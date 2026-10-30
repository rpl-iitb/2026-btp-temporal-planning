"""Temporal plans (parse, makespan) and the POPF / OPTIC / VAL wrappers; tool paths from configs/tools.yaml."""

"""This file is UN-VERIFIED"""

import os
import re
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import psutil
import yaml


# ==================== Temporal plans =============================================================
_LINE = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s*:\s*\(([^)]*)\)\s*(?:\[\s*([0-9]*\.?[0-9]+)\s*\])?")


@dataclass(frozen=True)
class TimedAction:
    start: float
    name: str
    args: Tuple[str, ...]
    duration: float

    @property
    def end(self) -> float:
        return self.start + self.duration

    def to_line(self) -> str:
        return "{:.3f}: ({}) [{:.3f}]".format(self.start, " ".join((self.name,) + self.args), self.duration)


@dataclass
class TemporalPlan:
    actions: List[TimedAction]

    @classmethod
    def parse(cls, text: str) -> "TemporalPlan":
        acts = []
        for line in text.splitlines():
            if line.strip().startswith(";"):
                continue
            m = _LINE.match(line)
            if m:
                toks = m.group(2).lower().split()
                acts.append(TimedAction(float(m.group(1)), toks[0], tuple(toks[1:]),
                                        float(m.group(3)) if m.group(3) else 0.0))
        return cls(sorted(acts, key=lambda a: (a.start, a.name, a.args)))

    def to_text(self) -> str:
        return "\n".join(a.to_line() for a in self.actions) + "\n"

    @property
    def makespan(self) -> float:
        return max((a.end for a in self.actions), default=0.0)

    @property
    def total_action_time(self) -> float:
        return sum(a.duration for a in self.actions)

    def __len__(self):
        return len(self.actions)

    def objects_used(self) -> set:
        return {x for a in self.actions for x in a.args}


# ==================== POPF / OPTIC / VAL wrappers ================================================
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CFG = os.path.join(ROOT, "configs", "tools.yaml")


def tools_config() -> dict:
    with open(_CFG) as f:
        cfg = yaml.safe_load(f)
    for k, v in list(cfg.get("binaries", {}).items()):
        cfg["binaries"][k] = v if os.path.isabs(v) else os.path.join(ROOT, v)
    return cfg


def binary(name: str) -> str:
    path = tools_config()["binaries"][name]
    if not os.access(path, os.X_OK):
        raise FileNotFoundError("{} not found or not executable at {}".format(name, path))
    return path






@dataclass
class PlannerResult:
    status: str                      # solved | unsolvable | timeout | error
    plan: Optional[TemporalPlan]
    runtime_s: float                 # planner wall-clock
    peak_rss_mb: Optional[float]     # sampled every 50 ms; approximate
    states_evaluated: Optional[int]
    num_plans_found: int
    planner: str
    cmd: list
    stdout: str = field(repr=False, default="")
    stderr: str = field(repr=False, default="")


def _run_sampled(cmd, timeout, cwd=None):
    """Run cmd in its own session; kill the whole group on timeout. Returns
    (returncode or None on timeout, stdout, stderr, wall seconds, peak RSS MB)."""
    t0 = time.monotonic()
    out_f, err_f = tempfile.TemporaryFile("w+"), tempfile.TemporaryFile("w+")
    proc = subprocess.Popen(cmd, stdout=out_f, stderr=err_f, cwd=cwd, start_new_session=True)
    peak, rc = 0.0, None
    try:
        ps = psutil.Process(proc.pid)
    except psutil.Error:
        ps = None
    while True:
        rc = proc.poll()
        if rc is not None:
            break
        if time.monotonic() - t0 > timeout:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
            rc = None
            break
        if ps is not None:
            try:
                rss = ps.memory_info().rss + sum(c.memory_info().rss for c in ps.children(recursive=True))
                peak = max(peak, rss / 2 ** 20)
            except psutil.Error:
                pass
        time.sleep(0.05)
    wall = time.monotonic() - t0
    out_f.seek(0), err_f.seek(0)
    return rc, out_f.read(), err_f.read(), wall, (peak or None)


def _last_plan(stdout):
    """POPF (-n) prints each improved plan; keep the last complete one."""
    blocks, cur = [], []
    for line in stdout.splitlines():
        if re.match(r"^\s*[0-9.]+\s*:\s*\(", line):
            cur.append(line)
        elif cur:
            blocks.append(cur)
            cur = []
    if cur:
        blocks.append(cur)
    return blocks


def run_temporal_planner(domain_file, problem_file, timeout=60.0, planner="popf", anytime=False, extra_args=()):
    exe = binary(planner)
    cmd = [exe] + (["-n"] if anytime else []) + list(extra_args) + [domain_file, problem_file]
    rc, out, err, wall, peak = _run_sampled(cmd, timeout)
    blocks = _last_plan(out)
    states = re.findall(r";\s*States evaluated[^:]*:\s*(\d+)", out)
    plan = TemporalPlan.parse("\n".join(blocks[-1])) if blocks else None
    low = out.lower()
    if plan is not None and ("solution found" in low or rc is None):
        status = "solved"          # anytime run killed at timeout still returns its best plan
    elif rc is None:
        status = "timeout"
    elif "unsolvable" in low or "problem unsolvable" in low or "no solution" in low:
        status = "unsolvable"
    else:
        status = "error"
    return PlannerResult(status, plan, wall, peak, int(states[-1]) if states else None, len(blocks),
                         planner, cmd, out, err)


@dataclass
class ValidationResult:
    valid: bool
    value: Optional[float]           # VAL's "Final value" (makespan for minimize total-time)
    errors: list
    stdout: str = field(repr=False, default="")


def validate_plan(domain_file, problem_file, plan, tolerance=None, timeout=60.0):
    """VAL on the given (normally the full, unreduced) problem."""
    tol = tolerance if tolerance is not None else tools_config()["val"]["tolerance"]
    with tempfile.NamedTemporaryFile("w", suffix=".plan", delete=False) as f:
        f.write(plan.to_text() if isinstance(plan, TemporalPlan) else plan)
        plan_file = f.name
    try:
        cmd = [binary("validate"), "-t", str(tol), "-v", domain_file, problem_file, plan_file]
        rc, out, err, _, _ = _run_sampled(cmd, timeout)
    finally:
        os.unlink(plan_file)
    valid = "Plan valid" in out and "Bad plan" not in out and "failed" not in out.split("Plan valid")[0][-200:]
    m = re.search(r"Final value:\s*([-0-9.]+)", out)
    errors = []
    if not valid:
        # VAL -v also prints progress; the diagnosis is the "Plan Repair Advice" block
        m_adv = re.search(r"Plan Repair Advice:(.*?)(?:Failed plans:|$)", out, re.S)
        block = m_adv.group(1) if m_adv else "\n".join(out.strip().splitlines()[-15:])
        errors = [l.strip() for l in block.splitlines() if l.strip()]
        if "Plan failed to execute" in out:
            errors.insert(0, "Plan failed to execute")
    if rc is None:
        errors.append("VAL timed out")
        valid = False
    return ValidationResult(valid, float(m.group(1)) if m else None, errors, out + err)


@dataclass
class ParseResult:
    ok: bool
    errors: int
    warnings: int
    output: str = field(repr=False, default="")


def parse_check(domain_file, problem_file=None):
    """VAL's Parser: syntax + type checking, independent of our own reader."""
    cmd = [binary("parser"), domain_file] + ([problem_file] if problem_file else [])
    rc, out, err, _, _ = _run_sampled(cmd, 30)
    m = re.search(r"Errors:\s*(\d+),\s*warnings:\s*(\d+)", out)
    e, w = (int(m.group(1)), int(m.group(2))) if m else (-1, -1)
    return ParseResult(e == 0, e, w, out + err)
