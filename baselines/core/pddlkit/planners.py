"""External tools: POPF / OPTIC (planning) and VAL (parsing, plan validation).

A binary is found by, in order: the environment variable TIFLAX_<NAME> (TIFLAX_POPF, TIFLAX_VALIDATE, TIFLAX_PARSER,
TIFLAX_OPTIC_CLP), the PATH, then TIFLAX_TOOLS (default: final/tools/bin). Every call returns
a result object with the raw output kept, so failed or timed-out runs are recorded rather than dropped. Planner
runtime = wall-clock of the planner subprocess only (not PDDL writing, not VAL).
"""
import os
import shutil

EXE = {"popf": "popf", "validate": "Validate", "parser": "Parser", "optic-clp": "optic-clp"}
VAL_TOLERANCE = 0.001        # epsilon separation; matches POPF's 0.001


def tools_dir() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.environ.get("TIFLAX_TOOLS") or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(here))), "tools", "bin")


def binary(name: str) -> str:
    exe = EXE.get(name, name)
    for path in (os.environ.get("TIFLAX_" + name.upper().replace("-", "_")), shutil.which(exe), os.path.join(tools_dir(), exe)):
        if path and os.access(path, os.X_OK):
            return path
    raise FileNotFoundError("{} not found: set TIFLAX_{} or put {} on PATH or in {}".format(name, name.upper().replace("-", "_"), exe, tools_dir()))

import re
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from typing import Optional

import psutil

from .plan import TemporalPlan



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
    tol = tolerance if tolerance is not None else VAL_TOLERANCE
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
