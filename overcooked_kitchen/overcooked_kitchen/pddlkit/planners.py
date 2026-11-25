"""External tools: POPF / OPTIC (planning) and VAL (parsing, plan validation).

Binaries come from $OVERCOOKED_KITCHEN_POPF / _OPTIC / _VAL / _PARSER, else from PATH (popf, optic-clp, Validate,
Parser). Every call returns a result object with the raw output kept. Planner runtime = wall-clock of the planner
subprocess only. Ported from earlier research code.
"""
"""This file is UN-VERIFIED"""
import os
import re
import shutil
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from typing import Optional

from .plan import TemporalPlan

TOOLS = {"popf": ("OVERCOOKED_KITCHEN_POPF", "popf"), "optic": ("OVERCOOKED_KITCHEN_OPTIC", "optic-clp"),
         "validate": ("OVERCOOKED_KITCHEN_VAL", "Validate"), "parser": ("OVERCOOKED_KITCHEN_PARSER", "Parser")}
VAL_TOLERANCE = 0.001


def binary(name: str) -> str:
    """Path of tool `name` (popf | optic | validate | parser); FileNotFoundError if missing."""
    if name not in TOOLS:
        raise ValueError("unknown tool '{}'; available: {}".format(name, ", ".join(TOOLS)))
    var, exe = TOOLS[name]
    path = os.environ.get(var) or shutil.which(exe)
    if not path or not os.access(path, os.X_OK):
        raise FileNotFoundError("{} not found: set ${} or put {} on PATH".format(name, var, exe))
    return path


def available(name: str) -> bool:
    try:
        binary(name)
    except FileNotFoundError:
        return False
    return True


@dataclass
class PlannerResult:
    status: str                      # solved | unsolvable | timeout | error
    plan: Optional[TemporalPlan]
    runtime_s: float                 # planner wall-clock
    states_evaluated: Optional[int]
    num_plans_found: int
    planner: str
    cmd: list
    stdout: str = field(repr=False, default="")
    stderr: str = field(repr=False, default="")


def _kill(proc):
    """Kill the process group of a still-running proc and reap it."""
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()


def _run(cmd, timeout, cwd=None):
    """Run cmd in its own session; kill the whole group on timeout or on any exception (e.g. KeyboardInterrupt).
    Returns (returncode or None on timeout, stdout, stderr, wall seconds)."""
    t0 = time.monotonic()
    with tempfile.TemporaryFile("w+") as out_f, tempfile.TemporaryFile("w+") as err_f:
        proc = subprocess.Popen(cmd, stdout=out_f, stderr=err_f, cwd=cwd, start_new_session=True)
        try:
            rc = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill(proc)
            rc = None
        except BaseException:
            _kill(proc)
            raise
        wall = time.monotonic() - t0
        out_f.seek(0)
        err_f.seek(0)
        return rc, out_f.read(), err_f.read(), wall


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
    rc, out, err, wall = _run(cmd, timeout)
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
    return PlannerResult(status, plan, wall, int(states[-1]) if states else None, len(blocks), planner, cmd, out, err)


@dataclass
class ValidationResult:
    valid: bool
    value: Optional[float]           # VAL's "Final value" (makespan for minimize total-time)
    errors: list
    stdout: str = field(repr=False, default="")


def validate_plan(domain_file, problem_file, plan, tolerance=VAL_TOLERANCE, timeout=60.0):
    with tempfile.NamedTemporaryFile("w", suffix=".plan", delete=False) as f:
        f.write(plan.to_text() if isinstance(plan, TemporalPlan) else plan)
        plan_file = f.name
    try:
        cmd = [binary("validate"), "-t", str(tolerance), "-v", domain_file, problem_file, plan_file]
        rc, out, err, _ = _run(cmd, timeout)
    finally:
        os.unlink(plan_file)
    valid = "Plan valid" in out and "Bad plan" not in out and "failed" not in out.split("Plan valid")[0][-200:]
    m = re.search(r"Final value:\s*([-0-9.]+)", out)
    errors = []
    if not valid:
        m_adv = re.search(r"Plan Repair Advice:(.*?)(?:Failed plans:|$)", out, re.S)
        block = m_adv.group(1) if m_adv else "\n".join(out.strip().splitlines()[-15:])
        errors = [ln.strip() for ln in block.splitlines() if ln.strip()]
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
    rc, out, err, _ = _run(cmd, 30)
    m = re.search(r"Errors:\s*(\d+),\s*warnings:\s*(\d+)", out)
    e, w = (int(m.group(1)), int(m.group(2))) if m else (-1, -1)
    return ParseResult(e == 0, e, w, out + err)
