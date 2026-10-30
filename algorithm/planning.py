"""Shared time budget with a ledger, cancellable POPF / OPTIC calls on Pi|S, VAL on the full task
(planning.attempt), and the PYTHONHASHSEED guard for reproducible runs.
"""
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional

from pddl_generator.planners import TemporalPlan, _last_plan, binary, validate_plan


# ==================== Budget and planner calls ===================================================
class Budget:
    def __init__(self, total_s: float):
        self.total = float(total_s)
        self.t0 = time.monotonic()
        self.ledger = defaultdict(float)
        self._lock = threading.Lock()

    def elapsed(self):
        return time.monotonic() - self.t0

    def remaining(self):
        return max(0.0, self.total - self.elapsed())

    def expired(self):
        return self.remaining() <= 0.0

    def cap(self, limit=None):
        r = self.remaining()
        return r if limit is None else min(r, limit)

    @contextmanager
    def charge(self, category):
        t = time.monotonic()
        try:
            yield
        finally:
            with self._lock:
                self.ledger[category] += time.monotonic() - t

    def report(self):
        return {"elapsed_s": round(self.elapsed(), 3), "budget_s": self.total,
                "ledger_s": {k: round(v, 3) for k, v in sorted(self.ledger.items())}}


@dataclass
class Attempt:
    stage: str
    n_objects: int
    outcome: str
    planner_s: float = 0.0
    val_s: float = 0.0
    states: Optional[int] = None
    makespan: Optional[float] = None
    plan: Optional[TemporalPlan] = field(default=None, repr=False)
    detail: str = ""
    t_end: float = 0.0          # budget-elapsed time when the attempt finished
    objects: Optional[frozenset] = field(default=None, repr=False)

    def row(self):
        return {"stage": self.stage, "n_objects": self.n_objects, "outcome": self.outcome,
                "planner_s": round(self.planner_s, 3), "val_s": round(self.val_s, 3), "states": self.states,
                "makespan": self.makespan, "detail": self.detail[:300], "t_end": round(self.t_end, 3)}


def run_planner(domain_file, problem_file, timeout, cancel: threading.Event = None, planner="popf",
                extra_args=(), anytime=False):
    """POPF (or OPTIC) with a hard time cap and cooperative cancellation.
    -> (status, plan, wall_s, states, stdout). status: solved|gave_up|timeout|cancelled|error.
    anytime=True: the planner prints improving plans until it is stopped; a run stopped by the
    cap that printed at least one plan counts as solved and returns the last (best) plan
    (callers validate it; `plans_in_output` gives the earlier ones as a fallback)."""
    exe = binary(planner)
    cmd = [exe] + list(extra_args) + [domain_file, problem_file]
    t0 = time.monotonic()
    out_f, err_f = tempfile.TemporaryFile("w+"), tempfile.TemporaryFile("w+")
    proc = subprocess.Popen(cmd, stdout=out_f, stderr=err_f, start_new_session=True)
    rc, why = None, None
    while True:
        rc = proc.poll()
        if rc is not None:
            break
        if time.monotonic() - t0 > timeout:
            why = "timeout"
        elif cancel is not None and cancel.is_set():
            why = "cancelled"
        if why:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
            break
        time.sleep(0.02)
    wall = time.monotonic() - t0
    out_f.seek(0)
    out = out_f.read()
    blocks = _last_plan(out)
    states = re.findall(r";\s*States evaluated[^:]*:\s*(\d+)", out)
    states = int(states[-1]) if states else None
    plan = TemporalPlan.parse("\n".join(blocks[-1])) if blocks else None
    low = out.lower()
    if why is None and plan is not None and "solution found" in low:
        status = "solved"
    elif anytime and plan is not None and why != "cancelled":
        status = "solved"
    elif why is not None:
        status = why
    elif "unsolvable" in low or "no solution" in low:
        status = "gave_up"
    else:
        status = "error"
    return status, plan, wall, states, out


def plans_in_output(out):
    """All plan blocks a planner printed, best (last) first."""
    return [TemporalPlan.parse("\n".join(b)) for b in reversed(_last_plan(out))]


class Workspace:
    """Temporary directory for reduced problems of one run."""

    def __init__(self, root=None, keep=False):
        self.dir = tempfile.mkdtemp(prefix="tiflax_", dir=root)
        self.keep = keep
        self._n = 0
        self._lock = threading.Lock()

    def path(self, tag):
        with self._lock:
            self._n += 1
            return os.path.join(self.dir, "{:04d}_{}".format(self._n, tag))

    def close(self):
        if not self.keep:
            shutil.rmtree(self.dir, ignore_errors=True)


def attempt(full_task, gt, S, budget: Budget, stage, ws: Workspace, cap=None, cancel=None,
            check_proof=True, planner="popf", planner_args=(), anytime=False, plan_task=None):
    """Plan on Pi|S and validate on Pi. Never returns an unvalidated plan.

    plan_task: plan on plan_task|S instead of full_task|S (e.g. a copy with tightened
    deadlines); the plan is still validated on full_task. gt (for the proof) must then
    describe plan_task."""
    plan_task = plan_task or full_task
    S = frozenset(S) | full_task.goal_objects
    n = len(S)
    if budget.expired():
        return Attempt(stage, n, "budget_exhausted", t_end=budget.elapsed(), objects=S)
    if check_proof and gt is not None:
        with budget.charge("proof"):
            pr = gt.proof(S)
        if pr is not None:
            return Attempt(stage, n, "proved_infeasible", detail="{kind}: {goals}".format(**pr),
                           t_end=budget.elapsed(), objects=S)
    with budget.charge("reduce"):
        sub = plan_task.restrict(S) if n < len(plan_task.objects) else plan_task
        dom, prob = sub.write(ws.dir, os.path.basename(ws.path(stage)))
    timeout = budget.cap(cap)
    if timeout <= 0.05:
        return Attempt(stage, n, "budget_exhausted", t_end=budget.elapsed(), objects=S)
    with budget.charge("plan"):
        status, plan, wall, states, out = run_planner(dom, prob, timeout, cancel, planner, planner_args, anytime)
    a = Attempt(stage, n, {"solved": "solved", "gave_up": "planner_gave_up"}.get(status, status),
                planner_s=wall, states=states, objects=S)
    if status == "solved":
        full_dom, full_prob = _full_files(full_task, ws)
        with budget.charge("validate"):
            t = time.monotonic()
            cands = plans_in_output(out)[:2] if anytime else [plan]
            for plan in cands:          # an anytime run killed mid-print may leave a truncated last plan
                v = validate_plan(full_dom, full_prob, plan)
                if v.valid:
                    break
            a.val_s = time.monotonic() - t
        if v.valid:
            a.outcome, a.plan, a.makespan = "valid", plan, plan.makespan
        else:
            a.outcome, a.detail = "invalid_on_full", "; ".join(v.errors[:5])
    elif status == "error":
        a.detail = out[-300:]
    a.t_end = budget.elapsed()
    return a


_FULL_LOCK = threading.Lock()


def _full_files(full_task, ws):
    with _FULL_LOCK:
        if not hasattr(ws, "_full") or ws._full[0] is not full_task:
            ws._full = (full_task, full_task.write(ws.dir, "full_problem"))
        return ws._full[1]


# ==================== Reproducibility ============================================================
def ensure_hash_seed(seed="0"):
    if os.environ.get("PYTHONHASHSEED") is None:
        os.environ["PYTHONHASHSEED"] = str(seed)
        os.execv(sys.executable, [sys.executable] + sys.orig_argv[1:])
