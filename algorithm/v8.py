"""T-iFLAX V8: learned object ranking + a constructive scheduler instead of POPF for the first plan.

What it does
  On large multi-step tasks even the right objects leave POPF a 150-300-step plan to find, so choosing objects
  cannot make POPF fast enough. V8 keeps what the GNN is good at -- ranking which objects (stations) matter --
  and builds the plan with a constructive scheduler that overlaps work while long actions run.
    1. scores: the GNN scorer's per-object scores (TIFLAX8), or none (TIFLAX8S: the scheduler ranks by itself)
    2. first plan: the scheduler's first plans for the best-ranked objects (all its dispatch policies, the learned
       ranking then its own ranking); the shortest one that VAL accepts on the full problem
    3. improvement with the rest of the budget: the scheduler's search (other object choices, policies, orders);
       a plan replaces the current one only if VAL accepts it and it is shorter
  Every plan is checked by VAL on the full problem, so no shortcut can return an invalid plan.

What it needs
  A scheduler object for the domain (supplied by the test harness, e.g. overcooked_test/kitchen_scheduler.py):
    first_plans(problem, score, k) -> [(plan, assignment)], shortest first
    search(problem, score, k, max_runs, deadline_fn) -> [(makespan, plan, assignment, policy)], shortest first
    assignment.objects() -> objects the assignment uses
  The algorithm itself has no domain code.

Earlier result (pre-final v2 kitchen benchmark, v2_test, 60 s): 60 / 60 solved, first plan in 1.1 s, 0.94 x certificate.
Plug-ins in final/: overcooked_test/kitchen_scheduler.py, factory_test/factory_scheduler.py (not evaluated yet).
"""

"""This file is UN-VERIFIED"""

import time

from .grounding import GroundTask
from .planning import Workspace, _full_files
from .pddlkit import validate_plan
from .v1 import RunResult


class V8Planner:
    def __init__(self, name, scheduler, scorer=None, k=4, max_runs=4000):
        self.name, self.scheduler, self.scorer, self.k, self.max_runs = name, scheduler, scorer, k, max_runs

    def solve(self, task, budget_s, mode="deploy", gt=None, graph=None) -> RunResult:
        SC = self.scheduler
        t0 = time.monotonic()
        el = lambda: time.monotonic() - t0       # noqa: E731
        ws = Workspace()
        attempts, notes = [], []
        best, first_s, first_ms = None, None, None
        try:
            dom, prob_file = _full_files(task, ws)
            score = None
            if self.scorer is not None:
                gt = gt if gt is not None else GroundTask(task)
                sc = self.scorer.scores(task, gt, graph=graph)
                score = {o: p[0] for o, p in sc.items()}
            t_gnn = el()

            def check(plan, stage, A):
                nonlocal best, first_s, first_ms
                t1 = el()
                v = validate_plan(dom, prob_file, plan)
                attempts.append({"stage": stage, "n_objects": len(A.objects()), "outcome": "valid" if v.valid else
                                 "invalid_on_full", "planner_s": 0.0, "val_s": round(el() - t1, 3),
                                 "makespan": plan.makespan if v.valid else None, "t_end": round(el(), 3)})
                if v.valid and (best is None or plan.makespan < best[0].makespan - 1e-6):
                    best = (plan, A)
                    if first_s is None:
                        first_s, first_ms = el(), plan.makespan
                return v.valid

            # 2. first plan
            for p, A in SC.first_plans(task.problem, score, self.k):
                if check(p, "first", A):
                    break
            # 3. improvement: the scheduler's search within the budget
            if mode == "deploy" or best is None:
                margin = 1.0
                res = SC.search(task.problem, score, self.k, max_runs=self.max_runs,
                                deadline_fn=lambda: el() > budget_s - margin - 2.0)
                for ms, p, A, pol in res[:5]:
                    if el() > budget_s - margin:
                        break
                    if best is not None and ms >= best[0].makespan - 1e-6:
                        break
                    if check(p, "search", A):
                        break
            if best is None:
                notes.append("no schedule")
        finally:
            ws.close()
        plan, A = best if best else (None, None)
        return RunResult(
            method=self.name, instance=task.name, budget_s=budget_s, solved=best is not None,
            makespan=plan.makespan if plan else None, first_makespan=first_ms, first_valid_s=first_s,
            total_s=el(), n_objects=len(plan.objects_used() & set(task.objects)) if plan else None,
            n_total=len(task.objects), branch="scheduler" if plan else None, attempts=attempts,
            ledger={"gnn": round(t_gnn, 3)}, planner_calls=0, timeouts=0, notes=notes,
            plan_text=plan.to_text() if plan else None,
            objects=sorted(plan.objects_used() & set(task.objects)) if plan else None)


def build(key, scheduler, scorer=None, **over):
    """TIFLAX8 (learned ranking, pass a scorer) or TIFLAX8S (scorer=None)."""
    return V8Planner(key, scheduler, scorer, **over)
