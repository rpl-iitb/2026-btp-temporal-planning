"""B1: POPF on the full temporal task (no object selection).

What it does: writes the whole problem as PDDL 2.1 and runs POPF on it with the full time budget; the plan is
VAL-checked on the full task. The plain temporal-planner reference every object-selection method must beat.
Needs: an instance (domain spec + HL problem) and POPF / VAL (TIFLAX_TOOLS). No training, no domain code.
Earlier result (pre-final Overcooked-AI benchmark, 60 stress kitchens, 30 s): 2 / 60 solved.
"""
from .core.planning import attempt

from .adapter import RunContext, plan_to_literals, run_result


class FullPlanner:
    name = "B1_full"

    def solve(self, task, budget_s, mode="deploy", **kw):
        ctx = RunContext(budget_s, deploy=True)
        try:
            a = attempt(task, None, set(task.objects), ctx.budget, "full", ctx.ws, check_proof=False)
            ctx.attempts.append(a)
            plan = None
            if a.outcome == "valid":
                plan = plan_to_literals(task, a.plan)
                plan.valid, plan.attempt = True, a
            return run_result(self.name, task, ctx, plan, "full", [])
        finally:
            ctx.close()


def build(key, spec, seed=0, **over):
    return FullPlanner()
