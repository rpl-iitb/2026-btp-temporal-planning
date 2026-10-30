"""B1: POPF on the full temporal task (no object selection)."""
from algorithm.planning import attempt

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
