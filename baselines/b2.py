"""B2: FLAX -- the original FlaxPlanner + GNNSearchGuidance (vendor/), POPF on the reduced temporal task.

What it does: FLAX (sair-lab/flax @ 42a1030, run unchanged through adapter.py): a GNN scores objects, a growing
score threshold picks a reduced problem, relaxation rules and complementary rules repair it, and POPF plans on
the reduced *temporal* problem; the plan is VAL-checked on the full task. Trained offline (301 epochs) on teacher
plans (shorter of POPF on the full task and the instance certificate).
Needs: FLAX's two rule files for the domain (TIFLAX_FLAX_COMPLEMENTARY, TIFLAX_FLAX_RELAXATION; e.g.
overcooked_test/flax_rules/). Earlier result (pre-final Overcooked-AI benchmark, 60 stress kitchens, 30 s): 1 / 60 solved.

    python -m baselines.b2 train --seed 0      # offline training -> results/b2/b2_seed0_*
"""
import os

from .adapter import (RULES, FrozenScores, RunContext, TemporalPOPF, domain_for, frozen_scores, graph_literals,
                      install_rule_wrappers, load_guidance, model_prefix, plan_to_literals, run_result, task_to_state,
                      train_main)


class FlaxBaseline:
    name = "B2_flax"

    def __init__(self, guidance, gamma=0.9):
        install_rule_wrappers()
        self.guidance = guidance
        self.gamma = gamma
        self.last_ctx = None

    def solve(self, task, budget_s, mode="deploy", **kw):
        from planning import PlanningFailure, PlanningTimeout
        from planning.my_planner import FlaxPlanner
        ctx = RunContext(budget_s, deploy=True)
        self.last_ctx = ctx
        notes, plan = [], None
        try:
            state = task_to_state(task)
            with ctx.budget.charge("gnn"):
                scores = frozen_scores(self.guidance, task, state)
            planner = FlaxPlanner(True, TemporalPOPF(ctx), FrozenScores(scores), seed=0, gamma=self.gamma,
                                  complementary_rules=RULES["complementary"], relaxation_rules=RULES["relaxation"])
            try:
                plan, _ = planner(domain_for(task.spec), state, budget_s - ctx.budget.elapsed())
            except (PlanningFailure, PlanningTimeout) as e:
                notes.append("{}: {}".format(type(e).__name__, e))
            relaxed_seen = any(a.stage == "relaxed" for a in ctx.attempts)
            return run_result(self.name, task, ctx, plan, "step3" if relaxed_seen else "step1", notes)
        finally:
            ctx.close()


# ---- training -----------------------------------------------------------------------------------------
def flax_guidance_class():
    from guidance.gnn_guidance import GNNSearchGuidance

    class FlaxGuidance(GNNSearchGuidance):
        """GNNSearchGuidance with the arity-2 GNN view (FLAX asserts arity <= 2) and
        plan objects limited to the state's objects (constants are not graph nodes)."""

        def _state_to_graph(self, state):
            return super()._state_to_graph(graph_literals(state))

        def _create_graph_dataset(self, training_data):
            states, targets = training_data
            states = [graph_literals(s) for s in states]
            targets = [{o for o in t if o in s.objects} for s, t in zip(states, targets)]
            return super()._create_graph_dataset((states, targets))

    return FlaxGuidance


class TeacherPlanner:
    """FLAX labels objects with optimal Fast Downward plans of the full task. No optimal temporal planner
    is used here: the teacher returns the shorter of POPF on the full task (cap) and the VAL-checked certificate."""

    def __init__(self, cap):
        self.cap = cap

    def __call__(self, domain, state, timeout=None):
        from planning import PlanningFailure
        from .core.pddlkit import TemporalPlan, validate_plan
        from .core.planning import attempt
        task = state.goal.task
        ctx = RunContext()
        try:
            a = attempt(task, None, set(task.objects), ctx.budget, "teacher", ctx.ws, cap=self.cap, check_proof=False)
            cands = [a.plan] if a.outcome == "valid" else []
            cert = os.path.join(getattr(task, "source_dir", ""), "certificate.plan")
            if os.path.exists(cert):
                with open(cert) as f:
                    cp = TemporalPlan.parse(f.read())
                dom, prob = task.write(ctx.ws.dir, "full")
                if validate_plan(dom, prob, cp).valid:
                    cands.append(cp)
            if not cands:
                raise PlanningFailure("no teacher plan")
            out = plan_to_literals(task, min(cands, key=lambda p: p.makespan))
            out.valid = True
            return out
        finally:
            ctx.close()


def make_guidance(seed, args=None):
    """FLAX defaults (flax/src/main.py): 301 epochs, BCE pos_weight 10, 3 message-passing steps, 16 hidden."""
    prefix = model_prefix("B2", seed)
    g = flax_guidance_class()(
        training_planner=TeacherPlanner(getattr(args, "label_cap", 30.0)),
        num_train_problems=10 ** 6, num_epochs=getattr(args, "flax_epochs", 301), criterion_name="bce",
        bce_pos_weight=getattr(args, "pos_weight", 10.0), load_from_file=True, load_dataset_from_file=True,
        dataset_file_prefix=prefix + "_dataset", save_model_prefix=prefix, is_strips_domain=True)
    g.seed(seed)
    return g


def build(key, spec, seed=0, **over):
    return FlaxBaseline(load_guidance("B2", seed, make_guidance))


if __name__ == "__main__":
    import sys
    from .core.planning import ensure_hash_seed
    ensure_hash_seed()
    if sys.argv[1:2] != ["train"]:
        raise SystemExit("usage: python -m baselines.b2 train [--seed 0 --flax_epochs 301 --label_cap 30]")
    train_main("B2", make_guidance, lambda a: a.label_cap, sys.argv[2:])
