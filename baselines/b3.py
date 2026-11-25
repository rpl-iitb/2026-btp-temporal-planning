"""B3: iFLAX -- IFlaxPlanner (Algorithm 1, parallel 3R) + IFlaxGNNGuidance (vendor/), trained bilevel.

What it does: the BTP iFLAX implementation, unchanged: score-threshold expansion with parallel Repair / Restart /
Rollback branches (first valid branch wins) and a GNN trained bilevel (each epoch plans the training tasks with
the current scores, the branch with the fewest evaluated states gives the labels). POPF on the reduced temporal
problem, VAL on the full task.
Needs: FLAX's rule files (as B2). Earlier result (pre-final Overcooked-AI benchmark, 60 stress kitchens, 30 s): 3 / 60 solved.

    python -m baselines.b3 train --seed 0      # bilevel training -> results/b3/b3_seed0_*
"""
from .adapter import (RULES, FrozenScores, RunContext, TemporalPOPF, domain_for, frozen_scores,
                      install_rule_wrappers, load_guidance, model_prefix, run_result, task_to_state, train_main)


class IFlaxBaseline:
    name = "B3_iflax"

    def __init__(self, guidance, gamma=0.9, expansion_frac=0.6):
        install_rule_wrappers()
        self.guidance = guidance
        self.gamma, self.expansion_frac = gamma, expansion_frac

    def solve(self, task, budget_s, mode="deploy", **kw):
        from iflax_planner import IFlaxPlanner
        from planning import PlanningFailure, PlanningTimeout
        ctx = RunContext(budget_s, deploy=(mode == "deploy"))
        notes, plan, branch = [], None, None
        try:
            state = task_to_state(task)
            with ctx.budget.charge("gnn"):
                scores = frozen_scores(self.guidance, task, state)
            planner = IFlaxPlanner(True, lambda: TemporalPOPF(ctx), FrozenScores(scores), seed=0, gamma=self.gamma,
                                   complementary_rules=RULES["complementary"], relaxation_rules=RULES["relaxation"],
                                   expansion_frac=self.expansion_frac, mode=mode)
            try:
                plan = planner(domain_for(task.spec), state, budget_s - ctx.budget.elapsed())
                branch = (planner.last_result or {}).get("branch")
            except (PlanningFailure, PlanningTimeout) as e:
                notes.append("{}: {}".format(type(e).__name__, e))
            return run_result(self.name, task, ctx, plan, branch, notes)
        finally:
            ctx.close()


def make_guidance(seed, args=None):
    """iFLAX defaults (iflax/main.py): gamma 0.9, T1 = 0.6 T, pos_weight 10; 6 bilevel epochs (as T-iFLAX)."""
    from bilevel_guidance import IFlaxGNNGuidance
    prefix = model_prefix("B3", seed)
    g = IFlaxGNNGuidance(num_train_problems=10 ** 6, num_epochs=getattr(args, "epochs", 6),
                         bce_pos_weight=getattr(args, "pos_weight", 10.0),
                         complementary_rules=RULES["complementary"], relaxation_rules=RULES["relaxation"],
                         gamma=0.9, expansion_frac=0.6, load_from_file=True, save_model_prefix=prefix,
                         log_file=prefix + "_history.pkl", seed=seed)
    g.seed(seed)
    return g


def build(key, spec, seed=0, **over):
    return IFlaxBaseline(load_guidance("B3", seed, make_guidance))


if __name__ == "__main__":
    import sys
    from .core.planning import ensure_hash_seed
    ensure_hash_seed()
    if sys.argv[1:2] != ["train"]:
        raise SystemExit("usage: python -m baselines.b3 train [--seed 0 --epochs 6 --budget 15]")
    train_main("B3", make_guidance, lambda a: a.budget, sys.argv[2:])
