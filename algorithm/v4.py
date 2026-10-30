"""T-iFLAX V4 / V4.1 / V4.2: seeds from work-optimal relaxed plans (+ two-agent seeds), uniform scores."""
from dataclasses import dataclass

from .gnn import UniformScorer
from .grounding import agent_type, comp_t
from .v3 import V3Config, V3Planner, agent_seeds


# ==================== seed rounds with two-agent seeds ================================================
def seed_rounds(gt, k=4, rank_key=None, universe=None, obj_cost=None):
    """V4.1: two rounds of minimal seeds.
    Round 1 = V4's mix (best work-optimal seeds + the best earliest-time seed).
    Round 2 = the remaining single-agent seeds, then two-agent work-optimal seeds for the best agent
    pairs (relaxed plan with exactly those two agents), for deadlines one agent cannot meet.
    Each round has at most k seeds; seeds are ranked by TRPG bound, then size."""
    task = gt.task
    ag = agent_type(task)
    if ag is None:
        return [], []
    U = set(task.objects) if universe is None else set(universe) | set(task.goal_objects)
    first = agent_seeds(gt, k, rank_key=rank_key, kind="mix", universe=universe, obj_cost=obj_cost)
    used = {S for _, _, S in first}
    rest = [x for x in agent_seeds(gt, 4 * k, rank_key=rank_key, kind="mix", universe=universe, obj_cost=obj_cost) if x[2] not in used]
    agents = [c for _, c, _ in agent_seeds(gt, 4 * k, rank_key=rank_key, kind="work", universe=universe, obj_cost=obj_cost)]
    best = list(dict.fromkeys(agents))[:3]
    pairs = []
    for i in range(len(best)):
        for j in range(i + 1, len(best)):
            keep = U - (set(task.of_type(ag)) - {best[i], best[j]})
            r = gt.trpg(keep)
            if not r.solvable_relaxed:
                continue
            plan = gt.cost_relaxed_plan(keep, windows_from=r, obj_cost=obj_cost)
            if not plan:
                continue
            S = frozenset(comp_t(gt, {o for a in plan for o in gt.actions[a].objects} | {best[i], best[j]}))
            if S not in used and gt.proof(S) is None:
                used.add(S)
                pairs.append((gt.trpg(S).lb, best[i] + "+" + best[j], S))
    pairs.sort(key=lambda x: (x[0], len(x[2])))
    second = (pairs[:2] + rest[: max(0, k - 2)])[:k]
    if len(second) < k:
        second += [x for x in rest[max(0, k - 2):] + pairs[2:] if x not in second][: k - len(second)]
    return first, second


# ==================== planner ======================================================================
@dataclass
class V4Config(V3Config):
    name: str = "TiFLAX_V4"
    seed_kind: str = "mix"
    seed_rounds: int = 1                 # V4.1: 2 = a second round (remaining seeds + two-agent seeds)
    seed_pairs: int = 0                  # V4.2: add this many two-agent seeds to the single first round


PRESETS = {
    "V4": dict(),
    "V4.1": dict(name="TiFLAX_V4.1", seed_rounds=2, seed_frac=0.3),
    "V4.2": dict(name="TiFLAX_V4.2", seed_pairs=2, seed_frac=0.6),
}


class V4Planner(V3Planner):
    def seed_rounds(self, R):
        cfg = self.cfg
        rank = lambda c: R.scores.get(c, [0.0])[0]  # noqa: E731
        if cfg.seed_pairs:
            first, second = seed_rounds(R.gt, cfg.seeds, rank_key=rank)
            pairs = [x for x in second if "+" in x[1]][: cfg.seed_pairs]
            return [first + pairs]
        if cfg.seed_rounds >= 2:
            return [r for r in seed_rounds(R.gt, cfg.seeds, rank_key=rank) if r]
        return super().seed_rounds(R)


def build(key, spec, seed=0, **over):
    return V4Planner(V4Config(**dict(PRESETS[key], **over)), UniformScorer(0.5, heads=2))
