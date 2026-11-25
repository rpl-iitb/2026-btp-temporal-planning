"""T-iFLAX V3 / V3U / V3.1: per-agent minimal seeds planned in parallel before V1's expansion.

What it does: "minimal first": per agent, COMP_T(agent + its earliest-time relaxed plan) as a seed set; the best 4
seeds by lower bound are planned in parallel (cap 0.4 T), V1's expansion as fallback. V3U: uniform scores (ablation:
learning adds nothing); V3.1: + POPF on the full task in the background during improvement.
Earlier result (pre-final Overcooked-AI benchmark, 60 stress kitchens, 30 s): V3 27, V3U 28, V3.1 27 / 60.
"""
import concurrent.futures
import threading
from dataclasses import dataclass

from .gnn import Scorer, UniformScorer
from .grounding import agent_type, comp_t
from .semantics import resource_predicates
from .v1 import MethodConfig, ObjectSelectionPlanner, model_prefix


# ==================== per-agent minimal seeds =========================================================
def agent_seeds(gt, k=4, rank_key=None, kind="earliest", universe=None, obj_cost=None):
    """V3 minimal seeds: for each agent c, S_c = COMP_T(c + objects of the temporal relaxed plan of Pi
    with the other agents removed) -- one agent, its earliest stations. Ranked by the TRPG lower bound
    of S_c, then by `rank_key(c)` (e.g. a learned score), then size. Returns [(lb, agent, S_c)].
    universe: relaxed plans only over these objects (V5/V6: after temporal filtration); COMP_T may add others.
    obj_cost: extra per-object cost in the work-optimal relaxed plan (V6: learned)."""
    task = gt.task
    ag = agent_type(task)
    if ag is None:
        return []
    agents = task.of_type(ag)
    U = set(task.objects) if universe is None else set(universe) | set(task.goal_objects)
    out = []
    work, early, seen = [], [], set()
    for c in agents:
        if c not in U:
            continue
        keep = U - (set(agents) - {c})
        r = gt.trpg(keep)
        if not r.solvable_relaxed:
            continue
        plans = []
        if kind in ("work", "mix"):          # V4: work-optimal relaxed plan (durations as costs)
            plans.append(("work", gt.cost_relaxed_plan(keep, windows_from=r, obj_cost=obj_cost)))
        if kind in ("earliest", "mix"):      # V3: earliest-time relaxed plan
            plans.append(("earliest", gt.relaxed_plan(r)))
        for kd, plan in plans:
            if not plan:
                continue
            S = frozenset(comp_t(gt, {o for i in plan for o in gt.actions[i].objects} | {c}))
            if S in seen or (kd == "work" and gt.proof(S) is not None):
                continue
            seen.add(S)
            (work if kd == "work" else early).append((gt.trpg(S).lb, -(rank_key(c) if rank_key else 0.0), len(S), c, S))
    work.sort(key=lambda x: x[:3] + (x[3],))
    early.sort(key=lambda x: x[:3] + (x[3],))
    if kind == "mix":                        # best work-optimal seeds + the best earliest-time seed
        out = work[: max(1, k - 1)] + early[:1]
        out += [x for x in work[max(1, k - 1):] + early[1:]][: max(0, k - len(out))]
    else:
        out = (work if kind == "work" else early)[:k]
    return [(lb, c, S) for lb, _, _, c, S in out]




# ==================== planner ======================================================================
@dataclass
class V3Config(MethodConfig):
    name: str = "TiFLAX_V3"
    seeds: int = 4                       # number of seeds planned in parallel
    seed_frac: float = 0.4               # per-seed cap = seed_frac * T
    seed_kind: str = "earliest"          # earliest (V3) | work | mix (V4: work-optimal + earliest-time seeds)
    # V3.1
    bulk_capacity: bool = False
    tighten_frac_small: float = 0.02
    full_in_background: bool = False     # plain POPF on the full task during the improvement phase


PRESETS = {
    "V3": dict(),
    "V3U": dict(name="TiFLAX_V3_uniform"),
    "V3.1": dict(name="TiFLAX_V3.1", full_in_background=True),
}


class V3Planner(ObjectSelectionPlanner):
    # ---- seed phase ----------------------------------------------------------------------------
    def first_plan(self, R):
        if self.cfg.seeds:
            best = self._seed_phase(R)
            if best is not None:
                return best, "seed"
        return self._iflax(R)

    def seed_rounds(self, R):
        """-> list of rounds, each a list of (lb, agent, S). V3: one round of per-agent seeds."""
        rank = lambda c: R.scores.get(c, [0.0])[0]  # noqa: E731
        return [agent_seeds(R.gt, self.cfg.seeds, rank_key=rank, kind=self.cfg.seed_kind)]

    def _seed_phase(self, R):
        """Plan on each round's seeds in parallel; the first valid plan wins."""
        cfg = self.cfg
        with R.budget.charge("seeds"):
            rounds = self.seed_rounds(R)
        cap = cfg.seed_frac * R.budget.total
        for rnd, seeds in enumerate(rounds):
            if not seeds or R.budget.remaining() < 1.0:
                continue
            cancel = threading.Event()

            def one(seed):
                a = R.plan(seed[2], "seed" if rnd == 0 else "seed2", cap=cap, cancel=cancel)
                if a.outcome == "valid" and R.mode == "deploy":
                    cancel.set()
                return a
            with concurrent.futures.ThreadPoolExecutor(len(seeds)) as pool:
                outs = list(pool.map(one, seeds))
            valid = [a for a in outs if a.outcome == "valid"]
            if valid:
                return min(valid, key=lambda a: a.t_end)
        R.notes.append("no seed solved")
        return None

    # ---- V3.1 improvement phase --------------------------------------------------------------------
    def refine_start(self, R, best):
        if not self.cfg.full_in_background or len(best.objects) >= len(R.task.objects):
            return None
        # plain POPF on the full task in the background for the rest of the budget; its plan
        # replaces ours if it is VAL-valid and shorter (quality >= B1 whenever B1 finishes in time)
        pool = concurrent.futures.ThreadPoolExecutor(1)
        return pool, pool.submit(R.plan, frozenset(R.task.objects), "improve_full", None)

    def refine_end(self, R, best, state):
        if state is None:
            return best
        pool, job = state
        a = job.result()               # bounded by the budget (attempt caps at the remaining time)
        pool.shutdown(wait=False)
        if a.outcome == "valid" and a.makespan < best.makespan - 1e-6:
            best = self._retime(R, a)
        return best

    def tighten_fraction(self, improved_last):
        cfg = self.cfg
        return cfg.tighten_frac if (not cfg.full_in_background or improved_last) else cfg.tighten_frac_small

    def front_sets(self, R, S):
        return self._bulk_sets(R, S) if self.cfg.bulk_capacity else []

    def _bulk_sets(self, R, S):
        """Capacity in bulk -- the plan's objects plus every usable agent (cook), and the
        same plus the best extra machine of each machine type (a resource type used by an action
        with no agent parameter, i.e. something that runs by itself: the pot). Closed with COMP_T.
        Returned largest first. Domain-level definitions, no Overcooked names."""
        task, gt = R.task, R.gt
        ag = agent_type(task)
        if ag is None:
            return []
        rel = getattr(gt, "_rel_by_obj", {})
        agents = {o for o in task.of_type(ag) if o in rel}
        rtypes = set(resource_predicates(task.spec).values())
        machine_types = set()
        for act in task.spec.actions.values():
            ptypes = [p.type for p in act.params]
            if not any(task.spec.is_subtype(t, ag) for t in ptypes):
                machine_types |= {rt for rt in rtypes for t in ptypes if task.spec.is_subtype(t, rt)}
        out = [frozenset(R.closure(S | agents))]
        ranked = self.capacity_candidates(R, set(out[0]))
        extra = set()
        for mt in machine_types:
            pick = [o for o in ranked if task.is_a(o, mt)]
            if pick:
                extra.add(pick[0])
        if extra:
            out.insert(0, frozenset(R.closure(set(out[0]) | extra)))
        return [B for B in out if B != frozenset(S)]


def build(key, spec, seed=0, **over):
    cfg = V3Config(**dict(PRESETS[key], **over))
    scorer = UniformScorer(0.5, heads=2) if key == "V3U" else Scorer.load(model_prefix("v1", seed), spec)
    return V3Planner(cfg, scorer)
