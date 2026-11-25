"""T-iFLAX V2 / V2.1: V1 with a gated reward alpha*p_obj + (1-alpha)*p_temp (third GNN output).

What it does: V1 with a third GNN output alpha: reward = alpha * p_obj + (1 - alpha) * p_temp, trained on "should keep"
labels. V2.1: same network, first plan by object score only, the gate used in the improvement phase.
Earlier result (pre-final Overcooked-AI benchmark, 60 stress kitchens, 30 s): 5 / 60 (both).

    python -m algorithm.v2 train --method V2|V2.1 --seed 0
"""
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from .gnn import Scorer
from .planning import ensure_hash_seed
from .v1 import BilevelTrainer, MethodConfig, ObjectSelectionPlanner, model_prefix


# ==================== scorer and labels =============================================================
class GatedScorer(Scorer):
    """reward = alpha * p_obj + (1 - alpha) * p_temp, alpha = head 2."""

    def __init__(self, spec, heads=3, temporal_features=True, hidden=16, steps=3, aggregate="sum",
                 combine="gated", weights=(1.0, 1.0), seed=0):
        super().__init__(spec, heads=heads, temporal_features=temporal_features, hidden=hidden, steps=steps,
                         aggregate=aggregate, combine=combine, weights=weights, seed=seed)
        self.weights = tuple(weights[:2])

    def reward(self, p):
        a = p[2]
        return a * p[0] + (1.0 - a) * p[1]

    @property
    def reward_scale(self):
        return 1.0


def keep_labels(task, objects, plan, info, m_tmp, sigma_keep=0.5):
    """V2 gate target: should the selection keep o?
        y_keep(o) = 1  if o is in the returned schedule or a goal object (needed), or
                       sigma(o) >= sigma_keep (removing / adding o changes the makespan by
                       >= sigma_keep * kappa, i.e. 5 % with the defaults; temporally needed)
                  = 0  otherwise.
    Mask: 1 for needed or counterfactually tested objects, else the temporal mask
    (untested resource objects outside the schedule are unknown)."""
    used = plan.objects_used() | set(task.goal_objects)
    sigma = info.get("sigma", {})
    y = np.zeros((len(objects), 1), dtype=np.float32)
    m = np.zeros((len(objects), 1), dtype=np.float32)
    for i, o in enumerate(objects):
        y[i, 0] = 1.0 if (o in used or sigma.get(o, 0.0) >= sigma_keep) else 0.0
        m[i, 0] = 1.0 if (o in used or o in sigma) else m_tmp[i, 0]
    return y, m


# ==================== planner ======================================================================
@dataclass
class V2Config(MethodConfig):
    name: str = "TiFLAX_V2"
    first_plan_score: str = "reward"     # reward (V2) | object (V2.1)
    improve_with_gate: bool = False


PRESETS = {
    "V2": dict(),
    "V2.1": dict(name="TiFLAX_V2.1", first_plan_score="object", improve_with_gate=True),
}


class V2Planner(ObjectSelectionPlanner):
    def rewards(self, R):
        R.gated = {o: self.scorer.reward(p) for o, p in R.scores.items()}
        if self.cfg.first_plan_score == "object":
            return {o: p[0] for o, p in R.scores.items()}
        return dict(R.gated)

    def temporal_score(self, R):
        if self.cfg.improve_with_gate:
            return lambda o: R.gated.get(o, 0.0)
        return lambda o: R.scores.get(o, [0.0, 0.0])[1]

    def extra_sets(self, R, S):
        if not self.cfg.improve_with_gate:
            return []
        # V2.1: the gate's own selection, on top of the current plan's objects
        return [frozenset(R.closure(set(S) | {o for o, r in R.gated.items() if r >= self.cfg.q_max}))]


def build(key, spec, seed=0, **over):
    cfg = V2Config(**dict(PRESETS[key], **over))
    return V2Planner(cfg, GatedScorer.load(model_prefix(key.lower(), seed), spec))


# ==================== training ======================================================================
class V2Trainer(BilevelTrainer):
    def __init__(self, method="V2"):
        self.method = method
        self.name = method.lower()

    def add_args(self, ap):
        ap.add_argument("--method", choices=["V2", "V2.1"], default=self.method,
                        help="V2.1: the training planner selects by object_score (as V1), so object labels "
                             "come from V1-like plans; the gate still learns from the keep labels")
        ap.add_argument("--lam_k", type=float, default=1.0)
        ap.add_argument("--lam_a", type=float, default=0.1)

    def config(self, args):
        return V2Config(**PRESETS["V2" if args.method == "V2" else "V2.1"])

    def planner(self, cfg, scorer):
        return V2Planner(cfg, scorer)

    def new_scorer(self, spec, args):
        return GatedScorer(spec, heads=3, temporal_features=True, hidden=args.hidden, steps=args.steps, seed=args.seed)

    def labels(self, scorer, ex, plan, S, scores, label_cfg, ws):
        # temporal labels read the temporal head (index 1), not the gate
        y, m, info = super().labels(scorer, ex, plan, S, {o: p[:2] for o, p in scores.items()}, label_cfg, ws)
        y_k, m_k = keep_labels(ex.task, ex.graph.objects, plan, info, m[:, 1:2])
        return np.concatenate([y, y_k], 1), np.concatenate([m, m_k], 1), info

    def loss(self, scorer, z, y, m, args, pos_weight):
        loss = super().loss(scorer, z, y, m, args, pos_weight)
        p = torch.sigmoid(z)
        a = p[:, 2:3]
        r = (a * p[:, 0:1] + (1 - a) * p[:, 1:2]).clamp(1e-6, 1 - 1e-6)
        mk = m[:, 2:3]
        loss = loss + args.lam_k * (nn.functional.binary_cross_entropy(r, y[:, 2:3], reduction="none") * mk).sum() \
            / mk.sum().clamp(min=1.0)
        loss = loss + args.lam_a * (((a - 0.5) ** 2) * (1 - mk)).sum() / (1 - mk).sum().clamp(min=1.0)
        return loss

    def finish(self, scorer, calib, log):
        """No temperature scaling; report how the gate behaves on held-out tasks instead."""
        gate = {}
        hits, n = 0, 0
        for g, y, m, task in calib:
            with torch.no_grad():
                p = torch.sigmoid(scorer.logits(g)).numpy()
            for i, o in enumerate(g.objects):
                gate.setdefault(task.type_of(o), []).append(float(p[i, 2]))
                if m[i, 2] > 0:
                    r = p[i, 2] * p[i, 0] + (1 - p[i, 2]) * p[i, 1]
                    hits += int((r >= 0.5) == (y[i, 2] >= 0.5))
                    n += 1
        log("validation keep-label accuracy (reward >= 0.5): {}".format("{:.3f}".format(hits / n) if n else "-"))
        log("mean gate alpha by object type (1 = object score, 0 = temporal score):",
            {k: round(float(np.mean(v)), 3) for k, v in sorted(gate.items())})


if __name__ == "__main__":
    import sys
    ensure_hash_seed()
    if sys.argv[1:2] != ["train"]:
        raise SystemExit("usage: python -m algorithm.v2 train [--method V2|V2.1] [--seed 0 ...]")
    argv = sys.argv[2:]
    method = argv[argv.index("--method") + 1] if "--method" in argv else "V2"
    V2Trainer(method).run(argv)
