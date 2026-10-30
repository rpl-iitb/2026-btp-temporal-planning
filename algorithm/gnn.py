"""Object graph of a task, encode-process-decode GNN, Scorer (calibrated per-head scores, reward) and
the training labels (object membership; temporal criticality + counterfactual sigma). See algorithm/README.md.
"""
import json
import math
import os
import time
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

from .grounding import INF, comp_t
from .planning import Budget, attempt
from .semantics import Simulator, analyze, bind_plan, resource_predicates, resource_types


# ==================== Object graph ===============================================================
N_TEMPORAL = 8
CAP = 5.0


class Vocabulary:
    def __init__(self, spec, temporal=True):
        self.spec = spec
        self.temporal = temporal
        self.types = sorted(set(spec.types) | {"object"})
        self.unary = sorted(p for p, s in spec.predicates.items() if len(s) == 1)
        self.unary_fn = sorted(f for f, s in spec.functions.items() if len(s) == 1)
        chans = []
        for p, s in sorted(spec.predicates.items()):
            k = len(s)
            chans += [("fact", p, i, j) for i in range(k) for j in range(k) if i != j and k >= 2]
        self.edge_fact = {c: n for n, c in enumerate(chans)}
        self.edge_goal = {c: len(chans) + n for n, c in enumerate(chans)}
        base = 2 * len(chans)
        fchans = []
        for f, s in sorted(spec.functions.items()):
            k = len(s)
            fchans += [(f, i, j) for i in range(k) for j in range(k) if i != j and k >= 2]
        self.edge_fn = {c: base + n for n, c in enumerate(fchans)}
        self.n_edge = base + len(fchans)
        self.n_node = len(self.types) + 2 * len(self.unary) + len(self.unary_fn) + (N_TEMPORAL if temporal else 0)
        self.n_global = 4

    def signature(self):
        return {"types": self.types, "unary": self.unary, "unary_fn": self.unary_fn,
                "n_edge": self.n_edge, "temporal": self.temporal}


@dataclass
class Graph:
    nodes: np.ndarray
    senders: np.ndarray
    receivers: np.ndarray
    edges: np.ndarray
    globals: np.ndarray
    objects: List[str]
    lb: float


def temporal_lb(gt):
    lb = gt.full.lb if gt is not None else INF
    return lb if (0 < lb < INF) else 1.0


def build_graph(task, vocab: Vocabulary, gt=None) -> Graph:
    spec = task.spec
    objs = sorted(task.objects)
    idx = {o: i for i, o in enumerate(objs)}
    L = temporal_lb(gt) if gt is not None else _numeric_scale(task)
    X = np.zeros((len(objs), vocab.n_node), dtype=np.float32)
    nt = len(vocab.types)
    nu = len(vocab.unary)
    for o in objs:
        for anc in spec.ancestors(task.objects[o]):
            if anc in vocab.types:
                X[idx[o], vocab.types.index(anc)] = 1.0
    uidx = {p: i for i, p in enumerate(vocab.unary)}
    for f in sorted(task.init):
        if len(f) == 2 and f[0] in uidx and f[1] in idx:
            X[idx[f[1]], nt + uidx[f[0]]] = 1.0
    for g in task.goals:
        if len(g) == 2 and g[0] in uidx and g[1] in idx:
            X[idx[g[1]], nt + nu + uidx[g[0]]] = 1.0
    fidx = {f: i for i, f in enumerate(vocab.unary_fn)}
    for k, v in task.fluents.items():
        if len(k) == 2 and k[0] in fidx and k[1] in idx:
            X[idx[k[1]], nt + 2 * nu + fidx[k[0]]] = min(v / L, CAP)

    E: Dict[tuple, np.ndarray] = {}

    def edge(a, b, ch, val=1.0):
        key = (idx[a], idx[b])
        e = E.get(key)
        if e is None:
            e = E[key] = np.zeros(vocab.n_edge, dtype=np.float32)
        e[ch] = val

    for f in sorted(task.init):
        k = len(f) - 1
        if k >= 2:
            for i in range(k):
                for j in range(k):
                    if i != j and f[1 + i] in idx and f[1 + j] in idx:
                        edge(f[1 + i], f[1 + j], vocab.edge_fact[("fact", f[0], i, j)])
    for g in task.goals:
        k = len(g) - 1
        if k >= 2:
            for i in range(k):
                for j in range(k):
                    if i != j and g[1 + i] in idx and g[1 + j] in idx:
                        edge(g[1 + i], g[1 + j], vocab.edge_goal[("fact", g[0], i, j)])
    for key, v in task.fluents.items():
        k = len(key) - 1
        if k >= 2:
            for i in range(k):
                for j in range(k):
                    if i != j and key[1 + i] in idx and key[1 + j] in idx:
                        edge(key[1 + i], key[1 + j], vocab.edge_fn[(key[0], i, j)], min(v / L, CAP))

    if vocab.temporal and gt is not None:
        _temporal_features(task, gt, X, idx, L, vocab.n_node - N_TEMPORAL)

    dl = [d for d in task.deadlines.values()]
    glob = np.array([L / 100.0, len(task.goals) / 10.0, (min(dl) / L) if dl else 0.0,
                     math.log1p(len(objs)) / 5.0], dtype=np.float32)
    keys = list(E.keys())
    snd = np.array([k[0] for k in keys], dtype=np.int64)
    rcv = np.array([k[1] for k in keys], dtype=np.int64)
    EA = np.stack([E[k] for k in keys]) if keys else np.zeros((0, vocab.n_edge), np.float32)
    return Graph(X, snd, rcv, EA, glob, objs, L)


def _numeric_scale(task):
    vals = [v for k, v in task.fluents.items() if len(k) >= 3]
    return float(np.median(vals)) if vals else 1.0


def _temporal_features(task, gt, X, idx, L, t0):
    full = gt.full
    use = gt.earliest_use(full)
    n_rel = {}
    for i in gt.relevant:
        for o in gt.actions[i].objects:
            n_rel[o] = n_rel.get(o, 0) + 1
    rp_objs = gt.relaxed_plan_objects(full)
    rtypes = set(resource_predicates(task.spec).values())
    goal_time = full.goal_time
    lb = full.lb
    for o, i in idx.items():
        e = use.get(o, INF)
        X[i, t0 + 0] = min(e / L, CAP) if e < INF else CAP
        X[i, t0 + 1] = 0.0 if o in use else 1.0
        X[i, t0 + 2] = 1.0 if o in rp_objs else 0.0
        X[i, t0 + 3] = 1.0 if any(task.is_a(o, ty) for ty in rtypes) else 0.0
        X[i, t0 + 4] = math.log1p(n_rel.get(o, 0)) / 5.0
    for g, d in task.deadlines.items():
        for o in g[1:]:
            if o in idx:
                X[idx[o], t0 + 5] = min(d / L, CAP)
                gtm = goal_time.get(g, INF)
                X[idx[o], t0 + 6] = max(-CAP, min(CAP, (d - gtm) / L)) if gtm < INF else -CAP
    everything = set(task.objects)
    for o, i in idx.items():
        if X[i, t0 + 3] > 0 and o not in task.goal_objects and o in use:
            r = gt.trpg(everything - {o})
            v = r.lb if r.solvable_relaxed else INF
            X[i, t0 + 7] = CAP if v == INF else min(CAP, max(0.0, (v - lb) / L))


# ==================== Encode-process-decode network ==============================================
def mlp(sizes, din):
    layers, d = [], din
    for i, s in enumerate(sizes):
        layers.append(nn.Linear(d, s))
        if i < len(sizes) - 1:
            layers.append(nn.ReLU())
        d = s
    if len(sizes) > 1:
        layers.append(nn.LayerNorm(d))
    return nn.Sequential(*layers)


class EncodeProcessDecode(nn.Module):
    def __init__(self, n_node, n_edge, n_global, heads=2, hidden=16, steps=3, aggregate="sum"):
        super().__init__()
        h = hidden
        self.steps, self.aggregate, self.heads = steps, aggregate, heads
        self.enc_n = mlp([h, h], n_node)
        self.enc_e = mlp([h, h], n_edge)
        self.enc_g = mlp([h, h], n_global)
        # inputs: [latent0 | latent] for receiver, sender, edge, plus globals
        self.edge_fn = mlp([h, h], 2 * (2 * h) + 2 * h + h)
        self.node_fn = mlp([h, h], 2 * h + h + h)
        # plain linear output: a LayerNorm over the head logits would couple the heads
        self.dec = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, heads))
        self.config = dict(n_node=n_node, n_edge=n_edge, n_global=n_global, heads=heads, hidden=hidden,
                           steps=steps, aggregate=aggregate)

    def forward(self, nodes, senders, receivers, edges, glob):
        n0, e0 = self.enc_n(nodes), self.enc_e(edges)
        g = self.enc_g(glob.unsqueeze(0))
        n, e = n0, e0
        N = n.shape[0]
        out = None
        for _ in range(self.steps):
            nn_ = torch.cat([n0, n], dim=1)
            ee = torch.cat([e0, e], dim=1)
            ge = g.expand(ee.shape[0], -1)
            e = self.edge_fn(torch.cat([nn_[receivers], nn_[senders], ee, ge], dim=1))
            agg = torch.zeros(N, e.shape[1], dtype=e.dtype).index_add_(0, receivers, e)
            if self.aggregate == "mean":
                deg = torch.zeros(N, dtype=e.dtype).index_add_(0, receivers, torch.ones_like(receivers, dtype=e.dtype))
                agg = agg / deg.clamp(min=1.0).unsqueeze(1)
            n = self.node_fn(torch.cat([nn_, agg, g.expand(N, -1)], dim=1))
            out = self.dec(n)
        return out          # logits [N, heads] of the last step (FLAX uses the last step's output)


# ==================== Scorer =====================================================================
def to_tensors(g):
    return (torch.from_numpy(g.nodes), torch.from_numpy(g.senders), torch.from_numpy(g.receivers),
            torch.from_numpy(g.edges), torch.from_numpy(g.globals))


class Scorer:
    def __init__(self, spec, heads=2, temporal_features=True, hidden=16, steps=3, aggregate="sum",
                 combine="sum", weights=(1.0, 1.0), seed=0):
        torch.manual_seed(seed)
        self.vocab = Vocabulary(spec, temporal=temporal_features)
        self.model = EncodeProcessDecode(self.vocab.n_node, self.vocab.n_edge, self.vocab.n_global,
                                         heads=heads, hidden=hidden, steps=steps, aggregate=aggregate)
        self.heads = heads
        self.temperature = [1.0] * heads
        self.combine = combine
        self.weights = tuple(weights[:heads])
        self.history = []

    # ---- inference ------------------------------------------------------------------------
    def graph(self, task, gt=None):
        return build_graph(task, self.vocab, gt if self.vocab.temporal else None)

    def logits(self, g):
        return self.model(*to_tensors(g))

    def scores(self, task, gt=None, graph=None):
        """-> {object: [p_head0, p_head1, ...]} calibrated probabilities."""
        g = graph or self.graph(task, gt)
        self.model.eval()
        with torch.no_grad():
            z = self.logits(g)
            T = torch.tensor(self.temperature, dtype=z.dtype)
            p = torch.sigmoid(z / T).numpy()
        return {o: [float(x) for x in p[i]] for i, o in enumerate(g.objects)}

    def reward(self, p):
        if self.heads == 1:
            return p[0]
        wo, wt = self.weights
        if self.combine == "sum":
            return wo * p[0] + wt * p[1]
        if self.combine == "max":
            return max(wo * p[0], wt * p[1])
        if self.combine == "noisy_or":
            return 1.0 - (1.0 - wo * p[0]) * (1.0 - wt * p[1])
        raise ValueError(self.combine)

    @property
    def reward_scale(self):
        if self.heads == 1:
            return 1.0
        wo, wt = self.weights
        return {"sum": wo + wt, "max": max(wo, wt), "noisy_or": 1.0}[self.combine]

    # ---- calibration ----------------------------------------------------------------------
    def calibrate(self, examples):
        """examples: [(graph, targets [N, heads] float, mask [N, heads] float)].
        Fits one temperature per head by grid + golden-section search on BCE."""
        self.model.eval()
        Z, Y, M = [], [], []
        with torch.no_grad():
            for g, y, m in examples:
                Z.append(self.logits(g))
                Y.append(torch.as_tensor(y))
                M.append(torch.as_tensor(m))
        if not Z:
            return self.temperature
        Z, Y, M = torch.cat(Z), torch.cat(Y), torch.cat(M)
        temps = []
        for k in range(self.heads):
            z, y, m = Z[:, k], Y[:, k], M[:, k]
            if m.sum() == 0:
                temps.append(1.0)
                continue

            def nll(T):
                return float((torch.nn.functional.binary_cross_entropy_with_logits(
                    z / T, y, reduction="none") * m).sum() / m.sum())
            grid = np.exp(np.linspace(np.log(0.05), np.log(20.0), 61))
            best = min(grid, key=nll)
            lo, hi = best / 1.25, best * 1.25
            for _ in range(30):
                a, b = lo + 0.382 * (hi - lo), lo + 0.618 * (hi - lo)
                if nll(a) < nll(b):
                    hi = b
                else:
                    lo = a
            temps.append(float((lo + hi) / 2))
        self.temperature = temps
        return temps

    # ---- persistence ----------------------------------------------------------------------
    def save(self, prefix, extra=None):
        os.makedirs(os.path.dirname(os.path.abspath(prefix)), exist_ok=True)
        torch.save(self.model.state_dict(), prefix + ".pt")
        meta = {"model": self.model.config, "vocab": self.vocab.signature(), "temperature": self.temperature,
                "combine": self.combine, "weights": list(self.weights), "history": self.history,
                "saved": time.strftime("%Y-%m-%d %H:%M:%S")}
        meta.update(extra or {})
        with open(prefix + ".json", "w") as f:
            json.dump(meta, f, indent=1)

    @classmethod
    def load(cls, prefix, spec, combine=None, weights=None):
        with open(prefix + ".json") as f:
            meta = json.load(f)
        mc = meta["model"]
        s = cls(spec, heads=mc["heads"], temporal_features=meta["vocab"]["temporal"], hidden=mc["hidden"],
                steps=mc["steps"], aggregate=mc["aggregate"], combine=combine or meta["combine"],
                weights=weights or meta["weights"])
        if s.vocab.signature() != meta["vocab"]:
            raise ValueError("vocabulary mismatch: model {} was trained on a different domain spec".format(prefix))
        s.model.load_state_dict(torch.load(prefix + ".pt", weights_only=True))
        s.temperature = meta["temperature"]
        s.history = meta.get("history", [])
        s.meta = meta
        return s


class UniformScorer:
    """No learning: every object gets the same score on every head (ablation; PLOI without guidance).
    heads=2 gives [v, v] so methods that read a temporal score work unchanged."""

    def __init__(self, value=0.5, heads=1):
        self.value = value
        self.heads = heads
        self.reward_scale = float(heads)

    def scores(self, task, gt=None, graph=None):
        return {o: [self.value] * self.heads for o in task.objects}

    def reward(self, p):
        return sum(p)


# ==================== Training labels ============================================================
@dataclass
class LabelConfig:
    counterfactual_k: int = 4
    counterfactual_cap: float = 8.0
    kappa: float = 0.10
    untested_resource_mask: float = 0.0


def object_labels(task, objects, plan):
    used = plan.objects_used() | set(task.goal_objects)
    return np.array([[1.0 if o in used else 0.0] for o in objects], dtype=np.float32)


def temporal_labels(task, gt, objects, plan, S, scores, cfg: LabelConfig, ws, cancel=None):
    """-> (y [N, 1], mask [N, 1], info dict)"""
    steps, durs, unknown = bind_plan(gt, plan)
    if unknown:
        raise ValueError("plan has actions outside the grounded task: {}".format(unknown[:3]))
    sim = Simulator(task).run(steps, durs)
    if not sim.valid:
        raise ValueError("internal simulator rejects a VAL-valid plan: {}".format(sim.violations[:2]))
    an = analyze(task, sim)
    C = an.makespan
    used = plan.objects_used()
    rtypes = resource_types(task.spec)
    is_res = {o: any(task.is_a(o, t) for t in rtypes) for o in objects}
    crit = {o: an.crit.get(o, 0.0) for o in objects}
    sigma, tested, info = {}, set(), {"removal": {}, "addition": {}, "critical_types": an.critical_resource_types,
                                      "parallelism": an.parallelism, "makespan": C}
    tscore = lambda o: scores.get(o, [0.0, 0.0])[-1]  # noqa: E731
    rel = getattr(gt, "_rel_by_obj", {})
    k = cfg.counterfactual_k
    # removal: resource objects in the schedule, most critical first
    rem = sorted([o for o in used if is_res.get(o) and o not in task.goal_objects],
                 key=lambda o: (-crit[o], o))[: (k + 1) // 2]
    for o in rem:
        S2 = set(S) - {o}
        pr = gt.proof(S2)
        if pr is not None and pr["kind"] == "unreachable":
            sigma[o], why = 0.0, "logically necessary"
        elif pr is not None:
            sigma[o], why = 1.0, "deadline proof"
        else:
            a = attempt(task, None, S2, Budget(cfg.counterfactual_cap), "cf_remove", ws, cancel=cancel,
                        check_proof=False)
            if a.outcome == "valid":
                sigma[o], why = float(np.clip((a.makespan - C) / (cfg.kappa * C), 0, 1)), "C'={}".format(a.makespan)
            else:
                why = a.outcome
        if o in sigma:
            tested.add(o)
        info["removal"][o] = why
    # addition: resource objects outside S, highest temporal score first
    add = sorted([o for o in objects if is_res.get(o) and o not in S and o in rel],
                 key=lambda o: (-tscore(o), o))[: k // 2]
    for o in add:
        S2 = comp_t(gt, set(S) | {o})
        a = attempt(task, None, S2, Budget(cfg.counterfactual_cap), "cf_add", ws, cancel=cancel, check_proof=False)
        if a.outcome == "valid":
            sigma[o] = float(np.clip((C - a.makespan) / (cfg.kappa * C), 0, 1))
            tested.add(o)
            info["addition"][o] = "C'={}".format(a.makespan)
        else:
            info["addition"][o] = a.outcome
    y = np.zeros((len(objects), 1), dtype=np.float32)
    m = np.ones((len(objects), 1), dtype=np.float32)
    for i, o in enumerate(objects):
        y[i, 0] = max(crit[o], sigma.get(o, 0.0))
        if is_res[o] and o not in used and o not in tested:
            m[i, 0] = cfg.untested_resource_mask
        if o in info["removal"] and o not in tested:
            m[i, 0] = 0.0
        if o in info["addition"] and o not in tested:
            m[i, 0] = 0.0
    info["sigma"] = sigma
    return y, m, info
