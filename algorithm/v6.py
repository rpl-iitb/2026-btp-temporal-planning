"""T-iFLAX V6: learned temporal triage (importance + makespan window per object) steering V4.2's seeds.

    python -m algorithm.v6 labels|train|tune|testA|testB|testC [--split ...]
"""
import argparse
import concurrent.futures
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn

from pddl_generator import TemporalPlan
from . import PDDL, results_dir, split_dirs
from .gnn import Scorer, temporal_lb, to_tensors
from .grounding import GroundTask, Task, agent_type, comp_t
from .planning import Budget, Workspace, attempt, ensure_hash_seed
from .semantics import resource_types
from .v1 import tightened_task
from .v3 import agent_seeds
from .v4 import seed_rounds

OUT = results_dir("v6")
CAPS = (10.0, 7.0, 5.0, 4.0, 4.0)          # one POPF at a time; k-th set gets CAPS[k] s (same as the V5 test)
RED_MARGIN = 0.25


def latest_deadline(task):
    ds = [g.deadline for g in task.problem.goals if g.deadline is not None]
    return max(ds) if ds else None


def candidates(gt, obj_cost=None):
    """V4.2's candidate sets (all agents: work-optimal + earliest-time; two-agent work-optimal seeds),
    optionally with learned object costs in the work-optimal relaxed plans. [(lb, label, S)]."""
    singles = agent_seeds(gt, 1000, kind="mix", obj_cost=obj_cost)
    _, second = seed_rounds(gt, 4, obj_cost=obj_cost)
    out, seen = [], set()
    for lb, c, S in singles + [x for x in second if "+" in x[1]]:
        if S not in seen and gt.proof(S) is None:
            seen.add(S); out.append((lb, c, S))
    return out


def machines(task):
    """Agents and the other resource objects (pots, counters), not tiles."""
    rts = [t for t in resource_types(task.spec) if t != "tile"]
    return {o for o in task.objects if any(task.is_a(o, t) for t in rts)}


# ---------------------------------------------------------------- labels
def label_instance(d, cap=10.0):
    task = Task.load(d)
    gt = GroundTask(task)
    lb = temporal_lb(gt)
    sets = {"%s_%d" % (c, i): S for i, (_, c, S) in enumerate(candidates(gt))}
    cert = os.path.join(d, "certificate.plan")
    if os.path.exists(cert):
        used = TemporalPlan.parse(open(cert).read()).objects_used() & set(task.objects)
        base = frozenset(comp_t(gt, used))
        sets["cert"] = base
        ag = agent_type(task)
        early = gt.earliest_use()
        for c in sorted(set(task.of_type(ag)) - base)[:3]:
            sets["cert+" + c] = frozenset(comp_t(gt, base | {c}))
        others = sorted((o for o in machines(task) - base - set(task.of_type(ag))), key=lambda o: (early.get(o, 1e9), o))
        for m in others[:3]:
            sets["cert+" + m] = frozenset(comp_t(gt, base | {m}))

    def one(item):
        ws = Workspace()
        try:
            return item[0], item[1], attempt(task, gt, item[1], Budget(cap), "label", ws, check_proof=False)
        finally:
            ws.close()
    with concurrent.futures.ThreadPoolExecutor(min(12, len(sets))) as ex:
        res = list(ex.map(one, sorted(sets.items(), key=lambda x: x[0])))
    valid = [(name, a.plan) for name, _, a in res if a.outcome == "valid"]
    tried = set().union(*[S for _, S, _ in res])
    lab = {}
    best = min(valid, key=lambda x: x[1].makespan)[1] if valid else None
    used_best = (best.objects_used() if best else set()) | set(task.goal_objects)
    for o in sorted(task.objects):
        ms = [p.makespan / lb for _, p in valid if o in p.objects_used()]
        lab[o] = {"imp": 1.0 if o in used_best else 0.0,
                  "tmin": min(ms) if ms else None, "tmax": max(ms) if ms else None,
                  "censored": (not ms) and o in tried}
    D = latest_deadline(task)
    return {"instance": os.path.basename(d), "lb": lb, "D_over_lb": (D / lb) if D else None,
            "sets": {name: {"n": len(S), "outcome": a.outcome, "makespan": a.makespan} for name, S, a in res},
            "labels": lab if valid else None}


def make_labels(split, cap, parallel):
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(parallel) as ex:
        rows = list(ex.map(lambda d: label_instance(d, cap), split_dirs(split)))
    out = os.path.join(OUT, "labels_%s.json" % split)
    json.dump(rows, open(out, "w"))
    n = sum(len(r["sets"]) for r in rows)
    ok = sum(s["outcome"] == "valid" for r in rows for s in r["sets"].values())
    print("%s: %d kitchens (%d with a valid plan), %d sets, %d solved, %.0f s -> %s"
          % (split, len(rows), sum(bool(r["labels"]) for r in rows), n, ok, time.time() - t0, out))


# ---------------------------------------------------------------- model
class TriageScorer(Scorer):
    def __init__(self, spec, hidden=16, steps=3, seed=0):
        super().__init__(spec, heads=3, temporal_features=True, hidden=hidden, steps=steps, seed=seed)
        self.combine = "v6"
        self.kappa = 0.0

    def outputs(self, g):
        z = self.model(*to_tensors(g))
        tmin = nn.functional.softplus(z[:, 1])
        return z[:, 0], tmin, tmin + nn.functional.softplus(z[:, 2])

    def predict(self, task, gt):
        g = self.graph(task, gt)
        self.model.eval()
        with torch.no_grad():
            zp, tmin, tmax = self.outputs(g)
        p = torch.sigmoid(zp).numpy()
        return {o: (float(p[i]), float(tmin[i]) * g.lb, float(tmax[i]) * g.lb) for i, o in enumerate(g.objects)}

    def triage(self, task, pred):
        D = latest_deadline(task)
        tiers = {}
        for o, (p, a, b) in pred.items():
            if D is None or o in task.goal_objects or b <= D:
                tiers[o] = "green"
            elif a <= D * (1 + RED_MARGIN):
                tiers[o] = "amber"
            else:
                tiers[o] = "red"
        return tiers

    def obj_cost(self, pred):
        return {o: self.kappa * (1.0 - p) for o, (p, _, _) in pred.items()}

    def predicted_makespan(self, task, pred, S):
        ms = machines(task) & set(S)
        return max((pred[o][1] for o in ms), default=0.0)

    def save(self, prefix, extra=None):
        super().save(prefix, dict(extra or {}, kappa=self.kappa))

    @classmethod
    def load(cls, prefix, spec):
        meta = json.load(open(prefix + ".json"))
        mc = meta["model"]
        s = cls(spec, hidden=mc["hidden"], steps=mc["steps"])
        if s.vocab.signature() != meta["vocab"]:
            raise ValueError("vocabulary mismatch for " + prefix)
        s.model.load_state_dict(torch.load(prefix + ".pt", weights_only=True))
        s.kappa = meta.get("kappa", 0.0)
        s.history = meta.get("history", [])
        return s


def examples(split, sc):
    out = []
    for r in json.load(open(os.path.join(OUT, "labels_%s.json" % split))):
        if not r["labels"]:
            continue
        task = Task.load(os.path.join(PDDL, split, r["instance"]))
        g = sc.graph(task, GroundTask(task))
        n = len(g.objects)
        y = np.zeros((n, 3), dtype=np.float32); m = np.zeros((n, 4), dtype=np.float32)
        for i, o in enumerate(g.objects):
            L = r["labels"][o]
            y[i, 0], m[i, 0] = L["imp"], 1.0
            if L["tmin"] is not None:
                # labels were divided by LB(Pi) at labelling time; the graph uses the same LB
                y[i, 1], y[i, 2] = L["tmin"], L["tmax"]
                m[i, 1] = m[i, 2] = 1.0
            elif L["censored"] and r["D_over_lb"]:
                m[i, 3] = 1.0
        out.append((r["instance"], g, torch.from_numpy(y), torch.from_numpy(m), float(r["D_over_lb"] or 0.0)))
    return out


def loss_of(sc, g, y, m, d):
    zp, tmin, tmax = sc.outputs(g)
    hub = lambda a, b, w: (nn.functional.huber_loss(a, b, reduction="none") * w).sum() / w.sum().clamp(min=1.0)  # noqa: E731
    l = nn.functional.binary_cross_entropy_with_logits(zp, y[:, 0])
    l = l + hub(tmin, y[:, 1], m[:, 1]) + hub(tmax, y[:, 2], m[:, 2])
    if m[:, 3].sum() > 0:
        l = l + (torch.relu(d - tmin) * m[:, 3]).sum() / m[:, 3].sum()
    return l, zp, tmin, tmax


def evaluate_split(sc, exs):
    sc.model.eval()
    L, acc, n, e1, e2, nt, cens, nc = 0.0, 0.0, 0, 0.0, 0.0, 0, 0.0, 0
    with torch.no_grad():
        for _, g, y, m, d in exs:
            l, zp, tmin, tmax = loss_of(sc, g, y, m, d)
            L += float(l)
            acc += float(((zp > 0).float() == y[:, 0]).sum()); n += len(y)
            k = m[:, 1] > 0
            e1 += float((tmin[k] - y[k, 1]).abs().sum()); e2 += float((tmax[k] - y[k, 2]).abs().sum()); nt += int(k.sum())
            c = m[:, 3] > 0
            cens += float((tmin[c] > d).float().sum()); nc += int(c.sum())
    return {"loss": L / len(exs), "imp_acc": acc / n, "tmin_mae": e1 / max(1, nt), "tmax_mae": e2 / max(1, nt),
            "censored_above_D": cens / max(1, nc)}


def train(epochs, lr, hidden, steps, seed):
    random.seed(seed); torch.manual_seed(seed)
    sc = TriageScorer(Task.load(split_dirs("train_large")[0]).spec, hidden=hidden, steps=steps, seed=seed)
    tr, va = examples("train_large", sc), examples("val_large", sc)
    mean_t = float(torch.cat([e[2][e[3][:, 1] > 0, 1] for e in tr]).mean())
    base = float(np.mean([float((e[2][e[3][:, 1] > 0, 1] - mean_t).abs().mean()) for e in va]))
    print("train kitchens %d, val kitchens %d; val MAE of a constant t_min predictor %.3f LB" % (len(tr), len(va), base))
    opt = torch.optim.Adam(sc.model.parameters(), lr=lr)
    best, state, t0 = None, None, time.time()
    for ep in range(epochs):
        sc.model.train()
        idx = list(range(len(tr))); random.shuffle(idx)
        for i in idx:
            l = loss_of(sc, *tr[i][1:])[0]
            opt.zero_grad(); l.backward(); opt.step()
        if ep % 10 == 9 or ep == epochs - 1:
            v = evaluate_split(sc, va)
            sc.history.append(dict(epoch=ep + 1, val=v))
            print("epoch %3d  val loss %.3f  imp acc %.3f  t_min MAE %.3f  t_max MAE %.3f  censored above D %.2f"
                  % (ep + 1, v["loss"], v["imp_acc"], v["tmin_mae"], v["tmax_mae"], v["censored_above_D"]))
            if best is None or v["loss"] < best:
                best, state = v["loss"], {k: t.clone() for k, t in sc.model.state_dict().items()}
    sc.model.load_state_dict(state)
    prefix = os.path.join(OUT, "v6_seed%d" % seed)
    sc.save(prefix, extra={"method": "V6", "train_seconds": round(time.time() - t0, 1), "best_val_loss": best})
    print("saved", prefix, "in %.0f s" % (time.time() - t0))


# ---------------------------------------------------------------- tests (one POPF at a time)
def sequential(task, gt, ranked, budget_s):
    t0, tried = time.monotonic(), 0
    for k, x in enumerate(ranked):
        left = budget_s - (time.monotonic() - t0)
        if left < 0.5:
            break
        ws = Workspace()
        try:
            a = attempt(task, gt, x[2], Budget(min(CAPS[min(k, len(CAPS) - 1)], left)), "seed", ws, check_proof=False)
        finally:
            ws.close()
        tried += 1
        if a.outcome == "valid":
            return {"solved": True, "rank": k, "tried": tried, "t": round(time.monotonic() - t0, 2),
                    "makespan": a.makespan, "n": len(x[2])}
    return {"solved": False, "tried": tried, "t": round(time.monotonic() - t0, 2)}


def variants(sc, task, gt, which):
    pred = sc.predict(task, gt)
    out = {}
    if "LB" in which:
        out["LB"] = sorted(candidates(gt), key=lambda x: (x[0], len(x[2]), x[1]))
    if "cost" in which or "cost_rank" in which:
        cands = candidates(gt, sc.obj_cost(pred))
        out["cost"] = sorted(cands, key=lambda x: (x[0], len(x[2]), x[1]))
        out["cost_rank"] = sorted(cands, key=lambda x: (round(sc.predicted_makespan(task, pred, x[2]), 1), x[0], len(x[2]), x[1]))
    return {k: v for k, v in out.items() if k in which}


def run_test(sc, split, which, budget_s, parallel, tag):
    out = os.path.join(OUT, "%s_%s.jsonl" % (tag, split))
    done = {json.loads(l)["instance"] for l in open(out)} if os.path.exists(out) else set()

    def run(d):
        task = Task.load(d)
        gt = GroundTask(task)
        t0 = time.monotonic()
        orders = variants(sc, task, gt, which)
        prep = time.monotonic() - t0
        row = {"instance": os.path.basename(d), "prep_s": round(prep, 2)}
        for k, ranked in orders.items():
            row[k] = sequential(task, gt, ranked, budget_s - prep / max(1, len(orders)))
        with open(out, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(row["instance"], {k: (v["solved"], v.get("rank"), v["t"]) for k, v in row.items() if isinstance(v, dict)}, flush=True)
    with concurrent.futures.ThreadPoolExecutor(parallel) as ex:
        list(ex.map(run, [d for d in split_dirs(split) if os.path.basename(d) not in done]))
    rows = [json.loads(l) for l in open(out)]
    for k in which:
        ok = [r[k] for r in rows if r[k]["solved"]]
        print("%-9s solved %d / %d, first-try %d, mean time to plan %.1f s, mean makespan %.1f"
              % (k, len(ok), len(rows), sum(o["rank"] == 0 for o in ok), np.mean([o["t"] for o in ok]) if ok else 0,
                 np.mean([o["makespan"] for o in ok]) if ok else 0))
    return rows


def on_time(a, b, D):
    if D is None:
        return 1.0
    if b - a < 1e-6:
        return 1.0 if a <= D else 0.0
    return min(1.0, max(0.0, (D - a) / (b - a)))


def triage_queue(sc, task, gt, budget_s, seed_share=0.5, green_until=0.8, step=3, cap=3.0):
    """Test B: seeds (TRPG-bound order, scaled caps) for seed_share * budget; then the user's queue:
    best seed + Green objects by importance (step at a time, COMP_T) until green_until * budget;
    then temporal rollback with Amber objects by p * P(on time); then a reset from the goals with all
    objects by importance. One POPF at a time; first VAL-valid plan wins."""
    t0 = time.monotonic()
    pred = sc.predict(task, gt)
    tiers = sc.triage(task, pred)
    D = latest_deadline(task)
    cands = sorted(candidates(gt), key=lambda x: (x[0], len(x[2]), x[1]))
    scaled = [c * seed_share for c in CAPS]
    el = lambda: time.monotonic() - t0  # noqa: E731

    def try_set(S, c, stage, k):
        ws = Workspace()
        try:
            a = attempt(task, gt, S, Budget(max(0.3, min(c, budget_s - el()))), stage, ws, check_proof=True)
        finally:
            ws.close()
        if a.outcome == "valid":
            return {"solved": True, "stage": stage, "k": k, "t": round(el(), 2), "makespan": a.makespan, "n": len(S)}
        return None
    for k, x in enumerate(cands):
        if el() > seed_share * budget_s - 0.3:
            break
        r = try_set(x[2], min(scaled[min(k, len(scaled) - 1)], seed_share * budget_s - el()), "seed", k)
        if r:
            return r
    base = set(cands[0][2]) if cands else set(task.goal_objects)
    green = sorted((o for o in task.objects if tiers[o] == "green" and o not in base), key=lambda o: -pred[o][0])
    amber = sorted((o for o in task.objects if tiers[o] == "amber" and o not in base),
                   key=lambda o: -pred[o][0] * on_time(pred[o][1], pred[o][2], D))
    S, k = set(base), 0
    for stage, queue, until in (("green", green, green_until), ("amber", amber, 1.0)):
        while queue and el() < until * budget_s - 0.3:
            S = set(comp_t(gt, S | set(queue[:step]))); queue = queue[step:]; k += 1
            r = try_set(S, min(cap, until * budget_s - el()), stage, k)
            if r:
                return r
    allq = sorted(task.objects, key=lambda o: -pred[o][0])
    S = set(task.goal_objects)
    while el() < budget_s - 0.3 and allq:
        S = set(comp_t(gt, S | set(allq[:2 * step]))); allq = allq[2 * step:]; k += 1
        r = try_set(S, min(cap, budget_s - el()), "reset", k)
        if r:
            return r
    return {"solved": False, "t": round(el(), 2), "tiers": {t: sum(v == t for v in tiers.values()) for t in ("green", "amber", "red")}}


def testB(sc, split, budget_s, parallel):
    out = os.path.join(OUT, "testB_%s.jsonl" % split)
    done = {json.loads(l)["instance"] for l in open(out)} if os.path.exists(out) else set()

    def run(d):
        task = Task.load(d)
        gt = GroundTask(task)
        row = {"instance": os.path.basename(d), "B": triage_queue(sc, task, gt, budget_s)}
        with open(out, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(row["instance"], row["B"], flush=True)
    with concurrent.futures.ThreadPoolExecutor(parallel) as ex:
        list(ex.map(run, [d for d in split_dirs(split) if os.path.basename(d) not in done]))
    rows = [json.loads(l) for l in open(out)]
    ok = [r["B"] for r in rows if r["B"]["solved"]]
    from collections import Counter
    print("B (seeds 50 %% + triage queue): solved %d / %d; by stage %s" % (len(ok), len(rows), dict(Counter(o["stage"] for o in ok))))


def capacity_test(sc, task, gt, budget_s=30.0, per=5.0):
    """Test C: first plan from the seeds (TRPG-bound order); then 3 one-object capacity additions chosen
    by the GNN vs 3 chosen V4.2-style (earliest-usable extra agent / machines by TRPG); POPF with every
    deadline tightened to C - max(1, 5 % C), `per` s each, VAL on the original task."""
    cands = sorted(candidates(gt), key=lambda x: (x[0], len(x[2]), x[1]))
    first = None
    t0 = time.monotonic()
    for k, x in enumerate(cands):
        left = budget_s - (time.monotonic() - t0)
        if left < 0.5:
            break
        ws = Workspace()
        try:
            a = attempt(task, gt, x[2], Budget(min(CAPS[min(k, len(CAPS) - 1)], left)), "seed", ws, check_proof=False)
        finally:
            ws.close()
        if a.outcome == "valid":
            first = (x[2], a.makespan)
            break
    if first is None:
        return None
    S, C = first
    pred = sc.predict(task, gt)
    tiers = sc.triage(task, pred)
    ms = sorted(machines(task) - set(S))
    gnn = sorted((o for o in ms if tiers[o] == "green" and pred[o][1] < C), key=lambda o: -(C - pred[o][1]) * pred[o][0])[:3]
    early = gt.earliest_use()
    ag = agent_type(task)
    agents = sorted((o for o in ms if task.is_a(o, ag)), key=lambda o: (early.get(o, 1e9), o))
    others = sorted((o for o in ms if not task.is_a(o, ag)), key=lambda o: (early.get(o, 1e9), o))
    sym = (agents[:1] + others[:2] + agents[1:] + others[2:])[:3]
    tt = tightened_task(task, C - max(1.0, 0.05 * C))
    gtt = gt.with_task(tt) if tt is not None else None

    def one(o):
        ws = Workspace()
        try:
            S2 = frozenset(comp_t(gt, set(S) | {o}))
            return o, attempt(task, gtt, S2, Budget(per), "cap", ws, plan_task=tt)
        finally:
            ws.close()
    with concurrent.futures.ThreadPoolExecutor(6) as ex:
        res = dict(ex.map(one, sorted(set(gnn) | set(sym))))
    best = lambda objs: min([res[o].makespan for o in objs if res[o].outcome == "valid"] + [C])  # noqa: E731
    return {"C": C, "gnn": gnn, "sym": sym, "C_gnn": best(gnn), "C_sym": best(sym)}


def testC(sc, split, parallel):
    out = os.path.join(OUT, "testC_%s.jsonl" % split)
    done = {json.loads(l)["instance"] for l in open(out)} if os.path.exists(out) else set()

    def run(d):
        task = Task.load(d)
        row = {"instance": os.path.basename(d), "C": capacity_test(sc, task, GroundTask(task))}
        with open(out, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(row["instance"], row["C"], flush=True)
    with concurrent.futures.ThreadPoolExecutor(parallel) as ex:
        list(ex.map(run, [d for d in split_dirs(split) if os.path.basename(d) not in done]))
    rows = [json.loads(l)["C"] for l in open(out)]
    rows = [r for r in rows if r]
    g = [r for r in rows if r["C_gnn"] < r["C"] - 0.5]; y = [r for r in rows if r["C_sym"] < r["C"] - 0.5]
    print("first plans %d; improved by GNN capacity %d, by V4.2-style capacity %d; mean makespan first %.1f, GNN %.1f, V4.2-style %.1f"
          % (len(rows), len(g), len(y), np.mean([r["C"] for r in rows]), np.mean([r["C_gnn"] for r in rows]), np.mean([r["C_sym"] for r in rows])))
    print("GNN better on %d, V4.2-style better on %d" % (sum(r["C_gnn"] < r["C_sym"] - 0.5 for r in rows), sum(r["C_sym"] < r["C_gnn"] - 0.5 for r in rows)))


def main():
    ensure_hash_seed()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["labels", "train", "tune", "testA", "testB", "testC"])
    ap.add_argument("--split", default="train_large")
    ap.add_argument("--cap", type=float, default=10.0)
    ap.add_argument("--parallel", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kappas", default="0,2,5,10")   # 0 = V4.2's candidates (reference)
    a = ap.parse_args()
    if a.cmd == "labels":
        return make_labels(a.split, a.cap, a.parallel)
    if a.cmd == "train":
        return train(a.epochs, a.lr, 16, 3, a.seed)
    prefix = os.path.join(OUT, "v6_seed%d" % a.seed)
    sc = TriageScorer.load(prefix, Task.load(split_dirs(a.split)[0]).spec)
    if a.cmd == "tune":                       # kappa on validation kitchens only
        res = {}
        for k in [float(x) for x in a.kappas.split(",")]:
            sc.kappa = k
            rows = run_test(sc, a.split, ["cost"], 30.0, 4, "tune_k%g" % k)
            res[k] = (sum(r["cost"]["solved"] for r in rows), -np.mean([r["cost"]["t"] for r in rows]))
        sc.kappa = max(res, key=lambda k: res[k])
        print("kappa ->", sc.kappa, res)
        sc.save(prefix, extra={"method": "V6", "kappa_tuning": {str(k): v[0] for k, v in res.items()}})
        return
    if a.cmd == "testC":
        return testC(sc, a.split, a.parallel)
    if a.cmd == "testB":
        return testB(sc, a.split, 30.0, a.parallel)
    run_test(sc, a.split, ["cost", "cost_rank"], 30.0, a.parallel, "testA")


if __name__ == "__main__":
    main()
