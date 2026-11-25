"""T-iFLAX V7: listwise set ranker (V5's set scorer) trained on V6's POPF labels of test-sized kitchens.

What it does: a listwise ranker of V4.2's candidate sets (V5's set scorer) trained on V6's POPF labels of test-sized
instances.
Earlier result (pre-final Overcooked-AI benchmark, seeds-only protocol): 39 / 60.

    python -m algorithm.v7 train|test [--split stress]
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

from .pddlkit import TemporalPlan
from . import PDDL, results_dir, split_dirs
from .grounding import GroundTask, Task, agent_type, comp_t
from .planning import ensure_hash_seed
from .v5 import SetScorer
from .v6 import OUT as V6_OUT
from .v6 import candidates, machines, sequential

OUT = results_dir("v7")

AUX = 0.5


def rebuild_sets(task, gt, d):
    """The object sets of v6.label_instance, by the same names."""
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
    return sets


def kitchens(split, sc):
    """[(instance, [(name, graph, solved, is_candidate, lb_bound, size)])] for kitchens with both outcomes."""
    rows = json.load(open(os.path.join(V6_OUT, "labels_%s.json" % split)))
    out = []
    for r in rows:
        d = os.path.join(PDDL, split, r["instance"])
        task = Task.load(d)
        gt = GroundTask(task)
        sets = rebuild_sets(task, gt, d)
        items = []
        for name, info in sorted(r["sets"].items()):
            S = sets.get(name)
            if S is None or len(S) != info["n"]:
                continue                       # rebuilt set differs from the labelled one: skip it
            sub = task.restrict(S)
            g = sc.graph(sub, GroundTask(sub))
            items.append((name, g, 1.0 if info["outcome"] == "valid" else 0.0,
                          not name.startswith("cert"), gt.trpg(S).lb, len(S)))
        ys = [x[2] for x in items]
        if items and 0 < sum(ys) < len(ys):
            out.append((r["instance"], items))
    return out


def kitchen_loss(sc, items):
    z = torch.stack([sc.outputs(g)[1] for _, g, _, _, _, _ in items])
    y = torch.tensor([x[2] for x in items])
    target = y / y.sum()
    listwise = -(target * torch.log_softmax(z, 0)).sum()
    return listwise + AUX * nn.functional.binary_cross_entropy_with_logits(z, y), z


def top1(sc, ks):
    """Fraction of kitchens whose top-ranked CANDIDATE set (inference only sees candidates) is solved:
    GNN order vs TRPG-bound order (V4.2)."""
    sc.model.eval()
    g_ok = b_ok = n = 0
    with torch.no_grad():
        for _, items in ks:
            cand = [x for x in items if x[3]]
            if not cand or not any(x[2] for x in cand):
                continue
            z = [float(sc.outputs(x[1])[1]) for x in cand]
            g_ok += cand[int(np.argmax(z))][2]
            b_ok += min(cand, key=lambda x: (x[4], x[5], x[0]))[2]
            n += 1
    return g_ok / max(1, n), b_ok / max(1, n), n


def train(epochs, lr, seed):
    random.seed(seed); torch.manual_seed(seed)
    sc = SetScorer(Task.load(split_dirs("train_large")[0]).spec, seed=seed)
    tr, va = kitchens("train_large", sc), kitchens("val_large", sc)
    print("kitchens with both outcomes: train %d (%d sets), val %d (%d sets)"
          % (len(tr), sum(len(i) for _, i in tr), len(va), sum(len(i) for _, i in va)))
    opt = torch.optim.Adam(sc.model.parameters(), lr=lr)
    best, state, t0 = None, None, time.time()
    for ep in range(epochs):
        sc.model.train()
        random.shuffle(tr)
        for _, items in tr:
            l = kitchen_loss(sc, items)[0]
            opt.zero_grad(); l.backward(); opt.step()
        if ep % 5 == 4 or ep == epochs - 1:
            sc.model.eval()
            with torch.no_grad():
                vl = float(np.mean([float(kitchen_loss(sc, i)[0]) for _, i in va]))
            gt1, bt1, n = top1(sc, va)
            trg, trb, _ = top1(sc, tr)
            sc.history.append(dict(epoch=ep + 1, val_loss=vl, val_top1=gt1, bound_top1=bt1))
            print("epoch %3d  val loss %.3f  top-1 solved: GNN %.2f vs bound %.2f (val, %d kitchens); train GNN %.2f vs bound %.2f"
                  % (ep + 1, vl, gt1, bt1, n, trg, trb))
            if best is None or vl < best:
                best, state = vl, {k: t.clone() for k, t in sc.model.state_dict().items()}
    sc.model.load_state_dict(state)
    prefix = os.path.join(OUT, "v7_seed%d" % seed)
    sc.save(prefix, extra={"method": "V7", "kind": "listwise set ranker", "best_val_loss": best,
                           "train_seconds": round(time.time() - t0, 1)})
    print("saved", prefix)


def test(split, seed, parallel, budget_s=30.0):
    dirs = split_dirs(split)
    sc = SetScorer.load(os.path.join(OUT, "v7_seed%d" % seed), Task.load(dirs[0]).spec)
    out = os.path.join(OUT, "test_%s.jsonl" % split)
    done = {json.loads(l)["instance"] for l in open(out)} if os.path.exists(out) else set()

    def run(d):
        task = Task.load(d)
        gt = GroundTask(task)
        t0 = time.monotonic()
        cands = candidates(gt)
        sc.model.eval()
        with torch.no_grad():
            z = {S: float(sc.outputs(sc.graph(task.restrict(S), GroundTask(task.restrict(S))))[1]) for _, _, S in cands}
        ranked = sorted(cands, key=lambda x: (-round(z[x[2]], 3), x[0], len(x[2]), x[1]))
        prep = time.monotonic() - t0
        row = {"instance": os.path.basename(d), "prep_s": round(prep, 2), "GNN": sequential(task, gt, ranked, budget_s - prep)}
        with open(out, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(row["instance"], row["GNN"], flush=True)
    with concurrent.futures.ThreadPoolExecutor(parallel) as ex:
        list(ex.map(run, [d for d in dirs if os.path.basename(d) not in done]))
    rows = {json.loads(l)["instance"]: json.loads(l) for l in open(out)}
    ref = {json.loads(l)["instance"]: json.loads(l)["cost"] for l in open(os.path.join(V6_OUT, "testA_%s.jsonl" % split))}
    ok = [r["GNN"] for r in rows.values() if r["GNN"]["solved"]]
    print("V7 GNN order: solved %d / %d, first-try %d, mean time to plan %.1f s | TRPG-bound order (same protocol, "
          "v6 test A): solved %d" % (len(ok), len(rows), sum(o["rank"] == 0 for o in ok),
                                     np.mean([o["t"] for o in ok]) if ok else 0, sum(v["solved"] for v in ref.values())))
    print("only V7:", [i for i in rows if rows[i]["GNN"]["solved"] and not ref[i]["solved"]])
    print("only bound:", [i for i in rows if ref[i]["solved"] and not rows[i]["GNN"]["solved"]])


def main():
    ensure_hash_seed()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["train", "test"])
    ap.add_argument("--split", default="stress")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--parallel", type=int, default=4)
    a = ap.parse_args()
    if a.cmd == "train":
        train(a.epochs, a.lr, a.seed)
    else:
        test(a.split, a.seed, a.parallel)


if __name__ == "__main__":
    main()
