"""POPF teacher labels (cached once), teacher losses for every model, and an offline teacher-trained GNN.

    python -m factory_test.teacher build  --bench factorysim --splits train val test [--cap 60] [--jobs 10]
    python -m factory_test.teacher train  --bench factorysim --train train --val val --test test
    python -m factory_test.teacher losses --bench factorysim [--models TIFLAX TIFLAX2 ...] [--splits ...]
    python -m factory_test.teacher plot   --bench factorysim

Goal of the learned part: POPF finds good schedules only with a long budget (or not at all on the full task);
the GNN should predict, in milliseconds, which objects such a schedule uses, so POPF then only has to plan on
a small set. The supervision target is therefore the best VAL-valid POPF schedule we can find offline.

Teacher plan per instance = shortest VAL-valid plan among (all VAL-checked on the full task):
  cert      the instance's certificate (POPF on the core, or the constructive serial schedule)
  full      POPF on the full task, cap --cap
  cert_set  POPF on COMP_T(objects of the certificate), cap --cap (often shorter than a serial certificate)
Labels (y), per object o of the full task, aligned with the graph (x):
  y_obj(o) = 1 if o is an argument of an action of the teacher plan or a goal object, else 0
  y_tmp(o) = crit(o) in [0, 1] from CPM over the teacher plan (1 = on the critical path), as the old tiflax/labels.py
Stored in results/teacher/<bench>/<split>/<instance>/ {teacher.plan, labels.json, x.npz}; `build` skips
instances that already have labels, so labels are generated once. x is rebuilt (cheap) when the graph
vocabulary changes; y never.

Teacher loss of a model on a split (the "test loss against POPF policies"), averaged over instances:
  obj  = class-balanced BCE(object logit, y_obj)       (positives and negatives weigh half each)
  tmp  = BCE(temporal logit, y_tmp)                    (models with a temporal head)
  plus AUC(obj) and recall@k: the share of teacher objects among the k highest-scored objects, k = |teacher set|.
"""
import argparse
import concurrent.futures
import glob
import hashlib
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn

import factory_test  # noqa: F401  (paths)
from factory_test import RESULTS
from algorithm import split_dirs as _split_dirs
from algorithm.pddlkit import TemporalPlan, validate_plan
from algorithm.semantics import analyze
from algorithm.grounding import comp_t
from algorithm.gnn import Graph
from algorithm.grounding import GroundTask
from algorithm.planning import Budget, Workspace, attempt, _full_files
from algorithm.gnn import Scorer, to_tensors
from algorithm.semantics import Simulator, bind_plan
from algorithm.grounding import Task

TEACHER = os.path.join(RESULTS, "teacher")
MODELS = os.path.join(RESULTS, "models")


def split_dirs(bench, split):
    return _split_dirs(split, bench)


def label_dir(bench, split, inst):
    return os.path.join(TEACHER, bench, split, inst)


# ==================== build: POPF teacher plans -> cached labels ====================================
def teacher_plan(d, bench, split, cap):
    """-> (best plan, source, {source: row}) over the sources in the module docstring."""
    task = Task.load(d)
    gt = GroundTask(task)
    inst = os.path.basename(d)
    rows, cands = {}, []
    ws = Workspace()
    try:
        dom, prob = _full_files(task, ws)
        cert_file = os.path.join(d, "certificate.plan")
        if os.path.exists(cert_file):
            cert = TemporalPlan.parse(open(cert_file).read())
            v = validate_plan(dom, prob, cert)
            rows["cert"] = {"outcome": "valid" if v.valid else "invalid", "makespan": cert.makespan, "seconds": 0.0}
            if v.valid:
                cands.append((cert.makespan, "cert", cert))
        jobs = [("full", frozenset(task.objects))]
        if cands:
            used = cands[0][2].objects_used() & set(task.objects)
            jobs.append(("cert_set", frozenset(comp_t(gt, used))))
        best = min((c[0] for c in cands), default=float("inf"))

        def run(job):
            w = Workspace()
            try:
                return job[0], attempt(task, gt, job[1], Budget(cap), job[0].replace(":", "_"), w, check_proof=False)
            finally:
                w.close()
        with concurrent.futures.ThreadPoolExecutor(min(4, len(jobs))) as ex:
            for name, a in ex.map(run, jobs):
                rows[name] = {"outcome": a.outcome, "makespan": a.makespan, "seconds": round(a.planner_s, 2),
                              "n_objects": a.n_objects}
                if a.outcome == "valid":
                    cands.append((a.makespan, name, a.plan))
    finally:
        ws.close()
    if not cands:
        return None, None, rows, task, gt
    ms, src, plan = min(cands, key=lambda c: (c[0], c[1]))
    return plan, src, rows, task, gt


def vocab_hash(vocab):
    return hashlib.sha256(json.dumps(vocab.signature(), sort_keys=True).encode()).hexdigest()[:16]


def save_x(path, g: Graph, vh):
    np.savez_compressed(path, nodes=g.nodes, senders=g.senders, receivers=g.receivers, edges=g.edges,
                        globals=g.globals, lb=np.array([g.lb]), objects=np.array(g.objects), vocab=np.array([vh]))


def load_x(path):
    z = np.load(path, allow_pickle=False)
    return Graph(z["nodes"], z["senders"], z["receivers"], z["edges"], z["globals"], [str(o) for o in z["objects"]],
                 float(z["lb"][0])), str(z["vocab"][0])


def label_instance(d, bench, split, cap, retry=False):
    inst = os.path.basename(d)
    out = label_dir(bench, split, inst)
    lp = os.path.join(out, "labels.json")
    if os.path.exists(lp) and (json.load(open(lp)).get("teacher") or not retry):
        return inst, "cached"
    t0 = time.time()
    plan, src, rows, task, gt = teacher_plan(d, bench, split, cap)
    os.makedirs(out, exist_ok=True)
    if plan is None:
        with open(os.path.join(out, "labels.json"), "w") as f:
            json.dump({"instance": inst, "teacher": None, "sources": rows}, f, indent=1)
        return inst, "no valid plan"
    return inst, write_labels(out, inst, bench, split, plan, src, rows, task, gt, cap, t0)


def write_labels(out, inst, bench, split, plan, src, rows, task, gt, cap, t0):
    """x.npz, teacher.plan and labels.json of a VAL-valid teacher plan. -> status message."""
    steps, durs, unknown = bind_plan(gt, plan)
    sim = Simulator(task).run(steps, durs)
    if unknown or not sim.valid:
        raise RuntimeError("{}: internal simulator rejects the VAL-valid teacher plan: {}".format(
            inst, (unknown or sim.violations)[:2]))
    an = analyze(task, sim)
    used = plan.objects_used() | set(task.goal_objects)
    sc = Scorer(task.spec, heads=2, temporal_features=True)
    g = sc.graph(task, gt)
    save_x(os.path.join(out, "x.npz"), g, vocab_hash(sc.vocab))
    with open(os.path.join(out, "teacher.plan"), "w") as f:
        f.write(plan.to_text())
    lab = {"instance": inst, "split": split, "bench": bench, "teacher": src, "makespan": round(plan.makespan, 3),
           "lb": gt.full.lb, "n_objects": len(task.objects), "n_teacher_objects": len(used & set(task.objects)),
           "teacher_popf_seconds": rows.get(src, {}).get("seconds"), "popf_cap": cap, "sources": rows,
           "objects": g.objects, "y_obj": [1.0 if o in used else 0.0 for o in g.objects],
           "y_tmp": [round(float(an.crit.get(o, 0.0)), 4) for o in g.objects],
           "seconds": round(time.time() - t0, 1)}
    with open(os.path.join(out, "labels.json"), "w") as f:
        json.dump(lab, f, indent=1)
    return "{} ms={:.1f} ({} of {} objects)".format(src, plan.makespan, lab["n_teacher_objects"], len(g.objects))


# ==================== losses of every saved model against the teacher labels ============================
def model_logit_fn(name, spec):
    """(logit_fn, graph_fn or None, scorer-for-vocab) for a saved model prefix in results/models."""
    prefix = os.path.join(MODELS, name)
    meta = json.load(open(prefix + ".json"))
    method = meta.get("method", name)
    if method == "TIFLAX5" or name.startswith("TIFLAX5") or method == "TIFLAX7" or name.startswith("TIFLAX7"):
        from algorithm.v5 import SetScorer
        s = SetScorer.load(prefix, spec)
        if method == "TIFLAX7" or name.startswith("TIFLAX7"):
            return None, None, s          # a set ranker: its per-object output is not trained
        return (lambda g: (s.outputs(g)[0], None)), None, s
    if method == "TIFLAX6" or name.startswith("TIFLAX6"):
        from algorithm.v6 import TriageScorer
        s = TriageScorer.load(prefix, spec)
        return (lambda g: (s.outputs(g)[0], None)), None, s
    s = Scorer.load(prefix, spec)
    return scorer_logits(s, 0, 1 if s.heads > 1 else None), None, s


def default_models(bench):
    names = []
    for f in sorted(glob.glob(os.path.join(MODELS, "*.json"))):
        n = os.path.basename(f)[:-5]
        try:
            meta = json.load(open(f))
        except Exception:      # noqa: BLE001 -- label files etc. are not models
            continue
        if isinstance(meta, dict) and "model" in meta and "vocab" in meta:
            if meta.get("bench") == bench:
                names.append(n)
    return names


def losses(args):
    splits = args.splits or sorted(os.path.basename(p) for p in glob.glob(os.path.join(TEACHER, args.bench, "*"))
                                   if os.path.isdir(p))
    spec = None
    for s in splits:
        dd = split_dirs(args.bench, s)
        if dd:
            spec = Task.load(dd[0]).spec
            break
    models = args.models or default_models(args.bench)
    table = {}
    for m in models:
        try:
            fn, gfn, sc = model_logit_fn(m, spec)
        except Exception as e:     # noqa: BLE001 -- a model of another domain etc.; reported
            print("  skip {}: {!r}".format(m, e))
            continue
        if fn is None:
            print("  skip {}: no trained per-object output (set ranker)".format(m))
            continue
        table[m] = {}
        for s in splits:
            exs = examples(args.bench, s, sc, gfn)
            table[m][s] = evaluate(fn, exs)
            r = table[m][s]
            if r:
                print("  {:<16} {:<14} obj BCE {:.4f}{}  AUC {:.3f}  recall@k {:.3f}  (n={})".format(
                    m, s, r["obj_loss"], "  tmp BCE {:.4f}".format(r["tmp_loss"]) if "tmp_loss" in r else "",
                    r["auc"], r["recall_at_k"], r["n"]), flush=True)
    os.makedirs(os.path.join(TEACHER, args.bench), exist_ok=True)
    with open(os.path.join(TEACHER, args.bench, "losses.json"), "w") as f:
        json.dump(table, f, indent=1)
    md = ["| Model | " + " | ".join(splits) + " |", "| --- |" + " --- |" * len(splits)]
    for m, row in table.items():
        md.append("| {} | ".format(m) + " | ".join(
            "{:.3f} / {:.2f}".format(row[s]["obj_loss"], row[s]["recall_at_k"]) if row.get(s) else "-" for s in splits) + " |")
    with open(os.path.join(TEACHER, args.bench, "losses.md"), "w") as f:
        f.write("Teacher object loss (class-balanced BCE) / recall@k against the POPF teacher plans.\n\n" + "\n".join(md) + "\n")
    print("written", os.path.relpath(os.path.join(TEACHER, args.bench, "losses.md")))


# ==================== curves ==============================================================================
def plot_history(history, path, title):
    """Per-epoch curves. history rows: {"epoch", "train"/"val"/"test:*": teacher metrics, "own": {"train", "val"}}.
    Panels: teacher loss, recall@k of the teacher objects, and the method's own training objective (if logged)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [h for h in history if isinstance(h, dict) and "epoch" in h]
    tkeys = [k for k in ("train", "val") if any(isinstance(h.get(k), dict) for h in rows)] + \
        sorted({k for h in rows for k in h if k.startswith("test:")})
    own = any(isinstance(h.get("own"), dict) for h in rows)
    n = (2 if tkeys else 0) + (1 if own else 0)
    if n == 0:
        return None
    fig, ax = plt.subplots(1, n, figsize=(5 * n, 3.6), squeeze=False)
    ax = ax[0]
    if tkeys:
        for k in tkeys:
            pts = [(h["epoch"], h[k]) for h in rows if isinstance(h.get(k), dict)]
            ax[0].plot([e for e, _ in pts], [r["loss"] for _, r in pts], marker=".", label=k)
            ax[1].plot([e for e, _ in pts], [r["recall_at_k"] for _, r in pts], marker=".", label=k)
        ax[0].set(xlabel="epoch", ylabel="teacher loss (vs POPF plans)", title=title)
        ax[1].set(xlabel="epoch", ylabel="recall@k of teacher objects")
    if own:
        a = ax[-1]
        for k in ("train", "val"):
            pts = [(h["epoch"], h["own"][k]) for h in rows if isinstance(h.get("own"), dict) and h["own"].get(k) is not None]
            a.plot([e for e, _ in pts], [v for _, v in pts], marker=".", label="own " + k)
        a.set(xlabel="epoch", ylabel="method's own training objective", title=title if not tkeys else "")
    for a in ax:
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot(args):
    for f in sorted(glob.glob(os.path.join(MODELS, "*.json"))):
        try:
            meta = json.load(open(f))
        except Exception:      # noqa: BLE001
            continue
        if isinstance(meta, dict) and meta.get("bench") == args.bench and meta.get("history"):
            n = os.path.basename(f)[:-5]
            p = plot_history(meta["history"], os.path.join(TEACHER, args.bench, "curves_{}.png".format(n)), n)
            if p:
                print("written", os.path.relpath(p))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--bench", default="factorysim")
    b.add_argument("--splits", nargs="+", required=True)
    b.add_argument("--cap", type=float, default=60.0, help="POPF seconds per teacher source")
    b.add_argument("--jobs", type=int, default=3, help="instances in parallel (each runs up to 4 POPF)")
    b.add_argument("--retry", action="store_true", help="recompute instances cached without a teacher plan")
    t = sub.add_parser("train")
    t.add_argument("--bench", default="factorysim")
    t.add_argument("--train", default="train")
    t.add_argument("--val", default="val")
    t.add_argument("--test", nargs="*", default=["test"])
    t.add_argument("--epochs", type=int, default=300)
    t.add_argument("--patience", type=int, default=40)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--hidden", type=int, default=16)
    t.add_argument("--steps", type=int, default=3)
    t.add_argument("--lam_t", type=float, default=1.0)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--name", default="TEACHER", help="model name prefix (TEACHER2: on the improved labels)")
    lo = sub.add_parser("losses")
    lo.add_argument("--bench", default="factorysim")
    lo.add_argument("--models", nargs="*")
    lo.add_argument("--splits", nargs="*")
    pl = sub.add_parser("plot")
    pl.add_argument("--bench", default="factorysim")
    args = ap.parse_args(argv)
    {"build": build, "train": train, "losses": losses, "plot": plot}[args.cmd](args)


if __name__ == "__main__":
    from algorithm.planning import ensure_hash_seed
    ensure_hash_seed()
    main()
