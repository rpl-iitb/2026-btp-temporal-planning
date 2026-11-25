"""POPF teacher labels (cached once), teacher losses for every model, and an offline teacher-trained GNN.

    python -m overcooked_test.teacher build  --bench overcooked_kitchen --splits v2_train v2_val v2_test [--cap 60] [--jobs 10]
    python -m overcooked_test.teacher train  --bench overcooked_kitchen --train v2_train --val v2_val --test v2_test
    python -m overcooked_test.teacher losses --bench overcooked_kitchen [--models TIFLAX TIFLAX2 ...] [--splits ...]
    python -m overcooked_test.teacher plot   --bench overcooked_kitchen
    python -m overcooked_test.teacher improve --bench overcooked_kitchen --splits v2_train v2_val v2_test

Goal of the learned part: POPF finds good schedules only with a long budget (or not at all on the full task);
the GNN should predict, in milliseconds, which objects such a schedule uses, so POPF then only has to plan on
a small set. The supervision target is therefore the best VAL-valid POPF schedule we can find offline.

Teacher plan per instance = shortest VAL-valid plan among (all VAL-checked on the full task):
  cert      the instance's certificate (POPF on the core, or the constructive serial schedule)
  full      POPF on the full task, cap --cap
  cert_set  POPF on COMP_T(objects of the certificate), cap --cap (often shorter than a serial certificate)
  sched     `improve`: the kitchen scheduler's shortest plan (VAL + overcooked_kitchen replay checked), when shorter
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

import overcooked_test  # noqa: F401  (paths)
from overcooked_test import RESULTS
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


# ==================== improve: better teacher plans from the kitchen scheduler ================================
def improve_instance(d, bench, split, max_runs=4000):
    """Replace the cached teacher plan by the shortest scheduler plan (kitchen_scheduler.py, both station rankings) if
    it is VAL-valid on the full task, replays in overcooked_kitchen (all served, nothing burnt, nothing late) and is
    shorter. The previous labels are kept as labels_prev.json / teacher_prev.plan."""
    from overcooked_test import kitchen_scheduler as SC
    from overcooked_kitchen.scenarios import load_scenario
    from overcooked_kitchen.suite import replay
    inst = os.path.basename(d)
    out = label_dir(bench, split, inst)
    lp = os.path.join(out, "labels.json")
    if not os.path.exists(lp):
        return inst, "no labels (run build first)"
    lab = json.load(open(lp))
    if lab.get("teacher") == "sched":
        return inst, "cached"
    t0 = time.time()
    task = Task.load(d)
    gt = GroundTask(task)
    old = lab.get("makespan") or float("inf")
    res = SC.search(task.problem, None, 4, max_runs=max_runs)
    try:
        tv = Scorer.load(os.path.join(MODELS, "TEACHER_{}_seed0".format(bench)), task.spec)
        score = {o: p[0] for o, p in tv.scores(task, gt).items()}
        res += SC.search(task.problem, score, 4, max_runs=max_runs)
    except FileNotFoundError:
        pass
    res.sort(key=lambda r: r[0])
    ws = Workspace()
    try:
        dom, prob = _full_files(task, ws)
        for ms, plan, A, pol in res[:6]:
            if ms >= old - 1e-6:
                break
            if not validate_plan(dom, prob, plan).valid:
                continue
            if not replay(load_scenario(os.path.join(d, "scenario.yaml")), plan)[0]:
                continue
            for n in ("labels.json", "teacher.plan"):
                src_f = os.path.join(out, n)
                if os.path.exists(src_f):
                    os.replace(src_f, os.path.join(out, n.replace(".", "_prev.", 1)))
            rows = dict(lab.get("sources", {}))
            rows["sched"] = {"outcome": "valid", "makespan": plan.makespan, "seconds": round(time.time() - t0, 2),
                             "policy": pol, "n_objects": len(A.objects())}
            msg = write_labels(out, inst, bench, split, plan, "sched", rows, task, gt, lab.get("popf_cap"), t0)
            return inst, "improved {:.1f} -> {}".format(old, msg)
    finally:
        ws.close()
    return inst, "kept {} ms={:.1f}".format(lab.get("teacher"), old)


def improve(args):
    for split in args.splits:
        dirs = split_dirs(args.bench, split)
        t0 = time.time()
        print("[{}] improve {}: {} instances".format(args.bench, split, len(dirs)), flush=True)
        with concurrent.futures.ProcessPoolExecutor(args.jobs) as ex:
            futs = {ex.submit(improve_instance, d, args.bench, split): d for d in dirs}
            for fu in concurrent.futures.as_completed(futs):
                try:
                    inst, msg = fu.result()
                except Exception as e:     # noqa: BLE001 -- reported
                    inst, msg = os.path.basename(futs[fu]), "ERROR {!r}".format(e)
                print("  {:<34} {}".format(inst, msg), flush=True)
        summarize(args.bench, split)
        print("  done in {:.0f}s".format(time.time() - t0), flush=True)


def build(args):
    for split in args.splits:
        dirs = split_dirs(args.bench, split)
        t0 = time.time()
        print("[{}] {}: {} instances".format(args.bench, split, len(dirs)), flush=True)
        with concurrent.futures.ProcessPoolExecutor(args.jobs) as ex:
            futs = {ex.submit(label_instance, d, args.bench, split, args.cap, args.retry): d for d in dirs}
            for fu in concurrent.futures.as_completed(futs):
                try:
                    inst, msg = fu.result()
                except Exception as e:     # reported; the instance is retried on the next build
                    inst, msg = os.path.basename(futs[fu]), "ERROR {!r}".format(e)
                print("  {:<34} {}".format(inst, msg), flush=True)
        summarize(args.bench, split)
        print("  done in {:.0f}s".format(time.time() - t0), flush=True)


def summarize(bench, split):
    rows = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(TEACHER, bench, split, "*", "labels.json")))]
    ok = [r for r in rows if r.get("teacher")]
    src = {}
    for r in ok:
        src[r["teacher"]] = src.get(r["teacher"], 0) + 1
    gain = [r["sources"]["cert"]["makespan"] / r["makespan"] for r in ok if "cert" in r["sources"]]
    full = sum(1 for r in rows if r["sources"].get("full", {}).get("outcome") == "valid")
    s = {"split": split, "instances": len(rows), "with_teacher": len(ok), "teacher_source": src,
         "full_popf_solved": full, "mean_cert_over_teacher_makespan": round(float(np.mean(gain)), 3) if gain else None,
         "mean_teacher_fraction_of_objects": round(float(np.mean([r["n_teacher_objects"] / r["n_objects"] for r in ok])), 3)
         if ok else None}
    with open(os.path.join(TEACHER, bench, split, "summary.json"), "w") as f:
        json.dump(s, f, indent=1)
    print("  summary:", json.dumps(s), flush=True)
    return s


# ==================== examples aligned with a model's graph ===========================================
class Example:
    """One instance: graph (x) for a model's vocabulary and teacher targets y [N, 2] in that graph's order."""

    def __init__(self, inst, g, y):
        self.instance, self.graph, self.y = inst, g, y


def examples(bench, split, scorer=None, graph_fn=None):
    """Teacher examples of a split. With a scorer: x from the cache when its vocabulary matches, else rebuilt
    with the scorer (graph_fn(task, gt) overrides the graph builder, e.g. for set scorers)."""
    out = []
    for d in split_dirs(bench, split):
        inst = os.path.basename(d)
        lp = os.path.join(label_dir(bench, split, inst), "labels.json")
        if not os.path.exists(lp):
            continue
        lab = json.load(open(lp))
        if not lab.get("teacher"):
            continue
        g = None
        xp = os.path.join(label_dir(bench, split, inst), "x.npz")
        if graph_fn is None and scorer is not None and os.path.exists(xp):
            g, vh = load_x(xp)
            if vh != vocab_hash(scorer.vocab):
                g = None
        if g is None:
            task = Task.load(d)
            gt = GroundTask(task)
            g = graph_fn(task, gt) if graph_fn else scorer.graph(task, gt)
        idx = {o: i for i, o in enumerate(lab["objects"])}
        y = np.zeros((len(g.objects), 2), dtype=np.float32)
        for i, o in enumerate(g.objects):
            j = idx[o]
            y[i, 0], y[i, 1] = lab["y_obj"][j], lab["y_tmp"][j]
        out.append(Example(inst, g, y))
    return out


def balanced_bce(z, y):
    """BCE where positives and negatives weigh half each (teacher sets are 20-50 % of the objects)."""
    l = nn.functional.binary_cross_entropy_with_logits(z, y, reduction="none")
    pos = y > 0.5
    if pos.all() or (~pos).all():
        return l.mean()
    return 0.5 * l[pos].mean() + 0.5 * l[~pos].mean()


def teacher_loss_terms(z_obj, z_tmp, y):
    """Differentiable teacher loss for one graph: (obj, tmp or None)."""
    y = torch.as_tensor(y)
    lo = balanced_bce(z_obj, y[:, 0])
    lt = nn.functional.binary_cross_entropy_with_logits(z_tmp, y[:, 1]) if z_tmp is not None else None
    return lo, lt


def _auc(s, y):
    pos, neg = s[y > 0.5], s[y <= 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))


def evaluate(logit_fn, exs):
    """logit_fn(graph) -> (z_obj [N] tensor, z_tmp [N] tensor or None). Mean teacher metrics over instances."""
    if not exs:
        return None
    lo, lt, auc, rec = [], [], [], []
    with torch.no_grad():
        for ex in exs:
            zo, zt = logit_fn(ex.graph)
            a, b = teacher_loss_terms(zo, zt, ex.y)
            lo.append(float(a))
            if b is not None:
                lt.append(float(b))
            s, y = zo.numpy(), ex.y[:, 0]
            auc.append(_auc(s, y))
            k = int(y.sum())
            rec.append(float(y[np.argsort(-s)[:k]].sum() / max(1, k)))
    r = {"obj_loss": float(np.mean(lo)), "auc": float(np.nanmean(auc)), "recall_at_k": float(np.mean(rec)), "n": len(exs)}
    if lt:
        r["tmp_loss"] = float(np.mean(lt))
        r["loss"] = r["obj_loss"] + r["tmp_loss"]
    else:
        r["loss"] = r["obj_loss"]
    return r


def scorer_logits(scorer, obj=0, tmp=1):
    """logit_fn for Scorer-like models with model(*tensors) -> [N, heads]."""
    def f(g):
        scorer.model.eval()
        z = scorer.model(*to_tensors(g))
        return z[:, obj], (z[:, tmp] if tmp is not None and z.shape[1] > tmp else None)
    return f


# ==================== train: offline GNN on the teacher labels =========================================
def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    spec = Task.load(split_dirs(args.bench, args.train)[0]).spec
    sc = Scorer(spec, heads=2, temporal_features=True, hidden=args.hidden, steps=args.steps, seed=args.seed)
    tr = examples(args.bench, args.train, sc)
    va = examples(args.bench, args.val, sc)
    te = {s: examples(args.bench, s, sc) for s in args.test}
    print("teacher examples: train {} val {} test {}".format(len(tr), len(va), {k: len(v) for k, v in te.items()}))
    opt = torch.optim.Adam(sc.model.parameters(), lr=args.lr)
    best, state, bad, t0 = None, None, 0, time.time()
    f = scorer_logits(sc)
    for ep in range(args.epochs):
        sc.model.train()
        order = list(range(len(tr)))
        random.shuffle(order)
        for i in order:
            z = sc.model(*to_tensors(tr[i].graph))
            lo, lt = teacher_loss_terms(z[:, 0], z[:, 1], tr[i].y)
            loss = lo + args.lam_t * lt
            opt.zero_grad()
            loss.backward()
            opt.step()
        rec = {"epoch": ep + 1, "train": evaluate(f, tr), "val": evaluate(f, va),
               **{"test:" + k: evaluate(f, v) for k, v in te.items()}}
        sc.history.append(rec)
        line = "epoch {:3d}  train {:.4f}  val {:.4f}".format(ep + 1, rec["train"]["loss"], rec["val"]["loss"])
        for k in te:
            if rec["test:" + k]:
                line += "  test[{}] {:.4f} (AUC {:.3f}, recall@k {:.3f})".format(
                    k, rec["test:" + k]["loss"], rec["test:" + k]["auc"], rec["test:" + k]["recall_at_k"])
        print(line, flush=True)
        if best is None or rec["val"]["loss"] < best - 1e-4:
            best, state, bad = rec["val"]["loss"], {k: t.clone() for k, t in sc.model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= args.patience:
                print("early stop (val loss did not improve for {} epochs)".format(args.patience))
                break
    sc.model.load_state_dict(state)
    sc.calibrate([(e.graph, e.y, np.ones_like(e.y)) for e in va])
    tag = "_" + args.bench
    prefix = os.path.join(MODELS, "{}{}_seed{}".format(args.name, tag, args.seed))
    sc.save(prefix, extra={"method": "TEACHER", "kind": "offline GNN on the cached teacher labels", "bench": args.bench,
                           "train_split": args.train, "val_split": args.val, "best_val_loss": best,
                           "train_seconds": round(time.time() - t0, 1), "args": vars(args)})
    print("saved", prefix)
    plot_history(sc.history, os.path.join(TEACHER, args.bench, "curves_{}{}.png".format(args.name, tag)),
                 "TEACHER (offline, POPF teacher labels)")


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
    b.add_argument("--bench", default="overcooked_kitchen")
    b.add_argument("--splits", nargs="+", required=True)
    b.add_argument("--cap", type=float, default=60.0, help="POPF seconds per teacher source")
    b.add_argument("--jobs", type=int, default=3, help="instances in parallel (each runs up to 4 POPF)")
    b.add_argument("--retry", action="store_true", help="recompute instances cached without a teacher plan")
    im = sub.add_parser("improve")
    im.add_argument("--bench", default="overcooked_kitchen")
    im.add_argument("--splits", nargs="+", required=True)
    im.add_argument("--jobs", type=int, default=4)
    t = sub.add_parser("train")
    t.add_argument("--bench", default="overcooked_kitchen")
    t.add_argument("--train", default="v2_train")
    t.add_argument("--val", default="v2_val")
    t.add_argument("--test", nargs="*", default=["v2_test"])
    t.add_argument("--epochs", type=int, default=300)
    t.add_argument("--patience", type=int, default=40)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--hidden", type=int, default=16)
    t.add_argument("--steps", type=int, default=3)
    t.add_argument("--lam_t", type=float, default=1.0)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--name", default="TEACHER", help="model name prefix (TEACHER2: on the improved labels)")
    lo = sub.add_parser("losses")
    lo.add_argument("--bench", default="overcooked_kitchen")
    lo.add_argument("--models", nargs="*")
    lo.add_argument("--splits", nargs="*")
    pl = sub.add_parser("plot")
    pl.add_argument("--bench", default="overcooked_kitchen")
    args = ap.parse_args(argv)
    {"build": build, "improve": improve, "train": train, "losses": losses, "plot": plot}[args.cmd](args)


if __name__ == "__main__":
    from algorithm.planning import ensure_hash_seed
    ensure_hash_seed()
    main()
