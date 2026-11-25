"""Closed-loop evaluation on FactorySim instances (the hard suite: FACTORY_SUITE=hard), random durations.

    FACTORY_SUITE=hard python -m factory_test.closed_eval --split test --methods dispatcher open-loop replan-2 V13_hard V14_hard \
        --noise 1 2 3 --seeds 3 [--procs 6] [--tag _ablations] [--replan_sizes M L XL]

Methods (the factory counterpart of overcooked_test/v3_eval.py; one rollout per instance x seed x noise level)
  dispatcher   the rule-based robot dispatcher (factory_dispatcher.py), deciding online (closed loop)
  open-loop    the dispatcher's plan at nominal durations from the start, executed time-triggered, never revised
  replan-<d>   that plan, re-planned (the dispatcher simulated at nominal durations from the actual state) when a robot
               is more than d steps late (at most 200 times); only on the sizes in --replan_sizes (default M, L: each
               re-plan is a full rollout)
  V12* / V13* / V14*   a trained model of that name (results_hard/models/<name>), greedy, closed loop
Success: every order delivered by its due date (reason "late": an order was not; "stuck" / "dead": nothing could
happen any more, e.g. a blocking or traffic deadlock; "horizon": 4 x the reference makespan passed). Writes results_hard/v13/eval_<split><tag>.jsonl (one row per run;
re-running resumes) and .md (tables by size and by family per noise level).
"""
import argparse
import json
import multiprocessing as mp
import os
import time
from collections import defaultdict

import numpy as np

import factory_test  # noqa: F401  (paths)
from factory_test import factory_v13 as K
from algorithm import v13
from algorithm.menv import MAEnv, PlanFollower, Replanner, sim_planner

v13.use(K)
OUT = v13.OUT
_W = {}


def _init():
    import torch
    torch.set_num_threads(1)
    _W["tasks"], _W["pol"] = {}, {}


def _task(split, name):
    tasks, key = _W["tasks"], (split, name)
    if key in tasks:
        tasks[key] = tasks.pop(key)
    else:
        while len(tasks) >= 2:                   # the large plants' grounded tasks take GBs: keep two per worker
            tasks.pop(next(iter(tasks)))
        tasks[key] = K.task(split, name)
    return tasks[key]


def late(T, r):
    """The run failed because an order was not delivered by its due date (hard deadlines: (open j) ended)."""
    due = {T.facts[f][1]: t for f, t in getattr(T, "window", {}).items() if T.facts[f][0] == "open"}
    done = {a.args[1] for _, a, _ in r["steps"] if a.name.endswith("unload-sink")}
    return any(t <= r["t_end"] + 1e-6 for j, t in due.items() if j not in done)


def plan_text(steps):
    return "\n".join("{:.3f}: ({}) [{}]".format(t, " ".join((a.name,) + tuple(a.args)), d if d is not None else a.lo) for t, a, d in steps)


def _policy(method):
    if method not in _W["pol"]:
        mod = __import__("algorithm." + method[:3].lower(), fromlist=["load"])
        _W["pol"][method] = mod.load(os.path.join(v13.MODELS, method))
    return _W["pol"][method]


def _run(job):
    split, name, cert, method, seed, nk = job
    from factory_test.factory_dispatcher import Dispatcher
    T = _task(split, name)
    env = MAEnv(T, K.AGENT, noise=v13.NOISE.scaled(nk), seed=seed, horizon=4 * cert)
    extra = {}
    t0 = time.time()
    if method == "dispatcher":
        ctl = Dispatcher(env)
    elif method == "open-loop":
        steps = sim_planner(env, lambda sim: Dispatcher(sim))(env.reset(seed))
        ctl = PlanFollower(env, plan_text(steps), timed=True)
    elif method.startswith("replan-"):
        ctl = Replanner(env, sim_planner(env, lambda sim: Dispatcher(sim)), tol=float(method.split("-")[1]))
    elif method[:3] in ("V12", "V13", "V14", "V15"):
        mod = __import__("algorithm." + method[:3].lower(), fromlist=["Controller"])
        ctl = (mod.Controller if method[:3] == "V12" else v13.Controller)(env, _policy(method), "greedy", label=False)
    else:
        raise ValueError(method)
    r = env.rollout(ctl, seed=seed)
    secs = time.time() - t0
    if not r["success"] and late(T, r):
        r["reason"] = "late"
    if method.startswith("replan-"):
        extra = {"replans": ctl.replans, "plan_s": round(ctl.plan_time, 2)}
    return {"split": split, "instance": name, "method": method, "seed": seed, "noise": nk, "success": r["success"],
            "reason": r["reason"], "served": r["served"], "goals": r["goals"],
            "ms_cert": round(r["makespan"] / cert, 4) if r["success"] else None, "seconds": round(secs, 2), **extra}


def table(rows, group, title):
    by = defaultdict(list)
    for r in rows:
        by[(r["noise"], r["method"], group[r["instance"]])].append(r)
        by[(r["noise"], r["method"], "all")].append(r)
    lines = []
    for nk in sorted({r["noise"] for r in rows}):
        lines.append("\nnoise x{} (by {})".format(nk, title))
        lines.append("| Method | {} | Runs | Success | Orders served | Makespan / cert | s / run | Replans |".format(title.capitalize()))
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for m in dict.fromkeys(r["method"] for r in rows):
            for t in ["all"] + sorted(set(group.values()), key=lambda x: (ORDER.get(x, 99), x)):
                rr = by.get((nk, m, t))
                if not rr:
                    continue
                ms = [r["ms_cert"] for r in rr if r["ms_cert"]]
                rp = [r.get("replans", 0) for r in rr]
                lines.append("| {} | {} | {} | {:.1%} | {:.1%} | {} | {:.1f} | {} |".format(
                    m, t, len(rr), np.mean([r["success"] for r in rr]), np.mean([r["served"] / r["goals"] for r in rr]),
                    "{:.3f}".format(np.mean(ms)) if ms else "-", np.mean([r["seconds"] for r in rr]),
                    "{:.1f}".format(np.mean(rp)) if any(rp) else "-"))
    return "\n".join(lines)


ORDER = {s: i for i, s in enumerate(("S", "M", "L", "XL", "XXL", "XXXL"))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test")
    ap.add_argument("--methods", nargs="+", default=["dispatcher", "open-loop", "replan-2"])
    ap.add_argument("--noise", type=float, nargs="+", default=[1.0, 2.0, 3.0])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--tag", default="")
    ap.add_argument("--sizes", nargs="*", default=None, help="only these sizes (default: all)")
    ap.add_argument("--replan_sizes", nargs="*", default=["M", "L"], help="re-planning only on these sizes (slow)")
    ap.add_argument("--max_size", nargs="*", default=[], metavar="METHOD:SIZE",
                    help="run METHOD only up to SIZE, e.g. V12_hard:XL")
    a = ap.parse_args()
    sp = [x for x in v13.split_rows(a.split) if a.sizes is None or x[3]["size"] in a.sizes]
    size = {n: inf["size"] for n, _, _, inf in sp}
    fam = {n: inf["family"] for n, _, _, inf in sp}
    cap = {m: ORDER[z] for m, z in (x.split(":") for x in a.max_size)}
    jobs = [(a.split, n, c, m, 9000 + s, nk) for nk in a.noise for m in a.methods for n, _, c, _ in sp for s in range(a.seeds)
            if not (m.startswith("replan-") and size[n] not in a.replan_sizes) and ORDER.get(size[n], 0) <= cap.get(m, 99)]
    jobs.sort(key=lambda j: -ORDER.get(size[j[1]], 0))      # largest first: the long runs do not trail at the end
    os.makedirs(OUT, exist_ok=True)
    out = os.path.join(OUT, "eval_{}{}.jsonl".format(a.split, a.tag))
    done = set()
    if os.path.exists(out):
        for line in open(out):
            r = json.loads(line)
            done.add((r["instance"], r["method"], r["seed"], r["noise"]))
    jobs = [j for j in jobs if (j[1], j[3], j[4], j[5]) not in done]
    print("{} runs to do ({} already done)".format(len(jobs), len(done)), flush=True)
    t0 = time.time()
    with mp.get_context("fork").Pool(a.procs, _init, maxtasksperchild=20) as pool, open(out, "a") as f:
        for k, r in enumerate(pool.imap_unordered(_run, jobs, chunksize=1)):
            f.write(json.dumps(r) + "\n")
            f.flush()
            if (k + 1) % 20 == 0:
                print("  {}/{} runs ({:.0f}s)".format(k + 1, len(jobs), time.time() - t0), flush=True)
    rows = [json.loads(line) for line in open(out)]
    rows = [r for r in rows if r["noise"] in a.noise and r["instance"] in size]      # every method run into this file
    md = table(rows, size, "size") + "\n" + table(rows, fam, "family")
    open(os.path.join(OUT, "eval_{}{}.md".format(a.split, a.tag)), "w").write(md + "\n")
    print(md, flush=True)


if __name__ == "__main__":
    main()
