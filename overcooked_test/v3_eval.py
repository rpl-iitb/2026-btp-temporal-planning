"""Closed-loop evaluation on the multi-cook kitchens (v3_* splits of data/: 3-4 cooks, random durations).

    python -m overcooked_test.v3_eval --split v3_test --methods dispatcher open-loop replan-2 V13 --noise 1 2 3 --seeds 3

Methods
  dispatcher   rule-based multi-cook dispatcher, decides online (closed loop)
  open-loop    the certificate plan (nominal durations), executed time-triggered, never revised
  replan-<d>   plan with nominal durations (dispatcher simulated without noise), execute time-triggered, re-plan
               from the actual state when a cook is more than d steps late
  V13          learned task choice (imitation + RL), greedy, closed loop
  V12          learned PDDL-move policy (GNN, imitation + RL), greedy, closed loop
  V14          tropical event-graph network (max-plus GNN, imitation + RL), greedy, closed loop
Writes results/v13/eval_<split>.jsonl (one row per run) and prints a table per noise level and tier.
"""
import argparse
import json
import multiprocessing as mp
import os
import time
from collections import defaultdict

import numpy as np

import overcooked_test  # noqa: F401  (paths)
from overcooked_test import kitchen_v13 as K
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
        tasks[key] = K.task(split, name)
        while len(tasks) > v13.TASK_CACHE:
            tasks.pop(next(iter(tasks)))
    return tasks[key]


def _run(job):
    split, name, cert, method, seed, nk = job
    from overcooked_test.kitchen_dispatcher import Dispatcher
    T = _task(split, name)
    env = MAEnv(T, "cook", noise=v13.NOISE.scaled(nk), seed=seed, horizon=4 * cert)
    extra = {}
    t0 = time.time()
    if method == "dispatcher":
        ctl = Dispatcher(env)
    elif method == "open-loop":
        ctl = PlanFollower(env, open(os.path.join(K.V3, split, name, "certificate.plan")).read(), timed=True)
    elif method.startswith("replan-"):
        ctl = Replanner(env, sim_planner(env, lambda sim: Dispatcher(sim)), tol=float(method.split("-")[1]))
    elif method.startswith("V13"):
        if method not in _W["pol"]:
            _W["pol"][method] = v13.load(os.path.join(v13.MODELS, method + "_v3" if method == "V13" else method))
        ctl = v13.Controller(env, _W["pol"][method], "greedy", label=False)
    elif method.startswith("V14"):
        from algorithm import v14
        if method not in _W["pol"]:
            _W["pol"][method] = v14.load(os.path.join(v13.MODELS, method + "_v3" if method == "V14" else method))
        ctl = v13.Controller(env, _W["pol"][method], "greedy", label=False)
    elif method.startswith("V12"):
        from algorithm import v12
        if method not in _W["pol"]:
            _W["pol"][method] = v12.load(os.path.join(v13.MODELS, method + "_v3" if method == "V12" else method))
        ctl = v12.Controller(env, _W["pol"][method], "greedy", label=False)
    else:
        raise ValueError(method)
    r = env.rollout(ctl, seed=seed)
    secs = time.time() - t0
    if method.startswith("replan-"):
        extra = {"replans": ctl.replans, "plan_s": round(ctl.plan_time, 2)}
    return {"split": split, "instance": name, "method": method, "seed": seed, "noise": nk, "success": r["success"],
            "reason": r["reason"], "served": r["served"], "goals": r["goals"],
            "ms_cert": round(r["makespan"] / cert, 4) if r["success"] else None, "seconds": round(secs, 2), **extra}


def table(rows, tiers):
    by = defaultdict(list)
    for r in rows:
        by[(r["noise"], r["method"], tiers[r["instance"]])].append(r)
        by[(r["noise"], r["method"], "all")].append(r)
    lines = []
    for nk in sorted({r["noise"] for r in rows}):
        lines.append("\nnoise x{}".format(nk))
        lines.append("| Method | Tier | Runs | Success | Orders served | Makespan / cert | s / run | Replans |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for m in dict.fromkeys(r["method"] for r in rows):
            for t in ["all"] + sorted(set(tiers.values())):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="v3_test")
    ap.add_argument("--methods", nargs="+", default=["dispatcher", "open-loop", "replan-2", "V13"])
    ap.add_argument("--noise", type=float, nargs="+", default=[1.0, 2.0, 3.0])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    sp = v13.split_rows(a.split)
    tiers = {n: inf["tier"] for n, _, _, inf in sp}
    jobs = [(a.split, n, c, m, 9000 + s, nk) for nk in a.noise for m in a.methods for n, _, c, _ in sp for s in range(a.seeds)]
    os.makedirs(OUT, exist_ok=True)
    out = os.path.join(OUT, "eval_{}{}.jsonl".format(a.split, a.tag))
    done = set()
    if os.path.exists(out):
        for l in open(out):
            r = json.loads(l)
            done.add((r["instance"], r["method"], r["seed"], r["noise"]))
    jobs = [j for j in jobs if (j[1], j[3], j[4], j[5]) not in done]
    print("{} runs to do ({} already done)".format(len(jobs), len(done)), flush=True)
    t0 = time.time()
    with mp.get_context("fork").Pool(a.procs, _init) as pool, open(out, "a") as f:
        for k, r in enumerate(pool.imap_unordered(_run, jobs, chunksize=1)):
            f.write(json.dumps(r) + "\n"); f.flush()
            if (k + 1) % 50 == 0:
                print("  {}/{} runs ({:.0f}s)".format(k + 1, len(jobs), time.time() - t0), flush=True)
    rows = [json.loads(l) for l in open(out)]
    rows = [r for r in rows if r["method"] in a.methods and r["noise"] in a.noise]
    md = table(rows, tiers)
    open(os.path.join(OUT, "eval_{}{}.md".format(a.split, a.tag)), "w").write(md + "\n")
    print(md, flush=True)


if __name__ == "__main__":
    main()
