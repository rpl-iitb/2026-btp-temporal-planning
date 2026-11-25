"""factory-sim command line: run, families, generate, verify, bench, pddl, solve."""
import argparse
import importlib
import importlib.util
import json
import os
import re
import sys
import time
from collections import defaultdict

import numpy as np

from . import baselines, scenarios
from .config import Factory
from .env import build_sim


def _policy(name, seed=0):
    if ":" in name:
        mod, fn = name.rsplit(":", 1)
        if mod.endswith(".py"):
            spec = importlib.util.spec_from_file_location("user_policy", mod)
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            return getattr(m, fn)
        sys.path.insert(0, os.getcwd())
        return getattr(importlib.import_module(mod), fn)
    return baselines.make(name, seed=seed)


def _instance(arg):
    return scenarios.load(arg) if arg.endswith(".json") else arg


def cmd_run(a):
    inst = _instance(a.instance)
    sim = build_sim(inst, seed=a.seed, record=bool(a.gif or a.gantt), **({"time_source": a.time_source} if a.time_source else {}))
    t0 = time.time()
    m = sim.run(_policy(a.policy, a.seed))
    m["wall_s"] = round(time.time() - t0, 3)
    print(json.dumps(m, indent=1))
    if a.gif:
        from .render import gif
        print("gif:", gif(sim, a.gif, dt=a.dt))
    if a.gantt:
        from .render import gantt
        print("gantt:", gantt(sim, a.gantt))
    if a.rfid:
        json.dump(sim.rfid, open(a.rfid, "w"), indent=1)
        print("rfid log:", a.rfid)


def cmd_families(a):
    for f, d in scenarios.FAMILIES.items():
        print("{:<13} {}".format(f, ", ".join("{}={}".format(k, v) for k, v in d.items()) or "(base)"))
    print("\nbase settings:", scenarios.BASE)
    print("sizes:", scenarios.SIZES)


def cmd_robots(a):
    """List the robot types (or the fleet of one config): parameters, programming interfaces and domain actions."""
    from . import pddl, robots
    if a.instance:
        F = Factory(_instance(a.instance)["config"]) if a.instance.endswith(".json") else Factory(a.instance)
        print("fleet of {}:".format(a.instance))
        for i, x in enumerate(F.fleet):
            print("  r{}  {:<16} {:<8} speed {} m/s  load {}  unload {}".format(i + 1, x["type"], x["handling"], x["speed"], x["load"], x["unload"]))
        print("PDDL actions:", ", ".join(re.findall(r"durative-action (\S+)", pddl.domain(F))))
        if "handler" in F.handling:
            print("handler pool(s):", ", ".join("{} x{}".format(p, F.operators[p]) for p in sorted({s.get("handler", F.handler_pool) for s in F.stations.values()})))
        return
    for name in robots.TYPES:
        print(robots.describe(name) + "\n")
    print("planned (not simulated yet):")
    for name, text in robots.PLANNED.items():
        print("  {}: {}".format(name, text))


def cmd_generate(a):
    t0 = time.time()
    a.out = a.out or "benchmarks/" + (("v" + a.version) if a.version[0].isdigit() else a.version)
    suite = scenarios.SUITES[a.version]
    idx = scenarios.generate_suite(a.out, suite=suite, splits=a.splits, families=a.families, count=a.count, cpsat_limit=a.cpsat, bound_limit=a.bounds)
    print("{} instances in {} ({:.0f}s)".format(len(idx), a.out, time.time() - t0))


def cmd_bench(a):
    rows = []
    t_all = time.time()
    for inst in scenarios.iter_suite(a.suite, a.split, a.families):
        gt = inst["ground_truth"]
        sched = None
        if a.policy == "cpsat":
            from . import ground_truth as G
            t0 = time.time()
            sched = G.cpsat(Factory(inst["config"]), inst["orders"], time_limit=a.cpsat_time)
            solve_s = time.time() - t0
        for s in range(a.seeds):
            sim = build_sim(inst, seed=inst["seed"] * 1000 + s)
            t0 = time.time()
            if sched is not None:
                from . import ground_truth as G
                m = sim.run(G.SchedulePolicy(sched)) if sched.get("robots") is not None and "makespan" in sched else dict(sim.metrics(), success=False, failure="no schedule")
                t0 -= solve_s
            else:
                m = sim.run(_policy(a.policy, s))
            rows.append({"name": inst["name"], "family": inst["family"], "size": inst["size"], "seed": s, "success": m["success"], "failure": m["failure"], "makespan": m["makespan"], "on_time": m["on_time"], "total_tardiness": m["total_tardiness"],
                         "makespan_over_lb": (m["makespan"] / gt["lower_bound"]) if m["makespan"] else None, "makespan_over_ref": (m["makespan"] / gt["reference_makespan"]) if (m["makespan"] and gt.get("reference_makespan")) else None, "makespan_over_best": (m["makespan"] / gt["best_known"]) if (m["makespan"] and gt.get("best_known")) else None, "weighted_tardiness": m.get("weighted_tardiness"), "decisions": m["decisions"], "wall_s": round(time.time() - t0, 3)})
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(summary(rows, a.policy))
    print("\n{} runs in {:.0f}s -> {}".format(len(rows), time.time() - t_all, a.out))


def cmd_verify(a):
    """Recompute stored ground truth (reference makespan, lower bound) for a sample of a suite with this code."""
    from . import __version__
    idx = scenarios.load(os.path.join(a.suite, "index.json"))
    print("suite {} generated by {} (this code: {})".format(idx.get("suite"), idx.get("code_version"), __version__))
    insts = idx["instances"][::max(1, len(idx["instances"]) // a.sample)][:a.sample]
    bad = 0
    for e in insts:
        inst = scenarios.load(os.path.join(a.suite, e["path"]))
        gt, now = inst["ground_truth"], scenarios.ground_truth(inst)
        lb = lambda g: max((g.get("lower_bound_parts") or {}).values(), default=g.get("lower_bound") or 0)  # noqa: E731  (analytic part; the CP-SAT bound is not recomputed)
        same = (abs((gt.get("reference_makespan") or 0) - (now.get("reference_makespan") or 0)) < 1e-3 and abs(lb(gt) - lb(now)) < 1e-3) or inst["family"] == "exact"
        bad += not same
        if not same or a.verbose:
            print("{:<28} stored ref {} LB {}   now ref {} LB {}".format(e["name"], gt.get("reference_makespan"), gt.get("lower_bound"), now.get("reference_makespan"), now.get("lower_bound")))
    print("{} / {} instances reproduce".format(len(insts) - bad, len(insts)))
    sys.exit(1 if bad else 0)


def cmd_table(a):
    """Compare bench result files (one per policy): success rate and mean makespan / best-known per family."""
    res = {}
    for path in a.files:
        pol = os.path.splitext(os.path.basename(path))[0]
        for line in open(path):
            r = json.loads(line)
            res.setdefault(r["family"], {}).setdefault(pol, []).append(r)
    pols = [os.path.splitext(os.path.basename(p))[0] for p in a.files]
    out = ["| family | " + " | ".join(pols) + " |", "| --- |" + " --- |" * len(pols)]
    for fam in sorted(res):
        cells = []
        for p in pols:
            rr = res[fam].get(p, [])
            ok = [r["makespan_over_best"] for r in rr if r.get("makespan_over_best")]
            cells.append("{:.0%} / {}".format(np.mean([r["success"] for r in rr]), "{:.2f}".format(np.mean(ok)) if ok else "-") if rr else "")
        out.append("| {} | {} |".format(fam, " | ".join(cells)))
    print("success rate / mean makespan over best-known (successful runs)\n")
    print("\n".join(out))


def summary(rows, policy):
    by = defaultdict(list)
    for r in rows:
        by[(r["family"], r["size"])].append(r)
        by[("all", "")].append(r)
    mean = lambda xs: "{:.3f}".format(float(np.mean(xs))) if xs else "-"  # noqa: E731
    lines = ["policy: {}".format(policy), "", "| family | size | runs | success | deadlock/stall | makespan / LB | makespan / ref | on-time | s / run |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for (fam, size), rr in sorted(by.items(), key=lambda x: (x[0][0] == "all", x[0])):
        lines.append("| {} | {} | {} | {:.0%} | {:.0%} | {} | {} | {:.0%} | {} |".format(fam, size, len(rr), np.mean([r["success"] for r in rr]), np.mean([r["failure"] in ("deadlock", "stalled") for r in rr]),
                                                                                    mean([r["makespan_over_lb"] for r in rr if r["makespan_over_lb"]]), mean([r["makespan_over_ref"] for r in rr if r["makespan_over_ref"]]), np.mean([r["on_time"] for r in rr]), mean([r["wall_s"] for r in rr])))
    return "\n".join(lines)


def cmd_pddl(a):
    from . import pddl
    sim = build_sim(_instance(a.instance), seed=a.seed, time_source="nominal")
    os.makedirs(a.out, exist_ok=True)
    d, p = pddl.write(sim, os.path.join(a.out, "domain.pddl"), os.path.join(a.out, "problem.pddl"), hard_deadlines=a.hard_deadlines)
    print("wrote", d, p)


def cmd_solve(a):
    from . import ground_truth as G
    inst = _instance(a.instance)
    if isinstance(inst, dict):
        F, orders = Factory(inst["config"]), inst["orders"]
    else:
        sim = build_sim(inst, seed=a.seed, time_source="nominal")
        F, orders = sim.F, [{"id": j.id, "product": j.product, "arrival": j.arrival, "due": j.due} for j in sim.jobs.values()]
    lb, parts = G.lower_bound(F, orders)
    sol = G.cpsat(F, orders, time_limit=a.time_limit)
    print(json.dumps({"lower_bound": lb, "parts": parts, "cpsat": {k: v for k, v in sol.items() if k in ("status", "makespan", "bound", "exact")}}, indent=1))
    if a.out:
        json.dump(sol, open(a.out, "w"), indent=1, default=str)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="factory-sim", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run one episode")
    r.add_argument("--instance", default="lab", help="built-in config name, config .yaml, or instance .json")
    r.add_argument("--policy", default="edd", help=", ".join(baselines.RULES) + " or module:function")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--time-source", choices=("sampled", "nominal"))
    r.add_argument("--gif"), r.add_argument("--gantt"), r.add_argument("--rfid")
    r.add_argument("--dt", type=float, default=15.0, help="simulated seconds per GIF frame")
    sub.add_parser("families", help="list scenario families")
    rb = sub.add_parser("robots", help="list robot types, or the fleet and PDDL actions of a config")
    rb.add_argument("instance", nargs="?", help="config name / .yaml / instance .json")
    g = sub.add_parser("generate", help="generate the benchmark suite")
    g.add_argument("--out", help="default benchmarks/<version>")
    g.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    g.add_argument("--families", nargs="+")
    g.add_argument("--count", type=int, help="instances per family and size (default: suite counts)")
    g.add_argument("--version", default="2.0", choices=list(scenarios.SUITES),
                   help="2.0: the people-free default lab (arm robots, no operators), families that need no people + disturbed; "
                        "1.0: the 25 families of suite 1.0 on the 1.x lab with its workforce (lab_v1); 1.1: + robot-fleet families "
                        "(tray_fleet, mixed_fleet); v1 / v2: earlier family sets")
    g.add_argument("--bounds", type=float, default=0.0, help="CP-SAT seconds per test-split M instance for relaxation lower bounds (0 = skip)")
    g.add_argument("--cpsat", type=float, default=30.0, help="CP-SAT seconds for the exact family (0 = skip)")
    v = sub.add_parser("verify", help="check that this code reproduces a suite's stored ground truth")
    v.add_argument("--suite", default="benchmarks/v2.0")
    v.add_argument("--sample", type=int, default=40)
    v.add_argument("--verbose", action="store_true")
    b = sub.add_parser("bench", help="evaluate a policy on a suite split")
    b.add_argument("--suite", default="benchmarks/v2.0")
    b.add_argument("--split", default="test")
    b.add_argument("--policy", default="edd", help="a rule (" + ", ".join(baselines.RULES) + "), cpsat (open-loop CP-SAT schedule), or module:function")
    b.add_argument("--cpsat-time", type=float, default=10.0, help="CP-SAT seconds per instance for --policy cpsat")
    b.add_argument("--families", nargs="+")
    b.add_argument("--seeds", type=int, default=3, help="episodes per instance (different random durations)")
    b.add_argument("--out", default="results/bench.jsonl")
    tb = sub.add_parser("table", help="compare bench result files (one JSONL per policy)")
    tb.add_argument("files", nargs="+")
    p = sub.add_parser("pddl", help="export PDDL 2.1 domain and problem")
    p.add_argument("--instance", default="lab")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="pddl_out")
    p.add_argument("--hard-deadlines", action="store_true")
    s = sub.add_parser("solve", help="lower bound and CP-SAT schedule")
    s.add_argument("--instance", default="lab")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--time-limit", type=float, default=60.0)
    s.add_argument("--out")
    a = ap.parse_args(argv)
    {"run": cmd_run, "families": cmd_families, "robots": cmd_robots, "generate": cmd_generate, "verify": cmd_verify, "bench": cmd_bench, "table": cmd_table, "pddl": cmd_pddl, "solve": cmd_solve}[a.cmd](a)


if __name__ == "__main__":
    main()
