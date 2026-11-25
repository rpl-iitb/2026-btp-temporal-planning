"""overcooked-kitchen command line: run, play, import, generate, bench, pddl, solve."""
"""This file is UN-VERIFIED"""
import argparse
import json
import os
import sys
import tempfile
import time

import yaml

from overcooked_kitchen import suite


def cmd_run(a):
    from overcooked_kitchen.core import Sim
    noise = None if a.noise == "none" or a.cv_scale == 0 else a.noise
    sim = Sim(a.scenario, clock=a.clock, seed=a.seed, noise=noise, cv_scale=a.cv_scale, record=bool(a.gif or a.gantt))
    m = sim.run(suite.load_policy(a.policy, a.seed))
    print(json.dumps(m, indent=1))
    if a.gif:
        from overcooked_kitchen.render import gif
        print("gif:", gif(sim, a.gif))
    if a.gantt:
        from overcooked_kitchen.render import gantt
        print("gantt:", gantt(sim, a.gantt))
    return 0 if m["success"] else 1


def cmd_import(a):
    rows = suite.import_existing(a.out, a.v2, a.v3, check=not a.no_check)
    kinds = {}
    for r in rows:
        kinds[r["cert_kind"]] = kinds.get(r["cert_kind"], 0) + 1
    print("{} instances -> {}, certificates {}".format(len(rows), os.path.join(a.out, "index.json"), kinds))
    for r in rows:
        if r["cert_valid"] is False:
            print("  {}: {}".format(r["name"], r["cert_note"]))


def cmd_generate(a):
    from overcooked_kitchen.generate import generate
    with open(a.config) as f:
        config = yaml.safe_load(f)
    t0 = time.time()
    try:
        rows = generate(a.out, config, a.splits, seed=a.seed, count=a.count, jobs=a.jobs, force=a.force)
    except (FileExistsError, ValueError) as e:
        sys.exit(str(e))
    kinds = {}
    for r in rows:
        kinds[r["cert_source"]] = kinds.get(r["cert_source"], 0) + 1
    print("{} instances in {} ({:.0f}s), certificates {}".format(len(rows), a.out, time.time() - t0, kinds))
    for split in [sp["dataset"] for sp in config["splits"] if not a.splits or sp["dataset"] in a.splits]:
        path = os.path.join(a.out, split, "dataset.json")
        if not os.path.isfile(path):
            continue
        with open(path) as f:
            ds = json.load(f)
        for tier, n in ds["requested"].items():
            if ds["made"][tier] < n:
                print("warning: {} tier {}: made {} of {} requested (raise max_tries_factor)".format(
                    split, tier, ds["made"][tier], n))


def cmd_bench(a):
    t0 = time.time()
    rows = suite.bench(a.suite, a.split, a.policy, seeds=a.seeds, clock=a.clock, cv_scale=a.cv_scale, out=a.out)
    print("policy: {}  split: {}  clock: {}  cv_scale: {}\n".format(a.policy, a.split, a.clock, a.cv_scale))
    print(suite.table(rows))
    print("\n{} runs in {:.0f}s{}".format(len(rows), time.time() - t0, " -> " + a.out if a.out else ""))


def cmd_pddl(a):
    from overcooked_kitchen import pddl
    from overcooked_kitchen.core import Sim
    d, p = pddl.write(Sim(a.scenario), a.out)
    print("wrote", d, p)


def cmd_solve(a):
    from overcooked_kitchen import pddl
    from overcooked_kitchen.core import Sim
    from overcooked_kitchen.pddlkit import planners
    if not planners.available(a.planner):
        sys.exit("{} not found: set $OVERCOOKED_KITCHEN_{} or put it on PATH".format(a.planner, a.planner.upper()))
    with tempfile.TemporaryDirectory() as tmp:
        d, p = pddl.write(Sim(a.scenario), tmp)
        res = planners.run_temporal_planner(d, p, timeout=a.timeout, planner=a.planner)
        valid = pddl.validate(d, p, res.plan) if res.plan is not None and planners.available("validate") else None
    out = {"status": res.status, "makespan": round(res.plan.makespan, 3) if res.plan else None,
           "solve_s": round(res.runtime_s, 3), "val_valid": valid}
    print(json.dumps(out, indent=1))
    if res.plan is not None and a.plan:
        with open(a.plan, "w") as f:
            f.write(res.plan.to_text())


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["play"]:
        from overcooked_kitchen import play
        return play.main(argv[1:])
    ap = argparse.ArgumentParser(prog="overcooked-kitchen", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run one episode and print its metrics")
    r.add_argument("--scenario", default="k01_one_soup", help="example name or scenario .yaml")
    r.add_argument("--policy", default="dispatcher", help="random, dispatcher, popf, popf-replan or module:function")
    r.add_argument("--clock", default="sampled", choices=("sampled", "grid"))
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--noise", default="none", choices=("none", "default"), help="duration noise (default: nominal)")
    r.add_argument("--cv-scale", type=float, default=1.0, help="scales the noise CVs (0 = nominal)")
    r.add_argument("--gif")
    r.add_argument("--gantt")
    pl = sub.add_parser("play", help="play a scenario (pygame); options as overcooked_kitchen.play")
    pl.add_argument("--scenario", default="k01_one_soup")
    pl.add_argument("--cooks", type=int, default=1)
    im = sub.add_parser("import", help="import the v2 / v3 instances and certificates into a suite")
    im.add_argument("--out", required=True)
    im.add_argument("--v2", help="v2 source folder (scenarios/<split>/<id>.yaml + <id>.certificate.plan)")
    im.add_argument("--v3", help="v3 source folder (generated_pddl/<split>/<id>/scenario.yaml + certificate.plan)")
    im.add_argument("--no-check", action="store_true", help="do not VAL-check / replay the certificates")
    g = sub.add_parser("generate", help="generate certified kitchens from a dataset config")
    g.add_argument("--config", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--splits", nargs="*")
    g.add_argument("--count", type=int, help="instances per tier (overrides the config)")
    g.add_argument("--seed", type=int, default=0, help="offset added to the config seeds")
    g.add_argument("--jobs", type=int, default=1)
    g.add_argument("--force", action="store_true", help="replace splits that already exist in --out")
    b = sub.add_parser("bench", help="evaluate a policy on a suite split")
    b.add_argument("--suite", required=True)
    b.add_argument("--split", required=True)
    b.add_argument("--policy", default="dispatcher")
    b.add_argument("--seeds", type=int, default=3)
    b.add_argument("--clock", default="sampled", choices=("sampled", "grid"))
    b.add_argument("--cv-scale", type=float, default=1.0, help="0 = nominal durations")
    b.add_argument("--out", help="JSONL rows")
    p = sub.add_parser("pddl", help="write domain.pddl and problem.pddl of a scenario's initial state")
    p.add_argument("--scenario", required=True)
    p.add_argument("--out", default="pddl_out")
    s = sub.add_parser("solve", help="POPF / OPTIC on a scenario's initial state, checked by VAL")
    s.add_argument("--scenario", required=True)
    s.add_argument("--planner", default="popf", choices=("popf", "optic"))
    s.add_argument("--timeout", type=float, default=300.0)
    s.add_argument("--plan", help="write the plan here")
    a = ap.parse_args(argv)
    return {"run": cmd_run, "import": cmd_import, "generate": cmd_generate, "bench": cmd_bench, "pddl": cmd_pddl,
            "solve": cmd_solve}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
