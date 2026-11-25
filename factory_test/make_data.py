"""Export FactorySim instances (its own scenario families, factory_sim/scenarios.py) as instance folders for
algorithm/ and baselines/:

    python -m factory_test.make_data [--families base bottleneck tight noisy ...] [--sizes S M] [--count 5]

data/<split>/<name>/  domain.pddl, problem.pddl  (FactorySim's PDDL 2.1 export of the factory at t = 0, nominal view)
                      domain.yaml, hl_problem.json (the same, in the algorithm's format: factory_test/pddl_convert.py)
                      instance.json               (the FactorySim instance: config, orders, simulator settings, ground truth)
data/<split>/dataset.json  {"instances": [{"instance_id", "family", "size", "best_known", "lower_bound"}]}
Splits: train / val / test as in FactorySim (test = larger factories than training).

    FACTORY_SUITE=hard python -m factory_test.make_data --suite hard [--procs 8]     (-> data_hard/)

The hard suite (FactorySim's SUITE_HARD): the hard_* families (lane traffic: corridor junctions hold one robot;
due dates from the workload; contention, blocking buffers, alternative routes, online orders, rush orders), sizes
S-L (the lab) for training and M-XXXL (up to a 4-line plant with 12 robots and 56 orders) for testing, and PDDL with
hard deadlines (an order cannot be delivered after its due date). Per instance: FactorySim's reference rules give a
makespan; the robot dispatcher (factory_dispatcher.py) runs on the PDDL in algorithm/menv.py at nominal durations;
the due dates are set from the slower of the two (so the dispatcher is on time at nominal durations under loose
due dates), best_known is the faster. Instances that neither finishes are skipped (the next index is used).
"""
import argparse
import json
import os

import yaml

import factory_test  # noqa: F401  (paths)
from factory_test import DATA
from factory_test.pddl_convert import read_domain, read_problem
from factory_sim import pddl as FP
from factory_sim.env import build_sim
from factory_sim.scenarios import make_instance

FAMILIES = ("base", "bottleneck", "tight", "noisy")   # people-free (default lab: arm robots); not "exact": its unlimited buffers become 1000 slot objects per station


def export(inst, out):
    sim = build_sim(inst, seed=0, time_source="nominal")
    d = os.path.join(out, inst["split"], inst["name"])
    os.makedirs(d, exist_ok=True)
    dom, prob = FP.domain(sim.F), FP.export(sim, name=inst["name"])
    for f, text in (("domain.pddl", dom), ("problem.pddl", prob)):
        with open(os.path.join(d, f), "w") as fh:
            fh.write(text)
    with open(os.path.join(d, "domain.yaml"), "w") as fh:
        yaml.safe_dump(read_domain(dom), fh, sort_keys=False, width=200)
    read_problem(prob, "factorysim", inst["name"].lower()).save(os.path.join(d, "hl_problem.json"))
    with open(os.path.join(d, "instance.json"), "w") as fh:
        json.dump(inst, fh)
    gt = inst.get("ground_truth", {})
    return {"instance_id": inst["name"], "family": inst["family"], "size": inst["size"], "best_known": gt.get("best_known"), "lower_bound": gt.get("lower_bound")}


def _menv_dispatcher(dom, prob, horizon):
    """The robot dispatcher on the PDDL in menv at nominal durations -> result dict."""
    from algorithm import v10
    from algorithm.menv import MAEnv
    from factory_test.factory_dispatcher import Dispatcher
    env = MAEnv(v10.Task(v10.Domain(dom), v10.Problem(prob)), "robot", seed=0, horizon=horizon)
    D = Dispatcher(env)
    return env.rollout(lambda e, S, a, lg: D(e, S, a, lg), seed=0, max_rounds=10 ** 6)


def hard_instance(job):
    """(family, size, split, index) -> (row, inst) or (None, reason)."""
    from factory_sim.scenarios import set_dues
    fam, size, split, i = job
    inst = make_instance(fam, size, split, i)
    gt = inst["ground_truth"]
    sim = build_sim(inst, seed=0, time_source="nominal")
    dom = FP.domain(sim.F)
    core = gt.get("best_known")
    horizon = 6 * (core or 3 * gt["lower_bound"])
    r = _menv_dispatcher(dom, FP.export(sim, name=inst["name"]), horizon)
    ms = r["makespan"] if r["success"] else None
    if core is None and ms is None:
        return None, "{}: no reference finishes ({})".format(inst["name"], r["reason"])
    set_dues(inst, max(x for x in (core, ms) if x is not None))
    gt["best_known"] = min(x for x in (core, ms) if x is not None)
    gt["dispatcher_menv_makespan"] = ms
    sim = build_sim(inst, seed=0, time_source="nominal")
    prob = FP.export(sim, hard_deadlines=True, name=inst["name"])
    r2 = _menv_dispatcher(dom, prob, 4 * gt["best_known"])
    gt["dispatcher_on_time_nominal"] = bool(r2["success"])
    d = os.path.join(inst["split"], inst["name"])
    return {"dir": d, "domain": dom, "problem": prob, "inst": inst, "on_time": r2["success"], "ms": ms}, None


def write_hard(res, out):
    d = os.path.join(out, res["dir"])
    os.makedirs(d, exist_ok=True)
    for f, text in (("domain.pddl", res["domain"]), ("problem.pddl", res["problem"])):
        with open(os.path.join(d, f), "w") as fh:
            fh.write(text)
    inst = res["inst"]
    with open(os.path.join(d, "domain.yaml"), "w") as fh:
        yaml.safe_dump(read_domain(res["domain"]), fh, sort_keys=False, width=200)
    read_problem(res["problem"], "factorysim", inst["name"].lower()).save(os.path.join(d, "hl_problem.json"))
    with open(os.path.join(d, "instance.json"), "w") as fh:
        json.dump(inst, fh)
    gt = inst["ground_truth"]
    return {"instance_id": inst["name"], "family": inst["family"], "size": inst["size"], "best_known": gt.get("best_known"),
            "lower_bound": gt.get("lower_bound"), "due_reference": gt.get("due_reference"), "dispatcher_menv_makespan": gt.get("dispatcher_menv_makespan"),
            "dispatcher_on_time_nominal": gt.get("dispatcher_on_time_nominal"), "robots": inst["config"]["robots"]["count"], "orders": len(inst["orders"])}


def main_hard(a):
    from multiprocessing import Pool
    from factory_sim.scenarios import SUITE_HARD
    families = a.families if a.families_given else SUITE_HARD["families"]
    for split in a.splits:
        sizes = a.sizes if a.sizes_given else SUITE_HARD["sizes"][split]
        count = a.count if a.count_given else SUITE_HARD["counts"][split]
        rows, skipped = [], []
        need = {(f, s): count for f in families for s in sizes}
        nxt = {k: 0 for k in need}
        with Pool(a.procs) as pool:
            while any(need.values()):
                jobs = [(f, s, split, nxt[(f, s)] + k) for (f, s), n in need.items() for k in range(n)]
                if not jobs or min(nxt[k] for k, n in need.items() if n) > 10 * count:
                    break
                for (f, s), n in need.items():
                    nxt[(f, s)] += n
                for job, (res, why) in zip(jobs, pool.imap(hard_instance, jobs)):
                    if res is None:
                        skipped.append(why)
                        continue
                    if need[(job[0], job[1])] > 0:
                        need[(job[0], job[1])] -= 1
                        rows.append(write_hard(res, a.out))
                        print("{:<5} {:<36} best {:>9} on time (nominal) {}".format(split, rows[-1]["instance_id"], round(rows[-1]["best_known"]), res["on_time"]), flush=True)
        os.makedirs(os.path.join(a.out, split), exist_ok=True)
        rows.sort(key=lambda r: r["instance_id"])
        with open(os.path.join(a.out, split, "dataset.json"), "w") as f:
            json.dump({"split": split, "suite": SUITE_HARD["name"], "instances": rows, "skipped": skipped}, f, indent=1)
        print("{:<5} {} instances ({} skipped) -> {}".format(split, len(rows), len(skipped), os.path.join(a.out, split)), flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", choices=["easy", "hard"], default="easy")
    ap.add_argument("--families", nargs="*", default=None)
    ap.add_argument("--sizes", nargs="*", default=None)
    ap.add_argument("--splits", nargs="*", default=["train", "val", "test"])
    ap.add_argument("--count", type=int, default=None)
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--out", default=DATA)
    a = ap.parse_args(argv)
    a.families_given, a.sizes_given, a.count_given = a.families is not None, a.sizes is not None, a.count is not None
    if a.suite == "hard":
        return main_hard(a)
    a.families = a.families if a.families_given else list(FAMILIES)
    a.sizes = a.sizes if a.sizes_given else ["S", "M"]
    a.count = a.count if a.count_given else 5
    for split in a.splits:
        rows = []
        for fam in a.families:
            for size in (a.sizes if fam != "exact" else ["S"]):
                for i in range(a.count):
                    rows.append(export(make_instance(fam, size, split, i), a.out))
        os.makedirs(os.path.join(a.out, split), exist_ok=True)
        with open(os.path.join(a.out, split, "dataset.json"), "w") as f:
            json.dump({"split": split, "instances": rows}, f, indent=1)
        print("{:<5} {} instances -> {}".format(split, len(rows), os.path.join(a.out, split)), flush=True)


if __name__ == "__main__":
    main()
