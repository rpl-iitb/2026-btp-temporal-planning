"""Export the overcooked_kitchen benchmark suite (final/overcooked_kitchen/benchmarks/v1) as instance folders for
algorithm/ and baselines/:

    python -m overcooked_test.make_data [--splits v2_train v2_val v2_test v3_train v3_val v3_test]

data/<split>/<name>/  hl_problem.json  (the kitchen's PDDL export of the scenario at t = 0: what T-iFLAX reads)
                      problem.pddl     (the same problem as PDDL text: what V10 / V11 / POPF read)
                      certificate.plan (the suite's VAL-checked certificate, if any)
                      scenario.yaml    (the kitchen scenario, to replay plans in overcooked_kitchen)
data/domain.pddl, data/domain.yaml     (the kitchen's domain; the algorithm finds domain.yaml in a parent folder)
data/<split>/dataset.json              {"instances": [{"instance_id", "certificate_makespan", "tier", "cooks", "orders"}]}
"""
import argparse
import json
import os
import shutil

import overcooked_test  # noqa: F401  (paths)
from overcooked_test import DATA, KITCHEN
from overcooked_kitchen import pddl as KP
from overcooked_kitchen.core import Sim
from overcooked_kitchen.scenarios import load_scenario
from overcooked_kitchen.suite import read_index

SUITE = os.path.join(KITCHEN, "benchmarks", "v1")
SPLITS = ("v2_train", "v2_val", "v2_test", "v3_train", "v3_val", "v3_test")


def export(row, out):
    sim = Sim(load_scenario(os.path.join(SUITE, row["path"])), clock="sampled", seed=0)
    prob = KP.export(sim)
    d = os.path.join(out, row["split"], row["name"])
    os.makedirs(d, exist_ok=True)
    prob.save(os.path.join(d, "hl_problem.json"))
    with open(os.path.join(d, "problem.pddl"), "w") as f:
        f.write(KP.problem_text(prob))
    shutil.copy(os.path.join(SUITE, row["path"]), os.path.join(d, "scenario.yaml"))
    if row.get("cert_plan") and row.get("cert_valid"):
        shutil.copy(os.path.join(SUITE, row["cert_plan"]), os.path.join(d, "certificate.plan"))
    return {"instance_id": row["name"], "certificate_makespan": row.get("plan_makespan"), "delivery_makespan": row.get("cert_makespan"),
            "tier": row.get("tier"), "cooks": row.get("cooks"), "orders": row.get("orders")}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", nargs="*", default=list(SPLITS))
    ap.add_argument("--out", default=DATA)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "domain.pddl"), "w") as f:
        f.write(KP.domain_text())
    shutil.copy(KP.DOMAIN_FILE, os.path.join(a.out, "domain.yaml"))
    rows = read_index(SUITE)
    for split in a.splits:
        meta = [export(r, a.out) for r in rows if r["split"] == split]
        with open(os.path.join(a.out, split, "dataset.json"), "w") as f:
            json.dump({"split": split, "suite": os.path.relpath(SUITE, a.out), "instances": meta}, f, indent=1)
        print("{:<9} {} instances -> {}".format(split, len(meta), os.path.join(a.out, split)), flush=True)


if __name__ == "__main__":
    main()
