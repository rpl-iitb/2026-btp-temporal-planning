"""V15's factory domain module: V13 / V14's (factory_v13.py) with a training curriculum that includes plants.

    python -m algorithm.v15 --domain factory_test.factory_v15 train --name V15_hard --init V14_hard

SPLITS: train15 = the hard suite's M and L training instances plus XL plant instances (2 production lines, 6
robots, 24 orders: make_curriculum below); val15 = the L validation instances plus XL ones. Everything else
(dispatcher, features, noise, tasks) is factory_v13's.
"""
import json
import os

from factory_test.factory_v13 import *  # noqa: F401,F403  (the domain protocol)
from factory_test.factory_v13 import DATA

SPLITS = ("train15", "val15")
XL_DATA = os.path.join(os.path.dirname(DATA), "data_hard_xl")


def make_curriculum(xl=XL_DATA):
    """Link the curriculum splits into DATA (instances stay where they are; XL ones come from make_data --out xl)."""
    for name, parts in (("train15", [(DATA, "train", ("M", "L")), (xl, "train", ("XL",))]),
                        ("val15", [(DATA, "val", ("L",)), (xl, "val", ("XL",))])):
        out = os.path.join(DATA, name)
        os.makedirs(out, exist_ok=True)
        rows = []
        for root, split, sizes in parts:
            for r in json.load(open(os.path.join(root, split, "dataset.json")))["instances"]:
                if r["size"] in sizes:
                    link = os.path.join(out, r["instance_id"])
                    if not os.path.exists(link):
                        os.symlink(os.path.join(root, split, r["instance_id"]), link)
                    rows.append(r)
        json.dump({"split": name, "instances": rows}, open(os.path.join(out, "dataset.json"), "w"), indent=1)
        print(name, len(rows), "instances")


if __name__ == "__main__":
    make_curriculum()
