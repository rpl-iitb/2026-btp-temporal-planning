"""Scenario files (YAML) to worlds: loading, order expansion, bundled examples."""
"""This file is UN-VERIFIED"""
import copy
import glob
import hashlib
import os
from typing import List

import yaml

from overcooked_kitchen.layout import INGREDIENTS
from overcooked_kitchen.world import World, sim_params

EXAMPLES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "examples")


def load_scenario(path_or_dict):
    if isinstance(path_or_dict, dict):
        sc = copy.deepcopy(path_or_dict)
        path = sc.get("scenario_id", "<dict>")
    else:
        path = str(path_or_dict)
        with open(path) as f:
            sc = yaml.safe_load(f)
    for k in ("scenario_id", "grid", "orders"):
        if k not in sc:
            raise ValueError("{}: missing '{}'".format(path, k))
    for o in sc["orders"]:
        bad = set(o["recipe"]) - set(INGREDIENTS)
        if bad:
            raise ValueError("{}: unknown ingredients {}".format(path, sorted(bad)))
    if not isinstance(path_or_dict, dict):
        sc["_path"] = os.path.abspath(path)
        with open(path, "rb") as f:
            sc["_sha256"] = hashlib.sha256(f.read()).hexdigest()
    return sc


def comp_of(recipe):
    return tuple(int(recipe.get(i, 0)) for i in INGREDIENTS)


def recipe_name(comp):
    return "soup_o{}_t{}_c{}".format(*comp)


def scenario_orders(sc) -> List[dict]:
    out = []
    for o in sc["orders"]:
        comp = comp_of(o["recipe"])
        for _ in range(o.get("count", 1)):
            out.append({"id": "o{}".format(len(out) + 1), "comp": comp, "deadline": o.get("deadline"),
                        "dish": o.get("dish", recipe_name(comp))})
    return out


def make_world(sc) -> World:
    pl = sc.get("plates", {}) or {}
    orders = scenario_orders(sc)
    return World(sc["grid"], sim_params(sc.get("sim")), plates_clean=pl.get("clean", len(orders)),
                 plates_dirty=pl.get("dirty", 0), dirty_pots=sc.get("dirty_pots", 0), orders=orders)


def example(name) -> str:
    path = os.path.join(EXAMPLES_DIR, name + ".yaml")
    if not os.path.isfile(path):
        raise ValueError("unknown example '{}'; available: {}".format(name, examples()))
    return path


def examples() -> List[str]:
    return sorted(os.path.basename(p)[:-5] for p in glob.glob(os.path.join(EXAMPLES_DIR, "*.yaml")))
