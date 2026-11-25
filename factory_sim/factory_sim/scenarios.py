"""Benchmark instances: scenario dimensions, named families, sizes, splits and ground truth.

An instance is a self-contained dict: {name, family, size, split, seed, dims, config, orders, sim, ground_truth}.
"""
import copy
import json
import os

import numpy as np

from .config import Factory, read
from .core import Sim, default_orders

DIMENSIONS = {
    "routing": ("fixed", "alt_machines", "alt_routes"),
    "contention": ("low", "medium", "high"),
    "uncertainty": ("deterministic", "stochastic", "high"),
    "reliability": ("none", "breakdowns"),
    "buffers": ("unlimited", "finite", "tight"),
    "arrivals": ("batch", "online"),
    "quality": ("none", "rework", "scrap"),
    "congestion": ("none", "docks", "high"),
    "deadlines": ("loose", "tight", "infeasible", "rush"),
    "energy": ("none", "battery"),
    "robot_reliability": ("none", "faults"),
    "maintenance": ("none", "preventive"),
    "flexibility": ("fixed", "reconfigurable"),
    "workforce": ("full", "breaks"),
    "materials": ("unlimited", "shortage"),
    "conveyors": ("smooth", "jams"),
    "fleet": ("arm", "tray", "mixed"),   # robot types (factory_sim/robots.py): arm = all tb3_manipulator
    "traffic": ("none", "lanes"),        # lanes: corridor junctions hold one robot, robots wait and back off (config.Factory._traffic)
    "due": ("flow", "load"),             # due dates from each job's own flow time, or from the whole factory's workload
}
BASE = {"routing": "fixed", "contention": "medium", "uncertainty": "stochastic", "reliability": "none", "buffers": "finite", "arrivals": "batch", "quality": "none", "congestion": "docks", "deadlines": "loose",
        "energy": "none", "robot_reliability": "none", "maintenance": "none", "flexibility": "fixed", "workforce": "full", "materials": "unlimited", "conveyors": "smooth",
        "traffic": "none", "due": "flow"}
FAMILIES = {
    "base": {},
    "independent": {"contention": "low", "buffers": "unlimited"},
    "bottleneck": {"contention": "high"},
    "flexible": {"routing": "alt_routes"},
    "alt_machines": {"routing": "alt_machines"},
    "congested": {"congestion": "high"},
    "breakdowns": {"reliability": "breakdowns"},
    "online": {"arrivals": "online"},
    "blocking": {"buffers": "tight"},
    "quality": {"quality": "rework"},
    "scrap": {"quality": "scrap"},
    "noisy": {"uncertainty": "high"},
    "tight": {"deadlines": "tight"},
    "exact": {"uncertainty": "deterministic", "buffers": "unlimited", "congestion": "none"},
    "infeasible": {"uncertainty": "deterministic", "deadlines": "infeasible"},
    "full": {"routing": "alt_routes", "contention": "high", "uncertainty": "high", "reliability": "breakdowns", "arrivals": "online", "quality": "rework", "congestion": "high", "deadlines": "tight"},
    "charging": {"energy": "battery"},
    "robot_faults": {"robot_reliability": "faults"},
    "maintenance": {"maintenance": "preventive"},
    "reconfig": {"flexibility": "reconfigurable", "reliability": "breakdowns", "maintenance": "preventive"},
    "breaks": {"workforce": "breaks"},
    "shortage": {"materials": "shortage"},
    "jams": {"conveyors": "jams"},
    "rush": {"deadlines": "rush"},
    "realistic": {"reliability": "breakdowns", "arrivals": "online", "quality": "rework", "deadlines": "rush", "energy": "battery", "robot_reliability": "faults", "maintenance": "preventive",
                  "flexibility": "reconfigurable", "workforce": "breaks", "materials": "shortage", "conveyors": "jams"},
}
FAMILIES.update({
    "tray_fleet": {"fleet": "tray"},     # TurtleBot3 tray robots; human handlers move the trays (handover actions)
    "mixed_fleet": {"fleet": "mixed"},   # arm robots + tray robots (+ a Burger on L): heterogeneous speeds and actions
    # every disturbance that involves no people (realistic without operator breaks)
    "disturbed": {"reliability": "breakdowns", "arrivals": "online", "quality": "rework", "deadlines": "rush", "energy": "battery", "robot_reliability": "faults",
                  "maintenance": "preventive", "flexibility": "reconfigurable", "materials": "shortage", "conveyors": "jams"},
})
# 3.0 "hard": lane traffic with junctions and intersections, due dates from the factory's workload, and only the
# features that the PDDL model (and so the closed-loop policies' environment, algorithm/menv.py) also has
HARD = {"traffic": "lanes", "due": "load"}
FAMILIES.update({
    "hard_base": dict(HARD),
    "hard_bottleneck": dict(HARD, contention="high"),
    "hard_blocking": dict(HARD, buffers="tight"),
    "hard_flexible": dict(HARD, routing="alt_routes"),
    "hard_online": dict(HARD, arrivals="online", deadlines="tight"),
    "hard_rush": dict(HARD, deadlines="rush"),
    "hard_full": dict(HARD, contention="high", routing="alt_routes", arrivals="online", deadlines="tight", uncertainty="high"),   # not with tight buffers: those two block for good (robots end up holding every tray)
})
HARD_FAMILIES = [f for f in FAMILIES if f.startswith("hard_")]
FAMILIES_1_0 = [f for f in FAMILIES if f not in ("tray_fleet", "mixed_fleet", "disturbed") and f not in HARD_FAMILIES]
# families that need people: operators with breaks, or human handlers for tray robots. The default lab has no people;
# for these families the workforce below is added to the config (configs that already have operators keep theirs).
PEOPLE_FAMILIES = ("breaks", "realistic", "tray_fleet", "mixed_fleet")
WORKFORCE = {"operators": {"kitter": 1, "assembler": 2, "technician": 1}, "technician": "technician",
             "operator_breaks": {"assembler": {"every": 7200, "duration": 900, "n": 1}, "kitter": {"every": 7200, "duration": 900, "n": 1, "first": 3600}},
             "station_operators": {"kitting": "kitter", "asm1": "assembler", "asm2": "assembler"}}
# instance seeds use a family's index; 1.0 families keep their 1.0 index, later families are numbered after them
FAMILY_INDEX = dict({f: i for i, f in enumerate(sorted(FAMILIES_1_0))}, **{f: len(FAMILIES_1_0) + i for i, f in enumerate(f for f in FAMILIES if f not in FAMILIES_1_0)})
V1_FAMILIES = ["base", "independent", "bottleneck", "flexible", "alt_machines", "congested", "breakdowns", "online", "blocking", "quality", "scrap", "noisy", "tight", "exact", "infeasible", "full"]
SIZES = {"S": {"robots": 2, "orders": 4}, "M": {"robots": 3, "orders": 8}, "L": {"robots": 4, "orders": 14},
         # larger plants (layouts.plant: production lines joined by cross aisles); used with lane traffic
         "XL": {"robots": 6, "orders": 24, "layout": "plant2"}, "XXL": {"robots": 8, "orders": 36, "layout": "plant3"},
         "XXXL": {"robots": 12, "orders": 56, "layout": "plant4"}}
SPLIT_SEED = {"train": 0, "val": 100000, "test": 200000}
UNLIMITED = 1000


def apply(cfg, dims, size, rng):
    """Turn a base factory config into one instance config for the given dimension settings."""
    cfg = copy.deepcopy(cfg)
    if (dims.get("workforce") == "breaks" or dims.get("fleet", "arm") != "arm") and not cfg.get("operators"):
        cfg["operators"], cfg["technician"] = dict(WORKFORCE["operators"]), WORKFORCE["technician"]
        cfg["operator_breaks"] = copy.deepcopy(WORKFORCE["operator_breaks"])
        for st, pool in WORKFORCE["station_operators"].items():
            if st in cfg["stations"]:
                cfg["stations"][st]["operator"] = pool
    cfg["robots"]["count"] = size["robots"]
    n = size["robots"]
    if dims.get("fleet", "arm") == "tray":
        cfg["robots"]["type"] = "tb3_tray"
        for k in ("speed", "travel_cv", "pick", "place", "load", "unload", "grasp_success", "place_success"):
            cfg["robots"].pop(k, None)
    elif dims.get("fleet", "arm") == "mixed":
        site = {k: v for k, v in cfg["robots"].items() if k in ("start", "parking", "battery", "fault", "handler_pool", "handlers")}
        fleet = [{"type": "tb3_manipulator", "count": (n + 1) // 2}, {"type": "tb3_tray", "count": n // 2 - (n >= 4)}] + ([{"type": "tb3_burger_tray", "count": 1}] if n >= 4 else [])
        cfg["robots"] = dict(site, fleet=[e for e in fleet if e["count"] > 0])
    o = cfg.setdefault("orders", {})
    o["count"] = size["orders"]
    prods = list(cfg["products"])
    mix = {"low": {p: 1 for p in prods}, "medium": {p: float(rng.uniform(0.5, 1.5)) for p in prods}, "high": {p: 1 for p in prods[:2]} if len(prods) < 4 else {prods[0]: 1, prods[2]: 1}}[dims["contention"]]
    o["mix"], o["random_mix"] = mix, dims["contention"] != "low"
    def sub(x, table):          # a route step (name or list of names, possibly with a line suffix _L<i>) through `table`
        def one(n):
            b, _, line = n.partition("_L")
            return table.get(b, b) + ("_L" + line if line else "")
        return one(x) if isinstance(x, str) else sorted({one(n) for n in x})
    if dims["routing"] == "alt_machines":
        alt = {"MC6": "MC7", "MC7": "MC6", "MC8": "MC9", "MC9": "MC8"}
        def widen(x):           # MC6 -> [MC6, MC7] (on every line the step already lists)
            xs = [x] if isinstance(x, str) else list(x)
            return sorted(set(xs) | set(sub(xs, alt))) if xs[0].partition("_L")[0] in alt else x
        for p in cfg["products"].values():
            p["routes"] = [[widen(x) for x in r] for r in p["routes"]]
    if dims["routing"] == "alt_routes":
        swap = {"MC1": "MC3", "MC2": "MC4", "MC3": "MC1", "MC4": "MC2"}
        for p in cfg["products"].values():
            r = p["routes"][0]
            p["routes"] = [r, [sub(x, swap) for x in r]]
    f = cfg.setdefault("features", {})
    f["stochastic"] = dims["uncertainty"] != "deterministic"
    f["breakdowns"] = dims["reliability"] == "breakdowns"
    f["rejects"] = dims["quality"] != "none"
    for s in cfg["stations"].values():
        if f["breakdowns"] and s.get("breakdown"):
            s["breakdown"]["mtbf"] = 3600
        if dims["buffers"] == "unlimited" and s["kind"] not in ("sink",):
            s["slots_in"], s["slots_out"] = UNLIMITED, UNLIMITED if s.get("slots_out", 1) > 0 else 0
        if dims["buffers"] == "tight" and s["kind"] in ("machine", "assembly", "inspection") and s.get("slots_out", 0) > 0:
            s["slots_out"] = 0
        if dims["quality"] == "none":
            s.pop("reject", None)
        if dims["quality"] in ("rework", "scrap") and s["kind"] == "inspection":
            s["reject"] = 0.1
            if dims["quality"] == "scrap":
                s.pop("rework_to", None)
    if dims["buffers"] == "tight":
        cfg["stations"] = {k: v for k, v in cfg["stations"].items() if v["kind"] != "buffer"}
    cfg["congestion"] = {"none": {"dock_capacity": UNLIMITED, "slowdown": 0.0}, "docks": {"dock_capacity": 1, "slowdown": 0.1}, "high": {"dock_capacity": 1, "slowdown": 0.3}}[dims["congestion"]]
    if dims.get("traffic", "none") == "lanes":   # robots really block each other: no extra flat slowdown
        cfg["traffic"] = dict(cfg.get("traffic") or {}, mode="lanes")
        cfg["congestion"]["slowdown"] = 0.0
    elif cfg.get("traffic"):
        cfg["traffic"]["mode"] = "none"
    if dims["arrivals"] == "online":
        F = Factory(cfg)
        o["arrivals"] = {"online": float(np.mean([F.nominal_flow(p) for p in F.products])) / (2 * size["robots"])}
    o["due_factor"] = {"loose": 3.0, "tight": 1.5, "infeasible": 0.6, "rush": 3.0}[dims["deadlines"]]
    if dims["deadlines"] == "rush":
        o["rush"] = {"prob": 0.3, "due_factor": 1.3, "weight": 3.0}
    f.update({"battery": dims["energy"] == "battery", "robot_faults": dims["robot_reliability"] == "faults", "maintenance": dims["maintenance"] == "preventive",
              "reconfiguration": dims["flexibility"] == "reconfigurable", "breaks": dims["workforce"] == "breaks", "materials": dims["materials"] == "shortage", "jams": dims["conveyors"] == "jams"})
    if f["battery"] and cfg["robots"].get("battery"):
        cfg["robots"]["battery"]["initial"] = [0.35, 0.9]
    if f["maintenance"]:
        for st in cfg["stations"].values():
            if st.get("maintenance"):
                st["maintenance"]["every"] = 900
        if "MC5" in cfg["stations"]:
            cfg["stations"]["MC5"]["downtime"] = [[1800, 600]]
    if f["materials"] and cfg.get("materials"):
        cfg["materials"] = {"stock": {p: 1 for p in prods}, "replenish": {"every": 1200, "qty": 1}}
    return cfg


def make_instance(family, size="M", split="train", index=0, base="lab", cpsat_limit=None):
    dims = dict(BASE, **FAMILIES[family])
    seed = SPLIT_SEED[split] + index
    rng = np.random.default_rng([seed, FAMILY_INDEX[family], list(SIZES).index(size)])
    lay = SIZES[size].get("layout")
    if lay:
        from .layouts import plant
        base = plant(int(lay[len("plant"):]))
    cfg = apply(read(base), dims, SIZES[size], rng)
    F = Factory(cfg)
    orders = default_orders(F, rng)
    sim = {"time_source": "sampled" if cfg["features"]["stochastic"] else "nominal", "cv_scale": 2.0 if dims["uncertainty"] == "high" else 1.0, "horizon": 20 * max(o["due"] - o["arrival"] for o in orders) + max(o["arrival"] for o in orders)}
    inst = {"name": "{}_{}_{}_{:03d}".format(family, size, split, index), "family": family, "size": size, "split": split, "seed": seed, "dims": dims, "config": cfg, "orders": orders, "sim": sim}
    inst["ground_truth"] = ground_truth(inst, cpsat_limit)
    if dims.get("due", "flow") == "load":
        set_dues(inst, inst["ground_truth"].get("best_known") or 2.0 * inst["ground_truth"]["lower_bound"])
    return inst


def set_dues(inst, ref):
    """Due dates from the factory's workload (dims due = load), with ref the reference makespan. loose (and the
    non-rush orders of rush): one common due date, 1.2 ref. tight: staggered, the k-th of n orders (by arrival) due
    at (0.35 + 0.8 k / n) ref, so the order of the work matters. Never before arrival + 1.3 (tight) / 1.5 x the
    order's own flow time. rush orders: half the slack of the others."""
    F = Factory(inst["config"])
    dims = inst["dims"]
    orders = sorted(inst["orders"], key=lambda o: (o["arrival"], o["id"]))
    n = len(orders)
    for k, o in enumerate(orders):
        a, flow = float(o["arrival"]), F.nominal_flow(o["product"])
        if dims["deadlines"] == "tight":
            due = max(a + 1.3 * flow, ref * (0.35 + 0.8 * (k + 1) / n))
        elif dims["deadlines"] == "infeasible":
            due = a + 0.6 * flow
        else:
            due = max(a + 1.5 * flow, 1.2 * ref)
        if o.get("rush"):
            due = a + max(1.3 * flow, 0.5 * (due - a))
        o["due"] = round(due, 3)
    inst["ground_truth"]["due_reference"] = round(float(ref), 3)
    inst["ground_truth"]["infeasible_deadlines"] = any(o["due"] < o["arrival"] + min(F.nominal_flow(o["product"], r) for r in range(len(F.products[o["product"]]["routes"]))) for o in inst["orders"])
    inst["sim"]["horizon"] = 20 * max(o["due"] - o["arrival"] for o in inst["orders"]) + max(o["arrival"] for o in inst["orders"])
    return inst


def ground_truth(inst, cpsat_limit=None):
    from . import baselines, ground_truth as G
    F = Factory(inst["config"])
    lb, parts = G.lower_bound(F, inst["orders"])
    out = {"lower_bound": round(lb, 3), "lower_bound_parts": {k: round(v, 3) for k, v in parts.items()}}
    ref = []
    for p in ("fifo", "edd", "spt", "cr", "mwkr", "nearest"):
        m = Sim(F, orders=inst["orders"], seed=inst["seed"], time_source="nominal").run(baselines.make(p))
        if m["success"]:
            ref.append((m["makespan"], p))
    if ref:
        out["reference_makespan"], out["reference_policy"] = min(ref)
        out["best_known"] = out["reference_makespan"]
    out["infeasible_deadlines"] = any(o["due"] < o["arrival"] + min(F.nominal_flow(o["product"], r) for r in range(len(F.products[o["product"]]["routes"]))) for o in inst["orders"])
    if cpsat_limit and F.reconfigurable:
        out["cpsat"] = {"status": "skipped: the CP-SAT model has fixed machine roles, so its bound is not valid under reconfiguration"}
    elif cpsat_limit:
        try:
            sol = G.cpsat(F, inst["orders"], time_limit=cpsat_limit)
            out["cpsat"] = {k: sol[k] for k in ("status", "bound", "exact") if k in sol}
            if "makespan" in sol:
                out["cpsat"]["makespan"] = sol["makespan"]
                if sol.get("exact"):
                    out["best_known"] = min(out.get("best_known", float("inf")), sol["makespan"])
            out["lower_bound"] = round(max(lb, sol["bound"]), 3)
        except ImportError:
            out["cpsat"] = {"status": "ortools not installed"}
    return out


SUITE_V1 = {"name": "factorysim-v1", "base": "lab_v1", "families": [f for f in V1_FAMILIES if f not in ("exact",)], "sizes": {"train": ["S", "M"], "val": ["S", "M"], "test": ["M", "L"]}, "counts": {"train": 50, "val": 10, "test": 10}, "exact": {"sizes": ["S"], "count": 10}}


SUITE_V2 = dict(SUITE_V1, name="factorysim-v2", families=[f for f in FAMILIES_1_0 if f not in ("exact",)])
SUITE_1_0 = dict(SUITE_V2, name="factorysim-1.0", families=[f for f in FAMILIES_1_0 if f not in ("exact",)])
SUITE_1_1 = dict(SUITE_1_0, name="factorysim-1.1", families=SUITE_1_0["families"] + ["tray_fleet", "mixed_fleet"])
# 2.0: the people-free default lab (arm robots, stations without operators); families that need people are left out
SUITE_2_0 = dict(SUITE_1_0, name="factorysim-2.0", base="lab",
                 families=[f for f in FAMILIES_1_0 if f not in ("exact",) + PEOPLE_FAMILIES] + ["disturbed"])
# 3.0 hard: lane traffic, workload due dates; train on the lab (S-L), test up to a four-line plant (XXXL)
SUITE_HARD = {"name": "factorysim-3.0-hard", "base": "lab", "families": HARD_FAMILIES,
              "sizes": {"train": ["S", "M", "L"], "val": ["S", "M", "L"], "test": ["M", "L", "XL", "XXL", "XXXL"]},
              "counts": {"train": 6, "val": 2, "test": 3}}
SUITES = {"hard": SUITE_HARD, "2.0": SUITE_2_0, "1.1": SUITE_1_1, "1.0": SUITE_1_0, "v2": SUITE_V2, "v1": SUITE_V1}


def code_version():
    """Package version and, when run from a git checkout, the commit (instances record what generated them)."""
    from . import __version__
    import subprocess
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=os.path.dirname(os.path.abspath(__file__)), capture_output=True, text=True, timeout=5).stdout.strip() or None
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", "."], cwd=os.path.dirname(os.path.abspath(__file__)), capture_output=True, text=True, timeout=5).stdout.strip())
    except (OSError, subprocess.SubprocessError):
        commit, dirty = None, None
    return {"version": __version__, "commit": commit, "dirty": dirty}


def generate_suite(out_dir, suite=SUITE_2_0, splits=("train", "val", "test"), families=None, count=None, cpsat_limit=30.0, bound_limit=0.0, verbose=True):
    """Write one JSON file per instance to out_dir/<split>/<name>.json and an index.json. bound_limit > 0 adds CP-SAT
    relaxation bounds (that many seconds each) to the test split's M instances."""
    index, ver = [], code_version()
    for split in splits:
        os.makedirs(os.path.join(out_dir, split), exist_ok=True)
        jobs = [(f, s, count or suite["counts"][split]) for f in (families or suite["families"]) for s in suite["sizes"][split]]
        if (families is None or "exact" in families) and "exact" in suite:
            jobs.append(("exact", suite["exact"]["sizes"][0], count or suite["exact"]["count"]))
        for fam, size, n in jobs:
            for i in range(n):
                limit = cpsat_limit if fam == "exact" else (bound_limit if (bound_limit and split == "test" and size == "M") else None)
                inst = make_instance(fam, size, split, i, base=suite.get("base", "lab"), cpsat_limit=limit)
                inst["code_version"] = ver
                path = os.path.join(out_dir, split, inst["name"] + ".json")
                with open(path, "w") as f:
                    json.dump(inst, f)
                index.append({"name": inst["name"], "split": split, "family": fam, "size": size, "path": os.path.relpath(path, out_dir)})
            if verbose:
                print("{:>5} {:<13} {} x{}".format(split, fam, size, n), flush=True)
    with open(os.path.join(out_dir, "index.json"), "w") as f:
        json.dump({"suite": suite.get("name", "factorysim"), "code_version": ver, "generated": __import__("time").strftime("%Y-%m-%d"), "dimensions": DIMENSIONS, "families": FAMILIES, "sizes": SIZES, "instances": index}, f, indent=1)
    return index


def load(path):
    with open(path) as f:
        return json.load(f)


def iter_suite(root, split="test", families=None):
    idx = load(os.path.join(root, "index.json"))
    for e in idx["instances"]:
        if e["split"] == split and (families is None or e["family"] in families):
            yield load(os.path.join(root, e["path"]))
