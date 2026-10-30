"""Scenario + simulator -> HLProblem (objects, facts, travel times, goals, deadlines); pinned simulator access."""

"""This file is UN-VERIFIED"""

import datetime
import glob
import hashlib
import json
import os
import random

import yaml

import overcooked_ai_py
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, Recipe
from overcooked_ai_py.utils import read_layout_dict
from pddl_generator import Obj, Goal, TimedLiteral, HLProblem
from .layout import Layout, PDDL_TYPE, PROVIDES, DIR_NAME, pose_name, tile_name

BENCHMARK = "overcooked_ai"
DOMAIN_NAME = "overcooked-v1"
ROOT = os.path.dirname(os.path.abspath(__file__))          # overcooked_ai/
SCENARIO_SCHEMA = os.path.join(ROOT, "scenario.schema.json")
SIM_REPO_DIR = os.path.join(ROOT, "repo")
SIM_VERSION = {"package": "overcooked_ai", "version": "1.1.0",
               "commit": "a60e31a6280718b8cdde08d394c8a6697b1d4829",
               "repository": "https://github.com/HumanCompatibleAI/overcooked_ai"}


# ---- simulator ---------------------------------------------------------------------------
def check_pinned():
    """Refuse to run against a different checkout than the pinned one in repo/."""
    here = os.path.realpath(os.path.dirname(overcooked_ai_py.__file__))
    want = os.path.realpath(os.path.join(SIM_REPO_DIR, "src", "overcooked_ai_py"))
    if here != want:
        raise RuntimeError("overcooked_ai_py imported from {}, expected the pinned copy {}".format(here, want))
    return SIM_VERSION


def layout_params(name):
    """(grid rows, other params) of a layout file shipped with the simulator."""
    d = read_layout_dict(name)
    rows = [r.strip() for r in d.pop("grid").split("\n")]
    return rows, d


def make_mdp(rows, params):
    """OvercookedGridworld from explicit rows. Note: Recipe configuration is
    class-global in 1.1.0, so read recipe times right after building the MDP."""
    return OvercookedGridworld.from_grid(list(rows), base_layout_params=dict(params))


def recipe_info(ingredients):
    """Time / value under the *current* global Recipe configuration."""
    r = Recipe(tuple(ingredients))
    return {"time": int(r.time), "value": int(r.value)}


# ---- scenario --------------------------------------------------------------------------
def load_scenario(path):
    import jsonschema
    with open(path) as f:
        sc = yaml.safe_load(f)
    with open(SCENARIO_SCHEMA) as f:
        jsonschema.validate(sc, json.load(f))
    sc["_path"] = os.path.abspath(path)
    with open(path, "rb") as f:
        sc["_sha256"] = hashlib.sha256(f.read()).hexdigest()
    return sc


def recipe_name(onions, tomatoes):
    return "soup_o{}_t{}".format(onions, tomatoes)


def _resolve_grid(sc):
    """Final grid rows (with player digits) and simulator params."""
    if "name" in sc["layout"]:
        rows, params = layout_params(sc["layout"]["name"])
        params["layout_name"] = sc["layout"]["name"]
    else:
        rows, params = list(sc["layout"]["grid"]), {}
    rows = [list(r) for r in rows]
    # pots: drop the ones not kept (they become plain counters)
    pots = [(x, y) for y, r in enumerate(rows) for x, ch in enumerate(r) if ch == "P"]
    spec = sc.get("pots", "all")
    if spec == "all":
        keep = set(pots)
    elif isinstance(spec, int):
        if spec > len(pots):
            raise ValueError("scenario asks for {} pots, layout has {}".format(spec, len(pots)))
        keep = set(pots[:spec])
    else:
        keep = {tuple(c) for c in spec}
        if not keep <= set(pots):
            raise ValueError("pots {} are not pots in the layout".format(sorted(keep - set(pots))))
    for x, y in pots:
        if (x, y) not in keep:
            rows[y][x] = "X"
    # cooks: keep the first n layout players, or add more on free floor tiles
    starts = {int(ch): (x, y) for y, r in enumerate(rows) for x, ch in enumerate(r) if ch.isdigit()}
    n = sc.get("cooks", len(starts))
    for k, (x, y) in starts.items():
        if k > n:
            rows[y][x] = " "
    if n > len(starts):
        rng = random.Random(sc.get("seed", 0))
        lay = Layout(["".join(r) for r in rows])
        comp = lay.regions()
        home = comp.get(starts.get(1)) if starts else None
        free = sorted(t for t in lay.floor_tiles() if rows[t[1]][t[0]] == " " and (home is None or comp[t] == home))
        rng.shuffle(free)
        for k in range(len(starts) + 1, n + 1):
            if not free:
                raise ValueError("no free floor tile for cook {}".format(k))
            x, y = free.pop()
            rows[y][x] = str(k)
    # recipes: the simulator rewards exactly the ordered recipes
    params.update(sc.get("sim", {}) or {})
    recipes = []
    for o in sc["orders"]:
        ing = ["onion"] * o["recipe"].get("onion", 0) + ["tomato"] * o["recipe"].get("tomato", 0)
        if {"ingredients": ing} not in recipes:
            recipes.append({"ingredients": ing})
    params["start_all_orders"] = recipes
    params.setdefault("start_bonus_orders", [])
    params.pop("start_state", None)
    return ["".join(r) for r in rows], params


def _code_sha256():
    h = hashlib.sha256()
    files = sorted(glob.glob(os.path.join(ROOT, "*.py")) + glob.glob(os.path.join(ROOT, "*.json")) +
                   glob.glob(os.path.join(ROOT, "*.yaml")))
    for f in files:
        with open(f, "rb") as fh:
            h.update(fh.read())
    return h.hexdigest()


def build_problem(sc):
    check_pinned()
    rows, params = _resolve_grid(sc)
    mdp = make_mdp(rows, params)
    state = mdp.get_standard_start_state()
    orders = []
    for o in sc["orders"]:
        for _ in range(o.get("count", 1)):
            orders.append((o["recipe"].get("onion", 0), o["recipe"].get("tomato", 0), o.get("deadline")))
    prob = encode_state(sc["scenario_id"], rows, params, mdp, state, orders,
                        counter_policy=sc.get("counters", "shared"))
    prob.provenance = {
        "scenario_file": os.path.relpath(sc["_path"], os.path.dirname(ROOT)) if "_path" in sc else None,
        "scenario_sha256": sc.get("_sha256"),
        "scenario": {k: v for k, v in sc.items() if not k.startswith("_")},
        "simulator": SIM_VERSION,
        "adapter_code_sha256": _code_sha256(),
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    }
    return prob


# ---- state encoding ----------------------------------------------------------------------
def encode_state(instance_id, rows, params, mdp, state, orders, counter_policy="shared", t_now=0):
    """Encode an OvercookedState as an HLProblem.

    orders: [(onions, tomatoes, deadline or None)] still to be delivered.
    t_now : simulator time of `state`; deadlines are made relative to it.
    """
    lay = Layout(rows)
    max_ing = Recipe.MAX_NUM_INGREDIENTS
    counts = ["n{}".format(i) for i in range(max_ing + 1)]

    # objects held on counters force those counters into the model
    obj_cells = [pos for pos, o in state.objects.items() if lay.char(pos) == "X"]
    stations = lay.stations(counter_policy, extra_counters=obj_cells)
    st_by_cell = {s.cell: s for s in stations}

    objects, facts, fluents, tls = [], set(), {}, []

    # recipes: every ordered recipe, plus any soup already in the world
    recipe_keys = {(on, to) for on, to, _ in orders}
    for o in state.objects.values():
        if o.name == "soup" and o.ingredients:
            recipe_keys.add((o.ingredients.count("onion"), o.ingredients.count("tomato")))
    for p in state.players:
        if p.held_object is not None and p.held_object.name == "soup":
            ing = p.held_object.ingredients
            recipe_keys.add((ing.count("onion"), ing.count("tomato")))
    recipes = {}
    for on, to in sorted(recipe_keys):
        if not 1 <= on + to <= max_ing:
            raise ValueError("recipe with {} onions + {} tomatoes is not cookable (max {})".format(on, to, max_ing))
        info = recipe_info(["onion"] * on + ["tomato"] * to)
        if info["time"] < 2:
            raise ValueError("cook time {} < 2 is not representable (cook-steps = time - 1)".format(info["time"]))
        name = recipe_name(on, to)
        recipes[name] = {"onion": on, "tomato": to, **info}
        objects.append(Obj(name, "recipe", dict(recipes[name])))
        facts.add(("recipe-of", name, counts[on], counts[to]))
        fluents[("cook-steps", name)] = info["time"] - 1

    for i, c in enumerate(counts):
        objects.append(Obj(c, "count", {"value": i}))
    facts.add(("is-zero", counts[0]))
    for i in range(max_ing):
        facts.add(("next", counts[i], counts[i + 1]))
    for a in range(max_ing + 1):
        for b in range(max_ing + 1 - a):
            facts.add(("fits", counts[a], counts[b]))

    # stations and their interaction poses
    poses = {}            # pose tuple -> name
    for s in stations:
        objects.append(Obj(s.name, PDDL_TYPE[s.kind], {"kind": s.kind, "cell": list(s.cell)}))
        if s.kind in PROVIDES:
            facts.add(("provides", s.name, PROVIDES[s.kind]))
        for p in lay.access_poses(s.cell):
            poses.setdefault(p, pose_name(p))
            facts.add(("faces", poses[p], s.name))
    station_poses = set(poses)
    # cooks: one per simulator player, at a (possibly station-less) start pose
    cooks = []
    for i, pl in enumerate(state.players):
        c = "c{}".format(i + 1)
        cooks.append(c)
        p = (pl.position[0], pl.position[1], tuple(pl.orientation))
        poses.setdefault(p, pose_name(p))
        objects.append(Obj(c, "cook", {"player_index": i, "start_pose": poses[p]}))
        facts |= {("at", c, poses[p]), ("idle", c)}
        held = pl.held_object
        if held is None:
            facts.add(("hand-empty", c))
        elif held.name == "soup":
            facts.add(("holding-soup", c, recipe_name(held.ingredients.count("onion"),
                                                     held.ingredients.count("tomato"))))
        else:
            facts.add(("holding", c, held.name))

    tiles = sorted({(x, y) for x, y, _ in poses})
    occupied = {tuple(pl.position) for pl in state.players}
    for t in tiles:
        objects.append(Obj(tile_name(t), "tile", {"cell": list(t)}))
        if t not in occupied:
            facts.add(("free", tile_name(t)))
    for p, n in sorted(poses.items(), key=lambda kv: kv[1]):
        objects.append(Obj(n, "pose", {"tile": tile_name(p[:2]), "cell": [p[0], p[1]],
                                       "facing": DIR_NAME[p[2]]}))
        facts.add(("pose-tile", n, tile_name(p[:2])))

    # travel: from every pose to every *station* pose (start poses are origins only)
    for p, n in poses.items():
        dist = lay.bfs_poses(p)
        for q in station_poses:
            if q == p or q not in dist:
                continue
            m = poses[q]
            facts.add(("connected", n, m) if p[:2] != q[:2] else ("same-tile", n, m))
            fluents[("travel", n, m)] = dist[q]

    # pots, counters
    for s in stations:
        has = state.has_object(s.cell)
        o = state.get_object(s.cell) if has else None
        if s.kind == "pot":
            if o is None:
                facts |= {("pot-idle", s.name), ("pot-has", s.name, counts[0], counts[0])}
                continue
            on, to = o.ingredients.count("onion"), o.ingredients.count("tomato")
            if o.is_idle:
                facts |= {("pot-idle", s.name), ("pot-has", s.name, counts[on], counts[to])}
            elif o.is_ready:
                facts.add(("soup-ready", s.name, recipe_name(on, to)))
            else:   # cooking: the soup becomes ready at a known time
                tls.append(TimedLiteral(float(o.cook_time_remaining), ("soup-ready", s.name, recipe_name(on, to))))
        elif s.kind == "counter":
            if o is None:
                facts.add(("counter-empty", s.name))
            elif o.name == "soup":
                facts.add(("counter-soup", s.name, recipe_name(o.ingredients.count("onion"),
                                                               o.ingredients.count("tomato"))))
            else:
                facts.add(("counter-holds", s.name, o.name))

    # orders -> goals with deadlines (relative to t_now)
    goals = []
    for k, (on, to, dl) in enumerate(orders):
        o = "o{}".format(k + 1)
        rel = None if dl is None else dl - t_now
        if rel is not None and rel <= 0:
            raise ValueError("order {} deadline {} already passed at t={}".format(o, dl, t_now))
        objects.append(Obj(o, "order", {"recipe": recipe_name(on, to), "deadline": dl}))
        facts |= {("order-recipe", o, recipe_name(on, to)), ("pending", o), ("open", o)}
        goals.append(Goal(("served", o), rel))

    return HLProblem(
        instance_id=instance_id, benchmark=BENCHMARK, domain_name=DOMAIN_NAME,
        objects=objects, facts=sorted(facts), fluents=fluents, goals=goals, timed_literals=tls,
        annotations={"grid": list(rows), "sim_params": params, "t_now": t_now,
                     "max_ingredients": max_ing, "counter_policy": counter_policy,
                     "cooks": cooks, "recipes": recipes})
