"""Generate many certified-feasible Overcooked scenarios (+ PDDL) from a configs/dataset_*.yaml.
See overcooked_ai/README.md.

    python -m overcooked_ai.make_dataset CONFIG [--count N] [--tiers ...] [--probe S] [--no-pddl]
"""

"""This file is UN-VERIFIED"""

import argparse
import collections
import json
import math
import os
import random
import shutil
import sys
import tempfile
import time

import yaml

from . import builder
from .benchmark import OvercookedV1

STATION_CHAR = {"pot": "P", "onion": "O", "tomato": "T", "dish": "D", "serve": "S"}


def pick(rng, v):
    """3 (fixed) | [lo, hi] (uniform; ints inclusive) | [a, b, c, ...] (choice)."""
    if isinstance(v, list) and len(v) == 2 and all(isinstance(x, (int, float)) for x in v):
        if all(isinstance(x, int) for x in v):
            return rng.randint(v[0], v[1])
        return rng.uniform(v[0], v[1])
    if isinstance(v, list):
        return rng.choice(v)
    return v


# ---- kitchen grids -------------------------------------------------------------------------
def make_room(rng, W, H, walls, split):
    """Border of counters; `walls` vertical counter walls with one gap each
    (alternating top / bottom -> serpentine detours); `split` = one extra wall
    with no gap (two rooms, hand-offs over shared counters)."""
    rows = [["X"] * W] + [["X"] + [" "] * (W - 2) + ["X"] for _ in range(H - 2)] + [["X"] * W]
    n = walls + (1 if split else 0)
    xs = [round((k + 1) * (W - 1) / (n + 1)) for k in range(n)]
    for k, x in enumerate(xs):
        if x <= 1 or x >= W - 2:
            continue
        no_gap = split and k == n // 2
        gap = 1 if k % 2 == 0 else H - 2
        for y in range(1, H - 1):
            if no_gap or abs(y - gap) > 0:
                rows[y][x] = "X"
    return rows


def floor_region_left(rows):
    """Floor tiles of the room containing the leftmost floor tile (cooks' home)."""
    from .layout import Layout
    lay = Layout(["".join(r) for r in rows])
    comp = lay.regions()
    tiles = sorted(comp, key=lambda t: (t[0], t[1]))
    home = comp[tiles[0]]
    return lay, comp, [t for t in tiles if comp[t] == home]


def border_slots(lay, rows):
    """Counter cells (not corners) with a floor neighbour: candidate station cells."""
    H, W = len(rows), len(rows[0])
    out = []
    for y in range(H):
        for x in range(W):
            if rows[y][x] == "X" and lay.access_poses((x, y)) and not (x in (0, W - 1) and y in (0, H - 1)):
                out.append((x, y))
    return out


def build_kitchen(rng, p, n_cooks, need_tomato):
    W, H = pick(rng, p["width"]), pick(rng, p["height"])
    split = rng.random() < p.get("split_prob", 0.0)
    rows = make_room(rng, W, H, pick(rng, p.get("walls", 0)), split)
    lay, comp, home = floor_region_left(rows)
    slots = border_slots(lay, rows)
    rng.shuffle(slots)
    used = set()
    third = W / 3.0

    def place(kind, near):
        for c in slots:
            if c in used:
                continue
            acc_regions = {comp[(x, y)] for x, y, _ in lay.access_poses(c)}
            if near and (c[0] >= third + 1 or comp[home[0]] not in acc_regions):
                continue
            if not near and c[0] < third and rng.random() < 0.8:
                continue                      # distractors prefer the middle / far side
            used.add(c)
            rows[c[1]][c[0]] = STATION_CHAR[kind]
            return c
        return None

    core = []
    for kind, n in (("pot", pick(rng, p.get("core_pots", 1))), ("onion", 1), ("tomato", 1 if need_tomato else 0),
                    ("dish", 1), ("serve", 1)):
        for _ in range(n):
            c = place(kind, near=True)
            if c is None:
                return None, "no near slot for core {}".format(kind)
            core.append(c)
    distract = []
    for kind in ("pot", "onion", "tomato", "dish", "serve"):
        for _ in range(pick(rng, p.get("distractor_" + kind, 0))):
            c = place(kind, near=False)
            if c is not None:
                distract.append(c)
    free = [t for t in home if rows[t[1]][t[0]] == " " and t[0] < max(third, 3)] or \
        [t for t in home if rows[t[1]][t[0]] == " "]
    rng.shuffle(free)
    if len(free) < n_cooks:
        return None, "not enough floor for cooks"
    for k, (x, y) in enumerate(free[:n_cooks]):
        rows[y][x] = str(k + 1)
    return {"rows": ["".join(r) for r in rows], "core_cells": core, "distractor_cells": distract,
            "W": W, "H": H, "split": split}, None


# ---- one sample ------------------------------------------------------------------------------
def sample(cfg, tier, index):
    rng = random.Random("{}|{}|{}".format(cfg["seed"], tier.get("name", ""), index))
    n_cooks = pick(rng, tier.get("cooks", 2))
    orders_n = pick(rng, tier.get("orders", 1))
    recipes = [dict(rng.choice(tier["recipes"])) for _ in range(orders_n)]
    need_tomato = any(r.get("tomato", 0) for r in recipes)
    sc = {"scenario_id": None, "seed": index, "counters": tier.get("counters", "shared")}
    info = {"tier": tier.get("name"), "sample_index": index}
    if tier["kind"] == "kitchen":
        k, why = build_kitchen(rng, tier["kitchen"], n_cooks, need_tomato)
        if k is None:
            return None, None, why
        sc["layout"] = {"grid": k["rows"]}
        info.update({k_: k[k_] for k_ in ("W", "H", "split")})
        info["core_cells"] = [list(c) for c in k["core_cells"]]
        info["distractor_cells"] = [list(c) for c in k["distractor_cells"]]
        info["core_cooks"] = min(n_cooks, pick(rng, tier.get("core_cooks", n_cooks)))
        sc["description"] = "{}x{} {}kitchen, {} cooks, {} distractor stations, {} orders".format(
            k["W"], k["H"], "split " if k["split"] else "", n_cooks, len(k["distractor_cells"]), orders_n)
    elif tier["kind"] == "layouts":
        name = rng.choice(tier["layouts"])
        rows, _ = builder.layout_params(name)
        if need_tomato and "T" not in "".join(rows):
            recipes = [{"onion": 3} for _ in recipes]
        sc["layout"] = {"name": name}
        sc["cooks"] = n_cooks
        if "pots" in tier:
            sc["pots"] = max(1, min(pick(rng, tier["pots"]), "".join(rows).count("P")))
        info["core_cooks"] = n_cooks
        sc["description"] = "{}, {} cooks, {} orders".format(name, n_cooks, orders_n)
    else:
        raise ValueError("tier kind must be kitchen or layouts")
    sc["orders"] = [{"recipe": r} for r in recipes]
    if "cook_time" in tier:
        sc["sim"] = {"cook_time": pick(rng, tier["cook_time"])}
    info["rho"] = pick(rng, tier.get("deadline", {}).get("rho", 1.0))
    if tier.get("deadline", {}).get("mode") == "fixed":
        for o in sc["orders"]:
            o["deadline"] = int(pick(rng, tier["deadline"]["value"]))
    return sc, info, None


def _write_yaml(path, sc):
    with open(path, "w") as f:
        yaml.safe_dump(sc, f, sort_keys=False, default_flow_style=None, width=200)


def core_problem(bench, prob, info):
    """Restrict the full HL problem to the core (stations, cooks, their poses / tiles)."""
    if "distractor_cells" not in info and info["core_cooks"] >= len(prob.of_type("cook")):
        return prob
    drop_cells = {tuple(c) for c in info.get("distractor_cells", [])}
    cooks = sorted(prob.of_type("cook"))[:info["core_cooks"]]
    stations = {o.name for o in prob.objects if "kind" in o.attrs and tuple(o.attrs["cell"]) not in drop_cells
                and (o.type != "counter" or info.get("split"))}
    faces = collections.defaultdict(set)
    for f in prob.facts:
        if f[0] == "faces":
            faces[f[1]].add(f[2])
    starts = {o.attrs["start_pose"] for o in prob.objects if o.type == "cook" and o.name in cooks}
    poses = {o.name for o in prob.objects if o.type == "pose" and (faces[o.name] & stations or o.name in starts)}
    tiles = {o.attrs["tile"] for o in prob.objects if o.name in poses}
    other = {o.name for o in prob.objects if o.type in ("recipe", "count", "order")}
    return prob.restrict(set(cooks) | stations | poses | tiles | other, prob.instance_id + "_core")


def serial_schedule(full, core):
    """Constructive certificate: the first core cook makes every order alone,
    one after another, at the first core pot, using exact travel times.
    Other cooks stay where they start (their tiles are avoided). Returns a
    TemporalPlan or None if a needed pose is blocked / unreachable."""
    from pddl_generator import TimedAction, TemporalPlan
    eps = 0.001
    facts = set(full.facts)
    core_names = {o.name for o in core.objects}
    tile = {f[1]: f[2] for f in facts if f[0] == "pose-tile"}
    cooks = sorted(o.name for o in core.objects if o.type == "cook")
    c = cooks[0]
    start = next(f[2] for f in facts if f[:2] == ("at", c))
    blocked = {tile[f[2]] for f in facts if f[0] == "at" and f[1] != c}
    poses_of = collections.defaultdict(list)
    for f in facts:
        if f[0] == "faces" and f[2] in core_names and tile[f[1]] not in blocked:
            poses_of[f[2]].append(f[1])
    kind = {o.name: o.attrs.get("kind") for o in core.objects}
    pots = sorted(n for n, k in kind.items() if k == "pot" and poses_of[n])
    by_item = {"onion": [], "tomato": [], "dish": []}
    for f in facts:
        if f[0] == "provides" and f[1] in core_names and poses_of[f[1]]:
            by_item[f[2]].append(f[1])
    serves = sorted(n for n, k in kind.items() if k == "serving" and poses_of[n])
    if not pots or not serves:
        return None
    tr = lambda a, b: 0 if a == b else full.fluents.get(("travel", a, b))  # noqa: E731
    pot = pots[0]
    pot_l = min(poses_of[pot], key=lambda l: (tr(start, l) is None, tr(start, l) or 0))
    plan, st = [], {"t": 0.0, "cur": start}

    def act(name, args, dur, at=None):
        t0 = st["t"] if at is None else max(st["t"], at)
        plan.append(TimedAction(round(t0, 3), name, tuple(args), float(dur)))
        st["t"] = t0 + dur + eps
        return t0 + dur

    def go(l):
        cur = st["cur"]
        if cur == l:
            return True
        d = tr(cur, l)
        if d is None:
            return False
        if tile[cur] == tile[l]:
            act("turn", (c, cur, l), d)
        else:
            act("move", (c, cur, l, tile[cur], tile[l]), d)
        st["cur"] = l
        return True

    def best(stations, then=None):
        cands = [(s_, l) for s_ in stations for l in poses_of[s_] if tr(st["cur"], l) is not None
                 and (then is None or tr(l, then) is not None)]
        if not cands:
            return None
        return min(cands, key=lambda x: tr(st["cur"], x[1]) + (tr(x[1], then) if then else 0))

    orders = sorted((o.name for o in full.objects if o.type == "order"), key=lambda n: int(n[1:]))
    recipe = {f[1]: f[2] for f in facts if f[0] == "order-recipe"}
    comp = {f[1]: (f[2], f[3]) for f in facts if f[0] == "recipe-of"}
    val = {o.name: o.attrs["value"] for o in full.objects if o.type == "count"}
    cnt = {v: k for k, v in val.items()}
    steps = {k[1]: v for k, v in full.fluents.items() if k[0] == "cook-steps"}
    for o in orders:
        r = recipe[o]
        need = ["onion"] * val[comp[r][0]] + ["tomato"] * val[comp[r][1]]
        a = b = 0
        for ing in need:
            pick_ = best(by_item[ing], pot_l)
            if pick_ is None or not go(pick_[1]):
                return None
            act("take", (c, pick_[1], pick_[0], ing), 1)
            if not go(pot_l):
                return None
            if ing == "onion":
                act("add-onion", (c, pot_l, pot, cnt[a], cnt[a + 1], cnt[b]), 1)
                a += 1
            else:
                act("add-tomato", (c, pot_l, pot, cnt[a], cnt[b], cnt[b + 1]), 1)
                b += 1
        started = act("start-cooking", (c, pot_l, pot, r, cnt[a], cnt[b]), 1)
        plan.append(TimedAction(round(started + eps, 3), "cook", (pot, r), float(steps[r])))
        ready = started + eps + steps[r]
        pick_ = best(by_item["dish"], pot_l)
        if pick_ is None or not go(pick_[1]):
            return None
        act("take", (c, pick_[1], pick_[0], "dish"), 1)
        if not go(pot_l):
            return None
        act("pick-soup", (c, pot_l, pot, r, cnt[0]), 1, at=ready + eps)
        pick_ = best(serves)
        if pick_ is None or not go(pick_[1]):
            return None
        act("deliver", (c, pick_[1], pick_[0], o, r), 1)
    return TemporalPlan(sorted(plan, key=lambda x: x.start))


def certify(bench, sc, info, how, timeout):
    """Certificate for the instance: POPF on the core problem (how = popf),
    the constructive serial schedule (serial), or POPF first then serial.
    The plan is checked with VAL on the full problem before it is accepted.
    Returns (plan, kind, seconds, sizes) or (None, reason, seconds, sizes)."""
    from pddl_generator import write_domain, write_problem, run_temporal_planner, validate_plan
    spec = bench.domain_spec()
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "s.yaml")
        _write_yaml(path, sc)
        prob = bench.build_problem(bench.load_scenario(path))
        core = core_problem(bench, prob, info).validate(spec)
        sizes = {"full_objects": len(prob.objects), "core_objects": len(core.objects),
                 "full_init_facts": len(prob.facts), "core_init_facts": len(core.facts)}
        d, pc, pf = (os.path.join(tmp, n) for n in ("d.pddl", "core.pddl", "full.pddl"))
        for fn, text in ((d, write_domain(spec)), (pc, write_problem(spec, core)), (pf, write_problem(spec, prob))):
            with open(fn, "w") as f:
                f.write(text)
        t0 = time.monotonic()
        plan, kind = None, None
        if how in ("popf", "popf_then_serial"):
            r = run_temporal_planner(d, pc, timeout=timeout)
            if r.status == "solved":
                plan, kind = r.plan, "popf_core"
        if plan is None and how in ("serial", "popf_then_serial"):
            plan, kind = serial_schedule(prob, core), "serial"
        secs = time.monotonic() - t0
        if plan is None:
            return None, "no certificate ({})".format(how), secs, sizes
        if not validate_plan(d, pf, plan).valid:
            return None, "{} certificate invalid on full problem".format(kind), secs, sizes
    return plan, kind, secs, sizes


# ---- main ----------------------------------------------------------------------------------------
def tiers_of(cfg):
    base = {k: v for k, v in cfg.items() if k not in ("tiers", "dataset", "seed")}
    return [dict(base, **t) for t in cfg.get("tiers", [{"name": None}])]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--count", type=int, help="instances per tier (overrides the config)")
    ap.add_argument("--tiers", nargs="*", help="only these tier names")
    ap.add_argument("--probe", type=float, default=0, help="also run POPF on each FULL problem for this many seconds")
    ap.add_argument("--no-pddl", action="store_true", help="only write scenario files")
    args = ap.parse_args(argv)
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    bench = OvercookedV1()
    out = os.path.join(bench.scenario_dir, cfg["dataset"])
    if os.path.isdir(out):
        shutil.rmtree(out)
    os.makedirs(out)
    instances, rejected = [], collections.Counter()
    for tier in tiers_of(cfg):
        if args.tiers and tier.get("name") not in args.tiers:
            continue
        count = args.count or tier.get("count", 10)
        made, index = 0, 0
        dl = tier.get("deadline", {"mode": "none"})
        need_plan = dl.get("mode") == "plan" or tier.get("require_plan", True)
        while made < count and index < count * tier.get("max_tries_factor", 10):
            sc, info, why = sample(cfg, tier, index)
            index += 1
            if sc is None:
                rejected[why] += 1
                continue
            sid = "{}_{}{:03d}".format(cfg["dataset"], (tier["name"] + "_") if tier.get("name") else "", made)
            sc["scenario_id"] = sid
            try:
                builder.make_mdp(*builder._resolve_grid(sc))
            except Exception as e:  # noqa: BLE001 -- invalid sample, counted
                rejected["invalid layout ({})".format(type(e).__name__)] += 1
                continue
            plan, secs = None, 0.0
            if need_plan:
                plan, kind, secs, sizes = certify(bench, sc, info, tier.get("certificate", "popf_then_serial"),
                                                  tier.get("certificate_timeout", 15))
                if plan is None:
                    rejected[kind] += 1
                    continue
                info.update(sizes, certificate_kind=kind)
                if dl.get("mode") == "plan":
                    ends = {a.args[3]: a.end for a in plan.actions if a.name == "deliver"}
                    for k, o in enumerate(sc["orders"]):
                        o["deadline"] = int(math.ceil(info["rho"] * ends["o{}".format(k + 1)]))
                    sc["description"] += ", deadlines = ceil({:.2f} x certificate)".format(info["rho"])
                with open(os.path.join(out, sid + ".certificate.plan"), "w") as f:
                    f.write(plan.to_text())
                info["certificate_makespan"] = round(plan.makespan, 3)
                info["certificate_popf_s"] = round(secs, 3)
            _write_yaml(os.path.join(out, sid + ".yaml"), sc)
            instances.append(dict(info, instance_id=sid, description=sc["description"]))
            made += 1
            print("  {:<34} {}".format(sid, sc["description"]), flush=True)
        if made < count:
            print("  tier {}: only {} of {} made (raise max_tries_factor?)".format(tier.get("name"), made, count))

    rc = 0
    if instances and not args.no_pddl:
        from pddl_generator import cli, validate_plan, TemporalPlan, run_temporal_planner
        rc = cli.main([out, "--quiet"])
        gen = os.path.join(bench.pddl_dir, cfg["dataset"])
        dom = os.path.join(bench.pddl_dir, "domain.pddl")
        for inst in instances:
            d = os.path.join(gen, inst["instance_id"])
            cert = os.path.join(out, inst["instance_id"] + ".certificate.plan")
            with open(os.path.join(d, "manifest.json")) as f:
                inst["size"] = json.load(f)["size"]
            if os.path.exists(cert):
                shutil.copy(cert, os.path.join(d, "certificate.plan"))
                with open(cert) as f:
                    v = validate_plan(dom, os.path.join(d, "problem.pddl"), TemporalPlan.parse(f.read()))
                inst["certificate_valid_on_full"] = v.valid
                if not v.valid:
                    rc = 1
                    print("CERTIFICATE INVALID", inst["instance_id"], v.errors[:4])
            if args.probe:
                t0 = time.monotonic()
                r = run_temporal_planner(dom, os.path.join(d, "problem.pddl"), timeout=args.probe)
                inst["probe_full_popf"] = {"timeout_s": args.probe, "status": r.status,
                                           "runtime_s": round(r.runtime_s, 2), "states": r.states_evaluated,
                                           "makespan": r.plan and round(r.plan.makespan, 3)}
                print("  probe {:<34} objects={:<4} full POPF: {:<9} {:6.1f}s  (certificate {:.1f}s on {} objects)"
                      .format(inst["instance_id"], inst["size"]["objects"], r.status, time.monotonic() - t0,
                              inst.get("certificate_popf_s", 0), inst.get("core_objects")))
    with open(os.path.join(out, "dataset.json"), "w") as f:
        json.dump({"config": cfg, "config_file": args.config, "instances": instances,
                   "rejected": dict(rejected)}, f, indent=1)
    print("{} scenarios -> {} (rejected samples: {})".format(len(instances), os.path.relpath(out),
                                                             dict(rejected) or "none"))
    return rc


if __name__ == "__main__":
    sys.exit(main())
