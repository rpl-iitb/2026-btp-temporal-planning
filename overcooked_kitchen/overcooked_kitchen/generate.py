"""Certified kitchens: single-cook rooms (a core of stations near the cook + distractors) and multi-cook rooms (3-4
cooks, several stations of every kind), from configs shaped as configs/dataset_single.yaml / dataset_multi.yaml.
Room generation is ported from earlier research code.

A tier with a `cooks` key is a multi-cook room (burn window >= the longest rack -> pot round trip + 8), else a
single-cook room. Certificate: the dispatcher's nominal run (sampled clock, no noise) must serve every
order with no burnt soup; its makespan (last served time) is the certificate makespan. With POPF and VAL available
(config `certificate: both`, the default; `dispatcher` skips POPF; any other value is an error), POPF also runs on
the full initial problem (`certificate_timeout` s); a VAL-valid POPF plan whose last deliver ends earlier and that
PlanPolicy replays in Sim (nominal, no burn, no lateness) wins and is written as <id>.certificate.plan. Service tiers
(`deadline: {rho: ...}`) get deadlines ceil(rho x the certificate's delivery time) and are certified again
(dispatcher on time, or the POPF plan VAL-valid with the deadlines). Uncertified samples are discarded and
the next index is tried (at most count x max_tries_factor samples per tier).

Writes out_dir/<split>/<id>.yaml (+ .certificate.plan), out_dir/<split>/dataset.json (config, requested and made per
tier, rejection reasons) and rows in out_dir/index.json (cert_kind "sim" for the dispatcher, "val" for POPF; see suite).
"""
"""This file is UN-VERIFIED"""
import collections
import concurrent.futures
import json
import math
import os
import random
import shutil
import tempfile
import time
from typing import List

import yaml

from overcooked_kitchen import pddl
from overcooked_kitchen.baselines import make
from overcooked_kitchen.core import Sim
from overcooked_kitchen.layout import INGREDIENTS, Layout
from overcooked_kitchen.pddlkit import planners
from overcooked_kitchen.suite import delivery_makespan, read_index, replay, update_index

CERTIFICATES = ("both", "dispatcher")
CHAR = {"onion": "O", "tomato": "T", "carrot": "C", "board": "B", "pot": "P", "rack": "D", "dropoff": "R",
        "sink": "W", "serve": "S"}


# ---- rooms ----------------------------------------------------------------------------------------------
def pick(rng, v):
    """3 (fixed) | [lo, hi] (uniform; ints inclusive) | [a, b, c, ...] (choice)."""
    if isinstance(v, list) and len(v) == 2 and all(isinstance(x, (int, float)) for x in v):
        if all(isinstance(x, int) for x in v):
            return rng.randint(v[0], v[1])
        return rng.uniform(v[0], v[1])
    if isinstance(v, list):
        return rng.choice(v)
    return v


def make_room(rng, W, H, walls):
    """Border of counters; `walls` vertical counter walls with one gap each (alternating top / bottom)."""
    rows = [["X"] * W] + [["X"] + [" "] * (W - 2) + ["X"] for _ in range(H - 2)] + [["X"] * W]
    xs = [round((k + 1) * (W - 1) / (walls + 1)) for k in range(walls)]
    for k, x in enumerate(xs):
        if x <= 1 or x >= W - 2:
            continue
        gap = 1 if k % 2 == 0 else H - 2
        for y in range(1, H - 1):
            if y != gap:
                rows[y][x] = "X"
    return rows


def border_slots(lay, rows):
    """Counter cells (not corners) with a floor neighbour: candidate station cells."""
    H, W = len(rows), len(rows[0])
    out = []
    for y in range(H):
        for x in range(W):
            if rows[y][x] == "X" and lay.access_poses((x, y)) and not (x in (0, W - 1) and y in (0, H - 1)):
                out.append((x, y))
    return out


def build_single(rng, p, need):
    """Single-cook room: a core (one crate per needed vegetable, board, pots, rack, drop-off, sink, serving window)
    near the cook in the left third, distractor stations preferring the middle / far side."""
    W, H = pick(rng, p["width"]), pick(rng, p["height"])
    rows = make_room(rng, W, H, pick(rng, p.get("walls", 0)))
    lay = Layout(["".join(r) for r in rows])
    comp = lay.regions()
    home = comp[min(comp)]
    slots = border_slots(lay, rows)
    rng.shuffle(slots)
    used = set()
    third = max(W / 3.0, 3)

    def place(kind, near):
        for c in slots:
            if c in used or not any(comp[(x, y)] == home for x, y, _ in lay.access_poses(c)):
                continue
            if near and c[0] > third + 1:
                continue
            if not near and c[0] < third and rng.random() < 0.8:
                continue
            used.add(c)
            rows[c[1]][c[0]] = CHAR[kind]
            return c
        return None

    core = []
    kinds = [(v, 1) for v in need] + [("board", 1), ("pot", pick(rng, p.get("core_pots", 1))), ("rack", 1),
                                      ("dropoff", 1), ("sink", 1), ("serve", 1)]
    for kind, n in kinds:
        for _ in range(n):
            c = place(kind, near=True)
            if c is None:
                return None, "no near slot for core {}".format(kind)
            core.append(c)
    distract = []
    for kind in ("onion", "tomato", "carrot", "board", "pot", "sink", "serve"):
        for _ in range(pick(rng, p.get("distractor_" + kind, 0))):
            c = place(kind, near=False)
            if c is not None:
                distract.append(c)
    free = [t for t in comp if comp[t] == home and rows[t[1]][t[0]] == " " and t[0] <= third]
    if not free:
        free = [t for t in comp if comp[t] == home and rows[t[1]][t[0]] == " "]
    if not free:
        return None, "no floor for the cook"
    x, y = rng.choice(sorted(free))
    rows[y][x] = "1"
    return {"rows": ["".join(r) for r in rows], "W": W, "H": H, "stations": len(core) + len(distract),
            "distractors": len(distract), "counters": "none"}, None


def build_multi(rng, p, need, n_cooks):
    """Multi-cook room: several stations of every kind spread over the largest floor region, cooks far apart."""
    W, H = pick(rng, p["width"]), pick(rng, p["height"])
    rows = make_room(rng, W, H, pick(rng, p.get("walls", 0)))
    lay = Layout(["".join(r) for r in rows])
    comp = lay.regions()
    home = max(set(comp.values()), key=lambda r: sum(1 for v in comp.values() if v == r))
    slots = [c for c in border_slots(lay, rows) if any(comp[(x, y)] == home for x, y, _ in lay.access_poses(c))]
    rng.shuffle(slots)
    kinds = []
    for v in need:
        kinds += [v] * pick(rng, p.get("crates", 1))
    for k in ("board", "pot", "rack", "dropoff", "sink", "serve"):
        kinds += [k] * pick(rng, p.get(k, 1))
    if len(kinds) > len(slots):
        return None, "room too small for {} stations".format(len(kinds))
    slots.sort(key=lambda c: (c[0] + c[1] * 0.37) % 7.3)
    for c, kind in zip(slots, kinds):
        rows[c[1]][c[0]] = CHAR[kind]
    rest = slots[len(kinds):]
    counters = [list(c) for c in rest[:n_cooks + 2]]
    free = sorted(t for t in comp if comp[t] == home and rows[t[1]][t[0]] == " ")
    if len(free) < n_cooks * 3:
        return None, "no floor for the cooks"
    pts = [rng.choice(free)]
    while len(pts) < n_cooks:
        pts.append(max(free, key=lambda t: min(abs(t[0] - q[0]) + abs(t[1] - q[1]) for q in pts)))
    for k, (x, y) in enumerate(pts):
        rows[y][x] = str(k + 1)
    return {"rows": ["".join(r) for r in rows], "W": W, "H": H, "stations": len(kinds), "distractors": 0,
            "counters": counters}, None


def sample(cfg, tier, index):
    """-> (scenario dict, info) or (None, reason). Same random draws as the earlier research generators."""
    rng = random.Random("{}:{}:{}".format(cfg["seed"], tier["name"], index))
    multi = "cooks" in tier
    n_orders = pick(rng, tier["orders"])
    n_cooks = pick(rng, tier["cooks"]) if multi else 1
    orders = [dict(rng.choice(tier["recipes"])) for _ in range(n_orders)]
    need = sorted({v for o in orders for v in o["recipe"]}, key=INGREDIENTS.index)
    if multi:
        k, why = build_multi(rng, tier["kitchen"], need, n_cooks)
    else:
        k, why = build_single(rng, tier["kitchen"], need)
    if k is None:
        return None, why
    clean = max(1, pick(rng, tier.get("plates_clean", n_orders)))
    clean = clean if multi else min(n_orders, clean)
    sc = {"scenario_id": "{}_{}_{:03d}".format(cfg["dataset"], tier["name"], index),
          "description": "generated: tier {}, {}x{}, {} cook(s), {} orders, {} stations ({} distractors)".format(
              tier["name"], k["W"], k["H"], n_cooks, n_orders, k["stations"], k["distractors"]),
          "grid": k["rows"], "counters": k["counters"],
          "plates": {"clean": clean, "dirty": pick(rng, tier.get("plates_dirty", 0))},
          "dirty_pots": pick(rng, tier.get("dirty_pots", 0)),
          "sim": {"burn": pick(rng, tier.get("burn", 20))},
          "orders": [{"dish": o["dish"], "recipe": dict(o["recipe"])} for o in orders], "seed": index}
    rho = round(pick(rng, tier["deadline"]["rho"]), 3) if tier.get("deadline") else None
    if multi:
        widen_burn(sc)
    return sc, {"tier": tier["name"], "cooks": n_cooks, "orders": n_orders, "rho": rho}


def widen_burn(sc):
    """Multi-cook rooms: burn window >= 2 x the longest (pot -> nearest rack) travel + 8, so a plate can always come in time."""
    lay = Layout(sc["grid"])
    cells = {}
    for y, r in enumerate(lay.terrain):
        for x, ch in enumerate(r):
            cells.setdefault(ch, []).append((x, y))
    racks = [q for c in cells.get("D", []) for q in lay.access_poses(c)]
    far = 0
    for pot in cells.get("P", []):
        best = None
        for a in lay.access_poses(pot):
            dist = lay.bfs_poses(a)
            d = min((dist[b] for b in racks if b in dist), default=None)
            if d is not None and (best is None or d < best):
                best = d
        far = max(far, best or 0)
    if sc["sim"]["burn"] < 2 * far + 8:
        sc["sim"]["burn"] = int(math.ceil(2 * far + 8))


# ---- certificates --------------------------------------------------------------------------------------
def run_dispatcher(sc) -> dict:
    sim = Sim(sc, clock="sampled", seed=0, noise=None)
    m = sim.run(make("dispatcher"))
    m["served_at"] = dict(sim.world.served)
    return m


def run_popf(sc, timeout):
    """POPF on the full initial problem, VAL-checked. -> (status, plan or None, seconds)."""
    if not (planners.available("popf") and planners.available("validate")):
        return "no_binary", None, 0.0
    sim = Sim(sc)
    with tempfile.TemporaryDirectory() as tmp:
        d, p = pddl.write(sim, tmp)
        res = planners.run_temporal_planner(d, p, timeout=timeout)
        if res.status != "solved" or res.plan is None:
            return res.status, None, res.runtime_s
        if not pddl.validate(d, p, res.plan):
            return "invalid", None, res.runtime_s
    return "valid", res.plan, res.runtime_s


def val_ok(sc, plan) -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        d, p = pddl.write(Sim(sc), tmp)
        return pddl.validate(d, p, plan)


def certify(sc, rho=None, how="both", timeout=20.0):
    """-> (certificate dict, None) or (None, reason). Adds deadlines to sc when rho is given."""
    m = run_dispatcher(sc)
    if not m["success"] or m["burnt"]:
        return None, "dispatcher: {}, served {}, burnt {}".format(m["failure"] or "burnt", m["served"], m["burnt"])
    cert = {"cert_kind": "sim", "cert_valid": True, "cert_note": None, "cert_source": "dispatcher",
            "cert_makespan": m["makespan"], "plan_makespan": None,
            "dispatcher_makespan": m["makespan"], "popf_status": "skipped", "popf_makespan": None, "popf_s": 0.0}
    plan = None
    if how == "both":
        status, plan, secs = run_popf(sc, timeout)
        cert.update(popf_status=status, popf_s=round(secs, 2))
        if plan is not None:
            cert["popf_makespan"] = delivery_makespan(plan)
            if cert["popf_makespan"] < m["makespan"] and replay(sc, plan)[0]:
                cert.update(cert_kind="val", cert_source="popf", cert_makespan=cert["popf_makespan"],
                            plan_makespan=round(plan.makespan, 3))
    if rho is None:
        return dict(cert, plan=plan if cert["cert_kind"] == "val" else None), None
    if cert["cert_kind"] == "val":
        served = {a.args[3]: a.end for a in plan.actions if a.name == "deliver"}
    else:
        served = m["served_at"]
    for k, o in enumerate(sc["orders"]):
        o["deadline"] = int(math.ceil(rho * served["o{}".format(k + 1)]))
    sc["description"] += ", deadlines = ceil({:.2f} x certificate)".format(rho)
    m2 = run_dispatcher(sc)
    if not m2["success"] or m2["burnt"]:
        return None, "with deadlines: dispatcher {}".format(m2["failure"])
    if cert["cert_kind"] == "val":
        if not val_ok(sc, plan) or not replay(sc, plan)[0]:
            return None, "with deadlines: POPF certificate invalid"
    elif m2["on_time"] < 1.0:
        return None, "with deadlines: dispatcher late"
    cert["dispatcher_makespan"] = m2["makespan"]
    if cert["cert_kind"] == "sim":
        cert["cert_makespan"] = m2["makespan"]
    return dict(cert, plan=plan if cert["cert_kind"] == "val" else None), None


# ---- datasets ------------------------------------------------------------------------------------------
def make_one(cfg, tier, index, out):
    """-> (row, None) or (None, reason). Writes <id>.yaml (+ .certificate.plan) into out."""
    sc, info = sample(cfg, tier, index)
    if sc is None:
        return None, info
    cert, why = certify(sc, info["rho"], tier.get("certificate", "both"), tier.get("certificate_timeout", 20))
    if cert is None:
        return None, why
    sid = sc["scenario_id"]
    with open(os.path.join(out, sid + ".yaml"), "w") as f:
        yaml.safe_dump(sc, f, sort_keys=False)
    plan = cert.pop("plan")
    split = cfg["dataset"]
    if plan is not None:
        with open(os.path.join(out, sid + ".certificate.plan"), "w") as f:
            f.write(plan.to_text())
    row = {"name": sid, "split": split, "tier": info["tier"], "source": "generated", "cooks": info["cooks"],
           "orders": info["orders"], "cert_plan": "{}/{}.certificate.plan".format(split, sid) if plan else None}
    row.update(cert)
    row.update(rho=info["rho"], path="{}/{}.yaml".format(split, sid))
    return row, None


def _discard(out, sid):
    for ext in (".yaml", ".certificate.plan"):
        if os.path.exists(os.path.join(out, sid + ext)):
            os.remove(os.path.join(out, sid + ext))


def _tier_rows(cfg, tier, n, out, ex, jobs, rejected):
    rows = []
    idx = 0
    limit = n * tier.get("max_tries_factor", 4)
    while len(rows) < n and idx < limit:
        batch = list(range(idx, min(limit, idx + max(jobs, n - len(rows)))))
        idx += len(batch)
        args = ([cfg] * len(batch), [tier] * len(batch), batch, [out] * len(batch))
        results = map(make_one, *args) if ex is None else ex.map(make_one, *args)
        for row, why in results:
            if row is None:
                rejected[str(why)[:80]] += 1
            elif len(rows) < n:
                rows.append(row)
            else:
                _discard(out, row["name"])
    return rows


def generate(out_dir, config: dict, splits=None, seed=0, count=None, jobs=1, force=False) -> List[dict]:
    """Generate the config's splits (all, or those named in `splits`) into out_dir; seed offsets the config seeds.
    count overrides the instances per tier. An existing split (folder or index rows) is replaced only with force=True
    (FileExistsError otherwise); other splits are kept. Returns the new index rows."""
    todo = [sp for sp in config["splits"] if not splits or sp["dataset"] in splits]
    for sp in todo:
        for tier in sp["tiers"]:
            how = {**config, **sp, **tier}.get("certificate", "both")
            if how not in CERTIFICATES:
                raise ValueError("split {} tier {}: certificate must be one of {}, got {!r}".format(
                    sp["dataset"], tier.get("name"), ", ".join(CERTIFICATES), how))
    have = {r["split"] for r in read_index(out_dir)}
    clash = [sp["dataset"] for sp in todo
             if sp["dataset"] in have or os.path.exists(os.path.join(out_dir, sp["dataset"]))]
    if clash and not force:
        raise FileExistsError("split(s) {} already exist in {}; use force=True (--force) to replace them".format(
            ", ".join(clash), out_dir))
    made = []
    for split in todo:
        cfg = {k: v for k, v in config.items() if k != "splits"}
        cfg.update({k: v for k, v in split.items() if k != "tiers"})
        cfg["seed"] = cfg.get("seed", 0) + seed
        out = os.path.join(out_dir, cfg["dataset"])
        shutil.rmtree(out, ignore_errors=True)
        os.makedirs(out)
        t0 = time.time()
        rows, rejected, requested, got = [], collections.Counter(), {}, {}
        ex = concurrent.futures.ProcessPoolExecutor(jobs) if jobs > 1 else None
        try:
            for tier in split["tiers"]:
                tier = dict(cfg, **tier)
                requested[tier["name"]] = count or tier.get("count", 10)
                new = _tier_rows(cfg, tier, requested[tier["name"]], out, ex, jobs, rejected)
                got[tier["name"]] = len(new)
                rows += new
        finally:
            if ex is not None:
                ex.shutdown()
        rows.sort(key=lambda r: r["name"])
        with open(os.path.join(out, "dataset.json"), "w") as f:
            json.dump({"config": cfg, "requested": requested, "made": got, "rejected": dict(rejected),
                       "seconds": round(time.time() - t0, 1)}, f, indent=1)
        update_index(out_dir, rows, drop_splits=[cfg["dataset"]])
        made += rows
    return made
