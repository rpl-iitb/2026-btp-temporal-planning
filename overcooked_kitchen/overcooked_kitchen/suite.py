"""Benchmark suites: import of the v2 / v3 instances with their certificates, iteration, bench runs, tables.

A suite root holds <split>/<id>.yaml (+ <id>.certificate.plan) and index.json {"instances": [row, ...]}; a row has
name, split, tier, source (v2 | v3 | generated), cooks, orders, cert_kind, cert_valid, cert_source, cert_plan
(relative path or null), cert_makespan, plan_makespan, path (relative).
  cert_kind "val": a PDDL plan checked by VAL against this package's domain and export of the instance (imported
  v2 / v3 certificates: also replayed by PlanPolicy on the sampled clock with nominal durations, all served, no burn,
  no lateness; POPF winners of generate);
  cert_kind "sim": simulator-certified (the dispatcher's nominal run serves every order with no burnt soup; no plan);
  cert_kind "plan": an imported plan that failed those checks (cert_valid false, cert_note says why; no
  cert_makespan) or could not be checked (no VAL binary; cert_valid null).
  Imported plans get this package's station names: v2 / v3 numbered only the scenario's policy counters, this
  package numbers every counter, so counters are renamed by cell.
  cert_source: who made the certificate (v2 | v3 | popf | dispatcher).
  cert_makespan: when the last order is served (plan: the latest end of a deliver action; sim: Sim makespan), the
  quantity Sim reports as makespan; plan_makespan: the latest end of any plan action (diners and timers included).
bench seeds: Sim seed = crc32(instance name) x 1000 + s, independent of the instance's position in the split.
"""
"""This file is UN-VERIFIED"""
import glob
import importlib
import importlib.util
import json
import os
import shutil
import sys
import time
import zlib
from collections import defaultdict
from typing import Iterator, List

import numpy as np

from overcooked_kitchen.layout import Layout
from overcooked_kitchen.pddlkit import TemporalPlan, TimedAction
from overcooked_kitchen.scenarios import load_scenario, scenario_orders

V2_SPLITS = ("v2_train", "v2_val", "v2_test")
V3_SPLITS = ("v3_train", "v3_val", "v3_test")
ROW_KEYS = ("name", "tier", "seed", "success", "failure", "makespan", "makespan_over_cert", "on_time",
            "total_lateness", "burnt", "decisions", "wall_s")


# ---- index ----------------------------------------------------------------------------------------
def read_index(root) -> List[dict]:
    path = os.path.join(root, "index.json")
    if not os.path.isfile(path):
        return []
    with open(path) as f:
        return json.load(f)["instances"]


def update_index(root, rows, drop_splits=()):
    """Merge rows into root/index.json (same name replaced; rows of drop_splits removed first)."""
    new = {r["name"] for r in rows}
    keep = [r for r in read_index(root) if r["name"] not in new and r["split"] not in drop_splits]
    allrows = sorted(keep + list(rows), key=lambda r: (r["split"], r["name"]))
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, "index.json"), "w") as f:
        json.dump({"instances": allrows}, f, indent=1)
    return allrows


def tier_of(name, split) -> str:
    """v3_test_m3_000 (split v3_test) -> m3."""
    return name[len(split) + 1:].rsplit("_", 1)[0]


def delivery_makespan(plan) -> float:
    """Latest end of a deliver action (the time the last order is served)."""
    return round(max((a.end for a in plan.actions if a.name == "deliver"), default=0.0), 3)


def bench_seed(name, s) -> int:
    return zlib.crc32(name.encode()) * 1000 + s


def _cooks(sc):
    return sum(ch.isdigit() for r in sc["grid"] for ch in r)


# ---- import -----------------------------------------------------------------------------------------
def _sources(v2_root, v3_root):
    """(source, split, id, scenario path, plan path) of every v2 / v3 instance found."""
    out = []
    for split in V2_SPLITS if v2_root else ():
        for p in sorted(glob.glob(os.path.join(v2_root, "scenarios", split, "*.yaml"))):
            sid = os.path.basename(p)[:-5]
            out.append(("v2", split, sid, p, os.path.join(os.path.dirname(p), sid + ".certificate.plan")))
    for split in V3_SPLITS if v3_root else ():
        for p in sorted(glob.glob(os.path.join(v3_root, "generated_pddl", split, "*", "scenario.yaml"))):
            d = os.path.dirname(p)
            out.append(("v3", split, os.path.basename(d), p, os.path.join(d, "certificate.plan")))
    return out


def rename_stations(plan, sc) -> TemporalPlan:
    """The plan with station names numbered as v2 / v3 did (counters: only the scenario's policy counters) replaced by
    this package's names (counters numbered over every counter), matched by cell."""
    lay = Layout(sc["grid"])
    ours = {st.cell: st.name for st in lay.stations("all")}
    names = {st.name: ours[st.cell] for st in lay.stations(sc.get("counters", "none") or "none")
             if st.cell in ours and ours[st.cell] != st.name}
    if not names or not plan.objects_used() & set(names):
        return plan
    return TemporalPlan([TimedAction(a.start, a.name, tuple(names.get(x, x) for x in a.args), a.duration)
                         for a in plan.actions])


def check_certificate(sc, plan):
    """(valid, note): VAL against this package's domain and export of the scenario at t = 0, then a PlanPolicy
    replay on the sampled clock with nominal durations (all served, no burn, no lateness). valid None: no VAL."""
    import tempfile
    from overcooked_kitchen import pddl
    from overcooked_kitchen.core import Sim
    from overcooked_kitchen.pddlkit import planners
    if not planners.available("validate"):
        return None, "VAL not found"
    with tempfile.TemporaryDirectory() as tmp:
        d, p = pddl.write(Sim(sc), tmp)
        res = planners.validate_plan(d, p, plan)
    if not res.valid:
        why = [x.strip() for x in res.stdout.splitlines() if any(k in x for k in ("rror", "unknown", "nsatisfied"))]
        return False, "VAL: " + ((why or res.errors or ["invalid"])[0])
    return replay(sc, plan)


def replay(sc, plan, clock="sampled"):
    """(ok, note): PlanPolicy runs the plan (made at t = 0) in Sim with nominal durations: all served, no burn, no
    lateness."""
    from overcooked_kitchen import pddl
    from overcooked_kitchen.core import Sim
    sim = Sim(sc, clock=clock)
    m = sim.run(pddl.PlanPolicy(plan, pddl.export(sim)))
    if not m["success"] or m["burnt"] or m["late"]:
        return False, "replay ({}): failure {}, served {}, burnt {}, late {}".format(
            clock, m["failure"], m["served"], m["burnt"], m["late"])
    return True, None


def import_existing(out_dir, v2_root, v3_root, check=True) -> List[dict]:
    """Copy the v2 / v3 instances (ids kept) and their certificate plans (station names renamed to this package's)
    into out_dir, checking each plan (check_certificate) unless check=False; returns the new rows."""
    rows = []
    for source, split, sid, path, plan_path in _sources(v2_root, v3_root):
        sc = load_scenario(path)
        if sc["scenario_id"] != sid:
            raise ValueError("{}: scenario_id {} != {}".format(path, sc["scenario_id"], sid))
        os.makedirs(os.path.join(out_dir, split), exist_ok=True)
        shutil.copyfile(path, os.path.join(out_dir, split, sid + ".yaml"))
        cert, kind, valid, note, makespan, plan_makespan = None, None, None, None, None, None
        if os.path.isfile(plan_path):
            cert = "{}/{}.certificate.plan".format(split, sid)
            with open(plan_path) as f:
                source_plan = TemporalPlan.parse(f.read())
            plan = rename_stations(source_plan, sc)
            if plan is source_plan:
                shutil.copyfile(plan_path, os.path.join(out_dir, cert))
            else:
                with open(os.path.join(out_dir, cert), "w") as f:
                    f.write("; station names renamed to overcooked_kitchen's (counters by cell)\n" + plan.to_text())
            valid, note = check_certificate(sc, plan) if check else (None, "not checked")
            kind = "val" if valid else "plan"
            makespan = delivery_makespan(plan) if valid is not False else None
            plan_makespan = round(plan.makespan, 3)
        rows.append({"name": sid, "split": split, "tier": tier_of(sid, split), "source": source,
                     "cooks": _cooks(sc), "orders": len(scenario_orders(sc)), "cert_kind": kind, "cert_valid": valid,
                     "cert_note": note, "cert_source": source if cert else None, "cert_plan": cert,
                     "cert_makespan": makespan, "plan_makespan": plan_makespan,
                     "path": "{}/{}.yaml".format(split, sid)})
    update_index(out_dir, rows)
    return rows


# ---- runs -------------------------------------------------------------------------------------------
def iter_suite(root, split) -> Iterator[dict]:
    """Index rows of a split (all splits if split is None), with "file" (absolute path) and "scenario"."""
    for r in read_index(root):
        if split is None or r["split"] == split:
            path = os.path.join(root, r["path"])
            yield dict(r, file=os.path.abspath(path), scenario=load_scenario(path))


def load_policy(name, seed=0):
    """A baseline name (baselines.make) or "module:function" / "file.py:function" (the attribute is the policy)."""
    if ":" in name:
        mod, fn = name.rsplit(":", 1)
        if mod.endswith(".py"):
            spec = importlib.util.spec_from_file_location("user_policy", mod)
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            return getattr(m, fn)
        if os.getcwd() not in sys.path:
            sys.path.insert(0, os.getcwd())
        return getattr(importlib.import_module(mod), fn)
    from overcooked_kitchen.baselines import make
    return make(name, seed=seed)


def bench(root, split, policy, seeds=3, clock="sampled", cv_scale=1.0, out=None) -> List[dict]:
    """Run `policy` (name or module:function) on every instance of a split, `seeds` episodes each
    (Sim seed = bench_seed(name, s); cv_scale > 0 -> default noise scaled, 0 -> nominal). makespan_over_cert =
    makespan / cert_makespan (last served time in both)."""
    from overcooked_kitchen.core import Sim
    rows = []
    for inst in iter_suite(root, split):
        for s in range(seeds):
            noise = "default" if cv_scale > 0 else None
            sim = Sim(inst["scenario"], clock=clock, seed=bench_seed(inst["name"], s), noise=noise, cv_scale=cv_scale)
            t0 = time.perf_counter()
            m = sim.run(load_policy(policy, s))
            cert = inst.get("cert_makespan")
            ratio = round(m["makespan"] / cert, 4) if m["makespan"] and cert else None
            rows.append({"name": inst["name"], "tier": inst["tier"], "seed": s, "success": m["success"],
                         "failure": m["failure"], "makespan": m["makespan"], "makespan_over_cert": ratio,
                         "on_time": m["on_time"], "total_lateness": m["total_lateness"], "burnt": m["burnt"],
                         "decisions": m["decisions"], "wall_s": round(time.perf_counter() - t0, 3)})
    if out:
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        with open(out, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    return rows


def _mean(xs, fmt="{:.3f}"):
    return fmt.format(float(np.mean(xs))) if xs else "-"


def table(rows) -> str:
    """Markdown table per tier (+ all): runs, success, deadlock / stall, makespan / certificate, on-time,
    lateness, burnt, seconds per run."""
    by = defaultdict(list)
    for r in rows:
        by[r["tier"]].append(r)
        by["all"].append(r)
    lines = ["| tier | runs | success | deadlock/stall | makespan / cert | on-time | lateness | burnt | s / run |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for tier in sorted(by, key=lambda t: (t == "all", t)):
        rr = by[tier]
        lines.append("| {} | {} | {:.0%} | {:.0%} | {} | {:.0%} | {} | {} | {} |".format(
            tier, len(rr), np.mean([r["success"] for r in rr]),
            np.mean([r["failure"] in ("deadlock", "stalled") for r in rr]),
            _mean([r["makespan_over_cert"] for r in rr if r["makespan_over_cert"]]),
            np.mean([r["on_time"] for r in rr]), _mean([r["total_lateness"] for r in rr], "{:.1f}"),
            _mean([r["burnt"] for r in rr], "{:.2f}"), _mean([r["wall_s"] for r in rr], "{:.2f}")))
    return "\n".join(lines)
