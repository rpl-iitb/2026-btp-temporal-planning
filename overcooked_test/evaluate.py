"""Evaluate baselines and T-iFLAX versions on a split; rows -> results/<method>/<tag>.jsonl, summary + plots
-> results/summary/<tag>.*  (tag = <split>_<budget>_seed<k>; resumable). See overcooked_test/README.md."""

"""This file is UN-VERIFIED"""

import argparse
import collections
import concurrent.futures
import csv
import importlib
import json
import os
import time

import numpy as np

from algorithm.pddlkit import TemporalPlan
from algorithm import PDDL, RESULTS, results_dir, split_dirs
from algorithm.grounding import Task
from algorithm.planning import ensure_hash_seed

# method id -> (module with build(key, spec, seed, **overrides), results folder)
METHODS = {"B1": ("baselines.b1", "b1"), "B2": ("baselines.b2", "b2"), "B3": ("baselines.b3", "b3"),
           "V1": ("algorithm.v1", "v1"), "V2": ("algorithm.v2", "v2"), "V2.1": ("algorithm.v2", "v2"),
           "V3": ("algorithm.v3", "v3"), "V3U": ("algorithm.v3", "v3"), "V3.1": ("algorithm.v3", "v3"),
           "V4": ("algorithm.v4", "v4"), "V4.1": ("algorithm.v4", "v4"), "V4.2": ("algorithm.v4", "v4"),
           "V5": ("algorithm.v5", "v5"), "V10": ("algorithm.v10", "v10"),
           # with kitchen parts / kitchen models: kitchen_methods.py
           "V8": ("overcooked_test.kitchen_methods", "v8"), "V8S": ("overcooked_test.kitchen_methods", "v8"),
           "V8.2": ("overcooked_test.kitchen_methods", "v8"), "V9": ("overcooked_test.kitchen_methods", "v9"),
           "V11": ("overcooked_test.kitchen_methods", "v11")}
# V12 / V13 are closed-loop controllers under random durations: python -m overcooked_test.v3_eval (kitchen_v13.py).


def build(method, spec, seed=0, overrides=None):
    return importlib.import_module(METHODS[method][0]).build(method, spec, seed, **(overrides or {}))


def rows_file(method, tag):
    return os.path.join(results_dir(METHODS[method][1]), tag + ".jsonl")


def load_rows(tag):
    """All methods' rows of one run tag."""
    rows = []
    for folder in sorted({f for _, f in METHODS.values()}):
        p = os.path.join(RESULTS, folder, tag + ".jsonl")
        if os.path.exists(p):
            rows += [json.loads(line) for line in open(p)]
    return rows


def certificate_makespan(d):
    p = os.path.join(d, "certificate.plan")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return TemporalPlan.parse(f.read()).makespan


MAX_LOAD = None        # set from --max_load: wait for a quiet machine before each run


def wait_for_quiet(max_load, poll=5.0, give_up=3600):
    """Block until the 1-min load average is below max_load (fair timing when the machine is shared)."""
    t0 = time.time()
    while os.getloadavg()[0] > max_load and time.time() - t0 < give_up:
        time.sleep(poll)


def run_one(method, d, budget, seed, overrides, label):
    if MAX_LOAD:
        wait_for_quiet(MAX_LOAD)
    task = Task.load(d)
    planner = build(method, task.spec, seed, overrides)
    t = time.time()
    res = planner.solve(task, budget)
    row = res.to_dict()
    row.update({"split_dir": os.path.relpath(d, PDDL), "budget_label": label, "seed": seed,
                "wall_s": round(time.time() - t, 3), "loadavg": os.getloadavg()[0],
                "certificate_makespan": certificate_makespan(d), "tier": os.path.basename(d).rsplit("_", 1)[0]})
    return row


def summarize(rows):
    best = {}
    for r in rows:
        k = r["split_dir"]
        cands = [x for x in (r["makespan"], r.get("certificate_makespan")) if x is not None]
        if cands:
            best[k] = min([best.get(k, float("inf"))] + cands)
    groups = {}
    for r in rows:
        groups.setdefault((r["method"], r["budget_label"]), []).append(r)
    out = []
    for (m, b), rs in sorted(groups.items()):
        sol = [r for r in rs if r["solved"]]
        out.append({
            "method": m, "budget": b, "instances": len(rs), "solved": len(sol),
            "coverage": round(len(sol) / len(rs), 3),
            "first_s": _mean([r["first_valid_s"] for r in sol]),
            "wpt_s": _mean([r["first_valid_s"] if r["solved"] else r["budget_s"] for r in rs]),
            "total_s": _mean([r["total_s"] for r in rs]),
            "makespan": _mean([r["makespan"] for r in sol]),
            "first_makespan": _mean([r["first_makespan"] for r in sol]),
            "gap": _mean([(r["makespan"] - best[r["split_dir"]]) / best[r["split_dir"]] for r in sol]),
            "kept": _mean([r["n_objects"] / r["n_total"] for r in sol if r["n_objects"]]),
            "calls": _mean([r["planner_calls"] for r in rs]),
            "timeouts": _mean([r["timeouts"] for r in rs]),
            "proofs": _mean([r["proofs"] for r in rs]),
            "invalid_on_full": sum(r["invalid_on_full"] for r in rs),
            "branches": _count([r["branch"] for r in sol]),
        })
    return out, best


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(float(np.mean(xs)), 3) if xs else None


def _count(xs):
    c = {}
    for x in xs:
        c[x] = c.get(x, 0) + 1
    return c


def paired(rows, a, b, budget):
    """Instances both methods attempted at this budget."""
    by = {}
    for r in rows:
        if r["budget_label"] == budget and r["method"] in (a, b):
            by.setdefault(r["split_dir"], {})[r["method"]] = r
    both = [v for v in by.values() if a in v and b in v]
    common = [v for v in both if v[a]["solved"] and v[b]["solved"]]
    return {"pair": "{} vs {}".format(a, b), "budget": budget, "instances": len(both),
            "only_{}".format(a): sum(1 for v in both if v[a]["solved"] and not v[b]["solved"]),
            "only_{}".format(b): sum(1 for v in both if v[b]["solved"] and not v[a]["solved"]),
            "both": len(common),
            "makespan_{}".format(a): _mean([v[a]["makespan"] for v in common]),
            "makespan_{}".format(b): _mean([v[b]["makespan"] for v in common]),
            "first_s_{}".format(a): _mean([v[a]["first_valid_s"] for v in common]),
            "first_s_{}".format(b): _mean([v[b]["first_valid_s"] for v in common])}


def write_summary(tag, rows):
    summ, best = summarize(rows)
    labels = sorted({r["budget_label"] for r in rows})
    methods = sorted({r["method"] for r in rows})
    ours = [m for m in methods if m.startswith("TiFLAX")]
    out_dir = results_dir("summary")
    pairs = [paired(rows, m, t, b) for b in labels for t in ours for m in methods if m not in ours]
    with open(os.path.join(out_dir, tag + ".json"), "w") as f:
        json.dump({"summary": summ, "paired": pairs, "best_known": best}, f, indent=1)
    cols = ["method", "budget", "solved", "instances", "coverage", "first_s", "wpt_s", "makespan", "gap", "kept",
            "calls", "timeouts", "proofs", "invalid_on_full"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for s in summ:
        lines.append("| " + " | ".join(str(s[c]) for c in cols) + " |")
    with open(os.path.join(out_dir, tag + ".md"), "w") as f:
        f.write("\n".join(lines) + "\n\n")
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    return summ


plt = None   # matplotlib.pyplot, imported by plot()


# reference categorical palette (light), validated: 3 slots all-pairs, 5 slots adjacent
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
GREEN, NEUTRAL = "#008300", "#8a8984"
VIOLET = "#4a3aa7"
PLOT_METHODS = [  # key, label, colour, marker -- reference palette slots 1-7 in order (adjacent pairs validated)
    ("V1", "T-iFLAX V1", BLUE, "o"),
    ("V4.2", "T-iFLAX V4.2", ORANGE, "*"),
    ("V4", "T-iFLAX V4", "#4a3aa7", "p"),
    ("V4.1", "T-iFLAX V4.1", "#e5a07f", "h"),
    ("V3", "T-iFLAX V3", "#c3c2b7", "P"),
    ("V3U", "V3, uniform scores", "#8a8984", "X"),
    ("V2.1", "T-iFLAX V2.1", AQUA, "P"),
    ("B1", "B1 full POPF", YELLOW, "D"),
    ("B2", "B2 FLAX", MAGENTA, "^"),
    ("B3", "B3 iFLAX", GREEN, "v"),
    ("V2", "T-iFLAX V2", "#a9a8a0", "s"),
    ("V3.1", "T-iFLAX V3.1", "#7a5c2e", "d"),
]


def by_problem(rows):
    by = collections.defaultdict(dict)
    for r in rows:
        by[r["instance"]][r["method_key"]] = r
    budget = rows[0]["budget_s"] if rows else 30.0
    return by, budget


def short(name):
    return name.replace("stress_", "").replace("_medium", "").replace("_large", "").replace("_split", "") \
        .replace("_counters", "").replace("_many_orders", "").replace("_xl", "")


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def per_problem(by, out_dir):
    solved_any = [k for k in sorted(by) if any(by[k][m]["solved"] for m, *_ in PLOT_METHODS if m in by[k])]
    best = {}
    for k in solved_any:
        cands = [by[k][m]["makespan"] for m, *_ in PLOT_METHODS if m in by[k] and by[k][m]["solved"]]
        cert = by[k]["B1"].get("certificate_makespan")
        best[k] = min(cands + ([cert] if cert else []))
    # dots overlap: at most 3 series (blue / orange validated all-pairs + neutral grey); V2 / V2.1 are in the CSV
    series = [("V4.2", "T-iFLAX V4.2 (final)", ORANGE, "*", -0.22), ("V1", "T-iFLAX V1", BLUE, "o", 0.0),
              ("BASE", "Baselines B1/B2/B3 (grey; label = which solved it)", NEUTRAL, "D", 0.22)]
    shown = ("V4.2", "V1", "B1", "B2", "B3")
    solved_any = [k for k in solved_any if any(by[k][m]["solved"] for m in shown if m in by[k])]
    xs = [by[k][m]["makespan"] / best[k] for k in solved_any for m in shown if m in by[k] and by[k][m]["solved"]]
    xmax = max(xs) if xs else 2.0
    lane = xmax + 0.25                      # "not solved" lane, right of the data
    fig, ax = plt.subplots(figsize=(9.5, 0.42 * len(solved_any) + 1.9), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    style(ax)
    for i, k in enumerate(solved_any):
        y = len(solved_any) - 1 - i
        if i % 2 == 0:
            ax.axhspan(y - 0.5, y + 0.5, color="#f1f0ec", zorder=0, linewidth=0)
        for key, lab, col, mk, dy in series:
            if key == "BASE":
                sol = [m for m in ("B1", "B2", "B3") if by[k][m]["solved"]]
                ms = min((by[k][m]["makespan"] for m in sol), default=None)
                tag = "+".join(sol)
            else:
                ms = by[k][key]["makespan"] if by[k][key]["solved"] else None
                tag = None
            if ms is not None:
                ax.scatter(ms / best[k], y + dy, s=95 if mk == "*" else 46, color=col, marker=mk, edgecolor=SURFACE,
                           linewidth=1.0, zorder=3)
                if tag:
                    ax.annotate(tag, (ms / best[k], y + dy), xytext=(7, -3), textcoords="offset points",
                                fontsize=7.5, color=INK2)
            else:
                ax.scatter(lane, y + dy, s=70 if mk == "*" else 40, facecolor="none", edgecolor=col, marker=mk,
                           linewidth=1.2, zorder=3)
    ax.axvline(lane - 0.12, color=GRID, linewidth=1.2)
    ax.set_yticks(range(len(solved_any)))
    ax.set_yticklabels([short(k) for k in reversed(solved_any)], fontsize=9, color=INK)
    ticks = [t for t in (1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0) if t <= xmax + 0.05]
    ax.set_xticks(ticks + [lane])
    ax.set_xticklabels(["{:g}×".format(t) for t in ticks] + ["not\nsolved"], color=INK2)
    ax.set_xlim(0.95, lane + 0.12)
    ax.set_ylim(-0.6, len(solved_any) - 0.4)
    ax.set_xlabel("makespan ÷ best known makespan for that problem (1× = best; lower is better)", color=INK2, fontsize=9)
    n_none = len(by) - len(solved_any)
    ax.set_title("Schedule quality per problem (V4.2, V1, baselines): {} solved by at least one, {} by none (not shown)"
                 .format(len(solved_any), n_none), color=INK, fontsize=10, loc="left")
    handles = [plt.Line2D([], [], linestyle="", marker=mk, color=col, markeredgecolor=SURFACE, markersize=7, label=lab)
               for _, lab, col, mk, _ in series]
    handles.append(plt.Line2D([], [], linestyle="", marker="o", markerfacecolor="none", markeredgecolor=INK2,
                              markersize=7, label="hollow = no valid plan within the budget"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.9 / (0.42 * len(solved_any) + 1.9)),
              ncol=2, frameon=False, fontsize=8.5, labelcolor=INK)
    fig.tight_layout()
    p = out_dir + "_per_problem_makespan.png"
    fig.savefig(p, facecolor=SURFACE)
    plt.close(fig)
    return p, best


def coverage(by, budget, out_dir):
    fig, ax = plt.subplots(figsize=(8.0, 4.4), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    style(ax)
    n = len(by)
    ends = []
    for key, lab, col, mk in PLOT_METHODS:
        ts = sorted(by[k][key]["first_valid_s"] for k in by if key in by[k] and by[k][key]["solved"])
        x = [0.0] + [t for t in ts for _ in (0, 1)] + [budget]
        y = [0] + [c for i in range(len(ts)) for c in (i, i + 1)] + [len(ts)]
        ax.plot(x, y, color=col, linewidth=2, drawstyle="default", solid_joinstyle="miter")
        ax.scatter(ts, range(1, len(ts) + 1), color=col, marker=mk, s=22, edgecolor=SURFACE, linewidth=1, zorder=3)
        ends.append((len(ts), lab, col))
    # legend outside the plot, ordered by final coverage (labels carry the count)
    order = sorted(ends, key=lambda e: -e[0])
    handles = [plt.Line2D([], [], color=col, linewidth=2, marker=dict((l, m) for _, l, _, m in PLOT_METHODS)[lab],
                          markeredgecolor=SURFACE, label="{}  {}/{}".format(lab, cnt, n)) for cnt, lab, col in order]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=8.5,
              labelcolor=INK)
    ax.set_xlim(0, budget)
    ax.set_ylim(0, max(c for c, *_ in ends) + 1.5)
    ax.set_xlabel("time since start (s)", color=INK2, fontsize=9)
    ax.set_ylabel("problems with a valid plan", color=INK2, fontsize=9)
    ax.set_title("Coverage over time — {} test problems, {:g} s budget".format(n, budget), color=INK, fontsize=10.5,
                 loc="left")
    fig.subplots_adjust(right=0.72)
    p = out_dir + "_coverage_vs_time.png"
    fig.savefig(p, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return p


def table(by, best, out_dir):
    p = out_dir + "_per_problem.csv"
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["problem", "objects", "certificate_makespan", "best_known"] +
                   ["{}_{}".format(m, x) for m, *_ in PLOT_METHODS for x in ("makespan", "first_s", "objects_kept")])
        for k in sorted(by):
            r0 = by[k]["B1"]
            row = [k, r0["n_total"], r0.get("certificate_makespan"), best.get(k)]
            for m, *_ in PLOT_METHODS:
                r = by[k].get(m, {})
                row += [r.get("makespan"), round(r["first_valid_s"], 2) if r.get("first_valid_s") else None,
                        r.get("n_objects")]
            w.writerow(row)
    return p


def plot(tag, rows):
    """Per-problem plots + CSV for one run tag -> results/summary/<tag>_*.png / .csv."""
    import matplotlib
    matplotlib.use("Agg")
    global plt
    import matplotlib.pyplot as plt
    out = os.path.join(results_dir("summary"), tag)
    by, budget = by_problem(rows)
    p1, best = per_problem(by, out)
    for p in (p1, coverage(by, budget, out), table(by, best, out)):
        print(p)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="v2_test")
    ap.add_argument("--tiers", nargs="*", help="instance-name prefixes, e.g. stress_t1 stress_t4")
    ap.add_argument("--limit", type=int, default=0, help="instances per tier (0 = all)")
    ap.add_argument("--methods", default="B1,V4.2", help="comma list of " + ",".join(METHODS))
    ap.add_argument("--budgets", default="30")
    ap.add_argument("--unlimited", type=float, default=0, help="single long cap in seconds instead of --budgets")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=2, help="instances run concurrently")
    ap.add_argument("--max_load", type=float, default=0, help="wait until the 1-min load average is below this "
                    "before each run (0 = no gate); the load at start is recorded per row")
    ap.add_argument("--sequential_recovery", action="store_true")
    ap.add_argument("--suffix", default="", help="appended to the run tag, e.g. _rerun (keeps earlier rows apart)")
    ap.add_argument("--plot", action="store_true", help="also write per-problem plots (needs B1 in the run)")
    ap.add_argument("--instance", metavar="DIR", help="solve one instance folder with each method, print the plan")
    ap.add_argument("--report", metavar="TAG", help="only rebuild summary + plots of a finished run, e.g. stress_30s_seed0")
    args = ap.parse_args(argv)
    if args.instance:
        for m in args.methods.split(","):
            task = Task.load(args.instance)
            b = args.unlimited or float(args.budgets.split(",")[0])
            r = build(m, task.spec, args.seed).solve(task, b)
            print("{}: solved={} makespan={} first_valid_s={} objects={}/{} branch={}".format(
                m, r.solved, r.makespan, r.first_valid_s, r.n_objects, r.n_total, r.branch))
            print(r.plan_text or "", flush=True)
        return
    if args.report:
        rows = load_rows(args.report)
        write_summary(args.report, rows)
        plot(args.report, rows)
        return
    global MAX_LOAD
    MAX_LOAD = args.max_load or None

    dirs = split_dirs(args.split)
    if args.tiers:
        dirs = [d for d in dirs if any(os.path.basename(d).startswith(t) for t in args.tiers)]
    if args.limit:
        by_tier = {}
        for d in dirs:
            by_tier.setdefault(os.path.basename(d).rsplit("_", 1)[0], []).append(d)
        dirs = [d for ds in by_tier.values() for d in ds[: args.limit]]
    if args.unlimited:
        budgets = [(args.unlimited, "unlimited<={:g}s".format(args.unlimited))]
    else:
        budgets = [(float(b), "{:g}s".format(float(b))) for b in args.budgets.split(",")]
    overrides = {"recovery_mode": "sequential"} if args.sequential_recovery else {}
    methods = args.methods.split(",")
    for m in methods:
        if m not in METHODS:
            raise SystemExit("unknown method {}; one of {}".format(m, ",".join(METHODS)))
    for b, label in budgets:
        tag = "{}_{}_seed{}{}".format(args.split, label.replace("unlimited<=", "unlimited"), args.seed, args.suffix)
        rows = load_rows(tag)
        done = {(r["method_key"], r["split_dir"]) for r in rows}
        jobs = [(m, d) for d in dirs for m in methods if (m, os.path.relpath(d, PDDL)) not in done]
        print("{}: {} runs to do ({} done)".format(tag, len(jobs), len(done)), flush=True)
        with concurrent.futures.ThreadPoolExecutor(args.jobs) as pool:
            futs = {pool.submit(run_one, m, d, b, args.seed, overrides, label): (m, d) for m, d in jobs}
            for fut in concurrent.futures.as_completed(futs):
                m, d = futs[fut]
                try:
                    r = fut.result()
                except Exception as e:
                    print("  ! {} {} {}: {!r}".format(m, os.path.basename(d), label, e), flush=True)
                    continue
                r["method_key"] = m
                rows.append(r)
                with open(rows_file(m, tag), "a") as f:
                    f.write(json.dumps(r) + "\n")
                print("  {:<5} {:<28} {:<14} solved={!s:<5} first={:<7} ms={!s:<9} kept={}/{} calls={} branch={}".format(
                    m, os.path.basename(d), label, r["solved"],
                    "{:.2f}".format(r["first_valid_s"]) if r["first_valid_s"] is not None else "-",
                    r["makespan"], r["n_objects"], r["n_total"], r["planner_calls"], r["branch"]), flush=True)
        for s in write_summary(tag, rows):
            print(s)
        if args.plot:
            plot(tag, rows)


if __name__ == "__main__":
    ensure_hash_seed()
    main()
