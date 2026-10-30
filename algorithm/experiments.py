"""Quick experiments behind the design decisions -> results/diagnostics/. See algorithm/README.md.

    python -m algorithm.experiments oracle|first_selection|agent_seeds|work_seeds|guided_seeds|cp_select|
                                    cert_scorer train|test|v5_ranked_seeds|capacity_oracle
"""

"""This file is UN-VERIFIED"""

import argparse
import collections
import concurrent.futures
import json
import os
import random
import statistics
import time

import numpy as np
import torch
import torch.nn as nn

from pddl_generator import TemporalPlan
from . import PDDL, results_dir, split_dirs
from .gnn import Scorer
from .grounding import INF, GroundTask, Task, agent_type, comp_t
from .planning import Budget, Workspace, attempt, ensure_hash_seed
from .v1 import model_prefix, tightened_task

OUT = results_dir("diagnostics")


def popf(task, S, cap, stage, gt=None, **kw):
    """One POPF call on Pi|S, VAL on the full task, own workspace."""
    ws = Workspace()
    try:
        return attempt(task, gt, S, Budget(cap), stage, ws, check_proof=False, **kw)
    finally:
        ws.close()


def over_stress(fn, jobs=2, split="stress"):
    with concurrent.futures.ThreadPoolExecutor(jobs) as pool:
        return list(pool.map(fn, split_dirs(split)))


def save(name, rows):
    with open(os.path.join(OUT, name), "w") as f:
        json.dump(rows, f, indent=1)


def load(name):
    with open(os.path.join(OUT, name)) as f:
        return json.load(f)


def cooks_pots(t, S):
    return {"cooks": sorted(o for o in S if t.is_a(o, "cook")), "pots": sorted(o for o in S if t.is_a(o, "pot"))}


# ==================== oracle selection ===============================================================
def oracle():
    """POPF on Pi|S*, S* = COMP_T(objects of the certificate plan), 30 s. Separates selection failures
    from planner failures."""
    def run(d):
        t = Task.load(d); g = GroundTask(t)
        cert = TemporalPlan.parse(open(os.path.join(d, "certificate.plan")).read())
        S = comp_t(g, cert.objects_used() & set(t.objects))
        a = popf(t, S, 30, "oracle")
        return {"instance": os.path.basename(d), "n_total": len(t.objects), "n_oracle": len(S), "outcome": a.outcome,
                "planner_s": round(a.planner_s, 2), "makespan": a.makespan, "cert_makespan": cert.makespan,
                "cooks_in_cert": len({o for o in cert.objects_used() if t.is_a(o, "cook")}),
                "pots_in_cert": len({o for o in cert.objects_used() if t.is_a(o, "pot")}), "orders": len(t.goals)}
    rows = over_stress(run)
    save("oracle_selection.json", rows)
    print("oracle selection solved", sum(r["outcome"] == "valid" for r in rows), "/", len(rows))


# ==================== which first object set lets POPF find a plan? ===================================
def first_selection():
    """30 s POPF on  rp: COMP_T(objects of the temporal relaxed plan of the full task) (no learning)
                and  v1: V1's first expansion set (reward >= q_max)."""
    def run(d):
        t = Task.load(d); g = GroundTask(t)
        s = Scorer.load(model_prefix("v1", 0), t.spec); sc = s.scores(t, g)
        sets = {"rp": comp_t(g, g.relaxed_plan_objects(g.full)),
                "v1": comp_t(g, {o for o, p in sc.items() if s.reward(p) >= 0.81 * s.reward_scale})}
        out = {"instance": os.path.basename(d), "n_total": len(t.objects)}
        for k, S in sets.items():
            a = popf(t, S, 30, k)
            out[k] = dict({"n": len(S), "outcome": a.outcome, "s": round(a.planner_s, 2), "makespan": a.makespan},
                          **cooks_pots(t, S))
        return out
    rows = over_stress(run)
    save("first_selection.json", rows)
    for k in ("rp", "v1"):
        print(k, "solved", sum(r[k]["outcome"] == "valid" for r in rows), "/", len(rows))


# ==================== per-agent seeds (V3, V4) ========================================================
def _plan_seeds(t, seeds, cap=15.0):
    """POPF on every seed in parallel; seeds = [(S, info)] -> [(rank, info, attempt)] of the valid ones."""
    with concurrent.futures.ThreadPoolExecutor(len(seeds) or 1) as pool:
        res = list(pool.map(lambda x: (x, popf(t, x[0], cap, "seed")), seeds))
    return [(i, x[1], a) for i, (x, a) in enumerate(res) if a.outcome == "valid"]


def agent_seeds():
    """For each agent c (cooks here), S_c = COMP_T(objects of the temporal relaxed plan of Pi without the
    other agents). The 4 seeds with the lowest TRPG bound are planned in parallel, POPF 15 s each."""
    def run(d):
        t = Task.load(d); g = GroundTask(t)
        agents = t.of_type(agent_type(t))
        seeds = []
        for c in agents:
            r = g.trpg(set(t.objects) - (set(agents) - {c}))
            if not r.solvable_relaxed:
                continue
            S = comp_t(g, {o for i in g.relaxed_plan(r) for o in g.actions[i].objects} | {c})
            seeds.append((g.trpg(S).lb, len(S), c, S))
        seeds = sorted(seeds, key=lambda x: (x[0], x[1]))[:4]
        t0 = time.time()
        ok = _plan_seeds(t, [(x[3], x) for x in seeds])
        return {"instance": os.path.basename(d), "n_total": len(t.objects), "n_seeds": len(seeds),
                "seed_sizes": [x[1] for x in seeds], "solved": bool(ok), "wall": round(time.time() - t0, 1),
                "winners": [{"rank": i, "agent": x[2], "n": x[1], "s": round(a.planner_s, 2), "makespan": a.makespan}
                            for i, x, a in ok]}
    rows = over_stress(run)
    save("agent_seeds.json", rows)
    print("earliest-time seeds solved", sum(r["solved"] for r in rows), "/", len(rows))


def work_seeds():
    """As agent_seeds, but each seed comes from the WORK-optimal relaxed plan (durations as costs, h_add
    supporters); seeds rejected by a TRPG proof are dropped. Also reports the union with agent_seeds."""
    def run(d):
        t = Task.load(d); g = GroundTask(t)
        agents = t.of_type(agent_type(t))
        cands = {}
        for c in agents:
            keep = set(t.objects) - (set(agents) - {c})
            r = g.trpg(keep)
            if not r.solvable_relaxed:
                continue
            plan = g.cost_relaxed_plan(keep, windows_from=r)
            if not plan:
                continue
            S = frozenset(comp_t(g, {o for i in plan for o in g.actions[i].objects} | {c}))
            if g.proof(S) is None and S not in cands:
                cands[S] = (g.trpg(S).lb, len(S), c)
        seeds = sorted(cands.items(), key=lambda x: x[1])[:4]
        t0 = time.time()
        ok = _plan_seeds(t, [(S, S) for S, _ in seeds])
        return {"instance": os.path.basename(d), "solved": bool(ok), "n_seeds": len(seeds),
                "sizes": [len(S) for S, _ in seeds], "wall": round(time.time() - t0, 1),
                "winners": [{"rank": i, "n": len(S), "s": round(a.planner_s, 2), "makespan": a.makespan}
                            for i, S, a in ok]}
    rows = over_stress(run)
    save("work_seeds.json", rows)
    a = {r["instance"]: r for r in load("agent_seeds.json")}
    print("work-optimal seeds solved", sum(r["solved"] for r in rows), "/ 60 | earliest-time seeds",
          sum(x["solved"] for x in a.values()), "| union", sum(r["solved"] or a[r["instance"]]["solved"] for r in rows))


def guided_seeds(tol=0.25, cap=15.0, rounds=1, par=4, jobs=2):
    """Per agent, two seeds: earliest relaxed plan (V3) and the score-guided relaxed plan (tolerance
    tol * LB, score = V1's learned object score). Ranked by TRPG bound, then mean learned score; `par`
    seeds per round in parallel, up to `rounds` rounds."""
    def run(d):
        t = Task.load(d); g = GroundTask(t)
        sc = Scorer.load(model_prefix("v1", 0), t.spec).scores(t, g)
        score = lambda o: sc.get(o, [0.0])[0]  # noqa: E731
        agents = t.of_type(agent_type(t))
        cands = {}
        for c in agents:
            r = g.trpg(set(t.objects) - (set(agents) - {c}))
            if not r.solvable_relaxed:
                continue
            for kind, plan in (("earliest", g.relaxed_plan(r)), ("guided", g.guided_relaxed_plan(r, score, tol * r.lb))):
                S = frozenset(comp_t(g, {o for i in plan for o in g.actions[i].objects} | {c}))
                if S not in cands:
                    cands[S] = (g.trpg(S).lb, -sum(score(o) for o in S) / len(S), len(S), c, kind)
        seeds = sorted(cands.items(), key=lambda x: x[1])
        res, t0 = [], time.time()
        for rnd in range(rounds):
            batch = seeds[par * rnd: par * rnd + par]
            if not batch:
                break
            with concurrent.futures.ThreadPoolExecutor(len(batch)) as pool:
                out = list(pool.map(lambda x: (x, popf(t, x[0], cap, "seed")), batch))
            res += out
            if any(a.outcome == "valid" for _, a in out):
                break
        ok = [(i, x, a) for i, (x, a) in enumerate(res) if a.outcome == "valid"]
        return {"instance": os.path.basename(d), "solved": bool(ok), "n_seeds": len(seeds), "tried": len(res),
                "wall": round(time.time() - t0, 1),
                "winners": [{"rank": i, "kind": x[1][4], "n": len(x[0]), "s": round(a.planner_s, 2),
                             "makespan": a.makespan} for i, x, a in ok]}
    rows = over_stress(run, jobs)
    save("guided_seeds_tol%g_cap%g_r%d_p%d.json" % (tol, cap, rounds, par), rows)
    print("tol", tol, "solved", sum(r["solved"] for r in rows), "/", len(rows))


# ==================== idea 1: object selection from a CP-SAT scheduling relaxation ======================
def _kitchen_data(t):
    facts = t.init
    faces = collections.defaultdict(list)
    for f in facts:
        if f[0] == "faces":
            faces[f[2]].append(f[1])
    start = {f[1]: f[2] for f in facts if f[0] == "at"}
    trav = {(k[1], k[2]): v for k, v in t.fluents.items() if k[0] == "travel"}
    provides = collections.defaultdict(list)
    for f in facts:
        if f[0] == "provides":
            provides[f[2]].append(f[1])
    nxt = {f[1]: f[2] for f in facts if f[0] == "next"}
    zero = [f[1] for f in facts if f[0] == "is-zero"][0]
    val, c, i = {}, zero, 0
    while c is not None:
        val[c] = i; c = nxt.get(c); i += 1
    recipe = {f[1]: (val[f[2]], val[f[3]]) for f in facts if f[0] == "recipe-of"}
    order_recipe = {f[1]: f[2] for f in facts if f[0] == "order-recipe"}
    steps = {k[1]: v for k, v in t.fluents.items() if k[0] == "cook-steps"}

    def d(a, b):            # station-to-station (or pose-to-station) travel, min over access poses
        A = faces[a] if a in faces else [a]
        B = faces[b] if b in faces else [b]
        return min((trav.get((x, y), 0 if x == y else INF) for x in A for y in B), default=INF)
    return dict(faces=faces, start=start, provides=provides, recipe=recipe, order_recipe=order_recipe,
                steps=steps, d=d)


def cp_selection(t, time_limit=5.0, mode="makespan"):
    """Overcooked-specific feasibility check (a domain-independent extraction is future work).
    Per order o (recipe r: n_onion, n_tomato), choose a cook c and a pot p. Stations are the ones
    nearest to p (a compact cluster): onion / tomato / dish dispenser, serving window.
    Intervals (travel = shortest pose-to-pose time, interact = 1):
      fill_o  (cook c): ingredient round trips pot <-> dispenser + start-cooking
      cook_o  (pot p) : cook-steps(r) + 1, right after fill
      serve_o (cook c): dish round trip pot <-> dish dispenser (may overlap the cooking), pick soup,
                        walk to the window, deliver; must end by the order's deadline
      pot p is busy from fill start to the soup pick; cook c's intervals do not overlap.
    mode: makespan = minimise makespan; min_resources = fewest distinct cooks + pots meeting all
    deadlines (makespan as tie-break). -> (selected objects or None, info)."""
    from ortools.sat.python import cp_model
    D = _kitchen_data(t)
    d = D["d"]
    cooks = [c for c in t.of_type("cook") if c in D["start"]]
    pots = [p for p in t.of_type("pot") if D["faces"].get(p)]
    disp = {i: [s for s in D["provides"][i] if D["faces"].get(s)] for i in ("onion", "tomato", "dish")}
    servs = [s for s in t.of_type("serving") if D["faces"].get(s)]
    near = lambda p, cands: min(cands, key=lambda s: d(p, s) + d(s, p)) if cands else None  # noqa: E731
    cluster = {p: dict(onion=near(p, disp["onion"]), tomato=near(p, disp["tomato"]), dish=near(p, disp["dish"]),
                       serve=min(servs, key=lambda s: d(p, s))) for p in pots}
    orders = [g[1] for g in t.goals]
    H = int(max(t.deadlines.values(), default=1000)) + 200
    m = cp_model.CpModel()
    cook_iv, pot_iv = collections.defaultdict(list), collections.defaultdict(list)
    choice, ends = {}, []
    for o in orders:
        r = D["order_recipe"][o]
        no, nt = D["recipe"][r]
        cookt = int(D["steps"][r]) + 1
        dl = t.deadlines.get(("served", o))
        lits = []
        for c in cooks:
            for p in pots:
                cl = cluster[p]
                if (no and cl["onion"] is None) or (nt and cl["tomato"] is None) or cl["dish"] is None:
                    continue
                need = [cl["dish"], cl["serve"]] + ([cl["onion"]] if no else []) + ([cl["tomato"]] if nt else [])
                first = cl["onion"] if no else cl["tomato"]
                if any(d(p, s_) == INF or d(s_, p) == INF for s_ in need) or d(D["start"][c], first) == INF:
                    continue                                     # unreachable for this cook / pot (split rooms)
                trip = lambda s: int(d(p, s) + d(s, p)) + 2  # noqa: E731  take + add
                fill = no * trip(cl["onion"]) + (nt * trip(cl["tomato"]) if nt else 0) + 1
                release = int(d(D["start"][c], first))
                dish = int(d(p, cl["dish"]) + d(cl["dish"], p)) + 1
                serve = dish + 1 + int(d(p, cl["serve"])) + 1
                x = m.NewBoolVar("x_%s_%s_%s" % (o, c, p))
                fs = m.NewIntVar(release, H, ""); fe = m.NewIntVar(0, H, "")
                ce = m.NewIntVar(0, H, ""); ss = m.NewIntVar(0, H, ""); se = m.NewIntVar(0, H, "")
                f_iv = m.NewOptionalIntervalVar(fs, fill, fe, x, "")
                s_iv = m.NewOptionalIntervalVar(ss, serve, se, x, "")
                m.Add(ce == fe + cookt).OnlyEnforceIf(x)
                m.Add(ss >= ce - dish).OnlyEnforceIf(x)             # dish fetched during cooking
                pot_end = m.NewIntVar(0, H, "")
                m.Add(pot_end == ss + dish + 1).OnlyEnforceIf(x)
                p_iv = m.NewOptionalIntervalVar(fs, m.NewIntVar(0, H, ""), pot_end, x, "")
                if dl is not None:
                    m.Add(se <= int(dl)).OnlyEnforceIf(x)
                cook_iv[c] += [f_iv, s_iv]; pot_iv[p].append(p_iv)
                e = m.NewIntVar(0, H, ""); m.Add(e == se).OnlyEnforceIf(x); m.Add(e == 0).OnlyEnforceIf(x.Not())
                ends.append(e)
                choice[(o, c, p)] = x
                lits.append(x)
        if not lits:
            return None, "no config for %s" % o
        m.AddExactlyOne(lits)
    for c in cooks:
        m.AddNoOverlap(cook_iv[c])
    for p in pots:
        m.AddNoOverlap(pot_iv[p])
    ms = m.NewIntVar(0, H, "ms"); m.AddMaxEquality(ms, ends)
    if mode == "min_resources":
        used = []
        for r_ in cooks + pots:
            u = m.NewBoolVar("u_" + r_)
            for (o_, c_, p_), x_ in choice.items():
                if r_ in (c_, p_):
                    m.AddImplication(x_, u)
            used.append(u)
        m.Minimize(10 * H * sum(used) + ms)
    else:
        m.Minimize(ms)
    s = cp_model.CpSolver(); s.parameters.max_time_in_seconds = time_limit; s.parameters.num_workers = 2
    st = s.Solve(m)
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None, "cp " + s.StatusName(st)
    S = set(t.goal_objects)
    for (o, c, p), x in choice.items():
        if s.Value(x):
            cl = cluster[p]
            S |= {c, p, cl["dish"], cl["serve"]} | ({cl["onion"]} if cl["onion"] else set()) | \
                 ({cl["tomato"]} if D["recipe"][D["order_recipe"][o]][1] and cl["tomato"] else set())
    return S, "cp ms %d (%s, %.2fs)" % (s.Value(ms), s.StatusName(st), s.WallTime())


def cp_select(mode="makespan"):
    """Selected objects = goal objects + chosen cooks / pots / stations, COMP_T; then POPF 15 s."""
    def run(d):
        t = Task.load(d); g = GroundTask(t)
        S, info = cp_selection(t, mode=mode)
        out = {"instance": os.path.basename(d), "n_total": len(t.objects), "info": info}
        if S is None:
            out.update(outcome="no_selection")
            return out
        S = comp_t(g, S)
        a = popf(t, S, 15, "cp")
        out.update(n=len(S), outcome=a.outcome, s=round(a.planner_s, 2), makespan=a.makespan, **cooks_pots(t, S))
        return out
    rows = over_stress(run)
    save("cp_select_%s.json" % mode, rows)
    print("cp selection (%s) solved" % mode, sum(r["outcome"] == "valid" for r in rows), "/", len(rows))


# ==================== idea 2: learn minimal sufficient object sets ======================================
CERT_MODEL = os.path.join(OUT, "cert_scorer_seed0")


def _cert_examples(split, scorer):
    """label(o) = 1 if o appears in the kitchen's certificate plan (the generator's minimal core:
    1-2 cooks, 1 pot, nearest stations) or in the goal, else 0."""
    out = []
    for d in split_dirs(split):
        t = Task.load(d); g = GroundTask(t)
        cert = TemporalPlan.parse(open(os.path.join(d, "certificate.plan")).read())
        pos = (cert.objects_used() | set(t.goal_objects)) & set(t.objects)
        gr = scorer.graph(t, g)
        out.append((gr, np.array([[1.0 if o in pos else 0.0] for o in gr.objects], dtype=np.float32)))
    return out


def cert_scorer_train():
    """Same graph / GNN as V1 (1 head, temporal features), offline BCE (FLAX-style), class-balanced."""
    torch.manual_seed(0); random.seed(0)
    sc = Scorer(Task.load(split_dirs("train")[0]).spec, heads=1, temporal_features=True, hidden=16, steps=3)
    tr, va = _cert_examples("train", sc), _cert_examples("val", sc)
    pos = sum(y.sum() for _, y in tr); neg = sum(len(y) - y.sum() for _, y in tr)
    pw = torch.tensor([neg / pos])
    opt = torch.optim.Adam(sc.model.parameters(), lr=1e-3)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    for ep in range(300):
        random.shuffle(tr)
        sc.model.train()
        for g, y in tr:
            loss = lossf(sc.logits(g), torch.from_numpy(y))
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 50 == 0 or ep == 299:
            sc.model.eval()
            with torch.no_grad():
                pr = [(torch.sigmoid(sc.logits(g)).numpy() >= 0.5, y >= 0.5) for g, y in va]
            acc = np.mean([(p == y).mean() for p, y in pr])
            rec = np.mean([(p & y).sum() / y.sum() for p, y in pr])
            prec = np.mean([(p & y).sum() / max(1, p.sum()) for p, y in pr])
            print("epoch %3d  val acc %.3f  recall %.3f  precision %.3f" % (ep, acc, rec, prec), flush=True)
    sc.save(CERT_MODEL, extra={"labels": "certificate membership", "pos_weight": float(pw)})


def cert_scorer_test(qs=(0.5, 0.3)):
    """sel@q: COMP_T(goal objects + {o : p(o) >= q}), POPF 15 s (diagnostic only, no recovery)."""
    def run(d):
        t = Task.load(d); g = GroundTask(t)
        sc = Scorer.load(CERT_MODEL, t.spec).scores(t, g)
        out = {"instance": os.path.basename(d), "n_total": len(t.objects)}
        for q in qs:
            S = comp_t(g, {o for o, p in sc.items() if p[0] >= q})
            a = popf(t, S, 15, "cert")
            out["q%g" % q] = dict({"n": len(S), "outcome": a.outcome, "s": round(a.planner_s, 2),
                                   "makespan": a.makespan}, **cooks_pots(t, S))
        return out
    rows = over_stress(run)
    save("cert_scorer_test.json", rows)
    a = {r["instance"]: r for r in load("agent_seeds.json")}
    for q in qs:
        k = "q%g" % q
        print("%s: solved %d/60, mean size %.0f%%, oracle 50, V3-seeds %d, union(sel, V3-seeds) %d" % (
            k, sum(r[k]["outcome"] == "valid" for r in rows), 100 * np.mean([r[k]["n"] / r["n_total"] for r in rows]),
            sum(x["solved"] for x in a.values()),
            sum(r[k]["outcome"] == "valid" or a[r["instance"]]["solved"] for r in rows)))


# ==================== V5a follow-up: GNN seeds by rank ==================================================
def v5_ranked_seeds():
    """On the problems V4.2's seeds miss (results/v5/v5a_seedtest_stress.json), GNN seeds by RANK instead
    of a threshold, with V5a's model (results/v5/v5a_seed0; retrain with `algorithm.v5 v5a-train`).
    G_size: top objects by r(o) = p_imp * on-time, as many as V4.2's best work-optimal seed has, then COMP_T.
    G_type: per type, the same count as in V4.2's best work-optimal seed (single cook) / best two-cook seed,
            chosen by r(o), then COMP_T. POPF 15 s each."""
    from .v5 import OUT as V5_OUT
    from .v5 import TimeScorer, latest_deadline, v42_seeds
    miss = [r["instance"] for r in json.load(open(os.path.join(V5_OUT, "v5a_seedtest_stress.json")))
            if not r["V4.2"]["solved"]]
    orc = {d["instance"]: d for d in load("oracle_selection.json")}
    sc, rows = None, []
    for inst in miss:
        t = Task.load(os.path.join(PDDL, "stress", inst)); g = GroundTask(t)
        sc = sc or TimeScorer.load(os.path.join(V5_OUT, "v5a_seed0"), t.spec)
        G, pr = sc.predict(t, g); D = latest_deadline(t) / G.lb
        r = {o: p * TimeScorer.on_time(a, b, D) for o, (p, a, b) in pr.items()}
        seeds = v42_seeds(g)
        single = [x for x in seeds if "+" not in x[1]][0][2]
        pair = [x for x in seeds if "+" in x[1]]
        by_r = sorted(t.objects, key=lambda o: -r[o])

        def type_of(o):
            return min((ty for ty in t.spec.types if t.is_a(o, ty)), key=lambda ty: len(t.of_type(ty)))

        def typed(ref):
            quota = collections.Counter(type_of(o) for o in ref)
            out, used = set(t.goal_objects), collections.Counter()
            for o in by_r:
                ty = type_of(o)
                if used[ty] < quota[ty]:
                    out.add(o); used[ty] += 1
            return frozenset(comp_t(g, out))
        cand = {"G_size": frozenset(comp_t(g, set(t.goal_objects) | set(by_r[:len(single)]))), "G_type1": typed(single)}
        if pair:
            cand["G_type2"] = typed(pair[0][2])
        with concurrent.futures.ThreadPoolExecutor(len(cand)) as ex:
            res = list(ex.map(lambda it: (it[0], len(it[1]), popf(t, it[1], 15, it[0], gt=g)), cand.items()))
        row = {"instance": inst, "oracle": orc[inst]["outcome"], "results": {k: (n, a.outcome, a.makespan) for k, n, a in res}}
        rows.append(row); print(row, flush=True)
    save("v5_ranked_seeds.json", rows)
    print("solved by a ranked GNN seed:", sum(any(v[1] == "valid" for v in r["results"].values()) for r in rows),
          "of", len(rows))


# ==================== headroom of capacity choice (V7) ==================================================
def capacity_oracle(split="stress"):
    """After the first plan (seeds, TRPG-bound order, one POPF at a time), try EVERY extra machine /
    agent (one-object additions), POPF with deadlines tightened to C - 5 %, 5 s each. Compares the best
    of all (oracle) with the V4.2-style top 3; stores the per-candidate outcomes (labels for a learned
    capacity chooser)."""
    from .v6 import CAPS, candidates, machines

    def first_plan(task, gt, budget_s=30.0):
        cands = sorted(candidates(gt), key=lambda x: (x[0], len(x[2]), x[1])); t0 = time.monotonic()
        for k, x in enumerate(cands):
            left = budget_s - (time.monotonic() - t0)
            if left < 0.5:
                return None
            a = popf(task, x[2], min(CAPS[min(k, 4)], left), "seed", gt=gt)
            if a.outcome == "valid":
                return x[2], a.makespan
        return None

    def run(d):
        task = Task.load(d); gt = GroundTask(task)
        fp = first_plan(task, gt)
        if fp is None:
            return {"instance": os.path.basename(d), "first": None}
        S, C = fp
        ms = sorted(machines(task) - set(S)); ag = agent_type(task); early = gt.earliest_use()
        agents = sorted((o for o in ms if task.is_a(o, ag)), key=lambda o: (early.get(o, 1e9), o))
        others = sorted((o for o in ms if not task.is_a(o, ag)), key=lambda o: (early.get(o, 1e9), o))
        sym = (agents[:1] + others[:2] + agents[1:] + others[2:])[:3]
        tt = tightened_task(task, C - max(1.0, 0.05 * C)); gtt = gt.with_task(tt)

        def one(o):
            S2 = frozenset(comp_t(gt, set(S) | {o}))
            ws = Workspace()
            try:
                a = attempt(task, gtt, S2, Budget(5.0), "cap", ws, plan_task=tt)
            finally:
                ws.close()
            return o, {"outcome": a.outcome, "makespan": a.makespan, "n": len(S2)}
        with concurrent.futures.ThreadPoolExecutor(6) as ex:
            res = dict(ex.map(one, ms))
        best = lambda objs: min([res[o]["makespan"] for o in objs if res[o]["outcome"] == "valid"] + [C])  # noqa: E731
        return {"instance": os.path.basename(d), "S": sorted(S), "C": C, "sym": sym, "res": res,
                "C_sym": best(sym), "C_all": best(ms), "n_cands": len(ms)}

    out = os.path.join(OUT, "capacity_oracle_%s.jsonl" % split)
    done = {json.loads(line)["instance"] for line in open(out)} if os.path.exists(out) else set()

    def go(d):
        r = run(d)
        with open(out, "a") as f:
            f.write(json.dumps(r) + "\n")
        print(r["instance"], r.get("C"), r.get("C_sym"), r.get("C_all"), r.get("n_cands"), flush=True)
    with concurrent.futures.ThreadPoolExecutor(2) as ex:
        list(ex.map(go, [d for d in split_dirs(split) if os.path.basename(d) not in done]))
    rows = [r for r in (json.loads(line) for line in open(out)) if r.get("C")]
    print("first plans %d; mean ratio to first: V4.2-style top-3 %.3f, oracle (all %.1f candidates) %.3f; "
          "improved: sym %d, oracle %d" % (
              len(rows), statistics.mean(r["C_sym"] / r["C"] for r in rows),
              statistics.mean(r["n_cands"] for r in rows), statistics.mean(r["C_all"] / r["C"] for r in rows),
              sum(r["C_sym"] < r["C"] - .5 for r in rows), sum(r["C_all"] < r["C"] - .5 for r in rows)))


def main():
    ensure_hash_seed()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["oracle", "first_selection", "agent_seeds", "work_seeds", "guided_seeds",
                                    "cp_select", "cert_scorer", "v5_ranked_seeds", "capacity_oracle"])
    ap.add_argument("sub", nargs="?", help="cert_scorer: train | test")
    ap.add_argument("--tol", type=float, default=0.25, help="guided_seeds: tolerance x LB")
    ap.add_argument("--cap", type=float, default=15.0, help="guided_seeds: POPF cap per seed")
    ap.add_argument("--rounds", type=int, default=1, help="guided_seeds: seed rounds")
    ap.add_argument("--par", type=int, default=4, help="guided_seeds: seeds per round")
    ap.add_argument("--jobs", type=int, default=2, help="guided_seeds: problems in parallel")
    ap.add_argument("--mode", default="makespan", choices=["makespan", "min_resources"], help="cp_select")
    ap.add_argument("--split", default="stress", help="capacity_oracle")
    a = ap.parse_args()
    if a.cmd == "guided_seeds":
        guided_seeds(a.tol, a.cap, a.rounds, a.par, a.jobs)
    elif a.cmd == "cp_select":
        cp_select(a.mode)
    elif a.cmd == "cert_scorer":
        {"train": cert_scorer_train, "test": cert_scorer_test}[a.sub or "test"]()
    elif a.cmd == "capacity_oracle":
        capacity_oracle(a.split)
    else:
        globals()[a.cmd]()


if __name__ == "__main__":
    main()
