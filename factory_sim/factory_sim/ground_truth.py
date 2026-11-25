"""Ground truth for benchmark instances: makespan lower bounds and CP-SAT schedules (nominal, deterministic).

CP-SAT model: processing on machines (no overlap; alternatives as optional intervals), operators (cumulative),
robot transport legs (each on exactly one robot) with sequence-dependent empty travel (one circuit per robot),
conveyors as fixed delays. It ignores finite buffers, blocking, dock queues and congestion, so its optimum is
exact only for instances without those features ("exact" in the result); otherwise it is a relaxation and its
proven bound is still a valid lower bound.
"""
from . import timing
from .core import DISPATCHER


def _legs(F, route):
    """Consecutive (from options, to options, conveyor?) pairs of a route, ending at the sinks."""
    out = []
    for i in range(len(route)):
        a, b = route[i], route[i + 1] if i + 1 < len(route) else F.sinks
        out.append((i, a, b, all(any((x, y) in F.transfers for y in b) for x in a)))   # every source has a conveyor onward
    return out


def lower_bound(F, orders):
    hop = timing.mean(F.robots["load"]) + timing.mean(F.robots["unload"])
    nr = F.robots["count"]
    lbs = {"job": 0.0, "machine": 0.0, "robot": 0.0, "operator": 0.0}
    load, ops, transport = {}, {}, 0.0
    for o in orders:
        p, a = o["product"], float(o.get("arrival", 0.0))
        lbs["job"] = max(lbs["job"], a + min(F.nominal_flow(p, r) for r in range(len(F.products[p]["routes"]))))
        best = min(range(len(F.products[p]["routes"])), key=lambda r: F.nominal_flow(p, r))
        route = F.products[p]["routes"][best]
        for opts in route:
            m = min(timing.mean(F.proc_model(s, p)) for s in opts)
            if len(opts) == 1 and all(len(rt) == len(route) for rt in F.products[p]["routes"]) and len(F.products[p]["routes"]) == 1:
                load[opts[0]] = load.get(opts[0], 0.0) + m
            pool = F.stations[opts[0]].get("operator")
            if pool and all(F.stations[s].get("operator") == pool for s in opts):
                ops[pool] = ops.get(pool, 0.0) + m
        for i, a_opts, b_opts, conv in _legs(F, route):
            if not conv:
                transport += hop + min(F.travel(F.stations[x]["node"], F.stations[y]["node"]) for x in a_opts for y in b_opts)
    first = min(float(o.get("arrival", 0.0)) for o in orders) if orders else 0.0
    groups = {r: {r} for r in load}
    for st in F.reconfigurable:
        for c in F.stations[st]["configs"].values():
            if c.get("replaces") in groups:
                groups[c["replaces"]].add(st)
    comps = []
    for r, g in groups.items():
        hit = [c for c in comps if c[0] & (g | {r})]
        roles, stations = {r}.union(*[c[1] for c in hit]), g.union(*[c[0] for c in hit])
        comps = [c for c in comps if c not in hit] + [(stations, roles)]
    lbs["machine"] = first + max((sum(load[r] for r in roles) / len(stations) for stations, roles in comps), default=0.0)
    lbs["robot"] = first + transport / nr
    lbs["operator"] = first + max((v / F.operators[k] for k, v in ops.items()), default=0.0)
    return max(lbs.values()), lbs


UNLIMITED = 1000


def exact_model(F):
    big = UNLIMITED
    finite = any(s["kind"] in ("machine", "assembly", "inspection", "kitting", "port") and (s["slots_in"] < big or 0 < s["slots_out"] < big) for s in F.stations.values())
    return not finite and F.homogeneous and F.handling == ["arm"] and F.slowdown == 0 and all(s["dock_capacity"] >= F.robots["count"] for s in F.stations.values()) and all(len(p["routes"]) == 1 for p in F.products.values())


def cpsat(F, orders, time_limit=60.0, workers=8, scale=10):
    """Solve the nominal scheduling model. Returns status, makespan, bound, exact flag and the robot / release plan."""
    from ortools.sat.python import cp_model
    S = lambda x: int(round(x * scale))  # noqa: E731
    hop = timing.mean(F.robots["load"]) + timing.mean(F.robots["unload"])
    robots = ["r{}".format(i + 1) for i in range(F.robots["count"])]
    start_node = {r: F.robots["start"][i % len(F.robots["start"])] for i, r in enumerate(robots)}
    m = cp_model.CpModel()
    H = S(sum(F.nominal_flow(o["product"]) for o in orders) + max((o.get("arrival", 0) for o in orders), default=0) + 10)
    machine, pools, legs, ends, kit, starts = {}, {}, [], [], {}, {}
    for o in orders:
        p, j = o["product"], o["id"]
        route = F.products[p]["routes"][0]
        pe, choose = [], []
        for i, opts in enumerate(route):
            st, en = m.NewIntVar(0, H, ""), m.NewIntVar(0, H, "")
            if i == 0:
                m.Add(st >= S(o.get("arrival", 0)))
                kit[j] = st
            xs = {}
            for s in opts:
                x = m.NewBoolVar("")
                d = S(timing.mean(F.proc_model(s, p)))
                iv = m.NewOptionalIntervalVar(st, d, en, x, "")
                machine.setdefault(s, []).append(iv)
                pool = F.stations[s].get("operator")
                if pool:
                    pools.setdefault(pool, []).append(iv)
                xs[s] = x
            m.AddExactlyOne(xs.values())
            pe.append((st, en))
            starts[(j, i)] = st
            choose.append(xs)
        for i, a_opts, b_opts, conv in _legs(F, route):
            a_end = pe[i][1]
            b_start = pe[i + 1][0] if i + 1 < len(route) else None
            if conv:
                if b_start is not None:
                    m.Add(b_start >= a_end + S(min(timing.mean(F.transfers[(x, y)]) for x in a_opts for y in b_opts if (x, y) in F.transfers)))
                continue
            fa, fb = sorted(a_opts, key=lambda s: s)[0], sorted(b_opts)[0]
            src = F.stations[min(a_opts, key=lambda s: 0)]["node"] if len(a_opts) == 1 else F.stations[fa]["node"]
            dst = F.stations[b_opts[0]]["node"] if len(b_opts) == 1 else F.stations[fb]["node"]
            d = S(hop + min(F.travel(F.stations[x]["node"], F.stations[y]["node"]) for x in a_opts for y in b_opts))
            ls, le = m.NewIntVar(0, H, ""), m.NewIntVar(0, H, "")
            m.Add(ls >= a_end)
            m.Add(le == ls + d)
            if b_start is not None:
                m.Add(b_start >= le)
            else:
                ends.append(le)
            pres = {r: m.NewBoolVar("") for r in robots}
            m.AddExactlyOne(pres.values())
            for r in robots:
                m.NewOptionalIntervalVar(ls, d, le, pres[r], "")
            legs.append({"job": j, "from": src, "to": dst, "start": ls, "end": le, "pres": pres, "to_station": b_opts[0] if len(b_opts) == 1 else None, "to_opts": b_opts, "step": i + 1})
    for s, ivs in machine.items():
        m.AddNoOverlap(ivs)
    for pool, ivs in pools.items():
        m.AddCumulative(ivs, [1] * len(ivs), F.operators[pool])
    for r in robots:
        arcs = []
        n = len(legs)
        for b, L in enumerate(legs):
            arcs.append((b + 1, b + 1, L["pres"][r].Not()))
            lit = m.NewBoolVar("")
            arcs.append((0, b + 1, lit))
            m.Add(L["start"] >= S(F.travel(start_node[r], L["from"]))).OnlyEnforceIf(lit)
            arcs.append((b + 1, 0, m.NewBoolVar("")))
            for a, K in enumerate(legs):
                if a != b:
                    lit2 = m.NewBoolVar("")
                    arcs.append((a + 1, b + 1, lit2))
                    m.Add(L["start"] >= K["end"] + S(F.travel(K["to"], L["from"]))).OnlyEnforceIf(lit2)
        if n:
            arcs.append((0, 0, m.NewBoolVar("")))
            m.AddCircuit(arcs)
    mk = m.NewIntVar(0, H, "makespan")
    for e in ends:
        m.Add(mk >= e)
    m.Minimize(mk)
    sol = cp_model.CpSolver()
    sol.parameters.max_time_in_seconds = time_limit
    sol.parameters.num_workers = workers
    st = sol.Solve(m)
    name = sol.StatusName(st)
    out = {"status": name, "bound": sol.BestObjectiveBound() / scale, "exact": exact_model(F)}
    if st in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        out["makespan"] = sol.Value(mk) / scale
        plan = {r: [] for r in robots}
        for L in legs:
            r = next(r for r in robots if sol.Value(L["pres"][r]))
            plan[r].append((sol.Value(L["start"]) / scale, L["job"], L["step"], L["to_opts"]))
        out["robots"] = {r: sorted(v) for r, v in plan.items()}
        out["releases"] = sorted((sol.Value(v) / scale, j) for j, v in kit.items())
        out["starts"] = {"{}:{}".format(j, k): sol.Value(v) / scale for (j, k), v in starts.items()}
    return out


class SchedulePolicy:
    """Execute a CP-SAT schedule in the simulator: releases in schedule order; each robot takes its earliest ready leg
    (schedule order) and otherwise waits at the station of its next leg."""

    def __init__(self, sched):
        self.rel = [j for _, j in sched.get("releases", [])]
        self.legs = {r: [(j, k, opts) for _, j, k, opts in v] for r, v in sched.get("robots", {}).items()}
        self.held = {}
        self.starts = {tuple(map(int, k.split(":"))): v for k, v in sched.get("starts", {}).items()}

    def __call__(self, sim, agent, legal):
        if not sim.priority and self.starts:
            sim.queue_rule, sim.priority = "priority", self.starts
        if agent == DISPATCHER:
            for a in legal:
                if a[0] == "release" and self.rel and a[1] == self.rel[0]:
                    self.rel.pop(0)
                    return a
            return ("wait",)
        q, r = self.legs.get(agent, []), sim.robots[agent]
        if r.carry is not None:
            opts = self.held.get(agent, ())
            dl = [a for a in legal if a[0] == "deliver" and (a[1] in opts or not opts)]
            return min(dl, key=lambda a: sim.stations[a[1]].spec["kind"] == "buffer") if dl else ("wait",)
        for i, (j, k, opts) in enumerate(q):
            if ("pickup", j) in legal and sim.jobs[j].k == k:
                q.pop(i)
                self.held[agent] = opts
                return ("pickup", j)
        if q:
            j = sim.jobs[q[0][0]]
            node = sim.stations[j.at].spec["node"] if j.at in sim.stations else None
            if node and ("move", node) in legal:
                return ("move", node)
        return ("wait",)
