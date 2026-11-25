"""Rule-based policies: policy(sim, agent, legal) -> action. Names: random, fifo, edd, spt, cr (critical ratio), mwkr (most
work remaining), nearest. Each rule is used for order release, robot pickups and (except fifo / nearest / random)
the machines' queues (policy.queue_rule)."""
import numpy as np

from . import timing

from .core import DISPATCHER

RULES = ("random", "fifo", "edd", "spt", "cr", "mwkr", "nearest")


def work_left(sim, j):
    """Nominal processing time of the job's remaining steps (fastest option per step)."""
    if j.route is None:
        return sim.F.nominal_flow(j.product, 0)
    return sum(min(timing.mean(sim.F.proc_model(s, j.product)) for s in opts) for opts in j.route[j.k:])


def rank(name, sim, j, station=None):
    """Priority key of job j under a dispatching rule (smaller first)."""
    if name == "edd":
        return (j.due, j.id)
    if name == "spt":
        return (timing.mean(sim.F.proc_model(station, j.product)) if station else work_left(sim, j), j.id)
    if name == "cr":
        return ((j.due - sim.t) / max(1.0, work_left(sim, j)), j.id)
    if name == "mwkr":
        return (-work_left(sim, j), j.id)
    return (j.arrival, j.id)


def setup_choice(sim, legal, patience=300.0, dwell=600.0):
    """Reconfigure a machine to take over a starved role (its machines down for > patience s, or no machine plays it)
    when more jobs wait for that role than for the role the machine would give up, at most once per `dwell` seconds
    per machine. Returns a ('setup', station, config) action or None."""
    sets = [a for a in legal if a[0] == "setup"]
    if not sets:
        return None
    need = {}
    for j in sim.jobs.values():
        if j.where in ("future", "done", "scrapped") or (j.where == "queue" and j.at in sim.stations and sim.stations[j.at].spec["kind"] != "kitting"):
            continue
        step = sim.F.products[j.product]["routes"][0][0] if j.where == "backlog" else (j.route[j.k] if j.route and j.k < len(j.route) else [])
        for role in step:
            need[role] = need.get(role, 0) + 1

    def players(role):
        return [n for n, st in sim.stations.items() if role in sim.F.covers(n, st.setup or st.config)]

    def starved(role):
        ps = players(role)
        return not ps or all(sim.stations[n].down and sim.t - (sim.stations[n].mark or sim.t) > patience for n in ps)

    best = None
    for a in sets:
        st = sim.stations[a[1]]
        gain = sum(need.get(role, 0) for role in sim.F.covers(a[1], a[2]) if starved(role))
        lose = sum(need.get(role, 0) for role in sim.F.covers(a[1], st.config) if len(players(role)) <= 1)
        recent = sim.t - (st.mark or -1e18) < dwell if st.n_setups else False
        if gain > lose and not recent and (best is None or gain - lose > best[0]):
            best = (gain - lose, a)
    return best[1] if best else None


def make(name="edd", seed=0, wip=None):
    if name not in RULES:
        raise ValueError("policy must be one of {}".format(RULES))
    rng = np.random.default_rng(seed)

    def key(sim, j):
        return rank(name, sim, j)

    def policy(sim, agent, legal):
        if name == "random":
            other = [a for a in legal if a[0] != "setup"]
            if len(other) < len(legal) and rng.random() < 0.02 or not other:
                return legal[rng.integers(len(legal))]
            return other[rng.integers(len(other))]
        if agent == DISPATCHER:
            su = setup_choice(sim, legal)
            if su:
                return su
            limit = wip if wip is not None else 2 * len(sim.robots) + 2
            active = sum(1 for j in sim.jobs.values() if j.where not in ("future", "backlog", "done", "scrapped"))
            rel = [a for a in legal if a[0] == "release"]
            if not rel or active >= limit:
                return ("wait",)
            return min(rel, key=lambda a: key(sim, sim.jobs[a[1]]) + (sim.F.nominal_flow(sim.jobs[a[1]].product, a[2]),))
        r = sim.robots[agent]
        here = r.node
        load = lambda s: len(sim.stations[s].q) + (sim.stations[s].cur is not None)  # noqa: E731
        dist = lambda s: sim.F.dist[here].get(sim.stations[s].spec["node"], 1e9)  # noqa: E731
        dl = [a for a in legal if a[0] == "deliver"]
        if dl:
            main = [a for a in dl if sim.stations[a[1]].spec["kind"] != "buffer"]
            return min(main or dl, key=lambda a: (load(a[1]), dist(a[1])))
        ready = lambda j: j.where != "buffer" or any(sim.cap_in(s) and sim.stations[s].spec["kind"] != "buffer" for s in sim.targets(j))  # noqa: E731
        pk = [a for a in legal if a[0] == "pickup" and ready(sim.jobs[a[1]])]
        if pk:
            blocked = lambda j: 0 if j.where == "machine" else 1  # noqa: E731
            if name == "nearest":
                return min(pk, key=lambda a: (blocked(sim.jobs[a[1]]), dist(sim.jobs[a[1]].at), a[1]))
            return min(pk, key=lambda a: (blocked(sim.jobs[a[1]]),) + key(sim, sim.jobs[a[1]]))
        if any(a[0] == "charge" for a in legal) and r.battery is not None and r.battery < 0.3 * r.spec["battery"]["capacity"]:
            return ("charge",)
        return ("wait",)

    if name in ("edd", "spt", "cr", "mwkr"):
        policy.queue_rule = lambda sim, station, jobs: min(jobs, key=lambda j: rank(name, sim, j, station))
    policy.rule = name
    return policy
