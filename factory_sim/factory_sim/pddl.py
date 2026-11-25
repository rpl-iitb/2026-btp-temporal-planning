"""PDDL 2.1 bridge: the factory as a temporal planning problem (nominal durations), and plan execution in the simulator.

Planning view = deterministic nominal model: no breakdowns, rejects or congestion (those are execution-time
events). Resources: robots (busy while moving / loading / unloading), machines (busy while processing and while
holding a finished tray: blocking), operators (held with their station during kitting / assembly), slots (input,
output, buffer), docks (one robot at a time). Processing and conveyors need no robot, so they overlap with transport.
"""
import re

from . import timing
from .core import DISPATCHER

DOMAIN = """(define (domain factorysim)
  (:requirements :typing :durative-actions :fluents :timed-initial-literals)
  (:types robot node station job step slot operator)
  (:predicates (robot-at ?r - robot ?n - node) (robot-free ?r - robot) (empty ?r - robot) (carrying ?r - robot ?j - job)
    (dock ?s - station ?n - node) (dock-free ?s - station) (is-sink ?s - station) (is-buffer ?s - station)
    (in-slot ?q - slot ?s - station) (out-slot ?q - slot ?s - station) (slot-free ?q - slot)
    (queued ?j - job ?s - station ?q - slot) (holding ?s - station ?j - job) (in-out ?j - job ?s - station ?q - slot) (stored ?j - job ?s - station ?q - slot)
    (machine-free ?s - station) (op-for ?o - operator ?s - station) (op-free ?o - operator)
    (step ?j - job ?k - step) (option ?j - job ?k - step ?s - station) (next ?k - step ?k2 - step) (final ?j - job ?k - step) (first ?j - job ?k - step)
    (robot-next ?j - job ?k - step ?s - station) (conveyor ?a - station ?b - station)
    (unreleased ?j - job) (arrived ?j - job) (open ?j - job) (delivered ?j - job))
  (:functions (travel ?a - node ?b - node) (ptime ?j - job ?k - step ?s - station) (ctime ?a - station ?b - station) (load-time) (unload-time) (release-time))
  (:durative-action move
    :parameters (?r - robot ?a - node ?b - node)
    :duration (= ?duration (travel ?a ?b))
    :condition (and (at start (robot-at ?r ?a)) (at start (robot-free ?r)))
    :effect (and (at start (not (robot-at ?r ?a))) (at start (not (robot-free ?r))) (at end (robot-at ?r ?b)) (at end (robot-free ?r))))
  (:durative-action release
    :parameters (?j - job ?k - step ?s - station ?q - slot)
    :duration (= ?duration (release-time))
    :condition (and (at start (unreleased ?j)) (at start (arrived ?j)) (at start (first ?j ?k)) (at start (option ?j ?k ?s)) (at start (in-slot ?q ?s)) (at start (slot-free ?q)))
    :effect (and (at start (not (unreleased ?j))) (at start (not (slot-free ?q))) (at end (queued ?j ?s ?q)) (at end (step ?j ?k))))
  (:durative-action process
    :parameters (?j - job ?s - station ?q - slot ?k - step ?k2 - step ?o - operator)
    :duration (= ?duration (ptime ?j ?k ?s))
    :condition (and (at start (queued ?j ?s ?q)) (at start (machine-free ?s)) (at start (op-free ?o)) (at start (op-for ?o ?s)) (at start (step ?j ?k)) (at start (option ?j ?k ?s)) (at start (next ?k ?k2)))
    :effect (and (at start (not (queued ?j ?s ?q))) (at start (slot-free ?q)) (at start (not (machine-free ?s))) (at start (not (op-free ?o))) (at start (not (step ?j ?k)))
      (at end (holding ?s ?j)) (at end (op-free ?o)) (at end (step ?j ?k2))))
  (:durative-action to-out
    :parameters (?j - job ?s - station ?q - slot ?k - step)
    :duration (= ?duration 0.01)
    :condition (and (at start (holding ?s ?j)) (at start (out-slot ?q ?s)) (at start (slot-free ?q)) (at start (step ?j ?k)) (at start (robot-next ?j ?k ?s)))
    :effect (and (at start (not (holding ?s ?j))) (at start (not (slot-free ?q))) (at end (in-out ?j ?s ?q)) (at end (machine-free ?s))))
  (:durative-action convey
    :parameters (?j - job ?a - station ?b - station ?k - step ?q - slot)
    :duration (= ?duration (ctime ?a ?b))
    :condition (and (at start (holding ?a ?j)) (at start (conveyor ?a ?b)) (at start (step ?j ?k)) (at start (option ?j ?k ?b)) (at start (in-slot ?q ?b)) (at start (slot-free ?q)))
    :effect (and (at start (not (holding ?a ?j))) (at start (machine-free ?a)) (at start (not (slot-free ?q))) (at end (queued ?j ?b ?q))))
  (:durative-action load-machine
    :parameters (?r - robot ?j - job ?s - station ?n - node ?k - step)
    :duration (= ?duration (load-time))
    :condition (and (at start (robot-at ?r ?n)) (at start (robot-free ?r)) (at start (empty ?r)) (at start (dock ?s ?n)) (at start (dock-free ?s)) (at start (holding ?s ?j)) (at start (step ?j ?k)) (at start (robot-next ?j ?k ?s)) (over all (robot-at ?r ?n)))
    :effect (and (at start (not (robot-free ?r))) (at start (not (empty ?r))) (at start (not (dock-free ?s))) (at start (not (holding ?s ?j))) (at start (machine-free ?s))
      (at end (carrying ?r ?j)) (at end (robot-free ?r)) (at end (dock-free ?s))))
  (:durative-action load-out
    :parameters (?r - robot ?j - job ?s - station ?n - node ?q - slot)
    :duration (= ?duration (load-time))
    :condition (and (at start (robot-at ?r ?n)) (at start (robot-free ?r)) (at start (empty ?r)) (at start (dock ?s ?n)) (at start (dock-free ?s)) (at start (in-out ?j ?s ?q)) (over all (robot-at ?r ?n)))
    :effect (and (at start (not (robot-free ?r))) (at start (not (empty ?r))) (at start (not (dock-free ?s))) (at start (not (in-out ?j ?s ?q))) (at start (slot-free ?q))
      (at end (carrying ?r ?j)) (at end (robot-free ?r)) (at end (dock-free ?s))))
  (:durative-action load-buffer
    :parameters (?r - robot ?j - job ?s - station ?n - node ?q - slot)
    :duration (= ?duration (load-time))
    :condition (and (at start (robot-at ?r ?n)) (at start (robot-free ?r)) (at start (empty ?r)) (at start (dock ?s ?n)) (at start (dock-free ?s)) (at start (stored ?j ?s ?q)) (over all (robot-at ?r ?n)))
    :effect (and (at start (not (robot-free ?r))) (at start (not (empty ?r))) (at start (not (dock-free ?s))) (at start (not (stored ?j ?s ?q))) (at start (slot-free ?q))
      (at end (carrying ?r ?j)) (at end (robot-free ?r)) (at end (dock-free ?s))))
  (:durative-action unload
    :parameters (?r - robot ?j - job ?s - station ?n - node ?k - step ?q - slot)
    :duration (= ?duration (unload-time))
    :condition (and (at start (robot-at ?r ?n)) (at start (robot-free ?r)) (at start (carrying ?r ?j)) (at start (dock ?s ?n)) (at start (dock-free ?s)) (at start (step ?j ?k)) (at start (option ?j ?k ?s)) (at start (in-slot ?q ?s)) (at start (slot-free ?q)) (over all (robot-at ?r ?n)))
    :effect (and (at start (not (robot-free ?r))) (at start (not (carrying ?r ?j))) (at start (not (dock-free ?s))) (at start (not (slot-free ?q)))
      (at end (queued ?j ?s ?q)) (at end (empty ?r)) (at end (robot-free ?r)) (at end (dock-free ?s))))
  (:durative-action unload-buffer
    :parameters (?r - robot ?j - job ?s - station ?n - node ?q - slot)
    :duration (= ?duration (unload-time))
    :condition (and (at start (robot-at ?r ?n)) (at start (robot-free ?r)) (at start (carrying ?r ?j)) (at start (dock ?s ?n)) (at start (dock-free ?s)) (at start (is-buffer ?s)) (at start (in-slot ?q ?s)) (at start (slot-free ?q)) (over all (robot-at ?r ?n)))
    :effect (and (at start (not (robot-free ?r))) (at start (not (carrying ?r ?j))) (at start (not (dock-free ?s))) (at start (not (slot-free ?q)))
      (at end (stored ?j ?s ?q)) (at end (empty ?r)) (at end (robot-free ?r)) (at end (dock-free ?s))))
  (:durative-action unload-sink
    :parameters (?r - robot ?j - job ?s - station ?n - node ?k - step)
    :duration (= ?duration (unload-time))
    :condition (and (at start (robot-at ?r ?n)) (at start (robot-free ?r)) (at start (carrying ?r ?j)) (at start (dock ?s ?n)) (at start (dock-free ?s)) (at start (is-sink ?s)) (at start (step ?j ?k)) (at start (final ?j ?k)) (over all (robot-at ?r ?n)) (over all (open ?j)))
    :effect (and (at start (not (robot-free ?r))) (at start (not (carrying ?r ?j))) (at start (not (dock-free ?s)))
      (at end (delivered ?j)) (at end (empty ?r)) (at end (robot-free ?r)) (at end (dock-free ?s))))
)
"""


LOADS = ("load-machine", "load-out", "load-buffer", "unload", "unload-buffer", "unload-sink")


MOVE = """  (:durative-action move
    :parameters (?r - robot ?a - node ?b - node)
    :duration (= ?duration (travel ?a ?b))
    :condition (and (at start (robot-at ?r ?a)) (at start (robot-free ?r)))
    :effect (and (at start (not (robot-at ?r ?a))) (at start (not (robot-free ?r))) (at end (robot-at ?r ?b)) (at end (robot-free ?r))))
"""
# lane traffic: robots drive along floor edges; a junction holds one robot (node-free), a pocket (dock, bay) any number.
# The robot keeps the junction it leaves until it is at the next node (so two robots cannot swap in a corridor).
LANE_MOVES = """  (:durative-action move
    :parameters (?r - robot ?a - node ?b - node)
    :duration (= ?duration (travel ?a ?b))
    :condition (and (at start (robot-at ?r ?a)) (at start (robot-free ?r)) (at start (link ?a ?b)) (at start (pocket ?b)))
    :effect (and (at start (not (robot-at ?r ?a))) (at start (not (robot-free ?r))) (at end (robot-at ?r ?b)) (at end (robot-free ?r)) (at end (node-free ?a))))
  (:durative-action move-j
    :parameters (?r - robot ?a - node ?b - node)
    :duration (= ?duration (travel ?a ?b))
    :condition (and (at start (robot-at ?r ?a)) (at start (robot-free ?r)) (at start (link ?a ?b)) (at start (junction ?b)) (at start (node-free ?b)))
    :effect (and (at start (not (robot-at ?r ?a))) (at start (not (robot-free ?r))) (at start (not (node-free ?b))) (at end (robot-at ?r ?b)) (at end (robot-free ?r)) (at end (node-free ?a))))
"""


def lanes(F):
    return F is not None and getattr(F, "lanes", False)


def domain_name(F):
    return ("factorysim" if legacy(F) else "factorysim-fleet") + ("-lanes" if lanes(F) else "")


def legacy(F):
    """FactorySim 1.0 fleets (one robot type that loads itself) keep the original domain text."""
    return F is None or (F.homogeneous and F.handling == ["arm"])


def domain(F=None):
    """The PDDL domain for this factory's fleet. Robots that load themselves (arm) get load-* / unload-*; robots
    with a passive tray (handler) get handover-* variants that also need a free handler (an operator) at the
    station. Times and travel are per robot (load-time ?r, travel ?r ?a ?b) once the fleet is mixed."""
    text = DOMAIN
    if lanes(F):
        assert MOVE in text
        text = text.replace(MOVE, LANE_MOVES).replace("(unreleased ?j - job)", "(link ?a - node ?b - node) (junction ?n - node) (pocket ?n - node) (node-free ?n - node)\n    (unreleased ?j - job)", 1)
        text = text.replace("(domain factorysim)", "(domain factorysim-lanes)", 1)
        # the slot of these actions is implied by a dynamic fact; the static condition says so, which keeps
        # grounders from building every job x station x slot (the plants have hundreds of slots)
        for act, pre, add in (("load-out", "(at start (in-out ?j ?s ?q))", "(at start (out-slot ?q ?s))"),
                              ("load-buffer", "(at start (stored ?j ?s ?q))", "(at start (in-slot ?q ?s))"),
                              ("process", "(at start (queued ?j ?s ?q))", "(at start (in-slot ?q ?s))")):
            a = text.index("(:durative-action {}\n".format(act))
            b = text.index(pre, a)
            text = text[:b] + add + " " + text[b:]
    if legacy(F):
        return text
    body, tail = text.rstrip().rsplit("\n", 1)          # tail: the domain's closing ")"
    head, *acts = (body + "\n").split("  (:durative-action ")
    head = head.replace("(unreleased ?j - job)", "(self-loading ?r - robot) (handled ?r - robot) (handler-for ?h - operator ?s - station)\n    (unreleased ?j - job)")
    head = head.replace("(travel ?a - node ?b - node)", "(travel ?r - robot ?a - node ?b - node)").replace("(load-time) (unload-time)", "(load-time ?r - robot) (unload-time ?r - robot)")
    out = [head]
    for a in acts:
        name = a.split()[0]
        a = a.replace("(travel ?a ?b)", "(travel ?r ?a ?b)").replace("(load-time)", "(load-time ?r)").replace("(unload-time)", "(unload-time ?r)")
        if name not in LOADS:
            out.append(a)
            continue
        if "arm" in F.handling:
            out.append(a.replace(":condition (and ", ":condition (and (at start (self-loading ?r)) ", 1))
        if "handler" in F.handling:
            h = ("handover-" + a).replace(")\n    :duration", " ?h - operator)\n    :duration", 1)
            h = h.replace(":condition (and ", ":condition (and (at start (handled ?r)) (at start (op-free ?h)) (at start (handler-for ?h ?s)) ", 1)
            h = h.replace(":effect (and ", ":effect (and (at start (not (op-free ?h))) (at end (op-free ?h)) ", 1)
            out.append(h)
    text = "  (:durative-action ".join(out) + tail + "\n"
    return re.sub(r"\(domain factorysim(-lanes)?\)", "(domain {})".format(domain_name(F)), text, count=1)


def _n(x):
    return re.sub(r"[^A-Za-z0-9_]", "_", str(x)).lower()


def _node(x):
    return "n_" + _n(x)


def export(sim, hard_deadlines=False, name="factory"):
    """The current simulator state as a PDDL 2.1 problem (time shifted so that now = 0). Returns problem text."""
    F, t = sim.F, sim.t
    old = legacy(F)
    load_t, unload_t = max(0.01, timing.mean(F.robots["load"])), max(0.01, timing.mean(F.robots["unload"]))
    hpools = {} if old or "handler" not in F.handling else {s: sp.get("handler", F.handler_pool) for s, sp in F.stations.items() if F.reachable(s)}
    objs = {"robot": [], "node": set(), "station": [], "job": [], "step": [], "slot": [], "operator": []}
    init, num, til = set(), [], []

    def at(dt, lit):
        til.append((max(0.001, round(dt, 3)), lit))
    for s, sp in F.stations.items():
        S = _n(s)
        objs["station"].append(S)
        if F.reachable(s):
            objs["node"].add(sp["node"])
            init |= {"(dock {} {})".format(S, _node(sp["node"])), "(dock-free {})".format(S)}
        if sp["kind"] == "sink":
            init.add("(is-sink {})".format(S))
            continue
        if sp["kind"] == "buffer":
            init.add("(is-buffer {})".format(S))
        st = sim.stations[s]
        ins = ["{}_in{}".format(S, i) for i in range(sp["slots_in"])]
        outs = ["{}_out{}".format(S, i) for i in range(sp["slots_out"])]
        objs["slot"] += ins + outs
        init |= {"(in-slot {} {})".format(q, S) for q in ins} | {"(out-slot {} {})".format(q, S) for q in outs}
        by_robot = sum(1 for r in sim.robots.values() if r.task and r.task[0] == "deliver" and r.task[2] == s)
        held = [("q", j) for j in st.q] + [("r", None)] * (st.reserved - by_robot)
        for q, (kind, j) in zip(ins, held):
            if kind == "q":
                init.add("(stored {} {} {})".format(_n("j{}".format(j.id)), S, q) if sp["kind"] == "buffer" else "(queued {} {} {})".format(_n("j{}".format(j.id)), S, q))
        init |= {"(slot-free {})".format(q) for q in ins[len(held):]}
        for q, j in zip(outs, st.out):
            init.add("(in-out {} {} {})".format(_n("j{}".format(j.id)), S, q))
        init |= {"(slot-free {})".format(q) for q in outs[len(st.out):]}
        if sp["kind"] != "buffer":
            if sp.get("operator"):
                pool = sp["operator"]
                ops = ["op_{}_{}".format(_n(pool), i) for i in range(F.operators[pool])]
                init |= {"(op-for {} {})".format(o, S) for o in ops}
            else:
                ops = ["op_self_{}".format(S)]
                init.add("(op-for {} {})".format(ops[0], S))
                if st.cur is None or st.done:
                    init.add("(op-free {})".format(ops[0]))
            objs["operator"] += [o for o in ops if o not in objs["operator"]]
            if st.cur is None:
                init.add("(machine-free {})".format(S))
            elif st.done:
                init.add("(holding {} {})".format(S, _n("j{}".format(st.cur.id))))
    for s, pool in hpools.items():
        ops = ["op_{}_{}".format(_n(pool), i) for i in range(F.operators[pool])]
        init |= {"(handler-for {} {})".format(o, _n(s)) for o in ops}
        objs["operator"] += [o for o in ops if o not in objs["operator"]]
    for pool, n in F.operators.items():
        if not any(st.get("operator") == pool for st in F.stations.values()) and pool not in hpools.values():
            continue
        busy = sum(1 for s in sim.stations.values() if s.spec.get("operator") == pool and s.cur is not None and not s.done)
        busy += sum(1 for r in sim.robots.values() if r.hand and r.hand[0] == pool)
        init |= {"(op-free op_{}_{})".format(_n(pool), i) for i in range(max(0, n - busy))}
    for a, b in F.transfers:
        init.add("(conveyor {} {})".format(_n(a), _n(b)))
        num.append("(= (ctime {} {}) {})".format(_n(a), _n(b), max(0.01, timing.mean(F.transfers[(a, b)]))))
    for n, r in sim.robots.items():
        R = _n(n)
        objs["robot"].append(R)
        if not old:
            init.add("({} {})".format("handled" if r.spec["handling"] == "handler" else "self-loading", R))
            num += ["(= (load-time {}) {})".format(R, max(0.01, timing.mean(r.spec["load"]))), "(= (unload-time {}) {})".format(R, max(0.01, timing.mean(r.spec["unload"])))]
        objs["node"] |= {r.node, r.dest}
        if r.task is None:
            init |= {"(robot-at {} {})".format(R, _node(r.node)), "(robot-free {})".format(R)}
        elif r.phase == "travel":
            at(r.t1 - t, "(robot-at {} {})".format(R, _node(r.dest)))
            at(r.t1 - t, "(robot-free {})".format(R))
            if lanes(F) and r.src in F.junctions and r.src != r.dest:
                at(r.t1 - t, "(node-free {})".format(_node(r.src)))
        elif r.phase == "blocked":
            init |= {"(robot-at {} {})".format(R, _node(r.node)), "(robot-free {})".format(R)}
        else:
            init.add("(robot-at {} {})".format(R, _node(r.node)))
            at(1.0 if r.phase in ("queue", "handler_wait") else max(0.001, max((v[3] or t) for k, v in sim.acts.items() if v[1] == n and v[0] in ("load", "unload", "charge")) - t), "(robot-free {})".format(R))
        if r.carry is not None:
            init.add("(carrying {} {})".format(R, _n("j{}".format(r.carry))))
        else:
            init.add("(empty {})".format(R))
    goals = []
    for j in sim.jobs.values():
        if j.where in ("scrapped",):
            continue
        J = _n("j{}".format(j.id))
        objs["job"].append(J)
        if j.where == "done":
            init.add("(delivered {})".format(J))
            continue
        goals.append("(delivered {})".format(J))
        if hard_deadlines and j.due < float("inf"):
            init.add("(open {})".format(J))
            at(j.due - t, "(not (open {}))".format(J))
        else:
            init.add("(open {})".format(J))
        routes = [(j.route_id, j.route)] if j.route else list(enumerate(F.products[j.product]["routes"]))
        for ri, route in routes:
            ks = ["{}_r{}_k{}".format(J, ri, i) for i in range(len(route) + 1)]
            objs["step"] += ks
            init |= {"(next {} {})".format(a, b) for a, b in zip(ks, ks[1:])} | {"(final {} {})".format(J, ks[-1]), "(first {} {})".format(J, ks[0])}
            for i, opts in enumerate(route):
                for s in opts:
                    init.add("(option {} {} {})".format(J, ks[i], _n(s)))
                    num.append("(= (ptime {} {} {}) {})".format(J, ks[i], _n(s), max(0.01, round(timing.mean(F.proc_model(s, j.product)), 3))))
            for i in range(1, len(route) + 1):
                nxt = route[i] if i < len(route) else F.sinks
                for s in route[i - 1]:
                    if not any((s, o) in F.transfers for o in nxt):
                        init.add("(robot-next {} {} {})".format(J, ks[i], _n(s)))
            if j.route is None:
                init.add("(unreleased {})".format(J))
                (init.add if j.arrival <= t else (lambda x: at(j.arrival - t, x)))("(arrived {})".format(J))
                if lanes(F) and j.arrival <= t:     # keeps `arrived` dynamic in every instance (as with online orders);
                    at(10 ** 6, "(arrived {})".format(J))   # long after any plan ends, so it never meets an action
            elif j.where == "machine" and not sim.stations[j.at].done:
                st = sim.stations[j.at]
                end = sim.acts[st.aid][3] if st.aid is not None and sim.acts[st.aid][3] is not None else t + timing.mean(F.proc_model(j.at, j.product))
                at(end - t, "(holding {} {})".format(_n(j.at), J))
                at(end - t, "(step {} {})".format(J, ks[j.k + 1]))
                if st.spec.get("operator"):
                    pass
                else:
                    at(end - t, "(op-free op_self_{})".format(_n(j.at)))
            elif j.where == "conveyor":
                kind, data, t0, t1 = next(v for v in sim.acts.values() if v[0] == "conv" and v[1][0] == j.id and (v[3] is None or v[3] >= t))
                tgt = data[2]
                q = next(x for x in sorted(objs["slot"]) if x.startswith(_n(tgt) + "_in") and "(slot-free {})".format(x) not in init and not any(x in y for y in init if y.startswith("(queued")))
                at((t1 or t) - t, "(queued {} {} {})".format(J, _n(tgt), q))
                init.add("(step {} {})".format(J, ks[j.k]))
            else:
                init.add("(step {} {})".format(J, ks[min(j.k, len(route))]))
    for pool in F.operators:
        for s in sim.stations.values():
            if s.spec.get("operator") == pool and s.cur is not None and not s.done:
                end = sim.acts[s.aid][3] if s.aid is not None and sim.acts[s.aid][3] is not None else t + 1
                for i in range(F.operators[pool]):
                    o = "(op-free op_{}_{})".format(_n(pool), i)
                    if o not in init and not any(o == x for _, x in til):
                        at(end - t, o)
                        break
    for r in sim.robots.values():  # handlers busy with a robot's tray come free when that pick / place ends
        if r.hand is not None:
            end = max((v[3] or t) for v in sim.acts.values() if v[1] == r.name and v[0] in ("load", "unload"))
            for i in range(F.operators[r.hand[0]]):
                o = "(op-free op_{}_{})".format(_n(r.hand[0]), i)
                if o not in init and not any(o == x for _, x in til):
                    at(end - t, o)
                    break
    if lanes(F):
        objs["node"] |= set(F.nodes)
        held = {r.dest if r.phase == "travel" else r.node for r in sim.robots.values()} | {r.src for r in sim.robots.values() if r.phase == "travel"}
        for n in F.nodes:
            init.add("({} {})".format("junction" if n in F.junctions else "pocket", _node(n)))
            if n in F.junctions and n not in held:
                init.add("(node-free {})".format(_node(n)))
    nodes = sorted(objs["node"])
    for a in nodes:
        for b in (sorted(F.adj[a]) if lanes(F) else nodes):
            if lanes(F):
                init.add("(link {} {})".format(_node(a), _node(b)))
                d = F.adj[a][b]
                if old:
                    num.append("(= (travel {} {}) {})".format(_node(a), _node(b), round(d / F.robots["speed"], 3)))
                else:
                    num += ["(= (travel {} {} {}) {})".format(_n(n), _node(a), _node(b), round(d / r.spec["speed"], 3)) for n, r in sim.robots.items()]
                continue
            if a != b and F.travel(a, b) < float("inf"):
                if old:
                    num.append("(= (travel {} {}) {})".format(_node(a), _node(b), round(F.travel(a, b), 3)))
                else:
                    num += ["(= (travel {} {} {}) {})".format(_n(n), _node(a), _node(b), round(F.travel(a, b, r.spec["speed"]), 3)) for n, r in sim.robots.items()]
    objs["node"] = [_node(x) for x in nodes]
    num += (["(= (load-time) {})".format(load_t), "(= (unload-time) {})".format(unload_t)] if old else []) + ["(= (release-time) 1)"]
    lines = ["(define (problem {}) (:domain {})".format(_n(name), domain_name(F)), "  (:objects"]
    for ty, xs in objs.items():
        if xs:
            lines.append("    {} - {}".format(" ".join(sorted(set(xs))), ty))
    lines += ["  )", "  (:init"] + ["    " + x for x in sorted(init)] + ["    " + x for x in num]
    lines += ["    (at {:.3f} {})".format(dt, x) for dt, x in sorted(til)]
    lines += ["  )", "  (:goal (and {}))".format(" ".join(goals)), "  (:metric minimize (total-time)))"]
    return "\n".join(lines) + "\n"


def write(sim, domain_path, problem_path, **kw):
    open(domain_path, "w").write(domain(sim.F))
    open(problem_path, "w").write(export(sim, **kw))
    return domain_path, problem_path


class PlanPolicy:
    """Execute a temporal plan in the simulator: its releases and robot pickups / deliveries (moves, processing,
    conveyors and out-slot moves are done by the simulator itself), each as soon as it is legal and the plan's order
    allows it. Each agent does its own steps in plan order. Across agents the plan's order is kept where it matters:
    a step waits for the earlier steps on its job; a delivery also waits for every earlier pickup and delivery at its
    station (free slots, and the station's first-come-first-served queue gets the plan's order); a pickup happens only
    at the station the plan picks the job up from; and orders are released in the order the plan processes them at
    their first station (kitting serves its queue in arrival order). Without these, a robot could take a job early
    from a station where the plan has another robot collect it, or fill a slot another robot's job needs first, and
    the run would stall although the plan is valid."""

    def __init__(self, sim, plan_text):
        jid = {_n("j{}".format(j)): j for j in sim.jobs}
        rob = {_n(r): r for r in sim.robots}
        sta = {_n(s): s for s in sim.stations}
        self.q = {a: [] for a in [DISPATCHER] + list(sim.robots)}
        rows = []
        for line in plan_text.splitlines():
            m = re.match(r"\s*([0-9.]+)\s*:\s*\(([^)]*)\)", line)
            if m:
                rows.append((float(m.group(1)), m.group(2).lower().split()))
        rows.sort(key=lambda x: x[0])
        first_process = {}              # job -> time the plan first processes it (at its first station)
        for t, a in rows:
            if a[0] == "process" and a[1] in jid:
                first_process.setdefault(jid[a[1]], t)
        self.steps = []                 # (agent, action, job, station)
        releases = []
        for t, a in rows:
            kind = a[0].replace("handover-", "")
            if a[0] == "release":
                j = jid[a[1]]
                releases.append((first_process.get(j, t), t, (DISPATCHER, ("release", j, int(a[2].split("_r")[-1].split("_")[0])), j, sta[a[3]])))
                continue
            if kind.startswith("load"):
                step = (rob[a[1]], ("pickup", jid[a[2]]), jid[a[2]], sta[a[3]])
            elif kind.startswith("unload"):
                step = (rob[a[1]], ("deliver", sta[a[3]]), jid[a[2]], sta[a[3]])
            else:
                continue
            self.steps.append(step)
        # releases first, in processing order (their jobs' later steps depend on them anyway)
        self.steps = [r[2] for r in sorted(releases, key=lambda r: (r[0], r[1]))] + self.steps
        for i, step in enumerate(self.steps):
            self.q[step[0]].append(i)
        last_job, at_station, self.after = {}, {}, []
        for i, (_, act, j, s) in enumerate(self.steps):
            need = [last_job[j]] if j in last_job else []
            if act[0] == "deliver":
                need += at_station.get(s, [])
            self.after.append(need)
            last_job[j] = i
            if act[0] in ("pickup", "deliver"):
                at_station.setdefault(s, []).append(i)
        self.issued = set()

    def __call__(self, sim, agent, legal):
        q = self.q.get(agent, [])
        if not q:
            return ("wait",)
        i = q[0]
        _, action, j, s = self.steps[i]
        if any(p not in self.issued for p in self.after[i]) or tuple(action) not in legal:
            return ("wait",)
        if action[0] == "pickup" and sim.pickup_station(sim.jobs[j]) != s:
            return ("wait",)
        q.pop(0)
        self.issued.add(i)
        return action
