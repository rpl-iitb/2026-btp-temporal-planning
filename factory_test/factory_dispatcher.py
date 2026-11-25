"""Rule-based robot dispatcher over FactorySim's PDDL states (algorithm/menv.py): the factory's plug-in for V8 (as a
constructive scheduler), V9 (macro-step environment) and V12 / V13 (option generator and DAgger expert).

In menv the actions without a robot argument start by themselves as soon as they are applicable (release, process,
to-out, convey); a robot chooses among its applicable actions. Options offered to a robot:
  load / unload / handover-* actions it can do where it stands (pick up a job, deliver it, store it in a buffer);
  go_pickup   move to the dock of a station with a job ready for a robot (not already claimed by another robot);
  go_deliver  move to the dock of a station that can take the job it carries now (free input slot, or a sink);
  go_buffer   move to an overflow buffer with a free slot (carrying, no target free);
  wait.
Expert (decide): deliver here > store here when no target is free > pick up here > go deliver (nearest target) >
go pick up (policy "nearest": the nearest ready job; "progress": the job furthest along its route; "score": the
station with the highest learned score) > go to a buffer > wait.

Lane traffic (instances whose PDDL has move-j: corridor junctions hold one robot, FactorySim's traffic: lanes). A move
is one floor edge, so a trip is many moves. Each go_* option is the next edge of the trip's shortest route (or of a
route at most 25 % / 30 s longer whose first edge still gets closer, when that edge is taken); a trip whose way is
taken is not offered. Two layers (Dispatcher._lane_options): the robot (rules, or a learned policy) chooses its trip
off the corridor (at a dock or bay) and, on a junction, only among the trips of its last choice that lead this way;
a rule-based traffic layer moves it on the corridor: it drives on when the way is free; when blocked it waits, or
backs into a bay (go_aside) when a still robot with priority (carrying before empty, then by name) is in the way
or after YIELD_AFTER s blocked (this breaks every deadlock), and it stays in the bay until that robot has passed;
an idle robot never waits on a junction. Pickups are shared out by a greedy matching (the nearest empty robot gets
a job). A robot picks up only a job it has somewhere to take (no blocking for good); under hard deadlines the
expert takes the most urgent of the near jobs, and once an order is late every robot stops (the run has failed).
"""
import heapq
import random
from collections import defaultdict

LOADS = ("load-machine", "load-out", "load-buffer")
UNLOADS = ("unload", "unload-sink", "unload-buffer")
MOVES = ("move", "move-j")
RULES = UNLOADS + LOADS + ("go_deliver", "go_pickup", "go_buffer", "go_aside", "wait")
DETOUR, DETOUR_S, YIELD_AFTER, HOLD = 1.25, 30.0, 120.0, 90.0


def base(name):
    return name[len("handover-"):] if name.startswith("handover-") else name


class View:
    """Factory facts of a state, decoded once per decision."""

    def __init__(self, env, S):
        T = env.T
        self.t = S.t
        self.at, self.carry, self.dock, self.holding, self.out, self.stored = {}, {}, {}, {}, {}, {}
        self.step, self.final, self.inslots = {}, {}, defaultdict(list)
        self.option, self.rnext = defaultdict(list), defaultdict(list)
        self.free, self.sinks, self.buffers, self.delivered, self.jobs = set(), set(), set(), set(), set()
        self.junctions, self.open = set(), set()
        if not hasattr(env, "_static_facts"):       # static facts (dock, option, robot-next, ...) are not in the state
            env._static_facts = list(T.static_init)
        if not hasattr(env, "_due"):                # hard deadlines: (open j) ends at the due date (a timed literal)
            env._due = {T.facts[f][1]: t for f, t in getattr(T, "window", {}).items() if T.facts[f][0] == "open"}
        self.due = env._due
        for x in [T.facts[f] for f in S.facts] + env._static_facts:
            p = x[0]
            if p == "robot-at": self.at[x[1]] = x[2]          # noqa: E701
            elif p == "carrying": self.carry[x[1]] = x[2]     # noqa: E701
            elif p == "dock": self.dock[x[1]] = x[2]          # noqa: E701
            elif p == "holding": self.holding[x[2]] = x[1]    # noqa: E701
            elif p == "in-out": self.out[x[1]] = x[2]         # noqa: E701
            elif p == "stored": self.stored[x[1]] = x[2]      # noqa: E701
            elif p == "step": self.step[x[1]] = x[2]          # noqa: E701
            elif p == "final": self.final.setdefault(x[1], set()).add(x[2])   # noqa: E701  (one per route)
            elif p == "in-slot": self.inslots[x[2]].append(x[1])  # noqa: E701
            elif p == "option": self.option[(x[1], x[2])].append(x[3])  # noqa: E701
            elif p == "robot-next": self.rnext[(x[1], x[2])].append(x[3])  # noqa: E701
            elif p == "slot-free": self.free.add(x[1])        # noqa: E701
            elif p == "is-sink": self.sinks.add(x[1])         # noqa: E701
            elif p == "is-buffer": self.buffers.add(x[1])     # noqa: E701
            elif p == "delivered": self.delivered.add(x[1])   # noqa: E701
            elif p == "junction": self.junctions.add(x[1])    # noqa: E701
            elif p == "open": self.open.add(x[1])             # noqa: E701
        self.heading = {}                     # robot -> node it is driving to
        self.moving = set()
        for i, st, lo, hi in S.run:
            a = T.actions[i]
            if a.name in MOVES:
                self.heading[a.args[0]] = a.args[2]
                self.moving.add(a.args[0])
            elif base(a.name) in LOADS + UNLOADS:
                self.heading[a.args[0]] = a.args[3]

    def ready(self):
        """{job: station} of jobs a robot can pick up now (on a machine waiting for a robot, in an output slot, stored)."""
        out = {}
        for j, s in self.holding.items():         # holding: job -> the machine that holds it
            if s in self.rnext.get((j, self.step.get(j)), ()):
                out[j] = s
        out.update(self.out)
        out.update(self.stored)
        return out

    def targets(self, j):
        """Stations that can take carried job j now."""
        k = self.step.get(j)
        if k is not None and k in self.final.get(j, ()):
            return sorted(self.sinks)
        return sorted(s for s in self.option.get((j, k), ()) if any(q in self.free for q in self.inslots.get(s, ())))

    def progress(self, j):
        k = self.step.get(j, "")
        return int(k.rsplit("_k", 1)[-1]) if "_k" in k else 0

    def missed(self):
        """Hard deadlines: some order is past its due date and not delivered (the run has failed)."""
        return any(j not in self.open and j not in self.delivered for j in self.due)

    def room(self, r):
        """Free input slots per station, less one for each other robot that carries a job it could take there."""
        out = {s: sum(1 for q in qs if q in self.free) for s, qs in self.inslots.items()}
        for o, j in self.carry.items():
            if o != r:
                for s in self.targets(j):
                    out[s] = out.get(s, 0) - 1
        return out

    def can_place(self, j, room):
        """A robot that picks up job j has somewhere to take it (a sink, or a next station with room)."""
        k = self.step.get(j)
        if k is not None and k in self.final.get(j, ()):
            return True
        return any(room.get(s, 0) > 0 for s in self.option.get((j, k), ()))

    def pos(self, r):
        """Where robot r is (or the node it is driving to)."""
        return self.at.get(r) or self.heading.get(r)

    def holder(self, n):
        """The robot on (or entering) junction n."""
        for o in sorted(set(self.at) | set(self.heading)):
            if self.pos(o) == n:
                return o
        return None


class Lanes:
    """The floor graph of a lane-traffic task (from its ground moves): edge times per robot, distances to a node."""

    def __init__(self, T, junctions):
        self.J = junctions
        self.adj = defaultdict(dict)                 # robot -> {a: {b: time}}
        for a in T.actions:
            if a.name in MOVES:
                self.adj[a.args[0]].setdefault(a.args[1], {})[a.args[2]] = a.lo
        self._to = {}

    def to(self, r, n):
        """{node: time to drive from it to n} (routes pass only through junctions)."""
        key = (r, n)
        if key not in self._to:
            adj, d, h = self.adj[r], {n: 0.0}, [(0.0, n)]
            while h:
                x, u = heapq.heappop(h)
                if x > d[u] or (u != n and u not in self.J):
                    continue
                for v, w in adj.get(u, {}).items():
                    if x + w < d.get(v, float("inf")):
                        d[v] = x + w
                        heapq.heappush(h, (x + w, v))
            self._to[key] = d
        return self._to[key]

    def path(self, r, a, n, limit=200):
        D, adj, out = self.to(r, n), self.adj[r], []
        while a != n and len(out) < limit and a in D:
            a = min(adj.get(a, {}), key=lambda v: (adj[a][v] + D.get(v, float("inf")), v), default=None)
            if a is None or D.get(a, float("inf")) == float("inf"):
                break
            out.append(a)
        return out


class Dispatcher:
    RULES = RULES

    def __init__(self, env, policy="nearest", score=None, seed=None):
        self.env, self.policy, self.score = env, policy, score or {}
        self.rng = random.Random(seed) if seed is not None else None
        self.L = None
        self._blk = {}                      # lanes: robot -> (node, time) where it has been blocked since
        self._last = (None, [])             # lanes: (robot, its trips that are blocked) of the last options() call
        self._pend, self._hold = {}, {}     # lanes: robot -> (junction, priority robot, its node[, since]) to make room for
        self._trip = {}                     # lanes: robot -> {"at": node, "trips": {trip: its next node}} of its last decision

    def lanes(self, V):
        if self.L is None and V.junctions:
            if not hasattr(self.env, "_lanes"):
                self.env._lanes = Lanes(self.env.T, V.junctions)
            self.L = self.env._lanes
        return self.L

    def travel_time(self, V, r, station):
        """Driving time of robot r from where it is to the dock of `station` (lane traffic only; else None)."""
        L = self.lanes(V)
        n, here = V.dock.get(station), V.pos(r)
        if L is None or n is None or here is None:
            return None
        return L.to(r, n).get(here)

    # ---- options
    def options(self, V, r, legal):
        if self.lanes(V) is not None:
            return self._lane_options(V, r, legal)
        T, here = self.env.T, V.at.get(r)
        opts, moves = [], {}
        for k, i in enumerate(legal):
            a = T.actions[i]
            if a.name == "move":
                moves[a.args[2]] = (k, a.lo)
            elif base(a.name) in LOADS + UNLOADS:
                opts.append({"k": k, "rule": base(a.name), "station": a.args[2], "dist": 0.0,
                             "info": {"job": a.args[1], "handover": a.name.startswith("handover-")}})
        claimed = {V.heading.get(o) for o in V.at if o != r and o not in V.carry}
        if r in V.carry:
            j = V.carry[r]
            for s in V.targets(j):
                n = V.dock.get(s)
                if n in moves and n != here:
                    opts.append({"k": moves[n][0], "rule": "go_deliver", "station": s, "dist": moves[n][1], "info": {"job": j}})
            for s in sorted(V.buffers):
                n = V.dock.get(s)
                if n in moves and n != here and any(q in V.free for q in V.inslots.get(s, ())):
                    opts.append({"k": moves[n][0], "rule": "go_buffer", "station": s, "dist": moves[n][1], "info": {"job": j}})
        else:
            for j, s in sorted(V.ready().items()):
                n = V.dock.get(s)
                if n in moves and n != here and n not in claimed:
                    opts.append({"k": moves[n][0], "rule": "go_pickup", "station": s, "dist": moves[n][1], "info": {"job": j}})
        opts.append({"k": None, "rule": "wait", "station": None, "dist": 0.0, "info": {}})
        return opts

    def _assign(self, V, r):
        """Greedy matching of empty robots to ready jobs by driving time: {job: robot}."""
        L, ready = self.L, V.ready()
        idle = [o for o in sorted(set(V.at) | set(V.heading)) if o not in V.carry and (o == r or o in V.moving or o in V.at)]
        pairs = []
        for j, s in ready.items():
            n = V.dock.get(s)
            for o in idle:
                d = L.to(o, n).get(V.pos(o), float("inf"))
                if d < float("inf"):
                    pairs.append((d, o != r, o, j))
        got, used = {}, set()
        for d, _, o, j in sorted(pairs):
            if j not in got and o not in used:
                got[j] = o
                used.add(o)
        return got

    def _held(self, V, r, here):
        """A robot that backed into a bay for a robot with priority stays there until that robot has passed (HOLD s
        at most): -> the junction it must not enter yet, or None."""
        if here is not None and here not in self.L.J and r in self._pend:
            u, q, b = self._pend.pop(r)
            if here in self.L.adj[r].get(u, {}):
                self._hold[r] = (u, q, b, V.t)
        h = self._hold.get(r)
        if h is None:
            return None
        u, q, b, t0 = h
        if V.pos(q) in (u, b) and V.t - t0 <= HOLD and here not in self.L.J:
            return u
        del self._hold[r]
        return None

    def _lane_options(self, V, r, legal):
        """Lane traffic. The robot chooses its trip (and what to load / unload) where it stands off the corridor (a
        dock or bay), and again on a junction only between trips of its last choice that still drive this way. On a
        junction the traffic layer decides alone when no trip can drive on: wait, or go aside when a still robot with
        priority is in the way or after YIELD_AFTER s; with nothing to do, go aside. Every option names its trip's
        next edge."""
        T, L, here = self.env.T, self.L, V.at.get(r)
        WAIT = {"k": None, "rule": "wait", "station": None, "dist": 0.0, "info": {}}
        if V.due and V.missed():                    # a deadline is missed: the run has failed, stop working
            self._last = (r, [])
            return [WAIT]
        opts, hops = [], {}
        keep_off = self._held(V, r, here)
        room = V.room(r)
        for k, i in enumerate(legal):
            a = T.actions[i]
            if a.name in MOVES:
                if a.args[2] != keep_off:
                    hops[a.args[2]] = (k, a.lo)
            elif base(a.name) in LOADS + UNLOADS:
                if base(a.name) in LOADS and not V.can_place(a.args[1], room):
                    continue                        # no room downstream: picking it up could block for good
                opts.append({"k": k, "rule": base(a.name), "station": a.args[2], "dist": 0.0,
                             "info": {"job": a.args[1], "handover": a.name.startswith("handover-")}})
        trips = []
        if r in V.carry:
            j = V.carry[r]
            trips += [("go_deliver", s, j) for s in V.targets(j)]
            trips += [("go_buffer", s, j) for s in sorted(V.buffers) if any(q in V.free for q in V.inslots.get(s, ()))]
        else:
            mine = self._assign(V, r)
            trips += [("go_pickup", s, j) for j, s in sorted(V.ready().items()) if mine.get(j, r) == r and V.can_place(j, room)]
        on_j = here in L.J
        c = self._trip.get(r)
        if on_j and c is not None:                  # in transit: keep to the trips chosen before that lead through here
            keep = set(c["trips"]) if c["at"] == here else {t for t, b in c["trips"].items() if b == here}
            kept = [t for t in trips if t in keep]
            trips = kept or trips
        occ = {V.pos(o): o for o in sorted(set(V.at) | set(V.heading)) if o != r}
        adj, blocked, first = L.adj[r].get(here, {}), [], {}
        for t in trips:
            rule, s, j = t
            n = V.dock.get(s)
            if n is None or n == here:
                continue
            D = L.to(r, n)
            dist = D.get(here, float("inf"))
            if dist == float("inf"):
                continue
            via = sorted((adj[b] + D.get(b, float("inf")), b) for b in adj if (b in L.J or b == n))
            if not via or via[0][0] == float("inf"):
                continue
            ok = [(cst, b) for cst, b in via if b in hops and cst <= dist * DETOUR + DETOUR_S and D.get(b, float("inf")) < dist]   # every edge gets closer: no back and forth
            if not ok:
                blocked.append((rule, s, j, via[0][1], dist))
                first[t] = via[0][1]
                continue
            cst, b = ok[0]
            first[t] = b
            route = L.path(r, here, n)
            cong = sum(1 for x in route if x in occ and x in L.J) / max(1, len(route))
            opts.append({"k": hops[b][0], "rule": rule, "station": s, "dist": dist,
                         "info": {"job": j, "detour": cst - dist, "congestion": cong, "first_free": float(b == via[0][1])}})
        self._trip[r] = {"at": here, "trips": first}
        self._last = (r, blocked)
        self._pend.pop(r, None)
        for _, _, _, b, _ in blocked:       # a robot with priority stands in the way: going aside makes room for it
            q = V.holder(b)
            if q is not None and q not in V.moving and self._prio(V, q) > self._prio(V, r):
                self._pend[r] = (here, q, b)
                break
        bl = float(bool(blocked))
        for o in opts:
            o["info"]["blocked"] = bl
        if not on_j:                                # off the corridor: the robot's choice (or wait)
            self._blk.pop(r, None)
            return opts + [dict(WAIT, info={"blocked": bl})]
        aside = sorted((d, b) for b, (k, d) in hops.items() if b not in L.J)
        go_aside = {"k": hops[aside[0][1]][0], "rule": "go_aside", "station": None, "dist": aside[0][0], "info": {"blocked": bl}} if aside else None
        if not blocked:                             # free way on: drive on (or wait); nothing to do: never wait here
            self._blk.pop(r, None)
            return opts + [dict(WAIT, info={"blocked": 0.0})] if opts else ([go_aside] if go_aside else [WAIT])
        node, t0 = self._blk.get(r, (None, None))
        if node != here:
            self._blk[r], t0 = (here, V.t), V.t
        if r in self._pend or V.t - t0 > YIELD_AFTER:   # it must make room: drive on another way or back off, not wait
            return opts + [go_aside] if go_aside else (opts or [WAIT])
        return opts + [dict(WAIT, info={"blocked": 1.0})]

    def _prio(self, V, r):
        return (r in V.carry, -int("".join(c for c in r if c.isdigit()) or 0))

    def _lane_decide(self, V, r, by):
        """Lanes: nothing else to start: the traffic layer's choice (go aside, when it is the option), else wait."""
        return by["go_aside"][0]["k"] if by["go_aside"] else None

    # ---- expert
    def _key(self, V, o):
        j = o["info"].get("job")
        noise = self.rng.random() if self.rng else 0.0
        if self.policy == "progress":
            return (-V.progress(j), o["dist"], noise)
        if self.policy == "score":
            return (-self.score.get(o["station"], 0.5), o["dist"], noise)
        return (o["dist"], noise, -V.progress(j))

    def decide(self, V, r, legal, opts=None):
        """-> index into legal, or None (wait)."""
        opts = opts if opts is not None else self.options(V, r, legal)
        return self._decide(V, r, legal, opts)

    def _decide(self, V, r, legal, opts):
        by = defaultdict(list)
        for o in opts:
            by[o["rule"]].append(o)
        if self.L is not None:                          # lanes: the preferred trip may be blocked; then wait for it
            for rule, st, j, _, dist in (self._last[1] if self._last[0] == r else []):
                by[rule].append({"k": "blocked", "rule": rule, "station": st, "dist": dist, "info": {"job": j}})
        carried = V.carry.get(r)
        if carried is not None:
            here = [o for o in by["unload"] + by["unload-sink"] if o["info"]["job"] == carried]
            if here:
                return here[0]["k"]
            if by["go_deliver"]:
                return self._go(V, r, by, min(by["go_deliver"], key=lambda o: self._key(V, o)))
            if by["unload-buffer"]:
                return by["unload-buffer"][0]["k"]
            if by["go_buffer"]:
                return self._go(V, r, by, min(by["go_buffer"], key=lambda o: o["dist"]))
            return self._lane_decide(V, r, by) if self.L is not None else None
        here = by["load-machine"] + by["load-out"] + by["load-buffer"]
        if here:
            return min(here, key=lambda o: -V.progress(o["info"]["job"]))["k"]
        if by["go_pickup"]:
            return self._go(V, r, by, self._urgent(V, by["go_pickup"]))
        return self._lane_decide(V, r, by) if self.L is not None else None

    def _urgent(self, V, opts):
        """Lanes with deadlines: among the trips at most 60 s longer than the nearest, the job due first."""
        if self.L is None or not V.due:
            return min(opts, key=lambda o: self._key(V, o))
        d0 = min(o["dist"] for o in opts)
        return min((o for o in opts if o["dist"] <= d0 + 60.0), key=lambda o: (V.due.get(o["info"].get("job"), float("inf")), o["dist"]))

    def _go(self, V, r, by, o):
        return self._lane_decide(V, r, by) if o["k"] == "blocked" else o["k"]

    def __call__(self, env, S, r, legal):
        return self.decide(View(env, S), r, legal)


def expert_index(D, V, r, legal, opts):
    k = D.decide(V, r, legal, opts)
    for j, o in enumerate(opts):
        if o["k"] == k:
            return j
    return len(opts) - 1                      # wait


NF = len(RULES) + 13


def features(D, V, r, opts):
    """[n options, NF]: rule one-hot, distance, handover, other robots heading there, job progress, carrying, free robots,
    jobs ready, jobs delivered, time; lane traffic: share of the route's junctions taken by other robots, detour (s),
    the robot's way to some target is blocked; slack of the option's job to its due date."""
    import numpy as np
    n_jobs = max(1, len(V.step) + len(V.delivered))
    heading = defaultdict(int)
    for o2, node in V.heading.items():
        if o2 != r:
            heading[node] += 1
    free = sum(1 for o2 in V.at if o2 not in V.heading)
    ready = len(V.ready())
    X = np.zeros((len(opts), NF), np.float32)
    for j, o in enumerate(opts):
        x = X[j]
        x[RULES.index(o["rule"])] = 1.0
        b = len(RULES)
        x[b] = min(o["dist"], 500.0) / 100.0
        x[b + 1] = float(o["info"].get("handover", False))
        x[b + 2] = heading[V.dock.get(o["station"])] / 2.0 if o["station"] else 0.0
        x[b + 3] = V.progress(o["info"]["job"]) / 10.0 if o["info"].get("job") else 0.0
        x[b + 4] = float(r in V.carry)
        x[b + 5] = free / max(1, len(V.at))
        x[b + 6] = ready / n_jobs
        x[b + 7] = len(V.delivered) / n_jobs
        x[b + 8] = min(V.t, 20000.0) / 5000.0
        info = o["info"]
        x[b + 9] = info.get("congestion", 0.0)
        x[b + 10] = min(info.get("detour", 0.0), 300.0) / 100.0
        x[b + 11] = info.get("blocked", 0.0)
        due = V.due.get(info.get("job")) if info.get("job") else None
        x[b + 12] = float(np.clip((due - V.t) / 5000.0, -2.0, 4.0)) if due is not None else 4.0
    return X
