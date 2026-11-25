"""FactorySim core: the authoritative task-level factory state, its event queue, rules and decision points.

Decision makers are the robots and one central dispatcher (order release); machines, operators, buffers and
conveyors are resources that run by themselves. Durations come from a time source: sampled ("sampled"),
nominal ("nominal") or reported by an external backend such as ROS 2 ("external", see Sim.complete).
"""
import heapq
import itertools

import numpy as np

from . import timing
from .config import Factory, PROCESSING

DISPATCHER = "dispatcher"


class Job:
    __slots__ = ("id", "product", "arrival", "due", "route", "route_id", "k", "where", "at", "claim", "t_release", "t_done", "origin", "reworks", "weight", "rush")

    def __init__(self, jid, product, arrival, due, origin=None, weight=1.0, rush=False):
        self.id, self.product, self.arrival, self.due, self.origin, self.weight, self.rush = jid, product, arrival, due, origin if origin is not None else jid, weight, rush
        self.route, self.route_id, self.k, self.where, self.at, self.claim, self.t_release, self.t_done, self.reworks = None, None, 0, "future", None, None, None, None, 0


class Station:
    __slots__ = ("name", "spec", "q", "cur", "done", "out", "reserved", "down", "paused", "docked", "dockq", "aid", "busy", "blocked", "since", "bsince",
                 "config", "setup", "service", "due_service", "wait", "worked", "t_down", "t_setup", "t_service", "mark", "n_setups", "n_services")

    def __init__(self, name, spec):
        self.name, self.spec = name, spec
        self.q, self.cur, self.done, self.out, self.reserved, self.down, self.paused, self.docked, self.dockq, self.aid = [], None, False, [], 0, False, None, 0, [], None
        self.busy = self.blocked = 0.0
        self.since = self.bsince = None
        self.config, self.setup, self.service, self.due_service, self.wait, self.worked = spec.get("config"), None, False, None, None, 0.0
        self.t_down = self.t_setup = self.t_service = 0.0
        self.mark, self.n_setups, self.n_services = None, 0, 0


class Robot:
    __slots__ = ("name", "node", "carry", "task", "phase", "src", "dest", "t0", "t1", "battery", "dist", "busy", "since", "fault", "fault_pending", "bt", "charges", "t_charge", "t_fault", "min_battery", "lowtok", "manual", "manual_pending", "spec", "hand", "route")

    def __init__(self, name, node, battery, spec=None):
        self.name, self.node, self.carry, self.task, self.phase, self.src, self.dest, self.t0, self.t1, self.battery, self.dist, self.busy, self.since = name, node, None, None, None, node, node, 0.0, 0.0, battery, 0.0, 0.0, None
        self.fault, self.fault_pending, self.bt, self.charges, self.t_charge, self.t_fault, self.min_battery, self.lowtok = False, False, 0.0, 0, 0.0, 0.0, battery, 0
        self.manual = self.manual_pending = False
        self.spec, self.hand = spec or {}, None   # per-robot type parameters; operator pool of the handler serving it
        self.route = []                         # lane traffic: nodes still to drive through


def default_orders(F, rng):
    spec = F.cfg.get("orders", {})
    n, mix = spec.get("count", 4), spec.get("mix", {p: 1 for p in F.products})
    names, w = list(mix), np.array([mix[p] for p in mix], float)
    arr, out, t = spec.get("arrivals", "batch"), [], 0.0
    for i in range(n):
        p = names[rng.choice(len(names), p=w / w.sum())] if spec.get("random_mix", False) else names[i % len(names)]
        if isinstance(arr, dict) and "online" in arr:
            t += float(rng.exponential(arr["online"])) if i else 0.0
        rush = spec.get("rush") and rng.random() < spec["rush"].get("prob", 0.0)
        due = (spec["rush"].get("due_factor", 1.2) if rush else spec.get("due_factor", 3.0)) * F.nominal_flow(p)
        out.append({"id": i, "product": p, "arrival": round(t, 3), "due": round(t + due, 3), **({"rush": True, "weight": spec["rush"].get("weight", 3.0)} if rush else {})})
    return out


class Sim:
    def __init__(self, factory, orders=None, seed=0, time_source="sampled", cv_scale=1.0, horizon=None, record=False, queue_rule="fifo", disturbances=None):
        self.F = factory if isinstance(factory, Factory) else Factory(factory)
        F = self.F
        self.rng = np.random.default_rng(seed)
        self.mode = time_source
        self.stochastic = time_source == "sampled" and F.features["stochastic"]
        self.cv_scale = cv_scale if self.stochastic else 0.0
        self.disturb = self.stochastic if disturbances is None else bool(disturbances)
        self.breakdowns = self.disturb and F.features["breakdowns"]
        self.rejects = self.disturb and F.features["rejects"]
        self.robot_faults = self.disturb and any(x.get("fault") for x in F.fleet)
        self.jams = F.jams if self.disturb else None
        self.horizon, self.record, self.queue_rule, self.priority = horizon, record, queue_rule, {}
        self.t, self.ev, self.seq, self.aid = 0.0, [], itertools.count(), itertools.count()
        self.acts, self.external_pending, self.cancelled, self.outcome, self.jammed = {}, {}, set(), {}, set()
        self.stations = {n: Station(n, s) for n, s in F.stations.items()}
        r = F.robots
        init = lambda b: (b["capacity"] * float(self.rng.uniform(*b["initial"])) if b.get("initial") else b["capacity"]) if b else None  # noqa: E731
        self.robots = {"r{}".format(i + 1): Robot("r{}".format(i + 1), r["start"][i % len(r["start"])], init(x["battery"]), x) for i, x in enumerate(F.fleet)}
        self.occ, self.waitq = {}, {}    # lane traffic: junction -> robot on it (or entering it); junction -> robots waiting for it
        if F.lanes:
            if time_source == "external":
                raise ValueError("lane traffic needs simulated time (time_source sampled or nominal)")
            for x in self.robots.values():
                if x.node in F.junctions:
                    if x.node in self.occ:          # two robots start on one junction: the second waits in a bay of it
                        x.node = x.src = x.dest = F.bays[x.node][0]
                    else:
                        self.occ[x.node] = x.name
        self.handq = {}   # operator pool -> robots docked and waiting for a handler
        self.hpos, self.hbusy = {}, {}   # operator pool -> [node of each handler], indices of handlers at work
        self.ops_free, self.ops_cap = dict(F.operators), dict(F.operators)
        self.bat = F.robots.get("battery")
        self.charger_at, self.chargeq = {c: None for c in F.chargers}, {c: [] for c in F.chargers}
        self.stock = dict(F.materials.get("stock", {})) if F.materials else None
        self.jobs, self.backlog = {}, []
        for o in (orders if orders is not None else default_orders(F, self.rng)):
            j = Job(o["id"], o["product"], float(o.get("arrival", 0.0)), float(o.get("due", float("inf"))), weight=float(o.get("weight", 1.0)), rush=bool(o.get("rush", False)))
            self.jobs[j.id] = j
            self._push(j.arrival, "arrive", j.id)
        self.n_target = len(self.jobs)
        self.completed, self.failed, self.decided = 0, None, set()
        self.progressed = False      # an agent took a non-wait action at the current instant
        self.stats = {"rejects": 0, "scraps": 0, "breakdowns": 0, "decisions": 0, "grasp_failures": 0, "robot_faults": 0, "jams": 0, "setups": 0, "services": 0, "replenishments": 0, "handovers": 0, "handler_wait": 0.0, "yields": 0, "parks": 0, "traffic_wait": 0.0}
        self.rfid, self.trace = [], []
        if self.breakdowns:
            for s in self.stations.values():
                if s.spec.get("breakdown"):
                    self._push(self.t + self.rng.exponential(s.spec["breakdown"]["mtbf"]), "fail", s.name)
        for s in self.stations.values():
            for start, dur in s.spec.get("downtime", []):
                self._push(float(start), "pm", (s.name, float(dur)))
        if self.robot_faults:
            for r in self.robots.values():
                if r.spec.get("fault"):
                    self._push(self.rng.exponential(r.spec["fault"]["mtbf"]), "rfail", r.name)
        for op, b in F.breaks.items():
            self._push(float(b.get("first", b["every"])), "break", (op, int(b.get("n", 1)), float(b["duration"])))
        if self.stock is not None and F.materials.get("replenish"):
            self._push(float(F.materials["replenish"]["every"]), "replenish", None)
        self._run_now()
        for r in self.robots.values():
            self._auto_charge(r)

    # ---------------- events and activities
    def _push(self, t, kind, data):
        heapq.heappush(self.ev, (t, next(self.seq), kind, data))

    def _activity(self, kind, data, model, nominal_scale=1.0, cv=None, extra=0.0):
        """Start an activity; its end is sampled (or nominal) here, or reported later by an external backend."""
        aid = next(self.aid)
        nom = timing.mean(model) * nominal_scale + extra
        if self.mode == "external":
            self.external_pending[aid] = (kind, data, self.t, nom)
            self.acts[aid] = (kind, data, self.t, None)
            return aid
        d = ((timing.sample(model, self.rng, self.cv_scale) * nominal_scale) if cv is None else ((nom - extra) * float(self.rng.lognormal(-np.log(1 + cv * cv) / 2, np.sqrt(np.log(1 + cv * cv)))) if cv > 0 else nom - extra)) + extra
        self.acts[aid] = (kind, data, self.t, self.t + d)
        self._push(self.t + d, "act", aid)
        return aid

    def complete(self, aid, t=None, ok=True):
        """External backends: activity aid has finished at time t (default: now); ok=False reports a failed pick/place (it is retried)."""
        if aid not in self.external_pending:
            raise KeyError("unknown or finished activity {}".format(aid))
        kind, data, t0, nom = self.external_pending.pop(aid)
        self.t = max(self.t, t if t is not None else self.t)
        self.acts[aid] = (kind, data, t0, self.t)
        self.outcome[aid] = ok
        self._finish(aid)
        self.decided.clear()
        self._run_now()

    def advance(self, t):
        """External backends: the clock has reached t (process scheduled events up to t, e.g. order arrivals)."""
        self.t = max(self.t, t)
        self.decided.clear()
        self._run_now()

    def _run_now(self):
        while self.ev and self.ev[0][0] <= self.t + 1e-9:
            self._pop()
        self._snap()

    def _pop(self):
        t, _, kind, data = heapq.heappop(self.ev)
        self.t = max(self.t, t)
        if kind == "act":
            if data in self.cancelled:
                self.cancelled.discard(data)
                return
            self._finish(data)
        elif kind == "arrive":
            j = self.jobs[data]
            if j.where == "future":
                j.where = "backlog"
                self.backlog.append(j.id)
        elif kind == "fail":
            self._fail(data)
        elif kind == "repair":
            self._repair(data)
        elif kind == "pm":
            st = self.stations[data[0]]
            st.due_service = data[1]
            self._try_start(data[0])
        elif kind == "break":
            op, n, dur = data
            self.ops_cap[op] -= n
            self.ops_free[op] -= n
            self._push(self.t + dur, "back", (op, n))
            self._push(self.t + float(self.F.breaks[op]["every"]), "break", data)
        elif kind == "back":
            op, n = data
            self.ops_cap[op] += n
            self.ops_free[op] += n
            self._wake_operator(op)
            self._assign_tech()
        elif kind == "replenish":
            rp = self.F.materials["replenish"]
            for p in self.F.products:
                if p in self.stock:
                    self.stock[p] += rp["qty"].get(p, 0) if isinstance(rp["qty"], dict) else rp["qty"]
            self.stats["replenishments"] += 1
            self._push(self.t + float(rp["every"]), "replenish", None)
        elif kind == "rfail":
            r = self.robots[data]
            r.fault_pending = True
            if r.task is None and r.phase is None:
                self._idle(r)
        elif kind == "rrepair":
            r = self.robots[data]
            r.fault, r.phase = False, None
            r.t_fault += self.t - r.t0
            self._drain(r)
            self._push(self.t + self.rng.exponential(r.spec["fault"]["mtbf"]), "rfail", r.name)
            self._auto_charge(r)
        elif kind == "lowbat":
            r = self.robots[data[0]]
            if data[1] == r.lowtok:
                self._auto_charge(r)

    def _finish(self, aid):
        kind, data = self.acts[aid][:2]
        if kind == "proc":
            self._proc_done(data)
        elif kind == "conv":
            self._conv_done(*data)
        elif kind == "travel":
            self._arrive(data)
        elif kind == "hop":
            self._hop_done(data)
        elif kind == "repair":
            self._repair(data)
        elif kind in ("load", "unload"):
            sp = self.robots[data].spec
            p = sp["grasp_success" if kind == "load" else "place_success"]
            if not self.outcome.pop(aid, True) or (self.stochastic and p < 1.0 and self.rng.random() >= p):
                self.stats["grasp_failures"] += 1
                self._activity(kind, data, sp[kind])
                return
            self._service_done(data, kind)
        elif kind == "charge":
            self._charge_done(data)
        elif kind == "setup":
            self._setup_done(data)
        elif kind == "service":
            self._service_end(data)

    # ---------------- stations
    def cap_in(self, s):
        st = self.stations[s]
        return st.spec["kind"] == "sink" or len(st.q) + st.reserved < st.spec["slots_in"]

    def _try_start(self, s):
        st = self.stations[s]
        sp = st.spec
        if sp["kind"] not in PROCESSING or st.cur is not None or st.down or st.setup is not None or st.service or st.wait:
            return
        if st.due_service is not None:
            st.wait = "service"
            return self._assign_tech()
        if not st.q:
            return
        op = sp.get("operator")
        if op and self.ops_free[op] <= 0:
            return
        if callable(self.queue_rule):
            j = self.queue_rule(self, s, list(st.q))
        elif self.queue_rule == "edd":
            j = min(st.q, key=lambda x: (x.due, x.id))
        elif self.queue_rule == "priority":
            j = min(st.q, key=lambda x: (self.priority.get((x.id, x.k), float("inf")), x.id))
        else:
            j = st.q[0]
        st.q.remove(j)
        st.cur, st.done, j.where, j.at, st.since = j, False, "machine", s, self.t
        if op:
            self.ops_free[op] -= 1
        st.aid = self._activity("proc", s, self.F.proc_model(s, j.product, st.config))
        self._unblock()

    def _proc_done(self, s):
        st = self.stations[s]
        j, sp = st.cur, st.spec
        st.busy += self.t - st.since
        st.worked += self.t - st.since
        st.aid = None
        mt = sp.get("maintenance")
        if mt and st.worked >= mt["every"] and st.due_service is None:
            st.due_service = mt["duration"]
        op = sp.get("operator")
        if op:
            self.ops_free[op] += 1
        j.k += 1
        if self.rejects and sp.get("reject", 0) > 0 and self.rng.random() < sp["reject"]:
            self.stats["rejects"] += 1
            rw = sp.get("rework_to")
            if rw:
                j.k = next(i for i, opts in enumerate(j.route) if rw in opts)
                j.reworks += 1
            else:
                self.stats["scraps"] += 1
                j.where, st.cur = "scrapped", None
                nid = max(self.jobs) + 1
                self.jobs[nid] = Job(nid, j.product, self.t, j.due, origin=j.origin)
                self.jobs[nid].where = "backlog"
                self.backlog.append(nid)
                self._try_start(s)
                self._wake_operator(op)
                return
        st.done = True
        self._flush(s)
        self._wake_operator(op)

    def _wake_operator(self, op):
        if op:
            while self.handq.get(op) and self.ops_free[op] > 0:
                self._handover(self.handq[op].pop(0), op)
            for n, x in self.stations.items():
                if x.spec.get("operator") == op:
                    self._try_start(n)

    def targets(self, j):
        return self.cands(self.F.sinks if j.k >= len(j.route) else j.route[j.k])

    def cands(self, opts):
        """Stations that can do a route step now: those whose current (or incoming) configuration plays one of the step's roles."""
        cs = [n for n in self.F.stations if self.F.stations[n].get("configs")]
        if not cs:
            return opts
        want, out = set(opts), []
        for n in list(opts) + cs:
            st = self.stations[n]
            if n in out:
                continue
            if not st.spec.get("configs"):
                out.append(n)
            elif self.F.covers(n, st.setup or st.config) & want:
                out.append(n)
        return out

    def _flush(self, s):
        """Move a finished job off its machine: by conveyor, to the output buffer, or stay (blocked after service)."""
        st = self.stations[s]
        j = st.cur
        conv = [o for o in self.targets(j) if (s, o) in self.F.transfers]
        if conv:
            ok = [o for o in conv if self.cap_in(o)]
            if not ok:
                return self._block(st)
            o = min(ok, key=lambda x: (len(self.stations[x].q) + (self.stations[x].cur is not None), x))
            self.stations[o].reserved += 1
            self._unblock_end(st)
            st.cur, st.done, j.where = None, False, "conveyor"
            self._log(s, "OUT", j)
            jam = self.jams and self.rng.random() < self.jams.get("prob", 0.0)
            aid = self._activity("conv", (j.id, s, o), self.F.transfers[(s, o)], extra=timing.sample(self.jams["clear"], self.rng) if jam else 0.0)
            if jam:
                self.stats["jams"] += 1
                self.jammed.add(aid)
        elif len(st.out) < st.spec["slots_out"]:
            self._unblock_end(st)
            st.out.append(j)
            st.cur, st.done, j.where = None, False, "out"
        else:
            return self._block(st)
        self._try_start(s)
        return True

    def _block(self, st):
        if st.bsince is None:
            st.bsince = self.t
        st.cur.where = "machine"
        return False

    def _unblock_end(self, st):
        if st.bsince is not None:
            st.blocked += self.t - st.bsince
            st.bsince = None

    def _unblock(self):
        changed = True
        while changed:
            changed = False
            for n, st in self.stations.items():
                if st.cur is not None and st.done and self._flush(n):
                    changed = True

    def _conv_done(self, jid, s, o):
        j, st = self.jobs[jid], self.stations[o]
        st.reserved -= 1
        st.q.append(j)
        j.where, j.at = "queue", o
        self._log(o, "IN", j)
        self._try_start(o)

    def _fail(self, s):
        st = self.stations[s]
        if st.setup is not None or st.service or st.wait:
            return self._push(self.t + self.rng.exponential(st.spec["breakdown"]["mtbf"]), "fail", s)
        st.down, st.wait, st.mark = True, "repair", self.t
        self.stats["breakdowns"] += 1
        if st.aid is not None and not st.done:
            kind, data, t0, t1 = self.acts[st.aid]
            if t1 is None:
                del self.external_pending[st.aid]
                t1 = self.t + max(0.0, t0 + self._nominal(data) - self.t)
            self.cancelled.add(st.aid)
            st.paused, st.aid = t1 - self.t, None
        self._assign_tech()

    def _nominal(self, s):
        st = self.stations[s]
        return timing.mean(self.F.proc_model(s, st.cur.product, st.config)) if st.cur is not None else 0.0

    def _assign_tech(self):
        """Give the technician crew (if any) to waiting repairs, then services, then changeovers; start that work."""
        tech = self.F.technician
        for want in ("repair", "service", "setup"):
            for n, st in self.stations.items():
                if st.wait != want:
                    continue
                if tech:
                    if self.ops_free[tech] <= 0:
                        return
                    self.ops_free[tech] -= 1
                st.wait = None
                if want == "repair":
                    self._activity("repair", n, st.spec["breakdown"]["mttr"])
                elif want == "service":
                    st.service, st.mark = True, self.t
                    self._activity("service", n, st.due_service)
                else:
                    st.mark = self.t
                    self._activity("setup", n, st.spec["changeover"])
                self._snap()

    def _free_tech(self):
        if self.F.technician:
            self.ops_free[self.F.technician] += 1
        self._assign_tech()

    def _setup_done(self, s):
        st = self.stations[s]
        st.config, st.setup = st.setup, None
        st.t_setup += self.t - st.mark
        st.n_setups += 1
        self.stats["setups"] += 1
        self._free_tech()
        self._try_start(s)
        self._unblock()

    def _service_end(self, s):
        st = self.stations[s]
        st.service, st.due_service, st.worked = False, None, 0.0
        st.t_service += self.t - st.mark
        st.n_services += 1
        self.stats["services"] += 1
        self._free_tech()
        self._try_start(s)

    def _repair(self, s):
        st = self.stations[s]
        st.down = False
        st.t_down += self.t - st.mark
        if self.F.technician:
            self.ops_free[self.F.technician] += 1
        if st.paused is not None:
            st.aid = self._activity("proc", s, {"fixed": st.paused})
            st.paused = None
        else:
            self._try_start(s)
        self._push(self.t + self.rng.exponential(st.spec["breakdown"]["mtbf"]), "fail", s)
        self._assign_tech()

    # ---------------- robots
    def pickup_station(self, j):
        if j.claim is not None or j.where not in ("out", "machine", "buffer"):
            return None
        st = self.stations[j.at]
        if j.where == "machine" and not (st.cur is j and st.done):
            return None
        if j.where != "buffer" and any((j.at, o) in self.F.transfers for o in self.targets(j)):
            return None
        return j.at

    def _travel(self, r, node, after):
        r.task = after
        if r.node == node:
            r.src = r.dest = node
            return self._arrive(r.name)
        if self.F.lanes:
            self._drain(r)
            if r.since is None and after[0] != "park":
                r.since = self.t
            r.route = self.F.path(r.node, node)[1:]
            return self._hop(r)
        moving = sum(1 for x in self.robots.values() if x.phase == "travel")
        d = self.F.dist[r.node][node]
        self._drain(r)
        r.phase, r.src, r.dest, r.t0, r.dist = "travel", r.node, node, self.t, r.dist + d
        if r.battery is not None:
            r.battery = max(0.0, r.battery - d * r.spec["battery"]["per_metre"])
            r.min_battery = min(r.min_battery, r.battery)
        if r.since is None:
            r.since = self.t
        v = r.spec["speed"]
        aid = self._activity("travel", r.name, {"fixed": d / v}, 1 + self.F.slowdown * moving, cv=r.spec["travel_cv"] * self.cv_scale)
        r.t1 = self.acts[aid][3] if self.acts[aid][3] is not None else self.t + d / v

    # ---------------- lane traffic (Factory.lanes): edge-by-edge driving with junction reservations
    def _hop(self, r):
        """Drive the next edge of r.route, or wait for its junction (deadlocks: _resolve)."""
        F, nxt = self.F, r.route[0]
        if nxt in F.junctions:
            holder = self.occ.get(nxt)
            if holder is not None and holder != r.name:
                q = self.robots[holder]
                if q.task is None and q.phase is None and not q.fault and not q.manual:
                    self._park(q)               # an idle robot on the junction makes way
                if r.phase != "blocked":
                    r.phase, r.src, r.dest, r.t0 = "blocked", r.node, r.node, self.t
                self.waitq.setdefault(nxt, []).append(r.name)
                self._resolve(r)
                return
            self.occ[nxt] = r.name
        if r.phase == "blocked":
            self.stats["traffic_wait"] += self.t - r.t0
        d = F.adj[r.node][nxt]
        r.phase, r.src, r.dest, r.t0, r.dist = "travel", r.node, nxt, self.t, r.dist + d
        if r.battery is not None:
            r.battery = max(0.0, r.battery - d * r.spec["battery"]["per_metre"])
            r.min_battery = min(r.min_battery, r.battery)
        v = r.spec["speed"]
        aid = self._activity("hop", r.name, {"fixed": d / v}, 1.0, cv=r.spec["travel_cv"] * self.cv_scale)
        r.t1 = self.acts[aid][3] if self.acts[aid][3] is not None else self.t + d / v

    def _hop_done(self, name):
        r = self.robots[name]
        prev, r.node = r.node, r.dest
        r.route.pop(0)
        if prev in self.F.junctions and self.occ.get(prev) == r.name:
            del self.occ[prev]
            self._wake(prev)
        if r.route:
            return self._hop(r)
        r.phase = None
        self._arrive(r.name)

    def _wake(self, node):
        """The junction came free: the robots waiting for it try again, in the order they came."""
        for n in self.waitq.pop(node, []):
            w = self.robots[n]
            if w.phase == "blocked" and w.route and w.route[0] == node:
                self._hop(w)

    def _park(self, q):
        """Move an idle robot off its junction into the nearest bay (no decision: the traffic layer does it)."""
        bays = self.F.bays.get(q.node, [])
        if not bays:
            return
        self.stats["parks"] += 1
        q.task, q.route = ("park", bays[0], None), [bays[0]]
        self._hop(q)

    def _resolve(self, r):
        """Deadlock check when r starts waiting: follow who waits for whom from r; on a cycle, the robot of the cycle
        with the lowest priority (a bay to back into first, then empty before carrying, then name) backs into a bay."""
        chain, x = [r.name], r
        while True:
            h = self.occ.get(x.route[0]) if x.route else None
            if h is None:
                return
            hq = self.robots[h]
            if hq.phase != "blocked":
                return
            if h in chain:
                cycle = chain[chain.index(h):]
                break
            chain.append(h)
            x = hq
        y = min((self.robots[n] for n in cycle), key=lambda z: (not self.F.bays.get(z.node), z.carry is not None, z.name))
        self._back_off(y)

    def _back_off(self, y):
        bays = self.F.bays.get(y.node, [])
        if not bays or y.node not in self.F.junctions:
            return
        q = self.waitq.get(y.route[0], [])
        if y.name in q:
            q.remove(y.name)
        goal = y.route[-1]
        self.stats["yields"] += 1
        y.route = [bays[0]] + (self.F.path(bays[0], goal)[1:] if goal != bays[0] else [])
        self._hop(y)

    def _arrive(self, name):
        r = self.robots[name]
        r.node, r.phase = r.dest, None
        kind = r.task[0]
        if kind in ("pickup", "deliver"):
            self._dock(r, r.task[2])
        elif kind == "charge":
            self._start_charge(r, r.task[2])
        else:
            self._idle(r)

    def _dock(self, r, s):
        st = self.stations[s]
        if st.docked < st.spec["dock_capacity"]:
            st.docked += 1
            if r.spec["handling"] == "handler":
                pool = st.spec.get("handler", self.F.handler_pool)
                if self.ops_free[pool] > 0:
                    return self._handover(r, pool)
                r.phase, r.t0 = "handler_wait", self.t
                self.handq.setdefault(pool, []).append(r)
                return
            self._serve(r)
        else:
            r.phase = "queue"
            st.dockq.append(r)

    def _handover(self, r, pool):
        """A handler from `pool` comes to a docked tray robot and moves the tray (the person is busy until it is done)."""
        if r.phase == "handler_wait":
            self.stats["handler_wait"] += self.t - r.t0
        self.stats["handovers"] += 1
        self.ops_free[pool] -= 1
        pos = self.hpos.setdefault(pool, [self.F.handler_home] * self.F.operators[pool])
        busy = self.hbusy.setdefault(pool, set())
        node = self.stations[r.task[2]].spec["node"]
        i = min((k for k in range(len(pos)) if k not in busy), key=lambda k: (self.F.dist[pos[k]].get(node, 1e9), k), default=None)
        walk = 0.0
        if i is not None:
            walk = self.F.dist[pos[i]].get(node, 0.0) / self.F.handler_speed
            busy.add(i)
            pos[i] = node
        r.hand = (pool, i)
        self._serve(r, walk)

    def _serve(self, r, walk=0.0):
        r.phase = "service"
        if r.battery is not None:
            self._drain(r)
            r.battery = max(0.0, r.battery - r.spec["battery"]["per_service"])
        kind = "load" if r.task[0] == "pickup" else "unload"
        self._activity(kind, r.name, r.spec[kind], extra=walk)

    def _service_done(self, name, kind):
        r = self.robots[name]
        s = r.task[2]
        st = self.stations[s]
        st.docked -= 1
        if r.hand is not None:
            (pool, i), r.hand = r.hand, None
            self.hbusy[pool].discard(i)
            self.ops_free[pool] += 1
            self._wake_operator(pool)
        if st.dockq:
            self._dock(st.dockq.pop(0), s)
        if kind == "load":
            j = self.jobs[r.task[1]]
            if j.where == "out":
                st.out.remove(j)
            elif j.where == "machine":
                self._unblock_end(st)
                st.cur, st.done = None, False
            else:
                st.q.remove(j)
            r.carry, j.where, j.at, j.claim = j.id, "robot", r.name, None
            self._log(s, "OUT", j)
            self._try_start(s)
            self._unblock()
        else:
            j = self.jobs[r.carry]
            st.reserved -= 1
            r.carry, j.at = None, s
            self._log(s, "IN", j)
            if st.spec["kind"] == "sink":
                j.where, j.t_done = "done", self.t
                self.completed += 1
            else:
                j.where = "buffer" if st.spec["kind"] == "buffer" else "queue"
                st.q.append(j)
                self._try_start(s)
        self._idle(r)

    def _idle(self, r):
        if r.since is not None:
            r.busy += self.t - r.since
        r.task, r.phase, r.since = None, None, None
        if r.fault_pending:
            r.fault_pending, r.fault, r.phase, r.t0 = False, True, "fault", self.t
            self.stats["robot_faults"] += 1
            self._push(self.t + timing.sample(r.spec["fault"]["mttr"], self.rng), "rrepair", r.name)
            return
        if r.manual_pending:
            r.manual_pending, r.manual, r.phase = False, True, "manual"
            return
        self._auto_charge(r)

    # ---------------- human control
    def hold(self, name):
        """A human takes a robot (keyboard driving): it gets no decisions; applied once its current activity ends."""
        r = self.robots[name]
        if r.manual or r.manual_pending:
            return
        if r.task is None and r.phase is None and not r.fault:
            r.manual, r.phase = True, "manual"
        else:
            r.manual_pending = True

    def release(self, name, node=None, distance=0.0):
        """Hand a held robot back at graph node `node` (where the human left it) after driving `distance` metres."""
        r = self.robots[name]
        r.manual_pending = False
        if not r.manual:
            return
        self._drain(r)
        r.manual, r.phase = False, None
        if node is not None:
            r.node = r.src = r.dest = node
        r.dist += distance
        if r.battery is not None:
            r.battery = max(0.0, r.battery - distance * r.spec["battery"]["per_metre"])
        self.decided.discard(name)
        self._auto_charge(r)

    # ---------------- energy
    def _drain(self, r):
        if r.battery is None:
            return
        if r.phase not in ("charging", "charge_queue"):
            r.battery = max(0.0, r.battery - r.spec["battery"]["per_second"] * (self.t - r.bt))
            r.min_battery = min(r.min_battery, r.battery)
        r.bt = self.t

    def _battery_ok(self, r, node):
        b = r.spec["battery"]
        if not b:
            return True
        home = min((self.F.dist[node].get(c, 1e18) for c in b["chargers"]), default=0.0)
        d = self.F.dist[r.node].get(node, 1e18)
        need = b["per_metre"] * (d + home) + 2 * b["per_service"] + b["per_second"] * (d / r.spec["speed"] + 120.0)
        return r.battery - need >= (b["reserve"] if b["auto"] else 0.0) - 1e-9

    def _auto_charge(self, r):
        """Battery below `low` while idle -> go to the nearest charger (queue there if it is taken)."""
        b = r.spec["battery"]
        if not b or not b["auto"] or r.fault or r.task is not None or r.phase is not None:
            return
        self._drain(r)
        if r.battery <= b["low"] + 1e-9:
            return self._go_charge(r)
        if b["per_second"] > 0:
            r.lowtok += 1
            self._push(self.t + (r.battery - b["low"]) / b["per_second"] + 1e-6, "lowbat", (r.name, r.lowtok))

    def _go_charge(self, r):
        cs = [c for c in self.bat["chargers"] if self.F.dist[r.node].get(c, float("inf")) < float("inf")]
        if not cs:
            return
        c = min(cs, key=lambda c: ((self.charger_at[c] is not None) + len(self.chargeq[c]), self.F.dist[r.node][c]))
        if r.since is None:
            r.since = self.t
        self._travel(r, c, ("charge", None, c))

    def _start_charge(self, r, c):
        self._drain(r)
        if self.charger_at[c] is None:
            self.charger_at[c], r.phase, r.t0 = r.name, "charging", self.t
            b = r.spec["battery"]
            self._activity("charge", r.name, {"fixed": max(1.0, (b["capacity"] - r.battery) / b["charge_rate"])})
        else:
            r.phase = "charge_queue"
            self.chargeq[c].append(r.name)

    def _charge_done(self, name):
        r = self.robots[name]
        c = r.task[2]
        r.battery, r.bt = r.spec["battery"]["capacity"], self.t
        r.t_charge += self.t - r.t0
        r.charges += 1
        self.charger_at[c] = None
        if self.chargeq[c]:
            self._start_charge(self.robots[self.chargeq[c].pop(0)], c)
        self._idle(r)

    # ---------------- decisions
    def legal(self, agent):
        """Legal actions of an agent now: ('wait',), ('release', job, route), ('pickup', job), ('deliver', station), ('move', node), ('charge',)."""
        out = [("wait",)]
        if agent == DISPATCHER:
            for jid in self.backlog:
                j = self.jobs[jid]
                if self.stock is not None and self.stock.get(j.product, 1) <= 0:
                    continue
                for ri, route in enumerate(self.F.products[j.product]["routes"]):
                    if any(self.cap_in(s) for s in self.cands(route[0])):
                        out.append(("release", jid, ri))
            for s in self.F.reconfigurable:
                st = self.stations[s]
                if st.cur is None and not st.q and st.reserved == 0 and not st.down and st.setup is None and not st.service and st.wait is None and st.due_service is None:
                    out += [("setup", s, c) for c in st.spec["configs"] if c != st.config]
            return out
        r = self.robots[agent]
        if r.task is not None or r.fault or r.manual:
            return out
        self._drain(r)
        if r.carry is None:
            for j in self.jobs.values():
                s = self.pickup_station(j)
                if s is not None and self._battery_ok(r, self.stations[s].spec["node"]):
                    out.append(("pickup", j.id))
            out += [("move", n) for n in self.F.move_targets if n != r.node]
            b = r.spec["battery"]
            if b and r.battery < b["capacity"] - 1e-6:
                out.append(("charge",))
        else:
            j = self.jobs[r.carry]
            for s in list(self.targets(j)) + self.F.buffers:
                node = self.stations[s].spec["node"]
                if self.cap_in(s) and self.F.dist[r.node].get(node, float("inf")) < float("inf") and self._battery_ok(r, node) and not (self.stations[s].spec["kind"] == "buffer" and j.at == s):
                    out.append(("deliver", s))
        return out

    def act(self, agent, a):
        a = tuple(a)
        if a not in self.legal(agent):
            raise ValueError("illegal action {} for {} at t={:.1f}".format(a, agent, self.t))
        self.decided.add(agent)
        self.stats["decisions"] += 1
        if a[0] == "wait":
            return
        self.progressed = True
        if a[0] == "release":
            j = self.jobs[a[1]]
            j.route, j.route_id, j.k, j.t_release = self.F.products[j.product]["routes"][a[2]], a[2], 0, self.t
            self.backlog.remove(j.id)
            if self.stock is not None and j.product in self.stock:
                self.stock[j.product] -= 1
            s = min((x for x in self.cands(j.route[0]) if self.cap_in(x)), key=lambda x: (len(self.stations[x].q), x))
            self.stations[s].q.append(j)
            j.where, j.at = "queue", s
            self._log(s, "IN", j)
            self._try_start(s)
        elif a[0] == "setup":
            st = self.stations[a[1]]
            st.setup, st.wait = a[2], "setup"
            self._assign_tech()
        else:
            r = self.robots[agent]
            if a[0] == "pickup":
                j = self.jobs[a[1]]
                j.claim = agent
                self._travel(r, self.stations[j.at].spec["node"], ("pickup", j.id, j.at))
            elif a[0] == "deliver":
                self.stations[a[1]].reserved += 1
                self._travel(r, self.stations[a[1]].spec["node"], ("deliver", r.carry, a[1]))
            elif a[0] == "move":
                self._travel(r, a[1], ("move", a[1], None))
            elif a[0] == "charge":
                self._go_charge(r)
        self._snap()

    def pending(self):
        """Agents that must decide now (agents whose only legal action is wait are skipped)."""
        out = []
        if DISPATCHER not in self.decided and (self.backlog or self.F.reconfigurable) and len(self.legal(DISPATCHER)) > 1:
            out.append(DISPATCHER)
        for n, r in self.robots.items():
            if n not in self.decided and r.task is None and len(self.legal(n)) > 1:
                out.append(n)
        return out

    @property
    def done(self):
        return self.completed >= self.n_target

    def next_decision(self):
        """Advance time until some agent must decide; return its name, or None (done, failed, or waiting for an external backend)."""
        while not self.done and self.failed is None:
            p = self.pending()
            if p:
                return p[0]
            if not self.ev:
                if self.progressed and self.decided:
                    # Agents that waited earlier at this instant are asked again: another agent's action since then
                    # (e.g. a robot taking a tray out of a full station) may have made one of their actions possible.
                    self.progressed = False
                    self.decided.clear()
                    continue
                if not self.external_pending:
                    self.failed = "stalled" if any(len(self.legal(a)) > 1 for a in [DISPATCHER] + list(self.robots)) else "deadlock"
                return None
            if self.horizon is not None and self.ev[0][0] > self.horizon:
                self.failed = "horizon"
                return None
            if self.mode == "external" and self.ev[0][0] > self.t + 1e-9:
                return None
            self._pop()
            while self.ev and self.ev[0][0] <= self.t + 1e-9:
                self._pop()
            self.decided.clear()
            self.progressed = False
            self._snap()
        return None

    def run(self, policy, max_decisions=10 ** 6):
        """Run to the end with policy(sim, agent, legal_actions) -> action. A policy with a `queue_rule` attribute also
        sets how machines pick from their queues (unless the Sim was given one)."""
        if self.queue_rule == "fifo" and getattr(policy, "queue_rule", None) is not None:
            self.queue_rule = policy.queue_rule
        for _ in range(max_decisions):
            a = self.next_decision()
            if a is None:
                break
            self.act(a, policy(self, a, self.legal(a)))
        return self.metrics()

    # ---------------- logs and metrics
    def _log(self, s, what, j):
        tag = self.F.stations[s].get("tag")
        if tag and what in self.F.stations[s].get("log", ["IN", "OUT"]):
            self.rfid.append({"t": round(self.t, 3), "event": "{} {}".format(tag, what), "job": j.id, "product": j.product})

    def _snap(self):
        if not self.record:
            return
        st = {n: (self.state(n), [self.jobs[x.id].product for x in s.q], [x.product for x in s.out], s.cur.product if s.cur else None) for n, s in self.stations.items()}
        rb = {n: (r.src, r.dest, r.t0, r.t1, r.phase, self.jobs[r.carry].product if r.carry is not None else None) for n, r in self.robots.items()}
        self.trace.append({"t": self.t, "stations": st, "robots": rb, "done": self.completed, "late": sum(1 for j in self.jobs.values() if j.t_done is not None and j.t_done > j.due + 1e-9)})

    def state(self, n):
        """Station state: down, repair_wait, setup, service, blocked, busy or idle."""
        s = self.stations[n]
        if s.down:
            return "down"
        if s.setup is not None:
            return "setup"
        if s.service or s.wait == "service":
            return "service"
        return "blocked" if (s.cur is not None and s.done) else "busy" if s.cur is not None else "idle"

    def metrics(self):
        fin = [j for j in self.jobs.values() if j.where == "done"]
        T = self.t if self.t > 0 else 1.0
        tard = [max(0.0, j.t_done - j.due) for j in fin]
        return {"success": self.done, "failure": self.failed, "makespan": round(self.t, 3) if self.done else None, "completed": self.completed, "orders": self.n_target,
                "on_time": sum(1 for x in tard if x <= 1e-9) / max(1, self.n_target), "total_tardiness": round(sum(tard), 3), "max_tardiness": round(max(tard, default=0.0), 3),
                "mean_flow_time": round(float(np.mean([j.t_done - j.arrival for j in fin])), 3) if fin else None,
                "station_utilization": {n: round(s.busy / T, 4) for n, s in self.stations.items() if s.spec["kind"] in PROCESSING},
                "station_blocked": {n: round((s.blocked + ((self.t - s.bsince) if s.bsince is not None else 0)) / T, 4) for n, s in self.stations.items() if s.spec["kind"] in PROCESSING},
                "robot_utilization": {n: round(r.busy / T, 4) for n, r in self.robots.items()}, "robot_distance": {n: round(r.dist, 2) for n, r in self.robots.items()},
                "rejects": self.stats["rejects"], "scraps": self.stats["scraps"], "breakdowns": self.stats["breakdowns"], "grasp_failures": self.stats["grasp_failures"], "decisions": self.stats["decisions"],
                "weighted_tardiness": round(sum(j.weight * max(0.0, j.t_done - j.due) for j in fin), 3), "rush_on_time": (sum(1 for j in fin if j.rush and j.t_done <= j.due + 1e-9) / max(1, sum(1 for j in self.jobs.values() if j.rush))) if any(j.rush for j in self.jobs.values()) else None,
                "station_down": {n: round(s.t_down / T, 4) for n, s in self.stations.items() if s.t_down > 0}, "setups": self.stats["setups"], "setup_time": round(sum(s.t_setup for s in self.stations.values()), 3),
                "services": self.stats["services"], "service_time": round(sum(s.t_service for s in self.stations.values()), 3), "jams": self.stats["jams"], "replenishments": self.stats["replenishments"],
                "robot_faults": self.stats["robot_faults"], "handovers": self.stats["handovers"], "handler_wait": round(self.stats["handler_wait"], 3), "robot_fault_time": {n: round(r.t_fault, 3) for n, r in self.robots.items() if r.t_fault > 0},
                "charges": {n: r.charges for n, r in self.robots.items()} if self.bat else None, "charge_time": {n: round(r.t_charge, 3) for n, r in self.robots.items()} if self.bat else None,
                "min_battery": {n: round(r.min_battery, 2) for n, r in self.robots.items()} if self.bat else None}
