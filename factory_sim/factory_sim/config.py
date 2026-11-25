"""Factory description: load and validate a YAML/dict config into a static Factory model."""
import copy
import heapq
import os

import yaml

from . import robots, timing

KINDS = {"kitting", "machine", "assembly", "inspection", "port", "buffer", "sink"}
PROCESSING = {"kitting", "machine", "assembly", "inspection", "port"}
HERE = os.path.dirname(os.path.abspath(__file__))


def builtin(name):
    return os.path.join(HERE, "configs", name + ".yaml")


def merge(base, over):
    """Deep merge (dicts merge recursively, everything else is replaced)."""
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def read(src):
    """Config dict from a dict, a YAML path or a builtin name; `extends: <config>` inherits and deep-merges."""
    if isinstance(src, dict):
        cfg = copy.deepcopy(src)
    else:
        path = src if os.path.exists(src) else builtin(src)
        with open(path) as f:
            cfg = yaml.safe_load(f)
    base = cfg.pop("extends", None)
    return merge(read(base), cfg) if base else cfg


class Factory:
    """Static part of a factory: floor graph, stations, products, routes, transfers, operators, robots."""

    def __init__(self, cfg):
        cfg = read(cfg)
        self.cfg, self.name = cfg, cfg.get("name", "factory")
        self.nodes = {k: tuple(map(float, v)) for k, v in cfg["nodes"].items()}
        self.adj = {n: {} for n in self.nodes}
        for a, b in cfg["edges"]:
            d = ((self.nodes[a][0] - self.nodes[b][0]) ** 2 + (self.nodes[a][1] - self.nodes[b][1]) ** 2) ** 0.5
            self.adj[a][b] = self.adj[b][a] = d
        self.prev = {}
        self.dist = {n: self._dijkstra(n) for n in self.nodes}
        self.stations = {}
        for name, s in cfg["stations"].items():
            s = dict(s)
            if s.get("kind") not in KINDS:
                raise ValueError("station {}: kind must be one of {}".format(name, sorted(KINDS)))
            s.setdefault("node", name)
            s.setdefault("slots_in", 10 ** 6 if s["kind"] in ("sink",) else 1)
            s.setdefault("slots_out", 0 if s["kind"] in ("buffer", "sink") else 1)
            s.setdefault("time", {"fixed": 0})
            s.setdefault("tag", None)
            s["dock_capacity"] = s.get("dock_capacity", cfg.get("congestion", {}).get("dock_capacity", 1))
            if s["node"] not in self.nodes:
                raise ValueError("station {}: node {} not in nodes".format(name, s["node"]))
            timing.mean(s["time"])
            if s.get("configs"):
                s["configs"] = {c: dict(v or {}) for c, v in s["configs"].items()}
                s.setdefault("config", next(iter(s["configs"])))
                s.setdefault("changeover", {"fixed": 0})
                if s["config"] not in s["configs"]:
                    raise ValueError("station {}: initial config {} not in configs".format(name, s["config"]))
            self.stations[name] = s
        for name, s in self.stations.items():
            for c, v in s.get("configs", {}).items():
                if v.get("replaces") and v["replaces"] not in self.stations:
                    raise ValueError("station {} config {}: replaces unknown station {}".format(name, c, v["replaces"]))
        self.operators = {k: (v["count"] if isinstance(v, dict) else v) for k, v in cfg.get("operators", {}).items()}
        self.breaks = {k: v for k, v in cfg.get("operator_breaks", {}).items() if k in self.operators}
        self.technician = cfg.get("technician") if cfg.get("technician") in self.operators else None
        for n, s in self.stations.items():
            if s.get("operator") and s["operator"] not in self.operators:
                raise ValueError("station {}: unknown operator pool {}".format(n, s["operator"]))
        self.transfers = {(a, b): m for a, b, m in cfg.get("transfers", [])}
        self.products = {}
        for p, d in cfg["products"].items():
            routes = [[[x] if isinstance(x, str) else list(x) for x in r] for r in d["routes"]]
            for r in routes:
                for opts in r:
                    for s in opts:
                        if s not in self.stations or self.stations[s]["kind"] not in PROCESSING:
                            raise ValueError("product {}: {} is not a processing station".format(p, s))
            self.products[p] = {"routes": routes, "color": tuple(d.get("color", (200, 200, 200)))}
        self.sinks = [n for n, s in self.stations.items() if s["kind"] == "sink"]
        self.buffers = [n for n, s in self.stations.items() if s["kind"] == "buffer"]
        if not self.sinks:
            raise ValueError("config needs a station of kind sink")
        self.kitting = [n for n, s in self.stations.items() if s["kind"] == "kitting"]
        r = dict(cfg.get("robots", {}))
        self.fleet = robots.resolve(r)
        if not self.fleet:
            raise ValueError("the fleet needs at least one robot")
        # fleet-wide summary (used for bounds and nominal times): the fastest robot and the shortest pick + place
        fast = min(self.fleet, key=lambda x: timing.mean(x["load"]) + timing.mean(x["unload"]))
        r.update({k: fast[k] for k in ("load", "unload", "grasp_success", "place_success")})
        r["count"], r["speed"], r["travel_cv"] = len(self.fleet), max(x["speed"] for x in self.fleet), min(x["travel_cv"] for x in self.fleet)
        r["types"] = sorted({x["type"] for x in self.fleet})
        r.setdefault("parking", [])
        r.setdefault("start", [next(iter(self.nodes))])
        self.robots = r
        self.handling = sorted({x["handling"] for x in self.fleet})
        self.handler_pool = r.get("handler_pool", "handler")
        if "handler" in self.handling:
            pools = {s.get("handler", self.handler_pool) for s in self.stations.values()}
            for pool in pools:
                if pool not in self.operators:
                    self.operators[pool] = int(r.get("handlers", 2))
        # handlers walk (floor graph, `handler_speed` m/s) from where they last worked; they start at `handler_home`
        self.handler_speed = float(r.get("handler_speed", 1.0))
        self.handler_home = r.get("handler_home") or (self.stations[self.kitting[0]]["node"] if self.kitting else next(iter(self.nodes)))
        self.slowdown = cfg.get("congestion", {}).get("slowdown", 0.0)
        self.move_targets = list(dict.fromkeys(list(r["parking"]) + [st["node"] for st in self.stations.values() if any(self.dist[n].get(st["node"], float("inf")) < float("inf") for n in r["start"])]))
        self.features = dict({"stochastic": True, "breakdowns": False, "rejects": True, "battery": True, "robot_faults": True, "maintenance": True, "reconfiguration": True, "breaks": True, "materials": True, "jams": True}, **cfg.get("features", {}))
        r["battery"] = self._battery(r.get("battery")) if self.features["battery"] else None
        r["fault"] = r.get("fault") if self.features["robot_faults"] else None
        for x in self.fleet:  # batteries need the site part (`robots.battery`: chargers ...); the type adds the robot part
            x["battery"] = self._battery(x["battery"]) if r["battery"] else None
            x["fault"] = x.get("fault") if self.features["robot_faults"] else None
        self.homogeneous = all(x == self.fleet[0] for x in self.fleet)
        self.chargers = r["battery"]["chargers"] if r["battery"] else []
        for c in self.chargers:
            if c not in self.nodes:
                raise ValueError("charger node {} not in nodes".format(c))
        self.materials = cfg.get("materials") if self.features["materials"] else None
        self.jams = cfg.get("conveyor_jams") if self.features["jams"] else None
        if not self.features["breaks"]:
            self.breaks = {}
        if not self.features["maintenance"]:
            self.technician = None
            for s in self.stations.values():
                s.pop("maintenance", None)
                s.pop("downtime", None)
        self.reconfigurable = [n for n, s in self.stations.items() if s.get("configs") and len(s["configs"]) > 1] if self.features["reconfiguration"] else []
        self._traffic(cfg.get("traffic") or {})

    def _traffic(self, t):
        """Lane traffic (`traffic: {mode: lanes}`): the corridor nodes are junctions that hold one robot at a time; the
        other nodes (station docks, chargers, `bays`) are pockets that hold any number. Robots drive edge by edge and
        reserve the next junction before they enter it, so they cannot pass each other in a corridor (see Sim._hop).
        `junctions` lists them explicitly; default: every corridor node (two or more edges) that is no station dock,
        charger or bay; dead ends are pockets."""
        self.lanes = t.get("mode", "none") == "lanes"
        docks = {s["node"] for s in self.stations.values()}
        bays = set(t.get("bays", [])) | set(self.chargers)
        js = t.get("junctions")
        self.junctions = set(js) if js is not None else {n for n in self.nodes if n not in docks and n not in bays and len(self.adj[n]) >= 2}
        self.pockets = {n for n in self.nodes if n not in self.junctions}
        self.bays = {j: sorted((p for p in self.adj[j] if p in self.pockets), key=lambda p: (self.adj[j][p], p)) for j in self.junctions}
        self.yield_after = float(t.get("yield_after", 120.0))      # s blocked at one junction before a robot backs off
        if self.lanes:
            # pockets are leaves of the floor graph: routes never pass through them
            self.dist, self.prev = {}, {}
            for n in self.nodes:
                self.dist[n] = self._dijkstra(n, through=self.junctions)

    def _battery(self, b):
        """Normalise a battery spec. New form: capacity (%), per_metre, per_second, per_service, low, reserve,
        charge_rate (% / s), chargers, auto. Old form {capacity: metres, charger, charge_time} is still accepted."""
        if not b:
            return None
        b = dict(b)
        old = "charger" in b and "per_metre" not in b
        b.setdefault("capacity", 100.0)
        b.setdefault("per_metre", 1.0 if old else 0.05)
        b.setdefault("per_second", 0.0)
        b.setdefault("per_service", 0.0)
        b.setdefault("low", 0.3 * b["capacity"])
        b.setdefault("reserve", 0.1 * b["capacity"])
        b.setdefault("chargers", [b["charger"]] if "charger" in b else [])
        b.setdefault("charge_rate", b["capacity"] / timing.mean(b.get("charge_time", {"fixed": 600})))
        b.setdefault("auto", not old)
        b.setdefault("charger", b["chargers"][0] if b["chargers"] else None)
        return b

    def covers(self, station, config=None):
        """Stations whose role `station` plays under `config` (its own role, or the role named by `replaces`)."""
        s = self.stations[station]
        c = s.get("configs", {}).get(config or s.get("config"))
        return {c["replaces"]} if c and c.get("replaces") else {station}

    def _dijkstra(self, src, through=None):
        """Shortest distances from src; with `through`, paths only pass through those nodes (src and ends excepted)."""
        d, h, pv = {src: 0.0}, [(0.0, src)], {}
        self.prev[src] = pv
        while h:
            x, n = heapq.heappop(h)
            if x > d[n]:
                continue
            if through is not None and n != src and n not in through:
                continue
            for m, w in self.adj[n].items():
                if x + w < d.get(m, float("inf")):
                    d[m], pv[m] = x + w, n
                    heapq.heappush(h, (x + w, m))
        return d

    def path(self, a, b):
        """Shortest node path from a to b."""
        out, n = [b], b
        while n != a and n in self.prev[a]:
            n = self.prev[a][n]
            out.append(n)
        return out[::-1]

    def travel(self, a, b, speed=None):
        """Nominal travel time between two nodes (inf if unreachable) at `speed` (default: the fleet's fastest robot)."""
        return self.dist[a].get(b, float("inf")) / (speed or self.robots["speed"])

    def reachable(self, station):
        return any(self.dist[n].get(self.stations[station]["node"], float("inf")) < float("inf") for n in self.robots["start"])

    def nominal_flow(self, product, route=0):
        """Nominal time for one job alone: processing + transfers / transport along the route, then to a sink."""
        r = self.products[product]["routes"][route]
        t = sum(min(timing.mean(self.proc_model(s, product)) for s in opts) for opts in r)
        hop = timing.mean(self.robots["load"]) + timing.mean(self.robots["unload"])
        for a, b in zip(r, r[1:] + [self.sinks]):
            t += min((timing.mean(self.transfers[(x, y)]) if (x, y) in self.transfers else self.travel(self.stations[x]["node"], self.stations[y]["node"]) + hop) for x in a for y in b)
        return t

    def proc_model(self, station, product, config=None):
        s = self.stations[station]
        c = s.get("configs", {}).get(config or s.get("config"))
        if c:
            if product in c.get("time_by_product", {}):
                return c["time_by_product"][product]
            if "time" in c:
                return c["time"]
            if c.get("replaces"):
                return self.proc_model(c["replaces"], product)
        return s.get("time_by_product", {}).get(product, s["time"])
