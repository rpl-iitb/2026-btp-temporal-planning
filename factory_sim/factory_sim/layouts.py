"""Generated layouts: a plant of several production lines for the large lane-traffic instances.

    plant(lines)  ->  config dict (like configs/lab.yaml)

Line i is a copy of the reference lab cell (its corridor k1..k9 with the CNC, assembly, conveyor and inspection
stations along it), shifted 14 m up per line; station and node names get the suffix _L<i>. Three cross aisles
(at k1, k5 and k9) join neighbouring lines, so the corridor graph has real intersections. Every line has its own
kitting, but there is one store (the sink), on line 0: every finished tray crosses to line 0. A route step can use
the station of any line (a job may change lines between steps), except inside the conveyor cell, whose transfers stay
on their line. Every junction has a bay next to it (a pocket robots can back into); the robots start in bays.
"""
import copy

from .config import read

LINE_GAP = 14.0
CROSS = ("k1", "k5", "k9")      # corridor nodes where cross aisles join the lines


def _sfx(name, i):
    return "{}_L{}".format(name, i)


def plant(lines=2, base="lab"):
    lab = read(base)
    if lines < 1:
        raise ValueError("a plant needs at least one line")
    cfg = {k: copy.deepcopy(v) for k, v in lab.items() if k not in ("nodes", "edges", "stations", "transfers", "products", "name")}
    cfg["name"] = "plant_{}_lines".format(lines)
    shared = {"store"}
    nodes, edges, stations, transfers, bays, park = {}, [], {}, [], [], []
    for i in range(lines):
        dy = LINE_GAP * i
        ren = {n: (n if n in shared and i == 0 else _sfx(n, i)) for n in lab["nodes"]}
        for n, (x, y) in lab["nodes"].items():
            if n in shared and i > 0:
                continue
            nodes[ren[n]] = [x, y + dy]
        for a, b in lab["edges"]:
            if (a in shared or b in shared) and i > 0:
                continue
            edges.append([ren[a], ren[b]])
        for s, sp in lab["stations"].items():
            if s in shared and i > 0:
                continue
            sp = copy.deepcopy(sp)
            name = ren.get(s, s if (s in shared and i == 0) else _sfx(s, i))
            if sp.get("node"):
                sp["node"] = ren[sp["node"]]
            if sp.get("rework_to"):
                sp["rework_to"] = _sfx(sp["rework_to"], i)
            for c in (sp.get("configs") or {}).values():
                if c and c.get("replaces"):
                    c["replaces"] = _sfx(c["replaces"], i)
            if sp.get("tag"):
                sp["tag"] = "{} L{}".format(sp["tag"], i)
            if sp["kind"] == "kitting":
                sp["slots_in"] = max(sp.get("slots_in", 1), 4)
            stations[name] = sp
        for a, b, m in lab.get("transfers", []):
            transfers.append([_sfx(a, i), _sfx(b, i), copy.deepcopy(m)])
        for k in ("k2", "k5", "k8") + (("k1",) if i > 0 else ()):   # parking bays along the line (k1: no store on lines > 0)
            p = _sfx("park_" + k, i)
            x, y = lab["nodes"][k]
            nodes[p] = [x + 0.5, y + dy - 1.0]
            edges.append([ren[k], p])
            bays.append(p)
            park.append(p)
        if i + 1 < lines:                      # cross aisles to the next line, each with a bay
            for k in CROSS:
                x, y = lab["nodes"][k]
                v = "x{}_{}_{}".format(k, i, i + 1)
                nodes[v] = [x, y + dy + LINE_GAP / 2]
                b = "bay_" + v
                nodes[b] = [x + 0.8, y + dy + LINE_GAP / 2]
                edges += [[_sfx(k, i), v], [v, _sfx(k, i + 1)], [v, b]]
                bays.append(b)
                park.append(b)
    for p, d in lab["products"].items():
        routes = []
        for r in d["routes"]:
            routes.append([sorted({_sfx(s, i) for s in ([x] if isinstance(x, str) else x) for i in range(lines)}) for x in r])
        stations_ok = all(s in stations for r in routes for opts in r for s in opts)
        assert stations_ok, p
        cfg.setdefault("products", {})[p] = dict(copy.deepcopy(d), routes=routes)
    r = cfg["robots"]
    r["start"], r["parking"] = list(park), list(park)
    if r.get("battery"):
        r["battery"]["chargers"] = [_sfx(c, i) for c in r["battery"].get("chargers", []) for i in range(lines)]
    cfg.update({"nodes": nodes, "edges": edges, "stations": stations, "transfers": transfers, "traffic": {"mode": "lanes", "bays": bays}})
    return cfg
