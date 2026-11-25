"""V12 / V13's kitchen side (the domain module of algorithm/v13.py): the rule-based multi-cook dispatcher
(kitchen_dispatcher.py) as option generator and DAgger expert, the 40 option features, duration noise and the kitchen
data (data/, make_data.py; v3_* = 3-4 cooks).

    python -m algorithm.v13 --domain overcooked_test.kitchen_v13 train
    python -m algorithm.v13 --domain overcooked_test.kitchen_v13 eval --split v3_test --seeds 5
"""
import json
import os
from collections import Counter

import numpy as np

import overcooked_test  # noqa: F401  (paths)
from overcooked_test import DATA
from overcooked_test.kitchen_dispatcher import Dispatcher, View  # noqa: F401  (part of the domain protocol)
from algorithm import v10
from algorithm.menv import Noise

AGENT = "cook"
SPLITS = ("v3_train", "v3_val")
V3 = DATA
NOISE = Noise({"move": 0.25, "turn": 0.25, "chop": 0.2, "cook": 0.1, "wash-plate": 0.2, "scrub-pot": 0.2,
               "diner-eats": 0.3}, default=0.1)
NR = 17
NF = NR + 23
_DOM = {}


def _domain_text():
    if "t" not in _DOM:
        _DOM["t"] = open(os.path.join(V3, "domain.pddl")).read()
    return _DOM["t"]


def task(split, name):
    return v10.Task(v10.Domain(_domain_text()), v10.Problem(open(os.path.join(V3, split, name, "problem.pddl")).read()))


def load_split(split):
    """-> [(name, v10 Task, certificate makespan, meta)]"""
    meta = json.load(open(os.path.join(V3, split, "dataset.json")))
    return [(inst["instance_id"], task(split, inst["instance_id"]), inst["certificate_makespan"], inst) for inst in meta["instances"]]


def features(D, V, c, opts):
    """[n options, NF] feature matrix."""
    rules = D.RULES
    t = V.t
    free = D.free_hands(V)
    others_free = [cc for cc in free if cc != c]
    hand = [c not in V.raw and c not in V.chop and c not in V.plate, c in V.raw, c in V.chop,
            c in V.plate and V.plate[c] in V.clean and V.plate[c] not in V.soup,
            c in V.plate and V.plate[c] in V.soup, c in V.plate and V.plate[c] in V.dirty]
    target, empty, new = D.targets(V)
    intents = Counter()
    for cc in V.at:
        if cc != c:
            st = D.intent(V, cc, target, D.needs(V, target, empty, new)[0])
            if st is not None:
                intents[st] += 1
    glob_f = [len(free) / 4.0, len(D.holders(V)) / 4.0, len(V.pending) / 10.0, len(V.on_stove) / 4.0,
              len(V.pot_dirty) / 4.0, sum(1 for pl in V.dirty) / 4.0, t / 400.0] + [float(h) for h in hand]
    rank = Counter()
    X = np.zeros((len(opts), NF), np.float32)
    for j, o in enumerate(opts):
        x = X[j]
        x[rules.index(o["rule"])] = 1.0
        st = o["station"]
        x[NR] = min(o["dist"], 60.0) / 20.0
        x[NR + 1] = rank[o["rule"]] / 3.0
        rank[o["rule"]] += 1
        inf = o["info"]
        if "ready" in inf:
            x[NR + 2] = float(np.clip((inf["ready"] - t) / 20.0, -2, 3))
            x[NR + 3] = 1.0 if inf.get("lit") else 0.0
        if "deadline" in inf and inf["deadline"] < 1e8:
            x[NR + 4] = float(np.clip((inf["deadline"] - t) / 100.0, -1, 3))
        if "window" in inf:
            x[NR + 5] = inf["window"] / 30.0
        if "missing" in inf:
            x[NR + 6] = inf["missing"] / 3.0
        if "short" in inf:
            x[NR + 7] = 1.0 if inf["short"] else 0.0
        if st is not None:
            od = min((D.dist_to(V.at.get(cc), st) for cc in others_free), default=60.0)
            x[NR + 8] = float(np.clip((od - o["dist"]) / 20.0, -3, 3))
            x[NR + 9] = intents[st] / 2.0
        x[NR + 10: NR + 10 + len(glob_f)] = glob_f
    return X




def agent_travel(D, V, agent, station):
    """V14: travel time of another cook to a station (compete edges of the event graph)."""
    return D.dist_to(V.at.get(agent), station)


def expert_index(D, V, c, legal, opts):
    k = D.decide(V, c, legal)
    from overcooked_test.kitchen_dispatcher import WAIT
    if k is WAIT or k is None:
        if not (k is WAIT and D.soon(V, c)):
            q = D.park(V, c, legal)
            if q is None and k is None:
                q = D.yield_tile(V, c, legal)
            if q is not None:
                k = q
            else:
                k = None
        else:
            k = None
    for j, o in enumerate(opts):
        if o["k"] == k:
            return j
    return len(opts) - 1                          # wait
