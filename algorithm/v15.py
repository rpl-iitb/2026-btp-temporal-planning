"""T-iFLAX V15: V14 (tropical event-graph network) with size-invariant inputs, for policies that must work on
factories much larger than the ones they were trained on.

Why. V14 trained on 2-4 robot factories saw, on a 12-robot plant, inputs far outside its training range: travel
delays 5x longer in its time unit, 4x the candidates, 13x the "another robot competes" edges, an option-rank feature
of 7.7 (training: 1.7), time-to-deadline features of 126 (training: 52); it missed the deadlines on almost every run.

What changes (the network, TENet, is V14's; only the event graph's numbers and the training differ):
  time scale   one per task from the run's reference makespan (horizon = 4 x reference: scale = reference / 50),
               so delays, travel, slack and deadlines are fractions of the job, whatever the factory's size
               (V14: 10 x the median action duration, the same for a small lab and a large plant)
  bounded      every count or time that grows with size is a fraction or clipped: option rank within its rule
  features     -> rank / (options of that rule - 1); running activities / (2 x agents); time-to-deadline / 10;
               the option features' travel and clock -> in the task's time scale, clipped
  competition  each candidate task gets "another agent competes" edges only from the K = 3 other agents closest to
               its station, with their travel time when the option generator reports it (D.travel_time); V14:
               one edge from every other agent, travel unknown
  training     V13's (DAgger with the dispatcher, then PPO), on a curriculum that includes larger factories (the
               domain module's SPLITS), warm-started from a trained V14 (--init): --init V14_hard
Ablations: --no_maxplus, --no_opt_feats as V14.

    python -m algorithm.v15 --domain factory_test.factory_v15 train --name V15_hard --init V14_hard
"""
import math
import os
from collections import Counter

import numpy as np
import torch

from . import v13, v14
from .v14 import (BIG, CEND, CSTART, CTASK, DEADLINE, E_COMPETE, E_HORIZON, E_NEEDS, E_NOW, E_READY, E_RELEASE,
                  E_SERVE, E_SERVE_RUN, E_STATION, E_STEP, E_TRAVEL, END, NOW, NX, OBJ, RUN, _cv, _h)

K_COMPETE = 3
CONFIG = dict(v14.CONFIG)
_CACHE = {}


def _static(env):
    """Per task (cached): object -> type, time scale (reference makespan / 50; V14's when there is no horizon)."""
    key = (id(env.T), env.horizon)
    if key not in _CACHE:
        T = env.T
        objs = dict(T.dom.constants); objs.update(T.prob.objects)
        if env.horizon:
            sc = env.horizon / 4.0 / 50.0
        else:
            ds = [a.lo for i, a in enumerate(T.actions) if env.agent_of[i] is not None and not a.instant and a.lo > 0]
            sc = 10.0 * float(np.median(ds) if ds else 1.0)
        _CACHE.clear()
        _CACHE[key] = (objs, sc)
    return _CACHE[key]


def encode(env, S, D, V, c, legal, opts):
    """V14's event graph of one decision with size-invariant numbers (see the module docstring)."""
    T = env.T
    objs, sc = _static(env)
    now = S.t
    agents = set(env.agents)
    typ, voc, X, t0, l0, v0 = [], [], [], [], [], []
    src, dst, et, ed, tm = [], [], [], [], []

    def node(tp, word, t=0.0, l=None, var=0.0, f=None):
        x = np.zeros(NX, np.float32)
        for k, val in (f or {}).items():
            x[k] = val
        x[0] = min(t, 20.0)
        typ.append(tp); voc.append(_h(word)); X.append(x)
        t0.append(t); l0.append(t + BIG if l is None else l); v0.append(var)
        return len(typ) - 1

    def edge(u, v, k, d=0.0, var=0.0, temporal=True, known=True):
        src.append(u); dst.append(v); et.append(k); ed.append((d, math.sqrt(var), float(known))); tm.append(temporal)

    ob = {}

    def obj(name):
        if name not in ob:
            ob[name] = node(OBJ, "o:" + objs.get(name, "object"), f={1: float(name == c), 2: float(name in agents)})
        return ob[name]

    def to_go(dt):
        return min(dt / sc / 10.0, 5.0)

    busy = {env.agent_of[i] for i, _, _, _ in S.run if env.agent_of[i] is not None}
    unmet = sum(1 for g in T.goals if g not in S.facts) / max(1, len(T.goals))
    n_now = node(NOW, "now", f={15: (len(agents) - len(busy)) / max(1, len(agents)), 16: min(len(S.run) / (2.0 * max(1, len(agents))), 3.0),
                                17: unmet, 18: min(now / sc / 50.0, 3.0)})
    n_end = node(END, "end")
    runs, run_dls = [], []
    for i, st, lo, hi in S.run:
        a = T.actions[i]
        fixed = a.lo == a.hi
        nom_end = st + a.lo
        rem = max(nom_end - now, 0.0)
        var = (_cv(a.name) * a.lo) ** 2 * (min(1.0, rem / a.lo) if a.lo > 0 else 0.0) if fixed else 0.0
        r = node(RUN, "a:" + a.name, rem / sc, var=var / sc ** 2,
                 f={1: float(env.agent_of[i] == c), 4: min((now - st) / sc, 5.0), 5: min(a.lo / sc, 5.0), 6: float(not fixed),
                    7: float(env.agent_of[i] is not None), 8: float(fixed and now > nom_end + 1e-6)})
        for x in a.args:
            if x in objs:
                edge(r, obj(x), E_RELEASE)
        edge(r, n_end, E_HORIZON)
        runs.append((r, set(a.args) - agents))
        if not fixed and hi > now:
            d = node(DEADLINE, "d:run:" + a.name, 0.0, l=(hi - now) / sc, f={13: 1.0, 14: to_go(hi - now)})
            edge(r, d, E_SERVE_RUN)
            run_dls.append((d, set(a.args) - agents))
    for ag in env.agents:
        obj(ag)
    dls = []
    for f, tl in T.window.items():
        if f in S.facts and tl > now:
            fa = T.facts[f]
            d = node(DEADLINE, "d:" + fa[0], 0.0, l=(tl - now) / sc, f={13: 1.0, 14: to_go(tl - now)})
            dls.append((d, set(fa[1:]) - agents))
    for r, args in runs:
        for d, os_ in dls:
            if args & os_:
                edge(r, d, E_SERVE_RUN)
    travel = getattr(D, "travel_time", None)
    others = [ag for ag in env.agents if ag != c]
    cv_move = _cv("move")
    per_rule = Counter(o["rule"] for o in opts)
    rank, extra, cands = Counter(), {}, []
    for o in opts:
        k = o["k"]
        a = T.actions[legal[k]] if k is not None else None
        name = a.name if a is not None else "wait"
        dur = 0.0 if a is None or a.instant else a.lo
        dist = o.get("dist") or 0.0
        dist = dist if dist < 1e5 else 0.0
        st = o.get("station")
        f = {1: 1.0, 9: min(dur / sc, 5.0), 10: min(dist / sc, 5.0), 11: float(a is None),
             12: rank[o["rule"]] / max(1, per_rule[o["rule"]] - 1), 19: float(st is not None)}
        rank[o["rule"]] += 1
        cs = node(CSTART, "a:" + name, f=f)
        ce = node(CEND, "a:" + name, dur / sc, f=f)
        ct = node(CTASK, "r:" + o["rule"], f=f)
        edge(n_now, cs, E_NOW)
        if a is not None:
            for x in a.args:
                if x in objs:
                    edge(obj(x), cs, E_NEEDS)
        edge(cs, ce, E_STEP, dur / sc, (_cv(name) * dur) ** 2 / sc ** 2)
        trav = max(0.0, dist - dur)
        edge(ce, ct, E_TRAVEL, trav / sc, (cv_move * trav) ** 2 / sc ** 2)
        if st is not None and st in objs:
            edge(obj(st), ct, E_STATION)
        info = o.get("info") or {}
        rd = info.get("ready")
        if isinstance(rd, (int, float)) and not isinstance(rd, bool) and rd < 1e8:
            edge(n_now, ct, E_READY, max(0.0, rd - now) / sc)
        concern = (set(a.args) if a is not None else set()) | ({st} if st else set())
        concern |= {val for val in info.values() if isinstance(val, str) and val in objs}
        concern -= agents
        for d, os_ in dls:
            if concern & os_:
                edge(ct, d, E_SERVE)
        if st is not None:
            for d, os_ in run_dls:
                if st in os_:
                    edge(ct, d, E_SERVE)
        dd = info.get("deadline")
        if isinstance(dd, (int, float)) and not isinstance(dd, bool) and now < dd < 1e8:
            key = round(float(dd), 3)
            if key not in extra:
                extra[key] = node(DEADLINE, "d:option", 0.0, l=(dd - now) / sc, f={13: 1.0, 14: to_go(dd - now)})
            edge(ct, extra[key], E_SERVE)
        if st is not None:              # competition: the K other agents closest to the task's station
            near = []
            for ag in others:
                tv = travel(V, ag, st) if travel is not None else None
                near.append((tv if tv is not None and tv < 1e5 else float("inf"), ag))
            for tv, ag in sorted(near)[:K_COMPETE]:
                known = tv < float("inf")
                edge(obj(ag), ct, E_COMPETE, min(tv / sc, 5.0) if known else 0.0, temporal=False, known=known)
        cands.append((cs, ce, ct))
    X = np.stack(X)
    for u, v, k in zip(src, dst, et):
        if k == E_RELEASE:
            X[v, 3] += 1.0 / 3.0
    np.minimum(X[:, 3], 3.0, out=X[:, 3])
    of = np.asarray(v13.features(D, V, c, opts), np.float32)
    nr = len(getattr(D, "RULES", ()))
    if nr and of.shape[1] > nr + 8:      # the robot dispatcher's travel and clock columns, in the task's time scale
        of[:, nr] = np.minimum([(o.get("dist") or 0.0) / sc for o in opts], 5.0)
        of[:, nr + 8] = min(now / sc / 50.0, 3.0)
    return {"x": X.astype(np.float16), "typ": np.array(typ, np.int8), "voc": np.array(voc, np.int16),
            "t0": np.array(t0, np.float32), "l0": np.array(l0, np.float32), "v0": np.array(v0, np.float32),
            "src": np.array(src, np.int32), "dst": np.array(dst, np.int32), "et": np.array(et, np.int8),
            "ed": np.array(ed, np.float32).reshape(-1, 3), "tm": np.array(tm, bool),
            "cand": np.array(cands, np.int32).reshape(-1, 3), "now": n_now, "end": n_end, "of": of.astype(np.float16)}


class TENet15(v14.TENet):
    """V14's network with V15's encoder."""

    @staticmethod
    def encode(env, S, D, V, c, legal, opts):
        return encode(env, S, D, V, c, legal, opts)


INIT = None      # a trained V14 / V15 to start training from (--init)


def make_policy(path=None):
    import json
    cfg = dict(CONFIG)
    src = path if path is not None else INIT
    if src is not None and os.path.exists(src + ".json"):
        cfg.update(json.load(open(src + ".json")))
    cfg["nf"] = cfg.get("nf") or v13.DOMAIN.NF
    pol = TENet15(**cfg)
    if path is None and INIT is not None:
        pol.load_state_dict(torch.load(INIT + ".pt", weights_only=True))
    return pol


def load(path):
    pol = make_policy(path)
    pol.load_state_dict(torch.load(path + ".pt", weights_only=True))
    pol.eval()
    return pol


def install(domain=None, **config):
    if domain is not None:
        v13.use(domain)
    CONFIG.update(config)
    v13.ENCODE, v13.POLICY = encode, make_policy
    v13.OUT = os.path.join(os.path.dirname(v13.MODELS), "v15")


def main():
    global INIT
    ap, t, e = v13.parser("V15")
    t.add_argument("--init", default=None, help="start from this trained V14 / V15 (a model name in the models folder)")
    t.add_argument("--no_maxplus", action="store_true")
    t.add_argument("--no_opt_feats", action="store_true")
    for p in (t, e):
        p.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    if a.cmd == "train":
        cfg = {}
        if a.no_maxplus:
            cfg["maxplus"] = False
        if a.no_opt_feats:
            cfg["opt_feats"] = False
        install(a.domain, **cfg)
        if a.init:
            INIT = os.path.join(v13.MODELS, a.init)
    else:
        install(a.domain)
    torch.set_num_threads(a.threads)
    {"train": v13.cmd_train, "eval": v13.cmd_eval}[a.cmd](a)


if __name__ == "__main__":
    main()
