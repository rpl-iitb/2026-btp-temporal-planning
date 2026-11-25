"""T-iFLAX V14: tropical event-graph network (TEGN) for multi-agent task choice under stochastic durations.

What it does
  V13's decision problem, options, expert and training, with a network in which time is structure, not a feature
  (plans/2026-11-15_new_algorithm_ideas.md, section 3). Whenever an agent can act it picks one of the tasks the
  dispatcher offers; the network sees the state as an EVENT GRAPH built from the PDDL state, not from domain code:
    nodes   now; the end event of every running durative action (nominal remaining time; the sampled actual end
            stays hidden); the objects those actions and the candidates use (agents, stations, jobs, ...);
            per candidate task three events: its next step starts / ends, the task reaches its station; deadline
            events (facts a timed literal deletes, a deadline the option reports, or the latest end of a running
            flexible action, e.g. a stove timer whose upper bound is the burn time); the current horizon (end).
    edges   "can happen after, with delay d": running end -> its objects (release), objects -> candidate start
            (needs), start -> end (the step's nominal duration), end -> task (remaining travel), station -> task
            (station free), now -> task (option ready time), task / running end -> deadline (shares an object;
            for a flexible action's own deadline: the task's station is one of its objects), running end -> horizon. Plus non-temporal "compete" edges: other agent -> candidate task, with the
            other agent's travel time when the domain module gives it (agent_travel).
  Message passing = learned max-plus (idea 4): per layer, earliest times
        t_v = max(t0_v, max over in-edges (t_u + d_uv + rho_uv(h_u, h_v, e_uv)))
  iterated `hops` times, and latest times by the dual min-plus pass from the deadlines and the horizon. rho is
  zero-initialised, so the untrained network computes exactly the critical-path method (CPM) on the event graph
  (checked by exact_times); learning adds what CPM ignores (contention, travel, blocking). The soft argmax of the
  in-edges at temperature tau (learned) is each edge's criticality: it weights messages and variances. Completion-time
  variances travel along the soft critical path (idea 10, simplified: variance of the soft-argmax predecessor
  plus the edge's, not Clark's approximation), giving each event a slack z-score and an on-time probability.
  Hidden states are updated with messages weighted by criticality (the tight in-edges dominate), forward and
  backward. Each candidate is scored from its three event nodes, their times, slack, on-time probability, how far
  it extends the horizon, optionally V13's option features, and a deep-set context (as V13).
  training / execution: V13's, unchanged (DAgger with the dispatcher, then PPO, or REINFORCE with --rl reinforce,
  with an imitation regulariser; algorithm/menv.py with random durations). V13 is therefore V14 without the event
  graph.

Ablations (train flags): --no_maxplus (same graph, times only as input features, mean aggregation: "time as a
number on a node"), --no_opt_feats (event graph only, no hand-made option features).
From the ideas list: implemented 2 (candidates as nodes), 4 (max-plus with learned residual), 10 (simplified),
1 partly (activity nodes join all their argument objects); not implemented: 3, 5-9.

What it needs: V13's domain module protocol; optional agent_travel(D, V, agent, station) -> time, TIME_SCALE.

    python -m algorithm.v14 --domain overcooked_test.kitchen_v13 train [--no_maxplus] [--no_opt_feats]
    python -m algorithm.v14 --domain overcooked_test.kitchen_v13 eval --split v3_test --seeds 5
"""
import json
import math
import os
import zlib
from collections import Counter

import numpy as np
import torch
import torch.nn as nn

from . import v13

# node types
NOW, RUN, OBJ, CSTART, CEND, CTASK, DEADLINE, END = range(8)
NT = 8
# edge types
E_RELEASE, E_NEEDS, E_STEP, E_TRAVEL, E_STATION, E_READY, E_SERVE, E_SERVE_RUN, E_HORIZON, E_NOW, E_COMPETE = range(11)
NE = 11
NV = 512          # hashed vocabulary (action schema, object type, option rule, deadline predicate)
NX = 20           # node input features
NTF = 5           # time features per node after propagation
BIG = 50.0        # "no deadline": latest time = earliest + BIG (scaled units)
CONFIG = dict(hidden=64, layers=3, hops=6, maxplus=True, opt_feats=True, tau=0.05)


# ==================== event graph ======================================================================
def _h(word):
    return 1 + zlib.crc32(word.encode()) % (NV - 1)


_CACHE = {}


def _static(env):
    """Per task (cached): object -> type, time scale."""
    key = id(env.T)
    if key not in _CACHE:
        T = env.T
        objs = dict(T.dom.constants); objs.update(T.prob.objects)
        ds = [a.lo for i, a in enumerate(T.actions) if env.agent_of[i] is not None and not a.instant and a.lo > 0]
        sc = getattr(v13.DOMAIN, "TIME_SCALE", None) or 10.0 * float(np.median(ds) if ds else 1.0)
        _CACHE.clear()
        _CACHE[key] = (objs, sc)
    return _CACHE[key]


def _cv(name):
    N = v13.NOISE
    return 0.0 if N is None else N.cv.get(name, N.default)


def encode(env, S, D, V, c, legal, opts):
    """The event graph of one decision of agent c among `opts` -> dict of numpy arrays."""
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

    busy = {env.agent_of[i] for i, _, _, _ in S.run if env.agent_of[i] is not None}
    unmet = sum(1 for g in T.goals if g not in S.facts) / max(1, len(T.goals))
    n_now = node(NOW, "now", f={15: (len(agents) - len(busy)) / max(1, len(agents)), 16: len(S.run) / 10.0,
                                17: unmet, 18: now / sc / 10.0})
    n_end = node(END, "end")
    # running activities: end events (nominal; the sampled end is hidden). A flexible one (duration window, e.g. a
    # stove timer whose upper bound is the burn time) must end by its latest end: a deadline event of its own.
    runs, run_dls = [], []
    for i, st, lo, hi in S.run:
        a = T.actions[i]
        fixed = a.lo == a.hi
        nom_end = st + a.lo
        rem = max(nom_end - now, 0.0)
        var = (_cv(a.name) * a.lo) ** 2 * (min(1.0, rem / a.lo) if a.lo > 0 else 0.0) if fixed else 0.0
        r = node(RUN, "a:" + a.name, rem / sc, var=var / sc ** 2,
                 f={1: float(env.agent_of[i] == c), 4: (now - st) / sc, 5: a.lo / sc, 6: float(not fixed),
                    7: float(env.agent_of[i] is not None), 8: float(fixed and now > nom_end + 1e-6)})
        for x in a.args:
            if x in objs:
                edge(r, obj(x), E_RELEASE)
        edge(r, n_end, E_HORIZON)
        runs.append((r, set(a.args) - agents))
        if not fixed and hi > now:
            d = node(DEADLINE, "d:run:" + a.name, 0.0, l=(hi - now) / sc, f={13: 1.0, 14: (hi - now) / sc})
            edge(r, d, E_SERVE_RUN)
            run_dls.append((d, set(a.args) - agents))
    for ag in env.agents:
        obj(ag)
    # deadlines: facts a timed literal deletes (still true, still ahead)
    dls = []
    for f, tl in T.window.items():
        if f in S.facts and tl > now:
            fa = T.facts[f]
            d = node(DEADLINE, "d:" + fa[0], 0.0, l=(tl - now) / sc, f={13: 1.0, 14: (tl - now) / sc})
            dls.append((d, set(fa[1:]) - agents))
    for r, args in runs:
        for d, os_ in dls:
            if args & os_:
                edge(r, d, E_SERVE_RUN)
    # candidates: start / end of the next step, task at its station
    hook = getattr(v13.DOMAIN, "agent_travel", None)
    others = [ag for ag in env.agents if ag != c]
    cv_move = _cv("move")
    rank, extra, cands = Counter(), {}, []
    for o in opts:
        k = o["k"]
        a = T.actions[legal[k]] if k is not None else None
        name = a.name if a is not None else "wait"
        dur = 0.0 if a is None or a.instant else a.lo
        dist = o.get("dist") or 0.0
        dist = dist if dist < 1e5 else 0.0
        st = o.get("station")
        f = {1: 1.0, 9: dur / sc, 10: dist / sc, 11: float(a is None), 12: rank[o["rule"]] / 3.0, 19: float(st is not None)}
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
        if st is not None:                      # a task at a station a flexible activity holds must meet its end
            for d, os_ in run_dls:
                if st in os_:
                    edge(ct, d, E_SERVE)
        dd = info.get("deadline")
        if isinstance(dd, (int, float)) and not isinstance(dd, bool) and now < dd < 1e8:
            key = round(float(dd), 3)
            if key not in extra:
                extra[key] = node(DEADLINE, "d:option", 0.0, l=(dd - now) / sc, f={13: 1.0, 14: (dd - now) / sc})
            edge(ct, extra[key], E_SERVE)
        for ag in others:
            tv = hook(D, V, ag, st) if (hook is not None and st is not None) else None
            known = tv is not None and tv < 1e5
            edge(obj(ag), ct, E_COMPETE, tv / sc if known else 0.0, temporal=False, known=known)
        cands.append((cs, ce, ct))
    X = np.stack(X)
    for u, v, k in zip(src, dst, et):
        if k == E_RELEASE:
            X[v, 3] += 1.0 / 3.0
    return {"x": X.astype(np.float16), "typ": np.array(typ, np.int8), "voc": np.array(voc, np.int16),
            "t0": np.array(t0, np.float32), "l0": np.array(l0, np.float32), "v0": np.array(v0, np.float32),
            "src": np.array(src, np.int32), "dst": np.array(dst, np.int32), "et": np.array(et, np.int8),
            "ed": np.array(ed, np.float32).reshape(-1, 3), "tm": np.array(tm, bool),
            "cand": np.array(cands, np.int32).reshape(-1, 3), "now": n_now, "end": n_end,
            "of": np.asarray(v13.features(D, V, c, opts), np.float16)}


def exact_times(G):
    """Exact CPM earliest times of an event graph (longest path over temporal edges, no learning): the reference the
    untrained network must reproduce."""
    t = G["t0"].astype(np.float64).copy()
    E = [(u, v, d) for u, v, d, m in zip(G["src"], G["dst"], G["ed"][:, 0], G["tm"]) if m]
    for _ in range(len(t)):
        ch = False
        for u, v, d in E:
            if t[u] + d > t[v] + 1e-9:
                t[v] = t[u] + d; ch = True
        if not ch:
            break
    return t


# ==================== network ==========================================================================
def _mlp(i, h, o, zero=False):
    m = nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, o))
    if zero:
        nn.init.zeros_(m[-1].weight); nn.init.zeros_(m[-1].bias)
    return m


def _smax(vals, idx, n, tau, base):
    """Max of vals grouped by idx, floored by base (exact, so the untrained network is exactly CPM; the gradient
    goes to the critical edge) -> ([n], per-edge soft-argmax weights at temperature tau = criticality)."""
    m = torch.full((n,), -1e9).scatter_reduce(0, idx, vals, "amax", include_self=True)
    z = torch.exp((vals - m.detach()[idx]) / tau)
    s = torch.zeros(n).index_add(0, idx, z)
    out = torch.where(s > 0, torch.maximum(base, m), base)
    return out, z / s[idx].clamp_min(1e-30)


def _seg_mean(vals, idx, n):
    s = torch.zeros(n, vals.shape[1]).index_add(0, idx, vals)
    c = torch.zeros(n).index_add(0, idx, torch.ones(len(idx))).clamp_min(1.0)
    return s / c.unsqueeze(1)


class Layer(nn.Module):
    def __init__(self, H, HE):
        super().__init__()
        self.res = _mlp(2 * H + HE, H, 1, zero=True)          # learned delay residual rho (0 at init: exact CPM)
        self.mf = _mlp(H + HE + 3, H, H)                      # forward message (predecessor -> event)
        self.mb = _mlp(H + HE + 3, H, H)                      # backward message (successor -> event)
        self.up = _mlp(3 * H + NTF, H, H)
        self.norm = nn.LayerNorm(H)


def _time_feats(t, l, var, t0):
    sd = torch.sqrt(var + 1e-6)
    slack = (l - t).clamp(-3.0, 5.0)
    p_on = 0.5 * (1 + torch.erf((l - t) / (sd * math.sqrt(2))))
    return torch.stack([t.clamp(0, 20.0), slack, sd.clamp(0, 5.0), p_on, (t - t0).clamp(0, 10.0)], 1)


class TENet(nn.Module):
    """Tropical event-graph network; config keys: hidden, layers, hops, maxplus, opt_feats, tau, nf."""

    def __init__(self, hidden=64, layers=3, hops=6, maxplus=True, opt_feats=True, tau=0.05, nf=0):
        super().__init__()
        self.config = dict(hidden=hidden, layers=layers, hops=hops, maxplus=maxplus, opt_feats=opt_feats, tau=tau, nf=nf)
        H, HE = hidden, 16
        self.hops, self.maxplus, self.opt_feats = hops, maxplus, opt_feats and nf > 0
        self.inp = nn.Linear(NX, H)
        self.etype, self.evoc = nn.Embedding(NT, H), nn.Embedding(NV, H)
        self.eemb, self.elin = nn.Embedding(NE, HE), nn.Linear(3, HE)
        self.layers = nn.ModuleList(Layer(H, HE) for _ in range(layers))
        self.log_tau = nn.Parameter(torch.tensor(math.log(tau)))
        self.optp = nn.Linear(nf, H) if self.opt_feats else None
        zin = 3 * H + NTF + 1 + (H if self.opt_feats else 0)
        self.phi = _mlp(zin, H, H)
        self.head = _mlp(3 * H, H, 1)

    # ---- batching
    @staticmethod
    def collate(Gs):
        off, cand, gid, nows, ends, sizes = 0, [], [], [], [], []
        keys = ("src", "dst")
        parts = {k: [] for k in ("x", "typ", "voc", "t0", "l0", "v0", "et", "ed", "tm", "of") + keys}
        for g, G in enumerate(Gs):
            n = len(G["typ"])
            for k in parts:
                parts[k].append(G[k] + off if k in keys else G[k])
            cand.append(G["cand"] + off); gid.append(np.full(len(G["cand"]), g))
            nows.append(G["now"] + off); ends.append(G["end"] + off); sizes.append(len(G["cand"]))
            off += n
        B = {k: torch.as_tensor(np.concatenate(v)) for k, v in parts.items()}
        B["x"], B["of"] = B["x"].float(), B["of"].float()
        B["typ"], B["voc"], B["et"] = B["typ"].long(), B["voc"].long(), B["et"].long()
        B["src"], B["dst"] = B["src"].long(), B["dst"].long()
        B["cand"] = torch.as_tensor(np.concatenate(cand)).long()
        B["gid"] = torch.as_tensor(np.concatenate(gid)).long()
        B["now"], B["end"] = torch.as_tensor(nows).long(), torch.as_tensor(ends).long()
        B["n"], B["sizes"] = off, sizes
        return B

    # ---- times
    def propagate(self, B, delay, tau):
        """Max-plus earliest times, variances and min-plus latest times over the temporal edges."""
        n, tm = B["n"], B["tm"]
        s, d, dl = B["src"][tm], B["dst"][tm], delay[tm]
        ev = B["ed"][:, 1][tm] ** 2
        t0, l0, v0 = B["t0"], B["l0"], B["v0"]
        has_in = torch.zeros(n, dtype=torch.bool)
        has_in[d] = True
        t, var, w = t0, v0, None
        for _ in range(self.hops):
            t, w = _smax(t[s] + dl, d, n, tau, t0)
            var = torch.where(has_in, torch.zeros(n).index_add(0, d, w * (var[s] + ev)), v0)
        is_end = torch.zeros(n, dtype=torch.bool)
        is_end[B["end"]] = True
        lb = torch.where(is_end, t, l0)
        u, wb = -lb, None
        for _ in range(self.hops):
            u, wb = _smax(u[d] + dl, s, n, tau, -lb)
        return t, -u, var, w, wb

    def forward(self, B):
        n, tm = B["n"], B["tm"]
        tau = self.log_tau.exp().clamp_min(1e-3)
        h = self.inp(B["x"]) + self.etype(B["typ"]) + self.evoc(B["voc"])
        e = self.eemb(B["et"]) + self.elin(B["ed"])
        src, dst = B["src"], B["dst"]
        st, dt = src[tm], dst[tm]
        t, l, var = B["t0"], B["l0"], B["v0"]
        base = B["ed"][:, 0]
        ones = torch.ones(int(tm.sum()))
        indeg = torch.zeros(n).index_add(0, dt, ones).clamp_min(1.0)
        outdeg = torch.zeros(n).index_add(0, st, ones).clamp_min(1.0)
        for L in self.layers:
            if self.maxplus:
                rho = L.res(torch.cat([h[src], h[dst], e], 1)).squeeze(1)
                delay = base + rho * tm
                t, l, var, wf, wb = self.propagate(B, delay, tau)
            else:
                delay = base
                wf, wb = 1.0 / indeg[dt], 1.0 / outdeg[st]
            tight_f = (t[dst] - t[src] - delay).clamp(0, 5.0)
            tight_b = (l[dst] - delay - l[src]).clamp(-5.0, 5.0)
            wfe = torch.ones(len(src)); wfe[tm] = wf
            wbe = torch.ones(len(src)); wbe[tm] = wb
            tmf = tm.float()
            mf = L.mf(torch.cat([h[src], e, torch.stack([tight_f, wfe, tmf], 1)], 1))
            mb = L.mb(torch.cat([h[dst], e, torch.stack([tight_b, wbe, tmf], 1)], 1))
            nt = ~tm
            af = torch.zeros(n, mf.shape[1]).index_add(0, dt, wfe[tm].unsqueeze(1) * mf[tm])
            ab = torch.zeros(n, mb.shape[1]).index_add(0, st, wbe[tm].unsqueeze(1) * mb[tm])
            if nt.any():
                af = af + _seg_mean(mf[nt], dst[nt], n)
                ab = ab + _seg_mean(mb[nt], src[nt], n)
            tf = _time_feats(t, l, var, B["t0"])
            h = L.norm(h + L.up(torch.cat([h, af, ab, tf], 1)))
        tf = _time_feats(t, l, var, B["t0"])
        cs, ce, ct = B["cand"][:, 0], B["cand"][:, 1], B["cand"][:, 2]
        ext = (t[ct] - t[B["end"]][B["gid"]]).clamp(-3.0, 5.0).unsqueeze(1)
        z = [h[ct], h[cs], h[ce], tf[ct], ext]
        if self.opt_feats:
            z.append(self.optp(B["of"]))
        z = self.phi(torch.cat(z, 1))
        G = len(B["sizes"])
        ctx = _seg_mean(z, B["gid"], G)[B["gid"]]
        s = self.head(torch.cat([z, ctx, h[B["now"]][B["gid"]]], 1)).squeeze(1)
        return list(torch.split(s, B["sizes"]))

    # ---- V13's policy interface
    @staticmethod
    def encode(env, S, D, V, c, legal, opts):
        return encode(env, S, D, V, c, legal, opts)

    def score(self, G):
        return self(self.collate([G]))[0]

    def batch(self, Gs):
        return self(self.collate(Gs))

    @torch.no_grad()
    def times(self, G):
        """Earliest / latest times the network computes on one graph (for checks)."""
        B = self.collate([G])
        tau = self.log_tau.exp().clamp_min(1e-3)
        h = self.inp(B["x"]) + self.etype(B["typ"]) + self.evoc(B["voc"])
        e = self.eemb(B["et"]) + self.elin(B["ed"])
        rho = self.layers[0].res(torch.cat([h[B["src"]], h[B["dst"]], e], 1)).squeeze(1)
        t, l, var, _, _ = self.propagate(B, B["ed"][:, 0] + rho * B["tm"], tau)
        return t.numpy(), l.numpy(), var.numpy()


# ==================== plug into V13 ====================================================================
def make_policy(path=None):
    cfg = dict(CONFIG)
    if path is not None and os.path.exists(path + ".json"):
        cfg.update(json.load(open(path + ".json")))
    cfg["nf"] = cfg.get("nf") or v13.DOMAIN.NF
    return TENet(**cfg)


def load(path):
    """A trained V14 policy (config from path.json); usable without install() (it carries its own encoder)."""
    pol = make_policy(path)
    pol.load_state_dict(torch.load(path + ".pt", weights_only=True))
    pol.eval()
    return pol


def install(domain=None, **config):
    """Use V14's event graph and network inside V13's training / evaluation."""
    if domain is not None:
        v13.use(domain)
    CONFIG.update(config)
    v13.ENCODE, v13.POLICY = encode, make_policy
    v13.OUT = os.path.join(os.path.dirname(v13.MODELS), "v14")


def main():
    ap, t, e = v13.parser("V14_v3")
    t.add_argument("--hidden", type=int, default=CONFIG["hidden"])
    t.add_argument("--layers", type=int, default=CONFIG["layers"])
    t.add_argument("--hops", type=int, default=CONFIG["hops"])
    t.add_argument("--tau", type=float, default=CONFIG["tau"])
    t.add_argument("--no_maxplus", action="store_true", help="ablation: times as input features only")
    t.add_argument("--no_opt_feats", action="store_true", help="ablation: no hand-made option features")
    for p in (t, e):
        p.add_argument("--threads", type=int, default=8, help="torch threads of the training process")
    a = ap.parse_args()
    if a.cmd == "train":
        install(a.domain, hidden=a.hidden, layers=a.layers, hops=a.hops, tau=a.tau,
                maxplus=not a.no_maxplus, opt_feats=not a.no_opt_feats)
    else:
        install(a.domain)
    torch.set_num_threads(a.threads)
    {"train": v13.cmd_train, "eval": v13.cmd_eval}[a.cmd](a)


if __name__ == "__main__":
    main()
