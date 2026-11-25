"""V11: a learned temporal planner for any PDDL 2.1 domain (trained per domain, no domain code).

    python -m algorithm.v11 train  --bench overcooked_kitchen               # imitation, then self-improvement
    python -m algorithm.v11 solve  --model results/models/V11_overcooked_kitchen domain.pddl problem.pddl
    python -m algorithm.v11 eval   --bench overcooked_kitchen --split v2_test --budget 60

Environment: the PDDL semantics of v10 (decision epochs over snap actions with explicit time):
    moves = START an applicable action now | END a running flexible action now | WAIT to the next
    event. Only legal moves are offered, so every finished rollout is a valid plan (VAL re-checks).
Policy: a relational GNN over the state, built only from the PDDL vocabulary of the domain.
    Nodes are the objects (+ one world node). Every atom sends a message to its argument objects
    through an MLP of its relation: static and dynamic facts, goals ("goal:p"), numeric values
    ("fn:f", value as a scalar), running actions ("run:a", remaining time as scalars) and facts a
    timed literal will delete or add ("til-:p", "til+:p", time left as a scalar). After L rounds
    each legal move is scored by an MLP of its schema over its argument embeddings and a global
    readout; WAIT by an MLP of the readout and the time to the next event.
Training (per domain): imitation of the training plans (cross-entropy over legal moves), then
    self-improvement (expert iteration): sample rollouts on the training problems, keep every valid
    plan shorter than the current target, retrain. No heuristic search is used anywhere.
Inference: one greedy rollout (one network call per decision); with time left, sampled rollouts
    keep the shortest plan (anytime).
"""
import argparse
import json
import math
import os
import random
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

from . import v10

from . import RESULTS, split_dirs as _split_dirs

MODELS = os.path.join(RESULTS, "models")
torch.set_num_threads(int(os.environ.get("V11_THREADS", "4")))
import warnings
warnings.filterwarnings("ignore", message="index_reduce")


def _sq(x):
    """Scale-free squashing of a time / numeric value."""
    return math.copysign(math.log1p(abs(x)), x) / 5.0


# ---------------------------------------------------------------- environment
class Env:
    """Legal moves and transitions of a grounded task (v10 semantics)."""

    def __init__(self, T: v10.Task, max_steps=None):
        self.T, self.E = T, v10.Engine(T)
        self.max_steps = max_steps or 20 * len(T.goals) * 50

    def initial(self):
        S = v10.State(self.T.init, 0.0, (), 0)
        if self.T.tils and self.T.tils[0][0] <= 0:
            w = self.E.wait(S)
            S = w if w is not None else S
        return S

    def moves(self, S):
        """-> list of (move, successor). move = ("start", i) | ("end", j) | ("wait",)."""
        out = [(("end", j), self.E.apply_end(S, j)) for _, j, _ in self.E.ends(S)]
        out += [(("start", i), self.E.apply_start(S, i)) for _, i, _ in self.E.starts(S)]
        w = self.E.wait(S)
        if w is not None:
            out.append((("wait",), w))
        return out

    def is_goal(self, S):
        return self.E.is_goal(S)


def plan_of(T, S):
    steps = v10.extract_plan(T, S)
    return v10.plan_text(steps), round(max((t + d for t, _, d in steps), default=0.0), 3)


def replay(env: Env, text):
    """Teacher plan -> [(state, legal moves, chosen index)] and the final state, or None.
    Happenings are taken in time order; WAIT is chosen while the next happening is still ahead
    and an event lies before it; fixed-duration ends happen inside WAIT."""
    T = env.T
    acts = {}
    for a in T.actions:
        acts[(a.name,) + a.args] = a.i
    hap = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith(";"):
            continue
        ts, rest = line.split(":", 1)
        name = rest[rest.index("(") + 1:rest.index(")")].split()
        dur = float(rest[rest.index("[") + 1:rest.index("]")]) if "[" in rest else 0.0
        i = acts.get(tuple(name))
        if i is None:
            return None
        hap.append((float(ts), "start", i, float(ts)))
        a = T.actions[i]
        if not a.instant and a.lo < a.hi:
            hap.append((float(ts) + dur, "end", i, float(ts)))
    hap.sort(key=lambda h: (h[0], h[1] != "end"))
    S, out, k = env.initial(), [], 0
    started = {}                               # (i, teacher start) -> our start time
    for _ in range(env.max_steps):
        if k == len(hap):
            if env.is_goal(S):
                return out, S
            mv = env.moves(S)
            w = [n for n, (m, _) in enumerate(mv) if m[0] == "wait"]
            if not w:
                return None
            out.append((S, mv, w[0])); S = mv[w[0]][1]
            continue
        ht, kind, i, tst = hap[k]
        mv = env.moves(S)
        pick = None
        for n, (m, _) in enumerate(mv):
            if kind == "start" and m == ("start", i):
                pick = n
            elif kind == "end" and m[0] == "end" and S.run[m[1]][0] == i \
                    and abs(S.run[m[1]][1] - started.get((i, tst), -1)) < 1e-6:
                pick = n
        w = [n for n, (m, _) in enumerate(mv) if m[0] == "wait"]
        if kind == "end" and (i, tst) not in started:
            return None
        if kind == "end" and not any(r[0] == i and abs(r[1] - started[(i, tst)]) < 1e-6 for r in S.run):
            k += 1; continue                   # ended at its latest end inside a WAIT
        ahead = ht > S.t + 0.05
        if w and (pick is None or ahead):
            nt = mv[w[0]][1].t
            if pick is None or nt <= ht + 0.05:
                out.append((S, mv, w[0])); S = mv[w[0]][1]
                continue
        if pick is None:
            return None
        out.append((S, mv, pick))
        if kind == "start":
            started[(i, tst)] = S.t
        S = mv[pick][1]
        k += 1
    return None


# ---------------------------------------------------------------- graph encoding
class Vocab:
    """Relation / schema names of one domain (the network's per-domain modules)."""

    def __init__(self, dom: v10.Domain, arity=None):
        self.types = sorted(set(dom.parent) | {t for t in dom.constants.values()})
        self.rel = {}                           # relation -> (arity incl. world, n scalars)
        ar = arity or {}
        for p, k in ar.items():
            self.rel[p] = k
        self.schemas = {s.name: len(s.params) for s in dom.schemas}

    def add(self, name, k, ns):
        if name not in self.rel:
            self.rel[name] = (k, ns)


def domain_vocab(dom: v10.Domain, tasks):
    """Collect every relation used by the encodings of the given tasks."""
    V = Vocab(dom)
    for T in tasks:
        for name, (k, ns) in Encoder(T, V, collect=True).relations().items():
            V.add(name, k, ns)
    V.rel = dict(sorted(V.rel.items()))
    return V


class Encoder:
    """Per-task constant part of the graph + per-state atoms."""

    def __init__(self, T: v10.Task, V: Vocab, collect=False):
        self.T, self.V = T, V
        dom, prob = T.dom, T.prob
        objs = dict(dom.constants); objs.update(prob.objects)
        self.objs = sorted(objs)
        self.oid = {o: k for k, o in enumerate(self.objs)}
        self.world = len(self.objs)
        self.n = len(self.objs) + 1
        tix = {t: k for k, t in enumerate(V.types)}
        x = np.zeros((self.n, len(V.types) + 1), np.float32)
        for o, t in objs.items():
            for a in dom.ancestors(t):
                if a in tix:
                    x[self.oid[o], tix[a]] = 1
        x[self.world, -1] = 1
        self.x = torch.tensor(x)
        self.static = defaultdict(list)          # rel -> [(args, scalars)]
        for f in T.static_init:
            self._add(self.static, f[0], f[1:], ())
        for g in prob.goals:
            self._add(self.static, "goal:" + g[0], g[1:], ())
        mx = defaultdict(float)
        for k, v in prob.fluents.items():
            mx[k[0]] = max(mx[k[0]], abs(v))
        for k, v in prob.fluents.items():
            self._add(self.static, "fn:" + k[0], k[1:], (v / mx[k[0]] if mx[k[0]] else 0.0, _sq(v)))
        self.fact_rel = [(f[0], self._args(f[1:])) for f in T.facts]
        self.act_args = [self._args(a.args) for a in T.actions]
        self.collect = collect
        self._static_t = None

    def _args(self, names):
        return tuple(self.oid.get(n, self.world) for n in names) + (self.world,)

    def _add(self, store, rel, names, scal):
        store[rel].append((self._args(names), tuple(scal)))

    def relations(self):
        rels = {r: (len(v[0][0]), len(v[0][1])) for r, v in self.static.items() if v}
        for f in self.T.facts:
            rels.setdefault(f[0], (len(f), 1))
        for a in self.T.actions:
            rels.setdefault("run:" + a.name, (len(a.args) + 1, 3))
        for _, f, pos in self.T.tils:
            p = self.T.facts[f]
            rels.setdefault(("til+:" if pos else "til-:") + p[0], (len(p), 1))
        return rels

    def atoms(self, S):
        """-> rel -> (LongTensor [m, k], FloatTensor [m, s]) for state S."""
        T, t = self.T, S.t
        dyn = defaultdict(list)
        left = {}
        for tl, f, pos in T.tils[S.k:]:
            p = T.facts[f]
            dyn[("til+:" if pos else "til-:") + p[0]].append((self.fact_rel[f][1], (_sq(tl - t),)))
            if not pos and f not in left:
                left[f] = tl - t
        for f in S.facts:
            rel, args = self.fact_rel[f]
            dyn[rel].append((args, (_sq(left[f]) if f in left else 2.0,)))
        for i, st, lo, hi in S.run:
            a = T.actions[i]
            dyn["run:" + a.name].append((self.act_args[i], (_sq(lo - t), _sq(hi - t), _sq(t - st))))
        out = {}
        if self._static_t is None:
            self._static_t = {r: (torch.tensor([a for a, _ in v], dtype=torch.long),
                                  torch.tensor([s for _, s in v], dtype=torch.float32).reshape(len(v), -1))
                              for r, v in self.static.items() if v}
        out.update(self._static_t)
        for r, v in dyn.items():
            a = torch.tensor([x for x, _ in v], dtype=torch.long)
            s = torch.tensor([y for _, y in v], dtype=torch.float32).reshape(len(v), -1)
            if r in out:
                out[r] = (torch.cat([out[r][0], a]), torch.cat([out[r][1], s]))
            else:
                out[r] = (a, s)
        return out

    def move_feats(self, S, moves):
        """-> list of (kind, schema, arg index tuple, scalars)."""
        T, t = self.T, S.t
        out = []
        for m, nxt in moves:
            if m[0] == "start":
                a = T.actions[m[1]]
                out.append(("start", a.name, self.act_args[m[1]], (_sq(a.lo), _sq(a.hi))))
            elif m[0] == "end":
                i, st, lo, hi = S.run[m[1]]
                a = T.actions[i]
                out.append(("end", a.name, self.act_args[i], (_sq(t - st), _sq(hi - t))))
            else:
                out.append(("wait", None, (self.world,), (_sq(nxt.t - t),)))
        return out


# ---------------------------------------------------------------- network
def _mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, o))


NS = 3          # scalars per atom (padded)


class Policy(nn.Module):
    """Relational GNN. Atoms of the same arity share one message MLP; the relation is a learned
    embedding, so the number of tensor operations does not grow with the vocabulary."""

    def __init__(self, V: Vocab, dim=64, layers=4):
        super().__init__()
        self.V, self.dim, self.layers = V, dim, layers
        self.rid = {r: k for k, r in enumerate(V.rel)}
        self.arities = sorted({k for k, _ in V.rel.values()})
        self.inp = nn.Linear(len(V.types) + 1, dim)
        self.remb = nn.Embedding(len(V.rel), dim)
        self.msg = nn.ModuleList([nn.ModuleDict({str(k): _mlp(k * dim + dim + NS, dim, k * dim)
                                                 for k in self.arities}) for _ in range(layers)])
        self.upd = nn.ModuleList([_mlp(3 * dim, dim, dim) for _ in range(layers)])
        self.norm = nn.ModuleList([nn.LayerNorm(dim) for _ in range(layers)])
        self.skey = {a: "s%d" % k for k, a in enumerate(V.schemas)}
        self.start = nn.ModuleDict({self.skey[a]: _mlp((k + 1) * dim + 2 * dim + 2, dim, 1) for a, k in V.schemas.items()})
        self.end = nn.ModuleDict({self.skey[a]: _mlp((k + 1) * dim + 2 * dim + 2, dim, 1) for a, k in V.schemas.items()})
        self.wait = _mlp(dim + 2 * dim + 1, dim, 1)

    def group(self, atoms):
        """rel -> (args, scal)  =>  arity -> (args [m,k], rel ids [m], scal [m,NS])"""
        g = defaultdict(lambda: ([], [], []))
        for r, (a, s) in atoms.items():
            rid = self.rid.get(r)
            if rid is None or a.shape[0] == 0:
                continue
            k = a.shape[1]
            if s.shape[1] < NS:
                s = torch.cat([s, torch.zeros(s.shape[0], NS - s.shape[1])], 1)
            g[k][0].append(a); g[k][1].append(torch.full((a.shape[0],), rid, dtype=torch.long)); g[k][2].append(s)
        return {k: (torch.cat(a), torch.cat(r), torch.cat(s)) for k, (a, r, s) in g.items()}

    def embed(self, x, atoms, n):
        h = self.inp(x)
        grp = self.group(atoms)
        idx_all = {k: a.reshape(-1) for k, (a, _, _) in grp.items()}
        deg = torch.zeros(h.shape[0], 1)
        for k, idx in idx_all.items():
            deg = deg.index_add(0, idx, torch.ones(idx.shape[0], 1))
        deg = deg.clamp(min=1)
        for L in range(self.layers):
            agg_s = torch.zeros_like(h)
            agg_m = torch.full_like(h, -1e4)
            for k, (a, rid, sc) in grp.items():
                if str(k) not in self.msg[L]:
                    continue
                inp = torch.cat([h[a].reshape(a.shape[0], -1), self.remb(rid), sc], 1)
                m = self.msg[L][str(k)](inp).reshape(a.shape[0] * k, self.dim)
                agg_s = agg_s.index_add(0, idx_all[k], m)
                agg_m = agg_m.index_reduce(0, idx_all[k], m, "amax", include_self=True)
            agg_m = torch.where(agg_m < -1e3, torch.zeros_like(agg_m), agg_m)
            h = self.norm[L](h + self.upd[L](torch.cat([h, agg_s / deg, agg_m], 1)))
        return h

    def forward(self, batch):
        """batch: dict with x, atoms (global indices), object -> graph ids, and moves
        [(graph id, kind, schema, args (global), scalars)] -> scores [n moves]"""
        h = self.embed(batch["x"], batch["atoms"], batch["n"])
        gid = batch["gid"]
        G = int(gid.max()) + 1
        cnt = torch.zeros(G, 1).index_add(0, gid, torch.ones(gid.shape[0], 1))
        gmean = torch.zeros(G, self.dim).index_add(0, gid, h) / cnt
        gmax = torch.full((G, self.dim), -1e4).index_reduce(0, gid, h, "amax", include_self=True)
        glob = torch.cat([gmean, gmax], 1)
        scores = torch.zeros(len(batch["moves"]))
        groups = defaultdict(list)
        for n, (g, kind, schema, args, scal) in enumerate(batch["moves"]):
            groups[(kind, schema)].append(n)
        for (kind, schema), idx in groups.items():
            mv = [batch["moves"][n] for n in idx]
            gi = torch.tensor([m[0] for m in mv])
            args = torch.tensor([m[3] for m in mv], dtype=torch.long)
            sc = torch.tensor([m[4] for m in mv], dtype=torch.float32)
            if kind == "wait":
                out = self.wait(torch.cat([h[args[:, 0]], glob[gi], sc], 1))
            else:
                net = (self.start if kind == "start" else self.end)[self.skey[schema]]
                out = net(torch.cat([h[args].reshape(len(mv), -1), glob[gi], sc], 1))
            scores[torch.tensor(idx)] = out.squeeze(1)
        return scores


def make_batch(items):
    """items: [(encoder, state, moves)] -> batch dict, move slices per item."""
    xs, gids, atoms, moves, slices = [], [], defaultdict(lambda: ([], [])), [], []
    off = 0
    for g, (enc, S, mv) in enumerate(items):
        xs.append(enc.x)
        gids.append(torch.full((enc.n,), g, dtype=torch.long))
        for r, (a, s) in enc.atoms(S).items():
            atoms[r][0].append(a + off); atoms[r][1].append(s)
        st = len(moves)
        for kind, schema, args, scal in enc.move_feats(S, mv):
            moves.append((g, kind, schema, tuple(x + off for x in args), scal))
        slices.append((st, len(moves)))
        off += enc.n
    at = {r: (torch.cat(a), torch.cat(s)) for r, (a, s) in atoms.items()}
    return {"x": torch.cat(xs), "gid": torch.cat(gids), "atoms": at, "moves": moves, "n": off}, slices


# ---------------------------------------------------------------- rollouts
@torch.no_grad()
def rollout(pol, env, enc, greedy=True, temp=1.0, rng=None, deadline=None, record=None):
    """-> goal state or None. record: list that receives the decisions (state, moves, chosen).
    Loop breaking without search: from a state seen before (same facts, running actions and timed
    literals, up to the absolute time) the policy never repeats a move it already took there, so a
    wrong turn can be undone but a cycle cannot repeat."""
    S = env.initial()
    tried = defaultdict(set)
    for _ in range(env.max_steps):
        if env.is_goal(S):
            return S
        if deadline and time.time() > deadline:
            return None
        mv = env.moves(S)
        key = S.key()
        if tried[key]:
            mv = [x for x in mv if x[0] not in tried[key]]
        if not mv:
            return None
        if len(mv) == 1:
            k = 0
        else:
            b, _ = make_batch([(enc, S, mv)])
            sc = pol(b)
            if greedy:
                k = int(sc.argmax())
            else:
                p = torch.softmax(sc / temp, 0).numpy().astype(np.float64)
                k = int((rng or np.random).choice(len(mv), p=p / p.sum()))
        tried[key].add(mv[k][0])
        if record is not None:
            record.append((S, mv, k))
        S = mv[k][1]
    return None


class Planner:
    """Learned planner: greedy rollout, then sampled rollouts until the budget (anytime)."""

    def __init__(self, pol, samples=64, temp=1.0):
        self.pol, self.samples, self.temp = pol, samples, temp

    def solve(self, T, budget=60.0, t0=None):
        t0 = t0 or time.time()
        env, enc = Env(T), Encoder(T, self.pol.V)
        end = t0 + budget
        best, first_s, n = None, None, 0
        rng = np.random.default_rng(0)
        while time.time() < end and n <= self.samples:
            S = rollout(self.pol, env, enc, greedy=(n == 0), temp=self.temp, rng=rng, deadline=end)
            n += 1
            if S is not None:
                txt, ms = plan_of(T, S)
                if best is None or ms < best[1]:
                    best = (txt, ms)
                    if first_s is None:
                        first_s = time.time() - t0
        return best, {"first_s": first_s, "rollouts": n}


# ---------------------------------------------------------------- data
def split_dirs(bench, split):
    return _split_dirs(split, bench)


def load_task(d, dom_text=None):
    from .grounding import Task as HT
    if dom_text is None:
        dom_text = open(HT.load(d).domain_file()).read()
    prob_text = open(os.path.join(d, "problem.pddl")).read()
    return v10.Task(v10.Domain(dom_text), v10.Problem(prob_text)), dom_text


def teacher_text(bench, split, d):
    p = os.path.join(RESULTS, "teacher", bench, split, os.path.basename(d), "teacher.plan")
    return open(p).read() if os.path.exists(p) else None


def load_data(bench, splits):
    """-> dom, {split: [(name, T, teacher text)]}"""
    data, dom_text = {}, None
    for sp in splits:
        rows = []
        for d in split_dirs(bench, sp):
            T, dom_text = load_task(d, dom_text)
            rows.append((os.path.basename(d), T, teacher_text(bench, sp, d), d))
        data[sp] = rows
    return v10.Domain(dom_text), dom_text, data


def examples(pol, env, enc, traj):
    return [(enc, S, mv, k) for S, mv, k in traj if len(mv) > 1]


def train_epochs(pol, opt, exs, epochs, bs=32, log=None):
    pol.train()
    for ep in range(epochs):
        random.shuffle(exs)
        tot, n, t0 = 0.0, 0, time.time()
        for b in range(0, len(exs), bs):
            chunk = exs[b:b + bs]
            batch, sl = make_batch([(e, S, mv) for e, S, mv, _ in chunk])
            sc = pol(batch)
            loss = 0.0
            for (st, en), (_, _, _, k) in zip(sl, chunk):
                loss = loss - torch.log_softmax(sc[st:en], 0)[k]
            loss = loss / len(chunk)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(pol.parameters(), 1.0)
            opt.step()
            tot += float(loss) * len(chunk); n += len(chunk)
        if log:
            log("  epoch {} loss {:.3f} ({:.0f}s)".format(ep, tot / max(n, 1), time.time() - t0))
    pol.eval()


def evaluate_greedy(pol, rows, log=None, budget=None):
    res = []
    for name, T, txt, d in rows:
        env, enc = Env(T), Encoder(T, pol.V)
        t0 = time.time()
        S = rollout(pol, env, enc, greedy=True, deadline=(t0 + budget) if budget else None)
        dt = time.time() - t0
        res.append((name, None if S is None else plan_of(T, S)[1], dt))
    return res


def save(pol, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(pol.state_dict(), path + ".pt")
    json.dump({"types": pol.V.types, "rel": {r: list(v) for r, v in pol.V.rel.items()}, "schemas": pol.V.schemas,
               "dim": pol.dim, "layers": pol.layers}, open(path + ".json", "w"))


def load(path):
    meta = json.load(open(path + ".json"))
    V = Vocab.__new__(Vocab)
    V.types, V.rel, V.schemas = meta["types"], {r: tuple(v) for r, v in meta["rel"].items()}, meta["schemas"]
    pol = Policy(V, meta["dim"], meta["layers"])
    pol.load_state_dict(torch.load(path + ".pt", weights_only=True))
    pol.eval()
    return pol


def cmd_train(a):
    random.seed(0); torch.manual_seed(0)
    path = os.path.join(MODELS, a.name or "V11_" + a.bench)
    logf = open(path + "_train.log", "a")

    def log(s):
        print(s, flush=True); logf.write(s + "\n"); logf.flush()
    dom, dom_text, data = load_data(a.bench, [a.train, a.val])
    tr, va = data[a.train], data[a.val]
    log("V11 train {} : {} train, {} val problems".format(a.bench, len(tr), len(va)))
    V = domain_vocab(dom, [T for _, T, _, _ in tr + va])
    pol = Policy(V, a.dim, a.layers)
    pol.eval()
    opt = torch.optim.Adam(pol.parameters(), lr=a.lr)
    # imitation
    target, exs = {}, []
    for name, T, txt, d in tr:
        if not txt:
            continue
        env, enc = Env(T), Encoder(T, V)
        r = replay(env, txt)
        if r is None:
            continue
        traj, S = r
        target[name] = (plan_of(T, S)[1], traj)
        exs += examples(pol, env, enc, traj)
    log("replayed {}/{} teacher plans, {} decisions".format(len(target), len(tr), len(exs)))
    train_epochs(pol, opt, exs, a.epochs, log=log)
    save(pol, path)

    def val_report(tag):
        r = evaluate_greedy(pol, va)
        ok = [x for x in r if x[1] is not None]
        ref = {n: plan_of(T, replay(Env(T), txt)[1])[1] for n, T, txt, d in va if txt and replay(Env(T), txt)}
        ratio = [x[1] / ref[x[0]] for x in ok if x[0] in ref]
        log("{}: val greedy solved {}/{}  makespan/teacher {:.3f}  mean time {:.2f}s".format(
            tag, len(ok), len(r), float(np.mean(ratio)) if ratio else float("nan"), float(np.mean([x[2] for x in r]))))
    val_report("imitation")
    # self-improvement (expert iteration)
    rng = np.random.default_rng(0)
    for it in range(a.iters):
        better, new = 0, 0
        t0 = time.time()
        for name, T, txt, d in tr:
            env, enc = Env(T), Encoder(T, V)
            for s in range(a.samples + 1):
                rec = []
                S = rollout(pol, env, enc, greedy=(s == 0), temp=a.temp, rng=rng, record=rec)
                if S is None:
                    continue
                ms = plan_of(T, S)[1]
                if name not in target or ms < target[name][0] - 1e-6:
                    better += name in target; new += name not in target
                    target[name] = (ms, rec)
        exs = []
        for name, T, txt, d in tr:
            if name in target:
                exs += examples(pol, None, Encoder(T, V), target[name][1])
        log("iter {}: improved {}, newly solved {}, targets {} ({:.0f}s)".format(it, better, new, len(target),
                                                                               time.time() - t0))
        train_epochs(pol, opt, exs, a.iter_epochs, log=log)
        save(pol, path)
        val_report("iter {}".format(it))


# ---------------------------------------------------------------- self-improvement (parallel)
_W = {}


def _winit(model, bench, split):
    torch.set_num_threads(1)
    _W["pol"] = load(model)
    _W["dom_text"] = None
    _W["bench"], _W["split"] = bench, split


def _wsample(job):
    """job = (dir, n samples, temp, seed) -> (name, best plan text, makespan, successes)"""
    d, n, temp, seed = job
    T, _ = load_task(d, _W.get("dom_text"))
    pol = _W["pol"]
    env, enc = Env(T), Encoder(T, pol.V)
    rng = np.random.default_rng(seed)
    best, ok = None, 0
    for s in range(n + 1):
        S = rollout(pol, env, enc, greedy=(s == 0), temp=temp, rng=rng, deadline=time.time() + 60)
        if S is not None:
            ok += 1
            txt, ms = plan_of(T, S)
            if best is None or ms < best[1]:
                best = (txt, ms)
    return os.path.basename(d), best, ok


def cmd_improve(a):
    """Expert iteration: sample rollouts on every training problem (in parallel), keep each
    problem's shortest valid plan as its target, retrain on the targets; repeat."""
    import multiprocessing as mp
    random.seed(0); torch.manual_seed(0)
    src = a.model or os.path.join(MODELS, "V11_" + a.bench)
    path = os.path.join(MODELS, a.name)
    logf = open(path + "_train.log", "a")

    def log(s):
        print(s, flush=True); logf.write(s + "\n"); logf.flush()
    dom, dom_text, data = load_data(a.bench, [a.train])
    tr = data[a.train]
    pol = load(src)
    save(pol, path)
    opt = torch.optim.Adam(pol.parameters(), lr=a.lr)
    target = {}
    for name, T, txt, d in tr:
        r = replay(Env(T), txt) if txt else None
        if r:
            target[name] = (plan_of(T, r[1])[1], plan_of(T, r[1])[0])
    log("improve {} from {}: {} targets from the teacher".format(a.name, src, len(target)))
    for it in range(a.iters):
        t0 = time.time()
        jobs = [(d, a.samples, a.temp, 1000 * it + k) for k, (_, _, _, d) in enumerate(tr)]
        with mp.get_context("fork").Pool(a.procs, _winit, (path, a.bench, a.train)) as pool:
            res = pool.map(_wsample, jobs, chunksize=1)
        better = solved = 0
        for name, best, ok in res:
            solved += ok > 0
            if best and (name not in target or best[1] < target[name][0] - 1e-6):
                better += 1
                target[name] = (best[1], best[0])
        log("iter {}: policy solved {}/{} problems ({} samples each), {} targets improved ({:.0f}s)".format(
            it, solved, len(tr), a.samples + 1, better, time.time() - t0))
        exs = []
        for name, T, txt, d in tr:
            if name in target:
                r = replay(Env(T), target[name][1])
                if r:
                    exs += examples(pol, None, Encoder(T, pol.V), r[0])
        train_epochs(pol, opt, exs, a.epochs, log=log)
        save(pol, path)
        json.dump({n: v[0] for n, v in target.items()}, open(path + "_targets.json", "w"))


def cmd_eval(a):
    pol = load(a.model or os.path.join(MODELS, "V11_" + a.bench))
    dom, dom_text, data = load_data(a.bench, [a.split])
    P = Planner(pol, samples=a.samples)
    rows = []
    from .pddlkit import validate_plan
    from .grounding import Task as HT
    for name, T, txt, d in data[a.split]:
        t0 = time.time()
        best, info = P.solve(T, a.budget, t0)
        ok = None
        if best:
            ht = HT.load(d)
            dom_f = ht.domain_file()
            ok = validate_plan(dom_f, os.path.join(d, "problem.pddl"), best[0]).valid
        rows.append({"instance": name, "solved": bool(best and ok), "makespan": best[1] if best else None,
                     "first_s": info["first_s"], "rollouts": info["rollouts"], "val": ok})
        print(rows[-1], flush=True)
    out = os.path.join(RESULTS, "teacher", a.bench, "v11_{}_{}.json".format(a.split, int(a.budget)))
    json.dump(rows, open(out, "w"), indent=1)
    s = [r for r in rows if r["solved"]]
    print("solved {}/{}  first plan {:.2f}s".format(len(s), len(rows), float(np.mean([r["first_s"] for r in s])) if s else -1))


def _weval(job):
    """job = (dir, budget) -> result row (worker of cmd_peval)."""
    d, budget = job
    from .pddlkit import validate_plan
    from .grounding import Task as HT
    t0 = time.time()
    T, _ = load_task(d, _W.get("dom_text"))
    best, info = Planner(_W["pol"], samples=10 ** 6).solve(T, budget, t0)
    ok = None
    if best:
        ok = validate_plan(HT.load(d).domain_file(), os.path.join(d, "problem.pddl"), best[0]).valid
    return {"instance": os.path.basename(d), "solved": bool(best and ok), "makespan": best[1] if best else None,
            "first_s": info["first_s"], "rollouts": info["rollouts"], "val": ok,
            "plan_text": best[0] if best and ok else None}


def cmd_peval(a):
    """Parallel evaluation: one process per kitchen, budget seconds each, every plan checked by VAL."""
    import multiprocessing as mp
    model = a.model or os.path.join(MODELS, "V11_" + a.bench)
    dirs = split_dirs(a.bench, a.split)
    with mp.get_context("fork").Pool(a.procs, _winit, (model, a.bench, a.split)) as pool:
        rows = pool.map(_weval, [(d, a.budget) for d in dirs], chunksize=1)
    tag = os.path.basename(model)
    out = os.path.join(RESULTS, "teacher", a.bench, "{}_{}_{}s.json".format(tag, a.split, int(a.budget)))
    json.dump(rows, open(out, "w"), indent=1)
    s = [r for r in rows if r["solved"]]
    print("{} on {} ({} s): solved {}/{}  first plan {:.2f}s".format(tag, a.split, a.budget, len(s), len(rows),
          float(np.mean([r["first_s"] for r in s])) if s else -1))
    print("written", out)


def cmd_solve(a):
    pol = load(a.model)
    t0 = time.time()
    T = v10.Task(v10.Domain(open(a.domain).read()), v10.Problem(open(a.problem).read()))
    best, info = Planner(pol, samples=a.samples).solve(T, a.budget, t0)
    print(info, best[1] if best else None)
    if best:
        print(best[0])


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--bench", default="overcooked_kitchen")
    t.add_argument("--train", default="v2_train"); t.add_argument("--val", default="v2_val")
    t.add_argument("--dim", type=int, default=32); t.add_argument("--layers", type=int, default=4)
    t.add_argument("--lr", type=float, default=1e-3); t.add_argument("--epochs", type=int, default=8)
    t.add_argument("--iters", type=int, default=3); t.add_argument("--iter_epochs", type=int, default=3)
    t.add_argument("--samples", type=int, default=4); t.add_argument("--temp", type=float, default=1.0)
    t.add_argument("--name")
    e = sub.add_parser("eval")
    e.add_argument("--bench", default="overcooked_kitchen"); e.add_argument("--split", default="v2_test")
    e.add_argument("--budget", type=float, default=60.0); e.add_argument("--samples", type=int, default=64)
    e.add_argument("--model")
    s = sub.add_parser("solve")
    s.add_argument("--model", required=True); s.add_argument("domain"); s.add_argument("problem")
    s.add_argument("--budget", type=float, default=60.0); s.add_argument("--samples", type=int, default=64)
    i = sub.add_parser("improve")
    i.add_argument("--bench", default="overcooked_kitchen"); i.add_argument("--train", default="v2_train")
    i.add_argument("--model"); i.add_argument("--name", default="V11X_overcooked_kitchen")
    i.add_argument("--iters", type=int, default=4); i.add_argument("--samples", type=int, default=16)
    i.add_argument("--temp", type=float, default=1.0); i.add_argument("--procs", type=int, default=10)
    i.add_argument("--epochs", type=int, default=2); i.add_argument("--lr", type=float, default=5e-4)
    pe = sub.add_parser("peval")
    pe.add_argument("--bench", default="overcooked_kitchen"); pe.add_argument("--split", default="v2_test")
    pe.add_argument("--budget", type=float, default=60.0); pe.add_argument("--procs", type=int, default=11)
    pe.add_argument("--model")
    a = ap.parse_args()
    {"train": cmd_train, "eval": cmd_eval, "solve": cmd_solve, "improve": cmd_improve, "peval": cmd_peval}[a.cmd](a)


if __name__ == "__main__":
    main()


# ==================== the evaluation interface (build / solve -> RunResult), as V1-V9 ==========================
def build(key, spec=None, seed=0, model=None, samples=64, **over):
    """V11 with a trained model (default results/models/V11_<benchmark> of the task's domain is not known here, so the
    harness passes model=...; else results/models/V11)."""
    pol = load(model or os.path.join(MODELS, "V11"))

    def solve_files(dom, prob, budget):
        t0 = time.time()
        T = v10.Task(v10.Domain(open(dom).read()), v10.Problem(open(prob).read()))
        best, info = Planner(pol, samples=samples).solve(T, budget, t0)
        return (best[0] if best else None), info
    return v10.TextPlanner(key, solve_files)
