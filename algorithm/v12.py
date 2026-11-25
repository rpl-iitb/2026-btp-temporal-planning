"""V12: GNN + imitation (DAgger) + RL over PDDL moves, for multi-agent kitchens with stochastic durations.

    python -m algorithm.v12 --domain <domain module, see v13.py> train
    python -m algorithm.v12 --domain <domain module> eval --split v3_test --seeds 5

Decision: whenever an agent can act (algorithm/menv.py), it chooses one of its applicable ground PDDL actions or WAIT.
No kitchen code in the policy: the state is encoded from the PDDL vocabulary (V11's relational GNN).
Encoding (V11's, with three changes):
  * the deciding agent is marked with a "me" atom (one network for every agent);
  * running actions carry nominal remaining / elapsed time only (the sampled durations stay hidden);
  * all-pairs static relations are dropped (e.g. `connected` on a complete graph), and numeric all-pairs
    functions (travel times) keep each object's 10 nearest neighbours (smaller graph, faster decisions).
Training: DAgger with the rule dispatcher as the expert (it labels every visited state: its own rollouts first,
then the learner's, with random durations), cross-entropy over the legal moves + WAIT; then REINFORCE
(sampled rollouts, reward -makespan / certificate, failure -1.5 - unserved fraction, per-kitchen baseline,
entropy bonus) with the imitation loss as a regulariser.
"""
import argparse
import json
import os
import random
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

from . import v10, v11
from .menv import MAEnv
from . import v13
from .v13 import load_split, reward, summary, MODELS, OUT, use  # noqa: F401  (use(domain) plugs in the domain module)

torch.set_num_threads(1)


class Encoder(v11.Encoder):
    def __init__(self, T, V, collect=False, knn=10):
        super().__init__(T, V, collect)
        for rel in list(self.static):
            lst = self.static[rel]
            if not lst or len(lst[0][0]) != 3:
                continue
            firsts = defaultdict(list)
            for args, sc in lst:
                firsts[args[0]].append((args, sc))
            n = len(firsts)
            if n < 6 or len(lst) < 0.8 * n * (n - 1):
                continue
            if rel.startswith("fn:"):
                keep = []
                for a0, xs in firsts.items():
                    keep += sorted(xs, key=lambda x: x[1][0] if x[1] else 0)[:knn]
                self.static[rel] = keep
            else:
                del self.static[rel]
        self._static_t = None

    def relations(self):
        r = super().relations()
        r["me"] = (2, 1)
        return r

    def state_atoms(self, S, agent):
        T, t = self.T, S.t
        dyn = defaultdict(list)
        left = {}
        for tl, f, pos in T.tils[S.k:]:
            p = T.facts[f]
            dyn[("til+:" if pos else "til-:") + p[0]].append((self.fact_rel[f][1], (v11._sq(tl - t),)))
            if not pos and f not in left:
                left[f] = tl - t
        for f in S.facts:
            rel, args = self.fact_rel[f]
            dyn[rel].append((args, (v11._sq(left[f]) if f in left else 2.0,)))
        for i, st, lo, hi in S.run:
            a = T.actions[i]
            dyn["run:" + a.name].append((self.act_args[i], (v11._sq(st + a.lo - t), v11._sq(st + a.hi - t), v11._sq(t - st))))
        dyn["me"].append(((self.oid[agent], self.world), (1.0,)))
        out = {}
        if self._static_t is None:
            self._static_t = {r: (torch.tensor([a for a, _ in v], dtype=torch.long),
                                  torch.tensor([s for _, s in v], dtype=torch.float32).reshape(len(v), -1))
                              for r, v in self.static.items() if v}
        out.update(self._static_t)
        for r, v in dyn.items():
            a = torch.tensor([x for x, _ in v], dtype=torch.long)
            s = torch.tensor([y for _, y in v], dtype=torch.float32).reshape(len(v), -1)
            out[r] = (torch.cat([out[r][0], a]), torch.cat([out[r][1], s])) if r in out else (a, s)
        return out

    def option_feats(self, legal):
        out = []
        for i in legal:
            a = self.T.actions[i]
            out.append(("start", a.name, self.act_args[i], (v11._sq(a.lo), v11._sq(a.hi))))
        out.append(("wait", None, (self.world,), (0.0,)))
        return out


def make_batch(items):
    """items: [(encoder, state, agent, legal)] -> batch dict, option slices."""
    xs, gids, atoms, moves, slices = [], [], defaultdict(lambda: ([], [])), [], []
    off = 0
    for g, (enc, S, ag, legal) in enumerate(items):
        xs.append(enc.x)
        gids.append(torch.full((enc.n,), g, dtype=torch.long))
        for r, (a, s) in enc.state_atoms(S, ag).items():
            atoms[r][0].append(a + off); atoms[r][1].append(s)
        st = len(moves)
        for kind, schema, args, scal in enc.option_feats(legal):
            moves.append((g, kind, schema, tuple(x + off for x in args), scal))
        slices.append((st, len(moves)))
        off += enc.n
    at = {r: (torch.cat(a), torch.cat(s)) for r, (a, s) in atoms.items()}
    return {"x": torch.cat(xs), "gid": torch.cat(gids), "atoms": at, "moves": moves, "n": off}, slices


def build_vocab(tasks):
    dom = tasks[0].dom
    V = v11.Vocab(dom)
    for T in tasks:
        for name, (k, ns) in Encoder(T, V).relations().items():
            V.add(name, k, ns)
    V.rel = dict(sorted(V.rel.items()))
    return V


# ---------------------------------------------------------------- controller
class Controller:
    def __init__(self, env, pol=None, mode="greedy", beta=0.0, rng=None, record=None, label=True, keep=1.0):
        self.D = v13.DOMAIN.Dispatcher(env) if (label or mode in ("expert", "mix")) else None
        self.pol, self.mode, self.beta = pol, mode, beta
        self.rng = rng or np.random.default_rng(0)
        self.record, self.keep = record, keep
        self.enc = Encoder(env.T, pol.V) if pol is not None else None

    def __call__(self, env, S, c, legal):
        n = len(legal) + 1
        ex = None
        if self.D is not None:
            k = self.D(env, S, c, legal)
            ex = len(legal) if k is None else k
        if self.mode == "expert" or (self.mode == "mix" and self.rng.random() < self.beta):
            j = ex
        else:
            b, _ = make_batch([(self.enc, S, c, legal)])
            with torch.no_grad():
                s = self.pol(b)
            if self.mode == "greedy":
                j = int(s.argmax())
            else:
                p = torch.softmax(s, 0).numpy().astype(np.float64)
                j = int(self.rng.choice(n, p=p / p.sum()))
        if self.record is not None and self.rng.random() < self.keep:
            self.record.append((S, c, list(legal), j, ex))
        return None if j == len(legal) else j


# ---------------------------------------------------------------- workers
_W = {}


def _winit(model_path):
    torch.set_num_threads(1)
    _W["pol"] = load(model_path) if model_path else None
    _W["tasks"] = {}


def _task(split, name):
    key = (split, name)
    tasks = _W["tasks"]
    if key in tasks:
        tasks[key] = tasks.pop(key)               # most recent last
    else:
        tasks[key] = v13.DOMAIN.task(split, name)
        while len(tasks) > v13.TASK_CACHE:            # grounding is cheap (~0.2 s); memory per worker stays small
            tasks.pop(next(iter(tasks)))
    return tasks[key]


def _wrun(job):
    split, name, cert, mode, seed, beta, rec, keep, noise_k = job
    T = _task(split, name)
    NOISE = v13.NOISE
    env = MAEnv(T, v13.DOMAIN.AGENT, noise=NOISE.scaled(noise_k) if noise_k != 1 else NOISE, seed=seed, horizon=4 * cert)
    record = [] if rec else None
    ctl = Controller(env, _W["pol"], mode, beta, np.random.default_rng(seed + 3), record, label=rec, keep=keep)
    res = env.rollout(ctl, seed=seed)
    res["reward"] = reward(res, cert)
    res.pop("steps", None)
    # states are not picklable cheaply across processes with parents: strip parents
    if record is not None:
        record = [(_strip(S), c, lg, j, ex) for S, c, lg, j, ex in record]
    return name, seed, res, record


def _strip(S):
    return v10.State(S.facts, S.t, S.run, S.k)


def pool_run(jobs, model_path, procs):
    import multiprocessing as mp
    with mp.get_context("fork").Pool(procs, _winit, (model_path,)) as pool:
        return pool.map(_wrun, jobs, chunksize=1)


def save(pol, path):
    v11.save(pol, path)
    meta = json.load(open(path + ".json"))
    json.dump(meta, open(path + ".json", "w"))


def load(path):
    return v11.load(path)


# ---------------------------------------------------------------- training
def bc_epochs(pol, opt, data, encs, epochs, bs=32, log=print):
    pol.train()
    for ep in range(epochs):
        random.shuffle(data)
        tot = n = acc = 0
        t0 = time.time()
        for b in range(0, len(data), bs):
            ch = data[b:b + bs]
            batch, sl = make_batch([(encs[nm], S, c, lg) for nm, S, c, lg, y in ch])
            sc = pol(batch)
            loss = 0
            for (s0, s1), (_, _, _, _, y) in zip(sl, ch):
                lp = torch.log_softmax(sc[s0:s1], 0)
                loss = loss - lp[y]
                acc += int(lp.argmax()) == y
            loss = loss / len(ch)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(pol.parameters(), 1.0)
            opt.step()
            tot += float(loss) * len(ch); n += len(ch)
        log("  bc epoch {} loss {:.4f} acc {:.3f} ({} decisions, {:.0f}s)".format(ep, tot / max(n, 1), acc / max(n, 1), n, time.time() - t0))
    pol.eval()


def cmd_train(a):
    random.seed(0); torch.manual_seed(0)
    trs, vas = v13.DOMAIN.SPLITS
    tr, va = load_split(trs), load_split(vas)
    certs = {n: c for n, _, c, _ in tr + va}
    V = build_vocab([T for _, T, _, _ in tr + va])
    pol = v11.Policy(V, a.dim, a.layers)
    pol.eval()
    encs = {n: Encoder(T, V) for n, T, _, _ in tr}
    path = os.path.join(MODELS, a.name)
    logf = open(path + "_train.log", "w")

    def log(s):
        print(s, flush=True); logf.write(s + "\n"); logf.flush()
    log("V12: {} relations, {} parameters".format(len(V.rel), sum(p.numel() for p in pol.parameters())))
    opt = torch.optim.Adam(pol.parameters(), lr=1e-3)
    data, hist = [], []

    def val(tag):
        save(pol, path)
        rows = pool_run([(vas, n, c, "greedy", 1000 + s, 0.0, False, 1.0, 1) for n, _, c, _ in va for s in range(a.val_seeds)],
                        path, a.procs)
        sm = summary(rows, certs)
        log("{}: val success {:.3f} served {:.3f} makespan/cert {}".format(tag, sm["success"], sm["served"],
                                                                           None if sm["ms_cert"] is None else round(sm["ms_cert"], 3)))
        hist.append(dict(sm, tag=tag))
        return sm
    for it in range(a.dagger):
        mode, beta = ("expert", 1.0) if it == 0 else ("mix", max(0.0, 0.5 - 0.25 * (it - 1)))
        t0 = time.time()
        if it > 0:
            save(pol, path)
        jobs = [(trs, n, c, mode, 100 * it + s, beta, True, a.keep, 1) for n, _, c, _ in tr for s in range(a.seeds)]
        rows = pool_run(jobs, path if it > 0 else None, a.procs)
        new = [(nm, S, c, lg, ex) for nm, _, _, rec in rows for S, c, lg, j, ex in rec]
        data += new
        sm = summary(rows, certs)
        log("dagger {} ({}, beta {:.2f}): {} new decisions, rollout success {:.3f} ({:.0f}s)".format(
            it, mode, beta, len(new), sm["success"], time.time() - t0))
        bc_epochs(pol, opt, data, encs, a.bc_epochs, log=log)
        val("dagger {}".format(it))
    opt = torch.optim.Adam(pol.parameters(), lr=a.rl_lr)
    best = None
    for ep in range(a.rl_epochs):
        t0 = time.time()
        save(pol, path)
        names = random.sample(tr, min(a.rl_kitchens, len(tr)))
        jobs = [(trs, n, c, "sample", 5000 + 100 * ep + s, 0.0, True, a.rl_keep, 1) for n, _, c, _ in names for s in range(a.rollouts)]
        rows = pool_run(jobs, path, a.procs)
        by = defaultdict(list)
        for nm, sd, res, rec in rows:
            by[nm].append((res["reward"], rec))
        pol.train(); opt.zero_grad()
        n_dec = 0
        for nm, lst in by.items():
            R = np.array([r for r, _ in lst])
            adv = R - R.mean()
            if np.allclose(adv, 0):
                continue
            for (r, rec), A in zip(lst, adv):
                if not rec:
                    continue
                for b in range(0, len(rec), 32):
                    ch = rec[b:b + 32]
                    batch, sl = make_batch([(encs[nm], S, c, lg) for S, c, lg, j, ex in ch])
                    sc = pol(batch)
                    l = 0
                    for (s0, s1), (_, _, _, j, _) in zip(sl, ch):
                        lp = torch.log_softmax(sc[s0:s1], 0)
                        l = l - float(A) * lp[j] - a.entropy * -(lp.exp() * lp).sum()
                    (l / (len(rec) * len(lst))).backward()
                    n_dec += len(ch)
        if a.bc_weight > 0 and data:
            ch = random.sample(data, min(256, len(data)))
            for b in range(0, len(ch), 32):
                cc = ch[b:b + 32]
                batch, sl = make_batch([(encs[nm], S, c, lg) for nm, S, c, lg, y in cc])
                sc = pol(batch)
                l = sum(-torch.log_softmax(sc[s0:s1], 0)[y] for (s0, s1), (_, _, _, _, y) in zip(sl, cc))
                (a.bc_weight * l / len(ch)).backward()
        nn.utils.clip_grad_norm_(pol.parameters(), 1.0)
        opt.step(); pol.eval()
        sm = summary(rows, certs)
        log("rl {}: sampled success {:.3f} served {:.3f} makespan/cert {} ({} decisions, {:.0f}s)".format(
            ep, sm["success"], sm["served"], None if sm["ms_cert"] is None else round(sm["ms_cert"], 3), n_dec, time.time() - t0))
        if (ep + 1) % a.val_every == 0 or ep == a.rl_epochs - 1:
            v = val("rl {}".format(ep))
            score = v["success"] - (v["ms_cert"] or 2) * 0.1
            if best is None or score > best[0]:
                best = (score, {k: t.clone() for k, t in pol.state_dict().items()})
    if best is not None:
        pol.load_state_dict(best[1])
    save(pol, path)
    json.dump(hist, open(path + "_hist.json", "w"), indent=1)
    log("saved " + path)


def cmd_eval(a):
    sp = load_split(a.split)
    certs = {n: c for n, _, c, _ in sp}
    tiers = {n: inf.get("tier", "all") for n, _, _, inf in sp}
    out = {}
    for nk in a.noise:
        rows = pool_run([(a.split, n, c, "greedy", 7000 + s, 0.0, False, 1.0, nk) for n, _, c, _ in sp for s in range(a.seeds)],
                        os.path.join(MODELS, a.name), a.procs)
        per = defaultdict(list)
        for nm, sd, r, rec in rows:
            per[tiers[nm]].append((nm, sd, r, rec))
        res = {"all": summary(rows, certs)}
        res.update({t: summary(rr, certs) for t, rr in per.items()})
        out["{} noise x{}".format(a.name, nk)] = res
        print(a.name, "noise x{}".format(nk), json.dumps({k: {kk: (round(vv, 3) if isinstance(vv, float) else vv)
                                                            for kk, vv in v.items()} for k, v in res.items()}), flush=True)
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(os.path.join(OUT, "{}_{}.json".format(a.name, a.split)), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, help="domain module, e.g. overcooked_test.kitchen_v13 (see v13.py)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--name", default="V12_v3"); t.add_argument("--procs", type=int, default=11)
    t.add_argument("--dim", type=int, default=32); t.add_argument("--layers", type=int, default=4)
    t.add_argument("--seeds", type=int, default=2); t.add_argument("--dagger", type=int, default=3)
    t.add_argument("--keep", type=float, default=0.35); t.add_argument("--bc_epochs", type=int, default=2)
    t.add_argument("--val_seeds", type=int, default=1)
    t.add_argument("--rl_epochs", type=int, default=6); t.add_argument("--rl_kitchens", type=int, default=16)
    t.add_argument("--rollouts", type=int, default=4); t.add_argument("--rl_keep", type=float, default=0.25)
    t.add_argument("--rl_lr", type=float, default=2e-4); t.add_argument("--entropy", type=float, default=0.01)
    t.add_argument("--bc_weight", type=float, default=0.5); t.add_argument("--val_every", type=int, default=3)
    e = sub.add_parser("eval")
    e.add_argument("--name", default="V12_v3"); e.add_argument("--split", default="v3_test")
    e.add_argument("--seeds", type=int, default=3); e.add_argument("--procs", type=int, default=11)
    e.add_argument("--noise", type=float, nargs="*", default=[1.0])
    a = ap.parse_args()
    use(a.domain)
    {"train": cmd_train, "eval": cmd_eval}[a.cmd](a)


if __name__ == "__main__":
    main()
