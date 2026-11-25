"""T-iFLAX V13: learned task choice for multi-agent execution with stochastic durations (imitation + RL).

What it does
  Closed-loop control instead of a fixed plan: whenever an agent can act, it chooses one task among all the
  tasks a rule-based dispatcher could give it now (e.g. in a kitchen: deliver at window s, add to pot p, claim a
  plate, light pot p, fetch from crate x, scrub, wash, park, wait; every alternative station). The task's next
  step is executed and the choice is re-made at the next decision, so it reacts to the actual, random durations.
  policy       each option -> a feature vector (rule, travel, timing / slack of the order it serves,
               coordination: how much closer the nearest other free agent is, how many agents already head there,
               global counts) -> MLP embedding; a deep-set context (mean of the option embeddings) is appended; an
               MLP scores each option; softmax.
  training     (1) DAgger: the dispatcher labels every decision of rollouts under random durations, first its own
               rollouts, then rollouts of the learner (the dispatcher is a pure function of the state, so it can
               label any state); cross-entropy. (2) RL on 8 sampled rollouts per instance with random durations,
               reward -makespan / certificate (failure: -1.5 - unserved fraction), advantage = reward minus the
               instance's mean (group baseline, no critic), entropy bonus, the imitation loss on the DAgger data as
               a regulariser. --rl ppo (default): the rollouts of an instance share their duration stream, the
               decisions keep the log-probability they were sampled with, and each batch is used for several
               passes of minibatches of PPO's clipped objective (early stop on the KL to the sampling policy); the
               imitation weight decays to 0.2 x. --rl reinforce: one policy-gradient step per batch, constant
               imitation weight (the earlier training).
  execution    algorithm/menv.py: the multi-agent temporal environment over any PDDL 2.1 domain (v10 semantics).

What it needs: a domain module (use(module) or --domain), e.g. overcooked_test/kitchen_v13.py, with
  AGENT (agent type), NOISE (menv.Noise), NF, SPLITS (train, val), Dispatcher(env) (.options(V, agent, legal) ->
  [{"k": move, "rule", "station", "dist", "info"}], .RULES), View(env, S), expert_index(D, V, agent, legal, opts),
  features(D, V, agent, opts) -> [n, NF], load_split(split) -> [(name, v10 Task, certificate, meta)],
  task(split, name) -> v10 Task.  The algorithm has no domain code.

Earlier results: pre-final v3 kitchen benchmark (3-4 cooks, random durations). Domain modules in final/:
overcooked_test/kitchen_v13.py, factory_test/factory_v13.py (not trained yet).

    python -m algorithm.v13 --domain overcooked_test.kitchen_v13 train
    python -m algorithm.v13 --domain overcooked_test.kitchen_v13 eval --split v3_test --seeds 5
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

from .menv import MAEnv

from . import RESULTS

MODELS = os.path.join(RESULTS, "models")
OUT = os.path.join(RESULTS, "v13")
torch.set_num_threads(1)
FAIL = -1.5
TASK_CACHE = 4         # grounded tasks kept per worker
DOMAIN = None          # the domain module (use())
NOISE = None
# Plug-ins for versions that keep V13's options, expert, training and execution but change the network (V14):
ENCODE = None          # encode(env, S, D, V, agent, legal, opts) -> X (default: features(D, V, agent, opts))
POLICY = None          # POLICY(path or None) -> module with .score(X) and .batch([X]) (default: Policy())


def use(domain):
    """Plug in a domain module (see the header); a module path string is imported."""
    global DOMAIN, NOISE
    if isinstance(domain, str):
        import importlib
        domain = importlib.import_module(domain)
    DOMAIN, NOISE = domain, domain.NOISE
    _results_from_env()
    return domain


def _results_from_env():
    """The domain module's harness sets TIFLAX_RESULTS on import (after this module read it): follow it."""
    global MODELS, OUT
    res = os.environ.get("TIFLAX_RESULTS")
    if res:
        MODELS, OUT = os.path.join(res, "models"), os.path.join(res, os.path.basename(OUT))


def load_split(split):
    """-> [(name, v10 Task, certificate makespan, meta)]"""
    return DOMAIN.load_split(split)


def split_rows(split):
    """-> [(name, None, certificate, meta)]: the split without its grounded tasks. The training / evaluation parent
    only needs names and certificates; keeping every grounded task in it would be copied into each forked worker."""
    import gc
    rows = [(n, None, c, m) for n, _, c, m in load_split(split)]
    gc.collect()
    return rows


def features(D, V, c, opts):
    return DOMAIN.features(D, V, c, opts)


def encode(env, S, D, V, c, legal, opts, pol=None):
    """The network's input for one decision: the policy's own encoder if it has one (V14), else ENCODE, else
    V13's option feature matrix."""
    if pol is not None and hasattr(pol, "encode"):
        return pol.encode(env, S, D, V, c, legal, opts)
    return ENCODE(env, S, D, V, c, legal, opts) if ENCODE is not None else features(D, V, c, opts)


def make_policy(path=None):
    """A fresh policy (path None) or the shell to load `path` into."""
    return POLICY(path) if POLICY is not None else Policy()


# ---------------------------------------------------------------- policy
class Policy(nn.Module):
    def __init__(self, hidden=128, nf=None):
        super().__init__()
        NF = nf or DOMAIN.NF
        self.phi = nn.Sequential(nn.Linear(NF, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, X):
        """X [n, NF] -> scores [n]"""
        h = self.phi(X)
        ctx = h.mean(0, keepdim=True).expand_as(h)
        return self.head(torch.cat([h, ctx], 1)).squeeze(1)

    def score(self, X):
        return self(torch.as_tensor(X))

    def batch(self, Xs):
        """list of [n_i, NF] -> list of score vectors (one forward pass)."""
        sizes = [x.shape[0] for x in Xs]
        X = torch.as_tensor(np.concatenate(Xs))
        h = self.phi(X)
        idx = torch.repeat_interleave(torch.arange(len(Xs)), torch.tensor(sizes))
        ctx = torch.zeros(len(Xs), h.shape[1]).index_add(0, idx, h) / torch.tensor(sizes, dtype=torch.float32).unsqueeze(1)
        s = self.head(torch.cat([h, ctx[idx]], 1)).squeeze(1)
        return list(torch.split(s, sizes))


# ---------------------------------------------------------------- controllers
class Controller:
    """mode: 'expert' (dispatcher), 'greedy' / 'sample' (policy), 'mix' (beta: expert prob). Records
    (features, chosen index, expert index, log-probability of the choice when sampled, else None) per decision
    when record is a list."""

    def __init__(self, env, pol=None, mode="greedy", beta=0.0, rng=None, record=None, label=True):
        self.D = DOMAIN.Dispatcher(env)
        self.pol, self.mode, self.beta = pol, mode, beta
        self.rng = rng or np.random.default_rng(0)
        self.record, self.label = record, label

    def expert_index(self, V, c, legal, opts):
        return DOMAIN.expert_index(self.D, V, c, legal, opts)

    def __call__(self, env, S, c, legal):
        V = DOMAIN.View(env, S)
        opts = self.D.options(V, c, legal)
        if len(opts) == 1 and self.record is None:
            return opts[0]["k"]
        ex = self.expert_index(V, c, legal, opts) if (self.label or self.mode in ("expert", "mix")) else None
        X = encode(env, S, self.D, V, c, legal, opts, self.pol) if (self.pol is not None or self.record is not None) else None
        logp = None
        if self.mode == "expert" or (self.mode == "mix" and self.rng.random() < self.beta):
            j = ex
        else:
            with torch.no_grad():
                s = self.pol.score(X)
            if self.mode == "greedy":
                j = int(s.argmax())
            else:
                ls = torch.log_softmax(s, 0)
                p = ls.exp().numpy().astype(np.float64)
                j = int(self.rng.choice(len(opts), p=p / p.sum()))
                logp = float(ls[j])
        if self.record is not None and len(opts) > 1:
            self.record.append((X, j, ex, logp))
        return opts[j]["k"]


def reward(res, cert):
    if res["success"]:
        return -res["makespan"] / cert
    return FAIL - (1 - res["served"] / max(1, res["goals"]))


def run_one(T, cert, mode, pol=None, seed=0, noise=None, beta=0.0, record=None, label=True, pseed=None):
    """seed: the durations' random stream; pseed: the policy's sampling stream (default: from seed)."""
    env = MAEnv(T, DOMAIN.AGENT, noise=noise if noise is not None else NOISE, seed=seed, horizon=4 * cert)
    ctl = Controller(env, pol, mode, beta, np.random.default_rng((seed if pseed is None else pseed) + 7), record, label)
    res = env.rollout(ctl, seed=seed)
    res["reward"] = reward(res, cert)
    res.pop("steps", None)
    return res


# ---------------------------------------------------------------- parallel workers
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
        tasks[key] = DOMAIN.task(split, name)
        while len(tasks) > TASK_CACHE:            # grounding is cheap (~0.2 s); memory per worker stays small
            tasks.pop(next(iter(tasks)))
    return tasks[key]


def _wrun(job):
    split, name, cert, mode, seed, beta, rec, noise_k = job[:8]
    pseed = job[8] if len(job) > 8 else None
    T = _task(split, name)
    record = [] if rec else None
    res = run_one(T, cert, mode, _W["pol"], seed, NOISE.scaled(noise_k) if noise_k != 1 else NOISE, beta, record,
                  label=rec, pseed=pseed)
    return name, seed, res, record


def pool_run(jobs, model_path, procs):
    import gc
    import multiprocessing as mp
    gc.collect()
    gc.freeze()            # the workers' garbage collector then leaves the parent's objects (e.g. DAgger data) unshared
    try:
        with mp.get_context("fork").Pool(procs, _winit, (model_path,)) as pool:
            return pool.map(_wrun, jobs, chunksize=1)
    finally:
        gc.unfreeze()


# ---------------------------------------------------------------- training
def save(pol, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(pol.state_dict(), path + ".pt")
    if getattr(pol, "config", None) is not None:
        json.dump(pol.config, open(path + ".json", "w"), indent=1)


def load(path):
    pol = make_policy(path)
    pol.load_state_dict(torch.load(path + ".pt", weights_only=True))
    pol.eval()
    return pol


def bc_epochs(pol, opt, data, epochs, bs=256, log=print):
    for ep in range(epochs):
        random.shuffle(data)
        tot = n = acc = 0
        for b in range(0, len(data), bs):
            ch = data[b:b + bs]
            ss = pol.batch([x for x, _ in ch])
            loss = 0
            for s, (_, y) in zip(ss, ch):
                loss = loss - torch.log_softmax(s, 0)[y]
                acc += int(s.argmax()) == y
            loss = loss / len(ch)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(ch); n += len(ch)
        log("  bc epoch {} loss {:.4f} acc {:.3f} ({} decisions)".format(ep, tot / n, acc / n, n))
    return tot / max(n, 1)


def evaluate(jobs_from, model_path, procs, mode):
    rows = pool_run([(sp, nm, cert, mode, sd, 0.0, False, nk) for sp, nm, cert, sd, nk in jobs_from], model_path, procs)
    return rows


def summary(rows, certs):
    ok = [r for _, _, r, _ in rows if r["success"]]
    ratio = [r["makespan"] / certs[nm] for nm, _, r, _ in rows if r["success"]]
    return {"runs": len(rows), "success": len(ok) / max(1, len(rows)),
            "served": float(np.mean([r["served"] / r["goals"] for _, _, r, _ in rows])),
            "ms_cert": float(np.mean(ratio)) if ratio else None}


def group_advantages(by):
    """{instance: [(reward, record)]} -> [(record, advantage)]: reward minus the mean of the instance's rollouts
    (instances whose rollouts all score the same carry no signal and are skipped)."""
    out = []
    for lst in by.values():
        R = np.array([r for r, _ in lst])
        adv = R - R.mean()
        if np.allclose(adv, 0):
            continue
        out += [(rec, float(A)) for (_, rec), A in zip(lst, adv) if rec]
    return out


def bc_loss(pol, data, n):
    """Imitation loss on n random DAgger decisions."""
    ch = random.sample(data, min(n, len(data)))
    ss = pol.batch([x for x, _ in ch])
    return sum(-torch.log_softmax(s, 0)[y] for s, (_, y) in zip(ss, ch)) / len(ch)


def reinforce_update(a, pol, opt, by, data, bc_w):
    """One REINFORCE step over the batch (at most max_dec decisions per rollout) -> number of decisions used."""
    n_dec = 0
    pol.train()
    opt.zero_grad()
    groups = {nm: len(lst) for nm, lst in by.items()}
    for nm, lst in by.items():
        for rec, A in group_advantages({nm: lst}):
            sub = rec if len(rec) <= a.max_dec else random.sample(rec, a.max_dec)
            ss = pol.batch([r[0] for r in sub])
            l = 0
            for s, r in zip(ss, sub):
                lp = torch.log_softmax(s, 0)
                l = l - A * lp[r[1]] - a.entropy * -(lp.exp() * lp).sum()
            (l / (len(sub) * groups[nm])).backward()
            n_dec += len(sub)
    if bc_w > 0 and data:
        (bc_w * bc_loss(pol, data, 2048)).backward()
    nn.utils.clip_grad_norm_(pol.parameters(), 1.0)
    opt.step()
    pol.eval()
    return n_dec


def ppo_update(a, pol, opt, by, data, bc_w):
    """PPO with group-relative advantages (no critic): the batch's decisions (at most ppo_decisions, sampled) with the
    log-probabilities they were sampled with; ppo_epochs passes of minibatches of the clipped surrogate objective +
    entropy bonus + imitation regulariser; a pass stops the update when the approximate KL to the sampling policy
    exceeds 1.5 x target_kl. -> (decisions used, log text)."""
    items = [(r[0], r[1], r[3], A) for rec, A in group_advantages(by) for r in rec if r[3] is not None]
    if not items:
        return 0, ", no signal"
    if len(items) > a.ppo_decisions:
        items = random.sample(items, a.ppo_decisions)
    sd = float(np.std([it[3] for it in items])) + 1e-8          # advantages scaled to unit spread
    pol.train()
    kl = clip = 0.0
    passes = 0
    for passes in range(1, a.ppo_epochs + 1):
        random.shuffle(items)
        kls, clips = [], []
        for b in range(0, len(items), a.ppo_batch):
            mb = items[b:b + a.ppo_batch]
            ls = [torch.log_softmax(s, 0) for s in pol.batch([it[0] for it in mb])]
            lp = torch.stack([l[it[1]] for l, it in zip(ls, mb)])
            old = torch.tensor([it[2] for it in mb], dtype=torch.float32)
            adv = torch.tensor([it[3] / sd for it in mb], dtype=torch.float32)
            ratio = torch.exp(lp - old)
            surr = torch.min(ratio * adv, ratio.clamp(1 - a.ppo_clip, 1 + a.ppo_clip) * adv)
            ent = torch.stack([-(l.exp() * l).sum() for l in ls])
            loss = -surr.mean() - a.entropy * ent.mean()
            if bc_w > 0 and data:
                loss = loss + bc_w * bc_loss(pol, data, a.ppo_batch)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(pol.parameters(), 1.0)
            opt.step()
            kls.append(float((old - lp.detach()).mean()))
            clips.append(float(((ratio.detach() - 1).abs() > a.ppo_clip).float().mean()))
        kl, clip = float(np.mean(kls)), float(np.mean(clips))
        if kl > 1.5 * a.target_kl:
            break
    pol.eval()
    return len(items), ", ppo passes {} kl {:.4f} clipped {:.3f}".format(passes, kl, clip)


def cmd_train(a):
    ts = getattr(a, "train_seed", 0)          # 0 reproduces the original models; others add training-seed repeats
    random.seed(ts); torch.manual_seed(ts)
    off = 100000 * ts                         # rollout seeds of DAgger / RL (validation seeds stay fixed)
    trs, vas = DOMAIN.SPLITS
    tr = split_rows(trs)
    va = split_rows(vas)
    certs = {n: c for n, _, c, _ in tr + va}
    path = os.path.join(MODELS, a.name)
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(MODELS, exist_ok=True)
    logf = open(path + "_train.log", "w")

    def log(s):
        print(s, flush=True); logf.write(s + "\n"); logf.flush()
    pol = make_policy()
    opt = torch.optim.Adam(pol.parameters(), lr=1e-3)
    data, hist = [], []

    def val(tag, mode="greedy"):
        save(pol, path)
        rows = evaluate([(vas, n, c, 1000 + s, 1) for n, _, c, _ in va for s in range(a.val_seeds)], path, a.procs, mode)
        sm = summary(rows, certs)
        log("{}: val success {:.3f} served {:.3f} makespan/cert {}".format(tag, sm["success"], sm["served"],
                                                                           None if sm["ms_cert"] is None else round(sm["ms_cert"], 3)))
        hist.append(dict(sm, tag=tag))
        return sm
    # expert reference on val
    rows = evaluate([(vas, n, c, 1000 + s, 1) for n, _, c, _ in va for s in range(a.val_seeds)], None, a.procs, "expert")
    sm = summary(rows, certs)
    log("dispatcher (expert): val success {:.3f} served {:.3f} makespan/cert {:.3f}".format(sm["success"], sm["served"], sm["ms_cert"] or 0))
    hist.append(dict(sm, tag="dispatcher"))
    # ---- DAgger
    for it in range(a.dagger):
        mode, beta = ("expert", 1.0) if it == 0 else ("mix", max(0.0, 0.5 - 0.25 * (it - 1)))
        t0 = time.time()
        jobs = [(trs, n, c, mode, off + 100 * it + s, beta, True, 1) for n, _, c, _ in tr for s in range(a.seeds)]
        if it > 0:
            save(pol, path)
        rows = pool_run(jobs, path if it > 0 else None, a.procs)
        new = [(r[0], r[2]) for _, _, _, rec in rows for r in rec]
        data += new
        sm = summary(rows, certs)
        log("dagger {} ({}, beta {:.2f}): {} new decisions, rollout success {:.3f} ({:.0f}s)".format(
            it, mode, beta, len(new), sm["success"], time.time() - t0))
        bc_epochs(pol, opt, data, a.bc_epochs, log=log)
        val("dagger {}".format(it))
    # ---- RL: PPO (default) or REINFORCE, both with per-instance group baselines and the imitation regulariser
    opt = torch.optim.Adam(pol.parameters(), lr=a.rl_lr)
    shared = a.shared_noise == "on" or (a.shared_noise == "auto" and a.rl == "ppo")
    decay = a.bc_decay if a.bc_decay is not None else (0.2 if a.rl == "ppo" else 1.0)
    log("rl: {}, shared duration noise per instance group: {}, imitation weight {} -> {}".format(
        a.rl, shared, a.bc_weight, round(a.bc_weight * decay, 4)))
    best = None
    for ep in range(a.rl_epochs):
        t0 = time.time()
        save(pol, path)
        names = random.sample(tr, min(a.rl_kitchens, len(tr)))
        if shared:      # every rollout of an instance sees the same duration stream; only the policy's sampling differs
            jobs = [(trs, n, c, "sample", off + 5000 + 100 * ep, 0.0, True, 1, off + 5000 + 100 * ep + s)
                    for n, _, c, _ in names for s in range(a.rollouts)]
        else:
            jobs = [(trs, n, c, "sample", off + 5000 + 100 * ep + s, 0.0, True, 1) for n, _, c, _ in names for s in range(a.rollouts)]
        rows = pool_run(jobs, path, a.procs)
        by = defaultdict(list)
        for nm, sd, res, rec in rows:
            by[nm].append((res["reward"], rec))
        bc_w = a.bc_weight * (1.0 + (decay - 1.0) * ep / max(1, a.rl_epochs - 1))
        if a.rl == "ppo":
            n_dec, info = ppo_update(a, pol, opt, by, data, bc_w)
        else:
            n_dec, info = reinforce_update(a, pol, opt, by, data, bc_w), ""
        sm = summary(rows, certs)
        log("rl {}: train sampled success {:.3f} served {:.3f} makespan/cert {} reward {:.3f} ({} decisions{}, {:.0f}s)".format(
            ep, sm["success"], sm["served"], None if sm["ms_cert"] is None else round(sm["ms_cert"], 3),
            float(np.mean([r["reward"] for _, _, r, _ in rows])), n_dec, info, time.time() - t0))
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
    sp = split_rows(a.split)
    certs = {n: c for n, _, c, _ in sp}
    tiers = {n: inf.get("tier", "all") for n, _, _, inf in sp}
    out = {}
    runs = [("dispatcher", None, "expert")]
    if a.name != "dispatcher":                     # --name dispatcher: the rule-based dispatcher alone (a baseline)
        runs.append((a.name, os.path.join(MODELS, a.name), "greedy"))
    for method, model, mode in runs:
        for nk in a.noise:
            rows = evaluate([(a.split, n, c, 7000 + s, nk) for n, _, c, _ in sp for s in range(a.seeds)], model, a.procs, mode)
            per = defaultdict(list)
            for nm, sd, r, _ in rows:
                per[tiers[nm]].append((nm, sd, r, None))
            res = {"all": summary(rows, certs)}
            for tname, rr in per.items():
                res[tname] = summary(rr, certs)
            out["{} noise x{}".format(method, nk)] = res
            print(method, "noise x{}".format(nk), json.dumps({k: {kk: (round(vv, 3) if isinstance(vv, float) else vv)
                                                                   for kk, vv in v.items()} for k, v in res.items()}), flush=True)
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(os.path.join(OUT, "{}_{}.json".format(a.name, a.split)), "w"), indent=1)


def parser(default_name="V13_v3"):
    """-> (parser, train subparser, eval subparser); V14 adds its own arguments to both."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, help="domain module, e.g. overcooked_test.kitchen_v13")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--name", default=default_name); t.add_argument("--procs", type=int, default=11)
    t.add_argument("--seeds", type=int, default=3); t.add_argument("--dagger", type=int, default=3)
    t.add_argument("--bc_epochs", type=int, default=4); t.add_argument("--val_seeds", type=int, default=2)
    t.add_argument("--rl_epochs", type=int, default=20); t.add_argument("--rl_kitchens", type=int, default=40)
    t.add_argument("--rollouts", type=int, default=8); t.add_argument("--rl_lr", type=float, default=3e-4)
    t.add_argument("--entropy", type=float, default=0.01); t.add_argument("--bc_weight", type=float, default=0.5)
    t.add_argument("--max_dec", type=int, default=400, help="REINFORCE: decisions per rollout")
    t.add_argument("--val_every", type=int, default=5)
    t.add_argument("--train_seed", type=int, default=0,
                   help="seed of initialisation, sampling and training rollouts (repeats with other seeds: V14_s1, ...)")
    t.add_argument("--rl", choices=["ppo", "reinforce"], default="ppo", help="RL phase after DAgger")
    t.add_argument("--ppo_epochs", type=int, default=4, help="PPO: passes over each batch")
    t.add_argument("--ppo_batch", type=int, default=512, help="PPO: decisions per minibatch")
    t.add_argument("--ppo_decisions", type=int, default=16384, help="PPO: decisions sampled from each batch")
    t.add_argument("--ppo_clip", type=float, default=0.2)
    t.add_argument("--target_kl", type=float, default=0.02, help="PPO: stop a batch's passes above 1.5 x this")
    t.add_argument("--shared_noise", choices=["auto", "on", "off"], default="auto",
                   help="the same duration stream for all rollouts of an instance (auto: on for PPO, off for REINFORCE)")
    t.add_argument("--bc_decay", type=float, default=None,
                   help="imitation weight at the last RL epoch, as a multiple of --bc_weight (default: 0.2 PPO, 1 REINFORCE)")
    e = sub.add_parser("eval")
    e.add_argument("--name", default=default_name); e.add_argument("--split", default="v3_test")
    e.add_argument("--seeds", type=int, default=5); e.add_argument("--procs", type=int, default=11)
    e.add_argument("--noise", type=float, nargs="*", default=[1.0])
    return ap, t, e


def main():
    ap, _, _ = parser()
    a = ap.parse_args()
    use(a.domain)
    {"train": cmd_train, "eval": cmd_eval}[a.cmd](a)


if __name__ == "__main__":
    main()
