"""T-iFLAX V9: reinforcement learning over a constructive scheduler's macro steps (replaces V8's dispatch rules).

What it does
  V8 builds plans with hand-written dispatch rules. V9 learns the dispatching: the scheduler becomes an RL
  environment whose decision points offer safe macro steps (e.g. "add the next ingredient into pot p", "light a
  full pot", "wash a plate"; safe = every running long action can still finish in time after the step).
  policy       an MLP scores each candidate step from its feature vector (step type, time it takes, slack of
               running actions / deadlines afterwards, resource states, distances, the GNN score of the object it
               uses); softmax over the candidates.
  reward       - makespan / certificate makespan; failure -1.5.
  training     behaviour cloning of the environment's rule (warm start), then REINFORCE with the mean of the
               instance's own rollouts as baseline (rollouts of one instance are compared with each other) +
               entropy bonus; the best validation-greedy policy is kept.
  inference    object choices: the GNN-best one, then the best distinct choices of the scheduler's search; a
               greedy rollout on each (first plan), then sampled rollouts round-robin while time remains; the
               shortest VAL-valid plan. Every plan comes from the policy (rules only rank object choices).

What it needs (supplied by the test harness, e.g. overcooked_test/kitchen_v9.py)
  an environment env with   rollout(choose, record=False) -> (plan or None, [(features, index)])
                            rule_choice(candidates) -> index;  objects() -> objects it uses;  n_features
     where choose(state, candidates, features) -> index
  for planning: a domain object with  envs(task, score) -> [env]  (the GNN-best object choice first) and
                                      search_envs(task, score, deadline_fn) -> [env]  (more choices, best first)

Earlier result (pre-final v2 kitchen benchmark, v2_test, 60 s): 59 / 60 solved, 0.92 x certificate makespan.
Plug-ins in final/: overcooked_test/kitchen_v9.py, factory_test/factory_v9.py (not trained yet).
"""

"""This file is UN-VERIFIED"""

import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn

from .grounding import GroundTask
from .planning import Workspace, _full_files
from .pddlkit import validate_plan
from .v1 import RunResult

FAIL = -1.5


# ==================== policy ===============================================================================
class Policy(nn.Module):
    def __init__(self, n_features=20, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_features, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1))

    def forward(self, F):
        return self.net(torch.as_tensor(F)).squeeze(-1)


def chooser(pol, greedy, rng=None):
    def f(S, cands, F):
        with torch.no_grad():
            z = pol(F)
        if greedy or len(cands) == 1:
            return int(torch.argmax(z))
        p = torch.softmax(z, 0).numpy().astype(np.float64)
        p /= p.sum()
        return int((rng or np.random).choice(len(cands), p=p))
    return f


def evaluate_greedy(pol, eps):
    """eps: [(name, env, certificate makespan, instance dir)] -> (mean reward, failures)."""
    r, fails = [], 0
    for name, env, cert, d in eps:
        plan, _ = env.rollout(chooser(pol, True))
        if plan is None:
            fails += 1
            r.append(FAIL)
        else:
            r.append(-plan.makespan / cert)
    return float(np.mean(r)), fails


# ==================== training ==============================================================================
def train(tr, va, out_prefix, epochs=20, bc_epochs=3, rollouts=8, lr=3e-4, entropy=0.01, hidden=64, seed=0, meta=None):
    """tr / va: [(name, env, certificate makespan, instance dir)]. Saves <out_prefix>.pt and .json; returns the policy."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    print("episodes: train {} val {}".format(len(tr), len(va)), flush=True)
    pol = Policy(tr[0][1].n_features, hidden)
    opt = torch.optim.Adam(pol.parameters(), lr=lr)
    hist = []
    data = []                                    # behaviour cloning of the environment's rule
    for name, env, cert, d in tr:
        _, traj = env.rollout(lambda S, c, F, env=env: env.rule_choice(c), record=True)
        data += traj
    for ep in range(bc_epochs):
        random.shuffle(data)
        tot = 0.0
        for F, j in data:
            if len(F) < 2:
                continue
            loss = nn.functional.cross_entropy(pol(F).unsqueeze(0), torch.tensor([j]))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss)
        vr, vf = evaluate_greedy(pol, va)
        print("bc epoch {}  loss {:.4f}  val reward {:.4f} (fails {})".format(ep + 1, tot / max(1, len(data)), vr, vf), flush=True)
    best, state = evaluate_greedy(pol, va)[0], {k: v.clone() for k, v in pol.state_dict().items()}
    hist.append({"epoch": 0, "val_reward": best})
    rng = np.random.RandomState(seed)
    for ep in range(epochs):
        t0 = time.time()
        random.shuffle(tr)
        rewards, fails = [], 0
        for name, env, cert, d in tr:
            trajs, rs = [], []
            for _ in range(rollouts):
                plan, traj = env.rollout(chooser(pol, False, rng), record=True)
                fails += plan is None
                trajs.append(traj)
                rs.append(FAIL if plan is None else -plan.makespan / cert)
            rs = np.array(rs)
            adv = rs - rs.mean()
            rewards.append(rs.mean())
            if np.allclose(adv, 0):
                continue
            loss = 0.0
            for traj, a in zip(trajs, adv):
                for F, j in traj:
                    if len(F) < 2:
                        continue
                    lp = torch.log_softmax(pol(F), 0)
                    loss = loss - float(a) * lp[j] - entropy * -(lp.exp() * lp).sum()
            opt.zero_grad()
            (loss / rollouts).backward()
            nn.utils.clip_grad_norm_(pol.parameters(), 1.0)
            opt.step()
        vr, vf = evaluate_greedy(pol, va)
        hist.append({"epoch": ep + 1, "train_reward": float(np.mean(rewards)), "train_fail_rate": fails / (len(tr) * rollouts), "val_reward": vr, "val_fails": vf})
        print("rl epoch {:2d}  train reward {:.4f} (fail {:.3f})  val greedy {:.4f} (fails {})  {:.0f}s".format(
            ep + 1, np.mean(rewards), fails / (len(tr) * rollouts), vr, vf, time.time() - t0), flush=True)
        if vr > best + 1e-5:
            best, state = vr, {k: v.clone() for k, v in pol.state_dict().items()}
    pol.load_state_dict(state)
    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
    torch.save(pol.state_dict(), out_prefix + ".pt")
    args = dict(epochs=epochs, bc_epochs=bc_epochs, rollouts=rollouts, lr=lr, entropy=entropy, hidden=hidden, seed=seed, n_features=tr[0][1].n_features)
    with open(out_prefix + ".json", "w") as f:
        json.dump(dict(meta or {}, method="TIFLAX9", best_val_reward=best, history=hist, args=args), f, indent=1)
    print("saved {} (best val greedy reward {:.4f})".format(out_prefix, best))
    return pol


def load_policy(prefix):
    meta = json.load(open(prefix + ".json"))
    pol = Policy(meta["args"].get("n_features", 20), meta["args"]["hidden"])
    pol.load_state_dict(torch.load(prefix + ".pt", weights_only=True))
    pol.eval()
    return pol


# ==================== planner ==============================================================================
class V9Planner:
    """RL dispatch inside object choices: the GNN-best choice first, then the best distinct choices of the
    scheduler's search. Greedy rollout on every choice (first plan), then sampled rollouts round-robin while time
    remains. Plans come from the RL policy only."""

    def __init__(self, domain, scorer, pol, name="TiFLAX_V9", max_samples=400, n_assign=4):
        self.domain, self.scorer, self.pol, self.name, self.max_samples, self.n_assign = domain, scorer, pol, name, max_samples, n_assign

    def solve(self, task, budget_s, mode="deploy", gt=None, graph=None) -> RunResult:
        t0 = time.monotonic()
        el = lambda: time.monotonic() - t0       # noqa: E731
        ws = Workspace()
        attempts, best, first_s, first_ms = [], None, None, None
        try:
            dom, prob_file = _full_files(task, ws)
            gt = gt if gt is not None else GroundTask(task)
            score = {o: p[0] for o, p in self.scorer.scores(task, gt, graph=graph).items()}
            envs = list(self.domain.envs(task, score))

            def check(plan, stage, env):
                nonlocal best, first_s, first_ms
                t1 = el()
                v = validate_plan(dom, prob_file, plan)
                attempts.append({"stage": stage, "n_objects": len(env.objects()), "outcome": "valid" if v.valid else
                                 "invalid_on_full", "planner_s": 0.0, "val_s": round(el() - t1, 3),
                                 "makespan": plan.makespan if v.valid else None, "t_end": round(el(), 3)})
                if v.valid and (best is None or plan.makespan < best.makespan - 1e-6):
                    best = plan
                    if first_s is None:
                        first_s, first_ms = el(), plan.makespan

            def greedy(env, stage):
                g, _ = env.rollout(chooser(self.pol, True))
                if g is not None and (best is None or g.makespan < best.makespan - 1e-6):
                    check(g, stage, env)

            for env in envs:
                greedy(env, "rl_greedy")
            for env in self.domain.search_envs(task, score, lambda: el() > 0.3 * budget_s):
                if len(envs) >= self.n_assign + 1:
                    break
                envs.append(env)
                greedy(env, "rl_greedy_search")
            rng = np.random.RandomState(0)
            cands = {}
            n = 0
            while mode == "deploy" and envs and n < self.max_samples and el() < budget_s - 2.0:
                env = envs[n % len(envs)]
                p, _ = env.rollout(chooser(self.pol, False, rng))
                n += 1
                if p is not None and (id(env) not in cands or p.makespan < cands[id(env)][0].makespan):
                    cands[id(env)] = (p, env)
            for p, env in sorted(cands.values(), key=lambda x: x[0].makespan)[:3]:
                if best is not None and p.makespan >= best.makespan - 1e-6:
                    break
                check(p, "rl_sampled", env)
        finally:
            ws.close()
        return RunResult(
            method=self.name, instance=task.name, budget_s=budget_s, solved=best is not None,
            makespan=best.makespan if best else None, first_makespan=first_ms, first_valid_s=first_s, total_s=el(),
            n_objects=len(best.objects_used() & set(task.objects)) if best else None, n_total=len(task.objects),
            branch="rl" if best else None, attempts=attempts, planner_calls=0, timeouts=0,
            notes=[] if best else ["no plan"], plan_text=best.to_text() if best else None,
            objects=sorted(best.objects_used() & set(task.objects)) if best else None)
