"""V9's kitchen side: the kitchen scheduler (kitchen_scheduler.py) as an RL environment over macro steps, its 20 step features, the
episodes (GNN-selected station spaces) and the CLI that trains / compares algorithm/v9.py on the kitchens.

    python -m overcooked_test.kitchen_v9 train [--epochs 20 ...]     # -> results/models/TIFLAX9_overcooked_kitchen_seed0.{pt,json}
    python -m overcooked_test.kitchen_v9 compare --split v2_test     # RL greedy / sampled vs the scheduler's rules

Macro steps at a decision point (only *safe* ones: after the step every cooking soup can still be served before it
burns and before its deadline): add the next ingredient of one of the next two dishes (for a dish without a pot:
into which clean pot), light a full pot, park a chopped ingredient, scrub a dirty pot, wash a plate, serve a cooking
soup. Features (20): step type (6), time it takes, soups cooking, burn slack, deadline slack, dish rank, dish
progress, GNN score of the station, distance, counts of cooking / filling pots, clean / dirty plates, open dishes,
wait before serving.
"""
import argparse
import json
import os

import numpy as np

import overcooked_test  # noqa: F401  (paths)
from overcooked_test import RESULTS  # noqa: E402
from overcooked_test import kitchen_scheduler as SC
from algorithm import split_dirs as _split_dirs
from algorithm.gnn import Scorer
from algorithm.grounding import GroundTask, Task
from algorithm.pddlkit import TemporalPlan
from algorithm import v9
from algorithm.v9 import chooser, load_policy

MODELS = os.path.join(RESULTS, "models")
TEACHER = os.path.join(RESULTS, "teacher")
BENCH = "overcooked_kitchen"
TYPES = ("add", "light", "park", "scrub", "wash", "serve")
NF = 20


def split_dirs(bench, split):
    return _split_dirs(split, bench)


# ==================== environment ==========================================================================
class Env(SC.Scheduler):
    """The scheduler as an RL environment over macro steps."""
    n_features = NF

    def __init__(self, K, A, score=None):
        super().__init__(K, A, "fill")
        self.score = score or {}

    def objects(self):
        return self.A.objects()

    def unlit(self, S, w=2):
        return [i for i, d in enumerate(S.dishes) if not d["lit"]][:w]

    def raw_candidates(self, S):
        """[(type, dish or pot, station, fn)] before the safety check."""
        out = []
        for i in self.unlit(S):
            d = S.dishes[i]
            if d["need"]:
                if d["pot"] is not None:
                    out.append(("add", i, d["pot"], lambda S, i=i: self.step_add(S, i)))
                else:
                    taken = {x["pot"] for x in S.dishes if x["pot"] is not None and not x["served"]}
                    for p in [p for p in self.A.pots if S.pot[p]["st"] == "clean" and p not in taken][:3]:
                        out.append(("add", i, p, lambda S, i=i, p=p: self.step_add(S, i, p)))
                out.append(("park", i, None, lambda S, i=i: self.step_park(S, i)))
            elif d["pot"] is not None and S.pot[d["pot"]]["st"] == "filling":
                out.append(("light", i, d["pot"], lambda S, i=i: self.step_light(S, i)))
        for p in [p for p in self.A.pots if S.pot[p]["st"] == "dirty"][:2]:
            out.append(("scrub", None, p, lambda S, p=p: self.step_scrub(S, p)))
        if S.dirty and len(S.rack) < sum(1 for d in S.dishes if not d["served"]):
            out.append(("wash", None, self.A.sink, lambda S: self.step_wash(S)))
        for _, i in self.cooking(S)[:2]:
            out.append(("serve", i, S.dishes[i]["pot"], lambda S, i=i: self.step_serve(S, i)))
        return out

    def candidates(self, S):
        """Safe candidates: [(type, key, station, next state)]."""
        out = []
        for typ, key, st, fn in self.raw_candidates(S):
            Tn = S.clone()
            try:
                fn(Tn)
            except SC.Infeasible:
                continue
            if self.cooking(Tn) and not self.serve_all_ok(Tn):      # a soup cooking after the step must stay safe
                continue
            out.append((typ, key, st, Tn))
        return out

    def features(self, S, cand):
        K = self.K
        typ, key, st, Tn = cand
        f = np.zeros(NF, dtype=np.float32)
        f[TYPES.index(typ)] = 1.0
        f[6] = (Tn.t - S.t) / 20.0
        f[7] = 1.0 if self.cooking(S) else 0.0
        sl = [(d["light"] + K.fl[("burn-window", d["recipe"])] - max(Tn.t, d["ready"])) / 20.0
              for d in Tn.dishes if d["lit"] and not d["served"]]
        f[8] = min(sl) if sl else 1.5
        dl = [(K.deadline[d["order"]] - Tn.t) / 100.0 for d in Tn.dishes if not d["served"] and d["order"] in K.deadline]
        f[9] = float(np.clip(min(dl), -1, 3)) if dl else 3.0
        unl = self.unlit(S)
        f[10] = float(unl.index(key)) if (key is not None and typ in ("add", "park", "light") and key in unl) else 0.0
        if typ in ("add", "park", "light", "serve") and key is not None:
            d = S.dishes[key]
            n = sum(K.comp[d["recipe"]])
            f[11] = 1.0 - len(d["need"]) / max(1, n)
        f[12] = self.score.get(st, 0.5) if st else 0.5
        f[13] = (K.dist(self.A.boards[0], st) / 10.0) if st else 0.0
        f[14] = len(self.cooking(Tn)) / 3.0
        f[15] = sum(1 for v in Tn.pot.values() if v["st"] == "filling") / 3.0
        f[16] = len(Tn.rack) / 3.0
        f[17] = len(Tn.dirty) / 3.0
        f[18] = sum(1 for d in Tn.dishes if not d["served"]) / max(1, len(Tn.dishes))
        if typ == "serve":
            f[19] = max(0.0, S.dishes[key]["ready"] - S.t) / 20.0
        return f

    def rule_choice(self, cands):
        """V8's "fill" preference among safe candidates: light > add (nearest pot, current dish) > scrub > wash
        > park > serve."""
        pri = {"light": 0, "add": 1, "scrub": 2, "wash": 3, "park": 4, "serve": 5}
        return min(range(len(cands)), key=lambda j: (pri[cands[j][0]], cands[j][1] if cands[j][1] is not None else 0,
                                                     self.K.dist(self.A.boards[0], cands[j][2]) if cands[j][2] else 0))

    def rollout(self, choose, record=False, max_steps=400):
        """choose(S, cands, feats) -> index. -> (plan or None, [(feats, index)] if record)."""
        S = SC._State(self.K, self.A)
        traj = []
        for _ in range(max_steps):
            if all(d["served"] for d in S.dishes):
                plan = TemporalPlan(sorted(S.acts, key=lambda a: (a.start, a.name, a.args)))
                plan.served = S.served
                return plan, traj
            cands = self.candidates(S)
            if not cands:
                return None, traj
            F = np.stack([self.features(S, c) for c in cands])
            j = choose(S, cands, F)
            if record:
                traj.append((F, j))
            S = cands[j][3]
        return None, traj



# ==================== episodes: GNN station space ================================================================
def gnn_scores(task, scorer):
    gt = GroundTask(task)
    return {o: p[0] for o, p in scorer.scores(task, gt).items()}


def assignment(K, score, k=4):
    """The GNN-selected station space: best-ranked crate per vegetable, serving window, rack and sink; the top-k
    boards and pots (learned ranking interleaved with distance, scheduler.candidates)."""
    C = SC.candidates(K, score, k)
    if not (all(C["crate"].values()) and C["board"] and C["pot"] and C["serve"] and C["rack"] and C["sink"]):
        return None
    return SC.Assignment({i: v[0] for i, v in C["crate"].items()}, list(C["board"]), list(C["pot"]), C["serve"][0],
                         C["rack"][0], C["sink"][0], list(next(SC.orders_to_try(K))))


def load_episodes(split, scorer, cache):
    """[(name, Env, certificate makespan)]; GNN scores cached in `cache` (json)."""
    sc_cache = json.load(open(cache)) if os.path.exists(cache) else {}
    out = []
    for d in split_dirs(BENCH, split):
        name = os.path.basename(d)
        task = Task.load(d)
        if name not in sc_cache:
            sc_cache[name] = gnn_scores(task, scorer)
        K = SC.Kitchen(task.problem)
        A = assignment(K, sc_cache[name])
        if A is None:
            continue
        cert = TemporalPlan.parse(open(os.path.join(d, "certificate.plan")).read()).makespan
        out.append((name, Env(K, A, sc_cache[name]), cert, d))
    with open(cache, "w") as f:
        json.dump(sc_cache, f)
    return out


def rule_best(env):
    """Shortest plan of V8's four dispatch policies on the same stations (the rules V9 has to beat)."""
    best = None
    for pol in SC.POLICIES:
        p = SC.schedule(env.K, env.A, pol)
        if p is not None and (best is None or p.makespan < best.makespan):
            best = p
    return best



def compare(args):
    """Greedy RL policy vs the best of V8's four rules, on the same GNN-selected stations; plus sampled RL."""
    scorer = teacher_scorer(args.split)
    eps = load_episodes(args.split, scorer, os.path.join(MODELS, "v9_scores_overcooked_kitchen_{}.json".format(args.split)))
    pol = load_policy(os.path.join(MODELS, "TIFLAX9_overcooked_kitchen_seed{}".format(args.seed)))
    rng = np.random.RandomState(0)
    rows = []
    for name, env, cert, d in eps:
        rule = rule_best(env)
        g, _ = env.rollout(chooser(pol, True))
        samp = [env.rollout(chooser(pol, False, rng))[0] for _ in range(args.samples)]
        samp = [p for p in samp + [g] if p is not None]
        sb = min(samp, key=lambda p: p.makespan) if samp else None
        rows.append({"instance": name, "cert": cert, "rule": rule.makespan if rule else None,
                     "rl_greedy": g.makespan if g else None, "rl_sampled": sb.makespan if sb else None})
        print("  {:<24} cert {:6.1f}  rules {}  RL greedy {}  RL best of {} {}".format(
            name, cert, "%6.1f" % rule.makespan if rule else "  fail", "%6.1f" % g.makespan if g else "  fail",
            args.samples + 1, "%6.1f" % sb.makespan if sb else "  fail"), flush=True)

    def agg(k):
        ok = [r for r in rows if r[k] is not None]
        return len(ok), float(np.mean([r[k] / r["cert"] for r in ok])) if ok else float("nan")
    both = [r for r in rows if r["rule"] is not None and r["rl_greedy"] is not None]
    win = sum(r["rl_greedy"] < r["rule"] - 1e-6 for r in both)
    lose = sum(r["rl_greedy"] > r["rule"] + 1e-6 for r in both)
    print("solved / mean makespan / cert:  rules {} {:.4f}   RL greedy {} {:.4f}   RL sampled {} {:.4f}".format(
        *agg("rule"), *agg("rl_greedy"), *agg("rl_sampled")))
    print("RL greedy vs rules on {} common kitchens: shorter {}, longer {}, equal {}".format(
        len(both), win, lose, len(both) - win - lose))
    with open(os.path.join(TEACHER, BENCH, "v9_compare_{}.json".format(args.split)), "w") as f:
        json.dump(rows, f, indent=1)


class KitchenDomain:
    """The planning side of V9's protocol: environments for a task's station choices."""

    def __init__(self, n_assign=4):
        self.n_assign = n_assign

    def _key(self, A):
        return (tuple(sorted(A.crate.items())), tuple(A.boards), tuple(A.pots), A.serve, A.rack, A.sink, tuple(A.order))

    def envs(self, task, score):
        self.K, self.seen = SC.Kitchen(task.problem), set()
        A0 = assignment(self.K, score)
        if A0 is None:
            return []
        self.seen.add(self._key(A0))
        return [Env(self.K, A0, score)]

    def search_envs(self, task, score, deadline_fn):
        """Station choices from the scheduler's search (ranked by its rule-based makespans), new ones only."""
        for ms, p, A, pol in SC.search(task.problem, score, 4, max_runs=2000, deadline_fn=deadline_fn):
            if A is not None and self._key(A) not in self.seen:
                self.seen.add(self._key(A))
                yield Env(self.K, A, score)


def teacher_scorer(split="v2_train", seed=0):
    return Scorer.load(os.path.join(MODELS, "TEACHER_overcooked_kitchen_seed{}".format(seed)), Task.load(split_dirs(BENCH, split)[0]).spec)


def train(args):
    scorer = teacher_scorer()
    tr = load_episodes("v2_train", scorer, os.path.join(MODELS, "v9_scores_overcooked_kitchen_v2_train.json"))
    va = load_episodes("v2_val", scorer, os.path.join(MODELS, "v9_scores_overcooked_kitchen_v2_val.json"))
    v9.train(tr, va, os.path.join(MODELS, "TIFLAX9_overcooked_kitchen_seed{}".format(args.seed)), epochs=args.epochs, bc_epochs=args.bc_epochs,
             rollouts=args.rollouts, lr=args.lr, entropy=args.entropy, hidden=args.hidden, seed=args.seed, meta={"bench": BENCH})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--epochs", type=int, default=20)
    t.add_argument("--bc_epochs", type=int, default=3)
    t.add_argument("--rollouts", type=int, default=8)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--entropy", type=float, default=0.01)
    t.add_argument("--hidden", type=int, default=64)
    t.add_argument("--seed", type=int, default=0)
    c = sub.add_parser("compare")
    c.add_argument("--split", default="v2_test")
    c.add_argument("--samples", type=int, default=31)
    c.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    {"train": train, "compare": compare}[args.cmd](args)


if __name__ == "__main__":
    main()
