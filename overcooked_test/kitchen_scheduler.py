"""Overlapping constructive scheduler for overcooked_kitchen (one cook; with more cooks only the first one works): station assignment + dispatch rules -> plan.

    plan = schedule(prob, Assignment(...), policy="fill")      # TemporalPlan or None

Unlike make_dataset.serial_schedule (one dish completely after another), the cook keeps working while soups cook:
it prepares the next dish into another pot, chops ingredients and parks them on a free board, scrubs pots and
washes plates, and comes back to dish up before the soup burns and to deliver before the order's deadline.
Every candidate step is checked by a look-ahead: after the step, every cooking soup must still be dished up within
its burn window and delivered before its deadline (simulated with the same timings). The plan is built with the
domain's exact durations and EPS = 0.001 separations, so VAL on the full problem decides validity.

Dispatch, at every decision point (policy "fill"; "strict" never delays a ready soup; "loose" may light a pot
before a clean plate is ready for it):
  1. if a soup is cooking: do the best other step (light > add > scrub > wash > park) if the look-ahead says all
     cooking soups are still served in time (strict: and the step ends before the soup is ready); else serve it
  2. else light a full pot, add the next ingredient of the next dish, scrub a dirty pot, wash a plate, park.

search(prob, ...) tries station assignments (crates, boards, pots, serving window; ranked by a score function,
e.g. a GNN's or distance from the cook) x policies x dish orders, and returns the shortest plans first.
"""

"""This file is UN-VERIFIED"""

import collections
import copy
import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from algorithm.pddlkit import TemporalPlan, TimedAction
from overcooked_kitchen.layout import INGREDIENTS

EPS = 0.001
CRATE_OF = {"onion_crate": "onion", "tomato_crate": "tomato", "carrot_crate": "carrot"}


POLICIES = ("fill", "strict", "loose", "batch")   # loose: no clean-plate guard before lighting; batch: fill the
                                                  # next dish's pot before lighting the current one


class Infeasible(Exception):
    pass


@dataclass
class Assignment:
    crate: Dict[str, str]                 # ingredient -> crate
    boards: List[str]                     # chopping / parking boards, preferred first
    pots: List[str]                       # pots the cook may use, preferred first
    serve: str
    rack: str
    sink: str
    order: List[str] = field(default_factory=list)    # order names in the sequence they are made

    def objects(self):
        return set(self.crate.values()) | set(self.boards) | set(self.pots) | {self.serve, self.rack, self.sink}


class Kitchen:
    """Static data of a problem (HLProblem)."""

    def __init__(self, prob):
        self.prob = prob
        F = set(map(tuple, prob.facts))
        self.F = F
        self.fl = prob.fluents
        self.kind = {o.name: o.attrs.get("kind") for o in prob.objects if "kind" in o.attrs}
        self.type = {o.name: o.type for o in prob.objects}
        self.pose_tile = {f[1]: f[2] for f in F if f[0] == "pose-tile"}
        self.faces = collections.defaultdict(list)
        for f in F:
            if f[0] == "faces":
                self.faces[f[2]].append(f[1])
        for v in self.faces.values():
            v.sort()
        self.cook = sorted(prob.of_type("cook"))[0]
        self.start = prob.obj(self.cook).attrs.get("start_pose") or prob.obj(self.cook).attrs["pose"]
        self.returns = {f[1]: f[2] for f in F if f[0] == "returns-to"}
        self.provides = {f[1]: f[2] for f in F if f[0] == "provides"}
        self.orders = sorted((o for o in prob.objects if o.type == "order"), key=lambda o: o.name)
        self.deadline = {g.fact[1]: float(g.deadline) for g in prob.goals if g.deadline is not None}
        self.recipe = {o.name: o.attrs["recipe"] for o in self.orders}
        self.comp = {r: prob.obj(r).attrs["comp"] for r in set(self.recipe.values())}
        self.max_ing = max(int(f[1][1:]) for f in F if f[0] == "fits") if any(f[0] == "fits" for f in F) else 4
        self._d = {}

    def of_kind(self, *kinds):
        return sorted(s for s, k in self.kind.items() if k in kinds)

    def travel(self, a, b):
        if a == b:
            return 0.0
        return self.fl.get(("travel", a, b))

    def dist(self, a, b):
        """Shortest travel between any pose facing station/pose a and any pose facing station b."""
        key = (a, b)
        if key not in self._d:
            pa = self.faces.get(a) or [a]
            pb = self.faces.get(b) or [b]
            ds = [self.travel(x, y) for x in pa for y in pb]
            ds = [d for d in ds if d is not None]
            self._d[key] = min(ds) if ds else float("inf")
        return self._d[key]


class _State:
    def __init__(self, K: Kitchen, A: Assignment):
        self.t = 0.0
        self.cur = K.start
        self.acts: List[TimedAction] = []
        F = K.F
        self.pot = {}
        for p in A.pots:
            self.pot[p] = {"st": "dirty" if ("pot-dirty", p) in F else "clean", "dish": None, "have": [0, 0, 0]}
        self.parked = {}                                  # board -> (ingredient, dish index)
        self.rack = sorted(f[1] for f in F if f[0] == "plate-at" and f[2] == A.rack and ("plate-clean", f[1]) in F)
        self.dirty = sorted((0.0, f[1]) for f in F if f[0] == "plate-at" and K.kind.get(f[2]) == "dropoff"
                            and ("plate-dirty", f[1]) in F and f[2] == K.returns.get(A.serve))
        self.dishes = []
        for o in A.order:
            r = K.recipe[o]
            need = [ing for k, ing in enumerate(INGREDIENTS) for _ in range(K.comp[r][k])]
            self.dishes.append({"order": o, "recipe": r, "need": need, "pot": None, "lit": False, "served": False,
                                "ready": None, "light": None, "timer": None})
        self.served = {}

    def clone(self):
        c = _State.__new__(_State)
        c.t, c.cur, c.acts = self.t, self.cur, list(self.acts)       # TimedActions are immutable
        c.pot = {p: {"st": v["st"], "dish": v["dish"], "have": list(v["have"])} for p, v in self.pot.items()}
        c.parked, c.rack, c.dirty, c.served = dict(self.parked), list(self.rack), list(self.dirty), dict(self.served)
        c.dishes = [dict(d, need=list(d["need"])) for d in self.dishes]
        return c


class Scheduler:
    def __init__(self, K: Kitchen, A: Assignment, policy="fill"):
        self.K, self.A, self.policy = K, A, policy

    # ---- primitive actions ---------------------------------------------------------------------------
    def act(self, S, name, args, dur, at=None):
        s = S.t if at is None else at
        S.acts.append(TimedAction(round(s, 3), name, tuple(args), float(dur)))
        if at is None:
            S.t = s + dur + EPS
        return s

    def go(self, S, station):
        K = self.K
        cands = [q for q in K.faces[station] if q == S.cur or K.travel(S.cur, q) is not None]
        if not cands:
            raise Infeasible("unreachable " + station)
        q = min(cands, key=lambda q: (K.travel(S.cur, q), q))
        if q != S.cur:
            if K.pose_tile[q] == K.pose_tile[S.cur]:
                self.act(S, "turn", (K.cook, S.cur, q), K.travel(S.cur, q))
            else:
                self.act(S, "move", (K.cook, S.cur, q, K.pose_tile[S.cur], K.pose_tile[q]), K.travel(S.cur, q))
            S.cur = q
        return q

    # ---- macro steps ---------------------------------------------------------------------------------
    def free_board(self, S, near=None):
        bs = [b for b in self.A.boards if b not in S.parked]
        if not bs:
            return None
        if near is None:
            return bs[0]
        return min(bs, key=lambda b: (self.K.dist(near, b), self.A.boards.index(b)))

    def next_dish(self, S):
        for i, d in enumerate(S.dishes):
            if not d["lit"]:
                return i
        return None

    def pot_for(self, S, i):
        d = S.dishes[i]
        if d["pot"] is not None:
            return d["pot"]
        cl = [p for p in self.A.pots if S.pot[p]["st"] == "clean"]
        if not cl:
            return None
        ref = self.A.boards[0]
        return min(cl, key=lambda p: (self.K.dist(ref, p), self.A.pots.index(p)))

    def step_add(self, S, i, pot=None):
        """Add dish i's next ingredient to its pot (chopping it, or fetching a parked one). pot: the clean pot to
        start dish i in (default: the clean pot nearest the first board)."""
        K, d = self.K, S.dishes[i]
        p = d["pot"] or pot or self.pot_for(S, i)
        if p is None or (d["pot"] is None and S.pot[p]["st"] != "clean"):
            raise Infeasible("no clean pot")
        ing = d["need"][0]
        pk = [b for b, v in S.parked.items() if v == (ing, i)] or [b for b, v in S.parked.items() if v[0] == ing]
        if pk:
            b = pk[0]
            l = self.go(S, b)
            self.act(S, "take-chopped", (K.cook, l, b, ing), 1)
            del S.parked[b]
        else:
            b = self.free_board(S, near=self.A.crate[ing])
            if b is None:
                raise Infeasible("no free board")
            l = self.go(S, self.A.crate[ing])
            self.act(S, "take-raw", (K.cook, l, self.A.crate[ing], ing), 1)
            l = self.go(S, b)
            self.act(S, "put-raw", (K.cook, l, b, ing), 1)
            self.act(S, "chop", (K.cook, l, b, ing), K.fl[("chop-steps", ing)])
            self.act(S, "take-chopped", (K.cook, l, b, ing), 1)
        P = S.pot[p]
        if P["st"] == "clean":
            P["st"], P["dish"], d["pot"] = "filling", i, p
        l = self.go(S, p)
        k = INGREDIENTS.index(ing)
        cnt = ["n%d" % x for x in P["have"]]
        nxt = "n%d" % (P["have"][k] + 1)
        args = {0: (cnt[0], nxt, cnt[1], cnt[2]), 1: (cnt[0], cnt[1], nxt, cnt[2]), 2: (cnt[0], cnt[1], cnt[2], nxt)}[k]
        self.act(S, "add-" + ing, (K.cook, l, p) + args, 1)
        P["have"][k] += 1
        d["need"].pop(0)

    def step_park(self, S, i):
        """Chop dish i's next not-yet-parked ingredient and leave it on a free board."""
        K, d = self.K, S.dishes[i]
        parked_i = collections.Counter(v[0] for v in S.parked.values() if v[1] == i)
        todo = list(d["need"])
        for ing, n in parked_i.items():
            for _ in range(n):
                todo.remove(ing)
        if not todo:
            raise Infeasible("nothing to park")
        ing = todo[0]
        b = self.free_board(S, near=self.A.crate[ing])
        if b is None or len(S.parked) >= len(self.A.boards) - 1:      # keep one board free for chopping
            raise Infeasible("no board to park on")
        l = self.go(S, self.A.crate[ing])
        self.act(S, "take-raw", (K.cook, l, self.A.crate[ing], ing), 1)
        l = self.go(S, b)
        self.act(S, "put-raw", (K.cook, l, b, ing), 1)
        self.act(S, "chop", (K.cook, l, b, ing), K.fl[("chop-steps", ing)])
        S.parked[b] = (ing, i)

    def step_light(self, S, i):
        K, d = self.K, S.dishes[i]
        p = d["pot"]
        l = self.go(S, p)
        s = S.t
        d["timer"] = len(S.acts)
        comp = K.comp[d["recipe"]]
        self.act(S, "start-cooking", (K.cook, l, p, d["recipe"]) + tuple("n%d" % x for x in comp), 1, at=s)
        self.act(S, "finish-lighting", (K.cook, l, p, d["recipe"]), 1, at=s + EPS)
        S.t = s + EPS + 1 + EPS
        cs = K.fl[("cook-steps", d["recipe"])]
        self.act(S, "cook", (p, d["recipe"]), cs, at=S.t)
        d["ready"], d["light"], d["lit"] = S.t + cs, s, True
        S.pot[p]["st"] = "cooking"

    def step_scrub(self, S, p):
        K = self.K
        l = self.go(S, p)
        self.act(S, "scrub-pot", (K.cook, l, p, "n0"), K.fl[("scrub-steps", p)])
        S.pot[p] = {"st": "clean", "dish": None, "have": [0, 0, 0]}

    def step_wash(self, S):
        K, A = self.K, self.A
        if not S.dirty:
            raise Infeasible("no dirty plate")
        S.dirty.sort()
        ret, pl = S.dirty.pop(0)
        drop = K.returns[A.serve]
        l = self.go(S, drop)
        S.t = max(S.t, ret + EPS)
        self.act(S, "take-dirty-plate", (K.cook, l, drop, pl), 1)
        l = self.go(S, A.sink)
        self.act(S, "put-in-sink", (K.cook, l, A.sink, pl), 1)
        self.act(S, "wash-plate", (K.cook, l, A.sink, pl), K.fl[("wash-steps", A.sink)])
        self.act(S, "take-from-sink", (K.cook, l, A.sink, pl), 1)
        l = self.go(S, A.rack)
        self.act(S, "rack-plate", (K.cook, l, A.rack, pl), 1)
        S.rack.append(pl)

    def step_serve(self, S, i):
        K, A, d = self.K, self.A, S.dishes[i]
        if not S.rack:
            self.step_wash(S)
        pl = S.rack.pop(0)
        l = self.go(S, A.rack)
        self.act(S, "take-clean-plate", (K.cook, l, A.rack, pl), 1)
        p = d["pot"]
        l = self.go(S, p)
        S.t = max(S.t, d["ready"] + EPS)
        dish_end = self.act(S, "dish-up", (K.cook, l, p, d["recipe"], pl), 1) + 1
        timer = dish_end + EPS - d["light"]
        if timer > K.fl[("burn-window", d["recipe"])] + 1e-9:
            raise Infeasible("burnt")
        a = S.acts[d["timer"]]
        S.acts[d["timer"]] = TimedAction(a.start, a.name, a.args, round(timer, 3))
        S.pot[p] = {"st": "dirty", "dish": None, "have": [0, 0, 0]}
        l = self.go(S, A.serve)
        end = self.act(S, "deliver", (K.cook, l, A.serve, d["order"], d["recipe"], pl), 1) + 1
        dl = K.deadline.get(d["order"])
        if dl is not None and end > dl - EPS:
            raise Infeasible("late")
        eat = K.fl[("eat-steps", A.serve)]
        self.act(S, "diner-eats", (A.serve, K.returns[A.serve], pl), eat, at=end + EPS)
        S.dirty.append((end + EPS + eat, pl))
        d["served"] = True
        S.served[d["order"]] = end

    # ---- dispatch -------------------------------------------------------------------------------------
    def cooking(self, S):
        return sorted((d["ready"], i) for i, d in enumerate(S.dishes) if d["lit"] and not d["served"])

    def other_steps(self, S):
        """Candidate non-serving steps, best first: [(name, fn)]."""
        out = []
        i = self.next_dish(S)
        if i is not None:
            d = S.dishes[i]
            p = self.pot_for(S, i)
            lit = sum(1 for x in S.dishes if x["lit"] and not x["served"])
            guard = self.policy != "loose"
            if p is not None and not d["need"] and (len(S.rack) > lit or not guard):   # a plate waits for every soup
                if self.policy == "batch":
                    # fill the next dish's pot first, then light both close together
                    j = next((j for j in range(i + 1, len(S.dishes)) if not S.dishes[j]["lit"]), None)
                    if j is not None and S.dishes[j]["need"] and (S.dishes[j]["pot"] is not None or any(
                            S.pot[q]["st"] == "clean" for q in self.A.pots)):
                        out.append(("add-next", lambda S, j=j: self.step_add(S, j)))
                out.append(("light", lambda S, i=i: self.step_light(S, i)))
            elif p is not None and d["need"] and sum(S.pot[p]["have"]) < self.K.max_ing:
                out.append(("add", lambda S, i=i: self.step_add(S, i)))
        for p in self.A.pots:
            if S.pot[p]["st"] == "dirty":
                out.append(("scrub", lambda S, p=p: self.step_scrub(S, p)))
                break
        need_plates = sum(1 for d in S.dishes if not d["served"])
        if S.dirty and len(S.rack) < need_plates:
            lit = sum(1 for x in S.dishes if x["lit"] and not x["served"])
            w = ("wash", lambda S: self.step_wash(S))
            if len(S.rack) <= lit:
                out.insert(0, w)                    # the next soup has no plate yet: wash first
            else:
                out.append(w)
        if i is not None:
            for j in range(i, len(S.dishes)):
                if S.dishes[j]["pot"] is None or j == i:
                    out.append(("park", lambda S, j=j: self.step_park(S, j)))
                    break
        return out

    def serve_all_ok(self, S):
        """Look-ahead: serving every cooking soup now, in ready order, burns nothing and misses no deadline."""
        T = S.clone()
        try:
            for _, i in self.cooking(T):
                self.step_serve(T, i)
        except Infeasible:
            return False
        return True

    def run(self):
        S = _State(self.K, self.A)
        guard = 0
        while not all(d["served"] for d in S.dishes):
            guard += 1
            if guard > 2000:
                raise Infeasible("no progress")
            cook = self.cooking(S)
            done = False
            for name, fn in self.other_steps(S):
                T = S.clone()
                try:
                    fn(T)
                except Infeasible:
                    continue
                if cook:
                    if self.policy == "strict" and T.t > cook[0][0] + EPS:
                        continue
                    if not self.serve_all_ok(T):
                        continue
                S, done = T, True
                break
            if done:
                continue
            if not cook:
                raise Infeasible("stuck")
            self.step_serve(S, cook[0][1])
        plan = TemporalPlan(sorted(S.acts, key=lambda a: (a.start, a.name, a.args)))
        plan.served = S.served
        return plan


def schedule(prob_or_kitchen, A: Assignment, policy="fill") -> Optional[TemporalPlan]:
    K = prob_or_kitchen if isinstance(prob_or_kitchen, Kitchen) else Kitchen(prob_or_kitchen)
    try:
        return Scheduler(K, A, policy).run()
    except Infeasible:
        return None


# ---- search over assignments ------------------------------------------------------------------------------
def candidates(K: Kitchen, score=None, k=3):
    """Per station kind, up to k candidates ranked by score (higher first) or by distance from the cook."""
    def rank(objs):
        by_dist = sorted(objs, key=lambda o: (K.dist(K.start, o), o))
        if score is None:
            return by_dist
        by_score = sorted(objs, key=lambda o: (-score.get(o, 0.0), K.dist(K.start, o), o))
        out = []                               # interleave: learned first, then distance, alternating
        for a, b in itertools.zip_longest(by_score, by_dist):
            for x in (a, b):
                if x is not None and x not in out:
                    out.append(x)
        return out
    crates = {}
    for c, ing in K.provides.items():
        crates.setdefault(ing, []).append(c)
    need = sorted({ing for r in K.comp for k_, ing in enumerate(INGREDIENTS) if K.comp[r][k_]}, key=INGREDIENTS.index)
    return {"crate": {ing: rank(crates.get(ing, []))[:k] for ing in need},
            "board": rank(K.of_kind("board"))[:k + 1],
            "pot": rank(K.of_kind("pot"))[:k + 1],
            "serve": rank(K.of_kind("serving"))[:k],
            "rack": rank(K.of_kind("rack"))[:k],
            "sink": rank(K.of_kind("sink"))[:k]}


def orders_to_try(K: Kitchen, limit=12):
    names = [o.name for o in K.orders]
    if K.deadline:
        yield sorted(names, key=lambda o: (K.deadline.get(o, float("inf")), o))
        return
    yield names
    seen = {tuple(names)}
    by_cook = sorted(names, key=lambda o: (-K.fl[("cook-steps", K.recipe[o])], o))      # long soups first
    for cand in [by_cook, list(reversed(by_cook))] + [list(p) for p in itertools.islice(itertools.permutations(names), 24)]:
        if tuple(cand) not in seen and len(seen) < limit:
            seen.add(tuple(cand))
            yield cand


def search(prob, score=None, k=3, max_runs=400, deadline_fn=None):
    """-> list of (makespan, plan, Assignment, policy), shortest first. Coordinate descent over station choices,
    starting from the best-ranked station of every kind, for 1..k pots and 1..k boards, both policies."""
    K = Kitchen(prob)
    C = candidates(K, score, k)
    if any(not v for v in C["crate"].values()) or not C["board"] or not C["pot"] or not C["serve"] \
            or not C["rack"] or not C["sink"]:
        return []
    results, runs, tried = [], 0, set()

    def run(choice, npots, nboards, order, policy):
        nonlocal runs
        key = (tuple(sorted(choice["crate"].items())), choice["serve"], choice["rack"], choice["sink"],
               tuple(choice["pots"][:npots]),
               tuple(choice["boards"][:nboards]), tuple(order), policy)
        if key in tried:
            return None
        tried.add(key)
        runs += 1
        A = Assignment(dict(choice["crate"]), choice["boards"][:nboards], choice["pots"][:npots], choice["serve"],
                       choice["rack"], choice["sink"], list(order))
        plan = schedule(K, A, policy)
        if plan is not None:
            results.append((plan.makespan, plan, A, policy))
            return plan.makespan
        return None

    base = {"crate": {ing: v[0] for ing, v in C["crate"].items()}, "serve": C["serve"][0],
            "pots": list(C["pot"]), "boards": list(C["board"]), "rack": C["rack"][0], "sink": C["sink"][0]}
    orders = list(orders_to_try(K))
    for order, npots, nboards, policy in itertools.product(orders, range(len(C["pot"]), 0, -1),
                                                           range(len(C["board"]), 0, -1), POLICIES):
        run(base, npots, nboards, order, policy)
        if deadline_fn and deadline_fn():
            break
    if not results:
        return []
    # coordinate descent on the best configuration: swap one station choice at a time
    improved = True
    while improved and runs < max_runs and not (deadline_fn and deadline_fn()):
        improved = False
        ms0, _, A0, pol0 = min(results, key=lambda r: r[0])
        cur = {"crate": dict(A0.crate), "serve": A0.serve, "pots": list(A0.pots), "boards": list(A0.boards),
               "rack": A0.rack, "sink": A0.sink}
        moves = []
        for ing, alts in C["crate"].items():
            for c in alts:
                if c != cur["crate"][ing]:
                    moves.append(("crate", ing, c))
        moves += [("serve", None, s) for s in C["serve"] if s != cur["serve"]]
        moves += [("rack", None, s) for s in C["rack"] if s != cur["rack"]]
        moves += [("sink", None, s) for s in C["sink"] if s != cur["sink"]]
        moves += [("pots", None, p) for p in C["pot"] if p not in cur["pots"]]
        moves += [("boards", None, b) for b in C["board"] if b not in cur["boards"]]
        for kind, ing, v in moves:
            ch = copy.deepcopy(cur)
            if kind == "crate":
                ch["crate"][ing] = v
            elif kind in ("serve", "rack", "sink"):
                ch[kind] = v
            elif kind == "pots":
                for j in range(len(ch["pots"])):          # replace each used pot by v
                    c2 = copy.deepcopy(ch)
                    c2["pots"][j] = v
                    ms = run(c2, len(c2["pots"]), len(c2["boards"]), A0.order, pol0)
                    if ms is not None and ms < ms0 - 1e-6:
                        improved = True
                continue
            else:
                for j in range(len(ch["boards"])):
                    c2 = copy.deepcopy(ch)
                    c2["boards"][j] = v
                    ms = run(c2, len(c2["pots"]), len(c2["boards"]), A0.order, pol0)
                    if ms is not None and ms < ms0 - 1e-6:
                        improved = True
                continue
            ms = run(ch, len(ch["pots"]), len(ch["boards"]), A0.order, pol0)
            if ms is not None and ms < ms0 - 1e-6:
                improved = True
            if runs >= max_runs:
                break
    results.sort(key=lambda r: r[0])
    return results


# ==================== the scheduler protocol of algorithm/v8.py ==============================================
class KitchenScheduler:
    """first_plans / search for algorithm/v8.py (first plans: best-ranked station of every kind, top-k boards / pots,
    every dispatch policy; with a learned score also the scheduler's own distance ranking)."""

    def first_plans(self, problem, score, k):
        K = Kitchen(problem)
        C = candidates(K, score, k)
        if not (all(C["crate"].values()) and C["board"] and C["pot"] and C["serve"] and C["rack"] and C["sink"]):
            return []
        orders = next(orders_to_try(K))
        firsts = []
        bases = [C] if score is None else [C, candidates(K, None, k)]    # learned, then distance
        for B in bases:
            for pol in POLICIES:
                A = Assignment({i: v[0] for i, v in B["crate"].items()}, list(B["board"]), list(B["pot"]),
                                  B["serve"][0], B["rack"][0], B["sink"][0], list(orders))
                p = schedule(K, A, pol)
                if p is not None:
                    firsts.append((p.makespan, p, A))
        return [(p, A) for _, p, A in sorted(firsts, key=lambda x: x[0])]

    def search(self, problem, score, k, max_runs=4000, deadline_fn=None):
        return search(problem, score, k, max_runs=max_runs, deadline_fn=deadline_fn)
