"""V10: a domain-independent temporal planner for PDDL 2.1 durative actions (no kitchen code).

    python -m algorithm.v10 solve domain.pddl problem.pddl [--budget 60] [--out plan.txt]

Input  : any domain/problem with typed STRIPS durative actions (also instantaneous :action),
         positive/negative literal conditions, durations given by numbers / static functions /
         arithmetic or by inequalities (flexible durations), timed initial literals.
         Not supported (refused with an error): numeric conditions/effects, quantifiers,
         conditional effects, disjunctive or negative goals.
Search : decision epochs over snap actions with explicit time. A state is
         (facts, time, running actions with [earliest, latest] end, next timed literal).
         Moves: START an applicable action now, END a running flexible action now, or WAIT to
         the next event (fixed-duration ends, flexible deadlines and timed literals there are
         applied automatically). Happenings are 0.001 apart, so the mutex rule holds trivially;
         over-all conditions are protected against every later happening.
Heuristic: temporal relaxed planning graph from the state (running actions' end effects and
         timed literals arrive at their times; facts deleted by a timed literal and never
         re-added must be used before then) -> FF relaxed-plan size, helpful moves, dead ends.
Strategy: enforced hill-climbing on helpful moves (as FF / POPF), then greedy best-first
         search; the plan is VAL-checked and re-timed (earliest start) afterwards.
"""
import argparse
import heapq
import itertools
import math
import re
import time
from collections import defaultdict, deque

INF = math.inf
SEP = 0.001          # separation between consecutive happenings (VAL tolerance 0.001, as POPF)


class Unsupported(Exception):
    pass


# ---------------------------------------------------------------- PDDL parsing
def _sexpr(text):
    text = re.sub(r";[^\n]*", "", text).lower()
    toks = re.findall(r"\(|\)|[^\s()]+", text)
    stack, cur = [], []
    for t in toks:
        if t == "(":
            stack.append(cur); cur = []
        elif t == ")":
            done, cur = cur, stack.pop(); cur.append(done)
        else:
            cur.append(t)
    return cur[0]


def _typed(lst, default="object"):
    """['a','b','-','t','c'] -> [('a','t'),('b','t'),('c','object')]"""
    out, pend, i = [], [], 0
    while i < len(lst):
        if lst[i] == "-":
            ty = lst[i + 1]
            if isinstance(ty, list):
                raise Unsupported("(either ...) types")
            out += [(n, ty) for n in pend]; pend = []; i += 2
        else:
            pend.append(lst[i]); i += 1
    return out + [(n, default) for n in pend]


def _lits(e, where):
    """Conjunction of literals -> list of (positive?, atom tuple)."""
    if not e:
        return []
    if e[0] == "and":
        return [l for x in e[1:] for l in _lits(x, where)]
    if e[0] == "not":
        if e[1][0] in ("=",):
            raise Unsupported("equality in " + where)
        return [(False, tuple(e[1]))]
    if e[0] in ("forall", "exists", "when", "or", "imply", "increase", "decrease", "assign",
                "scale-up", "scale-down", "<", ">", "<=", ">=", "="):
        raise Unsupported("'{}' in {}".format(e[0], where))
    return [(True, tuple(e))]


def _timed(e, where, kinds):
    out = {k: [] for k in kinds}
    if not e:
        return out
    items = e[1:] if e[0] == "and" else [e]
    for x in items:
        k = x[0] + " " + x[1] if x[0] in ("at", "over") else None
        if k not in out:
            raise Unsupported("untimed {} '{}'".format(where, x))
        out[k] += _lits(x[2], where)
    return out


class Schema:
    def __init__(self, name, params, dur, cond, eff, instant=False):
        self.name, self.params, self.dur, self.cond, self.eff, self.instant = \
            name, params, dur, cond, eff, instant


class Domain:
    def __init__(self, text):
        e = _sexpr(text)
        self.name = e[1][1]
        self.parent, self.constants, self.schemas, self.preds = {"object": None}, {}, [], set()
        for item in e[2:]:
            h = item[0]
            if h == ":types":
                for n, t in _typed(item[1:]):
                    self.parent[n] = t
                    self.parent.setdefault(t, "object" if t != "object" else None)
            elif h == ":constants":
                self.constants.update(_typed(item[1:]))
            elif h == ":predicates":
                self.preds |= {p[0] for p in item[1:]}
            elif h == ":durative-action":
                kv = dict(zip(item[2::2], item[3::2]))
                params = _typed(kv.get(":parameters", []))
                cond = _timed(kv.get(":condition", []), "condition", ("at start", "over all", "at end"))
                eff = _timed(kv.get(":effect", []), "effect", ("at start", "at end"))
                self.schemas.append(Schema(item[1], params, self._dur(kv[":duration"]), cond, eff))
            elif h == ":action":
                kv = dict(zip(item[2::2], item[3::2]))
                params = _typed(kv.get(":parameters", []))
                cond = {"at start": _lits(kv.get(":precondition", []), "precondition"),
                        "over all": [], "at end": []}
                eff = {"at start": _lits(kv.get(":effect", []), "effect"), "at end": []}
                self.schemas.append(Schema(item[1], params, (0, 0), cond, eff, instant=True))
        self.static = self.preds - {a[0] for s in self.schemas for k in s.eff for _, a in s.eff[k]}

    @staticmethod
    def _dur(d):
        """-> (lo expr, hi expr); expr = number | ?var-free term tree."""
        if d[0] == "=":
            return (d[2], d[2])
        if d[0] == "and":
            lo = hi = None
            for c in d[1:]:
                if c[0] in (">=", ">"):
                    lo = c[2]
                elif c[0] in ("<=", "<"):
                    hi = c[2]
            if hi is None:
                raise Unsupported("unbounded duration")
            return (lo if lo is not None else "0", hi)
        if d[0] in ("<=", "<"):
            return ("0", d[2])
        raise Unsupported("duration " + str(d))

    def ancestors(self, t):
        while t is not None:
            yield t
            t = self.parent.get(t, "object" if t != "object" else None)


class Problem:
    def __init__(self, text):
        e = _sexpr(text)
        self.name = e[1][1]
        self.objects, self.init, self.fluents, self.tils, self.goals = {}, set(), {}, [], []
        for item in e[2:]:
            h = item[0]
            if h == ":objects":
                self.objects.update(_typed(item[1:]))
            elif h == ":init":
                for x in item[1:]:
                    if x[0] == "=":
                        self.fluents[tuple(x[1]) if isinstance(x[1], list) else (x[1],)] = float(x[2])
                    elif x[0] == "at" and len(x) == 3 and re.fullmatch(r"[0-9.]+", str(x[1])):
                        pos, atom = _lits(x[2], "timed literal")[0]
                        self.tils.append((float(x[1]), atom, pos))
                    else:
                        self.init.add(tuple(x))
            elif h == ":goal":
                for pos, atom in _lits(item[1], "goal"):
                    if not pos:
                        raise Unsupported("negative goal")
                    self.goals.append(atom)
        self.tils.sort()


# ---------------------------------------------------------------- grounding
class Action:
    __slots__ = ("i", "name", "args", "lo", "hi", "instant", "cs", "csn", "co", "con", "ce", "cen",
                 "as_", "ds", "ae", "de", "pre")

    def __str__(self):
        return "(" + " ".join((self.name,) + self.args) + ")"


class Task:
    """Grounded task: facts are ints, actions carry frozensets of fact ids."""

    def __init__(self, dom: Domain, prob: Problem):
        self.dom, self.prob = dom, prob
        self.fid, self.facts = {}, []
        objs = dict(dom.constants); objs.update(prob.objects)
        typed = defaultdict(list)
        for n in sorted(objs):
            for a in dom.ancestors(objs[n]):
                typed[a].append(n)
        self.static = dom.static - {a[0] for _, a, _ in prob.tils}   # timed literals make a predicate dynamic
        init_by = defaultdict(list)
        for f in prob.init:
            init_by[f[0]].append(f)
        self.static_init = {f for f in prob.init if f[0] in self.static}
        raw = []
        for s in dom.schemas:
            raw += self._ground(s, typed, init_by)
        self.init = frozenset(self.f(x) for x in prob.init if x[0] not in self.static)
        self.goals = frozenset(self.f(g) for g in prob.goals)
        self.tils = [(t, self.f(a), pos) for t, a, pos in prob.tils]
        self.actions = self._prune(raw)
        for i, a in enumerate(self.actions):
            a.i = i
        self._index()

    def f(self, atom):
        i = self.fid.get(atom)
        if i is None:
            i = self.fid[atom] = len(self.facts); self.facts.append(atom)
        return i

    def _eval(self, e, b):
        if isinstance(e, str):
            try:
                return float(e)
            except ValueError:
                raise Unsupported("duration term " + e)
        if e[0] in ("+", "-", "*", "/") and len(e) == 3:
            x, y = self._eval(e[1], b), self._eval(e[2], b)
            return {"+": x + y, "-": x - y, "*": x * y, "/": x / y if y else INF}[e[0]]
        if e[0] == "-" and len(e) == 2:
            return -self._eval(e[1], b)
        key = (e[0],) + tuple(b.get(x, x) for x in e[1:])
        return self.prob.fluents.get(key)

    def _ground(self, s, typed, init_by):
        static = self.static
        ptype = dict(s.params)
        allc = [(k, pos, a) for k, ls in s.cond.items() for pos, a in ls]
        st_pos = sorted({a for _, pos, a in allc if pos and a[0] in static},
                        key=lambda a: len(init_by[a[0]]))
        st_neg = [a for _, pos, a in allc if not pos and a[0] in static]
        params = [p for p, _ in s.params]

        def match(atom, b):
            for fct in init_by[atom[0]]:
                if len(fct) != len(atom):
                    continue
                nb = b
                for x, v in zip(atom[1:], fct[1:]):
                    if x.startswith("?"):
                        if x in nb:
                            if nb[x] != v:
                                break
                        else:
                            if v not in typed[ptype[x]]:
                                break
                            nb = dict(nb); nb[x] = v
                    elif x != v:
                        break
                else:
                    yield nb

        def binds(k, b):
            if k == len(st_pos):
                free = [p for p in params if p not in b]
                for combo in itertools.product(*[typed[ptype[p]] for p in free]):
                    nb = dict(b); nb.update(zip(free, combo))
                    yield nb
                return
            for nb in match(st_pos[k], b):
                yield from binds(k + 1, nb)

        sub = lambda a, b: (a[0],) + tuple(b.get(x, x) for x in a[1:])   # noqa: E731
        seen, out = set(), []
        for b in binds(0, {}):
            args = tuple(b[p] for p in params)
            if args in seen:
                continue
            seen.add(args)
            if any(sub(a, b) in self.static_init for a in st_neg):
                continue
            if s.instant:
                lo = hi = 0.0
            else:
                lo, hi = self._eval(s.dur[0], b), self._eval(s.dur[1], b)
                if lo is None or hi is None or hi < lo or hi < 0:
                    continue
                lo = max(lo, 0.0)
            a = Action()
            a.name, a.args, a.lo, a.hi, a.instant = s.name, args, lo, hi, s.instant

            def dyn(ls, want):
                return frozenset(self.f(sub(x, b)) for pos, x in ls if pos == want and x[0] not in static)
            a.cs, a.csn = dyn(s.cond["at start"], True), dyn(s.cond["at start"], False)
            a.co, a.con = dyn(s.cond["over all"], True), dyn(s.cond["over all"], False)
            a.ce, a.cen = dyn(s.cond["at end"], True), dyn(s.cond["at end"], False)
            a.as_, a.ds = dyn(s.eff["at start"], True), dyn(s.eff["at start"], False)
            a.ae, a.de = dyn(s.eff["at end"], True), dyn(s.eff["at end"], False)
            out.append(a)
        return out

    def _prune(self, raw):
        """Relaxed reachability (positive conditions), then backward goal relevance when the
        domain has no negative conditions (then an action adding nothing relevant cannot help)."""
        reach = set(self.init) | {f for _, f, pos in self.tils if pos}
        pre_s = [a.cs | (a.co - a.as_) for a in raw]          # start snap
        pre_e = [a.ce - a.as_ for a in raw]                   # end snap (after the start snap)
        st, en, changed = [False] * len(raw), [False] * len(raw), True
        while changed:
            changed = False
            for k, a in enumerate(raw):
                if not st[k] and pre_s[k] <= reach:
                    st[k] = changed = True
                    reach |= a.as_
                if st[k] and not en[k] and pre_e[k] <= reach:
                    en[k] = changed = True
                    reach |= a.ae
        acts = [a for k, a in enumerate(raw) if en[k]]
        if any(a.csn or a.con or a.cen for a in acts):
            return acts
        need, keep, changed = set(self.goals), [False] * len(acts), True
        while changed:
            changed = False
            for k, a in enumerate(acts):
                if not keep[k] and (a.as_ | a.ae) & need:
                    keep[k] = changed = True
                    need |= a.cs | a.co | a.ce
        return [a for k, a in enumerate(acts) if keep[k]]

    def _index(self):
        # relaxed snap actions: start snap 2i (needs at-start and over-all conditions), end snap
        # 2i+1 (needs the virtual fact "a_i started" = n_facts + i, and its at-end conditions)
        self.consumers = defaultdict(list)
        nf = len(self.facts)
        self.snap_pre = []
        for a in self.actions:
            a.pre = a.cs | (a.co - a.as_)
            self.snap_pre.append(a.pre)
            self.snap_pre.append(frozenset({nf + a.i}) | (a.ce - a.as_))
        for k, pre in enumerate(self.snap_pre):
            for f in pre:
                self.consumers[f].append(k)
        self.nf = nf
        readded = {f for a in self.actions for f in a.as_ | a.ae} | {f for _, f, pos in self.tils if pos}
        self.window = {}
        for t, f, pos in self.tils:
            if not pos and f in self.init and f not in readded:
                self.window[f] = min(self.window.get(f, INF), t)
        self.by_start = defaultdict(list)       # fact -> actions with it in at-start condition
        for a in self.actions:
            for f in a.cs:
                self.by_start[f].append(a.i)
        self.nocond = [a.i for a in self.actions if not a.cs]
        self.free_snaps = [2 * a.i for a in self.actions if not a.pre]


# ---------------------------------------------------------------- search state
class State:
    __slots__ = ("facts", "t", "run", "k", "parent", "move", "g")

    def __init__(self, facts, t, run, k, parent=None, move=None):
        self.facts, self.t, self.run, self.k, self.parent, self.move = facts, t, run, k, parent, move
        self.g = 0 if parent is None else parent.g + 1

    def key(self):
        return (self.facts, tuple((i, round(lo - self.t, 4), round(hi - self.t, 4)) for i, _, lo, hi in self.run),
                self.k)


def _r(x):
    return round(x, 6)


class Engine:
    def __init__(self, T: Task):
        self.T = T

    # ---- moves
    def protected(self, run, skip=None):
        pos, neg = set(), set()
        for j, (i, _, _, _) in enumerate(run):
            if j != skip:
                a = self.T.actions[i]
                pos |= a.co; neg |= a.con
        return pos, neg

    def starts(self, S):
        T, F = self.T, S.facts
        ppos, pneg = self.protected(S.run)
        cand = set(T.nocond)
        for f in F:
            cand.update(T.by_start.get(f, ()))
        out = []
        for i in sorted(cand):
            a = T.actions[i]
            if not a.cs <= F or a.csn & F or a.ds & ppos or a.as_ & pneg:
                continue
            F2 = (F - a.ds) | a.as_
            if not a.co <= F2 or a.con & F2:
                continue
            if a.instant:
                out.append(("start", i, None))
                continue
            out.append(("start", i, None))
        return out

    def apply_start(self, S, i):
        a = self.T.actions[i]
        F = (S.facts - a.ds) | a.as_
        run = S.run
        if not a.instant:
            run = tuple(sorted(run + ((i, S.t, _r(S.t + a.lo), _r(S.t + a.hi)),), key=lambda r: (r[3], r[0], r[1])))
        return State(F, _r(S.t + SEP), run, S.k, S, ("start", i, S.t))

    def ends(self, S):
        """Flexible actions that may end now (fixed ones end automatically on WAIT)."""
        out = []
        for j, (i, st, lo, hi) in enumerate(S.run):
            if lo < hi and lo <= S.t + 1e-9 and self._can_end(S.facts, S.run, j):
                out.append(("end", j, None))
        return out

    def _can_end(self, F, run, j):
        a = self.T.actions[run[j][0]]
        if not a.ce <= F or a.cen & F:
            return False
        ppos, pneg = self.protected(run, skip=j)
        return not (a.de & ppos or a.ae & pneg)

    def apply_end(self, S, j, t=None):
        i, st, lo, hi = S.run[j]
        a = self.T.actions[i]
        F = (S.facts - a.de) | a.ae
        t = S.t if t is None else t
        return State(F, _r(t + SEP), S.run[:j] + S.run[j + 1:], S.k, S, ("end", i, st, t))

    def wait(self, S):
        """-> state after advancing to the next event (timed literals, fixed-duration ends and
        flexible actions reaching their latest end are applied there), or None if nothing to wait
        for or the wait is dead (a protected fact is deleted, an end condition fails)."""
        T = self.T
        cands = [lo for _, _, lo, hi in S.run if lo > S.t + 1e-9] + [hi for _, _, _, hi in S.run]
        if S.k < len(T.tils):
            cands.append(T.tils[S.k][0])
        if not cands:
            return None
        tn = max(min(cands), S.t)
        F, k, run = S.facts, S.k, S.run
        happened = False
        while k < len(T.tils) and T.tils[k][0] <= tn + 1e-9:
            _, f, pos = T.tils[k]
            ppos, pneg = self.protected(run)
            if (not pos and f in ppos) or (pos and f in pneg):
                return None
            F = (F | {f}) if pos else (F - {f})
            k += 1; happened = True
        node = State(frozenset(F), S.t, run, k, S, ("wait", tn))
        while True:
            due = [j for j, r in enumerate(node.run) if r[3] <= tn + 1e-9]
            if not due:
                break
            j = due[0]
            if not self._can_end(node.facts, node.run, j):
                return None
            i, st, lo, hi = node.run[j]
            a = T.actions[i]
            node = State((node.facts - a.de) | a.ae, node.t, node.run[:j] + node.run[j + 1:], k, node,
                         ("end", i, st, hi))
            happened = True
        node.t = _r(tn + SEP) if happened else _r(tn)
        return node

    def successors(self, S):
        out = [("end", j, self.apply_end(S, j)) for _, j, _ in self.ends(S)]
        out += [("start", i, self.apply_start(S, i)) for _, i, _ in self.starts(S)]
        w = self.wait(S)
        if w is not None:
            out.append(("wait", None, w))
        return out

    def is_goal(self, S):
        return self.T.goals <= S.facts and not S.run

    # ---- heuristics: relaxed planning graphs from S
    def graph(self, S, unit):
        """Relaxed planning graph over snap actions from S (deletes ignored).
        unit=True : cost = number of actions (FF levels); running actions' end effects and timed
                    literals are free.
        unit=False: cost = time; an end snap comes >= duration after its start snap; end effects
                    of running actions arrive at their earliest end, timed literals at their time;
                    a fact deleted by a timed literal and never re-added can only be used before
                    then (deadlines).
        -> (cost of each fact, achiever of each fact, cost of each start snap)"""
        T, t0, nf = self.T, (0.0 if unit else S.t), self.T.nf
        cost, ach, heap = {}, {}, []

        def push(f, c, why):
            if c < cost.get(f, INF):
                cost[f] = c; ach[f] = why; heapq.heappush(heap, (c, f))
        for f in S.facts:
            push(f, t0, None)
        for j, (i, st, lo, hi) in enumerate(S.run):
            for f in T.actions[i].ae:
                push(f, t0 if unit else max(lo, t0), ("run", j))
        for tl, f, pos in T.tils[S.k:]:
            if pos:
                push(f, t0 if unit else tl, ("til",))
        acts, pres, cnt, start = T.actions, T.snap_pre, {}, {}
        win = None if unit else T.window

        def fire(k):
            i, end = k >> 1, k & 1
            a = acts[i]
            s = t0
            for f in pres[k]:
                c = cost[f]
                if c > s:
                    s = c
            if end:
                if win:
                    for f in (a.co | a.ce) - a.as_:
                        if f in win and s > win[f] - 1e-9:
                            return
                for f in a.ae:
                    push(f, s, ("e", i))
                return
            if win:
                for f in a.cs:
                    if f in win and s >= win[f] - 1e-9:
                        return
            if s < start.get(i, INF):
                start[i] = s
                c1 = s + 1 if unit else s
                for f in a.as_:
                    push(f, c1, ("s", i))
                if not a.instant:
                    push(nf + i, s + 1 if unit else s + a.lo, ("v", i))

        for k in T.free_snaps:
            fire(k)
        done = set()
        cons = T.consumers
        while heap:
            c, f = heapq.heappop(heap)
            if c > cost.get(f, INF) or f in done:
                continue
            done.add(f)
            for k in cons.get(f, ()):
                m = cnt.get(k, 0) + 1
                cnt[k] = m
                if m == len(pres[k]):
                    fire(k)
        return cost, ach, start

    def h(self, S):
        """-> (h_ff, makespan lower bound, helpful action ids, running actions the relaxed plan
        waits for) or None if S is a dead end (a goal, or the end condition of a running action,
        cannot be reached in time)."""
        T = self.T
        goals = T.goals
        tc, _, _ = self.graph(S, unit=False)
        lb = S.t
        for g in goals:
            c = tc.get(g, INF)
            if c == INF:
                return None
            lb = max(lb, c)
        for i, st, lo, hi in S.run:
            a = T.actions[i]
            for f in a.ce:
                if tc.get(f, INF) > hi + 1e-9:
                    return None
        cost, ach, start = self.graph(S, unit=True)
        plan, seen, runs = set(), set(), set()
        stack = [g for g in goals if g not in S.facts]
        for i, st, lo, hi in S.run:                 # running actions must be able to end
            stack += [f for f in T.actions[i].ce if f not in S.facts]
        acts = T.actions
        while stack:
            f = stack.pop()
            if f in seen:
                continue
            seen.add(f)
            src = ach.get(f)
            if src is None or src[0] == "til":
                continue
            if src[0] == "run":
                runs.add(src[1]); continue
            i = src[1]
            if i not in plan:
                plan.add(i)
                stack.extend(acts[i].pre)
                if src[0] == "e":
                    stack.extend(acts[i].ce - acts[i].as_)
        helpful = {i for i in plan if start.get(i, INF) <= 0}
        return len(plan) + 0.5 * len(S.run), lb, helpful, runs


# ---------------------------------------------------------------- planner
class Planner:
    def __init__(self, T: Task, budget=60.0, policy=None, mode="ehc"):
        self.T, self.E, self.budget, self.policy, self.mode = T, Engine(T), budget, policy, mode
        self.expanded = self.evaluated = 0

    def _deadline(self):
        return time.time() > self.t_end

    def solve(self):
        self.t_end = time.time() + self.budget
        S0 = State(self.T.init, 0.0, (), 0)
        # timed literals at time 0
        if self.T.tils and self.T.tils[0][0] <= 0:
            w = self.E.wait(S0)
            if w is not None:
                S0 = w
        if self.mode == "gbfs":
            return self.gbfs(S0)
        r = self.ehc(S0)
        if r is None and not self._deadline():
            r = self.gbfs(S0)
        return r

    def _eval(self, S):
        self.evaluated += 1
        return self.E.h(S)

    def _ordered(self, S, hv):
        """Successors, helpful ones first; ends and waits count as helpful when a running action
        must finish (it supplies a relaxed-plan fact, or nothing else is helpful)."""
        _, _, helpful, runs = hv
        succ = self.E.successors(S)
        pref, rest = [], []
        for kind, x, n in succ:
            if kind == "start" and x in helpful:
                pref.append(n)
            elif kind == "end":
                pref.append(n)
            elif kind == "wait" and (runs or not helpful):
                pref.append(n)
            else:
                rest.append(n)
        if self.policy is not None:
            pref = self.policy.rank(self, S, pref)
        return pref, rest

    def ehc(self, S):
        hv = self._eval(S)
        if hv is None:
            return None
        best = (hv[0], hv[1])
        while not self.E.is_goal(S):
            if self._deadline():
                return None
            # breadth-first from S on helpful moves until strictly better h
            q = deque([(S, hv)])
            seen = {S.key()}
            found = None
            while q and found is None:
                if self._deadline():
                    return None
                X, xh = q.popleft()
                self.expanded += 1
                pref, _ = self._ordered(X, xh)
                for n in pref:
                    k = n.key()
                    if k in seen:
                        continue
                    seen.add(k)
                    if self.E.is_goal(n):
                        return n
                    nh = self._eval(n)
                    if nh is None:
                        continue
                    if (nh[0], nh[1]) < best:
                        found = (n, nh); break
                    q.append((n, nh))
                if len(seen) > 20000:
                    return None
            if found is None:
                return None
            S, hv = found
            best = (hv[0], hv[1])
        return S

    def gbfs(self, S0):
        """Lazy greedy best-first search with two open lists (all moves / helpful moves),
        alternated with a boost for the helpful list after progress (as in Fast Downward).
        Children are queued with the parent's h and evaluated when popped."""
        hv0 = self._eval(S0)
        if hv0 is None:
            return None
        cnt = itertools.count()
        opens = ([], [])                      # 0: all, 1: helpful
        heapq.heappush(opens[1], (hv0[0], hv0[1], next(cnt), S0, None))
        best_t = {S0.key(): S0.t}
        best_h, boost, turn = INF, 0, 0
        while opens[0] or opens[1]:
            if self._deadline():
                return None
            if boost > 0 and opens[1]:
                q = 1; boost -= 1
            else:
                turn ^= 1
                q = turn if opens[turn] else 1 - turn
            ph, plb, _, S, _ = heapq.heappop(opens[q])
            if best_t.get(S.key(), INF) < S.t:
                continue
            hv = self._eval(S)
            if hv is None:
                continue
            if self.E.is_goal(S):
                return S
            if hv[0] < best_h:
                best_h, boost = hv[0], boost + 1000
            self.expanded += 1
            pref, rest = self._ordered(S, hv)
            for group, helpful in ((pref, True), (rest, False)):
                for n in group:
                    k = n.key()
                    if best_t.get(k, INF) <= n.t:
                        continue
                    best_t[k] = n.t
                    if self.E.is_goal(n):
                        return n
                    entry = (hv[0], hv[1], next(cnt), n, None)
                    heapq.heappush(opens[0], entry)
                    if helpful:
                        heapq.heappush(opens[1], entry)
        return None


def extract_plan(T: Task, S):
    """Follow parents; pair starts with ends -> [(start, Action, duration)]."""
    moves = []
    while S is not None:
        if S.move is not None:
            moves.append(S.move)
        S = S.parent
    moves.reverse()
    open_, out = {}, []
    for m in moves:
        if m[0] == "start":
            _, i, t = m
            a = T.actions[i]
            if a.instant:
                out.append([t, a, 0.0])
            else:
                rec = [t, a, None]; out.append(rec)
                open_.setdefault((i, t), rec)
        elif m[0] == "end":
            _, i, st, t = m
            rec = open_.pop((i, st))
            rec[2] = _r(t - st)
    for rec in out:
        if rec[2] is None:
            rec[2] = rec[1].lo
    return sorted(out, key=lambda r: r[0])


def plan_text(steps):
    return "".join("{:.3f}: {} [{:.3f}]\n".format(t, a, d) for t, a, d in steps)


def plan(domain_file, problem_file, budget=60.0, policy=None, mode="ehc"):
    t0 = time.time()
    T = Task(Domain(open(domain_file).read()), Problem(open(problem_file).read()))
    tg = time.time() - t0
    P = Planner(T, budget=max(1.0, budget - tg), policy=policy, mode=mode)
    S = P.solve()
    info = {"ground_s": round(tg, 3), "search_s": round(time.time() - t0 - tg, 3), "actions": len(T.actions),
            "facts": len(T.facts), "expanded": P.expanded, "evaluated": P.evaluated, "solved": S is not None}
    if S is None:
        return None, info
    steps = extract_plan(T, S)
    info["makespan"] = round(max((t + d for t, _, d in steps), default=0.0), 3)
    info["length"] = len(steps)
    return plan_text(steps), info


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("solve")
    s.add_argument("domain"); s.add_argument("problem")
    s.add_argument("--budget", type=float, default=60.0)
    s.add_argument("--out")
    s.add_argument("--val", action="store_true")
    s.add_argument("--mode", default="ehc", choices=("ehc", "gbfs"))
    a = ap.parse_args()
    txt, info = plan(a.domain, a.problem, a.budget, mode=a.mode)
    print(info)
    if txt:
        if a.out:
            open(a.out, "w").write(txt)
        if a.val:
            from .pddlkit import validate_plan
            v = validate_plan(a.domain, a.problem, txt)
            print("VAL", v.valid, v.value, v.errors[:5])
        if not a.out:
            print(txt)


if __name__ == "__main__":
    main()


# ==================== the evaluation interface (build / solve -> RunResult), as V1-V9 ==========================
class TextPlanner:
    """Runs a planner on the task's PDDL files and checks its plan with VAL on the full task. solve_files(dom, prob,
    budget) -> (plan text or None, info)."""

    def __init__(self, name, solve_files):
        self.name, self.solve_files = name, solve_files

    def solve(self, task, budget_s, mode="deploy", gt=None, graph=None):
        import tempfile
        from .pddlkit import TemporalPlan, validate_plan
        from .v1 import RunResult
        t0 = time.time()
        with tempfile.TemporaryDirectory(prefix="tiflax_text_") as d:
            dom, prob = task.write(d, "full")
            txt, info = self.solve_files(dom, prob, budget_s)
            plan = TemporalPlan.parse(txt) if txt else None
            ok = plan is not None and validate_plan(dom, prob, plan).valid
        el = time.time() - t0
        return RunResult(method=self.name, instance=task.name, budget_s=budget_s, solved=ok,
                         makespan=plan.makespan if ok else None, first_makespan=plan.makespan if ok else None,
                         first_valid_s=info.get("first_s", el) if ok else None, total_s=el, n_total=len(task.objects),
                         branch="search" if ok else None, invalid_on_full=int(plan is not None and not ok),
                         notes=[] if ok else ["no valid plan"], plan_text=plan.to_text() if ok else None,
                         ledger={k: v for k, v in info.items() if isinstance(v, (int, float))})


def build(key, spec=None, seed=0, mode="ehc", **over):
    """V10: no model, no domain code."""
    return TextPlanner(key, lambda dom, prob, b: plan(dom, prob, b, mode=mode))
