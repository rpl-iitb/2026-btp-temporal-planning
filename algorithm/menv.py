"""Multi-agent temporal environment with stochastic durations, for any PDDL 2.1 domain (built on v10's semantics).

    env = MAEnv(task, agent_type="cook", noise=Noise({"move": 0.25, "chop": 0.2}, default=0.1), seed=0)
    result = env.rollout(controller)            # controller(env, S, agent, legal) -> index into legal or None (wait)

Execution model
  * The planner/controller sees nominal durations (the PDDL domain). When an action with a fixed duration
    starts, its actual duration is drawn from a lognormal with that nominal mean and the action's coefficient
    of variation (Noise). The end time is hidden until the end event happens. Flexible-duration actions
    (duration inequalities, e.g. a stove timer) keep their declared [min, max] window.
  * Actions without an agent argument (e.g. a pot cooking, a diner eating) start as soon as they are
    applicable; flexible actions end as soon as their end conditions allow (both are not decisions).
  * Decision rounds: at every event time each agent with at least one applicable action, in a fixed order,
    chooses one of its applicable START actions or waits. Then time advances to the next event.
  * A rollout fails if a forced end cannot happen (e.g. a soup would burn), if nothing can happen any more
    (stuck, e.g. a deadline passed), or at the horizon.
"""
import math

import numpy as np

from . import v10

SEP = v10.SEP


class Noise:
    """Lognormal multiplicative noise with mean 1 per action schema: d = nominal * X, E[X] = 1, CV[X] = cv."""

    def __init__(self, cv=None, default=0.0, clip=(0.4, 3.0)):
        self.cv, self.default, self.clip = dict(cv or {}), default, clip

    def scaled(self, k):
        return Noise({a: c * k for a, c in self.cv.items()}, self.default * k, self.clip)

    def sample(self, name, nominal, rng):
        cv = self.cv.get(name, self.default)
        if cv <= 0 or nominal <= 0:
            return nominal
        s2 = math.log(1 + cv * cv)
        x = rng.lognormal(-s2 / 2, math.sqrt(s2))
        return round(nominal * float(np.clip(x, *self.clip)), 3)


class SEngine(v10.Engine):
    """v10 engine whose fixed-duration actions take sampled durations."""

    def __init__(self, T, noise=None, rng=None):
        super().__init__(T)
        self.noise, self.rng = noise, rng

    def apply_start(self, S, i):
        a = self.T.actions[i]
        F = (S.facts - a.ds) | a.as_
        run, d = S.run, a.lo
        if not a.instant:
            if a.lo == a.hi and self.noise is not None:
                d = self.noise.sample(a.name, a.lo, self.rng)
                lo = hi = v10._r(S.t + d)
            else:
                lo, hi = v10._r(S.t + a.lo), v10._r(S.t + a.hi)
            run = tuple(sorted(run + ((i, S.t, lo, hi),), key=lambda r: (r[3], r[0], r[1])))
        return v10.State(F, v10._r(S.t + SEP), run, S.k, S, ("start", i, S.t, d))


class MAEnv:
    def __init__(self, T: v10.Task, agent_type, noise=None, seed=0, horizon=None):
        self.T = T
        self.rng = np.random.default_rng(seed)
        self.E = SEngine(T, noise, self.rng)
        dom, prob = T.dom, T.prob
        objs = dict(dom.constants); objs.update(prob.objects)
        is_agent_type = lambda ty: agent_type in set(dom.ancestors(ty))     # noqa: E731
        self.agents = sorted(o for o, ty in objs.items() if is_agent_type(ty))
        ptypes = {s.name: [ty for _, ty in s.params] for s in dom.schemas}
        self.agent_of = []
        for a in T.actions:
            ag = None
            for x, ty in zip(a.args, ptypes[a.name]):
                if is_agent_type(ty):
                    ag = x
                    break
            self.agent_of.append(ag)
        self.horizon = horizon
        self.cache_legal, self._legal = True, None

    # ---- helpers
    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.E.rng = self.rng
        S = v10.State(self.T.init, 0.0, (), 0)
        if self.T.tils and self.T.tils[0][0] <= 0:
            w = self.E.wait(S)
            S = w if w is not None else S
        return S

    def auto(self, S):
        """Start agent-less applicable actions and end flexible actions that may end, until nothing changes."""
        for _ in range(1000):
            ch = False
            for _, j, _ in self.E.ends(S):
                S = self.E.apply_end(S, j); ch = True
                break
            if ch:
                continue
            for _, i, _ in self.starts(S):
                if self.agent_of[i] is None:
                    S = self.E.apply_start(S, i); ch = True
                    break
            if not ch:
                return S
        return S

    def starts(self, S):
        """E.starts(S), computed once per state (auto() and every agent of a decision round that waits ask about the
        same state); states are immutable, so the cache is exact."""
        if not self.cache_legal:
            return self.E.starts(S)
        if self._legal is None or self._legal[0] is not S:
            self._legal = (S, self.E.starts(S))
        return self._legal[1]

    def legal(self, S, agent):
        return [i for _, i, _ in self.starts(S) if self.agent_of[i] == agent]

    def goal(self, S):
        return self.T.goals <= S.facts

    # ---- rollout
    def nominal_snapshot(self, S):
        """S as a planner sees it: no history, running fixed-duration actions end at their nominal time."""
        run = []
        for i, st, lo, hi in S.run:
            a = self.T.actions[i]
            if a.lo == a.hi:
                e = max(v10._r(st + a.lo), v10._r(S.t + SEP))
                run.append((i, st, e, e))
            else:
                run.append((i, st, lo, hi))
        return v10.State(S.facts, S.t, tuple(sorted(run, key=lambda r: (r[3], r[0], r[1]))), S.k)

    max_rounds = 20000      # decision rounds per rollout (a domain with long episodes may raise it)

    def rollout(self, controller, record=None, seed=None, max_rounds=None, start=None):
        """-> dict(success, makespan, served, reason, plan steps). record(S, agent, legal, choice) is called per decision.
        start: begin from this state instead of the initial state."""
        max_rounds = self.max_rounds if max_rounds is None else max_rounds
        S = self.reset(seed) if start is None else start
        for _ in range(max_rounds):
            S = self.auto(S)
            if self.goal(S):
                while S.run:                       # let running actions (e.g. diners) finish
                    w = self.E.wait(S)
                    if w is None:
                        break
                    S = self.auto(w)
                return self._result(S, True, "ok")
            if self.horizon and S.t > self.horizon:
                return self._result(S, False, "horizon")
            started = False
            for ag in self.agents:
                lg = self.legal(S, ag)
                if not lg:
                    continue
                k = controller(self, S, ag, lg)
                if record is not None:
                    record(S, ag, lg, k)
                if k is not None:
                    S = self.E.apply_start(S, lg[k])
                    started = True
            if started:
                continue                           # new state at t + eps: let other agents react
            w = self.E.wait(S)
            wake = controller.wakeup(S) if hasattr(controller, "wakeup") else None
            if wake is not None and wake > S.t + 1e-9 and (w is None or wake < self._next_event(S) - 1e-9):
                S = v10.State(S.facts, v10._r(wake), S.run, S.k, S, None)     # time-triggered controllers
                continue
            if w is None:
                return self._result(S, False, "dead" if S.run else "stuck")
            S = w
        return self._result(S, False, "rounds")

    def _next_event(self, S):
        c = [lo for _, _, lo, hi in S.run if lo > S.t + 1e-9] + [hi for _, _, _, hi in S.run]
        if S.k < len(self.T.tils):
            c.append(self.T.tils[S.k][0])
        return min(c) if c else float("inf")

    def _result(self, S, ok, why):
        steps = extract(self.T, S)
        served = sum(1 for g in self.T.goals if g in S.facts)
        return {"success": ok, "reason": why, "served": served, "goals": len(self.T.goals),
                "makespan": round(max((t + d for t, _, d in steps), default=0.0), 3) if ok else None,
                "t_end": round(S.t, 3), "steps": steps}


def extract(T, S):
    """Parent chain -> [(start, Action, actual duration)]."""
    moves = []
    while S is not None:
        if S.move is not None:
            moves.append(S.move)
        S = S.parent
    moves.reverse()
    out, open_ = [], {}
    for m in moves:
        if m[0] == "start":
            i, t, d = m[1], m[2], m[3] if len(m) > 3 else T.actions[m[1]].lo
            a = T.actions[i]
            rec = [t, a, d if a.lo == a.hi else None]
            out.append(rec)
            if a.lo != a.hi:
                open_[(i, t)] = rec
        elif m[0] == "end":
            _, i, st, t = m
            rec = open_.pop((i, st), None)
            if rec is not None:
                rec[2] = v10._r(t - st)
    for rec in out:
        if rec[2] is None:
            rec[2] = rec[1].lo
    return sorted(out, key=lambda r: r[0])


def plan_text(steps):
    return "".join("{:.3f}: {} [{:.3f}]\n".format(t, a, d) for t, a, d in steps)


class PlanFollower:
    """Open-loop execution of a fixed (nominal) plan under random durations: every agent performs its own
    actions of the plan in their planned order, each as soon as it is applicable (a partial-order dispatch;
    agent-less actions are started by the environment). Baseline for classical planners' output."""

    def __init__(self, env, plan_text, timed=True):
        T = env.T
        idx = {(a.name,) + a.args: a.i for a in T.actions}
        self.queue = {ag: [] for ag in env.agents}
        for line in plan_text.splitlines():
            if "(" not in line:
                continue
            name = tuple(line[line.index("(") + 1:line.index(")")].split())
            i = idx.get(name)
            if i is not None and env.agent_of[i] is not None:
                self.queue[env.agent_of[i]].append((float(line.split(":")[0]), i))
        self.pos = {ag: 0 for ag in env.agents}
        self.timed = timed

    def __call__(self, env, S, agent, legal):
        q, p = self.queue[agent], self.pos[agent]
        if p < len(q):
            t, i = q[p]
            if i in legal and (not self.timed or S.t >= t - 1e-6):
                self.pos[agent] = p + 1
                return legal.index(i)
        return None

    def wakeup(self, S):
        """Earliest planned start still ahead (time-triggered execution)."""
        ts = [self.queue[a][p][0] for a, p in self.pos.items() if p < len(self.queue[a]) and self.queue[a][p][0] > S.t + 1e-9]
        return min(ts) if (ts and self.timed) else None


class Replanner:
    """Plan -> execute time-triggered -> re-plan from the actual state when a robot is late by more than `tol`
    (its next planned action cannot start within tol of its planned time) or has run out of planned actions
    while goals remain. planner(S) -> [(start, Action, duration)] from state S (nominal durations)."""

    def __init__(self, env, planner, tol=3.0, max_replans=200):
        self.planner, self.tol, self.max_replans = planner, tol, max_replans
        self.queue, self.pos, self.replans, self.plan_t, self.last_check = {}, {}, 0, None, None
        self.agents = env.agents
        self.plan_time = 0.0

    def _plan(self, env, S):
        import time as _t
        t0 = _t.time()
        steps = self.planner(S)
        self.plan_time += _t.time() - t0
        self.queue = {ag: [] for ag in self.agents}
        for t, a, d in steps:
            ag = env.agent_of[a.i]
            if ag is not None:
                self.queue[ag].append((t, a.i))
        self.pos = {ag: 0 for ag in self.agents}
        self.replans += 1
        self.plan_t = S.t

    def _late(self, env, S):
        if self.replans >= self.max_replans:
            return False
        for ag in self.agents:
            q, p = self.queue.get(ag, []), self.pos.get(ag, 0)
            if p < len(q) and S.t > q[p][0] + self.tol:
                return True
        return False

    def __call__(self, env, S, agent, legal):
        if self.plan_t is None:
            self._plan(env, S)
        elif self.last_check != S.t and self._late(env, S):
            self._plan(env, S)
        elif self.pos.get(agent, 0) >= len(self.queue.get(agent, [])) and not env.goal(S) \
                and self.last_check != S.t and S.t > (self.plan_t or 0) + self.tol and self.replans < self.max_replans:
            self._plan(env, S)
        self.last_check = S.t
        q, p = self.queue[agent], self.pos[agent]
        if p < len(q):
            t, i = q[p]
            if i in legal and S.t >= t - 1e-6:
                self.pos[agent] = p + 1
                return legal.index(i)
        return None

    def wakeup(self, S):
        ts = [self.queue[a][p][0] for a, p in self.pos.items() if p < len(self.queue[a]) and self.queue[a][p][0] > S.t + 1e-9]
        late = [self.queue[a][p][0] + self.tol + 0.01 for a, p in self.pos.items()
                if p < len(self.queue[a]) and self.queue[a][p][0] + self.tol + 0.01 > S.t + 1e-9]
        ts += late
        return min(ts) if ts else None


def sim_planner(env, policy_factory):
    """A planner that simulates a closed-loop policy with nominal durations from the current state (constructive
    symbolic planning); its trajectory is the plan."""
    T = env.T

    def plan(S):
        sim = MAEnv(T, "cook", noise=None)
        sim.agents, sim.agent_of = env.agents, env.agent_of
        res = sim.rollout(policy_factory(sim), start=env.nominal_snapshot(S))
        return res["steps"]
    return plan
