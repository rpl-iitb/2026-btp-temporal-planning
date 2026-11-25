"""Baseline policies: uniform random over legal tasks (WAIT only when nothing else is legal), a rule-based multi-cook
dispatcher, and the POPF planner (popf: plan once; popf-replan: re-plan when execution drifts; see pddl.py).

    from overcooked_kitchen.baselines import make
    Sim("m01_three_cooks").run(make("dispatcher"))

The dispatcher is a pure function of (sim.world, sim.running, sim.t): it first applies the pending effects of the
other cooks' running tasks to a copy of the world (their claims), then picks by priority (ported from earlier
research code):
  1. carry what you hold where it is needed: chopped veg -> a pot that needs it (else park it); raw veg -> a board;
     a plated soup -> the serving window; a dirty plate -> a sink; a clean plate -> its soup (dish up when ready),
     else the rack
  2. empty-handed: claim a soup ready soon (fetch a clean plate, or wash one); light a full recipe pot when a plate
     can reach it before it burns; chop; take a needed chopped vegetable; scrub a dirty pot when no clean pot is
     free; fetch the most needed vegetable at the nearest crate; wash plates when clean plates are short;
     scrub anyway; else wait.
A task that would keep the cook away past the point where an unclaimed soup could still be plated is replaced by
the claim (burn safety). An idle cook standing at a station another cook holds something for takes a harmless task
elsewhere (fetch a dirty or washed plate, a needed vegetable, move a spare chopped one off a board).
Times are nominal (task durations, cook times); only observed state is read.
"""
"""This file is UN-VERIFIED"""
import copy
from collections import Counter

import numpy as np

from overcooked_kitchen.layout import INGREDIENTS, PROVIDES
from overcooked_kitchen.tasks import WAIT, stations_of
from overcooked_kitchen.world import cook_time

CLAIM_SLACK = 4
BURN_MARGIN = 3
ALLOWANCE = 1.25
INF = float("inf")


def make(name: str, seed: int = 0, **kwargs):
    """Policy callable policy(sim, cook, legal) -> an item of legal.

    popf / popf-replan take PlannerPolicy keywords (planner, timeout, mode, tol, max_replans, hard_deadlines)."""
    if name == "random":
        return RandomPolicy(seed)
    if name == "dispatcher":
        return Dispatcher()
    if name in ("popf", "popf-replan"):
        from overcooked_kitchen.pddl import PlannerPolicy
        return PlannerPolicy(replan=name == "popf-replan", **kwargs)
    raise ValueError("unknown baseline '{}'; available: random, dispatcher, popf, popf-replan".format(name))


class RandomPolicy:
    """Uniform over the legal tasks; WAIT only when it is the only legal item."""

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def __call__(self, sim, cook, legal):
        tasks = [t for t in legal if t != WAIT]
        if not tasks:
            return WAIT
        return tasks[int(self.rng.integers(len(tasks)))]


def _dist(lay, pose, cell):
    """Steps from pose to the nearest pose facing cell (other cooks ignored)."""
    d = lay.bfs_poses(pose)
    return min((d[p] for p in lay.access_poses(cell) if p in d), default=INF)


class View:
    """The kitchen after the other cooks' running tasks: projected world, cook poses, soups and plate holders."""

    def __init__(self, sim, allowance=1.0):
        w = sim.world
        memo = {id(w.lay): w.lay, id(w.events): [], id(w.burnt): [], id(w.spoiled): []}
        cache = w.__dict__.get("_stations_cache")
        if cache is not None:
            memo[id(cache)] = cache
        P = copy.deepcopy(w, memo)
        self.w, self.t, self.counters = P, sim.t, sim.counters
        self.pose = [(p.pos[0], p.pos[1], tuple(p.facing)) for p in P.players]
        for c in sim.cooks:
            r = sim.running.get(c)
            if r is None:
                continue
            i, task = sim.cooks.index(c), r.task
            self.pose[i] = task.pose
            if r.applied or task.cell is None or P.interact_event(i, task.pose) != task.event:
                continue
            ta = max(sim.t, r.start + task.duration - 1)
            P.apply(i, task.event, task.cell, ta, task.strokes)
        burn = P.params.get("burn")
        self.burn = INF if burn is None else burn
        self.soups = []
        for c, pot in P.pots.items():
            if pot.state == "ready":
                self.soups.append((pot.ready_at, c))
            elif pot.state == "cooking" and pot.recipe:
                self.soups.append((max(sim.t, pot.cook_start + cook_time(P.params, pot.comp())), c))
        self.soups.sort()
        self.holders = [i for i in range(len(P.players)) if self.plate(i) == "clean"]
        self.assign, self.uncovered = {}, []
        free = list(self.holders)
        for ready, c in self.soups:
            near = [(_dist(P.lay, self.pose[h], c), h) for h in free]
            near = [x for x in near if x[0] * allowance < self.burn]
            if near:
                h = min(near)[1]
                self.assign[h] = c
                free.remove(h)
            else:
                self.uncovered.append((ready, c))
        self.spare = free

    def held(self, i):
        return self.w.players[i].held

    def plate(self, i):
        h = self.held(i)
        return self.w.plates[h[1]].state if h is not None and h[0] == "plate" else None


class Dispatcher:
    """Rule-based multi-cook dispatcher on tasks (see the module docstring)."""

    def __init__(self, claim_slack=CLAIM_SLACK, burn_margin=BURN_MARGIN, allowance=ALLOWANCE):
        self.slack, self.margin, self.allowance = claim_slack, burn_margin, allowance

    def lead(self, V, pose, sources, pot):
        """Steps from pose to fetch a clean plate at one of sources and dish up at pot."""
        lay = V.w.lay
        d = lay.bfs_poses(pose)
        best = INF
        for s in sources:
            for q in lay.access_poses(s):
                if q in d:
                    best = min(best, d[q] + 1 + _dist(lay, q, pot) + 1)
        return best

    # ---- kitchen bookkeeping (on the projected world) ---------------------------------------
    def sources(self, V):
        """Cells holding a clean plate to take: racks and sinks."""
        w = V.w
        out = [c for c, s in w.stack.items() if w.kind[c] == "rack" and s]
        out += [c for c, pid in w.sink.items() if pid is not None and w.plates[pid].state == "clean"]
        return sorted(out)

    def safe_pot(self, V, pot):
        """A plate fetched from a rack can reach the pot before its soup burns."""
        racks = [c for c, k in V.w.kind.items() if k == "rack"]
        if V.burn == INF or not racks:
            return True
        lay = V.w.lay
        return any(_dist(lay, p, pot) < V.burn for r in racks for p in lay.access_poses(r))

    def plan(self, V):
        """pot -> target recipe for filling pots and for empty pots taking new dishes; open needs (ing, pot)."""
        w = V.w
        busy = Counter()
        for _, c in V.soups:
            busy[w.pots[c].comp()] += 1
        for pl in w.plates.values():
            if pl.state == "soup":
                busy[tuple(pl.soup)] += 1
        orders = sorted(w.pending_orders(), key=lambda o: (o.get("deadline") is None, o.get("deadline") or 0,
                                                            int(o["id"][1:])))
        free = []
        for o in orders:
            comp = tuple(o["comp"])
            if busy[comp] > 0:
                busy[comp] -= 1
            else:
                free.append(comp)
        target = {}
        idle = sorted((c for c, p in w.pots.items() if p.state == "idle"), key=lambda c: (c[1], c[0]))
        filling = sorted((c for c in idle if w.pots[c].contents), key=lambda c: (-len(w.pots[c].contents), c))
        for c in filling:
            have = w.pots[c].comp()
            cands = [r for r in free if all(h <= n for h, n in zip(have, r))]
            if cands:
                r = min(cands, key=lambda r: (sum(r) - sum(have), free.index(r)))
                target[c] = r
                free.remove(r)
        empty = [c for c in idle if not w.pots[c].contents and self.safe_pot(V, c)]
        for c, r in zip(empty, free):
            target[c] = r
        need = []
        for c, r in sorted(target.items()):
            have = w.pots[c].comp()
            for k, ing in enumerate(INGREDIENTS):
                need += [(ing, c)] * max(0, r[k] - have[k])
        supply = Counter()
        for p in w.players:
            if p.held is not None and p.held[0] in ("raw", "chopped"):
                supply[p.held[1]] += 1
        for item in w.surface.values():
            if item is not None and item[0] in ("raw", "chopped"):
                supply[item[1]] += 1
        open_need = []
        for ing, c in need:
            if supply[ing] > 0:
                supply[ing] -= 1
            else:
                open_need.append((ing, c))
        unplaced = free[len(empty):]
        later = Counter()
        for r in unplaced:
            later.update({ing: n for ing, n in zip(INGREDIENTS, r) if n})
        later -= supply
        return target, empty, unplaced, need, open_need, later

    # ---- the rules ------------------------------------------------------------------------------
    def __call__(self, sim, cook, legal):
        V = View(sim, self.allowance)
        return self.decide(V, sim.cooks.index(cook), legal)

    @staticmethod
    def pick(legal, pred):
        best = None
        for t in legal:
            if t != WAIT and pred(t) and (best is None or t.travel < best.travel):
                best = t
        return best

    def decide(self, V, i, legal):
        held = V.held(i)
        target, empty, unplaced, need, open_need, later = self.plan(V)
        if held is not None:
            return self.carry(V, i, legal, held, need) or WAIT
        claim = self.claim(V, i, legal)
        if claim is not None and claim[1]:
            return claim[0]
        options = self.work(V, i, legal, target, empty, unplaced, need, open_need, later)
        for t in options:
            if t is None:
                continue
            if claim is None or self.in_time(V, t, claim[2]):
                return t
            return claim[0]
        if self.blocking(V, i, need):
            return self.unblock(V, legal, open_need, later) or WAIT
        return WAIT

    def blocking(self, V, i, need):
        """An idle cook stands at a station another cook holds something for (chopped veg for a pot, a plate for
        its soup, raw veg for a board, a dirty plate for a sink, a soup for the window, a spare plate for a rack)."""
        w = V.w
        x, y = V.pose[i][:2]
        here = {(x + d[0], y + d[1]) for d in ((0, -1), (0, 1), (1, 0), (-1, 0))}
        loose = {c for _, c in V.uncovered}
        for j in range(len(w.players)):
            h = V.held(j)
            if j == i or h is None:
                continue
            if h[0] == "chopped":
                wanted = {c for ing, c in need if ing == h[1]}
            elif h[0] == "raw":
                wanted = {c for c, k in w.kind.items() if k == "board"}
            else:
                state = V.plate(j)
                kinds = {"dirty": "sink", "soup": "serving", "clean": "rack"}
                wanted = {c for c, k in w.kind.items() if k == kinds.get(state)}
                if state == "clean":
                    wanted = {V.assign[j]} if j in V.assign else wanted | loose
            if here & wanted:
                return True
        return False

    def unblock(self, V, legal, open_need, later):
        """A harmless task that takes an idle cook elsewhere: fetch a dirty plate, a washed plate or a still needed
        vegetable, move a spare chopped vegetable off a board, else take a clean plate."""
        w = V.w
        sink_free = any(pid is None for pid in w.sink.values())
        miss = Counter(ing for ing, _ in open_need) + later
        return ((self.pick(legal, lambda t: t.action == "take-dirty-plate") if sink_free else None)
                or self.pick(legal, lambda t: t.action == "take-from-sink")
                or self.pick(legal, lambda t: t.event == "take_raw" and miss[self.provides(w, t)] > 0)
                or (self.pick(legal, lambda t: t.event == "take_chopped" and w.kind[t.cell] == "board")
                    if self.free_counter(V) else None)
                or self.pick(legal, lambda t: t.action == "take-clean-plate"))

    @staticmethod
    def free_counter(V):
        return any(st.kind == "counter" and V.w.surface[st.cell] is None for st in stations_of(V.w, V.counters))

    def carry(self, V, i, legal, held, need):
        w = V.w
        if held[0] == "chopped":
            pots = {c for ing, c in need if ing == held[1]}
            if pots:
                return self.pick(legal, lambda t: t.event == "add" and t.cell in pots)
            return (self.pick(legal, lambda t: t.event == "put_chopped" and w.kind[t.cell] == "counter")
                    or self.pick(legal, lambda t: t.event == "put_chopped"))
        if held[0] == "raw":
            return self.pick(legal, lambda t: t.event == "put_raw" and w.kind[t.cell] == "board")
        state = V.plate(i)
        if state == "soup":
            return self.pick(legal, lambda t: t.event == "deliver")
        if state == "dirty":
            return self.pick(legal, lambda t: t.event == "put_in_sink")
        if i in V.assign:
            mine = V.assign[i]
            loose = {c for _, c in V.uncovered}
            return (self.pick(legal, lambda t: t.event == "dish_up" and t.cell == mine)
                    or self.pick(legal, lambda t: t.event == "dish_up" and t.cell in loose))
        reach = {c for _, c in V.uncovered if _dist(w.lay, V.pose[i], c) < V.burn}
        if reach:
            return self.pick(legal, lambda t: t.event == "dish_up" and t.cell in reach) or WAIT
        return (self.pick(legal, lambda t: t.event == "rack_plate")
                or self.pick(legal, lambda t: t.event == "put_plate"))

    def claim(self, V, i, legal):
        """(task, now?, soup) for the first soup no plate holder covers: fetch a clean plate or wash one."""
        if not V.uncovered:
            return None
        w = V.w
        ready, pot = V.uncovered[0]
        sources = self.sources(V)
        lay = w.lay
        takes = [t for t in legal if t != WAIT and t.event == "take_plate" and t.cell in sources
                 and _dist(lay, t.pose, pot) < V.burn]
        if takes:
            best = min(takes, key=lambda t: (_dist(lay, t.pose, pot) * self.allowance >= V.burn,
                                             t.travel + 1 + _dist(lay, t.pose, pot), t.travel))
            lead = best.travel + 1 + _dist(lay, best.pose, pot) + 1
            now = ready - V.t <= lead * self.allowance + self.slack
            return best, now, (V.uncovered[0], sources)
        if sources:
            return None
        washing = sum(1 for pid in w.sink.values() if pid is not None) + sum(
            1 for k in range(len(w.players)) if V.plate(k) == "dirty")
        wash = self.pick(legal, lambda t: t.event == "wash")
        if wash is not None:
            return wash, True, None
        if washing < len(V.uncovered):
            drop = self.pick(legal, lambda t: t.action == "take-dirty-plate")
            if drop is not None:
                return drop, True, None
        return None

    def in_time(self, V, task, soup):
        """After task the cook can still fetch a plate and dish up the soup before it burns."""
        if soup is None:
            return True
        (ready, pot), sources = soup
        lead = self.lead(V, task.pose, sources, pot)
        return V.t + (task.duration + lead) * self.allowance + self.margin < ready + V.burn

    def work(self, V, i, legal, target, empty, unplaced, need, open_need, later):
        """Productive tasks in priority order (None where a rule has nothing)."""
        w = V.w
        out = []
        full = [c for c, r in target.items() if w.pots[c].contents and w.pots[c].comp() == r and self.plate_ok(V, c)]
        out.append(self.pick(legal, lambda t: t.event == "start" and t.cell in full))
        out.append(self.pick(legal, lambda t: t.event == "chop"))
        want = Counter(ing for ing, _ in need)
        for p in w.players:
            if p.held is not None and p.held[0] == "chopped":
                want[p.held[1]] -= 1
        out.append(self.pick(legal, lambda t: t.event == "take_chopped" and want[w.surface[t.cell][1]] > 0))
        dirty = [c for c, p in w.pots.items() if p.state in ("dirty", "burnt", "spoiled")]
        if len(empty) <= len(unplaced):
            out.append(self.pick(legal, lambda t: t.event == "scrub" and t.cell in dirty))
        boards = sum(1 for c, item in w.surface.items() if w.kind[c] == "board" and item is None)
        raw_held = sum(1 for p in w.players if p.held is not None and p.held[0] == "raw")
        if boards - raw_held <= 0 and (raw_held or open_need) and self.free_counter(V):
            out.append(self.pick(legal, lambda t: t.event == "take_chopped" and w.kind[t.cell] == "board"
                                 and want[w.surface[t.cell][1]] <= 0))
        if open_need and boards - raw_held > 0:
            miss = Counter(ing for ing, _ in open_need)
            out.append(self.pick(legal, lambda t: t.event == "take_raw" and miss[self.provides(w, t)] > 0))
        out.append(self.wash(V, legal))
        out.append(self.pick(legal, lambda t: t.event == "scrub"))
        if later and boards - raw_held > 0:
            out.append(self.pick(legal, lambda t: t.event == "take_raw" and later[self.provides(w, t)] > 0))
        return out

    @staticmethod
    def provides(w, t):
        k = w.kind[t.cell]
        if k in PROVIDES:
            return PROVIDES[k]
        return w.surface[t.cell][1]

    def plate_ok(self, V, pot):
        """A plate is free for one more soup and can reach the pot in its burn window."""
        w = V.w
        clean = sum(1 for c, s in w.stack.items() if w.kind[c] == "rack" for _ in s)
        sink = sum(1 for pid in w.sink.values() if pid is not None and w.plates[pid].state == "clean")
        if clean + sink + len(V.spare) - len(V.uncovered) < 1:
            return False
        return self.safe_pot(V, pot)

    def wash(self, V, legal):
        """Wash plates when the clean ones (with the plates being washed) are fewer than the open orders."""
        w = V.w
        clean = sum(1 for c, s in w.stack.items() if w.kind[c] == "rack" for _ in s) + len(V.holders)
        pipe = sum(1 for pid in w.sink.values() if pid is not None) + sum(
            1 for k in range(len(w.players)) if V.plate(k) == "dirty")
        plated = sum(1 for pl in w.plates.values() if pl.state == "soup")
        short = clean + pipe < len(w.pending_orders()) - plated
        wash = self.pick(legal, lambda t: t.event == "wash")
        if wash is not None:
            return wash
        if short:
            return self.pick(legal, lambda t: t.action == "take-dirty-plate")
        return None
