"""Rule-based multi-cook dispatcher over overcooked_kitchen PDDL states (menv; baseline, DAgger expert of V12 / V13,
and the option generator of V13).

    from overcooked_test.kitchen_dispatcher import Dispatcher
    res = env.rollout(Dispatcher(env))

A pure function of the state (no hidden bookkeeping), so it can label any state a learned policy visits
(DAgger). For an agent with applicable actions it returns one of them, or None (wait). Priorities:
  0. finish lighting; chop a raw vegetable in front of you; take a chopped vegetable a pot needs
  1. carry what you hold where it is needed: chopped veg -> a pot that needs it (else park it on a free surface);
     raw veg -> a clear board; a clean plate -> the soup that is ready first (dish up when ready);
     a plated soup -> the serving window (open order of that dish, earliest deadline first);
     a dirty plate -> a sink; a clean plate nobody needs -> the rack
  2. empty-handed: claim a soup that will be ready soon (fetch a clean plate, or wash one first);
     light a pot that holds a full recipe; take a needed chopped vegetable from a board;
     scrub a dirty pot when no clean pot is free; fetch the vegetable that is needed most (nearest crate);
     wash a dirty plate if clean plates are short; else wait.
Walking: to the free pose facing the target with the shortest travel.
"""
from collections import Counter, defaultdict

INGS = ("onion", "tomato", "carrot")
CLAIM_SLACK = 4.0
WAIT = object()          # an intentional wait (do not yield the tile)


class View:
    """Kitchen facts of a state, decoded once per decision."""

    def __init__(self, env, S):
        T = env.T
        self.t = S.t
        F = [T.facts[f] for f in S.facts]
        self.at, self.raw, self.chop, self.plate, self.lighting = {}, {}, {}, {}, {}
        self.pot_has, self.pot_idle, self.pot_dirty, self.on_stove, self.ready = {}, set(), set(), {}, set()
        self.clear, self.on_raw, self.on_chop = set(), {}, {}
        self.plate_at, self.clean, self.dirty, self.soup = {}, set(), set(), {}
        self.pending, self.open, self.free_tiles = set(), set(), set()
        for f in F:
            p = f[0]
            if p == "at": self.at[f[1]] = f[2]
            elif p == "holding-raw": self.raw[f[1]] = f[2]
            elif p == "holding-chopped": self.chop[f[1]] = f[2]
            elif p == "holding-plate": self.plate[f[1]] = f[2]
            elif p == "lighting": self.lighting[f[1]] = f[2]
            elif p == "pot-has": self.pot_has[f[1]] = tuple(int(x[1:]) for x in f[2:5])
            elif p == "pot-idle": self.pot_idle.add(f[1])
            elif p == "pot-dirty": self.pot_dirty.add(f[1])
            elif p == "on-stove": self.on_stove[f[1]] = f[2]
            elif p == "soup-ready": self.ready.add(f[1])
            elif p == "clear": self.clear.add(f[1])
            elif p == "on-raw": self.on_raw[f[1]] = f[2]
            elif p == "on-chopped": self.on_chop[f[1]] = f[2]
            elif p == "plate-at": self.plate_at[f[1]] = f[2]
            elif p == "plate-clean": self.clean.add(f[1])
            elif p == "plate-dirty": self.dirty.add(f[1])
            elif p == "plate-soup": self.soup[f[1]] = f[2]
            elif p == "pending": self.pending.add(f[1])
            elif p == "open": self.open.add(f[1])
            elif p == "free": self.free_tiles.add(f[1])
        # soups: ready time of each lit pot (cook action end, or now if ready)
        self.ready_at = {}
        for i, st, lo, hi in S.run:
            a = T.actions[i]
            if a.name == "cook":
                self.ready_at[a.args[0]] = lo          # actual end is hidden; lo is the nominal end
        for p in self.on_stove:
            if p in self.ready:
                self.ready_at[p] = self.t
            self.ready_at.setdefault(p, self.t + 1e9 if p not in self.ready else self.t)


class Dispatcher:
    def __init__(self, env, claim_slack=CLAIM_SLACK):
        self.env, self.slack = env, claim_slack
        T = env.T
        st = defaultdict(list)
        for f in T.static_init:
            st[f[0]].append(f)
        self.faces = defaultdict(list)                  # station -> poses facing it
        for f in st["faces"]:
            self.faces[f[2]].append(f[1])
        self.tile = {f[1]: f[2] for f in st["pose-tile"]}
        self.provides = defaultdict(list)
        for f in st["provides"]:
            self.provides[f[2]].append(f[1])
        self.comp = {f[1]: tuple(int(x[1:]) for x in f[2:5]) for f in st["recipe-of"]}
        self.order_recipe = {f[1]: f[2] for f in st["order-recipe"]}
        objs = dict(T.dom.constants); objs.update(T.prob.objects)
        kind = lambda ty: set(T.dom.ancestors(ty))      # noqa: E731
        self.of = defaultdict(list)
        for o, ty in sorted(objs.items()):
            for k in ("board", "pot", "rack", "dropoff", "sink", "serving", "counter", "surface", "crate"):
                if k in kind(ty):
                    self.of[k].append(o)
        self.travel = {(k[1], k[2]): v for k, v in T.prob.fluents.items() if k[0] == "travel"}
        self.fl = {k: v for k, v in T.prob.fluents.items() if k[0] in ("cook-steps", "burn-window")}
        self.deadline = {}
        for tl, f, pos in T.tils:
            fa = T.facts[f]
            if fa[0] == "open" and not pos:
                self.deadline[fa[1]] = tl
        self.acts = T.actions

    # ---------------------------------------------------------------- helpers
    def d(self, a, b):
        return 0.0 if a == b else self.travel.get((a, b), 1e6)

    def dist_to(self, pose, station):
        return min((self.d(pose, q) for q in self.faces[station]), default=1e6)

    def faces_now(self, V, c, station):
        return V.at.get(c) in self.faces[station]

    def pick(self, legal, name, pred=lambda a: True):
        best = None
        for k, i in enumerate(legal):
            a = self.acts[i]
            if a.name == name and pred(a):
                best = k
                break
        return best

    def go(self, V, c, station, legal):
        """Index of the move/turn towards the nearest free pose facing station, or None."""
        cur = V.at.get(c)
        best, bd = None, 1e18
        for k, i in enumerate(legal):
            a = self.acts[i]
            if a.name in ("move", "turn") and a.args[2] in self.faces[station]:
                dd = self.d(cur, a.args[2])
                if dd < bd:
                    best, bd = k, dd
        return best

    def nearest(self, V, c, stations):
        cur = V.at.get(c)
        return min(stations, key=lambda s: (self.dist_to(cur, s), s)) if stations else None

    # ---------------------------------------------------------------- kitchen bookkeeping
    def targets(self, V):
        """pot -> target recipe for pots being filled; plus pots free for new dishes and unassigned orders."""
        busy = Counter()                                           # recipe -> dishes in progress
        for p, r in V.on_stove.items():
            busy[r] += 1
        for pl, r in V.soup.items():
            busy[r] += 1
        need_orders = Counter(self.order_recipe[o] for o in V.pending)
        target = {}
        filling = [p for p in self.of["pot"] if p in V.pot_idle and sum(V.pot_has.get(p, (0, 0, 0))) > 0]
        free_need = need_orders - busy
        for p in sorted(filling, key=lambda p: -sum(V.pot_has[p])):
            have = V.pot_has[p]
            cands = [r for r in free_need if free_need[r] > 0 and all(h <= c for h, c in zip(have, self.comp[r]))]
            if cands:
                r = min(cands, key=lambda r: (sum(self.comp[r]) - sum(have), r))
                target[p] = r
                free_need[r] -= 1
        empty = [p for p in self.of["pot"] if p in V.pot_idle and sum(V.pot_has.get(p, (0, 0, 0))) == 0]
        new = []
        for r in sorted(free_need.elements()):
            new.append(r)
        return target, empty, new

    def needs(self, V, target, empty, new):
        """ingredient -> list of pots needing it (with multiplicity), after counting vegetables already in hands/boards."""
        need = []
        for p, r in target.items():
            for k, ing in enumerate(INGS):
                need += [(ing, p)] * max(0, self.comp[r][k] - V.pot_has[p][k])
        # new dishes go to free pots (one dish per pot)
        for p, r in zip(sorted(empty), new):
            for k, ing in enumerate(INGS):
                need += [(ing, p)] * self.comp[r][k]
        supply = Counter(list(V.raw.values()) + list(V.chop.values()) + list(V.on_raw.values()) + list(V.on_chop.values()))
        out = []
        for ing, p in need:
            if supply[ing] > 0:
                supply[ing] -= 1
            else:
                out.append((ing, p))
        return need, out

    def soups(self, V, target):
        """Soups that need a plate, in order of (estimated) ready time: lit pots and full pots about to be lit."""
        return sorted((V.ready_at[p], p) for p in V.on_stove)

    def holders(self, V):
        return sorted(cc for cc, pp in V.plate.items() if pp in V.clean and pp not in V.soup)

    # ---------------------------------------------------------------- the rules
    def __call__(self, env, S, c, legal):
        V = View(env, S)
        k = self.decide(V, c, legal)
        if k is WAIT or k is None:          # waiting: not on a station's access tile, park at a counter
            if k is WAIT and self.soon(V, c):
                return None
            q = self.park(V, c, legal)
            if q is not None:
                return q
            return None if k is WAIT else self.yield_tile(V, c, legal)
        return k

    def soon(self, V, c, within=3.0):
        """Waiting at a lit pot whose soup is ready soon."""
        for p in V.on_stove:
            if self.faces_now(V, c, p) and V.ready_at[p] - V.t <= within:
                return True
        return False

    def park(self, V, c, legal):
        cur = V.at.get(c)
        if not cur or any(cur in self.faces[k] for k in self.of["counter"]) or \
                not any(cur in self.faces[s] for s in self.faces if s not in self.of["counter"]):
            return None
        best, bd = None, 1e18
        for k, i in enumerate(legal):
            a = self.acts[i]
            if a.name == "move" and any(a.args[2] in self.faces[ct] for ct in self.of["counter"]):
                dd = self.d(cur, a.args[2])
                if dd < bd:
                    best, bd = k, dd
        return best

    def intent(self, V, c, target, need_all):
        """Station a loaded cook is heading for (None if empty-handed)."""
        if c in V.chop:
            pots = sorted({p for i_, p in need_all if i_ == V.chop[c]})
            return self.nearest(V, c, pots) if pots else None
        if c in V.raw:
            return self.nearest(V, c, [b for b in self.of["board"] if b in V.clear])
        if c in V.plate:
            pl = V.plate[c]
            if pl in V.soup:
                return self.nearest(V, c, self.of["serving"])
            if pl in V.dirty:
                return self.nearest(V, c, [w for w in self.of["sink"] if w in V.clear])
            soups, hs = self.soups(V, target), self.holders(V)
            if c in hs and hs.index(c) < len(soups):
                return soups[hs.index(c)][1]
        return None

    def yield_tile(self, V, c, legal):
        target, empty, new = self.targets(V)
        need_all, _ = self.needs(V, target, empty, new)
        mine = self.tile.get(V.at.get(c))
        wanted = set()
        for cc in V.at:
            if cc == c:
                continue
            st = self.intent(V, cc, target, need_all)
            if st is not None and not self.faces_now(V, cc, st):
                wanted |= {self.tile[q] for q in self.faces[st]}
        if mine not in wanted:
            return None
        best, bd = None, 1e18
        for k, i in enumerate(self.acts[j] for j in legal):
            a = self.acts[legal[k]]
            if a.name == "move" and self.tile[a.args[2]] not in wanted:
                dd = self.d(V.at.get(c), a.args[2])
                if dd < bd:
                    best, bd = k, dd
        return best

    def to(self, V, c, station, legal, name, pred=None):
        """Act on station if facing it (action `name`), else walk there. None if neither is possible."""
        if station is None:
            return None
        if self.faces_now(V, c, station):
            return self.pick(legal, name, lambda a: a.args[2] == station and (pred is None or pred(a)))
        return self.go(V, c, station, legal)

    def decide(self, V, c, legal):
        A = self.acts
        cur = V.at.get(c)
        target, empty, new = self.targets(V)
        need_all, need_open = self.needs(V, target, empty, new)

        k = self.pick(legal, "finish-lighting")
        if k is not None:
            return k
        # ---- holding something
        if c in V.chop:
            ing = V.chop[c]
            pots = sorted({p for i_, p in need_all if i_ == ing})
            k = self.to(V, c, self.nearest(V, c, pots), legal, "add-" + ing) if pots else None
            if k is None:
                k = self.to(V, c, self.nearest(V, c, [s for s in self.of["surface"] if s in V.clear]), legal, "put-chopped")
            return k
        if c in V.raw:
            return self.to(V, c, self.nearest(V, c, [b for b in self.of["board"] if b in V.clear]), legal, "put-raw")
        if c in V.plate:
            pl = V.plate[c]
            if pl in V.soup:
                s = self.nearest(V, c, self.of["serving"])
                if self.faces_now(V, c, s):
                    cands = [(self.deadline.get(A[i].args[3], 1e9), kk) for kk, i in enumerate(legal)
                             if A[i].name == "deliver" and A[i].args[2] == s]
                    return min(cands)[1] if cands else None
                return self.go(V, c, s, legal)
            if pl in V.dirty:
                return self.to(V, c, self.nearest(V, c, [w for w in self.of["sink"] if w in V.clear]), legal, "put-in-sink")
            # clean plate: the soup that is ready first and not claimed by another plate holder
            soups = self.soups(V, target)
            idx = self.holders(V).index(c)
            if idx < len(soups):
                ready_t, p = soups[idx]
                lit = p in V.on_stove
                if self.faces_now(V, c, p):
                    if p in V.ready:
                        return self.pick(legal, "dish-up", lambda a: a.args[2] == p)
                    if not lit:                       # do not block an unlit pot: step aside
                        return None
                    return WAIT
                if lit and ready_t - V.t <= self.dist_to(cur, p) + 2:
                    k = self.go(V, c, p, legal)
                    return k if k is not None else WAIT
                return WAIT if not self.blocking(V, c, target, need_all) else None
            return self.to(V, c, self.nearest(V, c, self.of["rack"]), legal, "rack-plate")
        # ---- empty hand: act on what is in front of you
        for name in ("chop", "wash-plate", "take-from-sink"):
            k = self.pick(legal, name)
            if k is not None:
                return k
        # (a) soups that need a plate soon
        soups = self.soups(V, target)
        nh = len(self.holders(V))
        if len(soups) > nh:
            ready_t, p = soups[nh]
            racks = [d for d in self.of["rack"] if any(V.plate_at.get(pl) == d and pl in V.clean for pl in V.plate_at)]
            d = self.nearest(V, c, racks)
            if d is not None:
                lead = self.dist_to(cur, d) + self.dist_to(self.faces[d][0], p) + 2
                if ready_t - V.t <= lead + self.slack or p not in V.on_stove:
                    k = self.to(V, c, d, legal, "take-clean-plate")
                    if k is not None:
                        return k
            else:
                drops = [x for x in self.of["dropoff"] if any(V.plate_at.get(pl) == x and pl in V.dirty for pl in V.plate_at)]
                washing = any(V.plate_at.get(pl) in self.of["sink"] for pl in V.plate_at) or \
                    any(pp in V.dirty for pp in V.plate.values())
                if drops and not washing:
                    k = self.to(V, c, self.nearest(V, c, drops), legal, "take-dirty-plate")
                    if k is not None:
                        return k
        # (b) light a full pot whose plate holder can reach it within the burn window
        full = [p for p, r in target.items() if V.pot_has[p] == self.comp[r] and self.plate_reachable(V, p, r, soups)]
        k = self.to(V, c, self.nearest(V, c, full), legal, "start-cooking")
        if k is not None:
            return k
        # (c) take a needed chopped vegetable from a board / counter
        need_ings = Counter(i_ for i_, _ in need_all)
        held = Counter(V.chop.values())
        avail = [s for s, ing in V.on_chop.items() if need_ings[ing] - held[ing] > 0]
        k = self.to(V, c, self.nearest(V, c, avail), legal, "take-chopped")
        if k is not None:
            return k
        # (d) scrub a dirty pot when no clean pot is free for the remaining dishes
        if V.pot_dirty and len(empty) < len(new) + 1:
            k = self.to(V, c, self.nearest(V, c, sorted(V.pot_dirty)), legal, "scrub-pot")
            if k is not None:
                return k
        # (e) fetch a needed vegetable (board capacity permitting)
        boards_free = sum(1 for b in self.of["board"] if b in V.clear)
        if need_open and boards_free - len(V.raw) > 0:
            opts = sorted((self.dist_to(cur, cr), cr, ing) for ing in sorted({i_ for i_, _ in need_open})
                          for cr in self.provides[ing])
            for _, cr, ing in opts:
                k = self.to(V, c, cr, legal, "take-raw", lambda a, ing=ing: a.args[3] == ing)
                if k is not None:
                    return k
        # (f) wash plates if clean ones are short
        clean_avail = sum(1 for pl in V.clean if V.plate_at.get(pl) in self.of["rack"]) + len(self.holders(V))
        if clean_avail < len(V.pending):
            drops = [x for x in self.of["dropoff"] if any(V.plate_at.get(pl) == x and pl in V.dirty for pl in V.plate_at)]
            k = self.to(V, c, self.nearest(V, c, drops), legal, "take-dirty-plate")
            if k is not None:
                return k
        # (g) scrub anyway
        if V.pot_dirty:
            k = self.to(V, c, self.nearest(V, c, sorted(V.pot_dirty)), legal, "scrub-pot")
            if k is not None:
                return k
        return None

    def plate_reachable(self, V, p, r, soups):
        """A clean plate can be at pot p within its burn window: a spare plate holder close enough, or a rack
        with a clean plate close enough to fetch after lighting."""
        window = self.fl.get(("burn-window", r), 20) - 3
        on_racks = sum(1 for pl in V.clean if V.plate_at.get(pl) in self.of["rack"])
        soon = sum(1 for pl, at in V.plate_at.items() if at in self.of["sink"]) + \
            sum(1 for pp in V.plate.values() if pp in V.dirty)
        if on_racks + len(self.holders(V)) + soon - len(soups) < 1:
            return False                                  # every plate is already spoken for
        spare = self.holders(V)[len(soups):]
        if any(self.dist_to(V.at.get(h), p) + 2 <= window for h in spare):
            return True
        for d in self.of["rack"]:
            pd = min(self.d(q, q2) for q in self.faces[p] for q2 in self.faces[d])
            if 2 * pd + 3 <= window:
                return True
        return False

    def free_hands(self, V, but=None):
        return [cc for cc in V.at if cc != but and cc not in V.raw and cc not in V.chop and cc not in V.plate]

    def blocking(self, V, c, target, need_all):
        mine = self.tile.get(V.at.get(c))
        for cc in V.at:
            if cc != c:
                st = self.intent(V, cc, target, need_all)
                if st is not None and mine in {self.tile[q] for q in self.faces[st]}:
                    return True
        return False

    # ---------------------------------------------------------------- all options (for learned task choice, V13)
    RULES = ("finish", "add", "park-chopped", "put-raw", "deliver", "sink", "dish", "rack", "here", "claim",
             "wash", "light", "take-chopped", "scrub", "fetch", "park", "wait")

    def options(self, V, c, legal):
        """Every task cook c could take now: list of dicts(rule, station, k, info), k = index into legal of its
        next step (None for an intentional wait). Same task set as decide(), with every alternative station."""
        out = []
        cur = V.at.get(c)
        target, empty, new = self.targets(V)
        need_all, need_open = self.needs(V, target, empty, new)

        def add(rule, station, k, **info):
            if k is not None or rule == "wait":
                out.append({"rule": rule, "station": station, "k": k, "info": info,
                            "dist": self.dist_to(cur, station) if station else 0.0})
        k = self.pick(legal, "finish-lighting")
        if k is not None:
            add("finish", None, k)
            return out
        if c in V.chop:
            ing = V.chop[c]
            for p in sorted({p for i_, p in need_all if i_ == ing}):
                add("add", p, self.to(V, c, p, legal, "add-" + ing))
            for s in sorted(s for s in self.of["surface"] if s in V.clear):
                add("park-chopped", s, self.to(V, c, s, legal, "put-chopped"))
        elif c in V.raw:
            for b in sorted(b for b in self.of["board"] if b in V.clear):
                add("put-raw", b, self.to(V, c, b, legal, "put-raw"))
        elif c in V.plate:
            pl = V.plate[c]
            if pl in V.soup:
                for s in self.of["serving"]:
                    if self.faces_now(V, c, s):
                        cands = [(self.deadline.get(self.acts[i].args[3], 1e9), kk) for kk, i in enumerate(legal)
                                 if self.acts[i].name == "deliver" and self.acts[i].args[2] == s]
                        if cands:
                            add("deliver", s, min(cands)[1], deadline=min(cands)[0])
                    else:
                        add("deliver", s, self.go(V, c, s, legal))
            elif pl in V.dirty:
                for w in sorted(w for w in self.of["sink"] if w in V.clear):
                    add("sink", w, self.to(V, c, w, legal, "put-in-sink"))
            else:
                for ready_t, p in self.soups(V, target):
                    lit = p in V.on_stove
                    if self.faces_now(V, c, p):
                        if p in V.ready:
                            add("dish", p, self.pick(legal, "dish-up", lambda a: a.args[2] == p), ready=ready_t, lit=lit)
                    else:
                        add("dish", p, self.go(V, c, p, legal), ready=ready_t, lit=lit)
                for d in self.of["rack"]:
                    add("rack", d, self.to(V, c, d, legal, "rack-plate"))
        else:
            for name in ("chop", "wash-plate", "take-from-sink"):
                k = self.pick(legal, name)
                if k is not None:
                    add("here", self.acts[legal[k]].args[2], k, what=name)
            soups = self.soups(V, target)
            nh = len(self.holders(V))
            if len(soups) > nh:
                ready_t, p = soups[nh]
                for d in self.of["rack"]:
                    if any(V.plate_at.get(pl) == d and pl in V.clean for pl in V.plate_at):
                        add("claim", d, self.to(V, c, d, legal, "take-clean-plate"), ready=ready_t, lit=p in V.on_stove)
            for x in self.of["dropoff"]:
                if any(V.plate_at.get(pl) == x and pl in V.dirty for pl in V.plate_at):
                    add("wash", x, self.to(V, c, x, legal, "take-dirty-plate"))
            for p, r in target.items():
                if V.pot_has[p] == self.comp[r] and self.plate_reachable(V, p, r, soups):
                    add("light", p, self.to(V, c, p, legal, "start-cooking"), window=self.fl.get(("burn-window", r), 20))
            need_ings = Counter(i_ for i_, _ in need_all)
            held = Counter(V.chop.values())
            for s, ing in V.on_chop.items():
                if need_ings[ing] - held[ing] > 0:
                    add("take-chopped", s, self.to(V, c, s, legal, "take-chopped"))
            for p in sorted(V.pot_dirty):
                add("scrub", p, self.to(V, c, p, legal, "scrub-pot"), short=len(empty) < len(new) + 1)
            if sum(1 for b in self.of["board"] if b in V.clear) - len(V.raw) > 0:
                missing = Counter(i_ for i_, _ in need_open)
                for ing in sorted(missing):
                    for cr in self.provides[ing]:
                        add("fetch", cr, self.to(V, c, cr, legal, "take-raw", lambda a, ing=ing: a.args[3] == ing),
                            missing=missing[ing])
        q = self.park(V, c, legal)
        if q is None:
            q = self.yield_tile(V, c, legal)
        if q is not None:
            add("park", None, q)
        add("wait", None, None)
        # de-duplicate identical next steps (keep the first, i.e. the higher-priority rule)
        seen, uniq = set(), []
        for o in out:
            key = ("wait",) if o["k"] is None else o["k"]
            if key in seen:
                continue
            seen.add(key)
            uniq.append(o)
        return uniq
