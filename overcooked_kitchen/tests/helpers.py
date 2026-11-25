"""Test helpers: a deterministic serial policy (one order at a time, one cook per floor region)."""
"""This file is UN-VERIFIED"""
from collections import Counter

from overcooked_kitchen.layout import INGREDIENTS, PROVIDES
from overcooked_kitchen.tasks import WAIT


def _comp_counter(comp):
    return Counter({ing: n for ing, n in zip(INGREDIENTS, comp) if n})


def _first(legal, pred):
    for t in legal:
        if t != WAIT and pred(t):
            return t
    return None


def _reach(w, i):
    """Station cells the cook can reach (its floor region)."""
    regions = w.lay.regions()
    r = regions[tuple(w.players[i].pos)]
    return {c for c in w.kind if any(regions.get(p[:2]) == r for p in w.lay.access_poses(c))}


def _safe_pots(w, reach):
    """Pots a cook waiting at the rack can dish up from before the soup burns (all pots if none)."""
    pots = sorted((c for c in w.pots if c in reach), key=lambda c: (c[1], c[0]))
    racks = sorted(c for c, k in w.kind.items() if k == "rack" and c in reach)
    if w.params.get("burn") is None or not racks:
        return pots
    dist = w.lay.bfs_poses(w.lay.access_poses(racks[0])[0])
    travel = {c: min((dist[p] for p in w.lay.access_poses(c) if p in dist), default=float("inf")) for c in pots}
    return [c for c in pots if travel[c] < w.params["burn"]] or pots


def serial_policy(sim, cook, legal):
    """Serve the earliest pending order: prep, cook, plate, deliver; scrub and wash only when needed."""
    w = sim.world
    i = sim.cooks.index(cook)
    held = w.players[i].held
    orders = w.pending_orders()
    if not orders:
        return WAIT
    want = _comp_counter(orders[0]["comp"])
    comp = tuple(orders[0]["comp"])
    reach = _reach(w, i)
    soup_on = any(p.state in ("cooking", "ready") and p.comp() == comp for c, p in w.pots.items() if c in reach)
    safe = _safe_pots(w, reach)
    idle = [c for c in safe if w.pots[c].state == "idle"]
    building = [c for c in idle if w.pots[c].contents and not (Counter(w.pots[c].contents) - want)]
    empty = [c for c in idle if not w.pots[c].contents]
    target_pot = (building or empty or [None])[0]
    have = Counter(w.pots[target_pot].contents) if target_pot is not None else Counter()
    if held is not None and held[0] in ("raw", "chopped"):
        have[held[1]] += 1
    for c, item in w.surface.items():
        if c in reach and item is not None and item[0] in ("raw", "chopped"):
            have[item[1]] += 1
    need = Counter() if soup_on else want - have
    pot_need = Counter() if soup_on else want - Counter(w.pots[target_pot].contents if target_pot else [])
    held_plate = w.plates[held[1]] if held is not None and held[0] == "plate" else None
    clean_free = any(w.plates[x].state == "clean" for c, s in w.stack.items() if c in reach for x in s) or any(
        x is not None and w.plates[x].state == "clean" for c, x in w.sink.items() if c in reach)
    plate_needed = soup_on and held is None

    choice = (
        _first(legal, lambda t: t.event == "deliver")
        or _first(legal, lambda t: t.event == "dish_up")
        or (plate_needed and _first(legal, lambda t: t.action in ("take-clean-plate", "take-from-sink")))
        or (held_plate is not None and held_plate.state == "dirty" and _first(legal, lambda t: t.event == "put_in_sink"))
        or (plate_needed and not clean_free and _first(legal, lambda t: t.event == "wash"))
        or (plate_needed and not clean_free and _first(legal, lambda t: t.action == "take-dirty-plate"))
        or (held_plate is not None and held_plate.state == "clean" and not soup_on
            and _first(legal, lambda t: t.event == "rack_plate"))
        or (held is None and target_pot is None and not soup_on and _first(legal, lambda t: t.event == "scrub" and t.cell in safe))
        or _first(legal, lambda t: t.event == "start" and w.pots[t.cell].comp() == comp)
        or (held is not None and held[0] == "chopped" and _first(legal, lambda t: t.event == "add" and t.cell == target_pot))
        or _first(legal, lambda t: t.event == "take_chopped" and pot_need[w.surface[t.cell][1]] > 0)
        or _first(legal, lambda t: t.event == "chop")
        or (held is not None and held[0] == "raw" and _first(legal, lambda t: t.event == "put_raw" and w.kind[t.cell] == "board"))
        or (held is None and target_pot is not None
            and _first(legal, lambda t: t.action == "take-raw" and need[PROVIDES[w.kind[t.cell]]] > 0))
    )
    return choice or WAIT
