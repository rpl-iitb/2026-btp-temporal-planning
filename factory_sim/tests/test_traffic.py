"""Lane traffic (traffic: {mode: lanes}): junctions hold one robot, robots wait and back off, nothing deadlocks;
the generated plants; the hard scenario families and their PDDL."""
import copy

import pytest

from factory_sim import Factory, Sim, baselines, scenarios
from factory_sim import pddl as FP
from factory_sim.config import read
from factory_sim.env import build_sim
from factory_sim.layouts import plant


def lanes(cfg, robots, orders):
    cfg = copy.deepcopy(cfg)
    cfg["traffic"] = dict(cfg.get("traffic") or {}, mode="lanes")
    cfg["congestion"]["slowdown"] = 0.0
    cfg["robots"]["count"], cfg["orders"]["count"] = robots, orders
    return Factory(cfg)


class Watch:
    """A policy wrapper that checks at every decision that no junction holds two robots."""

    def __init__(self, policy):
        self.p, self.queue_rule = policy, getattr(policy, "queue_rule", None)
        self.max_on_junction = 0

    def __call__(self, sim, agent, legal):
        on = {}
        for r in sim.robots.values():
            for n in ([r.src, r.dest] if r.phase == "travel" else [r.node]):
                if n in sim.F.junctions:
                    on.setdefault(n, set()).add(r.name)
        assert all(len(v) == 1 for v in on.values()), (sim.t, on)
        return self.p(sim, agent, legal)


def test_junctions_and_pockets():
    F = lanes(read("lab"), 4, 14)
    assert F.lanes and {"k1", "k5", "k9"} <= F.junctions
    assert {"kitting", "store", "charger", "MC1"} <= F.pockets     # docks and dead ends hold any number
    assert all(F.bays[j] for j in F.junctions)                      # every junction has a pocket to back into
    for a in ("store", "kitting", "MC1"):                           # routes pass through junctions only
        for b in ("inspect", "asm1"):
            assert all(n in F.junctions for n in F.path(a, F.stations[b]["node"])[1:-1])


@pytest.mark.parametrize("policy", ["fifo", "nearest", "edd", "spt"])
@pytest.mark.parametrize("time_source", ["nominal", "sampled"])
def test_lab_lanes_no_collision_no_deadlock(policy, time_source):
    F = lanes(read("lab"), 4, 14)
    for seed in range(2):
        sim = Sim(F, seed=seed, time_source=time_source, cv_scale=2.0)
        m = sim.run(Watch(baselines.make(policy)))
        assert m["success"], (m["failure"], seed)
        assert sim.stats["traffic_wait"] > 0                        # robots really wait for each other


def test_traffic_makes_it_slower():
    cfg = read("lab")
    a = Sim(lanes(cfg, 4, 14), seed=0, time_source="nominal").run(baselines.make("nearest"))["makespan"]
    c = copy.deepcopy(cfg)
    c["robots"]["count"], c["orders"]["count"], c["congestion"]["slowdown"] = 4, 14, 0.0
    b = Sim(Factory(c), seed=0, time_source="nominal").run(baselines.make("nearest"))["makespan"]
    assert a > 1.2 * b


@pytest.mark.parametrize("lines", [1, 2, 3])
def test_plant(lines):
    cfg = plant(lines)
    F = lanes(cfg, 2 + 2 * lines, 6 * lines)
    deg = {j: sum(1 for m in F.adj[j] if m in F.junctions) for j in F.junctions}
    assert (max(deg.values()) >= 3) == (lines > 1)                  # cross aisles make intersections
    assert all(F.bays[j] for j in F.junctions)
    assert len(F.sinks) == 1 and len(F.kitting) == lines
    sim = Sim(F, seed=1, time_source="sampled")
    m = sim.run(Watch(baselines.make("nearest")))
    assert m["success"], m["failure"]


def test_head_on_resolved():
    """Two robots meet head-on in the lab corridor many times; the one with the lower priority backs off."""
    F = lanes(read("lab"), 2, 8)
    sim = Sim(F, seed=0, time_source="nominal")
    m = sim.run(baselines.make("fifo"))
    assert m["success"] and sim.stats["yields"] > 0


def test_export_mid_run():
    """The PDDL export of a state with robots between junctions and robots waiting for one."""
    F = lanes(read("lab"), 4, 14)
    sim = Sim(F, seed=0, time_source="nominal")
    pol, seen = baselines.make("nearest"), set()
    for _ in range(4000):
        a = sim.next_decision()
        if a is None:
            break
        sim.act(a, pol(sim, a, sim.legal(a)))
        phases = {r.phase for r in sim.robots.values()}
        if "blocked" in phases and "travel" in phases and "blocked" not in seen:
            seen.add("blocked")
            text = FP.export(sim)
            assert "(node-free" in text and "(link n_k1 n_k2)" in text
            held = {r.node for r in sim.robots.values() if r.phase != "travel" and r.node in F.junctions}
            assert not any("(node-free n_{})".format(n.lower()) in text.split("(:init")[1].split("(at ")[0] for n in held)
    assert "blocked" in seen
    dom = FP.domain(F)
    assert "(:durative-action move-j" in dom and "(domain factorysim-lanes)" in dom


@pytest.mark.parametrize("family", scenarios.HARD_FAMILIES)
def test_hard_families(family):
    for size in ("S", "XL"):
        inst = scenarios.make_instance(family, size, "test", 1)
        assert inst["config"]["traffic"]["mode"] == "lanes"
        gt = inst["ground_truth"]
        assert gt["lower_bound"] < float("inf") and "due_reference" in gt
        due = [o["due"] for o in inst["orders"]]
        assert all(d > o["arrival"] for d, o in zip(due, inst["orders"]))
        sim = build_sim(inst, seed=0, time_source="nominal")
        prob = FP.export(sim, hard_deadlines=True, name=inst["name"])
        assert prob.count("(not (open j") == len(inst["orders"])


def test_old_families_unchanged():
    """Adding traffic left the earlier families as they were (no traffic key, the same orders)."""
    inst = scenarios.make_instance("tight", "M", "test", 1)
    assert inst["config"].get("traffic") is None and inst["dims"]["traffic"] == "none"
    assert not Factory(inst["config"]).lanes
