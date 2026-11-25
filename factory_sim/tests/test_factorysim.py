import os
import shutil
import subprocess
import tempfile

import numpy as np
import pytest

from factory_sim import Factory, Sim, baselines, scenarios
from factory_sim.backends import run_external
from factory_sim.config import read
from factory_sim.env import FactoryEnv, aec_env, build_sim


def small(robots=1, unlimited=False):
    cfg = read("lab")
    cfg["robots"]["count"] = robots
    if unlimited:
        for s in cfg["stations"].values():
            if s["kind"] != "sink":
                s["slots_in"], s["slots_out"] = 1000, (1000 if s.get("slots_out", 1) > 0 else 0)
        cfg["congestion"] = {"dock_capacity": 1000, "slowdown": 0.0}
    return Factory(cfg)


def invariants(sim):
    seen = []
    for st in sim.stations.values():
        seen += [j.id for j in st.q] + [j.id for j in st.out] + ([st.cur.id] if st.cur is not None else [])
    seen += [r.carry for r in sim.robots.values() if r.carry is not None]
    assert len(seen) == len(set(seen)), "a tray is in two places"
    for st in sim.stations.values():
        assert len(st.q) + st.reserved <= st.spec["slots_in"] or st.spec["kind"] == "sink"
        assert len(st.out) <= st.spec["slots_out"]
    assert all(v >= 0 for v in sim.ops_free.values())


def test_bad_config_is_rejected():
    cfg = read("lab")
    cfg["stations"]["MC1"]["kind"] = "lathe"
    with pytest.raises(ValueError):
        Factory(cfg)


@pytest.mark.parametrize("policy", baselines.RULES)
def test_rules_finish_and_keep_invariants(policy):
    sim = Sim(Factory("lab"), seed=4)
    pol = baselines.make(policy, seed=4)
    while True:
        a = sim.next_decision()
        if a is None:
            break
        sim.act(a, pol(sim, a, sim.legal(a)))
        invariants(sim)
    assert sim.done and sim.metrics()["completed"] == sim.n_target


def test_routes_followed_in_order():
    sim = Sim(Factory("lab"), seed=1, time_source="nominal")
    sim.run(baselines.make("edd"))
    for j in sim.jobs.values():
        tags = [e["event"] for e in sim.rfid if e["job"] == j.id]
        ins = [t[:-3] for t in tags if t.endswith(" IN")]
        order = [s for opts in j.route for s in opts if sim.F.stations[s].get("tag") and "IN" in sim.F.stations[s].get("log", ["IN", "OUT"])]
        assert [x for x in ins if x != "INSPECTION"] == [sim.F.stations[s]["tag"] for s in order if sim.F.stations[s]["tag"] != "INSPECTION"]


def test_seeds_are_reproducible():
    a = Sim(Factory("lab"), seed=7).run(baselines.make("edd"))
    b = Sim(Factory("lab"), seed=7).run(baselines.make("edd"))
    c = Sim(Factory("lab"), seed=8).run(baselines.make("edd"))
    assert a == b and a["makespan"] != c["makespan"]


def test_external_clock_matches_nominal():
    F = Factory("lab")
    a = Sim(F, seed=0, time_source="nominal").run(baselines.make("edd"))
    b = run_external(Sim(F, seed=0, time_source="external"), baselines.make("edd"))
    assert a["makespan"] == pytest.approx(b["makespan"])


def test_gymnasium_api():
    from gymnasium.utils.env_checker import check_env
    env = FactoryEnv("lab")
    check_env(env, skip_render_check=True)
    obs, info = env.reset(seed=1)
    rng = np.random.default_rng(0)
    for _ in range(5000):
        obs, r, term, trunc, info = env.step(rng.choice(np.flatnonzero(info["action_mask"])))
        assert env.observation_space.contains(obs)
        if term or trunc:
            break
    assert term and info["metrics"]["success"]


def test_pettingzoo_api():
    pytest.importorskip("pettingzoo")
    from pettingzoo.test import api_test
    api_test(aec_env(instance="lab"), num_cycles=200, verbose_progress=False)


@pytest.mark.parametrize("family", list(scenarios.FAMILIES))
def test_every_family_generates_and_runs(family):
    inst = scenarios.make_instance(family, "S", "test", 0)
    m = build_sim(inst, seed=1).run(baselines.make("nearest"))
    assert m["completed"] > 0 or m["failure"] in ("deadlock", "stalled")
    assert inst["ground_truth"]["lower_bound"] > 0


def test_lower_bound_cpsat_and_execution_agree():
    pytest.importorskip("ortools")
    from factory_sim import ground_truth as G
    F = small(robots=2, unlimited=True)
    orders = [{"id": i, "product": p, "arrival": 0, "due": 1e9} for i, p in enumerate(["PR1", "PR2", "PR4"])]
    lb, _ = G.lower_bound(F, orders)
    sol = G.cpsat(F, orders, time_limit=20)
    ex = Sim(F, orders=orders, time_source="nominal").run(G.SchedulePolicy(sol))
    assert sol["exact"] and lb <= sol["makespan"] + 1e-6 <= ex["makespan"] + 1e-6
    assert ex["makespan"] <= 1.1 * sol["makespan"]


def _tool(name):
    return os.environ.get("FACTORYSIM_" + name.upper()) or shutil.which(name)


@pytest.mark.skipif(not (_tool("popf") and _tool("validate")), reason="set FACTORYSIM_POPF and FACTORYSIM_VALIDATE to run")
def test_pddl_plan_is_val_valid_and_executes():
    from factory_sim import pddl
    sim = Sim(small(robots=1), orders=[{"id": 0, "product": "PR1", "arrival": 0, "due": 1e9}], time_source="nominal")
    d = tempfile.mkdtemp()
    dom, prob = pddl.write(sim, os.path.join(d, "domain.pddl"), os.path.join(d, "problem.pddl"))
    out = subprocess.run([_tool("popf"), dom, prob], capture_output=True, text=True, timeout=300).stdout
    plan = "\n".join(l for l in out.split(";;;; Solution Found")[-1].splitlines() if l[:1].isdigit())
    open(os.path.join(d, "plan.txt"), "w").write(plan)
    val = subprocess.run([_tool("validate"), "-t", "0.001", dom, prob, os.path.join(d, "plan.txt")], capture_output=True, text=True).stdout
    assert "Plan valid" in val
    m = Sim(small(robots=1), orders=[{"id": 0, "product": "PR1", "arrival": 0, "due": 1e9}], time_source="nominal").run(pddl.PlanPolicy(sim, plan))
    assert m["success"]


def fleet_factory(fleet, **robots):
    cfg = read("lab")
    cfg["robots"] = dict({k: v for k, v in cfg["robots"].items() if k in ("start", "parking", "battery", "fault")}, fleet=fleet, **robots)
    return Factory(cfg)


@pytest.mark.skipif(not (_tool("popf") and _tool("validate")), reason="set FACTORYSIM_POPF and FACTORYSIM_VALIDATE to run")
def test_mixed_fleet_pddl_plan_uses_handovers_and_executes():
    from factory_sim import pddl
    F = fleet_factory([{"type": "tb3_tray", "count": 1}])
    orders = [{"id": 0, "product": "PR1", "arrival": 0, "due": 1e9}]
    sim = Sim(F, orders=orders, time_source="nominal")
    d = tempfile.mkdtemp()
    dom, prob = pddl.write(sim, os.path.join(d, "domain.pddl"), os.path.join(d, "problem.pddl"))
    out = subprocess.run([_tool("popf"), dom, prob], capture_output=True, text=True, timeout=300).stdout
    plan = "\n".join(l for l in out.split(";;;; Solution Found")[-1].splitlines() if l[:1].isdigit())
    assert "handover-" in plan
    open(os.path.join(d, "plan.txt"), "w").write(plan)
    val = subprocess.run([_tool("validate"), "-t", "0.001", dom, prob, os.path.join(d, "plan.txt")], capture_output=True, text=True).stdout
    assert "Plan valid" in val
    assert Sim(F, orders=orders, time_source="nominal").run(pddl.PlanPolicy(sim, plan))["success"]


def test_robot_types_and_fleets():
    from factory_sim import pddl, robots
    assert Factory("lab").homogeneous and pddl.domain(Factory("lab")) == pddl.DOMAIN   # 1.0 fleets keep the 1.0 domain
    F = fleet_factory([{"type": "tb3_manipulator", "count": 2}, {"type": "tb3_tray", "count": 1, "speed": 0.2}])
    assert [x["type"] for x in F.fleet] == ["tb3_manipulator", "tb3_manipulator", "tb3_tray"] and F.fleet[2]["speed"] == 0.2
    assert F.fleet[0]["fault"] == read("lab")["robots"]["fault"] or not F.features["robot_faults"]
    assert F.handling == ["arm", "handler"] and "handler" in F.operators and not F.homogeneous
    dom = pddl.domain(F)
    assert dom.count("(") == dom.count(")") and "handover-unload-sink" in dom and "(self-loading ?r)" in dom
    assert "handover-" not in pddl.domain(fleet_factory([{"type": "tb3_manipulator", "count": 2}, {"type": "tb3_manipulator", "count": 1, "speed": 0.1}]))
    with pytest.raises(ValueError):
        fleet_factory([{"type": "roller_amr", "count": 1}])
    with pytest.raises(ValueError):
        fleet_factory([{"type": "no_such_robot", "count": 1}])
    assert "handover" in robots.describe("tb3_tray")
    for fleet in ([{"type": "tb3_tray", "count": 3}], [{"type": "tb3_burger_tray", "count": 2}, {"type": "tb3_manipulator", "count": 1}]):
        sim = Sim(fleet_factory(fleet), seed=2)
        m = sim.run(baselines.make("nearest"))
        invariants(sim)
        assert m["success"] and all(v == c for v, c in zip(sim.ops_free.values(), sim.ops_cap.values()))


def test_observation_and_actions_fixed_across_fleets():
    import gymnasium as gym
    from factory_sim import scenarios
    shapes = set()
    for inst in ["lab", "lab_mixed_fleet", scenarios.make_instance("tray_fleet", "S", "test", 0), scenarios.make_instance("mixed_fleet", "L", "test", 0)]:
        e = gym.make("FactorySim-v0", instance=inst)
        o, _ = e.reset(seed=0)
        shapes.add((o.shape, e.action_space.n))
    assert len(shapes) == 1


def test_handler_robot_waits_for_a_free_handler():
    F = fleet_factory([{"type": "tb3_tray", "count": 3}], handlers=1)
    sim = Sim(F, seed=0, time_source="nominal")
    seen = set()
    while not sim.done and sim.failed is None:
        a = sim.next_decision()
        if a is None:
            break
        sim.act(a, baselines.make("nearest")(sim, a, sim.legal(a)))
        seen |= {r.phase for r in sim.robots.values()}
        assert sum(bool(r.hand) for r in sim.robots.values()) <= 1
    assert sim.done and "handler_wait" in seen


def test_grasp_failures_are_retried():
    cfg = read("lab")
    cfg["robots"]["grasp_success"] = 0.6
    m = Sim(Factory(cfg), seed=3).run(baselines.make("edd"))
    assert m["success"] and m["grasp_failures"] > 0
    cfg["robots"]["grasp_success"] = 1.0
    cfg["robots"]["place_success"] = 1.0
    assert Sim(Factory(cfg), seed=3).run(baselines.make("edd"))["grasp_failures"] == 0


def test_external_backend_reports_failed_pick():
    sim = Sim(small(robots=1), orders=[{"id": 0, "product": "PR1", "arrival": 0, "due": 1e9}], time_source="external")
    pol, failed = baselines.make("fifo"), False
    while not sim.done and sim.failed is None:
        while True:
            a = sim.next_decision()
            if a is None:
                break
            sim.act(a, pol(sim, a, sim.legal(a)))
        aid, (kind, _, t0, nom) = min(sim.external_pending.items(), key=lambda kv: kv[1][2] + kv[1][3])
        ok = not (kind == "load" and not failed)
        failed = failed or not ok
        sim.complete(aid, t0 + nom, ok=ok)
    assert sim.done and sim.metrics()["grasp_failures"] == 1


def test_world_map_and_arm_reach():
    from factory_sim.ros import robot, world
    d = tempfile.mkdtemp()
    out = world.write("lab", d, jobs=[(0, "PR1"), (1, "PR2")])
    lay = __import__("json").load(open(out["layout"]))
    assert all(c >= 0.24 for c in lay["clearance"].values())
    assert {"part_0", "part_1"} <= set(lay["parts"].values()) and lay["conveyors"]
    assert open(os.path.join(d, "factory.pgm"), "rb").read(2) == b"P5"
    w = open(out["world"]).read()
    assert all("robot{}_arm".format(i + 1) in w for i in range(3)) and "libgazebo_ros_state.so" in w
    media = os.path.join(d, "media")
    assert os.path.exists(os.path.join(media, "materials", "scripts", "factorysim.material")) and len(os.listdir(os.path.join(media, "meshes"))) > 50
    assert os.path.exists(os.path.join(media, "materials", "textures", "fs_sign_MC1.png")) and "FS/" in w
    assert "intel_realsense" not in robot.robot_sdf("<model><sensor name=\"intel_realsense_r200_depth\"></sensor></model>")
    for s, info in lay["stations"].items():
        node = Factory(read("lab")).stations[s]["node"]
        for lane in ("in", "out"):
            if info.get(lane) and lay["docks"][node][2] != 0.0 or s in ("kitting", "MC1"):
                p = robot.to_base(lay["docks"][node], info[lane][0]) if info.get(lane) else None
                assert p is None or robot.ik(p[0], p[1], p[2] + robot.GRASP_Z) is not None, (s, lane)


def realistic(**features):
    cfg = read("lab_realistic")
    cfg["features"].update(features)
    return cfg


def test_auto_charging_keeps_batteries_alive_and_chargers_exclusive():
    cfg = realistic(robot_faults=False, breakdowns=False)
    cfg["robots"]["battery"].update({"per_second": 0.03, "low": 30})
    sim = Sim(Factory(cfg), seed=1, record=True)
    m = sim.run(baselines.make("edd"))
    assert m["success"] and sum(m["charges"].values()) >= 3 and min(m["min_battery"].values()) > 0
    for snap in sim.trace:
        charging = [r for r, v in snap["robots"].items() if v[4] == "charging"]
        assert len(charging) <= len(sim.F.chargers)


def test_reconfiguration_lets_a_machine_take_over_a_role():
    sim = Sim(Factory(realistic(breakdowns=False, robot_faults=False, materials=False)), seed=0, time_source="nominal")
    assert ("setup", "MC1", "as_MC3") in sim.legal("dispatcher")
    sim.act("dispatcher", ("setup", "MC1", "as_MC3"))
    assert sim.state("MC1") == "setup" and "MC1" in sim.cands(["MC3"]) and "MC1" not in sim.cands(["MC1"])
    while sim.stations["MC1"].setup is not None:
        sim._pop()
    assert sim.stations["MC1"].config == "as_MC3" and sim.stats["setups"] == 1
    assert sim.F.proc_model("MC1", "PR2", "as_MC3") == {"lognormal": [360, 0.12]}


def test_single_technician_serialises_repairs_services_and_setups():
    sim = Sim(Factory(realistic(robot_faults=False, materials=False)), seed=2)
    peak = [0]
    orig = sim._assign_tech

    def watch():
        orig()
        busy = sum(1 for s in sim.stations.values() if s.service or (s.setup is not None and s.wait is None) or (s.down and s.wait is None))
        peak[0] = max(peak[0], busy)
    sim._assign_tech = watch
    m = sim.run(baselines.make("edd"))
    active = sum(1 for s in sim.stations.values() if s.service or (s.setup is not None and s.wait is None) or (s.down and s.wait is None))
    assert m["success"] and peak[0] <= 1 and sim.ops_free["technician"] == 1 - active


def test_operator_breaks_and_material_stock():
    cfg = realistic(breakdowns=False, robot_faults=False, battery=False, jams=False)
    cfg["operator_breaks"] = {"assembler": {"every": 600, "duration": 300, "n": 2}}
    cfg["materials"] = {"stock": {"PR1": 1, "PR2": 1, "PR3": 1, "PR4": 1}, "replenish": {"every": 400, "qty": 1}}
    sim = Sim(Factory(cfg), seed=0)
    releasable = {a[1] for a in sim.legal("dispatcher") if a[0] == "release"}
    assert all(sim.stock[sim.jobs[j].product] > 0 for j in releasable)
    m = sim.run(baselines.make("edd"))
    assert m["success"] and m["replenishments"] >= 1 and sim.ops_cap["assembler"] in (0, 2)


def test_codec_round_trips_setup_actions_and_env_runs():
    env = FactoryEnv(instance="lab_realistic")
    obs, info = env.reset(seed=0)
    pol, inner = baselines.make("edd"), env.unwrapped
    for _ in range(20000):
        sim = inner.sim
        obs, r, term, trunc, info = env.step(inner.codec.encode(pol(sim, info["agent"], sim.legal(info["agent"]))))
        assert env.observation_space.contains(obs)
        if term or trunc:
            break
    assert term and inner.sim.done
    sim = Sim(Factory(realistic()), seed=0)
    from factory_sim.env import Codec
    c = Codec(sim, 16)
    for a in [("setup", "MC2", "as_MC4"), ("wait",), ("charge",), ("release", 3, 0)]:
        assert c.decode(c.encode(a)) == a


def test_human_hold_and_release():
    sim = Sim(small(robots=2), orders=[{"id": i, "product": "PR1", "arrival": 0, "due": 1e9} for i in range(2)], time_source="nominal")
    sim.hold("r1")
    assert sim.robots["r1"].manual and sim.legal("r1") == [("wait",)]
    pol = baselines.make("edd")
    for _ in range(30):
        a = sim.next_decision()
        if a is None:
            break
        assert a != "r1"
        sim.act(a, pol(sim, a, sim.legal(a)))
    sim.release("r1", node="k5", distance=4.0)
    assert not sim.robots["r1"].manual and sim.robots["r1"].node == "k5" and sim.robots["r1"].dist >= 4.0
    assert sim.run(pol)["success"]


def test_default_lab_has_no_people_and_arm_robots():
    """FactorySim 2.0: the default lab's arm robots pick, carry and place; no operators, technician or handlers, and the
    Gazebo world draws no human figures. Workforce configs keep their people."""
    from factory_sim.ros import world
    F = Factory(read("lab"))
    assert not F.operators and F.technician is None and not any(st.get("operator") for st in F.stations.values())
    assert set(F.handling) == {"arm"}
    m = Sim(F, seed=0).run(baselines.make("edd"))
    assert m["success"]
    sdf, lay, _, _ = world.sdf(F)
    assert not lay["technicians"] and not lay["handlers"]
    assert "op_" not in "".join(n for n in lay["stations"] if "operator" in lay["stations"][n])
    for name in ("lab_realistic", "lab_v1"):     # the technician works only with `maintenance` on (lab_realistic)
        G = Factory(read(name))
        assert G.operators.get("kitter") == 1 and G.stations["asm1"]["operator"] == "assembler"
        assert G.technician == ("technician" if name == "lab_realistic" else None)


def test_suites_people_free_by_default_and_1x_reproducible():
    """Suite 2.0 (default) leaves out the families that need people; a people family generated on the default lab gets
    the workforce; the 1.x suites are built on lab_v1 with its workforce."""
    s2 = scenarios.SUITES["2.0"]
    assert s2["base"] == "lab" and not set(s2["families"]) & set(scenarios.PEOPLE_FAMILIES) and "disturbed" in s2["families"]
    assert all(scenarios.SUITES[v]["base"] == "lab_v1" for v in ("1.0", "1.1", "v1", "v2"))
    base = scenarios.make_instance("base", "S", "test", 0)
    assert not base["config"].get("operators")
    breaks = scenarios.make_instance("breaks", "S", "test", 0)
    assert breaks["config"]["operators"]["assembler"] == 2 and breaks["config"]["features"]["breaks"]
    old = scenarios.make_instance("base", "S", "test", 0, base="lab_v1")
    assert old["config"]["operators"] == {"kitter": 1, "assembler": 2, "technician": 1}
    m = build_sim(scenarios.make_instance("disturbed", "S", "test", 0), seed=0).run(baselines.make("edd"))
    assert m["success"] or m["failure"] == "horizon"


def test_observation_size_same_with_and_without_people():
    import gymnasium as gym
    shapes = {gym.make("FactorySim-v0", instance=i).reset(seed=0)[0].shape for i in ("lab", "lab_v1", "lab_realistic", "lab_mixed_fleet")}
    assert len(shapes) == 1
