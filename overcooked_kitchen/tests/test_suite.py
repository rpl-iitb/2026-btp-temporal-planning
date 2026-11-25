"""Tests for suites: import of v2 / v3 instances, generation with certificates, bench tables, the CLI."""
"""This file is UN-VERIFIED"""
import json
import os
import shutil
import subprocess
import sys
import zlib

import pytest
import yaml

from overcooked_kitchen import cli, generate, pddl, suite
from overcooked_kitchen.baselines import make
from overcooked_kitchen.core import Sim
from overcooked_kitchen.layout import Layout
from overcooked_kitchen.pddlkit import TemporalPlan, TimedAction, planners
from overcooked_kitchen.scenarios import example, load_scenario

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
V2 = os.environ.get("OVERCOOKED_KITCHEN_V2_SOURCE", "")     # folders the shipped suite was imported from (optional)
V3 = os.environ.get("OVERCOOKED_KITCHEN_V3_SOURCE", "")
RECIPES = [{"dish": "tomato_soup", "recipe": {"tomato": 3}}, {"dish": "onion_soup", "recipe": {"onion": 2}},
           {"dish": "carrot_soup", "recipe": {"carrot": 2}},
           {"dish": "tomato_onion_soup", "recipe": {"tomato": 2, "onion": 1}},
           {"dish": "veggie_stew", "recipe": {"onion": 1, "tomato": 1, "carrot": 1}},
           {"dish": "carrot_onion_stew", "recipe": {"carrot": 2, "onion": 1}}]
TIER_S = {"name": "s", "kitchen": {"width": [7, 9], "height": [5, 6], "core_pots": 1, "distractor_board": [0, 1],
                                   "distractor_pot": [0, 1]}, "orders": [1, 2], "burn": [15, 30]}
TIER_M3 = {"name": "m3", "cooks": 3, "certificate": "dispatcher",
           "kitchen": {"width": [12, 14], "height": [7, 8], "crates": [1, 2], "board": [3, 4], "pot": [3, 4],
                       "rack": 1, "dropoff": 1, "sink": [1, 2], "serve": [1, 2]},
           "orders": [5, 7], "plates_clean": [2, 3], "plates_dirty": [0, 1], "dirty_pots": [0, 1], "burn": [12, 20]}


@pytest.fixture
def tools():
    return planners.available("popf") and planners.available("validate")


def _copy_sources(tmp_path):
    """3 v2_test and 3 v3_test instances in the v2 / v3 folder layouts."""
    if not (os.path.isdir(V2) and os.path.isdir(V3)):
        pytest.skip("set OVERCOOKED_KITCHEN_V2_SOURCE / OVERCOOKED_KITCHEN_V3_SOURCE to the v2 / v3 source folders")
    v2, v3 = tmp_path / "v2", tmp_path / "v3"
    src2 = os.path.join(V2, "scenarios", "v2_test")
    dst2 = v2 / "scenarios" / "v2_test"
    dst2.mkdir(parents=True)
    ids2 = sorted(f[:-5] for f in os.listdir(src2) if f.endswith(".yaml"))[:3]
    for sid in ids2:
        for ext in (".yaml", ".certificate.plan"):
            shutil.copy(os.path.join(src2, sid + ext), str(dst2))
    src3 = os.path.join(V3, "generated_pddl", "v3_test")
    ids3 = sorted(d for d in os.listdir(src3) if os.path.isdir(os.path.join(src3, d)))[:3]
    for sid in ids3:
        d = v3 / "generated_pddl" / "v3_test" / sid
        d.mkdir(parents=True)
        for f in ("scenario.yaml", "certificate.plan"):
            shutil.copy(os.path.join(src3, sid, f), str(d))
    return str(v2), str(v3), ids2, ids3


def test_import_keeps_ids_and_certificates(tmp_path):
    v2, v3, ids2, ids3 = _copy_sources(tmp_path)
    out = str(tmp_path / "suite")
    rows = suite.import_existing(out, v2, v3)
    assert sorted(r["name"] for r in rows) == sorted(ids2 + ids3)
    with open(os.path.join(out, "index.json")) as f:
        index = json.load(f)["instances"]
    assert sorted(r["name"] for r in index) == sorted(ids2 + ids3)
    for r in index:
        src = "v2" if r["name"] in ids2 else "v3"
        assert r["source"] == src and r["split"] == src + "_test"
        assert r["tier"] == r["name"][len(r["split"]) + 1:].rsplit("_", 1)[0]
        with open(os.path.join(out, r["path"])) as f:
            sc = yaml.safe_load(f)
        assert sc["scenario_id"] == r["name"]
        assert r["cooks"] == (1 if src == "v2" else sum(ch.isdigit() for row in sc["grid"] for ch in row)) >= 1
        assert r["orders"] == len(sc["orders"])
        orig = (os.path.join(V2, "scenarios", "v2_test", r["name"] + ".certificate.plan") if src == "v2"
                else os.path.join(V3, "generated_pddl", "v3_test", r["name"], "certificate.plan"))
        with open(os.path.join(out, r["cert_plan"])) as f:
            text = f.read()
        source = TemporalPlan.parse(open(orig).read())
        renamed = suite.rename_stations(source, sc)
        assert text == open(orig).read() if renamed is source else TemporalPlan.parse(text) == renamed
        assert r["cert_source"] == src
        if planners.available("validate"):
            assert r["cert_kind"] == "val" and r["cert_valid"] is True and r["cert_note"] is None
        else:
            assert r["cert_kind"] == "plan" and r["cert_valid"] is None
        assert r["plan_makespan"] == pytest.approx(TemporalPlan.parse(text).makespan, abs=1e-3)
        assert r["cert_makespan"] < r["plan_makespan"]
    assert [r["name"] for r in suite.iter_suite(out, "v3_test")] == sorted(ids3)
    assert {r["tier"] for r in index} >= {"l", "l4"}
    # by hand from the plan files: last deliver 390.137 + 1 (v2), 324.101 + 1 (v3); last diner-eats ends 403.138 /
    # 337.102 (v2_test_l_000 / v3_test_l4_001)
    by = {r["name"]: r for r in index}
    assert (by["v2_test_l_000"]["cert_makespan"], by["v2_test_l_000"]["plan_makespan"]) == (391.137, 403.138)
    assert (by["v3_test_l4_001"]["cert_makespan"], by["v3_test_l4_001"]["plan_makespan"]) == (325.101, 337.102)


def test_generate_small(tmp_path, tools):
    config = {"seed": 31, "recipes": RECIPES, "certificate_timeout": 5,
              "splits": [{"dataset": "g_test", "count": 2, "tiers": [TIER_S, TIER_M3]}]}
    out = str(tmp_path / "gen")
    rows = generate.generate(out, config, ["g_test"])
    assert sorted(r["tier"] for r in rows) == ["m3", "m3", "s", "s"]
    assert [r["name"] for r in suite.iter_suite(out, "g_test")] == sorted(r["name"] for r in rows)
    for r in rows:
        sc = yaml.safe_load(open(os.path.join(out, r["path"])))
        assert sc["scenario_id"] == r["name"] and r["source"] == "generated"
        assert r["cooks"] == (3 if r["tier"] == "m3" else 1)
        assert sum(ch.isdigit() for row in sc["grid"] for ch in row) == r["cooks"]
        m = Sim(sc, clock="sampled", noise=None).run(make("dispatcher"))
        assert m["success"] and m["burnt"] == 0 and m["makespan"] == r["dispatcher_makespan"]
        assert r["cert_makespan"] <= r["dispatcher_makespan"]
        assert r["cert_source"] in ("dispatcher", "popf")
        if r["tier"] == "s" and tools:
            assert r["popf_status"] in ("valid", "timeout", "unsolvable", "error", "invalid")
        if r["popf_status"] == "valid":
            assert r["popf_makespan"] >= r["cert_makespan"] - 1e-6
        if r["cert_kind"] == "val":
            assert r["cert_source"] == "popf"
            plan = TemporalPlan.parse(open(os.path.join(out, r["cert_plan"])).read())
            d, p = pddl.write(Sim(sc), str(tmp_path / r["name"]))
            assert pddl.validate(d, p, plan)
            ends = [a.start + a.duration for a in plan.actions if a.name == "deliver"]
            assert r["cert_makespan"] == pytest.approx(max(ends), abs=1e-3) and r["cert_makespan"] < m["makespan"]
            assert r["plan_makespan"] == pytest.approx(plan.makespan, abs=1e-3)
        else:
            assert r["cert_kind"] == "sim" and r["cert_source"] == "dispatcher" and r["cert_plan"] is None
            assert r["cert_makespan"] == m["makespan"] and r["plan_makespan"] is None
    if tools:
        assert any(r["popf_status"] == "valid" for r in rows if r["tier"] == "s")
    ds = json.load(open(os.path.join(out, "g_test", "dataset.json")))
    assert ds["requested"] == {"s": 2, "m3": 2} and ds["made"] == {"s": 2, "m3": 2}
    again = str(tmp_path / "again")
    generate.generate(again, dict(config, certificate="dispatcher"), ["g_test"])
    for r in rows:
        assert open(os.path.join(out, r["path"])).read() == open(os.path.join(again, r["path"])).read()


def test_generate_keeps_existing_splits(tmp_path, capsys):
    root = str(tmp_path / "suite")
    _tiny_suite(root)
    config = {"seed": 31, "recipes": RECIPES, "certificate": "dispatcher",
              "splits": [{"dataset": "t", "count": 1, "tiers": [TIER_S]},
                         {"dataset": "g", "count": 1, "tiers": [dict(TIER_S, max_tries_factor=1, burn=1)]}]}
    with pytest.raises(FileExistsError, match="--force"):
        generate.generate(root, config, ["t"])
    assert sorted(os.listdir(os.path.join(root, "t"))) == ["k01_one_soup.yaml", "k02_two_dishes.yaml"]
    path = str(tmp_path / "config.yaml")
    with open(path, "w") as f:
        yaml.safe_dump(config, f)
    with pytest.raises(SystemExit, match="already exist"):
        cli.main(["generate", "--config", path, "--out", root, "--splits", "t"])
    cli.main(["generate", "--config", path, "--out", root, "--splits", "g"])
    assert "warning: g tier s: made 0 of 1 requested" in capsys.readouterr().out
    assert {r["split"] for r in suite.read_index(root)} == {"t"}
    rows = generate.generate(root, config, ["t"], force=True)
    index = suite.read_index(root)
    assert [r["name"] for r in index] == [r["name"] for r in rows] == ["t_s_000"]


def _tiny_suite(root):
    rows = []
    for k, (name, tier) in enumerate((("k01_one_soup", "a"), ("k02_two_dishes", "b"))):
        os.makedirs(os.path.join(root, "t"), exist_ok=True)
        shutil.copy(example(name), os.path.join(root, "t", name + ".yaml"))
        m = Sim(example(name)).run(make("dispatcher"))
        rows.append({"name": name, "split": "t", "tier": tier, "source": "example", "cooks": 1, "orders": 1,
                     "cert_kind": "sim", "cert_source": "dispatcher", "cert_plan": None,
                     "cert_makespan": m["makespan"], "plan_makespan": None, "path": "t/{}.yaml".format(name)})
    suite.update_index(root, rows)
    return rows


def test_bench_table_has_tiers(tmp_path):
    root = str(tmp_path / "suite")
    _tiny_suite(root)
    out = str(tmp_path / "res" / "bench.jsonl")
    rows = suite.bench(root, "t", "dispatcher", seeds=2, cv_scale=0.0, out=out)
    assert len(rows) == 4
    assert all(tuple(r) == suite.ROW_KEYS for r in rows)
    assert all(r["success"] and r["makespan_over_cert"] == 1.0 for r in rows)
    assert [json.loads(line) for line in open(out)] == rows
    noisy = suite.bench(root, "t", "random", seeds=2, cv_scale=1.0)
    assert [r["seed"] for r in noisy] == [0, 1, 0, 1]
    assert suite.bench_seed("k02_two_dishes", 1) == zlib.crc32(b"k02_two_dishes") * 1000 + 1
    os.remove(os.path.join(root, "t", "k01_one_soup.yaml"))
    suite.update_index(root, [r for r in suite.read_index(root) if r["name"] != "k01_one_soup"], drop_splits=["t"])
    alone = suite.bench(root, "t", "random", seeds=2, cv_scale=1.0)
    assert alone == [dict(r, wall_s=a["wall_s"]) for r, a in zip(noisy[2:], alone)]
    txt = suite.table(rows + noisy)
    lines = txt.splitlines()
    assert lines[0].startswith("| tier |")
    assert [ln.split("|")[1].strip() for ln in lines[2:]] == ["a", "b", "all"]


def test_cli_run_k01():
    exe = os.path.join(os.path.dirname(sys.executable), "overcooked-kitchen")
    cmd = [exe] if os.access(exe, os.X_OK) else [sys.executable, "-m", "overcooked_kitchen.cli"]
    r = subprocess.run(cmd + ["run", "--scenario", "k01_one_soup"], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert '"success": true' in r.stdout


def test_cli_pddl_writes_domain_and_problem(tmp_path, capsys):
    cli.main(["pddl", "--scenario", "k01_one_soup", "--out", str(tmp_path)])
    assert "(define (domain" in (tmp_path / "domain.pddl").read_text()
    assert "(define (problem" in (tmp_path / "problem.pddl").read_text()


def test_cli_bench_prints_table(tmp_path, capsys):
    root = str(tmp_path / "suite")
    _tiny_suite(root)
    out = str(tmp_path / "b.jsonl")
    cli.main(["bench", "--suite", root, "--split", "t", "--policy", "dispatcher", "--seeds", "1", "--cv-scale", "0",
              "--out", out])
    txt = capsys.readouterr().out
    assert "| a |" in txt and "| b |" in txt
    assert len(open(out).read().splitlines()) == 2


def test_cli_run_exit_code_and_noise(capsys):
    assert cli.main(["run", "--scenario", "k01_one_soup", "--noise", "default", "--cv-scale", "2", "--seed", "4"]) == 0
    noisy = json.loads(capsys.readouterr().out)
    nominal = Sim("k01_one_soup").run(make("dispatcher"))
    assert noisy["success"] and noisy["makespan"] != nominal["makespan"]
    assert cli.main(["run", "--scenario", "k01_one_soup", "--policy", "random", "--seed", "0"]) == 1
    assert not json.loads(capsys.readouterr().out)["success"]


def test_generate_rejects_unknown_certificate(tmp_path):
    config = {"seed": 1, "recipes": RECIPES, "certificate": "popf",
              "splits": [{"dataset": "g", "count": 1, "tiers": [TIER_S]}]}
    with pytest.raises(ValueError, match="certificate must be one of both, dispatcher"):
        generate.generate(str(tmp_path / "x"), config)
    assert not os.path.exists(str(tmp_path / "x"))


def test_load_policy_adds_cwd_once(tmp_path, monkeypatch):
    (tmp_path / "pol_mod.py").write_text("from overcooked_kitchen.tasks import WAIT\n\ndef act(sim, cook, legal):\n"
                                        "    return legal[-1]\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "path", [p for p in sys.path if p != str(tmp_path)])
    for _ in range(3):
        assert callable(suite.load_policy("pol_mod:act"))
    assert sys.path.count(str(tmp_path)) == 1


def test_rename_stations_by_cell():
    """v3 numbered only the policy counters (ctr1..); this package numbers every counter."""
    sc = yaml.safe_load(open(os.path.join(ROOT, "benchmarks", "v1", "v3_test", "v3_test_l4_016.yaml")))
    lay = Layout(sc["grid"])
    theirs = {s.name: s.cell for s in lay.stations(sc["counters"])}
    ours = {s.cell: s.name for s in lay.stations("all")}
    plan = TemporalPlan([TimedAction(0.0, "put-raw", ("c1", "p6_6s", "ctr6", "onion"), 1.0),
                         TimedAction(1.0, "take-raw", ("c1", "p1_1n", "tcrate1", "tomato"), 1.0)])
    out = suite.rename_stations(plan, sc)
    assert out.actions[0].args[2] == ours[theirs["ctr6"]] != "ctr6"
    assert out.actions[1] == plan.actions[1]
    plain = TemporalPlan(plan.actions[1:])
    assert suite.rename_stations(plain, sc) is plain


@pytest.mark.slow
@pytest.mark.skipif(not planners.available("validate"), reason="VAL not found")
def test_shipped_val_certificates_are_val_valid(tmp_path):
    """Every certificate labelled "val" in benchmarks/v1 is VAL-valid against this package's domain and export."""
    root = os.path.join(ROOT, "benchmarks", "v1")
    rows = [r for r in suite.read_index(root) if r["cert_kind"] == "val"]
    assert len(rows) == 440
    bad = []
    for r in rows:
        sc = load_scenario(os.path.join(root, r["path"]))
        plan = TemporalPlan.parse(open(os.path.join(root, r["cert_plan"])).read())
        d, p = pddl.write(Sim(sc), str(tmp_path / r["name"]))
        if not pddl.validate(d, p, plan):
            bad.append(r["name"])
        assert r["cert_valid"] is True and r["cert_makespan"] == suite.delivery_makespan(plan)
    assert bad == []


def test_check_certificate_reports_invalid_plans(monkeypatch):
    sc = load_scenario(example("k01_one_soup"))
    ok, note = suite.replay(sc, TemporalPlan([]))
    assert ok is False and note.startswith("replay (sampled): failure stalled")
    if planners.available("validate"):
        ok, note = suite.check_certificate(sc, TemporalPlan([]))
        assert ok is False and note.startswith("VAL: ")
    monkeypatch.setenv("OVERCOOKED_KITCHEN_VAL", "/nonexistent/Validate")
    monkeypatch.setenv("PATH", "")
    assert suite.check_certificate(sc, TemporalPlan([])) == (None, "VAL not found")
