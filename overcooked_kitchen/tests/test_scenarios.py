"""Tests for scenario loading and bundled examples."""
"""This file is UN-VERIFIED"""
import pytest

from overcooked_kitchen.scenarios import (load_scenario, scenario_orders, make_world, example,
                                          examples)

PLAYERS = {"k01_one_soup": 1, "k02_two_dishes": 1, "k03_service_cleanup": 1, "k04_burn_risk": 1,
           "m01_three_cooks": 3, "m02_four_cooks": 4}


def test_load_all_examples():
    assert examples() == sorted(PLAYERS)
    for name, n in PLAYERS.items():
        sc = load_scenario(example(name))
        assert "_sha256" in sc and sc["_path"].endswith(name + ".yaml")
        assert len(make_world(sc).players) == n


def test_orders_expand_count():
    sc = {"scenario_id": "x", "grid": ["XPX", "B1S", "XDX"],
          "orders": [{"recipe": {"onion": 1}, "count": 2, "deadline": 50}, {"recipe": {"tomato": 2}}]}
    orders = scenario_orders(load_scenario(sc))
    assert [o["id"] for o in orders] == ["o1", "o2", "o3"]
    assert orders[0]["comp"] == (1, 0, 0) and orders[0]["deadline"] == 50
    assert orders[2]["comp"] == (0, 2, 0) and orders[2]["deadline"] is None


def test_unknown_ingredient_rejected():
    with pytest.raises(ValueError):
        load_scenario({"scenario_id": "x", "grid": [], "orders": [{"recipe": {"kale": 1}}]})


def test_v3_scenario_keeps_id_and_counters():
    sc = load_scenario(example("m01_three_cooks"))
    assert sc["scenario_id"] == "v3_test_m3_000"
    assert isinstance(sc["counters"], list)
    assert len(scenario_orders(sc)) == 6
    sc4 = load_scenario(example("m02_four_cooks"))
    assert sc4["scenario_id"] == "v3_test_l4_001"
