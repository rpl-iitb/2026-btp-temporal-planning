"""Tests for grid layout analysis."""
"""This file is UN-VERIFIED"""
import collections

from overcooked_kitchen.layout import Layout, NORTH, SOUTH

K02 = ["XOTCXPXPX",
       "B       S",
       "X   1   X",
       "B       R",
       "XXDXXXWXX"]


def test_bfs_turn_costs_one_step():
    lay = Layout(["XXXXX", "X 1 X", "XXPXX"])
    assert lay.bfs_poses((2, 1, NORTH))[(2, 1, SOUTH)] == 1


def test_access_poses_face_station():
    lay = Layout(["XXXXX", "X 1 X", "XXPXX"])
    assert lay.access_poses((2, 2)) == [(2, 1, SOUTH)]


def test_stations_v2_alphabet():
    kinds = collections.Counter(s.kind for s in Layout(K02).stations(counter_policy="none"))
    assert dict(kinds) == {"onion_crate": 1, "tomato_crate": 1, "carrot_crate": 1, "pot": 2,
                           "board": 2, "serving": 1, "dropoff": 1, "rack": 1, "sink": 1}


def test_next_step_avoids_blocked():
    lay = Layout(["XXXXX", "X   X", "XXXXX"])
    assert lay.next_step((1, 1, SOUTH), (3, 1, SOUTH)) is not None
    assert lay.next_step((1, 1, SOUTH), (3, 1, SOUTH), blocked=frozenset({(2, 1)})) is None
