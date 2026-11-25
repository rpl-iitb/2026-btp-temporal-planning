"""Tests for human play (window-free: handle_key / update)."""
"""This file is UN-VERIFIED"""
import json
import os

import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
pytest.importorskip("pygame")

from overcooked_kitchen import grid
from overcooked_kitchen.layout import NORTH, SOUTH, EAST, WEST
from overcooked_kitchen.play import Play
from overcooked_kitchen.tasks import WAIT

KEY = {NORTH: "up", SOUTH: "down", EAST: "right", WEST: "left"}


def _path_keys(play, cook, dst):
    w = play.sim.world
    i = play.sim.cooks.index(cook)
    p = w.players[i]
    pose = (p.pos[0], p.pos[1], tuple(p.facing))
    keys = []
    while pose != dst:
        d = w.lay.next_step(pose, dst)
        keys.append(KEY[d])
        n = (pose[0] + d[0], pose[1] + d[1])
        pose = (n[0], n[1], d) if w.lay.is_floor(n) else (pose[0], pose[1], d)
    return keys


def test_drive_keys_pick_onion():
    play = Play("k01_one_soup")
    play.handle_key("1")
    play.handle_key("v")
    assert play.sim.mode["c1"] == "drive"
    w = play.sim.world
    crate = next(c for c, k in w.kind.items() if k == "onion_crate")
    dst = next(q for q in w.lay.access_poses(crate) if q in w.lay.bfs_poses((4, 2, tuple(w.players[0].facing))))
    for k in _path_keys(play, "c1", dst):
        play.handle_key(k)
        play.update(0.25)
    play.handle_key("space")
    play.update(0.25)
    assert play.sim.world.players[0].held == ("raw", "onion")


def test_task_menu_lists_legal_tasks():
    play = Play("m01_three_cooks")
    play.handle_key("t")
    want = ["{} @ {}".format(t.action, t.station) for t in play.sim.legal("c1") if t != WAIT]
    assert want and play.menu("c1") == want
    play.handle_key("]")
    assert play.highlight("c1") == 1 % len(want)
    play.handle_key("return")
    assert "c1" in play.sim.running
    assert play.menu("c1") == []


def test_auto_teammates_finish_m01():
    play = Play("m01_three_cooks", cooks=[1], rate=1000.0)
    for _ in range(400):
        play.update(0.1)
        if play.over:
            break
    m = play.sim.metrics()
    assert m["failure"] != "deadlock" and m["failure"] != "stalled"
    assert m["served"] >= 1


def test_save_trajectory(tmp_path):
    path = tmp_path / "t.json"
    play = Play("k01_one_soup", rate=1000.0, seed=3)
    play.update(0.05)
    play.save(str(path))
    data = json.loads(path.read_text())
    assert data["scenario"] == "k01_one_soup" and data["seed"] == 3
    assert len(data["steps"]) == play.sim.t > 0
    step = data["steps"][0]
    assert set(step["actions"]) == {"c1"} and step["actions"]["c1"] in grid.ACTION_NAMES
    assert step["modes"] == {"c1": "auto"}
    assert data["log"] == json.loads(json.dumps(play.sim.log, default=str))


def test_pause_speed_restart_quit():
    play = Play("k01_one_soup", rate=4.0)
    play.handle_key("p")
    play.update(5)
    assert play.sim.t == 0
    play.handle_key("p")
    play.handle_key("+")
    assert play.rate == 8.0
    play.update(1)
    assert play.sim.t == 8
    play.handle_key("r")
    assert play.sim.t == 0 and play.steps == []
    play.handle_key("q")
    assert play.quit


def test_main_runs_headless(monkeypatch):
    import pygame
    from overcooked_kitchen import play as mod
    events = [[], [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_q, unicode="q")]]
    monkeypatch.setattr(pygame.event, "get", lambda: events.pop(0) if events else [])
    mod.main(["--scenario", "k01_one_soup", "--cooks", "1"])


def test_menu_highlight_follows_label_not_index():
    play = Play("m01_three_cooks")
    play.handle_key("t")
    play.handle_key("]")
    label = play.menu("c1")[play.highlight("c1")]
    other = next(c for c in play.sim.cooks if c != "c1")
    task = next(t for t in play.sim.legal(other) if t != WAIT and "{} @ {}".format(t.action, t.station) == label)
    play.sim.act(other, task)
    play.sim.tick()
    labels = play.menu("c1")
    assert label not in labels or labels.index(label) == play.highlight("c1")
    before = dict(play.sim.running)
    play.handle_key("return")
    if label in labels:
        assert play.sim.running["c1"].task.station == label.split(" @ ")[1]
    else:
        assert "c1" not in play.sim.running and play.sim.running == before
        assert play.menu("c1") == labels
        assert 0 <= play.highlight("c1") < len(labels) or not labels


def test_menu_highlight_moves_to_remaining_entry():
    play = Play("m01_three_cooks")
    play.handle_key("t")
    play.cursor["c1"] = "gone @ nowhere"
    play._at["c1"] = 10 ** 6
    k = play.highlight("c1")
    assert k == len(play.menu("c1")) - 1
    assert play.cursor["c1"] == play.menu("c1")[k]


def test_play_keeps_default_stuck_detection():
    assert Play("k01_one_soup").sim.stuck_steps == 50


def test_main_clamps_cooks_and_hints_missing_pygame(monkeypatch, capsys):
    import builtins
    import pygame
    from overcooked_kitchen import play as mod
    events = [[pygame.event.Event(pygame.KEYDOWN, key=pygame.K_q, unicode="q")]]
    monkeypatch.setattr(pygame.event, "get", lambda: events.pop(0) if events else [])
    mod.main(["--scenario", "k01_one_soup", "--cooks", "3"])
    assert "has 1 cook(s); you drive 1" in capsys.readouterr().out
    real = builtins.__import__

    def no_pygame(name, *a, **kw):
        if name == "pygame":
            raise ImportError(name)
        return real(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", no_pygame)
    with pytest.raises(SystemExit, match=r"pip install -e '.\[play\]'"):
        mod.main(["--scenario", "k01_one_soup"])


def test_play_rejects_unknown_cook():
    with pytest.raises(ValueError):
        Play("k01_one_soup", cooks=["c5"])
