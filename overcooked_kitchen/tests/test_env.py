"""Tests: Gymnasium and PettingZoo environments."""
"""This file is UN-VERIFIED"""
import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from overcooked_kitchen.env import KitchenEnv, KitchenGridEnv, aec_env, build_sim, decode, encode_task, mask
from overcooked_kitchen.scenarios import examples
from overcooked_kitchen.tasks import WAIT


def test_gym_env_checker():
    check_env(KitchenGridEnv("k01_one_soup"), skip_render_check=True)
    check_env(KitchenEnv("m01_three_cooks"), skip_render_check=True)


def test_invalid_index_waits_with_penalty():
    env = KitchenEnv("k01_one_soup")
    _, info = env.reset(seed=0)
    bad = int(np.flatnonzero(info["action_mask"] == 0)[0])
    _, r, term, trunc, info = env.step(bad)
    assert info["invalid"] and info["applied"] == WAIT
    assert r == pytest.approx(-10.01)


def test_invalid_index_raises():
    env = KitchenEnv("k01_one_soup", invalid_action="raise")
    _, info = env.reset(seed=0)
    with pytest.raises(ValueError):
        env.step(int(np.flatnonzero(info["action_mask"] == 0)[0]))


def test_wait_applied_and_certain_stall_fails():
    env = KitchenEnv("k01_one_soup")
    env.reset(seed=0)
    _, r, term, trunc, info = env.step(0)
    assert trunc and not term and not info["invalid"] and info["applied"] == WAIT
    assert r <= -10.0


def test_applied_matches_decoded():
    env = KitchenEnv("k01_one_soup")
    _, info = env.reset(seed=0)
    i = int(np.flatnonzero(info["action_mask"])[1])
    expect = decode(env.sim, info["agent"], i)
    _, _, _, _, info = env.step(i)
    assert info["applied"] == expect and not info["invalid"]


def test_registered():
    assert isinstance(gym.make("OvercookedKitchen-v0").unwrapped, KitchenEnv)
    assert isinstance(gym.make("OvercookedKitchenGrid-v0").unwrapped, KitchenGridEnv)


def test_masked_random_episode_terminates():
    env = KitchenEnv("m01_three_cooks")
    obs, info = env.reset(seed=1)
    rng = np.random.default_rng(0)
    term = trunc = False
    for _ in range(20000):
        assert info["agent"] in ("c1", "c2", "c3")
        obs, r, term, trunc, info = env.step(rng.choice(np.flatnonzero(info["action_mask"])))
        assert env.observation_space.contains(obs)
        if term or trunc:
            break
    assert term or trunc
    assert "metrics" in info


def test_grid_env_steps():
    env = KitchenGridEnv("k01_one_soup")
    obs, info = env.reset(seed=0)
    obs, r, term, trunc, info = env.step(env.action_space.sample())
    assert r < 0 and not term
    assert env.observation_space.contains(obs)


@pytest.mark.filterwarnings("ignore:The old environment creation API:DeprecationWarning")
@pytest.mark.filterwarnings("ignore:We recommend agents to be named:UserWarning")
def test_pettingzoo_api():
    pytest.importorskip("pettingzoo")
    from pettingzoo.test import api_test
    api_test(aec_env(instance="m01_three_cooks"), num_cycles=200, verbose_progress=False)


def test_encode_decode_roundtrip():
    sim = build_sim("m01_three_cooks")
    for _ in range(40):
        cook = sim.next_decision()
        if cook is None:
            break
        m = mask(sim, cook)
        tasks = [t for t in sim.legal(cook) if t != WAIT]
        assert m.sum() == 1 + len(tasks)
        for t in tasks:
            i = encode_task(sim, t)
            assert m[i] == 1 and decode(sim, cook, i) == t
        assert decode(sim, cook, 0) == WAIT
        sim.act(cook, tasks[0] if tasks else WAIT)


def test_too_many_stations_rejected():
    with pytest.raises(ValueError):
        KitchenEnv("m01_three_cooks", max_stations=5)


def _random_episode(env, seed=None, steps=30):
    obs, info = env.reset(seed=seed)
    rng = np.random.default_rng(0)
    trace = [env.sim.rng.random()]
    for _ in range(steps):
        obs, r, term, trunc, info = env.step(rng.choice(np.flatnonzero(info["action_mask"])))
        trace.append((info["t"], round(r, 6)))
        if term or trunc:
            break
    return trace


def test_unseeded_resets_differ_and_seeded_reproduce():
    env = KitchenEnv("m01_three_cooks", noise="default")
    env.reset(seed=3)
    a, b = _random_episode(env), _random_episode(env)
    assert a != b
    c, d = _random_episode(env, seed=7), _random_episode(env, seed=7)
    assert c == d
    e = KitchenEnv("m01_three_cooks", noise="default")
    assert _random_episode(e, seed=7) == c


def test_defaults_fit_every_bundled_example():
    for name in examples():
        env = KitchenEnv(name)
        obs, info = env.reset(seed=0)
        obs, r, term, trunc, info = env.step(int(np.flatnonzero(info["action_mask"])[-1]))
        assert env.observation_space.contains(obs)
        g = KitchenGridEnv(name)
        g.reset(seed=0)
        assert g.observation_space.contains(g.step(g.action_space.sample())[0])


def test_step_after_end_raises():
    env = KitchenEnv("k01_one_soup")
    with pytest.raises(gym.error.ResetNeeded):
        env.step(0)
    env.reset(seed=0)
    _, _, term, trunc, _ = env.step(0)
    assert term or trunc
    with pytest.raises(gym.error.ResetNeeded):
        env.step(0)
    env.reset(seed=0)
    env.step(int(np.flatnonzero(env.action_masks())[-1]))


def test_render_rgb_array():
    env = KitchenEnv("k01_one_soup", render_mode="rgb_array")
    env.reset(seed=0)
    img = env.render()
    assert img.dtype == np.uint8 and img.ndim == 3 and img.shape[2] == 3
    assert KitchenEnv("k01_one_soup").render() is None
    with pytest.raises(ValueError):
        KitchenEnv("k01_one_soup", render_mode="human")
