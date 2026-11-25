"""Gymnasium and PettingZoo interfaces.

Action index layout (same for every agent, illegal ones masked):
  0 wait | 1..J pickup job j | deliver to station s (S) | move to parking node p (P) | charge | release job j with route r (J*R)
Reward: -(elapsed time)/time_scale - (new tardiness)/time_scale, and -fail_penalty on deadlock / stall / horizon.
"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .config import Factory
from .core import DISPATCHER, Sim

MAX_ROBOTS = 4    # robot slots in the flat observation (padded)
MAX_JOBS = 32      # job slots in the flat observation and the action codec: the same size for every benchmark instance
MAX_POOLS = 4     # people pools in the flat observation (padded; the people-free default lab has only the handler pool)
OBS_BOUND = 50.0   # observations are clipped to [-OBS_BOUND, OBS_BOUND]
WHERE = ("backlog", "queue", "machine", "out", "robot", "conveyor", "buffer", "done")


def build_sim(instance, seed=0, **kw):
    """instance: a built-in config name, a config path / dict, or a benchmark instance dict (config + orders + sim options)."""
    if isinstance(instance, dict) and "config" in instance:
        opts = dict(instance.get("sim", {}), **kw)
        return Sim(Factory(instance["config"]), orders=instance.get("orders"), seed=seed, **opts)
    return Sim(Factory(instance), seed=seed, **kw)


class Codec:
    """Maps between action tuples and integer indices for one factory and job budget."""

    def __init__(self, sim, max_jobs):
        F = sim.F
        self.J, self.stations, self.parking = max_jobs, list(F.stations), list(F.move_targets)
        self.R = max(len(p["routes"]) for p in F.products.values())
        self.S, self.P = len(self.stations), len(self.parking)
        self.o_del, self.o_mov = 1 + self.J, 1 + self.J + self.S
        self.o_chg, self.o_rel = self.o_mov + self.P, self.o_mov + self.P + 1
        self.setups = [(s, c) for s, st in F.stations.items() if len(st.get("configs", {})) > 1 for c in st["configs"]]
        self.o_set = self.o_rel + self.J * self.R
        self.n = self.o_set + len(self.setups)

    def encode(self, a):
        k = a[0]
        if k == "wait":
            return 0
        if k == "pickup":
            return 1 + a[1]
        if k == "deliver":
            return self.o_del + self.stations.index(a[1])
        if k == "move":
            return self.o_mov + self.parking.index(a[1])
        if k == "charge":
            return self.o_chg
        if k == "setup":
            return self.o_set + self.setups.index((a[1], a[2]))
        return self.o_rel + a[1] * self.R + a[2]

    def decode(self, i):
        i = int(i)
        if i == 0:
            return ("wait",)
        if i < self.o_del:
            return ("pickup", i - 1)
        if i < self.o_mov:
            return ("deliver", self.stations[i - self.o_del])
        if i < self.o_chg:
            return ("move", self.parking[i - self.o_mov])
        if i == self.o_chg:
            return ("charge",)
        if i >= self.o_set:
            return ("setup",) + tuple(self.setups[i - self.o_set])
        j = i - self.o_rel
        return ("release", j // self.R, j % self.R)

    def mask(self, legal):
        m = np.zeros(self.n, np.int8)
        for a in legal:
            i = self.encode(a)
            if i < self.n:
                m[i] = 1
        return m


def observe(sim, codec, agent, time_scale):
    F, t = sim.F, sim.t
    xs, ys = [p[0] for p in F.nodes.values()], [p[1] for p in F.nodes.values()]
    W, H = max(xs) or 1.0, max(ys) or 1.0
    agents = [DISPATCHER] + list(sim.robots)
    out = [t / time_scale, sim.completed / max(1, sim.n_target), len(sim.backlog) / max(1, sim.n_target)] + [1.0 if a == agent else 0.0 for a in agents]
    for n in codec.stations:
        s = sim.stations[n]
        rem = (sim.acts[s.aid][3] - t) / time_scale if (s.aid is not None and sim.acts[s.aid][3] is not None) else 0.0
        cfgs = list(s.spec.get("configs", {})) or [None]
        out += [float(s.cur is not None and not s.done), float(s.cur is not None and s.done), float(s.down), len(s.q) / max(1, s.spec["slots_in"] if s.spec["slots_in"] < 10 ** 5 else 10), len(s.out) / max(1, s.spec["slots_out"]), s.reserved / 4.0, rem]
        out += [cfgs.index(s.config) / max(1, len(cfgs) - 1) if s.config in cfgs else 0.0, float(s.setup is not None), float(s.service or s.wait == "service"), float(s.due_service is not None)]
    for n, r in sim.robots.items():
        x, y = F.nodes[r.dest]
        b = r.spec["battery"]
        out += [x / W, y / H, float(r.carry is not None), float(r.task is not None), max(0.0, r.t1 - t) / time_scale if r.phase == "travel" else 0.0, (r.battery / b["capacity"]) if b else 1.0]
        out += [float(r.fault), float(r.phase in ("charging", "charge_queue"))]
        out += [float(r.spec["handling"] == "handler"), r.spec["speed"] / 0.3, float(r.phase == "handler_wait")]   # robot type
    out += [0.0] * (12 * max(0, MAX_ROBOTS - len(sim.robots)))
    out += [min(1.0, sim.stock.get(p, 10) / 10.0) if sim.stock is not None else 1.0 for p in F.products]
    pools = sorted(set(F.operators) | {F.handler_pool})[:MAX_POOLS]    # padded: the same size with or without people
    out += [sim.ops_free[o] / max(1, F.operators[o]) if o in F.operators else 0.0 for o in pools] + [0.0] * (MAX_POOLS - len(pools))
    prods = list(F.products)
    for jid in range(codec.J):
        j = sim.jobs.get(jid)
        if j is None or j.where in ("future", "scrapped"):
            out += [0.0] * (2 + len(WHERE) + len(prods) + 4)
            continue
        node = sim.stations[j.at].spec["node"] if j.at in sim.stations else (sim.robots[j.at].dest if j.at in sim.robots else None)
        x, y = F.nodes[node] if node else (0.0, 0.0)
        out += [1.0, float(j.claim is not None)] + [1.0 if j.where == w else 0.0 for w in WHERE] + [1.0 if j.product == p else 0.0 for p in prods]
        out += [j.k / max(1, len(j.route)) if j.route else 0.0, (j.due - t) / time_scale if j.due < float("inf") else 10.0, x / W, y / H]
    return np.clip(np.asarray(out, np.float32), -OBS_BOUND, OBS_BOUND)


def graph(sim):
    """Graph view for GNN policies: nodes (stations, robots, active jobs) with features, typed directed edges."""
    F = sim.F
    names, types, feats = [], [], []
    for n, s in sim.stations.items():
        names.append(n), types.append("station"), feats.append([*F.nodes[s.spec["node"]], float(s.cur is not None), float(s.done), float(s.down), len(s.q), len(s.out)])
    for n, r in sim.robots.items():
        names.append(n), types.append("robot"), feats.append([*F.nodes[r.dest], float(r.carry is not None), float(r.task is not None), 0.0, 0.0, 0.0])
    act = [j for j in sim.jobs.values() if j.where not in ("future", "done", "scrapped")]
    for j in act:
        names.append("job{}".format(j.id)), types.append("job"), feats.append([j.k, len(j.route or []), (j.due - sim.t), float(j.where == "backlog"), float(j.claim is not None), 0.0, 0.0])
    idx = {n: i for i, n in enumerate(names)}
    edges, etype = [], []
    for j in act:
        if j.at in idx:
            edges.append((idx["job{}".format(j.id)], idx[j.at])), etype.append("at")
        if j.route:
            for s in sim.targets(j):
                edges.append((idx["job{}".format(j.id)], idx[s])), etype.append("next")
    for (a, b) in F.transfers:
        edges.append((idx[a], idx[b])), etype.append("conveyor")
    return {"names": names, "types": types, "x": np.asarray(feats, np.float32), "edges": np.asarray(edges, np.int64).reshape(-1, 2).T, "edge_types": etype}


class FactoryEnv(gym.Env):
    """Single-controller Gymnasium env: each step decides for the agent named in info['agent']."""
    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(self, instance="lab", max_jobs=None, time_scale=1000.0, fail_penalty=10.0, render_mode=None, invalid_action="wait", invalid_penalty=0.01, **sim_kw):
        self.instance, self.time_scale, self.fail_penalty, self.render_mode, self.sim_kw = instance, time_scale, fail_penalty, render_mode, sim_kw
        self.invalid_action, self.invalid_penalty = invalid_action, invalid_penalty
        sim = build_sim(instance, 0, **sim_kw)
        self.codec = Codec(sim, max_jobs or max(MAX_JOBS, 2 * sim.n_target))
        self.action_space = spaces.Discrete(self.codec.n)
        n = len(observe(sim, self.codec, DISPATCHER, time_scale))
        self.observation_space = spaces.Box(-OBS_BOUND, OBS_BOUND, (n,), np.float32)
        self.sim, self.agent = None, None

    def _info(self):
        legal = self.sim.legal(self.agent) if self.agent else [("wait",)]
        return {"agent": self.agent, "action_mask": self.codec.mask(legal), "t": self.sim.t}

    def action_masks(self):
        """Boolean mask of legal actions now (the sb3-contrib MaskablePPO convention)."""
        return self._info()["action_mask"].astype(bool)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.sim = build_sim(self.instance, seed if seed is not None else 0, **self.sim_kw)
        self.agent = self.sim.next_decision()
        return observe(self.sim, self.codec, self.agent, self.time_scale), self._info()

    def step(self, action):
        sim = self.sim
        t0, tard0 = sim.t, self._tardiness()
        a, bad = self.codec.decode(action), False
        if a not in sim.legal(self.agent):
            if self.invalid_action == "raise":
                raise ValueError("illegal action {} for {}".format(a, self.agent))
            a, bad = ("wait",), True
        sim.act(self.agent, a)
        self.agent = sim.next_decision()
        r = -(sim.t - t0) / self.time_scale - (self._tardiness() - tard0) / self.time_scale - (self.invalid_penalty if bad else 0.0)
        term, trunc = sim.done, sim.failed is not None
        if trunc:
            r -= self.fail_penalty
        info = self._info()
        info["invalid"] = bad
        if term or trunc:
            info["metrics"] = sim.metrics()
        return observe(sim, self.codec, self.agent, self.time_scale), r, term, trunc, info

    def _tardiness(self):
        return sum(j.weight * max(0.0, j.t_done - j.due) for j in self.sim.jobs.values() if j.t_done is not None)

    def graph(self):
        return graph(self.sim)

    def render(self):
        from .render import frame
        return frame(self.sim)


try:
    from pettingzoo import AECEnv

    class FactoryAEC(AECEnv):
        """PettingZoo AEC env: agents are the dispatcher and the robots; one shared team reward."""
        metadata = {"name": "factorysim_v0", "render_modes": ["rgb_array"], "is_parallelizable": False}

        def __init__(self, instance="lab", max_jobs=None, time_scale=1000.0, fail_penalty=10.0, render_mode=None, **sim_kw):
            super().__init__()
            self.inner = FactoryEnv(instance, max_jobs, time_scale, fail_penalty, render_mode, **sim_kw)
            sim = build_sim(instance, 0, **sim_kw)
            self.possible_agents = [DISPATCHER] + list(sim.robots)
            self.render_mode = render_mode

        def observation_space(self, agent):
            return self.inner.observation_space

        def action_space(self, agent):
            return self.inner.action_space

        def reset(self, seed=None, options=None):
            obs, info = self.inner.reset(seed=seed)
            self.agents = list(self.possible_agents)
            self.rewards = {a: 0.0 for a in self.agents}
            self._cumulative_rewards = {a: 0.0 for a in self.agents}
            self.terminations = {a: False for a in self.agents}
            self.truncations = {a: False for a in self.agents}
            self.infos = {a: {} for a in self.agents}
            self._obs = obs
            self.agent_selection = info["agent"] or self.agents[0]
            self.infos[self.agent_selection] = info

        def observe(self, agent):
            return observe(self.inner.sim, self.inner.codec, agent, self.inner.time_scale)

        def step(self, action):
            if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
                return self._was_dead_step(action)
            self._cumulative_rewards[self.agent_selection] = 0.0
            obs, r, term, trunc, info = self.inner.step(action)
            for a in self.agents:
                self.rewards[a] = r
                self.terminations[a], self.truncations[a] = term, trunc
            self._accumulate_rewards()
            if info["agent"]:
                self.agent_selection = info["agent"]
                self.infos[info["agent"]] = info
            if term or trunc:
                self.infos = {a: dict(info) for a in self.agents}

        def render(self):
            return self.inner.render()

        def close(self):
            pass
except ImportError:
    FactoryAEC = None


def aec_env(**kw):
    if FactoryAEC is None:
        raise ImportError("pip install pettingzoo  (or pip install -e '.[marl]')")
    return FactoryAEC(**kw)
