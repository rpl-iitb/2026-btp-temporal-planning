"""ROS 2 node: the FactorySim core on an external clock. Robots drive with Nav2 and pick/place with their arms; machines,
operators and conveyors run as timers; workpieces and stack lights are animated in Gazebo via /gazebo/set_entity_state.

    python3 -m factory_sim.ros.factory_node --instance lab --policy edd --layout run/layout.json --out run/result.json --ros-args -p use_sim_time:=true
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist
from nav_msgs.msg import Odometry
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
import threading
from rclpy.parameter import Parameter
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from .. import baselines, scenarios, timing
from ..config import Factory
from ..core import DISPATCHER, Sim, default_orders
from std_msgs.msg import String
from . import robot as arm

try:
    from gazebo_msgs.srv import SetEntityState
except ImportError:
    SetEntityState = None


def scenario(instance, robots=0, orders=0, seed=0):
    """-> (config dict, [(job id, product)]) exactly as the node will run it (also used by the launch file to place parts)."""
    inst = scenarios.load(instance) if instance.endswith(".json") else None
    cfg = dict(inst["config"] if inst else Factory(instance).cfg)
    cfg["robots"] = dict(cfg["robots"], **({"count": robots} if robots else {}))
    od = inst["orders"] if inst else None
    if orders:
        od = (od or default_orders(Factory(cfg), np.random.default_rng(seed)))[:orders]
    sim = Sim(Factory(cfg), orders=od, seed=seed, time_source="external")
    return cfg, [(j.id, j.product) for j in sim.jobs.values()], od


APPROACH, QUEUE_IN, QUEUE_GAP, DOCK_SPEED, NEAR = 0.55, 0.8, 0.7, 0.1, 0.25


def _quat(yaw):
    return math.sin(yaw / 2), math.cos(yaw / 2)


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class FactoryNode(Node):
    def __init__(self, a):
        super().__init__("factory_sim", parameter_overrides=[Parameter("use_sim_time", Parameter.Type.BOOL, True)] if a.sim_time else [])
        cfg, _, orders = scenario(a.instance, a.robots or 0, a.orders or 0, a.seed)
        self.F = Factory(cfg)
        self.sim = Sim(self.F, orders=orders, seed=a.seed, time_source="external", disturbances=a.disturbances)
        self.SetEntityState = SetEntityState
        self.policy, self.a, self.rng = baselines.make(a.policy, seed=a.seed), a, np.random.default_rng(a.seed + 1)
        self.ns = {r: "{}{}".format(a.prefix, i + 1) for i, r in enumerate(self.sim.robots)}
        self.lay = json.load(open(a.layout)) if a.layout and os.path.exists(a.layout) else None
        from .world import build
        self.docks = self.lay["docks"] if self.lay else build(self.F)[1]["docks"]
        self.nav = {r: ActionClient(self, NavigateToPose, "/{}/navigate_to_pose".format(ns)) for r, ns in self.ns.items()}
        self.init = {r: self.create_publisher(PoseWithCovarianceStamped, "/{}/initialpose".format(ns), 10) for r, ns in self.ns.items()}
        self.arm_pub = {r: self.create_publisher(JointTrajectory, "/{}/set_joint_trajectory".format(ns), 10) for r, ns in self.ns.items()}
        self.state = self.create_client(SetEntityState, "/gazebo/set_entity_state") if (SetEntityState and self.lay) else None
        self.started, self.done_q, self.travel, self.t0, self.wall0, self.retries = {}, [], [], None, time.time(), {}
        self.motion, self.conv, self.finished, self.picks = {}, {}, [], []
        self.part = dict(self.lay["parts"]) if self.lay else {}
        self.spares = {p: list(v) for p, v in self.lay["spares"].items()} if self.lay else {}
        self.cmd = {r: self.create_publisher(Twist, "/{}/cmd_vel".format(ns), 10) for r, ns in self.ns.items()}
        self.motion_group, self.anim_group, self.lock, self.odom_t = MutuallyExclusiveCallbackGroup(), MutuallyExclusiveCallbackGroup(), threading.RLock(), {}
        self.odom, self.drives, self.waits, self.phys = {}, {}, [], {r: ("free", None) for r in self.ns}
        for r, ns in self.ns.items():
            self.create_subscription(Odometry, "/{}/odom".format(ns), lambda m, r=r: self.on_odom(r, m), 10, callback_group=self.motion_group)
        self.tilt, self.uprights, self.nav_gen, self.making_way, self.spin_since, self.closing = {}, [], {}, set(), {}, False
        self.mode, self.manual_q, self.manual_dist, self.manual_last = {r: ("task" if a.teleop_mode == "task" else "auto") for r in self.ns}, {}, {}, {}
        self.by_ns = {ns: r for r, ns in self.ns.items()}
        self.create_subscription(String, "/factory_sim/teleop", self.on_teleop, 10)
        self.status_pub = self.create_publisher(String, "/factory_sim/status", 10)
        self.create_timer(0.5, self.publish_status)
        self.dock_station = {st["node"]: n for n, st in self.F.stations.items() if self.F.adj[st["node"]]}
        from .animator import Animator
        self.anim = Animator(self) if self.state else None
        self.create_timer(0.1, self.tick)
        self.create_timer(0.05, self.carry_tick, callback_group=self.anim_group)
        self.create_timer(0.05, self.drive_tick, callback_group=self.motion_group)
        self.get_logger().info("factory_sim: {} robots {}, {} orders, visuals {}".format(len(self.ns), list(self.ns.values()), self.sim.n_target, "on" if self.state else "off"))

    def now(self):
        return self.get_clock().now().nanoseconds / 1e9 - (self.t0 or 0.0)

    def approach(self, s, k=0):
        """Nav2 goal for station (or charger node) s: APPROACH metres back from its dock (k = 0), or the k-th waiting spot
        where its spur meets the aisle (k > 0), clear of the dock entrance and of the departing robot's path."""
        node = self.F.stations[s]["node"] if s in self.F.stations else s
        x, y, yaw = self.docks[node]
        if k == 0:
            return (x - math.cos(yaw) * APPROACH, y - math.sin(yaw) * APPROACH, yaw)
        nx, ny = self.F.nodes[next(iter(self.F.adj[node]))]
        d = math.hypot(x - nx, y - ny) or 1.0
        wx, wy = (x - nx) / d, (y - ny) / d
        return (nx + wx * QUEUE_IN - wy * QUEUE_GAP * (k - 1), ny + wy * QUEUE_IN + wx * QUEUE_GAP * (k - 1), math.atan2(wy, wx))

    def pose(self, node):
        x, y, yaw = node if isinstance(node, tuple) else self.docks[node]
        p = PoseStamped()
        p.header.frame_id = "map"
        p.pose.position.x, p.pose.position.y = float(x), float(y)
        p.pose.orientation.z, p.pose.orientation.w = _quat(yaw)
        return p

    # ---------------- arm
    def arm_send(self, r, keys):
        """Joint trajectory for the arm model; gazebo_ros_joint_pose_trajectory reads time_from_start as the delay after the previous point."""
        msg = JointTrajectory()
        msg.header.frame_id = "world"
        msg.joint_names = arm.JOINTS
        for t, q in arm.sample(keys, dt=0.05):
            pt = JointTrajectoryPoint()
            pt.positions = [float(x) for x in q]
            pt.time_from_start.nanosec = 50000000
            msg.points.append(pt)
        self.arm_pub[r].publish(msg)

    def arm_q(self, r, t):
        m = self.motion.get(r)
        if self.mode.get(r) == "drive" and r in self.manual_q:
            return tuple(self.manual_q[r][:4])
        if m is None:
            rob = self.sim.robots[r]
            return arm.CARRY if rob.carry is not None else arm.HOME
        keys, t0 = m["keys"], m["t0"]
        dt = t - t0
        for (ta, a, _), (tb, b, _) in zip(keys, keys[1:]):
            if dt < tb:
                u = max(0.0, (dt - ta) / max(1e-6, tb - ta))
                u = u * u * (3 - 2 * u)
                return tuple(x + (y - x) * u for x, y in zip(a, b))[:4]
        return keys[-1][1][:4]

    # ---------------- Gazebo entities
    def model(self, j):
        key = str(j.id)
        if key not in self.part:
            pool = self.spares.get(j.product, [])
            self.part[key] = pool.pop(0) if pool else None
        return self.part[key]

    def slot(self, s, lane, i):
        pts = self.lay["stations"][s].get(lane) or [self.lay["stations"][s]["work"]]
        p = list(pts[min(i, len(pts) - 1)])
        p[2] += 0.06 * max(0, i - len(pts) + 1)
        return p

    def where(self, j, t):
        """World position of job j's workpiece, or ('held', robot) when it is in a gripper."""
        sim, w = self.sim, j.where
        for r, m in self.motion.items():
            if m["jid"] == j.id and m["ok"]:
                if m["kind"] == "load":
                    return ("held", r) if t >= m["t0"] + m["event"] else m["target"]
                return m["target"] if t >= m["t0"] + m["event"] else ("held", r)
        if w == "robot":
            return ("held", j.at)
        if w in ("queue", "buffer"):
            st = sim.stations[j.at]
            i = next((k for k, x in enumerate(st.q) if x is j), 0)
            if st.spec["kind"] == "kitting":
                x, y, z = self.lay["stations"][j.at]["work"]
                _, _, yaw = self.docks[st.spec["node"]]
                return [x - math.sin(yaw) * 0.22 * (i + 1), y + math.cos(yaw) * 0.22 * (i + 1), z]
            return self.slot(j.at, "in", i)
        if w == "machine":
            st = sim.stations[j.at]
            return self.slot(j.at, "out", len(st.out)) if st.done and self.lay["stations"][j.at].get("out") and self.F.stations[j.at]["kind"] != "port" else self.lay["stations"][j.at]["work"] or self.slot(j.at, "in", 0)
        if w == "out":
            st = sim.stations[j.at]
            return self.slot(j.at, "out", next((k for k, x in enumerate(st.out) if x is j), 0))
        if w == "conveyor" and j.id in self.conv:
            path, ta, tb = self.conv[j.id]
            return self.along(path, min(1.0, max(0.0, (t - ta) / max(1e-6, tb - ta))))
        if w == "done":
            if j.id not in self.finished:
                self.finished.append(j.id)
            store = self.lay.get("store") or []
            return store[self.finished.index(j.id) % len(store)] if store else list(self.lay["hidden"])
        return list(self.lay["hidden"])

    @staticmethod
    def along(path, f):
        segs = [math.dist(a[:2], b[:2]) for a, b in zip(path, path[1:])]
        d = f * sum(segs)
        for (a, b), L in zip(zip(path, path[1:]), segs):
            if d <= L or L == segs[-1]:
                u = d / L if L else 1.0
                return [a[k] + (b[k] - a[k]) * min(1.0, u) for k in range(3)]
            d -= L
        return list(path[-1])

    def carry_tick(self):
        if self.anim:
            self.anim.held(self.now())

    # ---------------- main loop
    def tick(self):
        with self.lock:
            self._tick()
        if self.anim and self.t0 is not None:
            self.anim.update(self.now())

    def _tick(self):
        sim = self.sim
        if self.t0 is None:
            if self.anim:
                self.anim.update(0.0)
            for r, pub in self.init.items():
                m = PoseWithCovarianceStamped()
                m.header.frame_id = "map"
                m.pose.pose = self.pose(sim.robots[r].node).pose
                m.pose.pose.orientation.z, m.pose.pose.orientation.w = 0.0, 1.0
                m.pose.covariance[0] = m.pose.covariance[7] = 0.05
                m.pose.covariance[35] = 0.05
                pub.publish(m)
            if all(c.server_is_ready() for c in self.nav.values()) and time.time() - self.wall0 > self.a.warmup:
                self.t0 = self.get_clock().now().nanoseconds / 1e9
                for r in self.ns:
                    self.arm_send(r, [(0.0, tuple(arm.HOME) + (arm.OPEN, arm.OPEN), None), (1.0, tuple(arm.HOME) + (arm.OPEN, arm.OPEN), None)])
                self.get_logger().info("Nav2 ready: factory starts")
            return
        t = self.now()
        sim.advance(t)
        for aid, (end, ok) in list(self.started.items()):
            if aid not in sim.external_pending:
                del self.started[aid]
            elif not isinstance(end, str) and end <= t:
                sim.complete(aid, end, ok=ok)
                del self.started[aid]
        while self.done_q:
            aid, te, ok = self.done_q.pop(0)
            if aid in sim.external_pending:
                kind, data = sim.external_pending[aid][:2]
                sim.complete(aid, max(te, sim.t), ok=ok)
                if kind in ("load", "unload"):
                    self.motion.pop(data, None)
            self.started.pop(aid, None)
        while True:
            agent = sim.next_decision()
            if agent is None:
                break
            if agent != DISPATCHER and self.mode[agent] == "task":
                sim.decided.add(agent)
                continue
            sim.act(agent, self.policy(sim, agent, sim.legal(agent)))
        for aid, (kind, data, t_start, nom) in list(sim.external_pending.items()):
            if aid in self.started:
                continue
            if kind == "travel":
                self.started[aid] = ("nav", True)
                self.go(aid, data)
            elif kind in ("load", "unload"):
                self.started[aid] = self.service(aid, kind, data, t)
            elif kind == "charge":
                self.started[aid] = self.charge(aid, data, nom * self.a.charge_scale)
            else:
                d = nom if self.a.nominal else self.duration(kind, data, nom)
                d *= self.a.proc_scale
                self.started[aid] = (t + d, True)
                if kind == "conv" and self.lay:
                    self.conv[data[0]] = (self.lay["conveyors"]["{}|{}".format(data[1], data[2])], t, t + d)
        if sim.done or sim.failed is not None:
            self.finish()

    def duration(self, kind, data, nom):
        """Sampled duration from the FactorySim timing model of this activity (the backend owns physical time)."""
        sim = self.sim
        extra = 0.0
        if kind == "proc":
            st = sim.stations[data]
            model = self.F.proc_model(data, st.cur.product, st.config) if st.cur is not None else {"fixed": nom}
            model = model if abs(timing.mean(model) - nom) < 1e-6 else {"fixed": nom}
        elif kind == "conv":
            model = self.F.transfers[(data[1], data[2])]
            extra = nom - timing.mean(model)
        elif kind == "setup":
            model = self.F.stations[data]["changeover"]
        elif kind == "repair":
            model = self.F.stations[data]["breakdown"]["mttr"]
        else:
            model = {"fixed": nom}
        return float(timing.sample(model, self.rng, 1.0)) + extra

    def charge(self, aid, r, d):
        """Charging: dock into the charger, stay d seconds, reverse out."""
        c = self.sim.robots[r].task[2]
        dock = tuple(self.docks[c])

        def undocked():
            self.phys[r] = ("approach", c)
            self.done_q.append((aid, self.now(), True))

        def docked():
            self.phys[r] = ("docked", c)
            self.get_logger().info("t={:.0f}s {} charging at {} ({:.0f}s, battery {:.0f}%)".format(self.now(), self.ns[r], c, d, self.sim.robots[r].battery))
            self.waits.append((self.now() + d, lambda: self.drive(r, self.approach(c), -1, undocked)))
        if self.phys[r][0] == "queue":
            self.when_clear(r, self.approach(c), lambda: self.nav_to(r, self.approach(c), lambda: self.drive(r, dock, 1, docked)))
        else:
            self.drive(r, dock, 1, docked)
        return ("svc", True)

    def service(self, aid, kind, r, t):
        """Pick ('load') or place ('unload') with the arm: sampled duration, grasp outcome, joint trajectory."""
        sim, rob = self.sim, self.sim.robots[r]
        sp = rob.spec
        d = float(timing.sample(sp[kind], self.rng, 0.0 if self.a.nominal else 1.0))
        ok = self.a.nominal or self.rng.random() < sp["grasp_success" if kind == "load" else "place_success"]
        s = rob.task[2]
        if sp["handling"] == "handler":
            return self.handover(aid, kind, r, t, d, ok)
        if kind == "load":
            j = sim.jobs[rob.task[1]]
            target = self.where(j, t) if self.lay else [0, 0, 0]
            target = target if not isinstance(target, tuple) else self.slot(s, "out", 0)
        else:
            j = sim.jobs[rob.carry]
            target = self.slot(s, "in", len(sim.stations[s].q)) if self.lay and self.lay["stations"][s].get("in") else (self.lay["stations"][s]["work"] if self.lay else [0, 0, 0])
        keys = arm.motion(kind, self.docks[sim.stations[s].spec["node"]], target, d)
        if not ok:
            keys = arm.failed_grasp(keys) if kind == "load" else keys
        rec = {"robot": self.ns[r], "kind": "pick" if kind == "load" else "place", "station": s, "job": j.id, "t": round(t, 2), "arm_seconds": round(d, 2), "ok": bool(ok)}

        def undocked():
            self.phys[r] = ("approach", s)
            rec["seconds"] = round(self.now() - t, 2)
            self.picks.append(rec)
            self.done_q.append((aid, self.now(), ok))

        def arm_done():
            self.drive(r, self.approach(s), -1, undocked)

        def docked():
            self.phys[r] = ("docked", s)
            ta = self.now()
            self.get_logger().info("t={:.0f}s {} {} job {} at {} ({:.1f}s){}".format(ta, self.ns[r], rec["kind"], j.id, s, d, "" if ok else " FAILED, will retry"))
            self.arm_send(r, keys)
            self.motion[r] = {"kind": kind, "jid": j.id, "keys": keys, "t0": ta, "target": target, "ok": ok, "event": arm.event_time(keys, "grasp" if kind == "load" else "release") or d * 0.55}
            self.waits.append((ta + d, arm_done))

        dock = tuple(self.docks[sim.stations[s].spec["node"]])
        if self.phys[r][0] == "queue":
            self.when_clear(r, self.approach(s), lambda: self.nav_to(r, self.approach(s), lambda: self.drive(r, dock, 1, docked)))
        else:
            self.drive(r, dock, 1, docked)
        return ("svc", ok)

    def handover(self, aid, kind, r, t, d, ok):
        """Tray robot: it docks while the nearest free handler walks over; the handler moves the tray (d seconds)."""
        sim, rob = self.sim, self.sim.robots[r]
        s = rob.task[2]
        dock = tuple(self.docks[sim.stations[s].spec["node"]])
        if kind == "load":
            j = sim.jobs[rob.task[1]]
            target = self.where(j, t) if self.lay else [0, 0, 0]
            target = target if not isinstance(target, tuple) else self.slot(s, "out", 0)
        else:
            j = sim.jobs[rob.carry]
            target = self.slot(s, "in", len(sim.stations[s].q)) if self.lay and self.lay["stations"][s].get("in") else (self.lay["stations"][s]["work"] if self.lay else [0, 0, 0])
        pool, i = rob.hand or (None, None)
        name = "handler_{}".format(i) if (self.lay and pool == self.F.handler_pool and i is not None and i < len(self.lay.get("handlers", []))) else None
        walk, hand = 0.0, None
        if name and self.anim:
            x, y, yaw = dock
            stand = (x - 0.45 * math.sin(yaw) + 0.1 * math.cos(yaw), y + 0.45 * math.cos(yaw) + 0.1 * math.sin(yaw), yaw - math.pi / 2)
            from .animator import Path
            path = Path(self.anim.hand_route(self.anim.hand_at[name], stand), t)
            walk, hand = path.length / path.speed, (name, path, stand)
        rec = {"robot": self.ns[r], "kind": "handover-" + ("load" if kind == "load" else "unload"), "station": s, "job": j.id, "t": round(t, 2), "hand_seconds": round(d, 2), "walk_seconds": round(walk, 2), "ok": True}
        if hand:
            self.motion[r] = {"kind": kind, "jid": j.id, "keys": None, "t0": t + walk, "target": target, "ok": False, "event": d * 0.5, "hand": hand}

        def undocked():
            self.phys[r] = ("approach", s)
            rec["seconds"] = round(self.now() - t, 2)
            self.picks.append(rec)
            self.done_q.append((aid, self.now(), True))

        def docked():
            self.phys[r] = ("docked", s)
            ta = max(self.now(), t + walk)
            self.get_logger().info("t={:.0f}s {} handover {} job {} at {} by {} ({:.1f}s, walk {:.0f}s)".format(self.now(), self.ns[r], "onto" if kind == "load" else "off", j.id, s, name or pool, d, walk))
            m = self.motion.get(r) or {"kind": kind, "jid": j.id, "keys": None, "target": target, "event": d * 0.5}
            m.update(t0=ta, ok=True)
            self.motion[r] = m
            self.waits.append((ta + d, lambda: self.drive(r, self.approach(s), -1, undocked)))

        if self.phys[r][0] == "queue":
            self.when_clear(r, self.approach(s), lambda: self.nav_to(r, self.approach(s), lambda: self.drive(r, dock, 1, docked)))
        else:
            self.drive(r, dock, 1, docked)
        return ("svc", True)

    def go(self, aid, robot):
        """Travel activity: Nav2 to the station's approach pose (or a queue spot if another robot is there), or to the node."""
        self.making_way.discard(robot)
        r = self.sim.robots[robot]
        s = self.dock_station.get(r.dest) or (r.dest if r.dest in self.F.chargers else None)
        if s is None:
            self.phys[robot], target = ("free", None), r.dest
        else:
            k = sum(1 for o, (st, x) in self.phys.items() if o != robot and x == s and st in ("approach", "queue", "docked"))
            self.phys[robot], target = (("approach" if k == 0 else "queue"), s), self.approach(s, k)
        t_send = self.now()

        def ok():
            self.travel.append({"robot": self.ns[robot], "from": r.src, "to": r.dest, "distance": self.F.dist[r.src].get(r.dest), "seconds": round(self.now() - t_send, 3)})
            self.done_q.append((aid, self.now(), True))
        self.nav_to(robot, target, ok)

    def when_clear(self, robot, point, fn, radius=1.0):
        """Run fn once no other robot is within `radius` of point; an idle robot parked there is asked to make way."""
        blockers = [o for o, p in self.odom.items() if o != robot and math.dist(p[:2], point[:2]) < radius]
        for o in blockers:
            rb = self.sim.robots[o]
            if rb.task is None and not rb.manual and o not in self.drives and o not in self.making_way:
                self.make_way(o, point)
        if blockers:
            self.waits.append((self.now() + 0.5, lambda: self.when_clear(robot, point, fn, radius)))
        else:
            fn()

    def make_way(self, robot, point):
        """Send an idle robot from a dock approach to the nearest free aisle node (the core still sees it at its node)."""
        taken = [p[:2] for o, p in self.odom.items() if o != robot]
        aisle = [n for n in self.F.nodes if n not in self.dock_station and n not in self.F.chargers and len(self.F.adj[n]) >= 2 and all(math.dist(self.F.nodes[n], q) > 0.8 for q in taken)]
        if not aisle:
            return
        n = min(aisle, key=lambda n: math.dist(self.F.nodes[n], point[:2]))
        self.making_way.add(robot)
        self.phys[robot] = ("free", None)
        self.get_logger().info("{} makes way: parks at {}".format(self.ns[robot], n))
        self.nav_to(robot, n, lambda: self.making_way.discard(robot))

    def nav_to(self, robot, target, on_ok, tries=0):
        """Nav2 goal; arrival is accepted once Nav2 reports < NEAR metres left (the docking controller does the last cm).
        A newer goal for the same robot supersedes this one (its retries and callbacks are dropped)."""
        if tries == 0:
            self.nav_gen[robot] = self.nav_gen.get(robot, 0) + 1
        gen = self.nav_gen[robot]
        goal = NavigateToPose.Goal()
        goal.pose = self.pose(target)
        state = {"done": False, "h": None}
        tx, ty = (target[0], target[1]) if isinstance(target, tuple) else self.docks[target][:2]

        def finish():
            if not state["done"] and gen == self.nav_gen.get(robot):
                state["done"] = True
                on_ok()

        def retry():
            if state["done"] or gen != self.nav_gen.get(robot):
                return
            state["done"] = True
            self.get_logger().warn("navigation failed for {} (try {}), retrying".format(self.ns[robot], tries + 1))
            if tries < 20:
                self.waits.append((self.now() + 2.0, lambda: self.nav_to(robot, target, on_ok, tries + 1)))

        def feedback(fb):
            p = self.map_pose(robot)
            if not state["done"] and fb.feedback.distance_remaining < NEAR and p and math.hypot(p[0] - tx, p[1] - ty) < NEAR:
                if state["h"] is not None:
                    state["h"].cancel_goal_async()
                finish()

        def accepted(f):
            h = f.result()
            if not h.accepted:
                return retry()
            state["h"] = h
            h.get_result_async().add_done_callback(lambda rf: finish() if rf.result().status == 4 else (finish() if (self.map_pose(robot) and math.hypot(self.map_pose(robot)[0] - tx, self.map_pose(robot)[1] - ty) < NEAR) else retry()))
        self.nav[robot].send_goal_async(goal, feedback_callback=feedback).add_done_callback(accepted)

    # ---------------- keyboard teleop (factory_sim.ros.teleop)
    def on_teleop(self, msg):
        try:
            c = json.loads(msg.data)
            r = self.by_ns[c["robot"]]
        except (ValueError, KeyError):
            return
        with self.lock:
            sim, rob = self.sim, self.sim.robots[r]
            if c["cmd"] == "mode" and c["mode"] in ("auto", "task", "drive") and c["mode"] != self.mode[r]:
                if self.mode[r] == "drive":
                    p = self.map_pose(r) or (0.0, 0.0, 0.0)
                    others = [n for n in self.F.nodes if n not in self.dock_station and n not in self.F.chargers]
                    node = min(others, key=lambda n: math.dist(self.F.nodes[n], p[:2]))
                    sim.release(r, node, self.manual_dist.pop(r, 0.0))
                    self.manual_last.pop(r, None)
                    self.cmd[r].publish(Twist())
                    self.phys[r] = ("free", None)
                if c["mode"] == "drive":
                    sim.hold(r)
                    self.manual_q[r] = list(arm.CARRY if rob.carry is not None else arm.HOME) + [arm.OPEN, arm.OPEN]
                self.mode[r] = c["mode"]
                sim.decided.discard(r)
                self.get_logger().info("teleop: {} -> {}".format(self.ns[r], c["mode"].upper()))
            elif c["cmd"] == "act" and self.mode[r] == "task":
                a = tuple(c["action"])
                if a in sim.legal(r):
                    sim.act(r, a)
                    self.get_logger().info("teleop: {} does {}".format(self.ns[r], a))
            elif c["cmd"] == "arm" and self.mode[r] == "drive" and rob.manual:
                q = [float(x) for x in c["q"]][:6]
                q = [min(hi, max(lo, v)) for v, (lo, hi) in zip(q, arm.LIMITS)]
                self.arm_send(r, [(0.0, tuple(self.manual_q.get(r, q)), None), (0.5, tuple(q), None)])
                self.manual_q[r] = q

    def publish_status(self):
        sim = self.sim
        with self.lock:
            out = {"t": round(self.now(), 1) if self.t0 is not None else None, "done": sim.completed, "orders": sim.n_target, "robots": {}, "jobs": {}}
            for r, ns in self.ns.items():
                rob = sim.robots[r]
                legal = [list(a) for a in sim.legal(r) if a[0] != "wait"] if self.mode[r] == "task" and rob.task is None else []
                out["robots"][ns] = {"mode": self.mode[r], "held": rob.manual, "phase": rob.phase or ("wait" if rob.task is None else rob.task[0]), "battery": None if rob.battery is None else round(rob.battery, 1),
                                     "carry": rob.carry, "node": rob.dest, "task": list(rob.task) if rob.task else None, "fault": rob.fault, "legal": legal, "q": self.manual_q.get(r) if rob.spec["tool"] == "arm" else None,
                                     "type": rob.spec["type"], "tool": rob.spec["tool"]}
            for j in sim.jobs.values():
                out["jobs"][str(j.id)] = [j.product, j.where, j.at]
        self.status_pub.publish(String(data=json.dumps(out)))

    def on_odom(self, r, m):
        self.odom_t[r] = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        if self.mode.get(r) == "drive":
            p = (m.pose.pose.position.x, m.pose.pose.position.y)
            if r in self.manual_last:
                self.manual_dist[r] = self.manual_dist.get(r, 0.0) + math.dist(p, self.manual_last[r])
            self.manual_last[r] = p
        q = m.pose.pose.orientation
        self.odom[r] = (m.pose.pose.position.x, m.pose.pose.position.y, math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)))
        tipped = abs(q.x) > 0.07 or abs(q.y) > 0.07
        tw, now, pos = m.twist.twist, self.now(), (m.pose.pose.position.x, m.pose.pose.position.y)
        if abs(tw.linear.x) > 0.04:
            since, anchor = self.spin_since.get(r) or (now, pos)
            if math.dist(anchor, pos) > 0.03:
                since, anchor = now, pos
            self.spin_since[r] = (since, anchor)
            tipped = tipped or now - since > 6.0
        else:
            self.spin_since.pop(r, None)
        if not tipped:
            self.tilt.pop(r, None)
        elif r not in self.tilt:
            self.tilt[r] = self.now()
        elif self.now() - self.tilt[r] > 3.0 and self.state:
            self.tilt.pop(r)
            self.spin_since.pop(r, None)
            x0, y0, yaw = self.odom[r]
            others = [self.odom[o][:2] for o in self.odom if o != r]
            free = [p for n, p in self.F.nodes.items() if n not in self.dock_station and n not in self.F.chargers and len(self.F.adj[n]) >= 2 and all(math.dist(p, q) > 0.8 for q in others)]
            x, y = min(free, key=lambda p: math.dist(p, (x0, y0))) if free else (x0, y0)
            self.get_logger().error("{} tipped, wedged or stuck at ({:.2f}, {:.2f}): an operator sets it upright at ({:.1f}, {:.1f})".format(self.ns[r], x0, y0, x, y))
            self.uprights.append({"robot": self.ns[r], "t": round(self.now(), 1), "x": round(x0, 2), "y": round(y0, 2)})
            req = SetEntityState.Request()
            req.state.name, req.state.reference_frame = self.ns[r], "world"
            req.state.pose.position.x, req.state.pose.position.y, req.state.pose.position.z = float(x), float(y), 0.05
            req.state.pose.orientation.z, req.state.pose.orientation.w = _quat(yaw)
            self.state.call_async(req)

    def map_pose(self, robot):
        """Robot pose in the map: Gazebo's diff-drive odometry is world-sourced, and the map frame is the world frame."""
        return self.odom.get(robot)

    def drive(self, robot, target, direction, then):
        """Docking controller: forward along the dock line to `target` (x, y, yaw), or straight back (direction -1) to it."""
        self.drives[robot] = {"target": target, "dir": direction, "t0": self.now(), "then": then}

    def drive_tick(self):
        if self.t0 is None:
            return
        with self.lock:
            self._drive_tick()

    def _drive_tick(self):
        t = self.now()
        clock = self.get_clock().now().nanoseconds / 1e9
        for w in [w for w in self.waits if w[0] <= t]:
            self.waits.remove(w)
            w[1]()
        for r, dv in list(self.drives.items()):
            p = self.map_pose(r)
            if t - dv["t0"] > 40:
                self.cmd[r].publish(Twist())
                del self.drives[r]
                dv["then"]()
                continue
            if p is None or clock - self.odom_t.get(r, clock) > 0.3:
                self.cmd[r].publish(Twist())
                continue
            tx, ty, ta = dv["target"]
            ux, uy = math.cos(ta), math.sin(ta)
            s, e = (p[0] - tx) * ux + (p[1] - ty) * uy, -(p[0] - tx) * uy + (p[1] - ty) * ux
            tw = Twist()
            done = s >= -0.005 if dv["dir"] > 0 else s <= 0.005
            if done or t - dv["t0"] > 40:
                self.cmd[r].publish(tw)
                del self.drives[r]
                dv["then"]()
                continue
            h = _wrap((ta - math.atan(3.0 * e) if dv["dir"] > 0 else ta) - p[2])
            if abs(h) > 0.35:
                tw.angular.z = max(-0.6, min(0.6, 1.5 * h))
            else:
                tw.linear.x, tw.angular.z = DOCK_SPEED * dv["dir"], max(-0.5, min(0.5, 2.0 * h))
            self.cmd[r].publish(tw)

    def finish(self, interrupted=False):
        if self.closing:
            return
        self.closing = True
        m = self.sim.metrics()
        out = {"interrupted": interrupted, "sim_time": round(self.sim.t, 2), "metrics": m, "rfid": self.sim.rfid, "travel": self.travel, "manipulation": self.picks, "uprights": self.uprights, "wall_seconds": round(time.time() - self.wall0, 1)}
        json.dump(out, open(self.a.out, "w"), indent=1)
        self.get_logger().info("factory done: success={} makespan={} grasp_failures={} failure={} -> {}".format(m["success"], m["makespan"], m["grasp_failures"], m["failure"], self.a.out))


def main(argv=None):
    argv = rclpy.utilities.remove_ros_args(argv if argv is not None else sys.argv)[1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", default="lab")
    ap.add_argument("--policy", default="edd")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--robots", type=int, default=0, help="override the number of robots")
    ap.add_argument("--orders", type=int, default=0, help="use only the first N orders")
    ap.add_argument("--proc-scale", type=float, default=1.0, help="scale machine, conveyor, changeover, service and repair times (e.g. 0.1 for quick tests)")
    ap.add_argument("--nominal", action="store_true", help="use nominal durations and always-successful grasps")
    ap.add_argument("--prefix", default="robot", help="robot namespaces: <prefix>1, <prefix>2, ...")
    ap.add_argument("--layout", help="layout.json from factory_sim.ros.world (enables workpiece / stack-light animation)")
    ap.add_argument("--sim-time", action="store_true", help="use the simulation clock (/clock), as in Gazebo")
    ap.add_argument("--teleop-mode", default="auto", choices=("auto", "task"), help="initial mode of every robot (task: you choose their tasks with factory_sim.ros.teleop)")
    ap.add_argument("--disturbances", action="store_true", help="let the core sample breakdowns, robot faults, jams and rejects")
    ap.add_argument("--charge-scale", type=float, default=1.0, help="scale charging times (e.g. 0.1 for quick tests)")
    ap.add_argument("--warmup", type=float, default=10.0, help="wall seconds to publish initial poses before starting")
    ap.add_argument("--out", default="factory_run.json")
    a = ap.parse_args(argv)
    import faulthandler
    import signal
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    rclpy.init(args=sys.argv if argv is None else None)
    node = FactoryNode(a)
    try:
        ex = MultiThreadedExecutor(num_threads=4)
        ex.add_node(node)
        while rclpy.ok() and not node.closing:
            ex.spin_once(timeout_sec=0.2)
        ex.shutdown()
    except (SystemExit, KeyboardInterrupt):
        pass
    except Exception as e:  # rclpy's ExternalShutdownException on Ctrl-C / launch shutdown
        if type(e).__name__ != "ExternalShutdownException":
            raise
    if not node.closing:   # stopped from outside (Ctrl-C, timeout): keep what happened so far
        with node.lock:
            node.finish(interrupted=True)
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
