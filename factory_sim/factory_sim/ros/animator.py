"""Gazebo animation of the FactorySim state through /gazebo/set_entity_state: workpieces (tweened moves), stack lights,
CNC doors and spindles, conveyor-station presses, operators (working, breaks), technicians (repairs, services, changeovers),
robot status LEDs and charger lamps. Pure visuals: no collisions, no effect on the simulation.
"""
import math

from . import robot as arm

LED_Z = 0.245
WALK = 1.0


def _q(roll, pitch, yaw):
    cr, sr, cp, sp, cy, sy = math.cos(roll / 2), math.sin(roll / 2), math.cos(pitch / 2), math.sin(pitch / 2), math.cos(yaw / 2), math.sin(yaw / 2)
    return (sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy, cr * cp * cy + sr * sp * sy)


class Path:
    """Constant-speed walk along a polyline [(x, y), ...] starting at time t0."""

    def __init__(self, pts, t0, speed=WALK):
        self.pts, self.t0, self.speed = [tuple(p[:2]) for p in pts], t0, speed
        self.seg = [math.dist(a, b) for a, b in zip(self.pts, self.pts[1:])]
        self.length = sum(self.seg)

    def at(self, t):
        d = min(self.length, max(0.0, (t - self.t0) * self.speed))
        for (a, b), L in zip(zip(self.pts, self.pts[1:]), self.seg):
            if d <= L and L > 0:
                u = d / L
                return a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u, math.atan2(b[1] - a[1], b[0] - a[0]), False
            d -= L
        a, b = (self.pts[-2], self.pts[-1]) if len(self.pts) > 1 else (self.pts[0], self.pts[0])
        return b[0], b[1], math.atan2(b[1] - a[1], b[0] - a[0]) if a != b else 0.0, True


class Animator:
    def __init__(self, node):
        self.n, self.lay, self.sim = node, node.lay, node.sim
        self.shown, self.pos, self.tween = {}, {}, {}
        self.walk, self.away, self.tech_job, self.tech_at = {}, {}, {}, {}
        self.home = {name: (self.lay["bay"]["home"][0], self.lay["bay"]["home"][1] - 0.6 * i, self.lay["bay"]["home"][2]) for i, name in enumerate(self.lay["technicians"])}
        self.ops = {s: info["operator"] for s, info in self.lay["stations"].items() if "operator" in info}
        self.people = self.lay.get("people", True)
        self.hand_at = {n: tuple(h) for n, h in zip(self.lay.get("handlers", []), self.lay.get("handler_home", []))}
        self.tools = self.lay.get("tools") or ["arm"] * len(node.ns)

    # ---------------- low level
    def put(self, name, xyz, yaw=0.0, pitch=0.0, frame="world", force=False):
        key = (frame, round(xyz[0], 3), round(xyz[1], 3), round(xyz[2], 3), round(yaw, 3), round(pitch, 3))
        if not force and self.shown.get(name) == key:
            return
        self.shown[name] = key
        req = self.n.SetEntityState.Request()
        req.state.name, req.state.reference_frame = name, frame
        p = req.state.pose
        p.position.x, p.position.y, p.position.z = float(xyz[0]), float(xyz[1]), float(xyz[2])
        p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = _q(0.0, pitch, yaw)
        self.n.state.call_async(req)

    def hide(self, name):
        self.put(name, self.lay["hidden"])

    def corridor(self, p):
        return self.lay["corridors"]["south" if p[1] < self.lay["aisle_y"] else "north"]

    def route(self, a, b):
        """Walking route for people: via the back corridors behind the machine rows and the west strip (robots never go there)."""
        west = self.lay["corridors"]["west"]
        pts = [a[:2]]
        on_west = lambda p: p[0] < west + 0.8  # noqa: E731
        ya = None if on_west(a) else self.corridor(a)
        yb = None if on_west(b) else self.corridor(b)
        if ya is not None:
            pts.append((a[0], ya))
        if ya != yb or ya is None or yb is None:
            pts.append((west, ya if ya is not None else a[1]))
            pts.append((west, yb if yb is not None else b[1]))
        if yb is not None:
            pts.append((b[0], yb))
        pts.append(b[:2])
        return [p for i, p in enumerate(pts) if i == 0 or math.dist(p, pts[i - 1]) > 1e-3]

    # ---------------- people
    def person(self, name, x, y, yaw, t, mode):
        """mode: 'work' (arms busy), 'idle', 'walk'."""
        bob = 0.012 * abs(math.sin(t * 2 * math.pi * 1.6)) if mode == "walk" else 0.0
        self.put(name, (x, y, bob), yaw)
        for side, l, ph in (("arm_l", 0.27, 0.0), ("arm_r", -0.27, math.pi)):
            if mode == "work":
                pitch = 0.15 + 0.3 * math.sin(t * 2 * math.pi * 1.2 + ph)
            elif mode == "walk":
                pitch = 1.25 + 0.35 * math.sin(t * 2 * math.pi * 1.6 + ph)
            else:
                pitch = 1.3
            self.put("{}::{}".format(name, side), (0.0, l, 1.12 + bob), 0.0, pitch, frame=name)

    def operators(self, t):
        sim = self.sim
        for pool, cap in sim.ops_cap.items():
            mine = [s for s in self.ops if sim.F.stations[s].get("operator") == pool]
            want = max(0, sim.F.operators[pool] - cap)
            out = [s for s in mine if self.away.get(s)]
            for s in out[want:] if len(out) > want else []:
                self.away[s] = False
                self.walk["op_" + s] = (Path(self.route(self.lay["exit"], self.ops[s]), t), "back")
            if len(out) < want:
                for s in sorted([s for s in mine if not self.away.get(s)], key=lambda s: sim.stations[s].cur is not None)[:want - len(out)]:
                    self.away[s] = True
                    self.walk["op_" + s] = (Path(self.route(self.ops[s], self.lay["exit"]), t), "out")
        for s, pose in self.ops.items():
            name = "op_" + s
            if name in self.walk:
                path, kind = self.walk[name]
                x, y, yaw, done = path.at(t)
                if done:
                    del self.walk[name]
                    if kind == "out":
                        self.hide(name)
                        continue
                else:
                    self.person(name, x, y, yaw, t, "walk")
                    continue
            if self.away.get(s):
                continue
            st = sim.stations[s]
            self.person(name, pose[0], pose[1], pose[2], t, "work" if st.cur is not None and not st.done else "idle")

    def technicians(self, t):
        sim = self.sim
        active = {n for n, st in sim.stations.items() if "service" in self.lay["stations"].get(n, {}) and ((st.down and st.wait is None) or st.service or (st.setup is not None and st.wait is None))}
        for tech, job in list(self.tech_job.items()):
            if job not in active:
                del self.tech_job[tech]
                self.walk[tech] = (Path(self.route(self.tech_at.get(tech, self.home[tech]), self.home[tech]), t), "home")
        for job in sorted(active - set(self.tech_job.values())):
            free = [x for x in self.home if x not in self.tech_job]
            if not free:
                break
            tech = free[0]
            self.tech_job[tech] = job
            dst = self.lay["stations"][job]["service"]
            self.walk[tech] = (Path(self.route(self.tech_at.get(tech, self.home[tech]), dst), t), "job")
        for tech, home in self.home.items():
            if tech in self.walk:
                path, kind = self.walk[tech]
                x, y, yaw, done = path.at(t)
                self.tech_at[tech] = (x, y, yaw)
                if not done:
                    self.person(tech, x, y, yaw, t, "walk")
                    continue
                del self.walk[tech]
            if tech in self.tech_job:
                d = self.lay["stations"][self.tech_job[tech]]["service"]
                self.tech_at[tech] = tuple(d)
                self.person(tech, d[0], d[1], d[2], t, "work")
            else:
                self.tech_at[tech] = home
                self.person(tech, home[0], home[1], home[2], t, "idle")

    # ---------------- machines
    def machines(self, t):
        sim = self.sim
        for s, info in self.lay["stations"].items():
            st = sim.stations[s]
            state = sim.state(s)
            cutting = st.cur is not None and not st.done and not st.down
            on = {"busy": "g", "blocked": "y", "setup": "y" if int(t * 2) % 2 == 0 else None, "service": "y" if int(t * 2) % 2 == 0 else None, "down": "r" if int(t * 3) % 2 == 0 else None}.get(state)
            for c, p in info["lamps"].items():
                if c == on:
                    self.put("lamp_{}_{}".format(s, c), p)
                else:
                    self.hide("lamp_{}_{}".format(s, c))
            if "door" in info:
                target = info["door"]["closed" if cutting or st.down else "open"]
                cur = self.pos.get("door_" + s, info["door"]["closed"])
                dx, dy = target[0] - cur[0], target[1] - cur[1]
                step = min(1.0, 0.05 / max(1e-6, math.hypot(dx, dy)))
                nxt = (cur[0] + dx * step, cur[1] + dy * step, target[2])
                self.pos["door_" + s] = nxt
                self.put("door_" + s, (nxt[0], nxt[1], 0.0), nxt[2])
            if "spindle" in info:
                z = -0.035 - 0.02 * math.sin(t * 2 * math.pi / 1.6) if cutting else 0.0
                self.put("spindle_" + s, (info["spindle"][0], info["spindle"][1], z), info["spindle"][2])
            if "press" in info:
                z = -0.035 * (0.5 - 0.5 * math.cos(t * 2 * math.pi / 1.2)) if cutting else 0.0
                self.put("press_" + s, (info["press"][0], info["press"][1], z), info["press"][2])

    # ---------------- parts
    def parts(self, t):
        n = self.n
        for j in self.sim.jobs.values():
            name = n.model(j)
            if name is None:
                continue
            p = n.where(j, t)
            if isinstance(p, tuple) and p[0] == "held":
                continue
            p = tuple(p)
            if j.where == "conveyor" or p[2] < -1 or self.pos.get(name) is None or self.pos[name][2] < -1:
                self.tween.pop(name, None)
                self.pos[name] = p
                self.put(name, p)
                continue
            tw = self.tween.get(name)
            if (tw is None and math.dist(self.pos[name], p) > 0.005) or (tw is not None and tw[1] != p):
                a = self.pos[name]
                self.tween[name] = (a, p, t, max(0.4, min(2.0, math.dist(a, p) / 0.35)))
                tw = self.tween[name]
            if tw:
                a, b, t0, dur = tw
                u = min(1.0, (t - t0) / dur)
                u = u * u * (3 - 2 * u)
                lift = 0.05 * math.sin(math.pi * u) if math.dist(a, b) > 0.15 else 0.0
                q = tuple(a[k] + (b[k] - a[k]) * u for k in range(3))
                q = (q[0], q[1], q[2] + lift)
                self.pos[name] = q if u < 1 else b
                self.put(name, self.pos[name])
                if u >= 1:
                    self.tween.pop(name)

    def held(self, t):
        """20 Hz: arms pinned to robots, carried workpieces in the gripper, status LEDs on the mast."""
        n, sim = self.n, self.sim
        for r, ns in n.ns.items():
            p = n.map_pose(r)
            if p is None:
                continue
            c, s = math.cos(p[2]), math.sin(p[2])
            self.put(ns + "_arm", (p[0], p[1], 0.0), p[2])
            rb, b = sim.robots[r], sim.bat
            led = "r" if rb.fault else "b" if rb.phase in ("charging", "charge_queue") else "y" if (b and rb.battery is not None and rb.battery <= b["low"]) else "g"
            blink = (led == "r" and int(t * 3) % 2) or (led == "b" and rb.phase == "charging" and int(t * 1.5) % 2)
            for col in ("g", "b", "y", "r"):
                name = "led_{}_{}".format(ns, col)
                if col == led and not blink:
                    self.put(name, (p[0] - 0.17 * c, p[1] - 0.17 * s, LED_Z))
                else:
                    self.hide(name)
        if n.t0 is None:
            return
        for j in sim.jobs.values():
            h = n.where(j, t)
            if not (isinstance(h, tuple) and h[0] == "held"):
                continue
            p = n.map_pose(h[1])
            name = n.model(j)
            if p is None or name is None:
                continue
            c, s = math.cos(p[2]), math.sin(p[2])
            if self.tool(h[1]) == "tray":
                x, y = arm.TRAY_AT
                pos, yaw = (p[0] + c * x - s * y, p[1] + s * x + c * y, arm.TRAY_TOP), p[2]
            else:
                q = n.arm_q(h[1], t)
                x, y, z = arm.fk(q)
                pos, yaw = (p[0] + c * x - s * y, p[1] + s * x + c * y, z - arm.GRASP_Z), p[2] + q[0]
            self.pos[name] = pos
            self.tween.pop(name, None)
            self.put(name, pos, yaw)

    def tool(self, r):
        return self.tools[list(self.n.ns).index(r)]

    def handlers(self, t):
        """Material handlers: walk (along the aisle) to a docked tray robot, move the tray, stay there until needed again."""
        busy = {m["hand"][0]: m for m in self.n.motion.values() if m.get("hand")}
        for name in self.hand_at:
            m = busy.get(name)
            if m is None:
                x, y, yaw = self.hand_at[name]
                self.person(name, x, y, yaw, t, "idle")
                continue
            x, y, yaw, arrived = m["hand"][1].at(t)
            if arrived:
                x, y, yaw = m["hand"][2]
                self.hand_at[name] = (x, y, yaw)
            self.person(name, x, y, yaw, t, "work" if arrived and t >= m["t0"] else "walk" if not arrived else "idle")

    def hand_route(self, a, b):
        """People's route between two points via the aisle (robots do not collide with the figures)."""
        y = self.lay["aisle_y"]
        pts = [a[:2], (a[0], y), (b[0], y), b[:2]]
        return [p for i, p in enumerate(pts) if i == 0 or math.dist(p, pts[i - 1]) > 1e-3]

    def chargers(self, t):
        busy = {self.sim.robots[r].task[2] for r in self.sim.robots if self.sim.robots[r].phase == "charging" and self.sim.robots[r].task}
        for c, info in self.lay.get("chargers", {}).items():
            if c in busy and int(t * 1.5) % 2 == 0:
                self.put("chg_" + c, info["lamp"], info["yaw"])
            else:
                self.hide("chg_" + c)

    def update(self, t):
        """10 Hz."""
        self.machines(t)
        self.parts(t)
        if self.people:
            self.operators(t)
            self.technicians(t)
            self.handlers(t)
        self.chargers(t)
