"""Realistic Gazebo shop floor from a FactorySim config: CNC cells, a modular conveyor line, kitting / assembly / inspection
benches with operators, pallet racking, robot transfer counters, floor markings, lights; plus the matching Nav2 map and a
layout.json (docks, counter slots, work poses, conveyor paths, stack lights, part models) used by the ROS node. No ROS needed.

    python -m factory_sim.ros.world --config lab --out /tmp/factory_world
"""
import argparse
import json
import math
import os

import numpy as np

from ..config import Factory
from . import media as mediamod

MEDIA = None

RES, CLEAR, LANE, PITCH, CH, BELT, WALL_H, MARGIN = 0.05, 0.26, 0.09, 0.085, 0.19, 0.86, 2.6, 1.2
HIDDEN = (0.0, 0.0, -2.0)
C = {"floor": (0.52, 0.55, 0.55), "aisle": (0.40, 0.43, 0.46), "outside": (0.42, 0.42, 0.42), "wall": (0.90, 0.90, 0.87), "dado": (0.45, 0.55, 0.52), "col": (0.70, 0.70, 0.68), "ral7035": (0.84, 0.85, 0.82), "dark": (0.16, 0.17, 0.18), "dark2": (0.25, 0.26, 0.28),
     "alu": (0.75, 0.77, 0.79), "steel": (0.45, 0.47, 0.50), "blue": (0.05, 0.32, 0.62), "station_blue": (0.0, 0.42, 0.72), "yellow": (0.98, 0.78, 0.05), "black": (0.05, 0.05, 0.05), "white": (0.95, 0.95, 0.95), "wood": (0.74, 0.56, 0.36),
     "esd": (0.33, 0.48, 0.43), "orange": (0.95, 0.45, 0.05), "rack": (0.10, 0.25, 0.60), "card": (0.66, 0.50, 0.32), "skin": (0.85, 0.65, 0.50), "jeans": (0.15, 0.20, 0.35), "vest": (1.0, 0.50, 0.05), "red": (0.85, 0.08, 0.08),
     "green": (0.15, 0.60, 0.25), "lane_in": (0.15, 0.60, 0.25), "lane_out": (0.15, 0.35, 0.75), "glass": (0.55, 0.68, 0.78), "screen": (0.10, 0.40, 0.70)}
INERT = "<inertial><mass>0.1</mass><inertia><ixx>1e-4</ixx><iyy>1e-4</iyy><izz>1e-4</izz><ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia></inertial>"
KINDMAP = {}
LED = {"g": (0.1, 1.0, 0.25), "b": (0.15, 0.45, 1.0), "y": (1.0, 0.75, 0.05), "r": (1.0, 0.1, 0.1)}
LAMP = {"g": ((0.1, 0.35, 0.12), (0.1, 1.0, 0.2)), "y": ((0.45, 0.35, 0.05), (1.0, 0.75, 0.05)), "r": ((0.40, 0.07, 0.07), (1.0, 0.1, 0.1))}


class Model:
    """A static model built from primitives in a local frame (x away from the robot dock); collisions also become map footprints."""

    def __init__(self, name, x=0.0, y=0.0, yaw=0.0):
        self.name, self.x, self.y, self.yaw, self.items, self.foot, self.anim, self.sign_at = name, x, y, yaw, [], [], {}, None

    def w(self, lx, ly, lz=0.0):
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return [round(self.x + c * lx - s * ly, 4), round(self.y + s * lx + c * ly, 4), round(lz, 4)]

    def _vis(self, geo, pose, rgb, glow=False, alpha=0.0, size=None, image=None):
        if MEDIA is not None and image:
            self.items.append("<visual name='v{}'><pose>{}</pose><geometry>{}</geometry>{}</visual>".format(len(self.items), pose, geo, MEDIA.material((1, 1, 1), "screen", unlit_image=image)))
            return
        if MEDIA is not None and not alpha and not glow:
            if not KINDMAP:
                KINDMAP.update({tuple(v): mediamod.KIND.get(k, "paint") for k, v in C.items()})
            kind = KINDMAP.get(tuple(rgb), "paint")
            if size and min(size) >= 0.004 and (min(size) >= 0.012 or max(size) >= 0.3):
                geo = "<mesh><uri>file://{}</uri></mesh>".format(MEDIA.mesh(*size, tile=mediamod.TILE.get(kind, 0.6)))
            self.items.append("<visual name='v{}'><pose>{}</pose><geometry>{}</geometry>{}</visual>".format(len(self.items), pose, geo, MEDIA.material(rgb, kind)))
            return
        emi = "<emissive>{0} {1} {2} 1</emissive>".format(*rgb) if glow else ""
        tr = "<transparency>{}</transparency>".format(alpha) if alpha else ""
        self.items.append("<visual name='v{}'><pose>{}</pose><geometry>{}</geometry>{}<material><ambient>{} {} {} 1</ambient><diffuse>{} {} {} 1</diffuse><specular>0.25 0.25 0.25 1</specular>{}</material></visual>".format(len(self.items), pose, geo, tr, *rgb, *rgb, emi))

    def box(self, x0, x1, y0, y1, z0, z1, rgb, yaw=0.0, **k):
        size = (abs(x1 - x0), abs(y1 - y0), abs(z1 - z0))
        self._vis("<box><size>{:.4f} {:.4f} {:.4f}</size></box>".format(*size), "{:.4f} {:.4f} {:.4f} 0 0 {:.4f}".format((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2, yaw), rgb, size=size, **k)

    def cbox(self, cx, cy, cz, sx, sy, sz, rgb, yaw=0.0, roll=0.0, pitch=0.0, **k):
        self._vis("<box><size>{:.4f} {:.4f} {:.4f}</size></box>".format(sx, sy, sz), "{:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}".format(cx, cy, cz, roll, pitch, yaw), rgb, size=(sx, sy, sz), **k)

    def panel(self, x0, x1, y0, y1, z0, z1, name, image, rgb=(0.1, 0.4, 0.7)):
        """Flat screen / sign facing local -x with a texture (glowing plain colour without media)."""
        geo = "<box><size>{:.4f} {:.4f} {:.4f}</size></box>".format(abs(x1 - x0), abs(y1 - y0), abs(z1 - z0))
        pose = "{:.4f} {:.4f} {:.4f} 0 0 0".format((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2)
        if MEDIA is not None:
            self._vis(geo, pose, (1, 1, 1), image=MEDIA.image(name, image()))
        else:
            self._vis(geo, pose, rgb, glow=True)

    def cyl(self, x, y, z0, z1, r, rgb, **k):
        self._vis("<cylinder><radius>{}</radius><length>{:.4f}</length></cylinder>".format(r, z1 - z0), "{:.4f} {:.4f} {:.4f} 0 0 0".format(x, y, (z0 + z1) / 2), rgb, **k)

    def hcyl(self, x, y, z, r, length, rgb, yaw=0.0, **k):
        self._vis("<cylinder><radius>{}</radius><length>{:.4f}</length></cylinder>".format(r, length), "{:.4f} {:.4f} {:.4f} 0 1.5708 {:.4f}".format(x, y, z, yaw), rgb, **k)

    def sphere(self, x, y, z, r, rgb, **k):
        self._vis("<sphere><radius>{}</radius></sphere>".format(r), "{:.4f} {:.4f} {:.4f} 0 0 0".format(x, y, z), rgb, **k)

    def solid(self, x0, x1, y0, y1, h):
        """Invisible collision block (seen by the lidar) and the same rectangle in the Nav2 map."""
        self.items.append("<collision name='c{}'><pose>{:.4f} {:.4f} {:.4f} 0 0 0</pose><geometry><box><size>{:.4f} {:.4f} {:.4f}</size></box></geometry></collision>".format(len(self.items), (x0 + x1) / 2, (y0 + y1) / 2, h / 2, x1 - x0, y1 - y0, h))
        cx, cy, _ = self.w((x0 + x1) / 2, (y0 + y1) / 2)
        self.foot.append((cx, cy, self.yaw, x1 - x0, y1 - y0))

    def pose(self, lx, ly, lyaw=0.0):
        """World [x, y, yaw] of a local pose."""
        x, y, _ = self.w(lx, ly)
        return [x, y, round(self.yaw + lyaw, 4)]

    def sdf(self):
        return "<model name='{}'><static>true</static><pose>{:.4f} {:.4f} 0 0 0 {:.4f}</pose><link name='link'>{}</link></model>".format(self.name, self.x, self.y, self.yaw, "".join(self.items))


def line(m, a, b, w, rgb, z=0.002):
    d = math.hypot(b[0] - a[0], b[1] - a[1])
    if d > 1e-6:
        m.cbox((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, z, d, w, 0.003, rgb, yaw=math.atan2(b[1] - a[1], b[0] - a[0]))


def hatch(m, a, b, w=0.06, step=0.1):
    d = math.hypot(b[0] - a[0], b[1] - a[1])
    n = max(1, int(d / step))
    for i in range(n):
        p, q = [a[k] + (b[k] - a[k]) * i / n for k in (0, 1)], [a[k] + (b[k] - a[k]) * (i + 1) / n for k in (0, 1)]
        line(m, p, q, w, C["yellow"] if i % 2 == 0 else C["black"], z=0.003)


def rect(m, cx, cy, yaw, sx, sy, fn):
    c, s = math.cos(yaw), math.sin(yaw)
    pts = [(cx + c * x - s * y, cy + s * x + c * y) for x, y in ((-sx / 2, -sy / 2), (sx / 2, -sy / 2), (sx / 2, sy / 2), (-sx / 2, sy / 2))]
    for i in range(4):
        fn(m, pts[i], pts[(i + 1) % 4])


def stack_light(m, x, y, z):
    m.cyl(x, y, z, z + 0.12, 0.012, C["steel"])
    lamps = {}
    for i, c in enumerate("gyr"):
        m.cyl(x, y, z + 0.12 + 0.07 * i, z + 0.185 + 0.07 * i, 0.034, LAMP[c][0], alpha=0.15)
        lamps[c] = m.w(x, y, z + 0.1525 + 0.07 * i)
    m.cyl(x, y, z + 0.33, z + 0.355, 0.036, C["black"])
    return lamps


def operator(m, x, y, face):
    """Spot of a standing worker (an animated model, see person()) at local (x, y) facing local angle `face`."""
    m.anim["operator"] = m.pose(x, y, face)
    m.solid(x - 0.22, x + 0.22, y - 0.22, y + 0.22, 1.7)


def person(name, pose, vest, trousers, helmet, tool=None):
    """Kinematic worker model: links body, arm_l, arm_r (forearms pivot at the elbows, animated by the ROS node)."""
    b = Model("b")
    for l in (-0.09, 0.09):
        b.cyl(0, l, 0.06, 0.86, 0.065, trousers)
        b.cbox(0.04, l, 0.03, 0.26, 0.11, 0.06, C["black"])
    b.cbox(0, 0, 1.16, 0.24, 0.44, 0.62, vest)
    for z in (1.0, 1.25):
        b.cbox(0.005, 0, z, 0.245, 0.445, 0.035, C["alu"])
    for l in (-0.27, 0.27):
        b.cyl(0, l, 1.12, 1.44, 0.05, vest)
    b.cyl(0, 0, 1.46, 1.52, 0.05, C["skin"])
    b.sphere(0, 0, 1.6, 0.1, C["skin"])
    b.sphere(0, 0, 1.65, 0.115, helmet)
    b.cyl(0.03, 0, 1.6, 1.615, 0.14, helmet)
    arms = []
    for side, l in (("arm_l", 0.27), ("arm_r", -0.27)):
        a = Model("a")
        a.cbox(0.13, 0, 0, 0.26, 0.085, 0.085, vest)
        a.cbox(0.29, 0, 0, 0.07, 0.07, 0.07, C["skin"])
        if tool and side == "arm_r":
            a.cbox(0.3, 0, -0.14, 0.12, 0.3, 0.2, tool)
        arms.append("<link name='{}'><pose>0 {} 1.12 0 0 0</pose><gravity>false</gravity><kinematic>true</kinematic>{}{}</link>".format(side, l, INERT, "".join(a.items)))
    return "<model name='{}'><pose>{} {} 0 0 0 {}</pose><link name='body'><gravity>false</gravity><kinematic>true</kinematic>{}{}</link>{}</model>".format(name, pose[0], pose[1], pose[2], INERT, "".join(b.items), "".join(arms))


def counter(m, n):
    """Low AMR transfer counter in front of the station: IN lane (green, left) and OUT lane (blue, right)."""
    x0, x1, hw = CLEAR, CLEAR + max(0.45, 0.1 + PITCH * n), 0.32
    for x in (x0 + 0.02, x1 - 0.02):
        for y in (-hw + 0.02, hw - 0.02):
            m.box(x - 0.015, x + 0.015, y - 0.015, y + 0.015, 0, CH - 0.02, C["alu"])
    m.box(x0, x0 + 0.012, -hw, hw, 0.0, CH - 0.02, C["dark2"])
    m.box(x0, x1, -hw, hw, CH - 0.02, CH, C["dark"])
    for i in range(8):
        m.box(x0 - 0.004, x0 + 0.004, -hw + i * 0.08, -hw + (i + 1) * 0.08, CH - 0.045, CH, C["yellow"] if i % 2 == 0 else C["black"])
    for y, rgb in ((LANE, C["lane_in"]), (-LANE, C["lane_out"])):
        m.box(x0 + 0.015, x1 - 0.015, y - 0.045, y + 0.045, CH, CH + 0.003, rgb)
        for e in (-0.05, 0.05):
            m.box(x0 + 0.03, x1 - 0.015, y + e - 0.004, y + e + 0.004, CH, CH + 0.018, C["alu"])
    m.solid(x0, x1, -hw, hw, CH)
    slots = lambda y: [m.w(x0 + 0.05 + PITCH * i, y, CH + 0.003) for i in range(n)]
    return x1, slots(LANE), slots(-LANE)


def cnc(m, a):
    """Enclosed CNC machining centre with window, pendant and stack light; returns (work pose, lamps)."""
    D, W, H = 1.0, 1.1, 1.85
    m.box(a, a + D, -W / 2, W / 2, 0, 0.12, C["dark2"])
    m.box(a + D - 0.04, a + D, -W / 2, W / 2, 0.12, H, C["ral7035"])
    for y in (-W / 2, W / 2 - 0.04):
        m.box(a, a + D, y, y + 0.04, 0.12, H, C["ral7035"])
    m.box(a, a + D, -W / 2, W / 2, H - 0.05, H, C["ral7035"])
    m.box(a + 0.04, a + D - 0.04, -W / 2 + 0.04, W / 2 - 0.04, 0.12, 0.6, C["dark2"])
    wy0, wy1, wz0, wz1 = -0.25, 0.35, 0.75, 1.35
    m.box(a, a + 0.04, -W / 2, W / 2, 0.12, wz0, C["ral7035"])
    m.box(a, a + 0.04, -W / 2, W / 2, wz1, H, C["ral7035"])
    m.box(a, a + 0.04, wy1, W / 2, wz0, wz1, C["ral7035"])
    m.box(a, a + 0.04, -W / 2, wy0, wz0, wz1, C["ral7035"])
    m.box(a - 0.004, a, wy0 - 0.04, wy1 + 0.04, wz0 - 0.04, wz0 - 0.02, C["alu"])
    m.box(a - 0.004, a, wy0 - 0.04, wy1 + 0.04, wz1 + 0.02, wz1 + 0.04, C["alu"])
    m.box(a - 0.006, a, -W / 2, W / 2, H - 0.26, H - 0.13, C["blue"])
    m.box(a + 0.25, a + 0.75, -0.15, 0.3, 0.6, 0.72, C["steel"])
    m.box(a + 0.48, a + 0.62, 0.02, 0.18, 0.72, 0.77, C["dark"])
    m.box(a + 0.62, a + 0.72, -0.05, 0.25, 1.38, H - 0.05, C["dark2"])
    m.box(a + 0.15, a + 0.25, -0.2, 0.3, 0.6, 0.62, (0.3, 0.55, 0.75), alpha=0.3)
    mid = (wy0 + wy1) / 2
    m.anim["door"] = {"closed": m.pose(a - 0.014, mid), "open": m.pose(a - 0.014, mid - 0.44), "size": [wy1 - wy0, (wz0 + wz1) / 2, wz1 - wz0]}
    m.anim["spindle"] = m.pose(a + 0.55, 0.1)
    m.sign_at = (a - 0.012, 0.0, H - 0.195, 0.9, 0.12)
    m.anim["service"] = m.pose(a + 0.5, -W / 2 - 0.5, math.pi / 2)
    py = -W / 2 + 0.02
    m.box(a - 0.09, a, py, py + 0.3, 1.0, 1.45, C["dark"])
    m.panel(a - 0.093, a - 0.09, py + 0.04, py + 0.26, 1.24, 1.42, "hmi_" + m.name, lambda: mediamod.screen(m.name[3:] + "  CNC", np.random.default_rng(len(m.name))))
    m.box(a - 0.093, a - 0.09, py + 0.04, py + 0.26, 1.04, 1.2, C["dark2"])
    m.hcyl(a - 0.1, py + 0.24, 1.08, 0.022, 0.02, C["red"], yaw=0.0)
    m.box(a + 0.3, a + 0.7, -0.2, 0.2, H, H + 0.04, C["white"])
    m.solid(a, a + D, -W / 2, W / 2, H)
    return m.w(a + 0.55, 0.1, 0.77), stack_light(m, a + 0.15, -W / 2 + 0.12, H)


def module(m, cx, belt=True):
    """Modular training-line station: aluminium-profile table, pneumatic press unit, valve manifold, HMI, stack light."""
    D, W, Ht = 0.6, 0.8, BELT - 0.02
    for x in (cx - D / 2 + 0.03, cx + D / 2 - 0.03):
        for y in (-W / 2 + 0.03, W / 2 - 0.03):
            m.box(x - 0.02, x + 0.02, y - 0.02, y + 0.02, 0.02, Ht, C["alu"])
            m.cyl(x, y, 0, 0.02, 0.03, C["black"])
    m.box(cx - D / 2, cx + D / 2, -W / 2, W / 2, 0.14, 0.16, C["steel"])
    m.box(cx - 0.2, cx + 0.05, -0.3, -0.05, 0.16, 0.36, C["station_blue"])
    m.box(cx + 0.08, cx + 0.22, 0.1, 0.32, 0.16, 0.22, C["dark2"])
    m.box(cx - D / 2, cx + D / 2, -W / 2, W / 2, Ht - 0.03, Ht, C["alu"])
    if belt:
        m.box(cx - D / 2, cx + D / 2, -0.08, 0.08, Ht, BELT, C["dark"])
    by = W / 2 - 0.06
    for y in (-by, by):
        m.box(cx + D / 2 - 0.08, cx + D / 2 - 0.04, y - 0.02, y + 0.02, Ht, Ht + 0.5, C["alu"])
    m.box(cx + D / 2 - 0.1, cx + D / 2 - 0.02, -by - 0.02, by + 0.02, Ht + 0.45, Ht + 0.53, C["alu"])
    m.box(cx - 0.05, cx + D / 2 - 0.04, -0.04, 0.04, Ht + 0.45, Ht + 0.5, C["alu"])
    m.cyl(cx, 0, Ht + 0.22, Ht + 0.45, 0.025, C["alu"])
    m.cyl(cx, 0, Ht + 0.42, Ht + 0.45, 0.03, C["station_blue"])
    m.anim["press"] = m.pose(cx, 0.0)
    m.sign_at = (cx - D / 2 - 0.006, 0.0, Ht - 0.08, 0.5, 0.08)
    m.anim["service"] = m.pose(cx, -W / 2 - 0.45, math.pi / 2)
    m.box(cx + D / 2 - 0.12, cx + D / 2 - 0.02, -0.15, 0.15, Ht + 0.53, Ht + 0.6, C["station_blue"])
    m.box(cx + D / 2 - 0.12, cx + D / 2 - 0.1, by - 0.2, by - 0.03, Ht + 0.2, Ht + 0.34, C["dark"])
    m.panel(cx + D / 2 - 0.123, cx + D / 2 - 0.12, by - 0.19, by - 0.04, Ht + 0.21, Ht + 0.33, "hmi_" + m.name, lambda: mediamod.screen(m.name[3:] + "  LINE", np.random.default_rng(len(m.name) + 3)))
    m.solid(cx - D / 2, cx + D / 2, -W / 2, W / 2, Ht)
    return m.w(cx, 0.0, BELT + 0.002), stack_light(m, cx + D / 2 - 0.06, -by, Ht + 0.53)


def bench(m, a, D, W, top, legs=C["blue"]):
    for x in (a + 0.04, a + D - 0.04):
        for y in (-W / 2 + 0.04, W / 2 - 0.04):
            m.box(x - 0.025, x + 0.025, y - 0.025, y + 0.025, 0, 0.86, legs)
    m.box(a + 0.04, a + D - 0.04, -W / 2 + 0.04, W / 2 - 0.04, 0.18, 0.2, C["steel"])
    m.box(a, a + D, -W / 2, W / 2, 0.86, 0.9, top)
    m.solid(a, a + D, -W / 2, W / 2, 0.9)


def kitting(m, a, colors):
    D, W = 0.7, 1.6
    bench(m, a, D, W, C["wood"])
    for i, y in enumerate((-0.45, -0.1, 0.25)):
        m.box(a + 0.15, a + 0.45, y - 0.13, y + 0.13, 0.9, 0.94, C["dark2"])
    m.box(a + 0.5, a + 0.62, 0.45, 0.7, 0.9, 1.08, C["dark"], yaw=0.0)
    m.panel(a + 0.503, a + 0.507, 0.47, 0.68, 0.92, 1.06, "hmi_kitting", lambda: mediamod.screen("KIT LIST", np.random.default_rng(11)))
    s0 = a + D + 0.05
    for y in (-W / 2, -0.01, W / 2 - 0.03):
        m.box(s0, s0 + 0.04, y, y + 0.04, 0, 1.9, C["steel"])
        m.box(s0 + 0.36, s0 + 0.4, y, y + 0.04, 0, 1.9, C["steel"])
    for k, z in enumerate((0.3, 0.75, 1.2, 1.65)):
        m.box(s0, s0 + 0.4, -W / 2, W / 2, z - 0.03, z, C["steel"])
        for i in range(6):
            y = -W / 2 + 0.08 + i * 0.25
            col = colors[(i + k) % len(colors)]
            m.box(s0 + 0.04, s0 + 0.36, y, y + 0.22, z, z + 0.16, col)
            m.box(s0 + 0.035, s0 + 0.04, y + 0.06, y + 0.16, z + 0.08, z + 0.13, C["white"])
    m.solid(s0, s0 + 0.4, -W / 2, W / 2, 1.9)
    m.sign_at = (s0 - 0.006, 0.0, 2.05, 1.1, 0.22)
    operator(m, a + 0.3, W / 2 + 0.45, -math.pi / 2)
    return m.w(a + 0.3, 0.0, 0.94)


def assembly(m, a):
    D, W = 0.75, 1.5
    bench(m, a, D, W, C["esd"], legs=C["steel"])
    for y in (-W / 2 + 0.02, W / 2 - 0.06):
        m.box(a + D - 0.06, a + D - 0.02, y, y + 0.04, 0.9, 1.95, C["steel"])
    m.box(a + D - 0.08, a + D - 0.02, -W / 2, W / 2, 1.88, 1.95, C["steel"])
    m.box(a + D - 0.2, a + D - 0.06, -W / 2 + 0.1, W / 2 - 0.1, 1.84, 1.88, C["white"], glow=True)
    m.box(a + D - 0.06, a + D - 0.04, -W / 2, W / 2, 1.3, 1.34, C["alu"])
    for i in range(7):
        m.cyl(a + D - 0.08, -0.6 + 0.2 * i, 1.12, 1.3, 0.012 + 0.004 * (i % 3), C["red"] if i % 2 else C["dark"])
    for i in range(5):
        y = -0.6 + 0.3 * i
        m.box(a + D - 0.24, a + D - 0.08, y - 0.12, y + 0.12, 0.9, 1.02, C["blue"])
    m.box(a + 0.3, a + 0.46, -0.08, 0.08, 0.9, 0.93, C["dark"])
    m.box(a + D - 0.14, a + D - 0.1, 0.35, 0.75, 1.15, 1.42, C["dark"])
    m.panel(a + D - 0.143, a + D - 0.14, 0.37, 0.73, 1.17, 1.4, "hmi_" + m.name, lambda: mediamod.screen("WORK INSTR.", np.random.default_rng(len(m.name) + 7)))
    operator(m, a + 0.35, W / 2 + 0.45, -math.pi / 2)
    m.sign_at = (a + D - 0.086, 0.0, 2.08, 1.0, 0.2)
    return m.w(a + 0.38, 0.0, 0.93)


def inspection(m, a):
    D, W = 0.7, 0.9
    bench(m, a, D, W, C["dark2"], legs=C["steel"])
    for y in (-0.38, 0.34):
        m.box(a + 0.3, a + 0.34, y, y + 0.04, 0.9, 1.55, C["alu"])
    m.box(a + 0.28, a + 0.36, -0.4, 0.4, 1.5, 1.56, C["alu"])
    m.box(a + 0.26, a + 0.38, -0.06, 0.06, 1.38, 1.5, C["dark"])
    m.cyl(a + 0.32, 0, 1.3, 1.38, 0.025, C["black"])
    m.cyl(a + 0.32, 0, 1.22, 1.235, 0.09, C["white"], glow=True)
    m.box(a + 0.55, a + 0.6, 0.15, 0.42, 1.0, 1.22, C["dark"])
    m.panel(a + 0.547, a + 0.55, 0.17, 0.4, 1.02, 1.2, "hmi_inspect", lambda: mediamod.screen("VISION  OK", np.random.default_rng(5), accent=(60, 220, 90)))
    m.box(a + 0.1, a + 0.5, W / 2 + 0.05, W / 2 + 0.45, 0, 0.55, C["red"])
    m.box(a + 0.12, a + 0.48, W / 2 + 0.07, W / 2 + 0.43, 0.55, 0.56, C["dark"])
    m.solid(a + 0.1, a + 0.5, W / 2 + 0.05, W / 2 + 0.45, 0.55)
    m.sign_at = (a + 0.274, 0.0, 1.66, 0.7, 0.14)
    return m.w(a + 0.32, 0.0, 0.9)


def flow_rack(m, a):
    D, W = 0.6, 1.1
    for x in (a, a + D - 0.04):
        for y in (-W / 2, W / 2 - 0.04):
            m.box(x, x + 0.04, y, y + 0.04, 0, 1.4, C["rack"])
    for z in (0.35, 0.8, 1.25):
        m.box(a, a + D, -W / 2, W / 2, z - 0.03, z, C["orange"])
        for i in range(3):
            y = -W / 2 + 0.08 + 0.34 * i
            m.box(a + 0.05, a + D - 0.05, y, y + 0.28, z, z + 0.2, C["dark2"] if i % 2 else C["steel"])
    m.solid(a, a + D, -W / 2, W / 2, 1.4)
    m.sign_at = (a - 0.006, 0.0, 1.55, 0.8, 0.16)


def pallet_rack(m, a, n):
    D, W = 1.0, 2.4
    for x in (a, a + D - 0.06):
        for y in (-W / 2, -0.03, W / 2 - 0.06):
            m.box(x, x + 0.06, y, y + 0.06, 0, 2.3, C["rack"])
    for z in (0.15, 1.1, 2.05):
        for x in (a, a + D - 0.05):
            m.box(x, x + 0.05, -W / 2, W / 2, z, z + 0.1, C["orange"])
    for z in (1.2, 2.15):
        for y in (-W / 2 + 0.1, 0.05):
            m.box(a + 0.05, a + D - 0.05, y, y + 1.0, z, z + 0.14, C["wood"])
            m.box(a + 0.12, a + D - 0.12, y + 0.08, y + 0.92, z + 0.14, z + 0.5 + 0.1 * (y > 0), C["card"])
    for y in (-W / 2 + 0.1, 0.05):
        m.box(a + 0.05, a + D - 0.05, y, y + 1.0, 0.25, 0.39, C["wood"])
        m.box(a + 0.1, a + D - 0.1, y + 0.05, y + 0.95, 0.39, 0.41, C["dark2"])
    m.solid(a, a + D, -W / 2, W / 2, 2.3)
    m.sign_at = (a - 0.006, 0.0, 2.5, 1.4, 0.26)
    return [m.w(a + 0.16 + 0.11 * ((i // 12) % 7), -W / 2 + 0.17 + 0.18 * (i % 12) + 0.1 * (i % 12 >= 6), 0.41) for i in range(n)]


def charger(m):
    m.box(0.3, 0.45, -0.2, 0.2, 0, 0.42, C["dark"])
    m.box(0.296, 0.3, -0.15, 0.15, 0.3, 0.33, (0.08, 0.25, 0.1))
    m.box(0.296, 0.3, -0.16, -0.1, 0.36, 0.4, C["white"])
    m.anim["lamp"] = m.w(0.293, 0.0, 0.315)
    for y in (-0.08, 0.08):
        m.box(0.27, 0.3, y - 0.03, y + 0.03, 0.06, 0.1, (0.8, 0.5, 0.2))
    m.box(-0.2, 0.3, -0.25, 0.25, 0, 0.006, C["dark2"])
    m.solid(0.3, 0.45, -0.2, 0.2, 0.42)


def conveyor(m, pts, belt_w=0.16):
    """Belt conveyor along a Manhattan polyline at belt height, with side rails and legs."""
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        L, yaw = math.hypot(x1 - x0, y1 - y0), math.atan2(y1 - y0, x1 - x0)
        if L < 1e-6:
            continue
        cx, cy, c, s = (x0 + x1) / 2, (y0 + y1) / 2, math.cos(yaw), math.sin(yaw)
        m.cbox(cx, cy, BELT - 0.03, L + belt_w, belt_w + 0.06, 0.06, C["alu"], yaw=yaw)
        m.cbox(cx, cy, BELT - 0.001, L + belt_w, belt_w, 0.004, C["dark"], yaw=yaw)
        for e in (-1, 1):
            m.cbox(cx - s * e * (belt_w / 2 + 0.03), cy + c * e * (belt_w / 2 + 0.03), BELT + 0.02, L + belt_w, 0.02, 0.04, C["alu"], yaw=yaw)
        for k in range(int(L / 0.8) + 2):
            f = min(k * 0.8, L) / L
            px, py = x0 + (x1 - x0) * f, y0 + (y1 - y0) * f
            for e in (-1, 1):
                m.cbox(px - s * e * 0.1, py + c * e * 0.1, (BELT - 0.06) / 2, 0.04, 0.04, BELT - 0.06, C["alu"], yaw=yaw)
        m.items.append("<collision name='c{}'><pose>{:.4f} {:.4f} {:.4f} 0 0 {:.4f}</pose><geometry><box><size>{:.4f} {:.4f} {:.4f}</size></box></geometry></collision>".format(len(m.items), cx, cy, BELT / 2, yaw, L + belt_w, belt_w + 0.06, BELT))
        m.foot.append((cx, cy, yaw, L + belt_w, belt_w + 0.06))


def manhattan(a, b):
    mx = (a[0] + b[0]) / 2
    pts = [a[:2], [mx, a[1]], [mx, b[1]], b[:2]]
    return [p for i, p in enumerate(pts) if i == 0 or math.hypot(p[0] - pts[i - 1][0], p[1] - pts[i - 1][1]) > 1e-6]


def build(F, parts=()):
    """-> (list of Model, layout dict). parts: [(model name, product)] dynamic workpieces to pre-place (hidden)."""
    models, lay = [], {"docks": {n: [p[0], p[1], 0.0] for n, p in F.nodes.items()}, "stations": {}, "conveyors": {}, "hidden": list(HIDDEN)}
    conv = {s for k in F.transfers for s in k}
    colors = [tuple(c / 255 for c in p.get("color", (150, 150, 150))) for p in F.products.values()]
    snode = {s["node"] for s in F.stations.values()}
    aisle = [p for n, p in F.nodes.items() if n not in snode and len(F.adj[n]) >= 2]
    aisle_y = sum(p[1] for p in aisle) / len(aisle) if aisle else 0.0
    frame = {}
    for name, s in F.stations.items():
        x, y = F.nodes[s["node"]]
        nb = list(F.adj[s["node"]])
        if nb:
            nx, ny = F.nodes[nb[0]]
            yaw = math.atan2(y - ny, x - nx)
            lay["docks"][s["node"]] = [x, y, yaw]
        frame[name] = (x, y, yaw if nb else 0.0, bool(nb))
    for name, s in F.stations.items():
        x, y, yaw, docked = frame[name]
        m, info, kind = Model("st_" + name, x, y, yaw), {}, s["kind"]
        n = min(6, max(2, s.get("slots_in", 1), s.get("slots_out", 1)))
        a = 0.0
        if docked:
            a, info["in"], info["out"] = counter(m, n)
            a += 0.06
        if name in conv or kind == "port":
            work, lamps = module(m, a + 0.3 if docked else 0.0)
        elif kind == "machine":
            work, lamps = cnc(m, a)
        elif kind == "kitting":
            work, lamps = kitting(m, a, colors), stack_light(m, a + 0.05, -0.75, 0.9)
        elif kind == "assembly":
            work, lamps = assembly(m, a), stack_light(m, a + 0.7, -0.7, 1.95)
        elif kind == "inspection":
            work, lamps = inspection(m, a), stack_light(m, a + 0.32, 0.38, 1.56)
        elif kind == "buffer":
            flow_rack(m, a)
            work, lamps = None, {}
        elif kind == "sink":
            lay["store"] = pallet_rack(m, a, max(8, len(parts)))
            work, lamps = None, {}
        else:
            work, lamps = m.w(a + 0.3, 0, 0.5), {}
            m.box(a, a + 0.6, -0.3, 0.3, 0, 0.5, C["steel"])
            m.solid(a, a + 0.6, -0.3, 0.3, 0.5)
        if not docked and work:
            info["in"] = [m.w(-0.18 - PITCH * i, 0.0, BELT + 0.002) for i in range(min(6, max(1, s.get("slots_in", 1))))]
            info["out"] = [m.w(0.18 + PITCH * i, 0.0, BELT + 0.002) for i in range(min(6, max(1, s.get("slots_out", 1))))]
        if m.sign_at:
            x0s, y0s, zs, ws, hs = m.sign_at
            sub = s.get("label") or {"machine": "CONVEYOR STATION" if (name in conv or kind == "port") else "CNC MACHINE", "port": "LINE OUTPUT", "kitting": "KIT PREPARATION", "assembly": "MANUAL ASSEMBLY", "inspection": "QUALITY CHECK", "buffer": "OVERFLOW BUFFER", "sink": "FINISHED GOODS"}.get(kind, kind.upper())
            title = (s.get("tag") or name).replace("ASSEMBLY STAGE", "ASSEMBLY")
            m.panel(x0s - 0.006, x0s, y0s - ws / 2, y0s + ws / 2, zs - hs / 2, zs + hs / 2, "sign_" + name, lambda t=title, u=sub: mediamod.sign(t, u))
        info["work"], info["lamps"], info["kind"] = work, lamps, kind
        info.update(m.anim)
        info["row"] = "south" if (work or [x, y])[1] < aisle_y else "north"
        lay["stations"][name] = info
        models.append(m)
    belt = Model("smc_conveyors")
    for (a, b) in F.transfers:
        pa, pb = lay["stations"][a]["work"], lay["stations"][b]["work"]
        pts = manhattan(pa, pb)
        conveyor(belt, pts)
        lay["conveyors"]["{}|{}".format(a, b)] = [[p[0], p[1], BELT + 0.002] for p in pts]
    if F.transfers:
        models.append(belt)
    lay["chargers"] = {}
    for c in (F.chargers or [n for n in F.nodes if n.startswith("charger")]):
        if c in snode:
            continue
        x, y = F.nodes[c]
        nb = list(F.adj[c])
        yaw = math.atan2(y - F.nodes[nb[0]][1], x - F.nodes[nb[0]][0]) if nb else 0.0
        m = Model("charging_station_" + c, x, y, yaw)
        charger(m)
        models.append(m)
        lay["docks"][c] = [x, y, yaw]
        lay["chargers"][c] = {"lamp": m.anim["lamp"], "yaw": yaw}
    lay["aisle_y"] = aisle_y
    return models, lay


def maintenance_bay(bounds):
    """Technicians' bay against the west wall: tool cabinet, workbench, spare-parts shelf, hatched floor zone."""
    x0, y0, x1, y1 = bounds
    ym = (y0 + y1) / 2 + 1.6
    m = Model("maintenance_bay", x0, ym, 0.0)
    m.box(0.02, 0.47, -0.55, 0.05, 0, 1.6, C["red"])
    for z in (0.3, 0.6, 0.9, 1.2):
        m.box(0.47, 0.475, -0.53, 0.03, z, z + 0.02, C["alu"])
    m.box(0.02, 0.62, 0.15, 1.05, 0.85, 0.9, C["wood"])
    for x in (0.05, 0.57):
        for y in (0.18, 1.0):
            m.box(x - 0.02, x + 0.02, y - 0.02, y + 0.02, 0, 0.85, C["steel"])
    m.box(0.1, 0.3, 0.3, 0.5, 0.9, 1.05, C["orange"])
    m.box(0.05, 0.25, 0.7, 0.95, 0.9, 0.98, C["dark2"])
    m.solid(0.0, 0.62, -0.55, 1.05, 1.6)
    rect(m, x0 + 0.5, ym + 0.25, 0.0, 1.0, 1.9, hatch)
    return m, {"home": [round(x0 + 0.9, 3), round(ym + 0.25, 3), math.pi], "exit": [round(x0 + 0.45, 3), round((y0 + y1) / 2, 3)]}


def markings(F, models, lay, bounds):
    """Floor: epoxy coat, yellow aisle lines, hatched robot docks, parking boxes, walkway zones and machine outlines."""
    m = Model("floor_markings")
    snode = {s["node"] for s in F.stations.values()}
    aisle = {n for n in F.nodes if n not in snode and len(F.adj[n]) >= 2}
    spurs = [(a, b) for a in F.nodes for b in F.adj[a] if a in aisle and b not in aisle]
    for a in F.nodes:
        for b in F.adj[a]:
            if a < b and a in aisle and b in aisle:
                (x0, y0), (x1, y1) = F.nodes[a], F.nodes[b]
                L = math.hypot(x1 - x0, y1 - y0)
                ux, uy = (x1 - x0) / L, (y1 - y0) / L
                ex0, ex1 = (0.7 if len([c for c in F.adj[a] if c in aisle]) == 1 else 0.0), (0.7 if len([c for c in F.adj[b] if c in aisle]) == 1 else 0.0)
                m.cbox((x0 + x1) / 2 + ux * (ex1 - ex0) / 2, (y0 + y1) / 2 + uy * (ex1 - ex0) / 2, 0.0008, L + ex0 + ex1, 1.4, 0.0016, C["aisle"], yaw=math.atan2(uy, ux))
                for e in (-0.7, 0.7):
                    p, q = (x0 - ux * ex0 - uy * e, y0 - uy * ex0 + ux * e), (x1 + ux * ex1 - uy * e, y1 + uy * ex1 + ux * e)
                    cuts = []
                    for (sa, sb) in spurs:
                        (ax, ay), (bx, by) = F.nodes[sa], F.nodes[sb]
                        dx, dy = q[0] - p[0], q[1] - p[1]
                        den = dx * (by - ay) - dy * (bx - ax)
                        if abs(den) < 1e-9:
                            continue
                        t = ((ax - p[0]) * (by - ay) - (ay - p[1]) * (bx - ax)) / den
                        v = ((ax - p[0]) * dy - (ay - p[1]) * dx) / den
                        if 0 <= t <= 1 and 0 <= v <= 1:
                            cuts.append(t)
                    Lq, segs, t0 = math.hypot(q[0] - p[0], q[1] - p[1]), [], 0.0
                    for t in sorted(cuts):
                        segs.append((t0, max(t0, t - 0.55 / Lq)))
                        t0 = t + 0.55 / Lq
                    segs.append((t0, 1.0))
                    for s0, s1 in segs:
                        if s1 > s0:
                            line(m, (p[0] + (q[0] - p[0]) * s0, p[1] + (q[1] - p[1]) * s0), (p[0] + (q[0] - p[0]) * s1, p[1] + (q[1] - p[1]) * s1), 0.06, C["yellow"])
    for n, (x, y, yaw) in lay["docks"].items():
        if n in snode or n == "charger":
            rect(m, x, y, yaw, 0.55, 0.5, hatch)
    for n in F.robots.get("parking", []):
        x, y = F.nodes[n]
        rect(m, x, y, 0.0, 0.5, 0.5, lambda mm, a, b: line(mm, a, b, 0.04, C["white"]))
    for st in models:
        for (cx, cy, yaw, sx, sy) in st.foot:
            if sx * sy > 0.3 and st.name.startswith("st_"):
                rect(m, cx, cy, yaw, sx + 0.2, sy + 0.2, lambda mm, a, b: line(mm, a, b, 0.03, C["yellow"], z=0.0015))
    x0, y0, x1, y1 = bounds
    m.cbox((x0 + x1) / 2, (y0 + y1) / 2, -0.004, x1 - x0, y1 - y0, 0.008, C["floor"])
    m.cbox((x0 + x1) / 2, (y0 + y1) / 2, -0.012, x1 - x0 + 12, y1 - y0 + 12, 0.008, C["outside"])
    return m


def walls(bounds):
    x0, y0, x1, y1 = bounds
    m, t = Model("building"), 0.15
    for (a, b, c, d) in ((x0 - t, x1 + t, y0 - t, y0), (x0 - t, x1 + t, y1, y1 + t), (x0 - t, x0, y0, y1), (x1, x1 + t, y0, y1)):
        m.box(a, b, c, d, 0, 1.2, C["dado"])
        m.box(a, b, c, d, 1.2, WALL_H, C["wall"])
        m.solid(a, b, c, d, WALL_H)
    cols = [(x, y) for x in [x0 + i * (x1 - x0) / max(1, round((x1 - x0) / 5)) for i in range(round((x1 - x0) / 5) + 1)] for y in (y0, y1)]
    cols += [(x, y) for y in [y0 + i * (y1 - y0) / max(1, round((y1 - y0) / 5)) for i in range(1, round((y1 - y0) / 5))] for x in (x0, x1)]
    for (x, y) in cols:
        cx, cy = min(max(x, x0 + 0.18), x1 - 0.18), min(max(y, y0 + 0.18), y1 - 0.18)
        m.box(cx - 0.18, cx + 0.18, cy - 0.18, cy + 0.18, 0, WALL_H + 0.2, C["col"])
        for k in range(5):
            m.box(cx - 0.185, cx + 0.185, cy - 0.185, cy + 0.185, k * 0.2, k * 0.2 + 0.1, C["yellow"] if k % 2 == 0 else C["black"])
        m.solid(cx - 0.18, cx + 0.18, cy - 0.18, cy + 0.18, WALL_H)
    m.box(x0 - t - 0.01, x0 - t, (y0 + y1) / 2 - 1.5, (y0 + y1) / 2 + 1.5, 0, 2.4, C["steel"])
    for k in range(12):
        m.box(x0 - t - 0.02, x0 - t - 0.01, (y0 + y1) / 2 - 1.5, (y0 + y1) / 2 + 1.5, k * 0.2, k * 0.2 + 0.02, C["dark2"])
    return m


def dynamic(name, visuals, x, y, z):
    """Kinematic, collision-free model moved by /gazebo/set_entity_state (workpieces, stack-light lamps)."""
    return "<model name='{}'><pose>{} {} {} 0 0 0</pose><link name='link'><gravity>false</gravity><kinematic>true</kinematic><inertial><mass>0.05</mass><inertia><ixx>1e-5</ixx><iyy>1e-5</iyy><izz>1e-5</izz><ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia></inertial>{}</link></model>".format(name, x, y, z, visuals)


def part_visuals(rgb):
    m = Model("p")
    m.cbox(0, 0, 0.005, 0.07, 0.06, 0.01, C["alu"])
    m.cbox(0, 0, 0.0275, 0.04, 0.04, 0.035, rgb)
    m.cyl(0, 0, 0.045, 0.057, 0.012, C["alu"])
    return "".join(m.items)


def lamp_visuals(rgb):
    m = Model("l")
    m.cyl(0, 0, -0.034, 0.034, 0.037, rgb, glow=True)
    return "".join(m.items)


def footprints(models):
    return [f for m in models for f in m.foot]


def bounds_of(F, models):
    pts = [p for p in F.nodes.values()]
    for (cx, cy, yaw, sx, sy) in footprints(models):
        c, s = math.cos(yaw), math.sin(yaw)
        pts += [(cx + c * x - s * y, cy + s * x + c * y) for x in (-sx / 2, sx / 2) for y in (-sy / 2, sy / 2)]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return (round(min(xs) - MARGIN, 2), round(min(ys) - MARGIN, 2), round(max(xs) + MARGIN, 2), round(max(ys) + MARGIN, 2))


def sdf(F, parts=(), record=None, fps=1.0, mesh_dir=None, people=True):
    """-> (world SDF text, layout, map footprints, bounds). people=False leaves out the human figures (operators,
    technicians, handlers); the simulated people still work, they are just not drawn."""
    models, lay = build(F, parts)
    bounds = bounds_of(F, models)
    models += [walls(bounds), markings(F, models, lay, bounds)]
    x0, y0, x1, y1 = bounds
    lights = "".join("<light name='lamp{}{}' type='point'><pose>{:.2f} {:.2f} 3.4 0 0 0</pose><diffuse>0.12 0.12 0.115 1</diffuse><specular>0.02 0.02 0.02 1</specular><attenuation><range>30</range><constant>0.9</constant><linear>0.01</linear><quadratic>0.0</quadratic></attenuation><cast_shadows>false</cast_shadows></light>".format(i, j, x0 + (x1 - x0) * (i + 0.5) / 4, y0 + (y1 - y0) * (j + 0.5) / 2) for i in range(4) for j in range(2))
    sun = "<light name='sun' type='directional'><cast_shadows>true</cast_shadows><pose>0 0 10 0 0 0</pose><diffuse>0.62 0.61 0.58 1</diffuse><specular>0.15 0.15 0.15 1</specular><direction>-0.35 0.25 -0.9</direction></light>"
    floor = "<model name='ground'><static>true</static><link name='link'><collision name='c'><geometry><plane><normal>0 0 1</normal><size>200 200</size></plane></geometry><surface><friction><ode><mu>100</mu><mu2>50</mu2></ode></friction></surface></collision></link></model>"
    dyn = []
    for name, product in parts:
        dyn.append(dynamic(name, part_visuals(tuple(c / 255 for c in F.products[product].get("color", (150, 150, 150)))), *HIDDEN))
    for s, info in lay["stations"].items():
        for c, p in info["lamps"].items():
            dyn.append(dynamic("lamp_{}_{}".format(s, c), lamp_visuals(LAMP[c][1]), *HIDDEN))
        if "door" in info:
            dyn.append(posed("door_" + s, door_visuals(*info["door"]["size"]), info["door"]["closed"]))
        if "spindle" in info:
            dyn.append(posed("spindle_" + s, spindle_visuals(), info["spindle"]))
        if "press" in info:
            dyn.append(posed("press_" + s, press_visuals(), info["press"]))
        if "operator" in info and people:
            dyn.append(person("op_" + s, info["operator"], C["vest"], C["jeans"], C["yellow"]))
    for c, info in lay["chargers"].items():
        dyn.append(dynamic("chg_" + c, "<visual name='v'><geometry><box><size>0.012 0.3 0.035</size></box></geometry><material><ambient>0.1 1 0.3 1</ambient><diffuse>0.1 1 0.3 1</diffuse><emissive>0.1 1 0.3 1</emissive></material></visual>", *HIDDEN))
    bay, lay["bay"] = maintenance_bay(bounds)
    models.append(bay)
    crew = F.operators.get(F.technician, 1) if F.technician else 0     # no technician in the config: nobody in the bay
    lay["technicians"] = ["tech_{}".format(i) for i in range(crew)]
    for i, n in enumerate(lay["technicians"]):
        h = lay["bay"]["home"]
        if people:
            dyn.append(person(n, [h[0], h[1] - 0.6 * i, h[2]], (0.12, 0.28, 0.6), (0.12, 0.28, 0.6), C["white"], tool=C["red"]))
    lay["people"] = bool(people)
    lay["handlers"], lay["handler_home"] = [], []
    if "handler" in F.handling:   # material handlers who move trays on and off tray robots (hi-vis orange vests)
        k = F.operators[F.handler_pool]
        for i in range(k):
            h = [round(x0 + 1.1, 3), round(lay["aisle_y"] + 0.7 * (i - (k - 1) / 2), 3), 0.0]
            lay["handlers"].append("handler_{}".format(i))
            lay["handler_home"].append(h)
            if people:
                dyn.append(person("handler_{}".format(i), h, (0.98, 0.45, 0.08), (0.2, 0.22, 0.3), C["white"]))
    for i in range(F.robots["count"]):
        for c, rgb in LED.items():
            dyn.append(dynamic("led_robot{}_{}".format(i + 1, c), "<visual name='v'><geometry><sphere><radius>0.022</radius></sphere></geometry><material><ambient>{0} {1} {2} 1</ambient><diffuse>{0} {1} {2} 1</diffuse><emissive>{0} {1} {2} 1</emissive></material></visual>".format(*rgb), *HIDDEN))
    lay["corridors"] = {"south": round(y0 + 0.6, 3), "north": round(y1 - 0.6, 3), "west": round(x0 + 0.45, 3)}
    lay["exit"] = [round(x0 + 0.3, 3), round((y0 + y1) / 2, 3)]
    cam = "<gui fullscreen='0'><camera name='user_camera'><pose>{:.2f} {:.2f} 11 0 0.85 1.5708</pose></camera></gui>".format((x0 + x1) / 2, y0 - 5.5)
    plugin = "<plugin name='gazebo_ros_state' filename='libgazebo_ros_state.so'><ros><namespace>/gazebo</namespace></ros><update_rate>20</update_rate></plugin>"
    scene = "<scene><ambient>0.36 0.36 0.37 1</ambient><background>0.72 0.78 0.85 1</background><shadows>true</shadows><grid>false</grid></scene>"
    physics = "<physics type='ode'><max_step_size>0.001</max_step_size><real_time_factor>1</real_time_factor><real_time_update_rate>1000</real_time_update_rate></physics>"
    from .robot import ROBOT_RGB, tool_model
    dyn += [tool_model(x["gazebo"], "robot{}_arm".format(i + 1), "robot{}".format(i + 1), mesh_dir, ROBOT_RGB[i % len(ROBOT_RGB)], pose=(1.0 * i, 0.0, -3.0)) for i, x in enumerate(F.fleet)]
    lay["tools"] = ["tray" if x["gazebo"] == "tb3_tray" else "arm" for x in F.fleet]
    rec = "".join(recorder(name, pose, record, fps) for name, pose in views(F, bounds, lay).items()) if record else ""
    world = "<?xml version='1.0'?><sdf version='1.6'><world name='factory'>{}{}{}{}{}{}{}{}{}{}</world></sdf>\n".format(physics, scene, cam, plugin, sun, lights, floor, "".join(m.sdf() for m in models), "".join(dyn), rec)
    return world, lay, footprints(models), bounds


def posed(name, visuals, pose):
    return "<model name='{}'><pose>{} {} 0 0 0 {}</pose><link name='link'><gravity>false</gravity><kinematic>true</kinematic>{}{}</link></model>".format(name, pose[0], pose[1], pose[2], INERT, visuals)


def door_visuals(w, zc, h):
    m = Model("d")
    m.cbox(0, 0, zc, 0.012, w + 0.06, h + 0.06, C["glass"], alpha=0.55)
    for dz in (-(h + 0.06) / 2, (h + 0.06) / 2):
        m.cbox(-0.004, 0, zc + dz, 0.02, w + 0.1, 0.035, C["dark"])
    for dy in (-(w + 0.06) / 2, (w + 0.06) / 2):
        m.cbox(-0.004, dy, zc, 0.02, 0.035, h + 0.1, C["dark"])
    m.cyl(-0.035, -w / 2 + 0.03, zc - 0.2, zc + 0.2, 0.012, C["alu"])
    for dz in (-0.18, 0.18):
        m.cbox(-0.02, -w / 2 + 0.03, zc + dz, 0.03, 0.02, 0.02, C["alu"])
    return "".join(m.items)


def spindle_visuals():
    m = Model("s")
    m.cbox(0.0, 0.0, 1.59, 0.14, 0.26, 0.42, C["dark2"])
    m.cyl(0, 0, 1.08, 1.38, 0.045, C["alu"])
    m.cyl(0, 0, 1.06, 1.1, 0.03, C["dark"])
    m.cyl(0, 0, 0.94, 1.06, 0.008, (0.8, 0.65, 0.2))
    return "".join(m.items)


def press_visuals():
    m = Model("p")
    ht = BELT - 0.02
    m.cyl(0, 0, ht + 0.14, ht + 0.32, 0.008, C["steel"])
    m.cbox(0, 0, ht + 0.125, 0.07, 0.07, 0.03, C["dark2"])
    return "".join(m.items)


def views(F, bounds, lay):
    """Recording cameras: an overview of the floor and a close-up of the robot docks of the busiest-looking stations."""
    x0, y0, x1, y1 = bounds
    out = {"overview": ((x0 + x1) / 2, y0 - 4.5, 10.5, 0.72, 1.5708)}
    for s in ("MC1", "asm1"):
        if s in F.stations and F.adj[F.stations[s]["node"]]:
            x, y, yaw = lay["docks"][F.stations[s]["node"]]
            out["close_" + s] = (x - 2.2 * math.cos(yaw + 0.6), y - 2.2 * math.sin(yaw + 0.6), 1.5, 0.35, yaw + 0.6)
    if lay.get("chargers") and not F.transfers:
        c = next(iter(lay["chargers"]))
        x, y, yaw = lay["docks"][c]
        out["close_" + c] = (x - 2.0 * math.cos(yaw - 0.5), y - 2.0 * math.sin(yaw - 0.5), 1.3, 0.35, yaw - 0.5)
    return out


def recorder(name, pose, path, fps):
    x, y, z, pitch, yaw = pose
    return "<model name='rec_{0}'><static>true</static><pose>{1:.2f} {2:.2f} {3:.2f} 0 {4:.2f} {5:.3f}</pose><link name='l'><sensor name='cam' type='camera'><always_on>true</always_on><update_rate>{7}</update_rate><camera><horizontal_fov>1.25</horizontal_fov><image><width>960</width><height>540</height></image><clip><near>0.05</near><far>100</far></clip><save enabled='true'><path>{6}/{0}</path></save></camera></sensor></link></model>".format(name, x, y, z, pitch, yaw, path, fps)


def occupancy(foot, bounds):
    """Rasterise collision footprints into a Nav2 occupancy map. -> (PGM bytes, map yaml dict)."""
    x0, y0, x1, y1 = bounds
    x0, y0, x1, y1 = x0 - 0.3, y0 - 0.3, x1 + 0.3, y1 + 0.3
    W, H = int(math.ceil((x1 - x0) / RES)) + 1, int(math.ceil((y1 - y0) / RES)) + 1
    gx, gy = np.meshgrid(x0 + (np.arange(W) + 0.5) * RES, y0 + (np.arange(H) + 0.5) * RES)
    occ = np.zeros((H, W), bool)
    for (cx, cy, yaw, sx, sy) in foot:
        c, s = math.cos(yaw), math.sin(yaw)
        occ |= (np.abs(c * (gx - cx) + s * (gy - cy)) <= sx / 2 + RES / 2) & (np.abs(-s * (gx - cx) + c * (gy - cy)) <= sy / 2 + RES / 2)
    grid = np.where(occ[::-1], 0, 254).astype(np.uint8).tobytes()
    return "P5\n{} {}\n255\n".format(W, H).encode() + grid, {"resolution": RES, "origin": [x0, y0, 0.0], "negate": 0, "occupied_thresh": 0.65, "free_thresh": 0.196, "mode": "trinary"}


def clearance(foot, x, y):
    """Distance from (x, y) to the nearest footprint (to check that docks and parking spots are free)."""
    best = float("inf")
    for (cx, cy, yaw, sx, sy) in foot:
        c, s = math.cos(yaw), math.sin(yaw)
        px, py = c * (x - cx) + s * (y - cy), -s * (x - cx) + c * (y - cy)
        best = min(best, math.hypot(max(0.0, abs(px) - sx / 2), max(0.0, abs(py) - sy / 2)))
    return best


def spawn(F):
    return [(F.nodes[F.robots["start"][i % len(F.robots["start"])]][0], F.nodes[F.robots["start"][i % len(F.robots["start"])]][1], 0.0) for i in range(F.robots["count"])]


def robots_arg(F, names=None):
    """Spawn poses in nav2_bringup's multi-robot format."""
    names = names or ["robot{}".format(i + 1) for i in range(F.robots["count"])]
    return ";".join("{}={{x: {}, y: {}, yaw: {}}}".format(n, *p) for n, p in zip(names, spawn(F)))


def part_names(F, jobs, spares=2):
    """Workpiece models: one per known job plus spares per product (for replacement orders after scrap)."""
    parts = {str(j): ("part_{}".format(j), p) for j, p in jobs}
    spare = {p: ["spare_{}_{}".format(p, k) for k in range(spares)] for p in F.products}
    return parts, spare


def write(cfg, out, jobs=(), record=False, mesh_dir=None, media=True, people=True):
    """Write factory.world, factory.pgm/.yaml (Nav2 map) and layout.json to `out` (plus textures and bevelled meshes in
    out/media unless media=False). jobs: [(job id, product)]."""
    global MEDIA
    F = Factory(cfg)
    os.makedirs(out, exist_ok=True)
    MEDIA = mediamod.Media(out) if media else None
    parts, spare = part_names(F, jobs)
    world, lay, foot, bounds = sdf(F, list(parts.values()) + [(n, p) for p, ns in spare.items() for n in ns], record=os.path.join(os.path.abspath(out), "frames") if record else None, mesh_dir=mesh_dir, people=people)
    open(os.path.join(out, "factory.world"), "w").write(world)
    if MEDIA is not None:
        MEDIA.write()
    MEDIA = None
    pgm, meta = occupancy(foot, bounds)
    open(os.path.join(out, "factory.pgm"), "wb").write(pgm)
    meta["image"] = "factory.pgm"
    with open(os.path.join(out, "factory.yaml"), "w") as f:
        for k, v in meta.items():
            f.write("{}: {}\n".format(k, v))
    lay.update({"parts": {j: n for j, (n, _) in parts.items()}, "spares": spare, "robots": robots_arg(F), "spawn": spawn(F), "bounds": bounds})
    lay["clearance"] = {n: round(clearance(foot, x, y), 3) for n, (x, y, _) in lay["docks"].items() if F.adj[n]}
    json.dump(lay, open(os.path.join(out, "layout.json"), "w"), indent=1)
    tight = {n: c for n, c in lay["clearance"].items() if c < 0.24}
    if tight:
        print("warning: docks closer than 0.24 m to equipment: {}".format(tight))
    return {"world": os.path.join(out, "factory.world"), "map": os.path.join(out, "factory.yaml"), "robots": robots_arg(F), "layout": os.path.join(out, "layout.json")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="lab")
    ap.add_argument("--out", default="factory_world")
    a = ap.parse_args()
    print(json.dumps(write(a.config, a.out), indent=1))


if __name__ == "__main__":
    main()
