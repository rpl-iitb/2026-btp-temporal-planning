"""Mobile manipulator: TurtleBot3 Waffle (stock nav2_bringup waffle.model) + OpenMANIPULATOR-X arm, and the arm's pick/place motions.

The arm is a separate kinematic model per robot (no collisions, so the lidar is not blocked; no effect on the robot's
physics), pinned to the robot's pose by the ROS node; its joints are driven by Gazebo's joint_pose_trajectory plugin on
/<robot>/set_joint_trajectory. Geometry follows open_manipulator_x.urdf.xacro.
"""
import math
import os

JOINTS = ["arm_joint1", "arm_joint2", "arm_joint3", "arm_joint4", "arm_gripper_left", "arm_gripper_right"]
LIMITS = [(-2.83, 2.83), (-1.79, 1.57), (-0.94, 1.38), (-1.79, 2.04), (-0.01, 0.019), (-0.01, 0.019)]
MOUNT = (0.04, 0.0, 0.10)
J1, J2Z, A3, L4, TOOL = (0.012, 0.0, 0.017), 0.0595, (0.024, 0.128), 0.124, 0.105
OPEN, CLOSED = 0.016, 0.0
HOME = (0.0, -1.0, 0.3, 0.7)
GRASP_Z = 0.03
ROBOT_RGB = [(0.9, 0.2, 0.2), (0.2, 0.7, 0.3), (0.2, 0.4, 0.9), (0.95, 0.75, 0.1), (0.7, 0.3, 0.8), (0.1, 0.8, 0.8)]


def _planar():
    l1, p3 = math.hypot(*A3), math.atan2(A3[1], A3[0])
    return MOUNT[0] + J1[0], MOUNT[2] + J1[2] + J2Z, l1, p3


def fk(q):
    """Tool (grasp point) position in the robot base frame."""
    q1, q2, q3, q4 = q[:4]
    s0, z0, l1, p3 = _planar()
    s = L4 * math.cos(q2 + q3) + l1 * math.cos(p3 - q2) + TOOL * math.cos(q2 + q3 + q4)
    z = z0 + l1 * math.sin(p3 - q2) - L4 * math.sin(q2 + q3) - TOOL * math.sin(q2 + q3 + q4)
    return (MOUNT[0] + J1[0] * math.cos(q1) + s * math.cos(q1), MOUNT[1] + J1[0] * math.sin(q1) + s * math.sin(q1), z)


def ik(x, y, z):
    """Joint angles reaching (x, y, z) in the base frame with the steepest feasible downward tool pitch, or None."""
    q1 = math.atan2(y - MOUNT[1], x - MOUNT[0])
    s = math.hypot(x - MOUNT[0], y - MOUNT[1]) - J1[0]
    _, z0, l1, p3 = _planar()
    for k in range(40):
        phi = math.pi / 2 - k * 0.04
        ws, wz = s - TOOL * math.cos(phi), z - z0 + TOOL * math.sin(phi)
        c = (ws * ws + wz * wz - l1 * l1 - L4 * L4) / (2 * l1 * L4)
        if abs(c) > 1:
            continue
        d = -math.acos(c)
        a = math.atan2(wz, ws) - math.atan2(L4 * math.sin(d), l1 + L4 * math.cos(d))
        q2 = p3 - a
        q3 = -(a + d) - q2
        q = (q1, q2, q3, phi - q2 - q3)
        if all(lo <= v <= hi for v, (lo, hi) in zip(q, LIMITS)):
            return q
    return None


def to_base(pose, p):
    """World point p -> robot base frame given robot pose (x, y, yaw)."""
    dx, dy = p[0] - pose[0], p[1] - pose[1]
    c, s = math.cos(pose[2]), math.sin(pose[2])
    return (c * dx + s * dy, -s * dx + c * dy, p[2])


def motion(kind, dock, target, duration):
    """Keyframes [(t, 6 joint values, event)] for a pick ('load') or place ('unload') at world point `target` (counter
    slot, part bottom) from the ideal dock pose; events 'grasp' / 'release' mark when the part changes hands."""
    b = to_base(dock, target)
    at = ik(b[0], b[1], b[2] + GRASP_Z) or ik(b[0] * 0.9, b[1] * 0.9, b[2] + GRASP_Z) or HOME
    pre = ik(b[0], b[1], b[2] + GRASP_Z + 0.07) or at
    carry = CARRY
    g = lambda q, w: tuple(q) + (w, w)
    if kind == "load":
        keys = [(0.0, g(HOME, OPEN), None), (0.3, g(pre, OPEN), None), (0.45, g(at, OPEN), None), (0.55, g(at, CLOSED), "grasp"), (0.7, g(pre, CLOSED), None), (1.0, g(carry, CLOSED), None)]
    else:
        keys = [(0.0, g(carry, CLOSED), None), (0.3, g(pre, CLOSED), None), (0.45, g(at, CLOSED), None), (0.55, g(at, OPEN), "release"), (0.7, g(pre, OPEN), None), (1.0, g(HOME, OPEN), None)]
    return [(t * duration, q, e) for t, q, e in keys]


CARRY = ik(0.16, 0.0, 0.33) or HOME


def failed_grasp(keys):
    """A missed grasp: same motion but the gripper closes on nothing and the arm returns to HOME."""
    return [(t, q, None) for t, q, _ in keys[:-1]] + [(keys[-1][0], tuple(HOME) + (OPEN, OPEN), None)]


def sample(keys, dt=0.05):
    """Dense, smoothed joint trajectory [(t, joints)] through the keyframes."""
    out = []
    for (t0, a, _), (t1, b, _) in zip(keys, keys[1:]):
        n = max(1, int((t1 - t0) / dt))
        for i in range(n):
            u = i / n
            u = u * u * (3 - 2 * u)
            out.append((t0 + (t1 - t0) * i / n, [x + (y - x) * u for x, y in zip(a, b)]))
    out.append((keys[-1][0], list(keys[-1][1])))
    return out


def event_time(keys, name):
    return next((t for t, _, e in keys if e == name), None)


def _inertial(m=0.05):
    return "<inertial><mass>{0}</mass><inertia><ixx>1e-4</ixx><iyy>1e-4</iyy><izz>1e-4</izz><ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia></inertial>".format(m)


def _vis(name, mesh_dir, mesh, pose, rgb, box=None):
    geo = "<mesh><uri>file://{}</uri><scale>0.001 0.001 0.001</scale></mesh>".format(os.path.join(mesh_dir, mesh)) if mesh_dir and os.path.exists(os.path.join(mesh_dir, mesh)) else "<box><size>{} {} {}</size></box>".format(*box)
    return "<visual name='{}'><pose>{}</pose><geometry>{}</geometry><material><ambient>{} {} {} 1</ambient><diffuse>{} {} {} 1</diffuse><specular>0.3 0.3 0.3 1</specular></material></visual>".format(name, pose, geo, *rgb, *rgb)


def arm_model(model_name, ns, mesh_dir, rgb, pose=(0.0, 0.0, -2.0)):
    """Standalone kinematic OpenMANIPULATOR-X (+ robot id plate) that the node pins to robot `ns` (model frame = robot base_footprint)."""
    mx, my, mz = MOUNT
    j1 = (mx + J1[0], my, mz + J1[2])
    j2 = (j1[0], my, j1[2] + J2Z)
    j3 = (j2[0] + A3[0], my, j2[2] + A3[1])
    j4 = (j3[0] + L4, my, j3[2])
    grey, dark = (0.85, 0.85, 0.87), (0.12, 0.12, 0.13)
    links = [("arm_link1", (mx, my, mz), "link1.stl", "0 0 0 0 0 0", dark, (0.06, 0.06, 0.04)), ("arm_link2", j1, "link2.stl", "0 0 0.019 0 0 0", grey, (0.04, 0.04, 0.06)),
             ("arm_link3", j2, "link3.stl", "0 0 0 0 0 0", grey, (0.04, 0.04, 0.13)), ("arm_link4", j3, "link4.stl", "0 0 0 0 0 0", grey, (0.124, 0.04, 0.04)),
             ("arm_link5", j4, "link5.stl", "0 0 0 0 0 0", dark, (0.07, 0.05, 0.04)), ("arm_gripper_left_link", (j4[0] + 0.0817, 0.021, j4[2]), "gripper_left_palm.stl", "0 0 0 0 0 0", dark, (0.04, 0.01, 0.03)),
             ("arm_gripper_right_link", (j4[0] + 0.0817, -0.021, j4[2]), "gripper_right_palm.stl", "0 0 0 0 0 0", dark, (0.04, 0.01, 0.03))]
    out = []
    for name, p, mesh, vp, col, box in links:
        out.append("<link name='{}'><pose>{:.4f} {:.4f} {:.4f} 0 0 0</pose><gravity>false</gravity><kinematic>true</kinematic>{}{}</link>".format(name, p[0], p[1], p[2], _inertial(), _vis(name + "_v", mesh_dir, mesh, vp, col, box)))
    out.append("<link name='arm_base'><gravity>false</gravity><kinematic>true</kinematic>{}<visual name='mast'><pose>-0.17 0 0.165 0 0 0</pose><geometry><cylinder><radius>0.006</radius><length>0.1</length></cylinder></geometry><material><ambient>0.2 0.2 0.2 1</ambient><diffuse>0.2 0.2 0.2 1</diffuse></material></visual><visual name='id'><pose>-0.2 0 0.105 0 0 0</pose><geometry><box><size>0.03 0.18 0.012</size></box></geometry><material><ambient>{} {} {} 1</ambient><diffuse>{} {} {} 1</diffuse><emissive>{} {} {} 1</emissive></material></visual></link>".format(_inertial(0.01), *rgb, *rgb, *rgb))
    joints = [("arm_mount", "fixed", "arm_base", "arm_link1", None, None), ("arm_joint1", "revolute", "arm_link1", "arm_link2", "0 0 1", LIMITS[0]), ("arm_joint2", "revolute", "arm_link2", "arm_link3", "0 1 0", LIMITS[1]),
              ("arm_joint3", "revolute", "arm_link3", "arm_link4", "0 1 0", LIMITS[2]), ("arm_joint4", "revolute", "arm_link4", "arm_link5", "0 1 0", LIMITS[3]),
              ("arm_gripper_left", "prismatic", "arm_link5", "arm_gripper_left_link", "0 1 0", LIMITS[4]), ("arm_gripper_right", "prismatic", "arm_link5", "arm_gripper_right_link", "0 -1 0", LIMITS[5])]
    for name, kind, parent, child, axis, lim in joints:
        ax = "<axis><xyz>{}</xyz><limit><lower>{}</lower><upper>{}</upper><effort>10</effort><velocity>5</velocity></limit><dynamics><damping>5</damping><friction>5</friction></dynamics></axis>".format(axis, *lim) if axis else ""
        out.append("<joint name='{}' type='{}'><parent>{}</parent><child>{}</child>{}</joint>".format(name, kind, parent, child, ax))
    out.append("<plugin name='arm_trajectory' filename='libgazebo_ros_joint_pose_trajectory.so'><ros><namespace>/{0}</namespace></ros><update_rate>20</update_rate></plugin>".format(ns))
    out.append("<plugin name='arm_joint_states' filename='libgazebo_ros_joint_state_publisher.so'><ros><namespace>/" + ns + "</namespace><remapping>~/out:=arm_joint_states</remapping></ros><update_rate>10</update_rate>{}</plugin>".format("".join("<joint_name>{}</joint_name>".format(j) for j in JOINTS)))
    return "<model name='{}'><pose>{} {} {} 0 0 0</pose>{}</model>".format(model_name, pose[0], pose[1], pose[2], "".join(out))


TRAY_TOP = 0.205   # tray deck height above the floor (parts carried by tray robots sit here)
TRAY_AT = (-0.02, 0.0)


def tray_model(model_name, rgb, pose=(0.0, 0.0, -2.0)):
    """Standalone kinematic tray top (+ id plate and LED mast) pinned to a tray robot like the arm: four posts, a deck
    with a rim and a rubber mat. No joints: trays are moved on and off by a handler (person)."""
    alu, dark, mat = (0.78, 0.79, 0.8), (0.14, 0.14, 0.15), (0.2, 0.21, 0.22)

    def box(name, x, y, z, sx, sy, sz, col, em=False):
        e = "<emissive>{} {} {} 1</emissive>".format(*col) if em else ""
        return "<visual name='{}'><pose>{} {} {} 0 0 0</pose><geometry><box><size>{} {} {}</size></box></geometry><material><ambient>{} {} {} 1</ambient><diffuse>{} {} {} 1</diffuse><specular>0.25 0.25 0.25 1</specular>{}</material></visual>".format(name, x, y, z, sx, sy, sz, *col, *col, e)
    cx, cy = TRAY_AT
    z = TRAY_TOP - 0.012
    v = [box("deck", cx, cy, z, 0.26, 0.26, 0.012, alu), box("mat", cx, cy, z + 0.007, 0.23, 0.23, 0.003, mat)]
    for i, (dx, dy, sx, sy) in enumerate(((0.13, 0, 0.008, 0.27), (-0.13, 0, 0.008, 0.27), (0, 0.13, 0.27, 0.008), (0, -0.13, 0.27, 0.008))):
        v.append(box("rim{}".format(i), cx + dx, cy + dy, z + 0.02, sx, sy, 0.03, alu))
    for i, (dx, dy) in enumerate(((0.1, 0.1), (0.1, -0.1), (-0.1, 0.1), (-0.1, -0.1))):
        v.append("<visual name='post{}'><pose>{} {} {} 0 0 0</pose><geometry><cylinder><radius>0.008</radius><length>{}</length></cylinder></geometry><material><ambient>{} {} {} 1</ambient><diffuse>{} {} {} 1</diffuse></material></visual>".format(i, cx + dx, cy + dy, (0.145 + z) / 2, z - 0.145, *dark, *dark))
    v.append(box("id", -0.2, 0.0, 0.105, 0.03, 0.18, 0.012, rgb, em=True))
    v.append("<visual name='mast'><pose>-0.17 0 0.165 0 0 0</pose><geometry><cylinder><radius>0.006</radius><length>0.1</length></cylinder></geometry><material><ambient>0.2 0.2 0.2 1</ambient><diffuse>0.2 0.2 0.2 1</diffuse></material></visual>")
    return "<model name='{}'><pose>{} {} {} 0 0 0</pose><link name='arm_base'><gravity>false</gravity><kinematic>true</kinematic>{}{}</link></model>".format(model_name, pose[0], pose[1], pose[2], _inertial(0.01), "".join(v))


def tool_model(gazebo, model_name, ns, mesh_dir, rgb, pose=(0.0, 0.0, -2.0)):
    """The top module of a robot type (robots.TYPES[...]['gazebo']): the OpenMANIPULATOR-X arm or a tray."""
    return tray_model(model_name, rgb, pose) if gazebo == "tb3_tray" else arm_model(model_name, ns, mesh_dir, rgb, pose)


def robot_sdf(template, models_dir=None):
    """Waffle SDF text (nav2_bringup/worlds/waffle.model) without the unused depth camera (it costs software rendering);
    model:// mesh URIs become absolute file:// paths so camera sensors render the robot too."""
    a = template.find("<sensor name=\"intel_realsense_r200_depth\"")
    t = template if a < 0 else template[:a] + template[template.index("</sensor>", a) + len("</sensor>"):]
    return t.replace("model://", "file://" + models_dir.rstrip("/") + "/") if models_dir else t
