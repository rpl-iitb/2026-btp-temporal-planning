"""Robot types: what robots the fleet is made of, how they are programmed, and which actions they add to the domain.

A config chooses robots by type (like choosing a robot model in a physics simulator), then overrides parameters:

    robots:                                  robots:
      type: tb3_manipulator                    start: [k2, k5, k8]
      count: 3                                 fleet:
      speed: 0.12      # override              - {type: tb3_manipulator, count: 2}
                                               - {type: tb3_tray, count: 1, speed: 0.2}

Single-type form: the keys next to `type` override the type. Fleet form: each entry overrides its type, and the keys
next to `fleet` are shared defaults below the type (start, parking, chargers, ...). Without `type` or `fleet` the
fleet is `custom` (the generic parameters, as in FactorySim <= 1.0).

`handling` decides who moves a tray between a robot and a station, and so which actions exist:

- `arm`: the robot picks and places by itself (an on-board manipulator). PDDL: load-* / unload-* (robot only).
  ROS 2: an arm joint trajectory per pick / place, keyboard arm jogging in DRIVE mode.
- `handler`: a passive tray; a person from an operator pool (the station's `handler`, else `robots.handler_pool`,
  default `handler`) moves the tray. The robot waits docked for a free handler. PDDL: handover-load-* /
  handover-unload-* (robot + handler). ROS 2: the robot docks and a handler figure hands the tray over.

All robots move on the floor graph (`move`, Nav2 goals / cmd_vel) and, with batteries, charge. Add a type by adding
an entry to TYPES (or `register()`); keys not given fall back to GENERIC.
"""
import copy

GENERIC = {
    "summary": "generic transport robot",
    "base": None,              # mobile base model (documentation / Gazebo)
    "tool": "arm",             # arm | tray
    "handling": "arm",         # who moves trays on and off: arm (the robot) | handler (a person)
    "capacity": 1,             # trays carried at once (only 1 is simulated so far)
    "payload_kg": None,
    "speed": 0.2,              # nominal driving speed, m/s (planning speed, below the base's maximum)
    "travel_cv": 0.0,          # travel-time noise (coefficient of variation)
    "load": {"fixed": 0},      # pick (arm) or handover onto the robot (handler)
    "unload": {"fixed": 0},    # place (arm) or handover off the robot (handler)
    "grasp_success": 1.0,
    "place_success": 1.0,
    "battery": None,           # robot-side battery parameters (capacity, per_metre, per_second, per_service)
    "fault": None,             # {mtbf, mttr}
    "gazebo": None,            # Gazebo model used by factory_sim/ros (None: not available in Gazebo)
    "footprint_m": None,
}

TYPES = {  # only what differs from GENERIC
    "custom": dict(summary="generic robot with the parameters given in the config (FactorySim <= 1.0 behaviour)", gazebo="tb3_manipulator"),
    "tb3_manipulator": dict(
        summary="TurtleBot3 Waffle Pi + OpenMANIPULATOR-X: picks and places trays with its own arm",
        base="turtlebot3_waffle_pi", tool="arm", handling="arm", payload_kg=0.5, footprint_m=[0.281, 0.306],
        speed=0.15, travel_cv=0.2, load={"lognormal": [15, 0.15]}, unload={"lognormal": [12, 0.15]}, grasp_success=0.95, place_success=0.99,
        battery={"capacity": 100, "per_metre": 0.08, "per_second": 0.01, "per_service": 0.05},
        gazebo="tb3_manipulator"),
    "tb3_tray": dict(
        summary="TurtleBot3 Waffle Pi with a passive tray (no arm): a human handler at the station moves the tray",
        base="turtlebot3_waffle_pi", tool="tray", handling="handler", payload_kg=5.0, footprint_m=[0.281, 0.306],
        speed=0.18, travel_cv=0.2, load={"lognormal": [10, 0.3]}, unload={"lognormal": [8, 0.3]},
        battery={"capacity": 100, "per_metre": 0.07, "per_second": 0.008, "per_service": 0.0},
        gazebo="tb3_tray"),
    "tb3_burger_tray": dict(
        summary="TurtleBot3 Burger with a small tray: cheaper and narrower, smaller battery, handler-loaded",
        base="turtlebot3_burger", tool="tray", handling="handler", payload_kg=2.0, footprint_m=[0.138, 0.178],
        speed=0.14, travel_cv=0.25, load={"lognormal": [10, 0.3]}, unload={"lognormal": [8, 0.3]},
        battery={"capacity": 100, "per_metre": 0.11, "per_second": 0.012, "per_service": 0.0},
        gazebo="tb3_tray"),
}

# Types sketched for later (listed by `factory-sim robots`, rejected in configs until they are simulated).
PLANNED = {
    "roller_amr": "AMR with a powered roller top: transfers trays at conveyor-height stations without an arm or a person",
    "cart_x4": "cart AMR with four tray slots: one trip moves several trays (multi-capacity transport)",
}

HANDLING = {
    "arm": {"pddl": ["load-machine", "load-out", "load-buffer", "unload", "unload-buffer", "unload-sink"],
            "ros": ["Nav2 NavigateToPose (/<r>/navigate_to_pose)", "cmd_vel (/<r>/cmd_vel)", "arm joint trajectory (/<r>_arm/set_joint_trajectory)"],
            "teleop": ["drive", "arm jog", "task"]},
    "handler": {"pddl": ["handover-load-machine", "handover-load-out", "handover-load-buffer", "handover-unload", "handover-unload-buffer", "handover-unload-sink"],
                "ros": ["Nav2 NavigateToPose (/<r>/navigate_to_pose)", "cmd_vel (/<r>/cmd_vel)", "handler figure animation (no arm)"],
                "teleop": ["drive", "task"]},
}
TASK_ACTIONS = ["pickup <job>", "deliver <station>", "move <node>", "charge (with batteries)", "wait"]
FLEET_KEYS = {"type", "count", "fleet", "start", "parking", "handler_pool", "handlers"}


def register(name, **spec):
    """Add a robot type at runtime (keys not given fall back to GENERIC)."""
    TYPES[name] = dict(spec)
    return full(name)


def full(name):
    """All parameters of a type (GENERIC + the type's own values)."""
    return _over(GENERIC, TYPES[name])


def _aliases(d):
    d = dict(d)
    if "pick" in d:
        d["load"] = d.pop("pick")
    if "place" in d:
        d["unload"] = d.pop("place")
    return d


def _over(base, over):
    out = dict(base)
    for k, v in over.items():
        out[k] = dict(out[k], **v) if k == "battery" and isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def resolve(robots_cfg):
    """Per-robot specs (one dict per robot, in robot order r1, r2, ...) from a `robots:` config section."""
    rc = _aliases(robots_cfg or {})
    shared = {k: v for k, v in rc.items() if k not in FLEET_KEYS}
    if rc.get("fleet"):
        entries = [(_aliases(e), True) for e in rc["fleet"]]
    else:
        entries = [(dict(shared, type=rc.get("type", "custom"), count=rc.get("count", 1)), False)]
    out = []
    for e, fleet_form in entries:
        t = e.get("type", "custom")
        if t in PLANNED:
            raise ValueError("robot type {} is planned but not simulated yet: {}".format(t, PLANNED[t]))
        if t not in TYPES:
            raise ValueError("unknown robot type {} (known: {})".format(t, ", ".join(sorted(TYPES))))
        own = {k: v for k, v in e.items() if k not in ("type", "count")}
        spec = _over(_over(_over(GENERIC, shared), TYPES[t]), own) if fleet_form else _over(_over(GENERIC, TYPES[t]), shared)
        if spec.get("capacity", 1) != 1:
            raise ValueError("robot type {}: capacity {} is not simulated yet (1 tray per robot)".format(t, spec["capacity"]))
        if spec["handling"] not in HANDLING:
            raise ValueError("robot type {}: handling must be one of {}".format(t, sorted(HANDLING)))
        spec["type"] = t
        out += [copy.deepcopy(spec) for _ in range(int(e.get("count", 1)))]
    return out


def describe(name):
    """Human-readable description of a type: parameters, programming interfaces and the actions it adds."""
    t = full(name)
    h = HANDLING[t["handling"]]
    lines = ["{}: {}".format(name, t["summary"]),
             "  base {}  tool {}  handling {}  payload {} kg  speed {} m/s  gazebo {}".format(t["base"] or "-", t["tool"], t["handling"], t["payload_kg"] or "-", t["speed"], t["gazebo"] or "no"),
             "  task actions (Gym / PettingZoo): " + ", ".join(TASK_ACTIONS),
             "  PDDL actions: move, " + ", ".join(h["pddl"]),
             "  ROS 2 interfaces: " + "; ".join(h["ros"]),
             "  keyboard teleop: " + ", ".join(h["teleop"])]
    return "\n".join(lines)
