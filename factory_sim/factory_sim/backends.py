"""Driving the core from an external clock (Gazebo / ROS 2 / hardware).

The core owns the task-level state; a backend owns the physical robots. The backend starts the physical activity
for each pending core activity (travel, load, unload, charge, processing, conveyor) and reports when it ends:

    sim = build_sim(instance, time_source="external")
    while not sim.done and sim.failed is None:
        agent = sim.next_decision()
        if agent:
            sim.act(agent, policy(sim, agent, sim.legal(agent)))
            continue
        # start new physical activities in sim.external_pending: {aid: (kind, data, t_start, nominal_duration)}
        # ... then, when one finishes at time t: sim.complete(aid, t); when nothing finishes before t: sim.advance(t)

run_external() below is the same loop with simulated end times; a ROS 2 node replaces `duration` by the robot's
reported arrival (Nav2 result) and the machines' reported completion.
"""


def run_external(sim, policy, duration=None, max_steps=10 ** 6):
    """Reference external driver: each pending activity ends after duration(kind, data, nominal) (default: nominal)."""
    duration = duration or (lambda kind, data, nominal: nominal)
    ends = {}
    for _ in range(max_steps):
        if sim.done or sim.failed is not None:
            break
        agent = sim.next_decision()
        if agent is not None:
            sim.act(agent, policy(sim, agent, sim.legal(agent)))
            continue
        for aid, (kind, data, t0, nom) in sim.external_pending.items():
            if aid not in ends:
                ends[aid] = t0 + duration(kind, data, nom)
        t_ext = min((ends[a], a) for a in sim.external_pending) if sim.external_pending else None
        t_int = sim.ev[0][0] if sim.ev else None
        if t_ext is None and t_int is None:
            sim.next_decision()
            break
        if t_ext is not None and (t_int is None or t_ext[0] <= t_int):
            sim.complete(t_ext[1], t_ext[0])
            ends.pop(t_ext[1])
        else:
            sim.advance(t_int)
    return sim.metrics()
