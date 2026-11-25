"""Example of a policy file for `factory-sim bench --policy examples.my_policy:policy` (run from the repository root)."""
from factory_sim import DISPATCHER


def policy(sim, agent, legal):
    if agent == DISPATCHER:
        rel = [a for a in legal if a[0] == "release"]
        return min(rel, key=lambda a: sim.jobs[a[1]].due) if rel else ("wait",)
    deliver = [a for a in legal if a[0] == "deliver" and sim.stations[a[1]].spec["kind"] != "buffer"]
    if deliver:
        return deliver[0]
    pickup = [a for a in legal if a[0] == "pickup"]
    return min(pickup, key=lambda a: sim.jobs[a[1]].due) if pickup else ("wait",)
