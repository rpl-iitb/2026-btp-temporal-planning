"""Fit the Gym layer's robot models from measured runs (Gazebo or hardware result.json files): travel speed and noise,
pick / place durations (dock + arm + undock) and grasp / place success rates.

    python -m factory_sim.ros.calibrate run1/result.json run2/result.json
"""
import json
import sys

import numpy as np


def lognormal(x):
    x = np.asarray(x, float)
    return [round(float(x.mean()), 2), round(float(x.std() / x.mean()), 3)] if len(x) > 1 else [round(float(x.mean()), 2), 0.0] if len(x) else None


def fit(paths):
    runs = [json.load(open(p)) for p in paths]
    rows = [t for r in runs for t in r["travel"] if t["distance"] and t["seconds"] > 0]
    d, s = np.array([r["distance"] for r in rows]), np.array([r["seconds"] for r in rows])
    speed = float(d.sum() / s.sum())
    ratio = s / (d / speed)
    out = {"samples": len(rows), "speed": round(speed, 4), "travel_cv": round(float(ratio.std() / ratio.mean()), 4), "mean_seconds": round(float(s.mean()), 2)}
    man = [m for r in runs for m in r.get("manipulation", [])]
    for kind, key, ok in (("pick", "pick", "grasp_success"), ("place", "place", "place_success")):
        rows = [m for m in man if m["kind"] == kind]
        if rows:
            out[key] = {"lognormal": lognormal([m["seconds"] for m in rows if m["ok"]])}
            out[ok] = round(sum(m["ok"] for m in rows) / len(rows), 3)
    return out


if __name__ == "__main__":
    r = fit(sys.argv[1:])
    print(json.dumps(r, indent=1))
    print("\nconfig snippet:\nrobots: {}".format(json.dumps({k: v for k, v in r.items() if k not in ("samples", "mean_seconds")})))
