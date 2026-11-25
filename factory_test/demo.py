"""Demo GIFs: a policy runs a FactorySim instance (hard suite) closed loop under random durations; the schedule it
actually executed is replayed in FactorySim (factory_sim.pddl.PlanPolicy, recorded) and rendered as an animated GIF of
the plant (robots, stations, conveyors, jobs), with a .json caption (task and outcome) next to it.

    FACTORY_SUITE=hard python -m factory_test.demo --method V15_hard --compare dispatcher --size L --out demos
        -> picks the first test instance (and seed) of that size where the method delivers every order on time and the
           dispatcher does not, under the same random durations (else the first where the method succeeds), and
           renders both: demos/factory_V15_hard.gif, demos/factory_dispatcher.gif (+ .json)
    FACTORY_SUITE=hard python -m factory_test.demo --method V14_hard --instance hard_base_M_test_000 --seed 1 --out demos

--method / --compare: dispatcher (the rule-based robot dispatcher) or a trained model in results_hard/models
(V13_hard, V14_hard, V15_hard, ...). --frames sets the GIF length (one frame per makespan / frames simulated seconds).
"""
import argparse
import json
import os

import factory_test  # noqa: F401  (paths)
from factory_test import factory_v13 as K
from factory_test.closed_eval import late, plan_text
from algorithm import v13
from algorithm.menv import MAEnv

v13.use(K)
_POL = {}


def controller(method, env):
    if method == "dispatcher":
        from factory_test.factory_dispatcher import Dispatcher
        return Dispatcher(env)
    if method not in _POL:
        mod = __import__("algorithm." + method[:3].lower(), fromlist=["load"])
        _POL[method] = mod.load(os.path.join(v13.MODELS, method))
    return v13.Controller(env, _POL[method], "greedy", label=False)


def closed_loop(method, T, seed, noise, cert):
    env = MAEnv(T, K.AGENT, noise=v13.NOISE.scaled(noise), seed=seed, horizon=4 * cert)
    r = env.rollout(controller(method, env), seed=seed)
    if not r["success"] and late(T, r):
        r["reason"] = "late"
    return r


def render(res, method, d, meta, a):
    """Replay the executed schedule in FactorySim, write <out>/factory_<method>.gif and its .json."""
    from factory_sim import pddl as FP
    from factory_sim import render as R
    from factory_sim import render_sim as RS
    from factory_sim.env import build_sim
    name = meta["instance_id"]
    inst = res.get("instance") or json.load(open(os.path.join(d, name, "instance.json")))
    plan = res.get("plan") or plan_text(res["steps"])
    policy = FP.PlanPolicy(build_sim(inst, seed=0, time_source="nominal"), plan)
    sim = build_sim(inst, seed=0, time_source="nominal", record=True)
    m = sim.run(policy)
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "factory_{}.gif".format(method))
    label = "dispatcher (rule-based)" if method == "dispatcher" else "ours: event-graph policy"
    title = "{}   {} robots, {} orders".format(label, meta.get("robots"), meta.get("orders"))
    if a.view == "schematic":
        R.gif(sim, path, dt=max(sim.trace[-1]["t"], 1.0) / a.frames, fps=a.fps, title=title)
    else:
        RS.gif(sim, path, frames=a.frames, fps=a.fps, title=title, panel_text=a.text or None)
    video = None
    if a.video:
        vw = 1900 if len(sim.robots) > 6 else 1500
        video, speed = RS.video(sim, path[:-4] + ".mp4", speed=a.speed or None, duration=a.duration, title=title,
                                width=vw, panel_text=round(a.text * vw / 1100) if a.text else None)
    cert = meta["best_known"]
    info = {"simulator": "factorysim", "method": method, "instance": name, "family": meta.get("family"),
            "size": meta.get("size"), "robots": meta.get("robots"), "orders": meta.get("orders"),
            "stations": len(sim.stations), "noise": a.noise, "seed": a.seed,
            "closed_loop": {k: res[k] for k in ("success", "reason", "served", "goals", "makespan")},
            "makespan_over_best_known": round(res["makespan"] / cert, 3) if res["success"] else None,
            "replay": {k: m.get(k) for k in ("success", "makespan", "failure") if k in m},
            "best_known": cert, "due_reference": meta.get("due_reference"), "gif": path,
            "gif_bytes": os.path.getsize(path), "video": video, "video_speed": round(speed, 1) if video else None,
            "meta": meta, "plan": plan}
    with open(os.path.join(a.out, name + ".instance.json"), "w") as fh:     # for --from_json without the data
        json.dump(inst, fh)
    json.dump(info, open(path + ".json", "w"), indent=1, default=str)
    print("wrote {} ({:.1f} MB): {} {} served {}/{} ({})".format(path, info["gif_bytes"] / 1e6, method, name,
                                                                  res["served"], res["goals"], res["reason"]))
    return info


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="V15_hard")
    ap.add_argument("--compare", default="dispatcher", help="also render this method on the same instance ('' = none)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--size", default="L", help="search instances of this size (M, L, XL, ...) when --instance is not given")
    ap.add_argument("--family", default=None, help="and of this family (default: any)")
    ap.add_argument("--max_instances", type=int, default=12)
    ap.add_argument("--instance", default=None)
    ap.add_argument("--seed", type=int, default=None, help="random durations (searched over --seeds if not given)")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--noise", type=float, default=1.0)
    ap.add_argument("--frames", type=int, default=240)
    ap.add_argument("--view", default="sim", choices=["sim", "schematic"],
                    help="sim: the plant floor with robots (render_sim.py); schematic: station-state boxes (render.py)")
    ap.add_argument("--video", type=int, default=1, help="also write an MP4 of the sim view (needs imageio-ffmpeg)")
    ap.add_argument("--speed", type=float, default=0, help="simulated seconds per video second (0: from --duration)")
    ap.add_argument("--duration", type=float, default=90, help="video length in seconds when --speed is 0")
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--text", type=int, default=26,
                    help="font size (px, in the 1100 px GIF) of the order panel; with more than 16 orders it becomes "
                         "a grid of numbered chips coloured by status; 0: the small default list")
    ap.add_argument("--from_json", nargs="+", default=None,
                    help="re-render saved runs (<gif>.json written by this script, with their plan) without a model")
    ap.add_argument("--out", default="demos")
    a = ap.parse_args(argv)
    if a.from_json:
        for f in a.from_json:
            info = json.load(open(f))
            if not info.get("plan"):
                raise SystemExit(f + ": no saved plan (rendered before plans were saved): re-run with the model")
            inst = os.path.join(os.path.dirname(f), info["instance"] + ".instance.json")
            a.seed, a.noise, a.out = info["seed"], info["noise"], a.out if a.out != "demos" else os.path.dirname(f)
            render(dict(info["closed_loop"], plan=info["plan"], instance=json.load(open(inst))), info["method"], None,
                   info["meta"], a)
        return
    d = os.path.join(factory_test.DATA, a.split)
    metas = json.load(open(os.path.join(d, "dataset.json")))["instances"]
    if a.instance:
        metas = [m for m in metas if m["instance_id"] == a.instance]
    else:
        metas = [m for m in metas if m.get("size") == a.size and (a.family is None or m.get("family") == a.family)]
        metas = metas[:a.max_instances]
    if not metas:
        raise SystemExit("no instance matches in " + d)
    seeds = [a.seed] if a.seed is not None else list(range(1, a.seeds + 1))
    best, top = None, 3 if a.compare else 2     # best: (score, meta, seed, res, res_compare)
    for meta in metas:
        T = K.task(a.split, meta["instance_id"])
        for seed in seeds:
            r = closed_loop(a.method, T, seed, a.noise, meta["best_known"])
            rc = closed_loop(a.compare, T, seed, a.noise, meta["best_known"]) if a.compare else None
            print("  {} seed {}: {} {} ({})  {} {}".format(meta["instance_id"], seed, a.method, r["success"], r["reason"],
                                                          a.compare or "", rc["reason"] if rc else ""), flush=True)
            score = 2 * r["success"] + (rc is not None and not rc["success"])
            if best is None or score > best[0]:
                best = (score, meta, seed, r, rc)
            if score == top:
                break
        if best[0] == top:
            break
    score, meta, a.seed, r, rc = best
    print("showcase: {} seed {} ({})".format(meta["instance_id"], a.seed, {3: "method succeeds, comparison fails",
                                                                         2: "both succeed"}.get(score, "method fails")))
    render(r, a.method, d, meta, a)
    if rc is not None:
        render(rc, a.compare, d, meta, a)


if __name__ == "__main__":
    main()
