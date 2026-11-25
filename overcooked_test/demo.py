"""Demo GIFs: a policy runs a multi-cook test kitchen closed loop under random durations; the schedule it actually
executed is replayed in overcooked_kitchen (recorded) and rendered as an animated GIF, with a .json caption (task and
outcome) next to it.

    python -m overcooked_test.demo --method V14 --compare dispatcher --tier xl4 --out demos
        -> picks the first test instance (and seed) of that tier where V14 delivers every order and the dispatcher
           does not, under the same random durations (else the first where V14 succeeds), and renders both runs:
           demos/kitchen_V14.gif, demos/kitchen_dispatcher.gif (+ .mp4 videos at --speed, + .json)
    python -m overcooked_test.demo --method V14 --instance v3_test_xl4_001 --seed 1 --out demos

--method / --compare: dispatcher (the rule-based expert) or a trained model in results/models: V14 (= V14_v3),
V13 (= V13_v3), or any model name (V14_nofeats, ...).
"""
import argparse
import json
import os

import overcooked_test  # noqa: F401  (paths)
from overcooked_test import kitchen_v13 as K
from algorithm import v13
from algorithm.menv import MAEnv, plan_text

v13.use(K)
_POL = {}


def controller(method, env):
    if method == "dispatcher":
        from overcooked_test.kitchen_dispatcher import Dispatcher
        return Dispatcher(env)
    if method not in _POL:
        name = method + "_v3" if method in ("V13", "V14") else method
        mod = __import__("algorithm." + method[:3].lower(), fromlist=["load"])
        _POL[method] = mod.load(os.path.join(v13.MODELS, name))
    return v13.Controller(env, _POL[method], "greedy", label=False)


def closed_loop(method, split, name, seed, noise, cert):
    env = MAEnv(K.task(split, name), K.AGENT, noise=v13.NOISE.scaled(noise), seed=seed, horizon=4 * cert)
    return env.rollout(controller(method, env), seed=seed)


def render(res, method, split, meta, a):
    """Replay the executed schedule in the kitchen simulator, write <out>/kitchen_<method>.gif and its .json."""
    from overcooked_kitchen import pddl as KP
    from overcooked_kitchen import render as R
    from overcooked_kitchen.core import Sim
    from overcooked_kitchen.pddlkit import TemporalPlan
    from overcooked_kitchen.scenarios import load_scenario
    name = meta["instance_id"]
    plan = res.get("plan") or plan_text(res["steps"])
    sim = Sim(load_scenario(os.path.join(K.V3, split, name, "scenario.yaml")), clock="sampled", seed=0, record=True)
    m = sim.run(KP.PlanPolicy(TemporalPlan.parse(plan), KP.export(sim), "asap"))
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "kitchen_{}.gif".format(method))
    big = a.text > 0                    # large-text HUD: one line per order, 3 columns (readable on a slide)
    R.gif(sim, path, fps=a.fps, scale=a.gif_scale, cols=(a.text_cols if big else a.cols),
          hud_text=(a.text if big else None))
    if a.speed > 0:
        R.video(sim, path[:-4] + ".mp4", speed=a.speed, scale=a.scale, cols=(a.text_cols if big else a.cols),
                hud_text=(round(a.text * a.scale / a.gif_scale) if big else None))
    cert = meta["certificate_makespan"]
    info = {"simulator": "overcooked-kitchen", "method": method, "instance": name, "tier": meta.get("tier"),
            "cooks": meta.get("cooks"), "orders": meta.get("orders"), "noise": a.noise, "seed": a.seed,
            "closed_loop": {k: res[k] for k in ("success", "reason", "served", "goals", "makespan")},
            "makespan_over_reference": round(res["makespan"] / cert, 3) if res["success"] else None,
            "replay": {k: m.get(k) for k in ("success", "served", "makespan", "burnt", "late", "failure") if k in m},
            "reference_makespan": cert, "gif": path, "gif_bytes": os.path.getsize(path),
            "video": path[:-4] + ".mp4" if a.speed > 0 else None, "video_speed": a.speed,
            "split": split, "meta": meta, "plan": plan}
    json.dump(info, open(path + ".json", "w"), indent=1, default=str)
    print("wrote {} ({:.1f} MB): {} {} served {}/{}".format(path, info["gif_bytes"] / 1e6, method, name,
                                                             res["served"], res["goals"]))
    return info


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="V14")
    ap.add_argument("--compare", default="dispatcher", help="also render this method on the same instance ('' = none)")
    ap.add_argument("--split", default="v3_test")
    ap.add_argument("--tier", default="xl4", help="search this tier (m3, l4, xl4, s3) when --instance is not given")
    ap.add_argument("--instance", default=None)
    ap.add_argument("--seed", type=int, default=None, help="random durations (searched over --seeds if not given)")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--noise", type=float, default=1.0)
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--scale", type=int, default=4, help="pixels per sprite pixel (the kitchen is 4 x 15 px per tile)")
    ap.add_argument("--gif_scale", type=int, default=3, help="the GIF's scale (the video uses --scale)")
    ap.add_argument("--cols", type=int, default=2, help="columns of the order list under the kitchen")
    ap.add_argument("--speed", type=float, default=3.0,
                    help="also write an MP4 at this many simulated seconds per second of video (0: no video)")
    ap.add_argument("--text", type=int, default=26,
                    help="font size (px, at --gif_scale) of the order panel under the kitchen; 0: the small full panel")
    ap.add_argument("--text_cols", type=int, default=3, help="order columns of the large-text panel")
    ap.add_argument("--from_json", nargs="+", default=None,
                    help="re-render saved runs (<gif>.json written by this script, with their plan) without a model")
    ap.add_argument("--out", default="demos")
    a = ap.parse_args(argv)
    if a.from_json:
        for f in a.from_json:
            info = json.load(open(f))
            if not info.get("plan"):
                raise SystemExit(f + ": no saved plan (rendered before plans were saved): re-run with the model")
            a.seed, a.noise = info["seed"], info["noise"]
            render(dict(info["closed_loop"], plan=info["plan"]), info["method"], info["split"], info["meta"], a)
        return
    metas = json.load(open(os.path.join(K.V3, a.split, "dataset.json")))["instances"]
    if a.instance:
        metas = [m for m in metas if m["instance_id"] == a.instance]
    else:
        metas = [m for m in metas if m.get("tier") == a.tier]
    seeds = [a.seed] if a.seed is not None else list(range(1, a.seeds + 1))
    best, top = None, 3 if a.compare else 2     # best: (score, meta, seed, res, res_compare)
    for meta in metas:
        for seed in seeds:
            cert = meta["certificate_makespan"]
            r = closed_loop(a.method, a.split, meta["instance_id"], seed, a.noise, cert)
            rc = closed_loop(a.compare, a.split, meta["instance_id"], seed, a.noise, cert) if a.compare else None
            print("  {} seed {}: {} {}  {} {}".format(meta["instance_id"], seed, a.method, r["success"],
                                                     a.compare or "", rc["success"] if rc else ""))
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
    render(r, a.method, a.split, meta, a)
    if rc is not None:
        render(rc, a.compare, a.split, meta, a)


if __name__ == "__main__":
    main()
