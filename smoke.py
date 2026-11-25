"""Light smoke check of final/ (no planner runs by default; under a minute, < 1 GB):

    .venv/bin/python smoke.py            # imports, load + ground one instance per simulator, VAL on a certificate
    .venv/bin/python smoke.py --plan     # also B1 (one POPF call, 15 s) on each; needs free memory (POPF can grow fast)

1. every algorithm version, baseline and test-harness module imports;
2. one overcooked_kitchen instance and one factory_sim instance load and ground (data exported if missing);
3. the kitchen instance's certificate is VAL-valid on the exported PDDL;
4. the plug-ins: V8S (scheduler) gives VAL-valid plans on both simulators (and V10 on the kitchen); the multi-cook
   kitchen dispatcher and the factory dispatcher finish V13's expert rollouts under random durations; the factory
   plan also runs open-loop in FactorySim;
5. with --plan: B1's plan is VAL-valid on each, and the factory plan runs in FactorySim.
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
OK = []


def step(name, fn):
    t0 = time.time()
    try:
        out = fn()
        OK.append(True)
        print("ok    {:<52} {:5.1f}s {}".format(name, time.time() - t0, out or ""), flush=True)
    except Exception as e:      # noqa: BLE001  (report and go on)
        OK.append(False)
        print("FAIL  {:<52} {:5.1f}s {}: {}".format(name, time.time() - t0, type(e).__name__, str(e)[:200]), flush=True)


PLAN = "--plan" in sys.argv


def child(code, harness):
    """Run code in a fresh interpreter with a harness package imported first (each harness sets its own paths)."""
    r = subprocess.run([PY, "-c", "PLAN = {}\nimport {}\n{}".format(PLAN, harness, code)], cwd=HERE, capture_output=True, text=True, timeout=600,
                       env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"})
    if r.returncode:
        raise RuntimeError((r.stderr or r.stdout).strip().splitlines()[-1])
    return r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""


def imports():
    mods = ["algorithm.v{}".format(i) for i in range(1, 15)] + ["algorithm.menv", "baselines.b1", "baselines.b2", "baselines.b3"]
    return child("import importlib\nfor m in {!r}: importlib.import_module(m)\nprint(len({!r}), 'modules')".format(mods, mods), "overcooked_test")


KITCHEN = """
import glob, os
from overcooked_test import DATA, make_data
need = [s for s in ("v2_train", "v3_test") if not os.path.exists(os.path.join(DATA, s, "dataset.json"))]
if need:
    make_data.main(["--splits"] + need)
from algorithm.grounding import Task, GroundTask
from overcooked_test.evaluate import build
from algorithm.pddlkit import TemporalPlan, validate_plan
d = os.path.join(DATA, "v2_train", "v2_train_s_001")
t = Task.load(d); gt = GroundTask(t)
cert = TemporalPlan.parse(open(os.path.join(d, "certificate.plan")).read())
assert validate_plan(os.path.join(DATA, "domain.pddl"), os.path.join(d, "problem.pddl"), cert).valid
msg = "{} objects, {} ground actions, certificate VAL-valid (makespan {})".format(len(t.objects), len(gt.actions), cert.makespan)
if PLAN:
    r = build("B1", t.spec).solve(t, 15)
    assert r.solved and validate_plan(os.path.join(DATA, "domain.pddl"), os.path.join(d, "problem.pddl"), TemporalPlan.parse(r.plan_text)).valid, r.notes
    msg += ", B1 makespan {}".format(r.makespan)
from algorithm.v8 import V8Planner
from algorithm import v10, v13
from algorithm.menv import MAEnv
from overcooked_test.kitchen_scheduler import KitchenScheduler
from overcooked_test import kitchen_v13 as K13
r = V8Planner("V8S", KitchenScheduler(), None, max_runs=50).solve(t, 15)
assert r.solved and r.invalid_on_full == 0, r.notes
r10 = v10.build("V10").solve(t, 15)
assert r10.solved, r10.notes
v13.use(K13)
import json
row = [x for x in json.load(open(os.path.join(DATA, "v3_test", "dataset.json")))["instances"] if "s3" in x["instance_id"]][0]
T, cert = K13.task("v3_test", row["instance_id"]), row["certificate_makespan"]
e = MAEnv(T, "cook", noise=K13.NOISE, seed=1, horizon=4 * cert)
assert e.rollout(v13.Controller(e, None, "expert", label=False))["success"]
msg += ", V8S {}, V10 {}, dispatcher (3 cooks, random durations) ok".format(r.makespan, r10.makespan)
print(msg)
"""

FACTORY = """
import os
from factory_test import DATA, make_data
d = os.path.join(DATA, "test", "base_S_test_000")
if not os.path.exists(d):
    make_data.main(["--splits", "test", "--count", "1"])
from algorithm.grounding import Task, GroundTask
from factory_test.evaluate import build
from factory_test.execute import run_plan
from algorithm.pddlkit import TemporalPlan, validate_plan
t = Task.load(d); gt = GroundTask(t)
assert gt.proof(set(t.objects)) is None
msg = "{} objects, {} ground actions, relaxed-solvable".format(len(t.objects), len(gt.actions))
if PLAN:
    r = build("B1", t.spec).solve(t, 15)
    assert r.solved, ("no plan in 15 s", r.notes)
    assert validate_plan(os.path.join(d, "domain.pddl"), os.path.join(d, "problem.pddl"), TemporalPlan.parse(r.plan_text)).valid
    m = run_plan(d, r.plan_text)
    assert m["success"], m
    msg += ", B1 makespan {}, FactorySim makespan {}".format(r.makespan, m["makespan"])
from algorithm.v8 import V8Planner
from algorithm import v13
from algorithm.menv import MAEnv
from factory_test.factory_scheduler import FactoryScheduler
from factory_test import factory_v13 as F13
r = V8Planner("V8S", FactoryScheduler(t.spec), None, max_runs=2).solve(t, 15)
assert r.solved and validate_plan(os.path.join(d, "domain.pddl"), os.path.join(d, "problem.pddl"), TemporalPlan.parse(r.plan_text)).valid, r.notes
ran = run_plan(d, r.plan_text)
assert ran["success"], ("the VAL-valid V8S plan did not run in FactorySim", ran)
v13.use(F13)
name, T, ref, meta = F13.load_split("test")[0]
e = MAEnv(T, "robot", noise=F13.NOISE, seed=1, horizon=4 * ref)
assert e.rollout(v13.Controller(e, None, "expert", label=False))["success"]
msg += ", V8S {} (VAL-valid, runs in FactorySim: makespan {}), dispatcher (random durations) ok".format(r.makespan, ran["makespan"])
print(msg)
"""

if __name__ == "__main__":
    step("imports: algorithm v1-v14, baselines b1-b3", imports)
    step("overcooked_kitchen: load, ground, VAL, V8S, V10, V13 expert" + (" + B1" if PLAN else ""), lambda: child(KITCHEN, "overcooked_test"))
    step("factory_sim: load, ground, V8S + run, V13 expert" + (" + B1" if PLAN else ""), lambda: child(FACTORY, "factory_test"))
    print("{} / {} passed".format(sum(OK), len(OK)))
    sys.exit(0 if all(OK) else 1)
