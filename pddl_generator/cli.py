"""Commands: scenarios -> HL problems + PDDL with checks (default), and `check`: POPF -> VAL -> simulator.

    python -m pddl_generator.cli [paths] [--quiet]
    python -m pddl_generator.cli check [generated_pddl dirs] [--timeout 60]
"""

"""This file is UN-VERIFIED"""

import argparse
import collections
import datetime
import glob
import hashlib
import json
import os
import sys
import time

from .planners import parse_check, run_temporal_planner, validate_plan
from .problem import HLProblem, get_benchmark, validate_json_schema
from .writer import check_domain_roundtrip, check_problem_roundtrip, deadline_literals, write_domain, write_problem


# ==================== scenarios -> PDDL ==========================================================
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def _code_sha():
    h = hashlib.sha256()
    for f in sorted(glob.glob(os.path.join(ROOT, "pddl_generator", "*.py"))):
        with open(f, "rb") as fh:
            h.update(fh.read())
    return h.hexdigest()


def problem_size(spec, prob):
    static = prob.static_predicates(spec)
    return {
        "objects": len(prob.objects),
        "objects_by_type": dict(sorted(collections.Counter(o.type for o in prob.objects).items())),
        "init_facts": len(prob.facts),
        "init_facts_static": sum(1 for f in prob.facts if f[0] in static),
        "numeric_fluents": len(prob.fluents),
        "goals": len(prob.goals),
        "goals_with_deadline": sum(1 for g in prob.goals if g.deadline is not None),
        "timed_literals": len(prob.timed_literals) + len(deadline_literals(spec, prob)),
    }


def write_shared_domain(bench):
    spec = bench.domain_spec()
    text = write_domain(spec)
    errs = check_domain_roundtrip(spec, text)
    if errs:
        raise RuntimeError("domain round trip failed: {}".format(errs))
    os.makedirs(bench.pddl_dir, exist_ok=True)
    path = os.path.join(bench.pddl_dir, "domain.pddl")
    with open(path, "w") as f:
        f.write(text)
    return path, _sha(text)


def generate(bench, scenario_path, domain_path, domain_sha):
    spec = bench.domain_spec()
    t0 = time.perf_counter()
    sc = bench.load_scenario(scenario_path)
    prob = bench.build_problem(sc)
    t_build = time.perf_counter() - t0
    t1 = time.perf_counter()
    text = write_problem(spec, prob)
    t_write = time.perf_counter() - t1

    out = os.path.join(bench.pddl_dir, os.path.basename(os.path.dirname(os.path.abspath(scenario_path))),
                       prob.instance_id)
    os.makedirs(out, exist_ok=True)
    ppath, hpath = os.path.join(out, "problem.pddl"), os.path.join(out, "hl_problem.json")
    with open(ppath, "w") as f:
        f.write(text)
    prob.save(hpath)

    checks = {"hl_typing": "ok"}             # build_problem already ran prob.validate(spec)
    try:
        validate_json_schema(prob.to_dict())
        checks["hl_json_schema"] = "ok"
    except Exception as e:  # noqa: BLE001 -- recorded and reported as a failure
        checks["hl_json_schema"] = "FAIL: {}".format(e)
    checks["problem_roundtrip"] = check_problem_roundtrip(spec, prob, text) or "ok"
    pc = parse_check(domain_path, ppath)
    checks["val_parser"] = {"ok": pc.ok, "errors": pc.errors, "warnings": pc.warnings}
    ok = all(v == "ok" for k, v in checks.items() if k != "val_parser") and pc.ok

    rel = lambda p: os.path.relpath(p, ROOT)  # noqa: E731
    manifest = {
        "instance_id": prob.instance_id,
        "benchmark": bench.name,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "inputs": {
            "scenario": rel(sc["_path"]), "scenario_sha256": sc["_sha256"],
            "domain_spec": [rel(f) for f in spec.source_files], "domain_spec_sha256": spec.sha256(),
            "simulator": prob.provenance.get("simulator"),
            "benchmark_code_sha256": prob.provenance.get("adapter_code_sha256"),
            "generator_code_sha256": _code_sha(),
        },
        "outputs": {
            "domain_pddl": rel(domain_path), "domain_pddl_sha256": domain_sha,
            "problem_pddl": rel(ppath), "problem_pddl_sha256": _sha(text),
            "hl_problem": rel(hpath), "hl_problem_sha256": prob.sha256(),
        },
        "checks": checks,
        "checks_passed": ok,
        "size": problem_size(spec, prob),
        "timing_s": {"build_hl": round(t_build, 4), "write_pddl": round(t_write, 4)},
    }
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    return manifest


def scenario_files(bench, paths):
    files = []
    for p in paths or [bench.scenario_dir]:
        if os.path.isdir(p):
            files += sorted(glob.glob(os.path.join(p, "**", "*.yaml"), recursive=True))
        else:
            files.append(p)
    return files


def generate_main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", help="scenario files or folders (default: the benchmark's scenarios/)")
    ap.add_argument("--benchmark", default="overcooked_ai")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    bench = get_benchmark(args.benchmark)
    dpath, dsha = write_shared_domain(bench)
    index = collections.defaultdict(list)
    failed = 0
    for f in scenario_files(bench, args.paths):
        m = generate(bench, f, dpath, dsha)
        s = m["size"]
        index[os.path.dirname(os.path.dirname(m["outputs"]["problem_pddl"]))].append(
            {"instance_id": m["instance_id"], "checks_passed": m["checks_passed"],
             **{k: s[k] for k in ("objects", "init_facts", "numeric_fluents", "goals", "goals_with_deadline")}})
        if not m["checks_passed"]:
            failed += 1
            print("FAIL", m["instance_id"], json.dumps(m["checks"]))
        elif not args.quiet:
            print("{:<32} OK  objects={:<3} facts={:<4} fluents={:<4} goals={} deadlines={}".format(
                m["instance_id"], s["objects"], s["init_facts"], s["numeric_fluents"], s["goals"],
                s["goals_with_deadline"]))
    for set_dir, rows in index.items():
        with open(os.path.join(ROOT, set_dir, "index.json"), "w") as f:
            json.dump(sorted(rows, key=lambda r: r["instance_id"]), f, indent=1)
    n = sum(len(r) for r in index.values())
    print("{} instances, {} failed checks -> {}".format(n, failed, os.path.relpath(bench.pddl_dir, ROOT)))
    return 1 if failed else 0


# ==================== check: PDDL -> POPF -> VAL -> simulator ====================================


def instance_dirs(paths, bench):
    found = []
    for p in paths or [bench.pddl_dir]:
        found += [os.path.dirname(m) for m in glob.glob(os.path.join(p, "**", "manifest.json"), recursive=True)]
    return sorted(set(found))


def check_instance(bench, d, timeout, plan_dir=None):
    with open(os.path.join(d, "manifest.json")) as f:
        man = json.load(f)
    dom = os.path.join(ROOT, man["outputs"]["domain_pddl"])
    pfile = os.path.join(d, "problem.pddl")
    prob = HLProblem.load(os.path.join(d, "hl_problem.json"))
    r = run_temporal_planner(dom, pfile, timeout=timeout)
    row = {"instance": os.path.relpath(d, bench.pddl_dir), "planner": "popf", "planner_status": r.status,
           "planner_runtime_s": round(r.runtime_s, 3), "states_evaluated": r.states_evaluated,
           "peak_rss_mb": r.peak_rss_mb and round(r.peak_rss_mb, 1),
           "problem_pddl_sha256": man["outputs"]["problem_pddl_sha256"]}
    if r.plan is not None:
        if plan_dir:
            with open(os.path.join(plan_dir, row["instance"].replace(os.sep, "__") + ".plan"), "w") as f:
                f.write(r.plan.to_text())
        v = validate_plan(dom, pfile, r.plan)
        row.update({"val_valid": v.valid, "val_value": v.value, "val_errors": v.errors[:10],
                    "plan_makespan": round(r.plan.makespan, 3), "plan_actions": len(r.plan),
                    "plan_total_action_time": round(r.plan.total_action_time, 3)})
        for mode in ("timed", "asap"):
            row["sim_" + mode] = bench.execute(prob, r.plan, mode=mode).summary()
    return row


def check(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", help="generated_pddl folders or instance folders")
    ap.add_argument("--benchmark", default="overcooked_ai")
    ap.add_argument("--timeout", type=float, default=60)
    args = ap.parse_args(argv)
    bench = get_benchmark(args.benchmark)
    run = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(ROOT, "results", "pddl_check", run)
    os.makedirs(os.path.join(out, "plans"), exist_ok=True)
    rows = []
    for d in instance_dirs(args.paths, bench):
        row = check_instance(bench, d, args.timeout, os.path.join(out, "plans"))
        rows.append(row)
        t, a = row.get("sim_timed", {}), row.get("sim_asap", {})
        print("{:<38} {:<9} {:>6.2f}s VAL={!s:<5} plan_ms={!s:<8} | timed: ok={!s:<5} deadlines={!s:<5} "
              "ms={!s:<4} maxdev={!s:<3} detours={!s:<3} | asap: ok={!s:<5} ms={!s:<4}".format(
                  row["instance"], row["planner_status"], row["planner_runtime_s"], row.get("val_valid"),
                  row.get("plan_makespan"), t.get("success"), t.get("deadlines_met"), t.get("sim_makespan"),
                  t.get("max_abs_deviation"), t.get("detour_steps"), a.get("success"), a.get("sim_makespan")))
    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump({"run": run, "timeout_s": args.timeout, "results": rows}, f, indent=1)
    print("written", os.path.relpath(out, ROOT))


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] == "check":
        return check(argv[1:])
    return generate_main(argv)


if __name__ == "__main__":
    sys.exit(main())
