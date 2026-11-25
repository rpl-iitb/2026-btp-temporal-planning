"""Tests: the shipped examples run."""
"""This file is UN-VERIFIED"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _env():
    env = dict(os.environ, SDL_VIDEODRIVER="dummy")
    env.pop("PYTHONPATH", None)
    return env


def test_quickstart_runs():
    r = subprocess.run([sys.executable, os.path.join(ROOT, "examples", "quickstart.py")], capture_output=True,
                       text=True, timeout=120, env=_env())
    assert r.returncode == 0, r.stderr
    assert "makespan" in r.stdout


def test_my_policy_bench(tmp_path):
    import json
    from overcooked_kitchen import suite
    root = os.path.join(ROOT, "benchmarks", "v1")
    rows = [r for r in suite.read_index(root) if r["split"] == "v2_val"][:2]
    small = tmp_path / "small"
    for r in rows:
        os.makedirs(os.path.dirname(small / r["path"]), exist_ok=True)
        (small / r["path"]).write_bytes(open(os.path.join(root, r["path"]), "rb").read())
    suite.update_index(str(small), rows)
    out = suite.bench(str(small), "v2_val", os.path.join(ROOT, "examples", "my_policy.py") + ":policy", seeds=1)
    assert len(out) == 2 and all(r["success"] for r in out)
    assert json.dumps(out)
