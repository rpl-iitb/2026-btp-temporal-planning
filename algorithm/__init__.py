"""T-iFLAX: core modules, versions v1 ... v7, evaluation. Paths shared by every module."""
import glob
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDDL = os.path.join(ROOT, "overcooked_ai", "generated_pddl")
RESULTS = os.path.join(ROOT, "results")


def results_dir(name):
    """results/<name>/ (created): b1 .. b3, v1 .. v7, diagnostics."""
    d = os.path.join(RESULTS, name)
    os.makedirs(d, exist_ok=True)
    return d


def split_dirs(split):
    """Instance folders of a dataset split (overcooked_ai/generated_pddl/<split>/*), sorted."""
    return sorted(d for d in glob.glob(os.path.join(PDDL, split, "*")) if os.path.isdir(d))
