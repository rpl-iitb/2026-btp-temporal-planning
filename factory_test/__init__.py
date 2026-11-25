"""factory_test: runs algorithm/ and baselines/ on final/factory_sim, and keeps the results.

Importing this package (or using run.sh) points algorithm/ and baselines/ at the exported factory instances (data/,
made by make_data.py from FactorySim's scenario families) and the results folder, and puts final/ and final/factory_sim
on the import path. Nothing in algorithm/, baselines/ or factory_sim/ imports this folder. Every path is inside final/.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FINAL = os.path.dirname(HERE)
FACTORY = os.path.join(FINAL, "factory_sim")
# FACTORY_SUITE=hard: the hard suite (lane traffic, deadlines, plants up to XXXL) in data_hard/ and results_hard/
SUITE = os.environ.get("FACTORY_SUITE", "easy")
DATA = os.environ.get("FACTORY_DATA") or os.path.join(HERE, "data_hard" if SUITE == "hard" else "data")
RESULTS = os.environ.get("FACTORY_RESULTS") or os.path.join(HERE, "results_hard" if SUITE == "hard" else "results")

os.environ.setdefault("TIFLAX_DATA", DATA)
os.environ.setdefault("TIFLAX_RESULTS", RESULTS)
os.environ.setdefault("TIFLAX_BENCHES", "factorysim={}".format(DATA))
os.environ.setdefault("TIFLAX_TOOLS", os.path.join(FINAL, "tools", "bin"))
os.environ.setdefault("TIFLAX_FLAX_COMPLEMENTARY", os.path.join(HERE, "flax_rules", "factory_complementary_rules.json"))
os.environ.setdefault("TIFLAX_FLAX_RELAXATION", os.path.join(HERE, "flax_rules", "factory_relaxation_rules.json"))
os.environ.setdefault("FACTORYSIM_POPF", os.path.join(FINAL, "tools", "bin", "popf"))
os.environ.setdefault("FACTORYSIM_VALIDATE", os.path.join(FINAL, "tools", "bin", "Validate"))
os.environ.setdefault("PYTHONHASHSEED", "0")
for p in (FACTORY, FINAL):
    if p not in sys.path:
        sys.path.insert(0, p)
