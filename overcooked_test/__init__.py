"""overcooked_test: runs algorithm/ and baselines/ on final/overcooked_kitchen, and keeps the results.

Importing this package (or using run.sh) points algorithm/ and baselines/ at the exported kitchen instances (data/,
made by make_data.py from overcooked_kitchen's benchmark suite) and the results folder, and puts final/ and
final/overcooked_kitchen on the import path. Nothing in algorithm/, baselines/ or overcooked_kitchen/ imports this
folder; it is the only place where they meet. Every path is inside final/.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FINAL = os.path.dirname(HERE)
KITCHEN = os.path.join(FINAL, "overcooked_kitchen")
DATA = os.path.join(HERE, "data")
RESULTS = os.path.join(HERE, "results")

os.environ.setdefault("TIFLAX_DATA", DATA)
os.environ.setdefault("TIFLAX_RESULTS", RESULTS)
os.environ.setdefault("TIFLAX_BENCHES", "overcooked_kitchen={}".format(DATA))
os.environ.setdefault("TIFLAX_DOMAINS", "overcooked_kitchen={}".format(os.path.join(KITCHEN, "overcooked_kitchen", "domain.yaml")))
os.environ.setdefault("TIFLAX_TOOLS", os.path.join(FINAL, "tools", "bin"))
os.environ.setdefault("TIFLAX_FLAX_COMPLEMENTARY", os.path.join(HERE, "flax_rules", "overcooked_complementary_rules.json"))
os.environ.setdefault("TIFLAX_FLAX_RELAXATION", os.path.join(HERE, "flax_rules", "overcooked_relaxation_rules.json"))
os.environ.setdefault("OVERCOOKED_KITCHEN_POPF", os.path.join(FINAL, "tools", "bin", "popf"))
os.environ.setdefault("OVERCOOKED_KITCHEN_OPTIC", os.path.join(FINAL, "tools", "bin", "optic-clp"))
os.environ.setdefault("OVERCOOKED_KITCHEN_VAL", os.path.join(FINAL, "tools", "bin", "Validate"))
os.environ.setdefault("OVERCOOKED_KITCHEN_PARSER", os.path.join(FINAL, "tools", "bin", "Parser"))
os.environ.setdefault("PYTHONHASHSEED", "0")
for p in (KITCHEN, FINAL):
    if p not in sys.path:
        sys.path.insert(0, p)
