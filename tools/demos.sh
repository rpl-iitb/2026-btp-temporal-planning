#!/usr/bin/env bash
# Demo GIFs for the slides (needs the trained models): each picks a test instance where the learned policy delivers
# every order and the rule-based dispatcher does not (same random durations), and renders both runs.
#   ./run.sh demos slides           -> re-render the three runs on the slides (see below)
#   ./run.sh demos [out_dir]        -> demos/kitchen_V14.gif, kitchen_dispatcher.gif, factory_V15_hard.gif,
#                                      factory_dispatcher.gif (+ .json captions: task and outcome)
# Override with KITCHEN_METHOD, KITCHEN_TIER (m3 l4 xl4 s3), FACTORY_METHOD, FACTORY_SIZE (M L XL ...).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export PYTHONHASHSEED=0          # as in the evaluation: the dispatchers iterate over sets
if [ "${1:-}" = slides ]; then   # ./run.sh demos slides: re-render the three runs shown on the slides (same instances
                                 # and random durations, no search), with the large-text order panel
  $PY -m overcooked_test.demo --method V14 --instance v3_test_xl4_001 --seed 1 --out demos
  FACTORY_SUITE=hard $PY -m factory_test.demo --method V15_hard --instance hard_rush_L_test_000 --seed 9000 --out demos/factory_L
  FACTORY_SUITE=hard $PY -m factory_test.demo --method V15_hard --instance hard_blocking_XXXL_test_002 --seed 9000 --out demos/factory_XXXL
  ls -la demos demos/factory_L demos/factory_XXXL
  exit 0
fi
OUT=${1:-demos}
[ -f overcooked_test/results/models/${KITCHEN_METHOD:-V14}_v3.pt ] || [ -f overcooked_test/results/models/${KITCHEN_METHOD:-V14}.pt ] \
  || echo "warning: no kitchen model ${KITCHEN_METHOD:-V14} in overcooked_test/results/models" >&2
$PY -m overcooked_test.demo --method "${KITCHEN_METHOD:-V14}" --compare dispatcher --tier "${KITCHEN_TIER:-xl4}" --out "$OUT"
[ -f factory_test/data_hard/test/dataset.json ] || FACTORY_SUITE=hard $PY -m factory_test.make_data --suite hard
FACTORY_SUITE=hard $PY -m factory_test.demo --method "${FACTORY_METHOD:-V15_hard}" --compare dispatcher \
  --size "${FACTORY_SIZE:-L}" --out "$OUT"
ls -la "$OUT"
