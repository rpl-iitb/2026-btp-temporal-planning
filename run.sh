#!/usr/bin/env bash
# Entry point for final/; see README.md.   ./run.sh help
set -euo pipefail
cd "$(dirname "$0")"
unset PYTHONPATH                       # a ROS PYTHONPATH leaks into the venv otherwise
export PYTHONHASHSEED=${PYTHONHASHSEED:-0}

py() { [ -d .venv ] || { echo "no .venv: run ./run.sh setup first" >&2; exit 1; }; .venv/bin/python "$@"; }

cmd=${1:-help}; shift || true
case "$cmd" in
  setup)
    python3 -m venv .venv
    .venv/bin/pip install --upgrade pip
    .venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.5.1"
    .venv/bin/pip install -r requirements.txt
    .venv/bin/pip install --no-deps -e factory_sim -e overcooked_kitchen
    chmod +x tools/bin/popf tools/bin/optic-clp tools/bin/Validate tools/bin/Parser ;;
  smoke) py smoke.py "$@" ;;
  data) py -m overcooked_test.make_data && py -m factory_test.make_data ;;
  kitchen) m=$1; b=${2:-30}; shift 2 || shift; py -m overcooked_test.evaluate --methods "$m" --budgets "$b" "$@" ;;
  factory) m=$1; b=${2:-30}; shift 2 || shift; py -m factory_test.evaluate --methods "$m" --budgets "$b" "$@" ;;
  test) py -m pytest -q overcooked_test/tests factory_test/tests "$@" ;;
  py) py "$@" ;;                                                   # the venv's python with the project environment
  kitchen-all) overcooked_test/run_kitchen_all.sh "$@" ;;          # group main (or the groups / stages given)
  kitchen-baselines) overcooked_test/run_kitchen_all.sh baselines ;;
  kitchen-ablations) overcooked_test/run_kitchen_all.sh ablations ;;
  factory-all) factory_test/run_factory_all.sh "$@" ;;
  factory-baselines) factory_test/run_factory_all.sh baselines ;;
  factory-ablations) factory_test/run_factory_all.sh ablations ;;
  factory-hard) factory_test/run_factory_hard.sh "$@" ;;               # the hard FactorySim suite (main, ablations, seeds)
  demos) tools/demos.sh "$@" ;;                                        # demo GIFs of the trained policies (for the slides)
  all)                                                             # data if missing, then everything on both simulators
    [ -f overcooked_test/data/v3_train/dataset.json ] || py -m overcooked_test.make_data
    [ -f factory_test/data/train/dataset.json ] || py -m factory_test.make_data
    overcooked_test/run_kitchen_all.sh main ablations
    factory_test/run_factory_all.sh main baselines ablations ;;
  *) sed -n '/^```bash/,/^```/p' README.md | sed '1d;$d' ;;
esac
