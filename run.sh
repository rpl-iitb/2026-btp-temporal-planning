#!/usr/bin/env bash
# One entry point for everything; see README.md.  ./run.sh help
set -euo pipefail
cd "$(dirname "$0")"
unset PYTHONPATH                       # a system PYTHONPATH (e.g. ROS) leaks into the venv otherwise
export PYTHONHASHSEED=${PYTHONHASHSEED:-0}

usage() { sed -n '/^# usage/,/^# end/p' "$0" | sed 's/^# \{0,1\}//; /^usage$/d; /^end$/d'; }
# usage
#   ./run.sh setup                          venv + packages (once)
#   ./run.sh <METHOD> [budget_s] [args]     evaluate one method on the 60 stress problems
#                                           METHOD: B1 B2 B3 V1 V2 V2.1 V3 V3U V3.1 V4 V4.1 V4.2 V5
#   ./run.sh eval  M1,M2 [budget_s] [args]  several methods (args: see python -m algorithm.evaluate -h)
#   ./run.sh solve METHOD DIR [budget_s]    one instance folder, prints the plan
#   ./run.sh train V1|V2|V2.1|B2|B3 [args]  bilevel / FLAX training -> results/<method>/
#   ./run.sh v5|v6|v7 CMD [args]            V5-V7 pipelines (labels, train, tests); see algorithm/README.md
#   ./run.sh exp NAME [args]                quick experiments (oracle, cp_select, ...) -> results/diagnostics/
#   ./run.sh report TAG                     summary + plots of a finished run, e.g. stress_30s_seed0
#   ./run.sh all [seed]                     train B2 B3 V1 V2 V2.1, then evaluate everything at 30 s and 300 s
#   ./run.sh pddl [paths]                   scenarios -> PDDL;   ./run.sh check [dir]: POPF -> VAL -> simulator
#   ./run.sh data CONFIG.yaml [args]        generate a dataset (configs/dataset_*.yaml)
#   ./run.sh test                           pytest
# end

py() { [ -d .venv ] || { echo "no .venv: run ./run.sh setup first" >&2; exit 1; }; .venv/bin/python "$@"; }

setup() {
  if [ ! -d overcooked_ai/repo ]; then
    git clone https://github.com/HumanCompatibleAI/overcooked_ai.git overcooked_ai/repo
    git -C overcooked_ai/repo checkout a60e31a && rm -rf overcooked_ai/repo/.git   # tag 1.1.0, pinned
  fi
  python3 -m venv .venv
  .venv/bin/pip install --upgrade pip
  .venv/bin/pip install "numpy<2" "gym==0.26.2" pygame tqdm ipython ipywidgets pyyaml jsonschema psutil setuptools
  .venv/bin/pip install --no-deps -e overcooked_ai/repo
  .venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.5.1"
  .venv/bin/pip install "numpy==1.26.4" "pandas<2.3" "matplotlib<3.10" ortools pytest   # pandas: FLAX's GNN code
  .venv/bin/pip install --no-deps --no-build-isolation -e .
  chmod +x tools/bin/popf tools/bin/optic-clp tools/bin/Validate tools/bin/Parser
}

cmd=${1:-help}; shift || true
case "$cmd" in
  setup) setup ;;
  B1|B2|B3|V1|V2|V2.1|V3|V3U|V3.1|V4|V4.1|V4.2|V5)
    b=${1:-30}; shift || true
    py -m algorithm.evaluate --methods "$cmd" --budgets "$b" "$@" ;;
  eval) m=$1; b=${2:-30}; shift 2 || shift; py -m algorithm.evaluate --methods "$m" --budgets "$b" "$@" ;;
  solve) py -m algorithm.evaluate --methods "$1" --instance "$2" --budgets "${3:-30}" ;;
  train)
    m=$1; shift
    case "$m" in
      V1) py -m algorithm.v1 train "$@" ;;
      V2|V2.1) py -m algorithm.v2 train --method "$m" "$@" ;;
      B2) py -m baselines.b2 train "$@" ;;
      B3) py -m baselines.b3 train "$@" ;;
      *) echo "train: V1 V2 V2.1 B2 B3 (V5-V7: ./run.sh v5|v6|v7 train)" >&2; exit 1 ;;
    esac ;;
  v5|v6|v7) py -m "algorithm.$cmd" "$@" ;;
  exp) py -m algorithm.experiments "$@" ;;
  report) py -m algorithm.evaluate --report "$1" ;;
  all)
    s=${1:-0}
    py -m baselines.b2 train --seed "$s"
    py -m baselines.b3 train --seed "$s" --epochs 6 --budget 15
    py -m algorithm.v1 train --seed "$s" --epochs 6 --budget 15
    py -m algorithm.v2 train --method V2 --seed "$s" --epochs 6 --budget 15
    py -m algorithm.v2 train --method V2.1 --seed "$s" --epochs 6 --budget 15
    py -m algorithm.evaluate --methods B1,B2,B3,V1,V2,V2.1,V3,V3U,V3.1,V4,V4.1,V4.2 --budgets 30 --seed "$s" \
       --max_load 10 --plot
    py -m algorithm.evaluate --methods B1,B2,B3,V1,V3,V3.1 --unlimited 300 --limit 2 --seed "$s" --max_load 10 ;;
  pddl) py -m pddl_generator.cli "$@" ;;
  check) py -m pddl_generator.cli check "$@" ;;
  data) py -m overcooked_ai.make_dataset "$@" ;;
  test) py -m pytest -q tests ;;
  *) usage ;;
esac
