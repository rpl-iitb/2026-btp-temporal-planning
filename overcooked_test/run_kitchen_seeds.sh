#!/usr/bin/env bash
# Training-seed repeats of the multi-cook policies and their ablations, for confidence intervals.
#
#   overcooked_test/run_kitchen_seeds.sh [<stage> ...]     (from final/; default: every stage below)
#   SEEDS="1 2" PROCS=8 DRY_RUN=1 ...
#
# Seed 0 is the existing models (V14_v3, V14_nofeats, ...). For each seed s in $SEEDS this trains V14_s<s>,
# V14_nofeats_s<s>, V14_nomaxplus_s<s>, V14_il_s<s> and V13_s<s> (algorithm.v13 --train_seed s), then seeds_eval
# evaluates all of them, with seed 0, on the 60 multi-cook test kitchens at noise x1-x3, 3 seeds
# (overcooked_test/results/v13/eval_v3_test_seeds.md). Logs, .done markers and resuming as in run_kitchen_all.sh.
set -uo pipefail
cd "$(dirname "$0")/.."
unset PYTHONPATH
export PYTHONHASHSEED=0
PY=.venv/bin/python
LOGS=overcooked_test/results/logs
mkdir -p "$LOGS"
DOM=overcooked_test.kitchen_v13
PROCS=${PROCS:-6}
SEEDS=${SEEDS:-"1 2"}
if [ "${DRY_RUN:-0}" != 1 ] && [ ! -f overcooked_test/data/v3_train/dataset.json ]; then
  echo "no kitchen data: run ./run.sh data first" >&2
  exit 1
fi
export TIFLAX_RESULTS=$PWD/overcooked_test/results TIFLAX_DATA=$PWD/overcooked_test/data
eval "$($PY -c "import overcooked_test, os, shlex; print('\n'.join('export {}={}'.format(k, shlex.quote(v)) for k, v in os.environ.items() if k.startswith(('TIFLAX', 'OVERCOOKED'))))")"

$PY tools/memguard.py --root $$ --min_mb "${MIN_MB:-1500}" --log "$LOGS/memguard.log" >/dev/null 2>&1 &
GUARD=$!
systemd-inhibit --what=sleep:idle --who=btp-experiments --why="kitchen seed repeats" --mode=block sleep infinity 2>/dev/null &
INHIBIT=$!
trap 'kill $GUARD $INHIBIT 2>/dev/null' EXIT

stage() {      # stage <name> <command...>
  local name=$1; shift
  if [ "${DRY_RUN:-0}" = 1 ]; then echo "$name: $*"; return; fi
  if [ -f "$LOGS/$name.done" ]; then echo "$(date '+%F %T') skip $name (done)" | tee -a "$LOGS/driver.log"; return; fi
  echo "$(date '+%F %T') start $name: $*" | tee -a "$LOGS/driver.log"
  local t0=$SECONDS
  if "$@" > "$LOGS/$name.log" 2>&1; then
    touch "$LOGS/$name.done"
    echo "$(date '+%F %T') done  $name ($((SECONDS - t0)) s)" | tee -a "$LOGS/driver.log"
  else
    echo "$(date '+%F %T') FAIL  $name (exit $?, $((SECONDS - t0)) s), see $LOGS/$name.log" | tee -a "$LOGS/driver.log"
  fi
}

V14=(--procs "$PROCS" --threads "$PROCS")
train() {      # train <seed> <variant>
  local s=$1 v=$2
  case "$v" in
    V14)           stage seed_V14_s$s           $PY -m algorithm.v14 --domain $DOM train --name V14_s$s --train_seed $s "${V14[@]}" ;;
    V14_nofeats)   stage seed_V14_nofeats_s$s   $PY -m algorithm.v14 --domain $DOM train --name V14_nofeats_s$s --no_opt_feats --train_seed $s "${V14[@]}" ;;
    V14_nomaxplus) stage seed_V14_nomaxplus_s$s $PY -m algorithm.v14 --domain $DOM train --name V14_nomaxplus_s$s --no_maxplus --train_seed $s "${V14[@]}" ;;
    V14_il)        stage seed_V14_il_s$s        $PY -m algorithm.v14 --domain $DOM train --name V14_il_s$s --rl_epochs 0 --train_seed $s "${V14[@]}" ;;
    V13)           stage seed_V13_s$s           $PY -m algorithm.v13 --domain $DOM train --name V13_s$s --train_seed $s --procs "$PROCS" ;;
  esac
}
VARIANTS=(V14 V14_nofeats V14_nomaxplus V14_il V13)

seeds_eval() {
  local m=(V13 V14 V14_nofeats V14_nomaxplus V14_il)
  for s in $SEEDS; do for v in "${VARIANTS[@]}"; do m+=("${v}_s$s"); done; done
  stage seeds_eval $PY -m overcooked_test.v3_eval --split v3_test --tag _seeds --noise 1 2 3 --seeds 3 --procs "$PROCS" --methods "${m[@]}"
}

[ $# -eq 0 ] && set -- train eval
echo "$(date '+%F %T') seed driver pid $$ seeds: $SEEDS stages: $*" | tee -a "$LOGS/driver.log"
for st in "$@"; do
  case "$st" in
    train) for s in $SEEDS; do for v in "${VARIANTS[@]}"; do train "$s" "$v"; done; done ;;
    eval)  seeds_eval ;;
    *) echo "unknown stage $st (train, eval)" >&2 ;;
  esac
done
echo "$(date '+%F %T') seed driver finished" | tee -a "$LOGS/driver.log"
