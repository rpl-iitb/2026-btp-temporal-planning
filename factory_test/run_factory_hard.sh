#!/usr/bin/env bash
# The hard FactorySim suite, unattended: data, training of the closed-loop policies, evaluation, ablations, seeds.
#
#   factory_test/run_factory_hard.sh [<group | stage> ...]   (from final/; default: main)
#   DRY_RUN=1 ... prints each stage's command without running it
#   ./run.sh factory-hard [<group | stage> ...]               (the same)
#
# The hard suite (FactorySim's SUITE_HARD, factory_test/make_data.py --suite hard): lane traffic (corridor junctions
# hold one robot; robots wait, back off and can deadlock), hard due dates from the workload, contention, blocking
# buffers, alternative routes, online and rush orders; train on the lab (S, M, L: 2-4 robots, 4-14 orders), test on
# M, L and plants of 2-4 production lines joined by cross aisles (XL, XXL, XXXL: 6-12 robots, 24-56 orders).
# Data in factory_test/data_hard/, results in factory_test/results_hard/ (the earlier suite in data/ and results/ is
# left as it is).
#
# Groups (a group name expands to its stages; default: main):
#   main       data_hard v13_train v14_train v12_train closed_eval plan_eval
#              closed_eval: dispatcher, open loop, re-planning (sizes M, L), V13, V14, V12 (up to XL) on the test split,
#              noise x1-x3, 3 seeds (results_hard/v13/eval_test.md, by size and by family)
#              plan_eval: the planners B1 (POPF), V4.2, V8S, V10 with 60 s per instance (results_hard/summary/)
#   ablations  V14 without RL / with REINFORCE / without max-plus / without option features, V13 without RL:
#              abl_*_train, then abl_eval on sizes M-XL (results_hard/v13/eval_test_ablations.md)
#   seeds      training seeds 1 and 2 of V14, V14 without option features and V13 (seed_*), then seeds_eval
#              on sizes M-XL (results_hard/v13/eval_test_seeds.md)
#   v15        training seeds 1 and 2 of V15 (from V14_hard, the plant curriculum of factory_v15.py, the settings of
#              V15_hard), then v15_eval: V15_hard and its seeds on all sizes (results_hard/v13/eval_test_v15.md)
# Several drivers can run at once on one list (e.g. "ablations seeds" and "v15 seeds"): a stage another driver is
# running (<stage>.running holds its pid) is skipped, and an evaluation stage first waits for the training stages
# it evaluates. WAIT_FOR=<stage> makes a driver wait until that stage is done before it starts.
# Each stage logs to factory_test/results_hard/logs/<stage>.log and writes <stage>.done when it succeeds; a stage
# already done is skipped (delete the .done file to redo it), a failed stage is logged and the next one runs.
# tools/memguard.py stops the largest stage of this driver if available memory gets low.
set -uo pipefail
cd "$(dirname "$0")/.."
unset PYTHONPATH
export PYTHONHASHSEED=0 FACTORY_SUITE=hard
PY=.venv/bin/python
LOGS=factory_test/results_hard/logs
mkdir -p "$LOGS"
DOM=factory_test.factory_v13
PROCS=${PROCS:-6}
export TIFLAX_RESULTS=$PWD/factory_test/results_hard TIFLAX_DATA=$PWD/factory_test/data_hard
eval "$($PY -c "import factory_test, os, shlex; print('\n'.join('export {}={}'.format(k, shlex.quote(v)) for k, v in os.environ.items() if k.startswith('TIFLAX')))")"
MAIN=(data_hard v13_train v14_train v12_train closed_eval plan_eval)
ABL=(V14_hard_il V14_hard_reinforce V14_hard_nomaxplus V14_hard_nofeats V13_hard_il)
ABLATIONS=($(for n in "${ABL[@]}"; do echo "abl_${n}_train"; done) abl_eval)
SEEDS=(seed_V14_hard_s1 seed_V14_hard_s2 seed_V14_hard_nofeats_s1 seed_V14_hard_nofeats_s2 seed_V13_hard_s1 seed_V13_hard_s2 seeds_eval)
V15=(v15_s1_train v15_s2_train v15_eval)
ALL=("${MAIN[@]}" "${ABLATIONS[@]}" "${SEEDS[@]}" "${V15[@]}")
[ $# -eq 0 ] && set -- main
args=()
for a in "$@"; do
  case "$a" in
    main) args+=("${MAIN[@]}") ;;
    ablations) args+=("${ABLATIONS[@]}") ;;
    seeds) args+=("${SEEDS[@]}") ;;
    v15) args+=("${V15[@]}") ;;
    all) args+=("${ALL[@]}") ;;
    *) args+=("$a") ;;
  esac
done
set -- "${args[@]}"

$PY tools/memguard.py --root $$ --min_mb "${MIN_MB:-1500}" --log "$LOGS/memguard.log" >/dev/null 2>&1 &
GUARD=$!
systemd-inhibit --what=sleep:idle --who=btp-experiments --why="factory hard suite" --mode=block sleep infinity 2>/dev/null &
INHIBIT=$!
trap 'kill $GUARD $INHIBIT 2>/dev/null' EXIT

stage() {      # stage <name> <command...>
  local name=$1; shift
  if [ "${DRY_RUN:-0}" = 1 ]; then echo "$name: $*"; return; fi
  if [ -f "$LOGS/$name.done" ]; then echo "$(date '+%F %T') skip $name (done)" | tee -a "$LOGS/driver.log"; return; fi
  if running "$name"; then echo "$(date '+%F %T') skip $name (another driver runs it)" | tee -a "$LOGS/driver.log"; return; fi
  echo $$ > "$LOGS/$name.running"
  if [ "$name" != data_hard ] && [ ! -f factory_test/data_hard/test/dataset.json ]; then
    echo "$(date '+%F %T') FAIL  $name: no hard data (stage data_hard)" | tee -a "$LOGS/driver.log"; return
  fi
  echo "$(date '+%F %T') start $name: $*" | tee -a "$LOGS/driver.log"
  local t0=$SECONDS
  if "$@" > "$LOGS/$name.log" 2>&1; then
    touch "$LOGS/$name.done"
    echo "$(date '+%F %T') done  $name ($((SECONDS - t0)) s)" | tee -a "$LOGS/driver.log"
  else
    echo "$(date '+%F %T') FAIL  $name (exit $?, $((SECONDS - t0)) s), see $LOGS/$name.log" | tee -a "$LOGS/driver.log"
  fi
  rm -f "$LOGS/$name.running"
}
running() {    # running <stage>: another live driver runs it
  local f=$LOGS/$1.running
  [ -f "$f" ] && [ "$(cat "$f")" != $$ ] && kill -0 "$(cat "$f")" 2>/dev/null
}
await() {      # await <stage...>: wait while another driver runs any of them
  [ "${DRY_RUN:-0}" = 1 ] && return
  local s
  for s in "$@"; do
    while running "$s"; do sleep 60; done
  done
}

flags() {      # the ablation / variant flags of a model name
  case "$1" in
    *_il*) echo --rl_epochs 0 ;; *_reinforce*) echo --rl reinforce ;;
    *_nomaxplus*) echo --no_maxplus ;; *_nofeats*) echo --no_opt_feats ;; esac
}
train() {      # train <stage> <model name> [extra flags]
  local st=$1 n=$2; shift 2
  local mod=v14 threads=(--threads "$PROCS"); [[ $n == V13* ]] && { mod=v13; threads=(); }; [[ $n == V12* ]] && { mod=v12; threads=(); }
  stage "$st" $PY -m algorithm.$mod --domain $DOM train --name "$n" $(flags "$n") "$@" --procs "$PROCS" "${threads[@]}"
}
EVAL=(--split test --noise 1 2 3 --seeds 3 --procs "$PROCS")
SMALL=(--sizes M L XL)        # ablations and seed repeats: the sizes up to XL (the main evaluation has all sizes)
run() {
  case "$1" in
    data_hard)  stage data_hard $PY -m factory_test.make_data --suite hard --procs "$PROCS" ;;
    v13_train)  train v13_train V13_hard ;;
    v14_train)  train v14_train V14_hard ;;
    v12_train)  train v12_train V12_hard ;;
    closed_eval) stage closed_eval $PY -m factory_test.closed_eval "${EVAL[@]}" --methods dispatcher open-loop replan-2 V13_hard V14_hard V12_hard \
                   --max_size V12_hard:XL ;;
    plan_eval)  stage plan_eval $PY -m factory_test.evaluate --split test --budgets 60 --jobs 2 --seed 0 --methods B1,V4.2,V8S,V10 ;;
    abl_*_train) local n=${1#abl_}; train "$1" "${n%_train}" ;;
    abl_eval)   await "${ABLATIONS[@]}"; stage abl_eval $PY -m factory_test.closed_eval "${EVAL[@]}" "${SMALL[@]}" --tag _ablations \
                  --methods V13_hard V13_hard_il V14_hard V14_hard_il V14_hard_reinforce V14_hard_nomaxplus V14_hard_nofeats ;;
    seed_*)     local n=${1#seed_}; train "$1" "$n" --train_seed "${n##*_s}" ;;
    seeds_eval) await "${SEEDS[@]}"; stage seeds_eval $PY -m factory_test.closed_eval "${EVAL[@]}" "${SMALL[@]}" --tag _seeds \
                  --methods V14_hard V14_hard_s1 V14_hard_s2 V14_hard_nofeats V14_hard_nofeats_s1 V14_hard_nofeats_s2 V13_hard V13_hard_s1 V13_hard_s2 ;;
    v15_s*_train) local k=${1#v15_s}; k=${k%_train}
                stage "$1" $PY -m algorithm.v15 --domain factory_test.factory_v15 train --name "V15_hard_s$k" --init V14_hard \
                  --train_seed "$k" --dagger 2 --bc_epochs 2 --seeds 1 --rl_epochs 6 --rl_kitchens 24 --rollouts 8 --val_seeds 1 \
                  --val_every 2 --procs "$PROCS" --threads 6 ;;
    v15_eval)   await "${V15[@]}"; stage v15_eval $PY -m factory_test.closed_eval "${EVAL[@]}" --tag _v15 \
                  --methods V15_hard V15_hard_s1 V15_hard_s2 ;;
    *) echo "$(date '+%F %T') unknown stage $1 (stages: ${ALL[*]})" | tee -a "$LOGS/driver.log" ;;
  esac
}

echo "$(date '+%F %T') factory hard driver pid $$ stages: $*" | tee -a "$LOGS/driver.log"
if [ -n "${WAIT_FOR:-}" ] && [ "${DRY_RUN:-0}" != 1 ]; then
  echo "$(date '+%F %T') driver $$ waits for $WAIT_FOR" | tee -a "$LOGS/driver.log"
  while [ ! -f "$LOGS/$WAIT_FOR.done" ]; do sleep 120; done
fi
for s in "$@"; do run "$s"; done
echo "$(date '+%F %T') factory hard driver finished" | tee -a "$LOGS/driver.log"
