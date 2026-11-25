#!/usr/bin/env bash
# All kitchen experiments, unattended: training of every learned method, then evaluation.
#
#   overcooked_test/run_kitchen_all.sh [<group | stage> ...]   (from final/; default: main)
#   DRY_RUN=1 ... prints each stage's command without running it
#   ./run.sh kitchen-all [<stage> ...]                     (the same)
#
# Groups (a group name expands to its stages; default: main):
#   main       the multi-cook kitchens first (V13, V14, V12 and their closed-loop evaluation, with the dispatcher,
#              open-loop and re-planning baselines), then the single-cook planning track (teacher / V9 / V11 / B2 /
#              V5-V7 / V1 / V2 / V2.1 / B3 training), and last the timed evaluation of every planner (v2_eval, with
#              the baselines B1-B3, V10 and the ablations V3U, V8S), measured with nothing else running
#   baselines  only the baselines (B2 / B3 training, then base_v2_eval and base_v3_eval)
#   ablations  the policy ablations (V14 without RL / with REINFORCE / without max-plus / without option features,
#              V13 without RL; abl_eval) and the single-cook RL-vs-rules comparison (abl_v9_compare)
# Needs ./run.sh data.
#
# Each stage logs to overcooked_test/results/logs/<stage>.log and writes <stage>.done when it succeeds; a stage
# already done is skipped, a failed stage is logged and the next one runs. tools/memguard.py watches memory and
# kills the process tree of this driver's largest stage (never anything else) if available memory gets low; systemd-inhibit keeps
# the machine from sleeping while the run lasts.
set -uo pipefail
cd "$(dirname "$0")/.."
unset PYTHONPATH
export PYTHONHASHSEED=0
PY=.venv/bin/python
LOGS=overcooked_test/results/logs
mkdir -p "$LOGS"
DOM=overcooked_test.kitchen_v13
PROCS=${PROCS:-6}
# stage groups (a group name on the command line expands to its stages)
MAIN=(v13_train v14_train v3_eval_fast v12_train v3_eval_slow
      teacher_build teacher_train teacher_improve teacher2_train v9_train v11_train
      b2_train v5_labels v5_train v6_all v7_all v1_train v2_train v21_train b3_train
      v2_eval)
BASELINES=(b2_train b3_train base_v2_eval base_v3_eval)
ABLATIONS=(abl_v14_il_train abl_v14_reinforce_train abl_v14_nomaxplus_train abl_v14_nofeats_train abl_v13_il_train
           abl_eval teacher_build teacher_train v9_train abl_v9_compare)
[ $# -eq 0 ] && set -- main
args=()
for a in "$@"; do
  case "$a" in
    main) args+=("${MAIN[@]}") ;;
    baselines) args+=("${BASELINES[@]}") ;;
    ablations) args+=("${ABLATIONS[@]}") ;;
    *) args+=("$a") ;;
  esac
done
set -- "${args[@]}"
if [ "${DRY_RUN:-0}" != 1 ] && [ ! -f overcooked_test/data/v3_train/dataset.json ]; then
  echo "no kitchen data: run ./run.sh data first" >&2
  exit 1
fi
export TIFLAX_RESULTS=$PWD/overcooked_test/results TIFLAX_DATA=$PWD/overcooked_test/data
# the harness's paths (tools, FLAX rules, benchmark registries) for modules run as -m algorithm.vN
eval "$($PY -c "import overcooked_test, os, shlex; print('\n'.join('export {}={}'.format(k, shlex.quote(v)) for k, v in os.environ.items() if k.startswith(('TIFLAX', 'OVERCOOKED'))))")"

$PY tools/memguard.py --root $$ --min_mb "${MIN_MB:-1500}" --log "$LOGS/memguard.log" >/dev/null 2>&1 &
GUARD=$!
systemd-inhibit --what=sleep:idle --who=btp-experiments --why="kitchen experiments" --mode=block sleep infinity 2>/dev/null &
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

run() {
  case "$1" in
    # ---- multi-cook kitchens, random durations, closed loop (v3_*)
    v13_train) stage v13_train $PY -m algorithm.v13 --domain $DOM train --procs "$PROCS" ;;
    v14_train) stage v14_train $PY -m algorithm.v14 --domain $DOM train --procs "$PROCS" --threads "$PROCS" ;;
    v12_train) stage v12_train $PY -m algorithm.v12 --domain $DOM train --procs 5 ;;
    v3_eval_fast) stage v3_eval_fast $PY -m overcooked_test.v3_eval --split v3_test \
                    --methods dispatcher open-loop V13 V14 --noise 1 2 3 --seeds 3 --procs "$PROCS" ;;
    v3_eval_slow) stage v3_eval_slow $PY -m overcooked_test.v3_eval --split v3_test \
                    --methods dispatcher open-loop V13 V14 replan-2 V12 --noise 1 2 3 --seeds 3 --procs "$PROCS" ;;
    # ---- one-cook kitchens, deterministic planning with a time budget (v2_*); V5-V7 use the split aliases
    #      train/val/train_large/val_large -> v2_train/v2_val and stress -> v2_test (symlinks in data/)
    teacher_build)   stage teacher_build $PY -m overcooked_test.teacher build --splits v2_train v2_val --jobs 2 ;;
    teacher_train)   stage teacher_train $PY -m overcooked_test.teacher train ;;
    teacher_improve) stage teacher_improve $PY -m overcooked_test.teacher improve --splits v2_train v2_val --jobs 3 ;;
    teacher2_train)  stage teacher2_train $PY -m overcooked_test.teacher train --name TEACHER2 ;;
    v9_train)        stage v9_train $PY -m overcooked_test.kitchen_v9 train ;;
    v11_train)       stage v11_train $PY -m algorithm.v11 train --bench overcooked_kitchen ;;
    v1_train)        stage v1_train $PY -m algorithm.v1 train --train v2_train --val v2_val --seed 0 ;;
    v2_train)        stage v2_train $PY -m algorithm.v2 train --method V2 --train v2_train --val v2_val --seed 0 ;;
    v21_train)       stage v21_train $PY -m algorithm.v2 train --method V2.1 --train v2_train --val v2_val --seed 0 ;;
    b2_train)        stage b2_train $PY -c "import overcooked_test, sys, runpy; sys.argv=['b2','train','--train','v2_train','--seed','0']; runpy.run_module('baselines.b2', run_name='__main__')" ;;
    b3_train)        stage b3_train $PY -c "import overcooked_test, sys, runpy; sys.argv=['b3','train','--train','v2_train','--seed','0']; runpy.run_module('baselines.b3', run_name='__main__')" ;;
    v5_labels)       stage v5_labels bash -c "$PY -m algorithm.v5 labels --split train && $PY -m algorithm.v5 labels --split val" ;;
    v5_train)        stage v5_train $PY -m algorithm.v5 train ;;
    v6_all)          stage v6_all bash -c "$PY -m algorithm.v6 labels --split train_large --parallel 2 && $PY -m algorithm.v6 labels --split val_large --parallel 2 && $PY -m algorithm.v6 train && $PY -m algorithm.v6 testA --split stress --parallel 2" ;;
    v7_all)          stage v7_all bash -c "$PY -m algorithm.v7 train && $PY -m algorithm.v7 test --split stress --parallel 2" ;;
    v2_eval)         stage v2_eval $PY -m overcooked_test.evaluate --split v2_test --budgets 60 --jobs 2 --seed 0 \
                       --methods B1,B2,B3,V1,V2,V2.1,V3,V3U,V3.1,V4,V4.1,V4.2,V5,V8,V8S,V8.2,V9,V10,V11 ;;
    # ---- baselines alone: classical planner, FLAX, iFLAX, symbolic V10 (single cook, 60 s); dispatcher, fixed plan
    #      open-loop and re-planning (multi-cook, noise x1-x3). MAIN's v2_eval / v3_eval_slow include all of these.
    base_v2_eval)    stage base_v2_eval $PY -m overcooked_test.evaluate --split v2_test --budgets 60 --jobs 2 --seed 0 --methods B1,B2,B3,V10 ;;
    base_v3_eval)    stage base_v3_eval $PY -m overcooked_test.v3_eval --split v3_test --methods dispatcher open-loop replan-2 \
                       --noise 1 2 3 --seeds 3 --procs "$PROCS" ;;
    # ---- ablations of the multi-cook policy (models V14_il, ... next to V14_v3): imitation only (no RL), REINFORCE
    #      instead of PPO, no max-plus propagation (times only as features), no hand-made option features; V13
    #      imitation only. abl_eval compares them with V13 and V14 (eval_v3_test_ablations.md). abl_v9_compare: the
    #      single-cook RL policy against the hand-written rules on the same stations. Planning ablations (V3U: uniform
    #      scores, V8S: no GNN) are in MAIN's v2_eval.
    abl_v14_il_train)        stage abl_v14_il_train $PY -m algorithm.v14 --domain $DOM train --name V14_il --rl_epochs 0 --procs "$PROCS" --threads "$PROCS" ;;
    abl_v14_reinforce_train) stage abl_v14_reinforce_train $PY -m algorithm.v14 --domain $DOM train --name V14_reinforce --rl reinforce --procs "$PROCS" --threads "$PROCS" ;;
    abl_v14_nomaxplus_train) stage abl_v14_nomaxplus_train $PY -m algorithm.v14 --domain $DOM train --name V14_nomaxplus --no_maxplus --procs "$PROCS" --threads "$PROCS" ;;
    abl_v14_nofeats_train)   stage abl_v14_nofeats_train $PY -m algorithm.v14 --domain $DOM train --name V14_nofeats --no_opt_feats --procs "$PROCS" --threads "$PROCS" ;;
    abl_v13_il_train)        stage abl_v13_il_train $PY -m algorithm.v13 --domain $DOM train --name V13_il --rl_epochs 0 --procs "$PROCS" ;;
    abl_eval)        stage abl_eval $PY -m overcooked_test.v3_eval --split v3_test --tag _ablations --noise 1 2 3 --seeds 3 --procs "$PROCS" \
                       --methods V13 V13_il V14 V14_il V14_reinforce V14_nomaxplus V14_nofeats ;;
    abl_v9_compare)  stage abl_v9_compare $PY -m overcooked_test.kitchen_v9 compare --split v2_test ;;
    # ---- wait_<stage>: block until another driver has finished <stage> (done or failed)
    wait_*) local w=${1#wait_}
            echo "$(date '+%F %T') waiting for $w" | tee -a "$LOGS/driver.log"
            until [ -f "$LOGS/$w.done" ] || grep -q "FAIL  $w " "$LOGS/driver.log"; do sleep 60; done ;;
    *) echo "$(date '+%F %T') unknown stage $1" | tee -a "$LOGS/driver.log" ;;
  esac
}

echo "$(date '+%F %T') driver pid $$ stages: $*" | tee -a "$LOGS/driver.log"
for s in "$@"; do run "$s"; done
echo "$(date '+%F %T') driver finished" | tee -a "$LOGS/driver.log"
