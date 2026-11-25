#!/usr/bin/env bash
# All FactorySim experiments, unattended: training of the closed-loop policies, then evaluation.
#
#   factory_test/run_factory_all.sh [<group | stage> ...]   (from final/; default: main)
#   DRY_RUN=1 ... prints each stage's command without running it
#   ./run.sh factory-all [<stage> ...]                   (the same)
#
# Groups (a group name expands to its stages; default: main):
#   main       v13_train v14_train v12_train (DAgger with the robot dispatcher, then RL; models in results/models/),
#              v13_eval v14_eval v12_eval (test split, noise x1 / x2 / x3, 5 seeds; each also scores the dispatcher),
#              plan_eval (planners on the test split, 60 s budget: B1 (POPF), V4.2, V8S, V10)
#   baselines  dispatcher_eval (the rule-based robot dispatcher alone, noise x1-x3), plan_eval
#   ablations  V14 without RL / with REINFORCE / without max-plus / without option features, V13 without RL:
#              abl_*_train, then abl_*_eval (results/v14/<name>_test.json, results/v13/V13_factory_il_test.json)
# V8 / V9 / B2 / B3 are not run: they need the factory teacher, which is incomplete (factory_test/teacher.py).
# Needs the data: ./run.sh data. Each stage logs to factory_test/results/logs/<stage>.log and writes <stage>.done
# when it succeeds; a stage already done is skipped (delete the .done file to redo it), a failed stage is logged and
# the next one runs. tools/memguard.py stops the largest stage of this driver if available memory gets low.
set -uo pipefail
cd "$(dirname "$0")/.."
unset PYTHONPATH
export PYTHONHASHSEED=0
PY=.venv/bin/python
LOGS=factory_test/results/logs
mkdir -p "$LOGS"
DOM=factory_test.factory_v13
PROCS=${PROCS:-6}
export TIFLAX_RESULTS=$PWD/factory_test/results TIFLAX_DATA=$PWD/factory_test/data
# the harness's paths (tools, FLAX rules, benchmark registries) for modules run as -m algorithm.vN
eval "$($PY -c "import factory_test, os, shlex; print('\n'.join('export {}={}'.format(k, shlex.quote(v)) for k, v in os.environ.items() if k.startswith('TIFLAX')))")"
MAIN=(v13_train v14_train v12_train v13_eval v14_eval v12_eval plan_eval)
BASELINES=(dispatcher_eval plan_eval)
ABL=(V14_factory_il V14_factory_reinforce V14_factory_nomaxplus V14_factory_nofeats V13_factory_il)
ABLATIONS=($(for n in "${ABL[@]}"; do echo "abl_${n}_train"; done) $(for n in "${ABL[@]}"; do echo "abl_${n}_eval"; done))
ALL=("${MAIN[@]}" dispatcher_eval "${ABLATIONS[@]}")
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

if [ "${DRY_RUN:-0}" != 1 ] && [ ! -f factory_test/data/train/dataset.json ]; then
  echo "no factory training data: run ./run.sh data first" >&2
  exit 1
fi

$PY tools/memguard.py --root $$ --min_mb "${MIN_MB:-1500}" --log "$LOGS/memguard.log" >/dev/null 2>&1 &
GUARD=$!
systemd-inhibit --what=sleep:idle --who=btp-experiments --why="factory experiments" --mode=block sleep infinity 2>/dev/null &
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

EVAL=(--split test --seeds 5 --noise 1 2 3 --procs "$PROCS")
run() {
  case "$1" in
    v13_train) stage v13_train $PY -m algorithm.v13 --domain $DOM train --name V13_factory --procs "$PROCS" ;;
    v14_train) stage v14_train $PY -m algorithm.v14 --domain $DOM train --name V14_factory --procs "$PROCS" --threads "$PROCS" ;;
    v12_train) stage v12_train $PY -m algorithm.v12 --domain $DOM train --name V12_factory --procs "$PROCS" ;;
    v13_eval)  stage v13_eval $PY -m algorithm.v13 --domain $DOM eval --name V13_factory "${EVAL[@]}" ;;
    v14_eval)  stage v14_eval $PY -m algorithm.v14 --domain $DOM eval --name V14_factory "${EVAL[@]}" ;;
    v12_eval)  stage v12_eval $PY -m algorithm.v12 --domain $DOM eval --name V12_factory "${EVAL[@]}" ;;
    dispatcher_eval) stage dispatcher_eval $PY -m algorithm.v13 --domain $DOM eval --name dispatcher "${EVAL[@]}" ;;
    abl_*_train|abl_*_eval)
      local n=${1#abl_}; local kind=${n##*_}; n=${n%_*}
      local mod=v14; [[ $n == V13* ]] && mod=v13
      local flags=(); case "$n" in
        *_il) flags=(--rl_epochs 0) ;; *_reinforce) flags=(--rl reinforce) ;;
        *_nomaxplus) flags=(--no_maxplus) ;; *_nofeats) flags=(--no_opt_feats) ;; esac
      local threads=(); [ $mod = v14 ] && threads=(--threads "$PROCS")
      if [ "$kind" = train ]; then
        stage "$1" $PY -m algorithm.$mod --domain $DOM train --name "$n" "${flags[@]}" --procs "$PROCS" "${threads[@]}"
      else
        stage "$1" $PY -m algorithm.$mod --domain $DOM eval --name "$n" "${EVAL[@]}"
      fi ;;
    plan_eval) stage plan_eval $PY -m factory_test.evaluate --split test --budgets 60 --jobs 2 --seed 0 --methods B1,V4.2,V8S,V10 ;;
    *) echo "$(date '+%F %T') unknown stage $1 (stages: ${ALL[*]})" | tee -a "$LOGS/driver.log" ;;
  esac
}

echo "$(date '+%F %T') factory driver pid $$ stages: $*" | tee -a "$LOGS/driver.log"
for s in "$@"; do run "$s"; done
echo "$(date '+%F %T') factory driver finished" | tee -a "$LOGS/driver.log"
