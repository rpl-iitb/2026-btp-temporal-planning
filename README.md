# final

The BTP code base: temporal planning algorithms (T-iFLAX), baselines and the two simulators they run on. Self-contained:
nothing here refers to a folder outside `final/`.

| Folder | What | Depends on |
| --- | --- | --- |
| `algorithm/` | T-iFLAX versions `v1.py` … `v14.py` (each file starts with what that version does) + their shared core and a private PDDL toolkit | nothing (reads PDDL instances) |
| `baselines/` | `b1.py` POPF, `b2.py` FLAX, `b3.py` iFLAX (original code in `vendor/`) + a private core | nothing |
| `ablation_algorithm/` | ablations of the final version (once it is chosen) | — |
| `overcooked_kitchen/` | the multi-cook kitchen simulator (own README) | nothing |
| `factory_sim/` | the multi-robot factory simulator, Gym + Gazebo (own README) | nothing |
| `overcooked_test/` | glue + tests + results: algorithm / baselines on `overcooked_kitchen` | the folders above |
| `factory_test/` | glue + tests + results: algorithm / baselines on `factory_sim` | the folders above |
| `plans/` | plans and designs made so far | — |
| `paper/` | the paper | — |
| `tools/` | POPF, OPTIC, VAL binaries (external tools, found via `TIFLAX_TOOLS` / `PATH`) | — |

Only the `*_test` folders connect things: they export a simulator's instances as PDDL instance folders (`data/`), point
`algorithm/` and `baselines/` at them through environment variables (`TIFLAX_DATA`, `TIFLAX_RESULTS`, ...; see
`algorithm/__init__.py`), run methods, check plans with VAL and in the simulator, and keep results.

```bash
./run.sh setup                          # once: .venv with torch (CPU), both simulators, test deps
./run.sh smoke                          # light check: imports, one instance per simulator (~30 s)
./run.sh data                           # build the instances: overcooked_test/data, factory_test/data (~10-15 min)
./run.sh kitchen V4.2 60 --split v2_test --limit 2   # one planner on kitchen instances (budget s; --limit per tier)
./run.sh factory V8S 60 --split test --limit 2       # ... on FactorySim instances
./run.sh kitchen-all v14_train          # one stage (here: train V14 on the multi-cook kitchens)
./run.sh all                            # every experiment on both simulators (days; run inside tmux)
./run.sh kitchen-all | factory-all      # main experiments of one simulator (every method + baselines)
./run.sh kitchen-baselines | factory-baselines   # only the baselines
./run.sh kitchen-ablations | factory-ablations   # only the ablations (after the main experiments)
./run.sh factory-hard [main|ablations|seeds]     # the hard FactorySim suite (lane traffic, deadlines, plants up to XXXL)
./run.sh demos [out_dir]                     # demo GIFs: trained V14 (kitchen) / V15_hard (factory) vs the dispatcher (needs the models)
./run.sh test                           # pytest of both *_test folders (runs POPF: needs free CPU / memory)
./run.sh py -m <module> ...             # the venv's python with the project environment
```

## Running the experiments

Run everything from this folder, inside `tmux` (or `screen`) on a remote machine: the long runs survive a disconnect.
`PROCS` (default 6, about half the cores) sets the parallel workers and `MIN_MB` (default 1500) the free memory below
which the memory guard stops the largest running stage (never anything else).

**1. Once: setup, smoke test, data**

```bash
./run.sh setup
./run.sh smoke        # imports + one instance per simulator; 3 / 3 must pass
./run.sh data         # 440 kitchen instances (overcooked_test/data) + FactorySim train / val / test (factory_test/data)
```
Nothing below works without the data (the drivers stop with a message).

**2. One method on one simulator**

There are two kinds of methods.

*Planners* get a time budget and return one plan per instance (single-cook kitchens `v2_test`, FactorySim `test`):

```bash
./run.sh kitchen B1,V4.2 60 --split v2_test --limit 2    # 2 kitchens per tier, 60 s each
./run.sh factory B1,V8S 60 --split test
```

| Planner | What it is | Needs training first (`./run.sh kitchen-all <stages>`) |
| --- | --- | --- |
| B1 | POPF on the full problem | no |
| B2 / B3 | FLAX / iFLAX (original code) | `b2_train` / `b3_train` |
| V1, V3, V3.1 | T-iFLAX with the learned object scorer | `v1_train` |
| V2 / V2.1 | gated scorer | `v2_train` / `v21_train` |
| V3U, V4, V4.1, V4.2 | symbolic object selection (uniform scores) | no |
| V5 | set scorer | `v5_labels v5_train` |
| V8 / V8.2 | learned station ranking + constructive scheduler | `teacher_build teacher_train` (+ `teacher_improve teacher2_train`) |
| V8S | the scheduler without the GNN | no |
| V9 | learned stations + RL policy | `teacher_build teacher_train v9_train` |
| V10 | domain-independent temporal planner | no |
| V11 | relational GNN policy over PDDL moves | `v11_train` |

On FactorySim the planners that need no training run (B1, V3U, V4.x, V8S, V10); the learned ones are not trained on
factory data by the drivers yet, and V8 / V9 also need the factory teacher (see Known issues).

*Closed-loop policies* (V12, V13, V14) act in the multi-agent simulator under random durations: train, then evaluate
against the rule-based dispatcher.

```bash
./run.sh kitchen-all v14_train                                    # train V14 on the multi-cook kitchens (hours)
./run.sh py -m overcooked_test.v3_eval --split v3_test --methods dispatcher V14 --noise 1 2 --seeds 1
./run.sh factory-all v14_train v14_eval                           # the same on FactorySim
```

**3. All experiments**

```bash
./run.sh all          # data if missing; kitchen main + ablations; factory main + baselines + ablations
```
or one simulator at a time with `./run.sh kitchen-all` / `./run.sh factory-all`. What `kitchen-all` (group `main`) does, in order:

| Stages | What |
| --- | --- |
| `v13_train v14_train` | train V13 (MLP) and V14 (event-graph GNN): DAgger with the dispatcher, then PPO, on the multi-cook kitchens |
| `v3_eval_fast` | 60 multi-cook test kitchens, noise x1 / x2 / x3, 3 seeds: dispatcher, fixed plan open-loop, V13, V14 |
| `v12_train v3_eval_slow` | train V12 (relational GNN); add re-planning and V12 to the same evaluation |
| `teacher_* v9_train v11_train b2_train v5_* v6_all v7_all v1_train v2_train v21_train b3_train` | train every single-cook planner |
| `v2_eval` | all 19 planners on the 60 single-cook test kitchens, 60 s each, run last on an otherwise idle machine |

`factory-all` (group `main`): train V13 / V14 / V12, evaluate them (with the dispatcher) at noise x1-x3, then the
planners B1, V4.2, V8S, V10. Rough durations on a 12-core laptop: data 15 min, kitchen main 1.5-2 days (the
multi-cook results after 6-10 h), factory main several hours, each ablation group 8-12 h.

**4. Baselines only**

```bash
./run.sh kitchen-baselines   # B2 / B3 training; B1, B2 (FLAX), B3 (iFLAX), V10 at 60 s on single-cook kitchens;
                             # dispatcher, fixed plan open-loop, re-planning on multi-cook kitchens (noise x1-x3)
./run.sh factory-baselines   # the robot dispatcher alone (noise x1-x3); B1, V4.2, V8S, V10 at 60 s
```
(`kitchen-all` / `factory-all` already include all of these.)

**5. Ablations** (run after the main experiments: they are compared with the trained V13 / V14)

```bash
./run.sh kitchen-ablations
./run.sh factory-ablations
```

| Ablation | Model | Shows |
| --- | --- | --- |
| no RL (imitation only) | `V14_il`, `V13_il` | what PPO adds over DAgger |
| REINFORCE instead of PPO | `V14_reinforce` | the RL algorithm |
| no max-plus propagation (times only as input features) | `V14_nomaxplus` | the event-graph time structure |
| no hand-made option features (event graph only) | `V14_nofeats` | domain-independence |
| V13 vs V14 | `V13` | the GNN itself (same options, training and execution) |
| RL vs hand-written rules on the same stations | `abl_v9_compare` | single-cook RL policy (V9) |
| uniform scores (V3U) / no GNN (V8S) | in `v2_eval` | the learned object scorer |

On FactorySim the models are named `V14_factory_il`, ... .

**6. Watching and resuming**

```bash
tail -f overcooked_test/results/logs/driver.log            # stage start / done / FAIL (also printed in the terminal)
tail -f overcooked_test/results/logs/<stage>.log           # one stage's output (factory: factory_test/results/logs/)
DRY_RUN=1 ./run.sh kitchen-ablations                       # print the commands a group would run, run nothing
```
A finished stage writes `<stage>.done`: running the same command again skips it, so a stopped run resumes. Delete the
`.done` file to redo a stage. A failed stage is logged and the next one runs.

**7. Where the results are**

| What | File |
| --- | --- |
| multi-cook comparison (V14, V13, V12, dispatcher, re-planning, open loop) | `overcooked_test/results/v13/eval_v3_test.md` |
| multi-cook ablations | `overcooked_test/results/v13/eval_v3_test_ablations.md` |
| single-cook planners (solved, first plan, makespan) | `overcooked_test/results/summary/` |
| RL vs rules (single cook) | `overcooked_test/results/teacher/overcooked_kitchen/v9_compare_v2_test.json` |
| FactorySim policies and ablations | `factory_test/results/v13/`, `factory_test/results/v14/`, `factory_test/results/v12/` |
| FactorySim planners | `factory_test/results/summary/` |
| trained models and their training logs | `overcooked_test/results/models/`, `factory_test/results/models/` |

**8. The hard FactorySim suite** (2026-11-21; the suite above saturates: every policy and the dispatcher reach 100 %)

```bash
./run.sh factory-hard                 # data_hard, V13 / V14 / V12 training, closed_eval, plan_eval (about a day)
./run.sh factory-hard ablations       # V14 without RL / with REINFORCE / without max-plus / without option features, V13 without RL
./run.sh factory-hard seeds           # training seeds 1, 2 of V14, V14 without option features and V13
PROCS=8 ./run.sh factory-hard data_hard   # one stage; DRY_RUN=1 prints the commands
```
What makes it hard (FactorySim `SUITE_HARD`, details in factory_sim/README.md section 7):
- **Lane traffic**: corridor junctions hold one robot; robots queue, meet head-on, back into bays, and can deadlock.
  In the PDDL (domain `factorysim-lanes`) a move is one floor edge and needs the next junction free (`node-free`).
- **Hard deadlines** in the PDDL (an order cannot be delivered after its due date): success = every order on time.
  Due dates come from the workload (loose: all at 1.2 x the reference makespan; tight: staggered; rush orders).
- **Families** `hard_base`, `hard_bottleneck`, `hard_blocking`, `hard_flexible`, `hard_online`, `hard_rush`, `hard_full`.
- **Sizes**: train on the lab S, M, L (2-4 robots, 4-14 orders); test M, L and plants of 2-4 production lines joined
  by cross aisles (intersections): XL 6 robots / 24 orders, XXL 8 / 36, XXXL 12 / 56 (127k ground actions).
- The policies' option generator (factory_test/factory_dispatcher.py) has two layers: the robot (rules or a learned
  policy) chooses its trip at a dock or bay; a rule-based traffic layer drives it along the corridor (wait, back off,
  priorities). New option features: route congestion, detour, blocked, slack to the due date.

| What | File |
| --- | --- |
| closed loop: dispatcher, open loop, re-planning (M, L), V13, V14, V12 (M-XL), by size and by family; ablations and seeds on M-XL | `factory_test/results_hard/v13/eval_test.md` |
| ablations / training seeds | `factory_test/results_hard/v13/eval_test_ablations.md`, `eval_test_seeds.md` |
| planners B1, V4.2, V8S, V10 | `factory_test/results_hard/summary/` |
| instances (with best known makespan, due reference, dispatcher on time at nominal durations) | `factory_test/data_hard/<split>/dataset.json` |

The earlier suite (`data/`, `results/`) is untouched; its V13 / V14 models do not load with the new option features.

State (2026-11-20): code complete, nothing trained or evaluated yet. On both simulators: B1–B3, V1–V7 (+ sub-versions),
V10, V11 through `evaluate.py`; V8 / V9 with their scheduler / environment plug-ins; V12 / V13 / V14 with their domain
modules (`*_v13.py`); FLAX rules for both domains. Checked by `./run.sh smoke` (no planner runs) and the simulators' test
suites. Learned versions (V1, V2, V5–V7, V8 / V9's scorer and policy, V11–V14, B2, B3) need training on the new data;
`overcooked_test/run_kitchen_all.sh` and `factory_test/run_factory_all.sh` run every stage unattended (see above).
Known issues: `factory_test/teacher.py` lacks the label / training functions of `overcooked_test/teacher.py` (build, train, losses fail);
Results from before `final/` were on earlier benchmarks; they stay in `../temporal_planning/results` for reference.
