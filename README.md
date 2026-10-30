# 2026-btp-temporal-planning

Long-horizon temporal task planning (B.Tech. Project). **T-iFLAX** selects the objects a temporal planner
(POPF) needs, plans on the smaller problem Π|S, and checks every plan with VAL on the full problem.
Benchmark: Overcooked-AI 1.1.0 kitchens with deadlines.

Pipeline: scenario → HL problem → temporal PDDL → object selection → POPF → VAL (full problem) → simulator.

| Folder | What | README |
| --- | --- | --- |
| `algorithm/` | T-iFLAX: shared core + one file per version (`v1.py` … `v7.py`), evaluation, experiments | [algorithm](algorithm/README.md) |
| `baselines/` | `b1.py` full POPF, `b2.py` FLAX, `b3.py` iFLAX (original code in `vendor/`) | [baselines](baselines/README.md) |
| `pddl_generator/` | HL problem → PDDL 2.1, POPF / VAL wrappers (benchmark-independent) | [pddl_generator](pddl_generator/README.md) |
| `overcooked_ai/` | Benchmark: domain, scenarios, datasets, generated PDDL, simulator (`repo/`, pinned) | [overcooked_ai](overcooked_ai/README.md) |
| `configs/` | tool paths, dataset generators | [configs](configs/README.md) |
| `tools/bin/` | POPF, OPTIC, VAL binaries | [tools](tools/README.md) |
| `results/` | one folder per method: trained models, evaluation rows, logs; `summary/` tables + plots | [results](results/README.md) |
| `tests/` | semantics vs VAL, Π\|S soundness, every version and baseline end to end | — |
| `run.sh` | single entry point for everything below | |

## Setup (once)

```bash
./run.sh setup        # .venv with Python deps, Overcooked-AI 1.1.0 (repo/), this project in editable mode
./run.sh test         # 66 tests, ~4 min
```

`run.sh` unsets `PYTHONPATH` (a system ROS path otherwise leaks in) and fixes `PYTHONHASHSEED=0` (object
order, hence selection, depends on it). If you call `.venv/bin/python -m ...` directly, do the same.

## Run

| Task | Command |
| --- | --- |
| Evaluate one method on the 60 test kitchens, 30 s each | `./run.sh V4.2` &nbsp; (`B1 B2 B3 V1 V2 V2.1 V3 V3U V3.1 V4 V4.1 V4.2 V5`) |
| Other budget / several methods | `./run.sh V4.2 60` · `./run.sh eval B1,V1,V4.2 30` |
| "No time limit" (single 300 s cap, 2 per tier) | `./run.sh eval B1,V3 30 --unlimited 300 --limit 2` |
| Subset of problems | append `--tiers stress_t1 stress_t4` or `--limit 1` |
| One instance, print the plan | `./run.sh solve V4.2 overcooked_ai/generated_pddl/examples/s04_asym_c2_o2_dl 30` |
| Train a learned method | `./run.sh train V1` · `V2` · `V2.1` · `B2` · `B3` (V3 / V3.1 reuse V1's model) |
| V5 / V6 / V7 pipelines | `./run.sh v5 labels --split train` … see [algorithm](algorithm/README.md#v5v7-pipelines) |
| Everything from scratch (train + 30 s + 300 s) | `./run.sh all 0` |
| Summary + plots of a finished run | `./run.sh report stress_30s_seed0` |
| Design experiments (oracle, CP-SAT, …) | `./run.sh exp oracle` |
| Scenarios → PDDL; semantic check | `./run.sh pddl` · `./run.sh check overcooked_ai/generated_pddl/examples` |
| New dataset | `./run.sh data configs/dataset_stress.yaml --count 10 --probe 30` |

All options of an evaluation: `.venv/bin/python -m algorithm.evaluate -h` (`--jobs`, `--max_load`, `--seed`,
`--suffix`, `--plot`, `--sequential_recovery`). Runs are resumable: finished (method, problem) rows are skipped.

## Where to change things

| You want to change | Where |
| --- | --- |
| Planner / VAL binaries, VAL tolerance | `configs/tools.yaml` |
| Datasets (kitchen sizes, cooks, orders, deadline tightness) | `configs/dataset_*.yaml`, then `./run.sh data …` |
| A single test problem | `overcooked_ai/scenarios/examples/*.yaml`, then `./run.sh pddl` |
| Domain (actions, durations, deadline encoding) | `overcooked_ai/domain.yaml` |
| A version's parameters (seeds, caps, recovery, improvement phase) | its `PRESETS` / `*Config` in `algorithm/vN.py`; shared defaults in `MethodConfig` (`algorithm/v1.py`) |
| Training hyper-parameters | command-line flags (`./run.sh train V1 --epochs 6 --budget 15 …`) |
| Which test split / budget | `--split`, `--budgets`, `--unlimited` of `algorithm.evaluate` |
| Add a new version | see [algorithm](algorithm/README.md#adding-a-version) |

## Results (seed 0, 60 stress kitchens, 6 tiers, 44–175 objects)

| Method | 30 s solved | 300 s (12 problems) |
| --- | --- | --- |
| B1 full POPF / B2 FLAX / B3 iFLAX | 2 / 1 / 3 | 1 / 1 / 1 |
| V1 / V2 / V2.1 | 9 / 5 / 5 | 3 / – / – |
| V3 / V3U / V3.1 | 27 / 28 / 27 | 7 / – / 7 |
| V4 / V4.1 / **V4.2 (final)** | 42 / 41 / **48** | – |
| V5 / V6 / V7 (seeds only, one POPF at a time; same-protocol TRPG-bound order: 42–44) | 39 / 43 / 39 | – |
| Oracle selection (certificate's objects) | 50 | – |

Every reported plan is VAL-valid on the full problem (0 invalid). A timeout means "unknown", never
"infeasible". Wall-clock noise: differences of 1–3 problems are not significant. Details:
`results/summary/stress_30s_seed0.md`, per-problem plots next to it.
