# results

One folder per method (`b1`–`b3`, `v1`–`v7`), plus `summary/`, `diagnostics/`, `pddl_check/`.

| File | Written by | Contents |
| --- | --- | --- |
| `<method>/<split>_<budget>_seed<k>.jsonl` | `algorithm.evaluate` | one row per (method, problem): solved, makespan, first-plan time, objects kept, every attempt, time ledger, plan text. Sub-versions share their parent's folder (`method_key` = V3, V3U, V3.1 …) |
| `<method>/<id>_seed<k>.{pt,json,log}` | `./run.sh train …`, `v5`/`v6`/`v7 train` | trained model, its metadata (vocabulary, temperatures, history), training log |
| `b2/b2_seed0_*`, `b3/b3_seed0_*` | `baselines.b2` / `b3` train | FLAX / iFLAX checkpoints (`_train.pt`), dataset / history pickles, `_meta.json` |
| `v5/`, `v6/`, `v7/` other files | V5–V7 pipelines | POPF labels (`sets_*`, `labels_*`, `v5a_labels_*`), seed-test rows (`sequential_*`, `test*_stress.jsonl`, `v5a_seedtest_*`), logs |
| `summary/<tag>.{md,json}` | `evaluate` / `./run.sh report <tag>` | coverage, times, makespan, gap to best known, paired comparisons |
| `summary/<tag>_*.png`, `_per_problem.csv` | `evaluate --plot` / `report` | per-problem makespan ÷ best known, coverage over time |
| `diagnostics/` | `./run.sh exp …` | outputs of the design experiments (see algorithm/README.md) |
| `pddl_check/<run>/` | `./run.sh check` | POPF → VAL → simulator check of the generated PDDL |

Reported runs: tags `stress_30s_seed0` (12 methods × 60 problems) and `stress_unlimited300s_seed0` (6 methods,
2 problems per tier). Both were run with `PYTHONHASHSEED=0`; V3U, V4, V4.1, V4.2 with `--max_load 10`.
New runs with the same tag resume (skip finished rows); use `--suffix _name` for a separate run.
