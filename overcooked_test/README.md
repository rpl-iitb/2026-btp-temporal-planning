# overcooked_test: algorithm / baselines on overcooked_kitchen

| File | What |
| --- | --- |
| `__init__.py` | paths: points `algorithm/` and `baselines/` at `data/` and `results/`, `final/tools/bin`, the FLAX rules |
| `make_data.py` | exports `overcooked_kitchen/benchmarks/v1` (440 instances: `v2_*` one cook, `v3_*` 3–4 cooks) to `data/<split>/<name>/` (`hl_problem.json`, `problem.pddl`, VAL-checked `certificate.plan`, `scenario.yaml`) + `data/domain.{pddl,yaml}` |
| `evaluate.py` | runs methods (`METHODS`: B1–B3, V1–V5, V8 / V8S / V8.2, V9, V10, V11) on a split → `results/<method>/`, summaries in `results/summary/` |
| `kitchen_scheduler.py` | the one-cook constructive kitchen scheduler; V8's scheduler plug-in (`KitchenScheduler`) |
| `kitchen_v9.py` | V9's macro-step environment over the scheduler, episodes, `train` / `compare` CLI |
| `kitchen_dispatcher.py` | the rule-based multi-cook dispatcher over PDDL states (baseline, expert, option generator) |
| `kitchen_v13.py` | V12 / V13's domain module (options, expert, 40 features, duration noise, data) |
| `kitchen_methods.py` | builders of V8 / V9 / V11 with their kitchen parts and models |
| `teacher.py` | POPF teacher plans → labels → the GNN scorer V8 / V9 use (`build`, `improve` with the scheduler + kitchen replay, `train`, `losses`, `plot`) |
| `v3_eval.py` | closed-loop evaluation on the multi-cook kitchens: dispatcher, open-loop certificate, re-planning, V12, V13, V14 |
| `run_kitchen_all.sh` | unattended driver for every kitchen experiment, in stages (training of each learned method, both evaluations); logs, `.done` markers and the memory guard in `results/logs/` |
| `flax_rules/` | FLAX's complementary / relaxation rules for the kitchen domain (B2, B3) |
| `tests/` | semantics vs VAL on certificates, Pi\|S soundness, re-timing, baselines' adapter, every version on a small kitchen |

```bash
../run.sh data                                                  # once (or python -m overcooked_test.make_data)
../run.sh kitchen B1,V4.2,V8S 30 --split v2_test --limit 2       # evaluate (budget s)
python -m overcooked_test.teacher build --splits v2_train v2_val  # then teacher train -> TEACHER_overcooked_kitchen_seed0
python -m overcooked_test.kitchen_v9 train                       # V9's policy
python -m algorithm.v13 --domain overcooked_test.kitchen_v13 train
python -m algorithm.v14 --domain overcooked_test.kitchen_v13 train
python -m overcooked_test.v3_eval --split v3_test --methods dispatcher open-loop replan-2 V13 V14
```
`data/` also holds split aliases for V5 / V6 / V7, which name their splits the old way: `train`, `train_large` -> `v2_train`, `val`, `val_large` -> `v2_val`, `stress` -> `v2_test` (symlinks).
Models trained here go to `results/models/` (`TEACHER_overcooked_kitchen_seed0`, `TIFLAX9_overcooked_kitchen_seed0`,
`V11_overcooked_kitchen`, `V13_v3`, `V14_v3`, ...).
