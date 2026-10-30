# configs

| File | Used by | Controls |
| --- | --- | --- |
| `tools.yaml` | `pddl_generator/planners.py` (all planner / VAL calls) | paths of POPF, OPTIC, VAL `Validate` / `Parser`; VAL tolerance; default planner timeout |
| `dataset_stress.yaml` | `./run.sh data` | the 60 test kitchens (6 tiers t1–t6: size, cooks, orders, distractors, walls, counters, deadline tightness `rho`, seed) |
| `dataset_train.yaml`, `dataset_val.yaml` | training of V1, V2, B2, B3, V5 | 40 / 12 smaller kitchens (seeds 101 / 202) |
| `dataset_train_large.yaml`, `dataset_val_large.yaml` | V6, V7 | 48 / 12 test-sized kitchens (stress generator, other seeds) |
| `dataset_random_small.yaml`, `dataset_layouts.yaml` | tests, checks | small random kitchens; the simulator's standard layouts |

Generate: `./run.sh data configs/dataset_stress.yaml [--count N] [--tiers t1_medium] [--probe 30]` → `overcooked_ai/scenarios/<set>/`
and `overcooked_ai/generated_pddl/<set>/`. Same config + seed = same instances; rerunning replaces the set.
