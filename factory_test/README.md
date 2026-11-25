# factory_test: algorithm / baselines on factory_sim

| File | What |
| --- | --- |
| `__init__.py` | paths: points `algorithm/` and `baselines/` at `data/` and `results/`, `final/tools/bin`, the FLAX rules |
| `make_data.py` | FactorySim instances (its scenario families on the people-free default lab; default base, bottleneck, tight, noisy; S / M) → `data/<split>/<name>/` with FactorySim's PDDL (`domain.pddl`, `problem.pddl`), the same in the algorithm's format (`domain.yaml`, `hl_problem.json`) and `instance.json` |
| `pddl_convert.py` | FactorySim PDDL → domain spec YAML + HL problem |
| `evaluate.py` | runs methods (B1–B3, V1–V5, V8 / V8S, V9, V10, V11) on a split → `results/` |
| `execute.py` | runs a plan in FactorySim (`factory_sim.pddl.PlanPolicy`) → FactorySim metrics |
| `factory_dispatcher.py` | rule-based robot dispatcher over FactorySim's PDDL states (menv): options, expert, features |
| `factory_scheduler.py` | the dispatcher's nominal rollouts as V8's scheduler plug-in |
| `factory_v9.py` | V9's environment (the dispatcher's decisions), `train` CLI |
| `factory_v13.py` | V12 / V13's domain module (dispatcher options, expert, features, duration noise, data) |
| `factory_methods.py` | builders of V8 / V9 / V11 with their factory parts and models |
| `teacher.py` | POPF teacher plans → labels → the GNN scorer V8 / V9 use |
| `flax_rules/` | FLAX's complementary / relaxation rules for the factory domain (B2, B3) |
| `tests/` | conversion grounds; B1's plan is VAL-valid on FactorySim's PDDL (and runs in FactorySim) |

```bash
../run.sh data                                                # once (or python -m factory_test.make_data)
../run.sh factory B1,V4.2,V8S 30 --split test --limit 2        # evaluate (budget s)
python -m factory_test.teacher build --splits train val        # then teacher train -> TEACHER_factorysim_seed0
python -m factory_test.factory_v9 train
python -m algorithm.v13 --domain factory_test.factory_v13 train
```

Notes. The family `exact` is left out by default: its unlimited buffers become 1000 slot objects per station (memory).
A fleet with tray robots (families `tray_fleet`, `mixed_fleet`, which bring human handlers) adds `handover-*` actions;
the converter keeps them. `execute.py` replays a plan with `factory_sim.pddl.PlanPolicy` (FactorySim 2.0): each robot
does its steps in plan order, and across robots the plan's order is kept on shared jobs and stations, so multi-robot
plans run to the end, with nominal and with sampled durations.
