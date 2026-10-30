# Benchmark 1: Overcooked-AI v1

The simulator is Overcooked-AI **1.1.0**, upstream tag `1.1.0`, commit `a60e31a`. This is the same code as the PyPI wheel used in `Neuro_paper_implementation/`. The source is in `repo/` (unmodified clone, no git history); the commit is recorded in `builder.py` (`SIM_VERSION`, checked by `check_pinned()`).

```
overcooked_ai/
├── domain.yaml          the planning domain: types, predicates, durative actions, durations
├── scenarios/           problem specs (YAML): examples/ hand-written, <dataset>/ generated
├── generated_pddl/      domain.pddl + <set>/<id>/{problem.pddl, hl_problem.json, manifest.json}
├── make_dataset.py      generates many scenarios (incl. hard ones) from a config in ../configs/
├── builder.py           scenario -> HL problem (objects, facts, travel times, goals, deadlines)
├── layout.py            grid analysis: stations, interaction poses, exact travel times
├── executor.py          runs a plan in the simulator, measures timing vs. the plan
├── benchmark.py         the Benchmark subclass the rest of the pipeline uses
├── scenario.schema.json scenario format
└── repo/                simulator source (pinned clone)
```

## Run

From the project root:

```bash
./run.sh pddl                                            # all scenarios -> PDDL
./run.sh pddl overcooked_ai/scenarios/examples
./run.sh check overcooked_ai/generated_pddl/examples     # POPF -> VAL -> simulator
```

`check` prints one line per instance and writes `results/pddl_check/<run>/`. Columns:

- `ms`: simulated makespan, in steps.
- `maxdev`: largest |simulated end − planned end| over all dispatched actions.
- `detours`: steps a cook walked around another cook.

Measured on the examples (2026-10-10): all solved and VAL-valid, all deadlines met. `maxdev` is 0 everywhere except s02, which has 1 step, caused by 3 detours.

## Many test cases: `make_dataset.py`

```bash
./run.sh data configs/dataset_stress.yaml          # hard, 6 tiers
./run.sh data configs/dataset_stress.yaml --count 30 --probe 60
./run.sh data configs/dataset_random_small.yaml    # small kitchens
./run.sh data configs/dataset_layouts.yaml         # the 1.1.0 layouts
```

- `--count N`: instances per tier.
- `--tiers t3_xl t4_split`: generate only those tiers.
- `--probe S`: also run POPF on every *full* problem for S seconds and record whether it blows up.
- `--no-pddl`: write only the scenario files.

Rerunning a config replaces that dataset. The same config and seed always give the same instances.

### How the hard instances stay feasible

Every generated kitchen has a small **core** near the cooks: pot(s), onion (tomato if a recipe needs it), dish, serving window, and `core_cooks` cooks. Everything else only makes the full problem larger:

- distractor pots, dispensers and windows, mostly in the middle and far side;
- extra cooks;
- serpentine counter walls (detours);
- split rooms (hand-offs over counters);
- `counters: all` (every counter becomes an object).

The core problem is solved with POPF. That plan is also a plan for the full problem, and VAL checks this on the generated PDDL. So every instance ships with `certificate.plan`, even when POPF on the full problem fails.

Deadlines are `ceil(rho × delivery time in the certificate)`, with `rho ∈ [1.0, 1.15]`. They can be met, but comfortably only by using the extra capacity. This is the "classically irrelevant, temporally relevant" case.

`scenarios/<dataset>/dataset.json` records, per instance:

- tier, sizes (full vs. core objects);
- certificate makespan and POPF time;
- whether the certificate is valid on the full problem;
- the full-problem probe result;
- every rejected sample and why.

### Tiers in `configs/dataset_stress.yaml`

| Tier | Kitchen | Cooks | Orders | Distractor stations | Feature |
| --- | --- | --- | --- | --- | --- |
| t1_medium | 11–14 × 6–8 | 2–3 | 2–3 | 5–11 | just above the earlier test split |
| t2_large | 15–19 × 7–9 | 2–4 | 3–4 | 10–17 | 1–2 serpentine walls |
| t3_xl | 20–25 × 9–11 | 3–6 | 4–6 | 14–25 | 2–3 walls |
| t4_split | 13–19 × 7–9 | 2–4 | 2–4 | 6–13 | two rooms, far side via counters |
| t5_counters | 12–16 × 6–8 | 2–3 | 2–4 | 5–9 | every counter is an object |
| t6_many_orders | 12–16 × 7–9 | 2–4 | 6–8 | 6–13 | long horizon, few pots |

**Measured** (2026-10-10, `--count 10 --probe 30`, 60 instances). Every certificate is VAL-valid on its full problem.

| Tier | Objects (full) | Objects (core) | Max init facts | Full-problem POPF, 30 s |
| --- | --- | --- | --- | --- |
| t1_medium | 44–65 | 23–30 | 424 | 2 solved, 8 timeout |
| t2_large | 72–89 | 27–36 | 834 | 10 timeout |
| t3_xl | 96–120 | 30–42 | 1551 | 10 timeout |
| t4_split | 70–98 | 35–61 | 667 | 10 timeout |
| t5_counters | 117–175 | 25–35 | 3977 | 10 timeout |
| t6_many_orders | 66–82 | 35–40 | 637 | 10 timeout |

For comparison:

- The simulator's own 46 layouts are at most 14 × 9, with at most 2 pots and 4 players, and mostly 1 order.
- The earlier T-iFLAX kitchens (`Neuro_paper_implementation/tiflax/kitchen_tasks.py`):
  - train: 40 kitchens, 9–11 × 5–6, 2–6 distractors, 2 orders;
  - test: 20 kitchens, 13–16 × 7–8, 8–14 distractors, 3 orders;
  - always 2 cooks.
- Other prior domains (FJSP, DriverLog, Satellite, Logistics) are not Overcooked.

## Where to change things

| You want | Change |
| --- | --- |
| A different single problem | Write/edit a YAML in `scenarios/examples/` (format below) and rerun `./run.sh pddl` |
| Many problems / harder ones | Edit or copy a config in `configs/dataset_*.yaml` (sizes, distractors, cooks, orders, recipes, cook time, deadline tightness `rho`, counters), then run `make_dataset` |
| Different action semantics or durations (e.g. interactions take 2 steps, new action) | `domain.yaml`. If a new predicate needs initial facts, also `builder.py::encode_state` |
| How deadlines are encoded | `deadline_encoding` in `domain.yaml` (predicate, offset); the general writer is `pddl_generator/writer.py` |
| Which objects exist (e.g. no poses, one pose per station) | `builder.py::encode_state`, and `layout.py` for travel times |
| PDDL text format | `pddl_generator/writer.py` (the writer and its round-trip reader) |
| Planner / VAL paths or tolerance | `configs/tools.yaml` |
| A new simulator (Benchmark 2/3) | New package with its own `domain.yaml` (can `extends:` this one) and a `Benchmark` subclass; register it in `BENCHMARKS` in `pddl_generator/problem.py` |

## Scenario format (`scenario.schema.json`)

```yaml
scenario_id: my_kitchen            # lowercase; becomes the instance id
layout: {name: cramped_room}       # any 1.1.0 layout file, or: grid: ["XXPXX", "O1 2O", ...]
cooks: 2                           # default: players in the layout; extras placed on free floor (seeded)
pots: all                          # all | k (keep the first k) | [[x, y], ...]; dropped pots become counters
counters: shared                   # none | shared (hand-off counters between rooms) | all
sim: {cook_time: 20}               # any OvercookedGridworld recipe parameter
orders:
  - recipe: {onion: 3}
    deadline: 60                   # simulator step by which it must be served (optional)
  - recipe: {onion: 1, tomato: 2}
    count: 2
seed: 0
```

Grid characters:

- `X`: counter
- `P`: pot
- `O` / `T`: onion / tomato dispenser
- `D`: dish dispenser
- `S`: serving window
- `1`–`9`: cooks
- space: floor

Borders must not be floor.

## The model (simulator → PDDL 2.1)

**Time.** One PDDL time unit = one simulator step. An action over `[s, s+d]` uses steps `s … s+d−1`, and its effects are visible in state `s+d`.

| Simulator | Model |
| --- | --- |
| Player | `cook`. `idle` makes each cook a unary resource |
| Floor tile + facing direction from which a station can be reached | `pose`. Every access side of a station is a separate pose |
| Floor tile | `tile`. `free` gives occupancy: `move` reserves the target tile at start and releases the old one |
| Walking / turning | `move` / `turn`. Duration `travel(a, b)` = exact single-agent shortest path in (x, y, orientation) space |
| INTERACT at O/T/D | `take` (1 step, empty hand) |
| INTERACT at pot with an ingredient | `add-onion` / `add-tomato` (1 step, pot idle, capacity via `fits`) |
| Empty-hand INTERACT at a non-empty pot | `start-cooking` (1 step). 1.1.0 never starts a full pot by itself |
| Pot cooking | `cook`: the pot runs alone for `recipe time − 1` steps, so the soup is ready exactly `recipe time` steps after the start interact |
| Dish + ready soup | `pick-soup` (pot becomes empty and idle) |
| Soup at S | `deliver` for a pending order of that recipe, while `open` |
| Counter put-down / pick-up | `put-down`, `pick-up`, `put-down-soup`, `pick-up-soup` |
| Order + deadline | goal `(served o)`. Deadline D → `(at D+0.5 (not (open o)))` |
| Recipe | `recipe-of r #onions #tomatoes`. Only ordered recipes can be cooked |

**Checked against the simulator (2026-10-09):**

- `travel` equals the simulator's step count for all pose pairs in 5 layouts.
- Soup is ready exactly `cook_time` steps after the start interact (cook times 2, 7, 20).
- Full pots do not auto-start; partly filled pots can be started; a cooking pot refuses ingredients.
- VAL rejects plans that break a rule, and names the broken condition.
- Single-cook plans execute in the simulator with 0 steps of deviation.

**Abstracted away:**

- Path interference between cooks. The simulator freezes *all* players on any collision; the executor measures `detour_steps` and `blocked_steps`.
- Per-order identity and deadlines are planning-only.
- Counter staging, unless `counters: all`.
- POPF's ε drift: the deadline literal at D + 0.5 assumes fewer than ~500 chained happenings.
