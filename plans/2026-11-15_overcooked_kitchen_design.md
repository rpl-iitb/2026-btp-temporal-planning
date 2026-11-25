# Overcooked kitchen v2 as an open-source package: design

Status: design, waiting for approval. Date: 2026-11-15.
Working name: `overcooked_kitchen` (folder and import name). The public name is still open (section 11).

## 1. Intent

Turn the v2 kitchen (`temporal_planning/overcooked_ai_v2/`) into a complete, self-contained, open-source
benchmark. Today its pieces are spread over three places: the step-based grid simulator (v2, tested only for one
cook), the multi-cook datasets and dispatcher (v3), and the stochastic multi-agent engine (`tiflax/menv.py`).

What the package must deliver:
- **our domain**: the temporal PDDL 2.1 kitchen domain (`overcooked_ai_v2/domain.yaml`, 26 durative actions:
  move, prep, chop, cook, plate, serve, diners eating, scrubbing, washing; burning is encoded as the stove timer's
  window, not as an action);
- **several agents** (cooks) acting in parallel and sharing stations;
- **multi-tasking**: things that run by themselves (pots cooking, soups burning, diners eating) overlap with the
  cooks' work, and a cook juggles several orders;
- **durative actions** as the unit of decision and of planning;
- **the original Overcooked look and feel**: the Overcooked-AI sprite art, grid characters, action names;
- **human play**, as FactorySim has (a person can control any cook);
- the same research uses as FactorySim: planners (PDDL), learning (Gymnasium / PettingZoo), benchmark suite.

Why: it is the second benchmark of the RSS paper next to FactorySim, and it will be released as open source.
Contributing upstream (HumanCompatibleAI/overcooked_ai) is optional, not a goal.

## 2. Goals, non-goals, anti-goals

**Goals**
1. One package, installable on its own (`pip install -e .`), own README, tests, examples, MIT licence.
2. Multi-cook (1–4 cooks) works and is tested on the grid and at task level.
3. One core, two clocks (section 4): the same task-level rules whether durations are sampled or come from the grid.
4. PDDL export of the current state, plan execution, VAL check, in both clocks.
5. Human play in a window, with AI teammates.
6. Rendering with the Overcooked-AI sprites (GIF / MP4 export, headless).
7. A benchmark suite with certified instances and baseline results.

**Non-goals**
- Merging into upstream overcooked_ai.
- Partial observability, communication, zero-shot coordination research (that is OvercookedV2, Oxford, ICLR 2025).
- A web / browser demo (upstream's overcooked-demo); a local window is enough.
- New planning or learning algorithms (those live in `tiflax/` and `new_algo_trials/`); the package ships
  baselines only.

**Anti-goals** (it fails even if it runs)
- The task-level core and the grid disagree about what is legal or what happens (a plan valid in one is invalid in
  the other, with nominal durations).
- A PDDL plan that VAL accepts but that fails, burns a soup or (with hard deadlines) serves late in the simulator
  with nominal durations. (The reverse is allowed: the world forgives more than PDDL.)
- Breaking the existing experiments: `overcooked_ai_v2/`, `overcooked_ai_v3/`, `tiflax/` stay untouched and keep
  working.
- A look that does not read as Overcooked.

## 3. Approaches considered

| | Approach | Trade-off |
| --- | --- | --- |
| 1 | **The grid is the truth.** Durative actions are compiled to walking + interact steps; PDDL must match the grid. | Closest to Overcooked, low-level control; random durations awkward; blocking only emerges on the grid. |
| 2 | **Durative actions are the truth.** Event-based core over the PDDL actions (like `menv`); grid only for drawing. | Simple, fast, random durations easy; not really Overcooked underneath (no step control, paths abstracted). |
| 3 | **Both: one core, two clocks.** The durative-action core is the truth at task level; durations are either sampled or produced by the grid simulator, which really walks and chops. | Keeps the look, planners and RL both work, plans can be checked in both modes; same architecture as FactorySim (sampled vs Gazebo). More to build. |

**Chosen: 3** (by the user, 2026-11-15).

## 4. Architecture

```
            scenario.yaml (grid, orders, plates, durations, noise, cooks)
                               |
                 +---------------------------+
                 |  core: task-level engine   |  state, durative actions (start / end), resources,
                 |  (single source of rules)  |  autonomous processes (cooking, burning, eating), deadlines,
                 +-------------+-------------+  decision points, metrics, event log
             clock:            |
        +----------------------+----------------------+
        |                                             |
   sampled clock                                 grid clock
   durations drawn from                          the grid simulator executes each started action
   nominal x noise (seeded)                      (walk with BFS + collision rules, interact strokes);
   fast; planning / RL                           it reports the end; durations and blocking emerge
        |                                             |
   Python API · Gymnasium · PettingZoo · PDDL bridge · renderer · human play
```

**Forgiving world, strict planner (decided by the user, 2026-11-15).** The simulator is the world and behaves like
Overcooked: a ready soup left too long burns (the pot becomes burnt and must be scrubbed, the soup is lost); an order
can be served after its deadline (served late, lateness counted); a pot lit with contents that match no order cooks a
spoiled soup that must be dumped and the pot scrubbed; chopping, washing or scrubbing can stop part-way and keep its
progress; a human driving a cook can do anything Overcooked allows. The PDDL domain is the planner's stricter model:
no burning (stove-timer window), optional hard deadlines, recipes only. Guarantee: every VAL-valid plan runs in the
world with no burnt soup and, with hard deadlines, no late order. The core therefore holds the world's rules, which
are a superset of the catalogue: the catalogue's 26 actions, plus the world events burn, spoil, dump, partial progress
and late delivery. Task-level decisions offered to policies and to the TASK menu are catalogue actions only (so a
policy's choices are always PDDL-expressible); only DRIVE mode and burn / lateness can create off-catalogue states.

**PDDL export from a messy state** (mapping rules): a burnt or spoiled pot becomes a dirty pot (needs `scrub-pot`);
a partly chopped / washed / scrubbed item gets the remaining strokes as its duration (per-object duration fluent); an
order already past its deadline gets no deadline; an order that can no longer meet a hard deadline is exported
without it and flagged in the export's metadata.

**Core.** The state and rules of the world, executed as events: an agent starts a durative action (its
at-start conditions hold now), the action holds its resources, and its end applies at-end effects. Actions without
an agent (pot cooking, soup burning, diner eating) start and end by themselves. Decisions are made at event times;
each free cook chooses a legal action or waits. This is the FactorySim decision loop (`next_decision`, `legal`,
`act`, `run(policy)`) and the `menv` execution model.

**One action catalogue.** `domain.yaml` (copied into the package) is the single description of the actions: it
generates the PDDL domain file and drives the core's legality and effects. The grid simulator maps each action to
its grid event (`sim.event` in `domain.yaml`, as the v2 executor already does).

**Sampled clock.** Each started action's duration = nominal × lognormal noise per action type (CV per action,
config), as in `menv.Noise`; noise off = nominal (deterministic, VAL-checkable). Walking is a `move` action whose
nominal duration is the BFS distance.

**Grid clock.** Uses the step simulator (ported from `overcooked_ai_v2/kitchen.py`, Overcooked rules: one tile per
step, moving into a non-floor cell only turns, two cooks wanting the same tile or swapping both stay). When the core
starts an action for a cook, the grid layer plans the cook's path to the interaction pose and sends interacts until
the grid event happens; then it reports the action's end to the core. Contention between cooks (blocked corridors,
waiting for a board) therefore lengthens actions on the grid. Optional grid noise: random stroke counts
(chop / wash / scrub) and cook / eat times drawn from the same noise model.

**Low-level mode.** The grid simulator can also be stepped directly with Overcooked actions per cook
(`north, south, east, west, stay, interact`), as upstream's `OvercookedEnv`. The core then follows the grid: it
recognises which durative action each completed interact belongs to (the mapping already exists in the v2
executor), so metrics, orders, burning and logs stay the core's. Human play in DRIVE mode uses this.

## 5. Interfaces

| Interface | Decision unit | Notes |
| --- | --- | --- |
| Python API | durative action per cook | `Sim(scenario, clock="sampled" \| "grid", seed)`, `legal(cook)`, `act`, `next_decision`, `run(policy)`, `metrics()`. |
| Gymnasium `OvercookedKitchen-v0` | durative action, one cook at a time | masked `Discrete`; `info["action_mask"]`, `info["agent"]`; flat vector observation and a graph view. |
| PettingZoo AEC | durative action per cook | shared team reward. |
| Gymnasium low-level `OvercookedKitchenGrid-v0` | joint Overcooked action per step | grid clock only. |
| PDDL bridge | — | `export(sim)`: current state as a problem (facts now, running actions' remaining effects as timed literals, deadlines shifted to now; as `tiflax/snapshot.py`); `PlanPolicy(plan)` executes a plan; VAL check with nominal durations. |

**Reward and metrics.** Reward: −elapsed time per step, −lateness, −large penalty on a burnt soup or a stuck run.
Metrics: success (all orders served), makespan, on-time rate, total lateness, burnt soups, cook utilisation,
idle and blocked time per cook, number of decisions, wall time.

## 6. Human play (as FactorySim)

`overcooked-kitchen play --scenario <file> --cooks 2` opens a window (pygame, optional dependency).

Modes per cook, switchable at any time, as FactorySim's teleop:
- **AUTO**: a policy controls the cook (default: the dispatcher).
- **TASK**: you are its policy: choose one of its legal durative actions from a menu; the cook walks and works by
  itself.
- **DRIVE**: you move it with the keys and press interact, Overcooked style.

Keys: `1`–`4` select cook; `z` AUTO, `t` TASK, `v` DRIVE; DRIVE: arrow keys + space (selected cook), and a second
person can use W/A/S/D + E for another cook; TASK: `[` / `]` choose, Enter do it; `p` pause, `+`/`−` speed,
`r` restart, `q` quit. The grid clock runs in real time at a configurable step rate when a human drives; otherwise
as fast as possible. Episodes played by humans can be saved (trajectory + event log) for imitation data.

## 7. Look (as the Overcooked-AI creators)

- Use the upstream sprite sheets unchanged (MIT, Center for Human-Compatible AI; attribution in README and
  `NOTICE`): chefs (four hat colours, held items, facing), counter, floor, pot, onion / tomato dispensers, dish
  rack, serving window, onions, tomatoes, dishes, soups, pot-explosion (for a burnt soup). They are in
  `overcooked_ai_v1/upstream/src/overcooked_ai_py/data/graphics/` (vendored 1.1.0).
- New sprites in the same pixel-art style and palette for what upstream lacks: carrot and carrot crate, cutting
  board (with a vegetable and chop progress), chopped vegetables, sink (with a plate and wash progress), dirty-plate
  drop-off, dirty plate, carrot soups, dirty / scrubbing pot, plate counter on the rack.
- HUD as upstream (orders, time) plus progress bars (cooking, burn countdown, eating) and an optional Gantt strip of
  each cook's and pot's activities (the v2 renderer's timeline).
- Same grid characters as v2 (`X B P O T C D R W S`, digits for cooks), so upstream-style layouts read naturally.

## 8. Content and baselines

- **Scenarios**: the v2 examples (k01–k04) and the generators of v2 (1 cook) and v3 (3–4 cooks, more orders,
  deadlines), ported into the package (`generate` command). Tiers by cooks and orders; train / val / test, test
  larger than training.
- **Certificates per instance**: the best of the dispatcher and POPF on the core problem, VAL-valid on the full
  problem, served by the simulator with no burnt soup (as v2/v3 do). Deadlines on service tiers from the
  certificate (as v3).
- **Baselines shipped**: random, the rule dispatcher (from `overcooked_ai_v3/dispatcher.py`), POPF / OPTIC through
  the PDDL bridge (open loop), POPF with replanning on deviation.
- **Benchmark command** (`bench`) writes JSONL and prints a table per tier, as FactorySim.

## 9. Package layout

```
overcooked_kitchen/
  pyproject.toml  README.md  LICENSE  NOTICE  docs/  examples/  tests/
  overcooked_kitchen/
    domain.yaml        the action catalogue (copied from overcooked_ai_v2, then maintained here)
    layout.py          grid analysis, poses, BFS (from overcooked_ai_v1/layout.py; no import of v1)
    core.py            task-level engine (state, actions, events, decisions, metrics)
    clocks.py          sampled clock, grid clock (grid executor)
    grid.py            step simulator (from overcooked_ai_v2/kitchen.py)
    pddl.py            domain / problem writer, state export, plan execution, VAL wrapper
    env.py             Gymnasium (task level, low level), PettingZoo
    baselines.py       random, dispatcher, POPF open loop, POPF replanning
    scenarios.py       scenario format, generators, certificates, splits, suite
    render.py          sprites, frames, GIF / MP4, Gantt
    play.py            human play window
    cli.py             overcooked-kitchen run | play | generate | bench | pddl | solve
    assets/            upstream sprites + new sprites
```

It lives next to `factory_sim/` in `temporal_planning/`, with its own `.venv`, like FactorySim.

## 10. Constraints

- Python ≥ 3.9. Core dependencies: numpy, pyyaml, gymnasium, pillow. Optional: pygame (play), pettingzoo, pytest.
- POPF, OPTIC and VAL are external binaries (not shipped; POPF is GPL). The package finds them on `PATH` or via an
  environment variable; the PDDL tests skip if they are missing. Locally: `temporal_planning/tools/bin/`.
- Code style as in `factory_sim/`: one statement per line, minimal comments, short module docstrings, compact
  README.
- The guide's rule: AI-written code must be tagged as unverified until a person has checked it (as the v2 files
  are, `"""This file is UN-VERIFIED"""`).
- Every certificate and every PDDL test plan is VAL-checked.

## 11. Decided by the user / left to the builder

**Decided:** approach 3 (one core, two clocks); human play as in FactorySim; our domain, several agents,
multi-tasking, durative actions; Overcooked look; upstream contribution optional; forgiving world, strict planner
(section 4).

**Builder's calls (marked as such):** working name `overcooked_kitchen`; new folder instead of changing
`overcooked_ai_v2/` in place (keeps current experiments working); keys and HUD details; sprite drawing method;
observation and graph features; dataset sizes per tier (start from v2 / v3 sizes); test list.

**Builder's calls after the review (2026-11-15):**
- *Stuck on the grid.* `move` reserves only its target tile, so cooks can block each other. The grid executor plans
  paths around other cooks every step. An idle cook (no running action) that blocks another cook's path gets a
  core `move` to a free neighbouring tile ("yield"), so the step aside is a core action and core and grid never
  disagree. If no cook makes progress for `stuck_steps` (config, default 50), the run ends with failure `deadlock`
  (as FactorySim).
- *Existing instances.* The package reads the existing v2 `scenario.yaml` files and v3 instances unchanged and keeps
  their ids, certificates and deadlines (an `import` command), so the current splits carry over. Old results stay
  with the old code; the paper's kitchen numbers are re-run with the package's `bench`. `tiflax/` is not changed.
- *Action space.* Decisions are compound tasks, "go to X and do Y" (walking is folded into the task, as in V13 and
  FactorySim's pickup / deliver); the PDDL `move` stays for planners. Gymnasium uses a fixed-size slot encoding
  (wait, then per station and per held-item kind, sized by maximum counts in the config) with a mask, as
  FactorySim; the Python API and the graph view also give the variable-size list of legal tasks for GNN policies.
  The TASK menu in human play lists these compound tasks.

**Open:** the public name (avoid "OvercookedV2", which is Oxford's ICLR 2025 environment).

## 12. Tests (minimum)

- Nominal agreement: for scripted and dispatcher runs, the sampled clock with noise off and the grid clock without
  contention give the same action order and end times (within one step), 1–4 cooks.
- POPF → VAL → core and → grid on the examples; every VAL-valid plan runs with no burn and no failure.
- Forgiving world: a soup left too long burns and the pot needs scrubbing; a late delivery is accepted and counted;
  a non-recipe pot spoils; partial chopping keeps its progress; export of each such state to PDDL is valid input for
  POPF (parses, and POPF solves the k01-style continuation).
- Multi-cook grid rules: same-tile and swap conflicts, a blocked corridor lengthens an action.
- Burning, dirty pots, plate cycle, deadlines, as the v2 tests.
- Determinism with a fixed seed; Gymnasium and PettingZoo API checks; headless render and a headless play smoke test.
