# overcooked_kitchen Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A self-contained, open-source package `overcooked_kitchen` (in `temporal_planning/overcooked_kitchen/`): the v2 kitchen with several cooks, durative tasks, a forgiving Overcooked-style world, one core with a sampled clock and a grid clock, PDDL bridge, Gymnasium / PettingZoo, Overcooked-AI sprites, human play, benchmark suite.

**Architecture:** One `World` (state + world rules, ported from `overcooked_ai_v2/kitchen.py`) is the single source of truth for what an interact does, used by both clocks. A task = "go to station X and do the interact event there" (catalogue actions only); its legality is `World.interact_event` evaluated at a hypothetical pose, so task legality and grid behaviour cannot disagree. The core `Sim` schedules task starts / ends and world timers; the sampled clock draws durations, the grid clock walks and strokes on the grid. PDDL is generated from the vendored domain catalogue (`domain.yaml`) and from the current `World`.

**Tech Stack:** Python ≥ 3.9, numpy, pyyaml, gymnasium, pillow; optional pettingzoo, pygame, pytest, matplotlib; external POPF / OPTIC / VAL binaries.

**Spec:** `temporal_planning/docs/specs/2026-11-15-overcooked-kitchen-design.md`

## Global Constraints

- Python ≥ 3.9. Core dependencies: numpy, pyyaml, gymnasium, pillow. Optional: pygame (play), pettingzoo, pytest.
- POPF, OPTIC and VAL are external binaries (not shipped; POPF is GPL). Found on `PATH` (`popf`, `optic-clp`, `Validate`) or via `OVERCOOKED_KITCHEN_POPF`, `OVERCOOKED_KITCHEN_OPTIC`, `OVERCOOKED_KITCHEN_VAL`; PDDL tests skip if missing. Locally: `temporal_planning/tools/bin/`.
- Code style as `factory_sim/`: one statement per line, minimal comments, short module docstrings, compact README.
- Every new module starts with its docstring followed by `"""This file is UN-VERIFIED"""` (the guide's rule).
- Every certificate and every PDDL test plan is VAL-checked.
- Do not modify `overcooked_ai_v1/`, `overcooked_ai_v2/`, `overcooked_ai_v3/`, `tiflax/`, `pddl_generator/` (copy, never import them).
- Upstream sprites: Overcooked-AI (MIT, © 2019 Center for Human-Compatible AI), attribution in README and `NOTICE`.
- Time unit = one grid step; a task's interact effect is applied at `t_end - 1` (v2 convention: effects visible at `t_end`).
- Avoid the name "OvercookedV2" anywhere (Oxford's ICLR 2025 environment).
- Commands below run from `temporal_planning/overcooked_kitchen/` with `.venv/bin/` and `unset PYTHONPATH` (ROS leak).

## Review Focus

1. Two cooks choose tasks at the same station in the same decision round → the second cook must not be offered it (station lock); test in Task 6 `test_station_lock_same_round`.
2. A soup burns while a cook is walking to dish it up (sampled clock) → the task ends with no effect, cook becomes free, run continues, burn counted; Task 6 `test_task_fails_softly_when_soup_burns_en_route`.
3. Two pending orders for the same dish, one already past its deadline → delivery goes to the earliest-deadline pending order and lateness is counted, not a failure; Task 2 `test_late_delivery_counts_lateness`.
4. A station whose access poses are unreachable from a cook (walled-off region) → no task offered, no crash, no deadlock false alarm; Task 5 `test_unreachable_station_not_offered`.
5. Export to PDDL while a cook holds a soup plate and a pot is cooking → the problem parses with VAL's Parser and POPF finds a plan; Task 9 `test_export_mid_run_holding_soup`.

---

### Task 1: Package skeleton and layout

**Files:**
- Create: `overcooked_kitchen/pyproject.toml`, `overcooked_kitchen/LICENSE` (MIT, "overcooked_kitchen authors"), `overcooked_kitchen/NOTICE`, `overcooked_kitchen/.gitignore` (`.venv/`, `__pycache__/`, `*.egg-info/`, `runs/`), `overcooked_kitchen/overcooked_kitchen/__init__.py`, `overcooked_kitchen/overcooked_kitchen/layout.py`
- Test: `overcooked_kitchen/tests/test_layout.py`

**Interfaces:**
- Produces (`layout.py`): `NORTH, SOUTH, EAST, WEST, DIRS, DIR_NAME, DIR_BY_NAME`; `INGREDIENTS = ("onion", "tomato", "carrot")`; `STATION_CHARS`, `NAME_PREFIX`, `PDDL_TYPE`, `PROVIDES` exactly as in `overcooked_ai_v2/kitchen.py`; `Cell = Tuple[int,int]`, `Pose = Tuple[int,int,Tuple[int,int]]`; `@dataclass Station(name: str, kind: str, cell: Cell)`; `class Layout(rows)` with `.rows .H .W .starts: Dict[int, Cell] .terrain`, `is_floor(c) -> bool`, `access_poses(cell) -> List[Pose]`, `bfs_poses(src: Pose) -> Dict[Pose, int]`, `next_step(src: Pose, dst: Pose, blocked=frozenset()) -> Optional[Tuple[int,int]]`, `stations(counter_policy="none", extra_counters=()) -> List[Station]` (v2 alphabet); `pose_name(p) -> str`, `tile_name(c) -> str`.

- [ ] **Step 1:** `git init` in `overcooked_kitchen/`; write `pyproject.toml` (name `overcooked-kitchen`, version `0.1.0`, deps `numpy>=1.22, pyyaml>=6, gymnasium>=0.29, pillow>=9`; extras `play=[pygame>=2.5]`, `marl=[pettingzoo>=1.24]`, `plot=[matplotlib>=3.5]`, `dev=[pytest>=7]`, `all=[...all of them]`; script `overcooked-kitchen = "overcooked_kitchen.cli:main"`; package data `assets/*`, `examples/*.yaml`, `domain.yaml`); create `.venv`, `pip install -e ".[all]"`.
- [ ] **Step 2: Write the failing tests** in `tests/test_layout.py`:
  - `test_bfs_turn_costs_one_step`: on `["XXXXX","X 1 X","XXPXX"]`, `bfs_poses((2,1,NORTH))[(2,1,SOUTH)] == 1`.
  - `test_access_poses_face_station`: pot at `(2,2)`: `access_poses((2,2)) == [(2,1,SOUTH)]`.
  - `test_stations_v2_alphabet`: k02 grid (copy from `overcooked_ai_v2/scenarios/examples/k02_two_dishes.yaml`) gives kinds counted `{"onion_crate":1,"tomato_crate":1,"carrot_crate":1,"pot":2,"board":2,"serving":1,"dropoff":1,"rack":1,"sink":1}` with `counter_policy="none"`.
  - `test_next_step_avoids_blocked`: returns `None` when the only path tile is in `blocked`.
- [ ] **Step 3:** Run `pytest tests/test_layout.py -v` → FAIL (module missing).
- [ ] **Step 4:** Implement `layout.py` by merging `overcooked_ai_v1/layout.py` with `KitchenLayout.stations` from `overcooked_ai_v2/kitchen.py` (copy, no imports from those packages).
- [ ] **Step 5:** Run → PASS. Commit `feat: package skeleton and grid layout`.

### Task 2: World (state and forgiving world rules)

**Files:**
- Create: `overcooked_kitchen/overcooked_kitchen/world.py`
- Test: `overcooked_kitchen/tests/test_world.py`

**Interfaces:**
- Consumes: Task 1 `Layout`, constants.
- Produces: `DEFAULTS` (v2 values: chop onion 4 / tomato 3 / carrot 5, cook_base 2, cook onion 5 / tomato 3 / carrot 6, burn 20, scrub_pot 4, wash_plate 3, eat 12, max_ingredients 4); `sim_params(overrides=None) -> dict`; `cook_time(params, comp) -> int`; dataclasses `Plate(pid, state="clean", soup=None)`, `Pot(contents, state="idle", ready_at=None, cook_start=None, scrub=0, recipe=True)` with pot states `idle | cooking | ready | burnt | spoiled | dirty`; `Player(pos, facing, held=None)`; `class World(rows, params, plates_clean=1, plates_dirty=0, dirty_pots=0, orders=())` with fields `lay t kind players surface progress pots stack sink plates orders served late diners events burnt spoiled`, and methods:
  - `facing_cell(i) -> Cell`
  - `pending_orders() -> List[dict]` (orders not yet served, late ones included)
  - `interact_event(i, pose: Optional[Pose] = None) -> Optional[str]` — what INTERACT by cook `i` would do, as if standing at `pose` when given. Events: `take_raw put_raw put_chopped put_plate chop take_chopped add start dish_up scrub deliver take_plate rack_plate put_in_sink wash`.
  - `is_recipe(comp) -> bool` (comp equals some order's comp)
  - `apply(i, event, cell, t, strokes=1) -> None` (strokes > 1 only for `chop`, `wash`, `scrub`)
  - `advance(t) -> None` (fires timers with time ≤ t: cooking → ready at `ready_at`; ready → burnt at `ready_at + burn`; a non-recipe pot → `spoiled` at `ready_at`; diners return the dirty plate at their return time; sets `self.t = t`)
  - `next_timer() -> Optional[float]`
  - `snapshot() -> dict` (v2 snapshot keys plus `late`, `spoiled`)
- World rules (spec §4, forgiving): `start` allowed for any non-empty idle pot (non-recipe → `pot.recipe=False`, becomes `spoiled` when cooked); `burnt` and `spoiled` pots need `scrub`; `dish_up` only from `ready`; `deliver` allowed for a plated soup matching any pending order, goes to the earliest-deadline pending order, `served[o] = t + 1`, `late[o] = max(0, t + 1 - deadline)` when it has a deadline; partial `chop` / `wash` / `scrub` progress is kept per cell when the cook leaves.

- [ ] **Step 1: Write the failing tests** (port the four world tests of `tests/test_overcooked_v2.py`, driving `World` directly instead of `Kitchen`): `test_chop_then_add`, `test_pot_timing_and_dirty_cycle`, `test_soup_burns_and_burnt_pot_needs_scrubbing`, plus:
  - `test_late_delivery_counts_lateness`: two orders of the same dish with deadlines 10 and 1000; deliver one plated soup at t=50 → it goes to the deadline-10 order, `served == {"o1": 51}`, `late == {"o1": 41}`.
  - `test_non_recipe_pot_spoils`: light a pot holding one onion when only tomato soups are ordered → at `ready_at` state `spoiled`, `interact_event` with empty hand = `scrub`, with a clean plate = `None`.
  - `test_partial_chop_keeps_progress`: 2 chop strokes on an onion (needs 4), cook walks away, returns → 2 more strokes finish it (`surface == ("chopped","onion")`).
  - `test_interact_event_at_hypothetical_pose`: a cook elsewhere holding a chopped onion: `interact_event(0, pose_facing_pot) == "add"`.
- [ ] **Step 2:** Run `pytest tests/test_world.py -v` → FAIL.
- [ ] **Step 3:** Implement `world.py` from `overcooked_ai_v2/kitchen.py` (`Kitchen` minus `step` / `_move`, which go to Task 3), adding the rules above.
- [ ] **Step 4:** Run → PASS. Commit `feat: world state and forgiving rules`.

### Task 3: Grid stepping (Overcooked motion rules)

**Files:**
- Create: `overcooked_kitchen/overcooked_kitchen/grid.py`
- Test: `overcooked_kitchen/tests/test_grid.py`

**Interfaces:**
- Consumes: `World`, `DIRS`.
- Produces: `STAY = "stay"`, `INTERACT = "interact"`, `ACTIONS = (NORTH, SOUTH, EAST, WEST, STAY, INTERACT)`, `ACTION_NAMES = ("north","south","east","west","stay","interact")`; `step(world, actions: Sequence) -> List[Optional[str]]` — one transition t → t+1 (interacts first in cook order, then movement with the v2 `_move` rules, then `world.advance(t+1)`); returns the event each cook performed (None if none).

- [ ] **Step 1: Write the failing tests:** `test_same_tile_conflict_both_stay`, `test_swap_conflict_both_stay`, `test_move_into_counter_only_turns`, `test_interact_returns_event` (cook facing an onion crate with empty hand → `["take_raw"]`).
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement from `Kitchen.step` / `Kitchen._move`. **Step 4:** Run → PASS. Commit `feat: grid stepping`.

### Task 4: Scenarios and examples

**Files:**
- Create: `overcooked_kitchen/overcooked_kitchen/scenarios.py`, `overcooked_kitchen/overcooked_kitchen/examples/k01_one_soup.yaml` … `k04_burn_risk.yaml` (copied from `overcooked_ai_v2/scenarios/examples/`), `overcooked_kitchen/overcooked_kitchen/examples/m01_two_cooks.yaml`, `m02_four_cooks.yaml` (copied from two v3 test instances `overcooked_ai_v3/generated_pddl/v3_test/<id>/scenario.yaml`: one 3-cook, one 4-cook; keep their `scenario_id`)
- Test: `overcooked_kitchen/tests/test_scenarios.py`

**Interfaces:**
- Produces: `load_scenario(path_or_dict) -> dict` (v2 / v3 `scenario.yaml` format unchanged: `scenario_id, grid, counters, plates, dirty_pots, sim, orders`; validates ingredients; adds `_path`, `_sha256`); `scenario_orders(sc) -> List[dict]` (`{"id": "o1", "comp", "deadline", "dish"}`, v2 rules); `make_world(sc) -> World`; `example(name) -> str` (path of a bundled example; name without `.yaml`); `examples() -> List[str]`.

- [ ] **Step 1: Write the failing tests:** `test_load_all_examples` (6 examples load, `make_world` gives `len(players)` 1,1,1,1,3 or 4 as in the files), `test_orders_expand_count`, `test_v3_scenario_keeps_id_and_counters`.
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement from `overcooked_ai_v2/builder.py` (`load_scenario`, `scenario_orders`, `make_kitchen`). **Step 4:** Run → PASS. Commit `feat: scenarios and examples`.

### Task 5: Tasks (compound catalogue actions) and legality

**Files:**
- Create: `overcooked_kitchen/overcooked_kitchen/domain.yaml` (copy of `overcooked_ai_v2/domain.yaml`), `overcooked_kitchen/overcooked_kitchen/tasks.py`
- Test: `overcooked_kitchen/tests/test_tasks.py`

**Interfaces:**
- Consumes: `World`, `Layout`.
- Produces:
  - `EVENT_ACTION: Dict[str, str]` — grid event → catalogue action name, read from `domain.yaml` `sim.event` (`take_raw → take-raw`, `chop → chop`, `add → add-onion|add-tomato|add-carrot` by held ingredient, `start → start-cooking`, …); events with no catalogue action are off-catalogue.
  - `@dataclass(frozen=True) Task(cook: int, station: str, cell: Cell, pose: Pose, event: str, action: str, strokes: int, travel: int)`; `Task.duration -> int = travel + strokes`.
  - `WAIT = ("wait",)`.
  - `stations_of(world) -> List[Station]` (cached on the world; counters per scenario `counters` policy).
  - `legal_tasks(world, i, locked_cells=frozenset(), blocked_tiles=frozenset()) -> List[Task]`: for every station not in `locked_cells`, the nearest access pose (by `bfs_poses` from the cook's pose) whose tile is not in `blocked_tiles`, event = `world.interact_event(i, pose)`; keep only catalogue events, and `start` only when `world.is_recipe(pot comp)`; `strokes` = remaining strokes for chop / wash / scrub (params minus kept progress), 1 otherwise; `travel` = BFS steps.
- [ ] **Step 1: Write the failing tests:**
  - `test_k01_first_tasks`: k01 start state → legal events ⊆ `{"take_raw"}` and every crate in the core is offered.
  - `test_events_match_domain_catalogue`: every event in `EVENT_ACTION` is the `sim.event` of an action in `domain.yaml`; 26 actions in the file.
  - `test_non_recipe_start_not_offered`: pot with one onion, only tomato orders → no `start` task.
  - `test_locked_station_not_offered`, `test_blocked_pose_uses_other_access_pose`.
  - `test_unreachable_station_not_offered` (Review Focus 4): grid with a walled-off pot → no task for it, other tasks still offered.
  - `test_remaining_strokes_after_partial_chop`: after 2 of 4 strokes, the chop task has `strokes == 2`.
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** Run → PASS. Commit `feat: compound tasks and legality`.

### Task 6: Core engine and sampled clock

**Files:**
- Create: `overcooked_kitchen/overcooked_kitchen/core.py`, `overcooked_kitchen/overcooked_kitchen/clocks.py` (sampled clock part)
- Test: `overcooked_kitchen/tests/test_core.py`

**Interfaces:**
- Consumes: `World`, `make_world`, `legal_tasks`, `Task`, `WAIT`.
- Produces:
  - `clocks.Noise(cv: Dict[str, float] = None, default: float = 0.0, clip=(0.4, 3.0))` with `sample(category, nominal, rng) -> float` (lognormal, mean 1 × nominal, as `tiflax/menv.Noise`); categories `move chop wash scrub interact cook eat`; `Noise.default_profile(scale=1.0)` = `{"move": 0.25, "chop": 0.2, "wash": 0.2, "scrub": 0.2, "interact": 0.1, "cook": 0.1, "eat": 0.3}` × scale.
  - `class Sim(scenario, clock="sampled", seed=0, noise=None, cv_scale=1.0, horizon=None, record=False, stuck_steps=50)`; attributes `world`, `t`, `cooks: List[str]` (`"c1".."cn"`), `running: Dict[str, RunningTask]`, `log: List[dict]` (task start / end / fail / world events), `frames` (when `record`).
  - `legal(cook: str) -> List` (`WAIT` first, then `legal_tasks` with stations locked by other cooks' running tasks and tiles targeted or occupied by other cooks blocked); `act(cook, task_or_wait)` (raises `ValueError` on an illegal task); `pending() -> List[str]` (free cooks with more than `WAIT`, in cook order, each once per event time); `next_decision() -> Optional[str]`; `run(policy, max_decisions=10**6) -> dict` with `policy(sim, cook, legal) -> item of legal`; `metrics() -> dict` with keys `success failure makespan served late total_lateness on_time burnt spoiled decisions cook_utilisation idle blocked wall_s`.
  - Sampled clock: duration = `noise.sample("move", travel) + noise.sample(category(event), strokes)`; the cook is shown at the target pose at the end; at `t_end` the core calls `world.advance(t_end - 1)`, checks `world.interact_event(i, task.pose) == task.event`, then `world.apply(i, event, cell, t_end - 1, strokes)` or logs `task_failed` (no effect); cooking and eating durations are drawn once per pot / diner from categories `cook` / `eat` and written into the timers.
  - Failure values: `"deadlock"` (no running task, no timer, pending orders, no cook has a non-wait task), `"stalled"` (policy waited with nothing scheduled), `"horizon"`.
- [ ] **Step 1: Write the failing tests:**
  - `test_k01_scripted_nominal_makespan`: k01, `noise=None`, a scripted policy repeating the first legal non-wait task → `success`, `makespan` equal to the v2 POPF makespan of k01 (69) or lower, `burnt == 0`.
  - `test_station_lock_same_round` (Review Focus 1): m01 (3 cooks) at t=0, after `act("c1", crate task)`, `legal("c2")` has no task at that crate's cell.
  - `test_task_fails_softly_when_soup_burns_en_route` (Review Focus 2): scenario with burn 1 and a cook far from the pot → the dish-up task ends with `task_failed` in `log`, cook free, `metrics()["burnt"] == 1`, run continues.
  - `test_noise_reproducible`: same seed → same metrics; different seeds → different makespans on k03 with `cv_scale=1`.
  - `test_deadlock_detected`: a scenario whose order needs a carrot and has no carrot crate → `failure == "deadlock"`.
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** Run → PASS. Commit `feat: core engine and sampled clock`.

### Task 7: Grid clock, yielding, and driving cooks

**Files:**
- Modify: `overcooked_kitchen/overcooked_kitchen/clocks.py`, `overcooked_kitchen/overcooked_kitchen/core.py`
- Test: `overcooked_kitchen/tests/test_grid_clock.py`

**Interfaces:**
- Consumes: `grid.step`, `Layout.next_step`, Task 6 `Sim`.
- Produces:
  - `Sim(clock="grid", step_rate=None)`: each running task walks (`next_step` with other cooks' tiles blocked, replanned every step) then sends `INTERACT` while `world.interact_event(i) == task.event` until `strokes` are done; the task ends when its last stroke is applied; time advances one step at a time; world timers fire inside `grid.step`.
  - Yield: an idle cook standing on the next tile of another cook's path for 2 consecutive steps gets a core `move` task (`Task(event="move", action="move", strokes=0)`) to the nearest free floor tile not on that path; logged as `yield`.
  - `Sim.stuck_steps`: no cook changes position, held item or station progress for `stuck_steps` steps while tasks run → `failure = "deadlock"`.
  - Driving: `Sim.set_mode(cook, mode)` with modes `"auto" | "task" | "drive"`; `Sim.drive(cook, action)` queues one grid action for the next step (grid clock only; a driven cook gets no tasks and is skipped by `pending()`); `Sim.tick() -> None` advances exactly one grid step (used by human play and the low-level env).
- [ ] **Step 1: Write the failing tests:**
  - `test_nominal_agreement_one_cook`: k01–k04 with the scripted policy of Task 6: grid clock and sampled clock (no noise) give the same sequence of `(cook, event, station)` task ends and end times within 1 step.
  - `test_nominal_agreement_cooks_apart`: a test grid with 2 and with 4 cooks in separate walled regions, each with its own crate, board, pot, rack, sink, window and one order → same task-end sequence and end times within 1 step on both clocks.
  - `test_blocked_corridor_lengthens_task`: 2 cooks, one-tile corridor → the grid task takes more steps than its nominal `duration`, and a `yield` appears in `log`.
  - `test_drive_mode_moves_and_interacts`: k01, `set_mode("c1","drive")`, drive to the onion crate and `INTERACT` → cook holds a raw onion; `pending()` never returns `c1`.
  - `test_grid_deadlock_after_stuck_steps`: two cooks wanting each other's tile in a dead-end with yielding impossible → `failure == "deadlock"` after `stuck_steps`.
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** Run → PASS. Commit `feat: grid clock, yielding, driving`.

### Task 8: Dispatcher and random baselines

**Files:**
- Create: `overcooked_kitchen/overcooked_kitchen/baselines.py`
- Test: `overcooked_kitchen/tests/test_baselines.py`

**Interfaces:**
- Consumes: `Sim.legal`, `Task`, `World`.
- Produces: `make(name: str, seed: int = 0) -> policy` for `"random"` and `"dispatcher"` (Task 9 adds `"popf"`, `"popf-replan"`); `class Dispatcher` — `__call__(sim, cook, legal) -> task or WAIT`, a pure function of `sim.world` and `sim.running` with the priorities of `overcooked_ai_v3/dispatcher.py` docstring (finish chopping / take needed chopped veg; carry what is held where needed; claim a soup ready soon (fetch or wash a plate); light a full recipe pot; scrub a dirty pot when no clean pot is free; fetch the most needed vegetable at the nearest crate; wash plates when clean plates are short; else wait), re-implemented on tasks (no PDDL state).
- [ ] **Step 1: Write the failing tests:**
  - `test_dispatcher_serves_examples`: k01–k04, m01, m02 with sampled clock, no noise → `success`, `burnt == 0`.
  - `test_dispatcher_under_noise`: m01, m02, seeds 0–4, `cv_scale=1` → success on ≥ 4 of 5 seeds each.
  - `test_dispatcher_grid_clock`: m01 on the grid clock → `success`.
  - `test_random_never_crashes`: all examples, 3 seeds, `horizon=3000` → `metrics()` returns (success not required).
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** Run → PASS. Commit `feat: dispatcher and random baselines`.

### Task 9: PDDL bridge (domain, export, planners, plan execution)

**Files:**
- Create: `overcooked_kitchen/overcooked_kitchen/pddlkit/{__init__,domain_spec,problem,writer,plan,planners}.py` (copied from `pddl_generator/`; `planners.binary(name)` resolves env vars / `PATH` per Global Constraints; no `configs/tools.yaml`), `overcooked_kitchen/overcooked_kitchen/pddl.py`
- Modify: `overcooked_kitchen/overcooked_kitchen/baselines.py` (add `"popf"`, `"popf-replan"`)
- Test: `overcooked_kitchen/tests/test_pddl.py`

**Interfaces:**
- Consumes: `World`, `Sim`, `Task`, `domain.yaml`.
- Produces:
  - `pddlkit.planners.available(name: str) -> bool` (`"popf" | "optic" | "validate" | "parser"`); `pddlkit.planners.binary(name) -> str` (raises `FileNotFoundError`)
  - `spec() -> DomainSpec`; `domain_text() -> str`
  - `export(sim, hard_deadlines=True) -> HLProblem` — `overcooked_ai_v2/builder.encode` generalised to the current state, time shifted so now = 0: held items → `holding-*` facts and no `hand-empty`; cooking pot → `on-stove` + timed literal `soup-ready` at its remaining time, ready pot → `on-stove` + `soup-ready` (stove-timer end as the burn deadline, as `tiflax/snapshot.py`); burnt / spoiled / dirty pot → `pot-dirty`; partial chop / wash / scrub progress → exported as not started (full strokes; pessimistic, keeps the domain unchanged); diners → timed literals returning the dirty plate; a running task → its cook not `idle` and its end effects as timed literals at its nominal end; deadlines shifted to now; an order past or unable to meet its deadline → no deadline, listed in `problem.annotations["dropped_deadlines"]`; `hard_deadlines=False` drops all deadlines.
  - `problem_text(problem) -> str`; `write(sim, out_dir, hard_deadlines=True) -> Tuple[str, str]`
  - `solve(sim, planner="popf", timeout=60.0) -> Optional[TemporalPlan]` (None if no binary, timeout or no plan)
  - `validate(domain_path, problem_path, plan) -> bool`
  - `class PlanPolicy(plan: TemporalPlan, problem: HLProblem, mode="asap")` — policy for `Sim.run`: per cook, the plan's interact actions in start order become tasks (moves are folded into tasks); a task is chosen when it is legal and (mode `"timed"`) its planned start has come and the cook's cross-cook predecessors (planned earlier at the same tile or station) have ended; else `WAIT`.
  - Baselines `"popf"` (solve at t=0, execute with `PlanPolicy`), `"popf-replan"` (re-solve from `export(sim)` when any task ends more than `tol=3` steps later than planned; keep the old plan if the new solve fails).
- [ ] **Step 1: Write the failing tests** (module-level skip if `planners.available("popf")` and `available("validate")` are false):
  - `test_domain_has_26_actions_and_parses` (VAL Parser accepts `domain_text()`).
  - `test_popf_val_core_and_grid` (parametrised k01, k02): POPF plan on the initial export is VAL-valid; executed with `PlanPolicy` on the sampled clock (no noise) and on the grid clock → `success`, `burnt == 0`, and k02 deadlines met.
  - `test_export_mid_run_holding_soup` (Review Focus 5): k02 with the dispatcher, stop at the first decision where a cook holds a plated soup and a pot is cooking; `export` → VAL Parser accepts it and POPF finds a plan within 60 s.
  - `test_export_messy_states`: a burnt pot, a spoiled pot, a half-chopped onion, a late order → export parses; burnt / spoiled pots appear as `pot-dirty`; the late order has no deadline and is in `dropped_deadlines`.
  - `test_popf_replan_baseline_k02_noise`: k02, `cv_scale=1`, seed 0 → `success`.
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** Run → PASS. Commit `feat: PDDL bridge and planner baselines`.

### Task 10: Gymnasium and PettingZoo

**Files:**
- Create: `overcooked_kitchen/overcooked_kitchen/env.py`
- Modify: `overcooked_kitchen/overcooked_kitchen/__init__.py` (register envs when gymnasium is importable)
- Test: `overcooked_kitchen/tests/test_env.py`

**Interfaces:**
- Consumes: `Sim`, `legal`, `Task`, `grid.ACTIONS`.
- Produces:
  - `build_sim(instance, seed=0, **kw) -> Sim` (instance: example name, scenario path, or scenario dict).
  - Task-level action encoding: `0` = wait, `1 + k` = the legal task at station `k` of `stations_of(world)` (at most one event per station for a given cook state, so this is unambiguous); `max_stations` pads; `encode_task(sim, task) -> int`, `decode(sim, cook, index) -> task or WAIT`, `mask(sim, cook) -> np.ndarray[int8]`.
  - `observe(sim, cook) -> np.ndarray[float32]` (time, per cook: position / held kind / busy time left / is-current; per station: kind one-hot, holds, progress fraction, pot state one-hot and time to ready / burn; per order: pending, comp, slack to deadline; fixed length from `max_cooks`, `max_stations`, `max_orders`) and `graph(sim) -> dict` (nodes cooks, stations, orders; edges `at`, `needs`, `holds`, travel weights), as `factory_sim/env.py`.
  - `KitchenEnv(instance="k01_one_soup", max_cooks=4, max_stations=40, max_orders=12, time_scale=100.0, fail_penalty=10.0, burn_penalty=1.0, render_mode=None, **sim_kw)` registered as `OvercookedKitchen-v0`; reward = `-(Δt)/time_scale - (new lateness)/time_scale - burn_penalty * new burns`, `-fail_penalty` on failure; `info["agent"]`, `info["action_mask"]`.
  - `KitchenGridEnv(instance, ...)` registered as `OvercookedKitchenGrid-v0`: `MultiDiscrete([6] * n_cooks)` over `grid.ACTIONS`, grid clock, `Sim.tick()` per step, same reward.
  - `aec_env(**kw)` (PettingZoo AEC, agents `c1..cn`, shared reward).
- [ ] **Step 1: Write the failing tests:** `test_gym_env_checker` (`gymnasium.utils.env_checker.check_env` on both envs, k01), `test_masked_random_episode_terminates` (m01, masked random, ends with `terminated or truncated`), `test_pettingzoo_api` (`pettingzoo.test.api_test`, skip if pettingzoo missing), `test_encode_decode_roundtrip`.
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** Run → PASS. Commit `feat: gymnasium and pettingzoo envs`.

### Task 11: Rendering with Overcooked-AI sprites

**Files:**
- Create: `overcooked_kitchen/overcooked_kitchen/assets/` (copy `chefs.png/.json`, `objects.png/.json`, `soups.png/.json`, `terrain.png/.json` from `overcooked_ai_v1/upstream/src/overcooked_ai_py/data/graphics/`), `overcooked_kitchen/overcooked_kitchen/assets/make_sprites.py` (draws the new 15×15 sprites in the upstream palette into `assets/extra.png` + `extra.json`: `carrot`, `carrots` crate, `board`, `onion-chopped`, `tomato-chopped`, `carrot-chopped`, `sink`, `dropoff`, `dish-dirty`, `pot-dirty`, `soup-carrot-*` frames), `overcooked_kitchen/overcooked_kitchen/render.py`
- Test: `overcooked_kitchen/tests/test_render.py`

**Interfaces:**
- Consumes: `World.snapshot`, `Sim.frames`, `Sim.log`.
- Produces: `class Sprites(scale=4)` loading all sheets; `frame(sim, hud=True, timeline=False) -> PIL.Image` (terrain, stations and their contents, chefs with facing, hat colour per cook and held item, pot progress bar, burn countdown bar in red, diner eating bar, HUD with orders, deadlines, lateness, time; chefs drawn between poses when the sampled clock is mid-walk, interpolating along the BFS path); `gif(sim, path, fps=8) -> str` (needs `record=True`); `gantt(sim, path) -> str` (one lane per cook and per pot, from `log`).
- [ ] **Step 1: Write the failing tests:** `test_frame_size_k02` (image width = grid W × 15 × scale, height ≥ grid H × 15 × scale), `test_all_extra_sprites_exist` (every name listed above is a frame in `extra.json`), `test_gif_writes` (m01 dispatcher run with `record=True` → GIF with > 10 frames), `test_burnt_pot_uses_explosion_sprite`.
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement (run `make_sprites.py` once, commit its output). **Step 4:** Run → PASS. Commit `feat: rendering with Overcooked-AI sprites`.

### Task 12: Human play

**Files:**
- Create: `overcooked_kitchen/overcooked_kitchen/play.py`
- Test: `overcooked_kitchen/tests/test_play.py`

**Interfaces:**
- Consumes: `Sim` (grid clock, `set_mode`, `drive`, `tick`), `baselines.make`, `render.frame`.
- Produces: `class Play(scenario, cooks=None, policy="dispatcher", rate=4.0, seed=0, record_path=None)` with `handle_key(key: str) -> None`, `update(dt: float) -> None` (advances `Sim.tick()` at `rate` steps / s; AUTO cooks use the policy, TASK cooks act when the human confirms a menu item), `menu(cook) -> List[str]`, `save(path)` (trajectory: per step the grid actions of every cook + `Sim.log`, JSON); `main(argv=None)` opens the pygame window.
- Keys (spec §6): `1`–`4` select cook; `z` AUTO, `t` TASK, `v` DRIVE; arrows + space drive the selected cook; W/A/S/D + E drive the next DRIVE cook; `[` / `]` move in the TASK menu, Enter confirms; `p` pause, `+` / `-` speed, `r` restart, `q` quit.
- [ ] **Step 1: Write the failing tests** (`SDL_VIDEODRIVER=dummy`; skip if pygame missing): `test_drive_keys_pick_onion` (k01, select 1, `v`, arrow keys to the crate, space → holds onion), `test_task_menu_lists_legal_tasks` (`t`, `menu("c1")` equals the legal tasks' labels), `test_auto_teammates_finish_m01` (cook 1 in DRIVE doing nothing, others AUTO → orders still served, no deadlock), `test_save_trajectory`.
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** Run → PASS. Commit `feat: human play`.

### Task 13: Suite (import, generate, certificates), bench, CLI

**Files:**
- Create: `overcooked_kitchen/overcooked_kitchen/suite.py`, `overcooked_kitchen/overcooked_kitchen/generate.py` (ports `overcooked_ai_v2/make_dataset.py` and `overcooked_ai_v3/make_dataset.py` room / order generation onto `World`; certificates from the dispatcher and, if available, POPF on the core problem, VAL-checked), `overcooked_kitchen/overcooked_kitchen/cli.py`
- Test: `overcooked_kitchen/tests/test_suite.py`

**Interfaces:**
- Consumes: everything above.
- Produces:
  - `suite.import_existing(out_dir, v2_root, v3_root) -> List[dict]` — copies every `scenario.yaml` of `overcooked_ai_v2` splits (`v2_train v2_val v2_test`) and `overcooked_ai_v3` splits (`v3_train v3_val v3_test`) into `out_dir/<split>/<id>.yaml`, keeps ids, copies existing certificate plans and makespans into `out_dir/index.json`.
  - `generate.generate(out_dir, config: dict, splits, seed=0) -> List[dict]` (tiers `s m l xl service` with 1 cook as v2, `m3 l4 …` with 3–4 cooks as v3; config keys as `configs/dataset_v2.yaml` / `dataset_v3.yaml`).
  - `suite.iter_suite(root, split) -> Iterator[dict]`; `suite.bench(root, split, policy, seeds=3, clock="sampled", cv_scale=1.0, out=None) -> List[dict]` (rows `name tier seed success failure makespan makespan_over_cert on_time total_lateness burnt decisions wall_s`) and `suite.table(rows) -> str` (per tier).
  - CLI `overcooked-kitchen {run, play, import, generate, bench, pddl, solve}`: `run --scenario --policy --clock --seed --gif --gantt`; `play --scenario --cooks`; `import --out --v2 --v3`; `generate --config --out --splits`; `bench --suite --split --policy --seeds --clock --cv-scale --out`; `pddl --scenario --out`; `solve --scenario --planner --timeout`.
- [ ] **Step 1: Write the failing tests:** `test_import_keeps_ids_and_certificates` (on a temp copy of 3 v2 and 3 v3 instances), `test_generate_small` (2 instances per tier `s` and `m3`, each certified and VAL-valid when VAL is available, dispatcher-certified otherwise), `test_bench_table_has_tiers`, `test_cli_run_k01` (subprocess `overcooked-kitchen run --scenario k01_one_soup` exits 0, prints `"success": true`).
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** Run → PASS. Commit `feat: suite, bench and CLI`.

### Task 14: Documentation, examples, release check

**Files:**
- Create: `overcooked_kitchen/README.md`, `overcooked_kitchen/examples/quickstart.py`, `overcooked_kitchen/examples/my_policy.py`, `overcooked_kitchen/docs/` (a GIF of m02 with the dispatcher, a Gantt chart, a play screenshot)
- Test: `overcooked_kitchen/tests/test_examples.py`

**Interfaces:**
- Consumes: the CLI and API above.
- README sections (compact, as `factory_sim/README.md`): what it is (our domain: several cooks, multi-tasking, durative actions; forgiving world, strict planner); install; quickstart (CLI, Python API, Gymnasium); human play keys; the kitchen (station alphabet, tasks, durations, burning, plates, diners); clocks; PDDL bridge (export rules incl. pessimistic partial progress); benchmark (tiers, splits, certificates, metrics, baseline table from `bench` with `dispatcher`, `random`, `popf`, `popf-replan`); layout; credits (Overcooked-AI sprites, MIT, CHAI; relation to Overcooked-AI 1.1.0; not OvercookedV2 / JaxMARL); licence.
- [ ] **Step 1: Write the failing test** `test_quickstart_runs` (runs `examples/quickstart.py` headless, exit 0).
- [ ] **Step 2:** Run → FAIL. **Step 3:** Write README, examples, docs media (generated with the CLI). **Step 4:** Run the full suite `pytest -q tests` → all pass (PDDL tests may skip only when binaries are missing; locally they must run with `OVERCOOKED_KITCHEN_POPF` etc. pointing at `../tools/bin/`). Check every module starts with the UN-VERIFIED tag (`grep -L "UN-VERIFIED" overcooked_kitchen/*.py` prints nothing). Commit `docs: README, examples, media`.
