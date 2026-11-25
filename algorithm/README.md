# algorithm: T-iFLAX

One file per version; sub-versions live in the parent file. Each file starts with what the version does, what it needs
and its earlier result. The algorithm imports no simulator: it reads instance folders (`hl_problem.json` + a domain
spec YAML, or `domain.pddl` / `problem.pddl` for V10–V13) and is pointed at them by a test harness
(`TIFLAX_DATA`, `TIFLAX_RESULTS`, `TIFLAX_DOMAINS`, `TIFLAX_BENCHES`, `TIFLAX_TOOLS`; see `__init__.py`).

| File | Version |
| --- | --- |
| `v1.py` | V1: iFLAX expansion + recovery branches + improvement phase, two-head GNN, bilevel training (engine of V2–V5) |
| `v2.py` | V2 / V2.1: gated reward of object and temporal scores |
| `v3.py` | V3 / V3U / V3.1: per-agent minimal seeds in parallel |
| `v4.py` | V4 / V4.1 / V4.2: work-optimal relaxed-plan seeds (V4.2: the BTP report's final version) |
| `v5.py` | V5 / V5a: set scorer on POPF outcomes |
| `v6.py` | V6: learned temporal triage |
| `v7.py` | V7: listwise set ranker |
| `v8.py` | V8: learned ranking + a constructive scheduler (scheduler plug-in) |
| `v9.py` | V9: RL over the scheduler's macro steps (environment plug-in) |
| `v10.py` | V10: domain-independent temporal planner over snap actions (no learning) |
| `v11.py` | V11: relational GNN policy over PDDL moves, imitation + expert iteration |
| `v12.py` | V12: V11's GNN for multi-agent execution with random durations, DAgger + RL (domain module) |
| `v13.py` | V13: learned task choice among a dispatcher's options, DAgger + RL (domain module) |
| `v14.py` | V14: tropical event-graph network (learned max-plus GNN over an event graph built from the PDDL state; exactly CPM at initialisation), V13's options and DAgger + RL |

Shared core: `grounding.py` (Task, Pi|S, grounding, temporal relaxed planning graph, COMP_T), `semantics.py`
(PDDL 2.1 semantics, plan simulator, schedule analysis, re-timing), `gnn.py` (object graph, GNN scorer, labels),
`planning.py` (budget, POPF / OPTIC calls, attempt = plan on Pi|S + VAL on Pi), `menv.py` (multi-agent execution
environment with random durations, on V10's semantics), `pddlkit/` (domain spec, HL problem, PDDL writer, plan,
POPF / VAL wrappers). Every version has `build(key, spec, seed=0, **overrides)` -> a planner with `solve(task, budget_s)` -> `RunResult`
(V10 / V11 through `v10.TextPlanner`; V8 / V9 / V11 are built by the harnesses' `*_methods.py`, which add their
simulator plug-ins and models). Adding a version: `vN.py` with `build`, then register it in the `METHODS` of the
`*_test/evaluate.py` that should run it.
