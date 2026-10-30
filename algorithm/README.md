# algorithm: T-iFLAX

## Files

| File | Contents |
| --- | --- |
| `grounding.py` | `Task` (Π, restriction Π\|S, PDDL files), grounding, temporal relaxed planning graph (TRPG): lower bound LB(S), infeasibility proofs, earliest / work-optimal relaxed plans; `comp_t` (COMP_T closure), `agent_type` |
| `semantics.py` | PDDL 2.1 durative-action semantics + plan simulator (agrees with VAL), schedule analysis (causal / threat precedence, CPM slack, critical path, resources), `retime` |
| `gnn.py` | object graph, encode-process-decode GNN, `Scorer` (per-head calibrated scores, reward), `UniformScorer`, training labels |
| `planning.py` | shared `Budget` with a time ledger, cancellable POPF / OPTIC calls, `attempt` = plan on Π\|S + VAL on Π, `ensure_hash_seed` |
| `v1.py` … `v7.py` | one file per version (sub-versions in the parent file), see below |
| `evaluate.py` | runs any method on a split, writes `results/<method>/`, summaries + plots to `results/summary/` |
| `experiments.py` | the quick experiments behind design decisions → `results/diagnostics/` |

Core ideas used by every version:
- **Π\|S is sound**: restriction keeps facts / timed literals over S; with positive conditions only, a plan
  for Π\|S is a plan for Π (`Task.check_restriction_sound` refuses other domains). Every plan is still VAL-checked on Π.
- **COMP_T**: every object in S must be usable by a reachable goal-relevant action, and the goals must be
  relaxed-reachable within their deadline windows; missing objects are added from relaxed plans. No hand-written rules.
- **TRPG proofs**: a goal unreachable in the TRPG of Π\|S (deadlines included) proves Π\|S unsolvable, so
  POPF is not called. A POPF timeout is "unknown", not "insufficient".

## Versions

`./run.sh <ID> [budget]` evaluates; `./run.sh train <ID>` trains. Solved = of 60 stress kitchens at 30 s.

| ID | File | Idea | Model | Solved |
| --- | --- | --- | --- | --- |
| V1 | `v1.py` | iFLAX skeleton: score-guided expansion (T1 = 0.6 T, growing per-attempt caps), parallel recovery Repair / Restart / Rollback / **Temporal** (adds resources that lower LB); two-head GNN (object + temporal score); improvement phase with the leftover budget (deadline tightening, OPTIC anytime, capacity candidates, re-timing) | bilevel, `results/v1/v1_seed0` | 9 |
| V2 | `v2.py` | third GNN output α: reward = α·p_obj + (1−α)·p_temp, trained on "should keep" labels | bilevel, `v2_seed0` | 5 |
| V2.1 | `v2.py` | V2's network; first plan by object score only, gate used in the improvement phase | `v2.1_seed0` | 5 |
| V3 | `v3.py` | **minimal first**: per agent, COMP_T(agent + its earliest-time relaxed plan); best 4 by LB in parallel (cap 0.4 T), V1 as fallback | V1's | 27 |
| V3U | `v3.py` | V3 with uniform scores (ablation: learning adds nothing) | none | 28 |
| V3.1 | `v3.py` | V3 + POPF on the full task in the background during improvement | V1's | 27 |
| V4 | `v4.py` | seeds from **work-optimal** relaxed plans (durations as costs): 3 work-optimal + 1 earliest seed | none | 42 |
| V4.1 | `v4.py` | V4 + second seed round (remaining + two-agent seeds), cap 0.3 T | none | 41 |
| **V4.2** | `v4.py` | V4's seeds + 2 two-agent seeds, all 6 in parallel, cap 0.6 T. **Final.** (6 POPF in parallel vs B1's 1) | none | **48** |
| V5 | `v5.py` | GNN on Π\|S predicts P(POPF solves S) + makespan from POPF trials; ranks V4.2's candidates, one POPF at a time | `v5_seed0` | 39 (TRPG order 42) |
| V5a | `v5.py` | first V5: 3-head GNN (importance, T_min, T_max), temporal filtration + GNN seed inside V4.2 | `v5a_seed0` (retrain) | seeds 47 vs V4.2 48 |
| V6 | `v6.py` | per-object importance + makespan window [t_min, t_max] → learned relaxed-plan costs, triage, seed order | `v6_seed0` | 43 (TRPG order 44) |
| V7 | `v7.py` | listwise ranker of V4.2's candidate sets (V5's set scorer) on V6's labels | `v7_seed0` | 39 |

V5–V7 were tested in the single-core "seeds only" protocol (their own `seedtest` / `test*` commands, outputs in
`results/v5..v7/`); conclusion: learning does not beat the symbolic order on this benchmark, V4.2 is final.
V5 is also available as a full planner (`./run.sh V5`); that run is not part of the reported results.

Parameters: each file has `PRESETS` (per sub-version) on top of its `*Config` dataclass; `MethodConfig` in
`v1.py` holds the shared ones (`q_max`, `gamma`, `expansion_frac`, `recovery`, `attempt_cap_frac`,
`refine_*`, `tighten_frac`, `planner`). One-off overrides: `build(key, spec, seed, **overrides)`.

### Adding a version

1. `algorithm/v8.py`: subclass a planner (e.g. `V4Planner`) and override hooks of `ObjectSelectionPlanner`:
   `rewards`, `first_plan`, `seed_rounds`, `temporal_score`, `front_sets` / `extra_sets`, `refine_start` / `refine_end`, `tighten_fraction`.
2. Add `PRESETS` and `build(key, spec, seed=0, **over)` returning the planner.
3. Register `"V8": ("algorithm.v8", "v8")` in `METHODS` (`evaluate.py`) and in the `case` list of `run.sh`.

### Training (V1, V2)

Bilevel (iFLAX): each epoch every training kitchen (`train`, 40) is planned in train mode with the current
scores (all recovery branches run, the shortest-makespan branch is the label), then one gradient step.
Labels: object = used by the returned plan or a goal object; temporal = max(criticality from CPM slack,
counterfactual σ from replanning without / with resource objects), masked when unknown. After training,
temperatures are fitted on `val` (12). V2 adds the keep label and gate loss. Flags: `--epochs --budget
--parallel --lr --hidden --steps --lam_t --pos_weight --cf_k --cf_cap --seed`.

## V5–V7 pipelines

```bash
./run.sh v5 labels --split train && ./run.sh v5 labels --split val   # POPF on every candidate set (10 s each)
./run.sh v5 train                                                     # -> results/v5/v5_seed0
./run.sh v5 seedtest --split stress --cap 30                          # GNN order vs TRPG-bound order
./run.sh v5 v5a-labels --split train | v5a-train | v5a-seedtest --split stress

./run.sh v6 labels --split train_large && ./run.sh v6 labels --split val_large   # test-sized kitchens
./run.sh v6 train && ./run.sh v6 tune --split val_large                          # kappa on validation
./run.sh v6 testA --split stress      # seeds only (A), triage queue (testB), capacity choice (testC)

./run.sh v7 train && ./run.sh v7 test --split stress
```

## Experiments (`./run.sh exp NAME`, outputs in `results/diagnostics/`)

| NAME | Question | Result |
| --- | --- | --- |
| `oracle` | POPF on the certificate's objects | 50 / 60 → selection, not the planner, is the bottleneck |
| `first_selection` | first set: relaxed-plan objects vs V1's expansion set | 17 vs 13 |
| `agent_seeds` / `work_seeds` | earliest-time vs work-optimal per-agent seeds | 30 vs 41 → V3, V4 |
| `guided_seeds --tol 0.25` | score-guided relaxed-plan seeds | 27–33, no gain |
| `cp_select --mode min_resources` | idea 1: CP-SAT scheduling relaxation picks objects (needs `ortools`) | 13 → dropped |
| `cert_scorer train` / `test` | idea 2: GNN trained on certificate membership | 12 → dropped |
| `v5_ranked_seeds` | V5a GNN seeds by rank on V4.2's 12 misses | 0 / 12 |
| `capacity_oracle` | trying every extra machine / agent in the improvement phase | ~1 % headroom |
