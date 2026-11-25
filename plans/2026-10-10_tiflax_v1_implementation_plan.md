# T-iFLAX V1: implementation plan

Status: being implemented (started 2026-10-10). Results go in `README.md` once measured.

## 0. What is reused, what is new

| Piece | Source | Reuse |
| --- | --- | --- |
| Domain spec, HL problem, Π\|S, PDDL writer, POPF/VAL wrappers | `pddl_generator/` | as is |
| Benchmark 1 (Overcooked-AI 1.1.0, certified feasible instances, simulator) | `overcooked_ai_v1/` | as is; new train/val configs only |
| Threshold expansion q ← γq, q_max = 0.81, γ = 0.9 | FLAX Step 1 / iFLAX Alg. 1 l.1–11 | ported |
| Relaxed-plan Repair, complementary rules | FLAX Steps 2–3 | ported (rules as JSON, same format) |
| Repair / Restart / Rollback, T1 = 0.6·T, first valid branch wins, training picks fewest evaluated states | iFLAX Alg. 1, §III-E | ported |
| Encode–process–decode GNN (3 steps, LayerNorm, sigmoid head) | FLAX `gnn/gnn.py` (MIT) | re-implemented compactly, K heads |
| Offline labels from full-problem plans | FLAX §IV-C | B2 |
| Bilevel pseudo-labels from plans found in pruned spaces | iFLAX Eq. 2 | B3, T-iFLAX |
| Temporal semantics, TRPG, LB/deadline proofs, temporal head + labels, action-grounded closure, temporal recovery | — | **new** |

## 1. Graph schema (`graph.py`)

Same object graph as FLAX/PLOI, generalised so it needs no per-domain vocabulary (built from the `DomainSpec`):

- **Nodes = objects.** Type one-hot (with ancestors); unary init facts; unary goal facts; unary numeric fluents divided by LB(Π).
- **Edges = predicates of arity ≥ 2.** An atom p(a1..ak) gives an edge for each ordered argument pair (i, j), with a feature for (p, i, j). Arity-3 facts such as `pot-has` therefore keep argument positions; FLAX asserts arity ≤ 2. Goal atoms get their own channel. Binary numeric fluents (`travel`) become edge weights divided by LB(Π).
- **Temporal node features** (domain-independent, from the temporal relaxed planning graph, all times / LB(Π)): earliest relaxed time the object becomes usable; whether it is in the temporal relaxed plan; whether it is a resource object (see §4); deadline / LB and relaxed deadline slack for goal objects; number of goal-relevant ground actions (log).
- **Globals**: LB(Π), #goals, tightest deadline / LB.

## 2. Two-output GNN and score calibration (`model.py`, `scorer.py`)

Heads: `object_score` (logit 0) and `temporal_score` (logit 1). Loss = BCE(object) + λ_t · BCE(temporal, soft target, masked).

Calibration: after training, per-head temperature scaling (one scalar per head, fit by NLL on a held-out validation split) so each head is a calibrated probability in [0, 1]. Then

`reward(o) = w_obj · p̂_obj(o) + w_temp · p̂_temp(o)` (default w = 1, 1; `combine: sum | max | noisy_or` configurable).

The threshold is in reward units: q_0 = q_max · (w_obj + w_temp), q ← γ q. Goal objects are always selected.

## 3. Temporal semantics (`semantics.py`, `grounding.py`, `analysis.py`)

- **Grounding** with static-fact filtering and type checks. Refuses domains with negative or quantified conditions, because then Π\|S soundness (a plan for Π\|S is a plan for Π) no longer follows.
- **TemporalState**: time, facts, numeric fluents, running actions (start, duration, end, invariants), pending timed literals. `start(a, t)` checks at-start conditions, applies start effects, registers invariants and the end event. `end(a)` checks at-end conditions and applies end effects. Invariants are checked at every happening in the open interval. Fact validity is an interval [added, deleted); a fact never deleted has `UNBOUNDED`, a sentinel, never a number.
- **Plan simulator**: executes a schedule happening by happening (ε = 0.001 as in VAL/POPF) and reports the first violated condition. Cross-checked against VAL in tests; VAL stays the authority.
- **Schedule analysis**: causal links (producer happening → consumer condition), resource hand-offs (a fact that one action restores and another consumes, e.g. `idle`, `free`, `pot-idle`), deadline edges from timed literals; CPM slack with deadlines; critical path; resource-wait time; parallelism = total action time / makespan.
- **TRPG** (temporal relaxed planning graph, delete-free, at-start and at-end effects at their own times, duration-weighted): earliest time of each fact. This gives
  - **LB(S)** = max over goals of the earliest achievement time in Π\|S. Admissible: it ignores resource contention.
  - **Infeasibility proof**: goal unreachable in Π\|S, or LB of a deadlined goal > its deadline.
  - **Temporal relaxed plan** (backchaining from goals through earliest achievers): used by Repair/Restart.

## 4. Action-grounded expansion (`closure.py`)

`COMP_T(S)`: fixpoint. For every selected object o, if no goal-relevant ground action that mentions o is executable in the relaxed Π\|S, add the parameters of the earliest such action (TRPG time) from Π. Goals must also be relaxed-reachable in Π\|S; otherwise the objects of the temporal relaxed plan of Π are added. This keeps action parameters (pose, tile, count objects) and complementary objects without hand-written rules, and it never sends a provably unsolvable Π\|S to the planner.

**Resource objects** are detected from the domain: a predicate deleted at start and re-added at end by one action (`idle`, `free`), or deleted by one action and restored by another (`pot-idle`), marks its argument type as a resource. Overcooked: cooks, tiles, pots, counters.

B2/B3 use FLAX's complementary rules (JSON) instead, to stay faithful.

## 5. Training labels (`labels.py`)

From the validated schedule π that the lower level returned (iFLAX pseudo-supervision):

- **object label**: 1 if o is a parameter of an action in π or a goal object (iFLAX Eq. 2), else 0.
- **temporal label** τ(o) = max(crit(o), σ(o)), with a mask.
  - crit(o) = max over actions a ∋ o of 1 − slack(a)/C_max. Slack comes from CPM over causal, resource and deadline edges.
  - σ(o): counterfactual, for at most k resource objects per task (config), with cap T_cf per call:
    - removal (o ∈ π): σ = clip((C(S∖o) − C(S)) / (κ · C(S)), 0, 1); proven infeasible or a missed deadline gives 1.
    - addition (o ∉ S, same type as a resource on the critical path): σ = clip((C(S) − C(S∪o)) / (κ · C(S)), 0, 1).
    - κ = 0.10, so a 10 % makespan change saturates.
  - A **timeout gives mask = 0** (unknown), never a label. Objects with no evidence: τ = 0 if not in π (consistent with object label), masked if counterfactuals were skipped and they are resources.

## 6. Algorithm and four recovery policies (`algorithm.py`)

Shared `Budget` (monotonic deadline; ledger: graph, gnn, reduce, plan, validate, recovery, refine). Every subprocess gets min(remaining, cap). Planner outcomes are kept apart:

- `solved` → VAL on the **full** Π. An invalid plan is logged as a reduction bug and never returned.
- `proved_infeasible` → only from TRPG proofs.
- `planner_gave_up` → POPF "unsolvable" (POPF is incomplete; this is not a proof).
- `timeout` → not evidence of insufficiency; triggers recovery as in iFLAX.

Expansion (T1 = 0.6 T): S_0 = COMP(goal objects); add o with reward(o) ≥ q; q ← γ q after an infeasible / gave-up attempt; a timeout triggers recovery. Before each planner call: TRPG proof check (skips the call if it fires).

Recovery (T2 = remaining budget, branches in parallel threads, cancellable, first valid plan wins; sequential mode available):

1. **Repair**: S ∪ objects of the temporal relaxed plan of Π, then COMP; plan.
2. **Restart**: COMP(S_0 ∪ relaxed-plan objects), then iterative expansion from there.
3. **Rollback**: S_{N−1}, then add objects one by one by reward.
4. **Temporal recovery**: diagnose (deadline-infeasible LB, timeout under tight deadlines, or, after success, critical-path resource waits / excess makespan). Then add the resource objects that most reduce LB(S) or the resource waits on the critical path (ranked by temporal score), COMP, and replan.

After the first valid plan, the leftover budget runs an **improvement phase** (revised 2026-10-10; the first version only added a resource and replanned with POPF, which never shortened a schedule because POPF returns its first plan and ignores the extra capacity). Each round, for the current best plan (makespan C, objects S):

- candidate object sets: S, and S plus one capacity candidate per resource type (types waiting on the critical path first, then types with fewer objects, so a second pot or cook is tried before floor tiles), each closed with COMP_T;
- on every set, in parallel: POPF on a copy whose goal deadlines are tightened to C − max(1, 5 % C) (the TRPG proof can reject it without a call), and OPTIC with its cost cap at C (metric = total-time, so cost = makespan);
- each result is VAL-checked on the original task, re-timed (`retime.py`: earliest start under the plan's causal, mutex and invariant orderings), and kept only if shorter.

Time to first plan and final makespan are both reported. Re-timing has given no gain so far: POPF already starts every action as early as its own orderings allow.

## 7. Methods

| Id | Scorer | Training | Closure | Recovery | Notes |
| --- | --- | --- | --- | --- | --- |
| B1 | — | — | — | — | POPF on full Π, same budget |
| B2 FLAX | FLAX GNN, 1 output | original `GNNSearchGuidance.train` (offline); labels from the shorter of POPF(full) and the certificate | FLAX complementary rules | original `FlaxPlanner` | `baselines/` |
| B3 iFLAX | FLAX GNN, 1 output | original `IFlaxGNNGuidance.train` (bilevel) | FLAX complementary rules | original `IFlaxPlanner` (3R, T1 = 0.6 T) | `baselines/` |
| **T-iFLAX V1** | 2 heads, calibrated, reward = sum | bilevel, object + temporal labels | action-grounded COMP_T | 3R + temporal recovery + TRPG proofs + improvement phase | `tiflax/` |
| T-iFLAX V2 | 3 outputs, reward = α·p_obj + (1−α)·p_temp | V1 + keep label for the gate | as V1 | as V1 | `tiflax_v2/` |
| T-iFLAX V2.1 | V2 network | retrained with object-score selection | as V1 | object score for the first plan, gate in improvement | `tiflax_v2/` (`TIFLAX21`) |
| **T-iFLAX V3** | V1 network | — | COMP_T | **per-agent minimal seeds first** (relaxed plan with one agent, ranked by TRPG bound, 4 in parallel), V1 as fallback; improvement phase adds capacity | `tiflax/` (`TIFLAX3`) |

| **T-iFLAX V4** | uniform | — | COMP_T | as V3, but seeds from **work-optimal relaxed plans** (`cost_relaxed_plan`: h_add supporters, durations as costs) + the best earliest-time seed | `tiflax/` (`TIFLAX4`) |

| **T-iFLAX V4.2** | uniform | — | COMP_T | V4 + 2 two-agent work-optimal seeds, 6 seeds in parallel, one round, cap 0.6 T | `tiflax/` (`TIFLAX42`) |

**Why V4.2 (2026-10-22).** V4's remaining misses needed a second cook (tight deadlines) or more than 12 s of POPF on the right set. One round with two-agent seeds and a longer cap: 48/60 (oracle 50).

**Why V4 (2026-10-22).** Ablation: V3 with uniform scores solves 28/60 vs 27 with learned scores (no benefit from learning). V3's misses had the right cook but far stations, because earliest-time relaxed plans ignore that one agent repeats a cycle per goal. Work-optimal relaxed plans pick compact station clusters: 41/60 as seeds alone, 42/60 for the full V4 at 30 s (clean re-run with a load gate).

**Why V3 (2026-10-11).** POPF on the certificate's objects (an oracle selection) solves 50/60 stress problems in about 1.3 s, while V1 solved 9 and every V1 failure was a timeout. Selection, not the planner, was the bottleneck. V3 starts from the smallest plausible set (one agent with its earliest stations) and adds capacity only in the improvement phase. Results: `RESULTS.md`.

B2/B3 run the original FLAX / iFLAX classes (`baselines/vendor`); `baselines/adapter.py` replaces only PDDLGym, Fast Downward and STRIPS VAL by temporal equivalents. Rules are in FLAX's JSON format (`baselines/rules/`): relaxation marks every tile free (ignore cook collisions); the relaxed task also drops deadlines (FLAX has no time). Complementary rules are applied to a fixpoint (iFLAX's closure definition); FLAX's arity-≤2 graph leaves arity-3 facts out of the GNN input only.

**V2 gate.** Third output α(o) ∈ (0, 1). Loss = BCE(p_obj) + BCE(p_temp) + BCE(reward, y_keep) + 0.1·(α − 0.5)² on objects without evidence. y_keep(o) = 1 if o is in the returned schedule or a goal object, or its counterfactual σ(o) ≥ 0.5 (≥ 5 % makespan change). The reward is in [0, 1], so FLAX's threshold q_max = 0.81 applies directly; no temperature scaling (the gate is trained on the thresholded quantity).

## 8. Evaluation protocol (`evaluate.py`)

- **Data.** Train: new small kitchens with distractors (`configs/dataset_train.yaml`, seed 101). Validation: same generator, seed 202 (calibration, model selection). Test: `stress` tiers t1–t6 (certified feasible). Sanity: `examples`, `random_small`.
- **Budgets.** 15 / 30 / 60 s. **Unlimited mode**: a long cap (default 1800 s, configurable) to measure planning time and makespan without the threshold; anything still running at the cap is reported as "not finished in 1800 s", never as infeasible.
- **Metrics.** Coverage (VAL-valid within budget), time to first valid plan, total time, weighted planning time (FLAX), makespan, Δ vs best known, objects kept (fraction), planner calls, states evaluated, budget ledger, recovery branch that won, proofs fired. Deadlines are hard, so every valid plan meets them.
- **Fairness.** Same planner, machine, budget and VAL check. Parallel recovery uses up to 4 cores; CPU time is reported too. Single-thread runs use `taskset`; results are paired by instance; ≥ 3 seeds before claims.

## 9. Build order (each step verified before the next)

1. grounding + TRPG + semantics + plan simulator; tests against VAL on example plans and certificates.
2. Π\|S + COMP_T; soundness test: random S ⊇ COMP_T(goals) → POPF → VAL on full Π.
3. Budget + planner interface + B1.
4. Non-learned skeleton (uniform scores) with all 4 recovery policies.
5. Graph + 2-head GNN + labels + bilevel training + calibration.
6. B2, B3.
7. Evaluation: budgets + unlimited mode.
