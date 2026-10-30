# baselines

| ID | File | Method | Train | Model |
| --- | --- | --- | --- | --- |
| B1 | `b1.py` | POPF on the full temporal task | — | — |
| B2 | `b2.py` | FLAX: original `FlaxPlanner` + `GNNSearchGuidance` (Steps 1–3, FLAX rules in `rules/`) | `./run.sh train B2` (offline, 301 epochs, pos_weight 10; teacher = shorter of POPF on the full task and the certificate) | `results/b2/b2_seed0_*` |
| B3 | `b3.py` | iFLAX: `IFlaxPlanner` (Algorithm 1, parallel Repair / Restart / Rollback) + `IFlaxGNNGuidance` | `./run.sh train B3` (bilevel, 6 epochs, 15 s per task) | `results/b3/b3_seed0_*` |

Evaluate: `./run.sh B1` (or `B2`, `B3`, `./run.sh eval B1,B2,B3 30`). Train flags: `--seed --flax_epochs
--label_cap --epochs --budget --pos_weight --max_tasks`.

**Original code, unchanged** in `vendor/`: FLAX (sair-lab/flax @ `42a1030`, MIT, `LICENSE_FLAX`) and the BTP
iFLAX implementation. Files marked `# ADAPTER` at the top (`planning/__init__.py`, `pddlgym/__init__.py`,
`fd_tracked.py`, `guidance/__init__.py`) replace imports only.

`adapter.py` replaces the three I/O points of that code, and holds what B1–B3 share:
- **State**: Task → PDDLGym `State` (pddlgym's own structs); the goal carries the Task, so a reduced state
  becomes a temporal problem again.
- **Planner**: `TemporalPOPF` implements FLAX's `Planner` interface with POPF on the reduced *temporal*
  problem (same planner as B1 and T-iFLAX); `last_num_evaluated` = POPF's evaluated states (iFLAX's label choice).
- **Validation**: temporal VAL on the full task inside `TemporalPOPF`; `validate_strips_plan` returns that verdict.

Adaptations forced by the temporal setting: relaxed tasks also drop deadlines / timed literals; complementary
rules are applied until a fixpoint (iFLAX: COMP is a closure); arity-3 facts are left out of the GNN input
only (FLAX asserts arity ≤ 2); unseen types / predicates (e.g. `counter` in test tiers) give no feature
instead of a `KeyError` (`tolerant_state_to_graph`). Scores are raw sigmoid outputs, as in FLAX / iFLAX.
