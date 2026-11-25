# baselines

| ID | File | Method |
| --- | --- | --- |
| B1 | `b1.py` | POPF on the full temporal task |
| B2 | `b2.py` | FLAX: original `FlaxPlanner` + `GNNSearchGuidance` (Steps 1–3), POPF on the reduced temporal task |
| B3 | `b3.py` | iFLAX: `IFlaxPlanner` (Algorithm 1, parallel Repair / Restart / Rollback) + `IFlaxGNNGuidance`, bilevel training |

Each file starts with what it does. Train: `python -m baselines.b2 train --seed 0` (B3 likewise), with a harness
imported for the paths (e.g. from `overcooked_test`). Independent of every other folder: `core/` is a private copy of the
PDDL toolkit, `Task` / grounding and planner calls; FLAX's per-domain rule files come from the harness
(`TIFLAX_FLAX_COMPLEMENTARY`, `TIFLAX_FLAX_RELAXATION`).

**Original code, unchanged** in `vendor/`: FLAX (sair-lab/flax @ `42a1030`, MIT, `LICENSE_FLAX`) and the BTP iFLAX
implementation. `adapter.py` replaces its three I/O points: Task → PDDLGym `State`; `TemporalPOPF` (FLAX's `Planner`
interface, POPF on the reduced temporal problem); temporal VAL on the full task. Scores are raw sigmoid outputs, as in
FLAX / iFLAX.
