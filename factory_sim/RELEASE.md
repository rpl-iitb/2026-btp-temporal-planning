# Release checklist (maintainers)

Done in the tree:

- reference lab described as *inspired by* course material on a university smart-factory teaching lab, with every
  number a placeholder; no institution, person or vendor names in configs, signs or docs;
- `__version__ = 2.0.0`: the default reference lab has no people (arm robots pick, carry and place; stations run without
  operators; no technician); suite `factorysim-2.0` is the default (`disturbed` added, families that need people left
  out); the 1.x lab is `lab_v1.yaml` and the 1.0 / 1.1 suites regenerate from it unchanged; every instance and
  `index.json` record `code_version` (version, commit, dirty flag); `factory-sim verify` re-checks stored ground truth;
- `pddl.PlanPolicy` runs multi-robot plans to the end (plan order kept on shared jobs and stations, releases in the
  plan's processing order), and the core asks agents that waited again when another agent's action at the same
  instant may have unblocked them (before, such runs ended as `stalled`);
- distinct baselines (fifo, edd, spt, cr, mwkr, nearest, random; `cpsat` open-loop in `bench`), pooled lower bounds under
  reconfiguration, CP-SAT relaxation bounds on test-split M instances, `best_known` per instance;
- fixed-size, bounded Gymnasium observation (812, with per-robot type features) and action space (105) for every instance; `action_masks()` for
  MaskablePPO (`examples/train_ppo.py`, extra `rl`);
- CI workflow (`.github/workflows/tests.yml`), `BENCHMARK.md`, `docker/Dockerfile` for the Gazebo twin;
- docs media under 2.5 MB; full-resolution GIFs in `docs/release/` (git-ignored) for the release page.

Still to decide / do by the maintainers:

1. **Name**: "FactorySim" is used by several projects (FactorySimPy on PyPI and others). Candidates with no clash found:
   ShopTwin, TrayBench, CellArena. Renaming means the package, CLI (`factory-sim`), Gym id and docs.
2. Create the repository (private first), push, check CI.
3. Release: attach `benchmarks/v2.0` (zip; optionally `v1.0`), `docs/release/*.gif` and the baseline result JSONL files as assets; tag `v2.0.0`.
4. Optional: publish the Docker image and the package on PyPI under the chosen name.
5. Related work to cite / compare: JobShopLab, FabricatioRL, joint-sim, FactorySimPy, AssemblyGrid (arXiv 2609.16075).
