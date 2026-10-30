# pddl_generator: HL representation → temporal PDDL (+ planner / VAL tools)

This package is benchmark-agnostic. Benchmarks 2 and 3 will use it unchanged.

| File | What it is |
| --- | --- |
| `domain_spec.py` | Loads a domain YAML: types, predicates, functions, durative actions (`extends:` for derived domains) |
| `problem.py` | The HL problem (objects, facts, fluents, timed literals, goals with deadlines, provenance), its validation, and `restrict(S)` = Π\|S; the `Benchmark` interface + `BENCHMARKS` registry |
| `hl_problem.schema.json` | JSON schema of the HL problem |
| `writer.py` | `write_domain(spec)`, `write_problem(spec, problem)`: PDDL 2.1 + timed initial literals; the round-trip check that reads generated PDDL back and compares it with the source |
| `planners.py` | Temporal plans (parsing, makespan, total action time); POPF and VAL wrappers (`run_temporal_planner`, `validate_plan`, `parse_check`) |
| `cli.py` | Commands: scenarios → PDDL (default), and `check`: PDDL → POPF → VAL → simulator |

## Run

```bash
./run.sh pddl                                          # every scenario of overcooked_ai
./run.sh pddl overcooked_ai/scenarios/stress           # one folder;  ./run.sh pddl path/to/one.yaml
./run.sh check overcooked_ai/generated_pddl/examples --timeout 60
```

`cli` writes into `<benchmark>/generated_pddl/`:

```
domain.pddl                      shared by every instance of the benchmark
<set>/index.json                 one row per instance (sizes, checks)
<set>/<id>/problem.pddl
<set>/<id>/hl_problem.json       source of truth for problem.pddl
<set>/<id>/manifest.json         sha256 of scenario, domain spec, simulator commit, code, outputs;
                                 check results; problem sizes; timings
```

Here `<set>` is the folder the scenario file is in, e.g. `examples` or a dataset name.

Checks on every instance; any failure makes the command exit non-zero:

1. HL typing;
2. JSON schema;
3. exact round trip of the PDDL;
4. VAL `Parser` with 0 errors and 0 warnings.

`cli check` adds the semantic test (POPF plan → VAL on the full problem → execution in the simulator) and writes `results/pddl_check/<run>/`.

## Mapping HL → PDDL

| HL | PDDL |
| --- | --- |
| spec types / constants / predicates / functions | `:types` / `:constants` / `:predicates` / `:functions` |
| durative action: conditions `at_start/over_all/at_end`, effects `at_start/at_end`, duration (number or function term) | `:durative-action` |
| `objects` | `:objects`, grouped by type |
| `init.facts`, `init.fluents` | `:init` atoms, `(= (f args) v)` |
| `timed_literals` | `(at t atom)` / `(at t (not atom))` |
| goal `(served o)` with `deadline: D` | goal + `(at D+0.5 (not (open o)))`, per the domain's `deadline_encoding` |
| `metric: makespan` | `(:metric minimize (total-time))` |

Output is sorted, so the same inputs always give byte-identical files.

## Adding a benchmark

1. Write `<bench>/domain.yaml`. It may start with `extends: ../overcooked_ai/domain.yaml`.
2. Subclass `Benchmark` (`domain_spec`, `load_scenario`, `build_problem`, `execute`) and add it to `BENCHMARKS` in `problem.py`.
3. Run `./run.sh pddl --benchmark <name>`.
