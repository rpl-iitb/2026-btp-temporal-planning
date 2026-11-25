# tools

Static Linux binaries in `bin/`, found by `algorithm/pddlkit/planners.py` (and the baselines' copy) via `TIFLAX_<NAME>`, the `PATH`, then `TIFLAX_TOOLS` (default: this folder). Checksums in `bin/SHA256SUMS`
(`sha256sum -c bin/SHA256SUMS` from this folder).

| Binary | Source | Used for |
| --- | --- | --- |
| `popf` | POPF Release 2, static i386 build (KCL ROSPlan) | every planner call of every method |
| `optic-clp` | OPTIC, CLP build (Dongbox/optic-clp-release) | anytime improvement in T-iFLAX's improvement phase |
| `Validate`, `libVAL.so` | VAL 4, KCL-Planning/VAL @ `3c7a1f3` | plan validation on the full problem |
| `Parser` | VAL's PDDL parser / type checker | checks of the generated PDDL |

Check the licences before redistributing the binaries publicly.

`memguard.py`: memory guard for unattended runs (`overcooked_test/run_kitchen_all.sh` starts one per driver). When available memory falls below a threshold it kills the whole process tree of that driver's largest stage (POPF included), so the stage fails cleanly and the driver moves on; nothing outside the driver's tree is touched.
