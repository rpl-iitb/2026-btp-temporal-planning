# tools

Static Linux binaries in `bin/`, paths configured in `configs/tools.yaml`, checksums in `bin/SHA256SUMS`
(`sha256sum -c bin/SHA256SUMS` from this folder).

| Binary | Source | Used for |
| --- | --- | --- |
| `popf` | POPF Release 2, static i386 build (KCL ROSPlan) | every planner call of every method |
| `optic-clp` | OPTIC, CLP build (Dongbox/optic-clp-release) | anytime improvement in T-iFLAX's improvement phase |
| `Validate`, `libVAL.so` | VAL 4, KCL-Planning/VAL @ `3c7a1f3` | plan validation on the full problem |
| `Parser` | VAL's PDDL parser / type checker | checks of the generated PDDL |

Check the licences before redistributing the binaries publicly.
