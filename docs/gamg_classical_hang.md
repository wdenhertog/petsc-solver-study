# `pc_gamg_type: classical` hangs under `nprocs > 1`

**Status:** Confirmed, unresolved. Excluded from the benchmark sweep via
`components.yaml` (`known_broken` field on the `gamg_classical` component). Do not remove that exclusion without
re-testing against the specific PETSc version in use at the time.

## Summary

`-pc_type gamg -pc_gamg_type classical` hangs indefinitely (no output, no crash, no error) whenever the benchmark is run
with more than one MPI rank.
`-pc_gamg_type agg` (the default GAMG coarsening strategy) does not exhibit this behavior under otherwise identical
conditions. Single-rank runs of
`classical` are unaffected and complete instantly.

## Environment

- PETSc version: `PETSc Release Version 3.23.5, Jul 30, 2025`
- Cluster: VSC tier-2 (wice), module `PETSc/3.23.5-foss-2024a`
- Also reproduced locally (WSL) against the same PETSc version
- Problem: `poisson`, `n=64`, `ksp_type=cg`

## How it was found

During a full benchmark sweep (`nprocs ∈ {1,2,4,8,16}`), SLURM array chunks at the same relative chunk index (index 1
and index 3) were stuck producing zero output after 20+ minutes, consistently across the `p2`, `p4`, `p8`,
`p16` job arrays — but *not* in the `p1` array. Because `p2/p4/p8/p16` share an identical (filtered) job ordering, the
same chunk index corresponds to the same underlying `(problem, mesh, solver)` spec across all of them, which is what
made the pattern visible rather than looking like scattered per-config slowness.

Inspecting the first job at those chunk indices showed:

```
poisson {'n': 64}  {'pc_type': 'gamg', 'pc_gamg_type': 'classical', 'ksp_type': 'cg'}
poisson {'n': 512} {'pc_type': 'gamg', 'pc_gamg_type': 'classical', ...}
```

## Local reproduction

```bash
# Hangs (no output, requires kill/timeout):
timeout 120 mpiexec -n 2 ./bin/benchmark -problem poisson -n 64 \
    -pc_type gamg -pc_gamg_type classical -ksp_type cg \
    -ksp_view -ksp_converged_reason -log_view

# Completes instantly:
mpiexec -n 1 ./bin/benchmark -problem poisson -n 64 \
    -pc_type gamg -pc_gamg_type classical -ksp_type cg

# Completes instantly (control — same everything, different coarsening):
mpiexec -n 2 ./bin/benchmark -problem poisson -n 64 \
    -pc_type gamg -pc_gamg_type agg -ksp_type cg
```

`-ksp_view`/`-log_view` output an empty log file when run under `timeout`, i.e. the process never reaches even the
post-setup diagnostic print — consistent with a hang during `PCSetUp` (specifically GAMG's classical coarsening phase),
not a slow-but-progressing solve.

## Why this rules out a bug in our own code

`main.cpp`'s solve/print logic is identical regardless of which
`pc_gamg_type` was requested — it has no branch on this option. Since `agg`
completes correctly under the exact same rank count, problem, and mesh size, and only `classical` hangs, the fault must
live inside PETSc's
`PCSetUp_GAMG` classical-coarsening code path itself, not in anything this repository controls (e.g. not the DMDA
RHS-indexing class of bug fixed earlier in `poisson.cpp` — that bug affected *all* multi-rank runs regardless of
preconditioner, which is a different, distinguishable signature from this one).

## Working theory (unconfirmed)

Classical (Ruge–Stüben-style) AMG coarsening has a less mature/less commonly exercised parallel implementation than
aggregation-based coarsening in PETSc's GAMG. A plausible cause is a communication deadlock in the parallel coarse-point
selection or interpolation-construction step of classical coarsening specifically. Not verified via `gdb` backtrace — if
this is revisited, attaching to both ranks mid-hang (`gdb -p <pid> -batch -ex "bt"`) and comparing where each rank is
blocked would be the next diagnostic step.

## Current mitigation

`gamg_classical` is defined in `components.yaml` with a `known_broken`
field; the sweep-expansion logic (`expand_components`) silently excludes any component with this field set, regardless
of which problem references it. No `pc_gamg_type: classical` runs are included in any sweep as of this note.

## Re-testing checklist (before removing the exclusion)

- [ ] Confirm PETSc version has changed from `3.23.5` (check release notes / changelog for GAMG-related fixes)
- [ ] Re-run the local reproduction commands above at `nprocs=2`
- [ ] If fixed, also test `nprocs ∈ {3, 4, 8}` before assuming it's fully resolved (this note's testing did not
  exhaustively check rank counts beyond 1 and 2)
- [ ] Only then remove `known_broken` from `components.yaml`

## Not done / out of scope for this note

- No minimal standalone PETSc reproducer has been built or reported upstream. Worth doing at some point given how clean
  and reproducible this finding is, but not required for this project's own progress.
