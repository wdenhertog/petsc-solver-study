# Burn-injury benchmark (planar and curved 2D)

Adapter: `python_backend/problems/burn_injury/benchmark.py` (problem `burn_injury`,
`-assembly_mode ufl_highlevel`). The model and mesh loaders are vendored in
`python_backend/problems/burn_injury/`; there is no dependency on `burn_injuries`.

## Workflow

1. Generate frozen meshes once, serially (needs gmsh):
   `pixi run python scripts/generate_burn_meshes.py --out $VSC_SCRATCH/meshes`
2. Runs only *load* `burn_{planar,curved}_L{level:02d}.xdmf` from `-burn_mesh_dir`,
   `$BURN_MESH_DIR`, `$VSC_SCRATCH/meshes`, or `meshes/burn` (in that order).
3. Run a campaign, e.g. `configs/benchmarks/burn_injury_screening.yaml`.

On Wice, copy/pull the repository to `$VSC_DATA/petsc-solver-study` and run:

```bash
bash scripts/submit_burn_injury.sh smoke
```

Inspect the smoke-task output and JSONL results before submitting the larger
screening campaign:

```bash
bash scripts/submit_burn_injury.sh screening
```

The submitter checks for the required frozen mesh files, dry-runs the campaign
for each requested rank count, submits one array per rank count, and submits a
result-merging job with an `afterany` dependency. It defaults to MPI sizes
1, 2, 4, 8, and 16 for screening (configurable with `NPROCS_LIST`), one
benchmark per array task, two concurrent tasks per array, a 4-hour task limit,
and a 3-hour per-run timeout. The smoke mode uses one rank, both L01 geometries,
and shorter time limits. Both modes assume the Wice batch account and paths
encoded in the scripts; edit the SBATCH account/email directives if needed.
The Python benchmark uses the Pixi environment's PETSc, so the worker
intentionally does not load a separate PETSc module.

## Options

See the module docstring of `benchmark.py` (`-burn_case`, `-burn_level`, `-burn_T_days`,
`-burn_dt_days`, `-burn_split`, `-burn_bounds`). All `-ts_*`, `-snes_*`, `-ksp_*`, `-pc_*`
flags come from the PETSc options database; the adapter only fills in defaults for flags
you did not pass (VI Newton, BDF2, LU/MUMPS).

## Recorded metrics

Standard fields plus `details` (JSON): setup breakdown (mesh load, forms), time in
residual/Jacobian assembly, PETSc event times (`PCSetUp`, `KSPSolve`, `MatLUFactorNum`, ...),
rejected steps, SNES failures, and `functionals` (integrals, squared L2 norms, field minima
of the final state). Functionals are partition-independent: compare them against the
LU/MUMPS run on the same mesh/parameters to check accuracy.

## Findings so far (small meshes, 1-2 ranks)

- Serial vs 2 ranks agree to about 1e-13 on the functionals.
- Unconstrained `newtonls` diverges at the first step even with LU: the non-negativity
  bounds are required (the lower bound on M is active). Use a VI SNES.
- `vinewtonssls` fails even with LU on this problem; use `vinewtonrsls`.
- `-pc_type fieldsplit` + `-burn_split` fails under `vinewtonrsls`: the reduced-space solver
  resets the PC when the active set changes, which drops user-supplied index sets
  ("Unhandled case, must have at least two fields, not 0"). Open problem; the
  `-burn_split` hook is only usable with bounds disabled, which does not converge.
- Jacobian assembly is a large share of the cost on small meshes (about 0.3-0.6 s per
  evaluation at 35k DOFs). Compare solvers on `PCSetUp` + `KSPSolve`, not just total time,
  and consider `-snes_lag_jacobian` / `-snes_lag_preconditioner`.
