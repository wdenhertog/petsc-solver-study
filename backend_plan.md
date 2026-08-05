# Python Backend Plan (condensed)

Goal: Python problems emit the same JSON shape as C++, so `rank_results.py`/
`plot_results.py`/`dashboard.py` work across backends with zero logic
changes beyond explicit `backend`/`assembly_mode` columns.

Status legend: `[x]` done · `[ ]` not started · `[~]` in progress

---

## Done

*(Reconciled against actual uploaded files — `benchmark_result.cpp`,
`registry.py`, `runner.py`, `poisson.py`, `run_benchmarks.py` — not just
conversation memory.)*

- **[!] Immediate fix needed**: `run_benchmarks.py`'s `main()` currently
  hardcodes `s["backend"] = "python"` for *every* spec, unconditionally.
  Running this against `default.yaml` (has both `poisson` and `bratu`)
  will route Bratu through the Python runner and fail — Bratu isn't
  registered in `python_backend` yet. Revert to `"cpp"` before running
  any real sweep; only flip to `"python"` for isolated single-problem
  configs like `poisson_python_smoke.yaml`.

- **[x] Step 0 — harness dispatch.** `build_command()` added, `run()`
  rewired to call it, `cwd=REPO_ROOT` passed to `subprocess.run`.
  Pushed to GitHub.
- **[x] `python_backend/` package scaffolded**: `__init__.py`,
  `problem.py` (`Problem` ABC, `ProblemKind` enum), `benchmark_result.py`
  (`BenchmarkResult`, `fill_provenance`, `fill_solve_results_ksp`,
  `fill_memory_usage`), `runner.py`, `problems/__init__.py`.
- **[x] `petsc4py.init(sys.argv)`** placed at the top of `runner.py`,
  before any `from petsc4py import PETSc` — required or the options
  database comes up empty and every `-flag` silently vanishes.
- **[x] Poisson manual assembly runs correctly** (`problems/poisson.py`):
  mesh/forcing/BC match `poisson.cpp`, assembles via
  `assemble_vector(L_compiled)` (not `create_vector` — signature changed
  in dolfinx 0.10+), solves via `ksp.setFromOptions()` (no hardcoded
  `setType`), fixed a segfault caused by an unreferenced `fem.Function`
  temporary (keep it on `self`, e.g. `self.u_sol`, or its backing memory
  gets GC'd before `ksp.solve()` touches it).
  Verified directly via:
  `pixi run mpiexec -n 1 python3 -m python_backend.runner -problem poisson -n 64 -ksp_type cg -pc_type jacobi`

- **[x] C++ boolean output fixed** — `to_json()` now emits `true`/`false`
  for `success`/`git_dirty` instead of raw `0`/`1`, matching Python's
  native `bool` serialization.

## Decided, not yet implemented

- **PETSc version divergence is permanent, not a bug to fix.** dolfinx
  0.11 needs a newer PETSc than the wice module (`3.23.5-foss-2024a`).
  Don't pin `petsc`/`petsc4py` in `pixi.toml` — let conda-forge resolve
  whatever `fenics-dolfinx` needs. C++ stays on 3.23.5 always. Record
  both resolved versions in a short `docs/backend_petsc_versions.md`
  once `pixi install` settles. Cross-validation tolerance: compare
  `||x||_2` and `converged_reason` *category*, not exact iteration
  counts — version/rounding differences can shift iteration count by a
  few without indicating a real bug.
- **Two orthogonal axes**: `backend` (`cpp`|`python`) and
  `assembly_mode` (`manual`|`ufl_highlevel`, python-only — cpp rows
  always stamp `"manual"`). `manual` = hand-assembled, directly
  comparable to C++. `ufl_highlevel` = idiomatic
  `dolfinx.fem.petsc.LinearProblem`/`NonlinearProblem`, answers "best
  solver for this weak form" rather than "matches C++."
- **Problem dispatch is file-path-driven, not a static registry.**
  YAML points at a `.py` file (`problem_file:`); no central dict to
  edit when adding a problem. Convention: every problem file ends with
  `PROBLEM_CLASS = YourClass`; internal imports must be **absolute**
  (`from python_backend.problem import Problem`), not relative — dynamic
  loading via `importlib.util.spec_from_file_location` breaks relative
  imports.
- **`LinearProblem`/`NonlinearProblem` need their options prefix
  cleared**, or global `-ksp_type`/`-pc_type` sweep flags silently don't
  reach them (dolfinx 0.10+ requires a `petsc_options_prefix` and
  applies it before `setFromOptions()`):
  ```python
  problem = LinearProblem(a, L, bcs=bcs, u=u, petsc_options_prefix="benchmark_")
  problem.solver.setOptionsPrefix("")
  problem.solver.setFromOptions()
  ```
  **Must verify with `-ksp_view`** before trusting any `ufl_highlevel`
  result — same "silently ignored, no exception" risk class as the
  missing-`-problem`-flag bug.

---

## TODO, in order

### 1. Rewire Poisson to the file-path/two-axis design
- [ ] Rename `python_backend/registry.py` → `loader.py`:
  `load_problem_class(problem_file)` (importlib-based, requires
  `PROBLEM_CLASS`), `make_problem(problem_file)`.
- [ ] Rename `problems/poisson.py` → `problems/poisson_manual.py`, add
  `PROBLEM_CLASS = PoissonProblemManual`, switch to absolute imports.
- [ ] `runner.py`: parse `-problem_file` + `-assembly_mode`
  (`opts.getString(..., "manual")`), keep `-problem` only for a
  name-consistency check against `problem.name`, stamp
  `result.assembly_mode`.
- [ ] Add `assembly_mode: str = "manual"` to `BenchmarkResult`; stamp
  `"manual"` in C++'s `to_json()` literal too.
- [ ] Add `assembly_mode` (and `backend`, if not already) to
  `NON_INSTANCE_COLS` in `rank_results.py` + dashboard's exclude set —
  otherwise `infer_group_cols`'s fallback silently treats them as sweep
  dimensions.

### 2. Cross-validate Poisson manual (C++ vs Python)
- [ ] Make `configs/benchmarks/poisson_python_smoke.yaml` (Poisson only,
    `n: [64]`, one `ksp`/`pc` combo).
- [ ] Run through `run_benchmarks.py` (temporarily hardcode
    `spec["backend"] = "python"` in `main()`, revert after).
- [ ] Compare `||x||_2`, `iterations` (order-of-magnitude), `converged_reason`
    against equivalent C++ run at same `n`/`nprocs`. Note findings in
    `docs/python_poisson_validation.md`.

### 3. Poisson UFL-highlevel variant
- [ ] `problems/poisson_ufl.py` using `LinearProblem` + the prefix-clear
    fix above, verified via `-ksp_view`.
- [ ] Decide `setup_time`/`solve_time` split (may not be cleanly
    separable — if not, put everything in `solve_time`, note it).

### 4. Fold into real sweep config
- [ ] `configs/benchmarks/default.yaml`: add `backends: [cpp, python]`
    and `python_variants: [{assembly_mode: manual, problem_file: ...},
    {assembly_mode: ufl_highlevel, problem_file: ...}]` to the `poisson`
    entry only (not `bratu` — not implemented yet).
- [ ] `build_run_specs()`: expand `python_variants` when
    `backend == "python"`.
- [ ] `build_command()`: python branch passes `-problem`,
    `-problem_file`, `-assembly_mode`.
- [ ] Remove the hardcoded `spec["backend"] = "cpp"` from `main()`.

### 5. Bratu (manual + ufl_highlevel)
Same pattern as Poisson: `bratu_manual.py` (mirrors `bratu.cpp`,
`SNES` not `KSP`), `bratu_ufl.py` (`NonlinearProblem`), cross-validate
before trusting sweeps.

### 6. Backend/mode comparison tooling (dashboard)
- [ ] `compare_backends(df, problem, join_cols=SOLVER_FLAG_COLS + ["n","nprocs"])`
    in `plot_results.py`: inner-join `backend=="cpp"` vs `"python"` rows,
    diff `solve_time`/`iterations`, flag `converged_reason` mismatches.
- [ ] Dashboard tab reusing it; CLI flag on `rank_results.py`
    (`--compare-backends`) writing `<run>_backend_diff.csv`.

---

## Deferred: timestepping (heat → wave → dressing model)

Not started. Key decisions already made, kept brief:

- New `ProblemKind::Transient` in C++ (`assemble_transient(TS, Vec&)`,
  wraps `TSSolve`); Python mirrors via `PETSc.TS`.
- **Heat first** (linear, one KSP/step) — simplest possible TS problem,
  isolates "does TS wiring work" before nonlinearity. Analytic
  reference: decaying sine mode,
  `u(x,y,t) = sin(πx)sin(πy)·exp(-α·2π²·t)`.
- **Wave second** — reduced to a first-order system (`v=u_t`, `dof=2`
  per node), *not* PETSc's native `TSSetI2Function` — matches the
  Poisson/heat dof-per-node pattern and has more reference material to
  check against. Analytic reference: standing wave,
  `u(x,y,t) = cos(c·π√2·t)·sin(πx)sin(πy)`; also check discrete energy
  conservation (`½∫(v² + c²|∇u|²)dx` ≈ const) — a more sensitive
  instability detector than `converged_reason` alone.
- New schema fields (additive): `n_timesteps`, `n_ksp_iterations_total`,
  `n_snes_iterations_total`, `l2_error_vs_exact`, `energy_drift`
  (wave-only). Add to `NON_INSTANCE_COLS`.
- Dressing model (nonlinear transient, already in repo as
  `dressing_model.py`) waits until heat+wave are fully validated in
  both backends — reuses the same `ProblemKind::Transient` plumbing but
  adds SNES-inside-TS + Robin BC on top, which heat/wave deliberately
  avoid debugging simultaneously. Move it into
  `python_backend/problems/dressing.py` at that point, not before.

Sequencing once started: C++ heat → analytic check → Python heat →
cross-validate (analytic + C++) → C++ wave → analytic + energy check →
Python wave → cross-validate → fold into `default.yaml` → dressing
model last.