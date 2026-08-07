"""Benchmark adapter for the wound-dressing FEniCSx model."""

import time
from pathlib import Path

from mpi4py import MPI

from ...benchmark_result import get_ts_converged_reason_str
from ...problem import Problem, ProblemKind
from .model import WoundDressingProblem, WoundDressingSolver
from .parameters import (
    get_char_scales,
    get_default_adaptive_params,
    get_default_physical_params,
    get_nondim_params,
)


class WoundDressingBenchProblem(Problem):
    """Wrap the wound-dressing transient solver as a benchmark problem."""

    @property
    def name(self) -> str:
        return "wound_dressing"

    @property
    def kind(self) -> ProblemKind:
        return ProblemKind.TRANSIENT

    def __init__(self) -> None:
        self._params = get_default_physical_params()
        self._adaptive_params = get_default_adaptive_params()
        self._problem = None

    def run_transient_benchmark(self, opts, result) -> None:
        params = dict(self._params)
        params["Q0"] = opts.getReal("Q0", params["Q0"])
        params["Q_inf"] = opts.getReal("Q_inf", params["Q_inf"])
        params["tau"] = opts.getReal("tau", params["tau"])
        params["h_m"] = opts.getReal("h_m", params["h_m"])
        params["S_eq"] = opts.getReal("S_eq", params["S_eq"])
        params["wound_size"] = opts.getReal("wound_size", params["wound_size"])
        params["H"] = opts.getReal("H", params["H"])
        params["L"] = opts.getReal("L", params["L"])
        params["W"] = opts.getReal("W", params["W"])

        nx = opts.getInt("nx", 40)
        ny = opts.getInt("ny", 40)
        nz = opts.getInt("nz", 6)
        t_tilde_final = opts.getReal("T_tilde", 0.5)
        t_star = opts.getReal("t_star", 1.0)
        save_vtx = bool(opts.getBool("wd_save_vtx", False))

        scales = get_char_scales(params)
        nd = get_nondim_params(params, scales)
        self._problem = WoundDressingProblem(nd, nx=nx, ny=ny, nz=nz)

        output_path = None
        if save_vtx:
            out_dir = Path(__file__).resolve().parents[3] / "results" / "vtx"
            out_dir.mkdir(parents=True, exist_ok=True)
            output_path = out_dir / f"wound_dressing_{int(time.time() * 1000)}"
            output_path.mkdir(parents=True, exist_ok=True)

        t1 = MPI.Wtime()
        solver = WoundDressingSolver(
            self._problem,
            dict(self._adaptive_params),
            output_path,
            T_tilde=float(t_tilde_final),
            t_star=float(t_star),
            verb=0,
        )
        solver.run()
        t2 = MPI.Wtime()

        # Global DOF count -- `.x.array` is the *local* (per-rank) slice
        # and silently under-reports under MPI; `.x.petsc_vec.getSize()`
        # is the global PETSc Vec size, same convention as poisson.py and
        # solve_transient_ts() in runner.py. Verify this against a serial
        # run before trusting parallel numbers, per the project's usual
        # serial-vs-parallel MPI-correctness check.
        result.dofs = self._problem.p.x.petsc_vec.getSize()
        result.setup_time = 0.0
        result.solve_time = float(t2 - t1)

        # Real solver stats pulled off the TS after solve(), mirroring
        # solve_transient_ts() in runner.py -- previously hardcoded to 0,
        # which threw away the entire point of sweeping ts/ksp/pc types.
        result.n_timesteps = solver.n_timesteps
        result.n_ksp_iterations_total = solver.n_ksp_iterations_total
        result.n_snes_iterations_total = solver.n_snes_iterations_total
        result.iterations = solver.n_ksp_iterations_total
        result.final_time = solver.final_time
        result.converged_reason = solver.converged_reason
        result.converged_reason_string = get_ts_converged_reason_str(
            solver.converged_reason
        )
        result.success = solver.converged_reason > 0
        result.residual = self._problem.p.x.petsc_vec.norm()
