"""Benchmark adapter for the wound-dressing FEniCSx model."""

import time
from pathlib import Path

from mpi4py import MPI

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

        result.dofs = int(self._problem.p.x.array.shape[0])
        result.setup_time = 0.0
        result.solve_time = float(t2 - t1)
        result.success = True
        result.final_time = float(t_tilde_final)
        result.iterations = 0
        result.n_ksp_iterations_total = 0
        result.n_snes_iterations_total = 0
        result.residual = 0.0
        result.converged_reason = 0
        result.converged_reason_string = "custom_transient_run"
        result.n_timesteps = 0
