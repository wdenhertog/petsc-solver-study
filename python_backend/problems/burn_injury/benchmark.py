"""Benchmark adapter for the burn-injury model (planar 2D and curved 2D).

Frozen meshes are *loaded*, never generated (see scripts/generate_burn_meshes.py).
All PETSc configuration comes from the options database: -ts_*, -snes_*,
-ksp_*, -pc_*, -fieldsplit_*. Defaults are applied only for options the user
did not pass, so sweeps are never silently overridden.

Adapter options
---------------
-burn_case            planar | curved                (default planar)
-burn_level           mesh level L<k>                (default 0)
-burn_mesh_dir        mesh directory; falls back to $BURN_MESH_DIR,
                      $VSC_SCRATCH/meshes, then <repo>/meshes/burn
-burn_T_days          final time in days             (default 20)
-burn_dt_days         >0: fixed step (days), TS adaptivity off;
                      0: adaptive stepping           (default 0)
-burn_dt0_days        initial step for adaptive stepping (days);
                      default: production value (0.05)
-burn_split           none | scalar_mech | fields | chem_mech (default none)
                      Builds index sets for -pc_type fieldsplit.
                      Splits are named: scalar_mech -> scalar, mech;
                      fields -> N, M, rho, c, v, eps
                      (so e.g. -fieldsplit_scalar_pc_type gamg).
                      chem_mech uses a DM field decomposition rebuilt on
                      the vinewtonrsls inactive set (chem=N,M,rho,c;
                      mech=v,eps).
-burn_bounds          true|false: enforce non-negativity with a VI SNES
                      (default true; needs -snes_type vi*)
"""

import os
import time
from pathlib import Path

import dolfinx.fem.petsc  # noqa: F401  (registers fem.petsc)
import numpy as np
import ufl
from dolfinx import fem
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, set_bc
from mpi4py import MPI
from petsc4py import PETSc

from ...benchmark_result import get_ts_converged_reason_str
from ...problem import Problem, ProblemKind
from .meshes.io import load_surface_mesh
from .model import BurnProblem2D, BurnProblemSurface
from .parameters import (
    get_default_char_scales,
    get_default_physical_params,
    get_standard_adaptive_params,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
SCALAR_FIELDS = ("N", "M", "rho", "c")
ALL_FIELDS = SCALAR_FIELDS + ("v", "eps")
SPLITS = {
    "scalar_mech": {"scalar": (0, 1, 2, 3), "mech": (4, 5)},
    "fields": {name: (i,) for i, name in enumerate(ALL_FIELDS)},
    "chem_mech": {"chem": (0, 1, 2, 3), "mech": (4, 5)},
}


def resolve_mesh_dir(opts: PETSc.Options) -> Path:
    explicit = opts.getString("burn_mesh_dir", "") or os.environ.get(
        "BURN_MESH_DIR", ""
    )
    if explicit:
        return Path(explicit)
    scratch = os.environ.get("VSC_SCRATCH")
    if scratch and (Path(scratch) / "meshes").is_dir():
        return Path(scratch) / "meshes"
    return REPO_ROOT / "meshes" / "burn"


def build_parameters(case: str):
    """Same parameter set as the production 2D / curved example scripts."""
    p = get_default_physical_params()
    p = p.updated(rho_eq=p.rho_eq * 0.8, nu=0.46, delta_M=p.delta_M * 0.5)
    scales = get_default_char_scales(p)
    if case == "curved":
        p = p.updated(
            k_sub=20.0 * (scales.sigma_c / scales.L_c**2),
            c_sub=10.0 * (p.mu_1 / scales.L_c**2),
            beta_h=10.0 * (scales.sigma_c / (scales.rho_c * scales.L_c)),
        )
    return p, scales


class TimedTSContext:
    """TS residual/Jacobian callbacks (same math as the production solver) with timers."""

    def __init__(self, problem: BurnProblem2D):
        self.problem = problem
        self.t_residual = 0.0
        self.t_jacobian = 0.0
        self.n_residual = 0
        self.n_jacobian = 0

    def _load_state(self, U, U_dot):
        U.copy(self.problem.U.x.petsc_vec)
        U_dot.copy(self.problem.U_dot.x.petsc_vec)
        self.problem.U.x.scatter_forward()
        self.problem.U_dot.x.scatter_forward()

    def ifunction(self, ts, t, U, U_dot, F):
        t0 = MPI.Wtime()
        self._load_state(U, U_dot)
        self.problem.update_time_dependent_terms(t)
        with F.localForm() as loc:
            loc.set(0.0)
        assemble_vector(F, self.problem.F_compiled)
        F.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        set_bc(F, self.problem.bcs, x0=U, alpha=-1.0)
        self.t_residual += MPI.Wtime() - t0
        self.n_residual += 1

    def ijacobian(self, ts, t, U, U_dot, shift, J, P):
        t0 = MPI.Wtime()
        self._load_state(U, U_dot)
        self.problem.update_time_dependent_terms(t)
        self.problem.alpha_const.value = shift
        J.zeroEntries()
        assemble_matrix(J, self.problem.J_compiled, bcs=self.problem.bcs)
        J.assemble()
        self.t_jacobian += MPI.Wtime() - t0
        self.n_jacobian += 1


def make_field_is(problem: BurnProblem2D, groups: dict[str, tuple[int, ...]]):
    """Global-numbered index sets of owned DOFs for each named group of fields."""
    imap = problem.V.dofmap.index_map
    n_local = imap.size_local
    offset = imap.local_range[0]
    out = []
    for name, comps in groups.items():
        dofs = np.unique(
            np.concatenate([problem.V.sub(i).dofmap.list.flatten() for i in comps])
        )
        owned = dofs[dofs < n_local].astype(PETSc.IntType) + PETSc.IntType(offset)
        out.append((name, PETSc.IS().createGeneral(owned, comm=problem.comm)))
    return out


def make_owned_global_dofs(
    problem: BurnProblem2D, fields: tuple[int, ...]
) -> np.ndarray:
    """Return owned mixed-space DOFs for selected fields in global numbering."""
    index_map = problem.V.dofmap.index_map
    n_owned = index_map.size_local
    start = index_map.local_range[0]
    local = np.unique(
        np.concatenate([problem.V.sub(i).dofmap.list.flatten() for i in fields])
    )
    return local[local < n_owned].astype(PETSc.IntType) + PETSc.IntType(start)


def configure_vi_fieldsplit(
    problem: BurnProblem2D, snes: PETSc.SNES, ksp: PETSc.KSP
) -> PETSc.DMShell:
    """Install a DM field decomposition that follows VINEWTONRSLS reduction."""
    comm = problem.comm
    chem_dofs = make_owned_global_dofs(
        problem, (problem.idx_N, problem.idx_M, problem.idx_rho, problem.idx_c)
    )
    mech_dofs = make_owned_global_dofs(problem, (problem.idx_v, problem.idx_eps))

    def create_field_decomposition(dm):
        inactive = snes.getVIInactiveSet()
        if inactive is None or inactive.getSize() == 0:
            chem, mech = chem_dofs, mech_dofs
        else:
            inactive_dofs = np.asarray(inactive.getIndices(), dtype=PETSc.IntType)
            reduced_offset = comm.exscan(inactive_dofs.size)
            # Map each rank's inactive entries into PETSc's rank-concatenated reduced numbering.
            reduced_dofs = np.arange(inactive_dofs.size, dtype=PETSc.IntType) + (
                reduced_offset or 0
            )
            in_chem = np.isin(inactive_dofs, chem_dofs)
            in_mech = np.isin(inactive_dofs, mech_dofs)
            if not np.all(in_chem | in_mech) or np.any(in_chem & in_mech):
                raise RuntimeError(
                    "VINEWTONRSLS inactive set does not map cleanly to the burn chem/mech fields"
                )
            chem, mech = reduced_dofs[in_chem], reduced_dofs[in_mech]

        is_chem = PETSc.IS().createGeneral(chem, comm=comm)
        is_mech = PETSc.IS().createGeneral(mech, comm=comm)
        return ["chem", "mech"], [is_chem, is_mech], None

    dm = PETSc.DMShell().create(comm=comm)
    dm.setCreateFieldDecomposition(create_field_decomposition)
    ksp.setDM(dm)
    ksp.setDMActive(PETSc.KSP.DMActive.ALL, False)
    return dm


def final_functionals(problem: BurnProblem2D) -> dict[str, float]:
    """Partition-independent integrals/L2 norms of the final state, for accuracy comparison."""
    parts = ufl.split(problem.U)
    dx = ufl.dx(domain=problem.domain)
    forms = {}
    for i, name in enumerate(SCALAR_FIELDS):
        forms[f"int_{name}"] = fem.form(parts[i] * dx)
        forms[f"l2sq_{name}"] = fem.form(parts[i] ** 2 * dx)
    forms["l2sq_v"] = fem.form(ufl.inner(parts[4], parts[4]) * dx)
    forms["l2sq_eps"] = fem.form(ufl.inner(parts[5], parts[5]) * dx)
    forms["area"] = fem.form(1.0 * dx)
    out = {
        key: float(problem.comm.allreduce(fem.assemble_scalar(form), op=MPI.SUM))
        for key, form in forms.items()
    }
    # Global minimum of each non-negative field: shows whether the VI bounds ever bind.
    n_local = problem.V.dofmap.index_map.size_local
    for i, name in enumerate(SCALAR_FIELDS):
        dofs = problem.V.sub(i).dofmap.list.flatten()
        dofs = dofs[dofs < n_local]
        local_min = float(problem.U.x.array[dofs].min()) if dofs.size else np.inf
        out[f"min_{name}"] = float(problem.comm.allreduce(local_min, op=MPI.MIN))
    return out


def event_times() -> dict[str, float]:
    out = {}
    for name in (
        "SNESSolve",
        "SNESJacobianEval",
        "SNESFunctionEval",
        "KSPSetUp",
        "PCSetUp",
        "KSPSolve",
        "PCApply",
        "MatLUFactorNum",
    ):
        try:
            info = PETSc.Log.Event(name).getPerfInfo()
            out[name] = float(info["time"])
        except Exception:
            pass
    return out


class BurnInjuryBenchProblem(Problem):
    @property
    def name(self) -> str:
        return "burn_injury"

    @property
    def kind(self) -> ProblemKind:
        return ProblemKind.TRANSIENT

    def run_transient_benchmark(self, opts, result) -> None:
        comm = MPI.COMM_WORLD
        PETSc.Log.begin()

        case = opts.getString("burn_case", "planar")
        if case not in ("planar", "curved"):
            raise ValueError(f"-burn_case must be planar or curved, got '{case}'")
        level = opts.getInt("burn_level", 0)
        T_days = opts.getReal("burn_T_days", 20.0)
        dt_days = opts.getReal("burn_dt_days", 0.0)
        dt0_days = opts.getReal("burn_dt0_days", 0.0)
        split = opts.getString("burn_split", "none")
        use_bounds = bool(opts.getBool("burn_bounds", True))
        if split != "none" and split not in SPLITS:
            raise ValueError(
                f"-burn_split must be none, {', '.join(SPLITS)}; got '{split}'"
            )
        vi_fieldsplit = split == "chem_mech"
        if vi_fieldsplit and not use_bounds:
            raise ValueError("-burn_split chem_mech requires -burn_bounds true")

        # Defaults only where the user gave nothing.
        defaults = {
            "snes_type": "vinewtonrsls",
            "snes_atol": 1e-8,
            "snes_rtol": 1e-6,
            "snes_stol": 1e-12,
            "snes_max_it": 20,
            "snes_linesearch_type": "bt",
            "ksp_type": "fgmres" if vi_fieldsplit else "preonly",
            "pc_type": "fieldsplit" if vi_fieldsplit else "lu",
            "ts_bdf_order": 2,
        }
        for key, val in defaults.items():
            if not opts.hasName(key):
                opts[key] = val
        if opts.getString("pc_type", "") == "lu" and not opts.hasName(
            "pc_factor_mat_solver_type"
        ):
            opts["pc_factor_mat_solver_type"] = "mumps"
        if vi_fieldsplit:
            for key, val in {
                "ksp_rtol": 1e-8,
                "ksp_atol": 1e-14,
                "ksp_max_it": 200,
                "pc_fieldsplit_type": "multiplicative",
                "fieldsplit_chem_ksp_type": "preonly",
                "fieldsplit_chem_pc_type": "lu",
                "fieldsplit_chem_pc_factor_mat_solver_type": "mumps",
                "fieldsplit_mech_ksp_type": "preonly",
                "fieldsplit_mech_pc_type": "lu",
                "fieldsplit_mech_pc_factor_mat_solver_type": "mumps",
            }.items():
                if not opts.hasName(key):
                    opts[key] = val

        # ---- mesh + problem setup -------------------------------------------------
        t_setup0 = MPI.Wtime()
        mesh_dir = resolve_mesh_dir(opts)
        mesh_path = mesh_dir / f"burn_{case}_L{level:02d}.xdmf"
        if not mesh_path.exists():
            raise FileNotFoundError(f"Frozen mesh not found: {mesh_path}")
        t0 = MPI.Wtime()
        tagged = load_surface_mesh(mesh_path, comm=comm)
        t_mesh = MPI.Wtime() - t0

        t0 = MPI.Wtime()
        params, scales = build_parameters(case)
        cls = BurnProblemSurface if case == "curved" else BurnProblem2D
        problem = cls(tagged, params, scales)
        problem.define_weak_forms()
        problem.define_boundary_conditions()
        problem.define_initial_conditions()
        t_forms = MPI.Wtime() - t0

        appctx = TimedTSContext(problem)
        F_vec = fem.petsc.create_vector(problem.V)
        J_mat = fem.petsc.create_matrix(problem.J_compiled)

        ts = PETSc.TS().create(comm)
        ts.setProblemType(PETSc.TS.ProblemType.NONLINEAR)
        ts.setEquationType(PETSc.TS.EquationType.IMPLICIT)
        ts.setType(PETSc.TS.Type.BDF)
        ts.setIFunction(appctx.ifunction, F_vec)
        ts.setIJacobian(appctx.ijacobian, J_mat)

        T_final = T_days / problem.t_c
        ts.setMaxTime(T_final)
        ts.setExactFinalTime(PETSc.TS.ExactFinalTime.MATCHSTEP)
        if dt_days > 0.0:
            ts.setTimeStep(dt_days / problem.t_c)
            if not opts.hasName("ts_adapt_type"):
                opts["ts_adapt_type"] = "none"
            ts.setMaxSNESFailures(0)
        else:
            ap = get_standard_adaptive_params(scales, dim=2)
            if not opts.hasName("ts_adapt_type"):
                opts["ts_adapt_type"] = "basic"
            for key, val in (
                ("ts_adapt_dt_min", ap["dt_min_value"]),
                ("ts_adapt_dt_max", ap["dt_max_value"]),
            ):
                if not opts.hasName(key):
                    opts[key] = val
            ts.setTolerances(
                atol=ap["ts_atol"], rtol=ap["ts_rtol"]
            )  # overridable via -ts_atol/-ts_rtol
            ts.setTimeStep(
                dt0_days / problem.t_c if dt0_days > 0.0 else ap["initial_dt"]
            )
            ts.setMaxSNESFailures(-1)

        ts.setFromOptions()
        snes = ts.getSNES()
        snes.setFromOptions()
        if use_bounds and snes.getType().startswith("vi"):
            lb = fem.Function(problem.V)
            ub = fem.Function(problem.V)
            lb.x.array[:] = -np.inf
            ub.x.array[:] = np.inf
            for idx in (problem.idx_N, problem.idx_M, problem.idx_rho, problem.idx_c):
                lb.x.array[problem.V.sub(idx).dofmap.list.flatten()] = 0.0
            snes.setVariableBounds(lb.x.petsc_vec, ub.x.petsc_vec)

        ksp = snes.getKSP()
        ksp.setFromOptions()
        pc = ksp.getPC()
        pc.setFromOptions()
        dm = None
        if split != "none":
            if pc.getType() != "fieldsplit":
                raise ValueError("-burn_split requires -pc_type fieldsplit")
            if vi_fieldsplit:
                if snes.getType() != "vinewtonrsls":
                    raise ValueError(
                        "-burn_split chem_mech currently requires -snes_type vinewtonrsls"
                    )
                if not PETSc.Options().hasName("pc_fieldsplit_type"):
                    pc.setFieldSplitType(PETSc.PC.CompositeType.MULTIPLICATIVE)
                dm = configure_vi_fieldsplit(problem, snes, ksp)
            else:
                pc.setFieldSplitIS(*make_field_is(problem, SPLITS[split]))

        U = problem.U.x.petsc_vec
        ts.setSolution(U)
        ts.setUp()
        t_setup = MPI.Wtime() - t_setup0

        # ---- solve ----------------------------------------------------------------
        error = ""
        t1 = MPI.Wtime()
        try:
            ts.solve(U)
        except PETSc.Error as e:
            error = f"PETSc error {e.ierr}: {e}"
        t_solve = MPI.Wtime() - t1

        reason = int(ts.getConvergedReason())
        result.dofs = U.getSize()
        result.setup_time = t_setup
        result.solve_time = t_solve
        result.n_timesteps = ts.getStepNumber()
        result.n_ksp_iterations_total = ts.getKSPIterations()
        result.iterations = result.n_ksp_iterations_total
        result.n_snes_iterations_total = ts.getSNESIterations()
        result.final_time = ts.getTime()
        result.converged_reason = reason
        result.converged_reason_string = error or get_ts_converged_reason_str(reason)
        result.success = (not error) and reason > 0
        result.residual = float(U.norm())

        details = {
            "case": case,
            "level": level,
            "mesh_file": mesh_path.name,
            "mesh_cells": tagged.num_cells,
            "mesh_vertices": tagged.num_vertices,
            "n_ranks": comm.size,
            "T_days": T_days,
            "T_reached_days": result.final_time * problem.t_c,
            "dt_fixed_days": dt_days,
            "split": split,
            "t_mesh_load": t_mesh,
            "t_forms": t_forms,
            "t_residual_assembly": appctx.t_residual,
            "t_jacobian_assembly": appctx.t_jacobian,
            "n_residual_evals": appctx.n_residual,
            "n_jacobian_evals": appctx.n_jacobian,
            "n_rejected_steps": ts.getStepRejections(),
            "n_snes_failures": ts.getSNESFailures(),
            "n_ksp_failures": ts.getKSPFailures()
            if hasattr(ts, "getKSPFailures")
            else None,
            "petsc_events": event_times(),
            "wall_clock_epoch": time.time(),
        }
        if not error:
            details["functionals"] = final_functionals(problem)
        else:
            details["error"] = error
        result.details = details

        ts.destroy()
        if dm is not None:
            dm.destroy()
