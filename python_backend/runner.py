"""
python_backend/runner.py

CLI entrypoint for the Python/FEniCSx benchmark backend -- the Python
analogue of src/main.cpp. Mirrors its structure: resolve the problem
name from the PETSc options database, dispatch on ProblemKind, time
setup/solve, fill a BenchmarkResult, print one JSON line from rank 0.
"""

import json
import sys
from dataclasses import dataclass

import petsc4py

# MUST run before any `from petsc4py import PETSc`,
# including in modules imported below (benchmark_result,
# problems/poisson, etc.) -- otherwise PETSc initializes
# with an empty options database and every -flag
# silently vanishes
petsc4py.init(sys.argv)

from mpi4py import MPI
from petsc4py import PETSc

from .benchmark_result import (
    BenchmarkResult,
    fill_memory_usage,
    fill_provenance,
    fill_solve_results_ksp,
    fill_solve_results_snes,
    get_ts_converged_reason_str,
)
from .problem import ProblemKind
from .registry import make_problem

VALID_ASSEMBLY_MODES = {"manual", "ufl_highlevel"}


@dataclass
class TransientTsCtx:
    mass: PETSc.Mat
    stiffness: PETSc.Mat
    work: PETSc.Vec
    imex_alpha: float

    def ifunction(self, ts, t, x, xdot, f):
        del ts, t
        self.mass.mult(xdot, f)
        self.stiffness.mult(x, self.work)
        f.axpy(self.imex_alpha, self.work)

    def ijacobian(self, ts, t, x, xdot, shift, A, B):
        del ts, t, x, xdot
        self.stiffness.copy(result=B)
        B.scale(self.imex_alpha)
        B.axpy(shift, self.mass, structure=PETSc.Mat.Structure.SUBSET_NONZERO_PATTERN)
        if A != B:
            B.copy(result=A)

    def rhsfunction(self, ts, t, x, g):
        del ts, t
        self.stiffness.mult(x, g)
        g.scale(-(1.0 - self.imex_alpha))

    def rhsjacobian(self, ts, t, x, A, B):
        del ts, t, x
        self.stiffness.copy(result=B)
        B.scale(-(1.0 - self.imex_alpha))
        if A != B:
            B.copy(result=A)


def solve_transient_ts(problem, result: BenchmarkResult, opts: PETSc.Options) -> None:
    n = opts.getInt("n", 64)
    alpha = float(opts.getReal("alpha", 1.0))
    dt = float(opts.getReal("ts_time_step", opts.getReal("ts_dt", 1.0e-3)))
    max_time = float(opts.getReal("ts_max_time", 0.1))
    max_steps = opts.getInt("ts_max_steps", 100000)
    imex_alpha = max(0.0, min(1.0, float(opts.getReal("ts_imex_alpha", 1.0))))

    mass, stiffness, x = problem.assemble_transient(n=n, alpha=alpha)

    ts = PETSc.TS().create(MPI.COMM_WORLD)
    ts.setProblemType(PETSc.TS.ProblemType.LINEAR)
    ts.setSolution(x)
    ts.setTime(0.0)
    ts.setTimeStep(dt)
    ts.setMaxTime(max_time)
    ts.setMaxSteps(max_steps)

    ctx = TransientTsCtx(
        mass=mass,
        stiffness=stiffness,
        work=x.duplicate(),
        imex_alpha=imex_alpha,
    )

    f = x.duplicate()
    J = stiffness.copy()
    ts.setIFunction(ctx.ifunction, f)
    ts.setIJacobian(ctx.ijacobian, J, J)

    g = None
    R = None
    if imex_alpha < 1.0 - 1.0e-14:
        g = x.duplicate()
        R = stiffness.copy()
        ts.setRHSFunction(ctx.rhsfunction, g)
        ts.setRHSJacobian(ctx.rhsjacobian, R, R)

    ts.setFromOptions()

    reason = PETSc.TS.ConvergedReason.CONVERGED_ITERATING
    try:
        t1 = MPI.Wtime()
        ts.setUp()
        t2 = MPI.Wtime()
        ts.solve(x)
        t3 = MPI.Wtime()

        result.setup_time = t2 - t1
        result.solve_time = t3 - t2
        result.dofs = x.getSize()
        result.iterations = ts.getKSPIterations()
        result.n_ksp_iterations_total = ts.getKSPIterations()
        result.n_snes_iterations_total = ts.getSNESIterations()
        result.n_timesteps = ts.getStepNumber()
        result.final_time = ts.getTime()
        reason = ts.getConvergedReason()
        result.converged_reason = int(reason)
        result.converged_reason_string = get_ts_converged_reason_str(reason)
        result.success = reason > 0
        result.residual = x.norm()
        l2_error = problem.transient_l2_error(x, result.final_time)
        if l2_error is not None:
            result.l2_error_vs_exact = float(l2_error)
    finally:
        if R is not None:
            R.destroy()
        if g is not None:
            g.destroy()
        J.destroy()
        f.destroy()
        ctx.work.destroy()
        ts.destroy()
        mass.destroy()
        stiffness.destroy()


def main() -> None:
    comm = MPI.COMM_WORLD
    opts = PETSc.Options()

    # Mirrors PetscOptionsGetString(nullptr, nullptr, "-problem", ...) in
    # main.cpp -- fetched through the PETSc options database (not a
    # manual sys.argv scan), so PETSc marks "-problem" as a recognized,
    # used option and doesn't warn about it as unused at PetscFinalize.
    problem_name = opts.getString("problem", "")
    if not problem_name:
        if comm.rank == 0:
            print("Missing required -problem flag", file=sys.stderr)
        sys.exit(1)

    assembly_mode = opts.getString("assembly_mode", "manual").strip().lower()
    if assembly_mode not in VALID_ASSEMBLY_MODES:
        if comm.rank == 0:
            allowed = ", ".join(sorted(VALID_ASSEMBLY_MODES))
            print(
                f"Unknown -assembly_mode '{assembly_mode}'. Allowed: {allowed}",
                file=sys.stderr,
            )
        sys.exit(1)
    if assembly_mode == "manual":
        if problem_name == "wound_dressing":
            if comm.rank == 0:
                print(
                    "Problem 'wound_dressing' uses the Python/UFL path; run with -assembly_mode ufl_highlevel.",
                    file=sys.stderr,
                )
            sys.exit(1)
        try:
            problem = make_problem(problem_name)
        except ValueError as e:
            if comm.rank == 0:
                print(str(e), file=sys.stderr)
            sys.exit(1)
    elif assembly_mode == "ufl_highlevel":
        if problem_name == "poisson":
            from .problems.poisson_ufl import PoissonProblemUFL

            problem = PoissonProblemUFL()
        elif problem_name == "bratu":
            from .problems.bratu_ufl import BratuProblemUFL

            problem = BratuProblemUFL()
        elif problem_name == "heat":
            from .problems.heat_ufl import HeatProblemUFL

            problem = HeatProblemUFL()
        elif problem_name == "wound_dressing":
            from .problems.wound_dressing.benchmark import WoundDressingBenchProblem

            problem = WoundDressingBenchProblem()
        else:
            if comm.rank == 0:
                print(
                    f"assembly_mode '{assembly_mode}' is currently only implemented for problems 'poisson', 'bratu', 'heat', and 'wound_dressing'",
                    file=sys.stderr,
                )
            sys.exit(1)

    result = BenchmarkResult(problem=problem.name, assembly_mode=assembly_mode)
    fill_provenance(result)

    if problem.kind == ProblemKind.LINEAR:
        n = opts.getInt("n", 64)
        if assembly_mode == "manual":
            A, b, x = problem.assemble_linear(n=n)

            ksp = PETSc.KSP().create(comm)
            ksp.setOperators(A)
            ksp.setFromOptions()  # options database only -- no hardcoded setType()

            t1 = MPI.Wtime()
            ksp.setUp()
            t2 = MPI.Wtime()
            ksp.solve(b, x)
            t3 = MPI.Wtime()

            result.dofs = problem.dofs()
            result.setup_time = t2 - t1
            result.solve_time = t3 - t2
        else:
            ksp, dofs, setup_time, solve_time = problem.solve_linear(n=n)
            result.dofs = dofs
            result.setup_time = setup_time
            result.solve_time = solve_time

        fill_solve_results_ksp(ksp, result)

    elif problem.kind == ProblemKind.NONLINEAR:
        n = opts.getInt("n", 64)
        lam = opts.getReal("lambda", 6.0)
        if assembly_mode == "manual":
            snes, x = problem.assemble_nonlinear(n=n, lam=lam)
            t1 = MPI.Wtime()
            snes.setUp()
            t2 = MPI.Wtime()
            snes.solve(None, x)
            t3 = MPI.Wtime()

            result.dofs = problem.dofs()
            result.setup_time = t2 - t1
            result.solve_time = t3 - t2
        else:
            snes, dofs, setup_time, solve_time = problem.solve_nonlinear(n=n, lam=lam)
            result.dofs = dofs
            result.setup_time = setup_time
            result.solve_time = solve_time

        fill_solve_results_snes(snes, result)

    elif problem.kind == ProblemKind.TRANSIENT:
        if problem_name == "wound_dressing":
            try:
                problem.run_transient_benchmark(opts, result)
            except Exception as e:
                if comm.rank == 0:
                    print(str(e), file=sys.stderr)
                sys.exit(1)
        else:
            try:
                solve_transient_ts(problem, result, opts)
            except ValueError as e:
                if comm.rank == 0:
                    print(str(e), file=sys.stderr)
                sys.exit(1)

    fill_memory_usage(result)

    if comm.rank == 0:
        print(json.dumps(result.to_json_dict()))


if __name__ == "__main__":
    main()
