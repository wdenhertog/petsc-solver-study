"""
python_backend/runner.py

CLI entrypoint for the Python/FEniCSx benchmark backend -- the Python
analogue of src/main.cpp. Mirrors its structure: resolve the problem
name from the PETSc options database, dispatch on ProblemKind, time
setup/solve, fill a BenchmarkResult, print one JSON line from rank 0.
"""

import json
import sys

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
)
from .problem import ProblemKind
from .registry import make_problem

VALID_ASSEMBLY_MODES = {"manual", "ufl_highlevel"}


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
        else:
            if comm.rank == 0:
                print(
                    f"assembly_mode '{assembly_mode}' is currently only implemented for problems 'poisson' and 'bratu'",
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
        raise NotImplementedError("Transient dispatch arrives in Phase E")

    fill_memory_usage(result)

    if comm.rank == 0:
        print(json.dumps(result.to_json_dict()))


if __name__ == "__main__":
    main()
