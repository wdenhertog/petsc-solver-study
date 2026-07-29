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
    fill_provenance,
    fill_memory_usage,
    fill_solve_results_ksp,
)
from .problem import ProblemKind
from .registry import make_problem


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

    try:
        problem = make_problem(problem_name)
    except ValueError as e:
        if comm.rank == 0:
            print(str(e), file=sys.stderr)
        sys.exit(1)

    result = BenchmarkResult(problem=problem.name)
    fill_provenance(result)

    if problem.kind == ProblemKind.LINEAR:
        n = opts.getInt("n", 64)
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
        fill_solve_results_ksp(ksp, result)

    elif problem.kind == ProblemKind.NONLINEAR:
        raise NotImplementedError("Nonlinear dispatch arrives in Step 3")

    elif problem.kind == ProblemKind.TRANSIENT:
        raise NotImplementedError("Transient dispatch arrives in Phase E")

    fill_memory_usage(result)

    if comm.rank == 0:
        print(json.dumps(result.to_json_dict()))


if __name__ == "__main__":
    main()
