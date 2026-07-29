# python_backend/runner.py
import json
import sys
import time

from mpi4py import MPI

from .benchmark_result import BenchmarkResult, fill_provenance, fill_memory_usage
from .problem import ProblemKind
from .registry import make_problem


def _extract_problem_name(argv: list[str]) -> str:
    if "-problem" not in argv:
        print("Missing required -problem flag", file=sys.stderr)
        sys.exit(1)
    idx = argv.index("-problem")
    return argv[idx + 1]


def main() -> None:
    comm = MPI.COMM_WORLD
    problem_name = _extract_problem_name(sys.argv)

    try:
        problem = make_problem(problem_name)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)

    result = BenchmarkResult(problem=problem.name)
    fill_provenance(result)

    t0 = MPI.Wtime()
    if problem.kind == ProblemKind.LINEAR:
        # Step 0: stub returns a plain dict; Step 1 replaces this branch
        # with real assemble + KSP solve + fill_solve_results_ksp(...)
        data = problem.assemble_linear()
        t1 = MPI.Wtime()
        result.dofs = data["dofs"]
        result.iterations = data["iterations"]
        result.residual = data["residual_norm"]
        result.success = data["success"]
        result.setup_time = t1 - t0
        result.solve_time = 0.0
    elif problem.kind == ProblemKind.NONLINEAR:
        raise NotImplementedError("Nonlinear dispatch arrives in Step 3")
    elif problem.kind == ProblemKind.TRANSIENT:
        raise NotImplementedError("Transient dispatch arrives in Phase E")

    fill_memory_usage(result)

    if comm.rank == 0:
        print(json.dumps(result.to_json_dict()))


if __name__ == "__main__":
    main()
