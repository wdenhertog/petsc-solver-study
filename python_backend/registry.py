from typing import Callable

from .problem import Problem, ProblemKind


class StubProblem(Problem):
    """Exists only to validate the Step 0 harness (dispatch, timeout,
    error-capture paths) before any real physics is implemented.
    Remove once Poisson/Bratu land, or keep — it's cheap and harmless."""

    @property
    def name(self) -> str:
        return "stub"

    @property
    def kind(self) -> ProblemKind:
        return ProblemKind.LINEAR

    def assemble_linear(self):
        # Returns fake but well-typed data so runner.py's timing/JSON
        # path can be exercised without PETSc involved yet.
        return {"dofs": 100, "iterations": 5, "residual_norm": 1e-10, "success": True}


ProblemRegistry = dict[str, Callable[[], Problem]]


def make_registry() -> ProblemRegistry:
    registry: ProblemRegistry = {}
    registry["stub"] = StubProblem
    # registry["poisson"] = PoissonProblem   # Step 1
    # registry["bratu"] = BratuProblem       # Step 3
    # registry["heat"] = HeatProblem         # Phase E
    # registry["wave"] = WaveProblem         # Phase E
    return registry


def make_problem(name: str) -> Problem:
    registry = make_registry()
    if name not in registry:
        available = ", ".join(sorted(registry.keys()))
        raise ValueError(f"Unknown problem '{name}'. Available: {available}")
    return registry[name]()
