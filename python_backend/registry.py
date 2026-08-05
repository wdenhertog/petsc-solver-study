from collections.abc import Callable

from .problem import Problem
from .problems.poisson import PoissonProblem

ProblemRegistry = dict[str, Callable[[], Problem]]


def make_registry() -> ProblemRegistry:
    registry: ProblemRegistry = {}
    registry["poisson"] = PoissonProblem
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
