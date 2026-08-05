import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from mpi4py import MPI
from petsc4py import PETSc

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class BenchmarkResult:
    problem: str = ""
    dofs: int = 0
    iterations: int = 0
    residual: float = 0.0
    setup_time: float = 0.0
    solve_time: float = 0.0
    peak_memory_bytes: float = 0.0
    total_memory_bytes: float = 0.0
    success: bool = False
    converged_reason: int = 0
    converged_reason_string: str = ""
    outer_iterations: int = 0
    git_sha: str = ""
    git_dirty: bool = False
    petsc_version: str = ""
    backend: str = "python"

    def to_json_dict(self) -> dict:
        return asdict(self)


def get_converged_reason_str(reason_int: int, is_nonlinear: bool = False) -> str:
    """
    Dynamically maps a petsc4py convergence integer to its string name.
    """
    reason_class = (
        PETSc.SNES.ConvergedReason if is_nonlinear else PETSc.KSP.ConvergedReason
    )

    # Create a reverse mapping dict: { 2: 'CONVERGED_RTOL', -3: 'DIVERGED_LINEAR_SOLVE', ... }
    # We ignore internal Python attributes that start with '_'
    mapping = {
        getattr(reason_class, attr): attr
        for attr in dir(reason_class)
        if not attr.startswith("_")
    }

    # Return the string, or fallback to the integer string if not found
    return mapping.get(reason_int, str(reason_int))


def fill_provenance(result: BenchmarkResult) -> None:
    """Runtime equivalent of the CMake compile-time git_version.h baking."""
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT
    ).stdout.strip()
    dirty_out = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True, cwd=REPO_ROOT
    ).stdout.strip()
    result.git_sha = sha or "unknown"
    result.git_dirty = bool(dirty_out)

    try:
        from petsc4py import PETSc

        major, minor, sub = PETSc.Sys.getVersion()[:3]
        result.petsc_version = f"{major}.{minor}.{sub}"
    except Exception:
        result.petsc_version = "unknown"


def fill_solve_results_ksp(ksp, result: BenchmarkResult) -> None:
    result.iterations = ksp.getIterationNumber()
    result.residual = ksp.getResidualNorm()
    reason = ksp.getConvergedReason()
    result.converged_reason = int(reason)
    result.converged_reason_string = get_converged_reason_str(
        reason, is_nonlinear=False
    )
    result.success = reason > 0


def fill_memory_usage(result: BenchmarkResult) -> None:
    """Mirrors fill_memory_usage() in benchmark_result.cpp. Uses process
    RSS via `resource` — note this measures a different quantity than
    PETSc's PetscMemoryGetMaximumUsage() on the C++ side; flag this
    explicitly in cross-validation notes rather than treating the two
    numbers as directly comparable until confirmed otherwise."""
    import resource

    local_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    comm = MPI.COMM_WORLD
    peak = comm.reduce(local_mem, op=MPI.MAX, root=0)
    total = comm.reduce(local_mem, op=MPI.SUM, root=0)
    if comm.rank == 0:
        result.peak_memory_bytes = float(peak)
        result.total_memory_bytes = float(total)
