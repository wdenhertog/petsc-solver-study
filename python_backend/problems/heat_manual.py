import math

from mpi4py import MPI
from petsc4py import PETSc

from ..problem import Problem, ProblemKind


class HeatProblemManual(Problem):
    def __init__(self):
        self._n = 64
        self._alpha = 1.0
        self._dofs = 0

    @property
    def name(self) -> str:
        return "heat"

    @property
    def kind(self) -> ProblemKind:
        return ProblemKind.TRANSIENT

    def dofs(self) -> int:
        return self._dofs

    def assemble_transient(self, n: int = 64, alpha: float = 1.0):
        self._n = n
        self._alpha = alpha
        ndofs = n * n
        h = 1.0 / (n - 1)

        mass = PETSc.Mat().createAIJ([ndofs, ndofs], nnz=5, comm=MPI.COMM_WORLD)
        stiffness = PETSc.Mat().createAIJ([ndofs, ndofs], nnz=5, comm=MPI.COMM_WORLD)
        mass.setUp()
        stiffness.setUp()

        x = PETSc.Vec().createMPI(ndofs, comm=MPI.COMM_WORLD)
        rstart, rend = x.getOwnershipRange()
        x.set(0.0)

        for row in range(rstart, rend):
            i = row % n
            j = row // n
            boundary = i == 0 or i == n - 1 or j == 0 or j == n - 1

            mass.setValue(row, row, 1.0)
            if boundary:
                stiffness.setValue(row, row, 1.0)
                x.setValue(row, 0.0)
                continue

            scale = alpha / (h * h)
            stiffness.setValues(
                row,
                [row, row - 1, row + 1, row - n, row + n],
                [4.0 * scale, -scale, -scale, -scale, -scale],
            )
            x.setValue(row, self._exact_value(i * h, j * h, 0.0))

        mass.assemble()
        stiffness.assemble()
        x.assemblyBegin()
        x.assemblyEnd()

        self._dofs = ndofs
        return mass, stiffness, x

    def transient_l2_error(self, x, time: float):
        h = 1.0 / (self._n - 1)
        rstart, rend = x.getOwnershipRange()
        x_local = x.getArray(readonly=True)
        local_sq = 0.0
        for offset, row in enumerate(range(rstart, rend)):
            i = row % self._n
            j = row // self._n
            diff = x_local[offset] - self._exact_value(i * h, j * h, time)
            local_sq += h * h * float(diff * diff)
        global_sq = MPI.COMM_WORLD.allreduce(local_sq, op=MPI.SUM)
        return math.sqrt(global_sq)

    def _exact_value(self, x: float, y: float, time: float) -> float:
        return (
            math.sin(math.pi * x)
            * math.sin(math.pi * y)
            * math.exp(-2.0 * self._alpha * math.pi * math.pi * time)
        )
