from dolfinx import fem, mesh
from dolfinx.fem.petsc import NonlinearProblem
from mpi4py import MPI
from ufl import TestFunction, TrialFunction, derivative, dx, exp, grad, inner

from ..problem import Problem, ProblemKind


class BratuProblemUFL(Problem):
    def __init__(self):
        self._dofs = 0
        self._domain = None
        self._V = None
        self._u = None
        self._u_zero = None
        self._problem = None

    @property
    def name(self) -> str:
        return "bratu"

    @property
    def kind(self) -> ProblemKind:
        return ProblemKind.NONLINEAR

    def dofs(self) -> int:
        return self._dofs

    def solve_nonlinear(self, n: int = 64, lam: float = 6.0):
        comm = MPI.COMM_WORLD
        self._domain = mesh.create_unit_square(
            comm, n - 1, n - 1, mesh.CellType.triangle
        )
        self._V = fem.functionspace(self._domain, ("Lagrange", 1))

        self._u = fem.Function(self._V)
        self._u.x.array[:] = 0.0
        v = TestFunction(self._V)
        du = TrialFunction(self._V)

        F = inner(grad(self._u), grad(v)) * dx - lam * exp(self._u) * v * dx
        J = derivative(F, self._u, du)

        tdim = self._domain.topology.dim
        fdim = tdim - 1
        self._domain.topology.create_connectivity(fdim, tdim)
        boundary_facets = mesh.exterior_facet_indices(self._domain.topology)
        boundary_dofs = fem.locate_dofs_topological(self._V, fdim, boundary_facets)
        self._u_zero = fem.Function(self._V)
        self._u_zero.x.array[:] = 0.0
        bc = fem.dirichletbc(self._u_zero, boundary_dofs)

        self._problem = NonlinearProblem(
            F,
            self._u,
            bcs=[bc],
            J=J,
            petsc_options_prefix="benchmark_",
        )
        snes = self._problem.solver
        snes.setOptionsPrefix("")
        snes.setFromOptions()

        t0 = MPI.Wtime()
        snes.setUp()
        t1 = MPI.Wtime()
        self._problem.solve()
        t2 = MPI.Wtime()

        self._dofs = self._problem.x.getSize()
        return snes, self._dofs, t1 - t0, t2 - t1
