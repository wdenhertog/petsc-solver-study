"""
High-level UFL/dolfinx Poisson variant backed by LinearProblem.
"""

from dolfinx import fem, mesh
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
from ufl import SpatialCoordinate, TestFunction, TrialFunction, dx, exp, grad, inner

from ..problem import Problem, ProblemKind


class PoissonProblemUFL(Problem):
    def __init__(self):
        self._dofs = 0
        self._domain = None
        self._V = None
        self._u_zero = None
        self._u_sol = None
        self._linear_problem = None

    @property
    def name(self) -> str:
        return "poisson"

    @property
    def kind(self) -> ProblemKind:
        return ProblemKind.LINEAR

    def dofs(self) -> int:
        return self._dofs

    def solve_linear(self, n: int = 64):
        comm = MPI.COMM_WORLD
        self._domain = mesh.create_unit_square(
            comm, n - 1, n - 1, mesh.CellType.triangle
        )
        self._V = fem.functionspace(self._domain, ("Lagrange", 1))

        u = TrialFunction(self._V)
        v = TestFunction(self._V)

        x = SpatialCoordinate(self._domain)
        f_expr = exp(-500.0 * ((x[0] - 0.3) ** 2 + (x[1] - 0.7) ** 2))
        a = inner(grad(u), grad(v)) * dx
        L = f_expr * v * dx

        tdim = self._domain.topology.dim
        fdim = tdim - 1
        self._domain.topology.create_connectivity(fdim, tdim)
        boundary_facets = mesh.exterior_facet_indices(self._domain.topology)
        boundary_dofs = fem.locate_dofs_topological(self._V, fdim, boundary_facets)

        self._u_zero = fem.Function(self._V)
        self._u_zero.x.array[:] = 0.0
        bc = fem.dirichletbc(self._u_zero, boundary_dofs)

        self._u_sol = fem.Function(self._V)
        self._linear_problem = LinearProblem(
            a, L, bcs=[bc], u=self._u_sol, petsc_options_prefix="benchmark_"
        )
        ksp = self._linear_problem.solver

        # Allow global -ksp_* / -pc_* sweep flags to apply.
        ksp.setOptionsPrefix("")
        ksp.setFromOptions()

        t0 = MPI.Wtime()
        self._linear_problem.solve()
        t1 = MPI.Wtime()

        self._dofs = self._u_sol.x.petsc_vec.getSize()
        return ksp, self._dofs, 0.0, t1 - t0
