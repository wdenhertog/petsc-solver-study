"""
FEniCSx/dolfinx Poisson problem -- Python counterpart to
include/problems/poisson.hpp / src/problems/poisson.cpp.

Deliberately mirrors the C++ implementation as closely as the two APIs
allow, so cross-validation is comparing the same problem
instance, not two different problems that happen to share a name:

  - same domain (unit square), same `n` = grid points per side
  - same Gaussian-bump forcing term, same center/width
  - same homogeneous Dirichlet BC on the full boundary
  - assembly only -- the caller (runner.py, mirroring main.cpp) owns
    KSP creation, setFromOptions(), timing, and result extraction
"""

from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem, mesh
from dolfinx.fem.petsc import (
    apply_lifting,
    assemble_matrix,
    assemble_vector,
    set_bc,
)
from ufl import SpatialCoordinate, TestFunction, TrialFunction, dx, exp, grad, inner

from ..problem import Problem, ProblemKind


class PoissonProblem(Problem):
    """
    -div(grad(u)) = f      on the unit square
                u = 0      on the full boundary (homogeneous Dirichlet)

    f(x, y) = exp(-500 * ((x - 0.3)^2 + (y - 0.7)^2)) -- localized
    Gaussian bump, identical center/width/sign to forcing() in
    src/problems/poisson.cpp.
    """

    def __init__(self):
        self.domain = None
        self.V = None
        self.u_sol = None
        self.bcs = []
        self._dofs = 0

    @property
    def name(self) -> str:
        return "poisson"

    @property
    def kind(self) -> ProblemKind:
        return ProblemKind.LINEAR

    def dofs(self) -> int:
        """Exposed so runner.py can record it in BenchmarkResult,
        mirroring PoissonProblem::dofs() in the C++ header."""
        return self._dofs

    def assemble_linear(self, n: int = 64):
        """
        Builds and returns (A, b, x) as raw PETSc Mat/Vec objects.
        Mirrors PoissonProblem::assemble_linear(Mat&, Vec&, Vec&) in
        poisson.cpp -- assembly only, no solve.

        n: grid points per side, same convention as the DMDA `-n`
           option read by the C++ binary (DMDACreate2d(..., n, n, ...)).
           nx = ny = n - 1 cells per side gives exactly n points per
           side -- verify `dofs()` matches the C++ output at the same
           `n` before trusting anything downstream of this.
        """
        comm = MPI.COMM_WORLD

        self.domain = mesh.create_unit_square(comm, n - 1, n - 1, mesh.CellType.triangle)
        self.V = fem.functionspace(self.domain, ("Lagrange", 1))

        u = TrialFunction(self.V)
        v = TestFunction(self.V)

        x = SpatialCoordinate(self.domain)
        dx_ = x[0] - 0.3
        dy_ = x[1] - 0.7
        f_expr = exp(-500.0 * (dx_ * dx_ + dy_ * dy_))

        a = inner(grad(u), grad(v)) * dx
        L = f_expr * v * dx

        # Homogeneous Dirichlet u = 0 on the *full* boundary -- same BC
        # type and value as the matrix-row-replacement approach in
        # poisson.cpp, not a Robin/Neumann approximation.
        tdim = self.domain.topology.dim
        fdim = tdim - 1
        self.domain.topology.create_connectivity(fdim, tdim)
        boundary_facets = mesh.exterior_facet_indices(self.domain.topology)
        boundary_dofs = fem.locate_dofs_topological(self.V, fdim, boundary_facets)

        u_zero = fem.Function(self.V)
        u_zero.x.array[:] = 0.0
        bc = fem.dirichletbc(u_zero, boundary_dofs)
        self.bcs = [bc]

        a_compiled = fem.form(a)
        L_compiled = fem.form(L)

        A = assemble_matrix(a_compiled, bcs=self.bcs)
        A.assemble()

        b = assemble_vector(L_compiled)
        apply_lifting(b, [a_compiled], bcs=[self.bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b, self.bcs)

        self.u_sol = fem.Function(self.V)
        x_vec = self.u_sol.x.petsc_vec
        x_vec.set(0.0)

        self._dofs = x_vec.getSize()

        return A, b, x_vec
