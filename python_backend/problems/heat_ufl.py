import math

import numpy as np
from dolfinx import fem, mesh
from dolfinx.fem.petsc import assemble_matrix
from mpi4py import MPI
from ufl import TestFunction, TrialFunction, dx, grad, inner

from ..problem import Problem, ProblemKind


class HeatProblemUFL(Problem):
    def __init__(self):
        self._alpha = 1.0
        self._dofs = 0
        self._domain = None
        self._V = None
        self._u_sol = None

    @property
    def name(self) -> str:
        return "heat"

    @property
    def kind(self) -> ProblemKind:
        return ProblemKind.TRANSIENT

    def dofs(self) -> int:
        return self._dofs

    def assemble_transient(self, n: int = 64, alpha: float = 1.0):
        self._alpha = alpha
        self._domain = mesh.create_unit_square(
            MPI.COMM_WORLD, n - 1, n - 1, mesh.CellType.triangle
        )
        self._V = fem.functionspace(self._domain, ("Lagrange", 1))

        u = TrialFunction(self._V)
        v = TestFunction(self._V)

        tdim = self._domain.topology.dim
        fdim = tdim - 1
        self._domain.topology.create_connectivity(fdim, tdim)
        boundary_facets = mesh.exterior_facet_indices(self._domain.topology)
        boundary_dofs = fem.locate_dofs_topological(self._V, fdim, boundary_facets)

        u_zero = fem.Function(self._V)
        u_zero.x.array[:] = 0.0
        bc = fem.dirichletbc(u_zero, boundary_dofs)

        mass = assemble_matrix(fem.form(u * v * dx), bcs=[bc])
        mass.assemble()

        stiffness = assemble_matrix(fem.form(inner(grad(u), grad(v)) * dx), bcs=[bc])
        stiffness.assemble()
        stiffness.scale(alpha)

        self._u_sol = fem.Function(self._V)
        self._u_sol.interpolate(lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))

        self._dofs = self._u_sol.x.petsc_vec.getSize()
        return mass, stiffness, self._u_sol.x.petsc_vec

    def transient_l2_error(self, x, time: float):
        del x

        u_exact = fem.Function(self._V)
        factor = math.exp(-2.0 * self._alpha * math.pi * math.pi * time)
        u_exact.interpolate(
            lambda x: factor * np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])
        )
        local_sq = fem.assemble_scalar(
            fem.form((self._u_sol - u_exact) * (self._u_sol - u_exact) * dx)
        )
        global_sq = self._domain.comm.allreduce(local_sq, op=MPI.SUM)
        return math.sqrt(global_sq)
