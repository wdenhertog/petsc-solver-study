from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import ufl
from basix.ufl import element, mixed_element
from dolfinx import fem, geometry, mesh
from mpi4py import MPI
from numpy.typing import NDArray
from petsc4py import PETSc
from ufl import TestFunction
from ufl.core.expr import Expr

from .meshes import CellTag, FacetTag, TaggedMesh
from .parameters import (
    CharScales,
    PhysicalParams,
    WoundGeometry1D,
)


def regularized_pow(u: Expr, power: float, eps: float = 1e-8) -> Expr:
    """
    Compute u^power with regularization near u=0.

    For 0 < power < 1, the unregularized u^power is not differentiable at u=0:
        d/du u^p |_{u=0} = lim_{h->0+} h^(p-1) = +inf  (since p - 1 < 0).
    This breaks Newton's method (singular Jacobian) when u is near zero.

    This form shifts the argument to obtain a finite derivative while
    preserving the exact zero at u=0:
        f(0)  = 0
        f'(0) = power * eps^(power - 1)   (finite for eps > 0)
        f(u)  -> u^power  for u >> eps

    Parameters
    ----------
    u : Expr
        The UFL expression to raise to a power.
    power : float
        The exponent to raise u to.
    eps : float, optional
        Regularization parameter to avoid singularity at u=0 (default is 1e-8).

    Returns
    -------
    Expr
        The regularized power expression (u + eps)^power - eps^power.
    """
    return (u + eps) ** power - eps**power


class Problem(ABC):
    """
    Abstract base class for burn injury problems.

    This class defines the common interface and shared functionality for all
    burn injury problem formulations. It handles the setup of mixed finite
    element spaces, solution functions, and characteristic scales.

    Parameters
    ----------
    domain : mesh.Mesh or TaggedMesh
        The computational mesh domain. The 2D (curved) simulations require a TaggedMesh to define the wound region.
    physical_params : PhysicalParams
        Physical parameters for the burn injury model.
    char_scales : CharScales
        Characteristic scales for non-dimensionalization.

    Attributes
    ----------
    domain : mesh.Mesh
        The computational mesh.
    params : PhysicalParams
        Physical parameters for the model.
    scales : CharScales
        Characteristic scales.
    V : fem.FunctionSpace
        The mixed finite element function space.
    U : fem.Function
        The solution function.
    dim : int
        The topological dimension of the mesh.
    gdim : int
        The geometric dimension of the mesh.
    """

    def __init__(
        self,
        domain: mesh.Mesh | TaggedMesh,
        physical_params: PhysicalParams,
        char_scales: CharScales,
    ) -> None:
        """
        Initialize the problem with domain, parameters, and scales.

        Parameters
        ----------
        domain : mesh.Mesh or TaggedMesh
            The computational mesh domain.
        physical_params : PhysicalParams
            Physical parameters for the burn injury model.
        char_scales : CharScales
            Characteristic scales for non-dimensionalization.

        Raises
        ------
        TypeError
            If physical_params is not a PhysicalParams instance or
            char_scales is not a CharScales instance.
        """

        if not isinstance(physical_params, PhysicalParams):
            raise TypeError(f"Expected PhysicalParams, got {type(physical_params).__name__}")
        if not isinstance(char_scales, CharScales):
            raise TypeError(f"Expected CharScales, got {type(char_scales).__name__}")

        self.physical_params_typed = physical_params
        self.params = physical_params
        self.scales = char_scales

        if isinstance(domain, TaggedMesh):
            self.tagged_mesh = domain
            self.domain = domain.mesh
            self.cell_tags = domain.cell_tags
            self.facet_tags = domain.facet_tags
        else:
            self.tagged_mesh = None
            self.domain = domain
            self.cell_tags = None
            self.facet_tags = None

        self.comm = self.domain.comm
        self.dim = self.domain.topology.dim
        self.gdim = self.domain.geometry.dim

        self.idx_N = 0
        self.idx_M = 1
        self.idx_rho = 2
        self.idx_c = 3
        self.idx_v = 4
        self.idx_eps = 5

        self.V = fem.functionspace(self.domain, self.build_mixed_element())

        self.U = fem.Function(self.V)
        self.U_dot = fem.Function(self.V)
        self.dU = ufl.TrialFunction(self.V)
        self.V_test = TestFunction(self.V)

        self.v_prev = None  # Save the previous velocity field for post-processing

        self.alpha_const = fem.Constant(self.domain, 1.0)

        self.N_sol = self._create_collapsed_subspace_function(self.idx_N, "N")
        self.M_sol = self._create_collapsed_subspace_function(self.idx_M, "M")
        self.rho_sol = self._create_collapsed_subspace_function(self.idx_rho, "rho")
        self.c_sol = self._create_collapsed_subspace_function(self.idx_c, "c")
        self.v_sol = self._create_collapsed_subspace_function(self.idx_v, "v")
        self.eps_sol = self._create_collapsed_subspace_function(self.idx_eps, "eps")
        self.u_sol = self._create_collapsed_subspace_function(self.idx_v, "u")

        self.x0 = self.domain.geometry.x.copy()

        # Characteristic scales
        self.L_c = self.scales.L_c  # length scale
        self.N_c = self.scales.N_c  # (myo)fibroblast scale
        self.rho_c = self.scales.rho_c  # collagen scale
        self.c_c = self.scales.c_c  # signalling molecule scale
        self.t_c = self.scales.t_c  # time scale
        self.v_c = self.scales.v_c  # velocity scale
        self.sigma_c = self.scales.sigma_c  # stress scale

        # Dimensionless parameters
        self.D_F_hat = fem.Constant(self.domain, self.params.D_F * self.N_c * self.t_c / self.L_c**2)
        self.r_F_hat = fem.Constant(self.domain, self.params.r_F * self.t_c * self.N_c**self.params.q)
        self.a_c_I_hat = fem.Constant(self.domain, self.params.a_c_I / self.c_c)
        self.kappa_F_hat = fem.Constant(self.domain, self.params.kappa_F * self.N_c)
        self.k_F_hat = fem.Constant(self.domain, self.params.k_F * self.t_c * self.c_c)
        self.delta_N_hat = fem.Constant(self.domain, self.params.delta_N * self.t_c)
        self.delta_M_hat = fem.Constant(self.domain, self.params.delta_M * self.t_c)

        self.D_c_hat = fem.Constant(self.domain, self.params.D_c * self.t_c / self.L_c**2)
        self.k_c_hat = fem.Constant(self.domain, self.params.k_c * self.N_c * self.t_c / self.c_c)
        self.a_c_II_hat = fem.Constant(self.domain, self.params.a_c_II / self.c_c)
        self.a_c_III_hat = fem.Constant(self.domain, self.params.a_c_III * self.c_c)
        self.delta_c_hat = fem.Constant(self.domain, self.params.delta_c * self.N_c * self.rho_c * self.t_c)

        self.k_rho_hat = fem.Constant(self.domain, self.params.k_rho * self.N_c * self.t_c / self.rho_c)
        self.a_c_IV_hat = fem.Constant(self.domain, self.params.a_c_IV / self.c_c)
        self.delta_rho_hat = fem.Constant(self.domain, self.params.delta_rho * self.N_c * self.rho_c * self.t_c)

        self.mu_hat = fem.Constant(self.domain, self.params.mu_2 / self.params.mu_1)
        self.El = fem.Constant(self.domain, self.params.E * np.sqrt(self.rho_c) / self.sigma_c)
        self.nu = fem.Constant(self.domain, self.params.nu)
        self.xi_hat = fem.Constant(self.domain, self.params.xi * self.N_c / (self.rho_c * self.sigma_c))
        self.R_hat = fem.Constant(self.domain, self.params.R / self.rho_c)
        self.zeta_hat = fem.Constant(self.domain, self.params.zeta * self.N_c * self.t_c * self.c_c)

        self.eta_I = fem.Constant(self.domain, self.params.eta_I)
        self.eta_II = fem.Constant(self.domain, self.params.eta_II)
        self.q = fem.Constant(self.domain, self.params.q)
        self.r_F_max = fem.Constant(self.domain, self.params.r_F_max)
        self.k_rho_max = fem.Constant(self.domain, self.params.k_rho_max)

        self.k_sub_hat = fem.Constant(self.domain, self.params.k_sub * (self.L_c**2) / self.sigma_c)
        self.c_sub_hat = fem.Constant(self.domain, self.params.c_sub * (self.L_c**2) / self.params.mu_1)
        self.beta_h_hat = fem.Constant(self.domain, self.params.beta_h * self.rho_c * self.L_c / self.sigma_c)

    @abstractmethod
    def build_mixed_element(self) -> mixed_element:
        """
        Build the mixed finite element for the full state vector.

        Subclasses must implement this method to return the mixed element
        appropriate for their dimensionality and state-variable layout
        (e.g. scalar state variables for 1D, vector velocity and packed
        symmetric strain for higher dimensions).

        Returns
        -------
        mixed_element
            The mixed finite element for the full state vector.
        """
        ...

    def get_mesh_metadata(self) -> dict[str, Any]:
        """
        Extract metadata dictionary from problem mesh.

        Returns
        -------
        dict[str, Any]
            Dictionary containing mesh metadata including number of cells,
            vertices, dimensions, cell type, and domain extent.
        """
        if self.tagged_mesh is not None:
            return self.tagged_mesh.to_dict()
        if self.domain is not None:
            comm = self.domain.comm
            tdim = self.domain.topology.dim
            gdim = self.domain.geometry.dim
            num_cells = self.domain.topology.index_map(tdim).size_global
            num_vertices = self.domain.topology.index_map(0).size_global
            cell_type = self.domain.topology.cell_name()

            x_coords = self.domain.geometry.x[:, 0]
            if len(x_coords) > 0:
                local_min_x = float(np.min(x_coords))
                local_max_x = float(np.max(x_coords))
            else:
                local_min_x = float("inf")
                local_max_x = float("-inf")
            min_x = comm.allreduce(local_min_x, op=MPI.MIN)
            max_x = comm.allreduce(local_max_x, op=MPI.MAX)

            return {
                "num_cells": num_cells,
                "num_vertices": num_vertices,
                "tdim": tdim,
                "gdim": gdim,
                "cell_type": cell_type,
                "domain_length": max_x - min_x,
                "points": [min_x, max_x],
            }
        return {}

    def _create_collapsed_subspace_function(self, subspace_index: int, name: str) -> fem.Function:
        """
        Create a named output Function matching the element of a mixed subspace.

        The Function is created on a fresh space built directly on the mesh instead
        of on the collapsed subspace itself. A collapsed subspace of a mixed space
        can have a DOF ordering that is permuted with respect to the mesh geometry
        (this notably happens for vector subspaces on Gmsh-generated meshes), and
        VTXWriter writes the raw DOF arrays against the geometry ordering of the
        first function, which would scramble the saved fields.

        Parameters
        ----------
        subspace_index : int
            The index of the subspace within the mixed space.
        name : str
            The name to assign to the function.

        Returns
        -------
        fem.Function
            A DOLFINx Function on a fresh function space with the same element
            as the collapsed subspace.
        """
        V_sub, _ = self.V.sub(subspace_index).collapse()
        V_out = fem.functionspace(self.domain, V_sub.ufl_element())
        return fem.Function(V_out, name=name)

    @abstractmethod
    def define_weak_forms(self) -> ufl.Form:
        """
        Calculate and store the weak forms for the problem.

        Subclasses must implement this method to define the variational
        formulation of the burn injury model.

        Returns
        -------
        ufl.Form
            The combined weak form.
        """
        ...

    @abstractmethod
    def define_boundary_conditions(self) -> list[fem.DirichletBC]:
        """
        Calculate and store the boundary conditions for the problem.

        Subclasses must implement this method to define Dirichlet boundary
        conditions for the burn injury model.

        Returns
        -------
        list[fem.DirichletBC]
            A list of Dirichlet boundary conditions.
        """
        ...

    @abstractmethod
    def define_initial_conditions(self) -> None:
        """
        Calculate and populate the initial state functions.

        Subclasses must implement this method to set the initial conditions
        for all state variables in the burn injury model.
        """
        ...

    def _interpolate_v_at_points(
        self, points_2d: NDArray[np.float64], v_subfunction: fem.Function
    ) -> NDArray[np.float64]:
        """
        Evaluate a vector Function at arbitrary points.

        Uses DOLFINx's cell-location machinery, supporting parallel MPI
        evaluation across partitions.

        Parameters
        ----------
        points_2d : NDArray[np.float64]
            Array of shape (n_points, 2) containing the 2D coordinates
            at which to evaluate the function.
        v_subfunction : fem.Function
            The collapsed vector function to evaluate.

        Returns
        -------
        NDArray[np.float64]
            Array of shape (n_points, v_dim) containing the evaluated
            function values at the given points.

        Raises
        ------
        RuntimeError
            If any points cannot be located in any cell across all MPI ranks.
        """
        n_points = points_2d.shape[0]
        points_3d = np.zeros((n_points, 3), dtype=np.float64)
        points_3d[:, :2] = points_2d

        if self.domain.comm.size == 1:
            tree = geometry.bb_tree(self.domain, self.dim)
            cell_candidates = geometry.compute_collisions_points(tree, points_3d)
            colliding_cells = geometry.compute_colliding_cells(self.domain, cell_candidates, points_3d)

            cells = np.full(n_points, -1, dtype=np.int32)
            for i in range(n_points):
                links = colliding_cells.links(i)
                if len(links) > 0:
                    cells[i] = links[0]

            missing = cells < 0
            if np.any(missing):
                midpoint_tree = geometry.create_midpoint_tree(
                    self.domain,
                    self.dim,
                    np.arange(self.domain.topology.index_map(self.dim).size_local, dtype=np.int32),
                )
                closest_cells = geometry.compute_closest_entity(tree, midpoint_tree, self.domain, points_3d[missing])
                cells[missing] = closest_cells

            return v_subfunction.eval(points_3d, cells)
        else:
            comm = self.domain.comm
            pod = geometry.determine_point_ownership(self.domain, points_3d, 0.1)

            # Mirror the serial branch: points that no rank could locate must not be
            # silently assigned zero velocity, as that would freeze those dofs and make
            # parallel results diverge from serial ones.
            n_missing_local = int(np.sum(np.asarray(pod.src_owner) < 0))
            n_missing = comm.allreduce(n_missing_local)
            if n_missing > 0:
                raise RuntimeError(
                    f"{n_missing} points could not be located in any cell across all ranks "
                    f"during velocity interpolation (e.g. point outside domain)."
                )

            dest_vals = v_subfunction.eval(pod.dest_points, pod.dest_cells)
            v_dim = dest_vals.shape[1] if dest_vals.ndim > 1 else 1

            send_data = [dest_vals[pod.dest_owner == r] for r in range(comm.size)]
            recv_data = comm.alltoall(send_data)

            result = np.zeros((n_points, v_dim), dtype=np.float64)
            for r in range(comm.size):
                mask = pod.src_owner == r
                if np.any(mask):
                    result[mask] = recv_data[r]

            return result

    def update_time_dependent_terms(self, t_hat: float) -> None:
        """
        Update any explicitly time-dependent terms of the problem.

        Called by the time-stepping solver before each residual/Jacobian
        evaluation, since PETSc TS evaluates the residual at intermediate stage
        times. The base problem has no explicitly time-dependent terms, so this
        is a no-op; subclasses (e.g. manufactured-solution problems) override
        it to refresh time-dependent sources and boundary data.

        Parameters
        ----------
        t_hat : float
            The current non-dimensional time.
        """

    def update_post_processing(self, dt_dimensional: float) -> None:
        """
        Update post-processing fields after a time step.

        Uses a second-order Runge-Kutta method to update the displacement field
        based on the current velocity field.

        Parameters
        ----------
        dt_dimensional : float
            The dimensional time step size in days.
        """
        # Use RK2 to update the displacement field u_sol based on the current velocity field v_sol.

        v_new = self.U.sub(self.idx_v).collapse()
        v_new.x.array[:] *= self.v_c  # dimensional velocity, in the v-subspace's own dofmap
        v_new.x.scatter_forward()

        if self.v_prev is None:
            # No history yet, just do Euler forward for 1 time step
            self.v_prev = fem.Function(v_new.function_space)
            self.v_prev.x.array[:] = v_new.x.array

        if self.dim == 1:
            x0 = self.x0[:, 0]
            current_x = x0 + self.u_sol.x.array

            k1 = np.interp(current_x, x0, self.v_prev.x.array)
            x_pred = current_x + k1 * dt_dimensional
            k2 = np.interp(x_pred, x0, v_new.x.array)

            self.u_sol.x.array[:] += 0.5 * (k1 + k2) * dt_dimensional
            self.u_sol.x.scatter_forward()
        elif self.dim == 2:
            dof_coords = self.u_sol.function_space.tabulate_dof_coordinates()[:, :2]
            u_array = self.u_sol.x.array.reshape(-1, 2)
            current_pos = dof_coords + u_array

            k1 = self._interpolate_v_at_points(current_pos, self.v_prev)
            x_pred = current_pos + k1 * dt_dimensional
            k2 = self._interpolate_v_at_points(x_pred, v_new)

            self.u_sol.x.array[:] += 0.5 * (k1 + k2).ravel() * dt_dimensional
            self.u_sol.x.scatter_forward()

        self.v_prev.x.array[:] = v_new.x.array
        self.v_prev.x.scatter_forward()

    def get_functions_for_saving(self) -> list[fem.Function]:
        """
        Split the mixed solution functions for saving.

        Interpolation (rather than a raw array copy) maps the mixed-space DOFs onto
        the geometry-ordered output spaces, which is required for correct VTX output.

        Returns
        -------
        list[fem.Function]
            List of functions containing N, M, rho, c, v, eps, and u,
            scaled to dimensional values.
        """
        self.N_sol.interpolate(self.U.sub(self.idx_N))
        self.N_sol.x.array[:] *= self.N_c
        self.M_sol.interpolate(self.U.sub(self.idx_M))
        self.M_sol.x.array[:] *= self.N_c
        self.rho_sol.interpolate(self.U.sub(self.idx_rho))
        self.rho_sol.x.array[:] *= self.rho_c
        self.c_sol.interpolate(self.U.sub(self.idx_c))
        self.c_sol.x.array[:] *= self.c_c
        self.v_sol.interpolate(self.U.sub(self.idx_v))
        self.v_sol.x.array[:] *= self.v_c
        self.eps_sol.interpolate(self.U.sub(self.idx_eps))
        return [
            self.N_sol,
            self.M_sol,
            self.rho_sol,
            self.c_sol,
            self.v_sol,
            self.eps_sol,
            self.u_sol,
        ]

    def g(self, N, M, rho, c):
        return (N + self.eta_II * M) * rho / (1 + self.a_c_III_hat * c)

    def G(self, N, M, c, eps):
        return self.zeta_hat * (N + self.eta_II * M) / (1 + self.a_c_III_hat * c) * c * eps

    def R_N(self, N, M, c):
        proliferation = (
            self.r_F_hat
            * (1 + self.r_F_max * c / (self.a_c_I_hat + c))
            * (1 - self.kappa_F_hat * (N + M))
            * regularized_pow(N, 1 + self.q)
        )
        return proliferation - self.k_F_hat * c * N - self.delta_N_hat * N

    def R_M(self, N, M, c):
        proliferation = (
            self.r_F_hat
            * ((1 + self.r_F_max) * c / (self.a_c_I_hat + c))
            * (1 - self.kappa_F_hat * (N + M))
            * regularized_pow(M, 1 + self.q)
        )
        return proliferation + self.k_F_hat * c * N - self.delta_M_hat * M

    def R_c(self, N, M, c, rho):
        return (
            self.k_c_hat * c / (self.a_c_II_hat + c) * (N + self.eta_I * M)
            - self.delta_c_hat * self.g(N, M, rho, c) * c
        )

    def R_rho(self, N, M, rho, c):
        return (
            self.k_rho_hat * (1 + self.k_rho_max * c / (self.a_c_IV_hat + c)) * (N + self.eta_I * M)
            - self.delta_rho_hat * self.g(N, M, rho, c) * rho
        )

    def f(self, M, rho):
        return self.xi_hat * (M * rho / (self.R_hat**2 + rho**2))


class BurnProblem1D(Problem):
    """
    Burn injury problem with the current 1D formulation.

    This class implements the 1D version of the burn injury model, including
    wound geometry setup, weak form definition, boundary conditions, and
    initial conditions.

    Parameters
    ----------
    domain : mesh.Mesh
        The 1D computational mesh.
    physical_params : PhysicalParams
        Physical parameters for the burn injury model.
    char_scales : CharScales
        Characteristic scales for non-dimensionalization.
    wound_geometry : WoundGeometry1D or None, optional
        Wound geometry specification (default is None, which creates a default
        wound centered at 5.0 cm with length 4.0 cm).

    Attributes
    ----------
    wound_geometry : WoundGeometry1D
        The wound geometry specification.
    left_wound_node_idx : int
        Index of the node closest to the left wound boundary.
    right_wound_node_idx : int
        Index of the node closest to the right wound boundary.
    X_left_0 : float
        Initial x-coordinate of the left wound boundary node.
    X_right_0 : float
        Initial x-coordinate of the right wound boundary node.
    initial_wound_length : float
        Initial length of the wound region.
    """

    def __init__(
        self,
        domain: mesh.Mesh,
        physical_params: PhysicalParams,
        char_scales: CharScales,
        wound_geometry: WoundGeometry1D | None = None,
    ) -> None:
        """
        Initialize the 1D burn problem.

        Parameters
        ----------
        domain : mesh.Mesh
            The 1D computational mesh.
        physical_params : PhysicalParams
            Physical parameters for the burn injury model.
        char_scales : CharScales
            Characteristic scales for non-dimensionalization.
        wound_geometry : WoundGeometry1D or None, optional
            Wound geometry specification (default is None).

        Raises
        ------
        ValueError
            If the mesh dimension is not 1 or if the wound boundary nodes
            are not in the correct order.
        """
        super().__init__(domain, physical_params, char_scales)
        if self.dim != 1:
            raise ValueError("BurnProblem1D requires a 1D mesh.")
        self.F = None
        self.bcs = None

        if wound_geometry is None:
            wound_geometry = WoundGeometry1D(center=5.0, length=4.0)

        self.wound_geometry = wound_geometry

        if self.wound_geometry is not None:
            center = self.wound_geometry.center
            length = self.wound_geometry.length
            left_bdy_x = center - length / 2.0
            right_bdy_x = center + length / 2.0

            # Find the node indices closest to the initial wound boundaries
            self.left_wound_node_idx = np.argmin(np.abs(self.x0[:, 0] - left_bdy_x))
            self.right_wound_node_idx = np.argmin(np.abs(self.x0[:, 0] - right_bdy_x))

            # Store the actual initial coordinates of these nodes
            self.X_left_0 = self.x0[self.left_wound_node_idx, 0]
            self.X_right_0 = self.x0[self.right_wound_node_idx, 0]

            self.initial_wound_length = self.X_right_0 - self.X_left_0

            if self.initial_wound_length <= 0:
                raise ValueError("Wound boundary nodes are not in the correct order or are the same.")
        else:
            self.initial_wound_length = None

    def build_mixed_element(self) -> mixed_element:
        """
        Build the 1D mixed finite element for the full state vector.

        Returns
        -------
        mixed_element
            A mixed element containing 6 scalar P1 Lagrange elements, one
            for each state variable (N, M, rho, c, v, eps).
        """
        P1 = element("Lagrange", self.domain.topology.cell_name(), 1)
        return mixed_element([P1] * 6)

    def define_weak_forms(self) -> None:
        """
        Calculate and store the weak forms for the 1D burn problem.

        Defines the variational formulation for all state variables (N, M, rho, c, v, eps)
        and stores the combined weak form in self.F and its Jacobian in self.J.
        """

        def J_N(N_ufl: Expr, M_ufl: Expr) -> Expr:
            return self.D_F_hat * (N_ufl + M_ufl) * N_ufl.dx(0)

        def J_M(N_ufl: Expr, M_ufl: Expr) -> Expr:
            return self.D_F_hat * (N_ufl + M_ufl) * M_ufl.dx(0)

        def J_c(c_ufl: Expr) -> Expr:
            return self.D_c_hat * c_ufl.dx(0)  # type: ignore

        def sigma(rho_ufl: Expr, v_ufl: Expr, epsilon_ufl: Expr) -> Expr:
            return v_ufl.dx(0) * (1 + self.mu_hat) + self.El * regularized_pow(rho_ufl, 0.5) * epsilon_ufl

        N, M, rho, c, v, eps = ufl.split(self.U)
        N_test, M_test, rho_test, c_test, v_test, eps_test = ufl.split(self.V_test)
        N_dot, M_dot, rho_dot, c_dot, v_dot, eps_dot = ufl.split(self.U_dot)

        F_N = (
            N_dot * N_test * ufl.dx
            + v * N.dx(0) * N_test * ufl.dx
            + N * v.dx(0) * N_test * ufl.dx
            + J_N(N, M) * N_test.dx(0) * ufl.dx
            - N_test * self.R_N(N, M, c) * ufl.dx
        )

        F_M = (
            M_dot * M_test * ufl.dx
            + v * M.dx(0) * M_test * ufl.dx
            + M * v.dx(0) * M_test * ufl.dx
            + J_M(N, M) * M_test.dx(0) * ufl.dx
            - M_test * self.R_M(N, M, c) * ufl.dx
        )

        F_c = (
            c_dot * c_test * ufl.dx
            + v * c.dx(0) * c_test * ufl.dx
            + c * v.dx(0) * c_test * ufl.dx
            + J_c(c) * c_test.dx(0) * ufl.dx
            - c_test * self.R_c(N, M, c, rho) * ufl.dx
        )

        F_rho = (
            rho_dot * rho_test * ufl.dx
            + v * rho.dx(0) * rho_test * ufl.dx
            + rho * v.dx(0) * rho_test * ufl.dx
            - rho_test * self.R_rho(N, M, rho, c) * ufl.dx
        )
        F_v = v_test.dx(0) * sigma(rho, v, eps) * ufl.dx + self.f(M, rho) * v_test.dx(0) * ufl.dx

        F_epsilon = (
            eps_dot * eps_test * ufl.dx
            + v * eps_test * eps.dx(0) * ufl.dx
            + (eps - 1) * v.dx(0) * eps_test * ufl.dx
            + self.G(N, M, c, eps) * eps_test * ufl.dx
        )

        self.F = F_N + F_M + F_c + F_rho + F_v + F_epsilon
        self.J = ufl.derivative(self.F, self.U, self.dU) + self.alpha_const * ufl.derivative(
            self.F, self.U_dot, self.dU
        )
        self.F_compiled = fem.form(self.F)
        self.J_compiled = fem.form(self.J)

    def define_boundary_conditions(self) -> None:
        """
        Calculate and store the boundary conditions for the 1D burn problem.

        Sets Dirichlet boundary conditions for N, M, c, and v at the domain boundaries
        using equilibrium values from the physical parameters.
        """
        boundary_facets = mesh.locate_entities_boundary(
            self.domain, self.domain.topology.dim - 1, lambda x: np.full(x.shape[1], True)
        )

        N_boundary = fem.Constant(self.domain, PETSc.ScalarType(self.params.N_eq / self.N_c))
        M_boundary = fem.Constant(self.domain, PETSc.ScalarType(self.params.M_eq / self.N_c))
        c_boundary = fem.Constant(self.domain, PETSc.ScalarType(self.params.c_eq / self.c_c))
        v_boundary = fem.Constant(self.domain, PETSc.ScalarType(self.params.v_bd / self.v_c))

        N_dofs = fem.locate_dofs_topological(self.V.sub(self.idx_N), self.domain.topology.dim - 1, boundary_facets)
        M_dofs = fem.locate_dofs_topological(self.V.sub(self.idx_M), self.domain.topology.dim - 1, boundary_facets)
        c_dofs = fem.locate_dofs_topological(self.V.sub(self.idx_c), self.domain.topology.dim - 1, boundary_facets)
        v_dofs = fem.locate_dofs_topological(self.V.sub(self.idx_v), self.domain.topology.dim - 1, boundary_facets)

        N_bc = fem.dirichletbc(N_boundary, N_dofs, self.V.sub(self.idx_N))
        M_bc = fem.dirichletbc(M_boundary, M_dofs, self.V.sub(self.idx_M))
        c_bc = fem.dirichletbc(c_boundary, c_dofs, self.V.sub(self.idx_c))
        v_bc = fem.dirichletbc(v_boundary, v_dofs, self.V.sub(self.idx_v))

        self.bcs = [N_bc, M_bc, c_bc, v_bc]

    def define_initial_conditions(self) -> None:
        """
        Calculate and populate the initial state functions for the 1D burn problem.

        Sets initial conditions for all state variables (N, M, rho, c, v, eps) using
        smooth wound indicators to create spatially varying initial distributions.
        """

        # Do some calculations for the initial conditions
        def smooth_wound_indicator(x_values: NDArray[np.float64], alpha: float = 5) -> NDArray[np.float64]:
            # Wound boundary layer ~1/alpha
            dx = np.abs(x_values[:] - self.wound_geometry.center)
            sx = 0.5 * (1 - np.tanh(alpha * (dx - self.wound_geometry.length / 2)))
            return sx

        def initial_N(x_values: NDArray[np.float64]) -> NDArray[np.float64]:
            w = smooth_wound_indicator(x_values[0])
            return (w * self.params.N_tilde + (1 - w) * self.params.N_eq) / self.N_c

        def initial_c(x_values: NDArray[np.float64]) -> NDArray[np.float64]:
            w = smooth_wound_indicator(x_values[0])
            return (w * self.params.c_tilde + (1 - w) * self.params.c_eq) / self.c_c

        def initial_rho(x_values: NDArray[np.float64]) -> NDArray[np.float64]:
            w = smooth_wound_indicator(x_values[0])
            return (w * self.params.rho_tilde + (1 - w) * self.params.rho_eq) / self.rho_c

        self.U.sub(self.idx_N).interpolate(initial_N)
        self.U.sub(self.idx_M).interpolate(lambda x: np.full(x.shape[1], self.params.M_eq / self.N_c))
        self.U.sub(self.idx_rho).interpolate(initial_rho)
        self.U.sub(self.idx_c).interpolate(initial_c)
        self.U.sub(self.idx_v).interpolate(lambda x: np.full(x.shape[1], self.params.v_bd / self.v_c))
        self.U.sub(self.idx_eps).interpolate(lambda x: np.zeros(x.shape[1]))


class BurnProblem2D(Problem):
    """
    Planar 2D burn injury formulation.

    This class provides a framework for 2D burn injury simulations with
    tagged meshes and complex wound geometries.

    Parameters
    ----------
    domain : TaggedMesh
        The tagged computational mesh.
    physical_params : PhysicalParams
        Physical parameters for the burn injury model.
    char_scales : CharScales
        Characteristic scales for non-dimensionalization.
    """

    def __init__(
        self,
        domain: TaggedMesh,
        physical_params: PhysicalParams,
        char_scales: CharScales,
    ) -> None:
        """
        Initialize the 2D burn problem.

        Parameters
        ----------
        domain : TaggedMesh
            The tagged computational mesh.
        physical_params : PhysicalParams
            Physical parameters for the burn injury model.
        char_scales : CharScales
            Characteristic scales for non-dimensionalization.

        Raises
        ------
        TypeError
            If domain is not a TaggedMesh.
        ValueError
            If the mesh dimension is not 2 or if the mesh is not flat (gdim != tdim).
        """
        if not isinstance(domain, TaggedMesh):
            raise TypeError(f"BurnProblem2D requires a TaggedMesh, got {type(domain).__name__} instead.")
        super().__init__(domain, physical_params, char_scales)
        if self.dim != 2:
            raise ValueError("BurnProblem2D requires a 2D mesh.")
        if type(self) is BurnProblem2D and self.gdim != self.dim:
            raise ValueError(
                f"BurnProblem2D requires a flat mesh (gdim == tdim == {self.dim}), "
                f"got gdim={self.gdim}. Use BurnProblemSurface for curved manifold meshes."
            )
        self.F = None
        self.bcs = None

    def build_mixed_element(self):
        """
        Build the 2D mixed element with vector velocity and packed symmetric strain.

        Returns
        -------
        mixed_element
            A mixed element containing P1 elements for N, M, rho, c, a vector P1 for velocity,
            and a packed symmetric P1 for strain.
        """
        P1 = element("Lagrange", self.domain.topology.cell_name(), 1)
        vector_P1 = element("Lagrange", self.domain.topology.cell_name(), 1, shape=(self.gdim,))
        symmetric_components = self.gdim * (self.gdim + 1) // 2
        packed_symmetric_P1 = element("Lagrange", self.domain.topology.cell_name(), 1, shape=(symmetric_components,))
        return mixed_element([P1, P1, P1, P1, vector_P1, packed_symmetric_P1])

    def unpack_symmetric_tensor(self, eps_packed: Expr) -> Expr:
        """
        Reconstruct a symmetric tensor from a packed vector.

        The components are packed using a lexicographical (row-major) order
        for the upper triangular part:
        - gdim=2 (flat 2D mesh): [eps_xx, eps_xy, eps_yy] -> 2x2 tensor
        - gdim=3 (2D surface embedded in 3D): [eps_xx, eps_xy, eps_xz, eps_yy, eps_yz, eps_zz] -> 3x3 tensor

        Parameters
        ----------
        eps_packed : Expr
            The packed vector representation of the symmetric tensor.

        Returns
        -------
        Expr
            The reconstructed symmetric tensor as a UFL tensor expression.

        Raises
        ------
        ValueError
            If geometric dimension is not 2 or 3.
        """
        if self.gdim == 2:
            return ufl.as_tensor([[eps_packed[0], eps_packed[1]], [eps_packed[1], eps_packed[2]]])
        elif self.gdim == 3:
            return ufl.as_tensor(
                [
                    [eps_packed[0], eps_packed[1], eps_packed[2]],
                    [eps_packed[1], eps_packed[3], eps_packed[4]],
                    [eps_packed[2], eps_packed[4], eps_packed[5]],
                ]
            )
        else:
            raise ValueError(f"Unsupported dimension {self.gdim}. Must be 2 or 3.")

    def J_N(self, N_ufl: Expr, M_ufl: Expr) -> Expr:
        """Diffusive flux of fibroblasts N (used by the weak form and MMS source terms)."""
        return self.D_F_hat * (N_ufl + M_ufl) * ufl.grad(N_ufl)

    def J_M(self, N_ufl: Expr, M_ufl: Expr) -> Expr:
        """Diffusive flux of myofibroblasts M (used by the weak form and MMS source terms)."""
        return self.D_F_hat * (N_ufl + M_ufl) * ufl.grad(M_ufl)

    def J_c(self, c_ufl: Expr) -> Expr:
        """Diffusive flux of the signalling molecule c (used by the weak form and MMS source terms)."""
        return self.D_c_hat * ufl.grad(c_ufl)

    def sigma(self, rho_ufl: Expr, v_ufl: Expr, epsilon_ufl: Expr) -> Expr:
        """Mechanical stress tensor (used by the weak form and MMS source terms)."""
        identitity_tensor = ufl.Identity(self.gdim)
        sym_grad_v = ufl.sym(ufl.grad(v_ufl))
        term2 = self.mu_hat * ufl.tr(sym_grad_v) * identitity_tensor
        elastic_prefactor = (self.El * regularized_pow(rho_ufl, 0.5)) / (1.0 + self.nu)
        strain_bracket = epsilon_ufl + ufl.tr(epsilon_ufl) * (self.nu / (1.0 - 2.0 * self.nu)) * identitity_tensor
        term3 = elastic_prefactor * strain_bracket
        return sym_grad_v + term2 + term3

    def weak_form_momentum(self, M: Expr, rho: Expr, v: Expr, eps: Expr, v_test: Expr) -> Expr:
        """
        Momentum balance weak form for a planar 2D mesh.

        This is the single extension point of :meth:`define_weak_forms`: subclasses
        on curved manifolds override it to add surface-specific contributions
        (e.g. subcutaneous foundation resistance and hypertrophic bulging in
        :class:`BurnProblemSurface`) while inheriting all remaining equations.

        Parameters
        ----------
        M : Expr
            Myofibroblast density (UFL split of ``self.U``).
        rho : Expr
            Collagen density (UFL split of ``self.U``).
        v : Expr
            Velocity field (UFL split of ``self.U``).
        eps : Expr
            Unpacked symmetric strain tensor (UFL split of ``self.U``).
        v_test : Expr
            Velocity test function (UFL split of ``self.V_test``).

        Returns
        -------
        Expr
            The momentum contribution to the combined weak form.
        """
        return (
            ufl.inner(self.sigma(rho, v, eps), ufl.sym(ufl.grad(v_test))) * ufl.dx
            + self.f(M, rho) * ufl.div(v_test) * ufl.dx
        )

    def define_weak_forms(self) -> None:
        """
        Calculate and store the weak forms for the 2D burn problem.

        Defines the variational formulation for all state variables (N, M, rho, c, v, eps)
        in the 2D setting and stores the combined weak form in self.F
        and its Jacobian in self.J.

        This definition is shared with :class:`BurnProblemSurface` (curved 2D manifolds),
        which inherits it unchanged and only overrides :meth:`weak_form_momentum`.
        """
        N, M, rho, c, v, eps_packed = ufl.split(self.U)
        N_test, M_test, rho_test, c_test, v_test, eps_test_packed = ufl.split(self.V_test)
        N_dot, M_dot, rho_dot, c_dot, v_dot, eps_dot_packed = ufl.split(self.U_dot)

        eps = self.unpack_symmetric_tensor(eps_packed)
        eps_test = self.unpack_symmetric_tensor(eps_test_packed)
        eps_dot = self.unpack_symmetric_tensor(eps_dot_packed)

        F_N = (
            N_dot * N_test * ufl.dx
            + ufl.dot(v, ufl.grad(N)) * N_test * ufl.dx
            + N * ufl.div(v) * N_test * ufl.dx
            + ufl.inner(self.J_N(N, M), ufl.grad(N_test)) * ufl.dx
            - N_test * self.R_N(N, M, c) * ufl.dx
        )

        F_M = (
            M_dot * M_test * ufl.dx
            + ufl.dot(v, ufl.grad(M)) * M_test * ufl.dx
            + M * ufl.div(v) * M_test * ufl.dx
            + ufl.inner(self.J_M(N, M), ufl.grad(M_test)) * ufl.dx
            - M_test * self.R_M(N, M, c) * ufl.dx
        )

        F_c = (
            c_dot * c_test * ufl.dx
            + ufl.dot(v, ufl.grad(c)) * c_test * ufl.dx
            + c * ufl.div(v) * c_test * ufl.dx
            + ufl.inner(self.J_c(c), ufl.grad(c_test)) * ufl.dx
            - c_test * self.R_c(N, M, c, rho) * ufl.dx
        )

        F_rho = (
            rho_dot * rho_test * ufl.dx
            + ufl.dot(v, ufl.grad(rho)) * rho_test * ufl.dx
            + rho * ufl.div(v) * rho_test * ufl.dx
            - rho_test * self.R_rho(N, M, rho, c) * ufl.dx
        )

        F_v = self.weak_form_momentum(M, rho, v, eps, v_test)

        grad_v = ufl.grad(v)
        identity = ufl.Identity(self.gdim)
        Y2 = identity - 2.0 * eps  # I-2E
        detY2 = ufl.det(Y2)
        sqrt_detY2 = ufl.sqrt(detY2)

        material_deriv = eps_dot + ufl.dot(v, ufl.nabla_grad(eps))
        rhs = ufl.sym(ufl.dot(Y2, grad_v) - 1 / sqrt_detY2 * ufl.dot(Y2, self.G(N, M, c, eps)))

        F_epsilon = ufl.inner(material_deriv, eps_test) * ufl.dx - ufl.inner(rhs, eps_test) * ufl.dx

        self.F = F_N + F_M + F_c + F_rho + F_v + F_epsilon
        self.J = ufl.derivative(self.F, self.U, self.dU) + self.alpha_const * ufl.derivative(
            self.F, self.U_dot, self.dU
        )
        self.F_compiled = fem.form(self.F)
        self.J_compiled = fem.form(self.J)

    def define_boundary_conditions(self) -> None:
        """Calculates and stores the boundary conditions in the 2D burn problem as self.bcs."""

        self.domain.topology.create_connectivity(self.domain.topology.dim - 1, self.domain.topology.dim)

        boundary_facets = self.facet_tags.find(FacetTag.OUTER_BOUNDARY)

        N_boundary = fem.Constant(self.domain, PETSc.ScalarType(self.params.N_eq / self.N_c))
        M_boundary = fem.Constant(self.domain, PETSc.ScalarType(self.params.M_eq / self.N_c))
        c_boundary = fem.Constant(self.domain, PETSc.ScalarType(self.params.c_eq / self.c_c))

        N_dofs = fem.locate_dofs_topological(self.V.sub(self.idx_N), self.domain.topology.dim - 1, boundary_facets)
        M_dofs = fem.locate_dofs_topological(self.V.sub(self.idx_M), self.domain.topology.dim - 1, boundary_facets)
        c_dofs = fem.locate_dofs_topological(self.V.sub(self.idx_c), self.domain.topology.dim - 1, boundary_facets)

        N_bc = fem.dirichletbc(N_boundary, N_dofs, self.V.sub(self.idx_N))
        M_bc = fem.dirichletbc(M_boundary, M_dofs, self.V.sub(self.idx_M))
        c_bc = fem.dirichletbc(c_boundary, c_dofs, self.V.sub(self.idx_c))

        v_val = self.params.v_bd / self.v_c
        V_v, _ = self.V.sub(self.idx_v).collapse()
        v_boundary_func = fem.Function(V_v)
        v_boundary_func.x.array[:] = v_val

        v_dofs = fem.locate_dofs_topological(
            (self.V.sub(self.idx_v), V_v), self.domain.topology.dim - 1, boundary_facets
        )
        v_bc = fem.dirichletbc(v_boundary_func, v_dofs, self.V.sub(self.idx_v))

        self.bcs = [N_bc, M_bc, c_bc, v_bc]

    def define_initial_conditions(self) -> None:
        """
        2D initial conditions for the burn problem.
        Initializes state variables based on cell_tags.
        """

        def initial_M(x_values: NDArray[np.float64]) -> NDArray[np.float64]:
            val = self.params.M_eq / self.N_c
            return np.full(x_values.shape[1], val)

        def initial_v(x_values: NDArray[np.float64]) -> NDArray[np.float64]:
            val = self.params.v_bd / self.v_c
            return np.full((self.gdim, x_values.shape[1]), val)

        def initial_eps(x_values: NDArray[np.float64]) -> NDArray[np.float64]:
            packed_size = self.gdim * (self.gdim + 1) // 2
            return np.zeros((packed_size, x_values.shape[1]))

        self.U.sub(self.idx_M).interpolate(initial_M)
        self.U.sub(self.idx_v).interpolate(initial_v)
        self.U.sub(self.idx_eps).interpolate(initial_eps)

        self.U.sub(self.idx_N).interpolate(lambda x: np.full(x.shape[1], self.params.N_eq / self.N_c))
        self.U.sub(self.idx_c).interpolate(lambda x: np.full(x.shape[1], self.params.c_eq / self.c_c))
        self.U.sub(self.idx_rho).interpolate(lambda x: np.full(x.shape[1], self.params.rho_eq / self.rho_c))

        wound_cells = self.cell_tags.find(CellTag.WOUND)

        # CRITICAL: locate_dofs_topological is a collective operation that ALL ranks must call,
        # even if they have 0 wound cells. Otherwise ranks with no wound cells skip the call
        # while others enter it, causing a deadlock.
        dofs_N = fem.locate_dofs_topological(self.V.sub(self.idx_N), self.domain.topology.dim, wound_cells)
        dofs_c = fem.locate_dofs_topological(self.V.sub(self.idx_c), self.domain.topology.dim, wound_cells)
        dofs_rho = fem.locate_dofs_topological(self.V.sub(self.idx_rho), self.domain.topology.dim, wound_cells)

        # Set wound values - this is safe even if dofs arrays are empty
        if len(dofs_N) > 0:
            self.U.x.array[dofs_N] = self.params.N_tilde / self.N_c
        if len(dofs_c) > 0:
            self.U.x.array[dofs_c] = self.params.c_tilde / self.c_c
        if len(dofs_rho) > 0:
            self.U.x.array[dofs_rho] = self.params.rho_tilde / self.rho_c

        self.U.x.scatter_forward()


class BurnProblemSurface(BurnProblem2D):
    """Burn injury problem formulated on a curved 2D surface embedded in 3D (tdim=2, gdim=3)."""

    def __init__(
        self,
        domain: TaggedMesh,
        physical_params: PhysicalParams,
        char_scales: CharScales,
        unilateral_foundation: bool = True,
    ) -> None:
        if not isinstance(domain, TaggedMesh):
            raise TypeError(f"BurnProblemSurface requires a TaggedMesh, got {type(domain).__name__} instead.")
        if domain.tdim != 2 or domain.gdim != 3:
            raise ValueError(
                f"BurnProblemSurface requires a manifold mesh with tdim=2 and gdim=3, "
                f"got tdim={domain.tdim}, gdim={domain.gdim}."
            )
        self.unilateral_foundation = unilateral_foundation
        super().__init__(domain, physical_params, char_scales)

    def weak_form_momentum(self, M: Expr, rho: Expr, v: Expr, eps: Expr, v_test: Expr) -> Expr:
        """
        Momentum balance weak form on a curved 2D manifold.

        Extends the planar formulation inherited from :class:`BurnProblem2D` with the
        terms that only make sense on a surface embedded in 3D:

        - a subcutaneous elastic/viscous foundation resisting motion along the surface
          normal ``n`` (optionally unilateral, i.e. only resisting penetration into the
          underlying tissue, not detachment from it);
        - an active hypertrophic force bulging the surface outward in proportion to the
          excess collagen density.

        Parameters
        ----------
        M : Expr
            Myofibroblast density (UFL split of ``self.U``).
        rho : Expr
            Collagen density (UFL split of ``self.U``).
        v : Expr
            Velocity field (UFL split of ``self.U``).
        eps : Expr
            Unpacked symmetric strain tensor (UFL split of ``self.U``).
        v_test : Expr
            Velocity test function (UFL split of ``self.V_test``).

        Returns
        -------
        Expr
            The momentum contribution to the combined weak form.
        """
        F_v = super().weak_form_momentum(M, rho, v, eps, v_test)

        n = ufl.CellNormal(self.domain)
        v_n = ufl.dot(v, n)
        v_test_n = ufl.dot(v_test, n)

        # Subcutaneous elastic & viscous foundation resistance
        u_hat = self.u_sol / self.L_c
        u_n = ufl.dot(u_hat, n)
        if self.unilateral_foundation:
            u_n_bed = ufl.min_value(0.0, u_n)
            F_foundation = (self.k_sub_hat * u_n_bed + self.c_sub_hat * v_n) * v_test_n * ufl.dx
        else:
            F_foundation = (self.k_sub_hat * u_n + self.c_sub_hat * v_n) * v_test_n * ufl.dx

        # Active hypertrophic outward bulging from excess collagen
        excess_rho = ufl.max_value(0.0, rho - 1.0)  # rho-rho_eq = rho-1.0 (since rho_eq = 1.0 in dimensionless form)
        F_hypertrophy = -self.beta_h_hat * excess_rho * v_test_n * ufl.dx

        return F_v + F_foundation + F_hypertrophy

    def update_post_processing(self, dt_dimensional: float) -> None:
        """Fixed-node Heun/RK2-in-time displacement integration on reference surface nodes:
        u(X, t_{n+1}) = u(X, t_n) + 0.5 * (v_prev(X) + v_new(X)) * dt_dimensional.
        """
        v_current = fem.Function(self.v_sol.function_space)
        v_current.interpolate(self.U.sub(self.idx_v))
        v_current.x.array[:] *= self.v_c
        v_current.x.scatter_forward()

        if self.v_prev is None:
            self.v_prev = fem.Function(self.v_sol.function_space)
            self.v_prev.x.array[:] = v_current.x.array

        self.u_sol.x.array[:] += 0.5 * (self.v_prev.x.array + v_current.x.array) * dt_dimensional
        self.u_sol.x.scatter_forward()

        self.v_prev.x.array[:] = v_current.x.array
        self.v_prev.x.scatter_forward()


class BurnProblem(BurnProblem1D):
    """Backward-compatible alias for the existing 1D burn problem."""
