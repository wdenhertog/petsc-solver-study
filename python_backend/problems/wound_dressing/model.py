"""
Wound dressing model
Nondimensional governing equation:

    dS(p)/dt_tilde - div(K_tilde * k_r(p) * grad(p + Bo * z_tilde)) = 0

Boundary conditions (all Neumann):

    Gamma_wound   : flux =  Q_tilde(t_tilde)   [inward exudate source]
    Gamma_healthy : flux =  0                   [impermeable healthy skin]
    Gamma_top     : flux =  h_tilde*(S - S_eq)  [Robin evaporation]
    Gamma_sides   : flux =  0                   [sealed dressing edges]

Usage:
    problem = WoundDressingProblem(nd)
    problem = WoundDressingProblem.from_dimensional(p)

    solver  = WoundDressingSolver(problem, ap)
    solver.run()
"""

from pathlib import Path

import numpy as np
from dolfinx import fem, io, log, mesh
from dolfinx.fem.petsc import (
    assemble_matrix,
    assemble_vector,
    create_matrix,
    create_vector,
)
from mpi4py import MPI
from numpy.typing import NDArray
from petsc4py import PETSc
from ufl import (
    Measure,
    SpatialCoordinate,
    TestFunction,
    TrialFunction,
    as_vector,
    conditional,
    derivative,
    dx,
    grad,
    gt,
    inner,
    lt,
    sqrt,
)

from .parameters import get_char_scales, get_nondim_params

# Facet tag constants
TAG_WOUND = 1
TAG_HEALTHY = 2
TAG_TOP = 3
# Sides are untagged — zero flux by default (homogeneous Neumann)


class WoundDressingProblem:
    def __init__(
        self,
        nd: dict[str, float],
        nx: int = 40,
        ny: int = 40,
        nz: int = 6,
    ):
        self.nd = nd
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.rank

        self._build_mesh(nx, ny, nz)
        self._build_function_spaces()
        self._tag_boundaries()
        self._tag_cells()
        self._set_initial_condition()
        self._build_weak_form()

    @classmethod
    def from_dimensional(
        cls, p: dict[str, float], nx: int = 40, ny: int = 40, nz: int = 6
    ) -> "WoundDressingProblem":
        s = get_char_scales(p)
        nd = get_nondim_params(p, s)
        return cls(nd, nx, ny, nz)

    def _build_mesh(self, nx: int, ny: int, nz: int) -> None:
        nd = self.nd
        self.domain = mesh.create_box(
            self.comm,
            [np.array([0.0, 0.0, 0.0]), np.array([nd["L_tilde"], nd["W_tilde"], 1.0])],
            [nx, ny, nz],
            mesh.CellType.tetrahedron,
        )

    def _build_function_spaces(self) -> None:
        self.V = fem.functionspace(self.domain, ("Lagrange", 1))
        self.p = fem.Function(self.V, name="Pressure")  # current timestep
        self.p_dot = fem.Function(self.V, name="PressureDot")
        self.dp = TrialFunction(self.V)
        self.S_out = fem.Function(self.V, name="Saturation")
        self.v = TestFunction(self.V)
        self.alpha_const = fem.Constant(self.domain, PETSc.ScalarType(1.0))

        self.V_DG0 = fem.functionspace(
            self.domain, ("DG", 0)
        )  # piecewise constant permeability
        self.anisotropy_field = fem.Function(self.V_DG0, name="anisotropy")

    def _tag_boundaries(self) -> None:
        nd = self.nd
        EPS = 1e-10
        fdim = self.domain.topology.dim - 1
        self.domain.topology.create_connectivity(fdim, self.domain.topology.dim)

        # Get bounds of wounded area
        lw = nd["lw_tilde"]
        Lx = nd["L_tilde"]
        Ly = nd["W_tilde"]
        xmin = (Lx - lw) / 2
        xmax = (Lx + lw) / 2
        ymin = (Ly - lw) / 2
        ymax = (Ly + lw) / 2

        def bottom_wound(x: NDArray[np.float64]) -> NDArray[np.bool_]:
            """Check if a point is within the wound region."""
            on_bottom = np.isclose(x[2], 0.0, atol=EPS)
            in_x = (x[0] >= xmin - EPS) & (x[0] <= xmax + EPS)
            in_y = (x[1] >= ymin - EPS) & (x[1] <= ymax + EPS)
            return on_bottom & in_x & in_y

        def bottom_healthy(x: NDArray[np.float64]) -> NDArray[np.bool_]:
            """Everything outside the wound region is healthy skin."""
            on_bottom = np.isclose(x[2], 0.0, atol=EPS)
            in_x = (x[0] >= xmin - EPS) & (x[0] <= xmax + EPS)
            in_y = (x[1] >= ymin - EPS) & (x[1] <= ymax + EPS)
            return on_bottom & ~(in_x & in_y)

        def top_surface(x: NDArray[np.float64]) -> NDArray[np.bool_]:
            """The top of the box is the dressing surface."""
            return np.isclose(x[2], 1.0, atol=EPS)

        fi_wound = mesh.locate_entities_boundary(self.domain, fdim, bottom_wound)
        fi_healthy = mesh.locate_entities_boundary(self.domain, fdim, bottom_healthy)
        fi_top = mesh.locate_entities_boundary(self.domain, fdim, top_surface)

        all_indices = np.concatenate([fi_wound, fi_healthy, fi_top])
        all_tags = np.concatenate(
            [
                np.full(len(fi_wound), TAG_WOUND, dtype=np.int32),
                np.full(len(fi_healthy), TAG_HEALTHY, dtype=np.int32),
                np.full(len(fi_top), TAG_TOP, dtype=np.int32),
            ]
        )

        sort_idx = np.argsort(all_indices)
        self.facet_tags = mesh.meshtags(
            self.domain, fdim, all_indices[sort_idx], all_tags[sort_idx]
        )

        self.ds = Measure("ds", domain=self.domain, subdomain_data=self.facet_tags)

    def _tag_cells(self) -> None:
        """Tag 3D cells into distinct layers and assign material properties to each layer."""

        z_contact = 0.1
        z_transfer = 0.5

        def layer_contact(x: NDArray[np.float64]) -> NDArray[np.bool_]:
            return x[2] <= z_contact + 1e-10

        def layer_transfer(x: NDArray[np.float64]) -> NDArray[np.bool_]:
            return (x[2] > z_contact - 1e-10) & (x[2] <= z_transfer + 1e-10)

        def layer_spreading(x: NDArray[np.float64]) -> NDArray[np.bool_]:
            return x[2] > z_transfer - 1e-10

        tdim = self.domain.topology.dim
        cells_contact = mesh.locate_entities(self.domain, tdim, layer_contact)
        cells_transfer = mesh.locate_entities(self.domain, tdim, layer_transfer)
        cells_spreading = mesh.locate_entities(self.domain, tdim, layer_spreading)

        # --- Assign the physical anisotropy values ---
        # k_lateral / k_vertical
        a_contact = 0.1  # High resistance laterally, forces fluid strictly upwards
        a_transfer = 0.05  # Continues pulling fluid vertically
        a_spreading = (
            10.0  # High lateral permeability, rapidly spreads fluid at the top
        )

        # Because DG0 DoFs map 1:1 with local cell indices, we can assign directly:
        self.anisotropy_field.x.array[cells_contact] = a_contact
        self.anisotropy_field.x.array[cells_transfer] = a_transfer
        self.anisotropy_field.x.array[cells_spreading] = a_spreading

    def _set_initial_condition(self) -> None:
        S_init = self.nd.get("S_init", self.nd["S_wr"] + 0.01)
        n = self.nd["n_vG"]
        m = 1.0 - 1.0 / n
        Se_init = (S_init - self.nd["S_wr"]) / (1 - self.nd["S_wr"])

        p_init = -((Se_init ** (-1.0 / m) - 1.0) ** (1.0 / n))
        self.p.x.array[:] = p_init

    def _build_weak_form(self) -> None:
        nd = self.nd
        p, p_dot, v = self.p, self.p_dot, self.v
        n_vG = nd["n_vG"]
        m_vG = 1.0 - 1.0 / n_vG
        S_wr = nd["S_wr"]

        self.Q_w = fem.Constant(self.domain, PETSc.ScalarType(nd["Q0_tilde"]))
        h_tilde = fem.Constant(self.domain, PETSc.ScalarType(nd["h_tilde"]))
        S_eq = fem.Constant(self.domain, PETSc.ScalarType(nd["S_eq"]))
        Bo = nd["Bo"]
        anisotropy_field = self.anisotropy_field
        z_coord = SpatialCoordinate(self.domain)[2]

        def S_vG(p_expr):
            # Use a non-negative pressure magnitude to avoid invalid fractional powers
            # on intermediate Newton iterates where p_expr may become positive.
            p_mag = conditional(lt(p_expr, 0.0), -p_expr, 0.0)
            Se = (1.0 + p_mag**n_vG) ** (-m_vG)
            return S_wr + (1.0 - S_wr) * Se

        def k_r(p_expr):
            """Calculates relative permeability."""
            p_mag = conditional(lt(p_expr, 0.0), -p_expr, 0.0)
            Se_raw = (1.0 + p_mag**n_vG) ** (-m_vG)
            eps = 1e-4
            Se = conditional(
                lt(Se_raw, eps),
                eps,
                conditional(gt(Se_raw, 1.0 - eps), 1.0 - eps, Se_raw),
            )
            return sqrt(Se) * (1.0 - (1.0 - Se ** (1.0 / m_vG)) ** m_vG) ** 2

        def darcy_flux(p_expr: fem.Function) -> fem.Function:
            """
            Nondimensional Darcy flux vector (Pressure-Based).
            u_tilde = K_tilde * k_r(p) * grad(p + Bo * z_tilde)
            K_tilde is diagonal (anisotropy, anisotropy, 1)
            """
            head = p_expr + Bo * z_coord
            g_h = grad(head)
            kr = k_r(p_expr)

            return kr * as_vector(
                [anisotropy_field * g_h[0], anisotropy_field * g_h[1], g_h[2]]
            )

        # Weak form
        # dS/dt_tilde - div(u_tilde) = 0
        # Integration by parts and applying Euler backwards for time integration gives:
        # (1/dt)*(S-S_n)*v dx + u_tilde \cdot grad(v) dx - BC terms = 0
        #
        # BC terms are all Neumann conditions:
        # Wound: u_tilde \cdot n = Q_tilde -> -Q_w*v*ds(wound)
        # Top: u_tilde \cdot n = h_tilde*(S-S_eq) -> -h_tilde*(S-S_eq)*v*ds(top)
        # Healthy and sides: u_tilde \cdot n = 0 -> no contribution

        # flux = darcy_flux(s)
        S = S_vG(self.p)
        dS_dt = derivative(S, p, p_dot)

        dx_c = dx(metadata={"quadrature_degree": 4})
        ds_c = self.ds(metadata={"quadrature_degree": 4})
        dx_lumped = dx(metadata={"quadrature_rule": "vertex"})

        self.S_expr = fem.Expression(S, self.V.element.interpolation_points)
        flux = darcy_flux(self.p)

        evap = conditional(gt(S, S_eq), h_tilde * (S - S_eq), 0.0)
        self.F = (
            dS_dt * v * dx_lumped
            + inner(flux, grad(v)) * dx_c
            - self.Q_w * v * ds_c(TAG_WOUND)
            + evap * v * ds_c(TAG_TOP)
        )
        self.J = derivative(self.F, self.p, self.dp) + self.alpha_const * derivative(
            self.F, self.p_dot, self.dp
        )
        self.F_compiled = fem.form(self.F)
        self.J_compiled = fem.form(self.J)
        self.bcs = []

    def update_flux(self, t_tilde: float) -> None:
        """
        Update the time-dependent wound flux Q_tilde(t_tilde)
        Exponential decay from Q0_tilde to Q_inf_tilde with timescale tau_tilde.
        """
        nd = self.nd
        Q = (nd["Q0_tilde"] - nd["Q_inf_tilde"]) * np.exp(
            -t_tilde / nd["tau_tilde"]
        ) + nd["Q_inf_tilde"]
        self.Q_w.value = Q


class _DressingTSContext:
    def __init__(self, problem: WoundDressingProblem):
        self.problem = problem

    def evalIFunction(self, ts, t, U, U_dot, F_vec):
        U.copy(self.problem.p.x.petsc_vec)
        U_dot.copy(self.problem.p_dot.x.petsc_vec)
        self.problem.p.x.scatter_forward()
        self.problem.p_dot.x.scatter_forward()
        self.problem.update_flux(t)

        with F_vec.localForm() as loc:
            loc.set(0.0)

        assemble_vector(F_vec, self.problem.F_compiled)
        F_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    def evalIJacobian(self, ts, t, U, U_dot, shift, J_mat, P_mat):
        self.problem.alpha_const.value = shift
        J_mat.zeroEntries()

        assemble_matrix(J_mat, self.problem.J_compiled, bcs=self.problem.bcs)
        J_mat.assemble()


class WoundDressingSolver:
    def __init__(
        self,
        problem: WoundDressingProblem,
        ap: dict[str, float],
        output_path: Path | None,
        T_tilde: float,
        t_star: float,  # time scale
        verb: int = 0,
    ):
        self.problem = problem
        self.ap = ap
        self.output_path = output_path
        self.T_tilde = T_tilde
        self.t_star = t_star
        self.verb = verb

        # Populated by run() after ts.solve() -- real solver stats pulled
        # off the TS, same convention as solve_transient_ts() in
        # python_backend/runner.py. Left at these defaults if run() is
        # never called or ts.solve() raises before completing.
        self.n_timesteps = 0
        self.n_ksp_iterations_total = 0
        self.n_snes_iterations_total = 0
        self.final_time = 0.0
        self.converged_reason = 0

        if problem.rank != 0:
            log.set_log_level(log.LogLevel.WARNING)

    def run(self) -> None:
        """Run the adaptive time loop, optionally writing VTX output."""
        prob = self.problem
        rank = prob.rank
        verb = self.verb
        appctx = _DressingTSContext(prob)

        F_vec = create_vector(prob.V)
        J_mat = create_matrix(prob.J_compiled)

        ts = PETSc.TS().create(prob.comm)
        ts.setProblemType(PETSc.TS.ProblemType.NONLINEAR)
        ts.setEquationType(PETSc.TS.EquationType.IMPLICIT)
        ts.setIFunction(appctx.evalIFunction, F_vec)
        ts.setIJacobian(appctx.evalIJacobian, J_mat)

        snes = ts.getSNES()
        snes.setTolerances(atol=1e-6, rtol=1e-5, stol=1e-8, max_it=15)
        ksp = snes.getKSP()
        ksp.setType("cg")
        pc = ksp.getPC()
        pc.setType("hypre")

        ts.setType(PETSc.TS.Type.BDF)
        PETSc.Options().setValue("-ts_bdf_order", 2)
        PETSc.Options().setValue("-ts_adapt_type", "basic")
        ts.setTimeStep(self.ap["initial_dt"])
        PETSc.Options().setValue("-ts_adapt_dt_min", self.ap["dt_min"])
        PETSc.Options().setValue("-ts_adapt_dt_max", self.ap["dt_max"])
        ts.setMaxSNESFailures(-1)
        ts.setMaxTime(self.T_tilde)
        ts.setExactFinalTime(PETSc.TS.ExactFinalTime.MATCHSTEP)

        if verb == 1:
            PETSc.Options().setValue("-ts_monitor", None)
            PETSc.Options().setValue("-snes_monitor", None)
            PETSc.Options().setValue("-snes_converged_reason", None)
            PETSc.Options().setValue("-ts_view", None)

        ts.setFromOptions()

        local_num_cells = prob.domain.topology.index_map(
            prob.domain.topology.dim
        ).size_global
        local_num_vertices = prob.domain.topology.index_map(0).size_global

        if rank == 0 and verb == 1:
            print("=========================================")
            print("Mesh Topography Metrics:")
            print(f"  Total Elements: {local_num_cells:,}")
            print(f"  Total Nodes: {local_num_vertices:,}")
            print("=========================================")

        if rank == 0 and verb == 1:
            print(f"Starting simulation: T_tilde={self.T_tilde:.4e}", flush=True)

        if self.output_path is not None:
            with io.VTXWriter(
                prob.comm, self.output_path, [prob.S_out], engine="BP5"
            ) as vtx:
                prob.S_out.interpolate(prob.S_expr)
                vtx.write(0.0)

                def monitor(ts, step, t, U):
                    U.copy(prob.p.x.petsc_vec)
                    prob.p.x.scatter_forward()
                    prob.S_out.interpolate(prob.S_expr)

                    is_final = t >= self.T_tilde - 1e-12
                    if step % self.ap["write_every"] == 0 or is_final:
                        vtx.write(t * self.t_star)

                    if rank == 0 and verb == 1 and step % 50 == 0:
                        pct = 100 * t / self.T_tilde
                        print(
                            f"[{pct:6.2f}%] step={step} t={t:.4e} dt={ts.getTimeStep():.2e}",
                            flush=True,
                        )

                ts.setMonitor(monitor)
                ts.solve(prob.p.x.petsc_vec)
                self._capture_ts_stats(ts)
        else:

            def monitor(ts, step, t, U):
                del U
                if rank == 0 and verb == 1 and step % 50 == 0:
                    pct = 100 * t / self.T_tilde
                    print(
                        f"[{pct:6.2f}%] step={step} t={t:.4e} dt={ts.getTimeStep():.2e}",
                        flush=True,
                    )

            ts.setMonitor(monitor)
            ts.solve(prob.p.x.petsc_vec)
            self._capture_ts_stats(ts)

    def _capture_ts_stats(self, ts) -> None:
        """Pull cumulative solve stats off the TS after solve() returns.
        ts.getKSPIterations()/getSNESIterations() are cumulative totals
        across the whole adaptive time loop, not per-step -- same
        quantities solve_transient_ts() in runner.py reports for the
        heat/wave backends, so results stay comparable across problems."""
        self.n_timesteps = ts.getStepNumber()
        self.n_ksp_iterations_total = ts.getKSPIterations()
        self.n_snes_iterations_total = ts.getSNESIterations()
        self.final_time = ts.getTime()
        self.converged_reason = int(ts.getConvergedReason())
