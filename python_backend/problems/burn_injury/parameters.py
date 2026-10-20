"""
Default parameter sets for wound healing simulations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, replace
from typing import Union

import numpy as np

NEWTON_TO_G_CM_DAY2 = 1e5 * (86400**2)  # 1 N = 1e5 g*cm/s^2 * (86400 s/day)^2


@dataclass(frozen=True)
class WoundGeometry1D:
    """
    1D wound placement: a single point on the interval, plus its width.

    Parameters
    ----------
    center : float, optional
        The center position of the wound in cm (default is 5.0).
    length : float, optional
        The length of the wound in cm (default is 4.0).

    Attributes
    ----------
    center : float
        The center position of the wound in cm.
    length : float
        The length of the wound in cm.
    radius : float
        The radius (half-length) of the wound in cm.
    """

    center: float = 5.0  # cm
    length: float = 4.0  # cm

    @property
    def radius(self) -> float:
        """
        Calculate the radius (half-length) of the wound.

        Returns
        -------
        float
            The radius of the wound in cm.
        """
        return self.length / 2.0


@dataclass(frozen=True)
class WoundGeometry2D:
    """
    2D wound placement: a centre point on the plane, plus a characteristic size.

    Parameters
    ----------
    center : tuple of float, optional
        The (x, y) center position of the wound in cm (default is (5.0, 5.0)).
    length : float, optional
        The characteristic size (diameter) of the wound in cm (default is 4.0).
    shape : str or None, optional
        The shape of the wound (default is None).

    Attributes
    ----------
    center : tuple of float
        The (x, y) center position of the wound in cm.
    length : float
        The characteristic size of the wound in cm.
    shape : str or None
        The shape of the wound.
    radius : float
        The radius (half-length) of the wound in cm.
    """

    center: tuple[float, float] = (5.0, 5.0)  # (x, y) cm
    length: float = 4.0  # cm (diameter/characteristic size, matches 1D's `length` semantics)
    shape: str | None = None

    def __post_init__(self) -> None:
        if len(self.center) != 2:
            raise ValueError(f"WoundGeometry2D.center must be an (x, y) pair, got {self.center!r}")

    @property
    def radius(self) -> float:
        """
        Calculate the radius (half-length) of the wound.

        Returns
        -------
        float
            The radius of the wound in cm.
        """
        return self.length / 2.0


WoundGeometry = Union[WoundGeometry1D, WoundGeometry2D]


@dataclass(frozen=True)
class PhysicalParams:
    """
    Physical (dimensional) parameters for the burn injury model.

    Independent parameters are ordinary fields with defaults matching the
    original `get_default_physical_params()`. Dependent parameters (`q`,
    `k_rho`) are computed automatically.

    Attributes
    ----------
    a_c_I : float
        Parameter a_c_I in g/cm^3.
    a_c_II : float
        Parameter a_c_II in g/cm^3.
    a_c_III : float
        Parameter a_c_III in cm^3/g.
    a_c_IV : float
        Parameter a_c_IV in g/cm^3.
    c_eq : float
        Equilibrium concentration of signaling molecule in g/cm^3.
    c_tilde : float
        Initial concentration of signaling molecule inside the wound in g/cm^3.
    D_c : float
        Diffusion coefficient for signaling molecule in cm^2/day.
    D_F : float
        Diffusion coefficient for fibroblasts in cm^5/(cell*day).
    delta_c : float
        Decay rate parameter for signaling molecule in cm^6/(cell*g*day).
    delta_M : float
        Decay rate for myofibroblasts in 1/day.
    delta_N : float
        Decay rate for fibroblasts in 1/day.
    delta_rho : float
        Decay rate for collagen in cm^6/(cell*g*day).
    eta_I : float
        Parameter eta_I (dimensionless).
    eta_II : float
        Parameter eta_II (dimensionless).
    E : float
        Elastic modulus in g^0.5*cm^0.5/day^2. Young's modulus is E*sqrt{rho}.
    k_c : float
        Production rate parameter for signaling molecule in g/(cell*day).
    k_F : float
        Cell differentiation rate parameter in cm^3/(g*day).
    k_rho_max : float
        Maximum collagen production rate parameter.
    kappa_F : float
        Crowding factor in cm^3/cell.
    mu_1 : float
        Shear viscosity in g/(day*cm).
    mu_2 : float
        Bulk viscosity in g/(day*cm).
    M_eq : float
        Equilibrium myofibroblast density in cell/cm^3.
    nu : float
        Poisson's ratio (dimensionless).
    N_eq : float
        Equilibrium fibroblast density in cell/cm^3.
    N_tilde : float
        Initial wound-region fibroblast density in cell/cm^3.
    r_F : float
        Cell division rate in cm^(3q)/(cell^q*day).
    r_F_max : float
        Maximum cell division enhancing factor (dimensionless).
    rho_eq : float
        Equilibrium collagen density in g/cm^3.
    rho_tilde : float
        Initial wound-region collagen density in g/cm^3.
    R : float
        Fixed constant in g/cm^3.
    v_bd : float
        Boundary deformation velocity in cm/day.
    xi : float
        Body force parameter in g^2/(day^2*cell*cm).
    zeta : float
        Rate of morphoelastic change in cm^6/(cell*g*day).
    k_sub : float
        Subcutaneous elastic foundation stiffness in g/(day^2*cm^2).
    c_sub : float
        Subcutaneous viscous foundation damping in g/(day*cm^2).
    beta_h : float
        Hypertrophic collagen outward swelling stress parameter in cm^2/day^2.
    q : float
        Derived exponent parameter (computed automatically).
    k_rho : float
        Derived collagen production parameter (computed automatically).
    """

    # --- Independent parameters ---
    a_c_I: float = 1e-8  # g/cm^3
    a_c_II: float = 1e-8  # g/cm^3
    a_c_III: float = 2.05e8  # cm^3/g
    a_c_IV: float = 1e-9  # g/cm^3
    c_eq: float = 0.0  # g/cm^3
    c_tilde: float = 1e-8  # g/cm^3
    D_c: float = 3.25e-3  # cm^2/day
    D_F: float = 1.2e-6  # cm^5/(cell*day)
    delta_c: float = 5e-4  # cm^6/(cell*g*day)
    delta_M: float = 6e-2  # 1/day
    delta_N: float = 1.9e-2  # 1/day
    delta_rho: float = 6.11e-6  # cm^6/(cell*g*day)
    eta_I: float = 2.0
    eta_II: float = 0.45
    E: float = 320 * NEWTON_TO_G_CM_DAY2  # g^0.5*cm^0.5/day^2
    k_c: float = 3e-13  # g/(cell*day)
    k_F: float = 1.14e7  # cm^3/(g*day)
    k_rho_max: float = 10.0
    kappa_F: float = 1e-6  # cm^3/cell
    mu_1: float = 100.0 * NEWTON_TO_G_CM_DAY2  # g/(day*cm)
    mu_2: float = 100.0 * NEWTON_TO_G_CM_DAY2  # g/(day*cm)
    M_eq: float = 0.0  # cell/cm^3
    nu: float = 4.9e-1
    N_eq: float = 1.5e4  # cell/cm^3
    N_tilde: float = 3.0e3  # cell/cm^3 (wound-region initial value; independent -- see below)
    r_F: float = 1.222  # cm^(3q)/(cell^q*day)
    r_F_max: float = 2.0
    rho_eq: float = 1.25e-1  # g/cm^3
    rho_tilde: float = 1.25e-2  # g/cm^3 (wound-region initial value; independent -- see below)
    R: float = 0.995  # g/cm^3
    v_bd: float = 0.0  # cm/day
    xi: float = 0.044 * NEWTON_TO_G_CM_DAY2  # g^2/(day^2*cell*cm)
    zeta: float = 3.8e2  # cm^6/(cell*g*day)
    k_sub: float = 0.0  # g/(day^2*cm^2) — subcutaneous elastic foundation stiffness
    c_sub: float = 0.0  # g/(day*cm^2) — subcutaneous viscous foundation damping
    beta_h: float = 0.0  # cm^2/day^2 — hypertrophic collagen outward swelling stress per excess collagen density

    # --- Dependent parameters. Never set these directly (see updated()). ---
    # Only q and k_rho are TRUE algebraic dependents: both come from requiring
    # the equilibrium point to actually be a fixed point of the reaction terms
    # (k_rho = delta_rho * rho_eq^2; q solves the N-equation's steady-state
    # condition). They must be recomputed whenever their inputs change, or the
    # model is internally inconsistent.
    #
    # N_tilde/rho_tilde are NOT algebraic dependents, even though the default
    # baseline happens to set them as 0.2*N_eq / 0.1*rho_eq.
    # That ratio is a baseline modelling choice, not a constraint the
    # model requires, the Sobol sweep already varies them independently of
    # N_eq/rho_eq, with their own bounds. So they stay ordinary independent
    # fields above, with literal defaults (not derived in __post_init__).
    q: float = field(init=False)
    k_rho: float = field(init=False)

    def __post_init__(self) -> None:
        """
        Compute dependent parameters after initialization.

        Computes the derived parameters `q` and `k_rho` from the independent
        parameters to ensure model consistency.
        """
        q = (np.log(self.delta_N) - np.log(self.r_F * (1 - self.kappa_F * self.N_eq))) / np.log(self.N_eq)
        object.__setattr__(self, "q", float(q))
        object.__setattr__(self, "k_rho", self.delta_rho * self.rho_eq**2)

    def updated(self, **changes: float) -> "PhysicalParams":
        """
        Return a new instance with updated independent parameters.

        Creates a new PhysicalParams instance with the specified independent
        parameters changed and all dependent parameters (q, k_rho) recomputed
        from scratch.

        Parameters
        ----------
        **changes : float
            Keyword arguments specifying the independent parameters to change
            and their new values.

        Returns
        -------
        PhysicalParams
            A new PhysicalParams instance with updated parameters.

        Raises
        ------
        ValueError
            If attempting to set dependent parameters (q or k_rho) directly.
        """
        dependent_names = {"q", "k_rho"}
        offending = dependent_names & set(changes)
        if offending:
            raise ValueError(
                f"Cannot set dependent parameter(s) {sorted(offending)} directly -- "
                "they are always derived from other parameters. Change the "
                "independent parameter(s) they depend on instead."
            )
        return replace(self, **changes)

    def to_dict(self) -> dict[str, float]:
        """
        Convert parameters to a flat dictionary.

        Returns a JSON-serializable dictionary of all parameters, suitable
        for saving as provenance sidecars.

        Returns
        -------
        dict[str, float]
            Dictionary mapping parameter names to their float values.
        """
        return {k: float(v) for k, v in asdict(self).items()}

    @classmethod
    def field_names(cls) -> tuple[str, ...]:
        """
        Get the names of all fields in the dataclass.

        Returns
        -------
        tuple[str, ...]
            A tuple containing the names of all fields defined in the class.
        """
        return tuple(f.name for f in fields(cls))


def get_default_physical_params() -> PhysicalParams:
    """
    Get default physical parameters for the burn injury model.

    Returns
    -------
    PhysicalParams
        A PhysicalParams instance with default parameter values.
    """
    return PhysicalParams()


@dataclass(frozen=True)
class CharScales:
    """
    Characteristic scales for non-dimensionalization.

    Independent scales (L_c, N_c, rho_c, c_c, t_c, mu_1) define the system.
    Dependent scales (`v_c`, `sigma_c`) are computed automatically in __post_init__.

    Parameters
    ----------
    L_c : float, optional
        Characteristic length scale in cm (default is 1.0).
    N_c : float, optional
        Characteristic fibroblast density scale in cell/cm^3 (default is 1.5e4).
    rho_c : float, optional
        Characteristic collagen density scale in g/cm^3 (default is 1.25e-1).
    c_c : float, optional
        Characteristic signaling molecule concentration scale in g/cm^3 (default is 1e-8).
    t_c : float, optional
        Characteristic time scale in day (default is 1.0 / 1.9e-2).
    mu_1 : float, optional
        Characteristic viscosity scale in g/(day*cm) (default is 100.0 * NEWTON_TO_G_CM_DAY2).

    Attributes
    ----------
    L_c : float
        Characteristic length scale in cm.
    N_c : float
        Characteristic fibroblast density scale in cell/cm^3.
    rho_c : float
        Characteristic collagen density scale in g/cm^3.
    c_c : float
        Characteristic signaling molecule concentration scale in g/cm^3.
    t_c : float
        Characteristic time scale in day.
    mu_1 : float
        Characteristic viscosity scale in g/(day*cm).
    v_c : float
        Characteristic velocity scale in cm/day (computed automatically).
    sigma_c : float
        Characteristic stress scale in g/(day^2*cm) (computed automatically).
    """

    L_c: float = 1.0  # cm
    N_c: float = 1.5e4  # cell/cm^3
    rho_c: float = 1.25e-1  # g/cm^3
    c_c: float = 1e-8  # g/cm^3
    t_c: float = 1.0 / 1.9e-2  # day (1 / delta_N)
    mu_1: float = 100.0 * NEWTON_TO_G_CM_DAY2  # g/(day*cm)

    # --- Dependent scales ---
    v_c: float = field(init=False)  # cm/day
    sigma_c: float = field(init=False)  # g/(day^2*cm)

    def __post_init__(self) -> None:
        """
        Compute dependent scales after initialization.

        Computes the derived scales `v_c` and `sigma_c` from the independent
        scales to ensure consistency.
        """
        object.__setattr__(self, "v_c", float(self.L_c / self.t_c))
        object.__setattr__(self, "sigma_c", float(self.mu_1 / self.t_c))

    @classmethod
    def from_params(cls, physical_params: PhysicalParams, L_c: float = 1.0) -> "CharScales":
        """
        Create characteristic scales from physical parameters.

        Parameters
        ----------
        physical_params : PhysicalParams
            The physical parameters from which to derive the scales.
        L_c : float, optional
            The characteristic length scale in cm (default is 1.0).

        Returns
        -------
        CharScales
            A CharScales instance with scales derived from the physical parameters.
        """
        return cls(
            L_c=L_c,
            N_c=physical_params.N_eq,
            rho_c=physical_params.rho_eq,
            c_c=physical_params.c_tilde,
            t_c=1.0 / physical_params.delta_N,
            mu_1=physical_params.mu_1,
        )

    def updated(self, **changes: float) -> "CharScales":
        """
        Return a new instance with updated scales.

        Creates a new CharScales instance with the specified scales changed
        and all dependent scales (v_c, sigma_c) recomputed from scratch.

        Parameters
        ----------
        **changes : float
            Keyword arguments specifying the scales to change and their new values.

        Returns
        -------
        CharScales
            A new CharScales instance with updated scales.

        Raises
        ------
        ValueError
            If attempting to set dependent scales (v_c or sigma_c) directly.
        """
        dependent_names = {"v_c", "sigma_c"}
        offending = dependent_names & set(changes)
        if offending:
            raise ValueError(
                f"Cannot set dependent scale(s) {sorted(offending)} directly -- "
                "they are always derived from other scales."
            )
        return replace(self, **changes)

    def to_dict(self) -> dict[str, float]:
        """
        Convert scales to a flat dictionary.

        Returns a JSON-serializable dictionary of all scales.

        Returns
        -------
        dict[str, float]
            Dictionary mapping scale names to their float values.
        """
        return {k: float(v) for k, v in asdict(self).items()}

    @classmethod
    def field_names(cls) -> tuple[str, ...]:
        """
        Get the names of all fields in the dataclass.

        Returns
        -------
        tuple[str, ...]
            A tuple containing the names of all fields defined in the class.
        """
        return tuple(f.name for f in fields(cls))


def get_default_char_scales(physical_params: PhysicalParams | None = None) -> CharScales:
    """
    Get default characteristic scales for non-dimensionalization.

    Parameters
    ----------
    physical_params : PhysicalParams or None, optional
        Physical parameters from which to derive scales. If None, returns
        default scales (default is None).

    Returns
    -------
    CharScales
        A CharScales instance with default or derived scales.

    Raises
    ------
    TypeError
        If physical_params is not None or a PhysicalParams instance.
    """
    if physical_params is None:
        return CharScales()
    if isinstance(physical_params, PhysicalParams):
        return CharScales.from_params(physical_params)
    raise TypeError(f"Expected PhysicalParams or None, got {type(physical_params).__name__}")


def get_standard_adaptive_params(char_scales: CharScales, dim: int) -> dict[str, float]:
    """
    Get standard adaptive time-stepping parameters.

    Parameters
    ----------
    char_scales : CharScales
        Characteristic scales for non-dimensionalization.
    dim : int
        The spatial dimension of the problem (1 or 2).

    Returns
    -------
    dict[str, float]
        Dictionary containing adaptive time-stepping parameters:
        - initial_dt : Initial time step size (non-dimensional)
        - dt_min_value : Minimum allowed time step size (non-dimensional)
        - dt_max_value : Maximum allowed time step size (non-dimensional)
        - ts_atol : Absolute tolerance for time stepper
        - ts_rtol : Relative tolerance for time stepper

    Raises
    ------
    TypeError
        If char_scales is not a CharScales instance.
    """
    if not isinstance(char_scales, CharScales):
        raise TypeError(f"Expected CharScales, got {type(char_scales).__name__}")
    t_c = char_scales.t_c
    adaptive_params = {
        "initial_dt": 5e-2 / t_c,
        "dt_min_value": 1e-3 / t_c,
        "dt_max_value": 10.0 / t_c,
    }
    if dim == 1:
        adaptive_params["ts_atol"] = 1e-7
        adaptive_params["ts_rtol"] = 1e-4
    elif dim == 2:
        adaptive_params["ts_atol"] = 1e-5
        adaptive_params["ts_rtol"] = 1e-2
    return adaptive_params
