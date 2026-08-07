"""Default parameters for the modeling of wound dressings"""


def get_default_physical_params() -> dict[str, float]:
    return {
        # geometry
        "H": 0.005,  # dressing thickness [m]
        "L": 0.10,  # dressing length [m]
        "W": 0.10,  # dressing width [m]
        "wound_size": 0.04,  # wound side length [m]
        # material
        "phi": 0.85,  # porosity [-]
        "kappa_z": 1e-14,  # vertical permeability [m²]
        "kappa_xy": 1e-13,  # horizontal permeability [m²]
        "mu": 5e-3,  # dynamic viscosity [Pa·s]
        "rho": 1000.0,  # density [kg/m³]
        "g": 9.81,  # gravitational acceleration [m/s²]
        "alpha_vG": 1e-4,  # van Genuchten alpha [Pa⁻¹]
        "n_vG": 1.8,  # van Genuchten n [-]
        "S_wr": 0.05,  # residual saturation [-]
        # boundary conditions
        "Q0": 1e-7,  # initial wound flux [m/s]
        "Q_inf": 5e-9,  # chronic baseline flux [m/s]
        "tau": 2 * 86400.0,  # decay timescale [s]
        "h_m": 1e-7,  # evaporative coefficient [m/s]
        "S_eq": 0.5,  # equilibrium saturation [-]
    }


def get_char_scales(p: dict[str, float]) -> dict[str, float]:
    t_star = p["phi"] * p["alpha_vG"] * p["mu"] * p["H"] ** 2 / p["kappa_z"]
    u_star = p["kappa_z"] / (p["alpha_vG"] * p["mu"] * p["H"])  # reference velocity
    return {
        "H": p["H"],  # length scale [m]
        "P": 1 / p["alpha_vG"],  # pressure scale [Pa]
        "t_star": t_star,  # time scale [s]
        "u_star": u_star,  # flux scale [m/s]
    }


def get_nondim_params(p: dict[str, float], s: dict[str, float]) -> dict[str, float]:
    return {
        # geometry
        "L_tilde": p["L"] / p["H"],
        "W_tilde": p["W"] / p["H"],
        "lw_tilde": p["wound_size"] / p["H"],
        # physics
        "Bo": p["alpha_vG"] * p["rho"] * p["g"] * p["H"],
        "anisotropy": p["kappa_xy"] / p["kappa_z"],
        "n_vG": p["n_vG"],
        "S_wr": p["S_wr"],
        # boundary conditions
        "Q0_tilde": p["Q0"] / s["u_star"],
        "Q_inf_tilde": p["Q_inf"] / s["u_star"],
        "tau_tilde": p["tau"] / s["t_star"],
        "h_tilde": p["h_m"] / s["u_star"],
        "S_eq": p["S_eq"],
        "S_init": p.get("S_init", p["S_wr"] + 0.01),
    }


def get_adaptive_params() -> dict[str, float]:
    return {
        # Time stepping is performed in nondimensional time.
        "initial_dt": 1e-5,
        "dt_min": 1e-8,
        "dt_max": 1,
        "factor_increase": 1.3,
        "factor_decrease": 0.5,
        "factor_fail": 0.2,
        "newton_its_lo": 3,
        "newton_its_hi": 7,
        "write_every": 1,
    }


# Convenience functions
def get_default_char_scales() -> dict[str, float]:
    return get_char_scales(get_default_physical_params())


def get_default_nondim_params() -> dict[str, float]:
    return get_nondim_params(get_default_physical_params(), get_default_char_scales())


def get_default_adaptive_params() -> dict[str, float]:
    return get_adaptive_params()
