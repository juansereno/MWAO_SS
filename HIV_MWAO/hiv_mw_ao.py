"""
=========================================================================
Mode-Wise Adaptive Observer for the HIV mutation model (4 genotypes)
=========================================================================
Python port of ``MW_AO_HIV_main.m``.

Reference: Sereno & Hernandez-Vargas, "Adaptive Estimators for a class of
           Autonomous Nonlinear Switched Systems", bioRxiv 2026.

Architecture:
  - 3 mode observers Pi_0, Pi_1, Pi_2 (one per drug therapy mode 0/1/2).
      sigma = 0 => no therapy
      sigma = 1 => drug therapy 1
      sigma = 2 => drug therapy 2
  - Each observer is a FULL 14-state adaptive observer
    (state-vector form [Ti1 Mi1 V1 | ... | T M]).
  - Each observer estimates theta_l = [KT_{1,l}; KT_{2,l}; KT_{3,l}; KT_{4,l}].
  - E-scaling factor: zhat = xhat/E, yhat = y/E.
  - At every active time interval only Pi_{sigma(t)} integrates; the other
    two FREEZE.
  - scipy ``solve_ivp`` (RK45, the equivalent of MATLAB ode45) for both plant
    and observers.

Output measurements:   y = [T; V1; V2; V3; V4]   (5x1)
"""

from __future__ import annotations

import time

import numpy as np
from scipy.integrate import solve_ivp

from params import Params, NG, NX, NP, NY, NS_MODES
from hiv_plant import hiv_plant
from get_drug_eff import get_drug_eff
from get_treatment import get_treatment
from combined_rhs import combined_rhs


def run(verbose: bool = True):
    """Run the full two-phase simulation and return a results dictionary."""
    # =====================================================================
    # HIV Model Parameters
    # =====================================================================
    p = Params()

    ng, nx, np_, ny, ns_modes = NG, NX, NP, NY, NS_MODES

    # =====================================================================
    # Simulation timing
    # =====================================================================
    dt = 1.0          # plant time step [days]
    Ts = 1.0          # observer sampling period [days]
    Tobs = 3 * 365    # observer start time [days]
    Tcont = 4 * 365   # treatment start time [days]
    tf = 6 * 365      # final simulation time [days]
    dts = 1 * 365     # SWATCH switching period [days]

    # --- solver tolerances (mirror odeset) -------------------------------
    abstol_plant = 1e-4 * np.ones(14)
    abstol_observer = 1e-10 * np.ones(14 + 14 * 14 + 14 * 4 + 4 * 4 + 4)
    rtol = 1e-6
    atol_plant = abstol_plant
    atol_combined = np.concatenate([abstol_plant, abstol_observer])

    # =====================================================================
    # Mode-wise AO tuning (paper notation)
    # =====================================================================
    E = 1e8                              # state-scaling factor
    rho_x = np.array([10.0, 10.0, 10.0])   # S decay rate
    rho_theta = np.array([1.0, 4.0, 8.0])  # Gamma decay rate
    Q = 1e1 * np.eye(ny)                 # output-error weighting

    # Initial AO matrices (same for every mode at observer start)
    S0 = 1e3 * np.eye(nx)
    Lambda0 = np.zeros((nx, np_))   # Assumption 3: Lambda(t_k)=0
    Gamma0 = 1e3 * np.eye(np_)

    # =====================================================================
    # Plant initial conditions
    #   x = [Ti1 Mi1 V1 | Ti2 Mi2 V2 | Ti3 Mi3 V3 | Ti4 Mi4 V4 | T M]
    # =====================================================================
    x0 = np.array([0, 0, 0.001,    # g1
                   0, 0, 0,        # g2
                   0, 0, 0,        # g3
                   0, 0, 0,        # g4
                   1000, 200], dtype=float)  # T, M

    # =====================================================================
    # Phase 1 - Pre-simulation from t=0 to t=Tobs (free virus evolution)
    # =====================================================================
    tPre = np.arange(0, Tobs + dt, dt)   # storage grid [days]
    Npre = tPre.size - 1

    t0 = time.perf_counter()
    sol_pre = solve_ivp(lambda t, xp: hiv_plant(xp, p, 0),
                        [tPre[0], tPre[-1]], x0, method="RK45",
                        t_eval=tPre, rtol=rtol, atol=atol_plant)
    xPre = sol_pre.y                     # nx x (Npre+1)
    xp = xPre[:, -1].copy()              # state at t = Tobs
    if verbose:
        print(f"Pre-simulation complete (0 to {Tobs} days) in "
              f"{time.perf_counter() - t0:.2f} s.")

    # =====================================================================
    # Phase 2 - Initialise the MW AO at t = Tobs (free V evolution)
    # =====================================================================
    x_plant = xp.copy()

    # Offset factors applied at the FIRST activation of every observer
    obs_factors = np.array([1.20, 0.80, 0.01,    # Ti1, Mi1, V1
                            1.20, 0.80, 0.01,    # Ti2, Mi2, V2
                            1.20, 0.80, 0.01,    # Ti3, Mi3, V3
                            1.20, 0.80, 0.01,    # Ti4, Mi4, V4
                            1.20, 1.05])         # T, M

    zhat0 = (obs_factors * x_plant) / E          # scaled initial state estimate

    # Pack each mode observer block (length = 14 + 196 + 56 + 16 + 4 = 286)
    state_len = nx + nx * nx + nx * np_ + np_ * np_ + np_
    mode_state = np.zeros((state_len, ns_modes))
    KT_obs_Factors = np.array([1.3, 1.3, 1.3, 1.3])
    mode_Factor = np.array([1.4, 1.2, 1.0])
    KT_true_per_mode = np.zeros((np_, ns_modes))
    for m in range(ns_modes):
        u_m = m                                      # mode value (0,1,2)
        nT_m, _ = get_drug_eff(u_m, p, ng)
        KT_mode = (1 - nT_m) * p.bT                  # true KT vector for mode u_m
        theta_hat0 = mode_Factor[m] * KT_obs_Factors * KT_mode
        KT_true_per_mode[:, m] = KT_mode

        # MATLAB column-major flatten -> order='F'
        mode_state[:, m] = np.concatenate([
            zhat0,
            S0.flatten(order="F"),
            Lambda0.flatten(order="F"),
            Gamma0.flatten(order="F"),
            theta_hat0,
        ])

    # =====================================================================
    # Phase 2 - Pre-allocate storage
    # =====================================================================
    Nsteps = round((tf - Tobs) / Ts)
    Nplant = round((tf - Tobs) / dt)

    xPlant = np.full((nx, Nplant + 1), np.nan)
    xPlant[:, 0] = x_plant

    xhatArray = np.full((nx, Nsteps + 1), np.nan)        # active-observer estimate
    KThatArray = np.full((np_, Nsteps + 1), np.nan)      # active-observer theta_hat
    KThatAll = np.full((np_, ns_modes, Nsteps + 1), np.nan)
    # Active-observer filter Lambda (14x4) at every step -- used to reconstruct
    # the information matrix N(t) = Lambda' C' Q C Lambda for the PE validation.
    LambdaActive = np.full((nx, np_, Nsteps + 1), np.nan)

    # Slice of the packed observer state holding vec(Lambda) (column-major).
    lam_lo, lam_hi = nx + nx * nx, nx + nx * nx + nx * np_   # 210 .. 266

    sigma_arr = np.zeros(Nsteps + 1)
    KTtrue_arr = np.full((np_, Nsteps + 1), np.nan)

    # Initial sample (t = Tobs)
    u_init = get_treatment(Tobs, Tcont, dts)
    ell_init = u_init                                    # mode index 0/1/2
    xhatArray[:, 0] = E * mode_state[:nx, ell_init]
    KThatArray[:, 0] = mode_state[-np_:, ell_init]
    for m in range(ns_modes):
        KThatAll[:, m, 0] = mode_state[-np_:, m]
    LambdaActive[:, :, 0] = mode_state[lam_lo:lam_hi, ell_init].reshape(nx, np_, order="F")
    sigma_arr[0] = u_init
    nT_init, _ = get_drug_eff(u_init, p, ng)
    KTtrue_arr[:, 0] = (1 - nT_init) * p.bT

    index = 0    # last filled column in observer arrays
    ip = 0       # last filled column in xPlant (initial state lives in col 0)
    n_per_Ts = max(1, round(Ts / dt))


    # =====================================================================
    # Phase 2 - Main loop (Tobs -> tf)
    # =====================================================================
    if verbose:
        print(f"Running mode-wise AO from {Tobs} to {tf} days ...")
    t0 = time.perf_counter()

    # t_k = Tobs+Ts, Tobs+2Ts, ..., tf
    t_grid = np.arange(Tobs + Ts, tf + Ts / 2, Ts)
    for t_k in t_grid:
        u_k = get_treatment(t_k - Ts, Tcont, dts)   # active mode 0/1/2
        ell = u_k                                    # mode index 0/1/2

        # --- Combined initial state for this Ts window -------------------
        y0_comb = np.concatenate([x_plant, mode_state[:, ell]])

        def odefun(t, y, u_k=u_k):
            return combined_rhs(y, u_k, p, E, rho_x, rho_theta, Q, nx)

        # --- Integrate over [t_k - Ts, t_k] -----------------------------
        if n_per_Ts >= 2:
            tspan = np.linspace(t_k - Ts, t_k, n_per_Ts + 1)
            sol = solve_ivp(odefun, [t_k - Ts, t_k], y0_comb, method="RK45",
                            t_eval=tspan, rtol=rtol, atol=atol_combined)
            x_plant_trace = sol.y[:nx, 1:]              # nx x n_per_Ts
        else:
            sol = solve_ivp(odefun, [t_k - Ts, t_k], y0_comb, method="RK45",
                            rtol=rtol, atol=atol_combined)
            x_plant_trace = sol.y[:nx, -1:].copy()      # nx x 1
        s_end = sol.y[nx:, -1].copy()                   # final observer state

        # --- Symmetrise S to suppress floating-point drift --------------
        S_block = s_end[nx:nx + nx * nx].reshape(nx, nx, order="F")
        S_block = 0.5 * (S_block + S_block.T)
        s_end[nx:nx + nx * nx] = S_block.flatten(order="F")

        # --- Commit updates ---------------------------------------------
        x_plant = x_plant_trace[:, -1].copy()
        mode_state[:, ell] = s_end

        # all the MW-AO share the same state estimations
        for m in range(ns_modes):
            if m != ell:
                mode_state[:nx, m] = s_end[:nx]

        # --- Store plant samples on the dt grid -------------------------
        xPlant[:, ip + 1: ip + 1 + n_per_Ts] = x_plant_trace
        ip += n_per_Ts

        # --- Store observer-grid samples --------------------------------
        index += 1
        sigma_arr[index] = u_k
        nT_now, _ = get_drug_eff(u_k, p, ng)
        KTtrue_arr[:, index] = (1 - nT_now) * p.bT

        xhatArray[:, index] = E * mode_state[:nx, ell]
        KThatArray[:, index] = mode_state[-np_:, ell]
        for m in range(ns_modes):
            KThatAll[:, m, index] = mode_state[-np_:, m]
        # active-mode filter Lambda (just committed in mode_state[:, ell])
        LambdaActive[:, :, index] = mode_state[lam_lo:lam_hi, ell].reshape(nx, np_, order="F")


    if verbose:
        print(f"Mode-wise AO done in {(time.perf_counter() - t0) / 60:.2f} m.")

    # =====================================================================
    # Root-Mean-Squared-Log-Error metrics (active-observer estimates)
    # =====================================================================
    tp_full = np.concatenate([tPre, Tobs + np.arange(1, Nplant + 1) * dt])
    xp_full = np.concatenate([xPre, xPlant[:, 1:]], axis=1)

    to = Tobs + np.arange(Nsteps + 1) * Ts

    # Plant samples at the observer's Ts grid (nearest plant index)
    idx_obs_in_full = np.array([
        np.argmax(np.abs(tp_full - tt) < dt / 2) for tt in to
    ])
    xp_at_obs = xp_full[:, idx_obs_in_full]

    Ne = to.size

    def rmsle(a, b):
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        return (1.0 / Ne) * np.sqrt(
            np.sum((np.log(np.maximum(a, 0) + 1) - np.log(np.maximum(b, 0) + 1)) ** 2)
        )

    KT_RMSLE = np.zeros(np_)
    for i in range(np_):
        KT_RMSLE[i] = rmsle(KTtrue_arr[i, :], KThatArray[i, :])

    V_RMSLE = np.zeros(ng)
    Ti_RMSLE = np.zeros(ng)
    Mi_RMSLE = np.zeros(ng)
    for i in range(ng):
        # MATLAB 1-indexed rows 3*i, 3*(i-1)+1, 3*(i-1)+2  (i=1..4)
        # Python 0-indexed: V -> 3*i+2, Ti -> 3*i, Mi -> 3*i+1  (i=0..3)
        V_RMSLE[i] = rmsle(xp_at_obs[3 * i + 2, :], xhatArray[3 * i + 2, :])
        Ti_RMSLE[i] = rmsle(xp_at_obs[3 * i, :], xhatArray[3 * i, :])
        Mi_RMSLE[i] = rmsle(xp_at_obs[3 * i + 1, :], xhatArray[3 * i + 1, :])
    T_RMSLE = rmsle(xp_at_obs[12, :], xhatArray[12, :])
    M_RMSLE = rmsle(xp_at_obs[13, :], xhatArray[13, :])

    if verbose:
        print(f"\nKT parameter RMSLE:  KT1={KT_RMSLE[0]:.3e}  KT2={KT_RMSLE[1]:.3e}  "
              f"KT3={KT_RMSLE[2]:.3e}  KT4={KT_RMSLE[3]:.3e}")
        print("V  RMSLE: " + "  ".join(f"{v:.3e}" for v in V_RMSLE))
        print("Ti RMSLE: " + "  ".join(f"{v:.3e}" for v in Ti_RMSLE))
        print("Mi RMSLE: " + "  ".join(f"{v:.3e}" for v in Mi_RMSLE))
        print(f"T  RMSLE: {T_RMSLE:.3e}")
        print(f"M  RMSLE: {M_RMSLE:.3e}")

    # sigma on the full plant grid, and the true theta of each mode -- both are
    # needed by the figures and were previously reconstructed in sim_io.py from
    # the saved .mat; deriving them here keeps run() self-contained.
    sigma_full = np.array([get_treatment(t, Tcont, dts) for t in tp_full], float)
    _sig = np.asarray(sigma_arr, int)
    KT_true_per_mode = np.zeros((np_, ns_modes))
    for _m in range(ns_modes):
        _c = np.where(_sig == _m)[0]
        KT_true_per_mode[:, _m] = KTtrue_arr[:, _c[0]]

    return {
        "p": p, "E": E, "ng": ng, "nx": nx, "np": np_, "ns_modes": ns_modes,
        "sigma_full": sigma_full, "KT_true_per_mode": KT_true_per_mode,
        "dt": dt, "Ts": Ts, "Tobs": Tobs, "Tcont": Tcont, "tf": tf, "dts": dts,
        "Nsteps": Nsteps, "Nplant": Nplant,
        "tPre": tPre, "xPre": xPre,
        "xPlant": xPlant, "tp_full": tp_full, "xp_full": xp_full, "to": to,
        "xhatArray": xhatArray, "KThatArray": KThatArray, "KThatAll": KThatAll,
        "LambdaActive": LambdaActive,
        "rho_x": rho_x, "rho_theta": rho_theta, "Q": Q,
        "sigma_arr": sigma_arr, "KTtrue_arr": KTtrue_arr,
        "xp_at_obs": xp_at_obs,
        "KT_RMSLE": KT_RMSLE, "V_RMSLE": V_RMSLE, "Ti_RMSLE": Ti_RMSLE,
        "Mi_RMSLE": Mi_RMSLE, "T_RMSLE": T_RMSLE, "M_RMSLE": M_RMSLE,
    }


# ----- persisted run data -----
# run() takes ~15 s; the figures take a couple of seconds.  run() is therefore
# executed once and its result dict is written to HIV_DATA, after which
# hiv_figures.py can be re-run (restyled, re-laid out, re-exported) as often as
# needed without touching the integrator -- and is guaranteed to show the same
# run rather than a fresh one.
HIV_DATA = "hiv_data.npz"


def save_data(res, path=HIV_DATA):
    """Write a run() result to `path` (npz).  Returns the path written.

    The Params dataclass under key "p" is not array data and is dropped; every
    figure and compute_pe() work from the numeric fields alone.
    """
    np.savez_compressed(path, **{k: np.asarray(v) for k, v in res.items()
                                 if k != "p"})
    return path


def load_data(path=HIV_DATA):
    """Read back a run() result written by save_data(); inverse of save_data().

    np.load returns every entry as an array, so the scalar entries are restored
    to the native types the figure code expects.
    """
    with np.load(path, allow_pickle=False) as z:
        res = {k: z[k] for k in z.files}
    for k in ("E", "dt", "Ts", "Tobs", "Tcont", "tf", "dts"):
        res[k] = float(res[k])
    for k in ("ng", "nx", "np", "ns_modes", "Nsteps", "Nplant"):
        res[k] = int(res[k])
    return res


if __name__ == "__main__":
    results = run(verbose=True)
    save_data(results)
    print(f"\nSaved run data to {HIV_DATA} -- run 'python hiv_figures.py' to plot it.")
