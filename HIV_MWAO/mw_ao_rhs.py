"""Right-hand side of ONE mode-wise adaptive observer Pi_l (full 14-state).

Python port of ``MW_AO_rhs.m`` -- implements paper Eq. (13).

The MATLAB version cached mode-independent matrices in ``persistent`` variables.
Here the same caching is done lazily in a module-level dict keyed on
``id(p)`` together with ``E`` and the entries of ``Q`` that matter.  Because the
driver builds ``Q`` and ``p`` once and reuses them, the cache is populated on
the first call and reused thereafter -- algebraically identical to the original.

State packing (length 286), 0-indexed, MATLAB column-major (Fortran) layout:
    z_hat       : indices   0 ..  13   (14)
    vec(S)      : indices  14 .. 209   (14*14 = 196)
    vec(Lambda) : indices 210 .. 265   (14*4  =  56)
    vec(Gamma)  : indices 266 .. 281   ( 4*4  =  16)
    theta_hat   : indices 282 .. 285   (4)
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import lu_factor, lu_solve

from get_drug_eff import get_drug_eff

# Module-level cache (mirrors MATLAB persistent variables).
_cache = {}


def _build_cache(p, E, Q):
    """Build the mode-independent / per-mode constant matrices once."""
    # --- Output matrix C (constant) -----------------------------------
    C = np.zeros((5, 14))
    C[0, 12] = 1.0    # T
    C[1, 2] = 1.0     # V1
    C[2, 5] = 1.0     # V2
    C[3, 8] = 1.0     # V3
    C[4, 11] = 1.0    # V4
    Cprime = C.T
    CtQC = Cprime @ Q @ C            # 14x14, constant

    # --- Per-mode constants: parts of A that depend only on (u_mode, p) -
    mode_A_skel = []
    mode_KM = []
    for um in range(3):
        nT_m, nM_m = get_drug_eff(um, p, 4)
        KM_m = (1 - nM_m) * p.bM
        PT_m = (1 - nT_m) * p.pTr
        PM_m = (1 - nM_m) * p.pMr

        A_skel = np.zeros((14, 14))
        # Diagonal damping (constant -- independent of u_mode and yhat)
        A_skel[0, 0] = -p.dTi
        A_skel[1, 1] = -p.dMi
        A_skel[2, 2] = -p.dV
        A_skel[3, 3] = -p.dTi
        A_skel[4, 4] = -p.dMi
        A_skel[5, 5] = -p.dV
        A_skel[6, 6] = -p.dTi
        A_skel[7, 7] = -p.dMi
        A_skel[8, 8] = -p.dV
        A_skel[9, 9] = -p.dTi
        A_skel[10, 10] = -p.dMi
        A_skel[11, 11] = -p.dV
        A_skel[12, 12] = -p.dT
        # Virus-production rows: constant given u_mode
        A_skel[2, 0] = PT_m[0]
        A_skel[2, 1] = PM_m[0]
        A_skel[5, 3] = PT_m[1]
        A_skel[5, 4] = PM_m[1]
        A_skel[8, 6] = PT_m[2]
        A_skel[8, 7] = PM_m[2]
        A_skel[11, 9] = PT_m[3]
        A_skel[11, 10] = PM_m[3]
        # Rows (1,13),(4,13),(7,13),(10,13),(13,13) are filled per call (yhat).

        mode_A_skel.append(A_skel)
        mode_KM.append(KM_m)

    return {
        "C": C, "Cprime": Cprime, "CtQC": CtQC,
        "A_skel": mode_A_skel, "KM": mode_KM,
    }


def mw_ao_rhs(state, yhat, u_mode, p, E, rho_x, rho_theta, Q):
    """Return d(state)/dt for one mode-wise adaptive observer.

    Parameters
    ----------
    state : ndarray, shape (286,)
        Packed observer state (see module docstring).
    yhat : ndarray, shape (5,)
        Scaled measurement (1/E) * [T; V1; V2; V3; V4].
    u_mode : int
        Observer mode (0, 1 or 2).
    p : Params
    E : float
        State-scaling factor.
    rho_x, rho_theta : sequence of length 3
        Decay rates per mode.
    Q : ndarray, shape (5, 5)
        Output-error weighting.
    """
    global _cache
    key = (id(p), float(E))
    if _cache.get("key") != key:
        _cache = _build_cache(p, E, Q)
        _cache["key"] = key

    C = _cache["C"]
    Cprime = _cache["Cprime"]
    CtQC = _cache["CtQC"]

    state = np.asarray(state, dtype=float)

    # --- Unpack state (MATLAB column-major reshape -> order='F') --------
    z = state[0:14]
    S = state[14:210].reshape(14, 14, order="F")
    Lambda = state[210:266].reshape(14, 4, order="F")
    Gamma = state[266:282].reshape(4, 4, order="F")
    theta = state[282:286]

    # --- Mode-specific cached skeleton ---------------------------------
    A = _cache["A_skel"][u_mode].copy()
    KM = _cache["KM"][u_mode]

    # --- Scaled outputs + reusable products ----------------------------
    yT = yhat[0]
    yV = yhat[1:5]
    YVT = yV.sum()
    mu = p.mu
    EyT = E * yT

    EyV1 = E * yV[0]
    EyV2 = E * yV[1]
    EyV3 = E * yV[2]
    EyV4 = E * yV[3]
    KMyV1 = KM[0] * EyV1
    KMyV2 = KM[1] * EyV2
    KMyV3 = KM[2] * EyV3
    KMyV4 = KM[3] * EyV4

    # --- Fill yhat-dependent entries of A ------------------------------
    A[1, 13] = (1 - 2 * mu) * KMyV1
    A[4, 13] = mu * KMyV1 + (1 - mu) * KMyV2
    A[7, 13] = mu * KMyV1 + (1 - mu) * KMyV3
    A[10, 13] = mu * KMyV2 + mu * KMyV3 + KMyV4

    pM_Obs = E * p.rhoM * YVT / (p.CM + E * YVT)
    A[13, 13] = pM_Obs - (KMyV1 + KMyV2 + KMyV3 + KMyV4) - p.dM

    # --- beta vector ---------------------------------------------------
    pT_Obs = E * p.rhoT * yT * YVT / (p.CT + E * YVT)
    beta = np.zeros(14)
    beta[12] = pT_Obs + p.sT / E
    beta[13] = p.sM / E

    # --- Psi matrix ----------------------------------------------------
    Psi = np.zeros((14, 4))
    EyTyV1 = EyT * yV[0]
    EyTyV2 = EyT * yV[1]
    EyTyV3 = EyT * yV[2]
    EyTyV4 = EyT * yV[3]
    Psi[0, 0] = (1 - 2 * mu) * EyTyV1
    Psi[3, 0] = mu * EyTyV1
    Psi[3, 1] = (1 - mu) * EyTyV2
    Psi[6, 0] = mu * EyTyV1
    Psi[6, 2] = (1 - mu) * EyTyV3
    Psi[9, 1] = mu * EyTyV2
    Psi[9, 2] = mu * EyTyV3
    Psi[9, 3] = EyTyV4
    Psi[12, 0] = -EyTyV1
    Psi[12, 1] = -EyTyV2
    Psi[12, 2] = -EyTyV3
    Psi[12, 3] = -EyTyV4

    # --- Innovation ----------------------------------------------------
    innov = yhat - C @ z                  # 5

    # --- Single LU decomposition of S (reused below) -------------------
    lu = lu_factor(S)

    # --- AO ODEs (algebraically restructured) --------------------------
    QInnov = Q @ innov                    # 5
    CtQinnov = Cprime @ QInnov            # 14, used twice

    # z-equation correction term:
    #   S\(C'Q innov) + Lambda*(Gamma\(Lambda'*(C'Q innov)))
    term1 = lu_solve(lu, CtQinnov)                                   # 14
    term2 = Lambda @ np.linalg.solve(Gamma, Lambda.T @ CtQinnov)     # 14
    correction = term1 + term2

    dz = A @ z + beta + Psi @ theta + correction

    # S-equation
    dS = -rho_x[u_mode] * S - A.T @ S - S @ A + CtQC

    # Lambda-equation: A*Lambda - S\(CtQC*Lambda) + Psi
    dLambda = A @ Lambda - lu_solve(lu, CtQC @ Lambda) + Psi

    # Gamma-equation
    LtCtQC = Lambda.T @ CtQC               # 4x14
    dGamma = -rho_theta[u_mode] * Gamma + LtCtQC @ Lambda

    # theta-equation
    dtheta = np.linalg.solve(Gamma, Lambda.T @ CtQinnov)

    # --- Pack (MATLAB column-major flatten -> order='F') ---------------
    ds = np.concatenate([
        dz,
        dS.flatten(order="F"),
        dLambda.flatten(order="F"),
        dGamma.flatten(order="F"),
        dtheta,
    ])
    return ds
