"""Full 14-state HIV nonlinear plant dynamics.

Python port of ``HIV_plant.m`` (vectorised).

State ordering (Kalman convention, 0-indexed in Python):
    x = [Ti1 Mi1 V1 | Ti2 Mi2 V2 | Ti3 Mi3 V3 | Ti4 Mi4 V4 | T M]
         0   1   2     3   4   5    6   7   8     9   10  11   12 13
"""

from __future__ import annotations

import numpy as np

# Index helpers (Kalman ordering)
_TI = [0, 3, 6, 9]    # infected T cells     (MATLAB [1 4 7 10])
_MI = [1, 4, 7, 10]   # infected macrophages (MATLAB [2 5 8 11])
_VI = [2, 5, 8, 11]   # viral loads          (MATLAB [3 6 9 12])
_T = 12               # healthy T cells      (MATLAB 13)
_M = 13               # macrophages          (MATLAB 14)


def hiv_plant(x, p, u):
    """Return dx/dt for the 14-state HIV plant.

    Parameters
    ----------
    x : ndarray, shape (14,)
        State vector in Kalman ordering.
    p : Params
        Parameter container.
    u : int
        Treatment mode (0, 1 or 2).
    """
    ng = 4
    x = np.asarray(x, dtype=float)

    # --- Unpack state ----------------------------------------------------
    Ti = x[_TI]              # 4
    Mi = x[_MI]              # 4
    Vi = x[_VI]              # 4
    T = x[_T]
    M = x[_M]
    VT = Vi.sum()            # total viral load

    # --- Drug efficiencies for current treatment -------------------------
    if u == 0:
        nT = np.zeros(ng)
        nM = np.zeros(ng)
    elif u == 1:
        nT = p.nTd[0, :]
        nM = p.nMd[0, :]
    else:
        nT = p.nTd[1, :]
        nM = p.nMd[1, :]

    # --- Effective rates (4-vectors) -------------------------------------
    KT = (1 - nT) * p.bT
    KM = (1 - nM) * p.bM
    PT = (1 - nT) * p.pTr
    PM = (1 - nM) * p.pMr

    # --- Per-genotype reaction terms -------------------------------------
    TVi = T * Vi                              # 4
    MVi = M * Vi                              # 4
    self_factor = 1 - p.Mf_out * p.mu         # 4, outflow correction

    # Self-infection minus death
    dTi = self_factor * KT * TVi - p.dTi * Ti
    dMi = self_factor * KM * MVi - p.dMi * Mi

    # Mutation inflow from donor genotypes (Mmut(i,j) is 0/1 indicator)
    dTi = dTi + p.Mmut @ (p.mu * KT * TVi)
    dMi = dMi + p.Mmut @ (p.mu * KM * MVi)

    # Virus production / clearance
    dVi = PT * Ti + PM * Mi - p.dV * Vi

    # --- Healthy T cells -------------------------------------------------
    pT_prolif = p.rhoT * T * VT / (VT + p.CT)
    dT = p.sT - p.dT * T - T * np.sum(KT * Vi) + pT_prolif

    # --- Macrophages -----------------------------------------------------
    pM_prolif = p.rhoM * M * VT / (VT + p.CM)
    dM = p.sM - p.dM * M - M * np.sum(KM * Vi) + pM_prolif

    # --- Assemble output (Kalman ordering) -------------------------------
    dx = np.zeros(14)
    dx[_TI] = dTi
    dx[_MI] = dMi
    dx[_VI] = dVi
    dx[_T] = dT
    dx[_M] = dM
    return dx
