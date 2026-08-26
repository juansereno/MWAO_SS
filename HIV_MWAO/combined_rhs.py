"""Combined RHS for the ODE solver: plant + active mode-wise observer.

Python port of ``combined_rhs.m``.

    y = [ x_plant ;     (0      .. nx-1)
          s_ell    ];   (nx     .. end )
"""

from __future__ import annotations

import numpy as np

from hiv_plant import hiv_plant
from mw_ao_rhs import mw_ao_rhs


def combined_rhs(y, u_mode, p, E, rho_x, rho_theta, Q, nx):
    """Return d(y)/dt for the stacked plant + active-observer system."""
    y = np.asarray(y, dtype=float)

    x_plant = y[:nx]
    s_ell = y[nx:]

    # --- Plant derivative ----------------------------------------------
    dx = hiv_plant(x_plant, p, u_mode)

    # --- Scaled measurement from current plant state -------------------
    # yhat = [T; V1; V2; V3; V4] / E   (0-indexed plant rows 12,2,5,8,11)
    yhat = np.array([x_plant[12],   # T
                     x_plant[2],    # V1
                     x_plant[5],    # V2
                     x_plant[8],    # V3
                     x_plant[11]]) / E  # V4

    # --- Observer derivative -------------------------------------------
    ds = mw_ao_rhs(s_ell, yhat, u_mode, p, E, rho_x, rho_theta, Q)

    return np.concatenate([dx, ds])
