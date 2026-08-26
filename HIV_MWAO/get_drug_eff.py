"""Drug efficiency vectors for a given treatment mode.

Python port of ``get_drug_eff.m``.
"""

from __future__ import annotations

import numpy as np


def get_drug_eff(u, p, ng):
    """Return (nT, nM) drug-efficiency vectors for treatment ``u``.

    u = 0     -> no treatment -> nT = nM = 0
    u = 1, 2  -> rows of the treatment tables in ``p``

    Parameters
    ----------
    u : int
        Treatment mode (0, 1 or 2).
    p : Params
        Parameter container.
    ng : int
        Number of genotypes.

    Returns
    -------
    (nT, nM) : tuple of ndarray, each shape (ng,)
    """
    if u == 0:
        nT = np.zeros(ng)
        nM = np.zeros(ng)
    else:
        # MATLAB rows are 1-indexed: p.nTd(u, :) -> Python p.nTd[u - 1]
        nT = p.nTd[u - 1, :].copy()
        nM = p.nMd[u - 1, :].copy()
    return nT, nM
