"""Treatment switching signal (SWATCH cycling).

Python port of ``get_treatment.m``.
"""

from __future__ import annotations

import math


def get_treatment(t, Tcont, dts):
    """Determine the treatment signal at time ``t``.

    u = 0 : no treatment (t < Tcont)
    u = 1 : Treatment 1 (first period after Tcont)
    u = 2 : Treatment 2 (alternating every ``dts`` days)
    """
    if t < Tcont:
        return 0
    if math.floor((t - Tcont) / dts) % 2 == 0:
        return 1
    return 2
