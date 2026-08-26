"""
HIV mutation-model parameters (4 genotypes).

Python port of the parameter block defined at the top of
``MW_AO_HIV_main.m``.  The MATLAB code used a struct ``p`` with dotted fields;
here the same data lives in a small dataclass-like container so the rest of the
code can write ``p.dTi`` exactly as the MATLAB did.

All quantities use the same units and numerical values as the original
manuscript code (Sereno & Hernandez-Vargas, bioRxiv 2026).
"""

from __future__ import annotations

import numpy as np


class Params:
    """Container mirroring the MATLAB ``p`` struct."""

    def __init__(self) -> None:
        # ------------------------------------------------------------------
        # Healthy cell dynamics
        # ------------------------------------------------------------------
        self.sT = 10.0
        self.sM = 0.15
        self.rhoT = 0.01
        self.rhoM = 0.0035
        self.CT = 300.0
        self.CM = 500.0
        self.dT = 0.01
        self.dM = 1e-3

        # Infected cell / virus death rates
        self.dTi = 0.4
        self.dMi = 1e-3
        self.dV = 2.4
        self.mu = 1e-5

        # ------------------------------------------------------------------
        # Genotype fitness factors
        # ------------------------------------------------------------------
        self.eff = np.array([1.0, 0.98, 0.92, 0.88])

        # Base infection rates [ng]
        self.bT = 4.7714e-5 * self.eff
        self.bM = 4.5333e-8 * self.eff
        # Base virus production rates [ng]
        self.pTr = 38.0 * self.eff
        self.pMr = 44.0 * self.eff

        # ------------------------------------------------------------------
        # Mutation matrix (Kalman convention)
        # ------------------------------------------------------------------
        self.Mmut = np.array([[0, 0, 0, 0],
                              [1, 0, 0, 0],
                              [1, 0, 0, 0],
                              [0, 1, 1, 0]], dtype=float)
        # Column sums -> outflow multiplicity per donor genotype (1 x ng)
        self.Mf_out = self.Mmut.sum(axis=0)

        # ------------------------------------------------------------------
        # Drug efficiencies [treatment x genotype]
        # ------------------------------------------------------------------
        self.nTd = np.array([[0.97, 0.10, 0.80, 0.40],    # Treatment 1 - T cells
                             [0.98, 0.95, 0.10, 0.01]])   # Treatment 2 - T cells
        self.nMd = np.array([[0.95, 0.10, 0.80, 0.40],    # Treatment 1 - macrophages
                             [0.88, 0.95, 0.10, 0.01]])   # Treatment 2 - macrophages


# Convenient dimension constants (match MATLAB scalars)
NG = 4             # number of genotypes
NX = 14            # full plant state dimension
NP = 4             # parameters per mode (KT_1..KT_4)
NY = 5             # measurable outputs [T; V1..V4]
NS_MODES = 3       # mode observers: u = 0, 1, 2
