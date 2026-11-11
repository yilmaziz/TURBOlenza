"""
coefficients.py
===============
The coefficient fields A, B, C, D, E of the stream-function equation in expanded form
[Aungier 2003 Eq. 5-21]. A, B and C carry the inverse metric of the (m, eta) map
[Eqs. 5-22 to 5-24]; D and E hold the first-derivative terms that appear when Eq. 5-20 is
expanded into Eq. 5-21.

The density enters these fields, so they are rebuilt on every pass of the density loop,
while the grid and the metrics stay fixed. If the equation sheet shows another flow
quantity in Eq. 5-21 (a right-hand side for a rotating row, for example), it is added here
as an argument and a field.

Status: interface only. None of Eqs. 5-20 to 5-24 is implemented.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from potential_flow_solver.metrics import GridMetrics


@dataclass(frozen=True)
class Coefficients:
    """Coefficient fields of Eq. 5-21, each (n_m, n_eta)."""

    A: NDArray[np.float64]
    B: NDArray[np.float64]
    C: NDArray[np.float64]
    D: NDArray[np.float64]
    E: NDArray[np.float64]


def compute_coefficients(metrics: GridMetrics, density: NDArray[np.float64]) -> Coefficients:
    """A to E at every node, for the density field (n_m, n_eta) of the current pass."""
    raise NotImplementedError
