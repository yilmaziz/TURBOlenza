"""
boundary_conditions.py
======================
The conditions on the four edges of the computational rectangle [Aungier 2003 Sec. 5.3,
Fig. 5-3]. This module says which node gets which condition and with what value;
discretization.py turns each condition into rows of the linear system.

Edges
    eta = 0 and eta = 1 on the blade stations: walls. A wall is a streamline, so psi is
        constant along it.
    eta = 0 and eta = 1 on the extensions: periodic lines. They are the same physical line
        one pitch apart, so psi on them differs by the flow through one passage,
        psi_wall1 - psi_wall0 [Eqs. 5-29 to 5-32].
    i = 0 and i = n_m - 1: far inlet and far exit [Eq. 5-33]. The grid lines there run
        along the flow angle, which is how the inlet and discharge angles enter.

Which wall gets which psi value is fixed on the equation sheet, together with the signs of
W_m and W_theta.

Status: interface only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from potential_flow_solver.grid import Grid

INTERIOR, WALL, PERIODIC, FAR_INLET, FAR_EXIT = 0, 1, 2, 3, 4      # kind of a node


@dataclass(frozen=True)
class BoundaryConditions:
    """The condition at every node and the wall values of psi."""

    kind: NDArray[np.int64]     # (n_m, n_eta): INTERIOR, WALL, PERIODIC, FAR_INLET or FAR_EXIT
    psi_wall0: float            # psi on the wall at eta = 0
    psi_wall1: float            # psi on the wall at eta = 1


def make_boundary_conditions(grid: Grid) -> BoundaryConditions:
    """Classify every node and set the wall values. The corners, where a far boundary meets
    a periodic line, and the leading- and trailing-edge stations, where a periodic line
    meets a wall, need a rule for which condition applies; that rule is part of
    implementing this function."""
    raise NotImplementedError
