"""
postprocess.py
==============
Results derived from a finished Solution, as arrays ready to plot or tabulate. Nothing here
plots, and nothing feeds back into the solve.

    surface Mach number     on both walls over the blade stations
    pressure coefficient    on the same walls; the reference (p_t1 - p_1, or the inlet
                            dynamic pressure) is chosen when this is implemented
    discharge angle         from the velocity at the far exit, a check on the angle the
                            Kutta loop settled on
    field quantities        solution.flow holds them at every node, and grid.node_theta
                            gives the node positions for contour plots

The eta = 0 wall is the upper surface of one blade and the eta = 1 wall the lower surface of
the next, which by periodicity is the lower surface of the same blade.

Status: interface only.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from potential_flow_solver.solver import Solution


def surface_mach(solution: Solution) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """m of the blade stations, and the Mach number on the eta = 0 and eta = 1 walls there."""
    raise NotImplementedError


def pressure_coefficient(solution: Solution) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """m of the blade stations, and the pressure coefficient on the eta = 0 and eta = 1
    walls there."""
    raise NotImplementedError


def discharge_angle_deg(solution: Solution) -> float:
    """Flow angle at the far exit from the velocity field, in degrees from the meridional
    direction, for reporting. It should equal degrees(solution.grid.exit_angle)."""
    raise NotImplementedError
