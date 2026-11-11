"""
kutta.py
========
The Kutta condition at the trailing edge, and the discharge angle that satisfies it.

For a given discharge angle the stream-function problem has one solution, but in general
the flow does not leave the trailing edge smoothly. Only one discharge angle makes it do
so. The outer loop of the solver moves the exit angle, and with it the exit extension of
the grid (grid.rebuild_exit), until kutta_error is zero.

Open: how the error is defined and where it is measured (the last blade station, an
extrapolation to the trailing edge, or a fixed station ahead of it).

Status: interface only. No Kutta condition and no angle iteration is implemented.
"""

from __future__ import annotations

from potential_flow_solver.flow import FlowField
from potential_flow_solver.grid import Grid


def kutta_error(flow: FlowField, grid: Grid) -> float:
    """How far the solution is from leaving the trailing edge smoothly; zero when the Kutta
    condition holds."""
    raise NotImplementedError


def next_exit_angle(exit_angles: list[float], errors: list[float]) -> float:
    """The next discharge angle to try, in radians, from the angles tried so far (radians)
    and their Kutta errors."""
    raise NotImplementedError
