"""
solver.py
=========
Runs the method by calling the other modules in order. It holds no equations of its own, so
every step can be traced to the module that does it.

Planned data flow (step numbers are the order of the method)

    spec = geometry.CascadeSpec(...);  cascade = geometry.make_cascade(spec)
    solve(cascade, config)
     0  config.flow.inlet_angle_deg, exit_angle_guess_deg -> radians, the one conversion
     1  grid.make_grid                              geometry -> grid
     2  flow.inlet_state                            first density field, for example the
                                                    inlet static density everywhere
     |
     |  Kutta loop: one pass per discharge angle
     |    solve_at_exit_angle(grid, ...)
     |       metrics.build_metrics, boundary_conditions.make_boundary_conditions
     |       flow.inlet_state, flow.passage_mass_flow
     |       density loop: one pass per density field
     |  3      coefficients.compute_coefficients    A to E from metrics and density
     |  4      discretization.assemble              sparse matrix and right-hand side
     |  5      linear_solver.solve                  psi
     |  6      flow.velocity_components             W_m, W_theta, W
     |  7      flow.static_state (eos)              new density, relaxed
     |  8      convergence.density_converged        stop, or repeat from 3
     |  9  kutta.kutta_error, convergence.kutta_converged; if not converged,
     |     kutta.next_exit_angle and grid.rebuild_exit, then the next pass
     v
    Solution

Status: skeleton only. Neither loop is implemented.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from potential_flow_solver.config import SolverConfig
from potential_flow_solver.convergence import History
from potential_flow_solver.flow import FlowField
from potential_flow_solver.geometry import Cascade, StreamSheet
from potential_flow_solver.grid import Grid
from potential_flow_solver.metrics import GridMetrics


@dataclass(frozen=True)
class Solution:
    """The final result and how it was reached. The discharge angle is grid.exit_angle."""

    grid: Grid
    metrics: GridMetrics
    psi: NDArray[np.float64]        # (n_m, n_eta)
    flow: FlowField
    history: History


def solve(cascade: Cascade, config: SolverConfig) -> Solution:
    """Steps 1 to 9: the solution that satisfies the Kutta condition."""
    raise NotImplementedError


def solve_at_exit_angle(grid: Grid, sheet: StreamSheet, config: SolverConfig,
                        density: NDArray[np.float64], history: History) -> Solution:
    """Steps 3 to 8 for the discharge angle the exit extension of grid was built for: the
    compressible solution without the Kutta condition. density is the starting field, so a
    Kutta pass can start from the previous one. Each density pass is appended to history."""
    raise NotImplementedError
