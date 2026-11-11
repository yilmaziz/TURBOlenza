"""
flow.py
=======
From the stream function to the flow at the nodes, once per pass of the density loop:

    psi -> W_m, W_theta, W       definition of psi [Aungier 2003 Eqs. 5-16, 5-17]
    W   -> h -> static state     energy equation, entropy at its inlet value, eos.py

W is the velocity relative to the blade row; in a stationary cascade it is the absolute
velocity. The static state gives the density for the next pass and the Mach number. psi is
a fraction of the passage mass flow, so turning its derivatives into a velocity needs that
mass flow, which comes from the state at the far inlet.

The signs of W_m and W_theta are fixed on the equation sheet together with psi.

Status: interface only. No velocity formula and no energy equation is implemented.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from potential_flow_solver.boundary_conditions import BoundaryConditions
from potential_flow_solver.config import FlowConditions
from potential_flow_solver.eos import GasState, PerfectGas
from potential_flow_solver.grid import Grid
from potential_flow_solver.metrics import GridMetrics


@dataclass(frozen=True)
class FlowField:
    """The flow at the nodes, each array (n_m, n_eta)."""

    W_m: NDArray[np.float64]        # meridional velocity
    W_theta: NDArray[np.float64]    # tangential velocity
    W: NDArray[np.float64]          # speed
    state: GasState                 # static state: p, T, rho, h, s, a
    mach: NDArray[np.float64]       # W / a


def inlet_state(conditions: FlowConditions, eos: PerfectGas) -> tuple[GasState, GasState]:
    """Total and static state at the far inlet, from the total conditions and the inlet
    Mach number."""
    raise NotImplementedError


def passage_mass_flow(inlet_static: GasState, conditions: FlowConditions, grid: Grid, metrics: GridMetrics) -> float:
    """Mass flow through one passage, from the density, speed and angle at the far inlet and
    the inlet area of one passage. The angle is grid.inlet_angle, already in radians."""
    raise NotImplementedError


def velocity_components(psi: NDArray[np.float64], density: NDArray[np.float64], mass_flow: float,
                        grid: Grid, metrics: GridMetrics,
                        bc: BoundaryConditions) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """W_m and W_theta at every node, from psi and the density field psi was computed with."""
    raise NotImplementedError


def static_state(W: NDArray[np.float64], inlet_total: GasState, metrics: GridMetrics, angular_speed: float,
                 eos: PerfectGas) -> GasState:
    """Static state at every node. The enthalpy comes from the energy equation (rothalpy is
    constant in a rotating row, total enthalpy in a stationary one) and the entropy is the
    inlet value."""
    raise NotImplementedError
