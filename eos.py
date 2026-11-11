"""
eos.py
======
The thermodynamic model. The solver asks it for two things only:

    state_from_pT(p, T)   the state at pressure p and temperature T (used for the inlet
                          total state)
    state_from_hs(h, s)   the state at enthalpy h and entropy s (used at every node)

Potential flow is isentropic, so s keeps its inlet value and the energy equation gives h at
each node (flow.py). This module turns (h, s) into p, T, rho and the speed of sound. Any
object with these two methods, returning a GasState, can replace PerfectGas in
config.gas without a change anywhere else.

The first model will be a calorically and thermally perfect gas: p = rho R T with constant
specific heats.

Status: interface only. No gas relation is implemented.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class GasState:
    """A thermodynamic state. Each field is a number, or an array with one value per node."""

    p: float | NDArray[np.float64]      # pressure
    T: float | NDArray[np.float64]      # temperature
    rho: float | NDArray[np.float64]    # density
    h: float | NDArray[np.float64]      # specific enthalpy
    s: float | NDArray[np.float64]      # specific entropy
    a: float | NDArray[np.float64]      # speed of sound


@dataclass(frozen=True)
class PerfectGas:
    """Calorically and thermally perfect gas."""

    gamma: float            # ratio of specific heats cp / cv
    gas_constant: float     # R = cp - cv

    def state_from_pT(self, p: float | NDArray[np.float64], T: float | NDArray[np.float64]) -> GasState:
        """The state at pressure p and temperature T."""
        raise NotImplementedError

    def state_from_hs(self, h: float | NDArray[np.float64], s: float | NDArray[np.float64]) -> GasState:
        """The state at enthalpy h and entropy s."""
        raise NotImplementedError
