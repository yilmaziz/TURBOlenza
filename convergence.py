"""
convergence.py
==============
Bookkeeping for the two nonlinear loops: how big each change is, the history of those
numbers, and the stop tests. Nothing here changes the solution.

    density loop (inner)   changes in density, psi and velocity; difference-equation residual
    Kutta loop (outer)     discharge angle and Kutta error

The whole history is kept, not only the last value, so a loop that oscillates or stalls can
be seen as such.

Status: interface only. The History container is usable; the norm and stop tests are not
implemented.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from potential_flow_solver.config import IterationSettings


@dataclass
class History:
    """Iteration history. The density-loop lists get one entry per density pass, the
    Kutta-loop lists one per Kutta pass; kutta_pass says which Kutta pass each density
    entry belongs to."""

    # density loop
    kutta_pass: list[int] = field(default_factory=list)
    density_change: list[float] = field(default_factory=list)
    psi_change: list[float] = field(default_factory=list)
    velocity_change: list[float] = field(default_factory=list)
    residual: list[float] = field(default_factory=list)        # see discretization.residual
    # Kutta loop
    exit_angle: list[float] = field(default_factory=list)      # radians
    kutta_error: list[float] = field(default_factory=list)


def change_norm(new: NDArray[np.float64], old: NDArray[np.float64]) -> float:
    """Size of the change of one field between two passes. Which norm (largest or RMS,
    absolute or relative) is chosen when this is implemented."""
    raise NotImplementedError


def density_converged(history: History, settings: IterationSettings) -> bool:
    """True when the last density pass changed the solution by less than the tolerance."""
    raise NotImplementedError


def kutta_converged(history: History, settings: IterationSettings) -> bool:
    """True when the last Kutta error is within the tolerance."""
    raise NotImplementedError
