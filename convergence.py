"""
convergence.py
==============
Bookkeeping for the nonlinear iteration: how big each change is, the history of those
numbers, and the stop tests. Nothing here changes the solution.

    density                changes in density, psi and velocity; difference-equation residual
    Kutta                  discharge angle and Kutta error
    The two run in one loop (solver.solve: every pass updates the density and corrects the
    discharge angle); solver.solve_at_exit_angle runs the density part alone.

The whole history is kept, not only the last value, so a loop that oscillates or stalls can
be seen as such.

The change between two passes (decided 2026-11-13)
    One definition for every field: the largest change relative to the largest magnitude of
    the previous field,
        change_norm(new, old) = max |new - old| / max |old|,
    scale-free and never divided by a local small value (a stagnation point has W ~ 0; rho > 0
    everywhere). The velocity is compared as a vector:
        velocity_change = max |(W_m, W_theta)_new - (W_m, W_theta)_old| / max |(W_m, W_theta)_old|.
    Pass n of the density loop records density_change = change_norm(rho(n+1), rho(n)) (the
    relaxed density against the resident one) and velocity_change = the change of W(n) against
    W(n-1); the first pass has no previous velocity, so its velocity_change is NaN and the pass
    cannot be converged.

Stop test of the density loop (density_converged)
    Aungier judges convergence on successive estimates of the density AND the velocity: the
    density change of the last pass must be at or below density_tolerance and its velocity
    change at or below the velocity tolerance, IterationSettings.velocity_tolerance, or
    density_tolerance when that is None (the default: the rule before the field existed,
    unchanged). No tolerance is chosen here; the caller gives it.

Stop test of the Kutta condition (kutta_converged) and of the coupled loop
    History.kutta_error holds eps_K of kutta.py, |W_SS - W_PS| / (0.5 (W_SS + W_PS)) at the TE
    station, one entry per pass of the coupled loop (the pass's own velocity field). The Kutta
    test needs the last one at or below kutta_tolerance; coupled_converged needs the density
    test AND the Kutta test on the same last pass: density, velocity and Kutta together,
    never the density first and the Kutta condition afterwards. A NaN entry never passes.

Status: change_norm, velocity_change and density_converged implemented 2026-11-13 (under
review); kutta_converged and coupled_converged 2026-11-13 with the Kutta coupling (under
review).
"""

from __future__ import annotations

import math
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
    residual: list[float] = field(default_factory=list)        # ||F Psi - Q||_inf of the pass's linear solve
    # Kutta loop (the coupled loop: one entry per pass, the pass's resident angle and its eps_K)
    exit_angle: list[float] = field(default_factory=list)      # radians
    kutta_error: list[float] = field(default_factory=list)      # eps_K of kutta.py, dimensionless


def change_norm(new: NDArray[np.float64], old: NDArray[np.float64]) -> float:
    """max |new - old| / max |old| (module docstring): the relative largest change of one
    field between two passes. ValueError for shapes that differ, non-finite values or an old
    field that is zero everywhere."""
    new, old = np.asarray(new, dtype=float), np.asarray(old, dtype=float)
    if new.shape != old.shape:
        raise ValueError(f"the two fields have shapes {new.shape} and {old.shape}")
    if not (np.all(np.isfinite(new)) and np.all(np.isfinite(old))):
        raise ValueError("a field compared between passes is not finite")
    scale = float(np.max(np.abs(old)))
    if not scale > 0.0:
        raise ValueError("the previous field is zero everywhere: its relative change is undefined")
    return float(np.max(np.abs(new - old))) / scale


def velocity_change(W_m_new, W_theta_new, W_m_old, W_theta_old) -> float:
    """max |dW| / max |W_old| for the velocity vectors (W_m, W_theta) of two passes."""
    new = np.stack([np.asarray(W_m_new, dtype=float), np.asarray(W_theta_new, dtype=float)])
    old = np.stack([np.asarray(W_m_old, dtype=float), np.asarray(W_theta_old, dtype=float)])
    if new.shape != old.shape:
        raise ValueError(f"the two velocity fields have shapes {new.shape[1:]} and {old.shape[1:]}")
    if not (np.all(np.isfinite(new)) and np.all(np.isfinite(old))):
        raise ValueError("a velocity field compared between passes is not finite")
    scale = float(np.max(np.hypot(old[0], old[1])))
    if not scale > 0.0:
        raise ValueError("the previous velocity is zero everywhere: its relative change is undefined")
    return float(np.max(np.hypot(new[0] - old[0], new[1] - old[1]))) / scale


def velocity_tolerance(settings: IterationSettings) -> float:
    """The tolerance of the velocity change: settings.velocity_tolerance, or
    settings.density_tolerance when that is None (module docstring)."""
    given = settings.velocity_tolerance
    return float(settings.density_tolerance if given is None else given)


def density_converged(history: History, settings: IterationSettings) -> bool:
    """True when the last density pass changed the density by at most
    settings.density_tolerance and the velocity by at most velocity_tolerance(settings)
    (module docstring); False after a first pass (no previous velocity) or with an empty
    history."""
    if not history.density_change or not history.velocity_change:
        return False
    d, w = history.density_change[-1], history.velocity_change[-1]
    if math.isnan(d) or math.isnan(w):
        return False
    return d <= float(settings.density_tolerance) and w <= velocity_tolerance(settings)


def kutta_converged(history: History, settings: IterationSettings) -> bool:
    """True when the last Kutta error (eps_K of kutta.py) is at most settings.kutta_tolerance;
    False with no Kutta entry or a NaN."""
    if not history.kutta_error:
        return False
    e = history.kutta_error[-1]
    return not math.isnan(e) and e <= float(settings.kutta_tolerance)


def coupled_converged(history: History, settings: IterationSettings) -> bool:
    """density_converged and kutta_converged on the same last pass (module docstring)."""
    return density_converged(history, settings) and kutta_converged(history, settings)
