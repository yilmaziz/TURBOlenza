"""
kutta.py
========
The trailing-edge Kutta condition and the discharge-angle correction that drives it to zero
[Aungier 2003 Eq. 5-34; Aungier 2000, Centrifugal Compressors, Sec. 13.3, Eqs. 13-57, 13-59].
It reads a solved flow field and returns numbers: the residual, the correction of the exit
angle, the damped next angle. It solves no PDE, updates no density, makes no thermodynamic
call and builds no matrix; solver.py runs the iteration (one correction per density pass).

1. The condition [Aungier 2003 Eq. 5-34]
    [W(m, 0)]_TE = [W(m, 1)]_TE: equal relative speeds on the two blade surfaces at the
    trailing edge. In the isentropic flow of one rothalpy, h = I - W^2 / 2 + (omega r)^2 / 2
    at one radius, so equal speeds mean equal h, and with the one entropy equal T, p and rho:
    the equal TE pressure that the condition stands for. The residual is the speed
    difference, never a pressure difference (the equivalence is a test, not the definition).

2. Where (trailing_edge_station)
    i = grid.last_blade_i, the last station whose line cuts the blade: both wall nodes exist
    and are the passage-facing surface nodes of grid.py's topology (sections 3, 8). Never
    l + 1 (the first exit station, periodic, no blade surface) and never the geometric tail
    end (not a station). The suction (SS) and pressure (PS) node are read from
    grid.wall_identity at that station, the named surface of each wall node, not assumed. On
    the admissible grids of this project the lower wall (j = 0, theta0) is the reference
    blade's suction surface and the upper wall (j = n_eta - 1, theta1) the pressure surface
    of its +pitch neighbor, but the code asks. A station whose two wall nodes are not one SS
    and one PS node (an inadmissible TE station, grid.py section 8) is refused: KuttaError.

3. The residual (kutta_residual)
    W_SS = hypot(W_m, W_theta) at the SS node and W_PS at the PS node,
        R_K   = W_SS - W_PS                          the velocity unit of the flow (m/s in SI)
        eps_K = |R_K| / (0.5 (|W_SS| + |W_PS|))      dimensionless, 0 <= eps_K <= 2
    Sign convention: R_K > 0 when the suction surface is the faster one at the TE station.
    Zero speed: the denominator vanishes only when both speeds are exactly zero; the two are
    then equal, Eq. 5-34 holds, and eps_K = 0 (documented fallback). Otherwise the
    denominator is positive and eps_K <= 2: no small denominator can make it unbounded. The
    dimensional R_K is reported beside eps_K, never hidden by it.

4. Angle conventions, and Aungier 2000 mapped onto this project
    Project (geometry.py, grid.py, metrics.py, FlowConditions; Aungier 2003 Eq. 5-3): beta
    from the meridional (axial) direction, positive toward +y (+theta),
        tan(beta) = r d(theta) / dm = dy / dm,    W_theta / W_m = tan(beta) along a grid line.
    Aungier 2000, Ch. 13: beta is the angle of a constant-eta curve with the TANGENTIAL
    direction (the text after Eq. 13-2), and Eq. 13-23 defines it by cot(beta) =
    [r d(theta) / dm]_eta. m, theta and the eta = 0 -> 1 orientation are the same in both
    (Eq. 13-4 is Eq. 5-2), so, exactly and with no sign change,
        cot(beta_2000) = tan(beta),    beta_2000 = pi/2 - beta,
        sin(beta_2000) = cos(beta),    cos(beta_2000) = sin(beta).
    Check on the velocity recovery: with it Eq. 13-12, W_U = mdot / (rho b) [(cot(beta) / S)
    psi_eta - psi_m], is flow.py's W_theta = mu [(tan(beta) / S) psi_eta - psi_m] (Eq. 5-19)
    term by term, and Eq. 13-11 is Eq. 5-18.

    The cos(beta) printed in Eqs. 13-57 and 13-58 is not cos(beta_2000). Aungier derives
    Eq. 13-57 from Stokes' theorem on the closed path of the two blade surfaces and the last
    two constant-m stations; its 1 / cos(beta) is the length of a wall element per unit m.
    In the chapter's own convention that length is dm / sin(beta_2000) (Eq. 13-3: d(xi) =
    dm / sin(beta_2000) along a constant-eta curve), which is dm / cos(beta) in ours.
    Eq. 13-58 agrees: (W_m)_te / (cos(beta0) + cos(beta1)) is half the speed W_m /
    cos(beta_mean) only for an angle from the meridional direction. Read as cos(beta_2000) =
    sin(beta), both equations would divide by zero for a blade leaving axially (beta -> 0).
    So the factor here is cos(beta) of this project's wall slopes at the TE station,
    1 / sqrt(1 + tan(beta_k)^2) with metrics.tan_beta0 and tan_beta1. Measured on the
    verification case (fixed-angle solution, 33 deg, n_passage 43): the discrete Stokes
    balance (S W_theta_bar)_(i+1) - (S W_theta_bar)_i over int_wall1 W ds - int_wall0 W ds,
    ds = dm / cos(beta), has median 0.992 over the interior blade elements (range 0.71 to
    1.24, an element-to-element alternation on the front half) and 0.993 over the rear half
    (0.83 to 1.10); with ds = dm / sin(beta) the medians are 0.298 and 0.232, off by about the
    wall slope tan(beta) (0.27 to 0.72 there). It fails only on the LE and TE transition
    elements (2.0 and 6.6), the transition defect of the audits, not revisited here.

5. The correction (exit_angle_correction) [Eqs. 13-57, 13-59 in this project's convention]
    Stokes' theorem on the last element (stations l - 1, l between the walls 0 and 1),
    irrotational flow, W_theta_bar = int_0^1 W_theta d(eta):
        (S W_theta_bar)_l - (S W_theta_bar)_(l-1) = int_wall1 W ds - int_wall0 W ds.
    A speed difference W0 - W1 at the TE thus removes about (W0 - W1) ds of the passage's
    angular momentum over the last element, which a smooth (Kutta) departure would keep.
    Aungier's correction of the TE angular momentum puts it back,
        Delta(r C_theta) = 2 (W0 - W1) r dm / (S (cos(beta0) + cos(beta1)))        (13-57)
    W0, W1 the speeds on the eta = 0 and eta = 1 walls at station l; S, beta0, beta1 the
    passage width and the wall angles there; dm the station spacing. The downstream boundary
    angle carries it to the far exit, where the flow follows the boundary (W_theta = W_m
    tan(beta_out), beta_out the same at every eta), so int_0^1 r W_theta d(eta) = r
    tan(beta_out) int_0^1 W_m d(eta) and
        Delta(tan(beta_out)) = Delta(r C_theta) / (r int_0^1 W_m d(eta))            (13-59)
    (Aungier's Delta(cot(beta_2000))), the integral taken at the far exit station, whose angle
    is being moved, by the trapezoidal rule in eta (our choice; the book names no rule). In
    the planar cascade r is one constant and cancels, dm / S is a ratio of grid lengths and
    (W0 - W1) / int W_m d(eta) one of speeds: the correction is dimensionless and needs no
    c_ref,
        Delta(tan(beta_out)) = 2 (W0 - W1) dm / (S (cos(beta0) + cos(beta1)) int_0^1 W_m d(eta)).
    Frames: C_theta = W_theta + omega r, and at one radius Delta(r C_theta) = r
    Delta(W_theta): the correction moves the RELATIVE discharge angle; in a stationary cascade
    the two frames coincide.
    Sign: W0 > W1 (the eta = 0 wall faster) gives Delta(tan(beta_out)) > 0. With the SS on
    eta = 0 that is R_K > 0 -> beta_out up: an SS faster than the PS at the TE is circulation
    beyond the Kutta value (the rear stagnation point on the PS), and a larger discharge angle
    of this cascade (less turning toward the axis) takes circulation away. The tests check the
    direction against the solver's own response (R_K falling as beta_out rises) on both sides
    of the root. The correction is written with Aungier's walls (W0 - W1), so it holds
    whichever wall the SS is: R_K = +-(W0 - W1) by the station's SS/PS identity.
    Not used: Eq. 13-58, Aungier's averaged speed W~ = W / 2 + W_m / (cos(beta0) +
    cos(beta1)) for blades whose two TE angles differ much. Its fixed point is not Eq. 5-34,
    and Aungier calls it usually unnecessary for compressors.

6. Damping and the next angle (next_exit_angle)
    tan(beta_out,n+1) = tan(beta_out,n) + lambda_K Delta(tan(beta_out)), 0 < lambda_K <= 1,
    on tan(beta) (Aungier's variable, his cot(beta_2000)), returned as beta = arctan(...) in
    radians within (-pi/2, pi/2), the convention of grid.rebuild_exit and Grid.exit_angle. A
    zero step (equal TE speeds) returns the angle itself, bit for bit.
    Aungier recommends suitable numerical damping without naming it; lambda_K has no default
    here: the caller passes config.IterationSettings.kutta_relaxation (project baseline
    config.KUTTA_RELAXATION).

Status: implemented 2026-11-13, under review, not frozen.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from turbolenza.b2b.flow import FlowField
from turbolenza.b2b.grid import Grid
from turbolenza.b2b.metrics import GridMetrics


class KuttaError(ValueError):
    """The Kutta residual or the exit-angle correction cannot be formed: the TE station is not
    one SS and one PS node, a speed or a slope is not finite, the far-exit meridional flow is
    not positive, the flow, metrics or residual belong to different grids, or a relaxation is
    outside (0, 1]."""


@dataclass(frozen=True)
class TrailingEdgeStation:
    """Where the Kutta condition is evaluated (module docstring, section 2)."""

    i: int                  # grid.last_blade_i
    m: float                # grid.m[i], the grid's length unit
    m_offset: float         # m - grid.m_te (negative: ahead of the chord-axis TE)
    ss_j: int               # eta index of the suction-surface node at station i
    ps_j: int               # eta index of the pressure-surface node


@dataclass(frozen=True)
class KuttaResidual:
    """Eq. 5-34 at the TE station (module docstring, section 3)."""

    station: TrailingEdgeStation
    W_ss: float             # |W| at the SS node
    W_ps: float             # |W| at the PS node
    residual: float         # R_K = W_ss - W_ps, the velocity unit of the flow
    normalized: float       # eps_K = |R_K| / (0.5 (|W_ss| + |W_ps|)), dimensionless
    wall_difference: float  # W0 - W1 = |W|(i, 0) - |W|(i, n_eta - 1), Aungier's walls theta0, theta1


@dataclass(frozen=True)
class ExitAngleCorrection:
    """Eqs. 13-57 and 13-59 in this project's convention (module docstring, section 5), not
    damped."""

    wall_difference: float          # W0 - W1 at the TE station
    delta_c_theta: float            # Delta(r C_theta) / r of Eq. 13-57, the velocity unit of the flow
    exit_meridional_integral: float # int_0^1 W_m d(eta) at the far exit station (trapezoidal rule)
    passage_width: float            # S at the TE station, the grid's length unit
    dm: float                       # station spacing, the grid's length unit
    cos_beta0: float                # cos of the eta = 0 wall angle at the TE station (from the meridional direction)
    cos_beta1: float                # cos of the eta = 1 wall angle
    delta_tan: float                # Delta(tan(beta_out)) of Eq. 13-59 (Aungier's Delta(cot(beta)))


def trailing_edge_station(grid: Grid) -> TrailingEdgeStation:
    """The last blade station and its SS and PS node (module docstring, section 2)."""
    if not isinstance(grid, Grid):
        raise TypeError(f"trailing_edge_station takes a Grid, got {type(grid).__name__}")
    i = grid.last_blade_i
    k = i - grid.first_blade_i
    identity = grid.wall_identity
    lower, upper = str(identity.lower.surface[k]), str(identity.upper.surface[k])
    last = len(grid.eta) - 1
    if (lower, upper) == ("suction", "pressure"):
        ss_j, ps_j = 0, last
    elif (lower, upper) == ("pressure", "suction"):
        ss_j, ps_j = last, 0
    else:
        raise KuttaError(f"the TE station {i} (m = {grid.m[i]:.6g}) has its wall nodes on the {lower} and the {upper} "
                         f"surface, not one suction and one pressure node: an inadmissible TE station (grid.py section 8)")
    return TrailingEdgeStation(i=i, m=float(grid.m[i]), m_offset=float(grid.m[i] - grid.m_te), ss_j=ss_j, ps_j=ps_j)


def kutta_residual(flow: FlowField, grid: Grid) -> KuttaResidual:
    """R_K = W_SS - W_PS and eps_K at the TE station of grid (module docstring, section 3),
    from the velocity components of flow (W = hypot(W_m, W_theta))."""
    station = trailing_edge_station(grid)
    shape = (len(grid.m), len(grid.eta))
    W_m, W_theta = np.asarray(flow.W_m, dtype=float), np.asarray(flow.W_theta, dtype=float)
    if W_m.shape != shape or W_theta.shape != shape:
        raise KuttaError(f"the flow has shape {W_m.shape}, the grid {shape}")
    i, last = station.i, shape[1] - 1
    speed = {j: float(np.hypot(W_m[i, j], W_theta[i, j])) for j in (0, last)}
    if not all(math.isfinite(value) for value in speed.values()):
        raise KuttaError(f"a wall speed at the TE station {i} is not finite")
    W_ss, W_ps = speed[station.ss_j], speed[station.ps_j]
    residual = W_ss - W_ps
    mean = 0.5 * (abs(W_ss) + abs(W_ps))
    normalized = abs(residual) / mean if mean > 0.0 else 0.0      # both speeds zero: equal, Eq. 5-34 holds
    return KuttaResidual(station=station, W_ss=W_ss, W_ps=W_ps, residual=residual, normalized=normalized,
                         wall_difference=speed[0] - speed[last])


def exit_angle_correction(residual: KuttaResidual, flow: FlowField, metrics: GridMetrics) -> ExitAngleCorrection:
    """The undamped change of tan(beta_out) that Eqs. 13-57 and 13-59 give for the TE speed
    difference of residual (module docstring, sections 4 and 5). flow gives the far-exit
    meridional velocity, metrics (of the grid the flow was solved on) the TE passage width,
    the wall slopes and, through metrics.grid, the station spacing."""
    if not isinstance(metrics, GridMetrics):
        raise TypeError(f"exit_angle_correction takes GridMetrics, got {type(metrics).__name__}")
    grid = metrics.grid
    i = residual.station.i
    if i != grid.last_blade_i:
        raise KuttaError(f"the residual is at station {i}, the TE station of these metrics is {grid.last_blade_i}")
    W_m = np.asarray(flow.W_m, dtype=float)
    if W_m.shape != (len(grid.m), len(grid.eta)):
        raise KuttaError(f"the flow has shape {W_m.shape}, the grid {(len(grid.m), len(grid.eta))}")
    tan0, tan1 = float(metrics.tan_beta0[i]), float(metrics.tan_beta1[i])
    width, dm = float(metrics.S[i]), float(grid.dm)
    integral = float(np.trapezoid(W_m[-1], grid.eta))
    if not all(math.isfinite(value) for value in (tan0, tan1, width, dm, integral, residual.wall_difference)):
        raise KuttaError("a TE slope, width, spacing, speed or the far-exit meridional integral is not finite")
    if not (width > 0.0 and dm > 0.0):
        raise KuttaError(f"the TE passage width {width} and the station spacing {dm} must be positive")
    if not integral > 0.0:
        raise KuttaError(f"int W_m d(eta) at the far exit is {integral}: no flow leaves through it, the exit angle cannot "
                         f"carry an angular-momentum correction (Eq. 13-59)")
    cos0, cos1 = 1.0 / math.sqrt(1.0 + tan0 * tan0), 1.0 / math.sqrt(1.0 + tan1 * tan1)
    delta_c_theta = 2.0 * residual.wall_difference * dm / (width * (cos0 + cos1))     # Eq. 13-57 over r
    return ExitAngleCorrection(wall_difference=residual.wall_difference, delta_c_theta=delta_c_theta,
                               exit_meridional_integral=integral, passage_width=width, dm=dm, cos_beta0=cos0,
                               cos_beta1=cos1, delta_tan=delta_c_theta / integral)                 # Eq. 13-59


def next_exit_angle(exit_angle: float, correction: ExitAngleCorrection, relaxation: float) -> float:
    """arctan(tan(exit_angle) + relaxation * correction.delta_tan), radians (module docstring,
    section 6). relaxation is lambda_K, 0 < lambda_K <= 1, given by the caller (no default)."""
    if isinstance(relaxation, bool) or not isinstance(relaxation, (int, float)) or not 0.0 < relaxation <= 1.0:
        raise KuttaError(f"the Kutta relaxation {relaxation!r} must lie in (0, 1]")
    angle = float(exit_angle)
    if not (math.isfinite(angle) and abs(angle) < 0.5 * math.pi):
        raise KuttaError(f"the exit angle {exit_angle!r} rad is not within (-pi/2, pi/2)")
    step = relaxation * correction.delta_tan
    if not math.isfinite(step):
        raise KuttaError(f"the correction of tan(beta_out) is {correction.delta_tan!r}: not finite")
    if step == 0.0:
        return angle                    # equal TE speeds: the angle stays exactly (no tan / arctan round trip)
    return math.atan(math.tan(angle) + step)
