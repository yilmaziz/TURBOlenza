"""
flow.py
=======
From the stream function to the flow at the nodes, once per pass of the density loop:

    psi -> psi_m, psi_eta -> W_m, W_theta, W       Aungier 2003 Eqs. 5-18, 5-19   (implemented)

The kinematics only. W -> h -> T, p, rho, a and the density update belong to eos.py
(static_state_from_velocity, relax_density).

W is the velocity relative to the blade row; in a stationary cascade it is the absolute
velocity. Aungier's sequence is: psi from the current density, then the velocities
(this module), then a new density from the gas model, then psi again. This module uses the
density it is given and changes nothing else: the lag rho(n) -> psi(n) -> W(n) -> rho(n+1)
belongs to the solver.

Velocity recovery [Aungier 2003 Eqs. 5-18, 5-19], in the normalization of coefficients.py
    Aungier writes W_m = mdot / (S b rho) psi_eta and W_theta = mdot / (b rho) [(tan(beta) / S)
    psi_eta - psi_m]. coefficients.py takes mdot' = mdot / b, the mass flow through one passage
    per unit stream-sheet thickness (mass_flow_per_depth): b is absorbed into it and appears
    nowhere, not even as 1. With mu = mdot' / rho, exactly the mu of A, B and C:
        W_m     = mu psi_eta / S,
        W_theta = mu [(tan(beta) / S) psi_eta - psi_m],
    S = L_ref S* (metrics.S times coefficients.length_scale), tan(beta) = metrics.tan_beta (the
    grid-line slope of Eq. 5-3, affine in eta), psi_m the derivative in the physical m
    (dm = L_ref grid.dm) and psi_eta in eta (deta = grid.deta). velocity_components reads
    mdot' and L_ref from the CoefficientFields that built the matrix, so the velocities use the
    normalization of the solved equation by construction. Units: c_ref given (metres), mdot' in
    kg/(m s) and rho in kg/m^3 give m/s; c_ref None (lengths in chords) gives the velocity unit
    of mdot' / (rho chord) that the caller chose. Equivalently W_theta = (B psi_eta - C psi_m) / S
    and W_m = C psi_eta / S^2 with B and C of Eqs. 5-23, 5-24.
    Signs, as in Eqs. 5-18, 5-19: psi rises from 0 on eta = 0 to 1 on eta = 1, so W_m > 0 is the
    flow from inlet to exit for mdot' > 0, and W_theta > 0 points to +y (+theta); a uniform flow
    at angle alpha has W_theta / W_m = tan(alpha), alpha from m toward +y as in geometry.py.

The derivatives (second order, uniform grid; the node classes are boundary_conditions.py's)
    psi_eta
        interior j = 1 .. n_eta - 2          (psi[i, j+1] - psi[i, j-1]) / (2 deta)
        blade walls (lower_wall, upper_wall) one-sided into the passage:
                                             (-3 psi[i, 0] + 4 psi[i, 1] - psi[i, 2]) / (2 deta),
                                             (3 psi[i, J] - 4 psi[i, J-1] + psi[i, J-2]) / (2 deta);
                                             no stencil reaches through the blade
        periodic lines (j = 0, J off the blade) central across the line with the periodic
                                             continuation of boundary_conditions.periodic_ghost,
                                             psi[i, -1] = psi[i, J-1] - 1 and psi[i, J+1] = psi[i, 1] + 1,
                                             the +1 applied there and only there; both images then
                                             get (psi[i, 1] - psi[i, J-1] + 1) / (2 deta)
    psi_m: by m region, the regions and the stencils of metrics.py (2026-11-13; below)
        regions                              upstream periodic 0 .. f - 1, blade f .. l,
                                             downstream periodic l + 1 .. n_m - 1 (f = first_blade_i,
                                             l = last_blade_i): metrics._runs, the regions its own
                                             d/dm of the boundaries and of tan(beta) use
        inside a region                      (psi[i+1, j] - psi[i-1, j]) / (2 dm), periodic lines
                                             included (along the line, real nodes only)
        region ends                          (-3 psi[i] + 4 psi[i+1] - psi[i+2]) / (2 dm) at the
                                             first station of a region, (3 psi[i] - 4 psi[i-1] +
                                             psi[i-2]) / (2 dm) at its last: metrics._d_dm_run.
                                             Hence backward at f - 1, forward at f, backward at l,
                                             forward at l + 1, and at the far stations i = 0 and
                                             i = n_m - 1 the forward and backward forms of
                                             boundary_conditions' FORWARD_DIFFERENCE and
                                             BACKWARD_DIFFERENCE, as before
        blade walls                          exactly 0, overriding the above: psi = 0 and 1 along the
                                             walls (Dirichlet), so d psi / dm = 0 there; hence
                                             W_theta = W_m tan(beta) on the walls
    No stencil leaves its region, so psi_m and the tan(beta) of Eq. 5-19 belong to one map at
    every node. Before 2026-11-13 psi_m was central at f - 1, f, l and l + 1, across the edge,
    while metrics.py's tan(beta) there is the one-sided slope of its own side: at the interior
    nodes of the first blade station the two belonged to different eta-lines (slopes 2.50 and
    1.86 at the NACA 65 case's critical node, 61 x 41), W_theta came out about 50 m/s too large,
    and the lagged density turned that one-station overspeed into a sonic node from M1 = 0.6175
    (README decision log, the M 0.60 against M 0.65 diagnosis). Every other node is bit for bit
    as before. The m differences are taken by metrics.py's own functions (_runs, _d_dm), not a
    copy, so the two modules cannot drift apart. No higher-order, spline or spectral
    derivative, and none of the special transition rows of discretization.py (those are PDE-row
    choices, not velocity rules). The far stations' psi comes rebuilt by
    matrix_assembly.reconstruct (linear_solver.py). The affine periodic +1 belongs to psi_eta
    only: psi_m runs along a station line's real nodes and takes no jump.

Checks (FlowRecoveryError; TypeError for objects of the wrong types)
    psi and rho (n_m, n_eta) and finite; rho > 0; S > 0, tan(beta), dm and deta finite (dm, deta
    > 0); coefficients, metrics and boundary conditions of one grid; and rho the density the
    coefficients were built from: C = mdot' S / rho and B = mdot' tan(beta) / rho at every node
    to round-off. mdot' > 0 and c_ref are those coefficients.py already accepted (it refuses a
    zero or negative mass flow). Results non-finite: refused. psi is not altered or checked
    against its boundary conditions: the velocities are those of the psi given.

Mass flow (station_mass_flow, a diagnostic)
    Eq. 5-18 gives rho S W_m = mdot' psi_eta at every node; across a station, with
    psi[i, J] - psi[i, 0] = 1, the integral of rho W_m S over eta is mdot'. station_mass_flow
    integrates the recovered rho W_m S by the trapezoidal rule. On a periodic station the
    central differences telescope with the periodic continuation and give mdot' to round-off;
    on a blade station the one-sided wall differences leave an O(deta^2) difference. Nothing is
    rescaled to make them agree.

Inlet closure (inlet_state, passage_mass_flow; stationary cascade)
    From the absolute inlet totals Pt1, Tt1 (= relative ones for omega = 0), the far-inlet
    relative Mach number M1 and angle beta1 (from the meridional direction) of FlowConditions:
        W1 = the root of W / a(W) = M1, a(W) = eos.static_state_from_velocity(W).sound_speed,
        T1, p1, rho1, a1 = eos.static_state_from_velocity(W1),
        W_m1 = W1 cos(beta1),   W_theta1 = W1 sin(beta1),   q1 = rho1 W1^2 / 2,
        mdot' = rho1 W_m1 S1,   S1 = the physical inlet pitch (the far station is periodic: its
                                passage width is the pitch), per unit stream-sheet depth b.
    The gas relations stay eos.py's. For its calorically perfect gas the root is W1 = M1
    sqrt(gamma R T1) with T1 = Tt1 / (1 + (gamma - 1) M1^2 / 2), p1 = Pt1 (T1 / Tt1)^(gamma /
    (gamma - 1)), rho1 = p1 / (R T1), but none of these formulas is written here: M = W / a(W)
    rises strictly with W on [0, sqrt(2 H')) (h falls, so a does), and W1 is found by
    bisection, to adjacent floating-point numbers, on eos's own sound speed. M1 = 0 gives W1 =
    0 and mdot' = 0 exactly (the helper allows it; the solver refuses a zero flow, as
    coefficients.py does). A rotating row is refused (its inlet needs velocity triangles, not
    implemented).

Not here at all: the static state and the density update (eos.py), the surface Mach number
(postprocess.py).

Status: velocity_components and station_mass_flow implemented 2026-11-13, under review, not
frozen; psi_m by m region since 2026-11-13 (the transition fix above); inlet_state and
passage_mass_flow implemented 2026-11-13, under review.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields

import numpy as np
from numpy.typing import NDArray

from potential_flow_solver.boundary_conditions import BACKWARD_DIFFERENCE, FORWARD_DIFFERENCE, BoundaryConditions
from potential_flow_solver.coefficients import CoefficientFields
from potential_flow_solver.config import FlowConditions
from potential_flow_solver import eos as _eos
from potential_flow_solver.eos import PerfectGas
from potential_flow_solver.grid import Grid
from potential_flow_solver.metrics import ROUND_OFF, GridMetrics, MetricsError
from potential_flow_solver.metrics import _d_dm as _region_d_dm      # the m-derivative of metrics.py: no stencil
from potential_flow_solver.metrics import _runs as _m_regions        # leaves its region (module docstring)

Field = NDArray[np.float64]


class FlowRecoveryError(ValueError):
    """The inputs of the velocity recovery are broken or do not belong together (shapes,
    non-finite or non-positive values, a density other than the coefficients' own), or the
    recovered velocities are not finite."""


@dataclass(frozen=True)
class FlowField:
    """The kinematics at the nodes, each array (n_m, n_eta) and read-only (module docstring):
    the derivatives of psi and the relative velocity of Eqs. 5-18, 5-19."""

    psi_m: Field        # d psi / dm, physical m (1 / length)
    psi_eta: Field      # d psi / d eta
    W_m: Field          # meridional velocity, Eq. 5-18
    W_theta: Field      # tangential velocity, Eq. 5-19, positive toward +y
    W2: Field           # W_m^2 + W_theta^2
    W: Field            # sqrt(W2)

    def __post_init__(self) -> None:
        for item in fields(self):
            array = np.array(getattr(self, item.name), dtype=float)
            array.flags.writeable = False
            object.__setattr__(self, item.name, array)


class InletStateError(ValueError):
    """The far-inlet state cannot be formed: no inlet Mach number, a Mach number or angle
    outside the real range, a non-positive pitch, or a rotating row (module docstring,
    "Inlet closure")."""


@dataclass(frozen=True)
class InletState:
    """The far-inlet state of a stationary cascade (module docstring, "Inlet closure"), in the
    units of the gas and the totals (SI: K, Pa, kg/m^3, m/s; mdot' in kg/(m s))."""

    mach: float                     # M1, relative = absolute (stationary)
    flow_angle: float               # beta1, radians from the meridional direction, positive toward +y
    total_pressure: float           # Pt1 (absolute = relative)
    total_temperature: float        # Tt1
    static_temperature: float       # T1
    static_pressure: float          # p1
    static_density: float           # rho1
    sound_speed: float              # a1
    velocity: float                 # W1 = M1 a1
    meridional_velocity: float      # W_m1 = W1 cos(beta1)
    tangential_velocity: float      # W_theta1 = W1 sin(beta1)
    dynamic_pressure: float         # q1 = rho1 W1^2 / 2
    pitch: float                    # S1, the physical inlet pitch (length unit of c_ref)
    mass_flow_per_depth: float      # mdot' = rho1 W_m1 S1


def inlet_state(conditions: FlowConditions, gas: PerfectGas, *, inlet_pitch: float) -> InletState:
    """The far-inlet state from conditions (Pt1, Tt1, M1, beta1) and the gas, and the passage
    mass flow per unit depth through the physical inlet pitch inlet_pitch (module docstring,
    "Inlet closure"). InletStateError for inputs it cannot close."""
    if conditions.angular_speed != 0.0:
        raise InletStateError("a rotating row needs the inlet velocity triangle (absolute totals, relative Mach number "
                              "and angle); only a stationary cascade (angular_speed = 0) is closed here")
    mach = conditions.inlet_mach
    if (mach is None or isinstance(mach, bool) or not isinstance(mach, (int, float))
            or not (math.isfinite(mach) and mach >= 0.0)):
        raise InletStateError(f"the inlet Mach number is {mach!r}: a finite number >= 0 is needed to derive the inlet state")
    beta = math.radians(float(conditions.inlet_angle_deg))
    if not (math.isfinite(beta) and abs(beta) < 0.5 * math.pi):
        raise InletStateError(f"the inlet angle {conditions.inlet_angle_deg!r} deg is not within (-90, 90)")
    if (isinstance(inlet_pitch, bool) or not isinstance(inlet_pitch, (int, float))
            or not (math.isfinite(inlet_pitch) and inlet_pitch > 0.0)):
        raise InletStateError(f"the inlet pitch {inlet_pitch!r} must be a positive length")
    total = _eos.TotalState(pressure=conditions.inlet_absolute_total_pressure,
                            temperature=conditions.inlet_absolute_total_temperature)
    reference = _eos.blade_row_reference(total, gas)
    W = _speed_at_mach(float(mach), gas, reference)
    state = _eos.static_state_from_velocity(W, gas, reference)
    W_m, W_theta = W * math.cos(beta), W * math.sin(beta)
    rho = float(state.density)
    return InletState(mach=float(mach), flow_angle=beta, total_pressure=total.pressure, total_temperature=total.temperature,
                      static_temperature=float(state.temperature), static_pressure=float(state.pressure),
                      static_density=rho, sound_speed=float(state.sound_speed), velocity=W, meridional_velocity=W_m,
                      tangential_velocity=W_theta, dynamic_pressure=0.5 * rho * W * W, pitch=float(inlet_pitch),
                      mass_flow_per_depth=passage_mass_flow(rho, W_m, float(inlet_pitch)))


def passage_mass_flow(density: float, meridional_velocity: float, inlet_pitch: float) -> float:
    """mdot' = rho1 W_m1 S1: the mass flow through one passage per unit stream-sheet depth, from
    the far-inlet density and meridional velocity and the physical inlet pitch S1 (module
    docstring, "Inlet closure")."""
    return float(density) * float(meridional_velocity) * float(inlet_pitch)


def _speed_at_mach(mach: float, gas: PerfectGas, reference) -> float:
    """W with W / a(W) = mach, a from eos.static_state_from_velocity, by bisection on
    [0, sqrt(2 H')) to adjacent floating-point numbers (module docstring)."""
    if mach == 0.0:
        return 0.0

    def excess(W: float) -> float:            # W - M a(W): rises strictly with W
        try:
            return W - mach * float(_eos.static_state_from_velocity(W, gas, reference).sound_speed)
        except _eos.ThermodynamicStateError:  # T <= 0: beyond any Mach number
            return math.inf

    low, high = 0.0, math.sqrt(2.0 * reference.relative_total_enthalpy)
    while True:
        middle = 0.5 * (low + high)
        if not low < middle < high:
            break
        if excess(middle) < 0.0:
            low = middle
        else:
            high = middle
    return low if abs(excess(low)) <= abs(excess(high)) else high


def velocity_components(psi: Field, density: Field, coefficients: CoefficientFields, metrics: GridMetrics,
                        boundary_conditions: BoundaryConditions) -> FlowField:
    """psi_m, psi_eta, W_m, W_theta and W at every node (module docstring) from the solved psi
    (n_m, n_eta), the density field psi was computed with, the CoefficientFields built from it
    (mdot' and the length scale are read there), the metrics and the boundary conditions of the
    same grid."""
    if not isinstance(coefficients, CoefficientFields):
        raise TypeError(f"velocity_components takes CoefficientFields, got {type(coefficients).__name__}")
    if not isinstance(metrics, GridMetrics):
        raise TypeError(f"velocity_components takes GridMetrics, got {type(metrics).__name__}")
    if not isinstance(boundary_conditions, BoundaryConditions):
        raise TypeError(f"velocity_components takes BoundaryConditions, got {type(boundary_conditions).__name__}")
    grid = metrics.grid
    if boundary_conditions.grid is not grid:
        raise FlowRecoveryError("the boundary conditions were built from another grid than the metrics")
    shape = (len(grid.m), len(grid.eta))
    psi = _field(psi, shape, "psi")
    rho = _field(density, shape, "the density")
    if not np.all(rho > 0.0):
        i, j = np.argwhere(~(rho > 0.0))[0]
        raise FlowRecoveryError(f"the density is {rho[i, j]!r} at node ({i}, {j}): it must be positive")
    if coefficients.A.shape != shape:
        raise FlowRecoveryError(f"the coefficient fields are {coefficients.A.shape}, the grid has {shape} nodes")
    length = coefficients.length_scale
    mdot = coefficients.mass_flow_per_depth
    S = length * np.asarray(metrics.S, dtype=float)
    tan_beta = np.asarray(metrics.tan_beta, dtype=float)
    dm, deta = length * float(grid.dm), float(grid.deta)
    if not (np.all(np.isfinite(S)) and np.all(S > 0.0)):
        raise FlowRecoveryError("S must be finite and positive at every station")
    if not np.all(np.isfinite(tan_beta)) or tan_beta.shape != shape:
        raise FlowRecoveryError(f"tan(beta) must be a finite {shape} field")
    for name, value in (("dm", dm), ("deta", deta)):
        if not (np.isfinite(value) and value > 0.0):
            raise FlowRecoveryError(f"{name} = {value!r}: a grid spacing must be finite and positive")
    _check_density_of(coefficients, rho, S, tan_beta, mdot)
    psi_m, psi_eta = _derivatives(psi, dm, deta, boundary_conditions)
    W_m, W_theta = _velocities(psi_m, psi_eta, rho, mdot, S[:, None], tan_beta)
    W2 = W_m ** 2 + W_theta ** 2
    out = FlowField(psi_m=psi_m, psi_eta=psi_eta, W_m=W_m, W_theta=W_theta, W2=W2, W=np.sqrt(W2))
    for item in fields(out):
        if not np.all(np.isfinite(getattr(out, item.name))):
            raise FlowRecoveryError(f"the recovered {item.name} is not finite")
    return out


def station_mass_flow(flow: FlowField, density: Field, coefficients: CoefficientFields,
                      metrics: GridMetrics) -> NDArray[np.float64]:
    """The mass flow per unit depth through every station, (n_m,): the integral over eta of
    rho W_m S by the trapezoidal rule (module docstring, "Mass flow"). A diagnostic, to
    compare with coefficients.mass_flow_per_depth."""
    if not isinstance(flow, FlowField):
        raise TypeError(f"station_mass_flow takes a FlowField, got {type(flow).__name__}")
    grid = metrics.grid
    shape = (len(grid.m), len(grid.eta))
    rho = _field(density, shape, "the density")
    if flow.W_m.shape != shape:
        raise FlowRecoveryError(f"the flow field is {flow.W_m.shape}, the grid has {shape} nodes")
    S = coefficients.length_scale * np.asarray(metrics.S, dtype=float)
    integrand = rho * flow.W_m * S[:, None]
    deta = float(grid.deta)
    return deta * (integrand.sum(axis=1) - 0.5 * (integrand[:, 0] + integrand[:, -1]))


# --- the two steps ---------------------------------------------------------------------------

def _derivatives(psi: Field, dm: float, deta: float, bcs: BoundaryConditions) -> tuple[Field, Field]:
    """psi_m and psi_eta by the rules of the module docstring; the node classes, the far
    stations and the periodic continuation are those of bcs."""
    n_m, n_eta = psi.shape
    if bcs.shape != (n_m, n_eta):
        raise FlowRecoveryError(f"psi is {psi.shape}, the boundary conditions are for {bcs.shape}")
    if n_m < 3 or n_eta < 3:
        raise FlowRecoveryError(f"{n_m} x {n_eta} nodes: the second-order differences need three a side")
    J = n_eta - 1
    psi_eta = np.empty_like(psi)
    psi_eta[:, 1:-1] = (psi[:, 2:] - psi[:, :-2]) / (2.0 * deta)
    blade = np.flatnonzero(np.asarray(bcs.lower_wall)[:, 0])
    if not np.array_equal(blade, np.flatnonzero(np.asarray(bcs.upper_wall)[:, -1])):
        raise FlowRecoveryError("the lower and upper walls are not on the same stations")
    f0, f1, f2 = FORWARD_DIFFERENCE
    b0, b1, b2 = BACKWARD_DIFFERENCE
    psi_eta[blade, 0] = (f0 * psi[blade, 0] + f1 * psi[blade, 1] + f2 * psi[blade, 2]) / (2.0 * deta)
    psi_eta[blade, J] = (b0 * psi[blade, J] + b1 * psi[blade, J - 1] + b2 * psi[blade, J - 2]) / (2.0 * deta)
    for i in np.flatnonzero(np.asarray(bcs.periodic_lower)[:, 0]):
        below = bcs.periodic_ghost(int(i), -1).value(psi)           # psi[i, -1] = psi[i, J - 1] - 1
        above = bcs.periodic_ghost(int(i), n_eta).value(psi)        # psi[i, J + 1] = psi[i, 1] + 1
        psi_eta[i, 0] = (psi[i, 1] - below) / (2.0 * deta)
        psi_eta[i, J] = (above - psi[i, J - 1]) / (2.0 * deta)

    upstream, downstream = np.asarray(bcs.upstream), np.asarray(bcs.downstream)
    if not (upstream[0].all() and downstream[-1].all() and upstream.sum() == n_eta == downstream.sum()):
        raise FlowRecoveryError("the far stations of the boundary conditions are not i = 0 and i = n_m - 1")
    if blade.size == 0 or not np.array_equal(blade, np.arange(blade[0], blade[-1] + 1)):
        raise FlowRecoveryError("the blade stations of the boundary conditions are not one run")
    try:
        regions = _m_regions(n_m, int(blade[0]), int(blade[-1]))
    except MetricsError as error:
        raise FlowRecoveryError(f"psi_m by m region: {error}") from error
    psi_m = _region_d_dm(psi, dm, regions)                         # central inside a region, one-sided at its ends
    psi_m[np.asarray(bcs.blade_wall)] = 0.0                        # the walls take precedence
    return psi_m, psi_eta


def _velocities(psi_m, psi_eta, rho, mdot: float, S, tan_beta) -> tuple[Field, Field]:
    """Eqs. 5-18, 5-19 with mu = mdot' / rho (module docstring); S broadcasts against the
    (n_m, n_eta) fields."""
    mu = mdot / rho
    return mu * psi_eta / S, mu * (tan_beta / S * psi_eta - psi_m)


def _field(value, shape, name: str) -> Field:
    try:
        array = np.array(value, dtype=float)
    except (TypeError, ValueError):
        raise FlowRecoveryError(f"{name} must be a numeric {shape} array") from None
    if array.shape != shape:
        raise FlowRecoveryError(f"{name} has shape {array.shape}, the grid needs {shape}")
    if not np.all(np.isfinite(array)):
        i, j = np.argwhere(~np.isfinite(array))[0]
        raise FlowRecoveryError(f"{name} is not finite at node ({i}, {j})")
    return array


def _check_density_of(coefficients: CoefficientFields, rho: Field, S: Field, tan_beta: Field, mdot: float) -> None:
    """FlowRecoveryError unless C = mdot' S / rho and B = mdot' tan(beta) / rho at every node to
    round-off: rho, mdot', the length scale and these metrics are the coefficients' own."""
    C, B = np.asarray(coefficients.C, dtype=float), np.asarray(coefficients.B, dtype=float)
    off_C = np.abs(C * rho - mdot * S[:, None]) / (mdot * S[:, None])
    off_B = np.abs(B * rho - mdot * tan_beta) / (mdot * np.maximum(1.0, np.abs(tan_beta)))
    for name, off in (("C = mdot' S / rho", off_C), ("B = mdot' tan(beta) / rho", off_B)):
        bad = np.argwhere(~(off <= ROUND_OFF))
        if bad.size:
            i, j = (int(k) for k in bad[0])
            raise FlowRecoveryError(f"{name} fails at node ({i}, {j}) by {off[i, j]:.3g} (relative): the density, the "
                                    f"metrics and the coefficients do not belong together. Recover the velocities with "
                                    f"the density and the coefficients psi was solved with")
