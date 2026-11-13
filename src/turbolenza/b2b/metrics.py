"""
metrics.py
==========
The geometric fields of Aungier's transformed blade-to-blade equations on a grid from
grid.py [Aungier 2003 Secs. 5.2, 5.3; Eqs. 5-1, 5-2, 5-3, 5-9]. Pure geometry: no flow,
gas, density or stream-sheet quantity enters, so the metrics of a grid are computed once
and again only when rebuild_exit turns its exit extension.

Planar cascade: y replaces r theta
    Aungier writes the equations on a stream sheet in (m, theta) at radius r. This solver
    works on a flat linear cascade whose tangential coordinate is already unrolled:
    y = r theta is the physical tangential distance of geometry.py and grid.py. So r theta
    is y wherever it appears and r d theta/dm is dy/dm; no radius is used, faked (r = 1)
    or stored.

Coordinates [Eqs. 5-1, 5-2]
    xi  = integral dm / cos(beta)                 (not stored: d xi/dm = 1 / cos(beta))
    eta = (theta - theta0) / (theta1 - theta0) = (y - y0(m)) / S(m)
    y0(m) is the lower (eta = 0) and y1(m) the upper (eta = 1) boundary of the grid: the
    side lines on the extensions and the passage-facing blade walls in between, exactly as
    grid.py chose them (metrics never choose a wall or read its suction/pressure name).

Passage spacing [Eq. 5-9]
    S(m) = r (theta1 - theta0) = y1(m) - y0(m),
    the tangential width of the mapped passage at constant m. It is not the geometric
    throat, the least Euclidean distance between the blades (geometry.py's
    throat_opening): S runs along y, the throat across the passage, and min S is not the
    throat (0.8779 c against 0.7215 c for the 65-(12A10)10 at 33 deg). On the periodic
    extensions S is the pitch.

Mapping and Jacobian
    y(m, eta) = y0(m) + eta S(m), so
        d(m, y) / d(m, eta) = [[1, 0], [(dy/dm)_eta, S]],     det J = S.
    S > 0 keeps the orientation, S = 0 collapses the map and S < 0 reverses it. That is a
    coordinate-transformation criterion, not a cell-volume one (this is a node grid).
    grid.py already refuses a smooth passage narrower than 1e-12 chord anywhere; here an
    S at or below the round-off of the tangential coordinates is refused as a local
    backstop. jacobian_det is S itself: no (n_m, n_eta) copy.

Grid-line angle [Eq. 5-3]
    tan(beta) = [r d theta / dm]_eta = (dy/dm)_eta. Differentiating the mapping at fixed eta,
        (dy/dm)_eta = dy0/dm + eta dS/dm,      dS/dm = dy1/dm - dy0/dm,
    so tan(beta) = tan(beta0) + eta (tan(beta1) - tan(beta0)), Eq. 5-3, with
    tan(beta0) = dy0/dm and tan(beta1) = dy1/dm the slopes of the two boundaries. tan(beta),
    not beta, is linear in eta; it is evaluated as (1 - eta) tan(beta0) + eta tan(beta1), the
    form grid.py uses for the nodes, so that eta = 0 and 1 return the boundary slopes
    exactly. beta = arctan(tan(beta)) lies in (-pi/2, pi/2): every station is a line x = m
    and every constant-eta line a graph over m. beta is the angle of the constant-eta grid
    line, a property of the grid. It is not the flow angle, except at the far inlet and exit
    stations, where the side lines are built along the flow angle for Aungier's far
    boundary condition (Eq. 5-33).

Derivatives in m, by region
    The boundaries change kind twice: inlet side line -> blade wall at first_blade_i, and
    blade wall -> exit side line after last_blade_i. The straight links across those
    transitions are not grid edges (grid.py section 6: they can cross the nose or the
    tail). So d/dm of a boundary is taken in three separate runs, and no stencil reaches
    across a transition:
        U  i = 0 .. first_blade_i - 1            inlet side lines
        B  i = first_blade_i .. last_blade_i     blade walls
        D  i = last_blade_i + 1 .. n_m - 1       exit side lines
    On the uniform dm of each run, second order throughout (the central and one-sided
    forms of Aungier's differences):
        interior   f'_i = (f_{i+1} - f_{i-1}) / (2 dm)
        first      f'_i = (-3 f_i + 4 f_{i+1} - f_{i+2}) / (2 dm)
        last       f'_i = (3 f_i - 4 f_{i-1} + f_{i-2}) / (2 dm)
    So the first blade slope uses stations first_blade_i .. first_blade_i + 2 and the last
    one last_blade_i - 2 .. last_blade_i, never a side-line node; the last inlet and first
    exit slopes use side-line nodes only. A run needs three stations; fewer is a
    MetricsError (raise n_passage or the extension), never a silent first-order fallback.
    That is the whole story for a C1 grid, and for the side lines of a C2 grid. The blade
    walls of a C2 grid take no difference at all (next section).

C2 blade walls: the authoritative curve's own derivatives
    A C2 grid (grid.has_wall_curves) has its blade-wall nodes on the parametric curves
    X(u) = (m(u), y(u)) of geometry.py, and grid.wall_curve_derivative(k) hands on X_u
    (k = 1) and X_uu (k = 2) of the same curve at the same u as each wall node (grid.py
    section 7; order 0 is the node itself, checked here against the nodes). With m a
    regular coordinate along the wall (m_u != 0),
        tan(beta_wall)       = dy/dm   = y_u / m_u,
        d tan(beta_wall)/dm  = d2y/dm2 = (y_uu m_u - y_u m_uu) / m_u^3,
    exact derivatives of the spline at the node, for tan_beta0 / tan_beta1 and
    dtan_beta0_dm / dtan_beta1_dm on first_blade_i .. last_blade_i. No neighbouring node
    enters, so nothing at the blade run's ends is one-sided: the first and last blade
    stations carry the wall's true slope, however steep. A rounded nose or tail turns
    through a vertical tangent, where m_u = 0: the curve and its curvature are smooth there,
    but y(m) is not (d2y/dm2 = kappa / cos^3(beta)). Grid stations never sit on that
    point (a line through a nose tip or a tail end only touches the blade and is periodic,
    grid.py section 3), so the blade walls' values are finite, though large next to it. A
    wall node with |m_u| <= SINGULAR_COS |X_u| (cos(beta) numerically zero) is a
    MetricsError naming the wall, the station, its m and |m_u| (its u is in
    grid.wall_identity): no clipping, capping or smoothing of beta, tan(beta) or its
    derivative, anywhere.

d tan(beta)/dm, d tan(beta)/d eta, and what is not stored
    dtan_beta0_dm, dtan_beta1_dm: exact on C2 blade walls (above); elsewhere the region-wise
    d/dm of tan_beta0, tan_beta1 (the stencils above, inside each run, no transition
    crossed). The C1 blade walls get differences of differences there: at the rounded
    ends that compounds the end error of the slopes (next section), so on a C1 grid these
    two are a like-for-like comparison of the C1 treatment, not a reference.
    In the passage, from the mapping at fixed m (no blade curve is evaluated inside):
        d tan(beta)/dm    = (1 - eta) dtan_beta0_dm + eta dtan_beta1_dm      (dtan_beta_dm),
        d tan(beta)/d eta = tan_beta1 - tan_beta0 = dS/dm,
    the second exactly the array dS_dm, read as the property dtan_beta_deta (no copy, as
    jacobian_det is S). d2S/dm2 = dtan_beta1_dm - dtan_beta0_dm is not stored: none of the
    coefficients A to E of Eq. 5-21 needs it, and a consumer forms it exactly from these
    two fields.

Accuracy of the slopes (C1, and the side lines of either; measured for the EXAMPLE;
geometric, not a flow convergence)
    These are the finite-difference slopes. scripts/plot_metrics_accuracy.py audited them
    against the smooth walls (2026-11-08): the differences are sound (second order with the
    textbook constants on manufactured functions), and their error is local, at the
    rounded edges, set by the station phase there. The C2 blade walls do not go through
    this approximation at all: they take the spline derivative itself.
    Side lines: y1 = y0 + pitch, so tan(beta0) = tan(beta1), dS/dm = 0 and S = pitch there to
    round-off; the lines are smooth and the slopes converge at second order to their exact
    law dy/dm = tan(beta(m)), beta linear in m (orders 1.95 to 2.03), the largest error at
    the one-sided run ends.
    Blade walls, away from their ends (>= 0.05 c): the walls are grid.py's smooth walls, C1
    piecewise cubics through the Cascade's vertices, and the slopes converge to those
    walls' own slopes, lower/upper wall 1.6e-2/3.9e-3 at 31 passage cells, 7.1e-4/8.4e-4 at
    201, between first and second order and not monotonically (station phase; a stencil
    across a vertex sees the walls' curvature jump). Those walls' slopes differ from the
    same blade at 1601 camber stations by up to 4.0e-3/4.6e-3, at the knots of the
    thickness table (station 0.50 above all), where geometry.py's C1 thickness steps in
    curvature: a floor set by n_camber_stations (median 7e-5). The Kovach junction
    (station 0.60) is a true slope break of 1.7e-3/2.1e-3 that a stencil across it keeps
    at any dm.
    Blade ends: at the first blade stations (the unresolved nose: f to about f + 3) and the
    last two, the walls turn toward a vertical tangent across the stencil, and the slopes
    carry errors of 0.04 to 3, set by where the stations fall, not shrinking with dm; at
    coarse grids the upper-wall slope at f can even have the wrong sign (31 and 51 cells).
    At a sharp TE the end slopes are finite and the errors smaller (below 0.2). The extreme
    values of the fields, min cos(beta) (0.30 to 0.45 for the EXAMPLE, at f on the lower
    wall) and max |dS/dm| (1.6 to 4.5), sit there. A resolution limit of the H-grid at a
    rounded edge, for the LE and TE treatment of coefficients.py and discretization.py to
    take into account.

Fields (GridMetrics) and units
    S, jacobian_det  (n_m,)         length: chords for a chord-1 cascade, metres when dimensional
    dS_dm            (n_m,)         dimensionless, = tan_beta1 - tan_beta0
    tan_beta0/1      (n_m,)         dimensionless, slopes of the eta = 0 and eta = 1 lines
    tan_beta         (n_m, n_eta)   dimensionless, Eq. 5-3
    beta             (n_m, n_eta)   radians, in (-pi/2, pi/2)
    sin_beta         (n_m, n_eta)   dimensionless
    cos_beta         (n_m, n_eta)   dimensionless, > sqrt(machine epsilon) (see below)
    dtan_beta0/1_dm  (n_m,)         1/length: d tan(beta0)/dm, d tan(beta1)/dm (d2y/dm2 of the boundaries)
    dtan_beta_dm     (n_m, n_eta)   1/length: affine in eta, (1 - eta) dtan_beta0_dm + eta dtan_beta1_dm
    dtan_beta_deta   (n_m,)         dimensionless: property, the array dS_dm (tan(beta) is affine in eta)
    No unit is converted: lengths are those of the Grid.

Singular grid angle
    d xi/dm = 1 / cos(beta) [Eq. 5-1] and the coefficient A of Eq. 5-22 carries
    1 / cos^2(beta); cos(beta) = 0 is a singularity of the coordinates. build_metrics
    refuses cos(beta) <= sqrt(machine epsilon) = 1.5e-8 (|tan(beta)| above 6.7e7), where
    1 / cos^2(beta) reaches 1 / epsilon and the difference equations lose all precision;
    no steep-but-finite blade slope is refused. On a C2 blade wall the same threshold is
    applied to the curve's tangent, |m_u| / |X_u|, before anything is divided by m_u.

Consumers (not implemented here)
    coefficients.py builds Aungier's coefficients of Eq. 5-21 from these fields, the
    density and the stream-sheet thickness b:
        A = mdot / (S b rho cos^2 beta),   B = mdot tan(beta) / (b rho),   C = mdot S / (b rho),
        D = dA/deta - dB/dm,               E = dC/dm - dB/deta.
    flow.py recovers the velocities:
        W_m = mdot / (S b rho) dpsi/deta,  W_theta = mdot / (b rho) [(tan(beta) / S) dpsi/deta - dpsi/dm],
    so errors in S and tan(beta) pass straight into the velocity. Density, b and mdot do
    not belong here; nor do xi, A to E, or any derivative of them.

Status: implemented; reopened 2026-11-11 for the C2 wall derivatives (C1 fields unchanged
bit for bit; the three dtan_beta fields are new for both). Not frozen: under review. The
solver goes with C2 only for now, so its blade-wall angles are the C2 curves' own
derivatives; a fair scientific comparison of C1 against C2 is planned for later, and the
C1 finite-difference path is kept unchanged until then.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np
from numpy.typing import NDArray

from turbolenza.b2b.grid import Grid

EPSILON = float(np.finfo(float).eps)
ROUND_OFF = 64.0 * EPSILON          # relative tolerance of the contract checks: times the m scale (largest of |m_in|,
                                    # |m_out| and their range) for m, times max(|y|, pitch) for y and S
SINGULAR_COS = float(np.sqrt(EPSILON))  # cos(beta) at or below this is numerically singular: 1/cos^2 >= 1/eps

Field = NDArray[np.float64]


class MetricsError(ValueError):
    """The grid's arrays break what the metric formulas assume (shapes, finite values,
    strictly increasing uniform m and eta, nodes on their stations and on the linear eta
    map, S = pitch on the extensions, three stations in each m region, C2 wall derivatives
    that belong to the wall nodes), or the mapping is singular: S not above round-off, or
    cos(beta) numerically zero (on a C2 blade wall: m_u numerically zero)."""


@dataclass(frozen=True)
class GridMetrics:
    """The geometric fields of one grid, at its nodes (module docstring for the equations
    and the units). grid is the Grid they belong to, kept by reference: m, eta, the pitch
    and the blade-station indices are read there, not copied. Arrays are read-only."""

    grid: Grid
    S: Field                        # (n_m,), length: y1 - y0, tangential passage width at constant m [Eq. 5-9]
    dS_dm: Field                    # (n_m,), dimensionless: tan_beta1 - tan_beta0
    tan_beta0: Field                # (n_m,), dimensionless: dy0/dm, slope of the eta = 0 line
    tan_beta1: Field                # (n_m,), dimensionless: dy1/dm, slope of the eta = 1 line
    tan_beta: Field                 # (n_m, n_eta), dimensionless: Eq. 5-3, affine in eta
    beta: Field                     # (n_m, n_eta), radians: grid-line angle, arctan(tan_beta)
    sin_beta: Field                 # (n_m, n_eta), dimensionless
    cos_beta: Field                 # (n_m, n_eta), dimensionless, > SINGULAR_COS
    dtan_beta0_dm: Field            # (n_m,), 1/length: d tan(beta0)/dm; exact on C2 blade walls
    dtan_beta1_dm: Field            # (n_m,), 1/length: d tan(beta1)/dm; exact on C2 blade walls
    dtan_beta_dm: Field             # (n_m, n_eta), 1/length: (1 - eta) dtan_beta0_dm + eta dtan_beta1_dm

    def __post_init__(self) -> None:
        if not isinstance(self.grid, Grid):
            raise TypeError(f"GridMetrics takes a Grid, got {type(self.grid).__name__}")
        n_m, n_eta = len(self.grid.m), len(self.grid.eta)
        for item in fields(self):
            if item.name == "grid":
                continue
            array = np.array(getattr(self, item.name), dtype=float)
            shape = (n_m,) if item.name in _STATION_FIELDS else (n_m, n_eta)
            if array.shape != shape:
                raise MetricsError(f"{item.name} has shape {array.shape}, the grid needs {shape}")
            array.flags.writeable = False
            object.__setattr__(self, item.name, array)

    @property
    def jacobian_det(self) -> Field:
        """det of d(m, y)/d(m, eta), (n_m,): S itself, for it does not depend on eta."""
        return self.S

    @property
    def dtan_beta_deta(self) -> Field:
        """d tan(beta)/d eta at constant m, (n_m,): tan_beta1 - tan_beta0, which is dS_dm
        itself, for tan(beta) is affine in eta."""
        return self.dS_dm

    @property
    def exact_wall_slopes(self) -> bool:
        """True when the blade-wall slopes and their m derivatives are the C2 curve's own
        derivatives (grid.has_wall_curves); False when they are differences of the nodes."""
        return self.grid.has_wall_curves


_STATION_FIELDS = ("S", "dS_dm", "tan_beta0", "tan_beta1", "dtan_beta0_dm", "dtan_beta1_dm")


def build_metrics(grid: Grid) -> GridMetrics:
    """The metrics of a ready Grid (module docstring). Reads grid.m, grid.eta,
    grid.nodes, the lower and upper boundaries, the blade-station indices, dm, deta and the
    pitch, and checks the contract the formulas rely on before using them. For a C2 grid
    (grid.has_wall_curves) it also reads grid.wall_curve_derivative(0, 1, 2), the
    authoritative curve at the wall nodes, after checking that order 0 is those nodes.

    Raises TypeError for anything but a Grid and MetricsError when the contract fails or
    the mapping is singular."""
    if not isinstance(grid, Grid):
        raise TypeError(f"build_metrics takes a Grid, got {type(grid).__name__}")
    nodes = np.asarray(grid.nodes, dtype=float)
    lower, upper = np.asarray(grid.lower_boundary, dtype=float), np.asarray(grid.upper_boundary, dtype=float)
    if nodes.ndim != 3 or lower.shape != (nodes.shape[0], 2) or upper.shape != (nodes.shape[0], 2):
        raise MetricsError(f"the boundaries have shapes {lower.shape} and {upper.shape}, the nodes {nodes.shape}: "
                           f"each boundary must be (n_m, 2)")
    if not (np.array_equal(nodes[:, 0], lower) and np.array_equal(nodes[:, -1], upper)):
        raise MetricsError("the lower and upper boundaries are not the eta = 0 and eta = 1 rows of the nodes")
    walls = _wall_derivatives(grid, nodes) if grid.has_wall_curves else None
    values = _metric_fields(grid.m, grid.eta, nodes, grid.first_blade_i, grid.last_blade_i, grid.pitch, walls)
    for name, spacing, expected in (("dm", grid.dm, values.pop("_dm")), ("deta", grid.deta, values.pop("_deta"))):
        if not abs(spacing - expected) <= ROUND_OFF * abs(expected):
            raise MetricsError(f"grid.{name} = {spacing:.17g} is not the spacing of its stations, {expected:.17g}")
    return GridMetrics(grid=grid, **values)


def _wall_derivatives(grid: Grid, nodes: Field) -> tuple[Field, Field, Field, Field]:
    """C2: (X_u lower, X_u upper, X_uu lower, X_uu upper) of the authoritative curves at the
    wall nodes of the blade stations, each (n_blade, 2), from grid.wall_curve_derivative,
    once its order-0 points are checked to be those very nodes (to round-off)."""
    blade = slice(grid.first_blade_i, grid.last_blade_i + 1)
    m = np.asarray(grid.m, dtype=float)
    scale = max(abs(m[0]), abs(m[-1]), m[-1] - m[0], float(np.max(np.abs(nodes[..., 1]))), grid.pitch)
    for name, points, wall in zip(("lower", "upper"), grid.wall_curve_derivative(0), (nodes[blade, 0], nodes[blade, -1])):
        points = np.asarray(points, dtype=float)
        off = np.abs(points - wall) if points.shape == wall.shape else np.array([np.inf])
        if not np.all(off <= ROUND_OFF * scale):
            raise MetricsError(f"the {name} wall's curve points are up to {float(np.max(off)):.3g} off its wall nodes: "
                               f"the curve derivatives would not belong to the nodes")
    (d1_lower, d1_upper), (d2_lower, d2_upper) = grid.wall_curve_derivative(1), grid.wall_curve_derivative(2)
    return d1_lower, d1_upper, d2_lower, d2_upper


def _curve_slopes(wall_derivatives: tuple[Field, Field, Field, Field], m: Field, first_blade_i: int,
                  last_blade_i: int) -> tuple[Field, Field, Field, Field]:
    """(tan_beta0, tan_beta1, dtan_beta0_dm, dtan_beta1_dm) on the blade stations from the
    curves' X_u and X_uu at the wall nodes (module docstring, "C2 blade walls"):
    tan = y_u / m_u and d tan / dm = (y_uu m_u - y_u m_uu) / m_u^3, refused where
    |m_u| <= SINGULAR_COS |X_u|."""
    n_blade = last_blade_i - first_blade_i + 1
    arrays = [np.asarray(a, dtype=float) for a in wall_derivatives]
    if len(arrays) != 4 or any(a.shape != (n_blade, 2) for a in arrays):
        raise MetricsError(f"the C2 wall derivatives must be four (n_blade, 2) = ({n_blade}, 2) arrays, got shapes "
                           f"{[a.shape for a in arrays]}")
    if not all(np.all(np.isfinite(a)) for a in arrays):
        raise MetricsError("the C2 wall derivatives are not finite")
    out = {}
    for name, d1, d2 in (("lower", arrays[0], arrays[2]), ("upper", arrays[1], arrays[3])):
        m_u, y_u, m_uu, y_uu = d1[:, 0], d1[:, 1], d2[:, 0], d2[:, 1]
        speed = np.hypot(m_u, y_u)
        singular = np.flatnonzero(~(np.abs(m_u) > SINGULAR_COS * speed))
        if singular.size:
            k = int(singular[0])
            i = first_blade_i + k
            cosine = abs(m_u[k]) / speed[k] if speed[k] > 0.0 else 0.0
            raise MetricsError(f"the {name} blade wall is numerically vertical at blade station {i} (m = {m[i]:.10g}): "
                               f"|dm/du| = {abs(m_u[k]):.3g} against |dX/du| = {speed[k]:.3g}, so cos(beta) = "
                               f"{cosine:.3g} is not above {SINGULAR_COS:.3g}. The curve is smooth there, but m is no "
                               f"coordinate along it: tan(beta) = y_u / m_u and d tan(beta)/dm = (y_uu m_u - y_u m_uu) "
                               f"/ m_u^3 do not exist")
        out[name] = (y_u / m_u, (y_uu * m_u - y_u * m_uu) / m_u ** 3)
    return out["lower"][0], out["upper"][0], out["lower"][1], out["upper"][1]


def _metric_fields(m: Field, eta: Field, nodes: Field, first_blade_i: int, last_blade_i: int,
                   pitch: float, wall_derivatives: tuple[Field, Field, Field, Field] | None = None
                   ) -> dict[str, Field | float]:
    """The metric fields from the raw arrays of a grid: the whole computation and every
    check of build_metrics but the Grid-object ones. A separate function so that the
    checks can be tried on arrays no valid Grid would hold. wall_derivatives, for a C2 grid,
    is (X_u lower, X_u upper, X_uu lower, X_uu upper) of the authoritative curves at the
    blade stations' wall nodes; None differentiates the wall nodes (C1)."""
    m, eta, nodes = (np.asarray(a, dtype=float) for a in (m, eta, nodes))
    if m.ndim != 1 or eta.ndim != 1 or nodes.shape != (len(m), len(eta), 2):
        raise MetricsError(f"m {m.shape}, eta {eta.shape} and nodes {nodes.shape} do not fit: nodes must be "
                           f"(n_m, n_eta, 2)")
    if len(eta) < 2:
        raise MetricsError(f"eta has {len(eta)} node(s); a passage needs at least its two walls")
    for name, array in (("m", m), ("eta", eta), ("nodes", nodes)):
        bad = np.argwhere(~np.isfinite(array))
        if bad.size:
            raise MetricsError(f"{name} is not finite at index {tuple(int(k) for k in bad[0])}")
    if not np.isfinite(pitch) or not pitch > 0.0:
        raise MetricsError(f"the pitch {float(pitch):.17g} must be finite and positive")

    m_scale = max(abs(m[0]), abs(m[-1]), m[-1] - m[0])
    dm, deta = _uniform_spacing(m, "m", m_scale), _uniform_spacing(eta, "eta", 1.0)
    off = np.flatnonzero(~(np.abs(nodes[:, :, 0] - m[:, None]) <= ROUND_OFF * m_scale).all(axis=1))
    if off.size:
        raise MetricsError(f"the nodes of station {off[0]} do not lie on its line x = m = {m[off[0]]:.17g}")
    runs = _runs(len(m), first_blade_i, last_blade_i)

    y0, y1, y = nodes[:, 0, 1], nodes[:, -1, 1], nodes[:, :, 1]
    scale = max(float(np.max(np.abs(y))), pitch)
    S = y1 - y0
    singular = np.flatnonzero(~(S > ROUND_OFF * scale))
    if singular.size:
        i = int(singular[0])
        raise MetricsError(f"S = y1 - y0 = {S[i]:.6g} at station {i} (m = {m[i]:.10g}) is not above round-off "
                           f"({ROUND_OFF * scale:.3g}): the map from (m, eta) collapses or reverses there")
    mapped = y0[:, None] + eta[None, :] * S[:, None]
    mismatch = np.abs(mapped - y)
    if not np.all(mismatch <= ROUND_OFF * scale):
        i, j = np.unravel_index(int(np.argmax(mismatch)), mismatch.shape)
        raise MetricsError(f"the node ({i}, {j}) is {mismatch[i, j]:.3g} off y0 + eta S: the nodes are not the "
                           f"linear eta map between the boundaries")
    for name, start, stop in runs:
        if name != "blade":
            drift = np.abs(S[start:stop] - pitch)
            if not np.all(drift <= ROUND_OFF * scale):
                i = start + int(np.argmax(drift))
                raise MetricsError(f"S = {S[i]:.17g} at {name} station {i} is not the pitch {pitch:.17g}: the side lines "
                                   f"must be one pitch apart")

    tan_beta0, tan_beta1 = _d_dm(y0, dm, runs), _d_dm(y1, dm, runs)
    blade, exact = slice(first_blade_i, last_blade_i + 1), None
    if wall_derivatives is not None:            # C2: the curve's own slope on the blade walls, no difference
        exact = _curve_slopes(wall_derivatives, m, first_blade_i, last_blade_i)
        tan_beta0[blade], tan_beta1[blade] = exact[0], exact[1]
    dtan_beta0_dm, dtan_beta1_dm = _d_dm(tan_beta0, dm, runs), _d_dm(tan_beta1, dm, runs)
    if exact is not None:
        dtan_beta0_dm[blade], dtan_beta1_dm[blade] = exact[2], exact[3]
    tan_beta = (1.0 - eta)[None, :] * tan_beta0[:, None] + eta[None, :] * tan_beta1[:, None]
    cos_beta = 1.0 / np.hypot(1.0, tan_beta)
    flat = np.flatnonzero(~(cos_beta > SINGULAR_COS).ravel())
    if flat.size:
        i, j = np.unravel_index(int(flat[0]), cos_beta.shape)
        raise MetricsError(f"cos(beta) = {cos_beta[i, j]:.3g} at node ({i}, {j}), beta = "
                           f"{np.degrees(np.arctan(tan_beta[i, j])):.10g} deg: the grid line there is numerically "
                           f"vertical (tan(beta) = {tan_beta[i, j]:.3g}) and 1/cos(beta) singular")
    dtan_beta_dm = (1.0 - eta)[None, :] * dtan_beta0_dm[:, None] + eta[None, :] * dtan_beta1_dm[:, None]
    return dict(S=S, dS_dm=tan_beta1 - tan_beta0, tan_beta0=tan_beta0, tan_beta1=tan_beta1, tan_beta=tan_beta,
                beta=np.arctan(tan_beta), sin_beta=tan_beta * cos_beta, cos_beta=cos_beta,
                dtan_beta0_dm=dtan_beta0_dm, dtan_beta1_dm=dtan_beta1_dm, dtan_beta_dm=dtan_beta_dm, _dm=dm, _deta=deta)


def _uniform_spacing(values: Field, name: str, scale: float) -> float:
    """The constant step of strictly increasing, uniformly spaced values."""
    steps = np.diff(values)
    if not np.all(steps > 0.0):
        k = int(np.flatnonzero(~(steps > 0.0))[0])
        raise MetricsError(f"{name} is not strictly increasing at index {k} ({values[k]:.17g} -> {values[k + 1]:.17g})")
    step = (values[-1] - values[0]) / (len(values) - 1)
    off = np.abs(steps - step)
    if not np.all(off <= ROUND_OFF * scale):
        k = int(np.argmax(off))
        raise MetricsError(f"{name} is not uniformly spaced: step {k} is {steps[k]:.17g}, the mean step {step:.17g}")
    return float(step)


def _runs(n_m: int, first_blade_i: int, last_blade_i: int) -> tuple[tuple[str, int, int], ...]:
    """The three m regions as (name, start, stop), each with the three stations a
    second-order one-sided difference needs."""
    if not 0 <= first_blade_i <= last_blade_i < n_m:
        raise MetricsError(f"blade stations {first_blade_i} to {last_blade_i} do not fit {n_m} stations")
    runs = (("upstream periodic", 0, first_blade_i), ("blade", first_blade_i, last_blade_i + 1),
            ("downstream periodic", last_blade_i + 1, n_m))
    cure = {"upstream periodic": "lengthen upstream_extension_chords or raise n_passage",
            "blade": "raise n_passage",
            "downstream periodic": "lengthen downstream_extension_chords or raise n_passage"}
    for name, start, stop in runs:
        if stop - start < 3:
            raise MetricsError(f"the {name} run holds {stop - start} station(s), and a second-order m derivative "
                               f"inside one run needs three: {cure[name]}")
    return runs


def _d_dm(values: Field, dm: float, runs: tuple[tuple[str, int, int], ...]) -> Field:
    """d/dm of station values (first axis along m), each run differentiated on its own by
    _d_dm_run: no stencil leaves its run."""
    out = np.empty_like(values, dtype=float)
    for _, start, stop in runs:
        out[start:stop] = _d_dm_run(values[start:stop], dm)
    return out


def _d_dm_run(f: Field, dm: float) -> Field:
    """d/dm of one contiguous, uniformly spaced run (first axis along m, at least three
    stations), second order: central inside, three-point one-sided at the two ends."""
    d = np.empty_like(f, dtype=float)
    d[1:-1] = (f[2:] - f[:-2]) / (2.0 * dm)
    d[0] = (-3.0 * f[0] + 4.0 * f[1] - f[2]) / (2.0 * dm)
    d[-1] = (3.0 * f[-1] - 4.0 * f[-2] + f[-3]) / (2.0 * dm)
    return d
