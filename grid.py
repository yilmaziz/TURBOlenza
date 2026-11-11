"""
grid.py
=======
The computational (m, eta) grid of one blade passage with its inlet and exit extensions
[Aungier 2003 Secs. 5.2, 5.3, Figs. 5-3, 5-5]: where the nodes are, which region each
station is in, and how the exit extension follows the discharge angle. The geometric
fields of the transformed equations (slopes, S, Jacobian) are in metrics.py.

Coordinates [Aungier 2003 Eqs. 5-2, 5-3]
    eta = (theta - theta0(m)) / (theta1(m) - theta0(m))
    tan(beta_k) = r d theta_k / dm,  k = 0, 1      (wall slopes)
    so eta = 0 on the wall theta0 and eta = 1 on the wall theta1. In the planar cascade
    m is the axial coordinate x and r theta the tangential coordinate y of geometry.py.

Walls [Aungier 2003 Fig. 5-3]
    On the blade stations theta0 is the upper surface of one blade and theta1 the lower
    surface of the next, one pitch up: the passage lies between two blades. With the
    conventions of geometry.py that is the suction surface of the reference blade (eta = 0,
    "lower" below) and the pressure surface of its +pitch neighbor (eta = 1, "upper"), each
    continued around the nose and the tail by the side of the blade that faces the passage
    (section 3). On the extensions theta0 and theta1 are periodic lines with theta1 =
    theta0 + pitch.

Extensions [Aungier 2003 Sec. 5.3, p. 86]
    The far boundary condition (Eq. 5-33) needs the grid lines at the far stations to run
    along the flow angle. So the inlet extension depends on the inlet angle, and the exit
    extension on the discharge angle, which the Kutta loop changes (rebuild_exit).
    Their lengths come from GridSettings in true blade chords:
    settings.upstream_extension_chords * cascade.active.chord, and the same downstream,
    never c cos(stagger). That works unchanged for chord-1 and dimensional cascades.

Angles
    Radians. The solver converts the _deg inputs of config.py once, before calling here.
    grid.py holds the current exit angle but not how it was found: the Kutta loop supplies
    it through rebuild_exit.

Indexing
    i runs along m and j along eta, so node arrays are (n_m, n_eta). dm and deta are
    constant [Aungier 2003 Sec. 5.3, p. 87].

Status: geometric grid implementation, frozen 2026-11-07 for C1 (with the tracked wall
identity of section 3) and REOPENED 2026-11-11 for the C2 hand-off of section 7 only: for a
C1 cascade every array is unchanged bit for bit; for a C2 cascade the blade walls are the
authoritative curves of geometry.py (under review, not frozen). The solver goes with C2
only for now: its grid takes the wall locations, and through metrics.py the wall angles,
from the C2 geometry. A fair scientific comparison of C1 against C2 is planned for later
(an exact piecewise-C1 derivative path first, so that only the surface continuity
differs); until then the C1 path is kept unchanged as the legacy baseline. make_passage_boundaries
builds the side boundaries of one passage; make_grid checks that the smooth passage is open
over the whole blade, then builds the stations, the blade-passage topology, the wall values
and the nodes; rebuild_exit turns the exit extension to a new discharge angle. Coordinates
only: no wall slope, metric or flow quantity is computed here (for C2, Grid hands on the
curves' own u derivatives at the wall nodes, section 7, and metrics.py forms the slopes).
node_index and node_theta are not implemented. Deferred, not resolved: the straight m-links
across the LE and TE transition intervals (section 6) are for metrics.py and
discretization.py to review.

1. Passage boundaries (make_passage_boundaries)
    Input: a Cascade from geometry.make_cascade, read through cascade.active only, and its
    blade angles kappa_le and kappa_te. Nothing of the blade is rebuilt and the CascadeSpec
    is not read.

    Blade walls: lower = active.suction_surface, upper = active.pressure_surface +
    active.periodic_translation, both LE -> TE exactly as geometry.py orders them: no
    sorting by m, no point removed. For a C2 cascade blade_curves also holds
    cascade.active_curves, by reference, not copied: the authoritative curves those arrays
    sample, from which the C2 grid takes its walls (section 7); None for C1. The exact landmarks
        m_le = active.leading_edge_point[0],    m_te = active.trailing_edge_point[0]
    are the chord-axis ends of the blade, not its extent in m: the rounded nose reaches
    ahead of m_le (the suction side when the LE circle center lies at +y of the LE point,
    the pressure side when at -y), a thin nose can fold back a second time a few
    thousandths of a chord behind m_le, and a Kovach TE passes slightly downstream of m_te
    before it closes on the TE point. The boundaries hand the walls on as they are.

    Inlet extension, L_up = settings.upstream_extension_chords * active.chord along m, so
    m_in = m_le - L_up; exit extension, L_down = settings.downstream_extension_chords *
    active.chord, so m_out = m_te + L_down.

    Side-boundary angle [Aungier 2003 Sec. 5.3, p. 86]: beta, not tan(beta), varies
    uniformly in m from the blade angle at the junction to the flow angle at the far
    station,
        inlet:  kappa_le at m_le  ->  inlet_angle at m_in,
        exit:   kappa_te at m_te  ->  exit_angle at m_out.
    kappa_le and kappa_te are the equivalent-arc blade angles that geometry.py gives; they
    are not recomputed from a surface or camberline slope. exit_angle = None means
    exit_angle = kappa_te: a starting value for this solver's Kutta loop, not a rule of
    Aungier's. The exit line is then the straight tangent to kappa_te.

    Lower side line: dy/dm = tan(beta(m)) through the exact junction point (the LE point
    for the inlet, integrated backward; the TE point for the exit, integrated forward),
        y(m) = y_j + (m - m_j) T(beta(m), kappa_j),
    where T(a, b) = ln(cos a / cos b) / (b - a) is the mean of tan over [a, b], the exact
    primitive of the linear-angle law. It is evaluated without dividing by b - a, so equal
    angles give the straight line y = y_j + (m - m_j) tan(kappa_j).

    Upper side line: the lower one plus periodic_translation, never integrated on its own,
    so the two are exactly one pitch apart at every node.

    The upstream_* and downstream_* arrays sample these closed-form lines uniformly from
    the far station to the junction, with as many cells as the length holds passage cells
    of (m_te - m_le) / n_passage, rounded, so both ends are exact. They describe the
    curves; the grid stations are Grid.m, where make_grid evaluates the same formula.

2. Stations (make_grid)
    One constant dm over the whole domain, with exact m_in and m_out:
        dm_target = (m_te - m_le) / n_passage,    n_cells = round((m_out - m_in) / dm_target),
        m = linspace(m_in, m_out, n_cells + 1).
    n_passage is a target resolution: dm differs from dm_target by at most
    dm_target / (2 n_cells), and neither edge is a station in general.

3. Blade-passage topology [Aungier 2003 Sec. 5.3, Fig. 5-5]
    Aungier asks only that the first and last grid points on the blade surfaces lie inside
    the blade passages; he prescribes no distance from the LE or TE, and none is imposed
    here. Every station line x = m[i] is cut with the actual blade row: the closed outline
    of the reference blade (suction surface TE -> LE, then pressure surface LE -> TE) and
    its periodic copies.
    - A station whose line meets no blade is periodic. Its eta line, one pitch along the
      side lines, lies in fluid from end to end, at any n_eta.
    - A station whose line cuts the blade is a blade station. The cut of the reference
      blade is one or more metal intervals on the line; its passage-facing top is the
      lower wall node, and its passage-facing bottom moved one pitch, the bottom of the
      +pitch neighbor's cut, is the upper wall node. The eta line between them touches
      metal at its two ends only: the other copies lie a pitch further out, and the cut
      must span less than a pitch (otherwise GridError).
    The blade's extent in m is one interval, so the blade stations are one contiguous run
    first_blade_i .. last_blade_i, with the inlet periodic stations before it and the exit
    periodic stations after: the first blade station is the first station behind the
    blade's most upstream point and the last blade station the last one ahead of its most
    downstream point, read from the cuts, not from the landmarks. The exact LE and TE need
    not be stations or region boundaries: actual_le_offset = m_first_blade - m_le is
    negative when the first blade station falls in the nose ahead of m_le, and
    actual_te_offset = m_last_blade - m_te positive when it falls in a Kovach tail behind
    m_te. Both offsets stay within one dm of the blade's extent, so they follow dm as the
    grid is refined. The classification uses the stations alone, never the eta nodes.

    A line through the blade's most upstream or most downstream point only touches the
    blade, and a cut thinner than TOUCH = 1e-12 chord counts as touching: such a station is
    periodic. That happens when a station lands on a sharp TE, as it does with round
    numbers (stagger 60 deg, so the blade is 0.5 c long in m, and one-chord extensions);
    with round-off the cut there is 1e-16 c either way, and as a blade station its two wall
    nodes would collapse onto the TE point. A touching periodic line meets the blade at that
    point only.

    Passage-facing walls and their named surfaces. The blade's most upstream point (the
    nose tip) and most downstream point (the tail end) split the closed outline into two
    contours: the upper contour, the top of every cut, and the lower contour, its bottom.
    The lower wall runs on the reference blade's upper contour and the upper wall on the
    +pitch neighbor's lower contour, each from the nose tip to the tail end; these are the
    walls a station line meets from inside the passage, and the only ones whose wall-to-wall
    line is free of metal. geometry.py names the blade's two halves at its chord-axis LE and
    TE points instead, and those lie on the contours, not at their ends: under stagger the
    nose tip lies on one named surface (for the EXAMPLE on the suction surface, 0.0046 c
    ahead of m_le), and a Kovach tail end lies on one too. So the walls keep their usual
    names, lower = suction and upper = pressure, except over the two short end stretches:
    between the nose tip and the LE point one wall runs on the other named surface (for
    positive stagger the upper wall, on the neighbor's nose underside, which is suction
    surface; for negative stagger the lower wall, on the pressure surface's nose), and
    between the TE point and a Kovach tail end the other one does (for positive stagger the
    lower wall, on the pressure surface's tail return; for negative stagger the upper wall,
    on the suction surface's). A station falls in such a stretch whenever dm is shorter than
    it (the EXAMPLE from 172 passage cells up, always), so the name change is a property of
    the blade and of geometry.py's naming, not of a resolution. Grid.wall_identity records,
    for every wall node of the blade stations, its named surface, its place on that surface
    (0 at the LE point, 1 at the TE point) and its place along the wall's contour (the
    outline's chord-length parameter t, counted from the nose tip; the PCHIP runs through t
    within about 1 % of arc length), and Grid refuses a grid in which a wall node is off its contour, a
    wall runs backward along its contour, or a wall's named surface changes anywhere but
    where its contour passes the LE or TE point. Inside a thin nose's second fold the line
    crosses one contour three times; the outermost crossing faces the passage, the two
    behind it are counted in wall_identity as hidden, and the wall steps forward over that
    pocket of the surface, which no station line can reach. Each blade surface is one
    streamline whatever its name, so the wall condition is the same on both stretches.

4. Wall values (C1; a C2 cascade takes its walls from its curves, section 7)
    The wall nodes are crossing points of the station line with a parametric PCHIP x(t),
    y(t) through the outline vertices, t the cumulative chord length: one cubic per polyline
    segment, monotone in x and in y on each segment, C1 through the LE point, and never
    reaching beyond the vertices in m. The polyline segment that the cut selects (the top
    or the bottom crossing) therefore holds exactly one crossing of the smooth curve, found
    by bisection. Against the same blade at sixteen times the camber stations the nodes lie
    within 8.3e-6 c for the 65-(12A10)10 at 33 deg, nose and tail included, and within
    2.6e-5 c over the sweep, largest where the Kovach taper starts (station 0.60, where the
    thickness of geometry.py has a slope break) and where the nose stations end; straight
    lines between the vertices would be off by up to about 4e-5 c. That error belongs to
    the Cascade's resolution (n_camber_stations), not to the grid spacing. The smooth walls
    lie within 3.5e-5 c (normal distance) of the polyline.

    Passage check (make_grid, before any station). The passage must be open wherever both
    walls exist, from the blade's most upstream to its most downstream point, and on the
    walls the grid uses: at every m, not only at the stations. Because the upper wall is the
    bottom of the reference cut moved one pitch, the smooth width there is
        S(m) = pitch - (top(m) - bottom(m)),
    the pitch less the smooth cut's vertical span, and the grid requires S(m) > TOUCH chord
    for every m. The m range is cut at every vertex m of the outline into pieces; inside a
    piece the top and the bottom stay on one polyline segment each, so on one PCHIP piece
    each, monotone in m (above). So on any interval of a piece
        S >= pitch - max(top at its two ends) + min(bottom at its two ends),
    a rigorous bound from two end values. make_grid evaluates top and bottom at both ends of
    every piece (each with that piece's own segments); a piece whose bound clears TOUCH is
    open throughout. Each piece it does not clear (only where the width is within the rise
    of the walls over one piece, i.e. near a waist of the passage) is cut into 16 cells, the
    same bound clears most cells, and every cell left is searched to round-off by zooming:
    17 points, keep the two cells around the smallest width, 16 rounds, each shrinking the
    cell 8 times. That last step assumes no more than one dip of the width inside a cell, a
    sixteenth of a vertex spacing. If the smallest width found is not above TOUCH chord, the
    smooth walls touch or cross and make_grid raises GridError with that width, its m, the
    blade (from cascade.metadata) and the solidity. The answer depends on the blade and the
    pitch only, so it is the same for every n_passage, n_eta and extension length, even
    where the straight-segment outline that geometry.py checks for overlap still leaves a
    gap. TOUCH = 1e-12 chord sits four orders above the round-off of S (about 1e-16 c) and
    far below any open passage seen (2e-6 c, blades at the overlap limit). Likewise the
    straight-segment cut must span less than a pitch; that span is linear in m on a piece,
    so the piece ends settle it exactly.

5. Nodes
    x = m at every node of a station, and the linear map of Aungier's Eq. 5-2,
        y[i, j] = (1 - eta[j]) y_lower[i] + eta[j] y_upper[i],    eta = linspace(0, 1, n_eta),
    which is y_lower + eta (y_upper - y_lower) written so that eta = 0 and eta = 1 return
    the wall values exactly.

6. What the grid guarantees, and what it leaves to the difference equations
    This is a node grid for central differences in m and eta, not a mesh of cells. It
    guarantees: no node lies in blade metal; every station's eta line is valid as a whole
    (a periodic line in fluid, or a wall-to-wall line touching metal at its ends only);
    the wall nodes sit on the passage-facing sides of the right blades, two distinct nodes
    at every blade station, each wall on one contour of its blade, never running backward
    along it and changing its named surface only at geometry.py's LE or TE point (section
    3, Grid.wall_identity); the smooth passage is open, wider than TOUCH chord at every m
    between the blade's ends (the passage check of section 4), so the width is positive at
    every station; the periodic stations are exact translates; the regions are contiguous;
    dm and deta are constant; and none of it depends on n_eta. Whether a cascade builds at
    all depends on the blade and the pitch, not on n_passage or n_eta, except that a grid
    too coarse to put a station on the blade, or extensions shorter than the nose, are
    refused. Grid itself refuses, with GridError, blade-station indices or nodes that break
    the topology, the positive width at the stations, the two distinct wall nodes or the
    wall identity, whoever built them.

    What "metal" means here: the blade inside the smooth walls of section 4, the surface
    the wall nodes lie on; the Cascade's straight-segment polyline is within 3.5e-5 c of it.
    Read against the polyline, the wall-to-wall lines meet it only in slivers of that size at
    their ends. Where blades at the overlap limit of geometry.py all but close the passage
    (widths of a few 1e-6 c), the polyline and the smooth walls leave their gaps at slightly
    different heights, and nodes between the walls can then lie up to 2e-5 c (vertically)
    inside the polyline, never inside the smooth walls. Where the smooth walls touch or
    cross while the polyline still leaves a gap, the passage check refuses the cascade at
    every resolution. Touches of zero length are allowed: a
    station on the blade's most upstream or downstream point, or so close to it that its
    line holds no more than TOUCH of metal (a few 1e-15 c from a round nose's tip, up to some
    1e-11 c from a thin sharp TE), is periodic and its line meets the blade there, so at a
    sharp TE its side node can sit on the TE point; a blade station through the tip of a thin
    nose's second fold meets that tip at one point.

    It does not make the straight segment between nodes (i, j) and (i + 1, j) a grid edge.
    Across the two transition intervals, last inlet station -> first blade station and
    last blade station -> first exit station, the boundary rows change from periodic side
    line to wall (they are neither a wall nor a fluid link there), and the nose or tail
    between the two stations can lie across the straight segment between nodes of the
    rows near the walls: over the sweeps up to 0.18 c of a link at the LE, reaching rows up
    to eta 0.33, and up to 0.04 c at the TE. The four nodes around such an interval are not
    a body-fitted quadrilateral and are not meant as one. These links are NOT
    finite-volume cell edges. Their use by future finite-difference stencils must be
    reviewed when the m-derivative operators and the LE/TE one-sided boundary stencils are
    implemented (metrics.py, discretization.py). The crossings are a deferred limitation,
    reported as a diagnostic, not a resolved or harmless one.

7. C2: the walls are the authoritative curves (surface_continuity "C2", 2026-11-11)
    A C2 cascade carries its surfaces as two ParametricBladeCurves X(u) = (m(u), y(u)),
    0 <= u <= 1 from the LE point to the TE point, in the active unit and the cascade frame
    (cascade.active_curves; its arrays are these curves sampled). For such a cascade the
    blade is that pair of curves everywhere grid.py looks at the blade: the station cuts,
    the topology, the wall nodes, the wall identity and the passage check. No PCHIP is
    built and the arrays are not read for it. Everything else (stations, extensions, the
    linear eta map, rebuild_exit, the rules of sections 2, 3, 5 and 6) is the same code as
    for C1.

    Branches. Each curve is cut at the u where m(u) or y(u) turns, into branches on which
    both are strictly monotone. The turning points are the roots of m'(u) and y'(u), splines
    of degree 4: found exactly per polynomial piece (scipy PPoly.roots), kept where the
    derivative changes sign (a root of even order is no turn), and refined by bisection on
    the curve's own derivative to round-off. So a station line x = m meets a branch at most
    once, and only if m lies in the branch's m range, half-open like a polyline segment
    (min <= m < max): a line through a nose tip counts it twice, a line through a tail end
    or a sharp TE corner not at all, exactly as for C1's vertices. The outline runs suction
    curve TE -> LE (u from 1 to 0), then pressure curve LE -> TE.

    Roots. On a crossed branch m(u) - m_i keeps its sign at one end; 64 bisection steps on
    the branch's u interval, evaluating curve.evaluate itself, take u to round-off, and the
    wall node is X(u_i): y_i = y(u_i) exactly and |m(u_i) - m_i| at the round-off of m. A
    station's crossings are all branch roots, in height order; the topology (section 3:
    metal on the line, TOUCH, one run of blade stations) is read from them exactly as from
    the polyline, so first_blade_i and last_blade_i follow the same rule on the curve. The
    curve reaches slightly past the extreme vertices of its own samples (1.9e-6 c ahead of
    the most upstream sample at the EXAMPLE's nose, 2.7e-7 c behind the last one at its
    tail), so a station inside such a sliver is a blade station of the curve where the
    sampled polyline would call it periodic: the only way the blade run can differ from a
    grid read from the arrays.

    Passage-facing roots. Several roots on one curve are ordinary at a rounded nose (the
    line ahead of the LE point meets the nose twice, both on the surface the nose tip lies
    on) and at a Kovach tail (the line behind the TE point meets the tail return and the
    other side of the same surface). The choice is the rule of section 3, not a surface
    name: the highest crossing is the lower wall, on the reference blade's upper contour,
    and the lowest crossing moved one pitch is the upper wall, on the neighbor's lower
    contour. The contours run from the nose tip (the branch end of least m) to the tail end
    (the branch end of greatest m); a wall node off its contour, a wall running backward
    along it, or a named surface changing away from the LE or TE point is a GridError, as
    for C1. The contour coordinate is tau = (1 - u) L_s on the suction curve and L_s + u L_p
    on the pressure curve (L = reference_length, so tau is the C1 arc length to 1e-5), and
    WallSide.parameter is u itself.

    What the Grid keeps (no new Grid field): boundaries.blade_curves, and for every wall
    node of the blade stations wall_identity.<lower|upper>.surface (which curve: "suction"
    or "pressure", on the reference blade for the lower wall and on its +pitch neighbor for
    the upper wall), .u (where on it) and .contour (where along the passage-facing contour).
    Grid checks that every C2 wall node is that curve point, within TOUCH chord (it is so to
    round-off), whoever built the nodes. Grid.wall_curve_derivative(order) hands on
    d^k X / du^k of the same curve at the same u for k = 0 (the nodes themselves), 1, 2, so
    metrics.py takes the C2 wall slopes from the curve without reading wall names; C1 grids
    have no such curves (has_wall_curves False).

    Passage check (section 4) on the curves: the blade's m range is cut at the m of every
    branch end and every interior knot of either curve into pieces; inside a piece the top
    and the bottom of the cut stay on one branch each, monotone in m and in y, so the same
    bound from the two piece ends holds, and the same zoom finds a narrow waist to
    round-off. The straight-segment span of section 4 becomes the curve's span at the piece
    ends.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields, replace
from functools import cached_property

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import BSpline, PchipInterpolator, PPoly

from potential_flow_solver.config import GridSettings
from potential_flow_solver.geometry import Cascade

INLET, PASSAGE, EXIT = 0, 1, 2      # region of a station: inlet side lines, blade walls, exit side lines
TOUCH = 1e-12                       # chords: a cut thinner than this touches the blade, it does not cut it, and a
                                    # passage no wider than this is closed (section 4, passage check)
_SAMPLES = 16                       # passage check: cells per doubtful piece and per zoom round
_ZOOMS = 16                         # passage check: zoom rounds, each shrinking a cell by _SAMPLES / 2 = 8
_BISECTIONS = 64                    # C2 (section 7): bisection steps of a root in u, past the round-off of u in [0, 1]
_TURN_BRACKET = 1e-9                # C2: a root of m'(u) or y'(u) is a turn if the derivative changes sign across +- this
_TURN_MERGE = 1e-12                 # C2: candidate roots of one derivative closer than this in u are one candidate

Points = NDArray[np.float64]


class GridError(ValueError):
    """The grid cannot be built from this cascade with these settings, or a Grid would break
    the blade-passage topology: the passage is not open over the whole blade (the smooth
    walls touch or cross somewhere, or the straight-segment cut spans a pitch), no station
    cuts the blade, the stations that cut it are not one run, the blade reaches a far
    station, a Grid's nodes leave no positive width at a station or put both wall nodes
    of a blade station on one point of the blade's cut, or a wall leaves its contour, runs
    backward along it or changes its named surface away from the LE and TE points."""


@dataclass(frozen=True)
class PassageBoundaries:
    """The side boundaries of one blade passage, from the far inlet to the far exit. Planar
    cascade, so m is the axial x of geometry.py; point arrays are (n, 2) of (m, y), in the
    active length unit of the Cascade, and read-only.

    "lower" is the eta = 0 side (theta0) and "upper" the eta = 1 side (theta1).

    blade_curves is cascade.active_curves of a C2 cascade, held by reference (geometry.py
    owns it; nothing is copied or refitted), or None for a C1 cascade (module docstring,
    section 7). Its type is geometry.SurfaceCurves; grid.py takes nothing from geometry but
    Cascade, so the name is not imported."""

    chord: float                            # active.chord, the unit of the extension lengths
    pitch: float                            # active.pitch
    periodic_translation: NDArray[np.float64]  # (0, pitch): upper = lower + this on the extensions
    inlet_angle: float                      # far-inlet flow angle, radians: upstream_beta[0]
    exit_angle: float                       # far-exit flow angle, radians: downstream_beta[-1]
    m_in: float                             # far inlet station, m_le - upstream_extension_chords * chord
    m_le: float                             # active.leading_edge_point[0]
    m_te: float                             # active.trailing_edge_point[0]
    m_out: float                            # far exit station, m_te + downstream_extension_chords * chord
    lower_blade_boundary: Points            # active.suction_surface, LE -> TE, unchanged
    upper_blade_boundary: Points            # active.pressure_surface + periodic_translation, LE -> TE
    blade_curves: SurfaceCurves | None      # C2: cascade.active_curves (the curves the two arrays above sample); C1: None
    upstream_m: NDArray[np.float64]         # uniform samples, upstream_m[0] = m_in, upstream_m[-1] = m_le
    upstream_beta: NDArray[np.float64]      # side-boundary angle, radians: inlet_angle -> kappa_le
    upstream_lower_boundary: Points         # (m, y); ends on the exact LE point
    upstream_upper_boundary: Points         # upstream_lower_boundary + periodic_translation
    downstream_m: NDArray[np.float64]       # uniform samples, downstream_m[0] = m_te, downstream_m[-1] = m_out
    downstream_beta: NDArray[np.float64]    # side-boundary angle, radians: kappa_te -> exit_angle
    downstream_lower_boundary: Points       # (m, y); starts on the exact TE point
    downstream_upper_boundary: Points       # downstream_lower_boundary + periodic_translation

    def __post_init__(self) -> None:
        for item in fields(self):
            value = getattr(self, item.name)
            if item.name == "blade_curves":
                if value is not None and not all(hasattr(value, name) for name in ("suction", "pressure")):
                    raise TypeError(f"blade_curves must be the cascade's active_curves or None, got "
                                    f"{type(value).__name__}")
            elif isinstance(value, np.ndarray):
                array = np.array(value, dtype=float)
                array.flags.writeable = False
                object.__setattr__(self, item.name, array)
            else:
                object.__setattr__(self, item.name, float(value))


@dataclass(frozen=True)
class WallSide:
    """One wall's nodes at the blade stations, in station order: the named surface of
    geometry.py each lies on, the place on that surface, and the place along the wall's
    contour (module docstring, sections 3 and 7). Read-only arrays.

    For a C2 cascade the named surface is also the authoritative curve the node lies on
    (boundaries.blade_curves.suction or .pressure: on the reference blade for the lower
    wall, on its +pitch neighbor for the upper wall), and u is the node's parameter on it:
    curve.evaluate(u) is the node (plus periodic_translation for the upper wall). C1 has no
    such curve and u is NaN there."""

    surface: NDArray[np.str_]               # "suction" or "pressure"
    parameter: NDArray[np.float64]          # on that surface: 0 at geometry.py's LE point, 1 at its TE point (C1: chord
                                            # length; C2: u itself)
    u: NDArray[np.float64]                  # C2: parameter of the node on the surface's curve, exact to round-off; C1: NaN
    contour: NDArray[np.float64]            # along the wall's contour from the nose tip: C1 the outline's chord-length
                                            # parameter, C2 tau (section 7); a length either way
    hidden: NDArray[np.int64]               # further crossings of the station line with that contour, behind the node

    def __post_init__(self) -> None:
        for item in fields(self):
            array = np.array(getattr(self, item.name))
            array.flags.writeable = False
            object.__setattr__(self, item.name, array)


@dataclass(frozen=True)
class WallIdentity:
    """Where the wall nodes of the blade stations lie on the blades (Grid.wall_identity).
    lower: on the reference blade's upper contour, which runs from the nose tip over the
    suction surface to the tail end; upper: on the +pitch neighbor's lower contour, from the
    nose tip over the pressure surface to the tail end. Parameters and lengths are those of
    the reference blade, of which the neighbor is a translate. geometry.py's LE and TE points
    lie on these contours, so between the nose tip and the LE point, or the TE point and a
    Kovach tail's end, a wall runs on the other named surface; nowhere else."""

    stations: NDArray[np.int64]             # first_blade_i .. last_blade_i
    lower: WallSide
    upper: WallSide

    def __post_init__(self) -> None:
        array = np.array(self.stations, dtype=np.int64)
        array.flags.writeable = False
        object.__setattr__(self, "stations", array)


@dataclass(frozen=True)
class Grid:
    """The structured grid of one passage: stations m (n_m,), eta (n_eta,) and the nodes
    (n_m, n_eta, 2) of (m, y), in the active length unit and read-only. nodes[i, :, 0] is
    m[i]; nodes[:, 0] is the lower (eta = 0) boundary and nodes[:, -1] the upper one.

    Stations first_blade_i to last_blade_i are the stations whose line cuts the blade, with
    the wall nodes on its passage-facing sides; the ones before lie on the inlet side lines
    and the ones after on the exit side lines, and their lines meet no blade. A Grid whose
    stations break this, or whose nodes close the passage or put both wall nodes of a blade
    station on one point of the blade's cut, is refused with GridError. The landmarks,
    lengths and angles are those of boundaries, read through the properties below."""

    boundaries: PassageBoundaries           # the side-boundary curves the stations sample
    m: NDArray[np.float64]                  # uniform stations, m[0] = m_in, m[-1] = m_out
    eta: NDArray[np.float64]                # uniform, eta[0] = 0, eta[-1] = 1
    nodes: NDArray[np.float64]              # (n_m, n_eta, 2)
    first_blade_i: int                      # first station whose line cuts the blade
    last_blade_i: int                       # last station whose line cuts the blade

    def __post_init__(self) -> None:
        for name in ("m", "eta", "nodes"):
            array = np.array(getattr(self, name), dtype=float)
            array.flags.writeable = False
            object.__setattr__(self, name, array)
        for name in ("first_blade_i", "last_blade_i"):
            object.__setattr__(self, name, int(getattr(self, name)))
        row = _row_of(self.boundaries)
        cut = row.cut(self.m)
        cutting = cut.cuts_blade
        expected = np.zeros(len(self.m), dtype=bool)
        expected[self.first_blade_i:self.last_blade_i + 1] = True
        if not (1 <= self.first_blade_i <= self.last_blade_i < len(self.m) - 1 and np.array_equal(cutting, expected)):
            raise GridError(f"stations {self.first_blade_i} to {self.last_blade_i} as blade stations break the "
                            f"blade-passage topology: exactly the stations whose line cuts the blade must be blade "
                            f"stations, with periodic stations on both sides")
        lower, upper = self.nodes[:, 0, 1], self.nodes[:, -1, 1]
        shut = np.flatnonzero(~(upper - lower > 0.0))
        if shut.size:
            raise GridError(f"the passage width is not positive at {shut.size} station(s), first at m = "
                            f"{self.m[shut[0]]:.6g}")
        walls = slice(self.first_blade_i, self.last_blade_i + 1)
        merged = np.flatnonzero(~(lower[walls] - (upper[walls] - self.boundaries.periodic_translation[1]) > 0.0))
        if merged.size:
            i = self.first_blade_i + int(merged[0])
            raise GridError(f"the wall nodes of blade station {i} (m = {self.m[i]:.6g}) are not two distinct points of "
                            f"the blade's cut: the lower node, its top, must lie above the upper node moved down one "
                            f"pitch, its bottom")
        object.__setattr__(self, "_wall_identity", row.identity(cut, walls, self.first_blade_i))
        if self.has_wall_curves:
            self._check_on_curves(walls, row.touch)

    def _check_on_curves(self, walls: slice, tolerance: float) -> None:
        """C2: GridError unless every wall node of the blade stations is the point of its
        authoritative curve at its u (module docstring, section 7), within TOUCH chord."""
        lower, upper = self.wall_curve_derivative(0)
        m = self.m[walls]
        for name, curve_points, nodes in (("lower", lower, self.nodes[walls, 0]), ("upper", upper, self.nodes[walls, -1])):
            off = np.maximum(np.abs(curve_points[:, 0] - m), np.abs(curve_points[:, 1] - nodes[:, 1]))
            bad = np.flatnonzero(~(off <= tolerance))
            if bad.size:
                i = self.first_blade_i + int(bad[0])
                raise GridError(f"the {name} wall node of blade station {i} (m = {self.m[i]:.6g}) is {off[bad[0]]:.3g} "
                                f"off its authoritative curve point: a C2 wall node must be X(u) of the curve and u that "
                                f"wall_identity records")

    @property
    def wall_identity(self) -> WallIdentity:
        """Which named surface of geometry.py each wall node of the blade stations lies on,
        where on it, and where along its wall's contour; checked when the Grid is made. For
        a C2 cascade also the node's parameter u on that surface's authoritative curve."""
        return self._wall_identity

    @property
    def has_wall_curves(self) -> bool:
        """True for a C2 cascade, whose blade walls are authoritative parametric curves
        (boundaries.blade_curves): wall_curve_derivative is then available."""
        return self.boundaries.blade_curves is not None

    def wall_curve_derivative(self, order: int) -> tuple[Points, Points]:
        """(lower, upper): d^order X / du^order of the authoritative curve at each wall node
        of the blade stations, first_blade_i .. last_blade_i, each (n_blade, 2) of (m, y)
        components and read-only (module docstring, section 7). The curve and u of every node
        are those of wall_identity, so order 0 returns the wall nodes themselves (the upper
        wall on the +pitch neighbor), and orders 1 and 2 are the same curve's exact
        derivatives at the same points, in the active length unit per unit u. A C2 cascade
        only: GridError for a grid whose walls are sampled surfaces (C1)."""
        curves = self.boundaries.blade_curves
        if curves is None:
            raise GridError("this grid's blade walls are sampled surfaces (a C1 cascade): there is no authoritative "
                            "curve to evaluate; has_wall_curves is False")
        out = []
        for side, shift in ((self._wall_identity.lower, None), (self._wall_identity.upper, self.boundaries.periodic_translation)):
            value = np.empty((len(side.u), 2))
            for name in ("suction", "pressure"):
                on = side.surface == name
                if np.any(on):
                    value[on] = getattr(curves, name).derivative(side.u[on], order)
            if order == 0 and shift is not None:
                value = value + shift
            value.flags.writeable = False
            out.append(value)
        return out[0], out[1]

    @property
    def lower_boundary(self) -> Points:
        """(m, y) of the eta = 0 nodes, (n_m, 2)."""
        return self.nodes[:, 0]

    @property
    def upper_boundary(self) -> Points:
        """(m, y) of the eta = 1 nodes, (n_m, 2)."""
        return self.nodes[:, -1]

    @property
    def dm(self) -> float:
        return float((self.m[-1] - self.m[0]) / (len(self.m) - 1))

    @property
    def deta(self) -> float:
        return float((self.eta[-1] - self.eta[0]) / (len(self.eta) - 1))

    @property
    def last_inlet_i(self) -> int:
        """Last inlet (periodic) station: first_blade_i - 1."""
        return self.first_blade_i - 1

    @property
    def first_exit_i(self) -> int:
        """First exit (periodic) station: last_blade_i + 1."""
        return self.last_blade_i + 1

    @property
    def m_first_blade(self) -> float:
        """First blade station, the first one whose line cuts the blade."""
        return float(self.m[self.first_blade_i])

    @property
    def m_last_blade(self) -> float:
        """Last blade station, the last one whose line cuts the blade."""
        return float(self.m[self.last_blade_i])

    @property
    def actual_le_offset(self) -> float:
        """m_first_blade - m_le: a diagnostic, negative when the first blade station lies in
        the nose ahead of the chord-axis LE."""
        return self.m_first_blade - self.m_le

    @property
    def actual_te_offset(self) -> float:
        """m_last_blade - m_te: a diagnostic, positive when the last blade station lies in a
        Kovach tail behind the chord-axis TE."""
        return self.m_last_blade - self.m_te

    @property
    def region(self) -> NDArray[np.int64]:
        """INLET, PASSAGE or EXIT at each station."""
        region = np.full(len(self.m), PASSAGE, dtype=np.int64)
        region[:self.first_blade_i] = INLET
        region[self.last_blade_i + 1:] = EXIT
        region.flags.writeable = False
        return region

    @property
    def m_in(self) -> float:
        return self.boundaries.m_in

    @property
    def m_out(self) -> float:
        return self.boundaries.m_out

    @property
    def m_le(self) -> float:
        """The exact geometric LE, active.leading_edge_point[0]; a landmark, not a station or
        region boundary in general."""
        return self.boundaries.m_le

    @property
    def m_te(self) -> float:
        """The exact geometric TE, active.trailing_edge_point[0]; a landmark, not a station or
        region boundary in general."""
        return self.boundaries.m_te

    @property
    def chord(self) -> float:
        return self.boundaries.chord

    @property
    def pitch(self) -> float:
        return self.boundaries.pitch

    @property
    def inlet_angle(self) -> float:
        return self.boundaries.inlet_angle

    @property
    def exit_angle(self) -> float:
        return self.boundaries.exit_angle


def make_passage_boundaries(cascade: Cascade, settings: GridSettings, inlet_angle: float,
                            exit_angle: float | None = None) -> PassageBoundaries:
    """The blade walls of one passage and its inlet and exit extensions.

    cascade is read through cascade.active, cascade.active_curves (handed on by reference as
    blade_curves: the C2 curves, or None for C1) and its blade angles only. settings gives the
    extension lengths (upstream_ and downstream_extension_chords) and the passage
    resolution (n_passage) that sets their sample spacing. inlet_angle and exit_angle are
    the far-inlet and far-exit flow angles in radians from the meridional (axial)
    direction, positive toward +y; a caller holding degrees converts them once with
    math.radians. exit_angle None takes cascade.kappa_te, the starting value of the Kutta
    loop.

    Raises TypeError for a wrong argument type and ValueError for an angle outside
    (-pi/2, pi/2), a non-positive extension length or passage resolution, or a blade whose
    TE is not downstream of its LE.
    """
    _check_inputs(cascade, settings, "make_passage_boundaries")
    _check_angle(inlet_angle, "inlet_angle")
    if exit_angle is None:
        exit_angle = cascade.kappa_te
    else:
        _check_angle(exit_angle, "exit_angle")
    _check_blade_angle(cascade.kappa_le, "kappa_le", "inlet")
    _check_blade_angle(cascade.kappa_te, "kappa_te", "exit")
    upstream = _positive_number(settings.upstream_extension_chords, "upstream_extension_chords")
    downstream = _positive_number(settings.downstream_extension_chords, "downstream_extension_chords")
    n_passage = _whole_number(settings.n_passage, "n_passage", 1)

    a = cascade.active
    leading_edge, trailing_edge = a.leading_edge_point, a.trailing_edge_point
    m_le, m_te = float(leading_edge[0]), float(trailing_edge[0])
    if not m_te > m_le:
        raise ValueError(f"the TE (m = {m_te}) must lie downstream of the LE (m = {m_le})")
    m_in, m_out = m_le - upstream * a.chord, m_te + downstream * a.chord

    upstream_m = _samples(m_in, m_le, n_passage, m_te - m_le)
    upstream_beta, upstream_y = _side_line(upstream_m, _sample_fraction(len(upstream_m), junction_last=True),
                                           leading_edge, cascade.kappa_le, inlet_angle)
    downstream_m = _samples(m_te, m_out, n_passage, m_te - m_le)
    downstream_beta, downstream_y = _side_line(downstream_m, _sample_fraction(len(downstream_m), junction_last=False),
                                               trailing_edge, cascade.kappa_te, exit_angle)
    upstream_lower = np.column_stack((upstream_m, upstream_y))
    downstream_lower = np.column_stack((downstream_m, downstream_y))

    return PassageBoundaries(
        chord=a.chord,
        pitch=a.pitch,
        periodic_translation=a.periodic_translation,
        inlet_angle=inlet_angle,
        exit_angle=exit_angle,
        m_in=m_in,
        m_le=m_le,
        m_te=m_te,
        m_out=m_out,
        lower_blade_boundary=a.suction_surface,
        upper_blade_boundary=a.pressure_surface + a.periodic_translation,
        blade_curves=cascade.active_curves,
        upstream_m=upstream_m,
        upstream_beta=upstream_beta,
        upstream_lower_boundary=upstream_lower,
        upstream_upper_boundary=upstream_lower + a.periodic_translation,
        downstream_m=downstream_m,
        downstream_beta=downstream_beta,
        downstream_lower_boundary=downstream_lower,
        downstream_upper_boundary=downstream_lower + a.periodic_translation,
    )


def make_grid(cascade: Cascade, settings: GridSettings, inlet_angle: float,
              exit_angle: float | None = None) -> Grid:
    """The structured grid of one passage of cascade: make_passage_boundaries, then the
    uniform stations, the blade-passage topology from the cuts of the station lines with
    the blade row, the wall values at the blade stations and the nodes (module docstring,
    sections 2 to 5). For a C2 cascade the blade is its authoritative curves throughout
    (section 7): the walls are the curves' points, with their u in Grid.wall_identity.
    Angles as in make_passage_boundaries, in radians; exit_angle None takes
    cascade.kappa_te.

    Raises what make_passage_boundaries raises, ValueError for n_eta below 2, and
    GridError when the passage is not open over the whole blade (the passage check of
    section 4: the straight-segment cut spans a pitch or more somewhere, or the smooth walls
    leave no width above TOUCH chords somewhere; checked before any station, so the same
    for every n_passage and n_eta), when no station cuts the blade (n_passage too small for
    the blade to be seen), the stations that cut it are not one run, or the blade reaches a
    far station. Only to name the blade in the passage-check message, make_grid reads
    cascade.metadata and cascade.stagger.
    """
    _check_inputs(cascade, settings, "make_grid")
    b = make_passage_boundaries(cascade, settings, inlet_angle, exit_angle)
    n_eta = _whole_number(settings.n_eta, "n_eta", 2)
    row = _row_of(b)
    _check_passage(cascade, b, row)

    dm_target = (b.m_te - b.m_le) / settings.n_passage
    m = np.linspace(b.m_in, b.m_out, max(1, round((b.m_out - b.m_in) / dm_target)) + 1)
    cut = row.cut(m)
    blade = np.flatnonzero(cut.cuts_blade)
    if blade.size == 0:
        raise GridError(f"no station cuts the blade: dm = {(m[1] - m[0]) / b.chord:.4g} chords lets the blade fall "
                        f"between two stations; raise n_passage")
    first, last = int(blade[0]), int(blade[-1])
    if blade.size != last - first + 1:
        raise GridError(f"the stations that cut the blade, {first} to {last}, are not one run")
    if first == 0 or last == len(m) - 1:
        raise GridError("the blade reaches a far station: lengthen the extensions")

    walls = slice(first, last + 1)
    top, bottom = row.passage_sides(cut, walls)
    inlet, exit_ = slice(0, first), slice(last + 1, None)
    lower = np.empty_like(m)
    lower[inlet] = _inlet_side(b, m[inlet])
    lower[walls] = top
    lower[exit_] = _exit_side(b, m[exit_])
    upper = lower + b.periodic_translation[1]
    upper[walls] = bottom + b.periodic_translation[1]

    eta = np.linspace(0.0, 1.0, n_eta)
    nodes = np.empty((len(m), n_eta, 2))
    nodes[..., 0] = m[:, None]
    nodes[..., 1] = _across(eta, lower, upper)
    return Grid(boundaries=b, m=m, eta=eta, nodes=nodes, first_blade_i=first, last_blade_i=last)


def rebuild_exit(item: Grid | PassageBoundaries, exit_angle: float) -> Grid | PassageBoundaries:
    """The same Grid or PassageBoundaries with the exit extension turned to a new discharge
    angle exit_angle, in radians. Only the exit side changes: exit_angle, downstream_beta
    and the downstream lines, and for a Grid the nodes of the exit stations. The stations,
    the eta nodes, the topology (it depends on the blade and the stations only), the inlet
    extension, the blade walls, the landmarks, chord and pitch stay exactly as they are,
    and the result equals a fresh build at that exit angle. How the angle was found (the
    Kutta loop) is not grid.py's business."""
    _check_angle(exit_angle, "exit_angle")
    if isinstance(item, PassageBoundaries):
        beta, y = _side_line(item.downstream_m, _sample_fraction(len(item.downstream_m), junction_last=False),
                             item.downstream_lower_boundary[0], item.downstream_beta[0], exit_angle)
        lower = np.column_stack((item.downstream_m, y))
        return replace(item, exit_angle=exit_angle, downstream_beta=beta, downstream_lower_boundary=lower,
                       downstream_upper_boundary=lower + item.periodic_translation)
    if isinstance(item, Grid):
        boundaries = rebuild_exit(item.boundaries, exit_angle)
        exit_ = slice(item.last_blade_i + 1, None)
        lower = _exit_side(boundaries, item.m[exit_])
        nodes = np.array(item.nodes)
        nodes[exit_, :, 1] = _across(item.eta, lower, lower + boundaries.periodic_translation[1])
        return replace(item, boundaries=boundaries, nodes=nodes)
    raise TypeError(f"rebuild_exit takes a Grid or PassageBoundaries, got {type(item).__name__}")


def _check_passage(cascade: Cascade, b: PassageBoundaries, row: _Row) -> None:
    """GridError unless the passage is open over the whole blade, wherever both walls exist:
    the straight-segment cut spans less than a pitch at every m, and the smooth width is
    above TOUCH chords at every m (module docstring, section 4). It looks at the blade and
    the pitch only, never at the stations, so its answer is the same for every n_passage
    and n_eta."""
    pitch = b.periodic_translation[1]
    metadata = cascade.metadata
    blade = (f"{metadata.get('designation', 'the blade')} (trailing edge: {metadata.get('trailing_edge', 'not given')}) "
             f"at stagger {np.degrees(cascade.stagger):.6g} deg and solidity {b.chord / pitch:.10g}")
    span, where = row.widest_cut()
    if not span < pitch:
        raise GridError(f"the {row.cut_name} of {blade} spans {span / b.chord:.6g} chords at m = {where:.6g}, "
                        f"at least the pitch: the blades overlap there and the passage closes")
    width, where = row.narrowest(pitch)
    if not width > row.touch:
        raise GridError(f"the smooth walls close the passage of {blade}: their narrowest width over the blade is "
                        f"{width / b.chord:.6e} chords, at m = {where:.6g} (m - m_le = {(where - b.m_le) / b.chord:.6g} "
                        f"chords), not above TOUCH = {TOUCH:g} chords. The check covers the whole blade, not the "
                        f"stations, so no n_passage or n_eta builds this cascade; the straight-segment outlines that "
                        f"geometry.py checks for overlap can still leave a gap")


def node_index(i: int, j: int, n_eta: int) -> int:
    """Number of node (i, j) in the unknown vector of the linear system. Planned:
    i * n_eta + j, the C order of an (n_m, n_eta) array, so that psi.ravel() is the
    unknown vector."""
    raise NotImplementedError


def node_theta(grid: Grid) -> NDArray[np.float64]:
    """theta of every node, (n_m, n_eta): Eq. 5-2 solved for theta."""
    raise NotImplementedError


# --- Blade row: cuts of the station lines, passage-facing wall points -----------------

def _row_of(b: PassageBoundaries) -> _Row:
    """The reference blade as grid.py reads it: the sampled outline and its PCHIP for a C1
    cascade, the authoritative curves for a C2 cascade (module docstring, section 7)."""
    return _BladeRow(b) if b.blade_curves is None else _CurveRow(b)


@dataclass(frozen=True)
class _Cut:
    """The cut of the reference blade's outline by the station lines x = m: per station,
    the polyline segments crossed (half-open in x, so a line through a vertex counts it
    once where the outline passes and twice where it turns), sorted by crossing height,
    the heights, and the metal on the line. For a C2 blade (_CurveRow) the pieces are the
    curves' monotone branches instead of segments, the heights are exact curve points, and
    u holds each crossing's curve parameter."""

    m: NDArray[np.float64]                  # the stations
    count: NDArray[np.int64]                # crossings per station
    segment: NDArray[np.int64]              # (n_stations, n_segments) segment index, by ascending crossing height
    height: NDArray[np.float64]             # (n_stations, n_segments) crossing height, ascending, inf past count
    metal: NDArray[np.float64]              # length of the reference blade's cut on the line
    touch: float                            # TOUCH chords in the active length unit
    u: NDArray[np.float64] | None = None    # C2: (n_stations, n_branches) curve parameter of each crossing, NaN past count

    @property
    def cuts_blade(self) -> NDArray[np.bool_]:
        """Lines that cut the blade with more than round-off: a line through the blade's most
        upstream or most downstream point (a sharp TE, say) only touches it."""
        return self.metal > self.touch

    @property
    def top(self) -> NDArray[np.float64]:
        """Highest crossing on the polyline (meaningful where the line cuts the blade)."""
        return np.take_along_axis(self.height, np.maximum(self.count - 1, 0)[:, None], axis=1)[:, 0]

    @property
    def bottom(self) -> NDArray[np.float64]:
        """Lowest crossing on the polyline."""
        return self.height[:, 0]


class _Row:
    """What the two readings of the reference blade share (_BladeRow for C1, _CurveRow for
    C2): the passage check's narrowest width, from the subclass's pieces, smooth and touch."""

    touch: float
    cut_name: str                       # what widest_cut measures, for the passage check's message

    def narrowest(self, pitch: float) -> tuple[float, float]:
        """The smallest smooth passage width pitch - (top - bottom) that the passage check of
        module docstring section 4 finds over the blade, and its m. Every piece is either
        cleared by the bound (its width exceeds TOUCH throughout) or searched to round-off,
        so the width is above TOUCH exactly when the passage is open everywhere, and when it
        is not, it is the narrowest width of the passage. When every piece is cleared at once
        (a passage much wider than the walls rise over one piece) it is only the smallest
        width at the piece ends, an upper estimate."""
        lo, hi, top, bottom = self.pieces

        def walls(piece: NDArray[np.int64], m: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
            """Smooth top and bottom of the cut at the points m (n, k) of the pieces (n, 1)."""
            piece, flat = np.broadcast_to(piece, m.shape).ravel(), m.ravel()
            return (self.smooth(top[piece], flat).reshape(m.shape), self.smooth(bottom[piece], flat).reshape(m.shape))

        def doubtful(upper: NDArray[np.float64], lower: NDArray[np.float64]) -> NDArray[np.bool_]:
            """Cells between neighboring points along the last axis whose bound does not clear
            TOUCH: top and bottom are monotone in m on a piece, so on a cell the width is at
            least pitch - (largest top) + (smallest bottom) of its two ends."""
            bound = pitch - np.maximum(upper[..., :-1], upper[..., 1:]) + np.minimum(lower[..., :-1], lower[..., 1:])
            return ~(bound > self.touch)

        found = []                                                # (width, m) of every point evaluated

        def width_at(piece: NDArray[np.int64], m: NDArray[np.float64]) -> tuple[NDArray[np.float64], ...]:
            upper, lower = walls(piece, m)
            width = pitch - (upper - lower)
            k = int(np.argsort(width, axis=None, kind="stable")[0])
            found.append((float(width.flat[k]), float(m.flat[k])))
            return width, upper, lower

        piece = np.arange(len(lo))
        _, upper, lower = width_at(piece[:, None], np.column_stack((lo, hi)))     # every vertex m, both sides
        piece = piece[doubtful(upper, lower)[:, 0]]
        if piece.size:
            fraction = np.linspace(0.0, 1.0, _SAMPLES + 1)
            m = lo[piece, None] + fraction * (hi - lo)[piece, None]
            _, upper, lower = width_at(piece[:, None], m)
            rows, cells = np.nonzero(doubtful(upper, lower))
            piece, a, b = piece[rows], m[rows, cells], m[rows, cells + 1]
            for _ in range(_ZOOMS if piece.size else 0):         # each doubtful cell to round-off
                m = a[:, None] + fraction * (b - a)[:, None]
                width, _, _ = width_at(piece[:, None], m)
                k = np.argsort(width, axis=1, kind="stable")[:, 0]
                rows = np.arange(len(k))
                a, b = m[rows, np.maximum(k - 1, 0)], m[rows, np.minimum(k + 1, _SAMPLES)]
        return found[int(np.argsort([width for width, _ in found], kind="stable")[0])]


def _check_walls(lower: WallSide, upper: WallSide, splits: dict[str, list[float]], tolerance: float,
                 first_blade_i: int) -> None:
    """GridError if a wall runs backward along its contour, or a wall's named surface
    changes between two stations whose contour stretch holds none of its splits (the
    contour coordinates of geometry.py's LE and TE points on that wall's contour)."""
    for name, wall in (("lower", lower), ("upper", upper)):
        back = np.flatnonzero(~(np.diff(wall.contour) >= -tolerance))
        if back.size:
            raise GridError(f"the {name} wall runs backward along its contour between blade stations "
                            f"{first_blade_i + back[0]} and {first_blade_i + back[0] + 1}")
        for k in np.flatnonzero(wall.surface[1:] != wall.surface[:-1]):
            if not any(wall.contour[k] - tolerance <= split <= wall.contour[k + 1] + tolerance
                       for split in splits[name]):
                raise GridError(f"the {name} wall changes from the {wall.surface[k]} to the {wall.surface[k + 1]} "
                                f"surface between blade stations {first_blade_i + k} and {first_blade_i + k + 1} "
                                f"without passing geometry.py's LE or TE point")


class _BladeRow(_Row):
    """The reference blade's closed outline, read from the passage walls (suction surface
    TE -> LE, then pressure surface LE -> TE), its cuts by station lines, and the
    passage-facing wall points on a parametric PCHIP through its vertices (module
    docstring, sections 3 and 4). Every other blade of the row is this outline moved by
    whole pitches, so the reference cut describes the whole row on a station line."""

    cut_name = "straight-segment cut"

    def __init__(self, b: PassageBoundaries) -> None:
        pressure = b.upper_blade_boundary - b.periodic_translation
        self.outline = np.vstack((b.lower_blade_boundary[::-1], pressure[1:]))
        self.touch = TOUCH * b.chord
        step = np.hypot(*np.diff(self.outline, axis=0).T)
        self.t = np.concatenate(([0.0], np.cumsum(step)))
        self.x_of_t = PchipInterpolator(self.t, self.outline[:, 0])
        self.y_of_t = PchipInterpolator(self.t, self.outline[:, 1])
        self.n_suction = len(b.lower_blade_boundary)              # vertex n_suction - 1 is geometry.py's LE point
        order = np.argsort(self.outline[:-1, 0], kind="stable")    # the last vertex closes the loop on vertex 0
        self.tip, self.tail = int(order[0]), int(order[-1])        # most upstream, most downstream vertex

    def cut(self, m: NDArray[np.float64]) -> _Cut:
        x0, y0 = self.outline[:-1, 0], self.outline[:-1, 1]
        x1, y1 = self.outline[1:, 0], self.outline[1:, 1]
        station = np.asarray(m, dtype=float)
        crossed = (np.minimum(x0, x1) <= station[:, None]) & (station[:, None] < np.maximum(x0, x1))
        fraction = (station[:, None] - x0) / np.where(x1 != x0, x1 - x0, 1.0)
        height = np.where(crossed, y0 + fraction * (y1 - y0), np.inf)
        order = np.argsort(height, axis=1, kind="stable")
        height = np.take_along_axis(height, order, axis=1)
        count = np.count_nonzero(crossed, axis=1)
        if np.any(count % 2):
            raise GridError("a station line crosses the blade outline an odd number of times: the outline is not closed")
        finite = np.where(np.isfinite(height), height, 0.0)
        exits = finite[:, 1::2]
        pairs = exits - finite[:, 0::2][:, :exits.shape[1]]          # metal intervals, bottom to top
        inside = np.arange(pairs.shape[1])[None, :] < (count // 2)[:, None]
        metal = np.where(inside, pairs, 0.0).sum(axis=1)
        return _Cut(m=station, count=count, segment=order, height=height, metal=metal, touch=self.touch)

    def passage_sides(self, cut: _Cut, stations: slice) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """y of the passage-facing top and bottom of the cut at the given blade stations, on
        the smooth outline: the top is the lower wall, the bottom moved one pitch the upper."""
        count, segments = cut.count[stations], cut.segment[stations]
        top = np.take_along_axis(segments, (count - 1)[:, None], axis=1)[:, 0]
        bottom = segments[:, 0]
        m = cut.m[stations]
        return self.smooth(top, m), self.smooth(bottom, m)

    def smooth(self, segment: NDArray[np.int64], m: NDArray[np.float64]) -> NDArray[np.float64]:
        """y where the smooth outline crosses x = m on the given polyline segments."""
        return _smooth_crossing(self.outline, self.t, self.x_of_t, self.y_of_t, segment, m)

    def identity(self, cut: _Cut, stations: slice, first_blade_i: int) -> WallIdentity:
        """Which named surface, where on it and where along its wall's contour the two wall
        nodes of the blade stations lie (module docstring, section 3), checked: GridError if
        a wall node is off its contour, a wall runs backward along it, or a wall's named
        surface changes anywhere but at geometry.py's LE or TE point."""
        count, segment, m = cut.count[stations], cut.segment[stations], cut.m[stations]
        top = np.take_along_axis(segment, (count - 1)[:, None], axis=1)[:, 0]
        bottom = segment[:, 0]
        crossing = np.arange(segment.shape[1])[None, :] < count[:, None]
        n_segments, tip, tail = len(self.outline) - 1, self.tip, self.tail
        lower_arc = (np.arange(n_segments) - tip) % n_segments < (tail - tip) % n_segments   # tip -> tail, increasing t
        if np.any(lower_arc[top]) or not np.all(lower_arc[bottom]):
            raise GridError("a wall node is not on its contour: the top of a cut must lie on the blade's upper "
                            "contour and the bottom on its lower contour")
        total, t_tip, t_le = self.t[-1], self.t[tip], self.t[self.n_suction - 1]
        t_top, t_bottom = self.smooth_parameter(top, m), self.smooth_parameter(bottom, m)

        def side(segments: NDArray[np.int64], t: NDArray[np.float64], contour: NDArray[np.float64],
                 hidden: NDArray[np.int64]) -> WallSide:
            suction = self.on_suction(segments)
            return WallSide(surface=np.where(suction, "suction", "pressure"),
                            parameter=np.where(suction, (t_le - t) / t_le, (t - t_le) / (total - t_le)),
                            u=np.full(len(t), np.nan), contour=contour, hidden=hidden)

        lower = side(top, t_top, np.mod(t_tip - t_top, total), np.count_nonzero(~lower_arc[segment] & crossing, axis=1) - 1)
        upper = side(bottom, t_bottom, np.mod(t_bottom - t_tip, total),
                     np.count_nonzero(lower_arc[segment] & crossing, axis=1) - 1)
        # geometry.py's LE point (vertex n_suction - 1) and TE point (vertex 0, t = 0 = total) on each contour
        on_lower = [(v - tip) % n_segments <= (tail - tip) % n_segments for v in (self.n_suction - 1, 0)]
        splits = {"lower": [np.mod(t_tip - t, total) for t, low in zip((t_le, 0.0), on_lower) if not low],
                  "upper": [np.mod(t - t_tip, total) for t, low in zip((t_le, 0.0), on_lower) if low]}
        _check_walls(lower, upper, splits, self.touch, first_blade_i)
        return WallIdentity(stations=np.arange(first_blade_i, first_blade_i + len(m)), lower=lower, upper=upper)

    def smooth_parameter(self, segment: NDArray[np.int64], m: NDArray[np.float64]) -> NDArray[np.float64]:
        """t where the smooth outline crosses x = m on the given polyline segments."""
        return _smooth_parameter(self.outline, self.t, self.x_of_t, segment, m)

    def on_suction(self, segment: NDArray[np.int64]) -> NDArray[np.bool_]:
        """Whether outline segments belong to geometry.py's suction surface (they come first,
        TE -> LE point); the rest belong to its pressure surface."""
        return segment < self.n_suction - 1

    @cached_property
    def pieces(self) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]:
        """The blade's m range cut at every vertex m into pieces lo..hi, with the polyline
        segments of the top and the bottom of the cut inside each piece: no vertex lies inside
        a piece, so the segments a line meets, and their order, stay the same across it."""
        vertices = np.unique(self.outline[:, 0])
        lo, hi = vertices[:-1], vertices[1:]
        top, bottom = [], []
        for middle in np.array_split(0.5 * (lo + hi), max(1, len(lo) // 256)):    # bounded memory
            cut = self.cut(middle)
            top.append(np.take_along_axis(cut.segment, (cut.count - 1)[:, None], axis=1)[:, 0])
            bottom.append(cut.segment[:, 0])
        return lo, hi, np.concatenate(top), np.concatenate(bottom)

    def widest_cut(self) -> tuple[float, float]:
        """The largest span top - bottom of the straight-segment cut over the whole blade, and
        its m. The span is linear in m on a piece, so it is largest at a piece end."""
        lo, hi, top, bottom = self.pieces
        ends, top, bottom = np.concatenate((lo, hi)), np.concatenate((top, top)), np.concatenate((bottom, bottom))
        span = self.linear(top, ends) - self.linear(bottom, ends)
        k = int(np.argsort(span, kind="stable")[-1])
        return float(span[k]), float(ends[k])

    def linear(self, segment: NDArray[np.int64], m: NDArray[np.float64]) -> NDArray[np.float64]:
        """y where the polyline segments cross x = m."""
        (x0, y0), (x1, y1) = self.outline[segment].T, self.outline[segment + 1].T
        return y0 + (m - x0) / (x1 - x0) * (y1 - y0)


class _CurveRow(_Row):
    """The reference blade of a C2 cascade as its two authoritative curves (module
    docstring, section 7): the closed outline suction curve TE -> LE, then pressure curve
    LE -> TE, cut into branches monotone in m and y; its cuts by station lines, with every
    crossing an exact curve point; the passage-facing wall points; and the pieces of the
    passage check. Every other blade of the row is this one moved by whole pitches."""

    cut_name = "cut of the curves (at a passage-check piece end)"

    def __init__(self, b: PassageBoundaries) -> None:
        self.curves = (b.blade_curves.suction, b.blade_curves.pressure)
        self.touch = TOUCH * b.chord
        self.lengths = tuple(curve.reference_length for curve in self.curves)
        self.total = self.lengths[0] + self.lengths[1]
        curve_of, u0, u1 = [], [], []
        for index, curve in enumerate(self.curves):
            splits = np.unique(np.concatenate(([0.0], _turning_points(curve, 0), _turning_points(curve, 1), [1.0])))
            curve_of += [index] * (len(splits) - 1)
            u0.append(splits[:-1])
            u1.append(splits[1:])
        self.curve_of = np.array(curve_of, dtype=np.int64)          # 0: suction, 1: pressure
        self.u0, self.u1 = np.concatenate(u0), np.concatenate(u1)  # branch ends, u0 < u1 on its curve
        branches = np.arange(len(self.curve_of))
        self.m0, self.m1 = self.point(branches, self.u0)[:, 0], self.point(branches, self.u1)[:, 0]
        self.lo_m, self.hi_m = np.minimum(self.m0, self.m1), np.maximum(self.m0, self.m1)
        ends_m = np.concatenate((self.m0, self.m1))
        ends_tau = np.concatenate((self.tau(branches, self.u0), self.tau(branches, self.u1)))
        order = np.argsort(ends_m, kind="stable")
        self.tau_tip, self.tau_tail = float(ends_tau[order[0]]), float(ends_tau[order[-1]])   # nose tip, tail end
        middle = self.tau(branches, 0.5 * (self.u0 + self.u1))
        self.lower_contour = np.mod(middle - self.tau_tip, self.total) < np.mod(self.tau_tail - self.tau_tip, self.total)

    def tau(self, branch: NDArray[np.int64], u: NDArray[np.float64]) -> NDArray[np.float64]:
        """Contour coordinate of curve parameter u on the given branches: (1 - u) L_s on the
        suction curve (TE at 0, LE at L_s), L_s + u L_p on the pressure curve."""
        return np.where(self.curve_of[branch] == 0, (1.0 - u) * self.lengths[0], self.lengths[0] + u * self.lengths[1])

    def point(self, branch: NDArray[np.int64], u: NDArray[np.float64]) -> Points:
        """X(u), (n, 2), of the curve each branch belongs to: curve.evaluate itself."""
        branch, u = np.asarray(branch).ravel(), np.asarray(u, dtype=float).ravel()
        out = np.empty((len(u), 2))
        for index, curve in enumerate(self.curves):
            on = self.curve_of[branch] == index
            if np.any(on):
                out[on] = curve.evaluate(u[on])
        return out

    def solve(self, branch: NDArray[np.int64], m: NDArray[np.float64]) -> NDArray[np.float64]:
        """u where the given branches cross x = m (any common shape): m(u) is strictly
        monotone on a branch and m lies in its closed m range, so m(u) - m keeps one sign on
        the lo side, and _BISECTIONS halvings of [u0, u1] reach the root to round-off."""
        m = np.asarray(m, dtype=float)
        branch = np.broadcast_to(np.asarray(branch), m.shape).ravel()
        target = m.ravel()
        lo, hi = self.u0[branch].copy(), self.u1[branch].copy()
        start = self.m0[branch] - target                     # m(u) - m at the branch start; its sign stays on the lo side
        for _ in range(_BISECTIONS):
            mid = 0.5 * (lo + hi)
            value = self.point(branch, mid)[:, 0] - target
            same = np.sign(value) == np.sign(start)
            lo, hi = np.where(same, mid, lo), np.where(same, hi, mid)
            start = np.where(same, value, start)
        return (0.5 * (lo + hi)).reshape(m.shape)

    def cut(self, m: NDArray[np.float64]) -> _Cut:
        station = np.asarray(m, dtype=float)
        crossed = (self.lo_m <= station[:, None]) & (station[:, None] < self.hi_m)
        rows, branches = np.nonzero(crossed)
        u = np.full(crossed.shape, np.nan)
        height = np.full(crossed.shape, np.inf)
        if rows.size:
            u[rows, branches] = self.solve(branches, station[rows])
            height[rows, branches] = self.point(branches, u[rows, branches])[:, 1]
        order = np.argsort(height, axis=1, kind="stable")
        height, u = np.take_along_axis(height, order, axis=1), np.take_along_axis(u, order, axis=1)
        count = np.count_nonzero(crossed, axis=1)
        if np.any(count % 2):
            raise GridError("a station line crosses the blade's curves an odd number of times: the outline is not closed")
        finite = np.where(np.isfinite(height), height, 0.0)
        exits = finite[:, 1::2]
        pairs = exits - finite[:, 0::2][:, :exits.shape[1]]          # metal intervals, bottom to top
        inside = np.arange(pairs.shape[1])[None, :] < (count // 2)[:, None]
        metal = np.where(inside, pairs, 0.0).sum(axis=1)
        return _Cut(m=station, count=count, segment=order, height=height, metal=metal, touch=self.touch, u=u)

    def passage_sides(self, cut: _Cut, stations: slice) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """y of the passage-facing top and bottom of the cut at the given blade stations: the
        curve points themselves (the top is the lower wall, the bottom moved one pitch the
        upper)."""
        count, height = cut.count[stations], cut.height[stations]
        return np.take_along_axis(height, (count - 1)[:, None], axis=1)[:, 0], height[:, 0]

    def identity(self, cut: _Cut, stations: slice, first_blade_i: int) -> WallIdentity:
        """As _BladeRow.identity, on the curves: which curve, which u and where along its
        wall's contour each wall node lies, checked the same way (module docstring, sections 3
        and 7)."""
        count, segment, u = cut.count[stations], cut.segment[stations], cut.u[stations]
        last = (count - 1)[:, None]
        top, bottom = np.take_along_axis(segment, last, axis=1)[:, 0], segment[:, 0]
        u_top, u_bottom = np.take_along_axis(u, last, axis=1)[:, 0], u[:, 0]
        crossing = np.arange(segment.shape[1])[None, :] < count[:, None]
        lower_contour = self.lower_contour
        if np.any(lower_contour[top]) or not np.all(lower_contour[bottom]):
            raise GridError("a wall node is not on its contour: the top of a cut must lie on the blade's upper "
                            "contour and the bottom on its lower contour")
        total, tau_tip, tau_tail = self.total, self.tau_tip, self.tau_tail

        def side(branches: NDArray[np.int64], at: NDArray[np.float64], contour: NDArray[np.float64],
                 hidden: NDArray[np.int64]) -> WallSide:
            return WallSide(surface=np.where(self.curve_of[branches] == 0, "suction", "pressure"), parameter=at, u=at,
                            contour=contour, hidden=hidden)

        lower = side(top, u_top, np.mod(tau_tip - self.tau(top, u_top), total),
                     np.count_nonzero(~lower_contour[segment] & crossing, axis=1) - 1)
        upper = side(bottom, u_bottom, np.mod(self.tau(bottom, u_bottom) - tau_tip, total),
                     np.count_nonzero(lower_contour[segment] & crossing, axis=1) - 1)
        # geometry.py's LE point (tau = L_s) and TE point (tau = 0 = total) on each contour
        edges = (self.lengths[0], 0.0)
        on_lower = [np.mod(t - tau_tip, total) <= np.mod(tau_tail - tau_tip, total) for t in edges]
        splits = {"lower": [np.mod(tau_tip - t, total) for t, low in zip(edges, on_lower) if not low],
                  "upper": [np.mod(t - tau_tip, total) for t, low in zip(edges, on_lower) if low]}
        _check_walls(lower, upper, splits, self.touch, first_blade_i)
        return WallIdentity(stations=np.arange(first_blade_i, first_blade_i + len(count)), lower=lower, upper=upper)

    def smooth(self, branch: NDArray[np.int64], m: NDArray[np.float64]) -> NDArray[np.float64]:
        """y where the given branches cross x = m: the exact curve points."""
        return self.point(branch, self.solve(branch, m))[:, 1].reshape(np.shape(m))

    @cached_property
    def pieces(self) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]:
        """The blade's m range cut at the m of every branch end and every interior knot of
        either curve into pieces lo..hi, with the branches of the top and the bottom of the
        cut inside each piece: no branch ends inside a piece, so the branches a line meets,
        and their order, stay the same across it (module docstring, section 7)."""
        knots = [curve.evaluate(curve.interior_knots)[:, 0] for curve in self.curves]
        vertices = np.unique(np.concatenate((self.m0, self.m1, *knots)))
        lo, hi = vertices[:-1], vertices[1:]
        cut = self.cut(0.5 * (lo + hi))
        top = np.take_along_axis(cut.segment, (cut.count - 1)[:, None], axis=1)[:, 0]
        return lo, hi, top, cut.segment[:, 0]

    def widest_cut(self) -> tuple[float, float]:
        """The largest span top - bottom of the curves' cut at the piece ends, and its m (the
        passage check's narrowest width settles the span between them)."""
        lo, hi, top, bottom = self.pieces
        ends, top, bottom = np.concatenate((lo, hi)), np.concatenate((top, top)), np.concatenate((bottom, bottom))
        span = self.smooth(top, ends) - self.smooth(bottom, ends)
        k = int(np.argsort(span, kind="stable")[-1])
        return float(span[k]), float(ends[k])


def _turning_points(curve: ParametricBladeCurve, axis: int) -> NDArray[np.float64]:
    """u in (0, 1) where coordinate axis (0: m, 1: y) of a C2 curve turns: the roots of its u
    derivative, a spline of one degree less, found per polynomial piece (PPoly.roots, exact
    for a polynomial), candidates closer than _TURN_MERGE taken as one, kept where the
    derivative changes sign across +- _TURN_BRACKET (a root of even order is no turn) and
    refined by bisection on the curve's own derivative to round-off."""
    derivative = BSpline(curve.knots, curve.control_points[:, axis], curve.degree).derivative()
    roots = PPoly.from_spline(derivative).roots(extrapolate=False)
    roots = np.unique(roots[np.isfinite(roots) & (roots > 0.0) & (roots < 1.0)])
    if roots.size > 1:
        roots = roots[np.concatenate(([True], np.diff(roots) > _TURN_MERGE))]
    if roots.size == 0:
        return roots
    gaps = np.diff(np.concatenate(([0.0], roots, [1.0])))
    half = np.minimum(_TURN_BRACKET, 0.5 * np.minimum(gaps[:-1], gaps[1:]))
    lo, hi = roots - half, roots + half
    start = curve.derivative(lo, 1)[:, axis]
    keep = np.sign(start) * np.sign(curve.derivative(hi, 1)[:, axis]) < 0.0
    lo, hi, start = lo[keep], hi[keep], start[keep]
    for _ in range(_BISECTIONS):
        mid = 0.5 * (lo + hi)
        value = curve.derivative(mid, 1)[:, axis]
        same = np.sign(value) == np.sign(start)
        lo, hi = np.where(same, mid, lo), np.where(same, hi, mid)
        start = np.where(same, value, start)
    return 0.5 * (lo + hi)


def _smooth_crossing(outline: Points, t: NDArray[np.float64], x_of_t: PchipInterpolator, y_of_t: PchipInterpolator,
                     segment: NDArray[np.int64], m: NDArray[np.float64]) -> NDArray[np.float64]:
    """y where the parametric PCHIP crosses x = m on the given polyline segments."""
    return y_of_t(_smooth_parameter(outline, t, x_of_t, segment, m))


def _smooth_parameter(outline: Points, t: NDArray[np.float64], x_of_t: PchipInterpolator,
                      segment: NDArray[np.int64], m: NDArray[np.float64]) -> NDArray[np.float64]:
    """t where the parametric PCHIP crosses x = m on the given polyline segments: x(t) is
    monotone on each segment, so bisection on [t_k, t_k+1] finds the one crossing."""
    lo, hi = t[segment].copy(), t[segment + 1].copy()
    start = outline[segment, 0] - m                 # x(t) - m at the segment start; its sign stays on the lo side
    for _ in range(64):
        mid = 0.5 * (lo + hi)
        value = x_of_t(mid) - m
        same = np.sign(value) == np.sign(start)
        lo, hi = np.where(same, mid, lo), np.where(same, hi, mid)
        start = np.where(same, value, start)
    return 0.5 * (lo + hi)


def _across(eta: NDArray[np.float64], lower: NDArray[np.float64], upper: NDArray[np.float64]) -> NDArray[np.float64]:
    """y of the nodes, (n_stations, n_eta): (1 - eta) lower + eta upper at each station."""
    return (1.0 - eta) * lower[:, None] + eta * upper[:, None]


# --- Side lines ------------------------------------------------------------------------

def _samples(start: float, stop: float, n_passage: int, blade_length: float) -> NDArray[np.float64]:
    """Uniform samples from start to stop, both exact, in passage cells of
    blade_length / n_passage, rounded and at least one."""
    n_cells = max(1, round((stop - start) * n_passage / blade_length))
    return np.linspace(start, stop, n_cells + 1)


def _sample_fraction(n_samples: int, junction_last: bool) -> NDArray[np.float64]:
    """How far uniform samples are from the junction toward the far station, from the
    sample index: exactly 0 at the junction and 1 at the far station, whatever the length
    unit."""
    k = np.arange(n_samples, dtype=float)
    n = n_samples - 1
    return (n - k) / n if junction_last else k / n


def _station_fraction(m: NDArray[np.float64], junction_m: float, far_m: float) -> NDArray[np.float64]:
    """How far grid stations m are from the junction toward the far station,
    (m - junction_m) / (far_m - junction_m): from 0 to 1, since inlet stations lie ahead
    of the blade and so of the LE, and exit stations behind it and so behind the TE."""
    return (m - junction_m) / (far_m - junction_m)


def _side_line(m: NDArray[np.float64], fraction: NDArray[np.float64], junction: NDArray[np.float64],
               junction_angle: float, far_angle: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """beta and y of the lower side line at m: beta linear in m from junction_angle
    (fraction 0) to far_angle (fraction 1), and y the integral of tan(beta) through the
    junction point. beta is interpolated from the nearer end, so both ends are exact and
    equal angles give exactly that angle everywhere: the straight line."""
    turn = far_angle - junction_angle
    beta = np.where(fraction <= 0.5, junction_angle + fraction * turn, far_angle - (1.0 - fraction) * turn)
    return beta, junction[1] + (m - junction[0]) * _mean_tan(beta, junction_angle)


def _inlet_side(b: PassageBoundaries, m: NDArray[np.float64]) -> NDArray[np.float64]:
    """y of the lower inlet side line at stations m: through the exact LE point, the end of
    upstream_lower_boundary, at kappa_le, the end of upstream_beta."""
    junction = b.upstream_lower_boundary[-1]
    return _side_line(m, _station_fraction(m, b.m_le, b.m_in), junction, b.upstream_beta[-1], b.inlet_angle)[1]


def _exit_side(b: PassageBoundaries, m: NDArray[np.float64]) -> NDArray[np.float64]:
    """y of the lower exit side line at stations m: through the exact TE point, the start
    of downstream_lower_boundary, at kappa_te, the start of downstream_beta."""
    junction = b.downstream_lower_boundary[0]
    return _side_line(m, _station_fraction(m, b.m_te, b.m_out), junction, b.downstream_beta[0], b.exit_angle)[1]


def _mean_tan(a: NDArray[np.float64], b: float) -> NDArray[np.float64]:
    """Mean of tan over the angle interval [a, b], ln(cos a / cos b) / (b - a), and tan(b)
    where a = b. With d = b - a, cos a / cos b = 1 + x, x = d q,
        q = tan(b) sinc(d) - sin(d / 2) sinc(d / 2),    sinc(z) = sin(z) / z,
    so the mean is q log1p(x) / x: no division by d, and no cancellation as d -> 0."""
    d = b - np.asarray(a, dtype=float)
    q = math.tan(b) * np.sinc(d / math.pi) - np.sin(0.5 * d) * np.sinc(0.5 * d / math.pi)
    x = d * q
    ratio = np.ones_like(x)
    nonzero = x != 0.0
    ratio[nonzero] = np.log1p(x[nonzero]) / x[nonzero]
    return q * ratio


# --- Input checks ----------------------------------------------------------------------

def _check_inputs(cascade: Cascade, settings: GridSettings, caller: str) -> None:
    if not isinstance(cascade, Cascade):
        raise TypeError(f"{caller} takes a Cascade, got {type(cascade).__name__}")
    if not isinstance(settings, GridSettings):
        raise TypeError(f"{caller} takes GridSettings, got {type(settings).__name__}")


def _check_angle(angle: float, name: str) -> None:
    if not (isinstance(angle, (int, float)) and not isinstance(angle, bool) and math.isfinite(angle)
            and abs(angle) < 0.5 * math.pi):
        raise ValueError(f"{name} must be a finite angle strictly between -pi/2 and pi/2 radians, got {angle!r}; "
                         f"convert an angle in degrees with math.radians")


def _check_blade_angle(kappa: float, name: str, side: str) -> None:
    if not abs(kappa) < 0.5 * math.pi:
        raise ValueError(f"{name} = {math.degrees(kappa):.6g} deg: the {side} side boundary cannot turn to a "
                         f"blade angle at or beyond 90 deg from axial")


def _positive_number(value: float, name: str) -> float:
    if not (isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0.0):
        raise ValueError(f"{name} must be a positive number of chords, got {value!r}")
    return float(value)


def _whole_number(value: int, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be a whole number of at least {minimum}, got {value!r}")
    return int(value)
