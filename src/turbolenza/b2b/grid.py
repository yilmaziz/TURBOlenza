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
Section 9 (2026-11-13, under review) adds the geometry-driven standard grids (five named
levels, build_grid, build_grid_family); make_grid's explicit n_passage path is unchanged bit
for bit. node_index and node_theta are not implemented. Deferred, not resolved: the straight m-links
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

8. Admissible n_passage: the wall-station topology (2026-11-12)
    The meridional grid stays uniform (section 2); not every n_passage gives the passage
    topology of Aungier's Fig. 5-3 at the blade's ends. A uniform grid is admissible when
    its first blade station cuts the reference blade exactly twice, once on geometry.py's
    suction and once on its pressure surface, and so does its last blade station, neither
    line passing through the LE or TE point itself (EdgeStationTopology.valid). The surfaces
    are those of the crossings (C2: the curve each branch belongs to; C1: the outline
    segment's side of the LE point), the same ones Grid.wall_identity names.

    Why it fails. The nose tip, the most upstream point, lies on one surface (the suction
    surface for positive stagger) a window w_LE = m_le - m_tip ahead of the LE point; a
    station line in (m_tip, m_le] crosses that surface twice (the surface runs from the LE
    point to the tip and back past m_le) and the other one not at all. The TE mirrors it
    with w_TE = m_tail - m_te. The first blade station is the first station past the tip,
    so it lies in (m_tip, m_tip + dm], and the last in [m_tail - dm, m_tail): which part of
    the nose a station meets is set by where the uniform stations fall, so admissible and
    inadmissible n_passage alternate. Once dm <= w (either window) every station placement
    fails, and since dm never grows with n_passage every finer grid fails too
    (n_passage_ceiling).

    The rule rejects; it never repairs. make_grid builds the stations of settings.n_passage
    as before and, by default, accepts any topology (the tracked-contour walls of section 3
    describe every such grid correctly); make_grid(..., require_ss_ps=True) and
    validate_uniform_grid_topology raise GridTopologyError for an inadmissible grid, naming
    the failing station, the nose tip (tail end) and the LE (TE) point, the nearest
    admissible n_passage on either side, the finest admissible one and the ceiling. No
    station is moved and no n_passage is substituted. find_valid_n_passage lists the
    admissible values in a range and wall_station_topology reads a built grid; both are
    read-only.

    What the rule does not change: the mapping. A valid first blade station lies past the
    LE point, but the side lines still meet the walls at the nose tip and tail end with the
    jump of their anchoring at the LE / TE point (section 1); the two are separate issues.

9. Geometry-driven standard grids (2026-11-13; UNDER REVIEW)
    Two modes. EXPLICIT (legacy; verification, regression and manufactured tests):
    make_grid with GridSettings(n_passage, n_eta), the lattice of section 2, unchanged bit for
    bit; whether its blade ends are admissible depends on where those stations happen to fall
    (section 8). STANDARD (the physical and scientific interface): build_grid(cascade, level)
    for one of GRID_LEVELS = very_coarse, coarse, medium, fine, very_fine, build_grid_family
    for all five (one GridPackage), or GridSettings(level=...) through make_grid and so the
    solver. The geometry owns admissibility; no n_passage is remembered.

    Transitions (passage_transitions). From the m ranges of the outline's monotone pieces (C2
    branches, C1 segments; half-open, as the cuts count them) the crossings per surface are
    known on every interval between piece ends. The SS + PS-valid interval (m_start, m_end) is
    the run of intervals with exactly one suction and one pressure crossing that holds the
    middle of the chord. Three kinds of blade end are kept apart: the outline's extremes
    (nose tip m_tip, tail end m_tail), geometry.py's construction points (m_le, m_te) and
    (m_start, m_end). For the 65-(12A10)10 at 33 deg (C2) m_start = m_le and m_end = m_te; the
    nose tip lies 0.004524 c ahead on the suction surface (lines between cross it twice) and a
    Kovach tail end 1.35e-5 c behind on the pressure surface (emery_base: none). The LE window
    (m_tip, m_start] and the TE window [m_end, m_tail) may hold no station at all.

    Lattice. The LE face (m_tip + m_start) / 2 and the TE face (m_end + m_tail) / 2, the
    middles of the two windows, are cell faces: n_cells cells of dm = (te_face - le_face) /
    n_cells, stations at the cell centres. So f - 1 and f lie (dm - w_LE) / 2 ahead of the nose
    tip and behind m_start, the same clearance on both sides of the window (l and l + 1 at the
    TE), at every level; a station can never land on the LE or TE point, and a lattice exists
    whenever dm exceeds both windows (the bound of section 8, now reached by every level, not
    by lucky n_passage). Each clearance must be at least TRANSITION_CLEARANCE = 0.1 dm (no
    station nearly through the nose tip or the LE point). f is then read, not assumed: the
    first station whose line crosses the blade once on each surface, f - 1 crosses it not at
    all; the same for l and l + 1 (checked on the cut, below).

    Levels and the candidate search. Target dm / c per level: MEDIUM_DM_OVER_CHORD = 0.020
    times LEVEL_REFINEMENT_RATIO = sqrt(2) to the power 2 - k: 0.0400, 0.0283, 0.0200, 0.0141,
    0.0100 (very_coarse, medium and very_fine a factor 2 apart). Candidates: every n_cells whose
    dm lies within CANDIDATE_BAND = 10 % of the target, tried nearest first (|ln(dm/target)|);
    the first that passes every condition is the level's grid. Neighboring bands cannot
    overlap (0.9 sqrt 2 > 1.1), so dm falls strictly from level to level. Rejection reasons
    (CandidateStatistics, GridGenerationError): invalid LE cut, invalid TE cut (a window too
    wide for the clearance, or a station beside the blade meeting it), wrong branch identity
    (a blade station not one suction + one pressure crossing, through the LE or TE point, or a
    wall changing surface), extension outside tolerance, eta-resolution issue, other topology
    failure (a GridError of the Grid itself, metrics.py refusing it).

    Eta. n_eta - 1 = round(pitch / (CELL_ASPECT dm)) with CELL_ASPECT = 1.8, the ratio of the
    periodic-line cell height pitch deta to dm that the project's hand-picked grids had (43 x
    29, 61 x 41, 87 x 59: 1.79 to 1.83). Not a requirement of Aungier's: it keeps the m and eta
    resolutions in one proportion across the levels (2-D refinement, deta strictly falling
    too). Fewer than MIN_ETA_CELLS = 6 cells reject the candidate.

    Extensions. With dm fixed, whole cells reach out to the nominal lengths (1 chord each, from
    m_le and m_te as in section 1): k_up = round((m_first - (m_le - L_up)) / dm) inlet stations,
    and so for the exit. The actual lengths differ from the nominal ones by at most dm / 2;
    more than EXTENSION_TOLERANCE_CHORDS = 0.05 c rejects the candidate. Exact uniform dm comes
    first, an exact 1.000 c extension second. The side lines are those of section 1 for these
    lengths; the stations are linspace(m_in, m_out), uniform to round-off (measured 2.6e-16 to
    5.4e-16 c on the five 65-(12A10)10 levels; UNIFORM_SPACING_TOLERANCE 1e-13 c).

    Validation (validate_standard_grid, on every accepted grid): distinct stations, uniform dm
    and deta, 1 <= f <= l <= n_m - 2, f - 1 and l + 1 meeting no blade, every blade station
    inside (m_start, m_end) with one suction and one pressure crossing, the wall identity
    constant along each wall (no flip) and no hidden crossing, finite nodes, positive width,
    and metrics.py building the grid's metrics with S > 0. A failed level raises
    GridGenerationError (level, geometry_id, every candidate and its reason); no malformed grid
    is ever returned. A geometry without an SS + PS-valid interval, or with a closed passage,
    fails at every level.

    Timing (time.perf_counter): StandardGrid.timing holds the geometry interrogation (row,
    transitions, passage check), the search, and inside it the intersections (station cuts),
    the construction (boundaries, Grid) and the validation; GridPackage.timing the shared
    interrogation, each level's search and the total. The stations do not depend on the flow
    angles: a level's grid is the same at any inlet or exit angle but for the side lines.
"""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
from functools import cached_property
from types import MappingProxyType

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import BSpline, PchipInterpolator, PPoly

from turbolenza.b2b.config import GridSettings
from turbolenza.b2b.geometry import Cascade

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


class GridTopologyError(GridError):
    """A uniform grid whose first or last blade station does not cut exactly one suction
    and one pressure surface (module docstring, section 8): its n_passage is not admissible
    for the cascade. Raised by validate_uniform_grid_topology and by
    make_grid(..., require_ss_ps=True); the grid is never changed to avoid it.

    edge: "LE", "TE" or "LE and TE", the station(s) that fail; n_passage: the requested
    value (None when not known); nearest_lower, nearest_higher: the admissible n_passage
    next to it on either side (None when there is none, or n_passage is not known); finest:
    the largest admissible n_passage (None when there is none); ceiling: the first
    n_passage from which every finer one fails (None when the blade has neither window)."""

    def __init__(self, message: str, *, edge: str, n_passage: int | None, nearest_lower: int | None,
                 nearest_higher: int | None, finest: int | None, ceiling: int | None) -> None:
        super().__init__(message)
        self.edge, self.n_passage = edge, n_passage
        self.nearest_lower, self.nearest_higher = nearest_lower, nearest_higher
        self.finest, self.ceiling = finest, ceiling


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
              exit_angle: float | None = None, *, require_ss_ps: bool = False) -> Grid:
    """The structured grid of one passage of cascade: make_passage_boundaries, then the
    uniform stations, the blade-passage topology from the cuts of the station lines with
    the blade row, the wall values at the blade stations and the nodes (module docstring,
    sections 2 to 5). For a C2 cascade the blade is its authoritative curves throughout
    (section 7): the walls are the curves' points, with their u in Grid.wall_identity.
    Angles as in make_passage_boundaries, in radians; exit_angle None takes
    cascade.kappa_te.

    require_ss_ps (keyword only): also require the first and the last blade station each to
    cut exactly one suction and one pressure surface (section 8) and raise
    GridTopologyError otherwise. The grid is never changed to meet it: the stations are
    those of settings.n_passage either way. False (the default) builds the grid as before,
    whatever its wall-station topology; validate_uniform_grid_topology checks such a grid
    afterwards.

    Raises what make_passage_boundaries raises, ValueError for n_eta below 2, and
    GridError when the passage is not open over the whole blade (the passage check of
    section 4: the straight-segment cut spans a pitch or more somewhere, or the smooth walls
    leave no width above TOUCH chords somewhere; checked before any station, so the same
    for every n_passage and n_eta), when no station cuts the blade (n_passage too small for
    the blade to be seen), the stations that cut it are not one run, or the blade reaches a
    far station; with require_ss_ps, GridTopologyError (a GridError) for an inadmissible
    n_passage. Only to name the blade in the messages, make_grid reads cascade.metadata
    and cascade.stagger.
    """
    _check_inputs(cascade, settings, "make_grid")
    if settings.level is not None:                  # the geometry-driven standard-grid mode (section 9)
        if settings.n_passage is not None or settings.n_eta is not None:
            raise ValueError(f"GridSettings with level = {settings.level!r} take no n_passage or n_eta (got "
                             f"{settings.n_passage!r}, {settings.n_eta!r}): the level sets the resolution; give either "
                             f"a level or n_passage and n_eta (the explicit mode)")
        return build_grid(cascade, settings.level, inlet_angle=inlet_angle, exit_angle=exit_angle,
                          extension_chords=(settings.upstream_extension_chords, settings.downstream_extension_chords)).grid
    b = make_passage_boundaries(cascade, settings, inlet_angle, exit_angle)
    n_eta = _whole_number(settings.n_eta, "n_eta", 2)
    row = _row_of(b)
    _check_passage(cascade, b, row)

    grid = _grid_on_stations(b, row, _stations(b, settings.n_passage), n_eta)
    if require_ss_ps:
        validate_uniform_grid_topology(grid, n_passage=settings.n_passage, stagger=cascade.stagger)
    return grid


def _grid_on_stations(b: PassageBoundaries, row: _Row, m: NDArray[np.float64], n_eta: int,
                      cut: _Cut | None = None) -> Grid:
    """The Grid on given stations m (make_grid's sections 3 to 5): the blade-passage topology
    from the cuts of the station lines, the wall values at the blade stations and the nodes.
    make_grid gives it its uniform stations of n_passage; build_grid (section 9) its
    geometry-anchored lattice, with the cut it has already made of those stations."""
    if cut is None:
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


def _stations(b: PassageBoundaries, n_passage: int) -> NDArray[np.float64]:
    """The uniform stations of make_grid (module docstring, section 2): one constant dm, as
    close to (m_te - m_le) / n_passage as the exact m_in and m_out allow."""
    dm_target = (b.m_te - b.m_le) / n_passage
    return np.linspace(b.m_in, b.m_out, max(1, round((b.m_out - b.m_in) / dm_target)) + 1)


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


# --- Admissible n_passage: the wall-station topology (section 8) ---------------------

@dataclass(frozen=True)
class EdgeStationTopology:
    """How the first (edge "LE") or the last (edge "TE") blade station of a uniform grid
    cuts the reference blade (module docstring, section 8): its crossings with the outline
    (C2: the authoritative curves) and on which of geometry.py's named surfaces they lie."""

    edge: str                   # "LE": the first blade station; "TE": the last blade station
    station: int                # its index i
    m: float                    # its m
    crossings: int              # crossings of the reference blade on the station line
    suction: int                # of them on the suction surface
    pressure: int               # of them on the pressure surface
    branch_point: float         # m of geometry.py's LE (TE) point, where the two surfaces meet
    extreme: float              # m of the blade's most upstream (downstream) point, its nose tip (tail end)
    on_branch_point: bool       # the line passes within TOUCH chord of that LE (TE) point itself

    @property
    def valid(self) -> bool:
        """One suction and one pressure crossing, neither of them on the branch point."""
        return self.crossings == 2 and self.suction == 1 and self.pressure == 1 and not self.on_branch_point


def wall_station_topology(grid: Grid) -> tuple[EdgeStationTopology, EdgeStationTopology]:
    """(LE, TE): the first and the last blade station of grid read against the blade
    (module docstring, section 8). Read-only."""
    if not isinstance(grid, Grid):
        raise TypeError(f"wall_station_topology takes a Grid, got {type(grid).__name__}")
    b = grid.boundaries
    row = _row_of(b)
    cut = row.cut(grid.m[[grid.first_blade_i, grid.last_blade_i]])
    return (_edge_topology(row, b, cut, 0, "LE", grid.first_blade_i),
            _edge_topology(row, b, cut, 1, "TE", grid.last_blade_i))


def validate_uniform_grid_topology(grid: Grid, *, n_passage: int | None = None, stagger: float | None = None) -> None:
    """GridTopologyError unless the first and the last blade station of grid each cut
    exactly one suction and one pressure surface (module docstring, section 8). Nothing is
    changed or chosen: the error names the station, the nose tip (tail end) and the LE (TE)
    point, and, for the n_passage that built the grid (keyword, optional), the nearest
    admissible values on either side; it also gives the finest admissible n_passage and the
    first n_passage from which every finer one fails. stagger (radians, optional) only
    names the cascade in the message."""
    le, te = wall_station_topology(grid)
    if le.valid and te.valid:
        return
    b = grid.boundaries
    row = _row_of(b)
    ceiling = _ceiling(b, row)

    def admissible(n: int) -> bool:
        topology = _lattice_topology(b, row, n)
        return topology is not None and topology[0].valid and topology[1].valid

    finest = None
    if ceiling is not None:
        finest = next((n for n in range(ceiling - 1, 0, -1) if admissible(n)), None)
    lower = higher = None
    if n_passage is not None:
        n_passage = _whole_number(n_passage, "n_passage", 1)
        if ceiling is not None and n_passage >= ceiling:
            lower = finest
        else:
            lower = next((n for n in range(n_passage - 1, 0, -1) if admissible(n)), None)
            top = ceiling if ceiling is not None else 4 * n_passage + 100
            higher = next((n for n in range(n_passage + 1, top) if admissible(n)), None)
    bad = [t for t in (le, te) if not t.valid]
    edge = " and ".join(t.edge for t in bad)
    named = f" (stagger {math.degrees(stagger):.6g} deg)" if stagger is not None else ""
    head = (f"n_passage = {n_passage} is not admissible for this cascade{named}" if n_passage is not None else
            f"this uniform grid is not admissible for its cascade{named}")
    lines = [head + ": the first and the last blade station must each cut one suction and one pressure surface."]
    lines += [_describe_edge(t) for t in bad]
    choice = "Choose another admissible n_passage; none is substituted."
    if n_passage is not None:
        choice += (f" Nearest admissible: lower {lower if lower is not None else 'none'}, higher "
                   f"{higher if higher is not None else 'none'}.")
    if ceiling is not None:
        tip, tail = row.extremes
        windows = {"LE window m_le - nose tip": b.m_le - tip, "TE window tail end - m_te": tail - b.m_te}
        name, width = max(((k, w) for k, w in windows.items() if w > row.touch), key=lambda item: item[1])
        choice += (f" Finest admissible n_passage: {finest if finest is not None else 'none'}; every n_passage from "
                   f"{ceiling} on fails (dm <= {width:.4g}, the {name}). Coarser values can fail too: which ones pass "
                   f"depends on where the uniform stations fall (find_valid_n_passage lists them).")
    else:
        choice += " This blade has no LE or TE window, so no refinement ceiling exists."
    lines.append(choice)
    raise GridTopologyError("\n".join(lines), edge=edge, n_passage=n_passage, nearest_lower=lower,
                            nearest_higher=higher, finest=finest, ceiling=ceiling)


def find_valid_n_passage(cascade: Cascade, settings: GridSettings, inlet_angle: float, n_values) -> tuple[int, ...]:
    """The admissible n_passage among n_values (module docstring, section 8), ascending:
    those whose uniform stations, make_grid's for this cascade and settings' extension
    lengths, give a first and a last blade station that each cut one suction and one
    pressure surface. settings.n_passage is not used. Read-only: no Grid is built and
    nothing is chosen. Raises what make_passage_boundaries raises, and GridError when the
    passage is not open (the passage check of section 4, as make_grid)."""
    _check_inputs(cascade, settings, "find_valid_n_passage")
    b = make_passage_boundaries(cascade, settings, inlet_angle)
    row = _row_of(b)
    _check_passage(cascade, b, row)
    out = set()
    for value in n_values:
        n = _whole_number(value, "n_passage", 1)
        topology = _lattice_topology(b, row, n)
        if topology is not None and topology[0].valid and topology[1].valid:
            out.add(n)
    return tuple(sorted(out))


def n_passage_ceiling(cascade: Cascade, settings: GridSettings, inlet_angle: float) -> int | None:
    """The first n_passage from which every finer n_passage fails the wall-station
    topology (module docstring, section 8): the smallest n_passage whose dm is at most the
    LE window (m_le - nose tip) or the TE window (tail end - m_te); None when the blade has
    neither window. Below it, admissibility depends on where the stations fall: use
    find_valid_n_passage. Read-only."""
    _check_inputs(cascade, settings, "n_passage_ceiling")
    b = make_passage_boundaries(cascade, settings, inlet_angle)
    return _ceiling(b, _row_of(b))


def _edge_topology(row: _Row, b: PassageBoundaries, cut: _Cut, k: int, edge: str, station: int) -> EdgeStationTopology:
    """EdgeStationTopology of row k of cut (the station's crossings, in height order)."""
    count = int(cut.count[k])
    names = row.surface_of(cut.segment[k, :count])
    m = float(cut.m[k])
    tip, tail = row.extremes
    branch, extreme = (b.m_le, tip) if edge == "LE" else (b.m_te, tail)
    return EdgeStationTopology(edge=edge, station=int(station), m=m, crossings=count,
                               suction=int(np.count_nonzero(names == "suction")),
                               pressure=int(np.count_nonzero(names == "pressure")), branch_point=float(branch),
                               extreme=float(extreme), on_branch_point=bool(abs(m - branch) <= row.touch))


def _lattice_topology(b: PassageBoundaries, row: _Row,
                      n_passage: int) -> tuple[EdgeStationTopology, EdgeStationTopology] | None:
    """(LE, TE) of the uniform grid of n_passage by make_grid's own stations and blade-run
    rule, without building it; None when make_grid would refuse the stations (no station
    cuts the blade, the stations that cut it are not one run, or the blade reaches a far
    station)."""
    m = _stations(b, n_passage)
    cut = row.cut(m)
    blade = np.flatnonzero(cut.cuts_blade)
    if blade.size == 0 or blade.size != blade[-1] - blade[0] + 1 or blade[0] == 0 or blade[-1] == len(m) - 1:
        return None
    first, last = int(blade[0]), int(blade[-1])
    return _edge_topology(row, b, cut, first, "LE", first), _edge_topology(row, b, cut, last, "TE", last)


def _ceiling(b: PassageBoundaries, row: _Row) -> int | None:
    """Smallest n_passage whose station spacing dm is at most a window, w = m_le - nose tip
    or tail end - m_te (the smaller of the two such n_passage); None without a window.

    Why every finer n_passage then fails: the first blade station is the first station
    past the nose tip, so it lies in (m_tip, m_tip + dm]; with dm <= w that is inside
    (m_tip, m_le], where the line crosses the surface that holds the nose tip at least twice
    (the surface runs from the LE point to the tip and back past m_le) or passes through
    the LE point itself. The TE mirrors it: the last blade station lies in [m_tail - dm,
    m_tail). dm = (m_out - m_in) / round((m_out - m_in) n / (m_te - m_le)) never grows with
    n, so the condition holds for every larger n."""
    tip, tail = row.extremes
    windows = [w for w in (b.m_le - tip, tail - b.m_te) if w > row.touch]
    if not windows:
        return None
    width, length = b.m_out - b.m_in, b.m_te - b.m_le

    def spacing(n: int) -> float:
        return width / max(1, round(width / (length / n)))

    found = []
    for w in windows:
        n = max(1, int(length / w) - 2)
        while spacing(n) > w:
            n += 1
        while n > 1 and spacing(n - 1) <= w:
            n -= 1
        found.append(n)
    return min(found)


def _describe_edge(t: EdgeStationTopology) -> str:
    """The error-message paragraph for one failing edge station."""
    if t.edge == "LE":
        where, point, extreme, inside = "First", "LE point", "nose tip", t.extreme < t.m < t.branch_point
        interval = "the nose-return interval between the nose tip and the LE point"
    else:
        where, point, extreme, inside = "Last", "TE point", "tail end", t.branch_point < t.m < t.extreme
        interval = "the tail-return interval between the TE point and the tail end"
    text = (f"{where} blade station i = {t.station}: m = {t.m:.6g}; {extreme} m = {t.extreme:.6g}; {point} "
            f"(suction / pressure branch point) m = {t.branch_point:.6g}. ")
    if t.on_branch_point:
        return text + (f"The station line passes through the {point} itself, where the two surfaces meet, so its "
                       f"wall nodes are not one suction and one pressure node.")
    if inside:
        surface = "suction" if t.suction >= t.pressure else "pressure"
        return text + (f"The station lies inside {interval}, so its line crosses the {surface} surface twice "
                       f"({t.suction} suction, {t.pressure} pressure crossings) instead of producing one suction and "
                       f"one pressure wall node.")
    return text + (f"Its line crosses the blade {t.crossings} times ({t.suction} suction, {t.pressure} pressure) "
                   f"instead of once on each surface.")


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

    def surface_of(self, segment: NDArray[np.int64]) -> NDArray[np.str_]:
        """geometry.py's named surface of each polyline segment: the outline runs over the
        suction surface (TE -> LE) up to vertex n_suction - 1, the LE point, then over the
        pressure surface."""
        return np.where(np.asarray(segment) < self.n_suction - 1, "suction", "pressure")

    @property
    def extremes(self) -> tuple[float, float]:
        """m of the blade's most upstream and most downstream point (the nose tip and the tail end)."""
        return float(self.outline[self.tip, 0]), float(self.outline[self.tail, 0])

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
        self.m_tip, self.m_tail = float(ends_m[order[0]]), float(ends_m[order[-1]])
        middle = self.tau(branches, 0.5 * (self.u0 + self.u1))
        self.lower_contour = np.mod(middle - self.tau_tip, self.total) < np.mod(self.tau_tail - self.tau_tip, self.total)

    def surface_of(self, branch: NDArray[np.int64]) -> NDArray[np.str_]:
        """geometry.py's named surface of each branch: the curve it belongs to."""
        return np.where(self.curve_of[np.asarray(branch, dtype=np.int64)] == 0, "suction", "pressure")

    @property
    def extremes(self) -> tuple[float, float]:
        """m of the blade's most upstream and most downstream point (the nose tip and the tail
        end): the first and the last branch end in m order."""
        return self.m_tip, self.m_tail

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


# --- 9. Geometry-driven standard grids (module docstring, section 9) -------------------

GRID_LEVELS = ("very_coarse", "coarse", "medium", "fine", "very_fine")
MEDIUM_DM_OVER_CHORD = 0.020                # target dm / chord of "medium"
LEVEL_REFINEMENT_RATIO = math.sqrt(2.0)     # target dm ratio between neighboring levels (two levels: a factor 2)
LEVEL_TARGET_DM_OVER_CHORD = MappingProxyType(
    {name: MEDIUM_DM_OVER_CHORD * LEVEL_REFINEMENT_RATIO ** (2 - k) for k, name in enumerate(GRID_LEVELS)})
CANDIDATE_BAND = 0.10                       # candidates: dm within +- 10 % of the level's target
TRANSITION_CLEARANCE = 0.10                 # f - 1, f (l, l + 1) at least this many dm clear of their window
CELL_ASPECT = 1.8                           # pitch * deta / dm (the project's earlier grids: 1.79 .. 1.83)
MIN_ETA_CELLS = 6                           # fewer eta cells than this reject a candidate
NOMINAL_EXTENSION_CHORDS = 1.0              # nominal upstream and downstream extension, chords
EXTENSION_TOLERANCE_CHORDS = 0.05           # |actual - nominal| extension allowed, chords
UNIFORM_SPACING_TOLERANCE = 1e-13           # chords: max |m[i + 1] - m[i] - dm| (measured 2.6e-16 .. 5.4e-16 c)

REJECT_LE = "invalid LE cut"
REJECT_TE = "invalid TE cut"
REJECT_BRANCH = "wrong branch identity"
REJECT_EXTENSION = "extension outside tolerance"
REJECT_ETA = "eta-resolution issue"
REJECT_OTHER = "other topology failure"
REJECTION_REASONS = (REJECT_LE, REJECT_TE, REJECT_BRANCH, REJECT_EXTENSION, REJECT_ETA, REJECT_OTHER)


@dataclass(frozen=True)
class PassageTransitions:
    """The blade ends along m, three kinds kept apart (module docstring, section 9), in the
    active length unit: the outline's extremes (nose tip, tail end), geometry.py's
    construction points (LE, TE) and the SS + PS-valid interval (m_start, m_end), the m
    range in which a station line crosses the reference blade exactly once on the suction
    and once on the pressure surface. coverage lists every m interval between branch (C1:
    segment) ends with its suction and pressure crossing counts."""

    m_tip: float
    m_le: float
    m_start: float
    m_end: float
    m_te: float
    m_tail: float
    chord: float
    coverage: tuple[tuple[float, float, int, int], ...]

    @property
    def length(self) -> float:
        """m_end - m_start: the SS + PS-valid interval that the lattice divides into cells."""
        return self.m_end - self.m_start

    @property
    def le_window(self) -> float:
        """m_start - m_tip: no station may lie in (m_tip, m_start], where a line meets the
        nose but not once on each surface."""
        return self.m_start - self.m_tip

    @property
    def te_window(self) -> float:
        """m_tail - m_end: no station may lie in [m_end, m_tail)."""
        return self.m_tail - self.m_end

    @property
    def le_face(self) -> float:
        """(m_tip + m_start) / 2: the middle of the LE window, where the lattice puts the cell
        face between f - 1 and f."""
        return 0.5 * (self.m_tip + self.m_start)

    @property
    def te_face(self) -> float:
        """(m_end + m_tail) / 2: the middle of the TE window, the cell face between l and l + 1."""
        return 0.5 * (self.m_end + self.m_tail)

    @property
    def lattice_length(self) -> float:
        """te_face - le_face: the m length the lattice divides into whole cells."""
        return self.te_face - self.le_face

    def valid_cut(self, m: float) -> bool:
        """Whether the line x = m crosses the reference blade exactly once on each surface."""
        return self.m_start < m < self.m_end


@dataclass(frozen=True)
class CandidateRecord:
    """One candidate spacing tried for a level: n_cells cells of dm across the SS + PS-valid
    interval; its rank in the search (1: nearest the target); why it was rejected."""

    n_cells: int
    dm_over_chord: float
    rank: int
    accepted: bool
    reason: str | None                      # one of REJECTION_REASONS, None when accepted
    detail: str


@dataclass(frozen=True)
class CandidateStatistics:
    """The search of one level: candidates tested and rejected, the accepted one's rank and
    the rejection reasons with their counts."""

    target_dm_over_chord: float
    band: tuple[float, float]               # dm / chord range searched
    records: tuple[CandidateRecord, ...]

    @property
    def tested(self) -> int:
        return len(self.records)

    @property
    def rejected(self) -> int:
        return sum(not r.accepted for r in self.records)

    @property
    def accepted_rank(self) -> int | None:
        return next((r.rank for r in self.records if r.accepted), None)

    @property
    def reasons(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in self.records:
            if r.reason is not None:
                out[r.reason] = out.get(r.reason, 0) + 1
        return out


@dataclass(frozen=True)
class StandardGridInfo:
    """What a standard grid is (module docstring, section 9): lengths in chords, points (m, y)
    in the active unit on the reference blade (the pressure-side wall node moved back one
    pitch), x / c along geometry.py's chord from the LE point."""

    level: str
    geometry_id: str
    target_dm_over_chord: float
    n_blade_cells: int                      # cells of dm across the SS + PS-valid interval
    dm: float
    dm_over_chord: float
    deta: float
    n_m: int
    n_eta: int
    total_nodes: int
    first_blade_i: int
    last_blade_i: int
    n_blade_stations: int
    m_first: float
    m_last: float
    first_ss_wall_point: tuple[float, float]
    first_ps_wall_point: tuple[float, float]
    last_ss_wall_point: tuple[float, float]
    last_ps_wall_point: tuple[float, float]
    first_ss_x_c: float
    first_ps_x_c: float
    last_ss_x_c: float
    last_ps_x_c: float
    le_gap_over_chord: float                # (m_first - m_start) / c: from the SS + PS-valid LE transition
    te_gap_over_chord: float                # (m_end - m_last) / c
    m_first_minus_m_le_over_chord: float    # against geometry.py's LE point
    m_te_minus_m_last_over_chord: float
    upstream_extension_chords: float        # m_le - m_in, chords
    downstream_extension_chords: float      # m_out - m_te, chords
    cell_aspect: float                      # pitch * deta / dm
    admissibility: str
    checks: tuple[str, ...]                 # the invariants verified on the built grid


@dataclass(frozen=True)
class StandardGrid:
    """A geometry-driven grid of one level: the Grid itself (what the solver takes), what it
    is, how it was found and how long it took (time.perf_counter seconds)."""

    level: str
    grid: Grid
    info: StandardGridInfo
    candidates: CandidateStatistics
    timing: Mapping[str, float]


@dataclass(frozen=True)
class GridPackage:
    """The standard grids of one geometry (all five levels, or those asked for), with the
    shared geometry interrogation and the family timing."""

    geometry_id: str
    transitions: PassageTransitions
    levels: tuple[str, ...]
    grids: Mapping[str, StandardGrid]
    timing: Mapping[str, float]             # interrogation, each level's search, total

    def __getitem__(self, level: str) -> StandardGrid:
        return self.grids[level]

    def __iter__(self):
        return iter([self.grids[level] for level in self.levels])

    def __len__(self) -> int:
        return len(self.levels)

    @property
    def candidates_tested(self) -> int:
        return sum(g.candidates.tested for g in self)

    @property
    def candidates_rejected(self) -> int:
        return sum(g.candidates.rejected for g in self)


class GridGenerationError(GridError):
    """No admissible standard grid for this geometry (module docstring, section 9). level: the
    level asked for ("family" for the ordering of a package, None for the geometry itself);
    geometry_id; candidates: every candidate tried, each with why it was rejected; reasons:
    the rejection reasons with their counts. A structured failure that an optimizer can
    catch and penalize; no grid is returned."""

    def __init__(self, message: str, *, level: str | None, geometry_id: str | None,
                 candidates: tuple[CandidateRecord, ...] = (), reasons: Mapping[str, int] | None = None) -> None:
        super().__init__(message)
        self.level, self.geometry_id = level, geometry_id
        self.candidates = tuple(candidates)
        self.reasons = dict(reasons or {})


class _Rejected(Exception):
    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(detail)
        self.reason, self.detail = reason, detail


@dataclass(frozen=True)
class _Interrogation:
    cascade: Cascade
    row: _Row
    template: PassageBoundaries
    transitions: PassageTransitions
    geometry_id: str
    seconds: float


def geometry_identifier(cascade: Cascade) -> str:
    """A short fingerprint of the blade row a grid is built for: sha256 of the active
    surfaces, the C2 curves, pitch, chord, stagger and blade angles (16 hex digits)."""
    if not isinstance(cascade, Cascade):
        raise TypeError(f"geometry_identifier takes a Cascade, got {type(cascade).__name__}")
    a = cascade.active
    digest = hashlib.sha256()
    for array in (a.suction_surface, a.pressure_surface, a.periodic_translation,
                  (a.chord, a.pitch, cascade.kappa_le, cascade.kappa_te, cascade.stagger)):
        digest.update(np.ascontiguousarray(array, dtype=float).tobytes())
    curves = cascade.active_curves
    if curves is not None:
        for curve in (curves.suction, curves.pressure):
            digest.update(np.ascontiguousarray(curve.knots, dtype=float).tobytes())
            digest.update(np.ascontiguousarray(curve.control_points, dtype=float).tobytes())
            digest.update(str(curve.degree).encode())
    return digest.hexdigest()[:16]


def passage_transitions(cascade: Cascade) -> PassageTransitions:
    """The blade ends of cascade along m (module docstring, section 9): nose tip, LE point,
    SS + PS-valid interval, TE point, tail end. Read-only; GridGenerationError when no
    SS + PS-valid interval holds the middle of the chord or the passage is closed."""
    return _interrogate(cascade).transitions


def build_grid(cascade: Cascade, level: str = "medium", *, inlet_angle: float, exit_angle: float | None = None,
               extension_chords: tuple[float, float] = (NOMINAL_EXTENSION_CHORDS, NOMINAL_EXTENSION_CHORDS)
               ) -> StandardGrid:
    """The geometry-driven standard grid of one level (module docstring, section 9): the
    candidate spacing nearest the level's target that passes every geometry and topology
    condition, built once. Angles in radians as for make_grid (exit_angle None: kappa_te);
    extension_chords the nominal (upstream, downstream) lengths in chords. Raises
    GridGenerationError when no candidate of the level is admissible (or the passage is
    closed); never returns a malformed grid. timing["total"] is the whole call."""
    started = time.perf_counter()
    level = _level_name(level)
    interrogation = _interrogate(cascade)
    out = _build_level(interrogation, level, inlet_angle, exit_angle, extension_chords)
    timing = dict(out.timing)
    timing["interrogation"] = interrogation.seconds
    timing["total"] = time.perf_counter() - started
    return replace(out, timing=MappingProxyType(timing))


def build_grid_family(cascade: Cascade, *, inlet_angle: float, exit_angle: float | None = None,
                      levels: tuple[str, ...] = GRID_LEVELS,
                      extension_chords: tuple[float, float] = (NOMINAL_EXTENSION_CHORDS, NOMINAL_EXTENSION_CHORDS)
                      ) -> GridPackage:
    """The standard grids of levels (all five by default) of one geometry, from one shared
    geometry interrogation; each is the grid build_grid gives for its level. Raises
    GridGenerationError for the first level without an admissible grid, and (level
    "family") when dm and deta do not both fall strictly from level to level."""
    started = time.perf_counter()
    names = tuple(_level_name(level) for level in levels)
    if not names or len(set(names)) != len(names):
        raise ValueError(f"levels must name each level once, got {levels!r}")
    names = tuple(sorted(names, key=GRID_LEVELS.index))
    interrogation = _interrogate(cascade)
    grids, timing = {}, {"interrogation": interrogation.seconds}
    for name in names:
        grids[name] = _build_level(interrogation, name, inlet_angle, exit_angle, extension_chords)
        timing[name] = grids[name].timing["search"]
    for coarse, fine in zip(names[:-1], names[1:]):
        a, b = grids[coarse].info, grids[fine].info
        if not (a.dm > b.dm and a.deta > b.deta):
            raise GridGenerationError(f"levels {coarse} and {fine} are not a refinement: dm / c {a.dm_over_chord:.6g} -> "
                                      f"{b.dm_over_chord:.6g}, deta {a.deta:.6g} -> {b.deta:.6g}", level="family",
                                      geometry_id=interrogation.geometry_id)
    timing["total"] = time.perf_counter() - started
    return GridPackage(geometry_id=interrogation.geometry_id, transitions=interrogation.transitions, levels=names,
                       grids=MappingProxyType(grids), timing=MappingProxyType(timing))


def validate_standard_grid(grid: Grid, transitions: PassageTransitions | None = None) -> tuple[str, ...]:
    """The invariants of a standard grid (module docstring, section 9), checked on the built
    Grid itself: the names of the checks passed, or GridError naming the first that fails.
    transitions None: read from the grid's own blade."""
    if not isinstance(grid, Grid):
        raise TypeError(f"validate_standard_grid takes a Grid, got {type(grid).__name__}")
    row = _row_of(grid.boundaries)
    if transitions is None:
        transitions = _transitions(grid.boundaries, row)
    try:
        return _standard_checks(grid, row, transitions, row.cut(grid.m))
    except _Rejected as error:
        raise GridError(f"{error.reason}: {error.detail}") from None


def _level_name(level: str) -> str:
    if not isinstance(level, str) or level not in GRID_LEVELS:
        raise ValueError(f"level must be one of {', '.join(GRID_LEVELS)}, got {level!r}")
    return level


def _interrogate(cascade: Cascade) -> _Interrogation:
    """What every level shares: the blade row, its transitions and the passage check."""
    started = time.perf_counter()
    if not isinstance(cascade, Cascade):
        raise TypeError(f"the standard grids take a Cascade, got {type(cascade).__name__}")
    geometry_id = geometry_identifier(cascade)
    template = make_passage_boundaries(cascade, GridSettings(n_passage=1, n_eta=2), 0.0, 0.0)
    row = _row_of(template)
    transitions = _transitions(template, row, geometry_id)
    try:
        _check_passage(cascade, template, row)
    except GridError as error:
        raise GridGenerationError(f"geometry {geometry_id} could not produce an admissible grid at any level: {error}",
                                  level=None, geometry_id=geometry_id, reasons={REJECT_OTHER: 1}) from error
    return _Interrogation(cascade=cascade, row=row, template=template, transitions=transitions, geometry_id=geometry_id,
                          seconds=time.perf_counter() - started)


def _pieces_of(row: _Row) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
    """m range [lo, hi) of every monotone piece of the outline (C2 branch, C1 segment) and
    whether it lies on the suction surface."""
    if isinstance(row, _CurveRow):
        return row.lo_m, row.hi_m, row.curve_of == 0
    x0, x1 = row.outline[:-1, 0], row.outline[1:, 0]
    return np.minimum(x0, x1), np.maximum(x0, x1), row.on_suction(np.arange(len(x0)))


def _transitions(b: PassageBoundaries, row: _Row, geometry_id: str | None = None) -> PassageTransitions:
    """PassageTransitions from the pieces' m ranges: between consecutive piece ends the
    crossings per surface are constant (half-open ranges, as the cuts count them)."""
    lo, hi, suction = _pieces_of(row)
    ends = np.unique(np.concatenate((lo, hi)))
    middle = 0.5 * (ends[:-1] + ends[1:])
    on = (lo[None, :] <= middle[:, None]) & (middle[:, None] < hi[None, :])
    ss = np.count_nonzero(on & suction[None, :], axis=1)
    ps = np.count_nonzero(on & ~suction[None, :], axis=1)
    coverage = tuple((float(a), float(z), int(s), int(p)) for a, z, s, p in zip(ends[:-1], ends[1:], ss, ps))
    valid = (ss == 1) & (ps == 1)
    centre = 0.5 * (b.m_le + b.m_te)
    k = int(np.searchsorted(ends, centre, side="right")) - 1
    if not (0 <= k < len(valid) and valid[k]):
        found = coverage[k] if 0 <= k < len(coverage) else None
        raise GridGenerationError(f"geometry {geometry_id} could not produce an admissible grid at any level: the line "
                                  f"through the middle of the chord does not cross the blade once on the suction and once "
                                  f"on the pressure surface ({found}): no SS + PS-valid passage interval",
                                  level=None, geometry_id=geometry_id, reasons={REJECT_BRANCH: 1})
    first = last = k
    while first > 0 and valid[first - 1]:
        first -= 1
    while last < len(valid) - 1 and valid[last + 1]:
        last += 1
    tip, tail = row.extremes
    return PassageTransitions(m_tip=float(tip), m_le=b.m_le, m_start=float(ends[first]), m_end=float(ends[last + 1]),
                              m_te=b.m_te, m_tail=float(tail), chord=b.chord, coverage=coverage)


def _build_level(it: _Interrogation, level: str, inlet_angle: float, exit_angle: float | None,
                 extension_chords: tuple[float, float]) -> StandardGrid:
    """The candidate search of one level (module docstring, section 9)."""
    started = time.perf_counter()
    tr = it.transitions
    up, down = (_positive_number(value, name) for value, name in
                zip(extension_chords, ("upstream extension_chords", "downstream extension_chords")))
    target = LEVEL_TARGET_DM_OVER_CHORD[level] * tr.chord
    low, high = target * (1.0 - CANDIDATE_BAND), target * (1.0 + CANDIDATE_BAND)
    length = tr.lattice_length
    counts = range(max(1, math.ceil(length / high)), math.floor(length / low) + 1)
    ranked = sorted(counts, key=lambda n: (abs(math.log(length / n / target)), n))
    records, clock = [], {"intersection": 0.0, "construction": 0.0, "validation": 0.0}
    band = (low / tr.chord, high / tr.chord)
    for rank, n_cells in enumerate(ranked, start=1):
        try:
            grid, info = _candidate(it, level, n_cells, target, inlet_angle, exit_angle, up, down, clock)
        except _Rejected as error:
            records.append(CandidateRecord(n_cells=n_cells, dm_over_chord=length / n_cells / tr.chord, rank=rank,
                                           accepted=False, reason=error.reason, detail=error.detail))
            continue
        records.append(CandidateRecord(n_cells=n_cells, dm_over_chord=info.dm_over_chord, rank=rank, accepted=True,
                                       reason=None, detail="accepted"))
        stats = CandidateStatistics(target_dm_over_chord=target / tr.chord, band=band, records=tuple(records))
        clock["search"] = time.perf_counter() - started
        return StandardGrid(level=level, grid=grid, info=info, candidates=stats, timing=MappingProxyType(dict(clock)))
    stats = CandidateStatistics(target_dm_over_chord=target / tr.chord, band=band, records=tuple(records))
    tried = "; ".join(f"{r.n_cells} cells (dm/c {r.dm_over_chord:.5f}): {r.reason}, {r.detail}" for r in records) or "none"
    raise GridGenerationError(f"geometry {it.geometry_id} could not produce an admissible {level} grid: target dm/c "
                              f"{target / tr.chord:.5f}, band {band[0]:.5f} .. {band[1]:.5f}, {len(records)} candidate(s) "
                              f"tried: {tried}", level=level, geometry_id=it.geometry_id, candidates=tuple(records),
                              reasons=stats.reasons)


def _candidate(it: _Interrogation, level: str, n_cells: int, target: float, inlet_angle: float, exit_angle: float | None,
               up: float, down: float, clock: dict) -> tuple[Grid, StandardGridInfo]:
    """One candidate: n_cells cells between the LE and TE faces (the middles of the two
    windows), stations at the cell centres, whole cells out to about the nominal extensions;
    _Rejected unless every condition holds."""
    tr, row, chord = it.transitions, it.row, it.transitions.chord
    dm = tr.lattice_length / n_cells
    m_first, m_last = tr.le_face + 0.5 * dm, tr.te_face - 0.5 * dm
    for reason, edge, window in ((REJECT_LE, "LE", tr.le_window), (REJECT_TE, "TE", tr.te_window)):
        clearance = 0.5 * (dm - window)          # of the transition stations from the window, on both sides
        if not clearance >= TRANSITION_CLEARANCE * dm:
            raise _Rejected(reason, f"the {edge} window {window / chord:.3e} c leaves the stations beside it "
                                    f"{clearance / chord:.3e} c of clearance, below {TRANSITION_CLEARANCE:g} dm "
                                    f"({TRANSITION_CLEARANCE * dm / chord:.3e} c): dm must exceed the window "
                                    f"by {2 * TRANSITION_CLEARANCE:g} dm")
    k_up = round((m_first - (tr.m_le - up * chord)) / dm)
    k_down = round(((tr.m_te + down * chord) - m_last) / dm)
    upstream = (tr.m_le - (m_first - k_up * dm)) / chord
    downstream = ((m_last + k_down * dm) - tr.m_te) / chord
    for name, actual, nominal in (("upstream", upstream, up), ("downstream", downstream, down)):
        if not abs(actual - nominal) <= EXTENSION_TOLERANCE_CHORDS:
            raise _Rejected(REJECT_EXTENSION, f"{name} extension {actual:.4f} c, nominal {nominal:g} c, tolerance "
                                              f"{EXTENSION_TOLERANCE_CHORDS:g} c")
    if k_up < 1 or k_down < 1:
        raise _Rejected(REJECT_EXTENSION, "an extension holds no whole cell")
    eta_cells = round(it.template.pitch / (CELL_ASPECT * dm))
    if eta_cells < MIN_ETA_CELLS:
        raise _Rejected(REJECT_ETA, f"{eta_cells} eta cells (pitch / ({CELL_ASPECT:g} dm)), fewer than {MIN_ETA_CELLS}")
    first, last = k_up, k_up + n_cells - 1

    started = time.perf_counter()
    settings = GridSettings(n_passage=n_cells, n_eta=eta_cells + 1, upstream_extension_chords=upstream,
                            downstream_extension_chords=downstream)
    b = make_passage_boundaries(it.cascade, settings, inlet_angle, exit_angle)
    m = np.linspace(b.m_in, b.m_out, k_up + n_cells + k_down)
    clock["construction"] += time.perf_counter() - started
    started = time.perf_counter()
    cut = row.cut(m)
    clock["intersection"] += time.perf_counter() - started
    _classify(row, it.transitions, cut, first, last)
    started = time.perf_counter()
    try:
        grid = _grid_on_stations(b, row, m, eta_cells + 1, cut)
    except GridError as error:
        raise _Rejected(REJECT_OTHER, str(error).splitlines()[0]) from None
    finally:
        clock["construction"] += time.perf_counter() - started
    if (grid.first_blade_i, grid.last_blade_i) != (first, last):
        raise _Rejected(REJECT_OTHER, f"blade run {grid.first_blade_i} .. {grid.last_blade_i}, expected {first} .. {last}")
    started = time.perf_counter()
    try:
        checks = _standard_checks(grid, row, tr, cut)
    finally:
        clock["validation"] += time.perf_counter() - started
    return grid, _info(it, level, grid, n_cells, target, checks)


def _crossings(row: _Row, cut: _Cut) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Suction and pressure crossings of every station line of cut."""
    present = np.arange(cut.segment.shape[1])[None, :] < cut.count[:, None]
    names = row.surface_of(cut.segment)
    return (np.count_nonzero(present & (names == "suction"), axis=1),
            np.count_nonzero(present & (names == "pressure"), axis=1))


def _classify(row: _Row, tr: PassageTransitions, cut: _Cut, first: int, last: int) -> None:
    """_Rejected unless no station before first or after last meets the blade at all and every
    station first .. last crosses it exactly once on each surface, off the LE and TE points."""
    count = cut.count
    ahead = np.flatnonzero(count[:first] > 0)
    if ahead.size:
        i = int(ahead[-1])
        raise _Rejected(REJECT_LE, f"station {i} ahead of the first SS + PS station meets the blade ({int(count[i])} "
                                   f"crossings) at (m - m_start)/c = {(cut.m[i] - tr.m_start) / tr.chord:+.3e}: the LE "
                                   f"window is {tr.le_window / tr.chord:.3e} c")
    behind = np.flatnonzero(count[last + 1:] > 0)
    if behind.size:
        i = last + 1 + int(behind[0])
        raise _Rejected(REJECT_TE, f"station {i} behind the last SS + PS station meets the blade ({int(count[i])} "
                                   f"crossings) at (m - m_end)/c = {(cut.m[i] - tr.m_end) / tr.chord:+.3e}: the TE window "
                                   f"is {tr.te_window / tr.chord:.3e} c")
    ss, ps = _crossings(row, cut)
    blade = slice(first, last + 1)
    bad = np.flatnonzero(~((count[blade] == 2) & (ss[blade] == 1) & (ps[blade] == 1)))
    if bad.size:
        i = first + int(bad[0])
        raise _Rejected(REJECT_BRANCH, f"blade station {i} crosses the blade {int(count[i])} times ({int(ss[i])} suction, "
                                       f"{int(ps[i])} pressure)")
    m = cut.m[blade]
    near = np.flatnonzero((np.abs(m - tr.m_le) <= row.touch) | (np.abs(m - tr.m_te) <= row.touch))
    if near.size:
        raise _Rejected(REJECT_BRANCH, f"blade station {first + int(near[0])} passes through the LE or TE point")


def _standard_checks(grid: Grid, row: _Row, tr: PassageTransitions, cut: _Cut) -> tuple[str, ...]:
    """The invariants of section 9 on a built Grid; _Rejected at the first that fails."""
    chord, m, eta = grid.chord, grid.m, grid.eta
    f, l = grid.first_blade_i, grid.last_blade_i
    checks = []

    def need(ok: bool, name: str, reason: str, detail: str) -> None:
        if not ok:
            raise _Rejected(reason, f"{name}: {detail}")
        checks.append(name)

    steps = np.diff(m)
    need(bool(np.all(steps > 0.0)), "distinct stations", REJECT_OTHER, "two stations coincide or run backward")
    spread = float(max(np.abs(steps - grid.dm)))                 # builtin max: no min/max attribute in grid.py
    need(spread <= UNIFORM_SPACING_TOLERANCE * chord, "uniform dm", REJECT_OTHER, f"spacing varies by {spread / chord:.3g} c")
    eta_spread = float(max(np.abs(np.diff(eta) - grid.deta)))
    need(bool(eta[0] == 0.0 and eta[-1] == 1.0 and eta_spread <= UNIFORM_SPACING_TOLERANCE), "uniform deta", REJECT_OTHER,
         f"eta spacing varies by {eta_spread:.3g}")
    need(1 <= f <= l <= len(m) - 2, "f <= l inside the far stations", REJECT_OTHER, f"f = {f}, l = {l}, n_m = {len(m)}")
    need(not tr.valid_cut(m[f - 1]) and int(cut.count[f - 1]) == 0, "f - 1 outside the blade", REJECT_LE,
         f"station {f - 1} meets the blade")
    need(not tr.valid_cut(m[l + 1]) and int(cut.count[l + 1]) == 0, "l + 1 outside the blade", REJECT_TE,
         f"station {l + 1} meets the blade")
    need(all(tr.valid_cut(value) for value in m[f:l + 1]), "every blade station inside the SS + PS-valid interval",
         REJECT_BRANCH, "a blade station lies outside it")
    ss, ps = _crossings(row, cut)
    need(bool(np.all((cut.count[f:l + 1] == 2) & (ss[f:l + 1] == 1) & (ps[f:l + 1] == 1))),
         "one SS + one PS crossing per blade station", REJECT_BRANCH, "a blade station has other crossings")
    identity = grid.wall_identity
    lower, upper = set(identity.lower.surface.tolist()), set(identity.upper.surface.tolist())
    need(len(lower) == 1 and len(upper) == 1 and lower != upper, "wall identity constant", REJECT_BRANCH,
         f"lower wall on {sorted(lower)}, upper wall on {sorted(upper)}")
    need(not np.any(identity.lower.hidden) and not np.any(identity.upper.hidden), "no hidden crossings", REJECT_BRANCH,
         "a wall node hides further crossings of its contour")
    need(bool(np.all(np.isfinite(grid.nodes))), "finite nodes", REJECT_OTHER, "a node is not finite")
    width = grid.nodes[:, -1, 1] - grid.nodes[:, 0, 1]
    need(bool(np.all(width > row.touch)), "positive width", REJECT_OTHER, f"width {float(min(width)) / chord:.3g} c")
    from turbolenza.b2b.metrics import build_metrics        # metrics imports grid: here, not at the top
    try:
        metrics = build_metrics(grid)
    except Exception as error:                                     # any refusal of metrics.py rejects the candidate
        raise _Rejected(REJECT_OTHER, f"metrics: {type(error).__name__}: {error}") from None
    need(bool(np.all(np.asarray(metrics.S) > 0.0)), "metrics with positive S", REJECT_OTHER, "S is not positive")
    return tuple(checks)


def _info(it: _Interrogation, level: str, grid: Grid, n_cells: int, target: float,
          checks: tuple[str, ...]) -> StandardGridInfo:
    tr, chord, b = it.transitions, it.transitions.chord, grid.boundaries
    f, l = grid.first_blade_i, grid.last_blade_i
    shift = np.asarray(b.periodic_translation)
    identity = grid.wall_identity
    le, te = np.asarray(b.lower_blade_boundary[0]), np.asarray(b.lower_blade_boundary[-1])
    axis = te - le

    def points(i: int) -> tuple[tuple[float, float], tuple[float, float]]:
        lower, upper = grid.nodes[i, 0], grid.nodes[i, -1] - shift
        ss, ps = (lower, upper) if identity.lower.surface[i - f] == "suction" else (upper, lower)
        return (float(ss[0]), float(ss[1])), (float(ps[0]), float(ps[1]))

    def x_c(point: tuple[float, float]) -> float:
        return float(np.dot(np.asarray(point) - le, axis) / np.dot(axis, axis))

    first_ss, first_ps = points(f)
    last_ss, last_ps = points(l)
    return StandardGridInfo(
        level=level, geometry_id=it.geometry_id, target_dm_over_chord=target / chord, n_blade_cells=n_cells,
        dm=grid.dm, dm_over_chord=grid.dm / chord, deta=grid.deta, n_m=len(grid.m), n_eta=len(grid.eta),
        total_nodes=len(grid.m) * len(grid.eta), first_blade_i=f, last_blade_i=l, n_blade_stations=l - f + 1,
        m_first=float(grid.m[f]), m_last=float(grid.m[l]), first_ss_wall_point=first_ss, first_ps_wall_point=first_ps,
        last_ss_wall_point=last_ss, last_ps_wall_point=last_ps, first_ss_x_c=x_c(first_ss), first_ps_x_c=x_c(first_ps),
        last_ss_x_c=x_c(last_ss), last_ps_x_c=x_c(last_ps),
        le_gap_over_chord=(float(grid.m[f]) - tr.m_start) / chord, te_gap_over_chord=(tr.m_end - float(grid.m[l])) / chord,
        m_first_minus_m_le_over_chord=(float(grid.m[f]) - tr.m_le) / chord,
        m_te_minus_m_last_over_chord=(tr.m_te - float(grid.m[l])) / chord,
        upstream_extension_chords=(tr.m_le - float(grid.m[0])) / chord,
        downstream_extension_chords=(float(grid.m[-1]) - tr.m_te) / chord,
        cell_aspect=b.pitch * grid.deta / grid.dm, admissibility="admissible", checks=checks)
