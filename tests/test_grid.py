"""
Tests for turbolenza.b2b.grid: make_passage_boundaries (the blade walls of one
passage and its inlet and exit extensions), make_grid (stations, the blade-passage
topology from the cuts of the station lines with the blade row, wall values and nodes) and
rebuild_exit (the exit extension turned to a new discharge angle).

The cascades come from geometry.make_cascade, as they will in the solver. The reference
values are independent of grid.py: the Cascade fields themselves, scipy.integrate.quad of
dy/dm = tan(beta) for the side lines, the cut of each station line with the closed blade
outline written out here segment by segment, blade metal and distances measured on that
outline and its periodic copies, the smooth walls written out here too (a parametric PCHIP
through the outline's vertices) where the two differ in a passage the blades all but close,
the same blade built with 1601 camber stations, and finite differences of the samples.
The wall identity (Grid.wall_identity) is checked against the outline segment each wall
node lies on, the named polyline nearest to it, and the m of geometry.py's LE and TE points.

C2 cascades (grid.py section 7) are checked against their curves read here, not through
grid.py: C2Blade finds the monotone pieces of m(u) from dense samples and scipy's bounded
minimize_scalar, and every crossing of a station line by brentq; the wall nodes, their u
and curve, the topology, metal on the eta lines and the walk along each contour are
compared with it. The pre-hand-off C2 grid is the same cascade read from its sampled arrays
(sampled_reading), and C1 is locked by fingerprints taken before the hand-off.

Admissible n_passage (grid.py section 8): the first and last blade station of the C2
65-(12A10)10 at stagger 13 to 53 deg are classified by C2Blade's own crossings (which curve
each lies on), against grid.wall_station_topology, Grid.wall_identity, the errors of
make_grid(..., require_ss_ps=True) and validate_uniform_grid_topology, find_valid_n_passage
over a sweep, and the refinement ceiling computed here from the lattice formula and
C2Blade's nose tip and tail end. The admissible, inadmissible and ceiling values were
measured by scripts/audit_uniform_grid_validity.py (n_passage 20 to 800).

Not tested here because not implemented: node_index and node_theta, and every metric,
slope or flow quantity. The straight segments between nodes of neighboring stations are
not grid edges (grid.py, section 6) and are not tested for blade metal.
"""

from __future__ import annotations

import ast
import math
import re
import sys
from dataclasses import fields, replace
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq, minimize_scalar

from turbolenza.b2b import grid as gr
from turbolenza.b2b.config import GridSettings
from turbolenza.b2b.geometry import Cascade, CascadeSpec, make_cascade

# A Dunavant-like cascade, a negative-stagger one with a sharp TE, and the first again with
# a physical chord. Test inputs only: grid.py knows none of these numbers.
SPEC = CascadeSpec(blade_type="NACA65", solidity=1.0, stagger_deg=33.0, cl0=1.2, t_max_over_c=0.10,
                   surface_continuity="C1")
OTHER = CascadeSpec(blade_type="NACA65", solidity=1.5, stagger_deg=-20.0, cl0=0.4, t_max_over_c=0.06,
                    profile_variant="emery_base", surface_continuity="C1")
DIMENSIONAL = replace(SPEC, physical_chord=0.05)
CASES = (SPEC, OTHER, DIMENSIONAL)
# A thin nose whose pressure side folds back in m 0.0024 to 0.0027 c behind the LE.
THIN = CascadeSpec(blade_type="NACA65", solidity=1.0, stagger_deg=50.0, cl0=1.6, t_max_over_c=0.04,
                   surface_continuity="C1")
# A nose that reaches 0.019 c ahead of the LE, more than a cell at 31 passage cells.
STEEP = CascadeSpec(blade_type="NACA65", solidity=1.0, stagger_deg=60.0, cl0=1.6, t_max_over_c=0.10,
                    surface_continuity="C1")
# A Kovach TE whose pressure side passes m_te by 0.004 c before it closes on the TE point.
LONG_TAIL = CascadeSpec(blade_type="NACA65", solidity=1.0, stagger_deg=60.0, cl0=0.0, t_max_over_c=0.04,
                        surface_continuity="C1")
EDGE_CASES = (SPEC, OTHER, DIMENSIONAL, THIN, STEEP, LONG_TAIL)
# A Kovach TE whose suction side passes m_te, by 0.0040 c: negative stagger.
SUCTION_TAIL = CascadeSpec(blade_type="NACA65", solidity=0.8, stagger_deg=-30.0, cl0=1.2, t_max_over_c=0.10,
                           surface_continuity="C1")
# Thin sharp TEs at 0 and 60 deg stagger.
SHARP_0 = CascadeSpec(blade_type="NACA65", solidity=1.0, stagger_deg=0.0, cl0=0.4, t_max_over_c=0.04,
                      profile_variant="emery_base", surface_continuity="C1")
SHARP_60 = replace(SHARP_0, stagger_deg=60.0)
SHARP_60_FLAT = replace(SHARP_60, cl0=0.0)
# Blades at the overlap limit of geometry.py: NEAR_CLOSED 2e-6 below the largest solidity it
# accepts (5.931831), CLOSED at the largest (to 1e-9).
NEAR_CLOSED = CascadeSpec(blade_type="NACA65", solidity=5.93182, stagger_deg=50.0, cl0=1.6, t_max_over_c=0.10,
                          profile_variant="emery_base", surface_continuity="C1")
CLOSED = CascadeSpec(blade_type="NACA65", solidity=6.4246233818121254, stagger_deg=50.0, cl0=0.4, t_max_over_c=0.10,
                     profile_variant="emery_base", surface_continuity="C1")
# The same blade as SPEC with 16 times the camber stations, as a reference surface.
FINE = replace(SPEC, n_camber_stations=1601)
INLET_ANGLES_DEG = (45.0, 20.0, -10.0, 60.0)
EXIT_TURNS_DEG = (2.0, 5.0, -8.0, 25.0)        # exit angle = kappa_te + this
SETTINGS = GridSettings(n_passage=31, n_eta=21)
INVALID_ANGLES = (math.pi / 2, -math.pi / 2, math.nan, math.inf, 45.0, True)
WALL_TOLERANCE = 5e-5           # chords: smooth wall nodes against the outline polyline (largest seen 3.5e-5)


@lru_cache(maxsize=None)
def cascade_of(spec: CascadeSpec) -> Cascade:
    return make_cascade(spec)


def exit_angle_of(spec: CascadeSpec, turn_deg: float | None) -> float | None:
    return None if turn_deg is None else cascade_of(spec).kappa_te + math.radians(turn_deg)


@lru_cache(maxsize=None)
def boundaries(spec: CascadeSpec, inlet_angle_deg: float = 45.0, settings: GridSettings = SETTINGS,
               exit_turn_deg: float | None = None) -> gr.PassageBoundaries:
    return gr.make_passage_boundaries(cascade_of(spec), settings, math.radians(inlet_angle_deg),
                                      exit_angle_of(spec, exit_turn_deg))


@lru_cache(maxsize=None)
def grid_of(spec: CascadeSpec, inlet_angle_deg: float = 45.0, settings: GridSettings = SETTINGS,
            exit_turn_deg: float | None = None) -> gr.Grid:
    return gr.make_grid(cascade_of(spec), settings, math.radians(inlet_angle_deg), exit_angle_of(spec, exit_turn_deg))


def reference_line_y(m: np.ndarray, junction: np.ndarray, junction_angle: float, far_m: float,
                     far_angle: float) -> np.ndarray:
    """y of a lower side line by adaptive quadrature of tan(beta(m)) from the junction point,
    beta linear in m from junction_angle at the junction to far_angle at far_m:
    independent of the closed form in grid.py."""
    junction_m = float(junction[0])

    def slope(mu: float) -> float:
        weight = (mu - far_m) / (junction_m - far_m)
        return math.tan(weight * junction_angle + (1.0 - weight) * far_angle)

    return np.array([junction[1] + quad(slope, junction_m, float(x), epsabs=1e-14, epsrel=1e-13, limit=200)[0]
                     for x in m])


def crossings(wall: np.ndarray, m: float) -> int:
    """How many segments of a wall polyline meet the line x = m (half-open in x, so a
    vertex on the line is counted once where the wall passes through it)."""
    x0, x1 = wall[:-1, 0], wall[1:, 0]
    return int(np.count_nonzero((np.minimum(x0, x1) <= m) & (m < np.maximum(x0, x1))))


def blade_walls(spec: CascadeSpec) -> tuple[np.ndarray, np.ndarray]:
    a = cascade_of(spec).active
    return a.suction_surface, a.pressure_surface + a.periodic_translation


def stations_with_one_at(spec: CascadeSpec, m_target: float, settings: GridSettings = SETTINGS) -> GridSettings:
    """settings with both extensions moved by less than half a cell so that the stations are
    spaced exactly (m_te - m_le) / n_passage and one of them falls on m_target: the stations
    are linspace(m_in, m_out, N + 1) with N = round((m_out - m_in) / ((m_te - m_le) /
    n_passage)), module docstring section 2, here a whole number to round-off."""
    a = cascade_of(spec).active
    m_le, m_te = a.leading_edge_point[0], a.trailing_edge_point[0]
    cell = (m_te - m_le) / settings.n_passage
    m_in = m_target - round((m_target - m_le + settings.upstream_extension_chords * a.chord) / cell) * cell
    m_out = m_target + round((m_te + settings.downstream_extension_chords * a.chord - m_target) / cell) * cell
    return replace(settings, upstream_extension_chords=(m_le - m_in) / a.chord,
                   downstream_extension_chords=(m_out - m_te) / a.chord)


def nudged(settings: GridSettings, ulps: int) -> GridSettings:
    """settings with upstream_extension_chords moved by ulps units in the last place, which
    moves the stations by an ulp of m or so each."""
    up = settings.upstream_extension_chords
    for _ in range(abs(ulps)):
        up = float(np.nextafter(up, math.inf if ulps > 0 else -math.inf))
    return replace(settings, upstream_extension_chords=up)


def outline(spec: CascadeSpec) -> np.ndarray:
    """The reference blade's closed outline: suction surface TE -> LE, then pressure surface
    LE -> TE (written here, not taken from grid.py or geometry.closed_loop)."""
    a = cascade_of(spec).active
    return np.vstack((a.suction_surface[::-1], a.pressure_surface[1:]))


def blade_extent(spec: CascadeSpec) -> tuple[float, float]:
    """The blade's most upstream and most downstream m (test side only)."""
    x = outline(spec)[:, 0]
    return float(x.min()), float(x.max())


def line_cut(spec: CascadeSpec, m: float) -> tuple[np.ndarray, np.ndarray]:
    """Where the line x = m crosses the reference outline: heights ascending, and the outline
    segment of each (half-open in x)."""
    loop = outline(spec)
    x0, x1 = loop[:-1, 0], loop[1:, 0]
    k = np.flatnonzero((np.minimum(x0, x1) <= m) & (m < np.maximum(x0, x1)))
    y = loop[k, 1] + (m - x0[k]) / (x1[k] - x0[k]) * (loop[k + 1, 1] - loop[k, 1])
    order = np.argsort(y)
    return y[order], k[order]


def on_suction_polyline(spec: CascadeSpec, segment: int) -> bool:
    """Whether outline segment `segment` belongs to the suction polyline (it comes first)."""
    return segment < len(cascade_of(spec).active.suction_surface) - 1


TOUCH = 1e-12           # chords: metal no longer than this is a touch (a line through a sharp TE), not a cut


def metal_on(spec: CascadeSpec, m: float, y0: float, y1: float) -> list[tuple[float, float]]:
    """The pieces of blade metal on the segment x = m, y0 <= y <= y1: the closed outline of
    the reference blade and its copies -2 to +3 pitches away, cut by the line x = m,
    crossings paired by the even-odd rule; pieces no longer than TOUCH chords are touches
    and left out."""
    a = cascade_of(spec).active
    loop0 = outline(spec)
    pieces = []
    for k in (-2, -1, 0, 1, 2, 3):
        loop = loop0 + k * a.periodic_translation
        x0, x1 = loop[:-1, 0], loop[1:, 0]
        cut = np.flatnonzero((np.minimum(x0, x1) <= m) & (m < np.maximum(x0, x1)))
        ys = np.sort(loop[cut, 1] + (m - x0[cut]) / (x1[cut] - x0[cut]) * (loop[cut + 1, 1] - loop[cut, 1]))
        pieces += [(lo, hi) for lo, hi in zip(ys[::2], ys[1::2])
                   if lo < y1 and hi > y0 and hi - lo > TOUCH * a.chord]
    return pieces


def in_metal(spec: CascadeSpec, points: np.ndarray) -> np.ndarray:
    """Whether each point lies inside a blade of the row (even-odd ray casting against the
    outline and its copies -2 to +3 pitches away)."""
    a = cascade_of(spec).active
    p = np.asarray(points, dtype=float).reshape(-1, 2)
    inside = np.zeros(len(p), dtype=bool)
    for k in (-2, -1, 0, 1, 2, 3):
        loop = outline(spec) + k * a.periodic_translation
        x0, y0, x1, y1 = loop[:-1, 0], loop[:-1, 1], loop[1:, 0], loop[1:, 1]
        for chunk in np.array_split(np.arange(len(p)), max(1, len(p) // 2000)):
            px, py = p[chunk, 0:1], p[chunk, 1:2]
            straddle = (y0 > py) != (y1 > py)
            x_cross = x0 + (py - y0) * (x1 - x0) / np.where(y1 != y0, y1 - y0, 1.0)
            inside[chunk] |= np.count_nonzero(straddle & (px < x_cross), axis=1) % 2 == 1
    return inside


@lru_cache(maxsize=None)
def smooth_outline(spec: CascadeSpec) -> tuple[np.ndarray, PchipInterpolator, PchipInterpolator]:
    """The smooth walls of grid.py section 4, written here: a parametric PCHIP x(t), y(t)
    through the reference outline's vertices, t the cumulative chord length."""
    loop = outline(spec)
    t = np.concatenate(([0.0], np.cumsum(np.hypot(*np.diff(loop, axis=0).T))))
    return t, PchipInterpolator(t, loop[:, 0]), PchipInterpolator(t, loop[:, 1])


def smooth_cut(spec: CascadeSpec, m: float) -> np.ndarray:
    """Heights where the smooth outline crosses the line x = m, ascending: one root of
    x(t) = m by brentq on each outline segment the line crosses (x(t) is monotone there)."""
    t, x_of_t, y_of_t = smooth_outline(spec)
    heights = []
    for k in line_cut(spec, m)[1]:
        def offset(u: float) -> float:
            return float(x_of_t(u)) - m
        lo, hi = t[k], t[k + 1]
        u = lo if offset(lo) == 0.0 else hi if offset(hi) == 0.0 else brentq(offset, lo, hi, xtol=1e-15, rtol=1e-15)
        heights.append(float(y_of_t(u)))
    return np.sort(np.array(heights))


@lru_cache(maxsize=None)
def smooth_narrowest(spec: CascadeSpec) -> tuple[float, float]:
    """The narrowest smooth passage width pitch - (top - bottom) over the blade and its m,
    measured here: the smooth cut (smooth_cut, exact crossings) on 401 lines across the
    blade, then scipy's bounded Brent search around the narrowest of them."""
    a = cascade_of(spec).active
    tip, tail = blade_extent(spec)

    def width(m: float) -> float:
        heights = smooth_cut(spec, m)
        return a.pitch - (heights[-1] - heights[0])

    lines = np.linspace(tip, tail, 403)[1:-1]
    k = int(np.argmin([width(m) for m in lines]))
    step = lines[1] - lines[0]
    best = minimize_scalar(width, bounds=(lines[k] - 2.0 * step, lines[k] + 2.0 * step), method="bounded",
                           options={"xatol": 1e-12 * a.chord})
    return float(best.fun), float(best.x)


def distance_to(points: np.ndarray, polyline: np.ndarray) -> np.ndarray:
    """Distance of each point to a polyline."""
    p = np.asarray(points, dtype=float).reshape(-1, 2)
    a, d = polyline[:-1], np.diff(polyline, axis=0)
    out = np.empty(len(p))
    for chunk in np.array_split(np.arange(len(p)), max(1, len(p) // 200)):
        q = p[chunk, None, :]
        t = np.clip(((q - a) * d).sum(-1) / (d * d).sum(-1), 0.0, 1.0)
        out[chunk] = np.sqrt(((q - (a + t[..., None] * d)) ** 2).sum(-1)).min(axis=1)
    return out


def assert_topology(spec: CascadeSpec, g: gr.Grid) -> None:
    """The hard invariants of a built grid, checked on the blade outline itself.

    Regions: exactly the stations whose line cuts the blade are blade stations, one run with
    periodic stations on both sides. Periodic stations: the line meets no blade, so the
    whole one-pitch eta line is fluid, and upper = lower + translation. Blade stations: the
    cut spans less than a pitch; the open line from its top to its bottom one pitch up holds
    no metal of any blade; the lower node lies on the top crossing's outline segment and the
    upper node, one pitch down, on the bottom crossing's (the passage-facing sides of the
    reference and neighbor blades), within WALL_TOLERANCE of the outline. Everywhere: the
    width is positive. Wall identity: each wall node's named surface in g.wall_identity is
    that of the outline segment it lies on, and, farther than 2e-3 c from geometry.py's LE
    and TE points, the named polyline nearest to it; a wall's named surface changes only
    between two stations on either side of the LE or the TE point; and each wall's contour
    coordinate never decreases."""
    a = cascade_of(spec).active
    s, tolerance = a.pitch, WALL_TOLERANCE * a.chord
    loop = outline(spec)
    first, last = g.first_blade_i, g.last_blade_i
    identity = g.wall_identity
    assert np.array_equal(identity.stations, np.arange(first, last + 1))
    named = {True: "suction", False: "pressure"}
    assert 1 <= first <= last <= len(g.m) - 2
    assert g.last_inlet_i == first - 1 and g.first_exit_i == last + 1
    np.testing.assert_array_equal(g.region, np.where(np.arange(len(g.m)) < first, gr.INLET,
                                                     np.where(np.arange(len(g.m)) > last, gr.EXIT, gr.PASSAGE)))
    assert np.all(g.upper_boundary[:, 1] - g.lower_boundary[:, 1] > 0.0)
    walls = []
    for i, m in enumerate(g.m):
        heights, segments = line_cut(spec, m)
        lower, upper = g.nodes[i, 0, 1], g.nodes[i, -1, 1]
        metal = float(np.sum(heights[1::2] - heights[0::2])) if heights.size else 0.0
        if first <= i <= last:
            assert metal > TOUCH * a.chord, f"station {i} is a blade station but its line misses the blade"
            assert heights[-1] - heights[0] < s, i
            assert not metal_on(spec, m, heights[-1] + 1e-12 * a.chord, heights[0] + s - 1e-12 * a.chord), i
            for node, segment in ((lower, segments[-1]), (upper - s, segments[0])):
                y_low, y_high = sorted((loop[segment, 1], loop[segment + 1, 1]))
                assert y_low - 1e-12 * a.chord <= node <= y_high + 1e-12 * a.chord, (i, node)
            assert identity.lower.surface[i - first] == named[on_suction_polyline(spec, segments[-1])], i
            assert identity.upper.surface[i - first] == named[on_suction_polyline(spec, segments[0])], i
            walls += [(m, lower), (m, upper - s)]
        else:
            assert metal <= TOUCH * a.chord, f"station {i} is periodic but its line cuts the blade"
            assert not metal_on(spec, m, lower, upper), i
            assert upper - lower == pytest.approx(s, abs=4.0 * np.spacing(max(abs(upper), s)))
    assert distance_to(np.array(walls), loop).max() <= tolerance
    nodes = np.array(walls).reshape(-1, 2, 2)                  # per blade station: lower node, upper node less s
    for k, wall in enumerate((identity.lower, identity.upper)):
        points = nodes[:, k]
        assert np.all(np.diff(wall.contour) >= -1e-12 * a.chord)
        clear = np.minimum(np.hypot(*(points - a.leading_edge_point).T),
                           np.hypot(*(points - a.trailing_edge_point).T)) > 2e-3 * a.chord
        nearer = np.where(distance_to(points, a.suction_surface) < distance_to(points, a.pressure_surface),
                          "suction", "pressure")
        assert np.array_equal(wall.surface[clear], nearer[clear])
        m = g.m[first:last + 1]
        for j in np.flatnonzero(wall.surface[1:] != wall.surface[:-1]):
            assert any(m[j] < edge <= m[j + 1] for edge in (a.leading_edge_point[0], a.trailing_edge_point[0])), j
    assert_walls_follow_their_contours(spec, g)


def assert_walls_follow_their_contours(spec: CascadeSpec, g: gr.Grid) -> None:
    """Between two neighboring blade stations each wall moves forward along its contour, by
    at least dm over the largest speed of the outline's parametric PCHIP (the contour
    coordinate is its chord-length parameter, not exact arc length), and the contour between
    the two wall nodes stays between the two station lines, unless a fold hides crossings at
    one of them: no part of a surface is skipped otherwise. The outline vertices passed are
    counted here from the crossing segments of this module's own cuts."""
    a = cascade_of(spec).active
    loop = outline(spec)
    n_segments = len(loop) - 1                              # the last vertex closes the loop on vertex 0
    t, x_of_t, y_of_t = smooth_outline(spec)
    samples = np.linspace(0.0, t[-1], 20 * len(t))
    fastest = np.hypot(x_of_t(samples, 1), y_of_t(samples, 1)).max()
    first, last = g.first_blade_i, g.last_blade_i
    cuts = [line_cut(spec, g.m[i])[1] for i in range(first, last + 1)]
    tolerance = 1e-12 * a.chord
    for wall, segments, forward in ((g.wall_identity.lower, [c[-1] for c in cuts], -1),
                                    (g.wall_identity.upper, [c[0] for c in cuts], 1)):
        assert np.all(np.diff(wall.contour) >= g.dm / fastest * (1.0 - 1e-9))
        for j in range(len(segments) - 1):
            k0, k1 = int(segments[j]), int(segments[j + 1])
            passed = [(k0 - q) % n_segments for q in range((k0 - k1) % n_segments)] if forward < 0 else \
                     [(k0 + 1 + q) % n_segments for q in range((k1 - k0) % n_segments)]
            x = loop[passed, 0]
            between = np.all((x >= g.m[first + j] - tolerance) & (x <= g.m[first + j + 1] + tolerance))
            assert between or wall.hidden[j] > 0 or wall.hidden[j + 1] > 0, (first + j, forward)


# --- Geometry handoff -------------------------------------------------------------------

@pytest.mark.parametrize("spec", CASES)
def test_blade_walls_are_the_geometry_surfaces_unchanged(spec):
    """lower = reference suction surface, upper = pressure surface + periodic translation,
    point for point and in the LE -> TE order of geometry.py: nothing re-split or sorted."""
    a, b = cascade_of(spec).active, boundaries(spec)
    np.testing.assert_array_equal(b.lower_blade_boundary, a.suction_surface)
    np.testing.assert_array_equal(b.upper_blade_boundary, a.pressure_surface + a.periodic_translation)
    np.testing.assert_array_equal(b.periodic_translation, a.periodic_translation)
    assert b.chord == a.chord and b.pitch == a.pitch


@pytest.mark.parametrize("spec", CASES)
def test_landmarks_are_the_exact_le_and_te_points(spec):
    c = cascade_of(spec)
    a, b = c.active, boundaries(spec)
    assert b.m_le == a.leading_edge_point[0]
    assert b.m_te == a.trailing_edge_point[0]
    # The walls start and end on those points: LE -> TE, and LE + s -> TE + s.
    np.testing.assert_array_equal(b.lower_blade_boundary[0], a.leading_edge_point)
    np.testing.assert_array_equal(b.lower_blade_boundary[-1], a.trailing_edge_point)
    np.testing.assert_array_equal(b.upper_blade_boundary[0], a.leading_edge_point + a.periodic_translation)
    np.testing.assert_array_equal(b.upper_blade_boundary[-1], a.trailing_edge_point + a.periodic_translation)
    # Chord-axis landmarks: LE at the origin, TE at chord (cos, sin) of the stagger.
    assert b.m_le == 0.0
    assert b.m_te == pytest.approx(a.chord * math.cos(c.stagger), rel=1e-14)


@pytest.mark.parametrize("spec", (SPEC, OTHER))
def test_the_rounded_nose_is_handed_on_with_its_fold(spec):
    """The nose leaves the LE point upstream, so min(x) of the walls is ahead of m_le and is
    not the landmark. The fold is on the wall whose side the LE circle center lies on, and
    it stays in the wall: one backward run from the LE vertex, then single-valued in m.

    Going downstream, the folding wall comes back to m = m_le well away from the LE point,
    so y(m) of that wall does not tend to y_le as m approaches m_le from inside the passage."""
    c = cascade_of(spec)
    a, b = c.active, boundaries(spec)
    center_above = a.leading_edge_center[1] > a.leading_edge_point[1]
    folding, other = ((b.lower_blade_boundary, b.upper_blade_boundary) if center_above
                      else (b.upper_blade_boundary, b.lower_blade_boundary))
    x = folding[:, 0]

    assert x.min() < b.m_le
    assert b.m_le == a.leading_edge_point[0] != x.min()
    assert b.m_in == pytest.approx(b.m_le - b.chord, abs=1e-15)

    backward = np.flatnonzero(np.diff(x) <= 0.0)
    nose = int(np.argmin(x))
    assert backward[0] == 0 and np.array_equal(backward[:nose], np.arange(nose))
    after = x[nose:]
    rising = np.diff(after) > 0.0
    k = int(np.argmax(after >= b.m_le))             # back at m_le, between vertices nose+k-1 and nose+k
    assert np.all(rising[:k])
    t = (b.m_le - after[k - 1]) / (after[k] - after[k - 1])
    y_back = folding[nose + k - 1, 1] + t * (folding[nose + k, 1] - folding[nose + k - 1, 1])
    assert abs(y_back - folding[0, 1]) > 0.1 * a.leading_edge_radius

    # The other wall leaves its LE point downstream.
    assert other[1, 0] > other[0, 0] == b.m_le


def test_dunavant_like_nose_and_tail_in_numbers():
    """The values the report quotes, for SPEC: the suction-side nose runs 0.0046 c ahead of
    m_le and comes back to m_le 0.028 c above the LE point; the Kovach pressure-side TE goes
    1.3e-5 c past m_te before it closes on the TE point. The lower wall is single-valued in
    m from its most upstream vertex to the TE, the upper one from the LE to its most
    downstream vertex."""
    b = boundaries(SPEC)
    lower, upper = b.lower_blade_boundary, b.upper_blade_boundary
    nose, tail = int(np.argmin(lower[:, 0])), int(np.argmax(upper[:, 0]))
    assert (b.m_le - lower[nose, 0]) / b.chord == pytest.approx(4.577e-3, abs=1e-6)
    assert (upper[tail, 0] - b.m_te) / b.chord == pytest.approx(1.325e-5, abs=1e-8)
    assert np.all(np.diff(lower[nose:, 0]) > 0.0) and lower[-1, 0] == lower[:, 0].max() == b.m_te
    assert np.all(np.diff(upper[:tail + 1, 0]) > 0.0) and np.all(np.diff(upper[tail:, 0]) < 0.0)


# --- Upstream length and samples -------------------------------------------------------

@pytest.mark.parametrize("spec", CASES)
@pytest.mark.parametrize("chords", (0.5, 1.0, 2.0))
def test_upstream_length_is_in_true_blade_chords(spec, chords):
    c = cascade_of(spec)
    b = boundaries(spec, settings=replace(SETTINGS, upstream_extension_chords=chords))
    length = chords * c.active.chord
    assert b.m_in == pytest.approx(b.m_le - length, rel=1e-14, abs=1e-16)
    assert b.m_in != pytest.approx(b.m_le - length * math.cos(c.stagger), rel=1e-6)


def test_one_chord_upstream_of_a_dimensional_cascade_is_its_physical_chord():
    b = boundaries(DIMENSIONAL)
    assert b.m_in == pytest.approx(b.m_le - 0.05, rel=1e-14)
    assert b.m_in != pytest.approx(b.m_le - 0.05 * math.cos(math.radians(DIMENSIONAL.stagger_deg)), rel=1e-6)
    normalized = boundaries(SPEC)
    assert (normalized.m_le, normalized.m_in) == (0.0, -1.0)


@pytest.mark.parametrize("spec", CASES)
@pytest.mark.parametrize("chords", (0.5, 1.0, 2.0))
def test_upstream_samples_are_uniform_and_end_exactly_on_m_in_and_m_le(spec, chords):
    b = boundaries(spec, settings=replace(SETTINGS, upstream_extension_chords=chords))
    m = b.upstream_m
    assert m[0] == b.m_in and m[-1] == b.m_le
    n_cells = len(m) - 1
    np.testing.assert_allclose(np.diff(m), (b.m_le - b.m_in) / n_cells, rtol=1e-12)
    # The count is the extension length in passage cells, rounded: the spacing differs from
    # the passage spacing by at most half a cell spread over the extension.
    passage_dm = (b.m_te - b.m_le) / SETTINGS.n_passage
    assert n_cells == round((b.m_le - b.m_in) / passage_dm)
    assert abs((b.m_le - b.m_in) / n_cells - passage_dm) <= passage_dm / (2 * n_cells) * (1 + 1e-12)


def test_upstream_sample_count_of_the_dunavant_like_cascade():
    """One chord ahead of a blade 0.8387 c long in m, at 31 passage cells: 36.96 cells -> 37."""
    b = boundaries(SPEC)
    assert len(b.upstream_m) == 38
    assert np.diff(b.upstream_m)[0] == pytest.approx(1.0 / 37.0, rel=1e-12)


# --- Inlet side-boundary angle and lines -----------------------------------------------

@pytest.mark.parametrize("spec", CASES)
@pytest.mark.parametrize("inlet_angle_deg", INLET_ANGLES_DEG)
def test_beta_runs_from_the_inlet_angle_to_kappa_le(spec, inlet_angle_deg):
    b = boundaries(spec, inlet_angle_deg)
    assert b.upstream_beta[0] == b.inlet_angle == math.radians(inlet_angle_deg)
    assert b.upstream_beta[-1] == cascade_of(spec).kappa_le


@pytest.mark.parametrize("spec", CASES)
@pytest.mark.parametrize("inlet_angle_deg", INLET_ANGLES_DEG)
def test_beta_is_linear_in_m(spec, inlet_angle_deg):
    """The angle itself, not its tangent, varies uniformly with m [Aungier p. 86]."""
    b = boundaries(spec, inlet_angle_deg)
    beta_in, kappa_le = math.radians(inlet_angle_deg), cascade_of(spec).kappa_le
    expected = beta_in + (b.upstream_m - b.m_in) / (b.m_le - b.m_in) * (kappa_le - beta_in)
    np.testing.assert_allclose(b.upstream_beta, expected, rtol=0, atol=1e-14)
    np.testing.assert_allclose(np.diff(b.upstream_beta, 2), 0.0, atol=1e-14)
    if abs(kappa_le - beta_in) > 1e-3:      # and tan(beta) is not linear
        assert np.abs(np.diff(np.tan(b.upstream_beta), 2)).max() > 1e-6


@pytest.mark.parametrize("spec", CASES)
@pytest.mark.parametrize("inlet_angle_deg", INLET_ANGLES_DEG)
def test_lower_extension_integrates_tan_beta_back_from_the_le(spec, inlet_angle_deg):
    a, b = cascade_of(spec).active, boundaries(spec, inlet_angle_deg)
    np.testing.assert_array_equal(b.upstream_lower_boundary[:, 0], b.upstream_m)
    reference = reference_line_y(b.upstream_m, a.leading_edge_point, cascade_of(spec).kappa_le, b.m_in,
                                 math.radians(inlet_angle_deg))
    np.testing.assert_allclose(b.upstream_lower_boundary[:, 1], reference, rtol=0, atol=1e-13 * b.chord)


def test_lower_extension_slope_converges_to_tan_beta():
    """Central differences of the samples approach tan(beta) at second order as the spacing
    halves: the line is the integral of tan(beta), checked without the closed form."""
    errors = []
    for n_passage in (31, 62, 124):
        b = boundaries(SPEC, 45.0, replace(SETTINGS, n_passage=n_passage))
        m, y = b.upstream_m, b.upstream_lower_boundary[:, 1]
        slope = (y[2:] - y[:-2]) / (m[2:] - m[:-2])
        errors.append(np.abs(slope - np.tan(b.upstream_beta[1:-1])).max())
    ratios = np.array(errors[:-1]) / np.array(errors[1:])
    assert errors[0] < 1e-4
    assert np.all((3.8 < ratios) & (ratios < 4.2)), (errors, ratios)


@pytest.mark.parametrize("spec", CASES)
def test_extensions_attach_to_the_exact_le_points(spec):
    a, b = cascade_of(spec).active, boundaries(spec)
    np.testing.assert_array_equal(b.upstream_lower_boundary[-1], a.leading_edge_point)
    np.testing.assert_array_equal(b.upstream_upper_boundary[-1], a.leading_edge_point + a.periodic_translation)
    np.testing.assert_array_equal(b.upstream_lower_boundary[-1], b.lower_blade_boundary[0])
    np.testing.assert_array_equal(b.upstream_upper_boundary[-1], b.upper_blade_boundary[0])
    # The last cell turns toward the blade angle: its secant direction lies between the
    # angles of its two samples, the second of which is kappa_le.
    last = np.diff(b.upstream_lower_boundary[-2:], axis=0)[0]
    low, high = sorted(b.upstream_beta[-2:])
    assert low <= math.atan2(last[1], last[0]) <= high


@pytest.mark.parametrize("spec", CASES)
@pytest.mark.parametrize("inlet_angle_deg", INLET_ANGLES_DEG)
def test_upper_extension_is_the_lower_one_moved_by_one_pitch(spec, inlet_angle_deg):
    b = boundaries(spec, inlet_angle_deg)
    lower, upper = b.upstream_lower_boundary, b.upstream_upper_boundary
    difference = upper - lower
    np.testing.assert_array_equal(difference[:, 0], 0.0)
    tolerance = 4.0 * np.spacing(np.maximum(np.abs(upper[:, 1]), b.pitch))
    assert np.all(np.abs(difference[:, 1] - b.pitch) <= tolerance)
    np.testing.assert_array_equal(b.periodic_translation, [0.0, b.pitch])


@pytest.mark.parametrize("spec", CASES)
def test_straight_line_when_the_inlet_angle_is_kappa_le(spec):
    """inlet_angle_deg = degrees(kappa_le): beta is constant and the line straight, with
    slope tan(kappa_le); no 0/0 where the two angles meet."""
    c = cascade_of(spec)
    b = gr.make_passage_boundaries(c, SETTINGS, math.radians(math.degrees(c.kappa_le)))
    m, y = b.upstream_m, b.upstream_lower_boundary[:, 1]
    assert np.all(np.isfinite(y))
    np.testing.assert_allclose(b.upstream_beta, c.kappa_le, rtol=0, atol=4e-16)
    line = c.active.leading_edge_point[1] - (b.m_le - m) * math.tan(c.kappa_le)
    np.testing.assert_allclose(y, line, rtol=0, atol=1e-15 * b.chord)
    np.testing.assert_allclose(np.diff(y) / np.diff(m), math.tan(c.kappa_le), rtol=1e-12)


@pytest.mark.parametrize("difference", (1e-4, 1e-7, 1e-10, 1e-13, -1e-10))
def test_nearly_equal_angles_stay_accurate(difference):
    """Inlet angles within a hair of kappa_le, where ln(cos a / cos b) / (b - a) would lose
    its digits, still match the quadrature."""
    c = cascade_of(SPEC)
    b = gr.make_passage_boundaries(c, SETTINGS, c.kappa_le - difference)
    reference = reference_line_y(b.upstream_m, c.active.leading_edge_point, c.kappa_le, b.m_in,
                                 c.kappa_le - difference)
    np.testing.assert_allclose(b.upstream_lower_boundary[:, 1], reference, rtol=0, atol=1e-14)


# --- Exit extension --------------------------------------------------------------------

@pytest.mark.parametrize("spec", CASES)
def test_without_an_exit_angle_the_exit_line_is_the_straight_kappa_te_tangent(spec):
    """exit_angle None means kappa_te, the starting value of the Kutta loop: beta is
    exactly kappa_te all along the exit extension and the line is exactly the tangent at
    the TE, y_te + (m - m_te) tan(kappa_te), bit for bit."""
    c = cascade_of(spec)
    a, b = c.active, boundaries(spec)
    assert b.exit_angle == c.kappa_te
    np.testing.assert_array_equal(b.downstream_beta, c.kappa_te)
    m, y = b.downstream_m, b.downstream_lower_boundary[:, 1]
    np.testing.assert_array_equal(y, a.trailing_edge_point[1] + (m - b.m_te) * math.tan(c.kappa_te))
    np.testing.assert_allclose(np.diff(y) / np.diff(m), math.tan(c.kappa_te), rtol=1e-12)
    explicit = gr.make_passage_boundaries(c, SETTINGS, math.radians(45.0), c.kappa_te)
    for item in fields(b):
        np.testing.assert_array_equal(getattr(explicit, item.name), getattr(b, item.name), err_msg=item.name)


@pytest.mark.parametrize("spec", CASES)
@pytest.mark.parametrize("chords", (0.5, 1.0, 2.0))
def test_downstream_length_is_in_true_blade_chords_with_uniform_samples(spec, chords):
    c = cascade_of(spec)
    b = boundaries(spec, settings=replace(SETTINGS, downstream_extension_chords=chords))
    length = chords * c.active.chord
    assert b.m_out == pytest.approx(b.m_te + length, rel=1e-14, abs=1e-16)
    assert b.m_out != pytest.approx(b.m_te + length * math.cos(c.stagger), rel=1e-6)
    m = b.downstream_m
    assert m[0] == b.m_te and m[-1] == b.m_out
    n_cells = len(m) - 1
    assert n_cells == round((b.m_out - b.m_te) / ((b.m_te - b.m_le) / SETTINGS.n_passage))
    np.testing.assert_allclose(np.diff(m), (b.m_out - b.m_te) / n_cells, rtol=1e-12)


@pytest.mark.parametrize("spec", CASES)
@pytest.mark.parametrize("turn_deg", EXIT_TURNS_DEG)
def test_exit_beta_runs_linearly_from_kappa_te_to_the_exit_angle(spec, turn_deg):
    c, b = cascade_of(spec), boundaries(spec, exit_turn_deg=turn_deg)
    exit_angle = exit_angle_of(spec, turn_deg)
    assert b.downstream_beta[0] == c.kappa_te
    assert b.downstream_beta[-1] == b.exit_angle == exit_angle
    expected = c.kappa_te + (b.downstream_m - b.m_te) / (b.m_out - b.m_te) * (exit_angle - c.kappa_te)
    np.testing.assert_allclose(b.downstream_beta, expected, rtol=0, atol=1e-14)
    np.testing.assert_allclose(np.diff(b.downstream_beta, 2), 0.0, atol=1e-14)
    assert np.abs(np.diff(np.tan(b.downstream_beta), 2)).max() > 1e-8     # beta linear, not tan(beta)


@pytest.mark.parametrize("spec", CASES)
@pytest.mark.parametrize("turn_deg", EXIT_TURNS_DEG)
def test_lower_exit_line_integrates_tan_beta_forward_from_the_te(spec, turn_deg):
    c, b = cascade_of(spec), boundaries(spec, exit_turn_deg=turn_deg)
    np.testing.assert_array_equal(b.downstream_lower_boundary[:, 0], b.downstream_m)
    reference = reference_line_y(b.downstream_m, c.active.trailing_edge_point, c.kappa_te, b.m_out,
                                 exit_angle_of(spec, turn_deg))
    np.testing.assert_allclose(b.downstream_lower_boundary[:, 1], reference, rtol=0, atol=1e-13 * b.chord)


@pytest.mark.parametrize("spec", CASES)
@pytest.mark.parametrize("turn_deg", (None, *EXIT_TURNS_DEG))
def test_exit_lines_start_on_the_exact_te_points_one_pitch_apart(spec, turn_deg):
    a, b = cascade_of(spec).active, boundaries(spec, exit_turn_deg=turn_deg)
    np.testing.assert_array_equal(b.downstream_lower_boundary[0], a.trailing_edge_point)
    np.testing.assert_array_equal(b.downstream_upper_boundary[0], a.trailing_edge_point + a.periodic_translation)
    np.testing.assert_array_equal(b.downstream_lower_boundary[0], b.lower_blade_boundary[-1])
    np.testing.assert_array_equal(b.downstream_upper_boundary[0], b.upper_blade_boundary[-1])
    difference = b.downstream_upper_boundary - b.downstream_lower_boundary
    np.testing.assert_array_equal(difference[:, 0], 0.0)
    tolerance = 4.0 * np.spacing(np.maximum(np.abs(b.downstream_upper_boundary[:, 1]), b.pitch))
    assert np.all(np.abs(difference[:, 1] - b.pitch) <= tolerance)


@pytest.mark.parametrize("difference", (1e-4, 1e-7, 1e-10, 1e-13, -1e-10))
def test_exit_angles_nearly_kappa_te_stay_accurate(difference):
    c = cascade_of(SPEC)
    b = gr.make_passage_boundaries(c, SETTINGS, math.radians(45.0), c.kappa_te + difference)
    reference = reference_line_y(b.downstream_m, c.active.trailing_edge_point, c.kappa_te, b.m_out,
                                 c.kappa_te + difference)
    np.testing.assert_allclose(b.downstream_lower_boundary[:, 1], reference, rtol=0, atol=1e-14)


EXIT_FIELDS = ("exit_angle", "downstream_beta", "downstream_lower_boundary", "downstream_upper_boundary")


@pytest.mark.parametrize("turn_deg", EXIT_TURNS_DEG)
def test_the_exit_angle_leaves_the_inlet_side_alone(turn_deg):
    reference, turned = boundaries(SPEC), boundaries(SPEC, exit_turn_deg=turn_deg)
    for item in fields(reference):
        if item.name not in EXIT_FIELDS:
            np.testing.assert_array_equal(getattr(turned, item.name), getattr(reference, item.name), err_msg=item.name)


# --- rebuild_exit on the boundaries ----------------------------------------------------

@pytest.mark.parametrize("spec", CASES)
def test_rebuilding_the_exit_at_the_same_angle_changes_nothing(spec):
    b = boundaries(spec, exit_turn_deg=5.0)
    again = gr.rebuild_exit(b, b.exit_angle)
    assert type(again) is gr.PassageBoundaries
    for item in fields(b):
        np.testing.assert_array_equal(getattr(again, item.name), getattr(b, item.name), err_msg=item.name)


@pytest.mark.parametrize("spec", CASES)
@pytest.mark.parametrize("turn_deg", EXIT_TURNS_DEG)
def test_rebuilding_the_exit_changes_only_the_exit_side(spec, turn_deg):
    start = boundaries(spec)
    rebuilt = gr.rebuild_exit(start, exit_angle_of(spec, turn_deg))
    for item in fields(start):
        if item.name not in EXIT_FIELDS:
            np.testing.assert_array_equal(getattr(rebuilt, item.name), getattr(start, item.name), err_msg=item.name)
    assert rebuilt.exit_angle == exit_angle_of(spec, turn_deg) != start.exit_angle
    np.testing.assert_array_equal(rebuilt.downstream_lower_boundary[:, 0], start.downstream_m)
    np.testing.assert_array_equal(rebuilt.downstream_lower_boundary[0], start.downstream_lower_boundary[0])
    assert np.all(rebuilt.downstream_beta[1:] != start.downstream_beta[1:])
    assert np.all(rebuilt.downstream_lower_boundary[1:, 1] != start.downstream_lower_boundary[1:, 1])


@pytest.mark.parametrize("spec", CASES)
@pytest.mark.parametrize("turn_deg", EXIT_TURNS_DEG)
def test_a_rebuilt_exit_is_a_fresh_build_at_that_angle(spec, turn_deg):
    rebuilt = gr.rebuild_exit(boundaries(spec), exit_angle_of(spec, turn_deg))
    fresh = boundaries(spec, exit_turn_deg=turn_deg)
    for item in fields(fresh):
        np.testing.assert_array_equal(getattr(rebuilt, item.name), getattr(fresh, item.name), err_msg=item.name)


@pytest.mark.parametrize("spec", CASES)
def test_a_sequence_of_exit_angles_returns_to_the_start(spec):
    """kappa_te -> +2 deg -> +5 deg -> kappa_te, as artificial Kutta updates: each rebuild
    starts again from the TE junction, so the end state is the start state bit for bit."""
    start = boundaries(spec)
    kappa_te = cascade_of(spec).kappa_te
    state = start
    for angle in (kappa_te + math.radians(2.0), kappa_te + math.radians(5.0), kappa_te):
        state = gr.rebuild_exit(state, angle)
    for item in fields(start):
        np.testing.assert_array_equal(getattr(state, item.name), getattr(start, item.name), err_msg=item.name)


# --- Stations ---------------------------------------------------------------------------

@pytest.mark.parametrize("spec", CASES)
@pytest.mark.parametrize("changes", ({}, {"n_passage": 62}, {"n_passage": 17}, {"n_passage": 101},
                                     {"upstream_extension_chords": 0.5}, {"downstream_extension_chords": 2.0}))
def test_stations_have_one_dm_and_end_exactly_on_m_in_and_m_out(spec, changes):
    settings = replace(SETTINGS, **changes)
    g = grid_of(spec, settings=settings)
    b = g.boundaries
    assert g.m[0] == g.m_in == b.m_in and g.m[-1] == g.m_out == b.m_out
    np.testing.assert_allclose(np.diff(g.m), g.dm, rtol=1e-12)
    dm_target = (b.m_te - b.m_le) / settings.n_passage
    n_cells = len(g.m) - 1
    assert n_cells == round((b.m_out - b.m_in) / dm_target)
    assert abs(g.dm - dm_target) <= dm_target / (2 * n_cells) * (1 + 1e-12)
    # n_passage is a target: the stations strictly between the exact LE and TE number
    # n_passage give or take one.
    inside = np.count_nonzero((g.m > b.m_le) & (g.m < b.m_te))
    assert abs(inside - settings.n_passage) <= 1


@pytest.mark.parametrize("n_eta", (2, 3, 21, 40, 161))
def test_eta_is_uniform_from_zero_to_one(n_eta):
    g = grid_of(SPEC, settings=replace(SETTINGS, n_eta=n_eta))
    assert len(g.eta) == n_eta and g.eta[0] == 0.0 and g.eta[-1] == 1.0
    np.testing.assert_allclose(np.diff(g.eta), g.deta, rtol=0, atol=1e-15)
    assert g.deta == pytest.approx(1.0 / (n_eta - 1), rel=1e-15)


def test_the_development_grid_in_numbers():
    """SPEC at 31 x 21: 105 cells of 0.0270350 c over 2.8387 c (target 0.0270539 c),
    106 x 21 = 2226 nodes. Stations 37 to 68 cut the blade. Station 37 lies 0.0003 c behind
    the exact LE, the station the old 0.01 c shift rule had made an inlet station through
    the nose; station 36, 0.0267 c ahead of the LE, clears the nose tip at -0.0046 c;
    station 68 lies 0.0003 c ahead of the TE and station 69, 0.0267 c behind it, clears the
    1.3e-5 c Kovach tail."""
    g = grid_of(SPEC)
    assert g.nodes.shape == (106, 21, 2)
    assert g.dm == pytest.approx(0.0270349578, rel=1e-9) and g.deta == 0.05
    assert (g.last_inlet_i, g.first_blade_i, g.last_blade_i, g.first_exit_i) == (36, 37, 68, 69)
    assert g.actual_le_offset == pytest.approx(2.934e-4, abs=1e-7)
    assert g.actual_te_offset == pytest.approx(-2.934e-4, abs=1e-7)
    assert g.m[g.last_inlet_i] - g.m_le == pytest.approx(-0.026742, abs=1e-6)
    assert g.m[g.first_exit_i] - g.m_te == pytest.approx(0.026742, abs=1e-6)
    assert g.exit_angle == cascade_of(SPEC).kappa_te
    assert math.degrees(g.exit_angle) == pytest.approx(17.9203, abs=1e-4)
    assert_topology(SPEC, g)


def test_the_101_by_101_development_grid_builds():
    """The fine grid the 0.01 c shift rule refused (dm 0.0083 c could not hold the shift and
    the 0.0046 c nose in one cell): 342 cells of 0.0083002 c, 343 x 101 nodes. The first
    blade station, 120, lies 0.003975 c AHEAD of the exact LE, in the nose; the last inlet
    station, 119, 0.012275 c ahead of it, clears the nose tip; the last blade station, 221,
    lies 0.004325 c ahead of the TE and the first exit station, 222, 0.003975 c behind it,
    past the tail."""
    g = grid_of(SPEC, settings=replace(SETTINGS, n_passage=101, n_eta=101))
    assert g.nodes.shape == (343, 101, 2)
    assert g.dm == pytest.approx(0.0083002063, rel=1e-8) and g.deta == pytest.approx(0.01, rel=1e-15)
    assert (g.last_inlet_i, g.first_blade_i, g.last_blade_i, g.first_exit_i) == (119, 120, 221, 222)
    assert g.actual_le_offset == pytest.approx(-0.003975239, abs=1e-9)
    assert g.actual_te_offset == pytest.approx(-0.004324967, abs=1e-9)
    assert g.m[g.last_inlet_i] - g.m_le == pytest.approx(-0.012275446, abs=1e-9)
    assert g.m[g.first_exit_i] - g.m_te == pytest.approx(0.003975239, abs=1e-9)
    assert_topology(SPEC, g)


def test_there_is_no_chordwise_shift_setting():
    """The blade stations follow from the blade and the stations; there is no prescribed
    distance from the LE (Aungier gives none), so the old setting is gone, not ignored."""
    assert "leading_edge_shift_chords" not in {item.name for item in fields(GridSettings)}
    with pytest.raises(TypeError):
        GridSettings(n_passage=31, n_eta=21, leading_edge_shift_chords=0.01)


# --- Blade-passage topology -------------------------------------------------------------

@pytest.mark.parametrize("spec", EDGE_CASES)
@pytest.mark.parametrize("n_passage", (17, 31, 62, 101, 141))
def test_the_blade_stations_are_exactly_the_stations_that_cut_the_blade(spec, n_passage):
    assert_topology(spec, grid_of(spec, 30.0, replace(SETTINGS, n_passage=n_passage, n_eta=11)))


@pytest.mark.parametrize("spec", EDGE_CASES)
@pytest.mark.parametrize("n_passage", (21, 31, 41, 61, 81, 101, 141))
def test_the_first_and_last_blade_stations_follow_the_blade_extent(spec, n_passage):
    """The last inlet station is at or ahead of the blade's most upstream point and the first
    blade station behind it, the last blade station ahead of its most downstream point and
    the first exit station at or behind it: each edge falls inside one cell, so the gaps
    shrink with dm."""
    g = grid_of(spec, 30.0, replace(SETTINGS, n_passage=n_passage, n_eta=11))
    tip, tail = blade_extent(spec)
    assert g.m[g.last_inlet_i] <= tip < g.m_first_blade <= tip + g.dm * (1 + 1e-12)
    assert tail - g.dm * (1 + 1e-12) <= g.m_last_blade < tail <= g.m[g.first_exit_i]


def test_the_le_offset_follows_the_nose_and_changes_sign():
    """actual_le_offset = m_first_blade - m_le is a diagnostic. It lies within one dm behind
    the nose tip, which reaches 0.0046 c ahead of the chord-axis LE, so it tends to -0.0046 c
    with refinement, not to zero, and its sign depends on where the stations fall: a fixed
    shift behind the LE would forbid every negative value."""
    tip = blade_extent(SPEC)[0]
    offsets = []
    for n_passage in (21, 31, 41, 61, 81, 101, 141):
        g = grid_of(SPEC, settings=replace(SETTINGS, n_passage=n_passage, n_eta=11))
        offsets.append(g.actual_le_offset)
        assert 0.0 < g.actual_le_offset - (tip - g.m_le) <= g.dm * (1 + 1e-12)
    assert min(offsets) < 0.0 < max(offsets)
    assert (tip - g.m_le) / g.chord == pytest.approx(-4.577e-3, abs=1e-6)


def test_a_station_in_the_nose_ahead_of_the_le_is_a_blade_station():
    """A station 0.002 c ahead of the exact LE lies in the part of the nose that reaches
    0.0046 c ahead of it. Its line crosses only the suction polyline, twice: out from the LE
    point to the nose tip and back. It is the first blade station; its lower wall node is
    the top of that cut and its upper wall node the bottom moved one pitch, on the
    neighbor's nose underside, which geometry.py files under the suction surface because its
    LE point is the chord-axis end, not the most upstream point."""
    a = cascade_of(SPEC).active
    m_le = a.leading_edge_point[0]
    g = gr.make_grid(cascade_of(SPEC), stations_with_one_at(SPEC, m_le - 0.002), math.radians(45.0))
    i = int(np.argmin(np.abs(g.m - (m_le - 0.002))))
    assert g.m[i] == pytest.approx(m_le - 0.002, abs=1e-12)
    assert i == g.first_blade_i and g.actual_le_offset < 0.0
    heights, segments = line_cut(SPEC, g.m[i])
    assert heights.size == 2 and all(on_suction_polyline(SPEC, k) for k in segments)
    assert g.nodes[i, 0, 1] == pytest.approx(heights[-1], abs=WALL_TOLERANCE)
    assert g.nodes[i, -1, 1] - a.pitch == pytest.approx(heights[0], abs=WALL_TOLERANCE)
    assert heights[-1] - heights[0] > 0.005 * a.chord            # two distinct wall points
    identity = g.wall_identity
    assert (identity.lower.surface[0], identity.upper.surface[0], identity.upper.surface[1]) == ("suction", "suction",
                                                                                               "pressure")
    assert_topology(SPEC, g)


def test_a_station_through_a_thin_nose_fold_takes_the_passage_facing_crossing():
    """THIN's pressure side folds back in m a little behind the LE, so a station in the fold
    crosses the outline four times: once on the suction side and three times on the
    pressure side. The grid builds, and the upper wall node is the lowest crossing moved one
    pitch, the side that faces the passage below the blade; the crossings above it, inside
    the fold, are not walls of this passage. The wall identity reports them: two crossings
    hidden behind the upper wall node, which lies on the pressure surface past the fold, so
    the upper contour coordinate jumps over the pocket, forward."""
    a = cascade_of(THIN).active
    m_le = a.leading_edge_point[0]
    x = blade_walls(THIN)[1][:, 0]
    back = np.flatnonzero(np.diff(x) <= 0.0)
    back = back[(x[back] > m_le) & (x[back] < m_le + 0.005)]
    zone = (x[back[-1] + 1], x[back[0]])
    target = 0.5 * (zone[0] + zone[1])
    g = gr.make_grid(cascade_of(THIN), stations_with_one_at(THIN, target), math.radians(45.0))
    i = int(np.argmin(np.abs(g.m - target)))
    assert g.m[i] == pytest.approx(target, abs=1e-12) and g.first_blade_i <= i <= g.last_blade_i
    heights, segments = line_cut(THIN, g.m[i])
    assert heights.size == 4
    assert [on_suction_polyline(THIN, k) for k in segments] == [False, False, False, True]
    assert g.nodes[i, -1, 1] - a.pitch == pytest.approx(heights[0], abs=WALL_TOLERANCE)
    assert g.nodes[i, 0, 1] == pytest.approx(heights[-1], abs=WALL_TOLERANCE)
    identity, k = g.wall_identity, i - g.first_blade_i
    assert (identity.upper.hidden[k], identity.lower.hidden[k], identity.upper.surface[k]) == (2, 0, "pressure")
    assert np.count_nonzero(identity.upper.hidden) == 1 and np.all(identity.lower.hidden == 0)
    assert_topology(THIN, g)


@pytest.mark.parametrize("physical_chord", (None, 0.05))
def test_a_station_on_a_sharp_te_touches_the_blade_and_is_periodic(physical_chord):
    """At 60 deg stagger the blade is 0.5 c long in m, so with one-chord extensions and 17
    passage cells (85 cells over 2.5 c) station 51 lands on the sharp emery_base TE, within
    an ulp. Its line only touches the blade there (with round-off, 0 or 2e-16 c of metal):
    it is the first exit station, not a blade station whose two wall nodes would collapse
    onto the TE point, and the chord-1 and physical cascades agree on it."""
    spec = CascadeSpec(blade_type="NACA65", solidity=0.8, stagger_deg=60.0, cl0=0.4, t_max_over_c=0.04,
                       profile_variant="emery_base", physical_chord=physical_chord, surface_continuity="C1")
    a = cascade_of(spec).active
    g = grid_of(spec, 30.0, replace(SETTINGS, n_passage=17, n_eta=11))
    assert abs(g.m[51] - g.m_te) < 1e-15 * a.chord
    assert (g.last_blade_i, g.first_exit_i) == (50, 51)
    heights = line_cut(spec, g.m[51])[0]
    assert float(np.sum(heights[1::2] - heights[0::2])) <= TOUCH * a.chord
    thickness = g.nodes[50, 0, 1] - (g.nodes[50, -1, 1] - a.pitch)
    assert thickness > 1e-3 * a.chord                  # the last blade station keeps two distinct wall nodes
    assert_topology(spec, g)


def test_a_station_in_a_kovach_tail_is_the_last_blade_station():
    """LONG_TAIL's pressure side passes m_te by 0.004 c before it closes on the TE point. A
    station 0.002 c behind m_te cuts only that tail, twice on the pressure polyline: it is
    the last blade station, its lower wall node is the top of the tail on the pressure
    polyline's return to the TE point, actual_te_offset is positive, and the next station's
    line is clear of the blade."""
    a = cascade_of(LONG_TAIL).active
    m_te = a.trailing_edge_point[0]
    tail = blade_extent(LONG_TAIL)[1]
    assert (tail - m_te) / a.chord == pytest.approx(4.0e-3, abs=1e-4)
    target = m_te + 0.002 * a.chord
    g = gr.make_grid(cascade_of(LONG_TAIL), stations_with_one_at(LONG_TAIL, target), math.radians(30.0))
    i = int(np.argmin(np.abs(g.m - target)))
    assert g.m[i] == pytest.approx(target, abs=1e-12)
    assert i == g.last_blade_i and g.actual_te_offset > 0.0
    heights, segments = line_cut(LONG_TAIL, g.m[i])
    assert heights.size == 2 and not any(on_suction_polyline(LONG_TAIL, k) for k in segments)
    assert g.nodes[i, 0, 1] == pytest.approx(heights[-1], abs=WALL_TOLERANCE)
    assert line_cut(LONG_TAIL, g.m[g.first_exit_i])[0].size == 0
    identity = g.wall_identity      # the lower wall reaches the pressure surface's tail return past the TE point only
    assert identity.lower.surface[-1] == "pressure" and np.all(identity.lower.surface[:-1] == "suction")
    assert np.all(identity.upper.surface == "pressure")
    assert_topology(LONG_TAIL, g)


def test_a_station_in_a_suction_side_kovach_tail_is_the_last_blade_station():
    """At -30 deg stagger the Kovach overshoot is on the suction side, which passes m_te by
    0.0040 c before it closes on the TE point. A station 0.002 c behind m_te cuts only that
    tail, both times on the suction polyline: the top on its way out (the lower wall node)
    and the bottom on its return to the TE point (the upper wall node, one pitch up, on the
    neighbor's suction polyline, which faces this passage there). It is the last blade
    station, and the next station's line is clear of the blade."""
    a = cascade_of(SUCTION_TAIL).active
    m_te = a.trailing_edge_point[0]
    suction = a.suction_surface
    turn = int(np.argmax(suction[:, 0]))                 # the suction side's most downstream vertex
    assert suction[turn, 0] == blade_extent(SUCTION_TAIL)[1] and turn < len(suction) - 1
    assert (suction[turn, 0] - m_te) / a.chord == pytest.approx(3.96e-3, abs=1e-5)
    target = m_te + 0.002 * a.chord
    g = gr.make_grid(cascade_of(SUCTION_TAIL), stations_with_one_at(SUCTION_TAIL, target), math.radians(30.0))
    i = int(np.argmin(np.abs(g.m - target)))
    assert g.m[i] == pytest.approx(target, abs=1e-12)
    assert i == g.last_blade_i and g.actual_te_offset > 0.0
    heights, segments = line_cut(SUCTION_TAIL, g.m[i])
    assert heights.size == 2 and all(on_suction_polyline(SUCTION_TAIL, k) for k in segments)
    top, bottom = (len(suction) - 2 - segments[-1], len(suction) - 2 - segments[0])    # suction-surface segments
    assert top < turn <= bottom
    assert g.nodes[i, 0, 1] == pytest.approx(heights[-1], abs=WALL_TOLERANCE)
    assert g.nodes[i, -1, 1] - a.pitch == pytest.approx(heights[0], abs=WALL_TOLERANCE)
    assert line_cut(SUCTION_TAIL, g.m[g.first_exit_i])[0].size == 0
    identity = g.wall_identity      # the upper wall reaches the suction surface's tail return past the TE point only
    assert identity.upper.surface[-1] == "suction" and np.all(identity.upper.surface[:-1] == "pressure")
    assert np.all(identity.lower.surface == "suction")
    assert_topology(SUCTION_TAIL, g)


@pytest.mark.parametrize("spec, end", ((SPEC, "tip"), (SPEC, "tail"), (THIN, "tip"), (LONG_TAIL, "tail"),
                                       (SHARP_0, "tail"), (SHARP_60, "tail")))
def test_stations_within_ulps_of_the_blade_ends_only_touch_the_blade(spec, end):
    """A station on the blade's most upstream point (the nose tip) or most downstream point
    (a Kovach tail, a sharp TE), or a few ulps either side of it, holds at most 2.2e-13 c of
    metal on its line (1e-15 c inside a round nose, whose polyline is nearly vertical at the
    tip), less than TOUCH: it is periodic, and the next station into the blade is a blade
    station with two distinct wall nodes. Thirteen grids each, the upstream extension moved
    by -6 to +6 ulps; some of the tail stations land exactly on the vertex."""
    a = cascade_of(spec).active
    x = blade_extent(spec)[0 if end == "tip" else 1]
    base = stations_with_one_at(spec, x, replace(SETTINGS, n_passage=40, n_eta=5))
    for ulps in range(-6, 7):
        g = gr.make_grid(cascade_of(spec), nudged(base, ulps), math.radians(45.0))
        i = int(np.argmin(np.abs(g.m - x)))
        assert abs(g.m[i] - x) < 1e-15 * a.chord
        heights = line_cut(spec, g.m[i])[0]
        assert float(np.sum(heights[1::2] - heights[0::2])) <= TOUCH * a.chord
        assert i == (g.last_inlet_i if end == "tip" else g.first_exit_i)
        j = g.first_blade_i if end == "tip" else g.last_blade_i
        assert g.nodes[j, 0, 1] - (g.nodes[j, -1, 1] - a.pitch) > 5e-4 * a.chord
        assert_topology(spec, g)


@pytest.mark.parametrize("upstream", (0.0025000000000001132, 0.0025000000000001644))
def test_two_stations_just_ahead_of_a_sharp_te_are_periodic(upstream):
    """Two cases found in review: with a 0.0025 c upstream extension and 200 passage cells,
    station 201 lands 1.5 and 2 ulps of m ahead of a sharp TE vertex, where the blade's cut is
    round-off thin. Before TOUCH it was a blade station whose two wall nodes both sat on the
    TE point; now it is the first exit station, and the last blade station, one dm (0.0025 c)
    ahead, has its wall nodes 3.6e-4 c apart."""
    a = cascade_of(SHARP_60_FLAT).active
    settings = GridSettings(n_passage=200, n_eta=9, upstream_extension_chords=upstream, downstream_extension_chords=1.0)
    g = gr.make_grid(cascade_of(SHARP_60_FLAT), settings, math.radians(45.0))
    assert (g.last_blade_i, g.first_exit_i) == (200, 201)
    assert -3e-16 < g.m[201] - blade_extent(SHARP_60_FLAT)[1] < 0.0
    assert g.nodes[200, 0, 1] - (g.nodes[200, -1, 1] - a.pitch) > 3.5e-4 * a.chord
    assert_topology(SHARP_60_FLAT, g)


def test_a_passage_the_blades_all_but_close_keeps_its_nodes_out_of_the_smooth_walls():
    """NEAR_CLOSED narrows the passage to 3.5e-6 c at 101 x 21. The grid builds, and no
    interior node lies inside the smooth walls the wall nodes lie on (vertical cuts of the
    PCHIP outline written here). Read against the polyline, whose gap lies a little higher or
    lower, interior nodes do lie inside the blade, by less than 1e-5 c vertically: the
    limitation of grid.py section 6."""
    a = cascade_of(NEAR_CLOSED).active
    g = grid_of(NEAR_CLOSED, 45.0, replace(SETTINGS, n_passage=101, n_eta=21))
    walls = slice(g.first_blade_i, g.last_blade_i + 1)
    width = g.upper_boundary[walls, 1] - g.lower_boundary[walls, 1]
    assert 0.0 < width.min() < 1e-5 * a.chord
    inside_polyline, depth = 0, 0.0
    for i in range(g.first_blade_i, g.last_blade_i + 1):
        smooth = np.concatenate([smooth_cut(NEAR_CLOSED, g.m[i]) + k * a.pitch for k in (-1, 0, 1, 2)])
        polyline = np.concatenate([line_cut(NEAR_CLOSED, g.m[i])[0] + k * a.pitch for k in (-1, 0, 1, 2)])
        for y in g.nodes[i, 1:-1, 1]:
            assert np.count_nonzero(smooth > y) % 2 == 0, (i, y)
            if np.count_nonzero(polyline > y) % 2:
                inside_polyline += 1
                depth = max(depth, float(np.abs(polyline - y).min()))
    assert inside_polyline > 0 and depth < 1e-5 * a.chord
    assert_topology(NEAR_CLOSED, g)


@pytest.mark.parametrize("n_eta", (2, 21, 101))
def test_blades_whose_smooth_walls_close_the_passage_are_refused_at_every_resolution(n_eta):
    """CLOSED's straight-segment outline, which geometry.py checks for overlap, stays open by
    3.6e-6 c, but its smooth walls cross, by up to 1.1e-5 c, over 0.0039 c of m near
    m = 0.236 c. The passage check looks at the whole blade before any station, so every
    n_passage from 17 to 200, every n_eta and either upstream length gets the same refusal,
    word for word. (Checked at the stations only, 17, 21, 31, 61 and 81 cells had built.)"""
    messages = set()
    for n_passage in (17, 21, 31, 41, 61, 81, 101, 141, 200):
        for upstream in (1.0, 0.5):
            settings = replace(SETTINGS, n_passage=n_passage, n_eta=n_eta, upstream_extension_chords=upstream)
            with pytest.raises(gr.GridError, match="smooth walls close the passage") as refused:
                gr.make_grid(cascade_of(CLOSED), settings, math.radians(45.0))
            messages.add(str(refused.value))
    assert len(messages) == 1


def test_the_refusal_names_the_narrowest_width_its_place_the_blade_and_the_solidity():
    """The message gives the narrowest smooth width, -1.1358e-5 c at m = 0.23614 c, which
    the search written here (exact crossings, scipy's bounded Brent) finds too, the blade as
    geometry.py names it, the stagger and the solidity."""
    width, where = smooth_narrowest(CLOSED)
    assert width == pytest.approx(-1.13580e-5, abs=1e-9) and where == pytest.approx(0.23614, abs=1e-5)
    with pytest.raises(gr.GridError) as refused:
        gr.make_grid(cascade_of(CLOSED), SETTINGS, math.radians(45.0))
    message = str(refused.value)
    found = re.search(r"narrowest width over the blade is (\S+) chords, at m = (\S+) ", message)
    assert float(found[1]) == pytest.approx(width, rel=1e-6) and float(found[2]) == pytest.approx(where, abs=1e-5)
    for part in ("NACA 65-(4A10)10", "sharp", "stagger 50 deg", "solidity 6.424623382", "TOUCH = 1e-12 chords"):
        assert part in message, part


def test_a_smooth_width_of_exactly_touch_counts_as_closed(monkeypatch):
    """The tolerance itself: with the narrowest width set to TOUCH chords times 1 + 1e-9,
    1 and 1 - 1e-9, the first builds and the other two are refused."""
    chord = cascade_of(SPEC).active.chord
    for factor, builds in ((1.0 + 1e-9, True), (1.0, False), (1.0 - 1e-9, False)):
        monkeypatch.setattr(gr._BladeRow, "narrowest", lambda self, pitch, w=factor * TOUCH * chord: (w, 0.3))
        if builds:
            gr.make_grid(cascade_of(SPEC), SETTINGS, math.radians(45.0))
        else:
            with pytest.raises(gr.GridError, match="smooth walls close the passage"):
                gr.make_grid(cascade_of(SPEC), SETTINGS, math.radians(45.0))


def with_pitch(spec: CascadeSpec, pitch: float) -> Cascade:
    """spec's cascade with the pitch and the periodic translation, and nothing else, set to
    pitch (the blade does not depend on the pitch; the throat fields go stale, and grid.py
    does not read them). Test side only. It bypasses make_cascade, whose spline throat is
    round-off small (1e-16 c) this close to closure and whose check refuses only an exact
    0.0, so that make_cascade accepts or refuses such pitches by round-off."""
    c = cascade_of(spec)
    return replace(c, active=replace(c.active, pitch=pitch, periodic_translation=np.array([0.0, pitch])))


@pytest.mark.parametrize("excess, builds", ((1e-9, True), (2.0 * TOUCH, True), (0.5 * TOUCH, False), (-1e-9, False)))
def test_a_blade_closes_its_passage_when_the_pitch_reaches_its_widest_smooth_cut(excess, builds):
    """The smooth width is the pitch less the smooth cut's span, so CLOSED's blade closes its
    passage when the pitch comes down to its widest smooth cut D = 0.1556625 c, measured
    here. With the pitch D + excess, 1e-9 c and 2 TOUCH build and 0.5 TOUCH and -1e-9 c are
    refused, at 17 and at 101 passage cells alike; a refusal reports the excess as the
    narrowest width."""
    width, _ = smooth_narrowest(CLOSED)
    widest = cascade_of(CLOSED).active.pitch - width
    assert widest == pytest.approx(0.1556625, abs=1e-7)
    cascade = with_pitch(CLOSED, widest + excess)
    for n_passage in (17, 101):
        settings = replace(SETTINGS, n_passage=n_passage, n_eta=5)
        if builds:
            gr.make_grid(cascade, settings, math.radians(45.0))
        else:
            with pytest.raises(gr.GridError, match="smooth walls close the passage") as refused:
                gr.make_grid(cascade, settings, math.radians(45.0))
            found = re.search(r"narrowest width over the blade is (\S+) chords", str(refused.value))
            assert float(found[1]) == pytest.approx(excess, rel=1e-3)


@pytest.mark.parametrize("spec", (NEAR_CLOSED, replace(NEAR_CLOSED, solidity=5.93183135)))
def test_passages_that_stay_open_at_the_overlap_limit_build_at_every_resolution(spec):
    """NEAR_CLOSED, and the same blade just under the largest solidity geometry.py accepts
    (5.9318313519), stay open on the smooth walls, by 2.3e-6 and 2.0e-6 c, measured here.
    They build at 17 to 200 passage cells and 5 and 41 eta nodes, so the passage check adds
    no refusal band of its own, and every station's width is at least that narrowest one."""
    width, _ = smooth_narrowest(spec)
    assert 1e-6 < width < 1e-5
    for n_passage in (17, 41, 101, 200):
        for n_eta in (5, 41):
            g = gr.make_grid(cascade_of(spec), replace(SETTINGS, n_passage=n_passage, n_eta=n_eta), math.radians(45.0))
            walls = slice(g.first_blade_i, g.last_blade_i + 1)
            assert np.all(g.upper_boundary[walls, 1] - g.lower_boundary[walls, 1] >= width - 1e-15)


@pytest.mark.parametrize("spec", (SPEC, OTHER, THIN, LONG_TAIL, SUCTION_TAIL, CLOSED))
def test_the_smooth_walls_are_monotone_in_x_and_y_on_every_segment(spec):
    """The passage check's bound rests on this: on each segment of the outline the PCHIP
    x(t) and y(t) are monotone (65 points a segment), so the smooth top and bottom of a cut
    are monotone in m between two vertex m."""
    t, x_of_t, y_of_t = smooth_outline(spec)
    u = t[:-1, None] + np.linspace(0.0, 1.0, 65) * np.diff(t)[:, None]
    for curve in (x_of_t, y_of_t):
        steps = np.diff(curve(u), axis=1)
        assert np.all(np.all(steps >= -1e-15, axis=1) | np.all(steps <= 1e-15, axis=1))


@pytest.mark.parametrize("n_passage, nose_stations", ((31, 0), (51, 0), (81, 0), (101, 1), (141, 1), (201, 1)))
def test_the_example_walls_keep_their_contours_at_every_resolution(n_passage, nose_stations):
    """The walls of the EXAMPLE follow the same two contours at every resolution: the lower
    wall the reference blade's upper contour, all on the suction surface; the upper wall the
    neighbor's lower contour, on the pressure surface except at a blade station ahead of the
    LE point, which cuts the nose and lies on the neighbor's nose underside (suction surface
    between the LE point and the nose tip). Which name a wall node carries is a function of
    its m alone. Each wall moves forward along its contour from station to station and
    skips no part of it (assert_walls_follow_their_contours, through assert_topology), no
    crossing is hidden, and the parameter on each named surface runs the way the contour
    does."""
    g = grid_of(SPEC, settings=replace(SETTINGS, n_passage=n_passage, n_eta=21))
    assert_topology(SPEC, g)
    identity, m = g.wall_identity, g.m[g.first_blade_i:g.last_blade_i + 1]
    assert np.array_equal(identity.upper.surface == "suction", m < g.m_le)
    assert np.count_nonzero(identity.upper.surface == "suction") == nose_stations
    assert np.all(identity.lower.surface == "suction")
    for wall in (identity.lower, identity.upper):
        assert np.all(wall.hidden == 0)
        for surface, direction in (("suction", 1.0 if wall is identity.lower else -1.0), ("pressure", 1.0)):
            run = wall.parameter[wall.surface == surface]
            assert np.all(direction * np.diff(run) > 0.0)


def test_the_upper_wall_runs_onto_the_nose_underside_only_ahead_of_the_le_point():
    """LE audit of the EXAMPLE at 101 passage cells. Station 120, 0.00398 c ahead of the LE
    point, crosses the suction surface twice and the pressure surface not at all
    (geometry.py starts the pressure surface at the LE point, and the nose tip is 0.00458 c
    further upstream, on the suction surface). Its lower wall node is the crossing past the
    nose tip (suction segment 62, parameter 0.0161) and its upper wall node, one pitch up,
    the crossing between the LE point and the tip (segment 40, parameter 0.0065), the
    neighbor's nose underside. From there the upper contour runs through the LE point, at
    an arc length from the tip between those of stations 120 and 121, onto the pressure
    surface. With named surfaces only, station 120 would have no upper wall node at all."""
    a = cascade_of(SPEC).active
    g = grid_of(SPEC, settings=replace(SETTINGS, n_passage=101, n_eta=21))
    i = g.first_blade_i
    assert i == 120 and g.m[i] - g.m_le == pytest.approx(-0.003975, abs=1e-6)
    heights, segments = line_cut(SPEC, g.m[i])
    assert heights.size == 2 and all(on_suction_polyline(SPEC, k) for k in segments)
    suction_segment = [len(a.suction_surface) - 2 - int(k) for k in segments]           # in suction LE -> TE order
    tip = int(np.argmin(a.suction_surface[:, 0]))
    assert suction_segment == [40, 62] and suction_segment[0] < tip <= suction_segment[1]
    identity = g.wall_identity
    assert (identity.lower.surface[0], identity.upper.surface[0], identity.upper.surface[1]) == ("suction", "suction",
                                                                                               "pressure")
    assert identity.lower.parameter[0] == pytest.approx(0.0161, abs=1e-4)
    assert identity.upper.parameter[0] == pytest.approx(0.0065, abs=1e-4)
    t = smooth_outline(SPEC)[0]
    le_vertex, tip_vertex = len(a.suction_surface) - 1, len(a.suction_surface) - 1 - tip
    to_le = t[le_vertex] - t[tip_vertex]                     # along the lower contour, from the nose tip
    assert identity.upper.contour[0] < to_le < identity.upper.contour[1]


def test_a_station_in_a_pressure_side_nose_has_its_lower_wall_on_the_pressure_surface():
    """At negative stagger the nose wraps on the pressure side: OTHER's nose tip lies
    4e-5 c ahead of the LE point, on the pressure surface. A station 2e-5 c ahead of the LE
    point crosses only the pressure surface; its lower wall node lies on the pressure
    surface's nose part between the tip and the LE point, the reference blade's upper
    contour there, and every other lower wall node on the suction surface. The sharp TE
    changes nothing: the upper wall stays on the pressure surface."""
    a = cascade_of(OTHER).active
    target = a.leading_edge_point[0] - 2e-5 * a.chord
    g = gr.make_grid(cascade_of(OTHER), stations_with_one_at(OTHER, target), math.radians(45.0))
    i = int(np.argmin(np.abs(g.m - target)))
    assert g.m[i] == pytest.approx(target, abs=1e-12) and i == g.first_blade_i
    heights, segments = line_cut(OTHER, g.m[i])
    assert heights.size == 2 and not any(on_suction_polyline(OTHER, k) for k in segments)
    identity = g.wall_identity
    assert identity.lower.surface[0] == "pressure" and np.all(identity.lower.surface[1:] == "suction")
    assert np.all(identity.upper.surface == "pressure")
    assert_topology(OTHER, g)


def test_a_wall_that_changes_its_named_surface_elsewhere_is_refused(monkeypatch):
    """Grid checks the identity itself: with the suction surface's middle stretch labelled
    pressure, the lower wall would change surface at mid-chord, away from the LE and TE
    points, and the grid is refused."""
    original = gr._BladeRow.on_suction

    def mislabelled(self, segment):
        middle = (segment > 60) & (segment < 120)
        return original(self, segment) & ~middle

    monkeypatch.setattr(gr._BladeRow, "on_suction", mislabelled)
    with pytest.raises(gr.GridError, match="without passing geometry.py's LE or TE point"):
        gr.make_grid(cascade_of(SPEC), SETTINGS, math.radians(45.0))


def test_a_wall_that_runs_backward_along_its_contour_is_refused(monkeypatch):
    """With the crossings' places on the outline handed back in reverse station order, the
    walls would run backward along their contours, and the grid is refused."""
    original = gr._BladeRow.smooth_parameter
    monkeypatch.setattr(gr._BladeRow, "smooth_parameter", lambda self, segment, m: original(self, segment, m)[::-1])
    with pytest.raises(gr.GridError, match="runs backward along its contour"):
        gr.make_grid(cascade_of(SPEC), SETTINGS, math.radians(45.0))


def test_a_wall_node_off_its_contour_is_refused():
    """With the nose tip and the tail end swapped, the top of every cut would lie on the
    lower contour, and the identity check refuses it."""
    b = boundaries(SPEC)
    row = gr._BladeRow(b)
    row.tip, row.tail = row.tail, row.tip
    g = grid_of(SPEC)
    walls = slice(g.first_blade_i, g.last_blade_i + 1)
    with pytest.raises(gr.GridError, match="not on its contour"):
        row.identity(row.cut(g.m), walls, g.first_blade_i)


@pytest.mark.parametrize("spec", (SPEC, OTHER, STEEP))
def test_the_exact_le_and_te_stay_landmarks_of_an_unchanged_cascade(spec):
    c = cascade_of(spec)
    a = c.active
    before = [array.copy() for array in (a.suction_surface, a.pressure_surface, a.leading_edge_point,
                                         a.trailing_edge_point)]
    g = grid_of(spec, settings=replace(SETTINGS, n_passage=101, n_eta=21))
    assert g.m_le == a.leading_edge_point[0] and g.m_te == a.trailing_edge_point[0]
    fresh = make_cascade(spec).active
    for array, kept, again in zip((a.suction_surface, a.pressure_surface, a.leading_edge_point, a.trailing_edge_point),
                                  before, (fresh.suction_surface, fresh.pressure_surface, fresh.leading_edge_point,
                                           fresh.trailing_edge_point)):
        np.testing.assert_array_equal(array, kept)
        np.testing.assert_array_equal(array, again)


def test_a_grid_whose_regions_disagree_with_the_blade_cannot_exist():
    """Grid checks the topology itself: moving either end of the blade run by one station,
    or making a station that cuts the blade periodic, is refused."""
    g = grid_of(SPEC)
    for first, last in ((g.first_blade_i + 1, g.last_blade_i), (g.first_blade_i - 1, g.last_blade_i),
                        (g.first_blade_i, g.last_blade_i - 1), (g.first_blade_i, g.last_blade_i + 1)):
        with pytest.raises(gr.GridError, match="topology"):
            replace(g, first_blade_i=first, last_blade_i=last)


def test_a_grid_whose_walls_close_or_merge_cannot_exist():
    """Grid checks its nodes as well: an upper node moved onto the lower one closes the
    passage, and the last blade station's lower wall node moved onto the upper one less a
    pitch (the reference blade's bottom) leaves one point of the blade for both walls."""
    g = grid_of(SPEC)
    nodes = np.array(g.nodes)
    i = (g.first_blade_i + g.last_blade_i) // 2
    nodes[i, -1, 1] = nodes[i, 0, 1]
    with pytest.raises(gr.GridError, match="width is not positive"):
        replace(g, nodes=nodes)
    nodes = np.array(g.nodes)
    nodes[g.last_blade_i, 0, 1] = nodes[g.last_blade_i, -1, 1] - g.pitch
    with pytest.raises(gr.GridError, match="not two distinct points"):
        replace(g, nodes=nodes)


def test_the_metal_check_sees_a_periodic_line_through_the_nose():
    """The checker behind assert_topology flags the old station-37 topology: the periodic eta
    line 0.0003 c behind the LE, on the kappa_le tangent, holds blade metal twice, in the
    reference nose and in the neighbor's."""
    c = cascade_of(SPEC)
    a, g = c.active, grid_of(SPEC)
    m = g.m[37]
    assert 0.0 < m - g.m_le < 0.001
    y = a.leading_edge_point[1] + (m - g.m_le) * math.tan(c.kappa_le)
    pieces = metal_on(SPEC, m, y, y + a.pitch)
    assert len(pieces) == 2
    assert pieces[0][0] < y < pieces[0][1] and pieces[1][0] < y + a.pitch < pieces[1][1]


# --- Eta lines and n_eta independence ---------------------------------------------------

@pytest.mark.parametrize("spec", EDGE_CASES)
def test_n_eta_changes_nothing_but_the_interior_nodes(spec):
    """For one m grid, n_eta = 11, 21, 41, 81, 101 and 161 give the same stations, the same
    blade run and the same boundary nodes bit for bit, and every one of them passes the
    whole-line checks: the topology is decided on the station lines, not on the nodes."""
    reference = grid_of(spec, 30.0, replace(SETTINGS, n_passage=61, n_eta=11))
    for n_eta in (11, 21, 41, 81, 101, 161):
        g = grid_of(spec, 30.0, replace(SETTINGS, n_passage=61, n_eta=n_eta))
        np.testing.assert_array_equal(g.m, reference.m)
        assert (g.first_blade_i, g.last_blade_i) == (reference.first_blade_i, reference.last_blade_i)
        np.testing.assert_array_equal(g.lower_boundary, reference.lower_boundary)
        np.testing.assert_array_equal(g.upper_boundary, reference.upper_boundary)
        for name in ("lower", "upper"):                     # the same named surfaces, places and contour choices
            for item in fields(gr.WallSide):
                np.testing.assert_array_equal(getattr(getattr(g.wall_identity, name), item.name),
                                              getattr(getattr(reference.wall_identity, name), item.name))
        assert_topology(spec, g)


@pytest.mark.parametrize("spec", EDGE_CASES)
@pytest.mark.parametrize("n_passage, n_eta", ((31, 21), (101, 101), (141, 161)))
def test_no_node_lies_in_blade_metal(spec, n_passage, n_eta):
    """Interior nodes and periodic side nodes lie outside every blade of the row (wall nodes
    are on the walls, within WALL_TOLERANCE, and are checked by assert_topology)."""
    g = grid_of(spec, 30.0, replace(SETTINGS, n_passage=n_passage, n_eta=n_eta))
    tip, tail = blade_extent(spec)
    near = (g.m >= tip) & (g.m <= tail)              # elsewhere no blade reaches the station line
    interior = g.nodes[near, 1:-1].reshape(-1, 2)
    assert not np.any(in_metal(spec, interior))
    periodic = np.concatenate((g.nodes[:g.first_blade_i].reshape(-1, 2), g.nodes[g.last_blade_i + 1:].reshape(-1, 2)))
    close = (periodic[:, 0] >= tip - g.dm) & (periodic[:, 0] <= tail + g.dm)
    assert not np.any(in_metal(spec, periodic[close]))


@pytest.mark.parametrize("n_passage, n_eta, interior, side", ((31, 21, 0.022551, 0.027904), (101, 101, 0.002708, 0.004090)))
def test_the_non_wall_nodes_keep_clear_of_the_blade(n_passage, n_eta, interior, side):
    """Distances of the non-wall nodes from the nearest blade surface, SPEC: the closest is
    the first interior node (j = 1) at the first blade station, beside the steep nose of the
    suction side, 0.0226 c at 31 x 21 and 0.0027 c at 101 x 101 (eta spacing 0.01); the
    closest periodic side node is 0.0279 and 0.0041 c away."""
    g = grid_of(SPEC, settings=replace(SETTINGS, n_passage=n_passage, n_eta=n_eta))
    a = cascade_of(SPEC).active
    rows = [outline(SPEC) + k * a.periodic_translation for k in (-1, 0, 1, 2)]
    wall = np.zeros(g.nodes.shape[:2], dtype=bool)
    wall[g.first_blade_i:g.last_blade_i + 1, [0, -1]] = True
    periodic = np.zeros_like(wall)
    periodic[:, [0, -1]] = True
    periodic &= ~wall
    for mask, expected in ((~wall & ~periodic, interior), (periodic, side)):
        distance = np.min([distance_to(g.nodes[mask], row) for row in rows], axis=0)
        assert distance.min() == pytest.approx(expected, abs=1e-6)
    i, j = np.argwhere(~wall & ~periodic)[int(np.argmin(np.min([distance_to(g.nodes[~wall & ~periodic], row)
                                                                  for row in rows], axis=0)))]
    assert (i, j) == (g.first_blade_i, 1)


# --- Refinement ---------------------------------------------------------------------------

def test_the_sampled_throat_converges_to_the_geometric_throat():
    """The closest pair of a lower and an upper wall node approaches geometry.py's throat as
    dm^2. The nodes lie on the walls, so no pair is closer than the walls are: the error is
    never below the 9e-8 c by which the smooth walls' throat and geometry.py's cubic-spline
    throat differ, and it is at most about 3.7 dm^2, by where the stations fall against the
    throat's two ends (2.3, 1.8, 0.9 and 3.2 dm^2 here)."""
    a = cascade_of(SPEC).active
    errors = []
    for n_passage in (21, 41, 81, 161):
        g = grid_of(SPEC, settings=replace(SETTINGS, n_passage=n_passage, n_eta=5))
        walls = slice(g.first_blade_i, g.last_blade_i + 1)
        lower, upper = g.lower_boundary[walls], g.upper_boundary[walls]
        closest = np.sqrt(((lower[:, None] - upper[None]) ** 2).sum(-1)).min()
        errors.append(closest - a.throat_opening)
        assert -1e-6 < errors[-1] <= 4.0 * g.dm ** 2 + 1e-6
    assert errors[-1] < errors[0] / 20.0


def test_the_smallest_pitchwise_width_converges_from_above():
    """The smallest width S = upper - lower over the blade stations converges to the smallest
    vertical gap of the same blade at 1601 camber stations, 0.877940 c at m = 0.273 c, which
    is not the throat (0.7215 c): S runs along y, the throat across the passage. S is a
    parabola of curvature 1.47 / c at its minimum and a station lies within dm / 2 of it, so
    the sampled minimum is at most 1.47 dm^2 / 8 above; it can be up to 2.5e-6 c below, by
    which the smooth walls of the 101-station blade narrow that gap."""
    fine = cascade_of(FINE).active
    gaps = []
    for m in fine.leading_edge_point[0] + np.linspace(0.1, 0.5, 4001) * fine.chord:
        heights = line_cut(FINE, m)[0]
        gaps.append(heights[0] + fine.pitch - heights[-1])
    smallest = min(gaps)
    assert smallest == pytest.approx(0.877940, abs=1e-6)
    for n_passage in (21, 41, 81, 161):
        g = grid_of(SPEC, settings=replace(SETTINGS, n_passage=n_passage, n_eta=5))
        walls = slice(g.first_blade_i, g.last_blade_i + 1)
        sampled = (g.upper_boundary[walls, 1] - g.lower_boundary[walls, 1]).min()
        assert -4e-6 <= sampled - smallest <= 1.47 / 8.0 * g.dm ** 2 + 4e-6


@pytest.mark.parametrize("n_passage", (31, 101, 141))
def test_wall_nodes_lie_on_the_blade_at_1601_camber_stations(n_passage):
    """Against the same blade at 16 times the camber stations (its outline densified by a
    parametric PCHIP) every wall node is within 1e-5 c, nose and tail included."""
    fine = cascade_of(FINE).active
    loop = np.vstack((fine.suction_surface[::-1], fine.pressure_surface[1:]))
    t = np.concatenate(([0.0], np.cumsum(np.hypot(*np.diff(loop, axis=0).T))))
    dense_t = np.linspace(0.0, t[-1], 20 * len(t))
    dense = np.column_stack((PchipInterpolator(t, loop[:, 0])(dense_t), PchipInterpolator(t, loop[:, 1])(dense_t)))
    g = grid_of(SPEC, settings=replace(SETTINGS, n_passage=n_passage, n_eta=5))
    walls = slice(g.first_blade_i, g.last_blade_i + 1)
    nodes = np.vstack((g.lower_boundary[walls], g.upper_boundary[walls] - fine.periodic_translation))
    assert distance_to(nodes, dense).max() < 1.0e-5


def test_wall_nodes_converge_with_the_blade_resolution():
    """The wall error belongs to the Cascade's sampling, not to the grid: at a fixed m grid,
    101, 201, 401 and 801 camber stations put the wall nodes 8e-6, 2e-6, 5e-7 and 1e-7 c from
    the 1601-station blade."""
    fine = cascade_of(FINE).active
    loop = np.vstack((fine.suction_surface[::-1], fine.pressure_surface[1:]))
    t = np.concatenate(([0.0], np.cumsum(np.hypot(*np.diff(loop, axis=0).T))))
    dense_t = np.linspace(0.0, t[-1], 20 * len(t))
    dense = np.column_stack((PchipInterpolator(t, loop[:, 0])(dense_t), PchipInterpolator(t, loop[:, 1])(dense_t)))
    errors = []
    for n_camber_stations in (101, 201, 401, 801):
        spec = replace(SPEC, n_camber_stations=n_camber_stations)
        g = gr.make_grid(cascade_of(spec), replace(SETTINGS, n_passage=101, n_eta=5), math.radians(45.0))
        walls = slice(g.first_blade_i, g.last_blade_i + 1)
        nodes = np.vstack((g.lower_boundary[walls], g.upper_boundary[walls] - fine.periodic_translation))
        errors.append(distance_to(nodes, dense).max())
    assert errors[0] < 1e-5 and all(later < 0.6 * earlier for earlier, later in zip(errors, errors[1:]))


# --- Wall values and side lines -----------------------------------------------------------

@pytest.mark.parametrize("spec", CASES)
@pytest.mark.parametrize("inlet_angle_deg", INLET_ANGLES_DEG)
def test_inlet_stations_are_on_the_inlet_side_lines(spec, inlet_angle_deg):
    """Every inlet station lies ahead of the blade, and its nodes follow the inlet line
    through the exact LE point; the upper line is the lower one plus one pitch."""
    c = cascade_of(spec)
    g = grid_of(spec, inlet_angle_deg)
    inlet = slice(0, g.first_blade_i)
    reference = reference_line_y(g.m[inlet], c.active.leading_edge_point, c.kappa_le, g.m_in,
                                 math.radians(inlet_angle_deg))
    np.testing.assert_allclose(g.lower_boundary[inlet, 1], reference, rtol=0, atol=1e-13 * g.chord)
    np.testing.assert_array_equal(g.upper_boundary[inlet, 0], g.lower_boundary[inlet, 0])
    tolerance = 4.0 * np.spacing(np.maximum(np.abs(g.upper_boundary[inlet, 1]), g.pitch))
    assert np.all(np.abs(g.upper_boundary[inlet, 1] - g.lower_boundary[inlet, 1] - g.pitch) <= tolerance)


@pytest.mark.parametrize("spec", CASES)
@pytest.mark.parametrize("turn_deg", (None, *EXIT_TURNS_DEG))
def test_exit_stations_are_on_the_exit_side_lines(spec, turn_deg):
    c = cascade_of(spec)
    g = grid_of(spec, exit_turn_deg=turn_deg)
    exit_ = slice(g.last_blade_i + 1, None)
    reference = reference_line_y(g.m[exit_], c.active.trailing_edge_point, c.kappa_te, g.m_out, g.exit_angle)
    np.testing.assert_allclose(g.lower_boundary[exit_, 1], reference, rtol=0, atol=1e-13 * g.chord)
    tolerance = 4.0 * np.spacing(np.maximum(np.abs(g.upper_boundary[exit_, 1]), g.pitch))
    assert np.all(np.abs(g.upper_boundary[exit_, 1] - g.lower_boundary[exit_, 1] - g.pitch) <= tolerance)
    if turn_deg is None:        # exactly the straight kappa_te tangent
        line = c.active.trailing_edge_point[1] + (g.m[exit_] - g.m_te) * math.tan(c.kappa_te)
        np.testing.assert_array_equal(g.lower_boundary[exit_, 1], line)


@pytest.mark.parametrize("spec", EDGE_CASES)
@pytest.mark.parametrize("n_passage", (17, 31, 101))
def test_the_passage_width_is_positive_at_every_station(spec, n_passage):
    g = grid_of(spec, 30.0, replace(SETTINGS, n_passage=n_passage))
    width = g.upper_boundary[:, 1] - g.lower_boundary[:, 1]
    assert np.all(width > 0.0)
    walls = slice(g.first_blade_i, g.last_blade_i + 1)
    assert np.all(width[walls] < g.pitch)


# --- Geometry sweep -------------------------------------------------------------------------

SWEEP = (
    CascadeSpec(blade_type="NACA65", solidity=1.0, stagger_deg=33.0, cl0=1.2, t_max_over_c=0.10,
                surface_continuity="C1"),
    CascadeSpec(blade_type="NACA65", solidity=1.2, stagger_deg=50.0, cl0=0.8, t_max_over_c=0.06,
                surface_continuity="C1"),
    CascadeSpec(blade_type="NACA65", solidity=1.0, stagger_deg=15.0, cl0=1.6, t_max_over_c=0.04,
                profile_variant="emery_base", surface_continuity="C1"),
    CascadeSpec(blade_type="NACA65", solidity=1.5, stagger_deg=-20.0, cl0=0.4, t_max_over_c=0.06,
                profile_variant="emery_base", surface_continuity="C1"),
    CascadeSpec(blade_type="NACA65", solidity=0.8, stagger_deg=-30.0, cl0=1.2, t_max_over_c=0.10,
                surface_continuity="C1"),
    CascadeSpec(blade_type="NACA65", solidity=1.0, stagger_deg=0.0, cl0=0.4, t_max_over_c=0.04,
                surface_continuity="C1"),
    CascadeSpec(blade_type="NACA65", solidity=2.0, stagger_deg=45.0, cl0=1.0, t_max_over_c=0.08,
                surface_continuity="C1"),
    CascadeSpec(blade_type="NACA65", solidity=1.0, stagger_deg=33.0, cl0=1.2, t_max_over_c=0.10, physical_chord=0.05,
                surface_continuity="C1"),
    CascadeSpec(blade_type="NACA65", solidity=0.8, stagger_deg=-30.0, cl0=1.2, t_max_over_c=0.10, physical_chord=0.2,
                surface_continuity="C1"),
)


@pytest.mark.parametrize("spec", SWEEP)
@pytest.mark.parametrize("n_passage, n_eta", ((17, 11), (31, 21), (62, 41), (101, 101)))
@pytest.mark.parametrize("inlet_angle_deg, exit_turn_deg", ((30.0, None), (-10.0, 8.0)))
def test_every_grid_of_a_geometry_sweep_holds_the_hard_invariants(spec, n_passage, n_eta, inlet_angle_deg,
                                                                   exit_turn_deg):
    """Positive and negative stagger, cl0 0.4 to 1.6, t/c 0.04 to 0.10, solidity 0.8 to 2,
    both TE variants, chord 1 and physical chords, two inlet and exit angles and four
    resolutions: every grid builds and holds the hard invariants, and a physical chord gives
    the same topology as its chord-1 twin."""
    settings = replace(SETTINGS, n_passage=n_passage, n_eta=n_eta)
    g = grid_of(spec, inlet_angle_deg, settings, exit_turn_deg)
    assert_topology(spec, g)
    np.testing.assert_allclose(np.diff(g.m), g.dm, rtol=1e-12)
    if spec.physical_chord is not None:
        unit = grid_of(replace(spec, physical_chord=None), inlet_angle_deg, settings, exit_turn_deg)
        assert (len(g.m), g.first_blade_i, g.last_blade_i) == (len(unit.m), unit.first_blade_i, unit.last_blade_i)


# --- Nodes ------------------------------------------------------------------------------

@pytest.mark.parametrize("spec", CASES)
def test_every_node_of_a_station_is_at_that_station(spec):
    g = grid_of(spec)
    assert g.nodes.shape == (len(g.m), len(g.eta), 2)
    np.testing.assert_array_equal(g.nodes[..., 0], np.broadcast_to(g.m[:, None], g.nodes.shape[:2]))


@pytest.mark.parametrize("spec", CASES)
@pytest.mark.parametrize("turn_deg", (None, 5.0))
def test_the_interior_is_the_linear_map_between_the_walls(spec, turn_deg):
    g = grid_of(spec, exit_turn_deg=turn_deg)
    lower, upper = g.lower_boundary[:, 1], g.upper_boundary[:, 1]
    np.testing.assert_array_equal(g.nodes[:, 0], g.lower_boundary)
    np.testing.assert_array_equal(g.nodes[:, -1], g.upper_boundary)
    scale = np.maximum(np.abs(lower), np.abs(upper))
    for j, eta in enumerate(g.eta):
        expected = lower + eta * (upper - lower)
        assert np.all(np.abs(g.nodes[:, j, 1] - expected) <= 4.0 * np.spacing(scale)), j


# --- rebuild_exit on the grid ----------------------------------------------------------

def assert_same_grid(left: gr.Grid, right: gr.Grid) -> None:
    for item in fields(left):
        if item.name == "boundaries":
            for sub in fields(left.boundaries):
                np.testing.assert_array_equal(getattr(left.boundaries, sub.name), getattr(right.boundaries, sub.name),
                                              err_msg=sub.name)
        else:
            np.testing.assert_array_equal(getattr(left, item.name), getattr(right, item.name), err_msg=item.name)


@pytest.mark.parametrize("spec", CASES)
def test_the_initial_grid_has_exit_angle_kappa_te(spec):
    g = grid_of(spec)
    assert g.exit_angle == g.boundaries.exit_angle == cascade_of(spec).kappa_te
    assert g.inlet_angle == math.radians(45.0)


@pytest.mark.parametrize("spec", CASES)
def test_rebuilding_the_grid_exit_at_the_same_angle_changes_nothing(spec):
    g = grid_of(spec)
    again = gr.rebuild_exit(g, g.exit_angle)
    assert type(again) is gr.Grid
    assert_same_grid(again, g)


@pytest.mark.parametrize("spec", CASES)
@pytest.mark.parametrize("turn_deg", EXIT_TURNS_DEG)
def test_rebuilding_the_grid_exit_moves_only_the_exit_nodes(spec, turn_deg):
    start = grid_of(spec)
    rebuilt = gr.rebuild_exit(start, exit_angle_of(spec, turn_deg))
    np.testing.assert_array_equal(rebuilt.m, start.m)
    np.testing.assert_array_equal(rebuilt.eta, start.eta)
    assert (rebuilt.first_blade_i, rebuilt.last_blade_i) == (start.first_blade_i, start.last_blade_i)
    kept = slice(0, start.last_blade_i + 1)
    np.testing.assert_array_equal(rebuilt.nodes[kept], start.nodes[kept])
    np.testing.assert_array_equal(rebuilt.nodes[..., 0], start.nodes[..., 0])
    assert np.all(rebuilt.nodes[start.last_blade_i + 1:, :, 1] != start.nodes[start.last_blade_i + 1:, :, 1])
    assert rebuilt.exit_angle == exit_angle_of(spec, turn_deg)


@pytest.mark.parametrize("spec", CASES)
@pytest.mark.parametrize("turn_deg", EXIT_TURNS_DEG)
def test_a_rebuilt_grid_is_a_fresh_grid_at_that_angle(spec, turn_deg):
    assert_same_grid(gr.rebuild_exit(grid_of(spec), exit_angle_of(spec, turn_deg)),
                     grid_of(spec, exit_turn_deg=turn_deg))


@pytest.mark.parametrize("spec", CASES)
def test_a_grid_returns_to_the_start_after_a_sequence_of_exit_angles(spec):
    start = grid_of(spec)
    kappa_te = cascade_of(spec).kappa_te
    state = start
    for angle in (kappa_te + math.radians(2.0), kappa_te + math.radians(5.0), kappa_te):
        state = gr.rebuild_exit(state, angle)
    assert_same_grid(state, start)


def test_many_artificial_kutta_updates_stay_stable():
    """Twenty rebuilds through exit angles from kappa_te - 10 to + 10 deg and back: every
    grid equals a fresh build at its angle, and the inlet and blade nodes never move."""
    start = grid_of(SPEC)
    kappa_te = cascade_of(SPEC).kappa_te
    state = start
    kept = slice(0, start.last_blade_i + 1)
    for turn in np.concatenate((np.linspace(0.0, 10.0, 6), np.linspace(10.0, -10.0, 11), np.linspace(-10.0, 0.0, 3))):
        angle = kappa_te + math.radians(float(turn))
        state = gr.rebuild_exit(state, angle)
        assert_same_grid(state, gr.make_grid(cascade_of(SPEC), SETTINGS, math.radians(45.0), angle))
        np.testing.assert_array_equal(state.nodes[kept], start.nodes[kept])
    assert_same_grid(state, start)


@pytest.mark.parametrize("spec", (SPEC, OTHER, LONG_TAIL))
def test_rebuilt_exits_keep_the_topology_and_clear_exit_lines(spec):
    """On the 101 x 101 grid, exit angles kappa_te + 2, + 5, - 5, 0, then - 15 .. + 15 deg:
    the stations, eta, the blade run and every inlet and wall node stay bit for bit, the exit
    lines stay exact translates and clear of the blade, the width stays positive, and the
    grid comes back bit for bit."""
    settings = replace(SETTINGS, n_passage=101, n_eta=101)
    start = grid_of(spec, 30.0, settings)
    kappa_te = cascade_of(spec).kappa_te
    kept = slice(0, start.last_blade_i + 1)
    state = start
    for turn in (2.0, 5.0, -5.0, 0.0, *np.arange(-15.0, 16.0, 3.0), 0.0):
        state = gr.rebuild_exit(state, kappa_te + math.radians(float(turn)))
        np.testing.assert_array_equal(state.m, start.m)
        np.testing.assert_array_equal(state.eta, start.eta)
        assert (state.first_blade_i, state.last_blade_i) == (start.first_blade_i, start.last_blade_i)
        np.testing.assert_array_equal(state.nodes[kept], start.nodes[kept])
        assert np.all(state.upper_boundary[:, 1] - state.lower_boundary[:, 1] > 0.0)
        for i in range(state.first_exit_i, min(state.first_exit_i + 3, len(state.m))):
            lower, upper = state.nodes[i, 0, 1], state.nodes[i, -1, 1]
            assert upper - lower == pytest.approx(state.pitch, abs=4.0 * np.spacing(max(abs(upper), state.pitch)))
            assert not metal_on(spec, state.m[i], lower, upper)
    assert_same_grid(state, start)


# --- Scaling ---------------------------------------------------------------------------

def test_a_physical_chord_scales_every_length_and_no_angle():
    unit, metric = boundaries(SPEC, exit_turn_deg=5.0), boundaries(DIMENSIONAL, exit_turn_deg=5.0)
    scale = DIMENSIONAL.physical_chord
    for name in ("chord", "pitch", "m_in", "m_le", "m_te", "m_out"):
        assert getattr(metric, name) == pytest.approx(scale * getattr(unit, name), rel=1e-13, abs=1e-17), name
    for name in ("periodic_translation", "lower_blade_boundary", "upper_blade_boundary", "upstream_m",
                 "upstream_lower_boundary", "upstream_upper_boundary", "downstream_m", "downstream_lower_boundary",
                 "downstream_upper_boundary"):
        np.testing.assert_allclose(getattr(metric, name), scale * getattr(unit, name), rtol=1e-13, atol=1e-17,
                                   err_msg=name)
    for name in ("inlet_angle", "exit_angle", "upstream_beta", "downstream_beta"):
        np.testing.assert_array_equal(getattr(metric, name), getattr(unit, name), err_msg=name)


@pytest.mark.parametrize("n_passage, n_eta", ((31, 21), (101, 101)))
@pytest.mark.parametrize("scale", (0.05, 0.2))
def test_a_physical_chord_scales_the_grid_but_not_its_topology(n_passage, n_eta, scale):
    """Every coordinate, dm, the pitch, the edge offsets and the extension lengths scale with
    the chord; eta, the angles, the station indices and the stations in chords do not."""
    settings = replace(SETTINGS, n_passage=n_passage, n_eta=n_eta)
    unit, metric = grid_of(SPEC, 45.0, settings, 5.0), grid_of(replace(SPEC, physical_chord=scale), 45.0, settings, 5.0)
    np.testing.assert_allclose(metric.nodes, scale * unit.nodes, rtol=1e-12, atol=1e-16 * scale)
    np.testing.assert_allclose(metric.m, scale * unit.m, rtol=1e-13, atol=1e-15 * scale)    # round-off of the range
    for name in ("dm", "pitch", "chord", "actual_le_offset", "actual_te_offset", "m_first_blade", "m_last_blade"):
        assert getattr(metric, name) == pytest.approx(scale * getattr(unit, name), rel=1e-11, abs=1e-15 * scale), name
    assert metric.m_le - metric.m_in == pytest.approx(scale * (unit.m_le - unit.m_in), rel=1e-13)
    assert metric.m_out - metric.m_te == pytest.approx(scale * (unit.m_out - unit.m_te), rel=1e-13)
    np.testing.assert_array_equal(metric.eta, unit.eta)
    assert (metric.first_blade_i, metric.last_blade_i, len(metric.m)) == (unit.first_blade_i, unit.last_blade_i,
                                                                          len(unit.m))
    assert (metric.inlet_angle, metric.exit_angle) == (unit.inlet_angle, unit.exit_angle)
    np.testing.assert_allclose((metric.m - metric.m_le) / metric.chord, (unit.m - unit.m_le) / unit.chord,
                               rtol=0, atol=1e-13)


# --- Output objects ----------------------------------------------------------------------

def test_the_boundaries_expose_exactly_these_fields():
    """blade_curves (2026-11-11) is the C2 cascade's active_curves by reference, None for C1."""
    assert [item.name for item in fields(gr.PassageBoundaries)] == [
        "chord", "pitch", "periodic_translation", "inlet_angle", "exit_angle", "m_in", "m_le", "m_te", "m_out",
        "lower_blade_boundary", "upper_blade_boundary", "blade_curves",
        "upstream_m", "upstream_beta", "upstream_lower_boundary", "upstream_upper_boundary",
        "downstream_m", "downstream_beta", "downstream_lower_boundary", "downstream_upper_boundary",
    ]
    assert [item.name for item in fields(gr.WallSide)] == ["surface", "parameter", "u", "contour", "hidden"]


def test_the_grid_holds_coordinates_only():
    """Fields are the boundaries, stations, eta, nodes and the blade-station indices; the
    landmarks, lengths and angles are read from the boundaries. No slope, metric or flow
    field, not even as a placeholder."""
    assert [item.name for item in fields(gr.Grid)] == ["boundaries", "m", "eta", "nodes", "first_blade_i",
                                                       "last_blade_i"]
    g = grid_of(SPEC)
    b = g.boundaries
    for name in ("m_in", "m_out", "m_le", "m_te", "chord", "pitch", "inlet_angle", "exit_angle"):
        assert getattr(g, name) == getattr(b, name), name
    assert g.last_inlet_i == g.first_blade_i - 1 and g.first_exit_i == g.last_blade_i + 1
    assert g.actual_le_offset == g.m[g.first_blade_i] - g.m_le
    assert g.actual_te_offset == g.m[g.last_blade_i] - g.m_te


def test_numbering_and_theta_are_still_unimplemented():
    with pytest.raises(NotImplementedError):
        gr.node_index(0, 0, 21)
    with pytest.raises(NotImplementedError):
        gr.node_theta(grid_of(SPEC))


def test_boundaries_and_grid_are_frozen_and_read_only():
    for built in (boundaries(SPEC), grid_of(SPEC)):
        for item in fields(built):
            value = getattr(built, item.name)
            if isinstance(value, np.ndarray):
                assert not value.flags.writeable, item.name
                with pytest.raises(ValueError):
                    value[0] = 0.0
        with pytest.raises(AttributeError):
            built.m_in = 0.0
    g = grid_of(SPEC)
    for view in (g.lower_boundary, g.upper_boundary, g.region, g.wall_identity.stations):
        assert not view.flags.writeable
    for side in (g.wall_identity.lower, g.wall_identity.upper):
        assert all(not getattr(side, item.name).flags.writeable for item in fields(side))
        with pytest.raises(AttributeError):
            side.contour = None
    assert type(g.first_blade_i) is int and type(g.last_blade_i) is int
    assert all(type(getattr(boundaries(SPEC), name)) is float for name in ("inlet_angle", "exit_angle", "m_out"))


# --- Inputs ----------------------------------------------------------------------------

@pytest.mark.parametrize("inlet_angle", INVALID_ANGLES)
def test_inlet_angle_must_be_radians_inside_plus_minus_ninety_degrees(inlet_angle):
    for build in (gr.make_passage_boundaries, gr.make_grid):
        with pytest.raises(ValueError, match="inlet_angle"):
            build(cascade_of(SPEC), SETTINGS, inlet_angle)


@pytest.mark.parametrize("exit_angle", INVALID_ANGLES)
def test_exit_angle_must_be_radians_inside_plus_minus_ninety_degrees(exit_angle):
    for build in (gr.make_passage_boundaries, gr.make_grid):
        with pytest.raises(ValueError, match="exit_angle"):
            build(cascade_of(SPEC), SETTINGS, math.radians(45.0), exit_angle)
    for item in (boundaries(SPEC), grid_of(SPEC)):
        with pytest.raises(ValueError, match="exit_angle"):
            gr.rebuild_exit(item, exit_angle)


@pytest.mark.parametrize("changes", ({"upstream_extension_chords": 0.0}, {"upstream_extension_chords": -1.0},
                                     {"upstream_extension_chords": math.nan}, {"downstream_extension_chords": 0.0},
                                     {"downstream_extension_chords": -0.5}, {"downstream_extension_chords": math.inf},
                                     {"n_passage": 0}, {"n_passage": 2.5}, {"n_passage": True}))
def test_invalid_boundary_settings_are_refused(changes):
    for build in (gr.make_passage_boundaries, gr.make_grid):
        with pytest.raises(ValueError):
            build(cascade_of(SPEC), replace(SETTINGS, **changes), math.radians(45.0))


@pytest.mark.parametrize("changes", ({"n_eta": 1}, {"n_eta": 0}, {"n_eta": 21.0}, {"n_eta": True}))
def test_invalid_grid_settings_are_refused(changes):
    with pytest.raises(ValueError):
        gr.make_grid(cascade_of(SPEC), replace(SETTINGS, **changes), math.radians(45.0))


def test_a_blade_that_falls_between_two_stations_is_a_grid_error():
    """At one passage cell (3 cells over the whole domain, dm 0.946 c) the stations land
    0.054 c ahead of the LE (0.049 c ahead of the nose tip) and 0.054 c behind the TE and its
    1.3e-5 c tail: no station cuts the blade."""
    with pytest.raises(gr.GridError, match="no station cuts the blade"):
        gr.make_grid(cascade_of(SPEC), replace(SETTINGS, n_passage=1), math.radians(45.0))


def test_only_a_cascade_and_grid_settings_are_accepted():
    for build in (gr.make_passage_boundaries, gr.make_grid):
        with pytest.raises(TypeError):
            build(SPEC, SETTINGS, 0.5)
        with pytest.raises(TypeError):
            build(cascade_of(SPEC), {"n_passage": 31, "n_eta": 21}, 0.5)
    with pytest.raises(TypeError):
        gr.rebuild_exit(cascade_of(SPEC), 0.3)


def test_a_blade_angle_at_ninety_degrees_is_refused():
    steep = cascade_of(replace(SPEC, stagger_deg=75.0))
    assert steep.kappa_le > math.pi / 2
    with pytest.raises(ValueError, match="kappa_le"):
        gr.make_passage_boundaries(steep, SETTINGS, math.radians(45.0))


def test_grid_reads_a_ready_cascade_and_nothing_else():
    """grid.py imports no plotting or project package beyond config and geometry (and
    scipy's PCHIP) at module level, takes only Cascade from geometry, never reads the
    CascadeSpec or the normalized lengths, and finds no landmark by min/max of x. The one
    exception, added 2026-11-13 (section 9): validate_standard_grid's checks import
    metrics.build_metrics inside the function, to refuse a standard grid whose metrics
    metrics.py would not build (metrics imports grid, so never at module level)."""
    tree = ast.parse(Path(gr.__file__).read_text(encoding="utf-8"))

    def imports(nodes):
        found = {alias.name for node in nodes if isinstance(node, ast.Import) for alias in node.names}
        found |= {node.module for node in nodes if isinstance(node, ast.ImportFrom) and node.module}
        return {name for name in found if name.split(".")[0] not in sys.stdlib_module_names}

    assert imports(tree.body) == {"numpy", "numpy.typing", "scipy.interpolate", "turbolenza.b2b.config",
                                  "turbolenza.b2b.geometry"}
    local = imports([node for function in ast.walk(tree) if isinstance(function, ast.FunctionDef)
                     for node in ast.walk(function)])
    assert local == {"turbolenza.b2b.metrics"}
    assert imports(list(ast.walk(tree))) == imports(tree.body) | local
    from_geometry = {alias.name for node in ast.walk(tree)
                     if isinstance(node, ast.ImportFrom) and node.module == "turbolenza.b2b.geometry"
                     for alias in node.names}
    assert from_geometry == {"Cascade"}
    attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert not {"spec", "normalized"} & attributes
    assert not {"CascadeSpec", "make_cascade", "naca65_profile"} & names
    assert not {"min", "max", "argmin", "argmax", "amin", "amax", "nanmin", "nanmax"} & attributes


# --- C1 regression lock ---------------------------------------------------------------------

C1_GRID_FINGERPRINT = {     # 101 x 101 at inlet 45 deg, taken from grid.py before the C2 hand-off (2026-11-11)
    "SPEC": {"stations": (343, 120, 221),
             "node_sums": (22733.467392841238, 57.77815088514863, 392.3895202602224),
             "first_walls": (0.01536757795962444, 1.0053489041693877),
             "last_walls": (0.5512869457942922, 1.5370613284703447),
             "contour_sums": (57.96080636981634, 52.68681882926347)},
    "OTHER": {"stations": (317, 108, 208),
              "node_sums": (2264.79956070725, -81.14042058154772, 125.98793663515661),
              "first_walls": (0.0035285296231439304, 0.6609599297124622),
              "last_walls": (-0.33899857931240857, 0.32732879708886486),
              "contour_sums": (49.9897499788154, 50.7573616180335)},
}


@pytest.mark.parametrize("name, spec", (("SPEC", SPEC), ("OTHER", OTHER)))
def test_c1_grid_fingerprint(name, spec):
    """A C1 cascade's grid is the validated C1 grid: stations, blade run, node and wall sums,
    the first and last wall nodes and the contour coordinates as before the C2 hand-off
    (every array was also compared bit for bit on 55 C1 grids when it was made). A C1 grid
    has no authoritative curves: blade_curves None, wall u NaN, no curve derivatives."""
    g = grid_of(spec, settings=replace(SETTINGS, n_passage=101, n_eta=101))
    f, l = g.first_blade_i, g.last_blade_i
    expected = C1_GRID_FINGERPRINT[name]
    assert (len(g.m), f, l) == expected["stations"]
    measured = {"node_sums": (g.nodes[..., 1].sum(), g.nodes[:, 0, 1].sum(), g.nodes[:, -1, 1].sum()),
                "first_walls": (g.nodes[f, 0, 1], g.nodes[f, -1, 1]), "last_walls": (g.nodes[l, 0, 1], g.nodes[l, -1, 1]),
                "contour_sums": (g.wall_identity.lower.contour.sum(), g.wall_identity.upper.contour.sum())}
    for key, value in measured.items():
        np.testing.assert_allclose(value, expected[key], rtol=1e-13, atol=1e-16, err_msg=key)
    assert g.boundaries.blade_curves is None and not g.has_wall_curves
    assert np.all(np.isnan(g.wall_identity.lower.u)) and np.all(np.isnan(g.wall_identity.upper.u))
    with pytest.raises(gr.GridError, match="sampled surfaces"):
        g.wall_curve_derivative(1)


# --- C2: the walls are the authoritative curves (grid.py section 7) --------------------------

C2 = replace(SPEC, surface_continuity="C2")
C2_OTHER = replace(OTHER, surface_continuity="C2")              # negative stagger, sharp TE: nose tip on the pressure side
C2_SUCTION_TAIL = replace(SUCTION_TAIL, surface_continuity="C2")  # negative stagger, Kovach: tail end on the suction side
C2_STEEP = replace(STEEP, surface_continuity="C2")              # round numbers: a station on the exact LE and one on m_te
C2_DIMENSIONAL = replace(C2, physical_chord=0.05)
C2_RESOLUTIONS = (31, 51, 81, 101, 141, 201, 401)
C2_ROOT_TOL = 1e-14         # chords and u: grid.py's crossings against brentq's (measured at most 1.0e-15 over 120 grids)


class C2Blade:
    """The reference blade of a C2 cascade as its two curves, read here independently of
    grid.py: monotone pieces of m(u) from 20001 samples plus the knots, each turning point
    refined by scipy's bounded minimize_scalar, crossings by brentq, each piece counted on
    its half-open m range [min, max) (grid.py section 3: a line through a point where the
    contour passes on meets it once, through a tip twice, through a tail end not at all)."""

    def __init__(self, cascade: Cascade) -> None:
        self.cascade, self.pitch, self.chord = cascade, cascade.active.pitch, cascade.active.chord
        self.curves = {"suction": cascade.active_curves.suction, "pressure": cascade.active_curves.pressure}
        self.length = {name: curve.reference_length for name, curve in self.curves.items()}
        self.pieces = {}
        for name, curve in self.curves.items():
            u = np.unique(np.concatenate((np.linspace(0.0, 1.0, 20001), curve.interior_knots)))
            step = np.sign(np.diff(curve.evaluate(u)[:, 0]))
            turns = []
            for k in np.flatnonzero(step[1:] != step[:-1]):
                sign = 1.0 if step[k] < 0 else -1.0
                turns.append(minimize_scalar(lambda x: sign * float(curve.evaluate(x)[0]), bounds=(u[k], u[k + 2]),
                                             method="bounded", options={"xatol": 1e-15}).x)
            splits = [0.0, *turns, 1.0]
            self.pieces[name] = list(zip(splits[:-1], splits[1:]))

    def m_of(self, name: str, u: float) -> float:
        return float(self.curves[name].evaluate(u)[0])

    def crossings(self, m: float) -> list[tuple[float, str, float]]:
        """(y, curve, u) of every crossing of the line x = m, ascending y."""
        out = []
        for name, curve in self.curves.items():
            for a, b in self.pieces[name]:
                ma, mb = self.m_of(name, a), self.m_of(name, b)
                if not min(ma, mb) <= m < max(ma, mb):
                    continue
                u = a if m == ma else b if m == mb else brentq(lambda x: self.m_of(name, x) - m, a, b, xtol=1e-16)
                out.append((float(curve.evaluate(u)[1]), name, u))
        return sorted(out)

    def ends(self) -> tuple[tuple[float, str, float], tuple[float, str, float]]:
        """(m, curve, u) of the most upstream point (nose tip) and the most downstream one."""
        points = [(self.m_of(name, x), name, x) for name in self.curves for a, b in self.pieces[name] for x in (a, b)]
        return min(points), max(points)

    def metal(self, m: float) -> float:
        ys = [y for y, _, _ in self.crossings(m)]
        return float(sum(ys[k + 1] - ys[k] for k in range(0, len(ys) - 1, 2)))

    def metal_on(self, m: float, y0: float, y1: float) -> list[tuple[float, float]]:
        """Metal of the reference blade and its copies -2 to +3 pitches away on x = m between
        y0 and y1, pieces longer than TOUCH chords."""
        ys = [y for y, _, _ in self.crossings(m)]
        return [(lo + k * self.pitch, hi + k * self.pitch) for k in range(-2, 4) for lo, hi in zip(ys[0::2], ys[1::2])
                if lo + k * self.pitch < y1 and hi + k * self.pitch > y0 and hi - lo > TOUCH * self.chord]

    def tau(self, name: str, u: float) -> float:
        """The contour coordinate of grid.py section 7, written here."""
        return (1.0 - u) * self.length["suction"] if name == "suction" else self.length["suction"] + u * self.length["pressure"]

    def m_at_tau(self, tau: float) -> float:
        if tau <= self.length["suction"]:
            return self.m_of("suction", min(1.0, max(0.0, 1.0 - tau / self.length["suction"])))
        return self.m_of("pressure", min(1.0, max(0.0, (tau - self.length["suction"]) / self.length["pressure"])))


@lru_cache(maxsize=None)
def c2_blade(spec: CascadeSpec) -> C2Blade:
    return C2Blade(cascade_of(spec))


def c2_grid(spec: CascadeSpec, n_passage: int, n_eta: int = 11, inlet_angle_deg: float = 45.0,
            exit_turn_deg: float | None = None) -> gr.Grid:
    return grid_of(spec, inlet_angle_deg, replace(SETTINGS, n_passage=n_passage, n_eta=n_eta), exit_turn_deg)


@lru_cache(maxsize=None)
def sampled_reading(spec: CascadeSpec) -> Cascade:
    """The C2 cascade without its curves: grid.py then reads the sampled arrays through the
    C1 path (parametric PCHIP), which is exactly the C2 grid before the hand-off."""
    return replace(cascade_of(spec), active_curves=None, normalized_curves=None)


def assert_c2_topology(spec: CascadeSpec, g: gr.Grid) -> None:
    """The hard invariants of a C2 grid against C2Blade: exactly the stations whose line cuts
    the curves are blade stations; the eta line of a periodic station meets no metal, and a
    blade station's wall-to-wall line none either; the lower node is the top crossing and
    the upper node, one pitch down, the bottom one, on the curve and u grid.py records; each
    wall's contour coordinate never decreases, and the contour between two neighbouring
    wall nodes stays between their station lines unless a fold hides crossings there (no
    branch is jumped, no stretch skipped)."""
    blade = c2_blade(spec)
    tip, tail = blade.ends()
    tolerance = C2_ROOT_TOL * blade.chord
    f, l = g.first_blade_i, g.last_blade_i
    identity = g.wall_identity
    for i, m in enumerate(g.m):
        crossings = blade.crossings(m) if tip[0] <= m <= tail[0] else []
        metal = blade.metal(m) if crossings else 0.0
        lower, upper = g.nodes[i, 0, 1], g.nodes[i, -1, 1]
        assert upper - lower > 0.0, i
        if f <= i <= l:
            assert metal > TOUCH * blade.chord, i
            top, bottom = crossings[-1], crossings[0]
            assert not blade.metal_on(m, top[0] + tolerance, bottom[0] + blade.pitch - tolerance), i
            for wall, node, crossing in ((identity.lower, lower, top), (identity.upper, upper - blade.pitch, bottom)):
                k = i - f
                assert wall.surface[k] == crossing[1], (i, wall.surface[k], crossing[1])
                assert abs(node - crossing[0]) <= tolerance and abs(wall.u[k] - crossing[2]) <= C2_ROOT_TOL, i
        else:
            assert metal <= TOUCH * blade.chord, i
            assert not blade.metal_on(m, lower, upper), i
    total = blade.length["suction"] + blade.length["pressure"]
    m = g.m[f:l + 1]
    for wall, forward in ((identity.lower, -1.0), (identity.upper, 1.0)):
        assert np.all(np.diff(wall.contour) >= -TOUCH * blade.chord)
        tau = [blade.tau(name, u) for name, u in zip(wall.surface, wall.u)]
        for k in range(len(tau) - 1):
            span = (forward * (tau[k + 1] - tau[k])) % total
            walked = [blade.m_at_tau((tau[k] + forward * s) % total) for s in np.linspace(0.0, span, 41)]
            between = all(m[k] - TOUCH * blade.chord <= x <= m[k + 1] + TOUCH * blade.chord for x in walked)
            assert between or wall.hidden[k] > 0 or wall.hidden[k + 1] > 0, (f + k, forward)


def test_c2_boundaries_hold_the_cascade_curves_by_reference():
    """blade_curves is the C2 cascade's active_curves object itself (geometry.py owns the
    curves; nothing is copied or refitted), the sampled arrays are handed on unchanged as
    for C1, and rebuild_exit keeps the reference."""
    c, g = cascade_of(C2), c2_grid(C2, 31)
    assert g.boundaries.blade_curves is c.active_curves and g.has_wall_curves
    np.testing.assert_array_equal(g.boundaries.lower_blade_boundary, c.active.suction_surface)
    np.testing.assert_array_equal(g.boundaries.upper_blade_boundary, c.active.pressure_surface + c.active.periodic_translation)
    assert gr.rebuild_exit(g, c.kappa_te + 0.05).boundaries.blade_curves is c.active_curves
    with pytest.raises(TypeError, match="blade_curves"):
        replace(g.boundaries, blade_curves="suction and pressure")


@pytest.mark.parametrize("n_passage", C2_RESOLUTIONS)
def test_c2_wall_nodes_lie_on_the_authoritative_curves(n_passage):
    """Every C2 wall node is the curve point at the station: its y is brentq's crossing of
    the passage-facing branch (top for the lower wall, bottom one pitch down for the upper)
    to 1e-14 c, its u brentq's root to 1e-14 (measured at most 1.0e-15 for both over 120
    grids), and m(u) is the station m to round-off: at most 2e-15 times the m scale
    (measured 4.4e-16 c)."""
    g = c2_grid(C2, n_passage)
    assert_c2_topology(C2, g)
    curves = cascade_of(C2).active_curves
    f, l = g.first_blade_i, g.last_blade_i
    for wall in (g.wall_identity.lower, g.wall_identity.upper):
        for name in ("suction", "pressure"):
            on = wall.surface == name
            if on.any():
                residual = getattr(curves, name).evaluate(wall.u[on])[:, 0] - g.m[f:l + 1][on]
                assert np.abs(residual).max() <= 2e-15 * max(abs(g.m_in), abs(g.m_out))


@pytest.mark.parametrize("spec", (C2, C2_OTHER))
def test_c2_stored_u_reproduces_the_wall_coordinates(spec):
    """For every wall node, the curve its wall_identity names, evaluated directly at its u,
    gives the node's y bit for bit (plus the pitch for the upper wall), and
    wall_curve_derivative(k) is that curve's derivative(u, k) at the same u for k = 0, 1, 2:
    one curve, one u, position and derivatives alike."""
    g = c2_grid(spec, 101)
    curves = cascade_of(spec).active_curves
    f, l = g.first_blade_i, g.last_blade_i
    for k in (0, 1, 2):
        handed = g.wall_curve_derivative(k)
        for wall, row, shift, value in ((g.wall_identity.lower, 0, 0.0, handed[0]),
                                        (g.wall_identity.upper, -1, g.pitch, handed[1])):
            direct = np.array([getattr(curves, name).derivative(u, k) for name, u in zip(wall.surface, wall.u)])
            if k == 0:
                np.testing.assert_array_equal(direct[:, 1] + shift, g.nodes[f:l + 1, row, 1])
                direct[:, 1] += shift
            np.testing.assert_array_equal(value, direct)
            assert not value.flags.writeable


@pytest.mark.parametrize("spec", (C2, C2_OTHER))
def test_c2_a_station_in_the_rounded_nose_takes_both_roots_on_the_nose_curve(spec):
    """A station halfway between the nose tip and the LE point (C2: suction nose, 0.0045 c
    ahead of m_le; C2_OTHER, negative stagger: pressure nose) meets only the curve the tip
    lies on, twice, one root on each side of the tip's u. The higher root is the lower wall
    and the lower one the upper wall (on the +pitch neighbor): the passage-facing contours,
    so one wall carries the raw name of the other side there (C2: the upper wall on the
    neighbor's suction nose underside; C2_OTHER: the lower wall on the pressure nose)."""
    blade = c2_blade(spec)
    (m_tip, curve, u_tip), _ = blade.ends()
    m_le = cascade_of(spec).active.leading_edge_point[0]
    target = 0.5 * (m_tip + m_le)
    settings = stations_with_one_at(spec, target, replace(SETTINGS, n_eta=5))
    g = grid_of(spec, 45.0, settings)
    i = int(np.argmin(np.abs(g.m - target)))
    assert abs(g.m[i] - target) < 1e-12 and i == g.first_blade_i
    crossings = blade.crossings(g.m[i])
    assert [name for _, name, _ in crossings] == [curve, curve]
    (y_low, _, u_low), (y_high, _, u_high) = crossings
    assert min(u_low, u_high) < u_tip < max(u_low, u_high)
    lower, upper = g.wall_identity.lower, g.wall_identity.upper
    assert (lower.surface[0], upper.surface[0]) == (curve, curve)
    assert lower.u[0] == pytest.approx(u_high, abs=C2_ROOT_TOL) and upper.u[0] == pytest.approx(u_low, abs=C2_ROOT_TOL)
    assert g.nodes[i, 0, 1] == pytest.approx(y_high, abs=C2_ROOT_TOL) and \
        g.nodes[i, -1, 1] - g.pitch == pytest.approx(y_low, abs=C2_ROOT_TOL)
    usual = {"lower": "suction", "upper": "pressure"}
    renamed = [side for side, wall in (("lower", lower), ("upper", upper)) if wall.surface[0] != usual[side]]
    assert renamed == (["upper"] if curve == "suction" else ["lower"])
    assert_c2_topology(spec, g)


@pytest.mark.parametrize("spec, curve", ((C2, "pressure"), (C2_SUCTION_TAIL, "suction")))
def test_c2_a_station_in_a_kovach_tail_takes_both_roots_on_the_tail_curve(spec, curve):
    """A station halfway between the TE point and the tail end of a Kovach TE (C2: pressure
    tail, 1.35e-5 c behind m_te; C2_SUCTION_TAIL: suction tail) is the last blade station and
    meets only the tail's curve, twice, on both sides of the tail end's u: the higher root
    is the lower wall, the lower root the upper wall, whatever their raw names."""
    blade = c2_blade(spec)
    _, (m_tail, tail_curve, u_tail) = blade.ends()
    assert tail_curve == curve
    m_te = cascade_of(spec).active.trailing_edge_point[0]
    target = 0.5 * (m_te + m_tail)
    settings = stations_with_one_at(spec, target, replace(SETTINGS, n_eta=5))
    g = grid_of(spec, 45.0, settings)
    i = int(np.argmin(np.abs(g.m - target)))
    assert abs(g.m[i] - target) < 1e-12 and i == g.last_blade_i
    crossings = blade.crossings(g.m[i])
    assert [name for _, name, _ in crossings] == [curve, curve]
    (y_low, _, u_low), (y_high, _, u_high) = crossings
    assert min(u_low, u_high) < u_tail < max(u_low, u_high)
    lower, upper = g.wall_identity.lower, g.wall_identity.upper
    assert (lower.surface[-1], upper.surface[-1]) == (curve, curve)
    assert lower.u[-1] == pytest.approx(u_high, abs=C2_ROOT_TOL) and upper.u[-1] == pytest.approx(u_low, abs=C2_ROOT_TOL)
    assert_c2_topology(spec, g)


@pytest.mark.parametrize("spec", (C2, C2_OTHER, C2_SUCTION_TAIL))
@pytest.mark.parametrize("n_passage", (21, 31, 41, 61, 81, 101, 141))
def test_c2_first_and_last_blade_stations_follow_the_curve_extent(spec, n_passage):
    """Aungier's topology on the curves: the first blade station is the first behind the
    nose tip and the last the last ahead of the tail end, both cutting the curves, and their
    neighbours clear of the blade; nothing ties them to m_le or m_te."""
    blade = c2_blade(spec)
    (m_tip, _, _), (m_tail, _, _) = blade.ends()
    g = c2_grid(spec, n_passage, 5)
    f, l = g.first_blade_i, g.last_blade_i
    assert g.m[f - 1] <= m_tip < g.m[f] and g.m[l] < m_tail <= g.m[l + 1]
    assert blade.metal(g.m[f]) > TOUCH * blade.chord and blade.metal(g.m[l]) > TOUCH * blade.chord
    assert not blade.crossings(g.m[f - 1]) or blade.metal(g.m[f - 1]) <= TOUCH * blade.chord
    assert not blade.crossings(g.m[l + 1]) or blade.metal(g.m[l + 1]) <= TOUCH * blade.chord


def test_c2_a_station_past_the_last_sample_but_inside_the_curve_is_a_blade_station():
    """The curve's nose tip lies 2.56e-6 c ahead of the most upstream of its own samples
    (1.9e-6 c with the legacy PCHIP thickness). A station between the two cuts the curve (by
    4.23e-4 c of metal, measured; 3.66e-4 c before), so it is the first blade station of the
    C2 grid; reading the sampled arrays (the PCHIP of the C1 path,
    which never reaches past its vertices) calls it periodic. The only way the C2 blade run
    can differ from the sampled reading."""
    blade = c2_blade(C2)
    (m_tip, _, _), _ = blade.ends()
    a = cascade_of(C2).active
    sampled_tip = min(a.suction_surface[:, 0].min(), a.pressure_surface[:, 0].min())
    assert sampled_tip - m_tip == pytest.approx(2.56e-6, abs=0.1e-6)            # measured 2.564e-6
    target = 0.5 * (m_tip + sampled_tip)
    settings = stations_with_one_at(C2, target, replace(SETTINGS, n_eta=5))
    g, legacy = grid_of(C2, 45.0, settings), gr.make_grid(sampled_reading(C2), settings, math.radians(45.0))
    i = int(np.argmin(np.abs(g.m - target)))
    assert abs(g.m[i] - target) < 1e-13 and i == g.first_blade_i == legacy.first_blade_i - 1
    assert blade.metal(g.m[i]) == pytest.approx(4.23e-4, rel=0.01)              # measured 4.228e-4
    assert_c2_topology(C2, g)


@pytest.mark.parametrize("n_passage", C2_RESOLUTIONS)
def test_c2_walls_move_off_the_sampled_reading_by_its_pchip_error_only(n_passage):
    """Against the C2 grid read from the sampled arrays (the grid before the hand-off), the
    stations and the blade run are the same and the wall nodes move by the PCHIP's error
    against the curve it samples: at most 1.0e-5 c, RMS at most 2.1e-6 c (measured 9.8e-6
    and 2.0e-6 c here; up to 5.2e-5 c on a thin long-tailed blade over the sweep); min S
    moves by at most 1.8e-6 c (measured 1.7e-6 c), far below the passage's 0.878 c. The
    extension nodes are the same bit for bit."""
    settings = replace(SETTINGS, n_passage=n_passage, n_eta=31)
    g, legacy = c2_grid(C2, n_passage, 31), gr.make_grid(sampled_reading(C2), settings, math.radians(45.0))
    np.testing.assert_array_equal(g.m, legacy.m)
    assert (g.first_blade_i, g.last_blade_i) == (legacy.first_blade_i, legacy.last_blade_i)
    walls = slice(g.first_blade_i, g.last_blade_i + 1)
    move = g.nodes[walls][:, [0, -1], 1] - legacy.nodes[walls][:, [0, -1], 1]
    assert np.abs(move).max() <= 1.0e-5 and np.sqrt(np.mean(move ** 2, axis=0)).max() <= 2.1e-6
    width = lambda grid: (grid.nodes[walls, -1, 1] - grid.nodes[walls, 0, 1]).min()     # noqa: E731
    assert abs(width(g) - width(legacy)) <= 1.8e-6
    np.testing.assert_array_equal(g.nodes[:g.first_blade_i], legacy.nodes[:g.first_blade_i])
    np.testing.assert_array_equal(g.nodes[g.last_blade_i + 1:], legacy.nodes[g.last_blade_i + 1:])


@pytest.mark.parametrize("n_passage", (31, 101))
def test_c2_a_physical_chord_scales_the_walls_but_not_their_curve_parameters(n_passage):
    """The 0.05 m twin: the same stations in chords, blade run and named surfaces; every wall
    node scaled within 1e-12 (relative), the contours too, and u the same within 1e-14
    (measured 1.1e-15: the root of a station m that differs by round-off); the curve
    derivatives scale with the chord, within 1e-9 relative to their largest value (u moves
    by round-off where they change fastest, near the nose tip)."""
    unit, metric = c2_grid(C2, n_passage), c2_grid(C2_DIMENSIONAL, n_passage)
    scale = C2_DIMENSIONAL.physical_chord
    assert (metric.first_blade_i, metric.last_blade_i, len(metric.m)) == (unit.first_blade_i, unit.last_blade_i, len(unit.m))
    np.testing.assert_allclose(metric.nodes, scale * unit.nodes, rtol=1e-12, atol=1e-16 * scale)
    for name in ("lower", "upper"):
        a, b = getattr(metric.wall_identity, name), getattr(unit.wall_identity, name)
        np.testing.assert_array_equal(a.surface, b.surface)
        np.testing.assert_allclose(a.u, b.u, rtol=0.0, atol=1e-14)
        np.testing.assert_allclose(a.contour, scale * b.contour, rtol=1e-12, atol=1e-16 * scale)
    for k in (1, 2):
        for a, b in zip(metric.wall_curve_derivative(k), unit.wall_curve_derivative(k)):
            np.testing.assert_allclose(a, scale * b, rtol=0.0, atol=1e-9 * scale * np.abs(b).max())


def test_c2_n_eta_changes_nothing_but_the_interior_nodes():
    """n_eta 2 to 101 at 101 passage cells: the walls, every field of the wall identity (u
    included) and the curve derivatives the grid hands on are identical bit for bit."""
    reference = c2_grid(C2, 101, 2)
    for n_eta in (2, 11, 31, 101):
        g = c2_grid(C2, 101, n_eta)
        np.testing.assert_array_equal(g.nodes[:, 0], reference.nodes[:, 0])
        np.testing.assert_array_equal(g.nodes[:, -1], reference.nodes[:, -1])
        for name in ("lower", "upper"):
            for item in fields(gr.WallSide):
                np.testing.assert_array_equal(getattr(getattr(g.wall_identity, name), item.name),
                                              getattr(getattr(reference.wall_identity, name), item.name))
        for k in (1, 2):
            for a, b in zip(g.wall_curve_derivative(k), reference.wall_curve_derivative(k)):
                np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("spec", (C2, C2_OTHER, C2_SUCTION_TAIL, C2_STEEP))
@pytest.mark.parametrize("n_passage", (17, 62, 101))
def test_c2_grids_hold_the_hard_invariants_with_no_branch_jumps(spec, n_passage):
    """Positive and negative stagger, Kovach and sharp TEs, a steep blade whose stations
    fall on the exact LE and on m_te (round numbers), three resolutions: topology, metal-free
    eta lines, the passage-facing root at every wall node and walls that run forward along
    their contours without jumping a branch (assert_c2_topology), and any change of a wall's
    named surface only between stations on either side of the LE or TE point."""
    g = c2_grid(spec, n_passage, 11, 30.0)
    assert_c2_topology(spec, g)
    a = cascade_of(spec).active
    m = g.m[g.first_blade_i:g.last_blade_i + 1]
    for wall in (g.wall_identity.lower, g.wall_identity.upper):
        for j in np.flatnonzero(wall.surface[1:] != wall.surface[:-1]):
            assert any(m[j] < edge <= m[j + 1] for edge in (a.leading_edge_point[0], a.trailing_edge_point[0])), j


def test_c2_a_station_on_the_exact_le_point_counts_it_once():
    """At stagger 60 deg the stations fall on m = 0, the exact LE point, which both curves
    start from: the outline passes through it, so the line meets it once (on the pressure
    branch, half-open), and the station is a blade station with the LE point as its upper
    wall node, not a three-crossing cut."""
    g = c2_grid(C2_STEEP, 17, 5, 30.0)
    i = g.first_blade_i
    assert g.m[i] == 0.0
    upper = g.wall_identity.upper
    assert upper.surface[0] == "pressure" and upper.u[0] == pytest.approx(0.0, abs=1e-15)
    assert g.nodes[i, -1, 1] - g.pitch == pytest.approx(0.0, abs=1e-15)
    assert len(c2_blade(C2_STEEP).crossings(0.0)) == 2


def test_c2_rebuilding_the_exit_keeps_the_authoritative_wall_data():
    """Exit angles kappa_te + 2, + 5, - 8, + 25 deg and back: every rebuilt C2 grid equals a
    fresh build at its angle, the blade and inlet nodes, the wall identity (u included) and
    the curve derivatives never change, and the start comes back."""
    start = c2_grid(C2, 101, 11)
    kappa_te = cascade_of(C2).kappa_te
    kept = slice(0, start.last_blade_i + 1)
    state = start
    for turn in (2.0, 5.0, -8.0, 25.0, 0.0):
        state = gr.rebuild_exit(state, kappa_te + math.radians(turn))
        fresh = gr.make_grid(cascade_of(C2), replace(SETTINGS, n_passage=101, n_eta=11), math.radians(45.0),
                             kappa_te + math.radians(turn))
        np.testing.assert_array_equal(state.nodes, fresh.nodes)
        np.testing.assert_array_equal(state.nodes[kept], start.nodes[kept])
        for name in ("lower", "upper"):
            for item in fields(gr.WallSide):
                np.testing.assert_array_equal(getattr(getattr(state.wall_identity, name), item.name),
                                              getattr(getattr(start.wall_identity, name), item.name))
        for k in (0, 1, 2):
            for a, b in zip(state.wall_curve_derivative(k), start.wall_curve_derivative(k)):
                np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(state.nodes, start.nodes)


@pytest.mark.parametrize("shift, refused", ((1e-9, True), (2e-12, True), (1e-13, False)))
def test_c2_a_wall_node_off_its_curve_is_refused(shift, refused):
    """Moving one blade station's whole eta line (topology intact) by more than TOUCH chord
    puts its wall nodes off their curve points, and Grid refuses it whoever built it; a
    move within TOUCH (round-off scale) is accepted."""
    g = c2_grid(C2, 31, 5)
    nodes = np.array(g.nodes)
    nodes[g.first_blade_i + 10, :, 1] += shift
    if refused:
        with pytest.raises(gr.GridError, match="off its authoritative curve point"):
            replace(g, nodes=nodes)
    else:
        replace(g, nodes=nodes)


def test_c2_the_passage_check_reads_the_curves():
    """The passage check on the curves. At the cascade's own pitch every piece clears the
    bound at once, so its narrowest width is the smallest width at the piece ends (grid.py,
    _Row.narrowest), an upper estimate: against an independent search on C2Blade (bounded
    Brent on the curves' cut) it lies above by 4.10e-8 c (measured; its piece end 2.5e-4 c
    from the widest cut; 1e-9 with the legacy PCHIP thickness, whose widest cut fell next to
    a piece end), and it is the width of the cut at the m it reports (to 1e-12). With the
    pitch set to the curves' widest cut D plus 1e-9 c the grid builds while D - 1e-9 c is
    refused. Either
    part of the check may refuse it, and each names what it measured: the cut of the
    curves at a piece end at least the pitch (D to 1e-6), or the narrowest width (-1e-9 c
    to 0.1 %)."""
    blade = c2_blade(C2)
    c = cascade_of(C2)

    def span(m: float) -> float:
        ys = [y for y, _, _ in blade.crossings(m)]
        return ys[-1] - ys[0]

    lines = np.linspace(0.1, 0.5, 401)
    k = int(np.argmax([span(m) for m in lines]))
    best = minimize_scalar(lambda m: -span(m), bounds=(lines[k - 2], lines[k + 2]), method="bounded",
                           options={"xatol": 1e-12})
    widest = -float(best.fun)
    row = gr._CurveRow(c2_grid(C2, 31).boundaries)
    width, where = row.narrowest(c.active.pitch)
    assert 0.0 <= width - (c.active.pitch - widest) <= 5e-8                     # measured 4.10e-8
    assert width == pytest.approx(c.active.pitch - span(where), abs=1e-12)
    for excess, builds in ((1e-9, True), (-1e-9, False)):
        narrow = replace(c, active=replace(c.active, pitch=widest + excess,
                                           periodic_translation=np.array([0.0, widest + excess])))
        settings = replace(SETTINGS, n_passage=31, n_eta=5)
        if builds:
            gr.make_grid(narrow, settings, math.radians(45.0))
        else:
            with pytest.raises(gr.GridError, match="close the passage|the passage closes") as refused:
                gr.make_grid(narrow, settings, math.radians(45.0))
            message = str(refused.value)
            found = re.search(r"narrowest width over the blade is (\S+) chords", message)
            if found:
                assert float(found[1]) == pytest.approx(excess, rel=1e-3)
            else:
                span = re.search(r"the cut of the curves \(at a passage-check piece end\) of .* spans (\S+) chords", message)
                assert float(span[1]) == pytest.approx(widest, abs=1e-6)


# --- Admissible n_passage: the wall-station topology (grid.py section 8, 2026-11-12) ---------

TOPOLOGY_STAGGERS = (13.0, 23.0, 33.0, 43.0, 53.0)
TOPOLOGY = {stagger: replace(C2, stagger_deg=stagger) for stagger in TOPOLOGY_STAGGERS}
# measured by scripts/audit_uniform_grid_validity.py: n_passage 20 .. 800, inlet 45 deg, one-chord extensions
ADMISSIBLE = {13.0: (31, 60, 169), 23.0: (61, 92, 101), 33.0: (51, 61, 76), 43.0: (51, 81, 106), 53.0: (25, 31, 48)}
LE_ONLY_INADMISSIBLE = {13.0: (41, 81), 23.0: (151, 201), 33.0: (101, 201), 43.0: (41, 61), 53.0: (51, 101)}
CEILING = {13.0: 576, 23.0: 320, 33.0: 186, 43.0: 111, 53.0: 63}   # 575, 319 at 13, 23 deg with the legacy PCHIP
                                                                    # thickness (its nose window was wider)
FINEST = {13.0: 570, 23.0: 312, 33.0: 176, 43.0: 106, 53.0: 48}


def topology_grid(stagger: float, n_passage: int, require: bool = False, **extensions) -> gr.Grid:
    settings = replace(SETTINGS, n_passage=n_passage, n_eta=5, **extensions)
    return gr.make_grid(cascade_of(TOPOLOGY[stagger]), settings, math.radians(45.0), require_ss_ps=require)


def crossing_curves(stagger: float, m: float) -> list[str]:
    """The curves of the station line's crossings, read by C2Blade (not grid.py), sorted."""
    return sorted(name for _, name, _ in c2_blade(TOPOLOGY[stagger]).crossings(m))


@pytest.mark.parametrize("stagger,n_passage", [(s, n) for s in TOPOLOGY_STAGGERS for n in ADMISSIBLE[s]])
def test_an_admissible_grid_builds_under_the_check_unchanged(stagger, n_passage):
    """A measured admissible n_passage: the first and the last blade station each cross
    the suction and the pressure curve once (C2Blade), wall_station_topology agrees, every
    lower wall node is on the suction and every upper one on the pressure surface, and the
    grid built with require_ss_ps is the default grid, array for array."""
    plain = topology_grid(stagger, n_passage)
    strict = topology_grid(stagger, n_passage, require=True)
    assert np.array_equal(strict.m, plain.m) and np.array_equal(strict.nodes, plain.nodes)
    assert (strict.first_blade_i, strict.last_blade_i) == (plain.first_blade_i, plain.last_blade_i)
    for i in (strict.first_blade_i, strict.last_blade_i):
        assert crossing_curves(stagger, float(strict.m[i])) == ["pressure", "suction"]
    le, te = gr.wall_station_topology(strict)
    for t in (le, te):
        assert t.valid and (t.crossings, t.suction, t.pressure, t.on_branch_point) == (2, 1, 1, False)
    assert le.m > le.branch_point and te.m < te.branch_point
    wid = strict.wall_identity
    assert set(wid.lower.surface) == {"suction"} and set(wid.upper.surface) == {"pressure"}
    gr.validate_uniform_grid_topology(strict, n_passage=n_passage)


@pytest.mark.parametrize("stagger,n_passage", [(s, n) for s in TOPOLOGY_STAGGERS for n in LE_ONLY_INADMISSIBLE[s]])
def test_an_le_inadmissible_n_passage_is_refused_not_repaired(stagger, n_passage):
    """A measured LE-inadmissible n_passage: the first blade station lies between the nose
    tip and the LE point and crosses the suction curve twice (C2Blade); require_ss_ps raises
    GridTopologyError naming the LE only, with the station, the nose tip and the LE point;
    the default build still makes the same grid, and validating it raises the same."""
    plain = topology_grid(stagger, n_passage)
    blade = c2_blade(TOPOLOGY[stagger])
    (tip, _, _), _ = blade.ends()
    m_f = float(plain.m[plain.first_blade_i])
    assert tip < m_f < plain.m_le
    assert crossing_curves(stagger, m_f) == ["suction", "suction"]
    with pytest.raises(gr.GridTopologyError) as refused:
        topology_grid(stagger, n_passage, require=True)
    error = refused.value
    assert isinstance(error, gr.GridError)
    assert error.edge == "LE" and error.n_passage == n_passage
    message = str(error)
    assert f"n_passage = {n_passage} is not admissible" in message
    assert f"First blade station i = {plain.first_blade_i}: m = {m_f:.6g}" in message
    assert f"nose tip m = {tip:.6g}" in message and "nose-return interval" in message and "LE point" in message
    assert "Last blade station" not in message and "none is substituted" in message
    with pytest.raises(gr.GridTopologyError) as again:
        gr.validate_uniform_grid_topology(plain, n_passage=n_passage, stagger=cascade_of(TOPOLOGY[stagger]).stagger)
    assert str(again.value) == message
    le, te = gr.wall_station_topology(plain)
    assert not le.valid and te.valid and (le.suction, le.pressure) == (2, 0)


def test_a_te_inadmissible_n_passage_is_refused():
    """With the default one-chord extensions the stations are symmetric about the blade's
    middle, so the first station's offset past the LE point equals the last one's ahead of
    the TE point, and a TE failure comes with an LE failure (the TE window is the smaller).
    A 1.25-chord exit extension breaks the symmetry: at 13 deg, n_passage 102, the last
    blade station lies between the TE point and the tail end (on the suction surface at
    this stagger) and crosses the suction curve twice; only the TE is named."""
    plain = topology_grid(13.0, 102, downstream_extension_chords=1.25)
    blade = c2_blade(TOPOLOGY[13.0])
    _, (tail, tail_curve, _) = blade.ends()
    m_l = float(plain.m[plain.last_blade_i])
    assert tail_curve == "suction" and plain.m_te < m_l < tail
    assert crossing_curves(13.0, m_l) == ["suction", "suction"]
    assert crossing_curves(13.0, float(plain.m[plain.first_blade_i])) == ["pressure", "suction"]
    with pytest.raises(gr.GridTopologyError) as refused:
        topology_grid(13.0, 102, require=True, downstream_extension_chords=1.25)
    error = refused.value
    assert error.edge == "TE"
    message = str(error)
    assert f"Last blade station i = {plain.last_blade_i}: m = {m_l:.6g}" in message
    assert f"tail end m = {tail:.6g}" in message and "tail-return interval" in message and "TE point" in message
    assert "First blade station" not in message


def test_both_edges_are_named_when_both_fail():
    """43 deg, n_passage 101: the first station crosses the suction curve twice, the last
    the pressure curve twice (the tail end is on the pressure surface at this stagger)."""
    plain = topology_grid(43.0, 101)
    assert crossing_curves(43.0, float(plain.m[plain.first_blade_i])) == ["suction", "suction"]
    assert crossing_curves(43.0, float(plain.m[plain.last_blade_i])) == ["pressure", "pressure"]
    with pytest.raises(gr.GridTopologyError) as refused:
        topology_grid(43.0, 101, require=True)
    assert refused.value.edge == "LE and TE"
    message = str(refused.value)
    assert "First blade station" in message and "Last blade station" in message


def test_a_station_through_the_le_point_itself_is_not_admissible():
    """At stagger 60 deg (C2_STEEP, n_passage 17) the first blade station is m = 0, the LE
    point, where the suction and the pressure curve meet: its upper wall node is the branch
    point itself, so the station is not one suction and one pressure wall node."""
    g = c2_grid(C2_STEEP, 17, 5, 30.0)
    le, _ = gr.wall_station_topology(g)
    assert g.m[g.first_blade_i] == 0.0
    assert le.on_branch_point and (le.crossings, le.suction, le.pressure) == (2, 1, 1) and not le.valid
    with pytest.raises(gr.GridTopologyError, match="passes through the LE point itself"):
        gr.validate_uniform_grid_topology(g, n_passage=17)


@pytest.mark.parametrize("stagger,n_passage", ((13.0, 81), (33.0, 101), (33.0, 201), (43.0, 61), (53.0, 51),
                                               (53.0, 101)))
def test_the_suggestions_are_admissible_and_nearest(stagger, n_passage):
    """The error's nearest lower and higher admissible n_passage are the neighbours of the
    requested value in find_valid_n_passage, each builds under the check, finest is the
    largest admissible value (as measured) and ceiling the measured ceiling."""
    with pytest.raises(gr.GridTopologyError) as refused:
        topology_grid(stagger, n_passage, require=True)
    error = refused.value
    settings = replace(SETTINGS, n_passage=n_passage, n_eta=5)
    valid = gr.find_valid_n_passage(cascade_of(TOPOLOGY[stagger]), settings, math.radians(45.0),
                                    range(1, CEILING[stagger] + 40))
    below, above = [v for v in valid if v < n_passage], [v for v in valid if v > n_passage]
    assert error.nearest_lower == (below[-1] if below else None)
    assert error.nearest_higher == (above[0] if above else None)
    assert error.finest == valid[-1] == FINEST[stagger] and error.ceiling == CEILING[stagger]
    for v in (error.nearest_lower, error.nearest_higher):
        if v is not None:
            topology_grid(stagger, v, require=True)


@pytest.mark.parametrize("stagger", TOPOLOGY_STAGGERS)
def test_the_refinement_ceiling(stagger):
    """n_passage_ceiling is the smallest n_passage whose station spacing, computed here from
    the lattice formula (one constant dm, exact m_in and m_out), is at most the LE window
    m_le - nose tip (C2Blade's ends; the TE window is smaller at every stagger here); from it
    on no n_passage is admissible, and between FINEST and it none either."""
    cascade = cascade_of(TOPOLOGY[stagger])
    settings = replace(SETTINGS, n_passage=101, n_eta=5)
    blade = c2_blade(TOPOLOGY[stagger])
    (tip, _, _), (tail, _, _) = blade.ends()
    m_le, m_te = cascade.active.leading_edge_point[0], cascade.active.trailing_edge_point[0]
    width = (m_te - m_le) + 2.0 * cascade.active.chord

    def spacing(n: int) -> float:
        return width / max(1, round(width / ((m_te - m_le) / n)))

    ceiling = gr.n_passage_ceiling(cascade, settings, math.radians(45.0))
    assert ceiling == CEILING[stagger]
    assert tail - m_te < m_le - tip
    assert spacing(ceiling) <= m_le - tip < spacing(ceiling - 1)
    assert gr.find_valid_n_passage(cascade, settings, math.radians(45.0), range(FINEST[stagger] + 1, ceiling + 60)) == ()
    assert gr.find_valid_n_passage(cascade, settings, math.radians(45.0), [FINEST[stagger]]) == (FINEST[stagger],)


@pytest.mark.parametrize("stagger,stop", ((33.0, 200), (53.0, 70)))
def test_find_valid_n_passage_matches_an_independent_sweep(stagger, stop):
    """Every n_passage from 20: the default grid is built and its first and last blade
    station are classified by C2Blade's crossings; the admissible set is
    find_valid_n_passage's, and it is aliased (not one interval)."""
    cascade = cascade_of(TOPOLOGY[stagger])
    expected = []
    for n in range(20, stop + 1):
        g = topology_grid(stagger, n)
        if all(crossing_curves(stagger, float(g.m[i])) == ["pressure", "suction"]
               for i in (g.first_blade_i, g.last_blade_i)):
            expected.append(n)
    found = gr.find_valid_n_passage(cascade, replace(SETTINGS, n_eta=5), math.radians(45.0), range(20, stop + 1))
    assert found == tuple(expected)
    assert len(found) < found[-1] - found[0] + 1


@pytest.mark.parametrize("spec,n_passage", ((SPEC, 101), (SPEC, 61), (C2, 101), (C2, 61), (C2_OTHER, 51),
                                            (C2_SUCTION_TAIL, 41)))
def test_wall_station_topology_reads_the_wall_identity(spec, n_passage):
    """C1 and C2, positive and negative stagger: the surfaces wall_station_topology counts at
    the first and last blade station are those Grid.wall_identity gives their two wall
    nodes (with no hidden crossings)."""
    g = grid_of(spec, 45.0, replace(SETTINGS, n_passage=n_passage, n_eta=5))
    le, te = gr.wall_station_topology(g)
    wid = g.wall_identity
    for t, k in ((le, 0), (te, -1)):
        names = [str(wid.lower.surface[k]), str(wid.upper.surface[k])]
        assert wid.lower.hidden[k] == wid.upper.hidden[k] == 0 and t.crossings == 2
        assert (t.suction, t.pressure) == (names.count("suction"), names.count("pressure"))
        assert t.station == (g.first_blade_i if k == 0 else g.last_blade_i)


def test_c1_surface_of_every_outline_segment():
    """C1: the surface grid.py gives each segment of the closed outline is the named
    polyline both of its end points are vertices of (matched to the Cascade's own
    suction_surface and pressure_surface arrays within 1e-12 chord; the LE and TE points are
    vertices of both), including the first pressure segment out of the LE point."""
    g = grid_of(SPEC, 45.0, replace(SETTINGS, n_passage=61, n_eta=5))
    row = gr._row_of(g.boundaries)
    a = cascade_of(SPEC).active

    def on(points: np.ndarray, vertex: np.ndarray) -> bool:
        return bool(np.min(np.hypot(*(points - vertex).T)) <= 1e-12)

    expected = []
    for k in range(len(row.outline) - 1):
        ends = (row.outline[k], row.outline[k + 1])
        suction = all(on(a.suction_surface, p) for p in ends)
        pressure = all(on(a.pressure_surface, p) for p in ends)
        assert suction != pressure
        expected.append("suction" if suction else "pressure")
    assert list(row.surface_of(np.arange(len(row.outline) - 1))) == expected
    assert expected[row.n_suction - 2:row.n_suction] == ["suction", "pressure"]


def test_topology_helpers_check_their_inputs():
    """wall_station_topology takes a Grid; n_passage values are whole numbers >= 1; without
    n_passage the error gives the finest admissible value and the ceiling but no neighbours."""
    with pytest.raises(TypeError):
        gr.wall_station_topology(object())
    settings = replace(SETTINGS, n_passage=101, n_eta=5)
    for bad in (0, -3, 2.5, True):
        with pytest.raises(ValueError, match="n_passage"):
            gr.find_valid_n_passage(cascade_of(TOPOLOGY[33.0]), settings, math.radians(45.0), [bad])
    with pytest.raises(gr.GridTopologyError) as refused:
        gr.validate_uniform_grid_topology(topology_grid(33.0, 101))
    error = refused.value
    assert str(error).startswith("this uniform grid is not admissible")
    assert error.n_passage is None and error.nearest_lower is None and error.nearest_higher is None
    assert (error.finest, error.ceiling) == (FINEST[33.0], CEILING[33.0])
