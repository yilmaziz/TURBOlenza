"""
Tests for turbolenza.b2b.metrics: build_metrics and GridMetrics, the geometric fields
of Aungier's transformed equations on a frozen grid (Eqs. 5-1 to 5-3, 5-9).

The references are independent of metrics.py: the grid's own arrays; a region-aware
second-order m derivative written here station by station (region_derivative); the exact
side-line law of grid.py's extensions, dy/dm = tan(beta(m)) with beta linear in m; and the
exact slope of the smooth passage-facing walls, found by brentq on a parametric PCHIP of the
same blade (or of the blade at 1601 camber stations); and manufactured functions (a
quadratic, m^3, sin m, and a circle with a vertical tangent) for the three difference
stencils of _d_dm_run themselves. Failure cases that no valid Grid can hold are built as
raw arrays and handed to the lowest-level helper, _metric_fields.

C2 grids (exact wall slopes from the curves) are checked against geometry.py's own
wall_slopes on the curve and u each wall node records, against Richardson differences of
y(m) taken on the curve itself (brentq roots, no derivative formula), against the width
S(m) of the curves differenced the same way, and against the same cascade read from its
sampled arrays (whose extension metrics must be the same bit for bit). C1 is locked by
fingerprints taken before the hand-off.
"""

from __future__ import annotations

import ast
import math
import sys
from dataclasses import fields, replace
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq

from turbolenza.b2b import geometry
from turbolenza.b2b import grid as gr
from turbolenza.b2b import metrics as mt
from turbolenza.b2b.config import GridSettings
from turbolenza.b2b.geometry import CascadeSpec, make_cascade

# The EXAMPLE (65-(12A10)10 at 33 deg), a negative-stagger sharp-TE blade, and edge cases of
# grid.py: a thin nose that folds, a steep nose, a long Kovach tail, a suction-side tail.
SPEC = CascadeSpec(blade_type="NACA65", solidity=1.0, stagger_deg=33.0, cl0=1.2, t_max_over_c=0.10)
OTHER = CascadeSpec(blade_type="NACA65", solidity=1.5, stagger_deg=-20.0, cl0=0.4, t_max_over_c=0.06,
                    profile_variant="emery_base")
THIN = CascadeSpec(blade_type="NACA65", solidity=1.0, stagger_deg=50.0, cl0=1.6, t_max_over_c=0.04)
STEEP = CascadeSpec(blade_type="NACA65", solidity=1.0, stagger_deg=60.0, cl0=1.6, t_max_over_c=0.10)
LONG_TAIL = CascadeSpec(blade_type="NACA65", solidity=1.0, stagger_deg=60.0, cl0=0.0, t_max_over_c=0.04)
SUCTION_TAIL = CascadeSpec(blade_type="NACA65", solidity=0.8, stagger_deg=-30.0, cl0=1.2, t_max_over_c=0.10)
EDGE_CASES = (SPEC, OTHER, THIN, STEEP, LONG_TAIL, SUCTION_TAIL)
FINE = replace(SPEC, n_camber_stations=1601)
SETTINGS = GridSettings(n_passage=31, n_eta=21)
EPS = np.finfo(float).eps


@lru_cache(maxsize=None)
def cascade_of(spec: CascadeSpec):
    return make_cascade(spec)


@lru_cache(maxsize=None)
def grid_of(spec: CascadeSpec, n_passage: int = 31, n_eta: int = 21, inlet_deg: float = 45.0,
            exit_turn_deg: float | None = None, upstream: float = 1.0, downstream: float = 1.0) -> gr.Grid:
    c = cascade_of(spec)
    exit_angle = None if exit_turn_deg is None else c.kappa_te + math.radians(exit_turn_deg)
    settings = GridSettings(n_passage=n_passage, n_eta=n_eta, upstream_extension_chords=upstream,
                            downstream_extension_chords=downstream)
    return gr.make_grid(c, settings, math.radians(inlet_deg), exit_angle)


@lru_cache(maxsize=None)
def metrics_of(spec: CascadeSpec, *args) -> mt.GridMetrics:
    return mt.build_metrics(grid_of(spec, *args))


def runs_of(g: gr.Grid) -> list[tuple[int, int]]:
    """The three m regions as [start, stop) station ranges, from the grid's blade indices."""
    return [(0, g.first_blade_i), (g.first_blade_i, g.last_blade_i + 1), (g.last_blade_i + 1, len(g.m))]


def region_derivative(values: np.ndarray, dm: float, runs: list[tuple[int, int]]) -> np.ndarray:
    """d/dm written out station by station: central inside a run, the three-point one-sided
    forms at its two ends, and nothing taken from another run."""
    values = np.asarray(values, dtype=float)
    out = np.full(values.shape, np.nan)
    for start, stop in runs:
        for i in range(start, stop):
            if i == start:
                out[i] = (-3.0 * values[i] + 4.0 * values[i + 1] - values[i + 2]) / (2.0 * dm)
            elif i == stop - 1:
                out[i] = (3.0 * values[i] - 4.0 * values[i - 1] + values[i - 2]) / (2.0 * dm)
            else:
                out[i] = (values[i + 1] - values[i - 1]) / (2.0 * dm)
    return out


def raw(g: gr.Grid) -> dict:
    """The arguments of _metric_fields from a grid, as writable copies."""
    return dict(m=np.array(g.m), eta=np.array(g.eta), nodes=np.array(g.nodes), first_blade_i=g.first_blade_i,
                last_blade_i=g.last_blade_i, pitch=g.pitch)


def side_line_slope(g: gr.Grid, m: np.ndarray, side: str) -> np.ndarray:
    """tan(beta(m)) of grid.py's side lines, beta linear in m from the blade angle at the
    junction to the flow angle at the far station (the exact derivative of the side line)."""
    b = g.boundaries
    if side == "inlet":
        junction_m, junction_angle, far_m, far_angle = b.m_le, b.upstream_beta[-1], b.m_in, b.inlet_angle
    else:
        junction_m, junction_angle, far_m, far_angle = b.m_te, b.downstream_beta[0], b.m_out, b.exit_angle
    return np.tan(junction_angle + (far_angle - junction_angle) * (m - junction_m) / (far_m - junction_m))


@lru_cache(maxsize=None)
def smooth_walls(spec: CascadeSpec):
    """The blade's closed outline and a parametric PCHIP through it by chord length."""
    a = cascade_of(spec).active
    loop = np.vstack((a.suction_surface[::-1], a.pressure_surface[1:]))
    t = np.concatenate(([0.0], np.cumsum(np.hypot(*np.diff(loop, axis=0).T))))
    return loop, t, PchipInterpolator(t, loop[:, 0]), PchipInterpolator(t, loop[:, 1])


def wall_slopes(spec: CascadeSpec, m: float) -> tuple[float, float]:
    """Exact slopes dy/dm of the smooth passage-facing walls at x = m: the top of the cut (the
    lower wall) and its bottom (the upper wall, one pitch up), brentq on the PCHIP."""
    loop, t, x_of_t, y_of_t = smooth_walls(spec)
    x0, x1 = loop[:-1, 0], loop[1:, 0]
    k = np.flatnonzero((np.minimum(x0, x1) <= m) & (m < np.maximum(x0, x1)))
    y = loop[k, 1] + (m - x0[k]) / (x1[k] - x0[k]) * (loop[k + 1, 1] - loop[k, 1])
    k = k[np.argsort(y)]
    out = []
    for segment in (k[-1], k[0]):
        def offset(u: float) -> float:
            return float(x_of_t(u)) - m
        lo, hi = t[segment], t[segment + 1]
        u = lo if offset(lo) == 0.0 else hi if offset(hi) == 0.0 else brentq(offset, lo, hi, xtol=1e-15, rtol=1e-15)
        out.append(float(y_of_t(u, 1)) / float(x_of_t(u, 1)))
    return out[0], out[1]


def blade_slope_errors(spec_reference: CascadeSpec, g: gr.Grid, metrics: mt.GridMetrics) -> tuple[float, float]:
    """Largest |tan_beta - exact wall slope| over the blade stations at least 0.05 c from
    both ends of the blade, lower and upper wall."""
    f, l = g.first_blade_i, g.last_blade_i
    m = g.m[f:l + 1]
    inner = np.flatnonzero((m - g.m_le > 0.05 * g.chord) & (g.m_te - m > 0.05 * g.chord))
    exact = np.array([wall_slopes(spec_reference, m[k]) for k in inner])
    return (float(np.abs(metrics.tan_beta0[f + inner] - exact[:, 0]).max()),
            float(np.abs(metrics.tan_beta1[f + inner] - exact[:, 1]).max()))


# --- Contract ---------------------------------------------------------------------------

def test_the_metrics_expose_exactly_these_fields():
    """The Grid by reference and the eight metric arrays, then (2026-11-11) the three
    d tan(beta)/dm arrays; nothing the Grid already owns (m, eta, pitch, blade indices) is
    copied, and no xi, r, b, density, coefficient or d2S/dm2. d tan(beta)/d eta is the
    property dtan_beta_deta, the array dS_dm itself."""
    assert [item.name for item in fields(mt.GridMetrics)] == ["grid", "S", "dS_dm", "tan_beta0", "tan_beta1",
                                                             "tan_beta", "beta", "sin_beta", "cos_beta",
                                                             "dtan_beta0_dm", "dtan_beta1_dm", "dtan_beta_dm"]
    metrics = metrics_of(SPEC)
    assert metrics.dtan_beta_deta is metrics.dS_dm and metrics.exact_wall_slopes is False


@pytest.mark.parametrize("spec", EDGE_CASES)
def test_shapes_and_finite_values(spec):
    g = grid_of(spec)
    metrics = metrics_of(spec)
    n_m, n_eta = len(g.m), len(g.eta)
    for name in ("S", "dS_dm", "tan_beta0", "tan_beta1"):
        assert getattr(metrics, name).shape == (n_m,), name
    for name in ("tan_beta", "beta", "sin_beta", "cos_beta"):
        assert getattr(metrics, name).shape == (n_m, n_eta), name
    for item in fields(metrics):
        if item.name != "grid":
            assert np.all(np.isfinite(getattr(metrics, item.name))), item.name


def test_the_metrics_are_frozen_and_read_only_and_keep_their_grid():
    g = grid_of(SPEC)
    metrics = mt.build_metrics(g)
    assert metrics.grid is g
    for item in fields(metrics):
        value = getattr(metrics, item.name)
        if isinstance(value, np.ndarray):
            assert not value.flags.writeable, item.name
            with pytest.raises(ValueError):
                value[0] = 0.0
    with pytest.raises(AttributeError):
        metrics.S = None
    assert not metrics.jacobian_det.flags.writeable


def test_every_field_states_its_unit():
    """The module docstring gives each field its unit: S a length (chords or metres), beta in
    radians, the slopes, sines and cosines dimensionless, the m derivatives of tan(beta) one
    over a length, its eta derivative dimensionless."""
    doc = mt.__doc__
    table = doc[doc.index("Fields (GridMetrics) and units"):doc.index("Singular grid angle")]
    for name, unit in (("S, jacobian_det", "length"), ("dS_dm", "dimensionless"), ("tan_beta0/1", "dimensionless"),
                       ("tan_beta ", "dimensionless"), ("beta ", "radians"), ("sin_beta", "dimensionless"),
                       ("cos_beta", "dimensionless"), ("dtan_beta0/1_dm", "1/length"), ("dtan_beta_dm", "1/length"),
                       ("dtan_beta_deta", "dimensionless")):
        line = next(row for row in table.splitlines() if row.strip().startswith(name.strip()))
        assert unit in line, name
    assert "chords for a chord-1 cascade, metres when dimensional" in table


def test_only_a_grid_is_accepted():
    for item in (cascade_of(SPEC), grid_of(SPEC).boundaries, None):
        with pytest.raises(TypeError):
            mt.build_metrics(item)


def test_metrics_reads_the_grid_and_nothing_else():
    """metrics.py imports numpy and Grid from grid.py only: no geometry, config, coefficient,
    flow, gas or solver module, no Cascade, no rebuilt geometry, and no suction/pressure
    names of the walls."""
    tree = ast.parse(Path(mt.__file__).read_text(encoding="utf-8"))
    imported = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    imported |= {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
    outside = {name for name in imported if name.split(".")[0] not in sys.stdlib_module_names}
    assert outside == {"numpy", "numpy.typing", "turbolenza.b2b.grid"}
    from_grid = {alias.name for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
                 and node.module == "turbolenza.b2b.grid" for alias in node.names}
    assert from_grid == {"Grid"}
    attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert not {"wall_identity", "boundaries", "active", "spec", "throat_opening"} & attributes
    assert not {"make_cascade", "make_grid", "Cascade", "StreamSheet"} & names


# --- S and the mapping ------------------------------------------------------------------

@pytest.mark.parametrize("spec", EDGE_CASES)
def test_S_is_the_upper_less_the_lower_boundary_and_positive(spec):
    g, metrics = grid_of(spec), metrics_of(spec)
    np.testing.assert_array_equal(metrics.S, g.upper_boundary[:, 1] - g.lower_boundary[:, 1])
    assert np.all(metrics.S > 0.0)
    walls = slice(g.first_blade_i, g.last_blade_i + 1)
    assert np.all(metrics.S[walls] < g.pitch)            # a blade's cut takes part of the pitch


@pytest.mark.parametrize("spec", EDGE_CASES)
def test_S_is_the_pitch_on_both_extensions(spec):
    g, metrics = grid_of(spec), metrics_of(spec)
    tolerance = 4.0 * np.spacing(max(np.abs(g.nodes[..., 1]).max(), g.pitch))
    for start, stop in (runs_of(g)[0], runs_of(g)[2]):
        assert np.abs(metrics.S[start:stop] - g.pitch).max() <= tolerance


def test_the_jacobian_determinant_is_S():
    """det d(m, y)/d(m, eta) = S for y = y0 + eta S: S itself, checked against a central
    difference of the node positions in eta (exact for a map linear in eta)."""
    g, metrics = grid_of(SPEC), metrics_of(SPEC)
    assert metrics.jacobian_det is metrics.S
    dy_deta = (g.nodes[:, 2:, 1] - g.nodes[:, :-2, 1]) / (2.0 * g.deta)
    np.testing.assert_allclose(dy_deta, np.broadcast_to(metrics.S[:, None], dy_deta.shape), rtol=0.0,
                               atol=64 * EPS * np.abs(g.nodes[..., 1]).max() / g.deta)


@pytest.mark.parametrize("spec", EDGE_CASES)
def test_the_nodes_are_the_lower_boundary_plus_eta_S(spec):
    """grid.py's nodes are y0 + eta S to round-off: the largest difference is one ulp
    (2.2e-16 for the EXAMPLE), the bound here a few."""
    g, metrics = grid_of(spec), metrics_of(spec)
    mapped = g.lower_boundary[:, 1][:, None] + g.eta[None, :] * metrics.S[:, None]
    assert np.abs(mapped - g.nodes[..., 1]).max() <= 8.0 * np.spacing(np.abs(g.nodes[..., 1]).max())


def test_min_S_is_the_passage_spacing_not_the_throat():
    """S runs along y at constant m, the throat across the passage: for the EXAMPLE min S is
    0.8779 c and the geometric throat 0.7215 c, quantities of different meaning."""
    metrics = metrics_of(SPEC, 201)
    throat = cascade_of(SPEC).active.throat_opening
    assert metrics.S.min() == pytest.approx(0.87794, abs=1e-4)
    assert throat == pytest.approx(0.72152, abs=1e-4)
    assert metrics.S.min() - throat > 0.15


# --- Grid-line angles ---------------------------------------------------------------------

@pytest.mark.parametrize("spec", EDGE_CASES)
def test_the_boundary_slopes_are_the_region_derivatives_of_the_boundaries(spec):
    g, metrics = grid_of(spec), metrics_of(spec)
    tolerance = 16.0 * EPS * np.abs(g.nodes[..., 1]).max() / g.dm
    for slope, boundary in ((metrics.tan_beta0, g.lower_boundary), (metrics.tan_beta1, g.upper_boundary)):
        np.testing.assert_allclose(slope, region_derivative(boundary[:, 1], g.dm, runs_of(g)), rtol=0.0,
                                   atol=tolerance)


@pytest.mark.parametrize("spec", (SPEC, STEEP))
def test_tan_beta_is_affine_in_eta_with_the_boundary_slopes_at_its_ends(spec):
    """Eq. 5-3: exactly tan_beta0 at eta = 0 and tan_beta1 at eta = 1, and linear between."""
    g, metrics = grid_of(spec), metrics_of(spec)
    np.testing.assert_array_equal(metrics.tan_beta[:, 0], metrics.tan_beta0)
    np.testing.assert_array_equal(metrics.tan_beta[:, -1], metrics.tan_beta1)
    expected = metrics.tan_beta0[:, None] + g.eta[None, :] * (metrics.tan_beta1 - metrics.tan_beta0)[:, None]
    scale = np.abs(metrics.tan_beta).max()
    assert np.abs(metrics.tan_beta - expected).max() <= 8.0 * EPS * scale
    second = metrics.tan_beta[:, 2:] - 2.0 * metrics.tan_beta[:, 1:-1] + metrics.tan_beta[:, :-2]
    assert np.abs(second).max() <= 16.0 * EPS * scale


@pytest.mark.parametrize("spec", (SPEC, STEEP, SUCTION_TAIL))
def test_beta_and_its_sine_and_cosine(spec):
    """beta = arctan(tan beta) in (-pi/2, pi/2), sin = tan cos, sin^2 + cos^2 = 1, cos > 0."""
    metrics = metrics_of(spec)
    np.testing.assert_allclose(metrics.beta, np.arctan(metrics.tan_beta), rtol=0.0, atol=4 * EPS)
    assert np.all(np.abs(metrics.beta) < math.pi / 2)
    np.testing.assert_allclose(metrics.sin_beta, np.sin(metrics.beta), rtol=0.0, atol=8 * EPS)
    np.testing.assert_allclose(metrics.cos_beta, np.cos(metrics.beta), rtol=0.0, atol=8 * EPS)
    np.testing.assert_allclose(metrics.sin_beta ** 2 + metrics.cos_beta ** 2, 1.0, rtol=0.0, atol=8 * EPS)
    assert np.all(metrics.cos_beta > mt.SINGULAR_COS)


@pytest.mark.parametrize("spec", EDGE_CASES)
def test_every_constant_eta_line_has_the_slope_of_eq_5_3(spec):
    """The same region-aware derivative applied to every constant-eta line of the nodes
    gives tan_beta at every node: the 2-D field agrees with the boundary-built Eq. 5-3."""
    g, metrics = grid_of(spec), metrics_of(spec)
    direct = region_derivative(g.nodes[:, :, 1], g.dm, runs_of(g))
    assert np.abs(direct - metrics.tan_beta).max() <= 32.0 * EPS * np.abs(g.nodes[..., 1]).max() / g.dm


@pytest.mark.parametrize("spec", EDGE_CASES)
def test_dS_dm_is_the_derivative_of_S(spec):
    """dS/dm = tan_beta1 - tan_beta0, and the region derivative of S itself agrees."""
    g, metrics = grid_of(spec), metrics_of(spec)
    np.testing.assert_array_equal(metrics.dS_dm, metrics.tan_beta1 - metrics.tan_beta0)
    direct = region_derivative(metrics.S, g.dm, runs_of(g))
    assert np.abs(direct - metrics.dS_dm).max() <= 32.0 * EPS * np.abs(g.nodes[..., 1]).max() / g.dm


@pytest.mark.parametrize("spec", EDGE_CASES)
def test_the_extensions_have_equal_boundary_slopes_and_no_change_of_S(spec):
    g, metrics = grid_of(spec), metrics_of(spec)
    tolerance = 16.0 * EPS * np.abs(g.nodes[..., 1]).max() / g.dm
    for start, stop in (runs_of(g)[0], runs_of(g)[2]):
        assert np.abs(metrics.tan_beta1[start:stop] - metrics.tan_beta0[start:stop]).max() <= tolerance
        assert np.abs(metrics.dS_dm[start:stop]).max() <= tolerance


# --- No stencil across a transition -------------------------------------------------------

@pytest.mark.parametrize("spec", (SPEC, STEEP, LONG_TAIL))
@pytest.mark.parametrize("station", ("last_inlet_i", "first_blade_i", "last_blade_i", "first_exit_i"))
def test_a_station_moved_beyond_a_transition_leaves_the_other_side_alone(spec, station):
    """Moving every node of one station next to a transition (both boundaries and the eta
    line between them, consistently) changes the metrics of its own region only: the slopes
    on the other side of the LE or TE transition stay bit for bit."""
    g = grid_of(spec, 61)
    before = mt._metric_fields(**raw(g))
    i = getattr(g, station)
    arrays = raw(g)
    arrays["nodes"][i, :, 1] += 0.01 * g.chord          # the whole station: S, and S = pitch on a side line, kept
    after = mt._metric_fields(**arrays)
    runs = runs_of(g)
    own = next(k for k, (start, stop) in enumerate(runs) if start <= i < stop)
    for k, (start, stop) in enumerate(runs):
        for name in ("tan_beta0", "tan_beta1", "dS_dm", "S"):
            if k != own:
                np.testing.assert_array_equal(after[name][start:stop], before[name][start:stop], err_msg=(name, k))
    start, stop = runs[own]
    assert np.any(after["tan_beta0"][start:stop] != before["tan_beta0"][start:stop])


def test_the_edge_stations_use_their_own_region_only():
    """The four slopes beside the transitions are the one-sided three-point differences of
    their own region: the first blade slope from stations f, f+1, f+2, the last from l, l-1,
    l-2, the last inlet slope from f-1, f-2, f-3 and the first exit slope from l+1, l+2, l+3."""
    g, metrics = grid_of(SPEC, 101), metrics_of(SPEC, 101)
    f, l, dm = g.first_blade_i, g.last_blade_i, g.dm
    for slope, boundary in ((metrics.tan_beta0, g.lower_boundary[:, 1]), (metrics.tan_beta1, g.upper_boundary[:, 1])):
        y = boundary
        for i, expected in ((f, (-3 * y[f] + 4 * y[f + 1] - y[f + 2]) / (2 * dm)),
                            (l, (3 * y[l] - 4 * y[l - 1] + y[l - 2]) / (2 * dm)),
                            (f - 1, (3 * y[f - 1] - 4 * y[f - 2] + y[f - 3]) / (2 * dm)),
                            (l + 1, (-3 * y[l + 1] + 4 * y[l + 2] - y[l + 3]) / (2 * dm))):
            assert slope[i] == pytest.approx(expected, rel=0.0, abs=16 * EPS * np.abs(y).max() / dm), i


def test_a_wall_name_change_is_no_discontinuity_of_the_metrics():
    """At 101 passage cells the EXAMPLE's upper wall runs on the neighbor's nose underside
    (suction surface) at station 120 and on its pressure surface from 121 (grid.py's wall
    identity). The metrics follow the passage-facing contour as grid.py chose it: the blade
    slopes there are the ordinary one-sided and central differences of the wall nodes."""
    g, metrics = grid_of(SPEC, 101), metrics_of(SPEC, 101)
    assert list(g.wall_identity.upper.surface[:2]) == ["suction", "pressure"]
    y = g.upper_boundary[:, 1]
    f, dm = g.first_blade_i, g.dm
    assert metrics.tan_beta1[f] == pytest.approx((-3 * y[f] + 4 * y[f + 1] - y[f + 2]) / (2 * dm), abs=1e-12)
    assert metrics.tan_beta1[f + 1] == pytest.approx((y[f + 2] - y[f]) / (2 * dm), abs=1e-12)


# --- Convergence ----------------------------------------------------------------------------

def observed_orders(errors: list[float], spacings: list[float]) -> list[float]:
    return [math.log(e0 / e1) / math.log(h0 / h1) for e0, e1, h0, h1 in
            zip(errors, errors[1:], spacings, spacings[1:])]


@pytest.mark.parametrize("inlet_deg", (30.0, 60.0))
def test_inlet_slopes_converge_at_second_order_to_the_side_line_law(inlet_deg):
    """On the inlet side lines dy/dm = tan(beta(m)), beta linear in m from the inlet angle to
    kappa_le: the boundary slopes converge to it at second order (orders 1.93 to 2.00 over
    three halvings of dm, four grids), far above round-off."""
    errors, spacings = [], []
    for n_passage in (21, 41, 81, 161):
        g, metrics = grid_of(SPEC, n_passage, 5, inlet_deg), metrics_of(SPEC, n_passage, 5, inlet_deg)
        inlet = slice(0, g.first_blade_i)
        exact = side_line_slope(g, g.m[inlet], "inlet")
        errors.append(max(np.abs(metrics.tan_beta0[inlet] - exact).max(), np.abs(metrics.tan_beta1[inlet] - exact).max()))
        spacings.append(g.dm)
    orders = observed_orders(errors, spacings)
    assert all(1.9 < order < 2.1 for order in orders), orders
    assert errors[-1] < 1e-5


@pytest.mark.parametrize("turn_deg", (5.0, -8.0))
def test_exit_slopes_converge_at_second_order_on_a_curved_exit_line(turn_deg):
    """With an explicit exit angle kappa_te + turn the exit side line is curved, beta linear
    in m from kappa_te to the exit angle: second-order convergence as on the inlet."""
    errors, spacings = [], []
    for n_passage in (21, 41, 81, 161):
        g = grid_of(SPEC, n_passage, 5, 45.0, turn_deg)
        metrics = metrics_of(SPEC, n_passage, 5, 45.0, turn_deg)
        exit_ = slice(g.last_blade_i + 1, None)
        exact = side_line_slope(g, g.m[exit_], "exit")
        errors.append(max(np.abs(metrics.tan_beta0[exit_] - exact).max(), np.abs(metrics.tan_beta1[exit_] - exact).max()))
        spacings.append(g.dm)
    orders = observed_orders(errors, spacings)
    assert all(1.9 < order < 2.1 for order in orders), orders


def test_a_straight_exit_line_has_exact_slopes():
    """exit_angle = kappa_te makes the exit line the straight kappa_te tangent: every
    difference is exact, to round-off."""
    g, metrics = grid_of(SPEC, 101), metrics_of(SPEC, 101)
    exit_ = slice(g.last_blade_i + 1, None)
    assert np.abs(metrics.tan_beta0[exit_] - math.tan(cascade_of(SPEC).kappa_te)).max() <= 1e-12


def test_blade_wall_slopes_converge_to_the_slopes_of_the_smooth_walls():
    """Away from the blade's ends (0.05 c) the wall slopes approach the exact slopes of the
    grid's own smooth walls: lower/upper wall 1.6e-2/3.9e-3 at 31 cells, 2.5e-3/2.5e-3 at
    101, 7.1e-4/8.4e-4 at 201. The walls are C1 piecewise cubics whose curvature jumps at the
    vertices, so the order lies between one and two; the errors are not monotone in
    n_passage (station phase), and these three levels are chosen to show the trend."""
    lower, upper = [], []
    for n_passage in (31, 101, 201):
        errors = blade_slope_errors(SPEC, grid_of(SPEC, n_passage, 5), metrics_of(SPEC, n_passage, 5))
        lower.append(errors[0])
        upper.append(errors[1])
    assert lower[0] > lower[1] > lower[2] and lower[2] < 1e-3
    assert upper[0] > upper[2] and upper[2] < 1e-3


def test_blade_wall_slopes_reach_the_floor_of_the_camber_stations():
    """Against the same blade at 1601 camber stations the slopes stop at 2.9e-3 to 4.3e-3
    from 81 passage cells on: the slopes of the 101-station walls themselves differ by up to
    4.0e-3 (lower) and 4.6e-3 (upper) from the 1601-station walls, at the knots of the
    thickness table (station 0.50 above all), a floor set by n_camber_stations, not by dm."""
    for n_passage in (81, 101, 201):
        errors = blade_slope_errors(FINE, grid_of(SPEC, n_passage, 5), metrics_of(SPEC, n_passage, 5))
        assert 1e-3 < max(errors) < 5e-3, (n_passage, errors)


def test_the_end_stations_carry_the_documented_slope_errors():
    """The limitation the docstring states, measured: at a rounded nose the slope of the
    first blade station is off by more than one (EXAMPLE at 101 cells, 0.07 dm behind the
    nose tip: 3.16 against the wall's 5.30), while at a sharp TE (OTHER) the last station's
    slope stays within 0.2 of the wall's: no vertical tangent there."""
    g, metrics = grid_of(SPEC, 101, 5), metrics_of(SPEC, 101, 5)
    f = g.first_blade_i
    assert abs(metrics.tan_beta0[f] - wall_slopes(SPEC, g.m[f])[0]) > 1.0
    for n_passage in (31, 101, 201):
        g, metrics = grid_of(OTHER, n_passage, 5), metrics_of(OTHER, n_passage, 5)
        last = g.last_blade_i
        exact = wall_slopes(OTHER, g.m[last])
        assert max(abs(metrics.tan_beta0[last] - exact[0]), abs(metrics.tan_beta1[last] - exact[1])) < 0.2


# --- The difference formulas on manufactured functions ---------------------------------------

def test_the_three_stencils_are_exact_on_quadratics():
    """Central, three-point forward and three-point backward differences are exact for a
    quadratic, on the shortest run (three stations) and on longer ones, to round-off."""
    for n in (3, 4, 17):
        for h in (1e-3, 0.37):
            m = 0.3 + h * np.arange(n)
            f = 2.0 - 1.5 * m + 0.8 * m ** 2
            assert np.abs(mt._d_dm_run(f, h) - (-1.5 + 1.6 * m)).max() <= 64 * EPS * (1.0 + np.abs(f).max() / h)


def test_on_a_cubic_the_errors_are_exactly_the_leading_truncation_terms():
    """f = m^3 has f''' = 6 and nothing higher, so the central difference errs by exactly
    h^2 f'''/6 = h^2 and both three-point one-sided forms by exactly -h^2 f'''/3 = -2 h^2,
    at every station: second order with the textbook constants."""
    for h in (0.1, 0.05, 0.025, 0.0125):
        m = -0.4 + h * np.arange(9)
        error = mt._d_dm_run(m ** 3, h) - 3.0 * m ** 2
        assert np.allclose(error[1:-1], h ** 2, rtol=1e-9, atol=0.0)
        assert np.allclose(error[[0, -1]], -2.0 * h ** 2, rtol=1e-9, atol=0.0)


def test_on_a_sine_all_three_stencils_converge_at_second_order():
    """f = sin(m) on [0.1, 1.1], h halved five times: the forward end, the midpoint (a
    station of every run) and the backward end converge at observed orders 1.995 to 2.06,
    with the leading constants -cos(m)/6 inside and +cos(m)/3 at both ends."""
    errors, spacings = {"forward": [], "central": [], "backward": []}, []
    for n_cells in (16, 32, 64, 128, 256, 512):
        m = np.linspace(0.1, 1.1, n_cells + 1)
        h = 1.0 / n_cells
        d = mt._d_dm_run(np.sin(m), h)
        errors["forward"].append(d[0] - math.cos(m[0]))
        errors["central"].append(d[n_cells // 2] - math.cos(m[n_cells // 2]))
        errors["backward"].append(d[-1] - math.cos(m[-1]))
        spacings.append(h)
    for name, values in errors.items():
        orders = observed_orders([abs(v) for v in values], spacings)
        assert all(1.95 < order < 2.1 for order in orders), (name, orders)
    h = spacings[-1]
    assert errors["central"][-1] / h ** 2 == pytest.approx(-math.cos(0.6) / 6.0, rel=1e-4)
    assert errors["forward"][-1] / h ** 2 == pytest.approx(math.cos(0.1) / 3.0, rel=1e-2)
    assert errors["backward"][-1] / h ** 2 == pytest.approx(math.cos(1.1) / 3.0, rel=1e-2)


def test_next_to_a_vertical_tangent_the_one_sided_slope_depends_on_the_tip_distance_in_cells():
    """y = sqrt(2 R s - s^2): a circle whose tangent is vertical at s = 0, as a rounded nose
    is seen along m. With the first station s0 = phi h behind the tip and h << R, the
    forward three-point slope is the fraction g(phi) = -3 phi + 4 sqrt(phi (phi + 1)) -
    sqrt(phi (phi + 2)) of the exact slope, whatever h: 0.51 at phi = 0.072 (the EXAMPLE's
    nose at 101 cells), 0.86 at phi = 0.548 (201 cells). Refining at a fixed phase gains
    nothing; at a fixed s0 the slope converges, at second order, once h << s0. The formula
    is right; the error is the tangent's."""
    def ratio(s0: float, h: float) -> float:
        s = s0 + h * np.arange(3)
        return mt._d_dm_run(np.sqrt(2.0 * s - s ** 2), h)[0] / ((1.0 - s0) / math.sqrt(2.0 * s0 - s0 ** 2))

    for phi in (0.072, 0.548, 2.0):
        g = -3.0 * phi + 4.0 * math.sqrt(phi * (phi + 1.0)) - math.sqrt(phi * (phi + 2.0))
        for h in (1e-4, 1e-6):
            assert ratio(phi * h, h) == pytest.approx(g, abs=1e-4)
    steps = [1e-2 / 2 ** q for q in range(3, 8)]
    orders = observed_orders([abs(ratio(1e-2, h) - 1.0) for h in steps], steps)
    assert all(1.8 < order < 2.1 for order in orders) and orders[-1] > 1.95, orders


# --- Independence and scaling --------------------------------------------------------------

@pytest.mark.parametrize("spec", (SPEC, THIN, LONG_TAIL))
def test_n_eta_changes_no_station_metric(spec):
    """For one m grid, n_eta = 11 to 161 leaves S, dS_dm and both boundary slopes bit for
    bit, and the 2-D fields agree at eta = 0.5, a node of every one of these grids."""
    reference = metrics_of(spec, 61, 11)
    middle = reference.tan_beta[:, 5]
    for n_eta in (11, 21, 41, 81, 101, 161):
        metrics = metrics_of(spec, 61, n_eta)
        for name in ("S", "dS_dm", "tan_beta0", "tan_beta1"):
            np.testing.assert_array_equal(getattr(metrics, name), getattr(reference, name), err_msg=name)
        j = (n_eta - 1) // 2
        assert metrics.grid.eta[j] == 0.5
        np.testing.assert_allclose(metrics.tan_beta[:, j], middle, rtol=0.0, atol=8 * EPS * np.abs(middle).max())


@pytest.mark.parametrize("n_passage", (31, 101))
def test_a_physical_chord_scales_S_and_no_angle(n_passage):
    """S is a length and scales with the chord; slopes, dS/dm and angles are dimensionless and
    do not change; the stations and blade indices are the same."""
    scale = 0.05
    unit, metric = metrics_of(SPEC, n_passage), metrics_of(replace(SPEC, physical_chord=scale), n_passage)
    assert (metric.grid.first_blade_i, metric.grid.last_blade_i, len(metric.grid.m)) == \
           (unit.grid.first_blade_i, unit.grid.last_blade_i, len(unit.grid.m))
    np.testing.assert_allclose(metric.S, scale * unit.S, rtol=1e-12)
    for name in ("dS_dm", "tan_beta0", "tan_beta1", "tan_beta", "beta", "sin_beta", "cos_beta"):
        np.testing.assert_allclose(getattr(metric, name), getattr(unit, name), rtol=0.0, atol=1e-11, err_msg=name)


def test_rebuilding_the_exit_changes_only_the_exit_angles():
    """rebuild_exit to kappa_te + 2, + 5 and - 5 deg: S stays everywhere (both exit lines move
    together), the inlet and blade metrics stay bit for bit, only the exit slopes change, and
    the original angle gives back the original metrics bit for bit."""
    g = grid_of(SPEC, 101)
    start = mt.build_metrics(g)
    kappa_te = cascade_of(SPEC).kappa_te
    kept = slice(0, g.last_blade_i + 1)
    exit_ = slice(g.last_blade_i + 1, None)
    tolerance = 4.0 * np.spacing(max(np.abs(g.nodes[..., 1]).max(), g.pitch) + 1.0)
    state = g
    for turn in (2.0, 5.0, -5.0):
        state = gr.rebuild_exit(state, kappa_te + math.radians(turn))
        metrics = mt.build_metrics(state)
        assert np.abs(metrics.S - start.S).max() <= tolerance
        for item in fields(metrics):
            if item.name != "grid":
                np.testing.assert_array_equal(getattr(metrics, item.name)[kept], getattr(start, item.name)[kept],
                                              err_msg=item.name)
        assert np.all(metrics.tan_beta0[exit_] != start.tan_beta0[exit_])
        exact = side_line_slope(state, state.m[exit_], "exit")
        assert np.abs(metrics.tan_beta0[exit_] - exact).max() < 1e-4
    back = mt.build_metrics(gr.rebuild_exit(state, kappa_te))
    for item in fields(back):
        if item.name != "grid":
            np.testing.assert_array_equal(getattr(back, item.name), getattr(start, item.name), err_msg=item.name)


@pytest.mark.parametrize("spec", EDGE_CASES)
@pytest.mark.parametrize("n_passage", (17, 51, 201))
def test_metrics_build_on_the_edge_cases_of_grid_py(spec, n_passage):
    """Every edge case of grid.py at coarse to fine resolution: metrics build, S > 0, the
    extensions keep S = pitch, and cos(beta) stays clear of the singular threshold."""
    g = grid_of(spec, n_passage)
    metrics = mt.build_metrics(g)
    assert np.all(metrics.S > 0.0) and np.all(metrics.cos_beta > 0.05)
    for start, stop in (runs_of(g)[0], runs_of(g)[2]):
        assert np.abs(metrics.S[start:stop] - g.pitch).max() <= 4.0 * np.spacing(max(np.abs(g.nodes).max(), g.pitch))


# --- Refusals -------------------------------------------------------------------------------

def test_a_short_extension_is_refused_and_names_its_region():
    """Each m region needs three stations for a second-order one-sided difference. A 0.04 c
    upstream (downstream) extension at 31 passage cells holds two stations or fewer on the
    inlet (exit) side lines: a valid Grid, refused here with the region and the cure."""
    g = gr.make_grid(cascade_of(SPEC), GridSettings(31, 5, upstream_extension_chords=0.04), math.radians(45.0))
    assert g.first_blade_i < 3
    with pytest.raises(mt.MetricsError, match="upstream periodic run holds .* lengthen upstream_extension_chords"):
        mt.build_metrics(g)
    g = gr.make_grid(cascade_of(SPEC), GridSettings(31, 5, downstream_extension_chords=0.04), math.radians(45.0))
    assert len(g.m) - 1 - g.last_blade_i < 3
    with pytest.raises(mt.MetricsError, match="downstream periodic run holds .* lengthen downstream_extension_chords"):
        mt.build_metrics(g)


def test_a_blade_run_of_two_stations_is_refused():
    """At two passage cells the EXAMPLE's blade holds two stations: a valid Grid, refused
    here with the cure (raise n_passage). The same check on raw arrays as well."""
    g = gr.make_grid(cascade_of(SPEC), GridSettings(2, 5), math.radians(45.0))
    assert g.last_blade_i - g.first_blade_i + 1 == 2
    with pytest.raises(mt.MetricsError, match="the blade run holds 2 station.*raise n_passage"):
        mt.build_metrics(g)
    arrays = raw(grid_of(SPEC))
    arrays["last_blade_i"] = arrays["first_blade_i"] + 1
    with pytest.raises(mt.MetricsError, match="the blade run holds 2 station.*raise n_passage"):
        mt._metric_fields(**arrays)


@pytest.mark.parametrize("width, refused", ((1e-14, True), (1e-12, False), (1e-10, False)))
def test_the_width_threshold_sits_at_round_off(width, refused):
    """The S backstop is round-off of the tangential coordinates, 64 eps max(|y|, pitch)
    (2.7e-14 c for the EXAMPLE), not an engineering tolerance: a width of 1e-14 c is refused
    with its station and value; 1e-12 c (grid.py's TOUCH) and 1e-10 c are kept."""
    g = grid_of(SPEC)
    i = (g.first_blade_i + g.last_blade_i) // 2
    arrays = raw(g)
    arrays["nodes"][i, :, 1] = arrays["nodes"][i, 0, 1] + arrays["eta"] * width
    if refused:
        with pytest.raises(mt.MetricsError, match=rf"S = y1 - y0 = [0-9.]+e-1[45] at station {i} "):
            mt._metric_fields(**arrays)
    else:
        assert mt._metric_fields(**arrays)["S"][i] == pytest.approx(width, rel=1e-3)


@pytest.mark.parametrize("name", ("m", "eta"))
@pytest.mark.parametrize("value", (math.nan, math.inf))
def test_stations_that_are_not_finite_are_refused(name, value):
    arrays = raw(grid_of(SPEC))
    arrays[name][2] = value
    with pytest.raises(mt.MetricsError, match=rf"{name} is not finite at index \(2,\)"):
        mt._metric_fields(**arrays)


@pytest.mark.parametrize("pitch", (0.0, -1.0, math.nan, math.inf))
def test_a_pitch_that_is_not_positive_and_finite_is_refused(pitch):
    arrays = raw(grid_of(SPEC))
    arrays["pitch"] = pitch
    with pytest.raises(mt.MetricsError, match="must be finite and positive"):
        mt._metric_fields(**arrays)


@pytest.mark.parametrize("boundary", ("lower_boundary", "upper_boundary"))
def test_boundaries_that_are_not_the_wall_rows_of_the_nodes_are_refused(monkeypatch, boundary):
    """build_metrics checks that the Grid's boundaries are its eta = 0 and eta = 1 rows."""
    g = grid_of(SPEC)
    monkeypatch.setattr(gr.Grid, boundary, property(lambda self: self.nodes[:, 1]))
    with pytest.raises(mt.MetricsError, match="are not the eta = 0 and eta = 1 rows"):
        mt.build_metrics(g)


def test_a_grid_whose_eta_spacing_attribute_disagrees_is_refused(monkeypatch):
    g = grid_of(SPEC)
    monkeypatch.setattr(gr.Grid, "deta", property(lambda self: 1.001 / (len(self.eta) - 1)))
    with pytest.raises(mt.MetricsError, match="grid.deta"):
        mt.build_metrics(g)


def test_a_collapsed_or_reversed_passage_is_refused():
    """S at or below round-off, or negative, at one blade station: the map from (m, eta) is
    singular or reversed there, and the error names the station and the value."""
    g = grid_of(SPEC)
    i = (g.first_blade_i + g.last_blade_i) // 2
    for width in (0.0, -1e-3, 1e-17):
        arrays = raw(g)
        y0 = arrays["nodes"][i, 0, 1]
        arrays["nodes"][i, :, 1] = y0 + arrays["eta"] * width
        with pytest.raises(mt.MetricsError, match=f"at station {i} "):
            mt._metric_fields(**arrays)


@pytest.mark.parametrize("name", ("m", "eta"))
def test_a_nonuniform_spacing_is_refused(name):
    arrays = raw(grid_of(SPEC))
    arrays[name][3] += 1e-6 * (arrays[name][4] - arrays[name][3])
    if name == "m":
        arrays["nodes"][3, :, 0] = arrays["m"][3]
    else:
        y0, y1 = arrays["nodes"][:, 0, 1], arrays["nodes"][:, -1, 1]
        arrays["nodes"][:, :, 1] = (1 - arrays["eta"])[None, :] * y0[:, None] + arrays["eta"][None, :] * y1[:, None]
    with pytest.raises(mt.MetricsError, match=f"{name} is not uniformly spaced"):
        mt._metric_fields(**arrays)


@pytest.mark.parametrize("name", ("m", "eta"))
def test_a_spacing_that_is_not_increasing_is_refused(name):
    arrays = raw(grid_of(SPEC))
    arrays[name] = arrays[name][::-1].copy()
    with pytest.raises(mt.MetricsError, match=f"{name} is not strictly increasing"):
        mt._metric_fields(**arrays)


@pytest.mark.parametrize("value", (math.nan, math.inf))
def test_a_node_that_is_not_finite_is_refused(value):
    arrays = raw(grid_of(SPEC))
    arrays["nodes"][40, 7, 1] = value
    with pytest.raises(mt.MetricsError, match=r"nodes is not finite at index \(40, 7, 1\)"):
        mt._metric_fields(**arrays)


def test_a_node_off_the_linear_eta_map_is_refused():
    arrays = raw(grid_of(SPEC))
    arrays["nodes"][50, 10, 1] += 1e-9
    with pytest.raises(mt.MetricsError, match=r"the node \(50, 10\) is .* off y0 \+ eta S"):
        mt._metric_fields(**arrays)


def test_a_node_off_its_station_line_is_refused():
    arrays = raw(grid_of(SPEC))
    arrays["nodes"][50, 10, 0] += 1e-9
    with pytest.raises(mt.MetricsError, match="the nodes of station 50 do not lie on its line"):
        mt._metric_fields(**arrays)


def test_side_lines_that_are_not_one_pitch_apart_are_refused():
    arrays = raw(grid_of(SPEC))
    eta = arrays["eta"]
    for i in (5, len(arrays["m"]) - 5):
        y0 = arrays["nodes"][i, 0, 1]
        arrays_i = {key: (value.copy() if isinstance(value, np.ndarray) else value) for key, value in arrays.items()}
        arrays_i["nodes"][i, :, 1] = y0 + eta * (arrays["pitch"] * (1.0 + 1e-9))
        with pytest.raises(mt.MetricsError, match=f"periodic station {i} is not the pitch"):
            mt._metric_fields(**arrays_i)


def test_a_numerically_vertical_grid_line_is_refused():
    """A boundary that jumps by 1.5e8 dm between two stations makes tan(beta) 7.5e7 and
    cos(beta) 1.3e-8, below sqrt(machine epsilon) = 1.49e-8: singular, refused with the node,
    beta in degrees and cos."""
    g = grid_of(SPEC)
    arrays = raw(g)
    i = (g.first_blade_i + g.last_blade_i) // 2
    jump = 1.5e8 * g.dm
    arrays["nodes"][i + 1:, :, 1] += jump                # every later station moves up together
    arrays["pitch"] = g.pitch
    with pytest.raises(mt.MetricsError, match=r"cos\(beta\) = .* at node \(\d+, \d+\), beta = 89.99"):
        mt._metric_fields(**arrays)


def test_steep_but_finite_grid_lines_are_kept():
    """STEEP's nose turns its lower wall to 84 deg at 201 passage cells (cos 0.107, the
    smallest of the edge cases at the resolutions tested here; it falls further with dm at a
    rounded nose, 0.069 at 401): an ordinary steep slope, far from the singular threshold, and
    kept."""
    metrics = metrics_of(STEEP, 201)
    assert np.degrees(np.abs(metrics.beta).max()) > 80.0
    assert metrics.cos_beta.min() > 1e6 * mt.SINGULAR_COS


def test_badly_shaped_arrays_are_refused():
    arrays = raw(grid_of(SPEC))
    for change in ({"nodes": arrays["nodes"][:, :, :1]}, {"eta": arrays["eta"][:1]}, {"m": arrays["m"][:, None]}):
        with pytest.raises(mt.MetricsError):
            mt._metric_fields(**{**arrays, **change})


def test_a_grid_whose_spacing_attributes_disagree_is_refused(monkeypatch):
    """build_metrics checks grid.dm and grid.deta against the stations themselves."""
    g = grid_of(SPEC)
    monkeypatch.setattr(gr.Grid, "dm", property(lambda self: 1.001 * (self.m[-1] - self.m[0]) / (len(self.m) - 1)))
    with pytest.raises(mt.MetricsError, match="grid.dm"):
        mt.build_metrics(g)


# --- C1 regression lock and the new fields on C1 ---------------------------------------------

C1_METRICS_FINGERPRINT = {      # 101 x 101 at inlet 45 deg, taken from metrics.py before the C2 hand-off (2026-11-11)
    "SPEC": {"slope_sums": (231.9675213796899, 228.96974153005675, 334.61136937507376),
             "first_slopes": (3.1594636242399625, -1.331100596411298),
             "cos_beta": (0.30175544835055756, 28633.163614053694)},
    "OTHER": {"slope_sums": (-54.898812459251545, -54.34584629952968, 207.12835721670433),
              "first_slopes": (0.19102507170187902, -0.5671144301672018),
              "cos_beta": (0.7071501675675371, 29500.143926543107)},
}


@pytest.mark.parametrize("name, spec", (("SPEC", SPEC), ("OTHER", OTHER)))
def test_c1_metrics_fingerprint(name, spec):
    """A C1 grid's metrics are the validated finite-difference metrics: sums of tan_beta0,
    tan_beta1 and S, the first blade slopes and cos(beta) as before the C2 hand-off (every
    field was also compared bit for bit on 55 C1 grids when it was made)."""
    metrics = metrics_of(spec, 101, 101)
    f = metrics.grid.first_blade_i
    measured = {"slope_sums": (metrics.tan_beta0.sum(), metrics.tan_beta1.sum(), metrics.S.sum()),
                "first_slopes": (metrics.tan_beta0[f], metrics.tan_beta1[f]),
                "cos_beta": (metrics.cos_beta.min(), metrics.cos_beta.sum())}
    for key, value in measured.items():
        np.testing.assert_allclose(value, C1_METRICS_FINGERPRINT[name][key], rtol=1e-13, atol=1e-16, err_msg=key)
    assert not metrics.exact_wall_slopes


@pytest.mark.parametrize("spec", (SPEC, OTHER, STEEP))
def test_c1_m_derivatives_of_the_slopes_are_their_region_differences(spec):
    """On a C1 grid the new dtan_beta0/1_dm are the region-wise second-order differences of
    tan_beta0/1 (region_derivative, written out here), no stencil crossing a transition, and
    dtan_beta_dm is their affine combination in eta."""
    g, metrics = grid_of(spec, 61), metrics_of(spec, 61)
    runs = runs_of(g)
    for slope, derivative in ((metrics.tan_beta0, metrics.dtan_beta0_dm), (metrics.tan_beta1, metrics.dtan_beta1_dm)):
        expected = region_derivative(slope, g.dm, runs)
        np.testing.assert_allclose(derivative, expected, rtol=0.0, atol=64 * EPS * np.abs(slope).max() / g.dm)
    eta = g.eta
    np.testing.assert_array_equal(metrics.dtan_beta_dm, (1.0 - eta) * metrics.dtan_beta0_dm[:, None]
                                  + eta * metrics.dtan_beta1_dm[:, None])


# --- C2: the blade-wall slopes are the authoritative curve's derivatives ----------------------

C2 = replace(SPEC, surface_continuity="C2")
C2_OTHER = replace(OTHER, surface_continuity="C2")
C2_DIMENSIONAL = replace(C2, physical_chord=0.05)
C2_RESOLUTIONS = (31, 51, 81, 101, 141, 201, 401)


def direct_wall_slopes(spec: CascadeSpec, g: gr.Grid) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """(tan(beta), d tan(beta)/dm) at each wall's nodes from geometry.py's own wall_slopes
    on the curve the wall identity names, differentiated directly at its u: no grid.py or
    metrics.py code path."""
    curves = cascade_of(spec).active_curves
    out = {}
    for side in ("lower", "upper"):
        wall = getattr(g.wall_identity, side)
        d1 = np.array([getattr(curves, name).derivative(u, 1) for name, u in zip(wall.surface, wall.u)])
        d2 = np.array([getattr(curves, name).derivative(u, 2) for name, u in zip(wall.surface, wall.u)])
        out[side] = geometry.wall_slopes(d1, d2)
    return out


def richardson_wall_slopes(spec: CascadeSpec, g: gr.Grid, side: str, k: int) -> tuple[float, float]:
    """dy/dm and d2y/dm2 of the wall at blade station k by Richardson-extrapolated central
    differences of y(m) on the curve itself, each y(m) a brentq root of m(u) = m near the
    node: no derivative formula at all."""
    curves = cascade_of(spec).active_curves
    wall = getattr(g.wall_identity, side)
    curve, u = getattr(curves, wall.surface[k]), float(wall.u[k])
    m = g.m[g.first_blade_i + k]
    m_u, m_uu = float(curve.derivative(u, 1)[0]), float(curve.derivative(u, 2)[0])
    h = max(1e-6, min(1e-4, 0.05 * m_u ** 2 / max(1e-12, abs(m_uu))))

    def y_at(x: float) -> float:
        reach = 4.0 * abs(x - m) / abs(m_u) + 1e-12
        root = brentq(lambda s: float(curve.evaluate(s)[0]) - x, max(0.0, u - reach), min(1.0, u + reach), xtol=1e-16)
        return float(curve.evaluate(root)[1])

    def first(step: float) -> float:
        return (y_at(m + step) - y_at(m - step)) / (2.0 * step)

    def second(step: float) -> float:
        return (y_at(m + step) - 2.0 * y_at(m) + y_at(m - step)) / step ** 2

    return (4.0 * first(h / 2) - first(h)) / 3.0, (4.0 * second(h / 2) - second(h)) / 3.0


@pytest.mark.parametrize("spec", (C2, C2_OTHER))
@pytest.mark.parametrize("n_passage", C2_RESOLUTIONS)
def test_c2_wall_slopes_are_the_curve_derivatives_at_the_wall_nodes(spec, n_passage):
    """On the C2 blade walls tan_beta0/1 = y_u / m_u and dtan_beta0/1_dm = (y_uu m_u - y_u m_uu)
    / m_u^3 of the curve and u each wall node lies on: equal to geometry.wall_slopes on the
    curve's own derivatives (measured: bit for bit), at every blade station, the first and
    last included; no one-sided difference is left at the ends. Finite throughout."""
    g, metrics = grid_of(spec, n_passage), metrics_of(spec, n_passage)
    assert metrics.exact_wall_slopes
    blade = slice(g.first_blade_i, g.last_blade_i + 1)
    reference = direct_wall_slopes(spec, g)
    for side, slope, derivative in (("lower", metrics.tan_beta0, metrics.dtan_beta0_dm),
                                    ("upper", metrics.tan_beta1, metrics.dtan_beta1_dm)):
        np.testing.assert_allclose(slope[blade], reference[side][0], rtol=1e-15, atol=0.0)
        np.testing.assert_allclose(derivative[blade], reference[side][1], rtol=1e-15, atol=0.0)
    for item in fields(metrics):
        if item.name != "grid":
            assert np.all(np.isfinite(getattr(metrics, item.name))), item.name


@pytest.mark.parametrize("spec", (C2, C2_OTHER))
@pytest.mark.parametrize("n_passage", (31, 101))
def test_c2_wall_slopes_match_a_difference_of_the_curve_itself(spec, n_passage):
    """Against Richardson differences of y(m) taken on the curve (brentq roots, no formula),
    at every third blade station 0.02 c or more from the nose tip and the tail end: tan(beta)
    within 1e-8 and d tan(beta)/dm within 1e-5, relative to max(1, |value|) (measured at
    most 4.8e-9 and 3.6e-6, median 1e-12 and 3e-8 over 31 to 401 cells; closer to a vertical
    tangent the difference itself degrades, not the exact value)."""
    g, metrics = grid_of(spec, n_passage), metrics_of(spec, n_passage)
    f, l = g.first_blade_i, g.last_blade_i
    curves = cascade_of(spec).active_curves
    samples = np.concatenate([curve.evaluate(np.linspace(0.0, 1.0, 4001))[:, 0] for curve in (curves.suction, curves.pressure)])
    tip, tail = np.min(samples), np.max(samples)
    checked = 0
    for k in range(0, l - f + 1, 3):
        if min(g.m[f + k] - tip, tail - g.m[f + k]) < 0.02 * g.chord:
            continue
        for side, slope, derivative in (("lower", metrics.tan_beta0, metrics.dtan_beta0_dm),
                                        ("upper", metrics.tan_beta1, metrics.dtan_beta1_dm)):
            tan_ref, dtan_ref = richardson_wall_slopes(spec, g, side, k)
            assert abs(slope[f + k] - tan_ref) <= 1e-8 * max(1.0, abs(tan_ref)), (side, k)
            assert abs(derivative[f + k] - dtan_ref) <= 1e-5 * max(1.0, abs(dtan_ref)), (side, k)
            checked += 1
    assert checked > 10


def test_c2_the_extensions_keep_their_differences():
    """Only the blade run changes path: on the inlet and exit side lines every field of the
    C2 metrics equals, bit for bit, the metrics of the same cascade read from its sampled
    arrays (the pre-hand-off C2 path), whose blade walls are differenced; so no stencil
    reaches from the blade run into an extension, and the extension model is unchanged."""
    c = cascade_of(C2)
    legacy = mt.build_metrics(gr.make_grid(replace(c, active_curves=None, normalized_curves=None),
                                           GridSettings(n_passage=61, n_eta=21), math.radians(45.0)))
    metrics = metrics_of(C2, 61)
    g = metrics.grid
    assert (g.first_blade_i, g.last_blade_i) == (legacy.grid.first_blade_i, legacy.grid.last_blade_i)
    sides = np.r_[0:g.first_blade_i, g.last_blade_i + 1:len(g.m)]
    for item in fields(metrics):
        if item.name != "grid":
            np.testing.assert_array_equal(getattr(metrics, item.name)[sides], getattr(legacy, item.name)[sides],
                                          err_msg=item.name)
    blade = slice(g.first_blade_i, g.last_blade_i + 1)
    assert not np.array_equal(metrics.tan_beta0[blade], legacy.tan_beta0[blade])


@pytest.mark.parametrize("spec", (C2, C2_OTHER))
def test_c2_the_passage_fields_follow_the_exact_eta_relations(spec):
    """Inside the passage nothing is evaluated on a curve: tan(beta) and d tan(beta)/dm are
    the affine combinations (1 - eta) wall0 + eta wall1 of the mapping, exact at eta = 0 and
    1; d tan(beta)/d eta is tan_beta1 - tan_beta0, the array dS_dm itself; and
    beta, sin and cos follow from tan(beta) by the stable identities."""
    g, metrics = grid_of(spec, 101, 31), metrics_of(spec, 101, 31)
    eta = g.eta
    np.testing.assert_array_equal(metrics.tan_beta, (1.0 - eta) * metrics.tan_beta0[:, None] + eta * metrics.tan_beta1[:, None])
    np.testing.assert_array_equal(metrics.dtan_beta_dm, (1.0 - eta) * metrics.dtan_beta0_dm[:, None]
                                  + eta * metrics.dtan_beta1_dm[:, None])
    np.testing.assert_array_equal(metrics.tan_beta[:, 0], metrics.tan_beta0)
    np.testing.assert_array_equal(metrics.tan_beta[:, -1], metrics.tan_beta1)
    np.testing.assert_array_equal(metrics.dtan_beta_dm[:, -1], metrics.dtan_beta1_dm)
    assert metrics.dtan_beta_deta is metrics.dS_dm
    np.testing.assert_array_equal(metrics.dS_dm, metrics.tan_beta1 - metrics.tan_beta0)
    cos = 1.0 / np.hypot(1.0, metrics.tan_beta)
    np.testing.assert_array_equal(metrics.cos_beta, cos)
    np.testing.assert_array_equal(metrics.sin_beta, metrics.tan_beta * cos)
    np.testing.assert_array_equal(metrics.beta, np.arctan(metrics.tan_beta))


def test_c2_dS_dm_and_its_derivative_are_those_of_the_curves_width():
    """S(m) = y_upper(m) - y_lower(m) taken on the curves (brentq roots), differenced by
    Richardson at every fifth inner blade station: dS_dm within 1e-7 (measured at most
    3.3e-8, median 1.7e-12), and the second difference within 1e-6 of
    dtan_beta1_dm - dtan_beta0_dm, relative to max(1, |value|) (measured 1e-7, median
    1.4e-8). d2S/dm2 is not a field: that difference is all a consumer needs."""
    g, metrics = grid_of(C2, 101), metrics_of(C2, 101)
    curves = cascade_of(C2).active_curves
    f, l = g.first_blade_i, g.last_blade_i
    identity = g.wall_identity

    def wall_y(name: str, u: float, x: float) -> float:
        curve = getattr(curves, name)
        root = brentq(lambda s: float(curve.evaluate(s)[0]) - x, max(0.0, u - 2e-3), min(1.0, u + 2e-3), xtol=1e-16)
        return float(curve.evaluate(root)[1])

    assert not hasattr(metrics, "d2S_dm2")
    for k in range(1, l - f, 5):
        m = g.m[f + k]

        def width(x: float) -> float:
            return wall_y(identity.upper.surface[k], identity.upper.u[k], x) + g.pitch \
                - wall_y(identity.lower.surface[k], identity.lower.u[k], x)

        h = 2e-4
        first = [(width(m + s) - width(m - s)) / (2 * s) for s in (h, h / 2)]
        second = [(width(m + s) - 2 * width(m) + width(m - s)) / s ** 2 for s in (h, h / 2)]
        d1, d2 = (4 * first[1] - first[0]) / 3, (4 * second[1] - second[0]) / 3
        assert abs(metrics.dS_dm[f + k] - d1) <= 1e-7, k
        assert abs(metrics.dtan_beta1_dm[f + k] - metrics.dtan_beta0_dm[f + k] - d2) <= 1e-6 * max(1.0, abs(d2)), k


def test_c2_n_eta_changes_no_station_metric():
    """n_eta 11 to 101 at 101 passage cells: every station field, the exact wall slopes and
    their m derivatives included, is identical bit for bit, and the passage fields at
    eta = 0.5 agree to round-off."""
    reference = metrics_of(C2, 101, 11)
    for n_eta in (11, 31, 61, 101):
        metrics = metrics_of(C2, 101, n_eta)
        for name in ("S", "dS_dm", "tan_beta0", "tan_beta1", "dtan_beta0_dm", "dtan_beta1_dm"):
            np.testing.assert_array_equal(getattr(metrics, name), getattr(reference, name), err_msg=name)
        j = (n_eta - 1) // 2
        assert metrics.grid.eta[j] == 0.5
        for name in ("tan_beta", "dtan_beta_dm"):
            middle = getattr(reference, name)[:, 5]
            np.testing.assert_allclose(getattr(metrics, name)[:, j], middle, rtol=0.0, atol=8 * EPS * np.abs(middle).max())


@pytest.mark.parametrize("n_passage", (31, 101))
def test_c2_a_physical_chord_scales_lengths_and_not_angles(n_passage):
    """The 0.05 m twin: S scales with the chord (1e-14 relative; measured 1.9e-15), tan(beta),
    beta and dS_dm do not change and d tan(beta)/dm scales with 1/c, each within 1e-12 and
    2e-11 of its largest value (measured 1.5e-13 and 4.5e-12: u moves by round-off with the
    station m, and the walls change fastest along u next to the nose tip)."""
    unit, metric = metrics_of(C2, n_passage), metrics_of(C2_DIMENSIONAL, n_passage)
    scale = C2_DIMENSIONAL.physical_chord
    assert (metric.grid.first_blade_i, metric.grid.last_blade_i) == (unit.grid.first_blade_i, unit.grid.last_blade_i)
    np.testing.assert_allclose(metric.S, scale * unit.S, rtol=1e-14, atol=0.0)
    for name in ("tan_beta", "beta", "dS_dm", "tan_beta0", "tan_beta1"):
        reference = getattr(unit, name)
        np.testing.assert_allclose(getattr(metric, name), reference, rtol=0.0, atol=1e-12 * np.abs(reference).max(),
                                   err_msg=name)
    for name in ("dtan_beta_dm", "dtan_beta0_dm", "dtan_beta1_dm"):
        reference = getattr(unit, name)
        np.testing.assert_allclose(scale * getattr(metric, name), reference, rtol=0.0,
                                   atol=2e-11 * np.abs(reference).max(), err_msg=name)


def test_c2_the_large_le_angle_is_the_curve_tangent():
    """At 101 cells the first blade station lies 0.066 dm behind the nose tip: the lower
    wall's beta there is 79.16 deg and the upper wall's -75.63 deg (79.25, -75.49 with the
    legacy PCHIP thickness), the angles of the curves' own tangents at those points (atan2
    of X_u), so the steep angle is geometry, not a difference; the one-sided difference of
    the sampled reading gave 72.3 and -56.1 deg (72.4, -56.1)."""
    g, metrics = grid_of(C2, 101), metrics_of(C2, 101)
    f = g.first_blade_i
    tangents = [d[0] for d in g.wall_curve_derivative(1)]
    for slope, tangent, expected in ((metrics.tan_beta0[f], tangents[0], 79.164),      # measured 79.16421
                                     (metrics.tan_beta1[f], tangents[1], -75.630)):    # measured -75.62994
        angle = math.degrees(math.atan(slope))
        assert angle == pytest.approx(expected, abs=1e-3)
        direction = math.degrees(math.atan2(tangent[1], tangent[0]))
        assert angle == pytest.approx(direction if abs(direction) <= 90.0 else direction - math.copysign(180.0, direction),
                                      abs=1e-9)
    c = cascade_of(C2)
    legacy = mt.build_metrics(gr.make_grid(replace(c, active_curves=None, normalized_curves=None),
                                           GridSettings(n_passage=101, n_eta=21), math.radians(45.0)))
    assert [round(math.degrees(math.atan(x)), 1) for x in (legacy.tan_beta0[f], legacy.tan_beta1[f])] == [72.3, -56.1]


def test_c2_a_numerically_vertical_wall_tangent_is_refused_not_capped():
    """A C2 wall node whose tangent has |m_u| <= SINGULAR_COS |X_u| is a MetricsError naming
    the wall and the station; at twice that threshold the values are kept as they come,
    y_u / m_u and the parametric formula, nothing clipped or capped (tan(beta) about 3.4e7,
    d tan(beta)/dm about 1e23 here), and they pass on into the passage fields."""
    g = grid_of(C2, 31)
    walls = list(g.wall_curve_derivative(1)) + list(g.wall_curve_derivative(2))
    arrays = raw(g)
    k, i = 3, g.first_blade_i + 3
    for factor, refused in ((0.5, True), (2.0, False)):
        changed = [np.array(w) for w in walls]
        speed = math.hypot(*changed[0][k])
        changed[0][k] = (factor * mt.SINGULAR_COS * speed, speed)
        if refused:
            with pytest.raises(mt.MetricsError, match=f"lower blade wall is numerically vertical at blade station {i} "):
                mt._metric_fields(**arrays, wall_derivatives=tuple(changed))
            continue
        values = mt._metric_fields(**arrays, wall_derivatives=tuple(changed))
        (m_u, y_u), (m_uu, y_uu) = changed[0][k], changed[2][k]
        assert values["tan_beta0"][i] == y_u / m_u
        assert values["dtan_beta0_dm"][i] == (y_uu * m_u - y_u * m_uu) / m_u ** 3
        assert values["tan_beta0"][i] == pytest.approx(1.0 / (2.0 * mt.SINGULAR_COS), rel=1e-6)
        assert values["tan_beta"][i, 0] == values["tan_beta0"][i] and np.isfinite(values["dtan_beta_dm"][i]).all()
        assert values["cos_beta"][i, 0] == pytest.approx(2.0 * mt.SINGULAR_COS, rel=1e-6)


def test_c2_wall_derivatives_that_do_not_belong_to_the_nodes_are_refused(monkeypatch):
    """build_metrics takes the C2 wall derivatives only after checking that the grid's
    order-0 curve points are its wall nodes."""
    g = grid_of(C2, 31)
    honest = gr.Grid.wall_curve_derivative

    def shifted(self, order):
        lower, upper = honest(self, order)
        return (lower + 1e-9, upper) if order == 0 else (lower, upper)

    monkeypatch.setattr(gr.Grid, "wall_curve_derivative", shifted)
    with pytest.raises(mt.MetricsError, match="lower wall's curve points"):
        mt.build_metrics(g)


def test_c2_rebuilding_the_exit_changes_only_the_exit_metrics():
    """A rebuilt exit leaves every inlet and blade-run field of the C2 metrics bit for bit,
    the exact wall slopes included."""
    g = grid_of(C2, 101)
    before = mt.build_metrics(g)
    after = mt.build_metrics(gr.rebuild_exit(g, cascade_of(C2).kappa_te + math.radians(6.0)))
    kept = slice(0, g.last_blade_i + 1)
    for item in fields(before):
        if item.name != "grid":
            np.testing.assert_array_equal(getattr(after, item.name)[kept], getattr(before, item.name)[kept], err_msg=item.name)
    assert np.any(after.tan_beta0[g.last_blade_i + 1:] != before.tan_beta0[g.last_blade_i + 1:])
