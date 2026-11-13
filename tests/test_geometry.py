"""
Tests for turbolenza.b2b.geometry.

Every test builds blades over a range of parameters and checks them against something the
module does not compute itself: a rotation matrix, Aungier's Eqs. 4-6, 4-30 and 4-31 written
out here, Table 4-1 as printed, a brute-force polyline distance, a finer build of the same
blade, or a second build with one parameter changed.

Two geometries are frozen on purpose. test_c1_baseline_fingerprint locks the validated C1
geometry (surface_continuity "C1", the legacy baseline, the default until 2026-11-14) to
numbers taken from the module before the C2 representation was added. Bit-for-bit equality
with that module was checked once on 200 cascades and every field; the lock allows 1e-13 so
that a platform's last-bit differences in libm do not fail it. test_c2_frozen_fingerprint
locks the C2 design geometry (the robust LE and the natural quintic thickness, the default
since 2026-11-14) to numbers taken when it was frozen; its 1e-9 covers the measured 4.2e-11
response of the C2 fit to a one-ulp input change. DEFAULT_SPEC below states "C1", so the
tests of the C1 construction keep testing C1.

The thickness reconstruction (CascadeSpec.thickness_reconstruction) defaults to the legacy
PCHIP for C1 and to the natural quintic for C2 (2026-11-13); C2 is compared with C1 built with
the same reconstruction, and the tests of the shared construction run with both.

The C2 tests (surface_continuity "C2") check the quintic curves against independent
evaluations: scipy's piecewise polynomials for the one-sided values at every knot, closed
forms for the curvature and slope formulas, the analytic C1 surface for the deviation, and
brute-force segment crossings for the topology. C2 has a redesigned LE (the robust LE): it
is held to 1e-6 c of C1 downstream of each LE anchor only, and the LE tests check what the
redesign promises: the legacy nose radius and apex, G2 joins measured on the numbers, no
inflection or reverse turn, the legacy defects gone, continuity in the blade parameters (a
halving test), independence of the anchor scan step, scaling with the chord, the engine
with a prescribed radius, and its least-squares solver against SLSQP. The previous C2 fit,
which keeps the legacy LE fair (fit_quintic_surface), keeps its own whole-surface tests.
"""

from __future__ import annotations

import ast
import inspect
import math
import subprocess
import sys
from dataclasses import FrozenInstanceError, fields, replace
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from turbolenza.b2b import geometry as g

# C1 stated explicitly: these tests check the validated C1 construction, which was the default until
# 2026-11-14; the C2 tests ask for "C2" through spec_with / build.
DEFAULT_SPEC = g.CascadeSpec(blade_type="NACA65", solidity=1.25, stagger_deg=25.0, cl0=0.8, t_max_over_c=0.08,
                             surface_continuity="C1")
NOMINAL_PEAK = 0.9992       # Table 4-1: the "10 %" 65-010 column peaks at 9.992 % (x/c 0.40)
LEGACY = g.DEFAULT_THICKNESS_RECONSTRUCTION["C1"]           # "pchip_sqrt"
C2_THICKNESS = g.DEFAULT_THICKNESS_RECONSTRUCTION["C2"]     # "quintic_natural_sqrt"


def spec_with(**changes) -> g.CascadeSpec:
    """DEFAULT_SPEC with some fields changed."""
    return replace(DEFAULT_SPEC, **changes)


@lru_cache(maxsize=None)
def _cached(spec: g.CascadeSpec) -> g.Cascade:
    return g.make_cascade(spec)


def build(**changes) -> g.Cascade:
    """The cascade of DEFAULT_SPEC with some fields changed. Cached: a spec is hashable and a
    Cascade is immutable."""
    return _cached(spec_with(**changes))


def rotated(points: np.ndarray, angle: float) -> np.ndarray:
    """Points turned counterclockwise by angle, written independently of the module."""
    c, s = math.cos(angle), math.sin(angle)
    return np.asarray(points) @ np.array([[c, -s], [s, c]]).T


def polyline_distance(points: np.ndarray, polyline: np.ndarray) -> np.ndarray:
    """Distance from each point to a polyline, by brute force over every segment."""
    start, step = polyline[None, :-1], np.diff(polyline, axis=0)[None]
    offset = points[:, None, :] - start
    along = np.clip(np.einsum("ijk,ijk->ij", offset, step) / np.einsum("ijk,ijk->ij", step, step), 0.0, 1.0)
    return np.min(np.linalg.norm(offset - along[..., None] * step, axis=2), axis=1)


def max_thickness(coordinates: g.CascadeCoordinates) -> float:
    """Largest distance from the suction surface to the pressure surface."""
    return float(polyline_distance(coordinates.suction_surface_unstaggered,
                                   coordinates.pressure_surface_unstaggered).max())


def shoelace(loop: np.ndarray) -> float:
    x, y = loop[:, 0], loop[:, 1]
    return 0.5 * float(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))


# --- independence -------------------------------------------------------------------------

def test_geometry_imports_only_the_standard_library_numpy_and_scipy():
    tree = ast.parse(Path(g.__file__).read_text(encoding="utf-8"))
    imported = {alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import)
                for alias in node.names}
    imported |= {node.module.split(".")[0] for node in ast.walk(tree)
                 if isinstance(node, ast.ImportFrom) and node.module}
    assert imported <= set(sys.stdlib_module_names) | {"numpy", "scipy"}, imported


def test_geometry_runs_with_the_other_project_packages_blocked():
    """A fresh interpreter in which naca65 and b2b cannot be imported still builds a cascade."""
    code = ("import sys; sys.modules.update(naca65=None, b2b=None); "
            "from turbolenza.b2b.geometry import CascadeSpec, make_cascade; "
            "spec = CascadeSpec(blade_type='NACA65', solidity=1.1, stagger_deg=20.0, cl0=0.6, t_max_over_c=0.06); "
            "c = make_cascade(spec); "
            "assert 0.0 < c.throat_over_pitch < 1.0")
    src_root = Path(g.__file__).resolve().parents[2]        # src/: turbolenza importable from the cwd
    subprocess.run([sys.executable, "-c", code], cwd=src_root, check=True)


# --- CascadeSpec: the one input object -----------------------------------------------------

def test_cascade_spec_holds_the_only_modelling_defaults():
    spec = g.CascadeSpec(blade_type="NACA65", solidity=1.4, stagger_deg=20.0, cl0=0.9, t_max_over_c=0.07)
    assert (spec.meanline_type, spec.profile_variant, spec.physical_chord, spec.n_camber_stations,
            spec.surface_continuity, spec.thickness_reconstruction) == \
        ("A10", "kovach_sandercock", None, 101, "C2", None)
    assert dict(g.DEFAULT_THICKNESS_RECONSTRUCTION) == {"C1": "pchip_sqrt", "C2": "quintic_natural_sqrt"}
    # Nothing behind make_cascade has a default of its own.
    assert list(inspect.signature(g.make_cascade).parameters) == ["spec"]
    builder = inspect.signature(g.naca65_profile).parameters
    for name in ("meanline_type", "profile_variant", "n_camber_stations", "thickness_reconstruction"):
        assert builder[name].default is inspect.Parameter.empty
    for function in (g.naca65_base_thickness, g.naca65_reference_surfaces):
        assert inspect.signature(function).parameters["thickness_reconstruction"].default is inspect.Parameter.empty
    # The cascade is built with the defaults the spec carries.
    c = g.make_cascade(spec)
    assert c.metadata["n_camber_stations"] == 101 and "A10" in c.metadata["designation"]
    assert c.normalized.trailing_edge_radius == pytest.approx(0.008, abs=1e-15)    # Kovach-Sandercock
    assert not c.is_dimensional and c.active.chord == pytest.approx(1.0, abs=1e-12)


def test_cascade_spec_is_frozen_and_kept_by_the_cascade():
    with pytest.raises(FrozenInstanceError):
        DEFAULT_SPEC.cl0 = 1.0
    c = build()
    assert c.spec == DEFAULT_SPEC
    # The inputs live on the spec only: no field of the cascade repeats one.
    assert {item.name for item in fields(g.Cascade)}.isdisjoint(item.name for item in fields(g.CascadeSpec))
    # The one degree-to-radian conversion.
    assert c.stagger == pytest.approx(math.radians(DEFAULT_SPEC.stagger_deg), rel=1e-15)


def test_make_cascade_takes_a_spec_and_nothing_else():
    with pytest.raises(TypeError):
        g.make_cascade("NACA65", solidity=1.25, stagger_deg=25.0, cl0=0.8, t_max_over_c=0.08)
    with pytest.raises(TypeError, match="CascadeSpec"):
        g.make_cascade({"blade_type": "NACA65", "solidity": 1.25})


def test_angle_names_carry_their_unit():
    """Degree inputs end in _deg; computed angles are radians and unsuffixed."""
    spec_angles = [item.name for item in fields(g.CascadeSpec) if "angle" in item.name or "stagger" in item.name]
    assert spec_angles == ["stagger_deg"]
    radian_fields = {"stagger", "camber_angle", "chi_le", "chi_te", "kappa_le", "kappa_te"}
    assert radian_fields <= {item.name for item in fields(g.Cascade)}
    c = build()
    assert set(c.angles_deg()) == {f"{name}_deg" for name in radian_fields}
    for name in radian_fields:
        assert c.angles_deg()[f"{name}_deg"] == pytest.approx(math.degrees(getattr(c, name)), rel=1e-15)


# --- source data ------------------------------------------------------------------------

def test_table_4_1_is_kept_as_printed_and_corrected_at_one_station():
    tables = g.base_tables()
    at_2_5 = np.isclose(tables.x, 0.025)
    assert len(tables.x) == 26 and tables.x[0] == 0.0 and tables.x[-1] == 1.0
    assert tables.thickness_printed[at_2_5] == pytest.approx(0.03480)
    assert tables.thickness_corrected[at_2_5] == pytest.approx(0.03148)
    np.testing.assert_array_equal(tables.thickness_corrected[~at_2_5], tables.thickness_printed[~at_2_5])
    assert dict(tables.corrections) == {0.025: 0.03148}
    assert tables.thickness_corrected.max() == pytest.approx(0.09992, abs=1e-15)
    assert tables.x[np.argmax(tables.thickness_corrected)] == pytest.approx(0.40)


def test_a10_camberline_is_table_4_1_scaled_by_cl0():
    tables = g.base_tables()
    # The printed table agrees with the closed form to 2.3e-5 c.
    assert np.max(np.abs(g.a10_camberline(tables.x, 1.0) - tables.camber)) < 2.5e-5
    for cl0 in (0.0, 0.45, 1.2, 1.6):
        np.testing.assert_allclose(g.a10_camberline(tables.x, cl0), cl0 * g.a10_camberline(tables.x, 1.0),
                                   rtol=1e-15, atol=0.0)


# the maximum of each reconstruction of the 0.10 table: PCHIP keeps the table's own (monotone
# between stations); the quintic peaks between 0.35 and 0.45 (the quadratic through the three
# largest ordinates: 0.099956 at 0.4093), measured 0.0999610502 at station 0.409773
RECONSTRUCTION_PEAK = {"pchip_sqrt": (0.09992, 0.40), "quintic_natural_sqrt": (0.0999610502, 0.409773)}


@pytest.mark.parametrize("reconstruction", g.THICKNESS_RECONSTRUCTIONS)
def test_nominal_thickness_0_10_is_table_4_1_unchanged(reconstruction):
    """t_max_over_c is the nominal NACA ratio: 0.10 is the corrected Table 4-1 itself at
    every station, for either reconstruction (round-off: measured <= 1.4e-17), not a table
    renormalized to peak at 0.10000. Between the stations: PCHIP keeps the table's own
    maximum 0.09992 at station 0.40; the quintic peaks at 0.0999610502 near 0.4098, which
    the table's own three largest ordinates put at 0.099956 near 0.409 (to 1e-5)."""
    tables = g.base_tables()
    assert g.thickness_scale(0.10) == 1.0
    np.testing.assert_allclose(g.naca65_base_thickness(tables.x, 0.10, thickness_reconstruction=reconstruction),
                               tables.thickness_corrected, rtol=0.0, atol=1e-15)
    station = np.linspace(0.0, 1.0, 400_001)
    law = g.naca65_base_thickness(station, 0.10, thickness_reconstruction=reconstruction)
    peak, at = RECONSTRUCTION_PEAK[reconstruction]
    assert law.max() == pytest.approx(peak, rel=1e-9)
    assert station[np.argmax(law)] == pytest.approx(at, abs=1e-5)
    k = int(np.argmax(tables.thickness_corrected))                           # station 0.40
    x, t = tables.x[k - 1:k + 2], tables.thickness_corrected[k - 1:k + 2]   # 0.35, 0.40, 0.45
    a, b, c = np.polyfit(x, t, 2)
    assert law.max() == pytest.approx(c - b * b / (4.0 * a), abs=1e-5 if reconstruction != LEGACY else 4e-5)
    with pytest.raises(ValueError, match="thickness_reconstruction"):
        g.naca65_base_thickness(station, 0.10, thickness_reconstruction="cubic")


@pytest.mark.parametrize("reconstruction", g.THICKNESS_RECONSTRUCTIONS)
@pytest.mark.parametrize("t", (0.03, 0.08, 0.12))
def test_nominal_thickness_scales_the_whole_distribution_by_t_over_0_10(t, reconstruction):
    station = np.linspace(0.0, 1.0, 2001)
    law = g.naca65_base_thickness(station, t, thickness_reconstruction=reconstruction)
    np.testing.assert_allclose(law, t / 0.10 * g.naca65_base_thickness(station, 0.10,
                                                                     thickness_reconstruction=reconstruction),
                               rtol=1e-14, atol=0.0)
    if reconstruction == LEGACY:
        assert law.max() == pytest.approx(NOMINAL_PEAK * t, rel=1e-13)


@pytest.mark.parametrize("variant", g.PROFILE_VARIANTS)
def test_the_quintic_thickness_is_c4_at_the_stations_and_the_pchip_c1(variant):
    """At every interior table station the one-sided derivatives in z = sqrt(station) of the
    half-thickness: the quintic's agree through the fourth (measured <= 3e-12 relative to the
    largest), the PCHIP's through the first only (its second derivative jumps: measured
    4.6 in z). The cascade metadata names the reconstruction and its maximum."""
    for name, smooth_orders in (("quintic_natural_sqrt", (0, 1, 2, 3, 4)), ("pchip_sqrt", (0, 1))):
        curve = g._HALF_THICKNESS[name]
        for order in range(5):
            d = curve.derivative(order) if order else curve
            left = np.array([np.polyval(d.c[:, i - 1], d.x[i] - d.x[i - 1]) for i in range(1, len(d.x) - 1)])
            right = d.c[-1, 1:]
            jump = float(np.max(np.abs(left - right)))
            if order in smooth_orders:
                assert jump <= 1e-11 * max(1.0, float(np.max(np.abs(right)))), (name, order, jump)
            elif order == 2:
                assert jump > 1.0, (name, jump)
        spec = spec_with(profile_variant=variant, surface_continuity="C2", thickness_reconstruction=name)
        c = g.make_cascade(spec)
        assert c.metadata["thickness_reconstruction"] == name
        peak, at = RECONSTRUCTION_PEAK[name]
        assert c.metadata["base_max_thickness"] == pytest.approx(peak, rel=1e-9)
        assert c.metadata["base_max_thickness_station"] == pytest.approx(at, abs=1e-6)
        assert c.metadata["max_thickness_over_c"] == pytest.approx(peak * spec.t_max_over_c / 0.10, rel=1e-9)
    assert build(profile_variant=variant).metadata["thickness_reconstruction"] == LEGACY
    assert build(profile_variant=variant, surface_continuity="C2").metadata["thickness_reconstruction"] == C2_THICKNESS


@pytest.mark.parametrize("value", ("pchip", "quintic", "cubic_spline", ""))
def test_unknown_thickness_reconstruction_is_rejected(value):
    with pytest.raises(ValueError, match="thickness_reconstruction"):
        g.make_cascade(spec_with(thickness_reconstruction=value))


# --- parameterization ---------------------------------------------------------------------

SWEEP = [  # cl0, t_max_over_c, solidity, stagger_deg
    (0.0, 0.06, 1.0, 0.0),
    (0.4, 0.04, 0.75, 10.0),
    (0.8, 0.08, 1.5, 30.0),
    (1.2, 0.10, 1.0, 40.0),
    (1.6, 0.10, 2.0, 50.0),
    (1.2, 0.05, 1.25, -15.0),
]


@pytest.mark.parametrize("variant", g.PROFILE_VARIANTS)
@pytest.mark.parametrize("cl0, t, solidity, stagger_deg", SWEEP)
def test_parameterized_cascades(cl0, t, solidity, stagger_deg, variant):
    c = build(cl0=cl0, t_max_over_c=t, solidity=solidity, stagger_deg=stagger_deg, profile_variant=variant)
    n, gamma = c.normalized, math.radians(stagger_deg)
    assert n.chord == pytest.approx(1.0, abs=1e-12)
    assert n.pitch == pytest.approx(1.0 / solidity, rel=1e-15)
    np.testing.assert_allclose(n.periodic_translation, [0.0, 1.0 / solidity], rtol=1e-15)
    np.testing.assert_allclose(n.leading_edge_point, [0.0, 0.0], atol=1e-15)
    np.testing.assert_allclose(n.trailing_edge_point, [math.cos(gamma), math.sin(gamma)], atol=1e-12)
    assert c.stagger == pytest.approx(gamma, rel=1e-15)
    assert c.metadata["max_thickness_over_c"] == pytest.approx(NOMINAL_PEAK * t, rel=1e-13)
    assert max_thickness(n) == pytest.approx(NOMINAL_PEAK * t, rel=5e-4)


def test_solidity_changes_pitch_and_throat_but_not_the_blade():
    loose, tight = build(solidity=0.8), build(solidity=1.6)
    for name in ("suction_surface", "pressure_surface", "camberline"):
        for suffix in ("", "_unstaggered"):
            np.testing.assert_array_equal(getattr(loose.normalized, name + suffix),
                                          getattr(tight.normalized, name + suffix))
    for c, sigma in ((loose, 0.8), (tight, 1.6)):
        assert c.normalized.pitch == pytest.approx(1.0 / sigma, rel=1e-15)
        np.testing.assert_allclose(c.normalized.periodic_translation, [0.0, 1.0 / sigma], rtol=1e-15)
    assert tight.normalized.throat_opening < loose.normalized.throat_opening
    assert np.max(np.abs(tight.normalized.throat_point_neighbor - loose.normalized.throat_point_neighbor)) > 1e-3


def test_stagger_turns_the_blade_without_reshaping_it():
    flat, turned = build(stagger_deg=0.0), build(stagger_deg=40.0)
    for name in ("suction_surface", "pressure_surface", "camberline", "equivalent_arc_camberline"):
        np.testing.assert_array_equal(getattr(flat.normalized, f"{name}_unstaggered"),
                                      getattr(turned.normalized, f"{name}_unstaggered"))
        np.testing.assert_array_equal(getattr(flat.normalized, name), getattr(flat.normalized, f"{name}_unstaggered"))
        np.testing.assert_allclose(getattr(turned.normalized, name),
                                   rotated(getattr(flat.normalized, name), math.radians(40.0)), atol=1e-15)
    chord = turned.normalized.trailing_edge_point - turned.normalized.leading_edge_point
    assert math.atan2(chord[1], chord[0]) == pytest.approx(math.radians(40.0), abs=1e-15)
    assert turned.kappa_le - turned.stagger == pytest.approx(flat.kappa_le - flat.stagger, abs=1e-15)


def test_cl0_changes_camber_but_not_thickness():
    low, high = build(cl0=0.4), build(cl0=1.2)
    np.testing.assert_array_equal(low.normalized.camberline_unstaggered[:, 0], high.normalized.camberline_unstaggered[:, 0])
    np.testing.assert_allclose(high.normalized.camberline_unstaggered[:, 1],
                               3.0 * low.normalized.camberline_unstaggered[:, 1], rtol=1e-13, atol=1e-17)
    for c in (low, high):
        assert max_thickness(c.normalized) == pytest.approx(NOMINAL_PEAK * DEFAULT_SPEC.t_max_over_c, rel=5e-4)
    assert high.normalized.leading_edge_radius == low.normalized.leading_edge_radius
    assert high.camber_angle > low.camber_angle


def test_thickness_changes_thickness_but_not_the_camberline():
    thin, thick = build(t_max_over_c=0.05), build(t_max_over_c=0.10)
    np.testing.assert_array_equal(thin.normalized.camberline_unstaggered, thick.normalized.camberline_unstaggered)
    assert (thin.camber_angle, thin.kappa_le, thin.kappa_te) == (thick.camber_angle, thick.kappa_le, thick.kappa_te)
    assert max_thickness(thin.normalized) == pytest.approx(NOMINAL_PEAK * 0.05, rel=5e-4)
    assert max_thickness(thick.normalized) == pytest.approx(NOMINAL_PEAK * 0.10, rel=5e-4)
    # The nose is parabolic, so the LE radius goes with the square of the thickness.
    assert thick.normalized.leading_edge_radius == pytest.approx(4.0 * thin.normalized.leading_edge_radius, rel=1e-14)


@pytest.mark.parametrize("cl0", (0.4, 1.2, 1.6))
def test_surfaces_lie_half_a_thickness_from_the_camberline_along_its_normal(cl0):
    """A point built as X_c + (t/2) n_c has X_c as the foot of its perpendicular to the
    camberline, so its distance to the camberline is half the thickness at the arc station
    of X_c. A thickness laid off vertically, or read at x/c instead, would miss."""
    t = 0.08
    n = build(cl0=cl0, t_max_over_c=t, profile_variant="emery_base").normalized
    miss_arc, miss_chord = [], []
    for surface in (n.suction_surface_unstaggered, n.pressure_surface_unstaggered):
        for point in surface[(surface[:, 0] > 0.05) & (surface[:, 0] < 0.9)]:
            def distance(x: float, point=point) -> float:
                return math.hypot(point[0] - x, point[1] - float(g.a10_camberline(x, cl0)))
            foot = minimize_scalar(distance, bounds=(max(point[0] - 0.1, 1e-9), min(point[0] + 0.1, 1.0 - 1e-9)),
                                   method="bounded", options={"xatol": 1e-12}).x
            miss_arc.append(abs(distance(foot) - 0.5 * float(g.naca65_base_thickness(
                g.camberline_station(foot, cl0), t, thickness_reconstruction=LEGACY))))
            miss_chord.append(abs(distance(foot) - 0.5 * float(g.naca65_base_thickness(
                foot, t, thickness_reconstruction=LEGACY))))
    assert max(miss_arc) < 1e-9
    assert max(miss_chord) > 1e-5      # the test tells the two station readings apart


# --- blade angles -------------------------------------------------------------------------

@pytest.mark.parametrize("cl0", (0.0, 0.5, 1.2, 1.6))
@pytest.mark.parametrize("stagger_deg", (-10.0, 20.0, 45.0))
def test_equivalent_circular_arc_angles(cl0, stagger_deg):
    c = build(cl0=cl0, stagger_deg=stagger_deg)
    camber = 4.0 * math.atan(0.1103 * cl0)        # Aungier 2003 Eq. 4-6
    gamma = math.radians(stagger_deg)
    assert c.camber_angle == pytest.approx(camber, abs=1e-15)
    assert c.chi_le == c.chi_te == pytest.approx(0.5 * camber, abs=1e-15)
    assert c.kappa_le == pytest.approx(gamma + 0.5 * camber, abs=1e-15)
    assert c.kappa_te == pytest.approx(gamma - 0.5 * camber, abs=1e-15)
    assert c.kappa_le - c.kappa_te == pytest.approx(c.camber_angle, abs=1e-15)
    assert c.angles_deg()["stagger_deg"] == pytest.approx(stagger_deg, rel=1e-15)


def test_equivalent_arc_is_kept_apart_from_the_actual_camberline():
    c = build(cl0=1.2)
    n = c.normalized
    arc, actual = n.equivalent_arc_camberline_unstaggered, n.camberline_unstaggered
    np.testing.assert_allclose(arc[[0, -1]], [[0.0, 0.0], [1.0, 0.0]], atol=1e-15)
    x = np.array([1e-7, 1.0 - 1e-7])
    y = g.circular_arc_camberline(x, c.camber_angle)
    assert y[0] / x[0] == pytest.approx(math.tan(c.chi_le), rel=1e-6)
    assert y[1] / (1.0 - x[1]) == pytest.approx(math.tan(c.chi_te), rel=1e-6)
    # The surfaces belong to the actual A10 line, which is not the arc.
    np.testing.assert_array_equal(actual[:, 1], g.a10_camberline(actual[:, 0], 1.2))
    assert np.max(np.abs(actual[:, 1] - arc[:, 1])) > 1e-3


# --- profile variants ---------------------------------------------------------------------

@pytest.mark.parametrize("cl0, t", ((0.4, 0.06), (1.2, 0.10), (1.6, 0.08)))
def test_kovach_keeps_the_forward_blade_and_rounds_the_trailing_edge(cl0, t):
    kovach_cascade = build(cl0=cl0, t_max_over_c=t, profile_variant="kovach_sandercock")
    kovach = kovach_cascade.normalized
    base = build(cl0=cl0, t_max_over_c=t, profile_variant="emery_base").normalized

    # Ahead of station 0.60 both variants are the same blade, point for point.
    for name in ("suction_surface_unstaggered", "pressure_surface_unstaggered"):
        k, b = getattr(kovach, name), getattr(base, name)
        forward = int(np.argmax(b[:, 0] > 0.5))
        np.testing.assert_array_equal(k[:forward], b[:forward])

    # Base: sharp TE. Kovach: a circle of radius 0.008 c centered on the camberline,
    # through the TE point, with both surfaces ending on it.
    assert base.trailing_edge_radius == 0.0
    np.testing.assert_array_equal(base.trailing_edge_center, base.trailing_edge_point)
    assert kovach.trailing_edge_radius == pytest.approx(0.008, abs=1e-15)
    center = rotated(kovach.trailing_edge_center, -kovach_cascade.stagger)
    assert center[1] == pytest.approx(float(g.a10_camberline(center[0], cl0)), abs=1e-12)
    assert math.dist(center, (1.0, 0.0)) == pytest.approx(0.008, abs=1e-10)
    for surface in (kovach.suction_surface_unstaggered, kovach.pressure_surface_unstaggered):
        np.testing.assert_allclose(np.hypot(*(surface[-10:] - center).T), 0.008, atol=1e-10)

    # Aft of station 0.60 the Kovach blade is thicker; the base closes to the sharp TE.
    def thickness_at(c: g.CascadeCoordinates, x: float) -> float:
        def y(surface: np.ndarray) -> float:
            aft = surface[(surface[:, 0] > 0.3) & (surface[:, 0] < 0.985)]
            return float(np.interp(x, aft[:, 0], aft[:, 1]))
        return y(c.suction_surface_unstaggered) - y(c.pressure_surface_unstaggered)

    for x in (0.7, 0.8, 0.9, 0.97):
        assert thickness_at(kovach, x) > thickness_at(base, x)
    assert thickness_at(kovach, 0.97) > 0.012 > thickness_at(base, 0.97)


# --- throat -------------------------------------------------------------------------------

@pytest.mark.parametrize("solidity, stagger_deg, cl0, t", ((0.6, 50.0, 1.6, 0.05), (1.0, 30.0, 0.8, 0.05),
                                                           (1.5, 0.0, 1.2, 0.10), (2.0, -15.0, 0.4, 0.08)))
def test_geometric_throat_is_the_shortest_gap_between_neighbors(solidity, stagger_deg, cl0, t):
    c = build(solidity=solidity, stagger_deg=stagger_deg, cl0=cl0, t_max_over_c=t)
    n = c.normalized
    assert math.isfinite(n.throat_opening) and 0.0 < n.throat_opening < n.pitch
    assert math.dist(n.throat_point_reference, n.throat_point_neighbor) == pytest.approx(n.throat_opening, abs=1e-14)
    assert c.throat_over_pitch == pytest.approx(n.throat_opening / n.pitch, rel=1e-15)
    assert c.throat_over_chord == pytest.approx(n.throat_opening, rel=1e-15)

    # Referee: the same blade built eight times finer, so its polyline lies on the true
    # surface to about 1e-6 c. Both throat ends lie on it to the accuracy of the spline
    # through the default 101 stations (about 2e-6 c: the PCHIP thickness makes the true
    # surface only C1 at the table knots), and nothing is closer.
    fine = g.naca65_profile(cl0, t, meanline_type=DEFAULT_SPEC.meanline_type,
                            profile_variant=DEFAULT_SPEC.profile_variant, n_camber_stations=801,
                            thickness_reconstruction=LEGACY)
    reference = rotated(fine.suction_surface, c.stagger)
    neighbor = rotated(fine.pressure_surface, c.stagger) + n.periodic_translation
    assert polyline_distance(n.throat_point_reference[None], reference)[0] < 5e-6
    assert polyline_distance(n.throat_point_neighbor[None], neighbor)[0] < 5e-6
    referee = min(polyline_distance(reference, neighbor).min(), polyline_distance(neighbor, reference).min())
    assert n.throat_opening == pytest.approx(referee, abs=5e-6)


def test_throat_is_given_by_its_two_end_points_only():
    unit, metric = build(), build(physical_chord=0.05)
    assert "throat_location" not in {item.name for item in fields(g.Cascade)}
    assert not hasattr(unit, "throat_location")
    for c, scale in ((unit, 1.0), (metric, 0.05)):
        a = c.active
        assert a.throat_point_reference.shape == a.throat_point_neighbor.shape == (2,)
        assert math.dist(a.throat_point_reference, a.throat_point_neighbor) == pytest.approx(a.throat_opening, rel=1e-14)
        for name in ("throat_point_reference", "throat_point_neighbor", "throat_opening"):
            np.testing.assert_allclose(getattr(a, name), scale * np.asarray(getattr(unit.normalized, name)),
                                       rtol=1e-15, atol=0.0)
        assert c.throat_over_chord == pytest.approx(a.throat_opening / a.chord, rel=1e-12)
        assert c.throat_over_pitch == pytest.approx(a.throat_opening / a.pitch, rel=1e-12)


@pytest.mark.parametrize("solidity, stagger_deg, cl0, t", ((1.5, 40.0, 0.8, 0.06), (0.9, 20.0, 1.4, 0.09)))
def test_throat_correlation_is_a_separate_diagnostic(solidity, stagger_deg, cl0, t):
    c = build(solidity=solidity, stagger_deg=stagger_deg, cl0=cl0, t_max_over_c=t)
    phi = stagger_deg * (1.0 - 0.05 * cl0**1.5) + 5.0 * cl0**1.5 - 2.0                   # Eq. 4-30, degrees
    expected = ((1.0 - t * math.sqrt(solidity)) * math.cos(math.radians(phi))) ** math.sqrt(solidity)   # Eq. 4-31
    assert c.throat_over_pitch_correlation == pytest.approx(expected, rel=1e-14)
    assert c.throat_over_pitch == pytest.approx(c.normalized.throat_opening / c.normalized.pitch, rel=1e-15)
    assert abs(c.throat_over_pitch - c.throat_over_pitch_correlation) > 1e-6


# --- physical chord -----------------------------------------------------------------------

def test_physical_chord_scales_every_length_of_the_same_blade():
    unit, metric = build(), build(physical_chord=0.05)
    assert not unit.is_dimensional and metric.is_dimensional
    assert metric.spec == replace(unit.spec, physical_chord=0.05)
    for item in fields(g.CascadeCoordinates):
        np.testing.assert_array_equal(getattr(metric.normalized, item.name), getattr(unit.normalized, item.name))
        np.testing.assert_allclose(getattr(metric.active, item.name),
                                   0.05 * np.asarray(getattr(unit.normalized, item.name)), rtol=1e-15, atol=0.0)
    assert metric.active.chord == pytest.approx(0.05, rel=1e-12)
    assert metric.active.pitch == pytest.approx(0.05 / DEFAULT_SPEC.solidity, rel=1e-15)
    assert metric.active.trailing_edge_radius == pytest.approx(0.008 * 0.05, rel=1e-15)
    for name in ("throat_over_pitch", "throat_over_chord", "throat_over_pitch_correlation", "stagger",
                 "camber_angle", "kappa_le", "kappa_te"):
        assert getattr(metric, name) == getattr(unit, name)
    assert (unit.metadata["length_unit"], metric.metadata["length_unit"]) == ("chord", "m")


def test_without_physical_chord_the_active_geometry_is_the_normalized_one():
    c = build()
    for item in fields(g.CascadeCoordinates):
        np.testing.assert_array_equal(getattr(c.active, item.name), getattr(c.normalized, item.name))
    assert c.spec.physical_chord is None and c.active.chord == pytest.approx(1.0, abs=1e-12)


# --- output contract for grid.py ----------------------------------------------------------

@pytest.mark.parametrize("variant", g.PROFILE_VARIANTS)
def test_output_contract_for_grid(variant):
    c = build(profile_variant=variant, physical_chord=0.08)
    a, n = c.active, c.normalized
    for surface in (a.suction_surface, a.pressure_surface):
        assert surface.ndim == 2 and surface.shape[1] == 2
        np.testing.assert_array_equal(surface[0], a.leading_edge_point)
        np.testing.assert_array_equal(surface[-1], a.trailing_edge_point)
        assert np.min(np.hypot(*np.diff(surface, axis=0).T)) > 0.0
        with pytest.raises(ValueError):
            surface[0, 0] = 1.0
    # The suction surface is the +y side of the blade frame and faces the neighbor blade.
    assert shoelace(np.vstack((n.suction_surface_unstaggered[::-1], n.pressure_surface_unstaggered[1:]))) > 0.0
    neighbor = np.vstack((a.suction_surface[::-1], a.pressure_surface[1:])) + a.periodic_translation
    mid = lambda surface: surface[np.argmin(np.abs(surface[:, 0] - 0.5 * a.trailing_edge_point[0]))][None]
    assert polyline_distance(mid(a.suction_surface), neighbor)[0] < polyline_distance(mid(a.pressure_surface), neighbor)[0]


# --- inputs and invalid geometry ----------------------------------------------------------

@pytest.mark.parametrize("changes, message", [
    ({"solidity": 0.0}, "solidity"), ({"solidity": -1.0}, "solidity"), ({"solidity": math.nan}, "solidity"),
    ({"t_max_over_c": 0.0}, "t_max_over_c"), ({"t_max_over_c": -0.05}, "t_max_over_c"),
    ({"cl0": -0.2}, "cl0"),
    ({"stagger_deg": 90.0}, "stagger_deg"), ({"stagger_deg": -95.0}, "stagger_deg"),
    ({"physical_chord": 0.0}, "physical_chord"), ({"physical_chord": -0.1}, "physical_chord"),
    ({"profile_variant": "kovach"}, "profile_variant"),
    ({"n_camber_stations": 5}, "n_camber_stations"),
])
def test_invalid_inputs_are_rejected(changes, message):
    with pytest.raises(ValueError, match=message):
        g.make_cascade(spec_with(**changes))


def test_unknown_blade_type_is_rejected():
    with pytest.raises(ValueError, match="blade_type"):
        g.make_cascade(spec_with(blade_type="NACA64"))


@pytest.mark.parametrize("blade_type", ("DCA", "C4"))
def test_dca_and_c4_are_in_the_api_but_not_implemented(blade_type):
    assert blade_type in g.BLADE_TYPES
    with pytest.raises(NotImplementedError, match=blade_type):
        g.make_cascade(spec_with(blade_type=blade_type))


def test_other_meanlines_are_not_implemented():
    with pytest.raises(NotImplementedError, match="A4K6"):
        g.make_cascade(spec_with(meanline_type="A4K6"))


def test_overlapping_blades_are_rejected():
    # Unstaggered blades stacked at a pitch smaller than their thickness must intersect.
    with pytest.raises(g.GeometryError, match="overlap"):
        g.make_cascade(spec_with(solidity=20.0, stagger_deg=0.0))


def test_validate_cascade_reports_swapped_surfaces():
    c = build()

    def swap(coordinates: g.CascadeCoordinates) -> g.CascadeCoordinates:
        return replace(coordinates,
                       suction_surface=coordinates.pressure_surface,
                       pressure_surface=coordinates.suction_surface,
                       suction_surface_unstaggered=coordinates.pressure_surface_unstaggered,
                       pressure_surface_unstaggered=coordinates.suction_surface_unstaggered)

    with pytest.raises(g.GeometryError, match=r"suction surface is not on the \+y side"):
        g.validate_cascade(replace(c, normalized=swap(c.normalized), active=swap(c.active)))


def test_validate_cascade_reports_inconsistent_angles_and_scaling():
    c = build(physical_chord=0.05)
    with pytest.raises(g.GeometryError, match="kappa"):
        g.validate_cascade(replace(c, kappa_le=c.kappa_le + 1e-6))
    with pytest.raises(g.GeometryError, match="active"):
        g.validate_cascade(replace(c, spec=replace(c.spec, physical_chord=0.06)))
    with pytest.raises(g.GeometryError, match="stagger"):
        g.validate_cascade(replace(c, spec=replace(c.spec, stagger_deg=c.spec.stagger_deg + 1.0)))


def test_outside_the_checked_range_warns_but_still_builds():
    with pytest.warns(g.GeometryWarning, match="outside"):
        c = g.make_cascade(spec_with(t_max_over_c=0.11))
    assert c.metadata["inside_checked_range"] is False


# --- surface continuity: C1 stays the validated baseline ------------------------------------

C1_FINGERPRINT = {      # DEFAULT_SPEC, taken from geometry.py before the C2 representation existed
    "kovach_sandercock": {
        "points": (208, 216),
        "sums": (95.51623194806086, 5.45654721970309, 95.61051486394044, -0.10768780395491823),
        "suction_vertex_1": (-2.9638002174164728e-05, 9.122775088683246e-05),
        "pressure_vertex_40": (0.002981156427180013, -0.0028803410559074885),
        "throat": 0.6375471262532315, "throat_point_reference": (0.33527155118444246, 0.24794559574877603),
        "area": 0.05589339484755166, "throat_over_pitch": 0.7969339078165394},
    "emery_base": {
        "points": (164, 172),
        "sums": (51.532844827633035, 4.948028676713854, 51.85430338817447, 0.28979156666927636),
        "suction_vertex_1": (-2.9638002174164728e-05, 9.122775088683246e-05),
        "pressure_vertex_40": (0.002981156427180013, -0.0028803410559074885),
        "throat": 0.6375471262532406, "throat_point_reference": (0.3352715512152051, 0.24794559576387984),
        "area": 0.05239652453066268, "throat_over_pitch": 0.7969339078165507},
}


@pytest.mark.parametrize("variant", g.PROFILE_VARIANTS)
def test_c1_baseline_fingerprint(variant):
    c = build(profile_variant=variant)
    n, expected = c.normalized, C1_FINGERPRINT[variant]
    ss, ps = n.suction_surface_unstaggered, n.pressure_surface_unstaggered
    assert (len(ss), len(ps)) == expected["points"]
    measured = {
        "sums": (ss[:, 0].sum(), ss[:, 1].sum(), ps[:, 0].sum(), ps[:, 1].sum()),
        "suction_vertex_1": ss[1], "pressure_vertex_40": ps[40], "throat": n.throat_opening,
        "throat_point_reference": n.throat_point_reference,
        "area": shoelace(np.vstack((ss[::-1], ps[1:]))), "throat_over_pitch": c.throat_over_pitch,
    }
    for key, value in measured.items():
        np.testing.assert_allclose(value, expected[key], rtol=1e-13, atol=1e-16, err_msg=key)


def test_surface_continuity_defaults_to_c2_which_carries_the_curves():
    """The default since 2026-11-14 is the frozen C2 geometry; C1 is asked for by name and carries no curves."""
    spec = g.CascadeSpec(blade_type="NACA65", solidity=1.4, stagger_deg=20.0, cl0=0.9, t_max_over_c=0.07)
    assert spec.surface_continuity == "C2" and g.SURFACE_CONTINUITIES == ("C1", "C2")
    c = g.make_cascade(spec)
    assert c.normalized_curves is not None and c.active_curves is not None
    assert c.metadata["thickness_reconstruction"] == "quintic_natural_sqrt"
    assert c.metadata["le_model"] == "monotone_curvature_fair"
    legacy = g.make_cascade(replace(spec, surface_continuity="C1"))
    assert legacy.normalized_curves is None and legacy.active_curves is None
    assert legacy.metadata["thickness_reconstruction"] == "pchip_sqrt"
    assert not any(key.startswith("c2") or key == "surface_representation" for key in legacy.metadata)


@pytest.mark.parametrize("variant", g.PROFILE_VARIANTS)
def test_explicit_c2_with_the_quintic_is_the_default_bit_for_bit(variant):
    spec = g.CascadeSpec(blade_type="NACA65", solidity=1.25, stagger_deg=25.0, cl0=0.8, t_max_over_c=0.08,
                         profile_variant=variant, physical_chord=0.07)
    default = g.make_cascade(spec)
    explicit = g.make_cascade(replace(spec, surface_continuity="C2", thickness_reconstruction="quintic_natural_sqrt"))
    for label in ("normalized", "active"):
        for item in fields(g.CascadeCoordinates):
            np.testing.assert_array_equal(getattr(getattr(explicit, label), item.name),
                                          getattr(getattr(default, label), item.name))
    for label in ("normalized_curves", "active_curves"):
        for name in ("suction", "pressure", "suction_unstaggered", "pressure_unstaggered"):
            ours, theirs = getattr(getattr(explicit, label), name), getattr(getattr(default, label), name)
            np.testing.assert_array_equal(ours.control_points, theirs.control_points)
            np.testing.assert_array_equal(ours.knots, theirs.knots)
    for name in ("stagger", "camber_angle", "chi_le", "chi_te", "kappa_le", "kappa_te", "throat_over_pitch",
                 "throat_over_chord", "throat_over_pitch_correlation"):
        assert getattr(explicit, name) == getattr(default, name)
    assert dict(explicit.metadata) == dict(default.metadata)


# --- surface continuity: C2 is the frozen design geometry (2026-11-14) ------------------------

C2_FINGERPRINT_SPECS = {     # the two variants of DEFAULT_SPEC, and the solver's validation blade
    "kovach_sandercock": replace(DEFAULT_SPEC, surface_continuity="C2", profile_variant="kovach_sandercock"),
    "emery_base": replace(DEFAULT_SPEC, surface_continuity="C2", profile_variant="emery_base"),
    "65-(12A10)10": g.CascadeSpec(blade_type="NACA65", solidity=1.0, stagger_deg=33.0, cl0=1.2, t_max_over_c=0.10),
}
C2_FINGERPRINT_U = (0.002, 0.05, 0.5)      # in the LE fair, behind the LE anchor, mid-chord
C2_FINGERPRINT = {      # taken from geometry.py on 2026-11-14, when the C2 geometry was frozen
    "kovach_sandercock": {
        "points": (208, 213),
        "control_points": (414, 447),
        "sums": (95.51925319753433, 5.454755393791725, 95.60263315329917, -0.1027031018718374),
        "control_sums": (113.16532177520041, 9.352622165762892, 110.94361364925783, -0.5100421030705986),
        "suction_at_u": ((-0.00018572399088258244, 0.0020346856699687546), (0.04052271018280367, 0.027666934938312696),
                         (0.49884715805263175, 0.08266231025910009)),
        "pressure_at_u": ((0.001047329920794516, -0.0016951019011210205), (0.04868700009715207, -0.0047758580050219),
                          (0.500801856945998, 0.005659301068102172)),
        "suction_curvature_at_u": (-196.43609524495673, -3.051058914827888, -0.6811995844468087),
        "pressure_curvature_at_u": (214.76556607730706, 0.46205280398001847, 0.1869199452931848),
        "le_anchor_u": (0.010365462277339265, 0.011216539080322127),
        "le_departure_u": (0.00154993052963812, 0.0016062413984590787),
        "le_fairness": (0.9625517174339578, 0.7353443543369287),
        "throat": 0.6375033027822187, "throat_over_pitch": 0.7968791284777733, "area": 0.055893418325073885},
    "emery_base": {
        "points": (164, 169),
        "control_points": (378, 413),
        "sums": (51.535762628711005, 4.945884497557422, 51.84652483447332, 0.29512755896580634),
        "control_sums": (84.28482896219545, 7.455014927523157, 84.66256542296111, -0.3005382692672832),
        "suction_at_u": ((-0.00018738344025728953, 0.002022555533755243), (0.04023679250255069, 0.02755768622600673),
                         (0.4957879988450233, 0.0827619074968153)),
        "pressure_at_u": ((0.001044840693386648, -0.0016925822224674464), (0.04859846425187905, -0.004777290298147942),
                          (0.4999169287826601, 0.0056282079384378)),
        "suction_curvature_at_u": (-197.31952753401478, -3.0630096889375786, -0.6826849441164754),
        "pressure_curvature_at_u": (214.87922593796807, 0.46372869750257956, 0.1878086667111122),
        "le_anchor_u": (0.010427381314652312, 0.011236339600498992),
        "le_departure_u": (0.0015591891814695103, 0.0016090768912070067),
        "le_fairness": (0.9625517174339578, 0.7353443543369287),
        "throat": 0.6375033025663435, "throat_over_pitch": 0.7968791282079293, "area": 0.05239448501711963},
    "65-(12A10)10": {
        "points": (208, 230),
        "control_points": (394, 437),
        "sums": (95.45551883197653, 7.3470114429032245, 95.80280606574028, 0.5907469290284107),
        "control_sums": (107.94615711956595, 11.8442327377534, 105.61110556403362, 0.33357990944533433),
        "suction_at_u": ((-0.0006491609356879365, 0.001993208516498273), (0.03432961544801684, 0.03542338496042315),
                         (0.4967283057330068, 0.11444423371043105)),
        "pressure_at_u": ((0.0011561246235971539, -0.001638461551000991), (0.048685757613781314, -0.0030638931673078605),
                          (0.5009695913098412, 0.018113295166576886)),
        "suction_curvature_at_u": (-145.5603619034723, -3.9666899224897194, -0.8951885434022345),
        "pressure_curvature_at_u": (145.56041098681587, 0.3006479822358992, 0.1659133215657833),
        "le_anchor_u": (0.013412143833950351, 0.017537823972696848),
        "le_departure_u": (0.0023686008612041403, 0.0025048554669506113),
        "le_fairness": (1.0441614367474414, 0.7093012656924348),
        "throat": 0.7215213941264458, "throat_over_pitch": 0.7215213941264458, "area": 0.06960680691787813},
}
# Measured 2026-11-14: the build is reproducible run to run on one machine. A one-ulp change of
# t_max_over_c, cl0, stagger_deg or solidity moves these numbers by at most 4.2e-11 relative
# (the curvatures; positions and sums 4e-12), so another platform's last-bit differences
# should stay below that. 1e-9 covers the measured 4.2e-11; any change to the construction
# moves them by far more.
C2_FINGERPRINT_RTOL = 1e-9


@pytest.mark.parametrize("case", C2_FINGERPRINT)
def test_c2_frozen_fingerprint(case):
    """The frozen C2 geometry (robust LE, natural quintic thickness), the default CascadeSpec's."""
    c = _cached(C2_FINGERPRINT_SPECS[case])
    n, curves, md, expected = c.normalized, c.normalized_curves, c.metadata, C2_FINGERPRINT[case]
    assert md["thickness_reconstruction"] == "quintic_natural_sqrt" and md["le_model"] == "monotone_curvature_fair"
    ss, ps = n.suction_surface_unstaggered, n.pressure_surface_unstaggered
    suction, pressure = curves.suction_unstaggered, curves.pressure_unstaggered
    assert (len(ss), len(ps)) == expected["points"]
    assert tuple(md["c2_control_points"]) == expected["control_points"]
    assert (len(suction.control_points), len(pressure.control_points)) == expected["control_points"]
    u = np.array(C2_FINGERPRINT_U)
    measured = {
        "sums": (ss[:, 0].sum(), ss[:, 1].sum(), ps[:, 0].sum(), ps[:, 1].sum()),
        "control_sums": (*suction.control_points.sum(axis=0), *pressure.control_points.sum(axis=0)),
        "suction_at_u": suction.evaluate(u), "pressure_at_u": pressure.evaluate(u),
        "suction_curvature_at_u": suction.curvature(u), "pressure_curvature_at_u": pressure.curvature(u),
        "le_anchor_u": (md["ss_le_anchor_u"], md["ps_le_anchor_u"]),
        "le_departure_u": (md["ss_le_departure_u"], md["ps_le_departure_u"]),
        "le_fairness": (md["ss_le_fairness"], md["ps_le_fairness"]),
        "throat": n.throat_opening, "throat_over_pitch": c.throat_over_pitch,
        "area": shoelace(np.vstack((ss[::-1], ps[1:]))),
    }
    for key, value in measured.items():
        np.testing.assert_allclose(value, expected[key], rtol=C2_FINGERPRINT_RTOL, atol=1e-15, err_msg=key)


@pytest.mark.parametrize("variant", g.PROFILE_VARIANTS)
def test_c1_surfaces_are_the_legacy_builder_output_bit_for_bit(variant):
    c = build(profile_variant=variant)
    profile = g.naca65_profile(DEFAULT_SPEC.cl0, DEFAULT_SPEC.t_max_over_c, meanline_type="A10",
                               profile_variant=variant, n_camber_stations=101, thickness_reconstruction=LEGACY)
    np.testing.assert_array_equal(c.normalized.suction_surface_unstaggered, profile.suction_surface)
    np.testing.assert_array_equal(c.normalized.pressure_surface_unstaggered, profile.pressure_surface)


# --- surface continuity: the one switch ------------------------------------------------------

def test_c2_is_accepted_and_carries_its_curves():
    c = build(surface_continuity="C2")
    assert c.spec.surface_continuity == "C2"
    for curves in (c.normalized_curves, c.active_curves):
        assert isinstance(curves, g.SurfaceCurves)
        for name in ("suction", "pressure", "suction_unstaggered", "pressure_unstaggered"):
            curve = getattr(curves, name)
            assert isinstance(curve, g.ParametricBladeCurve) and curve.degree == 5 and curve.domain == (0.0, 1.0)
    assert c.metadata["c2_max_deviation_over_c"] <= 1.01e-6          # downstream of the LE anchors
    assert len(c.metadata["c2_control_points"]) == 2
    assert c.metadata["le_model"] == "monotone_curvature_fair"
    assert c.metadata["le_radius_source"] == "legacy_naca65_reference"
    for side in ("ss", "ps"):
        for name in ("departure_u", "anchor_u", "max_deviation_over_c", "rms_deviation_over_c", "fairness"):
            assert np.isfinite(c.metadata[f"{side}_le_{name}"])
        assert 0.0 < c.metadata[f"{side}_le_departure_u"] < c.metadata[f"{side}_le_anchor_u"] < 0.1


@pytest.mark.parametrize("value", ("C3", "c2", "C0", "", "G2", "quintic"))
def test_unknown_surface_continuity_is_rejected(value):
    with pytest.raises(ValueError, match="surface_continuity"):
        g.make_cascade(spec_with(surface_continuity=value))


def test_surface_continuity_is_the_one_switch():
    names = [item.name for item in fields(g.CascadeSpec)]
    assert [name for name in names if "continuity" in name] == ["surface_continuity"]
    assert not [name for name in names if any(word in name for word in ("spline", "smooth", "quintic", "c2"))]


# --- the C1 construction as an analytic curve ------------------------------------------------

@pytest.mark.parametrize("reconstruction", g.THICKNESS_RECONSTRUCTIONS)
@pytest.mark.parametrize("variant", g.PROFILE_VARIANTS)
@pytest.mark.parametrize("cl0, t, n_stations", ((0.0, 0.06, 101), (0.8, 0.08, 101), (1.6, 0.04, 101),
                                                (1.2, 0.10, 401)))
def test_c1_vertices_lie_on_the_analytic_reference(variant, cl0, t, n_stations, reconstruction):
    profile = g.naca65_profile(cl0, t, meanline_type="A10", profile_variant=variant, n_camber_stations=n_stations,
                               thickness_reconstruction=reconstruction)
    references = g.naca65_reference_surfaces(cl0, t, profile_variant=variant, thickness_reconstruction=reconstruction)
    for reference, polyline in zip(references, (profile.suction_surface, profile.pressure_surface)):
        s, distance = reference.locate(polyline)
        assert np.max(distance) < 1e-12
        assert np.all(np.diff(s) > 0.0) and s[0] == pytest.approx(0.0, abs=1e-15)
        assert s[-1] == pytest.approx(reference.length, abs=1e-12)
        np.testing.assert_array_equal(reference.leading_edge_point, polyline[0])
        np.testing.assert_array_equal(reference.trailing_edge_point, polyline[-1])
    assert references[0].sharp_trailing_edge == references[1].sharp_trailing_edge == (variant == "emery_base")


@pytest.mark.parametrize("reconstruction", g.THICKNESS_RECONSTRUCTIONS)
@pytest.mark.parametrize("variant", g.PROFILE_VARIANTS)
def test_reference_derivatives_match_central_differences(variant, reconstruction):
    """Every piece's analytic d/dtau and d2/dtau2 against central differences of its own
    points; the tolerances are the truncation plus the round-off eps |X| / h^k of each. For
    either reconstruction: the PCHIP's slopes have their own closed form, the quintic's come
    from its interval polynomial."""
    eps = np.finfo(float).eps
    for reference in g.naca65_reference_surfaces(1.2, 0.08, profile_variant=variant,
                                                 thickness_reconstruction=reconstruction):
        for name, evaluate in zip(reference.piece_names, reference.pieces):
            tau = np.linspace(0.05, 0.95, 19)
            points, d1, d2 = evaluate(tau)
            scale1, scale2 = np.max(np.hypot(*d1.T)), np.max(np.hypot(*d2.T))
            h = 1e-6
            first = (evaluate(tau + h)[0] - evaluate(tau - h)[0]) / (2 * h)
            np.testing.assert_allclose(first, d1, rtol=0.0, atol=1e-7 * scale1 + 10 * eps / h, err_msg=name)
            h = 1e-4
            second = (evaluate(tau + h)[0] - 2 * points + evaluate(tau - h)[0]) / h**2
            np.testing.assert_allclose(second, d2, rtol=0.0, atol=1e-5 * max(scale2, scale1) + 10 * eps / h**2,
                                       err_msg=name)


@pytest.mark.parametrize("reconstruction", g.THICKNESS_RECONSTRUCTIONS)
@pytest.mark.parametrize("variant", g.PROFILE_VARIANTS)
def test_reference_circles_and_junctions_are_the_c1_construction(variant, reconstruction):
    t = 0.10
    suction, pressure = g.naca65_reference_surfaces(1.2, t, profile_variant=variant,
                                                    thickness_reconstruction=reconstruction)
    le_radius = 0.00687 * (t / 0.10) ** 2
    for reference, sign in ((suction, -1.0), (pressure, 1.0)):
        sample = reference.sample(41)
        np.testing.assert_allclose(sample.kappa[sample.piece == 0], sign / le_radius, rtol=1e-9)  # suction: negative
        junctions = {j.label: j for j in reference.junctions}
        circle_fair = junctions["LE circle / LE fair"]          # the quintic fair matches the circle to 2nd order
        assert abs(circle_fair.kappa_jump) < 1e-6 and abs(circle_fair.turn) < 1e-12
        if variant == "kovach_sandercock":
            np.testing.assert_allclose(sample.kappa[sample.piece == len(reference.pieces) - 1], sign / 0.008, rtol=1e-9)
            start = junctions["Kovach start, station 0.6"]      # the corner the C1 walls round off
            # the reconstruction's slope at 0.60 against the taper's: measured 0.0767 / 0.0792 deg (suction /
            # pressure) with the PCHIP, 0.00725 / 0.00749 deg with the quintic
            low, high = (0.07, 0.09) if reconstruction == LEGACY else (0.0065, 0.0085)
            assert low < math.degrees(abs(start.turn)) < high
            # Tangent kept (to the 1e-9 c tolerance in x of _nearest_station: 1e-7 rad), curvature jumps.
            fair = junctions["Kovach TE fair start"]
            assert abs(fair.turn) < 1e-6 and abs(fair.kappa_jump) > 100.0
        else:
            closing = [j for label, j in junctions.items() if label.startswith("closing segment")][0]
            assert math.degrees(abs(closing.turn)) > 1.0        # the closing segment meets the surface at a corner


# --- curve quantities ------------------------------------------------------------------------

def test_signed_curvature_and_wall_slopes_against_closed_forms():
    u = np.linspace(-1.0, 1.0, 41)
    radius = 0.37
    # Counterclockwise circle: kappa = +1/R; clockwise: -1/R.
    d1 = radius * np.column_stack((-np.sin(u), np.cos(u)))
    d2 = -radius * np.column_stack((np.cos(u), np.sin(u)))
    np.testing.assert_allclose(g.signed_curvature(d1, d2), 1.0 / radius, rtol=1e-14)
    np.testing.assert_allclose(g.signed_curvature(-d1, d2), -1.0 / radius, rtol=1e-14)
    # Parabola y = a m^2 with the nonuniform parameter m = u + u^3: the m-derivatives do
    # not depend on the parameter.
    a = 0.8
    m, m1, m2 = u + u**3, 1.0 + 3.0 * u**2, 6.0 * u
    d1 = np.column_stack((m1, 2.0 * a * m * m1))
    d2 = np.column_stack((m2, 2.0 * a * m1**2 + 2.0 * a * m * m2))
    tan_beta, second = g.wall_slopes(d1, d2)
    np.testing.assert_allclose(tan_beta, 2.0 * a * m, rtol=1e-13, atol=1e-15)
    np.testing.assert_allclose(second, 2.0 * a, rtol=1e-12)
    np.testing.assert_allclose(g.signed_curvature(d1, d2), 2.0 * a / (1.0 + (2.0 * a * m) ** 2) ** 1.5, rtol=1e-12)


def test_wall_slopes_mask_only_where_m_turns():
    """On a circle about the origin dm/du = 0 at u = 0 and pi: y(m) is singular there although
    the curve is smooth. Only those points are masked; right beside them the values are
    finite and equal kappa / cos(beta)^3."""
    u = np.array([0.0, 1e-3, 0.5, np.pi - 1e-3, np.pi])
    d1 = np.column_stack((-np.sin(u), np.cos(u)))
    d2 = -np.column_stack((np.cos(u), np.sin(u)))
    tan_beta, second = g.wall_slopes(d1, d2)
    assert np.isnan(tan_beta[[0, 4]]).all() and np.isnan(second[[0, 4]]).all()
    finite = [1, 2, 3]
    assert np.isfinite(second[finite]).all()
    cos_beta = d1[finite, 0] / np.hypot(*d1[finite].T)
    np.testing.assert_allclose(second[finite], 1.0 / cos_beta**3, rtol=1e-12)
    with pytest.raises(ValueError):
        g.wall_slopes(d1, d2, min_cos=0.0)


def test_curve_quantities_scale_with_the_length_unit():
    u = np.linspace(0.1, 1.2, 12)
    d1, d2 = np.column_stack((np.cos(u), 0.3 + u**2)), np.column_stack((-np.sin(u), 2.0 * u))
    tan_beta, second = g.wall_slopes(d1, d2)
    for factor in (0.05, 3.0):
        np.testing.assert_allclose(g.signed_curvature(factor * d1, factor * d2), g.signed_curvature(d1, d2) / factor,
                                   rtol=1e-14)
        tan_scaled, second_scaled = g.wall_slopes(factor * d1, factor * d2)
        np.testing.assert_allclose(tan_scaled, tan_beta, rtol=1e-15)
        np.testing.assert_allclose(second_scaled, second / factor, rtol=1e-14)


def test_parametric_curve_against_an_explicit_polynomial():
    """A single-span quintic (the Bezier form), written out in Bernstein polynomials here."""
    control = np.array([[0.0, 0.0], [0.2, 0.3], [0.4, -0.1], [0.7, 0.5], [0.9, 0.2], [1.0, 0.0]])
    curve = g.ParametricBladeCurve(np.r_[np.zeros(6), np.ones(6)], control, 1.0, 5)
    u = np.linspace(0.0, 1.0, 11)

    def bernstein(degree: int) -> np.ndarray:
        return np.array([math.comb(degree, k) * u**k * (1.0 - u) ** (degree - k) for k in range(degree + 1)]).T

    np.testing.assert_allclose(curve.evaluate(u), bernstein(5) @ control, atol=1e-15)
    np.testing.assert_allclose(curve.derivative(u, 1), 5.0 * bernstein(4) @ np.diff(control, axis=0), atol=1e-13)
    np.testing.assert_allclose(curve.derivative(u, 2), 20.0 * bernstein(3) @ np.diff(control, 2, axis=0), atol=1e-12)
    np.testing.assert_array_equal(curve.evaluate([0.0, 1.0]), control[[0, -1]])


def test_parametric_curve_is_read_only_and_checks_its_input():
    curve = build(surface_continuity="C2").normalized_curves.suction
    with pytest.raises(FrozenInstanceError):
        curve.degree = 3
    for array in (curve.knots, curve.control_points):
        with pytest.raises(ValueError):
            array[0] = 0.5
    with pytest.raises(ValueError, match="parameter"):
        curve.evaluate(1.01)
    with pytest.raises(ValueError, match="order"):
        curve.derivative(0.5, 6)
    with pytest.raises(ValueError, match="clamped"):
        g.ParametricBladeCurve(np.r_[np.zeros(6), 0.5, 0.5, np.ones(6)], np.zeros((8, 2)), 1.0, 5)


# --- C2 geometry -----------------------------------------------------------------------------

C2_CASES = [  # cl0, t_max_over_c, solidity, stagger_deg
    (1.2, 0.10, 1.0, 33.0),
    (0.0, 0.06, 1.0, 0.0),
    (0.4, 0.04, 0.8, -30.0),
    (1.6, 0.08, 2.0, 60.0),
    (0.8, 0.05, 1.5, 15.0),
]


def c2_pair(**changes) -> tuple[g.Cascade, g.Cascade]:
    """(C1, C2) of one blade, both with the C2 default thickness reconstruction: C2 represents
    the C1 construction of its own thickness."""
    return build(thickness_reconstruction=C2_THICKNESS, **changes), build(surface_continuity="C2", **changes)


@pytest.mark.parametrize("variant", g.PROFILE_VARIANTS)
@pytest.mark.parametrize("chord", (None, 0.05))
def test_c2_keeps_every_landmark_of_c1_exactly(variant, chord):
    c1, c2 = c2_pair(profile_variant=variant, physical_chord=chord)
    for label in ("normalized", "active"):
        a, b = getattr(c1, label), getattr(c2, label)
        for name in ("chord", "pitch", "leading_edge_point", "trailing_edge_point", "periodic_translation",
                     "leading_edge_center", "leading_edge_radius", "trailing_edge_center", "trailing_edge_radius",
                     "camberline", "equivalent_arc_camberline"):
            np.testing.assert_array_equal(getattr(b, name), getattr(a, name), err_msg=name)
        for name in ("suction_surface", "pressure_surface", "suction_surface_unstaggered",
                     "pressure_surface_unstaggered"):
            np.testing.assert_array_equal(getattr(b, name)[[0, -1]], getattr(a, name)[[0, -1]])
            assert len(getattr(b, name)) == len(getattr(a, name))
        curves = getattr(c2, f"{label}_curves")
        for name in ("suction", "pressure"):
            np.testing.assert_array_equal(getattr(curves, name).evaluate([0.0, 1.0]),
                                          getattr(b, f"{name}_surface")[[0, -1]])
    # The blade angles stay those of the equivalent circular arc, not of the spline tangent.
    for name in ("stagger", "camber_angle", "chi_le", "chi_te", "kappa_le", "kappa_te",
                 "throat_over_pitch_correlation"):
        assert getattr(c2, name) == getattr(c1, name)


def end_sample(reference: g.ReferenceSurface, last: bool) -> g.SurfaceSample:
    index = len(reference.pieces) - 1 if last else 0
    return reference.sample_piece(index, reference.piece_bounds[index])


@pytest.mark.parametrize("variant", g.PROFILE_VARIANTS)
def test_c2_ends_have_the_c1_tangent_and_curvature(variant):
    """LE: both surfaces leave the LE point on the C1 circle, with opposite unit tangents and
    curvature 1/R_LE, so the closed contour is G2 there. Kovach TE: the same on the TE circle.
    emery_base: each surface ends along its C1 closing segment, so the sharp TE keeps its
    wedge, an intentional corner."""
    c2 = build(surface_continuity="C2", profile_variant=variant, cl0=1.2, t_max_over_c=0.10)
    curves = c2.normalized_curves
    references = g.naca65_reference_surfaces(1.2, 0.10, profile_variant=variant, thickness_reconstruction=C2_THICKNESS)
    for reference, curve in zip(references, (curves.suction_unstaggered, curves.pressure_unstaggered)):
        first, last = end_sample(reference, False), end_sample(reference, True)
        np.testing.assert_allclose(curve.unit_tangent(0.0), first.tangent[0], atol=1e-12)
        np.testing.assert_allclose(curve.unit_tangent(1.0), last.tangent[-1], atol=1e-12)
        np.testing.assert_allclose(curve.curvature(0.0), first.kappa[0], rtol=1e-9)
        if variant == "kovach_sandercock":
            np.testing.assert_allclose(curve.curvature(1.0), last.kappa[-1], rtol=1e-9)
    suction, pressure = curves.suction_unstaggered, curves.pressure_unstaggered
    np.testing.assert_allclose(suction.unit_tangent(0.0), -pressure.unit_tangent(0.0), atol=1e-12)
    np.testing.assert_allclose(suction.curvature(0.0), -1.0 / 0.00687, rtol=1e-9)
    np.testing.assert_allclose(pressure.curvature(0.0), 1.0 / 0.00687, rtol=1e-9)
    cosine = float(suction.unit_tangent(1.0) @ pressure.unit_tangent(1.0))
    if variant == "kovach_sandercock":
        assert cosine == pytest.approx(-1.0, abs=1e-12)
        np.testing.assert_allclose([suction.curvature(1.0), pressure.curvature(1.0)], [-125.0, 125.0], rtol=1e-9)
    else:
        wedge_c1 = math.acos(float(end_sample(references[0], True).tangent[-1]
                                   @ end_sample(references[1], True).tangent[-1]))
        assert math.acos(cosine) == pytest.approx(wedge_c1, abs=1e-10)
        assert math.degrees(wedge_c1) > 1.0


def one_sided_at_knots(curve: g.ParametricBladeCurve, order: int) -> tuple[np.ndarray, np.ndarray]:
    """(left, right) values of the order-th derivative at every interior knot, from scipy's
    piecewise-polynomial form of the spline: the polynomial of the span on either side."""
    from scipy.interpolate import PPoly
    left, right = [], []
    for column in range(2):
        pp = PPoly.from_spline((curve.knots, curve.control_points[:, column], curve.degree))
        pp = pp.derivative(order) if order else pp
        spans = np.searchsorted(pp.x, curve.interior_knots)
        left.append([np.polyval(pp.c[:, i - 1], pp.x[i] - pp.x[i - 1]) for i in spans])
        right.append(pp.c[-1, spans])
    return np.array(left).T, np.array(right).T


@pytest.mark.parametrize("variant", g.PROFILE_VARIANTS)
@pytest.mark.parametrize("cl0, t", ((1.2, 0.10), (1.6, 0.04)))
def test_c2_is_continuous_to_the_second_derivative_at_every_knot(variant, cl0, t):
    c2 = build(surface_continuity="C2", profile_variant=variant, cl0=cl0, t_max_over_c=t)
    for name in ("suction_unstaggered", "pressure_unstaggered"):
        curve = getattr(c2.normalized_curves, name)
        length = curve.reference_length
        assert len(curve.interior_knots) > 300
        values = [one_sided_at_knots(curve, order) for order in range(3)]
        for order, (left, right) in enumerate(values):
            scale = length**order * (1.0 if order < 2 else max(1.0, np.max(np.abs(right))))
            assert np.max(np.hypot(*(left - right).T)) <= 1e-12 * scale, (name, order)
        kappa_left = g.signed_curvature(values[1][0], values[2][0])
        kappa_right = g.signed_curvature(values[1][1], values[2][1])
        assert np.max(np.abs(kappa_left - kappa_right)) <= 1e-9
        # The spans are genuine quintic pieces: the fifth derivative does jump.
        left5, right5 = one_sided_at_knots(curve, 5)
        assert np.max(np.hypot(*(left5 - right5).T)) > 1.0


@pytest.mark.parametrize("reconstruction", g.THICKNESS_RECONSTRUCTIONS)
@pytest.mark.parametrize("variant", g.PROFILE_VARIANTS)
def test_c2_replaces_each_c1_curvature_jump_by_a_monotone_ramp(variant, reconstruction):
    """Where C1's curvature only jumps (the Kovach TE fair start; with the legacy PCHIP also
    the thickness-table stations), C2 has no jump, and what it adds to C1 is the jump times
    a monotone step, Phi - H, rising from 0 to 1/2 before the junction and from -1/2 to 0
    after it: no overshoot and no ringing, however fast the curvature of the pieces
    themselves changes. Phi is the integral of the kernel (35/32)(1 - x^2)^3 on the
    transition's half-width, written out here. Measured against C1 at the same arc length
    over 90 % of the transition; the 3 % allowance covers the measured 1.9 % (PCHIP: smaller
    jumps near the LE, where the neighbours' shift compensations add about 0.007 /c and the
    knots resolve the step to about 1 %; the quintic's Kovach TE fair start 0.31 %). The
    quintic leaves no table-station junction at all: on emery_base nothing downstream of the
    LE fair jumps by 1 /c or more (its closing segments are corners)."""
    def step(x: np.ndarray) -> np.ndarray:
        return 0.5 + 35.0 / 32.0 * (x - x**3 + 0.6 * x**5 - x**7 / 7.0)

    cl0, t = 1.2, 0.10
    c2 = build(surface_continuity="C2", profile_variant=variant, cl0=cl0, t_max_over_c=t,
               thickness_reconstruction=reconstruction)
    fits = g._naca65_quintic_fits(cl0, t, variant, reconstruction)
    largest, checked = 0.0, 0
    labels = [tr.junction.label for fit in fits for tr in fit.source.transitions]
    assert any(label.startswith("table station") for label in labels) == (reconstruction == LEGACY)
    for fit, curve in zip(fits, (c2.normalized_curves.suction_unstaggered, c2.normalized_curves.pressure_unstaggered)):
        reference, bounds = fit.reference, fit.reference.piece_bounds
        for transition in fit.source.transitions:
            junction, half_width = transition.junction, transition.half_width
            jump = junction.kappa_jump
            # Junctions inside the LE zone are replaced by the LE fair, not smoothed.
            if abs(jump) < 1.0 or abs(junction.turn) > 1e-6 or junction.s - half_width <= fit.leading_edge.anchor:
                continue
            assert half_width > 0.0
            largest, checked = max(largest, abs(jump)), checked + 1
            at = float(fit.parameter(junction.s))
            across = curve.curvature([at - 1e-8, at + 1e-8])
            assert abs(across[1] - across[0]) < 1e-3 * abs(jump), junction.label
            for side, piece in ((-1.0, junction.index), (1.0, junction.index + 1)):
                s = np.sort(np.clip(junction.s + side * np.linspace(0.0, 0.9 * half_width, 181)[1:], *bounds[piece]))
                c1 = reference.sample_piece(piece, np.concatenate(([bounds[piece][0]], s, [bounds[piece][1]])))
                added = (curve.curvature(fit.parameter(c1.s[1:-1])) - c1.kappa[1:-1]) / jump
                designed = step((c1.s[1:-1] - junction.s) / half_width) - (side > 0)
                assert np.max(np.abs(added - designed)) < 0.03, (junction.label, side)
    if reconstruction == LEGACY or variant == "kovach_sandercock":
        assert checked > 0
    else:
        assert checked == 0
    if variant == "kovach_sandercock":
        assert largest > 100.0          # the Kovach TE fair start is among them


@pytest.mark.parametrize("variant", g.PROFILE_VARIANTS)
@pytest.mark.parametrize("cl0, t, solidity, stagger_deg", C2_CASES)
def test_c2_preserves_the_c1_shape_downstream_of_the_le(variant, cl0, t, solidity, stagger_deg):
    """Downstream of each LE anchor the C1 surface (of the same thickness reconstruction, the
    quintic default) is kept to the C2 representation's target: the largest normal distance
    from every analytic C1 piece is 1e-6 c within round-off (measured 1.0004e-6 c on these
    cases), the RMS below 3e-7 (measured 5.6e-8; 2.2e-7 with the legacy PCHIP, whose station
    junctions all carried transitions). Inside the LE zone the geometry is redesigned
    (measured at most 1.19e-3 c here, on the thin 0.4/0.04 blade's pressure side, whose
    anchor the quintic nose moves from 0.00716 to 0.00771 c of arc; 8.2e-4 c with the PCHIP;
    1.25e-3 covers it). The throat and o/s move with the LE where the throat reaches the
    neighbour's LE (solidity 2.0 at 60 deg: measured 1.38e-4 c and 2.77e-4; 2e-4 and 4e-4),
    the blade area by the LE change (measured 1.75e-4 relative, thin blades; 2e-4); the
    maximum thickness, far from the LE, by less than 2e-6 (measured 3.4e-10)."""
    c1, c2 = c2_pair(profile_variant=variant, cl0=cl0, t_max_over_c=t, solidity=solidity, stagger_deg=stagger_deg)
    for fit in g._naca65_quintic_fits(cl0, t, variant, C2_THICKNESS):
        sample = fit.reference.sample(201)
        distance = fit.curve.project(sample.points, fit.parameter(sample.s))[1]
        zone = sample.s <= fit.leading_edge.anchor
        weights = sample.weights * ~zone
        assert np.max(np.abs(distance[~zone])) <= 1.01e-6
        assert math.sqrt(np.sum(weights * distance**2) / np.sum(weights)) <= 3e-7
        assert np.max(np.abs(distance[zone])) <= 1.25e-3                     # measured 1.193e-3
        assert fit.max_deviation <= 1.01e-6 and fit.le_max_deviation <= 1.25e-3
    assert abs(c2.normalized.throat_opening - c1.normalized.throat_opening) < 2e-4
    assert abs(c2.throat_over_pitch - c1.throat_over_pitch) < 4e-4
    n1, n2 = c1.normalized, c2.normalized
    area1 = shoelace(np.vstack((n1.suction_surface_unstaggered[::-1], n1.pressure_surface_unstaggered[1:])))
    area2 = shoelace(np.vstack((n2.suction_surface_unstaggered[::-1], n2.pressure_surface_unstaggered[1:])))
    assert area2 == pytest.approx(area1, rel=2e-4)
    assert max_thickness(n2) == pytest.approx(max_thickness(n1), abs=2e-6)


@pytest.mark.parametrize("variant", g.PROFILE_VARIANTS)
@pytest.mark.parametrize("cl0, t", [(case[0], case[1]) for case in C2_CASES])
def test_previous_c2_fit_preserves_the_whole_c1_surface(variant, cl0, t):
    """fit_quintic_surface, the C2 fit that keeps the legacy LE fair, is unchanged: on every
    analytic C1 piece, LE included, the normal distance is 1e-6 c within round-off (measured
    1.001e-6 c over the 40-blade, 640-cascade sweep), the RMS below 3e-7."""
    for reference in g.naca65_reference_surfaces(cl0, t, profile_variant=variant, thickness_reconstruction=LEGACY):
        fit = g.fit_quintic_surface(reference)
        sample = reference.sample(201)
        distance = fit.curve.project(sample.points, sample.s / reference.length)[1]
        weights = sample.weights
        assert np.max(np.abs(distance)) <= 1.01e-6
        assert math.sqrt(np.sum(weights * distance**2) / np.sum(weights)) <= 3e-7


def turn_back(d1: np.ndarray, d2: np.ndarray, s: np.ndarray) -> tuple[float, float, float]:
    """The first run of negative curvature (pressure side): how far the tangent turns back over
    it (deg), and where it starts and ends; (0, nan, nan) if there is none."""
    negative = g.signed_curvature(d1, d2) < 0.0
    if not negative.any():
        return 0.0, math.nan, math.nan
    i0 = int(np.argmax(negative))
    i1 = i0 + int(np.argmax(~negative[i0:])) - 1
    angle = np.degrees(np.unwrap(np.arctan2(d1[:, 1], d1[:, 0])))
    return angle[i0] - angle[i1], s[i0], s[i1]


def test_pressure_side_le_s_bend_is_kept_by_c1_and_removed_by_the_robust_c2():
    """The PS LE fair of the 65-(12A10)10 turns back by 4.3 deg (curvature < 0 over m/c
    0.0069 to 0.0085 at 33 deg stagger). C1 keeps it (the validated baseline), so does the
    previous C2 fit (fit_quintic_surface, within 0.01 deg); the robust C2 LE has none."""
    c2 = build(surface_continuity="C2", cl0=1.2, t_max_over_c=0.10, solidity=1.0, stagger_deg=33.0)
    reference = g.naca65_reference_surfaces(1.2, 0.10, profile_variant="kovach_sandercock",
                                            thickness_reconstruction=LEGACY)[1]
    sample = reference.sample(4001)
    rows = sample.s / reference.length < 0.02
    c1_bend = turn_back(sample.tangent[rows], sample.second[rows], sample.s[rows] / reference.length)
    u = np.linspace(0.0, 0.02, 40001)
    previous = g.fit_quintic_surface(reference).curve
    previous_bend = turn_back(previous.derivative(u, 1), previous.derivative(u, 2), u)
    assert c1_bend[0] == pytest.approx(4.3, abs=0.05)
    assert previous_bend[0] == pytest.approx(c1_bend[0], abs=0.01)
    np.testing.assert_allclose(previous_bend[1:], c1_bend[1:], atol=2e-5)
    robust = c2.normalized_curves.pressure_unstaggered
    assert turn_back(robust.derivative(u, 1), robust.derivative(u, 2), u)[0] == 0.0
    assert g.leading_edge_fairness(robust, 0.02)["max_reverse_turn_deg"] == 0.0


def crossings(path_a: np.ndarray, path_b: np.ndarray, *, same: bool = False) -> int:
    """Proper crossings of two polylines, by brute force. Segments that only meet at a shared
    end point (the LE and TE of the two surfaces, neighbours along one path) do not count:
    a crossing must lie more than 1e-9 of a segment from the segment ends."""
    a0, ra = path_a[:-1, None], np.diff(path_a, axis=0)[:, None]
    b0, rb = path_b[None, :-1], np.diff(path_b, axis=0)[None]
    det = ra[..., 0] * rb[..., 1] - ra[..., 1] * rb[..., 0]
    gap = b0 - a0
    with np.errstate(divide="ignore", invalid="ignore"):
        along_a = (gap[..., 0] * rb[..., 1] - gap[..., 1] * rb[..., 0]) / det
        along_b = (gap[..., 0] * ra[..., 1] - gap[..., 1] * ra[..., 0]) / det
    inner = 1e-9
    hit = ((det != 0) & (along_a > inner) & (along_a < 1 - inner) & (along_b > inner) & (along_b < 1 - inner))
    if same:
        i, j = np.nonzero(hit)
        return int(np.sum(np.abs(i - j) > 1))
    return int(hit.sum())


@pytest.mark.parametrize("variant", g.PROFILE_VARIANTS)
@pytest.mark.parametrize("cl0, t, solidity, stagger_deg", C2_CASES)
def test_c2_blade_is_simple_and_its_passage_open(variant, cl0, t, solidity, stagger_deg):
    c2 = build(surface_continuity="C2", profile_variant=variant, cl0=cl0, t_max_over_c=t, solidity=solidity,
               stagger_deg=stagger_deg)
    u = np.linspace(0.0, 1.0, 1201)
    suction, pressure = c2.normalized_curves.suction.evaluate(u), c2.normalized_curves.pressure.evaluate(u)
    loop = np.vstack((suction[::-1], pressure[1:]))
    assert crossings(loop, loop, same=True) == 0
    assert crossings(suction, pressure) == 0
    assert shoelace(np.vstack((c2.normalized.suction_surface_unstaggered[::-1],
                               c2.normalized.pressure_surface_unstaggered[1:]))) > 0.0
    neighbor = loop + c2.normalized.periodic_translation
    assert crossings(loop, neighbor) == 0
    gap = polyline_distance(loop, neighbor).min()
    assert 0.0 < c2.normalized.throat_opening <= gap + 1e-6


@pytest.mark.parametrize("variant", g.PROFILE_VARIANTS)
def test_c2_physical_chord_is_an_exact_scale_twin(variant):
    unit = build(surface_continuity="C2", profile_variant=variant)
    metric = build(surface_continuity="C2", profile_variant=variant, physical_chord=0.05)
    for item in fields(g.CascadeCoordinates):
        np.testing.assert_array_equal(getattr(metric.normalized, item.name), getattr(unit.normalized, item.name))
        np.testing.assert_allclose(getattr(metric.active, item.name),
                                   0.05 * np.asarray(getattr(unit.normalized, item.name)), rtol=1e-15, atol=0.0)
    # The curves are exact twins (control points times the chord, same knots and parameters);
    # what is computed from them agrees to their evaluation round-off, which is
    # eps |P| / h^2 in X'' on the finest spans (h = 3e-5 c at a rounded corner): 1e-6 /c.
    u = np.linspace(0.0, 1.0, 3001)
    for name in ("suction", "pressure", "suction_unstaggered", "pressure_unstaggered"):
        chord_1, metres = getattr(unit.normalized_curves, name), getattr(metric.active_curves, name)
        np.testing.assert_array_equal(metres.control_points, 0.05 * chord_1.control_points)
        np.testing.assert_array_equal(metres.knots, chord_1.knots)
        np.testing.assert_allclose(metres.evaluate(u), 0.05 * chord_1.evaluate(u), rtol=1e-15, atol=1e-19)
        np.testing.assert_allclose(metres.curvature(u), chord_1.curvature(u) / 0.05, rtol=1e-9, atol=1e-6 / 0.05)
        tan_chord, second_chord = chord_1.wall_slope(u)
        tan_metres, second_metres = metres.wall_slope(u)
        d1 = chord_1.derivative(u, 1)
        steep = np.abs(d1[:, 0]) / np.hypot(*d1.T) > 0.05       # away from the m-turning points
        np.testing.assert_allclose(tan_metres[steep], tan_chord[steep], rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(second_metres[steep], second_chord[steep] / 0.05, rtol=1e-9, atol=1e-5 / 0.05)
    for name in ("suction_parameters", "pressure_parameters"):
        np.testing.assert_array_equal(getattr(metric.active_curves, name), getattr(unit.normalized_curves, name))
    for name in ("throat_over_pitch", "throat_over_chord", "kappa_le", "kappa_te", "stagger"):
        assert getattr(metric, name) == getattr(unit, name)


def test_c2_curve_does_not_depend_on_n_camber_stations():
    """n_camber_stations sets where the arrays sample the curve, not the curve: the arrays
    are the curve at the arc-length fractions of the C1 vertices (C1 of the same thickness
    reconstruction)."""
    coarse, fine = build(surface_continuity="C2"), build(surface_continuity="C2", n_camber_stations=301)
    for name in ("suction_unstaggered", "pressure_unstaggered"):
        np.testing.assert_array_equal(getattr(coarse.normalized_curves, name).control_points,
                                      getattr(fine.normalized_curves, name).control_points)
    assert len(fine.normalized.suction_surface) > len(coarse.normalized.suction_surface)
    for c in (coarse, fine):
        c1 = build(n_camber_stations=c.spec.n_camber_stations, thickness_reconstruction=C2_THICKNESS)
        for name in ("suction", "pressure"):
            u = getattr(c.normalized_curves, f"{name}_parameters")
            polyline = getattr(c1.normalized, f"{name}_surface_unstaggered")
            chord_length = np.concatenate(([0.0], np.cumsum(np.hypot(*np.diff(polyline, axis=0).T))))
            np.testing.assert_allclose(u, chord_length / chord_length[-1], atol=2e-4)
            np.testing.assert_allclose(getattr(c.normalized_curves, f"{name}_unstaggered").evaluate(u),
                                       getattr(c.normalized, f"{name}_surface_unstaggered"), rtol=0.0, atol=1e-15)


def test_validate_cascade_checks_the_curves():
    c2 = build(surface_continuity="C2")
    moved = replace(c2.normalized_curves, suction=c2.normalized_curves.suction.scaled(1.0 + 1e-9))
    with pytest.raises(g.GeometryError, match="sampled at its parameters"):
        g.validate_cascade(replace(c2, normalized_curves=moved, active_curves=moved))
    with pytest.raises(g.GeometryError, match="no parametric surface curves"):
        g.validate_cascade(replace(c2, normalized_curves=None, active_curves=None))
    with pytest.raises(g.GeometryError, match="C1 cascade carries"):
        g.validate_cascade(replace(build(), normalized_curves=c2.normalized_curves, active_curves=c2.active_curves))


# --- the frozen grid takes either representation ---------------------------------------------

@pytest.mark.parametrize("variant", g.PROFILE_VARIANTS)
def test_grid_accepts_c1_and_c2_alike(variant):
    """grid.py builds a grid from either cascade, with the same stations and blade run. Read
    the same way (the C2 cascade's sampled arrays through the C1 path, which is what grid.py
    did for C2 before the 2026-11-11 hand-off), the C2 blade is a different surface at the
    LE, so the wall nodes there move by about the LE deviation (measured 3.2e-5 c, the LE
    deviation itself is 1.27e-4 c); past both LE zones by less than 2e-6, as before. Read
    from its authoritative curves (grid.py section 7, the C2 path now), the C2 walls are the
    curves themselves while the C1 walls stay a PCHIP of the C1 vertices, so past the LE
    zones they differ by that PCHIP's own error as well: below 1.0e-5 c (measured 9.6e-6 c
    for both variants; grid.py measures 8.3e-6 c for that PCHIP against the blade at 16
    times the camber stations, and the C2 fit adds at most 1e-6 c)."""
    from turbolenza.b2b import grid
    from turbolenza.b2b.config import GridSettings
    settings = GridSettings(n_passage=61, n_eta=11, upstream_extension_chords=1.0, downstream_extension_chords=1.0)
    c1, c2 = c2_pair(profile_variant=variant, cl0=1.2, t_max_over_c=0.10, solidity=1.0, stagger_deg=33.0)
    sampled = replace(c2, active_curves=None, normalized_curves=None)
    first, second, curve_grid = (grid.make_grid(c, settings, math.radians(45.0)) for c in (c1, sampled, c2))
    assert not second.has_wall_curves and curve_grid.has_wall_curves
    le_deviation = max(c2.metadata["ss_le_max_deviation_over_c"], c2.metadata["ps_le_max_deviation_over_c"])
    curves = c2.normalized_curves
    past_le = max(float(np.max(curves.suction.evaluate(np.linspace(0, c2.metadata["ss_le_anchor_u"], 201))[:, 0])),
                  float(np.max(curves.pressure.evaluate(np.linspace(0, c2.metadata["ps_le_anchor_u"], 201))[:, 0])))
    blade = slice(first.first_blade_i, first.last_blade_i + 1)
    for item, past in ((second, 2e-6), (curve_grid, 1.0e-5)):
        np.testing.assert_array_equal(item.m, first.m)
        assert (item.first_blade_i, item.last_blade_i) == (first.first_blade_i, first.last_blade_i)
        assert np.max(np.abs(item.nodes - first.nodes)) <= le_deviation + past
        columns = item.m > past_le + settings_spacing(item.m)
        assert np.max(np.abs(item.nodes[columns] - first.nodes[columns])) < past
        assert np.min(item.upper_boundary[blade, 1] - item.lower_boundary[blade, 1]) > 0.0
        widths = [np.min(grid_.upper_boundary[blade, 1] - grid_.lower_boundary[blade, 1]) for grid_ in (first, item)]
        assert widths[1] == pytest.approx(widths[0], abs=le_deviation + past)
    assert np.min(first.upper_boundary[blade, 1] - first.lower_boundary[blade, 1]) > 0.0


def settings_spacing(m: np.ndarray) -> float:
    """The largest step of the m stations: a column farther than that from the LE zone has wall
    nodes interpolated from surface points beyond it only."""
    return float(np.max(np.diff(m)))


# --- the robust C2 leading edge ------------------------------------------------------------

LE_AUDIT = [(1.2, 0.10), (1.6, 0.10), (1.2, 0.05), (1.2, 0.04), (0.0, 0.04)]


def legacy_le_defects(reference: g.ReferenceSurface, end: float) -> tuple[float, int]:
    """(largest reverse turn in deg, curvature sign changes) of a C1 reference over s <= end,
    from its exact piece tangents and curvatures."""
    sample = reference.sample(2001)
    rows = sample.s <= end
    sense = np.sign(sample.kappa[0])
    angle = np.unwrap(np.arctan2(sample.tangent[rows, 1], sample.tangent[rows, 0])) * sense
    kappa = sample.kappa[rows]
    signs = np.sign(kappa[np.abs(kappa) > 0.0])
    return math.degrees(float(np.max(np.maximum.accumulate(angle) - angle))), int(np.sum(signs[1:] != signs[:-1]))


@pytest.mark.parametrize("chord", (None, 0.05))
def test_c2_le_radius_and_apex_are_the_legacy_nose(chord):
    """R_LE is the legacy reference radius, not one a spline prefers: 0.00687 (t / 0.10)^2,
    centred on the C1 LE circle; the apex is the C1 LE vertex, where both surfaces have
    curvature -+1/R_LE (to 1e-9) and opposite tangents."""
    for cl0, t in ((1.2, 0.10), (0.4, 0.04)):
        c2 = build(surface_continuity="C2", cl0=cl0, t_max_over_c=t, physical_chord=chord)
        radius = 0.00687 * (t / 0.10) ** 2
        unit = chord or 1.0
        assert c2.metadata["le_radius_over_c"] == pytest.approx(radius, rel=1e-15)
        assert c2.metadata["le_radius"] == pytest.approx(radius * unit, rel=1e-15)
        assert c2.metadata["le_radius"] == float(c2.active.leading_edge_radius)
        for fit in g._naca65_quintic_fits(cl0, t, "kovach_sandercock", C2_THICKNESS):
            nose = fit.leading_edge.nose
            assert nose.radius == pytest.approx(float(c2.normalized.leading_edge_radius), rel=1e-15)
            np.testing.assert_allclose(g.rotate_by_stagger(nose.center, c2.stagger), c2.normalized.leading_edge_center,
                                       atol=1e-15)
            np.testing.assert_array_equal(nose.apex, c2.normalized.leading_edge_point)
        curves = c2.active_curves
        for name, sense in (("suction_unstaggered", -1.0), ("pressure_unstaggered", 1.0)):
            kappa = float(np.ravel(getattr(curves, name).curvature(0.0))[0])
            assert kappa * radius * unit == pytest.approx(sense, abs=1e-9)
        np.testing.assert_allclose(curves.suction.unit_tangent(0.0), -curves.pressure.unit_tangent(0.0), atol=1e-12)
    with pytest.raises(ValueError, match="R_LE"):
        g.LeadingEdgeNose(np.zeros(2), np.array([0.01, 0.0]), 0.02, "test")


@pytest.mark.parametrize("cl0, t", LE_AUDIT)
def test_le_fair_joins_the_nose_and_the_source_with_g2(cl0, t):
    """Verified on the numbers, not taken from the construction: at the departure the fair has
    r = R_LE with r' = 0 and r - R_LE cubic in phi (its curvature leaves 1/R_LE to third
    order); at the anchor it has the source point (to 1e-12 c), unit tangent (1e-12) and
    curvature (1e-12 relative). On the C2 curve, which represents that target, the curvature
    across the anchor is the source one within its representation error (measured at most
    0.11 /c in the LE zone; 0.2 covers it)."""
    for fit in g._naca65_quintic_fits(cl0, t, "kovach_sandercock", C2_THICKNESS):
        le, source = fit.leading_edge, fit.source
        radius = le.nose.radius
        r, slope = le.radius(np.array([0.0, le.min_arc, le.min_arc + 1e-3]))
        assert r[0] == r[1] == radius and slope[1] == 0.0
        assert (r[2] - radius) / radius < 1e-6          # cubic in phi - min_arc: (1e-3)^3 scale
        point, d1, d2 = (value[0] for value in source.evaluate(np.array([le.anchor])))
        np.testing.assert_allclose(le.points(le.anchor_angle)[0], point, rtol=0.0, atol=1e-12)
        end = le.evaluate(np.array([1.0]))
        tangent = end[1][0] / np.hypot(*end[1][0])
        np.testing.assert_allclose(tangent, d1 / np.hypot(*d1), atol=1e-12)
        kappa_source = float(g.signed_curvature(d1[None], d2[None])[0])
        kappa_fair = float(g.signed_curvature(end[1], end[2])[0])
        assert kappa_fair == pytest.approx(kappa_source, rel=1e-12)
        u = fit.anchor_u
        across = fit.curve.curvature([u - 1e-6, u, u + 1e-6])
        assert np.max(np.abs(across - kappa_source)) < 0.2


@pytest.mark.parametrize("variant", g.PROFILE_VARIANTS)
@pytest.mark.parametrize("cl0, t", LE_AUDIT + [(case[0], case[1]) for case in C2_CASES])
def test_robust_le_has_no_inflection_and_turns_monotonically(variant, cl0, t):
    """Over its LE zone each C2 surface keeps the sign of its curvature (no inflection), its
    tangent never turns back (largest reverse turn 0 deg), and |kappa| does not rise beyond
    the spline representation ripple (measured 2.2e-6 / R_LE; 1e-5 covers it) nor exceed
    1/R_LE (measured 1.3e-5 relative; 5e-5)."""
    for fit in g._naca65_quintic_fits(cl0, t, variant, C2_THICKNESS):
        diag = g.leading_edge_fairness(fit.curve, fit.anchor_u)
        assert diag["sign_changes"] == 0 and diag["max_reverse_turn_deg"] == 0.0
        radius = fit.leading_edge.nose.radius
        kappa = np.abs(fit.curve.curvature(np.linspace(0.0, fit.anchor_u, 20001)))
        assert np.max(np.maximum(np.diff(kappa), 0.0)) * radius < 1e-5
        assert np.max(kappa) * radius < 1.0 + 5e-5
        assert fit.leading_edge.fairness <= g.LE_FAIRNESS_LIMIT
        assert np.all(np.diff(fit.leading_edge.control) >= 0.0)


@pytest.mark.parametrize("cl0, t", LE_AUDIT)
def test_robust_le_removes_the_legacy_le_defects(cl0, t):
    """The legacy fairs of the audit set turn back (the PS S-bend, the thin-blade chin) over
    what the robust LE replaces; the robust LE does not, on either surface."""
    fits = g._naca65_quintic_fits(cl0, t, "kovach_sandercock", C2_THICKNESS)
    legacy = [legacy_le_defects(fit.reference, fit.leading_edge.anchor) for fit in fits]
    assert max(reverse for reverse, _ in legacy) > 1.0          # each audit blade has a defect to remove
    for fit in fits:
        diag = g.leading_edge_fairness(fit.curve, fit.anchor_u)
        assert diag["max_reverse_turn_deg"] == 0.0 and diag["sign_changes"] == 0


def test_thin_blade_le_has_no_loop_or_crossing():
    """A dense close-up of the 65-(12A10)04 LE (both surfaces to twice their anchors): no
    self-crossing, no SS/PS crossing, and the two surfaces only meet at the apex."""
    fits = g._naca65_quintic_fits(1.2, 0.04, "kovach_sandercock", C2_THICKNESS)
    u = np.linspace(0.0, 2.0 * max(fit.anchor_u for fit in fits), 6001)
    suction, pressure = (fit.curve.evaluate(u) for fit in fits)
    loop = np.vstack((suction[::-1], pressure[1:]))
    assert crossings(loop, loop, same=True) == 0
    assert crossings(suction, pressure) == 0
    assert polyline_distance(suction[200:], pressure).min() > 0.0


def le_probe(cl0: float, t: float) -> dict[str, np.ndarray]:
    """The designed LE (the fair at fixed fractions of its arc length: points, tangents,
    curvature), anchors, departures, throat, and the C2 curve at fixed u near the LE."""
    fits = g._naca65_quintic_fits(cl0, t, "kovach_sandercock", C2_THICKNESS)
    fraction = np.linspace(0.0, 1.0, 41)
    c2 = build(surface_continuity="C2", cl0=cl0, t_max_over_c=t, solidity=1.0, stagger_deg=33.0)
    out = {name: [] for name in ("fair points", "fair tangents", "fair curvature")}
    for fit in fits:
        le = fit.leading_edge
        phi = le.angle_at(fraction * le.length)
        _, d1, d2 = le.evaluate(phi / le.anchor_angle)
        out["fair points"].append(le.points(phi).ravel())
        out["fair tangents"].append((d1 / np.hypot(*d1.T)[:, None]).ravel())
        out["fair curvature"].append(g.signed_curvature(d1, d2) * le.nose.radius)
    probe = {name: np.concatenate(values) for name, values in out.items()}
    u = np.linspace(0.0, 0.03, 61)
    probe.update({
        "curve points": np.concatenate([fit.curve.evaluate(u).ravel() for fit in fits]),
        "anchor": np.array([fit.leading_edge.anchor for fit in fits]),
        "earliest anchor": np.array([fit.leading_edge.earliest_anchor for fit in fits]),
        "departure": np.array([fit.leading_edge.departure_angle for fit in fits]),
        "throat": np.array([c2.normalized.throat_opening]),
    })
    return probe


@pytest.mark.parametrize("cl0, t, which, step", [
    (1.2, 0.0700, "t", 0.001),      # where an anchor rule by fairness budget switched branches (measured)
    (1.2, 0.0510, "t", 0.001),      # where the anchor leaves the source start
    (1.48, 0.10, "cl0", 0.02),      # a second branch switch of that rule
    (0.8, 0.10, "cl0", 0.02),       # near where the legacy PS S-bend first appears
])
def test_le_construction_responds_continuously_to_the_blade_parameters(cl0, t, which, step):
    """Halving test of continuity: for a continuous map, each half of a parameter step changes
    every LE output (the designed fair at fixed fractions of its length: points, tangents,
    curvature; anchors, departures, throat) by about half of what the whole step does, in
    the max norm over that output; a jump would show up in one half nearly whole. Allowance:
    3/4 of the whole step (smooth: 1/2) plus round-off. The C2 curve at fixed u gets the
    representation error on top (its knot count is an integer, so it is refitted on a new
    knot vector now and then: measured 1.3e-9 c; 1e-8 covers it)."""
    def at(fraction: float) -> dict[str, np.ndarray]:
        value = round((t if which == "t" else cl0) + fraction * step, 7)
        return le_probe(cl0, value) if which == "t" else le_probe(value, t)
    first, middle, last = at(0.0), at(0.5), at(1.0)
    for key in first:
        whole = float(np.max(np.abs(last[key] - first[key])))
        halves = max(float(np.max(np.abs(middle[key] - first[key]))), float(np.max(np.abs(last[key] - middle[key]))))
        floor = 1e-8 if key == "curve points" else 1e-12
        assert halves <= 0.75 * whole + floor, (key, halves, whole)


def test_le_result_does_not_depend_on_the_anchor_scan_step(monkeypatch):
    """The anchor is refined by bisection to the point where the fairness budget is met, so
    halving the scan step leaves it (and the curve) unchanged to round-off."""
    first = g._naca65_quintic_fits(1.2, 0.10, "kovach_sandercock", C2_THICKNESS)
    g._naca65_quintic_fits.cache_clear()
    monkeypatch.setattr(g, "LE_ANCHOR_STEP", g.LE_ANCHOR_STEP / 2.0)
    try:
        second = g._naca65_quintic_fits(1.2, 0.10, "kovach_sandercock", C2_THICKNESS)
    finally:
        g._naca65_quintic_fits.cache_clear()
    u = np.linspace(0.0, 1.0, 4001)
    for a, b in zip(first, second):
        assert abs(a.leading_edge.anchor - b.leading_edge.anchor) < 1e-12
        assert np.max(np.hypot(*(a.curve.evaluate(u) - b.curve.evaluate(u)).T)) < 1e-10


def test_c2_le_scales_with_the_chord():
    """The LE construction is scale invariant: the physical curves are the chord-1 ones times
    c, so R_LE scales with c, curvature with 1/c, dkappa/ds with 1/c^2 (checked by central
    differences along the curves, to their round-off), angles and the normalized LE metadata
    (departure and anchor u, deviations over c, fairness) are the same."""
    unit, metric = build(surface_continuity="C2"), build(surface_continuity="C2", physical_chord=0.05)
    for key, value in unit.metadata.items():
        if key.startswith(("ss_le", "ps_le", "le_")) and key != "le_radius":
            assert metric.metadata[key] == value, key
    assert metric.metadata["le_radius"] == pytest.approx(0.05 * unit.metadata["le_radius"], rel=1e-15)
    h = 1e-6
    for name in ("suction", "pressure"):
        a, b = getattr(unit.normalized_curves, name), getattr(metric.active_curves, name)
        u = np.linspace(h, unit.metadata[f"{name[0]}s_le_anchor_u"] - h, 801)
        rate_a = (a.curvature(u + h) - a.curvature(u - h)) / (2 * h * np.hypot(*a.derivative(u, 1).T))
        rate_b = (b.curvature(u + h) - b.curvature(u - h)) / (2 * h * np.hypot(*b.derivative(u, 1).T))
        np.testing.assert_allclose(rate_b, rate_a / 0.05**2, rtol=1e-6, atol=1e-6 * np.max(np.abs(rate_a)) / 0.05**2)
        np.testing.assert_allclose(b.unit_tangent(u), a.unit_tangent(u), atol=1e-14)


@pytest.mark.parametrize("scale", (0.8, 1.25))
def test_le_engine_takes_a_prescribed_le_radius(scale):
    """The family-independent engine, called as a custom blade would call it: its own nose
    (here R = scale x the NACA radius, the apex and the nose axis kept) and a source surface.
    The fair has curvature 1/R at the apex, lands on the source with G2, turns monotonically
    and sits LE_ANCHOR_OFFSET R past the earliest anchor with a fair."""
    reference = g.naca65_reference_surfaces(1.2, 0.10, profile_variant="kovach_sandercock",
                                            thickness_reconstruction=C2_THICKNESS)[1]
    legacy = g._naca65_nose(1.2, 0.10, reference.leading_edge_point)
    radius = scale * legacy.radius
    nose = g.LeadingEdgeNose(legacy.apex, legacy.apex + radius * legacy.axis, radius, "prescribed")
    source = g.SmoothedSurface(reference, 2)
    fair = g.fair_leading_edge(nose, -1.0, source)
    start = fair.evaluate(np.array([0.0]))
    assert float(g.signed_curvature(start[1], start[2])[0]) == pytest.approx(1.0 / radius, rel=1e-12)
    np.testing.assert_array_equal(fair.points(0.0)[0], nose.apex)
    point, d1, d2 = (value[0] for value in source.evaluate(np.array([fair.anchor])))
    np.testing.assert_allclose(fair.points(fair.anchor_angle)[0], point, atol=1e-12)
    end = fair.evaluate(np.array([1.0]))
    kappa = float(g.signed_curvature(d1[None], d2[None])[0])
    assert float(g.signed_curvature(end[1], end[2])[0]) == pytest.approx(kappa, rel=1e-12)
    r = fair.radius(np.linspace(0.0, fair.anchor_angle, 2001))[0]
    assert np.all(np.diff(r) >= -1e-15 * np.max(r)) and r[0] == radius and fair.fairness <= g.LE_FAIRNESS_LIMIT
    assert fair.anchor == pytest.approx(fair.earliest_anchor + g.LE_ANCHOR_OFFSET * radius, abs=1e-15)


@pytest.mark.parametrize("reconstruction", g.THICKNESS_RECONSTRUCTIONS)
def test_smoothed_surface_is_the_c2_target_with_exact_derivatives(reconstruction):
    """SmoothedSurface evaluates the target of fit_quintic_surface (kinks smoothed, shifts
    taken back) at any s: against the sampled target of _c2_target downstream of the LE
    junctions (measured 1.1e-9 c, its trapezoid displacement), its derivatives against central
    differences of its own points (measured 2.7e-9 for h = 1e-6), and its curvature across
    every piece boundary (continuous to the offset of the transition centre from the
    boundary: measured 1.2e-5 /c at most)."""
    for reference in g.naca65_reference_surfaces(1.2, 0.08, profile_variant="emery_base",
                                                 thickness_reconstruction=reconstruction):
        source = g.SmoothedSurface(reference, 2)
        transitions = g._junction_transitions(reference)
        grid_s, spacing = g._c2_knot_spacing(reference, transitions)
        sample = g._c2_samples(reference, grid_s, spacing)
        target, _ = g._c2_target(sample, transitions)
        rows = sample.s > source.start + 0.03
        assert np.max(np.hypot(*(source.evaluate(sample.s[rows])[0] - target[rows]).T)) < 1e-8
        s = np.linspace(source.start + 0.003, source.end - 0.003, 200)
        points, d1, _ = source.evaluate(s)
        h = 1e-6
        central = (source.evaluate(s + h)[0] - source.evaluate(s - h)[0]) / (2 * h)
        assert np.max(np.hypot(*(central - d1).T)) < 1e-8
        for transition in source.transitions:
            boundary = reference.piece_bounds[transition.junction.index + 1, 0]
            _, a1, a2 = source.evaluate(np.array([boundary - 1e-11, boundary]))
            kappa = g.signed_curvature(a1, a2)
            assert abs(kappa[1] - kappa[0]) < 5e-5, transition.junction.label


def test_lsei_matches_a_general_solver():
    """The equality- and bound-constrained least squares behind the LE fair, against SLSQP on
    random problems: same optimum (to 1e-9 relative), equalities to 1e-12, bounds kept; an
    empty feasible set gives None."""
    from scipy.optimize import minimize
    rng = np.random.default_rng(7)
    for _ in range(4):
        A, b = rng.normal(size=(40, 12)), rng.normal(size=40)
        E = rng.random((3, 12))
        f = E @ np.abs(rng.normal(size=12))
        x = g._lsei(A, b, E, f)
        reference = minimize(lambda v: np.sum((A @ v - b) ** 2), np.ones(12), method="SLSQP",
                             constraints=[{"type": "eq", "fun": lambda v: E @ v - f}], bounds=[(0, None)] * 12,
                             options=dict(ftol=1e-15, maxiter=1000))
        assert np.sum((A @ x - b) ** 2) == pytest.approx(reference.fun, rel=1e-9)
        assert np.max(np.abs(E @ x - f)) < 1e-12 and np.min(x) > -1e-12
    assert g._lsei(np.eye(2), np.zeros(2), np.array([[1.0, 1.0]]), np.array([-1.0])) is None
