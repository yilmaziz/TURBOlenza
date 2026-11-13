"""
Tests for scripts/audit_naca_thickness_reconstruction.py, the DIAGNOSTIC audit of how the NACA
65 thickness table is reconstructed between its stations. Its A_pchip and C_quintic_natural
are the production reconstructions since 2026-11-13 ("pchip_sqrt", legacy; "quintic_natural_sqrt",
the C2 default) and are built through CascadeSpec; a diagnostic candidate (B, D) is registered
in geometry.py only inside thickness_reconstruction().

- Every candidate is finite and positive with one maximum; its table errors are measured: the
  production PCHIP, the cubic splines and the quintic pass through every ordinate (round-off),
  the smoothing spline stays within the table's printed precision (5e-6 c) and is not exact.
- Maximum thickness and its location are measured (PCHIP 0.09992 at station 0.40, the table's
  own maximum; the C2 interpolants 0.09996 near station 0.41, where the quadratic through the
  three largest ordinates puts it).
- knot_continuity sees the jumps it should (PCHIP's second derivative, the cubic's third) and
  measures a constructed jump exactly; curve_continuity finds the final quintic curves C2 to
  round-off.
- The ripple metric is reproducible and recovers a ripple added to a smooth reference.
- thickness_reconstruction() leaves the module alone for a production reconstruction and
  restores the production state and clears the caches after a diagnostic one, also after an
  error; A_pchip and C_quintic_natural are the "pchip_sqrt" and default C2 geometry bit for
  bit; a diagnostic candidate build and solve leave the production geometry unchanged.
"""

from __future__ import annotations

import importlib.util
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy.interpolate import PPoly

from turbolenza.b2b import geometry as g

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def audit():
    spec = importlib.util.spec_from_file_location("audit_naca_thickness_reconstruction",
                                                  ROOT / "scripts" / "audit_naca_thickness_reconstruction.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def production_spec(variant="emery_base", **kw):
    return g.CascadeSpec(blade_type="NACA65", solidity=1.0, stagger_deg=33.0, cl0=1.2, t_max_over_c=0.10,
                         profile_variant=variant, surface_continuity="C2", **kw)


def build(spec):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return g.make_cascade(spec)


@pytest.fixture(scope="module")
def production_cascade():
    return build(production_spec())                     # the default C2: quintic_natural_sqrt


def same_curves(a, b) -> bool:
    return all(np.array_equal(getattr(a.active_curves, n).control_points, getattr(b.active_curves, n).control_points)
               and np.array_equal(getattr(a.active_curves, n).knots, getattr(b.active_curves, n).knots)
               for n in ("suction", "pressure"))


def test_candidates_are_finite_positive_and_their_table_errors_are_measured(audit):
    for name in audit.CANDIDATES:
        rec = audit.candidate(name)
        checks = audit.thickness_checks(name, rec)
        s, t, t1, t2 = audit.thickness_profile(rec)
        assert np.all(np.isfinite(t)) and np.all(np.isfinite(t1)) and np.all(np.isfinite(t2[s > 1e-3])), name
        assert checks["positive"] and checks["single_maximum"] and checks["wrong_slope_samples_in_monotone_intervals"] == 0
        assert np.all(t[(s > 1e-4) & (s < 1.0 - 1e-4)] > 0.0)
        assert checks["errors"].shape == (26,)
        if name == "D_smoothing_cubic":
            assert 1e-7 < checks["max_abs_table_error"] <= audit.TABLE_TOLERANCE * (1 + 1e-9)    # measured 5.00e-6
            assert not checks["table_exact"]
        else:
            assert checks["max_abs_table_error"] < 1e-15 and checks["table_exact"]             # measured <= 1.4e-17


def test_maximum_thickness_and_location_are_measured(audit):
    pchip = audit.thickness_checks("A_pchip", audit.candidate("A_pchip"))
    assert pchip["t_max"] == pytest.approx(0.09992, abs=1e-9) and pchip["x_t_max"] == pytest.approx(0.40, abs=1e-4)
    assert pchip["three_point_quadratic_peak"] == pytest.approx(0.099956, abs=1e-6)
    for name in ("B_cubic_not_a_knot", "C_quintic_natural"):
        c = audit.thickness_checks(name, audit.candidate(name))
        assert 0.09995 < c["t_max"] < 0.09997 and 0.405 < c["x_t_max"] < 0.415       # measured 0.099961 .. 0.099963, 0.410


def test_knot_continuity_sees_the_jumps_it_should(audit):
    pchip = audit.knot_continuity(audit.candidate("A_pchip"))
    assert pchip[0] < 1e-15 and pchip[1] < 1e-15 and pchip[2] > 1.0                   # C1: h'' jumps (measured 4.6)
    cubic = audit.knot_continuity(audit.candidate("B_cubic_not_a_knot"))
    assert cubic[2] < 1e-12 and cubic[3] > 1.0                                        # C2: h''' jumps
    quintic = audit.knot_continuity(audit.candidate("C_quintic_natural"))
    assert quintic[2] < 1e-12 and quintic[3] < 1e-12                                  # C4
    # x^2 + 2 x + 1 on [0, 0.5], continued with the same value 2.25 and slope 3 but c2 = 3.5: h'' jumps 2 -> 7
    jump = PPoly(np.array([[0.0, 0.0], [1.0, 3.5], [2.0, 3.0], [1.0, 2.25]]), np.array([0.0, 0.5, 1.0]))
    measured = audit.knot_continuity(jump)
    assert measured[0] < 1e-15 and measured[1] < 1e-15 and measured[2] == pytest.approx(5.0, abs=1e-12)


def test_the_ripple_metric_is_reproducible_and_recovers_a_known_ripple(audit):
    x = np.linspace(0.0, 1.0, 20001)
    smooth = 0.3 - 1.2 * x + 0.8 * x ** 2 - 0.1 * x ** 5
    wave = 0.05 * np.sin(2.0 * np.pi * x / 0.05)
    ref = audit.trend_fit(x, smooth)
    a = audit.ripple(x, smooth + wave, ref)
    b = audit.ripple(x, smooth + wave, audit.trend_fit(x, smooth))
    assert np.array_equal(a, b, equal_nan=True)
    inside = np.isfinite(a)
    assert np.max(np.abs(a[inside] - wave[inside])) < 1e-9


def test_the_context_restores_production_state_also_after_an_error(audit):
    production = audit.production_state()
    for name, reconstruction in audit.PRODUCTION_NAMES.items():       # production: the module is not touched
        with audit.thickness_reconstruction(name) as key:
            assert key == reconstruction and audit.production_state() == production
    assert audit.candidate("A_pchip") is g._HALF_THICKNESS["pchip_sqrt"]
    assert audit.candidate("C_quintic_natural") is g._HALF_THICKNESS["quintic_natural_sqrt"]
    with audit.thickness_reconstruction("B_cubic_not_a_knot") as key:
        assert key == "audit_B_cubic_not_a_knot" and key in g.THICKNESS_RECONSTRUCTIONS
        assert key in g._HALF_THICKNESS and key in g._RECONSTRUCTION_MAXIMUM
    assert audit.production_state() == production
    with pytest.raises(RuntimeError):
        with audit.thickness_reconstruction("B_cubic_not_a_knot"):
            raise RuntimeError("inside")
    assert audit.production_state() == production
    assert g._naca65_reference_surfaces.cache_info().currsize == 0
    assert all(name in g.THICKNESS_RECONSTRUCTIONS for name in audit.PRODUCTION_NAMES.values())


def test_the_production_candidates_are_the_production_geometries(audit, production_cascade):
    assert same_curves(audit.build_cascade("C_quintic_natural", "emery_base"), production_cascade)
    assert same_curves(audit.build_cascade("A_pchip", "emery_base"),
                       build(production_spec(thickness_reconstruction="pchip_sqrt")))
    assert not same_curves(audit.build_cascade("A_pchip", "emery_base"), production_cascade)


def test_a_diagnostic_candidate_geometry_is_valid_c2_and_leaves_production_unchanged(audit, production_cascade):
    cascade = audit.build_cascade("B_cubic_not_a_knot", "emery_base")           # make_cascade validates it
    for name in ("suction_unstaggered", "pressure_unstaggered"):
        jumps = audit.curve_continuity(getattr(cascade.normalized_curves, name))
        assert jumps["X"] < 1e-12 and jumps["X'"] < 1e-12 and jumps["kappa"] < 1e-10   # measured <= 4.5e-13
    assert cascade.metadata["thickness_reconstruction"] == "audit_B_cubic_not_a_knot"
    assert not same_curves(cascade, production_cascade)
    assert same_curves(build(production_spec()), production_cascade)


def test_a_diagnostic_candidate_solve_leaves_the_production_geometry_unchanged(audit, production_cascade):
    production = audit.production_state()
    result = audit.solve_job(dict(candidate="B_cubic_not_a_knot", level="very_coarse"))
    assert result["converged"] and np.all(np.isfinite(result["mach_ss"]))
    assert audit.production_state() == production
    assert same_curves(build(production_spec()), production_cascade)
