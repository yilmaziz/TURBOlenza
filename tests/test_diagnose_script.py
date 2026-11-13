"""
Tests for scripts/diagnose_mach060_vs_065.py: its helpers only (no solve is run here).

- The matched baseline: two specs that differ in the Mach number give CaseSettings that differ
  in the Mach number (and the name) only.
- A what-if variant is patched into the process and ALWAYS taken out again (a pooled worker
  runs other cases afterwards: a leaked patch once contaminated later rows), and
  check_production refuses to run a case while a patch is in place.
- The same-side differences of the variants are the exact one-sided three-point formulas (a
  quadratic in m is differentiated exactly at f - 1, f, l, l + 1; elsewhere the production
  result is kept).
- The velocity what-if V (and Vle) is production since the 2026-11-13 fix of flow.py (psi_m by
  m region): patched on top of production it changes no psi_m, psi_eta bit (the acceptance
  runs of scripts/verify_le_velocity_fix.py reproduce the what-if V runs bit for bit).
- late_ratio returns the ratio of a geometric sequence.
"""

from __future__ import annotations

import importlib.util
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from potential_flow_solver import boundary_conditions as bc
from potential_flow_solver import coefficients, flow
from potential_flow_solver import grid as gr

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def script():
    spec = importlib.util.spec_from_file_location("diagnose_mach060_vs_065", ROOT / "scripts" / "diagnose_mach060_vs_065.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_matched_cases_differ_in_the_mach_number_only(script):
    low = asdict(script.settings_of(script.spec("low", script.MACH_LOW)))
    high = asdict(script.settings_of(script.spec("high", script.MACH_HIGH)))
    assert {k for k in low if low[k] != high[k]} == {"case_name", "inlet_mach"}
    assert low["stop_at_sonic"] is False and low["solver_mode"] == "coupled_kutta"
    assert (low["n_passage"], low["n_eta"], low["density_relaxation"], low["kutta_relaxation"]) == (61, 41, 0.7, 0.7)
    fixed = asdict(script.settings_of(script.spec("fixed", 0.65, mode="fixed_exit", exit_deg=22.0, lam_rho=0.3, passes=7)))
    assert fixed["fixed_exit_angle_deg"] == 22.0 and fixed["density_relaxation"] == 0.3 and fixed["max_density_iterations"] == 7
    with pytest.raises(AssertionError):
        script.spec("bad", 0.6, relaxation=0.5)


@pytest.mark.parametrize("kind", ["V", "C", "Vle", "Cle", "V+Cle"])
def test_a_what_if_patch_is_always_taken_out_again(script, kind):
    production = (flow._derivatives, coefficients._d_dm)
    script.check_production()
    restore = script.apply_what_if(kind, 10, 20)
    assert (flow._derivatives, coefficients._d_dm) != production
    with pytest.raises(RuntimeError, match="patch leaked"):
        script.check_production()
    restore()
    assert (flow._derivatives, coefficients._d_dm) == production
    script.check_production()
    script.apply_what_if(None, 10, 20)()                        # no variant: nothing changes
    assert (flow._derivatives, coefficients._d_dm) == production


def test_same_side_differences_are_exact_for_a_quadratic(script):
    n_m, n_eta, dm, first, last = 30, 5, 0.1, 10, 20
    m = dm * np.arange(n_m)[:, None] * np.ones((1, n_eta))
    field = 2.0 + 3.0 * m - 0.7 * m ** 2
    field[first:] += 5.0                                         # a jump between f - 1 and f, as B and C have
    field[last + 1:] -= 2.0                                      # and between l and l + 1
    exact = 3.0 - 1.4 * m
    production = coefficients._d_dm(field, dm)
    assert abs(production[first, 2] - exact[first, 2]) > 10.0    # the central difference sees the jump
    for kind, stations in (("C", (first - 1, first, last, last + 1)), ("Cle", (first - 1, first))):
        restore = script.apply_what_if(kind, first, last)
        try:
            patched = coefficients._d_dm(field, dm)
        finally:
            restore()
        assert np.allclose(patched[list(stations)], exact[list(stations)], rtol=0.0, atol=1e-11)
        others = [i for i in range(n_m) if i not in stations]
        assert np.array_equal(patched[others], production[others])


@pytest.mark.parametrize("kind", ["V", "Vle"])
def test_the_velocity_what_if_is_now_production(script, kind):
    """flow.py takes psi_m by m region (one-sided at f - 1, f, l, l + 1, walls zero) since
    2026-11-13: the what-if V / Vle patch, applied on top of it, reproduces it bit for bit."""
    n_m, n_eta = 41, 21
    m, eta = np.linspace(0.0, 3.0, n_m), np.linspace(0.0, 1.0, n_eta)
    first, last = 16, 24
    region = np.full(n_m, gr.PASSAGE)
    region[:first], region[last + 1:] = gr.INLET, gr.EXIT
    bcs = bc._boundary_conditions(None, m, eta, first, last, np.arange(first, last + 1), region)
    M, H = np.meshgrid(m, eta, indexing="ij")
    psi = H + 0.3 * M ** 3 - 0.2 * M ** 2 * H + 0.1 * M * np.sin(6.0 * H)
    psi[first:] += 0.4                                           # region jumps, as two grid maps give
    psi[last + 1:] -= 0.25
    dm, deta = float(m[1] - m[0]), float(eta[1] - eta[0])
    production = flow._derivatives(psi, dm, deta, bcs)
    restore = script.apply_what_if(kind, first, last)
    try:
        patched = flow._derivatives(psi, dm, deta, bcs)
    finally:
        restore()
    assert np.array_equal(patched[0], production[0]) and np.array_equal(patched[1], production[1])


def test_late_ratio(script):
    values = 3.0 * 0.9 ** np.arange(60)
    assert script.late_ratio(values) == pytest.approx(0.9, rel=1e-12)
    assert script.late_ratio(np.array([1.0, np.nan, 0.5])) is None
