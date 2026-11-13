"""
Tests for the study option transition_form (discretization.py "Study option: face-flux rows at the
transitions", matrix_assembly.assemble_from_coefficients, config.IterationSettings, solver.py),
added 2026-11-14 for scripts/transition_study.py.

What is checked
- The default is "expanded" and the default assembly is the production system bit for bit.
- "flux" changes the rows of the four transition stations (j = 1 .. n_eta - 2) and no other row,
  and nothing on the right-hand side.
- The face-flux row by hand: its weights applied to a psi equal the face fluxes of Eq. 5-20
  written out separately at one node of each transition station.
- With constant coefficients the face-flux weights equal the expanded weights (measured 6e-18
  relative on the study's very_fine shape; 1e-13 allowed).
- Unknown values are refused (assembly and solver).

Tolerances: hand formula 1e-12 relative (the same products in another order); constant
coefficients 1e-13 relative (measured 6.0e-18 in scripts/transition_study.py phaseC).
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
import pytest

from turbolenza.b2b import config as cf
from turbolenza.b2b import discretization as dz
from turbolenza.b2b import eos
from turbolenza.b2b import solver as sv
from turbolenza.b2b.boundary_conditions import build_boundary_conditions
from turbolenza.b2b.coefficients import build_coefficients
from turbolenza.b2b.geometry import CascadeSpec, make_cascade
from turbolenza.b2b.grid import make_grid
from turbolenza.b2b.matrix_assembly import MatrixAssemblyError, assemble_from_coefficients
from turbolenza.b2b.metrics import build_metrics

INLET = math.radians(45.0)


@lru_cache(maxsize=None)
def case():
    cascade = make_cascade(CascadeSpec(blade_type="NACA65", solidity=1.0, stagger_deg=33.0, cl0=1.2, t_max_over_c=0.10))
    grid = make_grid(cascade, cf.GridSettings(n_passage=43, n_eta=21), INLET, cascade.kappa_te, require_ss_ps=True)
    metrics, bcs = build_metrics(grid), build_boundary_conditions(grid)
    rho = 1.0 + 0.05 * np.sin(np.asarray(grid.m))[:, None] * np.ones((1, len(grid.eta)))
    co = build_coefficients(metrics, rho, 1.0, c_ref=None)
    return grid, metrics, bcs, co


def test_the_default_is_expanded_and_unchanged():
    it = cf.IterationSettings(density_tolerance=1e-8, max_density_iterations=1, kutta_tolerance=1e-8,
                              max_kutta_iterations=1)
    assert it.transition_form == "expanded" and dz.TRANSITION_FORMS == ("expanded", "flux")
    grid, metrics, bcs, co = case()
    default = assemble_from_coefficients(co, metrics, bcs)
    explicit = assemble_from_coefficients(co, metrics, bcs, transition_form="expanded")
    assert (default.matrix != explicit.matrix).nnz == 0
    np.testing.assert_array_equal(default.rhs, explicit.rhs)


def test_flux_changes_the_four_transition_stations_only():
    grid, metrics, bcs, co = case()
    expanded = assemble_from_coefficients(co, metrics, bcs)
    flux = assemble_from_coefficients(co, metrics, bcs, transition_form="flux")
    np.testing.assert_array_equal(expanded.rhs, flux.rhs)
    changed = np.flatnonzero(np.abs(flux.matrix - expanded.matrix).max(axis=1).toarray().ravel() > 0.0)
    n_eta = len(grid.eta)
    owners = {divmod(int(expanded.active_nodes[r]), n_eta) for r in changed}
    f, l = grid.first_blade_i, grid.last_blade_i
    allowed = {(i, j) for i in (f - 1, f, l, l + 1) for j in range(1, n_eta - 1)}
    assert owners and owners <= allowed
    assert {i for i, _ in owners} == {f - 1, f, l, l + 1}


@pytest.mark.parametrize("which", ("f-1", "f", "l", "l+1"))
def test_the_flux_row_by_hand(which):
    grid, metrics, bcs, co = case()
    f, l = grid.first_blade_i, grid.last_blade_i
    i = {"f-1": f - 1, "f": f, "l": l, "l+1": l + 1}[which]
    j = len(grid.eta) // 2
    A, B, C = (np.asarray(getattr(co, k)) for k in "ABC")
    h, k = grid.deta, co.length_scale * grid.dm
    rng = np.random.default_rng(3)
    psi = rng.normal(size=A.shape)
    mean = lambda a, p, q: 0.5 * (a[p] + a[q])  # noqa: E731
    psi_m = lambda ii, jj: (psi[ii + 1, jj] - psi[ii - 1, jj]) / (2 * k)  # noqa: E731
    psi_e = lambda ii, jj: (psi[ii, jj + 1] - psi[ii, jj - 1]) / (2 * h)  # noqa: E731
    F_up = mean(A, (i, j), (i, j + 1)) * (psi[i, j + 1] - psi[i, j]) / h - mean(B, (i, j), (i, j + 1)) * 0.5 * (
        psi_m(i, j) + psi_m(i, j + 1))
    F_dn = mean(A, (i, j), (i, j - 1)) * (psi[i, j] - psi[i, j - 1]) / h - mean(B, (i, j), (i, j - 1)) * 0.5 * (
        psi_m(i, j) + psi_m(i, j - 1))
    G_rt = mean(B, (i, j), (i + 1, j)) * 0.5 * (psi_e(i, j) + psi_e(i + 1, j)) - mean(C, (i, j), (i + 1, j)) * (
        psi[i + 1, j] - psi[i, j]) / k
    G_lt = mean(B, (i, j), (i - 1, j)) * 0.5 * (psi_e(i, j) + psi_e(i - 1, j)) - mean(C, (i, j), (i - 1, j)) * (
        psi[i, j] - psi[i - 1, j]) / k
    by_hand = (F_up - F_dn) / h - (G_rt - G_lt) / k
    weights, rows = dz.transition_flux_weights(co, metrics)
    assert (i, j) in rows
    H = 2 * A[i, j] / h ** 2 + 2 * C[i, j] / k ** 2
    raw = -H * float(np.sum(weights[i, j] * psi[i - 1:i + 2, j - 1:j + 2]))
    assert raw == pytest.approx(by_hand, rel=1e-12)


def test_constant_coefficients_give_the_expanded_weights():
    shape = (40, 12)
    A, B, C = np.full(shape, 2.0), np.full(shape, 0.7), np.full(shape, 1.3)
    Z = np.zeros(shape)
    h, k = 0.05, 0.02
    expanded = dz._interior_stencil(A, B, C, Z, Z, k, h)
    flux, rows = dz._flux_transition_weights(A, B, C, k, h, 10, 30, np.asarray(expanded.H))
    reference = np.asarray(expanded.weights)
    for i, j in rows:
        np.testing.assert_allclose(flux[i, j], reference[i, j], rtol=0.0, atol=1e-13 * np.max(np.abs(reference[i, j])))


def test_unknown_forms_are_refused():
    grid, metrics, bcs, co = case()
    with pytest.raises(MatrixAssemblyError, match="transition_form"):
        assemble_from_coefficients(co, metrics, bcs, transition_form="conservative")
    cascade = make_cascade(CascadeSpec(blade_type="NACA65", solidity=1.0, stagger_deg=33.0, cl0=1.2, t_max_over_c=0.10))
    config = cf.SolverConfig(
        grid=cf.GridSettings(n_passage=43, n_eta=21),
        flow=cf.FlowConditions(inlet_mach=0.3, inlet_angle_deg=45.0, exit_angle_guess_deg=math.degrees(cascade.kappa_te),
                               inlet_absolute_total_pressure=101325.0, inlet_absolute_total_temperature=288.15,
                               angular_speed=0.0),
        gas=eos.CaloricallyPerfectGas(gamma=1.4, gas_constant=287.05),
        iteration=cf.IterationSettings(density_tolerance=1e-8, max_density_iterations=10, kutta_tolerance=1e-6,
                                       max_kutta_iterations=10, transition_form="conservative"))
    with pytest.raises(sv.SolverInputError, match="transition_form"):
        sv.solve(cascade, config, c_ref=0.1)
