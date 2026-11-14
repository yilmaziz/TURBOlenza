"""
Tests for the Kutta study options (kutta.py section 7, config.IterationSettings.kutta_eval and
friends, solver._kutta_step), added 2026-11-14 for scripts/kutta_study.py.

What is checked
- The defaults are the production rule: kutta_eval "station"; wall_values + residual_from_walls
  at the station reproduce kutta_residual bit for bit on a solved flow.
- "fixed_m" interpolates each wall's speed in m: exact (to round-off) for a cubic in m with the
  cubic, for a linear one with the linear; at m* on a station both return that station's value
  to round-off; the four cubic nodes bracket m* (two each side) where the run allows and shift
  inward at the run's end; m* outside the blade wall nodes is refused, never extrapolated; the
  SS / PS identity is the TE station's.
- The solver refuses unknown option values before any pass, and a coupled solve with "fixed_m"
  drives the interpolated residual below the Kutta tolerance.

- kutta_method: "current" is the default; "aungier_13_57" (the brief's formula, written
  separately) equals exit_angle_correction to round-off and has its fixed point; alpha above 1
  is taken by the Aungier path only; Eq. 13-58 by hand; "aungier_13_57_58" drives the averaged
  speeds.

Tolerances: polynomial reproduction 1e-11 relative (measured 4.4e-16 at most over 400 offsets
along the blade); the coupled solve's final eps_K at m* below the case's kutta_tolerance by
construction of the stop test; brief against production correction 1e-12 relative (measured
1.0e-15); converged angles: 1e-6 deg between "current" and "aungier_13_57" at relaxation 1
(measured 7.6e-13 deg, both 115 passes), 1e-5 deg against alpha 1.5 (measured 7.5e-7 deg, 77
passes: the Kutta tolerance 1e-6 of this case stops the two at different distances from the
root).
"""

from __future__ import annotations

import math
from dataclasses import replace
from functools import lru_cache
from types import SimpleNamespace

import numpy as np
import pytest

from turbolenza.b2b import config as cf
from turbolenza.b2b import eos, kutta
from turbolenza.b2b import solver as sv
from turbolenza.b2b.geometry import CascadeSpec, make_cascade
from turbolenza.b2b.grid import make_grid

AIR = eos.CaloricallyPerfectGas(gamma=1.4, gas_constant=287.05)
PT, TT, C_REF, INLET_DEG = 101325.0, 288.15, 0.1, 45.0


@lru_cache(maxsize=None)
def cascade():
    return make_cascade(CascadeSpec(blade_type="NACA65", solidity=1.0, stagger_deg=33.0, cl0=1.2, t_max_over_c=0.10,
                                    surface_continuity="C2"))


def config(**iteration) -> cf.SolverConfig:
    it = cf.IterationSettings(density_tolerance=1e-8, max_density_iterations=400, kutta_tolerance=1e-6,
                              max_kutta_iterations=400)
    return cf.SolverConfig(
        grid=cf.GridSettings(n_passage=43, n_eta=21),
        flow=cf.FlowConditions(inlet_mach=0.3, inlet_angle_deg=INLET_DEG, exit_angle_guess_deg=math.degrees(cascade().kappa_te),
                               inlet_absolute_total_pressure=PT, inlet_absolute_total_temperature=TT, angular_speed=0.0),
        gas=AIR, iteration=replace(it, **iteration))


@lru_cache(maxsize=None)
def grid():
    return make_grid(cascade(), config().grid, math.radians(INLET_DEG), cascade().kappa_te, require_ss_ps=True)


def polynomial_flow(g, coefficients0, coefficients1) -> SimpleNamespace:
    """W_theta = 0 and W_m = a polynomial in m on each wall (positive on the blade), zero elsewhere."""
    shape = (len(g.m), len(g.eta))
    W_m, W_theta = np.zeros(shape), np.zeros(shape)
    W_m[:, 0] = np.polyval(coefficients0, g.m)
    W_m[:, -1] = np.polyval(coefficients1, g.m)
    return SimpleNamespace(W_m=W_m, W_theta=W_theta)


CUBIC0, CUBIC1 = (30.0, -50.0, 20.0, 120.0), (-25.0, 40.0, -10.0, 130.0)
LINEAR0, LINEAR1 = (15.0, 120.0), (-12.0, 135.0)


def test_the_defaults_are_the_production_rule():
    it = cf.IterationSettings(density_tolerance=1e-8, max_density_iterations=1, kutta_tolerance=1e-8,
                              max_kutta_iterations=1)
    assert (it.kutta_eval, it.kutta_eval_offset_chords, it.kutta_eval_interpolation) == ("station", None, "cubic")
    assert kutta.KUTTA_EVALUATIONS == ("station", "fixed_m") and kutta.KUTTA_INTERPOLATIONS == ("cubic", "linear")


@lru_cache(maxsize=None)
def solved_flow():
    """A converged fixed-angle flow of the verification case (no Kutta correction)."""
    return sv.solve_density_coupled(cascade(), config(), c_ref=C_REF)


def test_station_values_are_kutta_residual_bit_for_bit():
    solution = solved_flow()
    expected = kutta.kutta_residual(solution.flow, solution.grid)
    walls = kutta.wall_values(solution.flow, solution.grid)
    assert walls.evaluation == "station" and walls.stations == (solution.grid.last_blade_i,)
    assert kutta.residual_from_walls(walls, solution.grid) == expected


@pytest.mark.parametrize("offset", (0.012, 0.03, 0.05, 0.31, 0.77))
def test_the_cubic_reproduces_a_cubic_and_the_linear_a_line(offset):
    g = grid()
    m_star = g.m_te - offset * g.chord
    cubic = kutta.wall_values(polynomial_flow(g, CUBIC0, CUBIC1), g, "fixed_m", offset, "cubic")
    linear = kutta.wall_values(polynomial_flow(g, LINEAR0, LINEAR1), g, "fixed_m", offset, "linear")
    assert cubic.m == pytest.approx(m_star, abs=1e-15) and cubic.interpolation == "cubic"
    for value, coefficients in ((cubic.W0, CUBIC0), (cubic.W1, CUBIC1), (cubic.W_m0, CUBIC0), (cubic.W_m1, CUBIC1),
                                (linear.W0, LINEAR0), (linear.W1, LINEAR1)):
        assert value == pytest.approx(np.polyval(coefficients, m_star), rel=1e-11)
    # the linear interpolation of the cubic is not exact: the two are told apart
    rough = kutta.wall_values(polynomial_flow(g, CUBIC0, CUBIC1), g, "fixed_m", offset, "linear")
    assert abs(rough.W0 - np.polyval(CUBIC0, m_star)) > 1e-9


def test_the_cubic_nodes_bracket_m_star_and_shift_inward_at_the_run_end():
    g = grid()
    f, l = g.first_blade_i, g.last_blade_i
    middle = kutta.wall_values(polynomial_flow(g, CUBIC0, CUBIC1), g, "fixed_m", 0.31, "cubic")
    p = middle.stations[1]
    assert len(middle.stations) == 4 and g.m[p] <= middle.m <= g.m[p + 1] and middle.stations == tuple(range(p - 1, p + 3))
    end = kutta.wall_values(polynomial_flow(g, CUBIC0, CUBIC1), g, "fixed_m",
                            (g.m_te - 0.5 * (g.m[l - 1] + g.m[l])) / g.chord, "cubic")
    assert end.stations == (l - 3, l - 2, l - 1, l)
    assert all(f <= k <= l for k in middle.stations + end.stations)


def test_at_a_station_both_interpolations_return_its_value():
    solution = solved_flow()
    g = solution.grid
    k = g.last_blade_i - 3
    offset = (g.m_te - g.m[k]) / g.chord
    speed = np.hypot(solution.flow.W_m[k], solution.flow.W_theta[k])
    for interpolation in kutta.KUTTA_INTERPOLATIONS:
        walls = kutta.wall_values(solution.flow, g, "fixed_m", offset, interpolation)
        assert walls.W0 == pytest.approx(speed[0], rel=1e-12) and walls.W1 == pytest.approx(speed[-1], rel=1e-12)


def test_outside_the_blade_nodes_is_refused_never_extrapolated():
    g = grid()
    flow = polynomial_flow(g, CUBIC0, CUBIC1)
    behind_last = 0.999 * (g.m_te - g.m[g.last_blade_i]) / g.chord          # m* past the last blade station
    ahead_of_first = 1.001 * (g.m_te - g.m[g.first_blade_i]) / g.chord
    for offset in (behind_last, ahead_of_first, -0.01):
        with pytest.raises(kutta.KuttaError, match="outside the blade wall nodes"):
            kutta.wall_values(flow, g, "fixed_m", offset, "cubic")
    for bad in (None, math.nan, True):
        with pytest.raises(kutta.KuttaError, match="finite offset"):
            kutta.wall_values(flow, g, "fixed_m", bad, "cubic")
    with pytest.raises(kutta.KuttaError, match="interpolation"):
        kutta.wall_values(flow, g, "fixed_m", 0.05, "quadratic")
    with pytest.raises(kutta.KuttaError, match="evaluation"):
        kutta.wall_values(flow, g, "nowhere")


def test_fixed_m_keeps_the_te_stations_ss_ps_identity():
    g = grid()
    walls = kutta.wall_values(polynomial_flow(g, CUBIC0, CUBIC1), g, "fixed_m", 0.05, "cubic")
    residual = kutta.residual_from_walls(walls, g)
    station = kutta.trailing_edge_station(g)
    assert residual.station == station
    assert residual.wall_difference == walls.W0 - walls.W1
    assert residual.residual == pytest.approx((walls.W0 - walls.W1) * (1 if station.ss_j == 0 else -1), rel=1e-15)


@pytest.mark.parametrize("changes, match", (
    (dict(kutta_eval="nowhere"), "kutta_eval"),
    (dict(kutta_eval="fixed_m"), "kutta_eval_offset_chords"),
    (dict(kutta_eval="fixed_m", kutta_eval_offset_chords=0.03, kutta_eval_interpolation="spline"), "interpolation"),
))
def test_the_solver_refuses_bad_options_before_any_pass(changes, match):
    with pytest.raises(sv.SolverInputError, match=match):
        sv.solve(cascade(), config(**changes), c_ref=C_REF)


def test_a_fixed_m_coupled_solve_drives_the_interpolated_residual_down():
    solution = sv.solve(cascade(), config(kutta_eval="fixed_m", kutta_eval_offset_chords=0.05), c_ref=C_REF)
    assert solution.converged
    walls = kutta.wall_values(solution.flow, solution.grid, "fixed_m", 0.05, "cubic")
    assert kutta.residual_from_walls(walls, solution.grid).normalized <= 1e-6
    assert solution.normalized_kutta_residual <= 1e-6
    station = kutta.kutta_residual(solution.flow, solution.grid)
    assert station.normalized > 1e-6        # the station's own Eq. 5-34 is not what was driven to zero


# --- kutta_method (kutta.py section 8) ---------------------------------------------------------------------------

def test_the_method_default_is_current():
    it = cf.IterationSettings(density_tolerance=1e-8, max_density_iterations=1, kutta_tolerance=1e-8,
                              max_kutta_iterations=1)
    assert it.kutta_method == "current"
    assert kutta.KUTTA_METHODS == ("current", "aungier_13_57", "aungier_13_57_58")


def test_the_brief_formula_equals_the_production_correction_to_round_off():
    solution = solved_flow()
    residual = kutta.kutta_residual(solution.flow, solution.grid)
    ours = kutta.exit_angle_correction(residual, solution.flow, solution.metrics)
    brief = kutta.aungier_exit_angle_correction(residual, solution.flow, solution.metrics)
    assert brief.delta_tan == pytest.approx(ours.delta_tan, rel=1e-12)
    assert brief.delta_c_theta == pytest.approx(ours.delta_c_theta, rel=1e-12)
    assert (brief.dm, brief.passage_width) == pytest.approx((ours.dm, ours.passage_width), rel=1e-12)


def test_eq_13_58_by_hand():
    solution = solved_flow()
    g, metrics = solution.grid, solution.metrics
    K, J = g.last_blade_i, len(g.eta) - 1
    walls = kutta.wall_values(solution.flow, g)
    averaged = kutta.inside_outside_average(walls, metrics)
    cos0 = math.cos(math.atan(float(metrics.tan_beta0[K])))
    cos1 = math.cos(math.atan(float(metrics.tan_beta1[K])))
    W_m = np.asarray(solution.flow.W_m)
    assert averaged.W0 == pytest.approx(0.5 * (walls.W0 + 2.0 * W_m[K, 0] / (cos0 + cos1)), rel=1e-14)
    assert averaged.W1 == pytest.approx(0.5 * (walls.W1 + 2.0 * W_m[K, J] / (cos0 + cos1)), rel=1e-14)
    assert (averaged.W_m0, averaged.W_m1) == (walls.W_m0, walls.W_m1)


def test_the_relaxed_update_takes_alpha_above_one_and_refuses_nonsense():
    correction = SimpleNamespace(delta_tan=0.02)
    angle = 0.4
    assert kutta.relaxed_exit_angle(angle, correction, 1.5) == pytest.approx(math.atan(math.tan(angle) + 0.03), rel=1e-15)
    assert kutta.relaxed_exit_angle(angle, SimpleNamespace(delta_tan=0.0), 3.0) == angle
    for bad in (0.0, -1.0, math.inf, math.nan, True):
        with pytest.raises(kutta.KuttaError, match="alpha"):
            kutta.relaxed_exit_angle(angle, correction, bad)
    with pytest.raises(kutta.KuttaError, match="relaxation"):
        kutta.next_exit_angle(angle, correction, 1.5)          # "current" keeps lambda_K <= 1


def test_the_solver_refuses_an_unknown_method():
    with pytest.raises(sv.SolverInputError, match="kutta_method"):
        sv.solve(cascade(), config(kutta_method="newton"), c_ref=C_REF)


def test_aungier_13_57_has_the_fixed_point_of_current():
    current = sv.solve(cascade(), config(kutta_relaxation=1.0), c_ref=C_REF)
    brief = sv.solve(cascade(), config(kutta_method="aungier_13_57", kutta_relaxation=1.0), c_ref=C_REF)
    faster = sv.solve(cascade(), config(kutta_method="aungier_13_57", kutta_relaxation=1.5), c_ref=C_REF)
    assert current.converged and brief.converged and faster.converged
    assert math.degrees(abs(brief.exit_angle - current.exit_angle)) < 1e-6
    assert math.degrees(abs(faster.exit_angle - current.exit_angle)) < 1e-5
    assert len(faster.passes) < len(brief.passes)


def test_aungier_13_58_drives_the_averaged_speeds_not_eq_5_34():
    solution = sv.solve(cascade(), config(kutta_method="aungier_13_57_58"), c_ref=C_REF)
    assert solution.converged
    averaged = kutta.inside_outside_average(kutta.wall_values(solution.flow, solution.grid), solution.metrics)
    assert kutta.residual_from_walls(averaged, solution.grid).normalized <= 1e-6
    assert solution.normalized_kutta_residual <= 1e-6
