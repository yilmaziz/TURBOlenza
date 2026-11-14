"""
Tests for scripts/sideline_anchoring_study.py, phase A: the side lines re-anchored at the nose tip
and the tail end in study-local grids (no src/ change). The script is loaded with importlib, as
tests/test_naca_thickness_reconstruction.py loads its audit script. Case: kutta_study.case_of
(NACA 65-(12A10)10, C2, kovach_sandercock, stagger 33 deg, solidity 1, inlet 45 deg), the
very_coarse level, and very_fine for the published delta values (about 0.5 s per level).

(a) The nose tip and the tail end from grid._turning_points(curve, 0) and the curve ends are the
    brentq roots of m'(u) on a dense u sample (another root finder, no PPoly, no bisection).
    Tolerance 1e-14 in u, m/c and y/c: measured 8.9e-16 at most (the tail's u and y; the tip
    3.1e-17), and brentq's own bound in u is xtol + 4 eps |u| = 1.9e-15. Also: no curve sample lies
    beyond them in m (round-off 1e-15 c), w_LE = 4.521e-3 c and w_TE = 1.352e-5 c (the plan's
    numbers; relative tolerance half a unit of the quoted last digit, 1.1e-4 and 3.7e-4: measured
    4.5206035e-3, relative 8.8e-5, and 1.352435e-5, relative 3.2e-4).
(b) The tip side lines at the inlet and exit stations are the adaptive quadrature of dy/dm =
    tan(beta(m)) from the tip / tail end (aungier_side of scripts/audit_aungier_side_boundaries.py,
    which owes grid.py nothing); the production side lines (the grid's own nodes) are the same
    quadrature from the LE / TE point (control). Tolerance 1e-14 c: measured 3.3e-16 c at most on
    very_coarse (4.4e-16 c on very_fine), and quad is asked for epsabs 1e-15, epsrel 1e-14.
(c) The position jump of the side line at the nose tip / tail end, side line minus blade point:
    with "tip" exactly 0 (measured 0.0 on every level: the fraction is 0 at the junction); with
    "le_point" -1.558e-2 c at the LE and +4.694e-4 c at the TE, relative tolerance half a unit of
    the quoted last digit (3.2e-4 and 1.07e-4): measured -1.5578225e-2 c (relative 1.1e-4;
    -1.557823e-2 to -1.557825e-2 over the five levels, whose m_in differ by up to 0.024 c) and
    4.6935379e-4 c (relative 9.8e-6, the same on every level: the exit law at exit angle kappa_te
    is the straight line).
(d) le_point max |delta|/c at f-1, f, l, l+1 is the transition study's published phase C table
    (outputs/transition_study/report.md, four decimals; hard-coded, so the test does not read the
    untracked outputs/). Tolerance half a unit of the fourth decimal, 5e-5: measured differences up
    to 4.85e-5 (very_coarse l: 0.0057485 against 0.0057).
(e) tip_grid: Grid.__post_init__ accepts it and still refuses a wrong blade run on it; m, eta, the
    blade run, the blade-station nodes and the wall identity are bit-identical to the production
    grid; its periodic-station nodes are the tip side lines, eta-linear and one pitch apart (bit for
    bit), and they moved (0.0156 c at f-1 on very_coarse, measured; bound 1e-3 c); its boundary
    samples end exactly on the tip / tail end at kappa_le / kappa_te; build_metrics builds on it.

Phase B (the operator on the exact uniform flow psi = (y - m tan gamma) / pitch, rho = 1, mdot' = 1):
(f) le_point, very_coarse, gamma 33 deg: max |raw residual| of f-1, f, l, l+1 is the transition
    study's published phase B table (outputs/transition_study/report.md: 19.5, 18.3, 5.15, 4.67;
    hard-coded), tolerance half a unit of the printed last digit (0.05, 0.05, 0.005, 0.005):
    measured 19.5354, 18.2937, 5.1464, 4.6656 (differences 0.035, 0.0063, 0.0036, 0.0044).
(g) tip, very_coarse, gamma 20, 33, 45 deg: the raw residual is finite at every interior row and
    every row class, and the term split adds up to the full residual: tolerance 1e-12 (raw units)
    on the interior and the periodic-line rows, measured 2.1e-13 and 2.3e-13 (the full residual
    reaches 11.3 there; 8.8e-12 over all five levels, residuals to 136).

Phase D (side checks):
(h) side_line_exact, the closed form of the side line that D2 compares the discrete ingredients with,
    at the very_coarse inlet stations, both anchors: its y0 is the grid's lower side-line node
    (tolerance 2e-14 c, measured 4.0e-15); central differences of its own y0 and tan(beta) at
    h = 0.05 c reproduce its tan(beta) and d tan(beta)/dm (tolerances 1e-5, 1e-6 and 2e-6: measured
    6.2e-6, 7.3e-7 and 1.45e-6) and are second order: halving h divides each error by 4.0000
    (bound 3.9 .. 4.1).

Seeded mutations (2026-11-14, each run once and reverted; recorded in the phase reports): phase A,
in anchored_side, the inlet junction y at the tip but its m left at m_le: (b) and (c) fail. Phase
B, in operator_level, psi built with the wrong sign of the m term, (y + m tan gamma) / pitch: (f)
fails (f-1 18.61 against 19.5; f 18.40, l 4.970, l+1 4.966 are outside their tolerances too), the
others pass. Phase D, in side_line_exact, d tan(beta)/dm with the wrong sign of dbeta/dm: (h) fails for
both anchors (d tan/dm from y0 off by 0.244), the others pass.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from turbolenza.b2b import grid as gr  # noqa: E402
from turbolenza.b2b.metrics import build_metrics  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

TOL_EXTREME = 1e-14         # (a) u, m/c, y/c: measured 8.9e-16 (tail u and y)
TOL_QUADRATURE = 1e-14      # (b) chords: measured 3.3e-16 on very_coarse, 4.4e-16 on very_fine
PUBLISHED_DELTA = {         # (d) outputs/transition_study/report.md, phase C table: f-1, f, l, l+1
    "very_coarse": (0.0390, 0.0202, 0.0057, 0.0116),
    "very_fine": (0.0310, 0.0196, 0.0071, 0.0083),
}
TOL_PUBLISHED = 5e-5        # (d) half a unit of the fourth decimal: measured 4.85e-5 at most
PUBLISHED_RESIDUAL = {      # (f) report.md phase B table, very_coarse, gamma 33: (printed value, half its last digit)
    "f-1": (19.5, 0.05),    # measured 19.5354
    "f": (18.3, 0.05),      # measured 18.2937
    "l": (5.15, 0.005),     # measured 5.1464
    "l+1": (4.67, 0.005),   # measured 4.6656
}
TOL_SPLIT = 1e-12           # (g) raw units: measured 2.1e-13 (interior rows), 2.3e-13 (periodic-line rows)
FD_STEP = 0.05              # (h) chords
TOL_Y0_CLOSED = 2e-14       # (h) chords: measured 4.0e-15 (le_point), 3.3e-15 (tip)
TOL_FD = (1e-5, 1e-6, 2e-6)  # (h) tan from y0, dtan from y0, dtan from tan at FD_STEP: measured 6.2e-6, 7.3e-7, 1.45e-6


@pytest.fixture(scope="module")
def study():
    spec = importlib.util.spec_from_file_location("sideline_anchoring_study",
                                                  ROOT / "scripts" / "sideline_anchoring_study.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def very_coarse(study):
    case = study.case_of("very_coarse")
    return case, study.tip_and_tail(case.cascade)


def test_tip_and_tail_are_the_brentq_roots_of_m_prime(study, very_coarse):
    case, ends = very_coarse
    c = case.grid.chord
    check = study.brentq_extremes(case.cascade)
    assert (ends["tip"]["curve"], ends["tail"]["curve"]) == ("suction", "pressure")
    for key in ("tip", "tail"):
        assert check[key]["curve"] == ends[key]["curve"]
        assert 0.0 < ends[key]["u"] < 1.0                            # an interior turning point, not a curve end
        assert abs(check[key]["u"] - ends[key]["u"]) <= TOL_EXTREME
        assert abs(check[key]["m"] - ends[key]["m"]) / c <= TOL_EXTREME
        assert abs(check[key]["y"] - ends[key]["y"]) / c <= TOL_EXTREME
        assert np.array_equal(ends[key]["point"], [ends[key]["m"], ends[key]["y"]])
    for name in ("suction", "pressure"):
        m = getattr(case.cascade.active_curves, name).evaluate(np.linspace(0.0, 1.0, 20001))[:, 0]
        assert m.min() >= ends["tip"]["m"] - 1e-15 * c and m.max() <= ends["tail"]["m"] + 1e-15 * c
    b = case.grid.boundaries
    assert abs((b.m_le - ends["tip"]["m"]) / c / 4.521e-3 - 1.0) <= 1.1e-4
    assert abs((ends["tail"]["m"] - b.m_te) / c / 1.352e-5 - 1.0) <= 3.7e-4


def test_side_lines_are_the_quadrature_of_the_beta_law(study, very_coarse):
    case, ends = very_coarse
    grid, cascade = case.grid, case.cascade
    b, c = grid.boundaries, grid.chord
    f, l = grid.first_blade_i, grid.last_blade_i
    m = np.asarray(grid.m)
    lines = study.tip_side_lines(grid, cascade, ends)
    assert np.array_equal(lines["inlet"]["m"], m[:f]) and np.array_equal(lines["exit"]["m"], m[l + 1:])
    k_le, k_te = float(cascade.kappa_le), float(cascade.kappa_te)
    tip = (ends["tip"]["m"], ends["tip"]["y"])
    tail = (ends["tail"]["m"], ends["tail"]["y"])
    quad_in = study.quadrature_side(tip, k_le, b.m_in, b.inlet_angle)(m[:f])
    quad_out = study.quadrature_side(tail, k_te, b.m_out, b.exit_angle)(m[l + 1:])
    assert np.max(np.abs(lines["inlet"]["y"] - quad_in)) / c <= TOL_QUADRATURE
    assert np.max(np.abs(lines["exit"]["y"] - quad_out)) / c <= TOL_QUADRATURE
    # the control: the production side lines (the grid's own lower nodes) from the LE / TE point
    y0 = np.asarray(grid.nodes)[:, 0, 1]
    quad_le = study.quadrature_side(cascade.active.leading_edge_point, k_le, b.m_in, b.inlet_angle)(m[:f])
    quad_te = study.quadrature_side(cascade.active.trailing_edge_point, k_te, b.m_out, b.exit_angle)(m[l + 1:])
    assert np.max(np.abs(y0[:f] - quad_le)) / c <= TOL_QUADRATURE
    assert np.max(np.abs(y0[l + 1:] - quad_te)) / c <= TOL_QUADRATURE


def test_position_jump_at_the_nose_tip_and_the_tail_end(study, very_coarse):
    case, ends = very_coarse
    grid, cascade = case.grid, case.cascade
    b, c = grid.boundaries, grid.chord
    tip, tail = ends["tip"], ends["tail"]
    assert study.anchored_side(grid, cascade, ends, "inlet", [tip["m"]])[1][0] - tip["y"] == 0.0
    assert study.anchored_side(grid, cascade, ends, "exit", [tail["m"]])[1][0] - tail["y"] == 0.0
    jump_le = (gr._inlet_side(b, np.array([tip["m"]]))[0] - tip["y"]) / c
    jump_te = (gr._exit_side(b, np.array([tail["m"]]))[0] - tail["y"]) / c
    assert abs(jump_le / -1.558e-2 - 1.0) <= 3.2e-4
    assert abs(jump_te / 4.694e-4 - 1.0) <= 1.07e-4


@pytest.mark.parametrize("level", sorted(PUBLISHED_DELTA))
def test_le_point_delta_is_the_transition_study_table(study, level):
    grid = study.case_of(level).grid
    delta = study.corner_delta(grid)
    for row, published in zip(study.ROWS, PUBLISHED_DELTA[level]):
        assert abs(delta[row]["max_abs"] / grid.chord - published) <= TOL_PUBLISHED, (level, row)


def test_the_tip_grid_changes_only_the_periodic_stations(study, very_coarse):
    case, ends = very_coarse
    grid, cascade = case.grid, case.cascade
    b, c = grid.boundaries, grid.chord
    f, l = grid.first_blade_i, grid.last_blade_i
    eta = np.asarray(grid.eta)
    tg = study.tip_grid(grid, cascade, ends)
    assert isinstance(tg, gr.Grid) and tg is not grid
    assert np.array_equal(tg.m, grid.m) and np.array_equal(tg.eta, grid.eta)
    assert (tg.first_blade_i, tg.last_blade_i) == (f, l)
    assert np.array_equal(tg.nodes[f:l + 1], grid.nodes[f:l + 1])
    for side in ("lower", "upper"):
        new, old = getattr(tg.wall_identity, side), getattr(grid.wall_identity, side)
        assert np.array_equal(new.surface, old.surface) and np.array_equal(new.u, old.u)
    lines = study.tip_side_lines(grid, cascade, ends)
    pitch = b.periodic_translation[1]
    for stations, side in ((slice(0, f), "inlet"), (slice(l + 1, None), "exit")):
        y = lines[side]["y"]
        assert np.array_equal(tg.nodes[stations, :, 0], grid.nodes[stations, :, 0])
        assert np.array_equal(tg.nodes[stations, 0, 1], y)
        assert np.array_equal(tg.nodes[stations, -1, 1], y + pitch)
        assert np.array_equal(tg.nodes[stations, :, 1], (1.0 - eta) * y[:, None] + eta * (y + pitch)[:, None])
    assert abs(tg.nodes[f - 1, 0, 1] - grid.nodes[f - 1, 0, 1]) > 1e-3 * c      # measured 0.0156 c
    tb = tg.boundaries
    assert np.array_equal(tb.upstream_lower_boundary[-1], ends["tip"]["point"])
    assert np.array_equal(tb.downstream_lower_boundary[0], ends["tail"]["point"])
    assert np.array_equal(tb.upstream_upper_boundary, tb.upstream_lower_boundary + b.periodic_translation)
    assert np.array_equal(tb.downstream_upper_boundary, tb.downstream_lower_boundary + b.periodic_translation)
    assert (tb.upstream_m[0], tb.upstream_m[-1]) == (b.m_in, ends["tip"]["m"])
    assert (tb.downstream_m[0], tb.downstream_m[-1]) == (ends["tail"]["m"], b.m_out)
    assert (tb.upstream_beta[0], tb.upstream_beta[-1]) == (b.inlet_angle, cascade.kappa_le)
    assert (tb.downstream_beta[0], tb.downstream_beta[-1]) == (cascade.kappa_te, b.exit_angle)
    met = build_metrics(tg)
    assert np.array_equal(met.tan_beta0[f:l + 1], build_metrics(grid).tan_beta0[f:l + 1])
    with pytest.raises(gr.GridError):                                           # __post_init__ re-validates
        replace(tg, first_blade_i=f + 1)


def test_le_point_operator_residual_is_the_transition_study_table(study):
    grid = study.case_of("very_coarse").grid
    stats = study.operator_level(grid, 33.0)["stats"]
    for row, (published, tolerance) in PUBLISHED_RESIDUAL.items():
        assert abs(stats[row]["full"]["max"] - published) <= tolerance, (row, stats[row]["full"]["max"])


def test_tip_operator_residual_is_finite_and_its_split_adds_up(study, very_coarse):
    case, ends = very_coarse
    tg = study.tip_grid(case.grid, case.cascade, ends)
    manufactured = study.manufactured_of(tg)
    for gamma in study.GAMMAS_DEG:
        r = study.operator_level(tg, gamma, manufactured)
        assert np.all(np.isfinite(r["full"][1:-1, 1:-1]))
        assert r["split_gap"] <= TOL_SPLIT and r["split_gap_periodic"] <= TOL_SPLIT, (gamma, r["split_gap"],
                                                                                        r["split_gap_periodic"])
        for row_class, terms in r["stats"].items():
            assert terms["full"]["n"] > 0 and np.isfinite(terms["full"]["max"]), (gamma, row_class)
        for row in ("f-1", "f", "l", "l+1"):
            assert r["stats"][row]["full"]["max"] > 0.0


@pytest.mark.parametrize("anchor", ["le_point", "tip"])
def test_side_line_closed_form_against_finite_differences(study, anchor):
    case = study.case_of("very_coarse")
    grid = study.anchored_grid("very_coarse", anchor)
    f = grid.first_blade_i
    m = np.asarray(grid.m)[:f]
    args = study.side_junctions(grid, case.cascade, anchor)["inlet"]
    exact = study.side_line_exact(*args, m)
    assert np.max(np.abs(exact["y"] - np.asarray(grid.nodes)[:f, 0, 1])) / grid.chord <= TOL_Y0_CLOSED
    errors = []
    for h in (FD_STEP, FD_STEP / 2):
        plus, minus = study.side_line_exact(*args, m + h), study.side_line_exact(*args, m - h)
        errors.append((np.max(np.abs((plus["y"] - minus["y"]) / (2 * h) - exact["tan"])),
                       np.max(np.abs((plus["y"] - 2 * exact["y"] + minus["y"]) / h ** 2 - exact["dtan"])),
                       np.max(np.abs((plus["tan"] - minus["tan"]) / (2 * h) - exact["dtan"]))))
    for k, tolerance in enumerate(TOL_FD):
        assert errors[0][k] <= tolerance, (k, errors[0][k])
        assert 3.9 <= errors[0][k] / errors[1][k] <= 4.1, (k, errors[0][k] / errors[1][k])
