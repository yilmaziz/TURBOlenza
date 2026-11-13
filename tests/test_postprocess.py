"""
Tests for turbolenza.b2b.postprocess (surface quantities, discharge angle, summary,
measured-data import) and turbolenza.b2b.plots (the figures), on the dimensional
stationary verification case of test_solver.py (33 deg, n_passage 43, n_eta 21, chord 0.1 m,
inlet 45 deg at M 0.3, the mass flow derived from the inlet Mach number).

What is checked
- surface_mach: the blade stations only, solution.mach itself at the two wall rows.
- blade_surfaces: the suction / pressure walls from grid.wall_identity (an SS/PS-swapped
  identity swaps them, a wall whose surface changes raises), static pressure from the solution.
- x/c: the blade-frame x of geometry.py, checked by turning the wall nodes back by the stagger
  about the LE point independently; the LE point at 0 and the TE point at 1; the upper wall
  moved back by one pitch; x/c is not m/c under stagger.
- The pressure coefficient: Dunavant's S = (Pt1 - p) / q1 and (p - p1) / q1 by hand from the
  solution's static pressure and its inlet state; S = (Pt1 - p1) / q1 - Cp; refusals without
  an inlet state, with q1 = 0 and for an unknown convention.
- discharge_angle_deg = degrees(grid.exit_angle) exactly; the far-exit velocity angle agrees
  (measured mismatch: 1.1e-14 deg at the fixed angle kappa_te, a straight exit line; 5.9e-5 deg
  for the coupled solution, the finite-difference slope of the turned exit line).
- summarize: the numbers are those of the arrays; the text holds no loss quantity.
- read_surface_points: header-only file empty, rows read by surface, bad files refused.
- plots (Agg backend, no window): one axes with the SS and PS curves of blade_surfaces against
  x/c, labels, legend, grid, the M = 1 line, no forced limits, the y axis of S not inverted,
  measured points overlaid, the convergence figure from the pass records; show() never called.
"""

from __future__ import annotations

import math
from dataclasses import replace
from functools import lru_cache
from unittest import mock

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from turbolenza.b2b import config as cf  # noqa: E402
from turbolenza.b2b import eos, plots  # noqa: E402
from turbolenza.b2b import postprocess as pp  # noqa: E402
from turbolenza.b2b import solver as sv  # noqa: E402
from turbolenza.b2b.geometry import CascadeSpec, make_cascade  # noqa: E402
from turbolenza.b2b.grid import Grid, WallIdentity, WallSide  # noqa: E402

AIR = eos.CaloricallyPerfectGas(gamma=1.4, gas_constant=287.05)
C_REF = 0.1


@lru_cache(maxsize=None)
def cascade():
    return make_cascade(CascadeSpec(blade_type="NACA65", solidity=1.0, stagger_deg=33.0, cl0=1.2, t_max_over_c=0.10,
                                    surface_continuity="C2"))


def config(tolerance=1e-8, passes=400):
    return cf.SolverConfig(
        grid=cf.GridSettings(n_passage=43, n_eta=21),
        flow=cf.FlowConditions(inlet_mach=0.3, inlet_angle_deg=45.0, exit_angle_guess_deg=math.degrees(cascade().kappa_te),
                               inlet_absolute_total_pressure=101325.0, inlet_absolute_total_temperature=288.15,
                               angular_speed=0.0),
        gas=AIR, iteration=cf.IterationSettings(density_tolerance=tolerance, max_density_iterations=passes,
                                                kutta_tolerance=tolerance, max_kutta_iterations=passes))


@lru_cache(maxsize=None)
def fixed():
    return sv.solve_density_coupled(cascade(), config(), c_ref=C_REF)


@lru_cache(maxsize=None)
def coupled():
    return sv.solve(cascade(), config(), c_ref=C_REF)


# --- surfaces --------------------------------------------------------------------------------------------------

def test_surface_mach_is_the_solution_mach_on_the_blade_walls_only():
    sol = fixed()
    g = sol.grid
    m, eta0, eta1 = pp.surface_mach(sol)
    blade = slice(g.first_blade_i, g.last_blade_i + 1)
    assert np.array_equal(m, g.m[blade]) and len(m) == g.last_blade_i - g.first_blade_i + 1
    assert np.array_equal(eta0, sol.mach[blade, 0]) and np.array_equal(eta1, sol.mach[blade, -1])
    assert m[0] > g.m[g.last_inlet_i] and m[-1] < g.m[g.first_exit_i]          # no periodic station


def test_blade_surfaces_name_the_walls_from_the_wall_identity():
    sol = fixed()
    g = sol.grid
    s = pp.blade_surfaces(sol)
    blade = slice(g.first_blade_i, g.last_blade_i + 1)
    assert set(g.wall_identity.lower.surface) == {"suction"} and set(g.wall_identity.upper.surface) == {"pressure"}
    assert (s.ss_j, s.ps_j) == (0, len(g.eta) - 1)
    assert np.array_equal(s.stations, np.arange(g.first_blade_i, g.last_blade_i + 1))
    assert np.array_equal(s.mach_ss, sol.mach[blade, 0]) and np.array_equal(s.mach_ps, sol.mach[blade, -1])
    assert np.array_equal(s.pressure_ss, sol.static_state.pressure[blade, 0])
    assert np.array_equal(s.pressure_ps, sol.static_state.pressure[blade, -1])


def identity_with(g: Grid, lower=None, upper=None) -> WallIdentity:
    identity = g.wall_identity

    def side(s: WallSide, names) -> WallSide:
        if names is None:
            return s
        return WallSide(surface=np.array(names), parameter=s.parameter, u=s.u, contour=s.contour, hidden=s.hidden)

    return WallIdentity(stations=identity.stations, lower=side(identity.lower, lower), upper=side(identity.upper, upper))


def test_a_swapped_or_changing_identity_is_followed_or_refused():
    sol = fixed()
    g = sol.grid
    n = g.last_blade_i - g.first_blade_i + 1
    blade = slice(g.first_blade_i, g.last_blade_i + 1)
    with mock.patch.object(Grid, "wall_identity", new_callable=mock.PropertyMock,
                           return_value=identity_with(g, ["pressure"] * n, ["suction"] * n)):
        s = pp.blade_surfaces(sol)
    assert (s.ss_j, s.ps_j) == (len(g.eta) - 1, 0) and np.array_equal(s.mach_ss, sol.mach[blade, -1])
    mixed = ["suction"] * n
    mixed[0] = "pressure"                                                   # a nose stretch on the other surface
    with mock.patch.object(Grid, "wall_identity", new_callable=mock.PropertyMock,
                           return_value=identity_with(g, lower=mixed)):
        with pytest.raises(pp.PostprocessError, match="changes"):
            pp.blade_surfaces(sol)
    with mock.patch.object(Grid, "wall_identity", new_callable=mock.PropertyMock,
                           return_value=identity_with(g, upper=["suction"] * n)):
        with pytest.raises(pp.PostprocessError, match="not one suction and one pressure"):
            pp.blade_surfaces(sol)


def test_x_over_c_is_the_blade_frame_chordwise_station():
    """Turn the wall nodes back by the stagger about the LE point (geometry.py's blade frame:
    LE (0, 0), TE (1, 0)); their x is x/c. The upper wall's nodes first move back one pitch."""
    sol = fixed()
    g = sol.grid
    s = pp.blade_surfaces(sol)
    stagger, a = cascade().stagger, cascade().active
    le = np.asarray(a.leading_edge_point)
    c, sn = math.cos(stagger), math.sin(stagger)
    blade = np.arange(g.first_blade_i, g.last_blade_i + 1)
    lower = g.nodes[blade, 0] - le
    upper = g.nodes[blade, -1] - np.asarray(a.periodic_translation) - le
    assert np.max(np.abs(s.x_c_ss - (lower[:, 0] * c + lower[:, 1] * sn) / a.chord)) < 1e-14
    assert np.max(np.abs(s.x_c_ps - (upper[:, 0] * c + upper[:, 1] * sn) / a.chord)) < 1e-14
    ends = pp.chordwise_coordinate(np.array([a.leading_edge_point, a.trailing_edge_point]), a.leading_edge_point,
                                   a.trailing_edge_point)
    assert ends[0] == 0.0 and ends[1] == pytest.approx(1.0, abs=1e-15)
    assert 0.0 < s.x_c_ss[0] < s.x_c_ss[-1] < 1.0 and 0.0 < s.x_c_ps[0] < s.x_c_ps[-1] < 1.0
    assert np.all(np.diff(s.x_c_ss) > 0) and np.all(np.diff(s.x_c_ps) > 0)
    assert np.max(np.abs(s.x_c_ss - s.m_c)) > 0.05                     # x/c is not the axial m/c under stagger


# --- pressure coefficient --------------------------------------------------------------------------------------

def test_pressure_coefficients_by_hand_on_the_inlet_reference():
    sol = fixed()
    inlet = sol.inlet
    q1 = 0.5 * inlet.static_density * inlet.velocity ** 2
    assert inlet.dynamic_pressure == pytest.approx(q1, rel=1e-15)
    s = pp.blade_surfaces(sol)
    S = pp.surface_pressure_coefficient(sol)                            # default: Dunavant's S
    Cp = pp.surface_pressure_coefficient(sol, pp.INLET_STATIC)
    assert S.convention == pp.DUNAVANT and "Pt1 - p" in S.label and "p - p1" in Cp.label
    assert np.allclose(S.values_ss, (inlet.total_pressure - s.pressure_ss) / q1, rtol=1e-14, atol=0.0)
    assert np.allclose(S.values_ps, (inlet.total_pressure - s.pressure_ps) / q1, rtol=1e-14, atol=0.0)
    assert np.allclose(Cp.values_ss, (s.pressure_ss - inlet.static_pressure) / q1, rtol=1e-12, atol=1e-15)
    offset = (inlet.total_pressure - inlet.static_pressure) / q1        # 1.0227 at M 0.3 (compressibility)
    assert offset == pytest.approx(1.0227029548085425, rel=1e-12)
    assert np.allclose(S.values_ss + Cp.values_ss, offset, rtol=0.0, atol=1e-12)
    assert np.all(S.values_ss > S.values_ps[: len(S.values_ss)].min())   # the suction side has the higher S
    m, c0, c1 = pp.pressure_coefficient(sol)
    assert np.array_equal(c0, S.values_ss) and np.array_equal(c1, S.values_ps) and np.array_equal(m, s.m)


def test_pressure_coefficient_refusals():
    sol = fixed()
    with pytest.raises(pp.PostprocessError, match="no inlet state"):
        pp.pressure_coefficient(replace(sol, inlet=None))
    with pytest.raises(pp.PostprocessError, match="q1 > 0"):
        pp.pressure_coefficient(replace(sol, inlet=replace(sol.inlet, dynamic_pressure=0.0)))
    with pytest.raises(pp.PostprocessError, match="convention"):
        pp.pressure_coefficient(sol, "pt_static")
    with pytest.raises(TypeError):
        pp.surface_mach(None)


# --- discharge angle and summary ---------------------------------------------------------------------------------

@pytest.mark.parametrize("which, tolerance", [("fixed", 1e-12), ("coupled", 1e-4)])
def test_discharge_angle_is_the_grid_angle_and_the_exit_velocity_agrees(which, tolerance):
    sol = fixed() if which == "fixed" else coupled()
    assert pp.discharge_angle_deg(sol) == math.degrees(sol.grid.exit_angle)
    mismatch = pp.exit_velocity_angle_deg(sol) - pp.discharge_angle_deg(sol)
    assert abs(mismatch) < tolerance                                    # measured 1.1e-14 (fixed), 5.9e-5 (coupled) deg


def test_summary_numbers_are_those_of_the_arrays():
    sol = coupled()
    summary = pp.summarize(sol)
    s = pp.blade_surfaces(sol)
    assert summary.discharge_angle_deg == math.degrees(sol.exit_angle)
    assert summary.turning_deg == pytest.approx(45.0 - summary.discharge_angle_deg, abs=1e-12)
    assert summary.ss_peak_mach == np.max(s.mach_ss) and summary.ps_peak_mach == np.max(s.mach_ps)
    assert summary.ss_peak_x_c == s.x_c_ss[np.argmax(s.mach_ss)] and 0.0 < summary.ss_peak_x_c < 1.0
    assert summary.surface_mach_min == min(np.min(s.mach_ss), np.min(s.mach_ps))
    assert summary.mach_max == np.max(sol.mach) >= summary.surface_mach_max
    assert summary.density_min == np.min(sol.density) and summary.pressure_max == np.max(sol.static_state.pressure)
    assert summary.normalized_kutta_residual == sol.normalized_kutta_residual <= 1e-8
    assert summary.iterations == sol.iterations and summary.converged
    assert summary.inlet_mach == 0.3 and summary.mass_flow_source == sv.MASS_FLOW_FROM_INLET
    text = pp.format_summary(summary)
    for word in ("beta_out", "turning", "SS peak", "PS peak", "Kutta", "max M_rel", "mass flow"):
        assert word in text
    assert "loss" not in text.lower() and "boundary layer" not in text.lower()


# --- measured data ---------------------------------------------------------------------------------------------

def test_surface_points_are_read_by_surface(tmp_path):
    empty = tmp_path / "empty.csv"
    empty.write_text("surface,x_c,mach\n", encoding="utf-8")
    assert pp.read_surface_points(empty).empty
    data = tmp_path / "points.csv"
    data.write_text("surface,x_c,mach\nsuction,0.1,0.8\npressure,0.1,0.55\nsuction,0.5,0.7\n", encoding="utf-8")
    p = pp.read_surface_points(data)
    assert np.array_equal(p.x_c_ss, [0.1, 0.5]) and np.array_equal(p.values_ss, [0.8, 0.7])
    assert np.array_equal(p.x_c_ps, [0.1]) and np.array_equal(p.values_ps, [0.55]) and not p.empty
    bad = tmp_path / "bad.csv"
    bad.write_text("surface,x_c,mach\nupper,0.1,0.8\n", encoding="utf-8")
    with pytest.raises(pp.PostprocessError, match="not suction or pressure"):
        pp.read_surface_points(bad)
    with pytest.raises(pp.PostprocessError, match="lacks"):
        pp.read_surface_points(data, quantity="S")


# --- plots -----------------------------------------------------------------------------------------------------

@pytest.fixture
def no_show(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("a plotting helper called show()")
    monkeypatch.setattr(plt, "show", refuse)
    before = set(plt.get_fignums())
    yield
    for number in set(plt.get_fignums()) - before:
        plt.close(number)


def test_surface_mach_plot(no_show, tmp_path):
    sol = fixed()
    s = pp.blade_surfaces(sol)
    points = tmp_path / "points.csv"
    points.write_text("surface,x_c,mach\nsuction,0.2,0.4\npressure,0.2,0.25\n", encoding="utf-8")
    fig, ax = plots.plot_surface_mach(sol, experiment=pp.read_surface_points(points))
    lines = {line.get_label(): line for line in ax.get_lines()}
    assert np.array_equal(lines["Suction side"].get_xdata(), s.x_c_ss)
    assert np.array_equal(lines["Suction side"].get_ydata(), s.mach_ss)
    assert np.array_equal(lines["Pressure side"].get_xdata(), s.x_c_ps)
    assert np.array_equal(lines["Pressure side"].get_ydata(), s.mach_ps)
    assert "M = 1" in lines and set(lines["M = 1"].get_ydata()) == {1.0}
    assert "Experiment, suction side" in lines and "Experiment, pressure side" in lines
    assert ax.get_xlabel() == "x/c" and "Mach" in ax.get_ylabel() and ax.get_title()
    assert ax.get_legend() is not None and ax.get_autoscalex_on() and ax.get_autoscaley_on()
    assert any(line.get_visible() for line in ax.get_xgridlines())
    assert fig is ax.figure and len(fig.axes) == 1


def test_pressure_coefficient_plot(no_show):
    sol = fixed()
    S = pp.surface_pressure_coefficient(sol)
    fig, ax = plots.plot_pressure_coefficient(sol)
    lines = {line.get_label(): line for line in ax.get_lines()}
    assert np.array_equal(lines["Suction side"].get_xdata(), pp.blade_surfaces(sol).x_c_ss)   # the Mach plot's x/c
    assert np.array_equal(lines["Suction side"].get_ydata(), S.values_ss)
    assert np.array_equal(lines["Pressure side"].get_ydata(), S.values_ps)
    assert ax.get_ylabel() == pp.LABELS[pp.DUNAVANT] and not ax.yaxis_inverted() and ax.get_xlabel() == "x/c"
    fig2, ax2 = plots.plot_pressure_coefficient(sol, convention=pp.INLET_STATIC)
    assert ax2.get_ylabel() == pp.LABELS[pp.INLET_STATIC]


def test_convergence_plot_from_the_pass_records(no_show):
    sol = fixed()
    fig, axes = plots.plot_convergence(sol, config().iteration)
    assert set(axes) == {"changes", "linear", "angle", "mach"} and len(fig.axes) == 4
    density = [line for line in axes["changes"].get_lines() if "density" in line.get_label()][0]
    assert len(density.get_xdata()) == len(sol.passes)
    assert np.array_equal(density.get_ydata(), [p.density_change for p in sol.passes])
    tolerances = [line for line in axes["changes"].get_lines() if "tol" in line.get_label()]
    assert tolerances and all(set(line.get_ydata()) == {1e-8} for line in tolerances)


def test_a_non_converged_solution_is_labeled(no_show):
    sol = replace(fixed(), converged=False)
    _, ax = plots.plot_surface_mach(sol)
    assert "NOT CONVERGED" in ax.get_title()


def test_mach_contour_is_node_based_over_the_whole_domain(no_show):
    """plot_mach_contour shades the solver's own node values on the grid's own nodes (one
    Gouraud-shaded mesh per periodic copy, the copies shifted by whole pitches), with the
    blades on top and a colour bar; a window only sets the view."""
    from matplotlib.collections import QuadMesh

    sol = fixed()
    nodes, shift = np.asarray(sol.grid.nodes), np.asarray(sol.grid.boundaries.periodic_translation)
    fig, ax = plots.plot_mach_contour(sol)
    meshes = [c for c in ax.collections if isinstance(c, QuadMesh)]
    assert len(meshes) == 3
    for k, mesh in zip((-1, 0, 1), meshes):
        assert mesh._shading == "gouraud"
        assert np.array_equal(np.asarray(mesh.get_array()).reshape(sol.mach.shape), sol.mach)
        assert np.array_equal(mesh.get_coordinates(), nodes + k * shift)
    assert len(ax.patches) == 4                                             # blades -1 .. 2
    assert len(fig.axes) == 2 and ax.get_aspect() == 1.0                    # the colour bar
    fig2, ax2 = plots.plot_mach_contour(sol, copies=(0,), window=(0.0, 1.0, -0.5, 1.5), isolines=None, colorbar=False)
    assert ax2.get_xlim() == (0.0, 1.0) and ax2.get_ylim() == (-0.5, 1.5) and len(fig2.axes) == 1
