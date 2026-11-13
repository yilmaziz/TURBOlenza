"""
Tests for turbolenza.b2b.kutta: the trailing-edge Kutta residual (Aungier 2003 Eq.
5-34) and the exit-angle correction (Aungier 2000 Eqs. 13-57, 13-59 mapped onto this
project's tan(beta) convention), checked without the coupled solver.

What is checked
- The angle convention: Aungier 2000's cot(beta) (Eq. 13-23, angle from the tangential
  direction) is this project's tan(beta) (angle from the meridional direction): Eq. 13-12
  with cot -> tan reproduces flow.py's W_theta exactly, the reciprocal reading does not; at
  the far exit W_theta / W_m is tan(exit angle).
- The 1 / cos(beta) of Eq. 13-57 is the wall length per dm of THIS project's beta: the discrete
  Stokes balance of the blade elements holds with it (medians 0.992 interior, 0.993 rear
  half) and fails with 1 / sin(beta) (0.298, 0.232).
- The TE station is grid.last_blade_i with the SS and PS nodes of grid.wall_identity, never
  l + 1; an SS/PS-swapped station reverses R_K and not the wall difference; a station of two
  suction nodes is refused.
- R_K, eps_K and the correction by hand; the zero-speed fallback; zero residual -> exactly zero
  correction and the angle unchanged bit for bit; the damped update and its range.
- Pressure equivalence (a verification, not the residual): at one rothalpy and entropy p is a
  strictly falling function of W, dp/dW = -rho W, so W_SS = W_PS <=> p_SS = p_PS, and W_SS >
  W_PS <=> p_SS < p_PS on the solved TE nodes.
- Kutta alone at a frozen density (the converged fixed-angle density of the verification
  case, every flow from direct module calls): the solver's own response dR_K/d(beta_out) < 0
  on both sides of the root, measured by central differences; the first correction moves
  beta_out up where R_K > 0 and down where R_K < 0 (the sign test, both directions); from
  kappa_te (R_K > 0) and kappa_te + 10 deg (R_K < 0) the iteration contracts monotonically,
  never changes sign and closes on the root found independently by a secant on R_K(beta_out).
- kutta.py solves, relaxes and assembles nothing: no density, EOS, matrix or linear-solver
  name in it.

Tolerances name the measurement they cover: Eq. 13-12 against flow.py to 1e-12 relative
(the same products, round-off); the far-exit slope to 1e-5 (measured 8e-15 at the fixed angle
kappa_te, whose exit line is straight; 1.2e-6 in a coupled solution, whose turned exit line
metrics.py differentiates by finite differences); the Stokes medians within 0.02 of 1 with cos
(measured 0.992, 0.993) and below 0.5 with sin (0.298, 0.232); dp/dW to 1e-8 (measured
4e-10); the frozen-density contraction of R_K per step in [0.87, 0.88] (measured 0.8772 to
0.8773), the distance to the root after 25 steps below 5 % of the start (measured 3.6 % from
below, 3.9 % from above).
"""

from __future__ import annotations

import ast
import math
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from turbolenza.b2b import config as cf
from turbolenza.b2b import eos, kutta
from turbolenza.b2b import solver as sv
from turbolenza.b2b.boundary_conditions import build_boundary_conditions
from turbolenza.b2b.coefficients import build_coefficients
from turbolenza.b2b.flow import velocity_components
from turbolenza.b2b.geometry import CascadeSpec, make_cascade
from turbolenza.b2b.grid import EXIT, PASSAGE, Grid, WallIdentity, WallSide, make_grid, rebuild_exit
from turbolenza.b2b.linear_solver import solve_linear_system
from turbolenza.b2b.matrix_assembly import assemble_from_coefficients
from turbolenza.b2b.metrics import build_metrics

AIR = eos.CaloricallyPerfectGas(gamma=1.4, gas_constant=287.05)
PT, TT = 101325.0, 288.15
C_REF = 0.1                          # m
INLET_DEG = 45.0


@lru_cache(maxsize=None)
def cascade():
    return make_cascade(CascadeSpec(blade_type="NACA65", solidity=1.0, stagger_deg=33.0, cl0=1.2, t_max_over_c=0.10,
                                    surface_continuity="C2"))


def mass_flow() -> float:
    """mdot' of the far-inlet relative Mach number 0.3 (test_solver.py's derivation)."""
    T = TT / (1.0 + 0.5 * (AIR.gamma - 1.0) * 0.3 ** 2)
    W = 0.3 * math.sqrt(AIR.gamma * AIR.gas_constant * T)
    state = eos.static_state_from_velocity(W, AIR, eos.blade_row_reference(eos.TotalState(PT, TT), AIR))
    return state.density * W * math.cos(math.radians(INLET_DEG)) * float(cascade().active.pitch) * C_REF


def config() -> cf.SolverConfig:
    return cf.SolverConfig(
        grid=cf.GridSettings(n_passage=43, n_eta=21),
        flow=cf.FlowConditions(inlet_mach=0.3, inlet_angle_deg=INLET_DEG, exit_angle_guess_deg=math.degrees(cascade().kappa_te),
                               inlet_absolute_total_pressure=PT, inlet_absolute_total_temperature=TT, angular_speed=0.0),
        gas=AIR, iteration=cf.IterationSettings(density_tolerance=1e-8, max_density_iterations=200, kutta_tolerance=1.0,
                                                max_kutta_iterations=1))


@lru_cache(maxsize=None)
def start_grid() -> Grid:
    c = config()
    return make_grid(cascade(), c.grid, math.radians(INLET_DEG), cascade().kappa_te, require_ss_ps=True)


@lru_cache(maxsize=None)
def fixed_angle():
    """The converged fixed-angle density-coupled solution at kappa_te (the frozen density)."""
    return sv.solve_at_exit_angle(start_grid(), config(), mass_flow_per_depth=mass_flow(), c_ref=C_REF)


@lru_cache(maxsize=None)
def flow_at(beta: float):
    """(grid, metrics, flow) at the exit angle beta with the frozen density: module calls only,
    no solver.py."""
    rho = np.array(fixed_angle().density)
    g = start_grid() if beta == start_grid().exit_angle else rebuild_exit(start_grid(), beta)
    metrics, bcs = build_metrics(g), build_boundary_conditions(g)
    co = build_coefficients(metrics, rho, mass_flow(), c_ref=C_REF)
    linear = solve_linear_system(assemble_from_coefficients(co, metrics, bcs))
    return g, metrics, velocity_components(linear.psi, rho, co, metrics, bcs)


def residual_at(beta: float) -> float:
    g, _, flow = flow_at(beta)
    return kutta.kutta_residual(flow, g).residual


def synthetic(g: Grid, te: dict[int, tuple[float, float]], exit_W_m: float = 70.0) -> SimpleNamespace:
    """A flow object with (W_m, W_theta) set at the TE wall nodes {j: (W_m, W_theta)}, a uniform
    W_m at the far exit and zeros elsewhere."""
    shape = (len(g.m), len(g.eta))
    W_m, W_theta = np.zeros(shape), np.zeros(shape)
    for j, (a, b) in te.items():
        W_m[g.last_blade_i, j], W_theta[g.last_blade_i, j] = a, b
    W_m[-1] = exit_W_m
    return SimpleNamespace(W_m=W_m, W_theta=W_theta)


# --- the angle convention -------------------------------------------------------------------------------

def test_aungier_2000_cot_beta_is_this_projects_tan_beta():
    """Eq. 13-12, W_U = mdot / (rho b) [(cot(beta) / S) psi_eta - psi_m], with cot(beta_2000) =
    tan(beta) (Eq. 13-23 against Eq. 5-3) is flow.py's W_theta; read with the reciprocal it is
    not. At the far exit (psi_m = 0) W_theta / W_m = tan(beta) of the grid line = tan(exit angle)."""
    sol = fixed_angle()
    flow, met = sol.flow, sol.metrics
    mu = sol.mass_flow_per_depth / sol.density
    S = C_REF * np.asarray(met.S)[:, None]
    cot_2000 = np.asarray(met.tan_beta)
    W_U = mu * (cot_2000 / S * flow.psi_eta - flow.psi_m)
    assert np.max(np.abs(W_U - flow.W_theta)) <= 1e-12 * np.max(np.abs(flow.W_theta))
    with np.errstate(divide="ignore"):
        reciprocal = mu * (1.0 / cot_2000 / S * flow.psi_eta - flow.psi_m)
    assert np.max(np.abs(reciprocal - flow.W_theta)) > 0.1 * np.max(np.abs(flow.W_theta))
    beta_2000 = 0.5 * math.pi - sol.grid.exit_angle             # angle from the tangential direction
    assert 1.0 / math.tan(beta_2000) == pytest.approx(math.tan(sol.grid.exit_angle), rel=1e-14)
    assert np.max(np.abs(flow.W_theta[-1] / flow.W_m[-1] - math.tan(sol.grid.exit_angle))) < 1e-5   # measured 8e-15


def test_the_wall_length_factor_is_cos_of_this_projects_beta():
    """Stokes' theorem on a blade element (stations i, i + 1): (S W_theta_bar)_(i+1) -
    (S W_theta_bar)_i = int_wall1 W ds - int_wall0 W ds with ds = dm / cos(beta), beta from the
    meridional direction; the same balance with ds = dm / sin(beta) (cos(beta_2000) read
    literally) fails by about the wall slope."""
    sol = fixed_angle()
    g, f, met = sol.grid, sol.flow, sol.metrics

    def circulation(i):
        return met.S[i] * np.trapezoid(f.W_theta[i], g.eta)

    def ratio(a, trig):
        b, total = a + 1, 0.0
        for j, slope, sign in ((0, met.tan_beta0, -1.0), (-1, met.tan_beta1, 1.0)):
            t = 0.5 * (slope[a] + slope[b])
            c = 1.0 / math.sqrt(1.0 + t * t)
            total += sign * 0.5 * (f.W[a, j] + f.W[b, j]) * g.dm / (c if trig == "cos" else abs(t) * c)
        return (circulation(b) - circulation(a)) / total

    mid = (g.first_blade_i + g.last_blade_i) // 2
    for lo, hi in ((g.first_blade_i + 2, g.last_blade_i - 3), (mid, g.last_blade_i - 4)):
        cos_median = float(np.median([ratio(a, "cos") for a in range(lo, hi + 1)]))
        sin_median = float(np.median([ratio(a, "sin") for a in range(lo, hi + 1)]))
        assert abs(cos_median - 1.0) < 0.02                        # measured 0.992 (interior), 0.993 (rear half)
        assert sin_median < 0.5                                    # measured 0.298, 0.232


# --- the TE station and the residual -------------------------------------------------------------------

def test_the_te_station_is_the_last_blade_station_with_its_named_nodes():
    g = start_grid()
    station = kutta.trailing_edge_station(g)
    assert station.i == g.last_blade_i and station.i != g.first_exit_i
    assert g.region[station.i] == PASSAGE and g.region[station.i + 1] == EXIT
    assert station.m == g.m[station.i] and station.m_offset == pytest.approx(g.m[station.i] - g.m_te, abs=0)
    k = station.i - g.first_blade_i
    surfaces = {0: g.wall_identity.lower.surface[k], len(g.eta) - 1: g.wall_identity.upper.surface[k]}
    assert surfaces[station.ss_j] == "suction" and surfaces[station.ps_j] == "pressure"
    assert (station.ss_j, station.ps_j) == (0, len(g.eta) - 1)       # this project's grids: SS on eta = 0


def swapped_identity(g: Grid, lower: str, upper: str) -> WallIdentity:
    identity = g.wall_identity
    k = g.last_blade_i - g.first_blade_i

    def side(s: WallSide, name: str) -> WallSide:
        surface = np.array(s.surface)
        surface[k] = name
        return WallSide(surface=surface, parameter=s.parameter, u=s.u, contour=s.contour, hidden=s.hidden)

    return WallIdentity(stations=identity.stations, lower=side(identity.lower, lower), upper=side(identity.upper, upper))


def test_residual_by_hand_and_the_ss_ps_identity_decides_its_sign():
    g = start_grid()
    J = len(g.eta) - 1
    flow = synthetic(g, {0: (3.0, 4.0), J: (6.0, 8.0)})          # |W| = 5 on eta = 0, 10 on eta = 1
    r = kutta.kutta_residual(flow, g)
    assert (r.W_ss, r.W_ps, r.residual, r.wall_difference) == (5.0, 10.0, -5.0, -5.0)
    assert r.normalized == pytest.approx(5.0 / 7.5, rel=1e-15)
    with mock.patch.object(Grid, "wall_identity", new_callable=mock.PropertyMock,
                           return_value=swapped_identity(g, "pressure", "suction")):
        s = kutta.kutta_residual(flow, g)
    assert (s.station.ss_j, s.station.ps_j) == (J, 0)
    assert (s.W_ss, s.W_ps, s.residual, s.wall_difference) == (10.0, 5.0, 5.0, -5.0)   # R_K flips, W0 - W1 does not
    with mock.patch.object(Grid, "wall_identity", new_callable=mock.PropertyMock,
                           return_value=swapped_identity(g, "suction", "suction")):
        with pytest.raises(kutta.KuttaError, match="not one suction and one pressure"):
            kutta.kutta_residual(flow, g)


def test_zero_speed_fallback_and_bad_flows():
    g = start_grid()
    J = len(g.eta) - 1
    r = kutta.kutta_residual(synthetic(g, {0: (0.0, 0.0), J: (0.0, 0.0)}), g)
    assert r.residual == 0.0 and r.normalized == 0.0
    assert kutta.kutta_residual(synthetic(g, {0: (1e-300, 0.0), J: (0.0, 0.0)}), g).normalized == pytest.approx(2.0)
    with pytest.raises(kutta.KuttaError, match="not finite"):
        kutta.kutta_residual(synthetic(g, {0: (np.nan, 0.0), J: (1.0, 0.0)}), g)
    with pytest.raises(kutta.KuttaError, match="shape"):
        kutta.kutta_residual(SimpleNamespace(W_m=np.zeros((3, 3)), W_theta=np.zeros((3, 3))), g)


# --- the correction ---------------------------------------------------------------------------------------

def test_the_correction_by_hand():
    """Delta(tan(beta_out)) = 2 (W0 - W1) dm / (S (cos(beta0) + cos(beta1)) int W_m d(eta)),
    from the grid, the metrics and the synthetic speeds, computed here on its own."""
    g, met, _ = flow_at(start_grid().exit_angle)
    J, i = len(g.eta) - 1, g.last_blade_i
    for te, exit_W_m in (({0: (60.0, 20.0), J: (40.0, 10.0)}, 70.0), ({0: (40.0, 10.0), J: (60.0, 20.0)}, 55.0)):
        flow = synthetic(g, te, exit_W_m)
        r = kutta.kutta_residual(flow, g)
        c = kutta.exit_angle_correction(r, flow, met)
        W0, W1 = math.hypot(*te[0]), math.hypot(*te[J])
        cos0 = math.cos(math.atan(float(met.tan_beta0[i])))
        cos1 = math.cos(math.atan(float(met.tan_beta1[i])))
        expected = 2.0 * (W0 - W1) * g.dm / (float(met.S[i]) * (cos0 + cos1) * exit_W_m)
        assert c.delta_tan == pytest.approx(expected, rel=1e-13)
        assert c.delta_c_theta == pytest.approx(expected * exit_W_m, rel=1e-13)
        assert c.exit_meridional_integral == pytest.approx(exit_W_m, rel=1e-15)
        assert math.copysign(1.0, c.delta_tan) == math.copysign(1.0, W0 - W1)       # the sign follows W0 - W1
    with pytest.raises(kutta.KuttaError, match="no flow leaves"):
        flow = synthetic(g, {0: (60.0, 20.0), J: (40.0, 10.0)}, exit_W_m=0.0)
        kutta.exit_angle_correction(kutta.kutta_residual(flow, g), flow, met)


@pytest.mark.parametrize("lam", [1.0, 0.5, 0.25])
def test_zero_residual_leaves_the_angle_exactly(lam):
    g, met, _ = flow_at(start_grid().exit_angle)
    J = len(g.eta) - 1
    flow = synthetic(g, {0: (3.0, 4.0), J: (5.0, 0.0)})          # equal speeds 5, different directions
    r = kutta.kutta_residual(flow, g)
    c = kutta.exit_angle_correction(r, flow, met)
    assert r.residual == 0.0 and r.normalized == 0.0 and c.delta_tan == 0.0 and c.delta_c_theta == 0.0
    # kappa_te, and two angles at which arctan(tan(beta)) is not beta in floating point
    round_trip_off = (0.4885603254065163, -0.4684534809916385)
    assert all(math.atan(math.tan(angle)) != angle for angle in round_trip_off)
    for angle in (g.exit_angle, *round_trip_off):
        assert kutta.next_exit_angle(angle, c, lam) == angle        # bit for bit


def test_the_damped_update_and_its_range():
    g, met, _ = flow_at(start_grid().exit_angle)
    J = len(g.eta) - 1
    flow = synthetic(g, {0: (60.0, 20.0), J: (40.0, 10.0)})
    c = kutta.exit_angle_correction(kutta.kutta_residual(flow, g), flow, met)
    beta = g.exit_angle
    for lam in (1.0, 0.5, 0.25):
        assert kutta.next_exit_angle(beta, c, lam) == math.atan(math.tan(beta) + lam * c.delta_tan)
    for bad in (0.0, -0.1, 1.5, float("nan"), True, None):
        with pytest.raises(kutta.KuttaError, match="relaxation"):
            kutta.next_exit_angle(beta, c, bad)
    for angle in (0.5 * math.pi, -2.0, float("nan")):
        with pytest.raises(kutta.KuttaError, match="exit angle"):
            kutta.next_exit_angle(angle, c, 1.0)


def test_metrics_of_another_grid_are_refused():
    g, met, flow = flow_at(start_grid().exit_angle)
    r = kutta.kutta_residual(flow, g)
    other = make_grid(cascade(), cf.GridSettings(n_passage=31, n_eta=21), math.radians(INLET_DEG), cascade().kappa_te)
    with pytest.raises(kutta.KuttaError):
        kutta.exit_angle_correction(r, flow, build_metrics(other))
    with pytest.raises(TypeError):
        kutta.exit_angle_correction(r, flow, None)


# --- pressure equivalence (verification only) ---------------------------------------------------------------

def test_equal_te_speeds_mean_equal_te_pressures():
    ref = eos.blade_row_reference(eos.TotalState(PT, TT), AIR)
    W = np.linspace(0.0, 300.0, 3001)
    p = eos.static_state_from_velocity(W, AIR, ref).pressure
    assert np.all(np.diff(p) < 0.0)                                 # one W, one p: W_SS = W_PS <=> p_SS = p_PS
    a, b = eos.static_state_from_velocity(70.75, AIR, ref), eos.static_state_from_velocity(70.75, AIR, ref)
    assert (a.pressure, a.temperature, a.density) == (b.pressure, b.temperature, b.density)
    for W0 in (60.0, 120.0):
        h = 1e-3
        slope = (eos.static_state_from_velocity(W0 + h, AIR, ref).pressure
                 - eos.static_state_from_velocity(W0 - h, AIR, ref).pressure) / (2 * h)
        assert slope / (-eos.static_state_from_velocity(W0, AIR, ref).density * W0) == pytest.approx(1.0, abs=1e-8)
    sol = fixed_angle()                                             # R_K = +48 m/s at kappa_te
    st, r = sol.static_state, kutta.kutta_residual(sol.flow, sol.grid)
    i = r.station.i
    assert r.residual > 0.0 and st.pressure[i, r.station.ss_j] < st.pressure[i, r.station.ps_j]


# --- the Kutta correction alone, at a frozen density ---------------------------------------------------------

@lru_cache(maxsize=None)
def frozen_root() -> float:
    """The exit angle of R_K = 0 at the frozen density, by a secant on R_K(beta_out): solves
    only, no Aungier correction."""
    a, b = cascade().kappa_te, cascade().kappa_te + math.radians(10.0)
    Ra, Rb = residual_at(a), residual_at(b)
    for _ in range(30):
        a, Ra, b = b, Rb, b - Rb * (b - a) / (Rb - Ra)
        Rb = residual_at(b)
        if abs(Rb) < 1e-9:
            return b
    raise AssertionError(f"the secant did not close: R_K = {Rb}")


def test_the_correction_moves_against_the_solvers_own_response_on_both_sides():
    """Sign test. The solver says how R_K answers the exit angle: dR_K/d(beta_out) < 0 at
    kappa_te (R_K > 0) and at kappa_te + 10 deg (R_K < 0). So R_K > 0 needs a larger angle and
    R_K < 0 a smaller one, and that is the direction of the Aungier correction in each case."""
    root = frozen_root()
    h = math.radians(0.5)
    for start in (cascade().kappa_te, cascade().kappa_te + math.radians(10.0)):
        slope = (residual_at(start + h) - residual_at(start - h)) / (2.0 * h)
        assert slope < 0.0                                          # measured -471 and -547 m/s per rad
        g, met, flow = flow_at(start)
        r = kutta.kutta_residual(flow, g)
        step = kutta.next_exit_angle(start, kutta.exit_angle_correction(r, flow, met), 1.0) - start
        assert math.copysign(1.0, step) == math.copysign(1.0, -r.residual * slope)
        assert math.copysign(1.0, step) == math.copysign(1.0, root - start)
    assert residual_at(cascade().kappa_te) > 0.0 > residual_at(cascade().kappa_te + math.radians(10.0))


@pytest.mark.parametrize("offset_deg", [0.0, 10.0])
def test_kutta_alone_converges_monotonically_from_both_sides(offset_deg):
    """At the frozen density, 25 undamped corrections from kappa_te (below the root) and from
    kappa_te + 10 deg (above): |R_K| falls by the same factor every step (0.8772 measured), R_K
    keeps its sign, beta_out stays on its side of the root and closes on it."""
    root = frozen_root()
    beta = cascade().kappa_te + math.radians(offset_deg)
    start = beta
    residuals, angles = [], []
    for _ in range(25):
        g, met, flow = flow_at(beta)
        r = kutta.kutta_residual(flow, g)
        residuals.append(r.residual)
        angles.append(beta)
        beta = kutta.next_exit_angle(beta, kutta.exit_angle_correction(r, flow, met), 1.0)
    ratios = [b / a for a, b in zip(residuals, residuals[1:])]
    assert all(0.87 <= q <= 0.88 for q in ratios)                  # measured 0.8772 to 0.8773: monotonic, no sign change
    side = math.copysign(1.0, root - start)
    assert all(math.copysign(1.0, root - a) == side for a in angles + [beta])
    assert all(side * (b - a) > 0.0 for a, b in zip(angles, angles[1:]))
    assert abs(beta - root) < 0.05 * abs(start - root)             # measured 3.6 % (below), 3.9 % (above) after 25 steps


# --- what kutta.py does not do --------------------------------------------------------------------------------

def test_kutta_module_solves_relaxes_and_assembles_nothing():
    tree = ast.parse(Path(kutta.__file__).read_text(encoding="utf-8"))
    imported = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
    imported |= {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    assert imported <= {"__future__", "math", "dataclasses", "numpy", "turbolenza.b2b.flow",
                        "turbolenza.b2b.grid", "turbolenza.b2b.metrics"}
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    names |= {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    assert not {"eos", "density", "relax_density", "build_coefficients", "assemble_from_coefficients",
                "solve_linear_system", "velocity_components", "static_state_from_velocity", "rebuild_exit"} & names
