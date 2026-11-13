"""
Tests for potential_flow_solver.flow: velocity recovery, psi -> psi_m, psi_eta -> W_m, W_theta
(Aungier 2003 Eqs. 5-18, 5-19) in the normalization of coefficients.py.

What is checked
- The derivative rules by node class on synthetic grids (boundary_conditions' own masks and
  periodic continuation): interior, far inlet / exit, lower / upper periodic lines, lower /
  upper blade walls, against closed-form derivatives of manufactured fields (the brief's
  affine-periodic psi, which is also 0 / 1 on eta = 0 / 1; a phase-shifted affine-periodic psi
  that is not, for the periodic lines' m differences; a wall field with different slopes on
  the two walls), with second-order convergence; psi = eta exactly (the +1 of the periodic
  continuation applied once).
- The velocity formulas: Eqs. 5-18, 5-19 against the independent flux form W_theta =
  (B psi_eta - C psi_m) / S, W_m = C psi_eta / S^2 to round-off; exact velocities of smooth
  rho, S, tan(beta) fields recovered at second order.
- Real solved psi (33 deg, n_passage 43, n_eta 21, rho = 1; kinematic verification only): the
  two images of every periodic node give one velocity, W_theta = W_m tan(beta) on both blade
  walls, rho S W_m = mdot' psi_eta at every node, the station mass flow equal to mdot' on the
  periodic stations and within O(deta^2) on the blade stations (shrinking with deta), a
  uniform flow recovered with W_theta / W_m = tan(alpha) (the sign convention), physically
  similar twins (c_ref, mdot' ~ c_ref) with identical velocities, inputs left untouched.
- Refusals: shapes, non-finite values, rho <= 0, a density other than the coefficients' own,
  objects of other grids, wrong types.

Tolerances name what they cover: round-off identities to 1e-13 relative or 1e-14 absolute
(measured 1e-16 to 1.2e-14); orders of convergence at least 1.9 (measured 1.97 to 2.03); the
blade-station mass-flow defect below 5e-3 at deta = 0.05 (measured 3.4e-3, at the first blade
station) and falling under eta refinement (ratios measured in that test's docstring).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from potential_flow_solver import boundary_conditions as bc
from potential_flow_solver import flow as fl
from potential_flow_solver import grid as gr
from potential_flow_solver import linear_solver as ls
from potential_flow_solver.coefficients import build_coefficients

import test_matrix_assembly as tma        # the assembly tests' real pipeline

TWO_PI = 2.0 * math.pi


# --- manufactured fields --------------------------------------------------------------------------

def brief_psi(M, H):
    """eta + 0.07 cos(1.3 m) sin(2 pi eta) + 0.03 sin(0.7 m) sin(4 pi eta): affine periodic, and
    0 / 1 on eta = 0 / 1 (a valid wall field too)."""
    psi = H + 0.07 * np.cos(1.3 * M) * np.sin(TWO_PI * H) + 0.03 * np.sin(0.7 * M) * np.sin(2 * TWO_PI * H)
    psi_m = -0.091 * np.sin(1.3 * M) * np.sin(TWO_PI * H) + 0.021 * np.cos(0.7 * M) * np.sin(2 * TWO_PI * H)
    psi_eta = 1.0 + 0.14 * math.pi * np.cos(1.3 * M) * np.cos(TWO_PI * H) + 0.12 * math.pi * np.sin(0.7 * M) * np.cos(2 * TWO_PI * H)
    return psi, psi_m, psi_eta


def shifted_psi(M, H):
    """Affine periodic, not constant on eta = 0, 1: exercises the m differences along the
    periodic lines."""
    psi = H + 0.07 * np.cos(1.3 * M) * np.cos(TWO_PI * H) + 0.03 * np.sin(0.7 * M) * np.sin(TWO_PI * H + 0.4)
    psi_m = -0.091 * np.sin(1.3 * M) * np.cos(TWO_PI * H) + 0.021 * np.cos(0.7 * M) * np.sin(TWO_PI * H + 0.4)
    psi_eta = 1.0 - 0.14 * math.pi * np.cos(1.3 * M) * np.sin(TWO_PI * H) + 0.06 * math.pi * np.sin(0.7 * M) * np.cos(TWO_PI * H + 0.4)
    return psi, psi_m, psi_eta


def wall_psi(M, H):
    """0 on eta = 0 and 1 on eta = 1, with different slopes on the two walls (not periodic)."""
    psi = H + 0.1 * np.cos(M) * np.sin(math.pi * H) + 0.05 * np.sin(1.7 * M) * H * (1 - H) * (1 + 2 * H)
    psi_eta = (1.0 + 0.1 * math.pi * np.cos(M) * np.cos(math.pi * H)
               + 0.05 * np.sin(1.7 * M) * (1 + 2 * H - 6 * H ** 2))
    return psi, 0.0 * M, psi_eta


def synthetic(k):
    """A grid of 4k + 1 stations on m in [0, 3] and 2k + 1 eta nodes, blade stations from 40 % to
    60 % of the stations; boundary_conditions' own classification (no geometry)."""
    n_m, n_eta = 4 * k + 1, 2 * k + 1
    m, eta = np.linspace(0.0, 3.0, n_m), np.linspace(0.0, 1.0, n_eta)
    f, l = round(0.4 * (n_m - 1)), round(0.6 * (n_m - 1))
    region = np.full(n_m, gr.PASSAGE)
    region[:f], region[l + 1:] = gr.INLET, gr.EXIT
    bcs = bc._boundary_conditions(None, m, eta, f, l, np.arange(f, l + 1), region)
    M, H = np.meshgrid(m, eta, indexing="ij")
    return bcs, M, H, float(m[1] - m[0]), float(eta[1] - eta[0])


def classes(bcs):
    n_m, n_eta = bcs.shape
    interior = np.zeros(bcs.shape, dtype=bool)
    interior[1:-1, 1:-1] = True
    up, down = np.asarray(bcs.upstream), np.asarray(bcs.downstream)
    off_far = ~(up | down)
    return {"interior": interior, "upstream": up, "downstream": down,
            "lower periodic": np.asarray(bcs.periodic_lower) & off_far,
            "upper periodic": np.asarray(bcs.periodic_upper) & off_far,
            "lower wall": np.asarray(bcs.lower_wall), "upper wall": np.asarray(bcs.upper_wall)}


KS = (10, 20, 40, 80)       # k = 5 (11 eta nodes for sin(4 pi eta)) is pre-asymptotic: orders 1.29 .. 1.86


@pytest.mark.parametrize("field, which, node_classes", [
    (brief_psi, "psi_eta", ("interior", "upstream", "downstream", "lower periodic", "upper periodic", "lower wall", "upper wall")),
    (brief_psi, "psi_m", ("interior", "upstream", "downstream")),
    (shifted_psi, "psi_m", ("interior", "upstream", "downstream", "lower periodic", "upper periodic")),
    (shifted_psi, "psi_eta", ("interior", "upstream", "downstream", "lower periodic", "upper periodic")),
    (wall_psi, "psi_eta", ("lower wall", "upper wall")),
])
def test_derivatives_converge_at_second_order_by_node_class(field, which, node_classes):
    errors = {c: [] for c in node_classes}
    hs = []
    for k in KS:
        bcs, M, H, dm, deta = synthetic(k)
        psi, exact_m, exact_eta = field(M, H)
        psi_m, psi_eta = fl._derivatives(psi, dm, deta, bcs)
        got, exact = (psi_m, exact_m) if which == "psi_m" else (psi_eta, exact_eta)
        cls = classes(bcs)
        for c in node_classes:
            errors[c].append(np.max(np.abs(got - exact)[cls[c]]))
        hs.append(dm)
    for c, e in errors.items():
        orders = [math.log(e[q] / e[q + 1]) / math.log(hs[q] / hs[q + 1]) for q in range(len(e) - 1)]
        assert min(orders) > 1.8, (c, e, orders)                 # measured 1.83 .. 2.40 from k = 10
        assert min(orders[-2:]) > 1.9, (c, e, orders)            # measured 1.94 .. 2.22 from k = 20
        assert e[-1] < 1e-3, (c, e)


def test_psi_m_is_exactly_zero_on_the_walls_and_tangency_follows():
    bcs, M, H, dm, deta = synthetic(10)
    psi, _, _ = brief_psi(M, H)
    psi_m, psi_eta = fl._derivatives(psi, dm, deta, bcs)
    walls = np.asarray(bcs.blade_wall)
    assert np.all(psi_m[walls] == 0.0)
    rho = 1.2 + 0.1 * np.sin(M) * np.cos(TWO_PI * H)
    S = (0.9 + 0.05 * np.cos(M[:, 0]))[:, None]
    tan_beta = (0.4 + 0.2 * np.sin(M)) * (1 - H) + (0.7 - 0.1 * np.cos(M)) * H
    W_m, W_theta = fl._velocities(psi_m, psi_eta, rho, 2.5, S, tan_beta)
    for mask in (np.asarray(bcs.lower_wall), np.asarray(bcs.upper_wall)):
        assert np.max(np.abs(W_theta - W_m * tan_beta)[mask]) <= 1e-15 * np.max(np.abs(W_theta[mask]))


def test_linear_psi_is_differentiated_exactly_with_the_periodic_jump_once():
    """psi = eta has psi_eta = 1 and psi_m = 0 everywhere: on a periodic line the continuation
    must add exactly one passage flow (a missing +1 gives 1 - 1/(2 deta) there, a doubled one
    1 + 1/(2 deta))."""
    bcs, M, H, dm, deta = synthetic(10)
    psi_m, psi_eta = fl._derivatives(H.copy(), dm, deta, bcs)
    assert np.max(np.abs(psi_eta - 1.0)) < 1e-13 and np.max(np.abs(psi_m)) < 1e-13
    periodic = np.flatnonzero(np.asarray(bcs.periodic_lower)[:, 0])
    for i in periodic:
        ghost = bcs.periodic_ghost(int(i), bcs.shape[1])
        assert ghost.constant == bc.PSI_PERIODIC_JUMP == 1.0


def test_periodic_images_give_one_velocity():
    bcs, M, H, dm, deta = synthetic(10)
    psi, _, _ = shifted_psi(M, H)
    psi_m, psi_eta = fl._derivatives(psi, dm, deta, bcs)
    periodic = np.flatnonzero(np.asarray(bcs.periodic_lower)[:, 0])
    rho = 1.0 + 0.2 * np.cos(M) * np.cos(TWO_PI * H)                     # periodic in eta
    S = (0.8 + 0.1 * np.sin(M[:, 0]))[:, None]
    tan_beta = np.broadcast_to(0.5 + 0.3 * np.sin(M[:, :1]), M.shape)     # equal at the two images
    W_m, W_theta = fl._velocities(psi_m, psi_eta, rho, 1.7, S, tan_beta)
    for a in (psi_m, psi_eta, W_m, W_theta):
        assert np.max(np.abs(a[periodic, 0] - a[periodic, -1])) < 1e-12


# --- the velocity formulas ---------------------------------------------------------------------------

def test_velocity_formulas_against_the_flux_form():
    """Eqs. 5-18, 5-19 against W_m = C psi_eta / S^2 and W_theta = (B psi_eta - C psi_m) / S, with
    B = mdot' tan(beta) / rho and C = mdot' S / rho (Eqs. 5-23, 5-24): another arrangement of the
    same algebra."""
    rng = np.random.default_rng(4)
    shape = (13, 9)
    psi_m, psi_eta = rng.standard_normal(shape), 1.0 + 0.3 * rng.standard_normal(shape)
    rho, S = 0.5 + rng.random(shape), (0.6 + rng.random(shape[0]))[:, None]
    tan_beta, mdot = rng.standard_normal(shape), 3.7
    W_m, W_theta = fl._velocities(psi_m, psi_eta, rho, mdot, S, tan_beta)
    B, C = mdot * tan_beta / rho, mdot * S / rho
    assert np.max(np.abs(W_m - C * psi_eta / S ** 2)) <= 1e-14 * np.max(np.abs(W_m))
    assert np.max(np.abs(W_theta - (B * psi_eta - C * psi_m) / S)) <= 1e-14 * np.max(np.abs(W_theta))
    assert np.max(np.abs(rho * S * W_m - mdot * psi_eta)) <= 1e-14 * np.max(np.abs(mdot * psi_eta))


def test_exact_velocities_are_recovered_at_second_order():
    """Smooth rho, S, tan(beta) and the brief's psi: the recovered W_m, W_theta against the
    velocities of the exact derivatives, at every node class."""
    errors, hs = {"W_m": [], "W_theta": []}, []
    for k in KS:
        bcs, M, H, dm, deta = synthetic(k)
        psi, exact_m, exact_eta = brief_psi(M, H)
        rho = 1.1 + 0.15 * np.sin(0.9 * M) * np.cos(TWO_PI * H)
        S = (0.85 + 0.1 * np.cos(0.7 * M[:, 0]))[:, None]
        tan_beta = (0.3 + 0.2 * np.sin(M)) * (1 - H) + (0.6 + 0.1 * np.cos(M)) * H
        psi_m, psi_eta = fl._derivatives(psi, dm, deta, bcs)
        W = fl._velocities(psi_m, psi_eta, rho, 2.0, S, tan_beta)
        W_exact = fl._velocities(exact_m, exact_eta, rho, 2.0, S, tan_beta)
        for name, got, want in zip(("W_m", "W_theta"), W, W_exact):
            errors[name].append(np.max(np.abs(got - want)))
        hs.append(dm)
    for name, e in errors.items():
        orders = [math.log(e[q] / e[q + 1]) / math.log(hs[q] / hs[q + 1]) for q in range(len(e) - 1)]
        assert min(orders) > 1.9, (name, e, orders)


# --- the real solved psi ---------------------------------------------------------------------------

def solved(case=(33.0, 43, 21)):
    grid, bcs, co, metrics, stencil, rows, res = tma.real(*case)
    psi = ls.solve_linear_system(res).psi
    return grid, bcs, co, metrics, psi


def test_real_solved_psi_kinematic_consistency():
    """KINEMATIC / NUMERICAL VERIFICATION ONLY (rho = 1, mdot' = 1, normalized mode): not a
    physical compressor flow."""
    grid, bcs, co, metrics, psi = solved()
    rho = np.ones(psi.shape)
    flow = fl.velocity_components(psi, rho, co, metrics, bcs)
    for name in ("psi_m", "psi_eta", "W_m", "W_theta", "W2", "W"):
        a = getattr(flow, name)
        assert a.shape == psi.shape and np.all(np.isfinite(a))
    assert np.array_equal(flow.W2, flow.W_m ** 2 + flow.W_theta ** 2) and np.array_equal(flow.W, np.sqrt(flow.W2))
    periodic = np.flatnonzero(np.asarray(bcs.periodic_lower)[:, 0])
    for a in (flow.W_m, flow.W_theta):
        assert np.max(np.abs(a[periodic, 0] - a[periodic, -1])) < 1e-13
    tan_beta = np.asarray(metrics.tan_beta)
    for mask in (np.asarray(bcs.lower_wall), np.asarray(bcs.upper_wall)):
        assert np.all(flow.psi_m[mask] == 0.0)
        assert np.max(np.abs(flow.W_theta - flow.W_m * tan_beta)[mask]) < 1e-14
    S = np.asarray(metrics.S)[:, None]
    assert np.max(np.abs(rho * S * flow.W_m - co.mass_flow_per_depth * flow.psi_eta)) < 1e-14
    assert np.all(flow.W_m > 0.0)                         # inlet -> exit for mdot' > 0
    mass = fl.station_mass_flow(flow, rho, co, metrics)
    blade = np.zeros(len(grid.m), dtype=bool)
    blade[grid.first_blade_i:grid.last_blade_i + 1] = True
    assert np.max(np.abs(mass[~blade] / co.mass_flow_per_depth - 1.0)) < 1e-13
    assert np.max(np.abs(mass[blade] / co.mass_flow_per_depth - 1.0)) < 5e-3


def test_blade_station_mass_flow_defect_falls_with_deta():
    """The defect of the trapezoidal station mass flow on the blade stations (one-sided wall
    differences) shrinks under eta refinement at fixed n_passage: along the blade (middle half
    of the blade stations) by 3.4 and 3.6 per halving (measured), close to deta^2; its maximum,
    at the first blade station beside the LE transition, by 2.4 and 3.0 (measured; 3.5 at the
    next halving)."""
    largest, middle = [], []
    for n_eta in (21, 41, 81):
        grid, bcs, co, metrics, psi = solved((33.0, 43, n_eta))
        rho = np.ones(psi.shape)
        mass = fl.station_mass_flow(fl.velocity_components(psi, rho, co, metrics, bcs), rho, co, metrics)
        d = np.abs(mass[grid.first_blade_i:grid.last_blade_i + 1] - 1.0)
        largest.append(d.max())
        middle.append(d[len(d) // 4:3 * len(d) // 4].max())
    assert middle[0] / middle[1] > 3.0 and middle[1] / middle[2] > 3.0, middle
    assert largest[0] / largest[1] > 2.0 and largest[1] / largest[2] > 2.0, largest


def test_uniform_flow_has_the_flow_angle_of_the_sign_convention():
    """psi = (y - m tan(alpha)) / pitch on the real grid nodes is a uniform flow at alpha: on the
    periodic stations away from the run ends the recovered W_theta / W_m is tan(alpha) (the
    central m difference of psi and metrics' central slope of the side line cancel), W_m =
    mdot' / (rho pitch): W_theta > 0 is the +y direction."""
    grid, bcs, co, metrics, _ = solved((33.0, 43, 21))
    alpha = math.radians(30.0)
    pitch = float(grid.boundaries.periodic_translation[1])
    y = np.asarray(grid.nodes)[..., 1]
    psi = (y - np.asarray(grid.m)[:, None] * math.tan(alpha)) / pitch
    flow = fl.velocity_components(psi, np.ones(psi.shape), co, metrics, bcs)
    f, l, n_m = grid.first_blade_i, grid.last_blade_i, len(grid.m)
    stations = list(range(2, f - 2)) + list(range(l + 3, n_m - 2))
    ratio = flow.W_theta[stations] / flow.W_m[stations]
    assert np.max(np.abs(ratio - math.tan(alpha))) < 1e-12
    assert np.max(np.abs(flow.W_m[stations] - 1.0 / pitch)) < 1e-12


def test_similar_twins_share_one_velocity_field():
    """c_ref = L with mdot' = L k gives the velocities of the normalized case with mdot' = k
    (coefficients.py's twins); with mdot' held, W scales as 1 / L."""
    grid, bcs, co, metrics, psi = solved((33.0, 29, 7))
    rho = np.ones(psi.shape)
    base = fl.velocity_components(psi, rho, build_coefficients(metrics, rho, 2.0), metrics, bcs)
    L = 0.137
    twin = fl.velocity_components(psi, rho, build_coefficients(metrics, rho, 2.0 * L, c_ref=L), metrics, bcs)
    held = fl.velocity_components(psi, rho, build_coefficients(metrics, rho, 2.0, c_ref=L), metrics, bcs)
    for name in ("W_m", "W_theta"):
        assert np.max(np.abs(getattr(twin, name) - getattr(base, name))) < 1e-13
        assert np.max(np.abs(getattr(held, name) * L - getattr(base, name))) < 1e-13
    assert np.max(np.abs(twin.psi_m * L - base.psi_m)) < 1e-13           # d/dm in the physical m


def test_the_density_given_is_used_and_nothing_is_changed():
    grid, bcs, co, metrics, psi = solved((33.0, 29, 7))
    rng = np.random.default_rng(2)
    rho = 1.0 + 0.2 * rng.random(psi.shape)
    rho[:, -1] = np.where(np.asarray(bcs.periodic_upper)[:, -1], rho[:, 0], rho[:, -1])   # periodic, as coefficients.py needs
    co_rho = build_coefficients(metrics, rho, 1.0)
    psi_copy, rho_copy = psi.copy(), rho.copy()
    flow = fl.velocity_components(psi, rho, co_rho, metrics, bcs)
    assert np.array_equal(psi, psi_copy) and np.array_equal(rho, rho_copy)
    unit = fl.velocity_components(psi, np.ones(psi.shape), co, metrics, bcs)
    assert np.max(np.abs(flow.W_m * rho - unit.W_m)) < 1e-13               # W ~ 1 / rho at fixed psi
    with pytest.raises(fl.FlowRecoveryError, match="density"):
        fl.velocity_components(psi, np.ones(psi.shape), co_rho, metrics, bcs)  # not the coefficients' density
    with pytest.raises(ValueError):
        flow.W_m[0, 0] = 1.0


def test_refusals():
    grid, bcs, co, metrics, psi = solved((33.0, 29, 7))
    other_grid, other_bcs, other_co, other_metrics, other_psi = solved((33.0, 43, 11))
    rho = np.ones(psi.shape)
    bad = psi.copy()
    bad[3, 3] = np.nan
    cases = [
        (lambda: fl.velocity_components(psi[:-1], rho, co, metrics, bcs), fl.FlowRecoveryError, "shape"),
        (lambda: fl.velocity_components(bad, rho, co, metrics, bcs), fl.FlowRecoveryError, "not finite"),
        (lambda: fl.velocity_components(psi, rho[:, :-1], co, metrics, bcs), fl.FlowRecoveryError, "shape"),
        (lambda: fl.velocity_components(psi, np.where(bad == bad, rho, np.inf), co, metrics, bcs), fl.FlowRecoveryError, "not finite"),
        (lambda: fl.velocity_components(psi, -rho, co, metrics, bcs), fl.FlowRecoveryError, "positive"),
        (lambda: fl.velocity_components(psi, 0.0 * rho, co, metrics, bcs), fl.FlowRecoveryError, "positive"),
        (lambda: fl.velocity_components(psi, rho, co, metrics, other_bcs), fl.FlowRecoveryError, "grid"),
        (lambda: fl.velocity_components(psi, rho, other_co, metrics, bcs), fl.FlowRecoveryError, "coefficient"),
        (lambda: fl.velocity_components(psi, rho, None, metrics, bcs), TypeError, "CoefficientFields"),
        (lambda: fl.velocity_components(psi, rho, co, None, bcs), TypeError, "GridMetrics"),
        (lambda: fl.velocity_components(psi, rho, co, metrics, None), TypeError, "BoundaryConditions"),
        (lambda: fl.station_mass_flow(None, rho, co, metrics), TypeError, "FlowField"),
    ]
    for call, error, match in cases:
        with pytest.raises(error, match=match):
            call()


# --- inlet closure: inlet Mach number -> inlet state -> mdot' (2026-11-13) ------------------------------------

from potential_flow_solver import eos as _eos                       # noqa: E402
from potential_flow_solver.config import FlowConditions             # noqa: E402

INLET_AIR = _eos.CaloricallyPerfectGas(gamma=1.4, gas_constant=287.05)
INLET_PT, INLET_TT = 101325.0, 288.15


def inlet_conditions(mach, angle_deg=45.0, omega=0.0, **rotating):
    return FlowConditions(inlet_mach=mach, inlet_angle_deg=angle_deg, exit_angle_guess_deg=18.0,
                          inlet_absolute_total_pressure=INLET_PT, inlet_absolute_total_temperature=INLET_TT,
                          angular_speed=omega, **rotating)


@pytest.mark.parametrize("mach", [0.0, 0.3, 0.65])
def test_inlet_state_from_the_inlet_mach_number(mach):
    """M1 -> T1, p1, rho1, a1, W1 -> W_m1 -> mdot' against the closed-form perfect-gas
    relations written out here (flow.py writes none of them: it inverts eos's W / a(W))."""
    pitch, beta = 0.1, math.radians(45.0)
    s = fl.inlet_state(inlet_conditions(mach), INLET_AIR, inlet_pitch=pitch)
    g, R = 1.4, 287.05
    T1 = INLET_TT / (1.0 + 0.5 * (g - 1.0) * mach ** 2)
    p1 = INLET_PT * (T1 / INLET_TT) ** (g / (g - 1.0))
    rho1 = p1 / (R * T1)
    a1 = math.sqrt(g * R * T1)
    W1 = mach * a1
    expected = dict(static_temperature=T1, static_pressure=p1, static_density=rho1, sound_speed=a1, velocity=W1,
                    meridional_velocity=W1 * math.cos(beta), tangential_velocity=W1 * math.sin(beta),
                    dynamic_pressure=0.5 * rho1 * W1 ** 2, mass_flow_per_depth=rho1 * W1 * math.cos(beta) * pitch)
    for name, value in expected.items():
        assert getattr(s, name) == pytest.approx(value, rel=1e-14, abs=0.0), name   # measured: equal to the last bit
    assert (s.mach, s.flow_angle, s.pitch, s.total_pressure, s.total_temperature) == (mach, beta, pitch, INLET_PT, INLET_TT)
    if mach == 0.0:
        assert s.velocity == 0.0 and s.mass_flow_per_depth == 0.0 and s.dynamic_pressure == 0.0
    else:
        assert abs(s.velocity / s.sound_speed - mach) <= 4 * np.spacing(mach)
    # the static state is eos's own at W1
    state = _eos.static_state_from_velocity(s.velocity, INLET_AIR, _eos.blade_row_reference(
        _eos.TotalState(INLET_PT, INLET_TT), INLET_AIR))
    assert (s.static_temperature, s.static_pressure, s.static_density, s.sound_speed) == \
        (state.temperature, state.pressure, state.density, state.sound_speed)


def test_passage_mass_flow_and_the_angle_convention():
    assert fl.passage_mass_flow(1.25, 40.0, 0.05) == 2.5                 # rho1 W_m1 S1
    axial = fl.inlet_state(inlet_conditions(0.5, angle_deg=0.0), INLET_AIR, inlet_pitch=0.1)
    turned = fl.inlet_state(inlet_conditions(0.5, angle_deg=-30.0), INLET_AIR, inlet_pitch=0.1)
    assert axial.tangential_velocity == 0.0 and axial.meridional_velocity == axial.velocity
    assert turned.tangential_velocity < 0.0                               # angle from m, positive toward +y
    assert turned.mass_flow_per_depth == pytest.approx(axial.mass_flow_per_depth * math.cos(math.radians(30.0)), rel=1e-14)


def test_inlet_state_refusals():
    cases = [
        (lambda: fl.inlet_state(inlet_conditions(0.3, omega=100.0, stream_surface_radius=0.3,
                                                 inlet_absolute_tangential_velocity=10.0), INLET_AIR, inlet_pitch=0.1),
         "rotating"),
        (lambda: fl.inlet_state(inlet_conditions(None), INLET_AIR, inlet_pitch=0.1), "Mach"),
        (lambda: fl.inlet_state(inlet_conditions(-0.1), INLET_AIR, inlet_pitch=0.1), "Mach"),
        (lambda: fl.inlet_state(inlet_conditions(float("nan")), INLET_AIR, inlet_pitch=0.1), "Mach"),
        (lambda: fl.inlet_state(inlet_conditions(True), INLET_AIR, inlet_pitch=0.1), "Mach"),
        (lambda: fl.inlet_state(inlet_conditions(0.3, angle_deg=90.0), INLET_AIR, inlet_pitch=0.1), "angle"),
        (lambda: fl.inlet_state(inlet_conditions(0.3), INLET_AIR, inlet_pitch=0.0), "pitch"),
        (lambda: fl.inlet_state(inlet_conditions(0.3), INLET_AIR, inlet_pitch=float("nan")), "pitch"),
    ]
    for call, match in cases:
        with pytest.raises(fl.InletStateError, match=match):
            call()


def test_inlet_closure_writes_no_gas_formula():
    """flow.py reads no gas property: gamma, R, cp and the isentropic relations stay eos.py's."""
    import ast
    from pathlib import Path
    tree = ast.parse(Path(fl.__file__).read_text(encoding="utf-8"))
    attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    assert not {"gamma", "gas_constant", "cp", "cv", "isentropic_pressure", "density_of"} & attributes
    assert "_eos.static_state_from_velocity" in Path(fl.__file__).read_text(encoding="utf-8")


def test_flow_is_kinematic_only_the_static_state_is_eos():
    """The obsolete static-state stub of flow.py is gone (eos.static_state_from_velocity owns
    W -> h, T, p, rho, a) and nothing in the package, the tests or the scripts refers to it."""
    import re
    from pathlib import Path
    assert not hasattr(fl, "static_state")
    root = Path(fl.__file__).resolve().parents[1]
    pattern = re.compile(r"(\bflow|\bfl)\.static_state\b|def static_state\(")
    hits = [f"{path.relative_to(root)}:{n}" for folder in ("potential_flow_solver", "tests", "scripts")
            for path in sorted((root / folder).glob("*.py"))
            for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1) if pattern.search(line)]
    assert hits == []


# --- psi_m by m region: the LE / TE transition fix (2026-11-13) ----------------------------------------------
# The three regions of metrics.py: upstream periodic 0 .. f - 1, blade f .. l, downstream periodic
# l + 1 .. n_m - 1; inside each a central difference, at each end the three-point one-sided one (module
# docstring of flow.py). These tests name every transition stencil, so restoring a central difference at
# f - 1, f, l or l + 1, turning a one-sided stencil the wrong way, moving it one station, lowering it to
# first order or dropping the wall zero each fails one of them.

from potential_flow_solver import metrics as _metrics          # noqa: E402


def regime_psi(M, H, f, l):
    """A field with a different quadratic in m in each of the three regions and a jump at each
    region boundary (as two grid maps meeting at the LE and TE give), eta-dependent; its exact
    psi_m per region."""
    a = (0.2 + 0.1 * H, 1.5 + 0.3 * np.sin(TWO_PI * H), -0.4 + 0.2 * H)
    b = (0.05, -0.12 + 0.04 * H, 0.08)
    c = (H, H + 0.5, H - 0.3)                                    # the jumps between regions
    region = np.zeros(M.shape, dtype=int)
    region[f:l + 1], region[l + 1:] = 1, 2
    psi = np.choose(region, [c[k] + a[k] * M + b[k] * M ** 2 for k in range(3)])
    exact = np.choose(region, [a[k] + 2.0 * b[k] * M for k in range(3)])
    return psi, exact


def regions_of(bcs):
    blade = np.flatnonzero(np.asarray(bcs.lower_wall)[:, 0])
    return int(blade[0]), int(blade[-1])


def test_psi_m_never_crosses_a_region_boundary():
    """A quadratic in m per region, with jumps between the regions: psi_m is exact (round-off) at
    every node off the walls, the four transition stations included; the old central difference
    across the edges is off by the jump there."""
    bcs, M, H, dm, deta = synthetic(10)
    f, l = regions_of(bcs)
    psi, exact = regime_psi(M, H, f, l)
    psi_m, _ = fl._derivatives(psi, dm, deta, bcs)
    walls = np.asarray(bcs.blade_wall)
    assert np.max(np.abs(psi_m - exact)[~walls]) < 1e-10              # measured ~1e-13
    assert np.all(psi_m[walls] == 0.0)
    central = (psi[2:] - psi[:-2]) / (2.0 * dm)                        # the pre-fix rule at i = 1 .. n_m - 2
    for i in (f - 1, f, l, l + 1):
        assert np.max(np.abs(central[i - 1, 1:-1] - exact[i, 1:-1])) > 1.0, i   # the jump / (2 dm) shows


def test_psi_m_transition_stencils_by_name():
    """Each transition station's own one-sided stencil (bit for bit), the walls zero, and the
    central difference at every other station; a cubic field, so a stencil one station off
    differs from the central one there by O(dm^2), not by round-off."""
    bcs, M, H, dm, deta = synthetic(10)
    f, l = regions_of(bcs)
    psi = H + 0.3 * M ** 3 - 0.2 * M ** 2 * H + 0.1 * M * np.sin(TWO_PI * H)
    psi_m, _ = fl._derivatives(psi, dm, deta, bcs)
    n_m = psi.shape[0]
    rows = slice(1, -1)                                                # the interior nodes of a blade station
    backward = lambda i: (3.0 * psi[i] - 4.0 * psi[i - 1] + psi[i - 2]) / (2.0 * dm)    # noqa: E731
    forward = lambda i: (-3.0 * psi[i] + 4.0 * psi[i + 1] - psi[i + 2]) / (2.0 * dm)    # noqa: E731
    central = lambda i: (psi[i + 1] - psi[i - 1]) / (2.0 * dm)                          # noqa: E731
    assert np.array_equal(psi_m[f - 1], backward(f - 1))               # upstream periodic, all nodes
    assert np.array_equal(psi_m[f, rows], forward(f)[rows])            # blade side
    assert np.array_equal(psi_m[l, rows], backward(l)[rows])           # blade side
    assert np.array_equal(psi_m[l + 1], forward(l + 1))                # downstream periodic, all nodes
    for i in (f, l):
        assert psi_m[i, 0] == 0.0 and psi_m[i, -1] == 0.0              # the walls take precedence
    assert np.array_equal(psi_m[0], forward(0)) and np.array_equal(psi_m[-1], backward(n_m - 1))   # far stations
    for i in set(range(1, n_m - 1)) - {f - 1, f, l, l + 1}:
        expected = central(i)
        if f <= i <= l:
            expected = np.array(expected)
            expected[[0, -1]] = 0.0
        assert np.array_equal(psi_m[i], expected), i
    for i in (f - 1, f, l, l + 1):                                     # a central or a misplaced stencil is visible
        assert np.max(np.abs(central(i)[rows] - psi_m[i, rows])) > 1e-4, i


def test_psi_m_at_a_transition_station_reads_its_own_side_only():
    """Perturbing every node of the other side of an edge changes no transition station's psi_m on
    this side, and vice versa (the old central differences read across)."""
    bcs, M, H, dm, deta = synthetic(10)
    f, l = regions_of(bcs)
    psi, _ = regime_psi(M, H, f, l)
    base, _ = fl._derivatives(psi, dm, deta, bcs)
    for side, stations in ((slice(0, f), (f, l)), (slice(f, l + 1), (f - 1, l + 1)), (slice(l + 1, None), (f, l))):
        changed = np.array(psi)
        changed[side] += 0.37 + 0.2 * np.cos(3.0 * M[side])
        psi_m, _ = fl._derivatives(changed, dm, deta, bcs)
        for i in stations:
            assert np.array_equal(psi_m[i], base[i]), (side, i)


def test_psi_m_is_metrics_own_region_derivative():
    """flow.py takes psi_m with metrics.py's region derivative itself (one convention, two
    users): identical to metrics._d_dm over metrics._runs, the walls set to zero."""
    bcs, M, H, dm, deta = synthetic(20)
    f, l = regions_of(bcs)
    psi, _, _ = shifted_psi(M, H)
    psi_m, _ = fl._derivatives(psi, dm, deta, bcs)
    expected = _metrics._d_dm(psi, dm, _metrics._runs(psi.shape[0], f, l))
    expected[np.asarray(bcs.blade_wall)] = 0.0
    assert np.array_equal(psi_m, expected)


def test_transition_psi_m_is_second_order():
    """At f - 1, f, l, l + 1 (off the walls) the error of a smooth field falls by about 4 per
    halving of dm: the one-sided stencils are second order. The thresholds of the node-class test
    above (> 1.8 throughout, > 1.9 on the last two halvings); measured 1.84 .. 2.05, the 1.84 at
    l + 1 on the first halving (k = 10, pre-asymptotic), 1.93 and 1.97 after it."""
    errors, hs = [], []
    for k in KS:
        bcs, M, H, dm, deta = synthetic(k)
        f, l = regions_of(bcs)
        psi, exact, _ = brief_psi(M, H)
        psi_m, _ = fl._derivatives(psi, dm, deta, bcs)
        errors.append([np.max(np.abs(psi_m[i, 1:-1] - exact[i, 1:-1])) for i in (f - 1, f, l, l + 1)])
        hs.append(dm)
    errors = np.array(errors)
    orders = np.log(errors[:-1] / errors[1:]) / np.log(np.array(hs[:-1]) / np.array(hs[1:]))[:, None]
    assert np.all(orders > 1.8), orders                       # measured 1.84 .. 2.05
    assert np.all(orders[-2:] > 1.9), orders                  # measured 1.93 .. 2.02
    assert np.all(errors[-1] < 1e-4), errors[-1]


def test_uniform_flow_is_now_exact_at_the_transition_periodic_stations():
    """psi = (y - m tan(alpha)) / pitch on the real grid: at f - 1 and l + 1 flow.py's psi_m and
    metrics.py's tan(beta) now come from one one-sided stencil of one side line, so W_theta / W_m
    = tan(alpha) there exactly as at the other periodic stations (before the fix psi_m was central
    across the edge and the ratio was off)."""
    grid, bcs, co, metrics, _ = solved((33.0, 43, 21))
    alpha = math.radians(30.0)
    pitch = float(grid.boundaries.periodic_translation[1])
    psi = (np.asarray(grid.nodes)[..., 1] - np.asarray(grid.m)[:, None] * math.tan(alpha)) / pitch
    flow = fl.velocity_components(psi, np.ones(psi.shape), co, metrics, bcs)
    f, l = grid.first_blade_i, grid.last_blade_i
    for i in (f - 1, l + 1):
        assert np.max(np.abs(flow.W_theta[i] / flow.W_m[i] - math.tan(alpha))) < 1e-12, i

