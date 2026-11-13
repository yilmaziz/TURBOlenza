"""
Tests for turbolenza.b2b.solver (the density loop at a fixed discharge angle, and the
density + Kutta coupled loop of solve) and convergence (the change norms and the stop tests).

    solve_at_exit_angle: DENSITY-COUPLED / FIXED-DISCHARGE-ANGLE SOLUTION, NOT FINAL AUNGIER
    KUTTA-CONVERGED SOLUTION; its tests below are those of 2026-11-13, unchanged, plus a lock of
    the Kutta residual it now reports as a diagnostic.
    solve: the coupled solution (Kutta condition at the last blade station, kutta.py), verified
    on the constructed case, not yet validated.

What is checked
- convergence.change_norm, velocity_change and density_converged on numbers worked out by
  hand, the first pass (no previous velocity) never converged.
- One density pass of the solver equals independent calls of the verified modules, array for
  array (coefficients, matrix, psi, velocities, static state, target and next density): the
  orchestration changes no mathematics.
- The relaxation is read from config.iteration.density_relaxation: lambda = 1 gives the EOS
  density itself (Aungier's direct replacement), lambda = 0.5 the midpoint, any lambda the
  relax_density formula.
- A dimensional stationary verification case (air, Pt 101 325 Pa, Tt 288.15 K, 33 deg,
  n_passage 43, n_eta 21, chord 0.1 m, a 45 deg inlet at a chosen far-inlet Mach number 0.3;
  the mass flow derived below): converged; both changes of the last loop pass within the
  tolerance; the reported density, psi, flow and static state rebuilt exactly from the
  reported density by one independent pass; the far-inlet Mach number and density those of
  the derivation; the changes falling monotonically; lambda = 1 converging to the same field;
  the discharge angle unchanged (no Kutta).
- Stops: a pass limit without convergence raises with the last pass attached, a relative Mach
  number of 1 raises TransonicFlowError, inputs outside the scope are refused (no chord, a
  rotating row, a dimensional cascade, a start density of the wrong shape, no pass allowed).
- solver.py calls modules only: no gas, spacing or coefficient attribute is read there.

The verification case is a constructed low-Mach case, not the NACA test case: its mass flow
comes from the far-inlet state at M = 0.3 by the perfect-gas relations (T = Tt / (1 + (gamma -
1) / 2 M^2), W = M sqrt(gamma R T), rho from eos at that W), the axial velocity W_m = W cos(45
deg) (angles from m, positive toward +y, as in geometry.py) and the far-inlet passage width
S = pitch c_ref: mdot' = rho W cos(alpha) S = 8.382 kg/(m s).

The coupled case (solve) is the same case with the discharge angle left free: first angles
kappa_te - 5, kappa_te and kappa_te + 10 deg, tolerances 1e-8 on the density change, the
velocity change (velocity_tolerance None: the density tolerance) and eps_K, the undamped Kutta
correction (kutta_relaxation 1.0, the project baseline). Measured (with the region-wise psi_m
of flow.py, 2026-11-13): 143, 138 and 138 passes, all three at 23.5022008 deg within 1.3e-7
deg, the densities within 5.3e-10 and the speeds within 9.1e-9 of each other, eps_K of the
reported flow 7.8e-9 to 8.7e-9; |R_K| falls every pass (ratios 0.873 to 0.902, the last 20 at
0.8767 to 0.8769) without a change of sign, positive from below the converged angle and
negative from above. Before that fix (psi_m central across the LE and TE transitions): 140,
135, 134 passes, 23.4684347 deg, the last 20 ratios 0.8735 to 0.8736; the fixed-angle run 28
passes and R_K 48.3785 m/s (now 27 and 48.3464). The fix changes the velocity at the four
transition stations, hence the EOS density there, hence the converged field everywhere.

Tolerances name the measurement they cover: equality where the same calls are repeated;
the far-inlet Mach number to 1e-4 on the station mean and 1e-3 pointwise (measured 1.7e-5 and
4.1e-4: the far station sits only one chord ahead of the blade, whose potential field has not
died out there), the far-inlet density to 1e-5 mean and 1e-4 pointwise (measured 1.5e-6,
3.6e-5); lambda = 1 against lambda = 0.5 to 1e-6 (both converged to density_tolerance 1e-8;
measured 8.0e-9 in rho, 7.2e-9 in W).
"""

from __future__ import annotations

import ast
import math
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from turbolenza.b2b import config as cf
from turbolenza.b2b import convergence, eos, kutta
from turbolenza.b2b import solver as sv
from turbolenza.b2b.boundary_conditions import build_boundary_conditions
from turbolenza.b2b.coefficients import build_coefficients
from turbolenza.b2b.flow import velocity_components
from turbolenza.b2b.geometry import CascadeSpec, make_cascade
from turbolenza.b2b.grid import make_grid, rebuild_exit, validate_uniform_grid_topology
from turbolenza.b2b.linear_solver import solve_linear_system
from turbolenza.b2b.matrix_assembly import assemble_from_coefficients
from turbolenza.b2b.metrics import build_metrics

AIR = eos.CaloricallyPerfectGas(gamma=1.4, gas_constant=287.05)
PT, TT = 101325.0, 288.15
C_REF = 0.1                          # m
INLET_DEG = 45.0
TOL = 1e-8


@lru_cache(maxsize=None)
def cascade():
    return make_cascade(CascadeSpec(blade_type="NACA65", solidity=1.0, stagger_deg=33.0, cl0=1.2, t_max_over_c=0.10,
                                    surface_continuity="C2"))


def inlet_mass_flow(mach: float) -> tuple[float, float, float]:
    """mdot' of a far-inlet relative Mach number (module docstring): (mdot', W_in, rho_in)."""
    T = TT / (1.0 + 0.5 * (AIR.gamma - 1.0) * mach ** 2)
    W = mach * math.sqrt(AIR.gamma * AIR.gas_constant * T)
    state = eos.static_state_from_velocity(W, AIR, eos.blade_row_reference(eos.TotalState(PT, TT), AIR))
    assert W / state.sound_speed == pytest.approx(mach, rel=1e-14)
    pitch = float(cascade().active.pitch) * C_REF
    return state.density * W * math.cos(math.radians(INLET_DEG)) * pitch, W, state.density


def config(relaxation=None, tolerance=TOL, passes=200, omega=0.0, kutta_tolerance=1.0, kutta_passes=1, exit_offset_deg=0.0,
           velocity_tolerance=None, kutta_relaxation=None, stop_at_sonic=True, **flow):
    iteration = dict(density_tolerance=tolerance, max_density_iterations=passes, kutta_tolerance=kutta_tolerance,
                     max_kutta_iterations=kutta_passes, velocity_tolerance=velocity_tolerance, stop_at_sonic=stop_at_sonic)
    if relaxation is not None:
        iteration["density_relaxation"] = relaxation
    if kutta_relaxation is not None:
        iteration["kutta_relaxation"] = kutta_relaxation
    conditions = dict(inlet_mach=0.3, inlet_angle_deg=INLET_DEG,
                      exit_angle_guess_deg=math.degrees(cascade().kappa_te) + exit_offset_deg,
                      inlet_absolute_total_pressure=PT, inlet_absolute_total_temperature=TT, angular_speed=omega)
    conditions.update(flow)
    return cf.SolverConfig(grid=cf.GridSettings(n_passage=43, n_eta=21), flow=cf.FlowConditions(**conditions), gas=AIR,
                           iteration=cf.IterationSettings(**iteration))


@lru_cache(maxsize=None)
def solved(relaxation=None, mach=0.3):
    mdot, _, _ = inlet_mass_flow(mach)
    return sv.solve_density_coupled(cascade(), config(relaxation), mass_flow_per_depth=mdot, c_ref=C_REF)


@lru_cache(maxsize=None)
def grid():
    c = config()
    return make_grid(cascade(), c.grid, math.radians(c.flow.inlet_angle_deg), math.radians(c.flow.exit_angle_guess_deg),
                     require_ss_ps=True)


# --- convergence.py -------------------------------------------------------------------------------

def test_change_norm_by_hand():
    old = np.array([[1.0, 2.0], [4.0, -0.5]])
    new = np.array([[1.1, 2.0], [3.0, -0.5]])
    assert convergence.change_norm(new, old) == pytest.approx(1.0 / 4.0, rel=1e-15)     # max |d| 1 over max |old| 4
    assert convergence.change_norm(old, old) == 0.0
    assert convergence.velocity_change(np.array([3.0, 0.0]), np.array([4.0, 1.0]), np.array([3.0, 0.0]),
                                       np.array([0.0, 1.0])) == pytest.approx(4.0 / 3.0, rel=1e-15)  # |dW| 4 over max |W| 3
    assert convergence.velocity_change(np.array([3.0, 1.0]), np.array([4.0, 2.0]), np.array([3.0, 1.0]),
                                       np.array([4.0, 0.0])) == pytest.approx(2.0 / 5.0, rel=1e-15)  # over |W| 5, not W_m 3
    for call in (lambda: convergence.change_norm(np.ones(2), np.ones(3)), lambda: convergence.change_norm(np.ones(2), np.zeros(2)),
                 lambda: convergence.change_norm(np.array([np.nan]), np.ones(1))):
        with pytest.raises(ValueError):
            call()


def test_density_converged_needs_both_changes_and_a_previous_velocity():
    settings = cf.IterationSettings(density_tolerance=1e-3, max_density_iterations=5, kutta_tolerance=1.0,
                                    max_kutta_iterations=1)
    h = convergence.History()
    assert not convergence.density_converged(h, settings)
    h.density_change.append(1e-9)
    h.velocity_change.append(float("nan"))                  # pass 0: no previous velocity
    assert not convergence.density_converged(h, settings)
    for d, w, expected in ((1e-4, 1e-2, False), (1e-2, 1e-4, False), (1e-3, 1e-3, True), (1e-4, 1e-4, True)):
        h.density_change.append(d)
        h.velocity_change.append(w)
        assert convergence.density_converged(h, settings) is expected


# --- one pass ----------------------------------------------------------------------------------------

def test_one_pass_is_exactly_the_module_calls():
    """rho -> coefficients -> F Psi = Q -> psi -> W -> EOS target -> relaxed next, through the
    solver and through independent calls: the same arrays."""
    mdot, _, _ = inlet_mass_flow(0.3)
    g = grid()
    c = config()
    ref = eos.blade_row_reference(eos.TotalState(PT, TT), AIR)
    rho = np.array(eos.initial_density(AIR, ref, shape=(len(g.m), len(g.eta))))
    p = sv._density_pass(rho, sv._context(g, c, mdot, C_REF))
    metrics, bcs = build_metrics(g), build_boundary_conditions(g)
    co = build_coefficients(metrics, rho, mdot, c_ref=C_REF)
    lin = solve_linear_system(assemble_from_coefficients(co, metrics, bcs))
    flow = velocity_components(lin.psi, rho, co, metrics, bcs)
    state = eos.static_state_from_velocity(flow.W, AIR, ref)
    nxt = eos.relax_density(rho, state.density, relaxation=0.5)
    for name in "ABCDE":
        assert np.array_equal(getattr(p.coefficients, name), getattr(co, name))
    assert np.array_equal(p.linear.active_solution, lin.active_solution) and np.array_equal(p.linear.psi, lin.psi)
    for name in ("psi_m", "psi_eta", "W_m", "W_theta", "W"):
        assert np.array_equal(getattr(p.flow, name), getattr(flow, name))
    for name in ("temperature", "pressure", "density", "enthalpy", "sound_speed"):
        assert np.array_equal(getattr(p.state, name), getattr(state, name))
    assert np.array_equal(p.density_target, state.density) and np.array_equal(p.density_next, nxt)
    assert np.array_equal(p.mach, flow.W / state.sound_speed)
    assert p.density_resident is rho or np.array_equal(p.density_resident, rho)


@pytest.mark.parametrize("lam", [1.0, 0.5, 0.3])
def test_the_relaxation_is_read_from_the_config(lam):
    mdot, _, _ = inlet_mass_flow(0.3)
    g = grid()
    ref = eos.blade_row_reference(eos.TotalState(PT, TT), AIR)
    rho = np.array(eos.initial_density(AIR, ref, shape=(len(g.m), len(g.eta))))
    p = sv._density_pass(rho, sv._context(g, config(lam), mdot, C_REF))
    if lam == 1.0:
        assert np.array_equal(p.density_next, p.density_target)               # Aungier: direct replacement
    elif lam == 0.5:
        assert np.array_equal(p.density_next, (rho + p.density_target) / 2.0)  # the project baseline
    assert np.array_equal(p.density_next, eos.relax_density(rho, p.density_target, relaxation=lam))
    assert sv._context(g, config(), mdot, C_REF).relaxation == eos.DENSITY_RELAXATION == 0.5   # the default


# --- the dimensional stationary verification case ---------------------------------------------------------

def test_physical_case_converges_with_a_self_consistent_final_state():
    sol = solved()
    assert sol.converged and sol.limitation == sv.LIMITATION and "NOT FINAL AUNGIER KUTTA-CONVERGED" in sol.limitation
    loop = [p for p in sol.passes if not p.final]
    final = sol.passes[-1]
    assert final.final and len(loop) == sol.iterations and sol.passes[:-1] == tuple(loop)
    assert loop[-1].density_change <= TOL and loop[-1].velocity_change <= TOL
    assert math.isnan(loop[0].velocity_change)                        # no previous velocity on pass 0
    changes = [p.density_change for p in loop]
    assert all(b < a for a, b in zip(changes, changes[1:]))           # monotonic
    # the reported state is one pass: rebuild it from the reported density
    again = sv._density_pass(np.array(sol.density), sv._context(sol.grid, config(), sol.mass_flow_per_depth, C_REF))
    assert np.array_equal(again.linear.psi, sol.psi) and sol.psi is sol.linear.psi
    for name in ("W_m", "W_theta", "W"):
        assert np.array_equal(getattr(again.flow, name), getattr(sol.flow, name))
    for name in ("temperature", "pressure", "density"):
        assert np.array_equal(getattr(again.state, name), getattr(sol.static_state, name))
    assert np.array_equal(again.density_target, sol.density_target) and np.array_equal(sol.mach, sol.flow.W / sol.static_state.sound_speed)
    assert final.target_mismatch == convergence.change_norm(sol.density_target, sol.density) < 2.5 * TOL
    # the physics of the constructed case: the far inlet is the derived inlet state
    _, W_in, rho_in = inlet_mass_flow(0.3)
    assert abs(np.mean(sol.mach[0]) / 0.3 - 1.0) < 1e-4                   # measured 1.7e-5
    assert np.max(np.abs(sol.mach[0] / 0.3 - 1.0)) < 1e-3                 # measured 4.1e-4 across the station
    assert abs(np.mean(sol.density[0]) / rho_in - 1.0) < 1e-5             # measured 1.5e-6
    assert np.max(np.abs(sol.density[0] / rho_in - 1.0)) < 1e-4           # measured 3.6e-5
    assert np.max(sol.mach) < 1.0 and np.all(sol.static_state.temperature > 0) and np.all(sol.static_state.pressure > 0)
    assert np.all(np.isfinite(sol.psi)) and final.linear_residual < 1e-10 and sol.linear.relative_residual < 1e-13
    assert final.mass_flow_error_max < 5e-3                           # the O(deta^2) blade-station defect, not forced
    # the discharge angle was not touched: the grid of the solution is the one built at it
    assert sol.grid.exit_angle == pytest.approx(math.radians(config().flow.exit_angle_guess_deg), rel=0, abs=0)
    assert len(sol.history.density_change) == len(sol.passes) and set(sol.history.kutta_pass) == {0}


def test_aungier_direct_replacement_converges_to_the_same_field():
    half, direct = solved(), solved(1.0)
    assert direct.converged and direct.density_relaxation == 1.0 and half.density_relaxation == 0.5
    assert np.max(np.abs(direct.density / half.density - 1.0)) < 1e-6
    assert np.max(np.abs(direct.flow.W / half.flow.W - 1.0)) < 1e-6


def test_the_loop_starts_from_the_relative_total_density_and_feeds_the_relaxed_density_forward():
    """Pass 0's resident density is eos.initial_density (Pt / (R Tt) here); pass 1's resident is
    relax_density(rho(0), rho_target(1)), never rho_target(1) itself."""
    mdot, _, _ = inlet_mass_flow(0.3)
    g = grid()
    with pytest.raises(sv.DensityNotConvergedError) as info:
        sv.solve_at_exit_angle(g, config(tolerance=1e-14, passes=2), mass_flow_per_depth=mdot, c_ref=C_REF)
    partial = info.value.solution
    rho0 = PT / (AIR.gas_constant * TT)
    first = partial.passes[0]
    assert first.density_min == pytest.approx(rho0, rel=1e-15) and first.density_max == pytest.approx(rho0, rel=1e-15)
    ref = eos.blade_row_reference(eos.TotalState(PT, TT), AIR)
    start = np.array(eos.initial_density(AIR, ref, shape=(len(g.m), len(g.eta))))
    target = sv._density_pass(start, sv._context(g, config(), mdot, C_REF)).density_target
    assert np.array_equal(partial.density, eos.relax_density(start, target))
    assert not np.array_equal(partial.density, target)


def test_the_first_pass_is_never_converged():
    """A tolerance every change meets still needs two passes: pass 0 has no previous velocity."""
    mdot, _, _ = inlet_mass_flow(0.3)
    sol = sv.solve_density_coupled(cascade(), config(tolerance=10.0), mass_flow_per_depth=mdot, c_ref=C_REF)
    assert sol.converged and sol.iterations == 2 and len(sol.passes) == 3 and sol.passes[-1].final


def test_no_convergence_within_the_pass_limit_raises_with_the_last_pass():
    mdot, _, _ = inlet_mass_flow(0.3)
    with pytest.raises(sv.DensityNotConvergedError, match="did not converge in 3 passes") as info:
        sv.solve_density_coupled(cascade(), config(tolerance=1e-14, passes=3), mass_flow_per_depth=mdot, c_ref=C_REF)
    partial = info.value.solution
    assert partial is not None and not partial.converged and partial.iterations == 3
    assert len(partial.passes) == 3 and not any(p.final for p in partial.passes)


def test_a_sonic_node_stops_the_subsonic_loop():
    mdot, _, _ = inlet_mass_flow(0.7)                  # the peak relative Mach number passes 1 near the LE
    with pytest.raises(sv.TransonicFlowError, match="Mach number reaches") as info:
        sv.solve_density_coupled(cascade(), config(inlet_mach=0.7), mass_flow_per_depth=mdot, c_ref=C_REF)
    partial = info.value.solution
    assert not partial.converged and partial.passes[-1].mach_max >= 1.0 and np.max(partial.mach) >= 1.0


def test_inputs_outside_the_scope_are_refused():
    mdot, _, _ = inlet_mass_flow(0.3)
    g = grid()
    for c_ref in (None, 0.0, -0.1, float("nan"), True):
        with pytest.raises(sv.SolverInputError, match="c_ref"):
            sv.solve_at_exit_angle(g, config(), mass_flow_per_depth=mdot, c_ref=c_ref)
    with pytest.raises(sv.SolverInputError, match="rotating"):
        sv.solve_at_exit_angle(g, config(omega=100.0, stream_surface_radius=0.3, inlet_absolute_tangential_velocity=10.0),
                               mass_flow_per_depth=mdot, c_ref=C_REF)
    with pytest.raises(sv.SolverInputError, match="start density"):
        sv.solve_at_exit_angle(g, config(), mass_flow_per_depth=mdot, c_ref=C_REF, density=np.ones((3, 3)))
    with pytest.raises(sv.SolverInputError, match="at least one pass"):
        sv.solve_at_exit_angle(g, config(passes=0), mass_flow_per_depth=mdot, c_ref=C_REF)
    dimensional = make_cascade(CascadeSpec(blade_type="NACA65", solidity=1.0, stagger_deg=33.0, cl0=1.2,
                                           t_max_over_c=0.10, surface_continuity="C2", physical_chord=0.1))
    with pytest.raises(sv.SolverInputError, match="physical chord"):
        sv.solve_density_coupled(dimensional, config(), mass_flow_per_depth=mdot, c_ref=C_REF)
    with pytest.raises(TypeError):
        sv.solve_at_exit_angle(g, None, mass_flow_per_depth=mdot, c_ref=C_REF)


def test_the_fixed_angle_solution_reports_its_kutta_residual_without_correcting_it():
    """solve_at_exit_angle is unchanged by the Kutta coupling: its angle stays, no Kutta pass is
    recorded, and its TE speed difference is reported (R_K = +48.33 m/s at kappa_te)."""
    sol = solved()
    assert sol.exit_angle == sol.grid.exit_angle == cascade().kappa_te and sol.kutta_relaxation is None
    assert sol.history.exit_angle == [] and sol.history.kutta_error == []
    assert sol.kutta.station.i == sol.grid.last_blade_i
    # measured on the quintic thickness (2026-11-13); 48.34643594393599 and 0.6563355777339204 with the legacy
    # PCHIP thickness; 48.378526603618205 and 0.6567739456255879 before the region-wise psi_m of flow.py (psi_m
    # central across the transitions: a different velocity at f - 1, f, l, l + 1)
    assert sol.kutta_residual == pytest.approx(48.32878674594479, rel=1e-9)
    assert sol.normalized_kutta_residual == pytest.approx(0.6561158650596688, rel=1e-9)
    for record in sol.passes:
        assert record.exit_angle == record.exit_angle_next == sol.exit_angle and math.isnan(record.tan_correction)


# --- the density + Kutta coupled solver (solve) -------------------------------------------------------------

KTOL = 1e-8
GUESSES = (-5.0, 0.0, 10.0)          # exit_angle_guess_deg - kappa_te: both sides of the converged angle


def coupled_config(offset=0.0, passes=400, **kwargs):
    return config(tolerance=kwargs.pop("tolerance", TOL), passes=passes, kutta_tolerance=kwargs.pop("kutta_tolerance", KTOL),
                  kutta_passes=kwargs.pop("kutta_passes", passes), exit_offset_deg=offset, **kwargs)


@lru_cache(maxsize=None)
def coupled(offset=0.0):
    mdot, _, _ = inlet_mass_flow(0.3)
    return sv.solve(cascade(), coupled_config(offset), mass_flow_per_depth=mdot, c_ref=C_REF)


@lru_cache(maxsize=None)
def start_grid(offset=0.0):
    c = coupled_config(offset)
    return make_grid(cascade(), c.grid, math.radians(c.flow.inlet_angle_deg), math.radians(c.flow.exit_angle_guess_deg),
                     require_ss_ps=True)


def test_kutta_and_coupled_stop_tests_by_hand():
    settings = cf.IterationSettings(density_tolerance=1e-3, max_density_iterations=5, kutta_tolerance=1e-4,
                                    max_kutta_iterations=5, velocity_tolerance=1e-2)
    assert convergence.velocity_tolerance(settings) == 1e-2
    assert convergence.velocity_tolerance(cf.IterationSettings(density_tolerance=1e-3, max_density_iterations=5,
                                                               kutta_tolerance=1.0, max_kutta_iterations=1)) == 1e-3
    h = convergence.History()
    assert not convergence.kutta_converged(h, settings) and not convergence.coupled_converged(h, settings)
    for d, w, e, density, kutta_ok in ((1e-4, 5e-3, 1e-5, True, True),        # velocity within its own 1e-2
                                       (1e-4, 2e-2, 1e-5, False, True),
                                       (1e-4, 5e-3, 2e-4, True, False),          # density converged, Kutta not
                                       (2e-3, 5e-3, 1e-5, False, True),
                                       (1e-4, 5e-3, float("nan"), True, False)):
        h.density_change.append(d)
        h.velocity_change.append(w)
        h.kutta_error.append(e)
        assert convergence.density_converged(h, settings) is density
        assert convergence.kutta_converged(h, settings) is kutta_ok
        assert convergence.coupled_converged(h, settings) is (density and kutta_ok)


def test_solve_is_the_density_and_kutta_coupled_entry_point():
    sol = coupled()
    assert sol.converged and sol.limitation == sv.COUPLED_LIMITATION and sol.kutta_relaxation == 1.0
    assert "FIXED-DISCHARGE-ANGLE" not in sol.limitation and "NOT YET VALIDATED" in sol.limitation
    assert cf.KUTTA_RELAXATION == 1.0 == cf.IterationSettings(**dict(density_tolerance=1.0, max_density_iterations=1,
                                                                     kutta_tolerance=1.0, max_kutta_iterations=1)).kutta_relaxation


@pytest.mark.parametrize("offset", GUESSES)
def test_coupled_case_meets_all_three_tolerances_on_one_pass(offset):
    sol = coupled(offset)
    loop = [p for p in sol.passes if not p.final]
    final = sol.passes[-1]
    assert sol.converged and final.final and len(loop) == sol.iterations
    last = loop[-1]
    assert last.density_change <= TOL and last.velocity_change <= TOL and last.kutta_normalized <= KTOL
    assert not any(p.density_change <= TOL and p.velocity_change <= TOL and p.kutta_normalized <= KTOL for p in loop[:-1])
    assert sol.normalized_kutta_residual <= KTOL                 # of the reported flow; measured 7.8e-9 to 8.3e-9
    assert abs(sol.kutta_residual) <= KTOL * 0.5 * (sol.kutta.W_ss + sol.kutta.W_ps)
    assert math.isnan(loop[0].velocity_change)
    # the Kutta history: |R_K| falls every pass, never changes sign (monotonic), at the measured rate
    R = [p.kutta_residual for p in loop]
    assert len({math.copysign(1.0, r) for r in R}) == 1
    ratios = [b / a for a, b in zip(R, R[1:])]
    assert all(0.0 < q < 1.0 for q in ratios)                   # measured 0.873 to 0.902
    assert all(0.876 <= q <= 0.877 for q in ratios[-20:])       # measured 0.87673 to 0.87685 (0.87350 to 0.87362
                                                                # before the region-wise psi_m)
    assert math.copysign(1.0, R[0]) == (1.0 if offset < 6.0 else -1.0)   # R_K > 0 below the root, < 0 above it
    assert max(p.mach_max for p in sol.passes) < 1.0 and np.max(sol.mach) < 1.0
    assert final.linear_residual < 1e-10 and final.mass_flow_error_max < 5e-3 and final.target_mismatch < 2.5 * TOL


def test_coupled_case_converges_to_one_angle_and_one_field_from_every_guess():
    """kappa_te - 5, kappa_te, kappa_te + 10 deg: one discharge angle, one density, one
    velocity, within what the tolerances leave (measured: 1.3e-7 deg; rho 5.3e-10; W 9.1e-9)."""
    sols = [coupled(offset) for offset in GUESSES]
    angles = [s.exit_angle for s in sols]
    assert math.degrees(max(angles) - min(angles)) < 1e-6
    assert all(math.degrees(abs(a - math.radians(23.5002381))) < 1e-6 for a in angles)   # measured 23.50023808 deg
    # (23.50220080 deg with the legacy PCHIP thickness; 23.46843468 deg before the region-wise psi_m of flow.py:
    # +0.034 deg from the transition fix)
    for other in sols[::2]:
        assert np.max(np.abs(other.density / sols[1].density - 1.0)) < 1e-8
        assert np.max(np.abs(other.flow.W / sols[1].flow.W - 1.0)) < 1e-7
    assert all(s.kutta.station.i == s.grid.last_blade_i for s in sols)


def test_coupled_final_state_is_one_evaluation_at_the_final_density_and_angle():
    """The reported grid, density, psi, flow, static state and Kutta residual are one pass at
    (rho_final, beta_final): rebuilt here from the start grid and the reported density with
    module calls, they agree array for array."""
    sol = coupled()
    mdot = sol.mass_flow_per_depth
    g = rebuild_exit(start_grid(), sol.exit_angle)
    assert np.array_equal(g.nodes, sol.grid.nodes) and sol.grid.exit_angle == sol.exit_angle
    metrics, bcs = build_metrics(g), build_boundary_conditions(g)
    co = build_coefficients(metrics, np.array(sol.density), mdot, c_ref=C_REF)
    lin = solve_linear_system(assemble_from_coefficients(co, metrics, bcs))
    flow = velocity_components(lin.psi, np.array(sol.density), co, metrics, bcs)
    assert np.array_equal(lin.psi, sol.psi)
    for name in ("W_m", "W_theta", "W"):
        assert np.array_equal(getattr(flow, name), getattr(sol.flow, name))
    state = eos.static_state_from_velocity(flow.W, AIR, eos.blade_row_reference(eos.TotalState(PT, TT), AIR))
    for name in ("temperature", "pressure", "density"):
        assert np.array_equal(getattr(state, name), getattr(sol.static_state, name))
    assert sol.kutta == kutta.kutta_residual(flow, g)
    loop = [p for p in sol.passes if not p.final]
    assert sol.passes[-1].exit_angle == sol.exit_angle == loop[-1].exit_angle_next     # beta(n+1), not beta(n)
    assert sol.history.exit_angle[-1] == sol.exit_angle and sol.history.kutta_error[-1] == sol.normalized_kutta_residual


def test_coupled_passes_start_from_both_updated_quantities():
    """Pass 1 is built from (rho(1), beta(1)) of pass 0: the relaxed density and the corrected
    angle, both checked against independent module calls on the start grid."""
    mdot, _, _ = inlet_mass_flow(0.3)
    with pytest.raises(sv.CoupledNotConvergedError, match="did not converge in 2 passes") as info:
        sv.solve(cascade(), coupled_config(passes=2), mass_flow_per_depth=mdot, c_ref=C_REF)
    partial = info.value.solution
    first, second = partial.passes
    g = start_grid()
    ref = eos.blade_row_reference(eos.TotalState(PT, TT), AIR)
    rho0 = np.array(eos.initial_density(AIR, ref, shape=(len(g.m), len(g.eta))))
    metrics, bcs = build_metrics(g), build_boundary_conditions(g)
    co = build_coefficients(metrics, rho0, mdot, c_ref=C_REF)
    lin = solve_linear_system(assemble_from_coefficients(co, metrics, bcs))
    flow = velocity_components(lin.psi, rho0, co, metrics, bcs)
    target = eos.static_state_from_velocity(flow.W, AIR, ref).density
    residual = kutta.kutta_residual(flow, g)
    beta1 = kutta.next_exit_angle(g.exit_angle, kutta.exit_angle_correction(residual, flow, metrics), 1.0)
    assert first.exit_angle == g.exit_angle and first.kutta_residual == residual.residual
    assert first.exit_angle_next == beta1 and second.exit_angle == beta1 != g.exit_angle
    assert partial.exit_angle == beta1 and partial.grid.exit_angle == beta1
    assert np.array_equal(partial.density, eos.relax_density(rho0, target))            # rho(1), the relaxed one
    assert partial.limitation == sv.COUPLED_LIMITATION and not partial.converged and partial.iterations == 2


def test_coupled_topology_never_changes():
    sol, g = coupled(10.0), start_grid(10.0)
    kept = slice(0, g.last_blade_i + 1)
    assert np.array_equal(sol.grid.m, g.m) and np.array_equal(sol.grid.eta, g.eta)
    assert (sol.grid.first_blade_i, sol.grid.last_blade_i) == (g.first_blade_i, g.last_blade_i)
    assert np.array_equal(sol.grid.nodes[kept], g.nodes[kept]) and not np.array_equal(sol.grid.nodes, g.nodes)
    validate_uniform_grid_topology(sol.grid)
    assert sol.grid.exit_angle != g.exit_angle


def test_an_exit_rebuild_that_moves_more_than_the_exit_is_refused(monkeypatch):
    mdot, _, _ = inlet_mass_flow(0.3)
    other = make_grid(cascade(), cf.GridSettings(n_passage=43, n_eta=21, downstream_extension_chords=1.25),
                      math.radians(INLET_DEG), cascade().kappa_te)
    monkeypatch.setattr(sv, "rebuild_exit", lambda grid, angle: other)
    with pytest.raises(sv.ExitRebuildError, match="exit nodes only"):
        sv.solve(cascade(), coupled_config(), mass_flow_per_depth=mdot, c_ref=C_REF)


def test_coupled_density_convergence_alone_does_not_stop_the_loop():
    """A density tolerance met within a few passes and a Kutta tolerance not met: the loop goes
    on (it never converges the density first and the Kutta condition afterwards) and ends at
    its pass limit, min(max_density_iterations, max_kutta_iterations)."""
    mdot, _, _ = inlet_mass_flow(0.3)
    c = coupled_config(passes=50, tolerance=0.5, kutta_tolerance=1e-12, kutta_passes=6)
    with pytest.raises(sv.CoupledNotConvergedError, match="did not converge in 6 passes") as info:
        sv.solve(cascade(), c, mass_flow_per_depth=mdot, c_ref=C_REF)
    partial = info.value.solution
    assert len(partial.passes) == 6 and not partial.converged
    assert any(p.density_change <= 0.5 and p.velocity_change <= 0.5 for p in partial.passes[1:])
    assert all(p.kutta_normalized > 1e-12 for p in partial.passes)


def test_coupled_first_pass_is_never_converged():
    mdot, _, _ = inlet_mass_flow(0.3)
    sol = sv.solve(cascade(), coupled_config(tolerance=10.0, kutta_tolerance=10.0), mass_flow_per_depth=mdot, c_ref=C_REF)
    assert sol.converged and sol.iterations == 2 and len(sol.passes) == 3 and sol.passes[-1].final


def test_coupled_sonic_node_stops_the_loop():
    mdot, _, _ = inlet_mass_flow(0.7)
    with pytest.raises(sv.TransonicFlowError, match="Mach number reaches") as info:
        sv.solve(cascade(), coupled_config(inlet_mach=0.7), mass_flow_per_depth=mdot, c_ref=C_REF)
    assert info.value.solution.limitation == sv.COUPLED_LIMITATION and np.max(info.value.solution.mach) >= 1.0


def test_coupled_inputs_outside_the_scope_are_refused():
    mdot, _, _ = inlet_mass_flow(0.3)
    with pytest.raises(sv.SolverInputError, match="max_kutta_iterations"):
        sv.solve(cascade(), coupled_config(kutta_passes=0), mass_flow_per_depth=mdot, c_ref=C_REF)
    with pytest.raises(sv.SolverInputError, match="c_ref"):
        sv.solve(cascade(), coupled_config(), mass_flow_per_depth=mdot, c_ref=None)
    with pytest.raises(kutta.KuttaError, match="relaxation"):
        sv.solve(cascade(), coupled_config(kutta_relaxation=1.5), mass_flow_per_depth=mdot, c_ref=C_REF)


def test_records_are_numbers_only():
    sol = solved()
    for record in sol.passes:
        for value in record.__dict__.values():
            assert isinstance(value, (int, float, bool))


def test_solver_holds_no_equations():
    """solver.py reads no gas constant, spacing, coefficient or metric field: the formulas are
    in the modules it calls."""
    tree = ast.parse(Path(sv.__file__).read_text(encoding="utf-8"))
    attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    assert not {"cp", "gamma", "gas_constant", "dm", "deta", "A", "B", "C", "D", "E", "S", "tan_beta", "W2", "psi_eta",
                "psi_m"} & attributes
    imported = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
    assert not {name for name in imported if name.split(".")[0] in ("scipy",)}
    assert "scipy" not in {alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    assert "matplotlib" not in imported and not any(name.startswith("matplotlib") for name in imported)


# --- the mass flow from the inlet Mach number (2026-11-13) -------------------------------------------------

def same_records(a, b) -> bool:
    """PassRecord sequences equal field by field, NaN equal to NaN (pass 0 has no velocity
    change)."""
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        for name, value in x.__dict__.items():
            other = getattr(y, name)
            if not (value == other or (isinstance(value, float) and math.isnan(value) and math.isnan(other))):
                return False
    return True


def test_mass_flow_is_derived_from_the_inlet_mach_number():
    """Normal use: no mdot' given. The derived mdot' is this file's derivation to the last bit
    (measured: equal), so the solution equals the override solve array for array, and both
    keep the inlet state."""
    derived = sv.solve_density_coupled(cascade(), config(), c_ref=C_REF)
    override = solved()
    assert derived.mass_flow_source == sv.MASS_FLOW_FROM_INLET and override.mass_flow_source == sv.MASS_FLOW_CHECKED
    assert derived.mass_flow_per_depth == inlet_mass_flow(0.3)[0] == override.mass_flow_per_depth
    for name in ("density", "psi"):
        assert np.array_equal(getattr(derived, name), getattr(override, name))
    assert same_records(derived.passes, override.passes)
    assert derived.inlet == override.inlet and derived.inlet.mach == 0.3
    assert derived.inlet.mass_flow_per_depth == derived.mass_flow_per_depth


def test_an_override_must_agree_with_the_inlet_mach_number():
    mdot, _, _ = inlet_mass_flow(0.3)
    g = grid()
    assert sv._context(g, config(), mdot * (1.0 + 1e-13), C_REF).mass_flow_source == sv.MASS_FLOW_CHECKED   # round-off
    for wrong in (mdot * (1.0 + 1e-9), mdot * 0.9, inlet_mass_flow(0.31)[0]):
        with pytest.raises(sv.SolverInputError, match="disagree"):
            sv.solve_at_exit_angle(g, config(), c_ref=C_REF, mass_flow_per_depth=wrong)
    pure = sv._context(g, config(inlet_mach=None), 7.0, C_REF)              # a verification override alone
    assert pure.mass_flow_per_depth == 7.0 and pure.inlet is None and pure.mass_flow_source == sv.MASS_FLOW_OVERRIDE
    with pytest.raises(sv.SolverInputError, match="no mass flow"):
        sv.solve_at_exit_angle(g, config(inlet_mach=None), c_ref=C_REF)
    with pytest.raises(sv.SolverInputError, match="zero flow"):
        sv.solve_at_exit_angle(g, config(inlet_mach=0.0), c_ref=C_REF)     # the policy of coefficients.py
    with pytest.raises(sv.SolverInputError, match="inlet state cannot be derived"):
        sv.solve_at_exit_angle(g, config(inlet_mach=-0.2), c_ref=C_REF)


@pytest.mark.parametrize("which", ["fixed", "coupled"])
def test_the_history_lists_are_the_pass_records(which):
    """Every pass appends one entry to each History list, taken from its PassRecord: index n
    of every list is pass n (exit_angle and kutta_error in the coupled loop only)."""
    sol = solved() if which == "fixed" else coupled()
    h, passes = sol.history, sol.passes
    for name, field_name in (("density_change", "density_change"), ("velocity_change", "velocity_change"),
                             ("psi_change", "psi_change"), ("residual", "linear_residual")):
        values = getattr(h, name)
        assert len(values) == len(passes)
        assert all(a == getattr(p, field_name) or (math.isnan(a) and math.isnan(getattr(p, field_name)))
                   for a, p in zip(values, passes))
    if which == "coupled":
        assert h.kutta_pass == [p.index for p in passes]
        assert h.exit_angle == [p.exit_angle for p in passes]
        assert h.kutta_error == [p.kutta_normalized for p in passes]
        assert all(b.exit_angle == a.exit_angle_next for a, b in zip(passes, passes[1:]))   # pass n+1 resident = n's next
    else:
        assert h.kutta_pass == [0] * len(passes) and h.exit_angle == [] and h.kutta_error == []


# --- the sonic stop switched off (observation only, 2026-11-13) ----------------------------------------------

@pytest.mark.parametrize("mode", ["fixed", "coupled"])
def test_the_sonic_stop_can_be_switched_off(mode):
    """stop_at_sonic False at M1 0.72: the loop goes on past the first sonic pass (measured:
    pass 9, either mode) with one SonicFlowWarning, until the gas model refuses T <= 0
    (measured: pass 85, either mode). That error keeps its type and carries the last
    completed pass (.solution) and the failed pass's index (.failed_pass); the limitation
    string says the stop was off. (M1 0.70 until the quintic thickness of 2026-11-13: T <= 0
    at passes 172 / 181, first sonic 13 / 14, fixed / coupled; on the quintic geometry 0.70
    no longer reaches T <= 0 in 400 passes but ends at the pass limit.)"""
    mdot, _, _ = inlet_mass_flow(0.72)
    c = config(inlet_mach=0.72, stop_at_sonic=False, tolerance=TOL, passes=400, kutta_tolerance=TOL, kutta_passes=400)
    entry = sv.solve_density_coupled if mode == "fixed" else sv.solve
    with pytest.warns(sv.SonicFlowWarning) as warned:
        with pytest.raises(eos.ThermodynamicStateError, match="T <= 0") as info:
            entry(cascade(), c, c_ref=C_REF)
    assert len([w for w in warned if issubclass(w.category, sv.SonicFlowWarning)]) == 1
    error = info.value
    partial = error.solution
    assert isinstance(partial, sv.Solution) and not partial.converged
    assert error.failed_pass == len(partial.passes) == partial.iterations
    first_sonic = next(p.index for p in partial.passes if p.mach_max >= 1.0)
    assert first_sonic < error.failed_pass - 10                        # it went on well past sonic
    assert all(p.index == k for k, p in enumerate(partial.passes))
    assert "SONIC STOP OFF" in partial.limitation
    assert partial.mass_flow_per_depth == pytest.approx(mdot, rel=1e-14)   # measured 3e-16 (last bit)
    with pytest.raises(sv.TransonicFlowError):                         # the default still stops
        entry(cascade(), config(inlet_mach=0.72, passes=400, kutta_tolerance=TOL, kutta_passes=400), c_ref=C_REF)


def test_without_sonic_flow_the_switch_changes_nothing():
    off = sv.solve_density_coupled(cascade(), config(stop_at_sonic=False), c_ref=C_REF)
    on = solved()
    assert np.array_equal(off.density, on.density) and np.array_equal(off.psi, on.psi)
    assert off.limitation == on.limitation and same_records(off.passes, on.passes)


# --- initial guesses and the per-pass field hand-out (diagnosis / restart plumbing, 2026-11-13) --------------

def test_an_initial_density_is_a_start_not_a_change():
    """A restart from the converged density (and, coupled, from the converged angle) converges at
    once to the same solution; the default start is untouched; a wrong shape is refused."""
    base = solved()
    again = sv.solve_density_coupled(cascade(), config(), c_ref=C_REF, initial_density=np.array(base.density))
    assert again.converged and again.iterations <= 3                     # measured 2 (pass 0 never converges)
    assert again.passes[0].density_min == np.min(base.density) and again.passes[0].density_max == np.max(base.density)
    assert np.max(np.abs(again.density / base.density - 1.0)) < 1e-7     # both within the 1e-8 tolerance of one fixed point
    with pytest.raises(sv.SolverInputError, match="start density"):
        sv.solve_density_coupled(cascade(), config(), c_ref=C_REF, initial_density=np.ones((3, 3)))
    full = coupled()
    offset = math.degrees(full.exit_angle) - math.degrees(cascade().kappa_te)
    warm = sv.solve(cascade(), coupled_config(offset), c_ref=C_REF, initial_density=np.array(full.density))
    assert warm.converged and warm.iterations <= 3
    assert math.degrees(abs(warm.exit_angle - full.exit_angle)) < 1e-6
    assert np.max(np.abs(warm.density / full.density - 1.0)) < 1e-7


def test_an_inspecting_observer_gets_read_only_pass_fields(monkeypatch):
    class Inspector:
        def __init__(self):
            self.seen = []

        def start(self, info):
            pass

        def update(self, record):
            pass

        def finish(self, solution, error):
            pass

        def inspect(self, fields):
            for name in ("density_resident", "density_target", "density_next", "psi", "mach"):
                array = getattr(fields, name)
                assert array.shape == fields.mach.shape and not array.flags.writeable
                with pytest.raises(ValueError):
                    array[0, 0] = 0.0
            assert fields.grid is fields.metrics.grid and fields.mass_flow_per_depth > 0.0 and fields.c_ref == C_REF
            self.seen.append((fields.record, float(fields.mach.max()), np.array(fields.density_resident)))

    inspector = Inspector()
    sol = sv.solve_density_coupled(cascade(), config(), c_ref=C_REF, monitor=inspector)
    base = solved()
    assert np.array_equal(sol.density, base.density) and np.array_equal(sol.psi, base.psi)    # observation only
    assert len(inspector.seen) == len(sol.passes)
    assert all(record is sol.passes[k] and peak == record.mach_max for k, (record, peak, _) in enumerate(inspector.seen))
    assert np.array_equal(inspector.seen[-1][2], sol.density)

    class Plain:                                                           # no inspect: nothing is handed out
        def start(self, info):
            pass

        def update(self, record):
            pass

        def finish(self, solution, error):
            pass

    def refuse(*args, **kwargs):
        raise AssertionError("PassFields built although no observer inspects")

    monkeypatch.setattr(sv, "PassFields", refuse)
    plain = sv.solve_density_coupled(cascade(), config(), c_ref=C_REF, monitor=Plain())
    assert np.array_equal(plain.density, base.density)
