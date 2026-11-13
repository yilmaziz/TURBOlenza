"""
Tests for potential_flow_solver.eos: the calorically perfect gas, the energy relation
h = I + (r omega)^2 / 2 - W^2 / 2, the static state at the relative speed, Aungier's initial
density and the relaxed density update; against closed-form relations computed here, not
the module itself.

What is checked
- The gas: cp = gamma R / (gamma - 1), cv, h = cp T, p = rho R T, a^2 = gamma R T; refusals of
  R <= 0, gamma <= 1, non-finite and non-real values.
- References: stationary I = cp Tt; rotating I = cp Tt - omega r C_theta_in and H' = I + (r
  omega)^2 / 2 on hand-calculable numbers; a rotating row without radius or swirl, a
  non-positive radius, non-finite omega / swirl, non-positive totals refused.
- The static state: stationary T = Tt - W^2 / (2 cp), p = Pt (T / Tt)^(gamma / (gamma - 1)),
  rho = p / (R T), a = sqrt(gamma R T) computed independently; the omega = 0 reduction of the
  rotating reference; W = 0 returns the inlet total state; isentropy cp ln(T / Tt) - R ln(p /
  Pt) = 0; energy h + W^2 / 2 = I + (r omega)^2 / 2 (stationary and rotating); the closed-form
  ratios in Mach number for 0 <= M <= 1.5 for three gases; arrays of the flow.py shape equal
  to scalars element by element; read-only results; T <= 0 refused, never clipped; negative or
  non-finite speeds refused.
- The initial density: Pt / (R Tt) stationary; the relative total density of a rotating row
  (Tt' = H' / cp, Pt' isentropic); its omega = 0 limit; a uniform field of a given shape.
- The relaxation: the default midpoint, lambda = 1 equal to the target bit for bit (Aungier's
  direct replacement), the formula for several lambda, inputs untouched, shape kept, the result
  between old and target, the error halved exactly per pass for a fixed target, refusals of
  lambda outside (0, 1], shapes that differ and non-positive densities.
- The replacement contract: a test-only gas object with the same methods runs through
  static_state_from_velocity; the module runs no loop and imports no solver module.

Tolerances: closed-form comparisons to 1e-15 relative, 4e-15 where the power of T enters
(measured T 2.2e-16, a 2.2e-16, p 8.9e-16, rho 7.8e-16 for air up to M 0.95); the isentropy
residual below 1e-12 J/(kg K) absolute (measured 5.7e-14) and 1e-12 relative to cp |ln(T / Tt)|;
the energy residual to 1e-15 relative (measured 0 stationary, 2.0e-16 rotating).
"""

from __future__ import annotations

import ast
import inspect
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from potential_flow_solver import eos

AIR = eos.CaloricallyPerfectGas(gamma=1.4, gas_constant=287.05)
GASES = [AIR, eos.CaloricallyPerfectGas(gamma=5.0 / 3.0, gas_constant=2077.1),
         eos.CaloricallyPerfectGas(gamma=1.289, gas_constant=188.92)]
TOTAL = eos.TotalState(pressure=101325.0, temperature=288.15)
REL = 1e-15


def close(a, b, rel=REL):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return bool(np.all(np.abs(a - b) <= rel * np.maximum(np.abs(b), 1e-300)))


# --- the gas -------------------------------------------------------------------------------------

@pytest.mark.parametrize("gas", GASES)
def test_gas_constants_and_relations(gas):
    g, R = gas.gamma, gas.gas_constant
    assert close(gas.cp, g * R / (g - 1.0)) and close(gas.cv, R / (g - 1.0))
    assert close(gas.cp - gas.cv, R, rel=1e-14) and close(gas.cp / gas.cv, g, rel=1e-14)
    T = np.array([150.0, 288.15, 600.0])
    assert close(gas.enthalpy(T), gas.cp * T) and close(gas.temperature(gas.cp * T), T)
    p = np.array([5e4, 1e5, 3e5])
    rho = gas.density(p, T)
    assert close(p, rho * R * T, rel=1e-15)
    assert close(gas.sound_speed(T) ** 2, g * R * T, rel=1e-15)


def test_perfect_gas_name_is_kept():
    assert eos.PerfectGas is eos.CaloricallyPerfectGas and eos.GasState is eos.StaticState


@pytest.mark.parametrize("gamma, R", [(1.0, 287.0), (0.9, 287.0), (1.4, 0.0), (1.4, -1.0), (float("nan"), 287.0),
                                      (1.4, float("inf")), (True, 287.0), ("1.4", 287.0)])
def test_gas_refusals(gamma, R):
    with pytest.raises(eos.ThermodynamicStateError):
        eos.CaloricallyPerfectGas(gamma=gamma, gas_constant=R)


@pytest.mark.parametrize("p, T", [(0.0, 288.0), (-1.0, 288.0), (1e5, 0.0), (1e5, -5.0), (float("nan"), 288.0), (1e5, float("inf"))])
def test_total_state_refusals(p, T):
    with pytest.raises(eos.ThermodynamicStateError):
        eos.TotalState(pressure=p, temperature=T)


# --- references --------------------------------------------------------------------------------------

def test_stationary_reference_is_the_inlet_total_enthalpy():
    ref = eos.blade_row_reference(TOTAL, AIR)
    assert ref.omega == 0.0 and ref.radius is None and ref.blade_speed == 0.0
    assert ref.rothalpy == AIR.cp * TOTAL.temperature == ref.relative_total_enthalpy


def test_rotating_reference_by_hand():
    """cp = 1004.5 (gamma 1.4, R 287), Tt = 300: H_in = 301 350; omega 1000, r 0.3, C_theta 50:
    omega r C_theta = 15 000, I = 286 350; U = 300, H' = I + 45 000 = 331 350."""
    gas = eos.CaloricallyPerfectGas(gamma=1.4, gas_constant=287.0)
    total = eos.TotalState(pressure=1e5, temperature=300.0)
    ref = eos.blade_row_reference(total, gas, omega=1000.0, radius=0.3, inlet_c_theta=50.0)
    assert gas.cp == pytest.approx(1004.5, rel=1e-15)
    assert ref.rothalpy == pytest.approx(286350.0, rel=1e-15)
    assert ref.blade_speed == pytest.approx(300.0, rel=1e-15)
    assert ref.relative_total_enthalpy == pytest.approx(331350.0, rel=1e-15)
    W = np.array([0.0, 120.0, 250.0])
    state = eos.static_state_from_velocity(W, gas, ref)
    assert close(state.enthalpy, 331350.0 - 0.5 * W ** 2)
    assert close(state.enthalpy + 0.5 * W ** 2, ref.rothalpy + 0.5 * (0.3 * 1000.0) ** 2)


def test_omega_zero_is_the_stationary_limit_of_the_one_model():
    """A row built with omega = 0 but a radius and an inlet swirl is the stationary row: the
    general I - omega r C_theta and H' = I + (r omega)^2 / 2 reduce to cp Tt, and the static state
    to T = Tt - W^2 / (2 cp)."""
    ref0 = eos.blade_row_reference(TOTAL, AIR, omega=0.0, radius=0.25, inlet_c_theta=80.0)
    ref = eos.blade_row_reference(TOTAL, AIR)
    assert ref0.relative_total_enthalpy == ref.relative_total_enthalpy == AIR.cp * TOTAL.temperature
    W = np.linspace(0.0, 250.0, 11)
    a, b = eos.static_state_from_velocity(W, AIR, ref0), eos.static_state_from_velocity(W, AIR, ref)
    assert np.array_equal(a.temperature, b.temperature) and np.array_equal(a.density, b.density)
    assert close(a.temperature, TOTAL.temperature - W ** 2 / (2.0 * AIR.cp), rel=1e-15)


@pytest.mark.parametrize("kwargs", [dict(omega=100.0), dict(omega=100.0, radius=0.3), dict(omega=100.0, inlet_c_theta=10.0),
                                    dict(omega=100.0, radius=0.0, inlet_c_theta=10.0),
                                    dict(omega=100.0, radius=-0.3, inlet_c_theta=10.0),
                                    dict(omega=float("nan"), radius=0.3, inlet_c_theta=10.0),
                                    dict(omega=100.0, radius=0.3, inlet_c_theta=float("inf")),
                                    dict(omega=0.0, radius=-1.0)])
def test_reference_refusals(kwargs):
    with pytest.raises(eos.ThermodynamicStateError):
        eos.blade_row_reference(TOTAL, AIR, **kwargs)


# --- the static state --------------------------------------------------------------------------------------

def test_stationary_static_state_against_closed_forms():
    ref = eos.blade_row_reference(TOTAL, AIR)
    W = np.array([0.0, 30.0, 100.0, 180.0, 250.0, 320.0])
    s = eos.static_state_from_velocity(W, AIR, ref)
    cp, R, g = AIR.gamma * AIR.gas_constant / (AIR.gamma - 1.0), AIR.gas_constant, AIR.gamma
    T = TOTAL.temperature - W ** 2 / (2.0 * cp)
    p = TOTAL.pressure * (T / TOTAL.temperature) ** (g / (g - 1.0))
    assert close(s.temperature, T, rel=1e-15)
    assert close(s.pressure, p, rel=1e-15 * 4)                     # the power carries (gamma/(gamma-1)) roundings of T
    assert close(s.density, p / (R * T), rel=1e-15 * 4)
    assert close(s.sound_speed, np.sqrt(g * R * T), rel=1e-15)
    assert close(s.enthalpy, cp * TOTAL.temperature - 0.5 * W ** 2, rel=1e-15)


def test_zero_speed_returns_the_inlet_total_state():
    ref = eos.blade_row_reference(TOTAL, AIR)
    s = eos.static_state_from_velocity(0.0, AIR, ref)
    assert isinstance(s.temperature, float)
    assert s.temperature == pytest.approx(TOTAL.temperature, rel=1e-15)
    assert s.pressure == pytest.approx(TOTAL.pressure, rel=1e-15)
    assert s.density == pytest.approx(TOTAL.pressure / (AIR.gas_constant * TOTAL.temperature), rel=1e-15)
    assert s.sound_speed == pytest.approx(math.sqrt(AIR.gamma * AIR.gas_constant * TOTAL.temperature), rel=1e-15)


@pytest.mark.parametrize("gas", GASES)
@pytest.mark.parametrize("rotating", [False, True])
def test_isentropy_and_energy_over_subsonic_speeds(gas, rotating):
    ref = (eos.blade_row_reference(TOTAL, gas, omega=1200.0, radius=0.2, inlet_c_theta=40.0) if rotating
           else eos.blade_row_reference(TOTAL, gas))
    a_total = math.sqrt(gas.gamma * gas.gas_constant * TOTAL.temperature)
    W = np.linspace(0.0, 0.95 * a_total, 40)
    s = eos.static_state_from_velocity(W, gas, ref)
    residual = gas.cp * np.log(s.temperature / TOTAL.temperature) - gas.gas_constant * np.log(s.pressure / TOTAL.pressure)
    assert np.max(np.abs(residual)) < 1e-12
    scale = gas.cp * np.abs(np.log(s.temperature / TOTAL.temperature))
    moving = scale > 1e-6
    assert np.max(np.abs(residual[moving]) / scale[moving]) < 1e-12
    energy = s.enthalpy + 0.5 * W ** 2
    assert close(energy, ref.rothalpy + 0.5 * ref.blade_speed ** 2, rel=1e-15)


@pytest.mark.parametrize("gas", GASES)
def test_closed_form_ratios_in_mach_number(gas):
    """Stationary row: T / Tt = 1 / (1 + (gamma - 1) / 2 M^2), p / pt = (T / Tt)^(gamma / (gamma - 1)),
    rho / rho_t = (T / Tt)^(1 / (gamma - 1)) for 0 <= M <= 1.5, W = M a(T)."""
    g, R = gas.gamma, gas.gas_constant
    M = np.linspace(0.0, 1.5, 31)
    ratio = 1.0 / (1.0 + 0.5 * (g - 1.0) * M ** 2)
    T = TOTAL.temperature * ratio
    W = M * np.sqrt(g * R * T)
    s = eos.static_state_from_velocity(W, gas, eos.blade_row_reference(TOTAL, gas))
    rho_t = TOTAL.pressure / (R * TOTAL.temperature)
    assert close(s.temperature / TOTAL.temperature, ratio, rel=4e-15)
    assert close(s.pressure / TOTAL.pressure, ratio ** (g / (g - 1.0)), rel=2e-14)
    assert close(s.density / rho_t, ratio ** (1.0 / (g - 1.0)), rel=2e-14)
    assert close(W / s.sound_speed, M, rel=4e-15)


def test_arrays_of_the_flow_shape_match_scalars_and_are_read_only():
    ref = eos.blade_row_reference(TOTAL, AIR, omega=900.0, radius=0.25, inlet_c_theta=30.0)
    rng = np.random.default_rng(6)
    W = 50.0 + 200.0 * rng.random((147, 21))
    s = eos.static_state_from_velocity(W, AIR, ref)
    for name in ("temperature", "pressure", "density", "enthalpy", "sound_speed"):
        value = getattr(s, name)
        assert value.shape == W.shape and not value.flags.writeable
        with pytest.raises(ValueError):
            value[0, 0] = 0.0
    for (i, j) in ((0, 0), (17, 5), (146, 20)):
        one = eos.static_state_from_velocity(float(W[i, j]), AIR, ref)
        for name in ("temperature", "pressure", "density", "enthalpy", "sound_speed"):
            assert getattr(one, name) == getattr(s, name)[i, j]


def test_no_temperature_floor():
    """W^2 / 2 >= H' gives T <= 0: refused, nothing clipped."""
    ref = eos.blade_row_reference(TOTAL, AIR)
    W_limit = math.sqrt(2.0 * ref.relative_total_enthalpy)
    for W in (W_limit, 1.01 * W_limit, np.array([100.0, 2.0 * W_limit])):
        with pytest.raises(eos.ThermodynamicStateError, match="T <= 0"):
            eos.static_state_from_velocity(W, AIR, ref)
    assert eos.static_state_from_velocity(0.999 * W_limit, AIR, ref).temperature > 0.0


@pytest.mark.parametrize("W", [-1.0, np.array([10.0, -0.5]), float("nan"), np.array([1.0, np.inf]), "fast"])
def test_speed_refusals(W):
    with pytest.raises(eos.ThermodynamicStateError):
        eos.static_state_from_velocity(W, AIR, eos.blade_row_reference(TOTAL, AIR))


# --- the initial density ----------------------------------------------------------------------------------

def test_initial_density_stationary():
    ref = eos.blade_row_reference(TOTAL, AIR)
    rho0 = eos.initial_density(AIR, ref)
    assert rho0 == pytest.approx(TOTAL.pressure / (AIR.gas_constant * TOTAL.temperature), rel=1e-15)
    field = eos.initial_density(AIR, ref, shape=(147, 21))
    assert field.shape == (147, 21) and np.all(field == rho0) and not field.flags.writeable


def test_initial_density_rotating_is_the_relative_total_density():
    omega, r, c_theta = 1100.0, 0.3, 60.0
    ref = eos.blade_row_reference(TOTAL, AIR, omega=omega, radius=r, inlet_c_theta=c_theta)
    H_rel = AIR.cp * TOTAL.temperature - omega * r * c_theta + 0.5 * (r * omega) ** 2
    T_rel = H_rel / AIR.cp
    p_rel = TOTAL.pressure * (T_rel / TOTAL.temperature) ** (AIR.gamma / (AIR.gamma - 1.0))
    assert eos.initial_density(AIR, ref) == pytest.approx(p_rel / (AIR.gas_constant * T_rel), rel=4e-15)
    stationary = eos.initial_density(AIR, eos.blade_row_reference(TOTAL, AIR, omega=0.0, radius=r, inlet_c_theta=c_theta))
    assert stationary == eos.initial_density(AIR, eos.blade_row_reference(TOTAL, AIR))


# --- the relaxation --------------------------------------------------------------------------------------------

def test_default_relaxation_is_the_project_baseline_midpoint():
    assert eos.DENSITY_RELAXATION == 0.5 and eos.AUNGIER_DIRECT_REPLACEMENT == 1.0
    assert inspect.signature(eos.relax_density).parameters["relaxation"].default == eos.DENSITY_RELAXATION
    assert np.array_equal(eos.relax_density(np.array([1.0]), np.array([2.0])), [1.5])
    rng = np.random.default_rng(1)
    old, target = 1.0 + rng.random((7, 5)), 1.0 + rng.random((7, 5))
    assert np.array_equal(eos.relax_density(old, target), (old + target) / 2.0)


def test_lambda_one_is_aungier_direct_replacement_bit_for_bit():
    rng = np.random.default_rng(2)
    old, target = 0.5 + rng.random((147, 21)), 0.5 + rng.random((147, 21))
    assert np.array_equal(eos.relax_density(old, target, relaxation=eos.AUNGIER_DIRECT_REPLACEMENT), target)


@pytest.mark.parametrize("lam", [0.1, 0.25, 0.5, 0.7, 0.9, 1.0])
def test_relaxation_formula_shape_bounds_and_inputs_untouched(lam):
    rng = np.random.default_rng(3)
    old, target = 0.8 + rng.random((13, 9)), 0.8 + rng.random((13, 9))
    old_copy, target_copy = old.copy(), target.copy()
    nxt = eos.relax_density(old, target, relaxation=lam)
    assert nxt.shape == old.shape and not nxt.flags.writeable
    assert np.max(np.abs(nxt - ((1.0 - lam) * old + lam * target))) <= 2.3e-16 * np.max(old + target)
    assert old.tobytes() == old_copy.tobytes() and target.tobytes() == target_copy.tobytes()
    lo, hi = np.minimum(old, target), np.maximum(old, target)
    assert np.all(nxt >= lo) and np.all(nxt <= hi)


def test_fixed_target_error_halves_exactly_per_pass():
    """rho(n+1) - rho* = 0.5 (rho(n) - rho*): exact on dyadic values, to round-off otherwise. A
    property of the relaxation only, not of the coupled nonlinear iteration."""
    target = np.full((4, 3), 1.25)
    rho = np.full((4, 3), 2.0)
    for _ in range(20):
        nxt = eos.relax_density(rho, target)
        assert np.array_equal(nxt - target, 0.5 * (rho - target))
        rho = nxt
    rng = np.random.default_rng(8)
    target, rho = 1.0 + rng.random((6, 6)), 1.0 + rng.random((6, 6))
    for _ in range(10):
        nxt = eos.relax_density(rho, target)
        assert np.max(np.abs((nxt - target) - 0.5 * (rho - target))) < 1e-15
        rho = nxt


@pytest.mark.parametrize("lam", [0.0, -0.5, 1.0000001, float("nan"), True])
def test_relaxation_factor_refusals(lam):
    with pytest.raises(eos.ThermodynamicStateError):
        eos.relax_density(np.ones(3), np.ones(3), relaxation=lam)


@pytest.mark.parametrize("old, target", [(np.ones(3), np.ones(4)), (np.ones(3), np.array([1.0, 0.0, 1.0])),
                                         (np.array([1.0, -1.0, 1.0]), np.ones(3)), (np.ones(3), np.array([1.0, np.nan, 1.0]))])
def test_relaxation_input_refusals(old, target):
    with pytest.raises(eos.ThermodynamicStateError):
        eos.relax_density(old, target)


# --- the contract -------------------------------------------------------------------------------------------

def test_a_stand_in_gas_with_the_same_methods_runs_through():
    """static_state_from_velocity uses only the gas's methods: an object that is not a
    CaloricallyPerfectGas but has them (here: a scaled copy) is accepted, so nothing outside
    the gas object depends on the perfect-gas class."""
    class StandIn:
        gamma, gas_constant = AIR.gamma, AIR.gas_constant
        cp = AIR.cp

        def enthalpy(self, T):
            return AIR.enthalpy(T)

        def temperature(self, h):
            return AIR.temperature(h)

        def isentropic_pressure(self, T, total):
            return AIR.isentropic_pressure(T, total)

        def density(self, p, T):
            return AIR.density(p, T)

        def sound_speed(self, T):
            return AIR.sound_speed(T)
    ref = eos.blade_row_reference(TOTAL, AIR)
    W = np.array([10.0, 150.0])
    a, b = eos.static_state_from_velocity(W, StandIn(), ref), eos.static_state_from_velocity(W, AIR, ref)
    assert np.array_equal(a.density, b.density) and np.array_equal(a.sound_speed, b.sound_speed)


def test_the_module_is_a_state_layer_only():
    """No iteration controller, no tolerance, no solver module: eos.py imports numpy and the
    standard library only, and holds no while loop and no for loop over a range (its only for
    statements walk the fields of a dataclass for their checks)."""
    tree = ast.parse(Path(eos.__file__).read_text(encoding="utf-8"))
    assert not [node for node in ast.walk(tree) if isinstance(node, ast.While)]
    ranged = [node for node in ast.walk(tree) if isinstance(node, (ast.For, ast.AsyncFor)) and isinstance(node.iter, ast.Call)
              and getattr(node.iter.func, "id", "") == "range"]
    assert not ranged
    imported = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
    imported |= {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    assert {name for name in imported if name.split(".")[0] not in sys.stdlib_module_names} == {"numpy", "numpy.typing"}
