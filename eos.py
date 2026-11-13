"""
eos.py
======
The thermodynamic layer of the density loop: from the relative speed W at the nodes to the
static state, and the relaxed density for the next pass. A property and state layer only: it
calls no coefficient, matrix, solve or velocity function and runs no iteration.

    W (flow.py) -> h = H' - W^2 / 2 -> T, p, rho_EOS, a          static_state_from_velocity
    rho_next = (1 - lambda) rho + lambda rho_EOS                 relax_density

The gas: thermally and calorically perfect [CaloricallyPerfectGas]
    p = rho R T, constant specific heats, R > 0 and gamma > 1:
        cp = gamma R / (gamma - 1),   cv = R / (gamma - 1),   h = cp T (h = 0 at T = 0),
        a = sqrt(gamma R T).
    Isentropic between two states: cp ln(T2 / T1) - R ln(p2 / p1) = 0, so p2 = p1 (T2 / T1)^(gamma /
    (gamma - 1)). Entropy is never given an absolute value: every state here has the entropy of
    the inlet total state (inviscid, isentropic stream sheet), so
        p = Pt_in (T / Tt_in)^(gamma / (gamma - 1)),
    with TotalState (Pt_in, Tt_in) the specified inlet ABSOLUTE total state. Units: any one
    consistent system, as config.py says (SI: R in J/(kg K), T in K, p in Pa, W in m/s); the
    velocities of flow.py must be in it (physical mode, c_ref in metres and mdot' in kg/(m s)).

The energy relation [Aungier 2003, Ch. 5]
    h = H' - W^2 / 2 = I + (r omega)^2 / 2 - W^2 / 2,
    I the rothalpy, r the stream-surface radius, omega the blade-row angular speed. One stream
    surface, one radius, so H' = I + (r omega)^2 / 2 is one number over the blade-to-blade plane.
    BladeRowThermoReference holds (inlet total state, I, omega, r); blade_row_reference builds
        I = H_in - omega r C_theta_in,   H_in = cp Tt_in,
    C_theta_in the inlet ABSOLUTE tangential velocity, which a rotating row must be given (it
    is not inferred from a flow angle here). omega = 0 (a stationary cascade, no radius needed):
    I = H_in = cp Tt_in, H' = I and h = cp Tt_in - W^2 / 2, T = Tt_in - W^2 / (2 cp): the
    stationary formulas are the omega = 0 values of the one model, not a second one.

The static state (static_state_from_velocity), on scalars or (n_m, n_eta) arrays
    h = H' - W^2 / 2,  T = h / cp,  p = Pt_in (T / Tt_in)^(gamma / (gamma - 1)),  rho_EOS = p / (R T),
    a = sqrt(gamma R T). T <= 0 (more kinetic energy than H' holds, W >= sqrt(2 H')) is a
    ThermodynamicStateError: nothing is clipped, floored or limited.

The initial density (initial_density)
    Aungier's first guess is the local relative total density: the state at W = 0,
        Tt' = H' / cp,  Pt' = Pt_in (Tt' / Tt_in)^(gamma / (gamma - 1)),  rho' = Pt' / (R Tt'),
    which is static_state_from_velocity(0) itself. Stationary: rho(0) = Pt_in / (R Tt_in),
    uniform over the stream sheet.

The density update (relax_density)
    rho_target = rho_EOS(W(n)) of static_state_from_velocity; rho_next = (1 - lambda) rho(n) +
    lambda rho_target, 0 < lambda <= 1 (config.IterationSettings.density_relaxation's range).
        lambda = 1    Aungier: the new density replaces the old one directly (exactly
                      rho_target)
        lambda = 0.5  this project's baseline (DENSITY_RELAXATION): rho_next = (rho(n) +
                      rho_target) / 2, an under-relaxation added by us for robustness of the
                      nonlinear iteration; Aungier does not prescribe it.
    The two densities stay apart by name: rho_target (the gas model's answer to W(n)) is not
    the density of the next pass when lambda < 1, rho_next is.

The loop this serves (solver.py, not here)
    rho(n) -> coefficients, F Psi = Q, psi(n), W(n) (flow.py) -> rho_target = rho_EOS(W(n)) ->
    rho(n+1) = 0.5 rho(n) + 0.5 rho_target -> coefficients of pass n + 1 from rho(n+1), never from
    rho_target. Stopping rules and tolerances are convergence.py's, not chosen here.

Names kept for the modules that import them: PerfectGas is CaloricallyPerfectGas and GasState
is StaticState (config.py, flow.py's interfaces). The planned state_from_pT / state_from_hs
pair is replaced by the isentropic relation to the inlet total state, which needs no
absolute entropy.

Status: implemented 2026-11-13, under review, not frozen.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from numbers import Real

import numpy as np
from numpy.typing import NDArray

DENSITY_RELAXATION = 0.5            # this project's baseline lambda (Aungier: 1, direct replacement)
AUNGIER_DIRECT_REPLACEMENT = 1.0    # lambda that reproduces Aungier's direct density replacement

Value = float | NDArray[np.float64]


class ThermodynamicStateError(ValueError):
    """A gas, a reference state or a requested state that is not physical: R <= 0, gamma <= 1,
    a total pressure or temperature <= 0, a non-finite input, a negative speed, a rotating row
    without its radius or inlet swirl, T <= 0 from the energy relation, or a relaxation outside
    0 < lambda <= 1."""


@dataclass(frozen=True)
class CaloricallyPerfectGas:
    """p = rho R T with constant cp and cv (module docstring)."""

    gamma: float            # ratio of specific heats cp / cv, > 1
    gas_constant: float     # R = cp - cv, > 0

    def __post_init__(self) -> None:
        gamma, R = _real(self.gamma, "gamma"), _real(self.gas_constant, "the gas constant R")
        if not gamma > 1.0:
            raise ThermodynamicStateError(f"gamma = {gamma!r}: a perfect gas needs gamma > 1")
        if not R > 0.0:
            raise ThermodynamicStateError(f"R = {R!r}: the gas constant must be positive")
        object.__setattr__(self, "gamma", gamma)
        object.__setattr__(self, "gas_constant", R)

    @property
    def cp(self) -> float:
        return self.gamma * self.gas_constant / (self.gamma - 1.0)

    @property
    def cv(self) -> float:
        return self.gas_constant / (self.gamma - 1.0)

    def enthalpy(self, T: Value) -> Value:
        """h = cp T."""
        return self.cp * T

    def temperature(self, h: Value) -> Value:
        """T = h / cp."""
        return h / self.cp

    def isentropic_pressure(self, T: Value, total: TotalState) -> Value:
        """p at temperature T and the entropy of the total state: Pt (T / Tt)^(gamma / (gamma - 1))."""
        return total.pressure * (T / total.temperature) ** (self.gamma / (self.gamma - 1.0))

    def density(self, p: Value, T: Value) -> Value:
        """rho = p / (R T)."""
        return p / (self.gas_constant * T)

    def sound_speed(self, T: Value) -> Value:
        """a = sqrt(gamma R T)."""
        return np.sqrt(self.gamma * self.gas_constant * T)


PerfectGas = CaloricallyPerfectGas   # the name config.py and flow.py import


@dataclass(frozen=True)
class TotalState:
    """The specified inlet absolute total state: the entropy reference of the stream sheet."""

    pressure: float         # Pt_in > 0
    temperature: float      # Tt_in > 0

    def __post_init__(self) -> None:
        for name in ("pressure", "temperature"):
            value = _real(getattr(self, name), f"the total {name}")
            if not value > 0.0:
                raise ThermodynamicStateError(f"the total {name} is {value!r}: it must be positive")
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class BladeRowThermoReference:
    """The invariants of one blade row on one stream surface: the inlet total state (entropy),
    the rothalpy I, the angular speed omega and the stream-surface radius r (None for a
    stationary row, which needs none). Build it with blade_row_reference."""

    inlet_total_state: TotalState
    rothalpy: float
    omega: float
    radius: float | None

    def __post_init__(self) -> None:
        if not isinstance(self.inlet_total_state, TotalState):
            raise TypeError(f"inlet_total_state must be a TotalState, got {type(self.inlet_total_state).__name__}")
        object.__setattr__(self, "rothalpy", _real(self.rothalpy, "the rothalpy"))
        omega = _real(self.omega, "omega")
        object.__setattr__(self, "omega", omega)
        if self.radius is None:
            if omega != 0.0:
                raise ThermodynamicStateError("a rotating row (omega != 0) needs its stream-surface radius")
        else:
            radius = _real(self.radius, "the radius")
            if not radius > 0.0:
                raise ThermodynamicStateError(f"the stream-surface radius is {radius!r}: it must be positive")
            object.__setattr__(self, "radius", radius)

    @property
    def blade_speed(self) -> float:
        """U = r omega (0 for a stationary row)."""
        return 0.0 if self.omega == 0.0 else self.radius * self.omega

    @property
    def relative_total_enthalpy(self) -> float:
        """H' = I + U^2 / 2, one number over the stream sheet."""
        return self.rothalpy + 0.5 * self.blade_speed ** 2


def blade_row_reference(total: TotalState, gas: CaloricallyPerfectGas, *, omega: float = 0.0, radius: float | None = None,
                        inlet_c_theta: float | None = None) -> BladeRowThermoReference:
    """The reference of a blade row: I = cp Tt_in - omega r C_theta_in (module docstring). A
    stationary row (omega = 0) needs neither radius nor inlet swirl, and I = cp Tt_in; a
    rotating row needs both."""
    if not isinstance(total, TotalState):
        raise TypeError(f"total must be a TotalState, got {type(total).__name__}")
    omega = _real(omega, "omega")
    H_in = gas.enthalpy(total.temperature)
    if omega == 0.0:
        if radius is not None:
            _real(radius, "the radius")
        return BladeRowThermoReference(inlet_total_state=total, rothalpy=H_in, omega=0.0, radius=radius)
    if radius is None or inlet_c_theta is None:
        raise ThermodynamicStateError("a rotating row (omega != 0) needs radius and inlet_c_theta (the inlet absolute "
                                      "tangential velocity) for its rothalpy")
    radius, c_theta = _real(radius, "the radius"), _real(inlet_c_theta, "the inlet absolute tangential velocity")
    if not radius > 0.0:
        raise ThermodynamicStateError(f"the stream-surface radius is {radius!r}: it must be positive")
    return BladeRowThermoReference(inlet_total_state=total, rothalpy=H_in - omega * radius * c_theta, omega=omega,
                                   radius=radius)


@dataclass(frozen=True)
class StaticState:
    """The static state at a speed W: numbers for a scalar W, read-only arrays of W's shape
    otherwise."""

    temperature: Value
    pressure: Value
    density: Value          # rho_EOS: the gas model's density, the target of relax_density
    enthalpy: Value
    sound_speed: Value

    def __post_init__(self) -> None:
        for item in fields(self):
            value = getattr(self, item.name)
            if isinstance(value, np.ndarray):
                value.flags.writeable = False


GasState = StaticState   # the name flow.py's interfaces import


def static_state_from_velocity(velocity, gas: CaloricallyPerfectGas, reference: BladeRowThermoReference) -> StaticState:
    """T, p, rho_EOS, h and a at the relative speed W (a number or an array, e.g. FlowField.W):
    h = H' - W^2 / 2, then the isentropic perfect-gas state (module docstring). W must be
    finite and >= 0; T <= 0 anywhere is a ThermodynamicStateError."""
    if not isinstance(reference, BladeRowThermoReference):
        raise TypeError(f"the reference must be a BladeRowThermoReference, got {type(reference).__name__}")
    W = _array(velocity, "the speed W")
    if np.any(W < 0.0):
        raise ThermodynamicStateError(f"the speed W is negative at {int(np.count_nonzero(W < 0.0))} node(s): W is a "
                                      f"magnitude")
    H_rel = reference.relative_total_enthalpy
    h = H_rel - 0.5 * W ** 2
    T = gas.temperature(h)
    bad = ~(T > 0.0)
    if np.any(bad):
        raise ThermodynamicStateError(f"T <= 0 at {int(np.count_nonzero(bad))} node(s) (largest W {float(np.max(W)):.6g}): "
                                      f"the kinetic energy W^2 / 2 reaches the relative total enthalpy H' = {H_rel:.6g}, "
                                      f"i.e. W >= sqrt(2 H') = {math.sqrt(max(2.0 * H_rel, 0.0)):.6g}. Nothing is clipped")
    total = reference.inlet_total_state
    p = gas.isentropic_pressure(T, total)
    rho = gas.density(p, T)
    a = gas.sound_speed(T)
    return StaticState(temperature=_out(T), pressure=_out(p), density=_out(rho), enthalpy=_out(h), sound_speed=_out(a))


def initial_density(gas: CaloricallyPerfectGas, reference: BladeRowThermoReference, shape=None) -> Value:
    """Aungier's first density: the relative total density, the state at W = 0 (module
    docstring); Pt_in / (R Tt_in) for a stationary row. A number, or a uniform read-only array
    of the given shape."""
    rho = float(static_state_from_velocity(0.0, gas, reference).density)
    if shape is None:
        return rho
    field = np.full(tuple(int(n) for n in np.atleast_1d(shape)), rho)
    field.flags.writeable = False
    return field


def relax_density(density_old, density_target, relaxation: float = DENSITY_RELAXATION) -> NDArray[np.float64]:
    """rho_next = (1 - lambda) rho_old + lambda rho_target, a new read-only array (the inputs are
    not touched). lambda = DENSITY_RELAXATION = 0.5 by default, this project's baseline;
    lambda = 1 is Aungier's direct replacement. 0 < lambda <= 1; both densities finite,
    positive and of one shape."""
    lam = _real(relaxation, "the density relaxation")
    if not 0.0 < lam <= 1.0:
        raise ThermodynamicStateError(f"the density relaxation is {lam!r}: it must satisfy 0 < lambda <= 1")
    old, target = _array(density_old, "the old density"), _array(density_target, "the target density")
    if old.shape != target.shape:
        raise ThermodynamicStateError(f"the old density has shape {old.shape}, the target {target.shape}")
    for name, value in (("old", old), ("target", target)):
        if not np.all(value > 0.0):
            raise ThermodynamicStateError(f"the {name} density is not positive everywhere")
    out = (1.0 - lam) * old + lam * target
    out.flags.writeable = False
    return out


def _real(value, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (Real, np.floating, np.integer)):
        raise ThermodynamicStateError(f"{name} must be a real number, got {value!r}")
    value = float(value)
    if not math.isfinite(value):
        raise ThermodynamicStateError(f"{name} is {value!r}: it must be finite")
    return value


def _array(value, name: str) -> NDArray[np.float64]:
    try:
        array = np.array(value, dtype=float)
    except (TypeError, ValueError):
        raise ThermodynamicStateError(f"{name} must be numeric") from None
    if not np.all(np.isfinite(array)):
        raise ThermodynamicStateError(f"{name} is not finite at {int(np.count_nonzero(~np.isfinite(array)))} node(s)")
    return array


def _out(value) -> Value:
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        return float(array)
    array = np.array(array)
    array.flags.writeable = False
    return array
