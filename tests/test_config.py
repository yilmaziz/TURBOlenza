"""
Tests for turbolenza.b2b.config: the naming and unit conventions of the solver
inputs. Nothing reads these values yet, so only the conventions themselves are checked.
"""

from __future__ import annotations

from dataclasses import fields

import pytest

from turbolenza.b2b import config
from turbolenza.b2b.geometry import CascadeSpec, make_cascade

CONFIG_CLASSES = (config.GridSettings, config.FlowConditions, config.IterationSettings, config.SolverConfig)


def test_every_angle_input_is_named_in_degrees():
    """A public angle input is in degrees and ends in _deg; no angle field lacks the suffix."""
    angles = [item.name for cls in CONFIG_CLASSES for item in fields(cls) if "angle" in item.name]
    assert angles == ["inlet_angle_deg", "exit_angle_guess_deg"]


def test_grid_extensions_are_in_blade_chords_and_default_to_one_each_side():
    """One chord of extension on each side, and no leading- or trailing-edge setting: where
    the blade stations begin and end follows from the blade and the stations. level (added
    2026-11-13, grid.py section 9) chooses a geometry-driven standard grid; None, the default,
    is the explicit n_passage x n_eta mode of every earlier case."""
    assert [item.name for item in fields(config.GridSettings)] == \
        ["n_passage", "n_eta", "upstream_extension_chords", "downstream_extension_chords", "level"]
    settings = config.GridSettings(n_passage=31, n_eta=21)
    assert settings.upstream_extension_chords == 1.0
    assert settings.downstream_extension_chords == 1.0
    assert settings.level is None


@pytest.mark.parametrize("physical_chord, one_chord", ((None, 1.0), (0.05, 0.05)))
def test_one_extension_chord_is_one_true_blade_chord(physical_chord, one_chord):
    """The extension length is extension_chords * cascade.active.chord: 1 for a chord-1
    cascade and physical_chord metres for a dimensional one, whatever the stagger."""
    settings = config.GridSettings(n_passage=31, n_eta=21)
    cascade = make_cascade(CascadeSpec(blade_type="NACA65", solidity=1.2, stagger_deg=40.0, cl0=0.8,
                                       t_max_over_c=0.08, physical_chord=physical_chord, surface_continuity="C1"))
    assert settings.upstream_extension_chords * cascade.active.chord == pytest.approx(one_chord, rel=1e-12)
    assert settings.downstream_extension_chords * cascade.active.chord == pytest.approx(one_chord, rel=1e-12)


# --- the thermodynamic conventions of the inputs (2026-11-13) -------------------------------------

from turbolenza.b2b import eos  # noqa: E402

STOPPING = dict(density_tolerance=1.0, max_density_iterations=1, kutta_tolerance=1.0, max_kutta_iterations=1)


def test_density_relaxation_defaults_to_the_eos_project_baseline():
    """0.5 is the project's under-relaxation (eos.DENSITY_RELAXATION, the one source of the
    number: config imports it) and relax_density's own default; 1.0 is Aungier's direct
    replacement. The stopping rules have no default, so they are given here."""
    settings = config.IterationSettings(**STOPPING)
    assert settings.density_relaxation == 0.5 == eos.DENSITY_RELAXATION
    assert config.DENSITY_RELAXATION is eos.DENSITY_RELAXATION
    default = next(item.default for item in fields(config.IterationSettings) if item.name == "density_relaxation")
    import inspect
    assert default == inspect.signature(eos.relax_density).parameters["relaxation"].default
    with pytest.raises(TypeError):
        config.IterationSettings()                   # no tolerance or iteration limit is chosen by default


def test_density_relaxation_of_one_is_aungier_direct_replacement():
    import numpy as np
    settings = config.IterationSettings(**STOPPING, density_relaxation=eos.AUNGIER_DIRECT_REPLACEMENT)
    assert settings.density_relaxation == 1.0
    old, target = np.array([1.0, 2.0]), np.array([1.7, 0.9])
    assert np.array_equal(eos.relax_density(old, target, relaxation=settings.density_relaxation), target)
    half = config.IterationSettings(**STOPPING)
    assert np.array_equal(eos.relax_density(old, target, relaxation=half.density_relaxation), (old + target) / 2.0)


def test_flow_conditions_name_their_frames():
    """The inlet totals are absolute-frame (Pt_in^abs, Tt_in^abs); the rotating-row inputs are
    the stream-surface radius and the absolute inlet tangential velocity, None by default."""
    names = [item.name for item in fields(config.FlowConditions)]
    assert names == ["inlet_mach", "inlet_angle_deg", "exit_angle_guess_deg", "inlet_absolute_total_pressure",
                     "inlet_absolute_total_temperature", "angular_speed", "stream_surface_radius",
                     "inlet_absolute_tangential_velocity"]
    assert "total_pressure" not in names and "total_temperature" not in names
    defaults = {item.name: item.default for item in fields(config.FlowConditions)}
    assert defaults["stream_surface_radius"] is None and defaults["inlet_absolute_tangential_velocity"] is None
    assert "ABSOLUTE" in config.__doc__ and "absolute" in config.FlowConditions.__doc__.lower()


def reference_of(conditions, gas):
    """The thermodynamic reference of the inputs, as the solver will build it: no angle is
    turned into a swirl."""
    total = eos.TotalState(pressure=conditions.inlet_absolute_total_pressure,
                           temperature=conditions.inlet_absolute_total_temperature)
    return eos.blade_row_reference(total, gas, omega=conditions.angular_speed, radius=conditions.stream_surface_radius,
                                   inlet_c_theta=conditions.inlet_absolute_tangential_velocity)


GAS = eos.CaloricallyPerfectGas(gamma=1.4, gas_constant=287.0)
STATIONARY = dict(inlet_mach=0.5, inlet_angle_deg=45.0, exit_angle_guess_deg=20.0, inlet_absolute_total_pressure=1e5,
                  inlet_absolute_total_temperature=300.0, angular_speed=0.0)


def test_stationary_conditions_need_no_radius_or_swirl():
    conditions = config.FlowConditions(**STATIONARY)
    ref = reference_of(conditions, GAS)
    assert ref.omega == 0.0 and ref.radius is None
    assert ref.rothalpy == GAS.cp * 300.0 == ref.relative_total_enthalpy
    assert eos.initial_density(GAS, ref) == pytest.approx(1e5 / (287.0 * 300.0), rel=1e-15)


@pytest.mark.parametrize("missing", ["stream_surface_radius", "inlet_absolute_tangential_velocity", "both"])
def test_rotating_conditions_without_radius_or_swirl_are_refused(missing):
    given = dict(stream_surface_radius=0.3, inlet_absolute_tangential_velocity=50.0)
    if missing == "both":
        given = {}
    else:
        given.pop(missing)
    conditions = config.FlowConditions(**{**STATIONARY, "angular_speed": 1000.0}, **given)
    with pytest.raises(eos.ThermodynamicStateError, match="radius|inlet_c_theta"):
        reference_of(conditions, GAS)


def test_rotating_conditions_with_radius_and_swirl_give_the_rothalpy():
    """I = cp Tt_abs - omega r C_theta_abs = 1004.5 * 300 - 1000 * 0.3 * 50 = 286 350."""
    conditions = config.FlowConditions(**{**STATIONARY, "angular_speed": 1000.0}, stream_surface_radius=0.3,
                                       inlet_absolute_tangential_velocity=50.0)
    assert reference_of(conditions, GAS).rothalpy == pytest.approx(286350.0, rel=1e-15)
