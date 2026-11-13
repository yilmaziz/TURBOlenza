"""
config.py
=========
Solver inputs and numerical settings as plain dataclasses. Nothing here computes anything;
the other modules read these values. The geometry has its own input object,
geometry.CascadeSpec, and nothing here repeats it.

Defaults
    Three, all conventions of the method rather than properties of one cascade: the grid
    extension lengths (a blade chord on each side), the density relaxation (0.5, this
    project's baseline, taken from eos.DENSITY_RELAXATION so the number lives in one place)
    and the Kutta relaxation (KUTTA_RELAXATION below, this project's baseline; Aungier
    recommends damping the Kutta correction but gives no value). Every other value is given
    by the caller, so a number chosen for one cascade cannot carry over unnoticed to the next;
    in particular no tolerance or iteration limit has a default. The velocity tolerance is
    optional: None (the default) applies the density tolerance to the velocity change too,
    the behaviour of the density loop before the field existed; it is no chosen number.

Frames
    The inlet total pressure and temperature are ABSOLUTE-frame totals, Pt_in^abs and
    Tt_in^abs: the entropy reference and, with the angular speed, the rothalpy reference of
    eos.blade_row_reference. The inlet Mach number and the flow angles are relative to the
    blade row. In a stationary cascade (angular_speed = 0) the two frames coincide.

Units
    An angle a person types is in degrees and its name ends in _deg; the solver converts
    it to radians once, and every angle it computes is in radians with no suffix. A length
    whose name ends in _chords is in true blade chords (cascade.active.chord each); other
    lengths are in the active length unit of the Cascade. Gas properties and total
    conditions are in any one consistent system (SI, or nondimensional).
"""

from __future__ import annotations

from dataclasses import dataclass

from turbolenza.b2b.eos import DENSITY_RELAXATION, PerfectGas


@dataclass(frozen=True)
class GridSettings:
    """Resolution and length of the computational grid.

    n_passage is the target resolution: the grid has one constant dm over its whole m range,
    as close to (m_te - m_le) / n_passage as the exact extension lengths allow (within half
    a cell over the whole range). So the number of cells between the exact LE and TE is
    n_passage give or take one, not exactly n_passage. n_eta is the number of nodes across
    the passage, uniform in eta from 0 to 1. Where the blade stations begin and end follows
    from the blade geometry and the stations (grid.py), so there is no leading- or
    trailing-edge setting.

    The extensions are counted in true blade chords: 1.0 is one chord, cascade.active.chord
    long (1 for a chord-1 cascade, physical_chord metres otherwise), never the chord's
    axial projection c cos(stagger). The upstream extension is therefore
    upstream_extension_chords * cascade.active.chord long, and the m range of the grid runs
    exactly from m_le - that length to m_te + the downstream one.

    Two modes (grid.py section 9, 2026-11-13). Explicit (legacy, verification and regression
    cases): n_passage and n_eta as above, level None. Geometry-driven standard grid: level one
    of grid.GRID_LEVELS ("very_coarse" .. "very_fine") and n_passage = n_eta = None; the
    level sets dm and n_eta, the stations are anchored to the blade's SS + PS-valid
    interval, and the extension lengths become nominal (the actual ones are whole cells,
    within grid.EXTENSION_TOLERANCE_CHORDS of them). make_grid refuses a mixture."""

    n_passage: int | None = None                # explicit mode: target number of cells along m from LE to TE
    n_eta: int | None = None                    # explicit mode: nodes across the passage, eta = 0 to 1
    upstream_extension_chords: float = 1.0      # upstream extension, in blade chords (nominal in level mode)
    downstream_extension_chords: float = 1.0    # downstream extension, in blade chords (nominal in level mode)
    level: str | None = None                    # standard-grid mode: one of grid.GRID_LEVELS; None: explicit mode


@dataclass(frozen=True)
class FlowConditions:
    """The operating point (module docstring, "Frames").

    Relative to the blade row: the inlet Mach number and the flow angles (measured from the
    meridional (axial) direction, positive toward +y, as in geometry.py).
    Absolute frame: the inlet total pressure and temperature, Pt_in^abs and Tt_in^abs, and the
    inlet tangential velocity C_theta_in^abs.
    A rotating row (angular_speed != 0) needs the stream-surface radius and C_theta_in^abs for
    its rothalpy I = cp Tt_in^abs - omega r C_theta_in^abs; a stationary one (angular_speed = 0)
    needs neither, and both stay None. They are never inferred from a flow angle here or in
    eos.py; eos.blade_row_reference refuses a rotating row without them."""

    inlet_mach: float | None                            # relative Mach number at the far inlet; None only when the
                                                        # solver is given mdot' itself (verification override, solver.py)
    inlet_angle_deg: float                              # relative flow angle at the far inlet, degrees
    exit_angle_guess_deg: float                         # first discharge angle tried, degrees; the Kutta loop moves it
    inlet_absolute_total_pressure: float                # Pt_in^abs, absolute-frame inlet total pressure
    inlet_absolute_total_temperature: float             # Tt_in^abs, absolute-frame inlet total temperature
    angular_speed: float                                # omega of the blade row, rad per unit time; 0 for a stationary cascade
    stream_surface_radius: float | None = None          # r of the stream surface; rotating rows only
    inlet_absolute_tangential_velocity: float | None = None  # C_theta_in^abs; rotating rows only


# The Kutta relaxation lambda_K of kutta.next_exit_angle, 0 < lambda_K <= 1: this project's baseline, chosen
# 2026-11-13 from a sensitivity study on the stationary verification case (README decision log). Aungier
# (2000, Sec. 13.3) recommends "suitable numerical damping" of the correction and gives no value; 1.0 is the
# undamped correction of Eqs. 13-57 and 13-59, which was stable and monotonic there.
KUTTA_RELAXATION = 1.0


@dataclass(frozen=True)
class IterationSettings:
    """Stopping rules and relaxations of the nonlinear iteration (see convergence.py). The
    stopping rules have no default (none is chosen yet); the relaxations default to the
    project baselines:
        density_relaxation lambda_rho, 0 < lambda_rho <= 1 (eos.relax_density enforces it):
            0.5  project default under-relaxation (eos.DENSITY_RELAXATION), our choice
            1.0  Aungier's direct density replacement (eos.AUNGIER_DIRECT_REPLACEMENT)
        kutta_relaxation lambda_K, 0 < lambda_K <= 1 (kutta.next_exit_angle enforces it):
            1.0  project default (KUTTA_RELAXATION), the undamped Aungier correction; Aungier
                 recommends damping and names no value, so any value is ours
    Tolerances (the change norms of convergence.py, all dimensionless):
        density_tolerance    change_norm(rho(n+1), rho(n)) of the last pass
        velocity_tolerance   the velocity change of W(n) against W(n-1); None (the default):
                             density_tolerance applies to it as well
        kutta_tolerance      eps_K = |W_SS - W_PS| / (0.5 (W_SS + W_PS)) at the TE station
                             (kutta.py), of the last pass
    Pass limits: solver.solve_at_exit_angle (fixed discharge angle) runs at most
    max_density_iterations passes; solver.solve (density + Kutta) makes one density update and
    one exit-angle correction in every pass, so both limits count its passes and it runs at
    most min(max_density_iterations, max_kutta_iterations).
    Sonic stop: stop_at_sonic True (the default, the verified behaviour) ends the loop with
    solver.TransonicFlowError at the first pass whose relative Mach number reaches 1 anywhere.
    False lets it go on, for observation only: the subsonic method does not apply there (no
    transonic extension), a SonicFlowWarning is issued once, and the Solution's limitation
    string says the sonic stop was off and M_rel >= 1 was reached."""

    density_tolerance: float        # density loop: change of rho per pass
    max_density_iterations: int     # density updates (passes) at most
    kutta_tolerance: float          # Kutta: eps_K, the normalized TE speed difference of kutta.py
    max_kutta_iterations: int       # exit-angle corrections (coupled passes) at most
    density_relaxation: float = DENSITY_RELAXATION  # lambda_rho, 0 < lambda_rho <= 1: 0.5 ours, 1.0 Aungier
    velocity_tolerance: float | None = None         # velocity change per pass; None: density_tolerance
    kutta_relaxation: float = KUTTA_RELAXATION      # lambda_K, 0 < lambda_K <= 1: ours (Aungier gives none)
    stop_at_sonic: bool = True                      # False: go on past M_rel >= 1 (observation only)


@dataclass(frozen=True)
class SolverConfig:
    """Everything the solver needs besides the geometry, which comes as a Cascade built
    from a geometry.CascadeSpec."""

    grid: GridSettings
    flow: FlowConditions
    gas: PerfectGas                 # gas model; any object with the methods of eos.py can stand in
    iteration: IterationSettings
