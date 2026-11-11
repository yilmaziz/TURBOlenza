"""
config.py
=========
Solver inputs and numerical settings as plain dataclasses. Nothing here computes anything;
the other modules read these values. The geometry has its own input object,
geometry.CascadeSpec, and nothing here repeats it.

Defaults
    Only the grid extension lengths have one: a blade chord on each side, a convention of
    the method rather than a property of one cascade. Every other value is given by the
    caller, so a number chosen for one cascade cannot carry over unnoticed to the next.

Units
    An angle a person types is in degrees and its name ends in _deg; the solver converts
    it to radians once, and every angle it computes is in radians with no suffix. A length
    whose name ends in _chords is in true blade chords (cascade.active.chord each); other
    lengths are in the active length unit of the Cascade. Gas properties and total
    conditions are in any one consistent system (SI, or nondimensional).
"""

from __future__ import annotations

from dataclasses import dataclass

from potential_flow_solver.eos import PerfectGas


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
    exactly from m_le - that length to m_te + the downstream one."""

    n_passage: int                              # target number of cells along m from leading to trailing edge
    n_eta: int                                  # nodes across the passage, eta = 0 to 1
    upstream_extension_chords: float = 1.0      # upstream extension, in blade chords
    downstream_extension_chords: float = 1.0    # downstream extension, in blade chords


@dataclass(frozen=True)
class FlowConditions:
    """The operating point. Velocities and Mach numbers are relative to the blade row,
    which in a stationary cascade is the same as absolute. Angles are measured from the
    meridional (axial) direction, positive toward +y, as in geometry.py."""

    inlet_mach: float               # Mach number at the far inlet
    inlet_angle_deg: float          # flow angle at the far inlet, degrees
    exit_angle_guess_deg: float     # first discharge angle tried, degrees; the Kutta loop moves it
    total_pressure: float           # inlet total pressure
    total_temperature: float        # inlet total temperature
    angular_speed: float            # rotation speed of the blade row in rad per unit time, 0 for a stationary cascade


@dataclass(frozen=True)
class IterationSettings:
    """Stopping rules and relaxation of the two nonlinear loops (see convergence.py)."""

    density_tolerance: float        # density loop stops when the change falls below this
    max_density_iterations: int
    density_relaxation: float       # fraction of each density update applied, 0 < w <= 1
    kutta_tolerance: float          # Kutta loop stops when |kutta_error| falls below this
    max_kutta_iterations: int


@dataclass(frozen=True)
class SolverConfig:
    """Everything the solver needs besides the geometry, which comes as a Cascade built
    from a geometry.CascadeSpec."""

    grid: GridSettings
    flow: FlowConditions
    gas: PerfectGas                 # gas model; any object with the methods of eos.py can stand in
    iteration: IterationSettings
