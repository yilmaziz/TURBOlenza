"""
controller.py
=============
The user-facing run layer: one case from flat user settings, one timed solve, the report, the
figures and the files. It holds no numerics: it calls geometry.make_cascade, grid.make_grid,
flow.inlet_state, solver.solve / solver.solve_density_coupled, postprocess, plots and monitor,
and reads the time and CPU clocks. scripts/run_full_solver.py fills CaseSettings from the
constants at its top; a later GUI can fill the same CaseSettings and call the same functions.

    settings = CaseSettings(...)                     every user input, one flat field each
    case = build_case(settings)                      cascade, config, grid (admissibility checked
                                                     before any solve), inlet state, derived values
    print(format_case_summary(case))                 the input echo: [user], [derived], [project default]
    run = run_case(case, live_monitor=True, verbose=True)   the timed solve (RunResult)
    print(format_run_report(run))                    the result, or the failure diagnostics
    figures = plot_run(run)                          surface Mach (runtime footer), S, convergence
    directory = export_run(run, figures, "outputs")  outputs/<case>/<run label>/ PNG, CSV, JSON

Timing (RuntimeStats), around the solver call and nothing else
    wall_time_s           time.perf_counter() after minus before solver.solve (coupled_kutta)
                          or solver.solve_density_coupled (fixed_exit): every density and Kutta
                          pass, exit rebuild, assembly, linear solve, velocity recovery and EOS
                          update; not the case construction before it, not the report, figures
                          or files after it. The live monitor and the console lines run inside
                          the call: their share is measured and reported beside it
                          (observer_wall_time_s), not hidden in it.
    process_cpu_time_s    time.process_time() difference over the same call: the CPU time of
                          the whole process (all threads, user + system)
    effective_cpu_cores   process_cpu_time_s / wall_time_s: the average CPU use in equivalents
                          of fully busy cores (0.97: about one busy core). Not a number of
                          cores used.
    logical_cpus_available   os.cpu_count() (None when unknown): what the machine offers, not
                          what the solve used.
    Threads are not controlled here. Measured 2026-11-13 on the verification case (33 deg,
    M 0.3, 135 passes): 0.97 to 0.99 effective cores by default; with OMP_NUM_THREADS =
    OPENBLAS_NUM_THREADS = MKL_NUM_THREADS = 1 the result changes in its last bits (beta_out
    23.468434679791 -> 23.468434679798 deg; each setting deterministic run to run), so a
    thread limit is not answer-neutral and none is set. The thread variables present in the
    environment are recorded in the run summary.

Inputs and the project's policies (nothing is decided here that the production API decides)
    The mass flow: inlet_mach given and mass_flow_per_depth_override None -> derived from the
    inlet Mach number (flow.inlet_state, physical pitch = chord / solidity); both given -> they
    must agree to solver.MASS_FLOW_CONSISTENCY (checked here before the solve, and again by the
    solver); inlet_mach None -> the override alone (no inlet state, no pressure coefficient).
    The pitch, the inlet state, the angle of attack and the pass limit are derived, never
    entered. The grid: grid_level (one of grid.GRID_LEVELS, n_passage = n_eta = None) builds the
    geometry-driven standard grid of grid.py section 9 (grid.build_grid; Case.standard_grid
    keeps its metadata, search and timing, and the solver's make_grid(level) builds the same
    grid); grid_level None the explicit n_passage x n_eta grid, which must be admissible
    (grid.validate_uniform_grid_topology via make_grid(require_ss_ps=True)): an inadmissible
    n_passage stops build_case with the production message, which names the nearest
    admissible values.

Failures: run_case returns them, it does not swallow them. RunResult.error is the exception
(DensityNotConvergedError, CoupledNotConvergedError, TransonicFlowError, a thermodynamic or
linear-solver error, ...), RunResult.solution the partial solution it carries (or None);
format_run_report prints the failure diagnostics, export_run saves what exists, and
RunResult.raise_error() re-raises it.

Pressure plot: Dunavant's S = (Pt1 - p) / q1 (postprocess.DUNAVANT; P1 is the upstream TOTAL
pressure in Dunavant et al. 1955, q1 = rho1 W1^2 / 2), labeled as such.

Status: implemented 2026-11-13, under review.
"""

from __future__ import annotations

import csv
import json
import math
import os
import platform
import time
from dataclasses import MISSING, asdict, dataclass, fields
from datetime import datetime
from pathlib import Path

import numpy as np

from potential_flow_solver import config as cf
from potential_flow_solver import eos, postprocess
from potential_flow_solver import solver as sv
from potential_flow_solver.flow import InletState, InletStateError, inlet_state
from potential_flow_solver.geometry import Cascade, CascadeSpec, make_cascade
from potential_flow_solver.grid import Grid, GridError, StandardGrid, build_grid, make_grid

COUPLED, FIXED_EXIT = "coupled_kutta", "fixed_exit"
SOLVER_MODES = (COUPLED, FIXED_EXIT)
BLADE_TE_ANGLE, MANUAL = "blade_te_angle", "manual"
EXIT_ANGLE_MODES = (BLADE_TE_ANGLE, MANUAL)
THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
SUMMARY_FORMAT = "potential_flow_solver run summary, version 1"


class CaseError(ValueError):
    """The settings do not make a case: an unknown mode, a missing angle, an inadmissible grid,
    an inlet state that cannot be formed, contradicting mass-flow inputs."""


@dataclass(frozen=True)
class CaseSettings:
    """Every user input of one run, one flat field each, in the units of the comments; no
    default (scripts/run_full_solver.py and a future GUI state every value)."""

    # 0. run control
    case_name: str
    solver_mode: str                            # "coupled_kutta" (density + Kutta) or "fixed_exit"
    fixed_exit_angle_deg: float | None          # fixed_exit only; None: the blade TE angle kappa_te
    # 1. gas / thermodynamics
    gamma: float
    gas_constant: float                         # J/(kg K)
    inlet_total_pressure: float                 # Pa, absolute frame
    inlet_total_temperature: float              # K, absolute frame
    omega: float                                # rad/s; 0 (a rotating row is refused by the solver)
    stream_surface_radius: float | None         # m; rotating rows only
    inlet_absolute_tangential_velocity: float | None   # m/s; rotating rows only
    # 2. blade / cascade geometry (geometry.CascadeSpec)
    blade_family: str                           # CascadeSpec.blade_type
    design_lift_coefficient: float              # cl0 (camber)
    thickness_ratio: float                      # t_max_over_c (nominal)
    meanline_type: str
    profile_variant: str
    surface_continuity: str                     # "C2": the solver's geometry
    n_camber_stations: int
    chord: float                                # m: the true chord, the solver's c_ref
    solidity: float                             # chord / pitch
    stagger_deg: float                          # from the axial direction
    # 3. inlet flow
    inlet_mach: float | None                    # far-inlet relative Mach number; None: override only
    inlet_flow_angle_deg: float                 # from the axial direction, positive toward +y
    mass_flow_per_depth_override: float | None  # kg/(m s); None: derived from the inlet Mach number
    # 4. grid: a standard level (geometry-driven, grid.py section 9) or the explicit n_passage x n_eta
    grid_level: str | None                      # one of grid.GRID_LEVELS; None: explicit n_passage and n_eta
    n_passage: int | None                       # explicit mode only (None with grid_level)
    n_eta: int | None                           # explicit mode only (None with grid_level)
    upstream_extension_chords: float            # nominal with grid_level (whole cells, within 0.05 c)
    downstream_extension_chords: float
    # 5. density
    density_tolerance: float
    velocity_tolerance: float | None            # None: density_tolerance (project rule)
    max_density_iterations: int
    density_relaxation: float                   # 0.5 project baseline; 1.0 Aungier's direct replacement
    stop_at_sonic: bool                         # True (project default): stop at M_rel >= 1; False: go on (observation)
    # 6. Kutta
    kutta_tolerance: float
    max_kutta_iterations: int
    kutta_relaxation: float                     # 1.0 project baseline (undamped Aungier correction)
    initial_exit_angle_mode: str                # "blade_te_angle" or "manual" (coupled_kutta)
    initial_exit_angle_deg: float | None        # manual only
    # 7. runtime
    measure_runtime: bool


@dataclass(frozen=True)
class Case:
    """A built case: the production objects and the derived values (build_case)."""

    settings: CaseSettings
    cascade: Cascade
    config: cf.SolverConfig
    grid: Grid                                  # the grid of the first (coupled) or the fixed angle
    standard_grid: StandardGrid | None          # grid_level: what the level's grid is, its search and timing
    inlet: InletState | None
    blade_name: str
    pitch: float                                # m: chord * grid.pitch (= chord / solidity)
    angle_of_attack_deg: float                  # inlet angle - stagger
    kappa_le_deg: float
    kappa_te_deg: float
    exit_angle_deg: float                       # the first (coupled) or the fixed (fixed_exit) discharge angle
    exit_angle_source: str
    mass_flow_per_depth: float | None           # the expected mdot' (derived or override)
    mass_flow_source: str
    pass_limit: int


@dataclass(frozen=True)
class RuntimeStats:
    """The clocks around the solver call (module docstring, "Timing")."""

    wall_time_s: float
    process_cpu_time_s: float
    effective_cpu_cores: float
    logical_cpus_available: int | None
    observer_wall_time_s: float = 0.0           # live monitor and console lines, inside wall_time_s
    observer_cpu_time_s: float = 0.0


@dataclass(frozen=True)
class RunResult:
    """One timed solve: the solution (converged, or the partial one an error carries, or
    None), the error (None after a converged solve), the clocks."""

    case: Case
    solution: sv.Solution | None
    error: BaseException | None
    runtime: RuntimeStats | None
    monitor: object | None = None               # the LiveConvergenceMonitor, when one ran
    started: str = ""                           # ISO time of the solve

    @property
    def converged(self) -> bool:
        return self.error is None and self.solution is not None and self.solution.converged

    @property
    def status(self) -> str:
        return "converged" if self.converged else f"failed: {type(self.error).__name__}"

    def raise_error(self) -> None:
        """Re-raise the error of a failed run (nothing after a converged one)."""
        if self.error is not None:
            raise self.error


# --- case construction -----------------------------------------------------------------------------------------

def build_case(settings: CaseSettings) -> Case:
    """The production objects of settings and the derived values (module docstring)."""
    s = settings
    if s.solver_mode not in SOLVER_MODES:
        raise CaseError(f"solver_mode {s.solver_mode!r}: one of {SOLVER_MODES}")
    if s.initial_exit_angle_mode not in EXIT_ANGLE_MODES:
        raise CaseError(f"initial_exit_angle_mode {s.initial_exit_angle_mode!r}: one of {EXIT_ANGLE_MODES}")
    if not (isinstance(s.chord, (int, float)) and math.isfinite(s.chord) and s.chord > 0.0):
        raise CaseError(f"chord {s.chord!r}: a positive length in metres")
    cascade = make_cascade(CascadeSpec(blade_type=s.blade_family, solidity=s.solidity, stagger_deg=s.stagger_deg,
                                       cl0=s.design_lift_coefficient, t_max_over_c=s.thickness_ratio,
                                       meanline_type=s.meanline_type, profile_variant=s.profile_variant,
                                       physical_chord=None, n_camber_stations=s.n_camber_stations,
                                       surface_continuity=s.surface_continuity))
    kappa_te = math.degrees(cascade.kappa_te)
    if s.solver_mode == FIXED_EXIT:
        angle, angle_source = ((kappa_te, "fixed: the blade TE angle kappa_te (fixed_exit_angle_deg None)")
                               if s.fixed_exit_angle_deg is None else (float(s.fixed_exit_angle_deg), "fixed: user input"))
    elif s.initial_exit_angle_mode == BLADE_TE_ANGLE:
        angle, angle_source = kappa_te, "first Kutta angle: the blade TE angle kappa_te"
    else:
        if s.initial_exit_angle_deg is None:
            raise CaseError("initial_exit_angle_mode 'manual' needs initial_exit_angle_deg")
        angle, angle_source = float(s.initial_exit_angle_deg), "first Kutta angle: user input (manual)"
    conditions = cf.FlowConditions(
        inlet_mach=s.inlet_mach, inlet_angle_deg=s.inlet_flow_angle_deg, exit_angle_guess_deg=angle,
        inlet_absolute_total_pressure=s.inlet_total_pressure, inlet_absolute_total_temperature=s.inlet_total_temperature,
        angular_speed=s.omega, stream_surface_radius=s.stream_surface_radius,
        inlet_absolute_tangential_velocity=s.inlet_absolute_tangential_velocity)
    if s.grid_level is not None and (s.n_passage is not None or s.n_eta is not None):
        raise CaseError(f"grid_level {s.grid_level!r} sets the resolution: give n_passage = n_eta = None with it (got "
                        f"{s.n_passage!r}, {s.n_eta!r}), or grid_level None for the explicit n_passage x n_eta grid")
    grid_settings = cf.GridSettings(n_passage=s.n_passage, n_eta=s.n_eta,
                                    upstream_extension_chords=s.upstream_extension_chords,
                                    downstream_extension_chords=s.downstream_extension_chords, level=s.grid_level)
    iteration = cf.IterationSettings(density_tolerance=s.density_tolerance, max_density_iterations=s.max_density_iterations,
                                     kutta_tolerance=s.kutta_tolerance, max_kutta_iterations=s.max_kutta_iterations,
                                     density_relaxation=s.density_relaxation, velocity_tolerance=s.velocity_tolerance,
                                     kutta_relaxation=s.kutta_relaxation, stop_at_sonic=s.stop_at_sonic)
    gas = eos.CaloricallyPerfectGas(gamma=s.gamma, gas_constant=s.gas_constant)
    config = cf.SolverConfig(grid=grid_settings, flow=conditions, gas=gas, iteration=iteration)
    standard = None
    try:
        if s.grid_level is None:
            grid = make_grid(cascade, grid_settings, math.radians(s.inlet_flow_angle_deg), math.radians(angle),
                             require_ss_ps=True)
        else:                                   # the same grid make_grid(level) gives the solver, with its metadata
            standard = build_grid(cascade, s.grid_level, inlet_angle=math.radians(s.inlet_flow_angle_deg),
                                  exit_angle=math.radians(angle),
                                  extension_chords=(s.upstream_extension_chords, s.downstream_extension_chords))
            grid = standard.grid
    except GridError as error:
        which = (f"n_passage {s.n_passage}, n_eta {s.n_eta}" if s.grid_level is None else f"grid_level {s.grid_level}")
        raise CaseError(f"the grid cannot be used ({which}): {error}") from error
    pitch = float(s.chord) * grid.pitch
    inlet = None
    if s.inlet_mach is not None:
        try:
            inlet = inlet_state(conditions, gas, inlet_pitch=pitch)
        except InletStateError as error:
            raise CaseError(f"the inlet state cannot be derived: {error}") from error
    override = s.mass_flow_per_depth_override
    if override is None:
        if inlet is None:
            raise CaseError("no mass flow: give inlet_mach (normal use) or mass_flow_per_depth_override")
        mdot, source = inlet.mass_flow_per_depth, sv.MASS_FLOW_FROM_INLET
    elif inlet is None:
        mdot, source = float(override), sv.MASS_FLOW_OVERRIDE
    else:
        if not abs(float(override) - inlet.mass_flow_per_depth) <= sv.MASS_FLOW_CONSISTENCY * abs(inlet.mass_flow_per_depth):
            raise CaseError(f"mass_flow_per_depth_override {override!r} contradicts the inlet Mach number {s.inlet_mach!r} "
                            f"(mdot' {inlet.mass_flow_per_depth!r}); set inlet_mach None for a pure override "
                            f"(solver.MASS_FLOW_CONSISTENCY = {sv.MASS_FLOW_CONSISTENCY:g})")
        mdot, source = float(override), sv.MASS_FLOW_CHECKED
    limit = (min(s.max_density_iterations, s.max_kutta_iterations) if s.solver_mode == COUPLED
             else s.max_density_iterations)
    return Case(settings=s, cascade=cascade, config=config, grid=grid, standard_grid=standard, inlet=inlet,
                blade_name=blade_name(s),
                pitch=pitch, angle_of_attack_deg=s.inlet_flow_angle_deg - s.stagger_deg,
                kappa_le_deg=math.degrees(cascade.kappa_le), kappa_te_deg=kappa_te, exit_angle_deg=angle,
                exit_angle_source=angle_source, mass_flow_per_depth=mdot, mass_flow_source=source, pass_limit=limit)


def blade_name(settings: CaseSettings) -> str:
    """The designation, e.g. NACA 65-(12A10)10 (NACA 65 only; the family name otherwise)."""
    if settings.blade_family.upper() == "NACA65":
        return (f"NACA 65-({settings.design_lift_coefficient * 10:.0f}{settings.meanline_type})"
                f"{settings.thickness_ratio * 100:.0f}")
    return settings.blade_family


# --- the timed solve --------------------------------------------------------------------------------------------

class _TimedObservers:
    """The run's observers behind one monitor= object, with the time spent in them."""

    def __init__(self, items) -> None:
        self.items, self.wall, self.cpu = list(items), 0.0, 0.0
        if any(callable(getattr(item, "inspect", None)) for item in self.items):
            self.inspect = self._inspect        # only then does the solver hand out the pass fields

    def _call(self, method: str, *args) -> None:
        w0, c0 = time.perf_counter(), time.process_time()
        try:
            for item in self.items:
                getattr(item, method)(*args)
        finally:
            self.wall += time.perf_counter() - w0
            self.cpu += time.process_time() - c0

    def start(self, info) -> None:
        self._call("start", info)

    def update(self, record) -> None:
        self._call("update", record)

    def finish(self, solution, error) -> None:
        self._call("finish", solution, error)

    def _inspect(self, fields) -> None:         # handed on to the observers that define inspect
        w0, c0 = time.perf_counter(), time.process_time()
        try:
            for item in self.items:
                if callable(getattr(item, "inspect", None)):
                    item.inspect(fields)
        finally:
            self.wall += time.perf_counter() - w0
            self.cpu += time.process_time() - c0


def run_case(case: Case, *, live_monitor: bool = False, verbose: bool = False, monitor_title: str | None = None,
             initial_density=None, observers=()) -> RunResult:
    """Solve case (solver.solve for coupled_kutta, solver.solve_density_coupled for fixed_exit),
    timed (module docstring). Failures are returned, never swallowed: RunResult.error.
    initial_density: the start field of pass 0 instead of the method's own (a restart or a
    diagnostic start); observers: further observer objects (monitor.py's protocol), e.g. a
    diagnostic trace."""
    from potential_flow_solver.monitor import ConsoleProgress, LiveConvergenceMonitor

    s = case.settings
    live = LiveConvergenceMonitor(title=monitor_title or s.case_name) if live_monitor else None
    watching = [item for item in (live, ConsoleProgress() if verbose else None, *observers) if item is not None]
    timed = _TimedObservers(watching) if watching else None
    entry = sv.solve if s.solver_mode == COUPLED else sv.solve_density_coupled
    started = datetime.now().isoformat(timespec="seconds")
    wall0, cpu0 = time.perf_counter(), time.process_time()
    try:
        solution, error = entry(case.cascade, case.config, c_ref=float(s.chord),
                                mass_flow_per_depth=s.mass_flow_per_depth_override, monitor=timed,
                                initial_density=initial_density), None
    except sv.DensityIterationError as caught:
        solution, error = caught.solution, caught
    except Exception as caught:                 # returned to the caller, never swallowed; a called module's
        solution, error = getattr(caught, "solution", None), caught   # error carries the last completed pass
    wall, cpu = time.perf_counter() - wall0, time.process_time() - cpu0
    runtime = None
    if s.measure_runtime:
        runtime = RuntimeStats(wall_time_s=wall, process_cpu_time_s=cpu,
                               effective_cpu_cores=cpu / wall if wall > 0.0 else float("nan"),
                               logical_cpus_available=os.cpu_count(),
                               observer_wall_time_s=timed.wall if timed else 0.0,
                               observer_cpu_time_s=timed.cpu if timed else 0.0)
    return RunResult(case=case, solution=solution, error=error, runtime=runtime, monitor=live, started=started)


# --- console text ------------------------------------------------------------------------------------------------

def _defaults(cls) -> dict:
    return {item.name: item.default for item in fields(cls) if item.default is not MISSING}


def _tag(name: str, value, defaults: dict) -> str:
    return "[user = project default]" if name in defaults and defaults[name] == value else "[user]"


def format_case_summary(case: Case) -> str:
    """Every effective input after the derived values are evaluated, tagged [user], [derived]
    or [user = project default] (the default of the production dataclass)."""
    s = case.settings
    grid_defaults, iteration_defaults = _defaults(cf.GridSettings), _defaults(cf.IterationSettings)
    spec_defaults = _defaults(CascadeSpec)
    lines = [_rule("CASE INPUTS")]

    def row(label, value, tag="[user]", note=""):
        lines.append(f"  {label:34s}: {value!s:>24s}  {tag}{('  ' + note) if note else ''}")

    def head(text):
        lines.append(f" {text}")

    head("0. run control")
    row("case name", s.case_name)
    row("solver mode", s.solver_mode)
    if s.solver_mode == FIXED_EXIT:
        row("fixed exit angle [deg]", repr(s.fixed_exit_angle_deg))
    head("1. gas / thermodynamics")
    row("gamma", repr(s.gamma))
    row("gas constant R [J/(kg K)]", repr(s.gas_constant))
    row("Pt,in absolute [Pa]", repr(s.inlet_total_pressure))
    row("Tt,in absolute [K]", repr(s.inlet_total_temperature))
    row("omega [rad/s]", repr(s.omega), note="0: stationary cascade (radius and swirl unused)")
    row("stream-surface radius [m]", repr(s.stream_surface_radius))
    row("C_theta,in absolute [m/s]", repr(s.inlet_absolute_tangential_velocity))
    head("2. blade / cascade geometry")
    row("blade family", s.blade_family)
    row("design lift coefficient cl0", repr(s.design_lift_coefficient))
    row("thickness ratio t/c", repr(s.thickness_ratio))
    row("mean line", s.meanline_type, _tag("meanline_type", s.meanline_type, spec_defaults))
    row("profile variant", s.profile_variant, _tag("profile_variant", s.profile_variant, spec_defaults))
    row("surface continuity", s.surface_continuity, note="C2: the solver's geometry")
    row("thickness reconstruction", str(case.cascade.metadata.get("thickness_reconstruction")), "[derived]",
        "CascadeSpec default: C2 quintic_natural_sqrt, C1 legacy pchip_sqrt")
    row("camber stations", repr(s.n_camber_stations), _tag("n_camber_stations", s.n_camber_stations, spec_defaults))
    row("chord c [m]", repr(s.chord))
    row("solidity sigma", repr(s.solidity))
    row("stagger [deg]", repr(s.stagger_deg))
    row("blade", case.blade_name, "[derived]")
    row("pitch [m]", f"{case.pitch:.9g}", "[derived]", "chord / solidity")
    row("angle of attack [deg]", f"{case.angle_of_attack_deg:.6g}", "[derived]", "beta1 - stagger")
    row("kappa_le, kappa_te [deg]", f"{case.kappa_le_deg:.4f}, {case.kappa_te_deg:.4f}", "[derived]")
    head("3. inlet flow")
    row("inlet Mach number M1", repr(s.inlet_mach))
    row("inlet flow angle beta1 [deg]", repr(s.inlet_flow_angle_deg), note="from the axial direction")
    row("mass-flow override [kg/(m s)]", repr(s.mass_flow_per_depth_override))
    row("mass flow per depth [kg/(m s)]", f"{case.mass_flow_per_depth:.9g}", "[derived]", case.mass_flow_source)
    head("4. grid")
    row("grid level", repr(s.grid_level), note="None: explicit n_passage x n_eta" if s.grid_level is None else
        "geometry-driven standard grid (grid.py section 9)")
    row("n_passage", repr(s.n_passage))
    row("n_eta", repr(s.n_eta))
    row("upstream extension [chords]", repr(s.upstream_extension_chords),
        _tag("upstream_extension_chords", s.upstream_extension_chords, grid_defaults))
    row("downstream extension [chords]", repr(s.downstream_extension_chords),
        _tag("downstream_extension_chords", s.downstream_extension_chords, grid_defaults))
    head("5. density / nonlinear")
    row("density tolerance", repr(s.density_tolerance))
    row("velocity tolerance", repr(s.velocity_tolerance), _tag("velocity_tolerance", s.velocity_tolerance, iteration_defaults),
        "None: the density tolerance")
    row("max density iterations", repr(s.max_density_iterations))
    row("density relaxation lambda_rho", repr(s.density_relaxation),
        _tag("density_relaxation", s.density_relaxation, iteration_defaults), "0.5 ours; 1.0 = Aungier direct replacement")
    row("stop at M_rel >= 1", repr(s.stop_at_sonic), _tag("stop_at_sonic", s.stop_at_sonic, iteration_defaults),
        "" if s.stop_at_sonic else "OFF: the run goes on past sonic, outside the subsonic method")
    head("6. Kutta")
    row("Kutta tolerance (eps_K)", repr(s.kutta_tolerance))
    row("max Kutta iterations", repr(s.max_kutta_iterations))
    row("Kutta relaxation lambda_K", repr(s.kutta_relaxation),
        _tag("kutta_relaxation", s.kutta_relaxation, iteration_defaults), "1.0 ours: the undamped Aungier correction")
    row("initial exit-angle mode", s.initial_exit_angle_mode)
    row("initial exit angle [deg]", repr(s.initial_exit_angle_deg))
    row("exit angle used [deg]", f"{case.exit_angle_deg:.6f}", "[derived]", case.exit_angle_source)
    row("pass limit", repr(case.pass_limit), "[derived]",
        "min(max density, max Kutta)" if s.solver_mode == COUPLED else "max density iterations")
    head("7. compute / runtime")
    row("measure runtime", repr(s.measure_runtime))
    row("logical CPUs available", repr(os.cpu_count()), "[machine]")
    threads = {name: os.environ.get(name) for name in THREAD_VARIABLES if os.environ.get(name) is not None}
    row("thread variables set", repr(threads) if threads else "none", "[environment]", "no thread limit is set by the run")
    lines.append(_rule())
    return "\n".join(lines)


def format_inlet_state(case: Case) -> str:
    """The derived inlet state, before the solve."""
    lines = [_rule("INLET STATE")]
    inlet = case.inlet
    if inlet is None:
        lines.append(f"  no inlet state (inlet_mach None): mdot' = {case.mass_flow_per_depth!r} kg/(m s), "
                     f"{case.mass_flow_source}")
    else:
        for label, value in (("M1", f"{inlet.mach:.6f}"), ("beta1", f"{math.degrees(inlet.flow_angle):.3f} deg"),
                             ("T1", f"{inlet.static_temperature:.4f} K"), ("p1", f"{inlet.static_pressure:.2f} Pa"),
                             ("rho1", f"{inlet.static_density:.6f} kg/m^3"), ("a1", f"{inlet.sound_speed:.4f} m/s"),
                             ("W1", f"{inlet.velocity:.4f} m/s"), ("Wm1", f"{inlet.meridional_velocity:.4f} m/s"),
                             ("Wtheta1", f"{inlet.tangential_velocity:.4f} m/s"),
                             ("q1", f"{inlet.dynamic_pressure:.3f} Pa"), ("pitch S1", f"{inlet.pitch:.6g} m"),
                             ("mdot/depth", f"{case.mass_flow_per_depth:.6f} kg/(m s)"),
                             ("source", case.mass_flow_source)):
            lines.append(f"  {label:20s}: {value}")
    lines.append(_rule())
    return "\n".join(lines)


def format_grid_summary(case: Case) -> str:
    """The grid built for the case (admissible: make_grid(require_ss_ps=True) passed)."""
    g, s = case.grid, case.settings
    n_m, n_eta = len(g.m), len(g.eta)
    lines = [_rule("GRID")]
    extra = ()
    if case.standard_grid is not None:
        info, found = case.standard_grid.info, case.standard_grid.candidates
        extra = (("level", f"{info.level} (target dm/c {info.target_dm_over_chord:.5f}, actual {info.dm_over_chord:.5f}; "
                           f"{info.n_blade_cells} cells between the LE and TE faces)"),
                 ("(m_f - m_LE)/c", f"{info.le_gap_over_chord:.6f}"), ("(m_TE - m_l)/c", f"{info.te_gap_over_chord:.6f}"),
                 ("extensions [chords]", f"{info.upstream_extension_chords:.4f} up, {info.downstream_extension_chords:.4f} "
                                         f"down"),
                 ("candidates", f"{found.tested} tested, {found.rejected} rejected, accepted rank {found.accepted_rank}"),
                 ("grid generation", f"{case.standard_grid.timing['total']:.4f} s"))
    for label, value in extra + (("n_m", n_m), ("n_eta", n_eta), ("total nodes", n_m * n_eta),
                         ("first blade station", g.first_blade_i), ("last blade station", g.last_blade_i),
                         ("blade stations", g.last_blade_i - g.first_blade_i + 1),
                         ("dm", f"{g.dm:.6g} chords = {g.dm * s.chord:.6g} m"), ("deta", f"{g.deta:.6g}"),
                         ("admissibility", "admissible: first and last blade station each cut one suction and one "
                                           "pressure surface")):
        lines.append(f"  {label:20s}: {value}")
    lines.append(_rule())
    return "\n".join(lines)


def format_run_report(run: RunResult) -> str:
    """The structured result of a converged run, or the failure diagnostics of a failed one."""
    s = run.case.settings
    out = ["=" * 64, "SOLVER RESULT" if run.converged else "SOLVER FAILURE", "=" * 64, ""]

    def block(title, rows):
        out.append(title)
        for label, value in rows:
            out.append(f"  {label:26s}: {value}")
        out.append("")

    block("Case", [("name", s.case_name), ("mode", s.solver_mode), ("status", run.status)])
    sol = run.solution
    if not run.converged:
        last = sol.passes[-1] if sol is not None and sol.passes else None
        rows = [("termination reason", f"{type(run.error).__name__}: {run.error}")]
        if getattr(run.error, "failed_pass", None) is not None:
            rows.append(("failed in pass", f"{run.error.failed_pass} (the report shows the last completed pass)"))
        if not s.stop_at_sonic:
            rows.append(("sonic stop", "OFF (stop_at_sonic False)"))
        if last is not None:
            rows += [("last pass", f"{last.index}{' (final evaluation)' if last.final else ''}"),
                     ("last beta_out", f"{math.degrees(last.exit_angle):.6f} deg"),
                     ("last density change", f"{last.density_change:.3e}"),
                     ("last velocity change", f"{last.velocity_change:.3e}"),
                     ("last Kutta error eps_K", f"{last.kutta_normalized:.3e}"),
                     ("peak Mach (last pass)", f"{last.mach_max:.4f}"),
                     ("peak Mach (all passes)", f"{max(p.mach_max for p in sol.passes):.4f}")]
        else:
            rows.append(("last pass", "none recorded"))
        block("Failure", rows)
    if sol is not None:
        summary = postprocess.summarize(sol)
        surfaces = postprocess.blade_surfaces(sol)
        last = sol.passes[-1]
        if run.converged:
            block("Convergence", [("converged", "True"), ("coupled passes" if s.solver_mode == COUPLED else "density passes",
                                                         f"{sol.iterations} (+ final evaluation)"),
                                  ("density error", f"{last.density_change:.3e}"), ("velocity error", f"{last.velocity_change:.3e}"),
                                  ("Kutta error eps_K", f"{sol.normalized_kutta_residual:.3e}"),
                                  ("Kutta residual", f"{sol.kutta_residual:+.3e} m/s")])
        block("Flow" + ("" if run.converged else " (last pass, NOT CONVERGED)"),
              [("beta_in", f"{summary.inlet_angle_deg:.4f} deg"), ("beta_out", f"{summary.discharge_angle_deg:.6f} deg"),
               ("turning", f"{summary.turning_deg:.4f} deg"),
               ("max Mach", f"{summary.mach_max:.4f} at node {summary.mach_max_node}"),
               ("min/max density", f"{summary.density_min:.6g} / {summary.density_max:.6g} kg/m^3"),
               ("min/max pressure", f"{summary.pressure_min:.6g} / {summary.pressure_max:.6g} Pa"),
               ("min/max temperature", f"{summary.temperature_min:.6g} / {summary.temperature_max:.6g} K")])
        block("Surface", [("SS peak Mach", f"{summary.ss_peak_mach:.4f} at x/c = {summary.ss_peak_x_c:.4f}"),
                          ("PS peak Mach", f"{summary.ps_peak_mach:.4f} at x/c = {summary.ps_peak_x_c:.4f}"),
                          ("TE SS Mach", f"{surfaces.mach_ss[-1]:.4f}"), ("TE PS Mach", f"{surfaces.mach_ps[-1]:.4f}")])
        block("Linear algebra", [("||F Psi - Q||_inf", f"{last.linear_residual:.3e}")])
        block("Mass flow", [("mdot/depth", f"{sol.mass_flow_per_depth:.6f} kg/(m s)"), ("source", sol.mass_flow_source),
                            ("max station error", f"{last.mass_flow_error_max:.3e}"),
                            ("blade median error", f"{last.mass_flow_error_blade_median:.3e}")])
    block("Runtime", runtime_rows(run.runtime))
    out.append("=" * 64)
    return "\n".join(out)


def runtime_rows(runtime: RuntimeStats | None) -> list[tuple[str, str]]:
    if runtime is None:
        return [("solver wall time", "not measured (measure_runtime False)")]
    rows = [("solver wall time", f"{runtime.wall_time_s:.3f} s"), ("process CPU time", f"{runtime.process_cpu_time_s:.3f} s"),
            ("effective CPU cores", f"{runtime.effective_cpu_cores:.2f}"),
            ("logical CPUs available", "unknown" if runtime.logical_cpus_available is None else str(runtime.logical_cpus_available))]
    if runtime.observer_wall_time_s > 0.0:
        rows.append(("of which monitor/console", f"{runtime.observer_wall_time_s:.3f} s"))
    return rows


def _rule(title: str = "") -> str:
    return f"--- {title} ".ljust(64, "-") if title else "-" * 64


# --- figures ------------------------------------------------------------------------------------------------------

def runtime_annotation(runtime: RuntimeStats | None) -> str:
    """The runtime lines every surface Mach figure carries."""
    if runtime is None:
        return "Solver time: not measured (measure_runtime False)"
    monitor = (f" (monitor/console {runtime.observer_wall_time_s:.3f} s of it)"
               if runtime.observer_wall_time_s > 0.0 else "")
    cpus = "unknown" if runtime.logical_cpus_available is None else str(runtime.logical_cpus_available)
    return (f"Solver time: {runtime.wall_time_s:.3f} s{monitor}\n"
            f"Effective CPU use: {runtime.effective_cpu_cores:.2f} cores\n"
            f"Logical CPUs available: {cpus}")


def case_annotation(run: RunResult) -> str:
    """The compact case lines of the figures."""
    s, sol = run.case.settings, run.solution
    mach = "-" if s.inlet_mach is None else f"{s.inlet_mach:.3f}"
    lines = [f"M1 = {mach}   beta1 = {s.inlet_flow_angle_deg:.1f} deg   stagger = {s.stagger_deg:.1f} deg   "
             f"sigma = {s.solidity:.2f}",
             (f"grid = {s.n_passage} x {s.n_eta} (n_passage x n_eta; {len(run.case.grid.m)} x {len(run.case.grid.eta)} nodes)"
              if s.grid_level is None else
              f"grid = {s.grid_level} ({len(run.case.grid.m)} x {len(run.case.grid.eta)} nodes)")]
    if sol is not None:
        lines.append(f"beta2 = {math.degrees(sol.exit_angle):.3f} deg   iterations = {sol.iterations}   "
                     f"{s.solver_mode}{'' if run.converged else '   NOT CONVERGED: ' + type(run.error).__name__}")
    return "\n".join(lines)


def plot_run(run: RunResult, *, sonic_line: bool = True) -> dict:
    """The figures of a run that has a solution (converged or partial): surface_mach (always,
    with the runtime and case footer), surface_pressure (Dunavant's S; skipped without an inlet
    state), convergence. {} when the run has no solution."""
    import matplotlib.pyplot as plt

    from potential_flow_solver import plots

    sol = run.solution
    if sol is None:
        return {}
    figures = {}
    fig, ax = plt.subplots(figsize=(9.0, 6.6))
    fig.subplots_adjust(left=0.09, right=0.98, top=0.93, bottom=0.25)
    plots.plot_surface_mach(sol, ax=ax, sonic_line=sonic_line, title=f"{run.case.blade_name}: blade-surface Mach number"
                            + ("" if run.converged else "  [NOT CONVERGED]"))
    _footer(fig, run)
    figures["surface_mach"] = fig
    if sol.inlet is not None:
        fig, ax = plt.subplots(figsize=(9.0, 6.6))
        fig.subplots_adjust(left=0.09, right=0.98, top=0.93, bottom=0.25)
        plots.plot_pressure_coefficient(sol, ax=ax, convention=postprocess.DUNAVANT,
                                        title=f"{run.case.blade_name}: blade-surface pressure"
                                        + ("" if run.converged else "  [NOT CONVERGED]"))
        ax.set_ylabel("S = (Pt1 - p) / q1   [Dunavant et al. 1955]")
        _footer(fig, run)
        figures["surface_pressure"] = fig
    fig, _ = plots.plot_convergence(sol, run.case.config.iteration, title=run.case.settings.case_name)
    fig.subplots_adjust(bottom=0.14)
    fig.text(0.01, 0.005, runtime_annotation(run.runtime).replace("\n", "   |   "), fontsize=8, va="bottom")
    figures["convergence"] = fig
    figures["mach_contour"] = _mach_contour_figure(run)
    return figures


def _mach_contour_figure(run: RunResult):
    """The node-based Mach field (plots.plot_mach_contour): the whole domain with its periodic
    copies on the left, the blade passage closer on the right, one colour scale."""
    import matplotlib.pyplot as plt
    import numpy as np

    from potential_flow_solver import plots

    sol = run.solution
    g = sol.grid
    mach = np.asarray(sol.mach)
    low, high = float(np.min(mach)), float(np.max(mach))
    fig, (whole, close) = plt.subplots(1, 2, figsize=(12.5, 9.0), gridspec_kw=dict(width_ratios=(1.0, 1.0)))
    fig.subplots_adjust(left=0.06, right=0.97, top=0.9, bottom=0.14, wspace=0.18)
    state = "" if run.converged else "  [NOT CONVERGED]"
    fig.suptitle(f"{run.case.blade_name}: relative Mach number, node values with smooth (Gouraud) shading{state}",
                 fontsize=12)
    plots.plot_mach_contour(sol, ax=whole, colorbar=False, vmin=low, vmax=high,
                            title="whole computational domain (and its periodic copies)")
    walls = slice(g.first_blade_i, g.last_blade_i + 1)
    window = (g.m_le - 0.3 * g.chord, g.m_te + 0.3 * g.chord, float(np.min(g.nodes[walls, 0, 1])) - 0.3 * g.chord,
              float(np.max(g.nodes[walls, -1, 1])) + 0.3 * g.chord)
    plots.plot_mach_contour(sol, ax=close, window=window, vmin=low, vmax=high,
                            title=f"blade passage (peak M {high:.3f}; white line M = 1 where reached)")
    for ax in (whole, close):
        ax.set_xlabel("m / c (axial)")
        ax.set_ylabel("y / c (tangential)")
    _footer(fig, run)
    return fig


def _footer(fig, run: RunResult) -> None:
    box = dict(boxstyle="round,pad=0.4", facecolor="#f4f4f4", edgecolor="#bbbbbb")
    fig.text(0.01, 0.015, runtime_annotation(run.runtime), fontsize=9, va="bottom", ha="left", bbox=box,
             gid="runtime")
    fig.text(0.99, 0.015, case_annotation(run), fontsize=9, va="bottom", ha="right", bbox=box, gid="case")


# --- files --------------------------------------------------------------------------------------------------------

def export_run(run: RunResult, figures: dict, root, label: str | None = None, *, dpi: int = 130) -> Path:
    """Write the run into root/<case_name>/<label>/ (label None: the start time); an existing
    directory is never overwritten: -2, -3, ... is appended. Returns the directory."""
    base = Path(root) / _safe(run.case.settings.case_name)
    name = _safe(label) if label else (run.started or datetime.now().isoformat(timespec="seconds")).replace(":", "")
    directory, k = base / name, 2
    while directory.exists():
        directory, k = base / f"{name}-{k}", k + 1
    directory.mkdir(parents=True)
    written = []
    for key, fig in figures.items():
        path = directory / f"{key}.png"
        fig.savefig(path, dpi=dpi)
        written.append(path.name)
    if run.solution is not None:
        write_surface_csv(run, directory / "surface_mach.csv")
        written.append("surface_mach.csv")
        if run.solution.inlet is not None:
            write_surface_pressure_csv(run, directory / "surface_pressure.csv")
            written.append("surface_pressure.csv")
        write_convergence_csv(run, directory / "convergence_history.csv")
        written.append("convergence_history.csv")
    summary = run_summary(run)
    summary["files"] = written + ["run_summary.json"]
    (directory / "run_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    return directory


SURFACE_COLUMNS = ("surface", "x_c", "m_c", "mach", "pressure", "dunavant_S")
CONVERGENCE_COLUMNS = ("iteration", "final", "density_change", "velocity_change", "psi_change", "kutta_error",
                       "kutta_residual", "exit_angle_deg", "exit_angle_next_deg", "tan_correction", "linear_residual",
                       "peak_mach", "target_mismatch", "density_min", "density_max", "speed_min", "speed_max",
                       "mass_flow_error_max", "mass_flow_error_blade_median")


def write_surface_csv(run: RunResult, path) -> None:
    """One row per wall node of the blade stations: suction side, then pressure side."""
    sol = run.solution
    s = postprocess.blade_surfaces(sol)
    S = postprocess.surface_pressure_coefficient(sol) if sol.inlet is not None else None
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(SURFACE_COLUMNS)
        for name, x_c, mach, pressure, coefficient in (
                ("suction", s.x_c_ss, s.mach_ss, s.pressure_ss, None if S is None else S.values_ss),
                ("pressure", s.x_c_ps, s.mach_ps, s.pressure_ps, None if S is None else S.values_ps)):
            for k in range(len(x_c)):
                writer.writerow([name, repr(float(x_c[k])), repr(float(s.m_c[k])), repr(float(mach[k])),
                                 repr(float(pressure[k])), "" if coefficient is None else repr(float(coefficient[k]))])


def write_surface_pressure_csv(run: RunResult, path) -> None:
    """S = (Pt1 - p) / q1 and (p - p1) / q1 at every wall node."""
    sol = run.solution
    S = postprocess.surface_pressure_coefficient(sol)
    Cp = postprocess.surface_pressure_coefficient(sol, postprocess.INLET_STATIC)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("surface", "x_c", "dunavant_S", "inlet_static_cp"))
        for name, x_c, a, b in (("suction", S.x_c_ss, S.values_ss, Cp.values_ss),
                                ("pressure", S.x_c_ps, S.values_ps, Cp.values_ps)):
            for k in range(len(x_c)):
                writer.writerow([name, repr(float(x_c[k])), repr(float(a[k])), repr(float(b[k]))])


def write_convergence_csv(run: RunResult, path) -> None:
    """One row per PassRecord, the final evaluation included: every column of a row is that pass."""
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(CONVERGENCE_COLUMNS)
        for p in run.solution.passes:
            writer.writerow([p.index, int(p.final), *(repr(float(v)) for v in (
                p.density_change, p.velocity_change, p.psi_change, p.kutta_normalized, p.kutta_residual,
                math.degrees(p.exit_angle), math.degrees(p.exit_angle_next), p.tan_correction, p.linear_residual,
                p.mach_max, p.target_mismatch, p.density_min, p.density_max, p.speed_min, p.speed_max,
                p.mass_flow_error_max, p.mass_flow_error_blade_median))])


def run_summary(run: RunResult) -> dict:
    """The JSON-compatible record of a run: inputs, derived values, grid, results, runtime
    (module docstring); no field arrays."""
    import scipy

    case, sol = run.case, run.solution
    g = case.grid
    inlet = None
    if case.inlet is not None:
        inlet = asdict(case.inlet)
        inlet["flow_angle_deg"] = math.degrees(case.inlet.flow_angle)
    result = None
    if sol is not None:
        summary = postprocess.summarize(sol)
        surfaces = postprocess.blade_surfaces(sol)
        result = {"iterations": sol.iterations, "passes_recorded": len(sol.passes), "limitation": sol.limitation,
                  "beta_in_deg": summary.inlet_angle_deg, "beta_out_deg": summary.discharge_angle_deg,
                  "exit_velocity_angle_deg": summary.exit_velocity_angle_deg, "turning_deg": summary.turning_deg,
                  "kutta_residual_m_s": sol.kutta_residual, "normalized_kutta_residual": sol.normalized_kutta_residual,
                  "surface": {"ss_peak_mach": summary.ss_peak_mach, "ss_peak_x_c": summary.ss_peak_x_c,
                              "ps_peak_mach": summary.ps_peak_mach, "ps_peak_x_c": summary.ps_peak_x_c,
                              "te_ss_mach": float(surfaces.mach_ss[-1]), "te_ps_mach": float(surfaces.mach_ps[-1]),
                              "min_mach": summary.surface_mach_min, "max_mach": summary.surface_mach_max},
                  "field": {"max_mach": summary.mach_max, "max_mach_node": list(summary.mach_max_node),
                            "density_min": summary.density_min, "density_max": summary.density_max,
                            "pressure_min": summary.pressure_min, "pressure_max": summary.pressure_max,
                            "temperature_min": summary.temperature_min, "temperature_max": summary.temperature_max},
                  "mass_flow": {"per_depth_kg_m_s": sol.mass_flow_per_depth, "source": sol.mass_flow_source,
                                "max_station_error": summary.mass_flow_error_max,
                                "blade_median_error": summary.mass_flow_error_blade_median},
                  "final_pass": asdict(sol.passes[-1]), "peak_mach_all_passes": max(p.mach_max for p in sol.passes)}
    data = {
        "format": SUMMARY_FORMAT,
        "started": run.started,
        "status": run.status,
        "termination": {"converged": run.converged, "error_type": None if run.error is None else type(run.error).__name__,
                        "error_message": None if run.error is None else str(run.error)},
        "inputs": asdict(case.settings),
        "derived": {"blade_name": case.blade_name, "pitch_m": case.pitch, "angle_of_attack_deg": case.angle_of_attack_deg,
                    "kappa_le_deg": case.kappa_le_deg, "kappa_te_deg": case.kappa_te_deg,
                    "exit_angle_deg": case.exit_angle_deg, "exit_angle_source": case.exit_angle_source,
                    "mass_flow_per_depth_kg_m_s": case.mass_flow_per_depth, "mass_flow_source": case.mass_flow_source,
                    "pass_limit": case.pass_limit, "inlet_state": inlet},
        "grid": {"grid_level": case.settings.grid_level,
                 "standard_grid": None if case.standard_grid is None else asdict(case.standard_grid.info),
                 "grid_generation_s": None if case.standard_grid is None else case.standard_grid.timing["total"],
                 "n_passage": case.settings.n_passage, "n_eta_setting": case.settings.n_eta, "n_m": len(g.m),
                 "n_eta": len(g.eta), "total_nodes": len(g.m) * len(g.eta), "first_blade_station": g.first_blade_i,
                 "last_blade_station": g.last_blade_i, "blade_stations": g.last_blade_i - g.first_blade_i + 1,
                 "dm_chords": g.dm, "dm_m": g.dm * case.settings.chord, "deta": g.deta, "admissible": True},
        "results": result,
        "runtime": None if run.runtime is None else asdict(run.runtime),
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                        "platform": platform.platform(),
                        "thread_variables": {name: os.environ.get(name) for name in THREAD_VARIABLES}},
    }
    return _jsonable(data)


def _jsonable(value):
    """Plain JSON values: numpy scalars to Python numbers, NaN and infinities to None."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        return value if math.isfinite(value) else None
    return value


def _safe(name: str) -> str:
    keep = "".join(c if c.isalnum() or c in "-_." else "_" for c in str(name))
    return keep or "run"
