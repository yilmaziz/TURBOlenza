"""
solver.py
=========
Runs the method by calling the other modules in order. It holds no equations of its own, so
every step can be traced to the module that does it. Two entry points:

    solve                   density + Kutta coupled (implemented 2026-11-13): the density field
                            and the discharge angle converge together, with the Kutta condition
                            of Aungier 2003 Eq. 5-34 at the last blade station (kutta.py)
    solve_at_exit_angle     DENSITY-COUPLED / FIXED-DISCHARGE-ANGLE SOLUTION, NOT FINAL AUNGIER
                            KUTTA-CONVERGED SOLUTION: the density loop at the angle the grid was
                            built for (solve_density_coupled builds that grid first); kept as the
                            fixed-angle inner and verification solver
Both share one density pass (_density_pass). First scope: a stationary cascade (angular_speed
= 0); a rotating row is refused for now (eos.py's rothalpy reference supports one, but no
rotating case is validated).

The density pass, pass n (_density_pass; every step is a call to a verified module)
    rho(n)                                     the resident density: it builds this pass's equation
     3  coefficients.build_coefficients(metrics, rho(n), mdot', c_ref)          A .. E
     4  matrix_assembly.assemble_from_coefficients                              F Psi = Q
     5  linear_solver.solve_linear_system                                       Psi, psi(n)
     6  flow.velocity_components (psi(n), rho(n), the same coefficients)        W_m, W_theta, W
        flow.station_mass_flow                                                  diagnostic only
     7  eos.static_state_from_velocity(W(n))                                    T, p, a, and the
        rho_target(n+1) = rho_EOS(W(n))                                         target density
        Mach = W / a >= 1 anywhere: TransonicFlowError (no transonic extension, nothing clipped),
        unless config.iteration.stop_at_sonic is False (below, "Sonic stop off")
        eos.relax_density(rho(n), rho_target(n+1), lambda_rho)                  rho(n+1), the next
     8  convergence.change_norm / velocity_change / density_converged: converged when
        change_norm(rho(n+1), rho(n)) <= density_tolerance and the velocity change of W(n)
        against W(n-1) <= the velocity tolerance (velocity_tolerance, or density_tolerance
        when None); pass 0 has no previous W: never converged
    The next pass builds its coefficients from rho(n+1), never from rho_target. Three
    densities, kept apart by name: resident (built the pass), target (the gas model's answer
    to the pass's W), next (the relaxed density of the following pass).

The coupled pass of solve, pass n: resident rho(n) AND resident beta_out(n)
     1  grid(n) = grid.rebuild_exit(grid(0), beta_out(n)): only the exit stations' nodes move;
        the stations, eta, the blade run and every inlet and blade node are checked unchanged
        and the grid admissible (grid.validate_uniform_grid_topology), else ExitRebuildError
     2  metrics.build_metrics(grid(n)), boundary_conditions.build_boundary_conditions(grid(n)):
        rebuilt for every angle, nothing reused from another angle (the exit change reaches
        the metrics and A .. E of the exit stations and, through D, the last blade station;
        the boundary relations do not change but belong to their grid)
     3 .. 7  the density pass above on grid(n) with rho(n): psi(n), W(n), rho(n+1)
     8  kutta.kutta_residual(W(n), grid(n)): R_K = W_SS - W_PS and eps_K at i = last_blade_i;
        kutta.exit_angle_correction (Eqs. 13-57, 13-59 in this project's tan(beta) convention);
        kutta.next_exit_angle(beta_out(n), correction, lambda_K)               beta_out(n+1)
     9  convergence.coupled_converged: the density test of step 8 above AND eps_K(n) <=
        kutta_tolerance, on the same pass; pass 0 never converged
    The next pass starts from BOTH updated quantities (rho(n+1), beta_out(n+1)); no pass
    mixes a new angle with old metrics or a new density with old coefficients. The density
    array is carried node by node (i, j) to the rebuilt grid: the exit nodes move with the
    angle, the indices do not.

Relaxations (config.iteration)
    rho(n+1) = (1 - lambda_rho) rho(n) + lambda_rho rho_target(n+1), lambda_rho =
    density_relaxation (default 0.5 = eos.DENSITY_RELAXATION):
        lambda_rho = 1.0   Aungier's density iteration: direct replacement by the EOS density
        lambda_rho = 0.5   this project's baseline: under-relaxation added for the robustness
                           of the nonlinear loop (ours, not Aungier's)
    tan(beta_out(n+1)) = tan(beta_out(n)) + lambda_K Delta(tan(beta_out)), lambda_K =
    kutta_relaxation (default config.KUTTA_RELAXATION):
        Aungier recommends numerical damping of the Kutta correction and gives no value;
        the baseline is this project's choice from a sensitivity study (README).

Start and end
    rho(0) = eos.initial_density, Aungier's relative total density (Pt / (R Tt), uniform, for a
    stationary row); beta_out(0) = the angle of the grid (solve: exit_angle_guess_deg). When
    pass n converges, one more evaluation is made with rho(n+1) (and, coupled, beta_out(n+1) on
    its rebuilt grid) as its resident state, and the Solution reports that evaluation: its
    density, exit angle, grid, metrics, coefficients, psi, flow, static state and Kutta
    residual belong to one pass (never rho(n+1), beta_out(n+1) with psi(n), W(n)); its target
    density and the correction it would make next are reported beside it, small, not zero.
    No convergence within the pass limit (fixed angle: max_density_iterations; coupled:
    min(max_density_iterations, max_kutta_iterations), each coupled pass being one density
    update and one angle correction): DensityNotConvergedError (fixed angle) or
    CoupledNotConvergedError (coupled), carrying the last pass as a Solution with converged
    False; never a silent success.

Inputs and units
    config: grid settings, flow conditions (the far-inlet relative Mach number and angle;
    exit_angle_guess_deg, the fixed discharge angle of solve_density_coupled and the first
    angle of solve; absolute inlet totals; angular_speed), the gas, the iteration settings.
    c_ref, the true chord (m in SI), is given here: the cascade must be chord-normalized
    (physical_chord None), as coefficients.py requires, and c_ref is required and positive
    (the coupled thermodynamic solve needs one dimensional system, W in m/s against cp T in
    J/kg; the normalized c_ref = None of the lower modules is refused here). Angles are radians
    inside, from the meridional direction, positive toward +y, as in grid.py.

The mass flow (mdot' = mdot / b, one passage per unit stream-sheet depth, kg/(m s) in SI)
    Normal use: derived. flow.inlet_state(config.flow, gas, inlet_pitch = c_ref grid.pitch)
    closes the stationary far inlet from Pt1, Tt1, M1 and beta1 with eos.py's gas, and
    mdot' = rho1 W1 cos(beta1) S1, S1 the physical pitch (the far station is periodic). The
    InletState is kept in the Solution (postprocess.py's pressure coefficient needs it).
    Verification override: mass_flow_per_depth given.
        inlet_mach None      the given mdot' is used as it is ("override"); the Solution has no
                             inlet state, so no pressure coefficient
        inlet_mach given     both describe the inlet: the given mdot' must equal the derived
                             one to MASS_FLOW_CONSISTENCY (1e-12 relative, round-off), else
                             SolverInputError; it is used ("override, checked")
    Neither: SolverInputError. M1 = 0 (zero flow): SolverInputError, the policy of
    coefficients.py, which refuses a zero mass flow; the inlet helper itself returns 0.

Sonic stop off (config.iteration.stop_at_sonic False; observation only)
    The loop records a pass with M_rel >= 1 and goes on, with one SonicFlowWarning at the first
    such pass; nothing else changes (no clipping, no transonic model). The subsonic method does
    not apply there: the limitation string of every Solution of such a run says the stop was
    off and M_rel >= 1 was reached. Such a run typically ends in an error of a called module
    (the speeds grow until T <= 0 in eos.py: measured on the NACA 65 case at M1 0.65).

Errors of the called modules in the middle of a run (a thermodynamic state with T <= 0, a
singular system, an exit rebuild, ...) keep their type and gain two attributes: .solution, the
last completed pass as a Solution (converged False; None if the first pass failed), and
.failed_pass, the index of the pass that failed. So the history up to the failure is kept.

Observers (monitor.py): live_monitor=True opens one live convergence window per solve,
verbose=True prints one line per pass, monitor= takes any observer object. They receive the
frozen PassRecord of every pass and the finished Solution and never feed back: the solution is
the same with or without them. Off by default; Matplotlib is imported only for live_monitor.
An observer that also defines inspect(fields) is handed the PassFields of every pass: the
pass's own arrays as read-only views (nothing is copied or built for it), for a diagnostic
trace of chosen nodes; no observer of the package defines it.

Initial guesses (restart / diagnosis; the defaults are the method's own start)
    initial_density (solve, solve_density_coupled; density of solve_at_exit_angle): the
    resident density of pass 0 instead of eos.initial_density, an array of the grid's shape.
    The first discharge angle is config.flow.exit_angle_guess_deg. Neither changes an equation:
    only where the iteration starts.

Diagnostics per pass (PassRecord, numbers only): the changes, the target mismatch, the
resident density, speed and Mach ranges, the linear residual, the station mass-flow errors,
the pass's exit angle, its Kutta residual R_K and eps_K, the next angle and the undamped
correction. At a fixed angle no correction is made (NaN, next angle = the angle), and R_K and
eps_K are NaN when that grid's TE station is not one SS and one PS node. The PassRecord is the
authoritative record of a pass: History's lists are appended from it, one entry each per pass
(exit_angle and kutta_error in the coupled loop only), so index n of every list is pass n.

Status: density loop implemented 2026-11-13, under review, not frozen; the coupled density +
Kutta loop implemented 2026-11-13, under review, verified on a constructed case only, not yet
validated (the NACA / Dunavant comparison is the next stage).
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field, replace

import numpy as np
from numpy.typing import NDArray

from potential_flow_solver import convergence, eos, kutta
from potential_flow_solver.boundary_conditions import BoundaryConditions, build_boundary_conditions
from potential_flow_solver.coefficients import CoefficientFields, build_coefficients
from potential_flow_solver.config import SolverConfig
from potential_flow_solver.convergence import History
from potential_flow_solver.flow import (FlowField, InletState, InletStateError, inlet_state, station_mass_flow,
                                        velocity_components)
from potential_flow_solver.geometry import Cascade
from potential_flow_solver.grid import Grid, make_grid, rebuild_exit, validate_uniform_grid_topology
from potential_flow_solver.linear_solver import LinearSolveResult, solve_linear_system
from potential_flow_solver.matrix_assembly import assemble_from_coefficients
from potential_flow_solver.metrics import GridMetrics, build_metrics
from potential_flow_solver.monitor import MonitorWarning, SolveInfo

LIMITATION = "DENSITY-COUPLED / FIXED-DISCHARGE-ANGLE SOLUTION; NOT FINAL AUNGIER KUTTA-CONVERGED SOLUTION"
MASS_FLOW_CONSISTENCY = 1e-12   # relative: a given mdot' and the one of the inlet Mach number agree to round-off
MASS_FLOW_FROM_INLET = "derived from the inlet Mach number"
MASS_FLOW_OVERRIDE = "override (no inlet Mach number)"
MASS_FLOW_CHECKED = "override, checked against the inlet Mach number"
COUPLED_LIMITATION = ("DENSITY + KUTTA COUPLED SOLUTION (Aungier 2003 Eq. 5-34 at the last blade station, exit angle by "
                      "Aungier 2000 Eqs. 13-57 and 13-59); VERIFIED ON A CONSTRUCTED CASE ONLY, NOT YET VALIDATED "
                      "AGAINST EXPERIMENT OR PUBLISHED RESULTS")


class SolverInputError(ValueError):
    """The inputs do not describe a case this solver handles: no positive physical chord
    c_ref, a rotating row, a cascade that is not chord-normalized, a start density of the wrong
    shape, no pass allowed."""


class DensityIterationError(RuntimeError):
    """The iteration stopped without a converged solution; .solution is the last pass
    (converged False)."""

    def __init__(self, message: str, solution: Solution | None = None) -> None:
        super().__init__(message)
        self.solution = solution


class DensityNotConvergedError(DensityIterationError):
    """max_density_iterations passes at a fixed angle without meeting the tolerances."""


class CoupledNotConvergedError(DensityIterationError):
    """The coupled density + Kutta loop used its passes without meeting the density, velocity
    and Kutta tolerances on one pass."""


class TransonicFlowError(DensityIterationError):
    """The relative Mach number reached 1 somewhere: the subsonic density iteration does not
    apply, and Aungier's transonic extension is not implemented."""


class SonicFlowWarning(RuntimeWarning):
    """The relative Mach number reached 1 with the sonic stop off: the loop goes on outside the
    subsonic method (module docstring, "Sonic stop off")."""


class ExitRebuildError(RuntimeError):
    """Turning the exit extension to a new discharge angle changed what the Kutta loop must
    keep: the stations, eta, the blade run, an inlet or blade node, or the grid's
    admissibility."""


@dataclass(frozen=True)
class PassRecord:
    """The numbers of one pass (no fields)."""

    index: int                              # pass n
    final: bool                             # the evaluation at the converged state, not a loop pass
    density_change: float                   # change_norm(rho(n+1), rho(n))
    velocity_change: float                  # change of W(n) against W(n-1); NaN on pass 0
    psi_change: float                       # change_norm(psi(n), psi(n-1)); NaN on pass 0
    target_mismatch: float                  # change_norm(rho_target(n+1), rho(n))
    density_min: float                      # resident rho(n)
    density_max: float
    speed_min: float                        # W(n)
    speed_max: float
    mach_max: float                         # max W(n) / a(n)
    linear_residual: float                  # ||F Psi - Q||_inf
    mass_flow_error_max: float              # max over the stations of |station mass flow / mdot' - 1|
    mass_flow_error_blade_median: float     # its median over the blade stations
    exit_angle: float                       # radians: the pass's discharge angle, grid(n).exit_angle
    kutta_residual: float                   # R_K = W_SS - W_PS of W(n) at the TE station; NaN if not formed
    kutta_normalized: float                 # eps_K of W(n); NaN if not formed
    exit_angle_next: float                  # radians: the next pass's angle (= exit_angle at a fixed angle)
    tan_correction: float                   # undamped Delta(tan(beta_out)) of Eq. 13-59; NaN at a fixed angle


@dataclass(frozen=True)
class PassFields:
    """The fields of one pass, for an observer that defines inspect(fields) (module docstring,
    "Observers"): the pass's own arrays as read-only views. A diagnostic hand-out; nothing in
    the solve reads it back."""

    record: PassRecord
    grid: Grid                              # the pass's grid (its exit angle)
    metrics: GridMetrics
    density_resident: NDArray[np.float64]   # rho(n)
    density_target: NDArray[np.float64]     # rho_EOS(W(n))
    density_next: NDArray[np.float64]       # rho(n+1)
    coefficients: CoefficientFields         # A .. E of rho(n)
    psi: NDArray[np.float64]
    flow: FlowField
    static_state: eos.StaticState
    mach: NDArray[np.float64]
    mass_flow_per_depth: float
    c_ref: float


@dataclass(frozen=True)
class Solution:
    """The result and how it was reached (module docstring). density, exit_angle, grid,
    metrics, coefficients, linear, psi, flow, static_state, mach and kutta all belong to one
    evaluation: the one at the converged state. limitation says which solver made it."""

    grid: Grid                              # the grid of the reported evaluation (its exit angle)
    metrics: GridMetrics
    psi: NDArray[np.float64]                # (n_m, n_eta), linear.psi
    flow: FlowField
    history: History
    density: NDArray[np.float64]            # the resident density of the reported evaluation
    density_target: NDArray[np.float64]     # rho_EOS of its W
    static_state: eos.StaticState
    mach: NDArray[np.float64]               # relative Mach number W / a
    coefficients: CoefficientFields
    linear: LinearSolveResult
    passes: tuple[PassRecord, ...]
    converged: bool
    iterations: int                         # loop passes (the final evaluation not counted)
    mass_flow_per_depth: float
    c_ref: float
    density_relaxation: float
    exit_angle: float                       # radians, grid.exit_angle of the reported evaluation
    kutta: kutta.KuttaResidual | None       # Eq. 5-34 of the reported flow (station, SS/PS nodes, speeds)
    kutta_residual: float                   # kutta.residual, R_K = W_SS - W_PS; NaN if not formed
    normalized_kutta_residual: float        # kutta.normalized, eps_K; NaN if not formed
    kutta_relaxation: float | None = None   # lambda_K of the coupled loop; None at a fixed angle
    limitation: str = LIMITATION
    inlet: InletState | None = None         # the far-inlet state of the inlet Mach number; None for an override alone
    mass_flow_source: str = MASS_FLOW_OVERRIDE


@dataclass(frozen=True)
class _Context:
    """What every pass of one loop shares; metrics and boundary_conditions are those of the
    pass's grid (the coupled loop replaces them for every angle)."""

    metrics: GridMetrics
    boundary_conditions: BoundaryConditions
    gas: object
    reference: eos.BladeRowThermoReference
    mass_flow_per_depth: float
    c_ref: float
    relaxation: float
    blade: slice                            # first_blade_i .. last_blade_i
    kutta_relaxation: float | None = None   # lambda_K; None: fixed angle, no correction
    inlet: InletState | None = None
    mass_flow_source: str = MASS_FLOW_OVERRIDE
    observers: _Observers = field(default_factory=lambda: _Observers(()))
    stop_at_sonic: bool = True              # config.iteration.stop_at_sonic


@dataclass(frozen=True)
class _Pass:
    """One pass: the resident state and everything built from it."""

    density_resident: NDArray[np.float64]
    coefficients: CoefficientFields
    linear: LinearSolveResult
    flow: FlowField
    state: eos.StaticState
    mach: NDArray[np.float64]
    density_target: NDArray[np.float64]
    density_next: NDArray[np.float64]
    mass_flow_error: NDArray[np.float64]    # (n_m,)
    kutta: kutta.KuttaResidual | None = None
    correction: kutta.ExitAngleCorrection | None = None
    exit_angle: float = math.nan            # the resident angle, grid(n).exit_angle
    exit_angle_next: float = math.nan


def solve(cascade: Cascade, config: SolverConfig, *, c_ref: float, mass_flow_per_depth: float | None = None,
          live_monitor: bool = False, verbose: bool = False, monitor=None,
          initial_density: NDArray[np.float64] | None = None) -> Solution:
    """The density + Kutta coupled solution (module docstring): make_grid at the relative
    inlet angle and the first discharge angle config.flow.exit_angle_guess_deg (admissible
    n_passage required), then the coupled loop, in which every pass updates the density and
    corrects the discharge angle until the density, velocity and Kutta tolerances hold on one
    pass. The converged discharge angle is Solution.exit_angle (radians). The mass flow comes
    from config.flow.inlet_mach; mass_flow_per_depth is the verification override (module
    docstring). live_monitor, verbose and monitor: observers, off by default. initial_density:
    the start field of pass 0 instead of eos.initial_density (a restart or a diagnostic start).

    Raises SolverInputError for inputs outside this solver's scope, TransonicFlowError when a
    relative Mach number reaches 1, CoupledNotConvergedError after the pass limit,
    ExitRebuildError if an exit rebuild changes the topology, and what the called modules
    raise."""
    grid = _first_grid(cascade, config)
    return _solve_coupled(grid, config, mass_flow_per_depth=mass_flow_per_depth, c_ref=c_ref, density=initial_density,
                          observers=_observers(live_monitor, verbose, monitor))


def solve_density_coupled(cascade: Cascade, config: SolverConfig, *, c_ref: float,
                          mass_flow_per_depth: float | None = None, live_monitor: bool = False, verbose: bool = False,
                          monitor=None, initial_density: NDArray[np.float64] | None = None) -> Solution:
    """make_grid with the relative inlet angle and the fixed discharge angle
    config.flow.exit_angle_guess_deg (admissible n_passage required), then
    solve_at_exit_angle (initial_density is its start field). DENSITY-COUPLED /
    FIXED-DISCHARGE-ANGLE: no Kutta condition."""
    grid = _first_grid(cascade, config)
    return solve_at_exit_angle(grid, config, mass_flow_per_depth=mass_flow_per_depth, c_ref=c_ref,
                               density=initial_density, live_monitor=live_monitor, verbose=verbose, monitor=monitor)


def solve_at_exit_angle(grid: Grid, config: SolverConfig, *, c_ref: float, mass_flow_per_depth: float | None = None,
                        density: NDArray[np.float64] | None = None, history: History | None = None,
                        live_monitor: bool = False, verbose: bool = False, monitor=None) -> Solution:
    """Steps 3 to 8 at the discharge angle grid was built for (module docstring): the
    compressible solution without the Kutta condition. density is the start field (default
    eos.initial_density); every pass is appended to history (a new one by default). The mass
    flow and the observers as in solve.

    Raises SolverInputError for inputs outside this solver's scope, TransonicFlowError when a
    relative Mach number reaches 1, DensityNotConvergedError after max_density_iterations
    passes, and what the called modules raise (an invalid thermodynamic state, a singular
    system, ...)."""
    context = _context(grid, config, mass_flow_per_depth, c_ref)
    rho = _start_density(grid, context, density)
    settings = config.iteration
    limit = _limit(settings.max_density_iterations, "max_density_iterations")
    history = History() if history is None else history
    context = replace(context, observers=_observers(live_monitor, verbose, monitor))

    def loop() -> Solution:
        records: list[PassRecord] = []
        previous: _Pass | None = None
        current_rho, n = rho, 0
        try:
            for n in range(limit):
                current = _checked_pass(current_rho, context, grid, history, records, previous, n, final=False)
                previous = current
                if convergence.density_converged(history, settings):
                    n += 1
                    final = _checked_pass(current.density_next, context, grid, history, records, current, n, final=True)
                    return _solution(grid, context, final, history, records, True, n)
                current_rho = current.density_next
        except DensityIterationError:
            raise
        except Exception as error:              # a called module failed: keep the history (module docstring)
            _attach(error, (grid, context), previous, history, records, n)
            raise
        raise DensityNotConvergedError(f"the density loop did not converge in {limit} passes (last density change "
                                       f"{history.density_change[-1]:.3e}, velocity change "
                                       f"{history.velocity_change[-1]:.3e}, tolerance {settings.density_tolerance:g})",
                                       _solution(grid, context, previous, history, records, False, limit))

    return _observed(context.observers, _info("fixed angle", settings, limit, grid, context, None), loop)


def _solve_coupled(grid: Grid, config: SolverConfig, *, mass_flow_per_depth: float | None, c_ref: float,
                   density: NDArray[np.float64] | None = None, history: History | None = None,
                   observers: _Observers | None = None) -> Solution:
    """The coupled loop of solve from a built grid, whose exit angle is beta_out(0) (module
    docstring)."""
    base = _context(grid, config, mass_flow_per_depth, c_ref)
    settings = config.iteration
    base = replace(base, kutta_relaxation=settings.kutta_relaxation,
                   observers=observers if observers is not None else _Observers(()))
    validate_uniform_grid_topology(grid)
    rho0 = _start_density(grid, base, density)
    limit = min(_limit(settings.max_density_iterations, "max_density_iterations"),
                _limit(settings.max_kutta_iterations, "max_kutta_iterations"))
    history = History() if history is None else history

    def loop() -> Solution:
        records: list[PassRecord] = []
        previous: _Pass | None = None
        last = None                             # (grid, context) of the last completed pass
        rho, beta, n = rho0, grid.exit_angle, 0
        try:
            for n in range(limit):
                current_grid, context = _at_angle(grid, beta, base)
                current = _checked_pass(rho, context, current_grid, history, records, previous, n, final=False)
                previous, last = current, (current_grid, context)
                if convergence.coupled_converged(history, settings):
                    n += 1
                    final_grid, final_context = _at_angle(grid, current.exit_angle_next, base)
                    final = _checked_pass(current.density_next, final_context, final_grid, history, records, current, n,
                                          final=True)
                    return _solution(final_grid, final_context, final, history, records, True, n)
                rho, beta = current.density_next, current.exit_angle_next
        except DensityIterationError:
            raise
        except Exception as error:              # a called module failed: keep the history (module docstring)
            _attach(error, last, previous, history, records, n)
            raise
        current_grid, context = last
        raise CoupledNotConvergedError(
            f"the coupled density + Kutta loop did not converge in {limit} passes (last density change "
            f"{history.density_change[-1]:.3e} against {settings.density_tolerance:g}, velocity change "
            f"{history.velocity_change[-1]:.3e} against {convergence.velocity_tolerance(settings):g}, eps_K "
            f"{history.kutta_error[-1]:.3e} against {settings.kutta_tolerance:g}; last exit angle "
            f"{math.degrees(previous.exit_angle):.6f} deg)",
            _solution(current_grid, context, previous, history, records, False, limit))

    return _observed(base.observers, _info("coupled", settings, limit, grid, base, settings.kutta_tolerance), loop)


# --- observers (monitor.py): observation only ---------------------------------------------------------------

class _Observers:
    """The observers of one solve. A call that raises switches that observer off with a
    MonitorWarning; nothing an observer returns or does reaches the solve."""

    def __init__(self, items) -> None:
        self.items = list(items)

    def wants(self, method: str) -> bool:
        """True when an observer defines the optional method (inspect)."""
        return any(callable(getattr(item, method, None)) for item in self.items)

    def notify(self, method: str, *args, optional: bool = False) -> None:
        for item in list(self.items):
            if optional and not callable(getattr(item, method, None)):
                continue
            try:
                getattr(item, method)(*args)
            except Exception as error:      # an observer never stops or changes the solve
                self.items.remove(item)
                warnings.warn(f"the observer {type(item).__name__} raised in {method} and is switched off: "
                              f"{type(error).__name__}: {error}", MonitorWarning, stacklevel=3)


def _observers(live_monitor: bool, verbose: bool, monitor) -> _Observers:
    for name, value in (("live_monitor", live_monitor), ("verbose", verbose)):
        if not isinstance(value, bool):
            raise SolverInputError(f"{name} = {value!r}: True or False")
    items = [] if monitor is None else [monitor]
    if live_monitor:
        from potential_flow_solver.monitor import LiveConvergenceMonitor
        items.append(LiveConvergenceMonitor())
    if verbose:
        from potential_flow_solver.monitor import ConsoleProgress
        items.append(ConsoleProgress())
    return _Observers(items)


def _info(mode: str, settings, limit: int, grid: Grid, context: _Context, kutta_tolerance) -> SolveInfo:
    return SolveInfo(mode=mode, density_tolerance=float(settings.density_tolerance),
                     velocity_tolerance=convergence.velocity_tolerance(settings),
                     kutta_tolerance=None if kutta_tolerance is None else float(kutta_tolerance), max_passes=limit,
                     exit_angle_start=float(grid.exit_angle), inlet_angle=float(grid.inlet_angle),
                     mass_flow_per_depth=context.mass_flow_per_depth, mass_flow_source=context.mass_flow_source,
                     n_m=len(grid.m), n_eta=len(grid.eta))


def _observed(observers: _Observers, info: SolveInfo, loop) -> Solution:
    """start, the loop, finish (with the solution, or the error and its partial solution)."""
    observers.notify("start", info)
    try:
        solution = loop()
    except DensityIterationError as error:
        observers.notify("finish", error.solution, error)
        raise
    except BaseException as error:
        observers.notify("finish", getattr(error, "solution", None), error)
        raise
    observers.notify("finish", solution, None)
    return solution


# --- one pass -------------------------------------------------------------------------------------------

def _first_grid(cascade: Cascade, config: SolverConfig) -> Grid:
    if not isinstance(config, SolverConfig):
        raise TypeError(f"the solver takes a SolverConfig, got {type(config).__name__}")
    if cascade.is_dimensional:
        raise SolverInputError("the cascade has a physical chord: build it with physical_chord=None and give the chord "
                               "as c_ref (coefficients.py applies the length scale once)")
    return make_grid(cascade, config.grid, math.radians(config.flow.inlet_angle_deg),
                     math.radians(config.flow.exit_angle_guess_deg), require_ss_ps=True)


def _limit(value, name: str) -> int:
    limit = int(value)
    if limit < 1:
        raise SolverInputError(f"{name} = {value!r}: at least one pass is needed")
    return limit


def _start_density(grid: Grid, context: _Context, density) -> NDArray[np.float64]:
    shape = (len(grid.m), len(grid.eta))
    if density is None:
        return np.array(eos.initial_density(context.gas, context.reference, shape=shape))
    rho = np.array(density, dtype=float)
    if rho.shape != shape:
        raise SolverInputError(f"the start density has shape {rho.shape}, the grid {shape}")
    return rho


def _context(grid: Grid, config: SolverConfig, mass_flow_per_depth: float, c_ref: float) -> _Context:
    if not isinstance(config, SolverConfig):
        raise TypeError(f"the solver takes a SolverConfig, got {type(config).__name__}")
    if not isinstance(grid, Grid):
        raise TypeError(f"the solver takes a Grid, got {type(grid).__name__}")
    if isinstance(c_ref, bool) or not isinstance(c_ref, (int, float)) or not (math.isfinite(c_ref) and c_ref > 0.0):
        raise SolverInputError(f"c_ref = {c_ref!r}: the coupled solve needs the true chord as a positive length "
                               f"(metres in SI); the normalized mode of the lower modules is no physical solution")
    conditions = config.flow
    if conditions.angular_speed != 0.0:
        raise SolverInputError("a rotating row (angular_speed != 0) is not validated in the density-coupled solver yet: "
                               "the first scope is a stationary cascade")
    total = eos.TotalState(pressure=conditions.inlet_absolute_total_pressure,
                           temperature=conditions.inlet_absolute_total_temperature)
    reference = eos.blade_row_reference(total, config.gas, omega=conditions.angular_speed,
                                        radius=conditions.stream_surface_radius,
                                        inlet_c_theta=conditions.inlet_absolute_tangential_velocity)
    inlet, mdot, source = _mass_flow(grid, config, mass_flow_per_depth, float(c_ref))
    return _Context(metrics=build_metrics(grid), boundary_conditions=build_boundary_conditions(grid), gas=config.gas,
                    reference=reference, mass_flow_per_depth=mdot, c_ref=float(c_ref),
                    relaxation=config.iteration.density_relaxation,
                    blade=slice(grid.first_blade_i, grid.last_blade_i + 1), inlet=inlet, mass_flow_source=source,
                    stop_at_sonic=bool(config.iteration.stop_at_sonic))


def _mass_flow(grid: Grid, config: SolverConfig, given, c_ref: float) -> tuple[InletState | None, float, str]:
    """(inlet state, mdot', where mdot' came from): module docstring, "The mass flow"."""
    inlet = None
    if config.flow.inlet_mach is not None:
        try:
            inlet = inlet_state(config.flow, config.gas, inlet_pitch=c_ref * grid.pitch)
        except InletStateError as error:
            raise SolverInputError(f"the inlet state cannot be derived: {error}") from error
    if given is None:
        if inlet is None:
            raise SolverInputError("no mass flow: give config.flow.inlet_mach (normal use) or mass_flow_per_depth "
                                   "(verification override)")
        if not inlet.mass_flow_per_depth > 0.0:
            raise SolverInputError(f"the inlet Mach number {inlet.mach!r} gives mdot' = {inlet.mass_flow_per_depth!r}: the "
                                   f"solver refuses a zero flow (coefficients.py needs a positive mass flow)")
        return inlet, inlet.mass_flow_per_depth, MASS_FLOW_FROM_INLET
    mdot = float(given)
    if inlet is None:
        return None, mdot, MASS_FLOW_OVERRIDE
    derived = inlet.mass_flow_per_depth
    if not abs(mdot - derived) <= MASS_FLOW_CONSISTENCY * abs(derived):
        raise SolverInputError(f"mass_flow_per_depth = {mdot!r} and the inlet Mach number {inlet.mach!r} (mdot' = "
                               f"{derived!r}) disagree by more than {MASS_FLOW_CONSISTENCY:g} relative: give one of them, "
                               f"or inlet_mach None for a pure override")
    return inlet, mdot, MASS_FLOW_CHECKED


def _at_angle(start: Grid, angle: float, base: _Context) -> tuple[Grid, _Context]:
    """Step 1 and 2 of the coupled pass: the grid at angle (start itself at its own angle) with
    its own metrics and boundary relations; ExitRebuildError if anything but the exit
    stations' nodes changed or the grid is no longer admissible."""
    grid = start if angle == start.exit_angle else rebuild_exit(start, angle)
    if grid is not start:
        kept = slice(0, start.last_blade_i + 1)
        if not (np.array_equal(grid.m, start.m) and np.array_equal(grid.eta, start.eta)
                and (grid.first_blade_i, grid.last_blade_i) == (start.first_blade_i, start.last_blade_i)
                and np.array_equal(grid.nodes[kept], start.nodes[kept])):
            raise ExitRebuildError(f"the exit rebuild at {math.degrees(angle):.6f} deg changed the stations, eta, the blade "
                                   f"run or an inlet or blade node: the Kutta loop may move the exit nodes only")
        try:
            validate_uniform_grid_topology(grid)
        except ValueError as error:
            raise ExitRebuildError(f"the exit rebuild at {math.degrees(angle):.6f} deg left an inadmissible grid: "
                                   f"{error}") from error
    return grid, replace(base, metrics=build_metrics(grid), boundary_conditions=build_boundary_conditions(grid))


def _density_pass(rho: NDArray[np.float64], context: _Context) -> _Pass:
    """Steps 3 to 7 with the resident density rho (module docstring): calls only."""
    co = build_coefficients(context.metrics, rho, context.mass_flow_per_depth, c_ref=context.c_ref)
    assembly = assemble_from_coefficients(co, context.metrics, context.boundary_conditions)
    linear = solve_linear_system(assembly)
    flow = velocity_components(linear.psi, rho, co, context.metrics, context.boundary_conditions)
    state = eos.static_state_from_velocity(flow.W, context.gas, context.reference)
    mach = np.asarray(flow.W) / np.asarray(state.sound_speed)
    target = np.asarray(state.density)
    rho_next = eos.relax_density(rho, target, relaxation=context.relaxation)
    mass = station_mass_flow(flow, rho, co, context.metrics)
    return _Pass(density_resident=rho, coefficients=co, linear=linear, flow=flow, state=state, mach=mach,
                 density_target=target, density_next=rho_next,
                 mass_flow_error=np.abs(mass / context.mass_flow_per_depth - 1.0))


def _kutta_step(current: _Pass, context: _Context) -> _Pass:
    """Step 8 of the coupled pass (calls only): the Kutta residual of the pass's W and, in the
    coupled loop, the corrected next angle. At a fixed angle (kutta_relaxation None) the
    residual is a diagnostic only, left out (None) when that grid's TE station is not one SS
    and one PS node, and the angle stays."""
    grid = context.metrics.grid
    angle = grid.exit_angle
    if context.kutta_relaxation is None:
        try:
            residual = kutta.kutta_residual(current.flow, grid)
        except kutta.KuttaError:
            residual = None
        return replace(current, kutta=residual, exit_angle=angle, exit_angle_next=angle)
    residual = kutta.kutta_residual(current.flow, grid)
    correction = kutta.exit_angle_correction(residual, current.flow, context.metrics)
    return replace(current, kutta=residual, correction=correction, exit_angle=angle,
                   exit_angle_next=kutta.next_exit_angle(angle, correction, context.kutta_relaxation))


def _checked_pass(rho, context: _Context, grid: Grid, history: History, records: list, previous: _Pass | None,
                  n: int, final: bool) -> _Pass:
    """One pass, recorded; a relative Mach number of 1 or more stops the loop, or, with the
    sonic stop off, warns once and goes on (module docstring)."""
    current = _kutta_step(_density_pass(rho, context), context)
    _record(current, previous, n, final, context, history, records)
    peak = float(np.max(current.mach))
    if not peak < 1.0:
        i, j = np.unravel_index(int(np.argmax(current.mach)), current.mach.shape)
        if not context.stop_at_sonic:
            if not any(r.mach_max >= 1.0 for r in records[:-1]):
                warnings.warn(f"pass {n}: the relative Mach number reaches {peak:.4f} at node ({i}, {j}), m = "
                              f"{grid.m[i]:.4g}; the sonic stop is off, so the subsonic iteration goes on outside its "
                              f"validity", SonicFlowWarning, stacklevel=2)
            return current
        raise TransonicFlowError(f"pass {n}: the relative Mach number reaches {peak:.4f} at node ({i}, {j}), m = "
                                 f"{grid.m[i]:.4g}: the subsonic density iteration does not apply and no transonic "
                                 f"extension is implemented", _solution(grid, context, current, history, records, False, n))
    return current


def _record(current: _Pass, previous: _Pass | None, n: int, final: bool, context: _Context, history: History,
            records: list) -> None:
    """The PassRecord of the pass, then History's entries from it (one each), then the
    observers."""
    d_rho = convergence.change_norm(current.density_next, current.density_resident)
    if previous is None:
        d_w = d_psi = float("nan")
    else:
        d_w = convergence.velocity_change(current.flow.W_m, current.flow.W_theta, previous.flow.W_m, previous.flow.W_theta)
        d_psi = convergence.change_norm(current.linear.psi, previous.linear.psi)
    residual = current.kutta
    record = PassRecord(
        index=n, final=final, density_change=d_rho, velocity_change=d_w, psi_change=d_psi,
        target_mismatch=convergence.change_norm(current.density_target, current.density_resident),
        density_min=float(np.min(current.density_resident)), density_max=float(np.max(current.density_resident)),
        speed_min=float(np.min(current.flow.W)), speed_max=float(np.max(current.flow.W)),
        mach_max=float(np.max(current.mach)), linear_residual=current.linear.residual_inf,
        mass_flow_error_max=float(np.max(current.mass_flow_error)),
        mass_flow_error_blade_median=float(np.median(current.mass_flow_error[context.blade])),
        exit_angle=float(current.exit_angle),
        kutta_residual=float(residual.residual) if residual is not None else float("nan"),
        kutta_normalized=float(residual.normalized) if residual is not None else float("nan"),
        exit_angle_next=float(current.exit_angle_next),
        tan_correction=float(current.correction.delta_tan) if current.correction is not None else float("nan"))
    records.append(record)
    coupled = context.kutta_relaxation is not None
    history.kutta_pass.append(record.index if coupled else 0)
    history.density_change.append(record.density_change)
    history.velocity_change.append(record.velocity_change)
    history.psi_change.append(record.psi_change)
    history.residual.append(record.linear_residual)
    if coupled:
        history.exit_angle.append(record.exit_angle)
        history.kutta_error.append(record.kutta_normalized)
    context.observers.notify("update", record)
    if context.observers.wants("inspect"):
        context.observers.notify("inspect", PassFields(
            record=record, grid=context.metrics.grid, metrics=context.metrics,
            density_resident=_read_only(current.density_resident), density_target=_read_only(current.density_target),
            density_next=_read_only(current.density_next), coefficients=current.coefficients,
            psi=_read_only(current.linear.psi), flow=current.flow, static_state=current.state,
            mach=_read_only(current.mach), mass_flow_per_depth=context.mass_flow_per_depth, c_ref=context.c_ref),
            optional=True)


def _read_only(array) -> NDArray[np.float64]:
    view = np.asarray(array).view()
    view.flags.writeable = False
    return view


def _attach(error: Exception, last, previous: _Pass | None, history: History, records: list, failed_pass: int) -> None:
    """Give a called module's error the last completed pass (.solution) and the failed pass's
    index (.failed_pass), keeping its type (module docstring)."""
    try:
        if getattr(error, "solution", None) is None:
            error.solution = None if previous is None or last is None else _solution(
                last[0], last[1], previous, history, records, False, failed_pass)
        error.failed_pass = failed_pass
    except Exception:                           # an exception object that refuses attributes stays as it is
        pass


def _solution(grid: Grid, context: _Context, current: _Pass, history: History, records: list, converged: bool,
              iterations: int) -> Solution:
    residual = current.kutta
    coupled = context.kutta_relaxation is not None
    limitation = COUPLED_LIMITATION if coupled else LIMITATION
    if not context.stop_at_sonic:
        peak = max((r.mach_max for r in records), default=float(np.max(current.mach)))
        if peak >= 1.0:
            limitation += (f"; SONIC STOP OFF: M_rel >= 1 REACHED (peak {peak:.4f}), OUTSIDE THE SUBSONIC METHOD")
    return Solution(grid=grid, metrics=context.metrics, psi=current.linear.psi, flow=current.flow, history=history,
                    density=current.density_resident, density_target=current.density_target, static_state=current.state,
                    mach=current.mach, coefficients=current.coefficients, linear=current.linear, passes=tuple(records),
                    converged=converged, iterations=iterations, mass_flow_per_depth=context.mass_flow_per_depth,
                    c_ref=context.c_ref, density_relaxation=context.relaxation, exit_angle=float(grid.exit_angle),
                    kutta=residual,
                    kutta_residual=float(residual.residual) if residual is not None else float("nan"),
                    normalized_kutta_residual=float(residual.normalized) if residual is not None else float("nan"),
                    kutta_relaxation=context.kutta_relaxation,
                    limitation=limitation, inlet=context.inlet,
                    mass_flow_source=context.mass_flow_source)
