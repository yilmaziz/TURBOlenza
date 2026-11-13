"""
run_full_solver.py
==================
The user-facing controller of the blade-to-blade potential-flow solver: edit the values in
the sections below, then run

    python scripts/run_full_solver.py

It builds the case, prints every effective input and the derived inlet state and grid, runs
the solver with a live convergence window, prints the result, shows the surface Mach,
surface pressure and convergence figures and saves them with CSV and JSON files. No source
file needs editing for a normal case change.

Everything numerical stays in the package: this file only states the inputs and calls
turbolenza.b2b.controller (build_case, run_case, format_run_report, plot_run,
export_run), the same functions a later GUI will call with its own text boxes in place of
these constants.

The default case is the NACA 65-(12A10)10 cascade of Dunavant et al. (1955) at the
condition of Aungier (2003) Fig. 5-6. Note: at M1 = 0.65 the subsonic solver currently stops
at pass 9 with a relative Mach number of 1.003 at the first (leading-edge) blade station
(TransonicFlowError; the README's open decisions). That is a stop of the method, not an
iteration ceiling; the script reports it, plots the last pass labeled NOT CONVERGED and exits
with status 2. INLET_MACH = 0.30 or 0.60 converges.
"""

# ============================================================
# 0. RUN CONTROL
# ============================================================

CASE_NAME = "naca65_12a10_10_M065"      # names the output folder outputs/<CASE_NAME>/

SOLVER_MODE = "coupled_kutta"
# options:
#   "coupled_kutta"   density + Kutta coupled solve: the full solver (solver.solve)
#   "fixed_exit"      density loop at a fixed discharge angle, no Kutta condition
#                     (solver.solve_density_coupled)

FIXED_EXIT_ANGLE_DEG = None             # fixed_exit only: the discharge angle [deg];
                                        # None = the blade TE angle kappa_te

SHOW_LIVE_MONITOR = True                # one live convergence window for the solve
VERBOSE_SOLVER = True                   # one console line per pass

SHOW_FINAL_PLOTS = True                 # show the figures at the end (blocks until closed)
SAVE_RESULTS = True                     # write outputs/<CASE_NAME>/<RUN_LABEL>/

# ============================================================
# 1. GAS / THERMODYNAMIC INPUTS
# ============================================================

GAMMA = 1.4                             # ratio of specific heats (air)
GAS_CONSTANT = 287.05                   # J/(kg K) (air)

INLET_TOTAL_PRESSURE = 101325.0         # Pa, absolute-frame inlet total pressure
INLET_TOTAL_TEMPERATURE = 288.15        # K, absolute-frame inlet total temperature

OMEGA = 0.0                             # rad/s; only 0 (a stationary cascade) is solved now:
                                        # a rotating row is refused by the solver
STREAM_SURFACE_RADIUS = None            # m; rotating rows only, unused for OMEGA = 0
INLET_ABSOLUTE_TANGENTIAL_VELOCITY = None   # m/s; rotating rows only, unused for OMEGA = 0

# ============================================================
# 2. BLADE / CASCADE GEOMETRY
# ============================================================
# The inputs of geometry.CascadeSpec. The blade name, the pitch and the angle of attack are
# derived (printed below), never entered.

BLADE_FAMILY = "NACA65"                 # CascadeSpec.blade_type (NACA 65 implemented)
DESIGN_LIFT_COEFFICIENT = 1.2           # cl0: the "12" of 65-(12A10)10
THICKNESS_RATIO = 0.10                  # t/c: the "10" of 65-(12A10)10 (nominal)
MEANLINE_TYPE = "A10"                   # a = 1.0 mean line (project default)
PROFILE_VARIANT = "kovach_sandercock"   # thickness distribution (project default)
SURFACE_CONTINUITY = "C2"               # "C2": the solver's geometry
N_CAMBER_STATIONS = 101                 # camberline resolution (project default)

CHORD = 0.1                             # m, the true chord (the solver's c_ref)
SOLIDITY = 1.0                          # chord / pitch
STAGGER_DEG = 33.0                      # deg from the axial direction
# PITCH = CHORD / SOLIDITY                (derived)

# ============================================================
# 3. INLET FLOW CONDITIONS
# ============================================================

INLET_MACH = 0.65                       # far-inlet Mach number (stationary: relative = absolute)
INLET_FLOW_ANGLE_DEG = 45.0             # deg from the meridional (axial) direction, positive toward +y

MASS_FLOW_PER_DEPTH_OVERRIDE = None     # kg/(m s) per passage and unit depth
# None   -> mdot' is derived from Pt, Tt, M1, beta1 and the pitch (normal use)
# number -> verification override; with INLET_MACH given it must agree with it to 1e-12
#           (the production policy), so a pure override needs INLET_MACH = None

# ============================================================
# 4. GRID SETTINGS
# ============================================================

GRID_LEVEL = "medium"                       # None: the explicit N_PASSAGE x N_ETA grid below. Or a
                                        # geometry-driven standard grid (grid.py section 9):
                                        # "very_coarse", "coarse", "medium", "fine", "very_fine"
                                        # (then set N_PASSAGE = N_ETA = None; the extensions
                                        # below become nominal, actual ones are whole cells)
N_PASSAGE = None                          # target cells between LE and TE; must be admissible
                                        # (e.g. 43, 44, 45, 46, 49, 50, 51, ...; an inadmissible
                                        # value stops the run before the solve and names the nearest)
N_ETA = None                                  # nodes across the passage
UPSTREAM_EXTENSION_CHORDS = 1.0         # inlet extension [chords] (project default)
DOWNSTREAM_EXTENSION_CHORDS = 1.0       # exit extension [chords] (project default)

# ============================================================
# 5. NONLINEAR / DENSITY SETTINGS
# ============================================================

DENSITY_TOLERANCE = 1e-8                # relative density change per pass (validation script value)
VELOCITY_TOLERANCE = None               # relative velocity change; None = DENSITY_TOLERANCE (project rule)
MAX_DENSITY_ITERATIONS = 600            # pass ceiling (validation script value)

DENSITY_RELAXATION = 0.7                # lambda_rho: 0.5 is OUR project choice (under-relaxation);
                                        # 1.0 = Aungier's direct density replacement

STOP_AT_SONIC = True                   # True (project default): stop at the first pass with M_rel >= 1
                                        # (TransonicFlowError). False: go on past sonic to watch the
                                        # iteration: the subsonic method does not apply there; a run
                                        # that runs away ends in an error of the gas model (T <= 0) and
                                        # its last completed pass is still reported and plotted

# ============================================================
# 6. KUTTA SETTINGS
# ============================================================

KUTTA_TOLERANCE = 1e-8                  # eps_K = |W_SS - W_PS| / mean TE speed (validation script value)
MAX_KUTTA_ITERATIONS = 600              # exit-angle corrections; the coupled solve runs at most
                                        # min(MAX_DENSITY_ITERATIONS, MAX_KUTTA_ITERATIONS) passes
KUTTA_RELAXATION = 0.7                  # lambda_K: 1.0 is the project baseline, the undamped Aungier
                                        # correction (Aungier recommends damping, gives no value)

INITIAL_EXIT_ANGLE_MODE = "blade_te_angle"
# options:
#   "blade_te_angle"  start the Kutta iteration at the blade TE angle kappa_te
#   "manual"          start at INITIAL_EXIT_ANGLE_DEG
INITIAL_EXIT_ANGLE_DEG = None           # deg, "manual" only (e.g. 20.0)

# ============================================================
# 7. COMPUTE / RUNTIME SETTINGS
# ============================================================

MEASURE_RUNTIME = True                  # wall time, process CPU time and effective CPU cores of the
                                        # solver call (two clock reads; shown on every surface Mach plot)
# No thread-limit setting: the solve uses about one core (measured 0.97 to 0.99 effective
# cores), and limiting the numerical libraries' threads changes the result in its last bits
# (controller.py docstring), so this script sets none.

# ============================================================
# 8. PLOTTING / OUTPUT SETTINGS
# ============================================================

OUTPUT_ROOT = "outputs"                 # relative to the repository root
RUN_LABEL = None                        # subfolder name; None = the start time. An existing folder
                                        # is never overwritten (-2, -3, ... is appended)
PLOT_DPI = 130
SHOW_SONIC_LINE = True                  # dashed M = 1 line on the surface Mach plot

# ============================================================
# (end of the editable inputs)
# ============================================================

import sys                                                                      # noqa: E402
from pathlib import Path                                                        # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def settings_from_constants():
    """The constants above as one CaseSettings (the object a GUI would fill)."""
    from turbolenza.b2b.controller import CaseSettings

    return CaseSettings(
        case_name=CASE_NAME, solver_mode=SOLVER_MODE, fixed_exit_angle_deg=FIXED_EXIT_ANGLE_DEG,
        gamma=GAMMA, gas_constant=GAS_CONSTANT, inlet_total_pressure=INLET_TOTAL_PRESSURE,
        inlet_total_temperature=INLET_TOTAL_TEMPERATURE, omega=OMEGA, stream_surface_radius=STREAM_SURFACE_RADIUS,
        inlet_absolute_tangential_velocity=INLET_ABSOLUTE_TANGENTIAL_VELOCITY,
        blade_family=BLADE_FAMILY, design_lift_coefficient=DESIGN_LIFT_COEFFICIENT, thickness_ratio=THICKNESS_RATIO,
        meanline_type=MEANLINE_TYPE, profile_variant=PROFILE_VARIANT, surface_continuity=SURFACE_CONTINUITY,
        n_camber_stations=N_CAMBER_STATIONS, chord=CHORD, solidity=SOLIDITY, stagger_deg=STAGGER_DEG,
        inlet_mach=INLET_MACH, inlet_flow_angle_deg=INLET_FLOW_ANGLE_DEG,
        mass_flow_per_depth_override=MASS_FLOW_PER_DEPTH_OVERRIDE,
        grid_level=GRID_LEVEL, n_passage=N_PASSAGE, n_eta=N_ETA, upstream_extension_chords=UPSTREAM_EXTENSION_CHORDS,
        downstream_extension_chords=DOWNSTREAM_EXTENSION_CHORDS,
        density_tolerance=DENSITY_TOLERANCE, velocity_tolerance=VELOCITY_TOLERANCE,
        max_density_iterations=MAX_DENSITY_ITERATIONS, density_relaxation=DENSITY_RELAXATION,
        stop_at_sonic=STOP_AT_SONIC,
        kutta_tolerance=KUTTA_TOLERANCE, max_kutta_iterations=MAX_KUTTA_ITERATIONS, kutta_relaxation=KUTTA_RELAXATION,
        initial_exit_angle_mode=INITIAL_EXIT_ANGLE_MODE, initial_exit_angle_deg=INITIAL_EXIT_ANGLE_DEG,
        measure_runtime=MEASURE_RUNTIME)


def main() -> int:
    """Build, echo, run, report, plot, save. Returns 0 after a converged solve, 1 when the case
    cannot be built, 2 when the solve stopped (its last state is still reported and saved)."""
    if not (SHOW_LIVE_MONITOR or SHOW_FINAL_PLOTS):
        import matplotlib
        matplotlib.use("Agg")                   # nothing is shown: draw without any window
    from turbolenza.b2b import controller

    # BUILD CASE
    try:
        case = controller.build_case(settings_from_constants())
    except (controller.CaseError, ValueError) as error:
        print(f"THE CASE CANNOT BE BUILT: {type(error).__name__}: {error}")
        return 1
    # PRINT CASE SUMMARY
    print(controller.format_case_summary(case))
    print(controller.format_inlet_state(case))
    print(controller.format_grid_summary(case))
    # RUN SOLVER (with the live convergence window)
    run = controller.run_case(case, live_monitor=SHOW_LIVE_MONITOR, verbose=VERBOSE_SOLVER)
    # FINAL REPORT
    print()
    print(controller.format_run_report(run))
    # FIGURES: surface Mach (runtime footer), surface pressure, convergence
    figures = controller.plot_run(run, sonic_line=SHOW_SONIC_LINE)
    if not figures:
        print("no solution to plot (the solve failed before its first pass was recorded)")
    # FILES
    if SAVE_RESULTS:
        directory = controller.export_run(run, figures, ROOT / OUTPUT_ROOT, RUN_LABEL, dpi=PLOT_DPI)
        print(f"results written to {directory}")
    if SHOW_FINAL_PLOTS and (figures or run.monitor is not None):
        import matplotlib.pyplot as plt
        plt.ioff()
        plt.show()
    return 0 if run.converged else 2


if __name__ == "__main__":
    sys.exit(main())
