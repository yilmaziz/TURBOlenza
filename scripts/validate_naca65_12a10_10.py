"""
validate_naca65_12a10_10.py
===========================
Validation runner for the NACA 65-(12A10)10 cascade of Dunavant et al. (1955), the case of
Aungier (2003) Fig. 5-6: the density + Kutta coupled solver (turbolenza.b2b.solver.solve)
at the documented test condition, the blade-surface Mach number and pressure coefficient
against x/c, the discharge angle and the convergence summary.

This script RUNS the validation case; it does not validate anything by itself. Validation is
the comparison with the published measurements, and none is digitized in this repository
(see EXPERIMENT below): until a measured data file is given with --experiment, the output is
the prediction alone, and the external reference angles are printed for orientation only.
Nothing here tunes the solver; no case constant lives in the solver package.

Case inputs and their sources (CASE):
    blade        NACA 65-(12A10)10: Dunavant et al. 1955 (title, Fig. 10 caption), Aungier
                 2003 Fig. 5-6; built by geometry.py (A10 mean line, its default
                 kovach_sandercock thickness, C2 surfaces, the solver's geometry)
    solidity     1.0: Dunavant Fig. 10 caption (sigma = 1.0); Fig. 5-6 does not print it
    beta_1       45 deg: Fig. 5-6 (beta_in) and Dunavant Fig. 10 (beta_1), from the axial
                 direction ("perpendicular to blade row", Dunavant's list of symbols)
    alpha        12 deg: Fig. 5-6 and Dunavant Fig. 10 (alpha = 12.0 deg), the angle between
                 the entering flow and the chord (Dunavant's list of symbols)
    stagger      33 deg = beta_1 - alpha: derived from the two definitions above
    M_1          0.65: Fig. 5-6 (M_in); Dunavant Fig. 10(b): M_1 = 0.650, theta = 19.9 deg
    gas          air, gamma 1.4 (the standard value for the air of the tests; the model
                 choice of this script), R 287.05 J/(kg K)
    Pt, Tt, c    101 325 Pa, 288.15 K, 0.1 m: a numerical dimensionalization, not test data.
                 The inviscid perfect-gas solution depends on M_1, gamma and the geometry
                 only; these set the units (and the reported p, T, rho).
    Derived, not specified: the stagger (above) and the mass flow per passage and unit depth
    (flow.inlet_state from Pt, Tt, M_1, beta_1 and the pitch).

External references (REFERENCES, never used by the solver): experimental beta_out 25.0 deg
(Aungier 2003 p. 92 and the Fig. 5-6 legend; Dunavant Fig. 10(b) gives theta = 19.9 deg,
i.e. 45 - 19.9 = 25.1 deg); Aungier's potential-flow beta_out 24.4 deg (p. 92, Fig. 5-6).

EXPERIMENT: Dunavant et al. 1955 (NACA RM L55I08), Fig. 10(b), report p. 32 (PDF p. 33):
S = (P_1 - p_l) / q_1 against percent chord, upper and lower surface, at M_1 = 0.650; Aungier
converted these S values to Mach numbers for Fig. 5-6 (p. 91). Not digitized here. A
digitized file plugs in with --experiment PATH: a CSV with the columns surface (suction /
pressure), x_c and mach (the layout of naca65/data/cases/fig56_exp.csv, which holds the
header only), or --experiment-quantity S for Dunavant's own coefficient.

Usage
    python scripts/validate_naca65_12a10_10.py                 live monitor, console lines, plots shown
    python scripts/validate_naca65_12a10_10.py --no-monitor --no-show --save outputs/validation_naca65_12a10_10
Exit status 0 when the coupled solve converged, 2 when it stopped (a transonic stop, the pass
limit): its last pass is still summarized and plotted, labeled NOT CONVERGED.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from turbolenza.b2b import config as cf                                  # noqa: E402
from turbolenza.b2b import eos, postprocess                              # noqa: E402
from turbolenza.b2b import solver as sv                                  # noqa: E402
from turbolenza.b2b.geometry import CascadeSpec, make_cascade            # noqa: E402

CASE = {
    "blade": ("NACA 65-(12A10)10 (cl0 1.2, t/c 0.10, A10 mean line)",
              "Dunavant et al. 1955 title and Fig. 10; Aungier 2003 Fig. 5-6"),
    "solidity": (1.0, "Dunavant et al. 1955 Fig. 10 caption (sigma = 1.0)"),
    "inlet_angle_deg": (45.0, "Aungier 2003 Fig. 5-6 (beta_in); Dunavant Fig. 10 (beta_1)"),
    "angle_of_attack_deg": (12.0, "Aungier 2003 Fig. 5-6; Dunavant Fig. 10 (alpha = 12.0 deg)"),
    "inlet_mach": (0.65, "Aungier 2003 Fig. 5-6 (M_in); Dunavant Fig. 10(b) (M_1 = 0.650)"),
    "gamma": (1.4, "air: the standard value (model choice of this script)"),
    "gas_constant": (287.05, "air, J/(kg K): numerical dimensionalization"),
    "total_pressure": (101325.0, "Pa: numerical dimensionalization (not test data)"),
    "total_temperature": (288.15, "K: numerical dimensionalization (not test data)"),
    "chord": (0.1, "m: numerical dimensionalization (not test data)"),
}
REFERENCES = {
    "beta_out_experiment_deg": (25.0, "Aungier 2003 p. 92 and Fig. 5-6 legend"),
    "beta_out_from_dunavant_theta_deg": (45.0 - 19.9, "Dunavant Fig. 10(b): beta_1 - theta = 45 - 19.9"),
    "beta_out_aungier_prediction_deg": (24.4, "Aungier 2003 p. 92 and Fig. 5-6 legend"),
}


def stagger_deg() -> float:
    """beta_1 - alpha (Dunavant's definitions of both angles)."""
    return CASE["inlet_angle_deg"][0] - CASE["angle_of_attack_deg"][0]


def build(args) -> tuple:
    cascade = make_cascade(CascadeSpec(blade_type="NACA65", solidity=CASE["solidity"][0], stagger_deg=stagger_deg(),
                                       cl0=1.2, t_max_over_c=0.10, surface_continuity="C2"))
    gas = eos.CaloricallyPerfectGas(gamma=CASE["gamma"][0], gas_constant=CASE["gas_constant"][0])
    config = cf.SolverConfig(
        grid=cf.GridSettings(n_passage=args.n_passage, n_eta=args.n_eta),
        flow=cf.FlowConditions(inlet_mach=CASE["inlet_mach"][0], inlet_angle_deg=CASE["inlet_angle_deg"][0],
                               exit_angle_guess_deg=math.degrees(cascade.kappa_te),
                               inlet_absolute_total_pressure=CASE["total_pressure"][0],
                               inlet_absolute_total_temperature=CASE["total_temperature"][0], angular_speed=0.0),
        gas=gas,
        iteration=cf.IterationSettings(density_tolerance=args.tolerance, max_density_iterations=args.max_passes,
                                       kutta_tolerance=args.kutta_tolerance, max_kutta_iterations=args.max_passes))
    return cascade, config


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-passage", type=int, default=43, help="admissible n_passage (numerical choice; default 43)")
    parser.add_argument("--n-eta", type=int, default=21, help="nodes across the passage (default 21)")
    parser.add_argument("--tolerance", type=float, default=1e-8, help="density (and velocity) tolerance (default 1e-8)")
    parser.add_argument("--kutta-tolerance", type=float, default=1e-8, help="eps_K tolerance (default 1e-8)")
    parser.add_argument("--max-passes", type=int, default=600, help="coupled pass limit (default 600)")
    parser.add_argument("--no-monitor", action="store_true", help="no live convergence window")
    parser.add_argument("--quiet", action="store_true", help="no per-pass console line")
    parser.add_argument("--no-show", action="store_true", help="do not show the figures at the end")
    parser.add_argument("--save", type=Path, default=None, help="directory for surface_mach.png, surface_cp.png, convergence.png")
    parser.add_argument("--experiment", type=Path, default=None, help="CSV of measured points (surface, x_c, quantity)")
    parser.add_argument("--experiment-quantity", default="mach", choices=("mach", "S"),
                        help="the measured column: mach (as Fig. 5-6) or S (Dunavant's coefficient)")
    args = parser.parse_args(argv)

    if args.no_show:                    # batch run: draw without any window
        import matplotlib
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from turbolenza.b2b import plots
    from turbolenza.b2b.monitor import LiveConvergenceMonitor

    cascade, config = build(args)
    print("NACA 65-(12A10)10 cascade, Dunavant et al. (1955) / Aungier (2003) Fig. 5-6")
    for name, (value, source) in CASE.items():
        print(f"  {name:22s} {value!s:>46s}   [{source}]")
    print(f"  {'stagger_deg':22s} {stagger_deg()!s:>46s}   [derived: beta_1 - alpha]")

    monitor = None if args.no_monitor else LiveConvergenceMonitor(title="NACA 65-(12A10)10, Dunavant / Aungier Fig. 5-6")
    failure = None
    try:
        solution = sv.solve(cascade, config, c_ref=CASE["chord"][0], monitor=monitor, verbose=not args.quiet)
    except sv.DensityIterationError as error:
        failure, solution = error, error.solution
    inlet = solution.inlet
    print(f"\nderived inlet state: T1 {inlet.static_temperature:.4f} K, p1 {inlet.static_pressure:.2f} Pa, rho1 "
          f"{inlet.static_density:.6f} kg/m^3, a1 {inlet.sound_speed:.4f} m/s, W1 {inlet.velocity:.4f} m/s, "
          f"W_m1 {inlet.meridional_velocity:.4f} m/s, mdot' {inlet.mass_flow_per_depth:.6f} kg/(m s) "
          f"(pitch {inlet.pitch:.6g} m)")
    if failure is not None:
        print(f"\nTHE COUPLED SOLVE DID NOT CONVERGE: {type(failure).__name__}: {failure}")
        print("The figures and the summary below are of its LAST PASS, not a solution.")
    print("\n" + postprocess.format_summary(postprocess.summarize(solution)))
    print("\nexternal references (for orientation only; never used by the solver):")
    for name, (value, source) in REFERENCES.items():
        print(f"  {name:34s} {value:7.2f} deg   [{source}]")
    print(f"  predicted beta_out               {postprocess.discharge_angle_deg(solution):7.3f} deg"
          f"{'' if failure is None else '   (last pass, NOT CONVERGED)'}")

    experiment = None
    if args.experiment is not None:
        experiment = postprocess.read_surface_points(args.experiment, args.experiment_quantity)
    if experiment is None or experiment.empty:
        print("\nno measured surface data given: the quantitative Dunavant comparison is pending digitization of "
              "Dunavant et al. 1955 Fig. 10(b) (S against percent chord, M_1 = 0.650).")

    fig_mach, _ = plots.plot_surface_mach(solution, experiment=experiment if args.experiment_quantity == "mach" else None)
    fig_cp, _ = plots.plot_pressure_coefficient(solution, experiment=experiment if args.experiment_quantity == "S" else None)
    fig_conv = monitor.figure if monitor is not None and monitor.figure is not None else plots.plot_convergence(
        solution, config.iteration)[0]
    if args.save is not None:
        args.save.mkdir(parents=True, exist_ok=True)
        for name, fig in (("surface_mach.png", fig_mach), ("surface_cp.png", fig_cp), ("convergence.png", fig_conv)):
            fig.savefig(args.save / name, dpi=130)
            print(f"saved {args.save / name}")
    if not args.no_show:
        plt.ioff()
        plt.show()
    return 0 if failure is None else 2


if __name__ == "__main__":
    sys.exit(main())
