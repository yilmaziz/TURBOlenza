"""
compare_dunavant_aungier.py
===========================
The blade-surface Mach number of this solver on the geometry-driven MEDIUM grid against
Dunavant et al.'s measured points and Aungier's potential-flow curve for the NACA 65-(12A10)10
cascade at M1 = 0.65 (the digitized file validation/naca65_12a10_10/Dunavant_Aungier_Fix.csv: x/c
and surface Mach number; columns SS_experiment, PS_experiment, SS_potential, PS_potential).

    python scripts/compare_dunavant_aungier.py            # solve, plot, save, show
    python scripts/compare_dunavant_aungier.py --no-show  # solve, plot, save

The case is the one of scripts/run_full_solver.py (stagger 33 deg, solidity 1, inlet angle 45
deg, M1 0.65, lambda_rho = lambda_K = 0.7, tolerances 1e-8, the sonic stop on), with
GRID_LEVEL "medium" (grid.py section 9). Edit the settings below. Nothing is fitted or tuned to
the data: the data are drawn as they are (the experiment as points, Aungier's curve as a dashed
line, sorted along x/c), and this solver's wall nodes as solid lines.

Outputs (OUTPUT_DIR): surface_mach_comparison.png (the whole chord and a leading-edge close-up),
surface_mach_medium.csv (this solver's wall values) and comparison.txt (the peaks of all three).
"""

# ============================================================
# SETTINGS
# ============================================================

DATA = "validation/naca65_12a10_10/Dunavant_Aungier_Fix.csv"    # the corrected M 0.65 data (2026-11-13)
OUTPUT_DIR = "outputs/dunavant_aungier_comparison"
GRID_LEVEL = "very_fine"  # "coarse", "medium", "fine", "very_fine" (the grid.py section 9)
PROFILE_VARIANT = "kovach_sandercock"          # as scripts/run_full_solver.py; "kovach_sandercock" the project default
INLET_MACH = 0.65                       # the data's condition
LE_ZOOM = (0.0, 0.25)                   # x/c range of the leading-edge close-up

CASE = dict(case_name="dunavant_aungier_comparison", solver_mode="coupled_kutta", fixed_exit_angle_deg=None,
            gamma=1.4, gas_constant=287.05, inlet_total_pressure=101325.0, inlet_total_temperature=288.15, omega=0.0,
            stream_surface_radius=None, inlet_absolute_tangential_velocity=None, blade_family="NACA65",
            design_lift_coefficient=1.2, thickness_ratio=0.10, meanline_type="A10", surface_continuity="C2",
            n_camber_stations=101, chord=0.1, solidity=1.0, stagger_deg=33.0, inlet_flow_angle_deg=45.0,
            mass_flow_per_depth_override=None, upstream_extension_chords=1.0, downstream_extension_chords=1.0,
            density_tolerance=1e-8, velocity_tolerance=None, max_density_iterations=600, density_relaxation=0.7,
            stop_at_sonic=True, kutta_tolerance=1e-8, max_kutta_iterations=600, kutta_relaxation=0.7,
            initial_exit_angle_mode="blade_te_angle", initial_exit_angle_deg=None, measure_runtime=True)

# ============================================================

import argparse                                                                  # noqa: E402
import csv                                                                       # noqa: E402
import math                                                                      # noqa: E402
import sys                                                                       # noqa: E402
from pathlib import Path                                                         # noqa: E402

import numpy as np                                                               # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SERIES = ("SS_experiment", "PS_experiment", "SS_potential", "PS_potential")


def read_data(path: Path) -> dict:
    """{series: (x/c, M) arrays} from the digitized file: a row of series names (each over two
    columns), a row X, Y, then the points; empty cells end a shorter series. A series is
    recognized by its name: SS / PS first, then "exp..." (Dunavant's test) or "aungier" /
    "potential" (Aungier's curve), any case; so SS_Experimentt or SS_Aungier read as well."""
    with open(path, newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.reader(handle))
    names = [name.strip() for name in rows[0]]
    out = {}
    for k in range(0, len(names), 2):
        name = names[k].lower()
        if not name:
            continue
        side = "SS" if name.startswith("ss") else "PS" if name.startswith("ps") else None
        source = ("experiment" if "exp" in name else
                  "potential" if ("aungier" in name or "potential" in name) else None)
        if side is None or source is None:
            raise ValueError(f"{path}: cannot tell what series {names[k]!r} is (SS/PS and experiment/Aungier)")
        points = [(float(r[k]), float(r[k + 1])) for r in rows[2:] if len(r) > k + 1 and r[k].strip() and r[k + 1].strip()]
        xy = np.array(points)
        out[f"{side}_{source}"] = (xy[:, 0], xy[:, 1])
    missing = set(SERIES) - set(out)
    if missing:
        raise ValueError(f"{path} lacks the series {sorted(missing)}")
    return out


def solve():
    from turbolenza.b2b import controller as ct

    settings = ct.CaseSettings(**CASE, profile_variant=PROFILE_VARIANT, inlet_mach=INLET_MACH, grid_level=GRID_LEVEL,
                               n_passage=None, n_eta=None)
    case = ct.build_case(settings)
    print(ct.format_grid_summary(case))
    run = ct.run_case(case)
    print(ct.format_run_report(run))
    return case, run


def peak(x, m):
    k = int(np.argmax(m))
    return float(m[k]), float(x[k])


def figure(data: dict, surfaces, run, path: Path) -> None:
    import matplotlib.pyplot as plt

    from turbolenza.b2b import plots

    ss, ps = plots.SUCTION_COLOR, plots.PRESSURE_COLOR
    info = run.case.standard_grid.info
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.8), gridspec_kw=dict(width_ratios=(1.6, 1.0)))
    fig.subplots_adjust(left=0.06, right=0.98, top=0.86, bottom=0.19, wspace=0.16)
    for ax, zoom in ((axes[0], None), (axes[1], LE_ZOOM)):
        ax.plot(surfaces.x_c_ss, surfaces.mach_ss, "-", color=ss, lw=2.0, label="this solver, suction side", zorder=3)
        ax.plot(surfaces.x_c_ps, surfaces.mach_ps, "-", color=ps, lw=2.0, label="this solver, pressure side", zorder=3)
        for name, color, label in (("SS_potential", ss, "Aungier potential flow, suction side"),
                                   ("PS_potential", ps, "Aungier potential flow, pressure side")):
            x, m = data[name]
            order = np.argsort(x)
            ax.plot(x[order], m[order], "--", color=color, lw=1.4, alpha=0.9, label=label, zorder=2)
        for name, color, marker, label in (("SS_experiment", ss, "o", "Dunavant et al. 1955, suction side"),
                                           ("PS_experiment", ps, "s", "Dunavant et al. 1955, pressure side")):
            x, m = data[name]
            ax.plot(x, m, marker, color=color, mfc="white", mew=1.6, ms=7, linestyle="none", label=label, zorder=4)
        ax.grid(alpha=0.3)
        ax.set_xlabel("x / c")
        ax.set_ylabel("surface Mach number")
        if zoom is not None:
            ax.set_xlim(*zoom)
            inside = [m[(x >= zoom[0]) & (x <= zoom[1])] for x, m in
                      [(surfaces.x_c_ss, surfaces.mach_ss), (surfaces.x_c_ps, surfaces.mach_ps), *data.values()]]
            values = np.concatenate([v for v in inside if v.size])
            ax.set_ylim(values.min() - 0.03, values.max() + 0.03)
            ax.set_title("leading edge, x/c 0 to 0.25", fontsize=10)
        else:
            ax.set_xlim(-0.02, 1.02)
            ax.set_title("whole chord", fontsize=10)
            handles, labels = ax.get_legend_handles_labels()
            fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9, frameon=False)
    sol = run.solution
    state = "converged" if run.converged else f"NOT CONVERGED ({type(run.error).__name__})"
    fig.suptitle(f"{run.case.blade_name} ({PROFILE_VARIANT}), M1 = {INLET_MACH:.2f}, beta1 = "
                 f"{CASE['inlet_flow_angle_deg']:.0f} deg, solidity {CASE['solidity']:g}: surface Mach number\n"
                 f"this solver on the {GRID_LEVEL} grid ({info.n_m} x {info.n_eta} nodes, dm/c {info.dm_over_chord:.4f}; "
                 f"{state}, beta_out {math.degrees(sol.exit_angle):.2f} deg) against Dunavant's test and Aungier's "
                 f"potential flow", fontsize=11)
    fig.savefig(path, dpi=150)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="This solver (medium grid) against Dunavant's and Aungier's M 0.65 data")
    parser.add_argument("--no-show", action="store_true", help="do not open the window")
    args = parser.parse_args(argv)
    import matplotlib

    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from turbolenza.b2b import postprocess as pp

    data = read_data(ROOT / DATA)
    case, run = solve()
    if run.solution is None:
        print("no solution to compare")
        return 2
    surfaces = pp.blade_surfaces(run.solution)
    out = ROOT / OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "surface_mach_medium.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["surface", "x_c", "mach"])
        writer.writerows(("suction", x, m) for x, m in zip(surfaces.x_c_ss, surfaces.mach_ss))
        writer.writerows(("pressure", x, m) for x, m in zip(surfaces.x_c_ps, surfaces.mach_ps))
    lines = [f"surface Mach peaks (value at x/c), M1 = {INLET_MACH}, {GRID_LEVEL} grid, {PROFILE_VARIANT}"]
    for label, (x, m) in (("this solver, suction side", (surfaces.x_c_ss, surfaces.mach_ss)),
                          ("Aungier potential, suction side", data["SS_potential"]),
                          ("Dunavant experiment, suction side", data["SS_experiment"]),
                          ("this solver, pressure side", (surfaces.x_c_ps, surfaces.mach_ps)),
                          ("Aungier potential, pressure side", data["PS_potential"]),
                          ("Dunavant experiment, pressure side", data["PS_experiment"])):
        value, where = peak(np.asarray(x), np.asarray(m))
        lines.append(f"  {label:38s} {value:.4f} at x/c {where:.4f}")
    mach = np.asarray(run.solution.mach)
    lines.append(f"  beta_out this solver {math.degrees(run.solution.exit_angle):.3f} deg; status "
                 f"{'converged' if run.converged else 'NOT CONVERGED'}")
    lines.append(f"  this solver's far-inlet Mach (station 0, mean over eta) {float(np.mean(mach[0])):.5f}, far-exit "
                 f"{float(np.mean(mach[-1])):.5f}; mdot' {run.solution.mass_flow_per_depth:.6g} kg/(m s)")
    (out / "comparison.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    figure(data, surfaces, run, out / "surface_mach_comparison.png")
    print(f"written to {out}")
    if not args.no_show:
        plt.show()
    return 0 if run.converged else 2


if __name__ == "__main__":
    raise SystemExit(main())
