"""
analyze_grid_family.py
======================
The geometry-driven five-level grid family of grid.py (section 9) for the current NACA
65-(12A10)10 case, and what it does to the solution.

    python scripts/analyze_grid_family.py             # grids, timing, solver study, figures
    python scripts/analyze_grid_family.py --no-solve  # grids and timing only

What it does
 1. Builds the cascade (the frozen baseline of scripts/diagnose_mach060_vs_065.py: 65-(12A10)10,
    kovach_sandercock, C2, stagger 33 deg, solidity 1, inlet angle 45 deg).
 2. Audits the old explicit grids: the admissible n_passage of the section-2 lattice and the
    hand-picked grids of the earlier studies (where their first and last blade stations fall).
 3. Generates the five-level GridPackage, prints its metadata, validates its topology.
 4. Times grid generation (time.perf_counter): every level built TIMING_REPEATS times alone
    (build_grid, the cost an optimizer pays per candidate), the whole family as often; median,
    minimum, maximum. These repeated grids are never solved.
 5. Optionally runs M1 = INLET_MACH on all five grids with the production solver (the sonic
    stop on, one set of relaxations and tolerances for every level), and reports beta_out,
    passes, times, peak Mach, the Kutta residual, the surface peaks and the mass-flow error.
 6. Exports CSV tables: the grid family, the timing, the beta_out study, the transition
    diagnostics (S, tan beta, A .. E, psi_m, psi_eta, W_m, W_theta, M at f - 1, f, f + 1 and
    l - 1, l, l + 1), the Kutta / TE quantities, the old-grid comparison, the candidates.
 7. Figures: beta_out against dm / c, the surface Mach numbers of the five levels, and LAST
    the mandatory grid_transition_wall_nodes.png (the actual first / last wall nodes of all
    five grids at the LE and TE, with the stations f - 1 and l + 1).

Nothing here changes a numerical module; the grids are the package's own (no station is
reconstructed from metadata).
"""

# ============================================================
# SETTINGS
# ============================================================

OUTPUT_DIR = "outputs/grid_family_analysis"
RUN_SOLVER = True                       # the five M1 solves (about 1 to 2 minutes in parallel)
INLET_MACH = 0.65
N_WORKERS = 5
TIMING_REPEATS = 9                      # per level (single build) and for the family
OLD_GRIDS = ((43, 29), (61, 41), (87, 59), (51, 41))   # the hand-picked explicit grids of the earlier studies
OLD_RANGE = range(10, 201)              # n_passage scanned for the old admissibility list
OLD_BETA_OUT = "outputs/le_velocity_fix_verification/rows.json"   # the M 0.65 production runs on the old grids

GEOMETRY = dict(blade_family="NACA65", design_lift_coefficient=1.2, thickness_ratio=0.10, meanline_type="A10",
                profile_variant="kovach_sandercock", surface_continuity="C2", n_camber_stations=101,
                chord=0.1, solidity=1.0, stagger_deg=33.0)
SOLVER = dict(case_name="grid_family", solver_mode="coupled_kutta", fixed_exit_angle_deg=None, gamma=1.4,
              gas_constant=287.05, inlet_total_pressure=101325.0, inlet_total_temperature=288.15, omega=0.0,
              stream_surface_radius=None, inlet_absolute_tangential_velocity=None, inlet_flow_angle_deg=45.0,
              mass_flow_per_depth_override=None, upstream_extension_chords=1.0, downstream_extension_chords=1.0,
              density_tolerance=1e-8, velocity_tolerance=None, max_density_iterations=600, density_relaxation=0.7,
              stop_at_sonic=True, kutta_tolerance=1e-8, max_kutta_iterations=600, kutta_relaxation=0.7,
              initial_exit_angle_mode="blade_te_angle", initial_exit_angle_deg=None, measure_runtime=True)

# ============================================================

import argparse                                                                  # noqa: E402
import csv                                                                       # noqa: E402
import json                                                                      # noqa: E402
import math                                                                      # noqa: E402
import statistics                                                                # noqa: E402
import sys                                                                       # noqa: E402
import time                                                                      # noqa: E402
import warnings                                                                  # noqa: E402
from concurrent.futures import ProcessPoolExecutor                               # noqa: E402
from pathlib import Path                                                         # noqa: E402

import numpy as np                                                               # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / OUTPUT_DIR
LINES: list = []
LEVEL_INK = ("#9ecae1", "#6baed6", "#3182bd", "#08519c", "#08306b")      # one hue, light (coarse) -> dark (fine)
LEVEL_MARK = ("o", "s", "^", "D", "v")


# --- helpers ----------------------------------------------------------------------------------------------------

def say(text: str = "") -> None:
    print(text)
    LINES.append(text)


def fmt(value, digits=4) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (float, np.floating)):
        value = float(value)
        if value != 0.0 and (abs(value) < 1e-3 or abs(value) >= 1e5):
            return f"{value:.{digits}e}"
        return f"{value:.{digits}f}"
    return str(value)


def table(rows: list, columns: list, title: str, digits=4) -> None:
    say(title)
    cells = [[fmt(r.get(k), digits) for k, _ in columns] for r in rows]
    widths = [max([len(h)] + [len(c[n]) for c in cells]) for n, (_, h) in enumerate(columns)]
    say("  " + "  ".join(h.ljust(w) for (_, h), w in zip(columns, widths)))
    for c in cells:
        say("  " + "  ".join(x.ljust(w) for x, w in zip(c, widths)))
    say()


def write_csv(name: str, rows: list) -> None:
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(OUT / name, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: (json.dumps(v) if isinstance(v, (list, tuple, dict)) else v) for k, v in r.items()})


def cascade():
    from turbolenza.b2b.geometry import CascadeSpec, make_cascade

    g = GEOMETRY
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return make_cascade(CascadeSpec(blade_type=g["blade_family"], solidity=g["solidity"], stagger_deg=g["stagger_deg"],
                                        cl0=g["design_lift_coefficient"], t_max_over_c=g["thickness_ratio"],
                                        meanline_type=g["meanline_type"], profile_variant=g["profile_variant"],
                                        physical_chord=None, n_camber_stations=g["n_camber_stations"],
                                        surface_continuity=g["surface_continuity"]))


def inlet_angle() -> float:
    return math.radians(SOLVER["inlet_flow_angle_deg"])


def case_settings(level: str, mach: float):
    from turbolenza.b2b import controller as ct

    return ct.CaseSettings(**SOLVER, **GEOMETRY, inlet_mach=mach, grid_level=level, n_passage=None, n_eta=None)


# --- the figure data (tested: tests/test_grid_family_script.py) -------------------------------------------------

def figure_data(package) -> list:
    """Per level, exactly what grid_transition_wall_nodes.png draws, read from the generated
    grids themselves: the station indices f, l, f - 1, l + 1, their m, and the wall nodes of
    f and l (the suction node and the pressure node, the latter moved back one pitch onto the
    reference blade, chosen by the grid's wall_identity). Lengths in chords."""
    out = []
    for g in package:
        grid = g.grid
        c = grid.chord
        f, l = grid.first_blade_i, grid.last_blade_i
        shift = np.asarray(grid.boundaries.periodic_translation)
        identity = grid.wall_identity

        def walls(i):
            lower, upper = np.asarray(grid.nodes[i, 0]), np.asarray(grid.nodes[i, -1]) - shift
            return (lower, upper) if identity.lower.surface[i - f] == "suction" else (upper, lower)

        first_ss, first_ps = walls(f)
        last_ss, last_ps = walls(l)
        out.append(dict(level=g.level, dm_over_chord=grid.dm / c, f=f, l=l, f_minus_1=f - 1, l_plus_1=l + 1,
                        m_f=float(grid.m[f]) / c, m_f_minus_1=float(grid.m[f - 1]) / c, m_l=float(grid.m[l]) / c,
                        m_l_plus_1=float(grid.m[l + 1]) / c, first_ss=first_ss / c, first_ps=first_ps / c,
                        last_ss=last_ss / c, last_ps=last_ps / c, le_gap=g.info.le_gap_over_chord,
                        te_gap=g.info.te_gap_over_chord))
    return out


# --- 2. the old explicit grids ----------------------------------------------------------------------------------

def old_grid_audit(c, package) -> tuple[list, tuple, int | None]:
    from turbolenza.b2b import grid as gr
    from turbolenza.b2b.config import GridSettings

    tr = package.transitions
    settings = GridSettings(n_passage=61, n_eta=41)
    admissible = gr.find_valid_n_passage(c, settings, inlet_angle(), OLD_RANGE)
    ceiling = gr.n_passage_ceiling(c, settings, inlet_angle())
    levels = [(g.level, g.info.dm) for g in package]
    rows = []
    for n, n_eta in OLD_GRIDS:
        grid = gr.make_grid(c, GridSettings(n_passage=n, n_eta=n_eta), inlet_angle())
        le, te = gr.wall_station_topology(grid)
        nearest = min(levels, key=lambda item: abs(math.log(item[1] / grid.dm)))[0]
        rows.append(dict(old_n_passage=n, old_n_eta=n_eta, n_m=len(grid.m), old_dm_over_c=grid.dm / grid.chord,
                         f=grid.first_blade_i, l=grid.last_blade_i,
                         m_f_minus_m_le_over_c=(grid.m_first_blade - tr.m_le) / grid.chord,
                         m_te_minus_m_l_over_c=(tr.m_te - grid.m_last_blade) / grid.chord,
                         le_fraction_of_dm=(grid.m_first_blade - tr.m_start) / grid.dm,
                         te_fraction_of_dm=(tr.m_end - grid.m_last_blade) / grid.dm,
                         admissible=le.valid and te.valid, old_cell_aspect=grid.pitch * grid.deta / grid.dm,
                         nearest_new_level=nearest))
    return rows, admissible, ceiling


# --- 4. timing --------------------------------------------------------------------------------------------------

def timing_study(c) -> tuple[list, dict]:
    from turbolenza.b2b import grid as gr

    rows = []
    for level in gr.GRID_LEVELS:
        times, phases = [], None
        for _ in range(TIMING_REPEATS):
            built = gr.build_grid(c, level, inlet_angle=inlet_angle())
            times.append(built.timing["total"])
            phases = dict(built.timing)
        rows.append(dict(level=level, repeats=TIMING_REPEATS, median_s=statistics.median(times), min_s=min(times),
                         max_s=max(times), **{f"last_{k}_s": v for k, v in phases.items()}))
    family = []
    for _ in range(TIMING_REPEATS):
        started = time.perf_counter()
        gr.build_grid_family(c, inlet_angle=inlet_angle())
        family.append(time.perf_counter() - started)
    return rows, dict(repeats=TIMING_REPEATS, median_s=statistics.median(family), min_s=min(family), max_s=max(family))


# --- 5. the solver study ----------------------------------------------------------------------------------------

def solve_level(level: str) -> dict:
    """Build and solve the case on one level (a worker process); the diagnostics as plain data."""
    warnings.simplefilter("ignore")
    from turbolenza.b2b import controller as ct
    from turbolenza.b2b import kutta as kt
    from turbolenza.b2b import postprocess as pp
    from turbolenza.b2b import solver as sv

    case = ct.build_case(case_settings(level, INLET_MACH))
    std = case.standard_grid
    run = ct.run_case(case)
    sol, err = run.solution, run.error
    row = dict(level=level, dm_over_c=std.info.dm_over_chord, deta=std.info.deta, n_m=std.info.n_m, n_eta=std.info.n_eta,
               total_nodes=std.info.total_nodes, grid_s=std.timing["total"],
               solver_s=None if run.runtime is None else run.runtime.wall_time_s,
               status="converged" if run.converged else f"failed: {type(err).__name__}", failure="" if err is None else str(err)[:300])
    out = dict(row=row, transition=[], te=None, surface=None)
    if sol is None:
        return out
    summary = pp.summarize(sol)
    mach = np.asarray(sol.mach)
    i, j = np.unravel_index(int(np.argmax(mach)), mach.shape)
    grid = sol.grid
    f, l, J = grid.first_blade_i, grid.last_blade_i, len(grid.eta) - 1
    surfaces = pp.blade_surfaces(sol)
    row.update(passes=len(sol.passes), iterations=sol.iterations, beta_out_deg=math.degrees(sol.exit_angle),
               kutta_residual=sol.kutta_residual, eps_k=sol.normalized_kutta_residual, peak_mach=float(mach[i, j]),
               peak_node=f"({i}, {j})", peak_rel=f"(f{i - f:+d}, {j})", peak_mach_all_passes=max(p.mach_max for p in sol.passes),
               ss_peak=summary.ss_peak_mach, ss_peak_x_c=summary.ss_peak_x_c, ps_peak=summary.ps_peak_mach,
               ps_peak_x_c=summary.ps_peak_x_c, mass_flow_error_max=summary.mass_flow_error_max,
               mass_flow_error_blade_median=summary.mass_flow_error_blade_median,
               mass_flow_per_depth=sol.mass_flow_per_depth, exit_velocity_angle_deg=summary.exit_velocity_angle_deg)
    # transition diagnostics
    c, b = grid.chord, grid.boundaries
    le, te = np.asarray(b.lower_blade_boundary[0]), np.asarray(b.lower_blade_boundary[-1])
    axis = te - le
    shift = np.asarray(b.periodic_translation)
    met, co, flow = sol.metrics, sol.coefficients, sol.flow
    rows_j = sorted({0, max(1, round(0.1 * J)), J // 2, J})
    for label, k in (("f-1", f - 1), ("f", f), ("f+1", f + 1), ("l-1", l - 1), ("l", l), ("l+1", l + 1)):
        interior = slice(1, J)
        for jj in rows_j:
            point = np.asarray(grid.nodes[k, jj]) - (shift if jj == J else 0.0)
            out["transition"].append(dict(
                level=level, station=label, i=k, j=jj, eta=float(grid.eta[jj]), m_over_c=float(grid.m[k]) / c,
                x_c=float(np.dot(point - le, axis) / np.dot(axis, axis)), S_over_c=float(met.S[k]) / c,
                tan_beta=float(met.tan_beta[k, jj]), A=float(co.A[k, jj]), B=float(co.B[k, jj]), C=float(co.C[k, jj]),
                D=float(co.D[k, jj]), E=float(co.E[k, jj]), psi_m=float(flow.psi_m[k, jj]), psi_eta=float(flow.psi_eta[k, jj]),
                W_m=float(flow.W_m[k, jj]), W_theta=float(flow.W_theta[k, jj]), mach=float(mach[k, jj]),
                station_max_mach=float(np.max(mach[k])), station_max_abs_D=float(np.max(np.abs(co.D[k, interior]))),
                station_max_abs_E=float(np.max(np.abs(co.E[k, interior])))))
    # Kutta / TE
    residual = kt.kutta_residual(flow, grid)
    correction = kt.exit_angle_correction(residual, flow, met)
    st = residual.station
    out["te"] = dict(level=level, te_station=st.i, te_m_over_c=st.m / c, te_m_offset_over_c=st.m_offset / c,
                     W_ss=residual.W_ss, W_ps=residual.W_ps, residual=residual.residual, eps_k=residual.normalized,
                     wall_difference=correction.wall_difference, delta_c_theta=correction.delta_c_theta,
                     exit_meridional_integral=correction.exit_meridional_integral,
                     passage_width_over_c=correction.passage_width / c, dm_over_c=correction.dm / c,
                     cos_beta0=correction.cos_beta0, cos_beta1=correction.cos_beta1, delta_tan=correction.delta_tan,
                     te_wall_tan_beta_ss=float(met.tan_beta[st.i, st.ss_j]), te_wall_tan_beta_ps=float(met.tan_beta[st.i, st.ps_j]),
                     te_wall_x_c_ss=float(surfaces.x_c_ss[-1]), te_wall_x_c_ps=float(surfaces.x_c_ps[-1]))
    out["surface"] = dict(x_c_ss=np.asarray(surfaces.x_c_ss), mach_ss=np.asarray(surfaces.mach_ss),
                          x_c_ps=np.asarray(surfaces.x_c_ps), mach_ps=np.asarray(surfaces.mach_ps))
    # the first-blade-station Mach profile against its neighbours: the LE spike check
    out["row"].update(max_mach_f_minus_1=float(np.max(mach[f - 1])), max_mach_f=float(np.max(mach[f])),
                      max_mach_f_plus_1=float(np.max(mach[f + 1])))
    return out


def classify(dm: list, beta: list) -> tuple[str, str]:
    """The beta_out sequence ordered coarse -> fine: "monotonic convergence" (changes of one
    sign, each smaller than the one before), "oscillatory convergence" (alternating signs,
    shrinking), "apparently converged" (all levels within 0.05 deg), otherwise
    "non-convergent / scattered"; "insufficient evidence" below three levels. No Richardson
    extrapolation is forced."""
    if len(beta) < 3:
        return "insufficient evidence", "fewer than three converged levels"
    d = np.diff(beta)
    span = max(beta) - min(beta)
    changes = ", ".join(f"{x:+.4f}" for x in d)
    if span < 0.05:
        return "apparently converged", f"all levels within {span:.3f} deg"
    signs = np.sign(d)
    shrinking = all(abs(d[k + 1]) < abs(d[k]) for k in range(len(d) - 1))
    if np.all(signs == signs[0]) and shrinking:
        ratios = ", ".join(f"{abs(d[k] / d[k + 1]):.2f}" for k in range(len(d) - 1))
        return "monotonic convergence", f"one-signed changes {changes} deg, shrinking by {ratios} per level"
    if np.all(signs[1:] != signs[:-1]) and shrinking:
        return "oscillatory convergence", f"alternating changes {changes} deg, shrinking"
    return "non-convergent / scattered", (f"changes {changes} deg per level: neither one-signed and shrinking nor "
                                          f"alternating and shrinking; span {span:.4f} deg")


def growth(values: list, dm: list) -> list:
    """Apparent exponent p of |value| ~ dm^-p between neighbouring levels."""
    return [math.log(abs(values[k + 1]) / abs(values[k])) / math.log(dm[k] / dm[k + 1]) for k in range(len(values) - 1)]


# --- figures ----------------------------------------------------------------------------------------------------

def beta_figure(rows: list, old: list, path: Path) -> None:
    """beta_out against dm / c (left) and against the Kutta station's distance from the TE,
    (m_TE - m_l) / c (right), the five levels and the old explicit grids of the same case."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    done = [r for r in rows if r.get("beta_out_deg") is not None]
    for ax, key, old_key, label in ((axes[0], "dm_over_c", "dm_over_c", "dm / c"),
                                    (axes[1], "te_gap_over_c", "te_gap_over_c", "(m_TE - m_l) / c: Kutta station from the TE")):
        ax.plot([r[key] for r in done], [r["beta_out_deg"] for r in done], "-", color="#3182bd", lw=2, zorder=2,
                label="geometry-driven family (grid.py section 9)")
        for r in done:
            k = ["very_coarse", "coarse", "medium", "fine", "very_fine"].index(r["level"])
            ax.plot(r[key], r["beta_out_deg"], LEVEL_MARK[k], color=LEVEL_INK[k], mec="#08306b", ms=9, zorder=3)
            ax.annotate(r["level"].replace("_", " "), (r[key], r["beta_out_deg"]), textcoords="offset points",
                        xytext=(6, 6), fontsize=8, color="#333333")
        if old:
            ax.plot([o[old_key] for o in old], [o["beta_out_deg"] for o in old], "x", color="#7a7a7a", ms=9, mew=2,
                    label="old explicit grids (same case)")
            for o in old:
                ax.annotate(o["label"], (o[old_key], o["beta_out_deg"]), textcoords="offset points", xytext=(6, -12),
                            fontsize=8, color="#7a7a7a")
        ax.set_xlabel(label)
        ax.set_xlim(left=0.0)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("beta_out [deg]")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Discharge angle on the five standard levels and the old grids, M1 = {INLET_MACH:.2f}", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def surface_figure(results: list, path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.6))
    for k, res in enumerate(results):
        s = res.get("surface")
        if s is None:
            continue
        ax.plot(s["x_c_ss"], s["mach_ss"], color=LEVEL_INK[k], lw=1.6, label=f"{res['row']['level']} SS")
        ax.plot(s["x_c_ps"], s["mach_ps"], color=LEVEL_INK[k], lw=1.0, ls="--", label=f"{res['row']['level']} PS")
    ax.axhline(1.0, color="#c0392b", lw=1.0, ls=":")
    ax.set_xlabel("x / c")
    ax.set_ylabel("wall Mach number")
    ax.set_title(f"Blade-surface Mach number on the five levels, M1 = {INLET_MACH:.2f}")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def transition_figure(c, package, path: Path) -> None:
    """THE LAST FIGURE: the LE and TE of the blade with the first / last valid station of every
    level (its cut line and both wall nodes) and, dashed, the neighbouring station f - 1 /
    l + 1 outside the blade. Everything from figure_data, the grids' own arrays."""
    import matplotlib.pyplot as plt

    data = figure_data(package)
    tr = package.transitions
    chord = tr.chord
    u = np.linspace(0.0, 1.0, 40001)
    ss = c.active_curves.suction.evaluate(u) / chord
    ps = c.active_curves.pressure.evaluate(u) / chord
    le = np.asarray(c.active.leading_edge_point) / chord
    te = np.asarray(c.active.trailing_edge_point) / chord
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 6.2))
    spans = {"LE": (min(d["m_f_minus_1"] for d in data) - 0.006, max(d["m_f"] for d in data) + 0.006),
             "TE": (min(d["m_l"] for d in data) - 0.006, max(d["m_l_plus_1"] for d in data) + 0.006)}
    for ax, edge in ((left, "LE"), (right, "TE")):
        lo, hi = spans[edge]
        inside = lambda xy: (xy[:, 0] >= lo - 0.01) & (xy[:, 0] <= hi + 0.01)       # noqa: E731
        ax.plot(ss[inside(ss), 0], ss[inside(ss), 1], color="#111111", lw=2.4, zorder=4, label="suction surface (C2 curve)")
        ax.plot(ps[inside(ps), 0], ps[inside(ps), 1], color="#555555", lw=2.4, ls=(0, (6, 2)), zorder=4,
                label="pressure surface (C2 curve)")
        ys = np.concatenate([ss[inside(ss), 1], ps[inside(ps), 1]])
        y_lo, y_hi = ys.min() - 0.01, ys.max() + 0.01
        if edge == "LE":
            ax.plot(*le, "*", color="#c0392b", ms=14, zorder=6, label="LE point (construction; = SS+PS-valid start)")
            ax.axvline(tr.m_tip / chord, color="#c0392b", lw=1.0, ls=":", zorder=1, label="nose tip (most upstream point)")
            ax.axvspan(tr.m_tip / chord, tr.m_start / chord, color="#f4c7c3", alpha=0.5, zorder=0,
                       label="LE window: no station allowed")
        else:
            ax.plot(*te, "*", color="#c0392b", ms=14, zorder=6, label="TE point (construction; = SS+PS-valid end)")
            if tr.te_window > 0.0:
                ax.axvline(tr.m_tail / chord, color="#c0392b", lw=1.0, ls=":", zorder=1, label="tail end (most downstream point)")
        for k, d in enumerate(data):
            ink, mark = LEVEL_INK[k], LEVEL_MARK[k]
            if edge == "LE":
                m_cut, m_out, a, b = d["m_f"], d["m_f_minus_1"], d["first_ss"], d["first_ps"]
                text = f"{d['level']}: dm/c {d['dm_over_chord']:.4f}, (m_f - m_LE)/c {d['le_gap']:.4f}"
            else:
                m_cut, m_out, a, b = d["m_l"], d["m_l_plus_1"], d["last_ss"], d["last_ps"]
                text = f"{d['level']}: dm/c {d['dm_over_chord']:.4f}, (m_TE - m_l)/c {d['te_gap']:.4f}"
            ax.plot([m_cut, m_cut], [b[1], a[1]], color=ink, lw=2.0, zorder=3)
            ax.plot(a[0], a[1], mark, color=ink, mec="#08306b", ms=9, zorder=5, label=text)
            ax.plot(b[0], b[1], mark, color=ink, mec="#08306b", ms=9, mfc="white", zorder=5)
            ax.axvline(m_out, color=ink, lw=1.2, ls="--", alpha=0.8, zorder=1)
        ax.set_xlim(lo, hi)
        ax.set_ylim(y_lo, y_hi)
        ax.set_xlabel("m / c")
        ax.set_ylabel("y / c")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7, loc="upper left" if edge == "LE" else "lower left", framealpha=0.92)
    left.set_title("Leading edge: first valid station f (cut, SS node filled, PS node open);\n"
                   "dashed: f - 1, outside the blade (axes not to equal scale)", fontsize=10)
    right.set_title("Trailing edge: last valid station l (cut, SS node filled, PS node open);\n"
                    "dashed: l + 1, outside the blade (axes not to equal scale)", fontsize=10)
    fig.suptitle(f"{c.metadata.get('designation', 'blade')}, stagger {math.degrees(c.stagger):.0f} deg: transition wall "
                 f"nodes of the five standard grid levels (geometry {package.geometry_id})", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# --- main -------------------------------------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="The geometry-driven five-level grid family and its solver study")
    parser.add_argument("--no-solve", action="store_true", help="grids and timing only")
    args = parser.parse_args(argv)
    import matplotlib

    matplotlib.use("Agg")
    from turbolenza.b2b import grid as gr

    OUT.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    c = cascade()
    say(f"cascade: {c.metadata.get('designation', '?')} {GEOMETRY['profile_variant']}, stagger {GEOMETRY['stagger_deg']} deg, "
        f"solidity {GEOMETRY['solidity']}, {GEOMETRY['surface_continuity']} (built in {time.perf_counter() - started:.1f} s, "
        f"not grid time)")
    package = gr.build_grid_family(c, inlet_angle=inlet_angle())
    tr = package.transitions
    say(f"geometry id {package.geometry_id}")
    say()

    # 1. the blade ends
    say("1. BLADE ENDS ALONG m (chords from the LE point)")
    for name, value in (("nose tip (outline extreme)", tr.m_tip), ("construction LE point", tr.m_le),
                        ("SS + PS-valid start", tr.m_start), ("SS + PS-valid end", tr.m_end),
                        ("construction TE point", tr.m_te), ("tail end (outline extreme)", tr.m_tail),
                        ("LE face (middle of the LE window)", tr.le_face), ("TE face (middle of the TE window)", tr.te_face)):
        say(f"   {name:36s} {(value - tr.m_le) / tr.chord:+.9f}")
    say(f"   LE window {tr.le_window / tr.chord:.6e} c, TE window {tr.te_window / tr.chord:.6e} c; lattice length "
        f"{tr.lattice_length / tr.chord:.9f} c")
    say("   crossings per interval (m_lo, m_hi rel. LE in chords; SS, PS):")
    for lo, hi, s_, p_ in tr.coverage:
        say(f"     [{(lo - tr.m_le) / tr.chord:+.6e}, {(hi - tr.m_le) / tr.chord:+.6e}]  SS {s_}  PS {p_}")
    say()

    # 2. old grids
    old_rows, admissible, ceiling = old_grid_audit(c, package)
    write_csv("old_grid_comparison.csv", old_rows)
    say(f"2. OLD EXPLICIT GRIDS (make_grid section 2: m = linspace(m_le - 1c, m_te + 1c), dm from n_passage)")
    say(f"   admissible n_passage {OLD_RANGE.start} .. {OLD_RANGE.stop - 1}: {len(admissible)} of {len(OLD_RANGE)}; "
        f"inadmissible: {[n for n in OLD_RANGE if n not in admissible][:40]} ...; ceiling (every n from it fails) {ceiling}")
    table(old_rows, [("old_n_passage", "n_passage"), ("old_n_eta", "n_eta"), ("n_m", "n_m"), ("old_dm_over_c", "dm/c"),
                     ("f", "f"), ("l", "l"), ("m_f_minus_m_le_over_c", "(m_f-m_LE)/c"), ("m_te_minus_m_l_over_c", "(m_TE-m_l)/c"),
                     ("le_fraction_of_dm", "(m_f-m_LE)/dm"), ("te_fraction_of_dm", "(m_TE-m_l)/dm"),
                     ("old_cell_aspect", "aspect"), ("admissible", "admissible"), ("nearest_new_level", "nearest level")], "")

    # 3. the family
    fam = []
    for g in package:
        i, s = g.info, g.candidates
        fam.append(dict(level=i.level, target_dm_over_c=i.target_dm_over_chord, dm_over_c=i.dm_over_chord, deta=i.deta,
                        n_m=i.n_m, n_eta=i.n_eta, total_nodes=i.total_nodes, n_blade_cells=i.n_blade_cells, f=i.first_blade_i,
                        l=i.last_blade_i, n_blade_stations=i.n_blade_stations, m_f_over_c=i.m_first / tr.chord,
                        m_l_over_c=i.m_last / tr.chord, le_gap_over_c=i.le_gap_over_chord, te_gap_over_c=i.te_gap_over_chord,
                        first_ss=i.first_ss_wall_point, first_ps=i.first_ps_wall_point, last_ss=i.last_ss_wall_point,
                        last_ps=i.last_ps_wall_point, first_ss_x_c=i.first_ss_x_c, first_ps_x_c=i.first_ps_x_c,
                        last_ss_x_c=i.last_ss_x_c, last_ps_x_c=i.last_ps_x_c, upstream_ext_c=i.upstream_extension_chords,
                        downstream_ext_c=i.downstream_extension_chords, cell_aspect=i.cell_aspect,
                        admissibility=i.admissibility, checks=len(i.checks), candidates_tested=s.tested,
                        candidates_rejected=s.rejected, accepted_rank=s.accepted_rank, reasons=s.reasons,
                        search_band=s.band))
    for k in range(len(fam) - 1):
        fam[k + 1]["ratio_to_coarser"] = fam[k]["dm_over_c"] / fam[k + 1]["dm_over_c"]
        fam[k + 1]["deta_ratio_to_coarser"] = fam[k]["deta"] / fam[k + 1]["deta"]
    write_csv("grid_family_summary.csv", fam)
    table(fam, [("level", "level"), ("target_dm_over_c", "target dm/c"), ("dm_over_c", "dm/c"), ("ratio_to_coarser", "r"),
                ("deta", "deta"), ("n_m", "n_m"), ("n_eta", "n_eta"), ("total_nodes", "nodes"), ("f", "f"), ("l", "l"),
                ("n_blade_stations", "blade st."), ("le_gap_over_c", "(m_f-m_LE)/c"), ("te_gap_over_c", "(m_TE-m_l)/c"),
                ("upstream_ext_c", "L_up/c"), ("downstream_ext_c", "L_down/c"), ("cell_aspect", "aspect"),
                ("candidates_tested", "tested"), ("accepted_rank", "rank")], "3. THE FIVE-LEVEL FAMILY")
    table(fam, [("level", "level"), ("m_f_over_c", "m_f/c"), ("first_ss", "first SS node (m, y)"),
                ("first_ss_x_c", "x/c"), ("first_ps", "first PS node (m, y)"), ("first_ps_x_c", "x/c"),
                ("m_l_over_c", "m_l/c"), ("last_ss", "last SS node"), ("last_ss_x_c", "x/c"), ("last_ps", "last PS node"),
                ("last_ps_x_c", "x/c")], "   wall points (reference blade; chords)")
    say("   topology / invariants per level (validate_standard_grid on the built grid):")
    for g in package:
        checks = gr.validate_standard_grid(g.grid, tr)
        say(f"     {g.level:12s} {len(checks)} checks passed: {', '.join(checks)}")
    say()
    cand = [dict(level=g.level, n_cells=r.n_cells, dm_over_c=r.dm_over_chord, rank=r.rank, accepted=r.accepted, reason=r.reason,
                 detail=r.detail) for g in package for r in g.candidates.records]
    write_csv("grid_candidates.csv", cand)

    # 4. timing
    timing_rows, family_time = timing_study(c)
    write_csv("grid_timing.csv", timing_rows + [dict(level="family (all five)", **family_time)])
    table(timing_rows + [dict(level="family (all five)", **family_time)],
          [("level", "level"), ("repeats", "repeats"), ("median_s", "median s"), ("min_s", "min s"), ("max_s", "max s"),
           ("last_interrogation_s", "interrogation"), ("last_intersection_s", "intersections"),
           ("last_construction_s", "construction"), ("last_validation_s", "validation"), ("last_search_s", "search")],
          f"4. GRID-GENERATION TIME (time.perf_counter, {TIMING_REPEATS} repeats; never solved)")
    medium = next(r for r in timing_rows if r["level"] == "medium")["median_s"]
    say("   serial projection with the medium level, T_N = N t_grid,medium "
        f"(t = {medium:.4f} s): " + ", ".join(f"{n:,} geometries {n * medium:.1f} s ({n * medium / 60:.2f} min)"
                                               for n in (100, 1000, 10000)))
    say(f"   the package's own clock: {', '.join(f'{k} {v:.4f} s' for k, v in package.timing.items())}")
    say()

    results, beta_rows = [], []
    if RUN_SOLVER and not args.no_solve:
        with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
            results = list(pool.map(solve_level, gr.GRID_LEVELS))
        grid_time = {r["level"]: r["median_s"] for r in timing_rows}
        for res in results:
            row = res["row"]
            row["grid_s_median"] = grid_time[row["level"]]
            row["te_gap_over_c"] = package[row["level"]].info.te_gap_over_chord
            row["le_gap_over_c"] = package[row["level"]].info.le_gap_over_chord
            if row.get("solver_s"):
                row["grid_fraction"] = row["grid_s_median"] / (row["grid_s_median"] + row["solver_s"])
            beta_rows.append(row)
        write_csv("beta_out_grid_study.csv", beta_rows)
        table(beta_rows, [("level", "level"), ("dm_over_c", "dm/c"), ("deta", "deta"), ("n_m", "n_m"), ("n_eta", "n_eta"),
                          ("total_nodes", "nodes"), ("grid_s_median", "grid s"), ("solver_s", "solver s"),
                          ("grid_fraction", "grid/(grid+solver)"), ("status", "status"), ("passes", "passes"),
                          ("beta_out_deg", "beta_out deg"), ("kutta_residual", "R_K m/s"), ("eps_k", "eps_K")],
              f"5. SOLVER IMPACT, M1 = {INLET_MACH:.2f} (stop on, lambda_rho = lambda_K = 0.7, tolerances 1e-8)", digits=5)
        table(beta_rows, [("level", "level"), ("peak_mach", "peak M"), ("peak_rel", "at"), ("ss_peak", "SS peak"),
                          ("ss_peak_x_c", "x/c"), ("ps_peak", "PS peak"), ("ps_peak_x_c", "x/c"),
                          ("max_mach_f_minus_1", "max M f-1"), ("max_mach_f", "max M f"), ("max_mach_f_plus_1", "max M f+1"),
                          ("mass_flow_error_max", "mdot err max"), ("mass_flow_error_blade_median", "blade median")], "", digits=5)
        done = [r for r in beta_rows if r.get("beta_out_deg") is not None and r["status"] == "converged"]
        verdict, why = classify([r["dm_over_c"] for r in done], [r["beta_out_deg"] for r in done])
        say(f"   beta_out convergence: {verdict.upper()} ({why})")
        old_beta = []
        path = ROOT / OLD_BETA_OUT
        if path.exists():
            rows_old = json.loads(path.read_text(encoding="utf-8"))
            for tag, label, n, n_eta in (("P_43x29_M0.650", "43x29", 43, 29), ("P_M0.650", "61x41", 61, 41),
                                         ("P_87x59_M0.650", "87x59", 87, 59)):
                if tag in rows_old and rows_old[tag].get("beta_out_deg") is not None:
                    o = next((x for x in old_rows if x["old_n_passage"] == n), None)
                    old_beta.append(dict(label=label, dm_over_c=o["old_dm_over_c"] if o else None,
                                         te_gap_over_c=o["m_te_minus_m_l_over_c"] if o else None,
                                         beta_out_deg=rows_old[tag]["beta_out_deg"]))
            say("   old explicit grids, same case (scripts/verify_le_velocity_fix.py): " +
                ", ".join(f"{o['label']} {o['beta_out_deg']:.4f} deg (dm/c {o['dm_over_c']:.5f})" for o in old_beta))
            if done:
                span_old = max(o["beta_out_deg"] for o in old_beta) - min(o["beta_out_deg"] for o in old_beta)
                span_new = max(r["beta_out_deg"] for r in done) - min(r["beta_out_deg"] for r in done)
                say(f"   span of beta_out: old grids {span_old:.4f} deg, new family {span_new:.4f} deg")
        say()
        beta_figure(beta_rows, old_beta, OUT / "beta_out_vs_dm.png")
        surface_figure(results, OUT / "surface_mach_five_levels.png")
        transition = [t for res in results for t in res["transition"]]
        write_csv("transition_diagnostics.csv", transition)
        jc = {res["row"]["level"]: sorted({t["j"] for t in res["transition"]})[1] for res in results if res["transition"]}
        shown = [t for t in transition if t["j"] == jc.get(t["level"])]
        table(shown, [("level", "level"), ("station", "st."), ("i", "i"), ("j", "j"), ("m_over_c", "m/c"), ("x_c", "x/c"),
                      ("S_over_c", "S/c"), ("tan_beta", "tan b"), ("A", "A"), ("B", "B"), ("C", "C"), ("D", "D"), ("E", "E"),
                      ("station_max_abs_D", "max|D|"), ("station_max_abs_E", "max|E|")],
              "6. TRANSITION COEFFICIENTS at the row eta ~ 0.1 (all rows j = 0, ~0.1 J, J/2, J in transition_diagnostics.csv)",
              digits=4)
        table(shown, [("level", "level"), ("station", "st."), ("eta", "eta"), ("psi_m", "psi_m"), ("psi_eta", "psi_eta"),
                      ("W_m", "W_m"), ("W_theta", "W_theta"), ("mach", "M"), ("station_max_mach", "station max M")],
              "   velocity quantities (converged)", digits=4)
        say("   max |D| (interior rows) per station, coarse -> fine, and the apparent growth exponent p of |D| ~ dm^-p:")
        dms = [r["row"]["dm_over_c"] for r in results if r["transition"]]
        for station in ("f-1", "f", "f+1", "l-1", "l", "l+1"):
            values = [next(t["station_max_abs_D"] for t in r["transition"] if t["station"] == station) for r in results
                      if r["transition"]]
            say(f"     {station:4s} " + ", ".join(f"{v:9.1f}" for v in values) + "   p = " +
                ", ".join(f"{x:.2f}" for x in growth(values, dms)))
        say("   max |E| (interior rows) per station:")
        for station in ("f-1", "f", "f+1", "l-1", "l", "l+1"):
            values = [next(t["station_max_abs_E"] for t in r["transition"] if t["station"] == station) for r in results
                      if r["transition"]]
            say(f"     {station:4s} " + ", ".join(f"{v:9.2f}" for v in values) + "   p = " +
                ", ".join(f"{x:.2f}" for x in growth(values, dms)))
        say()
        te_rows = [res["te"] for res in results if res["te"] is not None]
        write_csv("kutta_te_diagnostics.csv", te_rows)
        table(te_rows, [("level", "level"), ("te_station", "l"), ("te_m_over_c", "m_l/c"), ("te_m_offset_over_c", "(m_l-m_TE)/c"),
                        ("W_ss", "W_SS"), ("W_ps", "W_PS"), ("residual", "R_K"), ("eps_k", "eps_K"),
                        ("passage_width_over_c", "S(l)/c"), ("cos_beta0", "cos b0"), ("cos_beta1", "cos b1"),
                        ("exit_meridional_integral", "int W_m deta"), ("dm_over_c", "dm/c"), ("te_wall_tan_beta_ss", "tan b SS"),
                        ("te_wall_tan_beta_ps", "tan b PS"), ("te_wall_x_c_ss", "x/c SS"), ("te_wall_x_c_ps", "x/c PS")],
              "7. KUTTA / TE (Eq. 5-34 at l; the quantities of the Aungier correction, Eqs. 13-57 / 13-59)", digits=5)
        if len(te_rows) > 1:
            w = [t["W_ss"] for t in te_rows]
            say("   TE speed W_SS = W_PS at l, coarse -> fine: " + ", ".join(f"{x:.3f}" for x in w) + " m/s; changes " +
                ", ".join(f"{b - a:+.3f}" for a, b in zip(w[:-1], w[1:])) + "; last-station x/c (SS) " +
                ", ".join(f"{t['te_wall_x_c_ss']:.4f}" for t in te_rows))
            say()
        (OUT / "solver_rows.json").write_text(json.dumps(beta_rows, indent=1, default=str), encoding="utf-8")

    # LAST: the mandatory transition figure
    transition_figure(c, package, OUT / "grid_transition_wall_nodes.png")
    say("8. FINAL FIGURE grid_transition_wall_nodes.png (the grids' own stations and wall nodes):")
    for d in figure_data(package):
        say(f"   {d['level']:12s} f {d['f']:3d} (m_f/c {d['m_f']:+.6f}, f-1 at {d['m_f_minus_1']:+.6f})  "
            f"(m_f - m_LE)/c {d['le_gap']:.6f}   l {d['l']:3d} (m_l/c {d['m_l']:.6f}, l+1 at {d['m_l_plus_1']:.6f})  "
            f"(m_TE - m_l)/c {d['te_gap']:.6f}")
    (OUT / "report.txt").write_text("\n".join(LINES) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
