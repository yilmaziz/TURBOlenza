"""
kutta_study.py
==============
Grid-convergence study of the trailing-edge Kutta condition for the NACA 65-(12A10)10 cascade
of Dunavant et al. (1955) at M1 = 0.65 on the five standard grid levels (grid.py section 9).
A STUDY: every method it tries sits behind an option of config.IterationSettings whose
default is the production behaviour; nothing here changes a default.

    python scripts/kutta_study.py phase0      # audit: the Kutta station and the TE grid of every level
    python scripts/kutta_study.py phase1      # Kutta at a fixed physical location m* = m_TE - delta
    python scripts/kutta_study.py phase2      # the Stokes loop of Eq. 13-57 in this project's frame
    python scripts/kutta_study.py phase3      # Aungier's 13-57 / 13-58 / 13-59 corrections, relaxation scan
    python scripts/kutta_study.py phase4      # figures and the CSV deliverables from the saved data

Each phase writes its data to OUTPUT_DIR/data/*.json and its tables to OUTPUT_DIR/*.csv, and
prints what it found. The case is the frozen baseline of scripts/analyze_grid_family.py (the
source of the beta_out 25.48 / 24.37 / 23.57 / 22.68 / 23.28 deg of the last report):
kovach_sandercock, C2, stagger 33 deg, solidity 1, inlet 45 deg, M1 0.65, lambda_rho =
lambda_K = 0.7, all tolerances 1e-8, at most 600 passes, the sonic stop on, the first exit
angle kappa_te.

Frames (this project = Aungier 2003 Ch. 5): beta from the meridional direction, W_m = W
cos(beta), W_theta = W sin(beta), tan(beta) = r dtheta/dm = dy/dm; eta = 0 is the wall
theta0, eta = 1 the wall theta1, S = y1 - y0. Planar cascade: r = 1.
"""

# ============================================================
# SETTINGS
# ============================================================

OUTPUT_DIR = "outputs/kutta_study"
N_WORKERS = 10
LEVELS = ("very_coarse", "coarse", "medium", "fine", "very_fine")
DELTAS = (0.01, 0.02, 0.03, 0.05)           # Phase 1: m* = m_TE - delta * chord
INLET_MACH = 0.65
DATA = "validation/naca65_12a10_10/Dunavant_Aungier_Fix.csv"
TARGETS = {"Aungier 2003 (potential flow)": 24.4, "experiment (Aungier 2003, p. 92)": 25.0}

GEOMETRY = dict(blade_family="NACA65", design_lift_coefficient=1.2, thickness_ratio=0.10, meanline_type="A10",
                profile_variant="kovach_sandercock", surface_continuity="C2", n_camber_stations=101,
                chord=0.1, solidity=1.0, stagger_deg=33.0)
SOLVER = dict(case_name="kutta_study", solver_mode="coupled_kutta", fixed_exit_angle_deg=None, gamma=1.4,
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
import sys                                                                       # noqa: E402
import warnings                                                                  # noqa: E402
from pathlib import Path                                                         # noqa: E402

import numpy as np                                                               # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / OUTPUT_DIR
LINES: list = []


# --- helpers ----------------------------------------------------------------------------------------------------

def say(text: str = "") -> None:
    print(text)
    LINES.append(text)


def fmt(value, digits=4) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        value = float(value)
        if value != 0.0 and (abs(value) < 1e-3 or abs(value) >= 1e5):
            return f"{value:.{digits}e}"
        return f"{value:.{digits}f}"
    return str(value)


def table(rows: list, columns: list, title: str, digits=4) -> None:
    """A plain markdown table (printed and kept for the report)."""
    say(f"**{title}**")
    say()
    say("| " + " | ".join(h for _, h in columns) + " |")
    say("|" + "|".join("---" for _ in columns) + "|")
    for r in rows:
        say("| " + " | ".join(fmt(r.get(k), digits) for k, _ in columns) + " |")
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


def save(name: str, value) -> None:
    (OUT / "data").mkdir(parents=True, exist_ok=True)
    (OUT / "data" / name).write_text(json.dumps(value, indent=1, default=_jsonable), encoding="utf-8")


def load(name: str):
    return json.loads((OUT / "data" / name).read_text(encoding="utf-8"))


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    raise TypeError(type(value).__name__)


def write_lines(name: str) -> None:
    (OUT / name).write_text("\n".join(LINES) + "\n", encoding="utf-8")


def case_of(level: str, **iteration):
    """The controller case of one level with the study's iteration options (dataclasses.replace
    on the built config: the controller itself is unchanged)."""
    from dataclasses import replace

    from turbolenza.b2b import controller as ct

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        case = ct.build_case(ct.CaseSettings(**SOLVER, **GEOMETRY, inlet_mach=INLET_MACH, grid_level=level,
                                             n_passage=None, n_eta=None))
    if iteration:
        case = replace(case, config=replace(case.config, iteration=replace(case.config.iteration, **iteration)))
    return case


def chordwise(grid, point) -> float:
    """x/c of a point on the reference blade: its projection on the chord axis LE -> TE."""
    b = grid.boundaries
    le, te = np.asarray(b.lower_blade_boundary[0]), np.asarray(b.lower_blade_boundary[-1])
    axis = te - le
    return float(np.dot(np.asarray(point) - le, axis) / np.dot(axis, axis))


def te_walls(grid):
    """(SS node, PS node) of the TE station l on the reference blade (the upper wall moved back
    one pitch), the SS / PS from the grid's wall identity."""
    from turbolenza.b2b import kutta as kt

    st = kt.trailing_edge_station(grid)
    shift = np.asarray(grid.boundaries.periodic_translation)
    J = len(grid.eta) - 1
    node = {0: np.asarray(grid.nodes[st.i, 0]), J: np.asarray(grid.nodes[st.i, J]) - shift}
    return st, node[st.ss_j], node[st.ps_j]


# --- Phase 0: the Kutta station and the TE grid -----------------------------------------------------------------

def phase0() -> None:
    from turbolenza.b2b import kutta as kt
    from turbolenza.b2b.metrics import build_metrics

    say("## Phase 0: where and how the Kutta condition is imposed")
    say()
    rows = []
    for level in LEVELS:
        case = case_of(level)
        grid, info = case.grid, case.standard_grid.info
        c = grid.chord
        met = build_metrics(grid)
        st, ss, ps = te_walls(grid)
        l = st.i
        tan0, tan1 = float(met.tan_beta0[l]), float(met.tan_beta1[l])
        cos0, cos1 = 1.0 / math.hypot(1.0, tan0), 1.0 / math.hypot(1.0, tan1)
        S = float(met.S[l])
        rows.append(dict(
            level=level, n_m=len(grid.m), n_eta=len(grid.eta), dm_over_c=grid.dm / c, deta=grid.deta,
            cell_pitchwise_over_c=grid.pitch * grid.deta / c, K=l, K_minus_f=l - grid.first_blade_i,
            m_K_over_c=float(grid.m[l]) / c, m_te_over_c=float(grid.m_te) / c,
            te_distance_m_pct=100.0 * float(grid.m_te - grid.m[l]) / c,
            te_distance_in_dm=float(grid.m_te - grid.m[l]) / grid.dm,
            te_window_over_c=info.te_gap_over_chord,
            ss_j=st.ss_j, ps_j=st.ps_j, x_c_ss=chordwise(grid, ss), x_c_ps=chordwise(grid, ps),
            tan_beta0=tan0, tan_beta1=tan1, beta0_deg=math.degrees(math.atan(tan0)), beta1_deg=math.degrees(math.atan(tan1)),
            cos_beta0=cos0, cos_beta1=cos1, S_over_c=S / c,
            gain_dm_over_S_cos=2.0 * grid.dm / (S * (cos0 + cos1)),
            path_harmonic_over_c=2.0 * grid.dm / (cos0 + cos1) / c,
            path_arithmetic_over_c=0.5 * grid.dm * (1.0 / cos0 + 1.0 / cos1) / c))
    table(rows, [("level", "level"), ("n_m", "n_m"), ("n_eta", "n_eta"), ("dm_over_c", "dm/c"),
                 ("deta", "deta"), ("cell_pitchwise_over_c", "s deta/c"), ("K", "K = l"),
                 ("te_distance_m_pct", "(m_TE - m_K)/c %"), ("te_distance_in_dm", "(m_TE - m_K)/dm"),
                 ("te_window_over_c", "TE window/c"), ("x_c_ss", "x/c SS node"), ("x_c_ps", "x/c PS node")],
          "Kutta station K and the TE grid spacing per level", digits=5)
    table(rows, [("level", "level"), ("ss_j", "SS j"), ("ps_j", "PS j"), ("beta0_deg", "beta0 deg"),
                 ("beta1_deg", "beta1 deg"), ("cos_beta0", "cos b0"), ("cos_beta1", "cos b1"), ("S_over_c", "S/c"),
                 ("gain_dm_over_S_cos", "2 dm/(S(cos b0+cos b1))")],
          "Wall angles and the gain factor of Eq. 13-57 at K", digits=5)
    write_csv("te_grid_spacing.csv", rows)
    save("phase0.json", rows)
    ratio = [r["gain_dm_over_S_cos"] / r["dm_over_c"] for r in rows]
    say(f"gain / (dm/c) per level: {', '.join(f'{x:.4f}' for x in ratio)} (the gain is dm times a factor near "
        f"2/(S(cos b0 + cos b1)) ~ 1)")
    say()
    write_lines("phase0.md")
    _ = kt


# --- the solves (worker processes) ------------------------------------------------------------------------------

def label_of(job: dict) -> str:
    it = job.get("iteration", {})
    method = it.get("kutta_method", "current")
    where = ("station" if it.get("kutta_eval", "station") == "station"
             else f"m*={100 * it['kutta_eval_offset_chords']:g}%c {it.get('kutta_eval_interpolation', 'cubic')}")
    alpha = it.get("kutta_relaxation", SOLVER["kutta_relaxation"])
    return f"{method} | {where} | relax {alpha:g}"


def kutta_reduction(history: list) -> dict:
    """The Kutta residual's reduction per pass: a least-squares slope of ln(eps_K) over the
    passes with 1e-7 < eps_K < 1e-2 (the geometric phase), and the median ratio there."""
    eps = np.array([h["eps_k"] for h in history], dtype=float)
    n = np.arange(len(eps))
    window = (eps > 1e-7) & (eps < 1e-2) & np.isfinite(eps)
    if window.sum() < 4:
        return dict(factor=math.nan, median_ratio=math.nan, window_passes=int(window.sum()))
    slope = np.polyfit(n[window], np.log(eps[window]), 1)[0]
    idx = np.flatnonzero(window)
    ratios = eps[idx[1:]] / eps[idx[:-1]]
    return dict(factor=float(math.exp(slope)), median_ratio=float(np.median(ratios)), window_passes=int(window.sum()))


def run_job(job: dict) -> dict:
    """One coupled solve of one level with the job's iteration options; plain data back."""
    warnings.simplefilter("ignore")
    from turbolenza.b2b import controller as ct
    from turbolenza.b2b import kutta as kt
    from turbolenza.b2b import postprocess as pp

    case = case_of(job["level"], **job.get("iteration", {}))
    run = ct.run_case(case)
    sol, err = run.solution, run.error
    out = dict(job=job, label=label_of(job), level=job["level"], converged=bool(run.converged), status=run.status,
               error="" if err is None else f"{type(err).__name__}: {str(err)[:400]}",
               wall_s=None if run.runtime is None else run.runtime.wall_time_s)
    if sol is None:
        return out
    grid = sol.grid
    c = grid.chord
    history = [dict(n=p.index, final=p.final, kutta_residual=p.kutta_residual, eps_k=p.kutta_normalized,
                    beta_out_deg=math.degrees(p.exit_angle), beta_next_deg=math.degrees(p.exit_angle_next),
                    tan_correction=p.tan_correction, density_change=p.density_change,
                    velocity_change=p.velocity_change, mach_max=p.mach_max) for p in sol.passes]
    station = kt.kutta_residual(sol.flow, grid)
    probes = {}
    for delta in DELTAS:
        for interpolation in ("cubic", "linear"):
            try:
                walls = kt.wall_values(sol.flow, grid, "fixed_m", delta, interpolation)
                probes[f"{delta:g}_{interpolation}"] = dict(W0=walls.W0, W1=walls.W1, W0_minus_W1=walls.W0 - walls.W1,
                                                           eps=kt.residual_from_walls(walls, grid).normalized)
            except kt.KuttaError:
                pass
    surfaces = pp.blade_surfaces(sol)
    out.update(beta_out_deg=math.degrees(sol.exit_angle), passes=len(sol.passes), iterations=sol.iterations,
               eps_k=sol.normalized_kutta_residual, R_K=sol.kutta_residual, dm_over_c=grid.dm / c,
               station_R_K=station.residual, station_eps_k=station.normalized, station_W_ss=station.W_ss,
               station_W_ps=station.W_ps, probes=probes, history=history, reduction=kutta_reduction(history),
               peak_mach=float(np.max(sol.mach)),
               surface=dict(x_c_ss=np.asarray(surfaces.x_c_ss), mach_ss=np.asarray(surfaces.mach_ss),
                            x_c_ps=np.asarray(surfaces.x_c_ps), mach_ps=np.asarray(surfaces.mach_ps)))
    return out


def run_all(jobs: list) -> list:
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=min(N_WORKERS, len(jobs))) as pool:
        results = list(pool.map(run_job, jobs))
    for r in results:
        flag = "" if r["converged"] else f"   <- {r['status']}: {r['error'][:160]}"
        beta = fmt(r.get("beta_out_deg"), 4)
        print(f"  {r['level']:12s} {r['label']:40s} beta_out {beta:>9s}  passes {r.get('passes', '-')}{flag}")
    return results


def admissible(level_row: dict, delta: float) -> bool:
    """m* = m_TE - delta c within [m(first blade station), m(K)]: wall nodes on both sides."""
    m_star = level_row["m_te_over_c"] - delta
    return level_row["m_first_over_c"] <= m_star <= level_row["m_K_over_c"]


def level_rows() -> dict:
    rows = {}
    for level in LEVELS:
        grid = case_of(level).grid
        c = grid.chord
        rows[level] = dict(m_te_over_c=float(grid.m_te) / c, m_K_over_c=float(grid.m[grid.last_blade_i]) / c,
                           m_first_over_c=float(grid.m[grid.first_blade_i]) / c, dm_over_c=grid.dm / c)
    return rows


def differences(betas: list) -> dict:
    """Successive differences coarse -> fine, and the observed order from each triple when the
    differences have one sign and shrink (refinement ratio from the levels' dm)."""
    dm = [level_rows_cache[level]["dm_over_c"] for level in LEVELS]
    d = [betas[k + 1] - betas[k] for k in range(len(betas) - 1)]
    orders = []
    for k in range(len(d) - 1):
        r = math.sqrt((dm[k] / dm[k + 1]) * (dm[k + 1] / dm[k + 2]))
        if d[k] * d[k + 1] > 0 and abs(d[k + 1]) < abs(d[k]):
            orders.append(math.log(abs(d[k]) / abs(d[k + 1])) / math.log(r))
        else:
            orders.append(math.nan)
    monotone = all(x * d[0] > 0 for x in d) and all(abs(d[k + 1]) < abs(d[k]) for k in range(len(d) - 1))
    return dict(diffs=d, orders=orders, monotone=monotone)


level_rows_cache: dict = {}


# --- Phase 1: Kutta at a fixed physical location ----------------------------------------------------------------

def phase1() -> None:
    level_rows_cache.update(level_rows())
    say("## Phase 1: Kutta at a fixed physical location m* = m_TE - delta")
    say()
    grid_rows = [dict(level=level, **{f"delta_{100 * d:g}pct": admissible(level_rows_cache[level], d) for d in DELTAS})
                 for level in LEVELS]
    table(grid_rows, [("level", "level")] + [(f"delta_{100 * d:g}pct", f"delta {100 * d:g} % c admissible")
                                             for d in DELTAS], "Wall nodes on both sides of m* (the rule of Phase 1)")
    full = [d for d in DELTAS if all(admissible(level_rows_cache[level], d) for level in LEVELS)]
    partial = [d for d in DELTAS if d not in full]
    say(f"deltas admissible on every level: {', '.join(f'{100 * d:g} %' for d in full)}; the others are run only on "
        f"the levels that admit them, as a supplementary check outside the rule.")
    say()
    jobs = [dict(level=level, phase=1, iteration={}) for level in LEVELS]
    for d in DELTAS:
        for interpolation in ("cubic", "linear"):
            for level in LEVELS:
                if admissible(level_rows_cache[level], d):
                    jobs.append(dict(level=level, phase=1, supplementary=d in partial,
                                     iteration=dict(kutta_eval="fixed_m", kutta_eval_offset_chords=d,
                                                    kutta_eval_interpolation=interpolation)))
    results = run_all(jobs)
    save("phase1.json", dict(results=results, levels=level_rows_cache, full=full, partial=partial))
    report_phase1(results, full, partial)
    write_lines("phase1.md")


def variant_key(r: dict) -> tuple:
    it = r["job"].get("iteration", {})
    return (it.get("kutta_method", "current"), it.get("kutta_eval", "station"), it.get("kutta_eval_offset_chords"),
            it.get("kutta_eval_interpolation", "cubic") if it.get("kutta_eval", "station") == "fixed_m" else None,
            it.get("kutta_relaxation", SOLVER["kutta_relaxation"]))


def by_variant(results: list) -> dict:
    out: dict = {}
    for r in results:
        out.setdefault(variant_key(r), {})[r["level"]] = r
    return out


def report_phase1(results: list, full: list, partial: list) -> None:
    variants = by_variant(results)
    rows, conv_rows = [], []
    for key, runs in variants.items():
        label = next(iter(runs.values()))["label"]
        row = dict(variant=label)
        for level in LEVELS:
            r = runs.get(level)
            row[level] = r.get("beta_out_deg") if r and r["converged"] else None
        betas = [row[level] for level in LEVELS]
        if all(b is not None for b in betas):
            info = differences(betas)
            row.update({f"d{k + 1}": x for k, x in enumerate(info["diffs"])})
            row.update(fine_vs_very_fine=abs(betas[-1] - betas[-2]), monotone=info["monotone"],
                       orders=", ".join(fmt(x, 2) for x in info["orders"]))
        rows.append(row)
        for level in LEVELS:
            r = runs.get(level)
            if r is None:
                continue
            conv_rows.append(dict(variant=label, level=level, status=r["status"], passes=r.get("passes"),
                                  reduction_per_pass=r.get("reduction", {}).get("factor"),
                                  median_ratio=r.get("reduction", {}).get("median_ratio"),
                                  final_eps_k=r.get("eps_k"), station_eps_k=r.get("station_eps_k"),
                                  wall_s=r.get("wall_s")))
    cols = [("variant", "variant")] + [(level, level) for level in LEVELS]
    table(rows, cols, "beta_out [deg] by level and Kutta evaluation (current correction, lambda_K 0.7)")
    table(rows, [("variant", "variant"), ("d1", "vc->c"), ("d2", "c->m"), ("d3", "m->f"), ("d4", "f->vf"),
                 ("fine_vs_very_fine", "|f - vf|"), ("monotone", "monotone"), ("orders", "observed order p")],
          "Successive differences [deg] and the observed order (only from one-signed, shrinking triples)")
    table(conv_rows, [("variant", "variant"), ("level", "level"), ("status", "status"), ("passes", "passes"),
                      ("reduction_per_pass", "eps_K factor/pass"), ("median_ratio", "median ratio"),
                      ("final_eps_k", "final eps_K (driven)"), ("station_eps_k", "eps_K at station K"),
                      ("wall_s", "wall s")], "Convergence of each solve")
    # the decision rule, on the deltas admissible on every level (cubic)
    say("**Decision rule (Phase 1)**, on the deltas admissible on every level, cubic interpolation:")
    say()
    for key, runs in variants.items():
        method, where, delta, interpolation, _ = key
        if where != "fixed_m" or delta not in full:
            continue
        betas = [runs[level].get("beta_out_deg") if level in runs and runs[level]["converged"] else None for level in LEVELS]
        if any(b is None for b in betas):
            say(f"- delta {100 * delta:g} % ({interpolation}): not every level converged")
            continue
        gap = abs(betas[-1] - betas[-2])
        case = "(a)" if gap < 0.1 and differences(betas)["monotone"] else "(b)"
        say(f"- delta {100 * delta:g} % ({interpolation}): |fine - very_fine| = {gap:.4f} deg, monotone "
            f"{differences(betas)['monotone']} -> case {case}")
    say()


# --- Phase 2: the Stokes loop in this project's frame -----------------------------------------------------------

def stokes_rows(sol) -> tuple[list, dict]:
    """Per pair of consecutive stations (k-1, k): Gamma_k - Gamma_(k-1) against the wall
    integrals with 1/cos(beta) (this frame) and 1/sin(beta) (the tangential convention left
    unconverted). Gamma_k = S_k int_0^1 W_theta d(eta) (= int W_theta r dtheta, r = 1; the
    trapezoidal rule in eta, the solver's); the wall integrals by the trapezoidal rule in m.
    Lengths in chords, speeds in m/s."""
    grid, met, flow = sol.grid, sol.metrics, sol.flow
    c = grid.chord
    f, l, J = grid.first_blade_i, grid.last_blade_i, len(grid.eta) - 1
    W = np.hypot(flow.W_m, flow.W_theta)
    gamma = np.asarray(met.S) / c * np.trapezoid(np.asarray(flow.W_theta), grid.eta, axis=1)
    b0, b1 = np.arctan(np.asarray(met.tan_beta0)), np.arctan(np.asarray(met.tan_beta1))
    g_cos = W[:, J] / np.cos(b1) - W[:, 0] / np.cos(b0)
    g_sin = W[:, J] / np.sin(b1) - W[:, 0] / np.sin(b0)
    rows = []
    for k in range(f + 1, l + 1):
        h = float(grid.m[k] - grid.m[k - 1]) / c
        lhs = float(gamma[k] - gamma[k - 1])
        rhs_cos = 0.5 * h * float(g_cos[k] + g_cos[k - 1])
        rhs_sin = 0.5 * h * float(g_sin[k] + g_sin[k - 1])
        x_mid = 0.5 * (chordwise(grid, grid.nodes[k, 0]) + chordwise(grid, grid.nodes[k - 1, 0]))
        rows.append(dict(element=f"({k - 1},{k})", k=k, k_from_f=k - f, k_to_l=l - k, m_mid_over_c=0.5 * float(grid.m[k] + grid.m[k - 1]) / c,
                         x_c_ss_mid=x_mid, gamma_km1=float(gamma[k - 1]), gamma_k=float(gamma[k]), lhs=lhs,
                         rhs_cos=rhs_cos, rhs_sin=rhs_sin, err_cos=abs(lhs - rhs_cos), err_sin=abs(lhs - rhs_sin),
                         ratio_cos=rhs_cos / lhs if lhs != 0.0 else math.nan,
                         ratio_sin=rhs_sin / lhs if lhs != 0.0 else math.nan,
                         beta0_deg=math.degrees(float(b0[k])), beta1_deg=math.degrees(float(b1[k]))))
    upstream, downstream = gamma[:f], gamma[l + 1:]
    periodic = dict(gamma_upstream_mean=float(np.mean(upstream)), gamma_downstream_mean=float(np.mean(downstream)),
                    upstream_spread=float(np.ptp(upstream)), downstream_spread=float(np.ptp(downstream)),
                    downstream_spread_rel=float(np.ptp(downstream) / abs(np.mean(downstream))),
                    upstream_spread_rel=float(np.ptp(upstream) / abs(np.mean(upstream))),
                    jump_le=float(gamma[f] - gamma[f - 1]), jump_te=float(gamma[l + 1] - gamma[l]),
                    blade_circulation=float(np.mean(downstream) - np.mean(upstream)),
                    gamma=gamma.tolist(), m_over_c=(np.asarray(grid.m) / c).tolist(), f=f, l=l)
    return rows, periodic


def path_rows() -> list:
    """The two path-length averages of the TE cell (stations K-1, K) on every level, beside
    the straight node-to-node lengths of the two walls over that cell."""
    from turbolenza.b2b.metrics import build_metrics

    rows = []
    for level in LEVELS:
        grid = case_of(level).grid
        met = build_metrics(grid)
        c, K, J = grid.chord, grid.last_blade_i, len(grid.eta) - 1
        cos0 = 1.0 / math.hypot(1.0, float(met.tan_beta0[K]))
        cos1 = 1.0 / math.hypot(1.0, float(met.tan_beta1[K]))
        dm = float(grid.m[K] - grid.m[K - 1]) / c
        chord0 = float(np.linalg.norm(np.asarray(grid.nodes[K, 0]) - np.asarray(grid.nodes[K - 1, 0]))) / c
        chord1 = float(np.linalg.norm(np.asarray(grid.nodes[K, J]) - np.asarray(grid.nodes[K - 1, J]))) / c
        cos0m = 1.0 / math.hypot(1.0, float(met.tan_beta0[K - 1]))
        cos1m = 1.0 / math.hypot(1.0, float(met.tan_beta1[K - 1]))
        trap0 = 0.5 * dm * (1.0 / cos0 + 1.0 / cos0m)
        trap1 = 0.5 * dm * (1.0 / cos1 + 1.0 / cos1m)
        rows.append(dict(level=level, dm_over_c=dm, cos_b0=cos0, cos_b1=cos1,
                         harmonic=2.0 * dm / (cos0 + cos1), arithmetic=0.5 * dm * (1.0 / cos0 + 1.0 / cos1),
                         rel_difference=(0.5 * dm * (1.0 / cos0 + 1.0 / cos1)) / (2.0 * dm / (cos0 + cos1)) - 1.0,
                         wall0_chord=chord0, wall1_chord=chord1, wall0_trapezoid=trap0, wall1_trapezoid=trap1))
    return rows


def phase2() -> None:
    from turbolenza.b2b import controller as ct

    say("## Phase 2: the Stokes loop of Eq. 13-57 in this project's frame")
    say()
    run = ct.run_case(case_of("very_fine"))
    sol = run.solution
    if not run.converged:
        say(f"very_fine did not converge ({run.status}): the check is made on the last completed pass")
    say(f"very_fine solution: {run.status}, beta_out {math.degrees(sol.exit_angle):.6f} deg, {len(sol.passes)} passes")
    say()
    rows, periodic = stokes_rows(sol)
    interior = [r for r in rows if r["k_from_f"] >= 3 and r["k_to_l"] >= 3]
    med = lambda key, rr: float(np.median([abs(r[key] - 1.0) for r in rr if math.isfinite(r[key])]))  # noqa: E731
    summary = dict(
        n_elements=len(rows), n_interior=len(interior),
        median_rel_err_cos_interior=med("ratio_cos", interior), median_rel_err_sin_interior=med("ratio_sin", interior),
        median_abs_err_cos_interior=float(np.median([r["err_cos"] for r in interior])),
        median_abs_err_sin_interior=float(np.median([r["err_sin"] for r in interior])),
        median_abs_lhs_interior=float(np.median([abs(r["lhs"]) for r in interior])),
        max_abs_err_cos_interior=float(max(r["err_cos"] for r in interior)),
        max_abs_err_sin_interior=float(max(r["err_sin"] for r in interior)))
    cols = [("element", "(k-1,k)"), ("k_to_l", "l-k"), ("x_c_ss_mid", "x/c (SS)"), ("beta0_deg", "b0 deg"),
            ("beta1_deg", "b1 deg"), ("lhs", "LHS dGamma"), ("rhs_cos", "RHS_cos"), ("rhs_sin", "RHS_sin"),
            ("err_cos", "|LHS-RHS_cos|"), ("err_sin", "|LHS-RHS_sin|")]
    table(rows[-8:], cols, "Near the TE: the last eight blade elements (very_fine; Gamma in m/s x chord)", digits=5)
    table(rows[:8], [("element", "(k-1,k)"), ("k_from_f", "k-f")] + cols[2:],
          "Near the LE: the first eight blade elements", digits=5)
    table([summary], [("n_interior", "interior elements"), ("median_abs_lhs_interior", "median |LHS|"),
                      ("median_abs_err_cos_interior", "median |LHS-RHS_cos|"),
                      ("median_abs_err_sin_interior", "median |LHS-RHS_sin|"),
                      ("median_rel_err_cos_interior", "median |RHS_cos/LHS-1|"),
                      ("median_rel_err_sin_interior", "median |RHS_sin/LHS-1|"),
                      ("max_abs_err_cos_interior", "max |LHS-RHS_cos|"), ("max_abs_err_sin_interior", "max |LHS-RHS_sin|")],
          "Interior elements (3 or more from either transition)", digits=5)
    table([periodic], [("gamma_upstream_mean", "Gamma upstream"), ("upstream_spread_rel", "upstream spread/|Gamma|"),
                       ("gamma_downstream_mean", "Gamma downstream"),
                       ("downstream_spread_rel", "downstream spread/|Gamma|"),
                       ("jump_le", "Gamma_f - Gamma_(f-1)"), ("jump_te", "Gamma_(l+1) - Gamma_l"),
                       ("blade_circulation", "Gamma_down - Gamma_up")],
          "Gamma on the periodic stations (no blade, omega = 0: constant expected)", digits=6)
    prow = path_rows()
    table(prow, [("level", "level"), ("dm_over_c", "dm/c"), ("cos_b0", "cos b0"), ("cos_b1", "cos b1"),
                 ("harmonic", "2dm/(cos b0+cos b1)"), ("arithmetic", "0.5dm(1/cos b0+1/cos b1)"),
                 ("rel_difference", "arith/harm - 1"), ("wall0_chord", "wall 0 node chord"),
                 ("wall1_chord", "wall 1 node chord"), ("wall0_trapezoid", "wall 0 trapz dm/cos"),
                 ("wall1_trapezoid", "wall 1 trapz dm/cos")],
          "Path-length averages of the TE cell (K-1, K), lengths in chords", digits=6)
    verdict = ("RHS_cos matches LHS to discretization error; RHS_sin does not"
               if summary["median_rel_err_cos_interior"] < 0.05 and summary["median_rel_err_sin_interior"] > 0.2 else
               "NEITHER matches cleanly: see the tables")
    say(f"Verdict on the interior elements: {verdict} (median |RHS/LHS - 1|: cos "
        f"{summary['median_rel_err_cos_interior']:.4f}, sin {summary['median_rel_err_sin_interior']:.4f}).")
    say()
    write_csv("stokes_check.csv", rows)
    write_csv("path_length_averages.csv", prow)
    save("phase2.json", dict(rows=rows, periodic=periodic, summary=summary, paths=prow,
                             beta_out_deg=math.degrees(sol.exit_angle), status=run.status, verdict=verdict))
    write_lines("phase2.md")


# --- Phase 3: Aungier's corrections behind kutta_method ---------------------------------------------------------

METHODS = ("current", "aungier_13_57", "aungier_13_57_58")
ALPHAS = (1.0, 0.5, 1.5)                    # Phase 3: the relaxation of the Aungier methods (1.0 the default)
STABILITY_ALPHAS = (2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0)    # aungier_13_57 at the station


def best_delta(p1: dict) -> float:
    """The delta of Phase 1 admissible on every level whose cubic run has the smallest
    |fine - very_fine| (ties: the larger delta)."""
    variants = by_variant(p1["results"])
    best, gap = None, math.inf
    for key, runs in variants.items():
        method, where, delta, interpolation, _ = key
        if where != "fixed_m" or interpolation != "cubic" or delta not in p1["full"]:
            continue
        if not all(level in runs and runs[level]["converged"] for level in ("fine", "very_fine")):
            continue
        g = abs(runs["very_fine"]["beta_out_deg"] - runs["fine"]["beta_out_deg"])
        if g < gap - 1e-12 or (abs(g - gap) <= 1e-12 and delta > best):
            best, gap = delta, g
    return best


def phase3_jobs(best: float) -> list:
    jobs = []
    for method in METHODS[1:]:
        for where in ("station", "fixed_m"):
            for alpha in ALPHAS:
                it = dict(kutta_method=method, kutta_relaxation=alpha)
                if where == "fixed_m":
                    it.update(kutta_eval="fixed_m", kutta_eval_offset_chords=best, kutta_eval_interpolation="cubic")
                jobs += [dict(level=level, phase=3, iteration=dict(it)) for level in LEVELS]
    for alpha in STABILITY_ALPHAS:
        jobs += [dict(level=level, phase=3, scan=True, iteration=dict(kutta_method="aungier_13_57", kutta_relaxation=alpha))
                 for level in LEVELS]
    return jobs


def sign_check() -> dict:
    """On the converged very_fine solution (current, station), at its frozen density: the
    change of W0 - W1 and of the far-exit circulation when tan(beta_out) moves by +-1e-4, by
    direct module calls (no solver loop)."""
    from turbolenza.b2b import controller as ct
    from turbolenza.b2b import kutta as kt
    from turbolenza.b2b.boundary_conditions import build_boundary_conditions
    from turbolenza.b2b.coefficients import build_coefficients
    from turbolenza.b2b.flow import velocity_components
    from turbolenza.b2b.grid import rebuild_exit
    from turbolenza.b2b.linear_solver import solve_linear_system
    from turbolenza.b2b.matrix_assembly import assemble_from_coefficients
    from turbolenza.b2b.metrics import build_metrics

    run = ct.run_case(case_of("very_fine"))
    sol = run.solution
    rho = np.array(sol.density)
    out = {}
    for sign in (-1.0, 1.0):
        beta = math.atan(math.tan(sol.exit_angle) + sign * 1e-4)
        g = rebuild_exit(sol.grid, beta)
        met, bcs = build_metrics(g), build_boundary_conditions(g)
        co = build_coefficients(met, rho, sol.mass_flow_per_depth, c_ref=sol.c_ref)
        flow = velocity_components(solve_linear_system(assemble_from_coefficients(co, met, bcs)).psi, rho, co, met, bcs)
        walls = kt.wall_values(flow, g)
        out[sign] = dict(W0_minus_W1=walls.W0 - walls.W1,
                         gamma_exit=float(met.S[-1] / g.chord * np.trapezoid(np.asarray(flow.W_theta)[-1], g.eta)),
                         gamma_K=float(met.S[g.last_blade_i] / g.chord
                                       * np.trapezoid(np.asarray(flow.W_theta)[g.last_blade_i], g.eta)))
    d = 2e-4
    return dict(d_W0_minus_W1_d_tan=(out[1.0]["W0_minus_W1"] - out[-1.0]["W0_minus_W1"]) / d,
                d_gamma_exit_d_tan=(out[1.0]["gamma_exit"] - out[-1.0]["gamma_exit"]) / d,
                d_gamma_K_d_tan=(out[1.0]["gamma_K"] - out[-1.0]["gamma_K"]) / d,
                beta_out_deg=math.degrees(sol.exit_angle))


def sign_changes(history: list) -> int:
    r = [h["kutta_residual"] for h in history if math.isfinite(h["kutta_residual"])]
    return int(sum(1 for a, b in zip(r, r[1:]) if a * b < 0.0))


def phase3() -> None:
    level_rows_cache.update(level_rows())
    p1 = load("phase1.json")
    best = best_delta(p1)
    say("## Phase 3: Aungier's Kutta correction (Eqs. 13-57, 13-58, 13-59) behind kutta_method")
    say()
    say(f"fixed_m uses the best delta of Phase 1: {100 * best:g} % chord (cubic).")
    say()
    results = run_all(phase3_jobs(best))
    keep = {("current", "station", None, None, SOLVER["kutta_relaxation"]),
            ("current", "fixed_m", best, "cubic", SOLVER["kutta_relaxation"])}
    current = [r for r in p1["results"] if variant_key(r) in keep]
    sign = sign_check()
    save("phase3.json", dict(results=results, current=current, best_delta=best, sign=sign, levels=level_rows_cache))
    report_phase3(results, current, best, sign)
    write_lines("phase3.md")


def report_phase3(results: list, current: list, best: float, sign: dict) -> None:
    main = by_variant(current + [r for r in results if not r["job"].get("scan")])
    rows, conv = [], []
    for key, runs in main.items():
        label = next(iter(runs.values()))["label"]
        row = dict(variant=label)
        for level in LEVELS:
            r = runs.get(level)
            row[level] = r.get("beta_out_deg") if r and r["converged"] else None
        betas = [row[level] for level in LEVELS]
        if all(b is not None for b in betas):
            info = differences(betas)
            row.update({f"d{k + 1}": x for k, x in enumerate(info["diffs"])})
            row.update(fine_vs_very_fine=abs(betas[-1] - betas[-2]), monotone=info["monotone"],
                       orders=", ".join(fmt(x, 2) for x in info["orders"]))
        rows.append(row)
        for level in LEVELS:
            r = runs.get(level)
            if r is not None:
                conv.append(dict(variant=label, level=level, status=r["status"], passes=r.get("passes"),
                                 reduction=r.get("reduction", {}).get("factor"), sign_changes=sign_changes(r.get("history", [])),
                                 final_eps_k=r.get("eps_k"), station_eps_k=r.get("station_eps_k"),
                                 first_tan_correction=(r.get("history") or [{}])[0].get("tan_correction")))
    table(rows, [("variant", "variant")] + [(level, level) for level in LEVELS],
          "beta_out [deg] by level: current (lambda_K 0.7), Aungier 13-57 and 13-57 + 13-58 (alpha 1.0, 0.5, 1.5)")
    table(rows, [("variant", "variant"), ("d1", "vc->c"), ("d2", "c->m"), ("d3", "m->f"), ("d4", "f->vf"),
                 ("fine_vs_very_fine", "|f - vf|"), ("monotone", "monotone"), ("orders", "observed order p")],
          "Successive differences [deg] and the observed order")
    table(conv, [("variant", "variant"), ("level", "level"), ("status", "status"), ("passes", "passes"),
                 ("reduction", "eps_K factor/pass"), ("sign_changes", "R_K sign changes"),
                 ("first_tan_correction", "pass-0 dtan (undamped)"), ("final_eps_k", "final eps (driven)"),
                 ("station_eps_k", "Eq. 5-34 eps at K")], "Convergence of each solve")
    scan = by_variant([r for r in results if r["job"].get("scan")]
                      + [r for r in results if variant_key(r)[:2] == ("aungier_13_57", "station")
                         and not r["job"].get("scan")])
    srows = []
    for key in sorted(scan, key=lambda k: k[4]):
        runs = scan[key]
        row = dict(alpha=key[4])
        for level in LEVELS:
            r = runs.get(level)
            row[level] = (f"{r['passes']} p" if r and r["converged"] else
                          ("fail: " + r["status"].split(": ")[-1]) if r else "-")
        srows.append(row)
    table(srows, [("alpha", "alpha")] + [(level, level) for level in LEVELS],
          "Relaxation scan of aungier_13_57 at the station: passes to converge, or the failure")
    limits = {}
    for level in LEVELS:
        stable = [k[4] for k, runs in scan.items() if level in runs and runs[level]["converged"]]
        failed = [k[4] for k, runs in scan.items() if level in runs and not runs[level]["converged"]]
        limits[level] = dict(largest_stable=max(stable) if stable else None,
                             smallest_failed=min(failed) if failed else None)
    table([dict(level=level, **limits[level]) for level in LEVELS],
          [("level", "level"), ("largest_stable", "largest converged alpha"), ("smallest_failed", "smallest failed alpha")],
          "Stability limit of the relaxation alpha (aungier_13_57, station)")
    say(f"Sign check (very_fine, frozen density, tan(beta_out) +- 1e-4): d(W0 - W1)/d tan(beta_out) = "
        f"{sign['d_W0_minus_W1_d_tan']:.4f} m/s; d Gamma_exit / d tan = {sign['d_gamma_exit_d_tan']:.4f}, "
        f"d Gamma_K / d tan = {sign['d_gamma_K_d_tan']:.4f} (m/s x chord).")
    say()
    save("phase3_limits.json", limits)


# --- Phase 4: the CSV deliverables and the figures ---------------------------------------------------------------

INK, INK2, MUTED, SURFACE, GRIDLINE = "#0b0b0b", "#52514e", "#8a8986", "#fcfcfb", "#e6e5e1"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300")     # validated, fixed order
MARKS = ("o", "s", "^", "D", "v", "P")
LEVEL_RAMP = ("#9ecae1", "#6baed6", "#3182bd", "#08519c", "#08306b")          # one hue, light (coarse) -> dark (fine)


def style(ax, title: str | None = None) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, color=GRIDLINE, linewidth=0.6)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, fontsize=9.5, color=INK, loc="left")
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)


def all_runs() -> list:
    p1, p3 = load("phase1.json"), load("phase3.json")
    seen, runs = set(), []
    for r in p1["results"] + p3["results"]:
        key = (variant_key(r), r["level"])
        if key not in seen:
            seen.add(key)
            runs.append(r)
    return runs


def read_dunavant() -> dict:
    rows = list(csv.reader(open(ROOT / DATA, encoding="utf-8-sig")))
    names = [n for n in rows[0] if n]
    series = {n: [] for n in names}
    for r in rows[2:]:
        for k, n in enumerate(names):
            pair = r[2 * k:2 * k + 2]
            if len(pair) == 2 and pair[0].strip() and pair[1].strip():
                series[n].append((float(pair[0]), float(pair[1])))
    return {n: np.array(sorted(p)) for n, p in series.items()}


def beta_axis(ax, title: str) -> None:
    style(ax, title)
    ax.set_xlabel("dm / c  (very_coarse at the left, very_fine at the right)")
    ax.set_ylabel("beta_out  [deg]")
    ax.invert_xaxis()
    for value, (name, _) in zip(TARGETS.values(), TARGETS.items()):
        ax.axhline(value, color=MUTED, linewidth=1.0, linestyle=":")
        ax.annotate(f"{name}: {value:g} deg", (1.0, value), xycoords=("axes fraction", "data"), xytext=(-4, 3),
                    textcoords="offset points", ha="right", fontsize=7.5, color=INK2)


def plot_variants(ax, variants: list, dm: list) -> None:
    for k, (label, betas, dashed) in enumerate(variants):
        color, mark = SERIES[k % len(SERIES)], MARKS[k % len(MARKS)]
        xs = [x for x, b in zip(dm, betas) if b is not None]
        ys = [b for b in betas if b is not None]
        ax.plot(xs, ys, color=color, linewidth=2.0, linestyle="--" if dashed else "-", marker=mark, markersize=6,
                markerfacecolor=SURFACE if dashed else color, markeredgewidth=1.4, label=label)
    ax.legend(fontsize=7.5, frameon=False, labelcolor=INK2, loc="best")


def phase4() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p1, p2, p3 = load("phase1.json"), load("phase2.json"), load("phase3.json")
    level_rows_cache.update(p1["levels"])
    dm = [level_rows_cache[level]["dm_over_c"] for level in LEVELS]
    runs = all_runs()
    variants = by_variant(runs)
    # CSVs
    matrix, history = [], []
    for key, by_level in variants.items():
        method, where, delta, interpolation, relax = key
        first = next(iter(by_level.values()))
        row = dict(variant=first["label"], method=method, kutta_eval=where, delta_chords=delta,
                   interpolation=interpolation, relaxation=relax, phase=first["job"].get("phase"),
                   supplementary=bool(first["job"].get("supplementary")), scan=bool(first["job"].get("scan")))
        for level in LEVELS:
            r = by_level.get(level)
            row[f"beta_out_{level}"] = r.get("beta_out_deg") if r and r["converged"] else None
            row[f"passes_{level}"] = r.get("passes") if r else None
            row[f"status_{level}"] = r["status"] if r else "not run"
        matrix.append(row)
        for level, r in by_level.items():
            for h in r.get("history", []):
                history.append(dict(variant=first["label"], level=level, pass_index=h["n"], final=h["final"],
                                    kutta_residual=h["kutta_residual"], eps_k=h["eps_k"], beta_out_deg=h["beta_out_deg"],
                                    density_change=h["density_change"], velocity_change=h["velocity_change"]))
    write_csv("beta_out_matrix.csv", matrix)
    write_csv("convergence_history.csv", history)
    # figure 1: Phase 1
    fig, ax = plt.subplots(figsize=(8.2, 4.8), facecolor=SURFACE)
    beta_axis(ax, "Phase 1: beta_out, Kutta at station K and at m* = m_TE - delta (lambda_K 0.7)")
    hue = {None: 0, **{d: k + 1 for k, d in enumerate(DELTAS)}}          # one hue per place, fixed order
    for key, by_level in sorted(variants.items(), key=lambda kv: (kv[0][2] or 0.0, kv[0][3] or "")):
        method, where, delta, interpolation, relax = key
        if method != "current" or relax != SOLVER["kutta_relaxation"]:
            continue
        dashed = interpolation == "linear"
        name = ("station K (production)" if where == "station" else
                f"m* = m_TE - {100 * delta:g} % c, {interpolation}" + ("  [not every level]" if delta in p1["partial"] else ""))
        betas = [by_level[level]["beta_out_deg"] if level in by_level and by_level[level]["converged"] else None
                 for level in LEVELS]
        k = hue[delta]
        xs = [x for x, b in zip(dm, betas) if b is not None]
        ax.plot(xs, [b for b in betas if b is not None], color=SERIES[k], linewidth=2.0, linestyle="--" if dashed else "-",
                marker=MARKS[k], markersize=6, markerfacecolor=SURFACE if dashed else SERIES[k], markeredgewidth=1.4,
                label=name)
    ax.legend(fontsize=7.5, frameon=False, labelcolor=INK2, loc="lower left")
    ax.annotate("solid: cubic in m, dashed: linear; very_fine at 5 % c did not converge in 600 passes",
                (0.0, -0.17), xycoords="axes fraction", fontsize=7.5, color=INK2)
    fig.tight_layout()
    fig.savefig(OUT / "beta_out_vs_level_phase1.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)
    # figure 2: Phase 3 at alpha 1 (current at its lambda_K 0.7)
    fig, ax = plt.subplots(figsize=(8.2, 4.8), facecolor=SURFACE)
    beta_axis(ax, f"Phase 3: beta_out against the grid by Kutta method; solid station K, dashed m* = m_TE - "
                  f"{100 * p3['best_delta']:g} % c")
    rows3 = []
    for method in METHODS:
        for where in ("station", "fixed_m"):
            relax = SOLVER["kutta_relaxation"] if method == "current" else 1.0
            key = (method, where, p3["best_delta"] if where == "fixed_m" else None,
                   "cubic" if where == "fixed_m" else None, relax)
            by_level = variants.get(key, {})
            betas = [by_level[level]["beta_out_deg"] if level in by_level and by_level[level]["converged"] else None
                     for level in LEVELS]
            rows3.append((f"{method} ({where}), relax {relax:g}", betas, where == "fixed_m", method))
    for k, (name, betas, dashed, method) in enumerate(rows3):
        color, mark = SERIES[METHODS.index(method)], MARKS[METHODS.index(method)]
        xs = [x for x, b in zip(dm, betas) if b is not None]
        ax.plot(xs, [b for b in betas if b is not None], color=color, linewidth=2.0, linestyle="--" if dashed else "-",
                marker=mark, markersize=6, markerfacecolor=SURFACE if dashed else color, markeredgewidth=1.4, label=name)
    ax.legend(fontsize=7.5, frameon=False, labelcolor=INK2, loc="lower left")
    ax.annotate("current and aungier_13_57 coincide at every level (the orange curves cover the blue ones)",
                (0.0, -0.17), xycoords="axes fraction", fontsize=7.5, color=INK2)
    fig.tight_layout()
    fig.savefig(OUT / "beta_out_vs_level_phase3.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)
    # figure 3: the relaxation scan
    fig, ax = plt.subplots(figsize=(8.2, 4.6), facecolor=SURFACE)
    style(ax, "Phase 3: passes to converge against the relaxation alpha (aungier_13_57, station); x = failed")
    for k, level in enumerate(LEVELS):
        pts = sorted((key[4], runs_[level]) for key, runs_ in variants.items()
                     if key[:2] == ("aungier_13_57", "station") and level in runs_)
        good = [(a, r["passes"]) for a, r in pts if r["converged"]]
        bad = [a for a, r in pts if not r["converged"]]
        if good:
            ax.plot(*zip(*good), color=LEVEL_RAMP[k], linewidth=2.0, marker="o", markersize=5, label=level)
        if bad:
            ax.plot(bad, [SOLVER["max_kutta_iterations"]] * len(bad), linestyle="none", marker="x", markersize=8,
                    color=LEVEL_RAMP[k], markeredgewidth=1.8)
    ax.set_xscale("log")
    ax.set_xlabel("alpha (relaxation of Eq. 13-59)")
    ax.set_ylabel("passes")
    ax.legend(fontsize=7.5, frameon=False, labelcolor=INK2, title="level", title_fontsize=7.5)
    fig.tight_layout()
    fig.savefig(OUT / "relaxation_scan.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)
    # figure 4: the Stokes check
    rows = p2["rows"]
    x = np.array([r["x_c_ss_mid"] for r in rows])
    lhs, rc, rs = (np.array([r[k] for r in rows]) for k in ("lhs", "rhs_cos", "rhs_sin"))
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2), facecolor=SURFACE)
    span = 3.0 * np.max(np.abs(lhs))
    for ax, sel, title in ((axes[0, 0], slice(None), "all blade elements"),
                           (axes[0, 1], slice(-15, None), "the last 15 elements (TE)"),
                           (axes[1, 0], slice(0, 15), "the first 15 elements (LE)")):
        style(ax, f"Stokes loop, very_fine: {title}")
        ax.plot(x[sel], lhs[sel], color=INK, linewidth=2.0, label="LHS  Gamma_k - Gamma_(k-1)")
        ax.plot(x[sel], rc[sel], color=SERIES[0], linewidth=2.0, linestyle="--", marker="o", markersize=4,
                label="RHS_cos (this frame)")
        ax.plot(x[sel], rs[sel], color=SERIES[1], linewidth=1.4, linestyle=":", marker="s", markersize=4,
                label="RHS_sin (unconverted)")
        ax.set_ylim(-span, span)
        ax.set_xlabel("x/c of the element (suction wall)")
        ax.set_ylabel("m/s x chord")
        ax.legend(fontsize=7, frameon=False, labelcolor=INK2)
    ax = axes[1, 1]
    style(ax, "|LHS - RHS| per element (log)")
    ax.semilogy(x, np.abs(lhs - rc), color=SERIES[0], linewidth=1.6, marker="o", markersize=3, label="|LHS - RHS_cos|")
    ax.semilogy(x, np.abs(lhs - rs), color=SERIES[1], linewidth=1.2, marker="s", markersize=3, label="|LHS - RHS_sin|")
    ax.semilogy(x, np.abs(lhs), color=MUTED, linewidth=1.0, label="|LHS|")
    ax.set_xlabel("x/c of the element (suction wall)")
    ax.legend(fontsize=7, frameon=False, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(OUT / "stokes_check.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)
    # figure 5: residual histories on very_fine (alpha-1 variants)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), facecolor=SURFACE)
    for k, (name, _, dashed, method) in enumerate(rows3):
        where = "fixed_m" if dashed else "station"
        relax = SOLVER["kutta_relaxation"] if method == "current" else 1.0
        key = (method, where, p3["best_delta"] if dashed else None, "cubic" if dashed else None, relax)
        r = variants.get(key, {}).get("very_fine")
        if not r or not r.get("history"):
            continue
        h = r["history"]
        n = [e["n"] for e in h]
        color = SERIES[METHODS.index(method)]
        ls = "--" if dashed else "-"
        axes[0].semilogy(n, [e["eps_k"] for e in h], color=color, linestyle=ls, linewidth=1.6, label=name)
        axes[1].plot(n, [e["beta_out_deg"] for e in h], color=color, linestyle=ls, linewidth=1.6, label=name)
        axes[2].semilogy(n, [e["density_change"] for e in h], color=color, linestyle=ls, linewidth=1.6, label=name)
    for ax, title, ylabel in zip(axes, ("Kutta residual eps_K (the driven one)", "beta_out per pass",
                                        "density change per pass"), ("eps_K", "deg", "max|d rho|/max|rho|")):
        style(ax, f"very_fine: {title}")
        ax.set_xlabel("pass")
        ax.set_ylabel(ylabel)
    axes[0].legend(fontsize=6.8, frameon=False, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(OUT / "residual_histories.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)
    # figure 6: surface Mach on very_fine against Dunavant and Aungier
    data = read_dunavant()
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.6), facecolor=SURFACE, sharex=True, sharey=True)
    for ax, (name, _, dashed, method) in zip(axes.flat, rows3):
        where = "fixed_m" if dashed else "station"
        relax = SOLVER["kutta_relaxation"] if method == "current" else 1.0
        key = (method, where, p3["best_delta"] if dashed else None, "cubic" if dashed else None, relax)
        r = variants.get(key, {}).get("very_fine")
        style(ax, name)
        for label, pts in data.items():
            pressure = label.upper().startswith("PS")
            if "xperiment" in label:
                ax.plot(pts[:, 0], pts[:, 1], linestyle="none", marker="s" if pressure else "o", markersize=6,
                        markerfacecolor=SURFACE, markeredgecolor=INK, markeredgewidth=1.2,
                        label=f"Dunavant, {'PS' if pressure else 'SS'}")
            else:
                ax.plot(pts[:, 0], pts[:, 1], color=MUTED, linewidth=1.2, linestyle="--" if pressure else "-.",
                        label=f"Aungier, {'PS' if pressure else 'SS'}")
        if r and r.get("surface"):
            s = r["surface"]
            beta = fmt(r.get("beta_out_deg"), 3) if r["converged"] else r["status"]
            ax.plot(s["x_c_ss"], s["mach_ss"], color=SERIES[0], linewidth=2.0, label=f"this solver, SS (beta_out {beta})")
            ax.plot(s["x_c_ps"], s["mach_ps"], color=SERIES[1], linewidth=2.0, label="this solver, PS")
        ax.set_xlabel("x/c")
        ax.set_ylabel("surface Mach number")
        ax.legend(fontsize=6.5, frameon=False, labelcolor=INK2, loc="lower center")
    fig.suptitle("Surface Mach number, very_fine, M1 0.65: each Kutta variant against Dunavant et al. (1955) and "
                 "Aungier (2003) (validation/naca65_12a10_10/Dunavant_Aungier_Fix.csv)", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "surface_mach_very_fine.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {OUT / 'beta_out_matrix.csv'}, convergence_history.csv and six figures")


# --- main -------------------------------------------------------------------------------------------------------

PHASES = {"phase0": phase0, "phase1": phase1, "phase2": phase2, "phase3": phase3, "phase4": phase4}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("phase", choices=sorted(PHASES))
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    PHASES[args.phase]()


if __name__ == "__main__":
    main()
