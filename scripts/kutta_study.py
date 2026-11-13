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
N_WORKERS = 16
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


# --- main -------------------------------------------------------------------------------------------------------

PHASES = {"phase0": phase0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("phase", choices=sorted(PHASES))
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    PHASES[args.phase]()


if __name__ == "__main__":
    main()
