"""
naca65_parameter_envelope.py
============================
The parameter envelope of geometry.py's NACA 65 generator, measured (issue #7). Nothing in
src/ is changed or tuned here: every number is what make_cascade / grid.build_grid return
for a stated spec, and every check of this script is computed by the script itself.

    python scripts/naca65_parameter_envelope.py                  # all parts, about 30 min serial
    python scripts/naca65_parameter_envelope.py --part profiles  # 1a only
    python scripts/naca65_parameter_envelope.py --part grids     # 1a + 1c (grids need the built blades)
    python scripts/naca65_parameter_envelope.py --part cascade history
    python scripts/naca65_parameter_envelope.py --quick          # a few blades, for the test

Each run writes a NEW folder outputs/naca65_parameter_envelope/<YYYYmmdd-HHMMSS>/ (a suffix -2,
-3 ... if that name exists; nothing is overwritten) with run.json (commit, dirty flag, command
line, versions, date, run time, every setting below), log.txt, summary.json, the CSVs, the PNGs
and captions.json.

Parts
 1a profiles  Solidity 1.0, stagger 33 deg, the default CascadeSpec otherwise (C2,
              quintic_natural_sqrt, n_camber_stations 101): cl0 0.0 .. 2.8 step 0.2 x t_max_over_c
              0.02 .. 0.15 step 0.01 x both profile variants (420 blades). Each blade is
              "built", "built_warning" (a geometry.GeometryWarning was emitted; its text is kept)
              or "refused" (GeometryError, ValueError, NotImplementedError: class and message);
              any other exception is "unexpected_error", recorded, never hidden. Per built
              blade: the LE fairness R_LE^3 int kappa_s^2 ds of both surfaces (metadata; refused
              above geometry.LE_FAIRNESS_LIMIT), the C2 deviations from C1, the LE-zone
              deviations, the checked-range flag, max thickness, R_LE, the equivalent-arc angles
              [Aungier 2003 Eq. 4-6], the geometric throat o/s and the correlation o/s
              [Aungier 2003 Eqs. 4-30, 4-31], the build time, and the independent shape check
              below. envelope_profiles.csv; a status map and a fairness map per variant.
 1c grids     For every built blade of 1a, grid.build_grid(cascade, "medium") at the inlet angle
              GRID_INLET_ANGLE_DEG (exit angle None = kappa_te), and again at the zero-incidence
              inlet angle kappa_le: success, or the exception class and message; n_m x n_eta.
              Columns added to envelope_profiles.csv; a status map per variant.
 1b cascade   For CASCADE_BLADES (kovach_sandercock): at every stagger -30 .. 70 deg step 5 the
              solidities SOLIDITY_SCAN are tried, then the largest accepted solidity is found by
              bisection between the last accepted and the first refused scan value, to a relative
              width SOLIDITY_RTOL; the refusal message at the bound is kept. If the smallest scan
              solidity is refused, the lower edge is bisected the same way. Near closure the
              geometric throat is round-off small and its check refuses only an exact 0.0 (README,
              "Continuous passage check (2026-11-07)"), so near the bound acceptance is decided by
              round-off; two spot checks at 1e-3 inside and outside the bound are recorded. The
              grid's blade-angle line kappa_le = stagger + chi_le < 90 deg (grid.py
              _check_blade_angle, called by make_passage_boundaries; kappa_te > -90 deg likewise)
              is recorded per blade, and grid.build_grid(cascade, "medium") is tried at each
              stagger at solidity GRID_CHECK_SOLIDITY. envelope_cascade.csv and a figure.
 1d history   Figures for the report that need "pchip_sqrt" and "C1" (issue #6 removes both):
              (i) the 65-(12A10)10 (cl0 1.2, t 0.10, sigma 1, 33 deg), both variants: curvature
              against x/c of the C2 curves with thickness_reconstruction "pchip_sqrt" and
              "quintic_natural_sqrt", both surfaces, the table stations marked; the half-thickness
              second derivative h''(station) of both reconstructions, rebuilt here with SciPy from
              geometry.base_tables() and checked against geometry's own interpolants; the
              sawtooth measures defined below. (ii) the legacy C1 LE (surface_continuity "C1",
              thickness_reconstruction "quintic_natural_sqrt", so only the LE differs) against the
              robust C2 LE on HISTORY_LE_BLADES: shape close-up, tangent angle and curvature
              against arc length from the apex, and the reverse turning defined below.

Definitions (ours, for this report; none is Aungier's)
 - Tangent angle turned: theta(s) = sense * unwrap(atan2(T_y, T_x)), T the unit tangent (curve:
   X'(u); polyline: the segment directions), sense = sign of the apex curvature (geometry.py's
   convention: suction -1, pressure +1), so the turning of the nose circle is positive.
 - Max reverse turn: max over the zone of (running max of theta - theta), the largest amount the
   tangent turns back below what it had reached (0 for a monotone turn).
 - Summed reverse turning: sum over consecutive samples of max(0, -(theta_k+1 - theta_k)); on a
   polyline that is the summed negative turning at its vertices.
 - LE zone: on a C2 curve u in [0, ss/ps_le_anchor_u] (metadata); on the C1 polyline and the C1
   analytic surface, arc length from the apex up to the same anchor in C1 reference arc length
   (metadata ss/ps_le_anchor_over_c of the C2 build of the same blade); the polyline's arc length
   is its cumulative chord length.
 - Curvature sign changes per surface: the signs of kappa at SHAPE_POINTS equally spaced u on
   [0, 1], points with |kappa| <= KAPPA_SIGN_THRESHOLD dropped, changes between neighbours counted;
   the x/c of each change is kept. In the LE zone (SHAPE_POINTS points on [0, anchor u]) nothing is
   dropped.
 - Sawtooth: kappa(x) of each C2 curve sampled at SHAPE_POINTS u, interpolated onto
   SAWTOOTH_X_POINTS equally spaced x/c in a window; (a) rms_difference = RMS of
   kappa_pchip - kappa_quintic, (b) rms_detrended = RMS of kappa minus its own least-squares
   polynomial of degree SAWTOOTH_TREND_DEGREE in x/c over the window. Windows SAWTOOTH_WINDOWS.
"""

# ============================================================
# SETTINGS
# ============================================================

OUTPUT_DIR = "outputs/naca65_parameter_envelope"

SOLIDITY = 1.0                          # 1a: the profile sweep's cascade
STAGGER_DEG = 33.0
CL0_STEP, CL0_STEPS = 0.2, range(0, 15)          # cl0 = round(k * 0.2, 10): 0.0 .. 2.8
T_STEP, T_STEPS = 0.01, range(2, 16)             # t/c = round(k * 0.01, 10): 0.02 .. 0.15
VARIANTS = ("kovach_sandercock", "emery_base")
SHAPE_POINTS = 20001                    # u samples of the independent shape check
KAPPA_SIGN_THRESHOLD = 1e-3             # 1/c: |kappa| at or below this is ignored by the sign-change count

GRID_LEVEL = "medium"
GRID_INLET_ANGLE_DEG = 45.0             # the reference case's inlet angle (scripts/analyze_grid_family.py)

CASCADE_BLADES = ((0.4, 0.04), (1.2, 0.10), (1.6, 0.10))
CASCADE_VARIANT = "kovach_sandercock"
CASCADE_STAGGERS_DEG = tuple(float(s) for s in range(-30, 75, 5))
SOLIDITY_SCAN = (0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0)
SOLIDITY_RTOL = 1e-6                    # bisection stops when (hi - lo) <= SOLIDITY_RTOL * hi
SPOT_CHECK = 1e-3                       # relative offsets of the spot checks inside / outside the bound
GRID_CHECK_SOLIDITY = 1.0               # 1b: solidity of the build_grid check at each stagger

HISTORY_BLADE = (1.2, 0.10)             # the 65-(12A10)10
HISTORY_LE_BLADES = ((1.2, 0.10), (1.6, 0.10), (1.2, 0.04))
CURVATURE_WINDOW = (0.05, 0.95)         # x/c of the curvature figure
SAWTOOTH_WINDOWS = ((0.20, 0.85), (0.20, 0.60))
SAWTOOTH_X_POINTS = 6501
SAWTOOTH_TREND_DEGREE = 6
H2_WINDOW = (0.05, 0.95)                # station window of the h'' figure
INTERPOLANT_TOL = 1e-15                 # c: our SciPy interpolants against geometry's (half-thickness <= 0.05 c)
LE_CLOSEUP_X = 0.02                     # x/c of the LE shape close-up
LE_PLOT_ANCHORS = 2.0                   # LE arc-length plots run to this many C2 anchor lengths
C1_REFERENCE_POINTS = 2001              # per piece, the C1 analytic surface samples

QUICK = dict(cl0=(1.2, 2.0), t=(0.03, 0.10), variants=("kovach_sandercock",), cascade_blades=((1.2, 0.10),),
             staggers=(33.0,), scan=(0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 50.0), rtol=1e-3, history_le=((1.2, 0.10),),
             history_variants=("kovach_sandercock",), shape_points=4001)

DPI = 200
FIG_WIDTH_IN = 6.3                      # about 16 cm
FONT_SIZE = 8.5
INK = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a"}          # categorical slots 1-3
STATUS_INK = {"built": "#0ca30c", "built_warning": "#fab219", "refused": "#d03b3b", "unexpected_error": "#6250d6"}
MUTED = "#8a8985"
GRID_INK = "#e4e3df"

# ============================================================

import argparse                                                                  # noqa: E402
import csv                                                                       # noqa: E402
import json                                                                      # noqa: E402
import math                                                                      # noqa: E402
import platform                                                                  # noqa: E402
import re                                                                        # noqa: E402
import subprocess                                                                # noqa: E402
import sys                                                                       # noqa: E402
import time                                                                      # noqa: E402
import warnings                                                                  # noqa: E402
from datetime import datetime                                                    # noqa: E402
from pathlib import Path                                                         # noqa: E402

import numpy as np                                                               # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from turbolenza.b2b import geometry as geo                                       # noqa: E402
from turbolenza.b2b import grid as gr                                            # noqa: E402

REFUSALS = (geo.GeometryError, ValueError, NotImplementedError)
# Refusal kinds, matched in this order on the message: (kind, map code, text in the message)
REFUSAL_KINDS = (
    ("le_fair_too_steep", "S", "the LE fair is too steep"),
    ("no_le_fair_in_range", "N", "no LE fair lands on the source within"),
    ("no_le_fair_at_anchor", "A", "no LE fair lands on the source at"),
    ("blades_overlap", "O", "neighboring blades overlap"),
    ("throat_zero", "Z", "throat opening"),
    ("throat_not_converged", "K", "throat refinement did not converge"),
    ("blades_closer_than_throat", "P", "come closer"),
    ("te_circle_cut", "T", "cuts into the TE circle"),
    ("construction_failed", "C", "failed for cl0="),
)
LINES: list[str] = []


# --- small helpers ------------------------------------------------------------------------------------------

def say(text: str = "") -> None:
    print(text, flush=True)
    LINES.append(text)


def exact_values(step: float, steps: range) -> tuple[float, ...]:
    """round(k * step, 10): reproducible decimals (0.6, not 0.6000000000000001)."""
    return tuple(round(k * step, 10) for k in steps)


def git(*args: str) -> str | None:
    try:
        out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def new_run_folder(root: Path, stamp: str | None = None) -> Path:
    """A folder root/<stamp> that did not exist before (stamp-2, stamp-3, ... if it did)."""
    stamp = stamp or datetime.now().strftime("%Y%m%d-%H%M%S")
    root.mkdir(parents=True, exist_ok=True)
    k = 1
    while True:
        folder = root / (stamp if k == 1 else f"{stamp}-{k}")
        try:
            folder.mkdir()
            return folder
        except FileExistsError:
            k += 1


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: (json.dumps(v) if isinstance(v, (list, tuple, dict)) else v) for k, v in r.items()})


def write_json(path: Path, data) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, default=_json_default)


def _json_default(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def one_line(message: str) -> str:
    return " | ".join(part.strip() for part in str(message).splitlines() if part.strip())


def refusal_kind(message: str) -> tuple[str, str]:
    for kind, code, text in REFUSAL_KINDS:
        if text in message:
            return kind, code
    return "other", "X"


def message_pattern(message: str) -> str:
    """The message with its numbers replaced by #, to group refusals."""
    return re.sub(r"[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?", "#", message)


def spec_for(cl0: float, t: float, variant: str, solidity: float = SOLIDITY, stagger_deg: float = STAGGER_DEG,
             **changes) -> geo.CascadeSpec:
    """The default CascadeSpec (C2, quintic_natural_sqrt, A10, 101 stations) at these values."""
    return geo.CascadeSpec(blade_type="NACA65", solidity=solidity, stagger_deg=stagger_deg, cl0=cl0, t_max_over_c=t,
                           profile_variant=variant, **changes)


# --- classification -----------------------------------------------------------------------------------------

def classify(spec: geo.CascadeSpec) -> tuple[dict, geo.Cascade | None]:
    """make_cascade(spec), classified: "built", "built_warning" (a GeometryWarning was emitted),
    "refused" (GeometryError, ValueError, NotImplementedError) or "unexpected_error" (anything
    else). Every warning is caught, GeometryWarning texts and others apart. build_seconds is
    perf_counter around the call (geometry.py's lru_caches can shorten a repeated build)."""
    cascade, error = None, None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        started = time.perf_counter()
        try:
            cascade = geo.make_cascade(spec)
        except Exception as exc:            # classified below; nothing is swallowed silently
            error = exc
        seconds = time.perf_counter() - started
    geometry_warnings = [str(w.message) for w in caught if issubclass(w.category, geo.GeometryWarning)]
    other_warnings = [f"{w.category.__name__}: {w.message}" for w in caught
                      if not issubclass(w.category, geo.GeometryWarning)]
    record = dict(status="", geometry_warning=" || ".join(geometry_warnings),
                  other_warnings=" || ".join(other_warnings), error_class="", error_message="", refusal_kind="",
                  build_seconds=seconds)
    if error is None:
        record["status"] = "built_warning" if geometry_warnings else "built"
    else:
        record["status"] = "refused" if isinstance(error, REFUSALS) else "unexpected_error"
        record["error_class"] = type(error).__name__
        record["error_message"] = one_line(error)
        if isinstance(error, geo.GeometryError):
            record["refusal_kind"] = refusal_kind(str(error))[0]
        elif isinstance(error, NotImplementedError):
            record["refusal_kind"] = "not_implemented"
        else:
            record["refusal_kind"] = "input" if isinstance(error, ValueError) else "unexpected"
    return record, cascade


def built(status: str) -> bool:
    return status in ("built", "built_warning")


# --- independent shape check ----------------------------------------------------------------------------------

def tangent_angle(vectors: np.ndarray, sense: float) -> np.ndarray:
    """sense * unwrap(atan2(v_y, v_x)) of tangent (or segment) vectors, radians."""
    vectors = np.asarray(vectors, dtype=float)
    return sense * np.unwrap(np.arctan2(vectors[:, 1], vectors[:, 0]))


def turning_metrics(angle: np.ndarray) -> dict:
    """Max reverse turn (largest drop below the running maximum) and summed reverse turning (sum of
    the negative steps) of a sense-oriented tangent angle, degrees; and the number of negative steps."""
    angle = np.asarray(angle, dtype=float)
    steps = np.diff(angle)
    return dict(max_reverse_turn_deg=math.degrees(float(np.max(np.maximum.accumulate(angle) - angle))),
                summed_reverse_turn_deg=math.degrees(float(np.sum(np.maximum(0.0, -steps)))),
                reverse_steps=int(np.sum(steps < 0.0)))


def sign_changes(kappa: np.ndarray, threshold: float) -> np.ndarray:
    """Indices (into kappa) of the first sample after each sign change, samples with |kappa| <=
    threshold dropped."""
    keep = np.flatnonzero(np.abs(kappa) > threshold)
    signs = np.sign(kappa[keep])
    return keep[1:][signs[1:] != signs[:-1]]


def curve_shape_check(curve: geo.ParametricBladeCurve, anchor_u: float, n: int = SHAPE_POINTS,
                      threshold: float = KAPPA_SIGN_THRESHOLD) -> dict:
    """The script's own check of one C2 surface curve (blade frame): curvature sign changes over the
    whole surface (|kappa| <= threshold ignored) with their x/c, and over the LE zone u in
    [0, anchor_u] the sign changes (none ignored), the minimum of sense * kappa and the reverse
    turning of the tangent angle (module docstring, "Definitions"). Uses only the curve's own
    evaluate / derivative / curvature, not geometry's fairness metadata."""
    u = np.linspace(0.0, 1.0, n)
    kappa = curve.curvature(u)
    changes = sign_changes(kappa, threshold)
    x = curve.evaluate(u)[:, 0]
    zone = np.linspace(0.0, float(anchor_u), n)
    kappa_zone = curve.curvature(zone)
    sense = float(np.sign(kappa_zone[0]))
    turning = turning_metrics(tangent_angle(curve.derivative(zone, 1), sense))
    return dict(sign_changes=int(len(changes)), sign_change_x=[round(float(v), 4) for v in x[changes]],
                apex_sense=sense, le_sign_changes=int(len(sign_changes(kappa_zone, 0.0))),
                le_min_sense_kappa=float(np.min(sense * kappa_zone)),
                le_max_reverse_turn_deg=turning["max_reverse_turn_deg"],
                le_summed_reverse_turn_deg=turning["summed_reverse_turn_deg"])


def polyline_turning(points: np.ndarray, sense: float, zone_length: float) -> dict:
    """Turning metrics of a polyline's segment directions over the segments that end within
    zone_length of cumulative chord length from its first point (the apex)."""
    segments = np.diff(np.asarray(points, dtype=float), axis=0)
    ends = np.cumsum(np.hypot(segments[:, 0], segments[:, 1]))
    inside = ends <= zone_length
    out = turning_metrics(tangent_angle(segments[inside], sense))
    out["segments"] = int(np.sum(inside))
    return out


# --- 1a profiles --------------------------------------------------------------------------------------------

def blade_record(cascade: geo.Cascade, shape_points: int) -> dict:
    m = cascade.metadata
    out = dict(designation=m["designation"],
               ss_le_fairness=m["ss_le_fairness"], ps_le_fairness=m["ps_le_fairness"],
               max_le_fairness=max(m["ss_le_fairness"], m["ps_le_fairness"]), le_fairness_limit=geo.LE_FAIRNESS_LIMIT,
               c2_max_deviation_over_c=m["c2_max_deviation_over_c"], c2_rms_deviation_over_c=m["c2_rms_deviation_over_c"],
               ss_le_max_deviation_over_c=m["ss_le_max_deviation_over_c"],
               ps_le_max_deviation_over_c=m["ps_le_max_deviation_over_c"],
               inside_checked_range=m["inside_checked_range"], max_thickness_over_c=m["max_thickness_over_c"],
               le_radius_over_c=m["le_radius_over_c"],
               camber_angle_deg=math.degrees(cascade.camber_angle), chi_le_deg=math.degrees(cascade.chi_le),
               kappa_le_deg=math.degrees(cascade.kappa_le), kappa_te_deg=math.degrees(cascade.kappa_te),
               throat_over_pitch=cascade.throat_over_pitch,
               throat_over_pitch_correlation=cascade.throat_over_pitch_correlation,
               ss_le_anchor_u=m["ss_le_anchor_u"], ps_le_anchor_u=m["ps_le_anchor_u"],
               ss_le_departure_deg=m["ss_le_departure_deg"], ps_le_departure_deg=m["ps_le_departure_deg"])
    curves = cascade.normalized_curves
    for prefix, curve in (("ss", curves.suction_unstaggered), ("ps", curves.pressure_unstaggered)):
        check = curve_shape_check(curve, m[f"{prefix}_le_anchor_u"], n=shape_points)
        out.update({f"{prefix}_{key}": value for key, value in check.items()})
    return out


def grid_check(cascade: geo.Cascade, inlet_angle: float) -> dict:
    """grid.build_grid(cascade, GRID_LEVEL): "ok" with the grid size, or "failed" with the exception
    class and message (GridError / ValueError), or "unexpected_error"."""
    started = time.perf_counter()
    try:
        standard = gr.build_grid(cascade, GRID_LEVEL, inlet_angle=inlet_angle)
    except ValueError as exc:
        reasons = getattr(exc, "reasons", None)
        return dict(status="failed", error_class=type(exc).__name__, error_message=one_line(exc),
                    reasons=dict(reasons) if reasons else "", n_m="", n_eta="", dm_over_chord="",
                    seconds=time.perf_counter() - started)
    except Exception as exc:                # recorded, not swallowed: its own status
        return dict(status="unexpected_error", error_class=type(exc).__name__, error_message=one_line(exc),
                    reasons="", n_m="", n_eta="", dm_over_chord="", seconds=time.perf_counter() - started)
    info = standard.info
    return dict(status="ok", error_class="", error_message="", reasons="", n_m=info.n_m, n_eta=info.n_eta,
                dm_over_chord=info.dm_over_chord, seconds=time.perf_counter() - started)


def profile_grid(quick: bool) -> tuple[tuple[float, ...], tuple[float, ...], tuple[str, ...]]:
    if quick:
        return QUICK["cl0"], QUICK["t"], QUICK["variants"]
    return exact_values(CL0_STEP, CL0_STEPS), exact_values(T_STEP, T_STEPS), VARIANTS


def run_profiles(quick: bool, with_grids: bool) -> list[dict]:
    cl0s, ts, variants = profile_grid(quick)
    shape_points = QUICK["shape_points"] if quick else SHAPE_POINTS
    rows = []
    total, k = len(cl0s) * len(ts) * len(variants), 0
    say(f"1a profiles: {len(cl0s)} cl0 x {len(ts)} t/c x {len(variants)} variants = {total} blades at solidity "
        f"{SOLIDITY}, stagger {STAGGER_DEG} deg" + (f"; 1c build_grid '{GRID_LEVEL}'" if with_grids else ""))
    for variant in variants:
        for cl0 in cl0s:
            for t in ts:
                k += 1
                record, cascade = classify(spec_for(cl0, t, variant))
                row = dict(profile_variant=variant, cl0=cl0, t_max_over_c=t, solidity=SOLIDITY, stagger_deg=STAGGER_DEG,
                           **record)
                if record["status"] == "refused" and record["refusal_kind"] == "le_fair_too_steep":
                    found = re.search(r"=\s*([-+0-9.eE]+)\s*>", record["error_message"])
                    row["refused_le_fairness"] = float(found.group(1)) if found else ""
                if cascade is not None:
                    row.update(blade_record(cascade, shape_points))
                    if with_grids:
                        for tag, angle in (("grid", math.radians(GRID_INLET_ANGLE_DEG)), ("grid_kappa_le", cascade.kappa_le)):
                            row.update({f"{tag}_{key}": value for key, value in grid_check(cascade, angle).items()})
                rows.append(row)
                grid_text = f" grid {row.get('grid_status', '-')}" if with_grids and cascade is not None else ""
                say(f"  [{k:3d}/{total}] {variant:17s} cl0 {cl0:4.2f} t {t:4.2f}: {record['status']:16s} "
                    f"{record['build_seconds']:5.2f} s{grid_text}"
                    + (f"  {record['error_class']}: {record['error_message'][:90]}" if record["error_message"] else ""))
    return rows


def profile_summary(rows: list[dict], with_grids: bool) -> dict:
    """Counts, boundaries, grouped refusals and ranges, per variant, from the rows only."""
    out = {}
    for variant in dict.fromkeys(r["profile_variant"] for r in rows):
        mine = [r for r in rows if r["profile_variant"] == variant]
        cl0s = sorted({r["cl0"] for r in mine})
        ts = sorted({r["t_max_over_c"] for r in mine})
        ok = {(r["cl0"], r["t_max_over_c"]): built(r["status"]) for r in mine}
        counts = {s: sum(r["status"] == s for r in mine) for s in ("built", "built_warning", "refused", "unexpected_error")}
        inside = [r for r in mine if r["cl0"] <= geo.CHECKED_CL0_MAX and r["t_max_over_c"] <= geo.CHECKED_T_MAX]
        counts_inside = {s: sum(r["status"] == s for r in inside) for s in counts}
        smallest_t = {}
        for c in cl0s:
            good = [t for t in ts if ok[(c, t)]]
            holes = [t for t in ts if good and t > min(good) and not ok[(c, t)]]
            smallest_t[str(c)] = dict(smallest_t_built=min(good) if good else None, refused_above_it=holes)
        largest_cl0 = {}
        for t in ts:
            good = [c for c in cl0s if ok[(c, t)]]
            holes = [c for c in cl0s if good and c < max(good) and not ok[(c, t)]]
            largest_cl0[str(t)] = dict(largest_cl0_built=max(good) if good else None, refused_below_it=holes)
        groups: dict[str, dict] = {}
        for r in mine:
            if r["status"] in ("refused", "unexpected_error"):
                key = f"{r['error_class']}: {message_pattern(r['error_message'])}"
                g = groups.setdefault(key, dict(kind=r["refusal_kind"], count=0, cells=[], example=r["error_message"]))
                g["count"] += 1
                g["cells"].append([r["cl0"], r["t_max_over_c"]])
        made = [r for r in mine if built(r["status"])]

        def span(key, rows_=made):
            values = [float(r[key]) for r in rows_ if r.get(key) not in (None, "")]
            return [min(values), max(values)] if values else None

        def argmax(key):
            best = max(made, key=lambda r: float(r[key])) if made else None
            return [best["cl0"], best["t_max_over_c"]] if best else None

        shape = {}
        for prefix in ("ss", "ps"):
            shape[prefix] = dict(
                le_max_reverse_turn_deg_max=max((r[f"{prefix}_le_max_reverse_turn_deg"] for r in made), default=None),
                le_summed_reverse_turn_deg_max=max((r[f"{prefix}_le_summed_reverse_turn_deg"] for r in made), default=None),
                blades_with_le_sign_change=sum(r[f"{prefix}_le_sign_changes"] > 0 for r in made),
                le_min_sense_kappa_min=min((r[f"{prefix}_le_min_sense_kappa"] for r in made), default=None),
                sign_changes_histogram={str(n): sum(r[f"{prefix}_sign_changes"] == n for r in made)
                                        for n in sorted({r[f"{prefix}_sign_changes"] for r in made})},
                apex_sense_values=sorted({r[f"{prefix}_apex_sense"] for r in made}))
        summary = dict(counts=counts, counts_inside_checked_range=counts_inside, smallest_t_built_per_cl0=smallest_t,
                       largest_cl0_built_per_t=largest_cl0, refusal_groups=groups,
                       ranges_over_built=dict(
                           ss_le_fairness=span("ss_le_fairness"), ps_le_fairness=span("ps_le_fairness"),
                           max_le_fairness=span("max_le_fairness"), max_le_fairness_at=argmax("max_le_fairness"),
                           c2_max_deviation_over_c=span("c2_max_deviation_over_c"),
                           ss_le_max_deviation_over_c=span("ss_le_max_deviation_over_c"),
                           ps_le_max_deviation_over_c=span("ps_le_max_deviation_over_c"),
                           ss_le_max_deviation_at=argmax("ss_le_max_deviation_over_c"),
                           ps_le_max_deviation_at=argmax("ps_le_max_deviation_over_c"),
                           throat_over_pitch=span("throat_over_pitch"),
                           build_seconds=span("build_seconds"),
                           refused_le_fairness=span("refused_le_fairness",
                                                    [r for r in mine if r.get("refused_le_fairness") not in (None, "")])),
                       shape_check=shape)
        if with_grids:
            fails = [dict(cl0=r["cl0"], t_max_over_c=r["t_max_over_c"], error_class=r["grid_error_class"],
                          error_message=r["grid_error_message"], reasons=r["grid_reasons"])
                     for r in made if r.get("grid_status") != "ok"]
            fails_kappa = [dict(cl0=r["cl0"], t_max_over_c=r["t_max_over_c"], error_class=r["grid_kappa_le_error_class"],
                                error_message=r["grid_kappa_le_error_message"])
                           for r in made if r.get("grid_kappa_le_status") != "ok"]
            sizes = sorted({(r["grid_n_m"], r["grid_n_eta"]) for r in made if r.get("grid_status") == "ok"})
            summary["grids"] = dict(level=GRID_LEVEL, inlet_angle_deg=GRID_INLET_ANGLE_DEG,
                                    ok=sum(r.get("grid_status") == "ok" for r in made), failed=fails,
                                    ok_at_kappa_le_inlet=sum(r.get("grid_kappa_le_status") == "ok" for r in made),
                                    failed_at_kappa_le_inlet=fails_kappa,
                                    n_m_range=[min(s[0] for s in sizes), max(s[0] for s in sizes)] if sizes else None,
                                    n_eta_range=[min(s[1] for s in sizes), max(s[1] for s in sizes)] if sizes else None,
                                    seconds=span("grid_seconds"))
        out[variant] = summary
    return out


def report_profiles(summary: dict) -> None:
    for variant, s in summary.items():
        say(f"--- 1a {variant}: counts {s['counts']}; inside cl0 <= {geo.CHECKED_CL0_MAX}, t/c <= {geo.CHECKED_T_MAX}: "
            f"{s['counts_inside_checked_range']}")
        say("  smallest t/c built per cl0: " + ", ".join(
            f"{c}: {v['smallest_t_built']}" + (f" (refused above: {v['refused_above_it']})" if v["refused_above_it"] else "")
            for c, v in s["smallest_t_built_per_cl0"].items()))
        say("  largest cl0 built per t/c: " + ", ".join(
            f"{t}: {v['largest_cl0_built']}" + (f" (refused below: {v['refused_below_it']})" if v["refused_below_it"] else "")
            for t, v in s["largest_cl0_built_per_t"].items()))
        for key, g in s["refusal_groups"].items():
            say(f"  refused x{g['count']} [{g['kind']}] {key}")
        for key, value in s["ranges_over_built"].items():
            say(f"  {key}: {value}")
        for prefix, v in s["shape_check"].items():
            say(f"  shape {prefix}: {v}")
        if "grids" in s:
            g = s["grids"]
            say(f"  1c grids '{g['level']}' at inlet {g['inlet_angle_deg']} deg: ok {g['ok']}, failed {len(g['failed'])}; "
                f"at kappa_le inlet ok {g['ok_at_kappa_le_inlet']}, failed {len(g['failed_at_kappa_le_inlet'])}; "
                f"n_m {g['n_m_range']}, n_eta {g['n_eta_range']}")
            for f in g["failed"]:
                say(f"    grid failed cl0 {f['cl0']} t {f['t_max_over_c']}: {f['error_class']}: {f['error_message'][:200]}")


# --- 1b cascade ---------------------------------------------------------------------------------------------

def solidity_envelope(cl0: float, t: float, variant: str, stagger_deg: float, scan: tuple[float, ...],
                      rtol: float) -> dict:
    """The solidities make_cascade accepts at one stagger: the scan pattern, the largest accepted
    solidity by bisection on the refusal (and the smallest one if the smallest scan value is
    refused), the refusal at each bound, spot checks around it."""
    started, attempts = time.perf_counter(), 0
    cache: dict[float, tuple[dict, geo.Cascade | None]] = {}

    def attempt(sigma: float) -> tuple[dict, geo.Cascade | None]:
        nonlocal attempts
        if sigma not in cache:
            attempts += 1
            cache[sigma] = classify(spec_for(cl0, t, variant, solidity=sigma, stagger_deg=stagger_deg))
        return cache[sigma]

    results = [attempt(s) for s in scan]
    accepted = [built(r[0]["status"]) for r in results]
    pattern = "".join("A" if a else "R" for a in accepted)
    row = dict(cl0=cl0, t_max_over_c=t, profile_variant=variant, stagger_deg=stagger_deg, scan_solidities=list(scan),
               scan_pattern=pattern, solidity_search_limit=scan[-1], solidity_rtol=rtol,
               scan_refusals=[[s, r[0]["refusal_kind"]] for s, r in zip(scan, results) if not built(r[0]["status"])],
               scan_is_refuse_accept_refuse=bool(re.fullmatch(r"R*A+R*", pattern)))
    if not any(accepted):
        row.update(largest_accepted_solidity="", smallest_refused_solidity="", accepted_at_search_limit=False,
                   note="no scan solidity accepted")
    else:
        top = max(i for i, a in enumerate(accepted) if a)
        lo = scan[top]
        if top == len(scan) - 1:
            row.update(largest_accepted_solidity=lo, smallest_refused_solidity="", accepted_at_search_limit=True,
                       refusal_class_at_bound="", refusal_kind_at_bound="", refusal_message_at_bound="")
        else:
            hi = scan[top + 1]
            while hi - lo > rtol * hi:
                mid = 0.5 * (lo + hi)
                if built(attempt(mid)[0]["status"]):
                    lo = mid
                else:
                    hi = mid
            refusal = attempt(hi)[0]
            inner, outer = attempt(lo * (1.0 - SPOT_CHECK))[0], attempt(hi * (1.0 + SPOT_CHECK))[0]
            row.update(largest_accepted_solidity=lo, smallest_refused_solidity=hi, accepted_at_search_limit=False,
                       refusal_class_at_bound=refusal["error_class"], refusal_kind_at_bound=refusal["refusal_kind"],
                       refusal_message_at_bound=refusal["error_message"],
                       spot_inside_accepted=built(inner["status"]), spot_outside_refused=not built(outer["status"]))
        bound = attempt(lo)[1]
        row["throat_over_pitch_at_largest"] = bound.throat_over_pitch if bound is not None else ""
        first = accepted.index(True)
        if first > 0:                       # refused below the accepted range: bisect the lower edge too
            low, high = scan[first - 1], scan[first]
            while high - low > rtol * high:
                mid = 0.5 * (low + high)
                if built(attempt(mid)[0]["status"]):
                    high = mid
                else:
                    low = mid
            refusal = attempt(low)[0]
            row.update(smallest_accepted_solidity=high, largest_refused_below=low,
                       refusal_kind_below=refusal["refusal_kind"], refusal_message_below=refusal["error_message"])
        else:
            row.update(smallest_accepted_solidity=scan[0], largest_refused_below="", refusal_kind_below="",
                       refusal_message_below="")
    camber, chi_le, chi_te = geo.equivalent_arc_angles(cl0)
    kappa_le, kappa_te = math.radians(stagger_deg) + chi_le, math.radians(stagger_deg) - chi_te
    row.update(chi_le_deg=math.degrees(chi_le), kappa_le_deg=math.degrees(kappa_le), kappa_te_deg=math.degrees(kappa_te),
               kappa_le_stagger_ceiling_deg=90.0 - math.degrees(chi_le),
               kappa_te_stagger_floor_deg=-90.0 + math.degrees(chi_te),
               grid_blade_angles_admissible=abs(kappa_le) < 0.5 * math.pi and abs(kappa_te) < 0.5 * math.pi)
    record, cascade = attempt(GRID_CHECK_SOLIDITY)
    row["grid_check_solidity"] = GRID_CHECK_SOLIDITY
    if cascade is None:
        row.update(grid_status="geometry refused", grid_error_class="", grid_error_message="")
    else:
        assert abs(cascade.kappa_le - kappa_le) <= 1e-12, "kappa_le of the cascade is not stagger + chi_le"
        check = grid_check(cascade, math.radians(GRID_INLET_ANGLE_DEG))
        row.update(grid_status=check["status"], grid_error_class=check["error_class"],
                   grid_error_message=check["error_message"])
    row.update(builds=attempts, seconds=time.perf_counter() - started)
    return row


def run_cascade(quick: bool) -> list[dict]:
    blades = QUICK["cascade_blades"] if quick else CASCADE_BLADES
    staggers = QUICK["staggers"] if quick else CASCADE_STAGGERS_DEG
    scan = QUICK["scan"] if quick else SOLIDITY_SCAN
    rtol = QUICK["rtol"] if quick else SOLIDITY_RTOL
    say(f"1b cascade: {len(blades)} blades x {len(staggers)} staggers, scan {scan}, rtol {rtol}")
    rows = []
    for cl0, t in blades:
        for stagger in staggers:
            row = solidity_envelope(cl0, t, CASCADE_VARIANT, stagger, scan, rtol)
            rows.append(row)
            say(f"  cl0 {cl0} t {t} stagger {stagger:5.1f}: {row['scan_pattern']} largest "
                f"{row.get('largest_accepted_solidity')} [{row.get('refusal_kind_at_bound', '')}] smallest "
                f"{row.get('smallest_accepted_solidity')} kappa_le {row['kappa_le_deg']:.3f} grid {row['grid_status']} "
                f"({row['builds']} builds, {row['seconds']:.1f} s)")
    return rows


# --- 1d history ---------------------------------------------------------------------------------------------

def thickness_interpolants() -> dict:
    """The two half-thickness reconstructions rebuilt from geometry.base_tables() (corrected column,
    z = sqrt(station)): PchipInterpolator, and the quintic interpolating spline with h''' = h'''' = 0
    at both ends; each checked against geometry's own interpolant (_HALF_THICKNESS) and against the
    public naca65_base_thickness at t/c 0.10, on 20001 z in [0, 1]. Raises AssertionError beyond
    INTERPOLANT_TOL."""
    from scipy.interpolate import PchipInterpolator, make_interp_spline

    tables = geo.base_tables()
    z, h = np.sqrt(tables.x), 0.5 * tables.thickness_corrected
    ours = {"pchip_sqrt": PchipInterpolator(z, h),
            "quintic_natural_sqrt": make_interp_spline(z, h, k=5, bc_type=([(3, 0.0), (4, 0.0)], [(3, 0.0), (4, 0.0)]))}
    zz = np.linspace(0.0, 1.0, 20001)
    checks = {}
    for name, spline in ours.items():
        private = float(np.max(np.abs(spline(zz) - geo._HALF_THICKNESS[name](zz))))
        public = float(np.max(np.abs(2.0 * spline(zz) - geo.naca65_base_thickness(zz**2, 0.10, thickness_reconstruction=name))))
        at_stations = float(np.max(np.abs(spline(z) - h)))
        checks[name] = dict(max_diff_vs_geometry_half_thickness=private, max_diff_vs_naca65_base_thickness=public,
                            max_error_at_table_stations=at_stations)
        assert private <= INTERPOLANT_TOL and public <= 2 * INTERPOLANT_TOL, (name, private, public)
    return dict(interpolants=ours, z=z, h=h, checks=checks)


def h_second_derivative(spline, station: np.ndarray) -> np.ndarray:
    """d2h/dstation2 of h(station) = H(z), z = sqrt(station): (H'' - H'/z) / (4 z^2)."""
    z = np.sqrt(station)
    return (spline(z, 2) - spline(z, 1) / z) / (4.0 * z * z)


def station_surface_x(cl0: float, t: float, stations: np.ndarray, side: float) -> np.ndarray:
    """x/c of the surface point of each table station (a station is s/S along the camberline): the
    camber point x_c, moved along the camberline normal by the table half-thickness (stations <= 0.60
    only, where both variants keep the table): x = x_c - side h sin(phi)."""
    from scipy.optimize import brentq

    tables = geo.base_tables()
    out = []
    for st in stations:
        x_c = brentq(lambda q: float(geo.camberline_station(q, cl0)) - st, 1e-9, 1.0 - 1e-9, xtol=1e-14)
        half = 0.5 * float(np.interp(st, tables.x, tables.thickness_corrected)) * t / geo.TABLE_T_MAX
        out.append(x_c - side * half * math.sin(float(geo.a10_slope_angle(x_c, cl0))))
    return np.array(out)


def curvature_along_x(curve: geo.ParametricBladeCurve, n: int) -> tuple[np.ndarray, np.ndarray]:
    u = np.linspace(0.0, 1.0, n)
    return curve.evaluate(u)[:, 0], curve.curvature(u)


def on_uniform_x(x: np.ndarray, kappa: np.ndarray, window: tuple[float, float], n: int) -> tuple[np.ndarray, np.ndarray]:
    inside = (x >= window[0] - 0.01) & (x <= window[1] + 0.01)
    xi, ki = x[inside], kappa[inside]
    assert np.all(np.diff(xi) > 0.0), "x/c is not monotone in the sawtooth window"
    xs = np.linspace(window[0], window[1], n)
    return xs, np.interp(xs, xi, ki)


def sawtooth_measures(curves: dict, n_x: int) -> list[dict]:
    """curves[(variant, surface, reconstruction)] = (x, kappa). Per variant, surface and window: the
    RMS and max of kappa_pchip - kappa_quintic and the detrended RMS of each (module docstring)."""
    rows = []
    for variant, surface in dict.fromkeys((v, s) for v, s, _ in curves):
        for window in SAWTOOTH_WINDOWS:
            xs, kp = on_uniform_x(*curves[(variant, surface, "pchip_sqrt")], window, n_x)
            _, kq = on_uniform_x(*curves[(variant, surface, "quintic_natural_sqrt")], window, n_x)
            row = dict(profile_variant=variant, surface=surface, x_lo=window[0], x_hi=window[1], x_points=n_x,
                       rms_difference=float(np.sqrt(np.mean((kp - kq) ** 2))), max_abs_difference=float(np.max(np.abs(kp - kq))))
            for name, k in (("pchip", kp), ("quintic", kq)):
                trend = np.polynomial.Polynomial.fit(xs, k, SAWTOOTH_TREND_DEGREE)
                row[f"rms_detrended_{name}"] = float(np.sqrt(np.mean((k - trend(xs)) ** 2)))
                row[f"rms_kappa_{name}"] = float(np.sqrt(np.mean(k**2)))
            rows.append(row)
    return rows


def le_comparison(cl0: float, t: float, n: int) -> dict:
    """The legacy C1 LE against the robust C2 LE of one blade (kovach_sandercock, sigma 1, 33 deg):
    reverse turning over the LE zone on the C1 polyline, the C1 analytic surface, the C2 polyline and
    the C2 curve, and the arrays the figure draws."""
    c2 = geo.make_cascade(spec_for(cl0, t, "kovach_sandercock"))
    c1 = geo.make_cascade(spec_for(cl0, t, "kovach_sandercock", surface_continuity="C1",
                                   thickness_reconstruction="quintic_natural_sqrt"))
    references = geo.naca65_reference_surfaces(cl0, t, profile_variant="kovach_sandercock",
                                               thickness_reconstruction="quintic_natural_sqrt")
    m = c2.metadata
    out = dict(cl0=cl0, t_max_over_c=t, designation=m["designation"], le_radius_over_c=m["le_radius_over_c"], surfaces={})
    for name, prefix, reference in (("suction", "ss", references[0]), ("pressure", "ps", references[1])):
        sense = -1.0 if name == "suction" else 1.0          # geometry.py: suction apex curvature negative
        anchor_s, anchor_u = float(m[f"{prefix}_le_anchor_over_c"]), float(m[f"{prefix}_le_anchor_u"])
        poly_c1 = np.asarray(getattr(c1.normalized, f"{name}_surface_unstaggered"))
        poly_c2 = np.asarray(getattr(c2.normalized, f"{name}_surface_unstaggered"))
        curve = getattr(c2.normalized_curves, f"{name}_unstaggered")
        u_c2 = np.asarray(getattr(c2.normalized_curves, f"{name}_parameters"))
        sample = reference.sample(C1_REFERENCE_POINTS)
        zone = sample.s <= anchor_s
        ref_turn = turning_metrics(tangent_angle(sample.tangent[zone], sense))
        seg = np.diff(poly_c2, axis=0)
        c2_poly_turn = turning_metrics(tangent_angle(seg[u_c2[1:] <= anchor_u], sense))
        check = curve_shape_check(curve, anchor_u, n=n)
        # plot arrays, to LE_PLOT_ANCHORS anchors of arc length
        s_max = LE_PLOT_ANCHORS * anchor_s
        u = np.linspace(0.0, min(1.0, LE_PLOT_ANCHORS * 1.2 * anchor_u), n)
        d1 = curve.derivative(u, 1)
        speed = np.hypot(d1[:, 0], d1[:, 1])
        s_c2 = np.concatenate(([0.0], np.cumsum(0.5 * (speed[1:] + speed[:-1]) * np.diff(u))))
        keep = s_c2 <= s_max
        plot_ref = sample.s <= s_max
        theta_ref = tangent_angle(sample.tangent, sense)
        out["surfaces"][name] = dict(
            sense=sense, anchor_over_c=anchor_s, anchor_u=anchor_u,
            anchor_s_c2=float(np.interp(anchor_u, u, s_c2)),
            c1_polyline=polyline_turning(poly_c1, sense, anchor_s), c1_analytic=ref_turn, c2_polyline=c2_poly_turn,
            c2_curve=dict(max_reverse_turn_deg=check["le_max_reverse_turn_deg"],
                          summed_reverse_turn_deg=check["le_summed_reverse_turn_deg"],
                          sign_changes=check["le_sign_changes"]),
            plot=dict(c1_points=poly_c1, c2_points=curve.evaluate(np.linspace(0.0, 0.06, 4001)), c1_s=sample.s[plot_ref],
                      c1_theta=np.degrees(theta_ref[plot_ref] - theta_ref[0]), c1_kappa=sample.kappa[plot_ref],
                      c2_s=s_c2[keep], c2_theta=np.degrees(tangent_angle(d1[keep], sense) - tangent_angle(d1[:1], sense)[0]),
                      c2_kappa=curve.curvature(u[keep]),
                      c2_anchor_point=curve.evaluate(anchor_u)))
    return out


def run_history(folder: Path, quick: bool, captions: dict) -> dict:
    n = QUICK["shape_points"] if quick else SHAPE_POINTS
    variants = QUICK["history_variants"] if quick else VARIANTS
    le_blades = QUICK["history_le"] if quick else HISTORY_LE_BLADES
    say(f"1d history: thickness interpolants, curvature {variants}, LE blades {le_blades}, {n} points")
    thick = thickness_interpolants()
    for name, c in thick["checks"].items():
        say(f"  {name}: |ours - geometry._HALF_THICKNESS| {c['max_diff_vs_geometry_half_thickness']:.3e}, "
            f"|2 ours - naca65_base_thickness| {c['max_diff_vs_naca65_base_thickness']:.3e}, "
            f"at stations {c['max_error_at_table_stations']:.3e} c")
    stations = thick["z"] ** 2
    pchip = thick["interpolants"]["pchip_sqrt"]
    inner = stations[(stations > 0.0) & (stations < 1.0)]
    jumps = [float(h_second_derivative(pchip, np.array([s + 1e-9]))[0] - h_second_derivative(pchip, np.array([s - 1e-9]))[0])
             for s in inner]
    h2 = dict(station=[float(s) for s in inner], pchip_h2_jump=jumps)
    say("  pchip h'' jump at the stations (1/c): " + ", ".join(f"{s:g}: {j:+.4f}" for s, j in zip(inner, jumps)))
    figure_h2(thick, folder / "history_thickness_h2.png", captions)

    cl0, t = HISTORY_BLADE
    curves, markers = {}, {}
    for variant in variants:
        for reconstruction in ("pchip_sqrt", "quintic_natural_sqrt"):
            cascade = geo.make_cascade(spec_for(cl0, t, variant, thickness_reconstruction=reconstruction))
            assert cascade.metadata["thickness_reconstruction"] == reconstruction
            for surface in ("suction", "pressure"):
                curves[(variant, surface, reconstruction)] = curvature_along_x(
                    getattr(cascade.normalized_curves, f"{surface}_unstaggered"), n)
        for surface, side in (("suction", 1.0), ("pressure", -1.0)):
            table_stations = stations[(stations > 0.0) & (stations <= (geo.KOVACH_START_STATION if variant == "kovach_sandercock" else 1.0) + 1e-12)]
            table_stations = table_stations[table_stations < 1.0]
            markers[(variant, surface)] = station_surface_x(cl0, t, table_stations, side)
    saw = sawtooth_measures(curves, SAWTOOTH_X_POINTS)
    for r in saw:
        say(f"  sawtooth {r['profile_variant']:17s} {r['surface']:8s} x/c {r['x_lo']}-{r['x_hi']}: rms(k_pchip - k_quintic) "
            f"{r['rms_difference']:.4f} /c (max {r['max_abs_difference']:.4f}); detrended rms pchip "
            f"{r['rms_detrended_pchip']:.4f}, quintic {r['rms_detrended_quintic']:.4f} /c")
    write_csv(folder / "history_sawtooth.csv", saw)
    for variant in variants:
        figure_curvature(curves, markers, variant, folder / f"history_curvature_{variant}.png", captions)

    le_rows = []
    for cl0_le, t_le in le_blades:
        data = le_comparison(cl0_le, t_le, n)
        tag = f"cl0_{cl0_le:g}_t_{t_le:.2f}"
        figure_le(data, folder / f"history_le_{tag}.png", captions)
        for name, s in data["surfaces"].items():
            row = dict(cl0=cl0_le, t_max_over_c=t_le, designation=data["designation"], surface=name,
                       anchor_over_c=s["anchor_over_c"], anchor_u=s["anchor_u"])
            for source in ("c1_polyline", "c1_analytic", "c2_polyline", "c2_curve"):
                row.update({f"{source}_{k}": v for k, v in s[source].items()})
            le_rows.append(row)
            say(f"  LE {data['designation']} {name:8s}: reverse turn max / summed (deg) C1 polyline "
                f"{s['c1_polyline']['max_reverse_turn_deg']:.4f} / {s['c1_polyline']['summed_reverse_turn_deg']:.4f} "
                f"({s['c1_polyline']['segments']} segments), C1 analytic {s['c1_analytic']['max_reverse_turn_deg']:.4f} / "
                f"{s['c1_analytic']['summed_reverse_turn_deg']:.4f}, C2 polyline {s['c2_polyline']['max_reverse_turn_deg']:.4g} / "
                f"{s['c2_polyline']['summed_reverse_turn_deg']:.4g}, C2 curve {s['c2_curve']['max_reverse_turn_deg']:.4g} / "
                f"{s['c2_curve']['summed_reverse_turn_deg']:.4g}")
    write_csv(folder / "history_le_reverse_turning.csv", le_rows)
    return dict(interpolant_checks=thick["checks"], pchip_h2_jumps=h2, sawtooth=saw, le_reverse_turning=le_rows)


# --- figures ------------------------------------------------------------------------------------------------

def pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": FONT_SIZE, "axes.titlesize": FONT_SIZE + 0.5, "axes.labelsize": FONT_SIZE,
                         "legend.fontsize": FONT_SIZE - 1, "xtick.labelsize": FONT_SIZE - 0.5,
                         "ytick.labelsize": FONT_SIZE - 0.5, "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
                         "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.labelcolor": "#0b0b0b",
                         "font.family": "DejaVu Sans", "savefig.facecolor": "white", "figure.facecolor": "white"})
    return plt


def _save(fig, path: Path, caption: str, captions: dict) -> None:
    fig.savefig(path, dpi=DPI)
    import matplotlib.pyplot as plt

    plt.close(fig)
    captions[path.name] = caption
    say(f"  figure {path.name}")


def _cell_grid(rows: list[dict], variant: str):
    mine = [r for r in rows if r["profile_variant"] == variant]
    cl0s = sorted({r["cl0"] for r in mine})
    ts = sorted({r["t_max_over_c"] for r in mine})
    lookup = {(r["cl0"], r["t_max_over_c"]): r for r in mine}
    return cl0s, ts, lookup


def _edges(values: list[float]) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    step = np.diff(v).min() if len(v) > 1 else 0.1
    return np.concatenate(([v[0] - step / 2], (v[:-1] + v[1:]) / 2, [v[-1] + step / 2]))


def _map_axes(plt, cl0s, ts, title):
    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, 0.62 * FIG_WIDTH_IN), layout="constrained")
    ax.set_xlabel("design lift coefficient cl0")
    ax.set_ylabel("thickness ratio t_max/c")
    ax.set_xticks(cl0s)
    ax.set_xticklabels([f"{c:.1f}" for c in cl0s])
    ax.set_yticks(ts)
    ax.set_yticklabels([f"{t:.2f}" for t in ts])
    ax.set_title(title, loc="left")
    return fig, ax


def _checked_box(ax, cl0s, ts):
    from matplotlib.patches import Rectangle

    xe, ye = _edges(cl0s), _edges(ts)
    x_hi = xe[np.searchsorted(cl0s, geo.CHECKED_CL0_MAX, side="right")]
    y_hi = ye[np.searchsorted(ts, geo.CHECKED_T_MAX, side="right")]
    ax.add_patch(Rectangle((xe[0], ye[0]), x_hi - xe[0], y_hi - ye[0], fill=False, lw=2.0, ec="#0b0b0b", zorder=5))


def _checked_handle():
    from matplotlib.lines import Line2D

    return Line2D([], [], color="#0b0b0b", lw=2.0,
                  label=f"checked range: cl0 <= {geo.CHECKED_CL0_MAX:g}, t/c <= {geo.CHECKED_T_MAX:g}")


def figure_status(rows: list[dict], variant: str, path: Path, captions: dict) -> None:
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    plt = pyplot()
    cl0s, ts, lookup = _cell_grid(rows, variant)
    order = ("built", "built_warning", "refused", "unexpected_error")
    codes = np.array([[order.index(lookup[(c, t)]["status"]) for c in cl0s] for t in ts], dtype=float)
    fig, ax = _map_axes(plt, cl0s, ts, f"make_cascade status, NACA 65 {variant}, solidity {SOLIDITY:g}, "
                                         f"stagger {STAGGER_DEG:g} deg")
    ax.pcolormesh(_edges(cl0s), _edges(ts), codes, cmap=ListedColormap([STATUS_INK[s] for s in order]), vmin=-0.5,
                  vmax=len(order) - 0.5, edgecolors="white", linewidth=1.5)
    letter = {"built": "B", "built_warning": "W", "unexpected_error": "E"}
    for (c, t), r in lookup.items():
        text = letter.get(r["status"]) or dict((k, code) for k, code, _ in REFUSAL_KINDS).get(r["refusal_kind"], "X")
        ax.text(c, t, text, ha="center", va="center", fontsize=FONT_SIZE - 2,
                color="#0b0b0b" if r["status"] == "built_warning" else "white")
    _checked_box(ax, cl0s, ts)
    used = sorted({r["refusal_kind"] for r in lookup.values() if r["status"] == "refused"})
    names = {k: code for k, code, _ in REFUSAL_KINDS}
    handles = [Patch(fc=STATUS_INK["built"], label="B built"), Patch(fc=STATUS_INK["built_warning"],
                                                                      label="W built, GeometryWarning")]
    handles += [Patch(fc=STATUS_INK["refused"], label=f"{names.get(k, 'X')} refused: {k.replace('_', ' ')}") for k in used]
    if any(r["status"] == "unexpected_error" for r in lookup.values()):
        handles.append(Patch(fc=STATUS_INK["unexpected_error"], label="E unexpected exception"))
    handles.append(_checked_handle())
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    counts = {s: sum(r["status"] == s for r in lookup.values()) for s in order}
    _save(fig, path, f"make_cascade outcome for {len(lookup)} NACA 65 blades ({variant}, C2 default spec, solidity "
                     f"{SOLIDITY:g}, stagger {STAGGER_DEG:g} deg): built {counts['built']}, built with GeometryWarning "
                     f"{counts['built_warning']}, refused {counts['refused']}"
                     + (f", unexpected {counts['unexpected_error']}" if counts["unexpected_error"] else "")
                     + f". The black frame is the range geometry.py states it was checked on (CHECKED_CL0_MAX "
                       f"{geo.CHECKED_CL0_MAX}, CHECKED_T_MAX {geo.CHECKED_T_MAX}); outside it the blade builds with a "
                       f"GeometryWarning. Letters name the refusal (legend).", captions)


def figure_fairness(rows: list[dict], variant: str, path: Path, captions: dict) -> None:
    plt = pyplot()
    cl0s, ts, lookup = _cell_grid(rows, variant)
    values = np.full((len(ts), len(cl0s)), np.nan)
    field = np.full_like(values, np.nan)
    for i, t in enumerate(ts):
        for j, c in enumerate(cl0s):
            r = lookup[(c, t)]
            if built(r["status"]):
                values[i, j] = field[i, j] = float(r["max_le_fairness"])
            elif r.get("refused_le_fairness") not in (None, ""):
                field[i, j] = float(r["refused_le_fairness"])
    fig, ax = _map_axes(plt, cl0s, ts, f"LE fairness max(SS, PS) R_LE^3 int kappa_s^2 ds, {variant}")
    mesh = ax.pcolormesh(_edges(cl0s), _edges(ts), np.ma.masked_invalid(values), cmap="Blues", vmin=0.5,
                         vmax=geo.LE_FAIRNESS_LIMIT, edgecolors="white", linewidth=1.0)
    for i, t in enumerate(ts):
        for j, c in enumerate(cl0s):
            if np.isfinite(values[i, j]):
                ax.text(c, t, f"{values[i, j]:.2f}", ha="center", va="center", fontsize=FONT_SIZE - 3.2,
                        color="white" if values[i, j] > 0.55 * geo.LE_FAIRNESS_LIMIT else "#0b0b0b")
    contour_note = ""
    finite = field[np.isfinite(field)]
    if finite.size and finite.min() < geo.LE_FAIRNESS_LIMIT < finite.max():
        cs = ax.contour(cl0s, ts, np.ma.masked_invalid(field), levels=[geo.LE_FAIRNESS_LIMIT], colors=INK["orange"],
                        linewidths=1.6)
        ax.clabel(cs, fmt={geo.LE_FAIRNESS_LIMIT: f"{geo.LE_FAIRNESS_LIMIT:g}"}, fontsize=FONT_SIZE - 1.5)
        contour_note = (f" The orange line is the {geo.LE_FAIRNESS_LIMIT:g} level (LE_FAIRNESS_LIMIT), interpolated "
                        "linearly between cell centres from the built values and the fairness printed in each "
                        "'LE fair is too steep' refusal (the first surface that failed: SS is tried first).")
    _checked_box(ax, cl0s, ts)
    ax.legend(handles=[_checked_handle()], loc="lower left", bbox_to_anchor=(0.0, 1.06), frameon=False)
    fig.colorbar(mesh, ax=ax, label="max(SS, PS) fairness", shrink=0.9)
    _save(fig, path, f"LE fairness R_LE^3 int kappa_s^2 ds, the larger of suction and pressure side, of every built "
                     f"blade ({variant}, solidity {SOLIDITY:g}, stagger {STAGGER_DEG:g} deg; cascade metadata "
                     f"ss/ps_le_fairness). Blank cells were refused. make_cascade refuses a fairness above "
                     f"{geo.LE_FAIRNESS_LIMIT:g}." + contour_note, captions)


def figure_grid_status(rows: list[dict], variant: str, path: Path, captions: dict) -> None:
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    plt = pyplot()
    cl0s, ts, lookup = _cell_grid(rows, variant)
    codes = np.full((len(ts), len(cl0s)), np.nan)
    for i, t in enumerate(ts):
        for j, c in enumerate(cl0s):
            r = lookup[(c, t)]
            if built(r["status"]):
                codes[i, j] = 0.0 if r.get("grid_status") == "ok" else 1.0
    fig, ax = _map_axes(plt, cl0s, ts, f"grid.build_grid(cascade, '{GRID_LEVEL}'), {variant}, inlet "
                                         f"{GRID_INLET_ANGLE_DEG:g} deg")
    ax.set_facecolor("#f0efec")
    ax.pcolormesh(_edges(cl0s), _edges(ts), np.ma.masked_invalid(codes),
                  cmap=ListedColormap([STATUS_INK["built"], STATUS_INK["refused"]]), vmin=-0.5, vmax=1.5,
                  edgecolors="white", linewidth=1.5)
    for (c, t), r in lookup.items():
        if built(r["status"]):
            ax.text(c, t, "G" if r.get("grid_status") == "ok" else "F", ha="center", va="center",
                    fontsize=FONT_SIZE - 2, color="white")
    _checked_box(ax, cl0s, ts)
    handles = [Patch(fc=STATUS_INK["built"], label="G grid built"), Patch(fc=STATUS_INK["refused"], label="F grid refused"),
               Patch(fc="#f0efec", label="geometry refused"), _checked_handle()]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    made = [r for r in lookup.values() if built(r["status"])]
    ok = sum(r.get("grid_status") == "ok" for r in made)
    _save(fig, path, f"Standard grid of level '{GRID_LEVEL}' (grid.build_grid, inlet angle {GRID_INLET_ANGLE_DEG:g} deg, "
                     f"exit angle kappa_te) for every built blade of the {variant} sweep: {ok} of {len(made)} built. "
                     f"Grey cells: make_cascade refused the blade.", captions)


def figure_cascade(rows: list[dict], path: Path, captions: dict) -> None:
    from matplotlib.lines import Line2D

    plt = pyplot()
    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, 0.62 * FIG_WIDTH_IN), layout="constrained")
    colors = (INK["blue"], INK["orange"], INK["aqua"])
    offsets = ((4, 5), (4, 6), (4, -12))            # direct labels at the left end, clear of each other
    blades = list(dict.fromkeys((r["cl0"], r["t_max_over_c"]) for r in rows))
    lower_note, ceilings = "", []
    for color, offset, (cl0, t) in zip(colors, offsets, blades):
        mine = [r for r in rows if (r["cl0"], r["t_max_over_c"]) == (cl0, t)]
        st = np.array([r["stagger_deg"] for r in mine])
        top = np.array([float(r["largest_accepted_solidity"]) if r.get("largest_accepted_solidity") not in ("", None)
                        else np.nan for r in mine])
        at_limit = np.array([bool(r.get("accepted_at_search_limit")) for r in mine])
        label = f"cl0 {cl0:g}, t/c {t:.2f}"
        ax.plot(st, top, "-o", color=color, lw=2.0, ms=4.5, label=f"{label}: largest accepted solidity")
        if at_limit.any():
            ax.plot(st[at_limit], top[at_limit], "o", mfc="white", mec=color, ms=6)
        low = np.array([float(r["smallest_accepted_solidity"]) if r.get("largest_refused_below") not in ("", None)
                        else np.nan for r in mine])
        if np.isfinite(low).any():
            ax.plot(st, low, ":", marker="v", color=color, lw=1.4, ms=4.5, label=f"{label}: smallest accepted solidity")
            lower_note = (" Dotted lines with triangles: the smallest accepted solidity where the smallest scan "
                          "solidity was refused (bisected the same way).")
        ceiling = mine[0]["kappa_le_stagger_ceiling_deg"]
        ax.axvline(ceiling, color=color, ls="--", lw=1.2)
        ceilings.append(Line2D([], [], color=color, ls="--", lw=1.2,
                               label=f"{label}: kappa_le = 90 deg at stagger {ceiling:.2f}"))
        last = np.flatnonzero(np.isfinite(top))
        if last.size:
            ax.annotate(label, (st[last[0]], top[last[0]]), textcoords="offset points", xytext=offset,
                        fontsize=FONT_SIZE - 1.5, color="#0b0b0b")
    ax.set_yscale("log")
    ax.set_xlim(min(r["stagger_deg"] for r in rows) - 3.0,
                max(max(r["stagger_deg"] for r in rows), max(r["kappa_le_stagger_ceiling_deg"] for r in rows)) + 3.0)
    ax.set_xlabel("stagger (deg from axial)")
    ax.set_ylabel("solidity c/s")
    ax.grid(True, which="both", color=GRID_INK, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_title(f"Largest solidity make_cascade accepts (NACA 65 {CASCADE_VARIANT}, C2)", loc="left")
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=handles + ceilings, loc="lower left", frameon=False, fontsize=FONT_SIZE - 1.5)
    r0 = rows[0]
    _save(fig, path, f"Largest solidity geometry.make_cascade accepts against stagger for three NACA 65 blades "
                     f"({CASCADE_VARIANT}, default C2 spec): scan {r0['scan_solidities']}, then bisection between the "
                     f"last accepted and the first refused value to a relative width {r0['solidity_rtol']:g}; open "
                     f"markers: accepted at the search limit {r0['solidity_search_limit']:g}. Dashed lines: the stagger "
                     f"at which kappa_le = stagger + chi_le reaches 90 deg, where grid.py refuses the blade (one per "
                     f"blade, same colour)." + lower_note, captions)


def figure_h2(thick: dict, path: Path, captions: dict) -> None:
    plt = pyplot()
    station = np.linspace(H2_WINDOW[0], H2_WINDOW[1], 9001)
    stations = thick["z"] ** 2
    fig, axes = plt.subplots(2, 1, figsize=(FIG_WIDTH_IN, 0.75 * FIG_WIDTH_IN), sharex=True, layout="constrained")
    h2 = {name: h_second_derivative(s, station) for name, s in thick["interpolants"].items()}
    for ax in axes:
        for st in stations[(stations >= H2_WINDOW[0]) & (stations <= H2_WINDOW[1])]:
            ax.axvline(st, color=GRID_INK, lw=0.8, zorder=0)
    axes[0].plot(station, h2["pchip_sqrt"], color=INK["orange"], lw=1.4, label="pchip_sqrt (legacy, C1)")
    axes[0].plot(station, h2["quintic_natural_sqrt"], color=INK["blue"], lw=1.6, label="quintic_natural_sqrt (C4)")
    axes[0].set_ylabel("h'' = d2h/dst2 (1/c)")
    axes[0].legend(frameon=False, loc="lower right")
    axes[0].set_title("Half-thickness second derivative of the 65-010 base form (t/c 0.10)", loc="left")
    axes[1].plot(station, h2["pchip_sqrt"] - h2["quintic_natural_sqrt"], color=INK["orange"], lw=1.2)
    axes[1].axhline(0.0, color=MUTED, lw=0.6)
    axes[1].set_ylabel("pchip - quintic (1/c)")
    axes[1].set_xlabel("thickness station s/S along the camberline (grey: Table 4-1 stations)")
    _save(fig, path, "Second derivative of the half-thickness h(station) of the base 65-010 form (t/c 0.10, Aungier 2003 "
                     "Table 4-1 corrected column) for the two reconstructions in z = sqrt(station), rebuilt here with "
                     "SciPy from geometry.base_tables() (PchipInterpolator; interpolating quintic with h''' = h'''' = 0 at "
                     "both ends) and checked against geometry.py's own interpolants to "
                     f"{INTERPOLANT_TOL:g} c. PCHIP's h'' jumps at every station (grey lines); the quintic's is continuous. "
                     "Lower panel: their difference.", captions)


def figure_curvature(curves: dict, markers: dict, variant: str, path: Path, captions: dict) -> None:
    plt = pyplot()
    fig, axes = plt.subplots(2, 1, figsize=(FIG_WIDTH_IN, 0.85 * FIG_WIDTH_IN), sharex=True, layout="constrained")
    lo, hi = CURVATURE_WINDOW
    for ax, surface in zip(axes, ("suction", "pressure")):
        for x in markers[(variant, surface)]:
            if lo <= x <= hi:
                ax.axvline(x, color=GRID_INK, lw=0.8, zorder=0)
        span = []
        for reconstruction, color, lw, label in (("pchip_sqrt", INK["orange"], 1.2, "pchip_sqrt (legacy)"),
                                                 ("quintic_natural_sqrt", INK["blue"], 1.6, "quintic_natural_sqrt (default)")):
            x, k = curves[(variant, surface, reconstruction)]
            inside = (x >= lo) & (x <= hi)
            ax.plot(x[inside], k[inside], color=color, lw=lw, label=label)
            span.append(k[inside])
        allk = np.concatenate(span)
        pad = 0.08 * (allk.max() - allk.min())
        ax.set_ylim(allk.min() - pad, allk.max() + pad)
        ax.set_ylabel(f"{surface} curvature (1/c)")
        ax.set_title(f"{surface} surface", loc="left")
    axes[0].legend(frameon=False, loc="best")
    axes[1].set_xlabel("x/c (grey: Table 4-1 stations on the surface"
                       + (", up to the Kovach start 0.60)" if variant == "kovach_sandercock" else ")"))
    cl0, t = HISTORY_BLADE
    _save(fig, path, f"Signed curvature (blade frame) of the C2 surfaces of the NACA 65-(12A10)10 ({variant}, cl0 {cl0:g}, "
                     f"t/c {t:.2f}, solidity {SOLIDITY:g}, stagger {STAGGER_DEG:g} deg) built with the legacy "
                     f"thickness_reconstruction 'pchip_sqrt' and the default 'quintic_natural_sqrt', x/c {lo:g} to {hi:g}. "
                     "Grey lines: the Table 4-1 stations, placed on each surface (camber point at station s/S moved "
                     "along the camberline normal by the half-thickness). The PCHIP curvature steps at the stations "
                     "(sawtooth); the quintic does not.", captions)


def figure_le(data: dict, path: Path, captions: dict) -> None:
    plt = pyplot()
    fig = plt.figure(figsize=(FIG_WIDTH_IN, 1.2 * FIG_WIDTH_IN), layout="constrained")
    grid = fig.add_gridspec(3, 2, height_ratios=(1.5, 1.0, 1.0))
    shape = fig.add_subplot(grid[0, :])
    styles = {"c1": dict(color=INK["orange"], lw=1.2, ls="--"), "c2": dict(color=INK["blue"], lw=1.6, ls="-")}
    x_lo, y_span = -0.001, []
    for name, s in data["surfaces"].items():
        p = s["plot"]
        c1 = p["c1_points"][p["c1_points"][:, 0] <= LE_CLOSEUP_X * 1.05]
        shape.plot(c1[:, 0], c1[:, 1], marker="o", ms=2.5, **styles["c1"], label="C1 legacy LE (polyline)" if name == "suction" else None)
        c2 = p["c2_points"]
        c2 = c2[c2[:, 0] <= LE_CLOSEUP_X * 1.05]
        shape.plot(c2[:, 0], c2[:, 1], **styles["c2"], label="C2 robust LE (curve)" if name == "suction" else None)
        shape.plot(*p["c2_anchor_point"], marker="D", ms=5, color="#0b0b0b", ls="none",
                   label="C2 anchor" if name == "suction" else None)
        y_span += [c1[c1[:, 0] <= LE_CLOSEUP_X, 1], c2[c2[:, 0] <= LE_CLOSEUP_X, 1]]
    y = np.concatenate(y_span)
    pad = 0.04 * (y.max() - y.min())
    shape.set_xlim(x_lo, LE_CLOSEUP_X)
    shape.set_ylim(y.min() - pad, y.max() + pad)
    shape.set_aspect("equal", adjustable="box")
    shape.set_xlabel("x/c")
    shape.set_ylabel("y/c")
    shape.grid(True, color=GRID_INK, lw=0.5)
    shape.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.suptitle(f"{data['designation']}, kovach_sandercock: the leading edge (blade frame; top panel to scale)",
                 x=0.01, ha="left", fontsize=FONT_SIZE + 0.5)
    for col, name in enumerate(("suction", "pressure")):
        s = data["surfaces"][name]
        p = s["plot"]
        theta = fig.add_subplot(grid[1, col])
        kappa = fig.add_subplot(grid[2, col], sharex=theta)
        theta.plot(p["c1_s"], p["c1_theta"], **styles["c1"], label="C1 legacy")
        theta.plot(p["c2_s"], p["c2_theta"], **styles["c2"], label="C2 robust")
        kappa.plot(p["c1_s"], s["sense"] * p["c1_kappa"] * data["le_radius_over_c"], **styles["c1"])
        kappa.plot(p["c2_s"], s["sense"] * p["c2_kappa"] * data["le_radius_over_c"], **styles["c2"])
        for ax in (theta, kappa):
            ax.axvline(s["anchor_over_c"], color=MUTED, ls=":", lw=1.0)
            ax.grid(True, color=GRID_INK, lw=0.5)
            ax.set_axisbelow(True)
        kappa.axhline(0.0, color=MUTED, lw=0.6)
        theta.set_title(f"{name}: reverse turn C1 {s['c1_polyline']['summed_reverse_turn_deg']:.2f}, "
                        f"C2 {s['c2_curve']['summed_reverse_turn_deg']:.2g} deg", loc="left", fontsize=FONT_SIZE - 1)
        theta.set_ylabel("tangent turned (deg)")
        kappa.set_ylabel("sense x kappa R_LE")
        kappa.set_xlabel("arc length from the apex s/c")
        if col == 0:
            theta.legend(frameon=False, loc="lower right")
    _save(fig, path, f"Leading edge of the {data['designation']} (kovach_sandercock, solidity {SOLIDITY:g}, stagger "
                     f"{STAGGER_DEG:g} deg): legacy C1 construction (surface_continuity 'C1' with thickness_reconstruction "
                     "'quintic_natural_sqrt', so only the LE differs; dashed, polyline vertices as dots; its curvature "
                     "from the analytic C1 surface) against the robust C2 LE (solid). Top: shape close-up; diamonds: "
                     "the C2 anchors. Middle: tangent angle turned since the apex, oriented so that the nose turning is "
                     "positive. Bottom: curvature times R_LE, same orientation (1 on the nose circle). Dotted: the C2 "
                     "anchor. Panel titles: summed reverse turning over the LE zone (apex to the anchor) on the C1 "
                     "polyline and on the C2 curve.", captions)


# --- main ---------------------------------------------------------------------------------------------------

PARTS = ("profiles", "grids", "cascade", "history")


def versions() -> dict:
    import matplotlib
    import scipy

    return dict(python=sys.version.split()[0], python_full=sys.version, numpy=np.__version__, scipy=scipy.__version__,
                matplotlib=matplotlib.__version__, platform=platform.platform(), executable=sys.executable)


def settings() -> dict:
    names = ("SOLIDITY", "STAGGER_DEG", "CL0_STEP", "T_STEP", "VARIANTS", "SHAPE_POINTS", "KAPPA_SIGN_THRESHOLD",
             "GRID_LEVEL", "GRID_INLET_ANGLE_DEG", "CASCADE_BLADES", "CASCADE_VARIANT", "CASCADE_STAGGERS_DEG",
             "SOLIDITY_SCAN", "SOLIDITY_RTOL", "SPOT_CHECK", "GRID_CHECK_SOLIDITY", "HISTORY_BLADE", "HISTORY_LE_BLADES",
             "CURVATURE_WINDOW", "SAWTOOTH_WINDOWS", "SAWTOOTH_X_POINTS", "SAWTOOTH_TREND_DEGREE", "H2_WINDOW",
             "INTERPOLANT_TOL", "LE_CLOSEUP_X", "LE_PLOT_ANCHORS", "C1_REFERENCE_POINTS", "QUICK", "DPI")
    out = {name: globals()[name] for name in names}
    out.update(cl0_values=exact_values(CL0_STEP, CL0_STEPS), t_values=exact_values(T_STEP, T_STEPS),
               geometry_constants=dict(CHECKED_CL0_MAX=geo.CHECKED_CL0_MAX, CHECKED_T_MAX=geo.CHECKED_T_MAX,
                                       LE_FAIRNESS_LIMIT=geo.LE_FAIRNESS_LIMIT, LE_ANCHOR_RANGE=geo.LE_ANCHOR_RANGE,
                                       LE_ANCHOR_OFFSET=geo.LE_ANCHOR_OFFSET, MIN_CAMBER_STATIONS=geo.MIN_CAMBER_STATIONS),
               default_spec=str(spec_for(1.2, 0.10, "kovach_sandercock")),
               definitions="see the script's module docstring, 'Definitions'")
    return out


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--part", nargs="+", choices=PARTS + ("all",), default=["all"],
                        help="parts to run; 'grids' also runs 'profiles' (it needs the built blades)")
    parser.add_argument("--quick", action="store_true", help="a few blades and staggers (the test's mini run)")
    parser.add_argument("--output-root", type=Path, default=ROOT / OUTPUT_DIR,
                        help="where the timestamped run folder is made (default: outputs/naca65_parameter_envelope)")
    args = parser.parse_args(argv)
    parts = set(PARTS) if "all" in args.part else set(args.part)
    if "grids" in parts:
        parts.add("profiles")
    LINES.clear()
    started, date = time.perf_counter(), datetime.now().isoformat(timespec="seconds")
    folder = new_run_folder(Path(args.output_root))
    status = git("status", "--porcelain")
    run = dict(script="scripts/naca65_parameter_envelope.py", issue=7, commit=git("rev-parse", "HEAD"),
               branch=git("rev-parse", "--abbrev-ref", "HEAD"), dirty=bool(status) if status is not None else None,
               git_status_porcelain=status.splitlines() if status else [],
               command_line=[sys.executable, *sys.argv] if argv is None else ["main", *argv],
               parts=sorted(parts, key=PARTS.index), quick=args.quick, date=date, folder=folder,
               geometry_module=geo.__file__, versions=versions(), settings=settings(), finished=False)
    write_json(folder / "run.json", run)
    say(f"run folder {folder}")
    say(f"commit {run['commit']} dirty {run['dirty']}; numpy {np.__version__}; geometry {geo.__file__}")
    captions: dict[str, str] = {}
    summary: dict = {}
    if "profiles" in parts:
        rows = run_profiles(args.quick, with_grids="grids" in parts)
        write_csv(folder / "envelope_profiles.csv", rows)
        summary["profiles"] = profile_summary(rows, "grids" in parts)
        report_profiles(summary["profiles"])
        for variant in dict.fromkeys(r["profile_variant"] for r in rows):
            figure_status(rows, variant, folder / f"profile_status_{variant}.png", captions)
            figure_fairness(rows, variant, folder / f"profile_fairness_{variant}.png", captions)
            if "grids" in parts:
                figure_grid_status(rows, variant, folder / f"grid_medium_status_{variant}.png", captions)
    if "cascade" in parts:
        rows_c = run_cascade(args.quick)
        write_csv(folder / "envelope_cascade.csv", rows_c)
        summary["cascade"] = [{k: r.get(k) for k in ("cl0", "t_max_over_c", "stagger_deg", "scan_pattern",
                                                      "largest_accepted_solidity", "refusal_kind_at_bound",
                                                      "smallest_accepted_solidity", "refusal_kind_below",
                                                      "accepted_at_search_limit", "spot_inside_accepted",
                                                      "spot_outside_refused", "throat_over_pitch_at_largest",
                                                      "kappa_le_deg", "grid_blade_angles_admissible", "grid_status")}
                              for r in rows_c]
        figure_cascade(rows_c, folder / "cascade_envelope.png", captions)
    if "history" in parts:
        summary["history"] = run_history(folder, args.quick, captions)
    write_json(folder / "summary.json", summary)
    write_json(folder / "captions.json", captions)
    seconds = time.perf_counter() - started
    say(f"done in {seconds:.1f} s: {folder}")
    (folder / "log.txt").write_text("\n".join(LINES) + "\n", encoding="utf-8")
    run.update(finished=True, run_seconds=seconds, files=sorted(p.name for p in folder.iterdir()) + ["run.json"])
    write_json(folder / "run.json", run)
    return folder


if __name__ == "__main__":
    main()
