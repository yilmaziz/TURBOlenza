"""
audit_naca_thickness_reconstruction.py
======================================
Geometry-only audit (with a limited solver what-if) of how the tabulated NACA 65 thickness
is reconstructed between its table stations. DIAGNOSTIC ONLY: the production geometry is
not changed. Run 2026-11-13 against the PCHIP geometry; its outcome, C_quintic_natural, has
since been the production C2 reconstruction (geometry.py, CascadeSpec.thickness_reconstruction
"quintic_natural_sqrt"), A_pchip the selectable legacy one ("pchip_sqrt", the C1 default).
"production" in the report and the figures means the geometry the audit was run against,
the legacy PCHIP. A_pchip and C_quintic_natural are built through CascadeSpec; a diagnostic
candidate (B, D) is registered in potential_flow_solver.geometry only inside
thickness_reconstruction() (under its own name, beside the production ones; caches cleared
on entry and exit) and taken out again; nothing else of the construction changes: the same
table, camber line, station reading (arc length along the camber line), LE nose and fair,
TE treatment and C2 fit.

    python scripts/audit_naca_thickness_reconstruction.py              # geometry + solver what-if
    python scripts/audit_naca_thickness_reconstruction.py --no-solve   # geometry only

Candidates (half-thickness h as a function of z = sqrt(station), as production reads it):
    A_pchip               legacy "pchip_sqrt", production when audited: PchipInterpolator (C1;
                          exact at the stations)
    B_cubic_natural       interpolating cubic spline, h'' = 0 at both ends (C2; exact)
    B_cubic_not_a_knot    interpolating cubic spline, not-a-knot ends (C2; exact)
    B_cubic_clamped_le    interpolating cubic spline, dh/dz(0) = sqrt(2 R_LE) (the nose
                          radius of the table, h ~ sqrt(2 R_LE s)), not-a-knot TE (C2; exact)
    C_quintic_natural     interpolating quintic spline, h''' = h'''' = 0 at both ends (C4;
                          exact; the interpolant of least int h'''^2); production C2 since
                          2026-11-13, "quintic_natural_sqrt"
    D_smoothing_cubic     penalized cubic smoothing spline (int h''^2), the largest penalty
                          that keeps every table ordinate within TABLE_TOLERANCE (C2; NOT
                          exact, bounded)

Outputs: OUTPUT_DIR (report.txt, candidate_summary.csv, table_fit_errors.csv,
geometry_displacements.csv, curvature_metrics.csv, continuity.csv, solver_what_if.csv and the
figures fig1 .. fig8, geometry_ripple_causality.png).
"""

# ============================================================
# SETTINGS
# ============================================================

OUTPUT_DIR = "outputs/naca_thickness_reconstruction_audit"
VARIANTS = ("emery_base", "kovach_sandercock")
SOLVER_VARIANT = "emery_base"           # the validation case of scripts/compare_dunavant_aungier.py
CL0, T_MAX, STAGGER, SOLIDITY = 1.2, 0.10, 33.0, 1.0
TABLE_TOLERANCE = 5e-6                  # thickness / c: half the last printed digit of Table 4-1 (0.001 % c)
RIPPLE_WINDOW = (0.20, 0.85)            # x/c of the curvature-ripple metric (the table spacing is 0.05 here)
REGIONS = (("LE table", 0.0, 0.05), ("forward table", 0.05, 0.20), ("mid-chord", 0.20, 0.60),
           ("aft", 0.60, 0.95), ("TE", 0.95, 1.0))
LE_FAIR_MARGIN = 0.003                  # x/c: "LE table" starts this far past the LE fair's anchor (the fair is
                                        # judged by its own fairness metric; its B-spline representation at the
                                        # G2 anchor depends on the knot layout, not on the thickness)
END_TOLERANCE = 0.10                    # relative: dh/dz(0) within this of the parabolic nose slope sqrt(2 R_LE)
TREND_DEGREE = 6                        # the low-frequency reference: this polynomial fit of the PRODUCTION curve
N_DENSE = 40001
SOLVER_LEVELS = ("medium", "very_fine")
FIVE_LEVELS = ("very_coarse", "coarse", "medium", "fine", "very_fine")
SAWTOOTH_RUNS = "outputs/ss_sawtooth_audit/runs.pkl"    # the production runs of scripts/audit_ss_sawtooth.py
N_WORKERS = 8

# ============================================================

import argparse                                                                  # noqa: E402
import csv                                                                       # noqa: E402
import math                                                                      # noqa: E402
import pickle                                                                    # noqa: E402
import sys                                                                       # noqa: E402
import warnings                                                                  # noqa: E402
from concurrent.futures import ProcessPoolExecutor                               # noqa: E402
from contextlib import contextmanager                                            # noqa: E402
from pathlib import Path                                                         # noqa: E402
from types import MappingProxyType                                               # noqa: E402

import numpy as np                                                               # noqa: E402
from scipy.interpolate import BSpline, CubicSpline, PPoly, make_smoothing_spline   # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / OUTPUT_DIR
LINES: list = []
CANDIDATES = ("A_pchip", "B_cubic_natural", "B_cubic_not_a_knot", "B_cubic_clamped_le", "C_quintic_natural",
              "D_smoothing_cubic")
# the candidates that are production reconstructions (geometry.THICKNESS_RECONSTRUCTIONS)
PRODUCTION_NAMES = {"A_pchip": "pchip_sqrt", "C_quintic_natural": "quintic_natural_sqrt"}
INK = {"A_pchip": "#e6550d", "B_cubic_natural": "#9ecae1", "B_cubic_not_a_knot": "#3182bd",
       "B_cubic_clamped_le": "#08519c", "C_quintic_natural": "#31a354", "D_smoothing_cubic": "#756bb1"}


def say(text: str = "") -> None:
    print(text)
    LINES.append(text)


# --- the table and the candidate reconstructions ---------------------------------------------------------------

def table():
    """(stations, corrected thickness t/c, z = sqrt(station), half-thickness) of Table 4-1 as
    geometry.py stores and corrects it (65-010, t/c 0.10)."""
    from potential_flow_solver import geometry as g

    x = np.array(g._TABLE_X_PCT) / 100.0
    t = g._corrected_thickness()
    return x, t, np.sqrt(x), 0.5 * t


def smoothing_penalty(z, h, tolerance_half: float) -> float:
    """The largest penalty of make_smoothing_spline keeping every |h - h_i| <= tolerance_half."""
    lo, hi = 1e-16, 1e-1
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        if np.max(np.abs(make_smoothing_spline(z, h, lam=mid)(z) - h)) <= tolerance_half:
            lo = mid
        else:
            hi = mid
        if hi / lo < 1.0 + 1e-9:
            break
    return lo


def candidate(name: str):
    """The half-thickness h(z) of candidate name as a PPoly (the production interpolant itself
    for A_pchip and C_quintic_natural)."""
    from potential_flow_solver import geometry as g

    _, _, z, h = table()
    if name in PRODUCTION_NAMES:
        return g._HALF_THICKNESS[PRODUCTION_NAMES[name]]
    if name == "B_cubic_natural":
        return CubicSpline(z, h, bc_type="natural")
    if name == "B_cubic_not_a_knot":
        return CubicSpline(z, h, bc_type="not-a-knot")
    if name == "B_cubic_clamped_le":
        return CubicSpline(z, h, bc_type=((1, math.sqrt(2.0 * g.TABLE_LE_RADIUS)), "not-a-knot"))
    if name == "D_smoothing_cubic":
        return PPoly.from_spline(make_smoothing_spline(z, h, lam=smoothing_penalty(z, h, 0.5 * TABLE_TOLERANCE)))
    raise ValueError(f"unknown candidate {name!r}")


def _clear_geometry_caches() -> None:
    from potential_flow_solver import geometry as g

    for name in dir(g):
        item = getattr(g, name)
        if callable(getattr(item, "cache_clear", None)):
            item.cache_clear()


def production_state() -> tuple:
    """The geometry.py objects a diagnostic registration replaces (and must restore)."""
    from potential_flow_solver import geometry as g

    return g._HALF_THICKNESS, g.THICKNESS_RECONSTRUCTIONS, g._RECONSTRUCTION_MAXIMUM


@contextmanager
def thickness_reconstruction(name: str):
    """Yields the CascadeSpec.thickness_reconstruction that builds candidate name. A production
    reconstruction (PRODUCTION_NAMES) yields its own name and touches nothing. DIAGNOSTIC for
    the others: the candidate is registered in geometry.py under "audit_<name>" beside the
    production ones (the slopes come from geometry's own polynomial path, any PPoly) inside
    the block; on exit the production state is back and the caches cleared."""
    from potential_flow_solver import geometry as g

    if name in PRODUCTION_NAMES:
        yield PRODUCTION_NAMES[name]
        return
    production = production_state()
    key = f"audit_{name}"
    rec = candidate(name)
    _clear_geometry_caches()
    g._HALF_THICKNESS = MappingProxyType({**production[0], key: rec})
    g.THICKNESS_RECONSTRUCTIONS = production[1] + (key,)
    try:
        g._RECONSTRUCTION_MAXIMUM = MappingProxyType({**production[2], key: g._reconstruction_maximum(key)})
        yield key
    finally:
        g._HALF_THICKNESS, g.THICKNESS_RECONSTRUCTIONS, g._RECONSTRUCTION_MAXIMUM = production
        _clear_geometry_caches()


def build_cascade(name: str, variant: str):
    from potential_flow_solver.geometry import CascadeSpec, make_cascade

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with thickness_reconstruction(name) as reconstruction:
            return make_cascade(CascadeSpec(blade_type="NACA65", solidity=SOLIDITY, stagger_deg=STAGGER, cl0=CL0,
                                            t_max_over_c=T_MAX, profile_variant=variant, surface_continuity="C2",
                                            thickness_reconstruction=reconstruction))


# --- thickness-function audit -----------------------------------------------------------------------------------

def thickness_profile(reconstruction, n: int = N_DENSE):
    """station s, t(s), dt/ds, d2t/ds2 (t = 2 h, base t/c 0.10) on a dense station grid."""
    s = np.linspace(1e-5, 1.0 - 1e-6, n)
    z = np.sqrt(s)
    b0, b1, b2 = reconstruction(z), reconstruction(z, 1), reconstruction(z, 2)
    return s, 2.0 * b0, 2.0 * b1 / (2.0 * z), 2.0 * (b2 - b1 / z) / (4.0 * s)


def thickness_checks(name: str, reconstruction) -> dict:
    x, t, z, h = table()
    errors = 2.0 * reconstruction(z) - t
    s, tt, t1, _ = thickness_profile(reconstruction)
    k = int(np.argmax(tt))
    inside = (s > 1e-4) & (s < 1.0 - 1e-4)
    rising, falling = (s < s[k]) & inside, (s > s[k]) & inside
    # a monotone table interval must stay monotone: count wrong-sign slopes inside such intervals,
    # except the two intervals next to the table's own maximum station, which hold the continuous
    # maximum of any smooth reconstruction (the table samples the hump; its peak need not be a station)
    peak_index = int(np.argmax(t))
    wrong = 0
    for i, (a, b, ta, tb) in enumerate(zip(x[:-1], x[1:], t[:-1], t[1:])):
        if i in (peak_index - 1, peak_index):
            continue
        on = (s > a) & (s < b)
        if tb > ta:
            wrong += int(np.count_nonzero(t1[on] < 0.0))
        elif tb < ta:
            wrong += int(np.count_nonzero(t1[on] > 0.0))
    from potential_flow_solver import geometry as g

    le_slope = float(reconstruction(0.0, 1))
    le_slope_nose = math.sqrt(2.0 * g.TABLE_LE_RADIUS)              # h = sqrt(2 R_LE s) = sqrt(2 R_LE) z, parabolic nose
    te_curvature = float(reconstruction(1.0, 2))
    # the second divided difference of the last three table points in z (information only: three points)
    z3, h3 = z[-3:], h[-3:]
    te_curvature_wedge = 2.0 * ((h3[2] - h3[1]) / (z3[2] - z3[1]) - (h3[1] - h3[0]) / (z3[1] - z3[0])) / (z3[2] - z3[0])
    fit = np.polyfit(x[peak_index - 1:peak_index + 2], t[peak_index - 1:peak_index + 2], 2)
    quadratic_peak_x = -fit[1] / (2.0 * fit[0])
    return dict(candidate=name, max_abs_table_error=float(np.max(np.abs(errors))),
                rms_table_error=float(np.sqrt(np.mean(errors ** 2))), table_exact=bool(np.max(np.abs(errors)) <= 1e-15),
                within_tolerance=bool(np.max(np.abs(errors)) <= TABLE_TOLERANCE * (1 + 1e-9)),
                t_max=float(tt[k]), x_t_max=float(s[k]), min_thickness_interior=float(np.min(tt[inside])),
                positive=bool(np.all(tt[inside] > 0.0)), single_maximum=bool(np.all(t1[rising] > 0.0) and np.all(t1[falling] < 0.0)),
                wrong_slope_samples_in_monotone_intervals=wrong, peak_excess_over_table=float(tt[k] - t[peak_index]),
                three_point_quadratic_peak=float(np.polyval(fit, quadratic_peak_x)),
                three_point_quadratic_x=float(quadratic_peak_x), le_slope=le_slope, le_slope_nose=le_slope_nose,
                le_slope_relative=abs(le_slope / le_slope_nose - 1.0), te_curvature=te_curvature,
                te_curvature_wedge=te_curvature_wedge, te_curvature_relative=abs(te_curvature / te_curvature_wedge - 1.0),
                errors=errors)


def knot_continuity(pp, orders=(0, 1, 2, 3)) -> dict:
    """Max |left - right| of the derivatives of a piecewise polynomial (PPoly or the PCHIP) at its
    interior breakpoints."""
    x = np.asarray(pp.x)
    out = {}
    for order in orders:
        d = pp.derivative(order) if order else pp
        c = d.c
        left = np.array([np.polyval(c[:, i - 1], x[i] - x[i - 1]) for i in range(1, len(x) - 1) if x[i] > x[i - 1]])
        right = np.array([np.polyval(c[:, i], 0.0) for i in range(1, len(x) - 1) if x[i] > x[i - 1]])
        out[order] = float(np.max(np.abs(left - right))) if left.size else 0.0
    return out


# --- final-curve audit ------------------------------------------------------------------------------------------

def curve_profile(curve, n: int = N_DENSE):
    """x, y, dy/dx, d2y/dx2, kappa, |dkappa/ds| of a blade-frame curve on a dense u grid."""
    u = np.linspace(0.0, 1.0, n)
    p, d1, d2 = curve.evaluate(u), curve.derivative(u, 1), curve.derivative(u, 2)
    d3 = curve.derivative(u, 3)
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = d1[:, 1] / d1[:, 0]
        second = (d2[:, 1] * d1[:, 0] - d1[:, 1] * d2[:, 0]) / d1[:, 0] ** 3
    speed = np.hypot(d1[:, 0], d1[:, 1])
    cross = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    kappa = cross / speed ** 3
    cross_u = d1[:, 0] * d3[:, 1] - d1[:, 1] * d3[:, 0]
    dkappa_du = cross_u / speed ** 3 - 3.0 * cross * (d1[:, 0] * d2[:, 0] + d1[:, 1] * d2[:, 1]) / speed ** 5
    return dict(u=u, x=p[:, 0], y=p[:, 1], slope=slope, second=second, kappa=kappa, dkappa_ds=dkappa_du / speed)


def curve_continuity(curve) -> dict:
    """Max |left - right| at the interior knots of a ParametricBladeCurve (exact one-sided
    piece polynomials) of X, X', X'', X''', kappa, dy/dx, d2y/dx2."""
    t, c, k = np.asarray(curve.knots), np.asarray(curve.control_points), curve.degree
    pps = [PPoly.from_spline(BSpline(t, c[:, d], k)) for d in range(2)]
    x = pps[0].x
    edges = [i for i in range(1, len(x) - 1) if x[i] > x[i - 1] and x[i + 1] > x[i] and 0.0 < x[i] < 1.0]

    def sides(order):
        left, right = [], []
        for pp in pps:
            dd = pp.derivative(order) if order else pp
            left.append([np.polyval(dd.c[:, i - 1], x[i] - x[i - 1]) for i in edges])
            right.append([np.polyval(dd.c[:, i], 0.0) for i in edges])
        return np.array(left).T, np.array(right).T

    out = {}
    values = {o: sides(o) for o in range(4)}
    for o, name in ((0, "X"), (1, "X'"), (2, "X''"), (3, "X'''")):
        out[name] = float(np.max(np.hypot(*(values[o][0] - values[o][1]).T))) if edges else 0.0

    def derived(lr):
        d1, d2 = values[1][lr], values[2][lr]
        sp = np.hypot(d1[:, 0], d1[:, 1])
        kap = (d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]) / sp ** 3
        slope = d1[:, 1] / d1[:, 0]
        second = (d2[:, 1] * d1[:, 0] - d1[:, 1] * d2[:, 0]) / d1[:, 0] ** 3
        return kap, slope, second

    (kl, sl, ql), (kr, sr, qr) = derived(0), derived(1)
    out["kappa"] = float(np.max(np.abs(kl - kr))) if edges else 0.0
    out["dy/dx"] = float(np.max(np.abs(sl - sr))) if edges else 0.0
    out["d2y/dx2"] = float(np.max(np.abs(ql - qr) / np.maximum(1.0, np.abs(ql)))) if edges else 0.0   # relative where large
    out["interior_knots"] = len(edges)
    return out


def table_station_x_c(variant: str):
    """x/c of the suction and pressure surface points at the table stations (legacy PCHIP
    construction; the candidates are exact at the stations, D within 2.5e-6 c)."""
    from scipy.optimize import brentq

    from potential_flow_solver import geometry as g

    stations = np.array(g._TABLE_X_PCT[1:-1]) / 100.0
    stations = stations[stations >= g.FIRST_TABLE_STATION]

    def half(st):
        return g._half_thickness(st, T_MAX, "pchip_sqrt")

    out = {}
    for name, side in (("suction", 1.0), ("pressure", -1.0)):
        xs = []
        for st in stations:
            xc = brentq(lambda q: g.camberline_station(q, CL0) - st, g.XI_MIN, 1.0 - g.XI_MIN, xtol=1e-14)
            xs.append(float(g._surface(np.array([xc]), CL0, half, side)[0, 0]))
        out[name] = np.array(xs)
    out["stations"] = stations
    return out


def trend_fit(x, q, window=RIPPLE_WINDOW, degree=TREND_DEGREE):
    inside = (x >= window[0]) & (x <= window[1]) & np.isfinite(q)
    return np.polyfit(x[inside], q[inside], degree)


def ripple(x, q, reference_coefficients, window=RIPPLE_WINDOW):
    """q minus the common low-frequency reference (the polynomial fitted to the PRODUCTION curve)
    inside window; NaN outside."""
    out = np.full(len(x), np.nan)
    inside = (x >= window[0]) & (x <= window[1]) & np.isfinite(q)
    out[inside] = q[inside] - np.polyval(reference_coefficients, x[inside])
    return out


def uniform(x, q, lo, hi, n=4001):
    xs = np.linspace(lo, hi, n)
    order = np.argsort(x)
    return xs, np.interp(xs, x[order], q[order])


def dominant_wavelength(x, q, lo, hi):
    xs, qs = uniform(x, q, lo, hi)
    qs = qs - np.nanmean(qs)
    spectrum = np.abs(np.fft.rfft(np.nan_to_num(qs) * np.hanning(len(qs))))
    freq = np.fft.rfftfreq(len(qs), d=xs[1] - xs[0])
    k = 1 + int(np.argmax(spectrum[1:]))
    return 1.0 / freq[k]


def steep_points(x, dkappa, lo, hi, threshold_fraction=0.35):
    """x/c of the local maxima of |dkappa/ds| above threshold_fraction of the largest in [lo, hi]."""
    inside = np.flatnonzero((x >= lo) & (x <= hi))
    a = np.abs(dkappa[inside])
    peak = np.nanmax(a)
    picks = [inside[i] for i in range(1, len(inside) - 1) if a[i] >= a[i - 1] and a[i] > a[i + 1] and a[i] > threshold_fraction * peak]
    return np.array([x[i] for i in picks])


# --- solver what-if ---------------------------------------------------------------------------------------------

def solve_job(job: dict) -> dict:
    """One M1 0.65 solve on candidate geometry (built inside the worker, a diagnostic
    registration taken out before the solve)."""
    warnings.simplefilter("ignore")
    import audit_ss_sawtooth as saw
    from potential_flow_solver import controller as ct
    from potential_flow_solver import solver as sv

    settings = saw.base_settings(0.65, job["level"])
    case = ct.build_case(settings)
    cascade = build_cascade(job["candidate"], SOLVER_VARIANT)
    sol = sv.solve(cascade, case.config, c_ref=settings.chord)
    g = sol.grid
    f, l = g.first_blade_i, g.last_blade_i
    from potential_flow_solver import postprocess as pp

    s = pp.blade_surfaces(sol)
    mach = np.asarray(sol.mach)
    return dict(candidate=job["candidate"], level=job["level"], x_c_ss=np.asarray(s.x_c_ss), mach_ss=np.asarray(s.mach_ss),
                x_c_ps=np.asarray(s.x_c_ps), mach_ps=np.asarray(s.mach_ps), beta_out=math.degrees(sol.exit_angle),
                kutta=sol.kutta_residual, eps_k=sol.normalized_kutta_residual, passes=len(sol.passes),
                converged=sol.converged, peak_mach=float(np.max(mach)), n_m=len(g.m), n_eta=len(g.eta),
                dm_c=g.dm / g.chord, f=f, l=l)


def wall_ripple(x, m):
    import audit_ss_sawtooth as saw

    w = (x >= saw.WINDOW[0]) & (x <= saw.WINDOW[1])
    r = saw.residual(x, m)
    return float(np.sqrt(np.nanmean(r[w] ** 2))), r


# --- figures ----------------------------------------------------------------------------------------------------

def station_lines(ax, xs, alpha=0.35):
    for x in xs:
        ax.axvline(x, color="#e6550d", lw=0.5, alpha=alpha, zorder=0)


def figures(thick, curves, displacement, stations, production_runs, what_if, best, path: Path) -> None:
    import matplotlib.pyplot as plt

    x_tab, t_tab, _, _ = table()
    # Figure 1-3: thickness and its derivatives
    for number, key, ylabel, zoom in ((1, "t", "t / c", None), (2, "t1", "dt / ds", (-0.5, 0.6)),
                                      (3, "t2", "d2t / ds2", (-3.0, 2.0))):
        fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), gridspec_kw=dict(width_ratios=(1.2, 1.0)))
        for ax, xlim in ((axes[0], (0.0, 1.0)), (axes[1], (0.15, 0.9))):
            for name in CANDIDATES:
                s, t, t1, t2 = thick[name]["profile"]
                q = dict(t=t, t1=t1, t2=t2)[key]
                ax.plot(s, q, "-", lw=1.6 if name == "A_pchip" else 1.0, color=INK[name], label=name)
            if key == "t":
                ax.plot(x_tab, t_tab, "ko", ms=4, mfc="white", label="Table 4-1 (corrected)", zorder=5)
            station_lines(ax, x_tab)
            ax.set_xlim(*xlim)
            if zoom and xlim[0] > 0.0:
                ax.set_ylim(*zoom)
            ax.set_xlabel("station s / S (thickness is read along the camber line)")
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.25)
        if key == "t":
            axes[1].set_ylim(0.04, 0.105)
        if key != "t":
            axes[0].set_ylim(*(np.array(zoom) * (3.0 if key == "t2" else 2.0)))
        axes[0].legend(fontsize=8)
        axes[0].set_title("whole chord", fontsize=10)
        axes[1].set_title("station 0.15 .. 0.90 (table spacing 0.05)", fontsize=10)
        fig.suptitle(f"Figure {number}: thickness reconstruction, {ylabel} (orange lines: table stations)", fontsize=11)
        fig.tight_layout()
        fig.savefig(path / f"fig{number}_thickness{'' if key == 't' else '_d' + key[1]}.png", dpi=125)
        plt.close(fig)
    # Figure 4-5: curvature of the final surfaces
    for number, surface in ((4, "suction"), (5, "pressure")):
        fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
        for name in CANDIDATES:
            p = curves[(name, SOLVER_VARIANT, surface)]["profile"]
            axes[0].plot(p["x"], p["kappa"], "-", lw=1.6 if name == "A_pchip" else 1.0, color=INK[name], label=name)
            axes[1].plot(p["x"], p["dkappa_ds"], "-", lw=1.2 if name == "A_pchip" else 0.8, color=INK[name], label=name)
        for ax in axes:
            station_lines(ax, stations[SOLVER_VARIANT][surface])
            ax.grid(alpha=0.25)
        inside = (p["x"] > 0.1) & (p["x"] < 0.95)
        ref = curves[("A_pchip", SOLVER_VARIANT, surface)]["profile"]["kappa"][inside]
        axes[0].set_ylim(np.nanmin(ref) - 0.4, np.nanmax(ref) + 0.4)
        axes[1].set_ylim(-150, 150)
        axes[0].set_xlim(0.05, 1.0)
        axes[0].set_ylabel("kappa [1/c]")
        axes[1].set_ylabel("dkappa / ds [1/c^2]")
        axes[1].set_xlabel("x / c (blade frame)")
        axes[0].legend(fontsize=8, ncol=3)
        fig.suptitle(f"Figure {number}: {surface}-surface curvature of the final C2 curves ({SOLVER_VARIANT}); orange lines: "
                     f"the surface points of the table stations", fontsize=11)
        fig.tight_layout()
        fig.savefig(path / f"fig{number}_{surface}_curvature.png", dpi=125)
        plt.close(fig)
    # Figure 6: surfaces and displacement
    fig, axes = plt.subplots(2, 1, figsize=(14, 9))
    p0s = curves[("A_pchip", SOLVER_VARIANT, "suction")]["profile"]
    p0p = curves[("A_pchip", SOLVER_VARIANT, "pressure")]["profile"]
    axes[0].plot(p0s["x"], p0s["y"], "-", color="#252525", lw=1.2, label="production suction")
    axes[0].plot(p0p["x"], p0p["y"], "-", color="#636363", lw=1.2, label="production pressure")
    for name in CANDIDATES[1:]:
        for surface, ls in (("suction", "-"), ("pressure", "--")):
            d = displacement[(name, SOLVER_VARIANT, surface)]
            axes[1].plot(d["x"], d["normal"] * 1e6, ls, color=INK[name], lw=1.0, label=f"{name}, {surface}")
    axes[0].set_aspect("equal")
    axes[0].legend(fontsize=8)
    axes[0].set_title("blade, production geometry (the candidates are indistinguishable at this scale)", fontsize=10)
    axes[1].set_ylabel("normal displacement from production [1e-6 c]")
    axes[1].set_xlabel("x / c")
    axes[1].set_ylim(-140, 140)
    axes[1].legend(fontsize=7, ncol=3)
    station_lines(axes[1], stations[SOLVER_VARIANT]["suction"])
    axes[1].grid(alpha=0.25)
    fig.suptitle(f"Figure 6: final surfaces and the candidates' displacement from production ({SOLVER_VARIANT})", fontsize=11)
    fig.tight_layout()
    fig.savefig(path / "fig6_surface_displacement.png", dpi=125)
    plt.close(fig)
    # Figure 7: curvature ripple
    fig, axes = plt.subplots(2, 1, figsize=(14, 8.5), sharex=True)
    for ax, surface in zip(axes, ("suction", "pressure")):
        for name in CANDIDATES:
            c = curves[(name, SOLVER_VARIANT, surface)]
            ax.plot(c["profile"]["x"], c["ripple"], "-", lw=1.4 if name == "A_pchip" else 0.9, color=INK[name], label=name)
        station_lines(ax, stations[SOLVER_VARIANT][surface])
        ax.set_ylabel(f"{surface}: kappa - reference [1/c]")
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8, ncol=3)
    axes[1].set_xlabel("x / c")
    axes[1].set_xlim(*RIPPLE_WINDOW)
    fig.suptitle("Figure 7: curvature ripple, kappa minus the common low-frequency reference (degree-"
                 f"{TREND_DEGREE} fit of the production curvature over x/c {RIPPLE_WINDOW})", fontsize=11)
    fig.tight_layout()
    fig.savefig(path / "fig7_curvature_ripple.png", dpi=125)
    plt.close(fig)
    # Figure 8: SS Mach, production vs best
    if what_if and best:
        fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
        for ax, level in zip(axes, SOLVER_LEVELS):
            prod = production_runs.get(level)
            cand = what_if.get((best, level))
            if prod is not None:
                ax.plot(prod["x_c_ss"], prod["mach_ss"], "o-", ms=3, lw=0.8, color=INK["A_pchip"], label="production (PCHIP)")
            if cand is not None:
                ax.plot(cand["x_c_ss"], cand["mach_ss"], "o-", ms=3, lw=0.8, color=INK[best], label=best)
            ax.set_title(f"{level}", fontsize=10)
            ax.set_xlabel("x / c")
            ax.set_ylabel("suction-wall Mach number")
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)
        fig.suptitle(f"Figure 8: suction-surface Mach at M1 0.65, production against {best} (geometry only changes)",
                     fontsize=11)
        fig.tight_layout()
        fig.savefig(path / "fig8_ss_mach_best_candidate.png", dpi=125)
        plt.close(fig)


def causality_figure(thick, curves, stations, production_runs, what_if, best, path: Path) -> None:
    import matplotlib.pyplot as plt

    import audit_ss_sawtooth as saw

    fig, axes = plt.subplots(5, 1, figsize=(13, 15), sharex=True)
    xs_ss = stations[SOLVER_VARIANT]["suction"]
    s, _, _, t2 = thick["A_pchip"]["profile"]
    axes[0].plot(s, t2, "-", color=INK["A_pchip"], lw=1.2, label="production (PCHIP): d2t/ds2")
    if best:
        s_b, _, _, t2_b = thick[best]["profile"]
        axes[0].plot(s_b, t2_b, "-", color=INK[best], lw=1.4, label=f"{best}: d2t/ds2")
    axes[0].set_ylim(-3.0, 1.5)
    axes[0].set_ylabel("d2t/ds2")
    axes[0].set_title("1-2. table stations (vertical lines) and the thickness second derivative (vs station)", fontsize=10)
    p = curves[("A_pchip", SOLVER_VARIANT, "suction")]["profile"]
    axes[1].plot(p["x"], p["kappa"], "-", color=INK["A_pchip"], lw=1.2, label="production SS curvature")
    if best:
        pb = curves[(best, SOLVER_VARIANT, "suction")]["profile"]
        axes[2].plot(pb["x"], pb["kappa"], "-", color=INK[best], lw=1.4, label=f"{best} SS curvature")
    for ax in axes[1:3]:
        inside = (p["x"] > 0.15) & (p["x"] < 0.9)
        ax.set_ylim(np.nanmin(p["kappa"][inside]) - 0.2, np.nanmax(p["kappa"][inside]) + 0.2)
        ax.set_ylabel("kappa [1/c]")
    axes[1].set_title("3. final production suction-surface curvature", fontsize=10)
    axes[2].set_title(f"5. best candidate ({best}) suction-surface curvature", fontsize=10)
    for k, (level, ink) in enumerate((("medium", "#3182bd"), ("very_fine", "#08306b"))):
        prod = production_runs.get(level)
        if prod is not None:
            _, r = wall_ripple(prod["x_c_ss"], prod["mach_ss"])
            axes[3].plot(prod["x_c_ss"], r, "o-", ms=2.5, lw=0.9, color=ink, label=f"production, {level}")
            cand = what_if.get((best, level)) if best else None
            if cand is not None:
                _, rc = wall_ripple(cand["x_c_ss"], cand["mach_ss"])
                axes[3].plot(cand["x_c_ss"], rc, "s--", ms=2.5, lw=0.9, color=INK[best], alpha=0.6 + 0.4 * k,
                             label=f"{best}, {level}")
    axes[3].set_ylabel("Mach - trend")
    axes[3].set_title("4. suction-wall Mach ripple (wall Mach minus its smooth trend), production and best candidate",
                      fontsize=10)
    for level, ink, marker in (("medium", INK["A_pchip"], "o"), ("very_fine", "#a63603", "s")):
        prod = production_runs.get(level)
        if prod is not None:
            axes[4].plot(prod["x_c_ss"], prod["mach_ss"], marker + "-", ms=2.5, lw=0.8, color=ink, label=f"production, {level}")
        cand = what_if.get((best, level)) if best else None
        if cand is not None:
            axes[4].plot(cand["x_c_ss"], cand["mach_ss"], marker + "-", ms=2.5, lw=0.8, color=INK[best] if level == "medium"
                         else "#3f007d", label=f"{best}, {level}")
    axes[4].set_ylabel("suction-wall Mach")
    axes[4].set_title("6. suction-wall Mach at M1 0.65", fontsize=10)
    axes[4].set_xlabel("x / c (table stations: their suction-surface points)")
    axes[4].set_xlim(0.1, 0.9)
    station_lines(axes[0], stations[SOLVER_VARIANT]["stations"], 0.5)   # panel 1 is against the station itself
    for ax in axes[1:]:
        station_lines(ax, xs_ss, 0.5)
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, loc="best")
    fig.suptitle("Geometry ripple causality: table stations -> thickness d2t/ds2 -> SS curvature -> SS Mach", fontsize=12)
    fig.tight_layout()
    fig.savefig(path / "geometry_ripple_causality.png", dpi=125)
    plt.close(fig)


def write_csv(name: str, rows: list) -> None:
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(OUT / name, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


# --- main -------------------------------------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="NACA 65 thickness reconstruction audit (diagnostic only)")
    parser.add_argument("--no-solve", action="store_true")
    parser.add_argument("--reuse-solver", action="store_true", help="load the solver what-if of the last run (what_if.pkl)")
    args = parser.parse_args(argv)
    import matplotlib

    matplotlib.use("Agg")
    warnings.simplefilter("ignore")
    from potential_flow_solver import geometry as g

    OUT.mkdir(parents=True, exist_ok=True)
    production_objects = production_state()

    # 1. the table and the thickness candidates
    x_tab, t_tab, z_tab, h_tab = table()
    say(f"TABLE: {g.TABLE_SOURCE}; {len(x_tab)} stations; corrections {g.TABLE_CORRECTIONS} (printed 3.480 % at 2.5 %); "
        f"half-thickness = t / 2, interpolated in z = sqrt(station); station = s / S along the camber line "
        f"(STATION_READING {g.STATION_READING!r}); scaled by t_max / {g.TABLE_T_MAX}")
    say("  x/c  " + " ".join(f"{v:.4f}" for v in x_tab))
    say("  t/c  " + " ".join(f"{v:.5f}" for v in t_tab))
    say()
    thick, summary, error_rows = {}, [], []
    for name in CANDIDATES:
        rec = candidate(name)
        checks = thickness_checks(name, rec)
        cont = knot_continuity(rec, (0, 1, 2, 3))
        thick[name] = dict(rec=rec, profile=thickness_profile(rec), checks=checks, continuity=cont)
        for xi, ti, ei in zip(x_tab, t_tab, checks["errors"]):
            error_rows.append(dict(candidate=name, x_c=xi, t_table=ti, t_candidate=ti + ei, error=ei))
    ch0 = thick["A_pchip"]["checks"]
    say(f"three-point quadratic through the table's largest ordinate and its neighbours: peak t/c "
        f"{ch0['three_point_quadratic_peak']:.6f} at station {ch0['three_point_quadratic_x']:.4f} (the table maximum "
        f"{np.max(t_tab):.5f} sits at station 0.40)")
    lam = smoothing_penalty(z_tab, h_tab, 0.5 * TABLE_TOLERANCE)
    say(f"D_smoothing_cubic penalty {lam:.4e} (largest keeping |t - t_i| <= {TABLE_TOLERANCE:g} c)")
    write_csv("table_fit_errors.csv", error_rows)

    # 2. geometry chain for every candidate and variant
    stations = {v: table_station_x_c(v) for v in VARIANTS}
    cascades = {}
    for v in VARIANTS:
        for name in CANDIDATES:
            cascades[(name, v)] = build_cascade(name, v)
    assert production_state() == production_objects, "production state not restored"
    for name, reconstruction in PRODUCTION_NAMES.items():
        check = g.make_cascade(g.CascadeSpec(blade_type="NACA65", solidity=SOLIDITY, stagger_deg=STAGGER, cl0=CL0,
                                             t_max_over_c=T_MAX, profile_variant=SOLVER_VARIANT, surface_continuity="C2",
                                             thickness_reconstruction=reconstruction))
        same = np.array_equal(check.active_curves.suction.control_points,
                              cascades[(name, SOLVER_VARIANT)].active_curves.suction.control_points)
        say(f"geometry {reconstruction!r} after the candidate builds: bitwise equal to {name} of the audit: {same}")
    curves, displacement, metric_rows, disp_rows, cont_rows = {}, {}, [], [], []
    for v in VARIANTS:
        for surface in ("suction", "pressure"):
            prod_curve = getattr(cascades[("A_pchip", v)].normalized_curves, f"{surface}_unstaggered")
            prod = curve_profile(prod_curve)
            ref_k = trend_fit(prod["x"], prod["kappa"])
            for name in CANDIDATES:
                cur = getattr(cascades[(name, v)].normalized_curves, f"{surface}_unstaggered")
                p = prod if name == "A_pchip" else curve_profile(cur)
                r = ripple(p["x"], p["kappa"], ref_k)
                curves[(name, v, surface)] = dict(profile=p, ripple=r)
                row = dict(variant=v, surface=surface, candidate=name)
                w = (p["x"] >= RIPPLE_WINDOW[0]) & (p["x"] <= RIPPLE_WINDOW[1])
                row["kappa_ripple_rms_window"] = float(np.sqrt(np.nanmean(r[w] ** 2)))
                row["dkappa_ds_rms_window"] = float(np.sqrt(np.nanmean(p["dkappa_ds"][w] ** 2)))
                anchor_u = cascades[(name, v)].metadata.get(f"{surface[0]}s_le_anchor_u")
                anchor_x = float(cur.evaluate(anchor_u)[0])
                row["le_anchor_x_c"] = anchor_x
                for region, lo, hi in REGIONS:
                    if region == "LE table":
                        lo = anchor_x + LE_FAIR_MARGIN
                    inside = (p["x"] >= lo) & (p["x"] <= hi)
                    row[f"dkappa_ds_rms {region}"] = float(np.sqrt(np.nanmean(p["dkappa_ds"][inside] ** 2)))
                    row[f"dkappa_ds_max {region}"] = float(np.nanmax(np.abs(p["dkappa_ds"][inside])))
                fair = (p["x"] >= 0.0) & (p["x"] < anchor_x + LE_FAIR_MARGIN) & np.isfinite(p["dkappa_ds"])
                row["dkappa_ds_rms LE fair (information)"] = float(np.sqrt(np.nanmean(p["dkappa_ds"][fair] ** 2)))
                row["dominant_wavelength_window"] = float(dominant_wavelength(p["x"][w], r[w], *RIPPLE_WINDOW))
                metric_rows.append(row)
                cont = curve_continuity(cur)
                cont_rows.append(dict(variant=v, surface=surface, candidate=name, **cont))
                # displacement: signed normal distance of the candidate's points from the production curve
                u = np.linspace(0.0, 1.0, 4001)
                pts = cur.evaluate(u)
                _, dist = prod_curve.project(pts)
                xn = pts[:, 0]
                displacement[(name, v, surface)] = dict(x=xn, normal=dist)
                drow = dict(variant=v, surface=surface, candidate=name, max_abs_normal=float(np.max(np.abs(dist))),
                            rms_normal=float(np.sqrt(np.mean(dist ** 2))))
                for region, lo, hi in REGIONS:
                    inside = (xn >= lo) & (xn <= hi)
                    drow[f"max_abs {region}"] = float(np.max(np.abs(dist[inside]))) if inside.any() else float("nan")
                disp_rows.append(drow)
    write_csv("curvature_metrics.csv", metric_rows)
    say("AGREEMENT of the smooth reconstructions with each other against their difference from PCHIP (max |normal "
        "displacement difference| [c] at equal x/c, interpolated):")
    for v in VARIANTS:
        for surface in ("suction", "pressure"):
            grid_x = np.linspace(0.01, 0.99, 3001)
            d = {}
            for name in CANDIDATES[1:]:
                dd = displacement[(name, v, surface)]
                keep = (dd["x"] > 0.005) & (dd["x"] < 0.995)
                order = np.argsort(dd["x"][keep])
                d[name] = np.interp(grid_x, dd["x"][keep][order], dd["normal"][keep][order])
            for region, lo, hi in REGIONS[1:4]:
                inside = (grid_x >= lo) & (grid_x <= hi)
                spread = max(float(np.max(np.abs(d[a][inside] - d[b][inside]))) for a in d for b in d if a < b)
                from_pchip = min(float(np.max(np.abs(d[a][inside]))) for a in d)
                say(f"  {v:17s} {surface:8s} {region:13s}: largest difference between smooth candidates "
                    f"{spread:.2e} c; smallest of their differences from PCHIP {from_pchip:.2e} c")
    write_csv("geometry_displacements.csv", disp_rows)
    write_csv("continuity.csv", cont_rows)
    # thickness ripple metric: t'' against the polynomial of the production t''
    s0, _, _, t2_0 = thick["A_pchip"]["profile"]
    ref_t2 = trend_fit(s0, t2_0)
    for name in CANDIDATES:
        s, _, _, t2 = thick[name]["profile"]
        r = ripple(s, t2, ref_t2)
        w = (s >= RIPPLE_WINDOW[0]) & (s <= RIPPLE_WINDOW[1])
        thick[name]["t2_ripple_rms"] = float(np.sqrt(np.nanmean(r[w] ** 2)))

    # 3. the C1 source: curvature jumps of the reference surfaces at the table stations and their C2 ramp widths
    ref_ss = g.naca65_reference_surfaces(CL0, T_MAX, profile_variant=SOLVER_VARIANT,
                                         thickness_reconstruction="pchip_sqrt")[0]
    transitions = g._junction_transitions(ref_ss)
    say()
    say("C1 SOURCE (production): curvature jump of the C1 suction surface at each junction and the half-width a of the "
        "C2 target's kernel ramp (SmoothedSurface):")
    for tr in transitions:
        lab = tr.junction.label
        say(f"  {lab:40s} kappa_jump {tr.junction.kappa_jump:+8.4f} /c  turn {math.degrees(tr.junction.turn):+.2e} deg  "
            f"a {tr.half_width:.5f} c")
    say()

    # 4. summary per candidate
    for name in CANDIDATES:
        ch, co = thick[name]["checks"], thick[name]["continuity"]
        row = dict(candidate=name, max_abs_table_error=ch["max_abs_table_error"], table_exact=ch["table_exact"],
                   within_table_precision=ch["within_tolerance"], t_max=ch["t_max"], x_t_max=ch["x_t_max"],
                   min_thickness_interior=ch["min_thickness_interior"], positive=ch["positive"],
                   single_maximum=ch["single_maximum"],
                   wrong_slope_samples=ch["wrong_slope_samples_in_monotone_intervals"], peak_excess=ch["peak_excess_over_table"],
                   quadratic_peak=ch["three_point_quadratic_peak"], quadratic_peak_x=ch["three_point_quadratic_x"],
                   le_slope=ch["le_slope"], le_slope_relative=ch["le_slope_relative"], te_h_zz=ch["te_curvature"],
                   te_h_zz_wedge=ch["te_curvature_wedge"], te_h_zz_relative=ch["te_curvature_relative"],
                   knot_jump_h=co[0], knot_jump_h1=co[1], knot_jump_h2=co[2], knot_jump_h3=co[3],
                   t2_ripple_rms=thick[name]["t2_ripple_rms"])
        for v in VARIANTS:
            for surface in ("suction", "pressure"):
                m = next(r for r in metric_rows if r["variant"] == v and r["surface"] == surface and r["candidate"] == name)
                d = next(r for r in disp_rows if r["variant"] == v and r["surface"] == surface and r["candidate"] == name)
                c = next(r for r in cont_rows if r["variant"] == v and r["surface"] == surface and r["candidate"] == name)
                tag = f"{v[:5]}_{surface[:2]}"
                row[f"{tag}_kappa_ripple"] = m["kappa_ripple_rms_window"]
                row[f"{tag}_dkds_rms"] = m["dkappa_ds_rms_window"]
                row[f"{tag}_max_disp"] = d["max_abs_normal"]
                row[f"{tag}_c2_kappa_jump"] = c["kappa"]
        cas = cascades[(name, SOLVER_VARIANT)]
        for side in ("ss", "ps"):
            for key in ("le_anchor_over_c", "le_fairness"):
                row[f"{side}_{key}"] = cas.metadata.get(f"{side}_{key}")
        summary.append(row)
    write_csv("candidate_summary.csv", summary)
    say("CANDIDATES (thickness; emery_base and kovach final curves): table fidelity, physical checks, continuity, ripple, "
        "displacement")
    for row in summary:
        say(f"  {row['candidate']:20s} max|e| {row['max_abs_table_error']:.2e}  exact {row['table_exact']!s:5s} "
            f"t_max {row['t_max']:.6f} at s {row['x_t_max']:.4f}  min t {row['min_thickness_interior']:.2e}  "
            f"positive {row['positive']}  single max {row['single_maximum']}  wrong-slope samples {row['wrong_slope_samples']}  "
            f"peak above the table max {row['peak_excess']:+.2e}")
        say(f"  {'':20s} ends: dh/dz(0) {row['le_slope']:.5f} (parabolic nose sqrt(2 R_LE) {math.sqrt(2.0 * g.TABLE_LE_RADIUS):.5f},"
            f" off {row['le_slope_relative']:.1%}); h_zz(1) {row['te_h_zz']:+.4f} (the last three table points imply "
            f"{row['te_h_zz_wedge']:+.4f}; information only)")
        say(f"  {'':20s} knot jumps h {row['knot_jump_h']:.1e} h' {row['knot_jump_h1']:.1e} h'' {row['knot_jump_h2']:.2e} "
            f"h''' {row['knot_jump_h3']:.2e}; t'' ripple rms {row['t2_ripple_rms']:.4f}")
        for v in VARIANTS:
            for surface in ("suction", "pressure"):
                tag = f"{v[:5]}_{surface[:2]}"
                say(f"  {'':20s} {v:17s} {surface:8s} kappa ripple rms {row[tag + '_kappa_ripple']:.4f} /c  dkappa/ds rms "
                    f"{row[tag + '_dkds_rms']:8.2f}  max |displacement| {row[tag + '_max_disp']:.2e} c  C2 kappa knot "
                    f"mismatch {row[tag + '_c2_kappa_jump']:.1e}")
    say("LE FAIR (emery_base) per candidate: anchor arc length past the apex [c] and fairness R_LE^3 int kappa_s^2 ds:")
    for row in summary:
        say(f"  {row['candidate']:20s} SS anchor {row['ss_le_anchor_over_c']:.5f} fairness {row['ss_le_fairness']:.4f}   "
            f"PS anchor {row['ps_le_anchor_over_c']:.5f} fairness {row['ps_le_fairness']:.4f}")
    say()
    say("FINAL-CURVE CONTINUITY (max |left - right| at the interior knots of the quintic B-splines; the curve parameter u):")
    for r in cont_rows:
        say(f"  {r['variant']:17s} {r['surface']:8s} {r['candidate']:20s} knots {r['interior_knots']:4d}  X {r['X']:.1e}  "
            f"X' {r[chr(88) + chr(39)]:.1e}  X'' {r[chr(88) + chr(39) * 2]:.1e}  X''' {r[chr(88) + chr(39) * 3]:.1e}  "
            f"kappa {r['kappa']:.1e}  dy/dx {r['dy/dx']:.1e}  d2y/dx2 {r['d2y/dx2']:.1e}")
    say()
    say("REGIONAL ROUGHNESS of the suction curvature (rms / max |dkappa/ds| [1/c^2], production vs candidates, "
        f"{SOLVER_VARIANT}; 'LE table' from {LE_FAIR_MARGIN} c past the LE-fair anchor to x/c 0.05):")
    for name in CANDIDATES:
        m = next(r for r in metric_rows if r["variant"] == SOLVER_VARIANT and r["surface"] == "suction" and r["candidate"] == name)
        say(f"  {name:20s} " + "  ".join(f"{region}: {m[f'dkappa_ds_rms {region}']:.1f}/{m[f'dkappa_ds_max {region}']:.0f}"
                                         for region, _, _ in REGIONS))
    say()

    # 5. table-station frequency test (production)
    p = curves[("A_pchip", SOLVER_VARIANT, "suction")]["profile"]
    table_ss = stations[SOLVER_VARIANT]["suction"]
    say("TABLE-STATION FREQUENCY (production suction surface): at each table station's surface point, the largest "
        "|dkappa/ds| within +-0.012 c, where it lies, and the median |dkappa/ds| 0.015 .. 0.035 c away")
    station_rows = []
    for st, xs_ in zip(stations[SOLVER_VARIANT]["stations"], table_ss):
        if not (RIPPLE_WINDOW[0] - 0.001 <= xs_ <= RIPPLE_WINDOW[1] + 0.001):
            continue
        near = np.abs(p["x"] - xs_) <= 0.012
        between = (np.abs(p["x"] - xs_) > 0.015) & (np.abs(p["x"] - xs_) < 0.035)
        k = np.flatnonzero(near)[int(np.argmax(np.abs(p["dkappa_ds"][near])))]
        station_rows.append(dict(station=float(st), x_c=float(xs_), peak=float(abs(p["dkappa_ds"][k])),
                                 offset=float(p["x"][k] - xs_), background=float(np.median(np.abs(p["dkappa_ds"][between])))))
    for r in station_rows:
        say(f"  station {r['station']:.2f} (x/c {r['x_c']:.4f}): peak |dkappa/ds| {r['peak']:7.1f} at {r['offset']:+.4f}, "
            f"background {r['background']:6.2f}, ratio {r['peak'] / r['background']:6.1f}")
    peaks_x = np.array([r["x_c"] + r["offset"] for r in station_rows])
    say(f"  spacing of the peaks: mean {np.diff(peaks_x).mean():.4f} c, range {np.diff(peaks_x).min():.4f} .. "
        f"{np.diff(peaks_x).max():.4f} c (table spacing 0.05 in station)")
    w = (p["x"] >= RIPPLE_WINDOW[0]) & (p["x"] <= RIPPLE_WINDOW[1])
    say(f"  dominant wavelength of the curvature ripple: {dominant_wavelength(p['x'][w], curves[('A_pchip', SOLVER_VARIANT, 'suction')]['ripple'][w], *RIPPLE_WINDOW):.4f} c")

    # 6. Mach correlation from the stored production runs
    production_runs = {}
    path = ROOT / SAWTOOTH_RUNS
    if path.exists():
        with open(path, "rb") as handle:
            runs = pickle.load(handle)
        for level in FIVE_LEVELS:
            r = runs[level]
            f, l = r["f"], r["l"]
            production_runs[level] = dict(x_c_ss=r["x_c_ss"], mach_ss=r["mach"][f:l + 1, 0],
                                          x_c_ps=None, mach_ps=r["mach"][f:l + 1, -1], beta_out=r["beta_out"],
                                          passes=r["passes"], n_m=r["n_m"], n_eta=r["n_eta"])
        vf = production_runs["very_fine"]
        _, mres = wall_ripple(vf["x_c_ss"], vf["mach_ss"])
        lo, hi = 0.25, 0.55
        kres = curves[("A_pchip", SOLVER_VARIANT, "suction")]["ripple"]
        # the solver's wall x/c is along the cascade chord; the blade-frame x is the same chord coordinate
        xs, ku = uniform(p["x"][np.isfinite(kres)], -kres[np.isfinite(kres)], lo, hi)
        ok = np.isfinite(mres)
        _, mu = uniform(vf["x_c_ss"][ok], mres[ok], lo, hi)
        lags = np.arange(-60, 61)
        step = xs[1] - xs[0]
        corr = [np.corrcoef(ku[max(0, -k):len(ku) - max(0, k)], mu[max(0, k):len(mu) - max(0, -k)])[0, 1] for k in lags]
        best_lag = int(lags[int(np.argmax(corr))])
        say(f"MACH CORRELATION (production very_fine, x/c {lo} .. {hi}): correlation of the wall-Mach ripple with "
            f"-(curvature ripple): {corr[60]:.3f} at zero lag, largest {max(corr):.3f} at lag {best_lag * step:+.4f} c; "
            f"Mach-ripple dominant wavelength {dominant_wavelength(vf['x_c_ss'][ok], mres[ok], lo, hi):.4f} c "
            f"(node spacing {np.median(np.diff(vf['x_c_ss'])):.4f})")
    say()

    # 7. geometry ranking (no Mach criterion)
    ranking = []
    for row in summary:
        fidelity = row["table_exact"] or row["within_table_precision"]
        ends_ok = row["le_slope_relative"] <= END_TOLERANCE
        fair_ok = all(row[f"{side}_le_fairness"] <= 1.10 * summary[0][f"{side}_le_fairness"] for side in ("ss", "ps"))
        physical = (row["positive"] and row["single_maximum"] and row["wrong_slope_samples"] == 0 and ends_ok
                    and fair_ok)
        c2 = all(row[f"{v[:5]}_{s[:2]}_c2_kappa_jump"] < 1e-6 for v in VARIANTS for s in ("suction", "pressure"))
        worse = []
        for v in VARIANTS:
            for surf in ("suction", "pressure"):
                m = next(r for r in metric_rows if r["variant"] == v and r["surface"] == surf and r["candidate"] == row["candidate"])
                m0 = next(r for r in metric_rows if r["variant"] == v and r["surface"] == surf and r["candidate"] == "A_pchip")
                for region, _, _ in REGIONS:
                    ratio = m[f"dkappa_ds_rms {region}"] / m0[f"dkappa_ds_rms {region}"]
                    if ratio > 1.10:
                        worse.append(f"{v} {surf} {region} x{ratio:.2f}")
        ripple_ss = row[f"{SOLVER_VARIANT[:5]}_su_kappa_ripple"]
        disp = max(row[f"{v[:5]}_{s[:2]}_max_disp"] for v in VARIANTS for s in ("suction", "pressure"))
        ranking.append(dict(candidate=row["candidate"], fidelity=fidelity, physical=physical, c2=c2, ripple=ripple_ss,
                            displacement=disp, rougher_regions=worse, ends_ok=ends_ok, fair_ok=fair_ok,
                            acceptable=fidelity and physical and c2 and not worse))
    exact = {row["candidate"]: row["table_exact"] for row in summary}
    order = sorted((r for r in ranking if r["acceptable"] and r["candidate"] != "A_pchip"),
                   key=lambda r: (not exact[r["candidate"]], round(r["ripple"], 3), r["displacement"]))
    say("GEOMETRY RANKING (priorities: table fidelity, physical checks, C2 quality, no region rougher than production by "
        "more than 10 % (rms dkappa/ds, both surfaces, both variants), curvature ripple, displacement; no Mach):")
    for r in ranking:
        say(f"  {r['candidate']:20s} fidelity {r['fidelity']!s:5s} physical {r['physical']!s:5s} C2 {r['c2']!s:5s} "
            f"SS ripple {r['ripple']:.4f} /c  max displacement {r['displacement']:.2e} c  -> "
            f"{'acceptable' if r['acceptable'] else 'REJECTED'}"
            + ("" if r["ends_ok"] else "; dh/dz(0) departs from the parabolic nose slope by more than 10 %")
            + ("" if r["fair_ok"] else "; the LE fair's fairness worsens by more than 10 %")
            + (f"; rougher than production (>10 %) in: {', '.join(r['rougher_regions'])}" if r["rougher_regions"] else ""))
    chosen = [r["candidate"] for r in order[:2]]
    best = chosen[0] if chosen else None
    say(f"  solver what-if candidates (geometry criteria only): {chosen}")
    say()

    # 8. solver what-if
    what_if = {}
    if not args.no_solve and chosen:
        saved = OUT / "what_if.pkl"
        if args.reuse_solver and saved.exists():
            with open(saved, "rb") as handle:
                what_if = pickle.load(handle)
        else:
            jobs = [dict(candidate="A_pchip", level="medium")]
            jobs += [dict(candidate=c, level=lv) for c in chosen for lv in SOLVER_LEVELS]
            jobs += [dict(candidate=best, level=lv) for lv in FIVE_LEVELS if lv not in SOLVER_LEVELS]
            with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
                for res in pool.map(solve_job, jobs):
                    what_if[(res["candidate"], res["level"])] = res
            with open(saved, "wb") as handle:
                pickle.dump(what_if, handle)
        rows = []
        for (name, level), res in sorted(what_if.items(), key=lambda kv: (kv[0][0], FIVE_LEVELS.index(kv[0][1]))):
            ss_r, _ = wall_ripple(res["x_c_ss"], res["mach_ss"])
            ps_r, _ = wall_ripple(res["x_c_ps"], res["mach_ps"])
            rows.append(dict(candidate=name, level=level, grid=f"{res['n_m']}x{res['n_eta']}", dm_c=res["dm_c"],
                             converged=res["converged"], passes=res["passes"], beta_out_deg=res["beta_out"],
                             kutta_residual=res["kutta"], eps_k=res["eps_k"], peak_field_mach=res["peak_mach"],
                             ss_peak=float(np.max(res["mach_ss"])), ss_ripple_rms=ss_r, ps_ripple_rms=ps_r))
        for level, prod in production_runs.items():
            ss_r, _ = wall_ripple(prod["x_c_ss"], prod["mach_ss"])
            rows.append(dict(candidate="A_pchip (stored, ss_sawtooth_audit)", level=level,
                             grid=f"{prod['n_m']}x{prod['n_eta']}", beta_out_deg=prod["beta_out"], passes=prod["passes"],
                             ss_peak=float(np.max(prod["mach_ss"])), ss_ripple_rms=ss_r))
        write_csv("solver_what_if.csv", rows)
        say("SOLVER WHAT-IF (M1 0.65, emery_base, production solver; only the thickness reconstruction differs):")
        for r in rows:
            say(f"  {r['candidate']:38s} {r['level']:12s} {r.get('grid', ''):9s} beta_out {r['beta_out_deg']:.4f}  "
                f"passes {r.get('passes', '-')}  SS peak {r['ss_peak']:.4f}  SS ripple {r['ss_ripple_rms']:.5f}  "
                f"PS ripple {r.get('ps_ripple_rms', float('nan')):.5f}  eps_K {r.get('eps_k', float('nan')):.2e}  "
                f"peak M {r.get('peak_field_mach', float('nan')):.4f}")
        if ("A_pchip", "medium") in what_if and "medium" in production_runs:
            same = np.array_equal(what_if[("A_pchip", "medium")]["mach_ss"], production_runs["medium"]["mach_ss"])
            say(f"  A_pchip rebuilt through the context reproduces the stored production medium wall Mach bit for bit: {same}")
        say()

    figures(thick, curves, displacement, stations, production_runs, what_if, best, OUT)
    causality_figure(thick, curves, stations, production_runs, what_if, best, OUT)
    assert production_state() == production_objects
    (OUT / "report.txt").write_text("\n".join(LINES) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
