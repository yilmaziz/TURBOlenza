"""
audit_ss_sawtooth.py
====================
Diagnosis only: where does the suction-side surface-Mach sawtooth of the NACA 65-(12A10)10 at
M1 = 0.65 come from? Production code as it is; nothing is changed or fixed.

    python scripts/audit_ss_sawtooth.py

The case is the one of scripts/compare_dunavant_aungier.py (emery_base, stagger 33 deg,
solidity 1, inlet 45 deg, lambda_rho = lambda_K = 0.7, tolerances 1e-8, the sonic stop on), on
the five standard grid levels of grid.py section 9. Further runs, each the production solver
with one thing changed and nothing patched:
    M030_medium      M1 = 0.30 (is it compressibility?)
    SMOOTH_<level>   DIAGNOSTIC GEOMETRY, NOT PRODUCTION: the suction curve with its curvature
                     ripple removed over x/c SMOOTH_RANGE (a normal offset of at most about
                     2e-4 c whose second derivative cancels the ripple; smoothed_cascade),
                     built in this script on a copy of the cascade; the pressure side, the LE
                     and TE and the solver are untouched. Does the sawtooth follow the
                     geometry? (A first attempt, averaging the control points, ADDED curvature
                     oscillations on this non-uniform knot vector and was discarded.)
Every run also keeps pass 0 (the first linear solve at the uniform start density): is the
sawtooth there before any density update?

Outputs (OUTPUT_DIR): figures (A) le_raw_nodes.png, te_raw_nodes.png, le_raw_quantities.png;
(B) station_differences.png; (C) ss_wall_sequences.png; (D) ss_five_grids.png; (E/F)
ss_causality.png; tables ss_wall_nodes.csv, station_differences.csv, sawtooth_metrics.csv and
report.txt.
"""

# ============================================================
# SETTINGS
# ============================================================

OUTPUT_DIR = "outputs/ss_sawtooth_audit"
N_WORKERS = 8
LEVELS = ("very_coarse", "coarse", "medium", "fine", "very_fine")
DETAIL_LEVEL = "medium"                 # the level of the raw-node and difference figures
SMOOTH_LEVELS = ("medium", "very_fine")
SMOOTH_RANGE = (0.12, 0.88)             # x/c where the diagnostic geometry removes the curvature ripple
SMOOTH_RAMP = 0.04                      # x/c of the cosine ramps at both ends of that range
WINDOW = (0.20, 0.60)                   # x/c of the sawtooth metrics
TREND_WINDOW = (0.12, 0.85)             # x/c of the smooth trend (least-squares polynomial)
TREND_DEGREE = 6
TABLE_STATIONS = tuple(round(0.05 * k, 2) for k in range(4, 20))    # NACA 65 table stations 0.20 .. 0.95

# ============================================================

import csv                                                                       # noqa: E402
import math                                                                      # noqa: E402
import sys                                                                       # noqa: E402
import warnings                                                                  # noqa: E402
from concurrent.futures import ProcessPoolExecutor                               # noqa: E402
from dataclasses import replace                                                  # noqa: E402
from pathlib import Path                                                         # noqa: E402

import numpy as np                                                               # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / OUTPUT_DIR
LINES: list = []
LEVEL_INK = ("#9ecae1", "#6baed6", "#3182bd", "#08519c", "#08306b")


def say(text: str = "") -> None:
    print(text)
    LINES.append(text)


# --- the cascade and the diagnostic geometry -------------------------------------------------------------------

def base_settings(mach: float, level: str):
    import compare_dunavant_aungier as cmp
    from turbolenza.b2b import controller as ct

    return ct.CaseSettings(**cmp.CASE, profile_variant=cmp.PROFILE_VARIANT, inlet_mach=mach, grid_level=level,
                           n_passage=None, n_eta=None)


def chord_axis(cascade):
    le, te = np.asarray(cascade.active.leading_edge_point), np.asarray(cascade.active.trailing_edge_point)
    return le, te - le


def x_over_c(points, cascade):
    le, axis = chord_axis(cascade)
    return (np.asarray(points) - le) @ axis / (axis @ axis)


def smoothed_cascade(cascade, span: tuple = SMOOTH_RANGE, ramp: float = SMOOTH_RAMP):
    """DIAGNOSTIC GEOMETRY, NOT PRODUCTION: the cascade with its suction curve's curvature
    ripple removed where the curve lies at x/c in span. The ripple r(s) = kappa(s) - trend(s)
    (trend: a degree-TREND_DEGREE polynomial in arc length s over span), cosine-tapered over
    ramp (x/c) at both ends and corrected by a linear function so that its first two moments
    vanish, is cancelled by a normal offset delta(s) with delta'' = -r, delta = delta' = 0 at
    both ends: X_new(u) = X(u) + delta N(u) (N the left normal, the fluid side). X_new is
    refitted on the curve's own knots (least squares on 40001 samples), its end control
    points kept exactly; LE, TE, the pressure curve and everything else untouched. Returns
    (cascade, new curve, max |delta|)."""
    from scipy.interpolate import make_lsq_spline

    from turbolenza.b2b.geometry import ParametricBladeCurve

    curves = cascade.active_curves
    c = curves.suction
    u = np.linspace(0.0, 1.0, 40001)
    X, d1, d2 = c.evaluate(u), c.derivative(u, 1), c.derivative(u, 2)
    speed = np.hypot(d1[:, 0], d1[:, 1])
    tangent = d1 / speed[:, None]
    normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    kappa = c.curvature(u)
    s = np.concatenate(([0.0], np.cumsum(0.5 * (speed[1:] + speed[:-1]) * np.diff(u))))
    xc = x_over_c(X, cascade)
    inside = (xc >= span[0]) & (xc <= span[1])
    k0, k1 = np.flatnonzero(inside)[[0, -1]]
    ss = s[k0:k1 + 1]
    trend = np.polyval(np.polyfit(ss, kappa[k0:k1 + 1], TREND_DEGREE), ss)
    r = kappa[k0:k1 + 1] - trend
    xw = xc[k0:k1 + 1]
    w = np.clip(np.minimum((xw - span[0]) / ramp, (span[1] - xw) / ramp), 0.0, 1.0)
    w = 0.5 - 0.5 * np.cos(math.pi * w)
    rt = w * r
    length = ss[-1] - ss[0]
    basis = (w, w * (ss - ss[0]) / length)

    def moments(f):
        return np.array([np.trapezoid(f, ss), np.trapezoid((ss[-1] - ss) * f, ss)])

    A = np.column_stack([moments(b) for b in basis])
    alpha = np.linalg.solve(A, moments(rt))
    rt = rt - alpha[0] * basis[0] - alpha[1] * basis[1]

    def integral(f):
        return np.concatenate(([0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(ss))))

    delta = np.zeros(len(u))
    delta[k0:k1 + 1] = integral(integral(-rt))
    points = X + delta[:, None] * normal
    fit = make_lsq_spline(u, points, c.knots, k=c.degree)
    control = np.array(fit.c)
    control[0], control[-1] = np.asarray(c.control_points)[0], np.asarray(c.control_points)[-1]
    new = ParametricBladeCurve(knots=c.knots, control_points=control, reference_length=c.reference_length,
                               degree=c.degree)
    smooth_curves = replace(curves, suction=new)
    return replace(cascade, active_curves=smooth_curves, normalized_curves=smooth_curves), new, float(np.max(np.abs(delta)))


def dense_curvature(curve, cascade, n: int = 40001):
    u = np.linspace(0.0, 1.0, n)
    xy = curve.evaluate(u)
    return x_over_c(xy, cascade), curve.curvature(u), xy


# --- one run (a worker process) -------------------------------------------------------------------------------

class PassZero:
    """Observer: keeps the fields of the first pass (index 0, the uniform start density)."""

    def __init__(self) -> None:
        self.data = None

    def start(self, info) -> None:
        pass

    def update(self, record) -> None:
        pass

    def finish(self, solution, error) -> None:
        pass

    def inspect(self, fields) -> None:
        if self.data is None and fields.record.index == 0:
            self.data = dict(mach=np.array(fields.mach), psi=np.array(fields.psi), psi_eta=np.array(fields.flow.psi_eta),
                             W_m=np.array(fields.flow.W_m), W_theta=np.array(fields.flow.W_theta))


def run_job(job: dict) -> dict:
    warnings.simplefilter("ignore")
    from turbolenza.b2b import controller as ct
    from turbolenza.b2b import postprocess as pp
    from turbolenza.b2b import solver as sv

    settings = base_settings(job["mach"], job["level"])
    case = ct.build_case(settings)
    cascade = case.cascade
    if job["smooth"]:
        cascade, _, _ = smoothed_cascade(cascade)
    observer = PassZero()
    sol = sv.solve(cascade, case.config, c_ref=settings.chord, monitor=observer)
    g = sol.grid
    f, l = g.first_blade_i, g.last_blade_i
    surfaces = pp.blade_surfaces(sol)
    d1, d2 = g.wall_curve_derivative(1)[0], g.wall_curve_derivative(2)[0]
    from turbolenza.b2b.geometry import signed_curvature

    rows = np.arange(4)
    nodes = np.asarray(g.nodes)
    out = dict(tag=job["tag"], level=job["level"], mach_in=job["mach"], smooth=job["smooth"], f=f, l=l,
               n_m=len(g.m), n_eta=len(g.eta), dm_c=g.dm / g.chord, deta=g.deta, beta_out=math.degrees(sol.exit_angle),
               passes=len(sol.passes), converged=sol.converged, m=np.asarray(g.m), eta=np.asarray(g.eta), nodes=nodes,
               pitch=float(g.pitch), chord=float(g.chord), mach=np.asarray(sol.mach), psi=np.asarray(sol.psi),
               psi_m=np.asarray(sol.flow.psi_m), psi_eta=np.asarray(sol.flow.psi_eta), W_m=np.asarray(sol.flow.W_m),
               W_theta=np.asarray(sol.flow.W_theta), W=np.asarray(sol.flow.W), rho=np.asarray(sol.density),
               a=np.asarray(sol.static_state.sound_speed), S=np.asarray(sol.metrics.S), tan_beta=np.asarray(sol.metrics.tan_beta),
               cos_beta=np.asarray(sol.metrics.cos_beta), x_c_ss=np.asarray(surfaces.x_c_ss), m_c=np.asarray(surfaces.m_c),
               kappa_ss=signed_curvature(d1, d2), wall_surface=np.asarray(g.wall_identity.lower.surface),
               x_c_rows=np.array([x_over_c(nodes[f:l + 1, j], cascade) for j in rows]),
               pass0={k: v[f:l + 1][:, rows] for k, v in observer.data.items()} if observer.data else None)
    return out


# --- metrics --------------------------------------------------------------------------------------------------

def alt(q):
    """alt_k = q[k+1] - 2 q[k] + q[k-1] (NaN at the ends)."""
    q = np.asarray(q, dtype=float)
    out = np.full(q.shape, np.nan)
    out[1:-1] = q[2:] - 2.0 * q[1:-1] + q[:-2]
    return out


def residual(x, q, window=TREND_WINDOW, degree=TREND_DEGREE):
    """q minus its least-squares polynomial trend over window (NaN outside)."""
    x, q = np.asarray(x, dtype=float), np.asarray(q, dtype=float)
    inside = (x >= window[0]) & (x <= window[1]) & np.isfinite(q)
    coef = np.polyfit(x[inside], q[inside], degree)
    out = np.full(q.shape, np.nan)
    out[inside] = q[inside] - np.polyval(coef, x[inside])
    return out


def metrics(x, q):
    a, r = alt(q), residual(x, q)
    w = (x >= WINDOW[0]) & (x <= WINDOW[1])
    aw, rw = a[w], r[w]
    signs = np.sign(aw[np.isfinite(aw)])
    flips = float(np.mean(signs[1:] != signs[:-1])) if signs.size > 2 else float("nan")
    k_alt = int(np.nanargmax(np.where(w, np.abs(a), np.nan)))
    k_res = int(np.nanargmax(np.where(w, np.abs(r), np.nan)))
    return dict(rms_alt=float(np.sqrt(np.nanmean(aw ** 2))), max_alt=float(np.abs(a[k_alt])), x_max_alt=float(x[k_alt]),
                alt_sign_flip_fraction=flips, rms_residual=float(np.sqrt(np.nanmean(rw ** 2))),
                max_residual=float(np.abs(r[k_res])), x_max_residual=float(x[k_res]), nodes_in_window=int(w.sum()))


def wavy_wall_prediction(x_dense, kappa_dense, xy_dense, mach_trend: float, x_nodes):
    """Linear subsonic wavy-wall theory, applied locally along the suction wall: a wall
    curvature perturbation kappa'(s) gives u'/U = -kappa'^(k) / (beta k) per Fourier
    component k (in arc length s), beta = sqrt(1 - M^2); dM/M = (1 + (gamma - 1) M^2 / 2) u'/U.
    kappa' is the curvature minus its trend over the window. An order-of-magnitude and
    phase check, not a solution."""
    inside = (x_dense >= TREND_WINDOW[0]) & (x_dense <= TREND_WINDOW[1])
    s = np.concatenate(([0.0], np.cumsum(np.hypot(*np.diff(xy_dense[inside], axis=0).T))))
    kp = residual(x_dense[inside], kappa_dense[inside])
    n = 4096
    su = np.linspace(s[0], s[-1], n)
    ku = np.interp(su, s, kp)
    ku = ku - ku.mean()
    taper = np.hanning(n) ** 0.25                     # mild end taper against wrap-around
    spectrum = np.fft.rfft(ku * taper)
    wavenumber = 2.0 * math.pi * np.fft.rfftfreq(n, d=su[1] - su[0])
    beta = math.sqrt(max(1.0 - mach_trend ** 2, 1e-6))
    gain = np.zeros_like(wavenumber)
    gain[1:] = -1.0 / (beta * wavenumber[1:])
    u = np.fft.irfft(spectrum * gain, n) / np.where(taper > 1e-3, taper, 1.0)
    dmach = mach_trend * (1.0 + 0.2 * mach_trend ** 2) * u
    xs = np.interp(su, s, x_dense[inside])
    return np.interp(x_nodes, xs, dmach, left=np.nan, right=np.nan)


# --- figures --------------------------------------------------------------------------------------------------

def outline(cascade):
    a = cascade.active
    return np.vstack((np.asarray(a.suction_surface)[::-1], np.asarray(a.pressure_surface)[1:]))


def raw_nodes_figure(run, cascade, edge: str, path: Path) -> None:
    """Raw node scatter of the five stations around the LE (f-2 .. f+2) or TE (l-2 .. l+2),
    coloured by node Mach number, each station joined by a thin line; no interpolation."""
    import matplotlib.pyplot as plt

    f, l, nodes, mach = run["f"], run["l"], run["nodes"], run["mach"]
    centre = f if edge == "LE" else l
    stations = list(range(centre - 2, centre + 3))
    names = [f"{'f' if edge == 'LE' else 'l'}{d:+d}".replace("+0", "") for d in range(-2, 3)]
    pitch = run["pitch"]
    loop = outline(cascade)
    low, high = float(np.min(mach[stations])), float(np.max(mach[stations]))
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 7.2), gridspec_kw=dict(width_ratios=(1.0, 1.25)))
    for panel, ax in enumerate(axes):
        for k in (0, 1):
            blade = loop + np.array([0.0, k * pitch])
            ax.fill(blade[:, 0], blade[:, 1], color="#e4e4e4", ec="#555555", lw=0.8, zorder=1)
        for i, name in zip(stations, names):
            ax.plot(nodes[i, :, 0], nodes[i, :, 1], "-", color="#888888", lw=0.6, zorder=2)
            sc = ax.scatter(nodes[i, :, 0], nodes[i, :, 1], c=mach[i], cmap="viridis", vmin=low, vmax=high, s=26,
                            edgecolors="black", linewidths=0.3, zorder=3)
            top = nodes[i, -1] if panel == 0 else nodes[i, min(6, len(nodes[i]) - 1)]
            ax.annotate(name, top, textcoords="offset points", xytext=(0, 6), ha="center", fontsize=9, weight="bold")
        if panel == 1:                                # close to the suction wall (eta = 0), equal aspect
            j_top = min(6, nodes.shape[1] - 1)
            xs, ys = nodes[stations, :j_top + 1, 0], nodes[stations, :j_top + 1, 1]
            pad = 0.01
            ax.set_xlim(xs.min() - pad, xs.max() + pad)
            ax.set_ylim(ys.min() - pad, ys.max() + 2 * pad)
            ax.set_aspect("equal")
            ax.set_title(f"suction-wall corner, eta rows 0 .. {j_top} (equal aspect)", fontsize=10)
        else:
            ax.set_xlim(nodes[stations, :, 0].min() - 0.02, nodes[stations, :, 0].max() + 0.02)
            ax.set_ylim(nodes[stations, :, 1].min() - 0.04, nodes[stations, :, 1].max() + 0.06)
            ax.set_title("whole station columns, wall to wall (axes not to scale)", fontsize=10)
        ax.set_xlabel("m / c")
        ax.set_ylabel("y / c")
        ax.grid(alpha=0.2)
    fig.colorbar(sc, ax=axes, label="node relative Mach number (raw, no interpolation)", shrink=0.85)
    fig.suptitle(f"{edge}: raw nodes of stations {names[0]} .. {names[-1]} ({run['level']} grid, M1 = {run['mach_in']})",
                 fontsize=11)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def quantities_figure(run, cascade, path: Path) -> None:
    """Raw nodes of f-2 .. f+2 near the suction wall coloured by psi, psi_m, W_m, W_theta, M."""
    import matplotlib.pyplot as plt

    f, nodes = run["f"], run["nodes"]
    stations = list(range(f - 2, f + 3))
    j_top = min(6, nodes.shape[1] - 1)
    loop = outline(cascade)
    fig, axes = plt.subplots(1, 5, figsize=(19, 5.0))
    for ax, name in zip(axes, ("psi", "psi_m", "W_m", "W_theta", "mach")):
        q = run[name][stations, :j_top + 1]
        ax.fill(loop[:, 0], loop[:, 1], color="#e4e4e4", ec="#555555", lw=0.8, zorder=1)
        for s_i, i in enumerate(stations):
            ax.plot(nodes[i, :j_top + 1, 0], nodes[i, :j_top + 1, 1], "-", color="#888888", lw=0.6, zorder=2)
        sc = ax.scatter(nodes[stations, :j_top + 1, 0].ravel(), nodes[stations, :j_top + 1, 1].ravel(), c=q.ravel(),
                        cmap="viridis", s=30, edgecolors="black", linewidths=0.3, zorder=3)
        for i, label in zip(stations, ("f-2", "f-1", "f", "f+1", "f+2")):
            ax.annotate(label, nodes[i, j_top], textcoords="offset points", xytext=(0, 5), ha="center", fontsize=8)
        xs, ys = nodes[stations, :j_top + 1, 0], nodes[stations, :j_top + 1, 1]
        ax.set_xlim(xs.min() - 0.01, xs.max() + 0.01)
        ax.set_ylim(ys.min() - 0.01, ys.max() + 0.02)
        ax.set_aspect("equal")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("m / c")
        fig.colorbar(sc, ax=ax, shrink=0.7)
    axes[0].set_ylabel("y / c")
    fig.suptitle(f"LE, suction-wall corner: raw nodes of f-2 .. f+2 coloured by each quantity ({run['level']})", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


QUANTITIES = ("psi", "psi_m", "W_m", "W_theta", "mach")


def station_difference_rows(run) -> list:
    """Delta_m q(i, j) = q(i+1, j) - q(i, j) for the pairs around the LE and the TE, every
    eta, and the pair's max over eta against the median of the blade-interior pairs; also
    the second difference q(i+1) - 2 q(i) + q(i-1)."""
    f, l, n_m = run["f"], run["l"], run["n_m"]
    interior = range(f + 5, l - 5)
    rows = []
    for name in QUANTITIES:
        q = run[name]
        d1 = np.abs(np.diff(q, axis=0))
        d2 = np.abs(q[2:] - 2.0 * q[1:-1] + q[:-2])
        ref1 = float(np.median([np.max(d1[i]) for i in interior]))
        ref2 = float(np.median([np.max(d2[i - 1]) for i in interior]))
        for edge, centre, label in (("LE", f, "f"), ("TE", l, "l")):
            for d in range(-3, 3):
                i = centre + d
                pair = f"{label}{d:+d}->{label}{d + 1:+d}".replace("+0", "").replace("-0", "")
                second = float(np.max(d2[i - 1])) if 1 <= i <= n_m - 2 else float("nan")
                rows.append(dict(level=run["level"], quantity=name, edge=edge, pair=pair, i=i,
                                 max_abs_delta=float(np.max(d1[i])), ratio_to_interior=float(np.max(d1[i])) / ref1,
                                 wall_delta=float(q[i + 1, 0] - q[i, 0]), row1_delta=float(q[i + 1, 1] - q[i, 1]),
                                 max_abs_second_diff_at_i=second, second_ratio_to_interior=second / ref2))
    return rows


def differences_figure(run, path: Path) -> None:
    import matplotlib.pyplot as plt

    f, l, eta = run["f"], run["l"], run["eta"]
    fig, axes = plt.subplots(2, 5, figsize=(20, 8.0))
    inks = ("#bdbdbd", "#969696", "#e6550d", "#3182bd", "#636363", "#252525")
    for row, (edge, centre, label) in enumerate((("LE", f, "f"), ("TE", l, "l"))):
        for col, name in enumerate(QUANTITIES):
            ax = axes[row, col]
            q = run[name]
            for k, d in enumerate(range(-3, 3)):
                i = centre + d
                pair = f"{label}{d:+d}->{label}{d + 1:+d}".replace("+0", "").replace("-0", "")
                lw = 2.2 if d in (-1, 0) else 1.0
                ax.plot(q[i + 1] - q[i], eta, "-", color=inks[k], lw=lw, label=pair)
            ax.axvline(0.0, color="black", lw=0.5)
            ax.set_title(f"{edge}: Delta_m {name}", fontsize=10)
            ax.set_xlabel(f"q(i+1, j) - q(i, j)")
            if col == 0:
                ax.set_ylabel("eta")
            ax.grid(alpha=0.25)
            if col == 4:
                ax.legend(fontsize=7)
    fig.suptitle(f"Station-to-station differences around the LE and TE ({run['level']}); bold: the transition pairs",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=125)
    plt.close(fig)


def wall_sequences_figure(run, path: Path) -> None:
    import matplotlib.pyplot as plt

    f, l = run["f"], run["l"]
    k = np.arange(l - f + 1)
    xc = run["x_c_ss"]
    seq = dict(mach=run["mach"][f:l + 1, 0], W_m=run["W_m"][f:l + 1, 0], W_theta=run["W_theta"][f:l + 1, 0],
               psi_eta=run["psi_eta"][f:l + 1, 0], tan_beta=run["tan_beta"][f:l + 1, 0], kappa=run["kappa_ss"],
               psi_row1=run["psi"][f:l + 1, 1])
    labels = dict(mach="wall Mach", W_m="W_m (wall)", W_theta="W_theta (wall)", psi_eta="psi_eta (wall, feeds W_m)",
                  tan_beta="tan beta (wall, geometry)", kappa="wall curvature (geometry) [1/c]",
                  psi_row1="psi at eta row 1")
    fig, axes = plt.subplots(4, 2, figsize=(15, 13))
    for ax, name in zip(axes.ravel(), list(seq) + ["alt"]):
        if name == "alt":
            for qn, ink in (("mach", "#3182bd"), ("kappa", "#e6550d")):
                a = alt(seq[qn])
                ax.plot(xc, a / np.nanmax(np.abs(a[(xc > 0.1) & (xc < 0.9)])), "o-", color=ink, ms=3, lw=0.8,
                        label=f"alt of {labels[qn]} (normalized)")
            ax.set_xlabel("x / c")
            ax.set_title("alternating metric alt_k = q[k+1] - 2 q[k] + q[k-1]", fontsize=10)
            ax.legend(fontsize=8)
        else:
            ax.plot(xc, seq[name], "o-", ms=3.5, lw=0.8, color="#08519c")
            ax.set_title(labels[name], fontsize=10)
            ax.set_xlabel("x / c")
            top = ax.secondary_xaxis("top", functions=(lambda x: np.interp(x, xc, k), lambda i: np.interp(i, k, xc)))
            top.set_xlabel("node index along the suction wall", fontsize=8)
        for xs in TABLE_STATIONS:
            ax.axvline(xs, color="#e6550d", lw=0.5, alpha=0.35)
        ax.grid(alpha=0.25)
    fig.suptitle(f"Raw suction-wall sequences ({run['level']}, M1 = {run['mach_in']}): orange lines = NACA 65 thickness "
                 f"table stations", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=125)
    plt.close(fig)


def five_grids_figure(runs, path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for k, level in enumerate(LEVELS):
        r = runs[level]
        f, l = r["f"], r["l"]
        m = r["mach"][f:l + 1, 0]
        xc = r["x_c_ss"]
        label = f"{level} (dm/c {r['dm_c']:.4f})"
        axes[0, 0].plot(xc, m, "o-", ms=2.8, lw=0.7, color=LEVEL_INK[k], label=label)
        axes[0, 1].plot(np.arange(len(m)), m, "o-", ms=2.8, lw=0.7, color=LEVEL_INK[k], label=label)
        axes[1, 0].plot(xc, residual(xc, m), "o-", ms=2.8, lw=0.9, color=LEVEL_INK[k], label=label)
        for j, ls in ((1, "--"), (2, ":")):
            if level in ("medium", "very_fine"):
                axes[1, 1].plot(r["x_c_rows"][j], residual(r["x_c_rows"][j], r["mach"][f:l + 1, j]), ls, lw=1.0,
                                color=LEVEL_INK[k], label=f"{level}, eta row {j}")
        if level in ("medium", "very_fine"):
            axes[1, 1].plot(xc, residual(xc, m), "-", lw=1.6, color=LEVEL_INK[k], label=f"{level}, wall")
    for ax in (axes[0, 0], axes[1, 0], axes[1, 1]):
        for xs in TABLE_STATIONS:
            ax.axvline(xs, color="#e6550d", lw=0.5, alpha=0.35)
        ax.set_xlabel("x / c")
    axes[0, 0].set_title("raw suction-wall Mach vs x/c", fontsize=10)
    axes[0, 1].set_title("raw suction-wall Mach vs node index", fontsize=10)
    axes[0, 1].set_xlabel("node index along the suction wall")
    axes[1, 0].set_title(f"wall Mach minus its smooth trend (degree-{TREND_DEGREE} fit, x/c {TREND_WINDOW[0]} .. "
                         f"{TREND_WINDOW[1]})", fontsize=10)
    axes[1, 0].set_xlim(*TREND_WINDOW)
    axes[1, 1].set_title("the same off the wall: eta rows 1 and 2 (medium, very fine)", fontsize=10)
    axes[1, 1].set_xlim(*TREND_WINDOW)
    for ax in axes.ravel():
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
    fig.suptitle("Suction-wall Mach on the five standard grids, raw nodes; orange lines = thickness table stations",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=125)
    plt.close(fig)


def causality_figure(runs, cascade, smooth_curve, path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    xd, kd, xyd = dense_curvature(cascade.active_curves.suction, cascade)
    xs_, ks_, _ = dense_curvature(smooth_curve, cascade)
    ax = axes[0, 0]
    ax.plot(xd, kd, "-", color="#e6550d", lw=1.0, label="production geometry (C2 curve)")
    ax.plot(xs_, ks_, "-", color="#252525", lw=1.4, label="diagnostic: curvature ripple removed (normal offset, refit)")
    ax.set_xlim(0.05, 0.95)
    inside = (xd > 0.1) & (xd < 0.9)
    ax.set_ylim(np.min(kd[inside]) - 0.2, np.max(kd[inside]) + 0.2)
    ax.set_title("suction-surface curvature [1/c]", fontsize=10)
    ax.legend(fontsize=8)
    for level, ax in (("medium", axes[0, 1]), ("very_fine", axes[1, 0])):
        base, smooth = runs[level], runs[f"SMOOTH_{level}"]
        for r, ink, label in ((base, "#e6550d", "production geometry"), (smooth, "#252525", "smoothed suction curve")):
            f, l = r["f"], r["l"]
            ax.plot(r["x_c_ss"], r["mach"][f:l + 1, 0], "o-", ms=3, lw=0.8, color=ink, label=label)
        ax.set_title(f"suction-wall Mach, {level}: same solver, only the geometry differs", fontsize=10)
        ax.set_xlim(0.0, 1.0)
        ax.legend(fontsize=8)
    ax = axes[1, 1]
    for tag, ink, ls, label in (("medium", "#3182bd", "-", "medium, converged M 0.65"),
                                ("very_fine", "#08306b", "-", "very fine, converged M 0.65"),
                                ("SMOOTH_very_fine", "#252525", "-", "very fine, smoothed geometry"),
                                ("M030_medium", "#31a354", "--", "medium, M1 0.30 (scaled to M 0.65's trend)")):
        r = runs[tag]
        f, l = r["f"], r["l"]
        m = r["mach"][f:l + 1, 0]
        res = residual(r["x_c_ss"], m)
        if tag == "M030_medium":
            res = res * np.nanmean(runs["medium"]["mach"][runs["medium"]["f"]:runs["medium"]["l"] + 1, 0]) / np.nanmean(m)
        ax.plot(r["x_c_ss"], res, ls, marker="o", ms=2.5, lw=1.0, color=ink, label=label)
    vf = runs["very_fine"]
    m_vf = vf["mach"][vf["f"]:vf["l"] + 1, 0]
    trend = float(np.nanmean(m_vf[(vf["x_c_ss"] > WINDOW[0]) & (vf["x_c_ss"] < WINDOW[1])]))
    pred = wavy_wall_prediction(xd, kd, xyd, trend, vf["x_c_ss"])
    ax.plot(vf["x_c_ss"], pred, "-", color="#e6550d", lw=1.8, alpha=0.8,
            label=f"linear wavy-wall theory from the curvature ripple (M {trend:.2f})")
    p0 = vf["pass0"]
    if p0 is not None:
        ax.plot(vf["x_c_ss"], residual(vf["x_c_ss"], p0["mach"][:, 0]) * trend / np.nanmean(p0["mach"][:, 0]), ":",
                color="#756bb1", lw=1.4, label="very fine, pass 0 (uniform start density, scaled)")
    ax.set_xlim(*TREND_WINDOW)
    ax.set_title("wall Mach minus its smooth trend: the ripple, its causes", fontsize=10)
    ax.legend(fontsize=7)
    for ax in axes.ravel():
        for xs in TABLE_STATIONS:
            ax.axvline(xs, color="#e6550d", lw=0.5, alpha=0.3)
        ax.grid(alpha=0.25)
        ax.set_xlabel("x / c")
    fig.suptitle("Causality: the suction-wall Mach ripple follows the surface-curvature ripple of the geometry", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=125)
    plt.close(fig)


def write_csv(name: str, rows: list) -> None:
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(OUT / name, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


# --- main -----------------------------------------------------------------------------------------------------

def main() -> int:
    import matplotlib

    matplotlib.use("Agg")
    warnings.simplefilter("ignore")
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [dict(tag=level, level=level, mach=0.65, smooth=False) for level in LEVELS]
    jobs += [dict(tag="M030_medium", level="medium", mach=0.30, smooth=False)]
    jobs += [dict(tag=f"SMOOTH_{level}", level=level, mach=0.65, smooth=True) for level in SMOOTH_LEVELS]
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        runs = {r["tag"]: r for r in pool.map(run_job, jobs)}

    from turbolenza.b2b import controller as ct

    cascade = ct.build_case(base_settings(0.65, DETAIL_LEVEL)).cascade
    smooth_cascade, smooth_curve, offset = smoothed_cascade(cascade)
    u = np.linspace(0.0, 1.0, 40001)
    shift = np.hypot(*(smooth_curve.evaluate(u) - cascade.active_curves.suction.evaluate(u)).T)
    say(f"diagnostic suction curve (curvature ripple removed over x/c {SMOOTH_RANGE[0]} .. {SMOOTH_RANGE[1]}): normal "
        f"offset at most {offset:.3e} c, refitted curve within {shift.max():.3e} c of the production curve")
    xd, kd, _ = dense_curvature(cascade.active_curves.suction, cascade)
    xs_, ks_, _ = dense_curvature(smooth_curve, cascade)
    w = (xd > WINDOW[0]) & (xd < WINDOW[1])
    say(f"suction curvature in x/c {WINDOW}: production {kd[w].min():+.3f} .. {kd[w].max():+.3f} /c (residual rms "
        f"{np.sqrt(np.nanmean(residual(xd, kd)[w] ** 2)):.3f}), smoothed {ks_[w].min():+.3f} .. {ks_[w].max():+.3f} /c "
        f"(residual rms {np.sqrt(np.nanmean(residual(xs_, ks_)[w] ** 2)):.3f})")
    say()

    # per-run metrics
    metric_rows, node_rows = [], []
    for tag, r in runs.items():
        f, l = r["f"], r["l"]
        xc = r["x_c_ss"]
        row = dict(run=tag, level=r["level"], mach_in=r["mach_in"], smooth=r["smooth"], n_m=r["n_m"], n_eta=r["n_eta"],
                   dm_c=r["dm_c"], converged=r["converged"], passes=r["passes"], beta_out_deg=r["beta_out"],
                   ss_peak=float(np.max(r["mach"][f:l + 1, 0])),
                   ss_peak_x_c=float(xc[int(np.argmax(r["mach"][f:l + 1, 0]))]))
        for name, q in (("mach", r["mach"][f:l + 1, 0]), ("mach_row1", r["mach"][f:l + 1, 1]),
                        ("mach_row2", r["mach"][f:l + 1, 2]), ("W_m", r["W_m"][f:l + 1, 0]),
                        ("tan_beta", r["tan_beta"][f:l + 1, 0]), ("psi_eta", r["psi_eta"][f:l + 1, 0]),
                        ("psi_row1", r["psi"][f:l + 1, 1]), ("kappa", r["kappa_ss"]), ("x_c_spacing", np.gradient(xc))):
            x_use = r["x_c_rows"][1] if name == "mach_row1" else r["x_c_rows"][2] if name == "mach_row2" else xc
            for key, value in metrics(x_use, q).items():
                row[f"{name}_{key}"] = value
        if r["pass0"] is not None:
            for key, value in metrics(xc, r["pass0"]["mach"][:, 0]).items():
                row[f"pass0_mach_{key}"] = value
        metric_rows.append(row)
        alt_m, res_m = alt(r["mach"][f:l + 1, 0]), residual(xc, r["mach"][f:l + 1, 0])
        for k in range(l - f + 1):
            i = f + k
            node_rows.append(dict(run=tag, k=k, i=i, x_c=xc[k], m_c=r["m_c"][k], mach=r["mach"][i, 0],
                                  mach_row1=r["mach"][i, 1], mach_row2=r["mach"][i, 2], W=r["W"][i, 0], W_m=r["W_m"][i, 0],
                                  W_theta=r["W_theta"][i, 0], psi_eta=r["psi_eta"][i, 0], psi_row1=r["psi"][i, 1],
                                  rho=r["rho"][i, 0], a=r["a"][i, 0], S=r["S"][i], tan_beta=r["tan_beta"][i, 0],
                                  cos_beta=r["cos_beta"][i, 0], kappa=r["kappa_ss"][k], alt_mach=alt_m[k],
                                  residual_mach=res_m[k],
                                  pass0_mach=r["pass0"]["mach"][k, 0] if r["pass0"] is not None else float("nan")))
    write_csv("sawtooth_metrics.csv", metric_rows)
    write_csv("ss_wall_nodes.csv", node_rows)

    say("SAWTOOTH METRICS on the suction wall, x/c window " + f"{WINDOW}")
    say(f"  {'run':18s} {'grid':>9s} {'dm/c':>7s} {'beta_out':>9s} {'SS peak':>8s} {'rms alt M':>10s} {'max|alt M|':>11s} "
        f"{'at x/c':>7s} {'flip frac':>9s} {'rms res M':>10s} {'max|res|':>9s} {'at x/c':>7s} {'res row1':>9s} "
        f"{'res row2':>9s} {'pass0 res':>9s}")
    for row in metric_rows:
        say(f"  {row['run']:18s} {row['n_m']:4d}x{row['n_eta']:<4d} {row['dm_c']:7.4f} {row['beta_out_deg']:9.3f} "
            f"{row['ss_peak']:8.4f} {row['mach_rms_alt']:10.5f} {row['mach_max_alt']:11.5f} {row['mach_x_max_alt']:7.3f} "
            f"{row['mach_alt_sign_flip_fraction']:9.2f} {row['mach_rms_residual']:10.5f} {row['mach_max_residual']:9.5f} "
            f"{row['mach_x_max_residual']:7.3f} {row['mach_row1_rms_residual']:9.5f} {row['mach_row2_rms_residual']:9.5f} "
            f"{row.get('pass0_mach_rms_residual', float('nan')):9.5f}")
    say()
    say("  the same ripple measure (rms of q minus its smooth trend, x/c window) for the inputs of the wall Mach:")
    say(f"  {'run':18s} {'kappa':>9s} {'tan_beta':>10s} {'psi_eta':>10s} {'W_m':>9s} {'psi row1':>10s} {'x/c step':>10s}")
    for row in metric_rows:
        say(f"  {row['run']:18s} {row['kappa_rms_residual']:9.4f} {row['tan_beta_rms_residual']:10.6f} "
            f"{row['psi_eta_rms_residual']:10.6f} {row['W_m_rms_residual']:9.4f} {row['psi_row1_rms_residual']:10.3e} "
            f"{row['x_c_spacing_rms_residual']:10.3e}")
    say()

    # the recovery chain on the wall of the detail level: which factor carries the ripple?
    r = runs[DETAIL_LEVEL]
    f, l = r["f"], r["l"]
    xc = r["x_c_ss"]
    W_m, tb, mach = r["W_m"][f:l + 1, 0], r["tan_beta"][f:l + 1, 0], r["mach"][f:l + 1, 0]
    W = r["W"][f:l + 1, 0]
    a = r["a"][f:l + 1, 0]
    w = (xc >= WINDOW[0]) & (xc <= WINDOW[1])
    parts = dict(ln_W_m=np.log(W_m), ln_secant=np.log(np.sqrt(1.0 + tb ** 2)), ln_W=np.log(W), ln_a=np.log(a),
                 ln_M=np.log(mach), ln_psi_eta=np.log(r["psi_eta"][f:l + 1, 0]), ln_rho=np.log(r["rho"][f:l + 1, 0]),
                 ln_S=np.log(r["S"][f:l + 1]))
    say(f"WALL RECOVERY CHAIN ({DETAIL_LEVEL}): ln M = ln W - ln a, ln W = ln W_m + ln sqrt(1 + tan^2 beta), "
        f"ln W_m = ln psi_eta - ln rho - ln S + const; rms of each term's ripple (minus trend), x/c {WINDOW}:")
    for name, value in parts.items():
        say(f"  {name:12s} {np.sqrt(np.nanmean(residual(xc, value)[w] ** 2)):.3e}")
    say()

    # station differences around the LE / TE (detail level)
    diff_rows = station_difference_rows(r)
    write_csv("station_differences.csv", diff_rows)
    say(f"STATION DIFFERENCES around the LE and TE ({DETAIL_LEVEL}): max over eta of |Delta_m q| per pair, as a "
        f"multiple of the median blade-interior pair (1 = ordinary); and the second difference at i")
    for edge in ("LE", "TE"):
        say(f"  {edge}:")
        pairs = [d for d in diff_rows if d["edge"] == edge and d["quantity"] == "psi"]
        say("    pair          " + "  ".join(f"{q:>14s}" for q in QUANTITIES))
        for p in pairs:
            cells = []
            for q in QUANTITIES:
                d = next(x for x in diff_rows if x["edge"] == edge and x["quantity"] == q and x["pair"] == p["pair"])
                cells.append(f"{d['ratio_to_interior']:6.2f} / {d['second_ratio_to_interior']:5.2f}")
            say(f"    {p['pair']:12s}  " + "  ".join(f"{c:>14s}" for c in cells))
    say("    (each cell: first-difference ratio / second-difference ratio)")
    say()

    # figures (A .. F)
    raw_nodes_figure(r, cascade, "LE", OUT / "le_raw_nodes.png")
    raw_nodes_figure(r, cascade, "TE", OUT / "te_raw_nodes.png")
    quantities_figure(r, cascade, OUT / "le_raw_quantities.png")
    differences_figure(r, OUT / "station_differences.png")
    wall_sequences_figure(r, OUT / "ss_wall_sequences.png")
    five_grids_figure(runs, OUT / "ss_five_grids.png")
    causality_figure(runs, cascade, smooth_curve, OUT / "ss_causality.png")

    # the location of the strongest ripple on each level against the table stations
    say("STRONGEST RIPPLE per level (wall Mach minus trend): x/c of the largest |residual| and of the residual's local "
        "extrema in the window")
    for level in LEVELS:
        rr = runs[level]
        x = rr["x_c_ss"]
        res = residual(x, rr["mach"][rr["f"]:rr["l"] + 1, 0])
        inside = np.flatnonzero((x >= WINDOW[0]) & (x <= WINDOW[1]))
        ext = [k for k in inside[1:-1] if np.isfinite(res[k - 1:k + 2]).all() and
               (res[k] - res[k - 1]) * (res[k + 1] - res[k]) < 0]
        say(f"  {level:12s} extrema at x/c " + ", ".join(f"{x[k]:.3f}({res[k]:+.4f})" for k in ext))
    say()
    (OUT / "report.txt").write_text("\n".join(LINES) + "\n", encoding="utf-8")
    import pickle

    with open(OUT / "runs.pkl", "wb") as handle:
        pickle.dump({k: {kk: vv for kk, vv in v.items()} for k, v in runs.items()}, handle)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
