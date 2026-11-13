"""
plot_geometry_continuity.py
===========================
The same CascadeSpec built twice, with surface_continuity "C1" (the validated construction)
and "C2" (its quintic representation), compared surface by surface.

C1 is drawn from its analytic construction (geometry.naca65_reference_surfaces): every
piece with its exact derivatives, both one-sided values at each junction, so its curvature
jumps show as they are; nothing is smoothed. C2 is drawn from the authoritative curves of
the C2 cascade (cascade.active_curves). Everything is in the cascade frame (m axial, y
tangential) except the blade-frame close-ups of figure 1. The horizontal coordinate of the
surface plots is s/L, arc length along the C1 surface over its length; the C2 curve
parameter u is that same arc-length fraction, and every C1 sample is compared with the C2
point nearest to it.

Figures
    1  geometry: C1 and C2 suction and pressure surfaces overlaid, with LE and TE close-ups
    2  deviation: signed normal distance of C1 from C2 along each surface, with max and RMS
    3  curvature * c, C1 against C2, both surfaces, with the construction junctions marked
    4  d2y/dm2 = d tan(beta) / dm, C1 against C2 (symlog), and an interior close-up
    5  beta = atan(dy/dm) and tan(beta), C1 against C2, and beta(C2) - beta(C1)
The preservation table (section 23 of the C2 task) is printed.

Dev-only: the package never imports matplotlib, and nothing here feeds back into it.

Usage
    python scripts/plot_geometry_continuity.py                 # the inputs below
    python scripts/plot_geometry_continuity.py --variant emery_base --physical-chord 0.05
    python scripts/plot_geometry_continuity.py --save outputs/geometry_continuity --no-show
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from turbolenza.b2b import geometry  # noqa: E402
from turbolenza.b2b.geometry import Cascade, CascadeSpec  # noqa: E402

# --- Development inputs: edit these ---------------------------------------------------
EXAMPLE = CascadeSpec(             # the Dunavant-like NACA 65-(12A10)10 of plot_cascade.py
    blade_type="NACA65",
    solidity=1.0,
    stagger_deg=33.0,
    cl0=1.2,
    t_max_over_c=0.10,
    meanline_type="A10",
    profile_variant="kovach_sandercock",
    physical_chord=None,
    n_camber_stations=101,
)
SETTINGS = {
    "c1_points_per_piece": 401,     # samples of every analytic C1 piece, both ends included
    "c2_points": 60001,             # uniform u samples of each C2 curve (its knots are added)
    "mask_cos": geometry.SLOPE_MIN_COS,   # dy/dm, d2y/dm2 masked where |cos beta| is below this
    "steep_cos": 0.05,              # figure 4 shades where |cos beta| < this: y(m) near-singular
    "le_window": (0.0, 0.03),       # s/L ranges of the close-ups
    "te_window": (0.975, 1.0),
    "interior_window": (0.03, 0.97),
    "save_dir": None,               # a folder to write the PNGs to, or None
    "show": True,
}
# ---------------------------------------------------------------------------------------

INK, MUTED, GRID = "#0b0b0b", "#6b6a66", "#e4e3df"
C1_COLOR, C2_COLOR = "#0b0b0b", "#d6402a"
SURFACES = ("suction", "pressure")
SURFACE_COLORS = {"suction": "#eb6834", "pressure": "#2a78d6"}
JUNCTION_COLORS = {"LE": "#8a5cd6", "table": "#b9b6ae", "Kovach start": "#1baf7a", "TE": "#b8327a"}

plt.rcParams.update({"figure.dpi": 100, "font.size": 8.5, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.spines.top": False,
                     "axes.spines.right": False, "legend.frameon": False, "savefig.facecolor": "white"})


@dataclass
class Trace:
    """One surface sampled with its derivatives in the cascade frame, active length unit."""
    s_over_l: np.ndarray
    points: np.ndarray
    d1: np.ndarray
    d2: np.ndarray
    piece: np.ndarray | None = None     # C1 only: the analytic piece of each sample

    @property
    def kappa(self) -> np.ndarray:
        return geometry.signed_curvature(self.d1, self.d2)

    def slopes(self, mask_cos: float) -> tuple[np.ndarray, np.ndarray]:
        return geometry.wall_slopes(self.d1, self.d2, mask_cos)

    @property
    def cos_beta(self) -> np.ndarray:
        return np.abs(self.d1[:, 0]) / np.hypot(self.d1[:, 0], self.d1[:, 1])


@dataclass
class Comparison:
    spec: CascadeSpec
    c1: Cascade
    c2: Cascade
    references: dict            # surface -> geometry.ReferenceSurface (blade frame, chord 1)
    c1_trace: dict              # surface -> Trace
    c2_trace: dict
    c2_at_c1: dict              # surface -> Trace of C2 at the projections of the C1 samples
    normal: dict                # surface -> signed normal distance C1 - C2 at the C1 samples
    junctions: dict             # surface -> list of (s/L, label)
    chord: float


def build(spec: CascadeSpec, settings: dict) -> Comparison:
    # C1 and C2 of the same thickness reconstruction (the C2 one): the plot shows the representation only
    reconstruction = spec.thickness_reconstruction or geometry.DEFAULT_THICKNESS_RECONSTRUCTION["C2"]
    c1 = geometry.make_cascade(replace(spec, surface_continuity="C1", thickness_reconstruction=reconstruction))
    c2 = geometry.make_cascade(replace(spec, surface_continuity="C2", thickness_reconstruction=reconstruction))
    chord = c1.active.chord
    references = dict(zip(SURFACES, geometry.naca65_reference_surfaces(spec.cl0, spec.t_max_over_c,
                                                                         profile_variant=spec.profile_variant,
                                                                         thickness_reconstruction=reconstruction)))
    c1_trace, c2_trace, c2_at_c1, normal, junctions = {}, {}, {}, {}, {}
    for name in SURFACES:
        reference = references[name]
        sample = reference.sample(settings["c1_points_per_piece"])
        turn = lambda v: geometry.rotate_by_stagger(v, c1.stagger)  # noqa: E731
        c1_trace[name] = Trace(sample.s / reference.length, turn(sample.points) * chord, turn(sample.tangent),
                               turn(sample.second) / chord, sample.piece)
        curve = getattr(c2.active_curves, name)
        u = np.union1d(np.linspace(0.0, 1.0, settings["c2_points"]), curve.interior_knots)
        c2_trace[name] = Trace(u, curve.evaluate(u), curve.derivative(u, 1), curve.derivative(u, 2))
        foot, distance = curve.project(c1_trace[name].points, c1_trace[name].s_over_l)
        c2_at_c1[name] = Trace(foot, curve.evaluate(foot), curve.derivative(foot, 1), curve.derivative(foot, 2))
        normal[name] = distance
        junctions[name] = [(j.s / reference.length, j.label) for j in reference.junctions]
    return Comparison(spec, c1, c2, references, c1_trace, c2_trace, c2_at_c1, normal, junctions, chord)


def junction_kind(label: str) -> str:
    if label.startswith("LE"):
        return "LE"
    if label.startswith("Kovach start"):
        return "Kovach start"
    if label.startswith(("Kovach TE", "closing")):
        return "TE"
    return "table"


def mark_junctions(ax, comparison: Comparison, name: str, window: tuple[float, float]) -> None:
    for position, label in comparison.junctions[name]:
        if window[0] <= position <= window[1]:
            kind = junction_kind(label)
            ax.axvline(position, color=JUNCTION_COLORS[kind], lw=0.7 if kind == "table" else 1.0,
                       ls=":" if kind == "table" else "--", zorder=0)


def piecewise(ax, trace: Trace, values: np.ndarray, window, **style) -> None:
    """C1 values piece by piece, so a jump at a junction is drawn as a jump."""
    label = style.pop("label", None)
    for k in np.unique(trace.piece):
        rows = (trace.piece == k) & (trace.s_over_l >= window[0]) & (trace.s_over_l <= window[1])
        if rows.any():
            ax.plot(trace.s_over_l[rows], values[rows], **style, label=label)
            label = None


def in_window(trace: Trace, window) -> np.ndarray:
    return (trace.s_over_l >= window[0]) & (trace.s_over_l <= window[1])


# --- Figure 1: geometry -----------------------------------------------------------------

def figure_geometry(comparison: Comparison) -> plt.Figure:
    fig = plt.figure(figsize=(13.5, 6.2), layout="constrained")
    grid = fig.add_gridspec(2, 3, width_ratios=(2.2, 1.0, 1.0))
    whole = fig.add_subplot(grid[:, 0])
    chord = comparison.chord
    for name in SURFACES:
        c1 = comparison.c1_trace[name]
        whole.plot(c1.points[:, 0] / chord, c1.points[:, 1] / chord, color=C1_COLOR, lw=1.6,
                   label="C1 (validated)" if name == "suction" else None)
        c2 = comparison.c2_trace[name]
        whole.plot(c2.points[:, 0] / chord, c2.points[:, 1] / chord, color=SURFACE_COLORS[name], lw=0.9, ls="--",
                   label=f"C2 {name} surface")
    worst = max(np.max(np.abs(comparison.normal[name])) for name in SURFACES) / chord
    whole.set(aspect="equal", xlabel="m / c", ylabel="y / c",
              title=f"{comparison.c1.metadata['designation']}, stagger {comparison.spec.stagger_deg:g} deg, "
                    f"{comparison.spec.profile_variant}")
    whole.legend(loc="upper left")
    whole.text(0.98, 0.03, f"max |C2 - C1| normal deviation = {worst:.2e} c\n(not visible at this scale; "
               "see figure 2)", transform=whole.transAxes, ha="right", va="bottom", color=MUTED)

    # Close-ups in the blade frame, chord 1, from the unstaggered arrays and curves:
    # (x range, y range) of each edge, wide and narrow.
    windows = {"LE": (((-0.008, 0.03), (-0.02, 0.035)), ((-0.004, 0.014), (-0.007, 0.011))),
               "TE": (((0.965, 1.004), (-0.03, 0.01)), ((0.986, 1.002), (-0.011, 0.005)))}
    for column, (edge, views) in enumerate(windows.items(), start=1):
        for row, (x_window, y_window) in enumerate(views):
            ax = fig.add_subplot(grid[row, column])
            for name in SURFACES:
                reference = comparison.references[name]
                sample = reference.sample(401)
                ax.plot(sample.points[:, 0], sample.points[:, 1], color=C1_COLOR, lw=1.8)
                points = getattr(comparison.c1.normalized, f"{name}_surface_unstaggered")
                ax.plot(points[:, 0], points[:, 1], "o", ms=2.2, color=C1_COLOR, mfc="white", mew=0.6)
                curve = getattr(comparison.c2.normalized_curves, f"{name}_unstaggered")
                xy = curve.evaluate(np.linspace(0.0, 1.0, 40001))
                ax.plot(xy[:, 0], xy[:, 1], color=SURFACE_COLORS[name], lw=0.9, ls="--")
            ax.set(xlim=x_window, ylim=y_window, aspect="equal", xlabel="x / c", ylabel="y / c",
                   title=f"{edge}, blade frame{', closer' if row else ''}")
            if row == 0:
                ax.text(0.02, 0.02, "solid: C1, dots: its vertices\ndashed: C2", transform=ax.transAxes,
                        fontsize=7.5, color=MUTED)
    return fig


# --- Figure 2: deviation ----------------------------------------------------------------

def figure_deviation(comparison: Comparison) -> plt.Figure:
    fig, axes = plt.subplots(2, 1, figsize=(12, 6.2), layout="constrained", sharex=True)
    for ax, name in zip(axes, SURFACES):
        trace, normal = comparison.c1_trace[name], comparison.normal[name] / comparison.chord
        weights = np.gradient(trace.s_over_l)
        rms = math.sqrt(float(np.sum(weights * normal**2) / np.sum(weights)))
        ax.plot(trace.s_over_l, normal, color=SURFACE_COLORS[name], lw=0.9)
        mark_junctions(ax, comparison, name, (0.0, 1.0))
        ax.axhline(0.0, color=MUTED, lw=0.6)
        ax.set(ylabel="normal deviation / c", title=f"{name} surface: C1 point minus nearest C2 point, along the "
                                                   f"C2 left normal; max |.| {np.max(np.abs(normal)):.2e} c, "
                                                   f"RMS {rms:.2e} c")
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    axes[-1].set_xlabel("s / L, arc length along the C1 surface")
    legend_for_junctions(axes[0])
    return fig


def legend_for_junctions(ax) -> None:
    for kind, color in JUNCTION_COLORS.items():
        ax.plot([], [], color=color, ls=":" if kind == "table" else "--", label={"LE": "LE circle / fair / surface",
                "table": "thickness-table station", "Kovach start": "Kovach start, station 0.60",
                "TE": "TE fair / circle / closing segment"}[kind])
    ax.legend(loc="upper right", ncols=4, fontsize=7.5)


# --- Figure 3: curvature ----------------------------------------------------------------

def figure_curvature(comparison: Comparison, settings: dict) -> plt.Figure:
    windows = [(0.0, 1.0), settings["le_window"], settings["interior_window"], settings["te_window"]]
    fig, axes = plt.subplots(2, 4, figsize=(17, 7.2), layout="constrained")
    chord = comparison.chord
    for row, name in enumerate(SURFACES):
        c1, c2 = comparison.c1_trace[name], comparison.c2_trace[name]
        for column, window in enumerate(windows):
            ax = axes[row, column]
            piecewise(ax, c1, c1.kappa * chord, window, color=C1_COLOR, lw=1.6, label="C1 (analytic pieces)")
            rows = in_window(c2, window)
            ax.plot(c2.s_over_l[rows], c2.kappa[rows] * chord, color=C2_COLOR, lw=0.8, label="C2")
            mark_junctions(ax, comparison, name, window)
            if column == 0:
                ax.set_yscale("symlog", linthresh=1.0)
            if column == 2:
                lo, hi = np.percentile(c1.kappa[in_window(c1, window)] * chord, [1, 99])
                ax.set_ylim(lo - 0.3 * (hi - lo), hi + 0.3 * (hi - lo))
            ax.set(xlim=window, title=f"{name}: s/L {window[0]:g} to {window[1]:g}",
                   xlabel="s / L", ylabel="curvature * c" if column == 0 else None)
    axes[0, 0].legend(loc="lower center")
    fig.suptitle("Signed curvature (counterclockwise positive, LE -> TE): the convex suction side is negative. "
                 "C1 is drawn piece by piece, so its jumps stay jumps.")
    return fig


# --- Figure 4: d2y/dm2 ------------------------------------------------------------------

def figure_second_derivative(comparison: Comparison, settings: dict) -> plt.Figure:
    fig, axes = plt.subplots(2, 3, figsize=(17, 7.2), layout="constrained")
    mask_cos, steep = settings["mask_cos"], settings["steep_cos"]
    windows = [(0.0, 1.0), settings["le_window"], settings["interior_window"]]
    chord = comparison.chord
    for row, name in enumerate(SURFACES):
        c1, c2 = comparison.c1_trace[name], comparison.c2_trace[name]
        d2_c1, d2_c2 = c1.slopes(mask_cos)[1] * chord, c2.slopes(mask_cos)[1] * chord
        for column, window in enumerate(windows):
            ax = axes[row, column]
            piecewise(ax, c1, d2_c1, window, color=C1_COLOR, lw=1.6, label="C1")
            rows = in_window(c2, window)
            ax.plot(c2.s_over_l[rows], d2_c2[rows], color=C2_COLOR, lw=0.8, label="C2")
            shade_steep(ax, c2, steep, window)
            mark_junctions(ax, comparison, name, window)
            if column < 2:
                ax.set_yscale("symlog", linthresh=1.0)
            else:
                values = d2_c1[in_window(c1, window)]
                lo, hi = np.nanpercentile(values, [1, 99])
                ax.set_ylim(lo - 0.3 * (hi - lo), hi + 0.3 * (hi - lo))
            ax.set(xlim=window, xlabel="s / L", title=f"{name}: s/L {window[0]:g} to {window[1]:g}",
                   ylabel="c * d2y/dm2" if column == 0 else None)
        masked = int(np.sum(np.isnan(d2_c2))), int(np.sum(np.isnan(d2_c1)))
        axes[row, 0].text(0.01, 0.02, f"masked (|cos beta| < {mask_cos:.1e}): C2 {masked[0]}, C1 {masked[1]} samples\n"
                          f"max |d2y/dm2| * c: C1 {np.nanmax(np.abs(d2_c1)):.3g}, C2 {np.nanmax(np.abs(d2_c2)):.3g}",
                          transform=axes[row, 0].transAxes, fontsize=7.5, color=MUTED)
    axes[0, 0].legend(loc="upper right")
    fig.suptitle("d2y/dm2 = d tan(beta)/dm = kappa / cos(beta)^3 in the cascade frame. Shaded: |cos beta| < "
                 f"{steep:g}, where the surface runs nearly across m (nose tip, tail). The growth to huge values "
                 "there comes from using m as the independent coordinate, not from a lack of C2 continuity; "
                 "nothing is clipped (symlog axes).")
    return fig


def shade_steep(ax, trace: Trace, steep: float, window) -> None:
    rows = in_window(trace, window)
    flag = trace.cos_beta[rows] < steep
    if flag.any():
        s = trace.s_over_l[rows]
        edges = np.flatnonzero(np.diff(np.concatenate(([0], flag.astype(int), [0]))))
        for a, b in zip(edges[::2], edges[1::2]):
            ax.axvspan(s[a], s[min(b, len(s) - 1)], color="#f8eec4", zorder=0)


# --- Figure 5: beta and tan(beta) -------------------------------------------------------

def figure_slope(comparison: Comparison, settings: dict) -> plt.Figure:
    fig, axes = plt.subplots(3, 2, figsize=(15, 8.6), layout="constrained", sharex="col")
    mask_cos = settings["mask_cos"]
    for column, name in enumerate(SURFACES):
        c1, c2, foot = comparison.c1_trace[name], comparison.c2_trace[name], comparison.c2_at_c1[name]
        tan_c1, tan_c2 = c1.slopes(mask_cos)[0], c2.slopes(mask_cos)[0]
        beta_c1, beta_c2 = np.degrees(np.arctan(tan_c1)), np.degrees(np.arctan(tan_c2))
        piecewise(axes[0, column], c1, beta_c1, (0, 1), color=C1_COLOR, lw=1.6, label="C1")
        axes[0, column].plot(c2.s_over_l, beta_c2, color=C2_COLOR, lw=0.8, label="C2")
        axes[0, column].set(ylabel="beta = atan(dy/dm), deg", title=f"{name} surface, cascade frame")
        piecewise(axes[1, column], c1, tan_c1, (0, 1), color=C1_COLOR, lw=1.6)
        axes[1, column].plot(c2.s_over_l, tan_c2, color=C2_COLOR, lw=0.8)
        axes[1, column].set(ylabel="tan(beta) = dy/dm", yscale="symlog")
        # beta of C2 at the point nearest each C1 sample, minus beta of C1, as tangent angles
        # so that nothing wraps at +-90 degrees.
        angle = lambda d: np.arctan2(d[:, 1], d[:, 0])  # noqa: E731
        difference = np.degrees(np.angle(np.exp(1j * (angle(foot.d1) - angle(c1.d1)))))
        piecewise(axes[2, column], c1, difference, (0, 1), color=SURFACE_COLORS[name], lw=0.9)
        axes[2, column].set(ylabel="beta(C2) - beta(C1), deg", xlabel="s / L")
        axes[2, column].set_yscale("symlog", linthresh=1e-3)
        for row in range(3):
            mark_junctions(axes[row, column], comparison, name, (0, 1))
        axes[2, column].set_title(f"largest |difference| {np.max(np.abs(difference)):.3g} deg "
                                  "(at the rounded junctions; C1 is one-sided there)", fontsize=8)
    axes[0, 0].legend()
    return fig


# --- Table ------------------------------------------------------------------------------

def preservation_table(comparison: Comparison) -> list[tuple[str, str, str]]:
    c1, c2, chord = comparison.c1, comparison.c2, comparison.chord
    rows = []
    for name in SURFACES:
        trace, normal = comparison.c1_trace[name], comparison.normal[name] / chord
        weights = np.gradient(trace.s_over_l)
        rms = math.sqrt(float(np.sum(weights * normal**2) / np.sum(weights)))
        rows.append((f"max |deviation| / c, {name}", "", f"{np.max(np.abs(normal)):.3e}"))
        rows.append((f"RMS deviation / c, {name}", "", f"{rms:.3e}"))
        length_c1 = comparison.references[name].length
        curve = getattr(c2.normalized_curves, f"{name}_unstaggered")
        rows.append((f"surface length / c, {name}", f"{length_c1:.10f}", f"{curve_length(curve):.10f}"))
    area_c1 = contour_area_reference(comparison.references)
    area_c2 = contour_area_curves(c2.normalized_curves)
    rows.append(("blade area / c^2", f"{area_c1:.10f}", f"{area_c2:.10f} ({100 * (area_c2 / area_c1 - 1):+.2e} %)"))
    t1, x1 = thickness(lambda name: comparison.references[name].sample(161).points)
    t2, x2 = thickness(lambda name: getattr(c2.normalized_curves, f"{name}_unstaggered").evaluate(
        np.linspace(0, 1, 4501)))
    rows.append(("max thickness / c", f"{t1:.8f}", f"{t2:.8f}"))
    rows.append(("x/c of max thickness", f"{x1:.5f}", f"{x2:.5f}"))
    x, local_c1 = local_thickness(comparison.references["suction"].sample(161).points,
                                  comparison.references["pressure"].sample(161).points)
    u = np.linspace(0.0, 1.0, 4501)
    x2_, local_c2 = local_thickness(c2.normalized_curves.suction_unstaggered.evaluate(u),
                                    c2.normalized_curves.pressure_unstaggered.evaluate(u))
    inside = (x > 0.02) & (x < 0.98)
    difference = np.interp(x[inside], x2_, local_c2) - local_c1[inside]
    k = int(np.argmax(np.abs(difference)))
    rows.append(("max |local thickness difference| / c, x/c 0.02-0.98", "",
                 f"{abs(difference[k]):.3e} at x/c {x[inside][k]:.4f}"))
    for label, attr in (("throat / c", "throat_over_chord"), ("o / s", "throat_over_pitch")):
        a, b = getattr(c1, attr), getattr(c2, attr)
        rows.append((label, f"{a:.10f}", f"{b:.10f} ({b - a:+.2e})"))
    for label, attr in (("LE point", "leading_edge_point"), ("TE point", "trailing_edge_point")):
        a, b = getattr(c1.normalized, attr), getattr(c2.normalized, attr)
        rows.append((label, f"({a[0]:.3e}, {a[1]:.3e})", f"identical: {np.array_equal(a, b)}"))
    rows.append(("kappa_le, kappa_te (equivalent arc), deg",
                 f"{math.degrees(c1.kappa_le):.6f}, {math.degrees(c1.kappa_te):.6f}",
                 f"{math.degrees(c2.kappa_le):.6f}, {math.degrees(c2.kappa_te):.6f}"))
    return rows


def curve_length(curve: geometry.ParametricBladeCurve) -> float:
    nodes, weights = np.polynomial.legendre.leggauss(8)
    breaks = np.unique(curve.knots)
    a, b = breaks[:-1], breaks[1:]
    u = (0.5 * (a + b)[:, None] + 0.5 * (b - a)[:, None] * nodes[None]).ravel()
    w = (0.5 * (b - a)[:, None] * weights[None]).ravel()
    d1 = curve.derivative(u, 1)
    return float(np.sum(w * np.hypot(d1[:, 0], d1[:, 1])))


def contour_area_curves(curves: geometry.SurfaceCurves) -> float:
    """Area inside suction (LE -> TE) and pressure (TE -> LE): 1/2 integral (x y' - y x'), Gauss."""
    nodes, weights = np.polynomial.legendre.leggauss(8)
    total = 0.0
    for name, sign in (("suction", -1.0), ("pressure", 1.0)):
        curve = getattr(curves, f"{name}_unstaggered")
        breaks = np.unique(curve.knots)
        a, b = breaks[:-1], breaks[1:]
        u = (0.5 * (a + b)[:, None] + 0.5 * (b - a)[:, None] * nodes[None]).ravel()
        w = (0.5 * (b - a)[:, None] * weights[None]).ravel()
        p, d1 = curve.evaluate(u), curve.derivative(u, 1)
        total += sign * 0.5 * float(np.sum(w * (p[:, 0] * d1[:, 1] - p[:, 1] * d1[:, 0])))
    return total


def contour_area_reference(references: dict) -> float:
    """The same integral on the analytic C1 pieces (arc-length samples, trapezoid on a fine grid)."""
    total = 0.0
    for name, sign in (("suction", -1.0), ("pressure", 1.0)):
        sample = references[name].sample(4001)
        integrand = sample.points[:, 0] * sample.tangent[:, 1] - sample.points[:, 1] * sample.tangent[:, 0]
        total += sign * 0.5 * float(np.sum(sample.weights * integrand))
    return total


def local_thickness(suction: np.ndarray, pressure: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(x/c, distance to the pressure polyline) of every suction point aft of the nose,
    increasing in x."""
    pressure = pressure[np.concatenate(([True], np.hypot(*np.diff(pressure, axis=0).T) > 0.0))]
    suction = suction[np.concatenate(([True], np.hypot(*np.diff(suction, axis=0).T) > 0.0))]
    suction = suction[np.argmax(suction[:, 0] > 0.01):]
    start, step = pressure[None, :-1], np.diff(pressure, axis=0)[None]
    out = np.empty(len(suction))
    for chunk in np.array_split(np.arange(len(suction)), max(1, len(suction) // 400)):
        offset = suction[chunk, None, :] - start
        along = np.clip(np.sum(offset * step, axis=2) / np.sum(step * step, axis=2), 0.0, 1.0)
        out[chunk] = np.min(np.linalg.norm(offset - along[..., None] * step, axis=2), axis=1)
    keep = np.concatenate(([True], np.diff(suction[:, 0]) > 0.0))
    return suction[keep, 0], out[keep]


def thickness(points_of) -> tuple[float, float]:
    """Largest distance from a suction point to the pressure polyline, and the x/c of that
    suction point."""
    suction, pressure = points_of("suction"), points_of("pressure")
    pressure = pressure[np.concatenate(([True], np.hypot(*np.diff(pressure, axis=0).T) > 0.0))]
    start, step = pressure[None, :-1], np.diff(pressure, axis=0)[None]
    best, where = 0.0, 0.0
    for chunk in np.array_split(np.arange(len(suction)), max(1, len(suction) // 400)):
        offset = suction[chunk, None, :] - start
        along = np.clip(np.sum(offset * step, axis=2) / np.sum(step * step, axis=2), 0.0, 1.0)
        distance = np.min(np.linalg.norm(offset - along[..., None] * step, axis=2), axis=1)
        i = int(np.argmax(distance))
        if distance[i] > best:
            best, where = float(distance[i]), float(suction[chunk][i, 0])
    return best, where


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare the C1 and C2 surfaces of one NACA 65 cascade.")
    parser.add_argument("--cl0", type=float, default=EXAMPLE.cl0)
    parser.add_argument("--t-max", type=float, default=EXAMPLE.t_max_over_c)
    parser.add_argument("--solidity", type=float, default=EXAMPLE.solidity)
    parser.add_argument("--stagger-deg", type=float, default=EXAMPLE.stagger_deg)
    parser.add_argument("--variant", default=EXAMPLE.profile_variant, choices=geometry.PROFILE_VARIANTS)
    parser.add_argument("--physical-chord", type=float, default=EXAMPLE.physical_chord)
    parser.add_argument("--save", type=Path, default=SETTINGS["save_dir"], help="folder to write the figures to")
    parser.add_argument("--no-show", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    spec = replace(EXAMPLE, cl0=args.cl0, t_max_over_c=args.t_max, solidity=args.solidity,
                   stagger_deg=args.stagger_deg, profile_variant=args.variant, physical_chord=args.physical_chord)
    comparison = build(spec, SETTINGS)
    print(f"{comparison.c1.metadata['designation']}, {spec.profile_variant}, stagger {spec.stagger_deg:g} deg, "
          f"solidity {spec.solidity:g}, chord {comparison.chord:g}")
    print(f"{'quantity':44s} {'C1':>30s}   C2")
    for label, a, b in preservation_table(comparison):
        print(f"{label:44s} {a:>30s}   {b}")
    figures = {"1_geometry": figure_geometry(comparison), "2_deviation": figure_deviation(comparison),
               "3_curvature": figure_curvature(comparison, SETTINGS),
               "4_d2y_dm2": figure_second_derivative(comparison, SETTINGS), "5_beta": figure_slope(comparison, SETTINGS)}
    if args.save is not None:
        args.save.mkdir(parents=True, exist_ok=True)
        for name, fig in figures.items():
            path = args.save / f"geometry_continuity_{name}.png"
            fig.savefig(path, dpi=130)
            print(f"saved {path}")
    if SETTINGS["show"] and not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
