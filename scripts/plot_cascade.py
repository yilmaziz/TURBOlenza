"""
plot_cascade.py
===============
Two windows, built from the same EXAMPLE, SETTINGS and angles.

Window 1, the cascade: the reference blade and its periodic neighbors, the suction and
pressure surfaces, the actual A10 camberline beside the equivalent circular arc and its
blade angles, the pitch and the geometric throat, with close-ups of the LE and TE in the
blade frame. On top of it the passage between the reference suction surface and the
neighbor pressure surface and the inlet extension ahead of it, from
grid.make_passage_boundaries, with the inlet side-boundary angle beta(m) in a panel of its
own.

Window 2, the computational grid from grid.make_grid: both families of grid lines, the
blades, the exact LE and TE beside the first and last blade-wall nodes, the far inlet and
outlet boundaries, the angles and lengths of both extensions, and close-ups of the LE and
TE with the grid of the passage above each blade and its periodic copy below. Each close-up
shades the transition cell between the last periodic station and the first (or last) blade
station, which holds the nose tip (or the tail end), and colours every wall node by the
named surface of geometry.py it lies on (orange suction, blue pressure; o lower wall,
s upper wall), so that a wall running on the other surface near the nose or a Kovach tail
shows.

Dev-only: the package never imports matplotlib, and nothing here feeds back into it.

Usage
    python scripts/plot_cascade.py                                    # the inputs below
    python scripts/plot_cascade.py --cl0 1.2 --t-max 0.10 --solidity 1.0 --stagger-deg 33
    python scripts/plot_cascade.py --variant emery_base --physical-chord 0.05
    python scripts/plot_cascade.py --inlet-angle-deg 30 --exit-angle-deg 22
    python scripts/plot_cascade.py --continuity C2                    # the quintic C2 surfaces
    python scripts/plot_cascade.py --level medium                     # a geometry-driven standard grid
    python scripts/plot_cascade.py --save cascade.png --no-show      # also writes cascade_grid.png

The normal workflow is to edit the inputs below and run the script with no options.
EXAMPLE is a CascadeSpec and every option defaults to its value; the options only replace
fields of that spec, and the cascade is then built the one way there is, make_cascade(spec).
SETTINGS, INLET_ANGLE_DEG and EXIT_ANGLE_DEG are the grid inputs; the angles are converted
to radians here once. EXIT_ANGLE_DEG None starts the exit extension at kappa_te, as the
solver will before its first Kutta update.

EXAMPLE.surface_continuity (or --continuity) picks the surfaces: "C1", the validated
polylines, or "C2", the quintic curves of geometry.py (under validation). The grid reads the
C1 arrays, and for C2 the curves themselves (grid.py section 7). With C2 the close-ups draw the curve itself as the line and its
samples, the arrays, as dots; with C1 the polyline is the geometry, so line and dots agree.

GRID_LEVEL (or --level) picks the grid. None (or --level explicit): the explicit grid of
SETTINGS.n_passage and SETTINGS.n_eta (make_grid, grid.py section 2). One of
grid.GRID_LEVELS ("very_coarse" .. "very_fine"): the geometry-driven standard grid of that
level (grid.build_grid, section 9); SETTINGS then gives only the nominal extension lengths,
the actual ones are whole cells, and both windows show that grid's own boundaries. Its
close-ups also shade the LE (TE) window where no station may lie, nose tip to the start of
the SS + PS-valid interval (its end to the tail end), and mark the cell face at the window's
middle, halfway between the last periodic and the first blade station.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from turbolenza.b2b import geometry, grid  # noqa: E402
from turbolenza.b2b.config import GridSettings  # noqa: E402
from turbolenza.b2b.geometry import Cascade, CascadeSpec  # noqa: E402
from turbolenza.b2b.grid import Grid, PassageBoundaries  # noqa: E402

# --- Development inputs: edit these ---------------------------------------------------
EXAMPLE = CascadeSpec(
    blade_type="NACA65",
    solidity=1.0,
    stagger_deg=33.0,
    cl0=1.2,
    t_max_over_c=0.10,
    meanline_type="A10",
    profile_variant="kovach_sandercock",
    physical_chord=None,
    n_camber_stations=101,
    surface_continuity="C2",            # "C1" validated polylines, "C2" quintic curves (under validation)
)
SETTINGS = GridSettings(
    n_passage=161,                       # target cells along m from LE to TE
    n_eta=71,                           # nodes across the passage
    upstream_extension_chords=1.0,
    downstream_extension_chords=1.0,
)                                       # the blade stations follow from the blade itself: no LE/TE setting
GRID_LEVEL = "coarse"             # None: the explicit SETTINGS grid above; or a standard level (grid.py
                                # section 9): "very_coarse", "coarse", "medium", "fine", "very_fine"
                                # (then only SETTINGS' extension lengths are used, as nominal ones)
INLET_ANGLE_DEG = 45.0         # far-inlet flow angle, degrees from axial
EXIT_ANGLE_DEG = None           # far-exit flow angle, degrees from axial; None = kappa_te, the initial state
# ---------------------------------------------------------------------------------------

BLADES = (0, 1)             # blades drawn, in pitches from the reference blade
CONNECTORS = 5                  # cross-passage lines drawn on the inlet extension
LE_WINDOW = (-0.012, 0.045)     # x/c range of the close-ups, blade frame
TE_WINDOW = (0.955, 1.012)
CURVE_SAMPLES = 20001           # uniform-u points per C2 surface curve drawn in the close-ups

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"
SUCTION = "#eb6834"
PRESSURE = "#2a78d6"
THROAT = "#1baf7a"
ARC = "#8a5cd6"
EXTENSION = "#a15c00"
EXIT_SIDE = "#b8327a"
TRANSITION_FILL = "#f8eec4"
WINDOW_FILL = "#f4c7c3"             # standard level: the LE / TE window, where no station may lie
GRID_LINE = "#8f8d88"
REFERENCE_FILL = "#f3e3d6"
NEIGHBOR_FILL = "#e8e6e0"
PASSAGE_FILL = "#e3eef5"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot a NACA 65 cascade from turbolenza.b2b.geometry.")
    parser.add_argument("--cl0", type=float, default=EXAMPLE.cl0, help="design lift coefficient")
    parser.add_argument("--t-max", type=float, default=EXAMPLE.t_max_over_c,
                        help="nominal thickness ratio of the NACA designation")
    parser.add_argument("--solidity", type=float, default=EXAMPLE.solidity, help="chord / pitch")
    parser.add_argument("--stagger-deg", type=float, default=EXAMPLE.stagger_deg, help="stagger, degrees from axial")
    parser.add_argument("--variant", default=EXAMPLE.profile_variant, choices=geometry.PROFILE_VARIANTS)
    parser.add_argument("--physical-chord", type=float, default=EXAMPLE.physical_chord, help="chord in metres")
    parser.add_argument("--continuity", default=EXAMPLE.surface_continuity, choices=geometry.SURFACE_CONTINUITIES,
                        help="surface representation: C1 validated polylines, C2 quintic curves")
    parser.add_argument("--inlet-angle-deg", type=float, default=INLET_ANGLE_DEG,
                        help="far-inlet flow angle, degrees from axial")
    parser.add_argument("--exit-angle-deg", type=float, default=EXIT_ANGLE_DEG,
                        help="far-exit flow angle, degrees from axial (default: kappa_te)")
    parser.add_argument("--level", default=GRID_LEVEL if GRID_LEVEL is not None else "explicit",
                        choices=("explicit", *grid.GRID_LEVELS),
                        help="the grid: explicit (SETTINGS.n_passage x n_eta) or a geometry-driven standard level")
    parser.add_argument("--save", type=Path, help="also write the cascade figure here and the grid figure beside it")
    parser.add_argument("--no-show", action="store_true", help="do not open the windows")
    return parser.parse_args()


def style(ax: plt.Axes, title: str, xlabel: str, ylabel: str, equal: bool = True) -> None:
    if equal:
        ax.set_aspect("equal")
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.set_xlabel(xlabel, fontsize=9, color=INK_SECONDARY)
    ax.set_ylabel(ylabel, fontsize=9, color=INK_SECONDARY)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8, length=0)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)


def unit_name(cascade: Cascade) -> str:
    return "m" if cascade.is_dimensional else "c"


def surface_lines(cascade: Cascade, unstaggered: bool) -> tuple[np.ndarray, np.ndarray]:
    """(suction, pressure) as the close-ups draw them: for C1 the polylines, which are the
    geometry; for C2 the authoritative quintic curves, densely sampled, of which the arrays
    are only samples. Blade frame in chords when unstaggered, else the cascade frame in the
    active unit."""
    if cascade.normalized_curves is None:
        n, a = cascade.normalized, cascade.active
        return ((n.suction_surface_unstaggered, n.pressure_surface_unstaggered) if unstaggered
                else (a.suction_surface, a.pressure_surface))
    curves = cascade.normalized_curves if unstaggered else cascade.active_curves
    names = ("suction_unstaggered", "pressure_unstaggered") if unstaggered else ("suction", "pressure")
    u = np.linspace(0.0, 1.0, CURVE_SAMPLES)
    return getattr(curves, names[0]).evaluate(u), getattr(curves, names[1]).evaluate(u)


def plot_cascade_view(ax: plt.Axes, cascade: Cascade, boundaries: PassageBoundaries) -> None:
    """The row in the cascade frame, in the active length unit, with the passage and its
    inlet extension from grid.make_passage_boundaries on top."""
    a, unit = cascade.active, unit_name(cascade)
    b = boundaries
    loop = geometry.closed_loop(a.suction_surface, a.pressure_surface)
    passage = np.vstack((b.upstream_lower_boundary, b.lower_blade_boundary[1:], b.upper_blade_boundary[::-1],
                         b.upstream_upper_boundary[-2::-1]))
    ax.fill(*passage.T, color=PASSAGE_FILL, linewidth=0, zorder=0.5, label="passage: suction to neighbor pressure")
    for k in BLADES:
        blade = loop + k * a.periodic_translation
        ax.fill(*blade.T, color=REFERENCE_FILL if k == 0 else NEIGHBOR_FILL, linewidth=0, zorder=1)
        if k != 0:
            ax.plot(*blade.T, color=INK_SECONDARY, linewidth=0.7, zorder=2)

    ax.plot(*a.suction_surface.T, color=SUCTION, linewidth=2.0, zorder=4, label="suction surface")
    ax.plot(*a.pressure_surface.T, color=PRESSURE, linewidth=2.0, zorder=4, label="pressure surface")
    ax.plot(*b.upper_blade_boundary.T, color=PRESSURE, linewidth=2.0, zorder=4)
    plot_inlet_extension(ax, cascade, b)
    ax.plot(*a.camberline.T, color=INK, linewidth=0.9, zorder=3, label="camberline (actual A10)")
    ax.plot(*a.equivalent_arc_camberline.T, color=ARC, linewidth=1.2, linestyle=(0, (4, 3)), zorder=3,
            label="equivalent circular arc")

    # Blade angles of the equivalent arc, drawn upstream of the LE and downstream of the TE.
    for point, angle, sign, name in ((a.leading_edge_point, cascade.kappa_le, -1.0, "κ_le"),
                                     (a.trailing_edge_point, cascade.kappa_te, 1.0, "κ_te")):
        end = point + sign * 0.4 * a.chord * np.array([math.cos(angle), math.sin(angle)])
        ax.plot(*np.vstack((point, end)).T, color=ARC, linewidth=1.0, linestyle="-.", zorder=3)
        ax.annotate(f"{name} = {math.degrees(angle):.1f}°", end, xytext=(6 * sign, 6), textcoords="offset points",
                    ha="right" if sign < 0 else "left", fontsize=9, color=ARC)
    ax.plot(*np.vstack((a.leading_edge_point, a.trailing_edge_point)).T, linestyle="none", marker="o",
            markersize=4, color=INK, zorder=5)

    # Geometric throat.
    ends = np.vstack((a.throat_point_reference, a.throat_point_neighbor))
    ax.plot(*ends.T, color=THROAT, linewidth=2.4, marker="o", markersize=4, zorder=6, label="geometric throat")
    ax.annotate(f"o/s = {cascade.throat_over_pitch:.4f}", ends.mean(axis=0), xytext=(8, 0), textcoords="offset points",
                va="center", fontsize=9, color=THROAT, bbox=dict(facecolor=SURFACE, edgecolor="none", alpha=0.85, pad=1.5),
                zorder=7)

    ax.margins(0.04)
    style(ax, "Cascade frame: reference blade (filled), its periodic neighbors, one passage and its inlet",
          f"x = m, axial [{unit}]", f"y, tangential [{unit}]")
    for m in (b.m_in, b.m_le):
        ax.axvline(m, color=INK_SECONDARY, linewidth=0.7, linestyle=(0, (5, 4)), zorder=0.6)


def plot_inlet_extension(ax: plt.Axes, cascade: Cascade, b: PassageBoundaries) -> None:
    """Lower and upper inlet extension lines, their end points, the pitch between them and
    the lengths and angles that define them."""
    unit, chords = unit_name(cascade), (b.m_le - b.m_in) / b.chord   # the actual length (whole cells with a level)
    lower, upper = b.upstream_lower_boundary, b.upstream_upper_boundary
    ax.plot(*lower.T, color=EXTENSION, linewidth=2.0, zorder=4, label="inlet extension, lower (integrated)")
    ax.plot(*upper.T, color=EXTENSION, linewidth=2.0, linestyle=(0, (6, 2)), zorder=4,
            label="inlet extension, upper = lower + pitch")

    # Cross-passage connectors: every one is the same vertical vector (0, pitch).
    picks = np.unique(np.round(np.linspace(0, len(b.upstream_m) - 1, CONNECTORS)).astype(int))
    for i in picks[1:-1]:
        ax.plot(*np.vstack((lower[i], upper[i])).T, color=INK_SECONDARY, linewidth=0.8, linestyle=(0, (1, 2)),
                zorder=3)
    first = picks[1] if len(picks) > 2 else picks[0]
    ax.annotate("", upper[first], lower[first], arrowprops=dict(arrowstyle="<->", color=INK, linewidth=1.0),
                zorder=6)
    ax.text(lower[first, 0] - 0.02 * b.chord, lower[first, 1] + 0.5 * b.pitch, f"pitch s = {b.pitch:.4g} {unit}",
            rotation=90, ha="right", va="center", fontsize=9, color=INK)

    # Exact LE points of both passage walls, and the far-inlet end points.
    ax.plot(*np.vstack((lower[-1], upper[-1])).T, linestyle="none", marker="o", markersize=5, color=INK,
            zorder=7, label="exact LE points (walls start here)")
    ax.plot(*np.vstack((lower[0], upper[0])).T, linestyle="none", marker="s", markersize=5,
            markerfacecolor=SURFACE, markeredgecolor=EXTENSION, markeredgewidth=1.4, zorder=7,
            label="far-inlet end points, m = m_in")

    # beta_in at the far inlet: the axial direction and the inlet flow direction.
    beta_in = b.upstream_beta[0]
    reach = 0.22 * b.chord
    for point in (lower[0], upper[0]):
        ray = point - reach * np.array([math.cos(beta_in), math.sin(beta_in)])
        ax.plot(*np.vstack((ray, point)).T, color=EXTENSION, linewidth=1.0, linestyle="-.", zorder=3)
        ax.plot(*np.vstack((point - [reach, 0.0], point)).T, color=INK_SECONDARY, linewidth=0.7,
                linestyle=(0, (1, 2)), zorder=3)
    ray_end = lower[0] - reach * np.array([math.cos(beta_in), math.sin(beta_in)])
    ax.annotate(f"β_in = {math.degrees(beta_in):.1f}°", ray_end, xytext=(6, -2), textcoords="offset points",
                ha="left", va="top", fontsize=9, color=EXTENSION)

    # L_up along m, above the upper extension line, from m_in to m_le.
    y = upper[:, 1].max() + 0.15 * b.chord
    ax.annotate("", (b.m_le, y), (b.m_in, y), arrowprops=dict(arrowstyle="<->", color=INK, linewidth=1.0))
    ax.plot([b.m_in, b.m_le], [y + 0.1 * b.chord] * 2, linestyle="none")   # room for the label in the autoscaled view
    length = f"{chords:.4g} c" if not cascade.is_dimensional else f"{chords:.4g} c = {b.m_le - b.m_in:.4g} {unit}"
    ax.text(0.5 * (b.m_in + b.m_le), y + 0.015 * b.chord, f"L_up = {length} along m, in true chords",
            ha="center", va="bottom", fontsize=9, color=INK)
    for m, name, align in ((b.m_in, "m_in", "left"), (b.m_le, "m_le", "right")):
        ax.annotate(f"{name} = {m:.4g}", (m, y), xytext=(3 if align == "left" else -3, -4), textcoords="offset points",
                    ha=align, va="top", fontsize=8.5, color=INK_SECONDARY)


def plot_upstream_angle(ax: plt.Axes, cascade: Cascade, b: PassageBoundaries) -> None:
    """beta(m) of the inlet side boundary: linear in m from beta_in to kappa_le."""
    unit = unit_name(cascade)
    beta = np.degrees(b.upstream_beta)
    ax.plot(b.upstream_m, beta, color=EXTENSION, linewidth=1.6, marker="o", markersize=2.4, zorder=3)
    rising = beta[-1] >= beta[0]
    for m, value, text, start in ((b.m_in, beta[0], f"β_in = {beta[0]:.2f}° at m_in", True),
                                  (b.m_le, beta[-1], f"κ_le = {beta[-1]:.2f}° at m_le", False)):
        ax.plot(m, value, marker="o", markersize=6, color=INK, zorder=4)
        below = start == rising             # put each label on the side of the line with room
        ax.annotate(text, (m, value), xytext=(8 if start else -8, -10 if below else 8), textcoords="offset points",
                    ha="left" if start else "right", va="top" if below else "bottom", fontsize=9, color=INK)
    span = max(abs(beta[-1] - beta[0]), 1.0)
    ax.set_ylim(min(beta[0], beta[-1]) - 0.35 * span, max(beta[0], beta[-1]) + 0.6 * span)
    ax.set_xlim(b.m_in - 0.03 * b.chord, b.m_le + 0.03 * b.chord)
    style(ax, f"Inlet side-boundary angle: linear in m, {len(b.upstream_m) - 1} uniform cells",
          f"m [{unit}]", "β [deg]", equal=False)


def plot_close_up(ax: plt.Axes, cascade: Cascade, end: str) -> None:
    """The LE or TE in the blade frame, chord 1, with the surface points and the edge circle."""
    n = cascade.normalized
    leading = end == "le"
    suction, pressure = n.suction_surface_unstaggered, n.pressure_surface_unstaggered
    center = geometry.rotate_by_stagger(n.leading_edge_center if leading else n.trailing_edge_center, -cascade.stagger)
    radius = n.leading_edge_radius if leading else n.trailing_edge_radius
    window = LE_WINDOW if leading else TE_WINDOW

    for surface, line, color in zip((suction, pressure), surface_lines(cascade, unstaggered=True),
                                    (SUCTION, PRESSURE)):
        ax.plot(*line.T, color=color, linewidth=1.6)
        ax.plot(*surface.T, linestyle="none", marker="o", markersize=2.2, color=color)
    ax.plot(*n.camberline_unstaggered.T, color=INK, linewidth=1.0)
    ax.plot(*n.equivalent_arc_camberline_unstaggered.T, color=ARC, linewidth=1.0, linestyle=(0, (4, 3)))
    if radius > 0.0:
        theta = np.linspace(0.0, 2.0 * math.pi, 361)
        ax.plot(center[0] + radius * np.cos(theta), center[1] + radius * np.sin(theta), color=THROAT,
                linewidth=1.0, linestyle=(0, (2, 2)))
        ax.plot(*center, marker="+", markersize=8, color=THROAT)

    shown = np.vstack([s[(s[:, 0] >= window[0]) & (s[:, 0] <= window[1])] for s in (suction, pressure)])
    pad = 0.15 * (window[1] - window[0])
    ax.set_xlim(*window)
    ax.set_ylim(shown[:, 1].min() - pad, shown[:, 1].max() + pad)
    if leading:
        title = f"Leading edge: R_LE = {radius:.5f} c"
    elif radius > 0.0:
        title = f"Trailing edge: R_TE = {radius:.4f} c (Kovach-Sandercock)"
    else:
        title = "Trailing edge: sharp (emery_base)"
    style(ax, title, "x / c, blade frame", "y / c")


def plot_information(ax: plt.Axes, cascade: Cascade, boundaries: PassageBoundaries, handles: list) -> None:
    """The summary numbers and the one legend that serves all the plots."""
    ax.axis("off")
    ax.text(0.0, 1.0, summary(cascade, boundaries), transform=ax.transAxes, va="top", ha="left", fontsize=8,
            family="monospace", color=INK, linespacing=1.3)
    edge = plt.Line2D([], [], color=THROAT, linewidth=1.0, linestyle=(0, (2, 2)), label="LE / TE circle")
    points = plt.Line2D([], [], color=INK_SECONDARY, linestyle="none", marker="o", markersize=3,
                        label="surface points (close-ups)" if cascade.normalized_curves is None else
                        "C2 curve samples, the arrays (close-ups)")
    ax.legend(handles=[*handles, edge, points], loc="upper right", frameon=False, fontsize=7.5,
              labelcolor=INK_SECONDARY)


def summary(cascade: Cascade, boundaries: PassageBoundaries) -> str:
    """Key numbers, in plain ASCII so that any console code page can print them."""
    a, spec, unit, angles = cascade.active, cascade.spec, unit_name(cascade), cascade.angles_deg()
    reference, neighbor = a.throat_point_reference, a.throat_point_neighbor
    b = boundaries
    cells = len(b.upstream_m) - 1
    return "\n".join((
        f"{cascade.metadata['designation']}, {spec.profile_variant}, {spec.surface_continuity}",
        f"chord {a.chord:.4g} {unit}, pitch {a.pitch:.4g} {unit}",
        f"solidity {spec.solidity:g}, t/c {spec.t_max_over_c:g} nominal",
        f"max thickness {cascade.metadata['max_thickness_over_c']:.5f} c",
        f"R_LE {a.leading_edge_radius:.4g} {unit}, R_TE {a.trailing_edge_radius:.4g} {unit}",
        f"stagger  {angles['stagger_deg']:7.2f} deg",
        f"camber   {angles['camber_angle_deg']:7.2f} deg (equiv. arc)",
        f"kappa_le {angles['kappa_le_deg']:7.2f} deg",
        f"kappa_te {angles['kappa_te_deg']:7.2f} deg",
        f"throat o = {a.throat_opening:.4g} {unit}",
        f"  o/s {cascade.throat_over_pitch:.4f} geometric",
        f"  o/s {cascade.throat_over_pitch_correlation:.4f} correlation",
        f"  from ({reference[0]:.4g}, {reference[1]:.4g}) {unit} suction",
        f"  to   ({neighbor[0]:.4g}, {neighbor[1]:.4g}) {unit} pressure (neighbor)",
        f"inlet: beta_in {math.degrees(b.upstream_beta[0]):.2f} deg -> kappa_le "
        f"{math.degrees(b.upstream_beta[-1]):.2f} deg",
        f"  L_up {b.m_le - b.m_in:.4g} {unit}, {cells} cells of {(b.m_le - b.m_in) / cells:.4g} {unit}",
        f"  m_in {b.m_in:.4g}, m_le {b.m_le:.4g}, m_te {b.m_te:.4g} {unit}",
        f"  far inlet y {b.upstream_lower_boundary[0, 1]:.4g} / {b.upstream_upper_boundary[0, 1]:.4g} {unit}",
    ))


def draw(cascade: Cascade, boundaries: PassageBoundaries) -> plt.Figure:
    """Cascade and passage on the left with beta(m) below it; summary and legend, then the
    LE and TE close-ups, on the right."""
    fig = plt.figure(figsize=(13.5, 10.0), facecolor=SURFACE)
    spec = cascade.spec
    fig.suptitle(f"{cascade.metadata['designation']} cascade, {spec.profile_variant}, {spec.surface_continuity} "
                 f"surfaces: solidity {spec.solidity:g}, stagger {spec.stagger_deg:g}°", x=0.06, ha="left",
                 fontsize=13, color=INK)
    outer = fig.add_gridspec(1, 2, width_ratios=(1.2, 1.0), left=0.06, right=0.98, top=0.93, bottom=0.06,
                             wspace=0.12)
    left = outer[0].subgridspec(2, 1, height_ratios=(3.4, 1.0), hspace=0.22)
    right = outer[1].subgridspec(3, 1, height_ratios=(1.15, 1.0, 1.0), hspace=0.36)
    view = fig.add_subplot(left[0])
    plot_cascade_view(view, cascade, boundaries)
    plot_upstream_angle(fig.add_subplot(left[1]), cascade, boundaries)
    plot_information(fig.add_subplot(right[0]), cascade, boundaries, view.get_legend_handles_labels()[0])
    plot_close_up(fig.add_subplot(right[1]), cascade, "le")
    plot_close_up(fig.add_subplot(right[2]), cascade, "te")
    return fig


# --- Window 2: the computational grid ----------------------------------------------------

def plot_grid_lines(ax: plt.Axes, g: Grid, offset: np.ndarray | None = None, alpha: float = 1.0,
                    label: bool = False) -> None:
    """Both families of grid lines: constant m (the stations) and constant eta. The two
    boundary rows change role across the LE and TE transition intervals (periodic side line
    on one side, blade wall on the other), so those links are not drawn: they are neither a
    wall nor a fluid link."""
    nodes = g.nodes if offset is None else g.nodes + offset
    stations = LineCollection(list(nodes), colors=GRID_LINE, linewidths=0.45, alpha=alpha, zorder=2,
                              label="constant-m grid lines (stations)" if label else None)
    rows = []
    for j in range(len(g.eta)):
        row = nodes[:, j]
        if j in (0, len(g.eta) - 1):
            pieces = (row[:g.first_blade_i], row[g.first_blade_i:g.last_blade_i + 1], row[g.last_blade_i + 1:])
            rows.extend(piece for piece in pieces if len(piece) > 1)
        else:
            rows.append(row)
    ax.add_collection(stations)
    ax.add_collection(LineCollection(rows, colors=GRID_LINE, linewidths=0.45, alpha=alpha, linestyles=(0, (4, 2)),
                                     zorder=2, label="constant-eta grid lines" if label else None))


def blade_row_outlines(cascade: Cascade, copies: tuple[int, ...] = (-1, 0, 1, 2)) -> list[np.ndarray]:
    """Closed outlines of the reference blade and its periodic copies (diagnostics only)."""
    a = cascade.active
    loop = np.vstack((a.suction_surface[::-1], a.pressure_surface[1:]))
    return [loop + k * a.periodic_translation for k in copies]


def crosses_metal(outlines: list[np.ndarray], p: np.ndarray, q: np.ndarray) -> bool:
    """Whether the open segment p-q properly crosses a blade outline (diagnostics only; the
    tests hold the authoritative checks)."""
    d = q - p
    for loop in outlines:
        e0, f = loop[:-1], np.diff(loop, axis=0)
        den = d[0] * f[:, 1] - d[1] * f[:, 0]
        ok = den != 0.0
        w = e0 - p
        t = (w[:, 0] * f[:, 1] - w[:, 1] * f[:, 0]) / np.where(ok, den, 1.0)
        u = (w[:, 0] * d[1] - w[:, 1] * d[0]) / np.where(ok, den, 1.0)
        if np.any(ok & (t > 1e-12) & (t < 1.0 - 1e-12) & (u >= 0.0) & (u <= 1.0)):
            return True
    return False


def plot_transition_links(ax: plt.Axes, cascade: Cascade, g: Grid, i: int, offsets: tuple) -> tuple[int, int]:
    """The interior m-links j = 1 .. n_eta - 2 across the transition interval i -> i + 1,
    green when clear of blade metal and red when not; returns (checked, through metal)."""
    outlines = blade_row_outlines(cascade)
    checked = bad = 0
    for offset in offsets:
        nodes = g.nodes + offset
        for j in range(1, len(g.eta) - 1):
            p, q = nodes[i, j], nodes[i + 1, j]
            hit = crosses_metal(outlines, p, q)
            checked, bad = checked + 1, bad + int(hit)
            ax.plot(*np.vstack((p, q)).T, color="#d62728" if hit else THROAT, linewidth=1.5 if hit else 0.9,
                    alpha=1.0 if hit else 0.8, zorder=5)
    return checked, bad


def plot_grid_view(ax: plt.Axes, cascade: Cascade, g: Grid, standard: grid.StandardGrid | None = None) -> None:
    """The whole grid with the blades, the landmarks, the far boundaries, and the angles and
    lengths that define the two extensions (the actual ones: whole cells with a level)."""
    a, b, unit = cascade.active, g.boundaries, unit_name(cascade)
    s = a.periodic_translation
    loop = geometry.closed_loop(a.suction_surface, a.pressure_surface)
    for k in BLADES:
        blade = loop + k * s
        ax.fill(*blade.T, color=REFERENCE_FILL if k == 0 else NEIGHBOR_FILL, linewidth=0, zorder=1)
        ax.plot(*blade.T, color=INK_SECONDARY, linewidth=0.6, zorder=1.5)
    plot_grid_lines(ax, g, label=True)

    first, last = g.first_blade_i, g.last_blade_i
    lower, upper = g.lower_boundary, g.upper_boundary
    inlet, walls, exit_ = slice(0, first), slice(first, last + 1), slice(last + 1, None)
    ax.plot(*lower[inlet].T, color=EXTENSION, linewidth=1.8, zorder=4, label="inlet side lines, upper = lower + s")
    ax.plot(*upper[inlet].T, color=EXTENSION, linewidth=1.8, linestyle=(0, (6, 2)), zorder=4)
    ax.plot(*lower[exit_].T, color=EXIT_SIDE, linewidth=1.8, zorder=4, label="exit side lines, upper = lower + s")
    ax.plot(*upper[exit_].T, color=EXIT_SIDE, linewidth=1.8, linestyle=(0, (6, 2)), zorder=4)
    ax.plot(*lower[walls].T, color=SUCTION, linewidth=1.0, marker="o", markersize=2.4, zorder=5,
            label="lower wall nodes, reference upper contour (suction)")
    ax.plot(*upper[walls].T, color=PRESSURE, linewidth=1.0, marker="o", markersize=2.4, zorder=5,
            label="upper wall nodes, neighbor lower contour (pressure)")
    ax.plot(*g.nodes[0].T, color=EXTENSION, linewidth=3.2, solid_capstyle="butt", zorder=4,
            label="far inlet boundary, m = m_in")
    ax.plot(*g.nodes[-1].T, color=EXIT_SIDE, linewidth=3.2, solid_capstyle="butt", zorder=4,
            label="far outlet boundary, m = m_out")

    # Exact landmarks against the first and last blade-wall nodes.
    for point, marker, name in ((a.leading_edge_point, "o", "exact LE"), (a.trailing_edge_point, "s", "exact TE")):
        ax.plot(*np.vstack((point, point + s)).T, linestyle="none", marker=marker, markersize=5.5, color=INK,
                zorder=8, label=f"{name} (landmark, not a node)")
    ax.plot(*np.vstack((lower[first], upper[first])).T, linestyle="none", marker="D", markersize=6.5,
            markerfacecolor=SURFACE, markeredgecolor=INK, markeredgewidth=1.5, zorder=8,
            label=f"first blade-wall nodes, i = {first}")
    ax.plot(*np.vstack((lower[last], upper[last])).T, linestyle="none", marker="v", markersize=6.5,
            markerfacecolor=SURFACE, markeredgecolor=INK, markeredgewidth=1.5, zorder=8,
            label=f"last blade-wall nodes, i = {last}")

    # Blade angles at the junctions and flow angles at the far boundaries.
    reach = 0.22 * a.chord
    for point, angle, sign, name in ((a.leading_edge_point, cascade.kappa_le, -1.0, "κ_le"),
                                     (a.trailing_edge_point, cascade.kappa_te, 1.0, "κ_te")):
        end = point + sign * reach * np.array([math.cos(angle), math.sin(angle)])
        ax.plot(*np.vstack((point, end)).T, color=ARC, linewidth=1.1, linestyle="-.", zorder=6)
        ax.annotate(f"{name} = {math.degrees(angle):.2f}°", point, xytext=(8, -12), textcoords="offset points",
                    ha="left", va="top", fontsize=9, color=ARC, zorder=9)
    beta_out_note = "current; = κ_te at start" if g.exit_angle == cascade.kappa_te else "current"
    for point, angle, sign, color, text, align in (
            (lower[0], g.inlet_angle, -1.0, EXTENSION, f"β_in = {math.degrees(g.inlet_angle):.2f}°", "right"),
            (lower[-1], g.exit_angle, 1.0, EXIT_SIDE, f"β_out = {math.degrees(g.exit_angle):.2f}°\n({beta_out_note})",
             "left")):
        end = point + sign * reach * np.array([math.cos(angle), math.sin(angle)])
        ax.plot(*np.vstack((point, end)).T, color=color, linewidth=1.1, linestyle="-.", zorder=6)
        ax.plot(*np.vstack((point, point + sign * np.array([reach, 0.0]))).T, color=INK_SECONDARY, linewidth=0.7,
                linestyle=(0, (1, 2)), zorder=6)
        ax.annotate(text, end, xytext=(4 * sign, 0), textcoords="offset points", ha=align, va="center", fontsize=9,
                    color=color, zorder=9)

    # Extension lengths along m, below the inlet and above the exit.
    for m0, m1, y, name, va, dy in (
            (g.m_in, g.m_le, lower[inlet, 1].min() - 0.12 * a.chord, "L_up", "top", -0.015),
            (g.m_te, g.m_out, upper[exit_, 1].max() + 0.12 * a.chord, "L_down", "bottom", 0.015)):
        ax.annotate("", (m1, y), (m0, y), arrowprops=dict(arrowstyle="<->", color=INK, linewidth=1.0))
        chords = (m1 - m0) / g.chord
        length = f"{chords:.4g} c" if not cascade.is_dimensional else f"{chords:.4g} c = {m1 - m0:.4g} {unit}"
        ax.text(0.5 * (m0 + m1), y + dy * a.chord, f"{name} = {length} along m", ha="center", va=va, fontsize=9,
                color=INK)
    for m in (g.m_in, g.m_le, g.m_te, g.m_out):
        ax.axvline(m, color=INK_SECONDARY, linewidth=0.6, linestyle=(0, (5, 4)), zorder=0.6)
    ax.margins(0.05)
    ax.set_xlim(g.m_in - 0.42 * a.chord, g.m_out + 0.62 * a.chord)       # room for the far-angle labels
    level = "" if standard is None else f" ({standard.level} level)"
    style(ax, f"Grid{level}: {len(g.m)} stations x {len(g.eta)} eta nodes, stations {first} to {last} on the blade walls",
          f"x = m, axial [{unit}]", f"y, tangential [{unit}]")


def plot_grid_close_up(ax: plt.Axes, cascade: Cascade, g: Grid, end: str,
                       transitions: grid.PassageTransitions | None = None) -> None:
    """The LE or TE of the reference blade in the cascade frame, with the grid of the passage
    above it and the periodic copy of that grid one pitch below, so that the upper wall
    nodes (minus one pitch) sit on this blade's pressure surface. The window follows dm: the
    transition interval between the last periodic station and the first (or last) blade
    station, and a few stations beside it. Notes sit in the free space right of the panel."""
    a, unit = cascade.active, unit_name(cascade)
    s, chord = a.periodic_translation, a.chord
    leading = end == "le"
    loop = geometry.closed_loop(a.suction_surface, a.pressure_surface)
    if leading:
        periodic_i, blade_i = g.last_inlet_i, g.first_blade_i
        window = (g.m[periodic_i] - max(0.9 * g.dm, 0.012 * chord), g.m[blade_i] + max(3.0 * g.dm, 0.05 * chord))
        extreme = loop[int(np.argmin(loop[:, 0]))]          # the nose tip, the blade's most upstream point
        landmark, landmark_name = a.leading_edge_point, "exact LE"
    else:
        periodic_i, blade_i = g.first_exit_i, g.last_blade_i
        window = (g.m[blade_i] - max(3.0 * g.dm, 0.05 * chord), g.m[periodic_i] + max(0.9 * g.dm, 0.012 * chord))
        extreme = loop[int(np.argmax(loop[:, 0]))]          # the tail end, the blade's most downstream point
        landmark, landmark_name = a.trailing_edge_point, "exact TE"
    span = (min(g.m[periodic_i], g.m[blade_i]), max(g.m[periodic_i], g.m[blade_i]))

    ax.fill(*loop.T, color=REFERENCE_FILL, linewidth=0, zorder=1)
    for surface, line, color in zip((a.suction_surface, a.pressure_surface), surface_lines(cascade, unstaggered=False),
                                    (SUCTION, PRESSURE)):
        ax.plot(*line.T, color=color, linewidth=1.4, zorder=3)
        ax.plot(*surface.T, linestyle="none", marker="o", markersize=1.6, color=color, zorder=3)
    plot_grid_lines(ax, g)
    plot_grid_lines(ax, g, offset=-s, alpha=0.5)
    for offset, alpha in ((np.zeros(2), 0.9), (-s, 0.45)):         # fluid interior nodes, j = 1 .. n_eta - 2
        ax.plot(*(g.nodes[:, 1:-1] + offset).reshape(-1, 2).T, linestyle="none", marker="o", markersize=1.8,
                color=INK_SECONDARY, alpha=alpha, zorder=4)
    checked, bad = plot_transition_links(ax, cascade, g, min(periodic_i, blade_i), (np.zeros(2), -s))
    lower, upper_down = g.lower_boundary, g.upper_boundary - s
    walls = slice(g.first_blade_i, g.last_blade_i + 1)
    identity = g.wall_identity                      # the named surface each wall node lies on
    for nodes, wall, marker in ((lower[walls], identity.lower, "o"), (upper_down[walls], identity.upper, "s")):
        for surface, color in (("suction", SUCTION), ("pressure", PRESSURE)):
            on = wall.surface == surface
            ax.plot(*nodes[on].T, linestyle="none", marker=marker, markersize=3.6, color=color, zorder=6)

    rows = np.vstack((lower[[periodic_i, blade_i]], upper_down[[periodic_i, blade_i]],
                      loop[(loop[:, 0] >= span[0]) & (loop[:, 0] <= span[1])]))
    pad = 0.012 * chord
    y_low, y_high = rows[:, 1].min() - 2.0 * pad, rows[:, 1].max() + 3.0 * pad
    for i in np.flatnonzero((g.m >= window[0]) & (g.m <= window[1])):
        if i not in (periodic_i, blade_i):          # those two are named in the notes
            ax.text(g.m[i], y_high - 0.3 * pad, f"i={i}", rotation=90, ha="right", va="top", fontsize=7,
                    color=INK_SECONDARY)

    periodic_color = EXTENSION if leading else EXIT_SIDE
    ax.axvspan(*span, color=TRANSITION_FILL, linewidth=0, zorder=0.4)
    if transitions is not None:                     # a standard level: the window and the lattice's cell face
        tr = transitions
        forbidden = (tr.m_tip, tr.m_start) if leading else (tr.m_end, tr.m_tail)
        face = tr.le_face if leading else tr.te_face
        if forbidden[1] > forbidden[0]:
            ax.axvspan(*forbidden, color=WINDOW_FILL, linewidth=0, zorder=0.45)
        ax.axvline(face, color=INK_SECONDARY, linewidth=1.0, linestyle=(0, (1, 1.5)), zorder=4)
    ax.axvline(g.m[periodic_i], color=periodic_color, linewidth=1.8, zorder=4)
    ax.axvline(g.m[blade_i], color=INK, linewidth=1.8, zorder=4)
    ax.axvline(landmark[0], color=INK, linewidth=0.8, linestyle=(0, (3, 2)), zorder=4)
    ax.plot(*landmark, marker="o" if leading else "s", markersize=6.5, color=INK, zorder=8)
    ax.text(landmark[0], y_low + 0.3 * pad, f" {landmark_name} (chord-axis landmark)", rotation=90, ha="left",
            va="bottom", fontsize=7, color=INK)
    ax.plot(*extreme, marker=">" if leading else "<", markersize=6, color=SUCTION, markeredgecolor=INK,
            markeredgewidth=0.8, zorder=8)
    ax.plot(*lower[periodic_i], marker="s", markersize=6, markerfacecolor=SURFACE, markeredgecolor=periodic_color,
            markeredgewidth=1.5, zorder=8)
    ax.plot(*np.vstack((lower[blade_i], upper_down[blade_i])).T, linestyle="none", marker="D" if leading else "v",
            markersize=7, markerfacecolor=SURFACE, markeredgecolor=INK, markeredgewidth=1.5, zorder=8)
    ax.set_anchor("W")              # free space for the notes on the right

    notes = dict(textcoords="axes fraction", ha="left", fontsize=7.5, zorder=9, annotation_clip=False)
    which, offset_name = ("first", "m - m_le") if leading else ("last", "m - m_te")
    offset = g.actual_le_offset if leading else g.actual_te_offset
    for node in (lower[blade_i], upper_down[blade_i]):
        ax.annotate("", node, xytext=(1.04, 0.90), arrowprops=dict(arrowstyle="-", color=INK, linewidth=0.6), **notes)
    ax.text(1.04, 0.90, f"{which} blade station, i = {blade_i}: the\n{which} line that cuts the blade,\n"
            f"{offset_name} = {offset / chord:+.4f} c (automatic);\nwall nodes: top and bottom of the\n"
            f"cut (the upper one less s)", transform=ax.transAxes, ha="left", va="center", fontsize=7.5, color=INK)
    edge, before, after = ("nose tip", "misses", "cuts") if leading else ("tail end", "cuts", "misses")
    ax.annotate(f"The {edge} lies between the\nlast line that {before} the blade\nand the first that {after} it",
                extreme, xytext=(1.04, 0.47), va="center", color=INK,
                bbox=dict(facecolor=TRANSITION_FILL, edgecolor="none", pad=2.0),
                arrowprops=dict(arrowstyle="->", color=INK, linewidth=0.8), **notes)
    other = [(name, int(i), wall_surface) for name, wall, usual in (("upper", identity.upper, "pressure"),
                                                                    ("lower", identity.lower, "suction"))
             for i, wall_surface in zip(identity.stations, wall.surface)
             if wall_surface != usual and window[0] <= g.m[i] <= window[1]]
    split = "LE point" if leading else "TE point"
    if other:
        name, _, wall_surface = other[0]
        stations = ", ".join(str(i) for wall_name, i, _ in other if wall_name == name)
        marker = "s" if name == "upper" else "o"
        identity_note = (f"{name} wall ({marker}) at i = {stations}: on the\n{wall_surface} surface, its contour from\n"
                         f"the {edge} to the {split} (orange SS, blue PS)")
    else:
        identity_note = "every wall node on its own surface\n(o lower, s upper wall;\norange suction, blue pressure)"
    ax.text(1.04, 0.66, identity_note, transform=ax.transAxes, ha="left", va="center", fontsize=7.0,
            color=INK_SECONDARY)
    kind, landmark_m = ("last inlet", g.m_le) if leading else ("first exit", g.m_te)
    ax.annotate(f"{kind} station, i = {periodic_i},\n{offset_name} = {(g.m[periodic_i] - landmark_m) / chord:+.4f} c,"
                f"\nits line clear of the blade", lower[periodic_i], xytext=(1.04, 0.30), va="center",
                color=periodic_color, arrowprops=dict(arrowstyle="-", color=periodic_color, linewidth=0.6), **notes)
    ax.text(1.04, 0.0, f"faint: grid one pitch down; dots:\nfluid nodes. Non-blocking diagnostic:\n{checked} straight "
            f"m-links across the\ntransition, {bad} through metal (red)", transform=ax.transAxes, ha="left",
            va="bottom", fontsize=6.5, color="#d62728" if bad else THROAT)
    title = (f"LE: first blade station at m_le {offset / chord:+.4f} c" if leading else
             f"TE: last blade station at m_te {offset / chord:+.4f} c")
    if transitions is not None:
        title += f"\n{'LE' if leading else 'TE'} window (red, no station) and its middle, the cell face (dotted)"
    ax.set_xlim(*window)
    ax.set_ylim(y_low, y_high)
    style(ax, title + ("" if transitions is not None else " (automatic)"), f"x = m [{unit}]", f"y [{unit}]")


def grid_summary(cascade: Cascade, g: Grid, standard: grid.StandardGrid | None = None) -> str:
    """Key numbers of the grid, in plain ASCII."""
    unit, chord = unit_name(cascade), g.chord
    n_m, n_eta = len(g.m), len(g.eta)
    exit_note = "kappa_te, initial" if g.exit_angle == cascade.kappa_te else "explicit"
    if standard is None:
        dm_target = (g.m_te - g.m_le) / SETTINGS.n_passage
        mode = (f"explicit: n_passage {SETTINGS.n_passage}, dm {g.dm:.6g} {unit} (target {dm_target:.6g}), "
                f"d eta {g.deta:.4g}",)
    else:
        i, found = standard.info, standard.candidates
        mode = (f"level {i.level} (grid.py sec. 9): {standard.timing['total']:.3f} s, {found.tested} candidate(s), "
                f"{len(i.checks)} checks",
                f"dm {i.dm_over_chord:.5f} c (target {i.target_dm_over_chord:.5f}), d eta {g.deta:.4g}, "
                f"aspect {i.cell_aspect:.3f}",
                f"(m_f-m_LE)/c {i.le_gap_over_chord:.5f}, (m_TE-m_l)/c {i.te_gap_over_chord:.5f}, "
                f"L {i.upstream_extension_chords:.4f}/{i.downstream_extension_chords:.4f} c")
    return "\n".join((
        f"grid {n_m} x {n_eta} = {n_m * n_eta} nodes, nodes array {g.nodes.shape}",
        *mode,
        f"m_in {g.m_in:.4g}, m_le {g.m_le:.4g}, m_te {g.m_te:.4g}, m_out {g.m_out:.4g} {unit}",
        f"stations: {g.first_blade_i} inlet, {g.last_blade_i - g.first_blade_i + 1} blade (the lines that cut it), "
        f"{n_m - g.last_blade_i - 1} exit",
        f"  last inlet station  i {g.last_inlet_i:4d}, m - m_le {(g.m[g.last_inlet_i] - g.m_le) / chord:+.5f} c",
        f"  first blade station i {g.first_blade_i:4d}, m - m_le {g.actual_le_offset / chord:+.5f} c",
        f"  last blade station  i {g.last_blade_i:4d}, m - m_te {g.actual_te_offset / chord:+.5f} c",
        f"  first exit station  i {g.first_exit_i:4d}, m - m_te {(g.m[g.first_exit_i] - g.m_te) / chord:+.5f} c",
        f"wall nodes on SS/PS: lower {np.count_nonzero(g.wall_identity.lower.surface == 'suction')}/"
        f"{np.count_nonzero(g.wall_identity.lower.surface == 'pressure')}, upper "
        f"{np.count_nonzero(g.wall_identity.upper.surface == 'suction')}/"
        f"{np.count_nonzero(g.wall_identity.upper.surface == 'pressure')}",
        f"beta_in {math.degrees(g.inlet_angle):.3f} deg, kappa_le {math.degrees(cascade.kappa_le):.3f} deg",
        f"kappa_te {math.degrees(cascade.kappa_te):.3f} deg, beta_out {math.degrees(g.exit_angle):.3f} deg ({exit_note})",
    ))


def plot_grid_information(ax: plt.Axes, cascade: Cascade, g: Grid, handles: list,
                          standard: grid.StandardGrid | None = None) -> None:
    """The grid numbers, and below them the legend of the main view."""
    ax.axis("off")
    ax.text(0.0, 1.0, grid_summary(cascade, g, standard), transform=ax.transAxes, va="top", ha="left", fontsize=7.6,
            family="monospace", color=INK, linespacing=1.22)
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, -0.2), ncol=2, frameon=False, fontsize=7.4,
              labelcolor=INK_SECONDARY, columnspacing=1.2, handlelength=2.2)


def draw_grid(cascade: Cascade, g: Grid, standard: grid.StandardGrid | None = None,
              transitions: grid.PassageTransitions | None = None) -> plt.Figure:
    """The grid on the left; summary and legend, then the LE and TE close-ups, on the right."""
    fig = plt.figure(figsize=(15.0, 9.8), facecolor=SURFACE)
    set_window_title(fig, "Computational grid" if standard is None else f"Computational grid ({standard.level})")
    spec = cascade.spec
    level = "" if standard is None else f", standard level {standard.level}"
    fig.suptitle(f"Computational grid of one passage, {cascade.metadata['designation']} ({spec.surface_continuity}) "
                 f"at stagger {spec.stagger_deg:g}°, solidity {spec.solidity:g}{level}: uniform dm = {g.dm:.5g}, "
                 f"d eta = {g.deta:.4g}", x=0.05, ha="left", fontsize=12.5, color=INK)
    outer = fig.add_gridspec(1, 2, width_ratios=(1.45, 1.0), left=0.05, right=0.985, top=0.93, bottom=0.06,
                             wspace=0.12)
    right = outer[1].subgridspec(3, 1, height_ratios=(1.35, 1.25, 1.25), hspace=0.38)
    view = fig.add_subplot(outer[0])
    plot_grid_view(view, cascade, g, standard)
    plot_grid_information(fig.add_subplot(right[0]), cascade, g, view.get_legend_handles_labels()[0], standard)
    for slot, end in ((right[1], "le"), (right[2], "te")):          # each close-up with room for its notes
        plot_grid_close_up(fig.add_subplot(slot.subgridspec(1, 2, width_ratios=(1.75, 1.0), wspace=0.0)[0]),
                           cascade, g, end, transitions)
    return fig


def set_window_title(fig: plt.Figure, title: str) -> None:
    manager = fig.canvas.manager
    if manager is not None:
        manager.set_window_title(title)


def main() -> None:
    args = parse_arguments()
    spec = replace(EXAMPLE, cl0=args.cl0, t_max_over_c=args.t_max, solidity=args.solidity,
                   stagger_deg=args.stagger_deg, profile_variant=args.variant, physical_chord=args.physical_chord,
                   surface_continuity=args.continuity)
    cascade = geometry.make_cascade(spec)
    inlet_angle = math.radians(args.inlet_angle_deg)
    exit_angle = None if args.exit_angle_deg is None else math.radians(args.exit_angle_deg)
    level = None if args.level == "explicit" else args.level
    standard = transitions = computational_grid = None
    if level is not None:                   # the geometry-driven standard grid: its own boundaries in both windows
        try:
            standard = grid.build_grid(cascade, level, inlet_angle=inlet_angle, exit_angle=exit_angle,
                                       extension_chords=(SETTINGS.upstream_extension_chords,
                                                         SETTINGS.downstream_extension_chords))
            transitions = grid.passage_transitions(cascade)
            computational_grid = standard.grid
        except grid.GridError as error:     # still show the cascade (nominal boundaries), and say why there is no grid
            print(f"no {level} grid: {error}")
    boundaries = (computational_grid.boundaries if computational_grid is not None else
                  grid.make_passage_boundaries(cascade, SETTINGS, inlet_angle, exit_angle))
    print(cascade.metadata["designation"], spec.profile_variant, spec.surface_continuity)
    print(summary(cascade, boundaries))
    figures = [(draw(cascade, boundaries), "")]
    set_window_title(figures[0][0], "Cascade geometry")
    if level is None:
        try:
            computational_grid = grid.make_grid(cascade, SETTINGS, inlet_angle, exit_angle)
        except grid.GridError as error:     # still show the cascade, and say why there is no grid
            print(f"no grid: {error}")
    if computational_grid is not None:
        print(grid_summary(cascade, computational_grid, standard))
        figures.append((draw_grid(cascade, computational_grid, standard, transitions), "_grid"))
    if args.save is not None:
        for fig, suffix in figures:
            path = args.save.with_name(args.save.stem + suffix + args.save.suffix)
            fig.savefig(path, dpi=160, facecolor=SURFACE)
            print(f"saved {path}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
