"""
TE topology and stencil footprints of the periodic PDE rows and the interior rows beside the
trailing edge, on the default C2 cascade (65-(12A10)10 at 33 deg, solidity 1, inlet 45 deg).
Diagnostic only: nothing of the solver is changed.

    python scripts/visualize_te_stencils.py                                    # show
    python scripts/visualize_te_stencils.py --save outputs/te_stencil_diagnostics --no-show

Figures (l = last_blade_i, J = n_eta - 1):
  01_te_geometry               the +pitch blade's tail, stations l .. l+4, the last wall node and
                               the first exit PDE node, n_passage 401 and 101 (n_eta 41)
  02_old_vs_new_te_stencil     the TE row (l+1, J): old central psi_m / psi_mm (the link to the
                               wall node (l, J) through the tail) against the revised forward
                               psi_m, psi_mm, psi_meta, n_passage 401, n_eta 41
  03_n401_interior_crossing    the interior row (l, J-1) whose NE diagonal crosses the tail,
                               n_passage 401, n_eta 121; the same row at n_eta 41 (clean)
  04_n401_index_stencil        that row's nine-point stencil in index space
  05_revised_te_footprint      which nodes the revised TE row reads, per derivative, in index
                               space and in the physical plane
  06_resolution_comparison     the old central TE link at n_passage 61, 101, 201, 401
  07_interior_proposal         the interior row of 03: reviewed stencil against a candidate
                               topology-aware mixed difference (PROPOSED ONLY, not production)
  08_straight_chord_sweep      straight-chord blade intersections of the reviewed rows around
                               both transitions, over n_passage x n_eta (after the TE revision)

Every chord drawn red or green is classified by link_geometry (exact crossings with the C2
curves, tolerance METAL_TOLERANCE = 1e-7 chord, ends touching a wall not counted), and the
script prints the table of them and a plain explanation at the end. A red chord is a
straight-chord blade intersection: a geometric fact, not by itself a proof that the stencil
is invalid (the differences act along the coordinate curves; scripts/audit_transition_mapping.py
audits those).
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, Path(__file__).resolve().parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402

import link_geometry as lg  # noqa: E402
from potential_flow_solver import discretization as dz  # noqa: E402
from potential_flow_solver.geometry import CascadeSpec, make_cascade  # noqa: E402

SPEC = CascadeSpec(blade_type="NACA65", solidity=1.0, stagger_deg=33.0, cl0=1.2, t_max_over_c=0.10,
                   surface_continuity="C2")
INK, MUTED, GRIDLINE = "#1f2328", "#6e7781", "#c9ced6"
BLADE_FILL, BLADE_EDGE = "#e4e7eb", "#57606a"
FLUID, WALL, METAL, OK = "#2a78d6", "#eb6834", "#cf222e", "#1baf7a"
STYLES = {"psi_m": "-", "psi_mm": "--", "psi_meta": ":"}

TABLE: list[dict] = []          # every link drawn red or green, for the printed table


def record(figure: str, g, target, node, direction: str, use: str, crossing: lg.Crossing) -> None:
    TABLE.append(dict(figure=figure, case=f"{len(g.m)}x{len(g.eta)}", target=target, node=node, direction=direction,
                      use=use, length=crossing.length, inside=crossing.inside_length, fraction=crossing.fraction,
                      clear=crossing.clear, copies=crossing.copies))


def style_axes(ax, title: str) -> None:
    ax.set_title(title, fontsize=9.5, color=INK, loc="left")
    ax.tick_params(labelsize=7.5, colors=MUTED)
    for spine in ax.spines.values():
        spine.set_color(GRIDLINE)


def draw_blades(ax, blade: lg.Blade, copies=(0, 1)) -> None:
    for k in copies:
        loop = blade.outline + np.array([0.0, k * blade.pitch])
        ax.fill(loop[:, 0], loop[:, 1], color=BLADE_FILL, zorder=1)
        ax.plot(loop[:, 0], loop[:, 1], color=BLADE_EDGE, lw=1.0, zorder=2)


def draw_grid(ax, g, stations, rows) -> None:
    for i in stations:
        ax.axvline(g.m[i], color=GRIDLINE, lw=0.6, zorder=0)
    for j in rows:
        ax.plot(g.nodes[list(stations), j, 0], g.nodes[list(stations), j, 1], color=GRIDLINE, lw=0.6, zorder=0)


def draw_link(ax, a, b, crossing: lg.Crossing, lw: float = 2.0, ls: str = "-", color_ok: str = OK) -> None:
    ax.plot([a[0], b[0]], [a[1], b[1]], color=color_ok, lw=lw, ls=ls, zorder=5)
    for p0, p1, _ in crossing.pieces:
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=METAL, lw=lw + 1.6, solid_capstyle="butt", zorder=6)


def node_kind_marker(ax, g, node, size: float = 4.5) -> None:
    i, j = node
    wall = g.first_blade_i <= i <= g.last_blade_i and j in (0, len(g.eta) - 1)
    ax.plot(*g.nodes[node], "D" if wall else "o", ms=size + (1.5 if wall else 0), color=WALL if wall else FLUID, zorder=7)


def window(ax, centre, half_m: float, half_y: float) -> None:
    ax.set_xlim(centre[0] - half_m, centre[0] + half_m)
    ax.set_ylim(centre[1] - half_y, centre[1] + half_y)
    ax.set_aspect("equal")


# --- figures ---------------------------------------------------------------------------------------

def figure_01(cascade, blade, cases) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.5))
    te = np.asarray(cascade.active.trailing_edge_point, dtype=float) + np.array([0.0, blade.pitch])
    tail_k = int(np.argmax(blade.outline[:, 0]))
    tail = blade.outline[tail_k] + np.array([0.0, blade.pitch])
    for ax, (n_passage, n_eta) in zip(axes, ((401, 41), (101, 41))):
        g = cases[(n_passage, n_eta)][0]
        l, J = g.last_blade_i, n_eta - 1
        stations = range(l - 2, l + 6)
        draw_blades(ax, blade)
        draw_grid(ax, g, stations, range(J - 3, J + 1))
        for i in stations:
            for j in range(J - 3, J + 1):
                node_kind_marker(ax, g, (i, j), 3.5 if j < J else 4.5)
            label = "l" if i == l else f"l+{i - l}" if i > l else f"l{i - l}"
            ax.text(g.m[i], 0.01, label, transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=8,
                    color=INK, bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))
        ax.plot(*te, "x", color=INK, ms=9, mew=2, zorder=8)
        ax.annotate("chord-axis TE (+pitch)", te, xytext=(12, -28), textcoords="offset points", fontsize=8, color=INK,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
        ax.plot(*tail, "+", color=INK, ms=11, mew=2, zorder=8)
        ax.annotate("tail end (most downstream point)", tail, xytext=(10, 22), textcoords="offset points", fontsize=8,
                    color=INK, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
        for i in range(l, l + 5):
            ax.annotate(f"({i}, {J})", g.nodes[i, J], xytext=(4, 6 if i != l else -14), textcoords="offset points",
                        fontsize=7.5, color=INK, zorder=9)
        ax.plot(*g.nodes[l, J], "D", ms=9, mfc="none", mec=INK, mew=1.4, zorder=9)
        ax.plot(*g.nodes[l + 1, J], "o", ms=11, mfc="none", mec=INK, mew=1.6, zorder=9)
        centre = 0.5 * (g.nodes[l, J] + g.nodes[l + 3, J])
        half = max(4.6 * g.dm, 0.014)
        window(ax, centre, half, 0.62 * half)
        style_axes(ax, f"n_passage {n_passage}, n_eta {n_eta}: last blade station l = {l}, first exit station l+1 = {l + 1}")
    handles = [plt.Line2D([], [], marker="D", ls="", color=WALL, ms=6, label="wall node (blade station)"),
               plt.Line2D([], [], marker="o", ls="", color=FLUID, ms=5, label="fluid / periodic-line node"),
               plt.Line2D([], [], marker="D", ls="", mfc="none", mec=INK, ms=9, label="last wall node (l, J)"),
               plt.Line2D([], [], marker="o", ls="", mfc="none", mec=INK, ms=10, label="first exit PDE node (l+1, J)")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8.5, frameon=False)
    fig.suptitle("01  TE of the +pitch blade (upper wall of the passage) and the stations around it", fontsize=11.5,
                 color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    return fig


def te_row(cases, key):
    g, _, _, rows = cases[key]
    return g, rows[g.last_blade_i + 1]


def figure_02(blade, cases) -> plt.Figure:
    g, row = te_row(cases, (401, 41))
    l, J = g.last_blade_i, len(g.eta) - 1
    target = g.nodes[l + 1, J]
    fig, axes = plt.subplots(1, 3, figsize=(19, 8.6), gridspec_kw=dict(width_ratios=(1.25, 1.25, 0.8)))
    stations = range(l - 1, l + 6)
    for ax in axes:
        draw_blades(ax, blade)
        draw_grid(ax, g, stations, range(J - 3, J + 1))
        for i in stations:
            for j in range(J - 2, J + 1):
                node_kind_marker(ax, g, (i, j), 3.2)
    box = dict(boxstyle="round,pad=0.4", fc="white", ec=GRIDLINE)
    # left: old central psi_m, psi_mm
    ax = axes[0]
    old = blade.crossing(target, g.nodes[l, J])
    east = blade.crossing(target, g.nodes[l + 2, J])
    draw_link(ax, target, g.nodes[l, J], old, lw=2.6, color_ok=FLUID)
    draw_link(ax, target, g.nodes[l + 2, J], east, lw=2.6, color_ok=FLUID)
    record("02 old", g, (l + 1, J), (l, J), "W", "psi_m, psi_mm (old central)", old)
    record("02 old", g, (l + 1, J), (l + 2, J), "E", "psi_m, psi_mm (old central)", east)
    mid = 0.5 * (target + g.nodes[l, J])
    ax.annotate(f"(l, J) -> (l+1, J): link {old.length:.2e} c,\n{old.inside_length:.2e} c inside the tail "
                f"({100 * old.fraction:.0f} %)", mid, xytext=(25, -75), textcoords="offset points", fontsize=8.5,
                color=METAL, arrowprops=dict(arrowstyle="->", color=METAL, lw=1.0),
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=METAL))
    ax.text(0.02, 0.98, "OLD (central in m)\n"
                        "psi_m  = (psi[l+2] - psi[l]) / (2 dm)\n"
                        "psi_mm = (psi[l+2] - 2 psi[l+1] + psi[l]) / dm^2\n"
                        "both read the wall node (l, J) through the +pitch tail",
            transform=ax.transAxes, va="top", fontsize=8.2, color=INK, family="monospace", bbox=box)
    # middle: revised psi_m, psi_mm along j = J
    ax = axes[1]
    for offset, style, name, rad in ((1, "-", "psi_m", 0.35), (2, "-", "psi_m", 0.35), (1, "--", "psi_mm", -0.35),
                                     (2, "--", "psi_mm", -0.35), (3, "--", "psi_mm", -0.35)):
        node = (l + 1 + offset, J)
        c = blade.crossing(target, g.nodes[node])
        ax.add_patch(FancyArrowPatch(target, g.nodes[node], connectionstyle=f"arc3,rad={rad}", arrowstyle="-",
                                     color=OK if c.clear else METAL, lw=1.8, ls=style, zorder=5))
        record("02 new", g, (l + 1, J), node, f"+{offset},J", name, c)
    ax.text(0.02, 0.98, "REVISED (forward one-sided in m)\n"
                        "psi_m  = (-3 psi[l+1] + 4 psi[l+2] - psi[l+3]) / (2 dm)   solid arcs\n"
                        "psi_mm = (2 psi[l+1] - 5 psi[l+2] + 4 psi[l+3] - psi[l+4]) / dm^2   dashed\n"
                        "the wall node (l, J) is not read",
            transform=ax.transAxes, va="top", fontsize=8.0, color=INK, family="monospace", bbox=box)
    # right: revised psi_meta
    ax = axes[2]
    for k in range(l + 1, l + 4):
        low, up = g.nodes[k, J - 1], lg.physical(g, (k, 1))
        for node, end in (((k, J - 1), low), ((k, 1), up)):
            c = blade.crossing(target, end)
            record("02 new", g, (l + 1, J), node, f"+{k - l - 1},{'1 (+pitch)' if node[1] == 1 else 'J-1'}",
                   "psi_meta" if k != l + 1 else "psi_meta, psi_eta, psi_etaeta", c)
            ax.plot([target[0], end[0]], [target[1], end[1]], color=OK if c.clear else METAL, lw=1.2, ls=":", zorder=4)
        ax.plot([low[0], up[0]], [low[1], up[1]], color=MUTED, lw=0.8, ls=(0, (1, 2)), zorder=3)
        ax.plot(*up, "s", ms=6, mfc="white", mec=FLUID, mew=1.4, zorder=8)
        ax.annotate(f"q[{k}]", up, xytext=(4, 4), textcoords="offset points", fontsize=7.5, color=INK)
    ax.text(0.03, 0.985, "REVISED psi_meta (forward in m)\n= (-3 q[l+1] + 4 q[l+2] - q[l+3]) / (2 dm)\n"
                         "q[k] = (psi[k,1] + 1 - psi[k,J-1]) / (2 deta)\n"
                         "squares: psi[k,J+1] = psi[k,1] + 1, one pitch up",
            transform=ax.transAxes, va="top", fontsize=7.6, color=INK, family="monospace", bbox=box)
    zoom = 0.5 * (g.nodes[l, J] + g.nodes[l + 4, J])
    tall = zoom + np.array([0.0, 0.0045])
    for ax, title, (centre, hm, hy) in zip(
            axes, (f"OLD central psi_m, psi_mm at ({l + 1}, {J})", f"REVISED psi_m, psi_mm at ({l + 1}, {J})",
                   f"REVISED psi_meta at ({l + 1}, {J})"),
            ((zoom, 0.0072, 0.0058), (zoom, 0.0072, 0.0058), (tall, 0.0105, 0.0325))):
        ax.plot(*target, "o", ms=12, mfc="white", mec=INK, mew=1.8, zorder=9)
        ax.plot(*g.nodes[l, J], "D", ms=9, mfc="none", mec=INK, mew=1.4, zorder=9)
        window(ax, centre, hm, hy)
        style_axes(ax, title)
    fig.suptitle(f"02  The TE periodic row (n_passage 401, n_eta 41, dm = {g.dm:.4f} c): the old central m differences "
                 f"read through the tail, the revised ones look downstream only", fontsize=11.5, color=INK, x=0.01,
                 ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def interior_panel(ax, g, blade, target, title: str, figure: str, highlight: bool = True) -> list:
    i, j = target
    stations = range(i - 2, i + 3)
    draw_blades(ax, blade)
    draw_grid(ax, g, stations, range(max(0, j - 2), min(len(g.eta), j + 3)))
    found = []
    start = g.nodes[target]
    for name, (di, dj) in lg.INTERIOR_DIRECTIONS.items():
        node = (i + di, j + dj)
        c = blade.crossing(start, g.nodes[node])
        record(figure, g, target, node, name, lg.INTERIOR_USE[name], c)
        draw_link(ax, start, g.nodes[node], c, lw=2.0)
        node_kind_marker(ax, g, node, 5)
        ax.annotate(f"{name} {node}", g.nodes[node], xytext=(6 * di + 3, 6 * dj + 2), textcoords="offset points",
                    fontsize=7.8, color=METAL if not c.clear else INK, zorder=10)
        if not c.clear:
            found.append((name, node, c))
            mid = 0.5 * (c.pieces[0][0] + c.pieces[-1][1])
            ax.annotate(f"metal intersection = {c.inside_length:.2e} c\n({100 * c.fraction:.0f} % of the "
                        f"{c.length:.2e} c link, blade copy {c.copies})", mid, xytext=(30, -70), textcoords="offset points",
                        fontsize=8.5, color=METAL, arrowprops=dict(arrowstyle="->", color=METAL),
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=METAL))
    ax.plot(*start, "o", ms=13, mfc="white", mec=INK, mew=1.8, zorder=11)
    ax.annotate(f"P {target}", start, xytext=(-62, -4), textcoords="offset points", fontsize=8.5, color=INK, zorder=11)
    reach = max(float(np.max(np.abs(g.nodes[i + di, j + dj] - start))) for di, dj in lg.INTERIOR_DIRECTIONS.values())
    window(ax, start, 1.35 * reach, 1.35 * reach)
    style_axes(ax, title)
    return found


def figure_03(blade, cases):
    g121 = cases[(401, 121)][0]
    g41 = cases[(401, 41)][0]
    l = g121.last_blade_i
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.8))
    found = interior_panel(axes[0], g121, blade, (l, 119), f"n_passage 401, n_eta 121: interior row (l, J-1) = ({l}, 119)",
                           "03 n_eta 121")
    interior_panel(axes[1], g41, blade, (l, 39), f"n_passage 401, n_eta 41: the same row (l, J-1) = ({l}, 39)",
                   "03 n_eta 41")
    handles = [plt.Line2D([], [], color=OK, lw=2, label="link stays in the fluid"),
               plt.Line2D([], [], color=METAL, lw=3.6, label="part of the link inside the blade"),
               plt.Line2D([], [], marker="D", ls="", color=WALL, ms=6, label="wall node"),
               plt.Line2D([], [], marker="o", ls="", color=FLUID, ms=5, label="fluid node")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8.5, frameon=False)
    fig.suptitle("03  The interior row beside the TE whose NE diagonal (psi_meta) runs through the +pitch tail", fontsize=11.5,
                 color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    return fig, found


def figure_04(g, target, found) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8.5, 7.2))
    i, j = target
    bad = {name for name, _, _ in found}
    for name, (di, dj) in {"P": (0, 0), **lg.INTERIOR_DIRECTIONS}.items():
        x, y = di, dj
        if name != "P":
            ax.plot([0, x], [0, y], color=METAL if name in bad else OK, lw=3.2 if name in bad else 1.8, zorder=1)
        circle_color = METAL if name in bad else (INK if name == "P" else FLUID)
        ax.plot(x, y, "o", ms=34, mfc="white", mec=circle_color, mew=2.2, zorder=2)
        ax.text(x, y + 0.06, name, ha="center", va="center", fontsize=11, color=circle_color, weight="bold", zorder=3)
        ax.text(x, y - 0.24, f"({i + di}, {j + dj})", ha="center", va="center", fontsize=8.5, color=INK, zorder=3)
    for name, node, c in found:
        ax.text(0.5, 1.32, f"{name} = {node}: {100 * c.fraction:.0f} % of its link inside the blade", ha="center",
                fontsize=9.5, color=METAL)
    ax.text(0, -1.55, "mixed term of Eq. 5-21:  -2 B psi_meta,\n"
                      "psi_meta = (psi_NE - psi_NW - psi_SE + psi_SW) / (4 dm deta)   [Eq. 5-39]\n"
                      "the NE weight  -B / (2 dm deta)  multiplies the value across the tail",
            ha="center", va="top", fontsize=9, color=INK, family="monospace")
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-2.2, 1.55)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"04  Index space: the nine-point stencil of P = {target} (n_passage 401, n_eta 121)", fontsize=11, color=INK,
                 loc="left")
    fig.tight_layout()
    return fig


def figure_05(blade, cases) -> plt.Figure:
    g, row = te_row(cases, (401, 41))
    l, J = g.last_blade_i, len(g.eta) - 1
    fig = plt.figure(figsize=(17, 8.5))
    grid = fig.add_gridspec(1, 4, width_ratios=(1, 1, 1.15, 1.0))
    titles = ("psi_m: forward, P, l+2, l+3", "psi_mm: forward, P, l+2, l+3, l+4",
              "psi_meta: q[k] at l+1 .. l+3 from (k, J-1), (k, 1) + 1")
    columns = list(range(l, l + 5))
    levels = ("J-1", "J", "J+1 = 1 (+1)")
    uses = ({(1, 1), (2, 1), (3, 1)}, {(1, 1), (2, 1), (3, 1), (4, 1)},
            {(1, 0), (1, 2), (2, 0), (2, 2), (3, 0), (3, 2)})
    weights = ({1: "-3", 2: "+4", 3: "-1"}, {1: "+2", 2: "-5", 3: "+4", 4: "-1"},
               {(1, 2): "-3", (2, 2): "+4", (3, 2): "-1", (1, 0): "+3", (2, 0): "-4", (3, 0): "+1"})
    for panel, (title, used, w) in enumerate(zip(titles, uses, weights)):
        ax = fig.add_subplot(grid[0, panel])
        for c, i in enumerate(columns):
            for r, level in enumerate(levels):
                blade_column = i == l
                on = (c, r) in used
                face = "white" if not on else (FLUID if r != 2 else "white")
                edge = GRIDLINE if blade_column and r != 2 and not on else (FLUID if on else MUTED)
                marker = "s" if r == 2 else ("D" if blade_column and r == 1 else "o")
                ax.plot(c, r, marker, ms=15 if on else 9, mfc=face if not blade_column else ("white" if not on else face),
                        mec=WALL if (blade_column and r == 1) else edge, mew=1.6, zorder=3)
                key = c if panel < 2 else (c, r)
                if on and key in w:
                    ax.text(c + 0.22, r + 0.18, w[key], fontsize=9, color=INK, zorder=4)
        ax.axvspan(-0.45, 0.45, color=BLADE_FILL, zorder=0)
        ax.text(0, -0.75, "station l\n(blade: not read)", ha="center", fontsize=8, color=MUTED)
        ax.plot(1, 1, "o", ms=22, mfc="none", mec=INK, mew=1.6, zorder=5)
        if panel == 2:
            for c in (1, 2, 3):
                ax.plot([c, c], [0, 2], color=FLUID, ls=":", lw=1.4, zorder=2)
                ax.text(c - 0.12, 1.55, f"q[l+{c}]", fontsize=8, color=INK, rotation=90)
        else:
            ax.plot([1, max(k for k in w)], [1, 1], color=FLUID, ls="-" if panel == 0 else "--", lw=1.6, zorder=2)
        ax.set_xticks(range(len(columns)), ["l", "l+1 (P)", "l+2", "l+3", "l+4"], fontsize=8)
        ax.set_yticks(range(len(levels)), levels, fontsize=8)
        ax.set_xlim(-0.6, 4.6)
        ax.set_ylim(-1.0, 2.7)
        style_axes(ax, title)
        for spine in ax.spines.values():
            spine.set_visible(False)
    ax = fig.add_subplot(grid[0, 3])
    draw_blades(ax, blade)
    stations = range(l - 1, l + 6)
    draw_grid(ax, g, stations, range(J - 3, J + 1))
    target = g.nodes[l + 1, J]
    for node in row.nodes:
        if node == row.target:
            continue
        end = lg.physical(g, node)
        use = lg.periodic_use(row, node, {o for o, _ in dz.MIXED_M_DIFFERENCE[row.kind]})
        c = blade.crossing(target, end)
        record("05", g, row.target, node, f"{node[0] - l - 1:+d},{node[1]}", use, c)
        ls = "-" if use.startswith("psi_m,") else ":"
        ax.plot([target[0], end[0]], [target[1], end[1]], color=OK if c.clear else METAL, ls=ls, lw=1.6, zorder=5)
        ax.plot(*end, "s" if node[1] == 1 else "o", ms=6, mfc="white" if node[1] == 1 else FLUID, mec=FLUID, zorder=7)
    ax.plot(*target, "o", ms=12, mfc="white", mec=INK, mew=1.8, zorder=9)
    ax.plot(*g.nodes[l, J], "D", ms=8, color=WALL, zorder=8)
    ax.annotate(f"(l, J) = ({l}, {J}): not read", g.nodes[l, J], xytext=(-120, -30), textcoords="offset points", fontsize=8,
                color=INK, arrowprops=dict(arrowstyle="-", color=MUTED))
    window(ax, 0.5 * (g.nodes[l, J] + g.nodes[l + 4, J]) + np.array([0.0, 0.006]), 0.0105, 0.031)
    style_axes(ax, f"physical plane: row ({l + 1}, {J}), every link checked")
    fig.suptitle("05  Which nodes the revised TE equation reads (n_passage 401, n_eta 41): only stations l+1 .. l+4",
                 fontsize=11.5, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def figure_06(blade, cases) -> plt.Figure:
    fig, axes = plt.subplots(1, 4, figsize=(19, 6.0))
    for ax, n_passage in zip(axes, (61, 101, 201, 401)):
        g = cases[(n_passage, 41)][0]
        l, J = g.last_blade_i, 40
        draw_blades(ax, blade)
        draw_grid(ax, g, range(l - 3, l + 4), range(J - 3, J + 1))
        for i in range(l - 3, l + 4):
            node_kind_marker(ax, g, (i, J), 4)
        c = blade.crossing(g.nodes[l + 1, J], g.nodes[l, J])
        record("06", g, (l + 1, J), (l, J), "W", "psi_m, psi_mm (old central)", c)
        draw_link(ax, g.nodes[l + 1, J], g.nodes[l, J], c, lw=2.4, color_ok=FLUID)
        ax.plot(*g.nodes[l + 1, J], "o", ms=10, mfc="white", mec=INK, mew=1.6, zorder=9)
        ax.text(0.03, 0.04, f"{c.inside_length:.1e} of {c.length:.1e} c inside ({100 * c.fraction:.0f} %)",
                transform=ax.transAxes, fontsize=8.5, color=METAL if not c.clear else INK,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GRIDLINE))
        half = max(3.2 * g.dm, 0.012)
        window(ax, 0.5 * (g.nodes[l, J] + g.nodes[l + 1, J]), half, 0.8 * half)
        style_axes(ax, f"n_passage {n_passage}: old link ({l + 1}, J) -> ({l}, J)")
    fig.suptitle("06  The old central TE link at four resolutions (n_eta 41): where the last blade station falls in the tail "
                 "sets how much of it is metal", fontsize=11.5, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def figure_07(blade, cases) -> plt.Figure:
    g = cases[(401, 121)][0]
    l, J = g.last_blade_i, 120
    target = (l, J - 1)
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.8))
    interior_panel(axes[0], g, blade, target, f"reviewed nine-point stencil at {target}", "07 reviewed")
    ax = axes[1]
    stations = range(l - 3, l + 3)
    draw_blades(ax, blade)
    draw_grid(ax, g, stations, range(J - 3, J + 1))
    start = g.nodes[target]
    proposal = [("W", (l - 1, J - 1), "psi_m, psi_mm (central, kept)"), ("E", (l + 1, J - 1), "psi_m, psi_mm (central, kept)"),
                ("S", (l, J - 2), "psi_eta, psi_etaeta; r[l]"), ("N", (l, J), "psi_eta, psi_etaeta; r[l]"),
                ("r[l-1] up", (l - 1, J), "psi_meta (backward)"), ("r[l-1] down", (l - 1, J - 2), "psi_meta (backward)"),
                ("r[l-2] up", (l - 2, J), "psi_meta (backward)"), ("r[l-2] down", (l - 2, J - 2), "psi_meta (backward)")]
    for name, node, use in proposal:
        c = blade.crossing(start, g.nodes[node])
        record("07 proposed", g, target, node, name, use, c)
        draw_link(ax, start, g.nodes[node], c, lw=2.0, ls=":" if "meta" in use else "-")
        node_kind_marker(ax, g, node, 5)
        ax.annotate(name, g.nodes[node], xytext=(4, 4), textcoords="offset points", fontsize=7.5, color=INK)
    ax.plot(*start, "o", ms=13, mfc="white", mec=INK, mew=1.8, zorder=11)
    ax.text(0.02, 0.98, "PROPOSED ONLY — NOT YET PRODUCTION\n"
                        "psi_meta = (3 r[l] - 4 r[l-1] + r[l-2]) / (2 dm),  r[k] = (psi[k, J] - psi[k, J-2]) / (2 deta)\n"
                        "(backward in m: stays on blade stations; psi_m, psi_mm, psi_eta, psi_etaeta unchanged)",
            transform=ax.transAxes, va="top", fontsize=8.5, color=INK,
            bbox=dict(boxstyle="round,pad=0.4", fc="#fff8e6", ec=WALL))
    half = max(2.6 * g.dm, 2.4 * g.deta * g.pitch)
    window(ax, start + np.array([-0.6 * g.dm, 0.0]), half * 1.2, half)
    style_axes(ax, "candidate topology-aware mixed difference (not implemented)")
    fig.suptitle("07  The interior row of figure 03: reviewed stencil and the smallest local candidate, for review only",
                 fontsize=11.5, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def figure_08(results) -> plt.Figure:
    n_passages = sorted({r["n_passage"] for r in results})
    n_etas = sorted({r["n_eta"] for r in results})
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))
    for ax, side in zip(axes, ("TE", "LE")):
        counts = np.full((len(n_etas), len(n_passages)), np.nan)
        for r in results:
            counts[n_etas.index(r["n_eta"]), n_passages.index(r["n_passage"])] = sum(x["side"] == side for x in r["straight_chord_intersections"])
        image = ax.imshow(counts, cmap="Oranges", vmin=0, vmax=max(5, np.nanmax(counts)), origin="lower", aspect="auto")
        for a in range(len(n_etas)):
            for b in range(len(n_passages)):
                if not np.isnan(counts[a, b]):
                    ax.text(b, a, f"{int(counts[a, b])}", ha="center", va="center", fontsize=10,
                            color="white" if counts[a, b] >= 3 else INK)
                else:
                    ax.text(b, a, "-", ha="center", va="center", fontsize=9, color=MUTED)
        ax.set_xticks(range(len(n_passages)), n_passages, fontsize=8.5)
        ax.set_yticks(range(len(n_etas)), n_etas, fontsize=8.5)
        ax.set_xlabel("n_passage", fontsize=9, color=INK)
        ax.set_ylabel("n_eta", fontsize=9, color=INK)
        style_axes(ax, f"{side}: straight-chord blade intersections, rows at the {side} transition")
        fig.colorbar(image, ax=ax, fraction=0.04, pad=0.02).ax.tick_params(labelsize=7.5)
    fig.suptitle("08  Straight-chord blade intersections after the TE row revision (periodic rows: none); every one is "
                 "an interior row beside a wall", fontsize=11.5, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


# --- report ---------------------------------------------------------------------------------------

def print_table() -> None:
    print("\nLinks drawn red or green (exact crossings with the C2 curves; tolerance 1e-7 chord):")
    print("  figure        case     target       neighbour    direction       chord [c]  inside [c]  inside %  clear  copy  "
          "derivative")
    for t in TABLE:
        print(f"  {t['figure']:13s} {t['case']:8s} {str(t['target']):12s} {str(t['node']):12s} {t['direction']:14s} "
              f"{t['length']:.3e}  {t['inside']:.3e}  {100 * t['fraction']:7.2f}  {str(t['clear']):5s}  "
              f"{str(t['copies']):5s} {t['use']}")


def print_sweep(results) -> None:
    print("\nSweep (after the TE row revision): straight-chord blade intersections, rows at l-3 .. l+4 and f-4 .. f+3")
    for r in results:
        old = r["old_te"]
        listing = "; ".join(f"{x['side']} {x['row'][:8]} {x['target']} [{x['station']}] {x['direction']} -> {x['node']} "
                            f"{x['crossing'].inside_length:.2e} c ({100 * x['crossing'].fraction:.0f} %) "
                            f"copy {x['crossing'].copies}, {x['use']}" for x in r["straight_chord_intersections"])
        print(f"  n_passage {r['n_passage']:3d}, n_eta {r['n_eta']:3d}: {r['checked']:5d} chords, "
              f"{len(r['straight_chord_intersections'])} intersect the blade; "
              f"old central TE link {100 * old.fraction:3.0f} % metal" + (f"  | {listing}" if listing else ""))


def print_explanation(found, results) -> None:
    rows_affected = {(r["n_passage"], r["n_eta"], x["target"]) for r in results for x in r["straight_chord_intersections"]}
    periodic_bad = sum(x["row"] != "interior" for r in results for x in r["straight_chord_intersections"])
    print("\nExplanation")
    print("1. (l, J) and (l+1, J) are neighbours in the index array, but (l, J) is the upper wall node of the last blade")
    print("   station: it sits on the +pitch blade's lower surface just ahead of the rounded tail, while (l+1, J) is on the")
    print("   periodic line just behind the tail. The straight segment between them cuts through the tail (figures 02, 06),")
    print("   so a difference across it is a difference through the solid.")
    print("2. The revised TE psi_m reads (l+1, J), (l+2, J), (l+3, J): weights -3, 4, -1 over 2 dm.")
    print("3. The revised TE psi_mm reads (l+1, J) .. (l+4, J): weights 2, -5, 4, -1 over dm^2.")
    print("4. The revised TE psi_meta reads q at l+1, l+2, l+3, each q[k] from (k, J-1) and (k, J+1) = (k, 1) + 1.")
    print("5. All of those lie at or behind the first exit station, downstream of the tail, on the periodic line, just")
    print("   below it, or one pitch up through the continuation: none of their links meets the blade (figure 05).")
    for name, node, c in found:
        print(f"6. The n_passage 401 interior crossing: row (l, J-1) at n_eta 121, its {name} neighbour {node}: "
              f"{c.inside_length:.2e} c of the {c.length:.2e} c link inside the +pitch blade ({100 * c.fraction:.1f} %).")
    print("7. That link is the NE diagonal of Eq. 5-39, so it enters the equation through -2 B psi_meta. The row sits one")
    print("   eta step below the upper wall at the last blade station; its NE neighbour is the first exit station's")
    print("   periodic-line node, behind the tail, and the diagonal skims the tail on the way.")
    print(f"8. Over the sweep, {len(rows_affected)} (grid, row) pairs of the reviewed interior stencil have a straight")
    print("   chord intersecting the blade, at both transitions, beside both walls (j = 1 and j = J-1), more as n_eta grows")
    print(f"   (figure 08); the periodic rows have {periodic_bad} after the TE revision.")
    print("9. A straight-chord intersection is a geometric fact about the segment, not by itself a proof that the stencil is")
    print("   invalid: the differences act along the computational coordinate curves. That question is audited by")
    print("   scripts/audit_transition_mapping.py.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--save", type=Path, default=None, help="folder to write the figures to")
    parser.add_argument("--no-show", action="store_true", help="do not open windows")
    args = parser.parse_args()

    cascade = make_cascade(SPEC)
    blade = lg.Blade(cascade)
    keys = [(61, 41), (101, 41), (201, 41), (401, 41), (401, 121)]
    cases = {key: lg.build_case(cascade, *key) for key in keys}
    figures = {"01_te_geometry": figure_01(cascade, blade, cases), "02_old_vs_new_te_stencil": figure_02(blade, cases)}
    fig03, found = figure_03(blade, cases)
    figures["03_n401_interior_crossing"] = fig03
    figures["04_n401_index_stencil"] = figure_04(cases[(401, 121)][0], (cases[(401, 121)][0].last_blade_i, 119), found)
    figures["05_revised_te_footprint"] = figure_05(blade, cases)
    figures["06_resolution_comparison"] = figure_06(blade, cases)
    if found:
        figures["07_interior_proposal"] = figure_07(blade, cases)
    levels = [(n, e) for n in (31, 51, 61, 101, 201, 401) for e in (21, 31, 41, 61)] + [(201, 121), (401, 121)]
    results = lg.sweep(cascade, blade, levels)
    figures["08_straight_chord_sweep"] = figure_08(results)
    print_table()
    print_sweep(results)
    print_explanation(found, results)
    if args.save is not None:
        args.save.mkdir(parents=True, exist_ok=True)
        for name, fig in figures.items():
            fig.savefig(args.save / f"{name}.png", dpi=140)
            print(f"wrote {args.save / (name + '.png')}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
