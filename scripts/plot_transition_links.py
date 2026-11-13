"""
The TE transition problem of the periodic PDE rows, drawn on the real grid: the default C2
cascade (65-(12A10)10 at 33 deg, solidity 1, inlet 45 deg).

    python scripts/plot_transition_links.py                       # show
    python scripts/plot_transition_links.py --save outputs/transition_links --no-show

The PDE row of the first exit station, (l+1, J) with l = last_blade_i and J = n_eta - 1, keeps
the central psi_m and psi_m,m (discretization.py, "The periodic PDE line"), so it reads the
upper wall node (l, J) of the last blade station. That node lies on the +pitch neighbor's
lower contour near its tail, and the straight link between the two nodes runs through the
tail: drawn red where it is inside the metal. The mixed term of the same row (forward,
stations l+1 .. l+3) stays downstream and is drawn too, with the nodes it reads through the
periodic continuation psi[k, J+1] = psi[k, 1] + 1 placed where they are physically, one pitch
above (k, 1). The LE counterpart, the row of the last inlet station and its link to the
first blade station's upper wall node, is drawn for contrast: free of metal.

Panels: an overview of the passage with the two close-up windows; the TE at n_passage 61,
101, 201 and 401; the LE at n_passage 61, 101 and 401.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402

from potential_flow_solver.boundary_conditions import build_boundary_conditions  # noqa: E402
from potential_flow_solver.coefficients import build_coefficients  # noqa: E402
from potential_flow_solver.config import GridSettings  # noqa: E402
from potential_flow_solver.discretization import build_periodic_pde_rows  # noqa: E402
from potential_flow_solver.geometry import CascadeSpec, make_cascade  # noqa: E402
from potential_flow_solver.grid import make_grid  # noqa: E402
from potential_flow_solver.metrics import build_metrics  # noqa: E402

SPEC = CascadeSpec(blade_type="NACA65", solidity=1.0, stagger_deg=33.0, cl0=1.2, t_max_over_c=0.10,
                   surface_continuity="C2")
INLET = math.radians(45.0)
LEVELS = {61: 21, 101: 31, 201: 61, 401: 121}

INK, MUTED, GRIDLINE = "#1f2328", "#6e7781", "#c9ced6"
BLADE_FILL, BLADE_EDGE = "#e4e7eb", "#57606a"
FLUID, WALL, METAL = "#2a78d6", "#eb6834", "#cf222e"


def outline(cascade) -> np.ndarray:
    """The closed C2 outline of the reference blade, densely sampled (TE -> LE on the suction
    curve, LE -> TE on the pressure curve)."""
    u = np.linspace(0.0, 1.0, 40001)
    curves = cascade.active_curves
    return np.vstack([curves.suction.derivative(u[::-1], 0), curves.pressure.derivative(u[1:], 0)])


def inside_runs(a: np.ndarray, b: np.ndarray, path: MplPath, samples: int = 4001):
    """The parts of the segment a -> b inside the blade: list of (start, end) points, and the
    metal length."""
    t = np.linspace(0.0, 1.0, samples)
    points = a[None, :] + t[:, None] * (b - a)[None, :]
    inside = path.contains_points(points)
    runs, k = [], 0
    while k < len(t):
        if inside[k]:
            start = k
            while k < len(t) and inside[k]:
                k += 1
            runs.append((points[start], points[k - 1]))
        k += 1
    return runs, float(inside.mean() * np.linalg.norm(b - a))


def case(cascade, n_passage: int):
    g = make_grid(cascade, GridSettings(n_passage=n_passage, n_eta=LEVELS[n_passage]), INLET)
    metrics = build_metrics(g)
    co = build_coefficients(metrics, np.ones((len(g.m), len(g.eta))), 1.0)
    rows = {row.target[0]: row for row in build_periodic_pde_rows(co, metrics, build_boundary_conditions(g))}
    return g, rows


def physical(g, node) -> np.ndarray:
    """Where a row node is: a grid node, or (k, 1) read through the continuation above the
    upper line, which is one pitch above (k, 1)."""
    return np.asarray(g.nodes[node], dtype=float)


def draw_close_up(ax, g, rows, path: MplPath, blade: np.ndarray, side: str, n_passage: int) -> None:
    f, l, J = g.first_blade_i, g.last_blade_i, len(g.eta) - 1
    if side == "TE":
        row, wall_i, centre = rows[l + 1], l, np.array([g.m_te, 0.0])
    else:
        row, wall_i, centre = rows[f - 1], f, np.array([g.m_le, 0.0])
    target = physical(g, row.target)
    wall = physical(g, (wall_i, J))
    half = max(3.2 * g.dm, 0.018)
    centre = 0.5 * (target + wall)
    ax.set_xlim(centre[0] - half, centre[0] + half)
    ax.set_ylim(centre[1] - 0.75 * half, centre[1] + 0.75 * half)
    ax.fill(blade[:, 0], blade[:, 1] + g.pitch, color=BLADE_FILL, zorder=1)
    ax.plot(blade[:, 0], blade[:, 1] + g.pitch, color=BLADE_EDGE, lw=1.0, zorder=2)
    stations = range(max(0, wall_i - 4), min(len(g.m), wall_i + 6))
    for i in stations:
        ax.axvline(g.m[i], color=GRIDLINE, lw=0.6, zorder=0)
    for j in range(max(0, J - 4), J + 1):
        ax.plot(g.nodes[list(stations), j, 0], g.nodes[list(stations), j, 1], color=GRIDLINE, lw=0.6, zorder=0)
    for i in stations:
        on_wall = g.first_blade_i <= i <= g.last_blade_i
        ax.plot(*g.nodes[i, J], "o", ms=4.5, color=WALL if on_wall else FLUID, zorder=4)
        ax.plot(*g.nodes[i, J - 1], "o", ms=3, color=MUTED, zorder=3)
    # the row's own links: centre -> each real node it reads (continued nodes one pitch up)
    for t in row.terms:
        if t.node == row.target:
            continue
        k, j = t.node
        point = physical(g, t.node) + (np.array([0.0, g.pitch]) if j == 1 else 0.0)
        ax.plot([target[0], point[0]], [target[1], point[1]], color=FLUID, lw=0.9, alpha=0.55, zorder=3)
        ax.plot(*point, "s" if j == 1 else "o", ms=4, mfc="white" if j == 1 else FLUID, mec=FLUID, zorder=5)
    runs, metal = inside_runs(target, wall, path)
    ax.plot([target[0], wall[0]], [target[1], wall[1]], color=FLUID, lw=2.4, zorder=6)
    for a, b in runs:
        ax.plot([a[0], b[0]], [a[1], b[1]], color=METAL, lw=3.4, solid_capstyle="butt", zorder=7)
    ax.plot(*target, "o", ms=8, mfc="white", mec=FLUID, mew=2.0, zorder=8)
    ax.plot(*wall, "D", ms=7, color=WALL, zorder=8)
    length = float(np.linalg.norm(wall - target))
    verdict = (f"{metal:.1e} c of the {length:.1e} c link inside the metal ({100 * metal / length:.0f} %)" if metal > 0
               else f"link {length:.1e} c, free of metal")
    name = "first exit station" if side == "TE" else "last inlet station"
    ax.set_title(f"{side}, n_passage {n_passage}: row {row.target} ({name})", fontsize=9.5, color=INK, loc="left")
    ax.text(0.02, 0.03, verdict, transform=ax.transAxes, fontsize=8.5, color=METAL if metal > 0 else INK,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GRIDLINE))
    ax.text(target[0], target[1], f"  ({row.target[0]}, J)", fontsize=8, color=INK, va="bottom", zorder=9)
    ax.text(wall[0], wall[1], f"({wall_i}, J)  ", fontsize=8, color=INK, ha="right", va="top", zorder=9)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=7.5, colors=MUTED)
    for spine in ax.spines.values():
        spine.set_color(GRIDLINE)


def draw_overview(ax, g, blade: np.ndarray, windows) -> None:
    for shift in (0.0, g.pitch):
        ax.fill(blade[:, 0], blade[:, 1] + shift, color=BLADE_FILL, zorder=1)
        ax.plot(blade[:, 0], blade[:, 1] + shift, color=BLADE_EDGE, lw=0.9, zorder=2)
    J = len(g.eta) - 1
    ax.plot(g.nodes[:, 0, 0], g.nodes[:, 0, 1], color=FLUID, lw=1.0, label="lower boundary line (j = 0)")
    ax.plot(g.nodes[:, J, 0], g.nodes[:, J, 1], color=WALL, lw=1.0, label="upper boundary line (j = J)")
    for name, (x0, y0, w, h) in windows.items():
        ax.add_patch(plt.Rectangle((x0, y0), w, h, fill=False, ec=METAL if name == "TE" else INK, lw=1.4, zorder=5))
        ax.text(x0 + w, y0 + h, f" {name}", color=METAL if name == "TE" else INK, fontsize=9, va="bottom")
    ax.set_xlim(-0.25, 1.15)
    ax.set_ylim(-0.15, 1.85)
    ax.set_aspect("equal")
    ax.set_title("Passage, n_passage 101: the two transition windows", fontsize=9.5, color=INK, loc="left")
    ax.legend(fontsize=7.5, loc="upper left", frameon=False)
    ax.tick_params(labelsize=7.5, colors=MUTED)
    for spine in ax.spines.values():
        spine.set_color(GRIDLINE)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--save", type=Path, default=None, help="folder to write the figure to")
    parser.add_argument("--no-show", action="store_true", help="do not open a window")
    args = parser.parse_args()

    cascade = make_cascade(SPEC)
    blade = outline(cascade)
    path = MplPath(blade + np.array([0.0, cascade.active.pitch]))
    cases = {n: case(cascade, n) for n in LEVELS}
    fig, axes = plt.subplots(2, 4, figsize=(20.5, 9.5))
    g101 = cases[101][0]
    windows = {"TE": (g101.m_te - 0.05, g101.pitch + 0.25, 0.1, 0.1), "LE": (g101.m_le - 0.05, g101.pitch - 0.05, 0.1, 0.1)}
    te_y = cascade.active.trailing_edge_point[1] + g101.pitch
    windows["TE"] = (g101.m_te - 0.05, te_y - 0.05, 0.1, 0.1)
    draw_overview(axes[0, 0], g101, blade, windows)
    for ax, n_passage in zip((axes[0, 1], axes[0, 2], axes[0, 3], axes[1, 0]), (61, 101, 201, 401)):
        draw_close_up(ax, *cases[n_passage], path, blade, "TE", n_passage)
    for ax, n_passage in zip(axes[1, 1:], (61, 101, 401)):
        draw_close_up(ax, *cases[n_passage], path, blade, "LE", n_passage)
    handles = [plt.Line2D([], [], color=FLUID, lw=2.4, label="link the row's central psi_m, psi_mm reads"),
               plt.Line2D([], [], color=METAL, lw=3.4, label="part of that link inside the blade"),
               plt.Line2D([], [], marker="o", ls="", mfc="white", mec=FLUID, mew=2, ms=8, label="the row's node (i, J)"),
               plt.Line2D([], [], marker="D", ls="", color=WALL, ms=7, label="upper wall node of the blade station"),
               plt.Line2D([], [], marker="o", ls="", color=FLUID, ms=4.5, label="periodic-line node (j = J)"),
               plt.Line2D([], [], marker="s", ls="", mfc="white", mec=FLUID, ms=4,
                          label="node read through the continuation (k, 1) + pitch"),
               plt.Line2D([], [], color=FLUID, lw=0.9, alpha=0.55, label="the row's other links")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8.5, frameon=False)
    fig.suptitle("Periodic PDE rows beside the blade: at the TE the central psi_m link runs through the +pitch blade's "
                 "tail; at the LE it does not", fontsize=11.5, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    if args.save is not None:
        args.save.mkdir(parents=True, exist_ok=True)
        out = args.save / "transition_links.png"
        fig.savefig(out, dpi=140)
        print(f"wrote {out}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
