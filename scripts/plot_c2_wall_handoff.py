"""
plot_c2_wall_handoff.py
=======================
Development diagnostic of the C2 geometry -> grid -> metrics hand-off (grid.py section 7,
metrics.py "C2 blade walls"). The same C2 cascade is gridded twice:

    new: grid.py on the authoritative curves (wall nodes X(u_i), slopes from the spline)
    old: grid.py on the same cascade with its curves removed, so it reads the sampled
         arrays through the C1 path (parametric PCHIP walls, finite-difference slopes):
         exactly the C2 grid and metrics before the hand-off.

Figures
    A  wall position: the PCHIP's normal distance from the curve it samples, along each
       surface (dense), with the old-to-new wall-node moves; LE and TE close-ups
    B  grid-line angle on the walls: beta exact against beta FD, and |beta_exact - beta_FD|
       for three resolutions
    C  exact d tan(beta)/dm along each passage-facing contour (dense and at the nodes),
       beside the curvature and |cos beta| = |m_u| / |X_u| of the same contour: the curve is
       C2 throughout, y(m) is not regular where m_u -> 0 (the nose tip, a Kovach tail end)
    D  first and last blade stations: the station lines, the authoritative wall points with
       their tangents, the passage-facing contour (colour) and the raw surface name (marker)
The audit tables are printed. Nothing here changes grid.py or metrics.py.

Usage
    python scripts/plot_c2_wall_handoff.py
    python scripts/plot_c2_wall_handoff.py --save outputs/c2_wall_handoff --no-show
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from potential_flow_solver import grid, metrics  # noqa: E402
from potential_flow_solver.config import GridSettings  # noqa: E402
from potential_flow_solver.geometry import CascadeSpec, make_cascade  # noqa: E402

# --- Development settings: edit these ---------------------------------------------------
EXAMPLE = CascadeSpec(blade_type="NACA65", solidity=1.0, stagger_deg=33.0, cl0=1.2, t_max_over_c=0.10,
                      surface_continuity="C2")
INLET_ANGLE_DEG = 45.0
N_ETA = 31
SHOWN = 101                         # figures A (node moves), C (nodes) and D (upper row)
RESOLUTIONS = (51, 101, 401)        # figure B and the edge tables
FINE = 401                          # figure D, lower row
EDGE_NODES = 5
DENSE = 40001                       # samples along each curve
TANGENT = 0.0015                    # chords: half-length of the tangent segments in figure D
# -----------------------------------------------------------------------------------------

NEW, OLD, THIRD = "#2a78d6", "#eb6834", "#1baf7a"     # validated categorical slots 1-3 (light)
INK, MUTED, GRID_INK = "#0b0b0b", "#52514e", "#e4e3df"
WALLS = ("lower", "upper")


@lru_cache(maxsize=None)
def cascade():
    return make_cascade(EXAMPLE)


@lru_cache(maxsize=None)
def grids(n_passage: int) -> tuple[grid.Grid, grid.Grid]:
    c = cascade()
    settings = GridSettings(n_passage=n_passage, n_eta=N_ETA)
    new = grid.make_grid(c, settings, math.radians(INLET_ANGLE_DEG))
    old = grid.make_grid(replace(c, active_curves=None, normalized_curves=None), settings, math.radians(INLET_ANGLE_DEG))
    return new, old


@lru_cache(maxsize=None)
def metric_pair(n_passage: int) -> tuple[metrics.GridMetrics, metrics.GridMetrics]:
    new, old = grids(n_passage)
    return metrics.build_metrics(new), metrics.build_metrics(old)


def curves() -> dict:
    a = cascade().active_curves
    return {"suction": a.suction, "pressure": a.pressure}


def wall_curve(g: grid.Grid, wall: str, k: int):
    return curves()[getattr(g.wall_identity, wall).surface[k]]


# --- The contours, read here from dense samples -----------------------------------------------

@lru_cache(maxsize=None)
def contour_samples() -> dict:
    """Dense samples of the closed outline (suction TE -> LE, pressure LE -> TE) with the
    contour coordinate tau of grid.py section 7, and the passage-facing contours: the upper
    one from the nose tip backward in tau to the tail end (the lower wall) and the lower one
    forward (the upper wall)."""
    cv = curves()
    u = np.linspace(0.0, 1.0, DENSE)
    length = {name: curve.reference_length for name, curve in cv.items()}
    parts = []
    for name, uu, along in (("suction", u[::-1], -1.0), ("pressure", u[1:], 1.0)):   # along: +1 where tau grows with u
        curve = cv[name]
        tau = (1.0 - uu) * length["suction"] if name == "suction" else length["suction"] + uu * length["pressure"]
        parts.append(dict(tau=tau, X=curve.evaluate(uu), d1=curve.derivative(uu, 1), d2=curve.derivative(uu, 2),
                          name=np.full(len(uu), name), along=np.full(len(uu), along)))
    loop = {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}
    tip, tail = int(np.argmin(loop["X"][:, 0])), int(np.argmax(loop["X"][:, 0]))
    n = len(loop["tau"])
    lower_rows = np.r_[tip:tail + 1] if tip <= tail else np.r_[tip:n, 0:tail + 1]           # forward from the tip
    upper_rows = np.r_[tip:-1:-1, n - 1:tail - 1:-1] if tip <= tail else np.r_[tip:tail - 1:-1]   # backward from the tip
    out = {}
    for wall, rows, direction in (("upper", lower_rows, 1.0), ("lower", upper_rows, -1.0)):
        d1, d2 = loop["d1"][rows], loop["d2"][rows]
        step = np.hypot(*np.diff(loop["X"][rows], axis=0).T)
        # the curvature turned to the walking direction (u may run against it); d tan/dm and |cos| do not depend on it
        sense = direction * loop["along"][rows]
        out[wall] = dict(s=np.concatenate(([0.0], np.cumsum(step))), X=loop["X"][rows], name=loop["name"][rows],
                         kappa=sense * (d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]) / np.hypot(d1[:, 0], d1[:, 1]) ** 3,
                         cos=np.abs(d1[:, 0]) / np.hypot(d1[:, 0], d1[:, 1]),
                         dtan=(d2[:, 1] * d1[:, 0] - d1[:, 1] * d2[:, 0]) / d1[:, 0] ** 3)
    out["tip"], out["tail"] = loop["X"][tip], loop["X"][tail]
    return out


def node_contour_s(g: grid.Grid, wall: str) -> np.ndarray:
    """Arc length along the passage-facing contour (contour_samples) of each wall node."""
    contour = contour_samples()[wall]
    walls = slice(g.first_blade_i, g.last_blade_i + 1)
    nodes = g.nodes[walls, 0 if wall == "lower" else -1].copy()
    if wall == "upper":
        nodes[:, 1] -= g.pitch
    return np.array([contour["s"][int(np.argmin(np.hypot(*(contour["X"] - p).T)))] for p in nodes])


# --- Tables --------------------------------------------------------------------------------

def report() -> None:
    a = cascade().active
    print(f"{cascade().metadata['designation']} C2, {EXAMPLE.profile_variant}, stagger {EXAMPLE.stagger_deg} deg, "
          f"solidity {EXAMPLE.solidity}; n_eta {N_ETA}")
    print("\n=== Old (sampled arrays -> PCHIP) against new (authoritative curve) wall nodes ===")
    print(f"{'n':>4} {'first':>5} {'last':>5} {'same':>5} {'max|dy| lo':>10} {'up':>9} {'rms lo':>8} {'up':>8} "
          f"{'min S old':>11} {'min S new':>11}")
    for n in (31, 51, 81, 101, 141, 201, 401):
        new, old = grids(n)
        w = slice(new.first_blade_i, new.last_blade_i + 1)
        same = (new.first_blade_i, new.last_blade_i) == (old.first_blade_i, old.last_blade_i)
        d = new.nodes[w][:, [0, -1], 1] - old.nodes[w][:, [0, -1], 1]
        s_new, s_old = (np.min(x.nodes[w, -1, 1] - x.nodes[w, 0, 1]) for x in (new, old))
        print(f"{n:4d} {new.first_blade_i:5d} {new.last_blade_i:5d} {str(same):>5} {np.abs(d[:, 0]).max() / a.chord:10.2e} "
              f"{np.abs(d[:, 1]).max() / a.chord:9.2e} {np.sqrt(np.mean(d[:, 0] ** 2)) / a.chord:8.1e} "
              f"{np.sqrt(np.mean(d[:, 1] ** 2)) / a.chord:8.1e} {s_old / a.chord:11.8f} {s_new / a.chord:11.8f}")
    for n in RESOLUTIONS:
        new, _ = grids(n)
        gm, fd = metric_pair(n)
        f, l = new.first_blade_i, new.last_blade_i
        d1 = new.wall_curve_derivative(1)
        print(f"\n=== n_passage {n}: first and last {EDGE_NODES} C2 blade stations (dm/c {new.dm / a.chord:.5f}) ===")
        print(f"{'i':>4} {'wall':5} {'m/c':>9} {'y/c':>10} {'curve':8} {'u':>12} {'tan_beta':>10} {'beta deg':>9} "
              f"{'FD deg':>8} {'dtan/dm c':>11} {'|dm/du|':>8} {'off curve':>9}")
        for ks in (range(EDGE_NODES), range(l - f - EDGE_NODES + 1, l - f + 1)):
            for k in ks:
                i = f + k
                for wall, row, slope, dslope, fd_slope, shift in (
                        ("lower", 0, gm.tan_beta0, gm.dtan_beta0_dm, fd.tan_beta0, 0.0),
                        ("upper", -1, gm.tan_beta1, gm.dtan_beta1_dm, fd.tan_beta1, new.pitch)):
                    side = getattr(new.wall_identity, wall)
                    point = new.nodes[i, row] - (0.0, shift)
                    off = float(np.hypot(*(wall_curve(new, wall, k).evaluate(side.u[k]) - point)))
                    print(f"{i:4d} {wall:5} {new.m[i] / a.chord:+9.5f} {new.nodes[i, row, 1] / a.chord:+10.6f} "
                          f"{side.surface[k]:8} {side.u[k]:12.9f} {slope[i]:+10.4f} {math.degrees(math.atan(slope[i])):+9.3f} "
                          f"{math.degrees(math.atan(fd_slope[i])):+8.3f} {dslope[i] * a.chord:+11.4e} "
                          f"{abs(d1[0 if wall == 'lower' else 1][k, 0]) / a.chord:8.3e} {off / a.chord:9.1e}")
            print("   ...")


# --- Figures -------------------------------------------------------------------------------

def style(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=10, loc="left", color=INK)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(True, color=GRID_INK, linewidth=0.6)
    ax.tick_params(colors=MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(GRID_INK)


def figure_a() -> plt.Figure:
    """PCHIP-through-samples against the curve: normal distance along each surface, the
    old -> new node moves at SHOWN cells, and LE/TE close-ups of the geometry."""
    c = cascade()
    a, chord = c.active, c.active.chord
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.5))
    fig.suptitle(f"A. Wall position: old grid PCHIP through the samples against the authoritative C2 curve "
                 f"(n_passage {SHOWN})", x=0.02, ha="left", fontsize=11)
    loop = np.vstack((a.suction_surface[::-1], a.pressure_surface[1:]))
    t = np.concatenate(([0.0], np.cumsum(np.hypot(*np.diff(loop, axis=0).T))))
    x_of_t, y_of_t = PchipInterpolator(t, loop[:, 0]), PchipInterpolator(t, loop[:, 1])
    n_s = len(a.suction_surface)
    for ax, name, rows in ((axes[0, 0], "suction", slice(0, n_s)), (axes[0, 1], "pressure", slice(n_s - 1, None))):
        tt = np.linspace(t[rows][0], t[rows][-1], 6000)
        points = np.column_stack((x_of_t(tt), y_of_t(tt)))
        curve = curves()[name]
        u_guess = curve.project(points[::50])[0]
        u_all = np.interp(np.arange(len(tt)), np.arange(len(tt))[::50], u_guess)
        _, distance = curve.project(points, u_all)
        ax.plot(points[:, 0] / chord, distance / chord, color=OLD, linewidth=1.0,
                label="PCHIP through the samples (old wall), normal distance from the curve")
        ax.axhline(0.0, color=NEW, linewidth=1.4, label="authoritative curve (new wall)")
        new, old = grids(SHOWN)
        w = slice(new.first_blade_i, new.last_blade_i + 1)
        for wall, row, marker in (("lower", 0, "o"), ("upper", -1, "s")):
            side = getattr(new.wall_identity, wall)
            on = side.surface == name
            move = (old.nodes[w, row, 1] - new.nodes[w, row, 1])[on]
            ax.plot(new.m[w][on] / chord, move / chord, marker, color=MUTED, markersize=3.5, markerfacecolor="none",
                    linestyle="none", label=f"old - new {wall}-wall node, vertical (y)")
        style(ax, f"{name} surface: max |normal| {np.abs(distance).max() / chord:.1e} c", "m / c", "distance / c")
        ax.legend(fontsize=7, loc="best")
    tt = np.linspace(0.0, t[-1], 60001)
    pchip = np.column_stack((x_of_t(tt), y_of_t(tt)))
    for ax, end in ((axes[1, 0], "LE"), (axes[1, 1], "TE")):
        new, old = grids(SHOWN)
        f, l = new.first_blade_i, new.last_blade_i
        stations = list(range(f - 1, f + EDGE_NODES)) if end == "LE" else list(range(l - EDGE_NODES + 1, l + 2))
        blade = [i for i in stations if f <= i <= l]
        ys = np.concatenate([new.nodes[blade, 0, 1], new.nodes[blade, -1, 1] - a.pitch])
        pad_m, pad_y = 0.3 * new.dm, 0.1 * (ys.max() - ys.min())
        box = (new.m[stations[0]] - pad_m, new.m[stations[-1]] + pad_m, ys.min() - pad_y, ys.max() + pad_y)
        for name, curve in curves().items():
            X = curve.evaluate(np.linspace(0.0, 1.0, DENSE))
            ax.plot(X[:, 0] / chord, X[:, 1] / chord, color=NEW, linewidth=1.6,
                    label="authoritative curve (new wall)" if name == "suction" else None)
        ax.plot(pchip[:, 0] / chord, pchip[:, 1] / chord, "--", color=OLD, linewidth=1.0, label="PCHIP of the samples (old wall)")
        inside = (loop[:, 0] > box[0]) & (loop[:, 0] < box[1])
        ax.plot(loop[inside, 0] / chord, loop[inside, 1] / chord, ".", color=OLD, markersize=4, label="samples (the arrays)")
        for i in stations:
            ax.axvline(new.m[i] / chord, color=MUTED, linewidth=0.7, linestyle=":" if i in (f - 1, l + 1) else "-", zorder=0)
        for i in blade:
            for row, shift in ((0, 0.0), (-1, a.pitch)):
                ax.plot(new.m[i] / chord, (new.nodes[i, row, 1] - shift) / chord, "o", color=NEW, markersize=6)
                ax.plot(old.m[i] / chord, (old.nodes[i, row, 1] - shift) / chord, "x", color=OLD, markersize=7)
        moved = np.abs(new.nodes[blade][:, [0, -1], 1] - old.nodes[blade][:, [0, -1], 1]).max()
        ax.plot([], [], "o", color=NEW, label="new wall nodes (upper wall moved down one pitch)")
        ax.plot([], [], "x", color=OLD, label="old wall nodes")
        ax.plot([], [], ":", color=MUTED, label="periodic neighbour station")
        ax.set_xlim(box[0] / chord, box[1] / chord)
        ax.set_ylim(box[2] / chord, box[3] / chord)
        style(ax, f"{end} close-up: the shown nodes moved by at most {moved / chord:.1e} c", "m / c", "y / c")
        ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    return fig


def figure_b() -> plt.Figure:
    chord = cascade().active.chord
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.5))
    fig.suptitle("B. Wall grid-line angle: exact C2 spline slope against the old finite difference of the PCHIP nodes",
                 x=0.02, ha="left", fontsize=11)
    for row, wall in enumerate(WALLS):
        gm, fd = metric_pair(SHOWN)
        g = gm.grid
        w = slice(g.first_blade_i, g.last_blade_i + 1)
        exact = gm.tan_beta0 if wall == "lower" else gm.tan_beta1
        old = fd.tan_beta0 if wall == "lower" else fd.tan_beta1
        ax = axes[row, 0]
        ax.plot(g.m[w] / chord, np.degrees(np.arctan(exact[w])), "-o", color=NEW, markersize=2.5, linewidth=1.0,
                label="beta exact (curve, new)")
        ax.plot(g.m[w] / chord, np.degrees(np.arctan(old[w])), "x", color=OLD, markersize=4, label="beta FD (old)")
        style(ax, f"{wall} wall, n_passage {SHOWN}", "m / c", "beta [deg]")
        ax.legend(fontsize=7)
        ax = axes[row, 1]
        for n, color, marker in zip(RESOLUTIONS, (NEW, OLD, THIRD), ("o", "s", "^")):
            gm, fd = metric_pair(n)
            g = gm.grid
            w = slice(g.first_blade_i, g.last_blade_i + 1)
            exact = gm.tan_beta0 if wall == "lower" else gm.tan_beta1
            old = fd.tan_beta0 if wall == "lower" else fd.tan_beta1
            diff = np.abs(np.degrees(np.arctan(exact[w]) - np.arctan(old[w])))
            ax.semilogy(g.m[w] / chord, np.maximum(diff, 1e-8), marker, color=color, markersize=2.5, linewidth=0.8,
                        linestyle="-", label=f"n_passage {n}")
        style(ax, f"{wall} wall: |beta_exact - beta_FD| (large only at the first and last nodes)", "m / c",
              "|delta beta| [deg]")
        ax.legend(fontsize=7)
    fig.tight_layout()
    return fig


def figure_c() -> plt.Figure:
    chord = cascade().active.chord
    contours = contour_samples()
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex="col")
    fig.suptitle("C. Along each passage-facing contour, from the nose tip to the tail end: exact d tan(beta)/dm, "
                 "the curvature and |cos beta| of the same C2 curve", x=0.02, ha="left", fontsize=11)
    gm, _ = metric_pair(SHOWN)
    g = gm.grid
    w = slice(g.first_blade_i, g.last_blade_i + 1)
    for col, wall in enumerate(WALLS):
        cont = contours[wall]
        s = cont["s"] / chord
        nodes_s = node_contour_s(g, wall) / chord
        dtan = (gm.dtan_beta0_dm if wall == "lower" else gm.dtan_beta1_dm)[w] * chord
        ax = axes[0, col]
        ax.plot(s, cont["dtan"] * chord, color=NEW, linewidth=1.0, label="d tan(beta)/dm * c of the curve (dense)")
        ax.plot(nodes_s, dtan, "o", color=OLD, markersize=2.5, label=f"metrics at the wall nodes, n_passage {SHOWN}")
        ax.set_yscale("symlog", linthresh=10.0)
        style(ax, f"{wall} wall: d tan(beta)/dm = (y_uu m_u - y_u m_uu) / m_u^3", "", "c d tan(beta)/dm")
        ax.legend(fontsize=7)
        ax = axes[1, col]
        ax.plot(s, cont["kappa"] * chord, color=NEW, linewidth=1.0)
        style(ax, "curvature * c, signed along the walk: continuous at the nose tip and tail end too", "", "kappa c")
        ax = axes[2, col]
        ax.semilogy(s, cont["cos"], color=NEW, linewidth=1.0, label="|cos beta| = |m_u| / |X_u|")
        ax.semilogy(nodes_s, np.abs(gm.cos_beta[w, 0 if wall == "lower" else -1]), "o", color=OLD, markersize=2.5,
                    label="wall nodes")
        style(ax, "|cos beta| -> 0 only at the contour's two ends (vertical tangent)",
              "arc length along the contour from the nose tip / c", "|cos beta|")
        ax.legend(fontsize=7)
    fig.tight_layout()
    return fig


def figure_d() -> plt.Figure:
    a = cascade().active
    chord = a.chord
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("D. First and last blade stations: authoritative wall points and tangents (upper-wall points moved down "
                 "one pitch onto the reference blade)", x=0.02, ha="left", fontsize=11)
    contour_color = {"lower": NEW, "upper": OLD}
    markers = {"suction": "o", "pressure": "s"}
    for row, n in enumerate((SHOWN, FINE)):
        new, _ = grids(n)
        gm, _ = metric_pair(n)
        f, l = new.first_blade_i, new.last_blade_i
        d1 = new.wall_curve_derivative(1)
        for col, end in enumerate(("LE", "TE")):
            ax = axes[row, col]
            stations = list(range(f, f + EDGE_NODES)) if end == "LE" else list(range(l - EDGE_NODES + 1, l + 1))
            neighbour = f - 1 if end == "LE" else l + 1
            for name, curve in curves().items():
                X = curve.evaluate(np.linspace(0.0, 1.0, DENSE))
                ax.plot(X[:, 0] / chord, X[:, 1] / chord, color=MUTED, linewidth=1.0, zorder=1)
            for i in stations:
                ax.axvline(new.m[i] / chord, color=GRID_INK, linewidth=0.8, zorder=0)
            ax.axvline(new.m[neighbour] / chord, color=MUTED, linewidth=0.8, linestyle=":", zorder=0)
            points = []
            for i in stations:
                k = i - f
                for wall, which, shift in (("lower", 0, 0.0), ("upper", 1, a.pitch)):
                    side = getattr(new.wall_identity, wall)
                    p = np.array([new.m[i], new.nodes[i, 0 if wall == "lower" else -1, 1] - shift])
                    tangent = d1[which][k] / np.hypot(*d1[which][k])
                    seg = np.array([p - TANGENT * chord * tangent, p + TANGENT * chord * tangent]) / chord
                    ax.plot(seg[:, 0], seg[:, 1], color=contour_color[wall], linewidth=1.2, zorder=2)
                    ax.plot(p[0] / chord, p[1] / chord, markers[side.surface[k]], color=contour_color[wall], markersize=6,
                            zorder=3, markeredgecolor="white", markeredgewidth=0.6)
                    points.append(p)
            points = np.array(points)
            pad = 0.004 * chord
            ax.set_xlim((points[:, 0].min() - pad) / chord, (points[:, 0].max() + pad) / chord)
            ax.set_ylim((points[:, 1].min() - pad) / chord, (points[:, 1].max() + pad) / chord)
            ax.set_aspect("equal")
            beta = [math.degrees(math.atan(x)) for x in (gm.tan_beta0[stations[0 if end == 'LE' else -1]],
                                                          gm.tan_beta1[stations[0 if end == 'LE' else -1]])]
            style(ax, f"{end}, n_passage {n}: {'first' if end == 'LE' else 'last'} station beta lower/upper "
                      f"{beta[0]:+.1f} / {beta[1]:+.1f} deg", "m / c", "y / c")
    handles = [plt.Line2D([], [], color=contour_color[wall], marker="o", linestyle="-",
                          label=f"{wall} wall: the {'upper' if wall == 'lower' else 'lower'} contour (passage-facing), "
                                f"with the wall tangent") for wall in WALLS]
    handles += [plt.Line2D([], [], color=MUTED, marker=marker, linestyle="none", label=f"raw surface name: {name}")
                for name, marker in markers.items()]
    handles += [plt.Line2D([], [], color=MUTED, linestyle=":", label="periodic neighbour station"),
                plt.Line2D([], [], color=GRID_INK, linestyle="-", label="blade stations")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--save", type=Path, default=None, help="folder to write the figures to")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    report()
    figures = {"A_wall_position": figure_a(), "B_beta": figure_b(), "C_dtan_dm": figure_c(), "D_edge_nodes": figure_d()}
    if args.save is not None:
        args.save.mkdir(parents=True, exist_ok=True)
        for name, fig in figures.items():
            fig.savefig(args.save / f"c2_wall_handoff_{name}.png", dpi=130)
        print(f"\nsaved {len(figures)} figures to {args.save}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
