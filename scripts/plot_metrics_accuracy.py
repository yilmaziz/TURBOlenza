"""
plot_metrics_accuracy.py
========================
Wall-slope accuracy audit of metrics.py: are the large grid-line angles near the rounded
LE and TE real geometry, or finite-difference error?

metrics.py takes tan(beta0) = dy0/dm and tan(beta1) = dy1/dm by second-order differences of
the wall nodes, inside each m region (three-point one-sided at a run's ends, central
inside). This script sets beside every blade-wall node the exact tangent of the smooth wall
the node lies on: the parametric PCHIP of grid.py section 4 (x(t), y(t) through the closed
outline suction[::-1] + pressure[1:], t the cumulative chord length), crossed with x = m on
the very polyline segment the frozen Grid picked (the top of the reference blade's cut for
the lower wall, its bottom for the upper wall, one pitch up: the passage-facing contours,
whatever geometry.py calls the surface there), and differentiated analytically,
tan_beta_ref = y'(t) / x'(t). No grid node is moved; that the reference point is the Grid's
wall node is checked (to round-off) for every node. Nothing here changes metrics.py.

Reading the audit (2026-11-11). It showed that the finite-difference implementation is
mathematically sound (second order with the textbook constants) and that its error is
local: at the rounded edges, set by where the stations fall against a vertical tangent,
and at the knots of the sampled walls. It still describes metrics.py for C1 grids, and the
EXAMPLE here is C1. For a C2 grid metrics.py no longer takes this approximation: the blade
walls are the authoritative curves and their slopes the spline's own derivatives
(grid.py section 7, metrics.py "C2 blade walls"); scripts/plot_c2_wall_handoff.py sets the
two side by side.

Figures: (1) beta_FD and beta_ref along both walls; (2) |beta error| and (3) |tan beta error|
for every resolution on one set of axes, with the construction knots of the blade marked;
(4) G = 1/cos^2 beta, FD against reference; (5) the first and last five blade nodes against
the dense smooth-wall angle; (6) local observed order of the error envelope. The tables of
the audit are printed. Development settings below; run without options.

Usage
    python scripts/plot_metrics_accuracy.py
    python scripts/plot_metrics_accuracy.py --save audit_dir --no-show
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from turbolenza.b2b import geometry, grid, metrics  # noqa: E402
from turbolenza.b2b.config import GridSettings  # noqa: E402
from turbolenza.b2b.geometry import CascadeSpec  # noqa: E402

# --- Development settings: edit these ---------------------------------------------------
EXAMPLE = CascadeSpec(blade_type="NACA65", solidity=1.0, stagger_deg=33.0, cl0=1.2, t_max_over_c=0.10,
                      surface_continuity="C1")
INLET_ANGLE_DEG = 45.0
N_ETA = 31
RESOLUTIONS = (31, 51, 81, 101, 141, 201, 301, 401)    # n_passage of the refinement study
SHOWN = 101                     # figure 1: angles along the walls
G_CASE = 201                    # figure 4: 1/cos^2 beta
CLOSE_UPS = (51, 101, 201, 401) # figure 5 and the LE/TE node tables
EDGE_NODES = 5                  # nodes shown at each end
INTERIOR_EXCLUDE = 5            # blade stations left out at each end for the interior norms
NODE_ORDER = 11                 # k = 0 .. NODE_ORDER - 1 from each end
N_ETA_CHECK = (11, 31, 61, 101)  # at SHOWN
CAMBER_STATIONS = (101, 201, 401, 801)   # geometry study, at GEOMETRY_CASE
GEOMETRY_CASE = 201
GEOMETRY_REFERENCE = 3201       # n_camber_stations of the high-resolution blade
AWAY = 0.05                     # chords from the nose tip / tail end: the supplementary interior
LOCAL_POINTS = 400              # common m locations of the local order p(m)
NEAR_ZERO = 1e-9                # |tan error| below this is masked in p(m)
MEDIAN_WINDOW = 0.02            # chords: running median of p(m)
# -----------------------------------------------------------------------------------------

WALLS = ("lower", "upper")
COLORS = plt.cm.viridis(np.linspace(0.0, 0.92, len(RESOLUTIONS)))
INK, MUTED, LANDMARK = "#0b0b0b", "#6b6a66", "#b8327a"


class SmoothWalls:
    """The smooth walls of one blade (grid.py section 4, written out here) and their exact
    slope where a station line crosses them on a given polyline segment."""

    def __init__(self, cascade: geometry.Cascade) -> None:
        a = cascade.active
        self.active = a
        self.n_suction = len(a.suction_surface)
        self.outline = np.vstack((a.suction_surface[::-1], a.pressure_surface[1:]))
        self.t = np.concatenate(([0.0], np.cumsum(np.hypot(*np.diff(self.outline, axis=0).T))))
        self.x, self.y = PchipInterpolator(self.t, self.outline[:, 0]), PchipInterpolator(self.t, self.outline[:, 1])

    def cut(self, m: float) -> tuple[int, int]:
        """Polyline segments of the top and the bottom crossing of x = m (half-open)."""
        x0, x1 = self.outline[:-1, 0], self.outline[1:, 0]
        k = np.flatnonzero((np.minimum(x0, x1) <= m) & (m < np.maximum(x0, x1)))
        h = self.outline[k, 1] + (m - x0[k]) / (x1[k] - x0[k]) * (self.outline[k + 1, 1] - self.outline[k, 1])
        k = k[np.argsort(h)]
        return int(k[-1]), int(k[0])

    def crossing(self, segment: int, m: float) -> tuple[float, float]:
        """(y, dy/dm) of the smooth wall where it crosses x = m on the segment."""
        lo, hi = self.t[segment], self.t[segment + 1]
        f = lambda u: float(self.x(u)) - m          # noqa: E731
        u = lo if f(lo) == 0.0 else hi if f(hi) == 0.0 else brentq(f, lo, hi, xtol=1e-15, rtol=1e-15, maxiter=200)
        return float(self.y(u)), float(self.y(u, 1)) / float(self.x(u, 1))

    def walls(self, m: float) -> dict[str, tuple[float, float]]:
        top, bottom = self.cut(m)
        return {"lower": self.crossing(top, m), "upper": self.crossing(bottom, m)}

    def dense_angle(self, which: str, m: np.ndarray) -> np.ndarray:
        """The smooth wall's grid-line angle in degrees at many m (for the close-ups)."""
        return np.degrees(np.arctan([self.walls(float(x))[which][1] for x in m]))


@lru_cache(maxsize=None)
def cascade_of(spec: CascadeSpec) -> geometry.Cascade:
    return geometry.make_cascade(spec)


@lru_cache(maxsize=None)
def smooth_of(spec: CascadeSpec) -> SmoothWalls:
    return SmoothWalls(cascade_of(spec))


@dataclass
class Audit:
    """FD and reference slopes at the blade-wall nodes of one grid."""
    n_passage: int
    dm: float
    m: np.ndarray               # blade stations
    tan_fd: dict                # wall -> (n_blade,)
    tan_ref: dict
    names: dict                 # wall -> geometry.py surface name per node
    y_check: float              # largest |reference y - wall node y|
    m_le: float
    m_te: float
    tip: float
    tail: float

    def beta(self, which: str, kind: str) -> np.ndarray:
        return np.degrees(np.arctan((self.tan_fd if kind == "fd" else self.tan_ref)[which]))

    def errors(self, which: str) -> dict[str, np.ndarray]:
        fd, ref = self.tan_fd[which], self.tan_ref[which]
        g_fd, g_ref = 1.0 + fd ** 2, 1.0 + ref ** 2            # 1/cos^2 = 1 + tan^2
        return {"tan": fd - ref, "beta": np.degrees(np.arctan(fd) - np.arctan(ref)),
                "G_rel": np.abs(g_fd - g_ref) / np.maximum(np.abs(g_ref), 1e-300), "G_fd": g_fd, "G_ref": g_ref}


@lru_cache(maxsize=None)
def audit(spec: CascadeSpec, n_passage: int, n_eta: int = N_ETA) -> Audit:
    c = cascade_of(spec)
    g = grid.make_grid(c, GridSettings(n_passage=n_passage, n_eta=n_eta), math.radians(INLET_ANGLE_DEG))
    gm = metrics.build_metrics(g)
    walls = smooth_of(spec)
    f, l = g.first_blade_i, g.last_blade_i
    m = g.m[f:l + 1]
    ref = {"lower": np.empty(len(m)), "upper": np.empty(len(m))}
    y_err = 0.0
    for k, x in enumerate(m):
        values = walls.walls(float(x))
        for which, node in (("lower", g.lower_boundary[f + k, 1]), ("upper", g.upper_boundary[f + k, 1] - g.pitch)):
            ref[which][k] = values[which][1]
            y_err = max(y_err, abs(values[which][0] - node))
    identity = g.wall_identity
    outline = walls.outline
    return Audit(n_passage, g.dm, m, {"lower": np.array(gm.tan_beta0[f:l + 1]), "upper": np.array(gm.tan_beta1[f:l + 1])},
                 ref, {"lower": np.array(identity.lower.surface), "upper": np.array(identity.upper.surface)}, y_err,
                 g.m_le, g.m_te, float(outline[:, 0].min()), float(outline[:, 0].max()))


def norms(values: np.ndarray) -> tuple[float, float]:
    """L_inf and L2 (root mean square over the nodes)."""
    return float(np.abs(values).max()), float(np.sqrt(np.mean(values ** 2)))


def order(e1: float, e2: float, h1: float, h2: float) -> float:
    return math.log(e1 / e2) / math.log(h1 / h2) if e1 > 0 and e2 > 0 else float("nan")


# --- Blade construction knots in m -------------------------------------------------------

def camber_stations(spec: CascadeSpec, points: np.ndarray) -> np.ndarray:
    """Thickness station of surface points: the arc fraction along the active camberline
    (geometry.py reads thickness at s / S) of each point's projection onto it."""
    camber = cascade_of(spec).active.camberline
    arc = np.concatenate(([0.0], np.cumsum(np.hypot(*np.diff(camber, axis=0).T))))
    arc /= arc[-1]
    a0, d = camber[:-1], np.diff(camber, axis=0)
    out = np.empty(len(points))
    for k, p in enumerate(points):
        s = np.clip(((p - a0) * d).sum(1) / (d * d).sum(1), 0.0, 1.0)
        j = int(np.argmin(np.hypot(*(a0 + s[:, None] * d - p).T)))
        out[k] = arc[j] + s[j] * (arc[j + 1] - arc[j])
    return out


def construction(spec: CascadeSpec, surface: np.ndarray) -> dict[str, int]:
    """Vertex indices of the construction junctions of one surface, in geometry.py's build
    order: LE circle arc, LE fair (N_LE_FAIR points, ending on the first table station),
    geometric nose stations (to x = X_NOSE_END), table stations, TE fair (N_TE_FAIR) and
    TE circle arc."""
    a = cascade_of(spec).active
    tol = 1e-12 * a.chord
    on_le = np.flatnonzero(np.abs(np.hypot(*(surface - a.leading_edge_center).T) - a.leading_edge_radius) < tol)
    out = {"LE circle end": int(on_le[-1])}
    out["LE fair end (station 0.005)"] = out["LE circle end"] + geometry.N_LE_FAIR - 1
    out["nose stations end (x 0.03)"] = out["LE fair end (station 0.005)"] + geometry.N_NOSE_STATIONS - 1
    if a.trailing_edge_radius > 0.0:
        on_te = np.flatnonzero(np.abs(np.hypot(*(surface - a.trailing_edge_center).T) - a.trailing_edge_radius) < tol)
        out["TE fair start"] = int(on_te[0]) - geometry.N_TE_FAIR
        out["TE circle start"] = int(on_te[0])
    return out


@lru_cache(maxsize=None)
def knots(spec: CascadeSpec) -> tuple[tuple[float, str, str], ...]:
    """(m, name, wall) of the blade's construction knots: the junctions above on both
    surfaces, the thickness-table stations and the Kovach junction (station 0.60), each put
    on the wall whose smooth crossing at its m it lies on ('hidden' in a fold)."""
    a = cascade_of(spec).active
    walls = smooth_of(spec)
    table = [x / 100.0 for x in geometry._TABLE_X_PCT if geometry.FIRST_TABLE_STATION < x / 100.0 < 1.0]
    if spec.profile_variant == "kovach_sandercock" and geometry.KOVACH_START_STATION not in table:
        table.append(geometry.KOVACH_START_STATION)
    found = []
    for tag, surface in (("SS", a.suction_surface), ("PS", a.pressure_surface)):
        idx = construction(spec, surface)
        found += [(surface[i], f"{tag} {name}") for name, i in idx.items()]
        start = idx["LE fair end (station 0.005)"]
        sigma, part = camber_stations(spec, surface[start:]), surface[start:]
        keep = np.concatenate(([True], np.diff(sigma) > 0)) & (sigma <= 1.0)
        sigma, part = sigma[keep], part[keep]
        for st in table:
            if sigma[0] < st < sigma[-1]:
                name = "Kovach junction 0.60" if st == geometry.KOVACH_START_STATION else f"table {st:g}"
                found.append((np.array([np.interp(st, sigma, part[:, 0]), np.interp(st, sigma, part[:, 1])]),
                              f"{tag} {name}"))
    out = []
    for point, name in found:     # table points are on the polyline, within ~1e-5 c of the smooth wall
        values = walls.walls(float(point[0]))
        gaps = {w: abs(values[w][0] - point[1]) for w in WALLS}
        wall = min(gaps, key=gaps.get)
        out.append((float(point[0]), name, wall if gaps[wall] <= 1e-4 * a.chord else "hidden"))
    return tuple(sorted(out, key=lambda k: (k[2], k[0])))


KNOT_STYLE = {"table": ("#c9c7c1", 0.6, "-"), "Kovach": ("#d62728", 1.0, (0, (4, 2))),
              "LE": (LANDMARK, 1.0, (0, (4, 2))), "nose": ("#2a78d6", 1.0, (0, (1, 2))), "TE": (LANDMARK, 1.0, (0, (4, 2)))}


def mark_knots(ax: plt.Axes, spec: CascadeSpec, which: str, chord: float) -> None:
    for m, name, wall in knots(spec):
        if wall == which:
            key = next(k for k in ("table", "Kovach", "nose", "LE", "TE") if k in name)
            color, width, line = KNOT_STYLE[key]
            ax.axvline(m / chord, color=color, linewidth=width, linestyle=line, zorder=0)


def reversals(spec: CascadeSpec) -> list[str]:
    """Stretches of the nose (LE point to the end of the nose stations) where a surface
    polyline turns against the LE circle: an S-bend of the outline itself."""
    a = cascade_of(spec).active
    lines = []
    for tag, surface in (("SS", a.suction_surface), ("PS", a.pressure_surface)):
        end = construction(spec, surface)["nose stations end (x 0.03)"]
        seg = np.diff(surface[:end + 1], axis=0)
        angle = np.unwrap(np.arctan2(seg[:, 1], seg[:, 0]))
        turn = np.diff(angle) / (0.5 * (np.hypot(*seg[1:].T) + np.hypot(*seg[:-1].T)))   # at vertices 1..end-1
        against = np.flatnonzero(np.sign(turn) == -np.sign(turn[0])) + 1
        if against.size == 0:
            lines.append(f"{tag}: turns one way from the LE point to the nose-station end (vertex {end}), "
                         f"curvature {np.abs(turn).min():.1f} to {np.abs(turn).max():.1f} /c")
            continue
        k0, k1 = int(against[0]), int(against[-1])
        drop = np.degrees(angle[k0 - 1:k1 + 1].max() - angle[k0 - 1:k1 + 1].min())
        lines.append(f"{tag}: vertices {k0}..{k1} (m/c {surface[k0, 0]:.5f} .. {surface[k1, 0]:.5f}) turn against the "
                     f"LE circle: the surface angle falls back {drop:.2f} deg; curvature there down to "
                     f"{np.abs(turn[against - 1]).max():.1f} /c the wrong way (LE circle {1 / a.leading_edge_radius:.1f} /c)")
    return lines


# --- Reports -----------------------------------------------------------------------------

def report_summary(spec: CascadeSpec) -> None:
    print("\n=== Refinement summary (n_eta %d): reference point = Grid wall node to max |dy| below ===" % N_ETA)
    print(f"{'n_pass':>6} {'dm':>9} {'n_blade':>7} {'|dy|max':>8} | {'wall':5} {'max|dbeta|deg':>13} {'where m/c':>9} "
          f"{'max|dtan|':>10} {'max rel dG':>10} | interior: {'Linf tan':>9} {'L2 tan':>9} {'Linf deg':>9} {'L2 deg':>9}"
          f" | full: {'Linf tan':>9} {'L2 tan':>9} {'Linf deg':>9} {'L2 deg':>9}")
    for n in RESOLUTIONS:
        r = audit(spec, n)
        inner = slice(INTERIOR_EXCLUDE, len(r.m) - INTERIOR_EXCLUDE)
        for which in WALLS:
            e = r.errors(which)
            k = int(np.argmax(np.abs(e["beta"])))
            it, i2 = norms(e["tan"][inner])
            ib, ib2 = norms(e["beta"][inner])
            ft, f2 = norms(e["tan"])
            fb, fb2 = norms(e["beta"])
            print(f"{n:6d} {r.dm:9.6f} {len(r.m):7d} {r.y_check:8.1e} | {which:5} {np.abs(e['beta']).max():13.3f} "
                  f"{r.m[k]:9.4f} {np.abs(e['tan']).max():10.3e} {e['G_rel'].max():10.3e} | "
                  f"{it:9.2e} {i2:9.2e} {ib:9.2e} {ib2:9.2e} | {ft:9.2e} {f2:9.2e} {fb:9.2e} {fb2:9.2e}")


def report_orders(spec: CascadeSpec) -> None:
    print("\n=== Observed orders p = log(E1/E2)/log(h1/h2), actual dm (tan beta error) ===")
    print(f"interior = without the first and last {INTERIOR_EXCLUDE} blade stations; supplementary 'away' = nodes "
          f"{AWAY} c or more from the nose tip and the tail end")
    for which in WALLS:
        rows = []
        for n in RESOLUTIONS:
            r = audit(spec, n)
            e = r.errors(which)["tan"]
            inner = e[INTERIOR_EXCLUDE:len(e) - INTERIOR_EXCLUDE]
            rows.append((n, r.dm, *norms(inner), *norms(e), *norms(e[away_from_ends(r)])))
        print(f"{which} wall: away from the ends, Linf / L2: "
              + "  ".join(f"{n}: {a:.2e}/{b:.2e}" for n, _, *_r, a, b in rows))
        print(f"{which} wall: {'pair':>9} {'int L2':>7} {'int Linf':>8} {'full L2':>8} {'full Linf':>9} "
              f"{'away L2':>8} {'away Linf':>9}")
        pairs = list(zip(rows, rows[1:])) + [(rows[0], rows[-1])]
        for k, (r1, r2) in enumerate(pairs):
            (n1, h1, *e1), (n2, h2, *e2) = r1, r2
            p = [order(e1[j], e2[j], h1, h2) for j in range(6)]
            print(f"{'':11}{n1:4d}>{n2:<4d} {p[1]:7.2f} {p[0]:8.2f} {p[3]:8.2f} {p[2]:9.2f} {p[5]:8.2f} {p[4]:9.2f}"
                  + ("   (overall)" if k == len(pairs) - 1 else ""))


def report_edges(spec: CascadeSpec) -> None:
    print(f"\n=== First and last {EDGE_NODES} blade nodes (k from the LE / TE end) ===")
    for n in CLOSE_UPS:
        r = audit(spec, n)
        print(f"n_passage {n}: dm {r.dm:.5f}; first node {(r.m[0] - r.tip) / r.dm:.3f} dm behind the nose tip, "
              f"last node {(r.tail - r.m[-1]) / r.dm:.3f} dm ahead of the tail end")
        for end, ks in (("LE", range(EDGE_NODES)), ("TE", range(len(r.m) - 1, len(r.m) - 1 - EDGE_NODES, -1))):
            for which in WALLS:
                e = r.errors(which)
                for k in ks:
                    kk = k if end == "LE" else len(r.m) - 1 - k
                    print(f"  {end} {which:5} k={kk} {r.names[which][k][:2].upper()}  m/c {r.m[k]:+.5f}  "
                          f"beta FD {r.beta(which, 'fd')[k]:+8.3f} ref {r.beta(which, 'ref')[k]:+8.3f} err {e['beta'][k]:+8.3f} deg  "
                          f"tan FD {r.tan_fd[which][k]:+9.4f} ref {r.tan_ref[which][k]:+9.4f} |err| {abs(e['tan'][k]):9.3e}  "
                          f"G FD {e['G_fd'][k]:9.4f} ref {e['G_ref'][k]:9.4f}")


def report_node_order(spec: CascadeSpec) -> None:
    print(f"\n=== Node-order errors, k = 0 .. {NODE_ORDER - 1} from each end: |beta err| deg / |tan err| ===")
    for end in ("LE", "TE"):
        for which in WALLS:
            print(f"{end} {which} wall")
            print("  n_pass " + " ".join(f"{'k=' + str(k):>15}" for k in range(NODE_ORDER)))
            for n in RESOLUTIONS:
                r = audit(spec, n)
                e = r.errors(which)
                idx = range(NODE_ORDER) if end == "LE" else range(len(r.m) - 1, len(r.m) - 1 - NODE_ORDER, -1)
                print(f"  {n:6d} " + " ".join(f"{abs(e['beta'][k]):6.2f}/{abs(e['tan'][k]):8.1e}" for k in idx))


def report_formula(spec: CascadeSpec) -> None:
    """The edge stencils applied to the smooth wall itself with steps dm, dm/4, dm/16,
    dm/64 at the fixed node: if the error falls as h^2, the formula is right and dm is
    simply too coarse for the local wall curvature."""
    walls = smooth_of(spec)
    print("\n=== Edge stencils on the smooth wall at the same node, step h = dm / 4^q: |tan error| ===")
    for n in RESOLUTIONS:
        r = audit(spec, n)
        for end, ks in (("LE", (0, 1, 2, 3)), ("TE", (len(r.m) - 1, len(r.m) - 2, len(r.m) - 3, len(r.m) - 4))):
            for which in WALLS:
                cells = []
                for k in ks:
                    x0 = float(r.m[k])
                    errs = []
                    for q in range(4):
                        h = r.dm / 4.0 ** q
                        if end == "LE" and k == 0:
                            pts, w = (x0, x0 + h, x0 + 2 * h), (-3.0, 4.0, -1.0)
                        elif end == "TE" and k == len(r.m) - 1:
                            pts, w = (x0, x0 - h, x0 - 2 * h), (3.0, -4.0, 1.0)
                        else:
                            pts, w = (x0 - h, x0 + h), (-1.0, 1.0)
                        ys = [walls.walls(p)[which][0] for p in pts]
                        errs.append(abs(sum(c * y for c, y in zip(w, ys)) / (2 * h) - r.tan_ref[which][k]))
                    kk = k if end == "LE" else len(r.m) - 1 - k
                    cells.append(f"k={kk} ref {math.degrees(math.atan(r.tan_ref[which][k])):+6.1f}deg: "
                                 + "/".join(f"{x:.1e}" for x in errs))
                if n in CLOSE_UPS:
                    print(f"  {n:4d} {end} {which:5}: " + "  ".join(cells))
    a = cascade_of(spec).active
    print("\n=== End node against the square-root tip model: a node phi = (distance to the tip) / dm from a vertical "
          "tangent gets FD/exact = g(phi) = -3 phi + 4 sqrt(phi (phi+1)) - sqrt(phi (phi+2)) for dm << R ===")
    print(f"{'n_pass':>6} | {'LE phi':>6} {'R_LE/dm':>7} {'g(phi)':>6} {'lower':>6} {'upper':>6} | {'TE phi':>6} "
          f"{'R_TE/dm':>7} {'g(phi)':>6} {'lower':>6} {'upper':>6}   (lower/upper: actual FD/ref at the end node)")
    for n in RESOLUTIONS:
        r = audit(spec, n)
        cells = []
        for k, distance, radius in ((0, r.m[0] - r.tip, a.leading_edge_radius), (-1, r.tail - r.m[-1], a.trailing_edge_radius)):
            phi = distance / r.dm
            g = -3.0 * phi + 4.0 * math.sqrt(phi * (phi + 1.0)) - math.sqrt(phi * (phi + 2.0))
            cells.append(f"{phi:6.3f} {radius / r.dm:7.2f} {g:6.3f} "
                         + " ".join(f"{r.tan_fd[w][k] / r.tan_ref[w][k]:6.3f}" for w in WALLS))
        print(f"{n:6d} | " + " | ".join(cells))


def away_from_ends(r: Audit) -> np.ndarray:
    """Blade nodes at least AWAY chords from the nose tip and the tail end."""
    return (r.m - r.tip >= AWAY) & (r.tail - r.m >= AWAY)


def report_knots(spec: CascadeSpec) -> None:
    print("\n=== Outline S-bends at the nose ===")
    for line in reversals(spec):
        print("  " + line)
    print("\n=== Construction knots: largest |tan err| within 1 dm of the knot / spike factor = that error over the "
          "median error of the nodes 2 to 6 dm from the knot ===")
    runs = {n: audit(spec, n) for n in RESOLUTIONS}
    print(f"{'knot':38} {'wall':5} {'m/c':>8} | " + " ".join(f"{n:>13d}" for n in RESOLUTIONS))
    for m_k, name, which in knots(spec):
        if which not in WALLS:
            print(f"{name:38} {which:5} {m_k:8.5f} |  (hidden in a fold: on neither wall)")
            continue
        cells = []
        for r in runs.values():
            e = np.abs(r.errors(which)["tan"])
            distance = np.abs(r.m - m_k) / r.dm
            near, ring = distance <= 1.0 + 1e-9, (distance > 2.0) & (distance <= 6.0)
            if near.any() and ring.any():
                cells.append(f"{e[near].max():7.1e}/{e[near].max() / np.median(e[ring]):5.1f}")
            else:
                cells.append(f"{'-':>13}")
        print(f"{name:38} {which:5} {m_k:8.5f} | " + " ".join(cells))
    r = runs[RESOLUTIONS[-1]]
    print(f"Median |tan err| at n_passage {RESOLUTIONS[-1]}, {AWAY} c from the ends, ahead of / behind the Kovach junction:")
    for m_k, name, which in knots(spec):
        if "Kovach" in name and which in WALLS:
            e = np.abs(r.errors(which)["tan"])
            sel = away_from_ends(r)
            print(f"  {which}: {np.median(e[sel & (r.m < m_k)]):.2e} / {np.median(e[sel & (r.m > m_k)]):.2e}")


def end_label(k: int, n: int) -> str:
    return f"LE+{k}" if k < n - 1 - k else f"TE-{n - 1 - k}"


def report_cases(spec: CascadeSpec) -> None:
    """Case A: the reference at a steep node is at least as steep as the FD angle (the
    large angle is geometry). Case B: FD steeper than the smooth wall (the large angle is
    made by the difference)."""
    print("\n=== Physical or numerical: steepest nodes, FD against the smooth wall ===")
    print(f"{'n_pass':>6} {'wall':5} | steepest FD node: {'FD':>7} {'ref':>7} case | steepest wall node: {'ref':>7} "
          f"{'FD':>7} | largest |FD|-|ref|: {'deg':>6} {'FD':>7} {'ref':>7} | end nodes |FD|<|ref| | sign differs at")
    for n in RESOLUTIONS:
        r = audit(spec, n)
        for which in WALLS:
            bf, br = r.beta(which, "fd"), r.beta(which, "ref")
            kf, kr = int(np.argmax(np.abs(bf))), int(np.argmax(np.abs(br)))
            excess = np.abs(bf) - np.abs(br)
            ke = int(np.argmax(excess))
            ends = all(abs(r.tan_fd[which][k]) < abs(r.tan_ref[which][k]) for k in (0, len(r.m) - 1))
            flips = [end_label(int(k), len(r.m)) for k in np.flatnonzero(np.sign(bf) != np.sign(br))]
            print(f"{n:6d} {which:5} | {end_label(kf, len(r.m)):>16}: {bf[kf]:+7.2f} {br[kf]:+7.2f}  {'A' if abs(br[kf]) >= abs(bf[kf]) else 'B'}"
                  f"   | {end_label(kr, len(r.m)):>17}: {br[kr]:+7.2f} {bf[kr]:+7.2f} | {end_label(ke, len(r.m)):>17}: "
                  f"{excess[ke]:6.2f} {bf[ke]:+7.2f} {br[ke]:+7.2f} | {str(ends):>20} | {', '.join(flips) or '-'}")


def report_eta(spec: CascadeSpec) -> None:
    print(f"\n=== n_eta spot check at n_passage {SHOWN} ===")
    base = audit(spec, SHOWN, N_ETA_CHECK[0])
    for n_eta in N_ETA_CHECK:
        r = audit(spec, SHOWN, n_eta)
        same = all(np.array_equal(r.tan_fd[w], base.tan_fd[w]) and np.array_equal(r.tan_ref[w], base.tan_ref[w])
                   for w in WALLS) and np.array_equal(r.m, base.m)
        print(f"  n_eta {n_eta:4d}: wall stations, FD and reference slopes identical to n_eta {N_ETA_CHECK[0]}: {same}")


def report_geometry(spec: CascadeSpec) -> None:
    print(f"\n=== Geometry resolution at n_passage {GEOMETRY_CASE}: reference blade {GEOMETRY_REFERENCE} camber stations ===")
    high = smooth_of(replace(spec, n_camber_stations=GEOMETRY_REFERENCE))
    print(f"{'n_camber':>8} | {'wall':5} | FD vs own walls: {'int Linf':>9} {'away Linf':>9} {'full Linf':>9} | "
          f"own walls vs high-res: {'int Linf':>9} {'away Linf':>9} {'full Linf':>9} {'away median':>11}")
    for n_camber in CAMBER_STATIONS:
        s = replace(spec, n_camber_stations=n_camber)
        r = audit(s, GEOMETRY_CASE)
        geo = {w: np.array([high.walls(float(x))[w][1] for x in r.m]) for w in WALLS}
        inner = slice(INTERIOR_EXCLUDE, len(r.m) - INTERIOR_EXCLUDE)
        away = away_from_ends(r)
        for which in WALLS:
            fd = r.errors(which)["tan"]
            rep = r.tan_ref[which] - geo[which]
            print(f"{n_camber:8d} | {which:5} | {'':16} {np.abs(fd[inner]).max():9.2e} {np.abs(fd[away]).max():9.2e} "
                  f"{np.abs(fd).max():9.2e} | {'':22} {np.abs(rep[inner]).max():9.2e} {np.abs(rep[away]).max():9.2e} "
                  f"{np.abs(rep).max():9.2e} {np.median(np.abs(rep[away])):11.2e}")
    r = audit(spec, GEOMETRY_CASE)
    for which in WALLS:
        geo = np.array([high.walls(float(x))[which][1] for x in r.m])
        total = r.tan_fd[which] - geo
        k = int(np.argmax(np.abs(total) * away_from_ends(r)))
        print(f"  default blade, {which} wall, away from the ends: FD against the {GEOMETRY_REFERENCE}-station blade "
              f"{np.abs(total[away_from_ends(r)]).max():.2e} at m/c {r.m[k]:.4f}")


# --- Figures -----------------------------------------------------------------------------

def style(ax: plt.Axes, title: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=10, loc="left")
    ax.set_xlabel("m / c")
    ax.set_ylabel(ylabel)
    ax.grid(True, color="#e4e3df", linewidth=0.6)


def landmarks(ax: plt.Axes, r: Audit, chord: float) -> None:
    for x, name, style_ in ((r.m[0], "first blade node", "-"), (r.m[-1], "last blade node", "-"),
                            (r.m_le, "exact LE", "--"), (r.m_te, "exact TE", "--"),
                            (r.tip, "nose tip", ":"), (r.tail, "tail end", ":")):
        ax.axvline(x / chord, color=LANDMARK if "node" in name else INK, linewidth=0.9, linestyle=style_, label=name,
                   zorder=1)


def figure_angles(spec: CascadeSpec) -> plt.Figure:
    r = audit(spec, SHOWN)
    chord = cascade_of(spec).active.chord
    walls = smooth_of(spec)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), gridspec_kw={"width_ratios": (2.2, 1, 1)})
    fig.suptitle(f"Grid-line angle on the walls, n_passage {SHOWN}: FD (metrics.py) against the smooth wall's tangent",
                 x=0.04, ha="left")
    for row, which in enumerate(WALLS):
        for col, span in enumerate((None, (r.tip - 0.004, r.m[0] + 6 * r.dm), (r.m[-1] - 6 * r.dm, r.tail + 0.004))):
            ax = axes[row, col]
            if span is not None:
                dense = np.linspace(max(span[0], r.tip + 1e-7), min(span[1], r.tail - 1e-7), 300)
                ax.plot(dense / chord, walls.dense_angle(which, dense), color=MUTED, linewidth=1.2,
                        label="smooth wall (dense)")
            ax.plot(r.m / chord, r.beta(which, "ref"), "o-", color="#2a78d6", markersize=2.5, linewidth=1.0,
                    label="beta_ref at the nodes")
            ax.plot(r.m / chord, r.beta(which, "fd"), "x", color="#eb6834", markersize=4, label="beta_FD")
            landmarks(ax, r, chord)
            if span is not None:
                ax.set_xlim(span[0] / chord, span[1] / chord)
            name = "lower wall (reference blade's upper contour)" if which == "lower" else \
                "upper wall (neighbor's lower contour)"
            style(ax, name if span is None else ("LE close-up" if col == 1 else "TE close-up"), "beta [deg]")
        axes[row, 0].legend(fontsize=7, ncol=2, loc="best")
    fig.tight_layout()
    return fig


def figure_errors(spec: CascadeSpec, kind: str) -> plt.Figure:
    chord = cascade_of(spec).active.chord
    fig, axes = plt.subplots(2, 1, figsize=(13, 8.5), sharex=True)
    label = "|beta_FD - beta_ref| [deg]" if kind == "beta" else "|tan beta_FD - tan beta_ref|"
    fig.suptitle(f"{label} along the walls, every resolution (grey: thickness-table stations; red dashed: Kovach "
                 f"junction; magenta dashed: LE circle / LE fair ends, TE fair / TE circle starts; blue dotted: end of "
                 f"the nose stations)", x=0.04, ha="left", fontsize=9)
    for ax, which in zip(axes, WALLS):
        mark_knots(ax, spec, which, chord)
        for n, color in zip(RESOLUTIONS, COLORS):
            r = audit(spec, n)
            e = np.abs(r.errors(which)[kind])
            ax.semilogy(r.m / chord, np.maximum(e, 1e-16), color=color, linewidth=1.0, marker="o", markersize=1.6,
                        label=f"n_passage {n}")
        style(ax, f"{which} wall", label)
        ax.set_ylim(1e-8, None)
    axes[0].legend(fontsize=7, ncol=4)
    fig.tight_layout()
    return fig


def figure_G(spec: CascadeSpec) -> plt.Figure:
    r = audit(spec, G_CASE)
    chord = cascade_of(spec).active.chord
    fig, axes = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True)
    fig.suptitle(f"G = 1/cos^2 beta (A of Eq. 5-22 carries it), n_passage {G_CASE}", x=0.04, ha="left")
    for ax, which in zip(axes, WALLS):
        e = r.errors(which)
        ax.semilogy(r.m / chord, e["G_ref"], "o-", color="#2a78d6", markersize=2.5, linewidth=1.0, label="G_ref")
        ax.semilogy(r.m / chord, e["G_fd"], "x", color="#eb6834", markersize=4, label="G_FD")
        landmarks(ax, r, chord)
        style(ax, f"{which} wall: largest relative error {e['G_rel'].max():.2f}", "1/cos^2 beta")
        ax.legend(fontsize=7, ncol=4)
    fig.tight_layout()
    return fig


def figure_edges(spec: CascadeSpec) -> plt.Figure:
    chord = cascade_of(spec).active.chord
    walls = smooth_of(spec)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f"First and last {EDGE_NODES} blade nodes: beta_FD (markers, per resolution) on the smooth wall's "
                 f"dense angle (grey)", x=0.04, ha="left")
    last = audit(spec, CLOSE_UPS[-1])
    for col, end in enumerate(("LE", "TE")):
        for row, which in enumerate(WALLS):
            ax = axes[row, col]
            lo, hi = np.inf, -np.inf
            for n, marker in zip(CLOSE_UPS, ("o", "s", "^", "D")):
                r = audit(spec, n)
                sel = slice(0, EDGE_NODES) if end == "LE" else slice(len(r.m) - EDGE_NODES, None)
                color = COLORS[RESOLUTIONS.index(n)]
                ax.plot(r.m[sel] / chord, r.beta(which, "fd")[sel], marker, color=color, markersize=5, linestyle="none",
                        label=f"FD, n_passage {n}")
                ax.plot(r.m[sel] / chord, r.beta(which, "ref")[sel], marker, color=color, markersize=5,
                        markerfacecolor="none", linestyle="none")
                lo, hi = min(lo, r.m[sel].min()), max(hi, r.m[sel].max())
            dense = np.linspace(max(lo, last.tip + 1e-7), min(hi, last.tail - 1e-7), 400)
            ax.plot(dense / chord, walls.dense_angle(which, dense), color=MUTED, linewidth=1.2, zorder=0,
                    label="smooth wall (open markers: beta_ref at the nodes)")
            for x, style_ in ((last.m_le if end == "LE" else last.m_te, "--"), (last.tip if end == "LE" else last.tail, ":")):
                ax.axvline(x / chord, color=INK, linewidth=0.8, linestyle=style_)
            style(ax, f"{end}, {which} wall", "beta [deg]")
            ax.legend(fontsize=7)
    fig.tight_layout()
    return fig


def local_orders(spec: CascadeSpec, which: str) -> tuple[np.ndarray, list[np.ndarray]]:
    """p(m) between successive resolutions on LOCAL_POINTS common m locations spanning the
    m range every grid has blade nodes in: |tan error| of each grid interpolated linearly
    between its nodes; NaN where either error is below NEAR_ZERO."""
    runs = [audit(spec, n) for n in RESOLUTIONS]
    x = np.linspace(max(r.m[0] for r in runs), min(r.m[-1] for r in runs), LOCAL_POINTS)
    e = [np.interp(x, r.m, np.abs(r.errors(which)["tan"])) for r in runs]
    out = []
    for k in range(len(runs) - 1):
        with np.errstate(divide="ignore", invalid="ignore"):
            p = np.log(e[k] / e[k + 1]) / math.log(runs[k].dm / runs[k + 1].dm)
        p[(e[k] < NEAR_ZERO) | (e[k + 1] < NEAR_ZERO)] = np.nan
        out.append(p)
    return x, out


def running_median(x: np.ndarray, p: np.ndarray, window: float) -> np.ndarray:
    return np.array([np.nanmedian(p[np.abs(x - xi) <= 0.5 * window]) if np.isfinite(p[np.abs(x - xi) <= 0.5 * window]).any()
                     else np.nan for xi in x])


def report_local_orders(spec: CascadeSpec) -> None:
    print(f"\n=== Local order p(m), {LOCAL_POINTS} common points: median p in the first / middle / last zone "
          f"(zones: within {AWAY} c of the common range's ends, and between) ===")
    for which in WALLS:
        x, ps = local_orders(spec, which)
        zones = (x < x[0] + AWAY, (x >= x[0] + AWAY) & (x <= x[-1] - AWAY), x > x[-1] - AWAY)
        cells = []
        for k, p in enumerate(ps):
            cells.append(f"{RESOLUTIONS[k]}>{RESOLUTIONS[k + 1]}: " + "/".join(
                f"{np.nanmedian(p[z]):+.2f}" if np.isfinite(p[z]).any() else "  nan" for z in zones))
        print(f"{which} wall (m/c {x[0]:.4f} .. {x[-1]:.4f}): " + "  ".join(cells))


def figure_order(spec: CascadeSpec) -> plt.Figure:
    chord = cascade_of(spec).active.chord
    fig, axes = plt.subplots(2, 1, figsize=(13, 8.5), sharex=True)
    fig.suptitle(f"Local observed order p(m) of the |tan beta| error, successive resolutions (thin: at "
                 f"{LOCAL_POINTS} common points; thick: running median over {MEDIAN_WINDOW} c). Second order: p = 2",
                 x=0.04, ha="left", fontsize=10)
    for ax, which in zip(axes, WALLS):
        mark_knots(ax, spec, which, chord)
        x, ps = local_orders(spec, which)
        for k, p in enumerate(ps):
            ax.plot(x / chord, p, color=COLORS[k + 1], linewidth=0.5, alpha=0.45)
            ax.plot(x / chord, running_median(x, p, MEDIAN_WINDOW), color=COLORS[k + 1], linewidth=1.8,
                    label=f"{RESOLUTIONS[k]} > {RESOLUTIONS[k + 1]} (median {np.nanmedian(p):.2f})")
        ax.axhline(2.0, color=INK, linewidth=0.8, linestyle="--")
        ax.axhline(1.0, color=MUTED, linewidth=0.8, linestyle=":")
        ax.set_ylim(-4, 6)
        style(ax, f"{which} wall", "p")
        ax.legend(fontsize=7, ncol=4)
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--save", type=Path, default=None, help="folder to write the figures to")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    spec = EXAMPLE
    report_summary(spec)
    report_orders(spec)
    report_local_orders(spec)
    report_edges(spec)
    report_node_order(spec)
    report_cases(spec)
    report_formula(spec)
    report_knots(spec)
    report_eta(spec)
    report_geometry(spec)
    figures = {"1_angles": figure_angles(spec), "2_beta_error": figure_errors(spec, "beta"),
               "3_tan_error": figure_errors(spec, "tan"), "4_G": figure_G(spec), "5_edges": figure_edges(spec),
               "6_local_order": figure_order(spec)}
    if args.save is not None:
        args.save.mkdir(parents=True, exist_ok=True)
        for name, fig in figures.items():
            fig.savefig(args.save / f"metrics_accuracy_{name}.png", dpi=130)
        print(f"\nsaved {len(figures)} figures to {args.save}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
