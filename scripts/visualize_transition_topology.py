"""
visualize_transition_topology.py
================================
Diagnostic and explanatory figures for the LE / TE transition topology of the
blade-to-blade (m, eta) grid. Read-only: it builds the current EXAMPLE cascade and grid
through the package (geometry -> grid -> boundary_conditions, and metrics ->
coefficients -> discretization only to read the periodic PDE rows the code writes today)
and changes nothing in the package.

    python scripts/visualize_transition_topology.py               # n_passage 101 and 41, n_eta 41
    python scripts/visualize_transition_topology.py --cases 101   # one case
    python scripts/visualize_transition_topology.py --show        # also open the windows

Writes outputs/transition_topology/np<N>/{A..E}_*.png and report.txt per case, and prints
the report.

The three things this script keeps apart
    1. A station line: the vertical line x = m[i] (planar cascade, x = m). At a blade
       station it DOES cut the blade; the grid only uses the part between the two
       passage-facing walls, which holds no metal (a grid.py invariant). Cutting the blade
       is therefore not a defect in itself.
    2. A same-eta link: the straight segment (i, j) -> (i +- 1, j). psi_m (Eq. 5-35) and
       psi_mm (Eq. 5-37) at (i, j) read (i - 1, j) and (i + 1, j) and assume psi varies
       smoothly along eta = const between them, so both links W-P and P-E must lie in fluid.
    3. The mixed-derivative relations: psi_m,eta (Eq. 5-39) reads the four corners
       (i +- 1, j +- 1). It needs (a) each corner to EXIST in the topology (beyond a wall
       at a blade station there is blade, not a node; on a periodic station the corner
       beyond eta = 0 / 1 exists through the periodic continuation psi[k, J + 1] =
       psi[k, 1] + 1, located at node (k, 1) + s), and (b) the corner relations P -> corner
       (and, as asked, the straight diagonals SW-NE and NW-SE) to lie in fluid.

"Metal" is what grid.py calls metal: the inside of the blade's smooth walls, the C2
quintic curves (Cascade.active_curves), for the reference blade and its periodic copies
k s, s = periodic_translation. Every crossing of a drawn link with a wall is solved on the
curve itself (brentq in u to 1e-15), inside/outside by the parity of a vertical ray's
crossings on monotone-in-m branches of the curves (the grid's own cut, done exactly), so
the colours in the figures are computed, not judged by eye. C1 cascades use the
polylines the same way.

Dev-only, diagnostic only: the package never imports this, and nothing here feeds back.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib

if "--show" not in sys.argv:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import brentq  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import plot_cascade as workflow  # noqa: E402  (EXAMPLE, SETTINGS and the angles of the current workflow)
from potential_flow_solver import discretization as dz  # noqa: E402
from potential_flow_solver import geometry  # noqa: E402
from potential_flow_solver import grid as grid_module  # noqa: E402
from potential_flow_solver.boundary_conditions import build_boundary_conditions  # noqa: E402
from potential_flow_solver.coefficients import build_coefficients  # noqa: E402
from potential_flow_solver.metrics import build_metrics  # noqa: E402

# --- Inputs ------------------------------------------------------------------------------
CASES = (101, 41)               # n_passage values; 101 is the main case, 41 the comparison
N_ETA = 41
OUTPUT = ROOT / "outputs" / "transition_topology"
# ---------------------------------------------------------------------------------------

LINK_TOUCH = 1e-10              # chords: metal on a link at or below this is a touch (round-off), not a crossing
COPIES = (-2, -1, 0, 1, 2)      # blade copies k s checked for metal
CURVE_SAMPLES = 20001           # coarse samples per surface, for bracketing only
LOCAL_SAMPLES = 2001            # local resampling of a curve piece near a link

FLUID, METAL, MISSING = "fluid", "metal", "missing"
OFFSETS = {"W": (-1, 0), "E": (1, 0), "S": (0, -1), "N": (0, 1),
           "SW": (-1, -1), "SE": (1, -1), "NW": (-1, 1), "NE": (1, 1)}
USED_BY = {"W": "psi_m, psi_mm", "E": "psi_m, psi_mm", "S": "psi_eta, psi_etaeta", "N": "psi_eta, psi_etaeta",
           "SW": "psi_meta", "SE": "psi_meta", "NW": "psi_meta", "NE": "psi_meta"}


# =========================================================================================
# Exact metal oracle
# =========================================================================================

class _Surface:
    """One blade surface X(u), 0 <= u <= 1, vectorized: the C2 curve or a C1 polyline."""

    def __init__(self, name: str, curve=None, polyline: np.ndarray | None = None) -> None:
        self.name = name
        if curve is not None:
            self._f = curve.evaluate
            self._df = lambda u: curve.derivative(u, 1)
        else:
            pts = np.asarray(polyline, dtype=float)
            s = np.linspace(0.0, 1.0, len(pts))
            self._f = lambda u: np.column_stack((np.interp(np.atleast_1d(u), s, pts[:, 0]),
                                                 np.interp(np.atleast_1d(u), s, pts[:, 1]))).squeeze()
            self._df = None
            self._knots = s
        self.u = np.linspace(0.0, 1.0, CURVE_SAMPLES)
        self.samples = np.asarray(self._f(self.u), dtype=float)
        self.tree = cKDTree(self.samples)
        self.spacing = float(np.max(np.hypot(*np.diff(self.samples, axis=0).T)))
        self.branches = self._monotone_branches()

    def __call__(self, u):
        return np.asarray(self._f(u), dtype=float)

    def _monotone_branches(self) -> list[tuple[float, float, float, float]]:
        """(u0, u1, m0, m1) pieces on which m(u) is monotone: split at the roots of m'(u)
        (exact for the curve, the vertices for a polyline)."""
        if self._df is not None:
            mu = np.asarray(self._df(self.u), dtype=float)[:, 0]
            cuts = [0.0]
            for k in np.flatnonzero(np.sign(mu[:-1]) * np.sign(mu[1:]) < 0):
                cuts.append(brentq(lambda v: float(self._df(v)[0]), self.u[k], self.u[k + 1], xtol=1e-15))
            cuts.append(1.0)
        else:
            m = self.samples[:, 0]
            dm = np.diff(m)
            cuts = [0.0] + [float(self.u[k + 1]) for k in np.flatnonzero(np.sign(dm[:-1]) * np.sign(dm[1:]) < 0)] + [1.0]
        out = []
        for u0, u1 in zip(cuts[:-1], cuts[1:]):
            m0, m1 = float(self(u0)[0]), float(self(u1)[0])
            out.append((u0, u1, m0, m1))
        return out

    def y_at(self, x: float) -> list[float]:
        """Every y with X(u) = (x, y): one root per monotone branch that spans x."""
        ys = []
        for u0, u1, m0, m1 in self.branches:
            lo, hi = min(m0, m1), max(m0, m1)
            if not lo < x < hi:
                continue
            u = brentq(lambda v: float(self(v)[0]) - x, u0, u1, xtol=1e-15, rtol=4 * np.finfo(float).eps)
            ys.append(float(self(u)[1]))
        return ys


@dataclass(frozen=True)
class LinkCheck:
    """A straight segment p -> q against the blade metal."""

    p: np.ndarray
    q: np.ndarray
    metal: float                            # length inside metal, chords
    intervals: tuple                        # ((x0, y0), (x1, y1)) pieces inside metal
    crossings: int                          # wall crossings strictly inside the segment

    @property
    def length(self) -> float:
        return float(np.hypot(*(self.q - self.p)))

    @property
    def status(self) -> str:
        return METAL if self.metal > LINK_TOUCH else FLUID


class BladeMetal:
    """Exact link / point tests against the reference blade and its copies k s."""

    def __init__(self, cascade) -> None:
        a = cascade.active
        self.s = np.asarray(a.periodic_translation, dtype=float)
        if abs(self.s[0]) > 0.0:
            raise ValueError("this diagnostic assumes a purely tangential periodic translation (0, pitch)")
        curves = cascade.active_curves
        if curves is not None:
            self.surfaces = [_Surface("suction", curves.suction), _Surface("pressure", curves.pressure)]
            self.kind = "C2 quintic curves (exact roots)"
        else:
            self.surfaces = [_Surface("suction", polyline=a.suction_surface),
                             _Surface("pressure", polyline=a.pressure_surface)]
            self.kind = "C1 polylines (exact roots)"
        self.margin = 3.0 * max(s.spacing for s in self.surfaces)

    def inside(self, point: np.ndarray) -> bool:
        """Parity of the upward vertical ray's crossings with every blade copy."""
        x, y = float(point[0]), float(point[1])
        count = 0
        for surface in self.surfaces:
            for yr in surface.y_at(x):
                for k in COPIES:
                    if yr + k * self.s[1] > y:
                        count += 1
        return count % 2 == 1

    def _roots(self, p: np.ndarray, d: np.ndarray, length: float) -> list[float]:
        """Parameters t in [0, 1] where p + t d meets a wall (exact roots on the curves)."""
        mid = p + 0.5 * d
        radius = 0.5 * length + self.margin
        ts = []
        for surface in self.surfaces:
            for k in COPIES:
                off = k * self.s
                idx = surface.tree.query_ball_point(mid - off, radius)
                if not idx:
                    continue
                idx = np.sort(np.asarray(idx))
                groups = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
                for g in groups:
                    ua = surface.u[max(int(g[0]) - 2, 0)]
                    ub = surface.u[min(int(g[-1]) + 2, CURVE_SAMPLES - 1)]
                    uu = np.linspace(ua, ub, LOCAL_SAMPLES)
                    X = surface(uu) + off
                    gv = d[0] * (X[:, 1] - p[1]) - d[1] * (X[:, 0] - p[0])

                    def g_of(v, surface=surface, off=off):
                        Xv = surface(v) + off
                        return float(d[0] * (Xv[1] - p[1]) - d[1] * (Xv[0] - p[0]))

                    roots = [float(uu[k2]) for k2 in np.flatnonzero(gv == 0.0)]
                    for k2 in np.flatnonzero(np.sign(gv[:-1]) * np.sign(gv[1:]) < 0):
                        roots.append(brentq(g_of, uu[k2], uu[k2 + 1], xtol=1e-15, rtol=4 * np.finfo(float).eps))
                    for u in roots:
                        Xr = surface(u) + off
                        t = float(np.dot(Xr - p, d) / length ** 2)
                        if -1e-12 <= t <= 1.0 + 1e-12:
                            ts.append(min(max(t, 0.0), 1.0))
        return ts

    def link(self, p, q) -> LinkCheck:
        p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
        d = q - p
        length = float(np.hypot(*d))
        if length == 0.0:
            return LinkCheck(p, q, 0.0, (), 0)
        ts = sorted(self._roots(p, d, length))
        merged = [0.0]
        for t in ts + [1.0]:
            if t - merged[-1] > 1e-13:
                merged.append(t)
        merged[-1] = 1.0
        metal, intervals = 0.0, []
        for ta, tb in zip(merged[:-1], merged[1:]):
            if self.inside(p + 0.5 * (ta + tb) * d):
                metal += (tb - ta) * length
                intervals.append((tuple(p + ta * d), tuple(p + tb * d)))
        crossings = sum(1 for t in ts if 1e-12 < t < 1.0 - 1e-12)
        return LinkCheck(p, q, metal, tuple(intervals), crossings)

    def cut(self, x: float, k: int = 0) -> tuple[float, float] | None:
        """(bottom, top) of blade copy k on the station line x, None if the line misses it."""
        ys = sorted(y + k * self.s[1] for surface in self.surfaces for y in surface.y_at(x))
        return (ys[0], ys[-1]) if ys else None


# =========================================================================================
# The case: cascade, grid, node classes, stencil requirements
# =========================================================================================

@dataclass(frozen=True)
class Neighbor:
    """One neighbor a central difference at P reads."""

    name: str                       # W, E, S, N, SW, SE, NW, NE
    node: tuple[int, int]           # (k, jj) in the local indexing, jj may be -1 or n_eta
    exists: bool
    real: tuple[int, int] | None    # the real node that holds the value, None if missing
    shift: int                      # copies of s added to the real node's position (periodic continuation)
    point: np.ndarray | None        # physical position of the neighbor, None if missing
    why: str                        # how it exists, or why it does not
    check: LinkCheck | None         # P -> neighbor, None if missing
    ghost_point: np.ndarray | None = None   # for a missing corner: where the continuation WOULD put it
    ghost_check: LinkCheck | None = None    # the eta pair at that station the would-be value enters
    ghost_link: LinkCheck | None = None     # and the straight P -> would-be link

    @property
    def status(self) -> str:
        if not self.exists:
            return MISSING
        return self.check.status


class Case:
    """Everything the figures and the report read for one n_passage."""

    def __init__(self, n_passage: int, n_eta: int = N_ETA) -> None:
        self.n_passage = n_passage
        self.settings = replace(workflow.SETTINGS, n_passage=n_passage, n_eta=n_eta)
        self.spec = workflow.EXAMPLE
        self.cascade = geometry.make_cascade(self.spec)
        self.inlet_angle = math.radians(workflow.INLET_ANGLE_DEG)
        self.exit_angle = None if workflow.EXIT_ANGLE_DEG is None else math.radians(workflow.EXIT_ANGLE_DEG)
        self.grid = grid_module.make_grid(self.cascade, self.settings, self.inlet_angle, self.exit_angle)
        self.bc = build_boundary_conditions(self.grid)
        self.metal = BladeMetal(self.cascade)
        g = self.grid
        self.n_m, self.n_eta = g.nodes.shape[:2]
        self.J = self.n_eta - 1
        self.f, self.l = g.first_blade_i, g.last_blade_i
        self.s = np.asarray(self.cascade.active.periodic_translation, dtype=float)
        self.periodic = np.zeros(self.n_m, dtype=bool)
        self.periodic[self.bc.periodic_stations] = True
        self._links: dict = {}
        # The rows the code writes today on the periodic line (read-only use of the package).
        metrics = build_metrics(g)
        co = build_coefficients(metrics, np.ones((self.n_m, self.n_eta)), 1.0)
        self.periodic_rows = {row.target[0]: row for row in dz.build_periodic_pde_rows(co, metrics, self.bc)}

    # --- node classes -----------------------------------------------------------------
    def node_class(self, i: int, j: int) -> str:
        b = self.bc
        if b.upstream[i, j]:
            return "inlet boundary"
        if b.downstream[i, j]:
            return "outlet boundary"
        if b.lower_wall[i, j]:
            return "lower wall"
        if b.upper_wall[i, j]:
            return "upper wall"
        if b.periodic_lower[i, j]:
            return "periodic (constraint side)"
        if b.periodic_upper[i, j]:
            return "periodic (PDE side)"
        return "interior"

    def has_pde_row(self, i: int, j: int) -> bool:
        """Interior nodes (InteriorStencil) and the periodic PDE side (i, J) off the far stations."""
        return bool(self.bc.interior[i, j] or (self.bc.periodic_upper[i, j] and 0 < i < self.n_m - 1))

    def point(self, i: int, j: int) -> np.ndarray:
        return np.asarray(self.grid.nodes[i, j], dtype=float)

    def link(self, p, q) -> LinkCheck:
        key = (tuple(np.round(p, 15)), tuple(np.round(q, 15)))
        if key not in self._links:
            self._links[key] = self.metal.link(p, q)
        return self._links[key]

    # --- neighbors the central formulas read ---------------------------------------------
    def neighbor(self, i: int, j: int, name: str) -> Neighbor:
        di, dj = OFFSETS[name]
        k, jj = i + di, j + dj
        P = self.point(i, j)
        if not 0 <= k < self.n_m:
            return Neighbor(name, (k, jj), False, None, 0, None, "outside the far boundary", None)
        if 0 <= jj <= self.J:
            q = self.point(k, jj)
            return Neighbor(name, (k, jj), True, (k, jj), 0, q, "real node", self.link(P, q))
        above = jj > self.J
        real = (k, jj - self.J) if above else (k, jj + self.J)
        shift = 1 if above else -1
        image = self.point(*real) + shift * self.s
        if self.periodic[k]:
            why = (f"periodic continuation psi[{k},{jj}] = psi[{real[0]},{real[1]}] {'+' if above else '-'} 1 "
                   f"at node ({real[0]},{real[1]}) {'+' if above else '-'} s")
            return Neighbor(name, (k, jj), True, real, shift, image, why, self.link(P, image))
        side = "above the upper wall" if above else "below the lower wall"
        blade = "+pitch blade" if above else "reference blade"
        why = (f"station {k} is a blade station: {side} lies the {blade}; no node and no continuation "
               f"(the continuation would put it at ({real[0]},{real[1]}) {'+' if above else '-'} s, across the cut)")
        # The eta difference q[k] = (psi[k, j + 1] - psi[k, j - 1]) / (2 deta) the mixed term would form at
        # station k: from the real node on the other side of P's row to the would-be value, across the cut.
        partner = (k, j - 1) if above else (k, j + 1)
        pair = self.link(self.point(*partner), image) if 0 <= partner[1] <= self.J else None
        return Neighbor(name, (k, jj), False, None, 0, None, why, None, image, pair, self.link(P, image))

    def stencil(self, i: int, j: int) -> dict[str, Neighbor]:
        return {name: self.neighbor(i, j, name) for name in OFFSETS}

    def diagonal(self, i: int, j: int, a: str, b: str) -> LinkCheck | None:
        """The straight full diagonal between corners a and b, None if either is missing."""
        na, nb = self.neighbor(i, j, a), self.neighbor(i, j, b)
        if not (na.exists and nb.exists):
            return None
        return self.link(na.point, nb.point)

    # --- scans --------------------------------------------------------------------------
    def window(self, end: str) -> range:
        return range(self.f - 2, self.f + 2) if end == "le" else range(self.l - 1, self.l + 3)

    def scan(self, end: str) -> list[dict]:
        """Every PDE node of the transition window and every central-difference neighbor that
        is not a clean fluid link."""
        out = []
        for i in self.window(end):
            for j in range(self.n_eta):
                if not self.has_pde_row(i, j):
                    continue
                st = self.stencil(i, j)
                for name, nb in st.items():
                    if nb.status != FLUID:
                        out.append(dict(i=i, j=j, cls=self.node_class(i, j), name=name, used_by=USED_BY[name],
                                        status=nb.status, node=nb.node, real=nb.real, why=nb.why,
                                        metal=(nb.check.metal if nb.check is not None else
                                               nb.ghost_check.metal if nb.ghost_check is not None else float("nan")),
                                        length=(nb.check.length if nb.check is not None else
                                                nb.ghost_check.length if nb.ghost_check is not None else float("nan"))))
                for a, b in (("SW", "NE"), ("NW", "SE")):
                    chk = self.diagonal(i, j, a, b)
                    if chk is not None and chk.status == METAL:
                        out.append(dict(i=i, j=j, cls=self.node_class(i, j), name=f"{a}-{b}", used_by="psi_meta (full diagonal)",
                                        status=METAL, node=None, real=None, why="straight diagonal", metal=chk.metal,
                                        length=chk.length))
        return out

    def same_eta_census(self) -> dict:
        """Every same-eta link (i, j) -> (i + 1, j) of the grid, by where it sits, and those through metal."""
        census = {}
        for i in range(self.n_m - 1):
            if i == self.f - 1:
                where = "LE transition interval"
            elif i == self.l:
                where = "TE transition interval"
            elif i < self.f - 1:
                where = "inlet"
            elif i > self.l:
                where = "exit"
            else:
                where = "blade"
            for j in range(self.n_eta):
                row = "boundary row" if j in (0, self.J) else "interior row"
                key = (where, row)
                entry = census.setdefault(key, dict(n=0, bad=[]))
                entry["n"] += 1
                chk = self.link(self.point(i, j), self.point(i + 1, j))
                if chk.status == METAL:
                    entry["bad"].append((i, j, chk.metal, chk.length))
        return census

    def where(self, i: int) -> str:
        """Where the m-interval i -> i + 1 sits."""
        return ("LE transition interval" if i == self.f - 1 else "TE transition interval" if i == self.l else
                "inlet" if i < self.f - 1 else "exit" if i > self.l else "blade")

    def diagonal_census(self) -> dict:
        """Every cell diagonal (i, j) -> (i + 1, j + 1) and (i, j + 1) -> (i + 1, j), j = 0 .. J - 1,
        and across the periodic seam of every pair of periodic stations, (i, J) -> (i + 1, 1) + s and
        (i, 1) + s -> ... written as (i + 1, J) -> (i, 1) + s, through the continuation."""
        census = {}
        for i in range(self.n_m - 1):
            entry = census.setdefault(self.where(i), dict(n=0, bad=[]))
            pairs = [(self.point(i, j), self.point(i + 1, j + 1), (i, j), (i + 1, j + 1)) for j in range(self.J)]
            pairs += [(self.point(i, j + 1), self.point(i + 1, j), (i, j + 1), (i + 1, j)) for j in range(self.J)]
            if self.periodic[i] and self.periodic[i + 1]:
                pairs.append((self.point(i, self.J), self.point(i + 1, 1) + self.s, (i, self.J), (i + 1, self.J + 1)))
                pairs.append((self.point(i + 1, self.J), self.point(i, 1) + self.s, (i + 1, self.J), (i, self.J + 1)))
            for p, q, a, b in pairs:
                entry["n"] += 1
                chk = self.link(p, q)
                if chk.status == METAL:
                    entry["bad"].append((a, b, chk.metal, chk.length))
        return census

    def code_row_links(self, i: int) -> list[tuple]:
        """The nodes the current periodic PDE row at (i, J) reads (discretization.build_periodic_pde_rows),
        each with the physical point it stands for and the straight link from the row's node to it.
        On that row a node (k, 1) always enters through the continuation psi[k, J + 1] = psi[k, 1] + 1,
        so it stands for (k, 1) + s."""
        row = self.periodic_rows[i]
        P = self.point(i, self.J)
        out = []
        for term in row.terms:
            k, jj = term.node
            if (k, jj) == (i, self.J):
                continue
            q = self.point(k, jj) + (self.s if jj == 1 else 0.0)
            out.append((term.node, term.coefficient, q, self.link(P, q)))
        return out


# =========================================================================================
# Drawing helpers
# =========================================================================================

SURFACE, INK, INK2 = workflow.SURFACE, workflow.INK, workflow.INK_SECONDARY
C_FLUID, C_METAL, C_METAL_DARK, C_MISSING = "#1baf7a", "#d62728", "#6e0b0b", "#8a5cd6"
C_CODE = "#0f5e9c"
C_SUCTION, C_PRESSURE = workflow.SUCTION, workflow.PRESSURE
C_PERIODIC, C_EXIT, C_FAR, C_INTERIOR = workflow.EXTENSION, workflow.EXIT_SIDE, "#6b6b6b", "#52514e"
METAL_FILL, TRANSITION_FILL, STATION = "#d8d4cc", "#f8eec4", "#9a9893"

CLASS_STYLE = {
    "interior": dict(marker="o", markersize=3.4, markerfacecolor=C_INTERIOR, markeredgecolor=C_INTERIOR,
                     markeredgewidth=0.0),
    "lower wall": dict(marker="o", markersize=6.0, markerfacecolor=C_SUCTION, markeredgecolor=INK, markeredgewidth=0.6),
    "upper wall": dict(marker="s", markersize=5.6, markerfacecolor=C_PRESSURE, markeredgecolor=INK, markeredgewidth=0.6),
    "periodic (constraint side)": dict(marker="o", markersize=8.0, markerfacecolor="white", markeredgecolor=C_PERIODIC,
                                       markeredgewidth=1.6),
    "periodic (PDE side)": dict(marker="D", markersize=4.6, markerfacecolor=C_PERIODIC, markeredgecolor=INK,
                                markeredgewidth=0.5),
    "inlet boundary": dict(marker=">", markersize=6.0, markerfacecolor=C_FAR, markeredgecolor=INK, markeredgewidth=0.5),
    "outlet boundary": dict(marker="<", markersize=6.0, markerfacecolor=C_FAR, markeredgecolor=INK, markeredgewidth=0.5),
}
CLASS_LABEL = {
    "interior": "interior node (Eq. 5-44 row)",
    "lower wall": "lower wall, j = 0 on a blade station (psi = 0)",
    "upper wall": "upper wall, j = J on a blade station (psi = 1)",
    "periodic (constraint side)": "periodic line, j = 0 (constraint psi[i,0] = psi[i,J] - 1)",
    "periodic (PDE side)": "periodic line, j = J (the periodic PDE row)",
    "inlet boundary": "far inlet, i = 0 (psi_m = 0)",
    "outlet boundary": "far outlet, i = n_m - 1 (psi_m = 0)",
}


def style(ax, title: str, xlabel: str = "x = m [c]", ylabel: str = "y [c]", equal: bool = True) -> None:
    workflow.style(ax, title, xlabel, ylabel, equal=equal)


def outline(case: Case) -> np.ndarray:
    """The reference blade's closed outline, densely sampled on the curves."""
    su, pr = case.metal.surfaces
    return np.vstack((su.samples[::-1], pr.samples[1:]))


def draw_blades(ax, case: Case, ks=(-1, 0, 1, 2), edge: bool = True) -> None:
    loop = outline(case)
    su, pr = case.metal.surfaces
    for k in ks:
        off = k * case.s
        ax.fill(*(loop + off).T, color=METAL_FILL, linewidth=0, zorder=1)
        if edge:
            ax.plot(*(su.samples + off).T, color=C_SUCTION, linewidth=1.1, zorder=2)
            ax.plot(*(pr.samples + off).T, color=C_PRESSURE, linewidth=1.1, zorder=2)


def draw_node(ax, case: Case, i: int, j: int, shift: int = 0, label: str | None = None, z: float = 7,
              scale: float = 1.0, dx: float = 4, dy: float = 4, fontsize: float = 6.5, ha: str = "left") -> np.ndarray:
    p = case.point(i, j) + shift * case.s
    cls = case.node_class(i, j)
    st = dict(CLASS_STYLE[cls])
    st["markersize"] *= scale
    ax.plot(*p, linestyle="none", zorder=z + (0.4 if cls == "periodic (PDE side)" else 0.0), **st)
    if label:
        ax.annotate(label, p, xytext=(dx, dy), textcoords="offset points", fontsize=fontsize, color=INK, ha=ha,
                    zorder=z + 1, bbox=dict(facecolor=SURFACE, edgecolor="none", alpha=0.75, pad=0.6))
    return p


def straddles(case: Case, a: int, b: int) -> bool:
    """Whether an m-relation between stations a and b crosses a transition interval."""
    lo, hi = min(a, b), max(a, b)
    return (lo <= case.f - 1 < hi) or (lo <= case.l < hi)


def draw_check(ax, chk: LinkCheck, shift_vec, across: bool, used: bool = True, lw: float | None = None,
               z: float = 5, emphasize: bool = True) -> None:
    """A link coloured by its exact metal test: green fluid, red metal (solid across a transition
    interval, dashed inside one region), its metal pieces overdrawn dark."""
    p, q = chk.p + shift_vec, chk.q + shift_vec
    bad = chk.status == METAL
    color = C_METAL if bad else C_FLUID
    width = lw if lw is not None else (2.0 if across else 1.3)
    ls = "-" if (across or not bad) else (0, (4, 2))
    if not used:
        ls, width = (0, (1, 1.6)), width * 0.85
    ax.plot([p[0], q[0]], [p[1], q[1]], color=color, linewidth=width, linestyle=ls, zorder=z,
            solid_capstyle="round", alpha=1.0 if used else 0.75)
    if not emphasize:
        return
    for a, b in chk.intervals:
        a, b = np.asarray(a) + shift_vec, np.asarray(b) + shift_vec
        ax.plot([a[0], b[0]], [a[1], b[1]], color=C_METAL_DARK, linewidth=width + 2.4, zorder=z + 0.2,
                solid_capstyle="butt")


@dataclass
class Frame:
    """A zoom around the reference blade's nose (LE) or tail (TE): rows 0 .. rows of the passage
    above it at their own place, rows J - rows .. J one pitch down (the same rows, periodic image:
    the passage below the reference blade, psi - 1), so both walls of one station sit on one blade."""

    stations: list
    low: list
    high: list
    xlim: tuple
    ylim: tuple
    i_periodic: int
    i_blade: int
    i_blade2: int

    def shift(self, j: int) -> int:
        return -1 if j in self.high else 0


def make_frame(case: Case, end: str, rows: int = 1) -> Frame:
    J, g = case.J, case.grid
    if end == "le":
        stations = list(range(case.f - 3, case.f + 3))
        i_per, i_bl, i_bl2 = case.f - 1, case.f, case.f + 1
    else:
        stations = list(range(case.l - 2, case.l + 4))
        i_per, i_bl, i_bl2 = case.l + 1, case.l, case.l - 1
    low, high = list(range(0, rows + 1)), list(range(J - rows, J + 1))
    pts = [case.point(i, j) for i in stations for j in low] + [case.point(i, j) - case.s for i in stations for j in high]
    pts = np.array(pts)
    gap = case.grid.deta * case.grid.pitch
    xlim = (g.m[stations[0]] - 0.45 * g.dm, g.m[stations[-1]] + 0.45 * g.dm)
    ylim = (pts[:, 1].min() - 0.45 * gap, pts[:, 1].max() + 0.55 * gap)
    return Frame(stations, low, high, xlim, ylim, i_per, i_bl, i_bl2)


def frame_base(ax, case: Case, fr: Frame, nodes: bool = True, labels: bool = False, station_labels: bool = True) -> None:
    g = case.grid
    draw_blades(ax, case, ks=(-1, 0, 1))
    a, b = sorted((g.m[fr.i_periodic], g.m[fr.i_blade]))
    ax.axvspan(a, b, color=TRANSITION_FILL, linewidth=0, zorder=0.3)
    for i in fr.stations:
        ax.axvline(g.m[i], color=STATION, linewidth=0.6, linestyle=(0, (2, 3)), zorder=0.8)
        if station_labels:
            ax.text(g.m[i], fr.ylim[1], f"i={i}", rotation=90, ha="right", va="top", fontsize=7, color=INK2, zorder=9)
    if nodes:
        for i in fr.stations:
            for j in fr.low + fr.high:
                draw_node(ax, case, i, j, fr.shift(j), label=f"({i},{j})" if labels else None)
    ax.plot(*case.cascade.active.leading_edge_point if fr.i_blade == case.f else case.cascade.active.trailing_edge_point,
            marker="*", markersize=9, color=INK, zorder=10, linestyle="none")
    ax.set_xlim(*fr.xlim)
    ax.set_ylim(*fr.ylim)


def caption(ax, text: str, y: float = -0.07) -> None:
    ax.text(0.0, y, text, transform=ax.transAxes, ha="left", va="top", fontsize=7.4, color=INK, linespacing=1.3,
            family="DejaVu Sans")


def fmt_check(chk: LinkCheck | None) -> str:
    if chk is None:
        return "n/a"
    if chk.status == FLUID:
        return f"fluid ({chk.length:.4f} c)"
    return f"METAL {chk.metal:.2e} c of {chk.length:.2e} c ({100 * chk.metal / chk.length:.0f} %)"


def legend_handles(extra: tuple = ()) -> list:
    h = [plt.Line2D([], [], linestyle="none", label=CLASS_LABEL[c], **CLASS_STYLE[c]) for c in
         ("interior", "lower wall", "upper wall", "periodic (constraint side)", "periodic (PDE side)")]
    h += [plt.Line2D([], [], color=C_FLUID, linewidth=2.0, label="link entirely in fluid (exact test)"),
          plt.Line2D([], [], color=C_METAL, linewidth=2.0, label="link through metal, across a transition interval"),
          plt.Line2D([], [], color=C_METAL, linewidth=1.3, linestyle=(0, (4, 2)),
                     label="straight chord through metal inside one region (the eta line itself follows the wall)"),
          plt.Line2D([], [], color=C_METAL_DARK, linewidth=4.0, label="the metal part of a link"),
          plt.Line2D([], [], color=C_MISSING, marker="X", markersize=8, linestyle=(0, (2, 1.5)),
                     label="missing neighbour: beyond a wall at a blade station (no node, no continuation)"),
          plt.Line2D([], [], color=STATION, linewidth=0.8, linestyle=(0, (2, 3)), label="station line x = m[i]"),
          plt.Rectangle((0, 0), 1, 1, color=TRANSITION_FILL, label="transition interval (holds the nose / tail)"),
          plt.Rectangle((0, 0), 1, 1, color=METAL_FILL, label="blade metal (C2 curves)"),
          plt.Line2D([], [], marker="*", color=INK, markersize=9, linestyle="none", label="exact LE / TE point (landmark)")]
    return h + list(extra)


# =========================================================================================
# Figure A: global overview
# =========================================================================================

def station_strip(ax, case: Case, end: str) -> None:
    """Stations around one transition over the whole passage height (x stretched)."""
    g, J = case.grid, case.J
    if end == "le":
        stations = list(range(case.f - 3, case.f + 4))
        named = {case.f - 1: ("last inlet", C_PERIODIC), case.f: ("first blade", INK)}
    else:
        stations = list(range(case.l - 3, case.l + 4))
        named = {case.l: ("last blade", INK), case.l + 1: ("first exit", C_EXIT)}
    draw_blades(ax, case, ks=(0, 1), edge=True)
    y_lo = min(case.point(i, 0)[1] for i in stations) - 0.06
    y_hi = max(case.point(i, J)[1] for i in stations) + 0.06
    for i in stations:
        x = g.m[i]
        ax.plot([x, x], [y_lo, y_hi], color=STATION, linewidth=0.6, linestyle=(0, (2, 3)), zorder=0.8)
        blade = case.f <= i <= case.l
        color = named.get(i, (None, INK if blade else (C_PERIODIC if i < case.f else C_EXIT)))[1]
        ax.plot(*g.nodes[i].T, color=color, linewidth=2.2 if i in named else 1.0, zorder=3)
        ax.plot(*g.nodes[i].T, linestyle="none", marker="o", markersize=1.6, color=INK2, zorder=4)
        for k in (0, 1):
            cut = case.metal.cut(x, k)
            if cut is not None:
                ax.plot([x, x], list(cut), color=C_METAL, linewidth=3.0, zorder=5, solid_capstyle="butt")
        if i in named:
            ax.text(x, y_hi, f" i = {i}\n {named[i][0]}", rotation=90, ha="center", va="top", fontsize=7.5,
                    color=named[i][1], bbox=dict(facecolor=SURFACE, edgecolor="none", alpha=0.8, pad=1))
    for j in range(0, case.n_eta, 4):
        row = np.array([case.point(i, j) for i in stations])
        ax.plot(*row.T, color=STATION, linewidth=0.5, zorder=2)
    for j in (0, J):
        row = np.array([case.point(i, j) for i in stations])
        ax.plot(*row.T, color=INK2, linewidth=0.7, linestyle=(0, (1, 1.5)), zorder=2)
    ax.set_xlim(g.m[stations[0]] - 0.6 * g.dm, g.m[stations[-1]] + 0.6 * g.dm)
    ax.set_ylim(y_lo, y_hi)
    which = "LE" if end == "le" else "TE"
    style(ax, f"{which} stations over the passage height (x stretched)", equal=False)
    ax.text(0.0, -0.075, "Red: the part of a station line x = m[i] inside metal (the cut).\nThe grid's eta line at that "
            f"station runs wall to wall\n(thick line) and never uses it. Dots: the {case.n_eta} nodes.", transform=ax.transAxes,
            fontsize=7.5, color=C_METAL, va="top")


def figure_global(case: Case) -> plt.Figure:
    g, J, a = case.grid, case.J, case.cascade.active
    fig = plt.figure(figsize=(19, 10), facecolor=SURFACE)
    fig.suptitle(f"A. Global overview: {case.cascade.metadata['designation']} ({case.spec.surface_continuity}), stagger "
                 f"{case.spec.stagger_deg:g} deg, solidity {case.spec.solidity:g}; grid {case.n_m} x {case.n_eta} "
                 f"(n_passage {case.n_passage}), dm = {g.dm:.5f} c, d eta = {g.deta:.4f}", x=0.02, ha="left",
                 fontsize=12.5, color=INK)
    gs = fig.add_gridspec(1, 3, width_ratios=(2.5, 1.0, 1.0), left=0.04, right=0.99, top=0.92, bottom=0.11, wspace=0.18)
    ax = fig.add_subplot(gs[0])
    draw_blades(ax, case, ks=(-1, 0, 1, 2))
    for i in range(case.n_m):
        ax.plot(*g.nodes[i].T, color=STATION, linewidth=0.25, zorder=2)
    for j in range(0, case.n_eta, 5):
        ax.plot(*g.nodes[:, j].T, color=STATION, linewidth=0.35, zorder=2)
    f, l = case.f, case.l
    inlet, walls, exit_ = slice(0, f), slice(f, l + 1), slice(l + 1, None)
    ax.plot(*g.nodes[inlet, 0].T, color=C_PERIODIC, linewidth=1.8, zorder=4, label="inlet periodic lines (j = 0, J)")
    ax.plot(*g.nodes[inlet, J].T, color=C_PERIODIC, linewidth=1.8, linestyle=(0, (6, 2)), zorder=4)
    ax.plot(*g.nodes[exit_, 0].T, color=C_EXIT, linewidth=1.8, zorder=4, label="exit periodic lines (j = 0, J)")
    ax.plot(*g.nodes[exit_, J].T, color=C_EXIT, linewidth=1.8, linestyle=(0, (6, 2)), zorder=4)
    ax.plot(*g.nodes[walls, 0].T, color=C_SUCTION, linewidth=1.8, zorder=4, label="lower wall j = 0 (reference blade)")
    ax.plot(*g.nodes[walls, J].T, color=C_PRESSURE, linewidth=1.8, zorder=4, label="upper wall j = J (+pitch blade)")
    ax.plot(*g.nodes[0].T, color=C_FAR, linewidth=3, zorder=4, label="far inlet i = 0 / far outlet i = n_m - 1")
    ax.plot(*g.nodes[-1].T, color=C_FAR, linewidth=3, zorder=4)
    marks = ((f - 1, "last inlet", C_PERIODIC, -1), (f, "first blade = first_blade_i", INK, 1),
             (l, "last blade = last_blade_i", INK, -1), (l + 1, "first exit", C_EXIT, 1))
    for i, name, color, side in marks:
        ax.plot(*g.nodes[i].T, color=color, linewidth=2.6, zorder=6)
        top = g.nodes[i, J]
        ax.annotate(f"i = {i}: {name}\nm - m_{'le' if i <= f else 'te'} = "
                    f"{(g.m[i] - (g.m_le if i <= f else g.m_te)) / g.chord:+.5f} c", top,
                    xytext=(-12 if side < 0 else 12, 34 if i in (f - 1, l) else 64), textcoords="offset points",
                    ha="right" if side < 0 else "left", fontsize=8, color=color,
                    arrowprops=dict(arrowstyle="-", color=color, linewidth=0.8),
                    bbox=dict(facecolor=SURFACE, edgecolor="none", alpha=0.9, pad=1.5), zorder=9)
    ax.plot(*a.leading_edge_point, marker="*", markersize=10, color=INK, zorder=8, linestyle="none", label="exact LE / TE")
    ax.plot(*a.trailing_edge_point, marker="*", markersize=10, color=INK, zorder=8, linestyle="none")
    ax.set_xlim(g.m_in - 0.08, g.m_out + 0.08)
    style(ax, f"Passage in (m, y): blade stations {f}..{l} (their lines cut the blade), {f} inlet and "
              f"{case.n_m - l - 1} exit periodic stations")
    ax.legend(loc="lower right", fontsize=7.5, frameon=False, labelcolor=INK2)
    txt = (f"first_blade_i = {f}, last_blade_i = {l}\n"
           f"last inlet station  i = {f - 1}: m - m_le = {(g.m[f - 1] - g.m_le) / g.chord:+.5f} c\n"
           f"first blade station i = {f}: m - m_le = {g.actual_le_offset / g.chord:+.5f} c\n"
           f"last blade station  i = {l}: m - m_te = {g.actual_te_offset / g.chord:+.5f} c\n"
           f"first exit station  i = {l + 1}: m - m_te = {(g.m[l + 1] - g.m_te) / g.chord:+.5f} c\n"
           f"LE transition interval: {f - 1} -> {f}; TE: {l} -> {l + 1}\n"
           f"inlet {workflow.INLET_ANGLE_DEG:g} deg, exit {'kappa_te' if case.exit_angle is None else case.exit_angle}, "
           f"extensions {case.settings.upstream_extension_chords:g} / {case.settings.downstream_extension_chords:g} c")
    ax.text(0.01, 0.99, txt, transform=ax.transAxes, va="top", ha="left", fontsize=8, family="monospace", color=INK,
            bbox=dict(facecolor=SURFACE, edgecolor=STATION, pad=4))
    station_strip(fig.add_subplot(gs[1]), case, "le")
    station_strip(fig.add_subplot(gs[2]), case, "te")
    return fig


# =========================================================================================
# Figures B and C: the LE / TE zooms
# =========================================================================================

def _names(case: Case, end: str) -> dict:
    if end == "le":
        return dict(edge="nose", which="LE", periodic="last inlet", blade="first blade", copy="+pitch blade's nose",
                    kind=dz.PERIODIC_LE_TRANSITION, side_word="east", missing="NE", partner="NW")
    return dict(edge="tail", which="TE", periodic="first exit", blade="last blade", copy="+pitch blade's tail",
                kind=dz.PERIODIC_TE_TRANSITION, side_word="west", missing="NW", partner="NE")


def _is_wall(case: Case, i: int, j: int) -> bool:
    return case.node_class(i, j) in ("lower wall", "upper wall")


def panel_stations(ax, case: Case, fr: Frame, end: str) -> str:
    """(a) station lines against the blade: the line cuts it, the eta line does not."""
    n, g, J = _names(case, end), case.grid, case.J
    frame_base(ax, case, fr)
    for i in fr.stations:
        x = g.m[i]
        blade = case.f <= i <= case.l
        color = INK if blade else (C_PERIODIC if i < case.f else C_EXIT)
        ax.plot([x, x], [case.point(i, 0)[1], fr.ylim[1]], color=color, linewidth=1.5, zorder=3)
        ax.plot([x, x], [(case.point(i, J) - case.s)[1], fr.ylim[0]], color=color, linewidth=1.5, zorder=3)
        cut = case.metal.cut(x, 0)
        if cut is not None:
            ax.plot([x, x], list(cut), color=C_METAL, linewidth=3.4, zorder=6, solid_capstyle="butt")
    ib, ip = fr.i_blade, fr.i_periodic
    cut = case.metal.cut(g.m[ib], 0)
    mid = np.array([g.m[ib], 0.5 * (cut[0] + cut[1])])
    side = 1 if end == "le" else -1
    ax.annotate(f"x = m[{ib}]\ncuts the blade:\n{cut[1] - cut[0]:.2e} c", mid, xytext=(side * 26, 0),
                textcoords="offset points", ha="left" if side > 0 else "right", va="center", fontsize=7,
                color=C_METAL, arrowprops=dict(arrowstyle="->", color=C_METAL, linewidth=0.8), zorder=11,
                bbox=dict(facecolor=SURFACE, edgecolor="none", alpha=0.9, pad=1))
    draw_node(ax, case, ib, 0, 0, label=f"({ib},0)", dx=-side * 5, dy=6, ha="right" if side > 0 else "left")
    draw_node(ax, case, ib, J, -1, label=f"({ib},{J}) - s", dx=-side * 5, dy=-12, ha="right" if side > 0 else "left")
    draw_node(ax, case, ip, 0, 0, label=f"({ip},0) =\n({ip},{J}) - s", dx=-side * 6, dy=-22,
              ha="right" if side > 0 else "left")
    style(ax, f"(a) Station lines vs the blade")
    return (f"Every station line x = m[i] is a straight vertical line (x = m). The {n['periodic']} station i = {ip} "
            f"misses the blade: its eta line runs through the periodic point ({ip},0) = ({ip},{J}) - s and is fluid "
            f"end to end.\nThe {n['blade']} station i = {ib} cuts it (red: {cut[1] - cut[0]:.3e} c of metal, y = "
            f"{cut[0]:.4f} .. {cut[1]:.4f} c). The grid uses only the fluid pieces: eta = 0 starts on top of the cut, "
            f"node ({ib},0); eta = 1 ends below the cut of the +pitch blade, node ({ib},{J}), drawn here one pitch "
            f"down.\nCutting the blade is what MAKES it a blade station; it is not a defect.\n"
            f"Rows j = {fr.low[0]}..{fr.low[-1]} at their place, rows j = {fr.high[0]}..{fr.high[-1]} one pitch down "
            f"(their periodic image), so both walls of a station sit on the one blade drawn.")


def panel_same_eta(ax, case: Case, fr: Frame, end: str) -> str:
    """(b) the same-eta links psi_m and psi_mm read."""
    n, J = _names(case, end), case.J
    frame_base(ax, case, fr)
    notes = []
    for i0, i1 in zip(fr.stations[:-1], fr.stations[1:]):
        for j in fr.low + fr.high:
            chk = case.link(case.point(i0, j), case.point(i1, j))
            used = case.has_pde_row(i0, j) or case.has_pde_row(i1, j)
            wall_chord = _is_wall(case, i0, j) and _is_wall(case, i1, j)
            draw_check(ax, chk, fr.shift(j) * case.s, straddles(case, i0, i1), used, emphasize=not wall_chord)
    i0, i1 = sorted((fr.i_periodic, fr.i_blade))
    for j, who in ((J, "PDE side: read by the periodic PDE row"), (0, "constraint side: no PDE row reads it today")):
        chk = case.link(case.point(i0, j), case.point(i1, j))
        p = 0.5 * (chk.p + chk.q) + fr.shift(j) * case.s
        tag = f"({i0},{j})-({i1},{j})"
        ax.annotate(f"{tag}\n{'METAL ' + format(chk.metal, '.2e') + ' c' if chk.status == METAL else 'fluid'}", p,
                    xytext=(0, 24 if j == 0 else -30), textcoords="offset points", ha="center", fontsize=6.8,
                    color=C_METAL if chk.status == METAL else C_FLUID, zorder=11,
                    arrowprops=dict(arrowstyle="-", color=INK2, linewidth=0.5),
                    bbox=dict(facecolor=SURFACE, edgecolor="none", alpha=0.9, pad=0.8))
        notes.append(f"  {tag} [{who}]: {fmt_check(chk)}")
    inside = [(a, j) for a, b in zip(fr.stations[:-1], fr.stations[1:]) for j in range(1, J)
              if straddles(case, a, b) and case.link(case.point(a, j), case.point(b, j)).status == METAL]
    style(ax, "(b) Same-eta links (psi_m, psi_mm)")
    return ("psi_m = (psi[i+1,j] - psi[i-1,j]) / 2dm and psi_mm = (psi[i+1,j] - 2 psi[i,j] + psi[i-1,j]) / dm^2 read "
            "W and E along eta = const, so the links W-P and P-E must be fluid connections. Solid: read by a PDE row; "
            "dotted: read by none (wall or constraint-side node).\nAcross the transition interval (yellow) the boundary "
            "row changes from periodic line to wall:\n" + "\n".join(notes) +
            f"\nInterior rows j = 1..{J - 1} across the interval through metal: "
            f"{', '.join(f'({a},{j})' for a, j in inside) if inside else 'none'}.\n"
            "Thin dashed red along the walls: wall-to-wall chords. There the eta = 0 (eta = 1) coordinate line IS the "
            "wall, and the nodes carry psi = 0 (1), so a chord through metal means nothing.")


def _would_be_pair(case: Case, k: int, j: int) -> LinkCheck:
    """The eta pair (k, j - 1) -> (k, j + 1) the central mixed term forms at station k for the
    periodic row j = J, through the continuation (k, 1) + s (valid on a periodic station only)."""
    return case.link(case.point(k, j - 1), case.point(k, 1) + case.s)


def panel_mixed_periodic(ax, case: Case, fr: Frame, end: str) -> str:
    """(c) the mixed derivative at the periodic PDE row next to the transition."""
    n, g, J = _names(case, end), case.grid, case.J
    frame_base(ax, case, fr)
    i = fr.i_periodic
    P = (i, J)
    sv = -case.s
    st = case.stencil(*P)
    for offset, c in dz.MIXED_M_DIFFERENCE[n["kind"]]:
        k = i + offset
        pair = _would_be_pair(case, k, J)
        ax.plot([pair.p[0], pair.q[0]], [pair.p[1] + sv[1], pair.q[1] + sv[1]], color=C_CODE, linewidth=9,
                alpha=0.18, zorder=3, solid_capstyle="butt")
        ax.annotate(f"q{k} {c:+g}", (g.m[k], pair.p[1] + sv[1]), xytext=(0, -16), textcoords="offset points",
                    ha="center", va="top", fontsize=6.6, color=C_CODE, zorder=11)
    notes = []
    for name in ("W", "E"):
        nb = st[name]
        draw_check(ax, nb.check, sv, straddles(case, i, nb.node[0]), lw=2.6, z=6)
        notes.append(f"  {name} = {nb.node}: {fmt_check(nb.check)}")
    for name in ("SW", "SE", "NW", "NE"):
        nb = st[name]
        if nb.exists:
            draw_check(ax, nb.check, sv, straddles(case, i, nb.node[0]), lw=1.4, z=6)
            q = nb.point + sv
            ax.plot(*q, marker="o", markersize=11, markerfacecolor="none", markeredgecolor=INK, markeredgewidth=0.9,
                    zorder=8)
            ax.annotate(name, q, xytext=(5, 4 if name[0] == "N" else -11), textcoords="offset points", fontsize=7,
                        color=INK, zorder=11, bbox=dict(facecolor=SURFACE, edgecolor="none", alpha=0.8, pad=0.5))
            notes.append(f"  {name} = {nb.node}" + (f" = continuation ({nb.real[0]},{nb.real[1]}) + s" if nb.shift else "")
                         + f": {fmt_check(nb.check)}")
        else:
            q = nb.ghost_point + sv
            pp = case.point(*P) + sv
            ax.plot([pp[0], q[0]], [pp[1], q[1]], color=C_MISSING, linewidth=1.4, linestyle=(0, (2, 1.5)), zorder=6)
            ax.plot(*q, marker="X", markersize=11, color=C_MISSING, markeredgecolor=INK, markeredgewidth=0.5, zorder=9)
            draw_check(ax, nb.ghost_check, sv, True, lw=2.2, z=7)
            ax.annotate(f"{name} MISSING", q, xytext=(7 if end == "le" else -7, 6), textcoords="offset points",
                        ha="left" if end == "le" else "right", fontsize=7, color=C_MISSING, zorder=11,
                        bbox=dict(facecolor=SURFACE, edgecolor="none", alpha=0.9, pad=0.6))
            notes.append(f"  {name} = {nb.node}: MISSING. Station {nb.node[0]} is a blade station: above its upper "
                         f"wall is the {n['copy']}. The continuation would take ({nb.node[0]},1) + s (purple X), on "
                         f"the far side of the cut, and q{nb.node[0]} = (psi[{nb.node[0]},{J + 1}] - "
                         f"psi[{nb.node[0]},{J - 1}]) / 2deta would straddle {nb.ghost_check.metal:.3e} c of metal "
                         f"(red vertical). The straight P -> X link itself is {fmt_check(nb.ghost_link)}: the defect "
                         f"is the topology, not that line.")
    other = i - 1 if end == "le" else i + 1
    draw_check(ax, _would_be_pair(case, other, J), sv, False, lw=1.4, z=5)
    draw_node(ax, case, *P, -1, label="P", scale=1.5, dx=-8 if end == "le" else 8, dy=-12,
              ha="right" if end == "le" else "left", z=10)
    style(ax, f"(c) psi_meta at P = ({i},{J})")
    return (f"P = ({i},{J}) is the periodic PDE row ({i},{J}) = ({i},0) + s, drawn at ({i},0). Central psi_meta = "
            f"(q[{i + 1}] - q[{i - 1}]) / 2dm, q[k] = (psi[k,{J + 1}] - psi[k,{J - 1}]) / 2deta: the four corners.\n"
            + "\n".join(notes) +
            f"\n  q{other} at the periodic station {other}: crosses the seam at ({other},0) = ({other},{J}) - s, "
            f"fluid; the continuation is valid there.\nBlue: what the code writes today ({n['kind']}): psi_meta "
            f"one-sided in m, q at stations {', '.join(str(i + o) for o, _ in dz.MIXED_M_DIFFERENCE[n['kind']])}, all "
            f"periodic, weights {', '.join(f'{c:+g}' for _, c in dz.MIXED_M_DIFFERENCE[n['kind']])} / 2dm. Along "
            f"j = {J} (psi_m, psi_mm) the code's row reads stations "
            f"{', '.join(str(k) for k in code_reads_on_row(case, i))}"
            f"{' (central)' if code_reads_on_row(case, i) == sorted((i - 1, i + 1)) else ' (one-sided, away from the blade)'}.")


def panel_mixed_rows(ax, case: Case, fr: Frame, end: str, nodes: list, title: str) -> str:
    """(d), (e), (f) the interior rows next to the boundary rows."""
    frame_base(ax, case, fr)
    notes = []
    for P in nodes:
        i, j = P
        sv = fr.shift(j) * case.s
        st = case.stencil(i, j)
        for name in ("W", "E", "SW", "SE", "NW", "NE"):
            nb = st[name]
            if nb.exists:
                draw_check(ax, nb.check, sv, straddles(case, i, nb.node[0]), lw=1.9 if name in "WE" else 1.2, z=6)
                if nb.real is not None and nb.shift == 0:
                    draw_node(ax, case, *nb.real, fr.shift(j), z=7.5)
            else:
                ax.plot(*(nb.ghost_point + sv), marker="X", markersize=9, color=C_MISSING, zorder=9)
        for a, b in (("SW", "NE"), ("NW", "SE")):
            chk = case.diagonal(i, j, a, b)
            if chk is not None:
                draw_check(ax, chk, sv, False, lw=0.7, z=5)
        draw_node(ax, case, i, j, fr.shift(j), label=f"P ({i},{j})", scale=1.8, dx=5, dy=5 if j < case.J // 2 else -12,
                  z=10)
        bad = [f"{name} {st[name].node} {st[name].status}"
               + (f" {st[name].check.metal:.2e} c of {st[name].check.length:.2e} c" if st[name].status == METAL else "")
               for name in ("W", "E", "SW", "SE", "NW", "NE") if st[name].status != FLUID]
        diag = [f"{a}-{b} {case.diagonal(i, j, a, b).metal:.2e} c" for a, b in (("SW", "NE"), ("NW", "SE"))
                if case.diagonal(i, j, a, b) is not None and case.diagonal(i, j, a, b).status == METAL]
        notes.append(f"  ({i},{j}) {case.node_class(i, j)}: " + ("W, E and all four corners fluid" if not bad else
                                                               "; ".join(bad))
                     + ("" if not diag else "; straight full diagonal " + ", ".join(diag)))
    style(ax, title)
    return ("Each P: W, E (thick) and the four corner relations P -> corner (thin); hairlines: the straight full "
            "diagonals SW-NE and NW-SE. Every corner exists (no wall lies between P and a corner in index space).\n"
            + "\n".join(notes) +
            "\nA metal crossing between two blade stations is a chord of a curved coordinate line: inside one region "
            "the map y = (1 - eta) y0(m) + eta y1(m) follows the smooth walls, so the connection itself is fluid; "
            "only across a transition interval is the straight link the only connection there is.")


def panel_mid(ax, case: Case, end: str) -> str:
    """(g) mid-passage: the full nine-point stencil, far from any metal."""
    g = case.grid
    fr = make_frame(case, end)
    jm = case.J // 2
    Ps = [(fr.i_periodic, jm), (fr.i_blade, jm)]
    ks = sorted({i + d for i, _ in Ps for d in (-1, 0, 1)})
    pts = np.array([case.point(k, j) for k in ks for j in (jm - 1, jm, jm + 1)])
    pad = 0.6 * g.dm
    a, b = sorted((g.m[fr.i_periodic], g.m[fr.i_blade]))
    ax.axvspan(a, b, color=TRANSITION_FILL, linewidth=0, zorder=0.3)
    notes = []
    for i, j in Ps:
        st = case.stencil(i, j)
        for name, nb in st.items():
            draw_check(ax, nb.check, 0.0, straddles(case, i, nb.node[0]), lw=1.6 if name in "WE" else 1.0)
        bad = [k for k, v in st.items() if v.status != FLUID]
        near = min(s.tree.query(case.point(i, j) - k * case.s)[0] for s in case.metal.surfaces for k in COPIES)
        notes.append(f"  ({i},{j}): all 8 neighbours {'fluid' if not bad else 'NOT fluid: ' + ', '.join(bad)}; "
                     f"nearest metal {near:.3f} c")
    for k in ks:
        ax.axvline(g.m[k], color=STATION, linewidth=0.6, linestyle=(0, (2, 3)), zorder=0.8)
        ax.text(g.m[k], pts[:, 1].max() + 0.6 * pad, f"i={k}", rotation=90, ha="right", va="top", fontsize=7,
                color=INK2)
        for j in (jm - 1, jm, jm + 1):
            draw_node(ax, case, k, j, label=f"({k},{j})" if (k, j) in Ps else None, scale=1.6 if (k, j) in Ps else 1.0)
    ax.set_xlim(pts[:, 0].min() - pad, pts[:, 0].max() + pad)
    ax.set_ylim(pts[:, 1].min() - pad, pts[:, 1].max() + pad)
    style(ax, f"(g) Mid-passage, j = {jm}")
    ib = fr.i_blade
    cut = case.metal.cut(g.m[ib], 0)
    return (f"The full nine-point stencil at j = {jm}, i = {fr.i_periodic} and {ib}. The station line x = m[{ib}] cuts "
            f"the blade at y = {cut[0]:.4f} .. {cut[1]:.4f} c (and one pitch up), about half a pitch from these nodes, "
            f"and nothing here notices: every same-eta and corner relation is fluid.\n" + "\n".join(notes))


def figure_zoom(case: Case, end: str) -> plt.Figure:
    import textwrap
    n = _names(case, end)
    fr = make_frame(case, end)
    J = case.J
    ip, ib, ib2 = fr.i_periodic, fr.i_blade, fr.i_blade2
    aspect = (fr.xlim[1] - fr.xlim[0]) / (fr.ylim[1] - fr.ylim[0])
    width, n_panels = 25.0, 7
    slot = width * 0.95 / n_panels
    h_panel = float(np.clip(slot / aspect, 3.8, 11.0))
    h_caption, h_legend, h_title = 3.5, 1.0, 0.7
    height = h_panel + h_caption + h_legend + h_title
    fig = plt.figure(figsize=(width, height), facecolor=SURFACE)
    fig.suptitle(f"{'B' if end == 'le' else 'C'}. {n['which']} zoom, physical plane (equal axes), n_passage "
                 f"{case.n_passage}, n_eta {case.n_eta}: {n['periodic']} station {ip}, {n['blade']} station {ib}. "
                 f"Every colour is an exact link-metal test on the C2 curves.", x=0.01, ha="left", fontsize=13,
                 color=INK, y=1.0 - 0.25 / height)
    gs = fig.add_gridspec(3, n_panels, height_ratios=(h_panel, h_caption, h_legend), left=0.025, right=0.995,
                          top=1.0 - h_title / height, bottom=0.005, wspace=0.28, hspace=0.12)
    panels = [
        lambda ax: panel_stations(ax, case, fr, end),
        lambda ax: panel_same_eta(ax, case, fr, end),
        lambda ax: panel_mixed_periodic(ax, case, fr, end),
        lambda ax: panel_mixed_rows(ax, case, fr, end, [(ip, 1), (ip, J - 1)],
                                    f"(d) Rows by the walls, i = {ip}"),
        lambda ax: panel_mixed_rows(ax, case, fr, end, [(ib, 1), (ib, J - 1)], f"(e) Rows by the walls, i = {ib}"),
        lambda ax: panel_mixed_rows(ax, case, fr, end, [(ib2, 1), (ib2, J - 1)], f"(f) Rows by the walls, i = {ib2}"),
        lambda ax: panel_mid(ax, case, end),
    ]
    for k, draw in enumerate(panels):
        ax = fig.add_subplot(gs[0, k])
        text = draw(ax)
        cap = fig.add_subplot(gs[1, k])
        cap.axis("off")
        wrapped = "\n".join(textwrap.fill(par, 56, subsequent_indent="  " if par.startswith("  ") else "")
                            for par in text.split("\n"))
        cap.text(0.0, 1.0, wrapped, transform=cap.transAxes, ha="left", va="top", fontsize=7.0, color=INK,
                 linespacing=1.25)
    leg = fig.add_subplot(gs[2, :])
    leg.axis("off")
    leg.legend(handles=legend_handles(), loc="center", ncol=5, fontsize=8, frameon=False, labelcolor=INK2)
    return fig


# =========================================================================================
# Figure D: index space, mapped back to the physical plane
# =========================================================================================

CLASS_CODES = ("interior", "lower wall", "upper wall", "periodic (constraint side)", "periodic (PDE side)",
               "inlet boundary", "outlet boundary")
CLASS_COLORS = ("#e9e7e2", C_SUCTION, C_PRESSURE, "#f0c48a", C_PERIODIC, "#9a9a9a", "#6b6b6b")


def class_map(ax, case: Case) -> None:
    from matplotlib.colors import ListedColormap
    codes = np.array([[CLASS_CODES.index(case.node_class(i, j)) for i in range(case.n_m)] for j in range(case.n_eta)])
    ax.imshow(codes, origin="lower", aspect="auto", interpolation="nearest", cmap=ListedColormap(CLASS_COLORS),
              vmin=-0.5, vmax=len(CLASS_CODES) - 0.5, extent=(-0.5, case.n_m - 0.5, -0.5, case.n_eta - 0.5))
    for i, name in ((case.f, "first_blade_i"), (case.l, "last_blade_i")):
        ax.axvline(i - 0.5 if i == case.f else i + 0.5, color=INK, linewidth=1.0)
        ax.text(i + (2 if i == case.f else -2), 0.5 * case.n_eta, f"{name} = {i}", ha="left" if i == case.f else "right",
                va="center", fontsize=8.5, color=INK, bbox=dict(facecolor=SURFACE, edgecolor="none", alpha=0.85, pad=1))
    for i, text in ((0, "i = 0: far inlet"), (case.n_m - 1, f"i = {case.n_m - 1}: far outlet")):
        ax.annotate(text, (i, 0.75 * case.n_eta), xytext=(18 if i == 0 else -18, 0), textcoords="offset points",
                    ha="left" if i == 0 else "right", va="center", fontsize=8, color=INK,
                    arrowprops=dict(arrowstyle="->", color=INK, linewidth=0.7),
                    bbox=dict(facecolor=SURFACE, edgecolor="none", alpha=0.85, pad=1))
    ax.set_xlabel("i (along m)", fontsize=9, color=INK2)
    ax.set_ylabel("j (along eta)", fontsize=9, color=INK2)
    ax.set_title("Node classes in index space (boundary_conditions.py masks): one passage, i x j = "
                 f"{case.n_m} x {case.n_eta}", loc="left", fontsize=10, color=INK)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, label=CLASS_LABEL[n]) for n, c in zip(CLASS_CODES, CLASS_COLORS)]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4, fontsize=7.5, frameon=False,
              labelcolor=INK2)
    ax.tick_params(labelsize=8, colors=INK2)


def _block_y(case: Case, i: int, j: int, upper_passage: bool) -> float:
    """Unrolled eta index across the seam: passage 0 rows j - J (<= 0), passage +1 rows j (>= 0);
    the two walls of a blade station sit either side of the slit."""
    if upper_passage:
        return 0.24 if (j == 0 and not case.periodic[i]) else float(j)
    return -0.24 if (j == case.J and not case.periodic[i]) else float(j - case.J)


def index_block(ax, case: Case, end: str, rows: int = 2) -> dict:
    """The stencil block around one transition, in index space, unrolled across the eta = 1 seam."""
    J = case.J
    n = _names(case, end)
    if end == "le":
        cols = list(range(case.f - 3, case.f + 3))
        P = (case.f - 1, J)
    else:
        cols = list(range(case.l - 2, case.l + 4))
        P = (case.l + 1, J)
    i = P[0]
    pos = {}
    # slit: the blade between the two walls of the blade stations
    blade_cols = [k for k in cols if not case.periodic[k]]
    x0 = (blade_cols[0] - 0.5) if end == "le" else cols[0] - 0.6
    x1 = cols[-1] + 0.6 if end == "le" else (blade_cols[-1] + 0.5)
    tip = x0 if end == "le" else x1
    xs = [tip, tip + (0.45 if end == "le" else -0.45)]
    ax.fill([x0, x1, x1, x0], [-0.12, -0.12, 0.12, 0.12], color="#7d7a73", zorder=2, linewidth=0)
    ax.fill([xs[0], xs[1], xs[1]], [0.0, -0.12, 0.12], color="#7d7a73", zorder=2, linewidth=0)
    ax.text(x1 - 0.05 if end == "le" else x0 + 0.05, 0.0, "+pitch blade (the slit)", ha="right" if end == "le" else "left",
            va="center", fontsize=6.8, color="white", zorder=3)
    for k in cols:
        for j in range(J - rows, J + 1):
            y = _block_y(case, k, j, False)
            st = dict(CLASS_STYLE[case.node_class(k, j)])
            ax.plot(k, y, linestyle="none", zorder=6, **st)
            pos[(k, j, 0)] = (k, y)
        for j in range(0, rows + 1):
            y = _block_y(case, k, j, True)
            if j == 0 and case.periodic[k]:
                ax.plot(k, y, linestyle="none", zorder=5.5, **CLASS_STYLE["periodic (constraint side)"])
            else:
                ax.plot(k, y, linestyle="none", zorder=6, **CLASS_STYLE[case.node_class(k, j)])
            pos[(k, j, 1)] = (k, y)
        ax.text(k, rows + 0.55, f"i={k}\n{'periodic' if case.periodic[k] else 'blade'}", ha="center", va="bottom",
                fontsize=7, color=C_PERIODIC if case.periodic[k] else INK)
    for j in range(J - rows, J + 1):
        ax.text(cols[0] - 0.75, j - J, f"j = {j}", ha="right", va="center", fontsize=7, color=INK2)
    for j in range(1, rows + 1):
        ax.text(cols[0] - 0.75, j, f"j = {j} of passage +1\n(= j {J + j} on periodic stations)", ha="right", va="center",
                fontsize=6.5, color=INK2)
    st = case.stencil(*P)
    Pxy = pos[(i, J, 0)]

    def arrow(to, color, ls="-", lw=1.6, z=7):
        ax.annotate("", to, Pxy, arrowprops=dict(arrowstyle="-|>", color=color, linewidth=lw, linestyle=ls,
                                                 shrinkA=6, shrinkB=6), zorder=z)

    for offset, c in dz.MIXED_M_DIFFERENCE[n["kind"]]:
        k = i + offset
        ax.plot([k, k], [-1, 1], color=C_CODE, linewidth=10, alpha=0.18, zorder=3, solid_capstyle="butt")
        ax.text(k + 0.08, -1.45, f"q{k}: {c:+g}", fontsize=6.6, color=C_CODE, ha="center")
    for name in ("W", "E"):
        nb = st[name]
        arrow(pos[(nb.node[0], nb.node[1], 0)], C_METAL if nb.status == METAL else C_FLUID, lw=2.4)
    for name in ("SW", "SE", "NW", "NE"):
        nb = st[name]
        k, jj = nb.node
        if nb.exists:
            to = pos[(k, jj - J, 1)] if jj > J else pos[(k, jj, 0)]
            arrow(to, C_METAL if nb.status == METAL else C_FLUID, ls=(0, (3, 1.5)), lw=1.3)
        else:
            to = (k, 0.0)
            arrow(to, C_MISSING, ls=(0, (2, 1.5)), lw=1.3)
            ax.plot(*to, marker="X", markersize=13, color=C_MISSING, markeredgecolor="white", markeredgewidth=0.8,
                    zorder=9)
            ax.annotate(f"{name} = ({k},{jj}): missing.\nAbove the upper wall of blade station {k}\nis the blade, "
                        f"not ({k},1) of passage +1:\nthe continuation psi[k,J+1] = psi[k,1] + 1\nholds on periodic "
                        f"stations only", (k, 0.0), xytext=(14 if end == "le" else -14, 34), textcoords="offset points",
                        ha="left" if end == "le" else "right", fontsize=7, color=C_MISSING, zorder=11,
                        arrowprops=dict(arrowstyle="-", color=C_MISSING, linewidth=0.6),
                        bbox=dict(facecolor=SURFACE, edgecolor="none", alpha=0.9, pad=1))
    ax.annotate(f"P = ({i},{J})\n= ({i},0) of passage +1", Pxy, xytext=(-10 if end == "le" else 10, -34),
                textcoords="offset points", ha="right" if end == "le" else "left", fontsize=7.4, color=INK, zorder=11,
                arrowprops=dict(arrowstyle="-", color=INK, linewidth=0.6),
                bbox=dict(facecolor=SURFACE, edgecolor="none", alpha=0.9, pad=1))
    ax.plot(*Pxy, marker="o", markersize=16, markerfacecolor="none", markeredgecolor=INK, markeredgewidth=1.2, zorder=8)
    ax.set_xlim(cols[0] - 2.4, cols[-1] + 0.8)
    ax.set_ylim(-rows - 1.0, rows + 1.4)
    ax.set_facecolor(SURFACE)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"{n['which']} transition in index space (i, j), unrolled across the eta = 1 seam: the periodic "
                 f"PDE row P = ({i},{J})", loc="left", fontsize=10, color=INK)
    ax.text(0.0, -0.02, "psi_m, psi_mm read W, E (solid arrows); psi_meta reads the four corners (dashed); blue: the "
            f"one-sided q's the code\nuses at this row ({n['kind']}). On a periodic station (i, J) of this passage and "
            "(i, 0) of the next are one node;\non a blade station they are the two walls with the blade between them, "
            "so the index lattice is cut (the slit).", transform=ax.transAxes, va="top", fontsize=7.4, color=INK)
    return dict(pos=pos, cols=cols, P=P, rows=rows)


def physical_block(ax, case: Case, end: str, rows: int = 1) -> dict:
    """The same nodes at their physical places around the +pitch blade (no shift), passage +1 rows
    at (i, j) + s."""
    J, g = case.J, case.grid
    n = _names(case, end)
    cols = list(range(case.f - 3, case.f + 2)) if end == "le" else list(range(case.l - 1, case.l + 4))
    i = case.f - 1 if end == "le" else case.l + 1
    P = (i, J)
    draw_blades(ax, case, ks=(0, 1, 2))
    a, b = sorted((g.m[i], g.m[case.f if end == 'le' else case.l]))
    ax.axvspan(a, b, color=TRANSITION_FILL, linewidth=0, zorder=0.3)
    pos = {}
    for k in cols:
        ax.axvline(g.m[k], color=STATION, linewidth=0.6, linestyle=(0, (2, 3)), zorder=0.8)
        for j in range(J - rows, J + 1):
            pos[(k, j, 0)] = draw_node(ax, case, k, j, 0)
        for j in range(0, rows + 1):
            pos[(k, j, 1)] = draw_node(ax, case, k, j, 1)
        cut = case.metal.cut(g.m[k], 1)
        if cut is not None:
            ax.plot([g.m[k]] * 2, list(cut), color=C_METAL, linewidth=3.0, zorder=6, solid_capstyle="butt")
    st = case.stencil(*P)
    for name in ("W", "E", "SW", "SE", "NW", "NE"):
        nb = st[name]
        if nb.exists:
            draw_check(ax, nb.check, 0.0, straddles(case, i, nb.node[0]), lw=2.4 if name in "WE" else 1.2, z=6)
        else:
            q = nb.ghost_point
            p0 = case.point(*P)
            ax.plot([p0[0], q[0]], [p0[1], q[1]], color=C_MISSING, linewidth=1.3, linestyle=(0, (2, 1.5)), zorder=6)
            ax.plot(*q, marker="X", markersize=11, color=C_MISSING, markeredgecolor=INK, markeredgewidth=0.5, zorder=9)
            draw_check(ax, nb.ghost_check, 0.0, True, lw=2.0, z=7)
            pos[("ghost",)] = q
    pts = np.array(list(v for v in pos.values()))
    pad = 0.5 * g.deta * g.pitch
    ax.set_xlim(g.m[cols[0]] - 0.5 * g.dm, g.m[cols[-1]] + 0.5 * g.dm)
    ax.set_ylim(pts[:, 1].min() - pad, pts[:, 1].max() + pad)
    style(ax, f"The same nodes in the physical plane, around the {n['copy'].split(chr(39))[0]}'s "
              f"{'nose' if end == 'le' else 'tail'} (equal axes)")
    ax.text(0.0, -0.06, f"Passage rows j = {J - rows}..{J} at their place, rows 0..{rows} of passage +1 at (i, j) + s. "
            f"Red bars: the station lines' cuts\nthrough the +pitch blade. Purple: the would-be corner (i, 1) + s and "
            f"the eta pair that would straddle the cut.", transform=ax.transAxes, va="top", fontsize=7.4, color=INK)
    return dict(pos=pos, P=P)


def connect(fig, ax_a, xy_a, ax_b, xy_b, text: str) -> None:
    from matplotlib.patches import ConnectionPatch
    con = ConnectionPatch(xyA=xy_a, coordsA=ax_a.transData, xyB=xy_b, coordsB=ax_b.transData, arrowstyle="->",
                          color="#b5b3ad", linewidth=0.6, linestyle=(0, (3, 2)), shrinkA=7, shrinkB=5, zorder=20,
                          alpha=0.9)
    fig.add_artist(con)
    ax_b.annotate(text, xy_b, xytext=(6, 6), textcoords="offset points", fontsize=6.6, color=INK, zorder=12,
                  bbox=dict(facecolor=SURFACE, edgecolor="none", alpha=0.85, pad=0.6))


def figure_index(case: Case) -> plt.Figure:
    J = case.J
    fig = plt.figure(figsize=(19, 21), facecolor=SURFACE)
    fig.suptitle(f"D. Index-space schematic and its physical image, n_passage {case.n_passage}, n_eta {case.n_eta}",
                 x=0.02, ha="left", fontsize=13, color=INK)
    gs = fig.add_gridspec(3, 2, height_ratios=(0.42, 1.0, 1.0), width_ratios=(1.0, 1.0), left=0.05, right=0.97,
                          top=0.955, bottom=0.04, hspace=0.34, wspace=0.10)
    class_map(fig.add_subplot(gs[0, :]), case)
    for row, end in ((1, "le"), (2, "te")):
        ax_i, ax_p = fig.add_subplot(gs[row, 0]), fig.add_subplot(gs[row, 1])
        blk = index_block(ax_i, case, end)
        phys = physical_block(ax_p, case, end)
        i = blk["P"][0]
        st = case.stencil(i, J)
        links = [((i, J, 0), f"P ({i},{J})")]
        for name in ("W", "E", "SE" if end == "le" else "SW"):
            k, jj = st[name].node
            links.append(((k, jj, 0), f"{name} ({k},{jj})"))
        for name in ("NW", "NE"):
            nb = st[name]
            k, jj = nb.node
            if nb.exists:
                links.append(((k, jj - J, 1), f"{name} ({k},{jj}) = ({k},{jj - J}) + s"))
        kb = case.f if end == "le" else case.l
        links.append(((kb, 0, 1), f"({kb},0) + s: lower wall\nof passage +1"))
        for key, text in links:
            if key in blk["pos"] and key in phys["pos"]:
                connect(fig, ax_i, blk["pos"][key], ax_p, phys["pos"][key], text)
        miss = [nb for nb in st.values() if not nb.exists]
        if miss and ("ghost",) in phys["pos"]:
            nb = miss[0]
            connect(fig, ax_i, (nb.node[0], 0.0), ax_p, phys["pos"][("ghost",)],
                    f"would-be {nb.name} ({nb.node[0]},1) + s:\nacross the cut")
    return fig


# =========================================================================================
# Figure E: the topology in 3D (schematic)
# =========================================================================================

def figure_3d(case: Case) -> plt.Figure:
    """The passage as a cylinder in (m, eta): eta wraps around (eta = 0 and eta = 1 are one line on a
    periodic station, the seam), and the blade is a slit along that seam from first_blade_i to
    last_blade_i. Schematic (not to scale): 7 real stations around each transition, the upper half of
    the cylinder only (rows J - 4 .. J and 0 .. 4 of the drawing), 16 eta steps around."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    f, l, J = case.f, case.l, case.J
    Js, R, eps = 16, 1.0, 0.07
    step = 2.0 * np.pi / Js

    def P3(x, th, r=R):
        return np.array([x, r * np.sin(th), r * np.cos(th)])

    fig = plt.figure(figsize=(19, 8.6), facecolor=SURFACE)
    fig.suptitle("E. Topology of one passage (schematic, not to scale): m along the axis, eta around it. On periodic "
                 "stations eta = 0 and eta = 1 are one line (brown seam);\non blade stations the seam is opened "
                 "into a slit, the blade, with two walls (orange: eta = 0, psi = 0; blue: eta = 1, psi = 1).",
                 x=0.02, ha="left", fontsize=12, color=INK)
    for k, end in enumerate(("le", "te")):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        ax.set_facecolor(SURFACE)
        if end == "le":
            stations, i, ib, io = list(range(f - 3, f + 4)), f - 1, f, f - 2
            kind = dz.PERIODIC_LE_TRANSITION
        else:
            stations, i, ib, io = list(range(l - 3, l + 4)), l + 1, l, l + 2
            kind = dz.PERIODIC_TE_TRANSITION
        x = {s: float(n) for n, s in enumerate(stations)}
        rows = range(-4, 5)
        for s in stations:
            arc = np.array([P3(x[s], t) for t in np.linspace(-4 * step, 4 * step, 60)])
            ax.plot(*arc.T, color=STATION, linewidth=0.7)
            for r in rows:
                if r == 0 and not case.periodic[s]:
                    continue
                ax.scatter(*P3(x[s], r * step), s=7, color=C_INTERIOR, depthshade=False)
        for r in rows:
            if r == 0:
                continue
            line = np.array([P3(x[s], r * step) for s in stations])
            ax.plot(*line.T, color=STATION, linewidth=0.6)
        per = [s for s in stations if case.periodic[s]]
        bl = [s for s in stations if not case.periodic[s]]
        ax.plot(*np.array([P3(x[s], 0.0) for s in per]).T, color=C_PERIODIC, linewidth=3.0, linestyle=(0, (5, 2)))
        ax.plot(*np.array([P3(x[s], eps) for s in bl]).T, color=C_SUCTION, linewidth=3.2)
        ax.plot(*np.array([P3(x[s], -eps) for s in bl]).T, color=C_PRESSURE, linewidth=3.2)
        xb0 = x[bl[0]] - (0.5 if end == "le" else 0.0)
        xb1 = x[bl[-1]] + (0.0 if end == "le" else 0.5)
        tip, far = (xb0, xb1) if end == "le" else (xb1, xb0)
        fin = [P3(tip, 0.0), P3(tip + (0.45 if end == "le" else -0.45), 0.0, 1.45), P3(far, 0.0, 1.45), P3(far, 0.0)]
        ax.add_collection3d(Poly3DCollection([fin], facecolor="#7d7a73", edgecolor=INK, linewidth=0.6, alpha=0.55))
        ax.text(*(0.5 * (fin[1] + fin[2]) + np.array([0, 0, 0.1])), "blade = the slit", fontsize=9.5, color=INK,
                ha="center")
        # the stencil of the periodic PDE row P on the seam
        P = P3(x[i], 0.0)
        for s, th, ok in ((io, 0.0, True), (ib, -eps, True)):
            ax.plot(*np.array([P, P3(x[s], th)]).T, color=C_FLUID, linewidth=3.2)
        for s in (io, ib):
            for r in (-1, 1):
                q = P3(x[s], r * step)
                missing = s == ib and r == 1
                ax.plot(*np.array([P, q]).T, color=C_MISSING if missing else C_FLUID, linewidth=1.8,
                        linestyle=(0, (3, 2)) if missing else "-")
                ax.scatter(*q, s=150 if missing else 40, marker="X" if missing else "o",
                           color=C_MISSING if missing else C_FLUID, edgecolor=INK if missing else None,
                           depthshade=False, zorder=10)
        for s, color in ((ib, C_METAL), (io, C_FLUID)):
            arc = np.array([P3(x[s], t, 1.01) for t in np.linspace(-step, step, 30)])
            ax.plot(*arc.T, color=color, linewidth=4.0)
        for offset, c in dz.MIXED_M_DIFFERENCE[kind]:
            s = i + offset
            arc = np.array([P3(x[s], t, 1.06) for t in np.linspace(-step, step, 30)])
            ax.plot(*arc.T, color=C_CODE, linewidth=7, alpha=0.25)
        ax.scatter(*P, s=140, color=C_PERIODIC, edgecolor=INK, depthshade=False, zorder=11)
        ax.text(*(P + np.array([0, -0.15, 0.32])), f"P = ({i},{J})", fontsize=10, color=INK, ha="center")
        ax.text(*(P3(x[ib], step) + np.array([0, 0.2, 0.25])), f"({ib},{J}+1): missing\n(through the blade)",
                fontsize=9, color=C_MISSING, ha="center")
        for s in stations:
            ax.text(x[s], 0.0, -0.25, f"{s}", fontsize=8, color=C_PERIODIC if case.periodic[s] else INK,
                    ha="center")
        ax.text(x[stations[0]] - 0.4, R * np.sin(3 * step), R * np.cos(3 * step), "eta small\n(rows 1, 2, ...)",
                fontsize=8, color=INK2, ha="right")
        ax.text(x[stations[0]] - 0.4, -R * np.sin(3 * step), R * np.cos(3 * step), "eta near 1\n(rows J-1, J-2, ...)",
                fontsize=8, color=INK2, ha="right")
        ax.set_xlim(-0.6, len(stations) - 0.4)
        ax.set_ylim(-1.1, 1.1)
        ax.set_zlim(-0.3, 1.5)
        try:
            ax.set_box_aspect((len(stations), 2.2, 1.8), zoom=1.25)
        except TypeError:                               # matplotlib < 3.6
            ax.set_box_aspect((len(stations), 2.2, 1.8))
        ax.view_init(elev=38, azim=-62)
        ax.set_axis_off()
        ax.set_title(f"{'LE' if end == 'le' else 'TE'}: stations {stations[0]}..{stations[-1]} (numbers below), "
                     f"P = ({i},{J}) on the seam", loc="left", fontsize=10.5, color=INK)
    fig.text(0.02, 0.03,
             f"Green: the neighbours P reads that exist (W, E along m; corners on both sides of the seam on the periodic "
             f"station). Purple X: the corner beyond the slit; the eta difference q at the blade station would have to "
             f"cross the blade (red arc),\nwhile at the periodic station it crosses the seam freely (green arc). Blue "
             f"bands: the three one-sided q's the code uses at this row (PERIODIC_LE/TE_TRANSITION). The station line "
             f"itself never appears: only the connections do.", fontsize=9.5, color=INK, va="bottom")
    return fig


# =========================================================================================
# The printed report
# =========================================================================================

def transition_census(case: Case) -> dict:
    """Every same-eta link and cell diagonal across the two transition intervals, by row."""
    out = {}
    for name, i in (("LE", case.f - 1), ("TE", case.l)):
        same = [(j, case.link(case.point(i, j), case.point(i + 1, j))) for j in range(case.n_eta)]
        diag = [((i, j), (i + 1, j + 1), case.link(case.point(i, j), case.point(i + 1, j + 1))) for j in range(case.J)]
        diag += [((i, j + 1), (i + 1, j), case.link(case.point(i, j + 1), case.point(i + 1, j))) for j in range(case.J)]
        out[name] = dict(same=same, diag=diag)
    return out


def _row_table(case: Case, rows: list[dict]) -> list[str]:
    lines = []
    for r in rows:
        if r["status"] == MISSING:
            what = (f"MISSING ({r['node'][0]},{r['node'][1]}): beyond the upper wall of blade station {r['node'][0]}; "
                    f"its eta pair would straddle {r['metal']:.3e} c of metal")
        else:
            region = "across the transition" if (r["node"] is not None and straddles(case, r["i"], r["node"][0])) else \
                "inside the blade region"
            target = f" -> {r['node']}" if r["node"] is not None else ""
            what = f"METAL{target}: {r['metal']:.3e} c of a {r['length']:.3e} c link ({region})"
        lines.append(f"    ({r['i']},{r['j']}) {r['cls']:20s} {r['name']:6s} [{r['used_by']}] {what}")
    return lines or ["    none"]


def case_section(case: Case) -> list[str]:
    g, J, f, l = case.grid, case.J, case.f, case.l
    cuts = [case.metal.cut(g.m[i], 0) for i in range(f, l + 1)]
    lengths = np.array([c[1] - c[0] for c in cuts])
    cen = case.same_eta_census()
    dcen = case.diagonal_census()
    blade_int = cen[("blade", "interior row")]
    blade_wall = cen[("blade", "boundary row")]
    tc = transition_census(case)
    L = [f"--- n_passage {case.n_passage}, n_eta {case.n_eta}: grid {case.n_m} x {case.n_eta}, dm = {g.dm:.6f} c, "
         f"d eta = {g.deta:.4f}, J = n_eta - 1 = {J}",
         f"    last inlet {f - 1} (m - m_le {(g.m[f - 1] - g.m_le) / g.chord:+.5f} c), first blade {f} "
         f"({g.actual_le_offset / g.chord:+.5f} c), last blade {l} ({g.actual_te_offset / g.chord:+.5f} c), "
         f"first exit {l + 1} ({(g.m[l + 1] - g.m_te) / g.chord:+.5f} c)",
         f"    cut of the first blade station {f}: {lengths[0]:.3e} c; of the last {l}: {lengths[-1]:.3e} c; "
         f"all {l - f + 1} blade stations cut the blade ({lengths.min():.2e} .. {lengths.max():.3f} c)",
         f"    same-eta links between blade stations, interior rows: {blade_int['n']}, through metal "
         f"{len(blade_int['bad'])}; wall-to-wall chords (rows 0, J): {blade_wall['n']}, {len(blade_wall['bad'])} "
         f"through metal (convex wall parts: a chord of the wall, which IS the eta line there)",
         f"    inlet and exit links: {sum(cen[k]['n'] for k in cen if k[0] in ('inlet', 'exit'))}, through metal "
         f"{sum(len(cen[k]['bad']) for k in cen if k[0] in ('inlet', 'exit'))}"]
    for name in ("LE", "TE"):
        bad_same = [(j, c) for j, c in tc[name]["same"] if c.status == METAL]
        bad_diag = [(a, b, c) for a, b, c in tc[name]["diag"] if c.status == METAL]
        i0 = f - 1 if name == "LE" else l
        L.append(f"    {name} interval {i0} -> {i0 + 1}: same-eta links through metal: "
                 + (", ".join(f"j={j} ({'PDE side' if j == J else 'constraint side' if j == 0 else 'interior row'}) "
                              f"{c.metal:.3e} of {c.length:.3e} c" for j, c in bad_same) or "none")
                 + f"; cell diagonals through metal: "
                 + (", ".join(f"{a}-{b} {c.metal:.2e} c" for a, b, c in bad_diag) or "none"))
    bad_d = dcen["blade"]["bad"]
    L.append(f"    cell diagonals inside the blade region through metal: "
             + (", ".join(f"{a}-{b} {m:.2e} of {ln:.2e} c" for a, b, m, ln in bad_d) or "none"))
    for end, i in (("LE", f - 1), ("TE", l + 1)):
        row = case.periodic_rows[i]
        L.append(f"    current code, {row.kind} row at ({i},{J}) reads:")
        for node, coef, q, chk in case.code_row_links(i):
            via = " (as continuation, at (k,1) + s)" if node[1] == 1 else ""
            L.append(f"        {str(node):10s} {coef:+.4f}{via:28s} link {fmt_check(chk)}")
    L.append("    every PDE row of the transition windows whose central stencil (Eqs. 5-35 to 5-39) is not all-fluid:")
    L.append("    LE window " + ", ".join(str(i) for i in case.window("le")) + ":")
    L += _row_table(case, case.scan("le"))
    L.append("    TE window " + ", ".join(str(i) for i in case.window("te")) + ":")
    L += _row_table(case, case.scan("te"))
    return L


def code_reads_on_row(case: Case, i: int) -> list[int]:
    """Stations k whose (k, J) the code's periodic PDE row at (i, J) reads (besides i itself)."""
    return sorted(k for (k, jj), _, _, _ in case.code_row_links(i) if jj == case.J)


def inside_region_metal(case: Case, end: str) -> str:
    """The non-fluid relations of the transition window that lie inside one region (chord artifacts)."""
    rows = [r for r in case.scan(end) if r["status"] == METAL
            and not (r["node"] is not None and straddles(case, r["i"], r["node"][0]))]
    if not rows:
        return "none"
    return ", ".join(f"({r['i']},{r['j']}) {r['name']}" + (f" -> {r['node']}" if r["node"] is not None else "")
                     + f" {r['metal']:.2e} c" for r in rows)


def interior_rows_across(case: Case, name: str) -> list[int]:
    tc = transition_census(case)
    return [j for j, chk in tc[name]["same"] if chk.status == METAL and 0 < j < case.J]


def build_report(cases: list[Case], eta_sweep: tuple = (41, 81, 161)) -> str:
    import textwrap
    main, J = cases[0], cases[0].J
    f, l = main.f, main.l
    W = lambda s: textwrap.fill(s, 104, subsequent_indent="    ")  # noqa: E731
    le = main.stencil(f - 1, J)
    te = main.stencil(l + 1, J)
    le_e, te_w = le["E"].check, te["W"].check
    le_ne, te_nw = le["NE"], te["NW"]
    le_c0 = main.link(main.point(f - 1, 0), main.point(f, 0))
    te_reads = code_reads_on_row(main, l + 1)
    te_central = l in te_reads
    te_code = ("central today, the open item of the README" if te_central else
               f"no longer central: discretization.py now writes this row's psi_m, psi_mm forward one-sided along "
               f"j = {J}, reading stations {', '.join(str(k) for k in te_reads)}, all exit periodic stations, and no "
               f"longer the wall node ({l},{J})")
    te_c0 = main.link(main.point(l, 0), main.point(l + 1, 0))
    out = ["TRANSITION-TOPOLOGY DIAGNOSTIC (read-only; the package is unchanged)",
           f"Cascade {main.cascade.metadata['designation']}, {main.spec.profile_variant}, "
           f"{main.spec.surface_continuity}, stagger {main.spec.stagger_deg:g} deg, solidity {main.spec.solidity:g}; "
           f"inlet {workflow.INLET_ANGLE_DEG:g} deg, exit kappa_te, extensions "
           f"{main.settings.upstream_extension_chords:g} / {main.settings.downstream_extension_chords:g} c "
           f"(scripts/plot_cascade.py EXAMPLE and angles)",
           f"Metal test: {main.metal.kind}; reference blade and copies {COPIES} x s; a link is 'metal' above "
           f"{LINK_TOUCH:g} c. Lengths in chords.", ""]
    for case in cases:
        out += case_section(case) + [""]
    out.append(f"--- n_eta sensitivity at n_passage {main.n_passage} (rows across the transition intervals)")
    eta_adds = False
    for n_eta in eta_sweep:
        c = main if n_eta == main.n_eta else Case(main.n_passage, n_eta)
        tc = transition_census(c)
        parts = []
        for name in ("LE", "TE"):
            rows = [j for j, chk in tc[name]["same"] if chk.status == METAL and 0 < j < c.J]
            bnd = [j for j, chk in tc[name]["same"] if chk.status == METAL and j in (0, c.J)]
            dg = sum(chk.status == METAL for _, _, chk in tc[name]["diag"])
            parts.append(f"{name}: boundary rows {bnd or 'none'}, interior rows {rows or 'none'}, diagonals {dg}")
            eta_adds = eta_adds or bool(rows) or dg > 0
        out.append(f"    n_eta {n_eta:3d} (d eta * pitch = {c.grid.deta * c.grid.pitch:.4f} c): " + "; ".join(parts))
    out.append("")
    cmp_ = cases[1] if len(cases) > 1 else None
    q = []
    q.append(("1) What is an m = constant station line?",
              f"In this planar cascade x = m, so station i is the straight vertical line x = m[i]. The grid puts "
              f"n_eta nodes on it, linear in eta between its two boundary nodes. On a periodic station (i < {f} or "
              f"i > {l}) the line misses the blade and runs from the lower periodic line to its image one pitch up. "
              f"On a blade station ({f}..{l}) the line cuts the blade; the eta line uses only the piece from the top "
              f"of the reference blade's cut (lower wall, j = 0) to the bottom of the +pitch blade's cut (upper "
              f"wall, j = J)."))
    q.append(("2) What is a same-eta connection?",
              "The straight link (i, j) -> (i +- 1, j): fixed eta, neighbouring stations. Inside one region it is the "
              "chord of the eta = const coordinate line, which follows the boundary curves and stays in fluid. Across "
              "a transition interval (last periodic -> first blade station, last blade -> first exit station) the "
              "boundary rows change identity from periodic line to wall and the nose / tail sits between the stations, "
              "so there is no continuous coordinate line: the straight link is the only connection a difference can "
              "mean, and it is valid only if it stays in fluid."))
    q.append(("3) Why can a station line cut the blade and still be acceptable?",
              f"Because no stencil uses the part of the line inside the blade. All {l - f + 1} blade stations cut the "
              f"blade (that is their definition), the cut at the first one is only "
              f"{main.metal.cut(main.grid.m[f], 0)[1] - main.metal.cut(main.grid.m[f], 0)[0]:.2e} c at the nose tip, "
              f"yet every wall-to-wall eta line is fluid (grid.py invariant) and every same-eta link between blade "
              f"stations in the interior rows is fluid ({main.same_eta_census()[('blade', 'interior row')]['n']} of "
              f"them, 0 through metal). Mid-passage nodes half a pitch from the cut see nothing of it (Figures B/C, "
              f"panel g)."))
    q.append(("4) For psi_m and psi_mm, exactly which neighbour links matter?",
              "At P = (i, j): W = (i - 1, j) and E = (i + 1, j); the links W-P and P-E (Eqs. 5-35, 5-37). On the "
              f"periodic PDE row (i, J) the same: W = (i - 1, J), E = (i + 1, J), real nodes; the continuation enters "
              f"only psi_eta and psi_etaeta."))
    q.append(("5) For psi_meta, exactly which extra neighbour relations matter?",
              "The four corners (i +- 1, j +- 1) of Eq. 5-39, which is (q[i+1] - q[i-1]) / 2dm with q[k] = "
              "(psi[k, j+1] - psi[k, j-1]) / 2deta. So it needs (a) every corner to EXIST in the topology, (b) the eta "
              "pairs q at stations i - 1 and i + 1 to lie in fluid, and (c) the m-connections from P to the corners "
              "(drawn as P -> corner and as the straight diagonals SW-NE, NW-SE) to be fluid connections. On a "
              "periodic station a corner beyond j = J is the continuation (k, 1) + s, valid; on a blade station there "
              "is none."))
    q.append(("6) LE transition, this EXAMPLE",
              f"n_passage {main.n_passage}: the periodic PDE row ({f - 1},{J}): its E link to ({f},{J}) is "
              f"{fmt_check(le_e)}, so central psi_m, psi_mm are valid connections there. Its NE corner ({f},{J + 1}) "
              f"is MISSING: station {f} is a blade station, and the continuation would put the value at ({f},1) + s, "
              f"on the far side of the +pitch blade's nose, so q{f} would straddle {le_ne.ghost_check.metal:.3e} c "
              f"of metal (the cut). The straight link from P to that point is {fmt_check(le_ne.ghost_link)}: the "
              f"failure is topological, not a line through metal. The code already writes this row with the "
              f"one-sided q{f - 1}, q{f - 2}, q{f - 3}. The constraint-side link ({f - 1},0)-({f},0) is "
              f"{fmt_check(le_c0)}. Interior rows across the LE interval through metal at n_eta {main.n_eta}: "
              f"{interior_rows_across(main, 'LE') or 'none'}. Inside the blade region (chords of curved coordinate "
              f"lines at the steep nose, not a topology defect): {inside_region_metal(main, 'le')}."
              + (f" n_passage {cmp_.n_passage}: same picture at ({cmp_.f - 1},{cmp_.J}) (E link "
                 f"{fmt_check(cmp_.stencil(cmp_.f - 1, cmp_.J)['E'].check)}, NE missing, q{cmp_.f} straddles "
                 f"{cmp_.stencil(cmp_.f - 1, cmp_.J)['NE'].ghost_check.metal:.3e} c), but now the constraint-side "
                 f"link ({cmp_.f - 1},0)-({cmp_.f},0) is "
                 f"{fmt_check(cmp_.link(cmp_.point(cmp_.f - 1, 0), cmp_.point(cmp_.f, 0)))}: the first blade "
                 f"station lies {cmp_.grid.actual_le_offset:+.5f} c from the LE point, behind the nose tip, so the "
                 f"straight line from the periodic node to the top of the cut cuts across the nose."
                 if cmp_ else "")))
    q.append(("7) TE transition, this EXAMPLE",
              f"n_passage {main.n_passage}: the periodic PDE row ({l + 1},{J}): its W link to the upper wall node "
              f"({l},{J}) is {fmt_check(te_w)}: it runs through the +pitch blade's rounded tail. The exit side lines "
              f"start at the TE point; the last blade station lies {main.grid.actual_te_offset:+.5f} c from it, "
              f"where the cut through the round tail is already {main.metal.cut(main.grid.m[l], 0)[1] - main.metal.cut(main.grid.m[l], 0)[0]:.3e} c "
              f"tall, so the straight link from the periodic node back to the bottom of the cut chords through the "
              f"tail. The other image fails too: ({l},0)-({l + 1},0) is {fmt_check(te_c0)} (over the top of the "
              f"reference tail). Its NW corner ({l},{J + 1}) is MISSING (q{l} would straddle "
              f"{te_nw.ghost_check.metal:.3e} c of the tail); the code uses the one-sided q{l + 1}, q{l + 2}, q{l + 3}. "
              f"So the TE row has both problems: a topological one in psi_meta (already treated) and a geometric one "
              f"in the central psi_m, psi_mm ({te_code})."
              + (f" n_passage {cmp_.n_passage}: the last blade station is only {cmp_.grid.actual_te_offset:+.5f} c "
                 f"from the TE point, the W link of ({cmp_.l + 1},{cmp_.J}) is "
                 f"{fmt_check(cmp_.stencil(cmp_.l + 1, cmp_.J)['W'].check)}, the j = 0 image "
                 f"{fmt_check(cmp_.link(cmp_.point(cmp_.l, 0), cmp_.point(cmp_.l + 1, 0)))}, NW is missing as "
                 f"always; inside the blade region: {inside_region_metal(cmp_, 'te')} (corner relations to the last "
                 f"blade station's wall nodes through the tail's rounding)." if cmp_ else "")))
    q.append(("8) Geometric intersection of the station line, or topological invalidity of the connection?",
              "Not the station line: every blade station's line cuts the blade and that is harmless. The problem is "
              "the finite-difference neighbour connection, in two forms. (i) Purely topological, at every resolution: "
              "the periodic PDE rows next to the transitions need a corner beyond eta = 1 at a blade station, and "
              "there is no node there (the seam has become a slit). (ii) Topological in nature, detected "
              "geometrically: across a transition interval the m-difference has no coordinate line to follow, so it "
              "means the straight link, and whether that link stays in fluid depends on where the stations fall "
              f"against the nose or tail: it changes with n_passage ({'and finer eta brings interior rows in' if eta_adds else f'here finer eta, up to n_eta {max(eta_sweep)}, adds no interior row: only the two boundary rows fail'}). "
              "Straight chords through metal INSIDE one region (wall chords, the corner relations at the steep first / "
              "last blade stations) are not connection failures: the coordinate lines there follow the walls."))
    probs = [f"({f - 1},{J}) psi_meta: NE ({f},{J + 1}) missing (treated: one-sided q)",
             f"({l + 1},{J}) psi_meta: NW ({l},{J + 1}) missing (treated: one-sided q)"]
    if te_w.status == METAL:
        probs.append(f"({l + 1},{J}) psi_m, psi_mm: W link to ({l},{J}) through the +pitch tail, {te_w.metal:.3e} c "
                     + ("(NOT treated: central today)" if te_central else
                        f"(treated in the current discretization.py: one-sided, reads {te_reads})"))
    if le_e.status == METAL:
        probs.append(f"({f - 1},{J}) psi_m, psi_mm: E link to ({f},{J}) through the +pitch nose, {le_e.metal:.3e} c")
    if te_c0.status == METAL:
        probs.append(f"({l},0)-({l + 1},0): {te_c0.metal:.3e} c through the reference tail (constraint side; read by "
                     f"no PDE row today, but by any row written on j = 0, and by the central m-differences of A, B, C "
                     f"in coefficients.py)")
    if le_c0.status == METAL:
        probs.append(f"({f - 1},0)-({f},0): {le_c0.metal:.3e} c through the reference nose (constraint side)")
    probs.append(f"minor, inside the blade region (chords, not connection failures): LE "
                 f"{inside_region_metal(main, 'le')}; TE {inside_region_metal(main, 'te')}")
    q.append(("9) Concrete problematic nodes at n_passage " + str(main.n_passage), "; ".join(probs) + "."))
    q.append(("10) Most natural next fix direction (from this diagnostic only)",
              "Yes to all three parts, with one refinement. (a) Keep central m-differences wherever both same-eta "
              f"links are fluid connections: at n_eta {main.n_eta} every interior row across both intervals"
              f"{'' if not (interior_rows_across(main, 'LE') or interior_rows_across(main, 'TE')) else ' except the listed ones'}"
              f", and the LE periodic row ({f - 1},{J}){' (clean)' if le_e.status == FLUID else ' (NOT clean here)'}. (b) Where a link across a transition interval crosses "
              f"metal, go one-sided in m into the side that stays in fluid, second order (psi_m = (-3 psi_i + "
              "4 psi_i+1 - psi_i+2) / 2dm forward, psi_mm = (2 psi_i - 5 psi_i+1 + 4 psi_i+2 - psi_i+3) / dm^2), "
              f"all on periodic stations: at n_passage {main.n_passage} that is ({l + 1},{J})"
              f"{'' if te_central else ', which the current discretization.py already writes this way (TE_M_DIFFERENCE, TE_MM_DIFFERENCE, the same formulas)'}"
              f". Decide it by an "
              f"exact link test like this one, not by index: the failing link moves between LE and TE, and between "
              f"the j = 0 and j = J images, with n_passage ({'interior rows join in at finer eta' if eta_adds else 'finer eta adds no interior row here, but the README grid notes record interior-row crossings at other resolutions, e.g. 81 x 81, and in the geometry sweeps up to eta 0.33'}). If a resolution-independent rule is preferred, the simple alternative is "
              f"one-sided psi_m, psi_mm at the two transition periodic rows always, at the price of a larger error "
              f"constant there"
              f"{(' (the current TE rule is of that kind: at n_passage ' + str(cmp_.n_passage) + ' it is one-sided although the W link is clean there)') if (cmp_ is not None and not te_central and cmp_.stencil(cmp_.l + 1, cmp_.J)['W'].status == FLUID and cmp_.l not in code_reads_on_row(cmp_, cmp_.l + 1)) else ''}"
              f". (c) Mixed rows: the missing corners at the periodic rows are topological and occur "
              f"at every resolution, so the existing one-sided q (PERIODIC_LE/TE_TRANSITION) is the right treatment; "
              f"the interior rows at the first / last blade stations have all four corners and need none (their "
              f"chord crossings are inside one region). (d) Whatever rule psi gets, coefficients.py's central "
              f"m-differences of A, B, C (D and E) at stations {f - 1}, {f}, {l}, {l + 1} read the same links and "
              f"should follow it; that is a review item, not changed here."))
    out.append("=== Answers")
    for head, body in q:
        out.append(head)
        out.append(W("    " + body))
        out.append("")
    return "\n".join(out)


# =========================================================================================
# Main
# =========================================================================================

def save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, facecolor=SURFACE)
    print(f"saved {path.relative_to(ROOT)}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transition-topology diagnostic (read-only).")
    parser.add_argument("--cases", type=int, nargs="+", default=list(CASES), help="n_passage values (first = main)")
    parser.add_argument("--n-eta", type=int, default=N_ETA)
    parser.add_argument("--no-3d", action="store_true", help="skip the 3D schematic (Figure E)")
    parser.add_argument("--show", action="store_true", help="also open the windows")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    cases = []
    for n_passage in args.cases:
        case = Case(n_passage, args.n_eta)
        cases.append(case)
        out = OUTPUT / f"np{n_passage}"
        save(figure_global(case), out / "A_global_overview.png")
        save(figure_zoom(case, "le"), out / "B_LE_zoom.png")
        save(figure_zoom(case, "te"), out / "C_TE_zoom.png")
        save(figure_index(case), out / "D_index_space.png")
        if not args.no_3d and "figure_3d" in globals():
            save(globals()["figure_3d"](case), out / "E_topology_3d.png")
        if not args.show:
            plt.close("all")
    if "build_report" in globals():
        text = globals()["build_report"](cases)
        text += "\n=== Figures (outputs/transition_topology/np<n_passage>/)\n" + "\n".join(
            f"    {name}" for name in ("A_global_overview.png  global passage, transition stations and their indices",
                                     "B_LE_zoom.png          LE: station lines, same-eta links, psi_meta relations",
                                     "C_TE_zoom.png          TE: the same",
                                     "D_index_space.png      index-space stencil block mapped to the physical plane",
                                     "E_topology_3d.png      the passage as a cylinder in (m, eta): seam and slit")) + "\n"
        print(text)
        path = OUTPUT / "report.txt"
        path.write_text(text, encoding="utf-8")
        print(f"saved {path.relative_to(ROOT)}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
