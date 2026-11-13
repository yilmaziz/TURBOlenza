"""
Exact geometry of the straight chords between the node of a PDE row and the nodes it reads,
for the audit and diagnostic scripts (not part of the package): does the straight Euclidean
segment between two stencil nodes run through blade metal?

A straight-chord blade intersection is a geometric fact about that segment, not by itself
a proof that the finite-difference stencil is invalid: the differences act in computational
(m, eta) coordinates, along the coordinate curves of the mapping, not along chords. Whether
the mapping between the nodes is continuous, one-to-one and smooth is the question of
scripts/audit_transition_mapping.py.

A link a -> b is checked against the blade copies k * pitch, k = -1, 0, 1, 2, each closed by the
two authoritative C2 curves (cascade.active_curves). Its crossings with each curve are the
roots of g(u) = cross(b - a, X(u) - a) at segment parameters 0 < t < 1: bracketed on 20001
samples per curve and refined by brentq on the curve itself, so they are exact to round-off
on the spline. Between consecutive crossings the segment is inside or outside the blade; a
piece's state is read at its midpoint (point in the densely sampled outline: wrong only for a
piece within the outline's sagitta, 2e-8 chord, of the boundary).

Tolerances (documented, not tuned):
    TOUCH_T          1e-9 of the link: a crossing this close to an end is that end. A link may
                     end on a wall node; touching the wall there is not entering it. A link
                     that enters the metal before its wall end is caught (the entry is a
                     crossing well inside the link).
    METAL_TOLERANCE  1e-7 chord: a chord counts as intersecting the blade when more than this
                     of it lies inside the metal. The real crossings measured are 2.5e-4 to 2.3e-2 chord; the state
                     of a piece is reliable to 2e-8 chord.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from matplotlib.path import Path as MplPath
from scipy.optimize import brentq

METAL_TOLERANCE = 1e-7
TOUCH_T = 1e-9
COPIES = (-1, 0, 1, 2)


@dataclass(frozen=True)
class Crossing:
    inside_length: float                  # chords of the open segment inside metal
    pieces: tuple                         # ((start point, end point, copy), ...) inside metal
    length: float

    @property
    def fraction(self) -> float:
        return self.inside_length / self.length if self.length > 0 else 0.0

    @property
    def clear(self) -> bool:
        """The straight chord does not intersect blade metal (beyond METAL_TOLERANCE)."""
        return self.inside_length <= METAL_TOLERANCE

    @property
    def copies(self) -> tuple:
        return tuple(sorted({p[2] for p in self.pieces}))


class Blade:
    """The reference blade's C2 curves and their pitch copies."""

    def __init__(self, cascade, samples: int = 20001):
        self.pitch = float(cascade.active.pitch)
        self.curves = (cascade.active_curves.suction, cascade.active_curves.pressure)
        self.u = np.linspace(0.0, 1.0, samples)
        self.points = [np.asarray(c.derivative(self.u, 0), dtype=float) for c in self.curves]
        self.outline = np.vstack([self.points[0][::-1], self.points[1][1:]])
        self.paths = {k: MplPath(self.outline + np.array([0.0, k * self.pitch])) for k in COPIES}
        self.boxes = {k: (self.outline.min(axis=0) + [0.0, k * self.pitch], self.outline.max(axis=0) + [0.0, k * self.pitch])
                      for k in COPIES}

    def crossing(self, a, b) -> Crossing:
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        d = b - a
        length = float(np.hypot(*d))
        lo, hi = np.minimum(a, b) - 1e-4, np.maximum(a, b) + 1e-4
        pieces, inside = [], 0.0
        for k in COPIES:
            box_lo, box_hi = self.boxes[k]
            if np.any(hi < box_lo) or np.any(lo > box_hi):
                continue
            shift = np.array([0.0, k * self.pitch])
            ts = []
            for curve, pts in zip(self.curves, self.points):
                p = pts + shift
                near = np.flatnonzero(np.all((p >= lo) & (p <= hi), axis=1))
                if near.size == 0:
                    continue
                idx = np.arange(max(near[0] - 1, 0), min(near[-1] + 2, len(self.u)))
                g = d[0] * (p[idx, 1] - a[1]) - d[1] * (p[idx, 0] - a[0])
                for q in np.flatnonzero(np.sign(g[:-1]) * np.sign(g[1:]) < 0):
                    def gu(u, curve=curve):
                        x = np.asarray(curve.derivative(np.array([u]), 0), dtype=float)[0] + shift
                        return d[0] * (x[1] - a[1]) - d[1] * (x[0] - a[0])
                    u0 = brentq(gu, self.u[idx[q]], self.u[idx[q + 1]], xtol=1e-15, rtol=9e-16)
                    x = np.asarray(curve.derivative(np.array([u0]), 0), dtype=float)[0] + shift
                    t = float(np.dot(x - a, d) / np.dot(d, d))
                    if TOUCH_T < t < 1.0 - TOUCH_T:
                        ts.append(t)
            cuts = np.array([0.0] + sorted(ts) + [1.0])
            for t0, t1 in zip(cuts[:-1], cuts[1:]):
                if t1 - t0 <= 0.0:
                    continue
                if self.paths[k].contains_point(a + 0.5 * (t0 + t1) * d):
                    pieces.append((a + t0 * d, a + t1 * d, k))
                    inside += (t1 - t0) * length
        return Crossing(inside_length=inside, pieces=tuple(pieces), length=length)


# --- which links a PDE row uses ------------------------------------------------------------------

INTERIOR_DIRECTIONS = {"W": (-1, 0), "E": (1, 0), "S": (0, -1), "N": (0, 1), "NW": (-1, 1), "NE": (1, 1),
                       "SW": (-1, -1), "SE": (1, -1)}
INTERIOR_USE = {"W": "psi_m, psi_mm", "E": "psi_m, psi_mm", "S": "psi_eta, psi_etaeta", "N": "psi_eta, psi_etaeta",
                "NW": "psi_meta", "NE": "psi_meta", "SW": "psi_meta", "SE": "psi_meta"}


def physical(g, node) -> np.ndarray:
    """Where a node a periodic PDE row reads is: a grid node, or (k, 1) read through the
    continuation psi[k, J + 1] = psi[k, 1] + 1, which lies one pitch above (k, 1)."""
    point = np.asarray(g.nodes[node], dtype=float)
    return point + (np.array([0.0, g.pitch]) if node[1] == 1 else 0.0)


def periodic_use(row, node, mixed_offsets) -> str:
    """Which derivatives of a PeriodicPDERow read a node (by its place in the row)."""
    i, J = row.target
    k, j = node
    on_line = j == J
    if on_line:
        return "psi_m, psi_mm"
    parts = []
    if k == i:
        parts.append("psi_eta, psi_etaeta" + (" (continued)" if j == 1 else ""))
    if k - i in mixed_offsets:
        parts.append("psi_meta" + (" (continued)" if j == 1 else ""))
    return ", ".join(parts)


def row_links(g, stencil_rows: dict, i: int, J: int, blade: Blade, mixed_table: dict):
    """Every link of the PDE rows at station i: the interior rows j = 1 .. J - 1 (Eq. 5-44, eight
    neighbors) and the periodic row at (i, J) if there is one. Records (dict) with the target,
    the neighbor, a direction or role, the derivatives that read it and the crossing."""
    out = []
    f, l = g.first_blade_i, g.last_blade_i
    for j in range(1, J):
        start = np.asarray(g.nodes[i, j], dtype=float)
        for name, (di, dj) in INTERIOR_DIRECTIONS.items():
            node = (i + di, j + dj)
            out.append(dict(row="interior", target=(i, j), node=node, direction=name, use=INTERIOR_USE[name],
                            crossing=blade.crossing(start, g.nodes[node])))
    row = stencil_rows.get(i)
    if row is not None:
        offsets = {o for o, _ in mixed_table[row.kind]}
        start = np.asarray(g.nodes[row.target], dtype=float)
        for t in row.terms:
            if t.node == row.target:
                continue
            out.append(dict(row=row.kind, target=row.target, node=t.node, direction=f"{t.node[0] - i:+d},{t.node[1]}",
                            use=periodic_use(row, t.node, offsets), crossing=blade.crossing(start, physical(g, t.node))))
    for r in out:
        r["station"] = f"l{i - l:+d}" if abs(i - l) <= abs(i - f) else f"f{i - f:+d}"
    return out


def build_case(cascade, n_passage: int, n_eta: int, inlet: float = math.radians(45.0)):
    """Grid, metrics, coefficients (rho = 1, mdot' = 1) and periodic rows of one resolution."""
    from turbolenza.b2b.boundary_conditions import build_boundary_conditions
    from turbolenza.b2b.coefficients import build_coefficients
    from turbolenza.b2b.config import GridSettings
    from turbolenza.b2b.discretization import build_periodic_pde_rows
    from turbolenza.b2b.grid import make_grid
    from turbolenza.b2b.metrics import build_metrics

    g = make_grid(cascade, GridSettings(n_passage=n_passage, n_eta=n_eta), inlet)
    metrics = build_metrics(g)
    co = build_coefficients(metrics, np.ones((len(g.m), len(g.eta))), 1.0)
    rows = {row.target[0]: row for row in build_periodic_pde_rows(co, metrics, build_boundary_conditions(g))}
    return g, metrics, co, rows


def sweep(cascade, blade: Blade, levels):
    """For each (n_passage, n_eta): every link of the PDE rows at l - 3 .. l + 4 (TE) and
    f - 4 .. f + 3 (LE), and the old central TE link (l + 1, J) -> (l, J). Returns a list of
    dicts: n_passage, n_eta, f, l, chords checked, the straight-chord blade intersections and
    the old central TE chord's crossing."""
    from turbolenza.b2b.discretization import MIXED_M_DIFFERENCE

    results = []
    for n_passage, n_eta in levels:
        g, _, _, rows = build_case(cascade, n_passage, n_eta)
        f, l, J = g.first_blade_i, g.last_blade_i, n_eta - 1
        records = []
        for side, stations in (("TE", range(l - 3, l + 5)), ("LE", range(f - 4, f + 4))):
            for i in stations:
                for r in row_links(g, rows, i, J, blade, MIXED_M_DIFFERENCE):
                    r["side"] = side
                    records.append(r)
        old = blade.crossing(g.nodes[l + 1, J], g.nodes[l, J])
        results.append(dict(n_passage=n_passage, n_eta=n_eta, f=f, l=l, checked=len(records),
                            straight_chord_intersections=[r for r in records if not r["crossing"].clear], old_te=old))
    return results
