"""
Audit of the computational mapping (m, eta) -> (m, y) across the LE and TE transition
intervals of the default C2 cascade (65-(12A10)10 at 33 deg, solidity 1, inlet 45 deg).
Diagnostic only: no production module is changed.

    python scripts/audit_transition_mapping.py --save outputs/transition_mapping_audit --no-show

The question
    The finite differences act in computational coordinates: psi_m is a difference along a
    constant-eta coordinate curve X_eta(m) = (m, y0(m) + eta S(m)), not along the straight
    Euclidean chord between two nodes. scripts/link_geometry.py measured the chords (a
    geometric fact about the segment). This script measures the coordinate curves and the
    mapping themselves: are they continuous, one-to-one, inside the fluid, and smooth enough
    for the Taylor series behind Eqs. 5-35 to 5-39?

The continuous mapping
    grid.py builds the boundary at each station by one rule (make_grid): a station whose line
    cuts the blade takes the passage-facing wall points (the top of the reference blade's cut,
    the bottom of the +pitch blade's cut), every other station the inlet or exit side line
    (and the side line + pitch). Mapping.walls applies exactly that rule, with the
    production functions, at any m; it reproduces the grid's boundary nodes bit for bit
    (checked). Between two stations the coordinate curve is that rule at every m: the
    mapping the nodes sample.

Aungier and our reading
    Aungier (2003, Sec. 5.3, Fig. 5-5) states that the first and last grid points on the blade
    surfaces must lie inside the blade passages; his reason is the eta derivative at the
    blade-surface points near the stagnation streamline, which is ambiguous for nodes outside
    the passage. That is what he says. That every straight chord of a stencil must lie in the
    fluid is not his statement; it was our own criterion, and it is the one this script
    re-examines.

Sections (printed) and figures
    1  the mapping at the transitions: blade extent, jumps of the coordinate curves, slopes
    2  validity over n_passage x n_eta: S, signed cell areas, folded cells, curves in metal,
       straight cells over metal, transition slope jumps and curvature, FD consistency of a
       smooth physical field across the transitions; straight-chord intersections apart
    3  metrics consistency: metrics.S and metrics.tan_beta against the mapping
    4  the n_passage 401, n_eta 121 interior row (878, 119)
    5  failure modes A (node exists, chord crosses), B (node does not exist), C (nodes exist,
       the mapping between them is discontinuous)
    6  TE row CENTRAL against FORWARD: smooth computational fields, a smooth physical field on
       the real grid, coefficients
    figures 01 .. 09 (outputs/transition_mapping_audit/)
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, Path(__file__).resolve().parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402

import link_geometry as lg  # noqa: E402
from potential_flow_solver import boundary_conditions as bc  # noqa: E402
from potential_flow_solver import discretization as dz  # noqa: E402
from potential_flow_solver import grid as gr  # noqa: E402
from potential_flow_solver.geometry import CascadeSpec, make_cascade  # noqa: E402

SPEC = CascadeSpec(blade_type="NACA65", solidity=1.0, stagger_deg=33.0, cl0=1.2, t_max_over_c=0.10,
                   surface_continuity="C2")
INK, MUTED, GRIDLINE = "#1f2328", "#6e7781", "#c9ced6"
BLADE_FILL, BLADE_EDGE = "#e4e7eb", "#57606a"
FLUID, WALL, METAL, OK, ALT = "#2a78d6", "#eb6834", "#cf222e", "#1baf7a", "#8250df"
logging.getLogger("matplotlib.axes._base").setLevel(logging.ERROR)   # equal aspect on data limits: x widened, by design
EPS_LIMIT = 1e-11           # one-sided limits at the blade's extreme points, chords
ALPHA = math.radians(40.0)  # direction of the smooth physical test field


# --- the mapping ----------------------------------------------------------------------------------

class Mapping:
    """make_grid's boundary rule at any m (module docstring), for one grid."""

    def __init__(self, g):
        self.g, self.b = g, g.boundaries
        self.row = gr._row_of(self.b)
        self.pitch = float(self.b.pitch)
        self.m_tip, self.m_tail = float(self.row.lo_m.min()), float(self.row.hi_m.max())
        y0, y1, blade = self.walls(g.m)
        if not (np.array_equal(y0, g.nodes[:, 0, 1]) and np.array_equal(y1, g.nodes[:, -1, 1])
                and np.array_equal(np.flatnonzero(blade), np.arange(g.first_blade_i, g.last_blade_i + 1))):
            raise RuntimeError("Mapping.walls does not reproduce the grid's boundary nodes")

    def walls(self, m):
        m = np.atleast_1d(np.asarray(m, dtype=float))
        cut = self.row.cut(m)
        blade = np.asarray(cut.cuts_blade)
        y0, y1 = np.empty_like(m), np.empty_like(m)
        middle = 0.5 * (self.b.m_le + self.b.m_te)
        for mask, side in ((~blade & (m < middle), gr._inlet_side), (~blade & (m >= middle), gr._exit_side)):
            if mask.any():
                y0[mask] = side(self.b, m[mask])
                y1[mask] = y0[mask] + self.pitch
        if blade.any():
            idx = np.flatnonzero(blade)
            top, bottom = self.row.passage_sides(cut, idx)
            y0[idx], y1[idx] = top, bottom + self.pitch
        return y0, y1, blade

    def y(self, m, eta):
        y0, y1, _ = self.walls(m)
        return y0 + eta * (y1 - y0)

    def side(self, m, which: str):
        """The analytic side line (lower), continued beyond the switch point: a smooth mapping."""
        m = np.atleast_1d(np.asarray(m, dtype=float))
        return (gr._inlet_side if which == "inlet" else gr._exit_side)(self.b, m)

    def jump(self, eta: float, where: str) -> float:
        """y(m+, eta) - y(m-, eta) at the nose tip ('LE') or the tail end ('TE')."""
        point = self.m_tip if where == "LE" else self.m_tail
        values = self.y([point - EPS_LIMIT, point + EPS_LIMIT], eta)
        return float(values[1] - values[0])


def smooth_side_mapping(mp: Mapping, i: int):
    """The smooth mapping on station i's own side of the transition: (Y0(m), S(m)) callables.
    Periodic stations: the side line continued analytically; blade stations: the walls (valid
    inside the blade's m extent)."""
    g = mp.g
    if g.first_blade_i <= i <= g.last_blade_i:
        def y0(m):
            return mp.walls(m)[0]

        def S(m):
            lo, hi, _ = mp.walls(m)
            return hi - lo
        return y0, S, ("blade", mp.m_tip, mp.m_tail)
    which = "inlet" if i < g.first_blade_i else "exit"
    return (lambda m: mp.side(m, which)), (lambda m: np.full(np.size(m), mp.pitch)), (which, -np.inf, np.inf)


def psi_phys(x, y, pitch: float):
    """A smooth physical field with psi(x, y + pitch) = psi + 1 (not a flow: a test function)."""
    s = y - x * math.tan(ALPHA)
    return s / pitch + 0.05 * np.sin(2.0 * math.pi * s / pitch) * np.cos(1.3 * x)


def exact_computational(mp: Mapping, i: int, eta: float):
    """psi_m, psi_mm and psi_meta of psi(m, eta) = psi_phys(m, Y0(m) + eta S(m)) on station i's
    own smooth mapping, at m_i: 5-point differences of the composed smooth function. Periodic
    stations: h = 5e-4 on the analytic side line (truncation ~h^4, 1e-11; round-off of the
    second difference ~1e-15 / h^2, 5e-9); blade stations: h = 0.05 of the distance to the
    blade's extreme point, at most 2e-5, so that the stencil stays inside the wall's smooth
    part (the wall has a branch point there)."""
    y0, S, (kind, lo, hi) = smooth_side_mapping(mp, i)
    m0 = float(mp.g.m[i])
    h = 5e-4 if kind != "blade" else min(2e-5, 0.05 * (m0 - lo), 0.05 * (hi - m0))
    k = 1e-4

    def psi(m, e):
        m = np.atleast_1d(np.asarray(m, dtype=float))
        return psi_phys(m, y0(m) + e * S(m), mp.pitch)

    ms = m0 + h * np.arange(-2, 3)
    v = psi(ms, eta)
    d1 = (v[0] - 8 * v[1] + 8 * v[3] - v[4]) / (12 * h)
    d2 = (-v[0] + 16 * v[1] - 30 * v[2] + 16 * v[3] - v[4]) / (12 * h * h)
    de = [(psi(ms, eta + k) - psi(ms, eta - k)) / (2 * k)]
    de = de[0]
    dme = (de[0] - 8 * de[1] + 8 * de[3] - de[4]) / (12 * h)
    return float(d1), float(d2), float(dme)


# --- validity measures ---------------------------------------------------------------------------

def cell_measures(g):
    """Signed areas and corner cross products of the straight quadrilateral cells
    (i, j) (i+1, j) (i+1, j+1) (i, j+1)."""
    P = g.nodes
    corners = [P[:-1, :-1], P[1:, :-1], P[1:, 1:], P[:-1, 1:]]
    x = [c[..., 0] for c in corners]
    y = [c[..., 1] for c in corners]
    area = 0.5 * sum(x[k] * y[(k + 1) % 4] - x[(k + 1) % 4] * y[k] for k in range(4))
    cross = []
    for k in range(4):
        a, b_, c = corners[k], corners[(k + 1) % 4], corners[(k + 2) % 4]
        u, v = b_ - a, c - b_
        cross.append(u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0])
    return area, np.min(np.stack(cross), axis=0)


def shoelace(P) -> float:
    return 0.5 * float(np.sum(P[:, 0] * np.roll(P[:, 1], -1) - np.roll(P[:, 0], -1) * P[:, 1]))


def clip_convex(P, window):
    """Sutherland-Hodgman: polygon P (any, closed implicitly) clipped to the convex,
    counterclockwise polygon window. The area of the result is exact for a concave P (the
    degenerate edges it may leave along the window have no area)."""
    for a, b in zip(window, np.roll(window, -1, axis=0)):
        if len(P) == 0:
            break
        d = (b[0] - a[0]) * (P[:, 1] - a[1]) - (b[1] - a[1]) * (P[:, 0] - a[0])
        Q, dq = np.roll(P, -1, axis=0), np.roll(d, -1)
        inside_p, inside_q = d >= 0.0, dq >= 0.0
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(inside_p != inside_q, d / (d - dq), 0.0)
        X = P + t[:, None] * (Q - P)
        P = np.stack([X, Q], axis=1)[np.stack([inside_p != inside_q, inside_q], axis=1)]
    return P


def cell_metal(g, blade: lg.Blade, i: int, j: int) -> float:
    """Metal area inside the straight cell (i, j) (convex: every corner turns the same way,
    checked by cell_measures), over the blade copies 0 and 1, as a fraction of the cell area."""
    quad = np.array([g.nodes[i, j], g.nodes[i + 1, j], g.nodes[i + 1, j + 1], g.nodes[i, j + 1]], dtype=float)
    area = shoelace(quad)
    if area < 0.0:
        quad, area = quad[::-1], -area
    lo, hi = quad.min(axis=0), quad.max(axis=0)
    metal = 0.0
    for k in (0, 1):
        box_lo, box_hi = blade.boxes[k]
        if np.any(hi < box_lo) or np.any(lo > box_hi):
            continue
        loop = blade.outline + np.array([0.0, k * blade.pitch])
        clipped = clip_convex(loop, quad)
        if len(clipped) >= 3:
            metal += abs(shoelace(clipped))
    return metal / area


def cells_over_metal(g, blade: lg.Blade) -> dict:
    """Straight cells beside the walls around both transitions (cells i = f-3 .. f+1 and
    l-2 .. l+2, rows j = 0, 1, 2 and J-3, J-2, J-1) with their metal fraction, in two kinds:
    'transition' (between the last periodic and the first blade station, i = f-1, or the
    last blade and the first exit station, i = l) and 'wall' (both stations blade stations:
    the straight edge between two wall nodes on a convex wall clips a sliver of metal, its
    sagitta). Returns {kind: [(i, j, fraction), ...]} with the cells whose fraction > 0."""
    f, l, J = g.first_blade_i, g.last_blade_i, len(g.eta) - 1
    out = {"transition": [], "wall": []}
    for i in list(range(f - 3, f + 2)) + list(range(l - 2, l + 3)):
        for j in (0, 1, 2, J - 3, J - 2, J - 1):
            fraction = cell_metal(g, blade, i, j)
            if fraction > 0.0:
                kind = "transition" if i in (f - 1, l) else ("wall" if f <= i < l else "periodic")
                out.setdefault(kind, []).append((i, j, fraction))
    return out


def implied_quadratics(mp: Mapping, blade: lg.Blade, samples: int = 1201) -> list:
    """The curve a central m difference assumes: the quadratic through (m_{i-1}, y_{i-1,j}),
    (m_i, y_{i,j}), (m_{i+1}, y_{i+1,j}) (psi_m and psi_mm central are exact for psi
    quadratic along it). At the four corner stations f-1, f, l, l+1, on the levels that carry
    a PDE row there: j = 1 and J-1 (interior rows) everywhere, and j = J on the periodic
    stations f-1 (the LE row, central psi_m and psi_m,m in production) and l+1 (the TE row:
    forward in production, so its central quadratic is the reviewed, old row's). Not j = 0
    (a periodic station's jump relation, a blade station's wall) nor j = J on a blade station
    (a wall): no PDE row there. The fraction of its m extent inside metal (copies 0, 1).
    Returns [(i, j, fraction), ...] for every one checked."""
    g = mp.g
    f, l, J = g.first_blade_i, g.last_blade_i, len(g.eta) - 1
    out = []
    for i in (f - 1, f, l, l + 1):
        for j in ((1, J - 1, J) if i in (f - 1, l + 1) else (1, J - 1)):
            ms = np.linspace(g.m[i - 1], g.m[i + 1], samples)
            y = np.zeros_like(ms)
            for a in (i - 1, i, i + 1):
                basis = np.ones_like(ms)
                for b in (i - 1, i, i + 1):
                    if b != a:
                        basis *= (ms - g.m[b]) / (g.m[a] - g.m[b])
                y += basis * g.nodes[a, j, 1]
            pts = np.column_stack([ms, y])[1:-1]
            inside = np.zeros(len(pts), dtype=bool)
            for k in (0, 1):
                inside |= blade.paths[k].contains_points(pts)
            out.append((i, j, float(inside.mean())))
    return out


def curves_in_metal(mp: Mapping, blade: lg.Blade, intervals, samples: int = 400) -> int:
    """Number of eta levels of the grid whose continuous coordinate curve has a point strictly
    inside blade metal over the given m intervals (the wall levels are checked off the blade's
    m extent only: on it they lie on the wall itself)."""
    g = mp.g
    ms = np.concatenate([np.linspace(a, b, samples) for a, b in intervals])
    y0, y1, onblade = mp.walls(ms)
    count = 0
    for j, eta in enumerate(g.eta):
        y = y0 + eta * (y1 - y0)
        keep = ~onblade if j in (0, len(g.eta) - 1) else np.ones_like(onblade)
        pts = np.column_stack([ms, y])[keep]
        if any(blade.paths[k].contains_points(pts, radius=-1e-12).any() for k in (0, 1)):
            count += 1
    return count


def transition_fd_errors(mp: Mapping, stations, js) -> dict:
    """For the smooth physical field sampled at the nodes, at each given station, max over js:
    |central psi_m - exact|, |one-sided psi_m - exact| (second order, reading the station's own
    region only: backward at f-1 and l, forward at f and l+1) and |exact| (exact on the
    station's own smooth mapping)."""
    g = mp.g
    P = psi_phys(g.nodes[..., 0], g.nodes[..., 1], mp.pitch)
    dm = g.dm
    f, l = g.first_blade_i, g.last_blade_i
    out = {}
    for i in stations:
        worst_c = worst_o = big = 0.0
        for j in js:
            exact = exact_computational(mp, i, float(g.eta[j]))[0]
            central = (P[i + 1, j] - P[i - 1, j]) / (2 * dm)
            if i in (f - 1, l):                      # own region lies behind: backward
                own = (3 * P[i, j] - 4 * P[i - 1, j] + P[i - 2, j]) / (2 * dm)
            else:                                     # f, l + 1: own region ahead: forward
                own = (-3 * P[i, j] + 4 * P[i + 1, j] - P[i + 2, j]) / (2 * dm)
            worst_c, worst_o = max(worst_c, abs(central - exact)), max(worst_o, abs(own - exact))
            big = max(big, abs(exact))
        out[i] = (worst_c, worst_o, big)
    return out


def validity(cascade, blade, n_passage: int, n_eta: int) -> dict:
    g, metrics, co, rows = lg.build_case(cascade, n_passage, n_eta)
    mp = Mapping(g)
    f, l, J = g.first_blade_i, g.last_blade_i, n_eta - 1
    area, cross = cell_measures(g)
    dense = np.linspace(g.m[0], g.m[-1], 20001)
    lo, hi, _ = mp.walls(dense)
    ratio = area / (g.dm * g.deta * np.minimum(metrics.S[:-1], metrics.S[1:])[:, None])
    worst = np.unravel_index(int(np.argmin(ratio)), ratio.shape)
    y = g.nodes[..., 1]
    d2 = np.abs(y[2:] - 2 * y[1:-1] + y[:-2]) / g.dm ** 2               # stations 1 .. n_m - 2
    corner = np.isin(np.arange(1, len(g.m) - 1), [f - 1, f, l, l + 1])
    tb = metrics.tan_beta
    fd = transition_fd_errors(mp, (f - 1, f, l, l + 1), range(1, J))
    cells = cells_over_metal(g, blade)
    implied = implied_quadratics(mp, blade)
    return dict(n_passage=n_passage, n_eta=n_eta, g=g, mp=mp, f=f, l=l,
                S_min=float(min(metrics.S.min(), (hi - lo).min())), S_max=float(max(metrics.S.max(), (hi - lo).max())),
                area_min=float(area.min()), ratio_min=float(ratio.min()), ratio_worst=(int(worst[0]), int(worst[1])),
                negative=int((area <= 0).sum()), nonconvex=int((cross <= 0).sum()), cells=cells,
                transition_metal=max([c[2] for c in cells["transition"]], default=0.0),
                wall_metal=max([c[2] for c in cells["wall"]], default=0.0),
                implied=implied, implied_max=max(q[2] for q in implied if q[:2] != (l + 1, J)),
                implied_old_te=next(q[2] for q in implied if q[:2] == (l + 1, J)),
                curves_metal=curves_in_metal(mp, blade, [(g.m[f - 2], g.m[f + 1]), (g.m[l - 1], g.m[l + 2])]),
                slope_jump_le=float(np.abs(tb[f] - tb[f - 1]).max()), slope_jump_te=float(np.abs(tb[l + 1] - tb[l]).max()),
                curvature_corner=float(d2[corner].max()), curvature_other=float(d2[~corner].max()),
                gap_le=float((g.m[f] - mp.m_tip) / g.dm), gap_te=float((mp.m_tail - g.m[l]) / g.dm),
                jump_le=mp.jump(0.0, "LE"), jump_te=mp.jump(1.0, "TE"), fd=fd)


# --- CENTRAL against FORWARD --------------------------------------------------------------------

@contextlib.contextmanager
def te_variant(name: str):
    """The TE row's psi_m, psi_m,m tables set to CENTRAL (the reviewed formulation) or left as
    FORWARD (production), restored afterwards. Diagnostic only."""
    saved = (dz.TE_M_DIFFERENCE, dz.TE_MM_DIFFERENCE)
    if name == "CENTRAL":
        dz.TE_M_DIFFERENCE, dz.TE_MM_DIFFERENCE = ((-1, -1.0), (1, 1.0)), ((-1, 1.0), (0, -2.0), (1, 1.0))
    try:
        yield
    finally:
        dz.TE_M_DIFFERENCE, dz.TE_MM_DIFFERENCE = saved


def lattice_rows(k: dict, dm: float, deta: float, first: int, last: int):
    n_m, n_eta = k["A"].shape
    region = np.full(n_m, gr.PASSAGE)
    region[:first], region[last + 1:] = gr.INLET, gr.EXIT
    bcs = bc._boundary_conditions(None, np.arange(n_m, dtype=float), np.linspace(0.0, 1.0, n_eta), first, last,
                                  np.arange(first, last + 1), region)
    return dz._periodic_pde_rows(k["A"], k["B"], k["C"], k["D"], k["E"], dm, deta, first, last,
                                 lambda s: bcs.periodic_ghost(s, n_eta))


def lattice_comparison() -> dict:
    """Smooth computational fields on lattices over [0, 1.2] x [0, 1] with the TE row at m = 0.78
    (n = 20 .. 320): psi_m and psi_m,m read through the row (A = 1, only E or only C), and the
    whole row against the exact operator (smooth coefficients, psi = eta + g, g periodic)."""
    out = {"CENTRAL": {}, "FORWARD": {}}
    for variant in out:
        with te_variant(variant):
            for which in ("psi_m", "psi_mm", "operator"):
                errors, spacings = [], []
                for n in (20, 40, 80, 160, 320):
                    dm, deta = 1.2 / n, 1.0 / (8 if which != "operator" else n)
                    n_eta = 9 if which != "operator" else n + 1
                    M, H = np.meshgrid(dm * np.arange(n + 1), deta * np.arange(n_eta), indexing="ij")
                    first, last = int(round(0.35 * n)), int(round(0.65 * n)) - 1
                    zeros, ones = np.zeros_like(M), np.ones_like(M)
                    if which == "operator":
                        w = math.pi
                        A = 1.5 + 0.4 * np.sin(1.7 * M + 0.3) * np.cos(w * H)
                        B = 0.6 * np.cos(1.3 * M) + 0.3 * np.sin(w * H + 0.4)
                        C = 0.9 + 0.3 * np.cos(2.1 * M + 0.1) * np.sin(w * H + 0.2)
                        D = 0.8 * np.sin(1.1 * M) + 0.5 * np.cos(w * H)
                        E = -0.7 * np.cos(0.9 * M + 0.5) + 0.4 * np.sin(2 * w * H)
                        k = dict(A=A, B=B, C=C, D=D, E=E)
                        tw = 2 * math.pi
                        s1, c1 = np.sin(1.7 * M + 0.3), np.cos(1.7 * M + 0.3)
                        psi = H + 0.1 * s1 * np.cos(tw * H)
                        d = dict(m=0.1 * 1.7 * c1 * np.cos(tw * H), mm=-0.1 * 2.89 * s1 * np.cos(tw * H),
                                 e=1.0 - 0.1 * s1 * tw * np.sin(tw * H), ee=-0.1 * s1 * tw * tw * np.cos(tw * H),
                                 me=-0.1 * 1.7 * c1 * tw * np.sin(tw * H))
                    else:
                        k = dict(A=ones, B=zeros, C=zeros if which == "psi_m" else ones, D=zeros,
                                 E=ones if which == "psi_m" else zeros)
                        psi = H + np.sin(2.0 * M + 0.3)
                    row = next(r for r in lattice_rows(k, dm, deta, first, last) if r.kind == dz.PERIODIC_TE_TRANSITION)
                    i, J = row.target
                    if which == "operator":
                        exact = (k["A"][i, J] * d["ee"][i, J] - 2 * k["B"][i, J] * d["me"][i, J] + k["C"][i, J] * d["mm"][i, J]
                                 + k["D"][i, J] * d["e"][i, J] + k["E"][i, J] * d["m"][i, J])
                        discrete = -row.H * row.residual(psi)
                    else:
                        m_i = M[i, J]
                        exact = 2.0 * math.cos(2.0 * m_i + 0.3) if which == "psi_m" else -4.0 * math.sin(2.0 * m_i + 0.3)
                        discrete = -row.H * row.residual(psi)
                    errors.append(abs(discrete - exact))
                    spacings.append(dm)
                out[variant][which] = (spacings, errors)
    return out


def real_grid_comparison(cascade, levels=(61, 101, 201, 401, 801, 1601), n_eta: int = 41) -> dict:
    """The smooth physical field sampled at the real grid's nodes: psi_m and psi_m,m at the TE row
    (l + 1, J) by CENTRAL and FORWARD, and at the LE row (f - 1, J) by central and backward,
    against the exact computational derivatives on each row's own (side-line) mapping."""
    out = {key: [] for key in ("dm", "TE psi_m CENTRAL", "TE psi_m FORWARD", "TE psi_mm CENTRAL", "TE psi_mm FORWARD",
                               "LE psi_m central", "LE psi_m backward", "LE psi_mm central", "LE psi_mm backward")}
    from potential_flow_solver.config import GridSettings

    for n in levels:
        g = gr.make_grid(cascade, GridSettings(n_passage=n, n_eta=n_eta), math.radians(45.0))
        mp = Mapping(g)
        f, l, J, dm = g.first_blade_i, g.last_blade_i, n_eta - 1, g.dm
        P = psi_phys(g.nodes[..., 0], g.nodes[..., 1], mp.pitch)
        te1, te2, _ = exact_computational(mp, l + 1, 1.0)
        le1, le2, _ = exact_computational(mp, f - 1, 1.0)
        out["dm"].append(dm)
        out["TE psi_m CENTRAL"].append(abs((P[l + 2, J] - P[l, J]) / (2 * dm) - te1))
        out["TE psi_m FORWARD"].append(abs((-3 * P[l + 1, J] + 4 * P[l + 2, J] - P[l + 3, J]) / (2 * dm) - te1))
        out["TE psi_mm CENTRAL"].append(abs((P[l + 2, J] - 2 * P[l + 1, J] + P[l, J]) / dm ** 2 - te2))
        out["TE psi_mm FORWARD"].append(abs((2 * P[l + 1, J] - 5 * P[l + 2, J] + 4 * P[l + 3, J] - P[l + 4, J]) / dm ** 2 - te2))
        out["LE psi_m central"].append(abs((P[f, J] - P[f - 2, J]) / (2 * dm) - le1))
        out["LE psi_m backward"].append(abs((3 * P[f - 1, J] - 4 * P[f - 2, J] + P[f - 3, J]) / (2 * dm) - le1))
        out["LE psi_mm central"].append(abs((P[f, J] - 2 * P[f - 1, J] + P[f - 2, J]) / dm ** 2 - le2))
        out["LE psi_mm backward"].append(abs((2 * P[f - 1, J] - 5 * P[f - 2, J] + 4 * P[f - 3, J] - P[f - 4, J]) / dm ** 2 - le2))
        out.setdefault("exact", []).append((te1, te2, le1, le2))
    return out


def coefficient_comparison(cascade) -> list:
    from potential_flow_solver.boundary_conditions import build_boundary_conditions
    rows = []
    for n_passage in (101, 401):
        g, metrics, co, _ = lg.build_case(cascade, n_passage, 41)
        bcs = build_boundary_conditions(g)
        for variant in ("CENTRAL", "FORWARD"):
            with te_variant(variant):
                te = next(r for r in dz.build_periodic_pde_rows(co, metrics, bcs) if r.kind == dz.PERIODIC_TE_TRANSITION)
            center = te.coefficient(te.target)
            off = [abs(t.coefficient) for t in te.terms if t.node != te.target]
            rows.append(dict(n_passage=n_passage, variant=variant, center=center, largest=max(abs(t.coefficient) for t in te.terms),
                             largest_node=max(te.terms, key=lambda t: abs(t.coefficient)).node, off_sum=sum(off),
                             dominance=sum(off) / abs(center), nodes=sorted(te.nodes)))
    return rows


# --- figures ---------------------------------------------------------------------------------------

def style(ax, title: str) -> None:
    ax.set_title(title, fontsize=9.5, color=INK, loc="left")
    ax.tick_params(labelsize=7.5, colors=MUTED)
    for spine in ax.spines.values():
        spine.set_color(GRIDLINE)


def draw_blades(ax, blade, copies=(0, 1)) -> None:
    for k in copies:
        loop = blade.outline + np.array([0.0, k * blade.pitch])
        ax.fill(loop[:, 0], loop[:, 1], color=BLADE_FILL, zorder=1)
        ax.plot(loop[:, 0], loop[:, 1], color=BLADE_EDGE, lw=1.0, zorder=2)


def plot_curve(ax, mp: Mapping, m_range, eta: float, **kw) -> None:
    """The continuous coordinate curve, with a break and a dotted connector at the switch point."""
    ms = np.linspace(*m_range, 3000)
    point = mp.m_tip if m_range[0] < mp.m_tip < m_range[1] else (mp.m_tail if m_range[0] < mp.m_tail < m_range[1] else None)
    if point is not None:
        ms = np.sort(np.concatenate([ms, [point - EPS_LIMIT, point + EPS_LIMIT]]))
    y = mp.y(ms, eta)
    if point is not None:
        k = int(np.searchsorted(ms, point))
        ax.plot(ms[:k], y[:k], **kw)
        kw2 = dict(kw)
        kw2.pop("label", None)
        ax.plot(ms[k:], y[k:], **kw2)
        ax.plot([point, point], [y[k - 1], y[k]], color=kw.get("color", INK), lw=0.9, ls=":", zorder=kw.get("zorder", 4))
    else:
        ax.plot(ms, y, **kw)


def eta_levels(g):
    J = len(g.eta) - 1
    return [(0, "0"), (1, "deta"), (2, "2 deta"), (J // 4, "0.25"), (J // 2, "0.5"), (3 * J // 4, "0.75"),
            (J - 2, "1 - 2 deta"), (J - 1, "1 - deta"), (J, "1")]


def figure_coordinate_curves(blade, case, side: str, chords) -> plt.Figure:
    g, mp = case["g"], case["mp"]
    f, l, J = case["f"], case["l"], len(g.eta) - 1
    a, b = (f - 3, f + 2) if side == "LE" else (l - 2, l + 3)
    m_range = (g.m[a], g.m[b])
    fig, axes = plt.subplots(1, 3, figsize=(19, 7.5), gridspec_kw=dict(width_ratios=(1, 1, 1.1)))
    levels = eta_levels(g)
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(levels)))
    point = mp.m_tip if side == "LE" else mp.m_tail
    corner = (f - 1, f) if side == "LE" else (l, l + 1)
    for panel, (ax, rows_of_interest, wall) in enumerate(zip(axes[:2], ((0, 1), (J - 1, J)), ("lower", "upper"))):
        draw_blades(ax, blade, copies=(0,) if wall == "lower" else (1,))
        for i in range(a, b + 1):
            ax.axvline(g.m[i], color=GRIDLINE, lw=0.6, zorder=0)
        for (j, label), color in zip(levels, colors):
            if j in rows_of_interest:
                plot_curve(ax, mp, m_range, float(g.eta[j]), color=color, lw=1.8, label=f"coordinate curve eta = {label}",
                           zorder=4)
                ax.plot(g.nodes[a:b + 1, j, 0], g.nodes[a:b + 1, j, 1], "o", ms=4, color=color, zorder=5)
        periodic = f - 1 if side == "LE" else l + 1
        quadratics = [(i, 1 if wall == "lower" else J - 1) for i in corner]
        if wall == "upper":
            quadratics.append((periodic, J))
        for (i, jq), ls in zip(quadratics, ("--", ":", "-.")):
            ms = np.linspace(g.m[i - 1], g.m[i + 1], 400)
            y = sum(np.prod([(ms - g.m[q]) / (g.m[p] - g.m[q]) for q in (i - 1, i, i + 1) if q != p], axis=0) * g.nodes[p, jq, 1]
                    for p in (i - 1, i, i + 1))
            what = ("the LE row (central, production)" if side == "LE" else "the old central TE row") if jq == J else "interior row"
            ax.plot(ms, y, color=ALT, lw=1.3, ls=ls, zorder=6, label=f"central quadratic at {(i, jq)}, {what}")
        for c in chords:
            if c["side"] == side and (c["target"][1] in rows_of_interest or c["node"][1] in rows_of_interest):
                p, q = g.nodes[c["target"]], g.nodes[c["node"]]
                ax.plot([p[0], q[0]], [p[1], q[1]], color=METAL, ls="--", lw=1.6, zorder=6,
                        label=f"straight chord {c['target']}->{c['node']} ({100 * c['crossing'].fraction:.0f} % in metal)")
        lo_y = min(mp.y([point - EPS_LIMIT, point + EPS_LIMIT], float(g.eta[j])).min() for j in rows_of_interest)
        hi_y = max(mp.y([point - EPS_LIMIT, point + EPS_LIMIT], float(g.eta[j])).max() for j in rows_of_interest)
        pad = max(2.5 * g.dm, 0.012)
        ax.set_xlim(*m_range)
        ax.set_ylim(lo_y - pad, hi_y + pad)
        ax.set_aspect("equal", adjustable="datalim")
        ax.legend(fontsize=7, loc="lower right" if wall == "lower" else "upper left", framealpha=0.9)
        style(ax, f"{side}, {wall} wall: the coordinate curves through the nodes")
    ax = axes[2]
    for (j, label), color in zip(levels, colors):
        plot_curve(ax, mp, m_range, float(g.eta[j]), color=color, lw=1.4, label=f"eta = {label}")
    ax.axvline(point, color=METAL, lw=0.8, ls=":")
    ax.text(point, 0.98, f" {'nose tip' if side == 'LE' else 'tail end'}: every curve jumps by "
                         f"{mp.jump(0.0, side):+.2e} c (eta 0) .. {mp.jump(1.0, side):+.2e} c (eta 1)",
            transform=ax.get_xaxis_transform(), fontsize=8, color=METAL, va="top")
    for i in range(a, b + 1):
        ax.axvline(g.m[i], color=GRIDLINE, lw=0.6, zorder=0)
    ax.set_xlim(*m_range)
    ax.legend(fontsize=7, loc="center right")
    style(ax, f"{side}: all nine eta levels, y(m, eta) (not to scale)")
    fig.suptitle(f"{'01' if side == 'LE' else '02'}  {side} transition, n_passage {case['n_passage']}, n_eta "
                 f"{case['n_eta']}: each coordinate curve follows the side line, jumps at the "
                 f"{'nose tip' if side == 'LE' else 'tail end'}, then follows the wall; purple = the quadratic a central "
                 f"difference assumes; red = straight chords that intersect the blade", fontsize=11, color=INK, x=0.01,
                 ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def figure_03(blade, cases, cascade) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(17, 8.2))
    te_point = np.asarray(cascade.active.trailing_edge_point, dtype=float) + np.array([0.0, blade.pitch])
    for ax, case in zip(axes, cases):
        g, mp = case["g"], case["mp"]
        l, J = case["l"], len(g.eta) - 1
        draw_blades(ax, blade, copies=(1,))
        m_range = (g.m[l - 1], g.m[l + 3])
        for i in range(l - 1, l + 4):
            ax.axvline(g.m[i], color=GRIDLINE, lw=0.6, zorder=0)
        plot_curve(ax, mp, m_range, 1.0, color=OK, lw=3.0, zorder=5, label="eta = 1 coordinate curve (the mapping)")
        ms = np.linspace(g.m[l], g.m[l + 2], 400)
        quad = sum(np.prod([(ms - g.m[q]) / (g.m[p] - g.m[q]) for q in (l, l + 1, l + 2) if q != p], axis=0) * g.nodes[p, J, 1]
                   for p in (l, l + 1, l + 2))
        ax.plot(ms, quad, color=ALT, lw=1.5, ls="-.", zorder=6,
                label="quadratic through (l, J), (l+1, J), (l+2, J): what the old central psi_m, psi_mm assume")
        c = blade.crossing(g.nodes[l + 1, J], g.nodes[l, J])
        ax.plot([g.nodes[l, J, 0], g.nodes[l + 1, J, 0]], [g.nodes[l, J, 1], g.nodes[l + 1, J, 1]], color=FLUID, lw=1.6,
                ls="--", zorder=6, label="straight chord (l, J) -> (l+1, J)")
        for p0, p1, _ in c.pieces:
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=METAL, lw=4, zorder=7, solid_capstyle="butt",
                    label=f"chord inside metal: {c.inside_length:.2e} of {c.length:.2e} c ({100 * c.fraction:.0f} %)")
        tail = mp.y([mp.m_tail - EPS_LIMIT], 1.0)[0]
        ax.plot(mp.m_tail, tail, "+", ms=13, mew=2, color=INK, zorder=9)
        ax.annotate(f"tail end: the eta = 1 curve leaves the wall here\nand jumps {mp.jump(1.0, 'TE'):+.2e} c to the exit "
                    f"side line", (mp.m_tail, tail), xytext=(18, -60), textcoords="offset points", fontsize=8, color=INK,
                    arrowprops=dict(arrowstyle="-", color=MUTED))
        ax.plot(*te_point, "x", ms=10, mew=2, color=INK, zorder=9)
        ax.annotate("chord-axis TE (+pitch)", te_point, xytext=(-20, 40), textcoords="offset points", fontsize=8, color=INK,
                    ha="right", arrowprops=dict(arrowstyle="-", color=MUTED))
        ax.plot(*g.nodes[l, J], "D", ms=9, color=WALL, zorder=10)
        ax.annotate(f"(l, J) = ({l}, {J})", g.nodes[l, J], xytext=(-90, -18), textcoords="offset points", fontsize=8.5,
                    color=INK)
        ax.plot(*g.nodes[l + 1, J], "o", ms=10, mfc="white", mec=INK, mew=1.8, zorder=10)
        ax.annotate(f"(l+1, J) = ({l + 1}, {J})", g.nodes[l + 1, J], xytext=(14, -4), textcoords="offset points",
                    fontsize=8.5, color=INK)
        centre = 0.5 * (g.nodes[l, J] + g.nodes[l + 1, J])
        half = 1.6 * max(abs(g.nodes[l + 1, J] - g.nodes[l, J]))
        ax.set_xlim(centre[0] - half, centre[0] + 1.6 * half)
        ax.set_ylim(centre[1] - half, centre[1] + half)
        ax.set_aspect("equal")
        ax.legend(fontsize=7.8, loc="lower right", framealpha=0.95)
        style(ax, f"n_passage {case['n_passage']}, n_eta {case['n_eta']}")
    fig.suptitle("03  TE: the straight chord (l, J) -> (l+1, J) and the quadratic of the old central differences cut the "
                 "tail.\nThe eta = 1 coordinate curve of the mapping runs on the wall to the tail end (vertical tangent), "
                 "then jumps to the side line: never in metal, but neither continuous nor smooth", fontsize=10.5,
                 color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def figure_04(blade, case, info) -> plt.Figure:
    g, mp = case["g"], case["mp"]
    l, J = case["l"], len(g.eta) - 1
    i, j = l, J - 1
    fig, axes = plt.subplots(1, 2, figsize=(17, 8.4), gridspec_kw=dict(width_ratios=(1.35, 1)))
    ax = axes[0]
    draw_blades(ax, blade, copies=(1,))
    m_range = (g.m[i - 1], g.m[i + 1])
    for k in range(i - 1, i + 2):
        ax.axvline(g.m[k], color=GRIDLINE, lw=0.6, zorder=0)
    for jj, color in zip((j - 1, j, j + 1), (ALT, FLUID, OK)):
        plot_curve(ax, mp, m_range, float(g.eta[jj]), color=color, lw=2.0, zorder=4, label=f"coordinate curve eta_{jj}")
    for di in (-1, 0):
        for dj in (-1, 0):
            quad = np.array([g.nodes[i + di, j + dj], g.nodes[i + di + 1, j + dj], g.nodes[i + di + 1, j + dj + 1],
                             g.nodes[i + di, j + dj + 1], g.nodes[i + di, j + dj]])
            ax.plot(quad[:, 0], quad[:, 1], color=MUTED, lw=0.7, zorder=3)
    c = blade.crossing(g.nodes[i, j], g.nodes[i + 1, j + 1])
    ax.plot([g.nodes[i, j, 0], g.nodes[i + 1, j + 1, 0]], [g.nodes[i, j, 1], g.nodes[i + 1, j + 1, 1]], color=METAL, lw=1.5,
            ls="--", zorder=6, label=f"straight NE chord ({100 * c.fraction:.0f} % in metal)")
    for di, dj, name in ((-1, -1, "SW"), (0, -1, "S"), (1, -1, "SE"), (-1, 0, "W"), (1, 0, "E"), (-1, 1, "NW"), (0, 1, "N"),
                         (1, 1, "NE")):
        node = (i + di, j + dj)
        wall = node[1] == J and g.first_blade_i <= node[0] <= g.last_blade_i
        ax.plot(*g.nodes[node], "D" if wall else "o", ms=7, color=WALL if wall else FLUID, zorder=8)
        ax.annotate(f"{name} {node}", g.nodes[node], xytext=(5, 4), textcoords="offset points", fontsize=7.8, color=INK)
    ax.plot(*g.nodes[i, j], "o", ms=13, mfc="white", mec=INK, mew=1.8, zorder=9)
    ax.annotate(f"P {(i, j)}", g.nodes[i, j], xytext=(-70, -4), textcoords="offset points", fontsize=8.5, color=INK)
    ax.axvline(mp.m_tail, color=METAL, lw=0.8, ls=":")
    reach = max(float(np.max(np.abs(g.nodes[i + a, j + b] - g.nodes[i, j]))) for a in (-1, 0, 1) for b in (-1, 0, 1))
    ax.set_xlim(g.nodes[i, j, 0] - 1.3 * reach, g.nodes[i, j, 0] + 1.3 * reach)
    ax.set_ylim(g.nodes[i, j, 1] - 1.3 * reach, g.nodes[i, j, 1] + 1.3 * reach)
    ax.set_aspect("equal")
    ax.legend(fontsize=7.8, loc="lower left", framealpha=0.95)
    style(ax, f"physical plane: the nine nodes of {(i, j)}, the coordinate curves through them, the straight cells")
    ax = axes[1]
    ax.axis("off")
    text = (f"psi_meta at P = {(i, j)} (n_passage 401, n_eta 121) is a Taylor approximation in (m, eta):\n"
            f"  psi_meta = d/dm of psi_eta  ~  (q[i+1] - q[i-1]) / (2 dm),   q[k] = (psi[k, j+1] - psi[k, j-1]) / (2 deta)\n"
            f"i.e. a difference in m of eta-differences at stations i-1 and i+1, along the coordinate curves;\n"
            f"the corner values are not samples along a straight diagonal ray.\n\n"
            f"Required computational nodes: all nine exist (indices in range, none is a periodic ghost).\n"
            f"Straight cells around P: signed areas {info['areas']}, all > 0.\n\n"
            f"But station i = l = {l} is the last blade station and i+1 = l+1 = {l + 1} the first exit station:\n"
            f"between them the mapping switches from the walls to the side lines at the tail end,\n"
            f"m_tail = {mp.m_tail:.6f}, {info['gap']:.2f} dm behind station l, where every coordinate curve jumps\n"
            f"(eta_j: {mp.jump(float(g.eta[j]), 'TE'):+.2e} c) and the wall side has a vertical tangent (y_m ~ d^-1/2).\n"
            f"Every m difference of this row (W/E, the four diagonals) straddles that switch, not only NE.\n\n"
            f"Smooth physical field sampled at the nodes, errors against the blade-side exact derivatives:\n"
            f"  psi_m    central {info['err_m']:.2e}  (exact {info['exact_m']:+.3f})\n"
            f"  psi_meta central {info['err_me']:.2e}  (exact {info['exact_me']:+.3f})\n"
            f"  the same row two stations upstream: psi_m {info['ref_m']:.2e}, psi_meta {info['ref_me']:.2e}\n"
            f"  (no switch, but still within reach of the branch point at the tail end)\n\n"
            f"So the NE chord's metal is a geometric fact, not the failure. The failure is the non-smooth\n"
            f"mapping between stations l and l+1, which every m difference of the row crosses.")
    ax.text(0.0, 1.0, text, va="top", fontsize=8.6, color=INK, family="monospace")
    fig.suptitle("04  The n_passage 401 interior row (878, 119): what the mixed derivative samples", fontsize=11.5,
                 color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def figure_cells(blade, cases, side: str) -> plt.Figure:
    """The mapped cells (the images of [m_i, m_i+1] x [eta_j, eta_j+1] under the mapping,
    bounded by the coordinate curves: two pieces across the switch) filled, and the straight
    quadrilaterals through the same four nodes dashed, red where a transition cell's straight
    quadrilateral holds metal."""
    fig, axes = plt.subplots(2, len(cases), figsize=(4.3 * len(cases), 9.6))
    for col, case in enumerate(cases):
        g, mp = case["g"], case["mp"]
        f, l, J = case["f"], case["l"], len(g.eta) - 1
        stations = range(f - 3, f + 2) if side == "LE" else range(l - 2, l + 3)
        point = mp.m_tip if side == "LE" else mp.m_tail
        metal = {(i, j): fr for kind in ("transition", "wall") for i, j, fr in case["cells"].get(kind, [])}
        for row_index, (js, copy) in enumerate((((0, 1), 0), ((J - 2, J - 1), 1))):
            ax = axes[row_index, col]
            draw_blades(ax, blade, copies=(copy,))
            for i in stations:
                transition = i in (f - 1, l)
                for j in js:
                    ms = np.linspace(g.m[i], g.m[i + 1], 600)
                    pieces = [ms[ms < point], ms[ms > point]] if g.m[i] < point < g.m[i + 1] else [ms]
                    for piece in pieces:
                        if piece.size < 2:
                            continue
                        lower, upper = mp.y(piece, float(g.eta[j])), mp.y(piece, float(g.eta[j + 1]))
                        ax.fill_between(piece, lower, upper, color=ALT if transition else OK, alpha=0.30 if transition else 0.15,
                                        lw=0, zorder=3)
                        ax.plot(piece, lower, color=MUTED, lw=0.6, zorder=4)
                        ax.plot(piece, upper, color=MUTED, lw=0.6, zorder=4)
                    quad = np.array([g.nodes[i, j], g.nodes[i + 1, j], g.nodes[i + 1, j + 1], g.nodes[i, j + 1], g.nodes[i, j]])
                    fr = metal.get((i, j), 0.0)
                    red = transition and fr > 0.0
                    ax.plot(quad[:, 0], quad[:, 1], color=METAL if red else INK, ls="--", lw=1.3 if red else 0.6, zorder=5)
                    if red:
                        c = quad[:4].mean(axis=0)
                        ax.annotate(f"{100 * fr:.1f} %", c, fontsize=7.5, color=METAL, ha="center", va="center", zorder=7,
                                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))
            sel = np.array([g.nodes[i, j] for i in stations for j in js] + [g.nodes[i + 1, j + 1] for i in stations for j in js])
            lo, hi = sel.min(axis=0), sel.max(axis=0)
            pad = 0.08 * max(hi - lo)
            ax.set_xlim(lo[0] - pad, hi[0] + pad)
            ax.set_ylim(lo[1] - pad, hi[1] + pad)
            ax.set_aspect("equal")
            n_tr = sum(1 for i, j, _ in case["cells"].get("transition", []) if j in js and i in stations)
            n_wall = sum(1 for i, j, _ in case["cells"].get("wall", []) if j in js and i in stations)
            style(ax, f"{case['n_passage']}x{case['n_eta']} {'lower' if copy == 0 else 'upper'}: "
                      f"{n_tr} transition, {n_wall} sagitta")
    fig.suptitle(f"{'05' if side == 'LE' else '06'}  {side}: mapped cells (filled; purple = the transition interval, two "
                 f"pieces across the switch) and the straight quadrilaterals through the same nodes (dashed; red with the "
                 f"metal fraction where a transition quadrilateral holds metal). 'sagitta' = wall cells whose straight edge "
                 f"clips the convex wall", fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def figure_07(case) -> plt.Figure:
    g, mp = case["g"], case["mp"]
    f, l = case["f"], case["l"]
    fig, axes = plt.subplots(3, 2, figsize=(16, 11), sharex="col")
    for col, (side, point, window) in enumerate((("LE", mp.m_tip, (g.m[f - 3], g.m[f + 2])),
                                                 ("TE", mp.m_tail, (g.m[l - 2], g.m[l + 3])))):
        ms = np.linspace(*window, 6001)
        ms = ms[np.abs(ms - point) > 2e-7]
        for eta, color in ((0.0, FLUID), (0.5, ALT), (1.0, WALL)):
            y = mp.y(ms, eta)
            left, right = ms < point, ms > point
            for part in (left, right):
                mm, yy = ms[part], y[part]
                d1 = np.gradient(yy, mm)
                d2 = np.gradient(d1, mm)
                axes[0, col].plot(mm, yy - (eta * mp.pitch), color=color, lw=1.4, label=f"eta = {eta:g}" if part is left else None)
                axes[1, col].plot(mm, d1, color=color, lw=1.2)
                axes[2, col].plot(mm, d2, color=color, lw=1.0)
        for r in range(3):
            axes[r, col].axvline(point, color=METAL, ls=":", lw=0.9)
            for i in range((f - 3 if side == "LE" else l - 2), (f + 3 if side == "LE" else l + 4)):
                axes[r, col].axvline(g.m[i], color=GRIDLINE, lw=0.6, zorder=0)
        axes[1, col].set_yscale("symlog", linthresh=1.0)
        axes[2, col].set_yscale("symlog", linthresh=10.0)
        axes[0, col].legend(fontsize=7.5)
        style(axes[0, col], f"{side}: y(m, eta) - eta pitch (the jump at the {'nose tip' if side == 'LE' else 'tail end'})")
        style(axes[1, col], f"{side}: y_m = tan(beta) along the curve (symlog): unbounded on the blade side")
        style(axes[2, col], f"{side}: y_mm (symlog)")
        axes[2, col].set_xlabel("m (chords); grey: stations of n_passage 101", fontsize=8.5, color=INK)
    fig.suptitle("07  The mapping along constant eta across the transitions (n_passage 101 stations shown): y jumps, "
                 "y_m and y_mm are unbounded next to the blade's extreme point", fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def figure_08(results) -> plt.Figure:
    n_ps = sorted({r["n_passage"] for r in results})
    n_es = sorted({r["n_eta"] for r in results})
    panels = (("mapping: jump at the nose tip / (2 dm)", lambda r: abs(r["jump_le"]) / (2 * r["g"].dm), "{:.2f}"),
              ("straight cells with area <= 0 or folded", lambda r: r["negative"] + r["nonconvex"], "{:.0f}"),
              ("mapping coordinate curves inside metal", lambda r: r["curves_metal"], "{:.0f}"),
              ("straight-chord blade intersections", lambda r: r["chords"], "{:.0f}"),
              ("max metal in a transition straight cell (%)", lambda r: 100 * r["transition_metal"], "{:.1f}"),
              ("production rows: max metal along a central quadratic (%)", lambda r: 100 * r["implied_max"], "{:.1f}"),
              ("central psi_m error across a switch (f-1 .. l+1)", lambda r: max(r["fd"][i][0] for i in r["fd"]), "{:.2f}"),
              ("one-sided psi_m error, periodic side (f-1, l+1)",
               lambda r: max(r["fd"][r["f"] - 1][1], r["fd"][r["l"] + 1][1]), "{:.0e}"),
              ("one-sided psi_m error, blade side (f, l)", lambda r: max(r["fd"][r["f"]][1], r["fd"][r["l"]][1]), "{:.2f}"))
    fig, axes = plt.subplots(3, 3, figsize=(17, 12.5))
    for ax, (title, value, fmt) in zip(axes.ravel(), panels):
        grid_values = np.full((len(n_es), len(n_ps)), np.nan)
        for r in results:
            grid_values[n_es.index(r["n_eta"]), n_ps.index(r["n_passage"])] = value(r)
        finite = grid_values[np.isfinite(grid_values)]
        vmax = finite.max() if finite.size and finite.max() > 0 else 1.0
        ax.imshow(grid_values, cmap="Oranges", vmin=min(0.0, finite.min()) if finite.size else 0, vmax=vmax, origin="lower",
                  aspect="auto")
        for a in range(len(n_es)):
            for b_ in range(len(n_ps)):
                v = grid_values[a, b_]
                ax.text(b_, a, fmt.format(v) if np.isfinite(v) else "-", ha="center", va="center", fontsize=8,
                        color=INK)
        ax.set_xticks(range(len(n_ps)), n_ps, fontsize=7.5)
        ax.set_yticks(range(len(n_es)), n_es, fontsize=7.5)
        style(ax, title)
    fig.suptitle("08  Over n_passage (x) and n_eta (y). Top: the mapping jumps by more grid spacings as the grid refines, "
                 "but is one-to-one, unfolded and never in metal; straight chords apart.\nMiddle (straight or implied "
                 "objects, not the mapping): straight transition cells hold metal; so does the central quadratic of the "
                 "interior row (f-1, 1) at two grids. Bottom: a smooth physical field,\npsi_m at the four corner stations, "
                 "rows j = 1 .. J-1: central "
                 "across the switch and one-sided at the blade-side stations do not converge; one-sided on the periodic "
                 "side does", fontsize=10.5, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig


def figure_09(lattice, real) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(19, 6.4))
    ax = axes[0]
    for which, marker in (("psi_m", "o"), ("psi_mm", "s"), ("operator", "^")):
        for variant, color in (("CENTRAL", FLUID), ("FORWARD", WALL)):
            h, e = lattice[variant][which]
            ax.loglog(h, e, marker=marker, color=color, ls="-" if variant == "CENTRAL" else "--", label=f"{which} {variant}")
    ax.set_xlabel("dm", fontsize=8.5)
    ax.set_ylabel("|discrete - exact|", fontsize=8.5)
    ax.legend(fontsize=7.5, ncol=2)
    style(ax, "smooth computational fields (lattice, no transition): both second order")
    for ax, quantity in zip(axes[1:], ("psi_m", "psi_mm")):
        for key, color, ls in ((f"TE {quantity} CENTRAL", FLUID, "-"), (f"TE {quantity} FORWARD", WALL, "--"),
                               (f"LE {quantity} central", ALT, "-"), (f"LE {quantity} backward", OK, "--")):
            ax.loglog(real["dm"], real[key], marker="o", color=color, ls=ls, label=key)
        ax.set_xlabel("dm (n_passage 61 .. 1601, n_eta 41)", fontsize=8.5)
        ax.legend(fontsize=7.5)
        style(ax, f"real grid, smooth physical field: {quantity} at the TE and LE periodic rows")
    fig.suptitle("09  CENTRAL against FORWARD at the TE row: both second order on smooth computational fields (central "
                 "with the smaller error constants, 1/2 and 1/11); on the real mapping the central differences straddle "
                 "the switch and do not converge", fontsize=10.5, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig


# --- main ----------------------------------------------------------------------------------------

def orders(e, h):
    return [math.log(e[k] / e[k + 1]) / math.log(h[k] / h[k + 1]) for k in range(len(e) - 1)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--save", type=Path, default=None)
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--quick", action="store_true", help="only the grids the figures need (for checking figures)")
    args = parser.parse_args()
    cascade = make_cascade(SPEC)
    blade = lg.Blade(cascade)

    # 1 the mapping at the transitions
    base = validity(cascade, blade, 101, 41)
    mp = base["mp"]
    print("1. The mapping at the transitions (production rule, reproduces the grid's boundary nodes bit for bit)")
    print(f"   blade m extent: nose tip m_tip = {mp.m_tip:.6f}, tail end m_tail = {mp.m_tail:.6f}; chord-axis m_le = "
          f"{mp.b.m_le:.6f}, m_te = {mp.b.m_te:.6f}")
    for side in ("LE", "TE"):
        print(f"   {side}: jump of y(m, eta) at the switch: " + ", ".join(f"eta {e:g}: {mp.jump(e, side):+.4e} c"
                                                                     for e in (0.0, 0.25, 0.5, 0.75, 1.0)))
    for side, point, sign in (("LE", mp.m_tip, 1), ("TE", mp.m_tail, -1)):
        slopes = []
        for d in (1e-3, 1e-4, 1e-5, 1e-6):
            x = np.array([point + sign * d, point + sign * d * 1.001])
            lo, hi, _ = mp.walls(x)
            slopes.append((d, (lo[1] - lo[0]) / (x[1] - x[0]), (hi[1] - hi[0]) / (x[1] - x[0])))
        which = "inlet" if side == "LE" else "exit"
        side_slope = (mp.side([point + 1e-6], which)[0] - mp.side([point - 1e-6], which)[0]) / 2e-6
        print(f"   {side}: side-line slope at the switch {side_slope:+.4f}; blade-side slopes y0_m, y1_m at distance d: "
              + "; ".join(f"d {d:.0e}: {a:+.2f}, {b:+.2f}" for d, a, b in slopes) + "  (~ d^-1/2: a vertical tangent)")

    # 2 validity sweep
    if args.quick:
        levels = [(61, 41), (101, 41), (201, 41), (401, 41), (401, 121)]
    else:
        levels = [(n, e) for n in (31, 51, 61, 101, 201, 401) for e in (21, 41, 61, 121)]
    print("\n2. Mapping validity (and, apart, straight-object geometry) over the sweep")
    print("   per grid: straight cells (min area / (dm deta S), area <= 0, folded); mapping curves in metal; straight-chord")
    print("   intersections; metal in transition straight cells / wall-sagitta cells; metal along the implied central")
    print("   quadratics; |d tan beta| across each switch; |y_mm| at the corner stations / elsewhere; first / last blade")
    print("   station's distance from the extreme point (dm); smooth physical field psi_m errors at the corner stations")
    print("   f-1, f, l, l+1 (central / one-sided, max over j = 1 .. J-1) and the largest |exact psi_m| there")
    results = []
    chord_results = {(r["n_passage"], r["n_eta"]): r for r in lg.sweep(cascade, blade, levels)}
    for n, e in levels:
        r = base if (n, e) == (101, 41) else validity(cascade, blade, n, e)
        r["chords"] = len(chord_results[(n, e)]["straight_chord_intersections"])
        results.append(r)
        f, l, fd = r["f"], r["l"], r["fd"]
        print(f"   {n:3d}x{e:3d}: cells {r['ratio_min']:.3f} at {r['ratio_worst']}, {r['negative']}, {r['nonconvex']}; curves "
              f"{r['curves_metal']}; chords {r['chords']}; transition cells {len(r['cells']['transition'])} (max "
              f"{100 * r['transition_metal']:.1f} %), sagitta cells {len(r['cells']['wall'])} (max {100 * r['wall_metal']:.2f} %); "
              f"central quadratics of the production rows max {100 * r['implied_max']:.1f} %, of the old central TE row "
              f"{100 * r['implied_old_te']:.1f} %; |d tan b| LE "
              f"{r['slope_jump_le']:.2f} TE {r['slope_jump_te']:.2f}; |y_mm| {r['curvature_corner']:.1e} / "
              f"{r['curvature_other']:.1e}; gaps LE {r['gap_le']:.3f} TE {r['gap_te']:.3f}")
        print("            psi_m " + "; ".join(f"{name} {fd[i][0]:.3f} / {fd[i][1]:.1e} (|exact| {fd[i][2]:.2f})"
                                          for name, i in (("f-1", f - 1), ("f", f), ("l", l), ("l+1", l + 1))))
    print(f"   S over all grids {min(r['S_min'] for r in results):.4f} .. {max(r['S_max'] for r in results):.4f}")

    # 3 metrics consistency
    print("\n3. Metrics consistency")
    for n, e in ((101, 41), (401, 121)):
        r = next(x for x in results if (x["n_passage"], x["n_eta"]) == (n, e))
        g, mpp = r["g"], r["mp"]
        metrics = lg.build_case(cascade, n, e)[1]
        S_nodes = g.nodes[:, -1, 1] - g.nodes[:, 0, 1]
        worst_blade = worst_side = 0.0
        for i in range(len(g.m)):
            y0, S, (kind, lo, hi) = smooth_side_mapping(mpp, i)
            h = min(1e-6, 0.2 * (g.m[i] - lo), 0.2 * (hi - g.m[i]))
            slope0 = float((y0([g.m[i] + h])[0] - y0([g.m[i] - h])[0]) / (2 * h))
            if kind == "blade":
                worst_blade = max(worst_blade, abs(slope0 - metrics.tan_beta0[i]))
            else:
                worst_side = max(worst_side, abs(slope0 - metrics.tan_beta0[i]))
        print(f"   {n}x{e}: metrics.S - (y1 - y0) {np.abs(metrics.S - S_nodes).max():.1e}; tan_beta0 against the mapping's "
              f"slope: blade stations {worst_blade:.1e} (exact spline slopes against a 1e-6 difference), side-line stations "
              f"{worst_side:.1e} (metrics' second-order region differences)")

    # 4 the 401 x 121 interior row
    r401 = next(x for x in results if (x["n_passage"], x["n_eta"]) == (401, 121))
    g4, mp4 = r401["g"], r401["mp"]
    i4, j4 = r401["l"], 119
    P = psi_phys(g4.nodes[..., 0], g4.nodes[..., 1], mp4.pitch)
    dm, de = g4.dm, g4.deta
    ex_m, _, ex_me = exact_computational(mp4, i4, float(g4.eta[j4]))
    cm = (P[i4 + 1, j4] - P[i4 - 1, j4]) / (2 * dm)
    cme = (P[i4 + 1, j4 + 1] - P[i4 - 1, j4 + 1] - P[i4 + 1, j4 - 1] + P[i4 - 1, j4 - 1]) / (4 * dm * de)
    ref_m, _, ref_me = exact_computational(mp4, i4 - 2, float(g4.eta[j4]))
    rcm = (P[i4 - 1, j4] - P[i4 - 3, j4]) / (2 * dm)
    rcme = (P[i4 - 1, j4 + 1] - P[i4 - 3, j4 + 1] - P[i4 - 1, j4 - 1] + P[i4 - 3, j4 - 1]) / (4 * dm * de)
    area, _ = cell_measures(g4)
    areas = [f"{area[i4 + a, j4 + b]:.2e}" for a in (-1, 0) for b in (-1, 0)]
    info = dict(areas=", ".join(areas), err_m=abs(cm - ex_m), exact_m=ex_m, err_me=abs(cme - ex_me), exact_me=ex_me,
                ref_m=abs(rcm - ref_m), ref_me=abs(rcme - ref_me), gap=r401["gap_te"])
    print(f"\n4. Interior row ({i4}, {j4}) at 401x121: nine nodes exist; cells around it {info['areas']} (all > 0); mapping "
          f"switch between l and l+1 at m_tail with a jump {mp4.jump(float(g4.eta[j4]), 'TE'):+.2e} c; smooth physical field: "
          f"central psi_m error {info['err_m']:.2e} (exact {ex_m:+.3f}), central psi_meta error {info['err_me']:.2e} (exact "
          f"{ex_me:+.3f}); two stations upstream: {info['ref_m']:.1e}, {info['ref_me']:.1e}")

    # 5 failure modes
    from potential_flow_solver.boundary_conditions import build_boundary_conditions
    bcs = build_boundary_conditions(base["g"])
    refused = []
    for station in (base["f"], base["l"]):
        try:
            bcs.periodic_ghost(station, len(base["g"].eta))
        except bc.BoundaryConditionError:
            refused.append(station)
    print("\n5. Failure modes (101x41)")
    print(f"   A  node exists, straight chord intersects the blade: {base['chords']} chords (geometry only)")
    print(f"   B  required node does not exist: the central mixed term of the periodic rows at f-1 and l+1 needs "
          f"(f, J+1), (l, J+1); periodic_ghost refuses stations {refused} (genuine, handled by the one-sided mixed terms)")
    print(f"   C  nodes exist, the mapping between them is discontinuous: every central m difference at stations "
          f"f-1, f, l, l+1 straddles a switch (jumps {mp.jump(0.0, 'LE'):+.2e} c LE, {mp.jump(1.0, 'TE'):+.2e} c TE): "
          f"{4 * (len(base['g'].eta) - 2)} interior rows, plus the LE periodic row (central psi_m, psi_mm)")

    # 6 CENTRAL against FORWARD
    lattice = lattice_comparison()
    print("\n6. TE row CENTRAL against FORWARD")
    for which in ("psi_m", "psi_mm", "operator"):
        hc, ec = lattice["CENTRAL"][which]
        hf, ef = lattice["FORWARD"][which]
        print(f"   lattice {which:8s}: CENTRAL {['%.2e' % x for x in ec]} orders {np.round(orders(ec, hc), 2)}; FORWARD "
              f"{['%.2e' % x for x in ef]} orders {np.round(orders(ef, hf), 2)}; error ratio FORWARD/CENTRAL at the finest "
              f"{ef[-1] / ec[-1]:.2f}")
    real = real_grid_comparison(cascade)
    for key in ("TE psi_m CENTRAL", "TE psi_m FORWARD", "TE psi_mm CENTRAL", "TE psi_mm FORWARD", "LE psi_m central",
                "LE psi_m backward", "LE psi_mm central", "LE psi_mm backward"):
        print(f"   real grid {key:20s}: {['%.2e' % x for x in real[key]]}  orders {np.round(orders(real[key], real['dm']), 2)}")
    print(f"   exact values at 1601: TE psi_m {real['exact'][-1][0]:+.4f}, psi_mm {real['exact'][-1][1]:+.4f}; "
          f"LE psi_m {real['exact'][-1][2]:+.4f}, psi_mm {real['exact'][-1][3]:+.4f}")
    for row in coefficient_comparison(cascade):
        print(f"   coefficients {row['n_passage']}x41 {row['variant']:8s}: center {row['center']:+.4f}, largest |c| "
              f"{row['largest']:.4f} at {row['largest_node']}, sum |off| / |center| {row['dominance']:.3f}, nodes "
              f"{len(row['nodes'])}, stations {sorted({n[0] for n in row['nodes']})}")

    figures = {
        "01_le_coordinate_curves": figure_coordinate_curves(blade, base, "LE", chord_results[(101, 41)]["straight_chord_intersections"]),
        "02_te_coordinate_curves": figure_coordinate_curves(blade, base, "TE", chord_results[(101, 41)]["straight_chord_intersections"]),
        "03_te_straight_chord_vs_coordinate_curve": figure_03(blade, [next(x for x in results if (x["n_passage"], x["n_eta"]) == key)
                                                                      for key in ((401, 41), (101, 41))], cascade),
        "04_n401_mixed_stencil_mapping": figure_04(blade, r401, info),
    }
    cell_cases = [next(x for x in results if (x["n_passage"], x["n_eta"]) == key)
                  for key in ((61, 41), (101, 41), (201, 41), (401, 41), (401, 121))]
    figures["05_le_mapped_cells"] = figure_cells(blade, cell_cases, "LE")
    figures["06_te_mapped_cells"] = figure_cells(blade, cell_cases, "TE")
    figures["07_mapping_derivatives"] = figure_07(base)
    figures["08_resolution_validity_summary"] = figure_08(results)
    figures["09_te_central_vs_forward_error"] = figure_09(lattice, real)
    if args.save is not None:
        args.save.mkdir(parents=True, exist_ok=True)
        for name, fig in figures.items():
            fig.savefig(args.save / f"{name}.png", dpi=140)
            print(f"wrote {args.save / (name + '.png')}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
