"""
Audit of where grid.py puts the first LE and last TE blade-wall stations, over stagger and
n_passage, against the authoritative C2 curves; and diagnostic candidate rules for the two
transitions. Diagnostic only: no production module is changed, nothing is frozen.

    python scripts/audit_wall_station_topology.py --save outputs/wall_station_topology_audit --no-show

The blade is the default 65-(12A10)10 (cl0 1.2, t/c 0.10, Kovach TE, C2 surfaces), solidity
1, inlet angle 45 deg, one-chord extensions; only the stagger changes (13 .. 53 deg).

Required topology (the user's): the first blade station cuts the reference blade on one
suction (SS) and one pressure (PS) surface, and so does the last; SS and PS are the two C2
curves of geometry.py, which meet at its LE point (u = 0) and TE point (u = 1).

Independent of grid.py
    Each C2 curve is split at its own turning points in m (dense sign scan of m'(u), brentq
    on the curve's derivative); a station line x = m crosses every monotone branch whose m
    range holds m once (brentq on the curve). Each root: curve (classification B, the
    authoritative surface), u, m, y, tangent angle, slope dy/dm, and the contour it lies on
    between the nose tip and the tail end (upper: the top of a cut, lower: its bottom).
    Classification A is grid.py's own Grid.wall_identity (surface and u of each wall node).

Sections (printed; also written as CSV next to the figures)
    1  reference points per stagger: construction LE/TE point (blade-frame origin and
       (1, 0) turned by the stagger), chord-axis LE/TE, the SS/PS branch points (curve ends),
       the LE/TE circle centres, the nose tip and tail end, grid.py's references, and the
       two-root limits m2_LE, m2_TE (the m range in which a station has one SS + one PS root)
    2  grid.py's station rule against make_grid (formula reproduced exactly)
    3  master tables LE and TE (production make_grid, 5 staggers x 10 n_passage)
    4  violations, and the reproduced failures explained
    5  dense refinement sweep n_passage 31 .. 401 (production stations; formula checked
       against make_grid every 10th n)
    6  candidate rules over the dense sweep, their spacing and mapping jump
    7  grid quality of the candidates (listed n_passage, n_eta 21)
    figures 01 .. 09
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

from potential_flow_solver import grid as gr  # noqa: E402
from potential_flow_solver.config import GridSettings  # noqa: E402
from potential_flow_solver.geometry import CascadeSpec, make_cascade, rotate_by_stagger  # noqa: E402

STAGGERS = (13.0, 23.0, 33.0, 43.0, 53.0)
N_LIST = (31, 41, 51, 61, 81, 101, 151, 201, 301, 401)
N_DENSE = tuple(range(31, 402))
INLET = math.radians(45.0)
N_ETA = 21
GAMMA = 0.5             # margin of the candidate rules beyond the window ends, in nominal dm (half a cell)
AMBIGUOUS_U = 1e-9      # a root this close to u = 0 or 1 sits on the LE / TE point itself
SUCTION, PRESSURE = "#eb6834", "#2a78d6"       # as scripts/plot_cascade.py
INK, MUTED, GRIDLINE, OK, BAD, ALT = "#1f2328", "#6e7781", "#c9ced6", "#1baf7a", "#cf222e", "#8250df"
CLASS_COLORS = {"VALID_SS_PS": OK, "SS_ONLY": SUCTION, "PS_ONLY": PRESSURE, "NO_INTERSECTION": MUTED,
                "AMBIGUOUS_NOSE_ROOTS": ALT, "MULTIPLE_EXTRA_ROOTS": BAD, "PERIODIC_CUTS_BLADE": BAD}


def spec_at(stagger_deg: float) -> CascadeSpec:
    return CascadeSpec(blade_type="NACA65", solidity=1.0, stagger_deg=stagger_deg, cl0=1.2, t_max_over_c=0.10,
                       surface_continuity="C2")


# --- independent geometry --------------------------------------------------------------------------

@dataclass(frozen=True)
class Root:
    curve: str          # "SS" | "PS" (classification B: the C2 curve the root lies on)
    u: float
    m: float
    y: float
    angle: float        # tangent angle, degrees from +m, curve direction LE -> TE
    slope: float        # dy/dm along the curve
    contour: str        # "upper" | "lower": side of the nose-tip / tail-end split


class Geometry:
    """The cascade at one stagger, read from its C2 curves only."""

    def __init__(self, stagger_deg: float):
        self.stagger_deg = stagger_deg
        self.cascade = make_cascade(spec_at(stagger_deg))
        self.a = self.cascade.active
        self.pitch = float(self.a.pitch)
        self.curves = {"SS": self.cascade.active_curves.suction, "PS": self.cascade.active_curves.pressure}
        self.turning, self.branches = {}, []
        for name, c in self.curves.items():
            u = np.linspace(0.0, 1.0, 200001)
            dx = c.derivative(u, 1)[:, 0]
            roots = []
            for k in np.flatnonzero(np.sign(dx[:-1]) * np.sign(dx[1:]) < 0):
                roots.append(brentq(lambda s, c=c: float(c.derivative(np.array([s]), 1)[0, 0]), u[k], u[k + 1],
                                    xtol=1e-15))
            self.turning[name] = roots
            splits = [0.0] + roots + [1.0]
            for u0, u1 in zip(splits[:-1], splits[1:]):
                m0, m1 = c.evaluate(np.array([u0, u1]))[:, 0]
                self.branches.append((name, u0, u1, float(m0), float(m1)))
        ends = [(m0, name, u0) for name, u0, u1, m0, m1 in self.branches] + \
               [(m1, name, u1) for name, u0, u1, m0, m1 in self.branches]
        self.m_tip, self.tip_curve, self.tip_u = min(ends)
        self.m_tail, self.tail_curve, self.tail_u = max(ends)
        self.tip = self.point(self.tip_curve, self.tip_u)
        self.tail = self.point(self.tail_curve, self.tail_u)
        # construction points: the blade frame's (0, 0) and (1, 0) turned by the stagger, the cascade's own points
        st = self.cascade.stagger
        self.construction_le = rotate_by_stagger(np.array([[0.0, 0.0]]), st)[0] * self.a.chord
        self.construction_te = rotate_by_stagger(np.array([[1.0, 0.0]]), st)[0] * self.a.chord
        self.le_point = np.asarray(self.a.leading_edge_point, dtype=float)
        self.te_point = np.asarray(self.a.trailing_edge_point, dtype=float)
        self.le_center, self.te_center = np.asarray(self.a.leading_edge_center), np.asarray(self.a.trailing_edge_center)
        self.r_le, self.r_te = float(self.a.leading_edge_radius), float(self.a.trailing_edge_radius)
        self.coincidence_le = max(float(np.abs(p - self.le_point).max()) for p in (
            self.construction_le, self.a.suction_surface[0], self.a.pressure_surface[0], self.a.camberline[0],
            self.point("SS", 0.0), self.point("PS", 0.0)))
        self.coincidence_te = max(float(np.abs(p - self.te_point).max()) for p in (
            self.construction_te, self.a.suction_surface[-1], self.a.pressure_surface[-1],
            self.point("SS", 1.0), self.point("PS", 1.0)))
        self.m_le, self.m_te = float(self.le_point[0]), float(self.te_point[0])
        self.m_mid = 0.5 * (self.m_le + self.m_te)
        # which side of the tip/tail split is the upper contour: the higher crossing at mid-chord
        self._upper_tag = None
        mids = self.roots(self.m_mid, tag=False)
        top = max(mids, key=lambda r: r.y)
        self._upper_tag = self._arc(top.curve, top.u)
        self._two_root_limits()

    def point(self, curve: str, u: float) -> np.ndarray:
        return np.asarray(self.curves[curve].evaluate(np.array([u]))[0], dtype=float)

    def _s(self, curve: str, u: float) -> float:
        """Position on the closed outline, 0 .. 1: SS from the TE (0) to the LE (0.5), then PS to the TE (1)."""
        return 0.5 * (1.0 - u) if curve == "SS" else 0.5 + 0.5 * u

    def _arc(self, curve: str, u: float) -> bool:
        """True on the arc that runs from the nose tip, increasing s, to the tail end."""
        s_tip, s_tail = self._s(self.tip_curve, self.tip_u), self._s(self.tail_curve, self.tail_u)
        return (self._s(curve, u) - s_tip) % 1.0 < (s_tail - s_tip) % 1.0

    def roots(self, m: float, tag: bool = True) -> list[Root]:
        out = []
        for name, u0, u1, m0, m1 in self.branches:
            if min(m0, m1) < m < max(m0, m1):
                c = self.curves[name]
                u = brentq(lambda s, c=c: float(c.evaluate(np.array([s]))[0, 0]) - m, u0, u1, xtol=1e-15)
                p = c.evaluate(np.array([u]))[0]
                t = c.derivative(np.array([u]), 1)[0]
                contour = ("upper" if self._arc(name, u) == self._upper_tag else "lower") if tag else "?"
                out.append(Root(curve=name, u=float(u), m=float(p[0]), y=float(p[1]),
                                angle=math.degrees(math.atan2(t[1], t[0])),
                                slope=float(t[1] / t[0]) if t[0] != 0 else math.inf, contour=contour))
        return sorted(out, key=lambda r: r.y)

    def counts(self, m: float) -> tuple[int, int]:
        ss = ps = 0
        for name, u0, u1, m0, m1 in self.branches:
            if min(m0, m1) < m < max(m0, m1):
                ss += name == "SS"
                ps += name == "PS"
        return ss, ps

    def classify(self, m: float) -> tuple[str, list[Root]]:
        rs = self.roots(m)
        n_ss, n_ps = sum(r.curve == "SS" for r in rs), sum(r.curve == "PS" for r in rs)
        if not rs:
            return "NO_INTERSECTION", rs
        if any(min(r.u, 1.0 - r.u) < AMBIGUOUS_U for r in rs):
            return "AMBIGUOUS_NOSE_ROOTS", rs
        if len(rs) > 2:
            return "MULTIPLE_EXTRA_ROOTS", rs
        return {(1, 1): "VALID_SS_PS", (2, 0): "SS_ONLY", (0, 2): "PS_ONLY"}.get((n_ss, n_ps), "MULTIPLE_EXTRA_ROOTS"), rs

    def _two_root_limits(self) -> None:
        """Root counts change only at branch-end m values. m2_LE .. m2_TE is the run of
        intervals around mid-chord with one SS and one PS root; m_first_two_root the first
        such interval met from the nose tip (equal to m2_LE unless a wiggle intervenes)."""
        crit = sorted({m for _, _, _, m0, m1 in self.branches for m in (m0, m1)})
        intervals = [(a, b, self.counts(0.5 * (a + b))) for a, b in zip(crit[:-1], crit[1:])]
        self.intervals = intervals
        k = next(i for i, (a, b, _) in enumerate(intervals) if a <= self.m_mid < b)
        if intervals[k][2] != (1, 1):
            raise RuntimeError("mid-chord station does not have one SS and one PS root")
        lo = hi = k
        while lo > 0 and intervals[lo - 1][2] == (1, 1):
            lo -= 1
        while hi < len(intervals) - 1 and intervals[hi + 1][2] == (1, 1):
            hi += 1
        self.m2_le, self.m2_te = intervals[lo][0], intervals[hi][1]
        self.m_first_two_root = next(a for a, b, cnt in intervals if cnt == (1, 1))
        self.m_last_two_root = next(b for a, b, cnt in reversed(intervals) if cnt == (1, 1))


# --- production stations ---------------------------------------------------------------------------

def production_stations(b, n: int) -> np.ndarray:
    """grid.make_grid's station rule, reproduced: one constant dm, exact m_in and m_out."""
    dm_target = (b.m_te - b.m_le) / n
    return np.linspace(b.m_in, b.m_out, max(1, round((b.m_out - b.m_in) / dm_target)) + 1)


def blade_run(row, m: np.ndarray) -> tuple[int, int, bool]:
    """first, last blade station by grid.py's own cut rule (metal > TOUCH), and whether the run is contiguous."""
    blade = np.flatnonzero(row.cut(m).cuts_blade)
    return int(blade[0]), int(blade[-1]), bool(blade.size == blade[-1] - blade[0] + 1)


def piecewise(intervals, D) -> np.ndarray:
    """Uniform stations on each closed interval (a, b) of intervals, about D apart; the gaps
    between consecutive intervals are single cells (the transitions)."""
    parts = []
    for a, b in intervals:
        k = max(1, round((b - a) / D))
        parts.append(np.linspace(a, b, k + 1))
    out = np.concatenate(parts)
    return out[np.concatenate(([True], np.diff(out) > 0.0))]


CANDIDATES = ("BASE", "TWO_ROOT_DROP", "SNAP", "REDIST", "EQ_LE_HALF", "EQ_TIP_HALF", "EQ_MID_HALF", "EQ_LE_GEO")
CANDIDATE_TEXT = {
    "BASE": "production: one uniform lattice from m_in, blade stations = lines that cut the blade",
    "TWO_ROOT_DROP": "lattice kept, stations in (m_tip, m2_LE + g dm) and (m2_TE - g dm, m_tail) removed",
    "SNAP": "lattice kept; the window stations removed and one put at m2_LE + g dm (m2_TE - g dm), unless a kept "
            "station lies within g dm of it",
    "REDIST": "transition [m_tip - g dm, m2_LE + g dm] (and [m2_TE - g dm, m_tail + g dm]) one cell, uniform "
              "stations about dm apart on the three regions",
    "EQ_LE_HALF": "equal shift about the construction LE / TE point, delta = dm / 2",
    "EQ_TIP_HALF": "equal shift about the nose tip / tail end, delta = dm / 2",
    "EQ_MID_HALF": "equal shift about the window midpoint, delta = dm / 2",
    "EQ_LE_GEO": "equal shift about the construction LE / TE point, delta = window + g dm",
}


def candidate_stations(name: str, geo: Geometry, b, n: int) -> np.ndarray:
    base = production_stations(b, n)
    D = float(base[1] - base[0])
    g = GAMMA * D
    if name == "BASE":
        return base
    if name == "TWO_ROOT_DROP":
        drop = ((base > geo.m_tip) & (base < geo.m2_le + g)) | ((base > geo.m2_te - g) & (base < geo.m_tail))
        return base[~drop]
    if name == "SNAP":
        out = base.copy()
        for lo, hi, target in ((geo.m_tip, geo.m2_le + g, geo.m2_le + g), (geo.m2_te - g, geo.m_tail, geo.m2_te - g)):
            window = (out > lo) & (out < hi)
            if window.any():
                kept = out[~window]
                near = np.abs(kept - target).min()
                out = kept if near < g else np.sort(np.append(kept, target))
        return out
    if name == "REDIST":
        a_le, b_le, a_te, b_te = geo.m_tip - g, geo.m2_le + g, geo.m2_te - g, geo.m_tail + g
    else:
        w_le, w_te = geo.m2_le - geo.m_tip, geo.m_tail - geo.m2_te
        if name == "EQ_LE_HALF":
            (r_le, d_le), (r_te, d_te) = (geo.m_le, D / 2), (geo.m_te, D / 2)
        elif name == "EQ_TIP_HALF":
            (r_le, d_le), (r_te, d_te) = (geo.m_tip, D / 2), (geo.m_tail, D / 2)
        elif name == "EQ_MID_HALF":
            (r_le, d_le), (r_te, d_te) = (0.5 * (geo.m_tip + geo.m2_le), D / 2), (0.5 * (geo.m2_te + geo.m_tail), D / 2)
        elif name == "EQ_LE_GEO":
            (r_le, d_le), (r_te, d_te) = (geo.m_le, w_le + g), (geo.m_te, w_te + g)
        else:
            raise ValueError(name)
        a_le, b_le, a_te, b_te = r_le - d_le, r_le + d_le, r_te - d_te, r_te + d_te
    return piecewise(((b.m_in, a_le), (b_le, a_te), (b_te, b.m_out)), D)


def evaluate_stations(geo: Geometry, row, m: np.ndarray, D: float) -> dict:
    f, l, run = blade_run(row, m)
    le_class, le_roots = geo.classify(float(m[f]))
    te_class, te_roots = geo.classify(float(m[l]))
    h = np.diff(m)
    ratios = np.maximum(h[1:] / h[:-1], h[:-1] / h[1:])
    return dict(f=f, l=l, run=run, m_f=float(m[f]), m_l=float(m[l]), m_fm1=float(m[f - 1]), m_lp1=float(m[l + 1]),
                le_class=le_class, te_class=te_class, le_roots=le_roots, te_roots=te_roots,
                valid=run and le_class == "VALID_SS_PS" and te_class == "VALID_SS_PS",
                trans_le=float((m[f] - m[f - 1]) / D), trans_te=float((m[l + 1] - m[l]) / D),
                max_ratio=float(ratios.max()), h_min=float(h.min() / D), h_max=float(h.max() / D),
                wall_offset_le=float((m[f] - geo.m2_le) / D), wall_offset_te=float((geo.m2_te - m[l]) / D),
                periodic_gap_le=float((geo.m_tip - m[f - 1]) / D), periodic_gap_te=float((m[l + 1] - geo.m_tail) / D))


# --- mapping and quality -----------------------------------------------------------------------------

def side_lines(geo: Geometry, b, anchor: str):
    """Lower inlet and exit side lines: production (through the construction LE / TE point) or
    re-anchored through the nose tip / tail end (same angles, gr._side_line itself)."""
    if anchor == "production":
        return (lambda m: gr._inlet_side(b, m)), (lambda m: gr._exit_side(b, m))
    tip, tail = geo.tip, geo.tail
    inlet = lambda m: gr._side_line(m, gr._station_fraction(m, tip[0], b.m_in), tip, b.upstream_beta[-1],  # noqa: E731
                                    b.inlet_angle)[1]
    exit_ = lambda m: gr._side_line(m, gr._station_fraction(m, tail[0], b.m_out), tail, b.downstream_beta[0],  # noqa: E731
                                    b.exit_angle)[1]
    return inlet, exit_


def walls_at(geo: Geometry, b, row, m: np.ndarray, anchor: str):
    """y0, y1 by grid.py's rule (cuts -> passage-facing curve points, else side lines) at any m."""
    m = np.atleast_1d(np.asarray(m, dtype=float))
    cut = row.cut(m)
    blade = np.asarray(cut.cuts_blade)
    inlet, exit_ = side_lines(geo, b, anchor)
    y0 = np.empty_like(m)
    front, back = ~blade & (m < geo.m_mid), ~blade & (m >= geo.m_mid)
    if front.any():
        y0[front] = inlet(m[front])
    if back.any():
        y0[back] = exit_(m[back])
    y1 = y0 + geo.pitch
    if blade.any():
        idx = np.flatnonzero(blade)
        top, bottom = row.passage_sides(cut, idx)
        y0[idx], y1[idx] = top, bottom + geo.pitch
    return y0, y1, blade


def mapping_jumps(geo: Geometry, b, row, anchor: str) -> dict:
    """The C0 jump of y(m, eta) at the switch: walls -> the curve point itself in the limit
    (the cut closes at the tip / tail), so the jump is side(m_tip) - y_tip exactly; checked
    by one-sided values at 1e-10 (the walls approach the tip like sqrt(d))."""
    inlet, exit_ = side_lines(geo, b, anchor)
    exact_le = float(inlet(np.array([geo.m_tip]))[0] - geo.tip[1])
    exact_te = float(exit_(np.array([geo.m_tail]))[0] - geo.tail[1])
    eps = 1e-10
    y0, y1, _ = walls_at(geo, b, row, np.array([geo.m_tip - eps, geo.m_tip + eps, geo.m_tail - eps, geo.m_tail + eps]),
                         anchor)
    return dict(le=exact_le, te=exact_te, le_numeric=float(y0[0] - y0[1]), te_numeric=float(y0[3] - y0[2]),
                le_upper=float(y1[0] - y1[1]), te_upper=float(y1[3] - y1[2]))


def quality(geo: Geometry, b, row, m: np.ndarray, anchor: str = "production") -> dict:
    y0, y1, _ = walls_at(geo, b, row, m, anchor)
    eta = np.linspace(0.0, 1.0, N_ETA)
    S = y1 - y0
    Y = y0[:, None] + eta[None, :] * S[:, None]
    X = np.broadcast_to(m[:, None], Y.shape)
    P = np.stack([X, Y], axis=-1)
    c = [P[:-1, :-1], P[1:, :-1], P[1:, 1:], P[:-1, 1:]]
    area = 0.5 * sum(c[k][..., 0] * c[(k + 1) % 4][..., 1] - c[(k + 1) % 4][..., 0] * c[k][..., 1] for k in range(4))
    cross = []
    for k in range(4):
        u, v = c[(k + 1) % 4] - c[k], c[(k + 2) % 4] - c[(k + 1) % 4]
        cross.append(u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0])
    folded = int((np.min(np.stack(cross), axis=0) <= 0).sum())
    h = np.diff(m)[:, None]
    height = (eta[1] - eta[0]) * 0.5 * (S[:-1] + S[1:])[:, None]
    aspect = np.maximum(h / height, height / h)
    return dict(S_min=float(S.min()), area_min=float(area.min()), negative=int((area <= 0).sum()), folded=folded,
                aspect_max=float(aspect.max()))


# --- figures -------------------------------------------------------------------------------------------

def style(ax, title: str) -> None:
    ax.set_title(title, fontsize=9, color=INK, loc="left")
    ax.tick_params(labelsize=7, colors=MUTED)
    for spine in ax.spines.values():
        spine.set_color(GRIDLINE)


def draw_blade(ax, geo: Geometry, shift: float = 0.0, lw: float = 2.0) -> None:
    u = np.linspace(0.0, 1.0, 6001)
    for name, color in (("SS", SUCTION), ("PS", PRESSURE)):
        p = geo.curves[name].evaluate(u)
        ax.plot(p[:, 0], p[:, 1] + shift, color=color, lw=lw, zorder=3)
    loop = np.vstack([geo.curves["SS"].evaluate(u[::-1]), geo.curves["PS"].evaluate(u[1:])])
    ax.fill(loop[:, 0], loop[:, 1] + shift, color="#eef0f3", zorder=1)


def station_panel(ax, geo: Geometry, case: dict, side: str) -> None:
    """Close-up of the nose (LE) or tail (TE) with the stations, all roots of the first (last)
    blade station, the selected wall nodes (the upper one moved down one pitch onto the
    reference blade) and the reference points."""
    g, m = case["grid"], case["grid"].m
    f, l = g.first_blade_i, g.last_blade_i
    draw_blade(ax, geo)
    D = g.dm
    if side == "LE":
        i, before, after = f, f - 1, f + 1
        ext, ref, ref_name, center, r = geo.tip, geo.le_point, "construction LE = chord-axis LE = SS/PS point", geo.le_center, geo.r_le
        w = geo.m2_le - geo.m_tip
        x0, x1 = geo.m_tip - 1.3 * max(D, w), geo.m_tip + 2.6 * max(D, w)
    else:
        i, before, after = l, l + 1, l - 1
        ext, ref, ref_name, center, r = geo.tail, geo.te_point, "construction TE = chord-axis TE = SS/PS point", geo.te_center, geo.r_te
        w = geo.m_tail - geo.m2_te
        x0, x1 = geo.m_tail - 2.6 * max(D, 4 * r), geo.m_tail + 1.3 * max(D, 4 * r)
    for k, ls, lw, name in ((before, "--", 1.0, "last periodic" if side == "LE" else "first exit periodic"),
                            (i, "-", 1.6, "first blade station" if side == "LE" else "last blade station"),
                            (after, ":", 1.0, "second blade station" if side == "LE" else "penultimate blade station")):
        ax.axvline(m[k], color=INK if k == i else MUTED, ls=ls, lw=lw, zorder=2, label=name)
    two = geo.m2_le if side == "LE" else geo.m2_te
    ax.axvspan(min(ext[0], two), max(ext[0], two), color=BAD, alpha=0.08, zorder=0,
               label="no SS + PS pair here (nose tip .. LE point)" if side == "LE" else "no SS + PS pair (TE point .. tail end)")
    th = np.linspace(0, 2 * math.pi, 300)
    ax.plot(center[0] + r * np.cos(th), center[1] + r * np.sin(th), color=MUTED, lw=0.6, ls=(0, (2, 2)), zorder=2)
    ax.plot(*center, "+", color=MUTED, ms=7, zorder=6)
    ax.plot(*ref, "o", ms=7, mfc="white", mec=INK, mew=1.6, zorder=8, label=ref_name)
    ax.plot(*ext, ">" if side == "LE" else "<", ms=8, color=INK, zorder=8,
            label="nose tip (most upstream point)" if side == "LE" else "tail end (most downstream point)")
    cls, roots = geo.classify(float(m[i]))
    for rt in roots:
        ax.plot(rt.m, rt.y, "o", ms=6, color=SUCTION if rt.curve == "SS" else PRESSURE, mec=INK, mew=0.8, zorder=9)
    lower, upper = g.nodes[i, 0], g.nodes[i, -1] - np.array([0.0, geo.pitch])
    ax.plot(*lower, "o", ms=15, mfc="none", mec=OK, mew=2.0, zorder=10, label="selected lower-wall node (top of cut)")
    ax.plot(*upper, "s", ms=15, mfc="none", mec=ALT, mew=2.0, zorder=10,
            label="selected upper-wall node - pitch (bottom of cut)")
    k = i - f
    wid = g.wall_identity
    short = {"suction": "SS", "pressure": "PS"}
    text = (f"{cls}: roots " + ", ".join(f"{rt.curve} u {rt.u:.4f}" for rt in roots) +
            f"\nlabels lower {short[wid.lower.surface[k]]}, upper {short[wid.upper.surface[k]]}; i {i}; "
            f"(m - m_{'tip' if side == 'LE' else 'tail'})/dm {(m[i] - ext[0]) / D:+.3f}")
    ys = [rt.y for rt in roots] + [ext[1], ref[1], lower[1], upper[1]]
    yc = 0.5 * (min(ys) + max(ys))
    half = 0.5 * max(x1 - x0, max(ys) - min(ys) + 0.4 * (x1 - x0))
    ax.set_xlim(x0, x1)
    ax.set_ylim(yc - half, yc + half)
    ax.set_aspect("equal", adjustable="box")
    style(ax, f"{geo.stagger_deg:.0f} deg, n_passage {case['n']}, {side}, dm {D:.2e} c\n{text}")
    ax.title.set_fontsize(7.5)
    ax.title.set_color(BAD if cls != "VALID_SS_PS" else INK)


def figure_01(cases) -> plt.Figure:
    ns = (61, 101, 201, 401)
    fig, axes = plt.subplots(2, len(ns), figsize=(5.2 * len(ns), 10.5))
    for col, n in enumerate(ns):
        case = cases[(33.0, n)]
        station_panel(axes[0, col], case["geo"], case, "LE")
        station_panel(axes[1, col], case["geo"], case, "TE")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.suptitle("01  Current algorithm at 33 deg: first (top) and last (bottom) blade station, all C2 roots (orange SS, "
                 "blue PS), the selected wall nodes and the reference points", fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    return fig


def figure_03(cases) -> plt.Figure:
    ns = (101, 401)
    fig, axes = plt.subplots(len(ns), len(STAGGERS), figsize=(4.6 * len(STAGGERS), 5.2 * len(ns)))
    for r, n in enumerate(ns):
        for c, st in enumerate(STAGGERS):
            case = cases[(st, n)]
            station_panel(axes[r, c], case["geo"], case, "LE")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.suptitle("03  First blade station at 13 .. 53 deg (n_passage 101, 401): orange SS, blue PS, every root of the "
                 "station line, the selected SS (circle) and PS (square) wall nodes; red band = nose tip .. LE point",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    return fig


def figure_02(dense) -> plt.Figure:
    fig, axes = plt.subplots(len(STAGGERS), 2, figsize=(17, 3.0 * len(STAGGERS)), sharex=True)
    for r, st in enumerate(STAGGERS):
        d = dense[st]
        ns = np.array(N_DENSE)
        for c, side in enumerate(("LE", "TE")):
            ax = axes[r, c]
            y = np.array([x[f"off_{side}"] for x in d["base"]])
            w = np.array([x[f"w_{side}"] for x in d["base"]])
            if side == "LE":
                ax.fill_between(ns, -w, 0.0, color=BAD, alpha=0.10, lw=0, label="nose tip .. LE point: SS + SS")
            else:
                ax.fill_between(ns, 0.0, w, color=BAD, alpha=0.10, lw=0, label="TE point .. tail end: one surface twice")
            cls = [x[f"class_{side}"] for x in d["base"]]
            for name, color in CLASS_COLORS.items():
                sel = np.array([k == name for k in cls])
                if sel.any():
                    ax.plot(ns[sel], y[sel], "o", ms=2.6, color=color, label=name)
            listed = np.isin(ns, N_LIST)
            ax.plot(ns[listed], y[listed], "o", ms=6, mfc="none", mec=INK, mew=1.0, label="make_grid (listed n)")
            ax.axhline(0.0, color=INK, lw=0.6)
            n_bad = sum(k != "VALID_SS_PS" for k in cls)
            style(ax, f"{st:.0f} deg, {side}: (m_{'first' if side == 'LE' else 'last'} - m_{'LE' if side == 'LE' else 'TE'}) "
                      f"/ dm; {n_bad} of {len(cls)} n_passage not SS + PS")
        axes[-1, 0].set_xlabel("n_passage", fontsize=8)
        axes[-1, 1].set_xlabel("n_passage", fontsize=8)
    handles, labels = {}, {}
    for ax in axes.ravel():
        for h, lab in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(lab, h)
    fig.legend(list(handles.values()), list(handles.keys()), loc="lower center", ncol=6, fontsize=8, frameon=False)
    fig.suptitle("02  Refinement aliasing of the production stations: first blade station relative to the construction LE "
                 "(left) and last relative to the TE (right), in dm, n_passage 31 .. 401", fontsize=11, color=INK, x=0.01,
                 ha="left")
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    return fig


def reference_panel(ax, geo: Geometry, side: str) -> None:
    draw_blade(ax, geo)
    if side == "LE":
        ext, ref, center, r, two = geo.tip, geo.le_point, geo.le_center, geo.r_le, geo.m2_le
        span = max(0.035, 2.5 * (geo.m2_le - geo.m_tip))
        x0, x1 = geo.m_tip - 0.3 * span, geo.m_tip + 0.9 * span
        chord_dir = (geo.construction_te - geo.construction_le) / np.linalg.norm(geo.construction_te - geo.construction_le)
        ax.plot([ref[0], ref[0] + 0.6 * span * chord_dir[0]], [ref[1], ref[1] + 0.6 * span * chord_dir[1]], color=MUTED,
                lw=0.9, ls="-.", zorder=4, label="chord axis")
    else:
        ext, ref, center, r, two = geo.tail, geo.te_point, geo.te_center, geo.r_te, geo.m2_te
        span = 0.035
        x0, x1 = geo.m_tail - 0.9 * span, geo.m_tail + 0.3 * span
        chord_dir = (geo.construction_le - geo.construction_te) / np.linalg.norm(geo.construction_te - geo.construction_le)
        ax.plot([ref[0], ref[0] + 0.6 * span * chord_dir[0]], [ref[1], ref[1] + 0.6 * span * chord_dir[1]], color=MUTED,
                lw=0.9, ls="-.", zorder=4, label="chord axis")
    th = np.linspace(0, 2 * math.pi, 300)
    ax.plot(center[0] + r * np.cos(th), center[1] + r * np.sin(th), color=MUTED, lw=0.7, ls=(0, (2, 2)), zorder=2,
            label=f"construction {'LE' if side == 'LE' else 'TE'} circle (R = {r:.4f} c)")
    ax.plot(*center, "+", color=MUTED, ms=8, zorder=6)
    ax.plot(*ref, "o", ms=8, mfc="white", mec=INK, mew=1.6, zorder=8,
            label=f"construction = chord-axis = SS/PS {'LE' if side == 'LE' else 'TE'} point")
    ax.plot(*ext, ">" if side == "LE" else "<", ms=9, color=INK, zorder=8,
            label="nose tip" if side == "LE" else "tail end")
    ax.axvline(ext[0], color=INK, lw=0.7, ls=":")
    ax.axvline(two, color=OK, lw=1.0, ls="--", label="two-root limit m2")
    ax.axvspan(min(ext[0], two), max(ext[0], two), color=BAD, alpha=0.10, zorder=0)
    yc = ext[1]
    ax.set_xlim(x0, x1)
    ax.set_ylim(yc - 0.6 * (x1 - x0), yc + 0.6 * (x1 - x0))
    ax.set_aspect("equal", adjustable="box")
    w = (geo.m2_le - geo.m_tip) if side == "LE" else (geo.m_tail - geo.m2_te)
    holder = geo.tip_curve if side == "LE" else geo.tail_curve
    style(ax, f"{geo.stagger_deg:.0f} deg: window {w:.2e} c, {'tip' if side == 'LE' else 'tail end'} on {holder}")


def figure_refs(geos, side: str) -> plt.Figure:
    fig, axes = plt.subplots(2, 3, figsize=(17, 11))
    for ax, st in zip(axes.ravel()[:5], STAGGERS):
        reference_panel(ax, geos[st], side)
    axes.ravel()[0].legend(fontsize=7, loc="upper right" if side == "LE" else "upper left")
    ax = axes.ravel()[5]
    st = np.array(STAGGERS)
    if side == "LE":
        w = np.array([geos[s].m2_le - geos[s].m_tip for s in STAGGERS])
        ax.plot(st, w, "o-", color=SUCTION, label="m_LE - m_tip (window, tip on SS)")
    else:
        w = np.array([geos[s].m_tail - geos[s].m2_te for s in STAGGERS])
        for s, val in zip(STAGGERS, w):
            ax.plot(s, val, "o", color=SUCTION if geos[s].tail_curve == "SS" else PRESSURE, ms=8)
        ax.plot(st, w, "-", color=MUTED, label="m_tail - m_TE (window; marker colour = surface of the tail end)")
    for n, ls in ((101, "--"), (401, ":")):
        D = np.array([(geos[s].m_te - geos[s].m_le) / n for s in STAGGERS])
        ax.plot(st, D, ls, color=INK, label=f"dm at n_passage {n}")
    ax.set_yscale("log")
    ax.set_xlabel("stagger (deg)", fontsize=8)
    ax.legend(fontsize=7)
    style(ax, "window against dm: wherever dm < window, every grid fails")
    fig.suptitle(f"0{4 if side == 'LE' else 5}  {side} reference points per stagger: construction / chord-axis / SS-PS point "
                 f"(one point, proved), construction circle, {'nose tip' if side == 'LE' else 'tail end'} and the two-root "
                 f"limit", fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def class_map(values) -> np.ndarray:
    order = ("VALID_SS_PS", "SS_ONLY", "PS_ONLY", "AMBIGUOUS_NOSE_ROOTS", "MULTIPLE_EXTRA_ROOTS", "NO_INTERSECTION",
             "PERIODIC_CUTS_BLADE")
    return np.array([[order.index(v) for v in row] for row in values])


CLASS_CMAP = ListedColormap([OK, SUCTION, PRESSURE, ALT, BAD, MUTED, BAD])


def figure_06(dense) -> plt.Figure:
    variants = ("EQ_LE_HALF", "EQ_TIP_HALF", "EQ_MID_HALF", "EQ_LE_GEO", "REDIST")
    fig, axes = plt.subplots(len(variants) + 1, 1, figsize=(17, 2.1 * (len(variants) + 1) + 1.5))
    for ax, v in zip(axes, variants):
        grid = np.array([[0 if x["valid"] else (1 if x["le_class"] != "VALID_SS_PS" and x["te_class"] == "VALID_SS_PS"
                                                 else (2 if x["le_class"] == "VALID_SS_PS" else 3))
                          for x in dense[st]["cand"][v]] for st in STAGGERS])
        ax.imshow(grid, aspect="auto", cmap=ListedColormap([OK, SUCTION, PRESSURE, BAD]), vmin=0, vmax=3,
                  extent=(N_DENSE[0] - 0.5, N_DENSE[-1] + 0.5, len(STAGGERS) - 0.5, -0.5), interpolation="nearest")
        ax.set_yticks(range(len(STAGGERS)), [f"{s:.0f}" for s in STAGGERS])
        n_ok = int((grid == 0).sum())
        style(ax, f"{v}: {CANDIDATE_TEXT[v]} -> {n_ok} of {grid.size} grids valid (green; orange LE fails, blue TE "
                  f"fails, red both)")
    ax = axes[-1]
    for st, color in zip(STAGGERS, plt.cm.viridis(np.linspace(0, 0.9, len(STAGGERS)))):
        for v, ls in (("EQ_LE_GEO", "--"), ("REDIST", "-")):
            ax.plot(N_DENSE, [x["trans_le"] for x in dense[st]["cand"][v]], ls, color=color, lw=1.1,
                    label=f"{st:.0f} deg {v}")
    ax.set_xlabel("n_passage", fontsize=8)
    ax.set_ylabel("LE transition / dm", fontsize=8)
    ax.legend(fontsize=6.5, ncol=5)
    style(ax, "LE transition cell (last periodic -> first wall) in dm: equal shift about the LE point is a full window "
              "longer than about the window midpoint (REDIST)")
    fig.suptitle("06  The equal-shift hypothesis over stagger and n_passage", fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def figure_07(dense) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(17, 10))
    names = CANDIDATES
    ax = axes[0, 0]
    total = len(STAGGERS) * len(N_DENSE)
    le = [sum(x["le_class"] == "VALID_SS_PS" for st in STAGGERS for x in dense[st]["cand"][v]) / total for v in names]
    te = [sum(x["te_class"] == "VALID_SS_PS" for st in STAGGERS for x in dense[st]["cand"][v]) / total for v in names]
    pos = np.arange(len(names))
    ax.bar(pos - 0.2, le, 0.38, color=SUCTION, label="LE SS + PS")
    ax.bar(pos + 0.2, te, 0.38, color=PRESSURE, label="TE SS + PS")
    ax.set_xticks(pos, names, rotation=20, fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7)
    style(ax, f"fraction of the {total} grids (5 staggers x n_passage 31 .. 401) with the required topology")
    for ax, key, title in ((axes[0, 1], "trans_le", "LE transition cell / dm"),
                           (axes[1, 0], "max_ratio", "largest ratio of neighbouring cell lengths"),
                           (axes[1, 1], "wall_offset_le", "(m_first_wall - m2_LE) / dm: < 0 = first wall in the window")):
        for v, color in zip(names, plt.cm.tab10(np.linspace(0, 0.9, len(names)))):
            vals = [x[key] for x in dense[53.0]["cand"][v]]
            ax.plot(N_DENSE, vals, lw=1.0, color=color, label=v)
        ax.set_xlabel("n_passage (53 deg, the widest window)", fontsize=8)
        ax.legend(fontsize=6.5, ncol=2)
        style(ax, title)
    fig.suptitle("07  Candidate rules compared", fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def figure_08(dense, recommended: str) -> plt.Figure:
    fig, axes = plt.subplots(4, 1, figsize=(17, 11))
    for ax, (v, side) in zip(axes, (("BASE", "le"), ("BASE", "te"), (recommended, "le"), (recommended, "te"))):
        values = [[x[f"{side}_class"] for x in dense[st]["cand"][v]] for st in STAGGERS]
        ax.imshow(class_map(values), aspect="auto", cmap=CLASS_CMAP, vmin=0, vmax=6, interpolation="nearest",
                  extent=(N_DENSE[0] - 0.5, N_DENSE[-1] + 0.5, len(STAGGERS) - 0.5, -0.5))
        ax.set_yticks(range(len(STAGGERS)), [f"{s:.0f}" for s in STAGGERS])
        bad = sum(k != "VALID_SS_PS" for row in values for k in row)
        style(ax, f"{v}, {side.upper()}: {bad} of {len(values) * len(values[0])} grids not SS + PS "
                  f"(green valid, orange SS only, blue PS only)")
    axes[-1].set_xlabel("n_passage", fontsize=8)
    fig.suptitle(f"08  Refinement invariance: production (top two) against {recommended} (bottom two), stagger x n_passage",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def figure_09(geos, rows, jumps, recommended: str) -> plt.Figure:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10.5))
    ax = axes[0, 0]
    st = np.array(STAGGERS)
    for key, ls, label in (("production", "-o", "production side lines (through the LE / TE point)"),
                           ("tip", "--s", "side lines through the nose tip / tail end")):
        ax.plot(st, [max(abs(jumps[s][key]["le"]), 1e-17) for s in STAGGERS], ls, color=SUCTION, label=f"LE, {label}")
        ax.plot(st, [max(abs(jumps[s][key]["te"]), 1e-17) for s in STAGGERS], ls, color=PRESSURE, label=f"TE, {label}")
    ax.set_yscale("log")
    ax.set_xlabel("stagger (deg)", fontsize=8)
    ax.legend(fontsize=6.5)
    style(ax, "|C0 jump of y(m, eta)| at the switch (the same for every station rule; 1e-17 = exactly 0)")
    axes[1, 0].axis("off")
    axes[1, 0].text(0.0, 1.0, "\n".join(
        [f"{s:.0f} deg: production LE {jumps[s]['production']['le']:+.3e} c, TE {jumps[s]['production']['te']:+.3e} c;"
         f" tip anchored LE {jumps[s]['tip']['le']:+.0e}, TE {jumps[s]['tip']['te']:+.0e}" for s in STAGGERS] +
        ["", "The switch from side lines to walls must sit at the nose tip / tail end:",
         "for m between the tip and the LE point the line cuts the blade, so a side-line",
         "strip there would hold nose metal. The stations decide only which part of the",
         "mapping is sampled, not where it jumps; the anchor of the side lines decides",
         "the jump."]), va="top", fontsize=8.5, color=INK, family="monospace", transform=axes[1, 0].transAxes)
    for k, s in enumerate((33.0, 53.0)):
        geo, b, row = geos[s], rows[s]["b"], rows[s]["row"]
        for j, side in enumerate(("LE", "TE")):
            ax = axes[k, 1 + j]
            draw_blade(ax, geo, lw=1.4)
            m_st = candidate_stations(recommended, geo, b, 101)
            if side == "LE":
                ms = np.linspace(geo.m_tip - 0.03, geo.m_tip + 0.03, 3001)
            else:
                ms = np.linspace(geo.m_tail - 0.03, geo.m_tail + 0.03, 3001)
            ext = geo.tip if side == "LE" else geo.tail
            seen = []
            for anchor, color, label in (("production", BAD, "production side lines"), ("tip", OK, "tip / tail anchored")):
                y0, y1, blade = walls_at(geo, b, row, ms, anchor)
                seen += [y0.min(), y0.max()]
                for part in (ms < ext[0], ms > ext[0]):
                    ax.plot(ms[part], y0[part], color=color, lw=1.6, label=f"eta = 0, {label}" if part[0] else None)
                    ax.plot(ms[part], y1[part] - geo.pitch, color=color, lw=1.0, ls="--",
                            label=f"eta = 1 - pitch, {label}" if part[0] else None)
            for x in m_st[(m_st > ms[0]) & (m_st < ms[-1])]:
                ax.axvline(x, color=GRIDLINE, lw=0.6, zorder=0)
            ax.plot(*ext, ">" if side == "LE" else "<", color=INK, ms=8, zorder=9)
            ax.set_xlim(ms[0], ms[-1])
            half = max(0.04, 0.5 * (max(seen) - min(seen)) + 0.004)
            centre = 0.5 * (max(seen) + min(seen))
            ax.set_ylim(centre - half, centre + half)
            ax.set_aspect("equal", adjustable="box")
            if k == 0 and j == 0:
                ax.legend(fontsize=6.5, loc="lower right")
            style(ax, f"{s:.0f} deg {side}: the mapping's walls\n(grey: {recommended} stations, n_passage 101)")
    fig.suptitle("09  C0 jump of the mapping: it sits at the nose tip / tail end whatever the stations; only the side-line "
                 "anchor decides it", fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


# --- main ----------------------------------------------------------------------------------------------

def fmt_roots(roots) -> str:
    return "; ".join(f"{r.curve}/{r.contour} u={r.u:.5f} y={r.y:+.5f} ang={r.angle:+.1f}" for r in roots)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--save", type=Path, default=None)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    if args.save is not None:
        args.save.mkdir(parents=True, exist_ok=True)

    geos, rows = {}, {}
    print("1. Reference points per stagger (cascade frame, chords; m = x)")
    for st in STAGGERS:
        geo = Geometry(st)
        geos[st] = geo
        g0 = gr.make_grid(geo.cascade, GridSettings(n_passage=101, n_eta=3), INLET)
        b, row = g0.boundaries, gr._row_of(g0.boundaries)
        rows[st] = dict(b=b, row=row)
        print(f"   {st:.0f} deg: construction LE = chord-axis LE = SS/PS LE point = camberline start (coincide to "
              f"{geo.coincidence_le:.1e}): m {geo.m_le:+.6f}, y {geo.le_point[1]:+.6f}; LE circle centre m "
              f"{geo.le_center[0]:+.6f} (R {geo.r_le:.4f})")
        print(f"            nose tip m {geo.m_tip:+.6f} y {geo.tip[1]:+.6f} on {geo.tip_curve} u {geo.tip_u:.6f} "
              f"(grid.py's tip {float(row.lo_m.min()):+.6f}, difference {abs(float(row.lo_m.min()) - geo.m_tip):.1e}); "
              f"window m_LE - m_tip {geo.m_le - geo.m_tip:.4e}")
        print(f"            two-root limit m2_LE {geo.m2_le:+.6f} (first SS + PS from the tip {geo.m_first_two_root:+.6f}); "
              f"grid.py references: m_le {b.m_le:+.6f} (lattice anchor m_in = m_le - 1 c, dm_target, side-line junction), "
              f"blade-station switch = the cut, i.e. m_tip")
        print(f"            construction TE = chord-axis TE = SS/PS TE point (coincide to {geo.coincidence_te:.1e}): m "
              f"{geo.m_te:+.6f} y {geo.te_point[1]:+.6f}; TE circle centre m {geo.te_center[0]:+.6f} (R {geo.r_te:.4f})")
        print(f"            tail end m {geo.m_tail:+.6f} y {geo.tail[1]:+.6f} on {geo.tail_curve} u {geo.tail_u:.6f} (grid.py "
              f"{float(row.hi_m.max()):+.6f}); window m_tail - m_TE {geo.m_tail - geo.m_te:.4e}; m2_TE {geo.m2_te:+.6f} "
              f"(last SS + PS {geo.m_last_two_root:+.6f})")
        print(f"            turning points in m: SS u {[round(u, 6) for u in geo.turning['SS']]}, PS u "
              f"{[round(u, 6) for u in geo.turning['PS']]}; tangent at the LE point: SS "
              f"{math.degrees(math.atan2(*geo.curves['SS'].derivative(np.array([0.0]), 1)[0][::-1])):+.2f} deg, PS "
              f"{math.degrees(math.atan2(*geo.curves['PS'].derivative(np.array([0.0]), 1)[0][::-1])):+.2f} deg; root counts "
              f"(SS, PS) by interval near the LE: " + ", ".join(f"[{a:+.5f}, {bb:+.5f}) {cnt}" for a, bb, cnt in
                                                                 geo.intervals[:3]))

    print("\n2. grid.py's station rule (make_grid): dm_target = (m_te - m_le) / n_passage, N = round((m_out - m_in) / "
          "dm_target), m = linspace(m_in, m_out, N + 1), m_in = m_le - 1 c, m_out = m_te + 1 c; blade stations = lines "
          "whose cut holds > 1e-12 c of metal, i.e. m_tip < m < m_tail; walls = top / bottom crossing of the cut")
    cases, master_le, master_te = {}, [], []
    checks = 0
    for st in STAGGERS:
        geo, b, row = geos[st], rows[st]["b"], rows[st]["row"]
        for n in N_LIST:
            g = gr.make_grid(geo.cascade, GridSettings(n_passage=n, n_eta=N_ETA), INLET)
            m = g.m
            if not np.array_equal(m, production_stations(b, n)):
                raise RuntimeError("station formula differs from make_grid")
            f, l, _ = blade_run(row, m)
            if (f, l) != (g.first_blade_i, g.last_blade_i):
                raise RuntimeError("blade run differs from make_grid")
            checks += 1
            cases[(st, n)] = dict(geo=geo, grid=g, n=n)
            D = g.dm
            for side, i, k in (("LE", f, 0), ("TE", l, l - f)):
                cls, roots = geo.classify(float(m[i]))
                lower_y, upper_y = g.nodes[i, 0, 1], g.nodes[i, -1, 1] - geo.pitch
                sel_lo = min(roots, key=lambda r: abs(r.y - lower_y))
                sel_up = min(roots, key=lambda r: abs(r.y - upper_y))
                wid = g.wall_identity
                a_lo, a_up = wid.lower.surface[k], wid.upper.surface[k]
                name = {"SS": "suction", "PS": "pressure"}
                row_out = dict(stagger=st, n_passage=n, dm=D, i=i,
                               m_extreme=geo.m_tip if side == "LE" else geo.m_tail,
                               m_construction=geo.m_le if side == "LE" else geo.m_te,
                               m_periodic=float(m[i - 1] if side == "LE" else m[i + 1]), m_wall=float(m[i]),
                               offset_dm=float((m[i] - (geo.m_tip if side == "LE" else geo.m_tail)) / D),
                               offset_construction_dm=float((m[i] - (geo.m_le if side == "LE" else geo.m_te)) / D),
                               roots=len(roots), ss=sum(r.curve == "SS" for r in roots),
                               ps=sum(r.curve == "PS" for r in roots), lower_B=sel_lo.curve, upper_B=sel_up.curve,
                               lower_A=a_lo, upper_A=a_up,
                               labels_agree=name[sel_lo.curve] == a_lo and name[sel_up.curve] == a_up,
                               node_error=max(abs(sel_lo.y - lower_y), abs(sel_up.y - upper_y)),
                               u_error=max(abs(sel_lo.u - wid.lower.u[k]), abs(sel_up.u - wid.upper.u[k])),
                               selected_top_bottom=(sel_lo is roots[-1]) and (sel_up is roots[0]),
                               cls=cls, roots_text=fmt_roots(roots))
                (master_le if side == "LE" else master_te).append(row_out)
    print(f"   formula and blade run reproduce make_grid at all {checks} listed grids")

    for side, table in (("LE", master_le), ("TE", master_te)):
        print(f"\n3. Master table {side}: stagger, n, dm, m_{'nose' if side == 'LE' else 'tail'}, m_construction, "
              f"m_{'last_periodic' if side == 'LE' else 'first_exit'}, m_{'first' if side == 'LE' else 'last'}_wall, "
              f"(m_wall - m_extreme)/dm, (m_wall - m_construction)/dm, roots, SS, PS, selected lower/upper (B), "
              f"production labels (A), A == B, |node - root|, |u_A - u_B|, class")
        for r in table:
            print(f"   {r['stagger']:4.0f} {r['n_passage']:4d} {r['dm']:.3e} {r['m_extreme']:+.6f} {r['m_construction']:+.6f} "
                  f"{r['m_periodic']:+.6f} {r['m_wall']:+.6f} {r['offset_dm']:+7.3f} {r['offset_construction_dm']:+7.3f} "
                  f"{r['roots']} {r['ss']} {r['ps']} {r['lower_B']}/{r['upper_B']} {r['lower_A'][:2].upper()}/"
                  f"{r['upper_A'][:2].upper()} {'yes' if r['labels_agree'] else 'NO'} {r['node_error']:.1e} "
                  f"{r['u_error']:.1e} {r['cls']}")
        if args.save is not None:
            with open(args.save / f"master_{side.lower()}.csv", "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(table[0].keys()))
                writer.writeheader()
                writer.writerows(table)

    print("\n4. Violations (listed grids) and why")
    for side, table in (("LE", master_le), ("TE", master_te)):
        bad = [r for r in table if r["cls"] != "VALID_SS_PS"]
        print(f"   {side}: {len(bad)} of {len(table)}")
        for r in bad:
            geo = geos[r["stagger"]]
            m = cases[(r["stagger"], r["n_passage"])]["grid"].m
            if side == "LE":
                nxt = next(k for k in range(r["i"], len(m)) if m[k] > geo.m2_le)
                why = (f"first lattice station past the tip (m_tip {geo.m_tip:+.6f} < m_f, m_f-1 {r['m_periodic']:+.6f} "
                       f"<= m_tip), but m_f <= m2_LE {geo.m2_le:+.6f}")
            else:
                nxt = next(k for k in range(r["i"], 0, -1) if m[k] < geo.m2_te)
                why = (f"last lattice station before the tail end (m_tail {geo.m_tail:+.6f}), but m_l >= m2_TE "
                       f"{geo.m2_te:+.6f}")
            print(f"     {r['stagger']:.0f} deg n {r['n_passage']} i {r['i']}: {why}; roots {r['roots_text']}; selected "
                  f"lower = top {r['lower_B']} ({r['lower_A']}), upper = bottom {r['upper_B']} ({r['upper_A']}); nearest SS + PS "
                  f"station i {nxt}, m {m[nxt]:+.6f} ({(m[nxt] - r['m_wall']) / r['dm']:+.0f} dm)")

    # dense production sweep and candidates
    print("\n5./6. Dense sweep n_passage 31 .. 401, production and candidate rules")
    dense, verified = {}, 0
    for st in STAGGERS:
        geo, b, row = geos[st], rows[st]["b"], rows[st]["row"]
        dense[st] = dict(base=[], cand={v: [] for v in CANDIDATES})
        for n in N_DENSE:
            base = production_stations(b, n)
            D = float(base[1] - base[0])
            if n % 10 == 0:
                g = gr.make_grid(geo.cascade, GridSettings(n_passage=n, n_eta=3), INLET)
                if not np.array_equal(g.m, base) or blade_run(row, base)[:2] != (g.first_blade_i, g.last_blade_i):
                    raise RuntimeError(f"formula differs from make_grid at {st} deg, n {n}")
                verified += 1
            for v in CANDIDATES:
                ev = evaluate_stations(geo, row, candidate_stations(v, geo, b, n), D)
                dense[st]["cand"][v].append(ev)
            e = dense[st]["cand"]["BASE"][-1]
            dense[st]["base"].append(dict(off_LE=(e["m_f"] - geo.m_le) / D, w_LE=(geo.m_le - geo.m_tip) / D,
                                          off_TE=(e["m_l"] - geo.m_te) / D, w_TE=(geo.m_tail - geo.m_te) / D,
                                          class_LE=e["le_class"], class_TE=e["te_class"]))
    print(f"   production stations checked against make_grid at {verified} dense grids (every 10th n)")
    total = len(N_DENSE)
    print("   candidate, then per stagger: LE valid / TE valid / both of 371; LE transition / dm min..max; max neighbour "
          "ratio; (m_f - m2_LE) / dm min; (m_tip - m_f-1) / dm min")
    for v in CANDIDATES:
        parts = []
        for st in STAGGERS:
            ev = dense[st]["cand"][v]
            parts.append(f"{st:.0f}: {sum(x['le_class'] == 'VALID_SS_PS' for x in ev)}/"
                         f"{sum(x['te_class'] == 'VALID_SS_PS' for x in ev)}/{sum(x['valid'] for x in ev)}, "
                         f"tr {min(x['trans_le'] for x in ev):.2f}..{max(x['trans_le'] for x in ev):.2f}, "
                         f"ratio {max(x['max_ratio'] for x in ev):.2f}, wall {min(x['wall_offset_le'] for x in ev):+.2f}, "
                         f"periodic {min(x['periodic_gap_le'] for x in ev):+.2f}")
        print(f"   {v:13s} " + " | ".join(parts))
    print(f"   (each stagger: {total} n_passage values)")
    for v in ("REDIST", "TWO_ROOT_DROP", "EQ_LE_GEO"):
        ev401 = [dense[st]["cand"][v][N_DENSE.index(401)] for st in STAGGERS]
        print(f"   {v} at n_passage 401: LE transition / dm " + ", ".join(f"{st:.0f}: {x['trans_le']:.2f}" for st, x in
                                                                     zip(STAGGERS, ev401)) +
              "; TE " + ", ".join(f"{st:.0f}: {x['trans_te']:.2f}" for st, x in zip(STAGGERS, ev401)))
    print("   lower bound for ANY rule with SS + PS at both ends: no station in (m_tip, m2_LE] or [m2_TE, m_tail), so "
          "the transition cells are at least window / dm long: at n_passage 401 LE " +
          ", ".join(f"{st:.0f}: {(geos[st].m2_le - geos[st].m_tip) / ((geos[st].m_te - geos[st].m_le) / 401):.2f}"
                    for st in STAGGERS))

    print("\n6b. Mapping jump y(m_switch+) - y(m_switch-) (the switch is the nose tip / tail end for every station rule: a "
          "line between them cuts the blade)")
    jumps = {}
    for st in STAGGERS:
        geo, b, row = geos[st], rows[st]["b"], rows[st]["row"]
        jumps[st] = {anchor: mapping_jumps(geo, b, row, anchor) for anchor in ("production", "tip")}
        p, t = jumps[st]["production"], jumps[st]["tip"]
        inlet, _ = side_lines(geo, b, "production")
        metal_at_le = float(np.ptp([r.y for r in geo.roots(0.5 * (geo.m_tip + geo.m_le))]))
        print(f"   {st:.0f} deg: production anchor LE {p['le']:+.4e} (one-sided {p['le_numeric']:+.4e}, upper "
              f"{p['le_upper']:+.4e}), TE {p['te']:+.4e} ({p['te_numeric']:+.4e}); tip/tail anchor LE {t['le']:+.1e} "
              f"(one-sided at 1e-10: {t['le_numeric']:+.1e}, the sqrt approach), TE {t['te']:+.1e} ({t['te_numeric']:+.1e}); "
              f"a switch at m_LE instead would leave {metal_at_le:.2e} c of nose metal inside the passage at the window "
              f"midpoint")

    print("\n7. Grid quality of the candidates (n_eta 21; S, areas in chords; production side lines)")
    for st in STAGGERS:
        geo, b, row = geos[st], rows[st]["b"], rows[st]["row"]
        for v in ("BASE", "TWO_ROOT_DROP", "SNAP", "REDIST", "EQ_LE_GEO"):
            parts = []
            for n in (61, 101, 201, 401):
                ms = candidate_stations(v, geo, b, n)
                q = quality(geo, b, row, ms)
                ev = dense[st]["cand"][v][N_DENSE.index(n)]
                parts.append(f"n {n}: S {q['S_min']:.4f}, area {q['area_min']:.1e}, <=0 {q['negative']}, folded "
                             f"{q['folded']}, aspect {q['aspect_max']:.0f}, trans {ev['trans_le']:.2f}/{ev['trans_te']:.2f}, "
                             f"ratio {ev['max_ratio']:.2f}")
            print(f"   {st:.0f} deg {v:13s} " + " | ".join(parts))
        q = quality(geo, b, row, candidate_stations("REDIST", geo, b, 401), anchor="tip")
        print(f"   {st:.0f} deg REDIST + tip-anchored side lines, n 401: S {q['S_min']:.4f}, area {q['area_min']:.1e}, "
              f"folded {q['folded']}")

    recommended = "REDIST"
    figures = {
        "01_current_algorithm_stagger33": figure_01(cases),
        "02_first_wall_roots_by_refinement": figure_02(dense),
        "03_stagger13_23_33_43_53": figure_03(cases),
        "04_le_reference_points": figure_refs(geos, "LE"),
        "05_te_reference_points": figure_refs(geos, "TE"),
        "06_equal_shift_candidate": figure_06(dense),
        "07_candidate_comparison": figure_07(dense),
        "08_refinement_invariance": figure_08(dense, recommended),
        "09_mapping_jump_comparison": figure_09(geos, rows, jumps, recommended),
    }
    if args.save is not None:
        for name, fig in figures.items():
            fig.savefig(args.save / f"{name}.png", dpi=130)
            print(f"wrote {args.save / (name + '.png')}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
