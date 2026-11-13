"""
Numerical audit of the REDIST station rule (diagnostic only; no production module changed).

    python scripts/audit_redist_numerics.py --save outputs/redist_numerical_audit --no-show

REDIST is reused exactly as audited by scripts/audit_wall_station_topology.py
(candidate_stations("REDIST", ...), GAMMA = 0.5): with D the production lattice step
(m_out - m_in) / round((m_out - m_in) / dm_target), dm_target = (m_te - m_le) / n_passage,
    a_LE = m_tip - D/2,  b_LE = m_le + D/2,  a_TE = m_te - D/2,  b_TE = m_tail + D/2,
    stations = linspace(m_in, a_LE, k1 + 1) U linspace(b_LE, a_TE, k2 + 1) U linspace(b_TE, m_out, k3 + 1),
    k = max(1, round(length / D)) per region;
the two-root limits m2_LE, m2_TE equal m_le, m_te exactly for this blade (previous audit).
The walls are grid.py's own rule at any m (passage-facing contours where a line cuts the
blade, side lines elsewhere); REDIST and MINIMAL use side lines re-anchored at the nose tip
and tail end (same angles, grid._side_line itself), the production grid its own.

MINIMAL (the one alternative): the transition centred on the window, width
max(D, w + 0.2 D) (margins 0.1 D past the tip and the LE point), and the neighbouring cells
graded geometrically (ratio 1.5) from the transition down to D, then uniform.

Everything m-differenced here uses nonuniform formulas derived for the actual coordinates
(three points m_i-1, m_i, m_i+1, h1 = m_i - m_i-1, h2 = m_i+1 - m_i):
    f_m  ~ -h2/(h1(h1+h2)) f_i-1 + (h2-h1)/(h1 h2) f_i + h1/(h2(h1+h2)) f_i+1,    error +h1 h2 f'''/6
    f_mm ~ 2 [f_i-1/(h1(h1+h2)) - f_i/(h1 h2) + f_i+1/(h2(h1+h2))],             error (h2-h1) f'''/3 + ...
checked against Fornberg weights; one-sided alternatives inside a region by Fornberg.

Sections (printed) and figures 01 .. 12 (outputs/redist_numerical_audit/).
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import solve_banded

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", Path(__file__).resolve().parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import matplotlib.pyplot as plt  # noqa: E402

import audit_wall_station_topology as wt  # noqa: E402
import link_geometry as lg  # noqa: E402
from turbolenza.b2b import grid as gr  # noqa: E402
from turbolenza.b2b.coefficients import build_coefficients  # noqa: E402
from turbolenza.b2b.config import GridSettings  # noqa: E402
from turbolenza.b2b.metrics import build_metrics  # noqa: E402

import logging  # noqa: E402

logging.getLogger("matplotlib.axes._base").setLevel(logging.ERROR)   # equal aspect on data limits: x widened, by design
INK, MUTED, GRIDLINE = wt.INK, wt.MUTED, wt.GRIDLINE
SUCTION, PRESSURE = wt.SUCTION, wt.PRESSURE
COLORS = {"PROD": "#cf222e", "REDIST": "#1baf7a", "MINIMAL": "#8250df"}
CASES = [(s, n) for s in (33.0, 43.0, 53.0) for n in (61, 101, 201, 401)] + \
        [(s, n) for s in (13.0, 23.0) for n in (61, 101, 401)]
SEQUENCE = (61, 101, 201, 401, 801, 1601)
N_ETA = 41
EPS_MIN, Q_MIN = 0.1, 1.5          # MINIMAL: margins in D, grading ratio


# --- stations --------------------------------------------------------------------------------------

def graded_region(start: float, stop: float, D: float, left_T: float | None, right_T: float | None) -> np.ndarray:
    """Stations on [start, stop]: cells graded by Q_MIN from left_T / Q_MIN down to D at the
    left end (if left_T), up to right_T / Q_MIN at the right end (if right_T), uniform between."""
    def ladder(T):
        out, s = [], (T / Q_MIN if T is not None else 0.0)
        while T is not None and s > D:
            out.append(s)
            s /= Q_MIN
        return out
    left, right = ladder(left_T), ladder(right_T)[::-1]
    middle = (stop - start) - sum(left) - sum(right)
    if middle <= 0:
        raise RuntimeError("region too short for the grading")
    k = max(1, round(middle / D))
    cells = left + [middle / k] * k + right
    return np.concatenate(([start], start + np.cumsum(cells)[:-1], [stop]))


def stations(kind: str, geo, b, n: int) -> np.ndarray:
    if kind == "PROD":
        return wt.production_stations(b, n)
    if kind == "REDIST":
        return wt.candidate_stations("REDIST", geo, b, n)
    base = wt.production_stations(b, n)
    D = float(base[1] - base[0])
    w_le, w_te = geo.m2_le - geo.m_tip, geo.m_tail - geo.m2_te
    T_le, T_te = max(D, w_le + 2 * EPS_MIN * D), max(D, w_te + 2 * EPS_MIN * D)
    c_le, c_te = 0.5 * (geo.m_tip + geo.m2_le), 0.5 * (geo.m2_te + geo.m_tail)
    a_le, b_le, a_te, b_te = c_le - T_le / 2, c_le + T_le / 2, c_te - T_te / 2, c_te + T_te / 2
    parts = [graded_region(b.m_in, a_le, D, None, T_le if T_le > Q_MIN * D else None),
             graded_region(b_le, a_te, D, T_le if T_le > Q_MIN * D else None, T_te if T_te > Q_MIN * D else None),
             graded_region(b_te, b.m_out, D, T_te if T_te > Q_MIN * D else None, None)]
    return np.concatenate(parts)


ANCHOR = {"PROD": "production", "REDIST": "tip", "MINIMAL": "tip"}


# --- nonuniform differences ------------------------------------------------------------------------

def fornberg(x0: float, x: np.ndarray, k: int) -> np.ndarray:
    """Weights of the k-th derivative at x0 from values at x (Fornberg 1988)."""
    n = len(x)
    c = np.zeros((n, k + 1))
    c1, c4 = 1.0, x[0] - x0
    c[0, 0] = 1.0
    for i in range(1, n):
        mn = min(i, k)
        c2, c5, c4 = 1.0, c4, x[i] - x0
        for j in range(i):
            c3 = x[i] - x[j]
            c2 *= c3
            if j == i - 1:
                for s in range(mn, 0, -1):
                    c[i, s] = c1 * (s * c[i - 1, s - 1] - c5 * c[i - 1, s]) / c2
                c[i, 0] = -c1 * c5 * c[i - 1, 0] / c2
            for s in range(mn, 0, -1):
                c[j, s] = (c4 * c[j, s] - s * c[j, s - 1]) / c3
            c[j, 0] = c4 * c[j, 0] / c3
        c1 = c2
    return c[:, k]


def d1(m: np.ndarray, f: np.ndarray) -> np.ndarray:
    """f_m at the interior stations (first axis), nonuniform three-point central; NaN at the ends."""
    h1, h2 = (m[1:-1] - m[:-2]), (m[2:] - m[1:-1])
    shape = (-1,) + (1,) * (f.ndim - 1)
    h1, h2 = h1.reshape(shape), h2.reshape(shape)
    out = np.full(f.shape, np.nan)
    out[1:-1] = (-h2 / (h1 * (h1 + h2)) * f[:-2] + (h2 - h1) / (h1 * h2) * f[1:-1] + h1 / (h2 * (h1 + h2)) * f[2:])
    return out


def d2(m: np.ndarray, f: np.ndarray) -> np.ndarray:
    h1, h2 = (m[1:-1] - m[:-2]), (m[2:] - m[1:-1])
    shape = (-1,) + (1,) * (f.ndim - 1)
    h1, h2 = h1.reshape(shape), h2.reshape(shape)
    out = np.full(f.shape, np.nan)
    out[1:-1] = 2.0 * (f[:-2] / (h1 * (h1 + h2)) - f[1:-1] / (h1 * h2) + f[2:] / (h2 * (h1 + h2)))
    return out


def d1_regions(m: np.ndarray, f: np.ndarray, runs) -> np.ndarray:
    """f_m inside each region on its own: central inside, three-point one-sided (Fornberg) at its ends."""
    out = np.empty(f.shape)
    for start, stop in runs:
        mm, ff = m[start:stop], f[start:stop]
        part = d1(mm, ff)
        for idx, nodes in ((0, (0, 1, 2)), (len(mm) - 1, (len(mm) - 3, len(mm) - 2, len(mm) - 1))):
            wgt = fornberg(mm[idx], mm[list(nodes)], 1)
            part[idx] = np.tensordot(wgt, ff[list(nodes)], axes=(0, 0))
        out[start:stop] = part
    return out


def check_formulas() -> tuple[float, float, float]:
    rng = np.random.default_rng(1)
    worst = [0.0, 0.0, 0.0]
    for _ in range(200):
        h1, h2 = rng.uniform(0.01, 1.0, 2)
        x = np.array([-h1, 0.0, h2])
        w1, w2 = fornberg(0.0, x, 1), fornberg(0.0, x, 2)
        mine1 = np.array([-h2 / (h1 * (h1 + h2)), (h2 - h1) / (h1 * h2), h1 / (h2 * (h1 + h2))])
        mine2 = 2.0 * np.array([1 / (h1 * (h1 + h2)), -1 / (h1 * h2), 1 / (h2 * (h1 + h2))])
        worst[0] = max(worst[0], np.abs(w1 - mine1).max() / np.abs(w1).max())
        worst[1] = max(worst[1], np.abs(w2 - mine2).max() / np.abs(w2).max())
        q = 0.3 - 1.2 * x + 2.5 * x * x            # exact for quadratics
        worst[2] = max(worst[2], abs(mine1 @ q + 1.2), abs(mine2 @ q - 5.0))
    return tuple(worst)


# --- grids, mapping, metrics -----------------------------------------------------------------------

class Case:
    def __init__(self, geo, b, row, kind: str, n: int, n_eta: int = N_ETA, metrics_needed: bool = True):
        self.geo, self.b, self.row, self.kind, self.n = geo, b, row, kind, n
        self.m = stations(kind, geo, b, n)
        base = wt.production_stations(b, n)
        self.D = float(base[1] - base[0])
        self.eta = np.linspace(0.0, 1.0, n_eta)
        self.y0, self.y1, blade = wt.walls_at(geo, b, row, self.m, ANCHOR[kind])
        idx = np.flatnonzero(blade)
        self.f, self.l = int(idx[0]), int(idx[-1])
        self.contiguous = bool(idx.size == self.l - self.f + 1)
        self.S = self.y1 - self.y0
        self.Y = self.y0[:, None] + self.eta[None, :] * self.S[:, None]
        self.runs = ((0, self.f), (self.f, self.l + 1), (self.l + 1, len(self.m)))
        if metrics_needed:
            self._metrics()

    def side(self, which: str):
        inlet, exit_ = wt.side_lines(self.geo, self.b, ANCHOR[self.kind])
        return inlet if which == "inlet" else exit_

    def _metrics(self) -> None:
        """tan(beta) and d tan(beta)/dm of the two walls at every station, exact: C2 curve
        slopes at the wall roots on blade stations, the side line's own angle elsewhere."""
        geo, m = self.geo, self.m
        n = len(m)
        tb0, tb1, dtb0, dtb1 = (np.empty(n) for _ in range(4))
        for i in range(n):
            if self.f <= i <= self.l:
                roots = geo.roots(float(m[i]))
                top, bottom = roots[-1], roots[0]
                for rt, tb, dtb in ((top, tb0, dtb0), (bottom, tb1, dtb1)):
                    c = geo.curves[rt.curve]
                    t1 = c.derivative(np.array([rt.u]), 1)[0]
                    t2 = c.derivative(np.array([rt.u]), 2)[0]
                    tb[i] = t1[1] / t1[0]
                    dtb[i] = (t1[0] * t2[1] - t1[1] * t2[0]) / t1[0] ** 3
            else:
                line = self.side("inlet" if i < self.f else "exit")
                h = 1e-6
                y = line(np.array([m[i] - 2 * h, m[i] - h, m[i], m[i] + h, m[i] + 2 * h]))
                slope = (y[0] - 8 * y[1] + 8 * y[3] - y[4]) / (12 * h)
                curv = (-y[0] + 16 * y[1] - 30 * y[2] + 16 * y[3] - y[4]) / (12 * h * h)
                tb0[i] = tb1[i] = slope
                dtb0[i] = dtb1[i] = curv
        self.tb0, self.tb1, self.dtb0, self.dtb1 = tb0, tb1, dtb0, dtb1
        self.tb = (1.0 - self.eta)[None, :] * tb0[:, None] + self.eta[None, :] * tb1[:, None]
        # the same slopes by nonuniform region-wise differences of the wall y (what an adapted metrics.py would do)
        self.tb0_fd, self.tb1_fd = d1_regions(m, self.y0, self.runs), d1_regions(m, self.y1, self.runs)

    def coefficients(self, rho: np.ndarray, across: bool = True, S=None, tb=None) -> dict:
        """A .. E with mdot' = 1; B_m and C_m nonuniform central over the whole m range
        (across the transitions, like coefficients.py) or region-wise; eta differences as
        coefficients.py. S, tb override the exact metrics (for the validation against
        coefficients.py with production's own metrics)."""
        S = self.S if S is None else S
        tb = self.tb if tb is None else tb
        cos2 = 1.0 / (1.0 + tb ** 2)
        A = 1.0 / (S[:, None] * rho * cos2)
        B = tb / rho
        C = S[:, None] / rho
        blade = np.zeros(len(self.m), dtype=bool)
        blade[self.f:self.l + 1] = True
        deta = self.eta[1] - self.eta[0]
        dme = lambda F: _d_deta(F, deta, blade)  # noqa: E731
        if across:
            Bm, Cm = _d_dm_nonuniform(self.m, B), _d_dm_nonuniform(self.m, C)
        else:
            Bm, Cm = d1_regions(self.m, B, self.runs), d1_regions(self.m, C, self.runs)
        return dict(A=A, B=B, C=C, D=dme(A) - Bm, E=Cm - dme(B))


def _d_dm_nonuniform(m, f):
    out = d1(m, f)
    for idx, nodes in ((0, (0, 1, 2)), (len(m) - 1, (len(m) - 3, len(m) - 2, len(m) - 1))):
        out[idx] = np.tensordot(fornberg(m[idx], m[list(nodes)], 1), f[list(nodes)], axes=(0, 0))
    return out


def _d_deta(f, deta, blade):
    d = np.empty_like(f)
    d[:, 1:-1] = (f[:, 2:] - f[:, :-2]) / (2.0 * deta)
    wall = f[blade]
    d[blade, 0] = (-3.0 * wall[:, 0] + 4.0 * wall[:, 1] - wall[:, 2]) / (2.0 * deta)
    d[blade, -1] = (3.0 * wall[:, -1] - 4.0 * wall[:, -2] + wall[:, -3]) / (2.0 * deta)
    periodic = f[~blade]
    d[~blade, 0] = d[~blade, -1] = (periodic[:, 1] - periodic[:, -2]) / (2.0 * deta)
    return d


def rho_field(case, kind: str) -> np.ndarray:
    if kind == "uniform":
        return np.ones((len(case.m), len(case.eta)))
    M, H = np.meshgrid(case.m, case.eta, indexing="ij")
    return 1.0 + 0.1 * np.sin(2.0 * M + 0.3) + 0.05 * np.cos(2.0 * math.pi * H)


# --- manufactured fields ---------------------------------------------------------------------------

FIELDS = {     # f, f_m, f_mm, f_meta, f_eta
    "F1": (lambda m, e: np.sin(4 * m) + 0.5 * m ** 2 + np.cos(2 * np.pi * e),
           lambda m, e: 4 * np.cos(4 * m) + m + 0 * e,
           lambda m, e: -16 * np.sin(4 * m) + 1 + 0 * e,
           lambda m, e: 0 * m * e,
           lambda m, e: -2 * np.pi * np.sin(2 * np.pi * e) + 0 * m),
    "F2": (lambda m, e: np.sin(3 * m + 0.4) * np.cos(2 * np.pi * e) + 0.3 * m ** 3,
           lambda m, e: 3 * np.cos(3 * m + 0.4) * np.cos(2 * np.pi * e) + 0.9 * m ** 2,
           lambda m, e: -9 * np.sin(3 * m + 0.4) * np.cos(2 * np.pi * e) + 1.8 * m,
           lambda m, e: -6 * np.pi * np.cos(3 * m + 0.4) * np.sin(2 * np.pi * e),
           lambda m, e: -2 * np.pi * np.sin(3 * m + 0.4) * np.sin(2 * np.pi * e)),
    "F3": (lambda m, e: np.exp(0.8 * m) * (1 + 0.2 * np.sin(2 * np.pi * e)),
           lambda m, e: 0.8 * np.exp(0.8 * m) * (1 + 0.2 * np.sin(2 * np.pi * e)),
           lambda m, e: 0.64 * np.exp(0.8 * m) * (1 + 0.2 * np.sin(2 * np.pi * e)),
           lambda m, e: 0.8 * np.exp(0.8 * m) * 0.4 * np.pi * np.cos(2 * np.pi * e),
           lambda m, e: np.exp(0.8 * m) * 0.4 * np.pi * np.cos(2 * np.pi * e)),
}


def manufactured(m: np.ndarray, eta: np.ndarray, field: str) -> dict:
    """Absolute errors (max over interior eta) of f_m, f_mm, f_meta at every interior station.
    f_meta is the nonuniform m difference of the exact f_eta, so that it measures the m
    differencing only (the eta difference is uniform central, second order in deta, the
    same on every grid here)."""
    F, Fm, Fmm, Fme, Fe = FIELDS[field]
    M, H = np.meshgrid(m, eta, indexing="ij")
    f = F(M, H)
    q = Fe(M, H)
    em = np.abs(d1(m, f) - Fm(M, H))[:, 1:-1].max(axis=1)
    emm = np.abs(d2(m, f) - Fmm(M, H))[:, 1:-1].max(axis=1)
    eme = np.abs(d1(m, q) - Fme(M, H))[:, 1:-1].max(axis=1)
    scale = dict(m=np.abs(Fm(M, H)).max(), mm=np.abs(Fmm(M, H)).max(), me=max(np.abs(Fme(M, H)).max(), 1e-300))
    return dict(m=em, mm=emm, me=eme, scale=scale)


def bvp_error(m: np.ndarray) -> np.ndarray:
    """1D model u'' = f on the stations, u = sin(4m) + 0.5 m^2, Dirichlet ends, nonuniform
    three-point scheme: the nodal error |u_h - u|."""
    u = np.sin(4 * m) + 0.5 * m ** 2
    rhs = -16 * np.sin(4 * m) + 1.0
    h1, h2 = m[1:-1] - m[:-2], m[2:] - m[1:-1]
    lo, di, up = 2 / (h1 * (h1 + h2)), -2 / (h1 * h2), 2 / (h2 * (h1 + h2))
    n = len(m) - 2
    ab = np.zeros((3, n))
    ab[0, 1:] = up[:-1]
    ab[1] = di
    ab[2, :-1] = lo[1:]
    r = rhs[1:-1].copy()
    r[0] -= lo[0] * u[0]
    r[-1] -= up[-1] * u[-1]
    uh = np.concatenate(([u[0]], solve_banded((1, 1), ab, r), [u[-1]]))
    return np.abs(uh - u)


# --- the smooth physical field on the real mapping -------------------------------------------------

ALPHA = math.radians(40.0)


def psi_phys(x, y, pitch):
    s = y - x * math.tan(ALPHA)
    return s / pitch + 0.05 * np.sin(2.0 * math.pi * s / pitch) * np.cos(1.3 * x)


def own_side_derivative(case: Case, i: int, eta: float, order: int = 1) -> float:
    """psi_m (order 1) or psi_mm (order 2) at station i on its own side's smooth mapping
    (side line continued analytically on a periodic station; the walls on a blade station,
    h within the wall's smooth part)."""
    geo, m0 = case.geo, float(case.m[i])
    if case.f <= i <= case.l:
        h = min(2e-5, 0.05 * (m0 - geo.m_tip), 0.05 * (geo.m_tail - m0))
        ms = m0 + h * np.arange(-2, 3)
        y0, y1, _ = wt.walls_at(geo, case.b, case.row, ms, ANCHOR[case.kind])
    else:
        h = 5e-4
        ms = m0 + h * np.arange(-2, 3)
        y0 = case.side("inlet" if i < case.f else "exit")(ms)
        y1 = y0 + geo.pitch
    v = psi_phys(ms, y0 + eta * (y1 - y0), geo.pitch)
    if order == 1:
        return float((v[0] - 8 * v[1] + 8 * v[3] - v[4]) / (12 * h))
    return float((-v[0] + 16 * v[1] - 30 * v[2] + 16 * v[3] - v[4]) / (12 * h * h))


def physical_errors(case: Case) -> dict:
    """|psi_m| and |psi_mm| errors (max over interior eta) at the corner stations: central
    nonuniform across the transition, and (psi_m) region-wise one-sided; the exact values on
    each station's own side."""
    P = psi_phys(case.m[:, None], case.Y, case.geo.pitch)
    central, second = d1(case.m, P), d2(case.m, P)
    region = d1_regions(case.m, P, case.runs)
    out = {}
    for name, i in (("f-1", case.f - 1), ("f", case.f), ("f+1", case.f + 1), ("l", case.l), ("l+1", case.l + 1)):
        ec = er = e2 = 0.0
        for j in range(1, len(case.eta) - 1):
            ex = own_side_derivative(case, i, float(case.eta[j]))
            ex2 = own_side_derivative(case, i, float(case.eta[j]), order=2)
            ec, er = max(ec, abs(central[i, j] - ex)), max(er, abs(region[i, j] - ex))
            e2 = max(e2, abs(second[i, j] - ex2))
        out[name] = (ec, er, e2)
    return out


# --- mapping through the windows -------------------------------------------------------------------

def continuity(case: Case) -> dict:
    geo, b, row = case.geo, case.b, case.row
    out = {}
    for name, point in (("m_tip", geo.m_tip), ("m_le", geo.m_le), ("m_te", geo.m_te), ("m_tail", geo.m_tail)):
        res = {}
        for eps in (1e-5, 1e-7):
            ms = np.array([point - 2 * eps, point - eps, point + eps, point + 2 * eps])
            y0, y1, _ = wt.walls_at(geo, b, row, ms, ANCHOR[case.kind])
            res[eps] = dict(y0_left=y0[1], y0_right=y0[2], y1_left=y1[1], y1_right=y1[2],
                            s0_left=(y0[1] - y0[0]) / eps, s0_right=(y0[3] - y0[2]) / eps,
                            s1_left=(y1[1] - y1[0]) / eps, s1_right=(y1[3] - y1[2]) / eps)
        out[name] = res
    return out


def window_alternatives(case: Case, blade: lg.Blade) -> dict:
    """Three ways through the LE window (m_tip, m_le) between the last periodic and the
    first wall station: (a) the passage-facing contours (grid.py's rule), (b) the side lines
    continued to the first wall station, (c) straight node-to-node segments. Metal inside
    each boundary curve, and the tiling defect: periodicity needs upper(m) - pitch to be the
    reference blade's bottom contour wherever the line cuts it."""
    geo, b, row = case.geo, case.b, case.row
    a, c = float(case.m[case.f - 1]), float(case.m[case.f])
    ms = np.linspace(a, c, 4001)[1:-1]
    y0c, y1c, _ = wt.walls_at(geo, b, row, ms, ANCHOR[case.kind])
    inlet = case.side("inlet")
    ys = inlet(ms)
    y0a, y0b = case.y0[case.f - 1], case.y0[case.f]
    y1a, y1b = case.y1[case.f - 1], case.y1[case.f]
    t = (ms - a) / (c - a)
    lin0, lin1 = y0a + t * (y0b - y0a), y1a + t * (y1b - y1a)
    cut = row.cut(ms)
    bottom = np.where(cut.cuts_blade, cut.height[:, 0], np.nan)

    def metal(y, copy):
        """m extent of the curve inside metal, tested 1e-9 c into the passage (a contour point
        itself lies on the outline)."""
        pts = np.column_stack([ms, y + (1e-9 if copy == 0 else -1e-9)])
        return float(blade.paths[copy].contains_points(pts).mean() * (c - a))

    def tiling(upper):
        d = np.abs(upper - geo.pitch - bottom)
        return float(np.nanmax(d)) if np.any(np.isfinite(d)) else 0.0

    out = {}
    for name, lower, upper in (("contour", y0c, y1c), ("side line continued", ys, ys + geo.pitch),
                               ("straight segments", lin0, lin1)):
        out[name] = dict(metal_lower=metal(lower, 0), metal_upper=metal(upper, 1), tiling=tiling(upper))
    return out


def arc(curve, u0: float, u1: float) -> float:
    u = np.linspace(u0, u1, 4001)
    d = curve.derivative(u, 1)
    s = np.hypot(d[:, 0], d[:, 1])
    return float(np.sum(0.5 * (s[1:] + s[:-1]) * np.diff(u)))


def unresolved(case: Case) -> dict:
    """Blade surface between the two wall nodes of the first (last) blade station, around the
    nose (tail): through the LE (TE) point when the nodes are SS + PS, around the tip on one
    surface when both are on the same surface."""
    geo = case.geo
    total = arc(geo.curves["SS"], 0, 1) + arc(geo.curves["PS"], 0, 1)
    out = {}
    for side, i, end in (("LE", case.f, 0.0), ("TE", case.l, 1.0)):
        roots = geo.roots(float(case.m[i]))
        top, bottom = roots[-1], roots[0]
        if top.curve != bottom.curve:
            length = sum(arc(geo.curves[r.curve], min(r.u, end), max(r.u, end)) for r in (top, bottom))
        else:
            length = arc(geo.curves[top.curve], min(top.u, bottom.u), max(top.u, bottom.u))
        out[side] = (length, 100 * length / total, f"{bottom.curve}+{top.curve}")
    return out


def quality(case: Case) -> dict:
    P = np.stack([np.broadcast_to(case.m[:, None], case.Y.shape), case.Y], axis=-1)
    c = [P[:-1, :-1], P[1:, :-1], P[1:, 1:], P[:-1, 1:]]
    area = 0.5 * sum(c[k][..., 0] * c[(k + 1) % 4][..., 1] - c[(k + 1) % 4][..., 0] * c[k][..., 1] for k in range(4))
    cross = []
    for k in range(4):
        u, v = c[(k + 1) % 4] - c[k], c[(k + 2) % 4] - c[(k + 1) % 4]
        cross.append(u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0])
    folded = int((np.min(np.stack(cross), axis=0) <= 0).sum())
    h = np.diff(case.m)
    r = h[1:] / h[:-1]
    return dict(S_min=float(case.S.min()), area_min=float(area.min()), area_max=float(area.max()),
                negative=int((area <= 0).sum()), folded=folded, h_min=float(h.min() / case.D),
                h_max=float(h.max() / case.D), r_max=float(r.max()), r_inv=float((1 / r).max()),
                t_le=float((case.m[case.f] - case.m[case.f - 1]) / case.D),
                t_te=float((case.m[case.l + 1] - case.m[case.l]) / case.D))


# --- figures -----------------------------------------------------------------------------------------

def style(ax, title):
    ax.set_title(title, fontsize=8.5, color=INK, loc="left")
    ax.tick_params(labelsize=7, colors=MUTED)
    for sp in ax.spines.values():
        sp.set_color(GRIDLINE)


def nose_panel(ax, geo, cases: dict, side: str, span: float) -> None:
    wt.draw_blade(ax, geo, lw=1.6)
    ext = geo.tip if side == "LE" else geo.tail
    ref = geo.le_point if side == "LE" else geo.te_point
    lo, hi = (geo.m_tip, geo.m_le) if side == "LE" else (geo.m_te, geo.m_tail)
    ax.axvspan(lo, hi, color="#cf222e", alpha=0.10, zorder=0)
    for kind, case in cases.items():
        sel = (case.m > ext[0] - span) & (case.m < ext[0] + span)
        for x in case.m[sel]:
            ax.axvline(x, color=COLORS[kind], lw=0.9 if kind != "PROD" else 0.6, alpha=0.8,
                       ls="-" if kind == "REDIST" else (":" if kind == "PROD" else "--"), zorder=2)
    ax.plot(*ext, ">" if side == "LE" else "<", color=INK, ms=8, zorder=9)
    ax.plot(*ref, "o", ms=7, mfc="white", mec=INK, mew=1.5, zorder=9)
    ax.set_xlim(ext[0] - span, ext[0] + span)
    ax.set_ylim(ext[1] - span, ext[1] + span)
    ax.set_aspect("equal", adjustable="box")


def spacing_panel(ax, cases: dict, centre: float, span: float, D: float) -> None:
    for kind, case in cases.items():
        mid = 0.5 * (case.m[1:] + case.m[:-1])
        h = np.diff(case.m) / D
        sel = (mid > centre - span) & (mid < centre + span)
        ax.plot(mid[sel], h[sel], "o-", ms=3, lw=1.0, color=COLORS[kind], label=kind)
    ax.axhline(1.0, color=MUTED, lw=0.7, ls=":")
    ax.set_ylabel("dm / nominal dm", fontsize=8)
    ax.legend(fontsize=7)


def figure_le(geos, rows, stagger, label) -> plt.Figure:
    geo, b, row = geos[stagger], rows[stagger]["b"], rows[stagger]["row"]
    ns = (101, 201, 401)
    fig, axes = plt.subplots(2, 3, figsize=(17, 10), gridspec_kw=dict(height_ratios=(1.4, 1)))
    for col, n in enumerate(ns):
        cases = {k: Case(geo, b, row, k, n, metrics_needed=False) for k in ("PROD", "REDIST", "MINIMAL")}
        D = cases["REDIST"].D
        span = max(4 * D, 2.2 * (geo.m_le - geo.m_tip))
        nose_panel(axes[0, col], geo, cases, "LE", span)
        T = (cases["REDIST"].m[cases["REDIST"].f] - cases["REDIST"].m[cases["REDIST"].f - 1]) / D
        style(axes[0, col], f"{stagger:.0f} deg, n_passage {n}: stations (red dotted PROD, green REDIST, purple "
                            f"dashed MINIMAL)\nREDIST LE transition {T:.2f} dm; window (red band) "
                            f"{(geo.m_le - geo.m_tip) / D:.2f} dm")
        spacing_panel(axes[1, col], cases, 0.5 * (geo.m_tip + geo.m_le), 1.2 * span, D)
        style(axes[1, col], "cell length / nominal dm near the LE")
    fig.suptitle(f"{label}  LE spacing at {stagger:.0f} deg: nose tip (>) and LE point (o), the station lines of the three "
                 f"rules, and the cell lengths", fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def figure_te(geos, rows) -> plt.Figure:
    fig, axes = plt.subplots(2, 3, figsize=(17, 10), gridspec_kw=dict(height_ratios=(1.4, 1)))
    for col, s in enumerate((33.0, 43.0, 53.0)):
        geo, b, row = geos[s], rows[s]["b"], rows[s]["row"]
        cases = {k: Case(geo, b, row, k, 401, metrics_needed=False) for k in ("PROD", "REDIST", "MINIMAL")}
        D = cases["REDIST"].D
        span = max(5 * D, 0.012)
        nose_panel(axes[0, col], geo, cases, "TE", span)
        c = cases["REDIST"]
        style(axes[0, col], f"{s:.0f} deg, n_passage 401, TE: REDIST transition "
                            f"{(c.m[c.l + 1] - c.m[c.l]) / D:.2f} dm; window {(geo.m_tail - geo.m_te) / D:.2f} dm")
        spacing_panel(axes[1, col], cases, 0.5 * (geo.m_te + geo.m_tail), 1.2 * span, D)
        style(axes[1, col], "cell length / nominal dm near the TE")
    fig.suptitle("05  TE spacing at n_passage 401: tail end (<) and TE point (o)", fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


# --- main --------------------------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--save", type=Path, default=None)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    if args.save is not None:
        args.save.mkdir(parents=True, exist_ok=True)

    w1, w2, wq = check_formulas()
    print(f"0. Nonuniform formulas against Fornberg (200 random h1, h2): f_m weights {w1:.1e}, f_mm {w2:.1e} relative; "
          f"exact for quadratics to {wq:.1e}")

    geos, rows = {}, {}
    for s in sorted({c[0] for c in CASES}):
        geo = wt.Geometry(s)
        g0 = gr.make_grid(geo.cascade, GridSettings(n_passage=101, n_eta=3), wt.INLET)
        geos[s], rows[s] = geo, dict(b=g0.boundaries, row=gr._row_of(g0.boundaries))

    print("\n1. Topology, C0 continuity, spacing, quality (n_eta 41)")
    print("   case        rule     f..l      SS+PS LE/TE  window stations  T_LE/dm  T_TE/dm  h min..max/dm  r max  "
          "1/r max  S_min    area min..max        folded")
    table, keep = {}, {}
    for s, n in CASES:
        geo, b, row = geos[s], rows[s]["b"], rows[s]["row"]
        for kind in ("PROD", "REDIST", "MINIMAL"):
            case = Case(geo, b, row, kind, n)
            le, _ = geo.classify(float(case.m[case.f]))
            te, _ = geo.classify(float(case.m[case.l]))
            in_window = int(((case.m > geo.m_tip) & (case.m < geo.m_le)).sum() + ((case.m > geo.m_te) & (case.m < geo.m_tail)).sum())
            q = quality(case)
            table[(s, n, kind)] = dict(case=case, le=le, te=te, in_window=in_window, q=q)
            print(f"   {s:4.0f} {n:4d}  {kind:8s} {case.f:4d}..{case.l:4d}  {le[:5]}/{te[:5]}  {in_window:3d}  "
                  f"{q['t_le']:7.3f}  {q['t_te']:7.3f}  {q['h_min']:.3f}..{q['h_max']:.3f}  {q['r_max']:5.2f}  "
                  f"{q['r_inv']:5.2f}  {q['S_min']:.4f}  {q['area_min']:.2e}..{q['area_max']:.2e}  {q['folded']}")

    print("\n2. Mapping continuity (REDIST, side lines through the tip / tail end): y and one-sided slopes at 1e-5 and 1e-7")
    for s in (13.0, 33.0, 53.0):
        case = table[(s, 401, "REDIST")]["case"]
        cont = continuity(case)
        for name, res in cont.items():
            r5, r7 = res[1e-5], res[1e-7]
            print(f"   {s:.0f} deg {name:6s}: y0 {r7['y0_left']:+.8f} | {r7['y0_right']:+.8f} (jump {r7['y0_right'] - r7['y0_left']:+.1e}), "
                  f"y1 jump {r7['y1_right'] - r7['y1_left']:+.1e}; slope y0 left {r5['s0_left']:+.3f} right {r5['s0_right']:+.2f} "
                  f"(1e-7: {r7['s0_right']:+.1f}); y1 left {r5['s1_left']:+.3f} right {r5['s1_right']:+.2f} "
                  f"(1e-7: {r7['s1_right']:+.1f})")

    print("\n3. Through the LE window: three boundary definitions (metal length inside each boundary curve, c; tiling "
          "defect |upper - pitch - bottom contour|, c)")
    blades = {s: lg.Blade(geos[s].cascade) for s in geos}
    for s in (13.0, 33.0, 53.0):
        for n in (101, 401):
            alt = window_alternatives(table[(s, n, "REDIST")]["case"], blades[s])
            print(f"   {s:.0f} deg n {n}: " + "; ".join(f"{k}: metal {v['metal_lower']:.2e}/{v['metal_upper']:.2e}, tiling "
                                                        f"{v['tiling']:.2e}" for k, v in alt.items()))

    print("\n4. Theoretical minimum and REDIST width, LE (TE)")
    for s in sorted(geos):
        geo = geos[s]
        w_le, w_te = geo.m_le - geo.m_tip, geo.m_tail - geo.m_te
        parts = []
        for n in (101, 401):
            D = table[(s, n, "REDIST")]["case"].D
            parts.append(f"n {n}: min {w_le / D:.2f} dm, REDIST {table[(s, n, 'REDIST')]['q']['t_le']:.2f} dm "
                         f"(x {table[(s, n, 'REDIST')]['q']['t_le'] * D / w_le:.2f}), MINIMAL "
                         f"{table[(s, n, 'MINIMAL')]['q']['t_le']:.2f} dm; TE min {w_te / D:.3f}, REDIST "
                         f"{table[(s, n, 'REDIST')]['q']['t_te']:.2f}")
        print(f"   {s:.0f} deg: window {w_le:.4e} c (TE {w_te:.3e} c); " + " | ".join(parts))

    print("\n5. Unresolved surface between the two wall nodes of the first / last station (arc c, % of the whole surface)")
    for s, n in CASES:
        parts = []
        for kind in ("PROD", "REDIST", "MINIMAL"):
            u = unresolved(table[(s, n, kind)]["case"])
            parts.append(f"{kind} LE {u['LE'][0]:.4f} ({u['LE'][1]:.2f} %, {u['LE'][2]}) TE {u['TE'][0]:.4f} ({u['TE'][1]:.2f} %)")
        print(f"   {s:.0f} deg n {n}: " + " | ".join(parts))

    print("\n6. Metrics: exact wall slopes at the corner stations; region-wise nonuniform FD of y against them; production "
          "metrics.py on the production grid against the exact")
    for s, n in ((33.0, 101), (53.0, 401), (53.0, 101), (13.0, 401)):
        for kind in ("PROD", "REDIST"):
            c = table[(s, n, kind)]["case"]
            f, l = c.f, c.l
            print(f"   {s:.0f} deg n {n} {kind:7s}: S f-1..f+1 {c.S[f - 1]:.4f} {c.S[f]:.4f} {c.S[f + 1]:.4f}; tan b0 "
                  f"{c.tb0[f - 1]:+.3f} {c.tb0[f]:+.3f} {c.tb0[f + 1]:+.3f}; tan b1 {c.tb1[f - 1]:+.3f} {c.tb1[f]:+.3f} "
                  f"{c.tb1[f + 1]:+.3f}; dtan b0/dm {c.dtb0[f]:+.1f} {c.dtb0[f + 1]:+.1f}, dtan b1/dm {c.dtb1[f]:+.1f} "
                  f"{c.dtb1[f + 1]:+.1f}; TE tan b0 l {c.tb0[l]:+.3f} b1 {c.tb1[l]:+.3f}, dtan b0/dm l "
                  f"{c.dtb0[l]:+.1f}")
            err = np.maximum(np.abs(c.tb0_fd - c.tb0), np.abs(c.tb1_fd - c.tb1))
            k = int(np.argmax(err))
            print(f"      region-wise nonuniform FD of the wall y against the exact slopes: max {err.max():.2e} at "
                  f"station {k} ({'f' if abs(k - f) <= abs(k - l) else 'l'}{k - (f if abs(k - f) <= abs(k - l) else l):+d}); "
                  f"at f {err[f]:.1e}, f+1 {err[f + 1]:.1e}, l-1 {err[l - 1]:.1e}, l {err[l]:.1e}; distance to the "
                  f"extreme point: f {c.m[f] - c.geo.m_tip:.2e} c, l {c.geo.m_tail - c.m[l]:.2e} c")
        g = gr.make_grid(geos[s].cascade, GridSettings(n_passage=n, n_eta=N_ETA), wt.INLET)
        met = build_metrics(g)
        c = table[(s, n, "PROD")]["case"]
        print(f"   {s:.0f} deg n {n} production metrics.py vs exact: tan b0 {np.abs(met.tan_beta0 - c.tb0).max():.1e}, "
              f"tan b1 {np.abs(met.tan_beta1 - c.tb1).max():.1e}, dtan b0/dm {np.abs(met.dtan_beta0_dm - c.dtb0).max():.1e}")
        keep[(s, n, "metrics")] = met

    print("\n7. Coefficients A .. E (mdot' = 1): global and LE / TE neighbourhoods (stations f-3..f+3, l-3..l+3)")
    coef = {}
    for s, n in ((33.0, 101), (33.0, 401), (53.0, 101), (53.0, 401), (13.0, 401)):
        g = gr.make_grid(geos[s].cascade, GridSettings(n_passage=n, n_eta=N_ETA), wt.INLET)
        met = build_metrics(g)
        for rk in ("uniform", "smooth"):
            prod_case = table[(s, n, "PROD")]["case"]
            rho = rho_field(prod_case, rk)
            ref = build_coefficients(met, rho, 1.0)
            mine = prod_case.coefficients(rho, S=met.S, tb=met.tan_beta)
            # validation: my reconstruction with production's own metrics reproduces coefficients.py
            entries = {"PROD (coefficients.py)": dict(A=ref.A, B=ref.B, C=ref.C, D=ref.D, E=ref.E)}
            rc = table[(s, n, "REDIST")]["case"]
            entries["REDIST across"] = rc.coefficients(rho_field(rc, rk), across=True)
            entries["REDIST region"] = rc.coefficients(rho_field(rc, rk), across=False)
            coef[(s, n, rk)] = entries
            line = []
            for name, e in entries.items():
                c = prod_case if name.startswith("PROD") else rc
                nb_le = slice(c.f - 3, c.f + 4)
                nb_te = slice(c.l - 3, c.l + 4)
                inner = np.ones(len(c.m), dtype=bool)
                inner[nb_le] = inner[nb_te] = False
                line.append(f"{name}: " + ", ".join(
                    f"{k} {np.nanmin(e[k]):+.2f}..{np.nanmax(e[k]):+.2f} (LE {np.nanmax(np.abs(e[k][nb_le])):.2f}, TE "
                    f"{np.nanmax(np.abs(e[k][nb_te])):.2f}, rest {np.nanmax(np.abs(e[k][inner])):.2f})" for k in ("A", "B", "C", "D", "E")))
            dA = np.abs(mine["D"] - ref.D).max()
            dE = np.abs(mine["E"] - ref.E).max()
            print(f"   {s:.0f} deg n {n} rho {rk}: my reconstruction with production's metrics reproduces "
                  f"coefficients.py to {dA:.1e} (D), {dE:.1e} (E)")
            for item in line:
                print(f"      {item}")

    print("\n8. Manufactured computational fields: |error| (max over eta) at the corner stations, and the maximum elsewhere")
    manu = {}
    for s in sorted(geos):
        geo, b, row = geos[s], rows[s]["b"], rows[s]["row"]
        for kind in ("PROD", "REDIST", "MINIMAL"):
            for n in SEQUENCE:
                case = Case(geo, b, row, kind, n, metrics_needed=False)
                for field in FIELDS:
                    e = manufactured(case.m, case.eta, field)
                    key = dict(fm1=case.f - 1, f=case.f, fp1=case.f + 1, l=case.l, lp1=case.l + 1)
                    rest = np.ones(len(case.m), dtype=bool)
                    rest[[0, len(case.m) - 1, case.f - 1, case.f, case.l, case.l + 1]] = False
                    manu[(s, kind, n, field)] = dict(
                        D=case.D, T=float(case.m[case.f] - case.m[case.f - 1]),
                        **{f"{d}_{k}": float(e[d][i]) for d in ("m", "mm", "me") for k, i in key.items()},
                        **{f"{d}_rest": float(np.nanmax(e[d][rest])) for d in ("m", "mm", "me")}, scale=e["scale"])
                manu[(s, kind, n, "bvp")] = dict(D=case.D, err=bvp_error(case.m),
                                                  near=slice(case.f - 3, case.f + 4), f=case.f)
    for s in (33.0, 53.0):
        for kind in ("PROD", "REDIST", "MINIMAL"):
            for field in ("F2",):
                rows_txt = []
                for n in SEQUENCE:
                    r = manu[(s, kind, n, field)]
                    rows_txt.append(f"n {n} (T/dm {r['T'] / r['D']:.2f}): f_m f-1 {r['m_fm1']:.1e} f {r['m_f']:.1e} rest "
                                    f"{r['m_rest']:.1e}; f_mm f-1 {r['mm_fm1']:.1e} f {r['mm_f']:.1e} rest {r['mm_rest']:.1e}; "
                                    f"f_meta f-1 {r['me_fm1']:.1e} f {r['me_f']:.1e} rest {r['me_rest']:.1e}")
                print(f"   {s:.0f} deg {kind} {field}:")
                for t in rows_txt:
                    print(f"      {t}")

    def order(vals, hs):
        return [math.log(vals[k] / vals[k + 1]) / math.log(hs[k] / hs[k + 1]) if vals[k + 1] > 0 and vals[k] > 0 else
                float("nan") for k in range(len(vals) - 1)]

    print("\n9. Observed orders (against the nominal dm; F2), corner stations and elsewhere, and the 1D model problem's "
          "nodal error")
    for s in sorted(geos):
        for kind in ("PROD", "REDIST", "MINIMAL"):
            hs = [manu[(s, kind, n, "F2")]["D"] for n in SEQUENCE]
            parts = []
            for d in ("m", "mm", "me"):
                for k in ("fm1", "f", "rest"):
                    vals = [manu[(s, kind, n, "F2")][f"{d}_{k}"] for n in SEQUENCE]
                    parts.append(f"{d}/{k} {np.round(order(vals, hs), 2).tolist()}")
            bv = [float(manu[(s, kind, n, "bvp")]["err"].max()) for n in SEQUENCE]
            parts.append(f"BVP max {['%.1e' % v for v in bv]} orders {np.round(order(bv, hs), 2).tolist()}")
            print(f"   {s:.0f} deg {kind}: " + "; ".join(parts))

    print("\n10. Smooth physical field on the real mapping: |psi_m error| (max over interior eta), central across the "
          "transition / region-wise one-sided")
    phys = {}
    for s in (33.0, 53.0, 13.0):
        geo, b, row = geos[s], rows[s]["b"], rows[s]["row"]
        for kind in ("PROD", "REDIST", "MINIMAL"):
            for n in (101, 201, 401, 801, 1601):
                case = Case(geo, b, row, kind, n, metrics_needed=False)
                phys[(s, kind, n)] = (case.D, physical_errors(case))
            for st_name in ("f-1", "f", "f+1", "l", "l+1"):
                print(f"   {s:.0f} deg {kind:7s} {st_name:4s}: psi_m central / one-sided, psi_mm central: " + " | ".join(
                    f"n {n}: {phys[(s, kind, n)][1][st_name][0]:.2e} / {phys[(s, kind, n)][1][st_name][1]:.1e}, "
                    f"{phys[(s, kind, n)][1][st_name][2]:.2e}" for n in (101, 401, 1601)))

    print("\n11. 53 deg, n_passage 401, the LE neighbourhood")
    c = table[(53.0, 401, "REDIST")]["case"]
    geo = geos[53.0]
    f = c.f
    print(f"   m_tip {geo.m_tip:+.6f}, m_le {geo.m_le:+.6f}, window {geo.m_le - geo.m_tip:.4e} c = "
          f"{(geo.m_le - geo.m_tip) / c.D:.3f} dm; nominal dm {c.D:.4e}")
    print(f"   stations f-3 .. f+2: " + ", ".join(f"{c.m[k]:+.6f}" for k in range(f - 3, f + 3)))
    print(f"   spacings: upstream {np.diff(c.m[f - 3:f]) / c.D}, transition {(c.m[f] - c.m[f - 1]) / c.D:.4f} dm = window "
          f"{(geo.m_le - geo.m_tip) / c.D:.4f} + 2 x 0.5 dm margin, blade {np.diff(c.m[f:f + 3]) / c.D}")
    e = manufactured(c.m, c.eta, "F2")
    print("   F2 |error| at f-3 .. f+3: f_m " + ", ".join(f"{e['m'][k]:.1e}" for k in range(f - 3, f + 4)) + "; f_mm " +
          ", ".join(f"{e['mm'][k]:.1e}" for k in range(f - 3, f + 4)) + "; f_meta " +
          ", ".join(f"{e['me'][k]:.1e}" for k in range(f - 3, f + 4)))
    h1, h2 = c.m[f - 1] - c.m[f - 2], c.m[f] - c.m[f - 1]
    print(f"   Taylor leading terms at f-1 (h1 {h1:.3e}, h2 {h2:.3e}): f_mm (h2 - h1)/3 = {(h2 - h1) / 3:.3e} x f''' "
          f"(|F2'''| <= 27 + 1.8 -> {(h2 - h1) / 3 * 28.8:.2e}); f_m h1 h2 / 6 = {h1 * h2 / 6:.2e} x f'''")

    figures = {}
    # 01 dm distribution
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=False)
    for ax, s in zip(axes, (33.0, 43.0, 53.0)):
        for kind in ("PROD", "REDIST", "MINIMAL"):
            cs = table[(s, 401, kind)]["case"]
            ax.semilogy(0.5 * (cs.m[1:] + cs.m[:-1]), np.diff(cs.m) / cs.D, ".-", ms=2, lw=0.8, color=COLORS[kind], label=kind)
        ax.axvline(geos[s].m_le, color=MUTED, lw=0.6, ls=":")
        ax.axvline(geos[s].m_te, color=MUTED, lw=0.6, ls=":")
        ax.legend(fontsize=7)
        style(ax, f"{s:.0f} deg, n_passage 401: cell length / nominal dm along m (dotted: LE and TE points)")
    axes[-1].set_xlabel("m (chords)", fontsize=8)
    fig.suptitle("01  Meridional spacing: production uniform, REDIST one large LE cell, MINIMAL a graded ladder", fontsize=11,
                 color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    figures["01_dm_distribution"] = fig
    figures["02_le_spacing_stagger33"] = figure_le(geos, rows, 33.0, "02")
    figures["03_le_spacing_stagger43"] = figure_le(geos, rows, 43.0, "03")
    figures["04_le_spacing_stagger53"] = figure_le(geos, rows, 53.0, "04")
    figures["05_te_spacing"] = figure_te(geos, rows)

    # 06 metrics comparison, 53 deg n 401 and 33 deg n 101
    fig, axes = plt.subplots(5, 4, figsize=(19, 15))
    for col, (s, n, side) in enumerate(((53.0, 401, "LE"), (53.0, 401, "TE"), (33.0, 101, "LE"), (33.0, 101, "TE"))):
        geo = geos[s]
        centre = geo.m_tip if side == "LE" else geo.m_tail
        span = 0.03
        ms = np.linspace(centre - span, centre + span, 3001)
        for kind in ("PROD", "REDIST"):
            cs = table[(s, n, kind)]["case"]
            y0, y1, _ = wt.walls_at(geo, rows[s]["b"], rows[s]["row"], ms, ANCHOR[kind])
            sel = (cs.m > centre - span) & (cs.m < centre + span)
            for r, (vals, cont, name) in enumerate(((cs.S, y1 - y0, "S"), (cs.tb0, None, "tan beta0 (lower wall)"),
                                                    (cs.tb1, None, "tan beta1 (upper wall)"),
                                                    (cs.dtb0, None, "d tan beta0 / dm"), (cs.dtb1, None, "d tan beta1 / dm"))):
                ax = axes[r, col]
                if cont is not None:
                    ax.plot(ms, cont, color=COLORS[kind], lw=0.8, alpha=0.6)
                ax.plot(cs.m[sel], vals[sel], "o", ms=4 if kind == "REDIST" else 3, color=COLORS[kind], label=kind)
                if r > 0:
                    ax.set_yscale("symlog", linthresh=1.0)
                ax.axvspan(*((geo.m_tip, geo.m_le) if side == "LE" else (geo.m_te, geo.m_tail)), color="#cf222e",
                           alpha=0.08)
                style(ax, f"{s:.0f} deg n {n} {side}: {name}")
        axes[0, col].legend(fontsize=7)
    fig.suptitle("06  Metrics at the stations (markers; exact C2 slopes on the walls, the side line's own angle elsewhere): "
                 "production samples the vertical tangent next to the tip, REDIST does not", fontsize=11, color=INK,
                 x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    figures["06_metrics_comparison"] = fig

    # 07 D / E
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    for col, (s, n) in enumerate(((33.0, 101), (53.0, 401), (53.0, 101))):
        entries = coef[(s, n, "uniform")]
        for r, k in enumerate(("D", "E")):
            ax = axes[r, col]
            for name, color in (("PROD (coefficients.py)", COLORS["PROD"]), ("REDIST across", COLORS["REDIST"]),
                                ("REDIST region", COLORS["MINIMAL"])):
                cs = table[(s, n, "PROD" if name.startswith("PROD") else "REDIST")]["case"]
                j = len(cs.eta) // 2
                sel = (cs.m > geos[s].m_tip - 0.05) & (cs.m < geos[s].m_tip + 0.05)
                ax.plot(cs.m[sel], entries[name][k][sel, j], "o-", ms=3, lw=0.9, color=color, label=name)
            ax.set_yscale("symlog", linthresh=1.0)
            ax.axvspan(geos[s].m_tip, geos[s].m_le, color="#cf222e", alpha=0.08)
            ax.legend(fontsize=7)
            style(ax, f"{s:.0f} deg n {n}: {k} at eta 0.5 near the LE (rho = 1)")
    fig.suptitle("07  D = A_eta - B_m and E = C_m - B_eta near the LE: production (central across the jump), REDIST with "
                 "nonuniform differences across the transition, REDIST region-wise", fontsize=11, color=INK, x=0.01,
                 ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    figures["07_DE_coefficients_comparison"] = fig

    # 08 / 09 manufactured errors near LE / TE (53 deg n 401, 33 deg n 101)
    for name, side in (("08_manufactured_error_near_LE", "LE"), ("09_manufactured_error_near_TE", "TE")):
        fig, axes = plt.subplots(2, 3, figsize=(18, 9))
        for r, (s, n) in enumerate(((53.0, 401), (33.0, 101))):
            for col, d in enumerate(("m", "mm", "me")):
                ax = axes[r, col]
                for k, kind in enumerate(("PROD", "REDIST", "MINIMAL")):
                    cs = Case(geos[s], rows[s]["b"], rows[s]["row"], kind, n, metrics_needed=False)
                    e = manufactured(cs.m, cs.eta, "F2")
                    centre = cs.f if side == "LE" else cs.l
                    idx = np.arange(centre - 4, centre + 5)
                    ax.semilogy(idx - centre + 0.12 * (k - 1), e[d][idx], "o-", ms=4, lw=1.0, color=COLORS[kind], label=kind)
                ax.set_xlabel(f"station - {'f (first wall)' if side == 'LE' else 'l (last wall)'}", fontsize=8)
                ax.legend(fontsize=7)
                style(ax, f"{s:.0f} deg n {n}: |f_{d} error| (F2), {side}")
        fig.suptitle(f"{name[:2]}  Manufactured computational field F2 near the {side}: nonuniform three-point differences",
                     fontsize=11, color=INK, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        figures[name] = fig

    # 10 convergence
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for col, d in enumerate(("m", "mm", "me")):
        ax = axes[0, col]
        for kind in ("PROD", "REDIST", "MINIMAL"):
            for k, ls in (("fm1", "-"), ("f", "--"), ("rest", ":")):
                hs = [manu[(53.0, kind, n, "F2")]["D"] for n in SEQUENCE]
                ax.loglog(hs, [manu[(53.0, kind, n, "F2")][f"{d}_{k}"] for n in SEQUENCE], ls, marker="o", ms=3,
                          color=COLORS[kind], label=f"{kind} {k}")
        ax.set_xlabel("nominal dm", fontsize=8)
        ax.legend(fontsize=6, ncol=3)
        style(ax, f"53 deg: |f_{d} error| (F2) at f-1, f and elsewhere")
    ax = axes[1, 0]
    for s, mk in ((33.0, "o"), (53.0, "s"), (13.0, "^")):
        for kind in ("PROD", "REDIST", "MINIMAL"):
            hs = [manu[(s, kind, n, "bvp")]["D"] for n in SEQUENCE]
            ax.loglog(hs, [manu[(s, kind, n, "bvp")]["err"].max() for n in SEQUENCE], "-", marker=mk, ms=3,
                      color=COLORS[kind], label=f"{s:.0f} deg {kind}")
    ax.set_xlabel("nominal dm", fontsize=8)
    ax.legend(fontsize=6, ncol=3)
    style(ax, "1D model u'' = f on the stations: max nodal error")
    for col, (s, which) in enumerate(((53.0, "f-1"), (53.0, "f")), start=1):
        ax = axes[1, col]
        for kind in ("PROD", "REDIST", "MINIMAL"):
            hs = [phys[(s, kind, n)][0] for n in (101, 201, 401, 801, 1601)]
            ax.loglog(hs, [phys[(s, kind, n)][1][which][0] for n in (101, 201, 401, 801, 1601)], "-o", ms=3,
                      color=COLORS[kind], label=f"{kind} central")
            ax.loglog(hs, [phys[(s, kind, n)][1][which][1] for n in (101, 201, 401, 801, 1601)], ":s", ms=3,
                      color=COLORS[kind], label=f"{kind} region one-sided")
        ax.set_xlabel("nominal dm", fontsize=8)
        ax.legend(fontsize=6)
        style(ax, f"{s:.0f} deg: smooth physical field on the real mapping, |psi_m error| at {which}")
    fig.suptitle("10  Refinement: computational fields (top), the 1D model problem and the real mapping (bottom)", fontsize=11,
                 color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    figures["10_refinement_convergence"] = fig

    # 11 53 / 401 zoom
    fig, ax = plt.subplots(figsize=(12, 10))
    c = table[(53.0, 401, "REDIST")]["case"]
    geo = geos[53.0]
    wt.draw_blade(ax, geo, lw=2.0)
    f = c.f
    ms = np.linspace(c.m[f - 2], c.m[f + 2], 4001)
    y0, y1, _ = wt.walls_at(geo, c.b, c.row, ms, "tip")
    ax.plot(ms, y0, color=COLORS["REDIST"], lw=2.4, label="eta = 0 boundary (contour where the line cuts the blade)")
    ax.plot(ms, y1 - geo.pitch, color=COLORS["REDIST"], lw=1.6, ls="--", label="eta = 1 boundary - pitch (contour)")
    ax.axvspan(c.m[f - 1], c.m[f], color="#1baf7a", alpha=0.10, label="REDIST transition cell")
    for k, name in ((f - 2, "f-2"), (f - 1, "last periodic f-1"), (f, "first wall f"), (f + 1, "second wall f+1")):
        ax.axvline(c.m[k], color=INK, lw=1.2 if k in (f - 1, f) else 0.7, ls="-" if k in (f - 1, f) else ":")
        ax.text(c.m[k], 0.985, f" {name}", transform=ax.get_xaxis_transform(), fontsize=8, rotation=90, va="top", color=INK)
        ax.plot(c.m[k], c.y0[k], "o", color=INK, ms=5, zorder=9)
        ax.plot(c.m[k], c.y1[k] - geo.pitch, "s", color=INK, ms=5, zorder=9)
    ax.plot(*geo.tip, ">", color=INK, ms=10, zorder=10, label="nose tip")
    ax.plot(*geo.le_point, "o", ms=9, mfc="white", mec=INK, mew=1.6, zorder=10, label="LE point (SS/PS branch)")
    ybar = geo.tip[1] - 0.012
    ax.annotate("", (c.m[f - 1], ybar), (c.m[f], ybar), arrowprops=dict(arrowstyle="<->", color=INK))
    ax.text(0.5 * (c.m[f - 1] + c.m[f]), ybar - 0.002, f"transition {(c.m[f] - c.m[f - 1]) / c.D:.2f} dm = window "
            f"{(geo.m_le - geo.m_tip) / c.D:.2f} dm + 2 x 0.5 dm", ha="center", va="top", fontsize=9, color=INK)
    under = 0.5 * (geo.m_tip + geo.m_le)
    y_under = float(wt.walls_at(geo, c.b, c.row, np.array([under]), "tip")[1][0] - geo.pitch)
    ax.annotate("inside the cell the upper boundary (- pitch) runs on the\nneighbour's suction-surface underside, tip -> LE "
                "point,\nthen on its pressure surface: unsampled, but the only\nboundary that holds no metal and tiles the "
                "passages", (under, y_under), xytext=(-260, -70), textcoords="offset points", fontsize=8, color=INK,
                arrowprops=dict(arrowstyle="->", color=MUTED))
    ax.annotate("", (c.m[f + 1], ybar + 0.03), (c.m[f + 2], ybar + 0.03), arrowprops=dict(arrowstyle="<->", color=MUTED))
    ax.text(c.m[f + 2], ybar + 0.03, "  nominal dm", va="center", fontsize=8, color=MUTED)
    ax.set_xlim(c.m[f - 2] - 0.4 * c.D, c.m[f + 2] + 0.4 * c.D)
    ax.set_ylim(geo.tip[1] - 0.03, geo.tip[1] + 0.05)
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(fontsize=8, loc="upper left")
    style(ax, "53 deg, n_passage 401, REDIST: SS orange, PS blue; nodes o (lower wall) and s (upper wall - pitch)")
    fig.suptitle("11  The LE transition cell at 53 deg / 401", fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    figures["11_53deg_401_transition_zoom"] = fig

    # 12 REDIST vs MINIMAL
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    ax = axes[0]
    for s, mk in zip(sorted(geos), ("o", "s", "^", "v", "D")):
        geo = geos[s]
        hs = [manu[(s, "REDIST", n, "F2")]["D"] for n in SEQUENCE]
        ax.plot(SEQUENCE, [manu[(s, "REDIST", n, "F2")]["T"] / h for n, h in zip(SEQUENCE, hs)], "-", marker=mk,
                color=COLORS["REDIST"], label=f"{s:.0f} REDIST")
        ax.plot(SEQUENCE, [manu[(s, "MINIMAL", n, "F2")]["T"] / h for n, h in zip(SEQUENCE, hs)], "--", marker=mk,
                color=COLORS["MINIMAL"], label=f"{s:.0f} MINIMAL")
        ax.plot(SEQUENCE, [(geo.m_le - geo.m_tip) / h for h in hs], ":", marker=mk, color=MUTED, label=f"{s:.0f} minimum")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("n_passage", fontsize=8)
    ax.legend(fontsize=6, ncol=3)
    style(ax, "LE transition / nominal dm against the geometric minimum")
    ax = axes[1]
    for kind in ("REDIST", "MINIMAL"):
        for s, mk in zip(sorted(geos), ("o", "s", "^", "v", "D")):
            hs = [manu[(s, kind, n, "F2")]["D"] for n in SEQUENCE]
            ax.loglog(hs, [max(manu[(s, kind, n, "F2")]["mm_fm1"], manu[(s, kind, n, "F2")]["mm_f"]) for n in SEQUENCE],
                      "-" if kind == "REDIST" else "--", marker=mk, ms=3, color=COLORS[kind], label=f"{s:.0f} {kind}")
    ax.set_xlabel("nominal dm", fontsize=8)
    ax.legend(fontsize=6, ncol=2)
    style(ax, "|f_mm error| at the transition stations (F2): no convergence for either")
    ax = axes[2]
    for kind in ("PROD", "REDIST", "MINIMAL"):
        for s, mk in zip(sorted(geos), ("o", "s", "^", "v", "D")):
            hs = [manu[(s, kind, n, "bvp")]["D"] for n in SEQUENCE]
            ax.loglog(hs, [manu[(s, kind, n, "bvp")]["err"].max() for n in SEQUENCE],
                      {"PROD": ":", "REDIST": "-", "MINIMAL": "--"}[kind], marker=mk, ms=3, color=COLORS[kind],
                      label=f"{s:.0f} {kind}")
    ax.set_xlabel("nominal dm", fontsize=8)
    ax.legend(fontsize=5.5, ncol=3)
    style(ax, "1D model problem: max nodal error")
    fig.suptitle("12  REDIST against the minimal alternative", fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    figures["12_redist_vs_minimum_candidate"] = fig

    if args.save is not None:
        for name, fg in figures.items():
            fg.savefig(args.save / f"{name}.png", dpi=125)
            print(f"wrote {args.save / (name + '.png')}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
