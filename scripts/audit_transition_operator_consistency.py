"""
Operator-consistency audit of the finite differences at the periodic-to-blade (f-1 / f) and
blade-to-periodic (l / l+1) station pairs (diagnostic only; no production file changed).

    python scripts/audit_transition_operator_consistency.py --save outputs/transition_operator_consistency --no-show

Every production row is read through the production objects themselves:
discretization._interior_stencil, discretization._periodic_pde_rows (with
boundary_conditions.periodic_ghost) and the relations of build_boundary_conditions. A row's
raw equation is recovered as raw = -H * residual, which is linear in A .. E, so constant fields
(A = 1 alone; A = 1 with one of B, C, D, E = 1) extract the very psi_eta,eta, psi_m,m, psi_eta,
psi_m and psi_m,eta each production row applies. Nothing of the production side is
re-implemented; the diagnostic central rows are written here and labelled so.

Two manufactured stream functions, both with psi(m, eta + 1) = psi(m, eta) + 1:
    "brief"    eta + 0.07 cos(1.3 m) sin(2 pi eta) + 0.03 sin(0.7 m) sin(4 pi eta)   (psi = 0, 1
               on eta = 0, 1 exactly: every m difference along a periodic line, and
               psi_eta,eta there by antisymmetry, is exact for it)
    "shifted"  eta + 0.07 cos(1.3 m) cos(2 pi eta) + 0.03 sin(0.7 m) sin(2 pi eta + 0.4)
               (not 0 / 1 on the lines, so the periodic rows' m differences are exercised;
               the wall values are irrelevant to an operator test: no row differentiates a
               wall relation)
and the matching synthetic coefficient sets (the brief's, and the same with phases 0.3 / 0.5
in eta so that A_eta and B_m do not vanish on eta = 0, 1).

Only SS+PS-admissible grids; n_eta = n_passage + 1 (deta and dm refined together). Default C2
65-(12A10)10, solidity 1, inlet 45 deg, exit kappa_te, one-chord extensions, rho = 1,
mdot' = 1, c_ref None (lengths in chords).
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", Path(__file__).resolve().parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import matplotlib.pyplot as plt  # noqa: E402

import audit_wall_station_topology as wt  # noqa: E402
from turbolenza.b2b import coefficients as cf  # noqa: E402
from turbolenza.b2b import discretization as dz  # noqa: E402
from turbolenza.b2b import grid as gr  # noqa: E402
from turbolenza.b2b.boundary_conditions import BoundaryConditionError, build_boundary_conditions  # noqa: E402
from turbolenza.b2b.coefficients import build_coefficients  # noqa: E402
from turbolenza.b2b.config import GridSettings  # noqa: E402
from turbolenza.b2b.geometry import make_cascade  # noqa: E402
from turbolenza.b2b.metrics import build_metrics  # noqa: E402

STAGGERS = (13.0, 23.0, 33.0, 43.0, 53.0)
SEQUENCES = {13.0: (60, 102, 173, 294), 23.0: (43, 66, 101, 159), 33.0: (29, 43, 65, 97),
             43.0: (26, 34, 45, 59), 53.0: (22, 31, 45)}
PRIMARY = 33.0
INLET = math.radians(45.0)
TWO_PI = 2.0 * math.pi
INK, MUTED, GRIDLINE = "#1f2328", "#6e7781", "#c9ced6"
CLASSES = ("ordinary interior", "f-1 interior", "f interior", "l interior", "l+1 interior",
           "periodic standard (i,J)", "LE periodic row (f-1,J)", "TE periodic row (l+1,J)")
COEF_CLASSES = ("ordinary periodic", "blade front (5-60 %)", "blade rear (65-95 %)", "f-1 interior", "f interior",
                "f+1 interior", "l-1 interior", "l interior", "l+1 interior", "periodic standard (i,J)",
                "LE periodic row (f-1,J)", "TE periodic row (l+1,J)")
COLORS = {"ordinary interior": MUTED, "f-1 interior": "#2a78d6", "f interior": "#eb6834", "l interior": "#8250df",
          "l+1 interior": "#1baf7a", "periodic standard (i,J)": INK, "LE periodic row (f-1,J)": "#cf222e",
          "TE periodic row (l+1,J)": "#9a6700", "ordinary periodic": INK, "blade front (5-60 %)": "#b3b9c2",
          "blade rear (65-95 %)": MUTED, "f+1 interior": "#f5a46b", "l-1 interior": "#b79ae8"}
MARKERS = {"ordinary interior": "o", "f-1 interior": "s", "f interior": "D", "l interior": "^", "l+1 interior": "v",
           "periodic standard (i,J)": "x", "LE periodic row (f-1,J)": "*", "TE periodic row (l+1,J)": "P",
           "ordinary periodic": "x", "blade front (5-60 %)": "o", "blade rear (65-95 %)": "o", "f+1 interior": "d",
           "l-1 interior": "<"}
OPERATORS = ("psi_m", "psi_mm", "psi_eta", "psi_etaeta", "psi_meta")
VARIANTS = ("brief", "shifted")


# --- manufactured stream functions and synthetic coefficients (computational space) ----------------

def manufactured(M, H, variant):
    a, b = 0.07, 0.03
    if variant == "brief":
        s2, s4, c2, c4 = np.sin(TWO_PI * H), np.sin(2 * TWO_PI * H), np.cos(TWO_PI * H), np.cos(2 * TWO_PI * H)
        return dict(
            psi=H + a * np.cos(1.3 * M) * s2 + b * np.sin(0.7 * M) * s4,
            psi_m=-1.3 * a * np.sin(1.3 * M) * s2 + 0.7 * b * np.cos(0.7 * M) * s4,
            psi_mm=-1.69 * a * np.cos(1.3 * M) * s2 - 0.49 * b * np.sin(0.7 * M) * s4,
            psi_eta=1.0 + TWO_PI * a * np.cos(1.3 * M) * c2 + 2 * TWO_PI * b * np.sin(0.7 * M) * c4,
            psi_etaeta=-TWO_PI ** 2 * a * np.cos(1.3 * M) * s2 - (2 * TWO_PI) ** 2 * b * np.sin(0.7 * M) * s4,
            psi_meta=-1.3 * TWO_PI * a * np.sin(1.3 * M) * c2 + 0.7 * 2 * TWO_PI * b * np.cos(0.7 * M) * c4)
    c2, s2 = np.cos(TWO_PI * H), np.sin(TWO_PI * H)
    sp, cp = np.sin(TWO_PI * H + 0.4), np.cos(TWO_PI * H + 0.4)
    return dict(
        psi=H + a * np.cos(1.3 * M) * c2 + b * np.sin(0.7 * M) * sp,
        psi_m=-1.3 * a * np.sin(1.3 * M) * c2 + 0.7 * b * np.cos(0.7 * M) * sp,
        psi_mm=-1.69 * a * np.cos(1.3 * M) * c2 - 0.49 * b * np.sin(0.7 * M) * sp,
        psi_eta=1.0 - TWO_PI * a * np.cos(1.3 * M) * s2 + TWO_PI * b * np.sin(0.7 * M) * cp,
        psi_etaeta=-TWO_PI ** 2 * a * np.cos(1.3 * M) * c2 - TWO_PI ** 2 * b * np.sin(0.7 * M) * sp,
        psi_meta=1.3 * TWO_PI * a * np.sin(1.3 * M) * s2 + 0.7 * TWO_PI * b * np.cos(0.7 * M) * cp)


def synthetic(M, H, variant):
    """Smooth A, B, C with D = A_eta - B_m and E = C_m - B_eta analytically ('brief': the
    brief's fields; 'shifted': phases 0.3 / 0.5 in eta)."""
    pa, pb = (0.0, 0.0) if variant == "brief" else (0.3, 0.5)
    A = 1.5 + 0.2 * np.sin(M) * np.cos(TWO_PI * H + pa)
    B = 0.25 + 0.08 * np.cos(0.8 * M) * np.sin(TWO_PI * H + pb)
    C = 1.2 + 0.15 * np.cos(1.1 * M) + 0.0 * H
    A_eta = -0.2 * TWO_PI * np.sin(M) * np.sin(TWO_PI * H + pa)
    B_m = -0.064 * np.sin(0.8 * M) * np.sin(TWO_PI * H + pb)
    B_eta = 0.08 * TWO_PI * np.cos(0.8 * M) * np.cos(TWO_PI * H + pb)
    C_m = -0.165 * np.sin(1.1 * M) + 0.0 * H
    return A, B, C, A_eta - B_m, C_m - B_eta


def source(A, B, C, D, E, ex):
    return A * ex["psi_etaeta"] - 2 * B * ex["psi_meta"] + C * ex["psi_mm"] + D * ex["psi_eta"] + E * ex["psi_m"]


# --- one grid ----------------------------------------------------------------------------------------

_CASCADES = {}


class Case:
    def __init__(self, stagger: float, n: int, n_eta: int | None = None):
        if stagger not in _CASCADES:
            _CASCADES[stagger] = make_cascade(wt.spec_at(stagger))
        self.stagger, self.n = stagger, n
        self.cascade = _CASCADES[stagger]
        n_eta = n + 1 if n_eta is None else n_eta
        self.grid = g = gr.make_grid(self.cascade, GridSettings(n_passage=n, n_eta=n_eta), INLET, require_ss_ps=True)
        self.metrics = build_metrics(g)
        self.bcs = build_boundary_conditions(g)
        self.shape = (len(g.m), len(g.eta))
        self.co = build_coefficients(self.metrics, np.ones(self.shape), 1.0)
        self.f, self.l, self.J = g.first_blade_i, g.last_blade_i, n_eta - 1
        self.dm, self.deta = float(g.dm), float(g.deta)
        self.M, self.H = np.meshgrid(g.m, g.eta, indexing="ij")
        self.blade = np.zeros(self.shape[0], dtype=bool)
        self.blade[self.f:self.l + 1] = True
        self.masks = self._classes()

    def _classes(self):
        n_m, n_eta = self.shape
        f, l, J = self.f, self.l, self.J
        I = np.arange(n_m)[:, None] + np.zeros((1, n_eta), dtype=int)
        Jj = np.arange(n_eta)[None, :] + np.zeros((n_m, 1), dtype=int)
        m = self.M
        b = self.grid.boundaries
        frac = (m - b.m_le) / (b.m_te - b.m_le)
        interior = (I >= 1) & (I <= n_m - 2) & (Jj >= 1) & (Jj <= J - 1)
        far_adjacent = (I == 1) | (I == n_m - 2)          # their final rows depend on the unimplemented far elimination
        far = (I <= f - 4) | ((I >= f + 3) & (I <= l - 3)) | (I >= l + 4)
        periodic_rows = (Jj == J) & (((I >= 1) & (I <= f - 1)) | ((I >= l + 1) & (I <= n_m - 2)))
        blade = (I >= f) & (I <= l)
        return {
            "ordinary interior": interior & far & ~far_adjacent,
            "f-1 interior": interior & (I == f - 1),
            "f interior": interior & (I == f),
            "f+1 interior": interior & (I == f + 1),
            "l-1 interior": interior & (I == l - 1),
            "l interior": interior & (I == l),
            "l+1 interior": interior & (I == l + 1),
            "periodic standard (i,J)": periodic_rows & (I != f - 1) & (I != l + 1) & ~far_adjacent,
            "LE periodic row (f-1,J)": periodic_rows & (I == f - 1),
            "TE periodic row (l+1,J)": periodic_rows & (I == l + 1),
            "ordinary periodic": interior & ((I <= f - 4) | (I >= l + 4)) & ~far_adjacent,
            "blade front (5-60 %)": interior & blade & (frac >= 0.05) & (frac <= 0.60) & (I >= f + 2) & (I <= l - 2),
            "blade rear (65-95 %)": interior & blade & (frac >= 0.65) & (frac <= 0.95) & (I >= f + 2) & (I <= l - 2),
        }

    def rows(self, A, B, C, D, E):
        n_eta = self.shape[1]
        st = dz._interior_stencil(A, B, C, D, E, self.dm, self.deta)
        rows = dz._periodic_pde_rows(A, B, C, D, E, self.dm, self.deta, self.f, self.l,
                                     lambda k: self.bcs.periodic_ghost(k, n_eta))
        return st, rows

    def raw(self, A, B, C, D, E, psi):
        """raw = -H * residual of every production PDE row, at its node; NaN elsewhere."""
        st, rows = self.rows(A, B, C, D, E)
        out = np.full(self.shape, np.nan)
        P = psi
        c = (slice(1, -1), slice(1, -1))
        R = (P[c] + st.west[c] * P[:-2, 1:-1] + st.east[c] * P[2:, 1:-1] + st.south[c] * P[1:-1, :-2]
             + st.north[c] * P[1:-1, 2:] + st.mixed[c] * (P[2:, 2:] - P[2:, :-2] - P[:-2, 2:] + P[:-2, :-2]) - st.rhs[c])
        out[c] = -st.H[c] * R
        for row in rows:
            out[row.target] = -row.H * row.residual(P)
        return out, rows

    def production_operators(self, psi):
        """The five difference operators exactly as the production rows apply them."""
        one, zero = np.ones(self.shape), np.zeros(self.shape)
        base, _ = self.raw(one, zero, zero, zero, zero, psi)
        ops = dict(psi_etaeta=base)
        ops["psi_mm"] = self.raw(one, zero, one, zero, zero, psi)[0] - base
        ops["psi_eta"] = self.raw(one, zero, zero, one, zero, psi)[0] - base
        ops["psi_m"] = self.raw(one, zero, zero, zero, one, psi)[0] - base
        ops["psi_meta"] = -0.5 * (self.raw(one, one, zero, zero, zero, psi)[0] - base)
        return ops


def stats(err, mask):
    v = np.abs(np.asarray(err)[mask])
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan"), 0
    return float(v.max()), float(np.sqrt(np.mean(v ** 2))), int(v.size)


def order(values, hs):
    if not (values[0] > 0 and values[-1] > 0):
        return float("nan")
    return math.log(values[0] / values[-1]) / math.log(hs[0] / hs[-1])


def orders_consecutive(values, hs):
    return [math.log(values[k] / values[k + 1]) / math.log(hs[k] / hs[k + 1]) if values[k] > 0 and values[k + 1] > 0
            else float("nan") for k in range(len(values) - 1)]


# --- conservative face-flux row (diagnostic; the form scripts/audit_discretization.py uses for Eq. 5-20) ---

def conservative(A, B, C, psi, dm, deta):
    Af, Bf = 0.5 * (A[1:-1, 1:] + A[1:-1, :-1]), 0.5 * (B[1:-1, 1:] + B[1:-1, :-1])
    F = Af * (psi[1:-1, 1:] - psi[1:-1, :-1]) / deta - Bf * (psi[2:, 1:] + psi[2:, :-1] - psi[:-2, 1:] - psi[:-2, :-1]) / (4.0 * dm)
    Bg, Cg = 0.5 * (B[1:, 1:-1] + B[:-1, 1:-1]), 0.5 * (C[1:, 1:-1] + C[:-1, 1:-1])
    G = Bg * (psi[1:, 2:] + psi[:-1, 2:] - psi[1:, :-2] - psi[:-1, :-2]) / (4.0 * deta) - Cg * (psi[1:, 1:-1] - psi[:-1, 1:-1]) / dm
    out = np.full(A.shape, np.nan)
    out[1:-1, 1:-1] = (F[:, 1:] - F[:, :-1]) / deta - (G[1:, :] - G[:-1, :]) / dm
    return out


# --- exact same-side D and E of the production fields (rho = 1, mdot' = 1) ---------------------------

def same_side_exact(case):
    """D, E at every node from the analytic derivatives on the node's own side: the side-line
    beta law on the periodic stations (A_eta = B_eta = C_m = 0 there), the C2 curves' exact
    derivatives on the blade stations. E = S' - (t1 - t0) = 0 on both sides."""
    g, met, b = case.grid, case.metrics, case.grid.boundaries
    m, eta = np.asarray(g.m), np.asarray(g.eta)
    n_m = case.shape[0]
    D = np.empty(case.shape)
    E = np.zeros(case.shape)
    dtb_side = np.zeros(n_m)
    k_le, k_te = float(b.upstream_beta[-1]), float(b.downstream_beta[0])
    for i in range(n_m):
        if i < case.f:
            slope = (b.inlet_angle - k_le) / (b.m_in - b.m_le)
            beta = k_le + (m[i] - b.m_le) * slope
        elif i > case.l:
            slope = (b.exit_angle - k_te) / (b.m_out - b.m_te)
            beta = k_te + (m[i] - b.m_te) * slope
        else:
            continue
        dtb_side[i] = (1.0 + math.tan(beta) ** 2) * slope
    per = ~case.blade
    D[per] = -dtb_side[per][:, None]
    t0, t1 = np.asarray(met.tan_beta0), np.asarray(met.tan_beta1)
    d0, d1 = np.asarray(met.dtan_beta0_dm), np.asarray(met.dtan_beta1_dm)
    S = np.asarray(met.S)
    tb = (1.0 - eta)[None, :] * t0[:, None] + eta[None, :] * t1[:, None]
    A_eta = 2.0 * tb * ((t1 - t0) / S)[:, None]
    B_m = (1.0 - eta)[None, :] * d0[:, None] + eta[None, :] * d1[:, None]
    D[case.blade] = (A_eta - B_m)[case.blade]
    return D, E, dtb_side


# --- the special rows and their pure central diagnostics ---------------------------------------------

def special_rows(case, ops, variant, ex):
    f, l, J = case.f, case.l, case.J
    m = np.asarray(case.grid.m)
    psi = ex["psi"]
    up, dn = 1.0 + case.deta, 1.0 - case.deta

    def q(k):          # eta difference at (k, J) with the analytic continuation above the line
        return float((manufactured(np.array(m[k]), np.array(up), variant)["psi"]
                      - manufactured(np.array(m[k]), np.array(dn), variant)["psi"]) / (2 * case.deta))
    e = {k: {op: float(ex[op][k, J]) for op in OPERATORS} for k in (f - 1, l + 1)}
    out = dict(
        le_m_prod=float(ops["psi_m"][f - 1, J]) - e[f - 1]["psi_m"],
        le_mm_prod=float(ops["psi_mm"][f - 1, J]) - e[f - 1]["psi_mm"],
        le_meta_prod=float(ops["psi_meta"][f - 1, J]) - e[f - 1]["psi_meta"],
        le_meta_central=(q(f) - q(f - 2)) / (2 * case.dm) - e[f - 1]["psi_meta"],
        te_m_prod=float(ops["psi_m"][l + 1, J]) - e[l + 1]["psi_m"],
        te_m_central=float((psi[l + 2, J] - psi[l, J]) / (2 * case.dm)) - e[l + 1]["psi_m"],
        te_mm_prod=float(ops["psi_mm"][l + 1, J]) - e[l + 1]["psi_mm"],
        te_mm_central=float((psi[l + 2, J] - 2 * psi[l + 1, J] + psi[l, J]) / case.dm ** 2) - e[l + 1]["psi_mm"],
        te_meta_prod=float(ops["psi_meta"][l + 1, J]) - e[l + 1]["psi_meta"],
        te_meta_central=(q(l + 2) - q(l)) / (2 * case.dm) - e[l + 1]["psi_meta"])
    A, B, C, D, E = synthetic(case.M, case.H, variant)
    Q = source(A, B, C, D, E, ex)
    raw, rows = case.raw(A, B, C, D, E, psi)
    for k, tag in ((f - 1, "le"), (l + 1, "te")):
        psi_m_c = (psi[k + 1, J] - psi[k - 1, J]) / (2 * case.dm)
        psi_mm_c = (psi[k + 1, J] - 2 * psi[k, J] + psi[k - 1, J]) / case.dm ** 2
        psi_meta_c = (q(k + 1) - q(k - 1)) / (2 * case.dm)
        central = (A[k, J] * ops["psi_etaeta"][k, J] - 2 * B[k, J] * psi_meta_c + C[k, J] * psi_mm_c
                   + D[k, J] * ops["psi_eta"][k, J] + E[k, J] * psi_m_c)
        out[f"{tag}_row_prod"] = float(raw[k, J] - Q[k, J])
        out[f"{tag}_row_central"] = float(central - Q[k, J])
    te_row = next(r for r in rows if r.target == (l + 1, J))
    out["te_center"] = te_row.coefficient((l + 1, J))
    out["te_largest"] = max(abs(t.coefficient) for t in te_row.terms)
    return out


# --- one grid: every measurement -----------------------------------------------------------------------

def measure(case):
    res = dict(dm=case.dm, deta=case.deta, f=case.f, l=case.l, J=case.J, n_m=case.shape[0])
    f, l, J = case.f, case.l, case.J
    for variant in VARIANTS:
        ex = manufactured(case.M, case.H, variant)
        ops = case.production_operators(ex["psi"])
        res[f"deriv_{variant}"] = {op: {c: stats(ops[op] - ex[op], case.masks[c]) for c in CLASSES} for op in OPERATORS}
        res[f"special_{variant}"] = special_rows(case, ops, variant, ex)
        A, B, C, D, E = synthetic(case.M, case.H, variant)
        Q = source(A, B, C, D, E, ex)
        raw, _ = case.raw(A, B, C, D, E, ex["psi"])
        res[f"synth_{variant}"] = {c: stats(raw - Q, case.masks[c]) for c in CLASSES}
        Dp = cf._d_deta(A, case.deta, case.blade) - cf._d_dm(B, case.dm)
        Ep = cf._d_dm(C, case.dm) - cf._d_deta(B, case.deta, case.blade)
        raw_p, _ = case.raw(A, B, C, Dp, Ep, ex["psi"])
        res[f"synthprod_{variant}"] = dict(tau={c: stats(raw_p - Q, case.masks[c]) for c in CLASSES},
                                           D={c: stats(Dp - D, case.masks[c]) for c in CLASSES},
                                           E={c: stats(Ep - E, case.masks[c]) for c in CLASSES})
    ex = manufactured(case.M, case.H, "brief")
    res["far_rows"] = (float(np.abs((-3 * ex["psi"][0] + 4 * ex["psi"][1] - ex["psi"][2]) / (2 * case.dm) - ex["psi_m"][0]).max()),
                       float(np.abs((3 * ex["psi"][-1] - 4 * ex["psi"][-2] + ex["psi"][-3]) / (2 * case.dm) - ex["psi_m"][-1]).max()))
    res["relations"] = max(abs(r.residual(ex["psi"])) for r in case.bcs.relations if r.kind in ("lower wall", "upper wall", "periodic"))

    # production coefficients
    co = case.co
    D_side, E_side, dtb_side = same_side_exact(case)
    masks = {c: case.masks[c] for c in COEF_CLASSES}
    res["coef"] = dict(D={c: stats(co.D - D_side, masks[c]) for c in COEF_CLASSES},
                       E={c: stats(co.E - E_side, masks[c]) for c in COEF_CLASSES},
                       Dmag={c: stats(D_side, masks[c]) for c in COEF_CLASSES})
    ex = manufactured(case.M, case.H, "shifted")          # non-degenerate on the periodic lines
    Q_side = source(co.A, co.B, co.C, D_side, E_side, ex)
    raw_a, _ = case.raw(co.A, co.B, co.C, co.D, co.E, ex["psi"])
    raw_b, _ = case.raw(co.A, co.B, co.C, D_side, E_side, ex["psi"])
    weight = np.abs(co.A) + 2 * np.abs(co.B) + np.abs(co.C) + np.abs(D_side) + np.abs(E_side)
    res["decomp"] = dict(production={c: stats(raw_a - Q_side, masks[c]) for c in COEF_CLASSES},
                         stencil={c: stats(raw_b - Q_side, masks[c]) for c in COEF_CLASSES},
                         production_rel={c: stats((raw_a - Q_side) / weight, masks[c]) for c in COEF_CLASSES},
                         stencil_rel={c: stats((raw_b - Q_side) / weight, masks[c]) for c in COEF_CLASSES})
    cons = conservative(co.A, co.B, co.C, ex["psi"], case.dm, case.deta)
    gap = raw_a - cons
    gap[~np.isfinite(cons)] = np.nan
    b = case.grid.boundaries
    rear = int(np.searchsorted(case.grid.m, b.m_le + 0.75 * (b.m_te - b.m_le)))
    blocks = {"LE f-2..f+1": slice(f - 2, f + 2), "TE l-1..l+2": slice(l - 1, l + 3),
              "blade rear": slice(rear, rear + 4), "ordinary periodic": slice(f // 2, f // 2 + 4)}
    res["gap"] = dict(pointwise={c: stats(gap, masks[c]) for c in COEF_CLASSES},
                      summed={name: float(np.nanmax(np.abs(np.nansum(gap[s, 1:-1], axis=0) * case.dm)))
                              for name, s in blocks.items()})
    # values, jumps and B_m / C_m estimates at the pair stations
    cols = {"j=0": 0, "j=J/2": J // 2, "j=J": J}
    B_, C_ = co.B, co.C
    h = case.dm
    rec = {}
    for side, (a, b_) in (("LE", (f - 1, f)), ("TE", (l, l + 1))):
        for cname, j in cols.items():
            rec[f"dB_{side}_{cname}"] = float(B_[b_, j] - B_[a, j])
        rec[f"dC_{side}"] = float(C_[b_, J // 2] - C_[a, J // 2])
    for cname, j in cols.items():
        rec[f"dB_rear_{cname}"] = float(B_[rear + 1, j] - B_[rear, j])
        rec[f"dB_f+1_{cname}"] = float(B_[f + 2, j] - B_[f + 1, j])
    rec["dB_periodic"] = float(B_[f // 2 + 1, J // 2] - B_[f // 2, J // 2])
    rec["dC_rear"] = float(C_[rear + 1, J // 2] - C_[rear, J // 2])
    met = case.metrics
    for cname, j in cols.items():
        bj = B_[:, j]
        eta_j = float(case.grid.eta[j])

        def exact_blade(i):
            return float((1 - eta_j) * met.dtan_beta0_dm[i] + eta_j * met.dtan_beta1_dm[i])
        rec[f"Bm_f-1_central_{cname}"] = (bj[f] - bj[f - 2]) / (2 * h)
        rec[f"Bm_f-1_side_{cname}"] = (3 * bj[f - 1] - 4 * bj[f - 2] + bj[f - 3]) / (2 * h)
        rec[f"Bm_f-1_exact_{cname}"] = float(dtb_side[f - 1])
        rec[f"Bm_f_central_{cname}"] = (bj[f + 1] - bj[f - 1]) / (2 * h)
        rec[f"Bm_f_side_{cname}"] = (-3 * bj[f] + 4 * bj[f + 1] - bj[f + 2]) / (2 * h)
        rec[f"Bm_f_exact_{cname}"] = exact_blade(f)
        rec[f"Bm_f+1_central_{cname}"] = (bj[f + 2] - bj[f]) / (2 * h)
        rec[f"Bm_f+1_exact_{cname}"] = exact_blade(f + 1)
        rec[f"Bm_l-1_central_{cname}"] = (bj[l] - bj[l - 2]) / (2 * h)
        rec[f"Bm_l-1_exact_{cname}"] = exact_blade(l - 1)
        rec[f"Bm_l_central_{cname}"] = (bj[l + 1] - bj[l - 1]) / (2 * h)
        rec[f"Bm_l_side_{cname}"] = (3 * bj[l] - 4 * bj[l - 1] + bj[l - 2]) / (2 * h)
        rec[f"Bm_l_exact_{cname}"] = exact_blade(l)
        rec[f"Bm_l+1_central_{cname}"] = (bj[l + 2] - bj[l]) / (2 * h)
        rec[f"Bm_l+1_side_{cname}"] = (-3 * bj[l + 1] + 4 * bj[l + 2] - bj[l + 3]) / (2 * h)
        rec[f"Bm_l+1_exact_{cname}"] = float(dtb_side[l + 1])
    c_ = C_[:, J // 2]
    S_m = np.asarray(met.dS_dm)
    rec.update({"Cm_f-1_central": (c_[f] - c_[f - 2]) / (2 * h), "Cm_f-1_side": (3 * c_[f - 1] - 4 * c_[f - 2] + c_[f - 3]) / (2 * h),
                "Cm_f_central": (c_[f + 1] - c_[f - 1]) / (2 * h), "Cm_f_side": (-3 * c_[f] + 4 * c_[f + 1] - c_[f + 2]) / (2 * h),
                "Cm_f_exact": float(S_m[f]), "Cm_f+1_central": (c_[f + 2] - c_[f]) / (2 * h), "Cm_f+1_exact": float(S_m[f + 1]),
                "Cm_l-1_central": (c_[l] - c_[l - 2]) / (2 * h), "Cm_l-1_exact": float(S_m[l - 1]),
                "Cm_l_central": (c_[l + 1] - c_[l - 1]) / (2 * h), "Cm_l_side": (3 * c_[l] - 4 * c_[l - 1] + c_[l - 2]) / (2 * h),
                "Cm_l_exact": float(S_m[l]), "Cm_l+1_central": (c_[l + 2] - c_[l]) / (2 * h),
                "Cm_l+1_side": (-3 * c_[l + 1] + 4 * c_[l + 2] - c_[l + 3]) / (2 * h),
                "cut_f": float(b.periodic_translation[1] - met.S[f]), "cut_l": float(b.periodic_translation[1] - met.S[l]),
                "m_f-m_le": float(case.grid.m[f] - b.m_le), "m_te-m_l": float(b.m_te - case.grid.m[l])})
    rec["window"] = {}
    for side, centre in (("LE", f), ("TE", l)):
        idx = np.arange(centre - 3, centre + 4)
        for cname, j in cols.items():
            rec["window"][(side, cname)] = dict(i=idx, m=np.asarray(case.grid.m)[idx], A=co.A[idx, j], B=co.B[idx, j],
                                                C=co.C[idx, j], D=co.D[idx, j], E=co.E[idx, j], D_side=D_side[idx, j])
    res["near"] = rec
    return res


# --- main ---------------------------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--save", type=Path, default=None)
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--quick", action="store_true", help="33 deg only")
    args = parser.parse_args()
    if args.save is not None:
        args.save.mkdir(parents=True, exist_ok=True)
    staggers = (PRIMARY,) if args.quick else STAGGERS

    print("1. Node / equation classification, read from the production objects (33 deg, n_passage 29, n_eta 7: J = 6)")
    demo = Case(33.0, 29, n_eta=7)
    f, l, J = demo.f, demo.l, demo.J
    print(f"   f = {f}, l = {l}, n_m = {demo.shape[0]}; blade stations f .. l; periodic 0 .. f-1 and l+1 .. n_m-1")
    A7, B7, C7, D7, E7 = synthetic(demo.M, demo.H, "shifted")
    st7, rows7 = demo.rows(A7, B7, C7, D7, E7)
    by_target = {r.target: r for r in rows7}
    for target in ((f - 2, J), (f - 1, J), (l + 1, J), (l + 2, J)):
        r = by_target[target]
        rel = sorted((t.node[0] - target[0], t.node[1]) for t in r.terms)
        print(f"   periodic row {target} '{r.kind}': reads (di, j) {rel}; rhs {r.rhs:+.4f}; center {r.coefficient(target):+.4f}")
    walls, per_side = demo.bcs.blade_wall, demo.bcs.periodic_side
    for name, (i, j) in (("(f-1,1)", (f - 1, 1)), ("(f-1,J-1)", (f - 1, J - 1)), ("(f,1)", (f, 1)), ("(f,J-1)", (f, J - 1)),
                         ("(f-1,j) mid", (f - 1, J // 2)), ("(f,j) mid", (f, J // 2)),
                         ("(l,1)", (l, 1)), ("(l,J-1)", (l, J - 1)), ("(l+1,1)", (l + 1, 1)), ("(l+1,J-1)", (l + 1, J - 1))):
        nb = [(i + di, j + dj) for di in (-1, 0, 1) for dj in (-1, 0, 1) if (di, dj) != (0, 0)]
        print(f"   interior row {name} = ({i},{j}): reads stations {sorted({p[0] for p in nb})}; wall nodes {[p for p in nb if walls[p]]}; "
              f"periodic-line nodes {[p for p in nb if per_side[p]]}")
    for k in (f - 1, l + 1):
        r = by_target[(k, J)]
        print(f"   periodic row ({k},J): wall nodes read {[t.node for t in r.terms if walls[t.node]]}; ghost (k, J+1) = (k, 1) + 1 "
              f"for k in {sorted({k} | {k + o for o, _ in dz.MIXED_M_DIFFERENCE[r.kind]})}")
    for k in (f, l):
        try:
            demo.bcs.periodic_ghost(k, J + 1)
            print(f"   periodic_ghost({k}, J+1): defined")
        except BoundaryConditionError as error:
            print(f"   periodic_ghost({k}, J+1), blade station: BoundaryConditionError: {str(error)[:120]} ...")
    kinds = {}
    for rel in demo.bcs.relations:
        kinds.setdefault(rel.kind, []).append(rel)
    print("   relations: " + "; ".join(f"{k} {len(v)} (terms per relation {sorted({len(r.terms) for r in v})})" for k, v in kinds.items()))
    print(f"   D, E computed but never read by a row: (f,0) (f,J) (l,0) (l,J) (Dirichlet), (f-1,0) (l+1,0) (constraint side)")

    results, hs = {}, {}
    for st in staggers:
        for n in SEQUENCES[st]:
            case = Case(st, n)
            results[(st, n)] = measure(case)
            hs[(st, n)] = case.dm
            r = results[(st, n)]
            print(f"   built {st:.0f} deg n {n}: dm {r['dm']:.5f}, deta {r['deta']:.5f}, n_m {r['n_m']}, f {r['f']}, l {r['l']}, "
                  f"m_f - m_le {r['near']['m_f-m_le']:.5f}, m_te - m_l {r['near']['m_te-m_l']:.5f}")

    def seq(st):
        keys = [(st, n) for n in SEQUENCES[st]]
        return keys, [hs[k] for k in keys]

    for variant in VARIANTS:
        print(f"\n2{'a' if variant == 'brief' else 'b'}. Manufactured psi '{variant}': production operators against the exact "
              f"derivatives, by node class. Linf; L2 = RMS; orders consecutive (33 deg) and first-to-last (all)")
        keys33, h33 = seq(PRIMARY)
        for op in OPERATORS:
            for c in CLASSES:
                vals = [results[k][f"deriv_{variant}"][op][c][0] for k in keys33]
                l2 = [results[k][f"deriv_{variant}"][op][c][1] for k in keys33]
                print(f"     33 {op:10s} {c:24s} Linf " + " ".join(f"{v:.2e}" for v in vals) + "  p "
                      + " ".join(f"{o:5.2f}" for o in orders_consecutive(vals, h33)) + "  L2 " + " ".join(f"{v:.2e}" for v in l2))
        for st in staggers:
            keys, h = seq(st)
            for op in OPERATORS:
                print(f"   {st:.0f} {op:10s}: " + ", ".join(
                    f"{c.split(' ')[0]} {results[keys[-1]][f'deriv_{variant}'][op][c][0]:.1e}/p{order([results[k][f'deriv_{variant}'][op][c][0] for k in keys], h):.2f}"
                    for c in CLASSES))
    print("   wall and periodic-jump relations on the brief psi: max |residual| "
          f"{max(r['relations'] for r in results.values()):.1e}; far rows (-3 4 -1)/(2 dm) psi_m error, 33 deg: "
          + " ".join(f"{results[k]['far_rows'][0]:.1e}/{results[k]['far_rows'][1]:.1e}" for k in seq(PRIMARY)[0])
          + f" (orders {order([results[k]['far_rows'][0] for k in seq(PRIMARY)[0]], seq(PRIMARY)[1]):.2f}, "
          f"{order([results[k]['far_rows'][1] for k in seq(PRIMARY)[0]], seq(PRIMARY)[1]):.2f})")

    for variant in VARIANTS:
        print(f"\n3{'a' if variant == 'brief' else 'b'}. Synthetic smooth coefficients '{variant}' with psi '{variant}': full-row "
              f"truncation raw - Q by class (analytic D, E | D, E by coefficients.py's differences)")
        for st in staggers:
            keys, h = seq(st)
            print(f"   {st:.0f} deg analytic: " + ", ".join(
                f"{c.split(' ')[0]} {results[keys[-1]][f'synth_{variant}'][c][0]:.1e}/p{order([results[k][f'synth_{variant}'][c][0] for k in keys], h):.2f}"
                for c in CLASSES))
            print(f"   {st:.0f} deg prod-diff: " + ", ".join(
                f"{c.split(' ')[0]} {results[keys[-1]][f'synthprod_{variant}']['tau'][c][0]:.1e}/p{order([results[k][f'synthprod_{variant}']['tau'][c][0] for k in keys], h):.2f}"
                for c in CLASSES))
            print(f"   {st:.0f} deg D error: " + ", ".join(
                f"{c.split(' ')[0]} {results[keys[-1]][f'synthprod_{variant}']['D'][c][0]:.1e}/p{order([results[k][f'synthprod_{variant}']['D'][c][0] for k in keys], h):.2f}"
                for c in CLASSES) + "; E error: " + ", ".join(
                f"{c.split(' ')[0]} {results[keys[-1]][f'synthprod_{variant}']['E'][c][0]:.1e}/p{order([results[k][f'synthprod_{variant}']['E'][c][0] for k in keys], h):.2f}"
                for c in CLASSES))

    print("\n4. Production coefficients (real grid, rho = 1, mdot' = 1)")
    print("   jumps across the pairs (smooth pairs fall like dm):")
    for st in staggers:
        for k in seq(st)[0]:
            r = results[k]["near"]
            print(f"   {st:.0f}/{k[1]:3d} dm {results[k]['dm']:.5f}: B f-1->f j=0 {r['dB_LE_j=0']:+.3f} J/2 {r['dB_LE_j=J/2']:+.3f} "
                  f"J {r['dB_LE_j=J']:+.3f} | f+1->f+2 j=0 {r['dB_f+1_j=0']:+.3f} J {r['dB_f+1_j=J']:+.3f} | rear pair j=0 "
                  f"{r['dB_rear_j=0']:+.1e} | inlet pair {r['dB_periodic']:+.1e};  C f-1->f {r['dC_LE']:+.4f} (-cut(m_f)); "
                  f"TE: B l->l+1 j=0 {r['dB_TE_j=0']:+.3f} J/2 {r['dB_TE_j=J/2']:+.3f} J {r['dB_TE_j=J']:+.3f}; C {r['dC_TE']:+.4f} "
                  f"(cut(m_l)); C rear pair {r['dC_rear']:+.1e}")
    print("   jump orders p (|jump| ~ dm^p; smooth: p = 1):")
    for st in staggers:
        keys, h = seq(st)
        names = ("dB_LE_j=0", "dB_LE_j=J", "dC_LE", "dB_TE_j=0", "dB_TE_j=J", "dC_TE", "dB_f+1_j=0", "dB_rear_j=0", "dB_periodic", "dC_rear")
        print(f"   {st:.0f} deg: " + ", ".join(f"{nm} {order([abs(results[k]['near'][nm]) for k in keys], h):+.2f}" for nm in names))
    for cname in ("j=0", "j=J/2", "j=J"):
        print(f"   B_m at {cname}: production central | same-side one-sided | exact same-side")
        for st in staggers:
            for k in seq(st)[0]:
                r = results[k]["near"]
                print(f"   {st:.0f}/{k[1]:3d}: f-1 {r[f'Bm_f-1_central_{cname}']:+9.2f} | {r[f'Bm_f-1_side_{cname}']:+8.4f} | "
                      f"{r[f'Bm_f-1_exact_{cname}']:+8.4f};  f {r[f'Bm_f_central_{cname}']:+9.2f} | {r[f'Bm_f_side_{cname}']:+9.2f} | "
                      f"{r[f'Bm_f_exact_{cname}']:+9.2f};  f+1 {r[f'Bm_f+1_central_{cname}']:+9.2f} | {r[f'Bm_f+1_exact_{cname}']:+9.2f};  "
                      f"l-1 {r[f'Bm_l-1_central_{cname}']:+9.2f} | {r[f'Bm_l-1_exact_{cname}']:+9.2f};  l {r[f'Bm_l_central_{cname}']:+9.2f} | "
                      f"{r[f'Bm_l_side_{cname}']:+9.2f} | {r[f'Bm_l_exact_{cname}']:+9.2f};  l+1 {r[f'Bm_l+1_central_{cname}']:+9.2f} | "
                      f"{r[f'Bm_l+1_side_{cname}']:+8.4f} | {r[f'Bm_l+1_exact_{cname}']:+8.4f}")
    print("   C_m (= S_m): production central | same-side one-sided | exact same-side (0 on periodic stations)")
    for st in staggers:
        for k in seq(st)[0]:
            r = results[k]["near"]
            print(f"   {st:.0f}/{k[1]:3d}: f-1 {r['Cm_f-1_central']:+8.3f} | {r['Cm_f-1_side']:+.0e} | 0;  f {r['Cm_f_central']:+8.3f} | "
                  f"{r['Cm_f_side']:+8.3f} | {r['Cm_f_exact']:+8.3f};  f+1 {r['Cm_f+1_central']:+8.3f} | {r['Cm_f+1_exact']:+8.3f};  "
                  f"l-1 {r['Cm_l-1_central']:+8.3f} | {r['Cm_l-1_exact']:+8.3f};  l {r['Cm_l_central']:+8.3f} | {r['Cm_l_side']:+8.3f} | "
                  f"{r['Cm_l_exact']:+8.3f};  l+1 {r['Cm_l+1_central']:+8.3f} | {r['Cm_l+1_side']:+.0e} | 0")
    for q in ("D", "E"):
        print(f"   Linf |{q} - {q}_side| by class (value at the finest / order first to last)" + ("; |D_side| for scale" if q == "D" else ""))
        for st in staggers:
            keys, h = seq(st)
            parts = []
            for c in COEF_CLASSES:
                vals = [results[k]["coef"][q][c][0] for k in keys]
                extra = f" [|D_side| {results[keys[-1]]['coef']['Dmag'][c][0]:.1e}]" if q == "D" else ""
                parts.append(f"{c.split(' (')[0]} {vals[-1]:.2e}/p{order(vals, h):+.2f}{extra}")
            print(f"   {st:.0f} deg: " + "; ".join(parts))
    print("   real-coefficient row truncation on psi 'shifted' against Q with the exact same-side D, E; relative = |tau| / "
          "(|A| + 2|B| + |C| + |D_side| + |E_side|). production D, E | exact same-side D, E (the stencil alone)")
    for st in staggers:
        keys, h = seq(st)
        parts = []
        for c in COEF_CLASSES:
            a = [results[k]["decomp"]["production_rel"][c][0] for k in keys]
            b_ = [results[k]["decomp"]["stencil_rel"][c][0] for k in keys]
            parts.append(f"{c.split(' (')[0]} {a[-1]:.1e}/p{order(a, h):+.2f} | {b_[-1]:.1e}/p{order(b_, h):+.2f}")
        print(f"   {st:.0f} deg: " + "; ".join(parts))
    print("   expanded production row minus a conservative face-flux row, same A, B, C, psi 'shifted': pointwise Linf | m-sum "
          "over 4-station blocks of gap * dm (max over j)")
    for st in staggers:
        keys, h = seq(st)
        pw = ", ".join(f"{c.split(' (')[0]} {results[keys[-1]]['gap']['pointwise'][c][0]:.1e}/p{order([results[k]['gap']['pointwise'][c][0] for k in keys], h):+.2f}"
                       for c in ("ordinary periodic", "blade rear (65-95 %)", "f-1 interior", "f interior", "f+1 interior",
                                 "l-1 interior", "l interior", "l+1 interior"))
        sm = ", ".join(f"{nm} {results[keys[-1]]['gap']['summed'][nm]:.1e}/p{order([results[k]['gap']['summed'][nm] for k in keys], h):+.2f}"
                       for nm in ("ordinary periodic", "blade rear", "LE f-2..f+1", "TE l-1..l+2"))
        print(f"   {st:.0f} deg pointwise: {pw}\n   {st:.0f} deg summed:    {sm}")

    print("\n5. Special production rows against pure central diagnostics (* = the central form reads (f or l, J+1), which does not "
          "exist; evaluated from the analytic psi)")
    for variant in VARIANTS:
        for st in staggers:
            keys, h = seq(st)
            for name, prod, cent in (("LE (f-1,J) psi_m,eta backward | central*", "le_meta_prod", "le_meta_central"),
                                     ("TE (l+1,J) psi_m forward | central", "te_m_prod", "te_m_central"),
                                     ("TE (l+1,J) psi_m,m forward | central", "te_mm_prod", "te_mm_central"),
                                     ("TE (l+1,J) psi_m,eta forward | central*", "te_meta_prod", "te_meta_central"),
                                     ("LE row full (synthetic) production | central*", "le_row_prod", "le_row_central"),
                                     ("TE row full (synthetic) production | central*", "te_row_prod", "te_row_central")):
                a = [abs(results[k][f"special_{variant}"][prod]) for k in keys]
                b_ = [abs(results[k][f"special_{variant}"][cent]) for k in keys]
                print(f"   {variant:7s} {st:.0f} {name:46s} " + " ".join(f"{x:.2e}|{y:.2e}" for x, y in zip(a, b_))
                      + f"  p {order(a, h):.2f}|{order(b_, h):.2f}; ratio finest {a[-1] / b_[-1] if b_[-1] else float('nan'):.2f}")
    for st in staggers:
        keys, h = seq(st)
        print(f"   {st:.0f} deg TE row: center coefficient " + " ".join(f"{results[k]['special_shifted']['te_center']:+.3f}" for k in keys)
              + "; largest |coefficient| " + " ".join(f"{results[k]['special_shifted']['te_largest']:.2f}" for k in keys)
              + "; LE row central psi_m / psi_m,m errors (shifted) " + " ".join(
                  f"{abs(results[k]['special_shifted']['le_m_prod']):.1e}/{abs(results[k]['special_shifted']['le_mm_prod']):.1e}" for k in keys)
              + f" (p {order([abs(results[k]['special_shifted']['le_m_prod']) for k in keys], h):.2f}/"
              f"{order([abs(results[k]['special_shifted']['le_mm_prod']) for k in keys], h):.2f})")

    if args.save is None:
        return
    write_csv(args.save, results, hs)
    figures = make_figures(demo, by_target, results, hs, staggers)
    for name, fg in figures.items():
        fg.savefig(args.save / f"{name}.png", dpi=120)
        print(f"wrote {args.save / (name + '.png')}")
    if not args.no_show:
        plt.show()


def write_csv(out, results, hs):
    with open(out / "derivative_errors.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["stagger_deg", "n_passage", "dm", "psi", "operator", "node_class", "linf", "l2", "nodes"])
        for (st, n), r in results.items():
            for variant in VARIANTS:
                for op, by in r[f"deriv_{variant}"].items():
                    for c, (a, b, k) in by.items():
                        w.writerow([st, n, hs[(st, n)], variant, op, c, a, b, k])
    with open(out / "pde_residual.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["stagger_deg", "n_passage", "dm", "test", "node_class", "linf", "l2", "nodes"])
        for (st, n), r in results.items():
            for variant in VARIANTS:
                for c, (a, b, k) in r[f"synth_{variant}"].items():
                    w.writerow([st, n, hs[(st, n)], f"synthetic {variant}, analytic D E", c, a, b, k])
                for c, (a, b, k) in r[f"synthprod_{variant}"]["tau"].items():
                    w.writerow([st, n, hs[(st, n)], f"synthetic {variant}, production-differentiated D E", c, a, b, k])
            for part in ("production", "stencil", "production_rel", "stencil_rel"):
                for c, (a, b, k) in r["decomp"][part].items():
                    w.writerow([st, n, hs[(st, n)], f"real coefficients, {part}", c, a, b, k])
            for c, (a, b, k) in r["gap"]["pointwise"].items():
                w.writerow([st, n, hs[(st, n)], "expanded minus conservative", c, a, b, k])
            for nm, v in r["gap"]["summed"].items():
                w.writerow([st, n, hs[(st, n)], "expanded minus conservative, m-summed 4-station block", nm, v, "", ""])
    with open(out / "coefficient_errors.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["stagger_deg", "n_passage", "dm", "quantity", "node_class", "linf", "l2", "nodes"])
        for (st, n), r in results.items():
            for q in ("D", "E", "Dmag"):
                for c, (a, b, k) in r["coef"][q].items():
                    w.writerow([st, n, hs[(st, n)], {"D": "D - D_side", "E": "E - E_side", "Dmag": "|D_side|"}[q], c, a, b, k])
    with open(out / "coefficient_transition.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        keys = sorted(k for k in next(iter(results.values()))["near"] if k != "window")
        w.writerow(["stagger_deg", "n_passage", "dm"] + keys)
        for (st, n), r in results.items():
            w.writerow([st, n, hs[(st, n)]] + [r["near"][k] for k in keys])
    with open(out / "coefficient_windows.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["stagger_deg", "n_passage", "edge", "column", "i", "offset_from_f_or_l", "m", "A", "B", "C", "D", "E", "D_side"])
        for (st, n), r in results.items():
            for (side, cname), v in r["near"]["window"].items():
                centre = r["f"] if side == "LE" else r["l"]
                for k in range(len(v["i"])):
                    w.writerow([st, n, side, cname, int(v["i"][k]), int(v["i"][k] - centre), v["m"][k], v["A"][k], v["B"][k],
                                v["C"][k], v["D"][k], v["E"][k], v["D_side"][k]])
    with open(out / "special_rows.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        names = sorted(next(iter(results.values()))["special_brief"].keys())
        w.writerow(["stagger_deg", "n_passage", "dm", "psi"] + names)
        for (st, n), r in results.items():
            for variant in VARIANTS:
                w.writerow([st, n, hs[(st, n)], variant] + [r[f"special_{variant}"][k] for k in names])


# --- figures -----------------------------------------------------------------------------------------

def style(ax, title):
    ax.set_title(title, fontsize=9, color=INK, loc="left")
    ax.tick_params(labelsize=7, colors=MUTED)
    if ax.get_xscale() == "log":          # short decades: minor labels collide
        ax.xaxis.set_minor_formatter(plt.NullFormatter())
    for spine in ax.spines.values():
        spine.set_color(GRIDLINE)
    ax.grid(True, color=GRIDLINE, lw=0.4, alpha=0.6)


def slope_guide(ax, hs, ref, p=2.0, label="slope 2"):
    h = np.array([min(hs), max(hs)])
    ax.loglog(h, ref * (h / h[0]) ** p, color=GRIDLINE, lw=1.2, ls="--", label=label)


def connectivity_panel(ax, demo, target, by_target, edge):
    f, l, J = demo.f, demo.l, demo.J
    centre = f if edge == "LE" else l + 1
    i_lo, i_hi = (centre - 3, centre + 2) if edge == "LE" else (centre - 3, centre + 3)
    walls, per = demo.bcs.blade_wall, demo.bcs.periodic_side
    for i in range(i_lo, i_hi + 1):
        x = i - centre
        for j in range(0, J + 1):
            if walls[i, j]:
                ax.plot(x, j, "s", color=INK, ms=7, zorder=3)
            elif per[i, j] and j == 0:
                ax.plot(x, j, "o", mfc="white", mec="#2a78d6", mew=1.5, ms=7, zorder=3)
            elif per[i, j]:
                ax.plot(x, j, "o", color="#2a78d6", ms=7, zorder=3)
            else:
                ax.plot(x, j, ".", color=MUTED, ms=6, zorder=3)
        if not demo.blade[i]:
            ax.plot(x, J + 1, "o", mfc="none", mec=MUTED, ms=7, zorder=2, alpha=0.6)
        else:
            ax.plot(x, J + 1, "x", color=GRIDLINE, ms=6, zorder=2)
    ax.axvline(-0.5, color="#cf222e", lw=1.0, ls=":")
    ax.text(-0.55, J + 1.6, "periodic" if edge == "LE" else "blade", ha="right", fontsize=7, color=MUTED)
    ax.text(-0.45, J + 1.6, "blade" if edge == "LE" else "periodic", ha="left", fontsize=7, color=MUTED)
    ti, tj = target
    color = "#cf222e"
    ghosts = []
    if target in by_target:
        row = by_target[target]
        reads = [t.node for t in row.terms if t.node != target]
        ghosts = sorted({ti} | {ti + o for o, _ in dz.MIXED_M_DIFFERENCE[row.kind]})
        kind = row.kind
    else:
        reads = [(ti + di, tj + dj) for di in (-1, 0, 1) for dj in (-1, 0, 1) if (di, dj) != (0, 0)]
        kind = "interior 9-point"
    for (i, j) in reads:
        ax.plot([ti - centre, i - centre], [tj, j], color=color, lw=1.0, alpha=0.7, zorder=2)
        ax.plot(i - centre, j, "o", mfc="none", mec=color, mew=1.6, ms=11, zorder=4)
    for k in ghosts:
        ax.plot([ti - centre, k - centre], [tj, J + 1], color="#9a6700", lw=1.0, ls="--", zorder=2)
        ax.plot(k - centre, J + 1, "o", mfc="none", mec="#9a6700", mew=1.6, ms=11, zorder=4)
        ax.annotate("", xy=(k - centre + 0.2, 1.2), xytext=(k - centre + 0.2, J + 0.8),
                    arrowprops=dict(arrowstyle="->", color="#9a6700", lw=0.8, alpha=0.55))
    ax.plot(ti - centre, tj, "*", color=color, ms=15, zorder=5)
    ax.set_xlim(i_lo - centre - 0.6, i_hi - centre + 0.6)
    ax.set_ylim(-0.7, J + 2.0)
    ticks = list(range(i_lo - centre, i_hi - centre + 1))
    ax.set_xticks(ticks)
    ref, base = (f, "f") if edge == "LE" else (l, "l")
    labels = []
    for k in ticks:
        off = k + centre - ref
        labels.append(f"{base}{off:+d}" if off else base)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_yticks(range(0, J + 2))
    ax.set_yticklabels([("j=" + str(j)) if j <= J else "J+1" for j in range(0, J + 2)], fontsize=7)
    names = {f - 2: "f-2", f - 1: "f-1", f: "f", l: "l", l + 1: "l+1", l + 2: "l+2"}
    style(ax, f"row ({names.get(ti, ti)}, {'J' if tj == J else tj}): {kind}")


def make_figures(demo, by_target, results, hs, staggers):
    figures = {}
    f, l, J = demo.f, demo.l, demo.J
    for name, edge, targets in (("01_node_classification_le", "LE", [(f - 2, J), (f - 1, J), (f - 1, 1), (f - 1, J - 1), (f, 1), (f, J - 1)]),
                                ("02_node_classification_te", "TE", [(l + 1, J), (l + 2, J), (l, 1), (l, J - 1), (l + 1, 1), (l + 1, J - 1)])):
        fig, axes = plt.subplots(2, 3, figsize=(17, 10.5))
        for ax, t in zip(axes.ravel(), targets):
            connectivity_panel(ax, demo, t, by_target, edge)
        handles = [plt.Line2D([], [], marker="s", color=INK, ls="none", ms=7, label="blade wall: Dirichlet psi = 0 / 1, no PDE"),
                   plt.Line2D([], [], marker="o", mfc="white", mec="#2a78d6", ls="none", ms=7, label="periodic j = 0: jump relation only"),
                   plt.Line2D([], [], marker="o", color="#2a78d6", ls="none", ms=7, label="periodic j = J: periodic PDE row + jump"),
                   plt.Line2D([], [], marker=".", color=MUTED, ls="none", ms=7, label="interior: 9-point PDE row"),
                   plt.Line2D([], [], marker="o", mfc="none", mec=MUTED, ls="none", ms=7, label="virtual j = J+1 (periodic stations)"),
                   plt.Line2D([], [], marker="x", color=GRIDLINE, ls="none", ms=7, label="no j = J+1 on a blade station"),
                   plt.Line2D([], [], marker="*", color="#cf222e", ls="none", ms=12, label="row target"),
                   plt.Line2D([], [], marker="o", mfc="none", mec="#cf222e", color="#cf222e", ms=10, label="node the production row reads"),
                   plt.Line2D([], [], marker="o", mfc="none", mec="#9a6700", ls="--", color="#9a6700", ms=10,
                              label="ghost read, replaced by (k, 1) + 1 (arrow)")]
        fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8, frameon=False)
        fig.suptitle(f"{name[:2]}  Production rows at the {edge}: 33 deg, n_passage 29, n_eta 7 (J = 6) for legibility; f = {f}, "
                     f"l = {l}; reads taken from the production row objects; red dotted line: the station pair", fontsize=10.5,
                     color=INK, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0.07, 1, 0.95))
        figures[name] = fig

    for name, op, title in (("03_psi_m_error_by_node_class", "psi_m", "psi_m"),
                            ("04_psi_mm_error_by_node_class", "psi_mm", "psi_m,m"),
                            ("05_psi_meta_error_by_node_class", "psi_meta", "psi_m,eta")):
        fig, axes = plt.subplots(2, len(staggers), figsize=(4.3 * len(staggers), 9.6), squeeze=False)
        for row_k, variant in enumerate(VARIANTS):
            for ax, st in zip(axes[row_k], staggers):
                keys = [(st, n) for n in SEQUENCES[st]]
                h = [hs[k] for k in keys]
                tops = []
                for c in CLASSES:
                    vals = np.array([results[k][f"deriv_{variant}"][op][c][0] for k in keys])
                    tops.append(vals[0])
                    ax.loglog(h, np.maximum(vals, 1e-17), marker=MARKERS[c], color=COLORS[c], lw=1.4, ms=6,
                              ls="--" if "periodic" in c else "-", label=c)
                slope_guide(ax, h, max(tops) * 1.5)
                ax.set_xlabel("dm (admissible sequence)", fontsize=8)
                style(ax, f"{st:.0f} deg, psi '{variant}': Linf |{title}_h - {title}|")
        axes[0][0].legend(fontsize=6.5)
        fig.suptitle(f"{name[:2]}  Manufactured psi: the production {title} operator by node class (extracted from the production "
                     f"rows; n_eta = n_passage + 1). Top: the brief's psi (0 and 1 on eta = 0, 1: periodic-row m differences exact, "
                     f"plotted at 1e-17); bottom: the shifted psi", fontsize=10, color=INK, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        figures[name] = fig

    fig, axes = plt.subplots(2, len(staggers), figsize=(4.3 * len(staggers), 9.6), squeeze=False)
    for col, st in enumerate(staggers):
        keys = [(st, n) for n in SEQUENCES[st]]
        h = [hs[k] for k in keys]
        for row_k, (field, label) in enumerate((("synth_brief", "brief fields, analytic D, E"),
                                                ("synthprod_shifted", "shifted fields, D, E by coefficients.py"))):
            ax = axes[row_k][col]
            tops = []
            for c in CLASSES:
                vals = [results[k][field][c][0] if field.startswith("synth_") else results[k][field]["tau"][c][0] for k in keys]
                tops.append(vals[0])
                ax.loglog(h, vals, marker=MARKERS[c], color=COLORS[c], lw=1.4, ms=6, ls="--" if "periodic" in c else "-", label=c)
            slope_guide(ax, h, max(tops) * 1.5)
            ax.set_xlabel("dm", fontsize=8)
            style(ax, f"{st:.0f} deg, {label}: Linf |raw - Q|")
    axes[0][0].legend(fontsize=6.5)
    fig.suptitle("06  Synthetic smooth A, B, C: truncation of the full production rows by node class (top: the brief's fields, "
                 "analytic D, E; bottom: phase-shifted fields, D, E by coefficients.py's own differences)", fontsize=10.5,
                 color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    figures["06_full_pde_residual_by_node_class"] = fig

    keys33 = [(PRIMARY, n) for n in SEQUENCES[PRIMARY]]
    h33 = [hs[k] for k in keys33]
    shades = plt.cm.viridis(np.linspace(0.0, 0.85, len(keys33)))

    def window_plot(ax, side, cname, quantity, show_side=False):
        for k, color in zip(keys33, shades):
            w = results[k]["near"]["window"][(side, cname)]
            off = w["i"] - (results[k]["f"] if side == "LE" else results[k]["l"])
            ax.plot(off, w[quantity], marker="o", color=color, lw=1.4, label=f"n {k[1]} (dm {results[k]['dm']:.4f})")
            if show_side:
                ax.plot(off, w["D_side"], marker="x", color=color, lw=0.9, ls=":")
        ax.axvline(-0.5 if side == "LE" else 0.5, color="#cf222e", lw=0.8, ls=":")
        ax.set_xlabel("station offset from " + ("f" if side == "LE" else "l"), fontsize=8)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for ax, cname in zip(axes[0], ("j=0", "j=J/2", "j=J")):
        window_plot(ax, "LE", cname, "B")
        style(ax, f"33 deg: B = tan(beta) at {cname} (rho = mdot' = 1), stations f-3 .. f+3")
        ax.legend(fontsize=7)
    ax = axes[1][0]
    for nm, color, label in (("dB_LE_j=0", "#eb6834", "f-1 -> f, j = 0"), ("dB_LE_j=J", "#8250df", "f-1 -> f, j = J"),
                             ("dB_f+1_j=0", "#f5a46b", "f+1 -> f+2, j = 0 (blade side, steep)"),
                             ("dB_rear_j=0", MUTED, "rear blade pair, j = 0"), ("dB_periodic", INK, "inlet pair")):
        ax.loglog(h33, [abs(results[k]["near"][nm]) for k in keys33], marker="o", color=color, label=label)
    slope_guide(ax, h33, abs(results[keys33[0]]["near"]["dB_rear_j=0"]) * 1.5, p=1.0, label="slope 1 (smooth field)")
    ax.set_xlabel("dm", fontsize=8)
    ax.legend(fontsize=7)
    style(ax, "|B[i+1] - B[i]| over station pairs")
    for ax, (st_i, label) in zip(axes[1][1:], (("f-1", "f-1 (periodic side)"), ("f", "f (blade side)"))):
        for cname, color in (("j=0", "#eb6834"), ("j=J/2", "#2a78d6"), ("j=J", "#8250df")):
            ax.semilogx(h33, [results[k]["near"][f"Bm_{st_i}_central_{cname}"] for k in keys33], marker="o", color=color,
                        label=f"production central, {cname}")
            ax.semilogx(h33, [results[k]["near"][f"Bm_{st_i}_side_{cname}"] for k in keys33], marker="s", color=color, ls="--",
                        label=f"same-side one-sided, {cname}")
            ax.semilogx(h33, [results[k]["near"][f"Bm_{st_i}_exact_{cname}"] for k in keys33], marker="x", color=color, ls=":",
                        label=f"exact on its side, {cname}")
        ax.set_yscale("symlog", linthresh=1.0)
        ax.set_xlabel("dm", fontsize=8)
        ax.legend(fontsize=6.5)
        style(ax, f"33 deg: B_m at {label}")
    fig.suptitle("07  B near the LE against refinement (33 deg, admissible n 29, 43, 65, 97): B jumps across f-1 -> f; the "
                 "jump does not fall with dm", fontsize=10.5, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    figures["07_B_near_le_vs_refinement"] = fig

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for ax, cname in zip(axes[0], ("j=0", "j=J/2", "j=J")):
        window_plot(ax, "LE", cname, "D", show_side=True)
        ax.set_yscale("symlog", linthresh=1.0)
        style(ax, f"33 deg: production D (o) and exact same-side D (x, dotted) at {cname}")
        ax.legend(fontsize=7)
    for ax, (q, title) in zip(axes[1][:2], (("D", "|D - D_side|"), ("E", "|E - E_side|, E_side = 0"))):
        for c in ("ordinary periodic", "blade rear (65-95 %)", "blade front (5-60 %)", "f-1 interior", "f interior", "f+1 interior",
                  "LE periodic row (f-1,J)"):
            ax.loglog(h33, [max(results[k]["coef"][q][c][0], 1e-17) for k in keys33], marker=MARKERS[c], color=COLORS[c],
                      ls="--" if "periodic" in c else "-", label=c)
        slope_guide(ax, h33, results[keys33[0]]["coef"][q]["f-1 interior"][0], p=-1.0, label="slope -1 (a jump over 2 dm)")
        ax.set_xlabel("dm", fontsize=8)
        ax.legend(fontsize=6.5)
        style(ax, f"33 deg: Linf {title} by node class")
    ax = axes[1][2]
    for st, color in zip(staggers, plt.cm.viridis(np.linspace(0, 0.9, len(staggers)))):
        keys = [(st, n) for n in SEQUENCES[st]]
        ax.loglog([hs[k] for k in keys], [results[k]["coef"]["D"]["f-1 interior"][0] for k in keys], marker="o", color=color,
                  label=f"{st:.0f} deg f-1")
        ax.loglog([hs[k] for k in keys], [results[k]["coef"]["D"]["f+1 interior"][0] for k in keys], marker="d", ls=":",
                  color=color, label=f"{st:.0f} deg f+1 (no crossing)")
    ax.set_xlabel("dm", fontsize=8)
    ax.legend(fontsize=6.5, ncol=2)
    style(ax, "all staggers: Linf |D - D_side| at f-1 (crossing) and f+1 (same side)")
    fig.suptitle("08  D near the LE against refinement: production D = A_eta - B_m, B_m central across f-1 / f, against the exact "
                 "derivative on the node's own side", fontsize=10.5, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    figures["08_D_near_le_vs_refinement"] = fig

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    window_plot(axes[0][0], "TE", "j=J/2", "C")
    style(axes[0][0], "33 deg: C = S (rho = mdot' = 1) at stations l-3 .. l+3")
    axes[0][0].legend(fontsize=7)
    window_plot(axes[0][1], "TE", "j=J/2", "E")
    axes[0][1].set_yscale("symlog", linthresh=0.01)
    style(axes[0][1], "33 deg: production E at j = J/2 (exact E = 0 on both sides)")
    window_plot(axes[0][2], "TE", "j=J", "E")
    axes[0][2].set_yscale("symlog", linthresh=0.01)
    style(axes[0][2], "33 deg: production E at j = J (the TE periodic row reads E[l+1, J])")
    ax = axes[1][0]
    for nm, color, label in (("dC_TE", "#8250df", "|C[l+1] - C[l]| = cut(m_l)"), ("dC_LE", "#2a78d6", "|C[f] - C[f-1]| = cut(m_f)"),
                             ("dC_rear", MUTED, "rear blade pair")):
        ax.loglog(h33, [abs(results[k]["near"][nm]) for k in keys33], marker="o", color=color, label=label)
    slope_guide(ax, h33, abs(results[keys33[0]]["near"]["dC_rear"]) * 1.5, p=1.0, label="slope 1 (smooth field)")
    ax.set_xlabel("dm", fontsize=8)
    ax.legend(fontsize=7)
    style(ax, "33 deg: jumps of C over station pairs")
    ax = axes[1][1]
    for c in ("ordinary periodic", "blade rear (65-95 %)", "l-1 interior", "l interior", "l+1 interior", "TE periodic row (l+1,J)"):
        ax.loglog(h33, [max(results[k]["coef"]["E"][c][0], 1e-17) for k in keys33], marker=MARKERS[c], color=COLORS[c],
                  ls="--" if "periodic" in c else "-", label=c)
    ax.set_xlabel("dm", fontsize=8)
    ax.legend(fontsize=6.5)
    style(ax, "33 deg: Linf |E| by node class (exact E = 0)")
    ax = axes[1][2]
    for st, color in zip(staggers, plt.cm.viridis(np.linspace(0, 0.9, len(staggers)))):
        keys = [(st, n) for n in SEQUENCES[st]]
        ax.loglog([hs[k] for k in keys], [results[k]["coef"]["E"]["l+1 interior"][0] for k in keys], marker="o", color=color,
                  label=f"{st:.0f} deg l+1")
        ax.loglog([hs[k] for k in keys], [max(results[k]["coef"]["E"]["l-1 interior"][0], 1e-17) for k in keys], marker="<",
                  ls=":", color=color, label=f"{st:.0f} deg l-1 (no crossing)")
    ax.set_xlabel("dm", fontsize=8)
    ax.legend(fontsize=6.5, ncol=2)
    style(ax, "all staggers: Linf |E| at l+1 (crossing) and l-1 (same side)")
    fig.suptitle("09  C and E near the TE against refinement: C = S jumps by the cut of the last blade station, and E = C_m - B_eta "
                 "takes that jump over 2 dm", fontsize=10.5, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    figures["09_C_and_E_near_te_vs_refinement"] = fig

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    panels = (("le_meta_prod", "le_meta_central", "LE (f-1, J) psi_m,eta: backward (production) vs central*"),
              ("te_m_prod", "te_m_central", "TE (l+1, J) psi_m: forward (production) vs central"),
              ("te_mm_prod", "te_mm_central", "TE (l+1, J) psi_m,m: forward (production) vs central"),
              ("te_meta_prod", "te_meta_central", "TE (l+1, J) psi_m,eta: forward (production) vs central*"),
              ("le_row_prod", "le_row_central", "LE row, full, shifted synthetic coefficients"),
              ("te_row_prod", "te_row_central", "TE row, full, shifted synthetic coefficients"))
    for ax, (prod, cent, title) in zip(axes.ravel(), panels):
        for st, color in zip(staggers, plt.cm.viridis(np.linspace(0, 0.9, len(staggers)))):
            keys = [(st, n) for n in SEQUENCES[st]]
            h = [hs[k] for k in keys]
            ax.loglog(h, [abs(results[k]["special_shifted"][prod]) for k in keys], marker="o", color=color, label=f"{st:.0f} deg production")
            ax.loglog(h, [abs(results[k]["special_shifted"][cent]) for k in keys], marker="s", ls="--", color=color,
                      label=f"{st:.0f} deg central")
        ax.set_xlabel("dm", fontsize=8)
        style(ax, title)
    axes[0][0].legend(fontsize=6.5, ncol=2)
    fig.suptitle("10  Special production rows against pure central diagnostics, psi 'shifted' (* = the central form reads (f or l, "
                 "J+1), a node that does not exist; taken from the analytic psi)", fontsize=10.5, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    figures["10_production_vs_central_transition_rows"] = fig
    return figures


if __name__ == "__main__":
    main()
