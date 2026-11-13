"""
Audit of coefficients.py on the default C2 cascade, the 65-(12A10)10 at 33 deg stagger,
solidity 1, inlet 45 deg: the numbers of the 2026-11-12 coefficient report. Print only.

    python scripts/audit_coefficients.py

1. Ranges of A to E at n_passage 101 (n_eta 31), rho = 1, mdot' = 1, normalized mode,
   with the node, region and m of each extremum.
2. The first and last five blade stations, both walls: S*, beta, cos(beta), rho, A to E,
   and A S* rho / mdot' against 1 / cos^2(beta).
3. The corner stations either side of each region transition (side line -> blade wall at
   first_blade_i, wall -> side line after last_blade_i): D and E there, how they grow
   with refinement, and the telescoping sum of the central difference across the corner.
4. D and E against an independent reference at the interior blade nodes (>= 0.05 c in m
   from the first and last blade stations): Aungier's D and E expanded by hand, with
   metrics.py's exact C2 derivatives (dS/dm, d tan(beta)/dm) and an analytic density,
   n_passage 51 to 401; the observed orders.
5. The chord scaling: fixed mdot', and physically similar twins (mdot' ~ c), physical mode
   with c_ref = 0.1 m.

Everything is read through the package (geometry -> grid -> metrics -> coefficients), and
the references of section 4 are written here, not taken from coefficients.py.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from turbolenza.b2b.coefficients import build_coefficients  # noqa: E402
from turbolenza.b2b.config import GridSettings  # noqa: E402
from turbolenza.b2b.geometry import CascadeSpec, make_cascade  # noqa: E402
from turbolenza.b2b.grid import make_grid  # noqa: E402
from turbolenza.b2b.metrics import build_metrics  # noqa: E402

SPEC = CascadeSpec(blade_type="NACA65", solidity=1.0, stagger_deg=33.0, cl0=1.2, t_max_over_c=0.10,
                   surface_continuity="C2")
INLET = math.radians(45.0)
N_ETA = 31
CASCADE = make_cascade(SPEC)


def metrics_at(n_passage: int, n_eta: int = N_ETA):
    return build_metrics(make_grid(CASCADE, GridSettings(n_passage=n_passage, n_eta=n_eta), INLET))


def region_of(g, i: int) -> str:
    return "inlet" if i < g.first_blade_i else ("exit" if i > g.last_blade_i else "blade")


def density(m: np.ndarray, eta: np.ndarray, m_in: float, m_out: float):
    """A smooth positive density, periodic in eta, and its analytic derivatives in m* and eta."""
    k = 2.0 * math.pi / (m_out - m_in)
    M, H = np.meshgrid(m, eta, indexing="ij")
    u = k * (M - m_in)
    w = 2.0 * math.pi * H
    rho = 1.0 + 0.10 * np.sin(u + 0.3) + 0.08 * np.sin(w) + 0.05 * np.cos(u) * np.sin(w + 0.5)
    rho_m = 0.10 * k * np.cos(u + 0.3) - 0.05 * k * np.sin(u) * np.sin(w + 0.5)
    rho_eta = 0.08 * 2.0 * math.pi * np.cos(w) + 0.05 * np.cos(u) * 2.0 * math.pi * np.cos(w + 0.5)
    return rho, rho_m, rho_eta


def expanded_reference(metrics, rho, rho_m, rho_eta, mdot: float, length: float):
    """D and E of Eqs. 5-25 and 5-26 expanded by the product rule, with t = tan(beta):
        A = mdot (1 + t^2) / (L S* rho),  B = mdot t / rho,  C = mdot L S* / rho,
        A_eta = mdot / (L S*) [2 t t_eta / rho - (1 + t^2) rho_eta / rho^2],   t_eta = dS*/dm*,
        B_m   = mdot / L [t_m* / rho - t rho_m* / rho^2],
        C_m   = mdot [S*_m* / rho - S* rho_m* / rho^2],
        B_eta = mdot [t_eta / rho - t rho_eta / rho^2],
    the metric derivatives from metrics.py (dS_dm, dtan_beta_dm: exact on C2 blade walls)."""
    S = metrics.S[:, None]
    t = metrics.tan_beta
    t_eta = metrics.dS_dm[:, None]
    t_m = metrics.dtan_beta_dm
    S_m = metrics.dS_dm[:, None]
    A_eta = mdot / (length * S) * (2.0 * t * t_eta / rho - (1.0 + t * t) * rho_eta / rho ** 2)
    B_m = mdot / length * (t_m / rho - t * rho_m / rho ** 2)
    C_m = mdot * (S_m / rho - S * rho_m / rho ** 2)
    B_eta = mdot * (t_eta / rho - t * rho_eta / rho ** 2)
    return A_eta - B_m, C_m - B_eta


def orders(errors, spacings):
    return [math.log(errors[k] / errors[k + 1]) / math.log(spacings[k] / spacings[k + 1])
            for k in range(len(errors) - 1)]


def section_1_and_2() -> None:
    metrics = metrics_at(101)
    g = metrics.grid
    rho = np.ones((len(g.m), len(g.eta)))
    co = build_coefficients(metrics, rho, 1.0)
    f, l = g.first_blade_i, g.last_blade_i
    print(f"1. Default C2 case, n_passage 101, n_eta {N_ETA}: n_m = {len(g.m)}, blade stations {f}..{l}, "
          f"dm* = {g.dm:.6g}; rho = 1, mdot' = 1, normalized mode")
    print("   field      min       at (i, j)  region   m*        max       at (i, j)  region   m*")
    for name in "ABCDE":
        F = getattr(co, name)
        (i0, j0), (i1, j1) = np.unravel_index(np.argmin(F), F.shape), np.unravel_index(np.argmax(F), F.shape)
        print(f"   {name}  {F.min():+11.4e} ({i0:4d},{j0:3d})  {region_of(g, i0):6s} {g.m[i0]:+8.4f}  "
              f"{F.max():+11.4e} ({i1:4d},{j1:3d})  {region_of(g, i1):6s} {g.m[i1]:+8.4f}")
    print(f"   finite everywhere: {all(np.all(np.isfinite(getattr(co, n))) for n in 'ABCDE')}")
    print("\n2. First and last five blade stations (j = 0 lower wall, j = -1 upper wall)")
    print("    i   wall    S*        beta deg   cos(beta)  rho   A          B          C          D           E"
          "          A S* rho cos^2 - 1")
    for i in list(range(f, f + 5)) + list(range(l - 4, l + 1)):
        for j, wall in ((0, "lower"), (-1, "upper")):
            check = co.A[i, j] * metrics.S[i] * rho[i, j] * metrics.cos_beta[i, j] ** 2 - 1.0
            print(f"   {i:4d} {wall:6s} {metrics.S[i]:.6f}  {math.degrees(metrics.beta[i, j]):+9.4f}  "
                  f"{metrics.cos_beta[i, j]:.6f}  {rho[i, j]:.2f}  {co.A[i, j]:10.4f} {co.B[i, j]:+10.4f} "
                  f"{co.C[i, j]:10.6f} {co.D[i, j]:+11.4f} {co.E[i, j]:+11.4e} {check:+.1e}")
        if i == f + 4:
            print("   ...")


def section_3() -> None:
    print("\n3. Corner stations: D and E at the region transitions (rho = 1, mdot' = 1)")
    print("   n_pass  dm*      |D| inlet  |D| f-1..f+1     |D| l..l+1      |D| blade>=0.05c  |E| f-1..f+1  |E| l..l+1"
          "  |E| inlet/exit")
    for n in (31, 51, 101, 201, 401):
        metrics = metrics_at(n)
        g = metrics.grid
        f, l = g.first_blade_i, g.last_blade_i
        co = build_coefficients(metrics, np.ones((len(g.m), len(g.eta))), 1.0)
        inner = (g.m >= g.m[f] + 0.05) & (g.m <= g.m[l] - 0.05)
        print(f"   {n:5d}  {g.dm:.5f}  {np.abs(co.D[:f - 1]).max():9.3g}  {np.abs(co.D[f - 1:f + 2]).max():12.4g}   "
              f"{np.abs(co.D[l:l + 2]).max():12.4g}   {np.abs(co.D[inner]).max():12.4g}   "
              f"{np.abs(co.E[f - 1:f + 2]).max():11.4g}  {np.abs(co.E[l:l + 2]).max():10.4g}  "
              f"{max(np.abs(co.E[:f - 1]).max(), np.abs(co.E[l + 2:]).max()):9.2g}")
    metrics = metrics_at(101)
    g = metrics.grid
    f, l = g.first_blade_i, g.last_blade_i
    co = build_coefficients(metrics, np.ones((len(g.m), len(g.eta))), 1.0)
    B, C, dm = co.B, co.C, g.dm
    B_m = np.empty_like(B)
    B_m[1:-1] = (B[2:] - B[:-2]) / (2.0 * dm)
    print("   n_passage 101, j = 0 / mid / n_eta-1:")
    for name, a, b in (("LE", f - 2, f + 2), ("TE", l - 2, l + 2)):
        for j in (0, N_ETA // 2, N_ETA - 1):
            total = B_m[a:b + 1, j].sum() * dm
            change = 0.5 * (B[b, j] + B[b + 1, j]) - 0.5 * (B[a - 1, j] + B[a, j])
            print(f"   {name} j={j:2d}: tan(beta) {B[a - 1, j]:+.4f} {B[a, j]:+.4f} | {B[a + 1, j]:+.4f} {B[a + 2, j]:+.4f} "
                  f"{B[a + 3, j]:+.4f} | {B[b, j]:+.4f} {B[b + 1, j]:+.4f};  sum(B_m dm) over {a}..{b} = {total:+.6f}, "
                  f"B change (ends averaged) = {change:+.6f}")
        print(f"   {name}: D at i = {a}..{b}, j = 0: {np.array2string(co.D[a:b + 1, 0], precision=4)}")
        print(f"   {name}: E at i = {a}..{b}, j = 0: {np.array2string(co.E[a:b + 1, 0], precision=4)}")
    print(f"   S* at f-1, f, f+1: {metrics.S[f - 1]:.6f} {metrics.S[f]:.6f} {metrics.S[f + 1]:.6f};  "
          f"l, l+1: {metrics.S[l]:.6f} {metrics.S[l + 1]:.6f}")


def section_4() -> None:
    print("\n4. D and E against the hand-expanded reference (exact C2 metric derivatives, analytic rho), mdot' = 1;"
          "\n   n_eta refined with n_passage. Windows in m: 'interior' >= 0.05 c from the first/last blade stations;"
          "\n   'rear' the smooth rear 35 % of the blade's m extent, up to 0.05 c before the last station")
    levels = ((51, 16), (101, 31), (201, 61), (401, 121), (801, 241))
    for label, variable in (("rho = 1", False), ("analytic rho(m, eta)", True)):
        rows = []
        for n, n_eta in levels:
            metrics = metrics_at(n, n_eta)
            g = metrics.grid
            f, l = g.first_blade_i, g.last_blade_i
            if variable:
                rho, rho_m, rho_eta = density(g.m, g.eta, g.m_in, g.m_out)
            else:
                rho = np.ones((len(g.m), len(g.eta)))
                rho_m = rho_eta = np.zeros_like(rho)
            co = build_coefficients(metrics, rho, 1.0)
            D_ref, E_ref = expanded_reference(metrics, rho, rho_m, rho_eta, 1.0, 1.0)
            row = [n, n_eta, g.dm]
            for lo in (g.m[f] + 0.05, g.m[f] + 0.65 * (g.m[l] - g.m[f])):
                window = (g.m >= lo) & (g.m <= g.m[l] - 0.05)
                eD, eE = np.abs(co.D - D_ref)[window], np.abs(co.E - E_ref)[window]
                row += [eD.max(), np.sqrt(np.mean(eD ** 2)), eE.max(), np.sqrt(np.mean(eE ** 2))]
                if lo == g.m[f] + 0.05:
                    worst = np.unravel_index(int(np.argmax(np.where(window[:, None], np.abs(co.D - D_ref), 0.0))),
                                             co.D.shape)
                    row += [g.m[worst[0]] - g.m_le, worst[1], np.abs(D_ref[window]).max(), np.abs(E_ref[window]).max()]
            rows.append(row)
        print(f"   {label}:")
        print("     n_pass n_eta  dm*      | interior: max|dD|  rms|dD|   max|dE|   rms|dE|  (worst D at m*-m_le, j; "
              "max|D_ref|, max|E_ref|) | rear: max|dD|  rms|dD|   max|dE|   rms|dE|")
        for r in rows:
            print(f"     {r[0]:5d} {r[1]:5d}  {r[2]:.5f}  |  {r[3]:.2e}  {r[4]:.2e}  {r[5]:.2e}  {r[6]:.2e}  "
                  f"({r[7]:.3f}, {r[8]:3d}; {r[9]:.3g}, {r[10]:.3g})  |  {r[11]:.2e}  {r[12]:.2e}  {r[13]:.2e}  {r[14]:.2e}")
        h = [r[2] for r in rows]
        for name, k in (("interior max|dD|", 3), ("interior rms|dD|", 4), ("interior max|dE|", 5),
                        ("interior rms|dE|", 6), ("rear max|dD|", 11), ("rear rms|dD|", 12), ("rear max|dE|", 13),
                        ("rear rms|dE|", 14)):
            print(f"       orders {name:17s}", np.round(orders([r[k] for r in rows], h), 2))


def section_5() -> None:
    metrics = metrics_at(101)
    g = metrics.grid
    rho = np.full((len(g.m), len(g.eta)), 1.2)
    c, lam = 0.1, 3.0
    w_m = 100.0 * math.cos(INLET)
    mdot = 1.2 * w_m * g.pitch * c                       # rho_in W_m,in pitch, pitch = pitch* c
    base = build_coefficients(metrics, rho, mdot, c_ref=c)
    fixed = build_coefficients(metrics, rho, mdot, c_ref=lam * c)
    twin = build_coefficients(metrics, rho, lam * mdot, c_ref=lam * c)
    print(f"\n5. Scaling, physical mode: c_ref = {c} m, rho = 1.2 kg/m^3, W_in = 100 m/s at 45 deg, "
          f"mdot' = {mdot:.6g} kg/(m s); lambda = {lam}")
    print("   field  fixed mdot': exponent (expected)    twins: exponent (expected)")
    for name, e_fixed, e_twin in (("A", -1, 0), ("B", 0, 1), ("C", 1, 2), ("D", -1, 0), ("E", 0, 1)):
        F0, F1, F2 = getattr(base, name), getattr(fixed, name), getattr(twin, name)
        big = np.abs(F0) > 1e-6 * np.abs(F0).max()
        p1 = np.log(np.abs(F1[big] / F0[big])) / math.log(lam)
        p2 = np.log(np.abs(F2[big] / F0[big])) / math.log(lam)
        print(f"   {name}      {np.median(p1):+.12f} ({e_fixed:+d}), spread {np.ptp(p1):.1e}     "
              f"{np.median(p2):+.12f} ({e_twin:+d}), spread {np.ptp(p2):.1e}")
    print(f"   ranges, physical: A {base.A.min():.4g}..{base.A.max():.4g} m/s, B {base.B.min():.4g}..{base.B.max():.4g} "
          f"m^2/s, C {base.C.min():.4g}..{base.C.max():.4g} m^3/s, D {base.D.min():.4g}..{base.D.max():.4g} m/s, "
          f"E {base.E.min():.4g}..{base.E.max():.4g} m^2/s")


if __name__ == "__main__":
    section_1_and_2()
    section_3()
    section_4()
    section_5()
