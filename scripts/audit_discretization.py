"""
Audit of discretization.py on the default C2 cascade, the 65-(12A10)10 at 33 deg stagger,
solidity 1, inlet 45 deg, rho = 1, mdot' = 1, normalized mode: the numbers of the
2026-11-12 discretization report. Print only.

    python scripts/audit_discretization.py

1. Ranges of the normalized weights of Eq. 5-44 and of H at n_passage 101 (n_eta 31), with
   their nodes; row properties; where neighbor weights turn positive.
2. The corner stations, first_blade_i - 2 .. + 2 and last_blade_i - 2 .. + 2: weight
   ranges over eta, D there, and |D| deta / (2 A), the ratio past which a south or north
   weight turns positive.
3. The curvature steps of the C2 walls (front 60 % of the chord): weights finite,
   deterministic, signs as their formulas give them; station-to-station change of the
   weights on the front against the smooth rear.
4. Expanded against conservative: for a smooth psi(m*, eta), Eq. 5-44 times -H (the expanded
   form, Eqs. 5-35 to 5-39 on Eq. 5-21) against a conservative face-flux discretization of
   Eq. 5-20 written here, on the real grid at n_passage 51 to 401 (n_eta refined alongside):
   the smooth rear, the front and the corner stations.
5. The periodic PDE line (build_periodic_pde_rows): the LE and TE transition rows and their
   standard neighbors with every node, coefficient, rhs and H; the mixed constants cancel
   and the eta jump stays on the rhs; ranges over the transition rows at n_passage 101 and
   401; and the straight links to the real-node neighbors of the rows around the
   transitions, checked against the blade's metal (the reference blade and its pitch copies,
   the C2 curves sampled densely): which cross it.
6. The TE row revision and the link audit (scripts/link_geometry.py: exact crossings with the
   C2 curves, tolerance 1e-7 chord, an end touching a wall not counted): the old central TE
   chord against the revised footprint at n_passage 101 and 401 (n_eta 41); every straight-chord
   blade intersection of the PDE rows at l-3 .. l+4 and f-4 .. f+3 over n_passage 31 to 401 x
   n_eta 21 to 61, and at n_eta 121 and 241 for n_passage 201 and 401; how many rows they touch.
   (A geometric fact about chords, not by itself stencil invalidity: see
   scripts/audit_transition_mapping.py for the computational mapping.)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from turbolenza.b2b import discretization as dz  # noqa: E402
from turbolenza.b2b.boundary_conditions import build_boundary_conditions  # noqa: E402
from turbolenza.b2b.coefficients import build_coefficients  # noqa: E402
from turbolenza.b2b.config import GridSettings  # noqa: E402
from turbolenza.b2b.discretization import build_interior_stencil  # noqa: E402
from turbolenza.b2b.geometry import CascadeSpec, make_cascade  # noqa: E402
from turbolenza.b2b.grid import make_grid  # noqa: E402
from turbolenza.b2b.metrics import build_metrics  # noqa: E402

SPEC = CascadeSpec(blade_type="NACA65", solidity=1.0, stagger_deg=33.0, cl0=1.2, t_max_over_c=0.10,
                   surface_continuity="C2")
INLET = math.radians(45.0)
CASCADE = make_cascade(SPEC)
NAMES = ("west", "east", "south", "north", "mixed")


def case(n_passage: int, n_eta: int):
    metrics = build_metrics(make_grid(CASCADE, GridSettings(n_passage=n_passage, n_eta=n_eta), INLET))
    co = build_coefficients(metrics, np.ones((len(metrics.grid.m), len(metrics.grid.eta))), 1.0)
    return metrics, co, build_interior_stencil(co, metrics)


def region(g, i: int) -> str:
    f, l = g.first_blade_i, g.last_blade_i
    return "inlet" if i < f else ("exit" if i > l else "blade")


def section_1() -> None:
    metrics, co, st = case(101, 31)
    g = metrics.grid
    f, l = g.first_blade_i, g.last_blade_i
    I = st.interior
    print(f"1. n_passage 101, n_eta 31: {len(g.m)} x {len(g.eta)} nodes, {int(I.sum())} interior; blade stations "
          f"{f}..{l}; dm = {st.dm:.6g}, deta = {st.deta:.6g}")
    print("   weight   min        at (i, j)  i-f   region   max        at (i, j)  i-f   region")
    for name in NAMES + ("H",):
        F = getattr(st, name)
        (i0, j0), (i1, j1) = np.unravel_index(int(np.nanargmin(F)), F.shape), np.unravel_index(int(np.nanargmax(F)), F.shape)
        print(f"   {name:6s} {np.nanmin(F):+.4e} ({i0:3d},{j0:2d}) {i0 - f:+4d}  {region(g, i0):6s}  {np.nanmax(F):+.4e} "
              f"({i1:3d},{j1:2d}) {i1 - f:+4d}  {region(g, i1):6s}")
    finite = sum(int(np.sum(~np.isfinite(getattr(st, n)[I]))) for n in NAMES + ("H", "center", "rhs"))
    print(f"   non-finite values at interior nodes: {finite};  NaN outside the interior: "
          f"{all(np.all(np.isnan(getattr(st, n)[~I])) for n in NAMES + ('H', 'center', 'rhs'))}")
    print(f"   min H = {np.nanmin(st.H):.6g} > 0: {bool(np.all(st.H[I] > 0))}")
    total = st.west + st.east + st.south + st.north
    neighbors = np.abs(st.west) + np.abs(st.east) + np.abs(st.south) + np.abs(st.north) + 4.0 * np.abs(st.mixed)
    print(f"   west + east + south + north + 1: max |.| = {np.nanmax(np.abs(total + 1.0)):.2e};  sum of |neighbor "
          f"weights| {np.nanmin(neighbors):.4f} .. {np.nanmax(neighbors):.4f} (median {np.nanmedian(neighbors[I]):.4f})")
    print(f"   max |mixed| = {np.nanmax(np.abs(st.mixed)):.5f};  max(|mixed| - |sin beta| / 8) = "
          f"{np.nanmax(np.abs(st.mixed) - np.abs(metrics.sin_beta) / 8.0):.2e}")
    for name in ("west", "east", "south", "north"):
        where = np.argwhere(I & (getattr(st, name) > 0.0))
        stations = sorted({int(i) - f for i, _ in where})
        print(f"   {name:5s} > 0 at {len(where):3d} nodes, stations i - f = {stations}")


def section_2() -> None:
    print("\n2. Corner stations (rho = 1): weights over j = 1 .. n_eta - 2, D and |D| deta / (2 A)")
    for n_passage, n_eta in ((101, 31), (401, 121)):
        metrics, co, st = case(n_passage, n_eta)
        g = metrics.grid
        f, l = g.first_blade_i, g.last_blade_i
        print(f"   n_passage {n_passage}, n_eta {n_eta}:")
        print("     i     rel     west             east             south            north            mixed"
              "            |D| max     |D|deta/2A max  finite")
        for i in list(range(f - 2, f + 3)) + list(range(l - 2, l + 3)):
            label = f"f{i - f:+d}" if abs(i - f) <= 2 else f"l{i - l:+d}"
            row = "  ".join(f"{np.min(getattr(st, n)[i, 1:-1]):+.3f}..{np.max(getattr(st, n)[i, 1:-1]):+.3f}" for n in NAMES)
            ratio = np.abs(co.D[i, 1:-1]) * st.deta / (2.0 * co.A[i, 1:-1])
            ok = all(np.all(np.isfinite(getattr(st, n)[i, 1:-1])) for n in NAMES + ("H",))
            print(f"     {i:4d}  {label:5s}  {row}  {np.abs(co.D[i, 1:-1]).max():9.3g}  {ratio.max():12.3f}  {ok}")


def section_3() -> None:
    print("\n3. Curvature steps (front 60 % of the chord) against the smooth rear, n_passage 101 and 401")
    for n_passage, n_eta in ((101, 31), (401, 121)):
        metrics, co, st = case(n_passage, n_eta)
        _, co2, st2 = case(n_passage, n_eta)
        g = metrics.grid
        f, l = g.first_blade_i, g.last_blade_i
        I = st.interior
        same = all(np.array_equal(getattr(st, n), getattr(st2, n), equal_nan=True) for n in NAMES + ("H",))
        signs = (np.array_equal(np.sign(st.mixed[I]), np.sign(co.B[I])) and np.all(st.west[I] < 0) and np.all(st.east[I] < 0)
                 and np.array_equal(st.south[I] > 0, (co.D * st.deta > 2.0 * co.A)[I])
                 and np.array_equal(st.north[I] > 0, (-co.D * st.deta > 2.0 * co.A)[I]))
        first, last = g.m[f], g.m[l]
        front = (g.m >= first + 0.05) & (g.m <= first + 0.6 * math.cos(math.radians(33.0)))
        rear = (g.m >= first + 0.65 * (last - first)) & (g.m <= last - 0.05)
        jumps = {}
        for name in ("south", "north", "west", "east", "mixed"):
            d = np.abs(np.diff(getattr(st, name)[:, 1:-1], axis=0))
            jumps[name] = (np.nanmax(d[front[1:] & front[:-1]]), np.nanmax(d[rear[1:] & rear[:-1]]))
        print(f"   n_passage {n_passage}: rebuilt bit for bit: {same}; signs follow the formulas: {bool(signs)}; "
              f"finite: {all(np.all(np.isfinite(getattr(st, n)[I])) for n in NAMES)}")
        print("     largest station-to-station change, front / rear: "
              + ", ".join(f"{k} {v[0]:.2e} / {v[1]:.2e}" for k, v in jumps.items()))


def conservative(A, B, C, psi, dm, deta):
    Af, Bf = 0.5 * (A[1:-1, 1:] + A[1:-1, :-1]), 0.5 * (B[1:-1, 1:] + B[1:-1, :-1])
    F = Af * (psi[1:-1, 1:] - psi[1:-1, :-1]) / deta - Bf * (psi[2:, 1:] + psi[2:, :-1] - psi[:-2, 1:] - psi[:-2, :-1]) / (4.0 * dm)
    Bg, Cg = 0.5 * (B[1:, 1:-1] + B[:-1, 1:-1]), 0.5 * (C[1:, 1:-1] + C[:-1, 1:-1])
    G = Bg * (psi[1:, 2:] + psi[:-1, 2:] - psi[1:, :-2] - psi[:-1, :-2]) / (4.0 * deta) - Cg * (psi[1:, 1:-1] - psi[:-1, 1:-1]) / dm
    return (F[:, 1:] - F[:, :-1]) / deta - (G[1:, :] - G[:-1, :]) / dm


def expanded(st, psi):
    c = (slice(1, -1), slice(1, -1))
    r = (psi[1:-1, 1:-1] + st.west[c] * psi[:-2, 1:-1] + st.east[c] * psi[2:, 1:-1] + st.south[c] * psi[1:-1, :-2]
         + st.north[c] * psi[1:-1, 2:] + st.mixed[c] * (psi[2:, 2:] - psi[2:, :-2] - psi[:-2, 2:] + psi[:-2, :-2]))
    return -r * st.H[c]


def section_4() -> None:
    print("\n4. Expanded (Eq. 5-44 x -H) against conservative face-flux Eq. 5-20, psi = eta + 0.05 sin(2 m*) sin(2 pi eta),"
          " real grid, rho = 1")
    print("   n_pass n_eta   rear: max gap   rel        front: max gap  rel        corners: max gap  rel")
    rows = []
    for n_passage, n_eta in ((51, 16), (101, 31), (201, 61), (401, 121)):
        metrics, co, st = case(n_passage, n_eta)
        g = metrics.grid
        f, l = g.first_blade_i, g.last_blade_i
        M, H = np.meshgrid(g.m, g.eta, indexing="ij")
        psi = H + 0.05 * np.sin(2.0 * M) * np.sin(2.0 * math.pi * H)
        e = expanded(st, psi)
        c = conservative(co.A, co.B, co.C, psi, st.dm, st.deta)
        gap = np.abs(e - c)
        m = g.m[1:-1]
        first, last = g.m[f], g.m[l]
        windows = {"rear": (m >= first + 0.65 * (last - first)) & (m <= last - 0.05),
                   "front": (m >= first + 0.05) & (m <= first + 0.6 * math.cos(math.radians(33.0))),
                   "corners": np.isin(np.arange(1, len(g.m) - 1), [f - 1, f, f + 1, l, l + 1])}
        row = [n_passage, n_eta, g.dm]
        for name in ("rear", "front", "corners"):
            w = windows[name]
            row += [gap[w].max(), gap[w].max() / np.abs(e[w]).max()]
        rows.append(row)
        print(f"   {n_passage:5d} {n_eta:5d}   {row[3]:.3e}  {row[4]:.2e}    {row[5]:.3e}  {row[6]:.2e}    "
              f"{row[7]:.3e}  {row[8]:.2e}")
    h = [r[2] for r in rows]
    for name, k in (("rear", 3), ("front", 5), ("corners", 7)):
        orders = [math.log(rows[q][k] / rows[q + 1][k]) / math.log(h[q] / h[q + 1]) for q in range(len(rows) - 1)]
        print(f"   orders of the {name} gap: {np.round(orders, 2)}")


def describe_row(row, f: int, l: int) -> None:
    i = row.target[0]
    where = f"f{i - f:+d}" if i < f else f"l{i - l:+d}"
    print(f"   {row.kind:24s} target {row.target} ({where}): rhs = {row.rhs:+.6e}, H = {row.H:.6e}")
    print("      " + ", ".join(f"{t.node}: {t.coefficient:+.5f}" for t in row.terms))


def blade_outline() -> np.ndarray:
    curves = CASCADE.active_curves
    u = np.linspace(0.0, 1.0, 10001)
    return np.vstack([curves.suction.derivative(u[::-1], 0), curves.pressure.derivative(u[1:], 0)])


def metal_on(a: np.ndarray, b: np.ndarray, paths, tree) -> float:
    """Length of the straight segment a -> b inside any of the blade copies (2001 samples);
    skipped (0) when neither end is within the segment's length of the outline."""
    length = float(np.linalg.norm(b - a))
    if min(tree.query(a)[0], tree.query(b)[0]) > length + 1e-3:
        return 0.0
    t = np.linspace(0.0, 1.0, 2001)[1:-1]
    points = a[None, :] + t[:, None] * (b - a)[None, :]
    inside = np.zeros(len(t), dtype=bool)
    for path in paths:
        inside |= path.contains_points(points)
    return float(inside.mean() * np.linalg.norm(b - a))


def section_5() -> None:
    from matplotlib.path import Path as MplPath          # the audit only; the package never imports matplotlib
    from scipy.spatial import cKDTree

    print("\n5. The periodic PDE line: rows at (i, n_eta - 1) on the periodic stations, rho = 1")
    outline = blade_outline()
    for n_passage, n_eta in ((101, 31), (401, 121)):
        metrics, co, _ = case(n_passage, n_eta)
        g = metrics.grid
        f, l, J = g.first_blade_i, g.last_blade_i, len(g.eta) - 1
        bcs = build_boundary_conditions(g)
        rows = dz.build_periodic_pde_rows(co, metrics, bcs)
        by = {row.target[0]: row for row in rows}
        kinds = {k: sum(row.kind == k for row in rows) for k in dz.PERIODIC_ROW_KINDS}
        bad = sum(not (all(np.isfinite(t.coefficient) for t in row.terms) and np.isfinite(row.rhs)) for row in rows)
        print(f"   n_passage {n_passage}, n_eta {n_eta}: f = {f}, l = {l}, J = {J}; rows {len(rows)} {kinds}; "
              f"non-finite rows {bad}; rebuilt bit for bit: {rows == dz.build_periodic_pde_rows(co, metrics, bcs)}")
        if n_passage == 101:
            for i in (f - 2, f - 1, l + 1, l + 2):
                describe_row(by[i], f, l)
            le, te = by[f - 1], by[l + 1]
            print(f"   the LE row's nodes at stations >= f: {[n for n in le.nodes if n[0] >= f]};  the TE row's at "
                  f"stations <= l: {[n for n in te.nodes if n[0] <= l]}")
            no_B = {r.target[0]: r for r in dz._periodic_pde_rows(co.A, 0.0 * co.B, co.C, co.D, co.E, g.dm, g.deta, f, l,
                                                                   lambda k: bcs.periodic_ghost(k, n_eta))}
            for row in (by[f - 2], le, te, by[l + 2]):
                i = row.target[0]
                north = -(co.A[i, J] / g.deta ** 2 + co.D[i, J] / (2.0 * g.deta)) / row.H
                print(f"   {row.kind:24s} at {i}: rhs {row.rhs:+.6e} = -north {-north:+.6e} (eta jump kept); rhs "
                      f"without B {no_B[i].rhs:+.6e} (mixed constants cancel: difference {row.rhs - no_B[i].rhs:+.1e})")
        for name, row in (("LE", by[f - 1]), ("TE", by[l + 1])):
            values = [t.coefficient for t in row.terms if t.node != row.target]
            print(f"   {name} row: coefficients {min(values):+.4f} .. {max(values):+.4f}, rhs {row.rhs:+.6f}, H {row.H:.6g}")
        copies = [outline + np.array([0.0, k * g.pitch]) for k in (-1, 0, 1)]
        paths, tree = [MplPath(c) for c in copies], cKDTree(np.vstack(copies))
        crossings = []
        for i in range(f - 2, f + 2):
            crossings.append(("interior", i))
        for i in range(l - 1, l + 3):
            crossings.append(("interior", i))
        report = []
        for _, i in crossings:
            for j in range(1, J):
                for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
                    metal = metal_on(g.nodes[i, j], g.nodes[i + di, j + dj], paths, tree)
                    if metal > 0.0:
                        report.append((i, j, di, dj, metal))
        for row in (by[f - 1], by[l + 1]):
            i = row.target[0]
            for node in ((i - 1, J), (i + 1, J), (i, J - 1)):
                metal = metal_on(g.nodes[i, J], g.nodes[node], paths, tree)
                if metal > 0.0:
                    report.append((i, J, node[0] - i, node[1] - J, metal))
        print(f"   links crossing metal among the rows at stations f-2..f+1 and l-1..l+2 (interior) and the LE/TE "
              f"periodic rows ({len(report)}):")
        for i, j, di, dj, metal in report:
            label = f"f{i - f:+d}" if abs(i - f) <= 2 else f"l{i - l:+d}"
            source = "periodic row" if j == J else "interior row"
            print(f"      {source} ({i}, {j}) [{label}] -> ({i + di}, {j + dj}): {metal:.3e} c of metal")


def section_6() -> None:
    import link_geometry as lg

    blade = lg.Blade(CASCADE)
    print("\n6. The TE periodic row: old central psi_m, psi_mm against the revised forward ones (n_eta 41)")
    for n_passage in (101, 401):
        g, _, _, rows = lg.build_case(CASCADE, n_passage, 41)
        l, J = g.last_blade_i, 40
        row = rows[l + 1]
        old = blade.crossing(g.nodes[l + 1, J], g.nodes[l, J])
        print(f"   n_passage {n_passage}: old link ({l + 1}, J) -> ({l}, J) {old.inside_length:.3e} of {old.length:.3e} c "
              f"inside ({100 * old.fraction:.1f} %, blade copy {old.copies})")
        bad = 0
        for t in row.terms:
            if t.node != row.target:
                c = blade.crossing(g.nodes[row.target], lg.physical(g, t.node))
                bad += not c.clear
        print(f"      revised row {row.target}: nodes {sorted(row.nodes)}; stations {sorted({n[0] for n in row.nodes})}; "
              f"chords intersecting the blade {bad}; center {row.coefficient(row.target):+.4f}, rhs {row.rhs:+.6f}, H {row.H:.6g}, "
              f"coefficients {min(t.coefficient for t in row.terms):+.4f} .. {max(t.coefficient for t in row.terms):+.4f}")
    levels = [(n, e) for n in (31, 51, 61, 101, 201, 401) for e in (21, 31, 41, 61)]
    levels += [(201, 121), (401, 121), (201, 241), (401, 241)]
    results = lg.sweep(CASCADE, blade, levels)
    print("   sweep: straight-chord blade intersections (rows at l-3 .. l+4 and f-4 .. f+3), after the TE revision")
    total_rows = set()
    for r in results:
        rows_hit = {x["target"] for x in r["straight_chord_intersections"]}
        total_rows |= {(r["n_passage"], r["n_eta"], t) for t in rows_hit}
        listing = "; ".join(f"{x['side']} {x['row'][:8]} {x['target']} [{x['station']}] {x['direction']} -> {x['node']} "
                            f"{100 * x['crossing'].fraction:.0f} % ({x['use']})" for x in r["straight_chord_intersections"])
        print(f"   n_passage {r['n_passage']:3d} n_eta {r['n_eta']:3d}: {r['checked']:5d} chords, "
              f"{len(r['straight_chord_intersections'])} intersecting in "
              f"{len(rows_hit)} rows; old central TE link {100 * r['old_te'].fraction:3.0f} %" + (f" | {listing}" if listing else ""))
    periodic = sum(x["row"] != "interior" for r in results for x in r["straight_chord_intersections"])
    print(f"   in all: {sum(len(r['straight_chord_intersections']) for r in results)} straight-chord intersections in "
          f"{len(total_rows)} (grid, row) pairs; periodic "
          f"rows among them: {periodic}")


if __name__ == "__main__":
    section_1()
    section_2()
    section_3()
    section_4()
    section_5()
    section_6()
