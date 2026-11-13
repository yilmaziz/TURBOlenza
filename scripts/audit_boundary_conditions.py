"""
Audit of boundary_conditions.py on the default C2 cascade, the 65-(12A10)10 at 33 deg
stagger, solidity 1, inlet 45 deg: the numbers of the 2026-11-12 boundary-condition report.
Print only.

    python scripts/audit_boundary_conditions.py

1. n_passage 101, n_eta 31: wall nodes, blade and periodic stations, relations by kind.
2. The upstream and downstream relations and their eliminations.
3. The stations around the region transitions, first_blade_i - 1 .. + 1 and last_blade_i
   - 1 .. + 1: classification, against the nodes (S = y1 - y0 against the pitch).
4. The four outer corners: the relations each takes part in, and the far-station
   dependency on random values.
5. The open item of the assembly step: the periodic-line rows of the last inlet and the
   first exit station reach diagonally into the first / last blade station, where no
   periodic continuation exists; the blade's cut there.
6. The same counts and checks over n_passage 31 to 401.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from turbolenza.b2b import boundary_conditions as bc  # noqa: E402
from turbolenza.b2b.config import GridSettings  # noqa: E402
from turbolenza.b2b.geometry import CascadeSpec, make_cascade  # noqa: E402
from turbolenza.b2b.grid import make_grid  # noqa: E402

SPEC = CascadeSpec(blade_type="NACA65", solidity=1.0, stagger_deg=33.0, cl0=1.2, t_max_over_c=0.10,
                   surface_continuity="C2")
INLET = math.radians(45.0)
CASCADE = make_cascade(SPEC)


def grid_and_bcs(n_passage: int, n_eta: int):
    g = make_grid(CASCADE, GridSettings(n_passage=n_passage, n_eta=n_eta), INLET)
    return g, bc.build_boundary_conditions(g)


def describe(relation: bc.Relation) -> str:
    terms = " ".join(f"{t.coefficient:+g} psi{t.node}" for t in relation.terms)
    return f"{relation.kind:10s} {terms} = {relation.rhs:g}" + ("" if relation.target is None else f"   target {relation.target}")


def node_class(b: bc.BoundaryConditions, node) -> str:
    names = [name for name in ("lower_wall", "upper_wall", "periodic_lower", "periodic_upper", "upstream", "downstream")
             if getattr(b, name)[node]]
    return "+".join(names) if names else "interior"


def main() -> None:
    g, b = grid_and_bcs(101, 31)
    f, l = g.first_blade_i, g.last_blade_i
    N, n = len(g.m) - 1, len(g.eta) - 1
    print(f"1. n_passage 101, n_eta 31: {len(g.m)} x {len(g.eta)} nodes; blade stations {f}..{l}")
    print(f"   lower wall nodes {int(b.lower_wall.sum())}, upper wall nodes {int(b.upper_wall.sum())}, blade stations "
          f"{len(b.blade_stations)}, periodic stations {len(b.periodic_stations)} ({f} inlet, {N - l} exit)")
    print(f"   relations: " + ", ".join(f"{k} {len(b.relations_of(k))}" for k in bc.KINDS) + f"; interior nodes "
          f"{int(b.interior.sum())}, periodic-line nodes off the far stations {int(b.periodic_line.sum())}")
    print(f"   nodes in two classes: {int((b.blade_wall & (b.periodic_side | b.meridional_boundary)).sum())} "
          f"(wall and periodic/far); corners (far and periodic): {len(b.corners)}")

    print("\n2. Far relations (j = 5) and their eliminations")
    for relation, elimination in ((b.relations_of(bc.UPSTREAM)[5], b.upstream_elimination(5)),
                                  (b.relations_of(bc.DOWNSTREAM)[5], b.downstream_elimination(5))):
        print(f"   {describe(relation)}")
        print(f"      psi{elimination.node} = " + " ".join(f"{t.coefficient:+.17g} psi{t.node}" for t in elimination.terms)
              + f" {elimination.constant:+g}")

    print("\n3. Transition stations (S = y1 - y0 from the nodes; blade stations have S < pitch)")
    S = g.nodes[:, -1, 1] - g.nodes[:, 0, 1]
    for i in (f - 1, f, f + 1, l - 1, l, l + 1):
        label = f"first_blade_i{i - f:+d}" if abs(i - f) <= 1 else f"last_blade_i{i - l:+d}"
        print(f"   i = {i:4d} ({label:16s}) m* = {g.m[i]:+.6f}  pitch - S = {g.pitch - S[i]:.3e}   j = 0: "
              f"{node_class(b, (i, 0)):15s} j = n_eta - 1: {node_class(b, (i, n))}")

    print("\n4. Outer corners")
    for corner in b.corners:
        print(f"   {corner}: {node_class(b, corner)}")
        for relation in b.relations_at(corner):
            print(f"      {describe(relation)}")
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(20):
        psi = rng.normal(size=b.shape)
        P = {i: b.periodic_relation(i).residual(psi) for i in (0, 1, 2, N - 2, N - 1, N)}
        U = [r.residual(psi) for r in b.relations_of(bc.UPSTREAM)]
        V = [r.residual(psi) for r in b.relations_of(bc.DOWNSTREAM)]
        worst = max(worst, abs((U[n] - U[0]) - (-3 * P[0] + 4 * P[1] - P[2])),
                    abs((V[n] - V[0]) - (3 * P[N] - 4 * P[N - 1] + P[N - 2])))
    print(f"   U(0, n) - U(0, 0) = -3 P(0) + 4 P(1) - P(2) and the downstream analogue: largest gap {worst:.2e} on 20 random psi")

    print("\n5. Open item for the assembly: the periodic-line rows beside the transitions")
    for station, blade_i, row_nodes in ((f - 1, f, ((f - 1, 0), (f - 1, n))), (l + 1, l, ((l + 1, 0), (l + 1, n)))):
        need = [(blade_i, -1), (blade_i, n + 1)]
        print(f"   periodic station {station}: its row at {row_nodes[0]} needs {need[0]}, at {row_nodes[1]} needs {need[1]} "
              f"(diagonal neighbors at blade station {blade_i}); periodic continuation there: ", end="")
        try:
            b.periodic_ghost(blade_i, -1)
            print("available")
        except bc.BoundaryConditionError:
            print("none (refused)")
        print(f"      blade cut at station {blade_i}: pitch - S = {g.pitch - S[blade_i]:.4e} chord, against deta S = "
              f"{g.deta * S[blade_i]:.4e}: a continuation 'as if periodic' would be off by that cut")

    print("\n6. Over resolutions")
    print("   n_pass n_eta  n_m   blade  periodic  inlet exit  walls  relations  nodes-vs-S  interior")
    for n_passage, n_eta in ((31, 5), (51, 11), (101, 31), (201, 61), (401, 121)):
        g, b = grid_and_bcs(n_passage, n_eta)
        S = g.nodes[:, -1, 1] - g.nodes[:, 0, 1]
        agree = np.array_equal(b.blade_stations, np.flatnonzero(S < g.pitch - 1e-12 * g.chord))
        print(f"   {n_passage:5d} {n_eta:5d}  {len(g.m):5d}  {len(b.blade_stations):5d}  {len(b.periodic_stations):8d}  "
              f"{g.first_blade_i:5d} {len(g.m) - 1 - g.last_blade_i:4d}  {int(b.blade_wall.sum()):5d}  {len(b.relations):9d}  "
              f"{str(agree):10s}  {int(b.interior.sum())}")


if __name__ == "__main__":
    main()
