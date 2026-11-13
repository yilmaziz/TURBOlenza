"""
Tests for potential_flow_solver.matrix_assembly: the local equations of discretization.py and
boundary_conditions.py gathered into F Psi = Q, with Aungier's far-boundary elimination.

What is checked
- Numbering: full_index / node_of exhaustively on small grids (C order, psi.ravel()), off-grid
  indices refused, never wrapped.
- Hand-checked rows of a 9 x 4 synthetic system (a hand-made InteriorStencil with distinct
  weights, periodic rows from discretization's own builder, boundary_conditions' relations):
  the nine-point row, Dirichlet rows, periodic jump rows, the far-boundary substitution, the
  periodic and LE / TE rows copied unscaled, the +1 of the periodic continuation neither lost
  nor added twice.
- The substitution algebra on explicit LocalEquation systems: an elimination constant, chained
  eliminations in any order, a cycle, duplicate contributions, structural refusals.
- Real admissible grids (13 to 53 deg, C2 65-(12A10)10, rho = 1): counts, the production
  eliminations themselves, local-row equivalence (F x - Q)[r] = the residual of the production
  object of row r on reconstruct(x) for every row class, manufactured known-vector recovery
  with a sparse direct solve (test only), the solved system satisfying every local equation
  and the implied far jumps, bandwidth from the local equations' reach, determinism, input
  refusals.
- The module holds no finite-difference physics and no solve (source guard).

Tolerances: entries copied without arithmetic are compared exactly; entries with one
substitution (4/3, -1/3) to rtol 1e-14 (one rounding, measured 0 to 1 ulp); residual
equivalence to 1e-12 absolute (measured at most 8.9e-16 on residuals of order 1 over eight
admissible grids, exactly 0 on rows without a substitution); recovery to 1e-10 relative
(measured 1.9e-14 to 3.4e-13); the solved system's residuals to 1e-11.
"""

from __future__ import annotations

import ast
import math
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from potential_flow_solver import boundary_conditions as bc
from potential_flow_solver import discretization as dz
from potential_flow_solver import grid as gr
from potential_flow_solver import matrix_assembly as ma
from potential_flow_solver.boundary_conditions import Elimination, Relation, Term, build_boundary_conditions
from potential_flow_solver.coefficients import build_coefficients
from potential_flow_solver.config import GridSettings
from potential_flow_solver.geometry import CascadeSpec, make_cascade
from potential_flow_solver.metrics import build_metrics

INLET = math.radians(45.0)
TWO_PI = 2.0 * math.pi


# --- numbering ---------------------------------------------------------------------------------

def test_full_index_is_the_c_order_exhaustively():
    for n_m in range(1, 7):
        for n_eta in range(1, 7):
            shape = (n_m, n_eta)
            psi = np.arange(n_m * n_eta, dtype=float).reshape(shape) * 1.5 + 0.25
            seen = []
            for i in range(n_m):
                for j in range(n_eta):
                    k = ma.full_index(i, j, shape)
                    assert k == i * n_eta + j == np.ravel_multi_index((i, j), shape)
                    assert ma.node_of(k, shape) == (i, j)
                    assert psi.ravel()[k] == psi[i, j]
                    seen.append(k)
            assert seen == list(range(n_m * n_eta))
            I, J = np.meshgrid(np.arange(n_m), np.arange(n_eta), indexing="ij")
            K = ma.full_index(I, J, shape)
            assert np.array_equal(K, np.arange(n_m * n_eta).reshape(shape))
            back_i, back_j = ma.node_of(K.ravel(), shape)
            assert np.array_equal(back_i, I.ravel()) and np.array_equal(back_j, J.ravel())


def test_nine_point_neighbours_are_the_documented_offsets():
    n_m, n_eta = 5, 6
    for i in range(1, n_m - 1):
        for j in range(1, n_eta - 1):
            k = ma.full_index(i, j, (n_m, n_eta))
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    assert ma.full_index(i + di, j + dj, (n_m, n_eta)) == k + di * n_eta + dj


@pytest.mark.parametrize("node", [(-1, 0), (0, -1), (4, 0), (0, 3), (2.0, 1), (True, 1)])
def test_full_index_refuses_nodes_off_the_grid(node):
    with pytest.raises(ma.MatrixAssemblyError):
        ma.full_index(node[0], node[1], (4, 3))


@pytest.mark.parametrize("k", [-1, 12, 13])
def test_node_of_refuses_indices_off_the_grid(k):
    with pytest.raises(ma.MatrixAssemblyError):
        ma.node_of(k, (4, 3))


# --- an 11 x 4 synthetic system ------------------------------------------------------------------

N_M, N_ETA, F_I, L_I = 11, 4, 4, 5           # blade stations 4, 5: the LE row (3, J) and the TE row (6, J) read no far station


@lru_cache(maxsize=None)
def synthetic():
    m = np.arange(N_M) * 0.1
    eta = np.linspace(0.0, 1.0, N_ETA)
    region = np.full(N_M, gr.PASSAGE)
    region[:F_I], region[L_I + 1:] = gr.INLET, gr.EXIT
    bcs = bc._boundary_conditions(None, m, eta, F_I, L_I, np.arange(F_I, L_I + 1), region)
    interior = np.zeros((N_M, N_ETA), dtype=bool)
    interior[1:-1, 1:-1] = True
    I, J = np.meshgrid(np.arange(N_M), np.arange(N_ETA), indexing="ij")

    def field(base):                          # distinct at every node, NaN off the interior
        return np.where(interior, base + 0.001 * I + 0.0001 * J, np.nan)
    stencil = dz.InteriorStencil(center=np.where(interior, 1.0, np.nan), west=field(-0.21), east=field(-0.23),
                                 south=field(-0.27), north=field(-0.29), mixed=field(0.031), rhs=field(0.5),
                                 H=field(10.0), interior=interior, dm=0.1, deta=1.0 / (N_ETA - 1))
    M, H = np.meshgrid(m, eta, indexing="ij")
    A = 1.5 + 0.2 * np.sin(M) * np.cos(TWO_PI * H + 0.3)
    B = 0.25 + 0.08 * np.cos(0.8 * M) * np.sin(TWO_PI * H + 0.5)
    C = 1.2 + 0.15 * np.cos(1.1 * M) + 0.0 * H
    D = 0.3 * np.sin(M + H)
    E = 0.2 * np.cos(M - H)
    rows = dz._periodic_pde_rows(A, B, C, D, E, 0.1, 1.0 / (N_ETA - 1), F_I, L_I, lambda k: bcs.periodic_ghost(k, N_ETA))
    return bcs, stencil, rows, ma.assemble_system(stencil, rows, bcs)


def entries(result, node):
    """{column node: value} of the row owned by node."""
    shape = result.shape
    r = int(result.full_to_active[ma.full_index(node[0], node[1], shape)])
    row = result.matrix[r]
    return {ma.node_of(int(result.active_nodes[c]), shape): v for c, v in zip(row.indices, row.data)}, result.rhs[r], r


def test_synthetic_counts_and_owners():
    bcs, stencil, rows, res = synthetic()
    J = N_ETA - 1
    assert res.n_unknowns == (N_M - 2) * N_ETA == 36
    assert res.matrix.shape == (36, 36)
    assert {ma.node_of(int(k), res.shape) for k in res.active_nodes} == {(i, j) for i in range(1, N_M - 1)
                                                                         for j in range(N_ETA)}
    assert {e.node for e in res.eliminations} == {(i, j) for i in (0, N_M - 1) for j in range(N_ETA)}
    kinds = {}
    for r, kind in enumerate(res.row_kind):
        kinds.setdefault(kind, []).append(ma.node_of(int(res.active_nodes[r]), res.shape))
    assert len(kinds[ma.INTERIOR]) == (N_M - 2) * (N_ETA - 2)
    assert sorted(kinds[bc.LOWER_WALL]) == [(4, 0), (5, 0)] and sorted(kinds[bc.UPPER_WALL]) == [(4, J), (5, J)]
    assert sorted(kinds[bc.PERIODIC]) == [(i, 0) for i in (1, 2, 3, 6, 7, 8, 9)]      # the constraint side
    assert kinds[dz.PERIODIC_LE_TRANSITION] == [(3, J)] and kinds[dz.PERIODIC_TE_TRANSITION] == [(6, J)]
    assert sorted(kinds[dz.PERIODIC_STANDARD]) == [(i, J) for i in (1, 2, 7, 8, 9)]
    assert sorted(r.nodes for r in res.implied) == sorted(((i, J), (i, 0)) for i in (0, N_M - 1))
    assert all(r.kind == bc.PERIODIC for r in res.implied)


def test_interior_row_is_the_nine_point_equation_at_its_columns():
    bcs, stencil, rows, res = synthetic()
    W = stencil.weights
    for i in range(2, N_M - 2):                       # rows that read no far station
        for j in range(1, N_ETA - 1):
            got, q, r = entries(res, (i, j))
            want = {(i + di, j + dj): W[i, j, 1 + di, 1 + dj] for di in (-1, 0, 1) for dj in (-1, 0, 1)}
            assert got == want                       # exact: copied, not computed
            assert q == stencil.rhs[i, j]
            assert res.row_kind[r] == ma.INTERIOR and not res.substituted[r]
            assert res.matrix[r, r] == 1.0           # the center, not rescaled


def test_dirichlet_rows():
    bcs, stencil, rows, res = synthetic()
    for relation in bcs.relations_of(bc.LOWER_WALL) + bcs.relations_of(bc.UPPER_WALL):
        got, q, r = entries(res, relation.target)
        assert got == {relation.target: 1.0}
        assert q == (0.0 if relation.kind == bc.LOWER_WALL else 1.0)


def test_periodic_jump_rows_have_two_coefficients_and_rhs_one():
    bcs, stencil, rows, res = synthetic()
    J = N_ETA - 1
    for i in (1, 2, 3, 6, 7, 8, 9):
        got, q, r = entries(res, (i, 0))
        assert got == {(i, J): 1.0, (i, 0): -1.0}
        assert q == 1.0
    assert all(abs(x) < 1e-15 for x in ma.implied_residuals(res, np.add.outer(np.zeros(N_M), np.linspace(0, 1, N_ETA))))


def test_far_substitution_is_the_documented_algebra():
    """Row of (1, j): psi[0, j'] = 4/3 psi[1, j'] - 1/3 psi[2, j'], so a coefficient a on (0, j')
    moves 4/3 a to (1, j') and -1/3 a to (2, j'); the rhs keeps q (the constant is 0)."""
    bcs, stencil, rows, res = synthetic()
    W = stencil.weights
    J = N_ETA - 1
    for i, far, near1, near2 in ((1, 0, 1, 2), (N_M - 2, N_M - 1, N_M - 2, N_M - 3)):
        for j in range(1, N_ETA - 1):
            local = {(i + di, j + dj): W[i, j, 1 + di, 1 + dj] for di in (-1, 0, 1) for dj in (-1, 0, 1)}
            want = {}
            for node, a in local.items():
                if node[0] == far:
                    want[(near1, node[1])] = want.get((near1, node[1]), 0.0) + a * (4.0 / 3.0)
                    want[(near2, node[1])] = want.get((near2, node[1]), 0.0) + a * (-1.0 / 3.0)
                else:
                    want[node] = want.get(node, 0.0) + a
            got, q, r = entries(res, (i, j))
            assert set(got) == set(want)
            for node in want:
                assert got[node] == pytest.approx(want[node], rel=1e-14, abs=0.0)
            assert q == stencil.rhs[i, j] and res.substituted[r]
    by_target = {row.target: row for row in rows}
    for i, far, near1, near2 in ((1, 0, 1, 2), (N_M - 2, N_M - 1, N_M - 2, N_M - 3)):
        row = by_target[(i, J)]
        want = {}
        for t in row.terms:
            if t.node[0] == far:
                want[(near1, t.node[1])] = want.get((near1, t.node[1]), 0.0) + t.coefficient * (4.0 / 3.0)
                want[(near2, t.node[1])] = want.get((near2, t.node[1]), 0.0) + t.coefficient * (-1.0 / 3.0)
            else:
                want[t.node] = want.get(t.node, 0.0) + t.coefficient
        got, q, r = entries(res, (i, J))
        assert set(got) == set(want)
        for node in want:
            assert got[node] == pytest.approx(want[node], rel=1e-14, abs=1e-16)
        assert q == row.rhs


def test_periodic_and_transition_rows_are_copied_unscaled():
    """A row that reads no eliminated node is the production row bit for bit: its coefficients,
    its rhs (with the +1 of the continuation that discretization.py moved there, once) and its
    center, which for the TE row is not 1 and is left so."""
    bcs, stencil, rows, res = synthetic()
    J = N_ETA - 1
    seen = set()
    for row in rows:
        if any(t.node[0] in (0, N_M - 1) for t in row.terms):
            continue
        got, q, r = entries(res, row.target)
        assert got == {t.node: t.coefficient for t in row.terms if t.coefficient != 0.0}
        assert q == row.rhs
        assert res.matrix[r, r] == row.coefficient(row.target)
        seen.add(row.kind)
    assert seen == {dz.PERIODIC_STANDARD, dz.PERIODIC_LE_TRANSITION, dz.PERIODIC_TE_TRANSITION}
    te = next(row for row in rows if row.kind == dz.PERIODIC_TE_TRANSITION)
    assert te.coefficient(te.target) != 1.0
    assert len(entries(res, te.target)[0]) == 10                  # ten reads: not a nine-slot row
    assert np.array_equal(res.constant, np.zeros(N_M * N_ETA))    # the far eliminations add no constant


def test_reconstruct_rebuilds_the_far_stations_and_keeps_the_unknowns():
    bcs, stencil, rows, res = synthetic()
    x = np.random.default_rng(3).standard_normal(res.n_unknowns)
    psi = ma.reconstruct(res, x)
    assert psi.shape == (N_M, N_ETA)
    assert np.array_equal(psi.ravel()[res.active_nodes], x)
    assert np.allclose(psi.ravel(), res.expansion @ x + res.constant, rtol=0.0, atol=1e-15)
    for relation in bcs.relations_of(bc.UPSTREAM) + bcs.relations_of(bc.DOWNSTREAM):
        assert abs(relation.residual(psi)) < 1e-14
    for j in range(N_ETA):
        for e, want in ((bcs.upstream_elimination(j), (0, j)), (bcs.downstream_elimination(j), (N_M - 1, j))):
            assert psi[want] == pytest.approx(e.value(psi), abs=1e-15)
    # the identity that makes the far jumps implied: P(0) = (4 P(1) - P(2)) / 3, P(N) likewise
    P = psi[:, -1] - psi[:, 0] - 1.0
    implied = ma.implied_residuals(res, psi)
    assert implied == pytest.approx([(4 * P[1] - P[2]) / 3, (4 * P[N_M - 2] - P[N_M - 3]) / 3], abs=1e-14)
    with pytest.raises(ma.MatrixAssemblyError):
        ma.reconstruct(res, x[:-1])
    with pytest.raises(ma.MatrixAssemblyError):
        ma.reconstruct(res, np.full(res.n_unknowns, np.nan))


# --- the substitution algebra on explicit systems -----------------------------------------------

def eq(owner, terms, rhs, kind="test"):
    return ma.LocalEquation(kind, owner, tuple(Term(n, c) for n, c in terms), rhs)


def elim(node, terms, constant):
    return Elimination(node=node, terms=tuple(Term(n, c) for n, c in terms), constant=constant)


def dense(result):
    return result.matrix.toarray(), np.asarray(result.rhs)


def test_elimination_constant_and_coefficients():
    """psi_b = 0.5 + 2 psi_x in 3 psi_b + psi_x - psi_y = 4 gives 7 psi_x - psi_y = 4 - 1.5."""
    b, x, y = (0, 0), (1, 0), (2, 0)
    res = ma.assemble_local_equations((3, 1), [eq(x, [(b, 3.0), (x, 1.0), (y, -1.0)], 4.0), eq(y, [(y, 1.0)], 1.0)],
                                      [elim(b, [(x, 2.0)], 0.5)])
    F, Q = dense(res)
    assert np.array_equal(F, [[7.0, -1.0], [0.0, 1.0]]) and np.array_equal(Q, [2.5, 1.0])
    assert res.substitutions == 1 and res.substituted.tolist() == [True, False]
    psi = ma.reconstruct(res, np.array([1.25, 1.0]))
    assert psi[b] == 0.5 + 2.0 * 1.25


@pytest.mark.parametrize("reverse", [False, True])
def test_chained_eliminations_in_any_order(reverse):
    """b1 = 1 + 2 b2, b2 = 2 + 3 x: b1 = 5 + 6 x. In 1.5 b1 + x = 2: 10 x = 2 - 7.5."""
    b1, b2, x = (0, 0), (0, 1), (1, 0)
    eliminations = [elim(b1, [(b2, 2.0)], 1.0), elim(b2, [(x, 3.0)], 2.0)]
    if reverse:
        eliminations.reverse()
    res = ma.assemble_local_equations((2, 2), [eq(x, [(b1, 1.5), (x, 1.0)], 2.0), eq((1, 1), [((1, 1), 2.0)], 0.0)],
                                      eliminations)
    F, Q = dense(res)
    assert np.array_equal(F, [[10.0, 0.0], [0.0, 2.0]]) and np.array_equal(Q, [-5.5, 0.0])
    assert [e.node for e in res.eliminations] == [b2, b1]            # dependencies first
    psi = ma.reconstruct(res, np.array([0.5, 7.0]))
    assert psi[b2] == 2.0 + 3.0 * 0.5 and psi[b1] == 1.0 + 2.0 * psi[b2]


def test_elimination_cycle_is_refused():
    b1, b2 = (0, 0), (0, 1)
    with pytest.raises(ma.MatrixAssemblyError, match="cycle"):
        ma.assemble_local_equations((2, 2), [eq((1, 0), [((1, 0), 1.0)], 0.0), eq((1, 1), [((1, 1), 1.0), (b1, 1.0)], 0.0)],
                                    [elim(b1, [(b2, 1.0)], 0.0), elim(b2, [(b1, 1.0)], 0.0)])


def test_duplicate_contributions_are_summed():
    x, y, b = (1, 0), (1, 1), (0, 0)
    res = ma.assemble_local_equations((2, 2), [eq(x, [(x, 0.25), (x, 0.5), (b, 0.5)], 1.0), eq(y, [(y, 1.0), (x, 2.0)], 0.0)],
                                      [elim(b, [(x, 1.0), (y, 1.0)], 0.0), elim((0, 1), [(y, 1.0)], 0.0)])
    F, Q = dense(res)
    assert np.array_equal(F, [[1.25, 0.5], [2.0, 1.0]])
    assert res.matrix.has_canonical_format and res.matrix.nnz == 4


@pytest.mark.parametrize("case, match", [
    ("missing", "has no equation"), ("twice", "owns 2 equations"), ("on_eliminated", "eliminated node"),
    ("off_grid", "off the"), ("negative", "off the"), ("nan", "non-finite"), ("empty_row", "is empty"),
    ("empty_column", "appears in no equation"), ("eliminated_twice", "eliminated twice"), ("float_node", "pair of integers"),
])
def test_structural_refusals(case, match):
    a, b = (0, 0), (0, 1)
    systems = {
        "missing": ([eq(a, [(a, 1.0)], 0.0)], []),
        "twice": ([eq(a, [(a, 1.0)], 0.0), eq(a, [(b, 1.0)], 0.0), eq(b, [(b, 1.0)], 0.0)], []),
        "on_eliminated": ([eq(a, [(a, 1.0)], 0.0), eq(b, [(b, 1.0)], 0.0)], [elim(b, [(a, 1.0)], 0.0)]),
        "off_grid": ([eq(a, [(a, 1.0), ((0, 2), 1.0)], 0.0), eq(b, [(b, 1.0)], 0.0)], []),
        "negative": ([eq(a, [(a, 1.0), ((0, -1), 1.0)], 0.0), eq(b, [(b, 1.0)], 0.0)], []),
        "nan": ([eq(a, [(a, float("nan"))], 0.0), eq(b, [(b, 1.0)], 0.0)], []),
        "empty_row": ([eq(a, [(a, 0.0)], 0.0), eq(b, [(b, 1.0), (a, 1.0)], 0.0)], []),
        "empty_column": ([eq(a, [(a, 1.0)], 0.0), eq(b, [(a, 1.0)], 0.0)], []),
        "eliminated_twice": ([eq(a, [(a, 1.0)], 0.0)], [elim(b, [(a, 1.0)], 0.0), elim(b, [(a, 2.0)], 0.0)]),
        "float_node": ([eq(a, [(a, 1.0), ((0.0, 1), 1.0)], 0.0), eq(b, [(b, 1.0)], 0.0)], []),
    }
    equations, eliminations = systems[case]
    with pytest.raises(ma.MatrixAssemblyError, match=match):
        ma.assemble_local_equations((1, 2), equations, eliminations)


# --- real admissible grids ----------------------------------------------------------------------

CASES = [(33.0, 29, 7), (33.0, 43, 11), (13.0, 60, 9), (23.0, 43, 9), (43.0, 26, 9), (53.0, 22, 9)]


@lru_cache(maxsize=None)
def cascade_at(stagger):
    return make_cascade(CascadeSpec(blade_type="NACA65", solidity=1.0, stagger_deg=stagger, cl0=1.2, t_max_over_c=0.10,
                                    surface_continuity="C2"))


@lru_cache(maxsize=None)
def real(stagger, n_passage, n_eta):
    grid = gr.make_grid(cascade_at(stagger), GridSettings(n_passage=n_passage, n_eta=n_eta), INLET, require_ss_ps=True)
    metrics = build_metrics(grid)
    bcs = build_boundary_conditions(grid)
    co = build_coefficients(metrics, np.ones((len(grid.m), n_eta)), 1.0)
    stencil = dz.build_interior_stencil(co, metrics)
    rows = dz.build_periodic_pde_rows(co, metrics, bcs)
    return grid, bcs, co, metrics, stencil, rows, ma.assemble_system(stencil, rows, bcs)


def local_residuals(stencil, rows, bcs, psi):
    """{owner node: residual} of every production equation, evaluated on the objects themselves."""
    n_m, n_eta = psi.shape
    W = stencil.weights
    R = -np.asarray(stencil.rhs)[1:-1, 1:-1].copy()
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            R += W[1:-1, 1:-1, 1 + di, 1 + dj] * psi[1 + di:n_m - 1 + di, 1 + dj:n_eta - 1 + dj]
    out = {(i, j): R[i - 1, j - 1] for i in range(1, n_m - 1) for j in range(1, n_eta - 1)}
    for row in rows:
        out[row.target] = row.residual(psi)
    for relation in bcs.relations:
        if relation.kind in (bc.LOWER_WALL, bc.UPPER_WALL):
            out[relation.target] = relation.residual(psi)
        elif relation.kind == bc.PERIODIC and 0 < relation.nodes[0][0] < n_m - 1:
            out[(relation.nodes[0][0], 0)] = relation.residual(psi)          # the constraint side owns it
    return out


@pytest.mark.parametrize("case", CASES)
def test_real_counts_and_eliminations(case):
    grid, bcs, co, metrics, stencil, rows, res = real(*case)
    n_m, n_eta = len(grid.m), len(grid.eta)
    f, l = grid.first_blade_i, grid.last_blade_i
    n_periodic = n_m - (l - f + 1)
    assert res.shape == (n_m, n_eta)
    assert res.n_unknowns == (n_m - 2) * n_eta == res.matrix.shape[0] == res.matrix.shape[1]
    assert len(res.eliminations) == 2 * n_eta and len(res.implied) == 2
    counts = {kind: res.row_kind.count(kind) for kind in set(res.row_kind)}
    assert counts == {ma.INTERIOR: (n_m - 2) * (n_eta - 2), bc.LOWER_WALL: l - f + 1, bc.UPPER_WALL: l - f + 1,
                      bc.PERIODIC: n_periodic - 2, dz.PERIODIC_STANDARD: n_periodic - 4, dz.PERIODIC_LE_TRANSITION: 1,
                      dz.PERIODIC_TE_TRANSITION: 1}
    produced = {e.node: e for e in res.eliminations}
    for j in range(n_eta):
        for e in (bcs.upstream_elimination(j), bcs.downstream_elimination(j)):
            assert produced[e.node] == e
    far = {(0, j) for j in range(n_eta)} | {(n_m - 1, j) for j in range(n_eta)}
    W = stencil.weights
    reads_far = sum(1 for (i, j) in np.argwhere(stencil.interior) for di in (-1, 0, 1) for dj in (-1, 0, 1)
                    if (i + di, j + dj) in far and W[i, j, 1 + di, 1 + dj] != 0.0)
    reads_far += sum(1 for row in rows for t in row.terms if t.node in far and t.coefficient != 0.0)
    assert res.substitutions == reads_far
    assert np.array_equal(res.constant, np.zeros(n_m * n_eta))
    assert np.array_equal(np.asarray(res.rhs), [_local_rhs(stencil, rows, bcs, ma.node_of(int(k), res.shape))
                                                for k in res.active_nodes])


def _local_rhs(stencil, rows, bcs, node):
    if stencil.interior[node]:
        return stencil.rhs[node]
    for row in rows:
        if row.target == node:
            return row.rhs
    for relation in bcs.relations:
        if relation.target == node or (relation.kind == bc.PERIODIC and node == (relation.nodes[0][0], 0)):
            return relation.rhs
    raise AssertionError(f"no equation for {node}")


@pytest.mark.parametrize("case", CASES)
def test_local_row_equivalence(case):
    """(F x - Q)[r] is the residual of row r's own production object on reconstruct(x), for every
    row class: the assembly changed no equation."""
    grid, bcs, co, metrics, stencil, rows, res = real(*case)
    x = np.random.default_rng(int(case[0] * 1000 + case[1])).standard_normal(res.n_unknowns)
    psi = ma.reconstruct(res, x)
    glob = res.matrix @ x - res.rhs
    local = local_residuals(stencil, rows, bcs, psi)
    assert len(local) == res.n_unknowns
    classes = {}
    for r, k in enumerate(res.active_nodes):
        node = ma.node_of(int(k), res.shape)
        kind = res.row_kind[r] + (" (substituted)" if res.substituted[r] else "")
        classes.setdefault(kind, []).append(abs(glob[r] - local[node]))
    for kind in (ma.INTERIOR, ma.INTERIOR + " (substituted)", bc.LOWER_WALL, bc.UPPER_WALL, bc.PERIODIC,
                 dz.PERIODIC_STANDARD, dz.PERIODIC_STANDARD + " (substituted)", dz.PERIODIC_LE_TRANSITION,
                 dz.PERIODIC_TE_TRANSITION):
        assert kind in classes, kind
        assert max(classes[kind]) < 1e-12, (kind, max(classes[kind]))


@pytest.mark.parametrize("case", CASES[:3])
def test_manufactured_known_vector_recovery(case):
    """Q* = F x*, then a sparse direct solve recovers x* (scipy, in the test only: the solve is
    linear_solver.py's)."""
    *_, res = real(*case)
    x_star = np.random.default_rng(7).standard_normal(res.n_unknowns)
    x = spla.spsolve(res.matrix.tocsc(), res.matrix @ x_star)
    assert np.max(np.abs(x - x_star)) / np.max(np.abs(x_star)) < 1e-10


@pytest.mark.parametrize("case", CASES[:2])
def test_the_solved_system_satisfies_every_local_equation_and_the_implied_jumps(case):
    grid, bcs, co, metrics, stencil, rows, res = real(*case)
    psi = ma.reconstruct(res, spla.spsolve(res.matrix.tocsc(), res.rhs))
    residual = local_residuals(stencil, rows, bcs, psi)
    assert max(abs(v) for v in residual.values()) < 1e-11
    assert np.max(np.abs(ma.implied_residuals(res, psi))) < 1e-11
    for relation in bcs.relations:
        assert abs(relation.residual(psi)) < 1e-11, relation.kind
    assert np.all(psi[:, 0][grid.first_blade_i:grid.last_blade_i + 1] == 0.0)
    assert np.all(psi[:, -1][grid.first_blade_i:grid.last_blade_i + 1] == 1.0)


@pytest.mark.parametrize("case", CASES)
def test_pattern_and_bandwidth_follow_the_local_reach(case):
    """Every row's columns are the nodes its local equation reads, far values replaced by their
    elimination's nodes; the bandwidths are those of the LE row (3 n_eta - 2 back) and the TE row
    (3 n_eta ahead); interior rows away from the far stations hold exactly nine entries."""
    grid, bcs, co, metrics, stencil, rows, res = real(*case)
    n_m, n_eta = res.shape
    shape = res.shape
    elim = {e.node: e for e in res.eliminations}

    def columns(nodes):
        out = set()
        for node in nodes:
            out |= {t.node for t in elim[node].terms} if node in elim else {node}
        return {int(res.full_to_active[ma.full_index(*m, shape)]) for m in out}
    low = high = 0
    F = res.matrix
    for r, k in enumerate(res.active_nodes):
        node = ma.node_of(int(k), shape)
        if res.row_kind[r] == ma.INTERIOR:
            nodes = [(node[0] + di, node[1] + dj) for di in (-1, 0, 1) for dj in (-1, 0, 1)]
        elif res.row_kind[r] in dz.PERIODIC_ROW_KINDS:
            nodes = [t.node for t in next(row for row in rows if row.target == node).terms]
        else:
            nodes = [t.node for rel in bcs.relations if rel.target == node or
                     (rel.kind == bc.PERIODIC and node == (rel.nodes[0][0], 0)) for t in rel.terms]
        cols = set(F.indices[F.indptr[r]:F.indptr[r + 1]].tolist())
        assert cols <= columns(nodes), node
        if res.row_kind[r] == ma.INTERIOR and not res.substituted[r]:
            assert len(cols) == 9
        low, high = max(low, r - min(cols)), max(high, max(cols) - r)
    s = ma.sparsity(res)
    assert (s["lower_bandwidth"], s["upper_bandwidth"]) == (low, high) == (3 * n_eta - 2, 3 * n_eta)
    assert s["row_nnz_min"] == 1 and s["row_nnz_max"] == 10 and s["zero_diagonals"] == 0
    assert s["nnz"] == F.nnz and F.has_canonical_format


def test_assembly_is_deterministic_and_matches_the_convenience_call():
    grid, bcs, co, metrics, stencil, rows, res = real(33.0, 29, 7)
    again = ma.assemble_system(stencil, rows, bcs)
    other = ma.assemble_from_coefficients(co, metrics, bcs)
    for x in (again, other):
        assert np.array_equal(x.matrix.data, res.matrix.data) and np.array_equal(x.matrix.indices, res.matrix.indices)
        assert np.array_equal(x.matrix.indptr, res.matrix.indptr) and np.array_equal(x.rhs, res.rhs)
        assert np.array_equal(x.active_nodes, res.active_nodes) and x.row_kind == res.row_kind


def test_input_refusals():
    grid, bcs, co, metrics, stencil, rows, res = real(33.0, 29, 7)
    other_grid, other_bcs, _, _, other_stencil, other_rows, _ = real(33.0, 43, 11)
    with pytest.raises(TypeError):
        ma.assemble_system(None, rows, bcs)
    with pytest.raises(TypeError):
        ma.assemble_system(stencil, rows, None)
    with pytest.raises(TypeError):
        ma.assemble_system(stencil, [None], bcs)
    with pytest.raises(ma.MatrixAssemblyError, match="grid"):
        ma.assemble_system(other_stencil, rows, bcs)
    with pytest.raises(ma.MatrixAssemblyError):
        ma.assemble_system(stencil, other_rows, bcs)
    with pytest.raises(ma.MatrixAssemblyError, match="without an equation"):
        ma.assemble_system(stencil, rows[1:], bcs)
    with pytest.raises(ma.MatrixAssemblyError, match="owns 2 equations"):
        ma.assemble_system(stencil, rows + rows[:1], bcs)


def test_result_arrays_are_read_only():
    *_, res = real(33.0, 29, 7)
    for array in (res.rhs, res.active_nodes, res.full_to_active, res.substituted, res.constant):
        with pytest.raises(ValueError):
            array[0] = 0


def test_module_holds_no_difference_physics_and_no_solve():
    """matrix_assembly.py reads the local equations and nothing else: no spacing, no stencil
    weight by name, no coefficient field, no geometry or grid import, no solve or factorization."""
    source = Path(ma.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    assert not {"dm", "deta", "west", "east", "south", "north", "mixed", "center", "H", "A", "B", "C", "D", "E",
                "pitch", "nodes_y", "first_blade_i", "last_blade_i"} & attributes
    called = {node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
              for node in ast.walk(tree) if isinstance(node, ast.Call)}
    assert not {"spsolve", "splu", "spilu", "factorized", "solve", "inv", "cg", "gmres", "bicgstab", "lsqr"} & called
    imported = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
    imported |= {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    outside = {name for name in imported if name.split(".")[0] not in sys.stdlib_module_names}
    assert outside == {"numpy", "numpy.typing", "scipy.sparse", "potential_flow_solver.boundary_conditions",
                       "potential_flow_solver.coefficients", "potential_flow_solver.discretization",
                       "potential_flow_solver.metrics"}
