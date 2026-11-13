"""
Tests for turbolenza.b2b.linear_solver: the sparse direct solve of the assembled
F Psi = Q and the rebuilt psi field.

What is checked
- Known systems: a diagonal one, a coupled nonsymmetric tridiagonal one, a random sparse
  nonsingular one, a zero-diagonal permutation (the solve pivots; no row is divided by its
  diagonal), a COO input (converted) and CSR / CSC inputs (passed as they are).
- Refusals: a non-square or empty matrix, a rhs of the wrong length or shape, NaN in the
  matrix, Inf in the rhs, a dense matrix, a singular matrix (SuperLU's MatrixRankWarning made
  an error), an overflowing solution, a solve that returns NaN (mutation-style), a non-finite
  reconstruction; other warnings are passed on, not swallowed.
- solve_linear_system on hand-made AssemblyResults: the eliminated nodes rebuilt by
  matrix_assembly.reconstruct, the diagnostics' definitions.
- Real admissible Aungier systems (33 deg n_passage 43 first, C2 65-(12A10)10, rho = 1): a
  known vector recovered, the solve's residuals, the residual reported against an
  independent F Psi - Q, psi on the grid, and the original local production equations
  (tests/test_matrix_assembly.local_residuals) satisfied by the reconstructed field.

Tolerances name what they cover: known small systems to 1e-14 relative (one or two roundings);
the random 400-unknown system to 1e-12 (measured below 1e-14); real-system recovery to 1e-10
(measured 2e-14 to 3e-13, test_matrix_assembly); real residuals to 1e-12 absolute (measured
below 3e-15 on entries of order 1).
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from turbolenza.b2b import linear_solver as ls
from turbolenza.b2b import matrix_assembly as ma
from turbolenza.b2b.boundary_conditions import Elimination, Term

import test_matrix_assembly as tma        # the assembly tests' real pipeline and local-equation evaluator


# --- known systems -------------------------------------------------------------------------------

def test_diagonal_system():
    F = sp.csr_matrix(np.diag([2.0, 3.0, 4.0]))
    x_star = np.array([1.0, 2.0, 3.0])
    x = ls.solve(F, F @ x_star)
    assert np.array_equal(x, x_star)
    assert ls.residual_norm(F, x, F @ x_star) == 0.0


def test_coupled_nonsymmetric_tridiagonal_system():
    n = 7
    main, lower, upper = np.full(n, 4.0), np.full(n - 1, -1.3), np.full(n - 1, -0.7)
    F = sp.diags([lower, main, upper], [-1, 0, 1], format="csr")
    x_star = np.sin(np.arange(1, n + 1))
    x = ls.solve(F, F @ x_star)
    assert np.max(np.abs(x - x_star)) <= 1e-14 * np.max(np.abs(x_star))


def test_random_sparse_nonsingular_system():
    rng = np.random.default_rng(5)
    n = 400
    R = sp.random(n, n, density=0.01, random_state=rng, format="csr")
    F = (R + sp.diags(np.abs(R).sum(axis=1).A1 + 1.0)).tocsr()       # strictly diagonally dominant
    x_star = rng.standard_normal(n)
    x = ls.solve(F, F @ x_star)
    assert np.max(np.abs(x - x_star)) / np.max(np.abs(x_star)) < 1e-12


def test_zero_diagonal_is_solved_by_pivoting_not_by_row_scaling():
    """A permutation has a zero diagonal: a solver that divides rows by their center fails, the
    factorization pivots."""
    F = sp.csr_matrix(np.array([[0.0, 2.0, 0.0], [0.0, 0.0, -1.0], [3.0, 0.0, 0.0]]))
    x_star = np.array([1.5, -2.0, 0.25])
    assert np.array_equal(ls.solve(F, F @ x_star), x_star)
    T = sp.csr_matrix(np.array([[1e-14, 1.0], [1.0, 1.0]]))                 # a near-zero pivot, like the TE row
    y_star = np.array([0.3, -0.8])
    assert np.max(np.abs(ls.solve(T, T @ y_star) - y_star)) < 1e-14


@pytest.mark.parametrize("fmt", ["csr", "csc", "coo", "lil"])
def test_sparse_formats(fmt):
    F = sp.diags([np.full(4, -1.0), np.full(5, 3.0), np.full(4, -1.0)], [-1, 0, 1]).asformat(fmt)
    x_star = np.arange(5.0) - 1.5
    with warnings.catch_warnings():
        warnings.simplefilter("error", sp.SparseEfficiencyWarning)     # no silent format conversion inside spsolve
        x = ls.solve(F, F @ x_star)
    assert np.max(np.abs(x - x_star)) < 1e-14


# --- refusals ----------------------------------------------------------------------------------------

@pytest.mark.parametrize("matrix, rhs, match", [
    (sp.csr_matrix(np.ones((2, 3))), np.ones(2), "square"),
    (sp.csr_matrix((0, 0)), np.ones(0), "empty"),
    (sp.csr_matrix(np.eye(3)), np.ones(2), "shape"),
    (sp.csr_matrix(np.eye(3)), np.ones((3, 1)), "shape"),
    (sp.csr_matrix(np.array([[1.0, np.nan], [0.0, 1.0]])), np.ones(2), "non-finite entries"),
    (sp.csr_matrix(np.eye(2)), np.array([1.0, np.inf]), "non-finite values"),
    (sp.csr_matrix(np.array([[1.0, 1.0], [1.0, 1.0]])), np.ones(2), "exactly singular"),
    (sp.csr_matrix(np.array([[1.0, 0.0], [0.0, 0.0]])), np.ones(2), "exactly singular"),         # an empty row
    (sp.csr_matrix(np.array([[1.0, 0.0], [2.0, 0.0]])), np.ones(2), "exactly singular"),         # an empty column
    (sp.csr_matrix(np.array([[1e-300]])), np.array([1e300]), "not finite"),              # overflow
])
def test_refusals(matrix, rhs, match):
    with pytest.raises(ls.LinearSolveError, match=match):
        ls.solve(matrix, rhs)


def test_dense_matrix_is_refused():
    with pytest.raises(TypeError, match="sparse"):
        ls.solve(np.eye(3), np.ones(3))
    with pytest.raises(TypeError):
        ls.solve_linear_system(sp.csr_matrix(np.eye(3)))


def test_a_nan_solution_from_the_factorization_is_refused(monkeypatch):
    """Mutation-style: if spsolve ever returned NaN without its warning, solve still refuses."""
    monkeypatch.setattr(ls.spla, "spsolve", lambda F, Q, use_umfpack=False: np.full(F.shape[0], np.nan))
    with pytest.raises(ls.LinearSolveError, match="not finite"):
        ls.solve(sp.csr_matrix(np.eye(2)), np.ones(2))


def test_other_warnings_are_passed_on(monkeypatch):
    def noisy(F, Q, use_umfpack=False):
        warnings.warn("something else", RuntimeWarning)
        return np.ones(F.shape[0])
    monkeypatch.setattr(ls.spla, "spsolve", noisy)
    with pytest.warns(RuntimeWarning, match="something else"):
        ls.solve(sp.csr_matrix(np.eye(2)), np.ones(2))


def test_singular_assembled_system_is_reported_not_regularized():
    """Two unknowns tied by the same equation twice: singular, refused with the solver's word
    for it; nothing pinned."""
    a, b = (0, 0), (0, 1)
    eq = lambda owner, terms, rhs: ma.LocalEquation("test", owner, tuple(Term(n, c) for n, c in terms), rhs)  # noqa: E731
    res = ma.assemble_local_equations((1, 2), [eq(a, [(a, 1.0), (b, 1.0)], 1.0), eq(b, [(a, 2.0), (b, 2.0)], 2.0)])
    with pytest.raises(ls.LinearSolveError, match="exactly singular"):
        ls.solve_linear_system(res)


# --- solve_linear_system on hand-made assemblies -------------------------------------------------------

def test_solve_linear_system_rebuilds_the_eliminated_nodes():
    """psi_b = 0.5 + 2 psi_x; 7 psi_x - psi_y = 2.5 (after substitution), psi_y = 1: psi_x = 0.5,
    psi_b = 1.5."""
    b, x, y = (0, 0), (1, 0), (2, 0)
    eq = lambda owner, terms, rhs: ma.LocalEquation("test", owner, tuple(Term(n, c) for n, c in terms), rhs)  # noqa: E731
    res = ma.assemble_local_equations((3, 1), [eq(x, [(b, 3.0), (x, 1.0), (y, -1.0)], 4.0), eq(y, [(y, 1.0)], 1.0)],
                                      [Elimination(node=b, terms=(Term(x, 2.0),), constant=0.5)])
    out = ls.solve_linear_system(res)
    assert out.active_solution == pytest.approx([0.5, 1.0], abs=1e-15)
    assert out.psi.shape == (3, 1)
    assert out.psi.ravel() == pytest.approx([1.5, 0.5, 1.0], abs=1e-15)
    assert np.array_equal(out.psi, ma.reconstruct(res, out.active_solution))
    assert out.matrix_shape == (2, 2) and out.nnz == 3 and out.finite
    assert out.psi_active_min == pytest.approx(0.5) and out.psi_active_max == pytest.approx(1.0)
    assert out.residual_inf < 1e-15 and out.implied_residual == 0.0


def test_diagnostics_follow_their_definitions(monkeypatch):
    """With a deliberately perturbed solution the reported norms are those of r = F Psi - Q."""
    *_, res = tma.real(33.0, 29, 7)
    exact = ls.solve(res.matrix, res.rhs)
    perturbed = exact + 1e-6 * np.cos(np.arange(exact.size))
    monkeypatch.setattr(ls, "solve", lambda F, Q: perturbed)
    out = ls.solve_linear_system(res)
    r = res.matrix @ perturbed - res.rhs
    assert out.residual_inf == np.max(np.abs(r)) and out.residual_l2 == np.linalg.norm(r)
    assert out.relative_residual == np.linalg.norm(r) / np.linalg.norm(res.rhs)
    norm_F = np.max(np.abs(res.matrix).sum(axis=1))
    assert out.backward_error == pytest.approx(np.max(np.abs(r)) / (norm_F * np.max(np.abs(perturbed))
                                                                     + np.max(np.abs(res.rhs))), rel=1e-14)
    assert out.residual_inf > 1e-8                     # the perturbation shows: the diagnostics are not hard-wired
    psi = ma.reconstruct(res, perturbed)
    assert np.array_equal(out.psi, psi)
    assert out.implied_residual == np.max(np.abs(ma.implied_residuals(res, psi))) > 1e-8
    assert (out.psi_active_min, out.psi_active_max) == (perturbed.min(), perturbed.max())


def test_result_arrays_are_read_only():
    *_, res = tma.real(33.0, 29, 7)
    out = ls.solve_linear_system(res)
    for array in (out.active_solution, out.psi):
        with pytest.raises(ValueError):
            array[0] = 0.0


def test_a_non_finite_reconstruction_is_refused(monkeypatch):
    *_, res = tma.real(33.0, 29, 7)
    monkeypatch.setattr(ls, "reconstruct", lambda assembly, x: np.full(assembly.shape, np.nan))
    with pytest.raises(ls.LinearSolveError, match="not finite"):
        ls.solve_linear_system(res)


# --- the real Aungier system ---------------------------------------------------------------------------

def test_known_vector_recovery_on_the_real_33_deg_system():
    *_, res = tma.real(33.0, 43, 11)
    x_star = np.random.default_rng(9).standard_normal(res.n_unknowns)
    x = ls.solve(res.matrix, res.matrix @ x_star)
    assert np.max(np.abs(x - x_star)) / np.max(np.abs(x_star)) < 1e-10


@pytest.mark.parametrize("case", [(33.0, 43, 21)] + tma.CASES)
def test_real_admissible_system_solves_and_satisfies_its_local_equations(case):
    """The assembled Aungier system of an admissible grid solves; F Psi - Q is at round-off and
    agrees with an independent evaluation; psi has the grid's shape, its walls are 0 and 1, the
    implied far jumps hold, and every original local production equation (interior Eq. 5-44,
    periodic PDE rows incl. LE / TE, walls, jumps) holds on the reconstructed field."""
    grid, bcs, co, metrics, stencil, rows, res = tma.real(*case)
    out = ls.solve_linear_system(res)
    n_m, n_eta = len(grid.m), len(grid.eta)
    assert out.matrix_shape == (res.n_unknowns, res.n_unknowns) == ((n_m - 2) * n_eta,) * 2
    assert out.nnz == res.matrix.nnz
    r = res.matrix @ out.active_solution - res.rhs
    assert out.residual_inf == np.max(np.abs(r)) < 1e-12
    assert out.residual_l2 == np.linalg.norm(r) and out.relative_residual < 1e-13 and out.backward_error < 1e-14
    assert ls.residual_norm(res.matrix, out.active_solution, res.rhs) == out.residual_inf
    assert out.psi.shape == (n_m, n_eta) and np.all(np.isfinite(out.psi))
    assert np.array_equal(out.psi.ravel()[res.active_nodes], out.active_solution)
    assert np.array_equal(out.psi, ma.reconstruct(res, out.active_solution))
    f, l = grid.first_blade_i, grid.last_blade_i
    assert np.all(out.psi[f:l + 1, 0] == 0.0) and np.all(out.psi[f:l + 1, -1] == 1.0)
    assert out.implied_residual < 1e-12
    local = tma.local_residuals(stencil, rows, bcs, out.psi)
    for k, node in enumerate(ma.node_of(int(k), res.shape) for k in res.active_nodes):
        assert abs(local[node] - r[k]) < 1e-12, node
    assert max(abs(v) for v in local.values()) < 1e-12
    for relation in bcs.relations:                    # the far relations too, through the reconstruction
        assert abs(relation.residual(out.psi)) < 1e-12, relation.kind
