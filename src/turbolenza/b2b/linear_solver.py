"""
linear_solver.py
================
Solves the assembled sparse system F Psi = Q of matrix_assembly.py and rebuilds the whole
psi field. The system is Aungier's [Aungier 2003 Sec. 5.4; Aungier 2000, Centrifugal
Compressors, Ch. 13], unchanged; only the linear algebra is ours: a sparse direct solve,
scipy.sparse.linalg.spsolve (SuperLU, its own column ordering and partial pivoting), where
Aungier used a problem-specific banded storage with forward and backward elimination. No
iterative method, preconditioner, row scaling, regularization or pinned node. It knows
nothing about the flow: no velocity, density, gas model or Kutta condition.

    matrix_assembly.py  AssemblyResult (F, Q, reconstruct) --> linear_solver.py --> Psi, psi(i, j), diagnostics

The pieces
    solve(matrix, rhs)            Psi from any square sparse system, checked (below)
    residual_norm(matrix, x, rhs) ||matrix @ x - rhs||_inf
    solve_linear_system(assembly) solve on assembly.matrix, assembly.rhs, then
                                  matrix_assembly.reconstruct for psi on the (n_m, n_eta) grid,
                                  with the diagnostics of LinearSolveResult
The active-node map, the eliminations and the reconstruction are matrix_assembly.py's; nothing
of them is repeated here.

The solve
    spsolve(F, Q, use_umfpack=False): SuperLU always, whatever optional packages are installed,
    so the factorization does not change with the environment. A CSR or CSC matrix is passed as
    it is (CSR, which matrix_assembly.py gives, is factorized as the transpose of a CSC matrix
    without a copy); any other sparse format is converted to CSC first. A dense matrix is
    refused: the production system is never made dense. Rows are not divided by their diagonal:
    the TE periodic row's diagonal is not 1 and can be near or through zero, and the
    factorization pivots by itself.

Failures (LinearSolveError; TypeError for an object that is not a sparse matrix or not an
AssemblyResult)
    a matrix that is not square or is empty; a rhs that is not one-dimensional of the matrix
    size; a non-finite matrix entry or rhs value; a singular matrix (SuperLU's MatrixRankWarning,
    turned into the error, never ignored); a non-finite solution (overflow); a reconstructed psi
    that is not finite or not of the grid's shape. Any other warning raised during the solve is
    passed on to the caller, not swallowed. A large residual is reported, not refused, and psi
    outside [0, 1] is no error at this stage.

Diagnostics (LinearSolveResult), r = F Psi - Q
    residual_inf       ||r||_inf
    residual_l2        ||r||_2
    relative_residual  ||r||_2 / ||Q||_2, or ||r||_2 itself when Q = 0 (no division by zero)
    backward_error     ||r||_inf / (||F||_inf ||Psi||_inf + ||Q||_inf), the normwise backward
                       error (0 when the denominator is 0, which needs r = 0)
    implied_residual   max |residual| of the relations no row carries (the far stations'
                       jumps) on psi: zero, to round-off, for a consistent system
    and the matrix shape, nnz, whether everything is finite, and min / max of Psi.

Status: implemented 2026-11-13 (direct sparse baseline), under review, not frozen. A
verification solver reproducing Aungier's banded elimination may be added later, to compare
its Psi with this one on the same F and Q.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, fields

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from numpy.typing import NDArray

from turbolenza.b2b.matrix_assembly import AssemblyResult, MatrixAssemblyError, implied_residuals, reconstruct


class LinearSolveError(ValueError):
    """The system cannot be solved as given: not square, sizes that disagree, non-finite
    entries, a singular matrix, or a solution or reconstructed field that is not finite."""


@dataclass(frozen=True)
class LinearSolveResult:
    """Psi of F Psi = Q, psi on the grid and the diagnostics of the solve (module docstring).
    Arrays are read-only."""

    active_solution: NDArray[np.float64]    # Psi, (N,): the unknowns, in matrix_assembly's active order
    psi: NDArray[np.float64]                # (n_m, n_eta): Psi with the eliminated far stations rebuilt
    residual_inf: float                     # ||F Psi - Q||_inf
    residual_l2: float                      # ||F Psi - Q||_2
    relative_residual: float                # ||r||_2 / ||Q||_2 (||r||_2 if Q = 0)
    backward_error: float                   # ||r||_inf / (||F||_inf ||Psi||_inf + ||Q||_inf)
    implied_residual: float                 # max |far-station jump residual| on psi
    matrix_shape: tuple[int, int]
    nnz: int
    finite: bool                            # Psi and psi finite (always True for a returned result)
    psi_active_min: float
    psi_active_max: float

    def __post_init__(self) -> None:
        for item in fields(self):
            value = getattr(self, item.name)
            if isinstance(value, np.ndarray):
                value.flags.writeable = False


def solve(matrix, rhs: NDArray[np.float64]) -> NDArray[np.float64]:
    """Psi of matrix @ Psi = rhs by a sparse direct solve (module docstring), checked before
    and after. TypeError for a matrix that is not a SciPy sparse matrix, LinearSolveError for
    any other failure."""
    F = _checked_matrix(matrix)
    Q = _checked_rhs(rhs, F.shape[0])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        x = spla.spsolve(F, Q, use_umfpack=False)
    singular = [w for w in caught if issubclass(w.category, spla.MatrixRankWarning)]
    for w in caught:
        if w not in singular:
            warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)
    if singular:
        raise LinearSolveError(f"the {F.shape[0]} x {F.shape[1]} matrix is singular: SuperLU reports "
                               f"'{singular[0].message}'. Nothing is regularized or pinned: the equations do not "
                               f"determine Psi")
    x = np.atleast_1d(np.asarray(x, dtype=float))
    if x.shape != (F.shape[0],):
        raise LinearSolveError(f"the solve returned shape {x.shape}, expected ({F.shape[0]},)")
    bad = np.flatnonzero(~np.isfinite(x))
    if bad.size:
        raise LinearSolveError(f"the solution is not finite at {bad.size} of {x.size} unknowns (first: {int(bad[0])}, "
                               f"{x[bad[0]]!r}): overflow or a numerically singular matrix")
    return x


def residual_norm(matrix, solution: NDArray[np.float64], rhs: NDArray[np.float64]) -> float:
    """||matrix @ solution - rhs||_inf, to check a solve."""
    F = _checked_matrix(matrix)
    Q = _checked_rhs(rhs, F.shape[0])
    x = _checked_rhs(solution, F.shape[1], name="solution")
    return float(np.max(np.abs(F @ x - Q)))


def solve_linear_system(assembly: AssemblyResult) -> LinearSolveResult:
    """Psi of the assembled F Psi = Q, psi on the whole grid by matrix_assembly.reconstruct, and
    the diagnostics (module docstring). TypeError for anything but an AssemblyResult,
    LinearSolveError when the solve or the reconstruction fails."""
    if not isinstance(assembly, AssemblyResult):
        raise TypeError(f"solve_linear_system takes a matrix_assembly.AssemblyResult, got {type(assembly).__name__}")
    F, Q = assembly.matrix, np.asarray(assembly.rhs, dtype=float)
    x = solve(F, Q)
    try:
        psi = reconstruct(assembly, x)
    except MatrixAssemblyError as error:
        raise LinearSolveError(f"the solved unknowns do not rebuild psi: {error}") from error
    psi = np.asarray(psi, dtype=float)
    if psi.shape != tuple(assembly.shape) or not np.all(np.isfinite(psi)):
        raise LinearSolveError(f"the reconstructed psi has shape {psi.shape} (grid {tuple(assembly.shape)}) or is not "
                               f"finite")
    r = F @ x - Q
    r_inf, r_l2, q_l2 = float(np.max(np.abs(r))), float(np.linalg.norm(r)), float(np.linalg.norm(Q))
    scale = float(spla.norm(F, np.inf)) * float(np.max(np.abs(x))) + float(np.max(np.abs(Q)))
    implied = implied_residuals(assembly, psi)
    return LinearSolveResult(active_solution=x, psi=psi, residual_inf=r_inf, residual_l2=r_l2,
                             relative_residual=r_l2 / q_l2 if q_l2 > 0.0 else r_l2,
                             backward_error=r_inf / scale if scale > 0.0 else 0.0,
                             implied_residual=float(np.max(np.abs(implied))) if implied.size else 0.0,
                             matrix_shape=(int(F.shape[0]), int(F.shape[1])), nnz=int(F.nnz), finite=True,
                             psi_active_min=float(x.min()), psi_active_max=float(x.max()))


def _checked_matrix(matrix):
    if not sp.issparse(matrix):
        raise TypeError(f"the matrix must be a SciPy sparse matrix (never a dense one), got {type(matrix).__name__}")
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise LinearSolveError(f"the matrix is {matrix.shape}: a linear system needs a square matrix")
    if matrix.shape[0] == 0:
        raise LinearSolveError("the matrix is empty")
    F = matrix if matrix.format in ("csr", "csc") else matrix.tocsc()
    if not np.all(np.isfinite(F.data)):
        raise LinearSolveError(f"the matrix has {int(np.count_nonzero(~np.isfinite(F.data)))} non-finite entries")
    return F


def _checked_rhs(rhs, n: int, name: str = "rhs") -> NDArray[np.float64]:
    try:
        Q = np.asarray(rhs, dtype=float)
    except (TypeError, ValueError):
        raise LinearSolveError(f"the {name} is not a numeric vector") from None
    if Q.shape != (n,):
        raise LinearSolveError(f"the {name} has shape {Q.shape}, the matrix needs ({n},)")
    if not np.all(np.isfinite(Q)):
        raise LinearSolveError(f"the {name} has {int(np.count_nonzero(~np.isfinite(Q)))} non-finite values")
    return Q
