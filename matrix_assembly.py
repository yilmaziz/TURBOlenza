"""
matrix_assembly.py
==================
The local linear equations of the stream-function problem gathered into one square, sparse
system F Psi = Q [Aungier 2003 Sec. 5.4: the difference equation written at the grid nodes
gives a square, highly sparse matrix; Aungier 2000, Centrifugal Compressors, Ch. 13: the
node numbering, [F_kn][Psi_n] = [Q_k], and the far-boundary values eliminated into the
adjacent equations and rebuilt after the solve]. This module adds no mathematics. Every
coefficient and right-hand side comes from the objects of discretization.py and
boundary_conditions.py; the only algebra here is the substitution of eliminated values and
the change from node indices to matrix indices. It solves nothing (linear_solver.py),
computes no coefficient, difference or boundary formula, and classifies no node from the
geometry.

    discretization.py        InteriorStencil, PeriodicPDERow ---+
    boundary_conditions.py   Relation, Elimination -------------+--> matrix_assembly.py --> linear_solver.py

1. The local equations consumed, as the upstream modules define them
    InteriorStencil: at every node with interior[i, j], the normalized Eq. 5-44
        sum over di, dj in (-1, 0, 1) of weights[i, j, 1 + di, 1 + dj] psi[i + di, j + dj] = rhs[i, j],
        center 1, the mixed term already expanded onto the four corners: the equation of (i, j).
    PeriodicPDERow: sum of term.coefficient psi[term.node] = rhs over distinct real nodes, the
        periodic continuation psi[k, n_eta] = psi[k, 1] + 1 already substituted by
        build_periodic_pde_rows (its +1 is in rhs): the equation of its target (i, n_eta - 1).
    Relation: sum of term.coefficient psi[term.node] = rhs.
        lower wall, upper wall   the equation of its target (Dirichlet psi = 0 / 1)
        periodic (jump)          target None: the equation of the image that has no PDE row,
                                 (i, 0), the constraint side of the discretization.py convention;
                                 found from the rows supplied, not from j
        upstream, downstream     not rows: each is solved for its target (Relation.solve_for),
                                 the eliminations of section 2
    Elimination: psi[node] = sum of term.coefficient psi[term.node] + constant.
    Nothing is re-derived. The +1 of the periodic continuation enters once, in the periodic rows'
    rhs (discretization.py), and the jump relations carry their own rhs (boundary_conditions.py);
    assembly copies both and adds no constant of its own other than an elimination's.

2. The unknowns
    Eliminated (Aungier 2000): every node of the two far stations, both periodic images and the
    corners, each by its own meridional relation solved for it, which is exactly
    boundary_conditions.upstream_elimination / downstream_elimination: psi[0, j] = 4/3 psi[1, j]
    - 1/3 psi[2, j] and psi[N, j] = 4/3 psi[N - 1, j] - 1/3 psi[N - 2, j]. reconstruct rebuilds them
    after the solve. The two far stations' periodic jumps are then carried by no row: with the
    meridional relations solved, P(0) = (4 P(1) - P(2)) / 3 for the jump residuals P of stations
    0, 1, 2 (boundary_conditions.py, "Corners"; likewise at the exit), so a row for them would make
    the system singular. They are kept as implied relations, checked by implied_residuals.
    Every other node is an unknown with exactly one equation: an interior node its Eq. 5-44, a wall
    node its Dirichlet relation, a periodic PDE-side node (i, n_eta - 1) its periodic row, a
    periodic constraint-side node (i, 0) its jump. Walls and jumps stay explicit rows, as
    boundary_conditions.py defines them (they are not eliminated). So there are N = (n_m - 2) n_eta
    unknowns and as many equations: checked, never patched.

3. Numbering
    full index     k = i n_eta + j, the C order of an (n_m, n_eta) array (psi.ravel()[k] = psi[i, j]):
                   full_index and node_of. The numbering grid.node_index plans (a stub there). The
                   nine-point neighbors of k are k +- 1, k +- n_eta and k +- n_eta +- 1.
    active index   the unknowns in increasing full index, 0 .. N - 1: active_nodes[r] = k and
                   full_to_active[k] = r, -1 for an eliminated node. Row r of F is the equation of
                   unknown r, its owner, so the diagonal holds the owner's own coefficient. With the
                   far stations eliminated r = k - n_eta, but nothing relies on that.

4. Substitution, exactly
    An eliminated psi_b = c_b + sum_r alpha_br psi_r in an equation sum_n a_n psi_n = q becomes
        sum_{n != b} a_n psi_n + sum_r a_b alpha_br psi_r = q - a_b c_b.
    All equations at once: psi_full = T psi + c, with T (n_full, N) holding 1 at (k, r(k)) for an
    unknown and the resolved alpha for an eliminated node, c its constant; then
        F = L T,    Q = q - L c,
    with L (N, n_full) the local coefficients (row: the owner's active index; column: the full
    index of the node). An elimination that refers to another eliminated node is resolved first
    (dependency order, depth first; a cycle is a MatrixAssemblyError, nothing is iterated). The
    far eliminations refer to stations 1, 2 and n_m - 3, n_m - 2 only, which are unknowns.

5. Storage
    COO triplets -> CSR (scipy.sparse): entries that meet on one column (a local coefficient and a
    substituted far value) are summed, exact zeros dropped. Not Aungier's nine-slot band storage:
    the periodic rows wrap from j = n_eta - 1 to j = 1, the LE transition row reaches 3 n_eta - 2
    columns back and the TE transition row reads ten nodes up to 3 n_eta columns ahead, so the
    matrix does not fit nine diagonals; sparsity reports what it is. No row is scaled or
    normalized: the TE row's center (2 A / deta^2 - 2 C / dm^2 + 3 E / (2 dm)) / H can be near or
    through zero, and scaling and pivoting belong to linear_solver.py.

6. Checks (MatrixAssemblyError)
    shapes agree, and the stencil's interior is the boundary conditions' interior; every node
    referenced lies on the grid (a negative index is refused, never wrapped); every coefficient,
    rhs and constant is finite; every unknown owns exactly one equation and no eliminated node owns
    one; every eliminated node has exactly one elimination, resolvable without a cycle; a periodic
    row stands on a periodic line; the matrix is square, finite, with no empty row and no empty
    column, and its pattern lies within the reach of the local equations through the
    eliminations.

Status: implemented 2026-11-13, under review, not frozen.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from potential_flow_solver.boundary_conditions import (DOWNSTREAM, LOWER_WALL, PERIODIC, UPPER_WALL, UPSTREAM,
                                                       BoundaryConditions, Elimination, Relation)
from potential_flow_solver.coefficients import CoefficientFields
from potential_flow_solver.discretization import (InteriorStencil, PeriodicPDERow, build_interior_stencil,
                                                  build_periodic_pde_rows)
from potential_flow_solver.metrics import GridMetrics

Node = tuple[int, int]

INTERIOR = "interior"           # row kind of an Eq. 5-44 row of InteriorStencil
_OFFSETS = tuple((di, dj) for di in (-1, 0, 1) for dj in (-1, 0, 1))


class MatrixAssemblyError(ValueError):
    """The local equations do not make one square system as section 2 of the module docstring
    defines it (an unknown without an equation or with two, an equation on an eliminated node,
    an elimination twice or in a cycle), or an input is broken (a node off the grid, a
    non-finite coefficient, shapes that disagree, an empty row or column)."""


@dataclass(frozen=True)
class LocalEquation:
    """sum of term.coefficient psi[term.node] = rhs, the equation of node owner. terms are
    objects with .node and .coefficient (boundary_conditions.Term, discretization.PDETerm)."""

    kind: str
    owner: Node
    terms: tuple
    rhs: float


@dataclass(frozen=True)
class AssemblyResult:
    """The system F Psi = Q of one grid and what it takes to read it (module docstring).
    Arrays are read-only."""

    matrix: sp.csr_matrix                   # F, (N, N), CSR, canonical (sorted, no duplicates)
    rhs: NDArray[np.float64]                # Q, (N,)
    shape: tuple[int, int]                  # (n_m, n_eta) of the grid
    active_nodes: NDArray[np.int64]         # (N,): full index of unknown r, the owner of row r
    full_to_active: NDArray[np.int64]       # (n_m n_eta,): unknown number of a node, -1 if eliminated
    row_kind: tuple[str, ...]               # (N,): the equation of row r (INTERIOR, a Relation kind, a periodic row kind)
    substituted: NDArray[np.bool_]          # (N,): the local equation of row r read an eliminated node
    eliminations: tuple[Elimination, ...]   # every eliminated node, dependencies first (reconstruct's order)
    implied: tuple[Relation, ...]           # relations carried by no row (the far stations' jumps)
    expansion: sp.csr_matrix                # T, (n_m n_eta, N): psi.ravel() = T Psi + constant
    constant: NDArray[np.float64]           # c, (n_m n_eta,)
    substitutions: int                      # local coefficients replaced by an elimination

    def __post_init__(self) -> None:
        for item in fields(self):
            value = getattr(self, item.name)
            if isinstance(value, np.ndarray):
                value.flags.writeable = False

    @property
    def n_unknowns(self) -> int:
        return int(self.matrix.shape[0])


# --- numbering --------------------------------------------------------------------------------

def full_index(i, j, shape: tuple[int, int]):
    """k = i n_eta + j of node (i, j) on an (n_m, n_eta) grid; i and j may be arrays.
    MatrixAssemblyError for a node off the grid (no negative-index wrap)."""
    n_m, n_eta = _shape(shape)
    i, j = np.asarray(i), np.asarray(j)
    if not (np.issubdtype(i.dtype, np.integer) and np.issubdtype(j.dtype, np.integer)):
        raise MatrixAssemblyError(f"node indices must be integers, got {i.dtype} and {j.dtype}")
    bad = ~((i >= 0) & (i < n_m) & (j >= 0) & (j < n_eta))
    if np.any(bad):
        where = np.flatnonzero(np.atleast_1d(bad))[0]
        raise MatrixAssemblyError(f"node ({int(np.atleast_1d(i)[where])}, {int(np.atleast_1d(j)[where])}) is off the "
                                  f"{n_m} x {n_eta} grid")
    k = i.astype(np.int64) * n_eta + j.astype(np.int64)
    return int(k) if k.ndim == 0 else k


def node_of(k, shape: tuple[int, int]):
    """(i, j) of full index k (the inverse of full_index); k may be an array, then a pair of
    arrays. MatrixAssemblyError for an index off the grid."""
    n_m, n_eta = _shape(shape)
    k = np.asarray(k)
    if not np.issubdtype(k.dtype, np.integer):
        raise MatrixAssemblyError(f"a full index must be an integer, got {k.dtype}")
    if np.any((k < 0) | (k >= n_m * n_eta)):
        raise MatrixAssemblyError(f"full index {k} is off the {n_m} x {n_eta} grid (0 .. {n_m * n_eta - 1})")
    i, j = np.divmod(k.astype(np.int64), n_eta)
    return (int(i), int(j)) if k.ndim == 0 else (i, j)


# --- assembly ---------------------------------------------------------------------------------

def assemble_system(stencil: InteriorStencil, periodic_rows, boundary_conditions: BoundaryConditions) -> AssemblyResult:
    """F Psi = Q from the production local equations of one grid: the Eq. 5-44 rows of stencil
    (discretization.build_interior_stencil), the periodic PDE rows (build_periodic_pde_rows) and
    the relations of boundary_conditions, with the far stations eliminated (module docstring).

    Raises TypeError for objects of the wrong types and MatrixAssemblyError when the equations
    do not make one square system or an input is broken (section 6)."""
    if not isinstance(stencil, InteriorStencil):
        raise TypeError(f"assemble_system takes an InteriorStencil, got {type(stencil).__name__}")
    if not isinstance(boundary_conditions, BoundaryConditions):
        raise TypeError(f"assemble_system takes BoundaryConditions, got {type(boundary_conditions).__name__}")
    rows = tuple(periodic_rows)
    for row in rows:
        if not isinstance(row, PeriodicPDERow):
            raise TypeError(f"periodic_rows must hold PeriodicPDERow objects, got {type(row).__name__}")
    bcs = boundary_conditions
    shape = _shape(bcs.shape)
    interior = np.asarray(stencil.interior, dtype=bool)
    if interior.shape != shape:
        raise MatrixAssemblyError(f"the stencil is for a {interior.shape} grid, the boundary conditions for {shape}")
    if not np.array_equal(interior, np.asarray(bcs.interior)):
        raise MatrixAssemblyError("the stencil's interior nodes are not the interior nodes of the boundary conditions: "
                                  "the two were built for different grids")
    periodic_line = np.asarray(bcs.periodic_line)
    for row in rows:
        _check_node(row.target, shape)
        if not periodic_line[row.target]:
            raise MatrixAssemblyError(f"the periodic PDE row at {row.target} is not on a periodic line of these boundary "
                                      f"conditions")

    eliminations = [r.solve_for(r.target) for r in bcs.relations if r.kind in (UPSTREAM, DOWNSTREAM)]
    eliminated = {e.node for e in eliminations}
    equations = [LocalEquation(row.kind, row.target, row.terms, row.rhs) for row in rows]
    equations += [LocalEquation(r.kind, r.target, r.terms, r.rhs) for r in bcs.relations if r.kind in (LOWER_WALL, UPPER_WALL)]
    owned = {eq.owner for eq in equations} | {tuple(int(v) for v in node) for node in np.argwhere(interior)}
    implied = []
    for relation in (r for r in bcs.relations if r.kind == PERIODIC):
        free = [n for n in relation.nodes if n not in eliminated and n not in owned]
        if all(n in eliminated for n in relation.nodes):
            implied.append(relation)
        elif len(free) == 1:
            equations.append(LocalEquation(relation.kind, free[0], relation.terms, relation.rhs))
            owned.add(free[0])
        else:
            raise MatrixAssemblyError(f"the periodic relation on {relation.nodes} has {len(free)} image(s) without an "
                                      f"equation ({free}): its owner is not determined by the rows supplied")
    for relation in bcs.relations:
        if relation.kind not in (LOWER_WALL, UPPER_WALL, PERIODIC, UPSTREAM, DOWNSTREAM):
            raise MatrixAssemblyError(f"unknown relation kind {relation.kind!r}")

    # the Eq. 5-44 rows, vectorized: nine entries per interior node
    owners_i, owners_j = np.nonzero(interior)
    weights = np.asarray(stencil.weights)[owners_i, owners_j]                     # (n_int, 3, 3)
    n_int = owners_i.size
    ent_i = np.concatenate([owners_i + di for di, dj in _OFFSETS])
    ent_j = np.concatenate([owners_j + dj for di, dj in _OFFSETS])
    ent_v = np.concatenate([weights[:, 1 + di, 1 + dj] for di, dj in _OFFSETS])
    ent_eq = np.tile(np.arange(n_int), len(_OFFSETS))
    block = _Block(kinds=[INTERIOR] * n_int, owner_i=owners_i, owner_j=owners_j,
                   rhs=np.asarray(stencil.rhs)[owners_i, owners_j], ent_eq=ent_eq, ent_i=ent_i, ent_j=ent_j, ent_v=ent_v)
    return _assemble(shape, [block, _block_of(equations)], eliminations, implied)


def assemble_from_coefficients(coefficients: CoefficientFields, metrics: GridMetrics,
                               boundary_conditions: BoundaryConditions) -> AssemblyResult:
    """build_interior_stencil, build_periodic_pde_rows and assemble_system in one call, for the
    solver's density loop; nothing else."""
    stencil = build_interior_stencil(coefficients, metrics)
    rows = build_periodic_pde_rows(coefficients, metrics, boundary_conditions)
    return assemble_system(stencil, rows, boundary_conditions)


def assemble_local_equations(shape: tuple[int, int], equations, eliminations=(), implied=()) -> AssemblyResult:
    """F Psi = Q from explicit LocalEquation objects and Elimination objects (the same core as
    assemble_system, for systems that are not one grid's production rows: the tests' hand-made
    systems). implied relations are only kept for implied_residuals."""
    shape = _shape(shape)
    equations = list(equations)
    for eq in equations:
        if not isinstance(eq, LocalEquation):
            raise TypeError(f"equations must be LocalEquation objects, got {type(eq).__name__}")
    for e in eliminations:
        if not isinstance(e, Elimination):
            raise TypeError(f"eliminations must be Elimination objects, got {type(e).__name__}")
    return _assemble(shape, [_block_of(equations)], list(eliminations), list(implied))


# --- after the solve ----------------------------------------------------------------------------

def reconstruct(result: AssemblyResult, solution: NDArray[np.float64]) -> NDArray[np.float64]:
    """psi on the whole (n_m, n_eta) grid from the solved unknowns: the unknowns copied, every
    eliminated node from its Elimination (Elimination.value) in dependency order. Nothing is
    solved or iterated."""
    if not isinstance(result, AssemblyResult):
        raise TypeError(f"reconstruct takes an AssemblyResult, got {type(result).__name__}")
    x = np.asarray(solution, dtype=float)
    if x.shape != (result.n_unknowns,):
        raise MatrixAssemblyError(f"the solution has shape {x.shape}, the system {result.n_unknowns} unknowns")
    if not np.all(np.isfinite(x)):
        raise MatrixAssemblyError("the solution is not finite")
    psi = np.full(result.shape, np.nan)
    psi.reshape(-1)[result.active_nodes] = x
    for elimination in result.eliminations:
        psi[elimination.node] = elimination.value(psi)
    if not np.all(np.isfinite(psi)):
        raise MatrixAssemblyError("a node is neither an unknown nor rebuilt by an elimination")
    return psi


def implied_residuals(result: AssemblyResult, psi: NDArray[np.float64]) -> NDArray[np.float64]:
    """Residuals of the relations no row carries (the far stations' periodic jumps) on a whole
    psi field, (n_m, n_eta): zero, to round-off, once the system is solved."""
    psi = np.asarray(psi, dtype=float)
    if psi.shape != result.shape:
        raise MatrixAssemblyError(f"psi has shape {psi.shape}, the grid {result.shape}")
    return np.array([r.residual(psi) for r in result.implied])


def sparsity(result: AssemblyResult) -> dict:
    """Size and structure of F: N, nnz, density, nonzeros per row, bandwidths (lower: max of
    row - column; upper: max of column - row), and the smallest |diagonal| (a row is never
    scaled here)."""
    F = result.matrix.tocoo()
    per_row = np.diff(result.matrix.indptr)
    n = result.n_unknowns
    diagonal = result.matrix.diagonal()
    return dict(n=n, nnz=int(F.nnz), density=F.nnz / float(n) ** 2, row_nnz_min=int(per_row.min()),
                row_nnz_max=int(per_row.max()), row_nnz_mean=float(per_row.mean()),
                lower_bandwidth=int((F.row - F.col).max()), upper_bandwidth=int((F.col - F.row).max()),
                zero_diagonals=int(np.count_nonzero(diagonal == 0.0)),
                min_abs_diagonal=float(np.abs(diagonal).min()))


# --- the core ----------------------------------------------------------------------------------

@dataclass
class _Block:
    """Equations as arrays: per equation kind, owner (i, j) and rhs; per entry the equation it
    belongs to, the node (i, j) and the coefficient."""

    kinds: list
    owner_i: NDArray
    owner_j: NDArray
    rhs: NDArray
    ent_eq: NDArray
    ent_i: NDArray
    ent_j: NDArray
    ent_v: NDArray


def _block_of(equations) -> _Block:
    kinds, oi, oj, rhs, eq_index, ei, ej, ev = [], [], [], [], [], [], [], []
    for e, eq in enumerate(equations):
        owner = _check_node(eq.owner, None)
        kinds.append(eq.kind)
        oi.append(owner[0])
        oj.append(owner[1])
        rhs.append(float(eq.rhs))
        for term in eq.terms:
            node = _check_node(term.node, None)
            eq_index.append(e)
            ei.append(node[0])
            ej.append(node[1])
            ev.append(float(term.coefficient))
    as_int = lambda values: np.asarray(values, dtype=np.int64)  # noqa: E731
    return _Block(kinds=kinds, owner_i=as_int(oi), owner_j=as_int(oj), rhs=np.asarray(rhs, dtype=float),
                  ent_eq=as_int(eq_index), ent_i=as_int(ei), ent_j=as_int(ej), ent_v=np.asarray(ev, dtype=float))


def _assemble(shape, blocks, eliminations, implied) -> AssemblyResult:
    n_m, n_eta = shape
    n_full = n_m * n_eta
    kinds, owners, rhs, ent_eq, ent_col, ent_val = [], [], [], [], [], []
    offset = 0
    for b in blocks:
        kinds += list(b.kinds)
        owners.append(full_index(b.owner_i, b.owner_j, shape))
        rhs.append(np.asarray(b.rhs, dtype=float))
        ent_eq.append(np.asarray(b.ent_eq, dtype=np.int64) + offset)
        ent_col.append(full_index(b.ent_i, b.ent_j, shape))
        ent_val.append(np.asarray(b.ent_v, dtype=float))
        offset += len(b.kinds)
    owners = np.concatenate(owners).astype(np.int64)
    rhs = np.concatenate(rhs)
    ent_eq, ent_col, ent_val = np.concatenate(ent_eq), np.concatenate(ent_col).astype(np.int64), np.concatenate(ent_val)
    for name, values in (("coefficient", ent_val), ("rhs", rhs)):
        if not np.all(np.isfinite(values)):
            raise MatrixAssemblyError(f"a local equation has a non-finite {name}")

    # eliminations: one per node, resolved in dependency order
    by_node = {}
    for e in eliminations:
        node = _check_node(e.node, shape)
        if node in by_node:
            raise MatrixAssemblyError(f"node {node} is eliminated twice")
        if not math.isfinite(float(e.constant)) or not all(math.isfinite(float(t.coefficient)) for t in e.terms):
            raise MatrixAssemblyError(f"the elimination of {node} is not finite")
        for t in e.terms:
            _check_node(t.node, shape)
        by_node[node] = e
    order, resolved = _resolve(by_node)

    active_mask = np.ones(n_full, dtype=bool)
    for node in by_node:
        active_mask[full_index(node[0], node[1], shape)] = False
    active_nodes = np.flatnonzero(active_mask).astype(np.int64)
    n = active_nodes.size
    full_to_active = np.full(n_full, -1, dtype=np.int64)
    full_to_active[active_nodes] = np.arange(n, dtype=np.int64)

    # ownership: every unknown exactly one equation, no equation on an eliminated node
    row_of_eq = full_to_active[owners]
    if np.any(row_of_eq < 0):
        e = int(np.flatnonzero(row_of_eq < 0)[0])
        raise MatrixAssemblyError(f"the {kinds[e]} equation of {node_of(int(owners[e]), shape)} stands on an eliminated node")
    count = np.bincount(row_of_eq, minlength=n)
    if np.any(count > 1):
        r = int(np.flatnonzero(count > 1)[0])
        twice = [kinds[e] for e in np.flatnonzero(row_of_eq == r)]
        raise MatrixAssemblyError(f"node {node_of(int(active_nodes[r]), shape)} owns {count[r]} equations ({twice})")
    if np.any(count == 0):
        r = int(np.flatnonzero(count == 0)[0])
        raise MatrixAssemblyError(f"the unknown {node_of(int(active_nodes[r]), shape)} has no equation: the system would "
                                  f"have {len(kinds)} equations for {n} unknowns")

    # L (N, n_full), T (n_full, N), c
    L = sp.coo_matrix((ent_val, (row_of_eq[ent_eq], ent_col)), shape=(n, n_full)).tocsr()
    t_rows, t_cols, t_vals = [active_nodes], [np.arange(n, dtype=np.int64)], [np.ones(n)]
    constant = np.zeros(n_full)
    for node in order:
        k = full_index(node[0], node[1], shape)
        terms, c = resolved[node]
        constant[k] = c
        cols = [full_to_active[full_index(m[0], m[1], shape)] for m in terms]
        t_rows.append(np.full(len(terms), k, dtype=np.int64))
        t_cols.append(np.asarray(cols, dtype=np.int64))
        t_vals.append(np.asarray(list(terms.values()), dtype=float))
    T = sp.coo_matrix((np.concatenate(t_vals), (np.concatenate(t_rows), np.concatenate(t_cols))), shape=(n_full, n)).tocsr()

    F = (L @ T).tocsr()
    F.sum_duplicates()
    F.eliminate_zeros()
    F.sort_indices()
    q = np.zeros(n)
    q[row_of_eq] = rhs
    Q = q - L @ constant

    # checks on the result
    if F.shape != (n, n):
        raise MatrixAssemblyError(f"the matrix is {F.shape}, not square")
    if not (np.all(np.isfinite(F.data)) and np.all(np.isfinite(Q))):
        raise MatrixAssemblyError("the assembled matrix or right-hand side is not finite")
    empty_rows = np.flatnonzero(np.diff(F.indptr) == 0)
    if empty_rows.size:
        r = int(empty_rows[0])
        raise MatrixAssemblyError(f"the row of {node_of(int(active_nodes[r]), shape)} ({kinds[int(np.flatnonzero(row_of_eq == r)[0])]}) "
                                  f"is empty")
    empty_cols = np.flatnonzero(np.bincount(F.indices, minlength=n) == 0)
    if empty_cols.size:
        r = int(empty_cols[0])
        raise MatrixAssemblyError(f"the unknown {node_of(int(active_nodes[r]), shape)} appears in no equation (an empty "
                                  f"column)")
    reach = (_pattern(L) @ _pattern(T)).tocsr()
    inside = _pattern(F).multiply(_pattern(reach))
    if inside.nnz != F.nnz:
        raise MatrixAssemblyError("the matrix has an entry outside the reach of its local equations")

    row_kind = [""] * n
    for e, r in enumerate(row_of_eq):
        row_kind[r] = kinds[e]
    reads_eliminated = ~active_mask[ent_col]
    substituted = np.zeros(n, dtype=bool)
    substituted[row_of_eq[ent_eq[reads_eliminated & (ent_val != 0.0)]]] = True
    return AssemblyResult(matrix=F, rhs=Q, shape=shape, active_nodes=active_nodes, full_to_active=full_to_active,
                          row_kind=tuple(row_kind), substituted=substituted, eliminations=tuple(by_node[m] for m in order),
                          implied=tuple(implied), expansion=T, constant=constant,
                          substitutions=int(np.count_nonzero(reads_eliminated & (ent_val != 0.0))))


def _resolve(by_node):
    """Every elimination in terms of unknowns only, dependencies first: (order, {node: ({unknown
    node: coefficient}, constant)}). Depth first; a cycle is a MatrixAssemblyError."""
    state, order, resolved = {}, [], {}

    def visit(node, path):
        if state.get(node) == "done":
            return
        if state.get(node) == "open":
            cycle = path[path.index(node):] + [node]
            raise MatrixAssemblyError(f"the eliminations form a cycle: {' -> '.join(map(str, cycle))}")
        state[node] = "open"
        terms, constant = {}, float(by_node[node].constant)
        for t in by_node[node].terms:
            m, a = tuple(t.node), float(t.coefficient)
            if m in by_node:
                visit(m, path + [node])
                sub_terms, sub_constant = resolved[m]
                for u, b in sub_terms.items():
                    terms[u] = terms.get(u, 0.0) + a * b
                constant += a * sub_constant
            else:
                terms[m] = terms.get(m, 0.0) + a
        resolved[node] = (terms, constant)
        state[node] = "done"
        order.append(node)

    for node in sorted(by_node):
        visit(node, [])
    return order, resolved


def _pattern(matrix):
    p = sp.csr_matrix(matrix, copy=True)
    p.data = np.ones_like(p.data)
    return p


def _check_node(node, shape) -> Node:
    try:
        values = tuple(node)
    except TypeError:
        values = ()
    if len(values) != 2 or not all(isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_)) for v in values):
        raise MatrixAssemblyError(f"a node must be a pair of integers (i, j), got {node!r}")
    i, j = int(values[0]), int(values[1])
    if shape is not None:
        full_index(i, j, shape)
    return i, j


def _shape(shape) -> tuple[int, int]:
    try:
        n_m, n_eta = (int(v) for v in shape)
    except (TypeError, ValueError):
        raise MatrixAssemblyError(f"a grid shape is (n_m, n_eta), got {shape!r}") from None
    if n_m < 1 or n_eta < 1:
        raise MatrixAssemblyError(f"a grid shape needs positive sizes, got {shape!r}")
    return n_m, n_eta
