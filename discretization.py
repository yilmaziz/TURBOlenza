"""
discretization.py
=================
Finite differences of the stream-function equation on the (m, eta) grid, and assembly of
the sparse linear system for psi [Aungier 2003 Eqs. 5-35 to 5-50].

Eq. 5-44 is the difference equation at a node. Its coefficients, Eqs. 5-45 to 5-50, come
from Eq. 5-21 with the difference formulas of Eqs. 5-35 to 5-43.

Pieces
    first and mixed derivatives   difference formulas for d/dm, d/deta and d2/dm deta,
                                  with the one-sided and periodic forms at the edges
    interior stencil              the nine weights that tie a node to its neighbors; the
                                  mixed derivative is why there are nine and not five
    residual                      the difference equation evaluated for a given psi
    assembly                      one row per node: stencil rows inside, rows for the
                                  conditions of boundary_conditions.py on the edges

b2b/SOLVER_PLAN.md (Step 2) also tests a conservative form of Eq. 5-20. If it is kept, it
gets its own stencil and residual functions next to these.

Status: interface only. No difference formula and no assembly is implemented.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from potential_flow_solver.boundary_conditions import BoundaryConditions
from potential_flow_solver.coefficients import Coefficients
from potential_flow_solver.grid import Grid


def first_derivatives(field: NDArray[np.float64], grid: Grid,
                      periodic_jump: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """d field / dm at fixed eta and d field / deta at fixed m, at every node. On a periodic
    line the neighbor across it is taken from the other side of the passage, shifted by
    periodic_jump (the passage flow for psi, 0 for density)."""
    raise NotImplementedError


def mixed_derivative(field: NDArray[np.float64], grid: Grid, periodic_jump: float) -> NDArray[np.float64]:
    """d2 field / dm deta at every node, with the edges treated as in first_derivatives."""
    raise NotImplementedError


def interior_stencil(coefficients: Coefficients, grid: Grid) -> NDArray[np.float64]:
    """Weights of the difference equation, (n_m, n_eta, 3, 3): entry [i, j, 1 + di, 1 + dj]
    multiplies psi[i + di, j + dj] in the equation of node (i, j) [Eqs. 5-44 to 5-50]."""
    raise NotImplementedError


def residual(psi: NDArray[np.float64], coefficients: Coefficients, grid: Grid) -> NDArray[np.float64]:
    """The difference equation at every interior node for a given psi. It is zero for the
    discrete solution, and on an exact solution of Eq. 5-20 it falls with grid refinement at
    the order of the scheme: the first thing to check when a solution looks wrong."""
    raise NotImplementedError


def assemble(coefficients: Coefficients, grid: Grid,
             bc: BoundaryConditions) -> tuple[sp.csr_matrix, NDArray[np.float64]]:
    """The sparse matrix and right-hand side of the linear system for psi, with the unknowns
    numbered by grid.node_index."""
    raise NotImplementedError
