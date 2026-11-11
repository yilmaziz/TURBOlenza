"""
linear_solver.py
================
Solves the sparse linear system matrix @ psi = rhs. It knows nothing about the flow: a
matrix and a vector in, a vector out, so it can be tested on any system with a known
answer.

Status: interface only. Whether the solve is direct or iterative is chosen when this is
implemented.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray


def solve(matrix: sp.csr_matrix, rhs: NDArray[np.float64]) -> NDArray[np.float64]:
    """The solution vector."""
    raise NotImplementedError


def residual_norm(matrix: sp.csr_matrix, solution: NDArray[np.float64], rhs: NDArray[np.float64]) -> float:
    """Size of matrix @ solution - rhs, to check a solve."""
    raise NotImplementedError
