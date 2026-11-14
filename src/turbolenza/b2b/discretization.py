"""
discretization.py
=================
Finite differences of the stream-function equation on the (m, eta) grid [Aungier 2003
Eqs. 5-35 to 5-50]. Implemented: the normalized nine-point difference equation, Eq. 5-44,
at every interior node, from the coefficient fields of coefficients.py and the spacings of
the grid. Not here yet (deliberately deferred, each a separate reviewed step): the rows of
the boundary conditions (walls, periodic lines, far inlet and exit, Eqs. 5-40 to 5-43 where
those need one-sided differences), the global numbering and sparse assembly, and the solve.

    geometry.py -> grid.py -> metrics.py -> coefficients.py -> discretization.py

Two different coefficient sets: do not mix them up
    PDE coefficients   A, B, C, D, E of Eq. 5-21 (coefficients.CoefficientFields), fields of
                       the continuous equation, in units of L/T to L^3/T.
    stencil weights    center, west, east, south, north, mixed (and the four diagonals) of
                       Eq. 5-44, dimensionless numbers of the difference equation at a node.
    Aungier writes the stencil weights with tildes over A to E (Eqs. 5-45 to 5-49), which
    are NOT the PDE coefficients of the same letters; this module never uses the letters A
    to E for a stencil weight.

The equation [Eq. 5-21]
    A psi_eta,eta - 2 B psi_m,eta + C psi_m,m + D psi_eta + E psi_m = Q,
    Q = 2 S omega sin(phi) in general, Q = 0 here (flat cascade at constant radius,
    sin(phi) = 0; coefficients.py). The expanded form is the production form; the
    conservative Eq. 5-20 is used by the tests as an independent check only.

Difference formulas [Eqs. 5-35 to 5-39]: second-order central, uniform grid
    psi_m        = (psi[i+1, j] - psi[i-1, j]) / (2 dm)                                (5-35)
    psi_eta      = (psi[i, j+1] - psi[i, j-1]) / (2 deta)                              (5-36)
    psi_m,m      = (psi[i+1, j] - 2 psi[i, j] + psi[i-1, j]) / dm^2                    (5-37)
    psi_eta,eta  = (psi[i, j+1] - 2 psi[i, j] + psi[i, j-1]) / deta^2                  (5-38)
    psi_m,eta    = (psi[i+1, j+1] - psi[i-1, j+1] - psi[i+1, j-1] + psi[i-1, j-1]) / (4 dm deta)   (5-39)
    "Second order" means a truncation error O(h^2), h = dm or deta, on a smooth psi. All
    five are centred on the node: no upwind or flow-direction-dependent difference is used
    anywhere in this solver. The one-sided boundary formulas (Eqs. 5-40 to 5-43) are
    second-order ONE-SIDED differences, not "upwind", and they belong to the boundary rows.

The raw nine-point equation (substitution of Eqs. 5-35 to 5-39 into Eq. 5-21)
    a_P psi_P + a_W psi_W + a_E psi_E + a_S psi_S + a_N psi_N
              + a_NW psi_NW + a_NE psi_NE + a_SW psi_SW + a_SE psi_SE = Q,
        a_P  = -2 A / deta^2 - 2 C / dm^2 = -H
        a_W  = C / dm^2 - E / (2 dm)          a_E = C / dm^2 + E / (2 dm)
        a_S  = A / deta^2 - D / (2 deta)      a_N = A / deta^2 + D / (2 deta)
        a_NE = -B / (2 dm deta)               a_NW = +B / (2 dm deta)
        a_SE = +B / (2 dm deta)               a_SW = -B / (2 dm deta)
    with W = (i-1, j), E = (i+1, j), S = (i, j-1), N = (i, j+1): west/east along m, south/
    north along eta. The diagonal signs come from -2 B times Eq. 5-39: -2 B / (4 dm deta) =
    -B / (2 dm deta) on the (+, +) and (-, -) corners, + on the (+, -) and (-, +) ones.

Aungier's normalization [Eqs. 5-44 to 5-50]
    H = 2 A / deta^2 + 2 C / dm^2 > 0 (A > 0 and C > 0), and the raw equation divided by -H:
        psi[i, j] + west psi[i-1, j] + east psi[i+1, j] + south psi[i, j-1] + north psi[i, j+1]
          + mixed (psi[i+1, j+1] - psi[i+1, j-1] - psi[i-1, j+1] + psi[i-1, j-1]) = rhs,    (5-44)
    Aungier's tilde weights and the names here:
        A~ = west  = -(C / dm^2 - E / (2 dm)) / H                                         (5-45)
        B~ = east  = -(C / dm^2 + E / (2 dm)) / H                                         (5-46)
        C~ = south = -(A / deta^2 - D / (2 deta)) / H                                     (5-47)
        D~ = north = -(A / deta^2 + D / (2 deta)) / H                                     (5-48)
        E~ = mixed = B / (2 dm deta H)                                                    (5-49)
        Q~ = rhs   = -Q / H = 0 here                                                      (5-50)
        center = 1;  northeast = southwest = +mixed;  southeast = northwest = -mixed.
    So the normalized residual is -(raw residual) / H. Row properties that follow (tested):
    west + east + south + north = -1, so with center = 1 and the mixed bracket a constant
    psi is an exact solution; the mixed weights carry no diagonal dominance (4 |mixed| > 0
    whenever B != 0), and west / east (south / north) can turn positive where the first
    derivative term outweighs the second, |E| dm / (2 C) > 1 (|D| deta / (2 A) > 1): a
    property of central differences, reported by the audit, not corrected here.

Lengths [coefficients.py]
    The grid is chord-normalized: grid.dm = dm*. The difference formulas need the m of the
    coefficient fields, so
        dm = coefficients.length_scale * grid.dm        (c_ref in metres, or 1 chord),
        deta = grid.deta                                 (eta is dimensionless),
    the one convention of coefficients.py, never reinterpreted here. InteriorStencil keeps
    the dm and deta it used.

Interior nodes and the representation
    An interior node has all eight neighbors of the nine-point stencil in the local (i, j)
    indexing: i = 1 .. n_m - 2 and j = 1 .. n_eta - 2 (InteriorStencil.interior). Every
    stencil array is (n_m, n_eta) for a later assembly by node, and holds NaN at every other
    node, so a node without a valid Eq. 5-44 row can never pass for one:
        j = 0, n_eta - 1 on the blade stations   walls, psi given (boundary_conditions.py)
        j = 0, n_eta - 1 on the periodic stations periodic lines: Eq. 5-21 holds there, but
                                                  its row needs the neighbor across the line
                                                  from the other side, with the jump of psi:
                                                  build_periodic_pde_rows (below), not here
        i = 0, n_m - 1                            far inlet and exit, psi_m = 0 [Eq. 5-33]
    The interior rows next to them read those nodes as ordinary neighbors. The stations
    either side of first_blade_i and last_blade_i are interior like any other (the m
    differences cross the region transitions, as in coefficients.py).

Inputs and checks
    build_interior_stencil(coefficients, metrics) reads dm*, deta and the shape from
    metrics.grid and the length scale from the coefficients, and refuses coefficient fields
    that do not belong to these metrics: Eqs. 5-22 to 5-24 give, whatever rho and mdot',
        C = A (L S*)^2 cos^2(beta)    and    B = C tan(beta) / (L S*),
    which are checked at every node to ROUND_OFF (a stale coefficient set after
    grid.rebuild_exit, a field of another grid or another length scale is a
    DiscretizationError). Then, as for raw arrays: fields of one (n_m, n_eta) shape with
    n_m, n_eta >= 3, finite; dm and deta finite and positive; H finite and positive at every
    interior node; every weight finite. Nothing is clipped or replaced.

Measured (2026-11-12; tests/test_discretization.py, scripts/audit_discretization.py; the
default C2 cascade, 65-(12A10)10 at 33 deg, inlet 45 deg, rho = 1)
    Smooth manufactured coefficients and psi: the operator converges at second order with dm
    and deta refined together, apart, and in the mixed term alone (orders 1.94 to 2.00);
    quadratics are exact; against a conservative face-flux discretization of Eq. 5-20 the
    gap is second order too (1.98 to 2.00).
    n_passage 101, n_eta 31: every weight finite at the 9889 interior nodes; H 2.84e4 to
    7.53e4; west -0.470 .. -0.186, east -0.483 .. -0.196, south -0.339 .. +0.067, north
    -0.279 .. +0.094, mixed -0.120 .. +0.121; west + east + south + north = -1 to 2.2e-16;
    |mixed| <= |sin beta| / 8 (B / sqrt(A C) = sin beta for coefficients of real metrics,
    and H >= 4 sqrt(A C) / (dm deta)), max 0.1212, so the sum of |neighbor weights| is 1.0001
    to 1.485: no row is diagonally dominant where beta != 0.
    Corner stations: south or north turn positive only at first_blade_i - 1 and + 1 (and
    last_blade_i + 1 at n_passage 401), where the large D of the corners (coefficients.py)
    gives |D| deta / (2 A) up to 3.9 (101) and 3.0 (401): 25 of 9889 nodes at 101, 144 of
    161364 at 401. Finite, unclipped. There the expanded and the conservative forms differ
    by an amount that does not fall with refinement (the expansion's derivative of a corner).
    Curvature steps of the C2 walls (front 60 % of the chord): weights finite, bit for bit
    reproducible, every sign the one its formula gives; the station-to-station change of
    the weights is 4 to 6 times that of the smooth rear (2.6e-3 against 4.4e-4 at 101), and
    the gap to the conservative form is about first order there (second order on the rear).

The periodic PDE line (build_periodic_pde_rows, added 2026-11-12)
    Outside the blade row Eq. 5-21 also holds on the periodic line, once per image pair. The
    convention for the assembly: j = 0 is the constraint side (psi[i, 0] = psi[i, J] - 1, the
    relation of boundary_conditions.py) and j = J = n_eta - 1 the PDE side, so a row is
    written at (i, J) on the periodic stations i = 1 .. first_blade_i - 1 and last_blade_i + 1
    .. n_m - 2 (the far stations take their psi_m = 0 relation; blade stations have walls).
    Neither image is eliminated here, and the j = 0 relation is not touched.
    The row is Eq. 5-44 at (i, J): center 1 and the pointwise weights west, east, south,
    north, mixed and H of Eqs. 5-45 to 5-49 at that node (the reviewed formulas, evaluated
    by the same code as InteriorStencil on the node's own coefficients), with every node
    above the line replaced by its periodic continuation,
        psi[k, J + 1] = psi[k, 1] + 1          (boundary_conditions.periodic_ghost, jump 1),
    so the weight w of (k, J + 1) moves to (k, 1) and w * 1 moves to the right-hand side:
        psi_eta     = (psi[i, 1] + 1 - psi[i, J - 1]) / (2 deta),
        psi_eta,eta = (psi[i, 1] + 1 - 2 psi[i, J] + psi[i, J - 1]) / deta^2.
    The physical source is still Q = 0, but the row's rhs is not: rhs = -north (the north
    weight times the jump, moved across), = (A / deta^2 + D / (2 deta)) / H. psi_m and
    psi_m,m stay central on the real nodes (i - 1, J) and (i + 1, J), except on the TE row.
    The mixed derivative is psi_m,eta = d q / dm with q[k] = (psi[k, 1] + 1 - psi[k, J - 1]) /
    (2 deta), the eta difference at (k, J), and its m difference sum c_k q[i + k] / (2 dm):
        PERIODIC_STANDARD       c = (-1 at i - 1, +1 at i + 1)       central, Eq. 5-39
        PERIODIC_LE_TRANSITION  c = (3 at i, -4 at i - 1, 1 at i - 2) at i = first_blade_i - 1
        PERIODIC_TE_TRANSITION  c = (-3 at i, 4 at i + 1, -1 at i + 2) at i = last_blade_i + 1
    In the weights: +c_k mixed on (k, 1) and -c_k mixed on (k, J - 1); the jump constants
    of the q's add up to sum c_k = 0, so the mixed term puts nothing on the rhs.
    Why the two transition rows differ: at first_blade_i - 1 the central mixed difference
    would need psi[first_blade_i, J + 1], above the upper wall of the first blade station,
    where the blade is and no periodic continuation exists (periodic_ghost refuses it), and
    likewise psi[last_blade_i, J + 1] at last_blade_i + 1. The second-order backward
    (forward) one-sided m difference keeps the mixed term on periodic stations upstream
    (downstream) of the transition. One-sided for the topology, not upwind: no direction of
    the flow enters. This is OUR topology-aware second-order extension of Aungier's
    Taylor-series finite differences (his LE/TE principle: no stencil through the blade,
    one-sided differences into the fluid), not a numbered Aungier formula.
    The final transition policy (the asymmetry is measured geometry, not a flow direction):
        LE row (first_blade_i - 1)  psi_m, psi_m,m  central: its links to (first_blade_i, J)
                                                    and (first_blade_i - 2, J) stay in the fluid
                                    psi_m,eta       backward one-sided (above)
        TE row (last_blade_i + 1)   psi_m   = (-3 psi[i, J] + 4 psi[i+1, J] - psi[i+2, J]) / (2 dm)
                                    psi_m,m = (2 psi[i, J] - 5 psi[i+1, J] + 4 psi[i+2, J]
                                               - psi[i+3, J]) / dm^2           (TE_M_DIFFERENCE,
                                                                                TE_MM_DIFFERENCE)
                                    psi_m,eta       forward one-sided (above)
    At the TE the central psi_m and psi_m,m read the upper wall node (last_blade_i, J), and the
    straight link from (last_blade_i + 1, J) to it runs through the +pitch blade's tail
    (6.3e-3 c of a 1.7e-2 c link at n_passage 61, 5.4e-3 of 1.2e-2 at 101, 2.0e-3 of 7.6e-3 at
    201, 5.9e-3 of 6.2e-3 at 401: scripts/visualize_te_stencils.py); the forward forms read
    only last_blade_i + 1 .. + 4 on the line j = J, downstream of the tail. Taylor series of
    the two: (-3, 4, -1) / (2 dm) is exact for quadratics, error -dm^2 f''' / 3; (2, -5, 4, -1)
    / dm^2 exact for cubics, error -11 dm^2 f'''' / 12: both second order. They act along one
    eta line, so no jump constant enters; psi_eta and psi_eta,eta stay as in every row. The
    TE row is still divided by the same H = 2 A / deta^2 + 2 C / dm^2, so its center is not 1:
    (2 A / deta^2 - 2 C / dm^2 + 3 E / (2 dm)) / H (the one-sided psi_m,m weighs the node itself
    with +2 C / dm^2), -0.78 at n_passage 101 and -0.98 at 401 (n_eta 41), and its largest
    weight sits on (last_blade_i + 2, J) (+2.2 to +2.5): not diagonally dominant, for the
    linear solver to know. Dividing by the center instead could divide by a number near
    zero. Error O(dm^2 + deta^2) on smooth fields, every row.
    A row is a PeriodicPDERow: sum of coefficient * psi[node] over its PDETerms = rhs, center
    coefficient 1 at target (i, J) (not on the TE row, above), nodes merged where two coincide
    (n_eta = 3), exact zeros dropped (never the target); H is its divisor. Requirements: at
    least three periodic stations on the inlet side and four on the exit side
    (DiscretizationError otherwise; no first-order fallback).
    Measured (tests/test_discretization.py, scripts/audit_discretization.py, the default C2
    cascade at rho = 1): the rows agree with Eq. 5-21 derived term by term to 2.8e-17;
    second order on a quasi-periodic psi (orders 1.95 to 2.00, the TE row 2.00, the mixed term
    alone too); the TE row's psi_m and psi_m,m, read through the row, exact for quadratics and
    cubics and second order (2.03, 2.00 at the finest). n_passage 101: 239 rows (237
    standard), all finite; the LE row's coefficients -0.440 .. +0.243, rhs 0.197. No link of a
    periodic row crosses metal at any of 28 grids (n_passage 31 to 401, n_eta 21 to 241).
    Not treated here (the reviewed interior stencil, InteriorStencil, unchanged): interior
    rows beside the transitions, at j = 1 and j = J - 1 of the stations next to them, have
    links through the nose or the tail at some resolutions (34 links in 32 rows over those 28
    grids, both edges, both walls, more as n_eta grows, at n_eta 241 an E/W link too), e.g.
    the NE diagonal of (last_blade_i, J - 1) at n_passage 401, n_eta 121: 2.38e-3 c of 1.42e-2 c.
    Their spread points to the grid topology at the LE and TE (grid.py section 6), not to
    more stencil exceptions.

Study option: face-flux rows at the transitions (transition_form, 2026-11-14;
scripts/transition_study.py; default "expanded", the rows above, unchanged)
    "flux" replaces the interior rows (j = 1 .. n_eta - 2) of the stations first_blade_i - 1,
    first_blade_i, last_blade_i and last_blade_i + 1 by the divergence form Eq. 5-20 on face
    fluxes, so that D and E never appear:
        [F(i, j+1/2) - F(i, j-1/2)] / deta - [G(i+1/2, j) - G(i-1/2, j)] / dm = 0,
        F = A psi_eta - B psi_m,   G = B psi_eta - C psi_m,
    the face coefficients the means of the two nodes (A(i, j+1/2) = (A[i, j] + A[i, j+1]) / 2,
    likewise B and C), the normal difference across the face two-point ((psi[i, j+1] -
    psi[i, j]) / deta), the tangential one the mean of the central differences of the face's two
    nodes ((psi_m)(i, j+1/2) = (psi[i+1, j] - psi[i-1, j] + psi[i+1, j+1] - psi[i-1, j+1]) /
    (4 dm), likewise (psi_eta)(i+1/2, j)). A nine-point row on the same neighbours, divided by the
    same H as the row it replaces. The periodic-line rows (j = n_eta - 1) are not replaced.
    Against the expanded row it differs, at every node, by
        (A[j+1] - 2 A[j] + A[j-1]) (psi[j+1] - 2 psi[j] + psi[j-1]) / (4 deta^2)
      + (C[i+1] - 2 C[i] + C[i-1]) (psi[i+1] - 2 psi[i] + psi[i-1]) / (4 dm^2)  + the same for B,
    O(h^2) where the fields are smooth, zero for constant coefficients (and for coefficients
    linear along a direction in which psi is linear); no single-valued face flux reproduces the
    central expanded row exactly for general coefficients (it is not conservative).

Status: build_interior_stencil and InteriorStencil implemented 2026-11-12, under review,
not frozen; build_periodic_pde_rows added 2026-11-12 and its TE row revised the same day
(forward psi_m, psi_m,m; InteriorStencil, the standard and the LE rows unchanged bit for
bit), under review. first_derivatives, mixed_derivative, residual and assemble remain
interfaces for the assembly step.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from numbers import Real

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from turbolenza.b2b.boundary_conditions import BoundaryConditions
from turbolenza.b2b.coefficients import CoefficientFields, Coefficients
from turbolenza.b2b.grid import Grid
from turbolenza.b2b.metrics import ROUND_OFF, GridMetrics

Field = NDArray[np.float64]
Node = tuple[int, int]

_WEIGHTS = ("center", "west", "east", "south", "north", "mixed", "rhs", "H")

TRANSITION_FORMS = ("expanded", "flux")     # study option (module docstring); "expanded" is the production form

PERIODIC_STANDARD = "periodic standard"
PERIODIC_LE_TRANSITION = "periodic LE transition"
PERIODIC_TE_TRANSITION = "periodic TE transition"
PERIODIC_ROW_KINDS = (PERIODIC_STANDARD, PERIODIC_LE_TRANSITION, PERIODIC_TE_TRANSITION)
# psi_m,eta on the periodic PDE line = sum c q[i + offset] / (2 dm), q the periodic eta difference (module docstring)
MIXED_M_DIFFERENCE = {
    PERIODIC_STANDARD: ((-1, -1.0), (1, 1.0)),                       # central, Eq. 5-39
    PERIODIC_LE_TRANSITION: ((0, 3.0), (-1, -4.0), (-2, 1.0)),       # backward one-sided: stays upstream
    PERIODIC_TE_TRANSITION: ((0, -3.0), (1, 4.0), (2, -1.0)),        # forward one-sided: stays downstream
}
# The TE row's psi_m and psi_m,m along j = J, forward one-sided (the central ones read (last_blade_i, J) through the tail)
TE_M_DIFFERENCE = ((0, -3.0), (1, 4.0), (2, -1.0))                 # psi_m   = sum c psi[i + o, J] / (2 dm)
TE_MM_DIFFERENCE = ((0, 2.0), (1, -5.0), (2, 4.0), (3, -1.0))      # psi_m,m = sum c psi[i + o, J] / dm^2


class DiscretizationError(ValueError):
    """An input breaks what the difference equation assumes: coefficient fields that do not
    belong to the metrics (or their length scale), arrays of the wrong or mixed shapes or
    fewer than three nodes a side, non-finite fields, a spacing that is not finite and
    positive, H = 2 A / deta^2 + 2 C / dm^2 not finite and positive at an interior node, or
    a stencil weight that comes out non-finite."""


@dataclass(frozen=True)
class InteriorStencil:
    """The normalized nine-point equation, Eq. 5-44, at every interior node (module
    docstring): each array (n_m, n_eta), read-only, NaN wherever interior is False. These
    are stencil weights, not the PDE coefficients A to E."""

    center: Field                   # 1                                          psi[i, j]
    west: Field                     # A~ (5-45): -(C/dm^2 - E/(2 dm)) / H        psi[i-1, j]
    east: Field                     # B~ (5-46): -(C/dm^2 + E/(2 dm)) / H        psi[i+1, j]
    south: Field                    # C~ (5-47): -(A/deta^2 - D/(2 deta)) / H    psi[i, j-1]
    north: Field                    # D~ (5-48): -(A/deta^2 + D/(2 deta)) / H    psi[i, j+1]
    mixed: Field                    # E~ (5-49): B / (2 dm deta H), times psi_NE - psi_SE - psi_NW + psi_SW
    rhs: Field                      # Q~ (5-50): -Q / H, 0 for this solver (Q = 0)
    H: Field                        # 2 A/deta^2 + 2 C/dm^2 > 0, the divisor (units of A / deta^2)
    interior: NDArray[np.bool_]     # i = 1 .. n_m - 2 and j = 1 .. n_eta - 2
    dm: float                       # coefficients.length_scale * grid.dm, the m spacing of the differences
    deta: float                     # grid.deta

    def __post_init__(self) -> None:
        interior = np.array(self.interior, dtype=bool)
        if interior.ndim != 2:
            raise DiscretizationError(f"interior has shape {interior.shape}: it must be (n_m, n_eta)")
        interior.flags.writeable = False
        object.__setattr__(self, "interior", interior)
        for item in fields(self):
            if item.name not in _WEIGHTS:
                continue
            array = np.array(getattr(self, item.name), dtype=float)
            if array.shape != interior.shape:
                raise DiscretizationError(f"{item.name} has shape {array.shape}, interior {interior.shape}")
            if not (np.all(np.isfinite(array[interior])) and np.all(np.isnan(array[~interior]))):
                raise DiscretizationError(f"{item.name} must be finite at the interior nodes and NaN elsewhere")
            array.flags.writeable = False
            object.__setattr__(self, item.name, array)
        for name in ("dm", "deta"):
            object.__setattr__(self, name, _spacing(getattr(self, name), name))

    @property
    def northeast(self) -> Field:
        """Weight of psi[i+1, j+1]: +mixed."""
        return self.mixed

    @property
    def southwest(self) -> Field:
        """Weight of psi[i-1, j-1]: +mixed."""
        return self.mixed

    @property
    def southeast(self) -> Field:
        """Weight of psi[i+1, j-1]: -mixed."""
        return _read_only(-self.mixed)

    @property
    def northwest(self) -> Field:
        """Weight of psi[i-1, j+1]: -mixed."""
        return _read_only(-self.mixed)

    @property
    def weights(self) -> NDArray[np.float64]:
        """All nine weights, (n_m, n_eta, 3, 3): entry [i, j, 1 + di, 1 + dj] multiplies
        psi[i + di, j + dj] in the equation of node (i, j); NaN outside the interior."""
        w = np.empty(self.center.shape + (3, 3))
        w[..., 1, 1] = self.center
        w[..., 0, 1], w[..., 2, 1] = self.west, self.east
        w[..., 1, 0], w[..., 1, 2] = self.south, self.north
        w[..., 2, 2] = w[..., 0, 0] = self.mixed
        w[..., 2, 0] = w[..., 0, 2] = -self.mixed
        return _read_only(w)


@dataclass(frozen=True)
class PDETerm:
    """coefficient * psi[node] in a periodic PDE row."""

    node: Node
    coefficient: float


@dataclass(frozen=True)
class PeriodicPDERow:
    """Eq. 5-44 at a periodic-line node (module docstring, "The periodic PDE line"): the sum
    of coefficient * psi[node] over the terms = rhs, center coefficient 1 at target (standard
    and LE rows; the TE row's is (2 A / deta^2 - 2 C / dm^2 + 3 E / (2 dm)) / H, from its
    one-sided m differences). H is the divisor of the normalization (2 A / deta^2 + 2 C / dm^2
    at target), the same for every kind."""

    kind: str
    target: Node
    terms: tuple[PDETerm, ...]
    rhs: float
    H: float

    def __post_init__(self) -> None:
        if self.kind not in PERIODIC_ROW_KINDS:
            raise DiscretizationError(f"unknown periodic row kind {self.kind!r}")
        nodes = [t.node for t in self.terms]
        if len(set(nodes)) != len(nodes):
            raise DiscretizationError(f"a periodic PDE row needs distinct nodes, got {nodes}")
        if not all(math.isfinite(t.coefficient) for t in self.terms) or not (math.isfinite(self.rhs)
                                                                             and math.isfinite(self.H) and self.H > 0.0):
            raise DiscretizationError(f"the periodic PDE row at {self.target} has a non-finite coefficient, rhs or H")
        if self.target not in nodes:
            raise DiscretizationError(f"the periodic PDE row at {self.target} does not read its own node")
        if self.kind != PERIODIC_TE_TRANSITION and self.coefficient(self.target) != 1.0:
            raise DiscretizationError(f"the periodic PDE row at {self.target} must have the center coefficient 1")

    @property
    def nodes(self) -> tuple[Node, ...]:
        return tuple(t.node for t in self.terms)

    def coefficient(self, node: Node) -> float:
        """The coefficient of psi[node] in the row, 0 for a node it does not read."""
        return next((t.coefficient for t in self.terms if t.node == tuple(node)), 0.0)

    def residual(self, psi: NDArray[np.float64]) -> float:
        """sum of coefficient * psi[node] - rhs: zero for a psi that satisfies the row."""
        return float(sum(t.coefficient * psi[t.node] for t in self.terms) - self.rhs)


def build_interior_stencil(coefficients: CoefficientFields, metrics: GridMetrics) -> InteriorStencil:
    """Eq. 5-44 at every interior node of metrics.grid from the PDE coefficient fields
    (module docstring). coefficients must be built from these metrics: checked through the
    rho-independent identities of Eqs. 5-22 to 5-24. dm = coefficients.length_scale *
    grid.dm, deta = grid.deta.

    Raises TypeError for objects of the wrong types and DiscretizationError for any broken
    input (module docstring, "Inputs and checks")."""
    if not isinstance(coefficients, CoefficientFields):
        raise TypeError(f"build_interior_stencil takes CoefficientFields, got {type(coefficients).__name__}")
    if not isinstance(metrics, GridMetrics):
        raise TypeError(f"build_interior_stencil takes GridMetrics, got {type(metrics).__name__}")
    grid = metrics.grid
    shape = (len(grid.m), len(grid.eta))
    if coefficients.A.shape != shape:
        raise DiscretizationError(f"the coefficient fields are {coefficients.A.shape}, the grid has {shape} nodes")
    _check_belongs(coefficients, metrics)
    return _interior_stencil(coefficients.A, coefficients.B, coefficients.C, coefficients.D, coefficients.E,
                             coefficients.length_scale * grid.dm, grid.deta)


def _check_belongs(coefficients: CoefficientFields, metrics: GridMetrics) -> None:
    """DiscretizationError unless C = A (L S*)^2 cos^2(beta) and B = C tan(beta) / (L S*) at
    every node to ROUND_OFF: the coefficient fields of these metrics at their length scale."""
    S = coefficients.length_scale * np.asarray(metrics.S, dtype=float)[:, None]
    cos, tan = np.asarray(metrics.cos_beta, dtype=float), np.asarray(metrics.tan_beta, dtype=float)
    A, B, C = coefficients.A, coefficients.B, coefficients.C
    with np.errstate(all="ignore"):
        off_C = np.abs(A * (S * cos) ** 2 - C) / np.abs(C)
        scale = np.abs(C / S) * np.maximum(1.0, np.abs(tan))
        off_B = np.abs(C * tan / S - B) / scale
    for name, off in (("C / A = (L S*)^2 cos^2(beta)", off_C), ("B / C = tan(beta) / (L S*)", off_B)):
        bad = np.argwhere(~(off <= ROUND_OFF))
        if bad.size:
            i, j = (int(k) for k in bad[0])
            raise DiscretizationError(f"the coefficient fields do not belong to these metrics: {name} fails at node "
                                      f"({i}, {j}) by {off[i, j]:.3g} (relative). Rebuild the coefficients from these "
                                      f"metrics (after grid.rebuild_exit too), at the length scale meant")


def build_periodic_pde_rows(coefficients: CoefficientFields, metrics: GridMetrics,
                            boundary_conditions: BoundaryConditions) -> tuple[PeriodicPDERow, ...]:
    """Eq. 5-44 at the PDE side (i, n_eta - 1) of every periodic station 1 .. first_blade_i - 1
    and last_blade_i + 1 .. n_m - 2, in station order (module docstring, "The periodic PDE
    line"). The coefficients must be those of metrics, and boundary_conditions those of
    metrics.grid, whose periodic_ghost gives the continuation psi[k, n_eta] = psi[k, 1] + 1.
    dm = coefficients.length_scale * grid.dm, deta = grid.deta, as for build_interior_stencil.

    Raises TypeError for objects of the wrong types and DiscretizationError for broken
    inputs (and BoundaryConditionError, from periodic_ghost, should a row ever ask for a
    continuation on a blade station)."""
    if not isinstance(coefficients, CoefficientFields):
        raise TypeError(f"build_periodic_pde_rows takes CoefficientFields, got {type(coefficients).__name__}")
    if not isinstance(metrics, GridMetrics):
        raise TypeError(f"build_periodic_pde_rows takes GridMetrics, got {type(metrics).__name__}")
    if not isinstance(boundary_conditions, BoundaryConditions):
        raise TypeError(f"build_periodic_pde_rows takes BoundaryConditions, got {type(boundary_conditions).__name__}")
    grid = metrics.grid
    if boundary_conditions.grid is not grid:
        raise DiscretizationError("the boundary conditions were built from another grid than the metrics")
    shape = (len(grid.m), len(grid.eta))
    if coefficients.A.shape != shape:
        raise DiscretizationError(f"the coefficient fields are {coefficients.A.shape}, the grid has {shape} nodes")
    _check_belongs(coefficients, metrics)
    n_eta = shape[1]
    return _periodic_pde_rows(coefficients.A, coefficients.B, coefficients.C, coefficients.D, coefficients.E,
                              coefficients.length_scale * grid.dm, grid.deta, grid.first_blade_i, grid.last_blade_i,
                              lambda k: boundary_conditions.periodic_ghost(k, n_eta))


def _periodic_pde_rows(A: Field, B: Field, C: Field, D: Field, E: Field, dm: float, deta: float, first_blade_i: int,
                       last_blade_i: int, ghost) -> tuple[PeriodicPDERow, ...]:
    """The rows of build_periodic_pde_rows from raw arrays, the blade stations and a ghost
    rule (ghost(k): the Elimination of psi[k, n_eta], as boundary_conditions.periodic_ghost
    gives it): a separate function so that the rows can be built on manufactured fields."""
    A, B, C, D, E = (np.asarray(a, dtype=float) for a in (A, B, C, D, E))
    if A.ndim != 2:
        raise DiscretizationError(f"A has shape {A.shape}: the fields must be (n_m, n_eta)")
    n_m, n_eta = A.shape
    J = n_eta - 1
    f, l = int(first_blade_i), int(last_blade_i)
    if not 1 <= f <= l <= n_m - 2:
        raise DiscretizationError(f"blade stations {f} to {l} do not fit {n_m} stations")
    if f < 3 or n_m - 1 - l < 4:
        raise DiscretizationError(f"{f} inlet and {n_m - 1 - l} exit periodic stations: the second-order one-sided "
                                  f"differences of the transition rows need three on the inlet side and four on the exit "
                                  f"side (the TE row's psi_m,m reaches last_blade_i + 4; no first-order fallback)")
    # The pointwise weights of Eqs. 5-45 to 5-49 at (i, J), by the reviewed core: on the columns (J - 1, J, J) its
    # interior column is column J, at stations 1 .. n_m - 2 (each weight depends on its node's coefficients only).
    columns = [J - 1, J, J]
    line = _interior_stencil(A[:, columns], B[:, columns], C[:, columns], D[:, columns], E[:, columns], dm, deta)
    rows = []
    for i in list(range(1, f)) + list(range(l + 1, n_m - 1)):
        kind = PERIODIC_LE_TRANSITION if i == f - 1 else PERIODIC_TE_TRANSITION if i == l + 1 else PERIODIC_STANDARD
        mixed = float(line.mixed[i, 1])
        terms: dict[Node, float] = {}
        rhs = float(line.rhs[i, 1])                                   # -Q / H, Q = 0

        def add(node: Node, weight: float) -> None:
            terms[node] = terms.get(node, 0.0) + weight

        def above(k: int, weight: float) -> float:
            """weight * psi[k, J + 1] through the periodic continuation: the rhs change."""
            rule = ghost(k)
            if rule.node != (k, n_eta):
                raise DiscretizationError(f"the ghost rule for station {k} is for {rule.node}, not {(k, n_eta)}")
            for t in rule.terms:
                add(t.node, weight * t.coefficient)
            return -weight * rule.constant

        if kind == PERIODIC_TE_TRANSITION:
            # psi_m, psi_m,m forward one-sided along j = J: the raw weights C c / dm^2 + E c / (2 dm) over -H, and the
            # eta part of the center, 2 A / deta^2 over H (psi_eta,eta's -2 psi[i, J] / deta^2 over -H)
            a_ij, c_ij, e_ij, H = float(A[i, J]), float(C[i, J]), float(E[i, J]), float(line.H[i, 1])
            along: dict[int, float] = {}
            for offset, c in TE_MM_DIFFERENCE:
                along[offset] = along.get(offset, 0.0) + c * c_ij / dm ** 2
            for offset, c in TE_M_DIFFERENCE:
                along[offset] = along.get(offset, 0.0) + c * e_ij / (2.0 * dm)
            add((i, J), 2.0 * a_ij / deta ** 2 / H)
            for offset in sorted(along):
                add((i + offset, J), -along[offset] / H)
        else:
            add((i, J), float(line.center[i, 1]))
            add((i - 1, J), float(line.west[i, 1]))
            add((i + 1, J), float(line.east[i, 1]))
        add((i, J - 1), float(line.south[i, 1]))
        rhs += above(i, float(line.north[i, 1]))
        for offset, c in MIXED_M_DIFFERENCE[kind]:
            rhs += above(i + offset, c * mixed)
            add((i + offset, J - 1), -c * mixed)
        rows.append(PeriodicPDERow(kind=kind, target=(i, J), rhs=rhs, H=float(line.H[i, 1]),
                                   terms=tuple(PDETerm(node, w) for node, w in terms.items()
                                               if w != 0.0 or node == (i, J))))
    return tuple(rows)


def _interior_stencil(A: Field, B: Field, C: Field, D: Field, E: Field, dm: float, deta: float) -> InteriorStencil:
    """Eq. 5-44 from raw coefficient arrays and spacings: the whole computation and every
    check of build_interior_stencil but the object ones (a separate function so that the
    formulas can be tried on fields no CoefficientFields of a grid would hold)."""
    dm, deta = _spacing(dm, "dm"), _spacing(deta, "deta")
    arrays = {}
    for name, value in (("A", A), ("B", B), ("C", C), ("D", D), ("E", E)):
        array = np.asarray(value, dtype=float)
        if array.ndim != 2 or (arrays and array.shape != arrays["A"].shape):
            raise DiscretizationError(f"{name} has shape {array.shape}: A to E must be (n_m, n_eta) arrays of one shape")
        bad = np.argwhere(~np.isfinite(array))
        if bad.size:
            raise DiscretizationError(f"{name} is not finite at node {tuple(int(k) for k in bad[0])}")
        arrays[name] = array
    n_m, n_eta = arrays["A"].shape
    if n_m < 3 or n_eta < 3:
        raise DiscretizationError(f"{n_m} x {n_eta} nodes hold no interior node: the nine-point stencil needs three a side")
    interior = np.zeros((n_m, n_eta), dtype=bool)
    interior[1:-1, 1:-1] = True
    A, B, C, D, E = (np.where(interior, arrays[name], np.nan) for name in "ABCDE")
    with np.errstate(all="ignore"):
        H = 2.0 * A / deta ** 2 + 2.0 * C / dm ** 2
    bad = np.argwhere(interior & ~(np.isfinite(H) & (H > 0.0)))
    if bad.size:
        i, j = (int(k) for k in bad[0])
        raise DiscretizationError(f"H = 2 A / deta^2 + 2 C / dm^2 = {H[i, j]!r} at node ({i}, {j}) (A = {A[i, j]:.6g}, "
                                  f"C = {C[i, j]:.6g}): it must be finite and positive")
    with np.errstate(all="ignore"):
        west = -(C / dm ** 2 - E / (2.0 * dm)) / H                     # Eq. 5-45
        east = -(C / dm ** 2 + E / (2.0 * dm)) / H                     # Eq. 5-46
        south = -(A / deta ** 2 - D / (2.0 * deta)) / H                # Eq. 5-47
        north = -(A / deta ** 2 + D / (2.0 * deta)) / H                # Eq. 5-48
        mixed = B / (2.0 * dm * deta * H)                              # Eq. 5-49
    center = np.where(interior, 1.0, np.nan)
    rhs = np.where(interior, 0.0, np.nan)                              # Eq. 5-50: -Q / H with Q = 0
    for name, weight in (("west", west), ("east", east), ("south", south), ("north", north), ("mixed", mixed)):
        bad = np.argwhere(interior & ~np.isfinite(weight))
        if bad.size:
            i, j = (int(k) for k in bad[0])
            raise DiscretizationError(f"the {name} weight is not finite at node ({i}, {j}) (H = {H[i, j]:.6g})")
    return InteriorStencil(center=center, west=west, east=east, south=south, north=north, mixed=mixed, rhs=rhs, H=H,
                           interior=interior, dm=dm, deta=deta)


def transition_flux_weights(coefficients: CoefficientFields, metrics: GridMetrics) -> tuple[NDArray[np.float64],
                                                                                           tuple[Node, ...]]:
    """The face-flux rows of the study option transition_form "flux" (module docstring): the
    normalized nine weights (n_m, n_eta, 3, 3) at the replaced rows, NaN elsewhere, and those
    rows' nodes. Same entry convention as InteriorStencil.weights."""
    if not isinstance(coefficients, CoefficientFields):
        raise TypeError(f"transition_flux_weights takes CoefficientFields, got {type(coefficients).__name__}")
    if not isinstance(metrics, GridMetrics):
        raise TypeError(f"transition_flux_weights takes GridMetrics, got {type(metrics).__name__}")
    grid = metrics.grid
    _check_belongs(coefficients, metrics)
    dm, deta = coefficients.length_scale * grid.dm, grid.deta
    A, C = coefficients.A, coefficients.C
    H = 2.0 * A / deta ** 2 + 2.0 * C / dm ** 2
    return _flux_transition_weights(A, coefficients.B, C, dm, deta, grid.first_blade_i, grid.last_blade_i, H)


def _flux_transition_weights(A: Field, B: Field, C: Field, dm: float, deta: float, first_blade_i: int,
                             last_blade_i: int, H: Field) -> tuple[NDArray[np.float64], tuple[Node, ...]]:
    """transition_flux_weights from raw arrays; H the divisor of the rows replaced."""
    A, B, C, H = (np.asarray(a, dtype=float) for a in (A, B, C, H))
    n_m, n_eta = A.shape
    f, l = int(first_blade_i), int(last_blade_i)
    stations = sorted({f - 1, f, l, l + 1})
    if stations[0] < 1 or stations[-1] > n_m - 2:
        raise DiscretizationError(f"the transition stations {stations} do not fit {n_m} stations")
    rows = tuple((i, j) for i in stations for j in range(1, n_eta - 1))
    out = np.full((n_m, n_eta, 3, 3), np.nan)
    h, k = float(deta), float(dm)
    for i, j in rows:
        w = _flux_raw_weights(A, B, C, i, j, h, k)
        divisor = float(H[i, j])
        if not (math.isfinite(divisor) and divisor > 0.0):
            raise DiscretizationError(f"H = {divisor!r} at the flux row ({i}, {j})")
        out[i, j] = w / -divisor
        if not np.all(np.isfinite(out[i, j])):
            raise DiscretizationError(f"a face-flux weight is not finite at ({i}, {j})")
    return out, rows


def _flux_raw_weights(A: Field, B: Field, C: Field, i: int, j: int, h: float, k: float) -> NDArray[np.float64]:
    """The raw nine weights [1 + di, 1 + dj] of the face-flux row at (i, j) (module docstring,
    study option): dF/deta - dG/dm, h = deta, k = dm, face coefficients the means of two nodes."""
    a_p, a_m = 0.5 * (A[i, j] + A[i, j + 1]), 0.5 * (A[i, j] + A[i, j - 1])
    c_p, c_m = 0.5 * (C[i, j] + C[i + 1, j]), 0.5 * (C[i, j] + C[i - 1, j])
    be_p, be_m = 0.5 * (B[i, j] + B[i, j + 1]), 0.5 * (B[i, j] + B[i, j - 1])     # B on the eta faces j +- 1/2
    bm_p, bm_m = 0.5 * (B[i, j] + B[i + 1, j]), 0.5 * (B[i, j] + B[i - 1, j])     # B on the m faces i +- 1/2
    q = 1.0 / (4.0 * h * k)
    w = np.zeros((3, 3))
    w[1, 1] = -(a_p + a_m) / h ** 2 - (c_p + c_m) / k ** 2
    w[1, 2] = a_p / h ** 2 - (bm_p - bm_m) * q                  # N (i, j+1)
    w[1, 0] = a_m / h ** 2 + (bm_p - bm_m) * q                  # S (i, j-1)
    w[2, 1] = c_p / k ** 2 - (be_p - be_m) * q                  # E (i+1, j)
    w[0, 1] = c_m / k ** 2 + (be_p - be_m) * q                  # W (i-1, j)
    w[2, 2] = -(be_p + bm_p) * q                                # NE
    w[0, 2] = (be_p + bm_m) * q                                 # NW
    w[2, 0] = (be_m + bm_p) * q                                 # SE
    w[0, 0] = -(be_m + bm_m) * q                                # SW
    return w


def _spacing(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    value = float(value)
    if not (math.isfinite(value) and value > 0.0):
        raise DiscretizationError(f"{name} = {value!r}: a grid spacing must be finite and positive")
    return value


def _read_only(array: Field) -> Field:
    array.flags.writeable = False
    return array


# --- interfaces of the boundary and assembly step (not implemented) ------------------------

def first_derivatives(field: NDArray[np.float64], grid: Grid,
                      periodic_jump: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """d field / dm at fixed eta and d field / deta at fixed m, at every node. On a periodic
    line the neighbor across it is taken from the other side of the passage, shifted by
    periodic_jump (the passage flow for psi, 0 for density)."""
    raise NotImplementedError


def mixed_derivative(field: NDArray[np.float64], grid: Grid, periodic_jump: float) -> NDArray[np.float64]:
    """d2 field / dm deta at every node, with the edges treated as in first_derivatives."""
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
