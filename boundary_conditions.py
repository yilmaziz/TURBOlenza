"""
boundary_conditions.py
======================
The boundary conditions of the stream-function problem on the (m, eta) grid, as explicit
local linear relations between node values of psi [Aungier 2003 Sec. 5.3, Fig. 5-3,
Eqs. 5-29 to 5-33 and the one-sided differences of Eqs. 5-40 to 5-43; the matrix strategy of
Aungier 2000, Centrifugal Compressors, Ch. 13]. This module classifies every boundary node
and owns the mathematics of each condition; it builds no matrix row, numbers no unknown,
substitutes nothing and solves nothing (the next step, matrix assembly, combines these
relations with the interior stencil of discretization.py). No Kutta condition (Eq. 5-34 is
an outer constraint on the discharge angle: kutta.py, solver.py), no density, and no
difference equation of the interior (Eq. 5-44 is discretization.py's).

    grid.py (topology) -> boundary_conditions.py ---+
    coefficients.py -> discretization.py -----------+--> matrix assembly (next step)

The relations
    Every condition is a relation  sum over k of a_k psi[node_k] = rhs  (Relation), with
    integer (i, j) nodes, i along m and j along eta, zero-based:
      lower wall   blade stations, j = 0:          psi[i, 0] = 0                       PSI_LOWER_WALL
      upper wall   blade stations, j = n_eta - 1:  psi[i, n_eta - 1] = 1               PSI_UPPER_WALL
      periodic     periodic stations:              psi[i, n_eta - 1] - psi[i, 0] = 1   PSI_PERIODIC_JUMP
      upstream     i = 0, every j:                 -3 psi[0, j] + 4 psi[1, j] - psi[2, j] = 0
      downstream   i = n_m - 1, every j:           3 psi[N, j] - 4 psi[N-1, j] + psi[N-2, j] = 0,  N = n_m - 1
    Walls: a blade wall is a streamline; the normalized psi of one passage is 0 on the lower
    wall (the reference blade's passage-facing contour) and 1 on the upper one (the +pitch
    neighbor's). These are definitions, exact, not computed from geometry, density, pitch or
    mass flow. Dirichlet: no stencil and no derivative.
    Periodic: outside the blade row eta = 0 and eta = 1 are one line and its image one pitch
    up (S = pitch there, metrics.py), so psi jumps by the passage flow, 1, across a pitch:
    psi_upper - psi_lower = 1, never psi_upper = psi_lower. A periodic scalar field (the
    density) has jump 0: periodic_relation(i, jump=0.0).
    Upstream and downstream: the grid lines of the far stations run along the flow angle
    (grid.py builds the extensions so), and Aungier's far condition is psi_m = 0 there
    [Eq. 5-33]. The relations are the second-order ONE-SIDED differences [Eqs. 5-40 to 5-43],
        psi_m[0]     = (-3 psi[0] + 4 psi[1] - psi[2]) / (2 dm) + O(dm^2)        forward
        psi_m[n_m-1] = (3 psi[N] - 4 psi[N-1] + psi[N-2]) / (2 dm) + O(dm^2)    backward
    set to zero and multiplied by 2 dm. One-sided because the far station has neighbors on
    one side only, not because of a flow direction: they are not "upwind" differences, and
    this solver uses no upwinding anywhere. dm is the physical m spacing of coefficients.py
    (dm = length_scale grid.dm), but the condition is homogeneous, so 2 dm cancels: the
    relations and their eliminations do not depend on c_ref. No first-order fallback.

Elimination (for the Aungier 2000 strategy)
    Relation.solve_for(node) solves a relation for any of its nodes (Elimination: psi[node] =
    sum b_k psi[k] + constant). The far-station ones, for substitution into the PDE rows of
    the adjacent stations and the reconstruction afterwards:
        upstream_elimination(j):    psi[0, j]   = 4/3 psi[1, j]   - 1/3 psi[2, j]
        downstream_elimination(j):  psi[N, j]   = 4/3 psi[N-1, j] - 1/3 psi[N-2, j]
    The coefficients 4/3 and -1/3 sum to 1, so a constant psi and the periodic jump pass
    through an elimination unchanged.

Node classes and the topology source
    The blade stations are grid.first_blade_i .. grid.last_blade_i, the stations whose line
    cuts the blade (grid.py section 3), cross-checked against grid.wall_identity.stations
    and grid.region; never inferred from m, the LE/TE landmarks or surface names. Masks, each
    (n_m, n_eta), read-only:
        lower_wall, upper_wall          j = 0 / n_eta - 1 of the blade stations
        periodic_lower, periodic_upper  j = 0 / n_eta - 1 of the periodic stations (far ones included)
        upstream, downstream            i = 0 / n_m - 1, every j (the corners included)
    No node is both a wall and periodic; the first blade station's and the last blade
    station's j = 0 and n_eta - 1 are walls like any other (no special LE/TE condition, no
    averaging). The nodes in none of the masks are the interior nodes of
    discretization.InteriorStencil (i = 1 .. n_m - 2, j = 1 .. n_eta - 2).

The periodic lines: duplicated endpoints, kept
    On a periodic station the grid stores both images, j = 0 and j = n_eta - 1; neither is
    removed or chosen here (the periodic relation's target is None). Eq. 5-21 holds on the
    periodic line (it is fluid), once per pair: its row, written at either image, needs the
    neighbor across the line, from the other side of the passage. periodic_ghost(i, j) gives
    that periodic continuation for the two virtual nodes:
        psi[i, -1]     = psi[i, n_eta - 2] - 1        (eta = -deta, the image of 1 - deta)
        psi[i, n_eta]  = psi[i, 1] + 1                (eta = 1 + deta, the image of deta)
    (jump 0 for a periodic scalar). Only on periodic stations: on a blade station the side
    below the lower wall and above the upper wall is the blade, and no periodic continuation
    exists (BoundaryConditionError). That matters at the last inlet station and the first
    exit station: their periodic-line rows reach diagonally into first_blade_i (to j = -1 or
    n_eta) and last_blade_i, beside the nose and the tail. How those two rows are written is
    an open decision of the assembly step, deliberately not settled here.

Corners
    The four outer corners (0, 0), (0, n_eta - 1), (n_m - 1, 0), (n_m - 1, n_eta - 1) lie on a
    far station and on its periodic pair: each keeps both relations, its own meridional one
    (as target) and the periodic jump of its station (shared with its image); nothing is
    discarded or given precedence. The three relations of a far station (meridional at j = 0
    and at j = n_eta - 1, and the jump) are not independent: with P(i) = psi[i, n_eta - 1] -
    psi[i, 0] - 1 the periodic residual of station i,
        U(0, n_eta - 1) - U(0, 0) = -3 P(0) + 4 P(1) - P(2),
        V(N, n_eta - 1) - V(N, 0) = 3 P(N) - 4 P(N - 1) + P(N - 2),
    U and V the upstream and downstream residuals (tested). So once the jumps of the two
    neighboring periodic stations hold, one of the three is implied, and an assembled system
    must carry exactly two of them per far station, or it is singular. The convention for
    the assembly (Aungier 2000): eliminate every far-station node, both images included, by
    its meridional relation; the far stations' periodic jumps then hold by themselves
    (4/3 - 1/3 = 1) and are not added as rows.

Precedence
    Every boundary node is in exactly one of: a wall (its Dirichlet value), a periodic line
    of a station 1 .. n_m - 2 (PDE row with continuation, plus the jump of its station), a far
    station off the corners (its meridional relation), a corner (meridional relation and
    jump, as above). Walls exist only on blade stations and far stations are periodic, so
    no wall condition ever meets a meridional or periodic one: the precedence wall >
    meridional > periodic never has to decide anything, and the corners are the only nodes
    with two relations.

Requirements (BoundaryConditionError otherwise)
    n_eta >= 3; at least three periodic stations on each side of the blade (the one-sided
    relations and the corner identity reach stations 0..2 and N-2..N, which must be periodic);
    1 <= first_blade_i <= last_blade_i <= n_m - 2 and the three topology sources agreeing;
    finite m and eta. A Grid satisfies all but the first two by construction.

Status: implemented 2026-11-12, under review, not frozen.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real

import numpy as np
from numpy.typing import NDArray

from potential_flow_solver.grid import EXIT, INLET, PASSAGE, Grid

Node = tuple[int, int]
Mask = NDArray[np.bool_]

PSI_LOWER_WALL = 0.0            # psi on the lower blade wall (eta = 0)
PSI_UPPER_WALL = 1.0            # psi on the upper blade wall (eta = 1)
PSI_PERIODIC_JUMP = PSI_UPPER_WALL - PSI_LOWER_WALL   # psi(eta = 1) - psi(eta = 0) on a periodic station
FORWARD_DIFFERENCE = (-3.0, 4.0, -1.0)    # x 1/(2 dm) at i = 0, 1, 2: psi_m at the far inlet [Eqs. 5-40 to 5-43]
BACKWARD_DIFFERENCE = (3.0, -4.0, 1.0)    # x 1/(2 dm) at i = N, N-1, N-2: psi_m at the far exit

LOWER_WALL, UPPER_WALL, PERIODIC, UPSTREAM, DOWNSTREAM = "lower wall", "upper wall", "periodic", "upstream", "downstream"
KINDS = (LOWER_WALL, UPPER_WALL, PERIODIC, UPSTREAM, DOWNSTREAM)


class BoundaryConditionError(ValueError):
    """The grid does not support the boundary conditions as defined (module docstring,
    "Requirements"), its topology sources disagree, or a relation is asked for where it does
    not exist (a periodic relation or continuation on a blade station, a far relation off
    the far stations)."""


@dataclass(frozen=True)
class Term:
    """coefficient * psi[node]."""

    node: Node
    coefficient: float


@dataclass(frozen=True)
class Elimination:
    """psi[node] = sum of the terms + constant. node may be virtual (a periodic ghost,
    j = -1 or n_eta)."""

    node: Node
    terms: tuple[Term, ...]
    constant: float

    def value(self, psi: NDArray[np.float64]) -> float:
        """psi[node] from the node values of psi on the right."""
        return float(sum(t.coefficient * psi[t.node] for t in self.terms) + self.constant)


@dataclass(frozen=True)
class Relation:
    """sum of coefficient * psi[node] over the terms = rhs. target is the node the relation
    determines (a wall node, a far-station node), None for the periodic jump, whose image to
    eliminate is the assembly's choice."""

    kind: str
    terms: tuple[Term, ...]
    rhs: float
    target: Node | None

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise BoundaryConditionError(f"unknown relation kind {self.kind!r}")
        nodes = [t.node for t in self.terms]
        if not nodes or len(set(nodes)) != len(nodes):
            raise BoundaryConditionError(f"a {self.kind} relation needs distinct nodes, got {nodes}")
        if not all(math.isfinite(t.coefficient) and t.coefficient != 0.0 for t in self.terms) or not math.isfinite(self.rhs):
            raise BoundaryConditionError(f"a {self.kind} relation needs finite non-zero coefficients and a finite rhs")
        if self.target is not None and self.target not in nodes:
            raise BoundaryConditionError(f"the target {self.target} is not a node of the relation")

    @property
    def nodes(self) -> tuple[Node, ...]:
        return tuple(t.node for t in self.terms)

    def residual(self, psi: NDArray[np.float64]) -> float:
        """sum of coefficient * psi[node] - rhs: zero when psi satisfies the relation."""
        return float(sum(t.coefficient * psi[t.node] for t in self.terms) - self.rhs)

    def solve_for(self, node: Node) -> Elimination:
        """The relation solved for one of its nodes: psi[node] = sum b_k psi[k] + constant."""
        own = [t for t in self.terms if t.node == node]
        if not own:
            raise BoundaryConditionError(f"node {node} is not in this {self.kind} relation ({self.nodes})")
        a = own[0].coefficient
        others = tuple(Term(t.node, -t.coefficient / a) for t in self.terms if t.node != node)
        return Elimination(node=node, terms=others, constant=self.rhs / a + 0.0)


@dataclass(frozen=True)
class BoundaryConditions:
    """The boundary classification and relations of one grid (module docstring). grid is
    kept by reference, as the topology the masks and relations were built from."""

    grid: Grid
    lower_wall: Mask            # blade stations, j = 0: psi = 0
    upper_wall: Mask            # blade stations, j = n_eta - 1: psi = 1
    periodic_lower: Mask        # periodic stations, j = 0
    periodic_upper: Mask        # periodic stations, j = n_eta - 1
    upstream: Mask              # i = 0, every j: psi_m = 0, forward one-sided
    downstream: Mask            # i = n_m - 1, every j: psi_m = 0, backward one-sided
    relations: tuple[Relation, ...]     # lower walls, upper walls, periodic jumps, upstream, downstream

    @property
    def shape(self) -> tuple[int, int]:
        return self.lower_wall.shape

    @property
    def blade_wall(self) -> Mask:
        return _read_only(self.lower_wall | self.upper_wall)

    @property
    def periodic_side(self) -> Mask:
        return _read_only(self.periodic_lower | self.periodic_upper)

    @property
    def meridional_boundary(self) -> Mask:
        return _read_only(self.upstream | self.downstream)

    @property
    def boundary(self) -> Mask:
        """Every node with a boundary relation or a periodic line: all but the interior."""
        return _read_only(self.blade_wall | self.periodic_side | self.meridional_boundary)

    @property
    def interior(self) -> Mask:
        """No boundary relation: the interior nodes of discretization.InteriorStencil."""
        return _read_only(~self.boundary)

    @property
    def periodic_line(self) -> Mask:
        """Periodic-line nodes off the far stations: Eq. 5-21 holds there (one row per pair,
        with periodic_ghost) besides the jump of their station."""
        return _read_only(self.periodic_side & ~self.meridional_boundary)

    @property
    def corners(self) -> tuple[Node, Node, Node, Node]:
        """(0, 0), (0, n_eta - 1), (n_m - 1, 0), (n_m - 1, n_eta - 1): far station and periodic pair."""
        n_m, n_eta = self.shape
        return (0, 0), (0, n_eta - 1), (n_m - 1, 0), (n_m - 1, n_eta - 1)

    @property
    def blade_stations(self) -> NDArray[np.int64]:
        return _read_only(np.flatnonzero(self.lower_wall[:, 0]))

    @property
    def periodic_stations(self) -> NDArray[np.int64]:
        return _read_only(np.flatnonzero(self.periodic_lower[:, 0]))

    def relations_of(self, kind: str) -> tuple[Relation, ...]:
        if kind not in KINDS:
            raise BoundaryConditionError(f"unknown relation kind {kind!r}; the kinds are {KINDS}")
        return tuple(r for r in self.relations if r.kind == kind)

    def relations_at(self, node: Node) -> tuple[Relation, ...]:
        """Every relation in which psi[node] appears (as target or not)."""
        node = (int(node[0]), int(node[1]))
        return tuple(r for r in self.relations if node in r.nodes)

    def upstream_elimination(self, j: int) -> Elimination:
        """psi[0, j] = 4/3 psi[1, j] - 1/3 psi[2, j]: the upstream relation solved for its target."""
        return self._far(self.upstream, 0, j).solve_for((0, int(j)))

    def downstream_elimination(self, j: int) -> Elimination:
        """psi[N, j] = 4/3 psi[N-1, j] - 1/3 psi[N-2, j], N = n_m - 1."""
        n_m = self.shape[0]
        return self._far(self.downstream, n_m - 1, j).solve_for((n_m - 1, int(j)))

    def periodic_relation(self, i: int, jump: float = PSI_PERIODIC_JUMP) -> Relation:
        """f[i, n_eta - 1] - f[i, 0] = jump on periodic station i: 1 for psi (the relation in
        self.relations), 0 for a periodic scalar such as the density."""
        i = self._periodic_station(i)
        return _periodic(i, self.shape[1], _real(jump, "jump"))

    def periodic_ghost(self, i: int, j: int, jump: float = PSI_PERIODIC_JUMP) -> Elimination:
        """The periodic continuation on periodic station i for the virtual node j = -1
        (f[i, -1] = f[i, n_eta - 2] - jump) or j = n_eta (f[i, n_eta] = f[i, 1] + jump)."""
        i = self._periodic_station(i)
        n_eta = self.shape[1]
        jump = _real(jump, "jump")
        if j == -1:
            return Elimination(node=(i, -1), terms=(Term((i, n_eta - 2), 1.0),), constant=-jump)
        if j == n_eta:
            return Elimination(node=(i, n_eta), terms=(Term((i, 1), 1.0),), constant=jump)
        raise BoundaryConditionError(f"a periodic ghost node is j = -1 or j = n_eta = {n_eta}, got j = {j}")

    def _periodic_station(self, i: int) -> int:
        n_m = self.shape[0]
        i = int(i)
        if not 0 <= i < n_m:
            raise BoundaryConditionError(f"station {i} is not one of the {n_m} stations")
        if not self.periodic_lower[i, 0]:
            raise BoundaryConditionError(f"station {i} is a blade station: its eta = 0 and 1 nodes are walls, and below "
                                         f"the lower wall and above the upper wall lies the blade, so there is no "
                                         f"periodic relation or continuation there")
        return i

    def _far(self, mask: Mask, i: int, j: int) -> Relation:
        j = int(j)
        if not 0 <= j < self.shape[1]:
            raise BoundaryConditionError(f"j = {j} is not one of the {self.shape[1]} eta nodes")
        kind = UPSTREAM if i == 0 else DOWNSTREAM
        for relation in self.relations:
            if relation.kind == kind and relation.target == (i, j):
                return relation
        raise BoundaryConditionError(f"no {kind} relation at ({i}, {j})")


def build_boundary_conditions(grid: Grid) -> BoundaryConditions:
    """The boundary classification and relations of a Grid (module docstring). Reads the
    station count, n_eta, first_blade_i, last_blade_i, wall_identity.stations, region, m
    and eta, nothing of the geometry.

    Raises TypeError for anything but a Grid and BoundaryConditionError when the grid does
    not meet the requirements of the module docstring."""
    if not isinstance(grid, Grid):
        raise TypeError(f"build_boundary_conditions takes a Grid, got {type(grid).__name__}")
    return _boundary_conditions(grid, grid.m, grid.eta, grid.first_blade_i, grid.last_blade_i,
                                grid.wall_identity.stations, grid.region)


def _boundary_conditions(grid: Grid, m, eta, first_blade_i: int, last_blade_i: int, stations, region) -> BoundaryConditions:
    """The whole classification and every check of build_boundary_conditions on raw topology
    arrays (a separate function so that the checks can be tried on arrays no valid Grid would
    hold)."""
    m, eta = np.asarray(m, dtype=float), np.asarray(eta, dtype=float)
    if m.ndim != 1 or eta.ndim != 1:
        raise BoundaryConditionError(f"m {m.shape} and eta {eta.shape} must be one-dimensional")
    for name, values in (("m", m), ("eta", eta)):
        if not np.all(np.isfinite(values)):
            raise BoundaryConditionError(f"{name} is not finite at index {int(np.flatnonzero(~np.isfinite(values))[0])}")
    n_m, n_eta = len(m), len(eta)
    if n_eta < 3:
        raise BoundaryConditionError(f"{n_eta} eta nodes: the walls and the periodic continuation need an interior "
                                     f"eta node, n_eta >= 3")
    f, l = int(first_blade_i), int(last_blade_i)
    if not 1 <= f <= l <= n_m - 2:
        raise BoundaryConditionError(f"blade stations {f} to {l} do not fit {n_m} stations with periodic stations on "
                                     f"both sides")
    expected = np.arange(f, l + 1)
    if not np.array_equal(np.asarray(stations, dtype=np.int64), expected):
        raise BoundaryConditionError(f"the wall identity's stations are not first_blade_i .. last_blade_i = {f} .. {l}: "
                                     f"the topology sources disagree")
    expected_region = np.full(n_m, PASSAGE)
    expected_region[:f], expected_region[l + 1:] = INLET, EXIT
    if not np.array_equal(np.asarray(region), expected_region):
        raise BoundaryConditionError("grid.region does not agree with first_blade_i and last_blade_i: the topology "
                                     "sources disagree")
    if f < 3 or n_m - 1 - l < 3:
        side, count = ("inlet", f) if f < 3 else ("exit", n_m - 1 - l)
        raise BoundaryConditionError(f"the {side} extension holds {count} periodic station(s): the second-order one-sided "
                                     f"psi_m = 0 relation reaches three stations, which must all be periodic (lengthen the "
                                     f"extension or raise n_passage); no first-order fallback")

    blade = np.zeros(n_m, dtype=bool)
    blade[f:l + 1] = True
    masks = {name: np.zeros((n_m, n_eta), dtype=bool) for name in
             ("lower_wall", "upper_wall", "periodic_lower", "periodic_upper", "upstream", "downstream")}
    masks["lower_wall"][blade, 0] = masks["upper_wall"][blade, -1] = True
    masks["periodic_lower"][~blade, 0] = masks["periodic_upper"][~blade, -1] = True
    masks["upstream"][0, :] = masks["downstream"][-1, :] = True
    walls = masks["lower_wall"] | masks["upper_wall"]
    if np.any(walls & (masks["periodic_lower"] | masks["periodic_upper"] | masks["upstream"] | masks["downstream"])):
        raise BoundaryConditionError("a wall node is also periodic or on a far station: contradictory classification")
    for mask in masks.values():
        mask.flags.writeable = False

    N = n_m - 1
    relations = [Relation(LOWER_WALL, (Term((i, 0), 1.0),), PSI_LOWER_WALL, (i, 0)) for i in range(f, l + 1)]
    relations += [Relation(UPPER_WALL, (Term((i, n_eta - 1), 1.0),), PSI_UPPER_WALL, (i, n_eta - 1)) for i in range(f, l + 1)]
    relations += [_periodic(i, n_eta, PSI_PERIODIC_JUMP) for i in np.flatnonzero(~blade)]
    relations += [Relation(UPSTREAM, tuple(Term((k, j), c) for k, c in zip((0, 1, 2), FORWARD_DIFFERENCE)), 0.0, (0, j))
                  for j in range(n_eta)]
    relations += [Relation(DOWNSTREAM, tuple(Term((k, j), c) for k, c in zip((N, N - 1, N - 2), BACKWARD_DIFFERENCE)), 0.0,
                           (N, j)) for j in range(n_eta)]
    return BoundaryConditions(grid=grid, relations=tuple(relations), **masks)


def _periodic(i: int, n_eta: int, jump: float) -> Relation:
    return Relation(PERIODIC, (Term((int(i), n_eta - 1), 1.0), Term((int(i), 0), -1.0)), jump, None)


def _real(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
        raise BoundaryConditionError(f"{name} must be a finite real number, got {value!r}")
    return float(value)


def _read_only(array: NDArray) -> NDArray:
    array.flags.writeable = False
    return array
