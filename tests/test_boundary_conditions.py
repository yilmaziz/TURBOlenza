"""
Tests for turbolenza.b2b.boundary_conditions: the classification of the boundary
nodes from the grid topology and the explicit relations of Aungier's boundary conditions
(blade walls psi = 0 / 1, periodic jump psi_upper - psi_lower = 1, psi_m = 0 at the far
inlet and exit by second-order one-sided differences), with their eliminations and the
periodic continuation. No matrix, no substitution, no solve.

Every reference is independent of boundary_conditions.py: the station classification read
from the nodes themselves (S = y1 - y0 is the pitch exactly on the periodic stations and
less on the blade stations), analytic fields with known values and derivatives, and the
one-sided formulas and their eliminations written out here. Each tolerance names the
measured value it covers.

Checks
The module
- the public names and constants; discretization still imports BoundaryConditions; the
  module imports numpy and grid only, reads the grid's topology and coordinates and nothing
  of the geometry, and has no stencil weight, matrix, solve or Kutta name
Classification
- the masks are read-only and cover every node once: interior (= InteriorStencil.interior),
  wall, periodic line, far station off the corners, corner; no wall node is periodic or on
  a far station
- the blade stations are those of the nodes (S < pitch): first_blade_i - 1 .. + 1 and
  last_blade_i - 1 .. + 1 are periodic, wall, wall and wall, wall, periodic, on several
  grids (a one-station shift would fail); the topology does not change with rebuild_exit,
  the representation (C1, C2) or a physical chord
Walls
- psi = 0 and psi = 1 exactly on every blade station (first and last named), one relation
  each, and none elsewhere
Periodic
- psi[i, n_eta - 1] - psi[i, 0] = 1 on every periodic station and on none of the blade
  stations; a periodic-with-jump field psi = g(m) + eta + q (q periodic) satisfies it to
  round-off, a field without the jump does not; jump 0 for a periodic scalar; the periodic
  continuation (ghost nodes) reproduces the field at eta = -deta and 1 + deta; refused on
  blade stations
Far stations
- the relations are Aungier's one-sided formulas (-3, 4, -1) and (3, -4, 1) times 2 dm;
  fields with psi_m = 0 at the far station satisfy them; exact for quadratics as a
  derivative formula, second order on a smooth function, the same with any dm (c_ref);
  the eliminations are 4/3 psi_1 - 1/3 psi_2, written out here, and substituting them back
  leaves round-off
Corners
- each corner holds exactly its far relation (as target) and its station's periodic jump;
  the dependency U(0, n-1) - U(0, 0) = -3 P(0) + 4 P(1) - P(2) (and downstream); eliminating
  both images of a far station keeps the jump
Errors
- anything but a Grid; n_eta = 2; fewer than three periodic stations on a side (no
  first-order fallback); disagreeing topology sources; non-finite coordinates; relations and
  eliminations asked for where they do not exist; malformed relations
"""

import ast
import math
import sys
from dataclasses import FrozenInstanceError, replace
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from turbolenza.b2b import boundary_conditions as bc
from turbolenza.b2b import coefficients as cf
from turbolenza.b2b import discretization as dz
from turbolenza.b2b import metrics as mt
from turbolenza.b2b.config import GridSettings
from turbolenza.b2b.geometry import CascadeSpec, make_cascade
from turbolenza.b2b.grid import make_grid, rebuild_exit

SPEC = CascadeSpec(blade_type="NACA65", solidity=1.0, stagger_deg=33.0, cl0=1.2, t_max_over_c=0.10,
                   surface_continuity="C2")
C1_SPEC = replace(SPEC, surface_continuity="C1")
INLET = math.radians(45.0)
LEVELS = ((31, 5), (51, 11), (101, 31), (201, 61), (401, 121))


@lru_cache(maxsize=None)
def cascade_of(spec: CascadeSpec):
    return make_cascade(spec)


@lru_cache(maxsize=None)
def grid_of(n_passage: int = 101, n_eta: int = 31, spec: CascadeSpec = SPEC, up: float = 1.0, down: float = 1.0):
    settings = GridSettings(n_passage=n_passage, n_eta=n_eta, upstream_extension_chords=up,
                            downstream_extension_chords=down)
    return make_grid(cascade_of(spec), settings, INLET)


@lru_cache(maxsize=None)
def bcs_of(*args) -> bc.BoundaryConditions:
    return bc.build_boundary_conditions(grid_of(*args))


def node_blade_stations(g) -> np.ndarray:
    """The blade stations read from the nodes: S = y1 - y0 is the pitch on a periodic
    station (to round-off, 1e-16 chord) and the pitch less the blade's cut (> TOUCH = 1e-12
    chord) on a blade station."""
    S = g.nodes[:, -1, 1] - g.nodes[:, 0, 1]
    return np.flatnonzero(S < g.pitch - 1e-12 * g.chord)


def field(g, values) -> np.ndarray:
    M, H = np.meshgrid(g.m, g.eta, indexing="ij")
    return values(M, H)


# --- the module -------------------------------------------------------------------------------

def test_the_public_names_and_constants():
    tree = ast.parse(Path(bc.__file__).read_text(encoding="utf-8"))
    functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef) and not node.name.startswith("_")}
    classes = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
    assert functions == {"build_boundary_conditions"}
    assert classes == {"BoundaryConditionError", "Term", "Elimination", "Relation", "BoundaryConditions"}
    assert issubclass(bc.BoundaryConditionError, ValueError)
    assert (bc.PSI_LOWER_WALL, bc.PSI_UPPER_WALL, bc.PSI_PERIODIC_JUMP) == (0.0, 1.0, 1.0)
    assert bc.FORWARD_DIFFERENCE == (-3.0, 4.0, -1.0) and bc.BACKWARD_DIFFERENCE == (3.0, -4.0, 1.0)
    assert bc.KINDS == ("lower wall", "upper wall", "periodic", "upstream", "downstream")
    assert dz.BoundaryConditions is bc.BoundaryConditions


def test_the_module_reads_the_grid_topology_only():
    """Imports numpy and grid.py only; of the grid it reads the coordinates and the topology
    (first/last blade station, wall identity stations, region), nothing of the geometry or
    the nodes; no stencil weight, no matrix, no solve, no Kutta."""
    tree = ast.parse(Path(bc.__file__).read_text(encoding="utf-8"))
    imported = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
    imported |= {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    assert {name for name in imported if name.split(".")[0] not in sys.stdlib_module_names} == {
        "numpy", "numpy.typing", "turbolenza.b2b.grid"}
    on = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            on.setdefault(node.value.id, set()).add(node.attr)
    assert on["grid"] <= {"m", "eta", "first_blade_i", "last_blade_i", "wall_identity", "region"}
    attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert not {"boundaries", "blade_curves", "m_le", "m_te", "west", "east", "north", "south", "mixed",
                "csr_matrix", "lil_matrix", "spsolve", "solve", "surface"} & attributes
    assert not {"kutta", "rho", "beta_in", "beta_out", "csr_matrix", "InteriorStencil"} & names


def test_the_object_is_frozen_and_its_masks_read_only():
    b = bcs_of()
    for name in ("lower_wall", "upper_wall", "periodic_lower", "periodic_upper", "upstream", "downstream", "blade_wall",
                 "periodic_side", "meridional_boundary", "boundary", "interior", "periodic_line", "blade_stations",
                 "periodic_stations"):
        assert not getattr(b, name).flags.writeable, name
    with pytest.raises(FrozenInstanceError):
        b.upstream = b.downstream
    assert b.grid is grid_of() and b.shape == (len(grid_of().m), len(grid_of().eta))


# --- classification ---------------------------------------------------------------------------

@pytest.mark.parametrize("level", LEVELS)
def test_every_node_has_one_class(level):
    """interior, wall, periodic line (stations 1 .. n_m - 2), far station off the corners, or
    corner: exactly one each. The interior is InteriorStencil.interior, built from the same
    grid's coefficients."""
    b = bcs_of(*level)
    n_m, n_eta = b.shape
    corner = np.zeros(b.shape, dtype=bool)
    for node in b.corners:
        corner[node] = True
    classes = (b.interior, b.blade_wall, b.periodic_line, b.meridional_boundary & ~corner, corner)
    np.testing.assert_array_equal(sum(c.astype(int) for c in classes), np.ones(b.shape, dtype=int))
    assert not np.any(b.blade_wall & (b.periodic_side | b.meridional_boundary))
    assert not np.any(b.lower_wall & b.upper_wall) and not np.any(b.periodic_lower & b.periodic_upper)
    metrics = mt.build_metrics(grid_of(*level))
    stencil = dz.build_interior_stencil(cf.build_coefficients(metrics, np.ones(b.shape), 1.0), metrics)
    np.testing.assert_array_equal(b.interior, stencil.interior)
    assert sorted(b.corners) == sorted({(0, 0), (0, n_eta - 1), (n_m - 1, 0), (n_m - 1, n_eta - 1)})


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("spec", (SPEC, C1_SPEC))
def test_the_blade_stations_are_those_of_the_nodes(level, spec):
    """The stations with walls are exactly the stations whose nodes span less than a pitch;
    around the transitions the classes run periodic, wall, wall and wall, wall, periodic."""
    g = grid_of(*level, spec)
    b = bc.build_boundary_conditions(g)
    np.testing.assert_array_equal(b.blade_stations, node_blade_stations(g))
    np.testing.assert_array_equal(b.periodic_stations, np.setdiff1d(np.arange(len(g.m)), node_blade_stations(g)))
    f, l = g.first_blade_i, g.last_blade_i
    n = len(g.eta) - 1
    expected = {f - 1: "periodic", f: "wall", f + 1: "wall", l - 1: "wall", l: "wall", l + 1: "periodic"}
    for i, kind in expected.items():
        if kind == "wall":
            assert b.lower_wall[i, 0] and b.upper_wall[i, n] and not b.periodic_side[i].any(), i
        else:
            assert b.periodic_lower[i, 0] and b.periodic_upper[i, n] and not b.blade_wall[i].any(), i


def test_a_one_station_shift_of_the_topology_is_refused():
    """The three topology sources must agree; each shifted by one station is refused."""
    g = grid_of()
    f, l, stations, region = g.first_blade_i, g.last_blade_i, g.wall_identity.stations, g.region
    bc._boundary_conditions(g, g.m, g.eta, f, l, stations, region)
    for args in ((f + 1, l, stations, region), (f, l - 1, stations, region), (f, l, stations + 1, region),
                 (f, l, stations, np.roll(region, 1))):
        with pytest.raises(bc.BoundaryConditionError, match="disagree"):
            bc._boundary_conditions(g, g.m, g.eta, *args)


def test_the_topology_alone_decides():
    """rebuild_exit turns the exit extension but keeps the blade stations: the same masks and
    relations. A physical-chord cascade (lengths in metres) has the same topology: the same
    relations, no length enters them."""
    b = bcs_of()
    turned = bc.build_boundary_conditions(rebuild_exit(grid_of(), cascade_of(SPEC).kappa_te + math.radians(6.0)))
    metres = bc.build_boundary_conditions(make_grid(cascade_of(replace(SPEC, physical_chord=0.1)),
                                                    GridSettings(n_passage=101, n_eta=31), INLET))
    for other in (turned, metres):
        for name in ("lower_wall", "upper_wall", "periodic_lower", "periodic_upper", "upstream", "downstream"):
            np.testing.assert_array_equal(getattr(other, name), getattr(b, name))
        assert other.relations == b.relations


# --- walls ------------------------------------------------------------------------------------

@pytest.mark.parametrize("level", LEVELS)
def test_blade_walls_are_zero_and_one(level):
    """One relation psi[i, 0] = 0 and one psi[i, n_eta - 1] = 1 per blade station, exactly (no
    tolerance), first_blade_i, last_blade_i and stations between included; no wall relation
    elsewhere."""
    g = grid_of(*level)
    b = bcs_of(*level)
    f, l, n = g.first_blade_i, g.last_blade_i, len(g.eta) - 1
    lower, upper = b.relations_of(bc.LOWER_WALL), b.relations_of(bc.UPPER_WALL)
    assert [r.target for r in lower] == [(i, 0) for i in range(f, l + 1)]
    assert [r.target for r in upper] == [(i, n) for i in range(f, l + 1)]
    for r in lower:
        assert r.terms == (bc.Term(r.target, 1.0),) and r.rhs == 0.0
    for r in upper:
        assert r.terms == (bc.Term(r.target, 1.0),) and r.rhs == 1.0
    for i in (f, f + 1, (f + l) // 2, l - 1, l):
        assert [(r.kind, r.rhs) for r in b.relations_at((i, 0))] == [(bc.LOWER_WALL, 0.0)]
        assert [(r.kind, r.rhs) for r in b.relations_at((i, n))] == [(bc.UPPER_WALL, 1.0)]
    psi = field(g, lambda M, H: H + 0.3 * np.sin(M) * H * (1.0 - H))    # exactly 0 at eta = 0, 1 at eta = 1
    assert all(r.residual(psi) == 0.0 for r in lower + upper)


# --- periodic ---------------------------------------------------------------------------------

def periodic_field(g, jump: float = 1.0):
    """psi = g(m) + jump eta + q(m, eta), q periodic in eta with period 1, and its analytic
    value anywhere (for the ghost nodes)."""
    def value(M, H):
        return 0.4 * np.sin(1.3 * M) + jump * H + 0.1 * np.cos(M) * np.sin(2.0 * math.pi * H + 0.3)
    return field(g, value), value


@pytest.mark.parametrize("level", LEVELS)
def test_the_periodic_jump(level):
    """psi[i, n_eta - 1] - psi[i, 0] = 1 on every periodic station, as (1, -1) terms with rhs
    1 and no target; none on a blade station. The periodic-with-jump field satisfies it to
    round-off (measured 2.2e-16), a field without the jump misses it by 1."""
    g = grid_of(*level)
    b = bcs_of(*level)
    n = len(g.eta) - 1
    periodic = b.relations_of(bc.PERIODIC)
    stations = [r.terms[0].node[0] for r in periodic]
    assert stations == list(np.setdiff1d(np.arange(len(g.m)), node_blade_stations(g)))
    for r in periodic:
        i = r.terms[0].node[0]
        assert r.terms == (bc.Term((i, n), 1.0), bc.Term((i, 0), -1.0)) and r.rhs == 1.0 and r.target is None
    psi, _ = periodic_field(g)
    assert max(abs(r.residual(psi)) for r in periodic) <= 4.5e-16
    flat, _ = periodic_field(g, jump=0.0)
    assert all(abs(r.residual(flat) + 1.0) <= 4.5e-16 for r in periodic)
    for i in b.blade_stations:
        assert not any(r.kind == bc.PERIODIC for r in b.relations_at((int(i), 0)))
        with pytest.raises(bc.BoundaryConditionError, match="blade station"):
            b.periodic_relation(int(i))


def test_solving_the_periodic_and_wall_relations():
    """Either image can be eliminated (the assembly's choice): solved for the upper image the
    jump gives psi[i, n - 1] = psi[i, 0] + 1, for the lower psi[i, 0] = psi[i, n - 1] - 1, both
    exactly, and each reproduces the periodic-with-jump field's node value to round-off
    (measured 2.2e-16). A wall relation solved for its node is its value."""
    g = grid_of()
    b = bcs_of()
    n = len(g.eta) - 1
    psi, _ = periodic_field(g)
    for relation in b.relations_of(bc.PERIODIC):
        i = relation.terms[0].node[0]
        upper, lower = relation.solve_for((i, n)), relation.solve_for((i, 0))
        assert upper == bc.Elimination((i, n), (bc.Term((i, 0), 1.0),), 1.0)
        assert lower == bc.Elimination((i, 0), (bc.Term((i, n), 1.0),), -1.0)
        assert abs(upper.value(psi) - psi[i, n]) <= 4.5e-16 and abs(lower.value(psi) - psi[i, 0]) <= 4.5e-16
    for relation in b.relations_of(bc.UPPER_WALL) + b.relations_of(bc.LOWER_WALL):
        assert relation.solve_for(relation.target) == bc.Elimination(relation.target, (), relation.rhs)


def test_a_periodic_scalar_has_jump_zero():
    g = grid_of()
    b = bcs_of()
    rho, _ = periodic_field(g, jump=0.0)
    for i in b.periodic_stations:
        relation = b.periodic_relation(int(i), jump=0.0)
        assert relation.rhs == 0.0 and abs(relation.residual(rho)) <= 4.5e-16
    assert b.periodic_relation(5) == b.relations_of(bc.PERIODIC)[5]


@pytest.mark.parametrize("level", ((31, 5), (101, 31)))
def test_the_periodic_continuation(level):
    """The ghost nodes of a periodic station: psi[i, -1] = psi[i, n_eta - 2] - 1 and psi[i, n_eta]
    = psi[i, 1] + 1 reproduce the analytic periodic-with-jump field at eta = -deta and
    1 + deta (measured 1.9e-16 and 4.4e-16); with jump 0 the periodic scalar. Refused on every blade
    station, and for any other j."""
    g = grid_of(*level)
    b = bcs_of(*level)
    n_eta = len(g.eta)
    for jump in (1.0, 0.0):
        psi, value = periodic_field(g, jump)
        for i in b.periodic_stations:
            i = int(i)
            below, above = b.periodic_ghost(i, -1, jump), b.periodic_ghost(i, n_eta, jump)
            assert below.node == (i, -1) and above.node == (i, n_eta)
            assert abs(below.value(psi) - value(g.m[i], -g.deta)) <= 4.5e-16 * 2
            assert abs(above.value(psi) - value(g.m[i], 1.0 + g.deta)) <= 4.5e-16 * 4
    for i in b.blade_stations[[0, -1]]:
        with pytest.raises(bc.BoundaryConditionError, match="blade station"):
            b.periodic_ghost(int(i), -1)
    with pytest.raises(bc.BoundaryConditionError, match="ghost"):
        b.periodic_ghost(0, 1)


# --- far stations -----------------------------------------------------------------------------

@pytest.mark.parametrize("level", LEVELS)
def test_the_far_relations_are_the_one_sided_formulas(level):
    """Upstream: -3 psi[0, j] + 4 psi[1, j] - psi[2, j] = 0, target (0, j); downstream:
    3 psi[N, j] - 4 psi[N - 1, j] + psi[N - 2, j] = 0, target (N, j); every j, corners included."""
    g = grid_of(*level)
    b = bcs_of(*level)
    N = len(g.m) - 1
    up, down = b.relations_of(bc.UPSTREAM), b.relations_of(bc.DOWNSTREAM)
    assert len(up) == len(down) == len(g.eta)
    for j, (u, d) in enumerate(zip(up, down)):
        assert u.terms == (bc.Term((0, j), -3.0), bc.Term((1, j), 4.0), bc.Term((2, j), -1.0))
        assert d.terms == (bc.Term((N, j), 3.0), bc.Term((N - 1, j), -4.0), bc.Term((N - 2, j), 1.0))
        assert u.rhs == d.rhs == 0.0 and u.target == (0, j) and d.target == (N, j)


def test_fields_with_zero_psi_m_at_the_far_stations_satisfy_them():
    """psi = eta + (m - m_in)^2 f(eta) has psi_m = 0 at the inlet, psi = eta + (m - m_out)^2 f(eta)
    at the exit, and the one-sided formula is exact for quadratics: measured residuals 3.3e-16
    (upstream and downstream)."""
    g = grid_of()
    b = bcs_of()
    up = field(g, lambda M, H: H + (M - g.m[0]) ** 2 * (1.0 + np.sin(3.0 * H)))
    down = field(g, lambda M, H: H + (M - g.m[-1]) ** 2 * (1.0 + np.sin(3.0 * H)))
    assert max(abs(r.residual(up)) for r in b.relations_of(bc.UPSTREAM)) <= 1e-15
    assert max(abs(r.residual(down)) for r in b.relations_of(bc.DOWNSTREAM)) <= 1e-15


def derivative_by(relation: bc.Relation, values, h: float) -> float:
    """The relation read as its derivative formula: sum of coefficient * f(m_node) / (2 h),
    with the node's station at m = i h measured from the far station."""
    far = relation.target[0]
    return sum(t.coefficient * values((t.node[0] - far) * h) for t in relation.terms) / (2.0 * h)


@pytest.mark.parametrize("h", (0.0083, 0.1 * 0.0083, 2.7))
def test_the_one_sided_formulas_are_exact_for_quadratics(h):
    """f = a + b x + c x^2 about the far station (x = m - m_far): the relation over 2 h is
    f'(0) = b exactly, for any h, so for any c_ref (dm = c_ref grid.dm: the factor 2 dm of the
    homogeneous condition cancels). The round-off is that of the three-term sum over 2 h:
    measured at most 6.9e-17 of sum |c f| / (2 h)."""
    b = bcs_of()
    up, down = b.relations_of(bc.UPSTREAM)[3], b.relations_of(bc.DOWNSTREAM)[3]
    for a, slope, c in ((1.7, 0.0, 0.0), (0.0, -2.3, 0.0), (0.4, 1.1, 5.0), (0.0, 0.0, -3.0)):
        def poly(x):
            return a + slope * x + c * x * x
        for relation in (up, down):
            size = sum(abs(t.coefficient * poly((t.node[0] - relation.target[0]) * h)) for t in relation.terms) / (2.0 * h)
            assert abs(derivative_by(relation, poly, h) - slope) <= 4.5e-16 * size


def test_the_one_sided_formulas_converge_at_second_order():
    """f = sin(2 x + 0.3) about the far station: the error of the forward (upstream) and backward
    (downstream) formulas against f'(0) = 2 cos(0.3) falls at second order, h = 0.1 / 2^k,
    k = 0 .. 5. Measured orders: forward 1.950, 1.979, 1.991, 1.996, 1.998; backward 2.018,
    2.013, 2.007, 2.004, 2.002."""
    b = bcs_of()
    up, down = b.relations_of(bc.UPSTREAM)[0], b.relations_of(bc.DOWNSTREAM)[-1]
    exact = 2.0 * math.cos(0.3)
    for relation in (up, down):
        steps = [0.1 / 2 ** k for k in range(6)]
        errors = [abs(derivative_by(relation, lambda x: math.sin(2.0 * x + 0.3), h) - exact) for h in steps]
        orders = [math.log(errors[k] / errors[k + 1]) / math.log(2.0) for k in range(len(errors) - 1)]
        assert min(orders) >= 1.94 and max(orders) <= 2.03 and abs(orders[-1] - 2.0) <= 0.01, orders


def test_the_eliminations_are_aungiers_and_substitute_back():
    """Solving Aungier's one-sided relation for the far node, written here: psi_0 = (4 psi_1 -
    psi_2) / 3 and psi_N = (4 psi_{N-1} - psi_{N-2}) / 3. The module's eliminations carry the
    coefficients 4/3 and -1/3 exactly; on random values they agree with the formula here to
    1.8e-16 relative (measured), and substituting them back into the relation leaves 1.1e-16
    of sum |c psi| (measured)."""
    b = bcs_of()
    N, n_eta = b.shape[0] - 1, b.shape[1]
    rng = np.random.default_rng(11)
    psi = rng.uniform(-3.0, 3.0, b.shape)
    for j in range(n_eta):
        for elimination, relation, near, far in ((b.upstream_elimination(j), b.relations_of(bc.UPSTREAM)[j], 1, 2),
                                                 (b.downstream_elimination(j), b.relations_of(bc.DOWNSTREAM)[j],
                                                  N - 1, N - 2)):
            assert elimination.terms == (bc.Term((near, j), 4.0 / 3.0), bc.Term((far, j), -1.0 / 3.0))
            assert elimination.constant == 0.0 and not math.copysign(1.0, elimination.constant) < 0.0
            reference = (4.0 * psi[near, j] - psi[far, j]) / 3.0
            assert abs(elimination.value(psi) - reference) <= 4.5e-16 * max(1.0, abs(reference))
            substituted = psi.copy()
            substituted[elimination.node] = elimination.value(psi)
            size = sum(abs(t.coefficient * substituted[t.node]) for t in relation.terms)
            assert abs(relation.residual(substituted)) <= 4.5e-16 * size


# --- corners ----------------------------------------------------------------------------------

def test_each_corner_holds_its_far_relation_and_its_periodic_jump():
    b = bcs_of()
    N, n = b.shape[0] - 1, b.shape[1] - 1
    expected = {(0, 0): bc.UPSTREAM, (0, n): bc.UPSTREAM, (N, 0): bc.DOWNSTREAM, (N, n): bc.DOWNSTREAM}
    for corner in b.corners:
        relations = b.relations_at(corner)
        assert sorted(r.kind for r in relations) == sorted([expected[corner], bc.PERIODIC])
        far = next(r for r in relations if r.kind != bc.PERIODIC)
        jump = next(r for r in relations if r.kind == bc.PERIODIC)
        assert far.target == corner and jump.target is None and corner in jump.nodes
        assert b.upstream[corner] or b.downstream[corner]
        assert b.periodic_lower[corner] or b.periodic_upper[corner]


def test_the_far_relations_and_the_jumps_are_dependent():
    """With P(i) the periodic residual of station i, U(0, n) - U(0, 0) = -3 P(0) + 4 P(1) - P(2)
    and V(N, n) - V(N, 0) = 3 P(N) - 4 P(N - 1) + P(N - 2), for any psi (measured 3.6e-15 on
    random normal values, nine terms): one of the three relations of a far station is implied
    by the others and the neighbors' jumps."""
    b = bcs_of()
    N, n = b.shape[0] - 1, b.shape[1] - 1
    rng = np.random.default_rng(12)
    for _ in range(5):
        psi = rng.normal(size=b.shape)
        P = {i: b.periodic_relation(i).residual(psi) for i in (0, 1, 2, N - 2, N - 1, N)}
        U = [r.residual(psi) for r in b.relations_of(bc.UPSTREAM)]
        V = [r.residual(psi) for r in b.relations_of(bc.DOWNSTREAM)]
        assert abs((U[n] - U[0]) - (-3.0 * P[0] + 4.0 * P[1] - P[2])) <= 8e-15
        assert abs((V[n] - V[0]) - (3.0 * P[N] - 4.0 * P[N - 1] + P[N - 2])) <= 8e-15


def test_eliminating_both_far_images_keeps_the_jump():
    """The Aungier 2000 convention: both far images eliminated by their meridional relation.
    If the jumps of stations 1, 2 (N - 1, N - 2) hold, the eliminated far values satisfy the
    jump of station 0 (N) to round-off (measured 2.2e-16), because 4/3 - 1/3 = 1."""
    g = grid_of()
    b = bcs_of()
    N, n = b.shape[0] - 1, b.shape[1] - 1
    psi, _ = periodic_field(g)
    psi[0, :] = psi[N, :] = np.nan                       # the far values are unknowns
    rebuilt = psi.copy()
    for j in range(n + 1):
        rebuilt[0, j] = b.upstream_elimination(j).value(psi)
        rebuilt[N, j] = b.downstream_elimination(j).value(psi)
    for i in (0, N):
        assert abs(b.periodic_relation(i).residual(rebuilt)) <= 4.5e-16


# --- errors -----------------------------------------------------------------------------------

def test_only_a_grid_is_accepted():
    metrics = mt.build_metrics(grid_of(31, 5))
    for item in (metrics, None, grid_of(31, 5).boundaries):
        with pytest.raises(TypeError, match="Grid"):
            bc.build_boundary_conditions(item)


def test_small_grids_are_refused():
    """n_eta = 2 has no interior eta node; an extension of fewer than three periodic stations
    cannot hold the second-order one-sided relation (measured: 1 and 2 inlet stations at
    0.02 and 0.05 chords of extension, 2 exit stations at 0.03 chords). No first-order
    fallback."""
    with pytest.raises(bc.BoundaryConditionError, match="n_eta >= 3"):
        bc.build_boundary_conditions(grid_of(31, 2))
    for up, down, side in ((0.02, 1.0, "inlet"), (0.05, 0.05, "inlet"), (1.0, 0.03, "exit")):
        g = grid_of(31, 5, SPEC, up, down)
        assert min(g.first_blade_i, len(g.m) - 1 - g.last_blade_i) < 3
        with pytest.raises(bc.BoundaryConditionError, match=f"the {side} extension holds"):
            bc.build_boundary_conditions(g)
    g = grid_of(31, 5, SPEC, 0.06, 1.0)
    assert g.first_blade_i == 3
    bc.build_boundary_conditions(g)


def test_broken_coordinates_and_topology_are_refused():
    g = grid_of(31, 5)
    f, l, stations, region = g.first_blade_i, g.last_blade_i, g.wall_identity.stations, g.region
    for name, value in (("m", np.where(np.arange(len(g.m)) == 4, np.nan, g.m)), ("eta", np.r_[g.eta[:-1], np.inf])):
        args = (value, g.eta) if name == "m" else (g.m, value)
        with pytest.raises(bc.BoundaryConditionError, match=f"{name} is not finite"):
            bc._boundary_conditions(g, *args, f, l, stations, region)
    with pytest.raises(bc.BoundaryConditionError, match="one-dimensional"):
        bc._boundary_conditions(g, g.m[:, None], g.eta, f, l, stations, region)
    for first, last in ((0, l), (f, len(g.m) - 1), (l, f)):
        with pytest.raises(bc.BoundaryConditionError, match="do not fit"):
            bc._boundary_conditions(g, g.m, g.eta, first, last, stations, region)


def test_relations_are_only_where_they_exist():
    b = bcs_of(31, 5)
    for call in (lambda: b.upstream_elimination(5), lambda: b.downstream_elimination(-1),
                 lambda: b.periodic_relation(len(b.periodic_stations) + 500), lambda: b.relations_of("wall"),
                 lambda: b.periodic_relation(3, jump=math.nan), lambda: b.periodic_relation(3, jump=True)):
        with pytest.raises(bc.BoundaryConditionError):
            call()
    relation = b.relations_of(bc.UPSTREAM)[0]
    with pytest.raises(bc.BoundaryConditionError, match="not in this"):
        relation.solve_for((5, 0))
    for kind, terms, target in (("wall", (bc.Term((0, 0), 1.0),), None),
                                (bc.PERIODIC, (bc.Term((0, 0), 1.0), bc.Term((0, 0), -1.0)), None),
                                (bc.PERIODIC, (bc.Term((0, 0), 0.0), bc.Term((0, 4), 1.0)), None),
                                (bc.LOWER_WALL, (bc.Term((3, 0), 1.0),), (4, 0)),
                                (bc.LOWER_WALL, (), None)):
        with pytest.raises(bc.BoundaryConditionError):
            bc.Relation(kind, terms, 0.0, target)
