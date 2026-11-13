"""
Tests for potential_flow_solver.discretization: the normalized nine-point difference
equation of Aungier Eq. 5-44 (weights Eqs. 5-45 to 5-50) at the interior nodes, from the
PDE coefficients A to E of Eq. 5-21 and the grid spacings. The boundary relations have
their own file, tests/test_boundary_conditions.py; the assembly and the solve are later
steps.

Every reference is written here, independently of discretization.py: the weights of
Eqs. 5-45 to 5-50 as formulas; the raw discretization, Eqs. 5-35 to 5-39 substituted into
Eq. 5-21 term by term (never through the module's weights); the exact operator of an
analytic psi; and a conservative (face-flux) discretization of Eq. 5-20. Each tolerance
names the measured value it covers.

Checks
Formulas (random coefficient fields and spacings, through the array core and through the
public API on the default C2 grid)
- Eqs. 5-45 to 5-49 (west, east, south, north, mixed), each against its formula; center 1;
  Eq. 5-50: rhs 0 at every interior node
- the nine weights against the raw nine-point weights divided by -H, through the named
  arrays, the diagonal properties and the (3, 3) weights layout
- raw-vs-normalized: for random psi, the normalized residual is -(raw residual) / H
- the mixed derivative alone (A = C = 1, D = E = 0, B != 0, psi = m eta): exact, with the
  -2 B sign; every one of the 15 sign flips of the four diagonal weights is caught
- polynomial exactness: psi = 1, m, eta, m^2, eta^2, m eta give the exact operator
Convergence (smooth manufactured coefficients and psi)
- the discrete operator converges at second order to the exact one: both directions
  refined, dm only, deta only, and the mixed derivative alone in each direction
- the expanded form and a conservative discretization of Eq. 5-20 approach each other at
  second order when D and E are the derivatives the expansion asks for
Lengths and scaling (the default C2 grid, coefficients from coefficients.py)
- dm = coefficients.length_scale * grid.dm, deta = grid.deta
- normalized mode: c_ref = None and c_ref = 1 give the same stencil bit for bit
- fixed mdot', physically similar twins and a change of mdot': the weights and the
  normalized residual of one psi(m*, eta) unchanged, H ~ 1/L, L^0 and mdot'
The default C2 case
- n_passage 101 (n_eta 31) and 401 (121): finite weights and H > 0 at every interior node,
  NaN elsewhere; row properties (west + east + south + north = -1, |mixed| <= |sin beta| / 8);
  deterministic; the sign of every weight follows its coefficients (no sign flip of the
  implementation, on the curvature steps too); positive south / north weights only at the
  corner stations of the region transitions, finite there
Inputs
- refused: coefficient fields of other metrics (rebuild_exit, another grid, another length
  scale); wrong types; shape mismatch; NaN or Inf fields; spacings that are zero, negative,
  NaN, Inf or not real; H zero, negative or overflowing; fewer than three nodes a side;
  direct InteriorStencil construction with non-NaN values outside the interior
- the module imports no solver and builds no sparse matrix; the boundary and assembly
  interfaces still raise NotImplementedError
The periodic PDE line (build_periodic_pde_rows)
- InteriorStencil unchanged: the weight sums of four cases from before the periodic rows
- one row per periodic station 1 .. first_blade_i - 1, last_blade_i + 1 .. n_m - 2, at
  (i, n_eta - 1); LE / TE transition kinds exactly at first_blade_i - 1 / last_blade_i + 1
- standard and transition rows against Eq. 5-21 at (i, J) derived here with the continuation
  psi[k, J + 1] = psi[k, 1] + 1 and the central / backward (3, -4, 1) / forward (-3, 4, -1)
  mixed difference, divided by -H
- the rhs is -north (the jump through psi_eta, psi_eta,eta), non-zero; the mixed constants
  cancel (sum c = 0)
- footprints: the LE row reaches station first_blade_i only at the wall node (f, J), with the
  east weight; the TE row reads no node of station last_blade_i at all (its psi_m and
  psi_m,m are forward one-sided, l + 1 .. l + 4); the mixed term stays at f - 3 .. f - 1 and
  l + 1 .. l + 3; a central mixed difference there is refused by periodic_ghost
- the TE row's psi_m and psi_m,m, read through the row itself: exact for quadratics and
  cubics, second order on a smooth function
- geometry (an independent polyline-intersection check of the C2 blade and its pitch
  copies): every link of the LE and TE rows stays in the fluid on five default grids, while
  the old central TE link (l + 1, J) -> (l, J) runs through the +pitch tail
- the LE and standard rows unchanged by the TE revision (values from before it)
- second order (whole operator and mixed term alone, LE, TE, standard) on a quasi-periodic
  psi; exact for psi = eta + p(m) s(eta) (mixed) and psi = eta + p(m) (rows), p quadratic
- the default C2 grid: finite, deterministic; dimensionless under the scalings, normalized
  mode = c_ref 1; inputs and malformed rows refused
"""

import ast
import itertools
import math
import sys
from dataclasses import FrozenInstanceError, fields, replace
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from potential_flow_solver import boundary_conditions as bc
from potential_flow_solver import coefficients as cf
from potential_flow_solver import discretization as dz
from potential_flow_solver import grid as gr
from potential_flow_solver import metrics as mt
from potential_flow_solver.config import GridSettings
from potential_flow_solver.geometry import CascadeSpec, make_cascade
from potential_flow_solver.grid import make_grid, rebuild_exit

SPEC = CascadeSpec(blade_type="NACA65", solidity=1.0, stagger_deg=33.0, cl0=1.2, t_max_over_c=0.10,
                   surface_continuity="C2")
INLET = math.radians(45.0)
LEVELS = (16, 32, 64, 128)          # cells per direction of the manufactured lattices
M_LENGTH = 1.2                      # m extent of the manufactured lattices


@lru_cache(maxsize=None)
def cascade_of(spec: CascadeSpec):
    return make_cascade(spec)


@lru_cache(maxsize=None)
def metrics_of(n_passage: int = 101, n_eta: int = 31) -> mt.GridMetrics:
    return mt.build_metrics(make_grid(cascade_of(SPEC), GridSettings(n_passage=n_passage, n_eta=n_eta), INLET))


def periodic_density(metrics: mt.GridMetrics, seed: int = 1) -> np.ndarray:
    """A random positive density, equal at eta = 0 and 1 (as coefficients.py requires on the
    periodic stations)."""
    rng = np.random.default_rng(seed)
    rho = rng.uniform(0.6, 1.4, (len(metrics.grid.m), len(metrics.grid.eta)))
    rho[:, -1] = rho[:, 0]
    return rho


@lru_cache(maxsize=None)
def c2_case(n_passage: int = 101, n_eta: int = 31, mdot: float = 1.0, c_ref: float | None = None, uniform: bool = True):
    metrics = metrics_of(n_passage, n_eta)
    rho = np.ones((len(metrics.grid.m), len(metrics.grid.eta))) if uniform else periodic_density(metrics)
    co = cf.build_coefficients(metrics, rho, mdot, c_ref=c_ref)
    return metrics, co, dz.build_interior_stencil(co, metrics)


def random_fields(shape, seed: int):
    """Random coefficient fields: A, C in [0.1, 10], B, D, E in [-10, 10]."""
    rng = np.random.default_rng(seed)
    A, C = rng.uniform(0.1, 10.0, shape), rng.uniform(0.1, 10.0, shape)
    B, D, E = (rng.uniform(-10.0, 10.0, shape) for _ in range(3))
    return A, B, C, D, E


def raw_weights(A, B, C, D, E, dm, deta) -> dict:
    """The raw nine-point weights: Eqs. 5-35 to 5-39 substituted into Eq. 5-21."""
    return dict(P=-2.0 * A / deta ** 2 - 2.0 * C / dm ** 2,
                W=C / dm ** 2 - E / (2.0 * dm), E=C / dm ** 2 + E / (2.0 * dm),
                S=A / deta ** 2 - D / (2.0 * deta), N=A / deta ** 2 + D / (2.0 * deta),
                NE=-B / (2.0 * dm * deta), NW=B / (2.0 * dm * deta), SE=B / (2.0 * dm * deta), SW=-B / (2.0 * dm * deta))


def raw_operator(A, B, C, D, E, psi, dm, deta) -> np.ndarray:
    """A psi_ee - 2 B psi_me + C psi_mm + D psi_e + E psi_m at the interior nodes, each
    derivative by its own difference formula, Eqs. 5-35 to 5-39 (no stencil weights)."""
    c = (slice(1, -1), slice(1, -1))
    psi_m = (psi[2:, 1:-1] - psi[:-2, 1:-1]) / (2.0 * dm)
    psi_e = (psi[1:-1, 2:] - psi[1:-1, :-2]) / (2.0 * deta)
    psi_mm = (psi[2:, 1:-1] - 2.0 * psi[1:-1, 1:-1] + psi[:-2, 1:-1]) / dm ** 2
    psi_ee = (psi[1:-1, 2:] - 2.0 * psi[1:-1, 1:-1] + psi[1:-1, :-2]) / deta ** 2
    psi_me = (psi[2:, 2:] - psi[:-2, 2:] - psi[2:, :-2] + psi[:-2, :-2]) / (4.0 * dm * deta)
    return A[c] * psi_ee - 2.0 * B[c] * psi_me + C[c] * psi_mm + D[c] * psi_e + E[c] * psi_m


def normalized_residual(stencil: dz.InteriorStencil, psi: np.ndarray, diagonals=None) -> np.ndarray:
    """Eq. 5-44 minus its rhs at the interior nodes, from the named weights (the diagonal
    weights may be overridden to try sign flips)."""
    c = (slice(1, -1), slice(1, -1))
    ne, nw, se, sw = diagonals or (stencil.northeast, stencil.northwest, stencil.southeast, stencil.southwest)
    return (stencil.center[c] * psi[1:-1, 1:-1] + stencil.west[c] * psi[:-2, 1:-1] + stencil.east[c] * psi[2:, 1:-1]
            + stencil.south[c] * psi[1:-1, :-2] + stencil.north[c] * psi[1:-1, 2:]
            + ne[c] * psi[2:, 2:] + nw[c] * psi[:-2, 2:] + se[c] * psi[2:, :-2] + sw[c] * psi[:-2, :-2]
            - stencil.rhs[c])


def lattice(n_m: int, n_eta: int, dm: float, deta: float, m0: float = 0.0, eta0: float = 0.0):
    M, H = np.meshgrid(m0 + dm * np.arange(n_m), eta0 + deta * np.arange(n_eta), indexing="ij")
    return M, H


def smooth_coefficients(M, H) -> dict:
    """Smooth coefficient fields with A, C > 0, and D, E the derivatives Eqs. 5-25 and 5-26
    ask for (D = A_eta - B_m, E = C_m - B_eta), all written out by hand."""
    w = math.pi
    A = 1.5 + 0.4 * np.sin(1.7 * M + 0.3) * np.cos(w * H)
    A_e = -0.4 * np.sin(1.7 * M + 0.3) * w * np.sin(w * H)
    B = 0.6 * np.cos(1.3 * M) + 0.3 * np.sin(w * H + 0.4)
    B_m, B_e = -0.6 * 1.3 * np.sin(1.3 * M), 0.3 * w * np.cos(w * H + 0.4)
    C = 0.9 + 0.3 * np.cos(2.1 * M + 0.1) * np.sin(w * H + 0.2)
    C_m = -0.3 * 2.1 * np.sin(2.1 * M + 0.1) * np.sin(w * H + 0.2)
    return dict(A=A, B=B, C=C, D=A_e - B_m, E=C_m - B_e)


def smooth_psi(M, H, km: float = 2.0, ke: float = 3.0) -> dict:
    """psi = sin(km m + 0.3) cos(ke eta + 0.2) + 0.5 eta and its derivatives."""
    s, c = np.sin(km * M + 0.3), np.cos(km * M + 0.3)
    se, ce = np.sin(ke * H + 0.2), np.cos(ke * H + 0.2)
    return dict(psi=s * ce + 0.5 * H, m=km * c * ce, e=-ke * s * se + 0.5, mm=-km * km * s * ce,
                ee=-ke * ke * s * ce, me=-km * ke * c * se)


def exact_operator(k: dict, d: dict) -> np.ndarray:
    return k["A"] * d["ee"] - 2.0 * k["B"] * d["me"] + k["C"] * d["mm"] + k["D"] * d["e"] + k["E"] * d["m"]


def observed_orders(errors, spacings) -> list[float]:
    return [math.log(errors[k] / errors[k + 1]) / math.log(spacings[k] / spacings[k + 1])
            for k in range(len(errors) - 1)]


def interior(field: np.ndarray) -> np.ndarray:
    return field[1:-1, 1:-1]


# --- the module ---------------------------------------------------------------------------

def test_the_public_names_and_the_deferred_interfaces():
    tree = ast.parse(Path(dz.__file__).read_text(encoding="utf-8"))
    functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef) and not node.name.startswith("_")}
    classes = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
    assert functions == {"build_interior_stencil", "build_periodic_pde_rows", "first_derivatives", "mixed_derivative",
                         "residual", "assemble"}
    assert classes == {"DiscretizationError", "InteriorStencil", "PDETerm", "PeriodicPDERow"}
    assert issubclass(dz.DiscretizationError, ValueError)
    assert dz.Coefficients is cf.CoefficientFields
    metrics = metrics_of(31, 5)
    for call in (lambda: dz.first_derivatives(np.zeros(3), metrics.grid, 0.0),
                 lambda: dz.mixed_derivative(np.zeros(3), metrics.grid, 0.0),
                 lambda: dz.residual(np.zeros(3), None, metrics.grid), lambda: dz.assemble(None, metrics.grid, None)):
        with pytest.raises(NotImplementedError):
            call()


def test_no_matrix_no_solve_no_upwind():
    """The module builds no sparse matrix, numbers no unknowns, solves nothing and has no
    flow-direction-dependent difference."""
    source = Path(dz.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    called = {node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
              for node in ast.walk(tree) if isinstance(node, ast.Call)}
    assert not {"csr_matrix", "lil_matrix", "coo_matrix", "spsolve", "cg", "gmres", "bicgstab", "node_index",
                "solve", "sign", "clip", "heaviside"} & called
    imported = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
    imported |= {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    outside = {name for name in imported if name.split(".")[0] not in sys.stdlib_module_names}
    assert outside == {"numpy", "numpy.typing", "scipy.sparse", "potential_flow_solver.boundary_conditions",
                       "potential_flow_solver.coefficients", "potential_flow_solver.grid",
                       "potential_flow_solver.metrics"}


def test_the_stencil_fields_and_their_meaning():
    assert [item.name for item in fields(dz.InteriorStencil)] == ["center", "west", "east", "south", "north", "mixed",
                                                                  "rhs", "H", "interior", "dm", "deta"]
    source = Path(dz.__file__).read_text(encoding="utf-8")
    for line in ("west: Field                     # A~ (5-45)", "east: Field                     # B~ (5-46)",
                 "south: Field                    # C~ (5-47)", "north: Field                    # D~ (5-48)",
                 "mixed: Field                    # E~ (5-49)", "rhs: Field                      # Q~ (5-50)"):
        assert line in source


def test_the_stencil_is_frozen_read_only_and_nan_outside_the_interior():
    metrics, _, st = c2_case()
    n_m, n_eta = len(metrics.grid.m), len(metrics.grid.eta)
    expected = np.zeros((n_m, n_eta), dtype=bool)
    expected[1:-1, 1:-1] = True
    np.testing.assert_array_equal(st.interior, expected)
    for name in ("center", "west", "east", "south", "north", "mixed", "rhs", "H", "northeast", "northwest",
                 "southeast", "southwest", "interior", "weights"):
        array = getattr(st, name)
        assert not array.flags.writeable, name
        if name not in ("interior", "weights"):
            assert array.shape == (n_m, n_eta)
            assert np.all(np.isfinite(array[st.interior])) and np.all(np.isnan(array[~st.interior])), name
    assert np.all(np.isnan(st.weights[~st.interior])) and np.all(np.isfinite(st.weights[st.interior]))
    with pytest.raises(FrozenInstanceError):
        st.west = st.east


# --- Eqs. 5-45 to 5-50 ------------------------------------------------------------------------

EQUATIONS = {
    "5-45 west": ("west", lambda A, B, C, D, E, dm, de, H: -(C / dm ** 2 - E / (2.0 * dm)) / H),
    "5-46 east": ("east", lambda A, B, C, D, E, dm, de, H: -(C / dm ** 2 + E / (2.0 * dm)) / H),
    "5-47 south": ("south", lambda A, B, C, D, E, dm, de, H: -(A / de ** 2 - D / (2.0 * de)) / H),
    "5-48 north": ("north", lambda A, B, C, D, E, dm, de, H: -(A / de ** 2 + D / (2.0 * de)) / H),
    "5-49 mixed": ("mixed", lambda A, B, C, D, E, dm, de, H: B / (2.0 * dm * de * H)),
}


@pytest.mark.parametrize("equation", EQUATIONS)
@pytest.mark.parametrize("seed", range(4))
def test_aungier_weights_on_random_fields(equation, seed):
    """Random A, C in [0.1, 10], B, D, E in [-10, 10], dm in [1e-3, 0.3], deta in [1e-2, 0.5]:
    each weight is its Eq. 5-45 to 5-49 with H = 2 A / deta^2 + 2 C / dm^2. Measured: 0, bit
    for bit (H and all five weights, four seeds)."""
    rng = np.random.default_rng(100 + seed)
    dm, deta = rng.uniform(1e-3, 0.3), rng.uniform(1e-2, 0.5)
    A, B, C, D, E = random_fields((9, 7), seed)
    st = dz._interior_stencil(A, B, C, D, E, dm, deta)
    H = 2.0 * A / deta ** 2 + 2.0 * C / dm ** 2
    name, formula = EQUATIONS[equation]
    expected = formula(A, B, C, D, E, dm, deta, H)
    np.testing.assert_allclose(interior(st.H), interior(H), rtol=2.3e-16, atol=0.0)
    scale = np.maximum.reduce([np.abs(interior(f(A, B, C, D, E, dm, deta, H))) for _, f in EQUATIONS.values()])
    assert np.max(np.abs(interior(getattr(st, name)) - interior(expected)) / scale) <= 4.5e-16
    assert np.all(interior(st.center) == 1.0)


@pytest.mark.parametrize("seed", range(4))
def test_eq_5_50_the_rhs_is_zero(seed):
    """Q = 0 (flat cascade at constant radius), so rhs = -Q / H = 0 at every interior node."""
    st = dz._interior_stencil(*random_fields((8, 6), seed), 0.05, 0.1)
    assert np.all(interior(st.rhs) == 0.0)
    assert np.all(np.isnan(st.rhs[~st.interior]))


@pytest.mark.parametrize("c_ref", (None, 0.07, 2.5))
def test_aungier_weights_through_the_public_api(c_ref):
    """On the default C2 grid, coefficients from a random density: the weights are Eqs. 5-45
    to 5-49 of the coefficient fields with dm = L_ref grid.dm. Measured: 0, bit for bit
    (the bound is 2 ulp of the row's largest weight)."""
    metrics, co, st = c2_case(101, 31, 3.7, c_ref, False)
    g = metrics.grid
    dm, deta = (1.0 if c_ref is None else c_ref) * g.dm, g.deta
    assert st.dm == dm and st.deta == deta
    H = 2.0 * co.A / deta ** 2 + 2.0 * co.C / dm ** 2
    scale = np.maximum.reduce([np.abs(interior(f(co.A, co.B, co.C, co.D, co.E, dm, deta, H)))
                               for _, f in EQUATIONS.values()])
    for name, formula in EQUATIONS.values():
        expected = formula(co.A, co.B, co.C, co.D, co.E, dm, deta, H)
        assert np.max(np.abs(interior(getattr(st, name)) - interior(expected)) / scale) <= 4.5e-16, name


@pytest.mark.parametrize("seed", range(4))
def test_the_nine_weights_are_the_raw_weights_over_minus_H(seed):
    """Every normalized weight is the raw weight of Eq. 5-21 with Eqs. 5-35 to 5-39 divided
    by -H, through the named arrays, the four diagonal properties and the (3, 3) layout
    [1 + di, 1 + dj]. Measured: at most 5.6e-17 (the normalized weights are O(1))."""
    rng = np.random.default_rng(200 + seed)
    dm, deta = rng.uniform(1e-3, 0.3), rng.uniform(1e-2, 0.5)
    A, B, C, D, E = random_fields((8, 9), seed)
    st = dz._interior_stencil(A, B, C, D, E, dm, deta)
    raw = raw_weights(A, B, C, D, E, dm, deta)
    H = -raw["P"]
    named = dict(P=st.center, W=st.west, E=st.east, S=st.south, N=st.north, NE=st.northeast, NW=st.northwest,
                 SE=st.southeast, SW=st.southwest)
    layout = dict(P=(1, 1), W=(0, 1), E=(2, 1), S=(1, 0), N=(1, 2), NE=(2, 2), NW=(0, 2), SE=(2, 0), SW=(0, 0))
    weights = st.weights
    for key in raw:
        expected = interior(-raw[key] / H)
        assert np.max(np.abs(interior(named[key]) - expected)) <= 4.5e-16, key
        assert np.max(np.abs(interior(weights[..., layout[key][0], layout[key][1]]) - expected)) <= 4.5e-16, key


@pytest.mark.parametrize("seed", range(4))
def test_raw_and_normalized_residuals_agree(seed):
    """Random coefficients, spacings and a random psi: the normalized residual of Eq. 5-44 is
    -(raw residual) / H, the raw one taken term by term from Eqs. 5-35 to 5-39. Measured:
    0.97e-15 to 2.1e-15 relative to the size of the raw terms over H (four seeds)."""
    rng = np.random.default_rng(300 + seed)
    dm, deta = rng.uniform(1e-3, 0.3), rng.uniform(1e-2, 0.5)
    A, B, C, D, E = random_fields((11, 9), seed)
    psi = rng.uniform(-2.0, 2.0, (11, 9))
    st = dz._interior_stencil(A, B, C, D, E, dm, deta)
    raw = raw_operator(A, B, C, D, E, psi, dm, deta)
    H = interior(2.0 * A / deta ** 2 + 2.0 * C / dm ** 2)
    size = (np.abs(interior(A)) / deta ** 2 + np.abs(interior(B)) / (dm * deta) + np.abs(interior(C)) / dm ** 2) / H
    assert np.max(np.abs(normalized_residual(st, psi) + raw / H) / size) <= 5e-15


def test_the_mixed_derivative_alone_has_the_minus_two_B_sign():
    """A = C = 1, D = E = 0, B = 0.7, psi = m eta (m, eta offset from 0 so that no neighbor
    value vanishes): psi_mm = psi_ee = 0 and psi_me = 1, so the raw operator is -2 B and the
    normalized residual 2 B / H, exactly. Each of the 15 sign flips of the four diagonal
    weights (alone, in pairs, in threes, all four) changes it by at least 1e-3 of the value.
    Measured: 7.1e-14 relative, the round-off of the four-point bracket over 4 dm deta."""
    dm, deta = 0.04, 0.06
    M, H = lattice(9, 8, dm, deta, m0=0.3, eta0=0.2)
    ones, zeros = np.ones_like(M), np.zeros_like(M)
    B = np.full_like(M, 0.7)
    st = dz._interior_stencil(ones, B, ones, zeros, zeros, dm, deta)
    psi = M * H
    Hs = 2.0 / deta ** 2 + 2.0 / dm ** 2
    np.testing.assert_allclose(raw_operator(ones, B, ones, zeros, zeros, psi, dm, deta), -1.4, rtol=1e-13, atol=0.0)
    residual = normalized_residual(st, psi)
    np.testing.assert_allclose(residual, 1.4 / Hs, rtol=1e-13, atol=0.0)
    diagonals = (st.northeast, st.northwest, st.southeast, st.southwest)
    for count in range(1, 5):
        for flipped in itertools.combinations(range(4), count):
            trial = tuple(-d if k in flipped else d for k, d in enumerate(diagonals))
            off = np.abs(normalized_residual(st, psi, trial) - 1.4 / Hs).max()
            assert off >= 1e-3 * 1.4 / Hs, flipped


POLYNOMIALS = {
    "1": (lambda M, H: np.ones_like(M), lambda k, M, H: np.zeros_like(M)),
    "m": (lambda M, H: M, lambda k, M, H: k["E"]),
    "eta": (lambda M, H: H, lambda k, M, H: k["D"]),
    "m^2": (lambda M, H: M * M, lambda k, M, H: 2.0 * k["C"] + 2.0 * k["E"] * M),
    "eta^2": (lambda M, H: H * H, lambda k, M, H: 2.0 * k["A"] + 2.0 * k["D"] * H),
    "m eta": (lambda M, H: M * H, lambda k, M, H: -2.0 * k["B"] + k["D"] * M + k["E"] * H),
}


@pytest.mark.parametrize("polynomial", POLYNOMIALS)
def test_polynomials_give_the_exact_operator(polynomial):
    """Quadratics are differentiated exactly by Eqs. 5-35 to 5-39, so for psi = 1, m, eta,
    m^2, eta^2, m eta the normalized residual is -L(psi) / H of the exact operator, on
    random coefficients. The round-off is that of a sum of nine O(1) weights times psi:
    measured 1.7e-16 (psi = 1), 3.7e-17 (m), 1.0e-16 (eta), 1.0e-17 (m^2), 7.9e-17 (eta^2),
    1.5e-17 (m eta), against max |psi| up to 1."""
    dm, deta = 0.037, 0.051
    M, H = lattice(12, 10, dm, deta, m0=-0.2, eta0=0.1)
    A, B, C, D, E = random_fields(M.shape, 7)
    k = dict(A=A, B=B, C=C, D=D, E=E)
    st = dz._interior_stencil(A, B, C, D, E, dm, deta)
    psi_of, operator_of = POLYNOMIALS[polynomial]
    psi = psi_of(M, H)
    exact = interior(operator_of(k, M, H)) / interior(st.H)
    assert np.max(np.abs(normalized_residual(st, psi) + exact)) <= 4.5e-16 * max(1.0, np.abs(psi).max())


# --- convergence ------------------------------------------------------------------------------

def manufactured_error(n_m: int, n_eta: int, case: str) -> tuple[float, float, float]:
    """(dm, deta, max |raw operator - exact|) on an (n_m + 1) x (n_eta + 1) lattice over
    [0, M_LENGTH] x [0, 1] with the smooth coefficients, for
        'both'   psi = sin(2 m + 0.3) cos(3 eta + 0.2) + 0.5 eta
        'dm'     psi = sin(2 m + 0.3) (1 + eta + 0.5 eta^2)    (eta differences exact)
        'deta'   psi = (1 + m + 0.5 m^2) cos(3 eta + 0.2)      (m differences exact)
        'mixed dm', 'mixed deta': the mixed term -2 B psi_me alone, with the psi of 'dm', 'deta'."""
    dm, deta = M_LENGTH / n_m, 1.0 / n_eta
    M, H = lattice(n_m + 1, n_eta + 1, dm, deta)
    k = smooth_coefficients(M, H)
    if case == "both":
        d = smooth_psi(M, H)
    elif case in ("dm", "mixed dm"):
        s, c, q = np.sin(2.0 * M + 0.3), np.cos(2.0 * M + 0.3), 1.0 + H + 0.5 * H * H
        d = dict(psi=s * q, m=2.0 * c * q, e=s * (1.0 + H), mm=-4.0 * s * q, ee=s, me=2.0 * c * (1.0 + H))
    else:
        s, c, p = np.sin(3.0 * H + 0.2), np.cos(3.0 * H + 0.2), 1.0 + M + 0.5 * M * M
        d = dict(psi=p * c, m=(1.0 + M) * c, e=-3.0 * p * s, mm=c, ee=-9.0 * p * c, me=-3.0 * (1.0 + M) * s)
    if case.startswith("mixed"):
        zeros = np.zeros_like(M)
        k = dict(A=zeros, B=k["B"], C=zeros, D=zeros, E=zeros)
    discrete = raw_operator(k["A"], k["B"], k["C"], k["D"], k["E"], d["psi"], dm, deta)
    return dm, deta, float(np.abs(discrete - interior(exact_operator(k, d))).max())


@pytest.mark.parametrize("case", ("both", "dm", "deta", "mixed dm", "mixed deta"))
def test_the_operator_converges_at_second_order(case):
    """Smooth manufactured coefficients and psi, 16 to 128 cells: 'both' refines dm and deta
    together; 'dm' refines dm with n_eta = 16 fixed and a psi quadratic in eta; 'deta'
    refines deta with n_m = 16 fixed and a psi quadratic in m; the mixed term alone in each
    direction. Measured orders (16 -> 32 -> 64 -> 128): both 1.961, 1.985, 1.994; dm 1.999,
    1.999, 2.000; deta 1.938, 1.980, 1.993; mixed dm 1.948, 1.979, 1.991; mixed deta 1.998,
    2.000, 2.000. The raw operator is the module's normalized residual times -H (checked
    above), so this is the order of Eq. 5-44 itself; it is also checked on the module
    directly."""
    errors, spacings = [], []
    for n in LEVELS:
        n_m, n_eta = {"both": (n, n), "dm": (n, 16), "mixed dm": (n, 16), "deta": (16, n), "mixed deta": (16, n)}[case]
        dm, deta, error = manufactured_error(n_m, n_eta, case)
        errors.append(error)
        spacings.append(deta if case.endswith("deta") else dm)
    orders = observed_orders(errors, spacings)
    assert min(orders) >= 1.93 and orders[-1] >= 1.99


def test_the_normalized_equation_converges_at_second_order():
    """The module's own Eq. 5-44 on the smooth case: max |residual + L(psi) / H| H falls at
    second order (measured 9.55e-2, 2.45e-2, 6.20e-3, 1.56e-3; orders 1.961, 1.985, 1.994)."""
    errors, spacings = [], []
    for n in LEVELS:
        dm, deta = M_LENGTH / n, 1.0 / n
        M, H = lattice(n + 1, n + 1, dm, deta)
        k, d = smooth_coefficients(M, H), smooth_psi(M, H)
        st = dz._interior_stencil(k["A"], k["B"], k["C"], k["D"], k["E"], dm, deta)
        Hs = interior(st.H)
        errors.append(float(np.abs((normalized_residual(st, d["psi"]) + interior(exact_operator(k, d)) / Hs) * Hs).max()))
        spacings.append(dm)
    orders = observed_orders(errors, spacings)
    assert min(orders) >= 1.95 and orders[-1] >= 1.99


def conservative_operator(k: dict, psi: np.ndarray, dm: float, deta: float) -> np.ndarray:
    """A conservative discretization of Eq. 5-20, dF/d eta - dG/dm with F = A psi_e - B psi_m and
    G = B psi_e - C psi_m, from fluxes on the cell faces between nodes (coefficients averaged
    to the face, the cross derivative from the four nodes around it): a different second-order
    scheme, written here only as a reference."""
    A, B, C = k["A"], k["B"], k["C"]
    # F on the faces (i, j + 1/2), i = 1 .. n_m - 2, j = 0 .. n_eta - 2
    Af, Bf = 0.5 * (A[1:-1, 1:] + A[1:-1, :-1]), 0.5 * (B[1:-1, 1:] + B[1:-1, :-1])
    F = (Af * (psi[1:-1, 1:] - psi[1:-1, :-1]) / deta
         - Bf * (psi[2:, 1:] + psi[2:, :-1] - psi[:-2, 1:] - psi[:-2, :-1]) / (4.0 * dm))
    # G on the faces (i + 1/2, j), i = 0 .. n_m - 2, j = 1 .. n_eta - 2
    Bg, Cg = 0.5 * (B[1:, 1:-1] + B[:-1, 1:-1]), 0.5 * (C[1:, 1:-1] + C[:-1, 1:-1])
    G = (Bg * (psi[1:, 2:] + psi[:-1, 2:] - psi[1:, :-2] - psi[:-1, :-2]) / (4.0 * deta)
         - Cg * (psi[1:, 1:-1] - psi[:-1, 1:-1]) / dm)
    return (F[:, 1:] - F[:, :-1]) / deta - (G[1:, :] - G[:-1, :]) / dm


def test_the_expanded_and_conservative_forms_approach_each_other():
    """With D = A_eta - B_m and E = C_m - B_eta (the smooth coefficients), the module's Eq.
    5-44 (times -H: the expanded form) and the conservative face-flux form of Eq. 5-20
    differ by O(h^2). Measured: max gap 3.8e-2, 9.7e-3, 2.4e-3, 6.1e-4 (16 to 128 cells),
    orders 1.979, 1.997, 1.999; each form alone is 9.5e-2 (expanded) and 1.2e-1
    (conservative) from the exact operator at 16 cells, 1.6e-3 and 1.9e-3 at 128."""
    gaps, spacings = [], []
    for n in LEVELS:
        dm, deta = M_LENGTH / n, 1.0 / n
        M, H = lattice(n + 1, n + 1, dm, deta)
        k, d = smooth_coefficients(M, H), smooth_psi(M, H)
        st = dz._interior_stencil(k["A"], k["B"], k["C"], k["D"], k["E"], dm, deta)
        expanded = -normalized_residual(st, d["psi"]) * interior(st.H)
        gaps.append(float(np.abs(expanded - conservative_operator(k, d["psi"], dm, deta)).max()))
        spacings.append(dm)
    assert min(observed_orders(gaps, spacings)) >= 1.97
    assert gaps[-1] <= 6.5e-4


# --- lengths and scaling on the default C2 grid ------------------------------------------------

def test_dm_is_the_length_scale_times_grid_dm():
    for c_ref in (None, 0.05, 1.0, 4.0):
        metrics, co, st = c2_case(101, 31, 1.0, c_ref)
        assert st.dm == co.length_scale * metrics.grid.dm and st.deta == metrics.grid.deta


def test_normalized_mode_is_c_ref_one_bit_for_bit():
    _, _, normalized = c2_case(101, 31, 1.0, None, False)
    _, _, unit = c2_case(101, 31, 1.0, 1.0, False)
    for name in ("center", "west", "east", "south", "north", "mixed", "rhs", "H", "interior"):
        np.testing.assert_array_equal(getattr(normalized, name), getattr(unit, name))
    assert normalized.dm == unit.dm == metrics_of().grid.dm


def c2_psi(metrics: mt.GridMetrics) -> np.ndarray:
    """A smooth dimensionless psi(m*, eta) on the grid's nodes."""
    g = metrics.grid
    M, H = np.meshgrid(g.m, g.eta, indexing="ij")
    return H + 0.05 * np.sin(2.0 * M) * np.sin(2.0 * math.pi * H)


@pytest.mark.parametrize("lam", (0.05, 3.0, 250.0))
@pytest.mark.parametrize("law", ("fixed mdot", "similar twins", "mass flow"))
def test_the_normalized_stencil_does_not_scale(law, lam):
    """Every weight of Eq. 5-44 is dimensionless and unchanged when the chord is scaled with
    mdot' held (A, D ~ 1/L, B, E ~ L^0, C ~ L, dm ~ L), for similar twins (mdot' ~ L as well),
    and when mdot' alone is scaled; so is the normalized residual of one psi(m*, eta). H ~ 1/L,
    L^0 and mdot' respectively. Measured: weights 2.2e-16 (absolute; they are O(1)), H 7.6e-16
    relative, residual 3.5e-16 absolute (the round-off of nine O(1) weights times psi, whose
    largest value is 1.05; the residual itself is up to 1.5e-2)."""
    c, mdot = 0.1, 2.0
    _, _, base = c2_case(101, 31, mdot, c, False)
    scaled_c, scaled_mdot, h_power = {"fixed mdot": (lam * c, mdot, -1.0), "similar twins": (lam * c, lam * mdot, 0.0),
                                      "mass flow": (c, lam * mdot, None)}[law]
    metrics, _, other = c2_case(101, 31, scaled_mdot, scaled_c, False)
    for name in ("west", "east", "south", "north", "mixed"):
        np.testing.assert_allclose(other.__getattribute__(name), base.__getattribute__(name), rtol=0.0, atol=1.5e-15)
    factor = lam if h_power is None else lam ** h_power
    np.testing.assert_allclose(interior(other.H), factor * interior(base.H), rtol=1e-15, atol=0.0)
    psi = c2_psi(metrics)
    r0, r1 = normalized_residual(base, psi), normalized_residual(other, psi)
    assert np.abs(r1 - r0).max() <= 1e-15


# --- the default C2 case --------------------------------------------------------------------

@pytest.mark.parametrize("level", ((101, 31), (401, 121)))
def test_c2_weights_are_finite_and_H_is_positive(level):
    """Measured min H 2.84e4 (101) and 4.47e5 (401), at j = 1 mid-blade; max H 7.53e4 and
    1.85e6 at the first blade station. Weights finite at every interior node, NaN elsewhere."""
    metrics, co, st = c2_case(*level)
    assert np.all(st.H[st.interior] > 0.0)
    for name in ("west", "east", "south", "north", "mixed", "H"):
        field = getattr(st, name)
        assert np.all(np.isfinite(field[st.interior])) and np.all(np.isnan(field[~st.interior]))
    assert np.nanmin(st.H) >= {101: 2.8e4, 401: 4.4e5}[level[0]]


@pytest.mark.parametrize("level", ((101, 31), (401, 121)))
def test_c2_row_properties(level):
    """west + east + south + north = -1 (so psi = const is exact), measured to 2.2e-16; and,
    since B / sqrt(A C) = sin(beta) for coefficients of real metrics and H >= 4 sqrt(A C) /
    (dm deta), |mixed| <= |sin beta| / 8 < 1/8: measured max |mixed| 0.1212 (101), 0.1212 (401),
    and the bound holds to 1e-16. So no row is diagonally dominant (sum of |neighbors| > 1
    wherever B != 0): measured 1.0001 to 1.485 (101), 1.000 to 1.492 (401)."""
    metrics, _, st = c2_case(*level)
    I = st.interior
    total = st.west + st.east + st.south + st.north
    assert np.abs(total[I] + 1.0).max() <= 4.5e-16
    sin_beta = np.abs(metrics.sin_beta)
    assert np.all(np.abs(st.mixed[I]) <= sin_beta[I] / 8.0 + 1e-16)
    assert np.abs(st.mixed[I]).max() <= 0.125


def test_c2_stencil_is_deterministic_and_its_signs_follow_the_coefficients():
    """Built twice from fresh coefficient objects: bit for bit the same. Every sign is the
    one its formula gives the coefficients (no flip of the implementation, through the
    curvature steps of the C2 walls too): sign(mixed) = sign(B); west and east negative
    (|E| dm / (2 C) measured at most 0.028); south > 0 exactly where D deta > 2 A, north > 0
    exactly where -D deta > 2 A (|D| deta / (2 A) measured up to 3.9, at the corners)."""
    metrics = metrics_of(101, 31)
    rho = np.ones((len(metrics.grid.m), len(metrics.grid.eta)))
    first = dz.build_interior_stencil(cf.build_coefficients(metrics, rho, 1.3), metrics)
    second = dz.build_interior_stencil(cf.build_coefficients(metrics, rho.copy(), 1.3), metrics)
    for name in ("west", "east", "south", "north", "mixed", "H"):
        np.testing.assert_array_equal(getattr(first, name), getattr(second, name))
    co = cf.build_coefficients(metrics, rho, 1.3)
    I = first.interior
    np.testing.assert_array_equal(np.sign(first.mixed[I]), np.sign(co.B[I]))
    assert np.all(first.west[I] < 0.0) and np.all(first.east[I] < 0.0)
    assert (np.abs(co.E) * first.dm / (2.0 * co.C))[I].max() <= 0.03
    np.testing.assert_array_equal(first.south[I] > 0.0, (co.D * first.deta > 2.0 * co.A)[I])
    np.testing.assert_array_equal(first.north[I] > 0.0, (-co.D * first.deta > 2.0 * co.A)[I])


@pytest.mark.parametrize("level", ((101, 31), (401, 121)))
def test_c2_corner_stations(level):
    """Around first_blade_i and last_blade_i (two stations either side) the weights are
    finite and unclipped. The large D of the corners (|D| up to 300 at 101) makes some south
    or north weights positive, and only there: measured at 101, 9 south (station f-1) and 16
    north (f-1, f+1) of 9889 interior nodes; at 401, 43 and 100 (f-1, f+1, l+1) of 161364
    (101 north with the legacy PCHIP thickness).
    Elsewhere every neighbor weight is negative."""
    metrics, _, st = c2_case(*level)
    g = metrics.grid
    f, l = g.first_blade_i, g.last_blade_i
    near = list(range(f - 2, f + 3)) + list(range(l - 2, l + 3))
    for name in ("west", "east", "south", "north", "mixed", "H"):
        assert np.all(np.isfinite(getattr(st, name)[near, 1:-1]))
    positive = st.interior & ((st.south > 0.0) | (st.north > 0.0))
    stations = set(np.flatnonzero(positive.any(axis=1)).tolist())
    assert stations and stations <= {f - 1, f, f + 1, l, l + 1}
    counts = (int((st.interior & (st.south > 0.0)).sum()), int((st.interior & (st.north > 0.0)).sum()))
    assert counts == {101: (9, 16), 401: (43, 100)}[level[0]]


# --- inputs ------------------------------------------------------------------------------------

def test_coefficients_of_other_metrics_are_refused():
    """A coefficient set no longer of these metrics: after rebuild_exit (exit tan(beta)
    changed: measured 2.2e-4 relative at the first exit station), of another grid with the
    same shape (another inlet angle), or carrying another length scale."""
    metrics, co, _ = c2_case(101, 31, 1.0, 0.1)
    g = metrics.grid
    turned = mt.build_metrics(rebuild_exit(g, cascade_of(SPEC).kappa_te + math.radians(5.0)))
    other = mt.build_metrics(make_grid(cascade_of(SPEC), GridSettings(n_passage=101, n_eta=31), math.radians(40.0)))
    for wrong in (turned, other):
        assert len(wrong.grid.m) == len(g.m)
        with pytest.raises(dz.DiscretizationError, match="do not belong"):
            dz.build_interior_stencil(co, wrong)
    with pytest.raises(dz.DiscretizationError, match="do not belong"):
        dz.build_interior_stencil(replace(co, c_ref=0.2), metrics)
    with pytest.raises(dz.DiscretizationError, match="do not belong"):
        dz.build_interior_stencil(replace(co, c_ref=None), metrics)
    with pytest.raises(dz.DiscretizationError, match="the grid has"):
        dz.build_interior_stencil(co, metrics_of(101, 21))


def test_wrong_types_are_refused():
    metrics, co, _ = c2_case()
    for args in ((metrics, co), (co, metrics.grid), (None, metrics), (co.A, metrics)):
        with pytest.raises(TypeError):
            dz.build_interior_stencil(*args)


def test_raw_arrays_are_validated():
    A, B, C, D, E = random_fields((6, 5), 3)
    for k, bad in enumerate((np.ones((6, 4)), np.ones(6), np.ones((5, 5)))):
        arrays = [A, B, C, D, E]
        arrays[k % 5] = bad
        with pytest.raises(dz.DiscretizationError, match="shape"):
            dz._interior_stencil(*arrays, 0.1, 0.1)
    for value in (np.nan, np.inf, -np.inf):
        for k in range(5):
            arrays = [a.copy() for a in (A, B, C, D, E)]
            arrays[k][2, 2] = value
            with pytest.raises(dz.DiscretizationError, match="not finite"):
                dz._interior_stencil(*arrays, 0.1, 0.1)
    with pytest.raises(dz.DiscretizationError, match="no interior node"):
        dz._interior_stencil(*(a[:, :2] for a in (A, B, C, D, E)), 0.1, 0.1)
    with pytest.raises(dz.DiscretizationError, match="no interior node"):
        dz._interior_stencil(*(a[:2] for a in (A, B, C, D, E)), 0.1, 0.1)


def test_spacings_are_validated():
    fields_ = random_fields((5, 5), 4)
    for name, position in (("dm", 0), ("deta", 1)):
        for bad in (0.0, -0.1, np.nan, np.inf, -np.inf):
            spacing = [0.1, 0.1]
            spacing[position] = bad
            with pytest.raises(dz.DiscretizationError, match=name):
                dz._interior_stencil(*fields_, *spacing)
        for bad in ("0.1", True, None, np.ones(1)):
            spacing = [0.1, 0.1]
            spacing[position] = bad
            with pytest.raises(TypeError, match=name):
                dz._interior_stencil(*fields_, *spacing)


def test_H_must_be_finite_and_positive():
    A, B, C, D, E = random_fields((6, 6), 5)
    for a, c in ((0.0, 0.0), (-1.0, -1.0), (-5.0, 0.01)):
        A2, C2 = A.copy(), C.copy()
        A2[3, 2], C2[3, 2] = a, c
        with pytest.raises(dz.DiscretizationError, match="H = "):
            dz._interior_stencil(A2, B, C2, D, E, 0.1, 0.1)
    A2 = A.copy()
    A2[3, 2] = 1e300
    with pytest.raises(dz.DiscretizationError, match="H = "):
        dz._interior_stencil(A2, B, C, D, E, 0.1, 1e-10)
    A2 = A.copy()
    A2[0, 0] = -1.0                                         # not an interior node: no row, no check
    dz._interior_stencil(A2, B, C, D, E, 0.1, 0.1)


def test_direct_construction_is_checked():
    st = dz._interior_stencil(*random_fields((5, 4), 6), 0.1, 0.2)
    values = {item.name: getattr(st, item.name) for item in fields(st)}
    dz.InteriorStencil(**values)
    leaky = np.array(st.west)
    leaky[0, 0] = 0.0
    with pytest.raises(dz.DiscretizationError, match="NaN elsewhere"):
        dz.InteriorStencil(**{**values, "west": leaky})
    with pytest.raises(dz.DiscretizationError, match="shape"):
        dz.InteriorStencil(**{**values, "north": np.array(st.north)[:, :-1]})
    with pytest.raises(dz.DiscretizationError, match="dm"):
        dz.InteriorStencil(**{**values, "dm": 0.0})


# --- the periodic PDE line (build_periodic_pde_rows) ------------------------------------------

PERIODIC_LEVELS = (32, 64, 128, 256)    # cells per direction of the manufactured periodic lattices


@lru_cache(maxsize=None)
def periodic_case(n_passage: int = 101, n_eta: int = 31, mdot: float = 1.0, c_ref: float | None = None,
                  uniform: bool = True):
    metrics, co, _ = c2_case(n_passage, n_eta, mdot, c_ref, uniform)
    bcs = bc.build_boundary_conditions(metrics.grid)
    return metrics, co, bcs, dz.build_periodic_pde_rows(co, metrics, bcs)


def lattice_bcs(n_m: int, n_eta: int, first: int, last: int) -> bc.BoundaryConditions:
    """boundary_conditions.py's own classification and continuation on a manufactured lattice
    (its array core; no Grid is needed for the relations)."""
    region = np.full(n_m, gr.PASSAGE)
    region[:first], region[last + 1:] = gr.INLET, gr.EXIT
    return bc._boundary_conditions(None, np.arange(n_m, dtype=float), np.linspace(0.0, 1.0, n_eta), first, last,
                                   np.arange(first, last + 1), region)


def lattice_rows(k: dict, dm: float, deta: float, first: int, last: int, B=None):
    n_m, n_eta = k["A"].shape
    bcs = lattice_bcs(n_m, n_eta, first, last)
    return dz._periodic_pde_rows(k["A"], k["B"] if B is None else B, k["C"], k["D"], k["E"], dm, deta, first, last,
                                 lambda station: bcs.periodic_ghost(station, n_eta))


def quasi_periodic(M, H) -> dict:
    """psi = eta + g(m, eta), g periodic in eta with period 1 (so psi(m, eta + 1) = psi + 1),
    and its derivatives, written out by hand."""
    w = 2.0 * math.pi
    s1, c1 = np.sin(1.7 * M + 0.3), np.cos(1.7 * M + 0.3)
    s2, c2 = np.sin(2.3 * M), np.cos(2.3 * M)
    a, ae = np.cos(w * H), -w * np.sin(w * H)
    b, be = np.sin(w * H + 0.4), w * np.cos(w * H + 0.4)
    psi = H + 0.10 * s1 * a + 0.05 * c2 * b
    return dict(psi=psi,
                m=0.10 * 1.7 * c1 * a - 0.05 * 2.3 * s2 * b,
                e=1.0 + 0.10 * s1 * ae + 0.05 * c2 * be,
                mm=-0.10 * 1.7 ** 2 * s1 * a - 0.05 * 2.3 ** 2 * c2 * b,
                ee=-0.10 * w * w * s1 * a - 0.05 * w * w * c2 * b,
                me=0.10 * 1.7 * c1 * ae - 0.05 * 2.3 * s2 * be)


def row_dict(row) -> dict:
    return {t.node: t.coefficient for t in row.terms}


def expected_row(k: dict, i: int, J: int, dm: float, deta: float, c: dict, m: dict | None = None,
                 mm: dict | None = None) -> tuple[dict, float]:
    """A periodic row derived here: the raw Eq. 5-21 at (i, J) with psi_eta, psi_eta,eta of
    Eqs. 5-36, 5-38, psi_m = sum m_o psi[i + o, J] / (2 dm) and psi_m,m = sum mm_o psi[i + o, J] /
    dm^2 (central, Eqs. 5-35, 5-37, unless given), the mixed difference sum_k c_k q_k / (2 dm),
    q_k = (psi[k, J + 1] - psi[k, J - 1]) / (2 deta), then psi[k, J + 1] = psi[k, 1] + 1, then
    divided by -H, H = 2 A / deta^2 + 2 C / dm^2. Returns (coefficients, rhs)."""
    m = {-1: -1.0, 1: 1.0} if m is None else m
    mm = {-1: 1.0, 0: -2.0, 1: 1.0} if mm is None else mm
    A, B, C, D, E = (float(k[name][i, J]) for name in "ABCDE")
    H = 2.0 * A / deta ** 2 + 2.0 * C / dm ** 2
    raw: dict = {}
    constant = 0.0

    def put(node, value):
        raw[node] = raw.get(node, 0.0) + value

    put((i, J), -2.0 * A / deta ** 2)
    for offset, w in mm.items():
        put((i + offset, J), C * w / dm ** 2)
    for offset, w in m.items():
        put((i + offset, J), E * w / (2.0 * dm))
    put((i, J - 1), A / deta ** 2 - D / (2.0 * deta))
    north = A / deta ** 2 + D / (2.0 * deta)          # weight of psi[i, J + 1] = psi[i, 1] + 1
    put((i, 1), north)
    constant += north
    for offset, ck in c.items():                      # -2 B sum c_k (psi[k, J+1] - psi[k, J-1]) / (4 dm deta)
        w = -2.0 * B * ck / (4.0 * dm * deta)
        put((i + offset, 1), w)
        constant += w
        put((i + offset, J - 1), -w)
    return {node: -value / H for node, value in raw.items()}, constant / H     # raw: sum + constant = 0


CENTRAL, BACKWARD, FORWARD = {-1: -1.0, 1: 1.0}, {0: 3.0, -1: -4.0, -2: 1.0}, {0: -3.0, 1: 4.0, 2: -1.0}
FORWARD_MM = {0: 2.0, 1: -5.0, 2: 4.0, 3: -1.0}       # psi_m,m = (2 f0 - 5 f1 + 4 f2 - f3) / dm^2, second order


def test_the_interior_stencil_fingerprint():
    """InteriorStencil before the periodic rows were added (2026-11-12): the sums of its
    weights on four cases, taken from the pre-change module (the same arrays were also
    compared bit for bit by hash in the change itself). Re-taken on the quintic thickness
    (2026-11-13; the geometry moved, the module did not): the sums move by at most 6.5e-5
    relative (the legacy PCHIP values, (101, 31, None, True): -4486.175595377946,
    -4486.739866390357, -457.1472692525272, -458.9372689791712, 379.8909912820828,
    525133870.30980074)."""
    expected = {
        (101, 31, None, True): (-4486.178696911572, -4486.742539634658, -457.1419401877804, -458.9368232659904,
                                379.8662082621802, 525129507.3447217),
        (101, 31, 0.1, False): (-4485.25117204214, -4487.670064504089, -457.0687124098048, -459.01005104396603,
                                379.86620826218007, 5548714074.997126),
        (401, 121, None, True): (-73117.41836798034, -73118.20847689317, -7561.196604186591, -7567.176550939898,
                                 6248.840800026614, 135069107490.14485),
        (51, 11, 2.0, False): (-741.2266352164877, -740.9186746569694, -32.734560090705415, -33.1201300358374,
                               41.68126521596664, 10526409.193919398),
    }
    for (n_passage, n_eta, c_ref, uniform), sums in expected.items():
        _, _, st = c2_case(n_passage, n_eta, 1.7, c_ref, uniform)
        got = [float(np.nansum(getattr(st, name))) for name in ("west", "east", "south", "north", "mixed", "H")]
        np.testing.assert_allclose(got, sums, rtol=1e-13, atol=0.0)


@pytest.mark.parametrize("level", ((31, 5), (101, 31), (401, 121)))
def test_periodic_rows_sit_on_the_upper_line_of_the_periodic_stations(level):
    """One row per periodic station 1 .. first_blade_i - 1 and last_blade_i + 1 .. n_m - 2, at
    (i, n_eta - 1), the PDE side; none on a blade or far station. The transition kinds are
    exactly first_blade_i - 1 and last_blade_i + 1."""
    metrics, co, bcs, rows = periodic_case(*level)
    g = metrics.grid
    f, l, J = g.first_blade_i, g.last_blade_i, len(g.eta) - 1
    stations = [row.target[0] for row in rows]
    assert stations == list(range(1, f)) + list(range(l + 1, len(g.m) - 1))
    assert stations == [int(i) for i in bcs.periodic_stations if 0 < i < len(g.m) - 1]
    assert all(row.target == (row.target[0], J) for row in rows)
    for row in rows:
        if row.kind == dz.PERIODIC_TE_TRANSITION:          # its one-sided m differences weigh its own node
            i, dm, deta = row.target[0], g.dm, g.deta
            center = (2.0 * co.A[i, J] / deta ** 2 - 2.0 * co.C[i, J] / dm ** 2 + 3.0 * co.E[i, J] / (2.0 * dm)) / row.H
            assert abs(row.coefficient(row.target) - center) <= 1e-15
        else:
            assert row.coefficient(row.target) == 1.0
    kinds = {row.target[0]: row.kind for row in rows}
    assert kinds[f - 1] == dz.PERIODIC_LE_TRANSITION and kinds[l + 1] == dz.PERIODIC_TE_TRANSITION
    assert sum(kind != dz.PERIODIC_STANDARD for kind in kinds.values()) == 2


@pytest.mark.parametrize("c_ref", (None, 0.1))
def test_standard_rows_are_eq_5_21_with_the_periodic_continuation(c_ref):
    """Away from the transitions the row is Eq. 5-21 at (i, J) with Eqs. 5-35 to 5-39 and
    psi[k, J + 1] = psi[k, 1] + 1, divided by -H, derived here. Measured: coefficients within
    6.9e-18 (they are O(1)), rhs within 6.9e-18."""
    metrics, co, _, rows = periodic_case(101, 31, 1.3, c_ref, False)
    g = metrics.grid
    f, l, J = g.first_blade_i, g.last_blade_i, len(g.eta) - 1
    k = dict(A=co.A, B=co.B, C=co.C, D=co.D, E=co.E)
    by_station = {row.target[0]: row for row in rows}
    for i in (1, 2, 40, f - 3, f - 2, l + 2, l + 3, 300, len(g.m) - 2):
        coefficients, rhs = expected_row(k, i, J, co.length_scale * g.dm, g.deta, CENTRAL)
        row = by_station[i]
        assert row.kind == dz.PERIODIC_STANDARD
        assert set(row_dict(row)) == set(coefficients)
        assert max(abs(row_dict(row)[node] - value) for node, value in coefficients.items()) <= 4.5e-16
        assert abs(row.rhs - rhs) <= 1e-16


@pytest.mark.parametrize("c_ref", (None, 0.1))
def test_transition_rows_use_the_one_sided_mixed_difference(c_ref):
    """At first_blade_i - 1 the mixed term is the backward (3, -4, 1) difference of the periodic
    q, everything else as in a standard row. At last_blade_i + 1 all three m derivatives are
    forward one-sided: psi_m (-3, 4, -1) / (2 dm), psi_m,m (2, -5, 4, -1) / dm^2 and the mixed
    (-3, 4, -1) of q; psi_eta and psi_eta,eta as in a standard row. Derived here; measured:
    coefficients within 2.8e-17 (LE) and 1.4e-17 (TE, whose weights reach 2.3), rhs within
    1.4e-17."""
    metrics, co, _, rows = periodic_case(101, 31, 1.3, c_ref, False)
    g = metrics.grid
    f, l, J = g.first_blade_i, g.last_blade_i, len(g.eta) - 1
    k = dict(A=co.A, B=co.B, C=co.C, D=co.D, E=co.E)
    by_station = {row.target[0]: row for row in rows}
    for i, c, m, mm, kind, bound in ((f - 1, BACKWARD, None, None, dz.PERIODIC_LE_TRANSITION, 4.5e-16),
                                     (l + 1, FORWARD, FORWARD, FORWARD_MM, dz.PERIODIC_TE_TRANSITION, 4.5e-16)):
        coefficients, rhs = expected_row(k, i, J, co.length_scale * g.dm, g.deta, c, m, mm)
        row = by_station[i]
        assert row.kind == kind and set(row_dict(row)) == set(coefficients)
        assert max(abs(row_dict(row)[node] - value) for node, value in coefficients.items()) <= bound
        assert abs(row.rhs - rhs) <= 1e-16


def test_the_rhs_carries_the_jump_and_the_mixed_constants_cancel():
    """Q = 0, but every row's rhs is -north = (A / deta^2 + D / (2 deta)) / H, the jump moved
    across through psi_eta and psi_eta,eta (non-zero: measured 0.032 to 0.197 at n_passage
    101, 0.197 on the LE and 0.038 on the TE row); the mixed difference adds nothing (sum c = 0
    for all three: exact), the rhs with and without B agreeing to round-off (measured 6.9e-18)."""
    for table in dz.MIXED_M_DIFFERENCE.values():
        assert sum(c for _, c in table) == 0.0
    metrics, co, _, rows = periodic_case()
    g = metrics.grid
    J, dm, deta = len(g.eta) - 1, g.dm, g.deta
    for row in rows:
        i = row.target[0]
        A, D = co.A[i, J], co.D[i, J]
        assert abs(row.rhs - (A / deta ** 2 + D / (2.0 * deta)) / row.H) <= 1e-16
        assert row.rhs != 0.0
    k = dict(A=co.A, B=co.B, C=co.C, D=co.D, E=co.E)
    with_B = lattice_rows(k, dm, deta, g.first_blade_i, g.last_blade_i)
    without_B = lattice_rows(k, dm, deta, g.first_blade_i, g.last_blade_i, B=np.zeros_like(co.B))
    assert max(abs(a.rhs - b.rhs) for a, b in zip(with_B, without_B)) <= 1e-16


def test_the_transition_footprints_stay_out_of_the_blade():
    """LE row (first_blade_i - 1): its only node at a station >= first_blade_i is the upper wall
    node (f, J), read by psi_m and psi_m,m with the east weight of Eq. 5-46 alone; no (f, 1),
    no (f, J - 1), nothing of the mixed term. TE row (last_blade_i + 1): no node of station
    last_blade_i at all; its stations are l + 1 .. l + 4, those on its own line j = J (psi_m,
    psi_m,m) exactly l + 1 .. l + 4. The mixed term's nodes (where the row changes with B) lie
    at f - 3 .. f - 1 and l + 1 .. l + 3."""
    metrics, co, bcs, rows = periodic_case(101, 31, 1.0, None, False)
    g = metrics.grid
    f, l, J, dm, deta = g.first_blade_i, g.last_blade_i, len(g.eta) - 1, g.dm, g.deta
    by_station = {row.target[0]: row for row in rows}
    le, te = by_station[f - 1], by_station[l + 1]
    assert [node for node in le.nodes if node[0] >= f] == [(f, J)]
    assert [node for node in te.nodes if node[0] <= l] == []
    assert {node[0] for node in te.nodes} == {l + 1, l + 2, l + 3, l + 4}
    assert sorted(node for node in te.nodes if node[1] == J) == [(l + 1, J), (l + 2, J), (l + 3, J), (l + 4, J)]
    east = -(co.C[f - 1, J] / dm ** 2 + co.E[f - 1, J] / (2.0 * dm)) / le.H
    assert abs(le.coefficient((f, J)) - east) <= 1e-16
    k = dict(A=co.A, B=co.B, C=co.C, D=co.D, E=co.E)
    no_B = {row.target[0]: row for row in lattice_rows(k, dm, deta, f, l, B=np.zeros_like(co.B))}
    for row, stations in ((le, {f - 3, f - 2, f - 1}), (te, {l + 1, l + 2, l + 3})):
        base = no_B[row.target[0]]
        moved = {node for node in set(row.nodes) | set(base.nodes) if row.coefficient(node) != base.coefficient(node)}
        assert {node[0] for node in moved} == stations
        assert all(node[1] in (1, J - 1) for node in moved)


@pytest.mark.parametrize("kind", (dz.PERIODIC_LE_TRANSITION, dz.PERIODIC_TE_TRANSITION))
def test_a_central_mixed_difference_at_a_transition_is_refused(kind, monkeypatch):
    """The guard: were a transition row's mixed difference central, it would ask for the
    continuation on the first (last) blade station, and periodic_ghost refuses it."""
    metrics, co, bcs, _ = periodic_case()
    monkeypatch.setitem(dz.MIXED_M_DIFFERENCE, kind, ((-1, -1.0), (1, 1.0)))
    station = metrics.grid.first_blade_i if kind == dz.PERIODIC_LE_TRANSITION else metrics.grid.last_blade_i
    with pytest.raises(bc.BoundaryConditionError, match=f"station {station} is a blade station"):
        dz.build_periodic_pde_rows(co, metrics, bcs)


def periodic_errors(n: int, mixed_only: bool = False) -> tuple[float, dict]:
    """On an (n + 1)^2 lattice over [0, 1.2] x [0, 1] with blade stations at about 35 % to 65 %
    of m, the smooth coefficients and the quasi-periodic psi: the largest |discrete - exact|
    per row kind, of the whole operator (-H times the residual against L(psi)) or of
    psi_m,eta alone (from the change of the row with B, at B = 0.8 everywhere)."""
    dm, deta = M_LENGTH / n, 1.0 / n
    M, Hh = lattice(n + 1, n + 1, dm, deta)
    k, d = smooth_coefficients(M, Hh), quasi_periodic(M, Hh)
    first, last = int(round(0.35 * n)), int(round(0.65 * n))
    J = n
    errors = {}
    if mixed_only:
        B = np.full_like(M, 0.8)
        with_B, without_B = lattice_rows(k, dm, deta, first, last, B), lattice_rows(k, dm, deta, first, last, 0.0 * B)
        for a, b in zip(with_B, without_B):
            i = a.target[0]
            discrete = a.H * (a.residual(d["psi"]) - b.residual(d["psi"])) / (2.0 * 0.8)
            errors[a.kind] = max(errors.get(a.kind, 0.0), abs(discrete - d["me"][i, J]))
    else:
        for row in lattice_rows(k, dm, deta, first, last):
            i = row.target[0]
            exact = exact_operator({name: k[name][i, J] for name in "ABCDE"},
                                   {key: value[i, J] for key, value in d.items()})
            errors[row.kind] = max(errors.get(row.kind, 0.0), abs(-row.H * row.residual(d["psi"]) - exact))
    return dm, errors


@pytest.mark.parametrize("mixed_only", (False, True))
def test_periodic_rows_converge_at_second_order(mixed_only):
    """Smooth coefficients, the quasi-periodic psi, 32 to 256 cells: the LE (backward), TE
    (forward) and standard rows, the whole operator and the mixed derivative alone, converge
    at second order. Measured orders: whole operator, standard 1.999, 2.000, 2.000, LE 1.986,
    1.990, 1.997, TE (forward psi_m, psi_m,m, psi_m,eta) 2.003, 2.003, 2.001; mixed alone,
    standard 1.949, 1.971, 1.992 (slower to settle: 1.927 from 16 cells), LE 1.978, 1.966,
    1.995, TE 1.994, 1.989, 1.998."""
    results = [periodic_errors(n, mixed_only) for n in PERIODIC_LEVELS]
    spacings = [r[0] for r in results]
    for kind in dz.PERIODIC_ROW_KINDS:
        orders = observed_orders([r[1][kind] for r in results], spacings)
        assert min(orders) >= 1.94 and orders[-1] >= 1.99, (kind, orders)


@pytest.mark.parametrize("p", ((0.3, -1.2, 0.0), (0.0, 0.7, 2.5)))
def test_periodic_rows_are_exact_where_the_differences_are(p):
    """psi = eta + p(m) s(eta), p linear or quadratic, s = sin(2 pi eta): the q of every station
    is 1 + p(m_k) sin(2 pi deta) / deta, so each row's mixed derivative is exactly
    p'(m_i) sin(2 pi deta) / deta (the one-sided and central m differences are exact for
    quadratics); and for psi = eta + p(m) every row is exact: the residual is -L(psi) / H, the
    jump handled through the rhs. Measured: 1.6e-14 relative (mixed: round-off over dm deta),
    2.3e-16 (rows)."""
    n = 24
    dm, deta = M_LENGTH / n, 1.0 / n
    M, Hh = lattice(n + 1, n + 1, dm, deta)
    k = smooth_coefficients(M, Hh)
    first, last, J = 8, 16, n
    a0, a1, a2 = p
    poly, slope = a0 + a1 * M + a2 * M * M, a1 + 2.0 * a2 * M
    psi = Hh + poly * np.sin(2.0 * math.pi * Hh)
    B = np.full_like(M, 0.8)
    for a, b in zip(lattice_rows(k, dm, deta, first, last, B), lattice_rows(k, dm, deta, first, last, 0.0 * B)):
        i = a.target[0]
        discrete = a.H * (a.residual(psi) - b.residual(psi)) / (2.0 * 0.8)
        exact = slope[i, J] * math.sin(2.0 * math.pi * deta) / deta
        assert abs(discrete - exact) <= 4e-14 * max(1.0, abs(exact)), a.kind
    psi = Hh + poly
    for row in lattice_rows(k, dm, deta, first, last):
        i = row.target[0]
        exact = k["C"][i, J] * 2.0 * a2 + k["D"][i, J] + k["E"][i, J] * slope[i, J]
        assert abs(row.residual(psi) + exact / row.H) <= 4.5e-16 * max(1.0, np.abs(psi).max()), row.kind


@pytest.mark.parametrize("level", ((101, 31), (401, 121)))
def test_c2_periodic_rows_are_finite_and_deterministic(level):
    """The default C2 grid: every coefficient, rhs and H finite, center 1; rebuilt from fresh
    objects the rows are the same, bit for bit."""
    metrics, co, bcs, rows = periodic_case(*level)
    again = dz.build_periodic_pde_rows(cf.build_coefficients(metrics, np.ones(co.A.shape), 1.0), metrics,
                                       bc.build_boundary_conditions(metrics.grid))
    assert rows == again
    for row in rows:
        assert all(math.isfinite(t.coefficient) for t in row.terms) and math.isfinite(row.rhs) and row.H > 0.0


@pytest.mark.parametrize("lam", (0.05, 3.0))
def test_periodic_rows_do_not_scale(lam):
    """Like the interior weights, the rows are dimensionless: unchanged when the chord is
    scaled at fixed mdot', for similar twins and for mdot' alone (measured 1.7e-16 absolute,
    coefficients and rhs); normalized mode is c_ref = 1 bit for bit."""
    _, _, _, base = periodic_case(101, 31, 2.0, 0.1, False)
    for c_ref, mdot in ((0.1 * lam, 2.0), (0.1 * lam, 2.0 * lam), (0.1, 2.0 * lam)):
        _, _, _, other = periodic_case(101, 31, mdot, c_ref, False)
        for a, b in zip(base, other):
            assert a.nodes == b.nodes and abs(a.rhs - b.rhs) <= 1e-15
            assert max(abs(x.coefficient - y.coefficient) for x, y in zip(a.terms, b.terms)) <= 1e-15
    assert periodic_case(101, 31, 1.0, None, False)[3] == periodic_case(101, 31, 1.0, 1.0, False)[3]


def test_periodic_rows_inputs_are_checked():
    metrics, co, bcs, _ = periodic_case()
    other = bc.build_boundary_conditions(make_grid(cascade_of(SPEC), GridSettings(n_passage=101, n_eta=31), INLET))
    with pytest.raises(dz.DiscretizationError, match="another grid"):
        dz.build_periodic_pde_rows(co, metrics, other)
    for args in ((metrics, co, bcs), (co, metrics.grid, bcs), (co, metrics, metrics.grid)):
        with pytest.raises(TypeError):
            dz.build_periodic_pde_rows(*args)
    with pytest.raises(dz.DiscretizationError, match="do not belong"):
        dz.build_periodic_pde_rows(replace(co, c_ref=0.3), metrics, bcs)
    k = smooth_coefficients(*lattice(20, 9, 0.05, 0.125))
    ghost = lattice_bcs(20, 9, 5, 14)
    for first, last in ((2, 14), (5, 17), (5, 16)):            # 2 inlet; 2 and 3 exit periodic stations
        with pytest.raises(dz.DiscretizationError, match="three on the inlet side and four on the exit"):
            dz._periodic_pde_rows(k["A"], k["B"], k["C"], k["D"], k["E"], 0.05, 0.125, first, last,
                                  lambda s: ghost.periodic_ghost(s, 9))
    with pytest.raises(dz.DiscretizationError, match="ghost rule"):
        dz._periodic_pde_rows(k["A"], k["B"], k["C"], k["D"], k["E"], 0.05, 0.125, 5, 14,
                              lambda s: ghost.periodic_ghost(s, -1))


def test_a_periodic_row_is_checked():
    term = dz.PDETerm((3, 8), 1.0)
    dz.PeriodicPDERow(dz.PERIODIC_STANDARD, (3, 8), (term, dz.PDETerm((2, 8), -0.4)), 0.1, 2.0)
    for kind, terms, rhs, H, words in (("periodic", (term,), 0.0, 1.0, "kind"),
                                       (dz.PERIODIC_STANDARD, (term, term), 0.0, 1.0, "distinct"),
                                       (dz.PERIODIC_STANDARD, (term,), math.nan, 1.0, "non-finite"),
                                       (dz.PERIODIC_STANDARD, (term,), 0.0, 0.0, "non-finite"),
                                       (dz.PERIODIC_STANDARD, (dz.PDETerm((3, 8), 0.5),), 0.0, 1.0, "center")):
        with pytest.raises(dz.DiscretizationError, match=words):
            dz.PeriodicPDERow(kind, (3, 8), terms, rhs, H)


# --- the TE transition row: forward one-sided psi_m and psi_m,m (2026-11-12 revision) -------------

def te_m_derivative(n: int, p, which: str) -> tuple[float, float, float]:
    """The TE row's own psi_m (which = "m") or psi_m,m ("mm"), read through the row: on an
    (n + 1) x 9 lattice over [0, 1.2] x [0, 1] with A = 1, B = D = 0 and E = 1, C = 0 (or C = 1,
    E = 0), psi = eta + p(m) makes psi_eta,eta and the mixed term vanish exactly and psi_eta = 1
    with D = 0, so the row's residual is -E psi_m / H (-C psi_m,m / H). The last blade station
    is 0.65 n - 1, so for n a multiple of 20 the TE row sits at m = 0.78 on every lattice.
    Returns (dm, discrete, m at the TE row)."""
    dm, deta = M_LENGTH / n, 1.0 / 8
    M, Hh = lattice(n + 1, 9, dm, deta)
    zeros, ones = np.zeros_like(M), np.ones_like(M)
    k = dict(A=ones, B=zeros, C=zeros if which == "m" else ones, D=zeros, E=ones if which == "m" else zeros)
    first, last = int(round(0.35 * n)), int(round(0.65 * n)) - 1
    row = next(r for r in lattice_rows(k, dm, deta, first, last) if r.kind == dz.PERIODIC_TE_TRANSITION)
    psi = Hh + p(M)
    return dm, -row.H * row.residual(psi), float(M[row.target])


def test_the_te_row_reads_its_m_derivatives_forward():
    """With only E (only C) the TE row's line nodes are l + 1 .. l + 3 (l + 1 .. l + 4): psi_m
    by (-3, 4, -1) / (2 dm) and psi_m,m by (2, -5, 4, -1) / dm^2 (their Taylor series: the
    first is exact to quadratics, error -dm^2 f''' / 3; the second to cubics, error
    -11 dm^2 f'''' / 12), and nothing of station l."""
    n = 20
    dm, deta = M_LENGTH / n, 1.0 / 8
    M, _ = lattice(n + 1, 9, dm, deta)
    zeros, ones = np.zeros_like(M), np.ones_like(M)
    first, last = int(round(0.35 * n)), int(round(0.65 * n)) - 1
    for C, E, stations in ((zeros, ones, {last + 1, last + 2, last + 3}), (ones, zeros, {last + 1, last + 2, last + 3,
                                                                                       last + 4})):
        k = dict(A=ones, B=zeros, C=C, D=zeros, E=E)
        row = next(r for r in lattice_rows(k, dm, deta, first, last) if r.kind == dz.PERIODIC_TE_TRANSITION)
        assert {node[0] for node in row.nodes if node[1] == 8} == stations


@pytest.mark.parametrize("which", ("m", "mm"))
def test_the_te_m_derivatives_are_exact_for_low_order_polynomials(which):
    """psi_m is exact for p quadratic, psi_m,m for p cubic. Measured: 6.3e-15 and 3.6e-13
    relative (round-off of differences over 2 dm and dm^2, dm = 0.06)."""
    p, exact = ((lambda m: 0.3 - 1.2 * m + 2.5 * m * m, lambda m: -1.2 + 5.0 * m) if which == "m" else
                (lambda m: 0.3 - 1.2 * m + 2.5 * m * m - 0.8 * m ** 3, lambda m: 5.0 - 4.8 * m))
    _, discrete, m = te_m_derivative(20, p, which)
    assert abs(discrete - exact(m)) <= (1e-13 if which == "m" else 1e-11) * max(1.0, abs(exact(m)))


@pytest.mark.parametrize("which", ("m", "mm"))
def test_the_te_m_derivatives_converge_at_second_order(which):
    """p = sin(2 m + 0.3), 20 to 320 cells, the TE row at m = 0.78 on each: measured orders
    psi_m 2.17, 2.10, 2.05, 2.03 (from above), psi_m,m 1.96, 1.98, 1.99, 2.00."""
    exact = (lambda m: 2.0 * math.cos(2.0 * m + 0.3)) if which == "m" else (lambda m: -4.0 * math.sin(2.0 * m + 0.3))
    errors, spacings = [], []
    for n in (20, 40, 80, 160, 320):
        dm, discrete, m = te_m_derivative(n, lambda x: np.sin(2.0 * x + 0.3), which)
        errors.append(abs(discrete - exact(m)))
        spacings.append(dm)
    orders = observed_orders(errors, spacings)
    assert min(orders) >= 1.9 and abs(orders[-1] - 2.0) <= 0.05, orders


@lru_cache(maxsize=None)
def blade_loop() -> np.ndarray:
    """The closed outline of the reference C2 blade: 20001 samples of each curve (TE -> LE on
    the suction curve, LE -> TE on the pressure curve)."""
    curves = cascade_of(SPEC).active_curves
    u = np.linspace(0.0, 1.0, 20001)
    return np.vstack([np.asarray(curves.suction.derivative(u[::-1], 0)), np.asarray(curves.pressure.derivative(u[1:], 0))])


def metal_length(a, b) -> float:
    """Length of the segment a -> b (a in the fluid) inside the C2 blade or a pitch copy, from
    its crossings with the sampled outline (a polyline intersection, independent of the
    audit scripts' spline-root check): the pieces after an odd crossing. Crossings within
    1e-9 of an end are the end itself (a link may end on a wall node)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    d = b - a
    length = float(np.hypot(*d))
    pitch = cascade_of(SPEC).active.pitch
    total = 0.0
    for copy in (-1, 0, 1, 2):
        P = blade_loop() + np.array([0.0, copy * pitch])
        P0, e = P[:-1], P[1:] - P[:-1]
        w = P0 - a
        with np.errstate(all="ignore"):
            denom = d[0] * e[:, 1] - d[1] * e[:, 0]
            t = (w[:, 0] * e[:, 1] - w[:, 1] * e[:, 0]) / denom
            s = (w[:, 0] * d[1] - w[:, 1] * d[0]) / denom
        cuts = sorted(t[(denom != 0.0) & (s >= 0.0) & (s < 1.0) & (t > 1e-9) & (t < 1.0 - 1e-9)])
        if len(cuts) % 2:
            cuts.append(1.0)                                  # it ends on the wall from inside
        total += sum(cuts[q + 1] - cuts[q] for q in range(0, len(cuts), 2)) * length
    return total


@pytest.mark.parametrize("level", ((61, 21), (101, 41), (201, 41), (401, 41), (401, 121)))
def test_the_transition_rows_stay_in_the_fluid(level):
    """Every straight chord of the LE and the TE row, from its node to each node it reads (a
    continued node (k, 1) at its place one pitch up), intersects no metal of the blade or its
    pitch copies. The old central TE chord (l + 1, J) -> (l, J) does: measured 6.3e-3,
    5.4e-3, 2.0e-3, 5.9e-3, 5.9e-3 chord at these five grids. A geometric fact about the
    straight chords, not by itself a stencil-validity criterion: the differences act along
    the coordinate curves of the mapping (scripts/audit_transition_mapping.py)."""
    metrics, _, _, rows = periodic_case(*level)
    g = metrics.grid
    f, l, J = g.first_blade_i, g.last_blade_i, len(g.eta) - 1
    by_station = {row.target[0]: row for row in rows}
    for row in (by_station[f - 1], by_station[l + 1]):
        start = g.nodes[row.target]
        for node in row.nodes:
            if node == row.target:
                continue
            end = g.nodes[node] + (np.array([0.0, g.pitch]) if node[1] == 1 else 0.0)
            assert metal_length(start, end) <= 1e-7, (row.kind, node)
    assert metal_length(g.nodes[l + 1, J], g.nodes[l, J]) >= 1.5e-3


def test_the_le_and_standard_rows_are_unchanged_by_the_te_revision():
    """Values of the LE row and of all standard rows from the module before the TE revision
    (bit for bit: compared by hash in the change itself), n_passage 101 (n_eta 31) and 401 (41):
    the LE row's nodes, the sum of |coefficients|, rhs and H; the standard rows' count and sums.
    Of the quintic thickness (2026-11-13) only the LE row's rhs moved: 0.19686130200463425 and
    0.07826755322366549 with the legacy PCHIP (it carries the first blade station's wall
    psi); the rest is the same to the last bit or two."""
    expected = {(101, 31): ((119, 30), 9, 2.6080486917691648, 0.19797276505127065, 33057.11078903566,
                            237, 510.37332854365025, 10.644875005575187, 7565847.866769484),
                (401, 41): ((475, 40), 9, 2.2292236938912913, 0.0791057371409257, 464209.5382632994,
                            951, 1956.0564768654688, 5.286205544200261, 439544531.45481366)}
    for level, (target, n_terms, total, rhs, H, count, std_total, std_rhs, std_H) in expected.items():
        _, _, _, rows = periodic_case(*level)
        le = next(r for r in rows if r.kind == dz.PERIODIC_LE_TRANSITION)
        standard = [r for r in rows if r.kind == dz.PERIODIC_STANDARD]
        assert le.target == target and len(le.terms) == n_terms
        np.testing.assert_allclose([sum(abs(t.coefficient) for t in le.terms), le.rhs, le.H], [total, rhs, H],
                                   rtol=1e-13, atol=0.0)
        assert len(standard) == count
        np.testing.assert_allclose([sum(sum(abs(t.coefficient) for t in r.terms) for r in standard),
                                    sum(r.rhs for r in standard), sum(r.H for r in standard)], [std_total, std_rhs, std_H],
                                   rtol=1e-13, atol=0.0)
