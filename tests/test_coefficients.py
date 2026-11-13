"""
Tests for potential_flow_solver.coefficients: the fields A, B, C, D, E of Aungier's Eq. 5-21
(Eqs. 5-22 to 5-26) from GridMetrics, a frozen density field and the mass flow per unit
depth, in physical (c_ref) or normalized (c_ref = None) mode.

Every reference is independent of coefficients.py: the closed form of a manufactured field,
the hand-expanded D and E with metrics.py's exact C2 derivatives, the stencil footprint
written out by hand, mu sqrt(g) g^ij of the (m, eta) map from a matrix inverse, the
conservative form of Eq. 5-20, or dimensional analysis. Each tolerance names the measured
value it covers.

Manufactured fields sit on real grids of the default C2 cascade (65-(12A10)10 at 33 deg,
inlet 45 deg) through a GridMetrics whose S, tan_beta, beta, sin_beta and cos_beta are
replaced: coefficients.py reads only S, tan_beta, cos_beta and the grid (checked), so the
real m, eta, dm, deta and blade-station topology stay, transitions, walls and periodic lines
included. They are periodic in eta (the periodic stations need it) and smooth in m.

Checks
API and inputs
- the public names, the fields of CoefficientFields, Coefficients is CoefficientFields (the
  name discretization.py imports), frozen and read-only; the module imports numpy and
  metrics only, reads S, tan_beta, cos_beta and the grid, and has no psi, stencil or matrix
- refused: rho of the wrong shape, non-numeric, NaN, Inf, zero or negative, or not periodic
  on the periodic stations (a round-off difference accepted); mass_flow_per_depth and c_ref
  zero, negative, NaN, Inf or not a real number; anything but a GridMetrics; a grid of a
  physical_chord cascade; n_eta = 2; direct CoefficientFields of mixed shapes or non-finite
A, B, C
- A, B, C = mu sqrt(g) (g^ee, -g^me, g^mm) of the (m, eta) map, the inverse metric from a
  2x2 inverse at every node of the default C2 grid, with S = L_ref S*
- constant rectangular case (S* const, beta = 0, rho const): B = D = E = 0 exactly, A and C
  the closed forms; constant skew case (beta = 37 deg): A, B, C constant, D = E = 0
D, E
- exact (to round-off) for fields quadratic in m, at every node, far stations and
  transitions included; exact for 1/rho quadratic in eta at the walls and inside, and zero
  at the periodic lines, where the periodic extension of that field has a symmetric kink
- the periodic lines: j = 0 and n_eta - 1 get the same D and E, the central difference with
  the neighbors n_eta - 2 below and 1 above (an analytic value), second order
- the stencil footprint: a density change at one node changes D and E exactly at the
  nodes listed here by hand (periodic wrap, one-sided walls, the m stencil across a
  transition, the far stations)
- second order on four grid levels: density in m only, in eta only, smooth geometry,
  everything together, in physical and normalized mode; no excess at the transitions
- signs: the expanded operator with D and E equals the conservative form of Eq. 5-20 to
  second order, for an analytic psi
- default C2 case: against the hand-expanded D and E with metrics' exact derivatives
  (rho = 1 and an analytic rho): second order on the smooth rear of the blade, the front
  bounded by the measured curvature-step values; on the extensions D is minus the side
  line's d tan(beta)/dm and E round-off
Units and scaling
- c_ref = None is c_ref = 1 bit for bit, and says so (physical_scale_defined False)
- fixed mdot': A ~ L^-1, B ~ L^0, C ~ L, D ~ L^-1, E ~ L^0, every term of Eq. 5-21 ~ L^-1
- physically similar twins (mdot' ~ L): A, D ~ L^0, B, E ~ L, C ~ L^2, every term unchanged
- dimensional analysis: new units of mass, length and time multiply each field by the
  factor of its dimension (A, D: L/T; B, E: L^2/T; C: L^3/T)
Default C2 case, n_passage 101
- finite everywhere; the first and last five blade stations finite and unclipped, A there
  1 / (S cos^2 beta) of the metric angle (79.25 deg at the first lower wall node)
"""

import ast
import math
import sys
from dataclasses import FrozenInstanceError, fields, replace
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from potential_flow_solver import coefficients as cf
from potential_flow_solver import metrics as mt
from potential_flow_solver.config import GridSettings
from potential_flow_solver.geometry import CascadeSpec, make_cascade
from potential_flow_solver.grid import make_grid

SPEC = CascadeSpec(blade_type="NACA65", solidity=1.0, stagger_deg=33.0, cl0=1.2, t_max_over_c=0.10,
                   surface_continuity="C2")
INLET = math.radians(45.0)
LEVELS = ((100, 33), (200, 65), (400, 129))           # (n_passage, n_eta): deta halves, dm about halves
S0, T0, RHO0 = 0.9, math.tan(math.radians(30.0)), 1.3  # manufactured base values
MDOT = 2.7                                             # mass flow per depth of the manufactured tests


@lru_cache(maxsize=None)
def cascade_of(spec: CascadeSpec):
    return make_cascade(spec)


@lru_cache(maxsize=None)
def metrics_of(n_passage: int = 101, n_eta: int = 31, spec: CascadeSpec = SPEC) -> mt.GridMetrics:
    return mt.build_metrics(make_grid(cascade_of(spec), GridSettings(n_passage=n_passage, n_eta=n_eta), INLET))


def ones(metrics: mt.GridMetrics) -> np.ndarray:
    return np.ones((len(metrics.grid.m), len(metrics.grid.eta)))


def blade_mask(metrics: mt.GridMetrics) -> np.ndarray:
    g = metrics.grid
    mask = np.zeros(len(g.m), dtype=bool)
    mask[g.first_blade_i:g.last_blade_i + 1] = True
    return mask


def with_geometry(metrics: mt.GridMetrics, S_station: np.ndarray, tan_beta: np.ndarray) -> mt.GridMetrics:
    """The real grid's GridMetrics with a manufactured S* (n_m,) and tan(beta) (n_m, n_eta);
    beta, sin and cos follow from tan(beta). The fields coefficients.py does not read
    (dS_dm, the boundary slopes, the dtan_beta fields) keep their real values."""
    tan_beta = np.broadcast_to(np.asarray(tan_beta, dtype=float), (len(metrics.grid.m), len(metrics.grid.eta)))
    cos = 1.0 / np.hypot(1.0, tan_beta)
    return replace(metrics, S=np.asarray(S_station, dtype=float), tan_beta=tan_beta, beta=np.arctan(tan_beta),
                   sin_beta=tan_beta * cos, cos_beta=cos)


def expanded(S, S_m, t, t_m, t_eta, rho, rho_m, rho_eta, mdot: float, length: float):
    """D and E of Eqs. 5-25, 5-26 expanded by the product rule, t = tan(beta), m derivatives
    in m* (so the length enters once, here):
        A_eta = mdot / (L S*) [2 t t_eta / rho - (1 + t^2) rho_eta / rho^2]
        B_m   = mdot / L [t_m / rho - t rho_m / rho^2]
        C_m   = mdot [S_m / rho - S* rho_m / rho^2]
        B_eta = mdot [t_eta / rho - t rho_eta / rho^2]"""
    A_eta = mdot / (length * S) * (2.0 * t * t_eta / rho - (1.0 + t * t) * rho_eta / rho ** 2)
    B_m = mdot / length * (t_m / rho - t * rho_m / rho ** 2)
    C_m = mdot * (S_m / rho - S * rho_m / rho ** 2)
    B_eta = mdot * (t_eta / rho - t * rho_eta / rho ** 2)
    return A_eta - B_m, C_m - B_eta


def manufactured(metrics: mt.GridMetrics, s: float = 0.0, t: float = 0.0, r_m: float = 0.0, r_eta: float = 0.0,
                 r_mix: float = 0.0) -> dict:
    """Smooth fields on the grid's nodes, periodic in eta, with u = 2 pi (m* - m_in) / (m_out - m_in)
    and w = 2 pi eta:
        S*  = S0 (1 + s sin u)
        t   = T0 + t (2 cos u + sin u sin(w + 0.7))
        rho = RHO0 (1 + r_m sin(u + 0.3) + r_eta (sin w + 0.5 cos 2w) + r_mix cos u sin(w + 0.5)),
    and their derivatives in m* and eta, written out by hand."""
    g = metrics.grid
    M, H = np.meshgrid(g.m, g.eta, indexing="ij")
    k = 2.0 * math.pi / (g.m[-1] - g.m[0])
    u, w = k * (M - g.m[0]), 2.0 * math.pi * H
    out = dict(S=S0 * (1.0 + s * np.sin(u)), S_m=S0 * s * k * np.cos(u),
               t=T0 + t * (2.0 * np.cos(u) + np.sin(u) * np.sin(w + 0.7)),
               t_m=t * k * (-2.0 * np.sin(u) + np.cos(u) * np.sin(w + 0.7)),
               t_eta=t * np.sin(u) * 2.0 * math.pi * np.cos(w + 0.7),
               rho=RHO0 * (1.0 + r_m * np.sin(u + 0.3) + r_eta * (np.sin(w) + 0.5 * np.cos(2.0 * w))
                           + r_mix * np.cos(u) * np.sin(w + 0.5)),
               rho_m=RHO0 * (r_m * k * np.cos(u + 0.3) - r_mix * k * np.sin(u) * np.sin(w + 0.5)),
               rho_eta=RHO0 * (r_eta * 2.0 * math.pi * (np.cos(w) - np.sin(2.0 * w))
                               + r_mix * np.cos(u) * 2.0 * math.pi * np.cos(w + 0.5)))
    out["metrics"] = with_geometry(metrics, out["S"][:, 0], out["t"])
    return out


def manufactured_errors(case: dict, c_ref: float | None, levels=LEVELS) -> list[tuple[float, float, float, float]]:
    """(dm*, deta, max |D - D_ref|, max |E - E_ref|) over every node, per grid level."""
    rows = []
    for n_passage, n_eta in levels:
        f = manufactured(metrics_of(n_passage, n_eta), **case)
        co = cf.build_coefficients(f["metrics"], f["rho"], MDOT, c_ref=c_ref)
        length = 1.0 if c_ref is None else c_ref
        D_ref, E_ref = expanded(f["S"], f["S_m"], f["t"], f["t_m"], f["t_eta"], f["rho"], f["rho_m"], f["rho_eta"],
                                MDOT, length)
        g = f["metrics"].grid
        rows.append((g.dm, g.deta, float(np.abs(co.D - D_ref).max()), float(np.abs(co.E - E_ref).max())))
    return rows


def observed_orders(errors, spacings) -> list[float]:
    return [math.log(errors[k] / errors[k + 1]) / math.log(spacings[k] / spacings[k + 1])
            for k in range(len(errors) - 1)]


# --- API and inputs ---------------------------------------------------------------------

def test_the_public_names():
    tree = ast.parse(Path(cf.__file__).read_text(encoding="utf-8"))
    functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef) and not node.name.startswith("_")}
    classes = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
    assert functions == {"build_coefficients"}
    assert classes == {"CoefficientError", "CoefficientFields"}
    assert cf.Coefficients is cf.CoefficientFields
    assert issubclass(cf.CoefficientError, ValueError)


def test_discretization_still_imports_its_name():
    """discretization.py (an interface, not changed here) imports Coefficients."""
    from potential_flow_solver import discretization
    assert discretization.Coefficients is cf.CoefficientFields


def test_the_fields_and_their_units_are_stated():
    assert [item.name for item in fields(cf.CoefficientFields)] == ["A", "B", "C", "D", "E", "mass_flow_per_depth",
                                                                    "c_ref"]
    source = Path(cf.__file__).read_text(encoding="utf-8")
    for line in ("A: Field                        # (n_m, n_eta), L/T:", "B: Field                        # (n_m, n_eta), L^2/T:",
                 "C: Field                        # (n_m, n_eta), L^3/T:", "D: Field                        # (n_m, n_eta), L/T:",
                 "E: Field                        # (n_m, n_eta), L^2/T:"):
        assert line in source


def test_the_fields_are_frozen_and_read_only():
    co = cf.build_coefficients(metrics_of(), ones(metrics_of()), 1.0)
    for name in "ABCDE":
        array = getattr(co, name)
        assert array.shape == (len(metrics_of().grid.m), len(metrics_of().grid.eta))
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array[0, 0] = 0.0
    with pytest.raises(FrozenInstanceError):
        co.A = co.B


def test_the_module_reads_metrics_and_nothing_else():
    """coefficients.py imports numpy and metrics.py only; of the metrics it reads S, tan_beta,
    cos_beta and the grid (no dS_dm or dtan_beta field, no geometry); of the grid its chord,
    m, eta, spacings and blade stations; and it has no psi, stencil, matrix or solve."""
    tree = ast.parse(Path(cf.__file__).read_text(encoding="utf-8"))
    imported = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    imported |= {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
    outside = {name for name in imported if name.split(".")[0] not in sys.stdlib_module_names}
    assert outside == {"numpy", "numpy.typing", "potential_flow_solver.metrics"}
    on = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            on.setdefault(node.value.id, set()).add(node.attr)
    assert on["metrics"] <= {"grid", "S", "tan_beta", "cos_beta"}
    assert on["grid"] <= {"chord", "m", "eta", "dm", "deta", "first_blade_i", "last_blade_i"}
    attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert not {"dS_dm", "dtan_beta_dm", "dtan_beta0_dm", "dtan_beta1_dm", "tan_beta0", "tan_beta1", "beta",
                "wall_identity", "boundaries", "wall_curve_derivative", "nodes"} & attributes
    assert not {"make_cascade", "make_grid", "Cascade", "psi", "assemble", "stencil", "csr_matrix"} & names


def test_mdot_and_c_ref_are_validated():
    metrics, rho = metrics_of(), ones(metrics_of())
    for bad in (0.0, -1.0, math.nan, math.inf, -math.inf):
        with pytest.raises(cf.CoefficientError, match="mass_flow_per_depth"):
            cf.build_coefficients(metrics, rho, bad)
        with pytest.raises(cf.CoefficientError, match="c_ref"):
            cf.build_coefficients(metrics, rho, 1.0, c_ref=bad)
    for bad in ("1.0", None, True, np.ones(1)):
        with pytest.raises(TypeError, match="mass_flow_per_depth"):
            cf.build_coefficients(metrics, rho, bad)
    for bad in ("0.1", True, np.ones(1)):
        with pytest.raises(TypeError, match="c_ref"):
            cf.build_coefficients(metrics, rho, 1.0, c_ref=bad)
    with pytest.raises(TypeError):
        cf.build_coefficients(metrics, rho, 1.0, 0.1)          # c_ref is keyword only
    assert cf.build_coefficients(metrics, rho, np.float64(2.0), c_ref=np.int64(3)).c_ref == 3.0


def test_rho_is_validated():
    metrics = metrics_of()
    rho = ones(metrics)
    n_m, n_eta = rho.shape
    for bad in (rho[:-1], rho[:, :-1], rho.T, rho.ravel(), 1.0):
        with pytest.raises(cf.CoefficientError, match="shape"):
            cf.build_coefficients(metrics, bad, 1.0)
    with pytest.raises(cf.CoefficientError, match="numeric"):
        cf.build_coefficients(metrics, np.full(rho.shape, "x"), 1.0)
    for value, words in ((math.nan, "not finite"), (math.inf, "not finite"), (-math.inf, "not finite"),
                         (0.0, "strictly positive"), (-1.0, "strictly positive")):
        bad = rho.copy()
        bad[n_m // 2, n_eta // 2] = value
        with pytest.raises(cf.CoefficientError, match=words):
            cf.build_coefficients(metrics, bad, 1.0)


def test_rho_must_be_periodic_on_the_periodic_stations():
    """On the periodic stations eta = 0 and eta = 1 are one point and its image: rho must agree
    there to ROUND_OFF max(rho). A blade station may differ (two walls); a round-off
    difference on a periodic station is accepted."""
    metrics = metrics_of()
    g = metrics.grid
    for i in (0, g.first_blade_i - 1, g.last_blade_i + 1, len(g.m) - 1):
        bad = ones(metrics)
        bad[i, -1] = 1.0 + 1e-9
        with pytest.raises(cf.CoefficientError, match=f"periodic station {i}"):
            cf.build_coefficients(metrics, bad, 1.0)
        fine = ones(metrics)
        fine[i, -1] = 1.0 + 8.0 * mt.EPSILON
        cf.build_coefficients(metrics, fine, 1.0)
    walls = ones(metrics)
    walls[g.first_blade_i, -1] = walls[g.last_blade_i, 0] = 1.5
    cf.build_coefficients(metrics, walls, 1.0)


def test_only_gridmetrics_is_accepted():
    for item in (metrics_of().grid, None, {"S": 1.0}):
        with pytest.raises(TypeError, match="GridMetrics"):
            cf.build_coefficients(item, ones(metrics_of()), 1.0)


def test_a_dimensional_grid_is_refused():
    """A physical_chord cascade's grid is already in metres; c_ref would scale it twice."""
    metrics = metrics_of(101, 31, replace(SPEC, physical_chord=0.1))
    for c_ref in (None, 0.1):
        with pytest.raises(cf.CoefficientError, match="chord-normalized"):
            cf.build_coefficients(metrics, ones(metrics), 1.0, c_ref=c_ref)


def test_two_eta_nodes_are_refused():
    metrics = metrics_of(31, 2)
    with pytest.raises(cf.CoefficientError, match="three"):
        cf.build_coefficients(metrics, ones(metrics), 1.0)


def test_metric_fields_that_break_the_formulas_are_refused():
    metrics = metrics_of()
    S = np.array(metrics.S)
    S[5] = 0.0
    with pytest.raises(cf.CoefficientError, match="S\\*"):
        cf.build_coefficients(replace(metrics, S=S), ones(metrics), 1.0)
    cos = np.array(metrics.cos_beta)
    cos[5, 3] = 0.0
    with pytest.raises(cf.CoefficientError, match="cos\\(beta\\)"):
        cf.build_coefficients(replace(metrics, cos_beta=cos), ones(metrics), 1.0)
    cos[5, 3] = 1e-160                                         # finite and positive, but 1/cos^2 overflows
    with pytest.raises(cf.CoefficientError, match="A is not finite"):
        cf.build_coefficients(replace(metrics, cos_beta=cos), ones(metrics), 1.0)


def test_direct_construction_is_checked():
    field = np.ones((4, 3))
    good = dict(A=field, B=field, C=field, D=field, E=field, mass_flow_per_depth=1.0, c_ref=None)
    cf.CoefficientFields(**good)
    with pytest.raises(cf.CoefficientError, match="shape"):
        cf.CoefficientFields(**{**good, "E": np.ones((4, 2))})
    with pytest.raises(cf.CoefficientError, match="shape"):
        cf.CoefficientFields(**{**good, "A": np.ones(4)})
    with pytest.raises(cf.CoefficientError, match="not finite"):
        cf.CoefficientFields(**{**good, "D": np.full((4, 3), np.nan)})
    with pytest.raises(cf.CoefficientError, match="mass_flow_per_depth"):
        cf.CoefficientFields(**{**good, "mass_flow_per_depth": 0.0})
    with pytest.raises(cf.CoefficientError, match="c_ref"):
        cf.CoefficientFields(**{**good, "c_ref": -1.0})


# --- A, B, C ----------------------------------------------------------------------------

@pytest.mark.parametrize("c_ref", (None, 0.08))
def test_A_B_C_are_mu_sqrt_g_times_the_inverse_metric(c_ref):
    """With x(m, eta) = (m, y0(m) + eta S(m)) the covariant basis is x_m = (1, tan beta),
    x_eta = (0, S), so Eq. 5-20 is d_i(mu sqrt(g) g^ij d_j psi): A = mu sqrt(g) g^ee,
    B = -mu sqrt(g) g^me, C = mu sqrt(g) g^mm, mu = mdot'/rho. The metric tensor is
    inverted here as a 2x2 matrix at every node, in physical lengths (S = L S*).
    Measured (normalized / c_ref 0.08), over each field's largest value: 2.5e-15 / 8.3e-16
    (A), 2.5e-15 / 1.2e-15 (B), 1.8e-15 / 1.0e-15 (C), the round-off of the inverse."""
    metrics = metrics_of()
    rho = manufactured(metrics, r_m=0.1, r_eta=0.1, r_mix=0.05)["rho"]
    co = cf.build_coefficients(metrics, rho, MDOT, c_ref=c_ref)
    length = 1.0 if c_ref is None else c_ref
    t = metrics.tan_beta
    S = length * np.broadcast_to(metrics.S[:, None], t.shape)
    g = np.empty(t.shape + (2, 2))
    g[..., 0, 0], g[..., 0, 1], g[..., 1, 0], g[..., 1, 1] = 1.0 + t * t, t * S, t * S, S * S
    inverse, root = np.linalg.inv(g), np.sqrt(np.linalg.det(g))
    mu = MDOT / rho
    for value, reference, bound in ((co.A, mu * root * inverse[..., 1, 1], 5e-15),
                                    (co.B, -mu * root * inverse[..., 0, 1], 5e-15),
                                    (co.C, mu * root * inverse[..., 0, 0], 4e-15)):
        assert np.max(np.abs(value - reference) / np.abs(reference).max()) <= bound


@pytest.mark.parametrize("c_ref", (None, 0.25))
def test_constant_rectangular_geometry(c_ref):
    """S* constant, beta = 0, rho constant: B = 0 exactly, A = mdot'/(L S* rho), C = mdot' L S* / rho
    (measured: exact), and D = E = 0 but for round-off: the far stations' one-sided
    -3 f + 4 f - f of a constant leaves a few ulp over 2 dm (measured |D| 0, |E| 2.7e-14)."""
    metrics = metrics_of()
    n_m, n_eta = ones(metrics).shape
    co = cf.build_coefficients(with_geometry(metrics, np.full(n_m, S0), np.zeros((n_m, n_eta))),
                               np.full((n_m, n_eta), RHO0), MDOT, c_ref=c_ref)
    length = 1.0 if c_ref is None else c_ref
    np.testing.assert_allclose(co.A, MDOT / (length * S0 * RHO0), rtol=2.3e-16, atol=0.0)
    np.testing.assert_allclose(co.C, MDOT * length * S0 / RHO0, rtol=2.3e-16, atol=0.0)
    assert np.all(co.B == 0.0)
    assert np.abs(co.D).max() <= 1e-13 and np.abs(co.E).max() <= 1e-13


@pytest.mark.parametrize("c_ref", (None, 0.25))
def test_constant_skew_geometry(c_ref):
    """S*, beta = 37 deg and rho constant: A, B, C are one number each (measured: exact), so
    D = E = 0 but for the round-off of the far stations' one-sided differences (measured
    |D| 1.3e-14 / 5.4e-14 with c_ref 0.25, |E| 2.7e-14)."""
    metrics = metrics_of()
    n_m, n_eta = ones(metrics).shape
    t = math.tan(math.radians(37.0))
    co = cf.build_coefficients(with_geometry(metrics, np.full(n_m, S0), np.full((n_m, n_eta), t)),
                               np.full((n_m, n_eta), RHO0), MDOT, c_ref=c_ref)
    length = 1.0 if c_ref is None else c_ref
    np.testing.assert_allclose(co.A, MDOT * (1.0 + t * t) / (length * S0 * RHO0), rtol=4.5e-16, atol=0.0)
    np.testing.assert_allclose(co.B, MDOT * t / RHO0, rtol=2.3e-16, atol=0.0)
    np.testing.assert_allclose(co.C, MDOT * length * S0 / RHO0, rtol=2.3e-16, atol=0.0)
    for name in "ABC":
        assert np.ptp(getattr(co, name)) == 0.0
    assert np.abs(co.D).max() <= 2e-13 and np.abs(co.E).max() <= 2e-13


# --- D, E: exactness, periodic lines, footprint ---------------------------------------

@pytest.mark.parametrize("c_ref", (None, 0.3))
def test_D_and_E_are_exact_for_fields_quadratic_in_m(c_ref):
    """S* = 0.9 + 0.05 m* + 0.02 m*^2, tan(beta) = 0.5 + 0.3 m* - 0.1 m*^2, rho constant:
    B and C are quadratic in m, A has no eta variation, so the central and one-sided
    differences are exact at every node, the far stations and the transitions included:
    D = -mdot' (0.3 - 0.2 m*) / (L rho), E = mdot' (0.05 + 0.04 m*) / rho.
    Measured, over the largest reference value: 3.7e-14 (D), 2.4e-13 (E), the round-off of
    differences over 2 dm."""
    metrics = metrics_of()
    m = metrics.grid.m
    n_eta = len(metrics.grid.eta)
    S = 0.9 + 0.05 * m + 0.02 * m * m
    t = np.repeat((0.5 + 0.3 * m - 0.1 * m * m)[:, None], n_eta, axis=1)
    co = cf.build_coefficients(with_geometry(metrics, S, t), np.full(t.shape, RHO0), MDOT, c_ref=c_ref)
    length = 1.0 if c_ref is None else c_ref
    D_ref = np.repeat((-MDOT * (0.3 - 0.2 * m) / (length * RHO0))[:, None], n_eta, axis=1)
    E_ref = np.repeat((MDOT * (0.05 + 0.04 * m) / RHO0)[:, None], n_eta, axis=1)
    assert np.abs(co.D - D_ref).max() / np.abs(D_ref).max() <= 5e-14
    assert np.abs(co.E - E_ref).max() / np.abs(E_ref).max() <= 4e-13


def test_D_and_E_are_exact_for_an_eta_quadratic_and_periodic_at_the_periodic_lines():
    """1/rho = q(eta) = 0.8 + 0.4 eta (1 - eta) with S*, beta constant: A, B, C are quadratic
    in eta, so at the blade stations (walls one-sided) and at every interior eta node,
        D = k q'(eta), k = mdot' (1 + t^2) / (L S*),   E = -mdot' t q'(eta),   q' = 0.4 (1 - 2 eta),
    exactly. q is periodic with q(0) = q(1) but its periodic extension has a kink at eta = 0
    (slopes +0.4 and -0.4 either side); the periodic central difference there takes the
    neighbors at eta = -deta and +deta and gives their average, 0, where a wall formula
    would give the one-sided slope k 0.4. Measured: 7.2e-15 (D over k), 2.7e-14 (E over
    mdot' t); on the periodic lines 1.3e-14 = 3.8e-15 k (eta[1] and 1 - eta[n - 2] differ
    by round-off)."""
    metrics = metrics_of()
    g = metrics.grid
    n_m, n_eta = ones(metrics).shape
    t = 0.4
    q = 0.8 + 0.4 * g.eta * (1.0 - g.eta)
    rho = np.repeat((1.0 / q)[None, :], n_m, axis=0)
    rho[:, -1] = rho[:, 0]
    co = cf.build_coefficients(with_geometry(metrics, np.full(n_m, S0), np.full((n_m, n_eta), t)), rho, MDOT)
    k = MDOT * (1.0 + t * t) / S0
    slope = np.repeat((0.4 * (1.0 - 2.0 * g.eta))[None, :], n_m, axis=0)
    blade = blade_mask(metrics)
    exact = np.zeros((n_m, n_eta), dtype=bool)
    exact[blade] = True
    exact[:, 1:-1] = True
    assert np.abs(co.D - k * slope)[exact].max() / k <= 1e-13
    assert np.abs(co.E + MDOT * t * slope)[exact].max() / (MDOT * t) <= 1e-13
    for name in "DE":
        assert np.abs(getattr(co, name)[~blade][:, [0, -1]]).max() <= 1e-14 * k


@pytest.mark.parametrize("n_eta", (5, 9, 17, 33))
def test_the_periodic_eta_neighbors(n_eta):
    """rho = RHO0 (1 + 0.1 sin(2 pi eta)) with S*, beta constant. On every periodic station the
    eta = 0 and eta = 1 nodes get the same D and E, the central difference of the periodic
    field with the neighbors eta = -deta (node n_eta - 2) and +deta (node 1):
        D = (k / 2 deta) [1/rho(deta) - 1/rho(-deta)],  k = mdot' (1 + t^2) / (L S*),
    evaluated here from the analytic rho. Measured: up to 7.2e-15 (D) and 2.5e-14 (E)
    relative, the round-off of rho at eta[n - 2] = 1 - deta against -deta."""
    metrics = metrics_of(31, n_eta)
    g = metrics.grid
    n_m = len(g.m)
    t = 0.4
    def rho_at(eta):
        return RHO0 * (1.0 + 0.1 * np.sin(2.0 * math.pi * eta))
    rho = np.repeat(rho_at(g.eta)[None, :], n_m, axis=0)
    rho[:, -1] = rho[:, 0]
    co = cf.build_coefficients(with_geometry(metrics, np.full(n_m, S0), np.full((n_m, n_eta), t)), rho, MDOT)
    periodic = ~blade_mask(metrics)
    k = MDOT * (1.0 + t * t) / S0
    D_line = k / (2.0 * g.deta) * (1.0 / rho_at(g.deta) - 1.0 / rho_at(-g.deta))
    E_line = -MDOT * t / (2.0 * g.deta) * (1.0 / rho_at(g.deta) - 1.0 / rho_at(-g.deta))
    for name, line in (("D", D_line), ("E", E_line)):
        field = getattr(co, name)[periodic]
        np.testing.assert_array_equal(field[:, 0], field[:, -1])
        assert np.abs(field[:, 0] - line).max() <= 5e-14 * abs(line)


def test_the_periodic_and_wall_eta_differences_converge_at_second_order():
    """The same periodic density: the error of D against d/d eta of k / rho at eta = 0 on the
    periodic stations (periodic central difference) and on the blade stations (one-sided wall
    difference) falls at second order with deta (n_eta 17, 33, 65, 129). Measured orders:
    periodic 1.996, 1.999, 2.000; wall 1.78, 1.90, 1.95 (the one-sided form reaches order 2
    more slowly)."""
    errors_periodic, errors_wall, spacings = [], [], []
    for n_eta in (17, 33, 65, 129):
        metrics = metrics_of(31, n_eta)
        g = metrics.grid
        n_m = len(g.m)
        t = 0.4
        rho = np.repeat((RHO0 * (1.0 + 0.1 * np.sin(2.0 * math.pi * g.eta)))[None, :], n_m, axis=0)
        rho[:, -1] = rho[:, 0]
        co = cf.build_coefficients(with_geometry(metrics, np.full(n_m, S0), np.full((n_m, n_eta), t)), rho, MDOT)
        k = MDOT * (1.0 + t * t) / S0
        exact = -k * 0.1 * 2.0 * math.pi / RHO0          # d/d eta of k / rho at eta = 0
        blade = blade_mask(metrics)
        errors_periodic.append(np.abs(co.D[~blade, 0] - exact).max())
        errors_wall.append(np.abs(co.D[blade, 0] - exact).max())
        spacings.append(g.deta)
    assert min(observed_orders(errors_periodic, spacings)) >= 1.99
    wall = observed_orders(errors_wall, spacings)
    assert min(wall) >= 1.75 and wall[-1] >= 1.9


def footprint(metrics: mt.GridMetrics, perturbed: list[tuple[int, int]]) -> set[tuple[int, int]]:
    """The nodes whose D or E change when rho changes at the given nodes."""
    rho = manufactured(metrics, r_m=0.1, r_eta=0.1)["rho"]
    base = cf.build_coefficients(metrics, rho, 1.0)
    changed = rho.copy()
    for i, j in perturbed:
        changed[i, j] *= 1.25
    after = cf.build_coefficients(metrics, changed, 1.0)
    nodes = set()
    for name in "DE":
        moved = np.argwhere(getattr(after, name) != getattr(base, name))
        nodes |= {(int(i), int(j)) for i, j in moved}
    return nodes


def test_the_stencil_footprint():
    """A density change at one node changes D and E exactly where a stencil reads A, B or C
    there (listed by hand from the module docstring): the m neighbors, central everywhere
    (across first_blade_i and last_blade_i too) and one-sided at the far stations; the eta
    neighbors, one-sided at blade walls and periodic at the periodic lines, where node 1 and
    node n - 2 are the neighbors of the line."""
    metrics = metrics_of(31, 9)
    g = metrics.grid
    f, l, N, n = g.first_blade_i, g.last_blade_i, len(g.m), len(g.eta)
    cases = {
        "periodic, next to the lower line": ([(5, 1)], {(4, 1), (6, 1), (5, 0), (5, 2), (5, n - 1)}),
        "periodic, next to the upper line": ([(5, n - 2)], {(4, n - 2), (6, n - 2), (5, n - 3), (5, n - 1), (5, 0)}),
        "periodic line and its image": ([(l + 3, 0), (l + 3, n - 1)],
                                        {(l + 2, 0), (l + 4, 0), (l + 2, n - 1), (l + 4, n - 1), (l + 3, 1),
                                         (l + 3, n - 2)}),
        "blade, next to the lower wall": ([(f + 5, 1)], {(f + 4, 1), (f + 6, 1), (f + 5, 0), (f + 5, 2)}),
        "blade, two from the lower wall": ([(f + 5, 2)], {(f + 4, 2), (f + 6, 2), (f + 5, 1), (f + 5, 3), (f + 5, 0)}),
        "blade, the lower wall": ([(f + 5, 0)], {(f + 4, 0), (f + 6, 0), (f + 5, 0), (f + 5, 1)}),
        "blade, the upper wall": ([(f + 5, n - 1)], {(f + 4, n - 1), (f + 6, n - 1), (f + 5, n - 1), (f + 5, n - 2)}),
        "first blade station": ([(f, 3)], {(f - 1, 3), (f + 1, 3), (f, 2), (f, 4)}),
        "last inlet station": ([(f - 1, 3)], {(f - 2, 3), (f, 3), (f - 1, 2), (f - 1, 4)}),
        "last blade station": ([(l, 3)], {(l - 1, 3), (l + 1, 3), (l, 2), (l, 4)}),
        "first exit station, lower line": ([(l + 1, 0), (l + 1, n - 1)],
                                           {(l, 0), (l + 2, 0), (l, n - 1), (l + 2, n - 1), (l + 1, 1), (l + 1, n - 2)}),
        "second station": ([(1, 3)], {(0, 3), (2, 3), (1, 2), (1, 4)}),
        "third station": ([(2, 3)], {(0, 3), (1, 3), (3, 3), (2, 2), (2, 4)}),
        "far exit station": ([(N - 1, 3)], {(N - 1, 3), (N - 2, 3), (N - 1, 2), (N - 1, 4)}),
    }
    for label, (perturbed, expected) in cases.items():
        assert footprint(metrics, perturbed) == expected, label


# --- D, E: convergence and signs --------------------------------------------------------

CASES = {
    "density in m": dict(r_m=0.1),
    "density in eta": dict(r_eta=0.1),
    "smooth geometry": dict(s=0.05, t=0.2),
    "everything": dict(s=0.05, t=0.2, r_m=0.1, r_eta=0.08, r_mix=0.05),
}


@pytest.mark.parametrize("c_ref", (None, 0.12))
@pytest.mark.parametrize("case", CASES)
def test_D_and_E_converge_at_second_order(case, c_ref):
    """Manufactured fields on three grid levels of the real topology, (100, 33), (200, 65),
    (400, 129): the largest error over every node (walls, periodic lines, far stations and
    transitions included) falls at second order, against dm (deta for the density in eta).
    Measured orders, both modes: density in m 2.011, 2.006; in eta 1.988, 1.990; smooth
    geometry 1.985, 1.999 (D), 2.06, 2.03 (E); everything 1.982, 1.992 (D), 1.978, 1.989 (E)."""
    rows = manufactured_errors(CASES[case], c_ref)
    h = [r[1] for r in rows] if case == "density in eta" else [r[0] for r in rows]
    for column in (2, 3):
        errors = [r[column] for r in rows]
        if max(errors) == 0.0:
            continue
        assert min(observed_orders(errors, h)) >= 1.95, (case, column, errors)


def test_no_excess_at_the_transitions_for_a_smooth_field():
    """With the smooth 'everything' field, the error at the stations either side of
    first_blade_i and last_blade_i is no larger than the error elsewhere: a stencil crossing a
    transition is an ordinary central difference. Measured on (100, 33): transition / rest
    max error 0.48 (D), 0.54 (E)."""
    metrics = metrics_of(100, 33)
    f = manufactured(metrics, **CASES["everything"])
    co = cf.build_coefficients(f["metrics"], f["rho"], MDOT)
    D_ref, E_ref = expanded(f["S"], f["S_m"], f["t"], f["t_m"], f["t_eta"], f["rho"], f["rho_m"], f["rho_eta"],
                            MDOT, 1.0)
    g = metrics.grid
    corner = np.zeros(len(g.m), dtype=bool)
    corner[[g.first_blade_i - 1, g.first_blade_i, g.last_blade_i, g.last_blade_i + 1]] = True
    for value, reference in ((co.D, D_ref), (co.E, E_ref)):
        error = np.abs(value - reference)
        assert error[corner].max() <= error[~corner].max()


def conservative_gap(n_passage: int, n_eta: int, c_ref: float | None) -> tuple[float, float]:
    """For psi = eta + 0.05 sin(u) sin(w) (analytic derivatives) and the 'everything' fields:
    the expanded operator A psi_ee - 2 B psi_em + C psi_mm + D psi_e + E psi_m against the
    conservative form dF/d eta - dG/dm of Eq. 5-20, F = A psi_e - B psi_m, G = B psi_e - C psi_m,
    whose derivatives are central differences of F and G taken here; interior nodes. Returns
    (max gap, max |D psi_e| + max |E psi_m|), m derivatives in the physical m."""
    metrics = metrics_of(n_passage, n_eta)
    f = manufactured(metrics, **CASES["everything"])
    co = cf.build_coefficients(f["metrics"], f["rho"], MDOT, c_ref=c_ref)
    g = metrics.grid
    length = 1.0 if c_ref is None else c_ref
    M, H = np.meshgrid(g.m, g.eta, indexing="ij")
    k = 2.0 * math.pi / (g.m[-1] - g.m[0])
    u, w = k * (M - g.m[0]), 2.0 * math.pi * H
    a = 0.05
    psi_e = 1.0 + a * np.sin(u) * 2.0 * math.pi * np.cos(w)
    psi_ee = -a * np.sin(u) * (2.0 * math.pi) ** 2 * np.sin(w)
    psi_m = a * k * np.cos(u) * np.sin(w) / length
    psi_mm = -a * k * k * np.sin(u) * np.sin(w) / length ** 2
    psi_em = a * k * np.cos(u) * 2.0 * math.pi * np.cos(w) / length
    expanded_form = co.A * psi_ee - 2.0 * co.B * psi_em + co.C * psi_mm + co.D * psi_e + co.E * psi_m
    F = co.A * psi_e - co.B * psi_m
    G = co.B * psi_e - co.C * psi_m
    dm = length * g.dm
    F_e = (F[1:-1, 2:] - F[1:-1, :-2]) / (2.0 * g.deta)
    G_m = (G[2:, 1:-1] - G[:-2, 1:-1]) / (2.0 * dm)
    gap = np.abs(expanded_form[1:-1, 1:-1] - (F_e - G_m)).max()
    size = np.abs(co.D * psi_e).max() + np.abs(co.E * psi_m).max()
    return float(gap), float(size)


@pytest.mark.parametrize("c_ref", (None, 0.12))
def test_the_signs_match_the_conservative_form(c_ref):
    """Eq. 5-21 with D = A_eta - B_m and E = C_m - B_eta is Eq. 5-20 expanded: the two
    agree to second order. A sign slip in D or E (or in -2 B) would leave a gap of the size
    of those terms. Measured: gap / (max|D psi_e| + max|E psi_m|) 1.5e-2, 3.9e-3, 9.7e-4 on
    (100, 33), (200, 65), (400, 129), both modes; orders 1.963, 1.997."""
    gaps, sizes, spacings = [], [], []
    for n_passage, n_eta in LEVELS:
        gap, size = conservative_gap(n_passage, n_eta, c_ref)
        gaps.append(gap)
        sizes.append(size)
        spacings.append(metrics_of(n_passage, n_eta).grid.deta)
    assert gaps[-1] <= 1.2e-3 * sizes[-1]
    assert min(observed_orders(gaps, spacings)) >= 1.95


# --- D, E on the default C2 cascade -----------------------------------------------------

def c2_reference_errors(n_passage: int, n_eta: int, variable_rho: bool, window: str) -> tuple[float, float]:
    """max |D - D_ref| and max |E - E_ref| on the default C2 grid, D_ref and E_ref the
    hand-expanded forms with metrics' exact C2 derivatives (dS_dm, dtan_beta_dm) and an
    analytic density, in a window of m: 'interior' >= 0.05 c from the first and last blade
    stations, 'rear' the last 35 % of the blade's m extent to 0.05 c before the last."""
    metrics = metrics_of(n_passage, n_eta)
    g = metrics.grid
    if variable_rho:
        f = manufactured(metrics, r_m=0.1, r_eta=0.08, r_mix=0.05)
        rho, rho_m, rho_eta = f["rho"], f["rho_m"], f["rho_eta"]
    else:
        rho = ones(metrics)
        rho_m = rho_eta = np.zeros_like(rho)
    co = cf.build_coefficients(metrics, rho, 1.0)
    D_ref, E_ref = expanded(metrics.S[:, None], metrics.dS_dm[:, None], metrics.tan_beta, metrics.dtan_beta_dm,
                            metrics.dS_dm[:, None], rho, rho_m, rho_eta, 1.0, 1.0)
    first, last = g.m[g.first_blade_i], g.m[g.last_blade_i]
    start = first + 0.05 if window == "interior" else first + 0.65 * (last - first)
    inside = (g.m >= start) & (g.m <= last - 0.05)
    return float(np.abs(co.D - D_ref)[inside].max()), float(np.abs(co.E - E_ref)[inside].max())


@pytest.mark.parametrize("variable_rho", (False, True))
def test_c2_D_and_E_converge_on_the_smooth_rear_of_the_blade(variable_rho):
    """Measured (rho = 1 / analytic rho): max |D error| 3.48e-3, 8.64e-4, 2.26e-4 / 8.47e-3,
    2.11e-3, 5.36e-4 at n_passage 201, 401, 801 (n_eta 61, 121, 241), orders 2.02, 1.94 /
    2.01, 1.98; max |E error| 8.5e-7, 2.2e-7, 5.4e-8 / 2.5e-3, 6.3e-4, 1.6e-4, orders 1.99,
    1.99 / 2.00, 2.00."""
    levels = ((201, 61), (401, 121), (801, 241))
    rows = [c2_reference_errors(n, k, variable_rho, "rear") for n, k in levels]
    spacings = [metrics_of(n, k).grid.dm for n, k in levels]
    for column in (0, 1):
        assert min(observed_orders([r[column] for r in rows], spacings)) >= 1.9
    assert rows[0][0] <= (3.6e-3 if not variable_rho else 8.7e-3)


@pytest.mark.parametrize("variable_rho", (False, True))
def test_c2_D_and_E_over_the_front_against_the_exact_derivative(variable_rho):
    """Over the front 60 % of the chord, the central difference of B against the exact
    derivative. Measured max |D error| (rho = 1 / analytic rho) on the quintic thickness
    (2026-11-13): 0.151 / 0.132 at n_passage 101 (n_eta 31), 0.0201 / 0.0163 at 401 (121),
    0.00557 / 0.00449 at 801 (241), ratio 3.6 from 401 to 801 (second order: 4); max |E
    error| 2.29e-3 / 1.34e-2 at 101, 1.56e-4 / 8.69e-4 at 401. With the legacy PCHIP
    thickness the walls carried the table's curvature steps, each over a span far shorter
    than dm, which the central difference averages and the exact derivative samples: D
    0.662 / 0.510, 0.221 / 0.170, 0.0685 / 0.0529 (ratio 3.2), E 2.7e-3 / 1.24e-2 and
    3.7e-4 / 1.03e-3."""
    bounds = {False: (0.155, 0.021, 0.0058, 2.4e-3, 1.6e-4), True: (0.135, 0.017, 0.0047, 1.36e-2, 8.8e-4)}[variable_rho]
    D101, E101 = c2_reference_errors(101, 31, variable_rho, "interior")
    D401, E401 = c2_reference_errors(401, 121, variable_rho, "interior")
    D801, _ = c2_reference_errors(801, 241, variable_rho, "interior")
    assert D101 <= bounds[0] and D401 <= bounds[1] and D801 <= bounds[2]
    assert D401 / D801 >= 3.5                       # measured 3.61 / 3.64
    assert E101 <= bounds[3] and E401 <= bounds[4]


def test_c2_extensions():
    """rho = 1, mdot' = 1. On the inlet extension (away from the last inlet station) A has no
    eta variation, so D is minus the m difference of the side line's tan(beta), which
    metrics.py takes on the same stations (|D| about 0.12): measured 2.4e-12 / 3.7e-11 apart
    at n_passage 101 / 401, the round-off of y0 against y0 + pitch differenced twice (eps /
    dm^2). The exit line is straight (exit angle kappa_te): |D| 8.0e-12 / 5.0e-11. E: S is
    the pitch, measured |E| 3.7e-13 / 1.5e-12."""
    for n_passage, bound in ((101, 5e-12), (401, 8e-11)):
        metrics = metrics_of(n_passage, 31)
        g = metrics.grid
        f, l = g.first_blade_i, g.last_blade_i
        co = cf.build_coefficients(metrics, ones(metrics), 1.0)
        inlet, exit_ = slice(0, f - 1), slice(l + 2, None)
        assert np.abs(co.D[inlet] + metrics.dtan_beta0_dm[inlet, None]).max() <= bound
        assert np.abs(co.D[exit_]).max() <= 2.0 * bound
        assert max(np.abs(co.E[inlet]).max(), np.abs(co.E[exit_]).max()) <= 3e-12


def test_c2_corner_stations_are_large_finite_and_unsmoothed():
    """The side line meets the wall with a corner, so tan(beta) changes by O(1) between the
    last inlet and the first blade station (and at the TE): measured at n_passage 101 the
    largest |D| (300) and the largest |E| (6.8) sit at the stations either side of the two
    corners, |D| 0.12 on the inlet extension; at least 5 times anything elsewhere (measured
    7.7 for D, against 38.9 at the second-last blade station; 87 for E). That the m difference there is the plain central difference across
    the corner is test_the_stencil_footprint."""
    metrics = metrics_of(101, 31)
    g = metrics.grid
    f, l = g.first_blade_i, g.last_blade_i
    co = cf.build_coefficients(metrics, ones(metrics), 1.0)
    corner = np.zeros(len(g.m), dtype=bool)
    corner[[f - 1, f, f + 1, l, l + 1]] = True
    for name in "DE":
        field = np.abs(getattr(co, name))
        assert np.all(np.isfinite(field))
        assert field[corner].max() == field.max()
        assert field[corner].max() >= 5.0 * field[~corner].max()
    assert np.abs(co.D[:f - 1]).max() <= 0.13


def test_c2_default_case_is_finite_and_unclipped():
    """n_passage 101, n_eta 31, rho = 1, mdot' = 1. All five fields finite; at the first and
    last five blade stations A = 1 / (S* cos^2 beta) of the metric angle, unclipped:
    measured within 2.2e-16 relative; A is largest at the first lower wall node, where
    beta = 79.164 deg (A = 28.557), B there tan(beta) = 5.224 (79.248 deg, 28.998 and 5.266
    with the legacy PCHIP thickness)."""
    metrics = metrics_of(101, 31)
    g = metrics.grid
    f, l = g.first_blade_i, g.last_blade_i
    co = cf.build_coefficients(metrics, ones(metrics), 1.0)
    for name in "ABCDE":
        assert np.all(np.isfinite(getattr(co, name)))
    edges = list(range(f, f + 5)) + list(range(l - 4, l + 1))
    for i in edges:
        for j in (0, -1):
            assert abs(co.A[i, j] * metrics.S[i] * metrics.cos_beta[i, j] ** 2 - 1.0) <= 4.5e-16
            assert co.B[i, j] == metrics.tan_beta[i, j]
    assert np.unravel_index(int(np.argmax(co.A)), co.A.shape) == (f, 0)
    assert math.degrees(metrics.beta[f, 0]) == pytest.approx(79.164, abs=1e-3)     # measured 79.16421
    assert co.A[f, 0] == pytest.approx(28.557, abs=1e-3)                            # measured 28.55676


# --- units and scaling ------------------------------------------------------------------

def test_normalized_mode_is_a_unit_reference_length():
    """c_ref = None builds exactly what c_ref = 1 builds, bit for bit, and records that no
    physical length is defined."""
    metrics = metrics_of()
    rho = manufactured(metrics, r_m=0.1, r_eta=0.1)["rho"]
    normalized = cf.build_coefficients(metrics, rho, MDOT)
    unit = cf.build_coefficients(metrics, rho, MDOT, c_ref=1.0)
    for name in "ABCDE":
        np.testing.assert_array_equal(getattr(normalized, name), getattr(unit, name))
    assert normalized.c_ref is None and normalized.length_scale == 1.0 and not normalized.physical_scale_defined
    assert unit.c_ref == 1.0 and unit.length_scale == 1.0 and unit.physical_scale_defined
    assert normalized.mass_flow_per_depth == MDOT


def psi_terms(co: cf.CoefficientFields, metrics: mt.GridMetrics) -> list[np.ndarray]:
    """The five terms of Eq. 5-21 for psi = eta + 0.05 sin(u) sin(w), m derivatives taken in
    the physical m = length_scale m*."""
    g = metrics.grid
    length = co.length_scale
    M, H = np.meshgrid(g.m, g.eta, indexing="ij")
    k = 2.0 * math.pi / (g.m[-1] - g.m[0])
    u, w = k * (M - g.m[0]), 2.0 * math.pi * H
    psi_e = 1.0 + 0.05 * np.sin(u) * 2.0 * math.pi * np.cos(w)
    psi_ee = -0.05 * np.sin(u) * (2.0 * math.pi) ** 2 * np.sin(w)
    psi_m = 0.05 * k * np.cos(u) * np.sin(w) / length
    psi_mm = -0.05 * k * k * np.sin(u) * np.sin(w) / length ** 2
    psi_em = 0.05 * k * np.cos(u) * 2.0 * math.pi * np.cos(w) / length
    return [co.A * psi_ee, -2.0 * co.B * psi_em, co.C * psi_mm, co.D * psi_e, co.E * psi_m]


def scaled_gap(value: np.ndarray, reference: np.ndarray) -> float:
    """max |value - reference| over max |reference|."""
    return float(np.abs(value - reference).max() / np.abs(reference).max())


@pytest.mark.parametrize("lam", (0.05, 3.0, 250.0))
def test_fixed_mass_flow_scaling(lam):
    """Normalized metrics, rho and mdot' held, c_ref from 0.1 to lam 0.1: A and D ~ 1/lam,
    B and E unchanged, C ~ lam; every term of Eq. 5-21 ~ 1/lam (so psi does not change).
    Measured, over each field's largest value: 4.0e-16 (A, B, C), 9.5e-16 (D), 6.4e-15 (E,
    which is round-off over much of the extensions), 1.2e-14 (terms)."""
    metrics = metrics_of()
    rho = manufactured(metrics, r_m=0.1, r_eta=0.1)["rho"]
    base = cf.build_coefficients(metrics, rho, MDOT, c_ref=0.1)
    scaled = cf.build_coefficients(metrics, rho, MDOT, c_ref=0.1 * lam)
    for name, power, bound in (("A", -1, 1e-15), ("B", 0, 1e-15), ("C", 1, 1e-15), ("D", -1, 5e-15),
                               ("E", 0, 1.5e-14)):
        assert scaled_gap(getattr(scaled, name), lam ** power * getattr(base, name)) <= bound, name
    for term, reference in zip(psi_terms(scaled, metrics), psi_terms(base, metrics)):
        assert scaled_gap(term, reference / lam) <= 3e-14


@pytest.mark.parametrize("lam", (0.05, 3.0, 250.0))
def test_physically_similar_twins(lam):
    """Geometrically similar cascades at the same rho, velocities and angles: mdot' =
    rho W_m S scales with the chord, so c_ref -> lam c_ref and mdot' -> lam mdot'. Then A and
    D are unchanged, B and E ~ lam, C ~ lam^2, and every term of Eq. 5-21 for the same
    psi(m*, eta) is unchanged: the twins share one normalized solution. Measured: 5.4e-16
    (A, B, C), 1.1e-15 (D), 6.8e-15 (E), 1.1e-14 (terms)."""
    metrics = metrics_of()
    rho = manufactured(metrics, r_m=0.1, r_eta=0.1)["rho"]
    base = cf.build_coefficients(metrics, rho, MDOT, c_ref=0.1)
    twin = cf.build_coefficients(metrics, rho, lam * MDOT, c_ref=0.1 * lam)
    for name, power, bound in (("A", 0, 1e-15), ("B", 1, 1e-15), ("C", 2, 1e-15), ("D", 0, 5e-15),
                               ("E", 1, 1.5e-14)):
        assert scaled_gap(getattr(twin, name), lam ** power * getattr(base, name)) <= bound, name
    for term, reference in zip(psi_terms(twin, metrics), psi_terms(base, metrics)):
        assert scaled_gap(term, reference) <= 3e-14


def test_dimensional_analysis():
    """The same physical inputs in other units (grams, millimetres, milliseconds, and an odd
    set): mdot' [M/(L T)], rho [M/L^3] and c_ref [L] convert by their dimensions, and each
    field must convert by its own, A and D by L/T, B and E by L^2/T, C by L^3/T. Measured,
    over each field's largest value: 4.1e-16 (A, B, C), 1.2e-15 (D), 1.2e-14 (E)."""
    metrics = metrics_of()
    rho = 1.2 * manufactured(metrics, r_m=0.1, r_eta=0.1)["rho"]
    base = cf.build_coefficients(metrics, rho, 8.5, c_ref=0.1)
    for mass, length, time in ((1e3, 1e3, 1e3), (1e3, 1.0, 1.0), (1.0, 1e3, 1.0), (1.0, 1.0, 1e3),
                               (0.37, 41.0, 2.9e-2)):
        other = cf.build_coefficients(metrics, rho * mass / length ** 3, 8.5 * mass / (length * time),
                                      c_ref=0.1 * length)
        for name, factor, bound in (("A", length / time, 2e-15), ("B", length ** 2 / time, 2e-15),
                                    ("C", length ** 3 / time, 2e-15), ("D", length / time, 5e-15),
                                    ("E", length ** 2 / time, 3e-14)):
            assert scaled_gap(getattr(other, name), factor * getattr(base, name)) <= bound, (name, mass, length, time)
