"""
coefficients.py
===============
The coefficient fields A, B, C, D, E of Aungier's stream-function equation in expanded form
[Aungier 2003 Eqs. 5-20 to 5-26], at the nodes of a grid from grid.py, built from its
metrics (metrics.py), a frozen density field and the mass flow per unit depth. The density
enters, so the fields are rebuilt on every pass of the density loop, while the grid and the
metrics stay fixed. This module ends at A to E: no stencil, no matrix, no boundary
condition and no psi (discretization.py, boundary_conditions.py, linear_solver.py).

    geometry.py -> grid.py -> metrics.py -> coefficients.py

The equation and its signs [Eqs. 5-20, 5-21; for discretization.py to take as they are]
    Conservative form (Eq. 5-20), with mu = mdot / (b rho):
        dF/d eta - dG/dm = 2 S omega sin(phi),
        F = A psi_eta - B psi_m,        G = B psi_eta - C psi_m,
    and expanded (Eq. 5-21), by the product rule on F and G:
        A psi_eta,eta  -  2 B psi_eta,m  +  C psi_m,m  +  D psi_eta  +  E psi_m  =  2 S omega sin(phi),
    with
        A = mu / (S cos^2 beta)          [Eq. 5-22]
        B = mu tan(beta)                 [Eq. 5-23]
        C = mu S                         [Eq. 5-24]
        D = dA/d eta - dB/dm             [Eq. 5-25]
        E = dC/dm - dB/d eta             [Eq. 5-26]
    Signs: + A, - 2 B, + C, + D, + E on the left; D = A_eta - B_m and E = C_m - B_eta. The
    mixed term carries -2 B (B once from F, once from G), and D and E are the first-
    derivative terms the expansion leaves; nothing else is folded into them.
    Right-hand side: sin(phi) = dr/dm = 0 on this solver's flat cascade at constant radius
    (y = r theta, metrics.py), so 2 S omega sin(phi) = 0. No radius, rotation speed or
    source term is implemented or accepted here: a rotating row on a radius-changing stream
    sheet is a later extension, with its own argument and field.

Per unit depth: mdot' = mdot / b
    Aungier carries the stream-sheet thickness b in mu = mdot / (b rho). This module takes
    mass_flow_per_depth = mdot' = mdot / b instead, the mass flow through ONE blade passage
    per unit thickness of the stream sheet (per unit span in a linear cascade), so
        mu = mdot' / rho.
    b has been absorbed into mdot'; no b, and no b = 1, appears anywhere. With psi rising by
    one across the passage (Aungier's normalization, W_m = mu / S psi_eta), mdot' = integral
    of rho W_m S d eta at any station; at the far inlet that is rho_in W_m,in pitch. Its
    units: kg/(m s) in physical SI mode; in normalized mode, those of rho times velocity
    times chord (below). Computing it from a Mach number, total state and inlet angle
    belongs to a later initialization layer, not here.

Lengths: the chain is chord-normalized; c_ref is the one physical scale
    geometry.py, grid.py and metrics.py work on the chord-1 blade here:
        m* = m / c,   y* = y / c,   S* = S / c = metrics.S,   grid.dm = dm*,
    all dimensionless. tan(beta), beta, cos(beta), dS/dm = dS*/dm* and eta do not change
    with the chord; a derivative in m does:  d/dm = (1 / c) d/dm*.
    c_ref sets the length scale L_ref once, here:
        c_ref = positive float  physical mode: c_ref is the true chord in metres, L_ref = c_ref.
        c_ref = None            normalized mode: L_ref = 1 reference length, the chord itself.
    Then
        S = L_ref S*,   m = L_ref m*,   dm = L_ref dm*,   d/dm = (1 / L_ref) d/dm*,
    and d/d eta needs no length (eta is dimensionless).
    Normalized mode (c_ref = None) is NOT a one-metre blade: no SI length is defined. The
    fields are then in reference units (lengths in chords, mdot' in the units of rho times
    velocity times chord that the caller chose), which is what a normalized result such as
    Mach number against x/c needs; dimensionless flow quantities downstream keep their
    meaning, but nothing derived here is a physical mass flow, and this module reports none.
    CoefficientFields.physical_scale_defined says which mode a field set was built in.
    One scale, applied once: geometry.py can also scale the blade itself
    (CascadeSpec.physical_chord), and a grid of such a cascade is already in metres. This
    module takes chord-normalized grids only (grid.chord = 1, to round-off) and refuses
    any other with CoefficientError: c_ref on a metre grid would apply the chord twice. Build
    the cascade with physical_chord = None and pass the chord as c_ref.

Density: a frozen two-dimensional field
    rho is the density at every node, (n_m, n_eta), whatever produced it (the inlet value
    everywhere on the first pass, the gas model on later ones): finite and strictly
    positive, else CoefficientError. On the periodic stations (outside first_blade_i ..
    last_blade_i) the eta = 0 and eta = 1 nodes are one physical point and its periodic
    image (y1 = y0 + pitch, S = pitch, metrics.py), so rho must agree there to round-off
    (ROUND_OFF times max rho); a density that differs there is not a field on the periodic
    passage, and its eta derivative there would be meaningless: CoefficientError.
    No gas law, temperature, pressure or Mach number enters this module.

A, B, C: pointwise [Eqs. 5-22 to 5-24]
    From the validated metric fields, never from the geometry (beta is not recomputed):
        A[i, j] = mdot' / (S[i] rho[i, j] cos^2 beta[i, j]),     S = L_ref S*,
        B[i, j] = mdot' tan beta[i, j] / rho[i, j],
        C[i, j] = mdot' S[i] / rho[i, j],
    S[i] broadcast along eta, not copied. Nothing is clipped or capped: A grows as
    1 / cos^2(beta) where the grid line is steep (the first and last blade stations of a
    rounded edge, metrics.py), and a large A there is the geometry, not an error. S <= 0,
    a non-finite or non-positive cos(beta), or an A, B or C that is not finite (cos(beta)
    so near zero that the division overflows) is a CoefficientError naming the node;
    metrics.py has already refused cos(beta) <= SINGULAR_COS.

D, E: second-order differences of the fields A, B, C [Eqs. 5-25, 5-26]
    D = A_eta - B_m and E = C_m - B_eta are taken by differencing the complete fields A, B
    and C, so every variation of rho, S and beta is in them. They are not expanded into
    rho_m, rho_eta, S_m, d tan(beta)/dm and so on, and metrics.dtan_beta_dm is not used:
    those serve the tests as an independent check.
    m (at fixed eta), with dm = L_ref grid.dm:
        interior     f_m[i] = (f[i+1] - f[i-1]) / (2 dm)
        i = 0        f_m[0] = (-3 f[0] + 4 f[1] - f[2]) / (2 dm)                    (far inlet)
        i = n_m - 1  f_m[i] = (3 f[i] - 4 f[i-1] + f[i-2]) / (2 dm)                 (far exit)
    One stencil over the whole m range: A, B and C are coefficient fields of the one
    computational domain, and their m difference crosses first_blade_i and last_blade_i
    like any other station (unlike the boundary slopes of metrics.py, which differentiate
    two different curves on either side). Across a transition the grid lines have a corner:
    the side line arrives at the blade angle kappa_le, while the wall beyond it leaves the
    nose at the curve's own, much steeper, angle (and likewise at the TE). So tan(beta), and
    with it B, changes by O(1) between the last inlet station and the first blade station
    (and between the last blade station and the first exit station), and S falls from the
    pitch by the metal of the first blade station's cut. The central differences at the
    stations either side of a corner are therefore large, about that change over 2 dm, and
    grow as 1 / dm under refinement (accuracy, below). They are not smoothed, and they are
    not an error: the central difference telescopes,
        sum over i = a .. b of f_m[i] dm  =  (f[b] + f[b+1]) / 2 - (f[a-1] + f[a]) / 2,
    so across a corner the large values add up exactly to the change of B (and C) there,
    which is what the fluxes of Eq. 5-20 carry past it; a one-sided difference inside each
    region would drop that change. How the psi equation treats these stations is
    discretization.py's choice.
    eta (at fixed m), dimensionless, deta = grid.deta, j = 0 .. n_eta - 1:
        interior          f_eta[j] = (f[j+1] - f[j-1]) / (2 deta)
        blade stations    the nodes j = 0 and n_eta - 1 are the walls: one-sided into the
                          fluid, f_eta[0] = (-3 f[0] + 4 f[1] - f[2]) / (2 deta) and
                          f_eta[-1] = (3 f[-1] - 4 f[-2] + f[-3]) / (2 deta); no stencil
                          reaches through the blade.
        periodic stations the passage repeats with period one in eta (S = pitch there), and
                          j = 0 and j = n_eta - 1 are the same point: the node below j = 0 is
                          j = n_eta - 2 (eta = 1 - deta, i.e. -deta) and the node above
                          j = n_eta - 1 is j = 1 (eta = 1 + deta, i.e. deta), so both get
                              f_eta[0] = f_eta[-1] = (f[1] - f[n_eta - 2]) / (2 deta),
                          the central difference of the periodic field, identical at the two.
    A second-order difference needs three nodes either way: n_eta >= 3 (CoefficientError
    otherwise); metrics.py already guarantees n_m >= 3 per region.

Dimensions (physical mode)
    [mdot'] = M / (L T),  [rho] = M / L^3,  [S] = L,  so [mu] = [mdot' / rho] = L^2 / T and
        [A] = L/T,  [B] = L^2/T,  [C] = L^3/T,
        [D] = [A_eta] = [B_m] = L/T,  [E] = [C_m] = [B_eta] = L^2/T.
    psi is dimensionless (W_m = mu / S psi_eta is a velocity), so every term of Eq. 5-21,
    A psi_eta,eta, B psi_eta,m, C psi_m,m, D psi_eta and E psi_m, is L/T, as is 2 S omega sin(phi).
    This holds only if psi_m and psi_m,m are taken in the physical m: the stencils of
    discretization.py must use the same dm = L_ref grid.dm (CoefficientFields.length_scale).

Scaling (checked by the tests)
    Fixed mdot', the chord scaled by lambda (normalized metrics and rho held):
        A ~ lambda^-1,  B ~ lambda^0,  C ~ lambda,  D ~ lambda^-1,  E ~ lambda^0.
    Physically similar twins (same rho, velocities, angles and solidity, so mdot' ~ lambda):
        A ~ lambda^0,  B ~ lambda,  C ~ lambda^2,  D ~ lambda^0,  E ~ lambda,
    and with psi(m*, eta) unchanged every term of Eq. 5-21 is unchanged: the twins share one
    normalized solution.

Accuracy of D and E (measured 2026-11-12, scripts/audit_coefficients.py; the default C2
cascade, 65-(12A10)10 at 33 deg, inlet 45 deg; geometric, not a flow convergence)
    Manufactured fields on the real grid topology (tests/test_coefficients.py): second
    order to their closed forms, with no excess at the transitions, the walls or the
    periodic lines; exact for quadratics.
    Against D and E expanded by hand with metrics.py's exact C2 derivatives (rho = 1, or a
    smooth analytic rho with n_eta refined alongside):
      smooth rear of the blade (its last 35 % in m, to 0.05 c before the last station):
        second order, max |D error| 8.2e-3 at n_passage 101, 8.6e-4 at 401; E 3.4e-6, 2.2e-7.
      front 60 % of the chord: the C2 walls follow the C1 thickness table to 1e-6 c, and with
        it its curvature steps at the table stations, each over a span far shorter than dm:
        d tan(beta)/dm is a sawtooth there with near-jumps of order 1. The central difference
        of B averages each step over 2 dm, the exact derivative samples it at the node, and
        the two differ by up to 0.66 at n_passage 101 (|D| up to 4.8 there), 0.22 at 401 and
        0.07 at 801: first order until dm resolves the steps (observed order 1.7 from 401 to
        801). E (through S): 2.7e-3 at 101, 3.7e-4 at 401.
      corner stations (first_blade_i - 1, first_blade_i, last_blade_i, last_blade_i + 1): no
        reference (the derivative of a corner); |D| up to 300 at n_passage 101 against 0.12
        on the inlet extension and about 5 on the blade, 1.6e3 at 401; |E| up to 6.8 and 7.9.
    The extensions: D is minus the m derivative of the side line's tan(beta) (A has no eta
    variation there), E is round-off (below 2e-12).

Status: implemented 2026-11-12 for the planar, non-rotating C2 solver path. Not frozen:
under review.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real

import numpy as np
from numpy.typing import NDArray

from turbolenza.b2b.metrics import ROUND_OFF, GridMetrics

Field = NDArray[np.float64]

_FIELDS = ("A", "B", "C", "D", "E")


class CoefficientError(ValueError):
    """An input breaks what the coefficient formulas assume: a density that is not a
    finite, strictly positive (n_m, n_eta) field periodic on the periodic stations, a mass
    flow per depth or reference chord that is not a finite positive number, a grid that is
    not chord-normalized or has fewer than three eta nodes, metric fields that make S or
    cos(beta) non-positive, or a coefficient that comes out non-finite."""


@dataclass(frozen=True)
class CoefficientFields:
    """A, B, C, D, E of Eq. 5-21 at every node, each (n_m, n_eta) and read-only, with the
    two scalars that fix their units (module docstring): mass_flow_per_depth = mdot' (all
    five fields are proportional to it) and c_ref, the physical chord in metres, or None in
    normalized mode."""

    A: Field                        # (n_m, n_eta), L/T: mdot' / (S rho cos^2 beta)          [Eq. 5-22]
    B: Field                        # (n_m, n_eta), L^2/T: mdot' tan(beta) / rho              [Eq. 5-23]
    C: Field                        # (n_m, n_eta), L^3/T: mdot' S / rho                      [Eq. 5-24]
    D: Field                        # (n_m, n_eta), L/T: dA/d eta - dB/dm                     [Eq. 5-25]
    E: Field                        # (n_m, n_eta), L^2/T: dC/dm - dB/d eta                   [Eq. 5-26]
    mass_flow_per_depth: float      # mdot' = mdot / b, one passage: kg/(m s), or reference units if c_ref is None
    c_ref: float | None             # physical chord in metres; None: normalized mode, L_ref = 1 chord

    def __post_init__(self) -> None:
        shape = None
        for name in _FIELDS:
            array = np.array(getattr(self, name), dtype=float)
            if array.ndim != 2 or (shape is not None and array.shape != shape):
                raise CoefficientError(f"{name} has shape {array.shape}: A to E must be (n_m, n_eta) arrays of one "
                                       f"shape{'' if shape is None else f' {shape}'}")
            if not np.all(np.isfinite(array)):
                raise CoefficientError(f"{name} is not finite everywhere")
            shape = array.shape
            array.flags.writeable = False
            object.__setattr__(self, name, array)
        object.__setattr__(self, "mass_flow_per_depth", _mass_flow(self.mass_flow_per_depth))
        object.__setattr__(self, "c_ref", _reference_chord(self.c_ref))

    @property
    def length_scale(self) -> float:
        """L_ref: c_ref in metres, or 1.0 reference length (the chord) in normalized mode. The
        physical m spacing is length_scale * grid.dm, for every m difference of Eq. 5-21."""
        return 1.0 if self.c_ref is None else self.c_ref

    @property
    def physical_scale_defined(self) -> bool:
        """True in physical mode (c_ref given, SI units); False in normalized mode, where no
        SI length, and so no physical mass flow, is defined."""
        return self.c_ref is not None


Coefficients = CoefficientFields     # the name the discretization.py interface imports


def build_coefficients(metrics: GridMetrics, rho: NDArray[np.float64], mass_flow_per_depth: float, *,
                       c_ref: float | None = None) -> CoefficientFields:
    """A to E of Eq. 5-21 at every node (module docstring).

    metrics: the GridMetrics of a chord-normalized grid; m, eta, dm, deta and the blade
    stations are read from metrics.grid, the fields S, tan_beta and cos_beta from metrics.
    rho: the frozen density field, (n_m, n_eta), finite, positive, periodic on the periodic
    stations. mass_flow_per_depth: mdot' = mdot / b through one passage, finite and
    positive. c_ref (keyword only): the chord in metres, or None for normalized mode.

    Raises TypeError for metrics that are not a GridMetrics or a mass flow or c_ref that is
    not a real number, and CoefficientError for any other broken input or a non-finite
    coefficient."""
    if not isinstance(metrics, GridMetrics):
        raise TypeError(f"build_coefficients takes a GridMetrics, got {type(metrics).__name__}")
    mdot = _mass_flow(mass_flow_per_depth)
    c_ref = _reference_chord(c_ref)
    length = 1.0 if c_ref is None else c_ref
    grid = metrics.grid
    if not abs(grid.chord - 1.0) <= ROUND_OFF:
        raise CoefficientError(f"the grid's chord is {grid.chord:.17g}, not 1: coefficients.py takes a chord-normalized "
                               f"grid and applies the physical chord once, as c_ref. Build the cascade with "
                               f"physical_chord=None and pass c_ref={grid.chord:.17g} instead")
    n_m, n_eta = len(grid.m), len(grid.eta)
    if n_eta < 3:
        raise CoefficientError(f"the grid has {n_eta} eta nodes; a second-order eta difference needs three")
    blade = np.zeros(n_m, dtype=bool)
    blade[grid.first_blade_i:grid.last_blade_i + 1] = True
    rho = _density(rho, (n_m, n_eta), blade)
    S, tan_beta, cos_beta = _metric_inputs(metrics, length)

    with np.errstate(all="ignore"):
        A = mdot / (S * rho * cos_beta ** 2)
        B = mdot * tan_beta / rho
        C = mdot * S / rho
    for name, field in (("A", A), ("B", B), ("C", C)):
        _finite(name, field, metrics, rho)
    dm, deta = length * grid.dm, grid.deta
    with np.errstate(all="ignore"):
        D = _d_deta(A, deta, blade) - _d_dm(B, dm)
        E = _d_dm(C, dm) - _d_deta(B, deta, blade)
    for name, field in (("D", D), ("E", E)):
        _finite(name, field, metrics, rho)
    return CoefficientFields(A=A, B=B, C=C, D=D, E=E, mass_flow_per_depth=mdot, c_ref=c_ref)


def _mass_flow(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"mass_flow_per_depth must be a real number, got {type(value).__name__}")
    value = float(value)
    if not (math.isfinite(value) and value > 0.0):
        raise CoefficientError(f"mass_flow_per_depth = {value!r}: the mass flow through one passage per unit depth must "
                               f"be finite and positive")
    return value


def _reference_chord(value: float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"c_ref must be None or a real number of metres, got {type(value).__name__}")
    value = float(value)
    if not (math.isfinite(value) and value > 0.0):
        raise CoefficientError(f"c_ref = {value!r}: the physical chord must be a finite positive length in metres, or "
                               f"None for normalized mode")
    return value


def _density(rho: NDArray[np.float64], shape: tuple[int, int], blade: NDArray[np.bool_]) -> Field:
    """The density field, checked (module docstring, "Density")."""
    try:
        rho = np.array(rho, dtype=float)
    except (TypeError, ValueError) as error:
        raise CoefficientError(f"rho must be a numeric (n_m, n_eta) = {shape} array: {error}") from None
    if rho.shape != shape:
        raise CoefficientError(f"rho has shape {rho.shape}, the grid needs the node field {shape} (n_m, n_eta)")
    bad = np.argwhere(~np.isfinite(rho))
    if bad.size:
        i, j = (int(k) for k in bad[0])
        raise CoefficientError(f"rho is not finite at node ({i}, {j}): {rho[i, j]!r}")
    bad = np.argwhere(~(rho > 0.0))
    if bad.size:
        i, j = (int(k) for k in bad[0])
        raise CoefficientError(f"rho = {rho[i, j]!r} at node ({i}, {j}): the density must be strictly positive")
    periodic = np.flatnonzero(~blade)
    jump = np.abs(rho[periodic, 0] - rho[periodic, -1])
    tolerance = ROUND_OFF * float(np.max(rho))
    if not np.all(jump <= tolerance):
        k = int(np.argmax(jump))
        i = int(periodic[k])
        raise CoefficientError(f"rho at the periodic station {i} is {rho[i, 0]:.17g} at eta = 0 and {rho[i, -1]:.17g} at "
                               f"eta = 1: the two nodes are one point and its periodic image, so rho must agree there "
                               f"(to {tolerance:.3g})")
    rho.flags.writeable = False
    return rho


def _metric_inputs(metrics: GridMetrics, length: float) -> tuple[Field, Field, Field]:
    """S = L_ref S* as an (n_m, 1) column for broadcasting, tan(beta) and cos(beta), checked."""
    S_star = np.asarray(metrics.S, dtype=float)
    bad = np.flatnonzero(~(np.isfinite(S_star) & (S_star > 0.0)))
    if bad.size:
        i = int(bad[0])
        raise CoefficientError(f"S* = {S_star[i]!r} at station {i}: the passage spacing must be finite and positive")
    tan_beta, cos_beta = np.asarray(metrics.tan_beta, dtype=float), np.asarray(metrics.cos_beta, dtype=float)
    for name, field, check in (("tan(beta)", tan_beta, np.isfinite(tan_beta)),
                               ("cos(beta)", cos_beta, np.isfinite(cos_beta) & (cos_beta > 0.0))):
        bad = np.argwhere(~check)
        if bad.size:
            i, j = (int(k) for k in bad[0])
            raise CoefficientError(f"{name} = {field[i, j]!r} at node ({i}, {j}): it must be finite"
                                   f"{' and positive' if name == 'cos(beta)' else ''}")
    return length * S_star[:, None], tan_beta, cos_beta


def _finite(name: str, field: Field, metrics: GridMetrics, rho: Field) -> None:
    bad = np.argwhere(~np.isfinite(field))
    if bad.size:
        i, j = (int(k) for k in bad[0])
        raise CoefficientError(f"{name} is not finite at node ({i}, {j}) (S* = {metrics.S[i]:.6g}, cos(beta) = "
                               f"{metrics.cos_beta[i, j]:.3g}, rho = {rho[i, j]:.6g}): the coefficient is undefined "
                               f"there, and nothing is clipped to hide it")


def _d_dm(f: Field, dm: float) -> Field:
    """d/dm at fixed eta of a node field (first axis along m), second order over the whole m
    range: central inside, three-point one-sided at the far inlet and exit stations."""
    d = np.empty_like(f)
    d[1:-1] = (f[2:] - f[:-2]) / (2.0 * dm)
    d[0] = (-3.0 * f[0] + 4.0 * f[1] - f[2]) / (2.0 * dm)
    d[-1] = (3.0 * f[-1] - 4.0 * f[-2] + f[-3]) / (2.0 * dm)
    return d


def _d_deta(f: Field, deta: float, blade: NDArray[np.bool_]) -> Field:
    """d/d eta at fixed m of a node field, second order: central inside; on the blade
    stations one-sided into the fluid at the two walls; on the periodic stations the
    periodic central difference at eta = 0 and 1 (module docstring)."""
    d = np.empty_like(f)
    d[:, 1:-1] = (f[:, 2:] - f[:, :-2]) / (2.0 * deta)
    wall = f[blade]
    d[blade, 0] = (-3.0 * wall[:, 0] + 4.0 * wall[:, 1] - wall[:, 2]) / (2.0 * deta)
    d[blade, -1] = (3.0 * wall[:, -1] - 4.0 * wall[:, -2] + wall[:, -3]) / (2.0 * deta)
    periodic = f[~blade]
    d[~blade, 0] = d[~blade, -1] = (periodic[:, 1] - periodic[:, -2]) / (2.0 * deta)
    return d
