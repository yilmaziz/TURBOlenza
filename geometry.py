"""
geometry.py
===========
Blade and cascade geometry for the blade-to-blade solver: one blade built from its family
parameters, set at a stagger angle, repeated at a pitch, with its throat. No grid, no
(m, eta) mapping and no flow here; grid.py consumes the Cascade this module returns.

Self-contained: this file needs only the Python standard library, NumPy and SciPy. The
NACA 65 data and every construction step below live in this file.

Use
    spec = CascadeSpec(blade_type="NACA65", solidity=..., stagger_deg=..., cl0=..., t_max_over_c=...)
    cascade = make_cascade(spec)
CascadeSpec holds the inputs and the only modelling defaults; Cascade is the generated,
checked geometry, and keeps its spec. surface_continuity="C2" gives the blade as quintic
curves with a robust leading edge (see "Surface continuity" and "The C2 leading edge").

Families
    "NACA65" is implemented, with the A10 (a = 1.0) mean line and two profile variants.
    "DCA" and "C4" are part of the API and raise NotImplementedError.

NACA 65 construction [Aungier 2003 Secs. 4.1, 4.2, Table 4-1; standard NACA practice]
    - Data: the NACA 65-series columns of Table 4-1 (from Emery et al. 1958), stored below
      as printed. The 65-010 thickness at x/c = 2.5 % is misprinted 3.480 %; the thickness
      law uses the correct 3.148 % (twice the standard 65-010 half-ordinate 1.574 %).
    - Mean line: the closed form of the a = 1.0 line,
          y/c = -(cl0 / 4 pi) [ (1 - x) ln(1 - x) + x ln x ],
      which Table 4-1 tabulates for Cl0 = 1; the printed values agree with it to 2.3e-5 c.
      Ordinates scale linearly with cl0.
    - Thickness: the corrected column, interpolated by PCHIP in sqrt(station). PCHIP cannot
      overshoot at the blunt nose or the thin TE, and a nose that grows like sqrt(x) is
      nearly straight in sqrt(x). The station is distance along the camberline, s / S,
      not x/c.
    - Surfaces: half the thickness laid off along the camberline normal,
      X = X_c +- (t / 2) n_c, from the first table station (0.005) to the TE.
    - LE: a circle of radius R_LE = 0.00687 f^2 (f the thickness scale factor; the nose is
      parabolic) through the LE point (0, 0), centered on the mean-line tangent at
      x/c = 0.005. Ahead of that station the mean line turns several degrees, so circle
      and surfaces miss or cross; each side is joined by a quintic fair in polar form about
      the circle center, matching radius, slope and curvature at both ends. It leaves the
      circle at the smallest angle >= 20 deg that minimizes the peak curvature of the fair.
    - "emery_base": the corrected table thickness all the way to a sharp TE (R_TE = 0).
      "kovach_sandercock": the table up to station 0.60, then a straight taper tangent to a
      TE circle of radius 0.008 c centered on the camberline and through (1, 0)
      [Kovach & Sandercock 1961]. The tangency holds along the straightened camberline;
      on the curved blade each surface ends at its own nearest point to the circle center,
      and a short radial fair reaches the circle.

    t_max_over_c is the nominal thickness ratio of the NACA designation (0.10 for a
    "65-(12A10)10"), and the corrected table is scaled by t_max_over_c / 0.10. So 0.10
    keeps Table 4-1 exactly, including its own maximum of 0.09992 at station 0.40: the
    table is not renormalized to peak at 0.10000. The true maximum thickness is
    metadata["max_thickness_over_c"] = 0.9992 t_max_over_c.

Two camberlines
    camberline is the actual A10 mean line, and it alone builds the surfaces. Its slope is
    infinite at both ends, so it gives no usable blade angles. The angles come from the
    equivalent circular arc instead [Aungier 2003 Eqs. 4-6, 4-32 to 4-36]:
        tan(camber_angle / 4) = 0.1103 Cl0,    chi_le = chi_te = camber_angle / 2
        kappa_le = stagger + chi_le,           kappa_te = stagger - chi_te
    equivalent_arc_camberline holds that arc for reference only; nothing is built from it.

Throat [Aungier 2003 Sec. 4.9]
    The geometric throat is the shortest distance between the suction surface of the
    reference blade and the pressure surface of its neighbor, measured on cubic splines
    through the surface points (closest_points). It is the preferred value. The empirical
    o / s of Eqs. 4-30 and 4-31 is kept beside it as a diagnostic and never replaces it.

Frames and signs
    Blade frame: x along the chord, y normal to it, LE at (0, 0), TE at (1, 0), camber
    toward +y. Cascade frame: x axial (the m direction of grid.py), y tangential. The blade
    turns counterclockwise by the stagger about its LE,
        X = x cos(stagger) - y sin(stagger),    Y = x sin(stagger) + y cos(stagger),
    and its neighbor is the same blade moved by periodic_translation = (0, pitch). Angles
    are measured from the axial direction, positive toward +y. A public input in degrees
    ends in _deg (CascadeSpec.stagger_deg, the one angle input); every angle that is
    computed is in radians and has no suffix (stagger, kappa_le, ...).

Suction and pressure surface
    The suction surface is the +y side of the blade frame: the convex side when cl0 > 0,
    and the side that faces the neighbor blade. For cl0 = 0 the blade is symmetric and the
    label is this convention only. Both surfaces run LE -> TE, share their two end points,
    and repeat no point.

Lengths
    The blade is always built with chord 1. physical_chord, in metres, is a uniform scale
    on that same blade. Cascade.normalized holds the chord-1 lengths and Cascade.active the
    lengths grid.py uses: the same as normalized when physical_chord is None.

Surface continuity (CascadeSpec.surface_continuity, the one switch)
    "C1", the default, is the construction above as it is: the validated baseline, unchanged
    bit for bit. Its surfaces are the polylines, and the curve they sample is C1 only: the
    PCHIP thickness jumps the curvature at every table station (up to 3 /c near the nose,
    1.2 /c mid-chord), the LE fair meets the surface with a 0.006-0.008 deg kink, the Kovach
    TE fair keeps the tangent (to 1e-7 rad) but jumps the curvature by about 1/R_TE, and the
    Kovach start, station 0.60, is a corner (0.08 deg at t/c 0.10); the emery_base closing
    segment meets the surface at x/c 0.999 at a corner of a few degrees.
    naca65_reference_surfaces gives each C1 surface as its chain of analytic pieces with
    exact derivatives; every C1 vertex lies on it to C2_ANCHOR_TOL.

    "C2" is the robust design geometry: per surface one clamped quintic B-spline
    X(u) = (m(u), y(u)) (C4 at its simple knots) representing, by least squares, a target
    made of two parts that meet G2 at an anchor (fit_c2_surface):
    - Downstream of the anchor, the C1 table surface as the C2 fit sees it (SmoothedSurface):
      at every junction where C1's curvature jumps or its tangent breaks, the kink of the
      tangent angle is replaced by its convolution with the positive kernel
      (35/32)(1 - x^2)^3 of half-width a, so the curvature runs monotonically from one side
      to the other, with no overshoot. a is the widest transition that leaves at most
      C2_DEVIATION_TARGET (1e-6 c) of shift (kappa_jump a^2 / 18) and cuts a corner by at
      most that (35/256 |turn| a); each shift is taken back over 6 a. There the C2 surface
      stays within 1e-6 c of C1.
    - Ahead of it, the C2 leading edge (below), which replaces the legacy LE circle end and
      polar fair. The legacy fair is a historical construction, not NACA data: it turns
      back on the pressure side of cambered blades (4.3 deg for the 65-(12A10)10, 8.1 deg at
      cl0 1.6) and wiggles on thin blades (a "chin", up to 82 deg of reverse turning on the
      polyline at t/c 0.04). The C2 LE is not held to 1e-6 c of it.
    - Imposed exactly: the LE point, its unit tangent and curvature 1/R_LE (the two surfaces
      join G2 at the apex); the TE point and tangent, plus the curvature of a rounded TE. A
      sharp emery_base TE stays a corner, each surface ending along its own C1 closing
      segment (the C1 wedge angle); C2 holds inside each surface up to the TE, not around it.
    The C2 arrays are the curves sampled where the C1 vertices are (as many points, both ends
    exactly the C1 LE and TE vertices; over the LE zone at the same fractions of its length,
    after it at the same arc lengths), so grid.py reads them as it reads C1's.
    Cascade.normalized_curves and .active_curves hold the curves (ParametricBladeCurve: X,
    X', X'', curvature, dy/dm and d2y/dm2 in u). The LE, TE, chord, pitch, stagger, LE and TE
    circles and the blade angles (the equivalent arc's) are those of C1; the throat is
    measured as for C1, on the arrays. The curves do not depend on n_camber_stations.

The C2 leading edge (fair_leading_edge; family independent, NACA 65 supplies its inputs)
    Inputs: a LeadingEdgeNose (apex, nose circle center, R_LE and its source: for NACA 65 the
    legacy reference radius 0.00687 (t/0.10)^2, metadata le_radius_source) and, per surface,
    a source surface in arc length (for NACA 65 the SmoothedSurface from station 0.005 on).
    Each surface, with the same rule:
    - Local LE frame: the nose axis from the apex to the circle center, and its normal. The
      fair is written in Whewell form, radius of curvature r against the turning angle phi
      from the apex tangent, so the tangent turns monotonically by construction.
    - r = R_LE on the nose circle, phi <= LE_FAIR_MIN_ARC (20 deg, the legacy minimum arc);
      beyond, a cubic B-spline in phi with non-decreasing control radii, the first three
      R_LE (r' = r'' = 0: G4 where it leaves the circle), the last the source's radius of
      curvature at the anchor (G2). Non-decreasing r: the curvature falls monotonically from
      1/R_LE, never exceeds it, never changes sign (no inflection, no S-bend).
    - The landing on the anchor point and its curvature are linear in the control radii;
      among all such fairs the one with least int (dkappa/ds)^2 ds (= int r'^2 / r^5 dphi,
      the minimum-variation-of-curvature energy, by reweighted least squares) is taken, an
      equality- and bound-constrained least-squares problem solved exactly (_lsei).
    - Anchor: LE_ANCHOR_OFFSET (0.5) R_LE of source arc length past the earliest anchor at
      which such a fair exists (found by scan and bisection), so the anchor moves
      continuously with the blade parameters. Not "the earliest anchor within a fairness
      budget": that energy is not monotone along the source and its first crossing switched
      branches (2e-5 c jumps of the LE, measured).
    The fair is G2 with the source and the C2 spline represents it to 1e-9 c, its curvature
    to 0.16 /c; metadata carries per surface the departure (curvature 0.1 % below 1/R_LE),
    the anchor, the fairness R_LE^3 int kappa_s^2 ds, and the LE-zone deviation from C1.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields, replace
from functools import lru_cache
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import BSpline, CubicSpline, PchipInterpolator
from scipy.linalg import lstsq, null_space, solve
from scipy.optimize import brentq, minimize_scalar, nnls
from scipy.special import xlogy

Points = NDArray[np.float64]
HalfThickness = Callable[[NDArray[np.float64]], NDArray[np.float64]]

BLADE_TYPES = ("NACA65", "DCA", "C4")
MEANLINE_TYPES = ("A10",)
PROFILE_VARIANTS = ("kovach_sandercock", "emery_base")

# Aungier 2003 Table 4-1, p. 63, NACA 65-series columns, in percent of chord, as printed.
_TABLE_X_PCT = (
    0.0, 0.5, 0.75, 1.25, 2.5, 5.0, 7.5, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0,
    45.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0, 100.0,
)
_TABLE_CAMBER_PCT = (       # a = 1.0 mean line, Cl0 = 1
    0.0, 0.250, 0.350, 0.535, 0.930, 1.580, 2.120, 2.585, 3.365, 3.980,
    4.475, 4.860, 5.150, 5.355, 5.475, 5.515, 5.475, 5.355, 5.150, 4.860,
    4.475, 3.980, 3.365, 2.585, 1.580, 0.0,
)
_TABLE_THICKNESS_PCT = (    # 65-010 thickness; 3.480 at 2.5 % is the misprint
    0.0, 1.544, 1.864, 2.338, 3.480, 4.354, 5.294, 6.080, 7.332, 8.286,
    9.006, 9.520, 9.848, 9.992, 9.926, 9.624, 9.060, 8.292, 7.364, 6.312,
    5.168, 3.974, 2.770, 1.620, 0.612, 0.0,
)
TABLE_CORRECTIONS = {0.025: 0.03148}    # x/c: corrected thickness / c
TABLE_LE_RADIUS = 0.00687               # R_LE / c of the tabulated 65-010 form
TABLE_T_MAX = 0.10                      # nominal thickness of the tabulated form
TABLE_SOURCE = "Aungier 2003 Table 4-1, p. 63 (Emery et al. 1958)"

STATION_READING = "arc"         # thickness station = s / S along the camberline [Aungier 2003 Sec. 4.1]
ARC_CAMBER_COEFF = 0.1103       # tan(camber_angle / 4) per unit Cl0 [Aungier 2003 Eq. 4-6]
THROAT_CORRELATION_COEFFS = (0.05, 1.5, 5.0, 2.0)   # a, n, b, d of Eq. 4-30, degrees
KOVACH_START_STATION = 0.60     # the table thickness is kept up to here
KOVACH_TE_RADIUS = 0.008        # TE radius / chord

XI_MIN = 1e-3                   # mean-line slope is singular at the ends; arc length is integrated inside
N_ARC_LENGTH = 2001             # cosine-spaced points of that integral
FIRST_TABLE_STATION = 0.005     # the surfaces start here; the LE closure covers the stretch ahead
LE_TANGENT_X = 0.005            # mean-line tangent that sets the nose axis [standard NACA practice]
LE_ARC_MIN_ANGLE = math.radians(20.0)
LE_ARC_SEARCH_STEP = math.radians(0.5)
LE_FAIR_MIN_SPAN = math.radians(10.0)
LE_ARC_STEP = math.radians(1.25)        # polyline step on the LE circle
N_LE_FAIR = 41
X_NOSE_END = 0.03               # geometric stations replace the cosine ones up to here
N_NOSE_STATIONS = 16
TE_FAIR_SPAN = math.radians(20.0)       # Kovach only; counts are per surface
TE_NEAREST_RANGE = 0.005
TE_MIN_STEP = 1e-4
N_TE_FAIR = 16
N_TE_ARC = 32
TE_POINT = np.array([1.0, 0.0])
TE_POINT.flags.writeable = False

CHECKED_CL0_MAX = 1.6           # range this construction has been checked on
CHECKED_T_MAX = 0.10
MIN_CAMBER_STATIONS = 21

CANDIDATE_MARGIN = 1e-3         # chords: local minima this close to the best are refined
MAX_THROAT_ITERATIONS = 200
THROAT_PARAMETER_TOL = 1e-13    # chords of spline parameter: refinement converged
ARMIJO = 1e-4                   # sufficient-decrease constant of the line search
MIN_STEP_FRACTION = 1e-12
NEWTON_ZONE = 1e-6              # chords: Newton steps this small are taken without line search

EXACT_TOL = 1e-12               # chords: quantities that hold by construction
EDGE_TOL = 1e-10                # chords: points on the LE and TE circles (the TE center is a root solve)
TE_CIRCLE_RTOL = 1e-3           # fraction of R_TE a surface may dip inside the TE circle, see _check_edges
THROAT_TOL = 1e-9               # chords: suction/pressure throat against the whole-blade one
THICKNESS_RTOL = 2e-3           # inscribed thickness against t_max_over_c

SURFACE_CONTINUITIES = ("C1", "C2")
SLOPE_MIN_COS = math.sqrt(np.finfo(float).eps)  # |cos(beta)| below this: y(m) is singular (metrics.py's threshold)

# C2 quintic representation, see "Surface continuity" in the module docstring. Lengths in chords.
C2_DEGREE = 5
C2_DEVIATION_TARGET = 1e-6      # the shift one junction transition leaves, and the cut of a rounded corner
C2_MAX_HALF_WIDTH = 0.01        # half-width a of the widest junction transition
C2_COMPENSATION_RATIO = 6.0     # a transition's shift is taken back over this many half-widths
C2_KNOTS_PER_HALF_WIDTH = 3.0   # knot spacing <= a / 3 inside a transition
C2_KNOTS_PER_RADIUS = 20.0      # knot spacing <= radius of curvature / 20
C2_MAX_KNOT_SPACING = 0.01
C2_KNOT_GRADING = 0.3           # the knot spacing grows by at most 0.3 per unit length along the surface
C2_SAMPLES_PER_SPAN = 8         # reference samples per knot span in the least-squares fit
C2_SIZE_GRID = 20001            # points of the knot-spacing field along the surface
C2_COARSE_POINTS = 65           # samples per piece for the curvature that sets the knot spacing
C2_KAPPA_JUMP_TOL = 1e-6        # 1/chord: curvature jumps below this are left as they are
C2_TURN_TOL = 1e-10             # radians: tangent breaks below this are left as they are
C2_ANCHOR_TOL = 1e-12           # a C1 vertex must lie this close to the analytic reference surface
REFERENCE_TABLE_POINTS = 257    # per construction piece, for its arc length
PROJECTION_ITERATIONS = 8       # Newton steps of ParametricBladeCurve.project
CURVE_PARAMETER_SLACK = 1e-12   # u this far outside [0, 1] is round-off and is clipped

# The C2 leading edge (see "The C2 leading edge" in the module docstring)
LE_MODEL = "monotone_curvature_fair"
LE_RADIUS_SOURCE_NACA65 = "legacy_naca65_reference"
LE_FAIR_MIN_ARC = LE_ARC_MIN_ANGLE      # nose circle kept on either side of the apex, the legacy minimum
LE_ANCHOR_OFFSET = 0.5          # the anchor lies this many R_LE of source arc length past the earliest
                                # anchor at which a fair exists
LE_FAIRNESS_LIMIT = 4.0         # R_LE^3 int (dkappa/ds)^2 ds above this is refused; the legacy 65-(12A10)10
                                # suction fair, which has no defect, measures 2.74
LE_FAIR_SPANS = 24              # spans of the cubic B-spline r(phi) of the fair
LE_FAIR_GAUSS_POINTS = 16       # Gauss-Legendre points per span
LE_FAIR_ITERATIONS = 6          # reweightings of int r'^2 / r^5 (= int kappa_s^2 ds)
LE_ANCHOR_STEP = 0.125          # anchor scan step, in LE radii of source arc length
LE_ANCHOR_RANGE = 0.1           # fraction of the source length the anchor scan covers
LE_ANCHOR_BISECTIONS = 48
LE_CLOSURE_TOL = 1e-12          # chords: a fair lands on its anchor this exactly
LE_RADIUS_TOL = 1e-12           # relative: the nose center lies R_LE from the apex
LE_FAIR_FIXED_CONTROLS = 3      # control radii held at R_LE: r' = r'' = 0 where the fair leaves the circle (G4)
LE_DEPARTURE_DROP = 1e-3        # the reported departure: where the curvature is 0.1 % below 1/R_LE
LE_KNOTS_PER_CURVATURE_CHANGE = 40.0    # in the LE fair the curvature changes by at most 1/40 of itself per knot span
LE_ANCHOR_KNOT_SPACING = 0.01   # knot spacing / R_LE within LE_ANCHOR_KNOT_ZONE R_LE of the departure and of the
LE_ANCHOR_KNOT_ZONE = 0.5       # anchor: a derivative of the curvature jumps at either, which a C4 spline
                                # resolves only over its spans
LDP_TOL = 1e-12                 # residual of the NNLS that signals an empty feasible set
SMOOTH_SUBDIVISIONS = 8         # sub-intervals of a transition support in SmoothedSurface
SMOOTH_GAUSS_POINTS = 16
SMOOTH_NEWTON_STEPS = 3         # arc length -> piece parameter


class GeometryError(ValueError):
    """A blade or cascade that breaks a geometric invariant."""


class GeometryWarning(UserWarning):
    """Inputs outside the range the NACA 65 construction has been checked on."""


# ---------------------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------------------

def _freeze_arrays(instance: object) -> None:
    """Replace every array field of a frozen dataclass by a read-only float copy."""
    for item in fields(instance):
        value = getattr(instance, item.name)
        if isinstance(value, np.ndarray):
            array = np.array(value, dtype=float)
            array.flags.writeable = False
            object.__setattr__(instance, item.name, array)


@dataclass(frozen=True)
class BaseTables:
    """The NACA 65-series columns of Aungier Table 4-1 as fractions of chord, kept for
    validation [Aungier 2003 Table 4-1, from Emery et al. 1958]. camber is the a = 1.0 mean
    line at Cl0 = 1; the thickness columns are the 65-010 form."""

    x: NDArray[np.float64]
    camber: NDArray[np.float64]
    thickness_printed: NDArray[np.float64]
    thickness_corrected: NDArray[np.float64]
    corrections: Mapping[float, float]      # x/c: corrected thickness
    leading_edge_radius: float              # of the tabulated 65-010 form
    source: str

    def __post_init__(self) -> None:
        _freeze_arrays(self)


@dataclass(frozen=True)
class BladeProfile:
    """One isolated blade in the blade frame, chord 1, before stagger. Point arrays are
    (n, 2) of (x, y); the surfaces run LE -> TE. Angles are radians."""

    camberline: Points                      # actual mean line: builds the surfaces
    equivalent_arc_camberline: Points       # reference only: defines the blade angles
    suction_surface: Points
    pressure_surface: Points
    leading_edge_center: NDArray[np.float64]
    leading_edge_radius: float
    trailing_edge_center: NDArray[np.float64]   # the TE point itself when the TE is sharp
    trailing_edge_radius: float                 # 0 for a sharp TE
    camber_angle: float                     # equivalent circular arc
    chi_le: float
    chi_te: float
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        _freeze_arrays(self)


@dataclass(frozen=True)
class CascadeCoordinates:
    """Every length of a cascade, in one unit: chords in Cascade.normalized, the active
    unit in Cascade.active. Point arrays are (n, 2); the surfaces run LE -> TE. Fields
    ending in _unstaggered are in the blade frame, all others in the cascade frame."""

    chord: float
    pitch: float
    camberline_unstaggered: Points
    equivalent_arc_camberline_unstaggered: Points
    suction_surface_unstaggered: Points
    pressure_surface_unstaggered: Points
    camberline: Points
    equivalent_arc_camberline: Points
    suction_surface: Points
    pressure_surface: Points
    leading_edge_point: NDArray[np.float64]
    trailing_edge_point: NDArray[np.float64]
    leading_edge_center: NDArray[np.float64]
    leading_edge_radius: float
    trailing_edge_center: NDArray[np.float64]
    trailing_edge_radius: float
    periodic_translation: NDArray[np.float64]   # neighbor blade = reference blade + this
    throat_opening: float
    throat_point_reference: NDArray[np.float64]  # on the suction surface of the reference blade
    throat_point_neighbor: NDArray[np.float64]   # on the pressure surface of the neighbor blade

    def __post_init__(self) -> None:
        _freeze_arrays(self)
        for item in fields(self):
            if not isinstance(getattr(self, item.name), np.ndarray):
                object.__setattr__(self, item.name, float(getattr(self, item.name)))

    def scaled(self, factor: float) -> CascadeCoordinates:
        """The same cascade with every length multiplied by factor: every field is a length."""
        return replace(self, **{item.name: getattr(self, item.name) * factor for item in fields(self)})


@dataclass(frozen=True, eq=False)
class ParametricBladeCurve:
    """One blade surface as a parametric curve X(u) = (m(u), y(u)), 0 <= u <= 1, from the LE
    (u = 0) to the TE (u = 1): a clamped B-spline of degree 5 whose interior knots are simple,
    so it is C4 at every knot (the contract is at least C2). m is the curve's first
    coordinate: the axial direction in the cascade frame, the chord in the blade frame.

    u is arc length along the validated C1 surface divided by its length, reference_length
    (in the curve's own length unit), so |X'(u)| stays within about 1e-5 of
    reference_length. Immutable: the arrays are read-only and every method returns new
    arrays; derivatives are with respect to u."""

    knots: NDArray[np.float64]
    control_points: Points
    reference_length: float
    degree: int

    def __post_init__(self) -> None:
        _freeze_arrays(self)
        object.__setattr__(self, "reference_length", float(self.reference_length))
        k, t, c = int(self.degree), self.knots, self.control_points
        if k < 1 or c.ndim != 2 or c.shape[1] != 2 or len(t) != len(c) + k + 1:
            raise ValueError("a ParametricBladeCurve needs (n, 2) control points and n + degree + 1 knots")
        inner = t[k:len(t) - k]
        if not (np.all(t[:k + 1] == 0.0) and np.all(t[-k - 1:] == 1.0) and np.all(np.diff(inner) > 0.0)):
            raise ValueError("the knot vector must be clamped on [0, 1] with simple interior knots")
        object.__setattr__(self, "_spline", BSpline(t, c, k))

    @property
    def domain(self) -> tuple[float, float]:
        return (0.0, 1.0)

    @property
    def interior_knots(self) -> NDArray[np.float64]:
        """The simple interior knots: the joins between polynomial pieces."""
        return self.knots[self.degree + 1:len(self.knots) - self.degree - 1]

    def evaluate(self, u: ArrayLike) -> NDArray[np.float64]:
        """X(u): shape (2,) for a scalar u, (n, 2) for n parameters. X(0) and X(1) are the
        end control points exactly, the LE and TE landmarks."""
        return self.derivative(u, 0)

    def derivative(self, u: ArrayLike, order: int = 1) -> NDArray[np.float64]:
        """The order-th derivative with respect to u, order 0 to degree."""
        if int(order) != order or not 0 <= order <= self.degree:
            raise ValueError(f"order must be an integer from 0 to {self.degree}, got {order}")
        u = _curve_parameter(u)
        value = np.asarray(self._spline(u, int(order)), dtype=float)
        if order == 0:      # de Boor at a clamped end is exact only to round-off with many knots
            value = np.where((u == 0.0)[..., None], self.control_points[0], value)
            value = np.where((u == 1.0)[..., None], self.control_points[-1], value)
        return value

    def unit_tangent(self, u: ArrayLike) -> NDArray[np.float64]:
        d1 = self.derivative(u, 1)
        return d1 / np.hypot(d1[..., 0], d1[..., 1])[..., None]

    def curvature(self, u: ArrayLike) -> NDArray[np.float64]:
        """Signed curvature, see signed_curvature."""
        return signed_curvature(self.derivative(u, 1), self.derivative(u, 2))

    def wall_slope(self, u: ArrayLike, min_cos: float = SLOPE_MIN_COS) -> tuple[NDArray[np.float64],
                                                                               NDArray[np.float64]]:
        """(tan(beta), d tan(beta) / dm) = (dy/dm, d2y/dm2), NaN where |cos(beta)| < min_cos;
        see wall_slopes."""
        return wall_slopes(self.derivative(u, 1), self.derivative(u, 2), min_cos)

    def project(self, points: ArrayLike, u_guess: ArrayLike | None = None) -> tuple[NDArray[np.float64],
                                                                                   NDArray[np.float64]]:
        """(u, signed normal distance) of the nearest curve point to each point, by Newton
        steps from u_guess (or from the nearest of 4097 curve samples). The distance is
        positive on the curve's left, the side its left normal points to."""
        points = np.atleast_2d(np.asarray(points, dtype=float))
        if u_guess is None:
            grid = np.linspace(0.0, 1.0, 4097)
            samples = self.evaluate(grid)
            u = grid[np.argmin(np.hypot(points[:, None, 0] - samples[None, :, 0],
                                        points[:, None, 1] - samples[None, :, 1]), axis=1)]
        else:
            u = np.array(np.broadcast_to(np.asarray(u_guess, dtype=float), len(points)))
        for _ in range(PROJECTION_ITERATIONS):
            gap = self.evaluate(u) - points
            d1, d2 = self.derivative(u, 1), self.derivative(u, 2)
            slope = np.sum(gap * d1, axis=1)
            u = np.clip(u - slope / (np.sum(d1 * d1, axis=1) + np.sum(gap * d2, axis=1)), 0.0, 1.0)
        tangent = self.unit_tangent(u)
        gap = points - self.evaluate(u)
        return u, tangent[:, 0] * gap[:, 1] - tangent[:, 1] * gap[:, 0]

    def scaled(self, factor: float) -> ParametricBladeCurve:
        """The same curve with every length multiplied by factor; u is unchanged."""
        return ParametricBladeCurve(self.knots, self.control_points * factor, self.reference_length * factor,
                                    self.degree)

    def rotated(self, angle: float) -> ParametricBladeCurve:
        """The curve turned counterclockwise by angle (radians) about the origin, as
        rotate_by_stagger turns points."""
        return ParametricBladeCurve(self.knots, rotate_by_stagger(self.control_points, angle),
                                    self.reference_length, self.degree)


@dataclass(frozen=True)
class SurfaceCurves:
    """The authoritative surfaces of a C2 cascade, in one length unit (chords in
    Cascade.normalized_curves, the active unit in Cascade.active_curves): both surfaces in
    the cascade frame and, with _unstaggered, in the blade frame, LE -> TE. The point arrays
    of CascadeCoordinates are these curves sampled at suction_parameters and
    pressure_parameters, the same u in either frame and unit."""

    suction: ParametricBladeCurve
    pressure: ParametricBladeCurve
    suction_unstaggered: ParametricBladeCurve
    pressure_unstaggered: ParametricBladeCurve
    suction_parameters: NDArray[np.float64]
    pressure_parameters: NDArray[np.float64]

    def __post_init__(self) -> None:
        _freeze_arrays(self)

    def scaled(self, factor: float) -> SurfaceCurves:
        """The same curves with every length multiplied by factor; the parameters stay."""
        names = ("suction", "pressure", "suction_unstaggered", "pressure_unstaggered")
        return replace(self, **{name: getattr(self, name).scaled(factor) for name in names})


@dataclass(frozen=True)
class CascadeSpec:
    """Everything that defines a cascade, as the user states it: geometry inputs only, no
    flow, grid or solver settings. make_cascade(spec) turns it into a Cascade.

    The defaults below are the only modelling defaults; no function behind make_cascade
    has its own. stagger_deg is in degrees from the axial direction, and make_cascade
    converts it to radians once. solidity is chord / pitch; the pitch follows from it. For
    NACA 65, t_max_over_c is the nominal thickness ratio of the designation (0.10 for a
    "...10" blade, whose true maximum is 0.9992 of that). physical_chord, in metres, scales
    the chord-1 blade uniformly; None keeps the cascade nondimensional. n_camber_stations
    sets the resolution of the camberline and of the surfaces built along it.

    surface_continuity chooses the representation of the surfaces, the one switch for it:
    "C1" is the validated construction itself, polylines on a surface that is C1 (C0 at the
    Kovach station 0.60), the legacy baseline; "C2" is the robust design geometry, two
    quintic B-spline curves, at least C2 inside each surface, equal to C1 within 1e-6 c
    downstream of the LE and with a redesigned, monotone-curvature leading edge, sampled into
    the same arrays (see "Surface continuity" and "The C2 leading edge" in the module
    docstring). Exactly "C1" or "C2".
    """

    blade_type: str
    solidity: float
    stagger_deg: float
    cl0: float
    t_max_over_c: float
    meanline_type: str = "A10"
    profile_variant: str = "kovach_sandercock"
    physical_chord: float | None = None
    n_camber_stations: int = 101
    surface_continuity: str = "C1"


@dataclass(frozen=True)
class Cascade:
    """A blade row as make_cascade generated and checked it; grid.py consumes it. The inputs
    it was built from are in spec, and only there. Lengths are in active (the unit grid.py
    uses) and in normalized (chord 1); every other field is dimensionless or an angle in
    radians.

    The blade angles belong to the equivalent circular arc, not to the actual camberline:
    they are reference angles, see the module docstring. Where the throat lies is given
    only by its two end points, active.throat_point_reference and .throat_point_neighbor.

    For surface_continuity "C2", normalized_curves and active_curves hold the authoritative
    parametric surfaces whose samples are the arrays of normalized and active; for "C1"
    both are None and the arrays are the geometry."""

    spec: CascadeSpec                       # the inputs; frozen, so they cannot drift from the geometry
    stagger: float                          # radians(spec.stagger_deg)
    camber_angle: float
    chi_le: float
    chi_te: float
    kappa_le: float                         # stagger + chi_le
    kappa_te: float                         # stagger - chi_te
    throat_over_pitch: float                # geometric: the preferred value
    throat_over_chord: float                # geometric
    throat_over_pitch_correlation: float    # Aungier Eqs. 4-30, 4-31: a diagnostic only
    normalized: CascadeCoordinates
    active: CascadeCoordinates
    normalized_curves: SurfaceCurves | None     # C2 only
    active_curves: SurfaceCurves | None         # C2 only
    metadata: Mapping[str, Any]

    @property
    def is_dimensional(self) -> bool:
        """True when spec.physical_chord is set, so active lengths are in metres."""
        return self.spec.physical_chord is not None

    def angles_deg(self) -> dict[str, float]:
        """The angles in degrees, for printing; every key carries the _deg suffix."""
        names = ("stagger", "camber_angle", "chi_le", "chi_te", "kappa_le", "kappa_te")
        return {f"{name}_deg": math.degrees(getattr(self, name)) for name in names}


@dataclass(frozen=True)
class StreamSheet:
    """The axisymmetric stream surface of the blade-to-blade flow [Aungier 2003 Fig. 5-2],
    sampled at increasing m: radius r and thickness b normal to the sheet. A plane cascade
    has r = b = 1 at every m."""

    m: NDArray[np.float64]
    r: NDArray[np.float64]
    b: NDArray[np.float64]


# ---------------------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------------------

def make_cascade(spec: CascadeSpec) -> Cascade:
    """The checked cascade that spec describes.

    Raises TypeError if spec is not a CascadeSpec, ValueError for an invalid input,
    NotImplementedError for a blade family or mean line not built yet, and GeometryError if
    the result breaks an invariant of validate_cascade.
    """
    if not isinstance(spec, CascadeSpec):
        raise TypeError(f"make_cascade takes a CascadeSpec, got {type(spec).__name__}")
    if spec.blade_type not in BLADE_TYPES:
        raise ValueError(f"unknown blade_type {spec.blade_type!r}; expected one of {BLADE_TYPES}")
    _check_cascade_inputs(spec.solidity, spec.stagger_deg, spec.physical_chord)
    if spec.surface_continuity not in SURFACE_CONTINUITIES:
        raise ValueError(f"unknown surface_continuity {spec.surface_continuity!r}; expected one of "
                         f"{SURFACE_CONTINUITIES}")
    stagger = math.radians(spec.stagger_deg)        # the one degree-to-radian conversion

    curves = None
    if spec.blade_type == "NACA65":
        profile = naca65_profile(spec.cl0, spec.t_max_over_c, meanline_type=spec.meanline_type,
                                 profile_variant=spec.profile_variant, n_camber_stations=spec.n_camber_stations)
        correlation = naca65_throat_correlation(spec.solidity, stagger, spec.cl0, spec.t_max_over_c)
        if spec.surface_continuity == "C2":
            profile, curves = _naca65_c2_profile(profile, spec.cl0, spec.t_max_over_c, spec.profile_variant)
    else:
        raise NotImplementedError(f"blade_type {spec.blade_type!r} is part of the API but not implemented yet; "
                                  "only 'NACA65' is")

    normalized = place_cascade(profile, spec.solidity, stagger)
    active = normalized if spec.physical_chord is None else normalized.scaled(spec.physical_chord)
    normalized_curves = active_curves = None
    if curves is not None:
        (suction, suction_u), (pressure, pressure_u) = curves
        normalized_curves = SurfaceCurves(suction.rotated(stagger), pressure.rotated(stagger), suction, pressure,
                                          suction_u, pressure_u)
        active_curves = (normalized_curves if spec.physical_chord is None
                         else normalized_curves.scaled(spec.physical_chord))
    cascade = Cascade(
        spec=spec,
        stagger=stagger,
        camber_angle=profile.camber_angle,
        chi_le=profile.chi_le,
        chi_te=profile.chi_te,
        kappa_le=stagger + profile.chi_le,
        kappa_te=stagger - profile.chi_te,
        throat_over_pitch=normalized.throat_opening / normalized.pitch,
        throat_over_chord=normalized.throat_opening / normalized.chord,
        throat_over_pitch_correlation=correlation,
        normalized=normalized,
        active=active,
        normalized_curves=normalized_curves,
        active_curves=active_curves,
        metadata=MappingProxyType({**profile.metadata,
                                   "length_unit": "chord" if spec.physical_chord is None else "m",
                                   **({"le_radius": float(active.leading_edge_radius)} if curves is not None else {})}),
    )
    validate_cascade(cascade)
    return cascade


def _check_cascade_inputs(solidity: float, stagger_deg: float, physical_chord: float | None) -> None:
    if not (math.isfinite(solidity) and solidity > 0.0):
        raise ValueError(f"solidity must be a positive finite number (sigma = c / s), got {solidity}")
    if not (math.isfinite(stagger_deg) and -90.0 < stagger_deg < 90.0):
        raise ValueError(f"stagger_deg must lie strictly between -90 and 90 degrees, got {stagger_deg}")
    if physical_chord is not None and not (math.isfinite(physical_chord) and physical_chord > 0.0):
        raise ValueError(f"physical_chord must be a positive length in metres or None, got {physical_chord}")


# ---------------------------------------------------------------------------------------
# NACA 65 family: data, mean line, thickness
# ---------------------------------------------------------------------------------------

def base_tables() -> BaseTables:
    """Table 4-1 as printed and as corrected, fractions of chord [Aungier 2003 Table 4-1]."""
    x = np.array(_TABLE_X_PCT) / 100.0
    return BaseTables(x, np.array(_TABLE_CAMBER_PCT) / 100.0, np.array(_TABLE_THICKNESS_PCT) / 100.0,
                      _corrected_thickness(), MappingProxyType(dict(TABLE_CORRECTIONS)), TABLE_LE_RADIUS,
                      TABLE_SOURCE)


def _corrected_thickness() -> NDArray[np.float64]:
    """The 65-010 thickness column with TABLE_CORRECTIONS applied, fractions of chord."""
    x, thickness = np.array(_TABLE_X_PCT) / 100.0, np.array(_TABLE_THICKNESS_PCT) / 100.0
    for station, value in TABLE_CORRECTIONS.items():
        thickness[np.isclose(x, station)] = value
    return thickness


BASE_MAX_THICKNESS = float(np.max(_corrected_thickness()))       # 0.09992 at station 0.40
_BASE_HALF_THICKNESS = PchipInterpolator(np.sqrt(np.array(_TABLE_X_PCT) / 100.0), 0.5 * _corrected_thickness())


def a10_camberline(x: ArrayLike, cl0: float) -> NDArray[np.float64]:
    """y/c of the a = 1.0 mean line at chord stations x/c [Table 4-1; thin-airfoil closed form].

        y/c = -(cl0 / 4 pi) [ (1 - x) ln(1 - x) + x ln x ]

    This is the Cl0 = 1 line of Table 4-1 scaled by cl0; x = 0 and 1 give the limit 0.
    """
    x = np.asarray(x, dtype=float)
    if np.any((x < 0.0) | (x > 1.0)):
        raise ValueError("x must lie in [0, 1]")
    return -cl0 / (4.0 * np.pi) * (xlogy(1.0 - x, 1.0 - x) + xlogy(x, x))


def a10_slope_angle(x: ArrayLike, cl0: float) -> NDArray[np.float64]:
    """Angle (radians) of the a = 1.0 mean line to the chord, from
    dy/dx = (cl0 / 4 pi) ln((1 - x) / x). Singular at both ends, so x must lie in (0, 1)."""
    return np.arctan(_a10_slope(x, cl0))


def _a10_slope(x: ArrayLike, cl0: float) -> NDArray[np.float64]:
    x = np.asarray(x, dtype=float)
    if np.any((x <= 0.0) | (x >= 1.0)):
        raise ValueError("the mean-line slope is singular at the ends: x must lie in (0, 1)")
    return cl0 / (4.0 * np.pi) * np.log((1.0 - x) / x)


@lru_cache(maxsize=64)
def _arc_length_spline(cl0: float) -> CubicSpline:
    """Mean-line arc length from x = XI_MIN, as a cubic spline in x on [XI_MIN, 1 - XI_MIN].

    Trapezoid rule on N_ARC_LENGTH cosine-clustered points, so the result does not depend
    on how the caller spaces x. The cubic keeps s smooth: a linear interpolant would put a
    small kink in the surfaces at every integration point.
    """
    edge = math.acos(1.0 - 2.0 * XI_MIN)
    x = 0.5 * (1.0 - np.cos(np.linspace(edge, math.pi - edge, N_ARC_LENGTH)))
    s = cumulative_trapezoid(np.sqrt(1.0 + _a10_slope(x, cl0) ** 2), x, initial=0.0)
    return CubicSpline(x, s)


def _end_piece(cl0: float) -> float:
    """The mean line between x = 0 and XI_MIN (and between 1 - XI_MIN and 1, by symmetry),
    taken as straight."""
    return math.hypot(XI_MIN, float(a10_camberline(XI_MIN, cl0)))


def camberline_length(cl0: float) -> float:
    """Whole mean-line length S / c, end pieces included."""
    return float(_arc_length_spline(cl0)(1.0 - XI_MIN)) + 2.0 * _end_piece(cl0)


def camberline_station(x: ArrayLike, cl0: float) -> NDArray[np.float64]:
    """Table station s(x) / S of the camber point at x/c: arc length from the LE over the
    whole mean-line length [Aungier 2003 Sec. 4.1]. 0 at the LE and 1 at the TE."""
    x = np.asarray(x, dtype=float)
    spline, piece = _arc_length_spline(cl0), _end_piece(cl0)
    s = (spline(np.clip(x, spline.x[0], spline.x[-1]))
         + piece * np.minimum(x / XI_MIN, 1.0)
         + piece * np.maximum((x - 1.0) / XI_MIN + 1.0, 0.0))
    return np.clip(s / camberline_length(cl0), 0.0, 1.0)


def thickness_scale(t_max_over_c: float) -> float:
    """Factor on the corrected 65-010 ordinates for a nominal thickness ratio t_max_over_c:
    t_max_over_c / 0.10, the standard NACA practice. 0.10 leaves the table unchanged."""
    return t_max_over_c / TABLE_T_MAX


def _half_thickness(station: ArrayLike, t_nominal: float) -> NDArray[np.float64]:
    """Half-thickness of the corrected base form at table stations, scaled by t_nominal / 0.10."""
    station = np.asarray(station, dtype=float)
    if np.any((station < 0.0) | (station > 1.0)):
        raise ValueError("station must lie in [0, 1]")
    return t_nominal / TABLE_T_MAX * _BASE_HALF_THICKNESS(np.sqrt(station))


def naca65_base_thickness(station: ArrayLike, t_max_over_c: float) -> NDArray[np.float64]:
    """Full thickness t/c of the corrected base distribution at table stations s / S, for
    the nominal thickness ratio t_max_over_c [Aungier 2003 Table 4-1, corrected]. Its
    maximum is 0.9992 t_max_over_c, at station 0.40. Without the Kovach modification; the
    two variants agree up to station 0.60."""
    return 2.0 * _half_thickness(station, TABLE_T_MAX * thickness_scale(t_max_over_c))


def _kovach_half_thickness(station: ArrayLike, end_station: float, end_half: float,
                           t_nominal: float) -> NDArray[np.float64]:
    """Half-thickness with the Kovach & Sandercock TE: the table up to station 0.60, then
    linear to end_half at end_station [Aungier 2003 Sec. 4.2, Kovach & Sandercock 1961]."""
    station = np.asarray(station, dtype=float)
    start = _half_thickness(KOVACH_START_STATION, t_nominal)
    taper = start + (end_half - start) * (station - KOVACH_START_STATION) / (end_station - KOVACH_START_STATION)
    return np.where(station <= KOVACH_START_STATION, _half_thickness(station, t_nominal), taper)


# ---------------------------------------------------------------------------------------
# NACA 65 family: angles and throat correlation
# ---------------------------------------------------------------------------------------

def equivalent_arc_angles(cl0: float) -> tuple[float, float, float]:
    """(camber_angle, chi_le, chi_te) of the equivalent circular arc, radians
    [Aungier 2003 Eq. 4-6]: tan(camber_angle / 4) = 0.1103 cl0, chi_le = chi_te = camber_angle / 2."""
    camber = 4.0 * math.atan(ARC_CAMBER_COEFF * cl0)
    return camber, 0.5 * camber, 0.5 * camber


def circular_arc_camberline(x: ArrayLike, camber_angle: float) -> NDArray[np.float64]:
    """y/c of the circular arc through (0, 0) and (1, 0) whose end slopes are
    +- tan(camber_angle / 2). Written so that it stays exact as camber_angle -> 0:

        y = sin(a) x (1 - x) / ( sqrt(1/4 - sin(a)^2 (x - 1/2)^2) + cos(a) / 2 ),   a = camber_angle / 2
    """
    x = np.asarray(x, dtype=float)
    half = 0.5 * camber_angle
    return math.sin(half) * x * (1.0 - x) / (np.sqrt(0.25 - (math.sin(half) * (x - 0.5)) ** 2) + 0.5 * math.cos(half))


def naca65_throat_correlation(solidity: float, stagger: float, cl0: float, t_max_over_c: float) -> float:
    """Empirical o / s of a NACA 65 cascade, stagger in radians [Aungier 2003 Eqs. 4-30, 4-31]:

        phi = gamma (1 - a Cl0^n) + b Cl0^n - d       (degrees; a, n, b, d = 0.05, 1.5, 5, 2)
        o/s = [ (1 - (t/c) sqrt(sigma)) cos(phi) ]^sqrt(sigma)

    A diagnostic to compare with the geometric throat, never a replacement for it.
    """
    if cl0 < 0.0:
        raise ValueError("the correlation is defined for cl0 >= 0")
    a, n, b, d = THROAT_CORRELATION_COEFFS
    camber = cl0**n
    phi = math.degrees(stagger) * (1.0 - a * camber) + b * camber - d
    root = math.sqrt(solidity)
    return float(((1.0 - t_max_over_c * root) * math.cos(math.radians(phi))) ** root)


# ---------------------------------------------------------------------------------------
# NACA 65 family: the blade
# ---------------------------------------------------------------------------------------

def naca65_profile(cl0: float, t_max_over_c: float, *, meanline_type: str, profile_variant: str,
                   n_camber_stations: int) -> BladeProfile:
    """An isolated NACA 65 blade in the blade frame, chord 1 [Aungier 2003 Secs. 4.1, 4.2].
    See the module docstring for the construction. No defaults here: they live in
    CascadeSpec only, so they cannot drift."""
    _check_naca65_inputs(cl0, t_max_over_c, meanline_type, profile_variant, n_camber_stations)
    t_nominal = TABLE_T_MAX * thickness_scale(t_max_over_c)
    try:
        contour = _naca65_contour(cl0, t_nominal, profile_variant, n_camber_stations)
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
        raise GeometryError(f"NACA 65 construction failed for cl0={cl0}, t_max_over_c={t_max_over_c}: {error}") from error

    suction, pressure = split_at_leading_edge(contour.loop)
    x = cosine_stations(n_camber_stations)
    camber_angle, chi_le, chi_te = equivalent_arc_angles(cl0)
    sharp = contour.te_center is None
    return BladeProfile(
        camberline=np.column_stack((x, a10_camberline(x, cl0))),
        equivalent_arc_camberline=np.column_stack((x, circular_arc_camberline(x, camber_angle))),
        suction_surface=suction,
        pressure_surface=pressure,
        leading_edge_center=contour.le_center,
        leading_edge_radius=contour.le_radius,
        trailing_edge_center=suction[-1] if sharp else contour.te_center,
        trailing_edge_radius=0.0 if sharp else KOVACH_TE_RADIUS,
        camber_angle=camber_angle,
        chi_le=chi_le,
        chi_te=chi_te,
        metadata=MappingProxyType(_naca65_metadata(cl0, t_max_over_c, meanline_type, profile_variant,
                                                   n_camber_stations)),
    )


def _check_naca65_inputs(cl0: float, t_max_over_c: float, meanline_type: str, profile_variant: str,
                         n_camber_stations: int) -> None:
    if meanline_type not in MEANLINE_TYPES:
        raise NotImplementedError(f"meanline_type {meanline_type!r} is not implemented; only {MEANLINE_TYPES} are")
    if profile_variant not in PROFILE_VARIANTS:
        raise ValueError(f"unknown profile_variant {profile_variant!r}; expected one of {PROFILE_VARIANTS}")
    if not (math.isfinite(cl0) and cl0 >= 0.0):
        raise ValueError(f"cl0 must be finite and >= 0, got {cl0}: a negative design lift coefficient "
                         "swaps the suction and pressure sides, which this module does not support")
    if not (math.isfinite(t_max_over_c) and t_max_over_c > 0.0):
        raise ValueError(f"t_max_over_c must be a positive finite number, got {t_max_over_c}")
    if int(n_camber_stations) != n_camber_stations or n_camber_stations < MIN_CAMBER_STATIONS:
        raise ValueError(f"n_camber_stations must be an integer >= {MIN_CAMBER_STATIONS}, got {n_camber_stations}")
    if cl0 > CHECKED_CL0_MAX or t_max_over_c > CHECKED_T_MAX:
        warnings.warn(f"cl0={cl0}, t_max_over_c={t_max_over_c} is outside the range the NACA 65 construction "
                      f"was checked on (cl0 <= {CHECKED_CL0_MAX}, t_max_over_c <= {CHECKED_T_MAX})",
                      GeometryWarning, stacklevel=3)


def _naca65_metadata(cl0: float, t_max_over_c: float, meanline_type: str, profile_variant: str,
                     n_camber_stations: int) -> dict[str, Any]:
    kovach = profile_variant == "kovach_sandercock"
    return {
        "designation": f"NACA 65-({10.0 * cl0:g}{meanline_type}){100.0 * t_max_over_c:g}",
        "source_tables": TABLE_SOURCE,
        "table_corrections": dict(TABLE_CORRECTIONS),
        "camberline": "a = 1.0 mean line, closed form, scaled by cl0",
        "thickness_station_reading": STATION_READING,
        "thickness_input": "nominal NACA thickness ratio; table scaled by t_max_over_c / 0.10",
        "base_max_thickness": BASE_MAX_THICKNESS,
        "thickness_scale_factor": thickness_scale(t_max_over_c),
        "max_thickness_over_c": thickness_scale(t_max_over_c) * BASE_MAX_THICKNESS,
        "leading_edge": f"nose circle R_LE = {TABLE_LE_RADIUS} (t_max_over_c / {TABLE_T_MAX})^2 "
                        "through the LE point, quintic fairs to station 0.005",
        "trailing_edge": (f"Kovach-Sandercock: table to station {KOVACH_START_STATION}, straight taper "
                          f"tangent to R_TE = {KOVACH_TE_RADIUS} c" if kovach else "sharp, R_TE = 0"),
        "blade_angles": "equivalent circular arc, Aungier 2003 Eq. 4-6",
        "throat_correlation": "Aungier 2003 Eqs. 4-30, 4-31",
        "n_camber_stations": int(n_camber_stations),
        "inside_checked_range": cl0 <= CHECKED_CL0_MAX and t_max_over_c <= CHECKED_T_MAX,
    }


@dataclass(frozen=True)
class _Contour:
    """A closed blade contour TE -> upper -> LE -> lower -> TE in the blade frame, with its
    LE circle and, for a rounded TE, the TE circle center (None when the TE is sharp)."""

    loop: Points
    le_center: NDArray[np.float64]
    le_radius: float
    te_center: NDArray[np.float64] | None


def _naca65_contour(cl0: float, t_nominal: float, profile_variant: str, n_camber_stations: int) -> _Contour:
    """Build the closed contour [standard NACA construction, Aungier 2003 Sec. 4.1, Table 4-1;
    "kovach_sandercock": Sec. 4.2, Kovach & Sandercock 1961].

        x_U = x_c - (t/2) sin(phi),    y_U = y_c + (t/2) cos(phi)
        x_L = x_c + (t/2) sin(phi),    y_L = y_c - (t/2) cos(phi)

    with phi the mean-line angle and t read at the station s / S. The surfaces start at the
    first table station; the LE circle and its fairs close the stretch ahead of it. Up to
    X_NOSE_END the stations are geometric, so the polyline stays as fair as the curve next
    to the LE closure.
    """
    if profile_variant == "kovach_sandercock":
        half, x_end, te_center = _kovach_closure(cl0, t_nominal)
    else:
        def half(station: NDArray[np.float64]) -> NDArray[np.float64]:
            return _half_thickness(station, t_nominal)
        x_end, te_center = 1.0 - XI_MIN, None

    edge = math.acos(1.0 - 2.0 * XI_MIN)
    x = 0.5 * (1.0 - np.cos(np.linspace(edge, math.pi - edge, n_camber_stations)))
    x_first = brentq(lambda q: camberline_station(q, cl0) - FIRST_TABLE_STATION, XI_MIN, 2.0 * LE_TANGENT_X)
    nose = np.geomspace(x_first, X_NOSE_END, N_NOSE_STATIONS)
    stations = np.concatenate((nose, x[(x > X_NOSE_END) & (x < x_end)], [x_end]))

    radius = TABLE_LE_RADIUS * (t_nominal / TABLE_T_MAX) ** 2
    phi_le = float(a10_slope_angle(LE_TANGENT_X, cl0))
    nose_axis = -np.array([np.cos(phi_le), np.sin(phi_le)])
    center = -radius * nose_axis

    def side_points(side: float) -> tuple[Points, float]:
        fair, arc_angle = _nose_fair(x_first, cl0, half, side, center, radius, nose_axis)
        if te_center is None:
            return np.vstack((fair, _surface(stations[1:], cl0, half, side), TE_POINT)), arc_angle
        x_last = _nearest_station(x_end, cl0, half, side, te_center)
        aft = stations[1:-1]
        surface = _surface(np.append(aft[aft < x_last - TE_MIN_STEP], x_last), cl0, half, side)
        return np.vstack((fair, surface, _te_cap(surface[-1], te_center, side))), arc_angle

    (upper, upper_angle), (lower, lower_angle) = side_points(1.0), side_points(-1.0)
    arc_angles = np.concatenate((_arc_steps(upper_angle), -_arc_steps(lower_angle)[-2::-1]))
    arc = center + radius * _polar(arc_angles, nose_axis)
    return _Contour(np.vstack((upper[::-1], arc[1:-1], lower)), center, radius, te_center)


def _surface(x: NDArray[np.float64], cl0: float, half: HalfThickness, side: float) -> Points:
    """Surface points at camber stations x; side is +1 upper, -1 lower."""
    h = half(camberline_station(x, cl0))
    phi = a10_slope_angle(x, cl0)
    return np.column_stack((x - side * h * np.sin(phi), a10_camberline(x, cl0) + side * h * np.cos(phi)))


def _arc_steps(angle: float) -> NDArray[np.float64]:
    """Angles from angle down to exactly 0, so that the LE stays a contour vertex."""
    return np.linspace(angle, 0.0, int(np.ceil(angle / LE_ARC_STEP)) + 1)


def _polar(angle: NDArray[np.float64], axis: NDArray[np.float64]) -> Points:
    """Unit vectors at angle from axis, positive toward the upper surface."""
    normal = np.array([axis[1], -axis[0]])
    return np.outer(np.cos(angle), axis) + np.outer(np.sin(angle), normal)


def _polar_coordinates(points: Points, center: NDArray[np.float64], axis: NDArray[np.float64],
                       side: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Radius about center and angle from axis, positive toward side."""
    relative = points - center
    normal = np.array([axis[1], -axis[0]])
    return np.hypot(relative[:, 0], relative[:, 1]), np.arctan2(side * (relative @ normal), relative @ axis)


def _fair_radius(u: NDArray[np.float64], radius: float, span: float, gap: float, slope: float,
                 curve: float) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Quintic r, dr/dtheta and d2r/dtheta2 of a fair over u in [0, 1].

    At u = 0 it sits on the circle to second order. At u = 1 it has r = radius + gap,
    dr/dtheta = slope and d2r/dtheta2 = curve.
    """
    slope, curve = slope * span, curve * span**2
    c3 = 10.0 * gap - 4.0 * slope + 0.5 * curve
    c4 = -15.0 * gap + 7.0 * slope - curve
    c5 = 6.0 * gap - 3.0 * slope + 0.5 * curve
    return (radius + u**3 * (c3 + u * (c4 + u * c5)),
            u**2 * (3.0 * c3 + u * (4.0 * c4 + 5.0 * u * c5)) / span,
            u * (6.0 * c3 + u * (12.0 * c4 + 20.0 * u * c5)) / span**2)


def _nose_fair(x_first: float, cl0: float, half: HalfThickness, side: float, center: NDArray[np.float64],
               radius: float, nose_axis: NDArray[np.float64]) -> tuple[Points, float]:
    """LE fair from the end of the circle arc to the surface at x_first, and that arc end.

    theta is measured from the nose axis, positive going aft on either surface. The end
    derivatives at the surface are one-sided aft, since the surface is only C1 at a table
    knot. Each side leaves the circle at the smallest angle from LE_ARC_MIN_ANGLE up that
    minimizes the peak |curvature| on its fair, so the circle carries the nose curvature.
    """
    probe = _surface(x_first + 2e-4 * np.arange(6), cl0, half, side)
    r, theta = _polar_coordinates(probe, center, nose_axis, side)
    fit = np.polyfit(theta - theta[0], r, 3)
    end = (r[0] - radius, fit[2], 2.0 * fit[1])

    def peak_curvature(start: float) -> float:
        rho, d1, d2 = _fair_radius(np.linspace(0.0, 1.0, 401), radius, theta[0] - start, *end)
        return float(np.max(np.abs(rho**2 + 2.0 * d1**2 - rho * d2) / (rho**2 + d1**2) ** 1.5))

    starts = np.arange(LE_ARC_MIN_ANGLE, theta[0] - LE_FAIR_MIN_SPAN, LE_ARC_SEARCH_STEP)
    peaks = np.array([peak_curvature(start) for start in starts])
    start = float(starts[np.argmax(peaks <= peaks.min() * (1.0 + 1e-6))])

    u = np.linspace(0.0, 1.0, N_LE_FAIR)
    span = theta[0] - start
    rho = _fair_radius(u, radius, span, *end)[0]
    return center + rho[:, None] * _polar(side * (start + span * u), nose_axis), start


def _kovach_closure(cl0: float, t_nominal: float) -> tuple[HalfThickness, float, NDArray[np.float64]]:
    """Thickness law, last surface station and TE circle center of the Kovach TE.

    The circle (radius KOVACH_TE_RADIUS) is centered on the mean line and passes through
    (1, 0). The taper is the straight line from the table thickness at station 0.60 that is
    tangent to that circle, written along the straightened camberline (s, h).
    """
    r = KOVACH_TE_RADIUS
    x_center = brentq(lambda q: np.hypot(1.0 - q, a10_camberline(q, cl0)) - r, 1.0 - 50.0 * r, 1.0 - XI_MIN)
    length = camberline_length(cl0)
    station_center = float(camberline_station(x_center, cl0))
    run = (station_center - KOVACH_START_STATION) * length
    h_start = float(_half_thickness(KOVACH_START_STATION, t_nominal))

    taper_angle = np.arccos(r / np.hypot(h_start, run)) - np.arctan2(run, h_start)
    station_tangent = station_center + r * np.sin(taper_angle) / length
    half_tangent = r * np.cos(taper_angle)
    x_tangent = brentq(lambda q: camberline_station(q, cl0) - station_tangent, x_center, 1.0 - XI_MIN)

    def half(station: NDArray[np.float64]) -> NDArray[np.float64]:
        return _kovach_half_thickness(station, station_tangent, half_tangent, t_nominal)

    return half, x_tangent, np.array([x_center, float(a10_camberline(x_center, cl0))])


def _nearest_station(x: float, cl0: float, half: HalfThickness, side: float, center: NDArray[np.float64]) -> float:
    """Camber station near x where a surface comes closest to center, which is where it runs
    tangent to the circles about center. The mean line turns fast near the TE, so this
    differs between the two surfaces."""
    def distance(q: float) -> float:
        return float(np.hypot(*(_surface(np.array([q]), cl0, half, side)[0] - center)))

    bounds = (x - TE_NEAREST_RANGE, min(x + TE_NEAREST_RANGE, 1.0 - XI_MIN))
    return float(minimize_scalar(distance, bounds=bounds, method="bounded", options={"xatol": 1e-9}).x)


def _te_cap(start: NDArray[np.float64], center: NDArray[np.float64], side: float) -> Points:
    """From a surface point tangent to the circles about center, onto the TE circle and
    along it to (1, 0), without the surface point.

    The fair closes the small radial gap as r = R + gap u^3 (4 - 3 u), so it sits on the
    circle to second order and keeps the surface tangent. theta is measured from the axis
    through (1, 0), positive going forward on either surface.
    """
    radius = KOVACH_TE_RADIUS
    axis = (TE_POINT - center) / np.hypot(*(TE_POINT - center))
    r, theta = _polar_coordinates(start[None], center, axis, -side)

    arc_end = theta[0] - TE_FAIR_SPAN
    u = np.linspace(1.0, 0.0, N_TE_FAIR + 1)[1:]
    rho = radius + (r[0] - radius) * u**3 * (4.0 - 3.0 * u)
    fair = center + rho[:, None] * _polar(-side * (arc_end + TE_FAIR_SPAN * u), axis)
    arc = center + radius * _polar(-side * np.linspace(arc_end, 0.0, N_TE_ARC + 1)[1:], axis)
    arc[-1] = TE_POINT
    return np.vstack((fair, arc))


# ---------------------------------------------------------------------------------------
# Surface continuity: curve quantities
# ---------------------------------------------------------------------------------------

def signed_curvature(d1: ArrayLike, d2: ArrayLike) -> NDArray[np.float64]:
    """Signed curvature of a parametric curve X(u) = (m(u), y(u)) from X' and X'' in any
    parameter u:

        kappa = (m' y'' - y' m'') / (m'^2 + y'^2)^(3/2)

    Positive where the curve turns counterclockwise (to its left) as u grows. A surface runs
    LE -> TE, so the convex suction surface has kappa < 0 and the convex nose of the pressure
    surface kappa > 0. Arrays (..., 2); a length unit c gives kappa in 1/c.
    """
    d1, d2 = np.asarray(d1, dtype=float), np.asarray(d2, dtype=float)
    return (d1[..., 0] * d2[..., 1] - d1[..., 1] * d2[..., 0]) / np.hypot(d1[..., 0], d1[..., 1]) ** 3


def wall_slopes(d1: ArrayLike, d2: ArrayLike,
                min_cos: float = SLOPE_MIN_COS) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """(tan(beta), d tan(beta) / dm) of a parametric curve X(u) = (m(u), y(u)), from X' and
    X'' in any parameter u:

        dy/dm = y_u / m_u,        d2y/dm2 = (y_uu m_u - y_u m_uu) / m_u^3

    Both are NaN where |m_u| < min_cos |X_u|, that is |cos(beta)| < min_cos: the curve runs
    across m there (a nose tip, a Kovach tail) and y(m) is singular, although the curve, its
    tangent and its curvature are smooth. The default sqrt(machine epsilon) is the threshold
    at which metrics.py calls cos(beta) singular. d2y/dm2 = kappa / cos(beta)^3 is huge but
    finite just outside it.
    """
    if not 0.0 < min_cos < 1.0:
        raise ValueError(f"min_cos must lie in (0, 1), got {min_cos}")
    d1, d2 = np.asarray(d1, dtype=float), np.asarray(d2, dtype=float)
    m1, y1, m2, y2 = d1[..., 0], d1[..., 1], d2[..., 0], d2[..., 1]
    valid = np.abs(m1) >= min_cos * np.hypot(m1, y1)
    safe = np.where(valid, m1, 1.0)
    return np.where(valid, y1 / safe, np.nan), np.where(valid, (y2 * safe - y1 * m2) / safe**3, np.nan)


def _curve_parameter(u: ArrayLike) -> NDArray[np.float64]:
    u = np.asarray(u, dtype=float)
    if not np.all(np.isfinite(u)) or np.any(u < -CURVE_PARAMETER_SLACK) or np.any(u > 1.0 + CURVE_PARAMETER_SLACK):
        raise ValueError("the curve parameter u must lie in [0, 1]")
    return np.clip(u, 0.0, 1.0)


# ---------------------------------------------------------------------------------------
# Surface continuity: the C1 construction as an analytic curve
# ---------------------------------------------------------------------------------------

PieceEvaluator = Callable[[NDArray[np.float64]], tuple[Points, Points, Points]]
_GAUSS_NODES, _GAUSS_WEIGHTS = np.polynomial.legendre.leggauss(8)


@dataclass(frozen=True)
class SurfaceJunction:
    """Where two analytic pieces of a reference surface meet. They share the point; the
    tangent turns by turn (radians, counterclockwise positive; nonzero at a corner) and the
    curvature goes from kappa_before to kappa_after (1/chord)."""

    index: int                  # pieces index and index + 1 meet here
    label: str
    s: float                    # arc length from the LE, chords
    kappa_before: float
    kappa_after: float
    turn: float

    @property
    def kappa_jump(self) -> float:
        return self.kappa_after - self.kappa_before


@dataclass(frozen=True)
class SurfaceSample:
    """Points of a reference surface with their exact arc-length derivatives: tangent = X_s,
    a unit vector, and second = X_ss = kappa N. Every piece is sampled with both its ends, so
    at a junction two samples share s and carry the one-sided values of either piece."""

    s: NDArray[np.float64]
    points: Points
    tangent: Points
    second: Points
    kappa: NDArray[np.float64]
    piece: NDArray[np.int64]

    def __post_init__(self) -> None:
        for item in fields(self):
            value = np.array(getattr(self, item.name), dtype=np.int64 if item.name == "piece" else float)
            value.flags.writeable = False
            object.__setattr__(self, item.name, value)

    @property
    def weights(self) -> NDArray[np.float64]:
        """Trapezoid weights in s; a junction's two samples each close their own piece."""
        step = np.diff(self.s)
        return np.concatenate((0.5 * step, [0.0])) + np.concatenate(([0.0], 0.5 * step))


def _concatenate_samples(samples: list[SurfaceSample]) -> SurfaceSample:
    return SurfaceSample(*(np.concatenate([getattr(sample, item.name) for sample in samples])
                           for item in fields(SurfaceSample)))


@dataclass(frozen=True, eq=False)
class ReferenceSurface:
    """One surface of the validated C1 blade, LE -> TE in the blade frame at chord 1, as the
    chain of analytic pieces its construction is made of, each with exact first and second
    derivatives (naca65_reference_surfaces builds them). Every vertex of the C1 polyline lies
    on it to C2_ANCHOR_TOL: it is the surface those vertices sample, not a fit to them, so its
    curvature shows every jump of the construction as it is. leading_edge_point and
    trailing_edge_point are the C1 contour's own end vertices, the exact landmarks.

    A piece maps tau in [0, 1] (LE -> TE) to (points, dX/dtau, d2X/dtau2); arc length along
    each piece comes from 8-point Gauss-Legendre on REFERENCE_TABLE_POINTS - 1 intervals."""

    name: str
    pieces: tuple[PieceEvaluator, ...]
    piece_names: tuple[str, ...]
    junction_labels: tuple[str, ...]
    sharp_trailing_edge: bool
    leading_edge_point: NDArray[np.float64]
    trailing_edge_point: NDArray[np.float64]

    def __post_init__(self) -> None:
        _freeze_arrays(self)
        if not (len(self.pieces) == len(self.piece_names) == len(self.junction_labels) + 1):
            raise ValueError("a ReferenceSurface needs one name per piece and one label per junction")
        tau = np.linspace(0.0, 1.0, REFERENCE_TABLE_POINTS)
        s_tables, point_tables, start = [], [], 0.0
        for evaluate in self.pieces:
            s = start + np.concatenate(([0.0], np.cumsum(_interval_lengths(evaluate, tau))))
            points = evaluate(tau)[0]
            for array in (s, points):
                array.flags.writeable = False
            s_tables.append(s)
            point_tables.append(points)
            start = float(s[-1])
        tau.flags.writeable = False
        object.__setattr__(self, "_tau", tau)
        object.__setattr__(self, "_s_tables", tuple(s_tables))
        object.__setattr__(self, "_point_tables", tuple(point_tables))

    @property
    def length(self) -> float:
        """Arc length LE -> TE, chords."""
        return float(self._s_tables[-1][-1])

    @property
    def piece_bounds(self) -> NDArray[np.float64]:
        """(n_pieces, 2): arc length at the start and end of every piece."""
        return np.array([(table[0], table[-1]) for table in self._s_tables])

    @property
    def junctions(self) -> tuple[SurfaceJunction, ...]:
        ends = [self.sample_piece(k, self._s_tables[k][[0, -1]]) for k in range(len(self.pieces))]
        out = []
        for k, (before, after) in enumerate(zip(ends[:-1], ends[1:])):
            a, b = before.tangent[-1], after.tangent[0]
            out.append(SurfaceJunction(k, self.junction_labels[k], float(before.s[-1]), float(before.kappa[-1]),
                                       float(after.kappa[0]), math.atan2(a[0] * b[1] - a[1] * b[0], float(a @ b))))
        return tuple(out)

    def sample_piece(self, index: int, s: ArrayLike) -> SurfaceSample:
        """Samples of piece index near the arc lengths s, which run up its range from its
        start to its end. The samples are where tau interpolated from s puts them; their own
        s is integrated exactly from the piece start."""
        table, evaluate = self._s_tables[index], self.pieces[index]
        tau = np.interp(np.asarray(s, dtype=float), table, self._tau)
        tau[0], tau[-1] = 0.0, 1.0
        arc = table[0] + np.concatenate(([0.0], np.cumsum(_interval_lengths(evaluate, tau))))
        points, d1, d2 = evaluate(tau)
        speed = np.hypot(d1[:, 0], d1[:, 1])
        tangent = d1 / speed[:, None]
        second = (d2 - np.sum(d2 * tangent, axis=1)[:, None] * tangent) / (speed**2)[:, None]
        return SurfaceSample(arc, points, tangent, second, signed_curvature(d1, d2), np.full(len(tau), index))

    def sample(self, points_per_piece: int = 201) -> SurfaceSample:
        """Every piece sampled uniformly in arc length, both ends included."""
        return _concatenate_samples([self.sample_piece(k, np.linspace(table[0], table[-1], points_per_piece))
                                     for k, table in enumerate(self._s_tables)])

    def locate(self, points: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """(s, distance) of the nearest point of this surface to each of points: Newton steps
        in tau from the nearest table point of each of the two pieces that come closest, the
        nearer foot kept. Two, because a point just past a junction is as near to the end
        table point of the piece before it, where Newton stops at the piece end."""
        points = np.atleast_2d(np.asarray(points, dtype=float))
        table = np.stack(self._point_tables)                                # (pieces, tau, 2)
        gaps = np.hypot(points[:, None, None, 0] - table[None, ..., 0], points[:, None, None, 1] - table[None, ..., 1])
        nearest = np.argmin(gaps, axis=2)                                    # per point and piece
        candidates = np.argsort(np.min(gaps, axis=2), axis=1)[:, :min(2, len(self.pieces))]
        s, distance = np.full(len(points), np.nan), np.full(len(points), np.inf)
        for column in range(candidates.shape[1]):
            piece = candidates[:, column]
            for k in np.unique(piece):
                rows = np.flatnonzero(piece == k)
                t, target = self._tau[nearest[rows, k]], points[rows]
                for _ in range(PROJECTION_ITERATIONS):
                    p, d1, d2 = self.pieces[k](t)
                    gap = p - target
                    curvature_term = np.sum(d1 * d1, axis=1) + np.sum(gap * d2, axis=1)
                    newton = np.where(curvature_term > 0.0, curvature_term, np.sum(d1 * d1, axis=1))
                    t = np.clip(t - np.sum(gap * d1, axis=1) / newton, 0.0, 1.0)
                found = np.hypot(*(self.pieces[k](t)[0] - target).T)
                better = found < distance[rows]
                distance[rows[better]] = found[better]
                s[rows[better]] = [self._arc_length(k, value) for value in t[better]]
        return s, distance

    def _arc_length(self, index: int, tau: float) -> float:
        i = min(int(np.searchsorted(self._tau, tau, side="right")) - 1, len(self._tau) - 2)
        return float(self._s_tables[index][i] + _interval_lengths(self.pieces[index], np.array([self._tau[i], tau]))[0])


def _interval_lengths(evaluate: PieceEvaluator, tau: NDArray[np.float64]) -> NDArray[np.float64]:
    """Arc length of a piece over each interval of tau, 8-point Gauss-Legendre."""
    a, b = tau[:-1], tau[1:]
    nodes = 0.5 * (a + b)[:, None] + 0.5 * (b - a)[:, None] * _GAUSS_NODES[None]
    d1 = evaluate(nodes.ravel())[1]
    speed = np.hypot(d1[:, 0], d1[:, 1]).reshape(nodes.shape)
    return np.sum(0.5 * (b - a)[:, None] * _GAUSS_WEIGHTS[None] * speed, axis=1)


def naca65_reference_surfaces(cl0: float, t_max_over_c: float, *,
                              profile_variant: str) -> tuple[ReferenceSurface, ReferenceSurface]:
    """(suction, pressure) of the C1 NACA 65 blade of naca65_profile as ReferenceSurfaces, in
    the blade frame at chord 1: the LE circle, the LE fair, the surface X_c + side (t/2) n_c
    cut at every thickness-table station (and at the Kovach start, station 0.60), then the
    Kovach TE fair and TE circle or the emery_base closing segment to the sharp TE. Their
    parameters (fair angles, end conditions, nearest stations, TE center) are those of
    _naca65_contour, from the same functions. They do not depend on n_camber_stations,
    which only sets where the C1 polyline samples them."""
    _check_naca65_inputs(cl0, t_max_over_c, MEANLINE_TYPES[0], profile_variant, MIN_CAMBER_STATIONS)
    return _naca65_reference_surfaces(cl0, t_max_over_c, profile_variant)


@lru_cache(maxsize=32)
def _naca65_reference_surfaces(cl0: float, t_max_over_c: float,
                               profile_variant: str) -> tuple[ReferenceSurface, ReferenceSurface]:
    t_nominal = TABLE_T_MAX * thickness_scale(t_max_over_c)
    if profile_variant == "kovach_sandercock":
        half, x_end, te_center = _kovach_closure(cl0, t_nominal)
    else:
        def half(station: NDArray[np.float64]) -> NDArray[np.float64]:
            return _half_thickness(station, t_nominal)
        x_end, te_center = 1.0 - XI_MIN, None
    x_first = brentq(lambda q: camberline_station(q, cl0) - FIRST_TABLE_STATION, XI_MIN, 2.0 * LE_TANGENT_X)
    radius = TABLE_LE_RADIUS * (t_nominal / TABLE_T_MAX) ** 2
    phi_le = float(a10_slope_angle(LE_TANGENT_X, cl0))
    nose_axis = -np.array([np.cos(phi_le), np.sin(phi_le)])
    center = -radius * nose_axis

    try:
        landmarks = split_at_leading_edge(_naca65_contour(cl0, t_nominal, profile_variant, MIN_CAMBER_STATIONS).loop)
        surfaces = []
        for name, side, c1 in (("suction", 1.0, landmarks[0]), ("pressure", -1.0, landmarks[1])):
            pieces, names, labels = _naca65_surface_pieces(cl0, t_nominal, half, side, x_first, x_end, te_center,
                                                           center, radius, nose_axis)
            surface = ReferenceSurface(name, pieces, names, labels, te_center is None, c1[0], c1[-1])
            _anchored_arc_length(surface, c1)
            surfaces.append(surface)
    except GeometryError:
        raise
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
        raise GeometryError(f"NACA 65 reference surfaces failed for cl0={cl0}, t_max_over_c={t_max_over_c}: "
                            f"{error}") from error
    return surfaces[0], surfaces[1]


def _anchored_arc_length(surface: ReferenceSurface, polyline: Points) -> NDArray[np.float64]:
    """Arc length along the analytic surface of every vertex of the C1 polyline, each of
    which must lie on it."""
    s, distance = surface.locate(polyline)
    if np.max(distance) > C2_ANCHOR_TOL:
        raise GeometryError(f"the C1 {surface.name} surface leaves its analytic reference by "
                            f"{np.max(distance):.3g} c at vertex {int(np.argmax(distance))}")
    return s


def _naca65_surface_pieces(cl0: float, t_nominal: float, half: HalfThickness, side: float, x_first: float,
                           x_end: float, te_center: NDArray[np.float64] | None, center: NDArray[np.float64],
                           radius: float, nose_axis: NDArray[np.float64]
                           ) -> tuple[tuple[PieceEvaluator, ...], tuple[str, ...], tuple[str, ...]]:
    """The analytic pieces of one surface (side +1 suction, -1 pressure), LE -> TE, their
    names and the labels of the junctions between them."""
    pieces: list[PieceEvaluator] = []
    names: list[str] = []
    labels: list[str] = []

    # LE circle and fair, with the start angle and the end conditions of _nose_fair (its
    # probe of the surface at x_first, repeated here).
    start = _nose_fair(x_first, cl0, half, side, center, radius, nose_axis)[1]
    probe = _surface(x_first + 2e-4 * np.arange(6), cl0, half, side)
    r, theta = _polar_coordinates(probe, center, nose_axis, side)
    fit = np.polyfit(theta - theta[0], r, 3)
    end = (r[0] - radius, fit[2], 2.0 * fit[1])
    span = theta[0] - start
    pieces.append(_polar_piece(center, nose_axis, side, 0.0, start, _constant_radius(radius)))
    names.append("LE circle")
    labels.append("LE circle / LE fair")

    def fair_radius(angle: NDArray[np.float64]) -> tuple[NDArray[np.float64], ...]:
        return _fair_radius((angle - start) / span, radius, span, *end)
    pieces.append(_polar_piece(center, nose_axis, side, start, float(theta[0]), fair_radius))
    names.append("LE fair")
    labels.append(f"LE fair / surface (station {FIRST_TABLE_STATION:g})")

    # The surface, cut at every thickness-table station between its two ends.
    x_last = x_end if te_center is None else _nearest_station(x_end, cl0, half, side, te_center)
    kovach = te_center is not None
    station_last = float(camberline_station(x_last, cl0))
    cuts = [(x_first, FIRST_TABLE_STATION)]
    for station in np.array(_TABLE_X_PCT[1:-1]) / 100.0:
        if FIRST_TABLE_STATION < station < station_last:
            cuts.append((brentq(lambda q, s=station: camberline_station(q, cl0) - s, XI_MIN, 1.0 - XI_MIN,
                                xtol=1e-15, rtol=4.0 * np.finfo(float).eps), float(station)))
    cuts.append((x_last, station_last))
    taper = None
    if kovach:      # the Kovach taper is linear in the station: its slope from two stations on it
        far = np.array([0.5 * (KOVACH_START_STATION + 1.0), 1.0])
        values = half(far)
        taper = float((values[1] - values[0]) / (far[1] - far[0]))
    for k, ((xa, sa), (xb, sb)) in enumerate(zip(cuts[:-1], cuts[1:])):
        on_taper = kovach and sa >= KOVACH_START_STATION
        slopes = _taper_slopes(taper) if on_taper else _table_slopes(0.5 * (sa + sb), t_nominal / TABLE_T_MAX)
        pieces.append(_surface_piece(cl0, half, side, xa, xb, slopes))
        names.append(f"surface, stations {sa:.4g}-{sb:.4g}" + (" (Kovach taper)" if on_taper else ""))
        if k < len(cuts) - 2:
            labels.append(f"Kovach start, station {sb:g}" if kovach and sb == KOVACH_START_STATION
                          else f"table station {sb:g}")

    if te_center is None:
        last = _surface(np.array([x_last]), cl0, half, side)[0]
        labels.append(f"closing segment (x/c = {x_last:g})")
        pieces.append(_segment_piece(last, TE_POINT))
        names.append("closing segment to the sharp TE")
        return tuple(pieces), tuple(names), tuple(labels)

    # Kovach TE fair and circle, as _te_cap builds them.
    te_radius = KOVACH_TE_RADIUS
    axis = (TE_POINT - te_center) / np.hypot(*(TE_POINT - te_center))
    r0, theta0 = (float(value[0]) for value in _polar_coordinates(_surface(np.array([x_last]), cl0, half, side),
                                                                  te_center, axis, -side))
    arc_end = theta0 - TE_FAIR_SPAN

    def te_fair_radius(angle: NDArray[np.float64]) -> tuple[NDArray[np.float64], ...]:
        u, gap = (angle - arc_end) / TE_FAIR_SPAN, r0 - te_radius
        return (te_radius + gap * u**3 * (4.0 - 3.0 * u), gap * (12.0 * u**2 - 12.0 * u**3) / TE_FAIR_SPAN,
                gap * (24.0 * u - 36.0 * u**2) / TE_FAIR_SPAN**2)
    labels.append("Kovach TE fair start")
    pieces.append(_polar_piece(te_center, axis, -side, theta0, arc_end, te_fair_radius))
    names.append("Kovach TE fair")
    labels.append("Kovach TE circle start")
    pieces.append(_polar_piece(te_center, axis, -side, arc_end, 0.0, _constant_radius(te_radius)))
    names.append("Kovach TE circle")
    return tuple(pieces), tuple(names), tuple(labels)


def _constant_radius(radius: float) -> Callable[[NDArray[np.float64]], tuple[NDArray[np.float64], ...]]:
    def radius_of(angle: NDArray[np.float64]) -> tuple[NDArray[np.float64], ...]:
        return np.full_like(angle, radius), np.zeros_like(angle), np.zeros_like(angle)
    return radius_of


def _polar_piece(center: NDArray[np.float64], axis: NDArray[np.float64], sign: float, theta_a: float, theta_b: float,
                 radius_of: Callable[[NDArray[np.float64]], tuple[NDArray[np.float64], ...]]) -> PieceEvaluator:
    """center + r(theta) e(theta), e = cos(theta) axis + sign sin(theta) normal, as _polar
    writes it, from theta_a to theta_b; radius_of gives r, dr/dtheta and d2r/dtheta2."""
    normal = np.array([axis[1], -axis[0]])
    span = theta_b - theta_a

    def evaluate(tau: NDArray[np.float64]) -> tuple[Points, Points, Points]:
        angle = theta_a + span * tau
        r, r1, r2 = radius_of(angle)
        e = np.outer(np.cos(angle), axis) + sign * np.outer(np.sin(angle), normal)
        e1 = -np.outer(np.sin(angle), axis) + sign * np.outer(np.cos(angle), normal)
        return (center + r[:, None] * e, span * (r1[:, None] * e + r[:, None] * e1),
                span**2 * ((r2 - r)[:, None] * e + 2.0 * r1[:, None] * e1))
    return evaluate


def _segment_piece(a: NDArray[np.float64], b: NDArray[np.float64]) -> PieceEvaluator:
    def evaluate(tau: NDArray[np.float64]) -> tuple[Points, Points, Points]:
        return a + np.outer(tau, b - a), np.tile(b - a, (len(tau), 1)), np.zeros((len(tau), 2))
    return evaluate


def _table_slopes(station: float, scale: float) -> Callable[[NDArray[np.float64]], tuple[NDArray[np.float64], ...]]:
    """dh/dstation and d2h/dstation2 on the table interval holding station: the PCHIP is
    cubic in z = sqrt(station) on each interval, h = scale B(z), so
    h' = scale B' / (2 z) and h'' = scale (B'' - B' / z) / (4 z^2), on that interval's
    polynomial even at its ends (one-sided values at a table knot)."""
    pchip = _BASE_HALF_THICKNESS
    interval = int(np.searchsorted(pchip.x, math.sqrt(station))) - 1
    c3, c2, c1, _ = pchip.c[:, interval]
    z0 = pchip.x[interval]

    def slopes(sigma: NDArray[np.float64]) -> tuple[NDArray[np.float64], ...]:
        z = np.sqrt(sigma)
        d = z - z0
        b1 = (3.0 * c3 * d + 2.0 * c2) * d + c1
        b2 = 6.0 * c3 * d + 2.0 * c2
        return scale * b1 / (2.0 * z), scale * (b2 - b1 / z) / (4.0 * sigma)
    return slopes


def _taper_slopes(slope: float) -> Callable[[NDArray[np.float64]], tuple[NDArray[np.float64], ...]]:
    def slopes(sigma: NDArray[np.float64]) -> tuple[NDArray[np.float64], ...]:
        return np.full_like(sigma, slope), np.zeros_like(sigma)
    return slopes


def _surface_piece(cl0: float, half: HalfThickness, side: float, xa: float, xb: float,
                   slopes: Callable[[NDArray[np.float64]], tuple[NDArray[np.float64], ...]]) -> PieceEvaluator:
    """X = X_c + side h n_c from camber station xa to xb, the points exactly as _surface
    builds them. With phi the mean-line angle, the station sigma(x) from the arc-length
    spline and H(x) = h(sigma(x)):

        X'  = (1, y_c')  + side (H' n - H phi' t)
        X'' = (0, y_c'') + side (H'' n - 2 H' phi' t - H (phi'' t + phi'^2 n))

    t = (cos phi, sin phi) and n = (-sin phi, cos phi)."""
    spline, length, k = _arc_length_spline(cl0), camberline_length(cl0), cl0 / (4.0 * math.pi)
    dx = xb - xa

    def evaluate(tau: NDArray[np.float64]) -> tuple[Points, Points, Points]:
        x = xa + dx * tau
        points = _surface(x, cl0, half, side)
        h = half(camberline_station(x, cl0))
        h1, h2 = slopes(camberline_station(x, cl0))
        sx, sxx = spline(x, 1) / length, spline(x, 2) / length
        dh, ddh = h1 * sx, h2 * sx**2 + h1 * sxx
        y1 = _a10_slope(x, cl0)
        y2 = -k / (x * (1.0 - x))
        y3 = k * (1.0 - 2.0 * x) / (x * (1.0 - x)) ** 2
        q = 1.0 + y1**2
        p1, p2 = y2 / q, (y3 * q - 2.0 * y1 * y2**2) / q**2
        phi = np.arctan(y1)
        t = np.column_stack((np.cos(phi), np.sin(phi)))
        n = np.column_stack((-np.sin(phi), np.cos(phi)))
        d1 = np.column_stack((np.ones_like(x), y1)) + side * (dh[:, None] * n - (h * p1)[:, None] * t)
        d2 = (np.column_stack((np.zeros_like(x), y2))
              + side * (ddh[:, None] * n - (2.0 * dh * p1 + h * p2)[:, None] * t - (h * p1**2)[:, None] * n))
        return points, dx * d1, dx**2 * d2
    return evaluate


# ---------------------------------------------------------------------------------------
# Surface continuity: the C2 quintic representation
# ---------------------------------------------------------------------------------------

_KERNEL_PSI0 = 35.0 / 256.0     # Psi(0) of the kernel below: a rounded corner's cut is Psi0 |turn| a


def _kernel_step(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Phi(x), the integral of the kernel K(x) = (35/32) (1 - x^2)^3 on [-1, 1]: 0 below -1,
    1 above 1, C3. K is positive, so a step smoothed by it rises monotonically."""
    x = np.clip(x, -1.0, 1.0)
    return 0.5 + 35.0 / 32.0 * x * (1.0 + x**2 * (-1.0 + x**2 * (0.6 - x**2 / 7.0)))


def _kernel_ramp(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Psi(x), the integral of Phi: 0 below -1, x above 1."""
    x = np.clip(x, -1.0, 1.0)
    return 0.5 * x + 35.0 / 32.0 * x**2 * (0.5 + x**2 * (-0.25 + x**2 * (0.1 - x**2 / 56.0))) + _KERNEL_PSI0


@dataclass(frozen=True)
class JunctionTransition:
    """How the C2 fit treats one junction of a reference surface. Its curvature jump and its
    tangent break are smoothed by the kernel K over s - half_width .. s + half_width, which
    leaves the shift vector behind it; the fit takes the shift back over
    C2_COMPENSATION_RATIO * half_width. half_width 0: the junction is G2 and stays as it is."""

    junction: SurfaceJunction
    half_width: float
    shift: NDArray[np.float64]

    def __post_init__(self) -> None:
        _freeze_arrays(self)


@dataclass(frozen=True, eq=False)
class QuinticSurfaceFit:
    """The C2 curve of one reference surface and what it cost. Deviations are normal
    distances from the reference samples to the curve, in chords; the representation error
    is the curve against the smoothed target it was fitted to."""

    curve: ParametricBladeCurve
    reference: ReferenceSurface
    transitions: tuple[JunctionTransition, ...]
    max_deviation: float
    rms_deviation: float            # arc-length weighted
    max_representation_error: float
    n_samples: int


def fit_quintic_surface(reference: ReferenceSurface) -> QuinticSurfaceFit:
    """The C2 representation of one C1 reference surface: a clamped quintic B-spline in
    u = s / L, s the arc length of the reference [this module's construction].

    1. At every junction where the curvature jumps or the tangent breaks, the tangent angle
       theta(s) is split into its smooth part and the kink, kappa_jump (s - s_j)_+ +
       turn H(s - s_j), and the kink is replaced by its convolution with the positive kernel K
       of half-width a_j. The curvature then rises monotonically across the junction, with
       no overshoot. a_j is the largest width that leaves a shift kappa_jump a^2 / 18 of at
       most C2_DEVIATION_TARGET and cuts a corner by at most that (Psi0 |turn| a), capped at
       C2_MAX_HALF_WIDTH and clear of the surface ends.
    2. The smoothed angle is integrated from the LE; each transition's shift is taken back
       smoothly over C2_COMPENSATION_RATIO a_j, so the surface rejoins the reference after
       it and the TE stays where it is. Smooth pieces, among them the LE fair and its
       S-bend, are not changed.
    3. A quintic B-spline is fitted to that target by least squares (arc-length weights),
       with knots no farther apart than a_j / C2_KNOTS_PER_HALF_WIDTH in a transition and a
       C2_KNOTS_PER_RADIUS-th of the radius of curvature, graded, and with the LE point, its
       unit tangent and curvature (u-derivatives L T and L^2 kappa N) imposed exactly, and
       at the TE the point and tangent, plus the curvature at a rounded TE. A sharp TE keeps
       its corner: its tangent is the C1 closing segment's.
    """
    length = reference.length
    transitions = _junction_transitions(reference)
    s_grid, spacing = _c2_knot_spacing(reference, transitions)
    sample = _c2_samples(reference, s_grid, spacing)
    target, transitions = _c2_target(sample, transitions)
    knots = _c2_knots(s_grid, spacing, length)
    u = np.clip(sample.s / length, 0.0, 1.0)
    u[-1] = 1.0
    curve = ParametricBladeCurve(knots, _c2_control_points(reference, sample, target, knots, u), length, C2_DEGREE)
    normal = curve.project(sample.points, u)[1]
    weights = sample.weights
    return QuinticSurfaceFit(curve, reference, transitions, float(np.max(np.abs(normal))),
                             math.sqrt(float(np.sum(weights * normal**2) / np.sum(weights))),
                             float(np.max(np.hypot(*(curve.evaluate(u) - target).T))), len(u))


def _junction_transitions(reference: ReferenceSurface) -> tuple[JunctionTransition, ...]:
    length, out = reference.length, []
    for junction in reference.junctions:
        jump, turn = abs(junction.kappa_jump), abs(junction.turn)
        half_width = 0.0
        if jump > C2_KAPPA_JUMP_TOL or turn > C2_TURN_TOL:
            half_width = C2_MAX_HALF_WIDTH
            if jump > C2_KAPPA_JUMP_TOL:
                half_width = min(half_width, math.sqrt(18.0 * C2_DEVIATION_TARGET / jump))
            if turn > C2_TURN_TOL:
                half_width = min(half_width, C2_DEVIATION_TARGET / (_KERNEL_PSI0 * turn))
            half_width = min(half_width, 0.45 * min(junction.s, length - junction.s) / C2_COMPENSATION_RATIO)
        out.append(JunctionTransition(junction, half_width, np.zeros(2)))
    return tuple(out)


def _c2_knot_spacing(reference: ReferenceSurface,
                     transitions: tuple[JunctionTransition, ...]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The knot spacing h(s) on C2_SIZE_GRID points: at most C2_MAX_KNOT_SPACING, a
    C2_KNOTS_PER_RADIUS-th of the radius of curvature and, near each transition, its
    half-width over C2_KNOTS_PER_HALF_WIDTH, growing by at most C2_KNOT_GRADING per unit length."""
    s = np.linspace(0.0, reference.length, C2_SIZE_GRID)
    coarse = reference.sample(C2_COARSE_POINTS)
    curvature = np.interp(s, coarse.s, np.abs(coarse.kappa))
    spacing = np.minimum(C2_MAX_KNOT_SPACING, 1.0 / (C2_KNOTS_PER_RADIUS * np.maximum(curvature, 1e-300)))
    for transition in transitions:
        if transition.half_width > 0.0:
            gap = np.maximum(np.abs(s - transition.junction.s) - transition.half_width, 0.0)
            spacing = np.minimum(spacing, transition.half_width / C2_KNOTS_PER_HALF_WIDTH + C2_KNOT_GRADING * gap)
    graded = C2_KNOT_GRADING * s
    spacing = np.minimum(spacing, graded + np.minimum.accumulate(spacing - graded))
    spacing = np.minimum(spacing, (np.minimum.accumulate((spacing + graded)[::-1]))[::-1] - graded)
    return s, spacing


def _c2_samples(reference: ReferenceSurface, s_grid: NDArray[np.float64], spacing: NDArray[np.float64]) -> SurfaceSample:
    """The reference sampled piece by piece, C2_SAMPLES_PER_SPAN samples per knot spacing."""
    density = C2_SAMPLES_PER_SPAN / spacing
    count = np.concatenate(([0.0], np.cumsum(0.5 * (density[1:] + density[:-1]) * np.diff(s_grid))))
    samples = []
    for k, (start, end) in enumerate(reference.piece_bounds):
        n_start, n_end = np.interp([start, end], s_grid, count)
        targets = np.interp(np.linspace(n_start, n_end, max(5, int(math.ceil(n_end - n_start)) + 1)), count, s_grid)
        samples.append(reference.sample_piece(k, targets))
    return _concatenate_samples(samples)


def _c2_target(sample: SurfaceSample,
               transitions: tuple[JunctionTransition, ...]) -> tuple[Points, tuple[JunctionTransition, ...]]:
    """The reference with every junction kink smoothed (step 1 of fit_quintic_surface) and
    every shift taken back (step 2), at the sample points; and the transitions with their shifts."""
    theta = np.unwrap(np.arctan2(sample.tangent[:, 1], sample.tangent[:, 0]))
    direction = np.column_stack((np.cos(theta), np.sin(theta)))
    weights = sample.weights[:, None]
    smoothed = theta.copy()
    compensation = np.zeros_like(sample.points)
    out = []
    for transition in transitions:
        a = transition.half_width
        if a == 0.0:
            out.append(transition)
            continue
        junction = transition.junction
        x = (sample.s - junction.s) / a
        inside = np.abs(x) < 1.0
        after = (sample.piece[inside] > junction.index).astype(float)
        change = (junction.kappa_jump * a * (_kernel_ramp(x[inside]) - np.maximum(x[inside], 0.0))
                  + junction.turn * (_kernel_step(x[inside]) - after))
        smoothed[inside] += change
        own = np.zeros_like(sample.points)
        own[inside] = np.column_stack((np.cos(theta[inside] + change), np.sin(theta[inside] + change))) - direction[inside]
        shift = np.sum(weights * own, axis=0)
        compensation -= shift[None] * _kernel_step((sample.s - junction.s) / (C2_COMPENSATION_RATIO * a))[:, None]
        out.append(replace(transition, shift=shift))
    change = np.column_stack((np.cos(smoothed), np.sin(smoothed))) - direction
    step = 0.5 * (change[1:] + change[:-1]) * np.diff(sample.s)[:, None]
    displacement = np.vstack((np.zeros(2), np.cumsum(step, axis=0)))
    return sample.points + displacement + compensation, tuple(out)


def _c2_knots(s_grid: NDArray[np.float64], spacing: NDArray[np.float64], length: float) -> NDArray[np.float64]:
    """Clamped quintic knots on [0, 1], interior knots equidistributed in 1 / spacing."""
    count = np.concatenate(([0.0], np.cumsum(0.5 * (1.0 / spacing[1:] + 1.0 / spacing[:-1]) * np.diff(s_grid))))
    n_spans = max(2, int(round(count[-1])))
    interior = np.interp(np.linspace(0.0, count[-1], n_spans + 1)[1:-1], count, s_grid) / length
    return np.concatenate((np.zeros(C2_DEGREE + 1), interior, np.ones(C2_DEGREE + 1)))


def _end_control_points(knots: NDArray[np.float64], at_start: bool, values: list[NDArray[np.float64]]
                        ) -> list[NDArray[np.float64]]:
    """The control points next to one end of a clamped spline, counted from that end, that
    give the end the point and u-derivatives in values (point, X', X'', as many as given).
    Only those control points act there, through a triangular matrix of basis derivatives."""
    n, u = len(knots) - C2_DEGREE - 1, 0.0 if at_start else 1.0
    basis = np.zeros((len(values), len(values)))
    for i in range(len(values)):
        unit = np.zeros(n)
        unit[i if at_start else n - 1 - i] = 1.0
        spline = BSpline(knots, unit, C2_DEGREE)
        for j in range(len(values)):
            basis[j, i] = float(spline(u, j))
    points: list[NDArray[np.float64]] = []
    for j, value in enumerate(values):
        points.append((value - sum(basis[j, i] * points[i] for i in range(j))) / basis[j, j])
    points[0] = np.array(values[0], dtype=float)     # the end point itself, exactly
    return points


def _c2_control_points(reference: ReferenceSurface, sample: SurfaceSample, target: Points,
                       knots: NDArray[np.float64], u: NDArray[np.float64]) -> Points:
    """Least squares for the control points, the end ones fixed (step 3 of fit_quintic_surface)."""
    length, n = reference.length, len(knots) - C2_DEGREE - 1

    def end_values(i: int, point: NDArray[np.float64], curvature: bool) -> list[NDArray[np.float64]]:
        tangent = sample.tangent[i]
        values = [point, length * tangent]
        if curvature:
            values.append(length**2 * sample.kappa[i] * np.array([-tangent[1], tangent[0]]))
        return values

    return _fit_clamped_quintic(knots, u, target, sample.weights,
                                end_values(0, reference.leading_edge_point, True),
                                end_values(-1, reference.trailing_edge_point, not reference.sharp_trailing_edge))


def _fit_clamped_quintic(knots: NDArray[np.float64], u: NDArray[np.float64], target: Points,
                         weights: NDArray[np.float64], start_values: list[NDArray[np.float64]],
                         end_values: list[NDArray[np.float64]]) -> Points:
    """Control points of the clamped quintic on knots that has the point and u-derivatives
    start_values at u = 0 and end_values at u = 1 exactly, and is otherwise the weighted
    least-squares fit to target at u."""
    n = len(knots) - C2_DEGREE - 1
    start = _end_control_points(knots, True, start_values)
    end = _end_control_points(knots, False, end_values)
    control = np.zeros((n, 2))
    control[:len(start)] = start
    control[n - len(end):] = end[::-1]
    free = slice(len(start), n - len(end))
    design = BSpline.design_matrix(u, knots, C2_DEGREE).tocsc()
    right = target - design @ control
    free_design = design[:, free]
    weighted = free_design.multiply(weights[:, None]).tocsc()
    normal = (free_design.T @ weighted).toarray()
    control[free] = solve(normal, weighted.T @ right, assume_a="pos")
    return control


# ---------------------------------------------------------------------------------------
# Surface continuity: the C2 leading edge (family independent)
# ---------------------------------------------------------------------------------------

_SMOOTH_NODES, _SMOOTH_WEIGHTS = np.polynomial.legendre.leggauss(SMOOTH_GAUSS_POINTS)
_LE_NODES, _LE_WEIGHTS = np.polynomial.legendre.leggauss(LE_FAIR_GAUSS_POINTS)


def _rotated_left(vectors: Points) -> Points:
    return np.column_stack((-vectors[:, 1], vectors[:, 0]))


def _ldp(G: NDArray[np.float64], h: NDArray[np.float64]) -> NDArray[np.float64] | None:
    """min |z| subject to G z >= h, through NNLS [Lawson & Hanson 1974, Ch. 23]; None when no z
    satisfies the inequalities."""
    n = G.shape[1]
    system = np.vstack((G.T, h[None, :]))
    target = np.zeros(n + 1)
    target[-1] = 1.0
    weights = nnls(system, target, maxiter=50 * (system.shape[1] + n + 10))[0]
    residual = system @ weights - target
    if float(np.linalg.norm(residual)) <= LDP_TOL or not abs(residual[n]) > LDP_TOL:
        return None
    return -residual[:n] / residual[n]


def _lsei(A: NDArray[np.float64], b: NDArray[np.float64], E: NDArray[np.float64],
          f: NDArray[np.float64]) -> NDArray[np.float64] | None:
    """min |A x - b| subject to E x = f and x >= 0 [Lawson & Hanson 1974, Chs. 20-23]: the
    equalities are eliminated through a null-space basis of E, what is left is a least-distance
    problem, solved exactly by _ldp. None when the constraints admit no x."""
    particular = lstsq(E, f)[0]
    basis = null_space(E)
    u, singular, vt = np.linalg.svd(A @ basis, full_matrices=False)
    keep = singular > singular[0] * 1e-13
    u, singular, vt = u[:, keep], singular[keep], vt[keep]
    offset = u.T @ (b - A @ particular)
    back = basis @ vt.T / singular                       # x = particular + back (z + offset)
    z = _ldp(back, -particular - back @ offset)
    if z is None:
        return None
    return particular + back @ (z + offset)


@dataclass(frozen=True, eq=False)
class SmoothedSurface:
    """A reference surface from its piece first_piece on, as the C2 fit sees it: the kink at
    every junction after that piece smoothed by its JunctionTransition and every shift taken
    back, as fit_quintic_surface builds its target, but evaluated at any reference arc length
    s in [start, end] with exact derivatives. Its curvature is continuous. The displacement
    the smoothed tangent leaves is integrated per transition support, split at its junction
    and its piece boundary, by Gauss-Legendre on SMOOTH_SUBDIVISIONS sub-intervals, each kept
    as the Legendre series of its antiderivative.

    evaluate(s) -> (points, dX/ds, d2X/ds2). s is the reference arc length, so |dX/ds| differs
    from 1 by the slope of the compensations (below 1e-4)."""

    reference: ReferenceSurface
    first_piece: int

    def __post_init__(self) -> None:
        reference = self.reference
        bounds = reference.piece_bounds
        if not 0 <= self.first_piece < len(reference.pieces):
            raise ValueError("first_piece must index a piece of the reference")
        start, end = float(bounds[self.first_piece, 0]), reference.length
        transitions = tuple(t for t in _junction_transitions(reference)
                            if t.half_width > 0.0 and t.junction.index >= self.first_piece)
        boundaries = tuple(float(bounds[t.junction.index + 1, 0]) for t in transitions)
        for name, value in (("_start", start), ("_end", end), ("_transitions", transitions),
                            ("_boundaries", boundaries)):
            object.__setattr__(self, name, value)
        shifts = np.zeros((len(transitions), 2))
        for i, t in enumerate(transitions):
            lo, hi = max(t.junction.s - t.half_width, start), min(t.junction.s + t.half_width, end)
            for a, b in self._pieces_of(lo, hi, (t.junction.s, boundaries[i])):
                s = 0.5 * (a + b) + 0.5 * (b - a) * _SMOOTH_NODES
                points, tangent, kappa, piece = self._raw(s)
                change = self._change(i, s, piece)[0]
                own = tangent * (np.cos(change) - 1.0)[:, None] + _rotated_left(tangent) * np.sin(change)[:, None]
                shifts[i] += 0.5 * (b - a) * (_SMOOTH_WEIGHTS @ own)
        shifts.flags.writeable = False
        object.__setattr__(self, "_shifts", shifts)
        self._build_displacement()

    @property
    def start(self) -> float:
        return self._start

    @property
    def end(self) -> float:
        return self._end

    @property
    def transitions(self) -> tuple[JunctionTransition, ...]:
        return self._transitions

    def _pieces_of(self, lo: float, hi: float, breaks: tuple[float, ...]) -> list[tuple[float, float]]:
        """[lo, hi] cut at breaks inside it, each part cut in SMOOTH_SUBDIVISIONS."""
        cuts = sorted({lo, hi, *(x for x in breaks if lo < x < hi)})
        out = []
        for a, b in zip(cuts[:-1], cuts[1:]):
            edges = np.linspace(a, b, SMOOTH_SUBDIVISIONS + 1)
            out.extend(zip(edges[:-1], edges[1:]))
        return out

    def _build_displacement(self) -> None:
        supports = sorted((max(t.junction.s - t.half_width, self._start), min(t.junction.s + t.half_width, self._end))
                          for t in self._transitions)
        merged: list[list[float]] = []
        for lo, hi in supports:
            if hi <= lo:
                continue
            if merged and lo <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], hi)
            else:
                merged.append([lo, hi])
        breaks = tuple(t.junction.s for t in self._transitions) + self._boundaries
        parts = [part for lo, hi in merged for part in self._pieces_of(lo, hi, breaks)]
        lows = np.array([a for a, _ in parts])
        halves = np.array([0.5 * (b - a) for a, b in parts])
        coefficients = np.zeros((len(parts), SMOOTH_GAUSS_POINTS + 1, 2))
        totals = np.zeros((len(parts), 2))
        if parts:
            s = (lows + halves)[:, None] + halves[:, None] * _SMOOTH_NODES[None]
            values = self._displacement_rate(s.ravel()).reshape(len(parts), SMOOTH_GAUSS_POINTS, 2)
            legendre = np.polynomial.legendre.legvander(_SMOOTH_NODES, SMOOTH_GAUSS_POINTS - 1)    # (nodes, degree)
            scale = (2.0 * np.arange(SMOOTH_GAUSS_POINTS) + 1.0) / 2.0
            series = np.einsum("n,nd,knc->kdc", _SMOOTH_WEIGHTS, legendre, values) * scale[None, :, None]
            for k in range(len(parts)):
                for c in range(2):
                    coefficients[k, :, c] = np.polynomial.legendre.legint(series[k, :, c], lbnd=-1.0)
                    totals[k, c] = halves[k] * np.polynomial.legendre.legval(1.0, coefficients[k, :, c])
        cumulative = np.vstack((np.zeros((1, 2)), np.cumsum(totals, axis=0)))[:-1]
        for name, value in (("_lows", lows), ("_halves", halves), ("_coefficients", coefficients),
                            ("_cumulative", cumulative)):
            value.flags.writeable = False
            object.__setattr__(self, name, value)

    def _tau_at(self, index: int, s: NDArray[np.float64]) -> NDArray[np.float64]:
        """Piece parameter at reference arc length s on piece index, by Newton on its arc length."""
        reference = self.reference
        table, taus, evaluate = reference._s_tables[index], reference._tau, reference.pieces[index]
        tau = np.interp(s, table, taus)
        for _ in range(SMOOTH_NEWTON_STEPS):
            i = np.clip(np.searchsorted(taus, tau, side="right") - 1, 0, len(taus) - 2)
            a = taus[i]
            nodes = 0.5 * (a + tau)[:, None] + 0.5 * (tau - a)[:, None] * _GAUSS_NODES[None]
            speed_nodes = np.hypot(*evaluate(nodes.ravel())[1].T).reshape(nodes.shape)
            arc = table[i] + 0.5 * (tau - a) * (speed_nodes @ _GAUSS_WEIGHTS)
            speed = np.hypot(*evaluate(tau)[1].T)
            tau = np.clip(tau - (arc - s) / speed, 0.0, 1.0)
        return tau

    def _raw(self, s: NDArray[np.float64]) -> tuple[Points, Points, NDArray[np.float64], NDArray[np.int64]]:
        """Reference points, unit tangents, curvatures and piece indices at arc lengths s (the
        piece after a boundary at a boundary)."""
        reference = self.reference
        piece = np.clip(np.searchsorted(reference.piece_bounds[:, 0], s, side="right") - 1,
                        self.first_piece, len(reference.pieces) - 1)
        points, d1, d2 = (np.zeros((len(s), 2)) for _ in range(3))
        for k in np.unique(piece):
            rows = piece == k
            points[rows], d1[rows], d2[rows] = reference.pieces[int(k)](self._tau_at(int(k), s[rows]))
        return points, d1 / np.hypot(*d1.T)[:, None], signed_curvature(d1, d2), piece

    def _change(self, i: int, s: NDArray[np.float64],
                piece: NDArray[np.int64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """What transition i adds to the tangent angle and to the curvature at s."""
        t = self._transitions[i]
        a, junction = t.half_width, t.junction
        x = (s - junction.s) / a
        inside = np.abs(x) < 1.0
        after = (piece > junction.index).astype(float)
        angle, curvature = np.zeros_like(s), np.zeros_like(s)
        xi = x[inside]
        angle[inside] = (junction.kappa_jump * a * (_kernel_ramp(xi) - np.maximum(xi, 0.0))
                         + junction.turn * (_kernel_step(xi) - after[inside]))
        curvature[inside] = (junction.kappa_jump * (_kernel_step(xi) - after[inside])
                             + junction.turn * 35.0 / 32.0 * (1.0 - xi**2) ** 3 / a)
        return angle, curvature

    def _changes(self, s: NDArray[np.float64],
                 piece: NDArray[np.int64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        angle, curvature = np.zeros_like(s), np.zeros_like(s)
        for i in range(len(self._transitions)):
            a, c = self._change(i, s, piece)
            angle += a
            curvature += c
        return angle, curvature

    def _displacement_rate(self, s: NDArray[np.float64]) -> Points:
        points, tangent, kappa, piece = self._raw(s)
        angle = self._changes(s, piece)[0]
        return tangent * (np.cos(angle) - 1.0)[:, None] + _rotated_left(tangent) * np.sin(angle)[:, None]

    def _displacement(self, s: NDArray[np.float64]) -> Points:
        out = np.zeros((len(s), 2))
        if len(self._lows) == 0:
            return out
        k = np.searchsorted(self._lows, s, side="right") - 1
        rows = k >= 0
        k = k[rows]
        x = np.clip((s[rows] - self._lows[k] - self._halves[k]) / self._halves[k], -1.0, 1.0)
        for c in range(2):
            partial = np.polynomial.legendre.legval(x, self._coefficients[k, :, c].T, tensor=False)
            out[rows, c] = self._cumulative[k, c] + self._halves[k] * partial
        return out

    def evaluate(self, s: ArrayLike) -> tuple[Points, Points, Points]:
        s = np.atleast_1d(np.asarray(s, dtype=float))
        if np.any(s < self._start - CURVE_PARAMETER_SLACK) or np.any(s > self._end + CURVE_PARAMETER_SLACK):
            raise ValueError(f"SmoothedSurface parameter outside [{self._start}, {self._end}]")
        s = np.clip(s, self._start, self._end)
        points, tangent, kappa, piece = self._raw(s)
        angle, curvature = self._changes(s, piece)
        direction = tangent * np.cos(angle)[:, None] + _rotated_left(tangent) * np.sin(angle)[:, None]
        shift0, shift1, shift2 = (np.zeros((len(s), 2)) for _ in range(3))
        for t, shift in zip(self._transitions, self._shifts):
            width = C2_COMPENSATION_RATIO * t.half_width
            x = (s - t.junction.s) / width
            inside = np.abs(x) < 1.0
            xi = np.clip(x, -1.0, 1.0)
            shift0 -= shift[None] * _kernel_step(x)[:, None]
            shift1 -= shift[None] * (inside * 35.0 / 32.0 * (1.0 - xi**2) ** 3 / width)[:, None]
            shift2 -= shift[None] * (inside * 35.0 / 32.0 * -6.0 * xi * (1.0 - xi**2) ** 2 / width**2)[:, None]
        return (points + self._displacement(s) + shift0, direction + shift1,
                (kappa + curvature)[:, None] * _rotated_left(direction) + shift2)


@dataclass(frozen=True, eq=False)
class LeadingEdgeNose:
    """The nose circle of a blade section, the fixed input of the LE fair: the LE apex, the
    circle's center and radius R_LE, and where R_LE comes from (radius_source). The apex lies
    on the circle. Local LE frame: axis is the unit vector from the apex to the center, along
    the nose axis into the blade; normal is axis turned +90 deg, toward the suction side. The
    suction surface leaves the apex along +normal and the pressure surface along -normal, both
    turning toward axis, with curvature -1/R_LE and +1/R_LE there (counterclockwise positive).
    A family supplies this; nothing in the fair depends on how the section was built."""

    apex: NDArray[np.float64]
    center: NDArray[np.float64]
    radius: float
    radius_source: str

    def __post_init__(self) -> None:
        _freeze_arrays(self)
        distance = float(np.hypot(*(self.center - self.apex)))
        if not (self.radius > 0.0 and abs(distance - self.radius) <= LE_RADIUS_TOL * self.radius):
            raise ValueError(f"a LeadingEdgeNose needs R_LE > 0 and its center R_LE from the apex "
                             f"(R_LE {self.radius}, distance {distance})")

    @property
    def axis(self) -> NDArray[np.float64]:
        return (self.center - self.apex) / self.radius

    @property
    def normal(self) -> NDArray[np.float64]:
        axis = self.axis
        return np.array([-axis[1], axis[0]])


@dataclass(frozen=True, eq=False)
class LeadingEdgeFair:
    """One surface's leading edge, apex -> anchor, in Whewell form: its radius of curvature r
    against the turning angle phi from the apex tangent (phi >= 0 runs along the surface;
    the tangent angle is the apex one plus sense * phi, sense = -side). r = R_LE on
    [0, min_arc], the nose circle. Beyond, r is the cubic B-spline in
    eta = (phi - min_arc) / (anchor_angle - min_arc) with the clamped uniform knots of
    LE_FAIR_SPANS spans and the non-decreasing control values control, the first
    LE_FAIR_FIXED_CONTROLS of them R_LE and the last the source's radius of curvature at the
    anchor. So the curvature falls monotonically from 1/R_LE (none above it: no overshoot),
    keeps its sign (no inflection), leaves the circle with zero slope and zero second
    derivative (G4 there), and meets the source with G2 at the anchor;
    the tangent turns monotonically by construction. Positions are the apex plus the integral
    of r (cos, sin)(tangent angle) d phi: closed form on the circle, Gauss-Legendre per span
    beyond. anchor is the source parameter of the anchor point; fairness is
    R_LE^3 int (dkappa/ds)^2 ds over the fair."""

    nose: LeadingEdgeNose
    side: float                      # +1 suction, -1 pressure
    min_arc: float
    anchor_angle: float
    anchor: float
    control: NDArray[np.float64]
    fairness: float

    def __post_init__(self) -> None:
        _freeze_arrays(self)
        if np.any(np.diff(self.control) < 0.0) or self.control[0] != self.nose.radius:
            raise ValueError("a LeadingEdgeFair needs non-decreasing control radii starting at R_LE")
        spline = BSpline(_le_fair_knots(len(self.control) - 3), self.control, 3)
        object.__setattr__(self, "_spline", spline)
        object.__setattr__(self, "_slope", spline.derivative())
        object.__setattr__(self, "_area", spline.antiderivative())
        edges = np.linspace(0.0, 1.0, len(self.control) - 2)
        span = self.anchor_angle - self.min_arc
        lows, halves = edges[:-1], 0.5 * np.diff(edges)
        eta = (lows + halves)[:, None] + halves[:, None] * _LE_NODES[None]
        integrand = (spline(eta.ravel()) - self.nose.radius)[:, None] * self._direction(self.min_arc + span * eta.ravel())
        parts = span * halves[:, None] * np.einsum("q,kqd->kd", _LE_WEIGHTS, integrand.reshape(eta.shape + (2,)))
        cumulative = np.vstack((np.zeros((1, 2)), np.cumsum(parts, axis=0)))
        for name, value in (("_edges", edges), ("_cumulative", cumulative)):
            value.flags.writeable = False
            object.__setattr__(self, name, value)

    @property
    def sense(self) -> float:
        return -self.side

    @property
    def apex_angle(self) -> float:
        tangent = self.side * self.nose.normal
        return math.atan2(tangent[1], tangent[0])

    def _direction(self, phi: NDArray[np.float64]) -> Points:
        angle = self.apex_angle + self.sense * phi
        return np.column_stack((np.cos(angle), np.sin(angle)))

    def _eta(self, phi: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.clip((phi - self.min_arc) / (self.anchor_angle - self.min_arc), 0.0, 1.0)

    def radius(self, phi: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """r and dr/dphi."""
        phi = np.atleast_1d(np.asarray(phi, dtype=float))
        eta = self._eta(phi)
        on_circle = phi <= self.min_arc
        r = np.where(on_circle, self.nose.radius, self._spline(eta))
        slope = np.where(on_circle, 0.0, self._slope(eta) / (self.anchor_angle - self.min_arc))
        return r, slope

    def points(self, phi: ArrayLike) -> Points:
        phi = np.atleast_1d(np.asarray(phi, dtype=float))
        start, sense, radius = self.apex_angle, self.sense, self.nose.radius
        angle = start + sense * phi
        circle = radius * sense * np.column_stack((np.sin(angle) - math.sin(start), math.cos(start) - np.cos(angle)))
        out = self.nose.apex + circle
        beyond = phi > self.min_arc
        if np.any(beyond):
            eta = self._eta(phi[beyond])
            k = np.clip(np.searchsorted(self._edges, eta, side="right") - 1, 0, len(self._edges) - 2)
            low = self._edges[k]
            half = 0.5 * (eta - low)
            nodes = (low + half)[:, None] + half[:, None] * _LE_NODES[None]
            span = self.anchor_angle - self.min_arc
            values = ((self._spline(nodes.ravel()) - radius)[:, None]
                      * self._direction(self.min_arc + span * nodes.ravel())).reshape(nodes.shape + (2,))
            out[beyond] += self._cumulative[k] + span * half[:, None] * np.einsum("q,nqd->nd", _LE_WEIGHTS, values)
        return out

    def evaluate(self, tau: NDArray[np.float64]) -> tuple[Points, Points, Points]:
        """The fair as a PieceEvaluator: phi = tau * anchor_angle, tau in [0, 1]."""
        tau = np.asarray(tau, dtype=float)
        phi = tau * self.anchor_angle
        r, slope = self.radius(phi)
        direction = self._direction(phi)
        d1 = self.anchor_angle * r[:, None] * direction
        d2 = self.anchor_angle**2 * (slope[:, None] * direction + (self.sense * r)[:, None] * _rotated_left(direction))
        return self.points(phi), d1, d2

    def arc_length(self, phi: ArrayLike) -> NDArray[np.float64]:
        phi = np.atleast_1d(np.asarray(phi, dtype=float))
        span = self.anchor_angle - self.min_arc
        beyond = span * (self._area(self._eta(phi)) - self._area(0.0))
        return self.nose.radius * np.minimum(phi, self.min_arc) + np.where(phi > self.min_arc, beyond, 0.0)

    @property
    def length(self) -> float:
        return float(self.arc_length(self.anchor_angle)[0])

    def angle_at(self, s: ArrayLike) -> NDArray[np.float64]:
        """phi at arc length s from the apex (Newton on the monotone arc length)."""
        s = np.atleast_1d(np.asarray(s, dtype=float))
        radius, circle = self.nose.radius, self.nose.radius * self.min_arc
        phi = np.where(s <= circle, s / radius, 0.0)
        beyond = s > circle
        if np.any(beyond):
            target = s[beyond]
            guess = np.interp(target, [circle, self.length], [self.min_arc, self.anchor_angle])
            for _ in range(40):
                step = (self.arc_length(guess) - target) / self.radius(guess)[0]
                guess = np.clip(guess - step, self.min_arc, self.anchor_angle)
                if np.max(np.abs(step)) <= 1e-15 * self.anchor_angle:
                    break
            phi[beyond] = guess
        return phi

    @property
    def earliest_anchor(self) -> float:
        """The earliest source parameter at which fair_leading_edge found a fair (the anchor
        lies LE_ANCHOR_OFFSET R_LE past it); the anchor itself for a fair built otherwise."""
        return float(getattr(self, "_earliest_anchor", self.anchor))

    @property
    def departure_angle(self) -> float:
        """Where the fair has left the nose circle: the first phi at which the curvature has fallen
        LE_DEPARTURE_DROP below 1/R_LE (r = R_LE (1 + LE_DEPARTURE_DROP)). A threshold, not the
        last knot with r = R_LE, so it moves continuously with the blade parameters."""
        level = self.nose.radius * (1.0 + LE_DEPARTURE_DROP)
        phi = np.linspace(self.min_arc, self.anchor_angle, 4 * LE_FAIR_SPANS * LE_FAIR_GAUSS_POINTS + 1)
        above = np.flatnonzero(self.radius(phi)[0] >= level)
        if len(above) == 0:
            return self.anchor_angle
        if above[0] == 0:
            return self.min_arc
        return float(brentq(lambda x: float(self.radius(x)[0][0]) - level, phi[above[0] - 1], phi[above[0]],
                            xtol=1e-14, rtol=4.0 * np.finfo(float).eps))


def _le_fair_knots(n_spans: int) -> NDArray[np.float64]:
    return np.concatenate((np.zeros(3), np.linspace(0.0, 1.0, n_spans + 1), np.ones(3)))


def _solve_le_fair(nose: LeadingEdgeNose, side: float, point: NDArray[np.float64], tangent: NDArray[np.float64],
                   kappa: float, anchor: float) -> LeadingEdgeFair | None:
    """The fairest LeadingEdgeFair from the nose to the anchor (point, unit tangent, signed
    curvature): r = R_LE + sum c_j S_j(eta) with S_j the sum of the cubic B-splines from j on
    (monotone, 0 -> 1) and c >= 0, so the control radii R_LE + cumsum(c) never decrease; the
    landing point (2 equations) and r = 1/|kappa| at the anchor are linear in c, and
    int (dr/dphi)^2 / r^5 d phi = int (dkappa/ds)^2 ds is minimized by LE_FAIR_ITERATIONS
    reweighted least-squares steps (_lsei). None when no such fair lands on the anchor."""
    sense, radius = -side, nose.radius
    start_tangent = side * nose.normal
    anchor_angle = sense * math.atan2(start_tangent[0] * tangent[1] - start_tangent[1] * tangent[0],
                                      float(start_tangent @ tangent))
    if not (kappa * sense > 0.0 and anchor_angle > LE_FAIR_MIN_ARC and 1.0 / abs(kappa) > radius):
        return None
    end_radius = 1.0 / abs(kappa)
    span = anchor_angle - LE_FAIR_MIN_ARC
    knots = _le_fair_knots(LE_FAIR_SPANS)
    n_control = LE_FAIR_SPANS + 3
    edges = np.linspace(0.0, 1.0, LE_FAIR_SPANS + 1)
    halves = 0.5 * np.diff(edges)
    eta = ((edges[:-1] + halves)[:, None] + halves[:, None] * _LE_NODES[None]).ravel()
    weight = (halves[:, None] * _LE_WEIGHTS[None]).ravel() * span          # d phi
    unit = BSpline(knots, np.eye(n_control), 3)
    first = LE_FAIR_FIXED_CONTROLS
    cumulative = np.cumsum(unit(eta)[:, ::-1], axis=1)[:, ::-1][:, first:]       # S_j, j = first .. n-1
    cumulative_slope = np.cumsum(unit.derivative()(eta)[:, ::-1], axis=1)[:, ::-1][:, first:] / span
    apex_angle = math.atan2(start_tangent[1], start_tangent[0])
    angle = apex_angle + sense * (LE_FAIR_MIN_ARC + span * eta)
    direction = np.column_stack((np.cos(angle), np.sin(angle)))
    end_angle = apex_angle + sense * anchor_angle
    circle = radius * sense * np.array([math.sin(end_angle) - math.sin(apex_angle),
                                        math.cos(apex_angle) - math.cos(end_angle)])
    E = np.vstack((np.einsum("q,qj,qd->dj", weight, cumulative, direction), np.ones((1, cumulative.shape[1]))))
    f = np.concatenate((point - nose.apex - circle, [end_radius - radius]))
    scale = end_radius - radius
    E_scaled = E * scale
    norms = np.linalg.norm(E_scaled, axis=1)
    E_scaled, f_scaled = E_scaled / norms[:, None], f / norms
    r_hat = radius + scale * eta
    c = None
    for _ in range(LE_FAIR_ITERATIONS):
        A = np.sqrt(radius**3 * weight / r_hat**5)[:, None] * cumulative_slope * scale
        if not np.all(np.isfinite(A)):
            return None
        try:
            gamma = _lsei(A, np.zeros(len(eta)), E_scaled, f_scaled)
        except (np.linalg.LinAlgError, ValueError):
            return None
        if gamma is None:
            return None
        c = scale * np.maximum(gamma, 0.0)
        r_hat = radius + cumulative @ c
    if not np.all(np.isfinite(c)):
        return None
    r = radius + cumulative @ c
    fairness = float(radius**3 * np.sum(weight * (cumulative_slope @ c) ** 2 / r**5))
    control = radius + np.concatenate((np.zeros(first), np.cumsum(c)))
    try:
        with np.errstate(all="raise"):
            fair = LeadingEdgeFair(nose, side, LE_FAIR_MIN_ARC, anchor_angle, anchor, control, fairness)
            landed = float(np.hypot(*(fair.points(anchor_angle)[0] - point)))
    except (ValueError, FloatingPointError):
        return None
    if not landed <= LE_CLOSURE_TOL:
        return None
    return fair


def fair_leading_edge(nose: LeadingEdgeNose, side: float, source: SmoothedSurface) -> LeadingEdgeFair:
    """The C2 leading edge of one surface (side +1 suction, -1 pressure): the fairest monotone-
    curvature fair (_solve_le_fair) from the nose circle to an anchor on source.

    The anchor is LE_ANCHOR_OFFSET R_LE of source arc length past the earliest anchor at which
    such a fair exists (the source start when one exists there). That earliest anchor is
    found by scanning the source from its start in steps of LE_ANCHOR_STEP R_LE (over
    LE_ANCHOR_RANGE of its length) and bisecting between the last anchor without a fair and
    the first with one: a boundary, where the fairness energy climbs steeply, so it moves
    continuously with the blade parameters. The offset gives the fair room to turn gently.
    The anchor is not chosen where the fairness energy first meets a budget: along the
    source that energy is not monotone, and its first crossing jumps between branches as the
    blade changes (measured: 2e-5 c steps of the LE geometry). Deterministic, the same rule on
    either surface; GeometryError if no fair exists, or if its fairness exceeds
    LE_FAIRNESS_LIMIT. source needs only start, end and evaluate(s) -> (X, X', X'')."""
    step = LE_ANCHOR_STEP * nose.radius
    limit = source.start + LE_ANCHOR_RANGE * (source.end - source.start)

    def trial(s: float) -> LeadingEdgeFair | None:
        point, d1, d2 = (value[0] for value in source.evaluate(np.array([s])))
        return _solve_le_fair(nose, side, point, d1 / math.hypot(*d1), float(signed_curvature(d1[None], d2[None])[0]), s)

    previous, k = None, 0
    while True:
        s = min(source.start + k * step, limit)
        if trial(s) is not None:
            break
        if s >= limit:
            raise GeometryError(f"no LE fair lands on the source within {limit - source.start:.4g} c of its start")
        previous, k = s, k + 1
    earliest = s
    if previous is not None:
        low, high = previous, s
        for _ in range(LE_ANCHOR_BISECTIONS):
            middle = 0.5 * (low + high)
            if not low < middle < high:
                break
            if trial(middle) is None:
                low = middle
            else:
                high = middle
        earliest = high
    anchor = min(earliest + LE_ANCHOR_OFFSET * nose.radius, limit)
    fair = trial(anchor)
    if fair is None:
        raise GeometryError(f"no LE fair lands on the source at {anchor - source.start:.4g} c past its start, "
                            f"{LE_ANCHOR_OFFSET} R_LE past the earliest anchor with one")
    if fair.fairness > LE_FAIRNESS_LIMIT:
        raise GeometryError(f"the LE fair is too steep: R_LE^3 int kappa_s^2 ds = {fair.fairness:.3g} "
                            f"> {LE_FAIRNESS_LIMIT}")
    object.__setattr__(fair, "_earliest_anchor", earliest)
    return fair


@dataclass(frozen=True, eq=False)
class C2SurfaceFit:
    """The C2 curve of one surface with its robust leading edge, and what it cost. The curve
    parameter is u = s' / length, s' the arc length of the LE fair from the apex to its anchor,
    then the source's reference arc length from the anchor on. Deviations are normal distances
    from the C1 reference to the curve, in chords: le_* over the LE zone (apex to anchor, where
    the geometry is redesigned), the others downstream of it (where C1 is kept to the C2
    representation tolerance). The representation error is the curve against its target."""

    curve: ParametricBladeCurve
    reference: ReferenceSurface
    source: SmoothedSurface
    leading_edge: LeadingEdgeFair
    length: float
    max_deviation: float
    rms_deviation: float
    le_max_deviation: float
    le_rms_deviation: float
    max_representation_error: float
    n_samples: int

    @property
    def anchor_u(self) -> float:
        """u of the anchor: the fraction of the surface the LE fair replaces."""
        return self.leading_edge.length / self.length

    @property
    def departure_u(self) -> float:
        return float(self.leading_edge.arc_length(self.leading_edge.departure_angle)[0]) / self.length

    def parameter(self, s: ArrayLike) -> NDArray[np.float64]:
        """u for reference arc lengths s: the LE zone [0, anchor] mapped proportionally onto the
        fair, the rest shifted onto the source part."""
        s = np.asarray(s, dtype=float)
        le, anchor = self.leading_edge.length, self.leading_edge.anchor
        return np.where(s <= anchor, s * le / anchor, le + (s - anchor)) / self.length


def fit_c2_surface(reference: ReferenceSurface, nose: LeadingEdgeNose, side: float,
                   first_piece: int) -> C2SurfaceFit:
    """The C2 representation of one surface with the robust LE: the reference from its piece
    first_piece on, smoothed as fit_quintic_surface smooths it (SmoothedSurface), joined at
    the anchor fair_leading_edge picks to the LE fair from the nose, and represented by a
    clamped quintic B-spline fitted to that target by arc-length weighted least squares. Knot
    spacing as in fit_quintic_surface (at most a twentieth of the radius of curvature and a
    third of each downstream transition half-width, graded), and in the LE fair at most
    1/LE_KNOTS_PER_CURVATURE_CHANGE of |kappa| / |dkappa/ds| and LE_ANCHOR_KNOT_SPACING R_LE
    within LE_ANCHOR_KNOT_ZONE R_LE of the departure and the anchor. Imposed exactly: the
    apex point, its unit tangent and curvature 1/R_LE; at the TE the point and tangent, plus
    the curvature at a rounded TE."""
    source = SmoothedSurface(reference, first_piece)
    le = fair_leading_edge(nose, side, source)
    le_length = le.length
    length = le_length + (source.end - le.anchor)

    def target_at(s: NDArray[np.float64]) -> tuple[Points, Points, Points]:
        points, d1, d2 = (np.zeros((len(s), 2)) for _ in range(3))
        on_le = s <= le_length
        if np.any(on_le):
            phi = le.angle_at(s[on_le])
            r = le.radius(phi)[0]
            direction = le._direction(phi)
            points[on_le] = le.points(phi)
            d1[on_le] = direction
            d2[on_le] = (le.sense / r)[:, None] * _rotated_left(direction)
        if np.any(~on_le):
            points[~on_le], d1[~on_le], d2[~on_le] = source.evaluate(s[~on_le] - le_length + le.anchor)
        return points, d1, d2

    s_grid = np.linspace(0.0, length, C2_SIZE_GRID)
    coarse = np.linspace(0.0, length, 64 * C2_COARSE_POINTS + 1)
    _, c1, c2 = target_at(coarse)
    curvature = np.interp(s_grid, coarse, np.abs(signed_curvature(c1, c2)))
    spacing = np.minimum(C2_MAX_KNOT_SPACING, 1.0 / (C2_KNOTS_PER_RADIUS * np.maximum(curvature, 1e-300)))
    on_le = s_grid <= le_length
    r, slope = le.radius(le.angle_at(s_grid[on_le]))
    rate = np.abs(slope) / r**2                                 # |dkappa/ds| / |kappa| = |r'| / r^2
    spacing[on_le] = np.minimum(spacing[on_le], 1.0 / (LE_KNOTS_PER_CURVATURE_CHANGE * np.maximum(rate, 1e-300)))
    departure = float(le.arc_length(le.departure_angle)[0])
    for join in (departure, le_length):         # where the fair leaves the circle and where it meets the source
        near = np.abs(s_grid - join) < LE_ANCHOR_KNOT_ZONE * le.nose.radius
        spacing[near] = np.minimum(spacing[near], LE_ANCHOR_KNOT_SPACING * le.nose.radius)
    for t in source.transitions:
        center = t.junction.s - le.anchor + le_length
        if center + t.half_width > le_length:
            gap = np.maximum(np.abs(s_grid - center) - t.half_width, 0.0)
            spacing = np.minimum(spacing, t.half_width / C2_KNOTS_PER_HALF_WIDTH + C2_KNOT_GRADING * gap)
    graded = C2_KNOT_GRADING * s_grid
    spacing = np.minimum(spacing, graded + np.minimum.accumulate(spacing - graded))
    spacing = np.minimum(spacing, (np.minimum.accumulate((spacing + graded)[::-1]))[::-1] - graded)

    density = C2_SAMPLES_PER_SPAN / spacing
    count = np.concatenate(([0.0], np.cumsum(0.5 * (density[1:] + density[:-1]) * np.diff(s_grid))))
    s = np.interp(np.linspace(0.0, count[-1], int(math.ceil(count[-1])) + 1), count, s_grid)
    s = np.unique(np.concatenate((s, [le_length])))
    s[0], s[-1] = 0.0, length
    target, d1, d2 = target_at(s)
    step = np.diff(s)
    weights = np.concatenate((0.5 * step, [0.0])) + np.concatenate(([0.0], 0.5 * step))
    knots = _c2_knots(s_grid, spacing, length)
    u = np.clip(s / length, 0.0, 1.0)
    u[-1] = 1.0

    def end_values(i: int, point: NDArray[np.float64], curvature: bool) -> list[NDArray[np.float64]]:
        tangent = d1[i] / math.hypot(*d1[i])
        values = [point, length * tangent]
        if curvature:
            kappa = float(signed_curvature(d1[i][None], d2[i][None])[0])
            values.append(length**2 * kappa * np.array([-tangent[1], tangent[0]]))
        return values

    if not np.array_equal(nose.apex, reference.leading_edge_point):
        raise GeometryError(f"the LE nose apex is not the {reference.name} reference LE point")
    control = _fit_clamped_quintic(knots, u, target, weights, end_values(0, reference.leading_edge_point, True),
                                   end_values(-1, reference.trailing_edge_point, not reference.sharp_trailing_edge))
    curve = ParametricBladeCurve(knots, control, length, C2_DEGREE)
    representation = float(np.max(np.hypot(*(curve.evaluate(u) - target).T)))

    fit = C2SurfaceFit(curve, reference, source, le, length, 0.0, 0.0, 0.0, 0.0, representation, len(u))
    sample = reference.sample(201)
    normal = curve.project(sample.points, fit.parameter(sample.s))[1]
    w = sample.weights
    zone = sample.s <= le.anchor
    out = {}
    for name, rows in (("le", zone), ("", ~zone)):
        weight = w * rows
        out[name] = (float(np.max(np.abs(normal[rows]))),
                     math.sqrt(float(np.sum(weight * normal**2) / np.sum(weight))))
    return replace(fit, max_deviation=out[""][0], rms_deviation=out[""][1],
                   le_max_deviation=out["le"][0], le_rms_deviation=out["le"][1])


def leading_edge_fairness(curve: ParametricBladeCurve, u_end: float, n: int = 4001) -> dict[str, float]:
    """Fairness of a curve over u in [0, u_end] (its LE zone), from its exact derivatives on n
    points: curvature sign changes, the largest reverse turn (the most the tangent turns back
    against the apex curvature's sense over a run), the total curvature variation
    int |dkappa| (/c), max |dkappa/ds| (/c^2), and the curvature range (/c)."""
    u = np.linspace(0.0, u_end, n)
    d1 = curve.derivative(u, 1)
    kappa = curve.curvature(u)
    sense = float(np.sign(kappa[0]))
    angle = np.unwrap(np.arctan2(d1[:, 1], d1[:, 0])) * sense
    # The largest drop of the sense-oriented tangent angle below its running maximum.
    reverse = float(np.max(np.maximum.accumulate(angle) - angle))
    signs = np.sign(kappa[np.abs(kappa) > 0.0])
    ds = np.hypot(*d1.T) * (u_end / (n - 1))
    rate = np.abs(np.diff(kappa)) / (0.5 * (ds[1:] + ds[:-1]))
    return {
        "sign_changes": int(np.sum(signs[1:] != signs[:-1])),
        "max_reverse_turn_deg": math.degrees(reverse),
        "curvature_variation": float(np.sum(np.abs(np.diff(kappa)))),
        "max_curvature_rate": float(np.max(rate)),
        "kappa_min": float(np.min(kappa)),
        "kappa_max": float(np.max(kappa)),
    }


# --- NACA 65 adapter ---------------------------------------------------------------------

def _naca65_nose(cl0: float, t_max_over_c: float, apex: NDArray[np.float64]) -> LeadingEdgeNose:
    """The legacy NACA 65 nose circle, from the same formulas as _naca65_contour: R_LE =
    TABLE_LE_RADIUS (t / 0.10)^2, centered on the mean-line tangent at x/c = LE_TANGENT_X."""
    t_nominal = TABLE_T_MAX * thickness_scale(t_max_over_c)
    radius = TABLE_LE_RADIUS * (t_nominal / TABLE_T_MAX) ** 2
    phi_le = float(a10_slope_angle(LE_TANGENT_X, cl0))
    nose_axis = -np.array([np.cos(phi_le), np.sin(phi_le)])
    return LeadingEdgeNose(np.array(apex, dtype=float), -radius * nose_axis, radius, LE_RADIUS_SOURCE_NACA65)


NACA65_FIRST_SURFACE_PIECE = 2      # pieces 0, 1 are the legacy LE circle and fair; 2 starts the table surface


@lru_cache(maxsize=32)
def _naca65_quintic_fits(cl0: float, t_max_over_c: float,
                         profile_variant: str) -> tuple[C2SurfaceFit, C2SurfaceFit]:
    suction, pressure = _naca65_reference_surfaces(cl0, t_max_over_c, profile_variant)
    try:
        return tuple(fit_c2_surface(reference, _naca65_nose(cl0, t_max_over_c, reference.leading_edge_point),
                                    side, NACA65_FIRST_SURFACE_PIECE)
                     for reference, side in ((suction, 1.0), (pressure, -1.0)))
    except GeometryError:
        raise
    except (ValueError, np.linalg.LinAlgError) as error:
        raise GeometryError(f"C2 fit failed for cl0={cl0}, t_max_over_c={t_max_over_c}: {error}") from error


def _naca65_c2_profile(profile: BladeProfile, cl0: float, t_max_over_c: float, profile_variant: str
                       ) -> tuple[BladeProfile, tuple[tuple[ParametricBladeCurve, NDArray[np.float64]], ...]]:
    """The C1 profile with its surfaces replaced by the C2 curves, sampled where the C1
    polyline has its vertices: their reference arc lengths mapped by C2SurfaceFit.parameter
    (proportionally over the LE zone, shifted after it), and those curves with their sample
    parameters. Every other field is the C1 one."""
    fits = _naca65_quintic_fits(cl0, t_max_over_c, profile_variant)
    surfaces, curves = [], []
    for fit, polyline in zip(fits, (profile.suction_surface, profile.pressure_surface)):
        reference, curve = fit.reference, fit.curve
        if not (np.array_equal(curve.control_points[0], polyline[0])
                and np.array_equal(curve.control_points[-1], polyline[-1])):
            raise GeometryError(f"the C2 {reference.name} curve does not end on the C1 LE and TE vertices")
        u = fit.parameter(_anchored_arc_length(reference, polyline))
        u[0], u[-1] = 0.0, 1.0
        if np.any(np.diff(u) <= 0.0):
            raise GeometryError(f"the C1 {reference.name} vertices do not run along their reference surface")
        points = curve.evaluate(u)
        surfaces.append(points)
        curves.append((curve, u))
    nose = fits[0].leading_edge.nose
    metadata = {
        **profile.metadata,
        "surface_representation": "C2: clamped quintic B-splines, C4 at their simple knots; downstream of each LE "
                                  "anchor fitted to the C1 construction with monotone junction transitions, "
                                  "ahead of it the monotone-curvature LE fair",
        "c2_deviation_target": C2_DEVIATION_TARGET,
        "c2_max_deviation_over_c": max(fit.max_deviation for fit in fits),
        "c2_rms_deviation_over_c": max(fit.rms_deviation for fit in fits),
        "c2_control_points": tuple(len(fit.curve.control_points) for fit in fits),
        "le_model": LE_MODEL,
        "le_radius_over_c": nose.radius,
        "le_radius_source": nose.radius_source,
        "le_anchor_offset_radii": LE_ANCHOR_OFFSET,
        "le_min_arc_deg": math.degrees(LE_FAIR_MIN_ARC),
    }
    for prefix, fit in (("ss", fits[0]), ("ps", fits[1])):
        le = fit.leading_edge
        metadata.update({
            f"{prefix}_le_departure_u": fit.departure_u,
            f"{prefix}_le_anchor_u": fit.anchor_u,
            f"{prefix}_le_departure_deg": math.degrees(le.departure_angle),
            f"{prefix}_le_anchor_deg": math.degrees(le.anchor_angle),
            f"{prefix}_le_anchor_over_c": le.anchor,
            f"{prefix}_le_anchor_past_source_start_over_c": le.anchor - fit.source.start,
            f"{prefix}_le_earliest_anchor_past_source_start_over_c": le.earliest_anchor - fit.source.start,
            f"{prefix}_le_fairness": le.fairness,
            f"{prefix}_le_max_deviation_over_c": fit.le_max_deviation,
            f"{prefix}_le_rms_deviation_over_c": fit.le_rms_deviation,
        })
    return (replace(profile, suction_surface=surfaces[0], pressure_surface=surfaces[1],
                    metadata=MappingProxyType(metadata)), tuple(curves))


# ---------------------------------------------------------------------------------------
# Family-independent pieces
# ---------------------------------------------------------------------------------------

def cosine_stations(n: int) -> NDArray[np.float64]:
    """n chord stations from 0 to 1, clustered at both ends."""
    return 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, n)))


def split_at_leading_edge(loop: Points) -> tuple[Points, Points]:
    """(upper, lower) surfaces, each LE -> TE, of a closed loop ordered TE -> upper -> LE ->
    lower -> TE whose LE vertex is the origin of the blade frame."""
    i_le = int(np.argmin(np.hypot(loop[:, 0], loop[:, 1])))
    if math.hypot(*loop[i_le]) > EXACT_TOL:
        raise GeometryError("the blade contour has no vertex at the leading-edge point (0, 0)")
    return loop[i_le::-1].copy(), loop[i_le:].copy()


def closed_loop(suction: Points, pressure: Points) -> Points:
    """The closed contour TE -> suction -> LE -> pressure -> TE; its first and last points
    are the TE. It runs counterclockwise when the suction surface is on the +y side."""
    return np.vstack((suction[::-1], pressure[1:]))


def rotate_by_stagger(points: ArrayLike, stagger: float) -> NDArray[np.float64]:
    """Blade-frame points turned counterclockwise by stagger (radians) about the LE
    [Aungier 2003 Sec. 4.10]: X = x cos - y sin, Y = x sin + y cos."""
    c, s = math.cos(stagger), math.sin(stagger)
    return np.asarray(points, dtype=float) @ np.array([[c, s], [-s, c]])


def place_cascade(profile: BladeProfile, solidity: float, stagger: float) -> CascadeCoordinates:
    """The profile at stagger (radians) and pitch 1 / solidity, chord 1, with its throat."""
    pitch = 1.0 / solidity
    translation = np.array([0.0, pitch])
    suction = rotate_by_stagger(profile.suction_surface, stagger)
    pressure = rotate_by_stagger(profile.pressure_surface, stagger)

    loop = closed_loop(suction, pressure)
    if _paths_cross(loop, loop + translation):
        raise GeometryError(f"neighboring blades overlap at solidity {solidity} and stagger "
                            f"{math.degrees(stagger):.6g} deg")
    opening, point_reference, point_neighbor = find_geometric_throat(suction, pressure + translation)

    return CascadeCoordinates(
        chord=math.hypot(*(profile.suction_surface[-1] - profile.suction_surface[0])),
        pitch=pitch,
        camberline_unstaggered=profile.camberline,
        equivalent_arc_camberline_unstaggered=profile.equivalent_arc_camberline,
        suction_surface_unstaggered=profile.suction_surface,
        pressure_surface_unstaggered=profile.pressure_surface,
        camberline=rotate_by_stagger(profile.camberline, stagger),
        equivalent_arc_camberline=rotate_by_stagger(profile.equivalent_arc_camberline, stagger),
        suction_surface=suction,
        pressure_surface=pressure,
        leading_edge_point=suction[0],
        trailing_edge_point=suction[-1],
        leading_edge_center=rotate_by_stagger(profile.leading_edge_center, stagger),
        leading_edge_radius=profile.leading_edge_radius,
        trailing_edge_center=rotate_by_stagger(profile.trailing_edge_center, stagger),
        trailing_edge_radius=profile.trailing_edge_radius,
        periodic_translation=translation,
        throat_opening=opening,
        throat_point_reference=point_reference,
        throat_point_neighbor=point_neighbor,
    )


def find_geometric_throat(suction_reference: Points,
                          pressure_neighbor: Points) -> tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    """Shortest distance between the suction surface of the reference blade and the pressure
    surface of its neighbor, and its two end points [Aungier 2003 Sec. 4.9]."""
    return closest_points(suction_reference, pressure_neighbor)


def closest_points(curve_a: Points, curve_b: Points) -> tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    """Shortest distance between two smooth curves given as polylines, and its end points.

    Each polyline becomes a cubic spline in chord length. Every vertex of one curve is
    projected onto the other polyline, and each local minimum of that distance, along
    either curve and within CANDIDATE_MARGIN of the smallest, starts a refinement on the
    splines (_refine_closest). The best refined pair wins. A single start is not enough:
    the small dent of the lower nose fair holds two minima a few thousandths of a chord
    apart, and a start near a curve end can stop there. Vertices are densest where the
    blade is most curved, at the LE and TE, so such a second minimum is not missed. A
    parameter that reaches the end of its curve stays there, so a minimum at a LE or TE
    needs no special case.
    """
    spline_a, spline_b = _chord_length_spline(curve_a), _chord_length_spline(curve_b)
    best: tuple[float, NDArray[np.float64], NDArray[np.float64]] | None = None
    for start in _candidate_starts(curve_a, spline_a.x, curve_b, spline_b.x):
        u, v = _refine_closest(spline_a, spline_b, start)
        point_a, point_b = spline_a(u), spline_b(v)
        distance = math.hypot(*(point_a - point_b))
        if best is None or distance < best[0]:
            best = (distance, point_a, point_b)
    assert best is not None     # the smallest vertex distance is always a candidate
    return best


def _vertex_distances(points: Points, polyline: Points) -> tuple[NDArray[np.float64], NDArray[np.int64],
                                                                  NDArray[np.float64]]:
    """Distance from each point to a polyline, the segment index of the nearest foot, and
    the fraction along that segment."""
    start, step = polyline[None, :-1], np.diff(polyline, axis=0)[None]
    offset = points[:, None, :] - start
    along = np.clip(np.sum(offset * step, axis=2) / np.sum(step * step, axis=2), 0.0, 1.0)
    gap = offset - along[..., None] * step
    distance = np.hypot(gap[..., 0], gap[..., 1])
    segment = np.argmin(distance, axis=1)
    rows = np.arange(len(points))
    return distance[rows, segment], segment, along[rows, segment]


def _candidate_starts(curve_a: Points, knots_a: NDArray[np.float64], curve_b: Points,
                      knots_b: NDArray[np.float64]) -> list[NDArray[np.float64]]:
    """(u, v) starting pairs for the refinement: the local minima of the vertex-to-polyline
    distance along each curve, within CANDIDATE_MARGIN of the smallest."""
    d_a, seg_b, along_b = _vertex_distances(curve_a, curve_b)
    d_b, seg_a, along_a = _vertex_distances(curve_b, curve_a)
    ceiling = min(d_a.min(), d_b.min()) + CANDIDATE_MARGIN
    v_feet = knots_b[seg_b] + along_b * np.diff(knots_b)[seg_b]
    u_feet = knots_a[seg_a] + along_a * np.diff(knots_a)[seg_a]
    starts = [np.array([knots_a[i], v_feet[i]]) for i in _local_minima(d_a, ceiling)]
    starts += [np.array([u_feet[j], knots_b[j]]) for j in _local_minima(d_b, ceiling)]
    return starts


def _local_minima(values: NDArray[np.float64], ceiling: float) -> NDArray[np.int64]:
    """Indices where values is no larger than its neighbors and no larger than ceiling."""
    padded = np.concatenate(([np.inf], values, [np.inf]))
    is_minimum = (values <= padded[:-2]) & (values <= padded[2:]) & (values <= ceiling)
    return np.flatnonzero(is_minimum)


def _chord_length_spline(points: Points) -> CubicSpline:
    """Cubic spline through points, parametrized by cumulative chord length."""
    steps = np.hypot(*np.diff(points, axis=0).T)
    return CubicSpline(np.concatenate(([0.0], np.cumsum(steps))), points)


def _refine_closest(spline_a: CubicSpline, spline_b: CubicSpline, p: NDArray[np.float64]) -> NDArray[np.float64]:
    """Minimize f(u, v) = |a(u) - b(v)|^2 / 2 with u and v kept inside their spline ranges.

    Projected Newton with a backtracking (Armijo) line search: a Newton step where the
    Hessian of the free parameters is positive definite, a steepest-descent step where it
    is not, and a parameter held at its end while the gradient pushes it outward. Every
    accepted step lowers f, so the iteration cannot cycle.
    """
    upper = np.array([spline_a.x[-1], spline_b.x[-1]])

    def value(q: NDArray[np.float64]) -> float:
        return 0.5 * float(np.sum((spline_a(q[0]) - spline_b(q[1])) ** 2))

    f = value(p)
    for _ in range(MAX_THROAT_ITERATIONS):
        a, a1, a2 = (spline_a(p[0], k) for k in range(3))
        b, b1, b2 = (spline_b(p[1], k) for k in range(3))
        gap = a - b
        grad = np.array([gap @ a1, -(gap @ b1)])
        hess = np.array([[a1 @ a1 + gap @ a2, -(a1 @ b1)],
                         [-(a1 @ b1), b1 @ b1 - gap @ b2]])
        free = ~(((p <= 0.0) & (grad > 0.0)) | ((p >= upper) & (grad < 0.0)))
        if not free.any():
            return p
        block = hess[np.ix_(free, free)]
        step = np.zeros(2)
        newton = bool(np.min(np.linalg.eigvalsh(block)) > 0.0)
        if newton:
            step[free] = np.linalg.solve(block, -grad[free])
        else:
            step[free] = -grad[free]

        if newton and np.max(np.abs(step)) <= NEWTON_ZONE:
            # Close to the minimum f changes by less than its round-off, so a decrease test
            # would stop near sqrt(eps). Newton on the gradient converges to round-off.
            trial = np.clip(p + step, 0.0, upper)
            if np.max(np.abs(trial - p)) <= THROAT_PARAMETER_TOL:
                return trial
            p, f = trial, value(trial)
            continue

        fraction = 1.0
        while True:
            trial = np.clip(p + fraction * step, 0.0, upper)
            f_trial = value(trial)
            if f_trial <= f + ARMIJO * float(grad @ (trial - p)) or fraction < MIN_STEP_FRACTION:
                break
            fraction *= 0.5
        if np.max(np.abs(trial - p)) <= THROAT_PARAMETER_TOL:
            return trial
        if f_trial > f:
            return p        # no step lowers f: stationary to round-off
        p, f = trial, f_trial
    raise GeometryError("geometric throat refinement did not converge")


def signed_area(loop: Points) -> float:
    """Shoelace area of a closed loop whose last point repeats its first; positive when the
    loop runs counterclockwise."""
    x, y = loop[:, 0], loop[:, 1]
    return 0.5 * float(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))


def inscribed_thickness(suction: Points, pressure: Points) -> float:
    """Largest distance from a suction-surface vertex to the pressure-surface polyline: the
    blade's maximum thickness measured without the camberline, a check on the construction."""
    return float(np.max(_vertex_distances(suction, pressure)[0]))


def _turn(origin: NDArray[np.float64], a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    """z component of (a - origin) x (b - origin), broadcast over leading axes."""
    return (a[..., 0] - origin[..., 0]) * (b[..., 1] - origin[..., 1]) - (a[..., 1] - origin[..., 1]) * (b[..., 0] - origin[..., 0])


def _paths_cross(path_a: Points, path_b: Points) -> bool:
    """True if any segment of path_a properly crosses a segment of path_b. Segments that
    only share an end point do not count, so a closed loop can be tested against itself."""
    a0, a1 = path_a[:-1, None, :], path_a[1:, None, :]
    b0, b1 = path_b[None, :-1, :], path_b[None, 1:, :]
    straddle_b = _turn(a0, a1, b0) * _turn(a0, a1, b1) < 0.0
    straddle_a = _turn(b0, b1, a0) * _turn(b0, b1, a1) < 0.0
    return bool(np.any(straddle_a & straddle_b))


# ---------------------------------------------------------------------------------------
# Invariants
# ---------------------------------------------------------------------------------------

def validate_cascade(cascade: Cascade) -> None:
    """Check the invariants grid.py relies on and raise GeometryError listing every one
    that fails. make_cascade calls this before it returns."""
    problems = _check_shapes(cascade)
    if not problems:
        for check in (_check_frame, _check_surfaces, _check_thickness, _check_periodicity, _check_throat,
                      _check_angles, _check_edges, _check_scaling, _check_curves):
            problems += check(cascade)
    if problems:
        raise GeometryError("invalid cascade geometry:\n- " + "\n- ".join(problems))


def _check_shapes(cascade: Cascade) -> list[str]:
    problems = []
    for coordinates, label in ((cascade.normalized, "normalized"), (cascade.active, "active")):
        for item in fields(coordinates):
            value = getattr(coordinates, item.name)
            if isinstance(value, np.ndarray):
                is_point = value.shape == (2,)
                is_curve = value.ndim == 2 and value.shape[1] == 2 and len(value) >= 3
                if not (is_point or is_curve) or not np.all(np.isfinite(value)):
                    problems.append(f"{label}.{item.name} is not a finite point or (n, 2) curve")
            elif not math.isfinite(value):
                problems.append(f"{label}.{item.name} is not finite")
    return problems


def _close(a: ArrayLike, b: ArrayLike, tol: float = EXACT_TOL) -> bool:
    return bool(np.max(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))) <= tol)


def _check_frame(cascade: Cascade) -> list[str]:
    n, problems = cascade.normalized, []
    te_expected = np.array([math.cos(cascade.stagger), math.sin(cascade.stagger)])
    if not _close(cascade.stagger, math.radians(cascade.spec.stagger_deg)):
        problems.append("stagger is not radians(spec.stagger_deg)")
    if not _close(n.chord, 1.0):
        problems.append(f"normalized chord is {n.chord!r}, not 1")
    if not _close(n.pitch, 1.0 / cascade.spec.solidity):
        problems.append(f"pitch {n.pitch!r} is not 1 / solidity")
    if not _close(n.periodic_translation, [0.0, n.pitch]):
        problems.append("periodic_translation is not (0, pitch)")
    if not (_close(n.leading_edge_point, [0.0, 0.0]) and _close(n.trailing_edge_point, te_expected)):
        problems.append("LE is not at the origin or TE is not at (cos(stagger), sin(stagger))")
    for name in ("suction_surface", "pressure_surface"):
        unstaggered, staggered = getattr(n, f"{name}_unstaggered"), getattr(n, name)
        if not (_close(unstaggered[0], [0.0, 0.0]) and _close(unstaggered[-1], [1.0, 0.0])):
            problems.append(f"{name}_unstaggered does not run from (0, 0) to (1, 0)")
        if not (_close(staggered[0], n.leading_edge_point) and _close(staggered[-1], n.trailing_edge_point)):
            problems.append(f"{name} does not run from the LE point to the TE point")
        if not _close(staggered, rotate_by_stagger(unstaggered, cascade.stagger)):
            problems.append(f"{name} is not {name}_unstaggered turned by the stagger")
    return problems


def _check_surfaces(cascade: Cascade) -> list[str]:
    n, problems = cascade.normalized, []
    suction, pressure = n.suction_surface_unstaggered, n.pressure_surface_unstaggered
    for name, points in (("suction", suction), ("pressure", pressure)):
        if np.min(np.hypot(*np.diff(points, axis=0).T)) <= EXACT_TOL:
            problems.append(f"{name} surface repeats a point")
    loop = closed_loop(suction, pressure)
    if _paths_cross(loop, loop):
        problems.append("suction and pressure surfaces cross")
    if signed_area(loop) <= 0.0:
        problems.append("the suction surface is not on the +y side of the blade frame")
    return problems


def _check_thickness(cascade: Cascade) -> list[str]:
    """The blade's measured maximum thickness against the one its family construction
    promises, metadata["max_thickness_over_c"] (0.9992 t_max_over_c for NACA 65)."""
    n = cascade.normalized
    expected = cascade.metadata.get("max_thickness_over_c")
    if expected is None:
        return ['metadata has no "max_thickness_over_c" to check the thickness against']
    measured = inscribed_thickness(n.suction_surface_unstaggered, n.pressure_surface_unstaggered)
    if abs(measured - expected) > THICKNESS_RTOL * expected:
        return [f"maximum thickness {measured:.6g} does not match the expected {expected:.6g}"]
    return []


def _check_periodicity(cascade: Cascade) -> list[str]:
    n = cascade.normalized
    loop = closed_loop(n.suction_surface, n.pressure_surface)
    return ["neighboring blades overlap"] if _paths_cross(loop, loop + n.periodic_translation) else []


def _check_throat(cascade: Cascade) -> list[str]:
    n, problems = cascade.normalized, []
    if not 0.0 < n.throat_opening < n.pitch:
        problems.append(f"throat opening {n.throat_opening!r} is not between 0 and the pitch")
    if not _close(np.hypot(*(n.throat_point_reference - n.throat_point_neighbor)), n.throat_opening):
        problems.append("the throat points are not one throat opening apart")
    if not (_close(cascade.throat_over_pitch, n.throat_opening / n.pitch)
            and _close(cascade.throat_over_chord, n.throat_opening / n.chord)):
        problems.append("throat ratios do not match the throat opening")
    loop = closed_loop(n.suction_surface, n.pressure_surface)
    whole_blades = closest_points(loop, loop + n.periodic_translation)[0]
    if whole_blades < n.throat_opening - THROAT_TOL:
        problems.append(f"the blades come closer ({whole_blades:.9g}) than the suction-to-pressure "
                        f"throat ({n.throat_opening:.9g}) away from those surfaces")
    return problems


def _check_angles(cascade: Cascade) -> list[str]:
    c, problems = cascade, []
    if not (_close(c.camber_angle, c.kappa_le - c.kappa_te) and _close(c.kappa_le, c.stagger + c.chi_le)
            and _close(c.kappa_te, c.stagger - c.chi_te)):
        problems.append("kappa_le, kappa_te, stagger and camber_angle are inconsistent")
    if not (_close(c.chi_le, 0.5 * c.camber_angle) and _close(c.chi_te, 0.5 * c.camber_angle)):
        problems.append("chi_le and chi_te are not camber_angle / 2")
    return problems


def _check_edges(cascade: Cascade) -> list[str]:
    """LE and TE circles. The Kovach taper is tangent to the TE circle along the
    straightened camberline only, so on the real, curved camberline the pressure surface
    dips slightly inside the circle before its radial fair reaches it: up to 3.1e-6 c
    (0.04 % of R_TE) at cl0 <= 1.6, t_max_over_c <= 0.10. TE_CIRCLE_RTOL allows that much."""
    n, problems = cascade.normalized, []
    if not _close(np.hypot(*(n.leading_edge_point - n.leading_edge_center)), n.leading_edge_radius, EDGE_TOL):
        problems.append("the LE circle does not pass through the LE point")
    radius, center = n.trailing_edge_radius, n.trailing_edge_center
    if not _close(np.hypot(*(n.trailing_edge_point - center)), radius, EDGE_TOL):
        problems.append("the TE circle does not pass through the TE point")
    variant = cascade.spec.profile_variant
    expected = KOVACH_TE_RADIUS if variant == "kovach_sandercock" else 0.0
    if not _close(radius, expected):
        problems.append(f"TE radius {radius!r} does not belong to profile_variant {variant!r}")
    for name in ("suction_surface", "pressure_surface"):
        distance = np.hypot(*(getattr(n, name) - center).T)
        if np.min(distance) < radius * (1.0 - TE_CIRCLE_RTOL) - EDGE_TOL:
            problems.append(f"the {name} cuts into the TE circle by {radius - np.min(distance):.3g} c")
        if radius > 0.0 and not _close(distance[-3:], radius, EDGE_TOL):
            problems.append(f"the {name} does not end on the TE circle")
    return problems


def _check_scaling(cascade: Cascade) -> list[str]:
    scale = 1.0 if cascade.spec.physical_chord is None else cascade.spec.physical_chord
    for item in fields(cascade.normalized):
        expected = getattr(cascade.normalized, item.name) * scale
        actual = getattr(cascade.active, item.name)
        if not np.allclose(actual, expected, rtol=EXACT_TOL, atol=EXACT_TOL * scale):
            return [f"active.{item.name} is not normalized.{item.name} times the chord {scale!r}"]
    return []


def _check_curves(cascade: Cascade) -> list[str]:
    """C1 carries no curves. C2: the arrays are the curves sampled at their parameters, in
    both frames and both units; the cascade-frame curves are the blade-frame ones turned by
    the stagger; the active curves are the normalized ones scaled; every curve is a clamped
    quintic with simple knots, so C4 inside each surface."""
    sets = (cascade.normalized_curves, cascade.active_curves)
    if cascade.spec.surface_continuity != "C2":
        return [] if sets == (None, None) else ["a C1 cascade carries parametric surface curves"]
    if any(item is None for item in sets):
        return ["a C2 cascade has no parametric surface curves"]
    scale = 1.0 if cascade.spec.physical_chord is None else cascade.spec.physical_chord
    problems = []
    for coordinates, curves, label, unit in ((cascade.normalized, sets[0], "normalized", 1.0),
                                             (cascade.active, sets[1], "active", scale)):
        for name in ("suction", "pressure"):
            u = getattr(curves, f"{name}_parameters")
            if not (u[0] == 0.0 and u[-1] == 1.0 and np.all(np.diff(u) > 0.0)
                    and len(u) == len(getattr(coordinates, f"{name}_surface"))):
                problems.append(f"{label} {name} parameters do not run from 0 to 1, one per surface point")
                continue
            for suffix in ("", "_unstaggered"):
                curve = getattr(curves, name + suffix)
                if curve.degree != C2_DEGREE:
                    problems.append(f"{label} {name}{suffix} curve is not of degree {C2_DEGREE}")
                if not _close(curve.evaluate(u), getattr(coordinates, f"{name}_surface{suffix}"), EXACT_TOL * unit):
                    problems.append(f"{label}.{name}_surface{suffix} is not its curve sampled at its parameters")
            turned = rotate_by_stagger(getattr(curves, f"{name}_unstaggered").control_points, cascade.stagger)
            if not _close(getattr(curves, name).control_points, turned, EXACT_TOL * unit):
                problems.append(f"{label} {name} curve is not the blade-frame curve turned by the stagger")
            reference = getattr(sets[0], name)
            if not _close(getattr(curves, name).control_points, reference.control_points * unit, EXACT_TOL * unit):
                problems.append(f"{label} {name} curve is not the normalized curve times the chord {unit!r}")
            problems += _check_leading_edge(cascade, curves, coordinates, name, label)
    return problems


def _check_leading_edge(cascade: Cascade, curves: SurfaceCurves, coordinates: CascadeCoordinates, name: str,
                        label: str) -> list[str]:
    """The C2 LE of one surface: curvature -+1/R_LE at the apex (suction negative), and over its
    LE zone (u up to the anchor, from the metadata) a curvature that keeps that sign, so the
    tangent turns monotonically: no inflection, no S-bend."""
    anchor = cascade.metadata.get(f"{'ss' if name == 'suction' else 'ps'}_le_anchor_u")
    if anchor is None:
        return [f"C2 metadata lacks the {name} LE anchor"]
    curve = getattr(curves, f"{name}_unstaggered")
    sense = 1.0 if name == "pressure" else -1.0
    radius = float(coordinates.leading_edge_radius)
    problems = []
    if abs(float(np.ravel(curve.curvature(0.0))[0]) * radius - sense) > 1e-9:
        problems.append(f"{label} {name} curve does not have curvature 1/R_LE at the LE")
    kappa = curve.curvature(np.linspace(0.0, float(anchor), 801))
    if not np.all(sense * kappa > 0.0):
        problems.append(f"{label} {name} LE fair changes the sign of its curvature (an inflection)")
    return problems
