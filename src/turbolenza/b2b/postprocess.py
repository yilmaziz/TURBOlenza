"""
postprocess.py
==============
Results derived from a finished Solution, as arrays and numbers ready to plot or tabulate.
Nothing here solves, recomputes the flow or feeds back into a solve; the plots are in plots.py.

Blade surfaces (blade_surfaces; surface_mach for the (m, eta = 0, eta = 1) form)
    Only the blade run, stations first_blade_i .. last_blade_i: no inlet or exit periodic
    station. Which wall is which surface comes from grid.wall_identity, never from j = 0
    versus j = n_eta - 1: every wall node of a wall must lie on one named surface of
    geometry.py, and the two walls on the suction and the pressure surface; a wall whose
    named surface changes along the blade (a nose or tail stretch of an inadmissible grid,
    grid.py sections 3 and 8) raises PostprocessError rather than giving mislabeled curves.
    The Mach number is solution.mach (W / a of the reported evaluation) at the wall nodes,
    and the static pressure solution.static_state.pressure there: nothing is recomputed.

Chordwise coordinate
    x/c is the blade-frame x of geometry.py, the percent-chord station of the NACA tables, of
    Dunavant et al. (1955, "Percent chord") and of Aungier (2003) Fig. 5-6 ("x / c"): the
    blade frame has the LE point at (0, 0) and the TE point at (1, 0) and is turned by the
    stagger about the LE, so for a point P of the reference blade
        x/c = (P - P_LE) . (P_TE - P_LE) / |P_TE - P_LE|^2.
    The upper wall's nodes lie on the +pitch neighbor; they are moved back by
    periodic_translation first. m/c = (m - m_le) / chord, the axial distance from the LE
    point in true chords, is returned beside it and is never labeled x/c: under stagger the
    two differ (by cos(stagger) and the camber).

Pressure coefficient (pressure_coefficient, surface_pressure_coefficient)
    Two conventions, both on the specified inlet state solution.inlet (the InletState of the
    inlet Mach number: Pt1, p1, rho1, W1), never on the computed far-inlet station:
        DUNAVANT (default)  S = (Pt1 - p) / q1,   q1 = rho1 W1^2 / 2
                            Dunavant et al. 1955 (NACA RM L55I08), list of symbols: "S
                            pressure coefficient, (P1 - p_l) / q1", P total pressure, p static
                            pressure, q dynamic pressure 1/2 rho V^2, subscript 1 upstream
                            undisturbed stream, l local on the blade surface; their Fig. 10
                            plots S against percent chord, upper surface above. Aungier (2003,
                            p. 91) converted these S data to Mach numbers for Fig. 5-6.
        INLET_STATIC        Cp = (p - p1) / q1, the static-reference form.
    They are related by S = (Pt1 - p1) / q1 - Cp. A solution without an inlet state (mdot'
    given without an inlet Mach number) or with q1 <= 0 has no pressure coefficient:
    PostprocessError.

Discharge angle (discharge_angle_deg, exit_velocity_angle_deg)
    The reported angle is the grid's: degrees(solution.grid.exit_angle), the Kutta loop's
    result (solve) or the fixed angle (solve_at_exit_angle). A check, never a replacement:
    the angle of the mean far-exit velocity, arctan(int W_theta d(eta) / int W_m d(eta)) at the
    far exit station (trapezoidal rule), which equals the boundary angle up to the
    finite-difference slope of the exit line (metrics.py).

Summary (summarize, format_summary): the inviscid numbers of a solution; no loss or boundary
layer quantity. Experimental surface data (read_surface_points): a CSV with the columns
surface (suction / pressure), x_c and the quantity (default mach), as
naca65/data/cases/fig56_exp.csv lays out; nothing is digitized in this package.

Status: implemented 2026-11-13, under review.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from turbolenza.b2b.solver import Solution

DUNAVANT = "dunavant"           # S = (Pt1 - p) / q1
INLET_STATIC = "inlet_static"   # Cp = (p - p1) / q1
CONVENTIONS = (DUNAVANT, INLET_STATIC)
LABELS = {DUNAVANT: "S = (Pt1 - p) / q1", INLET_STATIC: "Cp = (p - p1) / q1"}

Field = NDArray[np.float64]


class PostprocessError(ValueError):
    """The solution cannot give the asked quantity: wall identities that change along the
    blade, no inlet state or q1 <= 0 for a pressure coefficient, an unknown convention."""


@dataclass(frozen=True)
class BladeSurfaces:
    """The two blade walls over the blade stations (module docstring), each array (n_blade,)."""

    stations: NDArray[np.int64]     # first_blade_i .. last_blade_i
    m: Field                        # grid.m there, the grid's length unit
    m_c: Field                      # (m - m_le) / chord
    ss_j: int                       # eta index of the suction-surface wall
    ps_j: int                       # eta index of the pressure-surface wall
    x_c_ss: Field                   # chordwise x/c of the suction-surface nodes
    x_c_ps: Field                   # of the pressure-surface nodes (moved back onto the reference blade)
    mach_ss: Field                  # solution.mach at the suction-surface nodes
    mach_ps: Field
    pressure_ss: Field              # solution.static_state.pressure there
    pressure_ps: Field


@dataclass(frozen=True)
class SurfaceCoefficient:
    """A pressure coefficient on both surfaces against x/c (module docstring)."""

    convention: str
    label: str
    x_c_ss: Field
    values_ss: Field
    x_c_ps: Field
    values_ps: Field


@dataclass(frozen=True)
class SurfacePoints:
    """Measured surface values read from a CSV (read_surface_points); empty arrays when the
    file holds no points."""

    source: str
    quantity: str
    x_c_ss: Field
    values_ss: Field
    x_c_ps: Field
    values_ps: Field

    @property
    def empty(self) -> bool:
        return self.x_c_ss.size + self.x_c_ps.size == 0


@dataclass(frozen=True)
class SolutionSummary:
    """The inviscid result in numbers (summarize)."""

    limitation: str
    converged: bool
    iterations: int
    mass_flow_per_depth: float
    mass_flow_source: str
    inlet_mach: float | None
    inlet_angle_deg: float
    discharge_angle_deg: float
    exit_velocity_angle_deg: float
    exit_angle_mismatch_deg: float          # exit_velocity_angle_deg - discharge_angle_deg
    turning_deg: float                      # inlet_angle_deg - discharge_angle_deg
    surface_mach_max: float
    surface_mach_min: float
    ss_peak_mach: float
    ss_peak_x_c: float
    ps_peak_mach: float
    ps_peak_x_c: float
    kutta_residual: float
    normalized_kutta_residual: float
    mach_max: float                         # over every node
    mach_max_node: tuple[int, int]
    density_min: float
    density_max: float
    pressure_min: float
    pressure_max: float
    temperature_min: float
    temperature_max: float
    mass_flow_error_max: float
    mass_flow_error_blade_median: float
    linear_residual: float
    density_change: float                   # of the last recorded pass
    velocity_change: float


def blade_surfaces(solution: Solution) -> BladeSurfaces:
    """The suction and pressure wall over the blade stations (module docstring)."""
    _check(solution)
    grid = solution.grid
    stations = np.arange(grid.first_blade_i, grid.last_blade_i + 1)
    identity = grid.wall_identity
    lower, upper = _one_surface(identity.lower.surface, "lower (eta = 0)"), _one_surface(identity.upper.surface, "upper (eta = 1)")
    last = len(grid.eta) - 1
    if (lower, upper) == ("suction", "pressure"):
        ss_j, ps_j = 0, last
    elif (lower, upper) == ("pressure", "suction"):
        ss_j, ps_j = last, 0
    else:
        raise PostprocessError(f"the walls lie on the {lower} and the {upper} surface: not one suction and one pressure "
                               f"surface")
    boundaries = grid.boundaries
    le, te = np.asarray(boundaries.lower_blade_boundary[0]), np.asarray(boundaries.lower_blade_boundary[-1])
    shift = np.asarray(boundaries.periodic_translation)

    def x_c(j: int) -> Field:
        points = np.asarray(grid.nodes[stations, j])
        if j == last:
            points = points - shift                     # the upper wall lies on the +pitch neighbor
        return chordwise_coordinate(points, le, te)

    mach, pressure = np.asarray(solution.mach), np.asarray(solution.static_state.pressure)
    return BladeSurfaces(stations=stations, m=np.asarray(grid.m[stations]),
                         m_c=(np.asarray(grid.m[stations]) - grid.m_le) / grid.chord, ss_j=ss_j, ps_j=ps_j,
                         x_c_ss=x_c(ss_j), x_c_ps=x_c(ps_j), mach_ss=mach[stations, ss_j], mach_ps=mach[stations, ps_j],
                         pressure_ss=pressure[stations, ss_j], pressure_ps=pressure[stations, ps_j])


def chordwise_coordinate(points, leading_edge, trailing_edge) -> Field:
    """x/c = (P - P_LE) . (P_TE - P_LE) / |P_TE - P_LE|^2 for points (n, 2) of the reference
    blade in the cascade frame (module docstring)."""
    d = np.asarray(trailing_edge, dtype=float) - np.asarray(leading_edge, dtype=float)
    return (np.asarray(points, dtype=float) - np.asarray(leading_edge, dtype=float)) @ d / float(d @ d)


def surface_mach(solution: Solution) -> tuple[Field, Field, Field]:
    """m of the blade stations, and the Mach number on the eta = 0 and eta = 1 walls there
    (solution.mach). blade_surfaces names them by surface and gives x/c."""
    _check(solution)
    grid = solution.grid
    blade = slice(grid.first_blade_i, grid.last_blade_i + 1)
    mach = np.asarray(solution.mach)
    return np.asarray(grid.m[blade]), mach[blade, 0], mach[blade, -1]


def pressure_coefficient(solution: Solution, convention: str = DUNAVANT) -> tuple[Field, Field, Field]:
    """m of the blade stations, and the pressure coefficient of convention (DUNAVANT: S =
    (Pt1 - p) / q1; INLET_STATIC: (p - p1) / q1) on the eta = 0 and eta = 1 walls there."""
    _check(solution)
    grid = solution.grid
    blade = slice(grid.first_blade_i, grid.last_blade_i + 1)
    pressure = np.asarray(solution.static_state.pressure)
    return (np.asarray(grid.m[blade]), _coefficient(pressure[blade, 0], solution, convention),
            _coefficient(pressure[blade, -1], solution, convention))


def surface_pressure_coefficient(solution: Solution, convention: str = DUNAVANT) -> SurfaceCoefficient:
    """The pressure coefficient of convention on the suction and pressure surface against x/c."""
    s = blade_surfaces(solution)
    return SurfaceCoefficient(convention=convention, label=LABELS[convention], x_c_ss=s.x_c_ss,
                              values_ss=_coefficient(s.pressure_ss, solution, convention), x_c_ps=s.x_c_ps,
                              values_ps=_coefficient(s.pressure_ps, solution, convention))


def discharge_angle_deg(solution: Solution) -> float:
    """degrees(solution.grid.exit_angle): the discharge angle the solution was made with (the
    Kutta loop's in solve), from the meridional direction (module docstring)."""
    _check(solution)
    return math.degrees(solution.grid.exit_angle)


def exit_velocity_angle_deg(solution: Solution) -> float:
    """The angle of the mean velocity at the far exit station, degrees (module docstring): a
    check of discharge_angle_deg, not a replacement."""
    _check(solution)
    eta = np.asarray(solution.grid.eta)
    W_m = float(np.trapezoid(np.asarray(solution.flow.W_m)[-1], eta))
    W_theta = float(np.trapezoid(np.asarray(solution.flow.W_theta)[-1], eta))
    return math.degrees(math.atan2(W_theta, W_m))


def summarize(solution: Solution) -> SolutionSummary:
    """The numbers of SolutionSummary from solution (module docstring)."""
    s = blade_surfaces(solution)
    grid, state, mach = solution.grid, solution.static_state, np.asarray(solution.mach)
    beta_out, beta_velocity = discharge_angle_deg(solution), exit_velocity_angle_deg(solution)
    beta_in = math.degrees(grid.inlet_angle)
    surface = np.concatenate((s.mach_ss, s.mach_ps))
    i, j = np.unravel_index(int(np.argmax(mach)), mach.shape)
    last = solution.passes[-1]
    return SolutionSummary(
        limitation=solution.limitation, converged=solution.converged, iterations=solution.iterations,
        mass_flow_per_depth=solution.mass_flow_per_depth, mass_flow_source=solution.mass_flow_source,
        inlet_mach=None if solution.inlet is None else solution.inlet.mach, inlet_angle_deg=beta_in,
        discharge_angle_deg=beta_out, exit_velocity_angle_deg=beta_velocity,
        exit_angle_mismatch_deg=beta_velocity - beta_out, turning_deg=beta_in - beta_out,
        surface_mach_max=float(np.max(surface)), surface_mach_min=float(np.min(surface)),
        ss_peak_mach=float(np.max(s.mach_ss)), ss_peak_x_c=float(s.x_c_ss[int(np.argmax(s.mach_ss))]),
        ps_peak_mach=float(np.max(s.mach_ps)), ps_peak_x_c=float(s.x_c_ps[int(np.argmax(s.mach_ps))]),
        kutta_residual=solution.kutta_residual, normalized_kutta_residual=solution.normalized_kutta_residual,
        mach_max=float(mach[i, j]), mach_max_node=(int(i), int(j)),
        density_min=float(np.min(solution.density)), density_max=float(np.max(solution.density)),
        pressure_min=float(np.min(state.pressure)), pressure_max=float(np.max(state.pressure)),
        temperature_min=float(np.min(state.temperature)), temperature_max=float(np.max(state.temperature)),
        mass_flow_error_max=last.mass_flow_error_max, mass_flow_error_blade_median=last.mass_flow_error_blade_median,
        linear_residual=last.linear_residual, density_change=last.density_change, velocity_change=last.velocity_change)


def format_summary(summary: SolutionSummary) -> str:
    """A compact text block of summary (inviscid quantities only)."""
    state = "converged" if summary.converged else "NOT CONVERGED"
    inlet = "-" if summary.inlet_mach is None else f"{summary.inlet_mach:.4f}"
    lines = [
        f"{state} after {summary.iterations} passes; {summary.limitation}",
        f"inlet: M1 {inlet}, beta1 {summary.inlet_angle_deg:.4f} deg, mdot' {summary.mass_flow_per_depth:.6g} "
        f"({summary.mass_flow_source})",
        f"beta_out {summary.discharge_angle_deg:.6f} deg (far-exit velocity {summary.exit_velocity_angle_deg:.6f} deg, "
        f"mismatch {summary.exit_angle_mismatch_deg:+.2e} deg); turning {summary.turning_deg:.4f} deg",
        f"surface Mach {summary.surface_mach_min:.4f} .. {summary.surface_mach_max:.4f}; SS peak {summary.ss_peak_mach:.4f} at "
        f"x/c {summary.ss_peak_x_c:.4f}; PS peak {summary.ps_peak_mach:.4f} at x/c {summary.ps_peak_x_c:.4f}",
        f"Kutta: R_K {summary.kutta_residual:+.3e}, eps_K {summary.normalized_kutta_residual:.3e}",
        f"field: max M_rel {summary.mach_max:.4f} at node {summary.mach_max_node}; rho {summary.density_min:.6g} .. "
        f"{summary.density_max:.6g}; p {summary.pressure_min:.6g} .. {summary.pressure_max:.6g}; T "
        f"{summary.temperature_min:.6g} .. {summary.temperature_max:.6g}",
        f"mass flow: station error max {summary.mass_flow_error_max:.2e}, blade median "
        f"{summary.mass_flow_error_blade_median:.2e}; linear residual {summary.linear_residual:.2e}; last changes rho "
        f"{summary.density_change:.2e}, W {summary.velocity_change:.2e}",
    ]
    return "\n".join(lines)


def read_surface_points(path, quantity: str = "mach") -> SurfacePoints:
    """Measured points from a CSV with the columns surface (suction / pressure), x_c and
    quantity (module docstring). A file with the header only gives empty arrays."""
    path = Path(path)
    data = {"suction": ([], []), "pressure": ([], [])}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = {"surface", "x_c", quantity} - set(reader.fieldnames or ())
        if missing:
            raise PostprocessError(f"{path} lacks the column(s) {sorted(missing)}")
        for row in reader:
            surface = row["surface"].strip().lower()
            if surface not in data:
                raise PostprocessError(f"{path}: surface {row['surface']!r} is not suction or pressure")
            data[surface][0].append(float(row["x_c"]))
            data[surface][1].append(float(row[quantity]))
    return SurfacePoints(source=str(path), quantity=quantity, x_c_ss=np.array(data["suction"][0], dtype=float),
                         values_ss=np.array(data["suction"][1], dtype=float),
                         x_c_ps=np.array(data["pressure"][0], dtype=float),
                         values_ps=np.array(data["pressure"][1], dtype=float))


# --- helpers ---------------------------------------------------------------------------------------------------

def _check(solution) -> None:
    if not isinstance(solution, Solution):
        raise TypeError(f"postprocess takes a solver.Solution, got {type(solution).__name__}")


def _one_surface(names, wall: str) -> str:
    found = sorted({str(name) for name in np.asarray(names)})
    if len(found) != 1:
        raise PostprocessError(f"the {wall} wall lies on {' and '.join(found)} along the blade: its surface changes, so "
                               f"it cannot be labeled one side (an inadmissible grid; grid.py sections 3 and 8)")
    return found[0]


def _coefficient(pressure: Field, solution: Solution, convention: str) -> Field:
    if convention not in CONVENTIONS:
        raise PostprocessError(f"unknown pressure-coefficient convention {convention!r}; expected one of {CONVENTIONS}")
    inlet = solution.inlet
    if inlet is None:
        raise PostprocessError("the solution has no inlet state (mdot' given without an inlet Mach number): no reference "
                               "for a pressure coefficient")
    q1 = inlet.dynamic_pressure
    if not (math.isfinite(q1) and q1 > 0.0):
        raise PostprocessError(f"the inlet dynamic pressure is {q1!r}: a pressure coefficient needs q1 > 0")
    p = np.asarray(pressure, dtype=float)
    if convention == DUNAVANT:
        return (inlet.total_pressure - p) / q1
    return (p - inlet.static_pressure) / q1
