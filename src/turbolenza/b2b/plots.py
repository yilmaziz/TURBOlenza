"""
plots.py
========
Matplotlib figures of a finished Solution: the blade-surface Mach number, the surface
pressure coefficient and the convergence history. Matplotlib is imported only when one of
these is called (it is not a dependency of the numerical package); every function returns
(fig, ax) and none calls show() or writes a file: an interactive script shows them, a batch
script saves them (fig.savefig).

    plot_surface_mach          SS and PS relative Mach number against x/c on one axes, a dashed
                               M = 1 line (sonic_line), optional measured points
    plot_pressure_coefficient  SS and PS pressure coefficient against the same x/c; convention
                               DUNAVANT (S = (Pt1 - p) / q1, the source's own) by default, the
                               y axis not inverted (S rises toward the suction side, as in
                               Dunavant et al.'s figures)
    plot_convergence           the convergence history of solution.passes in the layout of the
                               live monitor (monitor.LiveConvergenceMonitor), tolerance lines when
                               the IterationSettings are given
    plot_mach_contour          the relative Mach number over the whole computational domain
                               (both extensions and the blade passage), node based: each grid
                               cell is Gouraud-shaded from its four node values (pcolormesh,
                               shading "gouraud"), so the colour varies smoothly across every
                               cell and is continuous from cell to cell; the periodic copies one
                               pitch up and down (the same node values) and the blades on top;
                               thin iso-Mach lines, the sonic line M = 1 when it exists

The coordinate and the values are postprocess.py's (blade_surfaces, surface_pressure_
coefficient): the surfaces named from grid.wall_identity, x/c along the chord. No axis
limits are imposed. A solution that did not converge is labeled NOT CONVERGED in the title.

Status: implemented 2026-11-13, under review.
"""

from __future__ import annotations

import math

from turbolenza.b2b import postprocess
from turbolenza.b2b.monitor import LiveConvergenceMonitor, SolveInfo

SUCTION_COLOR, PRESSURE_COLOR = "#1f77b4", "#d62728"
SUCTION_LABEL, PRESSURE_LABEL = "Suction side", "Pressure side"


def plot_surface_mach(solution, *, ax=None, title: str | None = None, sonic_line: bool = True,
                      experiment: postprocess.SurfacePoints | None = None):
    """The blade-surface relative Mach number on both surfaces against x/c (module docstring);
    returns (fig, ax)."""
    s = postprocess.blade_surfaces(solution)
    fig, ax = _axes(ax)
    ax.plot(s.x_c_ss, s.mach_ss, "-", color=SUCTION_COLOR, lw=1.8, label=SUCTION_LABEL)
    ax.plot(s.x_c_ps, s.mach_ps, "-", color=PRESSURE_COLOR, lw=1.8, label=PRESSURE_LABEL)
    if sonic_line:
        ax.axhline(1.0, color="#555555", ls="--", lw=0.9, label="M = 1")
    _experiment(ax, experiment)
    ax.set_xlabel("x/c")
    ax.set_ylabel("Surface relative Mach number")
    ax.set_title(title or _title("Blade-surface Mach number", solution))
    ax.grid(True, alpha=0.35)
    ax.legend(loc="best", fontsize=9)
    return fig, ax


def plot_pressure_coefficient(solution, *, convention: str = postprocess.DUNAVANT, ax=None, title: str | None = None,
                              experiment: postprocess.SurfacePoints | None = None):
    """The surface pressure coefficient of convention on both surfaces against x/c (module
    docstring); returns (fig, ax)."""
    c = postprocess.surface_pressure_coefficient(solution, convention)
    fig, ax = _axes(ax)
    ax.plot(c.x_c_ss, c.values_ss, "-", color=SUCTION_COLOR, lw=1.8, label=SUCTION_LABEL)
    ax.plot(c.x_c_ps, c.values_ps, "-", color=PRESSURE_COLOR, lw=1.8, label=PRESSURE_LABEL)
    _experiment(ax, experiment)
    ax.set_xlabel("x/c")
    ax.set_ylabel(c.label)
    ax.set_title(title or _title("Blade-surface pressure coefficient", solution))
    ax.grid(True, alpha=0.35)
    ax.legend(loc="best", fontsize=9)
    return fig, ax


def plot_convergence(solution, settings=None, *, title: str | None = None):
    """The convergence history of solution.passes, as the live monitor draws it; tolerance
    lines from settings (config.IterationSettings) when given. Returns (fig, axes), axes the
    monitor's dict of axes."""
    from turbolenza.b2b import convergence

    coupled = solution.kutta_relaxation is not None
    first = solution.passes[0]
    info = SolveInfo(mode="coupled" if coupled else "fixed angle",
                     density_tolerance=None if settings is None else float(settings.density_tolerance),
                     velocity_tolerance=None if settings is None else convergence.velocity_tolerance(settings),
                     kutta_tolerance=None if (settings is None or not coupled) else float(settings.kutta_tolerance),
                     max_passes=len(solution.passes), exit_angle_start=first.exit_angle,
                     inlet_angle=float(solution.grid.inlet_angle), mass_flow_per_depth=solution.mass_flow_per_depth,
                     mass_flow_source=solution.mass_flow_source, n_m=len(solution.grid.m), n_eta=len(solution.grid.eta))
    figure = LiveConvergenceMonitor(title=title, interactive=False)
    figure.start(info)
    for record in solution.passes:
        figure.update(record)
    figure.status = (f"converged in {solution.iterations} passes" if solution.converged
                     else f"NOT CONVERGED after {solution.iterations} passes")
    figure._refresh(force=True)
    return figure.figure, figure.axes


MACH_COLORMAP = "viridis"           # perceptually uniform, sequential: low Mach dark, high Mach light
BLADE_FILL, BLADE_EDGE = "#d9d9d9", "#333333"


def plot_mach_contour(solution, *, ax=None, title: str | None = None, copies: tuple[int, ...] = (-1, 0, 1),
                      window: tuple[float, float, float, float] | None = None, isolines: float | None = 0.05,
                      colorbar: bool = True, vmin: float | None = None, vmax: float | None = None):
    """The node-based relative Mach number over the whole computational domain (module
    docstring), in the grid's length unit (chords for the solver's cascade). copies: the
    periodic images drawn, in pitches (0 is the computational domain itself); window:
    (m_min, m_max, y_min, y_max) to zoom, None for everything drawn; isolines: the spacing
    of the thin iso-Mach lines, None for none; vmin, vmax: the colour range (default: the
    field's own). Returns (fig, ax); the shaded meshes are ax.collections[:len(copies)]."""
    import numpy as np

    grid = solution.grid
    nodes, mach = np.asarray(grid.nodes), np.asarray(solution.mach)
    shift = np.asarray(grid.boundaries.periodic_translation)
    low = float(np.min(mach)) if vmin is None else vmin
    high = float(np.max(mach)) if vmax is None else vmax
    fig, ax = _axes(ax)
    mesh = None
    for k in copies:
        x, y = nodes[..., 0] + k * shift[0], nodes[..., 1] + k * shift[1]
        mesh = ax.pcolormesh(x, y, mach, shading="gouraud", cmap=MACH_COLORMAP, vmin=low, vmax=high, zorder=1)
    if isolines:
        levels = np.arange(math.ceil(low / isolines) * isolines, high, isolines)
        for k in copies:
            x, y = nodes[..., 0] + k * shift[0], nodes[..., 1] + k * shift[1]
            if levels.size:
                ax.contour(x, y, mach, levels=levels, colors="black", linewidths=0.35, alpha=0.45, zorder=2)
            if high >= 1.0 > low:
                ax.contour(x, y, mach, levels=[1.0], colors="white", linewidths=1.4, zorder=3)
    b = grid.boundaries
    outline = np.vstack((np.asarray(b.lower_blade_boundary)[::-1], (np.asarray(b.upper_blade_boundary) - shift)[1:]))
    for k in range(min(copies), max(copies) + 2):          # copy k spans blade k (suction) to blade k + 1 (pressure)
        blade = outline + k * shift
        ax.fill(blade[:, 0], blade[:, 1], color=BLADE_FILL, edgecolor=BLADE_EDGE, linewidth=0.8, zorder=4)
    ax.set_aspect("equal")
    if window is not None:
        ax.set_xlim(window[0], window[1])
        ax.set_ylim(window[2], window[3])
    else:
        ax.autoscale_view()
    ax.set_xlabel("m (axial)")
    ax.set_ylabel("y (tangential)")
    ax.set_title(title or _title("Relative Mach number", solution))
    if colorbar and mesh is not None:
        fig.colorbar(mesh, ax=ax, label="relative Mach number (node values, smooth shading)", shrink=0.9)
    return fig, ax


def _axes(ax):
    if ax is None:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7.5, 5.0), constrained_layout=True)
        return fig, ax
    return ax.figure, ax


def _experiment(ax, experiment) -> None:
    if experiment is None or experiment.empty:
        return
    ax.plot(experiment.x_c_ss, experiment.values_ss, "o", mfc="none", color=SUCTION_COLOR, label="Experiment, suction side")
    ax.plot(experiment.x_c_ps, experiment.values_ps, "s", mfc="none", color=PRESSURE_COLOR, label="Experiment, pressure side")


def _title(what: str, solution) -> str:
    state = "" if solution.converged else "  [NOT CONVERGED]"
    return f"{what}: beta_out {math.degrees(solution.exit_angle):.3f} deg{state}"
