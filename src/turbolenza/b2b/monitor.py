"""
monitor.py
==========
Watching a solve while it runs: observers that the solver hands every pass record to. They
observe only. The solver gives them frozen numbers (PassRecord) and the finished Solution,
reads nothing back, and the same solve with or without them is the same solution to the
last bit (tests/test_monitor.py).

    solver physics  ──►  PassRecord (one per pass, the authoritative per-pass numbers)
                              │
                              ├──► ConsoleProgress           one line per pass (verbose=True)
                              └──► LiveConvergenceMonitor    one Matplotlib window per solve
                                                             (live_monitor=True)

The protocol (duck-typed; any object with these three methods can be passed as monitor=):
    start(info: SolveInfo)               once, after the inputs are checked, before pass 0
    update(record: PassRecord)           after every pass, the final evaluation included,
                                         and the pass that stopped the loop
    finish(solution, error)              once: the Solution (or the partial one an error
                                         carries, None if there is none) and the exception, or
                                         None after a converged solve
An observer that raises is switched off with a MonitorWarning; the solve goes on unchanged.

Matplotlib is not a dependency of the numerical package: this module imports it only when a
LiveConvergenceMonitor starts, and the solver imports nothing of it unless live_monitor=True.
The window: one figure per solve, created in start and never again; its axes and Line2D
objects are made once and updated with set_data after every pass. flush_events runs after
every pass, so the window stays responsive; the rescale and draw_idle run at most every
min_interval seconds (default 0.5 s) and always at start and finish (a redraw of the four
axes costs about 0.07 s with the fixed layout used here, 0.15 s with constrained layout:
drawing every pass would have more than doubled a 135-pass solve). No show() blocks here: an
interactive script calls matplotlib.pyplot.show() at its end to keep the window open. On a
non-interactive backend (Agg, the tests) the same figure is drawn without a window.

    axes                  curves (PassRecord fields)                       reference lines
    changes per pass      eps_rho density_change, eps_W velocity_change,  density, velocity and
                          eps_K kutta_normalized, dpsi psi_change (log)    Kutta tolerances of
                                                                           IterationSettings
    linear solve          ||F Psi - Q||_inf linear_residual (log)
    discharge angle       beta_out exit_angle, degrees
    peak Mach number      mach_max                                         M = 1

Status: implemented 2026-11-13, under review.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass

DENSITY_COLOR, VELOCITY_COLOR, KUTTA_COLOR, PSI_COLOR = "#1f77b4", "#ff7f0e", "#2ca02c", "#7f7f7f"


class MonitorWarning(RuntimeWarning):
    """An observer raised and was switched off; the solve went on without it."""


@dataclass(frozen=True)
class SolveInfo:
    """What an observer is told when a solve starts (numbers from IterationSettings and the
    grid; nothing an observer may change)."""

    mode: str                           # "coupled" (density + Kutta) or "fixed angle"
    density_tolerance: float | None     # None: not known (plots.plot_convergence without settings)
    velocity_tolerance: float | None
    kutta_tolerance: float | None       # None at a fixed angle (no Kutta stop test)
    max_passes: int
    exit_angle_start: float             # radians
    inlet_angle: float                  # radians
    mass_flow_per_depth: float
    mass_flow_source: str
    n_m: int
    n_eta: int


class ConsoleProgress:
    """One line per pass on stream (default sys.stdout), e.g.
        iter   34 | rho 2.1e-05 | W 3.0e-05 | Kutta 8.4e-04 | beta2 23.2100 deg | Mmax 0.401 | lin 6.0e-16
    the final evaluation marked "final"; a header at start and one closing line."""

    def __init__(self, stream=None) -> None:
        self.stream = stream

    def _print(self, text: str) -> None:
        print(text, file=self.stream if self.stream is not None else sys.stdout, flush=True)

    def start(self, info: SolveInfo) -> None:
        def tol(value) -> str:
            return "-" if value is None else f"{value:.1e}"

        self._print(f"{info.mode} solve: {info.n_m} x {info.n_eta} nodes, at most {info.max_passes} passes; tolerances "
                    f"rho {tol(info.density_tolerance)}, W {tol(info.velocity_tolerance)}, Kutta "
                    f"{tol(info.kutta_tolerance)}; mdot' "
                    f"{info.mass_flow_per_depth:.6g} ({info.mass_flow_source}); first beta2 "
                    f"{math.degrees(info.exit_angle_start):.4f} deg")

    def update(self, record) -> None:
        self._print(format_record(record))

    def finish(self, solution, error) -> None:
        if error is None:
            self._print(f"converged in {solution.iterations} passes: beta2 {math.degrees(solution.exit_angle):.6f} deg, "
                        f"eps_K {solution.normalized_kutta_residual:.2e}")
        else:
            self._print(f"stopped: {type(error).__name__}: {error}")


def format_record(record) -> str:
    """The console line of one PassRecord (ConsoleProgress)."""
    label = " final" if record.final else f"{record.index:6d}"
    return (f"iter{label} | rho {record.density_change:.1e} | W {record.velocity_change:.1e} | Kutta "
            f"{record.kutta_normalized:.1e} | beta2 {math.degrees(record.exit_angle):.4f} deg | Mmax "
            f"{record.mach_max:.3f} | lin {record.linear_residual:.1e}")


class LiveConvergenceMonitor:
    """The live convergence window (module docstring): one figure per solve, updated in place
    after every pass. figure, axes and lines are available after start; records holds every
    PassRecord received; status the closing text. close() closes the figure. Reusing one
    monitor object for a second solve makes a second figure (one per solve)."""

    def __init__(self, title: str | None = None, interactive: bool | None = None, min_interval: float = 0.5) -> None:
        self.title = title
        self.interactive = interactive      # None: from the backend; False: never a window (plots.plot_convergence)
        self.min_interval = float(min_interval)   # s between redraws; the line data is set after every pass
        self.draws = 0                      # redraws made (the figure itself is made once)
        self._last_draw = -math.inf
        self.figure = None
        self.axes: dict = {}
        self.lines: dict = {}
        self.records: list = []
        self.status = ""
        self.info: SolveInfo | None = None

    # -- the observer protocol ---------------------------------------------------------------------------

    def start(self, info: SolveInfo) -> None:
        import matplotlib.pyplot as plt

        self.info, self.records, self.status = info, [], "running"
        interactive = _interactive_backend() if self.interactive is None else bool(self.interactive)
        if interactive:
            plt.ion()
        fig, grid = plt.subplots(2, 2, figsize=(11.0, 7.0))
        fig.subplots_adjust(left=0.08, right=0.98, bottom=0.08, top=0.90, wspace=0.28, hspace=0.38)  # fixed: cheap redraws
        self.figure, self.draws, self._last_draw = fig, 0, -math.inf
        changes, linear, angle, mach = grid[0, 0], grid[0, 1], grid[1, 0], grid[1, 1]
        self.axes = {"changes": changes, "linear": linear, "angle": angle, "mach": mach}
        for ax in (changes, linear):
            ax.set_yscale("log")
        self.lines = {
            "density": changes.plot([], [], "-", color=DENSITY_COLOR, lw=1.6, label=r"$\epsilon_\rho$ density")[0],
            "velocity": changes.plot([], [], "-", color=VELOCITY_COLOR, lw=1.6, label=r"$\epsilon_W$ velocity")[0],
            "kutta": changes.plot([], [], "-", color=KUTTA_COLOR, lw=1.6, label=r"$\epsilon_K$ Kutta")[0],
            "psi": changes.plot([], [], "--", color=PSI_COLOR, lw=1.0, label=r"$\Delta\psi$")[0],
            "linear": linear.plot([], [], "-", color=DENSITY_COLOR, lw=1.4)[0],
            "angle": angle.plot([], [], "-", color=KUTTA_COLOR, lw=1.6)[0],
            "mach": mach.plot([], [], "-", color=VELOCITY_COLOR, lw=1.6)[0],
        }
        self.tolerance_lines = {}
        if info.density_tolerance is not None:
            self.tolerance_lines["density"] = changes.axhline(info.density_tolerance, color=DENSITY_COLOR, ls=":", lw=1.0,
                                                              label=f"$\\rho$ tol {info.density_tolerance:.0e}")
        if info.velocity_tolerance is not None and info.velocity_tolerance != info.density_tolerance:
            self.tolerance_lines["velocity"] = changes.axhline(info.velocity_tolerance, color=VELOCITY_COLOR, ls=":", lw=1.0,
                                                               label=f"W tol {info.velocity_tolerance:.0e}")
        if info.kutta_tolerance is not None:
            self.tolerance_lines["kutta"] = changes.axhline(info.kutta_tolerance, color=KUTTA_COLOR, ls=":", lw=1.0,
                                                            label=f"Kutta tol {info.kutta_tolerance:.0e}")
        mach.axhline(1.0, color="#d62728", ls=":", lw=1.0, label="M = 1")
        changes.set_title("changes per pass")
        changes.set_ylabel("relative change")
        linear.set_title("linear solve")
        linear.set_ylabel(r"$\|F\Psi - Q\|_\infty$")
        angle.set_title("discharge angle")
        angle.set_ylabel(r"$\beta_{out}$ [deg]")
        mach.set_title("peak relative Mach number")
        mach.set_ylabel(r"max $M_{rel}$")
        for ax in self.axes.values():
            ax.set_xlabel("pass")
            ax.grid(True, which="both", alpha=0.3)
        changes.legend(loc="upper right", fontsize=8, ncol=2)
        mach.legend(loc="lower right", fontsize=8)
        self._title()
        if interactive:
            plt.show(block=False)
        self._refresh(force=True)

    def update(self, record) -> None:
        self.records.append(record)
        x = [r.index for r in self.records]
        values = {
            "density": [r.density_change for r in self.records],
            "velocity": [r.velocity_change for r in self.records],
            "kutta": [r.kutta_normalized for r in self.records],
            "psi": [r.psi_change for r in self.records],
            "linear": [r.linear_residual for r in self.records],
            "angle": [math.degrees(r.exit_angle) for r in self.records],
            "mach": [r.mach_max for r in self.records],
        }
        for name, y in values.items():
            self.lines[name].set_data(x, y)
        self._refresh()

    def finish(self, solution, error) -> None:
        if error is None:
            self.status = f"converged in {solution.iterations} passes"
        else:
            self.status = f"stopped: {type(error).__name__}"
        self._refresh(force=True)

    # -- helpers ---------------------------------------------------------------------------------------------

    def close(self) -> None:
        """Close the figure (tests, batch runs)."""
        if self.figure is not None:
            import matplotlib.pyplot as plt

            plt.close(self.figure)

    def _title(self) -> None:
        if self.figure is None:
            return
        head = self.title or (f"{self.info.mode} solve" if self.info else "solve")
        last = self.records[-1] if self.records else None
        tail = "" if last is None else (f"  |  pass {last.index}, beta_out {math.degrees(last.exit_angle):.4f} deg, "
                                        f"eps_K {last.kutta_normalized:.1e}")
        self.figure.suptitle(f"{head}: {self.status}{tail}", fontsize=10)

    def _refresh(self, force: bool = False) -> None:
        """Process the window's events after every pass; rescale and redraw at most every
        min_interval seconds, and always when forced (start, finish)."""
        import time

        canvas = self.figure.canvas
        now = time.perf_counter()
        if force or now - self._last_draw >= self.min_interval:
            for ax in self.axes.values():
                ax.relim()
                ax.autoscale_view()
            self._title()
            canvas.draw_idle()
            self.draws += 1
            self._last_draw = now
        canvas.flush_events()


def _interactive_backend() -> bool:
    import matplotlib

    backend = str(matplotlib.get_backend()).lower()
    return backend not in ("agg", "pdf", "ps", "svg", "pgf", "cairo", "template") and "inline" not in backend
