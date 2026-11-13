"""
Tests for potential_flow_solver.monitor: the live convergence monitor and the console progress
lines, observers of a solve that never feed back. Matplotlib runs on the Agg backend: no window
opens, the figure is drawn all the same.

What is checked
- Off by default: no figure, and importing or running the solver without live_monitor does not
  import Matplotlib at all (a fresh interpreter).
- One figure per solve: a 138-pass coupled solve with live_monitor=True makes exactly one
  figure (pyplot.subplots called once); the monitor's figure, axes and lines are made once and
  updated in place; every line holds one point per pass record, its last values those of the
  last record; redraws are throttled (min_interval) and always made at start and finish.
- Tolerance lines are the IterationSettings' (density, a separate velocity line when the
  velocity tolerance differs, Kutta in the coupled loop only).
- Observation only: the same solve with and without the monitor, and with an observer that
  raises (switched off with a MonitorWarning), gives identical arrays and records.
- A transonic stop still gets its final update: the stopping pass is plotted and the status
  names the error.
- ConsoleProgress (verbose=True): a header, one line per pass in the documented format, one
  closing line; nothing printed by default.
- close() releases the figure.
"""

from __future__ import annotations

import math
import re
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from potential_flow_solver import config as cf  # noqa: E402
from potential_flow_solver import eos  # noqa: E402
from potential_flow_solver import monitor as mon  # noqa: E402
from potential_flow_solver import solver as sv  # noqa: E402
from potential_flow_solver.geometry import CascadeSpec, make_cascade  # noqa: E402

AIR = eos.CaloricallyPerfectGas(gamma=1.4, gas_constant=287.05)
C_REF = 0.1
ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=None)
def cascade():
    return make_cascade(CascadeSpec(blade_type="NACA65", solidity=1.0, stagger_deg=33.0, cl0=1.2, t_max_over_c=0.10,
                                    surface_continuity="C2"))


def config(tolerance=1e-8, kutta_tolerance=None, passes=400, mach=0.3, velocity_tolerance=None):
    return cf.SolverConfig(
        grid=cf.GridSettings(n_passage=43, n_eta=21),
        flow=cf.FlowConditions(inlet_mach=mach, inlet_angle_deg=45.0, exit_angle_guess_deg=math.degrees(cascade().kappa_te),
                               inlet_absolute_total_pressure=101325.0, inlet_absolute_total_temperature=288.15,
                               angular_speed=0.0),
        gas=AIR, iteration=cf.IterationSettings(density_tolerance=tolerance, max_density_iterations=passes,
                                                kutta_tolerance=tolerance if kutta_tolerance is None else kutta_tolerance,
                                                max_kutta_iterations=passes, velocity_tolerance=velocity_tolerance))


@lru_cache(maxsize=None)
def plain_coupled(tolerance):
    return sv.solve(cascade(), config(tolerance), c_ref=C_REF)


def same(a, b) -> bool:
    """Two solutions identical in every array and record (NaN equal to NaN)."""
    for name in ("density", "density_target", "psi", "mach"):
        if not np.array_equal(getattr(a, name), getattr(b, name)):
            return False
    if not (np.array_equal(a.flow.W, b.flow.W) and a.exit_angle == b.exit_angle and a.iterations == b.iterations
            and len(a.passes) == len(b.passes)):
        return False
    for x, y in zip(a.passes, b.passes):
        for name, value in x.__dict__.items():
            other = getattr(y, name)
            if not (value == other or (isinstance(value, float) and math.isnan(value) and math.isnan(other))):
                return False
    return True


@pytest.fixture
def figures():
    before = set(plt.get_fignums())
    yield before
    for number in set(plt.get_fignums()) - before:
        plt.close(number)


def test_the_solver_does_not_import_matplotlib_unless_asked():
    code = ("import sys; sys.path.insert(0, r'%s'); import potential_flow_solver.solver, potential_flow_solver.monitor; "
            "print('matplotlib' in sys.modules)" % ROOT)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
    assert out == "False"


def test_no_figure_without_a_monitor(figures):
    sv.solve_density_coupled(cascade(), config(), c_ref=C_REF)
    assert set(plt.get_fignums()) == figures


def test_one_window_for_a_135_pass_coupled_solve(figures, monkeypatch):
    calls = []
    original = plt.subplots

    def counting(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(plt, "subplots", counting)
    sol = sv.solve(cascade(), config(), c_ref=C_REF, live_monitor=True)
    assert sol.converged and sol.iterations == 138                       # the verification case, 138 + final
                                                                         # (135 before the region-wise psi_m)
    assert len(calls) == 1 and len(set(plt.get_fignums()) - figures) == 1
    assert same(sol, plain_coupled(1e-8))


def test_the_figure_is_made_once_and_updated_in_place(figures):
    class Watching(mon.LiveConvergenceMonitor):
        def start(self, info):
            super().start(info)
            self.made = (self.figure, dict(self.lines), {k: len(ax.get_lines()) for k, ax in self.axes.items()})

    m = Watching(min_interval=0.5)
    sol = sv.solve(cascade(), config(1e-5), c_ref=C_REF, monitor=m)
    figure, lines, counts = m.made
    assert m.figure is figure and m.lines == lines
    assert {k: len(ax.get_lines()) for k, ax in m.axes.items()} == counts      # no line added after start
    assert len(set(plt.get_fignums()) - figures) == 1
    n = len(sol.passes)
    assert m.records == list(sol.passes) or all(a is b for a, b in zip(m.records, sol.passes))
    for name, field_name in (("density", "density_change"), ("velocity", "velocity_change"),
                             ("kutta", "kutta_normalized"), ("psi", "psi_change"), ("linear", "linear_residual"),
                             ("mach", "mach_max")):
        x, y = m.lines[name].get_data()
        assert len(x) == len(y) == n
        assert y[-1] == getattr(sol.passes[-1], field_name)
    assert m.lines["angle"].get_ydata()[-1] == math.degrees(sol.passes[-1].exit_angle)
    assert list(m.lines["density"].get_xdata()) == [p.index for p in sol.passes]
    assert 2 <= m.draws < n and m.status.startswith("converged")          # throttled, start and finish forced
    assert same(sol, plain_coupled(1e-5))


def test_tolerance_lines_come_from_the_iteration_settings(figures):
    m = mon.LiveConvergenceMonitor()
    settings = config(1e-6, kutta_tolerance=1e-4, velocity_tolerance=1e-5, passes=3).iteration
    with pytest.raises(sv.CoupledNotConvergedError):
        sv.solve(cascade(), config(1e-6, kutta_tolerance=1e-4, velocity_tolerance=1e-5, passes=3), c_ref=C_REF, monitor=m)
    assert m.info.density_tolerance == settings.density_tolerance
    assert m.info.velocity_tolerance == settings.velocity_tolerance and m.info.kutta_tolerance == settings.kutta_tolerance
    assert {k: set(line.get_ydata()) for k, line in m.tolerance_lines.items()} == \
        {"density": {1e-6}, "velocity": {1e-5}, "kutta": {1e-4}}
    fixed = mon.LiveConvergenceMonitor()
    sv.solve_density_coupled(cascade(), config(1e-6), c_ref=C_REF, monitor=fixed)
    assert set(fixed.tolerance_lines) == {"density"} and fixed.info.kutta_tolerance is None


def test_a_transonic_stop_gets_its_final_update(figures):
    m = mon.LiveConvergenceMonitor()
    with pytest.raises(sv.TransonicFlowError) as info:
        sv.solve(cascade(), config(mach=0.7), c_ref=C_REF, monitor=m)
    partial = info.value.solution
    assert m.status == "stopped: TransonicFlowError"
    assert len(m.lines["mach"].get_xdata()) == len(partial.passes) and m.lines["mach"].get_ydata()[-1] >= 1.0
    assert "TransonicFlowError" in m.figure._suptitle.get_text()


def test_an_observer_that_raises_is_switched_off(figures):
    class Broken:
        def start(self, info):
            pass

        def update(self, record):
            raise RuntimeError("plotting went wrong")

        def finish(self, solution, error):
            pass

    with pytest.warns(mon.MonitorWarning, match="switched off"):
        sol = sv.solve_density_coupled(cascade(), config(), c_ref=C_REF, monitor=Broken())
    assert same(sol, sv.solve_density_coupled(cascade(), config(), c_ref=C_REF))


def test_console_progress(capsys):
    sol = sv.solve_density_coupled(cascade(), config(), c_ref=C_REF)
    assert capsys.readouterr().out == ""                                   # nothing by default
    sol = sv.solve_density_coupled(cascade(), config(), c_ref=C_REF, verbose=True)
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == len(sol.passes) + 2
    assert lines[0].startswith("fixed angle solve:") and lines[-1].startswith("converged in 27 passes")
    pattern = re.compile(r"^iter( +\d+| final) \| rho \S+ \| W \S+ \| Kutta \S+ \| beta2 \d+\.\d{4} deg \| Mmax \d\.\d{3} "
                         r"\| lin \S+$")
    assert all(pattern.match(line) for line in lines[1:-1])
    assert lines[-2].startswith("iter final") and mon.format_record(sol.passes[-1]) == lines[-2]


def test_inputs_and_close(figures):
    with pytest.raises(sv.SolverInputError, match="live_monitor"):
        sv.solve_density_coupled(cascade(), config(), c_ref=C_REF, live_monitor="yes")
    m = mon.LiveConvergenceMonitor()
    sv.solve_density_coupled(cascade(), config(1e-4), c_ref=C_REF, monitor=m)
    number = m.figure.number
    assert number in plt.get_fignums()
    m.close()
    assert number not in plt.get_fignums()
