"""
Tests for potential_flow_solver.controller and scripts/run_full_solver.py: the user-facing run
layer (case construction, the timed solve, reports, figures, files). Small fast cases only:
the 33 deg verification cascade at a fixed exit angle (27 passes; 28 before the region-wise
psi_m of flow.py, 2026-11-13) and the M1 0.7 transonic stop
(a few passes); Matplotlib on the Agg backend.

What is checked
- The inlet Mach number, the chord and the solidity drive the derived pitch and mdot' as
  flow.inlet_state does; contradicting mass-flow inputs are refused, a pure override has no
  inlet state; an inadmissible n_passage stops before any solve.
- RuntimeStats: wall time > 0, CPU time >= 0, effective cores finite and >= 0, the logical CPU
  count os.cpu_count() (or None, reported as unknown); measure_runtime False reports "not
  measured".
- The live monitor and the console lines change nothing (identical arrays); their time is
  reported beside the solver time.
- Every surface Mach figure carries "Solver time", "Effective CPU use" and "Logical CPUs
  available" with the measured numbers; the pressure figure is Dunavant's S.
- The files: surface_mach.csv equals postprocess's values to the last bit, the convergence CSV
  has one row per pass record with that record's numbers, run_summary.json is plain JSON with
  the inputs, the derived values and the results; an existing run folder is never overwritten.
- A failed solve gives a failure report (reason, last pass, last beta_out, errors, peak Mach,
  time), a failure summary file, and raise_error re-raises.
- The script runs headless from its constants and returns 0.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import os
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from potential_flow_solver import controller as ct  # noqa: E402
from potential_flow_solver import postprocess as pp  # noqa: E402
from potential_flow_solver import solver as sv  # noqa: E402
from potential_flow_solver.flow import inlet_state  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def settings(**changes) -> ct.CaseSettings:
    base = dict(case_name="test_case", solver_mode=ct.FIXED_EXIT, fixed_exit_angle_deg=None, gamma=1.4, gas_constant=287.05,
                inlet_total_pressure=101325.0, inlet_total_temperature=288.15, omega=0.0, stream_surface_radius=None,
                inlet_absolute_tangential_velocity=None, blade_family="NACA65", design_lift_coefficient=1.2,
                thickness_ratio=0.10, meanline_type="A10", profile_variant="kovach_sandercock", surface_continuity="C2",
                n_camber_stations=101, chord=0.1, solidity=1.0, stagger_deg=33.0, inlet_mach=0.3, inlet_flow_angle_deg=45.0,
                mass_flow_per_depth_override=None, grid_level=None, n_passage=43, n_eta=21, upstream_extension_chords=1.0,
                downstream_extension_chords=1.0, density_tolerance=1e-8, velocity_tolerance=None, max_density_iterations=600,
                density_relaxation=0.5, stop_at_sonic=True, kutta_tolerance=1e-8, max_kutta_iterations=600, kutta_relaxation=1.0,
                initial_exit_angle_mode=ct.BLADE_TE_ANGLE, initial_exit_angle_deg=None, measure_runtime=True)
    base.update(changes)
    return ct.CaseSettings(**base)


@lru_cache(maxsize=None)
def fixed_run():
    return ct.run_case(ct.build_case(settings()))


@lru_cache(maxsize=None)
def failed_run():
    return ct.run_case(ct.build_case(settings(solver_mode=ct.COUPLED, inlet_mach=0.7)))


@pytest.fixture
def figures():
    before = set(plt.get_fignums())
    yield
    for number in set(plt.get_fignums()) - before:
        plt.close(number)


# --- case construction ------------------------------------------------------------------------------------------

def test_inlet_mach_chord_and_solidity_drive_the_derived_mass_flow():
    cases = {M: ct.build_case(settings(inlet_mach=M)) for M in (0.3, 0.5)}
    for M, case in cases.items():
        expected = inlet_state(case.config.flow, case.config.gas, inlet_pitch=0.1 * case.grid.pitch)
        assert case.inlet == expected and case.mass_flow_per_depth == expected.mass_flow_per_depth
        assert case.mass_flow_source == sv.MASS_FLOW_FROM_INLET
    assert cases[0.5].mass_flow_per_depth > cases[0.3].mass_flow_per_depth
    double = ct.build_case(settings(chord=0.2))
    assert double.pitch == pytest.approx(0.2, rel=1e-15)
    assert double.mass_flow_per_depth == pytest.approx(2.0 * cases[0.3].mass_flow_per_depth, rel=1e-14)
    dense = ct.build_case(settings(solidity=1.25))
    assert dense.pitch == pytest.approx(0.1 / 1.25, rel=1e-14)
    assert dense.mass_flow_per_depth == pytest.approx(cases[0.3].mass_flow_per_depth / 1.25, rel=1e-14)
    assert cases[0.3].angle_of_attack_deg == 12.0 and cases[0.3].blade_name == "NACA 65-(12A10)10"


def test_mass_flow_inputs_follow_the_production_policy():
    mdot = ct.build_case(settings()).mass_flow_per_depth
    assert ct.build_case(settings(mass_flow_per_depth_override=mdot)).mass_flow_source == sv.MASS_FLOW_CHECKED
    with pytest.raises(ct.CaseError, match="contradicts"):
        ct.build_case(settings(mass_flow_per_depth_override=1.01 * mdot))
    pure = ct.build_case(settings(inlet_mach=None, mass_flow_per_depth_override=mdot))
    assert pure.inlet is None and pure.mass_flow_source == sv.MASS_FLOW_OVERRIDE
    with pytest.raises(ct.CaseError, match="no mass flow"):
        ct.build_case(settings(inlet_mach=None))


def test_an_inadmissible_grid_stops_before_any_solve():
    with pytest.raises(ct.CaseError, match="grid cannot be used") as info:
        ct.build_case(settings(n_passage=42))
    assert "admissible" in str(info.value)
    for bad in (dict(solver_mode="kutta"), dict(initial_exit_angle_mode="guess"),
                dict(solver_mode=ct.COUPLED, initial_exit_angle_mode=ct.MANUAL, initial_exit_angle_deg=None)):
        with pytest.raises(ct.CaseError):
            ct.build_case(settings(**bad))


def test_exit_angle_modes():
    kte = ct.build_case(settings()).kappa_te_deg
    assert ct.build_case(settings()).exit_angle_deg == kte
    assert ct.build_case(settings(fixed_exit_angle_deg=20.0)).exit_angle_deg == 20.0
    manual = ct.build_case(settings(solver_mode=ct.COUPLED, initial_exit_angle_mode=ct.MANUAL, initial_exit_angle_deg=21.5))
    assert manual.exit_angle_deg == 21.5 and manual.config.flow.exit_angle_guess_deg == 21.5 and manual.pass_limit == 600


def test_case_summary_tags_user_derived_and_project_defaults():
    text = ct.format_case_summary(ct.build_case(settings()))
    for line in ("pitch [m]", "mass flow per depth", "angle of attack", "exit angle used"):
        assert any(line in row and "[derived]" in row for row in text.splitlines()), line
    assert any("density relaxation" in row and "[user = project default]" in row for row in text.splitlines())
    assert any("inlet Mach number M1" in row and "[user]" in row for row in text.splitlines())
    inlet = ct.format_inlet_state(ct.build_case(settings()))
    for word in ("T1", "p1", "rho1", "a1", "W1", "Wm1", "Wtheta1", "mdot/depth", "source"):
        assert word in inlet
    grid = ct.format_grid_summary(ct.build_case(settings()))
    for word in ("n_m", "total nodes", "first blade station", "last blade station", "dm", "deta", "admissib"):
        assert word in grid


# --- runtime ---------------------------------------------------------------------------------------------------

def test_runtime_stats_of_a_solve(monkeypatch):
    run = fixed_run()
    r = run.runtime
    assert run.converged and run.solution.iterations == 27              # 28 before the region-wise psi_m
    assert r.wall_time_s > 0.0 and r.process_cpu_time_s >= 0.0
    assert math.isfinite(r.effective_cpu_cores) and r.effective_cpu_cores >= 0.0
    assert r.effective_cpu_cores == r.process_cpu_time_s / r.wall_time_s
    assert r.logical_cpus_available == os.cpu_count() and r.observer_wall_time_s == 0.0
    monkeypatch.setattr(os, "cpu_count", lambda: None)
    unknown = ct.run_case(ct.build_case(settings(density_tolerance=1e-3)))
    assert unknown.runtime.logical_cpus_available is None
    assert "Logical CPUs available: unknown" in ct.runtime_annotation(unknown.runtime)
    off = ct.run_case(ct.build_case(settings(density_tolerance=1e-3, measure_runtime=False)))
    assert off.runtime is None and "not measured" in ct.runtime_annotation(None)


def test_monitor_and_console_change_nothing(figures, capsys):
    watched = ct.run_case(ct.build_case(settings()), live_monitor=True, verbose=True)
    plain = fixed_run()
    for name in ("density", "psi", "mach"):
        assert np.array_equal(getattr(watched.solution, name), getattr(plain.solution, name))
    assert watched.solution.iterations == plain.solution.iterations
    assert watched.monitor is not None and len(watched.monitor.records) == len(watched.solution.passes)
    assert 0.0 < watched.runtime.observer_wall_time_s < watched.runtime.wall_time_s
    assert "iter" in capsys.readouterr().out


# --- figures ----------------------------------------------------------------------------------------------------

def test_every_surface_mach_figure_shows_the_runtime_and_cpu_use(figures):
    run = fixed_run()
    figs = ct.plot_run(run)
    assert set(figs) == {"surface_mach", "surface_pressure", "convergence", "mach_contour"}
    texts = "\n".join(t.get_text() for t in figs["surface_mach"].texts)
    r = run.runtime
    assert f"Solver time: {r.wall_time_s:.3f} s" in texts
    assert f"Effective CPU use: {r.effective_cpu_cores:.2f} cores" in texts
    assert f"Logical CPUs available: {r.logical_cpus_available}" in texts
    assert "M1 = 0.300" in texts and "beta1 = 45.0 deg" in texts and "stagger = 33.0 deg" in texts
    assert "sigma = 1.00" in texts and "grid = 43 x 21" in texts and f"iterations = {run.solution.iterations}" in texts
    ax = figs["surface_mach"].axes[0]
    assert {line.get_label() for line in ax.get_lines()} >= {"Suction side", "Pressure side"}
    assert ax.get_xlabel() == "x/c"
    pressure = figs["surface_pressure"].axes[0]
    assert "Pt1 - p" in pressure.get_ylabel() and not pressure.yaxis_inverted()
    failed = ct.plot_run(failed_run())
    assert "Solver time:" in "\n".join(t.get_text() for t in failed["surface_mach"].texts)
    assert "NOT CONVERGED" in failed["surface_mach"].axes[0].get_title()
    pure = ct.run_case(ct.build_case(settings(inlet_mach=None, density_tolerance=1e-3,
                                              mass_flow_per_depth_override=run.solution.mass_flow_per_depth)))
    assert "surface_pressure" not in ct.plot_run(pure)        # no inlet state, no S


# --- files ------------------------------------------------------------------------------------------------------

def test_exported_files_hold_the_solution_numbers(tmp_path, figures):
    run = fixed_run()
    directory = ct.export_run(run, ct.plot_run(run), tmp_path, "first")
    assert directory == tmp_path / "test_case" / "first"
    names = {p.name for p in directory.iterdir()}
    assert names == {"surface_mach.png", "surface_pressure.png", "convergence.png", "mach_contour.png", "surface_mach.csv",
                     "surface_pressure.csv", "convergence_history.csv", "run_summary.json"}
    s = pp.blade_surfaces(run.solution)
    S = pp.surface_pressure_coefficient(run.solution)
    with (directory / "surface_mach.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert list(rows[0]) == list(ct.SURFACE_COLUMNS)
    ss = [r for r in rows if r["surface"] == "suction"]
    ps = [r for r in rows if r["surface"] == "pressure"]
    assert len(ss) == len(s.x_c_ss) and len(ps) == len(s.x_c_ps)
    for column, values in (("x_c", s.x_c_ss), ("m_c", s.m_c), ("mach", s.mach_ss), ("pressure", s.pressure_ss),
                           ("dunavant_S", S.values_ss)):
        assert np.array_equal([float(r[column]) for r in ss], values), column
    assert np.array_equal([float(r["mach"]) for r in ps], s.mach_ps)
    with (directory / "convergence_history.csv").open(encoding="utf-8") as handle:
        history = list(csv.DictReader(handle))
    passes = run.solution.passes
    assert len(history) == len(passes) and list(history[0]) == list(ct.CONVERGENCE_COLUMNS)
    for row, record in zip(history, passes):
        assert int(row["iteration"]) == record.index and bool(int(row["final"])) == record.final
        assert float(row["density_change"]) == record.density_change and float(row["peak_mach"]) == record.mach_max
        assert float(row["linear_residual"]) == record.linear_residual
        assert float(row["exit_angle_deg"]) == math.degrees(record.exit_angle)
    summary = json.loads((directory / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "converged" and summary["termination"]["converged"] is True
    assert summary["inputs"]["inlet_mach"] == 0.3 and summary["inputs"]["n_passage"] == 43
    assert summary["inputs"] == json.loads(json.dumps(summary["inputs"]))
    assert summary["derived"]["mass_flow_per_depth_kg_m_s"] == run.solution.mass_flow_per_depth
    assert summary["derived"]["inlet_state"]["static_pressure"] == run.solution.inlet.static_pressure
    assert summary["grid"]["n_m"] == len(run.solution.grid.m) and summary["grid"]["admissible"] is True
    assert summary["results"]["beta_out_deg"] == math.degrees(run.solution.exit_angle)
    assert summary["results"]["iterations"] == 27 and summary["results"]["surface"]["ss_peak_mach"] == np.max(s.mach_ss)
    assert summary["runtime"]["wall_time_s"] == run.runtime.wall_time_s
    assert summary["runtime"]["logical_cpus_available"] == os.cpu_count()
    assert summary["results"]["final_pass"]["velocity_change"] is not None
    assert summary["results"]["passes_recorded"] == len(passes)
    again = ct.export_run(run, {}, tmp_path, "first")              # never overwritten
    assert again == tmp_path / "test_case" / "first-2" and (directory / "surface_mach.png").exists()


# --- failure ------------------------------------------------------------------------------------------------------

def test_a_failed_solve_is_reported_saved_and_re_raisable(tmp_path, figures):
    run = failed_run()
    assert not run.converged and isinstance(run.error, sv.TransonicFlowError) and run.solution is not None
    text = ct.format_run_report(run)
    for word in ("SOLVER FAILURE", "termination reason", "TransonicFlowError", "last pass", "last beta_out",
                 "last density change", "last Kutta error", "peak Mach", "solver wall time"):
        assert word in text, word
    directory = ct.export_run(run, ct.plot_run(run), tmp_path)
    summary = json.loads((directory / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "failed: TransonicFlowError"
    assert summary["termination"]["error_type"] == "TransonicFlowError" and summary["results"]["peak_mach_all_passes"] >= 1.0
    with pytest.raises(sv.TransonicFlowError):
        run.raise_error()
    fixed_run().raise_error()                                          # nothing after a converged run


# --- the script ---------------------------------------------------------------------------------------------------

def test_the_script_runs_headless_from_its_constants(tmp_path, capsys, figures):
    spec = importlib.util.spec_from_file_location("run_full_solver", ROOT / "scripts" / "run_full_solver.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    source = (ROOT / "scripts" / "run_full_solver.py").read_text(encoding="utf-8")
    for k, title in enumerate(("RUN CONTROL", "GAS / THERMODYNAMIC INPUTS", "BLADE / CASCADE GEOMETRY",
                               "INLET FLOW CONDITIONS", "GRID SETTINGS", "NONLINEAR / DENSITY SETTINGS", "KUTTA SETTINGS",
                               "COMPUTE / RUNTIME SETTINGS", "PLOTTING / OUTPUT SETTINGS")):
        assert f"# {k}. {title}" in source
    # every input constant set here (the user edits the script's values freely; the test must not
    # depend on them), the run control headless and the output in tmp_path
    for field_name, value in settings(case_name="headless_case").__dict__.items():
        assert hasattr(script, field_name.upper()), field_name
        setattr(script, field_name.upper(), value)
    for name, value in (("SHOW_LIVE_MONITOR", False), ("SHOW_FINAL_PLOTS", False), ("VERBOSE_SOLVER", False),
                        ("SAVE_RESULTS", True), ("OUTPUT_ROOT", str(tmp_path)), ("RUN_LABEL", "headless")):
        setattr(script, name, value)
    assert script.settings_from_constants() == settings(case_name="headless_case")
    assert script.main() == 0
    out = capsys.readouterr().out
    assert "CASE INPUTS" in out and "INLET STATE" in out and "GRID" in out and "SOLVER RESULT" in out
    assert "solver wall time" in out and "effective CPU cores" in out
    assert (tmp_path / script.CASE_NAME / "headless" / "run_summary.json").exists()


def test_a_run_past_sonic_keeps_its_history(figures):
    """stop_at_sonic False: the gas model's T <= 0 error ends the run, and the run still has the
    last completed pass to report and plot (measured at M1 0.7: 172 passes with the region-wise
    psi_m of flow.py, 2026-11-13; 30 before it)."""
    case = ct.build_case(settings(inlet_mach=0.7, stop_at_sonic=False))
    assert any("stop at M_rel >= 1" in row and "OFF" in row for row in ct.format_case_summary(case).splitlines())
    with pytest.warns(sv.SonicFlowWarning):
        run = ct.run_case(case)
    assert not run.converged and type(run.error).__name__ == "ThermodynamicStateError"
    assert run.solution is not None and len(run.solution.passes) == run.error.failed_pass
    text = ct.format_run_report(run)
    assert "failed in pass" in text and "sonic stop" in text and "OFF" in text
    figs = ct.plot_run(run)
    assert "NOT CONVERGED" in figs["surface_mach"].axes[0].get_title()
    assert "Solver time:" in "\n".join(t.get_text() for t in figs["surface_mach"].texts)


def test_run_case_takes_a_start_density_and_further_observers():
    """A restart through the controller: the converged density as the start converges at once;
    an observer with inspect receives every pass; neither changes the default run."""
    class Inspector:
        def __init__(self):
            self.passes = 0

        def start(self, info):
            pass

        def update(self, record):
            pass

        def finish(self, solution, error):
            pass

        def inspect(self, fields):
            self.passes += 1

    base = fixed_run()
    inspector = Inspector()
    again = ct.run_case(ct.build_case(settings()), initial_density=np.array(base.solution.density), observers=[inspector])
    assert again.converged and again.solution.iterations <= 3 and inspector.passes == len(again.solution.passes)
    assert np.max(np.abs(again.solution.density / base.solution.density - 1.0)) < 1e-7
    watched = ct.run_case(ct.build_case(settings()), observers=[Inspector()])
    assert np.array_equal(watched.solution.density, base.solution.density)
