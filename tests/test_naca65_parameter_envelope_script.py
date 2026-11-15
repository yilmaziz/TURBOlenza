"""
Tests for scripts/naca65_parameter_envelope.py (issue #7), without a window (Agg):

- A mini run (--part grids --quick: cl0 1.2 and 2.0 x t/c 0.03 and 0.10, kovach_sandercock, C2
  default spec, solidity 1, stagger 33 deg) writes into its own new folder under tmp_path, never
  into outputs/, and classifies (1.2, 0.03) as "refused" (GeometryError, "LE fair is too steep"),
  (2.0, 0.10) as "built_warning" (the GeometryWarning above CHECKED_CL0_MAX), (1.2, 0.10) as
  "built", and builds the medium grid of (1.2, 0.10); run.json carries the commit and versions.
  Expected outcomes are those geometry.py names for itself: the refusal text of fair_leading_edge
  and the warning of _check_naca65_inputs.
- A second folder with the same time stamp gets a new name (nothing is overwritten).
- The turning measure against a hand calculation: tangent angles 0, 10, 5, 15, 12, 20 deg have a
  largest reverse turn of 5 deg and a summed reverse turning of 5 + 3 = 8 deg, whatever the sense.
- The independent shape check finds zero reverse turning and no curvature sign change in the LE
  zone of both surfaces of the 65-(12A10)10 C2 blade (the robust LE's promise), while the same
  measure on the legacy C1 polyline of that blade (thickness_reconstruction quintic, so only the LE
  differs) finds the pressure-side S-bend: more than 1 deg (measured 3.81 deg). So the check can
  see a reverse turn when there is one.
- The SciPy interpolants rebuilt from geometry.base_tables() reproduce geometry's own half-
  thickness reconstructions to round-off (an independent rebuild of both).
"""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from turbolenza.b2b import geometry as geo  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def script():
    spec = importlib.util.spec_from_file_location("naca65_parameter_envelope",
                                                  ROOT / "scripts" / "naca65_parameter_envelope.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mini_run(script, tmp_path_factory):
    root = tmp_path_factory.mktemp("envelope")
    folder = script.main(["--part", "grids", "--quick", "--output-root", str(root)])
    with open(folder / "envelope_profiles.csv", newline="", encoding="utf-8") as handle:
        rows = {(float(r["cl0"]), float(r["t_max_over_c"])): r for r in csv.DictReader(handle)}
    return root, folder, rows


def test_mini_run_classifies_refusal_warning_and_built(mini_run):
    root, folder, rows = mini_run
    assert folder.parent == root and folder.is_dir()
    assert set(rows) == {(1.2, 0.03), (1.2, 0.10), (2.0, 0.03), (2.0, 0.10)}
    refused = rows[(1.2, 0.03)]
    assert refused["status"] == "refused"
    assert refused["error_class"] == "GeometryError"
    assert "LE fair is too steep" in refused["error_message"]
    assert refused["refusal_kind"] == "le_fair_too_steep"
    assert float(refused["refused_le_fairness"]) > geo.LE_FAIRNESS_LIMIT
    warned = rows[(2.0, 0.10)]
    assert warned["status"] == "built_warning"
    assert "outside the range the NACA 65 construction was checked on" in warned["geometry_warning"]
    assert warned["inside_checked_range"] == "False"
    ok = rows[(1.2, 0.10)]
    assert ok["status"] == "built" and ok["geometry_warning"] == "" and ok["error_class"] == ""
    assert ok["inside_checked_range"] == "True"
    assert float(ok["max_le_fairness"]) <= geo.LE_FAIRNESS_LIMIT
    assert ok["grid_status"] == "ok" and int(ok["grid_n_m"]) > 0 and int(ok["grid_n_eta"]) > 0


def test_mini_run_writes_its_own_folder_and_run_json(mini_run, script):
    root, folder, _ = mini_run
    assert not str(folder).startswith(str(ROOT / "outputs"))
    run = json.loads((folder / "run.json").read_text(encoding="utf-8"))
    assert run["finished"] is True and run["quick"] is True
    assert run["parts"] == ["profiles", "grids"]
    assert set(run["versions"]) >= {"python", "numpy", "scipy", "matplotlib"}
    assert "commit" in run and "dirty" in run and "command_line" in run and run["run_seconds"] > 0.0
    assert run["settings"]["KAPPA_SIGN_THRESHOLD"] == script.KAPPA_SIGN_THRESHOLD
    for name in ("envelope_profiles.csv", "summary.json", "captions.json", "log.txt",
                 "profile_status_kovach_sandercock.png", "profile_fairness_kovach_sandercock.png",
                 "grid_medium_status_kovach_sandercock.png"):
        assert (folder / name).is_file(), name
    captions = json.loads((folder / "captions.json").read_text(encoding="utf-8"))
    assert all(name in captions for name in run["files"] if name.endswith(".png"))
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    assert summary["profiles"]["kovach_sandercock"]["counts"] == {"built": 1, "built_warning": 2, "refused": 1,
                                                                  "unexpected_error": 0}


def test_run_folders_are_never_reused(script, tmp_path):
    first = script.new_run_folder(tmp_path, stamp="20000101-000000")
    second = script.new_run_folder(tmp_path, stamp="20000101-000000")
    assert first != second and first.is_dir() and second.is_dir()
    assert second.name == "20000101-000000-2"


@pytest.mark.parametrize("sense", [1.0, -1.0])
def test_turning_metrics_against_a_hand_calculation(script, sense):
    degrees = np.array([0.0, 10.0, 5.0, 15.0, 12.0, 20.0])
    vectors = np.column_stack((np.cos(np.radians(sense * degrees)), np.sin(np.radians(sense * degrees))))
    out = script.turning_metrics(script.tangent_angle(vectors, sense))
    assert out["max_reverse_turn_deg"] == pytest.approx(5.0, abs=1e-12)
    assert out["summed_reverse_turn_deg"] == pytest.approx(8.0, abs=1e-12)
    assert out["reverse_steps"] == 2
    monotone = script.turning_metrics(np.radians([0.0, 1.0, 1.0, 3.0]))
    assert monotone["max_reverse_turn_deg"] == 0.0 and monotone["summed_reverse_turn_deg"] == 0.0


def test_shape_check_sees_no_reverse_turn_on_c2_and_the_s_bend_on_c1(script):
    spec = dict(blade_type="NACA65", solidity=1.0, stagger_deg=33.0, cl0=1.2, t_max_over_c=0.10,
                profile_variant="kovach_sandercock")
    with warnings.catch_warnings():
        warnings.simplefilter("error")          # the 65-(12A10)10 lies inside the checked range
        c2 = geo.make_cascade(geo.CascadeSpec(**spec))
        c1 = geo.make_cascade(geo.CascadeSpec(**spec, surface_continuity="C1",
                                              thickness_reconstruction="quintic_natural_sqrt"))
    m = c2.metadata
    for name, prefix, sense in (("suction", "ss", -1.0), ("pressure", "ps", 1.0)):
        check = script.curve_shape_check(getattr(c2.normalized_curves, f"{name}_unstaggered"), m[f"{prefix}_le_anchor_u"])
        assert check["apex_sense"] == sense
        assert check["le_max_reverse_turn_deg"] == 0.0
        assert check["le_summed_reverse_turn_deg"] == 0.0
        assert check["le_sign_changes"] == 0 and check["le_min_sense_kappa"] > 0.0
    legacy = script.polyline_turning(c1.normalized.pressure_surface_unstaggered, 1.0, m["ps_le_anchor_over_c"])
    assert legacy["summed_reverse_turn_deg"] > 1.0 and legacy["max_reverse_turn_deg"] > 1.0
    assert math.isfinite(legacy["segments"]) and legacy["segments"] > 10


def test_rebuilt_thickness_interpolants_match_geometry(script):
    out = script.thickness_interpolants()
    for name, check in out["checks"].items():
        assert check["max_diff_vs_geometry_half_thickness"] <= script.INTERPOLANT_TOL, name
        assert check["max_error_at_table_stations"] <= 1e-16, name
