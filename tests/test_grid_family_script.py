"""
Tests for scripts/analyze_grid_family.py, without a solve and without a window (Agg):

- figure_data, the only source of grid_transition_wall_nodes.png, holds exactly the generated
  grids' own numbers for every family member: grid.first_blade_i, grid.last_blade_i, f - 1,
  l + 1, their m, and the wall-node arrays of f and l (the suction node and the pressure node
  moved back one pitch, by wall_identity); not the level's nominal target spacing.
- transition_figure takes its data from figure_data (a recording stand-in sees one call and
  the figure file is written), and main draws it after every other figure.
"""

from __future__ import annotations

import importlib.util
import inspect
import math
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from potential_flow_solver import grid as gr  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def script():
    spec = importlib.util.spec_from_file_location("analyze_grid_family", ROOT / "scripts" / "analyze_grid_family.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def built(script):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c = script.cascade()
    return c, gr.build_grid_family(c, inlet_angle=script.inlet_angle())


def test_figure_data_are_the_grids_own_arrays(script, built):
    c, package = built
    data = script.figure_data(package)
    assert [d["level"] for d in data] == list(gr.GRID_LEVELS)
    for d, g in zip(data, package):
        grid = g.grid
        chord = grid.chord
        f, l = grid.first_blade_i, grid.last_blade_i
        assert (d["f"], d["l"], d["f_minus_1"], d["l_plus_1"]) == (f, l, f - 1, l + 1)
        assert d["m_f"] == grid.m[f] / chord and d["m_l"] == grid.m[l] / chord
        assert d["m_f_minus_1"] == grid.m[f - 1] / chord and d["m_l_plus_1"] == grid.m[l + 1] / chord
        shift = np.asarray(grid.boundaries.periodic_translation)
        assert set(grid.wall_identity.lower.surface) == {"suction"}            # so the lower node is the SS node
        assert np.array_equal(d["first_ss"], np.asarray(grid.nodes[f, 0]) / chord)
        assert np.array_equal(d["first_ps"], (np.asarray(grid.nodes[f, -1]) - shift) / chord)
        assert np.array_equal(d["last_ss"], np.asarray(grid.nodes[l, 0]) / chord)
        assert np.array_equal(d["last_ps"], (np.asarray(grid.nodes[l, -1]) - shift) / chord)
        assert d["dm_over_chord"] == grid.dm / chord
        nominal = (package.transitions.le_face + 0.5 * g.info.target_dm_over_chord * chord) / chord
        assert d["m_f"] != nominal                                              # the actual station, not the target's
        assert d["le_gap"] == g.info.le_gap_over_chord and d["te_gap"] == g.info.te_gap_over_chord


def test_the_transition_figure_draws_figure_data_and_comes_last(script, built, tmp_path, monkeypatch):
    c, package = built
    calls = []
    original = script.figure_data

    def recording(item):
        out = original(item)
        calls.append(out)
        return out

    monkeypatch.setattr(script, "figure_data", recording)
    path = tmp_path / "grid_transition_wall_nodes.png"
    script.transition_figure(c, package, path)
    assert path.exists() and path.stat().st_size > 10_000
    assert len(calls) == 1 and [d["f"] for d in calls[0]] == [g.grid.first_blade_i for g in package]
    source = inspect.getsource(script.main)
    last = source.index("transition_figure(")
    assert all(source.index(name) < last for name in ("beta_figure(", "surface_figure("))
    assert "grid_transition_wall_nodes.png" in source[last:last + 120]
    assert math.isfinite(package.timing["total"])
