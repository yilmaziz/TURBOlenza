"""
Tests for the geometry-driven standard grids of grid.py (section 9, 2026-11-13): five named
levels, build_grid, build_grid_family, GridPackage, GridGenerationError.

The reference geometry is the project's NACA 65-(12A10)10 at 33 deg stagger, solidity 1 (C2,
kovach_sandercock), inlet angle 45 deg. What is checked, and against what:
- The levels: exactly five names in order; dm strictly falls (ratios measured 1.400 to
  1.429 against the target sqrt 2), deta strictly falls, pitch deta / dm stays at the
  CELL_ASPECT 1.8 within its rounding (measured 1.784 to 1.816).
- The blade ends, independently of grid.py's branch bisection: every curve sampled at 40001
  u, its crossings with each station line counted from the samples' m ranges. Every blade
  station crosses the suction curve once and the pressure curve once; f - 1 and l + 1 cross
  neither; the nose tip and tail end agree with the sampled extremes.
- The lattice: uniform dm and deta to round-off (measured 2.6e-16 .. 5.4e-16 c), the LE and
  TE faces at the middles of the windows, so (m_f - m_start) = (dm - w_LE) / 2 and
  (m_end - m_l) = (dm - w_TE) / 2 exactly (1e-12 c), both falling with refinement.
- The extensions within EXTENSION_TOLERANCE_CHORDS of one chord (measured 0.981 .. 1.007).
- Reproducible; one level asked alone equals the same level of the family, and
  make_grid(GridSettings(level=...)) and the solver's first grid are that grid.
- Optimization readiness: perturbed blades (stagger, camber, thickness, solidity) build a
  valid medium grid with stable SS / PS wall identity; blades without an admissible grid at a
  level (a nose window too wide for the level's dm, too few eta cells) raise
  GridGenerationError with the level, the geometry and every candidate's reason.
- Timing fields finite and non-negative (no speed is asserted).
"""

from __future__ import annotations

import math
import warnings
from functools import lru_cache

import numpy as np
import pytest

from potential_flow_solver import grid as gr
from potential_flow_solver.config import GridSettings
from potential_flow_solver.geometry import CascadeSpec, make_cascade

INLET = math.radians(45.0)
SAMPLES = 40001


@lru_cache(maxsize=None)
def cascade(**changes):
    spec = dict(blade_type="NACA65", solidity=1.0, stagger_deg=33.0, cl0=1.2, t_max_over_c=0.10,
                profile_variant="kovach_sandercock", surface_continuity="C2")
    spec.update(changes)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")            # GeometryWarning outside the checked NACA 65 range
        return make_cascade(CascadeSpec(**spec))


@lru_cache(maxsize=None)
def family(**changes):
    return gr.build_grid_family(cascade(**changes), inlet_angle=INLET)


def sampled_crossings(c, m):
    """(suction, pressure) crossings of the reference blade's curves with the lines x = m,
    from SAMPLES points per curve: the number of sample segments whose m range [lo, hi) holds
    the station (independent of grid.py's branches and bisection)."""
    u = np.linspace(0.0, 1.0, SAMPLES)
    out = []
    for curve in (c.active_curves.suction, c.active_curves.pressure):
        x = curve.evaluate(u)[:, 0]
        lo, hi = np.sort(np.minimum(x[:-1], x[1:])), np.sort(np.maximum(x[:-1], x[1:]))
        m = np.asarray(m, dtype=float)
        out.append(np.searchsorted(lo, m, side="right") - np.searchsorted(hi, m, side="right"))
    return out[0], out[1]


def test_exactly_five_named_levels_in_order():
    assert gr.GRID_LEVELS == ("very_coarse", "coarse", "medium", "fine", "very_fine")
    p = family()
    assert p.levels == gr.GRID_LEVELS and len(p) == 5
    assert [g.level for g in p] == list(gr.GRID_LEVELS) and all(p[name].level == name for name in gr.GRID_LEVELS)
    targets = [gr.LEVEL_TARGET_DM_OVER_CHORD[name] for name in gr.GRID_LEVELS]
    assert targets == pytest.approx([0.04, 0.02 * math.sqrt(2.0), 0.02, 0.02 / math.sqrt(2.0), 0.01], rel=1e-14)
    with pytest.raises(ValueError, match="level must be one of"):
        gr.build_grid(cascade(), "ultra_fine", inlet_angle=INLET)


def test_dm_and_deta_strictly_fall_as_a_refinement_family():
    infos = [g.info for g in family()]
    dm = np.array([i.dm for i in infos])
    deta = np.array([i.deta for i in infos])
    assert np.all(np.diff(dm) < 0.0) and np.all(np.diff(deta) < 0.0)
    ratios = dm[:-1] / dm[1:]
    assert np.all((ratios > 1.35) & (ratios < 1.48)), ratios                 # measured 1.400 .. 1.429
    for i in infos:
        assert abs(i.dm_over_chord / i.target_dm_over_chord - 1.0) < 0.02     # nearest candidate; measured < 0.8 %
        assert abs(i.cell_aspect / gr.CELL_ASPECT - 1.0) < 0.04, i.cell_aspect   # n_eta rounding; measured 1.784 .. 1.816
        assert i.n_eta - 1 == round(1.0 / (gr.CELL_ASPECT * i.dm_over_chord))   # pitch = 1 chord
        assert i.total_nodes == i.n_m * i.n_eta


def test_every_level_passes_its_invariants():
    for g in family():
        checks = gr.validate_standard_grid(g.grid)
        assert checks == g.info.checks and len(checks) == 13
        assert g.info.admissibility == "admissible"


def test_blade_stations_by_independent_curve_sampling():
    """One suction and one pressure crossing at every blade station f .. l; none at f - 1 and
    l + 1 (nor at any other periodic station): f is the first and l the last SS + PS station."""
    c = cascade()
    for g in family():
        grid = g.grid
        f, l = grid.first_blade_i, grid.last_blade_i
        ss, ps = sampled_crossings(c, grid.m)
        assert np.all(ss[f:l + 1] == 1) and np.all(ps[f:l + 1] == 1), g.level
        assert np.all(ss[:f] == 0) and np.all(ps[:f] == 0) and np.all(ss[l + 1:] == 0) and np.all(ps[l + 1:] == 0), g.level
        identity = grid.wall_identity
        assert set(identity.lower.surface) == {"suction"} and set(identity.upper.surface) == {"pressure"}


def test_the_three_kinds_of_blade_end():
    """Nose tip and tail end (the outline's extremes, here against the sampled curves), the
    construction LE / TE points and the SS + PS-valid interval are kept apart; for this blade
    m_start = m_le and m_end = m_te, the nose tip 4.521e-3 c ahead (suction side; 4.524e-3 c
    with the legacy PCHIP thickness), the Kovach tail end 1.35e-5 c behind (pressure side);
    emery_base has no TE window."""
    tr = family().transitions
    c = cascade()
    u = np.linspace(0.0, 1.0, SAMPLES)
    sampled = np.concatenate([curve.evaluate(u)[:, 0] for curve in (c.active_curves.suction, c.active_curves.pressure)])
    assert abs(tr.m_tip - sampled.min()) < 1e-8 and abs(tr.m_tail - sampled.max()) < 1e-8   # sampling error ~1e-10
    assert tr.m_start == tr.m_le == 0.0 and tr.m_end == tr.m_te
    assert tr.le_window == pytest.approx(4.520603e-3, abs=1e-9) and tr.te_window == pytest.approx(1.352435e-5, abs=1e-11)
    assert tr.le_face == 0.5 * (tr.m_tip + tr.m_start) and tr.te_face == 0.5 * (tr.m_end + tr.m_tail)
    plain = gr.passage_transitions(cascade(profile_variant="emery_base"))
    assert plain.te_window == 0.0 and plain.le_window == pytest.approx(tr.le_window, abs=1e-12)   # measured 1.4e-15 apart
    for lo, hi, ss, ps in tr.coverage:                                    # one SS + one PS inside, not beside
        if tr.m_start <= lo and hi <= tr.m_end:
            assert (ss, ps) == (1, 1)
        elif hi <= tr.m_start or lo >= tr.m_end:
            assert (ss, ps) != (1, 1)


def test_uniform_lattice_and_transition_gaps():
    tr = family().transitions
    le_gaps, te_gaps = [], []
    for g in family():
        grid, i = g.grid, g.info
        steps = np.diff(grid.m)
        assert np.max(np.abs(steps - grid.dm)) < 1e-14                     # measured 2.6e-16 .. 5.4e-16
        assert np.max(np.abs(np.diff(grid.eta) - grid.deta)) < 1e-15
        assert grid.m[0] == grid.boundaries.m_in and grid.m[-1] == grid.boundaries.m_out
        assert abs((i.m_first - tr.m_start) - 0.5 * (grid.dm - tr.le_window)) < 1e-12
        assert abs((tr.m_end - i.m_last) - 0.5 * (grid.dm - tr.te_window)) < 1e-12
        assert abs((tr.m_tip - grid.m[i.first_blade_i - 1]) - 0.5 * (grid.dm - tr.le_window)) < 1e-12
        assert i.le_gap_over_chord > gr.TRANSITION_CLEARANCE * i.dm_over_chord > 0.0
        assert i.n_blade_stations == i.n_blade_cells == i.last_blade_i - i.first_blade_i + 1
        le_gaps.append(i.le_gap_over_chord)
        te_gaps.append(i.te_gap_over_chord)
    assert np.all(np.diff(le_gaps) < 0.0) and np.all(np.diff(te_gaps) < 0.0)   # 0.0178 .. 0.0027, 0.0200 .. 0.0050


def test_extensions_within_tolerance():
    for g in family():
        i = g.info
        for value in (i.upstream_extension_chords, i.downstream_extension_chords):
            assert abs(value - 1.0) <= gr.EXTENSION_TOLERANCE_CHORDS          # measured 0.981 .. 1.007
            assert abs(value - 1.0) <= 0.5 * i.dm_over_chord + 1e-12          # whole cells: within half a cell


def test_reproducible_and_one_level_equals_the_family_member():
    a = gr.build_grid_family(cascade(), inlet_angle=INLET)
    b = family()
    for name in gr.GRID_LEVELS:
        assert np.array_equal(a[name].grid.nodes, b[name].grid.nodes) and np.array_equal(a[name].grid.m, b[name].grid.m)
        assert a[name].info == b[name].info and a[name].candidates == b[name].candidates
    single = gr.build_grid(cascade(), "medium", inlet_angle=INLET)
    assert np.array_equal(single.grid.nodes, b["medium"].grid.nodes) and single.info == b["medium"].info
    assert a.geometry_id == b.geometry_id == single.info.geometry_id == gr.geometry_identifier(cascade())
    assert gr.geometry_identifier(cascade(profile_variant="emery_base")) != a.geometry_id


def test_make_grid_level_mode_and_the_solver_take_the_same_grid():
    from potential_flow_solver import config as cf
    from potential_flow_solver import eos
    from potential_flow_solver import solver as sv

    settings = GridSettings(level="fine")
    grid = gr.make_grid(cascade(), settings, INLET, math.radians(20.0), require_ss_ps=True)
    expected = gr.build_grid(cascade(), "fine", inlet_angle=INLET, exit_angle=math.radians(20.0)).grid
    assert np.array_equal(grid.nodes, expected.nodes) and (grid.first_blade_i, grid.last_blade_i) == \
        (expected.first_blade_i, expected.last_blade_i)
    assert np.array_equal(grid.m, family()["fine"].grid.m)                  # the stations do not depend on the exit angle
    config = cf.SolverConfig(grid=settings,
                             flow=cf.FlowConditions(inlet_mach=0.3, inlet_angle_deg=45.0, exit_angle_guess_deg=20.0,
                                                    inlet_absolute_total_pressure=101325.0,
                                                    inlet_absolute_total_temperature=288.15, angular_speed=0.0),
                             gas=eos.CaloricallyPerfectGas(gamma=1.4, gas_constant=287.05),
                             iteration=cf.IterationSettings(density_tolerance=1e-8, max_density_iterations=10,
                                                            kutta_tolerance=1e-8, max_kutta_iterations=10))
    assert np.array_equal(sv._first_grid(cascade(), config).nodes, expected.nodes)
    with pytest.raises(ValueError, match="take no n_passage or n_eta"):
        gr.make_grid(cascade(), GridSettings(n_passage=61, n_eta=41, level="medium"), INLET)


def test_explicit_mode_is_still_the_explicit_lattice():
    """The legacy path: GridSettings(n_passage, n_eta) gives make_grid's section-2 lattice
    (exact one-chord extensions), not a standard level."""
    grid = gr.make_grid(cascade(), GridSettings(n_passage=61, n_eta=41), INLET)
    assert len(grid.m) == 207 and len(grid.eta) == 41 and (grid.first_blade_i, grid.last_blade_i) == (73, 133)
    assert grid.m[0] == -1.0 and grid.m[-1] == pytest.approx(grid.m_te + 1.0, abs=1e-15)


def test_timing_and_candidate_fields():
    p = family()
    for key in ("interrogation", *gr.GRID_LEVELS, "total"):
        assert math.isfinite(p.timing[key]) and p.timing[key] >= 0.0
    for g in p:
        for key in ("intersection", "construction", "validation", "search"):
            assert math.isfinite(g.timing[key]) and g.timing[key] >= 0.0
        s = g.candidates
        assert s.tested >= 1 and s.rejected == s.tested - 1 and s.accepted_rank == s.tested
        assert s.band[0] < g.info.dm_over_chord < s.band[1]
    single = gr.build_grid(cascade(), "coarse", inlet_angle=INLET)
    assert all(math.isfinite(v) and v >= 0.0 for v in single.timing.values())
    assert single.timing["total"] >= single.timing["interrogation"] + single.timing["search"] - 1e-9
    assert p.candidates_tested == sum(g.candidates.tested for g in p)


PERTURBED = [dict(stagger_deg=25.0), dict(stagger_deg=40.0), dict(cl0=0.8), dict(cl0=1.6), dict(t_max_over_c=0.06),
             dict(t_max_over_c=0.12), dict(solidity=0.7), dict(solidity=1.5), dict(stagger_deg=-10.0, cl0=0.4)]


@pytest.mark.parametrize("changes", PERTURBED, ids=lambda d: ",".join(f"{k}={v}" for k, v in d.items()))
def test_perturbed_blades_build_a_valid_medium_grid(changes):
    c = cascade(**changes)
    g = gr.build_grid(c, "medium", inlet_angle=INLET)
    grid = g.grid
    assert gr.validate_standard_grid(grid) == g.info.checks
    f, l = grid.first_blade_i, grid.last_blade_i
    ss, ps = sampled_crossings(c, grid.m)
    assert np.all(ss[f:l + 1] == 1) and np.all(ps[f:l + 1] == 1)
    assert ss[f - 1] == ps[f - 1] == ss[l + 1] == ps[l + 1] == 0
    assert set(grid.wall_identity.lower.surface) == {"suction"} and set(grid.wall_identity.upper.surface) == {"pressure"}


def test_a_nose_window_too_wide_for_the_level_is_a_structured_failure():
    """60 deg stagger: the nose window is 0.01355 c (0.01358 c with the legacy PCHIP
    thickness), so a level whose dm cannot exceed it by
    the clearance (fine: band 0.0127 .. 0.0156 c, needs dm >= 0.0170 c) has no admissible
    candidate; medium still builds."""
    c = cascade(stagger_deg=60.0)
    assert gr.passage_transitions(c).le_window == pytest.approx(0.013554, abs=1e-6)     # measured 0.0135539
    with pytest.raises(gr.GridGenerationError) as info:
        gr.build_grid(c, "fine", inlet_angle=INLET)
    error = info.value
    assert error.level == "fine" and error.geometry_id == gr.geometry_identifier(c)
    assert error.candidates and all(r.reason == gr.REJECT_LE and not r.accepted for r in error.candidates)
    assert error.reasons == {gr.REJECT_LE: len(error.candidates)}
    assert "could not produce an admissible fine grid" in str(error) and "LE window" in str(error)
    assert isinstance(error, gr.GridError)
    with pytest.raises(gr.GridGenerationError):
        gr.build_grid_family(c, inlet_angle=INLET)
    assert gr.build_grid(c, "medium", inlet_angle=INLET).info.admissibility == "admissible"


def test_too_few_eta_cells_is_a_structured_failure():
    """Solidity 3 (pitch c / 3): very_coarse would have 5 eta cells, below MIN_ETA_CELLS."""
    with pytest.raises(gr.GridGenerationError) as info:
        gr.build_grid(cascade(solidity=3.0, t_max_over_c=0.15, stagger_deg=10.0), "very_coarse", inlet_angle=INLET)
    assert set(info.value.reasons) == {gr.REJECT_ETA}


def test_rejected_candidates_are_counted_before_the_accepted_one():
    """50 deg stagger, very_fine: the nine candidates nearest the target leave too little
    clearance beside the 0.00858 c nose window; the tenth (at the band's edge) is taken."""
    g = gr.build_grid(cascade(stagger_deg=50.0), "very_fine", inlet_angle=INLET)
    s = g.candidates
    assert s.reasons == {gr.REJECT_LE: s.rejected} and s.rejected >= 1 and s.accepted_rank == s.rejected + 1
    assert all(not r.accepted for r in s.records[:-1]) and s.records[-1].accepted
