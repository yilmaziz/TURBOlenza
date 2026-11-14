"""
sideline_anchoring_study.py
===========================
The side lines of the passage anchored at the nose tip and the tail end instead of the chord-axis
LE and TE points: what that does to the eta-line corner offset delta at the transition stations.
A STUDY on the baseline of scripts/kutta_study.py and scripts/transition_study.py (Dunavant NACA
65-(12A10)10, C2, kovach_sandercock, 101 camber stations, stagger 33 deg, solidity 1, inlet 45 deg,
M1 0.65, exit angle kappa_te for the grid, the five standard levels). Script-first: nothing in
src/ changes. The tip-anchored grids are study-local Grid objects built here (tip_grid).

    python scripts/sideline_anchoring_study.py phaseA     # tip-anchored side lines, their checks, delta and Delta beta
    python scripts/sideline_anchoring_study.py phaseB     # the operator on an exact uniform flow, both anchors, gate B
    python scripts/sideline_anchoring_study.py phaseD     # side checks: curvature joints; the first-order periodic term
    (report: a later step)

Production side lines (grid.py section 1, make_passage_boundaries, _inlet_side, _exit_side): beta
linear in m from kappa_le at the LE point (m_le, y_le) to the inlet flow angle at m_in, y the
integral of tan(beta) through the LE point; the exit side the same from kappa_te at the TE point to
the exit angle at m_out. Aungier 2003 Ch. 5 p. 86 names neither the junction point on a rounded
nose nor which blade angle is meant, so both anchors are our choices within his rule (plan,
"Aungier"). The "tip" anchor keeps the beta law and moves only the junction: the inlet line ends at
the nose tip (most upstream blade point), the exit line starts at the tail end (most downstream).

Frames (Aungier 2003 Ch. 5): beta from the meridional direction, tan(beta) = dy/dm, eta = 0 the
suction wall of the reference blade, eta = 1 the pressure wall of the +pitch neighbour; planar,
r = 1. Lengths are reported in chords.
"""

# ============================================================
# SETTINGS
# ============================================================

OUTPUT_DIR = "outputs/sideline_anchoring_study"
ANCHORS = ("le_point", "tip")
DENSE_U = 200001                        # u samples per curve that bracket the sign changes of m'(u) (check of the tip)
BRENTQ_XTOL = 1e-15                     # brentq's absolute tolerance in u
CLOSEUP_LEVELS = ("very_coarse", "very_fine")
CLOSEUP_HALF_WIDTH_CHORDS = 0.09        # figure: half width of the LE / TE windows in m
GAMMAS_DEG = (20.0, 33.0, 45.0)         # phase B: uniform-flow angles from the meridional direction (transition_study's)
REPORT_GAMMA_DEG = 33.0                 # phase B: the gamma of the side-by-side, term and attribution tables
GATE_B_ORDER = -1.5                     # phase B: the plan's gate (order_max of f-1, f, l, l+1 with tip >= this); fixed
CONSISTENCY_REL = 1e-12                 # phase B: le_point rows against transition_study's phaseB.json (the plan's bound)
GEOMETRY_LEVEL = "medium"               # phase D1: the cascade (the geometry does not depend on the level)
JOINT_STEP_U = 1e-9                     # phase D1: one-sided values at u -+ this
DENSE_CURVE_U = 40001                   # phase D1: u samples per curve (figure, arc length, the derivative self-check)
X_C_MARKS = (0.64, 0.66)                # phase D1: the x/c of the brief
D2_GAMMA_DEG = 33.0                     # phase D2: the uniform-flow angle
STOKES_CSV = "outputs/kutta_study/stokes_check.csv"     # phase D1: where "x/c 0.66" was looked for

# ============================================================

import argparse                                                                  # noqa: E402
import csv                                                                       # noqa: E402
import json                                                                      # noqa: E402
import math                                                                      # noqa: E402
import os                                                                        # noqa: E402
import sys                                                                       # noqa: E402
from dataclasses import replace                                                  # noqa: E402
from functools import lru_cache                                                  # noqa: E402
from pathlib import Path                                                         # noqa: E402

import numpy as np                                                               # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import kutta_study as ks                                                         # noqa: E402  (the same baseline)
import transition_study as ts                                                    # noqa: E402  (power_law, the delta rule)
from turbolenza.b2b import grid as gr                                            # noqa: E402

OUT = ROOT / OUTPUT_DIR
LEVELS = ks.LEVELS
LINES: list = []
ROWS = ("f-1", "f", "l", "l+1")         # the transition stations of transition_study.corner_attribution


# --- output helpers (as in transition_study.py, pointing at this study's folder) ----------------------------------

def say(text: str = "") -> None:
    print(text)
    LINES.append(text)


def table(rows, columns, title, digits=4) -> None:
    say(f"**{title}**")
    say()
    say("| " + " | ".join(h for _, h in columns) + " |")
    say("|" + "|".join("---" for _ in columns) + "|")
    for r in rows:
        say("| " + " | ".join(ks.fmt(r.get(k), digits) for k, _ in columns) + " |")
    say()


def save(name: str, value) -> None:
    (OUT / "data").mkdir(parents=True, exist_ok=True)
    (OUT / "data" / name).write_text(json.dumps(value, indent=1, default=ks._jsonable), encoding="utf-8")


def write_csv(name: str, rows: list) -> None:
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(OUT / name, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: (json.dumps(v, default=ks._jsonable) if isinstance(v, (list, tuple, dict)) else v)
                             for k, v in r.items()})


def write_lines(name: str) -> None:
    (OUT / name).write_text("\n".join(LINES) + "\n", encoding="utf-8")
    LINES.clear()


@lru_cache(maxsize=None)
def case_of(level: str):
    """The controller case of one level, kutta_study's baseline (its grid has exit angle kappa_te)."""
    return ks.case_of(level)


# --- (1) the nose tip and the tail end ----------------------------------------------------------------------------

def _extreme(name: str, u: float, curve) -> dict:
    point = np.asarray(curve.evaluate(float(u)), dtype=float)
    return dict(curve=name, u=float(u), point=point, m=float(point[0]), y=float(point[1]))


def tip_and_tail(cascade) -> dict:
    """The nose tip (least m) and the tail end (greatest m) of the reference blade of a C2 cascade,
    from its authoritative curves cascade.active_curves.suction / .pressure: the candidates are
    grid.py's own turning points of m(u) (grid._turning_points(curve, 0)) and the curve ends u = 0,
    u = 1; each point is curve.evaluate(u). Returns {"tip": ..., "tail": ...}, each a dict of the
    curve name, u, point (m, y), m and y."""
    curves = cascade.active_curves
    if curves is None:
        raise ValueError("tip_and_tail needs a C2 cascade (active_curves is None for C1)")
    candidates = []
    for name in ("suction", "pressure"):
        curve = getattr(curves, name)
        for u in np.concatenate(([0.0], gr._turning_points(curve, 0), [1.0])):
            candidates.append(_extreme(name, float(u), curve))
    return dict(tip=min(candidates, key=lambda c: c["m"]), tail=max(candidates, key=lambda c: c["m"]))


def brentq_extremes(cascade, n: int = DENSE_U, xtol: float = BRENTQ_XTOL) -> dict:
    """The independent check of tip_and_tail: every sign change of m'(u) = curve.derivative(u, 1)[0]
    on n uniform samples of u in [0, 1] of each curve, solved with scipy.optimize.brentq (xtol),
    plus the curve ends and any sample where m'(u) is exactly zero; tip = least m, tail = greatest.
    Uses neither grid.py's PPoly roots nor its bisection."""
    from scipy.optimize import brentq

    candidates = []
    for name in ("suction", "pressure"):
        curve = getattr(cascade.active_curves, name)
        u = np.linspace(0.0, 1.0, n)
        sign = np.sign(curve.derivative(u, 1)[:, 0])
        roots = [0.0, 1.0] + [float(x) for x in u[sign == 0.0]]
        for k in np.flatnonzero(sign[:-1] * sign[1:] < 0.0):
            roots.append(brentq(lambda t: float(curve.derivative(t, 1)[0]), float(u[k]), float(u[k + 1]), xtol=xtol,
                                maxiter=500))
        candidates += [_extreme(name, r, curve) for r in roots]
    return dict(tip=min(candidates, key=lambda c: c["m"]), tail=max(candidates, key=lambda c: c["m"]))


def vertical_tangent(cascade, extreme: dict) -> float:
    """|m'(u)| / |X'(u)| of the curve at an extreme: the cosine of the wall angle there (0 = vertical)."""
    d1 = np.asarray(getattr(cascade.active_curves, extreme["curve"]).derivative(extreme["u"], 1), dtype=float)
    return float(abs(d1[0]) / math.hypot(d1[0], d1[1]))


# --- (3) the tip-anchored side lines ------------------------------------------------------------------------------

def anchored_side(grid, cascade, ends: dict, side: str, m) -> tuple[np.ndarray, np.ndarray]:
    """(beta, y) of the tip-anchored lower side line at m, by grid.py's own _side_line: the inlet
    line (side "inlet") through the nose tip, fraction (m - m_tip) / (m_in - m_tip), kappa_le at the
    tip -> b.inlet_angle at m_in; the exit line (side "exit") through the tail end, fraction
    (m - m_tail) / (m_out - m_tail), kappa_te at the tail -> b.exit_angle at m_out."""
    b = grid.boundaries
    m = np.atleast_1d(np.asarray(m, dtype=float))
    if side == "inlet":
        junction, far_m, junction_angle, far_angle = ends["tip"]["point"], b.m_in, cascade.kappa_le, b.inlet_angle
    elif side == "exit":
        junction, far_m, junction_angle, far_angle = ends["tail"]["point"], b.m_out, cascade.kappa_te, b.exit_angle
    else:
        raise ValueError(f"side must be 'inlet' or 'exit', got {side!r}")
    junction = np.asarray(junction, dtype=float)
    fraction = gr._station_fraction(m, float(junction[0]), far_m)
    return gr._side_line(m, fraction, junction, float(junction_angle), float(far_angle))


def tip_side_lines(grid, cascade, ends: dict | None = None) -> dict:
    """The tip-anchored lower side lines at the grid's periodic stations: "inlet" at 0 .. f-1 and
    "exit" at l+1 .. n_m-1, each (m, beta, y)."""
    ends = tip_and_tail(cascade) if ends is None else ends
    f, l = grid.first_blade_i, grid.last_blade_i
    m = np.asarray(grid.m, dtype=float)
    out = {}
    for side, stations in (("inlet", m[:f]), ("exit", m[l + 1:])):
        beta, y = anchored_side(grid, cascade, ends, side, stations)
        out[side] = dict(m=stations, beta=beta, y=y)
    return out


# --- (4) the independent check: adaptive quadrature of dy/dm = tan(beta(m)) ----------------------------------------

def quadrature_side(junction, junction_angle: float, far_m: float, far_angle: float):
    """y(m) of the lower side line by adaptive quadrature from the junction (scipy.integrate.quad),
    beta linear in m from junction_angle at the junction to far_angle at far_m: the function
    aungier_side of scripts/audit_aungier_side_boundaries.py (~line 74), imported (it imports
    headless with MPLBACKEND=Agg; its module sets a filter that ignores IntegrationWarning)."""
    os.environ.setdefault("MPLBACKEND", "Agg")
    import audit_aungier_side_boundaries as audit

    y, _ = audit.aungier_side(float(junction[0]), float(junction[1]), float(junction_angle), float(far_m),
                              float(far_angle))
    return y


def side_line_checks(grid, cascade, ends: dict) -> dict:
    """Max |side line - quadrature| / c at the stations: the tip-anchored lines (tip_side_lines)
    against the quadrature from the tip / tail, and, as a control, the production lines (the grid's
    own lower nodes at the periodic stations) against the quadrature from cascade.active's LE / TE
    point."""
    b, c = grid.boundaries, grid.chord
    f, l = grid.first_blade_i, grid.last_blade_i
    m = np.asarray(grid.m, dtype=float)
    y0 = np.asarray(grid.nodes, dtype=float)[:, 0, 1]
    tip = tip_side_lines(grid, cascade, ends)
    le_point = np.asarray(cascade.active.leading_edge_point, dtype=float)
    te_point = np.asarray(cascade.active.trailing_edge_point, dtype=float)
    k_le, k_te = float(cascade.kappa_le), float(cascade.kappa_te)
    q = dict(tip_inlet=quadrature_side(ends["tip"]["point"], k_le, b.m_in, b.inlet_angle)(m[:f]),
             tip_exit=quadrature_side(ends["tail"]["point"], k_te, b.m_out, b.exit_angle)(m[l + 1:]),
             le_point_inlet=quadrature_side(le_point, k_le, b.m_in, b.inlet_angle)(m[:f]),
             le_point_exit=quadrature_side(te_point, k_te, b.m_out, b.exit_angle)(m[l + 1:]))
    ours = dict(tip_inlet=tip["inlet"]["y"], tip_exit=tip["exit"]["y"], le_point_inlet=y0[:f], le_point_exit=y0[l + 1:])
    return {k: float(np.max(np.abs(ours[k] - q[k]))) / c for k in q}


# --- (5) the study-local tip-anchored Grid --------------------------------------------------------------------------

def tip_grid(grid, cascade, ends: dict | None = None):
    """The production grid with its side lines re-anchored at the nose tip and the tail end: a
    study-local Grid, dataclasses.replace on the production one.

    boundaries: upstream_m = linspace(m_in, m_tip, len(b.upstream_m)), upstream_beta and
    upstream_lower_boundary from grid._side_line with grid._sample_fraction(n, junction_last=True)
    through the tip, upstream_upper_boundary = lower + periodic_translation; downstream_* likewise
    from m_tail to m_out (junction_last=False) through the tail end. nodes: replaced at the inlet
    (0 .. f-1) and exit (l+1 .. n_m-1) stations only, lower = the tip side line (tip_side_lines),
    upper = lower + pitch, y = grid._across(eta, lower, upper), as grid._grid_on_stations builds
    them. Grid.__post_init__ re-validates the topology, the width and the wall identity; m, eta,
    first_blade_i, last_blade_i and the blade-station nodes (f .. l, all j) are checked to be
    bit-identical to the production grid (RuntimeError otherwise).

    NEVER call grid.rebuild_exit on this grid, nor feed its boundaries to grid._inlet_side /
    _exit_side: those measure the station fraction from b.m_le / b.m_te, so they would re-anchor
    the lines at the LE / TE point (and mix the two anchors). metrics.py, boundary_conditions.py,
    coefficients.py, discretization.py, matrix_assembly.py and flow.py read only the nodes and the
    topology (main session's grep: none reads the side-line arrays of boundaries or m_le / m_te),
    so build_metrics and what follows see the tip anchoring through the nodes alone."""
    ends = tip_and_tail(cascade) if ends is None else ends
    b = grid.boundaries
    tip, tail = np.asarray(ends["tip"]["point"], float), np.asarray(ends["tail"]["point"], float)
    n_up, n_down = len(b.upstream_m), len(b.downstream_m)
    up_m = np.linspace(b.m_in, tip[0], n_up)
    up_beta, up_y = gr._side_line(up_m, gr._sample_fraction(n_up, junction_last=True), tip, float(cascade.kappa_le),
                                  b.inlet_angle)
    down_m = np.linspace(tail[0], b.m_out, n_down)
    down_beta, down_y = gr._side_line(down_m, gr._sample_fraction(n_down, junction_last=False), tail,
                                      float(cascade.kappa_te), b.exit_angle)
    up_lower, down_lower = np.column_stack((up_m, up_y)), np.column_stack((down_m, down_y))
    boundaries = replace(b, upstream_m=up_m, upstream_beta=up_beta, upstream_lower_boundary=up_lower,
                         upstream_upper_boundary=up_lower + b.periodic_translation, downstream_m=down_m,
                         downstream_beta=down_beta, downstream_lower_boundary=down_lower,
                         downstream_upper_boundary=down_lower + b.periodic_translation)
    lines = tip_side_lines(grid, cascade, ends)
    f, l = grid.first_blade_i, grid.last_blade_i
    nodes = np.array(grid.nodes, dtype=float)
    for stations, lower in ((slice(0, f), lines["inlet"]["y"]), (slice(l + 1, None), lines["exit"]["y"])):
        nodes[stations, :, 1] = gr._across(np.asarray(grid.eta), lower, lower + b.periodic_translation[1])
    out = replace(grid, boundaries=boundaries, nodes=nodes)
    same = dict(m=np.array_equal(out.m, grid.m), eta=np.array_equal(out.eta, grid.eta),
                blade_run=(out.first_blade_i, out.last_blade_i) == (f, l),
                blade_nodes=np.array_equal(out.nodes[f:l + 1], grid.nodes[f:l + 1]))
    if not all(same.values()):
        raise RuntimeError(f"the tip grid changed more than its periodic-station nodes: {same}")
    return out


# --- (6) the eta-line corner offset delta ----------------------------------------------------------------------------

def corner_delta(grid) -> dict:
    """delta at the transition stations exactly as transition_study.corner_attribution defines it:
    at row i in {f-1, f, l, l+1}, the node across the transition (f, f-1, l+1, l) minus the
    quadratic extrapolation 3 y_a - 3 y_b + y_c of the row's own side of the eta-line, j = 1 .. J-1.
    Per row: max |delta|, delta at j = 1 and at j = J-1, the j of the max and the whole array (in the
    grid's length unit)."""
    y = np.asarray(grid.nodes, dtype=float)[..., 1]
    f, l, J = grid.first_blade_i, grid.last_blade_i, len(grid.eta) - 1
    js = np.arange(1, J)
    out = {}
    for name, across, own in (("f-1", f, (f - 1, f - 2, f - 3)), ("f", f - 1, (f, f + 1, f + 2)),
                              ("l", l + 1, (l, l - 1, l - 2)), ("l+1", l, (l + 1, l + 2, l + 3))):
        a, b, c3 = own
        delta = y[across, js] - (3 * y[a, js] - 3 * y[b, js] + y[c3, js])
        k = int(np.argmax(np.abs(delta)))
        out[name] = dict(max_abs=float(np.abs(delta[k])), j1=float(delta[0]), jJm1=float(delta[-1]), j_max=int(js[k]),
                         delta=delta)
    return out


# --- (7) phase A: per level, both anchors -----------------------------------------------------------------------------

def beta_law(m: float, junction_m: float, junction_angle: float, far_m: float, far_angle: float) -> float:
    """The side-line angle written out (not grid.py's): linear in m from junction_angle to far_angle."""
    return junction_angle + (m - junction_m) / (far_m - junction_m) * (far_angle - junction_angle)


def acute(angle_deg: float) -> float:
    """An angle between two line directions in (-90, 90] degrees."""
    a = (angle_deg + 90.0) % 180.0 - 90.0
    return 90.0 if a == -90.0 else a


def level_data(level: str) -> dict:
    """Everything phase A measures on one level, for both anchors."""
    from turbolenza.b2b.metrics import build_metrics

    case = case_of(level)
    grid, cascade = case.grid, case.cascade
    b, c = grid.boundaries, grid.chord
    f, l, J = grid.first_blade_i, grid.last_blade_i, len(grid.eta) - 1
    m = np.asarray(grid.m, dtype=float)
    ends, check = tip_and_tail(cascade), brentq_extremes(cascade)
    row_tip, row_tail = gr._row_of(b).extremes
    tg = tip_grid(grid, cascade, ends)
    tip, tail = ends["tip"], ends["tail"]
    k_le, k_te = float(cascade.kappa_le), float(cascade.kappa_te)
    le_point = np.asarray(cascade.active.leading_edge_point, dtype=float)
    te_point = np.asarray(cascade.active.trailing_edge_point, dtype=float)
    out = dict(level=level, dm_over_c=grid.dm / c, n_m=len(m), n_eta=J + 1, f=f, l=l, chord=c,
               kappa_le_deg=math.degrees(k_le), kappa_te_deg=math.degrees(k_te),
               inlet_angle_deg=math.degrees(b.inlet_angle), exit_angle_deg=math.degrees(b.exit_angle))
    # (1), (2): the extremes and their independent check
    out["ends"] = dict(
        tip_curve=tip["curve"], tip_u=tip["u"], tip_m_over_c=tip["m"] / c, tip_y_over_c=tip["y"] / c,
        tail_curve=tail["curve"], tail_u=tail["u"], tail_m_over_c=tail["m"] / c, tail_y_over_c=tail["y"] / c,
        w_le_over_c=(b.m_le - tip["m"]) / c, w_te_over_c=(tail["m"] - b.m_te) / c,
        brentq_tip_du=abs(check["tip"]["u"] - tip["u"]), brentq_tip_dm_over_c=abs(check["tip"]["m"] - tip["m"]) / c,
        brentq_tip_dy_over_c=abs(check["tip"]["y"] - tip["y"]) / c,
        brentq_tail_du=abs(check["tail"]["u"] - tail["u"]),
        brentq_tail_dm_over_c=abs(check["tail"]["m"] - tail["m"]) / c,
        brentq_tail_dy_over_c=abs(check["tail"]["y"] - tail["y"]) / c,
        brentq_same_curves=(check["tip"]["curve"], check["tail"]["curve"]) == (tip["curve"], tail["curve"]),
        row_extremes_dm_over_c=max(abs(row_tip - tip["m"]), abs(row_tail - tail["m"])) / c,
        cos_wall_tip=vertical_tangent(cascade, tip), cos_wall_tail=vertical_tangent(cascade, tail))
    # (4): the side lines against the quadrature
    out["side_checks"] = side_line_checks(grid, cascade, ends)
    yp, yt = np.asarray(grid.nodes, float)[:, 0, 1], np.asarray(tg.nodes, float)[:, 0, 1]
    out["tip_grid"] = dict(shift_fm1_over_c=(yt[f - 1] - yp[f - 1]) / c, shift_lp1_over_c=(yt[l + 1] - yp[l + 1]) / c,
                           max_shift_inlet_over_c=float(np.max(np.abs(yt[:f] - yp[:f]))) / c,
                           max_shift_exit_over_c=float(np.max(np.abs(yt[l + 1:] - yp[l + 1:]))) / c,
                           far_inlet_shift_over_c=(yt[0] - yp[0]) / c, far_exit_shift_over_c=(yt[-1] - yp[-1]) / c)
    # the distances
    out["distances"] = dict(f_minus_tip=(m[f] - tip["m"]) / c, tip_minus_fm1=(tip["m"] - m[f - 1]) / c,
                            tail_minus_l=(tail["m"] - m[l]) / c, lp1_minus_tail=(m[l + 1] - tail["m"]) / c,
                            f_minus_le=(m[f] - b.m_le) / c, te_minus_l=(b.m_te - m[l]) / c)
    wid = grid.wall_identity
    surfaces = dict(f_lower=str(wid.lower.surface[0]), f_upper=str(wid.upper.surface[0]),
                    l_lower=str(wid.lower.surface[-1]), l_upper=str(wid.upper.surface[-1]))
    # the wall slope at the LE / TE point (on the suction curve, u = 0 / 1) for the le_point kink
    s_curve = cascade.active_curves.suction
    d_le, d_te = (np.asarray(s_curve.derivative(u, 1), float) for u in (0.0, 1.0))
    wall_at_le, wall_at_te = (math.degrees(math.atan(d[1] / d[0])) for d in (d_le, d_te))
    out["anchors"] = {}
    for anchor, g in (("le_point", grid), ("tip", tg)):
        met = build_metrics(g)
        deg = lambda t: math.degrees(math.atan(float(t)))  # noqa: E731
        if anchor == "le_point":
            jump_le = float(gr._inlet_side(b, np.array([tip["m"]]))[0]) - tip["y"]
            jump_te = float(gr._exit_side(b, np.array([tail["m"]]))[0]) - tail["y"]
            j_le_m, j_te_m = b.m_le, b.m_te
            wall_le, wall_te = wall_at_le, wall_at_te
            anchor_le, anchor_te = le_point, te_point
        else:
            jump_le = float(anchored_side(grid, cascade, ends, "inlet", [tip["m"]])[1][0]) - tip["y"]
            jump_te = float(anchored_side(grid, cascade, ends, "exit", [tail["m"]])[1][0]) - tail["y"]
            j_le_m, j_te_m = tip["m"], tail["m"]
            wall_le, wall_te = 90.0, 90.0                      # vertical tangent at the nose tip / tail end
            anchor_le, anchor_te = tip["point"], tail["point"]
        side_fm1, side_lp1 = deg(met.tan_beta0[f - 1]), deg(met.tan_beta0[l + 1])
        law_fm1 = math.degrees(beta_law(m[f - 1], j_le_m, k_le, b.m_in, b.inlet_angle))
        law_lp1 = math.degrees(beta_law(m[l + 1], j_te_m, k_te, b.m_out, b.exit_angle))
        a = dict(jump_le_over_c=jump_le / c, jump_te_over_c=jump_te / c,
                 anchor_le_m_over_c=float(anchor_le[0]) / c, anchor_te_m_over_c=float(anchor_te[0]) / c,
                 wall_at_anchor_le_deg=wall_le, wall_at_anchor_te_deg=wall_te,
                 kink_at_anchor_le_deg=acute(wall_le - math.degrees(k_le)),
                 kink_at_anchor_te_deg=acute(wall_te - math.degrees(k_te)),
                 beta_f_lower_deg=deg(met.tan_beta0[f]), beta_f_upper_deg=deg(met.tan_beta1[f]),
                 beta_l_lower_deg=deg(met.tan_beta0[l]), beta_l_upper_deg=deg(met.tan_beta1[l]),
                 side_fm1_metrics_deg=side_fm1, side_fm1_law_deg=law_fm1, side_fm1_metrics_minus_law_deg=side_fm1 - law_fm1,
                 side_lp1_metrics_deg=side_lp1, side_lp1_law_deg=law_lp1, side_lp1_metrics_minus_law_deg=side_lp1 - law_lp1,
                 periodic_fm1_lp1_equal=bool(met.tan_beta0[f - 1] == met.tan_beta1[f - 1]
                                             and met.tan_beta0[l + 1] == met.tan_beta1[l + 1]),
                 **surfaces)
        a["dbeta_f_lower_deg"] = a["beta_f_lower_deg"] - side_fm1
        a["dbeta_f_upper_deg"] = a["beta_f_upper_deg"] - side_fm1
        a["dbeta_l_lower_deg"] = a["beta_l_lower_deg"] - side_lp1
        a["dbeta_l_upper_deg"] = a["beta_l_upper_deg"] - side_lp1
        a["delta"] = {k: {kk: (vv / c if kk != "j_max" else vv) for kk, vv in v.items()}
                      for k, v in corner_delta(g).items()}
        out["anchors"][anchor] = a
    return out


def published_le_point_delta() -> dict | None:
    """outputs/transition_study/data/phaseC_attribution.json (max |delta|/c by level), if present."""
    path = ROOT / ts.OUTPUT_DIR / "data" / "phaseC_attribution.json"
    if not path.exists():
        return None
    return {r["level"]: {k: r[f"{k} delta/c"] for k in ROWS} for r in json.loads(path.read_text(encoding="utf-8"))}


def phaseA() -> None:
    say("## Phase A: side lines anchored at the nose tip and the tail end (study-local grids)")
    say()
    say("Case: kutta_study.case_of(level) (NACA 65-(12A10)10, C2, kovach_sandercock, 101 camber stations, stagger 33 deg, "
        "solidity 1, inlet 45 deg, M1 0.65), grid = case.grid (exit angle kappa_te). le_point = the production grid; "
        "tip = tip_grid(case.grid): the same stations, eta and blade-station nodes, the inlet and exit side lines through "
        "the nose tip and the tail end with the same beta law. No solve. Lengths in chords.")
    say()
    data = [level_data(level) for level in LEVELS]
    c_rows, end_rows, jump_rows, le_rows, te_rows, dist_rows, delta_rows, detail_rows = [], [], [], [], [], [], [], []
    for d in data:
        e = d["ends"]
        end_rows.append(dict(level=d["level"], **e))
        c_rows.append(dict(level=d["level"], **d["side_checks"], **d["tip_grid"]))
        dist_rows.append(dict(level=d["level"], dm_over_c=d["dm_over_c"], n_m=d["n_m"], n_eta=d["n_eta"], f=d["f"],
                              l=d["l"], **d["distances"]))
        for anchor in ANCHORS:
            a = d["anchors"][anchor]
            base = dict(level=d["level"], anchor=anchor)
            jump_rows.append(dict(base, jump_le_over_c=a["jump_le_over_c"], jump_te_over_c=a["jump_te_over_c"],
                                  anchor_le_m_over_c=a["anchor_le_m_over_c"], anchor_te_m_over_c=a["anchor_te_m_over_c"]))
            le_rows.append(dict(base, kappa_le_deg=d["kappa_le_deg"], wall_at_anchor_le_deg=a["wall_at_anchor_le_deg"],
                                kink_at_anchor_le_deg=a["kink_at_anchor_le_deg"], surfaces=f"{a['f_lower']}/{a['f_upper']}",
                                beta_f_lower_deg=a["beta_f_lower_deg"], beta_f_upper_deg=a["beta_f_upper_deg"],
                                side_fm1_metrics_deg=a["side_fm1_metrics_deg"], side_fm1_law_deg=a["side_fm1_law_deg"],
                                side_fm1_metrics_minus_law_deg=a["side_fm1_metrics_minus_law_deg"],
                                dbeta_f_lower_deg=a["dbeta_f_lower_deg"], dbeta_f_upper_deg=a["dbeta_f_upper_deg"]))
            te_rows.append(dict(base, kappa_te_deg=d["kappa_te_deg"], wall_at_anchor_te_deg=a["wall_at_anchor_te_deg"],
                                kink_at_anchor_te_deg=a["kink_at_anchor_te_deg"], surfaces=f"{a['l_lower']}/{a['l_upper']}",
                                beta_l_lower_deg=a["beta_l_lower_deg"], beta_l_upper_deg=a["beta_l_upper_deg"],
                                side_lp1_metrics_deg=a["side_lp1_metrics_deg"], side_lp1_law_deg=a["side_lp1_law_deg"],
                                side_lp1_metrics_minus_law_deg=a["side_lp1_metrics_minus_law_deg"],
                                dbeta_l_lower_deg=a["dbeta_l_lower_deg"], dbeta_l_upper_deg=a["dbeta_l_upper_deg"]))
            delta_rows.append(dict(base, dm_over_c=d["dm_over_c"], **{k: a["delta"][k]["max_abs"] for k in ROWS}))
            for k in ROWS:
                v = a["delta"][k]
                detail_rows.append(dict(base, row=k, max_abs=v["max_abs"], j_max=v["j_max"], j1=v["j1"], jJm1=v["jJm1"],
                                        J=d["n_eta"] - 1))
    # (1), (2)
    table(end_rows, [("level", "level"), ("tip_curve", "tip on"), ("tip_u", "u_tip"), ("tip_m_over_c", "m_tip/c"),
                     ("tip_y_over_c", "y_tip/c"), ("tail_curve", "tail on"), ("tail_u", "u_tail"),
                     ("tail_m_over_c", "m_tail/c"), ("tail_y_over_c", "y_tail/c"), ("w_le_over_c", "w_LE = m_le - m_tip"),
                     ("w_te_over_c", "w_TE = m_tail - m_te")],
          "(1) Nose tip and tail end from grid._turning_points(curve, 0) and the curve ends (C2 curves)", digits=6)
    table(end_rows, [("level", "level"), ("brentq_tip_du", "tip |du|"), ("brentq_tip_dm_over_c", "tip |dm|/c"),
                     ("brentq_tip_dy_over_c", "tip |dy|/c"), ("brentq_tail_du", "tail |du|"),
                     ("brentq_tail_dm_over_c", "tail |dm|/c"), ("brentq_tail_dy_over_c", "tail |dy|/c"),
                     ("brentq_same_curves", "same curves"), ("row_extremes_dm_over_c", "|m - _CurveRow.extremes|/c"),
                     ("cos_wall_tip", "|m'|/|X'| at tip"), ("cos_wall_tail", "|m'|/|X'| at tail")],
          f"(2) Check: brentq root of m'(u) (sign changes on {DENSE_U} u samples per curve, xtol {BRENTQ_XTOL:g}) "
          f"against (1); the wall tangent at the extremes", digits=3)
    # (4), (5)
    table(c_rows, [("level", "level"), ("tip_inlet", "tip inlet"), ("tip_exit", "tip exit"),
                   ("le_point_inlet", "le_point inlet (control)"), ("le_point_exit", "le_point exit (control)"),
                   ("shift_fm1_over_c", "tip - le_point y at f-1"), ("shift_lp1_over_c", "at l+1"),
                   ("far_inlet_shift_over_c", "at i = 0"), ("far_exit_shift_over_c", "at i = n_m-1")],
          "(4) Max |side line - quadrature|/c at the periodic stations (scipy quad of tan(beta(m)) from the junction); "
          "(5) how far the tip grid moved the lower side-line node, /c", digits=3)
    say("(5) tip_grid: Grid.__post_init__ passed on every level (topology, width, wall identity); m, eta, "
        "first_blade_i, last_blade_i and the nodes of stations f .. l are bit-identical to the production grid "
        "(np.array_equal, checked inside tip_grid); build_metrics(tip grid) built on every level.")
    say()
    # (7) position jumps
    table(jump_rows, [("level", "level"), ("anchor", "anchor"), ("anchor_le_m_over_c", "LE anchor m/c"),
                      ("jump_le_over_c", "LE jump: side(m_tip) - y_tip"), ("anchor_te_m_over_c", "TE anchor m/c"),
                      ("jump_te_over_c", "TE jump: side(m_tail) - y_tail")],
          "Position jump of the side line at the nose tip and the tail end, /c", digits=4)
    table(le_rows, [("level", "level"), ("anchor", "anchor"), ("kappa_le_deg", "kappa_le"),
                    ("wall_at_anchor_le_deg", "wall at anchor"), ("kink_at_anchor_le_deg", "kink at anchor"),
                    ("surfaces", "f: j=0/j=J"), ("beta_f_lower_deg", "beta wall f, j=0"),
                    ("beta_f_upper_deg", "beta wall f, j=J"), ("side_fm1_metrics_deg", "beta side f-1 (metrics)"),
                    ("side_fm1_law_deg", "beta side f-1 (law)"), ("side_fm1_metrics_minus_law_deg", "metrics - law"),
                    ("dbeta_f_lower_deg", "dbeta j=0: wall f - side f-1"),
                    ("dbeta_f_upper_deg", "dbeta j=J: wall f - side f-1")],
          "Delta beta at the LE, degrees (walls from build_metrics: the C2 curves' exact slope; the side line at f-1 from "
          "the same metrics, one-sided in its run, and from the beta law; wall at the tip = vertical, 90 deg)", digits=3)
    table(te_rows, [("level", "level"), ("anchor", "anchor"), ("kappa_te_deg", "kappa_te"),
                    ("wall_at_anchor_te_deg", "wall at anchor"), ("kink_at_anchor_te_deg", "kink at anchor"),
                    ("surfaces", "l: j=0/j=J"), ("beta_l_lower_deg", "beta wall l, j=0"),
                    ("beta_l_upper_deg", "beta wall l, j=J"), ("side_lp1_metrics_deg", "beta side l+1 (metrics)"),
                    ("side_lp1_law_deg", "beta side l+1 (law)"), ("side_lp1_metrics_minus_law_deg", "metrics - law"),
                    ("dbeta_l_lower_deg", "dbeta j=0: wall l - side l+1"),
                    ("dbeta_l_upper_deg", "dbeta j=J: wall l - side l+1")],
          "Delta beta at the TE, degrees (wall at the tail end = vertical, 90 deg)", digits=3)
    say("The wall angle at the le_point anchor is the suction curve's slope angle at u = 0 (LE point) / u = 1 (TE "
        "point); kink at anchor = the acute angle between the wall tangent there and kappa (side line's junction angle).")
    say()
    # (6) delta and its orders
    order_rows = []
    dm = [d["dm_over_c"] for d in data]
    for anchor in ANCHORS:
        for k in ROWS:
            v = [r[k] for r in delta_rows if r["anchor"] == anchor]
            fit = ts.power_law(dm, v)
            order_rows.append(dict(anchor=anchor, row=k, very_coarse=v[0], very_fine=v[-1], order_fit=fit["p"],
                                   a_fit=fit["a"], order_last=math.log(v[-2] / v[-1]) / math.log(dm[-2] / dm[-1]),
                                   residual_order_estimate=fit["p"] - 2.0))
    delta_side = []
    for d in data:
        r = dict(level=d["level"], dm_over_c=d["dm_over_c"])
        for anchor in ANCHORS:
            for k in ROWS:
                r[f"{anchor} {k}"] = d["anchors"][anchor]["delta"][k]["max_abs"]
        delta_side.append(r)
    table(delta_side, [("level", "level"), ("dm_over_c", "dm/c")] +
          [(f"{anchor} {k}", f"{anchor} {k}") for anchor in ANCHORS for k in ROWS],
          "(6) max |delta|/c over j = 1 .. J-1 at the transition rows (transition_study.corner_attribution's rule)", digits=5)
    table(detail_rows, [("level", "level"), ("anchor", "anchor"), ("row", "row"), ("max_abs", "max |delta|/c"),
                        ("j_max", "j of max"), ("J", "J"), ("j1", "delta(j=1)/c"), ("jJm1", "delta(j=J-1)/c")],
          "(6) delta in detail", digits=5)
    table(order_rows, [("anchor", "anchor"), ("row", "row"), ("very_coarse", "very_coarse"), ("very_fine", "very_fine"),
                       ("order_fit", "order p (fit, |delta| = a dm^p)"), ("order_last", "order fine -> very_fine"),
                       ("residual_order_estimate", "p - 2 (residual ~ C delta / (s dm^2))")],
          "Orders of max |delta| over the five levels", digits=4)
    published = published_le_point_delta()
    if published is None:
        say("outputs/transition_study/data/phaseC_attribution.json is absent: le_point delta not compared with it.")
        gap = None
    else:
        gap = max(abs(d["anchors"]["le_point"]["delta"][k]["max_abs"] - published[d["level"]][k]) for d in data for k in ROWS)
        say(f"le_point max |delta|/c against outputs/transition_study/data/phaseC_attribution.json, largest difference "
            f"over the five levels and four rows: {gap:.3e}.")
    say()
    table(dist_rows, [("level", "level"), ("dm_over_c", "dm/c"), ("n_m", "n_m"), ("n_eta", "n_eta"), ("f", "f"), ("l", "l"),
                      ("f_minus_tip", "(m_f - m_tip)/c"), ("tip_minus_fm1", "(m_tip - m_f-1)/c"),
                      ("tail_minus_l", "(m_tail - m_l)/c"), ("lp1_minus_tail", "(m_l+1 - m_tail)/c"),
                      ("f_minus_le", "(m_f - m_le)/c"), ("te_minus_l", "(m_te - m_l)/c")],
          "Station distances from the anchors", digits=5)
    closeup_figure(OUT / "phaseA_le_te_closeup.png")
    say("Figure: phaseA_le_te_closeup.png (LE and TE close-ups on very_coarse and very_fine).")
    say()
    write_csv("phaseA_ends.csv", end_rows)
    write_csv("phaseA_checks.csv", c_rows)
    write_csv("phaseA_jumps.csv", jump_rows)
    write_csv("phaseA_dbeta_le.csv", le_rows)
    write_csv("phaseA_dbeta_te.csv", te_rows)
    write_csv("phaseA_delta.csv", delta_side)
    write_csv("phaseA_delta_detail.csv", detail_rows)
    write_csv("phaseA_delta_orders.csv", order_rows)
    write_csv("phaseA_distances.csv", dist_rows)
    save("phaseA.json", dict(levels=data, orders=order_rows, transition_study_gap=gap,
                             settings=dict(DENSE_U=DENSE_U, BRENTQ_XTOL=BRENTQ_XTOL, LEVELS=LEVELS)))
    write_lines("phaseA.md")


# --- Phase B: the operator on an exact uniform flow, both anchors -------------------------------------------------------

SIDE_CLASSES = ("f-1", "f", "f+1", "l-1", "l", "l+1", "periodic LE row (f-1, J)", "periodic TE row (l+1, J)",
                "blade near LE (f+2 .. 0.1 c)", "blade near TE (0.1 c .. l-2)", "blade middle", "blade middle, wall-adjacent",
                "periodic near transitions", "periodic far")
TRANSITION_STATIONS = ("f-1", "f", "f+1", "l-1", "l", "l+1")
TERM_COLUMNS = (("C_mm", "C psi_mm"), ("mixed", "-2B psi_me"), ("D_term", "D psi_e"), ("E_term", "E psi_m"),
                ("full", "full"), ("without_DE", "without D, E"))
FIGURE_CLASSES = ("f-1", "f", "l", "l+1", "f+1", "l-1", "blade middle", "periodic far")


def anchored_grid(level: str, anchor: str):
    """The level's grid for an anchor: "le_point" the production case.grid, "tip" tip_grid of it."""
    case = case_of(level)
    if anchor == "le_point":
        return case.grid
    if anchor == "tip":
        return tip_grid(case.grid, case.cascade)
    raise ValueError(f"anchor must be one of {ANCHORS}, got {anchor!r}")


def manufactured_of(grid) -> tuple:
    """transition_study.manufactured_case for a given grid: build_metrics, build_boundary_conditions
    and build_coefficients at rho = 1, mdot' = 1, normalized mode (c_ref None)."""
    from turbolenza.b2b.boundary_conditions import build_boundary_conditions
    from turbolenza.b2b.coefficients import build_coefficients
    from turbolenza.b2b.metrics import build_metrics

    met = build_metrics(grid)
    bcs = build_boundary_conditions(grid)
    co = build_coefficients(met, np.ones((len(grid.m), len(grid.eta))), 1.0, c_ref=None)
    return met, bcs, co


def operator_level(grid, gamma: float, manufactured: tuple | None = None, keep_split: bool = False) -> dict:
    """One grid, one gamma: the body of transition_study.operator_study's loop. psi of the uniform
    flow (ts.uniform_psi), the raw residual split by term (ts.term_split, "expanded"), the class
    statistics (ts.class_stats), the linearity gap of the split on the interior rows (full minus the
    sum of ts.SUMMED, the same check as operator_study) and on the periodic-line rows, and the scale
    max|C| / (length_scale dm)^2. Also returns the full interior field and the coefficients, and
    with keep_split the whole term split (phase D)."""
    met, bcs, co = manufactured_of(grid) if manufactured is None else manufactured
    psi = ts.uniform_psi(grid, gamma)
    split = ts.term_split(co, grid, bcs, psi, "expanded")
    f_ = split["full"][0]
    s_, m_, d_, e_ = (split[t][0] for t in ts.SUMMED)
    ok = np.isfinite(f_)
    gap = float(np.max(np.abs(f_[ok] - (s_[ok] + m_[ok] + d_[ok] + e_[ok])))) if ok.any() else math.nan
    per = split["full"][1]
    gap_periodic = max((abs(per[k][1] - sum(split[t][1][k][1] for t in ts.SUMMED)) for k in per), default=math.nan)
    out = dict(stats=ts.class_stats(split, grid), dm_over_c=grid.dm / grid.chord, deta=grid.deta, split_gap=gap,
               split_gap_periodic=float(gap_periodic), full_max=float(np.nanmax(np.abs(f_))),
               scale=float(np.nanmax(np.abs(np.asarray(co.C)))) / (co.length_scale * grid.dm) ** 2, full=f_, co=co)
    if keep_split:
        out["split"] = split
    return out


def attribution_of(grid, co, full) -> dict:
    """transition_study.corner_attribution for a given grid and its full raw residual (one gamma):
    at f-1, f, l, l+1 the row j of largest |residual|, C delta_j / (pitch dm^2) there (dm =
    co.length_scale * grid.dm) against the residual, and the ratio; the same keys as that function,
    plus the j, J and delta_j / c."""
    y = np.asarray(grid.nodes, float)[..., 1]
    C = np.asarray(co.C)
    dm = co.length_scale * grid.dm
    f, l, J = grid.first_blade_i, grid.last_blade_i, len(grid.eta) - 1
    js = np.arange(1, J)
    row = dict(dm_over_c=grid.dm / grid.chord)
    for name, i, across, own in (("f-1", f - 1, f, (f - 1, f - 2, f - 3)), ("f", f, f - 1, (f, f + 1, f + 2)),
                                 ("l", l, l + 1, (l, l - 1, l - 2)), ("l+1", l + 1, l, (l + 1, l + 2, l + 3))):
        a, b, c3 = own
        delta = y[across, js] - (3 * y[a, js] - 3 * y[b, js] + y[c3, js])
        predicted = C[i, js] * delta / (grid.pitch * dm ** 2)
        actual = full[i, js]
        k = int(np.argmax(np.abs(actual)))
        row[f"{name} delta/c"] = float(np.max(np.abs(delta))) / grid.chord
        row[f"{name} predicted"] = float(predicted[k])
        row[f"{name} actual"] = float(actual[k])
        row[f"{name} ratio"] = float(predicted[k] / actual[k])
        row[f"{name} j"] = int(js[k])
        row[f"{name} delta_j/c"] = float(delta[k]) / grid.chord
    row["J"] = J
    return row


def where_of(grid, full) -> dict:
    """(i, j) of the largest |full residual| of each transition-station class (ts.row_classes)."""
    masks = ts.row_classes(grid)
    f, l, J = grid.first_blade_i, grid.last_blade_i, len(grid.eta) - 1
    out = {}
    for cls in TRANSITION_STATIONS:
        a = np.where(masks[cls], np.abs(full), -1.0)
        i, j = (int(x) for x in np.unravel_index(int(np.argmax(a)), a.shape))
        out[cls] = dict(i=i, j=j, J=J, i_minus_f=i - f, l_minus_i=l - i, value=float(full[i, j]),
                        row="j = 1" if j == 1 else "j = J-1" if j == J - 1 else "interior j")
    return out


def operator_results(anchor: str) -> dict:
    """{(level, gamma): operator_level(...) without the fields, plus "where"; at REPORT_GAMMA_DEG
    also the corner "attribution"} for the five levels and GAMMAS_DEG."""
    results = {}
    for level in LEVELS:
        grid = anchored_grid(level, anchor)
        manufactured = manufactured_of(grid)
        for gamma in GAMMAS_DEG:
            r = operator_level(grid, gamma, manufactured)
            r["where"] = where_of(grid, r["full"])
            if gamma == REPORT_GAMMA_DEG:
                r["attribution"] = dict(level=level, **attribution_of(grid, r["co"], r["full"]))
            del r["full"], r["co"]
            results[(level, gamma)] = r
    return results


def operator_rows(results: dict, anchor: str) -> list:
    """The rows of transition_study.report_operator (max and RMS per class x term x gamma x level,
    order_max by ts.power_law, order_last fine -> very_fine, where_very_fine), without its printing,
    plus the anchor and the power-law order of the RMS. Terms that are exactly zero on a class (A
    psi_ee, E psi_m on the periodic lines) give log(0) = -inf in ts.power_law, which drops them
    (its isfinite mask): the divide warning is silenced here, nothing else changes."""
    with np.errstate(divide="ignore"):
        return _operator_rows(results, anchor)


def _operator_rows(results: dict, anchor: str) -> list:
    rows = []
    for gamma in GAMMAS_DEG:
        dm = [results[(level, gamma)]["dm_over_c"] for level in LEVELS]
        for cls in ts.CLASS_ORDER:
            for term in ts.TERMS:
                maxes = [results[(level, gamma)]["stats"][cls][term]["max"] for level in LEVELS]
                rmss = [results[(level, gamma)]["stats"][cls][term]["rms"] for level in LEVELS]
                fit = ts.power_law(dm, maxes)
                rows.append(dict(anchor=anchor, form="expanded", gamma_deg=gamma, row_class=cls, term=term,
                                 **{f"max_{level}": v for level, v in zip(LEVELS, maxes)},
                                 **{f"rms_{level}": v for level, v in zip(LEVELS, rmss)},
                                 order_max=fit["p"],
                                 order_last=(math.log(abs(maxes[-2]) / abs(maxes[-1])) / math.log(dm[-2] / dm[-1])
                                             if maxes[-1] and maxes[-2] and np.isfinite(maxes[-1]) else math.nan),
                                 order_rms=ts.power_law(dm, rmss)["p"]))
    for r in rows:
        r["where_very_fine"] = results[("very_fine", r["gamma_deg"])]["stats"][r["row_class"]][r["term"]]["where"]
    return rows


def _relative(a, b) -> float:
    """|a - b| / |b|; 0 for two NaNs or two zeros, |a| when b = 0, inf for one NaN."""
    a, b = float(a), float(b)
    if math.isnan(a) and math.isnan(b):
        return 0.0
    if math.isnan(a) or math.isnan(b):
        return math.inf
    if b == 0.0:
        return abs(a)
    return abs(a - b) / abs(b)


def consistency_check(rows: list) -> dict:
    """The le_point rows against outputs/transition_study/data/phaseB.json: every max_*, rms_*,
    order_max and order_last of every class x term x gamma, and the where_very_fine strings."""
    path = ROOT / ts.OUTPUT_DIR / "data" / "phaseB.json"
    if not path.exists():
        return dict(available=False, path=str(path))
    published = {(r["gamma_deg"], r["row_class"], r["term"]): r for r in json.loads(path.read_text(encoding="utf-8"))["rows"]}
    keys = [f"max_{level}" for level in LEVELS] + [f"rms_{level}" for level in LEVELS] + ["order_max", "order_last"]
    worst, where, compared, exact, where_diff = 0.0, None, 0, 0, 0
    ours = {(r["gamma_deg"], r["row_class"], r["term"]): r for r in rows}
    for key, p in published.items():
        r = ours.get(key)
        if r is None:
            return dict(available=True, missing=key)
        for k in keys:
            d = _relative(r[k], p[k])
            compared += 1
            exact += int(d == 0.0)
            if d > worst:
                worst, where = d, (key, k)
        where_diff += int(r["where_very_fine"] != p["where_very_fine"])
    return dict(available=True, rows=len(published), rows_ours=len(ours), values=compared, bit_identical=exact,
                max_relative=worst, worst_at=where, where_strings_differing=where_diff,
                passed=bool(worst <= CONSISTENCY_REL and len(published) == len(ours)))


def attribution_control(attribution: list) -> float | None:
    """le_point attribution against outputs/transition_study/data/phaseC_attribution.json: largest
    relative difference of its delta/c, predicted, actual and ratio (None when absent)."""
    path = ROOT / ts.OUTPUT_DIR / "data" / "phaseC_attribution.json"
    if not path.exists():
        return None
    published = {r["level"]: r for r in json.loads(path.read_text(encoding="utf-8"))}
    return max(_relative(a[f"{k} {field}"], published[a["level"]][f"{k} {field}"]) for a in attribution for k in ROWS
               for field in ("delta/c", "predicted", "actual", "ratio"))


def _cell(value: float, order: float) -> str:
    return f"{value:.4g} ({order:+.2f})"


def phaseB() -> None:
    say("## Phase B: the operator on an exact uniform flow, both anchors (no BCs, no Kutta, no solve)")
    say()
    say("psi = (y - m tan(gamma)) / pitch at the nodes (transition_study.uniform_psi), rho = 1, mdot' = 1, normalized "
        "mode, gamma = " + ", ".join(f"{g:g}" for g in GAMMAS_DEG) + " deg. Residual = the raw left side of each "
        "production PDE row (transition_study.term_split, 'expanded'), row classes and statistics of "
        "transition_study.row_classes / class_stats, orders by transition_study.power_law (|x| = a dm^p, p > 0 decays). "
        "le_point = case.grid (production side lines); tip = tip_grid(case.grid) (phase A). Five levels.")
    say()
    results = {"le_point": operator_results("le_point")}
    rows = {"le_point": operator_rows(results["le_point"], "le_point")}
    check = consistency_check(rows["le_point"])
    if not check.get("available"):
        say(f"CONSISTENCY CHECK NOT POSSIBLE: {check['path']} is absent. Stopped.")
        write_lines("phaseB.md")
        raise SystemExit(1)
    say(f"**Consistency check (before anything else):** the le_point rows against "
        f"outputs/transition_study/data/phaseB.json: {check.get('rows')} rows (ours {check.get('rows_ours')}), "
        f"{check.get('values')} values (max_*, rms_*, order_max, order_last), {check.get('bit_identical')} bit-identical, "
        f"largest relative difference {check.get('max_relative', math.nan):.3e} (at {check.get('worst_at')}); "
        f"where_very_fine strings differing: {check.get('where_strings_differing')}. Bound {CONSISTENCY_REL:g}: "
        f"{'PASSED' if check.get('passed') else 'FAILED'}.")
    say()
    if not check.get("passed"):
        say("Stopped: the le_point operator rows do not reproduce the transition study.")
        save("phaseB.json", dict(consistency=check))
        write_lines("phaseB.md")
        raise SystemExit(1)
    results["tip"] = operator_results("tip")
    rows["tip"] = operator_rows(results["tip"], "tip")
    by = {(r["anchor"], r["gamma_deg"], r["row_class"], r["term"]): r for a in ANCHORS for r in rows[a]}
    gaps = {a: max(r["split_gap"] for r in results[a].values()) for a in ANCHORS}
    gaps_per = {a: max(r["split_gap_periodic"] for r in results[a].values()) for a in ANCHORS}
    full_max = {a: max(r["full_max"] for r in results[a].values()) for a in ANCHORS}
    for a in ANCHORS:
        say(f"Linearity check of the term split ({a}): full - (second + mixed + D + E), largest over the interior rows, "
            f"levels and gammas {gaps[a]:.2e}, over the periodic-line rows {gaps_per[a]:.2e} (raw units; the full "
            f"residuals reach {full_max[a]:.4g}).")
    say()
    g0 = REPORT_GAMMA_DEG
    # (1) the side-by-side tables
    for a in ANCHORS:
        sel = [by[(a, g0, cls, "full")] for cls in SIDE_CLASSES]
        table(sel, [("row_class", "row class")] + [(f"max_{level}", level) for level in LEVELS]
              + [("order_max", "order (fit)"), ("order_last", "order f->vf"), ("order_rms", "RMS order (fit)"),
                 ("where_very_fine", "max at (very_fine)")],
              f"(1) {a}: max |raw residual| of the full row, gamma = {g0:g} deg", digits=3)
    side = []
    for cls in SIDE_CLASSES:
        le, tp = by[("le_point", g0, cls, "full")], by[("tip", g0, cls, "full")]
        side.append(dict(row_class=cls, le_vc=le["max_very_coarse"], tip_vc=tp["max_very_coarse"],
                         le_vf=le["max_very_fine"], tip_vf=tp["max_very_fine"], le_order=le["order_max"],
                         tip_order=tp["order_max"], order_change=tp["order_max"] - le["order_max"],
                         le_rms_order=le["order_rms"], tip_rms_order=tp["order_rms"]))
    table(side, [("row_class", "row class"), ("le_vc", "le_point vc"), ("tip_vc", "tip vc"), ("le_vf", "le_point vf"),
                 ("tip_vf", "tip vf"), ("le_order", "le_point order"), ("tip_order", "tip order"),
                 ("order_change", "tip - le_point"), ("le_rms_order", "le_point RMS order"),
                 ("tip_rms_order", "tip RMS order")],
          f"(1) side by side, gamma = {g0:g} deg (max |raw residual|; orders by the five-level fit)", digits=3)
    # (2) the term tables
    term_rows = []
    for a in ANCHORS:
        cells = []
        for cls in TRANSITION_STATIONS:
            cell = dict(row_class=cls)
            for term, _ in TERM_COLUMNS:
                r = by[(a, g0, cls, term)]
                cell[term] = _cell(r["max_very_fine"], r["order_max"])
            cells.append(cell)
        table(cells, [("row_class", "row class")] + list(TERM_COLUMNS),
              f"(2) {a}: terms at the transition stations, max |.| on very_fine, gamma = {g0:g} deg (order by the "
              f"five-level fit in brackets)")
    for a in ANCHORS:
        for gamma in GAMMAS_DEG:
            for cls in TRANSITION_STATIONS:
                for term, name in TERM_COLUMNS:
                    r = by[(a, gamma, cls, term)]
                    term_rows.append(dict(anchor=a, gamma_deg=gamma, row_class=cls, term=term, term_name=name,
                                          **{f"max_{level}": r[f"max_{level}"] for level in LEVELS},
                                          order_max=r["order_max"], order_last=r["order_last"]))
    # (3) the delta attribution
    attribution = {a: [results[a][(level, g0)]["attribution"] for level in LEVELS] for a in ANCHORS}
    att_rows = []
    for a in ("tip", "le_point"):
        long = []
        for r in attribution[a]:
            for k in ROWS:
                long.append(dict(anchor=a, level=r["level"], dm_over_c=r["dm_over_c"], row=k, delta_max=r[f"{k} delta/c"],
                                 j=r[f"{k} j"], J=r["J"], delta_j=r[f"{k} delta_j/c"], predicted=r[f"{k} predicted"],
                                 actual=r[f"{k} actual"], ratio=r[f"{k} ratio"]))
        att_rows += long
        table(long, [("level", "level"), ("row", "row"), ("delta_max", "max |delta|/c"), ("j", "j of max |residual|"),
                     ("J", "J"), ("delta_j", "delta_j/c"), ("predicted", "C delta_j/(s dm^2)"), ("actual", "residual"),
                     ("ratio", "ratio")],
              f"(3) {a}: the eta-line corner against the residual at each transition row's j of largest |residual| "
              f"(gamma = {g0:g} deg)", digits=4)
    control = attribution_control(attribution["le_point"])
    say("le_point attribution against outputs/transition_study/data/phaseC_attribution.json (delta/c, predicted, actual, "
        "ratio): " + ("absent, not compared." if control is None else f"largest relative difference {control:.3e}."))
    say()
    # (4) where the transition-class maxima sit on very_fine
    where_rows = []
    for gamma in GAMMAS_DEG:
        for a in ANCHORS:
            w = results[a][("very_fine", gamma)]["where"]
            for cls in TRANSITION_STATIONS:
                where_rows.append(dict(gamma_deg=gamma, anchor=a, row_class=cls, **w[cls]))
    table([r for r in where_rows if r["gamma_deg"] == g0], [("anchor", "anchor"), ("row_class", "row class"), ("i", "i"),
                                                            ("j", "j"), ("J", "J"), ("row", "row"), ("i_minus_f", "i - f"),
                                                            ("l_minus_i", "l - i"), ("value", "residual")],
          f"(4) Where the max |full residual| of each transition class sits on very_fine (gamma = {g0:g} deg)", digits=4)
    at_g0 = {(r["anchor"], r["row_class"]): (r["i"], r["j"]) for r in where_rows if r["gamma_deg"] == g0}
    moved = [r for r in where_rows if r["gamma_deg"] != g0 and (r["i"], r["j"]) != at_g0[(r["anchor"], r["row_class"])]]
    say(f"At gamma {', '.join(f'{g:g}' for g in GAMMAS_DEG if g != g0)} deg the maxima sit at the same (i, j) as at "
        f"{g0:g} deg, except: " + ("; ".join(f"{r['anchor']} {r['row_class']} ({r['i']}, {r['j']}) at {r['gamma_deg']:g}"
                                            for r in moved) if moved else "none") + ".")
    say()
    # (6) gate B
    gate_rows = []
    for a in ("tip", "le_point"):
        for gamma in GAMMAS_DEG:
            for cls in ROWS:
                p = by[(a, gamma, cls, "full")]["order_max"]
                gate_rows.append(dict(anchor=a, gamma_deg=gamma, row_class=cls, order_max=p, at_least_gate=bool(p >= GATE_B_ORDER),
                                      in_gate=a == "tip", in_le_only_reading=a == "tip" and cls in ("f-1", "f")))
    tip_gate = [r for r in gate_rows if r["in_gate"]]
    le_only = [r for r in gate_rows if r["in_le_only_reading"]]
    gate_pass = all(r["order_max"] >= GATE_B_ORDER for r in tip_gate)
    le_only_pass = all(r["order_max"] >= GATE_B_ORDER for r in le_only)
    say(f"### Gate B (the plan, read as written): order_max of max |raw residual| (five-level power-law fit) of f-1, f, "
        f"l and l+1, at gamma = {', '.join(f'{g:g}' for g in GAMMAS_DEG)} deg, all twelve with tip, must be >= "
        f"{GATE_B_ORDER:g}")
    say()
    say("| gamma | f-1 | f | l | l+1 |")
    say("|---|---|---|---|---|")
    for gamma in GAMMAS_DEG:
        say(f"| {gamma:g} | " + " | ".join(f"{by[('tip', gamma, cls, 'full')]['order_max']:.3f}" for cls in ROWS) + " |")
    say()
    say(f"GATE B: {'PASS' if gate_pass else 'FAIL'} (all twelve >= {GATE_B_ORDER:g}: {gate_pass}; the largest "
        f"{max(r['order_max'] for r in tip_gate):.3f}, the smallest {min(r['order_max'] for r in tip_gate):.3f}; "
        f"{sum(r['at_least_gate'] for r in tip_gate)} of 12 at or above).")
    say()
    say(f"LE-only reading (f-1, f; NOT the gate): " + ", ".join(
        f"{r['row_class']} @ {r['gamma_deg']:g}: {r['order_max']:.3f}" for r in le_only)
        + f" -> all >= {GATE_B_ORDER:g}: {le_only_pass}.")
    say()
    say("For reference, le_point (production) at the same rows: " + "; ".join(
        f"gamma {gamma:g}: " + ", ".join(f"{by[('le_point', gamma, cls, 'full')]['order_max']:.3f}" for cls in ROWS)
        for gamma in GAMMAS_DEG) + " (f-1, f, l, l+1).")
    say()
    # (5) the figure
    residual_figure(by, OUT / "residual_by_class_vs_dm.png")
    say("Figure: residual_by_class_vs_dm.png (max |raw residual| against dm, gamma 33 deg; le_point solid, tip dashed; "
        "reference slopes dm^-1.5 and dm^-2).")
    say()
    write_csv("phaseB_operator_residuals.csv", rows["le_point"] + rows["tip"])
    write_csv("phaseB_terms.csv", term_rows)
    write_csv("phaseB_corner_attribution.csv", att_rows)
    write_csv("phaseB_gate.csv", gate_rows)
    save("phaseB.json", dict(consistency=check, rows=rows["le_point"] + rows["tip"], side_by_side=side, terms=term_rows,
                             attribution=attribution, attribution_control=control, where=where_rows, gate=gate_rows,
                             gate_pass=gate_pass, le_only_pass=le_only_pass, split_gap=gaps, split_gap_periodic=gaps_per,
                             full_max=full_max, settings=dict(GAMMAS_DEG=GAMMAS_DEG, REPORT_GAMMA_DEG=REPORT_GAMMA_DEG,
                                                              GATE_B_ORDER=GATE_B_ORDER, CONSISTENCY_REL=CONSISTENCY_REL,
                                                              LEVELS=LEVELS)))
    write_lines("phaseB.md")


def residual_figure(by: dict, path: Path) -> None:
    """max |raw residual| against dm (log-log), gamma REPORT_GAMMA_DEG, FIGURE_CLASSES: le_point solid,
    tip dashed, with dm^-1.5 and dm^-2 reference slopes."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g0 = REPORT_GAMMA_DEG
    dm = np.array([case_of(level).grid.dm / case_of(level).grid.chord for level in LEVELS])
    colours = ks.SERIES + ("#6250d6", "#e34948", ks.INK)
    fig, ax = plt.subplots(figsize=(9.4, 5.8), facecolor=ks.SURFACE)
    ks.style(ax, f"Phase B: max |raw residual| on the uniform flow (gamma {g0:g} deg); solid le_point, dashed tip")
    for k, cls in enumerate(FIGURE_CLASSES):
        for a, dashed in (("le_point", False), ("tip", True)):
            r = by[(a, g0, cls, "full")]
            ax.loglog(dm, [r[f"max_{level}"] for level in LEVELS], color=colours[k], linestyle="--" if dashed else "-",
                      linewidth=1.4 if dashed else 2.0, marker=ks.MARKS[k % len(ks.MARKS)], markersize=5,
                      markerfacecolor=ks.SURFACE if dashed else colours[k],
                      label=f"{cls}  (le_point p = {by[('le_point', g0, cls, 'full')]['order_max']:.2f}, "
                            f"tip p = {by[('tip', g0, cls, 'full')]['order_max']:.2f})" if not dashed else None)
    top = by[("le_point", g0, "f-1", "full")]["max_very_coarse"]
    for p, style in ((-1.5, ":"), (-2.0, "-.")):
        ax.loglog(dm, 0.3 * top * (dm / dm[0]) ** p, color=ks.MUTED, linewidth=1.0, linestyle=style,
                  label=f"slope dm^{p:g} (reference)")
    ax.set_xlabel("dm / c")
    ax.set_ylabel("max |raw residual|")
    ax.invert_xaxis()
    ax.legend(fontsize=6.8, frameon=False, labelcolor=ks.INK2, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=ks.SURFACE)
    plt.close(fig)


# --- Phase D1: curvature along the C2 surfaces and the construction joints -----------------------------------------------

def chordwise_of(cascade, points) -> np.ndarray:
    """x/c of points of the reference blade: their projection on the chord axis LE -> TE (as
    kutta_study.chordwise, from cascade.active's LE and TE points)."""
    le = np.asarray(cascade.active.leading_edge_point, float)
    te = np.asarray(cascade.active.trailing_edge_point, float)
    axis = te - le
    return (np.atleast_2d(np.asarray(points, float)) - le) @ axis / float(axis @ axis)


def curvature_derivatives(d1, d2, d3, d4) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """kappa, dkappa/ds and d2kappa/ds2 of X(u) from its exact u derivatives X' .. X'''' (no
    differencing): with N = X' x X'', P = X'.X'', v = |X'|,
        kappa = N / v^3,  kappa_u = N_u / v^3 - 3 N P / v^5,
        kappa_uu = N_uu / v^3 - (6 N_u P + 3 N P_u) / v^5 + 15 N P^2 / v^7,
        kappa_s = kappa_u / v,  kappa_ss = kappa_uu / v^2 - kappa_u P / v^4,
    N_u = X' x X''', N_uu = X'' x X''' + X' x X'''', P_u = X''.X'' + X'.X'''. kappa has the sign of
    geometry.signed_curvature."""
    d1, d2, d3, d4 = (np.asarray(a, float) for a in (d1, d2, d3, d4))

    def cross(a, b):
        return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]

    def dot(a, b):
        return a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1]

    v = np.hypot(d1[..., 0], d1[..., 1])
    N, N_u, N_uu = cross(d1, d2), cross(d1, d3), cross(d2, d3) + cross(d1, d4)
    P, P_u = dot(d1, d2), dot(d2, d2) + dot(d1, d3)
    kappa = N / v ** 3
    kappa_u = N_u / v ** 3 - 3.0 * N * P / v ** 5
    kappa_uu = N_uu / v ** 3 - (6.0 * N_u * P + 3.0 * N * P_u) / v ** 5 + 15.0 * N * P ** 2 / v ** 7
    return kappa, kappa_u / v, kappa_uu / v ** 2 - kappa_u * P / v ** 4


def curve_curvature(curve, u) -> dict:
    """kappa, kappa_s, kappa_ss, the point and |X'| of a ParametricBladeCurve at u (its own
    derivatives, orders 1 to 4)."""
    u = np.atleast_1d(np.asarray(u, float))
    d = [np.asarray(curve.derivative(u, k), float) for k in range(5)]
    kappa, kappa_s, kappa_ss = curvature_derivatives(d[1], d[2], d[3], d[4])
    return dict(point=d[0], speed=np.hypot(d[1][:, 0], d[1][:, 1]), kappa=kappa, kappa_s=kappa_s, kappa_ss=kappa_ss)


def _piece_values(pps, u: float, interval: int) -> list:
    """X^(n)(u), n = 1 .. 5, from one polynomial piece of the two coordinate PPolys."""
    out = []
    for n in range(1, 6):
        out.append(np.array([np.polyval(np.polyder(pp.c[:, interval], n), u - pp.x[interval]) for pp in pps]))
    return out


def knot_jumps(curve, u_knot: float, pps) -> dict:
    """Jumps (right minus left polynomial piece, evaluated exactly at the knot) of kappa,
    kappa_s, kappa_ss and of X^(5), the first derivative a simple knot of the quintic may break."""
    x = pps[0].x
    i = int(np.searchsorted(x, u_knot, side="left"))
    if not (i < len(x) and x[i] == u_knot):
        raise ValueError(f"{u_knot!r} is not a breakpoint of the curve")
    sides = {}
    for side, k in (("left", i - 1), ("right", i)):
        d = _piece_values(pps, u_knot, k)
        kappa, kappa_s, kappa_ss = curvature_derivatives(d[0], d[1], d[2], d[3])
        sides[side] = dict(kappa=float(kappa), kappa_s=float(kappa_s), kappa_ss=float(kappa_ss), d5=d[4])
    return dict(kappa=sides["right"]["kappa"] - sides["left"]["kappa"],
                kappa_s=sides["right"]["kappa_s"] - sides["left"]["kappa_s"],
                kappa_ss=sides["right"]["kappa_ss"] - sides["left"]["kappa_ss"],
                d5=float(np.hypot(*(sides["right"]["d5"] - sides["left"]["d5"]))))


def c2_fits(cascade) -> tuple:
    """The two C2SurfaceFit objects of the cascade (geometry._naca65_quintic_fits, the cached
    construction make_cascade used), checked to carry the very knots of cascade.active_curves."""
    from turbolenza.b2b import geometry as ge

    spec = cascade.spec
    reconstruction = spec.thickness_reconstruction or ge.DEFAULT_THICKNESS_RECONSTRUCTION[spec.surface_continuity]
    fits = ge._naca65_quintic_fits(spec.cl0, spec.t_max_over_c, spec.profile_variant, reconstruction)
    for name, fit in zip(("suction", "pressure"), fits):
        if not np.array_equal(fit.curve.knots, getattr(cascade.active_curves, name).knots):
            raise RuntimeError(f"the {name} fit does not carry the knots of cascade.active_curves")
    return fits


def construction_joints(cascade) -> list:
    """Every joint of the C2 construction of both surfaces (module docstring of geometry.py,
    "Surface continuity" and "The C2 leading edge"), in the curve parameter u:
    - the LE fair: where it leaves the nose circle (phi = min_arc, G4 by construction), its cubic
      r(phi) knots, the departure (curvature 0.1 % below 1/R_LE: a threshold, not a joint) and the
      anchor (G2 join with the source);
    - the reference junctions downstream of the anchor (thickness table stations, the Kovach start,
      the Kovach TE fair and TE circle starts), mapped by C2SurfaceFit.parameter; those inside the
      LE zone are replaced by the C2 LE and are not on the curve;
    - the ends of each smoothed transition (junction s -+ a) and of its shift compensation
      (-+ C2_COMPENSATION_RATIO a);
    and, per joint, the one-sided kappa, kappa_s, kappa_ss at u -+ JOINT_STEP_U and their jumps,
    its x/c and the distance to the nearest interior knot of the quintic."""
    from turbolenza.b2b import geometry as ge

    out = []
    for name, fit in zip(("suction", "pressure"), c2_fits(cascade)):
        curve = getattr(cascade.active_curves, name)
        le, L = fit.leading_edge, fit.length
        knots = np.asarray(curve.interior_knots)
        dense = np.union1d(np.linspace(0.0, 1.0, DENSE_CURVE_U), knots)
        arc = _arc_length(curve, dense)
        items = [("LE circle end (phi = 20 deg)", "LE fair", float(le.arc_length(le.min_arc)[0]) / L, math.nan)]
        edges = np.linspace(0.0, 1.0, ge.LE_FAIR_SPANS + 1)[1:-1]
        for k, eta in enumerate(edges, 1):
            phi = le.min_arc + eta * (le.anchor_angle - le.min_arc)
            items.append((f"LE fair r(phi) knot {k}", "LE fair knot", float(le.arc_length(phi)[0]) / L, math.nan))
        items.append(("LE departure (kappa 0.1 % below 1/R_LE)", "LE marker", fit.departure_u, math.nan))
        items.append(("LE anchor (G2 join of fair and source)", "LE anchor", fit.anchor_u, math.nan))
        for j in fit.reference.junctions:
            if j.s > le.anchor:
                kind = ("Kovach start" if j.label.startswith("Kovach start") else "TE" if j.label.startswith("Kovach TE")
                        else "table station")
                items.append((j.label, kind, float(fit.parameter(j.s)), j.kappa_jump))
        for t in fit.source.transitions:
            for factor, what in ((1.0, "transition"), (ge.C2_COMPENSATION_RATIO, "compensation")):
                for sign, end in ((-1.0, "start"), (1.0, "end")):
                    s = t.junction.s + sign * factor * t.half_width
                    if le.anchor < s < fit.reference.length:
                        items.append((f"{t.junction.label}: {what} {end} (a = {t.half_width:.3g} c)", f"{what} end",
                                      float(fit.parameter(s)), math.nan))
        for label, kind, u, c1_jump in items:
            values = curve_curvature(curve, [u - JOINT_STEP_U, u, u + JOINT_STEP_U])
            nearest = float(knots[int(np.argmin(np.abs(knots - u)))])
            out.append(dict(surface=name, label=label, kind=kind, u=u, x_c=float(chordwise_of(cascade, values["point"][1])[0]),
                            s_over_c=float(np.interp(u, dense, arc)) / cascade.active.chord,
                            m_over_c=float(values["point"][1, 0]) / cascade.active.chord,
                            kappa=float(values["kappa"][1]), kappa_s=float(values["kappa_s"][1]),
                            kappa_ss=float(values["kappa_ss"][1]),
                            jump_kappa=float(values["kappa"][2] - values["kappa"][0]),
                            jump_kappa_s=float(values["kappa_s"][2] - values["kappa_s"][0]),
                            jump_kappa_ss=float(values["kappa_ss"][2] - values["kappa_ss"][0]),
                            smooth_bound_kappa=2.0 * JOINT_STEP_U * float(values["speed"][1] * abs(values["kappa_s"][1])),
                            smooth_bound_kappa_s=2.0 * JOINT_STEP_U * float(values["speed"][1] * abs(values["kappa_ss"][1])),
                            nearest_knot_du=abs(nearest - u), on_knot=bool(nearest == u), c1_kappa_jump=c1_jump))
    return out


def knot_statistics(cascade) -> dict:
    """Over every interior knot of each quintic: the largest jumps of kappa, kappa_s, kappa_ss and
    X^(5) between the two polynomial pieces at the knot (exact piece values), and the knots."""
    from scipy.interpolate import BSpline, PPoly

    out = {}
    for name in ("suction", "pressure"):
        curve = getattr(cascade.active_curves, name)
        pps = [PPoly.from_spline(BSpline(curve.knots, curve.control_points[:, a], curve.degree)) for a in (0, 1)]
        knots = np.asarray(curve.interior_knots)
        jumps = [knot_jumps(curve, float(u), pps) for u in knots]
        points = curve.evaluate(knots)
        out[name] = dict(n=len(knots), u=knots, x_c=chordwise_of(cascade, points), m=points[:, 0],
                         **{f"max_jump_{k}": float(max(abs(j[k]) for j in jumps)) for k in ("kappa", "kappa_s", "kappa_ss")},
                         max_jump_d5=float(max(j["d5"] for j in jumps)), jumps=jumps)
    return out


def derivative_selfcheck(cascade) -> dict:
    """curvature_derivatives against geometry.signed_curvature (kappa) and against central
    differences in arc length of its own kappa and kappa_s on DENSE_CURVE_U samples (away from the
    curve ends): a check of the formulas, not used for any result."""
    from turbolenza.b2b import geometry as ge

    out = {}
    for name in ("suction", "pressure"):
        curve = getattr(cascade.active_curves, name)
        u = np.linspace(0.0, 1.0, DENSE_CURVE_U)
        v = curve_curvature(curve, u)
        kappa_ref = ge.signed_curvature(curve.derivative(u, 1), curve.derivative(u, 2))
        ds = np.diff(_arc_length(curve, u))
        h1, h0 = ds[1:], ds[:-1]
        inner = slice(1, -1)

        def central(f):
            return (f[2:] * h0 ** 2 - f[:-2] * h1 ** 2 + f[1:-1] * (h1 ** 2 - h0 ** 2)) / (h0 * h1 * (h0 + h1))

        keep = (u[inner] > 0.03) & (u[inner] < 0.97)
        fd_s, fd_ss = central(v["kappa"]), central(v["kappa_s"])
        scale_s, scale_ss = np.max(np.abs(v["kappa_s"][inner][keep])), np.max(np.abs(v["kappa_ss"][inner][keep]))
        out[name] = dict(kappa_vs_signed_curvature=float(np.max(np.abs(v["kappa"] - kappa_ref))),
                         kappa_s_vs_difference_rel=float(np.max(np.abs(fd_s - v["kappa_s"][inner])[keep])) / scale_s,
                         kappa_ss_vs_difference_rel=float(np.max(np.abs(fd_ss - v["kappa_ss"][inner])[keep])) / scale_ss)
    return out


def _arc_length(curve, u) -> np.ndarray:
    """Arc length from u = 0 by the trapezoidal rule of |X'| on the given (dense) u."""
    speed = np.hypot(*np.asarray(curve.derivative(u, 1), float).T)
    return np.concatenate(([0.0], np.cumsum(0.5 * (speed[1:] + speed[:-1]) * np.diff(u))))


def blade_middle_rows(knots: dict, joints: list) -> list:
    """Per level and anchor (gamma REPORT_GAMMA_DEG): the (i, j) of the "blade middle" and "blade
    middle, wall-adjacent" maxima of |full residual|, the adjacent wall (j <= J/2: suction, j = 0
    side; else pressure, j = J side), the x/c of that wall's node at station i, the m distance to
    the nearest construction joint and to the nearest knot of that surface (in dm), and the term
    split at the node."""
    rows = []
    for level in LEVELS:
        case = case_of(level)
        for anchor in ANCHORS:
            grid = anchored_grid(level, anchor)
            r = operator_level(grid, REPORT_GAMMA_DEG, keep_split=True)
            masks = ts.row_classes(grid)
            J, dm, c = len(grid.eta) - 1, grid.dm, grid.chord
            for cls in ("blade middle", "blade middle, wall-adjacent"):
                a = np.where(masks[cls], np.abs(r["full"]), -1.0)
                i, j = (int(x) for x in np.unravel_index(int(np.argmax(a)), a.shape))
                surface = "suction" if j <= J / 2 else "pressure"
                node = (np.asarray(grid.nodes[i, 0], float) if surface == "suction"
                        else np.asarray(grid.nodes[i, J], float) - np.asarray(grid.boundaries.periodic_translation))
                own = [jt for jt in joints if jt["surface"] == surface and jt["kind"] not in ("LE fair knot", "LE marker")]
                near = min(own, key=lambda jt: abs(jt["m_over_c"] * c - grid.m[i]))
                k_m = knots[surface]["m"]
                kk = int(np.argmin(np.abs(k_m - grid.m[i])))
                ss_node = np.asarray(grid.nodes[i, 0], float)
                ps_node = np.asarray(grid.nodes[i, J], float) - np.asarray(grid.boundaries.periodic_translation)
                row = dict(level=level, anchor=anchor, row_class=cls, i=i, j=j, J=J, wall=surface,
                           dm_over_c=dm / c, m_over_c=grid.m[i] / c, x_c=float(chordwise_of(case.cascade, node)[0]),
                           x_c_ss_node=float(chordwise_of(case.cascade, ss_node)[0]),
                           x_c_ps_node=float(chordwise_of(case.cascade, ps_node)[0]),
                           nearest_joint=near["label"], nearest_joint_x_c=near["x_c"],
                           joint_distance_dm=(grid.m[i] - near["m_over_c"] * c) / dm,
                           nearest_knot_x_c=float(knots[surface]["x_c"][kk]),
                           knot_distance_dm=(grid.m[i] - k_m[kk]) / dm,
                           residual=float(r["full"][i, j]))
                for term in ("A_ee", "C_mm", "mixed", "D_term", "E_term"):
                    row[term] = float(r["split"][term][0][i, j])
                rows.append(row)
    return rows


def stokes_character() -> dict | None:
    """The source looked for behind the brief's "x/c 0.66": the kutta study's Stokes check
    (outputs/kutta_study/stokes_check.csv; its phase 2: very_fine, station K, converged), its
    elements by suction-side x/c with err_cos = LHS - RHS_cos. Read only; None when absent."""
    path = ROOT / STOKES_CSV
    if not path.exists():
        return None
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    elements = [dict(element=r["element"], x_c=float(r["x_c_ss_mid"]), err_cos=float(r["err_cos"]), lhs=float(r["lhs"]))
                for r in rows]
    window = [e for e in elements if 0.55 <= e["x_c"] <= 0.75]
    return dict(path=STOKES_CSV, elements=len(elements), window=window)


def curvature_figure(cascade, joints: list, middle: list, path: Path) -> None:
    """kappa and dkappa/ds (exact) against x/c for both C2 surfaces, the construction joints marked,
    and a band at the blade-middle max x/c of each level (on the surface it lies next to)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c = cascade.active.chord
    colours = {"LE fair": ks.SERIES[4], "LE anchor": ks.SERIES[4], "table station": ks.MUTED,
               "Kovach start": ks.SERIES[2], "TE": ks.SERIES[3], "transition end": ks.SERIES[2],
               "compensation end": ks.SERIES[5]}
    fig, axes = plt.subplots(3, 2, figsize=(12.0, 10.5), facecolor=ks.SURFACE)
    for col, name in enumerate(("suction", "pressure")):
        curve = getattr(cascade.active_curves, name)
        u = np.linspace(0.0, 1.0, DENSE_CURVE_U)
        v = curve_curvature(curve, u)
        x = chordwise_of(cascade, v["point"])
        for row, (key, label, scale, xlim, log) in enumerate((("kappa", "kappa * c", c, (0.0, 1.0), False),
                                                              ("kappa_s", "dkappa/ds * c^2", c * c, (0.0, 1.0), True),
                                                              ("kappa_s", "dkappa/ds * c^2 (x/c 0.5 .. 0.8)", c * c,
                                                               (0.5, 0.8), False))):
            ax = axes[row, col]
            ks.style(ax, f"{name} surface: {label}")
            sel = (x >= xlim[0]) & (x <= xlim[1])
            ax.plot(x[sel], v[key][sel] * scale, color=ks.INK, linewidth=1.0)
            if log:
                ax.set_yscale("symlog", linthresh=1.0)
            for jt in joints:
                if jt["surface"] == name and jt["kind"] in colours and xlim[0] <= jt["x_c"] <= xlim[1]:
                    ax.axvline(jt["x_c"], color=colours[jt["kind"]], linewidth=0.7,
                               linestyle=":" if jt["kind"] == "table station" else "--")
            for k, level in enumerate(LEVELS):
                for r in middle:
                    if r["level"] == level and r["anchor"] == "le_point" and r["wall"] == name:
                        half = 0.5 * r["dm_over_c"]
                        ax.axvspan(r["x_c"] - half, r["x_c"] + half, color=ks.LEVEL_RAMP[k], alpha=0.25, linewidth=0)
            for mark in X_C_MARKS:
                if xlim[0] <= mark <= xlim[1]:
                    ax.axvline(mark, color=ks.SERIES[1], linewidth=0.8, linestyle="-.")
            ax.set_xlim(*xlim)
            ax.set_xlabel("x / c")
    handles = [matplotlib.lines.Line2D([], [], color=colours[k], linestyle=":" if k == "table station" else "--", label=k)
               for k in ("LE anchor", "table station", "Kovach start", "transition end", "compensation end", "TE")]
    handles.append(matplotlib.lines.Line2D([], [], color=ks.SERIES[1], linestyle="-.", label="x/c 0.64, 0.66 (brief)"))
    handles += [matplotlib.patches.Patch(color=ks.LEVEL_RAMP[k], alpha=0.4, label=f"blade-middle max, {level} (+- dm/2)")
                for k, level in enumerate(LEVELS)]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=7.5, frameon=False, labelcolor=ks.INK2)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, dpi=150, facecolor=ks.SURFACE)
    plt.close(fig)


# --- Phase D2: the first-order term on the periodic rows ----------------------------------------------------------------

def side_line_exact(junction, junction_angle: float, far_m: float, far_angle: float, m) -> dict:
    """The side line in closed form, written out here (not grid.py's _side_line): beta linear in m,
    beta = junction_angle + (m - m_J) rate, rate = (far_angle - junction_angle) / (far_m - m_J),
    y0 = y_J + ln(cos(junction_angle) / cos(beta)) / rate (the integral of tan(beta); y_J + (m - m_J)
    tan(junction_angle) for rate 0), tan(beta), cos(beta), d tan(beta)/dm = sec^2(beta) rate and
    d2 tan(beta)/dm2 = 2 sec^2(beta) tan(beta) rate^2 (= d3 y0/dm3)."""
    m = np.asarray(m, float)
    junction = np.asarray(junction, float)
    rate = (far_angle - junction_angle) / (far_m - junction[0])
    beta = junction_angle + (m - junction[0]) * rate
    tan, cos = np.tan(beta), np.cos(beta)
    if rate == 0.0:
        y = junction[1] + (m - junction[0]) * math.tan(junction_angle)
    else:
        y = junction[1] + np.log(math.cos(junction_angle) / cos) / rate
    sec2 = 1.0 / cos ** 2
    return dict(beta=beta, tan=tan, cos=cos, y=y, rate=rate, dtan=sec2 * rate, d2tan=2.0 * sec2 * tan * rate ** 2)


def side_junctions(grid, cascade, anchor: str) -> dict:
    """(junction point, junction angle, far m, far angle) of the inlet and exit side lines of an
    anchor: le_point the LE / TE point of cascade.active, tip the nose tip / tail end."""
    b = grid.boundaries
    if anchor == "le_point":
        inlet_j, exit_j = cascade.active.leading_edge_point, cascade.active.trailing_edge_point
    elif anchor == "tip":
        ends = tip_and_tail(cascade)
        inlet_j, exit_j = ends["tip"]["point"], ends["tail"]["point"]
    else:
        raise ValueError(f"anchor must be one of {ANCHORS}, got {anchor!r}")
    return dict(inlet=(np.asarray(inlet_j, float), float(cascade.kappa_le), b.m_in, b.inlet_angle),
                exit=(np.asarray(exit_j, float), float(cascade.kappa_te), b.m_out, b.exit_angle))


def periodic_exact(grid, cascade, anchor: str) -> dict:
    """side_line_exact at every station, NaN on the blade stations: the inlet law on 0 .. f-1, the
    exit law on l+1 .. n_m-1."""
    f, l, n = grid.first_blade_i, grid.last_blade_i, len(grid.m)
    m = np.asarray(grid.m, float)
    junctions = side_junctions(grid, cascade, anchor)
    out = {k: np.full(n, np.nan) for k in ("beta", "tan", "cos", "y", "dtan", "d2tan")}
    rates = {}
    for side, stations in (("inlet", slice(0, f)), ("exit", slice(l + 1, n))):
        ex = side_line_exact(*junctions[side], m[stations])
        rates[side] = ex["rate"]
        for k in out:
            out[k][stations] = ex[k]
    out["rate"] = rates
    out["periodic"] = np.ones(n, bool)
    out["periodic"][f:l + 1] = False
    return out


def exact_metrics_of(met, ex: dict):
    """A copy of the GridMetrics (dataclasses.replace) with the closed-form side-line fields on the
    periodic stations: S = pitch, dS/dm = 0, tan(beta0) = tan(beta1) = tan(beta) (every eta),
    beta, sin, cos, d tan(beta)/dm; the blade stations as they are."""
    p = ex["periodic"]
    grid = met.grid
    n_eta = len(grid.eta)
    S = np.array(met.S)
    S[p] = grid.pitch
    dS = np.array(met.dS_dm)
    dS[p] = 0.0
    tan0, tan1 = np.array(met.tan_beta0), np.array(met.tan_beta1)
    tan0[p] = tan1[p] = ex["tan"][p]
    d0, d1 = np.array(met.dtan_beta0_dm), np.array(met.dtan_beta1_dm)
    d0[p] = d1[p] = ex["dtan"][p]
    fields2 = {}
    for name, value in (("tan_beta", ex["tan"]), ("beta", ex["beta"]), ("sin_beta", np.sin(ex["beta"])),
                        ("cos_beta", ex["cos"]), ("dtan_beta_dm", ex["dtan"])):
        a = np.array(getattr(met, name))
        a[p] = np.repeat(value[p][:, None], n_eta, axis=1)
        fields2[name] = a
    return replace(met, S=S, dS_dm=dS, tan_beta0=tan0, tan_beta1=tan1, dtan_beta0_dm=d0, dtan_beta1_dm=d1, **fields2)


def exact_coefficients_of(co, ex: dict, pitch: float) -> dict:
    """A .. E with the closed forms on the periodic stations (rho = 1, mdot' = 1, normalized mode):
    A = 1 / (pitch cos^2 beta), B = tan beta, C = pitch, D = A_eta - B_m = - d tan(beta)/dm, E = C_m -
    B_eta = 0 [Eqs. 5-22 to 5-26]; the blade stations keep the discrete fields."""
    if co.mass_flow_per_depth != 1.0 or co.c_ref is not None:
        raise ValueError("the closed forms here are for mdot' = 1 in normalized mode")
    p = ex["periodic"]
    out = {k: np.array(getattr(co, k)) for k in "ABCDE"}
    n_eta = out["A"].shape[1]
    for k, value in (("A", 1.0 / (pitch * ex["cos"] ** 2)), ("B", ex["tan"]), ("C", np.full(len(p), pitch)),
                     ("D", -ex["dtan"]), ("E", np.zeros(len(p)))):
        out[k][p] = np.repeat(value[p][:, None], n_eta, axis=1)
    return out


D2_CLASSES = ("periodic far", "periodic near transitions")
INGREDIENTS = ("y0", "S", "dS_dm", "tan_beta", "cos_beta", "dtan_beta_dm (not read by coefficients.py)", "A", "B", "C", "D",
               "E")


def periodic_level(level: str, anchor: str) -> dict:
    """D2 on one level and anchor: ingredient errors by class, the three residuals (discrete
    coefficients, exact metrics, exact coefficients) and the coefficient-error contributions."""
    from turbolenza.b2b.coefficients import build_coefficients

    case = case_of(level)
    grid = anchored_grid(level, anchor)
    met, bcs, co = manufactured_of(grid)
    ex = periodic_exact(grid, case.cascade, anchor)
    p, pitch = ex["periodic"], grid.pitch
    f, l, n, J = grid.first_blade_i, grid.last_blade_i, len(grid.m), len(grid.eta) - 1
    dm, deta = co.length_scale * grid.dm, grid.deta
    gamma = D2_GAMMA_DEG
    psi = ts.uniform_psi(grid, gamma)
    met_ex = exact_metrics_of(met, ex)
    co_met = build_coefficients(met_ex, np.ones((n, J + 1)), 1.0, c_ref=None)
    co_ex = exact_coefficients_of(co, ex, pitch)
    fields_of = {"discrete": {k: np.asarray(getattr(co, k)) for k in "ABCDE"},
                 "exact metrics": {k: np.asarray(getattr(co_met, k)) for k in "ABCDE"}, "exact coefficients": co_ex}
    R = {name: ts.raw_rows(*(fc[k] for k in "ABCDE"), grid, bcs, dm, deta, psi)[0] for name, fc in fields_of.items()}
    masks = ts.row_classes(grid)
    # the exact psi derivatives at the periodic nodes (psi = (y - m tan gamma) / pitch, y = y0(m) + eta S)
    t_g = math.tan(math.radians(gamma))
    col = lambda v: np.repeat(np.asarray(v)[:, None], J + 1, axis=1)  # noqa: E731
    psi_ex = dict(eta=col(np.full(n, 1.0)), m=col((ex["tan"] - t_g) / pitch), mm=col(ex["dtan"] / pitch))
    psi_h = dict(eta=np.full_like(psi, np.nan), m=np.full_like(psi, np.nan), mm=np.full_like(psi, np.nan),
                 me=np.full_like(psi, np.nan), ee=np.full_like(psi, np.nan))
    psi_h["eta"][1:-1, 1:-1] = (psi[1:-1, 2:] - psi[1:-1, :-2]) / (2 * deta)
    psi_h["m"][1:-1, 1:-1] = (psi[2:, 1:-1] - psi[:-2, 1:-1]) / (2 * dm)
    psi_h["mm"][1:-1, 1:-1] = (psi[2:, 1:-1] - 2 * psi[1:-1, 1:-1] + psi[:-2, 1:-1]) / dm ** 2
    psi_h["me"][1:-1, 1:-1] = (psi[2:, 2:] - psi[2:, :-2] - psi[:-2, 2:] + psi[:-2, :-2]) / (4 * dm * deta)
    psi_h["ee"][1:-1, 1:-1] = (psi[1:-1, 2:] - 2 * psi[1:-1, 1:-1] + psi[1:-1, :-2]) / deta ** 2
    disc = fields_of["discrete"]
    contrib = dict(A=(disc["A"] - co_ex["A"]) * 0.0, B=-2.0 * (disc["B"] - co_ex["B"]) * 0.0,
                   C=(disc["C"] - co_ex["C"]) * psi_ex["mm"], D=(disc["D"] - co_ex["D"]) * psi_ex["eta"],
                   E=(disc["E"] - co_ex["E"]) * psi_ex["m"])
    total = sum(contrib.values())
    y0 = np.asarray(grid.nodes, float)[:, 0, 1]
    station = {"y0": np.abs(y0 - ex["y"]), "S": np.abs(np.asarray(met.S) - pitch), "dS_dm": np.abs(np.asarray(met.dS_dm)),
               "tan_beta": np.abs(np.asarray(met.tan_beta0) - ex["tan"]),
               "cos_beta": np.abs(np.asarray(met.cos_beta)[:, 0] - ex["cos"]),
               "dtan_beta_dm (not read by coefficients.py)": np.abs(np.asarray(met.dtan_beta0_dm) - ex["dtan"])}
    out = dict(level=level, anchor=anchor, dm_over_c=grid.dm / grid.chord, f=f, l=l, n_m=n, J=J,
               rate_inlet=ex["rate"]["inlet"], rate_exit=ex["rate"]["exit"], classes={})
    for cls in D2_CLASSES:
        mask = masks[cls]
        stations = np.flatnonzero(mask.any(axis=1))
        c_out = dict(stations=[int(s) for s in stations])
        for name in INGREDIENTS:
            if name in station:
                values = station[name][stations]
                k = int(np.argmax(values))
                c_out[name] = dict(max=float(values[k]), i=int(stations[k]))
            else:
                values = np.where(mask, np.abs(disc[name] - co_ex[name]), -1.0)
                i, j = np.unravel_index(int(np.argmax(values)), values.shape)
                c_out[name] = dict(max=float(values[i, j]), i=int(i))
        for name, field in R.items():
            a = np.where(mask, np.abs(field), -1.0)
            i, j = (int(x) for x in np.unravel_index(int(np.argmax(a)), a.shape))
            c_out[f"R {name}"] = dict(max=float(a[i, j]), i=i, j=j)
        i, j = c_out["R discrete"]["i"], c_out["R discrete"]["j"]
        c_out["at_max"] = dict(i=i, j=j, i_from_far=min(i, n - 1 - i), i_minus_f=i - f, l_minus_i=l - i,
                               R=float(R["discrete"][i, j]), R_exact_coefficients=float(R["exact coefficients"][i, j]),
                               **{f"c_{k}": float(contrib[k][i, j]) for k in "CDE"}, c_sum=float(total[i, j]))
        c_out["at_max"]["ratio_D"] = c_out["at_max"]["c_D"] / c_out["at_max"]["R"]
        c_out["at_max"]["ratio_sum"] = (c_out["at_max"]["c_sum"] + c_out["at_max"]["R_exact_coefficients"]) / c_out["at_max"]["R"]
        gap = np.abs((R["discrete"] - R["exact coefficients"]) - total)[mask]
        c_out["decomposition_gap"] = float(np.max(gap)) if gap.size else math.nan
        for k in "CDE":
            a = np.abs(contrib[k])[mask]
            c_out[f"max |c_{k}|"] = float(np.max(a)) if a.size else math.nan
        c_out["psi_check"] = {key: float(np.nanmax(np.abs(psi_h[key] - psi_ex[key])[mask])) for key in ("eta", "m", "mm")}
        c_out["psi_check"]["me"] = float(np.nanmax(np.abs(psi_h["me"])[mask]))
        c_out["psi_check"]["ee"] = float(np.nanmax(np.abs(psi_h["ee"])[mask]))
        out["classes"][cls] = c_out
    # the run ends and far stations: the metric slope there (one-sided at 0, f-1, l+1, n_m-1)
    out["run_ends"] = {int(i): dict(tan_beta=float(station["tan_beta"][i]), D=float(np.max(np.abs(disc["D"][i] - co_ex["D"][i]))))
                       for i in (0, 1, 2, f - 2, f - 1, l + 1, l + 2, n - 3, n - 2, n - 1)}
    return out


def phaseD() -> None:
    say("## Phase D: side checks (report only; no fix, no solve)")
    say()
    case = case_of(GEOMETRY_LEVEL)
    cascade = case.cascade
    c = cascade.active.chord
    say(f"Geometry: kutta_study.case_of({GEOMETRY_LEVEL!r}).cascade (NACA 65-(12A10)10, C2, kovach_sandercock, "
        f"thickness {cascade.spec.thickness_reconstruction or 'C2 default (quintic_natural_sqrt)'}, stagger 33 deg; "
        f"chord {c:g}). Curvature derivatives from the quintic's exact u derivatives (curvature_derivatives).")
    say()
    # D1: the self-check of the derivative formulas
    check = derivative_selfcheck(cascade)
    say("**D1 self-check of curvature_derivatives** (not a result): " + "; ".join(
        f"{name}: |kappa - signed_curvature| {v['kappa_vs_signed_curvature']:.1e}, kappa_s against a central difference of "
        f"kappa in s {v['kappa_s_vs_difference_rel']:.1e} (relative to max |kappa_s|, x/c-range u 0.03 .. 0.97), kappa_ss "
        f"against one of kappa_s {v['kappa_ss_vs_difference_rel']:.1e}" for name, v in check.items()) + ".")
    say()
    joints = construction_joints(cascade)
    knots = knot_statistics(cascade)
    table([j for j in joints if j["kind"] != "LE fair knot"],
          [("surface", "surface"), ("label", "joint"), ("u", "u"), ("x_c", "x/c"), ("s_over_c", "s/c"), ("kappa", "kappa c"),
           ("kappa_s", "kappa_s c^2"), ("kappa_ss", "kappa_ss c^3"), ("jump_kappa", "jump kappa"),
           ("jump_kappa_s", "jump kappa_s"), ("jump_kappa_ss", "jump kappa_ss"),
           ("smooth_bound_kappa", "2e-9 |X'| |kappa_s|"), ("nearest_knot_du", "|u - nearest knot|"),
           ("c1_kappa_jump", "C1 kappa jump there")],
          f"D1: the construction joints of the C2 surfaces (one-sided values at u -+ {JOINT_STEP_U:g}; the 23 LE fair "
          f"r(phi) knots per surface are in phaseD_joints.csv)", digits=4)
    fair = [j for j in joints if j["kind"] == "LE fair knot"]
    say("LE fair r(phi) knots: " + "; ".join(
        f"{name}: {sum(1 for j in fair if j['surface'] == name)} knots, u {min(j['u'] for j in fair if j['surface'] == name):.5f}"
        f" .. {max(j['u'] for j in fair if j['surface'] == name):.5f}, largest |jump| kappa "
        f"{max(abs(j['jump_kappa']) for j in fair if j['surface'] == name):.2e}, kappa_s "
        f"{max(abs(j['jump_kappa_s']) for j in fair if j['surface'] == name):.2e}, kappa_ss "
        f"{max(abs(j['jump_kappa_ss']) for j in fair if j['surface'] == name):.2e}" for name in ("suction", "pressure")) + ".")
    say()
    on_knot = [j for j in joints if j["on_knot"]]
    say(f"Joints that are also knots of the quintic: {len(on_knot)} "
        f"({', '.join(j['surface'] + ' ' + j['label'] for j in on_knot) or 'none'}); every other joint lies inside one "
        f"polynomial piece of the quintic (closest knot distance in the table).")
    say()
    table([dict(surface=name, n=k["n"], max_jump_kappa=k["max_jump_kappa"], max_jump_kappa_s=k["max_jump_kappa_s"],
                max_jump_kappa_ss=k["max_jump_kappa_ss"], max_jump_d5=k["max_jump_d5"]) for name, k in knots.items()],
          [("surface", "surface"), ("n", "interior knots"), ("max_jump_kappa", "max |jump kappa|"),
           ("max_jump_kappa_s", "max |jump kappa_s|"), ("max_jump_kappa_ss", "max |jump kappa_ss|"),
           ("max_jump_d5", "max |jump X^(5)| (per u^5)")],
          "D1: the knots of the quintics, jumps between the two polynomial pieces at each knot (exact piece values)", digits=3)
    near_rows = []
    for name in ("suction", "pressure"):
        own = [j for j in joints if j["surface"] == name and j["kind"] not in ("LE fair knot", "LE marker")]
        k = knots[name]
        for mark in X_C_MARKS:
            jt = min(own, key=lambda j: abs(j["x_c"] - mark))
            kk = int(np.argmin(np.abs(k["x_c"] - mark)))
            kj = k["jumps"][kk]
            near_rows.append(dict(surface=name, x_c_mark=mark, joint=jt["label"], joint_x_c=jt["x_c"],
                                  joint_jump_kappa=jt["jump_kappa"], joint_jump_kappa_s=jt["jump_kappa_s"],
                                  joint_jump_kappa_ss=jt["jump_kappa_ss"], knot_x_c=float(k["x_c"][kk]),
                                  knot_jump_kappa=kj["kappa"], knot_jump_kappa_s=kj["kappa_s"],
                                  knot_jump_kappa_ss=kj["kappa_ss"], knot_jump_d5=kj["d5"]))
    table(near_rows, [("surface", "surface"), ("x_c_mark", "x/c"), ("joint", "nearest joint"), ("joint_x_c", "its x/c"),
                      ("joint_jump_kappa", "jump kappa"), ("joint_jump_kappa_s", "jump kappa_s"),
                      ("joint_jump_kappa_ss", "jump kappa_ss"), ("knot_x_c", "nearest knot x/c"),
                      ("knot_jump_kappa", "knot: jump kappa"), ("knot_jump_kappa_s", "jump kappa_s"),
                      ("knot_jump_kappa_ss", "jump kappa_ss"), ("knot_jump_d5", "jump X^(5)")],
          "D1: the joint and the knot nearest to x/c 0.64 and 0.66 on each surface", digits=4)
    middle = blade_middle_rows(knots, joints)
    table(middle, [("level", "level"), ("anchor", "anchor"), ("row_class", "class"), ("i", "i"), ("j", "j"), ("J", "J"),
                   ("wall", "next to"), ("m_over_c", "m/c"), ("x_c", "x/c (that wall's node)"),
                   ("x_c_ss_node", "x/c SS node"), ("x_c_ps_node", "x/c PS node"), ("nearest_joint", "nearest joint"),
                   ("joint_distance_dm", "(m - m_joint)/dm"), ("knot_distance_dm", "(m - m_knot)/dm"),
                   ("residual", "residual"), ("C_mm", "C psi_mm"), ("mixed", "-2B psi_me"), ("D_term", "D psi_e"),
                   ("E_term", "E psi_m"), ("A_ee", "A psi_ee")],
          f"D1: the blade-middle maxima (gamma {REPORT_GAMMA_DEG:g} deg), where they sit and their term split", digits=4)
    stokes = stokes_character()
    if stokes is None:
        say(f"x/c 0.66: {STOKES_CSV} is absent; the source of the brief's number could not be located.")
    else:
        say(f"x/c 0.66 (brief): the only data found that shows a change near it is {stokes['path']} (kutta study phase 2, "
            f"very_fine, station K, converged; {stokes['elements']} elements, err_cos = LHS - RHS_cos in m/s x chord). The "
            f"elements with suction-side x/c 0.55 .. 0.75:")
        say()
        table(stokes["window"], [("element", "element"), ("x_c", "x/c (SS, mid)"), ("err_cos", "err_cos"), ("lhs", "LHS")],
              "Stokes error by element (read from the kutta study's file, not recomputed)", digits=5)
        say("Whether this is the change the brief means is not stated in that study's report; it is cited as data only.")
        say()
    curvature_figure(cascade, joints, middle, OUT / "phaseD_curvature.png")
    say("Figure: phaseD_curvature.png.")
    say()
    # D2
    say(f"### D2: the first-order term on the periodic rows (gamma {D2_GAMMA_DEG:g} deg, rho = 1, mdot' = 1)")
    say()
    say("coefficients.py builds A, B, C from metrics.S, metrics.tan_beta, metrics.cos_beta only (pointwise), and D = A_eta - "
        "B_m, E = C_m - B_eta by second-order differences of those fields (coefficients.py:_d_dm, _d_deta: central in m over "
        "the whole range, one-sided at i = 0 and n_m-1). On a periodic station metrics.py takes tan_beta0 = tan_beta1 = "
        "dy0/dm by metrics.py:_d_dm_run inside each run (central inside, three-point one-sided at the run ends i = 0, f-1, "
        "l+1, n_m-1), S = y1 - y0, cos from tan. metrics.dtan_beta_dm is not read by coefficients.py. Exact closed forms "
        "(side_line_exact): beta linear in m, y0 its integral, S = pitch, dS/dm = 0, A = 1/(pitch cos^2), B = tan, "
        "C = pitch, D = -sec^2(beta) dbeta/dm, E = 0. psi = (y - m tan gamma)/pitch with y = y0(m) + eta S on a periodic "
        "station: psi_eta = S/pitch = 1, psi_m = (tan beta - tan gamma)/pitch, psi_mm = d tan(beta)/dm / pitch, "
        "psi_me = psi_ee = 0.")
    say()
    results = {(a, level): periodic_level(level, a) for a in ANCHORS for level in LEVELS}
    dm = [results[("le_point", level)]["dm_over_c"] for level in LEVELS]
    ing_rows, res_rows, contrib_rows, where_rows, end_rows = [], [], [], [], []
    with np.errstate(divide="ignore"):
        for a in ANCHORS:
            for cls in D2_CLASSES:
                for name in INGREDIENTS:
                    v = [results[(a, level)]["classes"][cls][name]["max"] for level in LEVELS]
                    ing_rows.append(dict(anchor=a, row_class=cls, ingredient=name, **dict(zip(LEVELS, v)),
                                         at_i_very_fine=results[(a, "very_fine")]["classes"][cls][name]["i"],
                                         order=ts.power_law(dm, v)["p"]))
                for name in ("discrete", "exact metrics", "exact coefficients"):
                    v = [results[(a, level)]["classes"][cls][f"R {name}"]["max"] for level in LEVELS]
                    res_rows.append(dict(anchor=a, row_class=cls, residual=name, **dict(zip(LEVELS, v)),
                                         order=ts.power_law(dm, v)["p"]))
                for level in LEVELS:
                    r = results[(a, level)]["classes"][cls]
                    contrib_rows.append(dict(anchor=a, row_class=cls, level=level, dm_over_c=results[(a, level)]["dm_over_c"],
                                             **r["at_max"], decomposition_gap=r["decomposition_gap"],
                                             **{f"max_c_{k}": r[f"max |c_{k}|"] for k in "CDE"}))
                    where_rows.append(dict(anchor=a, row_class=cls, level=level, i=r["at_max"]["i"], j=r["at_max"]["j"],
                                           n_m=results[(a, level)]["n_m"], f=results[(a, level)]["f"],
                                           i_from_far=r["at_max"]["i_from_far"], i_minus_f=r["at_max"]["i_minus_f"]))
                for k in "CDE":
                    v = [results[(a, level)]["classes"][cls][f"max |c_{k}|"] for level in LEVELS]
                    res_rows.append(dict(anchor=a, row_class=cls, residual=f"max |c_{k}| (coefficient error x exact psi "
                                         f"derivative)", **dict(zip(LEVELS, v)), order=ts.power_law(dm, v)["p"]))
            for level in LEVELS:
                r = results[(a, level)]
                for i, v in r["run_ends"].items():
                    end_rows.append(dict(anchor=a, level=level, i=i, f=r["f"], l=r["l"], n_m=r["n_m"], **v))
    for a in ANCHORS:
        table([r for r in ing_rows if r["anchor"] == a],
              [("row_class", "class"), ("ingredient", "ingredient"), *[(level, level) for level in LEVELS],
               ("order", "order"), ("at_i_very_fine", "at i (vf)")],
              f"D2 {a}: max |discrete - closed form| of each ingredient over the class's nodes", digits=3)
    say(f"Side-line beta rate dbeta/dm: inlet {results[('le_point', 'very_fine')]['rate_inlet']:.6g} (le_point), "
        f"{results[('tip', 'very_fine')]['rate_inlet']:.6g} (tip) rad per chord; exit "
        f"{results[('le_point', 'very_fine')]['rate_exit']:.6g}, {results[('tip', 'very_fine')]['rate_exit']:.6g} "
        f"(exit angle = kappa_te: a straight line).")
    say()
    table([r for r in end_rows if r["level"] in ("very_coarse", "very_fine")],
          [("anchor", "anchor"), ("level", "level"), ("i", "i"), ("f", "f"), ("l", "l"), ("n_m", "n_m"),
           ("tan_beta", "|tan_beta - tan|"), ("D", "max_j |D - D_exact|")],
          "D2: the metric slope and D at the run ends (0, f-1, l+1, n_m-1, one-sided slope) and their neighbours", digits=3)
    table(res_rows, [("anchor", "anchor"), ("row_class", "class"), ("residual", "residual"),
                     *[(level, level) for level in LEVELS], ("order", "order")],
          "D2: max |raw residual| of the class with the discrete coefficients (phase B), with exact metrics (tan, cos, S "
          "exact on the periodic stations; D, E still differenced by coefficients.py), with exact coefficients (truncation "
          "of the psi stencil only), and the coefficient-error contributions", digits=3)
    table(contrib_rows, [("anchor", "anchor"), ("row_class", "class"), ("level", "level"), ("i", "i"), ("j", "j"),
                         ("i_from_far", "i from far end"), ("i_minus_f", "i - f"), ("R", "residual"),
                         ("c_D", "(D - D_ex) psi_e"), ("c_C", "(C - C_ex) psi_mm"), ("c_E", "(E - E_ex) psi_m"),
                         ("R_exact_coefficients", "truncation"), ("ratio_D", "c_D / residual"),
                         ("ratio_sum", "(c sum + truncation) / residual"), ("decomposition_gap", "decomposition gap")],
          "D2: at the node of the class's max |residual|: the coefficient-error contributions against the residual",
          digits=4)
    psi_check = {cls: {k: max(results[(a, level)]["classes"][cls]["psi_check"][k] for a in ANCHORS for level in LEVELS)
                       for k in ("eta", "m", "mm", "me", "ee")} for cls in D2_CLASSES}
    say("Check of the exact psi derivatives against the discrete ones on the class nodes (largest over anchors and levels): "
        + "; ".join(f"{cls}: psi_eta {v['eta']:.1e}, psi_m {v['m']:.1e}, psi_mm {v['mm']:.1e}, discrete psi_me "
                    f"{v['me']:.1e}, psi_ee {v['ee']:.1e}" for cls, v in psi_check.items()) + ".")
    say()
    table(where_rows, [("anchor", "anchor"), ("row_class", "class"), ("level", "level"), ("i", "i"), ("j", "j"),
                       ("n_m", "n_m"), ("f", "f"), ("i_from_far", "i from the far end"), ("i_minus_f", "i - f")],
          "D2: where the class maximum sits", digits=4)
    write_csv("phaseD_joints.csv", [{k: v for k, v in j.items()} for j in joints])
    write_csv("phaseD_blade_middle.csv", middle)
    write_csv("phaseD_periodic_ingredients.csv", ing_rows + [dict(kind="run_end", **r) for r in end_rows])
    write_csv("phaseD_periodic_terms.csv", res_rows + [dict(kind="at_max", **r) for r in contrib_rows])
    save("phaseD.json", dict(selfcheck=check, joints=joints,
                             knots={k: {kk: vv for kk, vv in v.items() if kk != "jumps"} for k, v in knots.items()},
                             near_marks=near_rows, blade_middle=middle, stokes=stokes,
                             periodic={f"{a}|{level}": r for (a, level), r in results.items()}, ingredients=ing_rows,
                             residuals=res_rows, contributions=contrib_rows, where=where_rows, run_ends=end_rows,
                             psi_check=psi_check,
                             settings=dict(GEOMETRY_LEVEL=GEOMETRY_LEVEL, JOINT_STEP_U=JOINT_STEP_U,
                                           DENSE_CURVE_U=DENSE_CURVE_U, D2_GAMMA_DEG=D2_GAMMA_DEG, LEVELS=LEVELS)))
    write_lines("phaseD.md")


# --- the figure -------------------------------------------------------------------------------------------------------

def closeup_figure(path: Path) -> None:
    """LE and TE close-ups on CLOSEUP_LEVELS: the reference blade's walls, both anchors' side lines
    and eta-lines near the transitions (this passage's and, one pitch down, the passage below's, so
    both walls of the blade are seen), the nose tip / tail end and the LE / TE points."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    style, SERIES, SURFACE, INK, INK2, MUTED = ks.style, ks.SERIES, ks.SURFACE, ks.INK, ks.INK2, ks.MUTED
    fig, axes = plt.subplots(len(CLOSEUP_LEVELS), 2, figsize=(11.0, 5.2 * len(CLOSEUP_LEVELS)), facecolor=SURFACE)
    h = CLOSEUP_HALF_WIDTH_CHORDS
    for row, level in enumerate(CLOSEUP_LEVELS):
        case = case_of(level)
        grid, cascade = case.grid, case.cascade
        b, c = grid.boundaries, grid.chord
        ends = tip_and_tail(cascade)
        tg = tip_grid(grid, cascade, ends)
        f, l, J = grid.first_blade_i, grid.last_blade_i, len(grid.eta) - 1
        u = np.linspace(0.0, 1.0, 4001)
        walls = {name: np.asarray(getattr(cascade.active_curves, name).evaluate(u)) / c for name in ("suction", "pressure")}
        pitch = b.periodic_translation[1]
        for col, (edge, centre) in enumerate((("LE", ends["tip"]), ("TE", ends["tail"]))):
            ax = axes[row, col]
            style(ax, f"{level}: {edge} (n_m x n_eta = {len(grid.m)} x {J + 1}, dm/c = {grid.dm / c:.4f})")
            x0, y0 = centre["m"] / c, centre["y"] / c
            for name, colour in (("suction", INK), ("pressure", INK2)):
                ax.plot(walls[name][:, 0], walls[name][:, 1], color=colour, linewidth=1.6, label=f"{name} wall")
            inside = np.flatnonzero(np.abs(np.asarray(grid.m) / c - x0) <= h + grid.dm / c)
            sl = slice(max(int(inside[0]) - 1, 0), min(int(inside[-1]) + 2, len(grid.m)))
            for g, colour, dash, name in ((grid, SERIES[0], "-", "le_point"), (tg, SERIES[1], "--", "tip")):
                y = np.asarray(g.nodes, float)[sl, :, 1] / c
                x = np.asarray(g.m, float)[sl] / c
                for shift in (0.0, -pitch / c):
                    for j in range(J + 1):
                        ax.plot(x, y[:, j] + shift, color=colour, linestyle=dash, linewidth=0.55, alpha=0.8)
                ax.plot([], [], color=colour, linestyle=dash, linewidth=0.8, label=f"{name} eta-lines")
            mm = np.linspace(x0 - 1.5 * h, x0 + 1.5 * h, 801) * c
            if edge == "LE":
                le_m = mm[mm <= b.m_le]
                ax.plot(le_m / c, gr._inlet_side(b, le_m) / c, color=SERIES[0], linewidth=2.0, label="le_point side line")
                tip_m = mm[mm <= ends["tip"]["m"]]
                ax.plot(tip_m / c, anchored_side(grid, cascade, ends, "inlet", tip_m)[1] / c, color=SERIES[1],
                        linewidth=2.0, linestyle="--", label="tip side line")
                stations, point, pname = (f - 1, f), np.asarray(cascade.active.leading_edge_point) / c, "LE point"
                cname = "nose tip"
            else:
                te_m = mm[mm >= b.m_te]
                ax.plot(te_m / c, gr._exit_side(b, te_m) / c, color=SERIES[0], linewidth=2.0, label="le_point side line")
                tail_m = mm[mm >= ends["tail"]["m"]]
                ax.plot(tail_m / c, anchored_side(grid, cascade, ends, "exit", tail_m)[1] / c, color=SERIES[1],
                        linewidth=2.0, linestyle="--", label="tip side line")
                stations, point, pname = (l, l + 1), np.asarray(cascade.active.trailing_edge_point) / c, "TE point"
                cname = "tail end"
            for k, i in enumerate(stations):
                ax.axvline(grid.m[i] / c, color=MUTED, linewidth=0.8, linestyle=":")
                ax.annotate(f"i = {i}", (grid.m[i] / c, y0 + h * (0.92 - 0.07 * k)), fontsize=7, color=INK2,
                            ha="right" if k == 0 else "left")
            ax.plot(x0, y0, marker="o", markersize=7, color=SERIES[2], linestyle="none", label=cname)
            ax.plot(point[0], point[1], marker="s", markersize=7, color=SERIES[3], linestyle="none", label=pname)
            ax.set_xlim(x0 - h, x0 + h)
            ax.set_ylim(y0 - h, y0 + h)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel("m / c")
            ax.set_ylabel("y / c")
            if row == 0:
                ax.legend(fontsize=7, frameon=False, labelcolor=INK2, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------------------------------

PHASES = {"phaseA": phaseA, "phaseB": phaseB, "phaseD": phaseD}        # the report: a later step


class _Tee:
    """stdout copied to the phase's console file."""

    def __init__(self, stream, handle) -> None:
        self.stream, self.handle = stream, handle

    def write(self, text: str) -> int:
        self.stream.write(text)
        self.handle.write(text)
        return len(text)

    def flush(self) -> None:
        self.stream.flush()
        self.handle.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("phase", choices=sorted(PHASES))
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    stdout = sys.stdout
    with open(OUT / f"{args.phase}_console.txt", "w", encoding="utf-8") as handle:
        sys.stdout = _Tee(stdout, handle)
        try:
            PHASES[args.phase]()
        finally:
            sys.stdout = stdout


if __name__ == "__main__":
    main()
