"""
Audit of the side-boundary construction of grid.py against the Aungier specification given
for this task (diagnostic only; no production file changed):

    eta = (y - y0(m)) / (y1(m) - y0(m)),  S = y1 - y0,  tan(beta) = dy/dm at constant eta,
    tan(beta) = tan(beta0) + (tan(beta1) - tan(beta0)) eta;
    outside the blade passage y1 = y0 + s (periodic, beta0 = beta1), the side line from
    dy0/dm = tan(beta_side(m)), beta varying uniformly from the inlet flow angle to the
    blade LE angle (and from the blade TE angle to the exit flow angle downstream);
    inside the passage eta = 0 and 1 are the passage-facing blade surfaces.

    python scripts/audit_aungier_side_boundaries.py --save outputs/aungier_side_boundary_audit --no-show

Only SS+PS-admissible uniform grids are used (grid.find_valid_n_passage, the recommended
sequences of scripts/audit_uniform_grid_validity.py). The blade is the default C2
65-(12A10)10, solidity 1, inlet angle 45 deg, exit angle kappa_te (grid.py's default),
one-chord extensions.

Sections (printed) and figures 01 .. 09 (outputs/aungier_side_boundary_audit/)
    1  the six boundary pieces of grid.py and how they are joined; the production side line
       against an independent quadrature of the specified beta(m) law
    2  the points at each end: nose tip / tail end, construction LE / TE point (= SS/PS
       branch point), the side line's anchor, the first / last numerical wall nodes
    3  continuity at the switch the mapping actually uses, at the anchor and at the wall node
    4  the junction the specification implies: the C0 defect of any junction m_J
    5  Aungier-style diagnostic side lines (quadrature, independent of grid.py) anchored at
       the candidates; C0, C1, pitch separation, metal, far-boundary shift
    6  consequences on the admissible sequences: node steps, D near the LE, a smooth field's
       psi_m at the last periodic station, current against the tip / tail anchoring
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import warnings

import numpy as np
from scipy.integrate import IntegrationWarning, quad

# quad flags "roundoff" when the integral is already at machine precision (checked against grid.py to 1e-16 c below)
warnings.filterwarnings("ignore", category=IntegrationWarning)

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", Path(__file__).resolve().parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import matplotlib.pyplot as plt  # noqa: E402

import audit_redist_numerics as ar  # noqa: E402
import audit_wall_station_topology as wt  # noqa: E402
import link_geometry as lg  # noqa: E402
from turbolenza.b2b import grid as gr  # noqa: E402
from turbolenza.b2b.coefficients import build_coefficients  # noqa: E402
from turbolenza.b2b.config import GridSettings  # noqa: E402
from turbolenza.b2b.metrics import build_metrics  # noqa: E402

STAGGERS = (13.0, 23.0, 33.0, 43.0, 53.0)
SEQUENCES = {13.0: (60, 102, 173, 294), 23.0: (43, 66, 101, 159), 33.0: (29, 43, 65, 97),
             43.0: (26, 34, 45, 59), 53.0: (22, 31, 45)}
INLET = wt.INLET
N_ETA = 21
INK, MUTED, GRIDLINE = wt.INK, wt.MUTED, wt.GRIDLINE
SUCTION, PRESSURE = wt.SUCTION, wt.PRESSURE
CURRENT, AUNGIER, NODE = "#cf222e", "#1baf7a", "#8250df"


# --- the specified side line, written out here (quadrature, independent of grid.py) ----------------

def aungier_side(junction_m: float, junction_y: float, junction_angle: float, far_m: float, far_angle: float):
    """y(m) of the lower side line: dy/dm = tan(beta(m)), beta linear in m from junction_angle
    at the junction to far_angle at the far station, through the junction point; by
    adaptive quadrature."""
    def beta(m):
        return junction_angle + (m - junction_m) / (far_m - junction_m) * (far_angle - junction_angle)

    def y(m):
        m = np.atleast_1d(np.asarray(m, dtype=float))
        return np.array([junction_y + quad(lambda x: math.tan(beta(x)), junction_m, mm, epsabs=1e-15, epsrel=1e-14)[0]
                         for mm in m])
    return y, beta


class Ends:
    """The points at one stagger, all from the authoritative geometry or grid.py's own data."""

    def __init__(self, stagger: float):
        self.geo = geo = wt.Geometry(stagger)
        self.cascade = geo.cascade
        g = gr.make_grid(self.cascade, GridSettings(n_passage=SEQUENCES[stagger][0], n_eta=3), INLET)
        self.b, self.row = g.boundaries, gr._row_of(g.boundaries)
        b = self.b
        self.pitch = geo.pitch
        self.kappa_le, self.kappa_te = float(self.cascade.kappa_le), float(self.cascade.kappa_te)
        self.anchor_le = np.asarray(b.upstream_lower_boundary[-1])     # the side line's own end point
        self.anchor_te = np.asarray(b.downstream_lower_boundary[0])
        self.tip, self.tail = geo.tip, geo.tail
        # the current side lines and the Aungier-style lines (quadrature) at the candidate junctions
        self.current_in = lambda m: gr._inlet_side(b, np.atleast_1d(np.asarray(m, dtype=float)))  # noqa: E731
        self.current_out = lambda m: gr._exit_side(b, np.atleast_1d(np.asarray(m, dtype=float)))  # noqa: E731
        self.quad_le, self.beta_le = aungier_side(b.m_le, float(self.anchor_le[1]), self.kappa_le, b.m_in, b.inlet_angle)
        self.quad_te, self.beta_te = aungier_side(b.m_te, float(self.anchor_te[1]), self.kappa_te, b.m_out, b.exit_angle)
        self.tip_in, self.tip_beta = aungier_side(self.tip[0], self.tip[1], self.kappa_le, b.m_in, b.inlet_angle)
        self.tail_out, self.tail_beta = aungier_side(self.tail[0], self.tail[1], self.kappa_te, b.m_out, b.exit_angle)

    def walls(self, m):
        """Top and bottom of the reference blade's cut (the passage-facing walls; the upper
        wall is bottom + pitch), grid.py's own rule, where the line cuts the blade."""
        m = np.atleast_1d(np.asarray(m, dtype=float))
        cut = self.row.cut(m)
        blade = np.asarray(cut.cuts_blade)
        top, bottom = np.full(m.shape, np.nan), np.full(m.shape, np.nan)
        idx = np.flatnonzero(blade)
        if idx.size:
            t, bo = self.row.passage_sides(cut, idx)
            top[idx], bottom[idx] = t, bo
        return top, bottom, blade

    def cut_length(self, m):
        top, bottom, _ = self.walls(m)
        return top - bottom


def tip_grid(ends: Ends, n: int, anchor: str) -> gr.Grid:
    """The production grid of n, or the same stations with the Aungier-style side lines through
    the nose tip / tail end (same angles) at the periodic stations: a diagnostic Grid."""
    g = gr.make_grid(ends.cascade, GridSettings(n_passage=n, n_eta=N_ETA), INLET, require_ss_ps=True)
    if anchor == "production":
        return g
    y0, y1, _ = wt.walls_at(ends.geo, g.boundaries, ends.row, g.m, "tip")
    nodes = np.array(g.nodes)
    nodes[..., 1] = y0[:, None] + g.eta[None, :] * (y1 - y0)[:, None]
    return gr.Grid(boundaries=g.boundaries, m=g.m, eta=g.eta, nodes=nodes, first_blade_i=g.first_blade_i,
                   last_blade_i=g.last_blade_i)


def one_sided(f, m0: float, side: int, eps: float) -> tuple[float, float]:
    """value and slope of f a distance eps on one side of m0 (side -1 left, +1 right)."""
    a, c = m0 + side * eps, m0 + side * 2 * eps
    va, vc = float(f(np.array([a]))[0]), float(f(np.array([c]))[0])
    return va, (vc - va) / (c - a)


# --- main ------------------------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--save", type=Path, default=None)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    if args.save is not None:
        args.save.mkdir(parents=True, exist_ok=True)

    ends = {st: Ends(st) for st in STAGGERS}
    for st in STAGGERS:
        valid = gr.find_valid_n_passage(ends[st].cascade, GridSettings(n_passage=101, n_eta=3), INLET, SEQUENCES[st])
        if valid != SEQUENCES[st]:
            raise SystemExit(f"{st} deg: the sequence {SEQUENCES[st]} is not all admissible ({valid})")
    print("0. All grids used are SS+PS-admissible (grid.find_valid_n_passage): " +
          "; ".join(f"{st:.0f} deg {SEQUENCES[st]}" for st in STAGGERS))

    print("\n1. The six pieces and how grid.py joins them (make_passage_boundaries, make_grid sections 1-3)")
    print("   lower upstream side line  y0(m) = y_A + (m - m_A) T(beta(m), kappa_le), T(a, b) = ln(cos a / cos b) / (b - a),")
    print("                             beta linear in m: kappa_le at m_A -> inlet angle at m_in; A = active.leading_edge_point")
    print("   upper upstream side line  y1 = y0 + pitch (never integrated on its own)")
    print("   lower blade boundary      y0 = top of the reference blade's cut (passage-facing contour from the nose tip)")
    print("   upper blade boundary      y1 = bottom of the cut + pitch (the +pitch neighbour's passage-facing contour)")
    print("   lower downstream side     y0 = y_B + (m - m_B) T(beta(m), kappa_te), beta: kappa_te at m_B -> exit angle at m_out;")
    print("                             B = active.trailing_edge_point; exit angle None = kappa_te, so the default line is straight")
    print("   upper downstream side     y1 = y0 + pitch")
    print("   join rule: a station line that cuts the blade (metal > TOUCH) takes the walls, any other the side lines;")
    print("   so the switch is at the nose tip m_tip and the tail end m_tail, not at the anchors A and B")
    for st in STAGGERS:
        e = ends[st]
        b = e.b
        ms = np.linspace(b.m_in, b.m_le, 41)
        ms_out = np.linspace(b.m_te, b.m_out, 41)
        d_in = np.abs(e.current_in(ms) - e.quad_le(ms)).max()
        d_out = np.abs(e.current_out(ms_out) - e.quad_te(ms_out)).max()
        chord_line = e.anchor_le[1] + (ms - b.m_le) * (e.current_in([b.m_in])[0] - e.anchor_le[1]) / (b.m_in - b.m_le)
        print(f"   {st:.0f} deg: production upstream line vs quadrature of the specified law {d_in:.1e} c, downstream "
              f"{d_out:.1e} c; it is not the straight chord from the inlet to the anchor (that differs by up to "
              f"{np.abs(e.current_in(ms) - chord_line).max():.2e} c); beta at the anchor {math.degrees(float(b.upstream_beta[-1])):.3f} "
              f"deg = kappa_le {math.degrees(e.kappa_le):.3f}, at m_in {math.degrees(float(b.upstream_beta[0])):.3f} = inlet "
              f"45; exit {math.degrees(float(b.downstream_beta[0])):.3f} -> {math.degrees(float(b.downstream_beta[-1])):.3f} deg")
    for st in STAGGERS:
        # the default exit angle kappa_te makes the exit line straight; a turned exit angle tests the integration there
        e = ends[st]
        turned = e.kappa_te - math.radians(5.0)
        bt = gr.make_passage_boundaries(e.cascade, GridSettings(n_passage=SEQUENCES[st][0], n_eta=3), INLET, turned)
        line, _ = aungier_side(bt.m_te, float(bt.downstream_lower_boundary[0][1]), e.kappa_te, bt.m_out, turned)
        ms_out = np.linspace(bt.m_te, bt.m_out, 41)
        d_out = np.abs(gr._exit_side(bt, ms_out) - line(ms_out)).max()
        rb = gr.rebuild_exit(e.b, turned)
        print(f"   {st:.0f} deg, exit angle kappa_te - 5 deg: production exit line vs quadrature {d_out:.1e} c; anchor "
              f"still the TE point (rebuild_exit keeps downstream_lower_boundary[0]: "
              f"{np.abs(np.asarray(rb.downstream_lower_boundary[0]) - e.anchor_te).max():.0e})")

    print("\n2. The points at each end (m, y in chords)")
    for st in STAGGERS:
        e = ends[st]
        g = gr.make_grid(e.cascade, GridSettings(n_passage=SEQUENCES[st][-1], n_eta=3), INLET, require_ss_ps=True)
        f, l = g.first_blade_i, g.last_blade_i
        print(f"   {st:.0f} deg LE: nose tip ({e.tip[0]:+.6f}, {e.tip[1]:+.6f}) on {e.geo.tip_curve}; construction LE = SS/PS branch "
              f"point = side-line anchor ({e.anchor_le[0]:+.6f}, {e.anchor_le[1]:+.6f}) (coincide to {e.geo.coincidence_le:.0e}); "
              f"n {SEQUENCES[st][-1]}: last periodic m {g.m[f - 1]:+.6f}, first wall station m {g.m[f]:+.6f}, lower node "
              f"{g.nodes[f, 0, 1]:+.6f} (SS), upper node - pitch {g.nodes[f, -1, 1] - e.pitch:+.6f} (PS)")
        print(f"   {st:.0f} deg TE: tail end ({e.tail[0]:+.6f}, {e.tail[1]:+.6f}) on {e.geo.tail_curve}; construction TE = branch point "
              f"= anchor ({e.anchor_te[0]:+.6f}, {e.anchor_te[1]:+.6f}); last wall station m {g.m[l]:+.6f}, first exit periodic "
              f"m {g.m[l + 1]:+.6f}")

    print("\n3. Continuity of y0, y1 (production): at the switch (nose tip / tail end), one-sided at 1e-10 and 1e-6 c; the "
          "limit of the walls at the switch is the tip (tail) point itself, the cut closing like sqrt(d)")
    jumps = {}
    for st in STAGGERS:
        e = ends[st]
        lo = e.current_in(np.array([e.tip[0]]))[0]
        hi = e.current_out(np.array([e.tail[0]]))[0]
        jumps[st] = dict(le=e.tip[1] - lo, te=e.tail[1] - hi)
        rows = []
        for eps in (1e-10, 1e-6):
            v_side, s_side = one_sided(e.current_in, e.tip[0], -1, eps)
            v_top, s_top = one_sided(lambda m: e.walls(m)[0], e.tip[0], +1, eps)
            v_bot, s_bot = one_sided(lambda m: e.walls(m)[1], e.tip[0], +1, eps)
            rows.append(f"eps {eps:.0e}: y0 {v_side:+.6f} | {v_top:+.6f}, y1-s {v_side:+.6f} | {v_bot:+.6f}; slopes y0 "
                        f"{s_side:+.3f} | {s_top:+.1f}, y1 {s_side:+.3f} | {s_bot:+.1f}")
        print(f"   {st:.0f} deg LE: jump y0 = y1 jump = {jumps[st]['le']:+.4e} c (exact limit: tip y - side line at m_tip), "
              f"C0 DISCONTINUOUS; " + "; ".join(rows))
        v_side, s_side = one_sided(e.current_out, e.tail[0], +1, 1e-6)
        v_top, s_top = one_sided(lambda m: e.walls(m)[0], e.tail[0], -1, 1e-6)
        print(f"   {st:.0f} deg TE: jump {jumps[st]['te']:+.4e} c, C0 DISCONTINUOUS; slopes at 1e-6: side {s_side:+.3f}, top "
              f"{s_top:+.1f}")
        # at the anchor itself and at the first wall station
        top_le, bot_le, _ = e.walls(np.array([e.b.m_le + 1e-12]))
        print(f"      at the anchor m_le: side line y0 = {e.anchor_le[1]:+.6f} lies on the blade (the branch point, the bottom "
              f"of the cut there); the walls there are top {top_le[0]:+.6f} (jump {top_le[0] - e.anchor_le[1]:+.4e}) and "
              f"bottom {bot_le[0]:+.6f} (jump {bot_le[0] - e.anchor_le[1]:+.1e}); the cut is {top_le[0] - bot_le[0]:.4e} c")

    print("\n4. The junction the specification implies. Outside the passage y1 - y0 = pitch exactly (periodic); inside it, "
          "y1 - y0 = pitch - cut(m) (passage-facing walls). Both boundaries continuous at a junction m_J needs cut(m_J) = 0: "
          "the defect of any junction is at least cut(m_J) on one boundary")
    defects = {}
    for st in STAGGERS:
        e = ends[st]
        ms = np.concatenate(([e.tip[0] + 1e-12], np.linspace(e.tip[0], e.b.m_le + 0.02, 400)[1:]))
        cuts = e.cut_length(ms)
        defects[st] = (ms, cuts)
        g = gr.make_grid(e.cascade, GridSettings(n_passage=SEQUENCES[st][-1], n_eta=3), INLET, require_ss_ps=True)
        print(f"   {st:.0f} deg: cut(m_J) = 0 only at the nose tip (cut at the tip + 1e-12 c: {cuts[0]:.1e}); at the "
              f"construction LE {float(e.cut_length(np.array([e.b.m_le + 1e-12]))[0]):.4e} c; at the first wall station of n "
              f"{SEQUENCES[st][-1]} {float(e.cut_length(np.array([g.m[g.first_blade_i]]))[0]):.4e} c; TE: cut 0 only at the "
              f"tail end, {float(e.cut_length(np.array([e.b.m_te - 1e-12]))[0]):.3e} c at the construction TE")
    print("   the other reading, 'the passage is m_le .. m_te' (walls only from the construction LE point): the stations "
          "between the nose tip and m_le would be periodic, and their strip y0 .. y0 + pitch contains the nose metal")
    for st in STAGGERS:
        e = ends[st]
        rows = []
        for n in SEQUENCES[st]:
            m = gr._stations(e.b, n)
            nose = m[(m > e.tip[0]) & (m < e.b.m_le)]
            tail = m[(m > e.b.m_te) & (m < e.tail[0])]
            metal = [float(e.cut_length(np.array([x]))[0]) for x in np.concatenate((nose, tail))]
            rows.append(f"n {n}: {len(nose)} + {len(tail)} station(s), metal {max(metal) if metal else 0.0:.2e} c")
        print(f"   {st:.0f} deg: " + "; ".join(rows) + f"; at most cut(m_le) {float(e.cut_length(np.array([e.b.m_le - 1e-12]))[0]):.3e} c "
              f"(and the current anchor is then C0 for y1 only: the LE point is the bottom of the cut at m_le)")

    print("\n5. Aungier-style diagnostic side lines (quadrature of the specified law, through each candidate junction; beta "
          "at the junction = the blade angle kappa_le / kappa_te of geometry.py)")
    blades = {st: lg.Blade(ends[st].cascade) for st in STAGGERS}
    candidates = {}
    for st in STAGGERS:
        e = ends[st]
        b = e.b
        # local boundary tangents at the candidate junctions, for the beta_LE question
        cur = e.geo.curves
        ss0 = cur["SS"].derivative(np.array([0.0]), 1)[0]
        ps0 = cur["PS"].derivative(np.array([0.0]), 1)[0]
        g = gr.make_grid(e.cascade, GridSettings(n_passage=SEQUENCES[st][-1], n_eta=3), INLET, require_ss_ps=True)
        f = g.first_blade_i
        roots = e.geo.roots(float(g.m[f]))
        slope_f = [r.slope for r in (roots[-1], roots[0])]
        print(f"   {st:.0f} deg beta_LE candidates: kappa_le (equivalent arc) {math.degrees(e.kappa_le):.2f} deg; stagger "
              f"{st:.0f}; wall slopes at the nose tip +-inf (vertical); at the LE point the SS leaves at "
              f"{math.degrees(math.atan2(ss0[1], ss0[0])):.1f} deg, the PS at {math.degrees(math.atan2(ps0[1], ps0[0])):.1f} "
              f"deg; at the first wall node the lower (SS) slope {math.degrees(math.atan(slope_f[0])):.1f} deg and the upper "
              f"(PS) {math.degrees(math.atan(slope_f[1])):.1f} deg: no single angle meets both walls")
        rows = {}
        for name, line, (mj, yj) in (("construction LE point (current)", e.quad_le, e.anchor_le),
                                     ("nose tip", e.tip_in, e.tip)):
            at_tip = float(line(np.array([e.tip[0]]))[0])
            c0 = e.tip[1] - at_tip
            ms = np.linspace(b.m_in, e.tip[0] - 1e-9, 200)
            ys = line(ms)
            metal = sum(int(blades[st].paths[k].contains_points(np.column_stack([ms, ys + k2 * e.pitch])).sum())
                        for k in (-1, 0, 1, 2) for k2 in (0, 1))
            far = float(line(np.array([b.m_in]))[0]) - float(e.current_in(np.array([b.m_in]))[0])
            slope = math.tan(e.kappa_le) if name == "nose tip" else float(np.gradient(line(np.array([e.tip[0] - 2e-6, e.tip[0] - 1e-6, e.tip[0]])), 1e-6)[-1])
            rows[name] = dict(c0=c0, metal=metal, far=far, slope=slope)
            print(f"      junction {name:32s}: C0 jump at the switch {c0:+.3e} c (lower and upper alike: y1 = y0 + pitch "
                  f"exactly), side slope there {slope:+.3f} vs wall slopes +-inf (not C1), side-strip samples in metal "
                  f"{metal}, far-inlet y shift against production {far:+.3e} c")
        tail_c0 = e.tail[1] - float(e.tail_out(np.array([e.tail[0]]))[0])
        anchor_c0 = e.tail[1] - float(e.quad_te(np.array([e.tail[0]]))[0])
        rows["tail"] = dict(c0=tail_c0, anchor=anchor_c0)
        print(f"      TE: through the construction TE point {anchor_c0:+.3e} c, through the tail end {tail_c0:+.1e} c")
        candidates[st] = rows

    print("\n6. Consequences on the admissible sequences (n_eta 21): node steps y0[f] - y0[f-1], y1[f] - y1[f-1], max |D| at "
          "stations f-3 .. f+3 (coefficients.py, rho 1), smooth physical field |psi_m| error at f-1 (central / one-sided): "
          "production side lines against the Aungier-style lines through the nose tip and tail end (diagnostic Grid, same "
          "stations, same walls)")
    original = ar.stations
    ar.stations = lambda kind, geo, b, n: original("PROD", geo, b, n) if kind == "PROD_TIP" else original(kind, geo, b, n)
    ar.ANCHOR["PROD_TIP"] = "tip"
    consequences = {st: {"production": [], "tip": []} for st in STAGGERS}
    for st in STAGGERS:
        e = ends[st]
        for n in SEQUENCES[st]:
            for anchor in ("production", "tip"):
                g = tip_grid(e, n, anchor)
                met = build_metrics(g)
                co = build_coefficients(met, np.ones((len(g.m), len(g.eta))), 1.0)
                f, l = g.first_blade_i, g.last_blade_i
                case = ar.Case(e.geo, g.boundaries, e.row, "PROD" if anchor == "production" else "PROD_TIP", n,
                               n_eta=N_ETA, metrics_needed=False)
                if not np.allclose(case.Y, g.nodes[..., 1], atol=1e-14, rtol=0):
                    raise SystemExit("diagnostic case nodes differ from the grid")
                phys = ar.physical_errors(case)
                rec = dict(n=n, dm=g.dm, step0=float(g.nodes[f, 0, 1] - g.nodes[f - 1, 0, 1]),
                           step1=float(g.nodes[f, -1, 1] - g.nodes[f - 1, -1, 1]),
                           tb_jump0=float(met.tan_beta0[f] - met.tan_beta0[f - 1]),
                           tb_jump1=float(met.tan_beta1[f] - met.tan_beta1[f - 1]),
                           D_le=float(np.abs(co.D[f - 3:f + 4]).max()), E_le=float(np.abs(co.E[f - 3:f + 4]).max()),
                           D_te=float(np.abs(co.D[l - 3:l + 4]).max()), psi=phys["f-1"], psi_l1=phys["l+1"],
                           S_err=float(np.abs((g.nodes[:f, -1, 1] - g.nodes[:f, 0, 1]) - e.pitch).max()))
                rec["quality"] = wt.quality(e.geo, g.boundaries, e.row, g.m, anchor)
                consequences[st][anchor].append(rec)
            p, t = consequences[st]["production"][-1], consequences[st]["tip"][-1]
            print(f"   {st:.0f} deg n {n:3d} (dm {p['dm']:.4f}): steps y0 {p['step0']:+.4f} -> {t['step0']:+.4f}, y1 "
                  f"{p['step1']:+.4f} -> {t['step1']:+.4f}; d tan b0 {p['tb_jump0']:+.3f} -> {t['tb_jump0']:+.3f}; |D| LE "
                  f"{p['D_le']:.1f} -> {t['D_le']:.1f}, TE {p['D_te']:.1f} -> {t['D_te']:.1f}; |E| LE {p['E_le']:.2f} -> "
                  f"{t['E_le']:.2f}; psi_m f-1 central {p['psi'][0]:.3f} -> {t['psi'][0]:.3f} (one-sided {p['psi'][1]:.1e} -> "
                  f"{t['psi'][1]:.1e}); l+1 central {p['psi_l1'][0]:.3f} -> {t['psi_l1'][0]:.3f}; periodic S - pitch "
                  f"{p['S_err']:.0e} / {t['S_err']:.0e}")
    ar.stations = original
    print("   measured growth orders p (quantity ~ dm^-p, first to last grid of each sequence): current / Aungier-style")
    for st in STAGGERS:
        out = []
        for key, get in (("|D| LE", lambda r: r["D_le"]), ("|D| TE", lambda r: r["D_te"]),
                         ("psi_m f-1", lambda r: r["psi"][0]), ("d tan b0", lambda r: r["tb_jump0"])):
            orders = []
            for anchor in ("production", "tip"):
                a, z = consequences[st][anchor][0], consequences[st][anchor][-1]
                orders.append(math.log(get(z) / get(a)) / math.log(a["dm"] / z["dm"]))
            out.append(f"{key} {orders[0]:.2f} / {orders[1]:.2f}")
        print(f"   {st:.0f} deg: " + "; ".join(out))
    print("   cell quality on every admissible grid (n_eta 21): negative-area cells, folded cells, min S, max aspect; "
          "current / Aungier-style")
    for st in STAGGERS:
        q = [(r["quality"], t["quality"]) for r, t in zip(consequences[st]["production"], consequences[st]["tip"])]
        print(f"   {st:.0f} deg: negative {sum(a['negative'] for a, _ in q)} / {sum(b['negative'] for _, b in q)}, folded "
              f"{sum(a['folded'] for a, _ in q)} / {sum(b['folded'] for _, b in q)}, min S {min(a['S_min'] for a, _ in q):.4f} / "
              f"{min(b['S_min'] for _, b in q):.4f}, max aspect {max(a['aspect_max'] for a, _ in q):.1f} / "
              f"{max(b['aspect_max'] for _, b in q):.1f}, min cell area {min(a['area_min'] for a, _ in q):.2e} / "
              f"{min(b['area_min'] for _, b in q):.2e}")

    if args.save is None:
        return
    figures = {}

    def panel_ends(ax, st, side, current=True, aungier=False, span=None):
        e = ends[st]
        wt.draw_blade(ax, e.geo, lw=1.8)
        g = gr.make_grid(e.cascade, GridSettings(n_passage=SEQUENCES[st][-1], n_eta=3), INLET, require_ss_ps=True)
        f, l = g.first_blade_i, g.last_blade_i
        ext = e.tip if side == "LE" else e.tail
        anchor = e.anchor_le if side == "LE" else e.anchor_te
        span = span or max(0.03, 3 * g.dm)
        if side == "LE":
            ms_side = np.linspace(ext[0] - span, ext[0], 200)
            ms_wall = np.linspace(ext[0] + 1e-9, ext[0] + span, 600)
            line_cur, line_au = e.current_in, e.tip_in
            nodes = (f - 1, f)
        else:
            ms_side = np.linspace(ext[0], ext[0] + span, 200)
            ms_wall = np.linspace(ext[0] - span, ext[0] - 1e-9, 600)
            line_cur, line_au = e.current_out, e.tail_out
            nodes = (l, l + 1)
        top, bottom, _ = e.walls(ms_wall)
        ax.plot(ms_wall, top, color=INK, lw=1.0, ls="--", zorder=5, label="wall boundary y0 = top of the cut (passage-facing)")
        ax.plot(ms_wall, bottom, color=MUTED, lw=1.0, ls="--", zorder=5,
                label="wall boundary y1 - pitch = bottom of the cut")
        if current:
            ax.plot(ms_side, line_cur(ms_side), color=CURRENT, lw=2.0, label="current side line y0 (= y1 - pitch)")
            if side == "LE":
                ms_to = np.linspace(ext[0], anchor[0], 50)
            else:
                ms_to = np.linspace(anchor[0], ext[0], 50)
            ax.plot(ms_to, line_cur(ms_to), color=CURRENT, lw=1.0, ls=":", label="its continuation to the anchor (unused)")
        if aungier:
            ax.plot(ms_side, line_au(ms_side), color=AUNGIER, lw=2.0,
                    label="Aungier-style side line through the " + ("nose tip" if side == "LE" else "tail end"))
        for k in nodes:
            ax.plot(g.m[k], g.nodes[k, 0, 1], "o", color=NODE, ms=6, zorder=9)
            ax.plot(g.m[k], g.nodes[k, -1, 1] - e.pitch, "s", color=NODE, ms=6, zorder=9)
        ax.plot([], [], "o", color=NODE, label=f"nodes y0 (o), y1 - pitch (s) at {'f-1, f' if side == 'LE' else 'l, l+1'} "
                                               f"(finest admissible n of each stagger, in the panel title)")
        ax.plot(*ext, ">" if side == "LE" else "<", ms=9, color=INK, zorder=10, label="nose tip" if side == "LE" else "tail end")
        ax.plot(*anchor, "o", ms=8, mfc="white", mec=INK, mew=1.5, zorder=10,
                label="construction LE = SS/PS point = current anchor" if side == "LE" else "construction TE = current anchor")
        # a square view that holds the extreme point, the anchor and the four nodes
        ys = [ext[1], anchor[1]] + [g.nodes[k, 0, 1] for k in nodes] + [g.nodes[k, -1, 1] - e.pitch for k in nodes]
        ms_ = [ext[0], anchor[0]] + [g.m[k] for k in nodes]
        half = max(span, 0.6 * (max(ys) - min(ys)), 0.6 * (max(ms_) - min(ms_)))
        cm, cy = 0.5 * (max(ms_) + min(ms_)), 0.5 * (max(ys) + min(ys))
        ax.set_xlim(cm - half, cm + half)
        ax.set_ylim(cy - half, cy + half)
        ax.set_aspect("equal", adjustable="box")
        j = jumps[st]["le" if side == "LE" else "te"]
        wt.style(ax, f"{st:.0f} deg {side}, n {SEQUENCES[st][-1]}: current jump at the {'tip' if side == 'LE' else 'tail end'} "
                     f"{j:+.2e} c")

    for name, side in (("01_current_le_construction", "LE"), ("02_current_te_construction", "TE")):
        fig, axes = plt.subplots(1, 5, figsize=(24, 6.2))
        for ax, st in zip(axes, STAGGERS):
            panel_ends(ax, st, side, span=0.03 if side == "LE" else 0.012)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8, frameon=False)
        fig.suptitle(f"{name[:2]}  Current construction at the {side}: the side line ends at the construction "
                     f"{'LE' if side == 'LE' else 'TE'} point, the walls begin at the {'nose tip' if side == 'LE' else 'tail end'}"
                     f" (SS orange, PS blue)", fontsize=11, color=INK, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0.1, 1, 0.94))
        figures[name] = fig

    for name, side in (("03_aungier_intended_upstream_geometry", "LE"), ("04_aungier_intended_downstream_geometry", "TE")):
        fig, axes = plt.subplots(3, 2, figsize=(18, 14), gridspec_kw=dict(height_ratios=(1, 1.6, 1)))
        for col, st in enumerate((33.0, 53.0)):
            e = ends[st]
            b = e.b
            if side == "LE":
                ms = np.linspace(b.m_in, e.tip[0], 400)
                cur, au = e.current_in(ms), e.tip_in(ms)
                beta_c = [e.beta_le(m) for m in ms]
                beta_a = [e.tip_beta(m) for m in ms]
                note = ""
            else:
                # the default exit angle kappa_te gives a straight line; turned by -5 deg the integration shows
                turned = e.kappa_te - math.radians(5.0)
                bt = gr.make_passage_boundaries(e.cascade, GridSettings(n_passage=SEQUENCES[st][0], n_eta=3), INLET, turned)
                line_a, beta_fn = aungier_side(e.tail[0], e.tail[1], e.kappa_te, bt.m_out, turned)
                _, beta_cur = aungier_side(bt.m_te, float(e.anchor_te[1]), e.kappa_te, bt.m_out, turned)
                ms = np.linspace(e.tail[0], b.m_out, 400)
                cur, au = gr._exit_side(bt, ms), line_a(ms)
                beta_c = [beta_cur(m) for m in ms]
                beta_a = [beta_fn(m) for m in ms]
                note = " (exit angle kappa_te - 5 deg; the default kappa_te makes both lines straight)"
            ax = axes[0, col]
            ax.plot(ms, np.degrees(beta_c), color=CURRENT, lw=1.8, label="current beta(m): blade angle reached at the construction point")
            ax.plot(ms, np.degrees(beta_a), color=AUNGIER, lw=1.4, ls="--",
                    label="Aungier-style beta(m): blade angle reached at the " + ("nose tip" if side == "LE" else "tail end"))
            ax.set_ylabel("beta_side (deg)", fontsize=8)
            ax.legend(fontsize=7)
            wt.style(ax, f"{st:.0f} deg: side-boundary angle, uniform in m{note}")
            ax = axes[2, col]
            ax.plot(ms, au - cur, color=INK, lw=1.6)
            ax.axhline(0.0, color=GRIDLINE, lw=0.8)
            ax.set_xlabel("m (chords)", fontsize=8)
            ax.set_ylabel("y0 Aungier-style - y0 current (c)", fontsize=8)
            wt.style(ax, f"{st:.0f} deg: the two lines differ by an almost constant shift, the jump the current anchor leaves")
            ax = axes[1, col]
            for k in (0, 1):
                ax.fill(*(np.vstack([e.geo.curves["SS"].evaluate(np.linspace(1, 0, 400)),
                                     e.geo.curves["PS"].evaluate(np.linspace(0, 1, 400))]) + [0, k * e.pitch]).T,
                        color="#e8eaee")
            ax.plot(ms, cur, color=CURRENT, lw=1.6, label="current y0")
            ax.plot(ms, cur + e.pitch, color=CURRENT, lw=1.0, ls=":", label="current y1 = y0 + pitch")
            ax.plot(ms, au, color=AUNGIER, lw=1.4, ls="--", label="Aungier-style y0 (dy0/dm = tan beta, through the extreme point)")
            ax.plot(ms, au + e.pitch, color=AUNGIER, lw=1.0, ls="-.", label="Aungier-style y1 = y0 + pitch")
            ax.set_aspect("equal", adjustable="datalim")
            ax.legend(fontsize=7)
            wt.style(ax, f"{st:.0f} deg: the {'upstream' if side == 'LE' else 'downstream'} side boundaries, integrated")
        fig.suptitle(f"{name[:2]}  The specified {'upstream' if side == 'LE' else 'downstream'} construction: beta(m) "
                     f"uniform, y integrated, y1 = y0 + pitch; current and Aungier-style differ only in the junction",
                     fontsize=11, color=INK, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        figures[name] = fig

    for name, side in (("05_current_vs_aungier_le", "LE"), ("06_current_vs_aungier_te", "TE")):
        fig, axes = plt.subplots(2, 3, figsize=(19, 12))
        for ax, st in zip(axes.ravel()[:5], STAGGERS):
            panel_ends(ax, st, side, current=True, aungier=True, span=0.03 if side == "LE" else 0.012)
        ax = axes.ravel()[5]
        for st, color in zip(STAGGERS, plt.cm.viridis(np.linspace(0, 0.9, len(STAGGERS)))):
            e = ends[st]
            if side == "LE":
                ms, cuts = defects[st]
                ax.plot(ms - e.tip[0], cuts, color=color, label=f"{st:.0f} deg")
                ax.axvline(e.b.m_le - e.tip[0], color=color, lw=0.6, ls=":")
            else:
                ms = np.concatenate((np.linspace(e.b.m_te - 0.02, e.tail[0] - 1e-9, 400), [e.tail[0] - 1e-12]))
                ax.plot(e.tail[0] - ms, e.cut_length(ms), color=color, label=f"{st:.0f} deg")
                ax.axvline(e.tail[0] - e.b.m_te, color=color, lw=0.6, ls=":")
        ax.set_xlabel(f"distance of the junction m_J from the {'nose tip' if side == 'LE' else 'tail end'} (c); dotted: "
                      f"the construction {'LE' if side == 'LE' else 'TE'} point", fontsize=8)
        ax.set_ylabel("C0 defect = cut(m_J) (c)", fontsize=8)
        ax.legend(fontsize=7)
        wt.style(ax, "least C0 defect of a periodic side line\nat a junction m_J: zero only at the extreme point")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8, frameon=False)
        fig.suptitle(f"{name[:2]}  Current (red) against the Aungier-style side line through the "
                     f"{'nose tip' if side == 'LE' else 'tail end'} (green), with the walls and the first / last nodes",
                     fontsize=11, color=INK, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0.07, 1, 0.95))
        figures[name] = fig

    for name, deriv in (("07_boundary_y_vs_m", 0), ("08_boundary_slope_vs_m", 1)):
        fig, axes = plt.subplots(2, 2, figsize=(18, 10))
        for row_k, st in enumerate((33.0, 53.0)):
            e = ends[st]
            g = gr.make_grid(e.cascade, GridSettings(n_passage=SEQUENCES[st][-1], n_eta=3), INLET, require_ss_ps=True)
            for col, side in enumerate(("LE", "TE")):
                ax = axes[row_k, col]
                ext = e.tip if side == "LE" else e.tail
                ms = np.linspace(ext[0] - 0.03, ext[0] + 0.03, 6001)
                for anchor, color, label in (("production", CURRENT, "current"), ("tip", AUNGIER, "Aungier-style, extreme point")):
                    y0, y1, _ = wt.walls_at(e.geo, e.b, e.row, ms, anchor)
                    for part in (ms < ext[0], ms > ext[0]):
                        if deriv == 0:
                            ax.plot(ms[part], y0[part] - ext[1], color=color, lw=1.6, label=f"y0, {label}" if part[0] else None)
                            ax.plot(ms[part], y1[part] - e.pitch - ext[1], color=color, lw=1.0, ls="--",
                                    label=f"y1 - pitch, {label}" if part[0] else None)
                        else:
                            ax.plot(ms[part], np.gradient(y0[part], ms[part]), color=color, lw=1.4,
                                    label=f"dy0/dm, {label}" if part[0] else None)
                            ax.plot(ms[part], np.gradient(y1[part], ms[part]), color=color, lw=1.0, ls="--",
                                    label=f"dy1/dm, {label}" if part[0] else None)
                for k in range(len(g.m)):
                    if ms[0] < g.m[k] < ms[-1]:
                        ax.axvline(g.m[k], color=GRIDLINE, lw=0.6, zorder=0)
                ax.axvline(ext[0], color=INK, lw=0.8, ls=":")
                if deriv == 1:
                    ax.set_yscale("symlog", linthresh=1.0)
                ax.legend(fontsize=7)
                wt.style(ax, f"{st:.0f} deg {side}: {'y - y_extreme' if deriv == 0 else 'slopes'} along m "
                             f"(grey: stations of admissible n {SEQUENCES[st][-1]}; dotted: the extreme point)")
        fig.suptitle(f"{name[:2]}  {'Boundaries' if deriv == 0 else 'Boundary slopes'} through the transitions: current "
                     f"C0-discontinuous, Aungier-style through the extreme point C0 but not C1 (the walls leave it vertically)",
                     fontsize=11, color=INK, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        figures[name] = fig

    fig, axes = plt.subplots(1, 3, figsize=(19, 6))
    for st, color in zip(STAGGERS, plt.cm.viridis(np.linspace(0, 0.9, len(STAGGERS)))):
        for anchor, ls in (("production", "-o"), ("tip", "--s")):
            recs = consequences[st][anchor]
            hs = [r["dm"] for r in recs]
            label = f"{st:.0f} deg {'current' if anchor == 'production' else 'Aungier-style'}"
            axes[0].semilogx(hs, [abs(jumps[st]["le"]) if anchor == "production" else 1e-17 for _ in hs], ls, color=color,
                             label=label)
            axes[1].loglog(hs, [r["D_le"] for r in recs], ls, color=color, label=label)
            axes[2].loglog(hs, [r["psi"][0] for r in recs], ls, color=color, label=label)
    axes[0].set_yscale("log")
    wt.style(axes[0], "C0 jump of y at the nose tip (1e-17 = exactly 0): fixed by the anchor, not by refinement")
    wt.style(axes[1], "max |D| at f-3 .. f+3 (coefficients.py): the same with either anchor;\n"
                      "the tan(beta) jump at the corner stays, and D grows under refinement")
    wt.style(axes[2], "smooth field: central psi_m error at the last periodic station")
    for ax in axes:
        ax.set_xlabel("dm (admissible sequences)", fontsize=8)
        ax.legend(fontsize=6.5, ncol=2)
    fig.suptitle("09  The mapping jump against refinement on admissible grids, and what removing it changes", fontsize=11,
                 color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    figures["09_mapping_jump_vs_refinement"] = fig

    for name, fg in figures.items():
        fg.savefig(args.save / f"{name}.png", dpi=125)
        print(f"wrote {args.save / (name + '.png')}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
