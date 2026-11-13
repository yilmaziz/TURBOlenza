"""
Admissible uniform grids (diagnostic): which n_passage give a first and a last blade station
that each cut exactly one suction (SS) and one pressure (PS) surface, over stagger 13 .. 53
deg and every n_passage 20 .. 800. The meridional grid is production's, uniform; nothing is
moved or substituted.

    python scripts/audit_uniform_grid_validity.py --save outputs/uniform_grid_validity --no-show

Blade: the default 65-(12A10)10 (cl0 1.2, t/c 0.10, Kovach TE, C2 surfaces), solidity 1,
inlet angle 45 deg, one-chord extensions; only the stagger changes.

Per (stagger, n_passage) the production grid is built (make_grid, default path, n_eta 3)
and its first and last blade station are classified three ways that must agree:
    A  production metadata: Grid.wall_identity surfaces of the two wall nodes;
    P  production rule: grid.wall_station_topology (crossings per surface);
    B  independent: the C2 curves split at their own turning points, every root of
       m(u) = m_station found by brentq (scripts/audit_wall_station_topology.Geometry).
A disagreement stops the script. Then: the theory of the refinement ceiling against the
sweep, the admissible sets and their runs, refinement sequences for a convergence study,
how the first / last wall station approaches the LE / TE, the unresolved surface, and the
parts of the solver that exist (metrics, coefficients, the difference stencils on a smooth
field) on the recommended grids. No flow solution is computed (the solver does not exist).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, Path(__file__).resolve().parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

import audit_redist_numerics as ar  # noqa: E402
import audit_wall_station_topology as wt  # noqa: E402
from potential_flow_solver import grid as gr  # noqa: E402
from potential_flow_solver.coefficients import build_coefficients  # noqa: E402
from potential_flow_solver.config import GridSettings  # noqa: E402
from potential_flow_solver.metrics import build_metrics  # noqa: E402

STAGGERS = (13.0, 23.0, 33.0, 43.0, 53.0)
N_RANGE = tuple(range(20, 801))
INK, MUTED, GRIDLINE = wt.INK, wt.MUTED, wt.GRIDLINE
SUCTION, PRESSURE = wt.SUCTION, wt.PRESSURE
VALID, LE_BAD, TE_BAD, BOTH_BAD = "#1baf7a", "#eb6834", "#2a78d6", "#cf222e"
NAME = {"SS": "suction", "PS": "pressure"}


def classify_independent(geo, m: float):
    cls, roots = geo.classify(m)
    return cls, roots, sum(r.curve == "SS" for r in roots), sum(r.curve == "PS" for r in roots)


def sweep(st: float):
    geo = wt.Geometry(st)
    cas = geo.cascade
    rows = []
    for n in N_RANGE:
        g = gr.make_grid(cas, GridSettings(n_passage=n, n_eta=3), wt.INLET)
        f, l = g.first_blade_i, g.last_blade_i
        wid = g.wall_identity
        le_p, te_p = gr.wall_station_topology(g)
        rec = dict(stagger=st, n_passage=n, n_m=len(g.m), dm=g.dm, first_blade_i=f, last_blade_i=l,
                   m_tip=geo.m_tip, m_le=geo.m_le, m_te=geo.m_te, m_tail=geo.m_tail)
        for side, i, k, prod in (("LE", f, 0, le_p), ("TE", l, l - f, te_p)):
            m = float(g.m[i])
            cls, roots, ss, ps = classify_independent(geo, m)
            top, bottom = roots[-1], roots[0]
            lower_a, upper_a = str(wid.lower.surface[k]), str(wid.upper.surface[k])
            hidden = int(wid.lower.hidden[k]) + int(wid.upper.hidden[k])
            agree = (NAME[top.curve] == lower_a and NAME[bottom.curve] == upper_a and prod.crossings == len(roots)
                     == 2 + hidden and prod.suction == ss and prod.pressure == ps
                     and prod.valid == (cls == "VALID_SS_PS"))
            if not agree:
                raise SystemExit(f"classification disagreement at {st} deg n {n} {side}: production {lower_a}/{upper_a} "
                                 f"{prod}, independent {cls} {[(r.curve, r.u) for r in roots]}")
            rec.update({f"m_{'f' if side == 'LE' else 'l'}": m, f"{side}_roots": len(roots), f"{side}_SS": ss,
                        f"{side}_PS": ps, f"{side}_valid": cls == "VALID_SS_PS", f"{side}_class": cls,
                        f"{side}_lower_surface": lower_a, f"{side}_upper_surface": upper_a,
                        f"{side}_u_lower": top.u, f"{side}_u_upper": bottom.u})
        rec["m_f_minus_m_tip"] = rec["m_f"] - geo.m_tip
        rec["m_f_minus_m_le"] = rec["m_f"] - geo.m_le
        rec["m_l_minus_m_te"] = rec["m_l"] - geo.m_te
        rec["m_l_minus_m_tail"] = rec["m_l"] - geo.m_tail
        rec["valid"] = rec["LE_valid"] and rec["TE_valid"]
        rows.append(rec)
    return geo, rows


def runs(values):
    out = []
    for v in values:
        if out and v == out[-1][1] + 1:
            out[-1][1] = v
        else:
            out.append([v, v])
    return out


def theory(geo, b) -> dict:
    """The spacing ceiling, computed here from the lattice formula (independently of
    grid.n_passage_ceiling): smallest n with dm(n) <= w for each window."""
    width, length = b.m_out - b.m_in, b.m_te - b.m_le

    def dm(n):
        return width / max(1, round(width / (length / n)))

    out = {}
    for side, w in (("LE", geo.m_le - geo.m_tip), ("TE", geo.m_tail - geo.m_te)):
        n = 1
        while dm(n) > w:
            n += 1
        out[side] = (w, n)
    out["overall"] = min(out["LE"][1], out["TE"][1])
    return out


def recommend(valid, dm_of, delta_of=None, floors=(0.0,)):
    """Admissible grids for a convergence study: four (three if four are impossible),
    refinement ratios h_k / h_k+1 >= 1.2 and as equal as the admissible set allows (spread =
    max / min ratio), near 1.5, the finest as fine as possible. With delta_of (the first wall
    station's offset past the LE point in dm, equal to the last one's before the TE point
    on these symmetric lattices) only grids with delta >= floor are used, the first floor
    that works, and similar delta is preferred: the corner stations then sit alike on every
    grid. Returns (sequence, ratios, floor)."""
    for floor in floors:
        pool = [n for n in valid if delta_of is None or delta_of(n) >= floor]
        best = None
        for count in (4, 3):
            for n_top in sorted(pool)[-20:][::-1]:
                for r in np.arange(1.2, 2.05, 0.05):
                    seq = [n_top]
                    for k in range(1, count):
                        target = dm_of(n_top) * r ** k
                        seq.append(min(pool, key=lambda n: abs(dm_of(n) - target)))
                    seq = sorted(set(seq))
                    if len(seq) < count:
                        continue
                    hs = [dm_of(n) for n in seq]
                    ratios = [hs[i] / hs[i + 1] for i in range(len(hs) - 1)]
                    if min(ratios) < 1.2:
                        continue
                    spread = max(ratios) / min(ratios)
                    rank = sorted(pool)[::-1].index(n_top)
                    score = (spread - 1.0) + 0.01 * rank + 0.1 * abs(float(np.exp(np.mean(np.log(ratios)))) - 1.5)
                    if delta_of is not None:
                        ds = [delta_of(n) for n in seq]
                        score += 0.5 * (max(ds) - min(ds))
                    if best is None or score < best[0]:
                        best = (score, seq, ratios)
            if best is not None:
                return best[1], best[2], floor
    return [], [], None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--save", type=Path, default=None)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    out_dir = args.save
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    results, geos, extras = {}, {}, {}
    print(f"1. Dense sweep n_passage {N_RANGE[0]} .. {N_RANGE[-1]}: production grid, three classifications (A metadata, "
          f"P production rule, B independent C2 roots) agree at every station checked")
    for st in STAGGERS:
        geo, rows = sweep(st)
        geos[st], results[st] = geo, rows
        g0 = gr.make_grid(geo.cascade, GridSettings(n_passage=101, n_eta=3), wt.INLET)
        b, row = g0.boundaries, gr._row_of(g0.boundaries)
        valid = [r["n_passage"] for r in rows if r["valid"]]
        api = gr.find_valid_n_passage(geo.cascade, GridSettings(n_passage=101, n_eta=3), wt.INLET, N_RANGE)
        if tuple(valid) != api:
            raise SystemExit(f"find_valid_n_passage differs from the sweep at {st} deg")
        th = theory(geo, b)
        ceiling_api = gr.n_passage_ceiling(geo.cascade, GridSettings(n_passage=101, n_eta=3), wt.INLET)
        after = [r for r in rows if r["n_passage"] >= th["overall"]]
        last_valid = max(valid) if valid else None
        # the relations behind the classification
        le_rel = all((r["m_f"] > geo.m_le + 1e-12) == r["LE_valid"] for r in rows)
        te_rel = all((r["m_l"] < geo.m_te - 1e-12) == r["TE_valid"] for r in rows)
        le_in = all(geo.m_tip < r["m_f"] <= geo.m_tip + r["dm"] * (1 + 1e-12) for r in rows)
        te_in = all(geo.m_tail - r["dm"] * (1 + 1e-12) <= r["m_l"] < geo.m_tail for r in rows)
        le_fail = [r["n_passage"] for r in rows if not r["LE_valid"]]
        te_fail = [r["n_passage"] for r in rows if not r["TE_valid"]]
        extras[st] = dict(b=b, row=row, valid=valid, theory=th, ceiling_api=ceiling_api, last_valid=last_valid,
                          le_fail=le_fail, te_fail=te_fail)
        print(f"\n   {st:.0f} deg: w_LE = m_le - m_tip = {th['LE'][0]:.4e} c, w_TE = m_tail - m_te = {th['TE'][0]:.4e} c")
        print(f"      relations over the sweep: m_tip < m_f <= m_tip + dm {le_in}; m_tail - dm <= m_l < m_tail {te_in}; "
              f"LE valid <=> m_f > m_le {le_rel}; TE valid <=> m_l < m_te {te_rel}")
        print(f"      theory: dm <= w_LE from n = {th['LE'][1]}, dm <= w_TE from n = {th['TE'][1]}; overall ceiling "
              f"{th['overall']} (grid.n_passage_ceiling {ceiling_api}); every n >= ceiling in the sweep invalid: "
              f"{all(not r['valid'] for r in after)}; LE-invalid for every n >= {th['LE'][1]}: "
              f"{all(not r['LE_valid'] for r in rows if r['n_passage'] >= th['LE'][1])}")
        print(f"      admissible: {len(valid)} of {len(rows)}; largest admissible {last_valid}; every n > {last_valid} "
              f"invalid in the sweep (first permanently invalid n = {last_valid + 1 if last_valid else None})")
        print(f"      LE-invalid {len(le_fail)}, TE-invalid {len(te_fail)}, both "
              f"{len(set(le_fail) & set(te_fail))}")
        print(f"      runs: " + ", ".join(f"{a}-{b_}" if a != b_ else f"{a}" for a, b_ in runs(valid)))
        print(f"      valid_n_passage = {valid}")

    if out_dir is not None:
        keys = ["stagger", "n_passage", "n_m", "dm", "first_blade_i", "last_blade_i", "m_tip", "m_le", "m_f",
                "m_f_minus_m_tip", "m_f_minus_m_le", "LE_roots", "LE_SS", "LE_PS", "LE_valid", "LE_class",
                "LE_lower_surface", "LE_upper_surface", "m_te", "m_tail", "m_l", "m_l_minus_m_te", "m_l_minus_m_tail",
                "TE_roots", "TE_SS", "TE_PS", "TE_valid", "TE_class", "TE_lower_surface", "TE_upper_surface", "valid"]
        with open(out_dir / "uniform_grid_validity.csv", "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            for st in STAGGERS:
                writer.writerows(results[st])
        with open(out_dir / "valid_n_passage.json", "w", encoding="utf-8") as fh:
            json.dump({f"{st:.0f}": extras[st]["valid"] for st in STAGGERS}, fh, indent=1)
        print(f"\nwrote {out_dir / 'uniform_grid_validity.csv'} and {out_dir / 'valid_n_passage.json'}")

    print("\n2. Refinement sequences (admissible at LE and TE), spacing ratios r = h_coarse / h_finer. Recommended: the "
          "corner stations alike on every grid (delta = (m_f - m_le) / dm >= 0.2, similar); reach: as fine as admissible, "
          "delta free")
    sequences, reach = {}, {}
    for st in STAGGERS:
        rows = {r["n_passage"]: r for r in results[st]}
        dm_of = lambda n, rows=rows: rows[n]["dm"]  # noqa: E731
        delta_of = lambda n, rows=rows: rows[n]["m_f_minus_m_le"] / rows[n]["dm"]  # noqa: E731
        seq, ratios, floor = recommend(extras[st]["valid"], dm_of, delta_of, floors=(0.2, 0.1, 0.05, 0.0))
        seq_r, ratios_r, _ = recommend(extras[st]["valid"], dm_of)
        sequences[st], reach[st] = seq, seq_r
        spacings = ", ".join("%.4e" % rows[n]["dm"] for n in seq)
        print(f"   {st:.0f} deg recommended: n_passage {seq}, dm [{spacings}], ratios {[round(x, 3) for x in ratios]}, "
              f"delta {[round(delta_of(n), 3) for n in seq]} (floor {floor})")
        spacings = ", ".join("%.4e" % rows[n]["dm"] for n in seq_r)
        print(f"   {st:.0f} deg reach:       n_passage {seq_r}, dm [{spacings}], ratios {[round(x, 3) for x in ratios_r]}, "
              f"delta {[round(delta_of(n), 3) for n in seq_r]}")

    print("\n3. On the recommended grids: wall-station distances, unresolved surface, metrics, coefficients, the "
          "difference stencils on a smooth physical field, the side-line jump")
    seq_data = {}
    for st in STAGGERS:
        geo, b, row = geos[st], extras[st]["b"], extras[st]["row"]
        total = ar.arc(geo.curves["SS"], 0, 1) + ar.arc(geo.curves["PS"], 0, 1)
        jump = wt.mapping_jumps(geo, b, row, "production")
        seq_data[st] = []
        for n in sequences[st] + [n for n in reach[st] if n not in sequences[st]]:
            rec = next(r for r in results[st] if r["n_passage"] == n)
            g = gr.make_grid(geo.cascade, GridSettings(n_passage=n, n_eta=21), wt.INLET, require_ss_ps=True)
            met = build_metrics(g)
            co = build_coefficients(met, np.ones((len(g.m), len(g.eta))), 1.0)
            f, l = g.first_blade_i, g.last_blade_i
            arcs = dict(le_ss=ar.arc(geo.curves["SS"], 0.0, rec["LE_u_lower"]),
                        le_ps=ar.arc(geo.curves["PS"], 0.0, rec["LE_u_upper"]),
                        te_ss=ar.arc(geo.curves["SS"], rec["TE_u_lower"], 1.0),
                        te_ps=ar.arc(geo.curves["PS"], rec["TE_u_upper"], 1.0))
            case = ar.Case(geo, b, row, "PROD", n, metrics_needed=False)
            phys = ar.physical_errors(case)
            nb_le, nb_te = slice(f - 3, f + 4), slice(l - 3, l + 4)
            d = dict(n=n, dm=g.dm, d_le=rec["m_f_minus_m_le"], d_te=-rec["m_l_minus_m_te"], arcs=arcs, total=total,
                     tb0=float(met.tan_beta0[f]), tb1=float(met.tan_beta1[f]), dtb0=float(met.dtan_beta0_dm[f]),
                     dtb1=float(met.dtan_beta1_dm[f]), tb0_l=float(met.tan_beta0[l]), dtb0_l=float(met.dtan_beta0_dm[l]),
                     D_le=float(np.abs(co.D[nb_le]).max()), E_le=float(np.abs(co.E[nb_le]).max()),
                     D_te=float(np.abs(co.D[nb_te]).max()), E_te=float(np.abs(co.E[nb_te]).max()),
                     D_rest=float(np.abs(np.delete(co.D, list(range(f - 3, f + 4)) + list(range(l - 3, l + 4)), axis=0)).max()),
                     phys=phys, jump=jump)
            d["recommended"] = n in sequences[st]
            seq_data[st].append(d)
            print(f"   {st:.0f} deg n {n} ({'recommended' if d['recommended'] else 'reach'}): dm {g.dm:.4e}; m_f - m_le {d['d_le']:+.4e} c ({d['d_le'] / g.dm:.3f} dm), m_te - m_l "
                  f"{d['d_te']:+.4e} c ({d['d_te'] / g.dm:.3f} dm); unresolved arc LE SS {arcs['le_ss']:.4f} PS "
                  f"{arcs['le_ps']:.4f} ({100 * (arcs['le_ss'] + arcs['le_ps']) / total:.2f} %), TE SS {arcs['te_ss']:.4f} PS "
                  f"{arcs['te_ps']:.4f} ({100 * (arcs['te_ss'] + arcs['te_ps']) / total:.2f} %)")
            print(f"      metrics at f: tan b0 {d['tb0']:+.3f}, tan b1 {d['tb1']:+.3f}, dtan b0/dm {d['dtb0']:+.1f}, dtan b1/dm "
                  f"{d['dtb1']:+.1f}; at l: tan b0 {d['tb0_l']:+.3f}, dtan b0/dm {d['dtb0_l']:+.1f}; |D| LE {d['D_le']:.1f}, "
                  f"TE {d['D_te']:.1f}, elsewhere {d['D_rest']:.1f}; |E| LE {d['E_le']:.2f}, TE {d['E_te']:.2f}")
            print("      smooth physical field, |psi_m error| central / one-sided, |psi_mm| central: " + "; ".join(
                f"{k} {v[0]:.2e} / {v[1]:.1e}, {v[2]:.1e}" for k, v in phys.items()))
        print(f"   {st:.0f} deg side-line jump at the nose tip {jump['le']:+.4e} c, at the tail end {jump['te']:+.4e} c "
              f"(the same on every grid: set by the side-line anchor, not by the stations)")

    if out_dir is None:
        return
    figures = {}
    # 01 validity map
    fig, ax = plt.subplots(figsize=(18, 5.2))
    code = np.array([[0 if r["valid"] else (1 if not r["LE_valid"] and r["TE_valid"] else (2 if r["LE_valid"] else 3))
                      for r in results[st]] for st in STAGGERS])
    ax.imshow(code, aspect="auto", cmap=ListedColormap([VALID, LE_BAD, TE_BAD, BOTH_BAD]), vmin=0, vmax=3,
              interpolation="nearest", extent=(N_RANGE[0] - 0.5, N_RANGE[-1] + 0.5, len(STAGGERS) - 0.5, -0.5))
    for k, st in enumerate(STAGGERS):
        c = extras[st]["theory"]["overall"]
        ax.plot([c, c], [k - 0.45, k + 0.45], color=INK, lw=2.0)
        ax.text(c + 3, k - 0.25, f"ceiling {c}", fontsize=8, color=INK)
        lv = extras[st]["last_valid"]
        ax.plot([lv, lv], [k - 0.45, k + 0.45], color="white", lw=1.2, ls=":")
    ax.set_yticks(range(len(STAGGERS)), [f"{s:.0f} deg" for s in STAGGERS])
    ax.set_xlabel("n_passage", fontsize=9)
    for color, label in ((VALID, "valid (SS + PS at both ends)"), (LE_BAD, "LE-invalid"), (TE_BAD, "TE-invalid"),
                         (BOTH_BAD, "both invalid")):
        ax.plot([], [], "s", color=color, ms=10, label=label)
    ax.plot([], [], "-", color=INK, lw=2, label="theoretical ceiling (dm <= window)")
    ax.plot([], [], ":", color=MUTED, lw=1.2, label="largest admissible n")
    ax.legend(fontsize=8, ncol=6, loc="upper center", bbox_to_anchor=(0.5, -0.14), frameon=False)
    wt.style(ax, f"01  Admissible uniform grids, n_passage {N_RANGE[0]} .. {N_RANGE[-1]}: aliased below the ceiling, "
                 f"invalid at and beyond it")
    fig.tight_layout()
    figures["01_validity_map"] = fig

    # 02 validity by stagger: offsets in dm
    fig, axes = plt.subplots(len(STAGGERS), 1, figsize=(18, 3.0 * len(STAGGERS)), sharex=True)
    for ax, st in zip(axes, STAGGERS):
        rs = results[st]
        ns = np.array([r["n_passage"] for r in rs])
        le = np.array([r["m_f_minus_m_le"] / r["dm"] for r in rs])
        te = np.array([r["m_l_minus_m_te"] / r["dm"] for r in rs])
        w_le = np.array([(geos[st].m_le - geos[st].m_tip) / r["dm"] for r in rs])
        w_te = np.array([(geos[st].m_tail - geos[st].m_te) / r["dm"] for r in rs])
        ax.fill_between(ns, -w_le, 0, color=LE_BAD, alpha=0.12, lw=0)
        ax.fill_between(ns, 0, w_te, color=TE_BAD, alpha=0.12, lw=0)
        ok = np.array([r["valid"] for r in rs])
        ax.plot(ns[ok], le[ok], "o", ms=2.4, color=VALID, label="(m_f - m_le) / dm, admissible grid")
        ax.plot(ns[~ok], le[~ok], "o", ms=2.0, color=LE_BAD, alpha=0.6, label="(m_f - m_le) / dm, inadmissible")
        ax.plot(ns[ok], te[ok], "s", ms=2.0, color="#0b6e4f", label="(m_l - m_te) / dm, admissible")
        ax.plot(ns[~ok], te[~ok], "s", ms=1.8, color=TE_BAD, alpha=0.6, label="(m_l - m_te) / dm, inadmissible")
        ax.axhline(0, color=INK, lw=0.6)
        ax.axvline(extras[st]["theory"]["overall"], color=INK, lw=1.2)
        ax.set_ylim(-max(1.6, min(8, w_le.max() * 1.05)), 1.1)
        wt.style(ax, f"{st:.0f} deg: {len(extras[st]['valid'])} admissible; orange band: LE window (nose tip .. LE point) "
                     f"in dm, blue: TE window; black line: ceiling {extras[st]['theory']['overall']}")
    axes[0].legend(fontsize=7, ncol=4, loc="lower left")
    axes[-1].set_xlabel("n_passage", fontsize=9)
    fig.suptitle("02  First and last blade station relative to the LE / TE point, in dm: valid exactly when the first "
                 "lies past the LE point and the last before the TE point", fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    figures["02_validity_by_stagger"] = fig

    # 03 / 04 distances on admissible grids
    for name, key, label, sign in (("03_le_station_distance", "m_f_minus_m_le", "m_f - m_le", 1.0),
                                   ("04_te_station_distance", "m_l_minus_m_te", "m_te - m_l", -1.0)):
        fig, axes = plt.subplots(1, 2, figsize=(17, 6))
        for st, color in zip(STAGGERS, plt.cm.viridis(np.linspace(0, 0.9, len(STAGGERS)))):
            rs = [r for r in results[st] if r["valid"]]
            ns = [r["n_passage"] for r in rs]
            axes[0].plot(ns, [sign * r[key] for r in rs], "o", ms=2.5, color=color, label=f"{st:.0f} deg")
            axes[1].plot(ns, [sign * r[key] / r["dm"] for r in rs], "o", ms=2.5, color=color, label=f"{st:.0f} deg")
            nseq = sequences[st]
            axes[0].plot(nseq, [sign * next(r for r in rs if r["n_passage"] == n)[key] for n in nseq], "x", ms=9,
                         color=color, mew=2)
        axes[0].set_yscale("log")
        axes[0].set_xscale("log")
        axes[1].set_xscale("log")
        for ax in axes:
            ax.set_xlabel("n_passage (admissible grids only; x = recommended sequence)", fontsize=8)
            ax.legend(fontsize=7)
        wt.style(axes[0], f"({label}) / chord")
        wt.style(axes[1], f"({label}) / dm")
        fig.suptitle(f"{name[:2]}  {'First wall station past the LE point' if sign > 0 else 'Last wall station ahead of the TE point'}"
                     f" on the admissible grids", fontsize=11, color=INK, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        figures[name] = fig

    # 05 .. 09 per stagger: coarse valid, finest valid, first permanently invalid (LE); TE row
    for idx, st in enumerate(STAGGERS, start=5):
        geo = geos[st]
        valid = extras[st]["valid"]
        te_bad = [n for n in extras[st]["te_fail"] if n < extras[st]["theory"]["overall"]] or extras[st]["te_fail"]
        picks_le = [valid[0], valid[-1], valid[-1] + 1]
        picks_te = [valid[0], valid[-1], te_bad[-1] if te_bad else valid[-1] + 1]
        fig, axes = plt.subplots(2, 3, figsize=(17, 11))
        for row_k, (side, picks) in enumerate((("LE", picks_le), ("TE", picks_te))):
            for col, n in enumerate(picks):
                g = gr.make_grid(geo.cascade, GridSettings(n_passage=n, n_eta=3), wt.INLET)
                wt.station_panel(axes[row_k, col], geo, dict(geo=geo, grid=g, n=n), side)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8, frameon=False)
        fig.suptitle(f"{idx:02d}  {st:.0f} deg: LE (top) coarse admissible n {picks_le[0]}, finest admissible n {picks_le[1]}, "
                     f"first permanently invalid n {picks_le[2]}; TE (bottom) n {picks_te[0]}, {picks_te[1]}, "
                     f"{picks_te[2]} (a TE-invalid value)", fontsize=10.5, color=INK, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0.06, 1, 0.95))
        figures[f"{idx:02d}_stagger{st:.0f}_valid_grids"] = fig

    # 10 recommended sequences
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    ax = axes[0, 0]
    for k, st in enumerate(STAGGERS):
        ax.plot(extras[st]["valid"], [k] * len(extras[st]["valid"]), "|", ms=8, color=VALID)
        ax.plot(sequences[st], [k] * len(sequences[st]), "o", ms=8, mfc="none", mec=INK, mew=1.6)
        ax.plot(reach[st], [k + 0.25] * len(reach[st]), "x", ms=7, color=MUTED, mew=1.4)
        ax.text(max(extras[st]["valid"]) + 8, k, " " + ", ".join(str(n) for n in sequences[st]) + "  (reach " +
                ", ".join(str(n) for n in reach[st]) + ")", va="center", fontsize=7.5)
    ax.set_yticks(range(len(STAGGERS)), [f"{s:.0f} deg" for s in STAGGERS])
    ax.set_xlabel("n_passage", fontsize=8)
    wt.style(ax, "admissible n (green), recommended sequence (circles), finest-reaching sequence (crosses)")
    panels = (("d_le", "(m_f - m_le) / c", axes[0, 1]), ("d_te", "(m_te - m_l) / c", axes[0, 2]),
              ("arc_le", "unresolved LE arc (SS + PS) / surface length (%)", axes[1, 0]),
              ("dtb", "|d tan beta0/dm| at the first wall station", axes[1, 1]),
              ("D", "max |D| near the LE (coefficients.py, rho = 1)", axes[1, 2]))
    for key, title, ax in panels:
        for st, color in zip(STAGGERS, plt.cm.viridis(np.linspace(0, 0.9, len(STAGGERS)))):
            for which, ls in ((True, "o-"), (False, "x--")):
                ds = sorted([d for d in seq_data[st] if d["recommended"] == which or (not which and d["n"] in reach[st])],
                            key=lambda d: d["dm"]) if which else sorted([d for d in seq_data[st] if d["n"] in reach[st]],
                                                                         key=lambda d: d["dm"])
                hs = [d["dm"] for d in ds]
                vals = {"d_le": [d["d_le"] for d in ds], "d_te": [d["d_te"] for d in ds],
                        "arc_le": [100 * (d["arcs"]["le_ss"] + d["arcs"]["le_ps"]) / d["total"] for d in ds],
                        "dtb": [abs(d["dtb0"]) if key == "dtb" else 0 for d in ds], "D": [d["D_le"] for d in ds]}[key]
                ax.loglog(hs, vals, ls, color=color, label=f"{st:.0f} deg" + ("" if which else " reach"))
        ax.set_xlabel("dm (recommended: solid; finest-reaching: dashed)", fontsize=8)
        ax.legend(fontsize=7)
        wt.style(ax, title)
    fig.suptitle("10  Recommended refinement sequences and what changes along them", fontsize=11, color=INK, x=0.01,
                 ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    figures["10_recommended_refinement_sequences"] = fig

    for name, fg in figures.items():
        fg.savefig(out_dir / f"{name}.png", dpi=125)
        print(f"wrote {out_dir / (name + '.png')}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
