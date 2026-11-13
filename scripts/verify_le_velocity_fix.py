"""
verify_le_velocity_fix.py
=========================
Acceptance runs of the 2026-11-13 production fix of potential_flow_solver/flow.py: psi_m is
taken by m region (metrics.py's three regions, one-sided at the four transition stations
f - 1, f, l, l + 1) instead of by central differences across the leading and trailing edge.

    python scripts/verify_le_velocity_fix.py

Nothing is patched here: every case is the production solver, built by
potential_flow_solver.controller on the frozen baseline of scripts/diagnose_mach060_vs_065.py
(61 x 41, lambda_rho = lambda_K = 0.7, all tolerances 1e-8), with the normal sonic stop ON
(IterationSettings.stop_at_sonic True). One extra run at M 0.70 has the stop off, so that it
can be compared with the diagnosis run of the same case.

What it reports (OUTPUT_DIR: report.txt, CSV tables, PNG figures, the fields and traces)
 1. Equivalence with the diagnosis: every run against the saved what-if V run of the same case
    (outputs/mach060_vs_065_diagnosis, the variant that took the same one-sided stencils by a
    patch): fields, exit angle, pass count, and the per-pass traces of stations f - 3 .. f + 5.
 2. The critical node of the diagnosis (read from its trace: the peak-Mach node of the first
    pass with M >= 1 in the old M 0.65 run), before and after: psi_m, W_m, W_theta, W, M, rho.
 3. M(eta) at the first blade station, before and after (figure and CSV), and the f - 1, f,
    f + 1 table (max M, mean M, eta of the max).
 4. The blade-surface Mach distribution (SS / PS peak and x/c, TE Mach, first five PS points).
 5. The grid study at M 0.65 and the Mach sweep, both with the stop on, and the first case
    that stops: where its sonic node is.

"Before" is the saved production data of the diagnosis (psi_m central across the transitions),
read from DIAGNOSIS_DIR; this script never runs the old rule.
"""

# ============================================================
# SETTINGS
# ============================================================

OUTPUT_DIR = "outputs/le_velocity_fix_verification"
DIAGNOSIS_DIR = "outputs/mach060_vs_065_diagnosis"      # the "before" data and the what-if V runs
N_WORKERS = 8
SWEEP_MACHS = (0.60, 0.625, 0.65, 0.675, 0.68, 0.70)    # inlet Mach numbers, baseline grid, stop on
GRIDS = ((43, 29), (61, 41), (87, 59))                  # (n_passage, n_eta) of the grid study
GRID_MACH = 0.65
TRACED_MACHS = (0.60, 0.65)                             # traced runs (stations f - 3 .. f + 5, every pass)
EQUIVALENCE_TOLERANCE = 1e-8                            # the brief's floating-point agreement

# ============================================================

import csv                                                                       # noqa: E402
import importlib.util                                                            # noqa: E402
import json                                                                      # noqa: E402
import math                                                                      # noqa: E402
import sys                                                                       # noqa: E402
from concurrent.futures import ProcessPoolExecutor                               # noqa: E402
from pathlib import Path                                                         # noqa: E402

import numpy as np                                                               # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / OUTPUT_DIR
BEFORE = ROOT / DIAGNOSIS_DIR
FIELD_KEYS = ("density", "psi", "W", "mach", "pressure", "mach_ss", "mach_ps", "x_c_ss", "x_c_ps")
_DIAGNOSIS = None
LINES: list = []


def diagnosis():
    """scripts/diagnose_mach060_vs_065.py as a module (its baseline, spec and run_job), writing
    to OUTPUT_DIR."""
    global _DIAGNOSIS
    if _DIAGNOSIS is None:
        spec = importlib.util.spec_from_file_location("diagnose_mach060_vs_065", ROOT / "scripts" / "diagnose_mach060_vs_065.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.OUT = OUT
        _DIAGNOSIS = module
    return _DIAGNOSIS


def worker(job: dict) -> dict:
    return diagnosis().safe_job(job)


def tag_of(mach: float, grid=None) -> str:
    return f"P_{grid[0]}x{grid[1]}_M{mach:.3f}" if grid else f"P_M{mach:.3f}"


def jobs() -> list:
    dg = diagnosis()
    out = [dg.spec(tag_of(m), m, f"production M {m:.3f}, stop on", stop=True, trace=m in TRACED_MACHS) for m in SWEEP_MACHS]
    base = (dg.BASELINE["n_passage"], dg.BASELINE["n_eta"])
    for grid in GRIDS:
        if grid != base:
            out.append(dg.spec(tag_of(GRID_MACH, grid), GRID_MACH, f"production M {GRID_MACH:.2f}, {grid[0]} x {grid[1]}, stop on",
                               n_passage=grid[0], n_eta=grid[1], stop=True))
    out.append(dg.spec("P_M0.700_stop_off", 0.70, "production M 0.700, stop OFF (for the equivalence)", stop=False))
    return out


# --- output helpers ---------------------------------------------------------------------------------------------

def say(text: str = "") -> None:
    print(text)
    LINES.append(text)


def fmt(value, digits=4) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}g}" if (abs(value) < 1e-3 or abs(value) >= 1e5) and value != 0.0 else f"{value:.{digits}f}"
    return str(value)


def table(rows: list, columns: list, title: str, digits=4) -> None:
    say(title)
    cells = [[fmt(r.get(k), digits) for k, _ in columns] for r in rows]
    widths = [max(len(h), *(len(c[n]) for c in cells)) if cells else len(h) for n, (_, h) in enumerate(columns)]
    say("  " + "  ".join(h.ljust(w) for (_, h), w in zip(columns, widths)))
    for c in cells:
        say("  " + "  ".join(x.ljust(w) for x, w in zip(c, widths)))
    say()


def write_csv(name: str, rows: list) -> None:
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(OUT / name, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


# --- 1. equivalence with the diagnostic what-if V ---------------------------------------------------------------

EQUIVALENT = {tag_of(0.60): "W_V_M0.60", tag_of(0.65): "W_V_M0.65", tag_of(0.68): "W_V_M0.68",
              tag_of(0.65, (43, 29)): "W_V_43x29_M0.65", tag_of(0.65, (87, 59)): "W_V_87x59_M0.65",
              "P_M0.700_stop_off": "W_V_M0.70"}


def equivalence(rows: dict, before_rows: dict) -> list:
    out = []
    for new, old in EQUIVALENT.items():
        if new not in rows or not (OUT / "fields" / f"{new}.npz").exists():
            continue
        a, b = np.load(OUT / "fields" / f"{new}.npz"), np.load(BEFORE / "fields" / f"{old}.npz")
        diffs = {k: float(np.max(np.abs(a[k] - b[k]))) for k in FIELD_KEYS}
        row = dict(production=new, what_if=old, passes=rows[new].get("passes"), passes_what_if=before_rows[old].get("passes"),
                   exit_angle_deg_diff=abs(math.degrees(float(a["exit_angle"]) - float(b["exit_angle"]))),
                   **{f"max_diff_{k}": v for k, v in diffs.items()},
                   bitwise=all(np.array_equal(a[k], b[k]) for k in FIELD_KEYS) and float(a["exit_angle"]) == float(b["exit_angle"]))
        trace_new, trace_old = OUT / "traces" / f"{new}.npz", BEFORE / "traces" / f"{old}.npz"
        if trace_new.exists() and trace_old.exists():
            p, q = np.load(trace_new), np.load(trace_old)
            names = [*diagnosis().TRACE_NAMES, "peak", "passes"]
            same_shape = all(p[n].shape == q[n].shape for n in names)
            row["trace_max_diff"] = max(float(np.nanmax(np.abs(p[n] - q[n]))) for n in names) if same_shape else None
            row["trace_bitwise"] = same_shape and all(np.array_equal(p[n], q[n], equal_nan=True) for n in names)
            for n in ("psi_m", "W_m", "W_theta"):
                row[f"trace_diff_{n}"] = float(np.max(np.abs(p[n] - q[n]))) if same_shape else None
        row["agree"] = (row["passes"] == row["passes_what_if"] and max(diffs.values()) < EQUIVALENCE_TOLERANCE
                        and row["exit_angle_deg_diff"] < EQUIVALENCE_TOLERANCE
                        and (row.get("trace_max_diff") is None and not trace_new.exists() or
                             (row.get("trace_max_diff") is not None and row["trace_max_diff"] < EQUIVALENCE_TOLERANCE)))
        out.append(row)
    return out


# --- 2. / 3. the critical node and the first blade station ------------------------------------------------------

def pass_row(trace, index: int) -> int:
    """The row of a trace holding loop pass `index` (the final re-evaluation is the last row)."""
    hits = np.flatnonzero((trace["passes"][:, 0] == index) & (trace["passes"][:, 1] == 0))
    return int(hits[0])


def critical_node(old065) -> tuple:
    """The peak-Mach node of the first pass with M >= 1 in the old M 0.65 trace (the diagnosis'
    critical node), and that pass."""
    k = int(np.flatnonzero(old065["passes"][:, 7] >= 1.0)[0])
    i, j = int(old065["peak"][k, 0]), int(old065["peak"][k, 1])
    return i, j, int(old065["passes"][k, 0])


def node_values(trace, row: int, i: int, j: int) -> dict:
    k = i - int(trace["window_first"])
    return {name: float(trace[name][row, k, j]) for name in ("psi_m", "psi_eta", "W_m", "W_theta", "W", "mach", "rho_res", "a")}


def station_stats(trace, row: int, offsets=(-1, 0, 1)) -> list:
    f, first, eta = int(trace["f"]), int(trace["window_first"]), np.asarray(trace["eta"])
    out = []
    for d in offsets:
        mach = trace["mach"][row, f + d - first]
        out.append(dict(station=f"f{d:+d}" if d else "f", max_M=float(np.max(mach)), mean_M=float(np.mean(mach)),
                        eta_of_max=float(eta[int(np.argmax(mach))]), j_of_max=int(np.argmax(mach))))
    return out


# --- 4. the blade surface ----------------------------------------------------------------------------------------

def surface(path: Path) -> dict:
    s = np.load(path)
    ss, ps = int(np.argmax(s["mach_ss"])), int(np.argmax(s["mach_ps"]))
    return dict(ss_peak=float(s["mach_ss"][ss]), ss_peak_x_c=float(s["x_c_ss"][ss]), ps_peak=float(s["mach_ps"][ps]),
                ps_peak_x_c=float(s["x_c_ps"][ps]), te_ss=float(s["mach_ss"][-1]), te_ps=float(s["mach_ps"][-1]),
                first_ps=[(float(x), float(m)) for x, m in zip(s["x_c_ps"][:5], s["mach_ps"][:5])],
                first_ss=[(float(x), float(m)) for x, m in zip(s["x_c_ss"][:5], s["mach_ss"][:5])],
                beta_out_deg=math.degrees(float(s["exit_angle"])))


# --- figures ----------------------------------------------------------------------------------------------------

BEFORE_INK, AFTER_INK, AFTER_EARLY_INK = "#7a7a7a", "#2a6fdb", "#9cbcf0"


def profile_figure(curves: list, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, mach in zip(axes, TRACED_MACHS):
        for c in (c for c in curves if c["mach"] == mach):
            ax.plot(c["mach_profile"], c["eta"], color=c["ink"], ls=c["ls"], lw=1.8, label=c["label"],
                    marker="o" if c["ls"] == "-" else None, ms=3)
        ax.axvline(1.0, color="#c0392b", lw=1.0, ls=":", label="M = 1")
        ax.set_title(f"M1 = {mach:.2f}: first blade station f")
        ax.set_xlabel("M (relative)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="best")
    axes[0].set_ylabel("eta (0 = lower wall)")
    fig.suptitle("Region-wise psi_m (flow.py, 2026-11-13): M(eta) at f before and after", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def surface_figure(curves: list, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.6))
    for label, file, ink, ls in curves:
        s = np.load(file)
        ax.plot(s["x_c_ss"], s["mach_ss"], color=ink, ls=ls, lw=1.6, label=f"{label} SS")
        ax.plot(s["x_c_ps"], s["mach_ps"], color=ink, ls=ls, lw=1.0, alpha=0.8, label=f"{label} PS")
    ax.axhline(1.0, color="#c0392b", lw=1.0, ls=":", label="M = 1")
    ax.set_xlabel("x / c")
    ax.set_ylabel("wall Mach number")
    ax.set_title("Blade-surface Mach number (production, region-wise psi_m)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --- main -------------------------------------------------------------------------------------------------------

def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Acceptance runs of the region-wise psi_m of flow.py")
    parser.add_argument("--report-only", action="store_true", help="tabulate the runs on file, run nothing")
    args = parser.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    rows_file = OUT / "rows.json"
    if args.report_only:
        rows = json.loads(rows_file.read_text(encoding="utf-8"))
    else:
        todo = jobs()
        with ProcessPoolExecutor(max_workers=min(N_WORKERS, len(todo))) as pool:
            rows = {r["tag"]: r for r in pool.map(worker, todo)}
        rows_file.write_text(json.dumps(rows, indent=1, default=str), encoding="utf-8")
    before_rows = {r["tag"]: r for r in json.loads((BEFORE / "rows.json").read_text(encoding="utf-8"))} \
        if isinstance(json.loads((BEFORE / "rows.json").read_text(encoding="utf-8")), list) else \
        json.loads((BEFORE / "rows.json").read_text(encoding="utf-8"))

    from potential_flow_solver import flow as _flow
    say("verify_le_velocity_fix: production flow.py (region-wise psi_m), no patch")
    say(f"flow._derivatives from {_flow._derivatives.__module__}; every case checked by check_production before it ran")
    say()

    # 1. equivalence
    eq = equivalence(rows, before_rows)
    write_csv("equivalence_with_what_if_V.csv", eq)
    table(eq, [("production", "production run"), ("what_if", "diagnosis what-if V"), ("passes", "passes"),
               ("passes_what_if", "what-if passes"), ("max_diff_density", "max|d rho|"), ("max_diff_psi", "max|d psi|"),
               ("max_diff_W", "max|d W|"), ("max_diff_mach", "max|d M|"), ("exit_angle_deg_diff", "|d beta| deg"),
               ("trace_max_diff", "trace max|d|"), ("bitwise", "fields bitwise"), ("trace_bitwise", "traces bitwise"),
               ("agree", f"agree (< {EQUIVALENCE_TOLERANCE:g})")],
          "1. EQUIVALENCE: production against the diagnostic what-if V of the same case", digits=3)

    # 2. critical node
    old060, old065 = np.load(BEFORE / "traces" / "M060_cold.npz"), np.load(BEFORE / "traces" / "M065_cold.npz")
    new060, new065 = (np.load(OUT / "traces" / f"{tag_of(m)}.npz") for m in TRACED_MACHS)
    ci, cj, sonic_pass = critical_node(old065)
    f = int(old065["f"])
    say(f"2. CRITICAL NODE of the diagnosis: (i, j) = ({ci}, {cj}) = (f{ci - f:+d}, {cj}), the peak of pass {sonic_pass} "
        f"of the old M 0.65 run (its first pass with M >= 1)")
    node_rows = []
    for label, trace, row in (("M 0.60 before, converged", old060, -1), ("M 0.60 after, converged", new060, -1),
                              (f"M 0.65 before, pass {sonic_pass} (sonic stop)", old065, pass_row(old065, sonic_pass)),
                              (f"M 0.65 after, pass {sonic_pass}", new065, pass_row(new065, sonic_pass)),
                              ("M 0.65 after, converged", new065, -1)):
        node_rows.append(dict(case=label, **node_values(trace, row, ci, cj)))
    write_csv("critical_node_before_after.csv", node_rows)
    table(node_rows, [("case", "case"), ("psi_m", "psi_m"), ("psi_eta", "psi_eta"), ("W_m", "W_m m/s"),
                      ("W_theta", "W_theta m/s"), ("W", "W m/s"), ("mach", "M"), ("rho_res", "rho kg/m3"), ("a", "a m/s")], "")

    # 3. first blade station
    curves, prof_rows, stat_rows = [], [], []
    for mach, old, new in ((0.60, old060, new060), (0.65, old065, new065)):
        eta = np.asarray(old["eta"])
        k_old = pass_row(old, sonic_pass) if mach == 0.65 else -1
        cases = [(f"before ({'pass %d, sonic stop' % sonic_pass if mach == 0.65 else 'converged'})", old, k_old, BEFORE_INK, "--")]
        if mach == 0.65:
            cases.append((f"after, pass {sonic_pass}", new, pass_row(new, sonic_pass), AFTER_EARLY_INK, ":"))
        cases.append(("after, converged", new, -1, AFTER_INK, "-"))
        for label, trace, row, ink, ls in cases:
            profile = trace["mach"][row, int(trace["f"]) - int(trace["window_first"])]
            curves.append(dict(mach=mach, eta=eta, mach_profile=profile, ink=ink, ls=ls, label=label))
            for j, (e, m) in enumerate(zip(eta, profile)):
                prof_rows.append(dict(inlet_mach=mach, case=label, j=j, eta=float(e), mach=float(m)))
            for s in station_stats(trace, row):
                stat_rows.append(dict(inlet_mach=mach, case=label, **s))
    profile_figure(curves, OUT / "first_blade_station_mach_profiles.png")
    write_csv("first_blade_station_profiles.csv", prof_rows)
    write_csv("transition_station_stats.csv", stat_rows)
    shown = [dict(r) for r in prof_rows if r["j"] in (0, 1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 40)]
    pivot = {}
    for r in shown:
        pivot.setdefault((r["j"], r["eta"]), {})[f"M{r['inlet_mach']:.2f} {r['case']}"] = r["mach"]
    names = list(dict.fromkeys(k for v in pivot.values() for k in v))
    table([dict(j=j, eta=e, **v) for (j, e), v in pivot.items()], [("j", "j"), ("eta", "eta")] + [(n, n) for n in names],
          "3. M(eta) AT THE FIRST BLADE STATION f (selected j; all in first_blade_station_profiles.csv)")
    table(stat_rows, [("inlet_mach", "M1"), ("case", "case"), ("station", "station"), ("max_M", "max M"),
                      ("mean_M", "mean M"), ("eta_of_max", "eta of max"), ("j_of_max", "j of max")],
          "   f - 1 / f / f + 1")

    # 4. surface
    surf = []
    for label, path in (("M 0.60 before (converged)", BEFORE / "fields" / "M060_cold.npz"),
                        ("M 0.60 after (converged)", OUT / "fields" / f"{tag_of(0.60)}.npz"),
                        ("M 0.65 before (partial: sonic stop, pass 6)", BEFORE / "fields" / "M065_cold_production_stop.npz"),
                        ("M 0.65 after (converged)", OUT / "fields" / f"{tag_of(0.65)}.npz")):
        surf.append(dict(case=label, **surface(path)))
    write_csv("surface_summary.csv", [{k: (json.dumps(v) if isinstance(v, list) else v) for k, v in r.items()} for r in surf])
    table(surf, [("case", "case"), ("ss_peak", "SS peak M"), ("ss_peak_x_c", "at x/c"), ("ps_peak", "PS peak M"),
                 ("ps_peak_x_c", "at x/c"), ("te_ss", "TE SS M"), ("te_ps", "TE PS M"), ("beta_out_deg", "beta_out deg")],
          "4. BLADE SURFACE (the wall nodes of the blade stations f .. l)")
    for r in surf:
        say(f"   {r['case']}: first five PS points (x/c, M) " + ", ".join(f"({x:.4f}, {m:.4f})" for x, m in r["first_ps"]))
        say(f"   {'':{len(r['case'])}}  first five SS points (x/c, M) " + ", ".join(f"({x:.4f}, {m:.4f})" for x, m in r["first_ss"]))
    say()
    surface_figure([("M 0.60 before", BEFORE / "fields" / "M060_cold.npz", BEFORE_INK, "--"),
                    ("M 0.60 after", OUT / "fields" / f"{tag_of(0.60)}.npz", AFTER_EARLY_INK, "-"),
                    ("M 0.65 after", OUT / "fields" / f"{tag_of(0.65)}.npz", AFTER_INK, "-")],
                   OUT / "surface_mach.png")

    # 5. grid study and Mach sweep
    columns = [("label", "case"), ("status", "status"), ("passes", "passes"), ("first_sonic_pass", "1st sonic"),
               ("final_Mmax", "peak M"), ("final_Mmax_rel", "at (i, j)"), ("ss_wall_peak", "SS wall peak"),
               ("ps_wall_peak", "PS wall peak"), ("beta_out_deg", "beta_out deg"), ("kutta_err", "eps_K"),
               ("wall_s", "wall s")]
    grid_rows = [rows[t] for t in (tag_of(GRID_MACH, g) if g != (61, 41) else tag_of(GRID_MACH) for g in GRIDS) if t in rows]
    write_csv("grid_study.csv", grid_rows)
    table(grid_rows, columns, f"5a. GRID STUDY at M1 {GRID_MACH:.2f}, stop on")
    sweep = [rows[tag_of(m)] for m in SWEEP_MACHS if tag_of(m) in rows]
    write_csv("mach_sweep.csv", sweep)
    table(sweep, columns, "5b. MACH SWEEP, 61 x 41, stop on")
    stopped = [r for r in sweep if r.get("status") == "sonic stop (production)"]
    if stopped:
        r = stopped[0]
        say(f"   first case that stops: {r['label']} at pass {r['first_sonic_pass']}, sonic node {r['final_Mmax_node']} = "
            f"{r['final_Mmax_rel']} (j = 0 / {r.get('n_eta', 41) - 1} are the walls); message: {r['failure']}")
    else:
        say("   no case of the sweep reached M = 1")
    if "P_M0.700_stop_off" in rows:
        r = rows["P_M0.700_stop_off"]
        say(f"   M 0.70 with the stop off: {r['status']}, {r['passes']} passes, peak {fmt(r.get('final_Mmax'))} at "
            f"{r.get('final_Mmax_rel')}")
    say()
    (OUT / "report.txt").write_text("\n".join(LINES) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
