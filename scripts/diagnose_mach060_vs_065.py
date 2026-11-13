"""
diagnose_mach060_vs_065.py
==========================
Diagnosis of one question: why does the NACA 65-(12A10)10 case converge cleanly at M1 = 0.60
and reach a sonic interior node and diverge at M1 = 0.65? Controlled experiments on ONE frozen
baseline (below), every case built by potential_flow_solver.controller and solved by the
production solver. Nothing here tunes the solver or reproduces a published number.

    python scripts/diagnose_mach060_vs_065.py

Switch the experiment groups on and off below. Results go to OUTPUT_DIR: CSV tables, PNG
figures, report.txt (the console text), and the saved fields / traces of the runs (npz).

What each group answers
    MATCHED_BASELINE        M 0.60 against M 0.65, only the Mach number changed; traces of both
    INITIAL_DENSITY_STUDY   A: M 0.65 from three start densities (total, inlet static, the
                            converged M 0.60 field); E: one frozen-density pass with each
    WARM_START_STUDY        B: start density x start exit angle (2 x 2)
    MACH_CONTINUATION       C: cold starts and warm continuation 0.60 -> 0.65, the stability
                            boundary of each, and the contraction of the density map alone
                            (fixed-angle solves at each converged angle)
    FIXED_EXIT_STUDY        D: the density loop at fixed discharge angles (no Kutta)
    LE_FORENSICS            the critical node pass by pass, cross-passage profiles at the first
                            blade station, the f-1 / f / f+1 transition audit (production
                            against same-side derivatives), the first wall nodes
    RELAXATION_STUDY        lambda_rho = 0.1 .. 1.0 at M 0.65
    GRID_STUDY              a coarser and a finer admissible grid
    GEOMETRY_STUDY          the other thickness variant (emery_base, the plain 65-010 TE)
    WHAT_IF_STUDY           DIAGNOSTIC VARIANTS, NOT PRODUCTION: the same solver with the
                            m-derivatives at the transition stations taken on their own side
                            (one-sided, never across the edge): V in the velocity recovery, C
                            in d/dm of the coefficient fields behind D and E, at the four
                            stations f - 1, f, l, l + 1; Vle and Cle at the leading-edge pair
                            f - 1, f only. Patched inside the worker process only; no file of
                            the package is changed. It tests whether the transition
                            derivatives CAUSE the sonic node; V is also run on the other
                            grids and at higher inlet Mach numbers (where the wall itself
                            approaches M = 1).

SINCE THE 2026-11-13 FIX of potential_flow_solver/flow.py (psi_m by m region, metrics.py's
regions: the same one-sided stencils at f - 1, f, l, l + 1) the variants V and Vle ARE
production: patched on top of it they change nothing (tests/test_diagnose_script.py), and the
acceptance runs of scripts/verify_le_velocity_fix.py reproduce the saved W_V_* runs bit for bit.
The "production" rows and traces on file (M060_cold, M065_cold, ...) were made before the fix,
with psi_m central across the transitions: re-running a group now gives the fixed rule. The
same-side columns of the LE_FORENSICS audit now equal production for psi_m. C and Cle stay
diagnostic: d/dm of the coefficient fields behind D and E is still central across the edges
(a parked decision, README).

Every case runs in a pooled worker process. A what-if variant is patched in for its own case
and taken out again when the case ends, and every case first checks that the production
functions are in place (a leaked patch would silently change later cases).

The sonic stop is OFF in every diagnostic run (IterationSettings.stop_at_sonic False), so a run
that crosses M = 1 can still be classified; first_sonic_pass is the pass at which production
(stop on) would have raised TransonicFlowError. One production run (stop on) is included.
"""

# ============================================================
# EXPERIMENT GROUPS
# ============================================================

RUN_MATCHED_BASELINE = True
RUN_INITIAL_DENSITY_STUDY = True
RUN_WARM_START_STUDY = True
RUN_MACH_CONTINUATION = True
RUN_FIXED_EXIT_STUDY = True
RUN_LE_FORENSICS = True
RUN_RELAXATION_STUDY = False
RUN_GRID_STUDY = False
RUN_GEOMETRY_STUDY = False
RUN_WHAT_IF_STUDY = False

N_WORKERS = 8                           # independent cases run in parallel processes
OUTPUT_DIR = "outputs/mach060_vs_065_diagnosis"

# ============================================================
# THE FROZEN BASELINE (only the inlet Mach number changes between the two cases)
# ============================================================
# The settings of the user's runs of 2026-11-13 (outputs/naca65_12a10_10_M065: M 0.60 converged
# in 214 passes at beta_out 22.787 deg; M 0.65 stopped at pass 5): 61 x 41, both relaxations 0.7.

MACH_LOW, MACH_HIGH = 0.60, 0.65

BASELINE = dict(
    gamma=1.4, gas_constant=287.05, inlet_total_pressure=101325.0, inlet_total_temperature=288.15,
    omega=0.0, stream_surface_radius=None, inlet_absolute_tangential_velocity=None,
    blade_family="NACA65", design_lift_coefficient=1.2, thickness_ratio=0.10, meanline_type="A10",
    profile_variant="kovach_sandercock", surface_continuity="C2", n_camber_stations=101,
    chord=0.1, solidity=1.0, stagger_deg=33.0,
    inlet_flow_angle_deg=45.0, mass_flow_per_depth_override=None,
    grid_level=None, n_passage=61, n_eta=41, upstream_extension_chords=1.0, downstream_extension_chords=1.0,
    density_tolerance=1e-8, velocity_tolerance=None, max_density_iterations=600, density_relaxation=0.7,
    kutta_tolerance=1e-8, max_kutta_iterations=600, kutta_relaxation=0.7,
    measure_runtime=True,
)

CONTINUATION_MACHS = (0.61, 0.62, 0.63, 0.64, 0.65)     # warm continuation from MACH_LOW
REFINEMENT_STEP = 0.0025                                # Mach step around a stability boundary
RELAXATIONS = (0.1, 0.3, 0.5, 0.7, 1.0)                 # lambda_rho of the relaxation study
RELAXATION_PASSES = 2000                                # its pass ceiling (lambda 0.1 is slow)
GRIDS = ((43, 29), (61, 41), (87, 59))                  # (n_passage, n_eta): coarser, baseline, finer
FIXED_ANGLES_DEG = (19.7, 21.8, 24.4)                   # beyond beta*(M 0.60): two angles of the coupled M 0.65
                                                        # trajectory and Aungier's published 24.4 deg (a
                                                        # diagnostic fixed angle only, never a target)
CONTRACTION_MACHS = (0.30, 0.50)                        # further Mach numbers for the density-map contraction
LATE_PASSES = 20                                        # passes of the late contraction ratios
WHAT_IF_VARIANTS = (("V", MACH_LOW), ("V", MACH_HIGH), ("Vle", MACH_HIGH), ("Cle", MACH_HIGH), ("V+Cle", MACH_LOW),
                    ("V+Cle", MACH_HIGH), ("C", MACH_HIGH))     # (variant, inlet Mach) on the baseline grid
WHAT_IF_MACHS = (0.68, 0.70, 0.72)                      # further inlet Mach numbers for variant V

# ============================================================
# (end of the editable part)
# ============================================================

import csv                                                                       # noqa: E402
import json                                                                      # noqa: E402
import math                                                                      # noqa: E402
import sys                                                                       # noqa: E402
import time                                                                      # noqa: E402
import warnings                                                                  # noqa: E402
from concurrent.futures import ProcessPoolExecutor                               # noqa: E402
from pathlib import Path                                                         # noqa: E402

import numpy as np                                                               # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / OUTPUT_DIR

TRACE_NAMES = ("mach", "W", "W_m", "W_theta", "a", "T", "p", "rho_res", "rho_tgt", "rho_next", "psi", "psi_m",
               "psi_eta", "tan_beta", "A", "B", "C", "D", "E")
WINDOW_BEFORE, WINDOW_AFTER = 3, 5          # stations f - 3 .. f + 5 are traced


# --- one case ---------------------------------------------------------------------------------------------------

def spec(tag, mach, label="", **options) -> dict:
    """The description of one run. options: n_passage, n_eta, mode ("coupled_kutta" /
    "fixed_exit"), exit_deg (None: the blade TE angle), density ("total", "static" or the path of
    a saved field), lam_rho, lam_k, passes, stop (the sonic stop; default off), trace, what_if
    (None, "V", "C", "VC"), profile_variant."""
    base = dict(tag=tag, mach=mach, label=label or tag, n_passage=None, n_eta=None, mode="coupled_kutta", exit_deg=None,
                density="total", lam_rho=None, lam_k=None, passes=None, stop=False, trace=False, what_if=None,
                profile_variant=None)
    unknown = set(options) - set(base)
    assert not unknown, unknown
    base.update(options)
    return base


def settings_of(job: dict):
    from potential_flow_solver import controller as ct

    values = dict(BASELINE)
    values.update(case_name=job["tag"], inlet_mach=job["mach"], solver_mode=job["mode"], stop_at_sonic=bool(job["stop"]),
                  fixed_exit_angle_deg=job["exit_deg"] if job["mode"] == ct.FIXED_EXIT else None,
                  initial_exit_angle_mode=ct.BLADE_TE_ANGLE if job["exit_deg"] is None else ct.MANUAL,
                  initial_exit_angle_deg=job["exit_deg"])
    for key, name in (("n_passage", "n_passage"), ("n_eta", "n_eta"), ("lam_rho", "density_relaxation"),
                      ("lam_k", "kutta_relaxation"), ("profile_variant", "profile_variant")):
        if job[key] is not None:
            values[name] = job[key]
    if job["passes"] is not None:
        values["max_density_iterations"] = values["max_kutta_iterations"] = job["passes"]
    return ct.CaseSettings(**values)


class LETrace:
    """An observer with inspect(fields): every pass, the fields of the stations f - 3 .. f + 5
    (all eta) and the global peak-Mach node. It reads; it changes nothing."""

    def __init__(self, grid) -> None:
        self.f = grid.first_blade_i
        self.window = slice(self.f - WINDOW_BEFORE, self.f + WINDOW_AFTER + 1)
        self.rows = {name: [] for name in TRACE_NAMES}
        self.peak, self.passes = [], []

    def start(self, info) -> None:
        pass

    def update(self, record) -> None:
        pass

    def finish(self, solution, error) -> None:
        pass

    def inspect(self, fields) -> None:
        arrays = field_arrays(fields)
        for name in TRACE_NAMES:
            self.rows[name].append(np.array(arrays[name][self.window]))
        mach = arrays["mach"]
        i, j = np.unravel_index(int(np.argmax(mach)), mach.shape)
        self.peak.append([i, j] + [float(arrays[name][i, j]) for name in TRACE_NAMES])
        r = fields.record
        self.passes.append([r.index, int(r.final), r.density_change, r.velocity_change, r.psi_change, r.kutta_normalized,
                            math.degrees(r.exit_angle), r.mach_max, r.linear_residual])

    def save(self, path, case, fields_static) -> None:
        g = case.grid
        b = g.boundaries
        np.savez_compressed(
            path, **{name: np.array(values) for name, values in self.rows.items()}, peak=np.array(self.peak),
            passes=np.array(self.passes), f=self.f, l=g.last_blade_i, window_first=self.window.start,
            S=np.asarray(fields_static["S"])[self.window], nodes=np.asarray(g.nodes)[self.window],
            m=np.asarray(g.m)[self.window], eta=np.asarray(g.eta), dm=g.dm, c_ref=case.settings.chord,
            mdot=fields_static["mdot"], le=np.asarray(b.lower_blade_boundary[0]), te=np.asarray(b.lower_blade_boundary[-1]),
            shift=np.asarray(b.periodic_translation), n_m=len(g.m))


def field_arrays(fields) -> dict:
    """The traced node fields of a PassFields, each (n_m, n_eta)."""
    flow, state, co = fields.flow, fields.static_state, fields.coefficients
    return dict(mach=fields.mach, W=flow.W, W_m=flow.W_m, W_theta=flow.W_theta, a=state.sound_speed, T=state.temperature,
                p=state.pressure, rho_res=fields.density_resident, rho_tgt=fields.density_target,
                rho_next=fields.density_next, psi=fields.psi, psi_m=flow.psi_m, psi_eta=flow.psi_eta,
                tan_beta=fields.metrics.tan_beta, A=co.A, B=co.B, C=co.C, D=co.D, E=co.E)


def apply_what_if(kind, first: int, last: int) -> None:
    """DIAGNOSTIC VARIANTS (this process only): same-side three-point m-differences instead of
    central differences across the leading / trailing edge. V and Vle reproduce production
    since the 2026-11-13 fix of flow.py (a no-op on top of it); C and Cle still change d/dm of
    the coefficient fields. kind: tokens joined by "+": "V"
    (velocity recovery) and "C" (d/dm of the coefficient fields behind D and E) at the four
    transition stations f - 1 (backward), f (forward), l (backward), l + 1 (forward); "Vle" and
    "Cle" the same at the leading-edge pair f - 1, f only. Returns the function that puts the
    production functions back: run_job calls it when the case is done, because a pooled worker
    process goes on to other cases."""
    from potential_flow_solver import coefficients, flow

    production = (flow._derivatives, coefficients._d_dm)

    def restore() -> None:
        flow._derivatives, coefficients._d_dm = production

    if not kind:
        return restore

    tokens = set(kind.split("+"))
    assert tokens <= {"V", "C", "Vle", "Cle"}, tokens

    def same_side(field, derivative, dm, interior_only, trailing_edge):
        out = np.array(derivative)
        rows = slice(1, -1) if interior_only else slice(None)
        out[first - 1] = (3.0 * field[first - 1] - 4.0 * field[first - 2] + field[first - 3]) / (2.0 * dm)
        out[first, rows] = (-3.0 * field[first, rows] + 4.0 * field[first + 1, rows] - field[first + 2, rows]) / (2.0 * dm)
        if trailing_edge:
            out[last, rows] = (3.0 * field[last, rows] - 4.0 * field[last - 1, rows] + field[last - 2, rows]) / (2.0 * dm)
            out[last + 1] = (-3.0 * field[last + 1] + 4.0 * field[last + 2] - field[last + 3]) / (2.0 * dm)
        return out

    if tokens & {"V", "Vle"}:
        original_derivatives = flow._derivatives

        def derivatives(psi, dm, deta, bcs):
            psi_m, psi_eta = original_derivatives(psi, dm, deta, bcs)       # psi_m = 0 on the walls stays
            return same_side(psi, psi_m, dm, interior_only=True, trailing_edge="V" in tokens), psi_eta

        flow._derivatives = derivatives
    if tokens & {"C", "Cle"}:
        original_d_dm = coefficients._d_dm

        def d_dm(field, dm):
            return same_side(field, original_d_dm(field, dm), dm, interior_only=False, trailing_edge="C" in tokens)

        coefficients._d_dm = d_dm
    return restore


def check_production() -> None:
    """Refuse to run a case if a diagnostic variant is still patched into this process."""
    from potential_flow_solver import coefficients, flow

    for function, module, name in ((flow._derivatives, "potential_flow_solver.flow", "_derivatives"),
                                   (coefficients._d_dm, "potential_flow_solver.coefficients", "_d_dm")):
        if function.__module__ != module or function.__name__ != name:
            raise RuntimeError(f"{module}.{name} is not the production function in this process: a what-if patch leaked")


def run_job(job: dict) -> dict:
    """Build and solve one case in this process; save its fields (and trace); return its row."""
    warnings.simplefilter("ignore")
    from potential_flow_solver import controller as ct
    from potential_flow_solver import postprocess as pp
    from potential_flow_solver import solver as sv

    check_production()
    row = dict(tag=job["tag"], label=job["label"], mach=job["mach"], mode=job["mode"], what_if=job["what_if"] or "",
               init_density=job["density"] if job["density"] in ("total", "static") else "field of " + Path(job["density"]).stem)
    started = time.perf_counter()
    try:
        case = ct.build_case(settings_of(job))
    except Exception as error:                  # a case that cannot be built is a result, not a crash
        row.update(status="case not built", failure=f"{type(error).__name__}: {error}"[:300])
        return row
    grid, s = case.grid, case.settings
    f, last, J = grid.first_blade_i, grid.last_blade_i, len(grid.eta) - 1
    row.update(n_passage=s.n_passage, n_eta=s.n_eta, lam_rho=s.density_relaxation, lam_k=s.kutta_relaxation,
               init_beta_deg=case.exit_angle_deg, mdot=case.mass_flow_per_depth, first_blade_i=f)
    shape = (len(grid.m), len(grid.eta))
    if job["density"] == "total":
        rho0 = None
    elif job["density"] == "static":
        rho0 = np.full(shape, case.inlet.static_density)
    else:
        rho0 = np.load(job["density"])["density"]
        if rho0.shape != shape:
            row.update(status="case not built", failure=f"the start field is {rho0.shape}, the grid {shape}")
            return row
    trace = LETrace(grid) if job["trace"] else None
    restore = apply_what_if(job["what_if"], f, last)
    try:
        run = ct.run_case(case, initial_density=rho0, observers=[trace] if trace else ())
    finally:
        restore()                               # this worker process runs other cases afterwards
    sol, error = run.solution, run.error
    passes = sol.passes if sol is not None else ()
    peaks = [p.mach_max for p in passes]
    first_sonic = next((p.index for p in passes if p.mach_max >= 1.0), None)
    one_pass = job["passes"] == 1
    if one_pass and sol is not None:
        status = "one frozen-density pass"
    elif run.converged:
        status = ("converged subsonic" if max(peaks) < 1.0 else
                  "converged subsonic after a sonic transient" if peaks[-1] < 1.0 else "converged WITH M >= 1 nodes")
    elif isinstance(error, sv.TransonicFlowError):
        status = "sonic stop (production)"
    elif type(error).__name__ == "ThermodynamicStateError":
        status = "diverged: T <= 0"
    elif isinstance(error, sv.DensityIterationError):
        changes = [p.density_change for p in passes]
        status = f"not converged in {len(passes)} passes ({'growing' if changes[-1] > 2.0 * min(changes) else 'decaying'})"
    else:
        status = f"failed: {type(error).__name__}"
    row.update(status=status, passes=len(passes), first_sonic_pass=first_sonic,
               failure="" if error is None else f"{type(error).__name__}: {error}"[:200],
               wall_s=round(time.perf_counter() - started, 1))
    if sol is None:
        return row
    final = passes[-1]
    mach = np.asarray(sol.mach)
    i, j = np.unravel_index(int(np.argmax(mach)), mach.shape)
    surfaces = pp.blade_surfaces(sol)
    changes = np.array([p.density_change for p in passes if not p.final])
    velocity = np.array([p.velocity_change for p in passes if not p.final])
    row.update(history_Mmax=max(peaks), final_Mmax=float(mach[i, j]), final_Mmax_node=f"({i}, {j})",
               final_Mmax_rel=f"(f{i - f:+d}, {j})", ss_wall_peak=float(np.max(surfaces.mach_ss)),
               ps_wall_peak=float(np.max(surfaces.mach_ps)), beta_out_deg=math.degrees(sol.exit_angle),
               density_err=final.density_change, velocity_err=final.velocity_change, kutta_err=final.kutta_normalized,
               min_density_err=float(np.min(changes)), min_density_err_pass=int(np.argmin(changes)),
               rho0_min=passes[0].density_min, rho0_max=passes[0].density_max,
               far_inlet_mach=float(np.mean(mach[0])) if run.converged else None,
               late_r_rho=late_ratio(changes) if run.converged else None,
               late_r_W=late_ratio(velocity) if run.converged else None)
    (OUT / "fields").mkdir(parents=True, exist_ok=True)
    state = sol.static_state
    np.savez_compressed(OUT / "fields" / f"{job['tag']}.npz", density=np.asarray(sol.density), mach=mach,
                        psi=np.asarray(sol.psi), W=np.asarray(sol.flow.W), pressure=np.asarray(state.pressure),
                        exit_angle=sol.exit_angle, f=f, l=last, x_c_ss=surfaces.x_c_ss, x_c_ps=surfaces.x_c_ps,
                        m_c=surfaces.m_c, mach_ss=surfaces.mach_ss, mach_ps=surfaces.mach_ps)
    if trace is not None:
        (OUT / "traces").mkdir(parents=True, exist_ok=True)
        trace.save(OUT / "traces" / f"{job['tag']}.npz", case, dict(S=np.asarray(sol.metrics.S) * s.chord,
                                                                   mdot=sol.mass_flow_per_depth))
    return row


def late_ratio(values: np.ndarray):
    """The geometric-mean ratio of successive changes over the last LATE_PASSES loop passes."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v) & (v > 0.0)]
    if len(v) < 5:
        return None
    tail = v[-min(LATE_PASSES + 1, len(v)):]
    return float((tail[-1] / tail[0]) ** (1.0 / (len(tail) - 1)))


def continuation_job(start_tag: str, start_mach: float, machs: tuple, prefix: str = "warm", **options) -> list:
    """Warm continuation in Mach from the saved converged field of start_tag: each step starts
    from the previous converged density and discharge angle; after the first failure the interval
    back to the last converged Mach is searched in REFINEMENT_STEP steps."""
    rows = []
    previous = (start_mach, OUT / "fields" / f"{start_tag}.npz")

    def step(mach):
        nonlocal previous
        field = np.load(previous[1])
        row = run_job(spec(f"{prefix}_M{mach:.4f}", mach, f"M {mach:.4f} warm continuation from M {previous[0]:.4f}",
                           density=str(previous[1]), exit_deg=math.degrees(float(field["exit_angle"])), **options))
        row["continued_from"] = previous[0]
        rows.append(row)
        if row["status"].startswith("converged"):
            previous = (mach, OUT / "fields" / f"{row['tag']}.npz")
            return True
        return False

    for mach in machs:
        if not step(mach):
            trial = previous[0] + REFINEMENT_STEP
            while trial < mach - 1e-9 and step(round(trial, 6)):
                trial += REFINEMENT_STEP
            break
    return rows


# --- tables and text ---------------------------------------------------------------------------------------------

LOG: list[str] = []


def say(text: str = "") -> None:
    print(text, flush=True)
    LOG.append(text)


def write_csv(name: str, rows: list) -> None:
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with (OUT / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value, digits=4) -> str:
    if value is None or value == "":
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}g}" if (abs(value) < 1e-3 or abs(value) >= 1e5) and value != 0 else f"{value:.{digits}f}"
    return str(value)


def table(rows: list, columns: list, title: str) -> None:
    say(f"\n--- {title} " + "-" * max(4, 100 - len(title)))
    if not rows:
        say("  (no rows)")
        return
    cells = [[fmt(row.get(key)) for key, _ in columns] for row in rows]
    widths = [max(len(head), *(len(c[k]) for c in cells)) for k, (_, head) in enumerate(columns)]
    say("  " + "  ".join(head.ljust(w) for (_, head), w in zip(columns, widths)))
    for c in cells:
        say("  " + "  ".join(v.ljust(w) for v, w in zip(c, widths)))


SUMMARY_COLUMNS = [("label", "case"), ("status", "status"), ("passes", "passes"), ("first_sonic_pass", "1st sonic"),
                   ("history_Mmax", "hist Mmax"), ("final_Mmax", "final Mmax"), ("final_Mmax_rel", "at node"),
                   ("ss_wall_peak", "SS wall"), ("beta_out_deg", "beta_out"), ("density_err", "rho err"),
                   ("velocity_err", "W err"), ("kutta_err", "Kutta err")]


# --- the studies ---------------------------------------------------------------------------------------------------

def study_inputs() -> None:
    """Sections 2, 3 and 20 of the brief: the configuration of both cases, the start density
    against the inlet static density, the mass flow by hand."""
    from potential_flow_solver import controller as ct
    from potential_flow_solver import eos

    say("=" * 100)
    say("MATCHED CONFIGURATION: the two cases differ in the inlet Mach number only")
    say("=" * 100)
    rows = []
    for mach in (MACH_LOW, MACH_HIGH):
        case = ct.build_case(settings_of(spec(f"M{mach:.2f}", mach)))
        say(ct.format_case_summary(case))
        say(ct.format_inlet_state(case))
        g, R, Pt, Tt = BASELINE["gamma"], BASELINE["gas_constant"], BASELINE["inlet_total_pressure"], BASELINE["inlet_total_temperature"]
        T1 = Tt / (1.0 + 0.5 * (g - 1.0) * mach ** 2)
        p1 = Pt * (T1 / Tt) ** (g / (g - 1.0))
        rho1, a1 = p1 / (R * T1), math.sqrt(g * R * T1)
        W1 = mach * a1
        pitch = BASELINE["chord"] / BASELINE["solidity"]
        mdot = rho1 * W1 * math.cos(math.radians(BASELINE["inlet_flow_angle_deg"])) * pitch
        gas = eos.CaloricallyPerfectGas(gamma=g, gas_constant=R)
        rho_start = float(eos.initial_density(gas, eos.blade_row_reference(eos.TotalState(Pt, Tt), gas)))
        rows.append(dict(mach=mach, rho_total=Pt / (R * Tt), rho_start_solver=rho_start, T1=T1, p1=p1, rho1_static=rho1, a1=a1,
                         W1=W1, W_m1=W1 * math.cos(math.radians(45.0)), mdot_by_hand=mdot,
                         mdot_controller=case.mass_flow_per_depth,
                         mdot_relative_difference=case.mass_flow_per_depth / mdot - 1.0,
                         start_excess_over_static=(rho_start - rho1) / rho1))
    say(ct.format_grid_summary(case))
    write_csv("initial_density_and_mass_flow.csv", rows)
    table(rows, [("mach", "M1"), ("rho_total", "Pt/(R Tt)"), ("rho_start_solver", "solver start rho"),
                 ("rho1_static", "rho1 static"), ("start_excess_over_static", "(rho_t - rho1)/rho1"),
                 ("mdot_by_hand", "mdot' by hand"), ("mdot_controller", "mdot' controller"),
                 ("mdot_relative_difference", "rel. diff")], "START DENSITY AND MASS FLOW")


def analyse_traces(critical=None) -> tuple:
    """Critical-node history, first-blade-station profiles, the transition audit and the first
    wall nodes from the two baseline traces. Returns (critical node, selected passes)."""
    from potential_flow_solver import eos

    gas = eos.CaloricallyPerfectGas(gamma=BASELINE["gamma"], gas_constant=BASELINE["gas_constant"])
    reference = eos.blade_row_reference(eos.TotalState(BASELINE["inlet_total_pressure"],
                                                       BASELINE["inlet_total_temperature"]), gas)
    traces = {}
    for tag in ("M060_cold", "M065_cold"):
        path = OUT / "traces" / f"{tag}.npz"
        if path.exists():
            traces[tag] = dict(np.load(path))
    if "M065_cold" not in traces:
        say("no M065_cold trace: run the matched baseline first")
        return None, {}
    high = traces["M065_cold"]
    peaks = high["passes"][:, 7]
    sonic = int(np.argmax(peaks >= 1.0)) if np.any(peaks >= 1.0) else int(np.argmax(peaks))
    if critical is None:
        critical = (int(high["peak"][sonic, 0]), int(high["peak"][sonic, 1]))
    f, J = int(high["f"]), len(high["eta"]) - 1
    say(f"\ncritical node: (i, j) = {critical} = (f{critical[0] - f:+d}, {critical[1]}): the peak-Mach node of M 0.65 at its "
        f"first sonic pass ({sonic}); f = {f} is the first blade station, eta = {high['eta'][critical[1]]:.4f}")

    def node_class(t, i, j):
        blade = int(t["f"]) <= i <= int(t["l"])
        if j in (0, len(t["eta"]) - 1):
            return "blade wall" if blade else "periodic line"
        return "interior, blade station" if blade else ("interior, inlet station" if i < int(t["f"]) else "interior, exit station")

    def x_over_c(t, point):
        d = t["te"] - t["le"]
        return float((np.asarray(point) - t["le"]) @ d / (d @ d))

    def decomposition(t, values, S):
        mu = float(t["mdot"]) / values["rho_res"]
        first = values["tan_beta"] / S * values["psi_eta"]
        return dict(mu=mu, inv_S=1.0 / S, Wm_from_parts=mu * values["psi_eta"] / S, tanb_over_S_psi_eta=first,
                    minus_psi_m=-values["psi_m"], G_theta=first - values["psi_m"],
                    Wtheta_from_parts=mu * (first - values["psi_m"]))

    # critical-node history: the moving peak node and the fixed critical node, both cases
    history = []
    for tag, t in traces.items():
        w0 = int(t["window_first"])
        previous = {}
        for n in range(len(t["passes"])):
            i, j = int(t["peak"][n, 0]), int(t["peak"][n, 1])
            entries = [("peak", i, j, {name: float(t["peak"][n, 2 + k]) for k, name in enumerate(TRACE_NAMES)}, None)]
            if w0 <= critical[0] < w0 + t["mach"].shape[1]:
                k = critical[0] - w0
                entries.append(("fixed", critical[0], critical[1],
                                {name: float(t[name][n, k, critical[1]]) for name in TRACE_NAMES}, k))
            for kind, i, j, values, k in entries:
                row = dict(case=tag, kind=kind, iteration=int(t["passes"][n, 0]), i=i, j=j, i_minus_f=i - int(t["f"]),
                           eta=float(t["eta"][j]), node_class=node_class(t, i, j))
                if k is not None:
                    point = t["nodes"][k, j] - (t["shift"] if j == len(t["eta"]) - 1 else 0.0)
                    row.update(m=float(t["m"][k]), x_c=x_over_c(t, point), S=float(t["S"][k]))
                    row.update(values)
                    row.update(decomposition(t, values, float(t["S"][k])))
                    last = previous.get(kind)
                    if last is not None:
                        row.update(dlnM=math.log(values["mach"] / last["mach"]), dlnW=math.log(values["W"] / last["W"]),
                                   dlna=math.log(values["a"] / last["a"]), dln_mu=math.log(last["rho_res"] / values["rho_res"]),
                                   dln_psi_eta=math.log(abs(values["psi_eta"] / last["psi_eta"])))
                    previous[kind] = values
                else:
                    row.update(values)
                history.append(row)
    write_csv("critical_node_history.csv", history)

    # the passes shown in the profiles
    change = high["passes"][:, 2]
    after = np.arange(len(change)) > sonic
    plateau = int(np.argmin(np.where(after, change, np.inf))) if np.any(after) else sonic
    runaway = next((n for n in range(plateau, len(peaks)) if peaks[n] > peaks[plateau] + 0.15), len(peaks) - 1)
    selected = {"M065_cold": {"pass 0": 0, "early": 2, "before first sonic": max(sonic - 1, 0), "first sonic": sonic,
                              "plateau (least density change)": plateau, "before runaway": max(runaway - 3, plateau),
                              "last": len(peaks) - 1}}
    if "M060_cold" in traces:
        n = len(traces["M060_cold"]["passes"])
        selected["M060_cold"] = {"pass 0": 0, "mid-convergence": n // 2, "final": n - 1}
    profiles = []
    for tag, chosen in selected.items():
        t = traces[tag]
        w0 = int(t["window_first"])
        for name, n in chosen.items():
            for station in (-1, 0, 1, 2):
                k = int(t["f"]) + station - w0
                for j in range(len(t["eta"])):
                    row = dict(case=tag, selected=name, iteration=int(t["passes"][n, 0]), station=f"f{station:+d}",
                               i=int(t["f"]) + station, j=j, eta=float(t["eta"][j]), S=float(t["S"][k]))
                    row.update({key: float(t[key][n, k, j]) for key in TRACE_NAMES})
                    profiles.append(row)
    write_csv("first_blade_station_profiles.csv", profiles)

    # the transition audit: production against same-side derivatives at f - 1, f, f + 1
    audit = []
    chosen = {"M060_cold": [("final", selected["M060_cold"]["final"])]} if "M060_cold" in traces else {}
    chosen["M065_cold"] = [("first sonic", sonic), ("plateau", plateau)]
    for tag, list_of in chosen.items():
        t = traces[tag]
        w0, c_ref = int(t["window_first"]), float(t["c_ref"])
        dm = float(t["dm"]) * c_ref
        for name, n in list_of:
            k_f = int(t["f"]) - w0
            y = t["nodes"][:, :, 1] * c_ref

            def one_sided(field, k, forward):
                return ((-3.0 * field[k] + 4.0 * field[k + 1] - field[k + 2]) if forward
                        else (3.0 * field[k] - 4.0 * field[k - 1] + field[k - 2])) / (2.0 * dm)

            for station in (-1, 0, 1):
                k = k_f + station
                forward = station >= 0
                fields = {key: t[key][n] for key in TRACE_NAMES}
                central = {key: (fields[key][k + 1] - fields[key][k - 1]) / (2.0 * dm) for key in ("B", "C", "psi")}
                side = {key: one_sided(fields[key], k, forward) if station != 1 else central[key] for key in ("B", "C", "psi")}
                secant = (y[k + 1] - y[k - 1]) / (2.0 * dm)
                secant_side = one_sided(y, k, forward) if station != 1 else secant
                mu = float(t["mdot"]) / fields["rho_res"][k]
                S = float(t["S"][k])
                psi_m_side = np.array(side["psi"])
                if station >= 0:
                    psi_m_side[[0, -1]] = 0.0                       # the walls: psi = 0 / 1 along them
                W_theta_side = mu * (fields["tan_beta"][k] / S * fields["psi_eta"][k] - psi_m_side)
                W_side = np.hypot(fields["W_m"][k], W_theta_side)
                mach_side = W_side / np.asarray(eos.static_state_from_velocity(W_side, gas, reference).sound_speed)
                B_eta = np.gradient(fields["B"][k], t["eta"])
                for j in range(len(t["eta"])):
                    audit.append(dict(
                        case=tag, selected=name, iteration=int(t["passes"][n, 0]), station=f"f{station:+d}", j=j,
                        eta=float(t["eta"][j]), B=fields["B"][k, j], C=fields["C"][k, j],
                        B_m_production=central["B"][j], B_m_same_side=side["B"][j], C_m_production=central["C"][j],
                        C_m_same_side=side["C"][j], B_eta=B_eta[j], D_production=fields["D"][k, j],
                        D_same_side=fields["D"][k, j] + central["B"][j] - side["B"][j], E_production=fields["E"][k, j],
                        E_same_side=fields["E"][k, j] - central["C"][j] + side["C"][j], A=fields["A"][k, j],
                        tan_beta_metric=fields["tan_beta"][k, j], eta_line_slope_central=secant[j],
                        eta_line_slope_same_side=secant_side[j], psi_m_production=fields["psi_m"][k, j],
                        psi_m_same_side=psi_m_side[j], psi_eta=fields["psi_eta"][k, j], W_m=fields["W_m"][k, j],
                        W_theta_production=fields["W_theta"][k, j], W_theta_same_side=W_theta_side[j],
                        mach_production=fields["mach"][k, j], mach_same_side=mach_side[j]))
    write_csv("transition_audit.csv", audit)

    # the first wall nodes of both surfaces
    walls = []
    for tag, n, name in ([("M060_cold", selected["M060_cold"]["final"], "final")] if "M060_cold" in traces else []) + \
            [("M065_cold", sonic, "first sonic"), ("M065_cold", plateau, "plateau")]:
        t = traces[tag]
        k_f = int(t["f"]) - int(t["window_first"])
        for surface, j in (("suction", 0), ("pressure", len(t["eta"]) - 1)):
            for step in range(WINDOW_AFTER + 1):
                k = k_f + step
                point = t["nodes"][k, j] - (t["shift"] if j else 0.0)
                walls.append(dict(case=tag, selected=name, iteration=int(t["passes"][n, 0]), surface=surface,
                                  station=f"f+{step}", x_c=x_over_c(t, point),
                                  m_c=float(t["m"][k] - t["le"][0]), mach=float(t["mach"][n, k, j]),
                                  W=float(t["W"][n, k, j]), rho=float(t["rho_res"][n, k, j]), p=float(t["p"][n, k, j]),
                                  psi_eta=float(t["psi_eta"][n, k, j]),
                                  wall_angle_deg=math.degrees(math.atan(float(t["tan_beta"][n, k, j])))))
    write_csv("wall_le_nodes.csv", walls)
    return critical, dict(traces=traces, selected=selected, sonic=sonic, plateau=plateau, audit=audit, walls=walls,
                          history=history, profiles=profiles)


def report_forensics(critical, data: dict) -> None:
    """The console tables of the forensic CSVs (the full data are in the files)."""
    traces, selected, sonic, plateau = data["traces"], data["selected"], data["sonic"], data["plateau"]
    high = traces["M065_cold"]
    f = int(high["f"])
    fixed = [r for r in data["history"] if r["kind"] == "fixed"]
    for tag in traces:
        rows = [r for r in fixed if r["case"] == tag]
        pick = sorted(set(selected[tag].values()))
        table([rows[n] for n in pick if n < len(rows)],
              [("iteration", "pass"), ("mach", "M"), ("W", "W"), ("a", "a"), ("W_m", "W_m"), ("W_theta", "W_theta"),
               ("rho_res", "rho"), ("rho_tgt", "rho_tgt"), ("mu", "mdot'/rho"), ("psi_eta", "psi_eta"),
               ("tanb_over_S_psi_eta", "(tanb/S)psi_eta"), ("minus_psi_m", "-psi_m"), ("G_theta", "G_theta")],
              f"FIXED CRITICAL NODE {critical} = (f{critical[0] - f:+d}, {critical[1]}), {tag}")
    peak = [r for r in data["history"] if r["kind"] == "peak" and r["case"] == "M065_cold"]
    table([peak[n] for n in sorted(set(selected["M065_cold"].values()))],
          [("iteration", "pass"), ("i_minus_f", "i - f"), ("j", "j"), ("node_class", "class"), ("mach", "M"), ("W", "W"),
           ("a", "a"), ("rho_res", "rho")], "GLOBAL PEAK NODE, M065_cold")
    for tag, name in (("M060_cold", "final"), ("M065_cold", "first sonic")):
        if tag not in traces:
            continue
        for station in ("f-1", "f+0", "f+1"):
            rows = [r for r in data["profiles"] if r["case"] == tag and r["selected"] == name and r["station"] == station]
            rows = [r for r in rows if r["j"] <= 8 or r["j"] % 5 == 0 or r["j"] >= len(high["eta"]) - 3]
            table(rows, [("j", "j"), ("eta", "eta"), ("mach", "M"), ("W", "W"), ("W_m", "W_m"), ("W_theta", "W_theta"),
                         ("rho_res", "rho"), ("psi_eta", "psi_eta"), ("psi_m", "psi_m"), ("tan_beta", "tan_beta")],
                  f"PROFILE ACROSS THE PASSAGE, {tag} ({name}), station {station}")
    for tag, name in (("M060_cold", "final"), ("M065_cold", "first sonic")):
        for station in ("f-1", "f+0", "f+1"):
            rows = [r for r in data["audit"] if r["case"] == tag and r["selected"] == name and r["station"] == station]
            rows = [r for r in rows if r["j"] <= 6 or r["j"] % 10 == 0 or r["j"] >= len(high["eta"]) - 3]
            table(rows, [("j", "j"), ("tan_beta_metric", "tanb metric"), ("eta_line_slope_central", "eta-line slope (central)"),
                         ("psi_m_production", "psi_m prod"), ("psi_m_same_side", "psi_m same side"),
                         ("mach_production", "M prod"), ("mach_same_side", "M same side"), ("D_production", "D prod"),
                         ("D_same_side", "D same side"), ("E_production", "E prod"), ("E_same_side", "E same side")],
                  f"TRANSITION AUDIT, {tag} ({name}), station {station}")
    table([r for r in data["walls"] if r["selected"] in ("final", "first sonic")],
          [("case", "case"), ("surface", "surface"), ("station", "station"), ("x_c", "x/c"), ("m_c", "m/c"), ("mach", "M"),
           ("W", "W"), ("rho", "rho"), ("p", "p"), ("psi_eta", "psi_eta"), ("wall_angle_deg", "wall angle")],
          "FIRST WALL NODES")


def compare_what_if() -> None:
    """Station f and the walls, production against the what-if V runs (final or last traced pass)."""
    rows = []
    loaded = {}
    for tag in ("M060_cold", "W_V_M0.60", "W_V_M0.65"):
        path = OUT / "traces" / f"{tag}.npz"
        if path.exists():
            loaded[tag] = np.load(path)
    if len(loaded) < 3:
        return
    eta = loaded["M060_cold"]["eta"]
    k = int(loaded["M060_cold"]["f"]) - int(loaded["M060_cold"]["window_first"])
    for j in range(len(eta)):
        rows.append(dict(j=j, eta=float(eta[j]), **{f"mach_f_{tag}": float(t["mach"][-1, k, j]) for tag, t in loaded.items()},
                         **{f"mach_f+1_{tag}": float(t["mach"][-1, k + 1, j]) for tag, t in loaded.items()}))
    write_csv("what_if_station_f_profiles.csv", rows)
    shown = [r for r in rows if r["j"] <= 8 or r["j"] % 5 == 0 or r["j"] >= len(eta) - 3]
    table(shown, [("j", "j"), ("eta", "eta"), ("mach_f_M060_cold", "M 0.60 production, f"), ("mach_f_W_V_M0.60", "M 0.60 what-if V, f"),
                  ("mach_f+1_W_V_M0.60", "M 0.60 what-if V, f+1"), ("mach_f_W_V_M0.65", "M 0.65 what-if V, f"),
                  ("mach_f+1_W_V_M0.65", "M 0.65 what-if V, f+1")],
          "MACH ACROSS THE PASSAGE AT THE FIRST BLADE STATION: production against what-if V (final pass)")
    walls = []
    for tag in ("M060_cold", "W_V_M0.60", "W_V_M0.65"):
        path = OUT / "fields" / f"{tag}.npz"
        if path.exists():
            field = np.load(path)
            for surface in ("ss", "ps"):
                x, mach = field[f"x_c_{surface}"], field[f"mach_{surface}"]
                peak = int(np.argmax(mach))
                walls.append(dict(case=tag, surface=surface, first_x_c=float(x[0]), first_mach=float(mach[0]),
                                  second_mach=float(mach[1]), peak_mach=float(mach[peak]), peak_x_c=float(x[peak]),
                                  mach_at_x_c_0p5=float(np.interp(0.5, x, mach)), last_mach=float(mach[-1])))
    write_csv("what_if_wall_summary.csv", walls)
    table(walls, [("case", "case"), ("surface", "surface"), ("first_x_c", "first x/c"), ("first_mach", "M first node"),
                  ("second_mach", "M second node"), ("peak_mach", "peak M"), ("peak_x_c", "at x/c"),
                  ("mach_at_x_c_0p5", "M at x/c 0.5"), ("last_mach", "M last node")], "WALL MACH NUMBERS: production against what-if V")


def geometry_audit() -> None:
    """Section 19: what the generated profile is, from the cascade and its metadata."""
    from potential_flow_solver import controller as ct

    case = ct.build_case(settings_of(spec("geometry", MACH_HIGH)))
    c, a = case.cascade, case.cascade.active
    say("\n--- GEOMETRY ACTUALLY USED " + "-" * 73)
    for label, value in (
            ("designation", case.blade_name),
            ("mean line", f"{BASELINE['meanline_type']}: NACA a = 1.0 closed form, ordinates x cl0 = {BASELINE['design_lift_coefficient']}"),
            ("thickness", f"NACA 65-010 column of Aungier Table 4-1 (Emery et al. 1958) x {BASELINE['thickness_ratio'] / 0.10:g}; "
                          f"variant {BASELINE['profile_variant']}"),
            ("max thickness / chord", c.metadata.get("max_thickness_over_c")),
            ("LE radius / chord", a.leading_edge_radius / a.chord), ("TE radius / chord", a.trailing_edge_radius / a.chord),
            ("stagger [deg]", math.degrees(c.stagger)), ("solidity", BASELINE["solidity"]), ("pitch / chord", a.pitch / a.chord),
            ("inlet angle [deg]", BASELINE["inlet_flow_angle_deg"]),
            ("angle of attack = inlet angle - stagger [deg]", case.angle_of_attack_deg),
            ("kappa_le, kappa_te [deg] (equivalent arc)", f"{case.kappa_le_deg:.4f}, {case.kappa_te_deg:.4f}"),
            ("surfaces", f"{BASELINE['surface_continuity']} (quintic curves; LE circle and fair of geometry.py)")):
        say(f"  {label:48s}: {value}")
    say(f"  metadata: {json.dumps({k: v for k, v in c.metadata.items() if isinstance(v, (int, float, str))}, default=str)[:900]}")


# --- figures -------------------------------------------------------------------------------------------------------

def make_figures(rows_by_tag: dict, data: dict | None, continuation: list, contraction: list) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    traces = data["traces"] if data else {}
    if traces:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for tag, t in traces.items():
            p = t["passes"]
            axes[0, 0].plot(p[:, 0], p[:, 7], label=f"{tag}: peak M")
            axes[0, 1].plot(p[:, 0], p[:, 6], label=tag)
            axes[1, 0].semilogy(p[:, 0], p[:, 2], label=f"{tag}: density change")
            axes[1, 0].semilogy(p[:, 0], p[:, 3], "--", label=f"{tag}: velocity change")
            axes[1, 1].semilogy(p[:, 0], p[:, 5], label=f"{tag}: eps_K")
        axes[0, 0].axhline(1.0, color="k", ls=":", lw=0.8)
        axes[0, 0].set_ylim(0.6, 1.6)
        for ax, title in zip(axes.flat, ("peak relative Mach number (clipped at 1.6)", "discharge angle [deg]",
                                         "density and velocity change per pass", "Kutta error")):
            ax.set_title(title)
            ax.set_xlabel("pass")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(OUT / "baseline_histories.png", dpi=120)
        plt.close(fig)

        fixed = [r for r in data["history"] if r["kind"] == "fixed"]
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        for tag in traces:
            rows = [r for r in fixed if r["case"] == tag]
            x = [r["iteration"] for r in rows]
            for ax, key in zip(axes.flat, ("mach", "W", "a", "rho_res", "psi_eta", "G_theta")):
                ax.plot(x, [r[key] for r in rows], label=tag)
                ax.set_title(f"critical node: {key}")
        for ax in axes.flat:
            ax.set_xlabel("pass")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
        axes[0, 0].set_ylim(0.5, 1.6)
        axes[0, 1].set_ylim(150, 500)
        axes[1, 1].set_ylim(bottom=0)
        fig.tight_layout()
        fig.savefig(OUT / "critical_node_history.png", dpi=120)
        plt.close(fig)

        fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
        for row, tag in enumerate(("M060_cold", "M065_cold")):
            if tag not in traces:
                continue
            for column, station in enumerate(("f-1", "f+0", "f+1")):
                ax = axes[row, column]
                for name, n in data["selected"][tag].items():
                    rows = [r for r in data["profiles"] if r["case"] == tag and r["selected"] == name and r["station"] == station]
                    ax.plot([r["eta"] for r in rows], [r["mach"] for r in rows], marker=".", ms=3, label=f"{name} (pass {n})")
                ax.axhline(1.0, color="k", ls=":", lw=0.8)
                ax.set_title(f"{tag}, station {station}: Mach across the passage")
                ax.set_xlabel("eta (0 = suction wall)")
                ax.grid(alpha=0.3)
                ax.set_ylim(0.2, 1.4)
                if column == 0:
                    ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(OUT / "first_blade_station_mach_profiles.png", dpi=120)
        plt.close(fig)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
        for tag, name, style in (("M060_cold", "final", "-"), ("M065_cold", "first sonic", "--")):
            rows = [r for r in data["audit"] if r["case"] == tag and r["selected"] == name and r["station"] == "f+0"]
            if not rows:
                continue
            eta = [r["eta"] for r in rows]
            axes[0].plot(eta, [r["mach_production"] for r in rows], style, color="C3", label=f"{tag}: production")
            axes[0].plot(eta, [r["mach_same_side"] for r in rows], style, color="C0", label=f"{tag}: same-side psi_m")
            axes[1].plot(eta, [r["tan_beta_metric"] for r in rows], style, color="C3", label=f"{tag}: tan(beta) of the metrics")
            axes[1].plot(eta, [r["eta_line_slope_central"] for r in rows], style, color="C0", label=f"{tag}: slope of the central difference")
            axes[2].plot(eta, [r["D_production"] for r in rows], style, color="C3", label=f"{tag}: D production")
            axes[2].plot(eta, [r["D_same_side"] for r in rows], style, color="C0", label=f"{tag}: D same side")
        for ax, title in zip(axes, ("station f: Mach, production against same-side psi_m", "station f: the two eta-line slopes",
                                    "station f: D, production against same-side B_m")):
            ax.set_title(title)
            ax.set_xlabel("eta")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(OUT / "transition_audit_station_f.png", dpi=120)
        plt.close(fig)

    if continuation:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
        for kind, marker in (("cold", "o"), ("warm", "s")):
            rows = sorted((r for r in continuation if r["kind"] == kind), key=lambda r: r["mach"])
            ok = [r for r in rows if r["status"].startswith("converged")]
            bad = [r for r in rows if not r["status"].startswith("converged")]
            axes[0].plot([r["mach"] for r in ok], [r["final_Mmax"] for r in ok], marker + "-", label=f"{kind} start: converged")
            axes[0].plot([r["mach"] for r in bad], [min(r.get("history_Mmax") or 1.5, 1.5) for r in bad], marker, mfc="none",
                         color="k", label=f"{kind} start: failed (history peak, clipped 1.5)")
        cold_ok = sorted((r for r in continuation if r["kind"] == "cold" and r["status"].startswith("converged")),
                         key=lambda r: r["mach"])
        axes[0].plot([r["mach"] for r in cold_ok], [r["ss_wall_peak"] for r in cold_ok], "^:", color="C2",
                     label="production: peak on the suction WALL")
        what_if = sorted((r for t, r in rows_by_tag.items() if t.startswith("W_V_M") and r["status"].startswith("converged")),
                         key=lambda r: r["mach"])
        axes[0].plot([r["mach"] for r in what_if], [r["final_Mmax"] for r in what_if], "D-", color="C3",
                     label="what-if V (diagnostic variant): field peak")
        axes[0].axhline(1.0, color="k", ls=":", lw=0.8)
        axes[0].set_xlabel("inlet Mach number")
        axes[0].set_ylabel("peak relative Mach number")
        axes[0].set_title("stability map (production field peak is at the first blade station)")
        rows = sorted(contraction, key=lambda r: r["mach"])
        rows = [r for r in rows if r.get("late_r_rho")]
        axes[1].plot([r["mach"] for r in rows], [r["late_r_rho"] for r in rows], "o-", label="r_rho (fixed angle)")
        axes[1].plot([r["mach"] for r in rows], [r["late_r_W"] for r in rows], "s--", label="r_W (fixed angle)")
        axes[1].axhline(1.0, color="k", ls=":", lw=0.8)
        axes[1].set_xlabel("inlet Mach number")
        axes[1].set_title("late contraction of the density map alone")
        for ax in axes:
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(OUT / "continuation_stability_map.png", dpi=120)
        plt.close(fig)


# --- main ----------------------------------------------------------------------------------------------------------

GROUPS = {"baseline": "RUN_MATCHED_BASELINE", "density": "RUN_INITIAL_DENSITY_STUDY", "warm": "RUN_WARM_START_STUDY",
          "continuation": "RUN_MACH_CONTINUATION", "fixed": "RUN_FIXED_EXIT_STUDY", "forensics": "RUN_LE_FORENSICS",
          "relaxation": "RUN_RELAXATION_STUDY", "grid": "RUN_GRID_STUDY", "geometry": "RUN_GEOMETRY_STUDY",
          "whatif": "RUN_WHAT_IF_STUDY"}


def safe_job(job: dict) -> dict:
    """run_job; a crash of one case becomes its row, the others go on."""
    try:
        return run_job(job)
    except Exception as error:
        return dict(tag=job["tag"], label=job["label"], mach=job["mach"], status="job crashed",
                    failure=f"{type(error).__name__}: {error}"[:300])


def main(argv=None) -> int:
    """The switches at the top choose the groups; --groups a,b,c (names of GROUPS, or all)
    overrides them for one invocation. Rows of earlier invocations are kept in rows.json, so a
    group can be run or re-run alone; the tables always show what is on file."""
    import argparse

    parser = argparse.ArgumentParser(description="M 0.60 against M 0.65: controlled diagnosis")
    parser.add_argument("--groups", default=None, help=f"comma list of {', '.join(GROUPS)} or all (default: the switches)")
    parser.add_argument("--workers", type=int, default=N_WORKERS, help="parallel processes (default: N_WORKERS)")
    args = parser.parse_args(argv)
    on = {name: bool(globals()[flag]) for name, flag in GROUPS.items()}
    if args.groups:
        chosen = set(GROUPS) if args.groups == "all" else set(args.groups.split(","))
        assert chosen <= set(GROUPS), chosen - set(GROUPS)
        on = {name: name in chosen for name in GROUPS}

    OUT.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    store = OUT / "rows.json"
    by_tag: dict = json.loads(store.read_text(encoding="utf-8")) if store.exists() else {}
    low_tag, high_tag = "M060_cold", "M065_cold"
    low_field = OUT / "fields" / f"{low_tag}.npz"

    def keep(rows):
        by_tag.update({row["tag"]: row for row in rows})
        store.write_text(json.dumps(by_tag, indent=1), encoding="utf-8")
        return rows

    def have(*tags):
        return [by_tag[t] for t in tags if t in by_tag]

    study_inputs()
    geometry_audit()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        def run(jobs):
            return keep(list(pool.map(safe_job, jobs))) if jobs else []

        if on["baseline"]:
            run([spec(low_tag, MACH_LOW, f"M {MACH_LOW:.2f} cold (total density, TE angle)", trace=True),
                 spec(high_tag, MACH_HIGH, f"M {MACH_HIGH:.2f} cold (total density, TE angle)", trace=True),
                 spec("M065_cold_production_stop", MACH_HIGH, f"M {MACH_HIGH:.2f} cold, production sonic stop ON", stop=True)])
        if not low_field.exists() or low_tag not in by_tag:
            say(f"\nthe converged M {MACH_LOW:.2f} field is missing: run the baseline group first")
            return 1
        beta_low = by_tag[low_tag]["beta_out_deg"]
        table(have(low_tag, high_tag, "M065_cold_production_stop"), SUMMARY_COLUMNS, "A. MATCHED BASELINE")
        for row in have(low_tag, high_tag):
            say(f"  {row['tag']}: start density of pass 0 uniform {row['rho0_min']:.9f} .. {row['rho0_max']:.9f} kg/m^3; "
                f"far-inlet Mach of the converged field {fmt(row.get('far_inlet_mach'), 6)}; mdot' {row['mdot']:.9f}")

        # the jobs of every group (the tags are fixed, so the tables can also be made from rows on file)
        tags = {name: [] for name in ("A", "E", "B", "D", "C_cold", "R", "G", "Y", "W")}
        jobs = []

        def add(group, enabled, job):
            tags[group].append(job["tag"])
            if enabled:
                jobs.append(job)

        warm = str(low_field)
        add("A", on["density"], spec("A2_static_density", MACH_HIGH, "A2: M 0.65, uniform inlet-static density, TE angle",
                                     density="static"))
        add("A", on["density"] or on["warm"], spec("A3_warm_density", MACH_HIGH,
                                                   "A3 = B2: M 0.65, converged M 0.60 density, TE angle", density=warm))
        for name, density in (("total", "total"), ("static", "static"), ("M060", warm)):
            add("E", on["density"], spec(f"E_one_pass_{name}", MACH_HIGH, f"E: M 0.65, ONE pass, resident density {name}, "
                                         f"TE angle", mode="fixed_exit", density=density, passes=1))
            add("E", on["density"], spec(f"E_one_pass_{name}_beta060", MACH_HIGH, f"E: M 0.65, ONE pass, resident density "
                                         f"{name}, beta*(0.60)", mode="fixed_exit", density=density, passes=1, exit_deg=beta_low))
        add("E", on["density"], spec("E_one_pass_M060_own", MACH_LOW, "E: M 0.60, ONE pass, resident density total, TE angle",
                                     mode="fixed_exit", passes=1))
        add("B", on["warm"], spec("B3_warm_beta", MACH_HIGH, "B3: M 0.65, total density, beta*(0.60)", exit_deg=beta_low))
        add("B", on["warm"], spec("B4_full_warm", MACH_HIGH, "B4: M 0.65, M 0.60 density and beta*(0.60)", density=warm,
                                  exit_deg=beta_low))
        add("D", on["fixed"], spec("D_fixed_beta060", MACH_HIGH, f"D: M 0.65 fixed exit {beta_low:.3f} deg = beta*(0.60), "
                                   f"total density", mode="fixed_exit", exit_deg=beta_low))
        add("D", on["fixed"], spec("D_fixed_beta060_warm", MACH_HIGH, f"D: M 0.65 fixed exit {beta_low:.3f} deg, M 0.60 density",
                                   mode="fixed_exit", exit_deg=beta_low, density=warm))
        for angle in FIXED_ANGLES_DEG:
            add("D", on["fixed"], spec(f"D_fixed_{angle:.1f}", MACH_HIGH, f"D: M 0.65 fixed exit {angle:.1f} deg, total density",
                                       mode="fixed_exit", exit_deg=angle))
        add("D", on["fixed"], spec("D_fixed_M060", MACH_LOW, f"D: M 0.60 fixed exit {beta_low:.3f} deg (its own beta*), "
                                   f"total density", mode="fixed_exit", exit_deg=beta_low))
        for mach in CONTINUATION_MACHS[:-1]:
            add("C_cold", on["continuation"], spec(f"cold_M{mach:.4f}", mach, f"M {mach:.4f} cold"))
        for lam in RELAXATIONS:
            add("R", on["relaxation"], spec(f"R_lambda_{lam:.1f}", MACH_HIGH, f"R: M 0.65 cold, lambda_rho {lam:.1f}",
                                            lam_rho=lam, passes=RELAXATION_PASSES))
            add("R", on["relaxation"], spec(f"R_lambda_{lam:.1f}_M060", MACH_LOW, f"R: M 0.60 cold, lambda_rho {lam:.1f}",
                                            lam_rho=lam, passes=RELAXATION_PASSES))
        other_grids = [g for g in GRIDS if g != (BASELINE["n_passage"], BASELINE["n_eta"])]
        for n_passage, n_eta in other_grids:
            for mach in (MACH_LOW, MACH_HIGH):
                add("G", on["grid"], spec(f"G_{n_passage}x{n_eta}_M{mach:.2f}_cold", mach,
                                          f"G: {n_passage} x {n_eta}, M {mach:.2f} cold", n_passage=n_passage, n_eta=n_eta))
        for mach in (MACH_LOW, MACH_HIGH):
            add("Y", on["geometry"], spec(f"Y_emery_base_M{mach:.2f}", mach, f"Y: emery_base thickness, M {mach:.2f} cold",
                                          profile_variant="emery_base"))
        for kind, mach in WHAT_IF_VARIANTS:
            add("W", on["whatif"], spec(f"W_{kind.replace('+', '')}_M{mach:.2f}", mach,
                                        f"WHAT-IF {kind}: M {mach:.2f} cold (baseline grid)", what_if=kind,
                                        trace=(kind == "V")))
        for mach in WHAT_IF_MACHS:
            add("W", on["whatif"], spec(f"W_V_M{mach:.2f}", mach, f"WHAT-IF V: M {mach:.2f} cold (baseline grid)", what_if="V"))
        for n_passage, n_eta in other_grids:
            for mach in (MACH_LOW, MACH_HIGH):
                add("W", on["whatif"], spec(f"W_V_{n_passage}x{n_eta}_M{mach:.2f}", mach,
                                            f"WHAT-IF V: M {mach:.2f} cold, grid {n_passage} x {n_eta}", what_if="V",
                                            n_passage=n_passage, n_eta=n_eta))

        chain = pool.submit(continuation_job, low_tag, MACH_LOW, CONTINUATION_MACHS) if on["continuation"] else None
        run(jobs)

        table(have(high_tag, *tags["A"]), SUMMARY_COLUMNS, "B. INITIAL DENSITY: A1 = the cold baseline, A2, A3 (same TE-angle start)")
        table(have(*tags["E"]), [("label", "case"), ("final_Mmax", "Mmax of the pass"), ("final_Mmax_rel", "at node"),
                                 ("ss_wall_peak", "SS wall peak"), ("init_beta_deg", "exit angle")],
              "E. ONE FROZEN-DENSITY PASS (no density feedback)")
        table(have(high_tag, "A3_warm_density", *tags["B"]), SUMMARY_COLUMNS,
              "B. DENSITY x EXIT-ANGLE START: B1 = cold baseline, B2 = A3, B3, B4")
        table(have(*tags["D"]), SUMMARY_COLUMNS, "D. FIXED DISCHARGE ANGLE (density loop alone, no Kutta)")
        write_csv("initialization_study.csv", have(high_tag, *tags["A"], *tags["E"]))
        write_csv("warm_start_study.csv", have(high_tag, "A3_warm_density", *tags["B"]))
        write_csv("fixed_exit_study.csv", have(*tags["D"]))

        if on["continuation"]:
            warm_rows = keep(chain.result())
            cold = sorted(have(low_tag, *tags["C_cold"], high_tag), key=lambda r: r["mach"])
            good = [r["mach"] for r in cold if r["status"] == "converged subsonic"]
            bad = [r["mach"] for r in cold if r["status"] != "converged subsonic" and (not good or r["mach"] > max(good))]
            if good and bad and min(bad) - max(good) > REFINEMENT_STEP + 1e-9:
                run([spec(f"cold_M{m:.4f}", round(float(m), 6), f"M {m:.4f} cold")
                     for m in np.arange(max(good) + REFINEMENT_STEP, min(bad) - 1e-9, REFINEMENT_STEP)])
            fixed = [spec(f"contraction_M{row['mach']:.4f}", row["mach"], "fixed exit at its own beta*, cold", mode="fixed_exit",
                          exit_deg=row["beta_out_deg"])
                     for row in have(low_tag) + [r for r in warm_rows if r["status"].startswith("converged")]]
            fixed += [spec(f"contraction_M{mach:.4f}", mach, "fixed exit at beta*(0.60), cold", mode="fixed_exit", exit_deg=beta_low)
                      for mach in CONTRACTION_MACHS]
            run(fixed)
        cold = sorted((r for t, r in by_tag.items() if t.startswith("cold_M")), key=lambda r: r["mach"])
        cold = sorted(have(low_tag, high_tag) + cold, key=lambda r: r["mach"])
        warm_rows = sorted((r for t, r in by_tag.items() if t.startswith("warm_M")), key=lambda r: r["mach"])
        continuation = [dict(r, kind="cold") for r in cold] + [dict(r, kind="warm") for r in warm_rows]
        write_csv("mach_continuation.csv", continuation)
        columns = [("mach", "M1"), ("status", "status"), ("passes", "passes"), ("beta_out_deg", "beta_out"),
                   ("final_Mmax", "final Mmax"), ("history_Mmax", "hist Mmax"), ("first_sonic_pass", "1st sonic"),
                   ("final_Mmax_rel", "at node"), ("ss_wall_peak", "SS wall")]
        table(cold, columns, "C. MACH CONTINUATION: COLD STARTS (total density, TE angle)")
        table(warm_rows, columns + [("continued_from", "from M")], "C. MACH CONTINUATION: WARM STARTS (previous density and angle)")
        lam = BASELINE["density_relaxation"]
        contraction = sorted((r for t, r in by_tag.items() if t.startswith("contraction_M")), key=lambda r: r["mach"])
        for row in contraction:
            if row.get("late_r_rho"):
                row["unrelaxed_map_factor"] = (row["late_r_rho"] - (1.0 - lam)) / lam          # r = 1 - lam + lam g
        write_csv("contraction_study.csv", contraction)
        table(contraction, [("mach", "M1"), ("status", "status"), ("passes", "passes"), ("final_Mmax", "Mmax"),
                            ("late_r_rho", "r_rho"), ("late_r_W", "r_W"), ("unrelaxed_map_factor", "g = (r - 1 + lam)/lam")],
              f"CONTRACTION OF THE DENSITY MAP ALONE (fixed angle, lambda_rho {lam}; late ratio per pass)")
        table([r for r in have(low_tag) + warm_rows if r["status"].startswith("converged")],
              [("mach", "M1"), ("late_r_rho", "r_rho"), ("late_r_W", "r_W")], "LATE RATIOS OF THE COUPLED RUNS (set by the Kutta correction)")

        if on["grid"]:
            warm_jobs = []
            for n_passage, n_eta in other_grids:
                tag = f"G_{n_passage}x{n_eta}_M{MACH_LOW:.2f}_cold"
                field = OUT / "fields" / f"{tag}.npz"
                if tag in by_tag and by_tag[tag]["status"].startswith("converged") and field.exists():
                    warm_jobs.append(spec(f"G_{n_passage}x{n_eta}_M{MACH_HIGH:.2f}_warm", MACH_HIGH,
                                          f"G: {n_passage} x {n_eta}, M {MACH_HIGH:.2f} warm (its M {MACH_LOW:.2f} density and angle)",
                                          n_passage=n_passage, n_eta=n_eta, density=str(field), exit_deg=by_tag[tag]["beta_out_deg"]))
            run(warm_jobs)
        grid_rows = [dict(r, label="G: baseline grid, " + r["label"]) for r in have(low_tag, high_tag, "B4_full_warm")]
        grid_rows += [r for t, r in by_tag.items() if t.startswith("G_")]
        write_csv("grid_study.csv", grid_rows)
        table(sorted(grid_rows, key=lambda r: (r.get("n_passage", 0), r["mach"], r["tag"])),
              [("n_passage", "n_passage"), ("n_eta", "n_eta")] + SUMMARY_COLUMNS, "H. GRID SENSITIVITY")
        write_csv("relaxation_study.csv", have(*tags["R"]))
        table(have(*tags["R"]), SUMMARY_COLUMNS + [("lam_rho", "lambda_rho")], "DENSITY RELAXATION (lambda_K fixed)")
        write_csv("geometry_variant_study.csv", have(*tags["Y"]))
        table(have(*tags["Y"]), SUMMARY_COLUMNS + [("failure", "failure")], "GEOMETRY VARIANT (emery_base: the plain 65-010 trailing edge)")
        write_csv("what_if_study.csv", have(*tags["W"]))
        table(have(low_tag, high_tag, *tags["W"]), SUMMARY_COLUMNS,
              "WHAT-IF (DIAGNOSTIC VARIANTS, NOT PRODUCTION): same-side m-derivatives at the transition stations")
        compare_what_if()

    data = None
    if on["forensics"]:
        critical, data = analyse_traces()
        if data:
            report_forensics(critical, data)
    summary = have(low_tag, high_tag, "M065_cold_production_stop", "A2_static_density", "A3_warm_density", "B3_warm_beta",
                   "B4_full_warm", "D_fixed_beta060", "D_fixed_beta060_warm", "warm_M0.6500", "W_V_M0.65", "W_Vle_M0.65",
                   "W_Cle_M0.65", "W_VCle_M0.65")
    write_csv("case_summary.csv", list(by_tag.values()))
    write_csv("comparison_table.csv", summary)
    table(summary, SUMMARY_COLUMNS + [("failure", "failure reason")], "AUTOMATIC COMPARISON TABLE")
    make_figures(by_tag, data, continuation, contraction)
    say(f"\nall results in {OUT}  ({time.perf_counter() - started:.0f} s)")
    (OUT / "report.txt").write_text("\n".join(LOG), encoding="utf-8")       # the tables of everything on file
    return 0


if __name__ == "__main__":
    sys.exit(main())
