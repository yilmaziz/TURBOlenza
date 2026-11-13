"""
transition_study.py
===================
The LE / TE transition rows of the stream-function equation on the five standard grid levels:
is the beta_out grid drift the D, E differencing at the transitions? A STUDY on the baseline of
scripts/kutta_study.py (Dunavant NACA 65-(12A10)10, M1 = 0.65, kovach_sandercock, C2, stagger 33
deg, solidity 1, inlet 45 deg, lambda_rho 0.7, tolerances 1e-8, 600 passes, sonic stop on).
Production defaults stay unchanged; the candidate form sits behind
config.IterationSettings.transition_form.

    python scripts/transition_study.py phaseA     # the Stokes defect at the transitions on all five levels
    python scripts/transition_study.py phaseB     # operator consistency on an exact uniform flow (no BCs)
    python scripts/transition_study.py phaseC     # the "flux" transition rows on the same test (+ solves if they decay)
    python scripts/transition_study.py phaseD     # figures and CSVs from the saved data

Frames (Aungier 2003 Ch. 5): beta from the meridional direction, tan(beta) = dy/dm, eta = 0 the
wall theta0, eta = 1 the wall theta1, S = y1 - y0; planar, r = 1. Lengths in chords.
"""

# ============================================================
# SETTINGS
# ============================================================

OUTPUT_DIR = "outputs/transition_study"
N_WORKERS = 5
GAMMAS_DEG = (20.0, 33.0, 45.0)        # Phase B: uniform-flow angles from the meridional direction
TE_ELEMENTS = 3                         # Phase A: blade elements counted next to each transition
SENSITIVITY_STEP = 1e-4                 # Phase A: +- step of tan(beta_out) for d Gamma_exit / d tan

# ============================================================

import argparse                                                                  # noqa: E402
import json                                                                      # noqa: E402
import math                                                                      # noqa: E402
import sys                                                                       # noqa: E402
import warnings                                                                  # noqa: E402
from pathlib import Path                                                         # noqa: E402

import numpy as np                                                               # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import kutta_study as ks                                                         # noqa: E402  (the same baseline)

OUT = ROOT / OUTPUT_DIR
LEVELS = ks.LEVELS
LINES: list = []


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


def load(name: str):
    return json.loads((OUT / "data" / name).read_text(encoding="utf-8"))


def write_csv(name: str, rows: list) -> None:
    import csv

    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(OUT / name, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: (json.dumps(v) if isinstance(v, (list, tuple, dict)) else v) for k, v in r.items()})


def write_lines(name: str) -> None:
    (OUT / name).write_text("\n".join(LINES) + "\n", encoding="utf-8")
    LINES.clear()


def power_law(dm: list, values: list) -> dict:
    """|value| = a dm^p by least squares in log-log (p > 0: decays under refinement)."""
    x, y = np.log(np.asarray(dm, float)), np.log(np.abs(np.asarray(values, float)))
    ok = np.isfinite(y)
    if ok.sum() < 2:
        return dict(a=math.nan, p=math.nan)
    p, log_a = np.polyfit(x[ok], y[ok], 1)
    return dict(a=float(math.exp(log_a)), p=float(p))


# --- Phase A: the Stokes defect at the transitions ---------------------------------------------------------------

def circulation(grid, metrics, flow) -> np.ndarray:
    """Gamma_k = S_k int_0^1 W_theta d(eta) at every station (trapezoidal rule), m/s x chord."""
    return np.asarray(metrics.S) / grid.chord * np.trapezoid(np.asarray(flow.W_theta), grid.eta, axis=1)


def wall_integrand(grid, metrics, flow) -> np.ndarray:
    """g_k = W1 / cos(beta1) - W0 / cos(beta0) at every station: on the periodic stations the two
    walls are one line and its image, so g is zero there to round-off."""
    J = len(grid.eta) - 1
    W = np.hypot(flow.W_m, flow.W_theta)
    b0, b1 = np.arctan(np.asarray(metrics.tan_beta0)), np.arctan(np.asarray(metrics.tan_beta1))
    return W[:, J] / np.cos(b1) - W[:, 0] / np.cos(b0)


def stokes_elements(grid, metrics, flow) -> list:
    """Every element (k-1, k) from the LE jump (f-1, f) to the TE jump (l, l+1): LHS = Gamma_k -
    Gamma_(k-1), RHS_cos by the trapezoidal rule in m of g (the jump elements with g of their
    periodic station, zero to round-off)."""
    c = grid.chord
    f, l = grid.first_blade_i, grid.last_blade_i
    gamma, g = circulation(grid, metrics, flow), wall_integrand(grid, metrics, flow)
    rows = []
    for k in range(f, l + 2):
        h = float(grid.m[k] - grid.m[k - 1]) / c
        lhs, rhs = float(gamma[k] - gamma[k - 1]), 0.5 * h * float(g[k] + g[k - 1])
        kind = "LE jump" if k == f else "TE jump" if k == l + 1 else "blade"
        rows.append(dict(element=f"({k - 1},{k})", k=k, kind=kind, k_from_f=k - f, l_minus_k=l - k, lhs=lhs, rhs_cos=rhs,
                         defect=lhs - rhs, g_km1=float(g[k - 1]), g_k=float(g[k])))
    return rows


def te_location(grid, cascade) -> dict:
    """Where the two wall nodes of the Kutta station K = last_blade_i lie: distance from the TE
    circle and the curvature of their curve times R_TE (1 on the circle, ~0 on the straight
    Kovach taper), with beta0, beta1 at K."""
    from turbolenza.b2b import kutta as kt
    from turbolenza.b2b.metrics import build_metrics

    st = kt.trailing_edge_station(grid)
    K, f, J = st.i, grid.first_blade_i, len(grid.eta) - 1
    met = build_metrics(grid)
    center = np.asarray(cascade.active.trailing_edge_center, float)
    R = float(cascade.active.trailing_edge_radius)
    shift = np.asarray(grid.boundaries.periodic_translation, float)
    curves = grid.boundaries.blade_curves
    out = dict(K=K, beta0_deg=math.degrees(math.atan(float(met.tan_beta0[K]))),
               beta1_deg=math.degrees(math.atan(float(met.tan_beta1[K]))), R_TE_over_c=R / grid.chord)
    for side, wall, j, moved in (("lower", grid.wall_identity.lower, 0, np.zeros(2)), ("upper", grid.wall_identity.upper, J, shift)):
        surface, u = str(wall.surface[K - f]), float(wall.u[K - f])
        node = np.asarray(grid.nodes[K, j], float) - moved
        distance = (float(np.hypot(*(node - center))) - R) / grid.chord
        kappa = float(getattr(curves, surface).curvature(u)) * R
        where = ("TE circle" if abs(distance) < 1e-4 and abs(abs(kappa) - 1.0) < 0.05 else
                 "straight taper" if abs(kappa) < 0.05 else "fair / transition")
        out[f"{side}_surface"], out[f"{side}_u"] = surface, u
        out[f"{side}_distance_from_circle_over_c"], out[f"{side}_kappa_R_TE"], out[f"{side}_where"] = distance, kappa, where
    return out


def phaseA_job(level: str) -> dict:
    """The converged solve of one level (station Kutta, aungier_13_57, alpha 1), its Stokes
    elements, the Kutta-station location and d Gamma_exit / d tan(beta_out) at frozen density."""
    warnings.simplefilter("ignore")
    from turbolenza.b2b import controller as ct
    from turbolenza.b2b.boundary_conditions import build_boundary_conditions
    from turbolenza.b2b.coefficients import build_coefficients
    from turbolenza.b2b.flow import velocity_components
    from turbolenza.b2b.grid import rebuild_exit
    from turbolenza.b2b.linear_solver import solve_linear_system
    from turbolenza.b2b.matrix_assembly import assemble_from_coefficients
    from turbolenza.b2b.metrics import build_metrics

    case = ks.case_of(level, kutta_method="aungier_13_57", kutta_relaxation=1.0)
    run = ct.run_case(case)
    sol = run.solution
    out = dict(level=level, status=run.status, converged=bool(run.converged))
    if sol is None:
        return out
    grid = sol.grid
    rows = stokes_elements(grid, sol.metrics, sol.flow)
    gamma = circulation(grid, sol.metrics, sol.flow)
    rho = np.array(sol.density)
    gam = {}
    for sign in (-1.0, 1.0):
        beta = math.atan(math.tan(sol.exit_angle) + sign * SENSITIVITY_STEP)
        g = rebuild_exit(grid, beta)
        met, bcs = build_metrics(g), build_boundary_conditions(g)
        co = build_coefficients(met, rho, sol.mass_flow_per_depth, c_ref=sol.c_ref)
        flow = velocity_components(solve_linear_system(assemble_from_coefficients(co, met, bcs)).psi, rho, co, met, bcs)
        gam[sign] = float(circulation(g, met, flow)[-1])
    out.update(beta_out_deg=math.degrees(sol.exit_angle), passes=len(sol.passes), dm_over_c=grid.dm / grid.chord,
               elements=rows, gamma=gamma.tolist(), m_over_c=(np.asarray(grid.m) / grid.chord).tolist(),
               f=grid.first_blade_i, l=grid.last_blade_i, gamma_exit=float(gamma[-1]),
               d_gamma_exit_d_tan=(gam[1.0] - gam[-1.0]) / (2.0 * SENSITIVITY_STEP),
               location=te_location(grid, case.cascade))
    return out


def phaseA() -> None:
    from concurrent.futures import ProcessPoolExecutor

    say("## Phase A: the Stokes defect at the transitions, all five levels")
    say()
    say("Solves: station Kutta, kutta_method aungier_13_57, alpha 1 (the production correction and fixed point).")
    say()
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        results = list(pool.map(phaseA_job, LEVELS))
    rows, loc_rows, elem_rows = [], [], []
    for r in results:
        if not r.get("elements"):
            say(f"- {r['level']}: no solution ({r['status']})")
            continue
        el = r["elements"]
        blade = [e for e in el if e["kind"] == "blade"]
        te_jump = next(e for e in el if e["kind"] == "TE jump")
        le_jump = next(e for e in el if e["kind"] == "LE jump")
        last3, first3 = blade[-TE_ELEMENTS:], blade[:TE_ELEMENTS]
        dG_te = sum(e["defect"] for e in last3) + te_jump["defect"]
        dG_le = le_jump["defect"] + sum(e["defect"] for e in first3)
        interior = [e for e in blade if e["k_from_f"] > TE_ELEMENTS and e["l_minus_k"] >= TE_ELEMENTS]
        beta = math.radians(r["beta_out_deg"])
        dbeta = dG_te / r["d_gamma_exit_d_tan"] * math.cos(beta) ** 2
        rows.append(dict(level=r["level"], dm_over_c=r["dm_over_c"], beta_out_deg=r["beta_out_deg"], passes=r["passes"],
                         status=r["status"], dGamma_TE=dG_te, dGamma_LE=dG_le,
                         te_last3=sum(e["defect"] for e in last3), te_jump=te_jump["defect"],
                         le_jump=le_jump["defect"], le_first3=sum(e["defect"] for e in first3),
                         interior_sum=sum(e["defect"] for e in interior),
                         interior_median_abs=float(np.median([abs(e["defect"]) for e in interior])),
                         gamma_exit=r["gamma_exit"], d_gamma_exit_d_tan=r["d_gamma_exit_d_tan"],
                         dbeta_est_deg=math.degrees(dbeta), beta_corrected_deg=r["beta_out_deg"] - math.degrees(dbeta)))
        loc = r["location"]
        loc_rows.append(dict(level=r["level"], **loc))
        for e in el:
            elem_rows.append(dict(level=r["level"], **e))
    table(rows, [("level", "level"), ("dm_over_c", "dm/c"), ("beta_out_deg", "beta_out"), ("passes", "passes"),
                 ("te_last3", "last 3 elements"), ("te_jump", "jump l->l+1"), ("dGamma_TE", "dGamma_TE"),
                 ("le_jump", "jump f-1->f"), ("le_first3", "first 3 elements"), ("dGamma_LE", "dGamma_LE"),
                 ("interior_median_abs", "interior median |defect|")],
          "Stokes defect LHS - RHS_cos (m/s x chord) summed at each transition")
    table(loc_rows, [("level", "level"), ("K", "K"), ("beta0_deg", "beta0 deg"), ("beta1_deg", "beta1 deg"),
                     ("lower_surface", "wall 0"), ("lower_where", "wall 0 at K"),
                     ("lower_distance_from_circle_over_c", "wall 0: dist. from TE circle /c"),
                     ("lower_kappa_R_TE", "wall 0: kappa R_TE"), ("upper_surface", "wall 1"), ("upper_where", "wall 1 at K"),
                     ("upper_distance_from_circle_over_c", "wall 1: dist. /c"), ("upper_kappa_R_TE", "wall 1: kappa R_TE")],
          "Where the Kutta station's wall nodes lie (R_TE = 0.008 c)", digits=5)
    dm = [r["dm_over_c"] for r in rows]
    fits = {name: power_law(dm, [r[name] for r in rows]) for name in ("dGamma_TE", "dGamma_LE", "te_jump", "te_last3")}
    for name, fit in fits.items():
        say(f"- power law |{name}| = {fit['a']:.4g} dm^{fit['p']:.3f}   (signs: "
            f"{', '.join('+' if r[name] > 0 else '-' for r in rows)})")
    say()
    table(rows, [("level", "level"), ("beta_out_deg", "beta_out"), ("dGamma_TE", "dGamma_TE"),
                 ("d_gamma_exit_d_tan", "dGamma_exit/dtan"), ("dbeta_est_deg", "dbeta_est deg"),
                 ("beta_corrected_deg", "beta_out - dbeta_est")],
          "The estimated effect of the TE defect on beta_out: dbeta_est = dGamma_TE / (dGamma_exit/dtan) cos^2(beta_out)")
    for key in ("beta_out_deg", "beta_corrected_deg"):
        b = [r[key] for r in rows]
        d = [b[k + 1] - b[k] for k in range(len(b) - 1)]
        say(f"- {key}: successive differences {', '.join(f'{x:+.4f}' for x in d)} deg; |fine - very_fine| "
            f"{abs(b[-1] - b[-2]):.4f}; span {max(b) - min(b):.4f}")
    say()
    write_csv("phaseA_defects.csv", rows)
    write_csv("phaseA_elements.csv", elem_rows)
    write_csv("phaseA_te_location.csv", loc_rows)
    save("phaseA.json", dict(rows=rows, locations=loc_rows, fits=fits, results=results))
    write_lines("phaseA.md")


# --- Phase B / C: the operator on an exact uniform flow ---------------------------------------------------------

def uniform_psi(grid, gamma_deg: float) -> np.ndarray:
    """psi of a uniform flow at gamma from the meridional direction, constant density and b:
    rho W_m = mdot' dpsi/dy and rho W_theta = -mdot' dpsi/dm (flow.py's Eqs. 5-18, 5-19 with
    the eta derivatives turned back into y), mdot' = rho W cos(gamma) pitch, so
    psi = (y - m tan(gamma)) / pitch; it rises by one across a pitch."""
    nodes = np.asarray(grid.nodes, float)
    return (nodes[..., 1] - nodes[..., 0] * math.tan(math.radians(gamma_deg))) / grid.pitch


def manufactured_case(level: str):
    """The level's grid (the first coupled grid of the case), its metrics, boundary relations
    and coefficients at rho = 1, mdot' = 1, normalized mode (c_ref None)."""
    from turbolenza.b2b.boundary_conditions import build_boundary_conditions
    from turbolenza.b2b.coefficients import build_coefficients
    from turbolenza.b2b.metrics import build_metrics

    grid = ks.case_of(level).grid
    met = build_metrics(grid)
    bcs = build_boundary_conditions(grid)
    co = build_coefficients(met, np.ones((len(grid.m), len(grid.eta))), 1.0, c_ref=None)
    return grid, met, bcs, co


def raw_rows(A, B, C, D, E, grid, bcs, dm, deta, psi, form="expanded"):
    """The raw residual (the Eq. 5-21 / 5-20 left side, not divided by -H) of every PDE row on
    psi: the interior rows (n_m, n_eta), NaN elsewhere, and the periodic-line rows {(i, J): value}.
    form "flux" replaces the interior rows of the transition stations by the face-flux rows."""
    from turbolenza.b2b import discretization as dz

    st = dz._interior_stencil(A, B, C, D, E, dm, deta)
    w = np.asarray(st.weights)
    if form == "flux":
        w = np.array(w)
        w_flux, rows = dz._flux_transition_weights(A, B, C, dm, deta, grid.first_blade_i, grid.last_blade_i, st.H)
        for (i, j) in rows:
            w[i, j] = w_flux[i, j]
    n_m, n_eta = psi.shape
    norm = np.full((n_m, n_eta), np.nan)
    for i in range(1, n_m - 1):
        block = np.stack([psi[i - 1:i + 2, j - 1:j + 2] for j in range(1, n_eta - 1)])   # (n_eta - 2, 3, 3)
        norm[i, 1:-1] = np.sum(w[i, 1:-1] * block, axis=(1, 2)) - np.asarray(st.rhs)[i, 1:-1]
    raw = -np.asarray(st.H) * norm
    n_eta_ = n_eta
    per = {}
    for row in dz._periodic_pde_rows(A, B, C, D, E, dm, deta, grid.first_blade_i, grid.last_blade_i,
                                     lambda k: bcs.periodic_ghost(k, n_eta_)):
        value = sum(t.coefficient * psi[t.node] for t in row.terms) - row.rhs
        per[row.target] = (row.kind, -row.H * value)
    return raw, per


def term_split(co, grid, bcs, psi, form="expanded") -> dict:
    """The raw residual and its terms by linearity in A .. E (H only rescales the normalized row,
    the raw one is free of it): second-derivative terms A psi_ee + C psi_mm (B = 0), the mixed
    term -2 B psi_me, D psi_e, E psi_m; Q = 0. For "flux" the transition rows have no D, E:
    their split is the A, B and C parts of the face fluxes."""
    A, B, C, D, E = (np.asarray(getattr(co, k)) for k in "ABCDE")
    Z = np.zeros_like(A)
    dm, deta = co.length_scale * grid.dm, grid.deta
    full = raw_rows(A, B, C, D, E, grid, bcs, dm, deta, psi, form)
    no_de = raw_rows(A, B, C, Z, Z, grid, bcs, dm, deta, psi, form)
    ac = raw_rows(A, Z, C, Z, Z, grid, bcs, dm, deta, psi, form)
    a_only = raw_rows(A, Z, Z, Z, Z, grid, bcs, dm, deta, psi, form)
    with_d = raw_rows(A, B, C, D, Z, grid, bcs, dm, deta, psi, form)
    with_e = raw_rows(A, B, C, Z, E, grid, bcs, dm, deta, psi, form)
    out = {}
    for name, (a, b) in dict(full=(full, None), without_DE=(no_de, None), second=(ac, None), A_ee=(a_only, None),
                             C_mm=(ac, a_only), mixed=(no_de, ac), D_term=(with_d, no_de),
                             E_term=(with_e, no_de)).items():
        grid_part = a[0] if b is None else a[0] - b[0]
        per_part = {key: (a[1][key][0], a[1][key][1] - (0.0 if b is None else b[1][key][1])) for key in a[1]}
        out[name] = (grid_part, per_part)
    return out


EDGE_ZONE_CHORDS = 0.1      # blade rows this close (in m) to the first / last blade station are "near LE / TE"


def row_classes(grid) -> dict:
    """Masks of the interior rows (j = 1 .. J - 1) by class, and (by name) the periodic-line rows.
    "interior" is every row not in one of the named station classes; it is split further into
    the blade rows near the LE and the TE (within EDGE_ZONE_CHORDS of the first / last blade
    station), the smooth middle of the blade (its wall-adjacent rows j = 1, J - 1 apart), and the
    periodic stations near the transitions (f-3, f-2, l+2, l+3) and far from them."""
    n_m, n_eta = len(grid.m), len(grid.eta)
    J = n_eta - 1
    f, l = grid.first_blade_i, grid.last_blade_i
    m = np.asarray(grid.m)[:, None] * np.ones((1, n_eta))
    i_idx = np.arange(n_m)[:, None] * np.ones((1, n_eta), int)
    j_idx = np.ones((n_m, 1), int) * np.arange(n_eta)[None, :]
    base = (i_idx >= 1) & (i_idx <= n_m - 2) & (j_idx >= 1) & (j_idx <= J - 1)
    masks = {}
    taken = np.zeros_like(base)
    for i, name in {f - 1: "f-1", f: "f", f + 1: "f+1", l - 1: "l-1", l: "l", l + 1: "l+1"}.items():
        masks[name] = base & (i_idx == i)
        taken |= masks[name]
    interior = base & ~taken
    blade = (i_idx >= f) & (i_idx <= l)
    c = grid.chord
    near_le = blade & (m < grid.m[f] + EDGE_ZONE_CHORDS * c)
    near_te = blade & (m > grid.m[l] - EDGE_ZONE_CHORDS * c)
    middle = blade & ~near_le & ~near_te
    wall = (j_idx == 1) | (j_idx == J - 1)
    near_tr = ((i_idx >= f - 3) & (i_idx <= f - 2)) | ((i_idx >= l + 2) & (i_idx <= l + 3))
    masks["interior"] = interior
    masks["blade near LE (f+2 .. 0.1 c)"] = interior & near_le
    masks["blade near TE (0.1 c .. l-2)"] = interior & near_te
    masks["blade middle, wall-adjacent"] = interior & middle & wall
    masks["blade middle"] = interior & middle & ~wall
    masks["periodic near transitions"] = interior & ~blade & near_tr
    masks["periodic far"] = interior & ~blade & ~near_tr
    return masks


CLASS_ORDER = ("interior", "blade near LE (f+2 .. 0.1 c)", "blade middle", "blade middle, wall-adjacent",
               "blade near TE (0.1 c .. l-2)", "periodic near transitions", "periodic far",
               "f-1", "f", "f+1", "l-1", "l", "l+1",
               "periodic line", "periodic LE row (f-1, J)", "periodic TE row (l+1, J)")
TERMS = ("full", "without_DE", "second", "A_ee", "C_mm", "mixed", "D_term", "E_term")
SUMMED = ("second", "mixed", "D_term", "E_term")        # the terms that add up to "full"


def class_stats(split: dict, grid) -> dict:
    """max |.| and RMS per row class and term."""
    from turbolenza.b2b import discretization as dz

    masks = row_classes(grid)
    kind_name = {dz.PERIODIC_STANDARD: "periodic line", dz.PERIODIC_LE_TRANSITION: "periodic LE row (f-1, J)",
                 dz.PERIODIC_TE_TRANSITION: "periodic TE row (l+1, J)"}
    out = {}
    f, l = grid.first_blade_i, grid.last_blade_i
    for cls in CLASS_ORDER:
        out[cls] = {}
        for term in TERMS:
            field, per = split[term]
            where = ""
            if cls in masks:
                v = field[masks[cls]]
                if v.size and term == "full":
                    a = np.where(masks[cls], np.abs(field), -1.0)
                    i, j = np.unravel_index(int(np.argmax(a)), a.shape)
                    where = f"({i}, {j}): i-f {i - f:+d}, l-i {l - i:+d}, m/c {grid.m[i] / grid.chord:.4f}"
            else:
                v = np.array([val for (kind, val) in per.values() if kind_name[kind] == cls])
            v = v[np.isfinite(v)]
            out[cls][term] = dict(max=float(np.max(np.abs(v))) if v.size else math.nan,
                                  rms=float(np.sqrt(np.mean(v ** 2))) if v.size else math.nan, n=int(v.size), where=where)
    return out


def operator_study(form: str) -> dict:
    results = {}
    for level in LEVELS:
        grid, met, bcs, co = manufactured_case(level)
        for gamma in GAMMAS_DEG:
            psi = uniform_psi(grid, gamma)
            split = term_split(co, grid, bcs, psi, form)
            # the split adds up to the full raw residual (linearity check)
            f_ = split["full"][0]
            s_, m_, d_, e_ = (split[t][0] for t in SUMMED)
            ok = np.isfinite(f_)
            gap = float(np.max(np.abs(f_[ok] - (s_[ok] + m_[ok] + d_[ok] + e_[ok])))) if ok.any() else math.nan
            results[(level, gamma)] = dict(stats=class_stats(split, grid), dm_over_c=grid.dm / grid.chord,
                                           deta=grid.deta, split_gap=gap,
                                           scale=float(np.nanmax(np.abs(np.asarray(co.C)))) / (co.length_scale * grid.dm) ** 2)
    return results


def report_operator(results: dict, form: str, title: str) -> list:
    rows = []
    for gamma in GAMMAS_DEG:
        dm = [results[(level, gamma)]["dm_over_c"] for level in LEVELS]
        for cls in CLASS_ORDER:
            for term in TERMS:
                maxes = [results[(level, gamma)]["stats"][cls][term]["max"] for level in LEVELS]
                rmss = [results[(level, gamma)]["stats"][cls][term]["rms"] for level in LEVELS]
                fit = power_law(dm, maxes)
                rows.append(dict(form=form, gamma_deg=gamma, row_class=cls, term=term,
                                 **{f"max_{level}": v for level, v in zip(LEVELS, maxes)},
                                 **{f"rms_{level}": v for level, v in zip(LEVELS, rmss)},
                                 order_max=fit["p"],
                                 order_last=(math.log(abs(maxes[-2]) / abs(maxes[-1])) / math.log(dm[-2] / dm[-1])
                                             if maxes[-1] and maxes[-2] and np.isfinite(maxes[-1]) else math.nan)))
    for r in rows:
        r["where_very_fine"] = results[("very_fine", r["gamma_deg"])]["stats"][r["row_class"]][r["term"]]["where"]
    for gamma in GAMMAS_DEG:
        sel = [r for r in rows if r["gamma_deg"] == gamma and r["term"] == "full"]
        table(sel, [("row_class", "row class")] + [(f"max_{level}", level) for level in LEVELS]
              + [("order_max", "order (fit)"), ("order_last", "order f->vf"), ("where_very_fine", "max at (very_fine)")],
              f"{title}: max |raw residual| of the full row, gamma = {gamma:g} deg (p > 0 decays)", digits=3)
    sel = [r for r in rows if r["gamma_deg"] == 33.0 and r["term"] == "full"]
    table(sel, [("row_class", "row class")] + [(f"rms_{level}", level) for level in LEVELS],
          f"{title}: RMS raw residual of the full row, gamma = 33 deg", digits=3)
    gamma = 33.0
    for term in TERMS[1:]:
        sel = [r for r in rows if r["gamma_deg"] == gamma and r["term"] == term]
        table(sel, [("row_class", "row class")] + [(f"max_{level}", level) for level in LEVELS]
              + [("order_max", "order (fit)")], f"{title}: max |{term}| by row class, gamma = 33 deg", digits=3)
    gaps = [results[k]["split_gap"] for k in results]
    say(f"Linearity check of the split (full - sum of the terms), largest over all rows, levels and gammas: "
        f"{max(gaps):.2e} (raw units; the full residuals reach {max(r['max_very_fine'] for r in rows if r['term'] == 'full'):.3g}).")
    say()
    return rows


def phaseB() -> None:
    say("## Phase B: operator consistency on an exact uniform flow (no BCs, no Kutta)")
    say()
    say("psi = (y - m tan(gamma)) / pitch at the grid nodes (constant rho = 1, mdot' = 1, normalized mode) is an exact "
        "solution of Eq. 5-20/5-21 wherever the (m, eta) mapping is smooth. Residual = the raw left side of each PDE row "
        "as the solver assembles it (Eq. 5-44 times -H; the periodic-line rows with their ghost and jump), at every "
        "interior node and on the periodic line. Q = 0, so the RHS term is zero in every row; the periodic jump constants "
        "sit inside the terms they belong to.")
    say()
    results = operator_study("expanded")
    rows = report_operator(results, "expanded", "expanded (production)")
    write_csv("phaseB_operator_residuals.csv", rows)
    save("phaseB.json", dict(rows=rows))
    write_lines("phaseB.md")


PHASES = {"phaseA": phaseA, "phaseB": phaseB}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("phase", choices=sorted(PHASES))
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    PHASES[args.phase]()


if __name__ == "__main__":
    main()
