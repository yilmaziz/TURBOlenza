# turbolenza.b2b

> Part of **TURBOlenza** (see the repository README). Until 2026-11-13 this package was the
> stand-alone `potential_flow_solver`; the move to `src/turbolenza/b2b/` changed import paths only.

Blade-to-blade compressible potential flow after R. H. Aungier, *Axial-Flow Compressors*
(ASME Press, 2003), Chapter 5. The method solves a stream-function equation on a sheared
(m, η) grid, wraps it in a density loop for compressibility, and corrects the discharge
angle for the Kutta condition in the same loop.

> **Status:**
>
> | Module | State |
> |---|---|
> | `geometry.py` | **C2 geometry — FROZEN 2026-11-14, the `CascadeSpec` default** (`surface_continuity="C2"`): quintic curves, within 1e-6 c of C1 downstream of the LE, the robust monotone-curvature leading edge ahead of it, and the natural quintic thickness in √station (`"quintic_natural_sqrt"`, C4, exact at the table stations; the C2 default since 2026-11-13). Locked by `test_c2_frozen_fingerprint`; a change needs the user's approval and a new lock. **C1 geometry — VALIDATED LEGACY BASELINE** (asked for by name since 2026-11-14, `surface_continuity="C1"`; unchanged, bit for bit, with its legacy PCHIP thickness `"pchip_sqrt"`, which C2 can still select: the previous C2 bit for bit) |
> | `grid.py` | C1 path: FROZEN geometric grid implementation, unchanged bit for bit (passage boundaries, both extensions, a continuous check that the smooth passage is open, uniform (m, η) nodes, blade-passage topology from the station lines, tracked wall identity, exit rebuild). **C2 path reopened 2026-11-11 — UNDER REVIEW, not frozen:** the walls are the authoritative curves (section 7 of its docstring). **Admissible n_passage added 2026-11-12 — UNDER REVIEW:** the first and last blade station must each cut one suction and one pressure surface (section 8): `make_grid(..., require_ss_ps=True)` and `validate_uniform_grid_topology` raise `GridTopologyError` (never repair), `find_valid_n_passage` and `n_passage_ceiling` are read-only queries; the default build is unchanged. **Geometry-driven standard grids added 2026-11-13 — UNDER REVIEW (section 9):** five named levels (`GRID_LEVELS`, very_coarse … very_fine), `build_grid(cascade, level)`, `build_grid_family(cascade)` → `GridPackage`, `GridSettings(level=...)` through `make_grid` and so the solver; the lattice is anchored to the blade's own SS + PS transitions, so admissibility belongs to the geometry, not to a lucky n_passage; structured `GridGenerationError`; the explicit n_passage path unchanged bit for bit |
> | `metrics.py` | implemented: S, dS/dm, the grid-line angles of Eqs. 5-3, 5-9 and d tan β/dm. C1: finite differences, unchanged bit for bit. **C2 blade walls (2026-11-11) — UNDER REVIEW, not frozen:** exact spline slopes |
> | `coefficients.py` | **implemented 2026-11-12 — UNDER REVIEW, not frozen:** A to E of Eq. 5-21 (Eqs. 5-22 to 5-26) from the metrics, a frozen ρ field and the mass flow per unit depth; planar, non-rotating (no right-hand side); the one physical length scale `c_ref` |
> | `discretization.py` | **interior stencil implemented 2026-11-12 — UNDER REVIEW, not frozen:** Aungier's normalized nine-point Eq. 5-44 (weights Eqs. 5-45 to 5-50) at the interior nodes, second-order central differences (Eqs. 5-35 to 5-39), no upwinding. **Periodic PDE line added 2026-11-12 — UNDER REVIEW:** Eq. 5-44 at (i, n_η − 1) of the periodic stations with the periodic continuation; at the LE transition row a one-sided mixed difference, at the TE transition row one-sided ψ_m, ψ_mm and mixed (revised 2026-11-12: the central TE links ran through the tail). Assembly and the solve: deferred |
> | `boundary_conditions.py` | **implemented 2026-11-12 — UNDER REVIEW, not frozen:** node classification from the grid topology and explicit relations: walls ψ = 0 / 1, periodic ψ_upper − ψ_lower = 1, ψ_m = 0 at the far stations by second-order one-sided differences, with eliminations and periodic continuation. No matrix, no substitution, no Kutta |
> | `matrix_assembly.py` | **implemented 2026-11-13 — UNDER REVIEW, not frozen:** the local equations of `discretization.py` and `boundary_conditions.py` gathered into one square sparse system F Ψ = Q (Aungier 2000, Ch. 13): node number k = i n_η + j, both far stations eliminated by their ψ_m = 0 relations and rebuilt after the solve (`reconstruct`), walls and periodic jumps kept as explicit rows, COO → CSR, no row scaling, no solve |
> | `linear_solver.py` | **implemented 2026-11-13 — UNDER REVIEW, not frozen:** the sparse direct solve of the assembled F Ψ = Q (`scipy.sparse.linalg.spsolve`, SuperLU with its own pivoting; no iterative method, no row scaling, no regularization), ψ(i, j) rebuilt by `matrix_assembly.reconstruct`, residual diagnostics; singular systems and non-finite results refused. The linear algebra only: no velocity, density or Kutta condition |
> | `flow.py` | **implemented 2026-11-13 — UNDER REVIEW, not frozen:** ψ → ψ_m, ψ_η → W_m, W_θ, W², W (Eqs. 5-18, 5-19) from the solved ψ, the density it was solved with and its `CoefficientFields`; the station mass flow as a diagnostic; the stationary inlet closure (M₁, β₁, Pt₁, Tt₁ → inlet state and ṁ′). **ψ_m by m region since 2026-11-13 (the LE / TE transition fix):** `metrics.py`'s three regions (upstream periodic, blade, downstream periodic), central inside a region, three-point one-sided at its ends f − 1, f, l, l + 1, never across the leading or trailing edge; ψ_m = 0 on the walls; every other node bit for bit as before |
> | `eos.py` | **implemented 2026-11-13 — UNDER REVIEW, not frozen:** a calorically and thermally perfect gas (R, γ, cp = γR/(γ − 1), h = cp T); the energy relation h = I + (rω)²/2 − W²/2 with the rothalpy I = cp Tt,in − ω r C_θ,in (I = cp Tt,in for ω = 0); T, p (isentropic to the inlet total state), ρ_EOS = p/(RT), a; Aungier's initial density (the relative total density); the relaxed update ρ_next = (1 − λ)ρ + λρ_target with **λ = 0.5, our baseline** (λ = 1 is Aungier's direct replacement). A state layer: no loop, no solver call |
> | `kutta.py` | **implemented 2026-11-13 — UNDER REVIEW, not frozen:** Aungier 2003 Eq. 5-34 at the last blade station (i = `last_blade_i`, the SS and PS node read from `wall_identity`): R_K = W_SS − W_PS, ε_K = \|R_K\| / (½(W_SS + W_PS)); the exit-angle correction of Aungier 2000 Eqs. 13-57 and 13-59 mapped onto this project's convention (cot β₂₀₀₀ = tan β; the cos β of Eq. 13-57 is the cosine of OUR β, the meridional angle): Δ tan β_out = 2 (W0 − W1) Δm / (S (cos β0 + cos β1) ∫₀¹ W_m dη\|exit); the damped update on tan β. Reads a solved field: no PDE, density, EOS or matrix |
> | `convergence.py` | **change norms and the stop tests implemented 2026-11-13 — UNDER REVIEW:** change = max \|new − old\| / max \|old\| for every field (the velocity as a vector); density test: density change ≤ `density_tolerance` AND velocity change ≤ `velocity_tolerance` (None: `density_tolerance`); Kutta test: ε_K ≤ `kutta_tolerance`; coupled test: both on the same pass; the first pass (no previous velocity) never converges |
> | `solver.py` | **density loop and density + Kutta coupled loop implemented 2026-11-13 — UNDER REVIEW, not frozen:** `solve` runs (ρ⁽ⁿ⁾, β_out⁽ⁿ⁾) → `rebuild_exit`, metrics, boundary relations → A..E → F Ψ = Q → ψ⁽ⁿ⁾ → W⁽ⁿ⁾ → ρ⁽ⁿ⁺¹⁾ (λ_ρ = 0.5) and β_out⁽ⁿ⁺¹⁾ (λ_K = 1.0, ours) by calls only, until the density, velocity and Kutta tolerances hold on one pass, then reports one evaluation at (ρ_final, β_final). `solve_density_coupled` / `solve_at_exit_angle` stay the fixed-angle solver (**DENSITY-COUPLED / FIXED-DISCHARGE-ANGLE SOLUTION, NOT FINAL AUNGIER KUTTA-CONVERGED SOLUTION**), unchanged. Stationary cascades only, M_rel < 1 required. Verified on a constructed case; **not yet validated** against experiment or published results |
> | `postprocess.py` | **implemented 2026-11-13 — UNDER REVIEW:** blade-surface Mach number and static pressure on the blade run only, the suction / pressure wall from `grid.wall_identity` (refused if a wall's surface changes), x/c along the chord (geometry.py's blade frame), the pressure coefficient in Dunavant's convention S = (Pt₁ − p)/q₁ (default) or (p − p₁)/q₁, the discharge angle (the grid's) with the far-exit velocity angle as a check, a summary, a CSV import path for measured points |
> | `plots.py` | **implemented 2026-11-13 — UNDER REVIEW:** Matplotlib figures (imported on call only): surface Mach and pressure coefficient against x/c, SS and PS on one axes, the convergence history; the node-based Mach contour of the whole domain (Gouraud-shaded cells, periodic copies, blades on top; added 2026-11-13); return (fig, ax), never show() |
> | `monitor.py` | **implemented 2026-11-13 — UNDER REVIEW:** observers of a solve (`live_monitor=True`: one live Matplotlib window per solve, updated in place; `verbose=True`: one console line per pass; `monitor=`: any observer). Observation only: identical results with or without |
> | `controller.py` | **implemented 2026-11-13 — UNDER REVIEW:** the user-facing run layer: flat `CaseSettings` → `build_case` (production objects, derived pitch / inlet state / mass flow, admissibility checked before any solve) → `run_case` (the timed solver call: wall time, process CPU time, effective CPU cores, logical CPUs; failures returned, not swallowed) → reports, figures with a runtime footer, CSV / JSON export. No numerics; `scripts/run_full_solver.py` is its editable front end |
> | everything else | architecture only |
>
> **The solver goes with C2 only for now.** After the 2026-11-11 hand-off, `grid.py` takes
> the blade-wall locations and `metrics.py` the blade-wall angles from the C2 geometry (the
> authoritative quintic curves). Since 2026-11-14 C2 is frozen and the `CascadeSpec`
> default, so a spec that does not name `surface_continuity` gets the solver's geometry. A
> fair scientific comparison of C1 against C2 is planned for later (with an exact
> piecewise-C1 derivative path first, so that only the surface continuity differs); until
> then the C1 path stays as it is, the legacy baseline.
> `notebooks/Grid_Metrics.ipynb` walks through the workflow.
>
> The straight m-links across the LE and TE transition intervals are not used by
> `metrics.py` (its m derivatives never cross a transition) and stay deferred to
> `discretization.py` (see "Hard invariants and a diagnostic" below). Every other function
> raises `NotImplementedError`, and every test file except `test_geometry.py`,
> `test_config.py`, `test_grid.py`, `test_metrics.py`, `test_coefficients.py`,
> `test_discretization.py`, `test_boundary_conditions.py`, `test_matrix_assembly.py`,
> `test_linear_solver.py`, `test_flow.py`, `test_eos.py`, `test_kutta.py`, `test_solver.py`,
> `test_postprocess.py`, `test_monitor.py` and `test_controller.py` skips itself (none does now). The coupled solver `solve` gives a
> density + Kutta converged compressible solution (Eq. 5-34 at the last blade station) on the
> constructed stationary verification case; **no result is compared with measurements or
> published results yet** (the NACA 65 / Dunavant validation is the next stage), so none is
> called validated.
> The modules get implemented and reviewed one at a time.
>
> **Verification and validation are kept apart.** *Verification* (done for every module): manufactured and
> closed-form tests, module-to-module consistency (one pass equals the module calls; the final state is one
> evaluation), and the constructed dimensional cascade (33°, M₁ 0.3, test_solver.py). *Validation* is the
> comparison with external measurements, here the NACA 65-(12A10)10 cascade of Dunavant et al. (1955) that
> Aungier shows in Fig. 5-6. It has **not** been performed: `scripts/validate_naca65_12a10_10.py` runs the
> documented condition, but no measured surface data is digitized. At M₁ = 0.65 the subsonic solver used to
> stop at the first (LE) blade station; that was a defect of the velocity recovery, fixed 2026-11-13 (ψ_m by m
> region, decision log): the case now converges subsonically (61 × 41: suction-wall peak 0.906 at x/c 0.22).
> Running the script is still not a validation.

## Running the solver

Edit the constants at the top of `scripts/run_full_solver.py` (sections 0 to 8: run control, gas,
geometry, inlet flow, grid, density, Kutta, runtime, output) and run

```text
python scripts/run_full_solver.py
```

It prints every effective input tagged [user], [derived] or [user = project default], the derived
inlet state and the grid; runs the solver with one live convergence window; prints the result (or the
failure diagnostics); shows the surface Mach figure (with the solver time, the effective CPU use and
the logical CPU count in its footer), Dunavant's S, the convergence history and the Mach contour (node
values, smoothly shaded over the whole domain and the blade passage); and writes
`outputs/<CASE_NAME>/<RUN_LABEL or start time>/` (`surface_mach.png`, `surface_pressure.png`,
`convergence.png`, `mach_contour.png`, `surface_mach.csv`, `surface_pressure.csv`, `convergence_history.csv`,
`run_summary.json`; an existing folder is never overwritten). Exit status 0 converged, 1 case not
built, 2 solve stopped. The default is the NACA 65-(12A10)10 case of Fig. 5-6 at M₁ = 0.65, which converges
subsonically since the 2026-11-13 ψ_m fix (decision log). `python scripts/compare_dunavant_aungier.py` solves the same case on the medium standard grid
and plots its surface Mach number against Dunavant's measured points and Aungier's potential-flow curve
(digitized, `validation/naca65_12a10_10/Dunavant_Aungier_Fix.csv`: x/c and Mach; results in
`outputs/dunavant_aungier_comparison/`). The script only fills `controller.CaseSettings` and calls `controller.build_case`,
`run_case`, `format_run_report`, `plot_run`, `export_run`: a later GUI calls the same functions.

## Package tree

```text
src/turbolenza/b2b/
├── __init__.py
├── config.py                inputs and numerical settings
├── geometry.py              NACA 65 blade, stagger, pitch, throat; C1 or C2 surfaces, robust C2 LE (reopened)
├── grid.py                  (m, η) nodes, regions, exit rebuild; five standard levels (section 9)
├── metrics.py               S, dS/dm, tan β, β at the nodes          (implemented)
├── boundary_conditions.py   boundary classes and relations: walls, periodic jump, far ψ_m = 0 (implemented)
├── coefficients.py          A, B, C, D, E of Eq. 5-21                (implemented)
├── discretization.py        Eq. 5-44 interior stencil and periodic PDE rows (implemented)
├── matrix_assembly.py       local equations → sparse F Ψ = Q, far-boundary elimination (implemented)
├── linear_solver.py         sparse direct solve of F Ψ = Q, ψ(i, j) rebuilt (implemented)
├── eos.py                   perfect gas, W → static state, initial and relaxed density (implemented)
├── flow.py                  ψ → W_m, W_θ, W; inlet state and ṁ′ from the inlet Mach number (implemented)
├── kutta.py                 TE Kutta residual (Eq. 5-34), Aungier exit-angle correction (implemented)
├── convergence.py           change norms, iteration history, density / Kutta / coupled stop tests (implemented)
├── solver.py                density + Kutta coupled loop; density loop at a fixed angle (implemented)
├── postprocess.py           surface Mach, pressure coefficient, x/c, discharge angle, summary (implemented)
├── plots.py                 surface Mach, pressure-coefficient and convergence figures (Matplotlib, optional)
├── monitor.py               live convergence window, console progress (observers; Matplotlib optional)
├── controller.py            user-facing run layer: case, timed run, report, figures, files
└── README.md

tests/
├── test_config.py           unit and naming conventions of the solver inputs
├── test_geometry.py
├── test_grid.py
├── test_grid_family.py      the five standard levels: topology, refinement, perturbed blades, failures
├── test_grid_family_script.py  scripts/analyze_grid_family.py: the final figure's data are the grids' own
├── test_metrics.py
├── test_eos.py
├── test_coefficients.py
├── test_discretization.py
├── test_boundary_conditions.py
├── test_matrix_assembly.py
├── test_linear_solver.py
├── test_flow.py
├── test_kutta.py            Kutta residual, angle convention, correction, frozen-density Kutta loop
├── test_postprocess.py      surfaces, x/c, pressure coefficient, discharge angle, summary, plots
├── test_monitor.py          live monitor and console progress (Agg backend)
├── test_controller.py       run layer and scripts/run_full_solver.py (fast cases)
├── test_diagnose_script.py  helpers of scripts/diagnose_mach060_vs_065.py (no solve)
└── test_solver.py           fixed-angle and coupled solver, convergence, mass flow from the inlet Mach number
```

## Grid generation and the optimization chain

The intended long-term chain (the optimizer is not implemented):

```text
parametric blade geometry            geometry.make_cascade(CascadeSpec)
        |
geometry-driven GridPackage          grid.build_grid_family / grid.build_grid (section 9)
        |
selected grid level                  GridSettings(level="medium"), or StandardGrid.grid
        |
B2B solver                           solver.solve / controller.run_case
        |
objective / report                   postprocess, controller reports
```

Many geometry candidates normally take one level (`build_grid(cascade, "medium")`: about
0.13 s per blade, of which about 0.11 s is the blade's own passage check; a failed blade
raises `GridGenerationError`, which an optimizer can penalize). Promising and final
geometries take the five-level study (`build_grid_family`, about 0.21 s for all five) for
numerical verification. The explicit `GridSettings(n_passage, n_eta)` stays for the
verification, regression and manufactured cases. `python scripts/analyze_grid_family.py`
runs the study for the NACA 65-(12A10)10 (results in `outputs/grid_family_analysis/`).

## Modules

| Module | Responsibility | Defines |
|---|---|---|
| `config.py` | Every solver input except the geometry: grid size and extensions (in blade chords, one each side by default), operating point (relative inlet Mach number and angles in `_deg`; ABSOLUTE-frame inlet totals `inlet_absolute_total_pressure`, `inlet_absolute_total_temperature`; `angular_speed`; for rotating rows only `stream_surface_radius` and `inlet_absolute_tangential_velocity`), gas model, tolerances, iteration limits (no defaults; `velocity_tolerance` optional, None = `density_tolerance`), density relaxation (default 0.5 = `eos.DENSITY_RELAXATION`, our baseline; 1.0 = Aungier's direct replacement), Kutta relaxation (default 1.0 = `KUTTA_RELAXATION`, our baseline; Aungier gives no value) | `GridSettings`, `FlowConditions`, `IterationSettings`, `SolverConfig`, `KUTTA_RELAXATION` |
| `geometry.py` | Builds a blade from a `CascadeSpec` (NACA 65 / A10 now; DCA and C4 raise `NotImplementedError`), sets it at stagger and pitch, finds the geometric throat, checks every invariant `grid.py` relies on. Optional physical chord as a uniform scale. `surface_continuity` "C1" (validated construction) or "C2" (quintic B-spline curves with a robust, monotone-curvature leading edge, sampled into the same arrays). `thickness_reconstruction` between the table stations: "quintic_natural_sqrt" (the C2 default) or "pchip_sqrt" (legacy, the C1 default). No grid, no flow | `CascadeSpec`, `make_cascade(spec)`, `Cascade`, `CascadeCoordinates`, `BladeProfile`, `naca65_profile`, `validate_cascade`, `StreamSheet` (stub); C2: `ParametricBladeCurve`, `SurfaceCurves`, `naca65_reference_surfaces`, `ReferenceSurface`, `SmoothedSurface`, `LeadingEdgeNose`, `LeadingEdgeFair`, `fair_leading_edge`, `fit_c2_surface`, `C2SurfaceFit`, `leading_edge_fairness`, `fit_quintic_surface` (the previous C2 fit, kept for comparison), `signed_curvature`, `wall_slopes` |
| `grid.py` | The side boundaries of one passage (blade walls, inlet and exit extensions), a check that the smooth passage is open over the whole blade, uniform stations in m with exact far ends, the blade stations as the station lines that cut the blade with the wall nodes on its passage-facing sides, the linear η map to nodes, regions; the exit extension follows the discharge angle. Coordinates only: no slopes or metrics | `PassageBoundaries`, `make_passage_boundaries`, `Grid`, `WallIdentity`, `WallSide`, `make_grid`, `rebuild_exit`, `GridError`; standard grids: `GRID_LEVELS`, `build_grid`, `build_grid_family`, `GridPackage`, `StandardGrid`, `StandardGridInfo`, `CandidateStatistics`, `PassageTransitions`, `passage_transitions`, `validate_standard_grid`, `geometry_identifier`, `GridGenerationError`; `node_index`, `node_theta` (stubs) |
| `metrics.py` | Geometric fields of the transformed equations from a frozen `Grid` (Eqs. 5-1 to 5-3, 5-9): the passage spacing S and its m derivative, the boundary slopes tan β0, tan β1 and the grid-line angle β at every node. Pure geometry, planar (y = rθ), recomputed only when `rebuild_exit` moves the exit | `GridMetrics`, `build_metrics`, `MetricsError` |
| `boundary_conditions.py` | The boundary nodes from the grid topology (masks: lower/upper wall, periodic lower/upper, upstream, downstream; corners in two) and the explicit relations Σ aₖψₖ = rhs of Aungier's conditions: walls ψ = 0 / 1, periodic jump ψ_upper − ψ_lower = 1 (0 for a periodic scalar), ψ_m = 0 at the far stations by second-order one-sided differences; eliminations (ψ₀ = 4/3 ψ₁ − 1/3 ψ₂) and the periodic continuation for the assembly. No matrix, no stencil, no Kutta | `BoundaryConditions`, `build_boundary_conditions`, `Relation`, `Term`, `Elimination`, `BoundaryConditionError`, `PSI_LOWER_WALL`, `PSI_UPPER_WALL`, `PSI_PERIODIC_JUMP` |
| `coefficients.py` | A to E of Eq. 5-21 (Eqs. 5-22 to 5-26) from `GridMetrics`, the current frozen density field ρ(m, η) and the mass flow per passage and unit depth ṁ′ = ṁ/b; the physical chord `c_ref` (metres) or normalized mode (`c_ref=None`). Ends at A to E: no ψ, stencil or matrix | `CoefficientFields` (alias `Coefficients`, the name `discretization.py` imports), `build_coefficients`, `CoefficientError` |
| `discretization.py` | The normalized nine-point difference equation, Eq. 5-44, at the interior nodes from the PDE coefficients A to E and the spacings (Δm = `length_scale` × grid.dm, Δη = grid.deta); stencil weights named by direction (`west` = Ã, `east` = B̃, `south` = C̃, `north` = D̃, `mixed` = Ẽ, `rhs` = Q̃), never A to E; NaN outside the interior. The rows of Eq. 5-44 on the periodic PDE line (i, n_η − 1), as explicit local equations Σ aₖψₖ = rhs. Deferred interfaces: derivatives with the periodic jump, residual, assembly | `InteriorStencil`, `build_interior_stencil`, `PeriodicPDERow`, `PDETerm`, `build_periodic_pde_rows`, `PERIODIC_STANDARD`, `PERIODIC_LE_TRANSITION`, `PERIODIC_TE_TRANSITION`, `DiscretizationError`; stubs `first_derivatives`, `mixed_derivative`, `residual`, `assemble` (the last superseded by `matrix_assembly.py`; kept, its test pins it) |
| `matrix_assembly.py` | The local equations (the Eq. 5-44 rows of `InteriorStencil`, the `PeriodicPDERow`s, the wall and jump `Relation`s) → one square sparse system F Ψ = Q. Numbering k = i n_η + j; the far-station nodes eliminated by their own ψ_m = 0 relations (`Elimination`), substituted into the adjacent rows and rebuilt after the solve; the far stations' jumps implied. Only the substitution algebra and the index change: no difference formula, coefficient, node class from geometry, row scaling or solve | `AssemblyResult`, `assemble_system`, `assemble_from_coefficients`, `assemble_local_equations`, `LocalEquation`, `reconstruct`, `implied_residuals`, `sparsity`, `full_index`, `node_of`, `INTERIOR`, `MatrixAssemblyError` |
| `linear_solver.py` | Solves the assembled sparse system F Ψ = Q by a direct sparse factorization (`spsolve`, SuperLU, `use_umfpack=False`: the same factorization in every environment), the CSR matrix passed as it is, rows never scaled; rebuilds ψ on the (n_m, n_η) grid with `matrix_assembly.reconstruct`; reports ‖r‖∞, ‖r‖₂, ‖r‖₂/‖Q‖₂, the normwise backward error and the implied far-jump residual. Refuses non-square or non-finite input, singular matrices (SuperLU's `MatrixRankWarning` made an error) and non-finite solutions. Knows nothing about the flow | `solve_linear_system`, `LinearSolveResult`, `LinearSolveError`, `solve`, `residual_norm` |
| `eos.py` | The thermodynamic layer of the density loop: a calorically perfect gas, the inlet total state (the entropy reference), the blade-row reference (rothalpy, ω, r), the static state at the relative speed W (h = H′ − W²/2, T, p isentropic, ρ_EOS, a; arrays of the grid's shape), Aungier's initial density, the relaxed density update (λ = 0.5 default, λ = 1 Aungier). No loop and no call to the solver modules | `CaloricallyPerfectGas` (alias `PerfectGas`), `TotalState`, `BladeRowThermoReference`, `blade_row_reference`, `StaticState` (alias `GasState`), `static_state_from_velocity`, `initial_density`, `relax_density`, `DENSITY_RELAXATION`, `AUNGIER_DIRECT_REPLACEMENT`, `ThermodynamicStateError` |
| `flow.py` | ψ → ψ_m, ψ_η → W_m, W_θ, W², W (Eqs. 5-18, 5-19) from the solved ψ, the density it was solved with and the `CoefficientFields` built from that density (ṁ′ and L_ref read there, the density checked against them); ψ_m by `metrics.py`'s m regions (one-sided at the transition stations f − 1, f, l, l + 1, zero on the walls), ψ_η with the periodic continuation; the station mass flow as a diagnostic; the stationary inlet closure (inlet state, ṁ′ = ρ₁ W₁ cos β₁ S₁). The static state, the density update and the Mach number are not here (`eos.py`) | `FlowField` (ψ_m, ψ_η, W_m, W_θ, W2, W), `velocity_components`, `station_mass_flow`, `FlowRecoveryError`, `InletState`, `inlet_state`, `passage_mass_flow`, `InletStateError` |
| `kutta.py` | The Kutta condition at the TE station (the last blade station; SS and PS node from the grid's wall identity) and the exit-angle correction: R_K, ε_K, Aungier's Eqs. 13-57 / 13-59 in the tan β convention, the damped update. Reads a solved field; solves, relaxes and assembles nothing | `trailing_edge_station`, `kutta_residual`, `exit_angle_correction`, `next_exit_angle`, `TrailingEdgeStation`, `KuttaResidual`, `ExitAngleCorrection`, `KuttaError` |
| `convergence.py` | Change norms (max \|new − old\| / max \|old\|, the velocity as a vector), the iteration history, the stop tests: density (density change ≤ `density_tolerance`, velocity change ≤ the velocity tolerance, never on the first pass), Kutta (ε_K ≤ `kutta_tolerance`), coupled (both on one pass) | `History`, `change_norm`, `velocity_change`, `velocity_tolerance`, `density_converged`, `kutta_converged`, `coupled_converged` |
| `solver.py` | Orchestration only, by calls to the verified modules (no equation of its own): the coupled density + Kutta loop (`solve`: grid rebuilt for every exit angle, topology checked, metrics and boundary relations rebuilt, one density update and one angle correction per pass) and the fixed-angle density loop; resident / target / next densities kept apart; one final evaluation at the converged state; per-pass diagnostics (Kutta ones included); transonic, topology and non-convergence stops. Needs ṁ′ and c_ref (not in config) | `solve`, `solve_density_coupled`, `solve_at_exit_angle`, `Solution`, `PassRecord`, `LIMITATION`, `COUPLED_LIMITATION`, `SolverInputError`, `DensityIterationError`, `DensityNotConvergedError`, `CoupledNotConvergedError`, `TransonicFlowError`, `ExitRebuildError` |
| `postprocess.py` | From a finished solution, no recomputation: blade-surface Mach number and static pressure (blade stations only; SS / PS from `grid.wall_identity`), x/c along the chord and m/c, the pressure coefficient (Dunavant's S = (Pt₁ − p)/q₁ default, (p − p₁)/q₁ optional; q₁ = ½ρ₁W₁² of the inlet state), the discharge angle and its far-exit velocity check, a summary, measured points from CSV. No plotting | `blade_surfaces`, `BladeSurfaces`, `surface_mach`, `pressure_coefficient`, `surface_pressure_coefficient`, `SurfaceCoefficient`, `chordwise_coordinate`, `discharge_angle_deg`, `exit_velocity_angle_deg`, `summarize`, `SolutionSummary`, `format_summary`, `read_surface_points`, `SurfacePoints`, `DUNAVANT`, `INLET_STATIC`, `PostprocessError` |
| `plots.py` | Matplotlib figures of a finished solution (imported on call): SS and PS on one axes against x/c, the M = 1 line, optional measured points, the convergence history in the monitor's layout. Returns (fig, ax); no show(), no file | `plot_surface_mach`, `plot_pressure_coefficient`, `plot_convergence`, `plot_mach_contour` |
| `monitor.py` | Observers of a solve: the protocol start / update(PassRecord) / finish; the live window (one figure per solve, Line2D data set in place, throttled redraws, tolerance lines from `IterationSettings`) and the console line. An observer that raises is switched off (MonitorWarning); nothing feeds back | `LiveConvergenceMonitor`, `ConsoleProgress`, `SolveInfo`, `format_record`, `MonitorWarning` |
| `controller.py` | The run layer between a user (the script, later a GUI) and the solver: `CaseSettings` (every input, flat), `build_case` (cascade, config, admissible grid, inlet state, derived pitch, angle of attack, exit angle, mass flow, pass limit), `run_case` (the solver call timed with `perf_counter` and `process_time`; observers' share reported beside it), the console texts, `plot_run` (surface Mach with the runtime and case footer, S, convergence), `export_run` and `run_summary`. No numerics, no thread control | `CaseSettings`, `Case`, `RuntimeStats`, `RunResult`, `CaseError`, `build_case`, `run_case`, `format_case_summary`, `format_inlet_state`, `format_grid_summary`, `format_run_report`, `runtime_annotation`, `case_annotation`, `plot_run`, `export_run`, `run_summary`, `write_surface_csv`, `write_convergence_csv` |

## Planned data flow

```python
spec = CascadeSpec(blade_type="NACA65", solidity=1.2, stagger_deg=30.0, cl0=1.0, t_max_over_c=0.08)
cascade = make_cascade(spec)            # the one way to build a Cascade
solution = solve(cascade, config, c_ref=chord)        # mdot' from config.flow.inlet_mach; live_monitor=True for a window
fig, ax = plots.plot_surface_mach(solution)            # postprocess / plots read the finished solution
```

```text
CascadeSpec ──► make_cascade(spec) ──► Cascade ─┐
SolverConfig: grid, flow, gas, iteration ───────┤
                                                ▼
solve(cascade, config, mass_flow_per_depth, c_ref)
  0  config.flow *_deg angles ───────────────────► radians, the one conversion
  1  grid.make_grid(require_ss_ps=True) ─────────► grid(0) at β_out(0) = exit_angle_guess
  2  flow.inlet_state (M₁, β₁, Pt₁, Tt₁, pitch) ──► inlet state, ṁ′
     eos.initial_density ────────────────────────► ρ(0)
  │
  ├─ coupled loop (one pass per resident (ρ(n), β_out(n))) ──────────────────┐
  │    grid.rebuild_exit(grid(0), β_out(n)); topology and admissibility kept  │
  │    metrics.build_metrics, boundary_conditions.build_boundary_conditions   │
  │    3  coefficients.build_coefficients    → A, B, C, D, E                  │
  │    4  matrix_assembly.assemble_from_coefficients → F Ψ = Q                │
  │    5  linear_solver.solve_linear_system  → Ψ, ψ(i, j)                     │
  │    6  flow.velocity_components           → W_m, W_θ, W                    │
  │    7  eos.static_state_from_velocity     → ρ_target, M_rel < 1            │
  │       eos.relax_density (λ_ρ = 0.5)      → ρ(n+1)                         │
  │    8  kutta.kutta_residual               → R_K, ε_K of W(n)               │
  │       kutta.exit_angle_correction, next_exit_angle (λ_K = 1) → β_out(n+1) │
  │    9  convergence.coupled_converged ? ── no: next pass ───────────────────┘
  │    yes: one evaluation at (ρ(n+1), β_out(n+1)), reported
  ▼
Solution ──► postprocess
```

Imports point one way, from `eos` and `geometry` up to `solver` and `postprocess`, so each
module can be imported and tested without the modules above it.

Responsibilities in the density loop (2026-11-13):

```text
flow.py
    psi -> psi derivatives -> W_m, W_theta, W          (kinematics only)

eos.py
    inlet absolute total state / rothalpy
    + W
    -> h, T, p, rho_target, a
    -> relaxed density update

kutta.py
    W at the TE station -> R_K, eps_K -> Delta tan(beta_out) -> beta_out(n+1)
                                                       (no PDE, density or EOS)

solver.py
    iteration orchestration only (calls)
```

The density pass (implemented): ρ⁽ⁿ⁾ → ψ⁽ⁿ⁾ → W⁽ⁿ⁾ → ρ_target⁽ⁿ⁺¹⁾ → ρ⁽ⁿ⁺¹⁾, with
ρ⁽ⁿ⁺¹⁾ = 0.5 ρ⁽ⁿ⁾ + 0.5 ρ_target⁽ⁿ⁺¹⁾ (`config.IterationSettings.density_relaxation`,
default 0.5; 1.0 is Aungier's direct replacement), and the coefficients of the next pass
built from ρ⁽ⁿ⁺¹⁾. The coupled pass adds the Kutta correction from the same W⁽ⁿ⁾, and the
next pass is built from both ρ⁽ⁿ⁺¹⁾ and β_out⁽ⁿ⁺¹⁾.

## Conventions the stubs assume

- Node arrays are `(n_m, n_eta)`: `i` runs along m, `j` along η. Δm and Δη are constant.
- η = 0 is the wall θ0 and η = 1 the wall θ1. On the blade stations θ0 is the upper surface
  of one blade and θ1 the lower surface of the next (Fig. 5-3); on the extensions they are
  periodic lines one pitch apart.
- Cascade frame: x axial (the m direction), y tangential. The blade turns counterclockwise
  by the stagger about its LE, and the neighbor blade sits at `+periodic_translation =
  (0, pitch)`. The suction surface is the blade's +y (upper, convex) side and faces that
  neighbor. Both surfaces run LE → TE and share the LE and TE points.
- `grid.py` reads lengths from `Cascade.active` only. That is the chord-1 geometry, or the
  same blade scaled to `physical_chord` metres; `Cascade.normalized` always keeps chord 1.
- The grid extensions are counted in true blade chords: `upstream_extension_chords *
  cascade.active.chord` long (1 for chord-1 geometry, `physical_chord` metres otherwise),
  never the axial projection c cos(stagger).
- ∂/∂m holds η fixed, not θ. β is a property of the grid, not the flow angle.
- Angle names carry their unit. A public input in degrees ends in `_deg`
  (`CascadeSpec.stagger_deg`, `FlowConditions.inlet_angle_deg`, `exit_angle_guess_deg`);
  it is converted to radians once, where it enters the calculation. Every computed angle is
  in radians and unsuffixed (`stagger`, `kappa_le`, `grid.exit_angle`, ...). All are
  measured from the axial (meridional) direction, positive toward +y.
- The gas model is whatever object sits in `config.gas`. `eos.static_state_from_velocity`
  uses only its methods `enthalpy`, `temperature`, `isentropic_pressure`, `density` and
  `sound_speed` (and `cp`), so any object with them can replace `CaloricallyPerfectGas`
  (alias `PerfectGas`). Entropy is never absolute: every state has the entropy of the inlet
  total state.
- No case-specific numbers in the package. The cascade and the operating point arrive only
  through `Cascade` and `SolverConfig`.

## Open decisions

Left open on purpose. Each is settled when its module is implemented.

| Decision | Where |
|---|---|
| Signs of W_m and W_θ: settled 2026-11-13 by Eqs. 5-18, 5-19 with ψ = 0 / 1 on η = 0 / 1: W_m > 0 from inlet to exit, W_θ > 0 toward +y, a uniform flow at α has W_θ/W_m = tan α (α from m toward +y, as in `geometry.py`; tested) | `flow.py` |
| Numbering of the unknowns: settled 2026-11-13 as k = i n_η + j (`matrix_assembly.full_index`, `node_of`). Open: `grid.node_index` stays a stub (`grid.py` is frozen and its test pins the `NotImplementedError`); remove it or delegate | `grid.py` |
| The LE and TE grid topology. **2026-11-13: the geometry-driven standard grids (`grid.py` section 9) make admissibility a property of the geometry**: the lattice puts cell faces at the middles of the LE and TE windows, so every level of a blade whose windows are narrower than its dm (by 0.2 dm) is admissible; the explicit n_passage path below is kept for the verification cases. Decided 2026-11-12: the meridional grid stays uniform, and only admissible n_passage are used (first and last blade station one suction + one pressure wall node; `grid.py` section 8). Still open: (1) `make_grid`'s default accepts inadmissible grids, because the tests and fingerprints of `metrics.py`, `coefficients.py`, `discretization.py` and `boundary_conditions.py` are built on them (33°, n_passage 101 and 401 are inadmissible): make `require_ss_ps=True` the default once they are re-baselined on admissible grids; (2) the side lines still meet the walls at the nose tip and tail end with a C0 jump (anchored at the LE / TE point), on admissible grids too; (3) the last wall station can lie next to the tail's vertical tangent on admissible grids (the TE window is tiny); (4) the transition stencils, to be revisited on admissible grids only | `grid.py`, then `discretization.py` |
| A right-hand side of Eq. 5-21 for rotating rows on a radius-changing stream sheet (zero for the planar cascade, not implemented) | `coefficients.py` |
| Linear solve: decided 2026-11-13, a direct sparse solve (`spsolve`, SuperLU, its own pivoting; no row scaling). Open: a verification solver reproducing Aungier's banded forward / backward elimination, to compare Ψ_Aungier with Ψ_scipy on the same F and Q; an iterative method only if grids outgrow the direct solve | `linear_solver.py` |
| Kutta condition: settled 2026-11-13: Aungier 2003 Eq. 5-34, equal speeds W_SS = W_PS at the last blade station (`last_blade_i`, never l + 1 or the tail end), R_K = W_SS − W_PS and ε_K = \|R_K\| / (½(W_SS + W_PS)); Eq. 13-58's averaged speed not used. Open: the station's sensitivity to grid refinement (the deleted b2b attempt saw the last-node angle drift), and an extrapolated or fixed-station alternative, to be measured in the validation stage | `kutta.py` |
| Size of the Kutta step: Aungier's undamped correction (λ_K = 1.0, the baseline) removes 12.6 % of R_K per pass on the verification case, so the coupled loop takes about 138 passes (the density loop alone 27; 135 and 28 before the 2026-11-13 ψ_m fix, the λ_K counts below were measured before it). Eq. 13-57 puts the whole TE mismatch on the last element, which asks for about 8 × less than a Newton step on the solver's own slope (dR_K/d tan β_out ≈ −427 m/s at frozen density); its gain is proportional to Δm, so finer grids take smaller steps unless that slope grows with them (not measured yet). Damping only slows it (λ_K = 0.5: 280 passes, 0.25: 568). Over-relaxation (λ_K > 1, refused now), a secant or a Newton step would be ours, not Aungier's: to be decided with the refinement study | `kutta.py`, `config.py` |
| Mass flow from the inlet Mach number: settled 2026-11-13. `flow.inlet_state` closes the stationary inlet with `eos.py`'s gas (W₁ the root of W/a(W) = M₁) and ṁ′ = ρ₁ W₁ cos β₁ S₁; `solve` derives it. `mass_flow_per_depth` remains a verification override: alone with `inlet_mach=None`, or checked to 1e-12 against the inlet Mach number. M₁ = 0 is refused by the solver (zero flow, as `coefficients.py`) | `flow.py`, `solver.py` |
| A velocity tolerance of its own: settled 2026-11-13, `IterationSettings.velocity_tolerance`, optional; None (the default) applies `density_tolerance` to the velocity change, the behaviour before (no number chosen). The Kutta tolerance (`kutta_tolerance`, on ε_K) was already its own field; no tolerance has a default | `config.py`, `convergence.py` |
| Rotating rows in the density loop (refused now; the PDE and `eos.py` would carry ω at constant radius) | `solver.py` |
| Reference of the pressure coefficient: settled 2026-11-13 as the source's: Dunavant et al. (1955) S = (P₁ − p_l)/q₁, q₁ = ½ρ₁V₁² (their list of symbols; their Fig. 10 plots S against percent chord), the default of `postprocess.pressure_coefficient`; (p − p₁)/q₁ optional. Reference: the specified inlet state, not the computed far-inlet station | `postprocess.py` |
| The sonic node at the first blade station (M₁ ≥ 0.6175 on 61 × 41; the Fig. 5-6 condition M₁ = 0.65 stopped or diverged): **FIXED 2026-11-13** in `flow.py` (ψ_m by `metrics.py`'s m regions, one-sided at f − 1, f, l, l + 1; nothing else changed). With the sonic stop on, M₁ 0.60 to 0.68 converge subsonically on 61 × 41 with the field peak on the suction wall (0.906 at M₁ 0.65), and 0.65 on 43 × 29 and 87 × 59 too; the first stop is at M₁ 0.70, pass 11, on the suction wall (node (f + 9, 0)), a genuine wall-sonic case. The production runs reproduce the diagnostic what-if V bit for bit (`scripts/verify_le_velocity_fix.py`). The diagnosis, for the record. Cause: `flow.py`'s velocity recovery takes ψ_m at the interior nodes of the first blade station by a central difference across the leading edge (stations f − 1, f + 1) while `metrics.py`'s tan β there is the blade-side slope; the two belong to different η-lines (slopes 2.50 against 1.86 at the critical node), W_θ comes out about 50 m/s too large and the Mach number 0.10 to 0.17 too high over η ≈ 0.03 to 0.4 of that one station (recomputed on the production field; 0.22 against the converged variant at the critical node); the density feedback amplifies it. A diagnostic variant with same-side ψ_m at f − 1, f, l, l + 1 (nothing else changed) converges M₁ = 0.65 subsonically on three grids (peak 0.906 to 0.912 on the suction wall). Decision log, 2026-11-13 (the diagnosis and the fix); `scripts/diagnose_mach060_vs_065.py`, `scripts/verify_le_velocity_fix.py` | `flow.py` |
| D and E at the transition stations: production B_m, C_m cross the leading edge there and D has the wrong sign against same-side differences at f − 1 and f (−3556 against +10204 at the critical node; f + 1 agrees). Not the cause of the sonic node (the velocity variant alone removes it), but the naive same-side replacement is no fix either: it diverges on pass 0 or 1 at a wall node of the first or last blade station. **PARKED 2026-11-13:** the ψ_m fix of `flow.py` leaves D and E exactly as they were (bit for bit); needs its own study | `coefficients.py`, `discretization.py` |
| The discharge angle on grids. **2026-11-13, the systematic five-level family (grid.py section 9, M₁ 0.65, stop on):** β_out 25.478°, 24.372°, 23.570°, 22.683°, 23.282° from very_coarse to very_fine (dm/c 0.0400 … 0.0100): four steady drops of 0.80 to 1.11° per level, then a rise of 0.60°; NOT CONVERGED, so grid selection alone is not the explanation. The Kutta station l lies (dm − w_TE)/2 ahead of the TE on every level, so it samples x/c 0.987 → 0.9996 as the grid is refined and enters the TE-circle closure at the finest levels (the suction-wall tan β at l flips from +0.12 to −0.40, cos β of both walls falls to 0.93, the TE speed jumps back up 133.3 → 139.0 m/s); against (m_TE − m_l)/c the old grids lie within 0.33° of the family's curve (87 × 59 and very_fine have nearly the same dm, 0.0097 and 0.0100 c, but Kutta stations 0.0042 and 0.0050 c from the TE, and differ by 0.38°, as that curve's rising branch has it): the old scatter is largely this effect. Next suspects, not changed here: the Kutta station placement (Eq. 5-34 at the last blade station) and the transition D/E (growing as about dm^−1.5). Earlier record: the discharge angle changes by about 1° between admissible grids without a trend (M₁ 0.60: 23.75°, 22.79°, 23.92° on 43 × 29, 61 × 41, 87 × 59 before the 2026-11-13 ψ_m fix; after it 23.78°, 22.76°, 23.74° (the what-if V runs, which production reproduces bit for bit), and at M₁ 0.65 (production) 23.72°, 22.70°, 23.66°: 61 × 41 still the outlier), and by 0.39° with the thickness variant (kovach_sandercock against emery_base). **STILL OPEN** after the fix. To be understood before any comparison with 24.4° / 25.0° | `kutta.py`, `grid.py` |
| Kutta grid-convergence study (2026-11-14, branch `kutta-study`; `scripts/kutta_study.py`, report `outputs/kutta_study/report.md`, which the script regenerates since `outputs/` is untracked). The study options are `IterationSettings.kutta_eval`, `kutta_method` and an α > 1 for the Aungier methods; their defaults are the production rule. Outcome: case (b). β_out stays grid-dependent with the Kutta condition at a fixed m* (3 % c: \|fine − very_fine\| 0.96°) and with Eq. 13-58 (0.54°). The production correction already is Eqs. 13-57/13-59 in our frame. The Stokes balance fails on the two elements ending at the Kutta station. α ≥ 3 converges every level in 54–59 passes. **OPEN:** the D/E transition terms are the next suspect (not changed) | `kutta.py`, `coefficients.py` |
| Digitizing the Dunavant data: **settled 2026-11-13.** The user digitized Aungier's Fig. 5-6 (Dunavant's measured surface Mach numbers at M₁ 0.65 and Aungier's potential-flow curve, suction and pressure side) with WebPlotDigitizer: `validation/naca65_12a10_10/Dunavant_Aungier_Fix.csv`, the trusted validation data; `scripts/compare_dunavant_aungier.py` plots against it. Open: `scripts/validate_naca65_12a10_10.py` still reads measured points only through `--experiment` (a CSV of surface, x_c, mach or S) and does not use this file | `scripts/validate_naca65_12a10_10.py` |

## Decision log

Choices that are ours, not the book's, or where the spec and the data disagreed.

- **`geometry.py` is self-contained.** It imports only the standard library, NumPy and
  SciPy; the Table 4-1 data and every construction step are in the file. Two tests keep it
  that way: one reads its imports, the other builds a cascade with newblade's `naca65` and `b2b`
  packages blocked. The NACA 65 construction is the method of the repo's earlier naca65 toolchain,
  rewritten into this file. One-off checks outside the repository: at nominal thickness it
  reproduces naca65's frozen T10 reference contour to 7e-18 (o/s, angles and area
  identical) and naca65's generator on 40 blades to 3.5e-18. Nothing imports naca65.
- **Nominal thickness.** For NACA 65, `t_max_over_c` is the nominal ratio of the designation
  (0.10 for "...10"), and the corrected table is scaled by `t / 0.10`. So 0.10 keeps
  Table 4-1 exactly at every station, with its largest ordinate 0.09992 at station 0.40; the
  table is not renormalized to peak at 0.10000. The true maximum is
  `metadata["max_thickness_over_c"]` (0.9992 t for the legacy PCHIP, 0.99961 t at station
  0.4098 for the quintic: see the thickness reconstruction below), and the thickness check of
  `validate_cascade` compares against it. An earlier version scaled by `t / 0.09992`; it was
  changed on review.
- **`CascadeSpec` is the one geometry input.** `make_cascade(spec)` is the only way to build
  a `Cascade`, and the modelling defaults (A10, Kovach-Sandercock, no physical chord,
  101 camber stations) live in `CascadeSpec` only; `naca65_profile` has none. `Cascade`
  keeps its spec and no copy of any input, so an input is read from one place:
  `cascade.spec`. A future frozen case will pair a `CascadeSpec` with a `SolverConfig`.
- **Angle names carry their unit**, `_deg` for public degree inputs and none for radians,
  so `inlet_angle` became `inlet_angle_deg` and `exit_angle_guess` became
  `exit_angle_guess_deg`. The grid, Kutta and history stubs now take radians.
- **Extensions in true blade chords.** `upstream_extension_chords` and
  `downstream_extension_chords` (default 1.0) replace `inlet_length` and `exit_length`,
  which were in meridional chords. They set the m range exactly: `m_in = m_le - L_up`,
  `m_out = m_te + L_down` (grid entries below).
- **No `throat_location`.** It held the blade-frame x/c of each throat end and two fixed
  surface names, all derivable from the end points, and no solver module used it. The
  throat is located by `throat_point_reference` and `throat_point_neighbor` only.
- **Throat finder with many starts.** A single start refined by undamped Newton (the earlier
  toolchain's method) failed over a 640-cascade sweep: it did not converge on 11 Kovach
  searches and stopped in a non-global minimum on 9 others, 8 by up to 1e-4 c in a small
  dent of the lower nose fair and 1 by 0.021 c, held at a spline end. `closest_points`
  starts a line-searched projected Newton refinement from every local minimum of the
  vertex distance and keeps the best.
- **Kovach TE circle tolerance.** The taper is tangent to the 0.008 c circle along the
  straightened camberline only, so on the curved blade the pressure surface dips up to
  3.1e-6 c (0.04 % of R_TE) inside the circle before its radial fair. The check allows 0.1 %.
- **Grid stage 1 is its own object.** `make_passage_boundaries(cascade, settings,
  inlet_angle, exit_angle=None)` returns `PassageBoundaries`: the blade walls and both
  extensions as curves, and no field for anything not built. The angles are in radians,
  like every grid input; the caller converts `inlet_angle_deg` once.
- **Passage walls straight from the Cascade.** Lower (η = 0) is `active.suction_surface`,
  upper (η = 1) is `active.pressure_surface + periodic_translation`, both in the LE → TE
  order of `geometry.py`, nothing sorted, re-split or removed. The blade spans `m_le =
  leading_edge_point[0]` to `m_te = trailing_edge_point[0]`, never min/max of x.
- **The walls are not single-valued in m at the edges, and stage 1 leaves them so.** The
  rounded nose leaves the LE point upstream on the side the LE circle center lies on: for
  the 65-(12A10)10 at 33° stagger the suction side runs 0.0046 c ahead of `m_le` (68 of its
  208 points lie upstream of it) and comes back to `m_le` 0.028 c above the LE point. A thin nose can fold
  back a second time, 5e-6 to 6e-4 c deep and within 0.0033 c of `m_le` (12 cases of a
  sweep over cl0 0 to 1.6, t/c 0.04 to 0.10 and stagger -30° to 60°, all at t/c = 0.04).
  A Kovach TE can pass up to 0.005 c beyond `m_te` before closing on the TE point
  (1.3e-5 c on the pressure side of the case above). `PassageBoundaries` keeps them so;
  `make_grid` cuts them with the station lines and takes the passage-facing crossings
  (below).
- **Inlet extension.** `m_in = m_le - upstream_extension_chords * active.chord`. β, not
  tan β, is linear in m from the inlet angle to `kappa_le` (book p. 86). The lower line is
  the exact integral of tan β back from the LE point, through the mean of tan over an
  angle interval written so that equal angles give the straight line without 0/0; the
  upper line is the lower one plus `periodic_translation`.
- **Extension samples.** `upstream_*` and `downstream_*` sample the closed-form side lines
  uniformly from the far station to the junction, as many cells as the length holds
  passage cells, rounded, so both ends are exact (37 cells of 0.02703 c each side for the
  case above). They describe the curves for checks and plots; they are not grid stations.
  The angle weights come from the sample index, so a physical chord scales the lengths and
  leaves β bit for bit.
- **Exit extension and the initial exit angle.** `m_out = m_te +
  downstream_extension_chords * active.chord`; β linear in m from `kappa_te` at the TE to
  the exit angle at `m_out`, the line integrated forward from the exact TE point, the upper
  one the lower plus `periodic_translation`. `exit_angle=None` takes `kappa_te`: a starting
  value for this solver's Kutta loop, not a rule of Aungier's. β is interpolated from the
  nearer end, so both ends are exact and equal angles give exactly the straight tangent,
  bit for bit.
- **`rebuild_exit(grid_or_boundaries, exit_angle)`.** Takes a `Grid` or a
  `PassageBoundaries` and returns the same type with only the exit side recomputed from the
  TE junction: `exit_angle`, `downstream_beta`, the downstream lines and, for a `Grid`, the
  exit-station nodes. Stations, η, inlet, walls, landmarks, chord and pitch are carried
  over unchanged, and the result equals a fresh build at that angle bit for bit, so a
  sequence of angles that returns to `kappa_te` returns to the starting grid exactly.
  `grid.py` holds the current exit angle, never how it was found.
- **One Δm over the whole grid, exact far ends.** `m = linspace(m_in, m_out, n_cells + 1)`
  with exact extension lengths; only the whole number `n_cells` is free, so `n_passage`
  becomes a target, `dm_target = (m_te - m_le) / n_passage`, and neither edge is a station
  in general. This settles the spacing question stage 1 left open.
- **No prescribed leading-edge shift (2026-11-07).** Aungier asks only that the first and
  last grid points on the blade surfaces lie inside the blade passages; he gives no
  distance from the LE. An earlier version imposed `leading_edge_shift_chords = 0.01` (a
  development value) and required the whole nose plus that shift to fit in the cell before
  the first blade station. That rule refused fine grids: at 101 passage cells Δm =
  0.0083 c cannot hold 0.01 c plus the 0.0046 c nose. The setting and the rule are gone;
  `GridSettings` has no leading- or trailing-edge field. (The version before that put station
  37 of the case above, 0.0003 c behind the LE, on the inlet side lines, so that its
  periodic η line ran through both noses; the rule below makes that station a blade
  station.)
- **Blade-passage topology from the station lines.** Every station line x = m[i] is cut
  with the reference blade's closed outline (suction surface TE → LE, then pressure surface
  LE → TE) and its periodic copies. A line that meets no blade is a periodic station; a line
  that cuts the blade is a blade station. The blade's extent in m is one interval, so the
  regions are one inlet run, one blade run and one exit run, and the first and last blade
  stations are the first and last lines that cut the blade, within one Δm of its most
  upstream and most downstream points. `Grid` checks this itself and refuses any other
  split, and it refuses nodes that leave no positive width or put both wall nodes of a
  blade station on one point of the blade. The chord-axis LE and TE are landmarks only: `actual_le_offset = m_first_blade -
  m_le` and `actual_te_offset = m_last_blade - m_te` are diagnostics, and the LE offset tends to
  the nose's reach (-0.0046 c for the case above) with refinement, not to zero, and changes
  sign with the station positions. The classification never looks at the η nodes, so n_eta
  cannot change it. A cut thinner than 1e-12 c is a touch, not a cut: a station on the
  blade's most upstream or downstream point is periodic. That matters for round numbers: at
  60° stagger the blade is 0.5 c long in m, and with one-chord extensions and 17 passage
  cells a station lands on the sharp emery_base TE within an ulp; the outline's round-off
  gave its cut 2.2e-16 c, which made it a blade station with both wall nodes on the TE point.
  A first sweep with separate checking code found 192 such misclassified stations (and the
  chord-1 and physical twins disagreeing on 15 of them); with the touch rule there are none.
  Stations a few ulps either side of a nose tip, a Kovach tail or a sharp TE hold at most
  2.2e-13 c of metal and are periodic (tested), and the two stations review found 1.5 and
  2 ulps ahead of a sharp TE are periodic now, with the last blade station's wall nodes
  3.6e-4 c apart.
- **Geometry sweep.** 504 cascades (stagger -30° to 60°, cl0 0.4 to 1.6, t/c 0.04 to 0.10,
  solidity 0.8 to 1.5, both TE variants) at 17 × 11, 31 × 21, 61 × 41 and 101 × 101, each with
  inlet 30° and exit κ_te, and with inlet -10° and exit κ_te + 8°, plus a 0.05 m twin of every
  cascade at 31 × 21: 4536 grids, none refused, none breaking a hard invariant, the twins
  with the same topology. Straight m-links through metal (diagnostic): 718 across the LE
  interval in 252 grids, 24 across the TE interval in 24, and 4 between neighboring blade
  stations at the nose, all in rows next to a wall.
- **Passage-facing walls.** At a blade station the lower wall node is the top of the
  reference blade's cut and the upper wall node the bottom of that cut moved one pitch, the
  +pitch neighbor's passage-facing side; the η line between them holds no metal (the other
  copies lie a pitch further out; a cut spanning a pitch or more is a `GridError`). Away from
  the edges that is the suction surface and the neighbor's pressure surface. Where the
  outline folds in m the passage-facing side continues around it: a station in the nose
  ahead of `m_le` has its upper wall node on the neighbor's nose underside, which geometry.py
  files under the suction polyline because its LE point is the chord-axis end, not a corner;
  a station in a Kovach tail behind `m_te` cuts the overshooting side twice, which for
  positive stagger is the pressure side, with the lower wall node on its return to the TE
  point, and for strongly negative stagger the suction side (0.004 c at -30°), with the
  upper wall node on the neighbor's suction-side return; inside a thin nose's second fold
  (4 crossings) the outermost crossing faces the passage. Each blade surface is one
  streamline whatever the polyline label, so the wall condition is unchanged. This replaces
  the earlier rule that sampled each wall only where it was single-valued in m and refused
  folds.
- **Wall identity (2026-11-07).** At 101 passage cells the EXAMPLE's plot showed the
  upper wall "crossing the LE onto the suction side": station 120, 0.00398 c ahead of the
  LE point, crosses only the suction surface (twice), because geometry.py starts the
  pressure surface at its chord-axis LE point while the nose tip lies 0.00458 c further
  upstream, on the suction surface. Its only metal-free η line ends on the neighbor's nose
  underside, which is suction surface. That is not a sorting fault and cannot be avoided
  by branch tracking: whenever Δm is shorter than the nose reach a station falls there (the
  EXAMPLE from 172 cells up; 311 of the resolutions 17 to 401), and a Kovach tail does the
  same at the TE with the other wall. Rule (decided 2026-11-07): the walls are the two
  passage-facing contours, the reference blade's upper contour and the neighbor's lower
  contour from the nose tip to the tail end, and their identity is tracked.
  `Grid.wall_identity` (a `WallIdentity` of two `WallSide`s) gives for every wall node its
  named surface, its place on it (0 at the LE point, 1 at the TE point) and its place
  along the contour from the nose tip (the outline's chord-length parameter, within about
  1 % of arc length), and counts crossings hidden behind it in a fold.
  `Grid` refuses a grid whose wall node is off its contour, whose wall runs backward along
  its contour, or whose wall changes its named surface anywhere but at the LE or TE point.
  For the EXAMPLE from 21 to 201 cells: no backward or skipped step, no hidden crossing,
  the lower wall always on the suction surface, the upper wall on the pressure surface
  except at the one station ahead of the LE point at 21, 101, 141 and 201 cells; the name
  is a function of m alone. The grid plot colours wall nodes by named surface (o lower,
  s upper wall) and names any node on the other surface.
- **Wall points on a parametric PCHIP of the outline.** x(t), y(t) through the outline
  vertices, t the cumulative chord length: one cubic per polyline segment, monotone in x and
  y on each segment, C1 through the LE point, never beyond the vertices in m. The polyline
  segment the cut selects therefore holds exactly one crossing of the smooth curve, found by
  bisection, and the smooth curve has the polyline's extent in m, so the classification and
  the wall points agree. The nodes lie within 3.5e-5 c (normal distance) of the polyline over
  the sweep, on the very segment their crossing selected. Against the same blade at 1601
  camber stations they are within 8.3e-6 c for the case above, nose and tail included, and
  within 2.6e-5 c over the sweep, largest where the nose stations end and where the Kovach
  taper starts: the thickness slope breaks there, at station 0.60, by 0.08° at t/c 0.10 and
  0.67° at t/c 0.04, a corner the C1 walls round off (and wall slopes in metrics.py will
  see). That error belongs to the Cascade's sampling, not to the grid: at a fixed m grid
  101, 201, 401 and 801 camber stations give 8.1e-6, 2.2e-6, 4.7e-7 and 1.4e-7 c. A cubic
  spline in chord length (geometry.py's throat representation) can overshoot in m at a fold
  or tail, which would break the one-crossing property; PCHIP does not.
- **Hard invariants and a diagnostic.** Hard, and tested: no node in blade metal; every
  station's η line valid as a whole (periodic: in fluid end to end; blade: wall to wall,
  metal only at the ends); contiguous regions; two distinct passage-facing wall nodes at
  every blade station; positive width; uniform m and η; exact periodicity; topology
  independent of n_eta; `rebuild_exit` changing only exit-station nodes. "Metal" is the
  blade inside the smooth walls the wall nodes lie on; against the straight-segment
  polyline, 3.5e-5 c away at most, the wall-to-wall lines meet it in slivers of that size at
  their ends. Touches of zero length are allowed (a periodic line through a nose tip, a
  Kovach tail or a sharp TE point, a blade line through the tip of a thin nose's second
  fold). Not a grid property: the straight segment from node (i, j) to (i + 1, j). Across
  the two transition intervals (last inlet → first blade station, last blade → first exit
  station) the boundary rows change from periodic side line to wall and the nose or tail
  lies between the stations, so for the rows near the walls that segment can cross the nose,
  the more so the finer η. For the case above: no crossing at 21 × 21 to 61 × 61 or at
  101 × 101; one link at 81 × 81 (0.0155 c of metal), 61 × 101 (0.0092 c) and 61 × 161
  (0.024 c), all across the LE interval; over the sweeps up to 0.18 c of one link at the LE,
  in rows up to η 0.33, and 0.04 c at the TE. The four nodes around an interval are not a
  body-fitted quadrilateral and are not meant as one. These links are NOT finite-volume
  cell edges. Their use by future finite-difference stencils must be reviewed when the
  m-derivative operators and the LE/TE one-sided boundary stencils are implemented
  (metrics.py, discretization.py). This is a deferred limitation, kept as a diagnostic, not
  a resolved or a harmless one.
- **Continuous passage check (2026-11-07).** geometry.py refuses blades whose
  straight-segment outlines cross; the grid stands on the smooth walls, up to 3.5e-5 c away,
  so within about 0.01 % of the largest solidity geometry.py accepts the two can disagree
  about a passage a few 1e-6 to 1e-5 c wide. The smooth walls of 50° stagger, cl0 0.4,
  t/c 0.10, emery_base at solidity 6.4246234 cross by up to 1.1e-5 c over 0.0039 c of m
  while the outline stays open by 3.6e-6 c; checked at the stations only, that cascade was
  refused at 41, 101, 141 and 200 passage cells and built at 17, 21, 31, 61 and 81, with
  walls crossing between two stations. Now `make_grid` checks the passage over the whole
  blade before it places a station. The smooth width is S(m) = pitch − (top − bottom) of
  the smooth cut, and the grid needs S(m) > TOUCH = 1e-12 chord at every m between the
  blade's ends. The m range is cut at every vertex m of the outline; on each piece the top
  and the bottom stay on one PCHIP piece each, monotone in m (tested on six blades), so the
  two end values bound S over the piece from below. Pieces the bound clears are open,
  rigorously. Each piece it does not clear (only near a narrow waist) is cut into 16 cells,
  the bound clears most of them, and each cell left is searched to round-off by zooming
  (17 points, keep the two cells around the smallest, 16 rounds); that step assumes at most
  one dip of the width inside a cell, a sixteenth of a vertex spacing. A width not above
  TOUCH is a `GridError` naming the width, its m, the blade (from `cascade.metadata`), the
  stagger and the solidity. The answer depends on the blade and the pitch only. The case
  above is now refused, word for word, at every n_passage from 17 to 200, every n_eta and
  either upstream length; with its pitch set to its widest smooth cut D = 0.1556625 c plus
  1e-9 c or 2 TOUCH it builds, and with 0.5 TOUCH or −1e-9 c it is refused, at coarse and
  fine resolution alike, and a width of exactly TOUCH counts as closed. TOUCH sits four
  orders above the round-off of S and far below any open passage seen (2.0e-6 c at the
  largest solidity geometry.py accepts for 50°, cl0 1.6, t/c 0.10, which builds at every
  resolution). The search agrees with brute force at 400 001 points to 6e-16 c where the
  walls cross and is 1.4e-13 c sharper at a waist of 2.3e-6 c; it costs about 9 ms per
  `make_grid` for an ordinary passage and 60 ms at the overlap limit. The same piece ends
  settle the straight-segment cut exactly (it spans less than a pitch: linear on a piece),
  which replaces the old check at the stations. In a passage that stays open but narrows to
  a few 1e-6 c, nodes between the walls can lie up to 2e-5 c (vertically) inside the
  straight-segment outline, never inside the smooth walls (tested). Near such a closure
  geometry.py's own throat is round-off small (1e-16 c) and its check refuses only an exact
  0.0, so `make_cascade` accepts or refuses pitches there by round-off; the grid's decision
  does not depend on that.
- **Refinement behaviour (geometric only; no flow quantity exists yet).** Case above, from
  21 to 141 passage cells: the topology holds at every level; the gap from the nose tip to
  the first blade station and from the last blade station to the tail stay within one Δm
  but do not shrink monotonically, because where the uniform stations fall on the nose
  depends on n_cells; the smallest pitchwise width S converges from above, within
  1.47 Δm²/8, to the smallest vertical gap, 0.877937 c on these walls (0.877940 c on the
  1601-station blade), which is not the throat (0.721525 c): S runs along y, the throat
  across the passage; the closest pair of lower and upper wall nodes converges to the
  throat with an error between 0 and about 3.7 Δm², set by where the stations fall against
  the throat's two ends (3.7e-3 c at 21 cells, 9.4e-5 c at 81, 7.8e-6 c at 101, 3.9e-5 c at
  141), with no floor: the smooth walls' throat and geometry.py's cubic-spline throat
  differ by 9e-8 c; periodicity holds to 1.1e-16 c at every level; changing n_eta from 11
  to 161 at a fixed m grid changes no station, no blade index and no boundary node.
- **No trailing-edge rule needed.** The exit stations are the lines behind the blade's most
  downstream point, so a Kovach tail that passes `m_te` (by up to 0.004 c in the sweep) is
  either cut by the last blade station or lies wholly ahead of the first exit station; no
  exit η line can cross it. This closes the open TE item of the previous version, where 19
  of 1258 sweep grids had an exit station through the tail.
- **`Grid` holds coordinates only.** Fields: `boundaries`, `m`, `eta`, `nodes` (n_m, n_η, 2)
  and `first_blade_i`, `last_blade_i`. `lower_boundary`, `upper_boundary` (`nodes[:, 0]`,
  `nodes[:, -1]`), `dm`, `deta`, `last_inlet_i`, `first_exit_i`, `m_first_blade`,
  `m_last_blade`, `actual_le_offset`, `actual_te_offset`, `region`, `wall_identity` (derived
  from the boundaries and the stations when the Grid is made, and checked) and the
  landmarks, lengths and angles are read-only properties, so nothing is stored twice. No
  slope, metric or flow field. Nodes are `x = m`, `y = (1 - η) y_lower + η y_upper`, the linear map of Eq. 5-2
  written so that η = 0 and 1 return the walls exactly.
- **Class name `Cascade`.** The output object keeps the name that `grid.py` and `solver.py`
  already import, so neither changed (`metrics.py` takes only the `Grid`).
- **metrics.py: pure geometry on the frozen grid (2026-11-07).** `build_metrics(grid)`
  returns `GridMetrics`: the `Grid` by reference, S, dS/dm, tan β0 and tan β1 per station,
  tan β, β, sin β and cos β per node, read-only, in the Grid's length unit. No flow, gas,
  density, stream-sheet, coefficient or ξ field.
  - Metrics are pure geometry; the flat cascade's y replaces rθ, so S = y1 − y0 (Eq. 5-9)
    and tan β = (∂y/∂m)_η, with no radius used or faked. The stub's r, b, φ and
    `compute_metrics(grid, sheet)` went with this; `GridMetrics` replaces its `Metrics`, and
    the three stubs that used that name changed only the name (their import, annotations and
    one docstring line in solver.py).
  - S is the passage spacing at constant m, not the throat: min S = 0.8779 c against a
    throat of 0.7215 c here. S is also det ∂(m, y)/∂(m, η); S at or below round-off is a
    `MetricsError`.
  - β is the grid-line angle, not in general the flow angle (only the far stations are
    built along the flow); tan β, not β, is linear in η (Eq. 5-3).
  - Wall derivatives never cross the LE/TE topology transitions: d/dm is second order
    inside each of the inlet side lines, the blade walls and the exit side lines separately,
    each run needing three stations (else `MetricsError`, no first-order fallback).
  - Measured accuracy, limits and the minimum cos β are in metrics.py's docstring; in short:
    side lines second order; blade walls converge, between first and second order, to the
    C1 smooth walls, which sit up to 4.0e-3 / 4.6e-3 in slope from the 1601-station blade
    (an n_camber_stations floor at the thickness-table knots); slopes at the first few and
    last two blade stations carry O(1) errors at a rounded edge, where the extremes sit
    (EXAMPLE min cos β 0.30 to 0.45, at the first blade station; it falls with Δm at a
    rounded nose, STEEP 0.069 at 401 cells). A 2466-grid sweep found no invariant broken;
    cos β ≤ √ε, the singular threshold, is far off for any blade (the lowest seen, 0.007, is
    an exit side line built for an exit angle of 89.6°).
- **Wall-slope audit (2026-11-08), metrics.py unchanged.** `scripts/plot_metrics_accuracy.py`
  compares every blade-wall slope with the exact tangent of the smooth wall it lies on (the
  Grid's own contour, n_passage 31 to 401). The stencils are second order with the textbook
  constants on manufactured functions. Nodes 0.05 c or more from the ends converge at
  observed order 1.0 to 1.6 to the C1 smooth walls; the local order is about 2 between the
  outline's knots. The large angles at a rounded LE or TE are geometry: at the steepest node
  the wall is as steep as the difference, or steeper, and the first node's difference follows
  the square-root tip model g(φ), φ = (distance to the tip)/Δm. So the end-node error depends
  on φ, does not converge, and understates the angle there. Two outline features of the
  frozen geometry show in the errors: an S-bend on the pressure-side LE fair (the angle falls
  back 4.3° over m/c 0.0069 to 0.0085) and the curvature jump where the Kovach TE fair leaves
  the surface.
- **Surface continuity (2026-11-11), geometry.py reopened, grid.py and metrics.py unchanged.**
  `CascadeSpec.surface_continuity` (default "C1", or "C2"; exactly those strings, like every
  other name input) is the one switch. "C1" is the validated construction: checked bit for
  bit against the module before the change on 200 cascades and every field, metadata included,
  and locked by `test_c1_baseline_fingerprint`. "C2" is the same blade as clamped quintic
  B-splines in the C1 arc length, built from the analytic C1 surface
  (`naca65_reference_surfaces`: the C1 construction as chains of pieces with exact
  derivatives; every C1 vertex lies on it to 1e-12 c). The construction is in the module
  docstring; in short, every C1 junction where the curvature jumps or the tangent breaks
  becomes a monotone curvature transition of half-width a, set by a shift of at most
  1e-6 c, and nothing else is changed.
  - Why not a plain least-squares spline: it rings at every curvature jump (±12 % at the
    Kovach junction, and a spurious inflection on the suction side just ahead of it). Neither
    a penalty on dκ/ds (it smooths the legitimate LE fair: 2.7e-5 c at the S-bend) nor one on
    the fifth-derivative jumps (it moves the error onto the LE circle) removes it: a linear
    smoother with a sharp cutoff rings. A positive kernel in the curvature cannot overshoot,
    so the transitions are built that way and the spline only represents them.
  - Measured for the 65-(12A10)10, Kovach, 33° (`scripts/plot_geometry_continuity.py`): max
    normal deviation 1.001e-6 c on both surfaces (at the station 0.60 corner), RMS 1.6e-7 c;
    area −6.2e-6 %; throat −1.6e-8 c on the arrays (−2.9e-8 c on the curves; the array
    throat of either representation sits 4.5e-7 c below the throat of its own curve, from
    the cubic spline through 101 stations); surface lengths −5e-8 c; max thickness +2.4e-7 c.
    Kovach TE junction: C1 −9.4 → −128.4 /c (suction) and −11.1 → +121.0 /c (pressure);
    C2 continuous, rising monotonically over ±3.9e-4 c (max dκ/ds 3.9e5 /c²). Station 0.60
    corner: a curvature bump of area 0.077°, half-width 5.4e-3 c. The pressure-side S-bend is
    kept (4.315° → 4.316°, m/c 0.00687–0.00856). β differs by at most 0.35°, inside the
    Kovach transition; every interior knot is C2 to round-off (one-sided X, X', X'' agree
    to 7e-16 c, 9e-16 L and 4e-14 relative; curvature to 2.3e-13 /c). Grid (101 × 31): the same m stations and blade stations, wall nodes
    moved by at most 5.6e-7 c.
  - emery_base: the closing segment meets the surface at x/c 0.999 at a 3.6° corner (C1's own
    construction, not a fit artefact); within 1e-6 c it becomes a curvature spike of about
    −770 /c over 2e-4 c. The sharp TE itself keeps its C1 wedge exactly (4.839° here).
  - Sweep: 40 blades (cl0 0 to 1.6, t/c 0.04 to 0.10, both variants) × 16 cascades (stagger
    −30° to 60°, solidity 0.8 to 2.0), 160 physical twins at 0.05 m: all built in both
    representations, none refused by one and not the other, max deviation 1.001e-6 c, no
    self-crossing, no suction/pressure crossing, open passages, identical pitch, LE and TE,
    throat changes within ±7.5e-7 c, twins exact (arrays and control points bit for bit).
  - Not done here, on purpose: metrics.py still differentiates wall nodes. The curves give
    tan β = y_u / m_u, the curvature and d tan β / dm = (y_uu m_u − y_u m_uu) / m_u³ at any
    u (`ParametricBladeCurve.wall_slope`); a wall node's u follows from
    `ParametricBladeCurve.project`. That hand-off waits for review of the C2 plots.
  - The numbers above are those of that first C2 (`fit_quintic_surface`, which keeps the
    legacy LE fair and its S-bend). It is kept for comparison; the C2 surfaces now carry the
    robust LE of the next entry.
- **Robust C2 leading edge (2026-11-11), C2 only; C1, grid.py and metrics.py unchanged.** The
  legacy LE fair (a polar quintic from the LE circle to station 0.005) is a historical
  construction, not NACA data, and it is not fair across the family: the pressure side of
  cambered blades turns back (4.3° at 65-(12A10)10, 8.1° at cl0 1.6), thin blades wiggle (a
  "chin"; −82° of reverse turning summed over the polyline at t/c 0.04, cl0 1.2), and a fair
  G2 patch between the legacy ends cannot help: at station 0.005 the surface tangent points
  1.26° back across the chord from the circle end. So C2 replaces the LE, same rule on both
  surfaces (module docstring, "The C2 leading edge"): the nose circle (legacy R_LE, kept to
  20°) and a Whewell-form fair r(φ), radius of curvature against turning angle, a cubic
  B-spline with non-decreasing control radii (monotone curvature, no overshoot, no
  inflection by construction), landing with G2 on the C1 table surface (as the C2 fit
  smooths it); the fair minimizes ∫(dκ/ds)² ds by reweighted, equality- and bound-
  constrained least squares. The family-independent engine (`LeadingEdgeNose`,
  `fair_leading_edge`) takes any nose and any source surface; NACA 65 calls it.
  - Anchor rule, and why: half an LE radius of source past the earliest anchor at which such
    a fair exists. The first rule tried, "the earliest anchor whose fair stays within a
    fairness budget", gave the fairest short LEs but is not continuous: the optimal energy is
    not monotone along the source (its active set changes, and the knots quantize where the
    fair leaves the circle), so its first crossing of the budget switched branches, and the
    LE jumped by 2–4e-5 c between neighbouring blades (pressure side at cl0 1.48→1.50 and at
    t/c 0.070–0.073, measured on a 183-blade continuity sweep). The existence boundary is a
    clean edge (the energy climbs steeply toward it), so the new rule moves continuously.
  - Default 65-(12A10)10 (Kovach, 33°): S-bend gone (reverse turn 4.3° → 0, curvature sign
    changes 2 → 0); LE-zone deviation from C1 2.47e-4 c (suction) and 2.61e-4 c (pressure),
    RMS 1.3e-4 / 1.1e-4; anchors 0.0043 c and 0.0078 c past station 0.005; departures 20.8°
    and 21.0°; R_LE³∫κ_s² ds 1.10 and 0.76; downstream 1.000e-6 c as before. Thin 65-(12A10)04:
    chin gone, 6.1e-4 / 9.3e-4 c. High camber cl0 1.6: 3.2e-4 / 3.0e-4 c. Symmetric 65-010:
    4.9e-6 c (its legacy fair was already nearly the fairest one). On the suction side a short
    second curvature drop remains just ahead of the anchor (−58 → −30 /c at the default): the
    source there is flatter than the mean curvature a fair landing on it from this nose needs.
    It is monotone; reported, not hidden.
  - Representation: the C2 spline is within 1.3e-9 c of the fair, its nose curvature within
    6e-6 of 1/R_LE, |κ| rises by at most 2e-6/R_LE over an LE zone; knots are refined near
    the departure and the anchor (a derivative of the curvature jumps at either).
  - Sweep (the 40 blades × 16 cascades of the previous entry, 160 physical twins): every
    cascade built in both representations; on all 80 surfaces no curvature sign change and no
    reverse turn in the LE zone (legacy: up to 53.5° at cl0 1.6, t/c 0.04, 112 sign changes in
    all); LE-zone deviation at most 9.6e-4 c (thin, high camber), downstream 1.000e-6 c;
    departures 20.7°–21.5°; fairness at most 1.29; no crossing, open passages, twins exact. The
    throat moves where it ends on the neighbour's LE zone: up to +9.6e-4 c (o/s +1.9e-3) on
    the t/c 0.04 blades, the chin removed. Continuity (183 blades, t/c step 0.001, cl0 step
    0.02 and 0.04): anchors step evenly (at most 1.6 times their neighbours' steps), no jump.
  - Tuning knobs, measured on the 65-(12A10)10: a longer kept circle (`LE_FAIR_MIN_ARC` 30°,
    40°) brings the pressure side closer to C1 (1.1e-4 c at 40°) but the suction side away
    (6.5e-4 c) and steeper (max |dκ/ds| R² 2.4 → 7.1); `LE_ANCHOR_OFFSET` 0.25 instead of 0.5:
    1.6e-4 / 2.0e-4 c, suction max |dκ/ds| R² 3.6 instead of 2.4.

- **C2 grid → metrics hand-off (2026-11-11), grid.py and metrics.py reopened for C2 only;
  geometry.py unchanged, C1 unchanged bit for bit.** Before, a C2 grid read the sampled
  arrays through the C1 path (a parametric PCHIP through the samples) and metrics.py
  differenced the wall nodes, so the C2 curves reached the solver only through two
  interpolation layers. Now, for a C2 cascade:
  - `PassageBoundaries.blade_curves` holds `cascade.active_curves` by reference (None for
    C1), and grid.py reads the blade from the curves everywhere it looks at it: station cuts,
    topology, wall nodes, wall identity and the passage check (grid.py section 7). Each
    curve is cut into branches monotone in m and y at the exact roots of m′(u) and y′(u)
    (per-piece polynomial roots, polished by bisection on the curve's own derivative), and
    x = m_i is solved on every crossed branch by bisection to round-off: the wall node is
    X(u_i), y exact, |m(u_i) − m_i| ≤ 4.4e-16 c. The passage-facing root is chosen by the
    existing rule (top crossing = lower wall, bottom crossing one pitch up = upper wall, each
    on its contour from the nose tip to the tail end), never by a surface name.
  - `WallSide.u` (NaN for C1) records each wall node's parameter on the curve that
    `WallSide.surface` names; `Grid.wall_curve_derivative(k)` hands on X, X_u, X_uu of that
    curve at that u, and Grid refuses C2 wall nodes off their curve points (TOUCH chord).
  - metrics.py takes tan β = y_u/m_u and d tan β/dm = (y_uu m_u − y_u m_uu)/m_u³ on the C2
    blade walls, no difference stencil; |m_u| ≤ √ε |X_u| is a MetricsError (nothing
    capped). New fields for both representations: `dtan_beta0_dm`, `dtan_beta1_dm`,
    `dtan_beta_dm` (affine in η), and the property `dtan_beta_deta` (= `dS_dm`). d²S/dm² is
    not stored (no coefficient A–E needs it; it is `dtan_beta1_dm − dtan_beta0_dm`).
  - Unchanged: the stations, the extensions and `rebuild_exit`, the linear η map, the C1
    path (55 C1 grids and their metrics compared bit for bit, every array), the C2 extension
    metrics (bit for bit against the sampled reading).
  - Measured for the 65-(12A10)10 at 33° (`scripts/plot_c2_wall_handoff.py`): the same
    blade run as the sampled reading at n_passage 31 to 401 (and on all 120 grids of a
    12-cascade sweep); wall nodes moved by up to 9.8e-6 c (RMS 2e-6 c), the PCHIP's own error
    between samples, largest at m/c 0.2–0.5 (up to 5.2e-5 c on a thin long-tailed blade); min
    S moved by ≤ 1.7e-6 c. Wall slopes: interior (≥ 0.05 c from the ends) |Δ tan β| ≈ 3e-3
    at every resolution (the PCHIP floor, now gone); first/last nodes up to 57° at 31
    cells, 18–19° at 101 (the one-sided difference against a vertical tangent). The large LE
    angles are geometry: at 101 cells β = 79.2° / −75.5° at the first station, the curves'
    own tangents. A station can differ from the sampled reading only inside the 1.9e-6 c
    (nose) or 2.7e-7 c (tail) by which the curve reaches past its own extreme samples; it is
    then a blade station of the curve (tested).

- **coefficients.py (2026-11-12): A to E of Eq. 5-21, implemented, under review.**
  geometry.py, grid.py and metrics.py unchanged.
  - Equation and signs, as discretization.py must take them: A ψ_ηη − 2B ψ_ηm + C ψ_mm +
    D ψ_η + E ψ_m = 2Sω sin φ, with A = μ/(S cos²β), B = μ tan β, C = μS, D = A_η − B_m,
    E = C_m − B_η (the expansion of ∂F/∂η − ∂G/∂m = 0, F = Aψ_η − Bψ_m, G = Bψ_η − Cψ_m).
    The right-hand side is zero here: flat cascade at constant radius, sin φ = dr/dm = 0.
  - Per unit depth: μ = ṁ′/ρ with `mass_flow_per_depth` ṁ′ = ṁ/b, the mass flow through
    one passage per unit stream-sheet thickness (kg/(m s)); b never appears, not even as 1.
    Computing ṁ′ from the inlet state belongs to a later layer.
  - Lengths: geometry, grid and metrics stay chord-normalized (m* = m/c, S* = S/c);
    coefficients.py applies the one physical scale, `c_ref` in metres: S = c_ref S*,
    ∂/∂m = (1/c_ref) ∂/∂m*. `c_ref=None` is normalized mode, L_ref = 1 chord, **not** a
    one-metre blade: no SI length and no physical mass flow is defined
    (`CoefficientFields.physical_scale_defined`). A grid of a `physical_chord` cascade
    (already in metres) is refused, so the chord is never applied twice.
  - ρ is a frozen (n_m, n_η) field, finite, positive, and equal at η = 0 and 1 on the
    periodic stations (one point and its image); the module knows no gas law.
  - A, B, C pointwise from the metric fields, nothing clipped (A ∝ 1/cos²β is large at a
    steep first station: 29.0 at β = 79.25°, n_passage 101). D and E by second-order
    differences of the whole fields A, B, C (not expanded, `dtan_beta_dm` not used; the
    expanded form with metrics' exact derivatives is the tests' independent reference):
    m central over the whole range, across the region transitions too, one-sided at the
    far stations; η one-sided into the fluid at blade walls and the periodic central
    difference on the periodic stations (neighbors n_η − 2 below η = 0 and 1 above η = 1).
  - Dimensions: [A] = [D] = L/T, [B] = [E] = L²/T, [C] = L³/T, ψ dimensionless, every
    term of Eq. 5-21 L/T; discretization.py must use Δm = `length_scale` × grid.dm.
    Fixed ṁ′: A, D ∝ L⁻¹, B, E ∝ L⁰, C ∝ L; similar twins (ṁ′ ∝ L): every term of Eq. 5-21
    unchanged.
  - Measured on the default C2 cascade (`scripts/audit_coefficients.py`): second order on
    the smooth rear of the blade against the exact-derivative reference; over the front
    60 % of the chord the C2 walls carry the thickness table's curvature steps in spans
    far shorter than Δm, so the central difference of B averages what the exact derivative
    samples (|ΔD| up to 0.66 at n_passage 101, 0.07 at 801); at the four corner stations
    of the LE and TE transitions D and E are large (|D| 300 at 101, 1.6e3 at 401) because
    tan β changes by O(1) where a side line meets a wall. Unsmoothed; how the ψ equation
    treats those stations is discretization.py's choice.

- **discretization.py (2026-11-12): the interior difference equation, implemented, under
  review.** Earlier modules unchanged.
  - Production form: the expanded Eq. 5-21, A ψ_ηη − 2B ψ_mη + C ψ_mm + D ψ_η + E ψ_m = Q
    (Q = 0 here), with Aungier's second-order central differences, Eqs. 5-35 to 5-39 (truncation
    O(h²)); the printed difference equation, not the conservative Eq. 5-20, which the tests
    use only as an independent check. No upwind or flow-direction-dependent difference
    anywhere; the boundary formulas to come are second-order one-sided, not upwind.
  - Eq. 5-44: the raw nine-point equation divided by −H, H = 2A/Δη² + 2C/Δm² > 0, so the
    center weight is 1: west = −(C/Δm² − E/(2Δm))/H, east = −(C/Δm² + E/(2Δm))/H, south =
    −(A/Δη² − D/(2Δη))/H, north = −(A/Δη² + D/(2Δη))/H, mixed = B/(2ΔmΔηH) times
    ψ_NE − ψ_SE − ψ_NW + ψ_SW (so NE = SW = +mixed, SE = NW = −mixed), rhs = −Q/H = 0.
  - **Two coefficient sets:** the PDE coefficients A to E (`CoefficientFields`, L/T to
    L³/T) and the dimensionless stencil weights (`InteriorStencil`, named by direction).
    Aungier's tilde weights Ã to Ẽ are the latter; the code never calls a weight A to E.
  - Lengths: Δm = `coefficients.length_scale` × grid.dm, Δη = grid.deta, as coefficients.py
    defines them; the weights are dimensionless and unchanged under the chord and mass-flow
    scalings (tested).
  - Interior nodes only (i = 1 … n_m − 2, j = 1 … n_η − 2), NaN elsewhere. The wall,
    periodic-line and far-station rows, and the one-sided Eqs. 5-40 to 5-43 where they need
    them, are the boundary step; the global numbering, the sparse matrix and the solve are
    later steps, each reviewed on its own.
  - `build_interior_stencil` refuses coefficient fields that are not those of its metrics
    (C = A (L S*)² cos²β and B = C tan β/(L S*) at every node, whatever ρ): a coefficient set
    left over from before `rebuild_exit` is caught.
  - Measured on the default C2 cascade (`scripts/audit_discretization.py`): weights finite,
    H > 0 (min 2.8e4 at n_passage 101), |mixed| ≤ |sin β|/8, west + east + south + north = −1;
    no row diagonally dominant where β ≠ 0 (sum of |neighbor weights| up to 1.49); south or
    north positive only at the corner stations beside the LE (and, at 401, the TE)
    transition, where |D|Δη/(2A) reaches 3.9. Second order on smooth manufactured fields
    (and against the conservative form); about first order against it over the curvature
    steps of the front blade; no convergence at the corner stations.

- **discretization.py, the periodic PDE line (2026-11-12), under review; `InteriorStencil`
  unchanged bit for bit.**
  - Convention for the assembly: on a periodic station j = 0 is the constraint side
    (ψ[i, 0] = ψ[i, n_η − 1] − 1, `boundary_conditions.py`) and j = n_η − 1 the PDE side; a row
    of Eq. 5-44 is written at (i, n_η − 1) for i = 1 … first_blade_i − 1 and last_blade_i + 1 …
    n_m − 2 (`build_periodic_pde_rows`, rows of `PDETerm`s, center 1).
  - Ordinary periodic row: central differences, the node above the line by the periodic
    continuation ψ[k, n_η] = ψ[k, 1] + 1 (`periodic_ghost`). The jump enters through ψ_η and
    ψ_ηη: the row's rhs is −north = (A/Δη² + D/(2Δη))/H although Q = 0; in the mixed term
    the jumps cancel.
  - Transition rows: at first_blade_i − 1 (last inlet station) the central mixed difference
    would need ψ above the first blade station's upper wall, inside the blade, where no
    continuation exists (and `periodic_ghost` refuses one). There ψ_mη = (3q_i − 4q_{i−1} +
    q_{i−2})/(2Δm), at last_blade_i + 1 (first exit station) (−3q_i + 4q_{i+1} − q_{i+2})/(2Δm),
    with q = (ψ[k, 1] + 1 − ψ[k, n_η − 2])/(2Δη); second-order one-sided in m, chosen for the
    topology, not the flow direction (no upwinding). This is our topology-aware second-order
    extension of Aungier's Taylor-series differences (his rule: no stencil through the
    blade), not a numbered Aungier formula.
  - Final transition policy (revised the same day, from measured geometry):

    | Row | ψ_m | ψ_mm | ψ_mη |
    |---|---|---|---|
    | LE, first_blade_i − 1 | central | central | backward (3, −4, 1) |
    | TE, last_blade_i + 1 | forward (−3, 4, −1)/(2Δm) | forward (2, −5, 4, −1)/Δm² | forward (−3, 4, −1) |

    At the TE the central ψ_m, ψ_mm read the last blade station's upper wall node, and the
    straight link to it runs through the +pitch blade's tail (36 %, 44 %, 26 %, 96 % of the
    link at n_passage 61, 101, 201, 401); the LE links stay in the fluid. Every forward form is
    second order (Taylor: exact for quadratics / cubics) and reads only last_blade_i + 1 … + 4.
    The TE row keeps the divisor H, so its center is (2A/Δη² − 2C/Δm² + 3E/(2Δm))/H, not 1
    (−0.78 at n_passage 101): not diagonally dominant, for the linear solver to know. These
    one-sided formulas are topology-aware, not upwind.
  - Measured: second order on a quasi-periodic ψ for all three row kinds; no straight chord
    of a periodic row intersects the blade at any of 28 grids (exact crossings with the C2
    curves, `scripts/link_geometry.py`). Not treated: interior rows beside both transitions
    have straight chords through the nose or tail at some resolutions (open decision above;
    a geometric fact about the chords, audited against the mapping itself by
    `scripts/audit_transition_mapping.py`).

- **boundary_conditions.py (2026-11-12): the boundary relations, implemented, under
  review.** Earlier modules unchanged.
  - Topology: the blade stations are `first_blade_i` … `last_blade_i`, cross-checked against
    `wall_identity.stations` and `region`; never m, the LE/TE landmarks or surface names.
    The first and last blade stations have walls like any other (no LE/TE special condition).
  - Relations Σ aₖψₖ = rhs: blade walls ψ[i, 0] = 0 and ψ[i, n_η − 1] = 1 (Dirichlet, the
    normalized ψ of one passage); periodic stations ψ[i, n_η − 1] − ψ[i, 0] = 1, never
    ψ_upper = ψ_lower (a periodic scalar has jump 0); far stations ψ_m = 0 as Aungier's
    second-order one-sided differences −3ψ₀ + 4ψ₁ − ψ₂ = 0 and 3ψ_N − 4ψ_{N−1} + ψ_{N−2} = 0
    (the 2Δm of the derivative, Δm = `length_scale` × grid.dm, cancels in a homogeneous
    condition, so nothing depends on c_ref). One-sided because the boundary has neighbors on
    one side only: not upwind; the solver has no upwinding.
  - For the assembly (Aungier 2000, *Centrifugal Compressors*, Ch. 13: the far boundary
    values are eliminated from the adjacent PDE rows and rebuilt after the solve):
    eliminations ψ₀ = 4/3 ψ₁ − 1/3 ψ₂, ψ_N = 4/3 ψ_{N−1} − 1/3 ψ_{N−2}; `solve_for` on any
    relation; the periodic continuation ψ[i, −1] = ψ[i, n_η − 2] − 1, ψ[i, n_η] = ψ[i, 1] + 1
    for the PDE rows on the periodic lines. Both periodic images are kept; which one the
    assembly eliminates is its choice.
  - Corners: each keeps its far relation and its station's jump. Per far station the three
    relations are dependent (U(0, n) − U(0, 0) = −3P(0) + 4P(1) − P(2)), so the assembly
    carries two; convention: eliminate both far images by the meridional relation, and the
    far jump holds by itself (4/3 − 1/3 = 1).
  - Not here: the matrix and the substitution (next step), the Kutta condition (an outer
    constraint on the discharge angle, `kutta.py`), any density update. Open for the
    assembly: the periodic-line rows beside the transitions (their diagonal neighbors lie at
    the first / last blade station, across the blade's cut of 9.2e-3 c / 1.4e-2 c at
    n_passage 101), and Eq. 5-44 weights on the periodic lines (`InteriorStencil` has them
    NaN there by design).

- **grid.py admissible n_passage (2026-11-12/13): uniform grid kept, inadmissible values
  rejected, never repaired.** The user's decision: one constant Δm (Chapter 5), no local
  station redistribution, no LE/TE interface treatment, no change to the difference
  equations. A uniform grid is admissible when its first and its last blade station each
  cut the blade once on the suction and once on the pressure surface (neither through the
  LE or TE point itself); `grid.py` section 8.
  - Why it fails: the nose tip lies on the suction surface w_LE = m_le − m_tip ahead of the
    LE point (the SS/PS branch point), and a station in (m_tip, m_le] crosses the suction
    surface twice. The first blade station lies in (m_tip, m_tip + Δm] and the last in
    [m_tail − Δm, m_tail), so the pattern is aliased; once Δm ≤ w every grid fails
    (Δm never grows with n_passage). Measured over every n_passage 20 to 800 (C2
    65-(12A10)10, solidity 1, inlet 45°, one-chord extensions), with the production
    labels, the production rule and an independent C2 root count agreeing at all 7810
    stations, and with LE valid ⇔ m_f > m_le and TE valid ⇔ m_l < m_te on every grid:

    | stagger | w_LE (c) | w_TE (c) | admissible of 781 | largest admissible | ceiling (every n from it fails) |
    |---|---|---|---|---|---|
    | 13° | 1.696e-3 | 3.36e-4 | 274 | 570 | 575 |
    | 23° | 2.893e-3 | 5.41e-5 | 142 | 312 | 319 |
    | 33° | 4.524e-3 | 1.35e-5 | 72 | 176 | 186 |
    | 43° | 6.587e-3 | 2.16e-4 | 43 | 106 | 111 |
    | 53° | 9.680e-3 | 6.54e-4 | 14 | 48 | 63 |

  - With equal extensions the stations are symmetric about the blade's middle, so the first
    wall station's offset past the LE point equals the last one's ahead of the TE point, and
    (w_TE < w_LE here) a TE failure always comes with an LE failure. Unequal extensions can
    fail at the TE alone (13°, a 1.25-chord exit extension, n_passage 102).
  - API: `make_grid(..., require_ss_ps=True)` and `validate_uniform_grid_topology(grid,
    n_passage=..., stagger=...)` raise `GridTopologyError` (a `GridError`) naming the failing
    station(s), the nose tip / tail end, the LE / TE point, the nearest admissible
    n_passage either side, the finest admissible one and the ceiling; `find_valid_n_passage`
    and `n_passage_ceiling` are read-only; `wall_station_topology(grid)` reads a built
    grid. Nothing chooses or substitutes an n_passage.
  - Default kept: `make_grid` without the flag builds every grid as before (bit for bit),
    because the existing module tests and fingerprints use inadmissible grids. Our choice,
    flagged for review.
  - Refinement sequences for a convergence study (`scripts/audit_uniform_grid_validity.py`):
    recommended (the first wall station 0.42 to 0.57 Δm past the LE point on every grid, so
    the corner stations sit alike): 13° 60, 102, 173, 294 (r 1.70); 23° 43, 66, 101, 159
    (r 1.53–1.58); 33° 29, 43, 65, 97 (r 1.49–1.51); 43° 26, 34, 45, 59 (r 1.31–1.32); 53°
    only 22, 31, 45 (r 1.41, 1.46; the last with the wall station 0.05 Δm past the LE point).
    Finer admissible grids exist (33° up to 176) but put the first wall station within a
    few hundredths of Δm of the LE point and, by the symmetry, the last next to the tail's
    vertical tangent (d tan β/dm up to −1.5e4 there).
  - Not changed by the rule: the side-line C0 jump at the nose tip and tail end (−6.5e-3 to
    −4.7e-2 c at the LE, the same on every grid), so central m differences across the
    transitions stay inconsistent on admissible grids too.

- **matrix_assembly.py (2026-11-13): the local equations → F Ψ = Q, implemented, under
  review.** `grid.py`, `metrics.py`, `coefficients.py`, `discretization.py` and
  `boundary_conditions.py` unchanged bit for bit; no stencil, coefficient or boundary formula
  touched.
  - Aungier's matrix method (2003 Sec. 5.4; 2000 Ch. 13, as specified for the task): the
    node equations form one square sparse system [F][Ψ] = [Q]; nodes numbered k = i n_η + j
    (zero-based, C order: `full_index`, `node_of`), so the nine-point neighbors of k are
    k ± 1, k ± n_η, k ± n_η ± 1; the far-boundary values are eliminated into the adjacent
    equations and rebuilt after the solve.
  - The unknowns: every node except the two far stations. Each far node (both periodic
    images and the corners) is eliminated by its own ψ_m = 0 relation solved for it
    (`Relation.solve_for`, the same objects as `upstream_elimination` /
    `downstream_elimination`). The far stations' periodic jumps are then implied
    (P(0) = (4P(1) − P(2))/3, the identity of `boundary_conditions.py`) and checked after the
    solve (`implied_residuals`), not carried as rows. Walls and periodic jumps stay explicit
    rows; a jump row belongs to the image without a PDE row, which is j = 0, the constraint
    side of the `discretization.py` convention (found from the rows, not from j). So
    N = (n_m − 2) n_η equations for N unknowns: checked, never patched. Row r is the equation
    of unknown r.
  - Substitution in one place: ψ_full = T Ψ + c, F = L T, Q = q − L c (L the local
    coefficients), which is ψ_b = c + Σ α ψ_r substituted into every a_b ψ_b; chained
    eliminations resolved depth first, a cycle refused, nothing iterated. The periodic +1
    enters once, in `build_periodic_pde_rows`' rhs; assembly adds only an elimination's
    constant (zero for the far eliminations, so Q equals the local rhs row for row).
  - Storage, our deviation from Aungier 2000: general sparse COO → CSR (duplicates summed,
    exact zeros dropped) instead of his nine-slot banded storage and elimination. Memory is no
    longer the limit, the linear system is the same, and the current rows do not fit nine
    slots: the periodic rows wrap from j = n_η − 1 to j = 1, the LE transition row reaches
    3 n_η − 2 columns back and the TE transition row reads ten nodes up to 3 n_η ahead. The
    solve stays in `linear_solver.py`; Aungier's band solver may come back there as a
    reference. No row is scaled: the TE row's center is not 1 and can be near zero.
  - Measured (33°, n_passage 43, n_η 21): 3087 nodes, 42 eliminated, 3045 unknowns and rows
    (2755 interior, 86 wall, 102 jump, 100 standard + 1 LE + 1 TE periodic PDE rows),
    nnz 25 884 (density 2.8e-3), 1 to 10 per row (mean 8.50), bandwidth 61 below and 63 above
    (3 n_η − 2, 3 n_η), 120 substitutions. On eight admissible grids (13° to 53°): (F x − Q)
    equals the residual of each row's own production object on `reconstruct(x)` to at most
    8.9e-16 (exactly on rows without a substitution); a known vector is recovered by a sparse
    direct solve to 1.9e-14 … 3.4e-13 (tests only). Twelve mutants of the module (constant
    sign, dropped constant or rhs, chained coefficient, row scaling, transposed offsets,
    wrong jump owner, row index, missing exit elimination, reconstruction order, dropped
    elimination weights) are each caught by the tests.
  - `discretization.assemble` stays a stub (its test pins the `NotImplementedError`);
    `matrix_assembly.assemble_system` / `assemble_from_coefficients` take its place in the
    data flow.

- **linear_solver.py (2026-11-13): the sparse direct solve, implemented, under review.**
  `matrix_assembly.py` and everything upstream unchanged bit for bit.
  - The same matrix problem as Aungier: F Ψ = Q of `matrix_assembly.py`, unchanged. A
    different linear-algebra implementation: Aungier 2000 stores the band of his nine-point
    system in a problem-specific layout and solves it by forward and backward elimination;
    this solver passes the CSR matrix to `scipy.sparse.linalg.spsolve` (SuperLU, COLAMD
    column ordering, partial pivoting; `use_umfpack=False` so that no optional package changes
    the factorization). The discretized equations are not touched; only how F Ψ = Q is
    solved differs.
  - No iterative method, preconditioner, row scaling, regularization or pinned node: the TE
    periodic row's diagonal is not 1 and can be near zero, and SuperLU pivots (a zero-diagonal
    permutation system is tested). A singular F (SuperLU's `MatrixRankWarning`) and a
    non-finite solution are errors (`LinearSolveError`); other warnings are passed on; a large
    residual is reported, not refused; ψ outside [0, 1] is no error at this stage.
  - `solve_linear_system(assembly)` returns Ψ, ψ(i, j) rebuilt by `matrix_assembly.reconstruct`
    (no elimination formula repeated) and the diagnostics: ‖r‖∞, ‖r‖₂, ‖r‖₂/‖Q‖₂ (‖r‖₂ when
    Q = 0), the normwise backward error ‖r‖∞/(‖F‖∞‖Ψ‖∞ + ‖Q‖∞), the implied far-jump residual,
    shape, nnz, min/max Ψ. `solve(matrix, rhs)` and `residual_norm` are the matrix-level
    pieces, as planned.
  - Measured (33°, n_passage 43, n_η 21; ρ = 1, a frozen-density linear solve, not a flow
    result): 3045 unknowns, nnz 25 884, solved in 7.6 ms; ‖r‖∞ 6.9e-16, ‖r‖₂ 6.5e-15,
    relative 5.4e-16, backward error 1.0e-16, implied far jumps 0; Ψ from −0.040 to 1.038;
    every original local equation on the rebuilt ψ (Eq. 5-44 rows, periodic and LE / TE rows,
    walls, jumps, far relations) below 6.2e-16 and equal to (F Ψ − Q) row for row to 3.6e-16.
    The same on seven more admissible grids (13° to 53°, n_η 7 to 44): residuals at most
    8.3e-16. Twelve mutants of the module (singular warning ignored, finite check dropped,
    row scaling, residual and diagnostic definitions, reconstruction skipped, warnings
    swallowed, size check dropped) are each caught by the tests.
  - Possible later: a verification solver that reproduces Aungier's banded elimination, to
    compare Ψ_Aungier with Ψ_scipy on the same F and Q.

- **flow.py, velocity recovery (2026-11-13), implemented, under review.** Every upstream
  module unchanged bit for bit; the gas model, the static state and the density update are
  not here.
  - Aungier's Eqs. 5-18, 5-19, W_m = ṁ/(S b ρ) ψ_η and W_θ = ṁ/(b ρ) [(tan β/S) ψ_η − ψ_m], in
    the normalization `coefficients.py` already uses: ṁ′ = ṁ/b (`mass_flow_per_depth`), b
    absorbed and never an input, μ = ṁ′/ρ, so W_m = μ ψ_η/S and W_θ = μ [(tan β/S) ψ_η − ψ_m]
    with S = L_ref S* and ψ_m in the physical m (Δm = L_ref grid.dm). `velocity_components`
    takes the `CoefficientFields` that built the matrix and reads ṁ′ and L_ref there, and
    checks the density given against them (C = ṁ′S/ρ, B = ṁ′ tan β/ρ at every node to
    round-off): the velocities use the normalization and the density of the ψ they come
    from (Aungier's lag ρ⁽ⁿ⁾ → ψ⁽ⁿ⁾ → W⁽ⁿ⁾ → ρ⁽ⁿ⁺¹⁾). Velocity units: m/s with c_ref in metres,
    ṁ′ in kg/(m s), ρ in kg/m³; with c_ref = None those of ṁ′/(ρ chord).
  - Differences (second order, uniform grid), node classes from `boundary_conditions.py`:
    ψ_η central inside, one-sided into the passage on the blade walls, central across the
    periodic lines with `periodic_ghost` (the +1 applied there, once); ψ_m = 0 exactly on the
    blade walls (ψ = 0 and 1 along them, so W_θ = W_m tan β there), one-sided at the far
    stations, central elsewhere (along the periodic lines on real nodes). None of the
    special LE/TE PDE rows of `discretization.py` is used; the central m differences at the
    stations beside the transitions cross them, as they do everywhere (not revisited).
    **Superseded 2026-11-13:** that crossing was the defect behind the sonic node at the first
    blade station; ψ_m is now taken by m region (the entry "flow.py, ψ_m by m region" below).
  - The result `FlowField` is the kinematics only: ψ_m, ψ_η, W_m, W_θ, W2, W (the stub's
    `state` and `mach` are gone; the static state is `eos.static_state_from_velocity`'s
    `StaticState`).
  - Measured: every node class (interior, far inlet / exit, both periodic lines, both blade
    walls) second order on manufactured fields (orders 1.94 to 2.2 from 21 η nodes);
    Eqs. 5-18, 5-19 equal the flux form W_θ = (B ψ_η − C ψ_m)/S to round-off; a uniform flow
    on the real grid gives W_θ/W_m = tan α to 1e-12 on the extensions. On the real solved ψ
    (33°, n_passage 43, n_η 21, ρ = 1, ṁ′ = 1, normalized; kinematic verification values,
    not a physical flow): periodic images agree to 1.2e-14, W_θ − W_m tan β ≤ 2.2e-16 on
    both walls, ρ S W_m = ṁ′ ψ_η to 1.1e-16, the station mass flow equal to ṁ′ to 4.4e-16 on
    the periodic stations and within 3.4e-3 on the blade stations (median 2.0e-4; largest at
    the first blade station; it falls under η refinement). Twelve mutants (the +1 missing or
    doubled, ψ_m on the walls, a wall stencil through the blade, the W_θ sign, S or Δm
    unscaled, the density check, a first-order far difference, the mass-flow quadrature,
    one wall's tan β, W without the root) are each caught by the tests.

- **eos.py, the thermodynamic state and the relaxed density (2026-11-13), implemented,
  under review.** No other module changed; no density loop, tolerance or Kutta condition.
  - Density update:

    ```text
    Aungier baseline:          direct EOS density replacement, rho(n+1) = rho_EOS(W(n))
    Current project baseline:  density under-relaxation, lambda_rho = 0.5:
                               rho(n+1) = 0.5 rho(n) + 0.5 rho_target
    Reason:                    additional robustness of the nonlinear iteration (our choice,
                               not Aungier's)
    lambda_rho = 1:            reproduces Aungier's direct replacement exactly (tested bit for bit)
    ```

    `relax_density(density_old, density_target, relaxation=0.5)`; 0 < λ ≤ 1 (the range of
    `config.IterationSettings.density_relaxation`). The names keep the two densities apart:
    `rho_target` = ρ_EOS(W⁽ⁿ⁾) from `static_state_from_velocity`, `rho_next` the relaxed one.
    The coefficients of pass n + 1 are built from `rho_next`, never from `rho_target`.
  - The intended loop (for `solver.py`, not implemented): ρ⁽ⁿ⁾ → coefficients → F Ψ = Q →
    ψ⁽ⁿ⁾ → W⁽ⁿ⁾ → ρ_target = ρ_EOS(W⁽ⁿ⁾) → ρ⁽ⁿ⁺¹⁾ = 0.5 ρ⁽ⁿ⁾ + 0.5 ρ_target → coefficients of
    pass n + 1. Stopping rules and tolerances are not chosen here (`convergence.py`).
  - The gas: thermally and calorically perfect, `CaloricallyPerfectGas(gamma, gas_constant)`
    (the stub's `PerfectGas` fields; the name is kept as an alias), cp = γR/(γ − 1), h = cp T
    (zero at T = 0), a = √(γRT), ρ = p/(RT).
  - Entropy: never absolute. `TotalState(pressure, temperature)` is the specified inlet
    absolute total state and the entropy reference of the stream sheet; every state has its
    entropy, p = Pt,in (T/Tt,in)^(γ/(γ − 1)). This replaces the stub's `state_from_pT` /
    `state_from_hs`, which needed an absolute s; `GasState` is kept as an alias of
    `StaticState` (temperature, pressure, density, enthalpy, sound_speed).
  - Energy: h = H′ − W²/2, H′ = I + (rω)²/2, one stream surface and one radius, so H′ is one
    number. `blade_row_reference` builds I = cp Tt,in − ω r C_θ,in; a rotating row must be
    given r and the inlet absolute swirl C_θ,in (never inferred from an angle here); ω = 0
    gives I = cp Tt,in, so the stationary T = Tt,in − W²/(2cp) is the ω = 0 value of the one
    model. T ≤ 0 (W ≥ √(2H′)) is an error: nothing is clipped or floored.
  - Initial density (Aungier): the local relative total density, the state at W = 0:
    Tt′ = H′/cp, Pt′ = Pt,in (Tt′/Tt,in)^(γ/(γ − 1)), ρ′ = Pt′/(R Tt′); stationary
    ρ⁽⁰⁾ = Pt,in/(R Tt,in), uniform.
  - Units: one consistent system for the gas and the velocities (SI: W in m/s from `flow.py`
    in physical mode, c_ref in metres and ṁ′ in kg/(m s)).
  - Measured (air, γ 1.4, R 287.05, Pt 101 325 Pa, Tt 288.15 K, W up to M 0.95): T, p, ρ, a
    against the independent closed forms to 2.2e-16, 8.9e-16, 7.8e-16, 2.2e-16 relative;
    W = 0 gives the inlet total state exactly; isentropy cp ln(T/Tt) − R ln(p/Pt) ≤ 5.7e-14
    J/(kg K); energy h + W²/2 = I + (rω)²/2 to 0 (stationary) and 2.0e-16 (rotating); the
    Mach-number ratios to M 1.5 for three gases. Fourteen mutants (cp, the isentropic
    exponent, the ½ of the kinetic energy, the rothalpy sign, the blade speed, a temperature
    floor, the relaxation weights, the default, input mutation, the λ bounds, ρ = p/(RT),
    a = √(γRT), negative speeds, a static instead of total initial density) are each caught.

- **solver.py, the density loop at a fixed discharge angle (2026-11-13), implemented, under
  review.** DENSITY-COUPLED / FIXED-DISCHARGE-ANGLE SOLUTION, NOT FINAL AUNGIER KUTTA-CONVERGED
  SOLUTION. Every numerical module unchanged; `convergence.py` got its change norms and density
  stop test (its planned job).
  - Pass n, calls only: ρ⁽ⁿ⁾ (resident) → `build_coefficients` → `assemble_from_coefficients`
    → `solve_linear_system` → ψ⁽ⁿ⁾ → `velocity_components` → W⁽ⁿ⁾ →
    `static_state_from_velocity` → ρ_target⁽ⁿ⁺¹⁾ = ρ_EOS(W⁽ⁿ⁾) → `relax_density` → ρ⁽ⁿ⁺¹⁾ (next).
    The next pass is built from ρ⁽ⁿ⁺¹⁾, never from ρ_target (tested at loop level).
  - λ_ρ = `config.iteration.density_relaxation`: 0.5 default, the project baseline (ours, for
    robustness); 1.0 = Aungier's direct replacement (tested: next = target exactly).
  - Start: `eos.initial_density`, Aungier's relative total density (Pt/(R Tt) stationary).
  - Convergence (`convergence.py`, decided here): change = max |new − old| / max |old|, the
    velocity as a vector; converged when the density change ρ⁽ⁿ⁺¹⁾ vs ρ⁽ⁿ⁾ and the velocity
    change W⁽ⁿ⁾ vs W⁽ⁿ⁻¹⁾ are both ≤ `density_tolerance` (the one tolerance `IterationSettings`
    has; no new one); pass 0 has no previous velocity and never converges.
    `max_density_iterations` passes without it: `DensityNotConvergedError` with the last pass.
  - End: one more evaluation at the converged ρ⁽ⁿ⁺¹⁾; the `Solution` reports that evaluation
    (density, coefficients, ψ, flow, static state of one pass; target density beside it).
  - Stops: M_rel = W/a ≥ 1 anywhere → `TransonicFlowError` (no transonic extension, nothing
    clipped); invalid thermodynamic states are `eos.py`'s errors; a rotating row and a
    missing physical chord are refused.
  - Inputs: config plus ṁ′ (`mass_flow_per_depth`) and c_ref (the chord in metres), which
    config does not hold; c_ref = None (normalized) is refused here, so W is in m/s against
    cp T in J/kg. The discharge angle is `config.flow.exit_angle_guess_deg`, fixed: the grid is
    built once and never rebuilt.
  - Dimensional verification case (constructed, not the NACA case): air (R 287.05, γ 1.4), Pt
    101 325 Pa, Tt 288.15 K, ω = 0, 33° stagger, n_passage 43, n_η 21, chord 0.1 m, inlet angle
    45° (relative = absolute), discharge angle κ_te = 17.92°, far-inlet Mach 0.3 →
    T 283.06 K, W 101.18 m/s, ρ 1.17158 kg/m³, W_m = W cos 45°, S = pitch = 0.1 m:
    ṁ′ = 8.3822 kg/(m s). With tolerance 1e-8: λ = 0.5 converges in 28 passes (+ the final
    evaluation), the density change falling monotonically by 0.557 to 0.561 per pass (the
    linear estimate 1 − λ + λ M² at M ≈ 0.35); λ = 1 in 10 passes, 0.114 to 0.122 per pass; both
    give the same field to 7.5e-9. Final: ρ 1.1314 to 1.2121 kg/m³ (start 1.2250 uniform), target
    mismatch 6.2e-9; W_m 45.5 to 105.7, W_θ 0.23 to 114.4, W 49.5 to 134.7 m/s; M_rel 0.146 to
    0.402; T 279.13 to 286.93 K; p 90 648 to 99 834 Pa; ψ −0.043 to 1.037; ‖FΨ − Q‖∞ 6.3e-16;
    station mass flow within 3.3e-3 (blade median 1.4e-4, periodic stations exact); far-inlet
    Mach 0.29999 (mean). At far-inlet Mach 0.6 the loop still converges (76 passes, peak M_rel
    0.92); at 0.7 it stops at pass 5 with M_rel 1.011 at the first blade station.
  - Thirteen mutants (next pass from the target, no final evaluation, the next density
    reported, λ hard-wired, no Mach stop, the pass limit off by one, a start of ones, the c_ref
    and rotation guards, the norm, the vector scale, "or" in the stop test, a converged first
    pass) are each caught by the tests.

- **API cleanup for the coupled solver (2026-11-13).** No equation, difference formula,
  assembly, solve, velocity or EOS relation changed.
  - Density relaxation: `config.IterationSettings.density_relaxation` defaults to
    `eos.DENSITY_RELAXATION` (0.5), imported, so the number lives in `eos.py` only (config
    already imported `eos`; no cycle). 0.5 = project default under-relaxation, 1.0 = Aungier's
    direct replacement; 0 < λ ≤ 1, enforced by `eos.relax_density`. The field moved to the end
    of the dataclass (a default must follow the fields without one); the tolerances and
    iteration limits keep no default, so `IterationSettings()` without arguments is still
    refused (no stopping rule is chosen yet).
  - Frames: `FlowConditions.total_pressure` / `total_temperature` are now
    `inlet_absolute_total_pressure` / `inlet_absolute_total_temperature`, Pt_in^abs and
    Tt_in^abs, the same numbers with their frame in the name (nothing constructed them yet).
    The inlet Mach number and angles are relative to the row. Rotating rows only:
    `stream_surface_radius` and `inlet_absolute_tangential_velocity` (C_θ,in^abs), None by
    default; `eos.blade_row_reference` refuses ω ≠ 0 without them and never infers a swirl
    from an angle. A stationary row (ω = 0) needs neither.
  - `flow.static_state` removed: an unused stub (no caller in the package, tests or
    scripts); `eos.static_state_from_velocity` owns W → h, T, p, ρ, a, and `flow.py` is
    kinematic only. `solver.py`'s skeleton docstring now names the real steps.
- **Density + Kutta coupled solver (2026-11-13; `kutta.py`, `solver.solve`, `convergence.py`,
  `config.py`).** No stencil, coefficient, assembly, linear solve, velocity or EOS formula
  changed; `grid.py` is used through `rebuild_exit` only.
  - The condition: Aungier 2003 Eq. 5-34, equal relative speeds on the two surfaces at the
    TE, at the last blade station i = `last_blade_i` (both wall nodes exist; never l + 1, never
    the tail end). The SS and PS node are read from `Grid.wall_identity` (on this project's
    admissible grids the SS is the η = 0 wall). R_K = W_SS − W_PS (m/s; > 0: SS faster),
    ε_K = |R_K| / (½(W_SS + W_PS)) ≤ 2, ε_K = 0 when both speeds are exactly zero. A pressure
    difference is not the residual: p(W) at one rothalpy and entropy falls strictly
    (dp/dW = −ρW, checked to 4e-10), so equal speeds are equal pressures (a test).
  - Angle conventions. Ours (`geometry.py`, `grid.py`, `metrics.py`, `FlowConditions`): β from
    the meridional direction, tan β = r dθ/dm = dy/dm. Aungier 2000 Ch. 13: β is the angle of a
    constant-η line with the TANGENTIAL direction, defined by cot β = [r ∂θ/∂m]_η (Eq. 13-23);
    m, θ and η = 0 → 1 run the same way (Eq. 13-4 = Eq. 5-2). So cot β₂₀₀₀ = tan β exactly,
    with no sign change, and his Eq. 13-12 for W_U is `flow.py`'s W_θ term by term (tested; the
    reciprocal reading is not). His Eq. 13-59 Δ(cot β) is therefore our Δ(tan β_out).
  - The cos β of Eqs. 13-57 and 13-58 is NOT cos β₂₀₀₀. Eq. 13-57 is Stokes' theorem on the
    last element (the book says so); its 1/cos β is the wall length per Δm, which in the
    chapter's own convention is 1/sin β₂₀₀₀ (Eq. 13-3), i.e. 1/cos β in ours, and Eq. 13-58's
    W_m/(cos β0 + cos β1) is half a speed only for a meridional angle. Read literally as
    cos β₂₀₀₀ = sin β, both would divide by zero for an axially leaving blade. Measured on the
    fixed-angle verification solution: the discrete Stokes balance of the blade elements has
    median 0.992 (interior) and 0.993 (rear half) with ds = Δm / cos β, 0.298 and 0.232 with
    Δm / sin β; it fails only on the LE / TE transition elements (2.0, 6.6), the known
    transition defect (not revisited). Production uses cos of OUR wall angles at the TE station.
  - The correction (Eqs. 13-57, 13-59, planar cascade, r constant and cancelling; Δm / S and
    the speed ratio dimensionless, no c_ref): Δ tan β_out = 2 (W0 − W1) Δm / (S (cos β0 + cos β1)
    ∫₀¹ W_m dη), W0, W1 the η = 0 / η = 1 wall speeds, S and β0, β1 at the TE station, the
    integral at the far exit station (the boundary whose angle moves; trapezoidal rule, our
    choice). Frames: Δ(rC_θ) = r ΔW_θ at one radius, so it moves the relative angle; stationary
    here. Sign: W0 > W1 → β_out up (an SS faster at the TE is excess circulation, the rear
    stagnation point on the PS). Verified against the solver's own response: at frozen density
    dR_K/dβ_out = −471 m/s per rad at κ_te (R_K = +48.4 m/s) and −547 at κ_te + 10° (R_K =
    −39.8 m/s), and the first corrections are +0.718° and −0.514°, both toward the root found by
    an independent secant on R_K(β_out) (23.5916° at that density). Zero residual: the step is
    exactly zero and the angle stays bit for bit.
  - Damping: β_out(n+1) = arctan(tan β_out(n) + λ_K Δ tan β_out). Aungier recommends damping
    and gives no value; λ_K is ours, `IterationSettings.kutta_relaxation`, default
    `config.KUTTA_RELAXATION` = 1.0, chosen by this study (stationary case, λ_ρ = 0.5, all
    tolerances 1e-8):

    | λ_K | first angle | passes | R_K per pass | Kutta | density | β_out |
    |---|---|---|---|---|---|---|
    | 1.0 | κ_te | 135 | 0.8735 | monotonic | monotonic but one early rise | 23.4684347° |
    | 0.5 | κ_te | 280 | 0.937 | monotonic | monotonic but one early rise | 23.4684347° |
    | 0.25 | κ_te | 568 | 0.9686 | monotonic | monotonic but two early rises | 23.4684347° |

    The rate is 1 − 0.1265 λ_K: the undamped correction is stable and monotonic, and damping
    only slows it, so 1.0 is the baseline. λ_ρ = 1.0 with λ_K = 1.0 converges in 136 passes to
    the same angle: the Kutta part sets the rate.
  - The coupled pass: resident (ρ(n), β_out(n)) → `rebuild_exit(grid(0), β_out(n))` (only the
    exit nodes move; stations, η, the blade run and every inlet and blade node checked equal,
    the grid checked admissible, else `ExitRebuildError`) → `build_metrics`,
    `build_boundary_conditions` (rebuilt for every angle: the angle changes the metrics and
    A .. E of the exit stations 95 to 146 and, through D, the last blade station 94; the
    boundary relations are equal but belong to their grid) → the density pass → R_K, ε_K of
    W(n) → β_out(n+1). Converged when the density change ≤ `density_tolerance`, the velocity
    change ≤ the velocity tolerance AND ε_K ≤ `kutta_tolerance` on one pass (never pass 0);
    then one evaluation at (ρ(n+1), β_out(n+1)) is reported, its ε_K from its own flow. Pass
    limit min(`max_density_iterations`, `max_kutta_iterations`); `CoupledNotConvergedError`
    beyond it. `velocity_tolerance` (new, optional, None = `density_tolerance`) settles the old
    open decision without choosing a number.
  - Verification case (the density-coupled one, angle left free): air R 287.05, γ 1.4, Pt
    101 325 Pa, Tt 288.15 K, ω = 0, 33°, n_passage 43, n_η 21, c_ref 0.1 m, inlet 45°,
    ṁ′ = 8.3822 kg/(m s). First angles κ_te − 10, −5, 0, +5, +10° (7.92 to 27.92°): 142, 140,
    135, 118, 134 passes, all at 23.46843468° to 23.46843481° (spread 1.3e-7° over the three in
    the tests); densities within 5.3e-10, speeds within 9.1e-9 of each other; R_K never changes
    sign (positive from below, negative from above), |R_K| falls every pass (0.873 to 0.918,
    the last 20 at 0.8735). The density change has one transient rise (pass 6 or 7 from below,
    none from above), the velocity change one from above (pass 3); both then fall at the Kutta
    rate. Final (first angle κ_te): ε_K 8.3e-9, R_K 5.8e-7 m/s, W_SS = W_PS = 70.7568 m/s at
    station 94 (m = 0.82764 c, 0.01103 c ahead of the chord-axis TE; SS node j = 0, PS node
    j = 20), p_SS − p_PS = −5e-5 Pa; last density change 3.8e-11, velocity change 5.2e-10;
    ρ 1.13179 to 1.21034 kg/m³; W_m 46.53 to 105.04, W_θ 5.09 to 113.90, W 52.76 to 134.33 m/s;
    M_rel 0.1554 to 0.4010 (peak at the first blade station, node (52, 2)); T 279.170 to
    286.765 K; p 90 697 to 99 630 Pa; ψ −0.0438 to 1.0100; ‖FΨ − Q‖∞ 6.2e-16 (relative
    5.1e-16); station mass flow within 3.2e-3 (blade median 2.5e-4); far-inlet Mach 0.299995;
    far-exit W_θ / W_m = tan β_out to 1.2e-6. Not the NACA validation case, and not compared
    with any published or measured angle. The fixed-angle solver reproduces its 2026-11-13
    result unchanged (28 passes; its tests unchanged) and now also reports R_K = +48.38 m/s
    (ε_K 0.657) at κ_te as a diagnostic.
  - Kutta alone at a frozen density (the converged fixed-angle density, flows by direct module
    calls): from κ_te and κ_te + 10°, R_K contracts by 0.8772 to 0.8773 per undamped step, keeps
    its sign, and β_out closes on the independent root from its own side (3.6 % and 3.9 % of the
    start distance after 25 steps).
  - Mutation check (one change each, `test_kutta.py` + `test_solver.py`):
    twenty-nine mutants, all caught: in kutta.py the sign of R_K, the SS taken as the eta = 0
    wall regardless of wall_identity, the station l + 1, sin for cos, no factor 2, the sign of
    the correction, eps_K over the larger speed, the integral at the TE station, the update on
    beta instead of tan(beta), no zero-step shortcut (caught after two angles with an inexact
    arctan(tan) round trip were added to its test), the relaxation ignored, NaN at zero speed, the
    speed from W_m alone, the wall difference taken from R_K, S left out; in solver.py the angle
    not fed forward, the final evaluation at the old angle, stale metrics, the density stop test
    alone, the larger pass limit, lambda_rho used as lambda_K, no rebuild check, the signed R_K in
    the history, no final evaluation, a fixed angle that moves, no rebuild at all; in
    convergence.py the velocity tolerance ignored, "or" in the coupled test, a NaN eps_K passing.

- **Inlet closure, postprocessing, live monitor and the NACA validation runner (2026-11-13;
  `flow.py`, `postprocess.py`, `plots.py`, `monitor.py`, `solver.py`, `config.py`,
  `scripts/validate_naca65_12a10_10.py`).** No governing equation, difference formula,
  assembly, linear solve, density or Kutta algorithm changed; `geometry.py`, `grid.py`,
  `metrics.py`, `coefficients.py`, `discretization.py`, `boundary_conditions.py`,
  `matrix_assembly.py`, `linear_solver.py`, `eos.py` and `kutta.py` are byte-identical.
  - ṁ′ from the inlet Mach number: `flow.inlet_state` finds W₁ as the root of W/a(W) = M₁ by
    bisection (to adjacent floats) on `eos.static_state_from_velocity`'s sound speed and takes
    T₁, p₁, ρ₁, a₁ from the same call; W_m1 = W₁ cos β₁, W_θ1 = W₁ sin β₁, q₁ = ½ρ₁W₁², ṁ′ =
    ρ₁ W_m1 S₁ with S₁ = c_ref · grid.pitch. Against the closed form (T₁ = Tt/(1 + 0.2M²),
    p₁ = Pt (T₁/Tt)^3.5, ρ₁ = p₁/(R T₁), W₁ = M₁√(γRT₁)) at M₁ 0, 0.3, 0.65: equal to the last
    bit; M₁ = 0 gives ṁ′ = 0, which the solver refuses (`coefficients.py`'s policy). At the
    verification case the derived ṁ′ equals the old explicit one bit for bit, so the fixed-angle
    and coupled results are unchanged (28 and 135 passes, the same arrays).
  - API: `solve(cascade, config, *, c_ref, mass_flow_per_depth=None, live_monitor=False,
    verbose=False, monitor=None)`, the same keywords on `solve_density_coupled` and
    `solve_at_exit_angle`. `FlowConditions.inlet_mach` may be None (a pure override). Both given:
    they must agree to `MASS_FLOW_CONSISTENCY` = 1e-12 relative, else `SolverInputError`;
    `Solution.mass_flow_source` says which path was taken and `Solution.inlet` keeps the inlet
    state.
  - Postprocessing: surfaces on the blade run only, SS / PS from `grid.wall_identity` (a wall
    whose surface changes is refused); x/c the blade-frame x of `geometry.py` (projection on the
    LE → TE chord; the upper wall moved back one pitch), checked against turning the nodes back
    by the stagger to 1e-14; m/c kept apart. The pressure coefficient follows the source:
    Dunavant's S = (P₁ − p_l)/q₁, q₁ = ½ρ₁V₁² (RM L55I08, list of symbols), default; (p − p₁)/q₁
    optional; S + Cp = (Pt₁ − p₁)/q₁ = 1.0227 at M₁ 0.3. The discharge angle is the grid's; the
    far-exit velocity angle agrees to 1.1e-14° (fixed angle, straight exit line) and 5.9e-5°
    (coupled, the finite-difference slope of the turned exit line). On the M₁ 0.3 case the wall
    Mach peaks are 0.3849 (SS, x/c 0.21) and 0.2489 (PS), while the field maximum 0.4010 lies
    at node (52, 2), two rows off the suction wall at the first blade station.
  - Monitor: observers get the frozen PassRecord of each pass; History's lists are now appended
    from that record (index n of every list is pass n; tested for both loops). One figure per
    solve, made in `start`; Line2D data set every pass; redraws throttled to `min_interval`
    (0.5 s; forced at start and finish): a redraw of the four axes costs 0.07 s with the fixed
    layout and 0.15 s with constrained layout, and the 135-pass coupled solve takes 6.6 s
    without the monitor, 39 s redrawing every pass, 10 s throttled (20 redraws, Agg). Results
    with and without a monitor, or with an observer that raises, are identical. Matplotlib is
    imported only for `live_monitor` (or `plots`); importing or running the solver does not.
  - The NACA case, from the sources: Aungier 2003 Fig. 5-6 (α 12°, β_in 45°, M_in 0.65; predicted
    β_out 24.4°, experiment 25.0°, p. 92) and Dunavant et al. 1955 Fig. 10 (β₁ 45°, σ 1.0, α 12.0°;
    (b) M₁ 0.650, θ 19.9°). σ = 1.0 is from Dunavant (Fig. 5-6 does not print it); the stagger
    33° = β₁ − α from Dunavant's definitions of both angles. γ 1.4 for air; Pt, Tt and the chord
    are a numerical dimensionalization. The runner at that condition: ṁ′ 15.0211 kg/(m s)
    (T₁ 265.70 K, p₁ 76 280 Pa, ρ₁ 1.00015 kg/m³, W₁ 212.40 m/s); the coupled solve stops at
    pass 9, TransonicFlowError, M_rel 1.0028 at node (52, 2) (β_out 21.82° at that pass, still
    moving; wall Mach ≤ 0.893). The same at 0.63 and 0.64 (passes 19, 12; first blade station),
    from a first angle near the converged one, and on 87 × 41 (where 0.62 already stops); 0.62
    and 0.60 converge on 43 × 21 (peaks 0.976 and 0.919). So no converged prediction and no
    validation exist yet; the runner reports the stop and plots the last pass labeled NOT
    CONVERGED (exit status 2).
  - Mutation check (one change each; test_flow, test_postprocess, test_monitor, test_solver):
    twenty-three mutants, all caught: in flow.py sin for cos in W_m, no pitch in mdot', q1 without
    the 1/2, no rotating refusal; in postprocess.py the upper wall not moved back a pitch, the SS
    taken as eta = 0 regardless of wall_identity, the sign of S, p1 for Pt1 in S, one extra
    station, the exit angle at the inlet, no identity check, no q1 check, the discharge angle from
    the velocity, x/c as the axial distance; in monitor.py a new line each pass, a hard-wired
    tolerance line, no forced final redraw; in solver.py observer errors propagating, no finish on
    an error, no consistency check, the pitch without c_ref, no zero-flow refusal, a History list
    from another record field.

- **User-controlled full solver runner (2026-11-13; `controller.py`, `scripts/run_full_solver.py`).**
  No numerical module changed (no method, tolerance, relaxation, Kutta or TE treatment); the run
  layer only calls the production API.
  - Timing: `time.perf_counter()` and `time.process_time()` read immediately before and after the
    solver call (`solver.solve` or `solver.solve_density_coupled`), so the solver time holds every
    pass (density, Kutta, exit rebuild, assembly, linear solve, velocity recovery, EOS) and not the
    case construction, figures, report or files. effective_cpu_cores = process CPU time / wall
    time, an average in busy-core equivalents (never "cores used"); `os.cpu_count()` is reported
    as the logical CPUs available. The live monitor and the console lines run inside the call;
    their time is measured (a timing wrapper around the observers) and shown beside the solver
    time, not hidden in it. Measured on the verification case (33°, M₁ 0.3, 135 passes): 7.0 to
    8.1 s wall, 0.97 to 0.99 effective cores, 20 logical CPUs.
  - No thread limit. threadpoolctl is not installed, and the environment route changes the
    answer: with OMP_NUM_THREADS = OPENBLAS_NUM_THREADS = MKL_NUM_THREADS = 1 the same case gives a
    different result in its last bits (β_out 23.468434679798° against 23.468434679791°; the
    field hashes differ, each setting is deterministic run to run). The brief allowed a thread
    control only if answer-neutral, so none is offered; the variables present are recorded in
    `run_summary.json`.
  - Pressure figure: Dunavant's S = (Pt₁ − p)/q₁, labeled "S = (Pt1 - p) / q1 [Dunavant et al.
    1955]". Dunavant's P₁ is the upstream TOTAL pressure (capital P, his list of symbols), so the
    label states Pt1; (p₁ − p)/q₁ with the static p₁ would be the negative of the static-reference
    coefficient (`postprocess.INLET_STATIC`), a different quantity, not used for the S figure.
  - Inputs: only the production inputs, flat (CascadeSpec, FlowConditions, GridSettings,
    IterationSettings, the gas, c_ref); the pitch, the angle of attack, the inlet state, the mass
    flow, the exit angle used and the pass limit are derived and printed as such. Defaults of the
    script: the Fig. 5-6 case with the validation runner's numbers (tolerances 1e-8, 600 passes).

- **The sonic stop made switchable (2026-11-13, user request; `config.py`, `solver.py`,
  `controller.py`, `scripts/run_full_solver.py`).** `IterationSettings.stop_at_sonic`, default
  True (the verified behaviour, every earlier test unchanged); False lets the loop go on past
  M_rel >= 1 for observation, with one `SonicFlowWarning` and the limitation string marked
  "SONIC STOP OFF". No clipping, no transonic model: the numerics are untouched. An error of a
  called module in the middle of a run now keeps its type and carries `.solution` (the last
  completed pass) and `.failed_pass`, so a run that runs away still has its history, report and
  figures. Observed: at M1 0.7 on 43 x 21 the loop passes sonic at pass 5 and ends at pass 30
  with T <= 0 (peak M_rel 3.2 to 3.7); the NACA case at M1 0.65 on 61 x 171 (lambda_rho =
  lambda_K = 0.7) holds a peak M_rel near 1.14 at the first blade station (walls <= 0.90) while
  the changes fall to about 1e-6 by pass 90, then the density and velocity changes grow again,
  the peak Mach runs away after pass 130 and T <= 0 stops it at pass 157 (eps_K reached 1e-7 at
  pass 133, beta_out 22.47 deg). The subsonic density iteration is unstable once a node is
  supersonic, as the method expects.

- **Diagnosis: why M₁ = 0.60 converges and M₁ = 0.65 does not (2026-11-13;
  `scripts/diagnose_mach060_vs_065.py`, results in `outputs/mach060_vs_065_diagnosis/`).** No
  numerical module changed; a defect is identified and NOT fixed. New plumbing only:
  `initial_density` on `solve` / `solve_density_coupled` and on `controller.run_case`, and an
  optional observer method `inspect(PassFields)` (read-only views of a pass's arrays).
  - Matched baseline (the user's runs): NACA 65-(12A10)10, 61 × 41, λ_ρ = λ_K = 0.7, tolerances
    1e-8, 600 passes, TE-angle start; only M₁ differs. M 0.60: 214 passes, β_out 22.787°, field
    peak 0.9504 at node (f, 5) (f the first blade station), suction-wall peak 0.8173. M 0.65:
    first M ≥ 1 at pass 5, node (f, 4) (production stops there); with the stop off it diverges
    (T ≤ 0 at pass 96). Start density ρ_t = 1.22501 kg/m³ for both (uniform; 19.0 % and 22.5 %
    above the inlet static density); ṁ′ by hand equals the controller's to 2e-16; far-inlet Mach
    of the converged M 0.60 field 0.599992.
  - Not the cause (each tested on the same baseline): the start density (total, inlet-static and
    the converged M 0.60 field all diverge; first sonic pass 5, 3, 0); the start angle; both
    together; the Kutta coupling (fixed angles 19.7°, 21.8°, 22.787°, 24.4° all diverge, first
    sonic pass 5; M 0.60 at its fixed angle converges to the same field); λ_ρ (0.1 to 1.0: first
    sonic pass 39, 12, 7, 5, 3, all diverge; M 0.60 identical for all five); the grid (43 × 29
    and 87 × 59 fail the same way, always at station f, η ≈ 0.10 to 0.13); the thickness
    variant; the mass-flow closure.
  - Stability map (sonic stop off; cold and warm starts agree): strictly subsonic up to M₁ 0.615
    (peak 0.996); a converged field WITH a supersonic node at 0.6175 to 0.6325 (peak 1.005 to
    1.061); no convergence in 600 passes at 0.635; divergence at 0.64. The density map alone
    (fixed angle) contracts per pass by 0.39, 0.56, 0.80, 0.88, 0.92, 0.97, 0.98 at M₁ 0.30,
    0.50, 0.60, 0.61, 0.62, 0.63, 0.6325: the unrelaxed factor g = (r − 0.3)/0.7 is about
    0.86 M_peak² and reaches 1 near M_peak ≈ 1.08, M₁ ≈ 0.637. So the lagged-density iteration
    loses stability where its own theory says, at a local Mach number slightly above 1; the
    question is why a node is there at all.
  - The node. At station f the Mach number bulges over η ≈ 0.03 to 0.4 (0.83 to 0.95 at M 0.60)
    while station f − 1 has 0.42 to 0.62 and f + 1 has 0.67 to 0.71: smooth across η, one
    station wide in m, already present on a single frozen-density pass (0.7145 at (f, 3) against
    a wall peak 0.673) and then amplified by the density feedback (from pass 0 to the first sonic
    pass at the critical node: M × 1.43 = W × 1.36 / a × 0.955; ṁ′/ρ × 1.53 as ρ falls
    1.225 → 0.799, ψ_η × 0.93, G_θ × 0.87; S and tan β constant).
  - The defect. `flow.py` takes ψ_m at the interior nodes of f (and f − 1, l, l + 1) by a central
    difference across the edge; `metrics.py`'s tan β at f is the blade-side slope. At the
    critical node the η-line of that difference has slope 2.50, the metric 1.86; ψ_m is 4.52 by
    production and 7.54 by a blade-side three-point difference; recomputing the velocity at f
    with the latter gives M 0.798 instead of 0.949 (M 0.60) and 0.851 instead of 1.016 (M 0.65,
    first sonic pass). D and E are anomalous at the same stations (D −3556 against +10204
    same-side at the critical node; identical at f + 1).
  - Causality (diagnostic variants patched in the script's worker processes, production files
    untouched). V = same-side ψ_m in the velocity recovery at f − 1, f, l, l + 1, nothing else:
    M 0.60 converges with the field peak ON the suction wall (0.8162, production wall 0.8173,
    β_out 22.762° against 22.787°); M 0.65 converges subsonically, peak 0.9060 on the suction
    wall at x/c 0.22, β_out 22.697°; the same on 43 × 29 (0.9117) and 87 × 59 (0.9068); the
    leading-edge pair alone suffices (0.9060). With V the wall reaches M = 1 between M₁ 0.68
    (0.969) and 0.70 (1.027, converged with a sonic wall), 0.72 diverges: the genuine subsonic
    limit of this cascade (Dunavant's Fig. 10 labels the local peak 0.88 at M₁ 0.650 and 1.01 at
    0.708). The same-side D, E variants (at the leading edge only, or at all four stations)
    diverge on pass 0 or 1 at a wall node: no usable discretization, so nothing is concluded
    about D and E beyond the audit.
  - Fig. 5-6's pressure-side leading edge: our first pressure-side wall node sits at x/c 0.0047
    (61 × 41; 0.0102 and 0.0030 on the other grids) on the blunt nose next to the stagnation
    point, M 0.27 (0.32, 0.23), and the plateau 0.49 follows from the second node on; Aungier's
    curve starts near the inlet Mach number. The walls are grid-converged (suction peak 0.822,
    0.817, 0.816; pressure plateau 0.489 to 0.494). A difference of leading-edge resolution, not
    of the flow.
  - Geometry: the camber line, the forward 60 % of the thickness and R_LE = 0.687 % c are
    Aungier's Table 4-1; the default aft thickness is the Kovach–Sandercock taper (R_TE 0.8 % c),
    not the plain section of the 1955 test (emery_base), and the C2 nose fair is this project's.
    Neither is demonstrably Aungier's Fig. 5-6 geometry; neither touches the leading-edge
    finding (emery_base fails identically; β_out differs by 0.39°).
  - Verdict: LE / transition discretization (the velocity recovery), amplified by the
    density-lag instability once the spurious node passes M ≈ 1.07. Initialization, Kutta
    coupling, relaxation, mass flow: ruled out. A true transonic requirement: not at M₁ 0.65
    (from about 0.69). Recommended single next step: region-aware ψ_m in `flow.py`'s velocity
    recovery at the four transition stations.
  - A mistake of the diagnosis itself, found and corrected: the first what-if patches stayed in
    their pooled worker processes and contaminated five later rows (three cold refinement runs,
    two contraction runs). Every case now restores and checks the production functions; the
    affected groups were re-run; `tests/test_diagnose_script.py` pins the restore.

- **flow.py, ψ_m by m region: the LE / TE transition fix (2026-11-13), implemented, under
  review.** The one change the diagnosis above recommended; no other numerical module
  changed (`geometry.py`, `grid.py`, `metrics.py`, `coefficients.py`, `discretization.py`,
  `boundary_conditions.py`, `matrix_assembly.py`, `linear_solver.py`, `eos.py`, `kutta.py`
  bit for bit; D and E still central across the edges, parked).
  - The rule. ψ_m is `metrics.py`'s own region derivative (`metrics._runs`, `metrics._d_dm`):
    the three m regions upstream periodic [0, f), blade [f, l], downstream periodic
    [l + 1, n_m); central inside each, the three-point one-sided difference at its ends, so
    backward at f − 1, forward at f, backward at l, forward at l + 1 (and the far stations
    forward / backward as before, bit for bit); ψ_m = 0 on the blade walls takes precedence; no
    affine jump in ψ_m (the +1 periodic jump is ψ_η's). ψ_η, the wall rule, the far formulas
    and every node off the four stations unchanged bit for bit. The PDE transition rows of
    `discretization.py` are not used. One convention for both users: tan β and ψ_m at a
    transition station now come from the same side, so W_θ/W_m = tan α holds to 1e-12 at
    f − 1 and l + 1 for a uniform flow (before: off by the jump).
  - Equivalence with the diagnosis. The production runs (stop on) reproduce the saved what-if
    V runs bit for bit, fields, exit angle, pass count and the per-pass traces of f − 3 to
    f + 5: M₁ 0.60, 0.65, 0.68 on 61 × 41, 0.65 on 43 × 29 and 87 × 59, and 0.70 with the stop
    off (`scripts/verify_le_velocity_fix.py`, `outputs/le_velocity_fix_verification/`). The
    what-if V patch on top of the fixed production is a no-op (`test_diagnose_script.py`).
  - Acceptance (61 × 41, λ_ρ = λ_K = 0.7, tolerances 1e-8, sonic stop ON). Critical node
    (f, 4): M 0.949 → 0.733 (M₁ 0.60; ψ_m 4.52 → 5.98, W_θ 241.6 → 184.9 m/s, ρ 0.810 →
    0.949 kg/m³); at M₁ 0.65, pass 5, M 1.016 → 0.788, converged 0.801. The bulge at f is
    gone (max M at f 0.950 → 0.737 at M₁ 0.60; f − 1 / f / f + 1 now 0.688 / 0.737 / 0.719).
    M₁ 0.65 converges subsonically in 216 passes: field peak 0.9060 on the suction wall at
    x/c 0.22 (node (f + 9, 0)), pressure-wall peak 0.536 at x/c 0.055, TE 0.403 on both walls
    (the Kutta condition), β_out 22.697°. First pressure-side nodes (x/c, M): (0.0047, 0.295),
    (0.0204, 0.518), (0.0375, 0.534), (0.0546, 0.536), (0.0718, 0.533). Grid study at M₁ 0.65:
    peak 0.9117, 0.9060, 0.9068 on the suction wall (43 × 29, 61 × 41, 87 × 59), β_out
    23.72°, 22.70°, 23.66° (the scatter stays open). Mach sweep: 0.60, 0.625, 0.65, 0.675,
    0.68 converge subsonically (peak 0.816, 0.860, 0.906, 0.958, 0.969, always (f + 9, 0)); 0.70
    stops at pass 11 at M 1.0019 at (f + 9, 0), the suction wall: the first genuinely sonic
    case (stop off it converges with a sonic wall, 1.027, 511 passes). The user's runner
    (`scripts/run_full_solver.py`, its own constants, M₁ 0.65) converges with no workaround.
  - Changed regression values (the verification case of `test_solver.py`, 33°, M₁ 0.3;
    every one because the velocity at f − 1, f, l, l + 1 changed, hence the EOS density there,
    hence the converged field; the defect is the ψ_m across the transitions):

    | Test | Quantity | Before | After |
    |---|---|---|---|
    | `test_solver.py` | coupled passes from three start angles | 140, 135, 134 | 143, 138, 138 |
    | `test_solver.py` | coupled β_out | 23.4684347° | 23.5022008° |
    | `test_solver.py` | last 20 Kutta ratios | 0.87350 to 0.87362 | 0.87673 to 0.87685 |
    | `test_solver.py` | all Kutta ratios | 0.873 to 0.918 | 0.873 to 0.902 |
    | `test_solver.py` | ε_K of the reported flow | 7.8e-9 to 8.3e-9 | 7.8e-9 to 8.7e-9 |
    | `test_solver.py` | λ_ρ 1 against 0.5 (ρ, W) | 7.5e-9, 5.7e-9 | 8.0e-9, 7.2e-9 |
    | `test_solver.py` | fixed-angle R_K, ε_K | 48.378526603618205 m/s, 0.6567739456255879 | 48.34643594393599 m/s, 0.6563355777339204 |
    | `test_controller.py`, `test_monitor.py` | fixed-angle passes | 28 | 27 |
    | `test_monitor.py` | coupled passes | 135 | 138 |

    No tolerance was loosened. `test_kutta.py`'s far-exit slope comment was corrected (measured
    8e-15 at the fixed angle κ_te; the 1.2e-6 quoted there was a coupled-case figure), its
    tolerance unchanged. Not asserted, updated in a docstring: the verification case at M₁ 0.7
    with the stop off now ends with T ≤ 0 after 172 passes (30 before); with the stop on it
    stops at pass 13 at M 1.0012 on the suction wall (node (58, 0) = (f + 6, 0)).
  - Tests (`test_flow.py`, six new): a quadratic per region with jumps is differentiated exactly
    (2.7e-14) and the old central difference is off by the jump; each transition stencil by
    name, bit for bit; a transition station reads its own side only; identical to
    `metrics._d_dm` over `metrics._runs`; second order at the four stations (orders 1.84 to
    2.05, 1.93 to 2.02 on the last two halvings); the uniform flow at f − 1 and l + 1. Nine
    mutants (central at f, f − 1, l, l + 1; the stencil reversed at f and at l; the wall zero
    removed; the LE region edge one station too far; first-order one-sided at the four
    stations) are each caught by 3 to 7 tests of `test_flow.py`.


- **grid.py section 9: geometry-driven five-level standard grids (2026-11-13), implemented,
  under review.** No governing equation, coefficient, stencil, assembly, linear solve, EOS,
  density iteration or Kutta correction changed; flow.py's region-wise ψ_m is untouched.
  Changed: `grid.py` (new section 9; `make_grid` split into the explicit lattice and a
  shared `_grid_on_stations`, the explicit path bit for bit: 72 fingerprinted arrays of C1 and
  C2 grids at 13°, 33°, 53° identical), `config.GridSettings` (n_passage, n_eta optional, a
  `level` field), `controller` (`CaseSettings.grid_level`, `Case.standard_grid`, the summaries),
  `scripts/run_full_solver.py` (`GRID_LEVEL = None`: the user's explicit grid unchanged).
  - Audit of the old rule (explicit mode, section 2): m_in = m_le − 1 c, m_out = m_te + 1 c
    exactly, dm = (m_out − m_in) / round((m_out − m_in) n_passage / (m_te − m_le)); the
    stations that cut the blade are the blade run, f the first past the nose tip. For the
    65-(12A10)10 at 33° the nose tip lies 4.524e-3 c ahead of the LE point on the suction
    surface (lines between cross it twice, the pressure surface not at all) and the Kovach tail
    end 1.35e-5 c behind the TE point on the pressure surface. The lattice is anchored at the
    far inlet, blind to that window, so the station fraction (m_f − m_LE)/dm is pseudo-random
    in n_passage: 80 of n_passage 10 … 200 are admissible, from 186 on none (dm ≤ the LE
    window). The old grids: 43 × 29 fraction 0.568, 61 × 41 and 87 × 59 0.431, 51 × 41 0.056
    (the lattice is symmetric, so the TE fraction is the same).
  - The new rule (section 9): the SS + PS-valid interval from the outline's monotone pieces
    (here exactly (m_le, m_te)); cell faces at the middles of the LE and TE windows, stations
    at the cell centres, so f − 1 / f and l / l + 1 keep equal clearances (dm − w)/2 from their
    windows at every level, at least 0.1 dm; targets dm/c = 0.02 √2^(2 − k) (0.0400, 0.0283,
    0.0200, 0.0141, 0.0100), candidates within ±10 %, nearest first; n_η − 1 = round(pitch /
    (1.8 dm)); whole-cell extensions within 0.05 c of 1 c; f, l and the topology read from the
    cut and validated (13 checks, metrics.py included). Failures: `GridGenerationError`.
  - The NACA family (kovach_sandercock, 33°, σ 1): dm/c 0.04004, 0.02803, 0.02002, 0.01425,
    0.01001 (ratios 1.429, 1.400, 1.405, 1.424), n_m × n_η 71 × 15, 102 × 21, 142 × 29, 201 × 40,
    284 × 56; f 25, 36, 50, 71, 100; l 45, 65, 91, 129, 183; (m_f − m_LE)/c 0.01776, 0.01175,
    0.00775, 0.00486, 0.00274 and (m_TE − m_l)/c 0.02002, 0.01401, 0.01000, 0.00712, 0.00500,
    both strictly falling; extensions 0.981 … 1.007 c; every level the first candidate tried.
    emery_base gives the same stations up to the TE window (none there).
  - Timing (time.perf_counter, repeated; never solved): one level about 0.13 s (median 0.13 to
    0.14 s), of which about 0.11 s is the blade's passage check of section 4, about 0.02 s the
    search with its intersections, construction and validation; all five 0.21 s. Serial
    projection at the medium level (median 0.129 s): 100 blades about 13 s, 1,000 about 2.1 min,
    10,000 about 21 min; 2.4 % of a very_coarse solve's time and 0.3 % of a very_fine one's.
  - Perturbed blades (stagger 25°, 40°, 50°, cl0 0.8, 1.6, t/c 0.06, 0.12, 0.15, σ 0.7, 1.5,
    stagger −10° with cl0 0.4, stagger 0°): every level builds but where the nose window is
    too wide for the level: 60° stagger (window 0.0136 c) fails fine and very_fine, 55° with
    t/c 0.15 (0.019 c) medium and finer; σ 3 fails very_coarse (5 η cells). Each is a
    `GridGenerationError` naming the level, the geometry and every candidate's reason.
  - Solver study (M₁ 0.65, 61 × 41's settings: λ_ρ = λ_K = 0.7, tolerances 1e-8, stop on), very
    coarse to very fine: converged in 116, 142, 172, 211, 254 passes (5.3 to 43.7 s); β_out
    25.478°, 24.372°, 23.570°, 22.683°, 23.282°; ε_K 7.6e-9 to 8.9e-9; field peak 0.904, 0.904,
    0.905, 0.904, 0.903, always on the suction wall (x/c 0.21 to 0.23); pressure-wall peak 0.525
    → 0.571 moving to the nose (x/c 0.17 → 0.024); station mass-flow error 3.4e-3 → 1.1e-3.
    β_out: non-convergent (open decision above). Transition D at f − 1 (max |D|): 983, 1805,
    3110, 5152, 8592, growing as dm^−1.45 … −1.70 (at f as dm^−1.26 … −1.40; at l − 1 and l + 1 sharply once l enters the TE circle), with f − 1 and f negative and f + 1 positive
    at every level: the D/E anomaly is systematic and grows under refinement (parked). Velocity:
    no return of the large spike at f (max M at f − 1 / f / f + 1, very fine: 0.757 / 0.810 /
    0.763; it was 0.628 / 0.950 / 0.743 at M₁ 0.60 before the ψ_m fix); a mild local maximum at
    f remains at the three finer levels and does not grow (0.849 → 0.810).
  - Tests: `test_grid_family.py` (23: levels, refinement, topology against an independent
    40001-point sampling of the curves, lattice, extensions, reproducibility, level = family
    member = solver grid, nine perturbed blades, three structured failures, timing fields) and
    `test_grid_family_script.py` (2: the final figure's data are the grids' own arrays and the
    figure comes last). Migrated: `test_config.py` (GridSettings gains `level`),
    `test_grid.py`'s import contract (module-level imports unchanged; `metrics` allowed as the
    one function-level import of the standard-grid validation), `test_controller.py` and the
    diagnosis baseline (`grid_level=None`). Suite 1738 pass, none skipped.
  - Not done: no optimizer, no clustering, no change of the Kutta station or of D/E.
- **Thickness reconstruction (2026-11-13, approved by the user; geometry.py only).** Table 4-1
  gives the thickness at 26 stations; between them geometry.py interpolated the half-thickness
  by PCHIP in z = √station. That is exact at the stations but only C1: h'' jumps at every
  station, the C2 fit keeps each jump as a curvature ramp 0.008–0.02 c wide, and the C2 walls
  carry a curvature sawtooth locked to the stations (0.165 /c RMS over x/c 0.20–0.85, the
  |dκ/ds| peaks within ±0.0008 c of the stations) that the potential flow turns into the
  suction-side Mach ripple at M₁ 0.65 (`scripts/audit_ss_sawtooth.py`, then
  `scripts/audit_naca_thickness_reconstruction.py`, outputs/naca_thickness_reconstruction_audit/).
  - The choice: `CascadeSpec.thickness_reconstruction`, `"quintic_natural_sqrt"` (the
    interpolating quintic spline in z with h‴ = h⁗ = 0 at both ends, the interpolant of least
    ∫h‴²; C4, exact at every station to 1.4e-17) or `"pchip_sqrt"` (the legacy PCHIP). None,
    the default, takes `DEFAULT_THICKNESS_RECONSTRUCTION[surface_continuity]`: C1 keeps the
    PCHIP, so the validated C1 baseline is unchanged bit for bit (fingerprint test; checked on
    both variants, every field); C2 takes the quintic. C2 with `"pchip_sqrt"` is the previous
    C2 bit for bit (both variants). C2 is now compared with C1 of the same reconstruction.
    The cascade metadata carry `thickness_reconstruction`, `base_max_thickness` and its
    station; the controller's case summary prints the reconstruction.
  - Audit (65-(12A10)10, both variants): the quintic stays within 8.8e-5 c of the PCHIP
    surfaces, removes 86 % of the curvature ripple, improves the LE fairness, peaks at
    t/c 0.099961 at station 0.4098 (the quadratic through the three largest ordinates:
    0.099956 at 0.409; PCHIP pins it to 0.09992 at 0.40) and is positive and single-peaked.
    Cubic splines were rejected only for pressure-side LE-table roughness (×1.3–3.4), the
    smoothing spline for not being exact.
  - What it changes (measured): the C2 fits need fewer knots (kovach_sandercock SS / PS 533 /
    576 → 400 / 443, emery_base 574 / 622 → 373 / 419), no table-station transition is left
    (only the LE fair, the Kovach start and TE, the emery_base closing segment); the Kovach
    start corner falls from 0.077° to 0.0073°. The front-60 % D error of coefficients.py
    against the exact derivative falls 4 to 12 times (0.66 → 0.15 at n_passage 101, 0.069 →
    0.0056 at 801) and converges at about second order (ratio 3.6 per halving). The robust
    LE moves more of the thin 0.4 / 0.04 blade's nose (LE zone 8.2e-4 → 1.19e-3 c from C1:
    its pressure-side anchor moves from 0.00716 to 0.00771 c of arc). Solver, M₁ 0.65, the
    five-level family of kovach_sandercock (the production blade, whose aft is the Kovach
    taper from station 0.60 and so never sees the reconstruction): β_out 25.479°, 24.372°,
    23.571°, 22.683°, 23.282° (unchanged within 0.001°; the grid drift is not the geometry),
    passes 116 … 254 as before, suction peak 0.904, 0.902, 0.901, 0.901, 0.900 (before 0.904,
    0.904, 0.905, 0.904, 0.903: the sawtooth inflated the peak on the finer grids); emery_base
    (the audit's what-if): suction-wall Mach ripple −68 % at medium, β_out −0.02 to −0.04° on
    all five levels. Low-Mach verification case: coupled angle 23.50220° → 23.50024°, fixed-angle
    R_K 48.346 → 48.329 m/s.
  - Tests: `test_geometry.py` gains the reconstruction tests (exact at the stations and the
    maxima for both, C4 against C1 at the knots, metadata, refusal of unknown names, the spec
    default) and runs the shared C1 construction (vertices on the analytic reference,
    derivatives against central differences, circles and junctions, SmoothedSurface, the C2
    ramps) for both reconstructions. Moved locks, each with old value, new value and the
    reason in its docstring: test_geometry (LE zone 1e-3 → 1.25e-3 for the measured
    1.19e-3; Kovach start corner by reconstruction), test_coefficients (D/E bounds tightened
    to the new measurements, the default case's β 79.248° → 79.164°), test_discretization
    (stencil fingerprint, LE-row rhs, corner count 101 → 100 at 401), test_grid (nose-tip
    gap, ceilings 575 / 319 → 576 / 320 at 13° / 23°, the passage check stated as the upper
    estimate its docstring promises: 4.1e-8 c above the independent search, 1e-9 only because
    a piece end sat at the PCHIP blade's widest cut), test_grid_family (nose windows),
    test_metrics (LE angle 79.248° → 79.164°), test_solver (R_K, the coupled angle; the
    sonic-off test moves to M₁ 0.72, because 0.70 no longer reaches T ≤ 0 on the quintic
    geometry). `test_naca_thickness_reconstruction.py` builds A_pchip and C_quintic_natural
    through `CascadeSpec` and registers the diagnostic candidates under their own names. The
    C1 fingerprint and every C1 lock are untouched. Suite 1772 pass, none skipped.
- **C2 geometry frozen and made the default (2026-11-14, user's decision).** The C2 geometry
  with the robust LE (monotone-curvature Whewell fair, anchor 0.5 R_LE past the existence
  boundary) and the `"quintic_natural_sqrt"` thickness is FROZEN as it stands, and
  `CascadeSpec.surface_continuity` defaults to `"C2"` (`thickness_reconstruction` None takes
  the quintic for C2, as before). The open LE tuning item, a short second curvature drop just
  ahead of the suction-side anchor (about −58 → −30 /c at the default blade), is accepted as
  is. Nothing of the construction changed: an explicit `"C2"` spec gives the same cascade bit
  for bit, and `"C1"` stays the validated legacy baseline (its fingerprint and every C1 lock
  untouched).
  - The lock: `test_c2_frozen_fingerprint` on three blades (DEFAULT_SPEC of `test_geometry.py`
    in both variants and the 65-(12A10)10 at 33°, built with the default spec): point and
    control-point counts, sums of the surface arrays and control points, X(u) and curvature at
    u = 0.002, 0.05, 0.5, the LE anchor and departure parameters and fairness, throat, area.
    Measured first: the build is reproducible run to run; a one-ulp change of t_max_over_c,
    cl0, stagger_deg or solidity moves these by at most 4.2e-11 relative (the curvatures;
    positions and sums 4e-12), so the lock allows 1e-9 (C1's 1e-13 would be tighter than the
    C2 fit's own last-bit response).
  - What had to say "C1": the tests and the one script that meant the C1 construction and
    relied on the old default now name it (`test_geometry.py` DEFAULT_SPEC, `test_grid.py`,
    `test_metrics.py`, `test_config.py`, `scripts/plot_metrics_accuracy.py`), so they test and
    plot exactly what they did before. The two tests of the default were rewritten for C2
    (the default carries the curves; an explicit C2 + quintic spec equals the default bit for
    bit, curves included). The solver path (`controller.CaseSettings`, the scripts, the
    notebooks) already named its surface continuity and is unchanged. Suite 1623 pass; the
    only other results are the tests of scripts removed from the working copy the same day.

## Looking at the geometry

`python scripts/plot_cascade.py` opens two windows. The first, "Cascade geometry", is the
cascade: the blades and their periodic neighbors, suction and pressure surfaces, actual and
equivalent camberlines with κ_le and κ_te, the pitch and the geometric throat, and
close-ups of the LE and TE; over it the passage, the lower and upper inlet extension lines
with L_up, m_in, m_le, β_in and the pitch between them, and a panel of β(m) along the
extension. The second, "Computational grid", is the `make_grid` result: both families of
grid lines, the blades, the exact LE and TE against the first and last blade-wall nodes,
the far inlet and outlet boundaries, κ_le, κ_te, β_in, the current β_out, L_up and L_down,
the grid numbers, and close-ups of the LE and the TE, each with the grid one pitch down so
that the upper wall nodes sit on the drawn blade. Each close-up shades the transition
interval between the last periodic station and the first (or last) blade station, marks the
blade's most upstream (or downstream) point inside it, the exact LE (or TE) as a dashed
landmark, the selected wall nodes and the fluid nodes, gives the automatic offset from the
landmark, and draws the straight m-links across the interval as a non-blocking diagnostic,
red where one crosses metal. The normal workflow is to edit `EXAMPLE`, `SETTINGS`,
`INLET_ANGLE_DEG` and `EXIT_ANGLE_DEG` (None: κ_te) at the top of the script; options such as
`--cl0 1.2 --stagger-deg 33 --variant emery_base --physical-chord 0.05 --inlet-angle-deg 45
--exit-angle-deg 20 --save cascade.png` override them (`--save` also writes
`cascade_grid.png`). The package itself never imports matplotlib.

`python scripts/plot_geometry_continuity.py` builds its `EXAMPLE` twice, as "C1" and "C2",
prints the preservation table and draws five figures: the overlay with LE and TE close-ups,
the normal deviation along both surfaces, the curvature, d²y/dm² (symlog, the m-turning
points shaded) and β with β(C2) − β(C1). C1 is drawn from its analytic pieces, one piece at
a time, so its jumps stay jumps; every junction of the construction is marked. `--save DIR`
writes the PNGs (`outputs/geometry_continuity/` is gitignored).

`python scripts/plot_c2_wall_handoff.py` sets the C2 grid on its curves beside the same
cascade read from its sampled arrays (the pre-hand-off path): the PCHIP's distance from
the curve with LE/TE close-ups, exact against differenced β, the exact d tan β/dm with the
curvature and |cos β| along each passage-facing contour, and the first and last wall
nodes with their tangents and surface names; it prints the edge tables. `--save DIR`
writes the PNGs.

`python scripts/audit_coefficients.py` prints the coefficient audit of the default C2
cascade: the ranges of A to E with their nodes, the first and last five blade stations,
the corner stations of the region transitions, D and E against the hand-expanded
exact-derivative reference over n_passage 51 to 801, and the chord scaling.

`python scripts/audit_discretization.py` prints the stencil audit of the same cascade:
the ranges of the Eq. 5-44 weights and of H with their nodes, the row properties, the
corner stations of the region transitions, the curvature-step check, and the expanded
form against a conservative discretization of Eq. 5-20 over n_passage 51 to 401, and the
periodic PDE rows (the LE/TE transition rows with every node, coefficient and rhs, and the
links around the transitions that cross the blade).

`python scripts/visualize_te_stencils.py --save outputs/te_stencil_diagnostics` draws the TE
topology and footprints (01 geometry and stations, 02 old against revised TE row, 03 and 04
the n_passage 401 interior row whose NE diagonal crosses the tail, 05 the revised row's
footprint per derivative, 06 the old link at four resolutions, 07 a local candidate for that
interior row, PROPOSED ONLY, 08 the straight-chord sweep) and prints the crossing table. Its
geometry is `scripts/link_geometry.py`: exact crossings of the straight chord between two
stencil nodes with the C2 curves and their pitch copies; a chord counts as a straight-chord
blade intersection when more than 1e-7 c of it is inside metal (an end on a wall not
counted). That is a geometric fact about the segment, not by itself stencil invalidity: the
differences act along the computational coordinate curves.

`python scripts/audit_transition_mapping.py --save outputs/transition_mapping_audit` audits
the computational mapping (m, η) → (m, y) across the LE and TE transitions (diagnostic only):
make_grid's boundary rule evaluated at any m, the constant-η coordinate curves against the
straight chords, the jumps and slopes of the mapping at the blade's extreme points, signed
cell areas, folds, curves or cells in metal, metrics consistency, the n_passage 401 interior
row (878, 119), the TE row's central against forward m-derivatives, and a sweep over
n_passage 31 … 401 × n_eta 21 … 121 (figures 01 … 09).

`python scripts/audit_wall_station_topology.py --save outputs/wall_station_topology_audit`
classifies the first and last blade station against the C2 curves at stagger 13° to 53°
(the LE / TE reference points, the two-root limits m2_LE = m_le and m2_TE = m_te, the
production labels against independent roots) and compares station rules that were not
adopted (redistribution, equal shifts). `python scripts/audit_redist_numerics.py --save
outputs/redist_numerical_audit` measured the rejected REDIST rule numerically (nonuniform
differences, D/E, a smooth field on the real mapping). Both are diagnostics only.

`python scripts/audit_uniform_grid_validity.py --save outputs/uniform_grid_validity`
builds the production grid at every n_passage 20 … 800 for stagger 13° to 53°, classifies
its first and last blade station three ways (wall_identity, `wall_station_topology`,
independent C2 roots; a disagreement stops it), checks the ceiling theory and
`find_valid_n_passage`, writes `uniform_grid_validity.csv` and `valid_n_passage.json`,
picks refinement sequences and evaluates metrics, coefficients and a smooth field on them
(figures 01 … 10; `01_validity_map.png` is the overview). About 11 minutes.

`python scripts/plot_transition_links.py` draws the periodic PDE rows beside the blade on
the real grid: at the TE (n_passage 61, 101, 201, 401) the first exit station's central
ψ_m link to the last blade station's upper wall node, red where it runs through the +pitch
blade's tail, with the row's other nodes; at the LE (61, 101, 401) the same link, free of
metal. `--save DIR` writes the PNG.

`python scripts/audit_boundary_conditions.py` prints the boundary audit of the same
cascade: node and relation counts, the far relations and their eliminations, the
transition stations against the nodes, the corner memberships and the far-station
dependency, the open periodic-line rows beside the transitions, and the counts over
n_passage 31 to 401.

`python scripts/plot_le_fairing.py` compares the legacy C1 LE, the previous C2
(`fit_quintic_surface`) and the robust C2 LE on three blades (65-(12A10)10, the thin
65-(12A10)04, cl0 1.6), both surfaces, with the departure and the anchor marked: the LE
overlay with close-ups at the largest change, the turning angle, κc, d(tan β)/dm in the
cascade frame (exact derivatives, |cos β| < 0.05 shaded) and the deviation from C1 along
the surface. `--continuity` adds the anchor, departure, LE deviation and fairness against
t/c and cl0. `--save DIR` writes the PNGs (`outputs/le_fairing/`, gitignored).

## Tests

Each test file's docstring lists its checks, and each check compares against something the
module does not depend on: a closed form, a hand calculation, a book equation or separate
code. `test_geometry.py` (227 tests), `test_config.py` (12), `test_grid.py` (675),
`test_metrics.py` (174), `test_coefficients.py` (54), `test_discretization.py` (110),
`test_boundary_conditions.py` (52), `test_matrix_assembly.py` (60) and
`test_linear_solver.py` (34), `test_flow.py` (29), `test_eos.py` (69), `test_kutta.py` (16),
`test_solver.py` (39), `test_postprocess.py` (14), `test_monitor.py` (9), `test_controller.py` (13) and
`test_diagnose_script.py` (10, helpers only), `test_grid_family.py` (23), `test_grid_family_script.py` (2, no
solve) and `test_naca_thickness_reconstruction.py` (8, the audit script's helpers and one very_coarse solve) run; no test
file skips itself any more.
The plot and monitor tests use Matplotlib's Agg backend: no window opens.

```text
python -m pytest tests/test_config.py tests/test_geometry.py tests/test_grid.py tests/test_metrics.py tests/test_coefficients.py tests/test_discretization.py tests/test_boundary_conditions.py tests/test_matrix_assembly.py tests/test_linear_solver.py tests/test_flow.py tests/test_eos.py tests/test_kutta.py tests/test_solver.py tests/test_postprocess.py tests/test_monitor.py tests/test_controller.py tests/test_diagnose_script.py tests/test_grid_family.py tests/test_grid_family_script.py tests/test_naca_thickness_reconstruction.py
```
