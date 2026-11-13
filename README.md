# TURBOlenza

**Axial Compressor Aerodynamics and Blade Design Framework**

TURBOlenza is a Python framework for the aerodynamic design and analysis of axial-flow
compressors. It implements the methods of R. H. Aungier, *Axial-Flow Compressors: A Strategy
for Aerodynamic Design and Analysis* (ASME Press, 2003), one module at a time, each verified
on its own before the next one is built on it.

> **Status (November 2026): early development.** The first module, the blade-to-blade
> compressible potential-flow solver, runs and is *verified*: manufactured and closed-form
> tests, module-to-module consistency, and a constructed cascade case. It is **not yet
> validated** against measurements. The comparison with the NACA 65-(12A10)10 cascade data of
> Dunavant et al. (1955) is the next step.

## Scope

The framework is planned in eight stages. Each one builds on the stages before it.

| Stage | Module | Status |
|---|---|---|
| 1 | Blade-to-blade compressible potential flow (Aungier, Ch. 5) | `turbolenza.b2b`: implemented and verified; validation in progress |
| 2 | Blade-surface boundary layer, coupled to the B2B solution | planned |
| 3 | Meridional throughflow solver | planned |
| 4 | Empirical performance models (incidence, deviation, loss) | planned |
| 5 | Endwall boundary-layer analysis | planned |
| 6 | Aerodynamic performance analysis | planned |
| 7 | Compressor stage aerodynamic design | planned |
| 8 | Quasi-three-dimensional blade-passage flow field analysis | planned |

## The blade-to-blade solver today

- NACA 65 blades (A10 meanline) in cascade: stagger, pitch, geometric throat; C1 or C2 surface
  representation
- a sheared (m, η) grid in five geometry-driven standard levels, or an explicit size
- the stream-function equation (Aungier Eq. 5-21) on a nine-point stencil (Eq. 5-44), solved by
  a sparse direct factorization
- a density loop for compressibility, coupled with the Kutta condition at the trailing edge
  (Eq. 5-34) and Aungier's exit-angle correction
- surface Mach number, pressure coefficient and discharge angle; figures and CSV / JSON export

Limits: stationary, planar cascades, subsonic relative flow (a solve stops at M = 1), no
viscous effects yet.

## Installation

With conda:

```text
conda env create -f environment.yml
conda activate turbolenza
pip install -e . --no-deps
```

In an existing environment with NumPy, SciPy and Matplotlib: `pip install -e .`

The scripts and tests also run from a plain download without installing: they put `src/` on
the import path themselves.

## Quick start

Edit the constants at the top of `scripts/run_full_solver.py` (gas, geometry, inlet flow, grid,
iteration control, output), then run

```text
python scripts/run_full_solver.py
```

The script prints the case, runs the coupled density + Kutta solve with a live convergence
window, plots the surface Mach number, pressure coefficient, convergence history and Mach
contour, and writes them to `outputs/<CASE_NAME>/<run>/`. The default case is the NACA
65-(12A10)10 cascade at M₁ = 0.65 (Aungier, Fig. 5-6).

## Tests

```text
python -m pytest
```

Some tests are bit-for-bit regression checks. Their values were recorded in the reference
environment of `environment.yml`; other platforms or BLAS builds may differ in the last bits.

## Repository layout

```text
src/turbolenza/       the package
    b2b/              blade-to-blade potential flow (stage 1) and its detailed README
tests/                pytest suite
scripts/              run scripts, validation runs, audits and diagnostic plots
validation/           measured and published reference data, with sources
notebooks/            worked examples
```

## How it is developed

- One change at a time, each checked against something independent (a closed-form result, a
  second implementation, or a measurement) before the next one builds on it.
- Verification (does the code solve the equations correctly?) and validation (do the equations
  describe the measured flow?) are kept apart and reported separately.
- Where this code departs from the book, or the book leaves a choice open, the decision and its
  reason are recorded in the module's decision log (for B2B,
  [`src/turbolenza/b2b/README.md`](src/turbolenza/b2b/README.md)).

## References

- R. H. Aungier, *Axial-Flow Compressors: A Strategy for Aerodynamic Design and Analysis*,
  ASME Press, 2003.
- J. C. Dunavant et al., NACA RM L55I08, 1955: high-speed cascade tests of the
  NACA 65-(12A10)10 blade section.

## License

BSD 3-Clause. See [LICENSE](LICENSE).
