# potential_flow_solver

Blade-to-blade compressible potential flow after R. H. Aungier, *Axial-Flow Compressors*
(ASME Press, 2003), Chapter 5. Python standard library, NumPy and SciPy only (matplotlib for
the notebook). Clone the repository as `potential_flow_solver`, so that
`import potential_flow_solver` works from the folder that holds it.

## Status

| Module | State |
|---|---|
| `geometry.py` | NACA 65 blade and cascade. C1 surfaces: the validated legacy baseline. C2 surfaces: clamped quintic B-spline curves with a robust leading edge, under validation |
| `grid.py` | The (m, η) grid of one passage. C1 path frozen. C2 path (2026-11-11): the blade walls are the authoritative C2 curves, under review |
| `metrics.py` | S, dS/dm, tan β, d tan β/dm and β at the nodes. C2 blade walls: exact spline slopes, under review |
| everything else | interfaces only (`NotImplementedError`) |

**The solver goes with C2 only for now.** `grid.py` takes the blade-wall locations and
`metrics.py` the blade-wall angles from the C2 geometry: each wall node is a point X(u) of the
authoritative curve, and its slope and d tan β/dm are that curve's own derivatives at the same
u. No second wall representation sits in between.

**A fair scientific comparison of C1 against C2 is planned for later**, after an exact
piecewise-C1 derivative path, so that the two differ only in surface continuity. Until then
the C1 path stays as it is, the legacy baseline.

## Workflow

```python
import math
from potential_flow_solver.config import GridSettings
from potential_flow_solver.geometry import CascadeSpec, make_cascade
from potential_flow_solver.grid import make_grid
from potential_flow_solver.metrics import build_metrics

spec = CascadeSpec(blade_type="NACA65", solidity=1.0, stagger_deg=33.0, cl0=1.2, t_max_over_c=0.10,
                   surface_continuity="C2")
cascade = make_cascade(spec)                                   # blade, stagger, pitch, C2 curves
grid = make_grid(cascade, GridSettings(n_passage=101, n_eta=31), math.radians(45.0))
metrics = build_metrics(grid)                                  # S, tan beta, d tan beta / dm, ...
```

`notebooks/Grid_Metrics.ipynb` walks through this workflow and the theory behind it
(Aungier Eqs. 5-1 to 5-3, 5-9, Sec. 5.3 and Fig. 5-5 for the grid; Eqs. 5-16 to 5-26 for
what the metrics feed into), with independent checks of every step.
