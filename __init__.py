"""Blade-to-blade compressible potential flow after Aungier (2003), Chapter 5.

geometry.py is implemented, and so are grid.py (make_passage_boundaries, make_grid and
rebuild_exit, coordinates only, frozen), metrics.py (build_metrics, the geometric fields
of the transformed equations), coefficients.py (build_coefficients, A to E of the
stream-function equation), the interior difference equation of discretization.py
(build_interior_stencil, Eq. 5-44) and the boundary relations of boundary_conditions.py
(build_boundary_conditions). Everything else is architecture only: its functions raise
NotImplementedError and nothing here solves a flow yet. README.md has the module map, the
planned data flow and the decision log.
"""
