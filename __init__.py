"""Blade-to-blade compressible potential flow after Aungier (2003), Chapter 5.

geometry.py is implemented, and so are grid.py (make_passage_boundaries, make_grid and
rebuild_exit, coordinates only, frozen) and metrics.py (build_metrics, the geometric fields
of the transformed equations). Everything else is architecture only: its functions raise
NotImplementedError and nothing here solves a flow yet. README.md has the module map, the
planned data flow and the decision log.
"""
