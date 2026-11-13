"""TURBOlenza: Axial Compressor Aerodynamics and Blade Design Framework.

The methods follow R. H. Aungier, *Axial-Flow Compressors: A Strategy for Aerodynamic Design
and Analysis* (ASME Press, 2003).

Subpackages
-----------
b2b
    Blade-to-blade compressible potential flow (Aungier, Chapter 5): cascade geometry, grid,
    stream-function solver with the density and Kutta loops, post-processing.

Importing ``turbolenza`` imports no subpackage; import the one you need, e.g.
``from turbolenza.b2b import controller``.
"""

__version__ = "0.1.0.dev0"
