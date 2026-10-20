"""Vendored mesh utilities for the burn-injury benchmark.

Loading frozen meshes needs only dolfinx. The generators (which need gmsh) are
deliberately not imported here; use `from .generators import ...` explicitly.
"""

from .data import TaggedMesh
from .io import load_surface_mesh, save_mesh_xdmf
from .tags import CellTag, FacetTag

__all__ = ["CellTag", "FacetTag", "TaggedMesh", "load_surface_mesh", "save_mesh_xdmf"]
