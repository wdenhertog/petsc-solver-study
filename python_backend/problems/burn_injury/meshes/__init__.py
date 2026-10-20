"""Vendored mesh utilities for the burn-injury benchmark (no external burn_injuries dependency)."""

from .data import TaggedMesh
from .generators import generate_2d_surface_mesh, generate_curved_surface_mesh
from .io import load_surface_mesh, save_mesh_xdmf
from .tags import CellTag, FacetTag

__all__ = [
    "CellTag",
    "FacetTag",
    "TaggedMesh",
    "generate_2d_surface_mesh",
    "generate_curved_surface_mesh",
    "load_surface_mesh",
    "save_mesh_xdmf",
]
