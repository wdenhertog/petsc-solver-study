from dataclasses import dataclass
from typing import Any

from dolfinx.mesh import Mesh, MeshTags
from mpi4py import MPI

from .tags import CellTag, FacetTag


@dataclass(slots=True)
class TaggedMesh:
    mesh: Mesh
    cell_tags: MeshTags | None
    facet_tags: MeshTags | None
    wound_shape: str | None = None
    surface_shape: str | None = None
    surface_params: dict[str, Any] | None = None

    @property
    def tdim(self) -> int:
        return self.mesh.topology.dim

    @property
    def gdim(self) -> int:
        return self.mesh.geometry.dim

    @property
    def num_cells(self) -> int:
        return self.mesh.topology.index_map(self.tdim).size_global

    @property
    def num_vertices(self) -> int:
        return self.mesh.topology.index_map(0).size_global

    @property
    def comm(self) -> MPI.Comm:
        return self.mesh.comm

    @property
    def topology(self):
        return self.mesh.topology

    @property
    def geometry(self):
        return self.mesh.geometry

    @property
    def name(self) -> str:
        return self.mesh.name

    def to_dict(self) -> dict[str, Any]:
        """Convert mesh metadata to a dictionary."""
        data: dict[str, Any] = {
            "num_cells": self.num_cells,
            "num_vertices": self.num_vertices,
            "tdim": self.tdim,
            "gdim": self.gdim,
            "cell_type": self.mesh.topology.cell_name(),
        }
        if self.wound_shape is not None:
            data["wound_shape"] = self.wound_shape
        if self.surface_shape is not None:
            data["surface_shape"] = self.surface_shape
        if self.surface_params is not None:
            data["surface_params"] = self.surface_params
        if self.cell_tags is not None:
            comm = self.comm
            try:
                n_wound = comm.allreduce(len(self.cell_tags.find(CellTag.WOUND)), op=MPI.SUM)
                n_healthy = comm.allreduce(len(self.cell_tags.find(CellTag.HEALTHY)), op=MPI.SUM)
                data["num_wound_cells"] = n_wound
                data["num_healthy_cells"] = n_healthy
            except Exception:
                pass
        if self.facet_tags is not None:
            comm = self.comm
            try:
                n_outer = comm.allreduce(len(self.facet_tags.find(FacetTag.OUTER_BOUNDARY)), op=MPI.SUM)
                n_wound_bnd = comm.allreduce(len(self.facet_tags.find(FacetTag.WOUND_BOUNDARY)), op=MPI.SUM)
                data["num_outer_boundary_facets"] = n_outer
                data["num_wound_boundary_facets"] = n_wound_bnd
            except Exception:
                pass
        return data
