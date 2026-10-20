from pathlib import Path

from dolfinx.io import XDMFFile
from dolfinx.io import gmsh as dfx_gmsh
from mpi4py import MPI

from .data import TaggedMesh


def load_surface_mesh(
    path: str | Path,
    comm: MPI.Comm = MPI.COMM_WORLD,
    gdim: int = 2,
    model_rank: int = 0,
    wound_shape: str | None = None,
    surface_shape: str | None = None,
    surface_params: dict | None = None,
) -> TaggedMesh:
    """Load a tagged mesh from a file (.msh or .xdmf).

    For .msh files (Gmsh), expects physical group ids matching CellTag / FacetTag
    from tags.py.
    For .xdmf files, reads mesh, cell_tags, and facet_tags using XDMFFile.
    """
    path = Path(path)
    if wound_shape is None:
        known_shapes = {"circle", "ellipse", "superellipse", "dumbbell", "perturbed_circle"}
        stem = path.stem
        # Check direct token match or prefix match
        for shape in sorted(known_shapes, key=len, reverse=True):
            if shape in stem:
                wound_shape = shape
                break
        if wound_shape is None and ("surface_mesh" in stem or "curved" in stem):
            wound_shape = "circle"

    if path.suffix == ".msh":
        mesh_data = dfx_gmsh.read_from_msh(str(path), comm, model_rank, gdim=gdim)
        return TaggedMesh(
            mesh=mesh_data.mesh,
            cell_tags=mesh_data.cell_tags,
            facet_tags=mesh_data.facet_tags,
            wound_shape=wound_shape,
            surface_shape=surface_shape,
            surface_params=surface_params,
        )
    elif path.suffix == ".xdmf":
        with XDMFFile(comm, str(path), "r") as xdmf:
            try:
                mesh = xdmf.read_mesh(name="mesh")
            except Exception:
                mesh = xdmf.read_mesh()

            tdim = mesh.topology.dim
            mesh.topology.create_connectivity(tdim, tdim)
            mesh.topology.create_connectivity(tdim - 1, tdim)
            mesh.topology.create_entities(tdim)
            mesh.topology.create_entities(tdim - 1)

            cell_tags = None
            for ct_name in ["cell_tags", "CellTags", "cells"]:
                try:
                    cell_tags = xdmf.read_meshtags(mesh, name=ct_name)
                    break
                except Exception:
                    pass

            facet_tags = None
            for ft_name in ["facet_tags", "FacetTags", "facets"]:
                try:
                    facet_tags = xdmf.read_meshtags(mesh, name=ft_name)
                    break
                except Exception:
                    pass

            return TaggedMesh(
                mesh=mesh,
                cell_tags=cell_tags,
                facet_tags=facet_tags,
                wound_shape=wound_shape,
                surface_shape=surface_shape,
                surface_params=surface_params,
            )
    else:
        raise ValueError(f"Expected a .msh or .xdmf file, got {path.suffix} instead.")


def save_mesh_xdmf(
    tagged_mesh: TaggedMesh,
    path: str | Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with XDMFFile(
        tagged_mesh.mesh.comm,
        str(path),
        "w",
    ) as xdmf:
        xdmf.write_mesh(tagged_mesh.mesh)

        tagged_mesh.mesh.topology.create_connectivity(
            tagged_mesh.tdim - 1,
            tagged_mesh.tdim,
        )

        if tagged_mesh.cell_tags is not None:
            xdmf.write_meshtags(
                tagged_mesh.cell_tags,
                tagged_mesh.mesh.geometry,
            )

        if tagged_mesh.facet_tags is not None:
            xdmf.write_meshtags(
                tagged_mesh.facet_tags,
                tagged_mesh.mesh.geometry,
            )
