import inspect
import logging
from collections import defaultdict, deque
from collections.abc import Callable
from pathlib import Path
from typing import Any

import dolfinx
import gmsh
import numpy as np
from dolfinx.io import gmsh as dfx_gmsh
from mpi4py import MPI

from .data import TaggedMesh
from .io import save_mesh_xdmf
from .tags import CellTag, FacetTag

logger = logging.getLogger(__name__)

# Default save location only; the benchmark's frozen meshes live in <repo>/meshes/burn.
MESH_DIR = Path(__file__).resolve().parents[4] / "meshes"


def _fmt(val: Any) -> str:
    """Format numeric values cleanly for filename generation."""
    if isinstance(val, float):
        return f"{val:g}"
    return str(val)


def normalize_domain_size(domain_size: float | tuple[float, float] | list[float]) -> tuple[float, float]:
    """Normalize a scalar or pair domain size to an (Lx, Ly) float tuple.

    Raises:
        ValueError: If domain_size format is invalid or dimensions are non-positive.
    """
    if isinstance(domain_size, (int, float)):
        Lx = float(domain_size)
        Ly = float(domain_size)
    elif isinstance(domain_size, (list, tuple)) and len(domain_size) == 2:
        Lx = float(domain_size[0])
        Ly = float(domain_size[1])
    else:
        raise ValueError(f"Invalid domain_size: {domain_size}. Expected a float or a 2-tuple (Lx, Ly).")

    if Lx <= 0.0 or Ly <= 0.0:
        raise ValueError(f"Domain dimensions must be strictly positive, got Lx={Lx:g}, Ly={Ly:g}.")

    return Lx, Ly


def _check_wound_fits_domain(
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    Lx: float,
    Ly: float,
    shape_name: str = "wound",
) -> None:
    """Validate that the wound geometry fits strictly inside the domain (0, Lx) x (0, Ly).

    Raises:
        ValueError: If the wound touches or extends outside the domain.
    """
    if xmin <= 0.0 or xmax >= Lx or ymin <= 0.0 or ymax >= Ly:
        raise ValueError(
            f"{shape_name.capitalize()} with bounding box [x: ({xmin:g}, {xmax:g}), y: ({ymin:g}, {ymax:g})] "
            f"does not fit inside domain [0, {Lx:g}] x [0, {Ly:g}]."
        )


def _check_polygon_self_intersection(x: np.ndarray, y: np.ndarray) -> None:
    """Check if a 2D closed polygon (x, y) self-intersects.

    Raises:
        ValueError: If any non-adjacent segments intersect.
    """
    pts = np.column_stack((x, y))
    n = len(pts)
    if n < 3:
        raise ValueError(f"Polygon must have at least 3 points, got {n}.")

    p1 = pts
    p2 = np.roll(pts, -1, axis=0)

    i_idx, j_idx = np.triu_indices(n, k=2)
    mask = ~((i_idx == 0) & (j_idx == n - 1))
    i_idx = i_idx[mask]
    j_idx = j_idx[mask]

    a = p1[i_idx]
    b = p2[i_idx]
    c = p1[j_idx]
    d = p2[j_idx]

    def cross2d(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
        return v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]

    ab = b - a
    cd = d - c

    cp1 = cross2d(ab, c - a)
    cp2 = cross2d(ab, d - a)
    cp3 = cross2d(cd, a - c)
    cp4 = cross2d(cd, b - c)

    straddle = ((cp1 > 1e-12) & (cp2 < -1e-12) | (cp1 < -1e-12) & (cp2 > 1e-12)) & (
        (cp3 > 1e-12) & (cp4 < -1e-12) | (cp3 < -1e-12) & (cp4 > 1e-12)
    )

    if np.any(straddle):
        raise ValueError("Perturbed circle boundary curve is self-intersecting.")

    eps = 1e-12
    on_seg_c = (
        (np.abs(cp1) <= eps)
        & (np.minimum(a[:, 0], b[:, 0]) - eps <= c[:, 0])
        & (c[:, 0] <= np.maximum(a[:, 0], b[:, 0]) + eps)
        & (np.minimum(a[:, 1], b[:, 1]) - eps <= c[:, 1])
        & (c[:, 1] <= np.maximum(a[:, 1], b[:, 1]) + eps)
    )

    on_seg_d = (
        (np.abs(cp2) <= eps)
        & (np.minimum(a[:, 0], b[:, 0]) - eps <= d[:, 0])
        & (d[:, 0] <= np.maximum(a[:, 0], b[:, 0]) + eps)
        & (np.minimum(a[:, 1], b[:, 1]) - eps <= d[:, 1])
        & (d[:, 1] <= np.maximum(a[:, 1], b[:, 1]) + eps)
    )

    on_seg_a = (
        (np.abs(cp3) <= eps)
        & (np.minimum(c[:, 0], d[:, 0]) - eps <= a[:, 0])
        & (a[:, 0] <= np.maximum(c[:, 0], d[:, 0]) + eps)
        & (np.minimum(c[:, 1], d[:, 1]) - eps <= a[:, 1])
        & (a[:, 1] <= np.maximum(c[:, 1], d[:, 1]) + eps)
    )

    on_seg_b = (
        (np.abs(cp4) <= eps)
        & (np.minimum(c[:, 0], d[:, 0]) - eps <= b[:, 0])
        & (b[:, 0] <= np.maximum(c[:, 0], d[:, 0]) + eps)
        & (np.minimum(c[:, 1], d[:, 1]) - eps <= b[:, 1])
        & (b[:, 1] <= np.maximum(c[:, 1], d[:, 1]) + eps)
    )

    if np.any(on_seg_c | on_seg_d | on_seg_a | on_seg_b):
        raise ValueError("Perturbed circle boundary curve is self-intersecting.")


def build_mesh_filename(
    shape: str,
    domain_size: float | tuple[float, float],
    center: tuple[float, float],
    angle: float = 0.0,
    extension: str = ".xdmf",
    **kwargs: Any,
) -> str:
    """Build a descriptive, deterministic filename for a generated mesh based on parameters."""
    Lx, Ly = normalize_domain_size(domain_size)
    cx, cy = center
    geom_tokens = []
    for k, v in sorted(kwargs.items()):
        if v is None:
            continue
        if isinstance(v, (list, tuple)):
            v_str = "_".join(_fmt(x) for x in v)
        else:
            v_str = _fmt(v)
        geom_tokens.append(f"{k}{v_str}")

    tokens = [shape]
    if geom_tokens:
        tokens.extend(geom_tokens)
    tokens.append(f"c{_fmt(cx)}_{_fmt(cy)}")
    if abs(angle) > 1e-12:
        tokens.append(f"ang{_fmt(angle)}")

    if abs(Lx - Ly) < 1e-12:
        tokens.append(f"L{_fmt(Lx)}")
    else:
        tokens.append(f"L{_fmt(Lx)}x{_fmt(Ly)}")

    ext = extension if extension.startswith(".") else f".{extension}"
    return f"{'_'.join(tokens)}{ext}"


def _validate_generated_tags(tagged_mesh: TaggedMesh) -> None:
    """Validate that the generated mesh contains all required physical tags.

    Uses global MPI counts, because in parallel some ranks may own zero
    entities of a given tag even when the full distributed mesh is valid.
    """
    comm = tagged_mesh.mesh.comm

    if tagged_mesh.cell_tags is None:
        raise RuntimeError("Generated mesh has no cell_tags.")

    if tagged_mesh.facet_tags is None:
        raise RuntimeError("Generated mesh has no facet_tags.")

    n_wound_cells_local = len(tagged_mesh.cell_tags.find(CellTag.WOUND))
    n_healthy_cells_local = len(tagged_mesh.cell_tags.find(CellTag.HEALTHY))
    n_wound_facets_local = len(tagged_mesh.facet_tags.find(FacetTag.WOUND_BOUNDARY))
    n_outer_facets_local = len(tagged_mesh.facet_tags.find(FacetTag.OUTER_BOUNDARY))

    n_wound_cells = comm.allreduce(n_wound_cells_local, op=MPI.SUM)
    n_healthy_cells = comm.allreduce(n_healthy_cells_local, op=MPI.SUM)
    n_wound_facets = comm.allreduce(n_wound_facets_local, op=MPI.SUM)
    n_outer_facets = comm.allreduce(n_outer_facets_local, op=MPI.SUM)

    missing = []
    if n_wound_cells == 0:
        missing.append("CellTag.WOUND")
    if n_healthy_cells == 0:
        missing.append("CellTag.HEALTHY")
    if n_wound_facets == 0:
        missing.append("FacetTag.WOUND_BOUNDARY")
    if n_outer_facets == 0:
        missing.append("FacetTag.OUTER_BOUNDARY")

    if missing:
        raise RuntimeError("Generated mesh is missing required physical tags: " + ", ".join(missing))


def generate_2d_surface_mesh(
    domain_size: float | tuple[float, float] = (10.0, 10.0),
    wound_center: tuple[float, float] | None = None,
    wound_radius: float = 2.0,
    char_length_far: float = 1.0,
    char_length_wound: float = 0.5,
    refinement_radius: float | None = None,
    wound_creator: Callable[[Any], tuple[int, float]] | None = None,
    wound_shape: str | None = None,
    gdim: int = 2,
    comm: MPI.Comm = MPI.COMM_WORLD,
    model_rank: int = 0,
    gui: bool = False,
    save: bool = False,
    save_path: str | Path | None = None,
) -> TaggedMesh:
    """Generate a 2D square or rectangular skin patch with an embedded wound region.

    The wound is embedded as its own tagged surface (CellTag.WOUND) inside
    the healthy-skin surface (CellTag.HEALTHY), sharing a tagged interface
    curve (FacetTag.WOUND_BOUNDARY) for tracking boundary dynamics over time.
    The outer rectangle edges are tagged FacetTag.OUTER_BOUNDARY for Dirichlet BCs.

    Mesh size is graded from `char_length_wound` inside and near the wound boundary to
    `char_length_far` at the domain boundary (Distance + Threshold + Restrict fields).
    """
    Lx, Ly = normalize_domain_size(domain_size)

    if wound_center is None:
        cx, cy = Lx / 2.0, Ly / 2.0
    else:
        cx, cy = wound_center

    if wound_creator is None:
        if wound_radius <= 0.0:
            raise ValueError(f"Wound radius must be positive, got {wound_radius:g}.")
        _check_wound_fits_domain(
            cx - wound_radius,
            cx + wound_radius,
            cy - wound_radius,
            cy + wound_radius,
            Lx,
            Ly,
            shape_name="wound",
        )

    gmsh.initialize()
    try:
        gmsh.model.add("burn_2d_surface")

        if comm.rank == model_rank:
            occ = gmsh.model.occ

            outer = occ.addRectangle(0.0, 0.0, 0.0, Lx, Ly)

            if wound_creator is not None:
                wound_tag, approx_extent = wound_creator(occ)
            else:
                wound_tag = occ.addDisk(cx, cy, 0.0, wound_radius, wound_radius)
                approx_extent = wound_radius

            if refinement_radius is None:
                refinement_radius = 3.0 * approx_extent

            # Fragment so the wound disk and outer rectangle share conformal
            # edges/nodes at their interface -- required for a clean shared tag
            # and for the two regions to remain a single connected mesh.
            occ.fragment([(2, outer)], [(2, wound_tag)])
            occ.synchronize()

            surfaces = gmsh.model.getEntities(dim=2)

            if len(surfaces) != 2:
                raise RuntimeError(f"Expected 2 surfaces after fragmentation, got {len(surfaces)}.")
            # After fragmentation, tell the wound surface apart from the
            # (rectangle minus wound) healthy surface by area.
            surface_areas = []

            for _, tag in surfaces:
                area = occ.getMass(2, tag)
                surface_areas.append((area, tag))

            surface_areas.sort()

            # Smallest surface = wound
            wound_tag = surface_areas[0][1]

            # Largest surface = healthy tissue
            healthy_tag = surface_areas[-1][1]

            if wound_tag is None:
                raise RuntimeError("Failed to identify wound surface.")

            if healthy_tag is None:
                raise RuntimeError("Failed to identify healthy surface.")

            gmsh.model.addPhysicalGroup(2, [healthy_tag], CellTag.HEALTHY)
            gmsh.model.setPhysicalName(2, CellTag.HEALTHY, "healthy")
            gmsh.model.addPhysicalGroup(2, [wound_tag], CellTag.WOUND)
            gmsh.model.setPhysicalName(2, CellTag.WOUND, "wound")

            # Classify boundary curves: curves bounding both surfaces are the
            # wound/healthy interface; curves bounding only the healthy surface
            # are the outer rectangle boundary.
            healthy_curves = {c[1] for c in gmsh.model.getBoundary([(2, healthy_tag)], combined=False, oriented=False)}
            wound_curves = {c[1] for c in gmsh.model.getBoundary([(2, wound_tag)], combined=False, oriented=False)}
            interface_curves = healthy_curves & wound_curves
            outer_curves = healthy_curves - interface_curves

            gmsh.model.addPhysicalGroup(1, list(outer_curves), FacetTag.OUTER_BOUNDARY)
            gmsh.model.setPhysicalName(1, FacetTag.OUTER_BOUNDARY, "outer_boundary")
            gmsh.model.addPhysicalGroup(1, list(interface_curves), FacetTag.WOUND_BOUNDARY)
            gmsh.model.setPhysicalName(1, FacetTag.WOUND_BOUNDARY, "wound_boundary")

            # ------------------------------------------------------------
            # Mesh-size fields
            #
            # Goal:
            # - Entire wound region gets char_length_wound
            # - Small transition band outside wound
            # - Coarse mesh elsewhere
            # ------------------------------------------------------------

            field = gmsh.model.mesh.field

            # Distance from wound boundary
            dist_field = field.add("Distance")
            field.setNumbers(
                dist_field,
                "CurvesList",
                list(interface_curves),
            )
            field.setNumber(dist_field, "Sampling", 100)

            # Transition band around wound boundary
            thresh_field = field.add("Threshold")
            field.setNumber(thresh_field, "InField", dist_field)
            field.setNumber(thresh_field, "SizeMin", char_length_wound)
            field.setNumber(thresh_field, "SizeMax", char_length_far)

            # Fine at wound boundary
            field.setNumber(thresh_field, "DistMin", 0.0)

            # Transition to coarse outside wound
            field.setNumber(
                thresh_field,
                "DistMax",
                refinement_radius,
            )

            # Constant fine size inside the wound surface
            wound_size_field = field.add("MathEval")
            field.setString(
                wound_size_field,
                "F",
                str(char_length_wound),
            )

            # Restrict fine sizing to wound surface only
            wound_refine_field = field.add("Restrict")
            field.setNumber(
                wound_refine_field,
                "InField",
                wound_size_field,
            )
            field.setNumbers(
                wound_refine_field,
                "SurfacesList",
                [wound_tag],
            )

            # Take the smallest requested mesh size from both fields
            background_field = field.add("Min")
            field.setNumbers(
                background_field,
                "FieldsList",
                [
                    thresh_field,
                    wound_refine_field,
                ],
            )

            field.setAsBackgroundMesh(background_field)
            gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
            gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

            gmsh.model.mesh.generate(2)

        if gui and comm.rank == model_rank:
            gmsh.fltk.run()

        partitioner = dolfinx.mesh.create_cell_partitioner(
            dolfinx.graph.partitioner(),
            dolfinx.mesh.GhostMode.shared_facet,
            2,
        )
        mesh_data = dfx_gmsh.model_to_mesh(gmsh.model, comm, model_rank, gdim=gdim, partitioner=partitioner)
    finally:
        gmsh.finalize()

    if wound_shape is None and wound_creator is None:
        wound_shape = "circle"

    tagged_mesh = TaggedMesh(
        mesh=mesh_data.mesh,
        cell_tags=mesh_data.cell_tags,
        facet_tags=mesh_data.facet_tags,
        wound_shape=wound_shape,
    )

    _validate_generated_tags(tagged_mesh)

    if save:
        if save_path is None:
            default_name = build_mesh_filename("surface_mesh", domain_size=(Lx, Ly), center=(cx, cy))
            save_path = MESH_DIR / "generated" / default_name
        save_mesh_xdmf(tagged_mesh, save_path)

    return tagged_mesh


def generate_circle_mesh(
    domain_size: float | tuple[float, float] = 10.0,
    radius: float = 2.0,
    center: tuple[float, float] | None = None,
    char_length_far: float = 1.0,
    char_length_wound: float = 0.5,
    refinement_radius: float | None = None,
    gdim: int = 2,
    comm: MPI.Comm = MPI.COMM_WORLD,
    model_rank: int = 0,
    gui: bool = False,
    save: bool = False,
    save_dir: str | Path | None = None,
    save_path: str | Path | None = None,
) -> TaggedMesh:
    """Generate a square domain with an embedded circular wound."""
    Lx, Ly = normalize_domain_size(domain_size)
    cx, cy = (Lx / 2.0, Ly / 2.0) if center is None else center

    if radius <= 0.0:
        raise ValueError(f"Circle radius must be positive, got {radius:g}.")
    _check_wound_fits_domain(
        cx - radius,
        cx + radius,
        cy - radius,
        cy + radius,
        Lx,
        Ly,
        shape_name="circle",
    )

    def wound_creator(occ: Any) -> tuple[int, float]:
        tag = occ.addDisk(cx, cy, 0.0, radius, radius)
        return tag, radius

    if save and save_path is None:
        filename = build_mesh_filename("circle", domain_size=(Lx, Ly), center=(cx, cy), r=radius)
        target_dir = Path(save_dir) if save_dir is not None else MESH_DIR / "generated"
        save_path = target_dir / filename

    return generate_2d_surface_mesh(
        domain_size=(Lx, Ly),
        wound_center=(cx, cy),
        wound_radius=radius,
        char_length_far=char_length_far,
        char_length_wound=char_length_wound,
        refinement_radius=refinement_radius,
        wound_creator=wound_creator,
        wound_shape="circle",
        gdim=gdim,
        comm=comm,
        model_rank=model_rank,
        gui=gui,
        save=save,
        save_path=save_path,
    )


def generate_ellipse_mesh(
    domain_size: float | tuple[float, float] = 10.0,
    semi_axes: tuple[float, float] = (2.5, 1.5),
    center: tuple[float, float] | None = None,
    angle: float = 0.0,
    char_length_far: float = 1.0,
    char_length_wound: float = 0.5,
    refinement_radius: float | None = None,
    gdim: int = 2,
    comm: MPI.Comm = MPI.COMM_WORLD,
    model_rank: int = 0,
    gui: bool = False,
    save: bool = False,
    save_dir: str | Path | None = None,
    save_path: str | Path | None = None,
) -> TaggedMesh:
    """Generate a square domain with an embedded elliptical wound."""
    Lx, Ly = normalize_domain_size(domain_size)
    cx, cy = (Lx / 2.0, Ly / 2.0) if center is None else center
    a, b = semi_axes

    if a <= 0.0 or b <= 0.0:
        raise ValueError(f"Ellipse semi-axes must be positive, got semi_axes=({a:g}, {b:g}).")

    x_ext = np.sqrt((a * np.cos(angle)) ** 2 + (b * np.sin(angle)) ** 2)
    y_ext = np.sqrt((a * np.sin(angle)) ** 2 + (b * np.cos(angle)) ** 2)
    _check_wound_fits_domain(
        cx - x_ext,
        cx + x_ext,
        cy - y_ext,
        cy + y_ext,
        Lx,
        Ly,
        shape_name="ellipse",
    )

    def wound_creator(occ: Any) -> tuple[int, float]:
        tag = occ.addDisk(cx, cy, 0.0, a, b)
        if abs(angle) > 1e-12:
            occ.rotate([(2, tag)], cx, cy, 0.0, 0.0, 0.0, 1.0, angle)
        return tag, max(a, b)

    if save and save_path is None:
        filename = build_mesh_filename(
            "ellipse",
            domain_size=(Lx, Ly),
            center=(cx, cy),
            angle=angle,
            a=a,
            b=b,
        )
        target_dir = Path(save_dir) if save_dir is not None else MESH_DIR / "generated"
        save_path = target_dir / filename

    return generate_2d_surface_mesh(
        domain_size=(Lx, Ly),
        wound_center=(cx, cy),
        wound_radius=max(a, b),
        char_length_far=char_length_far,
        char_length_wound=char_length_wound,
        refinement_radius=refinement_radius,
        wound_creator=wound_creator,
        wound_shape="ellipse",
        gdim=gdim,
        comm=comm,
        model_rank=model_rank,
        gui=gui,
        save=save,
        save_path=save_path,
    )


def generate_superellipse_mesh(
    domain_size: float | tuple[float, float] = 10.0,
    semi_axes: tuple[float, float] = (2.5, 1.5),
    exponent: float = 4.0,
    center: tuple[float, float] | None = None,
    angle: float = 0.0,
    char_length_far: float = 1.0,
    char_length_wound: float = 0.5,
    refinement_radius: float | None = None,
    num_points: int = 120,
    gdim: int = 2,
    comm: MPI.Comm = MPI.COMM_WORLD,
    model_rank: int = 0,
    gui: bool = False,
    save: bool = False,
    save_dir: str | Path | None = None,
    save_path: str | Path | None = None,
) -> TaggedMesh:
    """Generate a square domain with an embedded superelliptical (Lamé curve) wound (|x/a|^n + |y/b|^n = 1)."""
    Lx, Ly = normalize_domain_size(domain_size)
    cx, cy = (Lx / 2.0, Ly / 2.0) if center is None else center
    a, b = semi_axes

    if a <= 0.0 or b <= 0.0:
        raise ValueError(f"Superellipse semi-axes must be positive, got semi_axes=({a:g}, {b:g}).")
    if exponent <= 0.0:
        raise ValueError(f"Superellipse exponent must be positive, got exponent={exponent:g}.")

    t = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    x_prime = a * np.sign(np.cos(t)) * (np.abs(np.cos(t)) ** (2.0 / exponent))
    y_prime = b * np.sign(np.sin(t)) * (np.abs(np.sin(t)) ** (2.0 / exponent))
    x_pts = cx + x_prime * np.cos(angle) - y_prime * np.sin(angle)
    y_pts = cy + x_prime * np.sin(angle) + y_prime * np.cos(angle)

    _check_wound_fits_domain(
        float(np.min(x_pts)),
        float(np.max(x_pts)),
        float(np.min(y_pts)),
        float(np.max(y_pts)),
        Lx,
        Ly,
        shape_name="superellipse",
    )

    def wound_creator(occ: Any) -> tuple[int, float]:
        pt_tags = [occ.addPoint(float(xi), float(yi), 0.0) for xi, yi in zip(x_pts, y_pts)]
        pt_tags.append(pt_tags[0])
        spline = occ.addBSpline(pt_tags)
        wire = occ.addWire([spline])
        surf = occ.addPlaneSurface([wire])
        return surf, max(a, b)

    if save and save_path is None:
        filename = build_mesh_filename(
            "superellipse",
            domain_size=(Lx, Ly),
            center=(cx, cy),
            angle=angle,
            a=a,
            b=b,
            n=exponent,
        )
        target_dir = Path(save_dir) if save_dir is not None else MESH_DIR / "generated"
        save_path = target_dir / filename

    return generate_2d_surface_mesh(
        domain_size=(Lx, Ly),
        wound_center=(cx, cy),
        wound_radius=max(a, b),
        char_length_far=char_length_far,
        char_length_wound=char_length_wound,
        refinement_radius=refinement_radius,
        wound_creator=wound_creator,
        wound_shape="superellipse",
        gdim=gdim,
        comm=comm,
        model_rank=model_rank,
        gui=gui,
        save=save,
        save_path=save_path,
    )


def generate_perturbed_circle_mesh(
    domain_size: float | tuple[float, float] = 10.0,
    base_radius: float = 2.0,
    perturbation_amplitude: float = 0.2,
    wavenumber: int = 3,
    phase: float = 0.0,
    modes: list[tuple[int, float]] | list[tuple[int, float, float]] | None = None,
    center: tuple[float, float] | None = None,
    angle: float = 0.0,
    char_length_far: float = 1.0,
    char_length_wound: float = 0.5,
    refinement_radius: float | None = None,
    num_points: int = 120,
    gdim: int = 2,
    comm: MPI.Comm = MPI.COMM_WORLD,
    model_rank: int = 0,
    gui: bool = False,
    save: bool = False,
    save_dir: str | Path | None = None,
    save_path: str | Path | None = None,
) -> TaggedMesh:
    """Generate a square domain with an embedded perturbed circular wound:
    r(theta) = r0 * (1 + sum eps * cos(m * theta + phi))."""
    Lx, Ly = normalize_domain_size(domain_size)
    cx, cy = (Lx / 2.0, Ly / 2.0) if center is None else center

    if base_radius <= 0.0:
        raise ValueError(f"Perturbed circle base_radius must be positive, got {base_radius:g}.")

    harmonic_modes: list[tuple[int, float, float]] = []
    if modes is not None:
        for mode in modes:
            if len(mode) == 2:
                harmonic_modes.append((int(mode[0]), float(mode[1]), 0.0))
            else:
                harmonic_modes.append((int(mode[0]), float(mode[1]), float(mode[2])))
    else:
        harmonic_modes.append((wavenumber, perturbation_amplitude, phase))

    theta = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    r = np.full_like(theta, base_radius)
    for m, eps, phi in harmonic_modes:
        r += base_radius * eps * np.cos(m * theta + phi)

    min_r = float(np.min(r))
    if min_r <= 0.0:
        raise ValueError(
            f"Perturbed circle radius must be strictly positive everywhere, got minimum radius {min_r:g} <= 0."
        )

    x_prime = r * np.cos(theta)
    y_prime = r * np.sin(theta)
    x_pts = cx + x_prime * np.cos(angle) - y_prime * np.sin(angle)
    y_pts = cy + x_prime * np.sin(angle) + y_prime * np.cos(angle)

    _check_polygon_self_intersection(x_pts, y_pts)
    _check_wound_fits_domain(
        float(np.min(x_pts)),
        float(np.max(x_pts)),
        float(np.min(y_pts)),
        float(np.max(y_pts)),
        Lx,
        Ly,
        shape_name="perturbed circle",
    )

    def wound_creator(occ: Any) -> tuple[int, float]:
        pt_tags = [occ.addPoint(float(xi), float(yi), 0.0) for xi, yi in zip(x_pts, y_pts)]
        pt_tags.append(pt_tags[0])
        spline = occ.addBSpline(pt_tags)
        wire = occ.addWire([spline])
        surf = occ.addPlaneSurface([wire])
        return surf, float(np.max(r))

    if save and save_path is None:
        if modes is None:
            filename = build_mesh_filename(
                "perturbed_circle",
                domain_size=(Lx, Ly),
                center=(cx, cy),
                angle=angle,
                r=base_radius,
                eps=perturbation_amplitude,
                m=wavenumber,
            )
        else:
            mode_str = "-".join(f"m{m}_eps{eps:g}" for m, eps, _ in harmonic_modes)
            filename = build_mesh_filename(
                "perturbed_circle",
                domain_size=(Lx, Ly),
                center=(cx, cy),
                angle=angle,
                r=base_radius,
                modes=mode_str,
            )
        target_dir = Path(save_dir) if save_dir is not None else MESH_DIR / "generated"
        save_path = target_dir / filename

    max_extent = float(np.max(r))

    return generate_2d_surface_mesh(
        domain_size=(Lx, Ly),
        wound_center=(cx, cy),
        wound_radius=max_extent,
        char_length_far=char_length_far,
        char_length_wound=char_length_wound,
        refinement_radius=refinement_radius,
        wound_creator=wound_creator,
        wound_shape="perturbed_circle",
        gdim=gdim,
        comm=comm,
        model_rank=model_rank,
        gui=gui,
        save=save,
        save_path=save_path,
    )


def generate_dumbbell_mesh(
    domain_size: float | tuple[float, float] = 10.0,
    lobe_radius: float = 1.0,
    lobe_separation: float = 1.5,
    neck_width: float = 0.5,
    center: tuple[float, float] | None = None,
    angle: float = 0.0,
    char_length_far: float = 1.0,
    char_length_wound: float = 0.5,
    refinement_radius: float | None = None,
    gdim: int = 2,
    comm: MPI.Comm = MPI.COMM_WORLD,
    model_rank: int = 0,
    gui: bool = False,
    save: bool = False,
    save_dir: str | Path | None = None,
    save_path: str | Path | None = None,
) -> TaggedMesh:
    """
    Generate a square domain with an embedded dumbbell wound (two circular lobes connected by a central neck bridge).
    """
    Lx, Ly = normalize_domain_size(domain_size)
    cx, cy = (Lx / 2.0, Ly / 2.0) if center is None else center

    d = lobe_separation
    r = lobe_radius
    w = neck_width
    half_w = w / 2.0

    if r <= 0.0 or d <= 0.0 or w <= 0.0:
        raise ValueError(f"Dumbbell parameters r, d, w must be positive, got r={r:g}, d={d:g}, w={w:g}.")

    c1_x = cx - d * np.cos(angle)
    c1_y = cy - d * np.sin(angle)
    c2_x = cx + d * np.cos(angle)
    c2_y = cy + d * np.sin(angle)

    local_corners = np.array(
        [
            [-d, -half_w],
            [-d, half_w],
            [d, -half_w],
            [d, half_w],
        ]
    )
    rot_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    rot_corners = (local_corners @ rot_matrix.T) + np.array([cx, cy])

    xmin = min(c1_x - r, c2_x - r, float(np.min(rot_corners[:, 0])))
    xmax = max(c1_x + r, c2_x + r, float(np.max(rot_corners[:, 0])))
    ymin = min(c1_y - r, c2_y - r, float(np.min(rot_corners[:, 1])))
    ymax = max(c1_y + r, c2_y + r, float(np.max(rot_corners[:, 1])))

    _check_wound_fits_domain(xmin, xmax, ymin, ymax, Lx, Ly, shape_name="dumbbell")

    def wound_creator(occ: Any) -> tuple[int, float]:
        d1 = occ.addDisk(cx - d, cy, 0.0, r, r)
        d2 = occ.addDisk(cx + d, cy, 0.0, r, r)
        rect = occ.addRectangle(cx - d, cy - half_w, 0.0, 2.0 * d, w)
        fused, _ = occ.fuse([(2, d1), (2, d2)], [(2, rect)])
        surf_tag = fused[0][1]
        if abs(angle) > 1e-12:
            occ.rotate([(2, surf_tag)], cx, cy, 0.0, 0.0, 0.0, 1.0, angle)
        return surf_tag, d + r

    if save and save_path is None:
        filename = build_mesh_filename(
            "dumbbell",
            domain_size=(Lx, Ly),
            center=(cx, cy),
            angle=angle,
            d=lobe_separation,
            r=lobe_radius,
            w=neck_width,
        )
        target_dir = Path(save_dir) if save_dir is not None else MESH_DIR / "generated"
        save_path = target_dir / filename

    return generate_2d_surface_mesh(
        domain_size=(Lx, Ly),
        wound_center=(cx, cy),
        wound_radius=d + r,
        char_length_far=char_length_far,
        char_length_wound=char_length_wound,
        refinement_radius=refinement_radius,
        wound_creator=wound_creator,
        wound_shape="dumbbell",
        gdim=gdim,
        comm=comm,
        model_rank=model_rank,
        gui=gui,
        save=save,
        save_path=save_path,
    )


def gaussian_bump_height(
    amplitude: float = 1.0,
    sigma: float | None = None,
    width: float | None = None,
    center: tuple[float, float] | None = None,
) -> Callable[[np.ndarray | float, np.ndarray | float], np.ndarray]:
    """Return a Gaussian bump height function: z = amplitude * exp(-((x-cx)^2 + (y-cy)^2)/(2*sigma^2))."""
    s = sigma if sigma is not None else (width if width is not None else 2.0)
    cx, cy = (0.0, 0.0) if center is None else center

    def f(x: np.ndarray | float, y: np.ndarray | float) -> np.ndarray:
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        return amplitude * np.exp(-((x_arr - cx) ** 2 + (y_arr - cy) ** 2) / (2.0 * s**2))

    return f


def sinusoid_height(
    amplitude: float = 1.0,
    wavelength: float | tuple[float, float] = 10.0,
    phase: tuple[float, float] = (0.0, 0.0),
) -> Callable[[np.ndarray | float, np.ndarray | float], np.ndarray]:
    """
    Return a sinusoidal height function:
    z = amplitude * sin(2*pi*x/lambda_x + phase_x) * sin(2*pi*y/lambda_y + phase_y).
    """
    if isinstance(wavelength, (int, float)):
        wx, wy = float(wavelength), float(wavelength)
    else:
        wx, wy = float(wavelength[0]), float(wavelength[1])
    px, py = phase

    def f(x: np.ndarray | float, y: np.ndarray | float) -> np.ndarray:
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        kx = 2.0 * np.pi / wx
        ky = 2.0 * np.pi / wy
        return amplitude * np.sin(kx * x_arr + px) * np.sin(ky * y_arr + py)

    return f


def lift_tagged_mesh(
    tagged_mesh: TaggedMesh,
    height_function: Callable[[np.ndarray | float, np.ndarray | float], np.ndarray],
) -> TaggedMesh:
    """Lift a TaggedMesh into 3D by setting node z-coordinates to f(x, y).

    Requires that the mesh has gdim == 3.
    """
    if tagged_mesh.gdim != 3:
        raise ValueError(f"lift_tagged_mesh requires a mesh with gdim=3, got gdim={tagged_mesh.gdim}.")
    coords = tagged_mesh.mesh.geometry.x
    x = coords[:, 0]
    y = coords[:, 1]
    z = np.asarray(height_function(x, y), dtype=float)
    coords[:, 2] = z
    return tagged_mesh


def generate_curved_surface_mesh(
    domain_size: float | tuple[float, float] = 10.0,
    wound_shape: str = "circle",
    wound_center: tuple[float, float] | None = None,
    wound_radius: float = 2.0,
    char_length_far: float = 1.0,
    char_length_wound: float = 0.5,
    refinement_radius: float | None = None,
    wound_creator: Callable[[Any], tuple[int, float]] | None = None,
    height_function: Callable[[np.ndarray | float, np.ndarray | float], np.ndarray] | str | None = "gaussian_bump",
    surface_params: dict[str, Any] | None = None,
    comm: MPI.Comm = MPI.COMM_WORLD,
    model_rank: int = 0,
    gui: bool = False,
    save: bool = False,
    save_dir: str | Path | None = None,
    save_path: str | Path | None = None,
    **wound_kwargs: Any,
) -> TaggedMesh:
    """Generate a curved 2D surface mesh embedded in 3D (tdim=2, gdim=3) with tagged regions."""
    Lx, Ly = normalize_domain_size(domain_size)
    cx, cy = (Lx / 2.0, Ly / 2.0) if wound_center is None else wound_center

    if wound_creator is not None:
        mesh = generate_2d_surface_mesh(
            domain_size=(Lx, Ly),
            wound_center=(cx, cy),
            wound_radius=wound_radius,
            char_length_far=char_length_far,
            char_length_wound=char_length_wound,
            refinement_radius=refinement_radius,
            wound_creator=wound_creator,
            wound_shape=wound_shape,
            gdim=3,
            comm=comm,
            model_rank=model_rank,
            gui=gui,
            save=False,
        )
    elif wound_shape == "circle":
        radius = wound_kwargs.get("radius", wound_radius)
        mesh = generate_circle_mesh(
            domain_size=(Lx, Ly),
            radius=radius,
            center=(cx, cy),
            char_length_far=char_length_far,
            char_length_wound=char_length_wound,
            refinement_radius=refinement_radius,
            gdim=3,
            comm=comm,
            model_rank=model_rank,
            gui=gui,
            save=False,
        )
    elif wound_shape == "ellipse":
        semi_axes = wound_kwargs.get("semi_axes", (wound_radius, wound_radius * 0.6))
        angle = wound_kwargs.get("angle", 0.0)
        mesh = generate_ellipse_mesh(
            domain_size=(Lx, Ly),
            semi_axes=semi_axes,
            center=(cx, cy),
            angle=angle,
            char_length_far=char_length_far,
            char_length_wound=char_length_wound,
            refinement_radius=refinement_radius,
            gdim=3,
            comm=comm,
            model_rank=model_rank,
            gui=gui,
            save=False,
        )
    elif wound_shape == "superellipse":
        semi_axes = wound_kwargs.get("semi_axes", (wound_radius, wound_radius * 0.6))
        exponent = wound_kwargs.get("exponent", 4.0)
        angle = wound_kwargs.get("angle", 0.0)
        num_points = wound_kwargs.get("num_points", 120)
        mesh = generate_superellipse_mesh(
            domain_size=(Lx, Ly),
            semi_axes=semi_axes,
            exponent=exponent,
            center=(cx, cy),
            angle=angle,
            char_length_far=char_length_far,
            char_length_wound=char_length_wound,
            refinement_radius=refinement_radius,
            num_points=num_points,
            gdim=3,
            comm=comm,
            model_rank=model_rank,
            gui=gui,
            save=False,
        )
    elif wound_shape == "perturbed_circle":
        base_radius = wound_kwargs.get("base_radius", wound_radius)
        perturbation_amplitude = wound_kwargs.get("perturbation_amplitude", 0.2)
        wavenumber = wound_kwargs.get("wavenumber", 3)
        phase = wound_kwargs.get("phase", 0.0)
        modes = wound_kwargs.get("modes", None)
        angle = wound_kwargs.get("angle", 0.0)
        num_points = wound_kwargs.get("num_points", 120)
        mesh = generate_perturbed_circle_mesh(
            domain_size=(Lx, Ly),
            base_radius=base_radius,
            perturbation_amplitude=perturbation_amplitude,
            wavenumber=wavenumber,
            phase=phase,
            modes=modes,
            center=(cx, cy),
            angle=angle,
            char_length_far=char_length_far,
            char_length_wound=char_length_wound,
            refinement_radius=refinement_radius,
            num_points=num_points,
            gdim=3,
            comm=comm,
            model_rank=model_rank,
            gui=gui,
            save=False,
        )
    elif wound_shape == "dumbbell":
        lobe_radius = wound_kwargs.get("lobe_radius", 1.0)
        lobe_separation = wound_kwargs.get("lobe_separation", 1.5)
        neck_width = wound_kwargs.get("neck_width", 0.5)
        angle = wound_kwargs.get("angle", 0.0)
        mesh = generate_dumbbell_mesh(
            domain_size=(Lx, Ly),
            lobe_radius=lobe_radius,
            lobe_separation=lobe_separation,
            neck_width=neck_width,
            center=(cx, cy),
            angle=angle,
            char_length_far=char_length_far,
            char_length_wound=char_length_wound,
            refinement_radius=refinement_radius,
            gdim=3,
            comm=comm,
            model_rank=model_rank,
            gui=gui,
            save=False,
        )
    else:
        raise ValueError(f"Unknown wound_shape: '{wound_shape}'. Provide wound_creator for custom shapes.")

    resolved_surface_params: dict[str, Any] = dict(surface_params or {})
    if isinstance(height_function, str) or height_function is None:
        name = "gaussian_bump" if height_function is None else height_function
        surface_shape_name = name
        if name == "gaussian_bump":
            params = {"amplitude": 1.0, "sigma": 2.0, "center": (cx, cy)}
            params.update(resolved_surface_params)
            resolved_surface_params = params
            h_fn = gaussian_bump_height(**params)
        elif name == "sinusoid":
            params = {"amplitude": 1.0, "wavelength": (Lx, Ly), "phase": (0.0, 0.0)}
            params.update(resolved_surface_params)
            resolved_surface_params = params
            h_fn = sinusoid_height(**params)
        elif name == "flat":
            surface_shape_name = "flat"

            def h_fn(x, y):
                return np.zeros_like(np.asarray(x, dtype=float))
        else:
            raise ValueError(f"Unknown height_function preset: '{name}'.")
    elif callable(height_function):
        surface_shape_name = getattr(height_function, "__name__", "custom")
        h_fn = height_function
    else:
        raise TypeError(f"height_function must be callable, str, or None, got {type(height_function)}.")

    lift_tagged_mesh(mesh, h_fn)
    mesh.surface_shape = surface_shape_name
    mesh.surface_params = resolved_surface_params

    _validate_generated_tags(mesh)

    if save:
        if save_path is None:
            filename_params = {k: v for k, v in resolved_surface_params.items() if k != "center"}
            filename = build_mesh_filename(
                f"curved_{wound_shape}",
                domain_size=(Lx, Ly),
                center=(cx, cy),
                **filename_params,
            )
            target_dir = Path(save_dir) if save_dir is not None else MESH_DIR / "generated"
            save_path = target_dir / filename
        save_mesh_xdmf(mesh, save_path)

    return mesh


def _geodesic_wound_mask(
    domain: dolfinx.mesh.Mesh,
    burn_center: tuple[float, float, float],
    burn_radius: float,
    comm: MPI.Comm,
) -> np.ndarray:
    """Classify owned/ghost cells using distance measured along the surface.

    A Euclidean ball is not the same object as a wound on a curved surface:
    it can cut through the hand and produce a jagged, mesh-dependent footprint.
    This routine builds the edge graph of the complete distributed surface and
    uses shortest-path distances from the mesh vertex nearest to ``burn_center``.
    The returned mask is indexed by the local cell numbering, including ghosts.

    The graph is assembled on rank 0 because this operation is performed once
    during mesh construction and the hand mesh is small compared with a solve.
    No field smoothing or mechanical regularization is involved.
    """
    geom_index_map = domain.geometry.index_map()
    num_owned_vertices = geom_index_map.size_local
    owned_vertex_ids = geom_index_map.local_to_global(np.arange(num_owned_vertices, dtype=np.int32))
    owned_coords = domain.geometry.x[:num_owned_vertices].copy()

    cell_index_map = domain.topology.index_map(domain.topology.dim)
    num_owned_cells = cell_index_map.size_local
    owned_cell_ids = cell_index_map.local_to_global(np.arange(num_owned_cells, dtype=np.int32))
    owned_cells = domain.geometry.dofmaps[0][:num_owned_cells, :3]
    owned_cells_global = geom_index_map.local_to_global(owned_cells.reshape(-1)).reshape(owned_cells.shape)

    gathered_coords = comm.gather((owned_vertex_ids, owned_coords), root=0)
    gathered_cells = comm.gather((owned_cell_ids, owned_cells_global), root=0)

    if comm.rank == 0:
        num_vertices_global = geom_index_map.size_global
        full_coords = np.empty((num_vertices_global, 3), dtype=np.float64)
        for vertex_ids, coords in gathered_coords:
            full_coords[vertex_ids] = coords

        num_cells_global = cell_index_map.size_global
        full_cells = np.empty((num_cells_global, 3), dtype=np.int64)
        for cell_ids, cells in gathered_cells:
            full_cells[cell_ids] = cells

        # scipy is already a project dependency and gives a robust Dijkstra
        # implementation for the sparse surface edge graph.
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import dijkstra

        graph_edges = np.vstack(
            [
                full_cells[:, [0, 1]],
                full_cells[:, [1, 2]],
                full_cells[:, [2, 0]],
            ]
        )
        graph_edges.sort(axis=1)
        graph_edges = np.unique(graph_edges, axis=0)
        edge_lengths = np.linalg.norm(full_coords[graph_edges[:, 0]] - full_coords[graph_edges[:, 1]], axis=1)
        rows = np.concatenate([graph_edges[:, 0], graph_edges[:, 1]])
        cols = np.concatenate([graph_edges[:, 1], graph_edges[:, 0]])
        weights = np.concatenate([edge_lengths, edge_lengths])
        graph = coo_matrix((weights, (rows, cols)), shape=(num_vertices_global, num_vertices_global)).tocsr()

        center = np.asarray(burn_center, dtype=np.float64)
        seed = int(np.argmin(np.linalg.norm(full_coords - center, axis=1)))
        vertex_distances = dijkstra(graph, indices=seed)
        cell_distances = np.mean(vertex_distances[full_cells], axis=1)
        global_wound_mask = cell_distances <= burn_radius

        # Keep tiny synthetic/open meshes usable when the radius is smaller
        # than one element: the cell containing the nearest surface vertex is
        # the only geometrically defensible discrete wound in that case.
        if not np.any(global_wound_mask):
            closest_cell = int(np.argmin(cell_distances))
            global_wound_mask[closest_cell] = True
    else:
        global_wound_mask = None

    global_wound_mask = comm.bcast(global_wound_mask, root=0)
    num_local_cells = cell_index_map.size_local + cell_index_map.num_ghosts
    local_cell_ids = cell_index_map.local_to_global(np.arange(num_local_cells, dtype=np.int32))
    return np.asarray(global_wound_mask, dtype=bool)[local_cell_ids]


def _compute_signed_volume(cells: np.ndarray, coords: np.ndarray) -> float:
    """Sum of v0 . (v1 x v2) / 6 over all triangles. Positive for a
    consistently outward-oriented (CCW-from-outside) surface; a standard,
    order-independent way to pin down absolute orientation sign, robust
    even for an open surface like the hand mesh (missing wrist cap) since
    it's dominated by the closed-ish bulk of the surface."""
    v0 = coords[cells[:, 0]]
    v1 = coords[cells[:, 1]]
    v2 = coords[cells[:, 2]]
    cross = np.cross(v1, v2)
    return float(np.einsum("ij,ij->i", v0, cross).sum() / 6.0)


def _make_orientation_consistent(cells: np.ndarray) -> tuple[np.ndarray, int]:
    """BFS-propagate a single consistent orientation across a connected,
    orientable 2-manifold triangle mesh (vertex-index array, shape (n, 3)).

    Two triangles sharing an edge (a, b) are consistently oriented iff they
    traverse that shared edge in OPPOSITE directions. Plaza refinement gives
    no guarantee of preserving parent-cell handedness in its children, so
    after n refinement passes a mesh's per-cell orientation can be an
    arbitrary patchwork rather than globally consistent -- this repairs that.

    Must be called with the FULL mesh (not a per-rank partition): the BFS
    needs the complete adjacency graph to propagate correctly, and a
    partitioned subset would each get its own independent (and possibly
    disagreeing) seed orientation.
    """
    n_cells = cells.shape[0]
    cells = cells.copy()

    edge_to_cells: dict[tuple[int, int], list[int]] = defaultdict(list)
    for ci in range(n_cells):
        tri = cells[ci]
        for k in range(3):
            a, b = int(tri[k]), int(tri[(k + 1) % 3])
            edge_to_cells[tuple(sorted((a, b)))].append(ci)

    adjacency: dict[int, list[int]] = defaultdict(list)
    for cs in edge_to_cells.values():
        if len(cs) == 2:
            c0, c1 = cs
            adjacency[c0].append(c1)
            adjacency[c1].append(c0)

    def edge_dir(ci: int, a: int, b: int) -> bool:
        tri = cells[ci]
        for k in range(3):
            if tri[k] == a and tri[(k + 1) % 3] == b:
                return True
        return False

    visited = np.zeros(n_cells, dtype=bool)
    n_flipped = 0
    for start in range(n_cells):
        if visited[start]:
            continue
        visited[start] = True
        queue = deque([start])
        while queue:
            c = queue.popleft()
            for nbr in adjacency[c]:
                if visited[nbr]:
                    continue
                shared = tuple(sorted(set(cells[c].tolist()) & set(cells[nbr].tolist())))
                if len(shared) != 2:
                    # Two cells sharing >2 vertices only happens at a
                    # non-manifold junction; skip -- there's no well-defined
                    # "opposite direction" to enforce there.
                    visited[nbr] = True
                    queue.append(nbr)
                    continue
                a, b = shared
                if edge_dir(c, a, b) == edge_dir(nbr, a, b):
                    cells[nbr] = cells[nbr][[0, 2, 1]]
                    n_flipped += 1
                visited[nbr] = True
                queue.append(nbr)

    return cells, n_flipped


def _enforce_consistent_orientation(domain: dolfinx.mesh.Mesh, comm: MPI.Comm) -> dolfinx.mesh.Mesh:
    """Gather the full mesh to rank 0, run the serial BFS orientation fix,
    and rebuild+redistribute via create_mesh -- same "full data on rank 0,
    empty elsewhere, let the partitioner redistribute" pattern as the
    initial OBJ load. Must be called collectively by all ranks.
    """
    geom_im = domain.geometry.index_map()
    num_owned_nodes = geom_im.size_local
    num_owned_cells = domain.topology.index_map(2).size_local

    owned_coords = domain.geometry.x[:num_owned_nodes].copy()
    owned_global_node_ids = geom_im.local_to_global(np.arange(num_owned_nodes, dtype=np.int32))

    owned_cells_local = domain.geometry.dofmaps[0][:num_owned_cells]
    owned_cells_global = geom_im.local_to_global(owned_cells_local.reshape(-1)).reshape(owned_cells_local.shape)

    gathered_coords = comm.gather((owned_global_node_ids, owned_coords), root=0)
    gathered_cells = comm.gather(owned_cells_global, root=0)

    if comm.rank == 0:
        num_vertices_global = geom_im.size_global
        full_coords = np.empty((num_vertices_global, 3), dtype=np.float64)
        for global_ids, coords in gathered_coords:
            full_coords[global_ids] = coords

        full_cells = np.concatenate(gathered_cells, axis=0).astype(np.int64)

        fixed_cells, n_flipped = _make_orientation_consistent(full_cells)

        signed_vol = _compute_signed_volume(fixed_cells, full_coords)
        if signed_vol < 0:
            fixed_cells = fixed_cells[:, [0, 2, 1]]
            logger.info(
                "Orientation fix: %d/%d cells flipped for internal consistency; "
                "global sign was inward (signed volume %.4e) -- flipped whole mesh to outward.",
                n_flipped,
                full_cells.shape[0],
                signed_vol,
            )
        else:
            logger.info(
                "Orientation fix: %d/%d cells flipped for internal consistency; "
                "global sign already outward (signed volume %.4e).",
                n_flipped,
                full_cells.shape[0],
                signed_vol,
            )
    else:
        full_coords = np.empty((0, 3), dtype=np.float64)
        fixed_cells = np.empty((0, 3), dtype=np.int64)

    import basix.ufl
    import ufl

    c_el = basix.ufl.element("Lagrange", "triangle", 1, shape=(3,))
    ufl_domain = ufl.Mesh(c_el)

    sig = inspect.signature(dolfinx.mesh.create_mesh)
    if "e" in sig.parameters:
        new_domain = dolfinx.mesh.create_mesh(comm=comm, cells=fixed_cells, e=ufl_domain, x=full_coords)
    else:
        new_domain = dolfinx.mesh.create_mesh(comm=comm, cells=fixed_cells, x=full_coords, domain=ufl_domain)

    return new_domain


def _loop_subdivide_surface(
    vertices: np.ndarray, faces: np.ndarray, n_subdivisions: int
) -> tuple[np.ndarray, np.ndarray]:
    """Loop-subdivide a triangulated surface `n_subdivisions` times (VTK's
    loop subdivision filter, via pyvista).

    Unlike Plaza bisection (naive midpoint subdivision),
    Loop subdivision repositions *every* vertex -- old and new -- toward a
    smooth C2 limit surface on each pass, instead of inserting flat edge
    midpoints. That difference is exactly why it can reduce a facet-to-facet
    normal discontinuity inherited from a low-poly base mesh, where Plaza
    bisection provably cannot (confirmed by direct measurement: Plaza-style
    midpoint subdivision left the worst-case adjacent-facet angle on
    `mano_right.obj` completely unchanged at 145.0 deg no matter how many
    passes were applied, while Loop subdivision brought it down to
    115.0/76.2/45.0 deg after 1/2/3 passes respectively, with the bounding
    box moving by under 1% -- i.e. genuinely smoothing, not just shrinking).

    A no-op (returns the inputs unchanged) when `n_subdivisions <= 0`.
    """
    if n_subdivisions <= 0:
        return vertices, faces

    import pyvista as pv

    pv_faces = np.column_stack([np.full(len(faces), 3, dtype=np.int64), faces]).ravel()
    dense = pv.PolyData(vertices, pv_faces).subdivide(nsub=n_subdivisions, subfilter="loop")
    dense_faces = dense.faces.reshape(-1, 4)[:, 1:].astype(np.int64)
    return np.asarray(dense.points, dtype=np.float64), dense_faces


def _wound_size_field(
    vertices: np.ndarray,
    burn_center: tuple[float, float, float],
    burn_radius: float,
    char_length_burn: float,
    char_length_far: float,
    buffer_radius: float,
    transition_width: float,
) -> np.ndarray:
    """Per-vertex target edge length h(x) for MMGS metric-driven remeshing:
    `char_length_burn` inside the wound and a `buffer_radius`-wide halo
    around it, tapering linearly out to `char_length_far` over the next
    `transition_width`, and held at `char_length_far` beyond that.
    """
    dist = np.linalg.norm(vertices - np.asarray(burn_center), axis=1)
    dist_past_buffer = np.maximum(dist - burn_radius - buffer_radius, 0.0)
    weight = np.clip(dist_past_buffer / max(transition_width, 1e-300), 0.0, 1.0)
    return char_length_burn + weight * (char_length_far - char_length_burn)


def _mmgs_remesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    size_field: np.ndarray,
    hgrad: float = 1.3,
    hausd: float = 0.001,
) -> tuple[np.ndarray, np.ndarray]:
    """Metric-driven adaptive surface remeshing via the external `mmgs_O3`
    CLI (MMG surface remesher), given a per-vertex target edge length field.

    Runs serially through temporary MEDIT (`.mesh`/`.sol`) files via
    `meshio` -- call only on rank 0, on a fully gathered (not
    per-rank-partitioned) mesh, same pattern as
    `_enforce_consistent_orientation`. `hgrad` caps the growth ratio between
    adjacent element edges (how fast the mesh may coarsen away from the
    wound); `hausd` is the maximum Hausdorff distance MMGS may deviate from
    the input surface while remeshing, i.e. how tightly it must track
    existing curvature.

    Raises:
        RuntimeError: If the `mmgs_O3` executable is missing from PATH, or
            it exits non-zero (e.g. malformed/non-manifold input).
    """
    import shutil
    import subprocess
    import tempfile
    from pathlib import Path as _Path

    import meshio

    if shutil.which("mmgs_O3") is None:
        raise RuntimeError(
            "The 'mmgs_O3' executable was not found on PATH. It is provided by the "
            "'mmgsuite' pixi dependency -- make sure this runs inside the pixi "
            "environment (e.g. via `pixi run ...`) so its bin directory is on PATH."
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = _Path(tmpdir)
        input_mesh = tmp / "input.mesh"
        input_sol = tmp / "input.sol"
        output_mesh = tmp / "output.mesh"

        meshio.write(str(input_mesh), meshio.Mesh(points=vertices, cells=[("triangle", faces)]), file_format="medit")

        with open(input_sol, "w") as f:
            f.write("MeshVersionFormatted 2\nDimension 3\n\nSolAtVertices\n")
            f.write(f"{len(size_field)}\n1 1\n\n")
            for h in size_field:
                f.write(f"{h:.6e}\n")
            f.write("\nEnd\n")

        cmd = [
            "mmgs_O3",
            str(input_mesh),
            "-sol",
            str(input_sol),
            "-hgrad",
            str(hgrad),
            "-hausd",
            str(hausd),
            "-out",
            str(output_mesh),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0 or not output_mesh.exists():
            raise RuntimeError(
                f"mmgs_O3 remeshing failed (exit code {result.returncode}):\n{result.stdout}\n{result.stderr}"
            )

        remeshed = meshio.read(str(output_mesh))

    remeshed_faces = remeshed.cells_dict.get("triangle")
    if remeshed_faces is None or len(remeshed_faces) == 0:
        raise RuntimeError("mmgs_O3 output contained no triangle cells.")
    return np.asarray(remeshed.points, dtype=np.float64), np.asarray(remeshed_faces, dtype=np.int64)


def generate_hand_surface_mesh_direct(
    mesh_path: str | Path,
    burn_center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    burn_radius: float = 0.015,
    char_length_far: float = 0.008,
    char_length_burn: float = 0.0015,
    refinement_radius: float = 0.03,
    scale_factor: float = 1.0,
    use_mmg_remeshing: bool = True,
    loop_subdivisions: int = 3,
    mmg_buffer_radius: float | None = None,
    mmg_transition_width: float | None = None,
    mmg_hausd: float | None = None,
    mmg_hgrad: float = 1.3,
    comm: MPI.Comm = MPI.COMM_WORLD,
    save: bool = False,
    save_path: str | Path | None = None,
    flip_normals: bool = False,
    **kwargs: Any,
) -> TaggedMesh:
    """Load MANO .obj directly into DOLFINx (tdim=2, gdim=3), remesh near the
    wound, and compute sub-domain tags.

    When `use_mmg_remeshing` is set (the default), the pipeline is:

    1. Loop-subdivide the raw input mesh `loop_subdivisions` times
       (`_loop_subdivide_surface`) to get a dense, genuinely *smooth* surface
       -- unlike Plaza bisection (naive midpoint subdivision), Loop
       subdivision repositions every vertex toward a smooth limit surface
       each pass, so it actually removes the facet-to-facet normal
       discontinuities inherited from MANO's low-poly base triangulation
       instead of just subdividing them unchanged.
    2. Build a per-vertex target edge-length field (`_wound_size_field`):
       `char_length_burn` inside the wound plus a `mmg_buffer_radius` halo,
       tapering to `char_length_far` over the next `mmg_transition_width`.
    3. Feed the dense mesh and that field to `mmgs_O3` (`_mmgs_remesh`), an
       external metric-driven surface remesher, producing a single mesh
       that is fine at the wound, coarse far away, and well-graded in
       between.

    Set `use_mmg_remeshing=False` to skip all of the above and use the raw
    (scaled/flip-normals-adjusted) input mesh as-is -- useful for fast,
    dependency-free tests of the tagging logic on tiny synthetic meshes
    where a size field tuned for a real hand-scale wound wouldn't make
    sense. `mmg_buffer_radius`/`mmg_transition_width` default to
    `burn_radius` (each) when not given.

    `refinement_radius` is accepted for interface parity with the other
    generators but is currently unused here.
    """
    path = Path(mesh_path)
    if not path.exists():
        raise FileNotFoundError(f"Surface mesh file not found at {path}")

    # 1. Load mesh using Trimesh on rank 0
    if comm.rank == 0:
        import trimesh

        tri_mesh = trimesh.load(str(path), process=False)

        vertices = np.ascontiguousarray(tri_mesh.vertices, dtype=np.float64) * scale_factor
        if flip_normals:
            faces = np.ascontiguousarray(np.fliplr(tri_mesh.faces), dtype=np.int64)
        else:
            faces = np.ascontiguousarray(tri_mesh.faces, dtype=np.int64)

        if use_mmg_remeshing:
            vertices, faces = _loop_subdivide_surface(vertices, faces, loop_subdivisions)
            size_field = _wound_size_field(
                vertices,
                burn_center=burn_center,
                burn_radius=burn_radius,
                char_length_burn=char_length_burn,
                char_length_far=char_length_far,
                buffer_radius=mmg_buffer_radius if mmg_buffer_radius is not None else burn_radius,
                transition_width=mmg_transition_width if mmg_transition_width is not None else burn_radius,
            )
            logger.info(
                "Loop-subdivided to %d verts / %d faces; running mmgs_O3 adaptive remeshing...",
                len(vertices),
                len(faces),
            )
            # hausd is an *absolute* geometric tolerance, but burn_radius/
            # char_length_* are given in whatever (possibly non-metric)
            # scaled units the caller uses -- a hardcoded default tuned for
            # real meters would be wildly mismatched against, e.g., cm-like
            # nondimensional units, forcing mmgs_O3 to fight a near-impossible
            # combination of a coarse size field and a razor-tight tolerance.
            # Scaling it off char_length_burn keeps it self-consistent.
            effective_hausd = mmg_hausd if mmg_hausd is not None else 0.1 * char_length_burn
            vertices, faces = _mmgs_remesh(vertices, faces, size_field, hgrad=mmg_hgrad, hausd=effective_hausd)
            logger.info("mmgs_O3 remeshing done: %d verts / %d faces.", len(vertices), len(faces))
    else:
        vertices = np.empty((0, 3), dtype=np.float64)
        faces = np.empty((0, 3), dtype=np.int64)

    # NOTE: deliberately no comm.bcast here. create_mesh expects the "full
    # data on rank 0, empty elsewhere" pattern set up above -- its internal
    # partitioner distributes cells across ranks from that. Broadcasting the
    # full arrays to every rank (as this used to do) gives every process a
    # complete duplicate of the mesh instead of nothing, which the
    # partitioner can't handle once there's more than one rank; it only
    # went unnoticed in serial because bcast-to-self is a no-op.

    # Guarantee 2D coordinate array shape (N, 3) for gdim=3
    vertices = np.atleast_2d(vertices).astype(np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"Expected coordinate array shape (N, 3), got {vertices.shape}")

    # 2. Construct UFL Domain & DOLFINx Mesh (tdim=2, gdim=3)
    import ufl

    # Define a 3D coordinate element for the 2D triangular surface
    try:
        import basix.ufl

        c_el = basix.ufl.element("Lagrange", "triangle", 1, shape=(3,))
    except ImportError:
        # Fallback for older FEniCSx versions
        c_el = ufl.VectorElement("Lagrange", "triangle", 1, dim=3)

    ufl_domain = ufl.Mesh(c_el)

    # Handle breaking API changes in dolfinx.mesh.create_mesh (v0.7.x vs v0.8.x+)
    sig = inspect.signature(dolfinx.mesh.create_mesh)
    if "e" in sig.parameters:
        # DOLFINx >= 0.8.0 signature
        domain = dolfinx.mesh.create_mesh(comm=comm, cells=faces, e=ufl_domain, x=vertices)
    else:
        # DOLFINx < 0.8.0 signature
        domain = dolfinx.mesh.create_mesh(comm=comm, cells=faces, x=vertices, domain=ufl_domain)

    if comm.rank == 0:
        logger.info("Hand mesh created (%d ranks). Enforcing consistent cell orientation...", comm.size)

    # mmgs_O3 (like Plaza refinement) gives no guarantee of preserving
    # consistent per-cell handedness -- repeated remeshing operations can
    # leave orientation as an arbitrary per-cell patchwork. ufl.CellNormal
    # (used directly in BurnProblemSurface's subcutaneous foundation and
    # hypertrophic bulging terms) depends entirely
    # on this, so a patchwork here corrupts those forces locally and produces
    # a wobbly, physically incoherent final surface. Must run before tagging,
    # since create_mesh below discards the old MeshTags along with the old
    # connectivity.
    domain = _enforce_consistent_orientation(domain, comm)

    if comm.rank == 0:
        logger.info("Orientation fix done. Computing cell/facet tags...")

    # 3. Compute Cell Tags (WOUND vs HEALTHY) via surface-geodesic distance
    #
    # wound_mask must cover ghost cells, not just owned ones: a facet on a
    # partition boundary has one owned neighbor and one ghost neighbor (a
    # mirror of a cell another rank owns), and dolfinx's local cell numbering
    # puts ghosts *after* the owned range (indices size_local..size_local+
    # num_ghosts-1). The facet-tagging loop below indexes wound_mask by
    # those neighbor indices directly, so an array sized to size_local alone
    # goes out of bounds the moment a facet's neighbor is a ghost.
    num_cells_owned = domain.topology.index_map(2).size_local
    num_ghosts = domain.topology.index_map(2).num_ghosts
    num_cells_total = num_cells_owned + num_ghosts

    # A wound is a patch on the hand surface, not the intersection of an
    # ambient 3-D ball with a folded surface.  Use surface geodesic distance
    # for the final classification; the Euclidean distance remains suitable
    # for selecting a conservative refinement region above.
    wound_mask = _geodesic_wound_mask(domain, burn_center, burn_radius, comm)
    if len(wound_mask) != num_cells_total:
        raise RuntimeError("Geodesic wound classification returned an invalid local cell mask.")

    # meshtags themselves only need values for cells this rank owns.
    cell_indices = np.arange(num_cells_owned, dtype=np.int32)
    cell_values = np.full(num_cells_owned, CellTag.HEALTHY, dtype=np.int32)
    cell_values[wound_mask[:num_cells_owned]] = CellTag.WOUND

    cell_tags = dolfinx.mesh.meshtags(domain, 2, cell_indices, cell_values)

    # 4. Compute Facet Tags: WOUND_BOUNDARY (wound/healthy interface) and
    #    OUTER_BOUNDARY (free edge of the surface, e.g. a wrist cut on the
    #    MANO mesh). A facet with exactly 2 neighboring cells is interior to
    #    the surface; one with wound/healthy neighbors on either side is the
    #    wound interface. The physical free edge is obtained separately with
    #    `exterior_facet_indices`; this is important on distributed meshes,
    #    where a partition-interface facet may expose only one local cell.
    #    If the surface is watertight (e.g. a full closed glove shape with no
    #    wrist opening), there simply is no such boundary -- see the warning
    #    below.
    domain.topology.create_connectivity(2, 1)
    domain.topology.create_connectivity(1, 2)
    f2c = domain.topology.connectivity(1, 2)

    num_facets = domain.topology.index_map(1).size_local
    interface_facets = []
    # `f2c.links(f)` is not sufficient for identifying the physical boundary
    # on a distributed mesh: a facet on a partition boundary can expose only
    # its local cell and therefore look like a one-cell boundary facet.  Use
    # DOLFINx's distributed exterior-facet routine for the actual free edge,
    # and reserve the manual cell-neighbour inspection for wound interfaces.
    outer_facets = dolfinx.mesh.exterior_facet_indices(domain.topology).tolist()

    for f in range(num_facets):
        neighbors = f2c.links(f)
        if len(neighbors) == 2:
            w0 = wound_mask[neighbors[0]]
            w1 = wound_mask[neighbors[1]]
            if w0 != w1:
                interface_facets.append(f)

    n_outer_global = domain.comm.allreduce(len(outer_facets), op=MPI.SUM)
    if n_outer_global == 0:
        # Checked globally, not per-rank: with the wrist boundary occupying
        # only a small part of the mesh, most ranks legitimately own zero
        # outer-boundary facets after partitioning even when the mesh
        # unambiguously has a free boundary somewhere. A per-rank check here
        # would fire this warning on those ranks even though the mesh as a
        # whole is fine -- only the true "nowhere on any rank" case should warn.
        import warnings

        warnings.warn(
            "No free-boundary facets found anywhere on this surface mesh -- it appears "
            "to be watertight (closed), so no FacetTag.OUTER_BOUNDARY facets were tagged. "
            "Dirichlet BCs keyed on OUTER_BOUNDARY (e.g. in BurnProblem2D) will have "
            "nothing to attach to on this mesh; you'll need an alternative BC strategy "
            "(e.g. pinning a reference patch to remove rigid-body modes) if that's the case.",
            stacklevel=2,
        )

    facet_indices = np.array(interface_facets + outer_facets, dtype=np.int32)
    facet_values = np.concatenate(
        [
            np.full(len(interface_facets), FacetTag.WOUND_BOUNDARY, dtype=np.int32),
            np.full(len(outer_facets), FacetTag.OUTER_BOUNDARY, dtype=np.int32),
        ]
    )

    # dolfinx.mesh.meshtags requires indices in sorted order; interface_facets
    # and outer_facets are each individually ascending (single pass over f in
    # increasing order) but interleaved with each other, so the concatenation
    # isn't globally sorted -- fix that up explicitly.
    sort_idx = np.argsort(facet_indices)
    facet_tags = dolfinx.mesh.meshtags(domain, 1, facet_indices[sort_idx], facet_values[sort_idx])

    # Gmsh-generated meshes (the other generators) get several topology
    # connectivities built as a side effect of dfx_gmsh.model_to_mesh's
    # partitioning step. This mesh is built directly via create_mesh, so
    # nothing populates them automatically -- including (dim, dim), which
    # locate_dofs_topological needs whenever entity_dim == topology.dim
    # (e.g. BurnProblem2D.define_initial_conditions tagging wound cells).
    # Conceptually this is just the trivial cell-to-itself map, but dolfinx
    # still requires it to be explicitly created.
    domain.topology.create_connectivity(domain.topology.dim, domain.topology.dim)

    if comm.rank == 0:
        logger.info("Tagging done.")

    tagged_mesh = TaggedMesh(
        mesh=domain,
        cell_tags=cell_tags,
        facet_tags=facet_tags,
        wound_shape="hand_mano_direct",
    )

    if save and save_path is not None:
        if comm.rank == 0:
            logger.info("Saving mesh to %s...", save_path)
        save_mesh_xdmf(tagged_mesh, save_path)
        if comm.rank == 0:
            logger.info("Saved mesh to %s.", save_path)

    return tagged_mesh
