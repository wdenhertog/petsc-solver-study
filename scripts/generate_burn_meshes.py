"""Generate the frozen mesh ladder for the burn-injury solver study.

Run ONCE, serially, in the burn_injuries environment (needs dolfinx + gmsh +
the burn_injuries package):

    pixi run -e <env> python scripts/generate_burn_meshes.py \
        --out meshes/burn --levels 0-10

Meshes are written as XDMF/HDF5 (cells + cell tags + facet tags) and indexed in
manifest.json (sha256, sizes, gmsh version). Benchmark runs must only *load*
these files, never regenerate them: gmsh output is not guaranteed identical
across versions/platforms, so the files are the single source of truth.

Level k uses h_far = H0 * 2**(-k/2), h_wound = h_far / 2, so nodes roughly
double per level. Every level is generated independently (no nesting).
"""

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import gmsh
from burn_injuries.meshes import generate_2d_surface_mesh, generate_curved_surface_mesh
from burn_injuries.meshes.io import save_mesh_xdmf
from mpi4py import MPI

H0 = 0.4
DOMAIN = 10.0
CENTER = (DOMAIN / 2.0, DOMAIN / 2.0)
WOUND_RADIUS = 2.0
BUMP = {"amplitude": 1.5, "sigma": 2.5, "center": CENTER}


def parse_levels(spec: str) -> list[int]:
    levels: list[int] = []
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        levels.extend(range(int(lo), int(hi or lo) + 1))
    return sorted(set(levels))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def build(case: str, h_far: float):
    common = {
        "domain_size": (DOMAIN, DOMAIN),
        "wound_shape": "circle",
        "wound_center": CENTER,
        "wound_radius": WOUND_RADIUS,
        "char_length_far": h_far,
        "char_length_wound": h_far / 2.0,
        "comm": MPI.COMM_SELF,
    }
    if case == "planar":
        return generate_2d_surface_mesh(**common)
    return generate_curved_surface_mesh(
        height_function="gaussian_bump", surface_params=BUMP, **common
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", type=Path, default=Path("meshes/burn"))
    ap.add_argument("--levels", default="0-10", help="e.g. '0-10' or '0,2,4-6'")
    ap.add_argument("--cases", default="planar,curved")
    ap.add_argument("--force", action="store_true", help="regenerate existing files")
    args = ap.parse_args()

    if MPI.COMM_WORLD.size != 1:
        sys.exit("Run serially (no mpirun).")

    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text())
        if manifest_path.exists()
        else {"meshes": {}}
    )
    manifest["environment"] = {
        "gmsh": gmsh.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "H0": H0,
        "domain": DOMAIN,
        "wound_radius": WOUND_RADIUS,
        "bump": {k: v for k, v in BUMP.items()},
    }

    for case in args.cases.split(","):
        for level in parse_levels(args.levels):
            name = f"burn_{case}_L{level:02d}"
            path = args.out / f"{name}.xdmf"
            if path.exists() and not args.force:
                print(f"skip {name} (exists)")
                continue
            h_far = H0 * 2.0 ** (-level / 2.0)
            t0 = time.time()
            tm = build(case, h_far)
            save_mesh_xdmf(tm, path)
            n_vert = tm.num_vertices
            manifest["meshes"][name] = {
                "case": case,
                "level": level,
                "file": path.name,
                "h_far": h_far,
                "h_wound": h_far / 2.0,
                "tdim": tm.tdim,
                "gdim": tm.gdim,
                "num_cells": tm.num_cells,
                "num_vertices": n_vert,
                "approx_dofs": n_vert * (4 + tm.gdim + tm.gdim * (tm.gdim + 1) // 2),
                "sha256_h5": sha256(path.with_suffix(".h5")),
                "gen_seconds": round(time.time() - t0, 1),
            }
            manifest_path.write_text(json.dumps(manifest, indent=2))
            print(
                f"{name}: {n_vert} vertices, ~{manifest['meshes'][name]['approx_dofs']} DOFs"
            )


if __name__ == "__main__":
    main()
