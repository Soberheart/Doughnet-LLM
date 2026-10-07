"""Recover the missing board mesh from the repository's existing SDF.

Run on a CPU from the project root:
    python scripts/restore_board_mesh.py

This extracts a zero surface using marching tetrahedra and maps voxel indices
back to mesh coordinates. It does not reproduce the author's original OBJ
triangulation. Keep the provenance JSON when reporting simulation results.
Only NumPy is required; the input SDF is the trusted pickle supplied with this
repository. Existing output files are never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED = PROJECT_ROOT / "sim/mpm/assets/meshes/processed"
CORNERS = np.array([
    [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
    [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1],
], dtype=np.int64)
TETRAHEDRA = (
    (0, 1, 3, 7), (0, 3, 2, 7), (0, 2, 6, 7),
    (0, 6, 4, 7), (0, 4, 5, 7), (0, 5, 1, 7),
)


def extract_mesh(values, mesh_to_voxels):
    values = np.asarray(values, dtype=np.float64)
    transform = np.asarray(mesh_to_voxels, dtype=np.float64)
    if values.ndim != 3 or min(values.shape) < 3 or not np.isfinite(values).all():
        raise ValueError("SDF must be a finite three-dimensional grid.")
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("Expected a finite 4 x 4 mesh-to-voxel transform.")
    if not np.allclose(transform[3], [0, 0, 0, 1]):
        raise ValueError("Only affine coordinate transforms are supported.")
    for axis in range(3):
        for side in (0, -1):
            if np.any(np.take(values, side, axis=axis) <= 0):
                raise ValueError("The surface reaches the grid boundary; cannot recover a closed mesh.")
    if np.any(values == 0):
        raise ValueError("Exact zero grid samples need special handling; this extractor refuses them.")

    inverse = np.linalg.inv(transform)
    negative = values < 0
    cell_shape = tuple(n - 1 for n in values.shape)
    any_inside = np.zeros(cell_shape, dtype=bool)
    all_inside = np.ones(cell_shape, dtype=bool)
    for offset in CORNERS:
        slices = tuple(slice(int(k), int(k) + n) for k, n in zip(offset, cell_shape))
        corner_inside = negative[slices]
        any_inside |= corner_inside
        all_inside &= corner_inside
    cells = np.argwhere(any_inside & ~all_inside)
    if len(cells) == 0:
        raise ValueError("No zero surface found in the SDF.")

    corner_indices = cells[:, None, :] + CORNERS
    corner_ids = np.ravel_multi_index(corner_indices.transpose(2, 0, 1), values.shape)
    corner_values = values.ravel()[corner_ids]
    triangle_edges, outward_directions = [], []
    for tetra in TETRAHEDRA:
        ids = corner_ids[:, tetra]
        points = corner_indices[:, tetra]
        inside = corner_values[:, tetra] < 0
        counts = inside.sum(axis=1)
        for count in (1, 2, 3):
            mask = counts == count
            if not mask.any():
                continue
            # Inside corners first, then outside corners, in stable index order.
            order = np.argsort(~inside[mask], axis=1, kind="stable")
            sorted_ids = np.take_along_axis(ids[mask], order, axis=1)
            sorted_points = np.take_along_axis(points[mask], order[..., None], axis=1)
            outward = (sorted_points[:, count:].mean(axis=1)
                       - sorted_points[:, :count].mean(axis=1)) @ inverse[:3, :3].T
            if count == 1:
                pairs = ((0, 1), (0, 2), (0, 3))
                edge_triangles = [pairs]
            elif count == 3:
                pairs = ((3, 0), (3, 1), (3, 2))
                edge_triangles = [pairs]
            else:
                edge_triangles = [((0, 2), (0, 3), (1, 2)),
                                  ((0, 3), (1, 3), (1, 2))]
            for pairs in edge_triangles:
                triangle_edges.append(np.stack([sorted_ids[:, pair] for pair in pairs], axis=1))
                outward_directions.append(outward)

    edges = np.concatenate(triangle_edges)
    edges.sort(axis=-1)
    unique_edges, indices = np.unique(edges.reshape(-1, 2), axis=0, return_inverse=True)
    endpoints = values.ravel()[unique_edges]
    fractions = -endpoints[:, 0] / (endpoints[:, 1] - endpoints[:, 0])
    a = np.stack(np.unravel_index(unique_edges[:, 0], values.shape), axis=-1)
    b = np.stack(np.unravel_index(unique_edges[:, 1], values.shape), axis=-1)
    voxel_vertices = a + fractions[:, None] * (b - a)
    vertices = voxel_vertices @ inverse[:3, :3].T + inverse[:3, 3]
    faces = indices.reshape(-1, 3)
    normals = np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]],
                       vertices[faces[:, 2]] - vertices[faces[:, 0]])
    outward = np.concatenate(outward_directions)
    flip = np.einsum("ij,ij->i", normals, outward) < 0
    faces[flip] = faces[flip][:, [0, 2, 1]]
    if not np.isfinite(vertices).all() or np.any(np.linalg.norm(normals, axis=1) == 0):
        raise ValueError("Recovered mesh contains invalid or degenerate triangles.")

    directed_edges = faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2)
    _, edge_inverse, edge_counts = np.unique(
        np.sort(directed_edges, axis=1), axis=0, return_inverse=True, return_counts=True)
    if np.any(edge_counts != 2):
        raise ValueError("Recovered mesh is not closed: each edge must have two faces.")
    winding = np.bincount(edge_inverse,
                          weights=np.where(directed_edges[:, 0] < directed_edges[:, 1], 1, -1))
    if np.any(winding != 0):
        raise ValueError("Recovered mesh has inconsistent face orientation.")
    volume = float(np.einsum("ij,ij->i", vertices[faces[:, 0]],
                            np.cross(vertices[faces[:, 1]], vertices[faces[:, 2]])).sum() / 6)
    if volume <= 0:
        raise ValueError("Expected a positive enclosed mesh volume.")
    return vertices, faces, volume


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdf", type=Path, default=PROCESSED / "board-128.sdf")
    parser.add_argument("--output", type=Path, default=PROCESSED / "board-board.obj")
    args = parser.parse_args()
    metadata_path = args.output.with_suffix(".provenance.json")
    for path in (args.output, metadata_path):
        if path.exists():
            parser.error(f"Output already exists; refusing to overwrite: {path}")
    raw = args.sdf.read_bytes()
    source = pickle.loads(raw)
    vertices, faces, volume = extract_mesh(source["voxels"], source["T_mesh_to_voxels"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="ascii", newline="\n") as stream:
        stream.write("# Recovered from the supplied SDF using marching tetrahedra.\n")
        stream.write("# This is not the original author's OBJ file.\n")
        for x, y, z in vertices:
            stream.write(f"v {x:.12g} {y:.12g} {z:.12g}\n")
        for a, b, c in faces + 1:
            stream.write(f"f {a} {b} {c}\n")
    metadata = {
        "source_sdf": str(args.sdf.resolve()),
        "source_sdf_sha256": hashlib.sha256(raw).hexdigest(),
        "output_obj": str(args.output.resolve()),
        "output_obj_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "method": "marching_tetrahedra_at_zero_with_inverse_mesh_to_voxels_transform",
        "original_author_obj": False,
        "vertices": len(vertices), "triangles": len(faces),
        "bounds": [vertices.min(axis=0).tolist(), vertices.max(axis=0).tolist()],
        "enclosed_volume": volume,
        "closed_and_consistently_oriented": True,
    }
    with metadata_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(metadata, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(f"Recovered mesh: {args.output}")
    print(f"Vertices: {len(vertices)}; triangles: {len(faces)}")
    print(f"Bounds: {metadata['bounds']}")
    print("Closed mesh and face orientation: PASS")
    print(f"Provenance: {metadata_path}")
    print("READY FOR SIMULATOR INTERFACE CHECK (recovered asset, not original OBJ)")


if __name__ == "__main__":
    main()
