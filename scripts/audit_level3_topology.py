"""Audit processed Level 3 surface topology against simulator annotations.

CPU only. Uses existing diagnostic metadata and source meshes without repairing,
smoothing, or remeshing them. Removes zero-area face padding and welds exactly
equal stored vertex coordinates before computing connectivity and Euler counts.
Open3D checks manifoldness, orientability, and self-intersections. Genus remains
unresolved if those checks fail. This audits the processed surface, not the raw
particle volume or the simulator's graph-based topology detector.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import pickle
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.check_level3_dyn import COLORS, sha256


def prepare_surface(vertices, faces):
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError("Invalid surface vertices")
    if faces.ndim != 2 or faces.shape[1] != 3 or not np.issubdtype(faces.dtype, np.integer):
        raise ValueError("Invalid surface triangles")
    if faces.size == 0:
        return vertices[:0], faces.astype(np.int64), 0, 0
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError("Out-of-range surface triangles")
    triangles = vertices[faces]
    valid = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                    triangles[:, 2] - triangles[:, 0]), axis=1) > 0
    removed = int((~valid).sum())
    if not valid.any():
        return vertices[:0], faces[:0], removed, 0
    coordinates = vertices[faces[valid]].reshape(-1, 3)
    welded, inverse = np.unique(coordinates, axis=0, return_inverse=True)
    welded_faces = inverse.reshape(-1, 3)
    duplicate_count = len(welded_faces) - len(np.unique(np.sort(welded_faces, axis=1), axis=0))
    return welded, welded_faces, removed, int(duplicate_count)


def surface_counts(faces):
    """Euler counts from the used vertices and unoriented unique edges."""
    faces = np.asarray(faces)
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    _, incidence = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
    v, e, f = len(np.unique(faces)), len(incidence), len(faces)
    return {"vertices": v, "edges": e, "triangles": f, "euler_characteristic": v - e + f,
            "boundary_edges": int((incidence == 1).sum()),
            "nonmanifold_edges": int((incidence > 2).sum())}


def audit_surface(o3d, vertices, faces):
    vertices, faces, removed, duplicates = prepare_surface(vertices, faces)
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices),
                                    o3d.utility.Vector3iVector(faces))
    if len(faces) == 0:
        return mesh, {"closed_edge_manifold": None, "vertex_manifold": None,
                      "orientable": None, "self_intersecting": None,
                      "removed_zero_area_triangles": removed, "duplicate_triangles": duplicates,
                      "surface_components": [], "surface_component_count": 0,
                      "reason": "no nonzero-area surface triangles"}
    clusters, cluster_counts, _ = mesh.cluster_connected_triangles()
    clusters = np.asarray(clusters)
    checks = {
        "closed_edge_manifold": bool(mesh.is_edge_manifold(allow_boundary_edges=False)),
        "vertex_manifold": bool(mesh.is_vertex_manifold()),
        "orientable": bool(mesh.is_orientable()),
        "self_intersecting": bool(mesh.is_self_intersecting()),
    }
    eligible = (checks["closed_edge_manifold"] and checks["vertex_manifold"]
                and checks["orientable"] and not checks["self_intersecting"] and duplicates == 0)
    components = []
    for index in range(len(cluster_counts)):
        counts = surface_counts(faces[clusters == index])
        chi = counts["euler_characteristic"]
        counts["genus"] = (2 - chi) // 2 if eligible and chi <= 2 and chi % 2 == 0 else None
        components.append(counts)
    return mesh, {**checks, "removed_zero_area_triangles": removed,
                  "duplicate_triangles": duplicates, "surface_components": components,
                  "surface_component_count": len(components)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dyn-artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="A directory that does not already exist")
    parser.add_argument("--scene-row", type=int, default=0)
    parser.add_argument("--frames", type=int, nargs="+", default=[0, 30, 35, 45, 50, 55, 60, 62])
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Choose a new output directory: {args.output}")
    root = args.dyn_artifacts.resolve()
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("kind") != "recorded_action_interface_diagnostic":
        raise ValueError("Expected artifacts from check_level3_dyn.py")
    processed_path = Path(metadata["processed"])
    log_path = Path(metadata["scene_dir"]) / "log.pkl"
    if sha256(processed_path) != metadata["processed_sha256"] or sha256(log_path) != metadata["source_log_sha256"]:
        raise ValueError("Source scene changed since the Dyn diagnostic")

    import hdf5plugin  # Register source compression filters.
    import h5py
    import open3d as o3d

    with h5py.File(processed_path, "r") as stream:
        if not 0 <= args.scene_row < stream["frame"].shape[0]:
            raise ValueError("Invalid processed scene row")
        data = {key: stream[key][args.scene_row] for key in (
            "scene", "frame", "obj_verts", "obj_faces", "obj_face_labels", "genus", "ee_verts", "ee_faces",
        )}
    count = len(data["frame"])
    with log_path.open("rb") as stream:
        logs = pickle.load(stream)
    if len(logs) < count or count != metadata["num_source_frames"]:
        raise ValueError("Processed frame padding or metadata mismatch")
    logs = logs[-count:]
    if not np.array_equal(data["frame"], np.arange(count)):
        raise ValueError("Processed frames must be consecutive from zero")
    frames = sorted(set(args.frames))
    if not frames or frames[0] < 0 or frames[-1] >= count:
        raise ValueError("Selected frame outside processed sequence")
    if not np.isfinite(data["obj_verts"]).all() or not np.isfinite(data["ee_verts"]).all():
        raise ValueError("Nonfinite processed mesh")
    rows, frame_results = [], []
    args.output.mkdir(parents=True, exist_ok=False)
    for frame in frames:
        genus = data["genus"][frame]
        if (genus.shape != (len(COLORS),) or not np.issubdtype(genus.dtype, np.integer)
                or genus.min() < -1 or genus[0] != -1):
            raise ValueError(f"Invalid genus slots at frame {frame}")
        annotation = genus[genus >= 0].tolist()
        raw_annotation = logs[frame]["topology"]["genus"]
        if sorted(annotation) != sorted(raw_annotation):
            raise ValueError(f"Processed and simulator genus annotations differ at frame {frame}")
        labels = data["obj_face_labels"][frame].reshape(-1)
        faces = data["obj_faces"][frame]
        if (labels.shape != (len(faces),) or not np.issubdtype(labels.dtype, np.integer)
                or labels.min() < 1 or labels.max() >= len(COLORS)):
            raise ValueError(f"Invalid object face labels at frame {frame}")
        _, full_result = audit_surface(o3d, data["obj_verts"][frame], faces)
        parts, colored_mesh = [], o3d.geometry.TriangleMesh()
        for label in sorted(set(np.unique(labels).tolist()) | set(np.flatnonzero(genus >= 0).tolist())):
            label_faces = faces[labels == label]
            if len(label_faces) == 0:
                part = {"label": label, "annotated_genus": int(genus[label]),
                        "status": "unresolved", "reason": "annotation has no mesh faces"}
            else:
                part_mesh, result = audit_surface(o3d, data["obj_verts"][frame], label_faces)
                part_mesh.paint_uniform_color((COLORS[label].astype(float) / 255).tolist())
                colored_mesh += part_mesh
                computed = [component["genus"] for component in result["surface_components"]]
                if len(computed) == 1 and computed[0] is not None and genus[label] >= 0:
                    status = "matches" if computed[0] == genus[label] else "differs"
                else:
                    status = "unresolved"
                part = {"label": label, "annotated_genus": int(genus[label]), "status": status, **result}
            parts.append(part)
            row = {"frame": frame, "step": int(logs[frame]["step"]), "label": label,
                   "annotated_genus": part["annotated_genus"], "status": part["status"],
                   "reason": part.get("reason", ""),
                   "surface_component_count": part.get("surface_component_count"),
                   "surface_genus": json.dumps([component["genus"] for component in part.get("surface_components", [])]),
                   **{key: part.get(key) for key in ("closed_edge_manifold", "vertex_manifold", "orientable",
                                                    "self_intersecting", "duplicate_triangles")}}
            rows.append(row)
        colored_mesh.compute_vertex_normals()
        if not o3d.io.write_triangle_mesh(str(args.output / f"gt_{frame:03d}.ply"), colored_mesh):
            raise RuntimeError("Failed to export object mesh")
        tool_vertices, tool_faces, _, _ = prepare_surface(data["ee_verts"][frame], data["ee_faces"][frame])
        tool_mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(tool_vertices),
                                             o3d.utility.Vector3iVector(tool_faces))
        tool_mesh.paint_uniform_color([0.5, 0.5, 0.5])
        tool_mesh.compute_vertex_normals()
        if not o3d.io.write_triangle_mesh(str(args.output / f"tool_{frame:03d}.ply"), tool_mesh):
            raise RuntimeError("Failed to export tool mesh")
        frame_result = {"frame": frame, "step": int(logs[frame]["step"]),
                        "annotation_components": int(logs[frame]["topology"]["components"]),
                        "annotation_genus": annotation, "whole_surface": full_result, "labels": parts}
        frame_results.append(frame_result)
        print(f"frame={frame} annotation={annotation} " + " ".join(
            f"label={part['label']} surface_genus="
            f"{[component['genus'] for component in part.get('surface_components', [])]} {part['status']}"
            for part in parts), flush=True)

    with (args.output / "mesh_topology.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "kind": "processed_surface_topology_audit", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dyn_artifacts": str(root), "scene_row": args.scene_row,
        "processed_sha256": metadata["processed_sha256"], "source_log_sha256": metadata["source_log_sha256"],
        "script_sha256": sha256(__file__), "environment": {"numpy": np.__version__, "open3d": o3d.__version__},
        "method": "Exact coordinate welding; zero-area padding excluded; closed manifold orientable nonintersecting surfaces; g=(2-chi)/2 per connected surface",
        "status_semantics": "matches/differs compare one valid connected surface per annotated part; otherwise unresolved",
        "limitations": "Processed meshes only; no raw-particle topology verification or prediction mesh verification. Multiple surface shells do not directly imply multiple solid components. No mesh repair is performed.",
        "frames": frame_results,
    }
    with (args.output / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
    print(f"SURFACE_TOPOLOGY_AUDIT_COMPLETE: {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
