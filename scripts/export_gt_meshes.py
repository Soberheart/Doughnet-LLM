#!/usr/bin/env python3
"""Export fixed DoughNet ground-truth mesh cases for visual inspection."""

import argparse
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401 - registers the dataset's HDF5 compression filters
import numpy as np
import open3d as o3d
import plotly.graph_objects as go


CASES = ((0, 51), (0, 52), (99, 0))
COLORS = np.array(
    [
        [0.18, 0.52, 0.72],
        [0.85, 0.39, 0.23],
        [0.31, 0.66, 0.43],
        [0.66, 0.43, 0.68],
        [0.85, 0.70, 0.24],
    ],
    dtype=np.float64,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/dataset.h5"))
    parser.add_argument(
        "--output", type=Path, default=Path("records/level1_ae_visual_cases/ground_truth")
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.dataset, "r") as file:
        group = file["val_test"]
        for scene, frame in CASES:
            vertices = np.asarray(group["obj_verts"][scene, frame], dtype=np.float64)
            faces = np.asarray(group["obj_faces"][scene, frame], dtype=np.int32)
            labels = np.asarray(group["obj_vert_labels"][scene, frame]).reshape(-1)
            component_count = int(group["num_components"][scene, frame])
            genus = np.asarray(group["genus"][scene, frame]).tolist()

            valid = np.all((faces >= 0) & (faces < len(vertices)), axis=1)
            faces = faces[valid]
            nondegenerate = (
                (faces[:, 0] != faces[:, 1])
                & (faces[:, 1] != faces[:, 2])
                & (faces[:, 0] != faces[:, 2])
            )
            faces = faces[nondegenerate]
            if not len(faces):
                raise RuntimeError(f"No valid faces for scene={scene}, frame={frame}")

            colors = COLORS[np.mod(labels.astype(np.int64), len(COLORS))]
            mesh = o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(vertices),
                o3d.utility.Vector3iVector(faces),
            )
            mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
            mesh.compute_vertex_normals()

            stem = f"val_test_scene{scene:03d}_frame{frame:03d}_gt"
            ply_path = args.output / f"{stem}.ply"
            html_path = args.output / f"{stem}.html"
            if not o3d.io.write_triangle_mesh(str(ply_path), mesh):
                raise RuntimeError(f"Could not write {ply_path}")

            vertex_colors = [
                f"rgb({r},{g},{b})"
                for r, g, b in (colors * 255).astype(np.uint8)
            ]
            figure = go.Figure(
                go.Mesh3d(
                    x=vertices[:, 0],
                    y=vertices[:, 1],
                    z=vertices[:, 2],
                    i=faces[:, 0],
                    j=faces[:, 1],
                    k=faces[:, 2],
                    vertexcolor=vertex_colors,
                    flatshading=True,
                    lighting={"ambient": 0.65, "diffuse": 0.75},
                    hoverinfo="skip",
                )
            )
            figure.update_layout(
                title=(
                    f"Ground truth | val_test scene {scene}, frame {frame}"
                    f" | components {component_count} | genus {genus}"
                ),
                scene={"aspectmode": "data"},
                margin={"l": 0, "r": 0, "b": 0, "t": 60},
            )
            figure.write_html(str(html_path), include_plotlyjs=True)
            print(f"{stem}: {len(vertices)} vertices, {len(faces)} faces")
            print(f"  {ply_path}")
            print(f"  {html_path}")


if __name__ == "__main__":
    main()
