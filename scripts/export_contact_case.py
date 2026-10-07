"""Export a clear, component-separated view of one val_test mesh case.

Run on the server from the DoughNet project root, for example:
    python scripts/export_contact_case.py --scene 11 --frame 51

The script exports opaque component meshes, PLY files, an interactive HTML
with front/side/top views, and a small metadata file.  It does not load or
modify any model checkpoint.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401: register HDF5 compression filters before reading
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.spatial import cKDTree


COLORS = ["#20a84b", "#f28e2b", "#3b82f6", "#c241a8", "#6b7280"]


def component_mesh(vertices, faces, vertex_labels, label):
    """Return a component mesh with compact vertex indices."""
    keep_v = vertex_labels == label
    keep_f = np.all(keep_v[faces], axis=1)
    selected_faces = faces[keep_f]
    selected_vertex_ids = np.flatnonzero(keep_v)
    remap = -np.ones(len(vertices), dtype=np.int64)
    remap[selected_vertex_ids] = np.arange(len(selected_vertex_ids))
    return vertices[selected_vertex_ids], remap[selected_faces]


def write_ply(path, vertices, faces):
    with path.open("w", encoding="ascii") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\nend_header\n")
        for xyz in vertices:
            f.write(f"{xyz[0]:.8g} {xyz[1]:.8g} {xyz[2]:.8g}\n")
        for tri in faces:
            f.write(f"3 {tri[0]} {tri[1]} {tri[2]}\n")


def add_mesh(fig, row, col, vertices, faces, color, name):
    if len(vertices) == 0 or len(faces) == 0:
        return
    fig.add_trace(
        go.Mesh3d(
            x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            color=color, opacity=1.0, flatshading=False,
            hoverinfo="skip", name=name, showlegend=True,
        ),
        row=row, col=col,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/dataset.h5")
    parser.add_argument("--scene", type=int, default=11)
    parser.add_argument("--frame", type=int, default=51)
    parser.add_argument(
        "--output", default="records/level1_ae_visual_cases/contact_candidates"
    )
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    prefix = f"val_test_scene{args.scene:03d}_frame{args.frame:03d}"

    with h5py.File(args.dataset, "r") as h5:
        group = h5["val_test"]
        vertices = np.asarray(group["obj_verts"][args.scene, args.frame])
        faces = np.asarray(group["obj_faces"][args.scene, args.frame])
        labels = np.asarray(group["obj_vert_labels"][args.scene, args.frame]).reshape(-1)
        num_components = int(group["num_components"][args.scene, args.frame])
        genus = np.asarray(group["genus"][args.scene, args.frame]).reshape(-1)

    labels_present = [int(x) for x in np.unique(labels) if int(x) > 0]
    components = []
    for index, label in enumerate(labels_present):
        v, f = component_mesh(vertices, faces, labels, label)
        components.append((label, v, f))
        write_ply(out / f"{prefix}_component{index + 1}_label{label}.ply", v, f)

    # Compute a simple nearest-vertex distance for the first two components.
    min_distance = float("nan")
    if len(components) >= 2:
        a, b = components[0][1], components[1][1]
        min_distance = float(cKDTree(a).query(b, k=1)[0].min())

    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{"type": "scene"}, {"type": "scene"}, {"type": "scene"}]],
        subplot_titles=("Front / XY", "Side / XZ", "Top / YZ"),
    )
    for col in range(1, 4):
        for index, (_, v, f) in enumerate(components):
            add_mesh(fig, 1, col, v, f, COLORS[index % len(COLORS)],
                     f"component {index + 1}")
        scene_name = "scene" if col == 1 else f"scene{col}"
        fig.layout[scene_name].aspectmode = "data"
        fig.layout[scene_name].xaxis.showbackground = False
        fig.layout[scene_name].yaxis.showbackground = False
        fig.layout[scene_name].zaxis.showbackground = False

    title = (
        f"{prefix} | num_components={num_components} | "
        f"labels={labels_present} | min vertex distance={min_distance:.6f}"
    )
    fig.update_layout(title=title, width=1800, height=700, margin=dict(l=0, r=0, t=70, b=0))
    fig.write_html(out / f"{prefix}_opaque_3views.html", include_plotlyjs=True)

    with (out / f"{prefix}_metadata.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["subset", "scene", "frame", "num_components", "labels", "genus", "min_vertex_distance"])
        writer.writerow(["val_test", args.scene, args.frame, num_components,
                         ";".join(map(str, labels_present)), ";".join(map(str, genus.tolist())),
                         f"{min_distance:.9f}"])

    print(f"Saved: {out / f'{prefix}_opaque_3views.html'}")
    print(f"Saved {len(components)} component PLY files and metadata CSV in {out}")
    print(f"num_components={num_components}, labels={labels_present}, min_distance={min_distance:.9f}")


if __name__ == "__main__":
    main()
