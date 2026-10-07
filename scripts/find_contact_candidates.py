#!/usr/bin/env python3
"""Rank two-component DoughNet frames by inter-component surface distance."""

import argparse
import csv
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401 - registers HDF5 compression filters
import numpy as np
from scipy.spatial import cKDTree


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/dataset.h5"))
    parser.add_argument("--output", type=Path, default=Path("records/level1_ae_visual_cases/contact_candidates.csv"))
    parser.add_argument("--max-candidates", type=int, default=20)
    args = parser.parse_args()

    results = []
    with h5py.File(args.dataset, "r") as file:
        group = file["val_test"]
        counts = group["num_components"][:]
        for scene in range(counts.shape[0]):
            best = None
            for frame in np.flatnonzero(counts[scene] == 2):
                labels = np.asarray(group["obj_vert_labels"][scene, frame]).reshape(-1)
                component_ids = np.unique(labels[labels >= 0])
                if len(component_ids) != 2:
                    continue
                vertices = np.asarray(group["obj_verts"][scene, frame])
                first = vertices[labels == component_ids[0]]
                second = vertices[labels == component_ids[1]]
                if not len(first) or not len(second):
                    continue
                minimum_distance = float(cKDTree(first).query(second, k=1, workers=1)[0].min())
                if best is None or minimum_distance < best[2]:
                    best = (scene, int(frame), minimum_distance, int(component_ids[0]), int(component_ids[1]))
            if best is not None:
                results.append(best)

    results.sort(key=lambda item: item[2])
    selected = results[: args.max_candidates]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="ascii") as stream:
        writer = csv.writer(stream)
        writer.writerow(("scene", "frame", "minimum_vertex_distance", "component_label_1", "component_label_2"))
        writer.writerows(selected)

    for scene, frame, distance, first, second in selected:
        print(
            f"scene={scene} frame={frame} distance={distance:.6f} "
            f"labels={first},{second}"
        )
    print(f"Saved {len(selected)} candidates to {args.output}")


if __name__ == "__main__":
    main()
