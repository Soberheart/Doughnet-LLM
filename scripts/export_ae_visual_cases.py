"""Export fixed val_test AE examples for visual inspection.

Run from the DoughNet project root on an allocated GPU node:
    python scripts/export_ae_visual_cases.py \
      settings.test_path=records/training_ae_20260928_134123/checkpoints/ae_best.pth

Outputs are saved below records/level1_ae_visual_cases/predictions_topology_verified.
The input cloud is the voxelized observation actually passed to the AE. The
prediction PLY contains query-grid points whose predicted class is a component;
the compressed NPZ retains every query point and its class probabilities.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

# Running a file under scripts/ puts that directory, not the project root, on
# sys.path. Add the repository root before importing DoughNet modules.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import hydra
import numpy as np
import torch
from torch.utils.data import default_collate
from omegaconf import OmegaConf

from net.dataset.dataset import DoughDataset
from net.prediction import PredictionWorkspace
from net.pipeline.builder import get_model


# scene 11/51 is the selected near-contact case. The dataset labels scenes
# 15/55 and 36/53 as two components as well, despite their fused appearance.
# scene 0/52 is a verified one-component merge; scene 99/0 is the hole case.
CASES = ((11, 51), (15, 55), (36, 53), (0, 52), (99, 0))
COLORS = np.asarray([
    [120, 120, 120],  # class 0: outside / outlier
    [32, 168, 75],
    [242, 142, 43],
    [59, 130, 246],
    [194, 65, 168],
], dtype=np.uint8)


def write_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    xyz = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
    rgb = np.asarray(rgb, dtype=np.uint8).reshape(-1, 3)
    with path.open("w", encoding="ascii") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(xyz)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("element face 0\nproperty list uchar int vertex_indices\nend_header\n")
        for point, color in zip(xyz, rgb):
            f.write(f"{point[0]:.7g} {point[1]:.7g} {point[2]:.7g} "
                    f"{int(color[0])} {int(color[1])} {int(color[2])}\n")


def scalar(value) -> float:
    if isinstance(value, torch.Tensor):
        value = value.detach().float().mean().cpu().item()
    return float(value)


@hydra.main(version_base=None, config_path="../net/config", config_name="ae")
def main(cfg):
    # This is a deterministic, single-GPU inspection run, not a training run.
    cfg.settings.test_only = True
    cfg.settings.ddp = False
    cfg.dataset.test.bs = 1
    cfg.log_wandb = False
    OmegaConf.resolve(cfg)

    if str(cfg.dataset.test.subset) != "val_test":
        raise ValueError(f"Expected dataset.test.subset=val_test, got {cfg.dataset.test.subset!r}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run this script on an allocated GPU node.")

    workspace = PredictionWorkspace(cfg)
    model = get_model(cfg, ddp=False)
    workspace.load(model, path=cfg.settings.test_path,
                   pop_condition=cfg.dataset.next_frames == 0)
    model.eval()

    dataset = DoughDataset(cfg.dataset.test)
    out_dir = Path("records/level1_ae_visual_cases/predictions_topology_verified")
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = out_dir / "ae_visual_case_metrics.csv"

    rows = []
    with torch.no_grad():
        for scene, frame in CASES:
            try:
                dataset_idx = dataset.consecutive_items.index([scene, frame])
            except ValueError as exc:
                raise ValueError(
                    f"Could not map val_test scene={scene}, frame={frame} to Dataset index. "
                    "Check dataset.test subset and next_frames/keyframe settings."
                ) from exc

            # Synthetic val_test items contain meshes; Pipeline renders their
            # partial observations. custom_collate is reserved for real-data
            # batches and expects an obj_observed field.
            batch = default_collate([dataset[dataset_idx]])
            _, _, _, acc_dict, result = workspace.forward(model, batch, val_test=True)

            input_points = result["obj_observed"][0].detach().cpu().numpy()
            query = result["query"][0].detach().cpu().numpy()
            logits = result["obj_predicted_part"][0].detach().float().cpu()
            probabilities = torch.softmax(logits, dim=-1).numpy()
            predicted_class = probabilities.argmax(axis=-1).astype(np.int16)
            predicted_genus = result["predicted_genus"][0].detach().cpu().numpy().astype(np.int16)
            gt_genus = result["genus"][0].detach().cpu().numpy().astype(np.int16)
            gt_num_components = int(dataset.data["num_components"][scene, frame])

            prefix = f"val_test_scene{scene:03d}_frame{frame:03d}"
            input_labels = np.rint(input_points[:, 3]).astype(np.int64)
            input_colors = COLORS[np.clip(input_labels, 0, len(COLORS) - 1)]
            write_ply(out_dir / f"{prefix}_input_voxelized.ply", input_points[:, :3], input_colors)

            occupied = predicted_class > 0
            pred_colors = COLORS[np.clip(predicted_class[occupied], 0, len(COLORS) - 1)]
            write_ply(out_dir / f"{prefix}_prediction_occupancy.ply", query[occupied], pred_colors)
            np.savez_compressed(
                out_dir / f"{prefix}_prediction_grid.npz",
                query=query.astype(np.float32),
                probabilities=probabilities.astype(np.float16),
                predicted_class=predicted_class,
            )

            active_slots = sorted(int(x) for x in np.unique(predicted_class[occupied]))
            row = {
                "subset": "val_test", "scene": scene, "frame": frame,
                "dataset_index": dataset_idx,
                "checkpoint": str(cfg.settings.test_path),
                "gt_num_components": gt_num_components,
                "gt_genus_slots": ";".join(map(str, gt_genus.tolist())),
                "input_point_count": len(input_points),
                "query_count": len(query),
                "predicted_occupied_query_count": int(occupied.sum()),
                "predicted_component_slots_present": ";".join(map(str, active_slots)),
                "predicted_genus_slots": ";".join(map(str, predicted_genus.tolist())),
            }
            for key, value in acc_dict.items():
                row[key] = scalar(value)
            rows.append(row)
            print(f"Exported {prefix}: input={len(input_points)}, "
                  f"occupied queries={int(occupied.sum())}, active slots={active_slots}, "
                  f"predicted genus={predicted_genus.tolist()}, "
                  f"GT components={gt_num_components}, GT genus={gt_genus.tolist()}")

    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with metadata_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved per-case metadata and metrics: {metadata_path}")
    print(f"All case outputs: {out_dir}")


if __name__ == "__main__":
    main()
