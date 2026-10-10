"""Compare an executed candidate scene with the recorded target frame.

This is a CPU-only comparison of artifacts produced by check_level3_ae.py. It
reports latent similarity and occupancy overlap, while keeping the candidate's
actual simulated geometry separate from its AE reconstruction.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_packet(root, frame):
    path = Path(root) / f"ae_{frame:03d}.npz"
    with np.load(path, allow_pickle=False) as packet:
        result = {key: packet[key] for key in packet.files}
    required = {"query", "predicted_class", "ground_truth_class", "ignored_mask",
                "z_reference", "ground_truth_genus", "predicted_genus", "frame"}
    if not required.issubset(result) or result["frame"].shape != () or int(result["frame"]) != frame:
        raise ValueError(f"Invalid AE packet: {path}")
    return result, path


def cosine(left, right):
    left, right = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 2 or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("Latent shapes or values differ")
    left_norm, right_norm = np.linalg.norm(left), np.linalg.norm(right)
    if left_norm == 0 or right_norm == 0:
        raise ValueError("Cosine is undefined for a zero latent")
    token_left = np.linalg.norm(left, axis=1)
    token_right = np.linalg.norm(right, axis=1)
    if (token_left == 0).any() or (token_right == 0).any():
        raise ValueError("Cosine is undefined for a zero latent token")
    return {
        "token_mean_cosine": float(np.mean(np.sum(left * right, axis=1) / (token_left * token_right))),
        "flattened_cosine": float(np.sum(left * right) / (left_norm * right_norm)),
    }


def occupancy_iou(estimate, target, estimate_mask, target_mask):
    if not (estimate.shape == target.shape == estimate_mask.shape == target_mask.shape):
        raise ValueError("Grid and mask shapes differ")
    valid = ~(estimate_mask.astype(bool) | target_mask.astype(bool))
    estimate, target = estimate > 0, target > 0
    union = (estimate | target) & valid
    if not union.any():
        raise ValueError("Target and candidate have no valid occupied voxels")
    return float(100 * np.count_nonzero(estimate & target & valid) / np.count_nonzero(union))


def compare(target_root, candidate_root, frame):
    target, target_path = load_packet(target_root, frame)
    candidate, candidate_path = load_packet(candidate_root, frame)
    if not np.array_equal(target["query"], candidate["query"]):
        raise ValueError("Target and candidate query grids differ")
    scores = cosine(candidate["z_reference"], target["z_reference"])
    result = {
        "frame": frame,
        **scores,
        "candidate_decode_vs_target_iou_pct": occupancy_iou(
            candidate["predicted_class"], target["ground_truth_class"],
            candidate["ignored_mask"], target["ignored_mask"],
        ),
        "candidate_actual_vs_target_iou_pct": occupancy_iou(
            candidate["ground_truth_class"], target["ground_truth_class"],
            candidate["ignored_mask"], target["ignored_mask"],
        ),
        "candidate_actual_genus": sorted(candidate["ground_truth_genus"][candidate["ground_truth_genus"] >= 0].tolist()),
        "target_genus": sorted(target["ground_truth_genus"][target["ground_truth_genus"] >= 0].tolist()),
        "candidate_predicted_genus": sorted(candidate["predicted_genus"][candidate["predicted_genus"] >= 0].tolist()),
        "target_packet_sha256": sha256(target_path),
        "candidate_packet_sha256": sha256(candidate_path),
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-ae-artifacts", type=Path, required=True)
    parser.add_argument("--candidate-ae-artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--frame", type=int, default=60,
        help="Primary comparison frame; use --frame 62 only for the auxiliary final-frame report",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")
    result = compare(args.target_ae_artifacts, args.candidate_ae_artifacts, args.frame)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"POSE_EXECUTION_COMPARISON_COMPLETE: {args.output}")


if __name__ == "__main__":
    main()
