"""Inspect goal scores using saved AE/Dyn artifacts, with NumPy on CPU.

Compare observed-state and predicted latents with the encoded final goal.
Cosine aggregation variants are diagnostic choices, not a verified original
planner. Geometry is compared on the saved grid with the union of both tool
masks excluded. No checkpoint loading, inference, action search, or execution
is performed. Existing artifacts are read only.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCORE_KEYS = ("token_mean_cosine", "flattened_cosine", "last_token_cosine")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cosine_scores(latent, goal):
    """Preserve token order; reject undefined cosine rather than hiding zeros."""
    latent, goal = np.asarray(latent, dtype=np.float64), np.asarray(goal, dtype=np.float64)
    if latent.ndim != 2 or latent.shape != goal.shape or min(latent.shape) < 1:
        raise ValueError("Latents must have the same nonempty token-by-feature shape")
    if not np.isfinite(latent).all() or not np.isfinite(goal).all():
        raise ValueError("Nonfinite latent")
    left, right = np.linalg.norm(latent, axis=1), np.linalg.norm(goal, axis=1)
    if (left == 0).any() or (right == 0).any():
        raise ValueError("Zero-norm latent token; cosine is undefined")
    tokens = np.clip(np.sum((latent / left[:, None]) * (goal / right[:, None]), axis=1), -1, 1)
    flattened = float(np.clip(np.sum(latent * goal) / (np.linalg.norm(latent) * np.linalg.norm(goal)), -1, 1))
    return dict(zip(SCORE_KEYS, (float(tokens.mean()), flattened, float(tokens[-1]))))


def goal_overlap(classes, goal_classes, ignored, goal_ignored):
    """Label-independent occupancy IoU; exclude both states' masked voxels."""
    if not (classes.shape == goal_classes.shape == ignored.shape == goal_ignored.shape):
        raise ValueError("Grid/mask shape mismatch")
    valid = ~(ignored | goal_ignored)
    target = (goal_classes > 0) & valid
    if not target.any():
        raise ValueError("No goal occupancy remains after excluding both tool masks")
    estimate = (classes > 0) & valid
    intersection, union = np.count_nonzero(estimate & target), np.count_nonzero(estimate | target)
    return {
        "goal_occupancy_iou_pct": float(100 * intersection / union),
        "comparison_voxels": int(valid.sum()),
        "goal_occupied_voxels": int(target.sum()),
    }


def spearman(x, y):
    """Average tied ranks; a constant array has no defined correlation."""
    def ranks(values):
        values = np.asarray(values)
        _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
        starts = np.cumsum(counts) - counts
        return (starts + (counts - 1) / 2)[inverse]

    if len(x) < 3:
        return None
    a, b = ranks(x), ranks(y)
    a, b = a - a.mean(), b - b.mean()
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return None if denominator == 0 else float(np.clip(np.dot(a, b) / denominator, -1, 1))


def load_npz(path, keys, hashes):
    hashes[str(path.resolve())] = sha256(path)
    with np.load(path, allow_pickle=False) as stream:
        return {key: stream[key] for key in keys}


def ae_fingerprints(root):
    return {
        "dyn_metadata_sha256": sha256(root / "metadata.json"),
        "observations_sha256": sha256(root / "observations.npz"),
        "dyn_metrics_sha256": sha256(root / "metrics.csv"),
        "model_config_sha256": sha256(root / "model_config.yaml"),
    }


def validate_ae_origin(path, root, fingerprints):
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("kind") != "same_observation_ae_reconstruction_diagnostic":
        raise ValueError(f"Not a completed AE diagnostic: {path}")
    if Path(metadata.get("dyn_artifacts", "")).resolve() != root.resolve():
        raise ValueError(f"AE artifacts refer to a different Dyn run: {path}")
    if any(metadata.get(key) != value for key, value in fingerprints.items()):
        raise ValueError(f"AE/Dyn provenance hashes do not match: {path}")
    return metadata


def discover_ae(root, fingerprints, records_root):
    candidates = set(records_root.glob("level3_ae_check_*/artifacts"))
    candidates.update(root.parent.parent.glob("ae_check_*/artifacts"))
    matches = []
    for path in sorted(candidates):
        try:
            validate_ae_origin(path, root, fingerprints)
        except (OSError, ValueError, TypeError):
            continue
        if (path / "summary.json").is_file():
            matches.append(path.resolve())
    if len(matches) != 1:
        details = "\n".join(str(path) for path in matches) or "none found"
        raise ValueError(f"Expected one matching completed AE run, found {len(matches)}. "
                         f"Pass --ae-artifacts explicitly. Matching paths:\n{details}")
    return matches[0]


def validate_classes(values, count, num_parts):
    if (values.shape != (count,) or not np.isfinite(values).all()
            or ((values < 0) | (values >= num_parts) | (values != np.rint(values))).any()):
        raise ValueError("Invalid cached class labels")


def validate_genus(values, num_parts):
    if (values.shape != (num_parts,) or not np.issubdtype(values.dtype, np.integer)
            or (values < -1).any()):
        raise ValueError("Invalid cached genus vector")


def build_report(root, ae_root, metadata, hashes):
    cache = load_npz(root / "observations.npz", (
        "frame_indices", "simulation_times", "z_reference", "genus_annotation",
    ), hashes)
    frames = cache["frame_indices"]
    if (frames.ndim != 1 or not len(frames) or not np.issubdtype(frames.dtype, np.integer)
            or frames[0] != 0 or not (np.diff(frames) > 0).all()
            or frames.tolist() != metadata["observed_frames"]
            or frames[-1] >= metadata["num_source_frames"]):
        raise ValueError("Invalid or misaligned observed frame schedule")
    times, references, genera = (cache[key] for key in ("simulation_times", "z_reference", "genus_annotation"))
    if (times.shape != frames.shape or not np.isfinite(times).all() or not (np.diff(times) > 0).all()
            or references.ndim != 3 or references.shape[0] != len(frames)
            or genera.ndim != 2 or genera.shape[0] != len(frames)):
        raise ValueError("Invalid cached time/latent/genus shapes")
    goal_frame = int(metadata["goal_frame"])
    if goal_frame != int(frames[-1]):
        raise ValueError("Final goal frame is missing from the observation cache")
    rollout_frames = metadata["rollout_frames"]
    if (len(rollout_frames) < 2 or rollout_frames[0] != 0
            or not set(rollout_frames).issubset(frames.tolist())
            or not np.all(np.diff(rollout_frames) == metadata["next_frame_offset"])
            or rollout_frames[-1] != metadata["final_evaluated_frame"]):
        raise ValueError("Invalid recorded prediction schedule")
    num_parts = genera.shape[1]
    goal = references[-1]
    packets, query = {}, None
    for index, frame in enumerate(frames.tolist()):
        packet = load_npz(ae_root / f"ae_{frame:03d}.npz", (
            "query", "predicted_class", "ground_truth_class", "ignored_mask",
            "z_reference", "ground_truth_genus", "predicted_genus", "frame",
        ), hashes)
        if query is None:
            query = packet["query"]
            if query.ndim != 2 or query.shape[1] != 3 or not len(query) or not np.isfinite(query).all():
                raise ValueError("Invalid saved query grid")
        if not np.array_equal(query, packet["query"]):
            raise ValueError(f"AE query grids differ at frame {frame}")
        if (packet["frame"].shape != () or int(packet["frame"]) != frame
                or not np.array_equal(packet["z_reference"], references[index])
                or not np.array_equal(packet["ground_truth_genus"], genera[index])):
            raise ValueError(f"AE observation/latent/annotation mismatch at frame {frame}")
        mask = packet["ignored_mask"]
        if mask.shape != (len(query),) or mask.dtype != np.bool_:
            raise ValueError(f"Invalid AE tool mask at frame {frame}")
        for key in ("predicted_class", "ground_truth_class"):
            validate_classes(packet[key], len(query), num_parts)
            if packet[key][mask].any():
                raise ValueError(f"Unmasked cached classes at frame {frame}")
        validate_genus(packet["ground_truth_genus"], num_parts)
        validate_genus(packet["predicted_genus"], num_parts)
        cosine_scores(references[index], goal)
        packets[frame] = packet
    goal_packet = packets[goal_frame]
    rows = []

    def add_row(mode, frame, index, latent, classes, genus):
        rows.append({
            "mode": mode, "frame": frame, "simulation_time": float(times[index]),
            **cosine_scores(latent, goal),
            **goal_overlap(classes, goal_packet["ground_truth_class"],
                           packets[frame]["ignored_mask"], goal_packet["ignored_mask"]),
            "topology_head_component_count": "" if mode == "reference" else int((genus >= 0).sum()),
            "genus_values_sorted": json.dumps(sorted(genus[genus >= 0].tolist())),
        })

    for index, frame in enumerate(frames.tolist()):
        packet = packets[frame]
        add_row("reference", frame, index, references[index], packet["ground_truth_class"], genera[index])
        add_row("ae_reconstruction", frame, index, references[index], packet["predicted_class"], packet["predicted_genus"])
    for source, frame in zip(rollout_frames[:-1], rollout_frames[1:]):
        index = frames.tolist().index(frame)
        for mode in ("single_step", "rollout"):
            prediction = load_npz(root / f"{mode}_{frame:03d}.npz", (
                "query", "predicted_class", "ground_truth_class", "ground_truth_genus",
                "predicted_genus", "z_predicted", "source_frame", "target_frame",
            ), hashes)
            if (prediction["source_frame"].shape != () or prediction["target_frame"].shape != ()
                    or int(prediction["source_frame"]) != source or int(prediction["target_frame"]) != frame
                    or not np.array_equal(prediction["query"], query)
                    or not np.array_equal(prediction["ground_truth_class"], packets[frame]["ground_truth_class"])
                    or not np.array_equal(prediction["ground_truth_genus"], genera[index])):
                raise ValueError(f"AE/Dyn frame, grid, or truth mismatch: {mode}/{frame}")
            validate_classes(prediction["predicted_class"], len(query), num_parts)
            validate_genus(prediction["predicted_genus"], num_parts)
            if prediction["predicted_class"][packets[frame]["ignored_mask"]].any():
                raise ValueError(f"Unmasked Dyn classes: {mode}/{frame}")
            add_row(mode, frame, index, prediction["z_predicted"], prediction["predicted_class"], prediction["predicted_genus"])

    mode_diagnostics = {}
    for mode in ("reference", "ae_reconstruction", "single_step", "rollout"):
        subset = [row for row in rows if row["mode"] == mode and row["frame"] != goal_frame]
        geometry = [row["goal_occupancy_iou_pct"] for row in subset]
        mode_diagnostics[mode] = {
            "num_frames_excluding_goal": len(subset),
            "best_geometry_frame": max(subset, key=lambda row: row["goal_occupancy_iou_pct"])["frame"] if subset else None,
            "score_variants": {
                key: {
                    "highest_score_frame": max(subset, key=lambda row: row[key])["frame"] if subset else None,
                    "spearman_with_goal_occupancy_iou": spearman([row[key] for row in subset], geometry),
                } for key in SCORE_KEYS
            },
        }
    summary = {
        "goal_frame": goal_frame, "final_predicted_frame": int(rollout_frames[-1]),
        "prediction_endpoint_gap_seconds": float(times[-1] - times[frames.tolist().index(rollout_frames[-1])]),
        "num_rows": len(rows),
        "goal_reference": next(row for row in rows if row["mode"] == "reference" and row["frame"] == goal_frame),
        "goal_ae_reconstruction": next(row for row in rows if row["mode"] == "ae_reconstruction" and row["frame"] == goal_frame),
        "initial_reference": rows[0],
        "last_predictions": [row for row in rows if row["mode"] in ("single_step", "rollout") and row["frame"] == rollout_frames[-1]],
        "mode_diagnostics": mode_diagnostics,
        "interpretation": "Cosine is a score, not a success probability. Correlations describe one recorded trajectory, not candidate-action ranking quality.",
    }
    return rows, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dyn-artifacts", type=Path, required=True)
    parser.add_argument("--ae-artifacts", type=Path, help="If omitted, discover one matching completed AE run")
    parser.add_argument("--output", type=Path, required=True, help="A directory that does not already exist")
    args = parser.parse_args(argv)
    started = time.monotonic()
    root = args.dyn_artifacts.resolve()
    if args.output.exists():
        raise FileExistsError(f"Choose a new output directory: {args.output}")
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("kind") != "recorded_action_interface_diagnostic":
        raise ValueError("Input must be the artifacts from check_level3_dyn.py")
    fingerprints = ae_fingerprints(root)
    ae_root = args.ae_artifacts.resolve() if args.ae_artifacts else discover_ae(root, fingerprints, PROJECT_ROOT / "records")
    validate_ae_origin(ae_root, root, fingerprints)
    hashes = {str((root / filename).resolve()): fingerprints[key] for key, filename in (
        ("dyn_metadata_sha256", "metadata.json"), ("observations_sha256", "observations.npz"),
        ("dyn_metrics_sha256", "metrics.csv"), ("model_config_sha256", "model_config.yaml"),
    )}
    hashes[str(ae_root / "metadata.json")] = sha256(ae_root / "metadata.json")
    rows, summary = build_report(root, ae_root, metadata, hashes)
    summary["elapsed_seconds"] = time.monotonic() - started
    output_metadata = {
        "kind": "cached_goal_score_diagnostic", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dyn_artifacts": str(root), "ae_artifacts": str(ae_root), "input_sha256": hashes,
        "checkpoint_sha256_as_recorded": metadata["checkpoint_sha256"], "script_sha256": sha256(__file__),
        "score_definitions": {
            "token_mean_cosine": "Mean of cosine over corresponding tokens (including the appended token)",
            "flattened_cosine": "Cosine after flattening all tokens and features into one vector",
            "last_token_cosine": "Cosine of the final latent token only; not the mean of the final token array",
        },
        "geometry": "Occupancy IoU against goal ground truth on the saved grid; exclude union of source-frame and goal ignored masks; ignore component labels; values in percent",
        "scope": "Reference and decoded AE/Dyn states from one recorded trajectory; no new inference, candidate actions, CEM, execution, or planning success measurement",
        "limitations": "Goal frame may be later than last prediction. Masks differ across pairs. Cosine aggregation is not verified against an original planner. Genus lists are unpaired topology-head outputs (reference uses graph annotations).",
        "environment": {"python": sys.version, "numpy": np.__version__},
    }
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / "goal_scores.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for filename, value in (("summary.json", summary), ("metadata.json", output_metadata)):
        (args.output / filename).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(f"Using AE artifacts: {ae_root}", flush=True)
    print(json.dumps(summary, indent=2, allow_nan=False), flush=True)
    print(f"GOAL_SCORE_DIAGNOSTIC_COMPLETE: {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
