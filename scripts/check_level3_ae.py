"""Compare direct AE reconstruction with an existing Level 3 Dyn diagnostic.

Run on an allocated CUDA node after check_level3_dyn.py. Reuse its saved
observations, reference latents, query grid, source meshes, and combined Dyn
checkpoint. The condition module is not called. No simulation or rendering is
repeated, and the existing diagnostic artifacts are left intact.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.check_level3_dyn import (
    FIELDS, canonical_model_state, sha256, validate_arrays, write_cloud,
)

METRICS = ("viou", "ciou", "accc", "accg")


def validate_observations(cache, count, n_points, n_tokens, latent_dim, num_parts, num_genus):
    """Reject misaligned or corrupt cached observations before using their latents."""
    frames = cache["frame_indices"]
    if frames.ndim != 1 or len(frames) == 0 or not np.issubdtype(frames.dtype, np.integer):
        raise ValueError("Invalid cached frame indices")
    if frames[0] < 0 or frames[-1] >= count or not (np.diff(frames) > 0).all():
        raise ValueError("Cached frame indices must be unique, ordered, and in range")
    expected = {
        "observed": (len(frames), n_points, 4),
        "z_reference": (len(frames), n_tokens, latent_dim),
        "genus_annotation": (len(frames), num_parts),
        "simulation_times": (len(frames),),
    }
    for key, shape in expected.items():
        if cache[key].shape != shape or not np.isfinite(cache[key]).all():
            raise ValueError(f"Invalid shape or nonfinite cached values: {key}")
    labels = cache["observed"][..., 3]
    if ((labels < 0) | (labels >= num_parts) | (labels != np.rint(labels))).any():
        raise ValueError("Invalid cached observation labels")
    if not (cache["observed"][..., :3] != 0).any(axis=(1, 2)).all():
        raise ValueError("Empty cached object observation")
    genus = cache["genus_annotation"]
    if not np.issubdtype(genus.dtype, np.integer) or genus.min() < -1 or genus.max() >= num_genus:
        raise ValueError("Invalid cached genus annotations")
    return frames.tolist()


def require_hash(path, expected, label):
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"{label} changed since the Dyn diagnostic: {path}")
    return actual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dyn-artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="A directory that does not already exist")
    parser.add_argument("--scene-row", type=int, default=0)
    parser.add_argument("--cuda-id", type=int, default=0)
    args = parser.parse_args()
    started_timer = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat()
    root = args.dyn_artifacts.resolve()
    if args.output.exists():
        raise FileExistsError(f"Choose a new output directory: {args.output}")
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("kind") != "recorded_action_interface_diagnostic":
        raise ValueError("Input must be the artifacts from check_level3_dyn.py")
    # Paths in metadata refer to the server on which the diagnostic was run.
    checkpoint_path = Path(metadata["checkpoint"])
    processed_path = Path(metadata["processed"])
    require_hash(checkpoint_path, metadata["checkpoint_sha256"], "Checkpoint")
    require_hash(processed_path, metadata["processed_sha256"], "Processed scene")
    checked_sources = {}
    for path in ("net/model/model.py", "net/pipeline/pipeline.py", "net/pipeline/evaluater.py"):
        checked_sources[path] = require_hash(
            PROJECT_ROOT / path, metadata["source_file_sha256"][path], "Evaluation source",
        )

    import hdf5plugin  # Register the source HDF5 compression filters.
    import h5py
    import torch
    from omegaconf import OmegaConf
    from net.model.model import Predictor
    from net.pipeline.evaluater import Evaluater
    from net.pipeline.pipeline import get_label
    from net.pipeline.random import manual_seed

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; run on an allocated GPU compute node")
    torch.cuda.set_device(args.cuda_id)
    device = torch.device(f"cuda:{args.cuda_id}")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    manual_seed(int(metadata["seed"]))
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)
    cfg = OmegaConf.load(root / "model_config.yaml")
    OmegaConf.resolve(cfg)
    cfg.cuda_id = args.cuda_id
    if cfg.dimensions.input_dim != 4 or cfg.dimensions.num_parts != 5 or cfg.loss.w_top <= 0:
        raise ValueError("Unsupported diagnostic configuration")
    with h5py.File(processed_path, "r") as stream:
        if not 0 <= args.scene_row < stream["scene"].shape[0]:
            raise ValueError("Invalid processed scene row")
        data = {key: stream[key][args.scene_row] for key in FIELDS}
    count = validate_arrays(data, cfg.dimensions.num_parts, cfg.dataset.n_points)
    with np.load(root / "observations.npz", allow_pickle=False) as stream:
        cache = {key: stream[key] for key in (
            "frame_indices", "observed", "z_reference", "genus_annotation", "simulation_times",
        )}
    n_tokens = cfg.dimensions.num_query_in + int(cfg.model.high_level_token in ("mean", "max"))
    frames = validate_observations(
        cache, count, cfg.dataset.n_points, n_tokens, cfg.dimensions.latent_dim,
        cfg.dimensions.num_parts, cfg.dimensions.num_genus,
    )
    if frames != metadata["observed_frames"]:
        raise ValueError("Cached frames do not match the Dyn metadata")
    if not np.array_equal(cache["genus_annotation"], data["genus"][frames]):
        raise ValueError("Cached annotations do not match the processed scene row")
    with (root / "metrics.csv").open(newline="", encoding="utf-8") as stream:
        dyn_rows = {}
        for row in csv.DictReader(stream):
            key = (row["mode"], int(row["target_frame"]))
            if key in dyn_rows or row["mode"] not in ("single_step", "rollout"):
                raise ValueError("Invalid or duplicate Dyn metric rows")
            dyn_rows[key] = {metric: float(row[metric]) for metric in METRICS}
    targets = metadata["rollout_frames"][1:]
    if set(dyn_rows) != {(mode, frame) for mode in ("single_step", "rollout") for frame in targets}:
        raise ValueError("Dyn metrics are incomplete or refer to different target frames")
    if not all(np.isfinite(value) for row in dyn_rows.values() for value in row.values()):
        raise ValueError("Nonfinite Dyn metrics")
    with np.load(root / f"single_step_{targets[0]:03d}.npz", allow_pickle=False) as stream:
        query_numpy = stream["query"]
    if query_numpy.ndim != 2 or query_numpy.shape[1] != 3 or not np.isfinite(query_numpy).all():
        raise ValueError("Invalid cached query grid")
    query = torch.as_tensor(query_numpy, device=device)[None]
    model = Predictor(cfg).to(device).eval()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    result = model.load_state_dict(canonical_model_state(checkpoint["model_state_dict"]), strict=True)
    del checkpoint
    evaluator = Evaluater(cfg.loss).cuda(device=device)
    print(f"Checkpoint load: missing={result.missing_keys}; unexpected={result.unexpected_keys}", flush=True)
    print("Direct AE reconstruction uses the AE weights embedded in this Dyn checkpoint.", flush=True)

    args.output.mkdir(parents=True, exist_ok=False)
    rows, comparisons = [], []
    with torch.no_grad(), (args.output / "ae_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "frame", "simulation_time", *METRICS, "gt_genus", "matched_predicted_genus",
            "gt_component_count", "predicted_component_count", "encoding_max_abs_error",
        ])
        writer.writeheader()
        for index, frame in enumerate(frames):
            observation = torch.as_tensor(cache["observed"][index], device=device)[None]
            reference = torch.as_tensor(cache["z_reference"][index], device=device)[None]
            _, _, encoded = model.reconstruct(observation, None, decode=False)
            if not torch.isfinite(encoded).all() or not torch.allclose(encoded, reference, rtol=1e-4, atol=1e-5):
                raise RuntimeError(f"Saved latent does not match this model/observation at frame {frame}")
            encoding_error = float((encoded - reference).abs().max())
            # Decode the exact reference latent used by the prior Dyn comparison.
            logits, genus_logits = model.decode(query, reference)
            if not torch.isfinite(logits).all() or not torch.isfinite(genus_logits).all():
                raise RuntimeError(f"Nonfinite AE reconstruction at frame {frame}")
            mesh = {key: torch.as_tensor(data[key][frame], device=device)[None]
                    for key in ("obj_verts", "obj_faces", "obj_face_labels", "ee_verts", "ee_faces", "genus")}
            true_part = get_label(query, mesh["obj_verts"], mesh["obj_faces"], mesh["obj_face_labels"])
            in_tool = get_label(query, mesh["ee_verts"], mesh["ee_faces"],
                                torch.ones_like(mesh["ee_faces"][..., :1])).squeeze(-1).bool()
            ignored = in_tool[0] | (query[0, :, 2] < 0)
            truth = true_part[0, :, 0].clone()
            truth[ignored] = 0
            if not torch.any(truth > 0):
                raise RuntimeError(f"No ground-truth object in the query grid at frame {frame}")
            if frame in targets:
                for mode in ("single_step", "rollout"):
                    with np.load(root / f"{mode}_{frame:03d}.npz", allow_pickle=False) as saved:
                        if not (np.array_equal(query_numpy, saved["query"])
                                and np.array_equal(truth.cpu().numpy(), saved["ground_truth_class"])
                                and np.array_equal(data["genus"][frame], saved["ground_truth_genus"])):
                            raise ValueError(f"AE/Dyn query or ground-truth mismatch at frame {frame}")
            predicted_genus = genus_logits.argmax(dim=-1) - 1
            metrics, best_perm = evaluator.accuracy({
                "query": query, "obj_true_part": true_part, "obj_predicted_part": logits,
                "ee_true_part": in_tool, "genus": mesh["genus"], "predicted_genus": predicted_genus,
            }, "rec", "")
            gt_genus = data["genus"][frame]
            matched_genus = predicted_genus[0, best_perm].cpu().numpy()
            row = {"frame": frame, "simulation_time": float(cache["simulation_times"][index]),
                   **{metric: float(metrics[f"rec_{metric}"]) * 100 for metric in METRICS},
                   "gt_genus": json.dumps(gt_genus[gt_genus >= 0].tolist()),
                   "matched_predicted_genus": json.dumps(matched_genus[gt_genus >= 0].tolist()),
                   "gt_component_count": int((gt_genus >= 0).sum()),
                   "predicted_component_count": int((predicted_genus >= 0).sum()),
                   "encoding_max_abs_error": encoding_error}
            if not all(np.isfinite(row[metric]) for metric in METRICS):
                raise RuntimeError(f"Nonfinite AE metric at frame {frame}")
            rows.append(row)
            writer.writerow(row)
            stream.flush()
            if frame in targets:
                comparison = {"frame": frame, "simulation_time": row["simulation_time"]}
                for mode, values in (("ae", row), ("single_step", dyn_rows[("single_step", frame)]),
                                     ("rollout", dyn_rows[("rollout", frame)])):
                    comparison.update({f"{mode}_{metric}": values[metric] for metric in METRICS})
                comparisons.append(comparison)
            probabilities = torch.softmax(logits[0], dim=-1)
            probabilities[ignored] = torch.tensor([1., 0, 0, 0, 0], device=device)
            classes = probabilities.argmax(dim=-1).cpu().numpy().astype(np.int16)
            np.savez_compressed(
                args.output / f"ae_{frame:03d}.npz", query=query_numpy,
                probabilities=probabilities.cpu().numpy(), predicted_class=classes,
                ground_truth_class=truth.cpu().numpy(), ignored_mask=ignored.cpu().numpy(),
                predicted_genus=predicted_genus[0].cpu().numpy(), matched_predicted_genus=matched_genus,
                ground_truth_genus=gt_genus, z_reference=reference[0].cpu().numpy(), frame=frame,
            )
            write_cloud(args.output / f"ae_{frame:03d}.ply", query_numpy[classes > 0], classes[classes > 0])
            print(f"frame={frame:02d} AE: " + " ".join(f"{metric}={row[metric]:.2f}%" for metric in METRICS)
                  + f" GT={row['gt_genus']} matched={row['matched_predicted_genus']}", flush=True)

    with (args.output / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)
    summary = {"num_observed_frames": len(rows), "num_comparison_frames": len(targets),
               "elapsed_seconds": time.monotonic() - started_timer,
               "ae_all_observed_mean": {metric: float(np.mean([row[metric] for row in rows])) for metric in METRICS},
               "same_target_frames_mean": {},
               "encoding_max_abs_error": max(row["encoding_max_abs_error"] for row in rows)}
    for mode in ("ae", "single_step", "rollout"):
        summary["same_target_frames_mean"][mode] = {
            metric: float(np.mean([row[f"{mode}_{metric}"] for row in comparisons])) for metric in METRICS}
    for label, frame in (("last_dyn_target", targets[-1]), ("goal", metadata["goal_frame"])):
        summary[f"ae_{label}"] = next(row for row in rows if row["frame"] == frame)
    output_metadata = {
        "kind": "same_observation_ae_reconstruction_diagnostic",
        "started_at_utc": started_at,
        "dyn_artifacts": str(root), "dyn_metadata_sha256": sha256(root / "metadata.json"),
        "observations_sha256": sha256(root / "observations.npz"),
        "dyn_metrics_sha256": sha256(root / "metrics.csv"),
        "model_config_sha256": sha256(root / "model_config.yaml"),
        "checkpoint": str(checkpoint_path), "checkpoint_sha256": metadata["checkpoint_sha256"],
        "processed_sha256": metadata["processed_sha256"], "scene_row": args.scene_row,
        "script_sha256": sha256(__file__), "checked_source_sha256": checked_sources,
        "observation": "Saved observations; encoding checked against saved latents; exact saved latents decoded",
        "metrics": "Repository Evaluater; same saved query grid; Boolean tool mask; values in percent",
        "scope": "Direct AE within the same combined checkpoint; no condition module, action search, or simulation",
        "limitations": "One scene; simulator topology annotations; not a dataset benchmark or geometric genus verification",
        "environment": {"python": sys.version, "numpy": np.__version__, "torch": torch.__version__,
                        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(device)},
    }
    for filename, value in (("summary.json", summary), ("metadata.json", output_metadata)):
        with (args.output / filename).open("w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
    print(json.dumps(summary, indent=2, allow_nan=False), flush=True)
    print(f"AE_RECONSTRUCTION_DIAGNOSTIC_PASS: {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
