"""Render processed simulator frames and inspect a recorded-action Dyn rollout.

Run on an allocated CUDA node. This is a new-scene interface diagnostic, not
the official dataset benchmark or a CEM planning experiment. It reads existing
scene data and weights, and writes results only to a new output directory.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle
import sys
import time

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

FIELDS = (
    "scene", "frame", "obj_verts", "obj_faces", "obj_vert_labels",
    "obj_face_labels", "ee_verts", "ee_faces", "ee_observed", "genus",
)
COLORS = np.array(
    [[120, 120, 120], [32, 168, 75], [242, 142, 43],
     [59, 130, 246], [194, 65, 168]], dtype=np.uint8,
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_arrays(data, num_parts, num_points):
    """Validate a single scene after removing the HDF5 scene dimension."""
    missing = set(FIELDS) - set(data)
    if missing:
        raise ValueError(f"Missing processed fields: {sorted(missing)}")
    count = len(data["frame"])
    if count < 2:
        raise ValueError("At least two processed frames are required")
    for key in FIELDS:
        value = data[key]
        if value.ndim == 0 or value.shape[0] != count or not np.isfinite(value).all():
            raise ValueError(f"Invalid frame count or nonfinite values: {key}")
    for key in ("scene", "frame", "genus"):
        if not np.issubdtype(data[key].dtype, np.integer):
            raise ValueError(f"Noninteger annotations: {key}")
    if data["scene"].shape != (count,):
        raise ValueError("Invalid scene annotation shape")
    if not np.array_equal(data["frame"], np.arange(count)):
        raise ValueError("Processed frame indices must be consecutive from zero")
    if np.unique(data["scene"]).size != 1:
        raise ValueError("Scene IDs vary within a processed sequence")
    if data["genus"].shape != (count, num_parts):
        raise ValueError("Genus annotation shape does not match the model")
    if data["ee_observed"].shape != (count, num_points, 3):
        raise ValueError("Tool point cloud shape does not match the model")
    for prefix in ("obj", "ee"):
        vertices, faces = data[f"{prefix}_verts"], data[f"{prefix}_faces"]
        if vertices.ndim != 3 or vertices.shape[-1] != 3:
            raise ValueError(f"Invalid {prefix} vertex shape")
        if faces.ndim != 3 or faces.shape[-1] != 3:
            raise ValueError(f"Invalid {prefix} face shape")
        if vertices.shape[1] == 0 or faces.shape[1] == 0:
            raise ValueError(f"Empty {prefix} vertex or face array")
        if not np.issubdtype(faces.dtype, np.integer):
            raise ValueError(f"Noninteger {prefix} face indices")
        if faces.min() < 0 or faces.max() >= vertices.shape[1]:
            raise ValueError(f"Out-of-range {prefix} face indices")
        triangles = vertices[np.arange(count)[:, None, None], faces]
        areas = np.linalg.norm(np.cross(
            triangles[:, :, 1] - triangles[:, :, 0],
            triangles[:, :, 2] - triangles[:, :, 0],
        ), axis=-1)
        if not (areas > 0).any(axis=1).all():
            raise ValueError(f"Empty {prefix} mesh in a processed frame")
    for key, size in (("obj_vert_labels", data["obj_verts"].shape[1]),
                      ("obj_face_labels", data["obj_faces"].shape[1])):
        labels = data[key]
        if labels.shape != (count, size, 1):
            raise ValueError(f"Invalid label shape: {key}")
        if not np.issubdtype(labels.dtype, np.integer):
            raise ValueError(f"Noninteger labels: {key}")
        if labels.min() < 1 or labels.max() >= num_parts:
            raise ValueError(f"Labels outside the model's component range: {key}")
    return count


def frame_schedule(count, offset):
    if offset < 1 or count <= offset:
        raise ValueError("Sequence is too short for next_frame_offset")
    rollout = list(range(0, count, offset))
    observed = sorted(set(rollout + [count - 1]))
    return rollout, observed


def canonical_model_state(state):
    """Keep condition weights; remove only Pipeline/DDP name prefixes."""
    mapped = {}
    for key, value in state.items():
        name = key.removeprefix("module.").removeprefix("model.")
        if name in mapped:
            raise ValueError(f"Duplicate checkpoint key after prefix removal: {name}")
        mapped[name] = value
    if not any(name.startswith("condition.") for name in mapped):
        raise ValueError("Checkpoint has no condition weights; a trained Dyn is required")
    return mapped


def write_cloud(path, points, labels):
    labels = np.asarray(labels).reshape(-1)
    points = np.asarray(points).reshape(-1, 3)
    if len(points) != len(labels):
        raise ValueError("Point and label counts differ")
    if not np.isfinite(points).all() or not np.isfinite(labels).all():
        raise ValueError("Nonfinite exported point cloud")
    if ((labels < 0) | (labels >= len(COLORS)) | (labels != np.rint(labels))).any():
        raise ValueError("Invalid exported component labels")
    colors = COLORS[labels.astype(int)]
    with Path(path).open("w", encoding="ascii") as stream:
        stream.write("ply\nformat ascii 1.0\n")
        stream.write(f"element vertex {len(points)}\n")
        stream.write("property float x\nproperty float y\nproperty float z\n")
        stream.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        stream.write("end_header\n")
        for point, color in zip(points, colors):
            stream.write("%.7g %.7g %.7g %d %d %d\n" % (*point, *color))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--processed", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True,
                        help="The Dyn training run's .hydra/config.yaml")
    parser.add_argument("--output", type=Path, required=True,
                        help="A directory that does not already exist")
    parser.add_argument("--scene-row", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cuda-id", type=int, default=0)
    parser.add_argument("--expected-sha256")
    args = parser.parse_args()
    started_at = datetime.now(timezone.utc).isoformat()
    started_timer = time.monotonic()

    import hdf5plugin  # Register the original HDF5 compression filters.
    import h5py
    import torch
    from omegaconf import OmegaConf
    from net.model.model import Predictor
    from net.pipeline.evaluater import Evaluater
    from net.pipeline.pipeline import get_label
    from net.pipeline.random import manual_seed
    from net.pipeline.renderer import Renderer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; run on an allocated GPU compute node")
    torch.cuda.set_device(args.cuda_id)
    device = torch.device(f"cuda:{args.cuda_id}")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    manual_seed(args.seed)
    if args.output.exists():
        raise FileExistsError(f"Choose a new output directory: {args.output}")
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)
    cfg = OmegaConf.load(args.config)
    OmegaConf.resolve(cfg)
    cfg.settings.ddp = False
    cfg.settings.test_only = True
    cfg.log_wandb = False
    cfg.cuda_id = args.cuda_id
    if cfg.model.condition.ee_shape != "both" or cfg.dimensions.input_dim != 4:
        raise ValueError("This diagnostic requires the trained 'both' tool / xyz-label input")
    if cfg.dimensions.num_parts != len(COLORS):
        raise ValueError("This diagnostic expects the repository's five part classes")
    if cfg.loss.w_top <= 0:
        raise ValueError("Topology metrics require the trained topology head")

    checkpoint_hash = sha256(args.checkpoint)
    if args.expected_sha256 and checkpoint_hash != args.expected_sha256.lower():
        raise ValueError("Dyn checkpoint SHA256 does not match the expected checkpoint")
    print(f"checkpoint_sha256={checkpoint_hash}", flush=True)
    with h5py.File(args.processed, "r") as stream:
        if "scene" not in stream or not 0 <= args.scene_row < stream["scene"].shape[0]:
            raise ValueError("Invalid scene row or incorrect processed HDF5 format")
        data = {key: stream[key][args.scene_row] for key in FIELDS}
    count = validate_arrays(data, cfg.dimensions.num_parts, cfg.dataset.n_points)
    if data["genus"].min() < -1 or data["genus"].max() >= cfg.dimensions.num_genus:
        raise ValueError("Genus annotations outside the model's class range")
    scene_cfg = OmegaConf.load(args.scene_dir / "config.yaml")
    OmegaConf.resolve(scene_cfg)
    if int(scene_cfg.scene_id) != int(data["scene"][0]):
        raise ValueError("Scene config and processed scene ID do not match")
    with (args.scene_dir / "log.pkl").open("rb") as stream:
        logs = pickle.load(stream)
    if len(logs) < count:
        raise ValueError("Processed sequence contains padding; use unpadded source frames")
    # process.py keeps the last N source frames when the original run is longer.
    logs = logs[-count:]
    steps = np.array([int(frame["step"]) for frame in logs])
    if not (np.diff(steps) == int(scene_cfg.log.n_iter)).all():
        raise ValueError("Source simulation logging intervals are irregular")
    raw_genus = [sorted(frame["topology"]["genus"]) for frame in logs]
    saved_genus = [sorted(row[row >= 0].tolist()) for row in data["genus"]]
    if raw_genus != saved_genus:
        raise ValueError("Processed genus labels disagree with source log annotations")

    offset = int(cfg.dataset.next_frame_offset)
    rollout_frames, observed_frames = frame_schedule(count, offset)
    step_seconds = offset * int(scene_cfg.log.n_iter) * float(scene_cfg.sim.step_dt)
    print(f"rollout_frames={rollout_frames}; interval_seconds={step_seconds:g}", flush=True)
    print(f"goal_frame={count - 1}; final_evaluated_frame={rollout_frames[-1]}", flush=True)
    model = Predictor(cfg).to(device).eval()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    result = model.load_state_dict(canonical_model_state(checkpoint["model_state_dict"]), strict=True)
    print(f"Checkpoint load: missing={result.missing_keys}; unexpected={result.unexpected_keys}", flush=True)
    del checkpoint
    # Renderer mutates intrinsics when downsampling, so give it an independent copy.
    render_cfg = OmegaConf.create(OmegaConf.to_container(cfg.augmentation.render, resolve=True))
    renderer = Renderer(render_cfg, device=device).eval()
    evaluator = Evaluater(cfg.loss).cuda(device=device)

    args.output.mkdir(parents=True, exist_ok=False)
    OmegaConf.save(cfg, args.output / "model_config.yaml")
    metadata = {
        "kind": "recorded_action_interface_diagnostic", "seed": args.seed,
        "started_at_utc": started_at,
        "scene_dir": str(args.scene_dir.resolve()), "processed": str(args.processed.resolve()),
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": checkpoint_hash,
        "processed_sha256": sha256(args.processed), "config_sha256": sha256(args.config),
        "source_log_sha256": sha256(args.scene_dir / "log.pkl"),
        "scene_config_sha256": sha256(args.scene_dir / "config.yaml"),
        "script_sha256": sha256(__file__),
        "num_source_frames": count, "next_frame_offset": offset,
        "prediction_interval_seconds": step_seconds,
        "rollout_frames": rollout_frames, "observed_frames": observed_frames,
        "goal_frame": count - 1, "final_evaluated_frame": rollout_frames[-1],
        "unit_to_meter": float(cfg.augmentation.render.unit_to_meter),
        "observation": "Original Renderer, training-config noise/camera settings, identity scene transform; per-frame seed=seed+frame",
        "rollout": "Only the first object observation initializes rollout; recorded tool point clouds supply actions",
        "metrics": "Repository Evaluater; grid spacing 0.01; explicit Boolean tool mask; values in percent",
        "latent_cosine": "Diagnostic mean of per-token cosine; not a verified official planning objective",
        "topology_reference": "Simulator annotations carried into processed data; geometric mesh genus not validated here",
        "limitations": "One recorded scene; no CEM, no newly selected action execution, no planning success rate",
        "environment": {"python": sys.version, "numpy": np.__version__,
                        "torch": torch.__version__, "cuda": torch.version.cuda,
                        "gpu": torch.cuda.get_device_name(device)},
    }
    source_files = (
        "net/model/model.py", "net/pipeline/renderer.py", "net/pipeline/pipeline.py",
        "net/pipeline/evaluater.py", "sim/process.py",
        "sim/mpm/assets/meshes/processed/board-board.provenance.json",
    )
    metadata["source_file_sha256"] = {
        path: sha256(PROJECT_ROOT / path) for path in source_files
        if (PROJECT_ROOT / path).is_file()
    }
    with (args.output / "metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, ensure_ascii=False, allow_nan=False)

    def mesh_frame(frame):
        return {key: torch.as_tensor(data[key][frame], device=device)[None]
                for key in ("obj_verts", "obj_faces", "obj_vert_labels", "obj_face_labels",
                            "ee_verts", "ee_faces", "genus")}

    observations, references = {}, {}
    with torch.no_grad():
        for frame in observed_frames:
            manual_seed(args.seed + frame)
            observation = renderer(mesh_frame(frame), postfix="", num_samples=cfg.dataset.n_points)
            if tuple(observation.shape) != (1, cfg.dataset.n_points, cfg.dimensions.input_dim):
                raise RuntimeError(f"Unexpected observation shape at frame {frame}")
            if not torch.isfinite(observation).all() or not torch.any(observation[..., :3] != 0):
                raise RuntimeError(f"Empty or nonfinite rendered observation at frame {frame}")
            _, _, latent = model.reconstruct(observation, None, decode=False)
            extra_token = int(cfg.model.high_level_token in ("mean", "max"))
            expected_shape = (1, cfg.dimensions.num_query_in + extra_token, cfg.dimensions.latent_dim)
            if tuple(latent.shape) != expected_shape:
                raise RuntimeError(f"Unexpected encoded latent shape at frame {frame}: {latent.shape}")
            if not torch.isfinite(latent).all():
                raise RuntimeError(f"Nonfinite encoded latent at frame {frame}")
            observations[frame], references[frame] = observation, latent
            cloud = observation[0].cpu().numpy()
            write_cloud(args.output / f"observed_{frame:03d}.ply", cloud[:, :3], cloud[:, 3])
        latent_shape = list(references[0].shape)
        np.savez_compressed(
            args.output / "observations.npz",
            frame_indices=np.array(observed_frames), simulation_steps=steps[observed_frames],
            simulation_times=steps[observed_frames] * float(scene_cfg.sim.step_dt),
            observed=np.stack([observations[f][0].cpu().numpy() for f in observed_frames]),
            ee_observed=data["ee_observed"][observed_frames],
            z_reference=np.stack([references[f][0].cpu().numpy() for f in observed_frames]),
            genus_annotation=data["genus"][observed_frames],
        )
        print(f"OBSERVATION_PASS: {len(observed_frames)} frames; latent_shape={latent_shape}", flush=True)

        # Match Pipeline's existing validation query-grid construction.
        resolution = 0.01
        bounds = torch.tensor([[-0.30, 0.30], [-0.52, 0.52],
                               [-resolution / 2, 0.11 + resolution / 2]], device=device)
        grid_steps = ((bounds[:, 1] - bounds[:, 0]) / resolution).int() + 1
        coords = [torch.linspace(bounds[i, 0], bounds[i, 1], int(grid_steps[i]), device=device)
                  for i in range(3)]
        query = torch.stack(torch.meshgrid(*coords, indexing="ij"), dim=-1).reshape(1, -1, 3)
        query_numpy = query[0].cpu().numpy()
        z_rollout = references[0]
        rows = []
        csv_path = args.output / "metrics.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            columns = ["mode", "source_frame", "target_frame", "viou", "ciou", "accc", "accg", "latent_cosine"]
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            for source, target in zip(rollout_frames[:-1], rollout_frames[1:]):
                target_mesh = mesh_frame(target)
                true_part = get_label(query, target_mesh["obj_verts"], target_mesh["obj_faces"],
                                      target_mesh["obj_face_labels"])
                tool_part = get_label(query, target_mesh["ee_verts"], target_mesh["ee_faces"],
                                      torch.ones_like(target_mesh["ee_faces"][..., :1]))
                in_tool = tool_part.squeeze(-1).bool()
                ignored = in_tool[0] | (query[0, :, 2] < 0)
                masked_truth = true_part[0, :, 0].clone()
                masked_truth[ignored] = 0
                if not torch.any(masked_truth > 0):
                    raise RuntimeError(f"No ground-truth object in the query grid at frame {target}")
                tools = torch.as_tensor(data["ee_observed"][[source, target]], device=device)[:, None]
                tools = torch.cat([tools, torch.zeros_like(tools[..., :1])], dim=-1)
                for mode, latent in (("single_step", references[source]), ("rollout", z_rollout)):
                    logits, genus_logits, prediction = model.predict(tools, query, latent, decode=True)
                    if not all(torch.isfinite(value).all() for value in (logits, genus_logits, prediction)):
                        raise RuntimeError(f"Nonfinite {mode} prediction at frame {target}")
                    genus_pred = genus_logits.argmax(dim=-1) - 1
                    evaluation = {
                        "query": query, "obj_true_part_nxt": true_part,
                        "obj_predicted_part_nxt": logits, "ee_true_part_nxt": in_tool,
                        "genus_nxt": target_mesh["genus"], "predicted_genus_nxt": genus_pred,
                    }
                    metrics, best_perm = evaluator.accuracy(evaluation, "pre", "_nxt")
                    row = {"mode": mode, "source_frame": source, "target_frame": target}
                    row.update({key: float(metrics[f"pre_{key}_nxt"]) * 100
                                for key in ("viou", "ciou", "accc", "accg")})
                    row["latent_cosine"] = float(torch.nn.functional.cosine_similarity(
                        prediction, references[target], dim=-1).mean())
                    if not all(np.isfinite(row[key]) for key in ("viou", "ciou", "accc", "accg", "latent_cosine")):
                        raise RuntimeError(f"Nonfinite metric at frame {target}")
                    rows.append(row)
                    writer.writerow(row)
                    stream.flush()
                    probabilities = torch.softmax(logits[0], dim=-1)
                    probabilities[ignored] = torch.tensor([1., 0, 0, 0, 0], device=device)
                    predicted_class = probabilities.argmax(dim=-1).cpu().numpy().astype(np.int16)
                    np.savez_compressed(
                        args.output / f"{mode}_{target:03d}.npz",
                        query=query_numpy, probabilities=probabilities.cpu().numpy(),
                        predicted_class=predicted_class,
                        ground_truth_class=masked_truth.cpu().numpy(),
                        predicted_genus=genus_pred[0].cpu().numpy(),
                        matched_predicted_genus=genus_pred[0, best_perm].cpu().numpy(),
                        ground_truth_genus=data["genus"][target],
                        z_predicted=prediction[0].cpu().numpy(),
                        source_frame=source, target_frame=target,
                    )
                    occupied = predicted_class > 0
                    write_cloud(args.output / f"{mode}_{target:03d}.ply",
                                query_numpy[occupied], predicted_class[occupied])
                    if mode == "rollout":
                        z_rollout = prediction
                print(f"predicted interval {source}->{target}", flush=True)

    summary = {"latent_shape": latent_shape, "num_intervals": len(rollout_frames) - 1,
               "elapsed_seconds": time.monotonic() - started_timer}
    for mode in ("single_step", "rollout"):
        subset = [row for row in rows if row["mode"] == mode]
        summary[f"{mode}_mean"] = {key: float(np.mean([row[key] for row in subset]))
                                   for key in ("viou", "ciou", "accc", "accg", "latent_cosine")}
    summary["rollout_last"] = {key: value for key, value in rows[-1].items()
                               if key not in ("mode", "source_frame")}
    with (args.output / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
    for name in ("single_step_mean", "rollout_mean", "rollout_last"):
        values = summary[name]
        print(f"{name}: " + " ".join(f"{key}={values[key]:.2f}%" for key in ("viou", "ciou", "accc", "accg")), flush=True)
    print(f"FIXED_ACTION_INFERENCE_PASS: {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
