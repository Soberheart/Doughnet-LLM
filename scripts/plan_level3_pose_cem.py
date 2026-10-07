"""Search grasp pose with a frozen tool and recorded closing schedule.

This is an initial Level 3 implementation, not the complete original planner.
Only planar translation and yaw are optimized. The tool, opening schedule,
and 12-step prediction horizon come from the recorded interface experiment.
Search uses only initial/goal object latents and candidate tool point clouds;
intermediate object states are used solely for the nominal interface check.
The selected actions are exported for subsequent independent simulation.
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
from scripts.check_level3_dyn import canonical_model_state, sha256, write_cloud


def transform_tools(tools, center, parameters, unit_to_meter):
    """Parameters are dx/dy in mm and yaw in degrees about the nominal grasp."""
    tools, center, parameters = (np.asarray(value, dtype=np.float64)
                                 for value in (tools, center, parameters))
    if (tools.ndim != 3 or tools.shape[-1] != 3 or center.shape != (3,)
            or parameters.ndim != 2 or parameters.shape[1] != 3
            or not all(np.isfinite(value).all() for value in (tools, center, parameters))
            or not np.isfinite(unit_to_meter) or unit_to_meter <= 0):
        raise ValueError("Invalid tool trajectory, pose parameters, or unit conversion")
    relative = tools - center
    angle = np.deg2rad(parameters[:, 2])[:, None, None]
    cosine, sine = np.cos(angle), np.sin(angle)
    output = np.zeros((len(parameters), *tools.shape[:2], 4), dtype=np.float32)
    output[..., 0] = relative[None, ..., 0] * cosine - relative[None, ..., 1] * sine + center[0]
    output[..., 1] = relative[None, ..., 0] * sine + relative[None, ..., 1] * cosine + center[1]
    output[..., 2] = tools[None, ..., 2]
    output[..., :2] += (parameters[:, :2] / (1000 * unit_to_meter))[:, None, None, :]
    return output


def compose_yaw(quaternion, degrees):
    """Premultiply a simulator wxyz quaternion by a world-z rotation."""
    q = np.asarray(quaternion, dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) == 0 or not np.isfinite(degrees):
        raise ValueError("Invalid grasp quaternion")
    w, x, y, z = q / np.linalg.norm(q)
    c, s = np.cos(np.deg2rad(degrees) / 2), np.sin(np.deg2rad(degrees) / 2)
    return [float(value) for value in (c*w - s*z, c*x - s*y, c*y + s*x, c*z + s*w)]


def bounded_gaussian(rng, count, mean, std, bounds):
    """Rejection sample an independent Gaussian inside symmetric bounds."""
    mean, std, bounds = (np.asarray(value, dtype=np.float64) for value in (mean, std, bounds))
    if (count < 0 or mean.shape != (3,) or std.shape != (3,) or bounds.shape != (3,)
            or not all(np.isfinite(value).all() for value in (mean, std, bounds))
            or (std <= 0).any() or (bounds <= 0).any() or (np.abs(mean) > bounds).any()):
        raise ValueError("Invalid bounded Gaussian parameters")
    samples = rng.normal(mean, std, (count, 3))
    for _ in range(10000):
        invalid = (np.abs(samples) > bounds).any(axis=1)
        if not invalid.any():
            return samples
        samples[invalid] = rng.normal(mean, std, (int(invalid.sum()), 3))
    raise RuntimeError("Bounded Gaussian could not supply valid samples")


def validated_scores(score_fn, parameters):
    scores = np.asarray(score_fn(parameters), dtype=np.float64)
    if scores.shape != (len(parameters),) or not np.isfinite(scores).all():
        raise ValueError("Candidate scorer returned invalid scores")
    return scores


def search_poses(score_fn, seed, population=64, elites=8, iterations=10,
                 bounds=(40., 40., 10.), initial_std=(8., 8., 10.),
                 minimum_std=(1., 1., .5), callback=None):
    """Retain scored elites; compare with random search from the same initial prior.

    Both methods share the first population. Each has population +
    (iterations-1)*(population-elites) scored candidates in its search budget.
    Carried elite rows are logged but not re-evaluated or charged again.
    """
    if not 1 <= elites < population or iterations < 1:
        raise ValueError("Require 1 <= elites < population and iterations >= 1")
    bounds, initial_std, minimum_std = (np.asarray(value, dtype=np.float64)
                                       for value in (bounds, initial_std, minimum_std))
    if (bounds.shape != (3,) or initial_std.shape != (3,)
            or minimum_std.shape != (3,)
            or not all(np.isfinite(value).all()
                       for value in (bounds, initial_std, minimum_std))
            or (bounds <= 0).any() or (initial_std <= 0).any()
            or (minimum_std <= 0).any() or (initial_std > bounds).any()
            or (minimum_std > bounds).any()):
        raise ValueError("Invalid search bounds or standard deviations")
    rng = np.random.default_rng(seed)
    mean = np.zeros(3)
    first = bounded_gaussian(rng, population, mean, initial_std, bounds)
    first_scores = validated_scores(score_fn, first)
    candidates, scores = first.copy(), first_scores.copy()
    candidate_ids = np.arange(population)
    next_id = population
    states, rows = [], []
    std = initial_std.copy()
    for iteration in range(iterations):
        is_new = np.ones(population, dtype=bool)
        if iteration:
            fresh = bounded_gaussian(rng, population - elites, mean, std, bounds)
            candidates = np.concatenate([kept_candidates, fresh])
            scores = np.concatenate([kept_scores, validated_scores(score_fn, fresh)])
            candidate_ids = np.concatenate([kept_ids, np.arange(next_id, next_id + len(fresh))])
            next_id += len(fresh)
            is_new[:elites] = False
        order = np.argsort(-scores, kind="stable")
        retained = order[:elites]
        elite_flags = np.zeros(population, dtype=bool)
        elite_flags[retained] = True
        for index, (parameters, score) in enumerate(zip(candidates, scores)):
            rows.append({
                "method": "cem", "iteration": iteration + 1, "candidate_id": int(candidate_ids[index]),
                "new_evaluation": bool(is_new[index]), "elite": bool(elite_flags[index]),
                "dx_mm": float(parameters[0]), "dy_mm": float(parameters[1]),
                "yaw_deg": float(parameters[2]), "token_mean_cosine": float(score),
            })
        kept_candidates, kept_scores, kept_ids = candidates[retained], scores[retained], candidate_ids[retained]
        next_mean = kept_candidates.mean(axis=0)
        next_std = np.maximum(kept_candidates.std(axis=0), minimum_std)
        state = {
            "iteration": iteration + 1, "sampling_mean_mm_mm_deg": mean.tolist(),
            "sampling_std_mm_mm_deg": std.tolist(),
            "refitted_mean_mm_mm_deg": next_mean.tolist(), "refitted_std_mm_mm_deg": next_std.tolist(),
            "best_score": float(kept_scores[0]), "best_pose_mm_mm_deg": kept_candidates[0].tolist(),
            "new_evaluations": int(is_new.sum()), "cumulative_evaluations": next_id,
        }
        states.append(state)
        if callback:
            callback(state, rows[-population:])
        mean, std = next_mean, next_std
    cem_best = {"parameters": kept_candidates[0].tolist(), "score": float(kept_scores[0])}
    budget = population + (iterations - 1) * (population - elites)
    random_rng = np.random.default_rng(seed + 1)
    random_candidates, random_scores = [first], [first_scores]
    # Keep batches small enough that GPU inference and progress remain visible.
    remaining = budget - population
    for chunk in range((remaining + population - 1) // population):
        count = min(population, remaining - chunk * population)
        fresh = bounded_gaussian(random_rng, count, np.zeros(3), initial_std, bounds)
        fresh_scores = validated_scores(score_fn, fresh)
        random_candidates.append(fresh)
        random_scores.append(fresh_scores)
        if callback:
            callback({"method": "random_progress", "evaluations": population + chunk * population + count,
                      "budget": budget}, [])
    random_candidates, random_scores = np.concatenate(random_candidates), np.concatenate(random_scores)
    for index, (parameters, score) in enumerate(zip(random_candidates, random_scores)):
        rows.append({
            "method": "random", "iteration": 0, "candidate_id": index,
            "new_evaluation": True, "elite": "",
            "dx_mm": float(parameters[0]), "dy_mm": float(parameters[1]),
            "yaw_deg": float(parameters[2]), "token_mean_cosine": float(score),
        })
    best_index = int(np.argmax(random_scores))
    return {
        "cem": cem_best,
        "random": {"parameters": random_candidates[best_index].tolist(), "score": float(random_scores[best_index])},
        "evaluations_per_method": budget, "shared_initial_evaluations": population,
        "unique_scorer_evaluations": 2 * budget - population,
    }, states, rows


def require_hash(path, expected):
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"Input changed since the recorded diagnostic: {path}")
    return actual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dyn-artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--cuda-id", type=int, default=0)
    parser.add_argument("--population", type=int, default=64)
    parser.add_argument("--elites", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()
    started = time.monotonic()
    if args.output.exists():
        raise FileExistsError(f"Choose a new output directory: {args.output}")
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    root = args.dyn_artifacts.resolve()
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("kind") != "recorded_action_interface_diagnostic":
        raise ValueError("Input must be the recorded-action Dyn artifacts")
    inputs = {str(root / name): sha256(root / name) for name in ("metadata.json", "observations.npz", "model_config.yaml")}
    checkpoint_path = Path(metadata["checkpoint"])
    inputs[str(checkpoint_path)] = require_hash(checkpoint_path, metadata["checkpoint_sha256"])
    scene_dir = Path(metadata["scene_dir"])
    inputs[str(scene_dir / "config.yaml")] = require_hash(scene_dir / "config.yaml", metadata["scene_config_sha256"])
    for filename in ("net/model/model.py", "net/model/encoder.py", "net/model/attention.py"):
        inputs[str(PROJECT_ROOT / filename)] = sha256(PROJECT_ROOT / filename)
        if filename in metadata["source_file_sha256"]:
            require_hash(PROJECT_ROOT / filename, metadata["source_file_sha256"][filename])
    with np.load(root / "observations.npz", allow_pickle=False) as stream:
        cache = {key: stream[key] for key in (
            "frame_indices", "simulation_times", "observed", "z_reference", "ee_observed",
        )}
    frames = cache["frame_indices"].tolist()
    schedule = metadata["rollout_frames"]
    if (frames != metadata["observed_frames"] or not frames or frames[0] != 0
            or frames[-1] != metadata["goal_frame"] or not (np.diff(frames) > 0).all()
            or len(schedule) < 2 or schedule[0] != 0 or not set(schedule).issubset(frames)
            or not np.all(np.diff(schedule) == metadata["next_frame_offset"])
            or schedule[-1] != metadata["final_evaluated_frame"]):
        raise ValueError("Invalid recorded frame schedule")
    if (cache["observed"].ndim != 3 or cache["observed"].shape[0] != len(frames)
            or cache["observed"].shape[-1] != 4 or cache["z_reference"].ndim != 3
            or cache["z_reference"].shape[0] != len(frames)
            or cache["ee_observed"].shape != (*cache["observed"].shape[:2], 3)
            or cache["simulation_times"].shape != (len(frames),)
            or not (np.diff(cache["simulation_times"]) > 0).all()
            or not all(np.isfinite(value).all() for value in cache.values())):
        raise ValueError("Invalid cached observations/tools/latents")
    nominal_path = root / f"rollout_{schedule[-1]:03d}.npz"
    inputs[str(nominal_path)] = sha256(nominal_path)
    with np.load(nominal_path, allow_pickle=False) as stream:
        nominal_latent = stream["z_predicted"]

    import torch
    from omegaconf import OmegaConf
    from net.model.model import Predictor
    from net.pipeline.random import manual_seed

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; submit the PBS job or use an allocated compute node")
    torch.cuda.set_device(args.cuda_id)
    device = torch.device(f"cuda:{args.cuda_id}")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    manual_seed(int(metadata["seed"]))
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)
    cfg = OmegaConf.load(root / "model_config.yaml")
    OmegaConf.resolve(cfg)
    if cfg.model.condition.ee_shape != "both" or cfg.dimensions.input_dim != 4:
        raise ValueError("Requires the trained two-endpoint xyz-label tool interface")
    unit_to_meter = float(cfg.augmentation.render.unit_to_meter)
    if not np.isclose(unit_to_meter, float(metadata["unit_to_meter"])):
        raise ValueError("Coordinate unit conversion differs from the recorded diagnostic")
    n_tokens = cfg.dimensions.num_query_in + int(cfg.model.high_level_token in ("mean", "max"))
    if (cache["z_reference"].shape[1:] != (n_tokens, cfg.dimensions.latent_dim)
            or cache["observed"].shape[1] != cfg.dataset.n_points):
        raise ValueError("Cached dimensions do not match the trained model")
    scene_cfg = OmegaConf.load(scene_dir / "config.yaml")
    OmegaConf.resolve(scene_cfg)
    if (scene_cfg.ee.type != "gripper" or list(scene_cfg.actions) != ["grasp"]
            or scene_cfg.actions.grasp.type != "grasp"
            or not np.allclose(np.asarray(scene_cfg.actions.grasp.from_offset, dtype=float), 0)
            or "init_pos" in scene_cfg.actions.grasp or "init_quat" in scene_cfg.actions.grasp):
        raise ValueError("This first planner supports one stationary grasp with no approach offset or separate init pose")
    center = np.asarray(scene_cfg.actions.grasp.to_pos, dtype=float)
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError("Invalid nominal grasp position")
    tool_sequence = cache["ee_observed"][[frames.index(frame) for frame in schedule]]
    model = Predictor(cfg).to(device).eval()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    result = model.load_state_dict(canonical_model_state(checkpoint["model_state_dict"]), strict=True)
    del checkpoint
    print(f"Checkpoint load: missing={result.missing_keys}; unexpected={result.unexpected_keys}", flush=True)
    initial = torch.as_tensor(cache["z_reference"][0:1], device=device)
    goal = torch.as_tensor(cache["z_reference"][-1:], device=device)
    with torch.no_grad():
        for index in (0, -1):
            observation = torch.as_tensor(cache["observed"][index:index+1] if index == 0 else cache["observed"][-1:], device=device)
            _, _, encoded = model.reconstruct(observation, None, decode=False)
            expected = initial if index == 0 else goal
            if not torch.isfinite(encoded).all() or not torch.allclose(encoded, expected, rtol=1e-4, atol=1e-5):
                raise ValueError("Initial/goal cache does not correspond to this model and observation")

    def rollout(parameters):
        tools = torch.as_tensor(transform_tools(tool_sequence, center, parameters, unit_to_meter), device=device)
        latent = initial.expand(len(parameters), -1, -1).clone()
        for interval in range(len(schedule) - 1):
            endpoints = torch.stack([tools[:, interval], tools[:, interval + 1]], dim=0)
            _, _, latent = model.predict(endpoints, None, latent, decode=False)
        if not torch.isfinite(latent).all():
            raise RuntimeError("Nonfinite candidate rollout")
        return latent

    @torch.no_grad()
    def score(parameters):
        values = []
        for begin in range(0, len(parameters), args.batch_size):
            predicted = rollout(parameters[begin:begin + args.batch_size])
            norms = torch.linalg.vector_norm(predicted, dim=-1)
            if (norms == 0).any() or (torch.linalg.vector_norm(goal, dim=-1) == 0).any():
                raise ValueError("Zero-norm latent; cosine is undefined")
            values.append(torch.nn.functional.cosine_similarity(predicted, goal, dim=-1).mean(dim=-1).cpu().numpy())
        return np.concatenate(values)

    with torch.no_grad():
        zero = np.zeros((1, 3))
        nominal = rollout(zero)
        expected = torch.as_tensor(nominal_latent, device=device)[None]
        if nominal.shape != expected.shape or not torch.allclose(nominal, expected, rtol=1e-4, atol=1e-5):
            raise ValueError("Nominal action does not reproduce the cached rollout; stop before searching")
        nominal_error = float((nominal - expected).abs().max())
        nominal_score = float(score(zero)[0])
    print(f"NOMINAL_ROLLOUT_PASS: max_abs_error={nominal_error:g}, goal_score={nominal_score:.8f}", flush=True)
    args.output.mkdir(parents=True, exist_ok=False)
    journal_path = args.output / "progress.jsonl"

    def progress(state, rows):
        with journal_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(state, allow_nan=False) + "\n")
        print(json.dumps(state, allow_nan=False), flush=True)
        if rows:
            path = args.output / f"cem_iteration_{state['iteration']:02d}.csv"
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

    results, distributions, rows = search_poses(
        score, args.seed, args.population, args.elites, args.iterations, callback=progress,
    )
    with (args.output / "candidates.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    actions = {}
    with torch.no_grad():
        for method in ("cem", "random"):
            parameters = np.asarray(results[method]["parameters"])
            selected_cfg = OmegaConf.create(OmegaConf.to_container(scene_cfg, resolve=True))
            position = center.copy()
            position[:2] += parameters[:2] / (1000 * unit_to_meter)
            selected_cfg.actions.grasp.to_pos = position.tolist()
            selected_cfg.actions.grasp.to_quat = compose_yaw(scene_cfg.actions.grasp.to_quat, parameters[2])
            selected_cfg.render = False
            selected_cfg.cuda_id = 0
            selected_cfg.scene_id = "0000"
            # Source config is copied; simulation must use a distinct, fresh root.
            selected_cfg.log.base_dir = str(args.output.resolve() / f"{method}_execution" / "scenes")
            Path(selected_cfg.log.base_dir).mkdir(parents=True, exist_ok=True)
            OmegaConf.save(selected_cfg, args.output / f"{method}_scene.yaml")
            latent = rollout(parameters[None]).cpu().numpy()[0]
            np.savez_compressed(args.output / f"{method}_selected.npz", z_predicted=latent, parameters=parameters)
            selected_tools = transform_tools(tool_sequence[[0, -1]], center, parameters[None], unit_to_meter)[0]
            for endpoint, tools in zip(("start", "end"), selected_tools):
                write_cloud(args.output / f"{method}_tool_{endpoint}.ply", tools[:, :3], np.zeros(len(tools), dtype=int))
            actions[method] = {
                "parameters_dx_mm_dy_mm_yaw_deg": parameters.tolist(),
                "to_pos_simulation_units": position.tolist(),
                "to_quat_wxyz": list(selected_cfg.actions.grasp.to_quat),
                "initial_opening_simulation_units": float(selected_cfg.actions.grasp.init_d),
                "target_opening_simulation_units": float(selected_cfg.actions.grasp.close_d),
                "predicted_score": results[method]["score"],
                "gain_over_nominal": results[method]["score"] - nominal_score,
                "exported_scene_config": str((args.output / f"{method}_scene.yaml").resolve()),
                "executed": False,
            }
    summary = {
        "kind": "fixed_tool_pose_cem_search", "goal_frame": int(metadata["goal_frame"]),
        "prediction_endpoint_frame": int(schedule[-1]),
        "prediction_endpoint_gap_seconds": float(cache["simulation_times"][-1] - cache["simulation_times"][frames.index(schedule[-1])]),
        "nominal_goal_score": nominal_score, "nominal_latent_max_abs_error": nominal_error,
        "evaluations_per_method": results["evaluations_per_method"],
        "shared_initial_evaluations": results["shared_initial_evaluations"],
        "unique_search_evaluations": results["unique_scorer_evaluations"],
        "actions": actions, "elapsed_seconds": time.monotonic() - started,
        "interpretation": "A higher model score does not establish better executed geometry or topology. Selected actions must be simulated independently.",
    }
    protocol = {
        "kind": "fixed_tool_pose_cem_search", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dyn_artifacts": str(root), "input_sha256": inputs, "script_sha256": sha256(__file__),
        "seed": args.seed, "batch_size": args.batch_size,
        "population": args.population, "elites": args.elites, "iterations": args.iterations,
        "score": "Maximize mean cosine over corresponding latent tokens, including appended token; cost=-score",
        "bounds_dx_mm_dy_mm_yaw_deg": [40, 40, 10],
        "initial_std_mm_mm_deg": [8, 8, 10], "minimum_std_mm_mm_deg": [1, 1, .5],
        "sampling": "Independent bounded Gaussian via rejection; elite mean/std refit; scored elites carried without re-evaluation",
        "random_baseline": "Same initial population, same initial prior, same candidate score budget; remaining candidates use seed+1",
        "fixed_tool": OmegaConf.to_container(scene_cfg.ee, resolve=True),
        "fixed_action_template": OmegaConf.to_container(scene_cfg.actions.grasp, resolve=True),
        "unit_to_meter": unit_to_meter, "prediction_frames": schedule,
        "goal_information": "Only the final rendered goal observation latent is used for scoring; no intermediate object references or future object geometry enter candidate scores",
        "limitations": "One demonstration-derived fixed tool/closing template; nominal pose is a reference for offsets. No tool category/closing-width search, collision rejection, executed-result evaluation, original-planner equivalence, or success-rate estimate. Initial/minimum std and cosine aggregation are declared implementation choices, not verified author settings. End state is frame 60 while goal is frame 62.",
        "nominal_check_evaluations": 2, "selected_action_export_evaluations": 2,
        "environment": {"python": sys.version, "numpy": np.__version__, "torch": torch.__version__,
                        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(device)},
    }
    for filename, value in (("summary.json", summary), ("metadata.json", protocol), ("distributions.json", distributions)):
        (args.output / filename).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, allow_nan=False), flush=True)
    print(f"POSE_SEARCH_COMPLETE_EXECUTION_PENDING: {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
