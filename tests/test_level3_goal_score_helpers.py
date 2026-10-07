"""CPU checks for scoring ambiguity, mask handling, and cache provenance."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.check_level3_goal_score import (
    ae_fingerprints, cosine_scores, discover_ae, goal_overlap, main, spearman,
)


def make_artifacts(directory):
    root = Path(directory) / "dyn_check_example" / "artifacts"
    ae = Path(directory) / "ae_check_example" / "artifacts"
    root.mkdir(parents=True)
    ae.mkdir(parents=True)
    frames = np.array([0, 5, 6])
    latents = np.array([[[0., 1], [1, 0]], [[1., 1], [0, 1]], [[1., 0], [0, 1]]])
    genera = np.tile([-1, 1, -1, -1, -1], (3, 1))
    query = np.array([[0., 0, .1], [.1, 0, .1], [.2, 0, .1]])
    classes = np.array([0, 1, 1], dtype=np.int32)
    metadata = {
        "kind": "recorded_action_interface_diagnostic", "checkpoint_sha256": "fixture",
        "observed_frames": frames.tolist(), "num_source_frames": 7, "goal_frame": 6,
        "rollout_frames": [0, 5], "next_frame_offset": 5, "final_evaluated_frame": 5,
    }
    (root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (root / "metrics.csv").write_text("fixture", encoding="utf-8")
    (root / "model_config.yaml").write_text("fixture", encoding="utf-8")
    np.savez(root / "observations.npz", frame_indices=frames, simulation_times=frames * .032,
             z_reference=latents, genus_annotation=genera)
    ae_metadata = {
        "kind": "same_observation_ae_reconstruction_diagnostic", "dyn_artifacts": str(root.resolve()),
        **ae_fingerprints(root),
    }
    (ae / "metadata.json").write_text(json.dumps(ae_metadata), encoding="utf-8")
    (ae / "summary.json").write_text("{}", encoding="utf-8")
    for index, frame in enumerate(frames):
        np.savez(ae / f"ae_{frame:03d}.npz", query=query, predicted_class=np.array([0, 1, 0]),
                 ground_truth_class=classes, ignored_mask=np.zeros(3, dtype=bool),
                 z_reference=latents[index], ground_truth_genus=genera[index],
                 predicted_genus=genera[index], frame=frame)
    for mode in ("single_step", "rollout"):
        np.savez(root / f"{mode}_005.npz", query=query, predicted_class=classes,
                 ground_truth_class=classes, ground_truth_genus=genera[1], predicted_genus=genera[1],
                 z_predicted=latents[1], source_frame=0, target_frame=5)
    return root, ae


class GoalScoreTests(unittest.TestCase):
    def test_cosine_aggregations_can_choose_different_candidates(self):
        goal = np.array([[10., 0], [1, 0]])
        a = np.array([[10., 0], [-1, 0]])
        b = np.array([[0., 10], [1, 0]])
        score_a, score_b = cosine_scores(a, goal), cosine_scores(b, goal)
        self.assertLess(score_a["token_mean_cosine"], score_b["token_mean_cosine"])
        self.assertGreater(score_a["flattened_cosine"], score_b["flattened_cosine"])
        self.assertLess(score_a["last_token_cosine"], score_b["last_token_cosine"])

    def test_rejects_undefined_or_misaligned_cosine(self):
        for value in (np.zeros((2, 2)), np.full((2, 2), np.nan), np.ones((3, 2))):
            with self.subTest(value=value), self.assertRaises(ValueError):
                cosine_scores(value, np.ones((2, 2)))

    def test_geometry_excludes_both_tool_masks_and_ignores_part_ids(self):
        estimate, goal = np.array([0, 2, 4, 1]), np.array([1, 0, 3, 1])
        result = goal_overlap(estimate, goal, np.array([True, False, False, False]),
                              np.array([False, True, False, False]))
        self.assertEqual(result["comparison_voxels"], 2)
        self.assertEqual(result["goal_occupancy_iou_pct"], 100)
        with self.assertRaisesRegex(ValueError, "No goal occupancy"):
            goal_overlap(estimate, goal, np.ones(4, dtype=bool), np.zeros(4, dtype=bool))

    def test_rank_correlation_handles_ties_and_constant_arrays(self):
        self.assertAlmostEqual(spearman([1, 1, 3], [2, 2, 4]), 1)
        self.assertAlmostEqual(spearman([1, 2, 3], [3, 2, 1]), -1)
        self.assertIsNone(spearman([1, 1, 1], [1, 2, 3]))

    def test_complete_report_distinguishes_self_score_from_decode_quality(self):
        with tempfile.TemporaryDirectory() as directory:
            root, ae = make_artifacts(directory)
            self.assertEqual(discover_ae(root, ae_fingerprints(root), Path(directory)), ae.resolve())
            output = Path(directory) / "output"
            with redirect_stdout(io.StringIO()):
                main(["--dyn-artifacts", str(root), "--ae-artifacts", str(ae), "--output", str(output)])
            result = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(result["num_rows"], 8)
            self.assertAlmostEqual(result["prediction_endpoint_gap_seconds"], .032)
            self.assertAlmostEqual(result["goal_reference"]["token_mean_cosine"], 1)
            self.assertAlmostEqual(result["goal_ae_reconstruction"]["token_mean_cosine"], 1)
            self.assertEqual(result["goal_reference"]["goal_occupancy_iou_pct"], 100)
            self.assertEqual(result["goal_ae_reconstruction"]["goal_occupancy_iou_pct"], 50)
            self.assertEqual(result["mode_diagnostics"]["reference"]["num_frames_excluding_goal"], 2)

    def test_rejects_stale_provenance_before_creating_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root, ae = make_artifacts(directory)
            (root / "metrics.csv").write_text("changed", encoding="utf-8")
            output = Path(directory) / "output"
            with self.assertRaisesRegex(ValueError, "provenance hashes"):
                main(["--dyn-artifacts", str(root), "--ae-artifacts", str(ae), "--output", str(output)])
            self.assertFalse(output.exists())

    def test_rejects_mixed_query_grid_before_creating_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root, ae = make_artifacts(directory)
            packet_path = root / "rollout_005.npz"
            with np.load(packet_path) as stream:
                packet = {key: stream[key] for key in stream.files}
            packet["query"][0, 0] = .7
            np.savez(packet_path, **packet)
            output = Path(directory) / "output"
            with self.assertRaisesRegex(ValueError, "grid, or truth mismatch"):
                main(["--dyn-artifacts", str(root), "--ae-artifacts", str(ae), "--output", str(output)])
            self.assertFalse(output.exists())

    def test_rejects_ambiguous_auto_discovery(self):
        with tempfile.TemporaryDirectory() as directory:
            root, ae = make_artifacts(directory)
            other = Path(directory) / "ae_check_other" / "artifacts"
            other.mkdir(parents=True)
            for filename in ("metadata.json", "summary.json"):
                (other / filename).write_bytes((ae / filename).read_bytes())
            with self.assertRaisesRegex(ValueError, "found 2"):
                discover_ae(root, ae_fingerprints(root), Path(directory))


if __name__ == "__main__":
    unittest.main()
