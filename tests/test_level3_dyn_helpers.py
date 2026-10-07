"""CPU checks for the recorded-scene diagnostic's input/export boundaries.

These do not verify rendering, CUDA inference, or prediction quality.
"""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.check_level3_dyn import (
    canonical_model_state, frame_schedule, validate_arrays, write_cloud,
)


def scene_arrays():
    # One valid triangle and one degenerate padding triangle in each mesh.
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    faces = np.array([[0, 1, 2], [0, 0, 0]], dtype=np.int32)
    frames = 6
    arrays = {
        "scene": np.zeros(frames, dtype=np.int32),
        "frame": np.arange(frames, dtype=np.int32),
        "genus": np.tile([-1, 1, -1, -1, -1], (frames, 1)).astype(np.int32),
        "obj_vert_labels": np.ones((frames, 3, 1), dtype=np.int32),
        "obj_face_labels": np.ones((frames, 2, 1), dtype=np.int32),
        "ee_observed": np.zeros((frames, 4, 3), dtype=np.float32),
    }
    for prefix in ("obj", "ee"):
        arrays[f"{prefix}_verts"] = np.tile(vertices, (frames, 1, 1))
        arrays[f"{prefix}_faces"] = np.tile(faces, (frames, 1, 1))
    return arrays


class InputValidationTests(unittest.TestCase):
    def test_accepts_degenerate_face_padding_with_nonempty_mesh(self):
        self.assertEqual(validate_arrays(scene_arrays(), 5, 4), 6)

    def test_rejects_corrupt_mesh_indices_labels_and_nonfinite_geometry(self):
        mutations = (
            ("obj_faces", (0, 0, 0), -1, "Out-of-range"),
            ("ee_faces", (0, 0, 0), 3, "Out-of-range"),
            ("obj_face_labels", (0, 0, 0), 5, "component range"),
            ("ee_observed", (0, 0, 0), np.nan, "nonfinite"),
        )
        for key, index, value, message in mutations:
            with self.subTest(key=key, value=value):
                arrays = scene_arrays()
                arrays[key][index] = value
                with self.assertRaisesRegex(ValueError, message):
                    validate_arrays(arrays, 5, 4)

    def test_rejects_a_missing_or_empty_frame(self):
        arrays = scene_arrays()
        arrays["obj_verts"][2] = 0
        with self.assertRaisesRegex(ValueError, "Empty obj mesh"):
            validate_arrays(arrays, 5, 4)
        arrays = scene_arrays()
        arrays["frame"][2] = 4
        with self.assertRaisesRegex(ValueError, "consecutive"):
            validate_arrays(arrays, 5, 4)

    def test_frame_schedule_never_shortens_last_prediction_interval(self):
        predicted, observed = frame_schedule(63, 5)
        self.assertEqual(len(predicted) - 1, 12)
        self.assertEqual(predicted[-1], 60)
        self.assertNotIn(62, predicted)
        self.assertEqual(observed[-1], 62)
        self.assertEqual(np.diff(predicted).tolist(), [5] * 12)
        self.assertEqual(frame_schedule(61, 5), (predicted, predicted))
        with self.assertRaises(ValueError):
            frame_schedule(5, 5)


class CheckpointAndExportTests(unittest.TestCase):
    def test_keeps_both_ae_and_dyn_weights_for_ddp_and_single_gpu(self):
        for prefix in ("module.model.", "model.", ""):
            with self.subTest(prefix=prefix):
                state = {prefix + "condition.weight": object(),
                         prefix + "encoder.basis": object()}
                mapped = canonical_model_state(state)
                self.assertIs(mapped["condition.weight"], state[prefix + "condition.weight"])
                self.assertIs(mapped["encoder.basis"], state[prefix + "encoder.basis"])
                self.assertEqual(len(state), 2)

    def test_rejects_ae_checkpoint_and_ambiguous_names(self):
        with self.assertRaisesRegex(ValueError, "no condition weights"):
            canonical_model_state({"model.encoder.basis": object()})
        with self.assertRaisesRegex(ValueError, "Duplicate checkpoint key"):
            canonical_model_state({"condition.weight": object(),
                                   "module.model.condition.weight": object()})

    def test_cloud_export_handles_empty_prediction_and_rejects_bad_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cloud.ply"
            write_cloud(path, np.empty((0, 3)), np.empty(0, dtype=int))
            self.assertIn("element vertex 0", path.read_text(encoding="ascii"))
            write_cloud(path, [[0.1, 0.2, 0.3]], [2])
            self.assertIn("242 142 43", path.read_text(encoding="ascii"))
            for label in (-1, 5, 1.5):
                with self.subTest(label=label), self.assertRaises(ValueError):
                    write_cloud(path, [[0, 0, 0]], [label])


if __name__ == "__main__":
    unittest.main()
