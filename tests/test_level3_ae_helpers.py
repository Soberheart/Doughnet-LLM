"""Check that corrupt or misaligned observation caches cannot be compared."""

import unittest

import numpy as np

from scripts.check_level3_ae import validate_observations


def observation_cache():
    observed = np.zeros((3, 4, 4), dtype=np.float32)
    observed[..., :3] = 0.1
    observed[..., 3] = 1
    return {
        "frame_indices": np.array([0, 5, 10], dtype=np.int32),
        "observed": observed,
        "z_reference": np.ones((3, 2, 8), dtype=np.float32),
        "genus_annotation": np.tile([-1, 1, -1, -1, -1], (3, 1)).astype(np.int32),
        "simulation_times": np.array([0, 0.16, 0.32]),
    }


def validate(cache):
    return validate_observations(cache, 11, 4, 2, 8, 5, 3)


class ObservationCacheTests(unittest.TestCase):
    def test_accepts_ordered_sparse_frames(self):
        self.assertEqual(validate(observation_cache()), [0, 5, 10])

    def test_rejects_duplicate_reordered_or_out_of_range_frames(self):
        for frames in ([0, 5, 5], [0, 10, 5], [-1, 5, 10], [0, 5, 11]):
            with self.subTest(frames=frames):
                cache = observation_cache()
                cache["frame_indices"] = np.array(frames, dtype=np.int32)
                with self.assertRaisesRegex(ValueError, "unique, ordered, and in range"):
                    validate(cache)

    def test_rejects_corrupt_latents_observations_and_topology(self):
        for key, index, value, message in (
            ("z_reference", (1, 0, 0), np.nan, "nonfinite"),
            ("observed", (1, 0, 3), 1.5, "observation labels"),
            ("genus_annotation", (1, 1), 3, "genus annotations"),
        ):
            with self.subTest(key=key):
                cache = observation_cache()
                cache[key][index] = value
                with self.assertRaisesRegex(ValueError, message):
                    validate(cache)
        cache = observation_cache()
        cache["observed"][1, :, :3] = 0
        with self.assertRaisesRegex(ValueError, "Empty cached object"):
            validate(cache)
        cache = observation_cache()
        cache["z_reference"] = cache["z_reference"][:2]
        with self.assertRaisesRegex(ValueError, "Invalid shape"):
            validate(cache)


if __name__ == "__main__":
    unittest.main()
