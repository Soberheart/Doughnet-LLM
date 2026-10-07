"""CPU tests for the Level 3 pose-search boundaries.

These tests do not load CUDA, checkpoints, or the simulator. They cover the
deterministic geometry and budget logic used before the PBS inference run.
"""

import unittest

import numpy as np

from scripts.plan_level3_pose_cem import (
    bounded_gaussian, compose_yaw, search_poses, transform_tools,
)


class PoseTransformTests(unittest.TestCase):
    def test_zero_pose_preserves_tool_trajectory_and_translation_is_converted(self):
        tools = np.array([
            [[0.0, 0.0, 0.1], [0.1, 0.0, 0.1]],
            [[0.0, 0.1, 0.1], [0.1, 0.1, 0.1]],
        ])
        center = np.array([0.0, 0.0, 0.1])
        zero = transform_tools(tools, center, [[0.0, 0.0, 0.0]], 0.2)
        np.testing.assert_allclose(zero[0, ..., :3], tools)
        shifted = transform_tools(tools, center, [[10.0, -20.0, 0.0]], 0.2)
        np.testing.assert_allclose(
            shifted[0, ..., :2], tools[..., :2] + [0.05, -0.1], atol=1e-6
        )
        np.testing.assert_array_equal(shifted[0, ..., 3], 0.0)

    def test_yaw_composition_uses_wxyz(self):
        np.testing.assert_allclose(
            compose_yaw([1.0, 0.0, 0.0, 0.0], 90.0),
            [np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)],
        )

    def test_invalid_transform_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            transform_tools(np.zeros((2, 3)), np.zeros(3), [[0, 0, 0]], 0.2)
        with self.assertRaises(ValueError):
            compose_yaw([0, 0, 0, 0], 0)


class SearchTests(unittest.TestCase):
    def test_search_budget_and_finite_bounds(self):
        seen = []

        def score(parameters):
            seen.extend(np.asarray(parameters).tolist())
            return -np.sum(np.asarray(parameters) ** 2, axis=1)

        result, states, rows = search_poses(
            score, seed=0, population=8, elites=2, iterations=3,
            bounds=(4, 4, 4), initial_std=(1, 1, 1), minimum_std=(.2, .2, .2),
        )
        expected = 8 + 2 * (8 - 2)
        self.assertEqual(result["evaluations_per_method"], expected)
        self.assertEqual(result["unique_scorer_evaluations"], 2 * expected - 8)
        self.assertEqual(len(states), 3)
        self.assertEqual(len([row for row in rows if row["method"] == "cem"]), 3 * 8)
        self.assertEqual(len([row for row in rows if row["method"] == "random"]), expected)
        self.assertTrue(np.isfinite(seen).all())
        self.assertTrue((np.abs(np.asarray(seen)) <= 4).all())

    def test_bad_search_scales_are_rejected(self):
        with self.assertRaises(ValueError):
            bounded_gaussian(np.random.default_rng(0), 2, [0, 0, 0], [1, 1], [2, 2, 2])
        with self.assertRaises(ValueError):
            search_poses(lambda values: np.zeros(len(values)), 0,
                         bounds=(1, 1, 1), initial_std=(2, 1, 1))


if __name__ == "__main__":
    unittest.main()
