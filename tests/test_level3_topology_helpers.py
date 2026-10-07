"""Analytic surface checks for Euler counts and face-expanded mesh welding."""

import unittest

import numpy as np

from scripts.audit_level3_topology import prepare_surface, surface_counts


class SurfaceTopologyTests(unittest.TestCase):
    def test_welds_face_expanded_tetrahedron_and_removes_padding(self):
        vertices = np.array([[0., 0, 0], [1., 0, 0], [0., 1, 0], [0., 0, 1]])
        faces = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]])
        expanded = vertices[faces].reshape(-1, 3)
        expanded = np.vstack([expanded, np.tile(expanded[0], (3, 1))])
        welded, triangles, removed, duplicates = prepare_surface(expanded, np.arange(15).reshape(-1, 3))
        self.assertEqual((len(welded), len(triangles), removed, duplicates), (4, 4, 1, 0))
        counts = surface_counts(triangles)
        self.assertEqual(counts["euler_characteristic"], 2)
        self.assertEqual(counts["boundary_edges"], 0)
        self.assertEqual(counts["nonmanifold_edges"], 0)

    def test_periodic_torus_has_euler_zero_and_no_boundary(self):
        nu, nv = 8, 6
        u, v = np.meshgrid(np.arange(nu) * 2 * np.pi / nu,
                           np.arange(nv) * 2 * np.pi / nv, indexing="ij")
        vertices = np.stack([(3 + np.cos(v)) * np.cos(u),
                             (3 + np.cos(v)) * np.sin(u), np.sin(v)], axis=-1).reshape(-1, 3)
        faces = []
        for i in range(nu):
            for j in range(nv):
                a, b = i * nv + j, ((i + 1) % nu) * nv + j
                c, d = ((i + 1) % nu) * nv + (j + 1) % nv, i * nv + (j + 1) % nv
                faces.extend([[a, b, c], [a, c, d]])
        _, triangles, removed, duplicates = prepare_surface(vertices, np.array(faces))
        counts = surface_counts(triangles)
        self.assertEqual(counts["euler_characteristic"], 0)
        self.assertEqual(counts["boundary_edges"], 0)
        self.assertEqual(counts["nonmanifold_edges"], 0)
        self.assertEqual((removed, duplicates), (0, 0))

    def test_detects_open_surface_and_duplicate_triangles(self):
        vertices = np.array([[0., 0, 0], [1., 0, 0], [0., 1, 0]])
        _, triangles, _, duplicates = prepare_surface(vertices, np.array([[0, 1, 2]]))
        self.assertEqual(surface_counts(triangles)["boundary_edges"], 3)
        self.assertEqual(duplicates, 0)
        _, _, _, duplicates = prepare_surface(vertices, np.array([[0, 1, 2], [2, 1, 0]]))
        self.assertEqual(duplicates, 1)
        with self.assertRaisesRegex(ValueError, "Out-of-range"):
            prepare_surface(vertices, np.array([[0, 1, 3]]))

    def test_padding_only_label_has_no_surface(self):
        vertices = np.zeros((3, 3))
        welded, faces, removed, duplicates = prepare_surface(vertices, np.array([[0, 1, 2]]))
        self.assertEqual(welded.shape, (0, 3))
        self.assertEqual(faces.shape, (0, 3))
        self.assertEqual((removed, duplicates), (1, 0))
        self.assertEqual(surface_counts(faces)["triangles"], 0)
        welded, faces, removed, duplicates = prepare_surface(vertices, np.empty((0, 3), dtype=int))
        self.assertEqual((len(welded), len(faces), removed, duplicates), (0, 0, 0, 0))


if __name__ == "__main__":
    unittest.main()
