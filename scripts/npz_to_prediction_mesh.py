import sys
from pathlib import Path

import numpy as np
import open3d as o3d
from skimage.measure import marching_cubes


src = Path(sys.argv[1])
dst = Path(sys.argv[2])

data = np.load(src)
query = data["query"]
pred = data["predicted_class"]

x = np.unique(query[:, 0])
y = np.unique(query[:, 1])
z = np.unique(query[:, 2])

volume = pred.reshape(len(x), len(y), len(z))
occupied = (volume > 0).astype(np.float32)

vertices, faces, _, _ = marching_cubes(
    occupied,
    level=0.5,
    spacing=(
        float(x[1] - x[0]),
        float(y[1] - y[0]),
        float(z[1] - z[0]),
    ),
)

vertices[:, 0] += x[0]
vertices[:, 1] += y[0]
vertices[:, 2] += z[0]

mesh = o3d.geometry.TriangleMesh(
    o3d.utility.Vector3dVector(vertices),
    o3d.utility.Vector3iVector(faces),
)

mesh.compute_vertex_normals()
o3d.io.write_triangle_mesh(str(dst), mesh)

print(f"saved: {dst}")
print(f"vertices: {len(vertices)}, faces: {len(faces)}")
