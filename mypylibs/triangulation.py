"""Step 4: intersect viewing rays, then check their projections in both images."""
import json

import cv2
import numpy as np

from mypylibs.sfm import common_points


def project(points, pose, K):
    camera = points @ pose[:, :3].T + pose[:, 3]
    pixels = camera @ K.T
    denominator = np.where(abs(pixels[:, 2:3]) > 1e-9, pixels[:, 2:3], 1e-9)
    return pixels[:, :2] / denominator, camera[:, 2]


def triangulate(p1, p2, pose1, pose2, K, max_error=2.0):
    if len(p1) == 0:
        return np.empty((0, 3)), np.zeros(0, bool), np.empty(0)
    homogeneous = cv2.triangulatePoints(K @ pose1, K @ pose2, p1.T.astype(float), p2.T.astype(float))
    with np.errstate(divide='ignore', invalid='ignore'):
        points = (homogeneous[:3] / homogeneous[3]).T
    q1, z1 = project(points, pose1, K)
    q2, z2 = project(points, pose2, K)
    errors = np.maximum(np.linalg.norm(q1 - p1, axis=1), np.linalg.norm(q2 - p2, axis=1))
    centre1 = -pose1[:, :3].T @ pose1[:, 3]
    centre2 = -pose2[:, :3].T @ pose2[:, 3]
    ray1, ray2 = points - centre1, points - centre2
    cosine = np.sum(ray1 * ray2, axis=1) / np.maximum(np.linalg.norm(ray1, axis=1) *
                                                               np.linalg.norm(ray2, axis=1), 1e-12)
    # Near-parallel rays have unreliable depth even when pixel error is small.
    angle = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
    keep = (np.isfinite(points).all(axis=1) & (z1 > 0) & (z2 > 0) &
            (errors < max_error) & (angle > 0.5))
    return points, keep, errors


def save_ply(path, points, colours=None):
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    if colours is None:
        colours = np.full(points.shape, 180, dtype=np.uint8)
    if not np.isfinite(points).all():
        raise ValueError('Point cloud contains non-finite coordinates')
    with path.open('w', encoding='ascii') as file:
        file.write(f'ply\nformat ascii 1.0\nelement vertex {len(points)}\n'
                   'property float x\nproperty float y\nproperty float z\n'
                   'property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n')
        np.savetxt(file, np.column_stack((points, colours)), fmt='%.6f %.6f %.6f %d %d %d')


def two_view_cloud(views, K, output_dir):
    poses = np.load(output_dir / 'relative_poses.npz')
    origin = np.column_stack((np.eye(3), np.zeros(3)))
    rows, best = [], np.empty((0, 3))
    for i, (R, t) in enumerate(zip(poses['R'], poses['t'])):
        if not np.isfinite(R).all():
            continue
        _, p1, p2 = common_points(views[i], views[i + 1])
        points, keep, errors = triangulate(p1, p2, origin, np.column_stack((R, t)), K)
        rows.append(dict(pair=[i, i + 1], points=int(keep.sum()),
                         mean_reprojection_px=float(errors[keep].mean()) if keep.any() else None))
        if keep.sum() > len(best):
            best = points[keep]
    # Each pair has its own unknown scale, so do not concatenate local clouds.
    save_ply(output_dir / 'two_view.ply', best)
    (output_dir / 'triangulation.json').write_text(json.dumps(rows, indent=2))
    print(f'Step 4: best consecutive pair retained {len(best)} triangulated points', flush=True)
    return rows
