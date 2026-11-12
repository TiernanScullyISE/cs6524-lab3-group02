"""Step 6: rectify registered views, match pixels, then merge their 3D estimates."""
import json

import cv2
import numpy as np

from mypylibs.sfm import read_frames
from mypylibs.triangulation import save_ply


def stereo_points(first, second, pose1, pose2, K, roi):
    R = pose2[:, :3] @ pose1[:, :3].T
    t = pose2[:, 3] - R @ pose1[:, 3]
    if np.linalg.norm(t) < 1e-6:
        return np.empty((0, 3)), np.empty((0, 3), np.uint8)
    size = (first.shape[1], first.shape[0])
    r1, r2, p1, p2, Q, _, _ = cv2.stereoRectify(K, None, K, None, size, R, t.reshape(3, 1),
                                                flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
    map1 = cv2.initUndistortRectifyMap(K, None, r1, p1, size, cv2.CV_32FC1)
    map2 = cv2.initUndistortRectifyMap(K, None, r2, p2, size, cv2.CV_32FC1)
    left = cv2.remap(first, *map1, cv2.INTER_LINEAR)
    right = cv2.remap(second, *map2, cv2.INTER_LINEAR)
    source_mask = np.zeros(first.shape[:2], np.uint8)
    x1, y1, x2, y2 = roi
    source_mask[int(y1 * size[1]):int(y2 * size[1]), int(x1 * size[0]):int(x2 * size[0])] = 255
    mask1 = cv2.remap(source_mask, *map1, cv2.INTER_NEAREST) != 0
    mask2 = cv2.remap(source_mask, *map2, cv2.INTER_NEAREST) != 0
    grey1, grey2 = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in (left, right)]
    # SGBM searches along rows; transpose a vertically rectified stereo pair.
    axis = int(abs(p2[1, 3]) > abs(p2[0, 3]))
    if axis == 1:
        grey1, grey2, mask1, mask2 = grey1.T.copy(), grey2.T.copy(), mask1.T, mask2.T
    minimum = -64 if p2[axis, 3] > 0 else 0

    def disparity(a, b, low):
        matcher = cv2.StereoSGBM_create(minDisparity=low, numDisparities=128, blockSize=5,
                                        P1=8 * 25, P2=32 * 25, uniquenessRatio=10,
                                        speckleWindowSize=100, speckleRange=2,
                                        disp12MaxDiff=1)
        return matcher.compute(a, b).astype(np.float32) / 16

    d1 = disparity(grey1, grey2, minimum)
    reverse_minimum = -minimum - 127
    d2 = disparity(grey2, grey1, reverse_minimum)
    y, x = np.indices(d1.shape)
    target_x = np.rint(x - d1).astype(int)
    inside = (target_x >= 0) & (target_x < d1.shape[1])
    target_x = np.clip(target_x, 0, d1.shape[1] - 1)
    # A pixel must match back to its original location, in valid image regions.
    valid = (inside & mask1 & mask2[y, target_x] & (d1 > minimum) &
             (d2[y, target_x] > reverse_minimum) & (abs(d1 + d2[y, target_x]) < 1.5))
    if axis == 1:
        d1, valid = d1.T.copy(), valid.T
    rectified = cv2.reprojectImageTo3D(d1, Q)
    finite = np.isfinite(rectified).all(axis=2)
    valid &= finite
    rectified[~finite] = 0
    camera = rectified @ r1
    world = (camera - pose1[:, 3]) @ pose1[:, :3]
    second_depth = (world @ pose2[:, :3].T + pose2[:, 3])[:, :, 2]
    valid &= np.isfinite(world).all(axis=2) & (camera[:, :, 2] > 0) & (second_depth > 0)
    return world[valid], cv2.cvtColor(left, cv2.COLOR_BGR2RGB)[valid]


def dense_cloud(video_path, K, D, undistort, config, poses, cloud, output_path):
    registered = sorted(int(frame) for frame in poses)
    # Spread at most 20 dense stereo pairs over the recovered trajectory.
    indices = np.unique(np.linspace(0, len(registered) - 2, min(20, len(registered) - 1), dtype=int))
    pairs = [(registered[i], registered[i + 1]) for i in indices]
    needed = {f for pair in pairs for f in pair}
    images = {i: frame for i, frame in enumerate(read_frames(video_path, K, D, undistort)) if i in needed}
    points, colours, counts = [], [], []
    sparse = np.asarray(list(cloud.values()))
    for first, second in pairs:
        xyz, rgb = stereo_points(images[first], images[second], poses[first], poses[second],
                                 K, config.get('roi', [0, 0, 1, 1]))
        # Bound distant stereo noise using the sparse map's depth in this camera.
        depth = (sparse @ poses[first][:, :3].T + poses[first][:, 3])[:, 2]
        positive = depth[depth > 0]
        if len(positive):
            dense_depth = (xyz @ poses[first][:, :3].T + poses[first][:, 3])[:, 2]
            keep = (dense_depth > np.median(positive) * 0.1) & (dense_depth < np.median(positive) * 10)
            xyz, rgb = xyz[keep], rgb[keep]
        points.append(xyz)
        colours.append(rgb)
        counts.append(dict(pair=[first, second], points=len(xyz)))
    xyz, rgb = np.concatenate(points), np.concatenate(colours)
    if len(xyz):
        # Average estimates in the same voxel to merge repeated surface samples.
        voxel_size = max(float(np.median(np.linalg.norm(sparse - np.median(sparse, axis=0), axis=1))) * 0.005, 1e-5)
        _, inverse = np.unique(np.floor(xyz / voxel_size).astype(np.int64), axis=0, return_inverse=True)
        weights = np.bincount(inverse)
        xyz = np.column_stack([np.bincount(inverse, weights=xyz[:, j]) / weights for j in range(3)])
        rgb = np.column_stack([np.bincount(inverse, weights=rgb[:, j]) / weights for j in range(3)]).astype(np.uint8)
    save_ply(output_path, xyz, rgb)
    result = dict(dense_points=len(xyz), pairs=counts, pixels='all candidates examined in selected pairs; invalid depths rejected')
    output_path.with_suffix('.json').write_text(json.dumps(result, indent=2))
    print(f'Step 6: {len(xyz)} dense points saved to {output_path.name}', flush=True)
    return result
