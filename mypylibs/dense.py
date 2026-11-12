"""Step 6: rectify registered views, match pixels, then merge their 3D estimates."""
import json

import cv2
import numpy as np
from scipy.spatial import cKDTree

from mypylibs.foreground import object_mask
from mypylibs.sfm import read_frames
from mypylibs.triangulation import project, save_ply


def silhouette_votes(points, poses, masks, K):
    """A real object point should project inside its silhouette in other views."""
    votes = np.zeros(len(points), dtype=int)
    for frame, mask in masks.items():
        pixels, depth = project(points, poses[frame], K)
        finite = np.isfinite(pixels).all(axis=1)
        pixels = np.rint(np.nan_to_num(pixels, nan=-1, posinf=-1, neginf=-1)).astype(int)
        x, y = pixels.T
        inside = finite & (depth > 0) & (x >= 0) & (x < mask.shape[1]) & (y >= 0) & (y < mask.shape[0])
        indices = np.flatnonzero(inside)
        votes[indices] += mask[y[inside], x[inside]] != 0
    return votes


def stereo_points(first, second, pose1, pose2, K, roi, first_mask=None, second_mask=None):
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
    mask1 = cv2.remap(source_mask if first_mask is None else first_mask, *map1, cv2.INTER_NEAREST) != 0
    mask2 = cv2.remap(source_mask if second_mask is None else second_mask, *map2, cv2.INTER_NEAREST) != 0
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
    intensity = grey1.astype(np.float32)
    variance = cv2.blur(intensity ** 2, (5, 5)) - cv2.blur(intensity, (5, 5)) ** 2
    valid = (inside & mask1 & mask2[y, target_x] & (d1 > minimum) & (abs(d1) >= 1.5) & (variance > 4) &
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
    sparse = np.asarray(list(cloud.values()))
    # Spread at most 20 dense stereo pairs over the recovered trajectory.
    indices = np.unique(np.linspace(0, len(registered) - 2, min(20, len(registered) - 1), dtype=int))
    pairs = []
    for i in indices:
        first, second = registered[i], registered[i + 1]
        centre1 = -poses[first][:, :3].T @ poses[first][:, 3]
        ray1 = sparse - centre1
        for candidate in registered[i + 1:i + 5]:
            centre2 = -poses[candidate][:, :3].T @ poses[candidate][:, 3]
            ray2 = sparse - centre2
            cosine = np.sum(ray1 * ray2, axis=1) / np.maximum(np.linalg.norm(ray1, axis=1) *
                                                                           np.linalg.norm(ray2, axis=1), 1e-12)
            angle = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
            second = candidate
            if np.median(angle) >= 1.0:
                break
        pairs.append((first, second))
    # Also load neighbouring views for independent silhouette checks.
    neighbours = {first: registered[max(0, registered.index(first) - 2):registered.index(first) + 4]
                  for first, _ in pairs}
    check_frames = [registered[i] for i in np.unique(np.linspace(0, len(registered) - 1,
                                                               min(10, len(registered)), dtype=int))]
    needed = ({f for group in neighbours.values() for f in group} |
              {f for pair in pairs for f in pair} | set(check_frames))
    images = {i: frame for i, frame in enumerate(read_frames(video_path, K, D, undistort)) if i in needed}
    masks = {i: object_mask(image, config) for i, image in images.items()}
    points, colours, counts = [], [], []
    for first, second in pairs:
        # Erosion removes mixed foreground/background pixels at silhouette edges.
        m1, m2 = [cv2.erode(masks[i], np.ones((3, 3), np.uint8)) for i in (first, second)]
        xyz, rgb = stereo_points(images[first], images[second], poses[first], poses[second],
                                 K, config.get('roi', [0, 0, 1, 1]), m1, m2)
        before = len(xyz)
        votes = silhouette_votes(xyz, poses, {f: masks[f] for f in neighbours[first]}, K)
        supported = votes >= min(3, len(neighbours[first]))
        xyz, rgb = xyz[supported], rgb[supported]
        # Bound distant stereo noise using the sparse map's depth in this camera.
        depth = (sparse @ poses[first][:, :3].T + poses[first][:, 3])[:, 2]
        positive = depth[depth > 0]
        if len(positive):
            dense_depth = (xyz @ poses[first][:, :3].T + poses[first][:, 3])[:, 2]
            keep = (dense_depth > np.median(positive) * 0.1) & (dense_depth < np.median(positive) * 10)
            xyz, rgb = xyz[keep], rgb[keep]
        points.append(xyz)
        colours.append(rgb)
        counts.append(dict(pair=[first, second], before_silhouette_check=before, points=len(xyz)))
    xyz, rgb = np.concatenate(points), np.concatenate(colours)
    before_global = len(xyz)
    check_masks = {f: masks[f] for f in check_frames}
    votes = silhouette_votes(xyz, poses, check_masks, K)
    # Require three spread-out views, rather than a majority of the entire path:
    # conservative segmentation can omit pins and surfaces in some views.
    global_support = votes >= min(3, len(check_frames))
    xyz, rgb = xyz[global_support], rgb[global_support]
    after_global = len(xyz)
    if len(xyz):
        # Average estimates in the same voxel to merge repeated surface samples.
        voxel_size = max(float(np.median(np.linalg.norm(sparse - np.median(sparse, axis=0), axis=1))) * 0.005, 1e-5)
        _, inverse = np.unique(np.floor(xyz / voxel_size).astype(np.int64), axis=0, return_inverse=True)
        weights = np.bincount(inverse)
        xyz = np.column_stack([np.bincount(inverse, weights=xyz[:, j]) / weights for j in range(3)])
        rgb = np.column_stack([np.bincount(inverse, weights=rgb[:, j]) / weights for j in range(3)]).astype(np.uint8)
        # Isolated samples are usually bad disparities, rather than a supported surface.
        if len(xyz) >= 10:
            distances = cKDTree(xyz).query(xyz, k=8)[0][:, 1:].mean(axis=1)
            median = np.median(distances)
            deviation = np.median(abs(distances - median))
            keep = distances <= median + 3 * max(1.4826 * deviation, voxel_size)
            xyz, rgb = xyz[keep], rgb[keep]
    save_ply(output_path, xyz, rgb)
    result = dict(dense_points=len(xyz), pairs=counts,
                  global_check_views=check_frames, before_global_check=before_global,
                  after_global_check=after_global,
                  pixels='all candidates examined; foreground masks, multi-view silhouette support and isolated-point rejection applied')
    output_path.with_suffix('.json').write_text(json.dumps(result, indent=2))
    print(f'Step 6: {len(xyz)} dense points saved to {output_path.name}', flush=True)
    return result
