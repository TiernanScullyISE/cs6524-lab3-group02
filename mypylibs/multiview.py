"""Step 5: shared 3D map, PnP camera poses and five-view bundle adjustment."""
import json

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from mypylibs.features import match_points
from mypylibs.sfm import common_points, relative_pose
from mypylibs.triangulation import project, save_ply, triangulate


def bundle_adjust(views, poses, cloud, K, window=5, max_points=250, iterations=15):
    """Optimise a local window (or all views); fix two poses to anchor scale."""
    frames = sorted(poses)[-window:]
    if len(frames) < 5:
        return None
    # Points observed in >=3 window views constrain more than a single pair.
    observations = {}
    for frame in frames:
        for track, pixel in zip(views[frame]['ids'], views[frame]['points']):
            if track in cloud:
                observations.setdefault(int(track), []).append((frame, pixel))
    tracks = sorted((t for t in observations if len(observations[t]) >= 3),
                    key=lambda t: -len(observations[t]))[:max_points]
    if len(tracks) < 10:
        return None
    variable_frames = frames[2:]
    parameters = np.concatenate([np.r_[cv2.Rodrigues(poses[f][:, :3])[0].ravel(), poses[f][:, 3]]
                                 for f in variable_frames] + [np.array([cloud[t] for t in tracks]).ravel()])
    rows = [(f, i, pixel) for i, t in enumerate(tracks) for f, pixel in observations[t]]
    offset = 6 * len(variable_frames)
    groups = []
    for frame in frames:
        indices = np.array([i for i, row in enumerate(rows) if row[0] == frame], dtype=int)
        if len(indices):
            groups.append((frame, indices, [rows[i][1] for i in indices],
                           np.array([rows[i][2] for i in indices])))

    def unpack(values):
        cameras = {f: poses[f] for f in frames[:2]}
        for i, frame in enumerate(variable_frames):
            block = values[6 * i:6 * i + 6]
            cameras[frame] = np.column_stack((cv2.Rodrigues(block[:3])[0], block[3:]))
        return cameras, values[offset:].reshape(-1, 3)

    def residual(values):
        cameras, points = unpack(values)
        errors = np.empty((len(rows), 2))
        for frame, indices, point_indices, observed in groups:
            errors[indices] = project(points[point_indices], cameras[frame], K)[0] - observed
        return errors.ravel()

    sparsity = lil_matrix((2 * len(rows), len(parameters)), dtype=int)
    for i, (frame, point, _) in enumerate(rows):
        sparsity[2 * i:2 * i + 2, offset + 3 * point:offset + 3 * point + 3] = 1
        if frame in variable_frames:
            j = variable_frames.index(frame) * 6
            sparsity[2 * i:2 * i + 2, j:j + 6] = 1
    before = float(np.sqrt(np.mean(residual(parameters) ** 2)))
    fit = least_squares(residual, parameters, jac_sparsity=sparsity.tocsr(),
                        loss='soft_l1', f_scale=1.0, max_nfev=iterations)
    after = float(np.sqrt(np.mean(residual(fit.x) ** 2)))
    accepted = np.isfinite(after) and after <= before
    if accepted:
        cameras, points = unpack(fit.x)
        poses.update(cameras)
        cloud.update(zip(tracks, points))
    return dict(frames=frames, points=len(tracks), before_rmse_px=before,
                after_rmse_px=after, accepted=bool(accepted))


def prune_map(views, poses, cloud, K):
    """Retain points supported by at least three views with small pixel errors."""
    errors = {}
    for frame, pose in poses.items():
        view = views[frame]
        known = [(int(t), p) for t, p in zip(view['ids'], view['points']) if t in cloud]
        if not known:
            continue
        tracks, pixels = zip(*known)
        projected, depth = project(np.array([cloud[t] for t in tracks]), pose, K)
        distances = np.linalg.norm(projected - pixels, axis=1)
        for track, distance, z in zip(tracks, distances, depth):
            errors.setdefault(track, []).append(float(distance) if z > 0 else float('inf'))
    return {t: cloud[t] for t, values in errors.items() if len(values) >= 3 and
            np.median(values) < 1.5 and np.percentile(values, 90) < 3.0}


def reconstruct(views, K, config, output_dir):
    step = config.get('keyframe_step', 5)
    origin = np.column_stack((np.eye(3), np.zeros(3)))
    poses, cloud, corrections = {}, {}, []
    # Bootstrap with sufficient baseline; never join independent unit-scale pairs.
    best = None
    search_frames = config.get('initialisation_frames', 60)
    for first in range(0, min(len(views) - step, search_frames), step):
        if first >= 60 and best is not None:
            break  # Keep a verified opening pair; search later only if needed.
        for second in range(first + step, min(first + 5 * step, len(views)), step):
            ids, p1, p2 = common_points(views[first], views[second])
            pose = relative_pose(p1, p2, K, config.get('min_motion', 0))
            if pose is None:
                continue
            camera = np.column_stack((pose[1], pose[2]))
            points, keep, _ = triangulate(p1, p2, origin, camera, K)
            keep &= pose[3]
            if keep.sum() >= 20:
                candidate = dict(zip(ids[keep], points[keep]))
                support = []
                # A third/fourth view must support the initial depth estimates.
                # This avoids accepting a plausible but unstable two-view pose.
                for check in (second + step, second + 2 * step):
                    if check >= len(views):
                        continue
                    known = [(candidate[t], p) for t, p in
                             zip(views[check]['ids'], views[check]['points']) if t in candidate]
                    if len(known) < 8:
                        break
                    objects, pixels = map(np.asarray, zip(*known))
                    ok, _, _, inliers = cv2.solvePnPRansac(objects, pixels, K, None,
                                                           reprojectionError=2.0, iterationsCount=100)
                    if not ok or inliers is None or len(inliers) < max(8, len(known) * 0.5):
                        break
                    support.append(len(inliers))
                if len(support) == 2:
                    score = min(support)
                    if best is None or score > best[0]:
                        best = (score, first, second, camera, candidate)
    if best is None:
        raise ValueError('No multi-view verified bootstrap; inspect calibration, ROI and texture')
    _, first, second, camera, cloud = best
    poses = {first: origin.copy(), second: camera}
    last = max(poses)
    skipped = []
    order = list(range(last + step, len(views), step)) + list(range(first - step, -1, -step))
    for frame in order:
        # Work backwards too when the best initial pair occurs after frame zero.
        last = min(poses, key=lambda index: abs(index - frame))
        view = views[frame]
        if view['descriptors'] is None:
            skipped.append(frame)
            continue
        # Match directly to the last registered keyframe as well. This can
        # recover a point whose consecutive-frame track was briefly lost.
        norm = cv2.NORM_HAMMING if view['descriptors'].dtype == np.uint8 else cv2.NORM_L2
        matches = match_points(cv2.BFMatcher(norm), views[last]['descriptors'], view['descriptors'])
        claimed = set(view['ids'])
        for match in matches:
            track = views[last]['ids'][match.queryIdx]
            if track in cloud and (track not in claimed or view['ids'][match.trainIdx] == track):
                view['ids'][match.trainIdx] = track
                claimed.add(track)
        known = [(cloud[int(t)], p) for t, p in zip(view['ids'], view['points']) if t in cloud]
        if len(known) < 8:
            skipped.append(frame)
            continue
        objects, pixels = map(np.asarray, zip(*known))
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(objects, pixels, K, None,
                                                   iterationsCount=100, reprojectionError=2.0,
                                                   confidence=0.999, flags=cv2.SOLVEPNP_EPNP)
        if not ok or inliers is None or len(inliers) < 8:
            skipped.append(frame)
            continue
        rvec, tvec = cv2.solvePnPRefineLM(objects[inliers.ravel()], pixels[inliers.ravel()],
                                         K, None, rvec, tvec)
        camera = np.column_stack((cv2.Rodrigues(rvec)[0], tvec.ravel()))
        poses[frame] = camera
        ids, p1, p2 = common_points(views[last], view)
        points, keep, _ = triangulate(p1, p2, poses[last], camera, K)
        for track, point in zip(ids[keep], points[keep]):
            if track not in cloud:
                cloud[int(track)] = point
        last = frame
        if len(poses) % 5 == 0 and frame > first:
            correction = bundle_adjust(views, poses, cloud, K)
            if correction:
                corrections.append(correction)
    # Finish with a common adjustment so local window changes do not leave
    # the rest of the trajectory and old map points inconsistent.
    global_correction = bundle_adjust(views, poses, cloud, K, window=len(poses),
                                      max_points=3000, iterations=30)
    if global_correction:
        global_correction['scope'] = 'global'
        corrections.append(global_correction)
    before_pruning = len(cloud)
    cloud = prune_map(views, poses, cloud, K)
    if len(cloud) < 10:
        raise ValueError('Too few multi-view supported points after map verification')
    # Express the final map in the earliest registered camera's coordinate system.
    anchor = poses[min(poses)].copy()
    for frame, pose in list(poses.items()):
        rotation = pose[:, :3] @ anchor[:, :3].T
        poses[frame] = np.column_stack((rotation, pose[:, 3] - rotation @ anchor[:, 3]))
    cloud = {track: anchor[:, :3] @ point + anchor[:, 3] for track, point in cloud.items()}
    frames = sorted(poses)
    np.savez_compressed(output_dir / 'map.npz', frames=frames, poses=[poses[f] for f in frames],
                        track_ids=list(cloud), points=list(cloud.values()))
    save_ply(output_dir / 'sparse.ply', list(cloud.values()))
    summary = dict(registered_frames=frames, skipped_frames=skipped, points=len(cloud),
                   bundle_adjustments=corrections, rejected_map_points=before_pruning - len(cloud),
                   scale='arbitrary monocular units')
    (output_dir / 'multiview.json').write_text(json.dumps(summary, indent=2))
    print(f'Step 5: {len(poses)} registered views, {len(cloud)} points, '
          f'{len(corrections)} bundle adjustments', flush=True)
    return poses, cloud
