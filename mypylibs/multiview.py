"""Step 5: shared 3D map, PnP camera poses and five-view bundle adjustment."""
import json

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from mypylibs.features import match_points
from mypylibs.sfm import common_points, relative_pose
from mypylibs.triangulation import project, save_ply, triangulate


def bundle_adjust(views, poses, cloud, K):
    """Optimise the last five cameras/points; fix two poses to anchor scale."""
    frames = sorted(poses)[-5:]
    if len(frames) < 5:
        return None
    # Points observed in >=3 window views constrain more than a single pair.
    observations = {}
    for frame in frames:
        for track, pixel in zip(views[frame]['ids'], views[frame]['points']):
            if track in cloud:
                observations.setdefault(int(track), []).append((frame, pixel))
    tracks = sorted((t for t in observations if len(observations[t]) >= 3),
                    key=lambda t: -len(observations[t]))[:250]
    if len(tracks) < 10:
        return None
    variable_frames = frames[2:]
    parameters = np.concatenate([np.r_[cv2.Rodrigues(poses[f][:, :3])[0].ravel(), poses[f][:, 3]]
                                 for f in variable_frames] + [np.array([cloud[t] for t in tracks]).ravel()])
    rows = [(f, i, pixel) for i, t in enumerate(tracks) for f, pixel in observations[t]]
    offset = 6 * len(variable_frames)

    def unpack(values):
        cameras = {f: poses[f] for f in frames[:2]}
        for i, frame in enumerate(variable_frames):
            block = values[6 * i:6 * i + 6]
            cameras[frame] = np.column_stack((cv2.Rodrigues(block[:3])[0], block[3:]))
        return cameras, values[offset:].reshape(-1, 3)

    def residual(values):
        cameras, points = unpack(values)
        errors = np.empty((len(rows), 2))
        for frame in frames:
            indices = [i for i, row in enumerate(rows) if row[0] == frame]
            point_indices = [rows[i][1] for i in indices]
            observed = np.array([rows[i][2] for i in indices])
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
                        loss='soft_l1', f_scale=1.0, max_nfev=15)
    after = float(np.sqrt(np.mean(residual(fit.x) ** 2)))
    accepted = np.isfinite(after) and after <= before
    if accepted:
        cameras, points = unpack(fit.x)
        poses.update(cameras)
        cloud.update(zip(tracks, points))
    return dict(frames=frames, points=len(tracks), before_rmse_px=before,
                after_rmse_px=after, accepted=bool(accepted))


def reconstruct(views, K, config, output_dir):
    step = config.get('keyframe_step', 5)
    origin = np.column_stack((np.eye(3), np.zeros(3)))
    poses, cloud, corrections = {}, {}, []
    # Bootstrap with sufficient baseline; never join independent unit-scale pairs.
    for first in range(0, min(len(views) - step, 60), step):
        for second in range(first + step, min(first + 5 * step, len(views)), step):
            ids, p1, p2 = common_points(views[first], views[second])
            pose = relative_pose(p1, p2, K, config.get('min_motion', 0))
            if pose is None:
                continue
            camera = np.column_stack((pose[1], pose[2]))
            points, keep, _ = triangulate(p1, p2, origin, camera, K)
            keep &= pose[3]
            if keep.sum() >= 20:
                poses = {first: origin.copy(), second: camera}
                cloud.update(zip(ids[keep], points[keep]))
                break
        if poses:
            break
    if not poses:
        raise ValueError('No reliable two-view bootstrap; inspect calibration, ROI and texture')
    last = max(poses)
    skipped = []
    for frame in range(last + step, len(views), step):
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
        if len(poses) % 5 == 0:
            correction = bundle_adjust(views, poses, cloud, K)
            if correction:
                corrections.append(correction)
    frames = sorted(poses)
    np.savez_compressed(output_dir / 'map.npz', frames=frames, poses=[poses[f] for f in frames],
                        track_ids=list(cloud), points=list(cloud.values()))
    save_ply(output_dir / 'sparse.ply', list(cloud.values()))
    summary = dict(registered_frames=frames, skipped_frames=skipped, points=len(cloud),
                   bundle_adjustments=corrections, scale='arbitrary monocular units')
    (output_dir / 'multiview.json').write_text(json.dumps(summary, indent=2))
    print(f'Step 5: {len(poses)} registered views, {len(cloud)} points, '
          f'{len(corrections)} five-view corrections', flush=True)
    return poses, cloud
