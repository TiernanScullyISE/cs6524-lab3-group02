"""Steps 3 onwards use world-to-camera poses: camera_point = R @ world_point + t."""
import json

import cv2
import numpy as np
import yaml

from mypylibs.features import feature_tools, match_points


def load_camera(input_dir, factor=1.0):
    config = yaml.safe_load((input_dir / 'a.yaml').read_text())
    K = np.loadtxt(input_dir / 'K.csv', delimiter=',')
    D = np.loadtxt(input_dir / 'D.csv', delimiter=',').ravel()
    if K.shape != (3, 3) or not np.isfinite(K).all() or not np.isfinite(D).all():
        raise ValueError('Invalid calibration CSV files')
    size = config['calibration_size']
    K = np.diag([640 / size[0], 360 / size[1], 1]) @ K
    K[0, 0] *= factor
    K[1, 1] *= factor
    return K, D


def read_frames(video_path, K, D, undistort=True):
    """Rotate portrait frames CCW to calibration orientation, then resize equally."""
    video = cv2.VideoCapture(str(video_path))
    if not video.isOpened():
        raise ValueError(f'Cannot open {video_path}')
    maps = cv2.initUndistortRectifyMap(K, D, None, K, (640, 360), cv2.CV_32FC1)
    try:
        while True:
            ok, frame = video.read()
            if not ok:
                break
            if frame.shape[0] > frame.shape[1]:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            frame = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
            yield cv2.remap(frame, *maps, cv2.INTER_LINEAR) if undistort else frame
    finally:
        video.release()


def extract_views(video_path, K, D, config, method, undistort=True):
    """Keep point coordinates and persistent IDs; discard images/descriptors as we go."""
    detector, matcher = feature_tools(method)
    views, previous, previous_ids = [], None, []
    next_id = 0
    mask = np.zeros((360, 640), np.uint8)
    x1, y1, x2, y2 = config.get('roi', [0, 0, 1, 1])
    mask[int(y1 * 360):int(y2 * 360), int(x1 * 640):int(x2 * 640)] = 255
    for frame in read_frames(video_path, K, D, undistort):
        points, descriptors = detector.detectAndCompute(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), mask)
        ids = np.arange(next_id, next_id + len(points))
        next_id += len(points)
        for match in match_points(matcher, previous, descriptors):
            ids[match.trainIdx] = previous_ids[match.queryIdx]
        views.append(dict(ids=ids, points=np.float32([p.pt for p in points]).reshape(-1, 2),
                          descriptors=descriptors))
        previous, previous_ids = descriptors, ids
    if len(views) < 2:
        raise ValueError('Need at least two readable frames')
    return views


def common_points(first, second):
    ids, i, j = np.intersect1d(first['ids'], second['ids'], return_indices=True)
    return ids, first['points'][i], second['points'][j]


def relative_pose(p1, p2, K, min_motion=0):
    """RANSAC rejects mismatches; cheirality chooses the pose with points in front."""
    moving = np.linalg.norm(p2 - p1, axis=1) >= min_motion
    if moving.sum() < 8:
        return None
    E, mask = cv2.findEssentialMat(p1[moving], p2[moving], K, cv2.RANSAC, 0.999, 1.0)
    if E is None or E.shape != (3, 3) or mask is None:
        return None
    count, R, t, mask = cv2.recoverPose(E, p1[moving], p2[moving], K, mask=mask)
    if count < 8:
        return None
    keep = np.zeros(len(p1), dtype=bool)
    keep[np.flatnonzero(moving)] = mask.ravel() != 0
    return E, R, t.ravel(), keep


def estimate_pairs(views, K, output_dir, min_motion=0):
    """Step 3: attempt E, R, t for EVERY original consecutive frame pair."""
    records, Es, rotations, translations = [], [], [], []
    for index in range(len(views) - 1):
        _, p1, p2 = common_points(views[index], views[index + 1])
        pose = relative_pose(p1, p2, K, min_motion)
        records.append(dict(pair=[index, index + 1], matches=len(p1),
                            inliers=int(pose[3].sum()) if pose else 0,
                            status='accepted' if pose else 'insufficient_geometry'))
        Es.append(pose[0] if pose else np.full((3, 3), np.nan))
        rotations.append(pose[1] if pose else np.full((3, 3), np.nan))
        translations.append(pose[2] if pose else np.full(3, np.nan))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'pairs.json').write_text(json.dumps(records, indent=2))
    np.savez_compressed(output_dir / 'relative_poses.npz', E=Es, R=rotations, t=translations)
    print(f'Step 3: {sum(r["status"] == "accepted" for r in records)}/{len(records)} pairs accepted',
          flush=True)
    return records
