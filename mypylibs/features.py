import cv2
import numpy as np


METHOD = 'ORB'  # Updated after the measured SIFT/ORB comparison.


def feature_tools(method, nfeatures=2000):
    """Binary ORB descriptors need Hamming; floating-point SIFT needs L2."""
    if method == 'ORB':
        return cv2.ORB_create(nfeatures=nfeatures), cv2.BFMatcher(cv2.NORM_HAMMING)
    if method == 'SIFT':
        return cv2.SIFT_create(nfeatures=nfeatures), cv2.BFMatcher(cv2.NORM_L2)
    raise ValueError(f'Unknown feature method: {method}')


def match_points(matcher, previous, current, ratio=0.75):
    """Reject ambiguous matches and allow only one match per current point."""
    if previous is None or current is None or len(current) < 2:
        return []
    neighbours = matcher.knnMatch(previous, current, k=2)
    candidates = [pair[0] for pair in neighbours
                  if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance]
    matches, used = [], set()
    for match in sorted(candidates, key=lambda match: match.distance):
        if match.trainIdx not in used:
            used.add(match.trainIdx)
            matches.append(match)
    return matches


def track_features(video_path, output_dir, nfeatures=2000, ratio=0.75, method=METHOD):
    """Match points in consecutive frames, keeping IDs while points survive.

    More features gives denser tracks; a lower ratio rejects more ambiguity.
    Save aligned arrays: frame_pairs (zero-based), track_ids, points1, points2.
    Coordinates are (x, y) pixels in the original frames, ready for Step 3.

    1. Detect points: ORB finds up to 2,000 distinctive points per frame, such as corners and textured areas.
    2. Describe them: ORB gives each point a binary pattern describing its surroundings.
    3. Compare consecutive frames: Hamming distance counts how many bits differ between patterns. 
    Fewer differences suggest a better match.
    4. Reject uncertain matches: the best match must have less than 75% of the second-best match’s distance. 
    Each current point can only be matched once.
    5. Keep track IDs: matched points inherit their previous ID; new points receive new IDs.

    """
    output_dir.mkdir(parents=True, exist_ok=True)
    video = cv2.VideoCapture(str(video_path))
    detector, matcher = feature_tools(method, nfeatures)
    label = method.lower()
    previous_descriptors = None
    next_id = 0
    frame_number = 0
    rows = []
    best_count = 0
    preview = None
    writer = None
    try:
        while True:
            ok, frame = video.read()
            if not ok:
                break
            if writer is None:
                height, width = frame.shape[:2]
                writer = cv2.VideoWriter(str(output_dir / f'{video_path.stem}_{label}.mp4'),
                                         cv2.VideoWriter_fourcc(*'mp4v'),
                                         video.get(cv2.CAP_PROP_FPS) or 30, (width, height))
                if not writer.isOpened():
                    raise OSError('Could not create the feature preview video')
            grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            keypoints, descriptors = detector.detectAndCompute(grey, None)
            # Every new point gets an ID; matched points inherit their old ID.
            ids = np.arange(next_id, next_id + len(keypoints))
            next_id += len(keypoints)
            matches = match_points(matcher, previous_descriptors, descriptors, ratio)
            for match in matches:
                ids[match.trainIdx] = previous_ids[match.queryIdx]
                x1, y1 = previous_keypoints[match.queryIdx].pt
                x2, y2 = keypoints[match.trainIdx].pt
                rows.append((frame_number - 1, ids[match.trainIdx], x1, y1, x2, y2))
            if len(matches) > best_count:
                best_count = len(matches)
                # Limit the drawing to 80 matches so it remains readable.
                preview = cv2.drawMatches(previous_grey, previous_keypoints, grey,
                                          keypoints, matches[:80], None,
                                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                preview_frames = (frame_number - 1, frame_number)
            # Orange points are new; green points matched the previous frame.
            display = cv2.drawKeypoints(frame, keypoints, None, color=(0, 165, 255))
            tracked = [keypoints[match.trainIdx] for match in matches]
            display = cv2.drawKeypoints(display, tracked, None, color=(0, 255, 0))
            cv2.rectangle(display, (0, 0), (width, 60), (0, 0, 0), -1)
            cv2.putText(display, 'Green: matched | Orange: new', (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.putText(display, f'Frame {frame_number} | Matches: {len(matches)}', (8, 47),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            writer.write(display)
            previous_grey, previous_keypoints = grey, keypoints
            previous_descriptors, previous_ids = descriptors, ids
            frame_number += 1
    finally:
        video.release()
        if writer is not None:
            writer.release()

    if frame_number < 2:
        raise ValueError(f'{video_path}: could not read at least two frames')
    if not rows:
        raise ValueError(f'{video_path}: no {method} matches found')
    data = np.asarray(rows)
    frames = data[:, 0].astype(np.int32)
    track_ids = data[:, 1].astype(np.int64)
    np.savez_compressed(output_dir / f'{video_path.stem}_{label}.npz',
                        frame_pairs=np.column_stack((frames, frames + 1)),
                        track_ids=track_ids,
                        points1=data[:, 2:4].astype(np.float32),
                        points2=data[:, 4:6].astype(np.float32))
    if not cv2.imwrite(str(output_dir / f'{video_path.stem}_{label}_matches.jpg'), preview):
        raise OSError('Could not save the feature match preview')
    _, lengths = np.unique(track_ids, return_counts=True)
    counts = np.bincount(frames, minlength=frame_number - 1)
    print(f'{video_path.name}: {frame_number} frames, {len(rows)} matches, '
          f'{len(lengths)} tracks; matches/pair min/mean/max = '
          f'{counts.min()}/{counts.mean():.1f}/{counts.max()}; '
          f'longest track = {lengths.max() + 1} frames; '
          f'preview frames = {preview_frames}')


def track_orb(video_path, output_dir, nfeatures=2000, ratio=0.75):
    """Keep the original ORB entry point available."""
    return track_features(video_path, output_dir, nfeatures, ratio, method='ORB')
