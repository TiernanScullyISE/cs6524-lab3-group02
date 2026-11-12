"""Step 7: repeat the actual pipeline for both scenes and all six settings."""
import csv
import hashlib
import json
import platform

import cv2
import numpy as np
import scipy
import yaml

from mypylibs.dense import dense_cloud
from mypylibs.multiview import reconstruct
from mypylibs.sfm import estimate_pairs, extract_views, load_camera
from mypylibs.triangulation import project, two_view_cloud


def run_experiments(root, method):
    results = []
    for name in ('a', 'b'):
        config = yaml.safe_load((root / 'in' / f'{name}.yaml').read_text())
        for factor in (0.8, 1.0, 1.2):
            for undistort in (True, False):
                cv2.setRNGSeed(42)
                label = f'{name}_{round(factor * 10):02d}{"u" if undistort else "d"}'
                output = root / 'out' / label
                output.mkdir(parents=True, exist_ok=True)
                print(f'\n=== {label}: {method} ===', flush=True)
                K, D = load_camera(root / 'in', factor)
                video = root / 'in' / f'{name}.mp4'
                views = extract_views(video, K, D, config, method, undistort)
                pairs = estimate_pairs(views, K, output, config.get('min_motion', 0))
                two_view_cloud(views, K, output)
                poses, cloud = reconstruct(views, K, config, output)
                dense = dense_cloud(video, K, D, undistort, config, poses, cloud,
                                    root / 'out' / f'{label}.ply')
                errors = []
                for frame, pose in poses.items():
                    observed = [(cloud[int(t)], p) for t, p in
                                zip(views[frame]['ids'], views[frame]['points']) if t in cloud]
                    if observed:
                        points, pixels = map(np.asarray, zip(*observed))
                        predicted, depth = project(points, pose, K)
                        errors.extend(np.linalg.norm(predicted - pixels, axis=1)[depth > 0])
                row = dict(case=label, method=method, focal_factor=factor, undistorted=undistort,
                           frames=len(views), accepted_pairs=sum(r['status'] == 'accepted' for r in pairs),
                           registered_views=len(poses), sparse_points=len(cloud),
                           dense_points=dense['dense_points'],
                           median_reprojection_px=float(np.median(errors)),
                           p90_reprojection_px=float(np.percentile(errors, 90)))
                results.append(row)
                (root / 'out' / 'sensitivity.json').write_text(json.dumps(results, indent=2))
                print(row, flush=True)
    with (root / 'out' / 'sensitivity.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    inputs = [root / 'in' / name for name in ('a.mp4', 'b.mp4', 'K.csv', 'D.csv', 'a.yaml', 'b.yaml')]
    metadata = dict(python=platform.python_version(), opencv=cv2.__version__, numpy=np.__version__,
                    scipy=scipy.__version__, seed=42,
                    calibration=yaml.safe_load((root / 'in' / 'a.yaml').read_text())['calibration_source'],
                    hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs})
    (root / 'out' / 'run_metadata.json').write_text(json.dumps(metadata, indent=2))
    return results
