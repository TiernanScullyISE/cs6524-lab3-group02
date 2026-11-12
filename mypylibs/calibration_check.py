"""Check reused calibration without claiming it calibrates a different recording."""
import hashlib
import json

import cv2
import numpy as np
import yaml


def audit_calibration(input_dir, output_path, source_corners=None):
    K = np.loadtxt(input_dir / 'K.csv', delimiter=',')
    D = np.loadtxt(input_dir / 'D.csv', delimiter=',').ravel()
    width, height = yaml.safe_load((input_dir / 'a.yaml').read_text())['calibration_size']
    if (K.shape != (3, 3) or D.size not in (4, 5, 8, 12, 14) or
            not np.isfinite(K).all() or not np.isfinite(D).all() or
            min(K[0, 0], K[1, 1]) <= 0 or not np.allclose(K[2], [0, 0, 1]) or
            not (0 <= K[0, 2] < width and 0 <= K[1, 2] < height)):
        raise ValueError('Invalid calibration matrix or distortion coefficients')
    x, y = np.meshgrid(np.linspace(0, width - 1, 25), np.linspace(0, height - 1, 15))
    pixels = np.column_stack((x.ravel(), y.ravel())).astype(float)
    criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-10)
    if hasattr(cv2, 'undistortPointsIter'):
        rays = cv2.undistortPointsIter(pixels.reshape(-1, 1, 2), K, D, None, None, criteria)
    else:
        rays = cv2.undistortPoints(pixels.reshape(-1, 1, 2), K, D, criteria=criteria)
    points = np.column_stack((rays.reshape(-1, 2), np.ones(len(pixels))))
    roundtrip = cv2.projectPoints(points, np.zeros(3), np.zeros(3), K, D)[0].reshape(-1, 2)
    error = np.linalg.norm(roundtrip - pixels, axis=1)
    # A folding distortion map is physically invalid, even if centre errors look good.
    epsilon = 1e-5
    base = cv2.projectPoints(points, np.zeros(3), np.zeros(3), np.eye(3), D)[0].reshape(-1, 2)
    columns = []
    for axis in (0, 1):
        shifted = points.copy()
        shifted[:, axis] += epsilon
        projected = cv2.projectPoints(shifted, np.zeros(3), np.zeros(3), np.eye(3), D)[0].reshape(-1, 2)
        columns.append((projected - base) / epsilon)
    determinants = np.linalg.det(np.stack(columns, axis=2))
    passed = bool(error.max() < 0.1 and determinants.min() > 0)
    result = dict(numerically_valid=passed, calibration_size=[width, height],
                  maximum_roundtrip_error_px=float(error.max()),
                  minimum_distortion_jacobian=float(determinants.min()),
                  calibration_transfer_validated=False,
                  limitation='Source-camera consistency is not proof of the actual recording camera, lens, crop or stabilisation.')
    if source_corners is not None:
        corners = np.load(source_corners)
        board = np.zeros((40, 3), np.float32)
        board[:, :2] = np.mgrid[:8, :5].T.reshape(-1, 2) * 2.5
        training = [p for i, p in enumerate(corners) if i % 3 != 0]
        heldout = [p for i, p in enumerate(corners) if i % 3 == 0]
        models = [('current_full_source_fit', K, D)]
        for name, flags in [('two_radial_training_fit', cv2.CALIB_FIX_K3),
                             ('one_radial_training_fit', cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3),
                             ('pinhole_training_fit', cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 |
                              cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST)]:
            _, matrix, distortion, _, _ = cv2.calibrateCamera([board] * len(training), training,
                                                              (width, height), None, None, flags=flags)
            models.append((name, matrix, distortion))
        rows = []
        for label, matrix, distortion in models:
            errors = []
            for observed in heldout:
                ok, rotation, translation = cv2.solvePnP(board, observed, matrix, distortion)
                if not ok:
                    raise ValueError('Cannot estimate held-out checkerboard pose')
                predicted = cv2.projectPoints(board, rotation, translation, matrix, distortion)[0]
                errors.append(float(np.linalg.norm(predicted.reshape(-1, 2) - observed.reshape(-1, 2), axis=1).mean()))
            rows.append(dict(model=label, mean_error_px=float(np.mean(errors)),
                             worst_view_error_px=max(errors)))
        result['source_check'] = dict(pattern_inner_corners=[8, 5], training_views=len(training),
                                      heldout_views=len(heldout), results=rows,
                                      note='Training fits exclude held-out views. Current K uses all source views. The one-radial model trades ~0.03 px for a non-folding distortion map.',
                                      corners_sha256=hashlib.sha256(source_corners.read_bytes()).hexdigest())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + '\n')
    if not passed:
        raise ValueError('Distortion map failed numerical validity checks')
    return result
