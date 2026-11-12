import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import cv2
import numpy as np

from mypylibs.calibration_check import audit_calibration
from mypylibs.dense import silhouette_votes, stereo_points
from mypylibs.foreground import object_mask
from mypylibs.multiview import prune_map, reconstruct
from mypylibs.triangulation import project


class OptimisationTests(unittest.TestCase):
    def test_calibration_checks_do_not_claim_camera_transfer(self):
        root = Path(__file__).resolve().parent.parent
        with TemporaryDirectory() as folder:
            result = audit_calibration(root / 'in', Path(folder) / 'audit.json')
        self.assertTrue(result['numerically_valid'])
        self.assertFalse(result['calibration_transfer_validated'])

    def test_original_folding_distortion_is_rejected(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'a.yaml').write_text('calibration_size: [854, 480]\n')
            np.savetxt(root / 'K.csv', [[800.3305, 0, 422.8597], [0, 800.115, 224.9106], [0, 0, 1]], delimiter=',')
            np.savetxt(root / 'D.csv', [0.264184, -0.995237, -0.010203, -0.004020, 0], delimiter=',')
            with self.assertRaises(ValueError):
                audit_calibration(root, root / 'audit.json')
            self.assertFalse(json.loads((root / 'audit.json').read_text())['numerically_valid'])

    def test_foreground_mask_excludes_surrounding_background(self):
        cv2.setRNGSeed(42)
        rng = np.random.default_rng(42)
        image = rng.integers(80, 150, (360, 640, 3), dtype=np.uint8)
        image[130:240, 280:350] = 240
        mask = object_mask(image, dict(object_roi=[0.30, 0.20, 0.65, 0.85]))
        self.assertEqual(mask[180, 310], 255)
        self.assertEqual(mask[180, 220], 0)
        self.assertLess(np.count_nonzero(mask), 15000)

    def test_background_only_pixels_produce_no_dense_points(self):
        rng = np.random.default_rng(42)
        first = rng.integers(0, 256, (180, 320, 3), dtype=np.uint8)
        second = np.roll(first, -10, axis=1)
        K = np.array([[300., 0, 160], [0, 300, 90], [0, 0, 1]])
        origin = np.column_stack((np.eye(3), np.zeros(3)))
        pose = np.column_stack((np.eye(3), [-0.2, 0, 0]))
        empty = np.zeros(first.shape[:2], np.uint8)
        points, _ = stereo_points(first, second, origin, pose, K, [0, 0, 1, 1], empty, empty)
        self.assertEqual(len(points), 0)

    def test_map_rejects_inconsistent_and_two_view_points(self):
        K = np.array([[600., 0, 320], [0, 600, 180], [0, 0, 1]])
        points = np.array([[0., 0., 5.], [0.1, 0.1, 6.], [-0.1, 0.2, 7.]])
        poses, views = {}, []
        for i in range(4):
            poses[i] = np.column_stack((np.eye(3), [-0.1 * i, 0., 0.]))
            ids = np.arange(3) if i < 2 else np.array([0, 2])
            views.append(dict(ids=ids, points=project(points[ids], poses[i], K)[0]))
        cloud = dict(enumerate(points.copy()))
        cloud[0] += [1, 0, 0]
        kept = prune_map(views, poses, cloud, K)
        self.assertEqual(set(kept), {2})

    def test_silhouette_votes_reject_background_and_behind_camera(self):
        K = np.array([[600., 0, 320], [0, 600, 180], [0, 0, 1]])
        pose = np.column_stack((np.eye(3), np.zeros(3)))
        mask = np.zeros((360, 640), np.uint8)
        mask[150:210, 290:350] = 255
        votes = silhouette_votes(np.array([[0., 0, 5], [1., 0, 5], [0., 0, -5]]),
                                  {0: pose, 1: pose, 2: pose}, {i: mask for i in range(3)}, K)
        np.testing.assert_array_equal(votes, [3, 0, 0])

    def test_backtracking_recovers_opening_frames_and_anchors_frame_zero(self):
        cv2.setRNGSeed(42)
        rng = np.random.default_rng(42)
        points = rng.uniform([-2, -1, 4], [2, 1, 9], (150, 3))
        descriptors = rng.integers(0, 256, (150, 32), dtype=np.uint8)
        K = np.array([[600., 0, 320], [0, 600, 180], [0, 0, 1]])
        views = []
        for i in range(9):
            pose = np.column_stack((np.eye(3), [-0.1 * max(0, i - 1), 0., 0.]))
            # IDs were lost before frame two, but descriptors can reconnect them.
            ids = np.arange(150) + (150 if i < 2 else 0)
            views.append(dict(ids=ids, points=project(points, pose, K)[0].astype(np.float32),
                              descriptors=descriptors.copy()))
        with TemporaryDirectory() as folder:
            poses, cloud = reconstruct(views, K, dict(keyframe_step=1, min_motion=0.1), Path(folder))
        self.assertIn(0, poses)
        self.assertIn(1, poses)
        self.assertIn(8, poses)
        np.testing.assert_allclose(poses[0], np.column_stack((np.eye(3), np.zeros(3))), atol=1e-6)
        self.assertGreater(len(cloud), 100)


if __name__ == '__main__':
    unittest.main()
