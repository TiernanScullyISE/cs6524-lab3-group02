"""Run with python -m unittest mypylibs.test_geometry."""
import unittest

import cv2
import numpy as np

from mypylibs.sfm import relative_pose
from mypylibs.multiview import bundle_adjust
from mypylibs.triangulation import project, triangulate


class GeometryTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.points = rng.uniform([-2, -1, 4], [2, 1, 9], (150, 3))
        self.K = np.array([[600., 0, 320], [0, 600, 180], [0, 0, 1]])
        self.first = np.column_stack((np.eye(3), np.zeros(3)))
        R = cv2.Rodrigues(np.array([0., 0.03, 0.]))[0]
        self.second = np.column_stack((R, [-0.5, 0, 0]))
        self.p1 = project(self.points, self.first, self.K)[0]
        self.p2 = project(self.points, self.second, self.K)[0]

    def test_recover_known_motion_with_outliers(self):
        cv2.setRNGSeed(42)
        p2 = self.p2.copy()
        p2[:20] = p2[20:40]
        result = relative_pose(self.p1, p2, self.K)
        self.assertIsNotNone(result)
        _, R, t, keep = result
        self.assertGreater(keep.sum(), 120)
        np.testing.assert_allclose(R, self.second[:, :3], atol=1e-4)
        self.assertGreater(np.dot(t, [-1, 0, 0]), 0.99)

    def test_no_motion_is_rejected(self):
        self.assertIsNone(relative_pose(self.p1, self.p1, self.K, min_motion=0.4))

    def test_known_structure_and_reprojection(self):
        points, keep, errors = triangulate(self.p1, self.p2, self.first, self.second, self.K)
        self.assertTrue(keep.all())
        np.testing.assert_allclose(points, self.points, atol=1e-6)
        self.assertLess(errors.max(), 1e-6)

    def test_parallel_rays_are_rejected(self):
        parallel_camera = np.column_stack((np.eye(3), [-0.5, 0, 0]))
        _, keep, _ = triangulate(self.p1, self.p1, self.first, parallel_camera, self.K)
        self.assertFalse(keep.any())

    def test_five_view_adjustment_reduces_error_and_keeps_anchors(self):
        rng = np.random.default_rng(8)
        views, poses = [], {}
        for i in range(5):
            pose = np.column_stack((np.eye(3), [-0.15 * i, 0., 0.]))
            pixels = project(self.points, pose, self.K)[0] + rng.normal(0, 0.15, (150, 2))
            views.append(dict(ids=np.arange(150), points=pixels))
            poses[i] = pose.copy()
            if i >= 2:
                poses[i][:, 3] += [0.01, 0.01, 0.005]
        anchors = [poses[i].copy() for i in range(2)]
        cloud = dict(enumerate(self.points + rng.normal(0, 0.01, self.points.shape)))
        result = bundle_adjust(views, poses, cloud, self.K)
        self.assertTrue(result['accepted'])
        self.assertLess(result['after_rmse_px'], result['before_rmse_px'])
        for i in range(2):
            np.testing.assert_array_equal(poses[i], anchors[i])


if __name__ == '__main__':
    unittest.main()
