import unittest

import cv2
import numpy as np

from mypylibs.dense import stereo_points


class DenseTests(unittest.TestCase):
    def test_known_plane_depth_for_horizontal_and_vertical_stereo(self):
        rng = np.random.default_rng(42)
        texture = rng.integers(0, 256, (360, 640), np.uint8)
        first = cv2.cvtColor(texture, cv2.COLOR_GRAY2BGR)
        K = np.array([[600., 0, 320], [0, 600, 180], [0, 0, 1]])
        origin = np.column_stack((np.eye(3), np.zeros(3)))
        for axis in (0, 1):
            with self.subTest(axis=axis):
                second = np.roll(first, -20, axis=1 - axis)
                translation = np.zeros(3)
                translation[axis] = -0.2
                pose = np.column_stack((np.eye(3), translation))
                xyz, rgb = stereo_points(first, second, origin, pose, K, [0, 0, 1, 1])
                self.assertGreater(len(xyz), 10000)
                # Z = focal * baseline / disparity = 600 * .2 / 20 = 6.
                self.assertAlmostEqual(float(np.median(xyz[:, 2])), 6.0, places=2)
                self.assertEqual(xyz.shape, rgb.shape)
                self.assertTrue(np.isfinite(xyz).all())


if __name__ == '__main__':
    unittest.main()
