import unittest
import numpy as np
from bettermdptools.envs.cartpole_model import DiscretizedCartPole


class TestDiscretizedCartPole(unittest.TestCase):
    def setUp(self):
        self.cartpole = DiscretizedCartPole(
            position_bins=10,
            velocity_bins=10,
            angular_velocity_bins=10,
            angular_center_resolution=0.1,
            angular_outer_resolution=0.5,
        )

    def test_adaptive_angle_bins(self):
        # Test with default angle range
        angle_range = (-12 * np.pi / 180, 12 * np.pi / 180)
        bins = self.cartpole.adaptive_angle_bins(angle_range, 0.1, 0.005)

        # Check output type
        self.assertIsInstance(bins, np.ndarray, "Bins should be a numpy array")

        # Check monotonically increasing
        self.assertTrue(np.all(np.diff(bins) > 0), "Bins should be strictly increasing")

        # Check range boundaries
        self.assertAlmostEqual(
            bins[0],
            angle_range[0],
            places=6,
            msg="First bin should match min angle",
        )
        self.assertAlmostEqual(
            bins[-1],
            angle_range[1],
            places=6,
            msg="Last bin should match max angle",
        )

        # Check for uniqueness
        self.assertEqual(
            len(np.unique(bins)), len(bins), "Bins should have no duplicates"
        )


if __name__ == "__main__":
    unittest.main()
