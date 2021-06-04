"""
Test module for the so2 data module.
"""
import unittest
from symdet.test_systems.so2_data import SO2
import numpy as np


class TestSO2(unittest.TestCase):
    """
    Class to test the double well potential.
    """

    @classmethod
    def setUpClass(cls) -> None:
        """
        Set up the class for the test.

        Returns
        -------
        Updates the class state.
        """
        cls.model = SO2(n_points=500, noise=True, variance=0.05)
        cls.model.generate_data()

    def test_check_input(self):
        """
        Test the check input method.

        Returns
        -------
        Asserts that the mean of the coordinate norms is approx 1.
        """
        self.assertAlmostEqual(
            float(np.mean(np.linalg.norm(self.model.data, axis=1))), 1, delta=0.01
        )


if __name__ == "__main__":
    unittest.main()
