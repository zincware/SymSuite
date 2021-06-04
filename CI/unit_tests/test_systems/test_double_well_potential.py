"""
Test module for the double well potential module.
"""
import unittest
from symdet.test_systems.double_well_potential import DoubleWellPotential
import tensorflow as tf
import numpy as np


class TestDoubleWell(unittest.TestCase):
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
        cls.model = DoubleWellPotential()

    def test_get_function_values(self):
        """
        Test the get function values method.
        Returns
        -------

        """
        coordinates = tf.convert_to_tensor([[0.0, 0.0], [1.0, 1.0], [3.0, 3.0]])
        reference_values = np.array(
            [[0.0, 0.0], [1.4142135, -0.59999967], [4.2426405, 282.59995]]
        )
        function_values = self.model.get_function_values(coordinates)
        np.testing.assert_almost_equal(reference_values, function_values, 5)


if __name__ == "__main__":
    unittest.main()
