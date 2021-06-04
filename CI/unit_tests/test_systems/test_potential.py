"""
Test module for the double well potential module.
"""
import unittest
from symdet.test_systems.potential import Potential


class TestPotential(unittest.TestCase):
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
        cls.model = Potential()

    def test_check_input(self):
        """
        Test the check input method.

        Returns
        -------

        """
        self.assertEqual(self.model.n_class_members, 1000)
        self.assertEqual(self.model.coordinate_tensor.shape, (1000000, 2))


if __name__ == "__main__":
    unittest.main()
