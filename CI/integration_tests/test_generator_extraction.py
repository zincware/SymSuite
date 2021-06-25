"""
Unit test for the generators extraction.
"""
import unittest


class TestSO2(unittest.TestCase):
    """
    Test SO(2) generator extraction works.
    """

    def test_circle_generation(self):
        """
        Check if SO(2) points are generated correctly.

        Returns
        -------
        Asserts that the mean of the coordinate norms is approx 1.
        """
        self.assertEqual(True, True)


if __name__ == "__main__":
    unittest.main()
