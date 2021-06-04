"""
Test module for the Data Clustering module.
"""
import unittest
from symdet.symmetry_groups.data_clustering import DataCluster
import numpy as np


class TestDataClustering(unittest.TestCase):
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
        cls.model = DataCluster(data=None)

    def test_count_bins(self):
        """
        Test the count bins method.

        Returns
        -------
        Asserts that the method counts the correct number of values in the
        list.
        """
        test_data = np.array(
            [[1, 2, 3], [1, 2, 3, 4], [1, 2], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]
        )
        reference = [3, 4, 2, 11]
        test_data = self.model._count_bins(test_data)

        assert np.array_equal(test_data, reference)

    def test_build_condlist(self):
        """
        Test the build_condlist method.

        Returns
        -------
        Assert that correct classes and conditions come out.
        """
        pass


if __name__ == "__main__":
    unittest.main()
