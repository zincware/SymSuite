"""
Unit test for the generators extraction.
"""
import unittest
from symdet.test_systems.so2_data import SO2
from symdet.generators.generators import GeneratorExtraction
import numpy as np


class SO2Test(unittest.TestCase):
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
        self.circle = SO2(n_points=500,
                          noise=True,
                          variance=0.05)
        self.circle.generate_data()
        self.assertAlmostEqual(float(np.mean(np.linalg.norm(self.circle.data, axis=1))), 1, delta=0.01)

    def test_generator_extraction(self):
        """
        Test whether the generators extracted from SymDet are correct.

        Returns
        -------
        Asserts the correct generator is found.
        """
        self.circle = SO2(n_points=500,
                          noise=True,
                          variance=0.05)
        self.circle.generate_data()
        generator_extractor = GeneratorExtraction(self.circle.data,  # clustered data
                                                  delta=0.5,  # distance of points to hyperplane
                                                  epsilon=0.3,  # distance between points connected by a generator
                                                  candidate_runs=5)  # Number of times to run the extraction loop
        self.generators, self.variance_list = generator_extractor.perform_generator_extraction(pca_components=4,
                                                                                               plot=False)
        self.assertAlmostEqual(self.variance_list[0], 1, delta=0.0001)
        self.assertAlmostEqual(self.variance_list[1], 0, delta=0.00000001)
        generator = np.sqrt(self.generators[0]**2)
        self.assertAlmostEqual(generator[0], 0.0, delta=0.05)
        self.assertAlmostEqual(generator[1], 1.0, delta=0.1)
        self.assertAlmostEqual(generator[2], 1.0, delta=0.1)
        self.assertAlmostEqual(generator[3], 0.0, delta=0.05)


if __name__ == '__main__':
    unittest.main()
