"""
Python module to show generator extraction of SO(2) Lie algebra generators.
"""

from symsuite.test_systems.so2_data import SO2
from symsuite.generators.generators import GeneratorExtraction
import numpy as np


def generator_extraction():
    """
    Function to extract the SO(2) Lie algebra generators from points on a circle
    given small amounts of noise.

    The steps involved for SymDet to perform the extraction are
    1.) Generate some data
    2.) We will plot this data for reference.
    3.) Define the generator extraction class and some parameters.
    4.) Perform the generator extraction.
    """

    circle = SO2(n_points=50, noise=True, variance=0.05)
    circle.plot_data()

    generator_extractor = GeneratorExtraction(
        circle.data,  # clustered data
        delta=0.5,  # distance of points to hyperplane
        epsilon=0.3,  # distance between points connected by a generator
        candidate_runs=5,
    )  # Number of times to run the extraction loop
    generators, variance_list = generator_extractor.perform_generator_extraction(
        pca_components=4, plot=True
    )
    print(generators[0])


if __name__ == "__main__":
    generator_extraction()
