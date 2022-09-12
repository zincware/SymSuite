"""
Python module to show generator extraction of SO(3) Lie algebra generators.
"""

from symsuite.generators.generators import GeneratorExtraction
from symsuite.test_systems.so3_data import SO3


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

    sphere = SO3(n_points=1000, noise=True, variance=0.05)
    sphere.plot_data()

    generator_extractor = GeneratorExtraction(
        sphere.data,  # clustered data
        delta=0.5,  # distance of points to hyperplane
        epsilon=0.3,  # distance between points connected by a generator
        candidate_runs=9,
    )  # Number of times to run the extraction loop
    generators, variance_list = generator_extractor.perform_generator_extraction(
        pca_components=9, plot=True
    )


if __name__ == "__main__":
    generator_extraction()
