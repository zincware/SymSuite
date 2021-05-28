"""
This file is part of the SymDet distribution (https://github.com/SamTov/SymDet).
Copyright (c) 2021 Samuel Tovey.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, version 3.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>.

Generators
==========
Python module to extract generators from data.
"""
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import random
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from typing import Tuple


class GeneratorExtraction:
    """
    Class to extract generators from point clouds

    Attributes
    ----------
    point_cloud : tf.Tensor
            Point cloud on which to perform regression.
    delta : float
                Width of the hyperplane to be considered.
    epsilon : float
                Distance between points to be considered as a point pair.
    candidate_runs : int
                Number of times to generate candidates before running PCA. Tune if convergence is not found.
    basis : tf.Tensor
            Orthonormal basis of the point cloud.
    hyperplane_set : tf.Tensor
            Set of all points in the hyperplane.
    point_pairs : list
            A list of tuples where each tuple value is an index of a point. The tuple indices correspond to point pairs.
            i.e. if the tuple is (0, 400), then indices 0 and 400 in the hyperplane set are a pair connected by the
            application/s of the generators.
    dimension : int
            Dimensionality of the points.
    generator_candidates : list
            Generator candidates on which to perform PCA.
    constrained_generators : np.ndarray:
            Constrained generator candidates.
    """

    def __init__(
        self,
        point_cloud: tf.Tensor,
        delta: float = 0.5,
        epsilon: float = 0.3,
        candidate_runs: int = 10,
    ):
        """
        Constructor for the GeneratorExtraction class.

        Parameters
        ----------
        point_cloud : tf.Tensor
                Point cloud on which to perform regression
        delta : float
                Width of the hyperplane to be considered
        epsilon : float
                Distance between points to be considered as a point pair.
        candidate_runs : int
                Number of times to generate candidates before running PCA. Tune if convergence is not found.
        """
        self.point_cloud = point_cloud
        self.delta = delta
        self.epsilon = epsilon
        self.candidate_runs = candidate_runs

        self.basis: tf.Tensor
        self.hyperplane_set: tf.Tensor
        self.point_pairs: list
        self.dimension = self._get_dimension()

        self.generator_candidates = []
        self.constrained_generators = []

    def _get_dimension(self):
        """
        Get the dimension of the points in the point cloud.

        Returns
        -------
        dimension : int
                Dimension of the points in the point cloud.
        """
        return len(self.point_cloud[0])

    def _remove_redundancy(self):
        """
        Remove the redundancy in the data with PCA.

        Returns
        -------
        Updates the class state.
        Notes
        -----
        Currently not implemented as it is not required.
        """
        pass

    def _generate_basis_set(self):
        """
        Build the basis set.

        Returns
        -------
        Updates the basis set and enforces a check.
        """
        basis = list(self._start_gs())

        if self.dimension > 2:
            basis_candidates = np.zeros((self.dimension, self.dimension))
            for i, vector in enumerate(basis_candidates):
                vector[i] = 1
            reduced_candidates = self._eliminate_closest_vector(
                basis, list(basis_candidates)
            )
            for item in reduced_candidates:
                basis.append(self._perform_gs(item, basis))

        self.basis = tf.convert_to_tensor(basis)  # set the class attribute
        self._gs_check()

    def _gs_check(self):
        """
        Check to see that all basis vectors are orthogonal to one another.

        If this assert fails, the session will end.

        Returns
        -------
        Will throw an exception if the assert fails.
        """
        for basis in self.basis:
            np.testing.assert_almost_equal(np.linalg.norm(basis), 1)
            for test in self.basis:
                if all(test == basis):
                    continue
                np.testing.assert_almost_equal(np.dot(basis, test), 0)

    def _perform_gs(self, vector: list, basis_set: list) -> np.ndarray:
        """
        Perform the Gram-Schmidt orthogonalization procedure.

        Parameters
        ----------
        vector : list
                Vector to orthonormalize
        basis_set : list
                Current basis set
        Returns
        -------
        basis_vector : list
                Orthonormalized basis vector
        """
        basis_vector = vector
        for basis_item in basis_set:
            basis_vector -= self._projection_operator(basis_item, basis_vector)

        return basis_vector / np.linalg.norm(basis_vector)

    def _eliminate_closest_vector(
        self, reference_vectors: list, test_vectors: list
    ) -> np.ndarray:
        """
        Remove the closest vectors in the theoretical basis set

        Parameters
        ----------
        reference_vectors : list
                First two basis vectors
        test_vectors : list
                Vectors to test against
        Returns
        -------
        basis_vectors : np.ndarray
                Returns the basis vectors which minimizes the sum of squares of the scalar product.
        """
        distances = []
        for vector in test_vectors:
            d_1 = np.dot(vector, reference_vectors[0])
            d_2 = np.dot(vector, reference_vectors[1])
            distances.append(d_1 ** 2 + d_2 ** 2)

        return np.array(test_vectors)[np.argsort(distances)][: int(self.dimension - 2)]

    def _start_gs(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pick the two first random basis vectors.

        Returns
        -------
        basis_vectors : list
                first two randomly selected, normalized basis vectors.
        """
        index_1 = random.randint(0, len(self.point_cloud) - 1)
        index_2 = random.randint(0, len(self.point_cloud) - 1)

        vector_1 = self.point_cloud[index_1] / np.linalg.norm(self.point_cloud[index_1])
        vector_2 = self.point_cloud[index_2] - self._projection_operator(
            vector_1, self.point_cloud[index_2]
        )
        vector_2 /= np.linalg.norm(vector_2)

        return vector_1, vector_2

    @staticmethod
    def _projection_operator(u, v) -> np.ndarray:
        """
        Perform the projection of u onto v.

        Returns
        -------
        proj_{u}(v) : np.ndarray
                Returns the projection of u on v. This is required for the Gram-Schmidt process.
        """
        return (np.dot(u, v) / np.dot(u, u)) * u

    def _construct_hyperplane_set(self):
        """
        Build the hyperplane set.

        Returns
        -------
        Updates the class state
        """
        if self.dimension == 2:
            self.hyperplane_set = self.point_cloud
        else:
            self.hyperplane_set = []
            for point in self.point_cloud:
                truth_table = []
                for vector in self.basis[2:]:
                    truth_table.append(np.dot(point, vector) < self.delta)

                if all(truth_table):
                    self.hyperplane_set.append(point)
        self.hyperplane_set = tf.convert_to_tensor(self.hyperplane_set)

    def _identify_point_pairs(self):
        """
        Identify point pairs within the hyperplane set.

        Returns
        -------
        Updates the class state.
        """
        self.point_pairs = []
        for index, origin in enumerate(self.hyperplane_set):
            for pair_index, reference in enumerate(self.hyperplane_set):
                # Skip the same point
                if pair_index == index:
                    continue
                else:
                    distance = np.linalg.norm(origin - reference)
                    if distance < self.epsilon:
                        self.point_pairs.append((index, pair_index))

    def _perform_regression(self):
        """
        Perform regression on the point pairs and extract generator candidates.

        Returns
        -------
        Updates the class state.
        """
        self._simple_regression()

        if self.dimension > 2:
            self._constrain_generators()

    def _constrain_generators(self):
        """
        If the dimension is greater than 2, the generators should be constrained by Equation 15 in the referenced paper.

        Returns
        -------
        Updates the class.
        """
        for generator in self.generator_candidates:
            outcome = []
            for basis in self.basis[2:]:
                data = np.matmul(
                    generator.reshape(self.dimension, self.dimension), basis
                )
                outcome = np.concatenate((outcome, data))

            if np.allclose(outcome, [0 for _ in range(len(outcome))], atol=1):
                self.constrained_generators.append(generator)

    def _full_regression(self):
        """
        Perform full regression to extract the generators.

        Returns
        -------
        Updates the class state.
        """
        self._simple_regression()
        unconstrained_generators = np.array(self.generator_candidates).reshape(
            (len(self.generator_candidates), self.dimension, self.dimension)
        )
        self.generator_candidates = []
        for generator in unconstrained_generators:
            truth_array = []
            for vector in self.basis[2:]:
                test = sum(generator * vector)
                truth_array.append(all(abs(test) < 1))
            if all(truth_array):
                self.generator_candidates.append(
                    np.array(generator).reshape((1, self.dimension * self.dimension))
                )

    def _simple_regression(self):
        """
        In the case where additional constraints are not needed, we simply perform regression on the problem to
        extract generator candidates.

        Returns
        -------
        Updates the class state.
        """
        Y = []
        X = []
        for pair in self.point_pairs:
            points = [self.hyperplane_set[pair[0]], self.hyperplane_set[pair[1]]]
            sigma = self._compute_sigma(points)
            Y.append(
                ((points[0] - points[1]) * np.linalg.norm(points[0]))
                / (sigma * np.linalg.norm(points[1] - points[0]))
            )
            X.append(points[0])

        generator = []
        for i in range(self.dimension):
            generator = np.concatenate(
                (generator, LinearRegression().fit(X, np.array(Y)[:, i]).coef_)
            )

        self.generator_candidates.append(generator)

    def _compute_sigma(self, pair) -> int:
        """
        Compute the directional information about the point pair.

        Parameters
        ----------
        pair : list
                A point pair from the point cloud.
        Returns
        -------
        sigma : int
                A direction measurement used to identify direction information in the basis set.
        """
        return np.sign(
            (np.dot(pair[0], self.basis[0]) * np.dot(pair[1], self.basis[1]))
            - (np.dot(pair[0], self.basis[1]) * np.dot(pair[1], self.basis[0]))
        )

    def _extract_generators(self, pca_components: int) -> Tuple:
        """
        Perform PCA on candidates and extract true generators.

        Parameters
        ----------
        pca_components : int
                Number of pca components to use.

        Returns
        -------
        pca_components : list
                pca components.
        variance : list
                The explained variance list.
        """
        if self.dimension > 2:
            pca = PCA(n_components=pca_components)
            pca.fit(self.constrained_generators)

        pca = PCA(n_components=pca_components)
        pca.fit(self.generator_candidates)

        return np.sqrt(self.dimension) * pca.components_, pca.explained_variance_ratio_

    def _plot_results(self, std_values, save: bool = False):
        """
        Plot the results of the analysis.

        Parameters
        ----------
        std_values : list
                explained variance list to be plotted.
        save : bool
                If true, the plot will be saved.

        Returns
        -------
        Plots and image and saves it if required.
        """
        plt.plot([i for i in range(len(std_values))], std_values, "o-")
        plt.xlabel("No. PCA Components")
        plt.ylabel("Explained Variance (%)")
        plt.xticks(np.arange(len(std_values)), [i + 1 for i in range(len(std_values))])
        plt.xlim(-0.1, len(std_values) - 0.9)
        plt.ylim(-0.1, 1.1)
        if save:
            plt.savefig("PCA_STD.svg", dpi=800, format="svg")
        plt.show()

    def perform_generator_extraction(
        self, pca_components: int = 4, plot: bool = False, save: bool = False
    ) -> Tuple:
        """
        Collect all methods and perform the generator extraction.

        Parameters
        ----------
        pca_components : int
                Number of pca components to checked in the reduction.
        plot : bool
                If True, the outcomes will be plotted.
        save : bool
                If True, and plot is also True, the plots will be saved.

        Returns
        -------
        generators : list
                Return a list of generators.
        std_array : list
                explained variance list.
        """

        for _ in tqdm(
            range(self.candidate_runs), ncols=100, desc="Producing generator candidates"
        ):
            self._remove_redundancy()
            self._generate_basis_set()
            self._construct_hyperplane_set()
            self._identify_point_pairs()
            self._perform_regression()

        generators, std_array = self._extract_generators(pca_components=pca_components)
        for i, item in enumerate(generators):
            print(f"Principle Component {i + 1}: Explained Variance: {std_array[i]}")
            print(item.reshape((self.dimension, self.dimension)))
            print("\n")

        if plot:
            self._plot_results(std_array, save=save)

        return generators, std_array
