"""
Python module to extract generators from data.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import random
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA


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
    """

    def __init__(self, point_cloud: tf.Tensor, delta: float = 0.5, epsilon: float = 0.3, candidate_runs: int = 10):
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

        """
        basis = list(self._start_gs())

        basis_candidates = np.zeros((self.dimension, self.dimension))
        if self.dimension > 2:
            for i, vector in enumerate(basis_candidates):
                vector[i] = 1

            reduced_candidates = self._eliminate_closest_vector(basis, list(basis_candidates))

            for item in reduced_candidates:
                basis.append(self._perform_gs(item, basis))

        self.basis = tf.convert_to_tensor(basis)  # set the class attribute

    def _perform_gs(self, vector: list, basis_set: list):
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

    def _eliminate_closest_vector(self, reference_vectors: list, test_vectors: list):
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

        """
        inv_distances = []
        for vector in test_vectors:
            d_1 = np.linalg.norm(vector - reference_vectors[0])
            d_2 = np.linalg.norm(vector - reference_vectors[1])
            inv_distances.append(np.mean([1 / d_1, 1 / d_2]))

        return np.array(test_vectors)[np.argsort(inv_distances)[:int(self.dimension - 2)]]

    def _start_gs(self):
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
        vector_2 = self.point_cloud[index_2] - self._projection_operator(vector_1, self.point_cloud[index_2])
        vector_2 /= np.linalg.norm(vector_2)

        return vector_1, vector_2

    @staticmethod
    def _projection_operator(u, v):
        """
        Perform the projection of u onto v.

        Returns
        -------

        """
        return (np.dot(u, v) / np.dot(u, u)) * u

    def _construct_hyperplane_set(self):
        """
        Build the hyperplane set
        Returns
        -------

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
        Identify point pairs within the hyperplane set
        Returns
        -------

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

        """
        if self.dimension == 2:
            self._simple_regression()
        else:
            self._full_regression()

    def _full_regression(self):
        """
        Perform full regression to extract the generators.
        Returns
        -------

        """
        self._simple_regression()
        print(self.generator_candidates)
        unconstrained_generators = np.array(self.generator_candidates).reshape((len(self.generator_candidates),
                                                                      self.dimension,
                                                                      self.dimension))
        self.generator_candidates = []
        for generator in unconstrained_generators:
            truth_array = []
            for vector in self.basis[2:]:
                test = sum(generator*vector)
                truth_array.append(all(abs(test) < 1))
            if all(truth_array):
                self.generator_candidates.append(np.array(generator).reshape((1, self.dimension*self.dimension)))

    def _simple_regression(self):
        """
        In the case where additional constraints are not needed, we simply perform regression on the problem to
        extract generator candidates.

        Returns
        -------

        """
        Y = []
        X = []
        for pair in self.point_pairs:
            points = [self.hyperplane_set[pair[0]], self.hyperplane_set[pair[1]]]
            sigma = self._compute_sigma(points)
            Y.append(((points[0] - points[1])*np.linalg.norm(points[0]))/(sigma*np.linalg.norm(points[1] - points[0])))
            X.append(points[0])

        top_row = LinearRegression().fit(X, np.array(Y)[:, 0]).coef_
        middle_row = LinearRegression().fit(X, np.array(Y)[:, 1]).coef_
        bottom_row = LinearRegression().fit(X, np.array(Y)[:, 2]).coef_
        self.generator_candidates.append(np.concatenate((top_row, middle_row, bottom_row)))

    def _compute_sigma(self, pair):
        """
        compute the directional information about the point pair
        Returns
        -------

        """
        return np.sign((np.dot(pair[0], self.basis[0])*np.dot(pair[1], self.basis[1])) -
                       (np.dot(pair[0], self.basis[1])*np.dot(pair[1], self.basis[0])))

    def _extract_generators(self):
        """
        Perform PCA on candidates and extract true generators.
        Returns
        -------

        """
        print(self.generator_candidates)
        pca = PCA(n_components=4)
        pca.fit(self.generator_candidates)

        print(np.sqrt(2)*pca.components_)
        print(pca.explained_variance_ratio_)

    def perform_generator_extraction(self):
        """
        Collect all methods and perform the generator extraction.
        Returns
        -------

        """

        for _ in tqdm(range(self.candidate_runs), ncols=100, desc="Producing generator candidates"):
            self._remove_redundancy()
            self._generate_basis_set()
            self._construct_hyperplane_set()
            self._identify_point_pairs()
            self._perform_regression()

        self._extract_generators()
