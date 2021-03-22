""" Class to extract the generators of symmetry groups from data """

import numpy as np
import random

from sklearn.decomposition import PCA


class GeneratorExtract:
    """ Generate extraction class

    Attributes
    ----------
    data_clusters : numpy ndarray
                        Data to be analyzed
    basis : list
                        Basis set for the analysis
    basis_data : list
                        A copy of the data_clusters array used when building the orthogonal basis. Data from this array
                        is removed during the analysis and thus a copy was made.
    hyperplane_points : list
                        A list of tuples corresponding to points within the dataset which lie within the hyperplane
    point_pairs : list
                        A list of tuple pairs containing points connected by the single action of a generator
    """

    def __init__(self, data_clusters, delta=0.5, epsilon=0.3):
        """ Python constructor

        Parameters
        ----------
        data_clusters : numpy ndarray
                            Data to be analyzed
        """

        self.data_clusters = data_clusters  # clusters of data connected by a symmetry group
        self.basis_data = list(np.copy(data_clusters))
        self.delta = delta
        self.epsilon = epsilon

        self.basis = []
        self.hyperplane_points = []
        self.point_pairs = []
        self.generators = []

    @staticmethod
    def _perform_PCA(data):
        """
        Perform PCA on the data to remove redundant data
        """

        pca = PCA(n_components=1)
        pca.fit(data)

        print(pca.components_)
        print(pca.explained_variance_ratio_)

    def _gram_schmidt_process(self, vectors):
        """ Implement the gram schmidt process on data """

        for vector in vectors:
            u = vector

            for basis_vector in self.basis:
                u -= self._projection_operator(basis_vector, vector)

            if any(np.isnan(u)):
                continue

            else:
                self.basis.append(u)

    def _normalize_basis_set(self):
        """ Normalize the generated basis set"""

        for i in range(len(self.basis)):
            self.basis[i] = self.basis[i] / np.linalg.norm(self.basis[i])

    @staticmethod
    def _projection_operator(u, v):
        """ Apply projection operator to vectors

        A method to apply the projection operator used in the Gram-Schmidt procedure.

        .. math::

            proj_{\mathbf{u}}(\mathbf{v}) = \frac{\langle\mathbf{u}, \mathbf{v}\rangle}{\langle\mathbf{u}, \mathbf{u} \rangle}\mathbf{u}

        Parameters
        ----------
        u : numpy ndarray
                            The u vector for the projection operation
        v : numpy ndarray
                            The v vector for the projection operation
        """

        return u*(np.dot(u, v) / np.dot(u, u))

    def _start_gs_method(self):
        """ Populate basis set with first two vectors

        Randomly sample the given data and construct the first two vectors for the orthonormal basis set
        """

        index = random.randint(0, len(self.basis_data) - 1)  # generate a random index
        u_1 = self.basis_data[index]  # get the first basis vector
        self.basis.append(u_1)  # assign first vector to basis
        self.basis_data.pop(index)  # remove the data point from the list

        index = random.randint(0, len(self.basis_data) - 1)  # generate a new random index
        v_2 = self.data_clusters[index]  # select data at that index
        u_2 = v_2 - self._projection_operator(u_1, v_2)  # calculate the orthogonal vector
        self.basis.append(u_2)  # add vector to the basis set
        self.basis_data.pop(index)  # remove the data point from the data

    def _construct_standard_basis(self):
        """
        Construct a standard basis dependant on the size of the point cloud.
        """
        zero_vector = np.zeros((len(self.data_clusters), len(self.data_clusters)))  # build the zero matrix

        for i, vector in enumerate(zero_vector):
            vector[i] = 1

        return zero_vector

    def _eliminate_closest_vectors(self, data: np.array):
        """
        Find and remove the vectors in data closest to the basis vectors of the class.
        """

        size_array = []  # instantiate empty array for the sizes of the data

        for vector in data:
            distance = np.linalg.norm

    def _build_orthonormal_basis(self):
        """ Construct and orthonormal basis

        Use the Gram-Schmidt procedure to build an orthonormal basis set out of the given data. The first two points are
        built manually by choosing points at random. The rest constructed in a loop automatically from these chosen
        basis vectors.
        """

        self._start_gs_method()  # get the first two vectors in the basis set
        tst_data = [[1, 0], [0, 1]]
        self._gram_schmidt_process(tst_data)  # perform the Gram-Schmidt process over the rest of the array
        self._normalize_basis_set()  # normalize the vectors in the orthogonal basis

    def _find_hyperplane_points(self):
        """ Get all points within the hyperplane thickness """

        if len(self.basis) == 2:
            self.hyperplane_points = self.data_clusters
        else:
            for point in self.data_clusters:
                for basis_vector in self.basis[2:]:
                    if abs(np.dot(point, basis_vector)) < self.delta:
                        self.hyperplane_points.append(point)
                    else:
                        continue

    def _find_point_pairs(self):
        """ Find all pairs connected by a generator """

        for origin in self.hyperplane_points:
            for vector in self.hyperplane_points:
                # check for origin = vector
                if all(origin == vector):
                    continue
                distance = np.linalg.norm(origin - vector)  # calculate the distance
                if distance < self.epsilon:
                    self.point_pairs.append([origin, vector])

    def _sigma_function(self, points):
        """ Get the directional information from points on hyperplane """

        return np.sign(((np.dot(points[0], self.basis[0])) * (np.dot(points[1], self.basis[1]))) -
                       ((np.dot(points[0], self.basis[1])) * (np.dot(points[1], self.basis[0]))))

    def _solve_for_generator(self, pair, reference):
        """ Perform regression task on generator equation

        Parameters
        ----------
        pair : list
                            The list of points connected by the generator in the point cloud
        reference : list
                            A basis vector to compare with the point from the point cloud
        """

        # get the information about the point
        point_1 = pair[0]
        point_2 = pair[1]
        distance = point_2 - point_1
        distance_magnitude = np.linalg.norm(distance)
        normalization = np.linalg.norm(point_1)
        sigma = self._sigma_function(pair)

        lhs = (normalization * distance) / (sigma * distance_magnitude)

        regression_tensor_1 = [lhs[0], 0]
        regression_tensor_2 = [lhs[1], 0]
        coefficient_tensor = [point_1, reference]

        first_row = np.linalg.inv(coefficient_tensor).dot(regression_tensor_1)
        second_row = np.linalg.inv(coefficient_tensor).dot(regression_tensor_2)

        return [first_row, second_row]

    def _evaluate_generator(self):
        """ Calculate the generator for the given data """

        temporary_generators = []
        # for points in self.point_pairs:
        #     for basis in self.basis[2:]:
        #         generator = self._solve_for_generator(points, basis)
        #         check = generator * basis
        #         for basis_check in self.basis[2:]:
        #             check += generator * basis_check
        #         condition = []
        #         for row in check:
        #             if all(v == 0 for v in row):
        #                 condition.append(True)
        #         if all(condition):
        #             temporary_generators.append(generator)
        #         else:
        #             continue

        for points in self.point_pairs:
            point_1 = points[0]
            point_2 = points[1]
            distance = point_2 - point_1
            distance_magnitude = np.linalg.norm(distance)
            normalization = np.linalg.norm(point_1)
            sigma = self._sigma_function(points)
            lhs = (normalization * distance) / (sigma * distance_magnitude)
            dat = np.linalg.lstsq(lhs, point_1)
            print(dat)
            temporary_generators.append(dat)

        if len(temporary_generators) > 2:
            self.generators.append(temporary_generators)

    def _clear_class_state(self):
        """ Reset the class back to its original state """

        self.basis_data = list(np.copy(self.data_clusters))

        self.basis = []
        self.hyperplane_points = []
        self.point_pairs = []

    def extract_generators(self):
        """ Collect all functions to extract the generators

        Steps are performed as follows:

        1.) Run PCA to determine if any points are not required
        2.) Construct orthonormal basis from remaining points
        3.) Filter points close enough to hyperplane spanned by b1 and b2 to be accessible by a generator action
        4.) Collect point pairs capable of being connected by a generator action
        5.) Perform regression to identify all possible generator candidates
        6.) Perform PCA to identify relevant generators
        """

        counter = 0
        while counter < 10:
            # self._perform_PCA()              # 1.) Perform pca on the input data
            self._build_orthonormal_basis()  # 2.) Build the orthonormal basis
            self._find_hyperplane_points()  # 3.) Fill the hyperplane points array
            self._find_point_pairs()  # 4.) Find all connected points
            self._evaluate_generator()  # 5.) Evaluate generators
            self._clear_class_state()  # Clear class state for next run
            counter += 1

        outcome = []
        for item in self.generators:
            for generator in item:
                outcome.append(generator)
        self._perform_PCA(np.reshape(outcome, (len(outcome), -1)))
