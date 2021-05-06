"""
Author: Samuel Tovey
Contact: stovey@icp.uni-stuttgart.de ; tovey.samuel@gmail.com
Affiliation: Institute for Computational Physics, University of Stuttgart, Stuttgart, Germany

Description: A parent class for the construction and sampling of an arbitray potential
"""

import numpy as np
import tensorflow as tf


class Potential:
    """ Parent class for potentials

    Attributes
    ----------
    k_range : list
            Bin width with which to sample the data, e.g. [-5, 5]
    bin_operation : list
            operations to apply to k values to get bins from multiplication to addition,
            e.g. [1/5, 1e-3] leads to bins of [k/5 - 1e-3, k/5 + 1e-3]
    n_class_members : int
            Amount of class representatives to have.
    coordinates : tf.Tensor
            Tensor of coordinates to use in the function evaluation
    """

    def __init__(self, k_range, bin_operation, n_class_members=None, coordinates=None):
        """
        Constructor for the Potential parent class.
        Parameters
        ----------
        k_range : list
            Bin width with which to sample the data, e.g. [-5, 5]
        bin_operation : list
            operations to apply to k values to get bins from multiplication to addition,
            e.g. [1/5, 1e-3] leads to bins of [k/5 - 1e-3, k/5 + 1e-3]
        n_class_members : int
            Amount of class representatives to have.
        coordinates : tf.Tensor
            Tensor of coordinates to use in the function evaluation
        """
        self.k_range = k_range
        self.bin_values = {}
        self.data_tensor = None
        self.n_class_members = n_class_members
        self.coordinate_tensor = coordinates
        self._check_input()

        for k in np.linspace(k_range[0], k_range[1], 11, dtype=int):
            self.bin_values[f"{k + abs(k_range[0])}"] = [bin_operation[0] * k - bin_operation[1],
                                                         bin_operation[0] * k + bin_operation[1]]

    def _check_input(self):
        """
        Check the input and set defaults in required.

        Returns
        -------
        Updates the class directly.
        """
        if self.n_class_members is None:
            self.n_class_members = 1000
        if self.coordinate_tensor is None:
            self.coordinate_tensor = tf.random.uniform([self.n_class_members * 1000, 2],
                                                       minval=0.0,
                                                       maxval=1.6)

    def _build_condlist(self, x):
        """
        Build the condition list for the piecewise implementation.
        """

        conditions = []
        classes = []
        for key in self.bin_values:
            conditions.append(np.logical_and(x >= (self.bin_values[key][0]), x <= (self.bin_values[key][1])))
            classes.append(key)

        return conditions, classes

    @staticmethod
    def _count_bins(data):
        """
        Count how many members are in a representative class.
        """
        summed_array = []
        for cls in data:
            summed_array.append(np.sum(cls))

        return summed_array

    def get_function_values(self, coordinates):
        """
        Evaluate the function at a point.
        """
        raise NotImplementedError("Implemented in child class")  # Raise error if this class method is called directly

    def plot_potential(self):
        """
        Plot the potential.
        """
        raise NotImplementedError("Implemented in child class")  # Raise error if this class method is called directly

    def _function_to_bins(self, function_values):
        """
        Sort function values into bins.
        """

        conditions, functions = self._build_condlist(function_values)

        return conditions

    def _reduce_bins(self, data, members=1000):
        """
        Reduce the bins from n members to m members.

        Parameters
        ----------
        data : np.array
                Data to reduce.

        members : int
                Number of entries in each bin.
        """
        pass

    def build_dataset(self):
        """
        Call all methods and build the dataset
        """
        raise NotImplementedError("Implemented in child class")  # Raise error if this class method is called directly

    @staticmethod
    def _cluster_data(predictions, training_data):
        """
        Cluster data by the norm of the potential at the minimum
        """

        correlated_classes = np.zeros(len(predictions))
        counter = 0

        truth_data = training_data[:, 1]
        radii = training_data[:, 0]
        categorical_truth_data = tf.keras.utils.to_categorical(truth_data)
        for i in range(len(predictions)):
            if np.linalg.norm(predictions[i] - categorical_truth_data[i]) <= 2e-1:
                correlated_classes[counter] = i
                counter += 1

        correlated_classes = correlated_classes[0:counter]

        colour_classes = np.zeros((len(correlated_classes)), dtype=int)

        visualization_data = np.zeros((len(correlated_classes), 1))

        for i in range(len(correlated_classes)):
            visualization_data[i] = np.copy(radii[int(correlated_classes[i])])
            colour_classes[i] = np.copy(truth_data[int(correlated_classes[i])])

        return colour_classes, correlated_classes, visualization_data
