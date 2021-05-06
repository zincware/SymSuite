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
    coordinate_tensor : tf.Tensor
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

    def get_function_values(self, coordinate_tensor: tf.Tensor):
        """
        Evaluate the function at a point.
        """
        raise NotImplementedError("Implemented in child class")  # Raise error if this class method is called directly

    def plot_potential(self):
        """
        Plot the potential.
        """
        raise NotImplementedError("Implemented in child class")  # Raise error if this class method is called directly