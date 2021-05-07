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
    n_class_members : int
            Amount of class representatives to have.
    coordinate_tensor : tf.Tensor
            Tensor of coordinates to use in the function evaluation
    """

    def __init__(self, n_class_members: int = None, coordinates=None):
        """
        Constructor for the Potential parent class.

        Parameters
        ----------
        coordinates : tf.Tensor
            Tensor of coordinates to use in the function evaluation
        """
        self.bin_values = {}
        self.data_tensor = None
        self.coordinate_tensor = coordinates
        self.n_class_members = n_class_members
        self._check_input()

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

        Parameters
        ----------
        coordinate_tensor : tf.Tensor
                coordinate tensor on which to apply a function.
        """
        raise NotImplementedError("Implemented in child class")  # Raise error if this class method is called directly

    def plot_potential(self):
        """
        Plot the potential.
        """
        raise NotImplementedError("Implemented in child class")  # Raise error if this class method is called directly