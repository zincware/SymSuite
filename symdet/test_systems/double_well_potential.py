"""
A class for the construction and sampling of a double well potential.
"""

import numpy as np
import matplotlib.pyplot as plt
from .potential import Potential
import tensorflow as tf


class Double_Well_Potential(Potential):
    """
    Double potential

    A double well potential is characterized by two wells in a two dimensional space.

    Notes
    -----
    The double well potential is defined mathematically as

    ..math::

            f(x, y) = -a(x^{2} + y^{2}) + (x^{2} + y^{2})^{2} = -ar^{2} + r^{4}

    Attributes
    ----------
    a : float
            prefactor in the double well calculation
    """

    def __init__(self, k_range, bin_operation, n_class_members=None, coordinates=None, a: float = 2.3):
        """
        Constructor for the double well potential data generator.

        Parameters
        ----------
        k_range : list
            Bin width with which to sample the data, e.g. [-5, 5].
        bin_operation : list
            operations to apply to k values to get bins from multiplication to addition,
            e.g. [1/5, 1e-3] leads to bins of [k/5 - 1e-3, k/5 + 1e-3].
        n_class_members : int
            Amount of class representatives to have.
        coordinates : tf.Tensor
            Tensor of coordinates to use in the function evaluation.
        a: float
                Pre-factor in the double well potential.
        """
        super().__init__(k_range, bin_operation, n_class_members, coordinates)
        self.a = a

    def get_function_values(self, coordinate_tensor: tf.Tensor = None) -> np.ndarray:
        """
        Evaluate the double well potential at a point

        Parameters
        ----------
        coordinate_tensor: tf.Tensor
                x, y values of the function in a 3 dimensional array
                [[x, y], [x, y], [x, y], ... ]

        Returns
        --------
        data: tuple
                Tuple of numpy arrays set of coordinates and the function values
        """
        if coordinate_tensor is None:
            coordinate_tensor = self.coordinate_tensor
        radial_tensor = tf.math.sqrt(tf.math.reduce_sum(coordinate_tensor ** 2, 1))
        function_tensor = -self.a * radial_tensor ** 2 + radial_tensor ** 4

        return np.array(list(zip(radial_tensor.numpy(), function_tensor.numpy())))

    def plot_potential(self):
        """
        Plot the double well potential
        """

        r = np.linspace(-self.a, self.a, 100)
        function = -1 * self.a * r ** 2 + r ** 4

        plt.plot(r, function)
        plt.ylim(-1.5, 1.0)
        plt.show()
