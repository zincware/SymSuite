"""
Author: Samuel Tovey
Contact: stovey@icp.uni-stuttgart.de ; tovey.samuel@gmail.com
Affiliation: Institute for Computational Physics, University of Stuttgart, Stuttgart, Germany

Description: A class for the construction and sampling of a double well potential.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

from .Potential import Potential

import tensorflow as tf


class Double_Well_Potential(Potential):
    """ Double potential

    A double well potential is characterized by two wells in a two dimensional space. This differs from the more popular
    mexican hat potential by a factor of an imaginary exponent, which implies a 'rim' to the well, and is often best
    to leave to a three dimensional space.

    The double well potential is defined mathematically as

            f(x, y) = -a(x**2 + y**2) + (x**2 + y**2)**2 = -ar**2 + r**4

    args:
        a (float) -- prefactor in the double well calculation
    """

    @staticmethod
    def get_function_values(coordinate_tensor: list) -> list:
        """ Evaluate the double well potential at a point

        args:
            :param coordinates: (list) -- x, y values of the function in a 3 dimensional array
                                          [[x, y], [x, y], [x, y], ... ]

            :return: python list of tensors for each set of coordinates and the function values
        """

        radial_tensor = tf.math.sqrt(tf.math.reduce_sum(coordinate_tensor ** 2, 1))

        function_tensor = -2.3 * radial_tensor ** 2 + radial_tensor ** 4

        return function_tensor.numpy(), radial_tensor.numpy()

    def plot_potential(self):
        """ Plot the double well potential """

        r = np.linspace(-2.3, 2.3, 100)
        function = -1 * 2.3 * r ** 2 + r ** 4

        plt.plot(r, function)
        plt.ylim(-1.5, 1.0)
        plt.show()

    def plot_clusters(self, data, x_val):
        """
        Plot the clustered data on the raw.
        """

        x_data = list(data.keys())
        radii = []

        for key in x_data:
            for function_value in data[key]:
                index = tf.where(function_value, x=x_val , y=data[key])
                radii.append(x_val[index])
        print(radii)

    def build_dataset(self):
        """ Call all methods and build the dataset """

        function_values, radial_values = self.get_function_values(self.coordinate_tensor)  # Get the function values
        bin_masks = self._function_to_bins(function_values)
        bin_count = self._count_bins(bin_masks)
        if all(bin_count) > self.n_class_members is False:
            print("Not enough data, please provide more")
        class_keys = list(self.bin_values.keys())  # Get the keys for the class
        potential_data = {}  # Instantiate dictionary to store the data
        for i in range(len(class_keys)):
            filtered_array = radial_values*bin_masks[i]
            filtered_array = filtered_array[filtered_array != 0]
            filtered_array = np.random.choice(filtered_array, size=self.n_class_members)
            potential_data[class_keys[i]] = filtered_array

        #self.plot_clusters(potential_data, radial_values)
        return potential_data
