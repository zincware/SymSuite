"""
This program and the accompanying materials are made available under the terms of the
Eclipse Public License v2.0 which accompanies this distribution, and is available at
https://www.eclipse.org/legal/epl-v20.html
SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the Zincware Project.

Description: Example data generator for the double well potential.
"""
from symdet.data.data_generator import DataGenerator
from symdet.utils.data_clustering import range_binning
from typing import Union
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm


class DoubleWellPotential(DataGenerator):
    """
    Class for the double well potential implementation.

    Attributes
    ----------
    a : float
            Scaling factor in the equation (see below)

    See Also
    --------
    symdet.data.data_generator.DataGenerator

    Examples
    --------
    >>> from symdet import DoubleWellPotential
    >>> generator = DoubleWellPotential()
    >>> generator.load_data()
    >>> generator.plot_data()

    Notes
    -----
    The double well potential is written mathematically:

    .. math:: V = -a \cdot (x^{2} + y^{2}) + (x^{2} + y^{2})^{2}

    We will require the data to be stored as x,y  coordinates in order to facilitate the generator extraction in this
    part of the process.
    """

    def __init__(self, a: float = 2.3):
        """
        Constructor for the double well potential.

        Parameters
        ----------
        a : float
                Scaling factor of the double well potential.
        """
        super().__init__()
        self.a = a

    def _double_well(self):
        """
        Generate point along a double well potential range.
        Returns
        -------

        """
        square_radii = tf.reduce_sum(tf.math.square(self.domain), 1)
        self.image = -self.a * square_radii + tf.square(square_radii)

    def _pick_points(self, n_points: int, min_val: float = 0, max_val: float = 1.6):
        """
        Generate random coordinates in the double well domain.

        Parameters
        ----------
        n_points : int
                Number of points to pick.
        min_val : float
                Minimum value to consider.
        max_val : float
                Maximum value to consider.

        Returns
        -------

        """
        self.domain = tf.random.uniform(shape=(n_points, 2),
                                        minval=min_val,
                                        maxval=max_val)

    def load_data(self, points: Union[int, np.ndarray], save: bool = False):
        """
        Load / generate the data.

        Parameters
        ----------
        points : Union[int, np.ndarray]
                Points to generate, either an np.ndarray or an integer. If an integer, N points will be generated, if
                an array, it will either be treated as input to a function to generate values or those indices will be
                drawn from a pool.
        save : bool
                If true, save the data after generating it.

        Returns
        -------
        Updates the class state.
        """
        # generate domain and image data.
        if type(points) is int:
            self._pick_points(points)
            self._double_well()

        # set domain and generate image data.
        else:
            self.domain = tf.convert_to_tensor(points)
            self._double_well()

    def plot_clusters(self, save: bool = False):
        """
        Plot the clusters generated.

        Parameters
        ----------
        save

        Returns
        -------

        """
        self.plot_data(show=False)

        for i, item in tqdm(enumerate(self.clustered_data), ncols=70, total=len(self.clustered_data)):
            r = tf.norm(self.clustered_data[item]['domain'], axis=1)
            v = self.clustered_data[item]['image']
            plt.plot(r, v, '.', label=f"Class {i}", markersize=15)

        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        plt.show()

    def plot_data(self, save: bool = False, show: bool = True):
        """
        Plot the data.

        Parameters
        ----------
        save : bool
                If true, save the plot.
        show : bool (default=True)
                If true, show the result
        Returns
        -------
        Plots the data.
        """
        if self.domain is None:
            self._pick_points(1000, min_val=0, max_val=1.2)
            self._double_well()
        plt.plot(tf.norm(self.domain, axis=1), self.image, '.')
        plt.xlabel('r')
        plt.ylabel('V')
        plt.xlim(-0.03, 1.7)
        plt.ylim(-1.5, 1.3)
        plt.grid()
        if save:
            plt.savefig(f'Double_Well_{len(self.domain)}.svg', dpi=600, format='dpi')
        if show:
            plt.show()

    def build_clusters(self, value_range: list = None, bin_operation: list = None, representatives=1000):
        """
        Split the raw function data into classes.

        Parameters
        ----------
        representatives : int
            Number of class representatives to have for each bin.
        value_range : list
            The parameters within which to bin e.g.  k in [-5, 5]
        bin_operation : list
            Operation to apply to the bins e.g [1/5, 1e-3] will lead
            to bins of the form [k/5 - 1e-3, k/5 + 1e-3]

        Returns
        -------
        Updates the class state.

        Notes
        -----
        In the double well potential we can simply use the range_binning clustering algorithm.
        """
        # Replace None type parameters.
        if bin_operation is None:
            bin_operation = [1 / 5, 1e-3]
        if value_range is None:
            value_range = [-5, 5]
        n_classes = (value_range[1] - value_range[0]) + 1

        # Generate data if it does not exist
        if self.domain is None:
            self.load_data(n_classes * representatives * 1000)
        if len(self.domain) < n_classes * representatives * 1000:
            print("Loading additional data.")
            self.load_data(n_classes * representatives * 1000)

        self.clustered_data = range_binning(image=self.image,
                                            domain=self.domain,
                                            value_range=value_range,
                                            bin_operation=bin_operation,
                                            representatives=representatives)
