"""
This program and the accompanying materials are made available under the terms of the
Eclipse Public License v2.0 which accompanies this distribution, and is available at
https://www.eclipse.org/legal/epl-v20.html
SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the Zincware Project.

Description: Module for the computation of so3 data
"""
from symdet.data.data_generator import DataGenerator
from typing import Union
import numpy as np
import matplotlib.pyplot as plt


class SO3(DataGenerator):
    """
    Class for the double well potential implementation.

    Attributes
    ----------
    noise : bool
                If true, noise is included in the data generation.
    variance : float
            Variance to use in the noise generation.
    radius : float
            Radius if the circle.
    radial_values : Union[float, list]
            Radial values to use in the data generation.

    See Also
    --------
    symdet.data.data_generator.DataGenerator

    Examples
    --------
    >>> from symdet import DoubleWellPotential
    >>> generator = SO3()
    >>> generator.load_data()
    >>> generator.plot_data()
    """

    def __init__(self, noise: bool = True, variance: float = 0.05, radius: float = 1.0):
        """
        Constructor for the double well potential.

        Parameters
        ----------
        noise : bool
                If true, noise is included in the data generation.
        variance : float
                Variance to use in the noise generation.
        radius : float
                Radius if the circle.
        """
        super().__init__()
        self.noise = noise
        self.variance = variance
        self.radius = radius
        self.radial_values = None

    def _sphere(self, points: int):
        """
        Generate point along a double well potential range.

        Parameters
        ----------
        points: int
                Number of points to use.

        Returns
        -------

        """
        if self.noise:
            self.radial_values = np.random.uniform(self.radius - self.variance,
                                                   self.radius + self.variance,
                                                   points)
        else:
            self.radial_values = self.radius

        theta = np.random.rand(points) * (np.pi * 2)  # generate angles randomly
        phi = np.random.rand(points) * np.pi  # generate angles randomly

        x = self.radial_values * np.cos(theta) * np.sin(phi)
        y = self.radial_values * np.sin(theta) * np.sin(phi)
        z = self.radial_values * np.cos(phi)

        self.domain = np.array(list(zip(x, y, z)))

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
            self._sphere(points)

        # set domain and generate image data.
        else:
            raise ValueError(f"Type {type(points)} is not valid for this data generator, try an integer")

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
            self._sphere(points=100)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(self.domain[:, 0],
                   self.domain[:, 1],
                   self.domain[:, 2],
                   marker=".",
                   color="k")
        if save:
            plt.savefig(f"SO(2)_{len(self.domain)}.svg",
                        dpi=800,
                        format="svg")
        plt.show()

    def build_clusters(self, **kwargs):
        """
        Split the raw function data into classes.

        Returns
        -------
        Updates the class state.

        Notes
        -----
        Not required for this data.
        """
        pass

    def plot_clusters(self, save: bool = False):
        """
        Plot the clusters generated.

        Parameters
        ----------
        save

        Notes
        -----
        Not required for this analysis.

        """
        pass
