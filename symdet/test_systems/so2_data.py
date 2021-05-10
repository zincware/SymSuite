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
"""

"""
Class to generate data for the SO(2) group
"""

import numpy as np
import matplotlib.pyplot as plt


class SO2:
    """
    Class for generating SO(2) data.

    Attributes
    ----------
    n_points : int
            Number of points to be generated.
    radius : float
            Radius of the circle on which to generate points.
    noise : bool
            Parameter to decide if noise should be added to the generated signal.
    variance : float
            Variance in the noise.
    theta : np.ndarray
            theta values as a numpy array
    data : np.ndarray
            x, y coordinate data of points on the circle. shape=(n_points, 2)

    """

    def __init__(self, n_points: int = 100, radius: float = 1, noise: bool = False, variance: float = 0.05):
        """
        Constructor for the SO(2) data.

        Parameters
        ----------
        n_points : int
                Number of points to be generated.
        radius : float
                Radius of the circle on which to generate points.
        noise : bool
                Parameter to decide if noise should be added to the generated signal.
        variance : float
                Variance in the noise.
        """
        self.n_points = n_points
        self.radius = radius
        self.noise = noise
        self.variance = variance

        self.theta = None
        self.data = None

    def generate_data(self):
        """
        Generate data on a circle

        Returns
        -------
        Updates the class state
        """

        if self.noise:
            self.radius = np.random.uniform(self.radius - self.variance, self.radius + self.variance, self.n_points)

        self.theta = np.random.rand(self.n_points) * (np.pi * 2)
        # generate x, y samples
        x = self.radius*np.cos(self.theta)
        y = self.radius*np.sin(self.theta)

        self.data = np.array(list(zip(x, y)))

    def plot_data(self, save: bool = False):
        """
        Plot the data

        Returns
        -------
        plots and saves an image.
        """

        if self.data is None:
            self.generate_data()  # generate the data

        plt.plot(self.data[:, 0], self.data[:, 1], 'k.')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.xlim(-1.5, 1.5)
        plt.ylim(-1.5, 1.5)
        plt.axis('equal')
        if save:
            plt.savefig(f'SO(2)_{self.n_points}.svg', dpi=800, format='svg')
        plt.show()

