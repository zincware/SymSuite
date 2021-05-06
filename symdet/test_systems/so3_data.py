""" Class to generate data for the SO(2) group """

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class SO3:
    """
    Class to generate data for the SO3 group i.e. points on a sphere.

    Attributes
    ----------
    n_points : int
                Number of points to generate.
    radius : float
                Radius of the sphere.
    noise : bool
                If true, noise is added to the data in the form of a fluctuating radius.
    variance : float
                If noise is added, the variance of the noise to add.
    data : np.array
            cartesian data for use in the regression.
    """
    def __init__(self, n_points: int = 100, radius: float = 1.0, noise: bool = False, variance: float = 0.05):
        """
        Constructor for the SO(3) data class.

        Parameters
        ----------
        n_points : int
                Number of points to generate.
        radius : float
                Radius of the sphere.
        noise : bool
                If true, noise is added to the data in the form of a fluctuating radius.
        variance : float
                If noise is added, the variance of the noise to add.
        """
        self.n_points = n_points
        self.radius = radius
        self.noise = noise
        self.variance = variance

        self.data = None

    def generate_data(self):
        """
        Generate points on the sphere.

        Returns
        -------

        """
        if self.noise:
            self.radius = np.random.uniform(self.radius - self.variance, self.radius + self.variance, self.n_points)

        theta = np.random.rand(self.n_points) * (np.pi * 2)  # generate angles randomly
        phi = np.random.rand(self.n_points) * (np.pi)  # generate angles randomly

        x = self.radius*np.cos(theta)*np.sin(phi)
        y = self.radius*np.sin(theta)*np.sin(phi)
        z = self.radius*np.cos(phi)

        self.data = np.array(list(zip(x, y, z)))

    def plot_data(self):
        """
        Plot the sphere data.
        Returns
        -------

        """
        if self.data is None:
            self.generate_data()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.data[:, 0], self.data[:, 1], self.data[:, 2], marker='.', color='k')
        plt.show()
