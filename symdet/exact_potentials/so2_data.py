""" Class to generate data for the SO(2) group """

import numpy as np
import matplotlib.pyplot as plt

class SO2:
    """ Class or generating SO(2) data """

    def __init__(self, n_points=100, radius=1, noise=False, variance: float = 0.05):
        """ Python constructor

        Parameters
        ----------
        n_points : int
                            Number of points to be generated
        radius : float
                            Radius of the circle on which to generate points.
        noise : bool
                            Parameter to decide if noise should be added to the generated signal
        """
        self.n_points = n_points  # number of points on the circle
        self.radius = radius  # radius of the circle
        self.noise = noise
        self.variance = variance

        self.theta = None  # angles
        self.data = None  # points on the circle

    def generate_data(self):
        """ Generate data on a circle"""

        if self.noise:
            self.radius = np.random.uniform(self.radius - self.variance, self.radius + self.variance, self.n_points)

        self.theta = np.random.rand(self.n_points) * (np.pi * 2)  # generate angles randomly
        # generate x, y samples
        x = self.radius*np.cos(self.theta)
        y = self.radius*np.sin(self.theta)

        self.data = np.array(list(zip(x, y)))

    def plot_data(self):
        """ Plot the data """

        if self.data is None:
            self.generate_data()  # generate the data

        plt.plot(self.data[:, 0], self.data[:, 1], 'k.')
        plt.xlim(-1.5, 1.5)
        plt.ylim(-1.5, 1.5)
        plt.axis('equal')
        plt.show()
