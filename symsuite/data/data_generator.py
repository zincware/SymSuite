"""
This program and the accompanying materials are made available under the terms of the
Eclipse Public License v2.0 which accompanies this distribution, and is available at
https://www.eclipse.org/legal/epl-v20.html
SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the Zincware Project.

Description: Parent class for the data generator. Data should be passed to the group detection and generator extraction
routines through a data generator child class.
"""
import abc
from typing import Union

import jax.numpy as jnp
import numpy as np


class DataGenerator(metaclass=abc.ABCMeta):
    """
    A class to generate data for use in the Symmetry analysis.

    Attributes
    ----------
    domain : tf.Tensor
            Domain values of the function.
    image : tf.Tensor
            Image values of the function, i.e. f(x) for all x.
    image_size : int
            Size of the data pool.
    domain_shape : tuple
            Shape of the domain points.
    clustered_data : dict
            A dictionary of clustered data.
    """

    def __init__(self):
        """
        Constructor for the DataGenerator class.
        """
        self.domain = None
        self.image = None
        self.image_size = None
        self.domain_shape = None
        self.clustered_data = None

    def plot_data(self, save: bool = False):
        """
        Plot the data.

        Parameters
        ----------
        save : bool
                If true the figure will be saved.

        Returns
        -------

        """
        raise NotImplementedError

    def plot_clusters(self, save: bool = False):
        """
        Plot the data clusters.

        Parameters
        ----------
        save : bool
                If true the figure will be saved.

        Returns
        -------

        """
        raise NotImplementedError

    def load_data(self, points: Union[int, np.ndarray], save: bool = False):
        """
        Load some data either from a computation or from a pool into the class state.

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

        """
        raise NotImplementedError

    def build_clusters(self, **kwargs):
        """
        Split the raw function data into classes.

        Parameters
        ----------
        **kwargs

        Returns
        -------
        Updates the class state.

        Notes
        -----
        In the double well potential we can simply use the range_binning clustering algorithm.
        """
        raise NotImplementedError
