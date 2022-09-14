"""
This program and the accompanying materials are made available under the terms
of the Eclipse Public License v2.0 which accompanies this distribution, and is
available at https://www.eclipse.org/legal/epl-v20.html
SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the Zincware Project.

Description: Parent class for the models.
"""
from typing import Any, Callable, Union

import jax.numpy as jnp
from jax.random import PRNGKeyArray


class Model:
    """
    Parent class for ZnRND Models.

    Attributes
    ----------
    model : Callable
            A callable class or function that takes a feature
            vector and returns something from it. Typically this is a
            neural network layer stack.
    """

    model: Callable

    def init_model(
        self,
        init_rng: Union[Any, PRNGKeyArray] = None,
        kernel_init: Callable = None,
        bias_init: Callable = None,
    ):
        """
        Initialize a model.

        Parameters
        ----------
        init_rng : Union[Any, PRNGKeyArray]
                Initial rng for train state that is immediately deleted.
        kernel_init : Callable
                Define the kernel initialization.
        bias_init : Callable
                Define the bias initialization.
        """
        raise NotImplementedError("Implemented in child class.")

    def train_model(
        self,
        train_ds: dict,
        test_ds: dict,
        epochs: int = 10,
        batch_size: int = 1,
        disable_loading_bar: bool = False,
    ):
        """
        Train the model on data.

        Parameters
        ----------
        train_ds : dict
                Train dataset with inputs and targets.
        test_ds : dict
                Test dataset with inputs and targets.
        epochs : int
                Number of epochs to train over.
        batch_size : int
                Size of the batch to use in training.
        disable_loading_bar : bool
                Disable the output visualization of the loading par.
        """
        raise NotImplementedError("Implemented in child class.")

    def compute_ntk(
        self,
        x_i: jnp.ndarray,
        x_j: jnp.ndarray = None,
        normalize: bool = True,
        infinite: bool = False,
    ):
        """
        Compute the NTK matrix for the model.

        Parameters
        ----------
        x_i : jnp.ndarray
                Dataset for which to compute the NTK matrix.
        x_j : jnp.ndarray (optional)
                Dataset for which to compute the NTK matrix.
        normalize : bool (default = True)
                If true, divide each row by its max value.
        infinite : bool (default = False)
                If true, compute the infinite width limit as well.

        Returns
        -------
        NTK : dict
                The NTK matrix for both the empirical and infinite width computation.
        """
        raise NotImplementedError("Implemented in child class")

    def __call__(self, feature_vector: jnp.ndarray):
        """
        Call the network.

        Parameters
        ----------
        feature_vector : jnp.ndarray
                Feature vector on which to apply operation.

        Returns
        -------
        output of the model.
        """
        self.model(feature_vector)
