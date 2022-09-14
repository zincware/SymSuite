"""
ZnRND: A zincwarecode package.

License
-------
This program and the accompanying materials are made available under the terms
of the Eclipse Public License v2.0 which accompanies this distribution, and is
available at https://www.eclipse.org/legal/epl-v20.html

SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the zincwarecode Project.

Contact Information
-------------------
email: zincwarecode@gmail.com
github: https://github.com/zincware
web: https://zincwarecode.com/

Citation
--------
If you use this module please cite us with:

Summary
-------
Module for the neural tangents infinite width network models.
"""
import logging
from typing import Any, Callable, Union

import jax
import jax.numpy as jnp
import neural_tangents as nt
import numpy as onp
from flax.training import train_state
from jax.random import PRNGKeyArray
from neural_tangents.stax import serial
from tqdm import trange

from symsuite.accuracy_functions.accuracy_function import AccuracyFunction
from symsuite.loss_functions.loss import Loss
from symsuite.ml_models.model import Model
from symsuite.utils import normalize_covariance_matrix

logger = logging.getLogger(__name__)


class NTModel(Model):
    """
    Class for a neural tangents model.
    """

    def __init__(
        self,
        loss_fn: Loss,
        optimizer: Callable,
        input_shape: tuple,
        nt_module: serial = None,
        data_pool: jnp.ndarray = None,
        accuracy_fn: AccuracyFunction = None,
        batch_size: int = 10,
    ):
        """
        Constructor for a Flax model.

        Parameters
        ----------
        loss_fn : SimpleLoss
                A function to use in the loss computation.
        accuracy_fn : AccuracyFunction
                Accuracy function to use for accuracy computation.
        optimizer : Callable
                optimizer to use in the training. OpTax is used by default and
                cross-compatibility is not assured.
        input_shape : tuple
                Shape of the NN input.
        batch_size : int (default=10)
                Batch size to use in the NTK computation.
        nt_module : serial
                NT stax module for training.
        data_pool : jnp.ndarray
                Data pool from which TTV is built.

        """
        self.rng = jax.random.PRNGKey(onp.random.randint(0, 500))
        self.init_fn = nt_module[0]
        self.apply_fn = jax.jit(nt_module[1])
        self.kernel_fn = nt.batch(nt_module[2], batch_size=batch_size)
        self.empirical_ntk = nt.batch(
            nt.empirical_ntk_fn(self.apply_fn), batch_size=batch_size
        )
        self.empirical_ntk_jit = jax.jit(self.empirical_ntk)
        self.loss_fn = loss_fn
        self.accuracy_fn = accuracy_fn
        self.optimizer = optimizer
        self.input_shape = input_shape

        self.data_pool = data_pool

        # initialize the model state
        self.model_state = None
        self.init_model()

    def init_model(
        self,
        init_rng: Union[Any, PRNGKeyArray] = None,
        kernel_init: Callable = None,
        bias_init: Callable = None,
    ):
        """
        Initialize a model.

        If no rng key is given, the key will be produced randomly.

        Parameters
        ----------
        init_rng : Union[Any, PRNGKeyArray]
                Initial rng for train state that is immediately deleted.
        kernel_init : Callable
                Define the kernel initialization.
        bias_init : Callable
                Define the bias initialization.
        """
        if kernel_init:
            raise NotImplementedError(
                "Currently, there is no option customize the weight initialization. "
            )
        if bias_init:
            raise NotImplementedError(
                "Currently, there is no option customize the bias initialization. "
            )
        if init_rng is None:
            init_rng = jax.random.PRNGKey(onp.random.randint(0, 1000000))
        self.model_state = self._create_train_state(init_rng)

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
        x_i : np.ndarray
                Dataset for which to compute the NTK matrix.
        x_j : np.ndarray (optional)
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
        if x_j is None:
            x_j = x_i
        empirical_ntk = self.empirical_ntk_jit(x_i, x_j, self.model_state.params)

        if infinite:
            infinite_ntk = self.kernel_fn(x_i, x_j, "ntk")
        else:
            infinite_ntk = None

        if normalize:
            empirical_ntk = normalize_covariance_matrix(empirical_ntk)
            if infinite:
                infinite_ntk = normalize_covariance_matrix(infinite_ntk)

        return {"empirical": empirical_ntk, "infinite": infinite_ntk}

    def _create_train_state(self, init_rng: Union[Any, PRNGKeyArray]):
        """
        Create a training state of the model.

        Parameters
        ----------
        init_rng : Union[Any, PRNGKeyArray]
                Initial rng for train state that is immediately deleted.

        Returns
        -------
        initial state of model to then be trained.

        Notes
        -----
        TODO: Make the TrainState class passable by the user as it can track custom
              model properties.
        """
        _, params = self.init_fn(init_rng, self.input_shape)

        return train_state.TrainState.create(
            apply_fn=self.apply_fn, params=params, tx=self.optimizer
        )

    def _compute_metrics(
        self,
        predictions: jnp.ndarray,
        targets: jnp.ndarray,
    ):
        """
        Compute the current metrics of the training.

        Parameters
        ----------
        predictions : np.ndarray
                Predictions made by the network.
        targets : np.ndarray
                Targets from the training data.

        Returns
        -------
        metrics : dict
                A dict of current training metrics, e.g. {"loss": ..., "accuracy": ...}
        """
        loss = self.loss_fn(predictions, targets)
        if self.accuracy_fn is not None:
            accuracy = self.accuracy_fn(predictions, targets)
            metrics = {"loss": loss, "accuracy": accuracy}

        else:
            metrics = {"loss": loss}

        return metrics

    def _train_step(self, state: train_state.TrainState, batch: dict):
        """
        Train a single step.

        Parameters
        ----------
        state : TrainState
                Current state of the neural network.
        batch : dict
                Batch of data to train on.

        Returns
        -------
        state : dict
                Updated state of the neural network.
        metrics : dict
                Metrics for the current model.
        """

        def loss_fn(params):
            """
            helper loss computation
            """
            inner_predictions = self.apply_fn(params, batch["inputs"])
            loss = self.loss_fn(inner_predictions, batch["targets"])
            return loss, inner_predictions

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

        (_, predictions), grads = grad_fn(state.params)

        state = state.apply_gradients(grads=grads)  # in place state update.
        metrics = self._compute_metrics(
            predictions=predictions, targets=batch["targets"]
        )

        return state, metrics

    def _evaluate_step(self, params: dict, batch: dict):
        """
        Evaluate the model on test data.

        Parameters
        ----------
        params : dict
                Parameters of the model.
        batch : dict
                Batch of data to test on.

        Returns
        -------
        metrics : dict
                Metrics dict computed on test data.
        """
        predictions = self.apply_fn(params, batch["inputs"])

        return self._compute_metrics(predictions, batch["targets"])

    def _train_epoch(
        self, state: train_state.TrainState, train_ds: dict, batch_size: int
    ):
        """
        Train for a single epoch.

        Performs the following steps:

        * Shuffles the data
        * Runs an optimization step on each batch
        * Computes the metrics for the batch
        * Return an updated optimizer, state, and metrics dictionary.

        Parameters
        ----------
        state : TrainState
                Current state of the model.
        train_ds : dict
                Dataset on which to train.
        batch_size : int
                Size of each batch.

        Returns
        -------
        state : TrainState
                State of the model after the epoch.
        metrics : dict
                Dict of metrics for current state.
        """
        # Some housekeeping variables.
        train_ds_size = len(train_ds["inputs"])
        steps_per_epoch = train_ds_size // batch_size

        if train_ds_size == 1:
            state, metrics = self._train_step(state, train_ds)
            batch_metrics = [metrics]

        else:
            # Prepare the shuffle.
            permutations = jax.random.permutation(self.rng, train_ds_size)
            permutations = permutations[: steps_per_epoch * batch_size]
            permutations = permutations.reshape((steps_per_epoch, batch_size))

            # Step over items in batch.
            batch_metrics = []
            for permutation in permutations:
                batch = {k: v[permutation, ...] for k, v in train_ds.items()}
                # print(batch)
                state, metrics = self._train_step(state, batch)
                batch_metrics.append(metrics)

        # Get the metrics off device for printing.
        batch_metrics_np = jax.device_get(batch_metrics)
        epoch_metrics_np = {
            k: onp.mean([metrics[k] for metrics in batch_metrics_np])
            for k in batch_metrics_np[0]
        }

        return state, epoch_metrics_np

    def _evaluate_model(self, params: dict, test_ds: dict) -> dict:
        """
        Evaluate the model.

        Parameters
        ----------
        params : dict
                Current state of the model.
        test_ds : dict
                Dataset on which to evaluate.
        Returns
        -------
        metrics : dict
                Loss of the model.
        """
        metrics = self._evaluate_step(params, test_ds)
        metrics = jax.device_get(metrics)
        summary = jax.tree_map(lambda x: x.item(), metrics)

        return summary

    def validate_model(self, dataset: dict, loss_fn: SimpleLoss):
        """
        Validate the model on some external data.

        Parameters
        ----------
        loss_fn : SimpleLoss
                Loss function to use in the computation.
        dataset : dict
                Dataset on which to validate the model.
                {"inputs": np.ndarray, "targets": np.ndarray}.

        Returns
        -------
        metrics : dict
                Metrics computed in the validation. {"loss": [], "accuracy": []}.
                Note, for ease of large scale experiments we always return both keywords
                whether they are computed or not.
        """
        predictions = self.apply_fn(self.model_state.params, dataset["inputs"])

        loss = loss_fn(predictions, dataset["targets"])

        if self.accuracy_fn is not None:
            accuracy = self.accuracy_fn(predictions, dataset["targets"])
        else:
            accuracy = None

        return {"loss": loss, "accuracy": accuracy}

    def train_model(
        self,
        train_ds: dict,
        test_ds: dict,
        epochs: int = 50,
        batch_size: int = 1,
        disable_loading_bar: bool = False,
    ):
        """
        Train the model.

        See the parent class for a full doc-string.
        """
        if self.model_state is None:
            self.init_model()

        state = self.model_state

        loading_bar = trange(
            1, epochs + 1, ncols=100, unit="batch", disable=disable_loading_bar
        )
        test_losses = []
        test_accuracy = []
        train_losses = []
        train_accuracy = []
        for i in loading_bar:
            loading_bar.set_description(f"Epoch: {i}")

            state, train_metrics = self._train_epoch(
                state, train_ds, batch_size=batch_size
            )
            metrics = self._evaluate_model(state.params, test_ds)

            loading_bar.set_postfix(test_loss=metrics["loss"])
            if self.accuracy_fn is not None:
                loading_bar.set_postfix(accuracy=metrics["accuracy"])
                test_accuracy.append(metrics["accuracy"])
                train_accuracy.append(train_metrics["accuracy"])

            test_losses.append(metrics["loss"])
            train_losses.append(train_metrics["loss"])

        # Update the final model state.
        self.model_state = state

        return {
            "test_losses": test_losses,
            "test_accuracy": test_accuracy,
            "train_losses": train_losses,
            "train_accuracy": train_accuracy,
        }

    def __call__(self, feature_vector: jnp.ndarray):
        """
        See parent class for full doc string.
        """
        state = self.model_state

        return self.apply_fn(state.params, feature_vector)
