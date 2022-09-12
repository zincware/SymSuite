"""
This program and the accompanying materials are made available under the terms of the
Eclipse Public License v2.0 which accompanies this distribution, and is available at
https://www.eclipse.org/legal/epl-v20.html
SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the Zincware Project.

Description: Methods to help with clustering data.
"""
import jax.numpy as jnp
import numpy as np
from typing import Tuple
import sys


def _build_condlist(data: np.array, bin_values: dict) -> Tuple:
    """
    Build the condition list for the piecewise implementation.

    Parameters
    ----------
    data : np.array
            Data on which to apply conditions.
    bin_values : np.array
            Bin numbers for the constraint.

    Returns
    -------
    conditions : list
            These are conditions applied to the data.
    classes : list
            Class keys.
    """

    conditions = []
    classes = []
    for key in bin_values:
        conditions.append(
            np.logical_and(
                data >= (bin_values[key][0]), data <= (bin_values[key][1])
            )
        )
        classes.append(key)

    return conditions, classes


def _function_to_bins(function_values: tf.Tensor, bin_values: dict) -> tf.Tensor:
    """
    Sort function values into bins.

    Parameters
    ----------
    function_values : np.array
            Function values corresponding to radii values.
    bin_values : dict
            bin dictionary where keys are the class numbers.

    Returns
    -------
    conditions : jnp.ndarrau
            Conditions from the cond list build.
    """

    conditions, functions = _build_condlist(function_values, bin_values)

    return jnp.array(conditions)


def range_binning(
        image: jnp.ndarrau,
        domain: jnp.ndarrau,
        value_range: list,
        bin_operation: list,
        representatives: int = 100) -> dict:
    """
    A method to apply simple range binning to some data.

    Parameters
    ----------
    image : jnp.ndarrau
            data to cluster.
    domain : jnp.ndarrau
            data pool to return clustered.
    representatives : int
            Number of class representatives to have for each bin.
    value_range : list
            The parameters within which to bin e.g.  k in [-5, 5]
    bin_operation : list
            Operation to apply to the bins e.g [1/5, 1e-3] will
            lead to bins of the form [k/5 - 1e-3, k/5 + 1e-3]

    Returns
    -------
    classes : dict
            Data class numbers and their data representatives as a dictionary.
    """
    # Construct the classes and their range.
    bin_values = {}
    n_classes = (value_range[1] - value_range[0]) + 1

    for k in np.linspace(value_range[0], value_range[1], n_classes, dtype=int):
        bin_values[f"{k + abs(value_range[0])}"] = [
            bin_operation[0] * k - bin_operation[1],
            bin_operation[0] * k + bin_operation[1],
        ]

    # Collect the bin masks
    bin_masks = _function_to_bins(image, bin_values)

    # Check that there is enough data in each class.
    bin_count = tf.reduce_sum(tf.cast(bin_masks, tf.int8), 1)
    if any(bin_count) < representatives:
        print("WARNING: Not enough data! Some classes will be under-represented.")

    class_keys = list(bin_values.keys())

    clustered_data = {}
    for i in range(len(class_keys)):
        clustered_data[class_keys[i]] = {}
        filtered_domain = tf.boolean_mask(domain, bin_masks[i])
        filtered_image = tf.boolean_mask(image, bin_masks[i])
        clustered_data[class_keys[i]]['domain'] = filtered_domain[0:representatives]
        clustered_data[class_keys[i]]['image'] = filtered_image[0:representatives]

    return clustered_data


def compute_com(data: np.ndarray):
    """
    Compute the center of mass of some data.

    Parameters
    ----------
    data : np.ndarray
            Data on which to compute the center of mass.

    Returns
    -------

    """
    return tf.reduce_mean(data, axis=0)


def compute_radius_of_gyration(data: np.ndarray, com: np.ndarray):
    """
    Compute the radius of gyration of some data.

    Parameters
    ----------
    data : np.ndarray
    com : np.ndarray

    Returns
    -------

    """
    rg_primitive = tf.reduce_sum((data - com)**2, axis=1)

    return tf.reduce_mean(rg_primitive, axis=0)
