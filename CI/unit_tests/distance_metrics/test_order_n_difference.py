"""
ZnRND: A Zincwarecode package.
License
-------
This program and the accompanying materials are made available under the terms
of the Eclipse Public License v2.0 which accompanies this distribution, and is
available at https://www.eclipse.org/legal/epl-v20.html
SPDX-License-Identifier: EPL-2.0
Copyright Contributors to the Zincwarecode Project.
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
Test the order n norm metric.
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import jax.numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal

from symsuite.distance_metrics.order_n_difference import OrderNDifference


class TestOrderNDifference:
    """
    Class to test the cosine distance measure module.
    """

    def test_order_2_distance(self):
        """
        Test the cosine similarity measure.

        Returns
        -------
        Assert the correct answer is returned for orthogonal, parallel, and
        somewhere in between.
        """
        metric = OrderNDifference(order=2, reduce_operation="sum")

        # Test orthogonal vectors
        point_1 = np.array([[1.0, 7.0, 0.0, 0.0]])
        point_2 = np.array([[1.0, 1.0, 0.0, 0.0]])
        assert_array_equal(metric(point_1, point_2), [36.0])

    def test_order_3_distance(self):
        """
        Test the cosine similarity measure.

        Returns
        -------
        Assert the correct answer is returned for orthogonal, parallel, and
        somewhere in between.
        """
        metric = OrderNDifference(order=3, reduce_operation="sum")

        # Test orthogonal vectors
        point_1 = np.array([[1.0, 1.0, 0.0, 0.0]])
        point_2 = np.array([[1.0, 7.0, 0.0, 0.0]])

        assert_almost_equal(metric(point_1, point_2), [-216.0], decimal=4)

    def test_multi_distance(self):
        """
        Test the cosine similarity measure.

        Returns
        -------
        Assert the correct answer is returned for orthogonal, parallel, and
        somewhere in between.
        """
        metric = OrderNDifference(order=3, reduce_operation="sum")

        # Test orthogonal vectors
        point_1 = np.array([[1.0, 7.0, 0.0, 0.0], [4, 7, 2, 1]])
        point_2 = np.array([[1.0, 1.0, 0.0, 0.0], [6, 3, 1, 8]])
        assert_almost_equal(metric(point_1, point_2), [216.0, -286.0], decimal=4)
