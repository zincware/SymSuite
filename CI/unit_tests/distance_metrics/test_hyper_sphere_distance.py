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
Test the hyper sphere distance module.
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import jax.numpy as np
from numpy.testing import assert_array_almost_equal

from symsuite.distance_metrics.hyper_sphere_distance import HyperSphere


class TestCosineDistance:
    """
    Class to test the cosine distance measure module.
    """

    def test_hyper_sphere_distance(self):
        """
        Test the hyper sphere distance.

        Returns
        -------
        Assert the correct answer is returned for orthogonal, parallel, and
        somewhere in between.
        """
        metric = HyperSphere(order=2)

        # Test orthogonal vectors
        point_1 = np.array([[1, 0, 0, 0]])
        point_2 = np.array([[0, 1, 0, 0]])
        assert_array_almost_equal(metric(point_1, point_2), [1.41421356])

        # Test parallel vectors
        point_1 = np.array([[1, 0, 0, 0]])
        point_2 = np.array([[1, 0, 0, 0]])
        assert_array_almost_equal(metric(point_1, point_2), [0])

        # Somewhere in between
        point_1 = np.array([[1.0, 0, 0, 0]])
        point_2 = np.array([[0.5, 1.0, 0, 3.0]])
        assert_array_almost_equal(
            metric(point_1, point_2), [0.84382623 * np.sqrt(10.25)]
        )

    def test_multiple_distances(self):
        """
        Test the hyper sphere distance.

        Returns
        -------
        Assert the correct answer is returned for orthogonal, parallel, and
        somewhere in between.
        """
        metric = HyperSphere(order=2)

        # Test orthogonal vectors
        point_1 = np.array([[1, 0, 0, 0], [1, 0, 0, 0], [1.0, 0, 0, 0]])
        point_2 = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0.5, 1.0, 0, 3.0]])
        assert_array_almost_equal(
            metric(point_1, point_2),
            [np.sqrt(2), 0, 0.84382623 * np.sqrt(10.25)],
            decimal=6,
        )
