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
Module for testing the loss functions

Notes
-----
As the loss functions come directly from distance metrics and the distance metrics are
heavily tested, here we test all loss functions on the same set of data and ensure that
the results are as expected.
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import jax.numpy as np
import pytest

from symsuite.loss_functions import (
    AngleDistanceLoss,
    CosineDistanceLoss,
    LPNormLoss,
    MeanPowerLoss,
)


class TestLossFunctions:
    """
    Class for the testing of the ZnRND loss functions.
    """

    @classmethod
    def setup_class(cls):
        """
        Prepare the test class
        """
        cls.linear_predictions = np.array([[1, 1, 2], [9, 9, 9], [0, 0, 0], [9, 1, 1]])
        cls.linear_targets = np.array([[9, 9, 9], [1, 1, 2], [9, 1, 1], [0, 0, 0]])

        cls.angular_predictions = np.array([[0, 0, 1], [1, 0, 0], [1, 1, 0], [1, 0, 1]])
        cls.angular_targets = np.array([[1, 0, 0], [0, 0, 1], [1, 0, 1], [1, 1, 0]])

    def test_absolute_angle(self):
        """
        Test the absolute angle loss
        """
        loss = AngleDistanceLoss()(
            self.angular_predictions / 9, self.angular_targets / 9
        )
        loss == pytest.approx(0.417, 0.0001)

    def test_cosine_distance(self):
        """
        Test the cosine_distance loss
        """
        loss = CosineDistanceLoss()(
            self.angular_predictions / 9, self.angular_targets / 9
        )
        loss == 0.75

    def test_l_p_norm(self):
        """
        Test the l_p norm loss
        """
        loss = LPNormLoss(order=2)(self.linear_predictions, self.linear_targets)
        loss == pytest.approx(11.207, 0.0001)

    def test_mean_power(self):
        """
        Test the mean_power loss
        """
        loss = MeanPowerLoss(order=2)(self.linear_predictions, self.linear_targets)
        loss == 130.0
