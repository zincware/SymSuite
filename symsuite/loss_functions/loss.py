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
Module for the simple loss for TensorFlow.
"""
from abc import ABC

import jax.numpy as np

from symsuite.distance_metrics.distance_metric import DistanceMetric


class Loss(ABC):
    """
    Class for the simple loss.

    Attributes
    ----------
    metric : DistanceMetric
    """

    def __init__(self):
        """
        Constructor for the simple loss parent class.
        """
        super().__init__()
        self.metric: DistanceMetric = None

    def __call__(self, point_1: np.array, point_2: np.array) -> float:
        """
        Summation over the tensor of the respective similarity measurement
        Parameters
        ----------
        point_1 : np.array
                first neural network representation of the considered points
        point_2 : np.array
                second neural network representation of the considered points

        Returns
        -------
        loss : float
                total loss of all points based on the similarity measurement
        """
        return np.mean(self.metric(point_1, point_2), axis=0)
