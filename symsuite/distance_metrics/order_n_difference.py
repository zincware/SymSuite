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
Raise a difference to a power of order n.

e.g. (a - b)^n
"""
import jax.numpy as np

from symsuite.distance_metrics.distance_metric import DistanceMetric


class OrderNDifference(DistanceMetric):
    """
    Compute the order n difference between points.
    """

    def __init__(self, order: float = 2, reduce_operation: str = "mean"):
        """
        Constructor for the order n distance.

        Parameters
        ----------
        order : float (default=2)
                Order to which the difference should be raised.
        reduce_operation : str (default = "mean")
                How to reduce the order N difference, either a sum or a mean.
        """
        self.order = order
        self.reduce_operation = reduce_operation

    def __call__(self, point_1: np.ndarray, point_2: np.ndarray, **kwargs):
        """
        Call the distance metric.

        Distance between points in the point_1 tensor will be computed between those in
        the point_2 tensor element-wise. Therefore, we will have:

                point_1[i] - point_2[i] for all i.

        Parameters
        ----------
        point_1 : np.ndarray (n_points, point_dimension)
            First set of points in the comparison.
        point_2 : np.ndarray (n_points, point_dimension)
            Second set of points in the comparison.
        kwargs
                Miscellaneous keyword arguments for the specific metric.

        Returns
        -------
        d(point_1, point_2) : np.ndarray : shape=(n_points, 1)
                Array of distances for each point.
        """
        diff = point_1 - point_2

        if self.reduce_operation == "mean":
            return np.mean(np.power(diff, self.order), axis=1)
        elif self.reduce_operation == "sum":
            return np.sum(np.power(diff, self.order), axis=1)
        else:
            raise ValueError(f"Invalid reduction operation: {self.reduce_operation}")
