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
Module for the Mahalanobis distance.
"""
import jax.numpy as np
import scipy.spatial.distance

from symsuite.distance_metrics.distance_metric import DistanceMetric


class MahalanobisDistance(DistanceMetric):
    """
    Compute the mahalanobis distance between points.
    """

    def __call__(self, point_1: np.array, point_2: np.array, **kwargs) -> np.array:
        """
        Call the distance metric.

        Mahalanobis Distance between points in the point_1 tensor will be computed
        between those in the point_2 tensor element-wise. Therefore, we will have:

                point_1[i] - point_2[i] for all i.

        Parameters
        ----------
        point_1 : tf.Tensor (n_points, point_dimension)
            First set of points in the comparison.
        point_2 : tf.Tensor (n_points, point_dimension)
            Second set of points in the comparison.
        kwargs
                Miscellaneous keyword arguments for the specific metric.
        Returns
        -------
        d(point_1, point_2) : tf.tensor : shape=(n_points, 1)
                Array of distances for each point.
        """
        inverted_covariance = np.linalg.inv(np.cov(point_1.T))
        distances = []
        for i in range(len(point_1.T[0, :])):
            distance = scipy.spatial.distance.mahalanobis(
                point_1[i], point_2[i], inverted_covariance
            )
            distances.append(distance)

        return distances
