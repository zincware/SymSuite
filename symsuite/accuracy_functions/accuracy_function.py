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
Parent class for the accuracy functions.
"""
import jax.numpy as np


class AccuracyFunction:
    """
    Class for computing accuracy.
    """

    def __call__(self, predictions: np.array, targets: np.array) -> float:
        """
        Accuracy function call method.

        Parameters
        ----------
        predictions : np.array
                First set of points to be compared.
        targets : np.array
                Second points to compare. This will be passed through any
                pre-processing of the child classes.

        Returns
        -------
        accuracy : float
                Accuracy of the points.
        """
        raise NotImplementedError("Implemented in child class.")
