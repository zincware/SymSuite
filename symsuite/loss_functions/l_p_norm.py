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
ZnRND L^{p} norm TF loss function.
"""
from symsuite.distance_metrics.l_p_norm import LPNorm
from symsuite.loss_functions.loss import Loss


class LPNormLoss(Loss):
    """
    Class for the mean power loss
    """

    def __init__(self, order: float):
        """
        Constructor for the L_p norm loss class.

        Parameters
        ----------
        order : float
                Order to which the difference should be raised.
        """
        super(LPNormLoss, self).__init__()
        self.metric = LPNorm(order=order)
