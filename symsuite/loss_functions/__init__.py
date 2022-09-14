"""
Symsuite

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
Package containing custom loss functions.
"""
from symsuite.loss_functions.absolute_angle_difference import AngleDistanceLoss
from symsuite.loss_functions.cosine_distance import CosineDistanceLoss
from symsuite.loss_functions.cross_entropy_loss import CrossEntropyLoss
from symsuite.loss_functions.l_p_norm import LPNormLoss
from symsuite.loss_functions.mahalanobis import MahalanobisLoss
from symsuite.loss_functions.mean_power_error import MeanPowerLoss
from symsuite.loss_functions.loss import Loss

__all__ = [
    AngleDistanceLoss.__name__,
    CosineDistanceLoss.__name__,
    LPNormLoss.__name__,
    MahalanobisLoss.__name__,
    MeanPowerLoss.__name__,
    Loss.__name__,
    CrossEntropyLoss.__name__,
]
