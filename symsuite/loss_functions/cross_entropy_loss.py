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
Implement a cross entropy loss function.
"""
import jax
import optax

from symsuite.loss_functions.loss import Loss


class CrossEntropyDistance:
    """
    Class for the cross entropy distance
    """

    def __init__(self, classes: int, apply_softmax: bool = False):
        """
        Constructor for the distance

        Parameters
        ----------
        classes : int
                Number of classes in the one-hot encoding.
        apply_softmax : bool (default = False)
                If true, softmax is applied to the prediction before computing the loss.
        """
        self.classes = classes
        self.apply_softmax = apply_softmax

    def __call__(self, prediction, target):
        """

        Parameters
        ----------
        prediction (batch_size, n_classes)
        target

        Returns
        -------

        """
        if self.apply_softmax:
            prediction = jax.nn.softmax(prediction)
        one_hot_labels = jax.nn.one_hot(target, num_classes=self.classes)
        return optax.softmax_cross_entropy(logits=prediction, labels=one_hot_labels)


class CrossEntropyLoss(Loss):
    """
    Class for the cross entropy loss
    """

    def __init__(self, classes: int = 10, apply_softmax: bool = False):
        """
        Constructor for the mean power loss class.

        Parameters
        ----------
        classes : int (default=10)
                Number of classes in the loss.
        apply_softmax : bool (default = False)
                If true, softmax is applied to the prediction before computing the loss.
        """
        super(CrossEntropyLoss, self).__init__()
        self.metric = CrossEntropyDistance(classes=classes, apply_softmax=apply_softmax)
