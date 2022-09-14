"""
SymSuite
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
init function for the accuracy functions.
"""
from symsuite.accuracy_functions.accuracy_function import AccuracyFunction
from symsuite.accuracy_functions.label_accuracy import LabelAccuracy

__all__ = [AccuracyFunction.__name__, LabelAccuracy.__name__]
