"""
This program and the accompanying materials are made available under the terms of the
Eclipse Public License v2.0 which accompanies this distribution, and is available at
https://www.eclipse.org/legal/epl-v20.html
SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the Zincware Project.

Description: Test the double_well_potential module.
"""
import unittest
from symdet.data.double_well_potential import DoubleWellPotential


class TestDoubleWellPotential(unittest.TestCase):
    """
    Class to test the DoubleWellPotential data generator.
    """

    @classmethod
    def setUpClass(cls) -> None:
        """
        Set up the data generator class for the tests.

        Returns
        -------

        """
        cls.generator = DoubleWellPotential()

    def test_pick_points(self):
        """
        test the pick points method.

        Returns
        -------
        Assert that n_points are picked out.
        """
        self.generator._pick_points(500)

        self.assertEqual(len(self.generator.domain), 500)

    def test_double_well(self):
        """
        test the double well dat
        Returns
        -------

        """
        self.generator._pick_points(500)
        self.generator._double_well()
        self.assertEqual(len(self.generator.image), 500)


if __name__ == '__main__':
    unittest.main()
