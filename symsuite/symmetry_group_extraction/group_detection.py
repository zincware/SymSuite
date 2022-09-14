"""
This program and the accompanying materials are made available under the terms of the
Eclipse Public License v2.0 which accompanies this distribution, and is available at
https://www.eclipse.org/legal/epl-v20.html
SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the Zincware Project.

Group Detection
===============
Cluster raw data into symmetry groups
"""
from typing import Tuple

import jax.numpy as jnp
import numpy as np

from symsuite.analysis.model_visualization import Visualizer
from symsuite.ml_models.dense_model import DenseModel
from symsuite.utils.data_clustering import (
    compute_com,
    compute_radius_of_gyration,
    to_categorical,
)


class GroupDetection:
    """
    A class to cluster raw data into symmetry groups.

    Attributes
    ----------
    model : DenseModel
            Model to use in the group detection.
    data_clusters : dict
            Data cluster class used for the partitioning of the data.
    representation_set : str
            Which set to use in the representation, train, validation, or test.
    """

    def __init__(
        self, model_representation: np.ndarray, data_classes: list
    ):
        """
        Constructor for the GroupDetection class.

        Parameters
        ----------
        model_representation : np.ndarray
                Model representation on which to perform the symmetry connection
                analysis.
        data_classes : list
                List of the data classes for better visualization
        """
        self.model_representations = model_representation
        self.data_classes = data_classes

    @staticmethod
    def _cluster_detection(function_data: np.ndarray, data: np.ndarray):
        """
        Use the results of the TSNE reduction to extract clusters.

        Parameters
        ----------
        function_data : tf.Tensor
                A tensor of the raw data to be collected.
        data : np.ndarray
                Results of the tsne representation

        Returns
        -------
        clusters : dict
                An unordered point cloud of data belonging to the same cluster.
                e.g. {1: [radial values], 2: [radial_values], ...}
        """
        net_array = np.concatenate((data, function_data), 1)
        sorted_data = jnp.take(net_array, jnp.argsort(net_array[:, -1]))
        class_array = np.unique(function_data[:, -1])

        point_cloud = {}
        # loop over the class array
        for i, item in enumerate(class_array):
            start = np.searchsorted(sorted_data[:, -1], item, side="left")
            stop = np.searchsorted(sorted_data[:, -1], item, side="right") - 1
            com = compute_com(sorted_data[start:stop, 0:2])
            rg = compute_radius_of_gyration(sorted_data[start:stop, 0:2], com)

            # print(f"Class: {item}, COM: {com}, Rg: {rg}")
            if rg > 1000:
                point_cloud[item] = sorted_data[start:stop, 2:-1]

        return point_cloud

    def run_symmetry_detection(self, plot: bool = True, save: bool = False):
        """
        Run the symmetry detection routine.

        Parameters
        ----------
        plot : bool
                Plot the TSNE visualization.
        save : bool
                Save the image plotted.
        Returns
        -------
        None
        """
        validation_data, predictions = self._get_model_predictions()
        accepted_data = self._filter_data(predictions, validation_data)
        representation = self.model.get_embedding_layer_representation(
            accepted_data
        )  # get the embedding layer

        visualizer = Visualizer(representation, accepted_data[:, -1])
        data = visualizer.tsne_visualization(plot=plot, save=save)

        # determine coupled groups in the tSNE representation.
        point_cloud = self._cluster_detection(validation_data, data)

        return point_cloud
