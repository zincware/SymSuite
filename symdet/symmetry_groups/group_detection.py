"""
This file is part of the SymDet distribution (https://github.com/SamTov/SymDet).
Copyright (c) 2021 Samuel Tovey.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, version 3.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>.

Group Detection
===============
Cluster raw data into symmetry groups
"""
from symdet.models.dense_model import DenseModel
from symdet.symmetry_groups.data_clustering import DataCluster
from symdet.analysis.model_visualization import Visualizer
from typing import Tuple
import numpy as np
from sklearn.cluster import MeanShift


class GroupDetection:
    """
    A class to cluster raw data into symmetry groups.

    Attributes
    ----------
    model : DenseModel
                    Model to use in the group detection.
    cluster : DataCluster
                Data cluster class used for the partitioning of the data.
    """

    def __init__(self, model: DenseModel, cluster: DataCluster):
        """
        Constructor for the GroupDetection class.

        Parameters
        ----------
        model : DenseModel
                Model to use in the group detection.
        cluster : DataCluster
                Data cluster class used for the partitioning of the data.
        """
        self.model = model
        self.cluster = cluster

    def _get_model_predictions(self) -> Tuple:
        """
        Train the attached model.

        Returns
        -------
        model_predictions : Tuple
                Embedding layer of the NN on validation data.
        """
        validation_data = self.model.train_model()
        predictions = self.model.model.predict(validation_data[:, 0])

        return validation_data, predictions

    def _run_visualization(self):
        """
        Perform a visualization on the TSNE data.
        Returns
        -------

        """
        pass

    def _cluster_detection(self, classes: np.ndarray, data: np.ndarray, vis_data: np.ndarray):
        """
        Use the results of the TSNE reduction to extract clusters.

        Parameters
        ----------
        classes : np.ndarray
                A numpy array stating which class the i'th data point is in
        data : np.ndarray
                Results of the tsne representation
        vis_data : np.ndarray
                Raw data to be returned by class.

        Returns
        -------
        clusters : dict
                An unordered point cloud of data belonging to the same cluster.
                e.g. {1: [radial values], 2: [radial_values], ...}
        """
        net_array = np.concatenate((data, vis_data, np.array([classes]).T), 1)
        sorted_data = net_array[np.argsort(net_array[:, -1])]
        class_array = np.unique(classes)

        ms = MeanShift()  # define the cluster algorithm

        point_cloud = {}
        # loop over the class array
        for i, item in enumerate(class_array):
            start = np.searchsorted(sorted_data[:, -1], item, side='left')
            stop = np.searchsorted(sorted_data[:, -1], item, side='right') - 1
            n_cluster = len(np.unique(ms.fit(sorted_data[start:stop, 0:len(data)]).labels_))
            if n_cluster == 2:
                point_cloud[item] = sorted_data[start:stop, len(data):-1]

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
        colour_map, clusters, visualization_data = self.cluster.cluster_data(
            predictions, validation_data
        )
        representation = self.model.get_embedding_layer_representation(
            visualization_data
        )  # get the embedding layer

        visualizer = Visualizer(representation, colour_map)
        data = visualizer.tsne_visualization(plot=plot, save=save)

        point_cloud = self._cluster_detection(colour_map, data, visualization_data)

        return point_cloud
