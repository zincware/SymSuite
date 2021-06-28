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
from symdet.ml_models.dense_model import DenseModel
from symdet.analysis.model_visualization import Visualizer
from typing import Tuple
import numpy as np
import tensorflow as tf
from symdet.utils.data_clustering import compute_com, compute_radius_of_gyration


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

    def __init__(self, model: DenseModel, data_clusters: dict, representation_set: str = 'train'):
        """
        Constructor for the GroupDetection class.

        Parameters
        ----------
        model : DenseModel
                Model to use in the group detection.
        data_clusters : dict
                Data cluster class used for the partitioning of the data.
        representation_set : str
                Which set to use in the representation, train, validation, or test.
        """
        self.model = model
        self.data = data_clusters
        self.representation_set = representation_set
        self.model.add_data(self.data)  # add the data to the model.

    def _get_model_predictions(self) -> Tuple:
        """
        Train the attached model.

        Returns
        -------
        val_data : tf.Tensor
                Data on which the prediction were made.
        model_predictions : Tuple
                Embedding layer of the NN on validation data.
        """
        self.model.train_model()
        if self.representation_set == 'train':
            val_data = self.model.train_ds
            predictions = self.model.model.predict(val_data[:, 0:self.model.input_shape])
        elif self.representation_set == 'test:':
            val_data = self.model.test_ds
            predictions = self.model.model.predict(val_data[:, 0:self.model.input_shape])
        else:
            val_data = self.model.val_ds
            predictions = self.model.model.predict(val_data[:, 0:self.model.input_shape])

        return val_data, predictions

    def _run_visualization(self):
        """
        Perform a visualization on the TSNE data.

        Returns
        -------

        """
        pass

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
        sorted_data = tf.gather(net_array, tf.argsort(net_array[:, -1])).numpy()
        class_array = np.unique(function_data[:, -1])

        point_cloud = {}
        # loop over the class array
        for i, item in enumerate(class_array):
            start = np.searchsorted(sorted_data[:, -1], item, side='left')
            stop = np.searchsorted(sorted_data[:, -1], item, side='right') - 1
            com = compute_com(sorted_data[start:stop, 0:2])
            rg = compute_radius_of_gyration(sorted_data[start:stop, 0:2], com)

            #print(f"Class: {item}, COM: {com}, Rg: {rg}")
            if rg > 1000:
                point_cloud[item] = sorted_data[start:stop, 2:-1]

        return point_cloud

    @staticmethod
    def _filter_data(predictions: tf.Tensor, targets: tf.Tensor):
        """
        Check which data points are predicted well and include them in the data.

        Parameters
        ----------
        targets : tf.Tensor
                Target values on which predictions were made.
        predictions : tf.Tensor
                Network predictions.

        Returns
        -------

        """
        accepted_candidates = np.zeros(len(predictions))
        target_values = tf.keras.utils.to_categorical(targets[:, -1])
        counter = 0
        for i, item in enumerate(predictions):
            if np.linalg.norm(predictions[i] - target_values[i]) <= 2e-1:
                accepted_candidates[counter] = i
                counter += 1
        accepted_candidates = tf.convert_to_tensor(accepted_candidates[0:counter], dtype=tf.int32)

        return tf.gather(targets, accepted_candidates)

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
        representation = self.model.get_embedding_layer_representation(accepted_data)  # get the embedding layer

        visualizer = Visualizer(representation, accepted_data[:, -1])
        data = visualizer.tsne_visualization(plot=plot, save=save)

        # determine coupled groups in the tSNE representation.
        point_cloud = self._cluster_detection(validation_data, data)

        return point_cloud
