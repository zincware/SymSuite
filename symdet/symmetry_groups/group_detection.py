"""
Cluster raw data into symmetry groups
"""
from symdet.models.dense_model import DenseModel
from symdet.symmetry_groups.data_clustering import DataCluster
from symdet.analysis.model_visualization import Visualizer
import tensorflow as tf
from typing import Tuple

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

    def _get_model_predictions(self) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Train the attached model.

        Returns
        -------
        model_predictions : tf.Tensor
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

    def _cluster_detection(self):
        """
        Use the results of the TSNE reduction to extract clusters.

        Returns
        -------
        clusters : dict
                An unordered point cloud of data belonging to the same cluster.
        """
        pass

    def run_symmetry_detection(self, plot: bool = True, save: bool = False):
        """
        Run the symmetry detection routine.
        Returns
        -------

        """
        validation_data, predictions = self._get_model_predictions()
        colour_map, clusters, visualization_data = self.cluster._cluster_data(predictions, validation_data)
        representation = self.model.get_embedding_layer_representation(visualization_data)  # get the embedding layer

        visualizer = Visualizer(representation, colour_map)
        naive_clusters = visualizer.tsne_visualization(plot=plot, save=save)

