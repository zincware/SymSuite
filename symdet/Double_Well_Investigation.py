""" Study of the double well potential """

import numpy as np
import tensorflow as tf
import os
import time

from Exact_Potentials.Double_Well_Potential import Double_Well_Potential
from Models.dense_model import DenseModel
from Analysis.model_visualization import Visualizer

from Exact_Potentials.SO2_Data import SO2
from generators.extract_generators import GeneratorExtract

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def main_clustering():
    """ Main function to study the clustering """

    # Instantiate the class and build the training data
    double_well_potential = Double_Well_Potential([-5, 5], [1 / 5, 1e-3], n_class_members=3000)
    double_well_potential.plot_potential()
    training_data = double_well_potential.build_dataset()

    # Build, train, and evaluate the model
    model = DenseModel(training_data,
                       n_layers=7,
                       units=80,
                       epochs=10,
                       batch_size=64,
                       lr=0.00025)  # Build the model
    truth_values = model.train_model()

    predictions = model.model.predict(truth_values[:, 0])  # get the model predictions for the training data
    colour_map, clusters, visualization_data = double_well_potential._cluster_data(predictions, truth_values)

    representation = model.get_embedding_layer_representation(visualization_data)  # get the embedding layer

    # Visualize the model
    visualizer = Visualizer(representation, colour_map)
    visualizer.tsne_visualization()


def main_generator_extraction():
    """ Main function to study the generator extraction

    We will use a points on a circle example to extract symmetries from there
    """

    data_generator = SO2(n_points=500)  # instantiate the class
    data_generator.plot_data()  # plot the data

    generator_extractor = GeneratorExtract(data_generator.data, delta=0.5, epsilon=0.3)  # instantiate the class
    generator_extractor.extract_generators()  # extract the generators


if __name__ == "__main__":
    start = time.time()

    #main_clustering()
    main_generator_extraction()

    print(f"Program ran in {(time.time() - start)/60} minutes")
