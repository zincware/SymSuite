""" Study of the double well potential """

from symdet.exact_potentials.double_well_potential import Double_Well_Potential
from symdet.models.dense_model import DenseModel
from symdet.analysis.model_visualization import Visualizer

from symdet.exact_potentials.so2_data import SO2
from symdet.exact_potentials.so3_data import SO3
from symdet.generators.generators import GeneratorExtraction


def main_clustering():
    """ Main function to study the clustering """

    # Instantiate the class and build the training data
    double_well_potential = Double_Well_Potential([-5, 5], [1 / 5, 1e-3], n_class_members=1000)
    double_well_potential.plot_potential()
    training_data = double_well_potential.build_dataset(plot_clusters=False)

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

    data_generator = SO2(n_points=200, noise=True)  # instantiate the class
    data_generator.plot_data()  # plot the data

    sphere = SO3(n_points=500, noise=True)
    sphere.plot_data()

    generator_extractor = GeneratorExtraction(data_generator.data,
                                              delta=0.5,
                                              epsilon=0.3,
                                              candidate_runs=5)
    generator_extractor.perform_generator_extraction()  # extract the generators


if __name__ == "__main__":
    #main_clustering()
    main_generator_extraction()