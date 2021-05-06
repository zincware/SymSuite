""" Study of the double well potential """

from symdet.test_systems.double_well_potential import Double_Well_Potential
from symdet.models.dense_model import DenseModel
from symdet.symmetry_groups.data_clustering import DataCluster
from symdet.symmetry_groups.group_detection import GroupDetection


def main():
    """
    Main function to study the clustering
    """

    # Instantiate the class and build the training data
    double_well_potential = Double_Well_Potential([-5, 5], [1 / 5, 1e-3], n_class_members=1000)
    double_well_potential.plot_potential()
    clustering = DataCluster(double_well_potential.get_function_values())
    training_data = clustering.range_binning(value_range=[-5, 5],
                                             bin_operation=[1 / 5, 1e-3],
                                             representatives=300,
                                             plot=False)
    # Build, train, and evaluate the model
    model = DenseModel(training_data,
                       n_layers=7,
                       units=80,
                       epochs=10,
                       batch_size=64,
                       lr=0.00025)
    sym_detector = GroupDetection(model, clustering)
    sym_detector.run_symmetry_detection(plot=True)


if __name__ == "__main__":
    """ Standard boilerplate. """
    main()
