"""
Python module for clustering data into classes. These may then be passed to the group detection class to
identify symmetries between the classes.
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf


class DataCluster:
    """
    A class to generate clusters of data from some raw input.

    Attributes
    ----------
    input_data: np.array
            Data to be clustered.
    """

    def __init__(self, data: np.array):
        """
        Constructor for the DatCluster class.

        Parameters
        ----------
        data : np.array
                Data to be clustered. This should be in an (n, d) shape array where n is the number of points and d is
                the dimension of the data. For 2 dimensional data this would be simple (x, y) coordinates.
        """
        self.input_data = data

    @staticmethod
    def _count_bins(data):
        """
        Count how many members are in a representative class.
        """
        summed_array = []
        for cls in data:
            summed_array.append(np.sum(cls))

        return summed_array

    @staticmethod
    def _build_condlist(data: np.array, bin_values: dict):
        """
        Build the condition list for the piecewise implementation.

        Parameters
        ----------
        data : np.array
        bin_values : np.array
        """

        conditions = []
        classes = []
        for key in bin_values:
            conditions.append(np.logical_and(data >= (bin_values[key][0]), data <= (bin_values[key][1])))
            classes.append(key)

        return conditions, classes

    @staticmethod
    def _cluster_data(predictions, training_data):
        """
        Cluster data by the norm of the potential at the minimum
        """

        correlated_classes = np.zeros(len(predictions))
        counter = 0

        truth_data = training_data[:, 1]
        radii = training_data[:, 0]
        categorical_truth_data = tf.keras.utils.to_categorical(truth_data)
        for i in range(len(predictions)):
            if np.linalg.norm(predictions[i] - categorical_truth_data[i]) <= 2e-1:
                correlated_classes[counter] = i
                counter += 1

        correlated_classes = correlated_classes[0:counter]

        colour_classes = np.zeros((len(correlated_classes)), dtype=int)

        visualization_data = np.zeros((len(correlated_classes), 1))

        for i in range(len(correlated_classes)):
            visualization_data[i] = np.copy(radii[int(correlated_classes[i])])
            colour_classes[i] = np.copy(truth_data[int(correlated_classes[i])])

        return colour_classes, correlated_classes, visualization_data

    def _function_to_bins(self, function_values, bin_values: dict):
        """
        Sort function values into bins.

        Parameters
        ----------
        function_values : np.array
        bin_values : dict
        """

        conditions, functions = self._build_condlist(function_values, bin_values)

        return conditions

    def plot_clusters(self, data, save: bool = False):
        """
        Plot the clustered data on the raw.

        Parameters
        ----------
        data : dict
                A dictionary of the data clusters. Values are x values of the function belonging to the class associated
                with the key.
        save : bool
                If true, save the resulting plot.

        Returns
        -------
        Plots the clustering.
        """

        f_val = []
        for i, key in enumerate(data.keys()):
            f_val.append([])
            for radial_point in tqdm(data[key]):
                index = list(self.input_data[:, 0]).index(radial_point)
                f_val[i].append([radial_point, self.input_data[:, 1][index]])

        f_val = np.array(f_val)
        for group in range(len(f_val)):
            plt.plot(f_val[group][:, 0], f_val[group][:, 1], '.')

        plt.plot(self.input_data)
        plt.ylim(-1.5, 1.0)
        plt.xlim(-0.1, 2.0)
        if save:
            plt.savefig('Data_Clusters.svg', dpi=800, format='svg')
        plt.show()

    def range_binning(self, value_range: list, bin_operation: list, axis: int = 1, representatives: int = 100,
                      plot: bool = False, save_plot: bool = False):
        """
        A method to apply simple range binning to some data.

        Parameters
        ----------
        representatives : int
                Number of class representatives to have for each bin.
        value_range : list
                The parameters within which to bin e.g.  k \in [-5, 5]
        bin_operation : list
                Operation to apply to the bins e.g [1/5, 1e-3] will lead to bins of the form [k/5 - 1e-3, k/5 + 1e-3]
        axis : int
                Axis along which to cluster data.
        plot : bool
                If true, plot the clusters after running.
        save_plot : bool
                If true, save the resulting plot

        Returns
        -------
        classes : dict
                Data class numbers and their data representatives as a dictionary.
        """
        bin_values = {}
        for k in np.linspace(value_range[0], value_range[1], 11, dtype=int):
            bin_values[f"{k + abs(value_range[0])}"] = [bin_operation[0] * k - bin_operation[1],
                                                        bin_operation[0] * k + bin_operation[1]]

        bin_masks = self._function_to_bins(self.input_data[:, axis], bin_values)
        bin_count = self._count_bins(bin_masks)
        if all(bin_count) > representatives is False:
            print("Not enough data, please provide more")
        class_keys = list(bin_values.keys())  # Get the keys for the class
        potential_data = {}  # Instantiate dictionary to store the data
        for i in range(len(class_keys)):
            filtered_array = self.input_data[:, 0] * bin_masks[i]
            filtered_array = filtered_array[filtered_array != 0]
            filtered_array = np.random.choice(filtered_array, size=representatives)
            potential_data[class_keys[i]] = filtered_array

        if plot:
            self.plot_clusters(potential_data, save=save_plot)

        return potential_data
