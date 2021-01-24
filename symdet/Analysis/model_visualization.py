""" Visualize the NN models in different ways """

import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
import sklearn as sk
from sklearn.manifold import TSNE


class Visualizer:
    """ Class for the visualization of NN models """

    def __init__(self, data, colour_map):
        """ Constructor the visualizer class """

        self.data = data  # data to be visualized
        self.colour_map = colour_map  # colour map to be used in the plotting - differs for different potentials


    def tsne_visualization(self, perplexity=80, n_components=2):
        """ display a tsne representation of the models embedding layer """

        tsne_model = TSNE(n_components=n_components, perplexity=perplexity, random_state=1)
        tsne_representation = tsne_model.fit_transform(self.data)
        plt.scatter(tsne_representation[:, 0], tsne_representation[:, 1],
                    c=self.colour_map, cmap='viridis', vmax=10, vmin=-1)
        plt.colorbar()
        plt.show()
