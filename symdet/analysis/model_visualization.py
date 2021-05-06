""" Visualize the NN models in different ways """

import matplotlib.pyplot as plt

from sklearn.manifold import TSNE


class Visualizer:
    """ Class for the visualization of NN models """

    def __init__(self, data, colour_map):
        """
        Constructor the visualizer class
        """

        self.data = data  # data to be visualized
        self.colour_map = colour_map  # colour map to be used in the plotting - differs for different potentials

    def tsne_visualization(self, perplexity=50, n_components=2, plot: bool = True, save: bool = False):
        """
        Display a TSNE representation of the models embedding layer
        """

        tsne_model = TSNE(n_components=n_components, perplexity=perplexity, random_state=1)
        tsne_representation = tsne_model.fit_transform(self.data)

        if plot:
            plt.scatter(tsne_representation[:, 0], tsne_representation[:, 1],
                        c=self.colour_map, cmap='viridis', vmax=1, vmin=-1)
            plt.colorbar()
            if save:
                plt.savefig(f'tsne_representation_{perplexity}_{n_components}.svg', dpi=800, format='svg')
            plt.show()

        return tsne_representation
