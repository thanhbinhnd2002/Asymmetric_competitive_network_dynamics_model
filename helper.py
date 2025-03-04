import random

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def get_random_competitor(n_nodes):
    return random.randint(0, n_nodes - 1)


def load_distance_matrix_from_csv(file_path):
    d = np.loadtxt(file_path, delimiter=",", dtype=int)
    return d


def visualize(net):
    plt.figure(figsize=(50, 50))
    nx.draw(net, with_labels=True, node_color='lightblue', node_size=800, font_size=10, font_weight='bold',
            arrows=True)
    plt.show()
