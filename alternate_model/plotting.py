import matplotlib.pyplot as plt
import networkx as nx

from graph import *

def plot_network(G):
    pos = nx.get_node_attributes(G, 'pos')
    nx.draw(G=G, with_labels=False, labels=nx.get_node_attributes(G, 'label'), pos=pos, node_color='blue', edge_color='gray', node_size=1)
    plt.show()


if __name__ == '__main__':
    filename1 = 'data/network/sij.json'

    G = read_from_json(filename=filename1)

    G = remove_secondary_nodes(G)
    print(G.number_of_nodes())

    plot_network(G)