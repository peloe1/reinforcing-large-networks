import networkx as nx
import time
import numpy as np
import random

from information_set import compute_extreme_points
from portfolio import portfolio_as_bitmask, generate_feasible_portfolios
from model.subnetwork import cost_efficient_portfolios
from model.path import simple_paths, terminal_pairs
from model.path import feasible_paths
from graph import construct_graph, generate_random_graph_with_positions



def speedTest() -> None:
    filename = 'data/network/sij.json'
    num_nodes = 40

    reliabilities = {i: 0.99 for i in range(num_nodes)}
    terminal_nodes = [2, 11, 32] # East, West, South
    for node in terminal_nodes:
        reliabilities[node] = 1.0
    
    G: nx.Graph = construct_graph(filename, reliabilities)

    for node in terminal_nodes:
        G.nodes[node]['reliability'] = 1

    terminal_node_pairs = terminal_pairs(terminal_nodes)

    paths = feasible_paths(G, terminal_node_pairs)

    r = 20
    node_reinforcements = [(i, 0.995) for i in range(r)]
    costs = [1.0 for _ in range(r)]
    budget = 12

    start = time.time()
    Q_F, feasible_portfolio_costs = generate_feasible_portfolios(r, costs, budget)
    end = time.time()
    print(f"It took {(end - start):.2f} seconds to compute the {len(Q_F)} feasible portfolios")
    

    print("-----------------------------------------")
    print("New version with binary representation")
    start = time.time()
    Q_CE, portfolio_costs = cost_efficient_portfolios(G, paths, node_reinforcements, Q_F, feasible_portfolio_costs , False)
    end = time.time()
    if end - start > 60:
        print(f"Time to compute cost-efficient portfolios: {(end - start)/60:.2f} minutes")
    else:
        print(f"Time to compute cost-efficient portfolios: {(end - start):.2f} seconds")

    print(f"Number of resulting cost-efficient portfolios: {len(Q_CE)}")

    return

if __name__ == "__main__":
    speedTest()
