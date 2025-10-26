import networkx as nx
import time
import numpy as np
import random

from information_set import compute_extreme_points
from portfolio import portfolio_as_bitmask
from ce_portfolios import cost_efficient_portfolios_old, cost_efficient_portfolios
from path import simple_paths, terminal_pairs
from path import feasible_paths
from graph import construct_graph, generate_random_graph_with_positions



def speedTest() -> None:
    preferences = np.array([1,1,1]) # Start with no preference between the two metrics

    extremePoints = np.array(compute_extreme_points(preferences))

    filename = 'data/network/sij.json'
    num_nodes = 40

    reliabilities = {i: 0.99 for i in range(num_nodes)}
    terminal_nodes = [2, 11, 32] # East, West, South
    for node in terminal_nodes:
        reliabilities[node] = 1.0
    
    G = construct_graph(filename, reliabilities)


    for node in terminal_nodes:
        G.nodes[node]['reliability'] = 1

    terminal_node_pairs = terminal_pairs(terminal_nodes)
    traffic_volumes = {t: 100.0 for t in terminal_node_pairs}

    paths = feasible_paths(G, terminal_node_pairs)

    r = 20

    node_reinforcements = [(i, 0.995) for i in range(r)]
    costs = [1.0 for _ in range(r)]
    #budget = r / 2.5
    budget = 12
    print("-----------------------------------------")
    print("Original version with lists of integers")
    start = time.time()
    Q_1, portfolio_costs = cost_efficient_portfolios_old(G, terminal_node_pairs, paths, traffic_volumes, extremePoints, node_reinforcements, costs, budget, False)
    #Q_1 = []
    end = time.time()
    if end - start > 60:
        print(f"Time to compute cost-efficient portfolios: {(end - start)/60:.2f} minutes")
    else:
        print(f"Time to compute cost-efficient portfolios: {(end - start):.2f} seconds")

    print(f"Number of resulting cost-efficient portfolios: {len(Q_1)}")

    print("-----------------------------------------")
    print("New version with binary representation")
    start = time.time()
    Q_2, portfolio_costs = cost_efficient_portfolios(G, terminal_node_pairs, paths, traffic_volumes, extremePoints, node_reinforcements, costs, budget, False)
    end = time.time()
    if end - start > 60:
        print(f"Time to compute cost-efficient portfolios: {(end - start)/60:.2f} minutes")
    else:
        print(f"Time to compute cost-efficient portfolios: {(end - start):.2f} seconds")

    print(f"Number of resulting cost-efficient portfolios: {len(Q_2)}")

    Q_1_set = set([portfolio_as_bitmask(q) for q in Q_1])
    Q_2_set = set(Q_2)
    print(f"The two sets of cost-efficient portfolios are equal: {Q_1_set == Q_2_set}")
    return


if __name__ == "__main__":
    speedTest()
