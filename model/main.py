import networkx as nx
import time
import numpy as np
import random

from information_set import compute_extreme_points
from portfolio import generate_feasible_portfolios
from subnetwork import cost_efficient_portfolios
from path import terminal_pairs, feasible_paths
from graph import construct_graph, generate_random_graph_with_positions



def main() -> None:
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
    
    travel_volumes = {pair: 100 for pair in terminal_node_pairs}

    paths = feasible_paths(G, terminal_node_pairs)

    r = 32
    node_reinforcements = [(i, 0.995) for i in range(r)]
    costs = {i: [1] for i in range(r)}
    budget = [10]

    print("Feasible portfolios")

    start = time.time()
    Q_F, feasible_portfolio_costs = generate_feasible_portfolios(r, costs, budget)
    end = time.time()
    print(f"It took {(end - start):.2f} seconds to compute the {len(Q_F)} feasible portfolios")
    

    print("-----------------------------------------")
    print("New version with binary representation")
    start = time.time()
    Q_CE, portfolio_costs = cost_efficient_portfolios(G, paths, node_reinforcements, Q_F, feasible_portfolio_costs, travel_volumes, False)
    end = time.time()
    if end - start > 60:
        print(f"Time to compute cost-efficient portfolios: {(end - start)/60:.2f} minutes")
    else:
        print(f"Time to compute cost-efficient portfolios: {(end - start):.2f} seconds")

    print(f"Number of resulting cost-efficient portfolios: {len(Q_CE)}")

    return
    

if __name__ == "__main__":
    main()



"""
RESULTS FOR r=25 with the new optimized version

Original version with lists of integers
It took 336.13 seconds to compute the 7119516 feasible portfolios
Time to compute cost-efficient portfolios: 18.38 minutes
Number of resulting cost-efficient portfolios: 73
-----------------------------------------
New version with binary representation
It took 360.09 seconds to compute the 7119516 feasible portfolios
Time to compute cost-efficient portfolios: 7.09 minutes
Number of resulting cost-efficient portfolios: 73
The two sets of cost-efficient portfolios are equal: True
"""

"""
With r=20, and b = r / 2.5
-----------------------------------------
Original version with lists of integers
It took 6.66 seconds to compute the 263950 feasible portfolios
Time to compute cost-efficient portfolios: 18.64 seconds
Number of resulting cost-efficient portfolios: 38
-----------------------------------------
New version with binary representation
It took 3.02 seconds to compute the 263950 feasible portfolios
Time to compute cost-efficient portfolios: 4.33 seconds
Number of resulting cost-efficient portfolios: 38
The two sets of cost-efficient portfolios are equal: True
"""

"""
With r = 20, b = 12

Original version with lists of integers
It took 30.27 seconds to compute the 910596 feasible portfolios
Time to compute cost-efficient portfolios: 1.14 minutes
Number of resulting cost-efficient portfolios: 42
-----------------------------------------
New version with binary representation
It took 12.93 seconds to compute the 910596 feasible portfolios
Time to compute cost-efficient portfolios: 17.95 seconds
Number of resulting cost-efficient portfolios: 42
The two sets of cost-efficient portfolios are equal: True

"""