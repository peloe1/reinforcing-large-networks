import networkx as nx
import time
import numpy as np
import random

from information_set import compute_extreme_points
from portfolio import portfolio_as_bitmask
from ce_portfolios import costEfficientPortfolios, costEfficientPortfolios_optimized 
from path import simple_paths, terminal_pairs
from path import feasible_paths
from graph import construct_graph, generate_random_graph_with_positions



def main() -> None:
    preferences = np.array([1,1,1]) # Start with no preference between the two metrics

    extremePoints = np.array(compute_extreme_points(preferences))
    IP = False
    if IP:
        # Example network from Ip. and Wang.
        G = nx.Graph()
        G.add_node(0, reliability=0.8)
        G.add_node(1, reliability=0.8)
        G.add_node(2, reliability=0.8)
        G.add_node(3, reliability=0.8)
        G.add_node(4, reliability=0.8)
        G.add_node(5, reliability=0.8)
        G.add_node(6, reliability=0.8)
        G.add_node(7, reliability=0.8)
        G.add_node(8, reliability=0.8)
        G.add_node(9, reliability=0.8)

        G.add_edge(0,1)
        G.add_edge(1,2)
        G.add_edge(1,3)
        G.add_edge(2,3)
        G.add_edge(3,4)
        G.add_edge(2,5)
        G.add_edge(3,5)
        G.add_edge(2,7)
        G.add_edge(5,6)
        G.add_edge(6,7)
        G.add_edge(6,8)
        G.add_edge(7,9)

        terminal_nodes = [1, 9, 3]

        for node in terminal_nodes:
            G.nodes[node]['reliability'] = 1

        terminal_node_pairs = terminal_pairs(terminal_nodes)
        traffic_volumes = {t: 1.0 for t in terminal_node_pairs}

        #paths = feasible_paths(G, terminal_node_pairs)
        paths = simple_paths(G, terminal_node_pairs)


        node_reinforcements = [(i, 0.9) for i in range(10)]
        costs = [1.0 for _ in range(10)]
        budget = 10.0

        start = time.time()
        Q, portfolio_costs = costEfficientPortfolios(G, terminal_node_pairs, paths, traffic_volumes, extremePoints, node_reinforcements, costs, budget)
        end = time.time()
        print(f"Time to compute cost-efficient portfolios: {(end - start):.2f}")
        print(f"Number of cost-efficient portfolios: {len(Q)}")
        print(f"Costs of the cost-efficient portfolios: {portfolio_costs}")

    SIJ = True
    if SIJ:
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
        Q_1, portfolio_costs = costEfficientPortfolios(G, terminal_node_pairs, paths, traffic_volumes, extremePoints, node_reinforcements, costs, budget, False)
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
        Q_2, portfolio_costs = costEfficientPortfolios_optimized(G, terminal_node_pairs, paths, traffic_volumes, extremePoints, node_reinforcements, costs, budget, False)
        end = time.time()
        if end - start > 60:
            print(f"Time to compute cost-efficient portfolios: {(end - start)/60:.2f} minutes")
        else:
            print(f"Time to compute cost-efficient portfolios: {(end - start):.2f} seconds")

        print(f"Number of resulting cost-efficient portfolios: {len(Q_2)}")

        Q_1_set = set([portfolio_as_bitmask(q) for q in Q_1])
        Q_2_set = set(Q_2)
        print(f"The two sets of cost-efficient portfolios are equal: {Q_1_set == Q_2_set}")

    RANDOM = True
    if RANDOM:
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