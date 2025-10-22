import networkx as nx
import numpy as np
import time
import multiprocessing as mp
from multiprocessing import Manager

from portfolio import dominates_with_cost, generate_feasible_portfolios_old, generate_feasible_portfolios
from performance import utility_functions


def bitmask_to_portfolio(mask: int, r: int) -> list[int]:
    return [(mask >> i) & 1 for i in range(r)]

# TODO: Make this work for general node reinforcement indices, not just 0, 1, ..., r
def cost_efficient_portfolios(G: nx.Graph, paths: dict[tuple[int, int], list[np.ndarray]], node_reinforcements: list[tuple[int, float]], Q_F: set[int], portfolio_costs: dict[int, float], verbose=False) -> tuple[set[int], dict[int, list[float]]]:
    """
    Parameters:
        G (networkx.Graph): The graph in its original state with no portfolios applied and no disruptions.
        extremePoints (np.array): List of extreme points of the set of feasible weights.
        nodeReinforcements (dict[int, float]): A dictionary where the key is the index of the node which can be reinforced and the value is the resulting reliability.
        costs (list[float]): The costs of the reinforcement actions.

    Returns:    
        Q (list[tuple[int, ..., int]]): The set of cost-efficient portfolios.
    """
    
    r = len(node_reinforcements)
    Q: set[int] = set([0])

    performance: list[float] = utility_functions(G, paths)
    performances: dict[int, list[float]] = {0: performance}

    for l in range(r):
        start = time.time()
        #print("---------------------------------")
        #print(f"Iteration {l+1}/{r}: ")
        
        newQ = set()
        mask = ~(1 << l)  # Bitmask to zero out the lth bit
        for q1 in Q_F:
            if (q1 >> l) & 1: # True if the lth bit is one
                q1_masked = q1 & mask
                if any((q2 & mask) == q1_masked for q2 in Q):
                    newQ.add(q1)
 
        #print(f"Number of portfolios considered in Q_{l+1}: {len(newQ)}")

        for portfolio in newQ:
            G_q = G.copy()

            # This for loops applies 'portfolio' to 'G_q'
            # the kth reinforcement action increases the reliability of 'node' to 'prob'
            for k, (node, prob) in enumerate(node_reinforcements):
                if (portfolio >> k) & 1: # True if the kth bit is one
                    G_q.nodes[node]['reliability'] = prob


            # Next we can calculate the expected performances of G_q
            performance = utility_functions(G_q, paths)
            performances[portfolio] = performance # Step 6

        # The approach from Joaquin's paper
        dominated = set(filter(lambda q1: any(dominates_with_cost(performances[q2], performances[q1], portfolio_costs[q2], portfolio_costs[q1]) for q2 in Q), newQ))
        newQ = newQ.difference(dominated)

        dominated_previous = set(filter(lambda q1: any(dominates_with_cost(performances[q2], performances[q1], portfolio_costs[q2], portfolio_costs[q1]) for q2 in newQ), Q))
        Q = Q.difference(dominated_previous)

        Q.update(newQ)

        #print(f"Number of portfolios in Q^{l+1}: {len(Q)}")
        if verbose:
            print(f"Q^{l+1}:")
            for portfolio in Q:
                print(bitmask_to_portfolio(portfolio, r))

        end = time.time()
        print(f"Time for iteration {l+1}: {(end - start):.2f} seconds.")

    return Q, performances