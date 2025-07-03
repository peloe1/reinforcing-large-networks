import networkx as nx
import numpy as np
import time
import multiprocessing as mp
from multiprocessing import Manager

from portfolio import dominates_with_cost, generate_feasible_portfolios_old, generate_feasible_portfolios
from performance import expected_traffic_volumes

# Old slow version using lists of integers representing portfolios
def cost_efficient_portfolios_old(G: nx.Graph, paths: dict[tuple[int, int], list[np.ndarray]], traffic_volumes: dict[tuple[int, int], float], extreme_points: np.ndarray, node_reinforcements: list[tuple[int, float]], costs: list[float], budget: float, verbose=False) -> tuple[list[list[int]], dict[tuple[int, ...], list[float]]]:
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

    start = time.time()
    Q_F, portfolio_costs = generate_feasible_portfolios_old(r, costs, budget)
    end = time.time()
    print(f"It took {(end - start):.2f} seconds to compute the {len(Q_F)} feasible portfolios")
    
    Q = [[0 for _ in range(r)]]
    #print(f"Q^0: {Q}")

    performance = expected_traffic_volumes(G, paths, traffic_volumes, extreme_points)
    performances: dict[tuple[int, ...], list[float]] = {tuple(Q[0]): performance}

    for l in range(r):
        start = time.time()
        #print("---------------------------------")
        #print(f"Iteration {l+1}/{r}: ")

        indexes = [k for k in range(r) if k != l]
        newQ = list(filter(lambda q1: not all(not (all(q1[k] == q2[k] for k in indexes)) or q1[l] != 1 for q2 in Q), Q_F)) # Step 5, newQ = Q^l

        #print(f"Number of portfolios considered in Q_{l+1}: {len(newQ)}")

        for portfolio in newQ:
            G_q = G.copy()

            # This for loops applies 'portfolio' to 'G_q'
            for k, (node, prob) in enumerate(node_reinforcements):
                if portfolio[k] == 1:
                    G_q.nodes[node]['reliability'] = prob


            # Next we can calculate the expected performances of G_q
            performance = expected_traffic_volumes(G_q, paths, traffic_volumes, extreme_points)
            performances[tuple(portfolio)] = performance # Step 6

        # The approach from Joaquin's paper
        dominated = list(filter(lambda q1: any(dominates_with_cost(performances[tuple(q2)], performances[tuple(q1)], portfolio_costs[tuple(q2)], portfolio_costs[tuple(q1)]) for q2 in Q), newQ))
        newQ = [q for q in newQ if q not in dominated]

        dominated_previous = list(filter(lambda q1: any(dominates_with_cost(performances[tuple(q2)], performances[tuple(q1)], portfolio_costs[tuple(q2)], portfolio_costs[tuple(q1)]) for q2 in newQ), Q))
        Q = [q for q in Q if q not in dominated_previous]

        Q.extend(newQ)

        #print(f"Number of portfolios in Q^{l+1}: {len(Q)}")
        if verbose:
            print(f"Q^{l+1}:")
            for portfolio in Q:
                print(portfolio)

        end = time.time()
        if verbose:
            print(f"Time for iteration {l+1}: {(end - start):.2f} seconds.")

    return Q, performances

def bitmask_to_portfolio(mask: int, r: int) -> list[int]:
    return [(mask >> i) & 1 for i in range(r)]

# TODO: Make this work for general node reinforcement indices, not just 0, 1, ..., r
def cost_efficient_portfolios(G: nx.Graph, paths: dict[tuple[int, int], list[np.ndarray]], traffic_volumes: dict[tuple[int, int], float], extreme_points: np.ndarray, node_reinforcements: list[tuple[int, float]], Q_F: set[int], portfolio_costs: dict[int, float], verbose=False) -> tuple[set[int], dict[int, list[float]]]:
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

    performance = expected_traffic_volumes(G, paths, traffic_volumes, extreme_points)
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
            performance = expected_traffic_volumes(G_q, paths, traffic_volumes, extreme_points)
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