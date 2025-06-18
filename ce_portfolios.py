import networkx as nx
import numpy as np
import time

from portfolio import dominatesWithCost, feasiblePortfolios, costEfficient, generateFeasiblePortfolios
from performance import expected_traffic_volumes
from path import terminal_pairs
from path import feasible_paths


def costEfficientPortfolios(G: nx.Graph, pairs: list[tuple[int, int]], paths: dict[tuple[int, int], list[np.ndarray]], traffic_volumes: dict[tuple[int, int], float], extreme_points: np.ndarray, node_reinforcements: list[tuple[int, float]], costs: list[float], budget: float) -> tuple[list[list[int]], dict[tuple[int, ...], list[float]]]:
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
    #Q_F, portfolio_costs = feasiblePortfolios(r, costs, budget)
    Q_F, portfolio_costs = generateFeasiblePortfolios(r, costs, budget)
    end = time.time()
    #print(f"Q_F: {Q_F}")
    #print(f"Costs: {portfolio_costs}")
    print(f"Time to compute feasible portfolios: {(end - start):.2f}")
    print(f"Number of feasible portfolios: {len(Q_F)}")
    
    Q = [[0 for _ in range(r)]]
    print(f"Q^0: {Q}")

    start = time.time()
    performance = expected_traffic_volumes(G, pairs, paths, traffic_volumes, extreme_points)
    performances: dict[tuple[int, ...], list[float]] = {tuple(Q[0]): performance}

    end = time.time()
    print(f"Time to compute the expected performances for [0,...,0]: {(end - start):.2f} seconds.")

    for l in range(r):
        start = time.time()
        print("---------------------------------")
        print(f"Iteration {l+1}/{r}: ")

        #indexes = np.array([k for k in range(r) if k != l], dtype=int)
        #newQ = list(filter(lambda q1: not all(not np.all(q1[indexes] == q2[indexes]) or q1[l] != 1 for q2 in Q), Q_F))

        indexes = [k for k in range(r) if k != l]
        newQ = list(filter(lambda q1: not all(not (all(q1[k] == q2[k] for k in indexes)) or q1[l] != 1 for q2 in Q), Q_F)) # Step 5, newQ = Q^l

        print(f"Number of portfolios considered in Q_{l+1}: {len(newQ)}")

        for portfolio in newQ:
            G_q = G.copy()

            # This for loops applies 'portfolio' to 'G_q'
            for k, (node, prob) in enumerate(node_reinforcements):
                if portfolio[k] == 1:
                    G_q.nodes[node]['reliability'] = prob


            # Next we can calculate the expected performances of G_q
            performance = expected_traffic_volumes(G_q, pairs, paths, traffic_volumes, extreme_points)
            performances[tuple(portfolio)] = performance # Step 6

        # Previous approach
        #new_cost_efficient = list(filter(lambda q1: all(not dominatesWithCost(performances[tuple(q2)], performances[tuple(q1)], portfolio_costs[tuple(q2)], portfolio_costs[tuple(q1)]) for q2 in q_0), newQ)) # Step 7
        #previous_cost_efficient = list(filter(lambda q1: all(not dominatesWithCost(performances[tuple(q2)], performances[tuple(q1)], portfolio_costs[tuple(q2)], portfolio_costs[tuple(q1)]) for q2 in new_cost_efficient), q_0)) # Step 8 part 1

        #new_cost_efficient.extend(previous_cost_efficient)
        # q_0 = new_cost_efficient


        # The approach from Joaquin's paper
        dominated = list(filter(lambda q1: any(dominatesWithCost(performances[tuple(q2)], performances[tuple(q1)], portfolio_costs[tuple(q2)], portfolio_costs[tuple(q1)]) for q2 in Q), newQ))
        newQ = [q for q in newQ if q not in dominated]

        dominated_previous = list(filter(lambda q1: any(dominatesWithCost(performances[tuple(q2)], performances[tuple(q1)], portfolio_costs[tuple(q2)], portfolio_costs[tuple(q1)]) for q2 in newQ), Q))
        Q = [q for q in Q if q not in dominated_previous]

        Q.extend(newQ)
        
        #q_0 = list(set(tuple(q) for q in q_0)) # Removes duplicates
        #q_0 = list(filter(lambda q1: all(not dominatesWithCost(performances[tuple(q2)], performances[tuple(q1)], portfolio_costs[tuple(q2)], portfolio_costs[tuple(q1)]) for q2 in q_0), q_0)) # Added this line, which compares each portfolio with itself

        print(f"Number of portfolios in Q^{l+1}: {len(Q)}")
        print(f"Q^{l+1}:")
        for portfolio in Q:
            print(portfolio)

        end = time.time()
        print(f"Time for iteration {l+1}: {(end - start):.2f} seconds.")

    #Q = list(filter(lambda q1: all(not dominatesWithCost(performances[tuple(q2)], performances[tuple(q1)], portfolio_costs[tuple(q2)], portfolio_costs[tuple(q1)]) for q2 in Q), Q)) # One last time to make sure no portfolios are dominated

    return Q, performances

if __name__ == "__main__":
    print("This file is not meant to be run directly!")

    Q_F = feasiblePortfolios(4, [1, 1, 1, 1], 2)
    q_0 = [[0,0,0,0]]
    r = 4
    l = 1
    
    indexes = [k for k in range(r) if k != l]
    newQ = list(filter(lambda q1: all(all(q1[k] == q2[k] for k in indexes) and q1[l] == 1 for q2 in q_0), Q_F)) # Step 5, newQ = Q^l
