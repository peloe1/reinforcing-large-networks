import networkx as nx
import numpy as np
from path import feasible_paths
from portfolio import generate_feasible_portfolios, dominates_with_cost
from subnetwork import cost_efficient_portfolios
import itertools
from performance import expected_travel


# TODO: Double check that this works and optimize it
def decomposition_solver(G: list[nx.Graph],
                         T: list[tuple[int, int]],
                         subnetworks: list[nx.Graph], 
                         terminal_pair_sets: list[list[tuple[int, int]]], 
                         travel_volumes: dict[tuple[int, int], float],
                         reinforcement_actions: list[list[tuple[int, float, float]]],
                         budget: list[float]) -> set[list[int]]:
    r = len(budget)
    k = len(subnetworks)

    all_paths = feasible_paths(G, T)

    Q_CE = []
    portfolio_costs = []
    #portfolio_performances = []

    for graph, terminal_pairs, actions in zip(subnetworks, terminal_pair_sets, reinforcement_actions):
        N = len(actions)
        action_costs: list[float] = []
        node_reinforcements: list[tuple[int, float]] = []
        for node, cost, prob in actions:
            action_costs.append(cost)
            node_reinforcements.append((node, prob))

        # Step 1
        feasible_portfolios, costs = generate_feasible_portfolios(N, action_costs, budget)
        paths = feasible_paths(graph, terminal_pairs)

        # Step 2
        Q_CE_j, _ = cost_efficient_portfolios(graph, paths, node_reinforcements, feasible_portfolios, costs)
        Q_CE.append(Q_CE_j)
        portfolio_costs.append(costs)
    
    # Step 3
    Q_star = set()
    combined_performances = {}
    combined_costs = {}

    for Q in Q_CE[0]:
        G_Q = G.copy()
        Q = [Q] + [0 for _ in range(2, k)]
        # This for loops applies portfolio 'Q' to 'G_Q'
        # the kth reinforcement action increases the reliability of 'node' to 'prob'
        for i, (node, prob) in enumerate(reinforcement_actions[j]):
            if (Q[0] >> i) & 1: # True if the ith bit of the portfolio corresponding to the 0th subnetwork is one
                G_Q.nodes[node]['reliability'] = prob
        
        Q_star.add(Q)
        combined_performances[tuple(Q)] = expected_travel(G_Q, all_paths, travel_volumes)
        combined_costs[tuple(Q)] = portfolio_costs[Q[0]]

    # Step 4
    for j in range(2, k):
        Q_j = set()
        for Q in Q_star:
            for l, q_j in enumerate(Q_CE[j]):
                cost_vector = [portfolio_costs[q_j][i] + combined_costs[tuple(Q)][i] for i in range(r)]
                if all(cost_vector[i] <= budget[i] for i in range(r)):
                    # Step 5
                    Q_copy = Q.copy()
                    Q_copy[l] = q_j
                    Q_j.add(Q_copy)
                    combined_costs[tuple(Q_copy)] = cost_vector

                    # Step 6
                    G_Q = G.copy()
                    for i in range(k): # Reinforcements to the ith subnetwork's nodes
                        for m, (node, prob) in enumerate(reinforcement_actions[j]):
                            if (Q_copy[i] >> m) & 1: # True if the kth bit of the portfolio corresponding to the jth subnetwork is one
                                G_Q.nodes[node]['reliability'] = prob
                                
                    combined_performances[tuple(Q_copy)] = expected_travel(G_Q, all_paths, travel_volumes)
        # Step 7

        # Do we still take cost into account? Probably not as long as the combined portfolio is feasible. In this case the lines below need to be modified to not take cost into account
        dominated = set(filter(lambda Q1: any(dominates_with_cost(combined_performances[tuple(Q2)], combined_performances[tuple(Q1)], combined_costs[tuple(Q2)], combined_costs[tuple(Q1)]) for Q2 in Q_star), Q_j))
        Q_j = Q_j.difference(dominated)

        dominated_previous = set(filter(lambda Q1: any(dominates_with_cost(combined_performances[tuple(Q2)], combined_performances[tuple(Q1)], combined_costs[tuple(Q2)], combined_costs[tuple(Q1)]) for Q2 in Q_j), Q_star))
        Q_star = Q_star.difference(dominated_previous)

        Q_star.update(Q_j)
    # Step 9 and 10
    return Q_star, combined_performances, combined_costs



                    
            


if __name__ == '__main__':
    print("This file is not meant to be run directly!")

