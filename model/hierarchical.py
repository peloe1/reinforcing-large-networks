import networkx as nx
from model.path import feasible_paths
from portfolio import generate_feasible_portfolios
from subnetwork import cost_efficient_portfolios
from performance import expected_travel


# TODO: Double check that this works and optimize it
def cost_efficient_combined_portfolios( G: nx.Graph,
                                        G_original: nx.Graph,
                                        terminal_pairs: list[tuple[str, str]],
                                        travel_volumes: dict[tuple[str, str], float],
                                        Q_CE: dict[str, set[int]],
                                        subnetworks: list[str],
                                        node_reinforcements: dict[str, list[tuple[str, float]]],
                                        portfolio_costs: dict[str, dict[int, list[float]]],
                                        budget: list[float],
                                        k: int) -> tuple[set[tuple[int, ...]], 
                                                       dict[tuple[int, ...], float], 
                                                       dict[tuple[int, ...], list[float]]
                                                       ]:
    r = len(budget)
    
    # Step 3
    Q_star: set[tuple[int, ...]] = set()
    combined_performances: dict[tuple[int, ...], float] = {}
    combined_costs: dict[tuple[int, ...], list[float]] = {}

    path_list = feasible_paths(G, G_original, terminal_pairs)

    j = 0
    for Q in Q_CE[subnetworks[j]]:
        cost_dict = portfolio_costs[subnetworks[j]]
        G_Q = G.copy()
        Q = [Q] + [0 for _ in range(2, k)]
        # This for loops applies portfolio 'Q' to 'G_Q'
        # the kth reinforcement action increases the reliability of 'node' to 'prob'
        for i, (node, prob) in enumerate(node_reinforcements[subnetworks[j]]):
            if (Q[0] >> i) & 1: # True if the ith bit of the portfolio corresponding to the 0th subnetwork is one
                G_Q.nodes[node]['reliability'] = prob
        
        Q_star.add(tuple(Q))
        combined_performances[tuple(Q)] = expected_travel(G_Q, path_list, travel_volumes)
        combined_costs[tuple(Q)] = cost_dict[Q[0]]

    # Step 4 for loop
    for j in range(1, k):
        cost_dict = portfolio_costs[subnetworks[j]]
        Q_j = set()
        # Step 5
        # Constructing Q^*_j
        for Q in Q_star: # Loop through the portfolios in Q^*_j-1
            # Try adding each q^j in Q^j_CE to it and if feasible keep it
            for q_j in Q_CE[subnetworks[j]]: # Q_CE[j] = Q^j_CE
                cost_vector = [combined_costs[tuple(Q)][i] + cost_dict[q_j][i] for i in range(r)]
                if all(cost_vector[i] <= budget[i] for i in range(r)):
                    Q_copy = list(Q).copy()
                    Q_copy[j] = q_j
                    Q_j.add(tuple(Q_copy))
                    combined_costs[tuple(Q_copy)] = cost_vector

                    # Step 6
                    G_Q = G.copy()
                    for i in range(j+1): # range(k) # looping all the way to k is redundant since we know that Q_copy[j+1], ... Q_copy[k-1] = 0
                        # Reinforcements to the ith subnetwork's nodes
                        for m, (node, prob) in enumerate(node_reinforcements[subnetworks[i]]):
                            if (Q_copy[i] >> m) & 1: # True if the mth bit of the portfolio corresponding to the ith subnetwork is one
                                G_Q.nodes[node]['reliability'] = prob
                                
                    combined_performances[tuple(Q_copy)] = expected_travel(G_Q, path_list, travel_volumes)
        # Step 7
        # dominates instead of dominates_with_cost
        dominated = set(filter(lambda Q1: any(combined_performances[tuple(Q2)] > combined_performances[tuple(Q1)] for Q2 in Q_star), Q_j))
        Q_j = Q_j.difference(dominated)

        # Step 8
        dominated_previous = set(filter(lambda Q1: any(combined_performances[tuple(Q2)] > combined_performances[tuple(Q1)] for Q2 in Q_j), Q_star))
        Q_star = Q_star.difference(dominated_previous)

        # Do we still take cost into account? Probably not as long as the combined portfolio is feasible. In this case the lines below need to be modified to not take cost into account
        # The approach below takes cost into account
        #dominated = set(filter(lambda Q1: any(dominates_with_cost(combined_performances[tuple(Q2)], combined_performances[tuple(Q1)], combined_costs[tuple(Q2)], combined_costs[tuple(Q1)]) for Q2 in Q_star), Q_j))
        #Q_j = Q_j.difference(dominated)

        #dominated_previous = set(filter(lambda Q1: any(dominates_with_cost(combined_performances[tuple(Q2)], combined_performances[tuple(Q1)], combined_costs[tuple(Q2)], combined_costs[tuple(Q1)]) for Q2 in Q_j), Q_star))
        #Q_star = Q_star.difference(dominated_previous)
        
        # Step 9
        Q_star.update(Q_j)
        
    # Step 11 and 12
    return Q_star, combined_performances, combined_costs


if __name__ == '__main__':
    print("This file is not meant to be run directly!")

