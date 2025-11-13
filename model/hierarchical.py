import networkx as nx
from path import feasible_paths
from performance import expected_travel_hierarchical
from portfolio import dominates_with_cost
import time
import numpy as np


# TODO: Double check that this works and optimize it
def cost_efficient_combined_portfolios(partitioned_paths: dict[tuple[str, str], tuple[list[tuple[str, str]], list[str]]], 
                                       terminal_pair_reliabilities: dict[str, dict[int, dict[tuple[str, str], float]]], 
                                       travel_volumes: dict[tuple[str, str], float], 
                                       Q_CE: dict[str, set[int]],
                                       subnetworks: list[str],
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

    start = time.time()
    j = 0
    for Q in Q_CE[subnetworks[j]]:
        cost_dict = portfolio_costs[subnetworks[j]]
        Q = [Q] + [0 for _ in range(1, k)]
        # This for loops applies portfolio 'Q' to 'G_Q'
        # the kth reinforcement action increases the reliability of 'node' to 'prob'
        
        #G_Q = G.copy()
        #for i, (node, prob) in enumerate(node_reinforcements[subnetworks[j]]):
        #    if (Q[0] >> i) & 1: # True if the ith bit of the portfolio corresponding to the 0th subnetwork is one
        #        G_Q.nodes[node]['reliability'] = prob
        
        Q_star.add(tuple(Q))
        combined_performances[tuple(Q)] = expected_travel_hierarchical(Q, subnetworks, partitioned_paths, terminal_pair_reliabilities, travel_volumes)
        combined_costs[tuple(Q)] = cost_dict[Q[0]]
    
    end = time.time()
    print(f"Time for iteration {j+1}: {(end - start):.2f} seconds.")
    #print("Combined portfolios: ", Q_star)

    # Step 4 for loop
    for j in range(1, k):
        start = time.time()
        cost_dict = portfolio_costs[subnetworks[j]]
        Q_j: set[tuple[int, ...]] = set()
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

                    ## Step 6
                    #G_Q = G.copy()
                    ##print(f"Applying portfolio {Q_copy}")
                    #for i in range(k):
                    #    #print(f"  Subnetwork {subnetworks[i]}: portfolio {Q_copy[i]}")
                    #    for m, (node, prob) in enumerate(node_reinforcements[subnetworks[i]]):
                    #        if (Q_copy[i] >> m) & 1:
                    #            #print(f"    Reinforcing node {node} from {G.nodes[node].get('reliability', 'N/A')} to {prob}")
                    #            G_Q.nodes[node]['reliability'] = prob
                    #for i in range(k):#j+1): # range(k) # looping all the way to k is redundant since we know that Q_copy[j+1], ... Q_copy[k-1] = 0
                    #    # Reinforcements to the ith subnetwork's nodes
                    #    for m, (node, prob) in enumerate(node_reinforcements[subnetworks[i]]):
                    #        if (Q_copy[i] >> m) & 1: # True if the mth bit of the portfolio corresponding to the ith subnetwork is one
                    #            G_Q.nodes[node]['reliability'] = prob
                                
                    combined_performances[tuple(Q_copy)] = expected_travel_hierarchical(Q_copy, subnetworks, partitioned_paths, terminal_pair_reliabilities, travel_volumes)
        # Step 7
        # dominates instead of dominates_with_cost
        dominated = set(filter(lambda Q1: any(dominates_with_cost(combined_performances[Q2], combined_performances[Q1], combined_costs[Q1], combined_costs[Q2]) for Q2 in Q_star), Q_j))
        #print("Dominated portfolios:")
        #for p in dominated:
        #    print(f"Portfolio {p} with performance {combined_performances[p]}")
        
        Q_j = Q_j.difference(dominated)

        # Step 8
        dominated_previous = set(filter(lambda Q1: any(dominates_with_cost(combined_performances[Q2], combined_performances[Q1], combined_costs[Q1], combined_costs[Q2]) for Q2 in Q_j), Q_star))
        #print("Dominated portfolios from previous:")
        #for p in dominated_previous:
        #    print(f"Portfolio {p} with performance {combined_performances[p]}")
        Q_star = Q_star.difference(dominated_previous)

        # Do we still take cost into account? Probably not as long as the combined portfolio is feasible. In this case the lines below need to be modified to not take cost into account
        # The approach below takes cost into account
        #dominated = set(filter(lambda Q1: any(dominates_with_cost(combined_performances[tuple(Q2)], combined_performances[tuple(Q1)], combined_costs[tuple(Q2)], combined_costs[tuple(Q1)]) for Q2 in Q_star), Q_j))
        #Q_j = Q_j.difference(dominated)

        #dominated_previous = set(filter(lambda Q1: any(dominates_with_cost(combined_performances[tuple(Q2)], combined_performances[tuple(Q1)], combined_costs[tuple(Q2)], combined_costs[tuple(Q1)]) for Q2 in Q_j), Q_star))
        #Q_star = Q_star.difference(dominated_previous)
        
        # Step 9
        Q_star.update(Q_j)
        
        #dominated_new = set(filter(lambda Q1: any(dominates_with_cost(combined_performances[Q2], combined_performances[Q1], combined_costs[Q1], combined_costs[Q2]) for Q2 in Q_star), Q_star))
        #Q_star = Q_star.difference(dominated_new)
        
        end = time.time()
        print(f"Time for iteration {j+1}: {(end - start):.2f} seconds.")
        print("Number of non-dominated portfolios: ", len(Q_star))
        #perf = np.array([combined_performances[portfolio] for portfolio in Q_star])
        #print("Average performance: ", np.mean(perf))
        #print("Variance of the performances: ", np.var(perf))
        #print("Non-dominated portfolios:")
        #for p in Q_star:
        #    print(f"Portfolio {p} with performance {combined_performances[p]}")
        
        #print("-"*50)
        #print("\n")
        
    # Step 11 and 12
    return Q_star, combined_performances, combined_costs


if __name__ == '__main__':
    print("This file is not meant to be run directly!")

