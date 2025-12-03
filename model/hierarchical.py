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
                                                         dict[tuple[int, ...], list[float]],
                                                         dict[tuple[int, ...], dict[tuple[str, str]]]
                                                        ]:
    r = len(budget)
    
    # Step 3
    Q_star: set[tuple[int, ...]] = set()
    combined_performances: dict[tuple[int, ...], float] = {}
    combined_costs: dict[tuple[int, ...], list[float]] = {}

    dict_reliabilities: dict[tuple[int, ...], dict[tuple[str, str]]] = {}

    start = time.time()
    j = 0
    for Q in Q_CE[subnetworks[j]]:
        cost_dict = portfolio_costs[subnetworks[j]]
        Q = [Q] + [0 for _ in range(1, k)]

        Q_star.add(tuple(Q))
        combined_performances[tuple(Q)] = expected_travel_hierarchical(Q, subnetworks, partitioned_paths, terminal_pair_reliabilities, travel_volumes)
        combined_costs[tuple(Q)] = cost_dict[Q[0]]
    
    end = time.time()
    print(f"Time for iteration {j+1}: {(end - start):.2f} seconds.")

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
                    performance, reliability_dict = expected_travel_hierarchical(Q_copy, subnetworks, partitioned_paths, terminal_pair_reliabilities, travel_volumes)
                    combined_performances[tuple(Q_copy)] = performance

                    dict_reliabilities[tuple(Q_copy)] = reliability_dict
        # Step 7
        dominated = set(filter(lambda Q1: any(dominates_with_cost(combined_performances[Q2], combined_performances[Q1], combined_costs[Q2], combined_costs[Q1]) for Q2 in Q_star), Q_j))
        Q_j = Q_j.difference(dominated)

        # Step 8
        dominated_previous = set(filter(lambda Q1: any(dominates_with_cost(combined_performances[Q2], combined_performances[Q1], combined_costs[Q2], combined_costs[Q1]) for Q2 in Q_j), Q_star))
        Q_star = Q_star.difference(dominated_previous)

        # Step 9
        Q_star.update(Q_j)
        
        #dominated_new = set(filter(lambda Q1: any(dominates_with_cost(combined_performances[Q2], combined_performances[Q1], combined_costs[Q1], combined_costs[Q2]) for Q2 in Q_star), Q_star))
        #Q_star = Q_star.difference(dominated_new)
        
        end = time.time()
        print(f"Time for iteration {j+1}: {(end - start):.2f} seconds.")
        print("Number of non-dominated portfolios: ", len(Q_star))
    
    dominated = set(filter(lambda Q1: any(dominates_with_cost(combined_performances[Q2], combined_performances[Q1], combined_costs[Q2], combined_costs[Q1]) for Q2 in Q_star), Q_star))
    Q_star = Q_star.difference(dominated)
    # Step 11 and 12
    return Q_star, combined_performances, combined_costs, dict_reliabilities


def random_portfolios(partitioned_paths: dict[tuple[str, str], tuple[list[tuple[str, str]], list[str]]], 
                                terminal_pair_reliabilities: dict[str, dict[int, dict[tuple[str, str], float]]], 
                                travel_volumes: dict[tuple[str, str], float], 
                                portfolios: dict[str, set[int]],
                                subnetworks: list[str],
                                portfolio_costs: dict[str, dict[int, list[float]]],
                                budget: list[float],
                                k: int):
    
    combinations: set[tuple[int, ...]] = set()
    combined_costs: dict[tuple[int, ...], list[float]] = {}
    combined_performances: dict[tuple[int, ...], float] = {}
    
    for q0 in portfolios[subnetworks[0]]:
        for q1 in portfolios[subnetworks[1]]:
            for q2 in portfolios[subnetworks[2]]:
                for q3 in portfolios[subnetworks[3]]:
                   for q4 in portfolios[subnetworks[4]]:
                        for q5 in portfolios[subnetworks[5]]:
                            for q6 in portfolios[subnetworks[6]]:
                                for q7 in portfolios[subnetworks[7]]:
                                    for q8 in portfolios[subnetworks[8]]:
                                        for q9 in portfolios[subnetworks[9]]:
                                            combin: list[int] = [q0, q1, q2, q3, q4, q5, q6, q7, q8, q9]
                                            cost_vector = [0.0 for _ in range(len(budget))]
                                            for j, q in enumerate(combin):
                                                for i in range(len(budget)):
                                                    cost_vector[i] += portfolio_costs[subnetworks[j]][q][i]
                                                

                                            #cost_vector = [sum(portfolio_costs[subnetworks[j]][q][i] for j, q in enumerate(combin)) for i in range(len(budget))]
                                            if all(cost_vector[i] <= budget[i] for i in range(len(budget))):
                                                combinations.add(tuple(combin))
                                                combined_costs[tuple(combin)] = cost_vector

                                                combined_performances[tuple(combin)] = expected_travel_hierarchical(combin, subnetworks, partitioned_paths, terminal_pair_reliabilities, travel_volumes)
    
    return combinations, combined_performances, combined_costs





if __name__ == '__main__':
    print("This file is not meant to be run directly!")

