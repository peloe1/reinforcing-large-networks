import networkx as nx
import numpy as np
from path import feasible_paths
from portfolio import generate_feasible_portfolios, dominates_with_cost
from model.subnetwork import cost_efficient_portfolios
import itertools


def partitions():
    return None

# TODO: Double check and optimize this
def generate_combinations_float(elements: list[list[float]]) -> list[list[float]]:
    first = elements.pop(0)
    combinations = [[i] for i in first]
    new_combinations: list[list[float]] = []

    for elem in elements:
        new_combinations = []
        for comb in combinations:
            for i in elem:
                new_elem = comb.copy()
                new_elem.append(i)
                new_combinations.append(new_elem)
        combinations = new_combinations.copy()
    
    return combinations

def generate_combinations_int(elements: list[list[int]]) -> list[list[int]]:
    first = elements.pop(0)
    combinations = [[i] for i in first]
    new_combinations: list[list[int]] = []

    for elem in elements:
        new_combinations = []
        for comb in combinations:
            for i in elem:
                new_elem = comb.copy()
                new_elem.append(i)
                new_combinations.append(new_elem)
        combinations = new_combinations.copy()
    
    return combinations
    

# TODO: Double check that this works and optimize it
def uncertainty_set(confidence_intervals: list[list[list[float]]]) -> list[list[list[float]]]:
    D_f: list[list[list[float]]] = []
    k = len(confidence_intervals)

    for j in range(k):
        conf_ints: list[list[float]] = confidence_intervals[j].copy()
        combinations = generate_combinations_float(conf_ints)
        D_f.append(combinations)
    return D_f

# This seems to work 10.8.
def all_combined_portfolios(Q: list[list[int]]) -> list[list[int]]:
    portfolios = Q.copy()
    combinations = generate_combinations_int(portfolios)
    return combinations

# This seems to work 10.8.
def feasible_combined_portfolios(Q: list[list[int]], 
                                 portfolio_costs: list[dict[int, float]], 
                                 budget: float) -> tuple[list[list[int]], dict[tuple[int, ...], float]]:
    combinations = all_combined_portfolios(Q)
    feasible = []
    costs: dict[tuple[int, ...], float] = {}
    for comb in combinations:
        cost = 0
        for i, elem in enumerate(comb):
            cost += portfolio_costs[i][elem]
        
        if cost <= budget:
            feasible.append(comb)
            costs[tuple(comb)] = cost

    return feasible, costs


# TODO: Double check that this works and optimize it
def decomposition_solver(G: list[nx.Graph], 
                         T: list[list[tuple[int, int]]], 
                         D_f: list[list[list[float]]],
                         reinforcement_actions: list[list[tuple[int, float, float]]],
                         budget: float) -> set[list[int]]:
    k = len(G)
    m = len(D_f[0])
    Q = []
    portfolio_costs = []
    portfolio_performances = []

    for graph, terminal_pairs, actions in zip(G, T, reinforcement_actions):
        r = len(actions)
        action_costs: list[float] = []
        node_reinforcements: list[tuple[int, float]] = []
        for node, cost, prob in actions:
            action_costs.append(cost)
            node_reinforcements.append((node, prob))

        # Step 1
        feasible_portfolios, costs = generate_feasible_portfolios(r, action_costs, budget)
        paths = feasible_paths(graph, terminal_pairs)

        # Step 2
        Q_CE, performances = cost_efficient_portfolios(graph, paths, node_reinforcements, feasible_portfolios, costs)
        Q.append(Q_CE)
        portfolio_costs.append({portfolio: costs[portfolio] for portfolio in Q_CE})
        portfolio_performances.append(performances)
    
    # Step 3
    feasible_ones: tuple[list[list[int]], dict[tuple[int, ...], float]] = feasible_combined_portfolios(Q, portfolio_costs, budget)
    Q_F, combined_costs = feasible_ones

    combined_performances: dict[tuple[int, ...], list[float]] = {}

    # Step 4
    for portfolio in Q_F:
        performance = [0.0 for _ in range(m)]
        for j in range(k):
            # |T^j| number of utilities (terminal pair reliabilities) corresponding to the jth subnetwork
            u_j: list[float] = portfolio_performances[j][portfolio[j]] 
            for i in range(m):
                # |T^j| number of travel volumes corresponding to the jth subnetwork
                volumes: list[float] = D_f[j][i]

                # Dot product
                performance[i] += sum([volume * utility for volume, utility in zip(volumes, u_j)])
        
        combined_performances[tuple(portfolio)] = performance

    # TODO: Filter the dominated portfolios away
    # Idea 1: Naive brute force approach, do filtering passes until no portfolio gets pruned out => only non-dominated ones left
    # Idea 2: Figure out a more elegant approach utilizing dynamic programming similar to the approach by Kangaspunta and Salo 2014

    # Step 5
    
    brute_force = True
    if brute_force:
        Q_feasible: set[list[int]] = set(Q_F)
        dominated = set(filter(lambda q1: any(dominates_with_cost(combined_performances[tuple(q2)], 
                                                                  combined_performances[tuple(q1)], 
                                                                  combined_costs[tuple(q2)], 
                                                                  combined_costs[tuple(q1)]
                                                                  ) for q2 in Q_feasible), Q_feasible))
        Q_cost_efficient = Q_feasible.difference(dominated)
        return Q_cost_efficient
    else:
        # More elegant approach here which avoids comparing all portfolios to all other ones?
        return set()


if __name__ == '__main__':
    Q: list[list[int]] = [[0, 1, 2, 3, 4], [0, 1]]
    lengths = [len(i) for i in Q]
    feasible = all_combined_portfolios(Q)
    for f in feasible:
        print(f)

    print("Number of combined portfolios: ", len(feasible))
    print("and it should be: ", np.prod(lengths))

