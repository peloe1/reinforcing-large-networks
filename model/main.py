import networkx as nx
import time
import numpy as np
import random

#from information_set import compute_extreme_points
from portfolio import generate_feasible_portfolios
from subnetwork import cost_efficient_portfolios
from path import terminal_pairs, feasible_paths
from graph import construct_graph, generate_random_graph_with_positions, add_reliabilities
from travel_volumes import read_travel_volumes
from result_handling import save_cost_efficient_portfolios, save_combined_portfolios
#from hierarchical import cost_efficient_combined_portfolios



def main(verbose = False) -> None:
    subnetworks = ["apt", "jki", "knh", "kuo", "lna", "sij", "skm", "sor", "te", "toi"]
    filenames = ["data/network/dipan_data/" + subnetwork + ".json" for subnetwork in subnetworks]
    travel_volumes = ["data/travel_volumes/subnetwork_yearly/" + subnetwork.upper() + "_volumes.json" for subnetwork in subnetworks]

    all_terminal_nodes = {"apt": ["LNA V0001", "SIJ V0632", "APT V0002", "APT V0001"],
                          "jki": ["KNH V0381", "LUI V0511", "JKI V0411", "JKI V0422"],
                          "knh": ["JKI V0411", "SKM V0262", "KNH V0381"],
                          "kuo": ["SOR V0001", "KRM V0001|V0002", "KUO V0941", "KUO V0002"], # Check that the 2nd and 4th are correct
                          "lna": ["TE V0001", "APT V0002", "LNA V0002", "LNA V0001"],
                          "sij": ["SKM V0271", "APT V0001", "TOI V0002", "SIJ V0642", "SIJ V0632", "SIJ V0611"],
                          "skm": ["SIJ V0642", "KNH V0381", "SKM V0262", "SKM V0271"],
                          "sor": ["TOI V0001", "KUO V0002", "SOR V0001"],
                          "te": ["OHM V0002", "LNA V0002", "TE V0001", "TE V0002"],
                          "toi": ["SIJ V0611", "SOR V0001", "TOI V0001", "TOI V0002"]}
    
    dict_Q_CE: dict[str, set[int]] = {}
    dict_portfolio_costs: dict[str, dict[int, list[float]]] = {}
    dict_reinforcement_actions: dict[str, list[tuple[str, float]]] = {}

    
    for filename, subnetwork, travel_volume_path in zip(filenames, subnetworks, travel_volumes):
        #if subnetwork == "kuo":
            print("\n\n\n")
            print("-"*50)
            print("Looking at subnetwork: ", subnetwork.upper())
            
            G, G_original = construct_graph(filename)

            #node_to_idx = {node: i for i, node in enumerate(G.nodes())}

            node_list = sorted(G.nodes())
            print("With " + str(len(node_list)) + " nodes")

            if verbose:
                print("Node list ", node_list)
            
            terminal_nodes = all_terminal_nodes[subnetwork]
            reliabilities = {node: 0.99 for node in node_list} # Every node gets reliability 0.99 at the start
            artificial_nodes = []
            for node in terminal_nodes:
                if node[:3].lower().replace(" ", "") != subnetwork: # Fix the artificial nodes's reliability to 1.0 so they can't be disrupted
                    artificial_nodes.append(node)
                    reliabilities[node] = 1.0

            
            G = add_reliabilities(G, reliabilities)
            
            terminal_node_pairs = terminal_pairs(terminal_nodes)
            
            if verbose:
                print("Terminal node pairs: ", terminal_node_pairs)
                print("Number of terminal node pairs: ", len(terminal_node_pairs))

            travel_volumes = read_travel_volumes(travel_volume_path)
            
            pairs_to_remove = []
            for (u, v) in terminal_node_pairs:
                if (u, v) not in travel_volumes:
                    if (u, v) in terminal_node_pairs:
                        pairs_to_remove.append((u, v))
                if (v, u) not in travel_volumes:
                    if (v, u) in terminal_node_pairs:
                        pairs_to_remove.append((v, u))
            
            for pair in pairs_to_remove:
                if pair in terminal_node_pairs:
                    terminal_node_pairs.remove(pair)
            
            if verbose:
                print("Terminal node pairs, after filtering: ", terminal_node_pairs)
                print("And number of them ", len(terminal_node_pairs))

            paths = feasible_paths(G_original, G, terminal_node_pairs)

            node_reinforcements = []
            costs: dict[str, list[float]] = {}
            for node in node_list:
                if node not in artificial_nodes:
                    node_reinforcements.append((node, 0.995))
                    costs[node] = [1] # Start with uniform cost of 2 units for each action and only one resource

            r = len(node_reinforcements)

            print("and " + str(r) + " of nodes to be considered for reinforcing")

            budget = [40.0]

            #print("Computing the feasible portfolios")

            #start = time.time()
            #Q_F, feasible_portfolio_costs = generate_feasible_portfolios(r, costs, budget)
            #end = time.time()
            #print(f"It took {(end - start):.2f} seconds to compute the {len(Q_F)} feasible portfolios")
            

            print("-----------------------------------------")
            start = time.time()
            Q_CE, performances, portfolio_costs = cost_efficient_portfolios(G, paths, node_reinforcements, costs, budget, travel_volumes, False)#, Q_F, feasible_portfolio_costs, travel_volumes, False)
            end = time.time()
            if end - start > 60:
                print(f"Time to compute cost-efficient portfolios: {(end - start)/60:.2f} minutes")
            else:
                print(f"Time to compute cost-efficient portfolios: {(end - start):.2f} seconds")

            print(f"Number of resulting cost-efficient portfolios for subnetwork {subnetwork.upper()}: {len(Q_CE)}")

            dict_Q_CE[subnetwork] = Q_CE
            dict_portfolio_costs[subnetwork] = portfolio_costs
            dict_reinforcement_actions[subnetwork] = node_reinforcements


            save_cost_efficient_portfolios(Q_CE, performances, portfolio_costs, node_reinforcements, filename="model/results/" + subnetwork + "_ce_portfolios.json")
    
    #filename = "data/network/dipan_data/@network.json"
    #G, G_original = construct_graph(filename)
#
    #node_list = sorted(G.nodes())
    #
    ## TODO: Fix this
    #terminal_nodes = []
    #terminal_node_pairs = terminal_pairs(terminal_nodes)
#
    ## TODO: Fix the terminal pairs in this file to be the actual nodes
    #travel_volume_path = "data/network/travel_volumes/2024_volumes.json"
    #travel_volumes = read_travel_volumes(travel_volume_path)
    #
    #pairs_to_remove = []
    #for (u, v) in terminal_node_pairs:
    #    if (u, v) not in travel_volumes:
    #        if (u, v) in terminal_node_pairs:
    #            pairs_to_remove.append((u, v))
    #    if (v, u) not in travel_volumes:
    #        if (v, u) in terminal_node_pairs:
    #            pairs_to_remove.append((v, u))
    #
    #for pair in pairs_to_remove:
    #    if pair in terminal_node_pairs:
    #        terminal_node_pairs.remove(pair)
#
    #Q_star, combined_performances, combined_costs = cost_efficient_combined_portfolios(G,
    #                                                                                   G_original, 
    #                                                                                   terminal_node_pairs, 
    #                                                                                   travel_volumes, 
    #                                                                                   dict_Q_CE, 
    #                                                                                   subnetworks, 
    #                                                                                   dict_reinforcement_actions, 
    #                                                                                   dict_portfolio_costs, 
    #                                                                                   budget, 
    #                                                                                   len(subnetworks)
    #                                                                                   )
    #
    #save_combined_portfolios(Q_star, 
    #                         combined_performances, 
    #                         combined_costs, 
    #                         dict_reinforcement_actions, 
    #                         subnetworks, 
    #                         filename="model/results/whole_network_ce_portfolios.json")

if __name__ == "__main__":
    main(verbose = True)



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