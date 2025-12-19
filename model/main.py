import networkx as nx
import time
import numpy as np
import random
import matplotlib.pyplot as plt

#from information_set import compute_extreme_points
#from portfolio import generate_feasible_portfolios
from subnetwork import cost_efficient_portfolios
from path import terminal_pairs, feasible_paths, intermediate_terminal_pairs
from graph import construct_graph, add_reliabilities #generate_random_graph_with_positions 
from travel_volumes import read_travel_volumes, subnetwork_travel_volumes
from result_handling import *
from hierarchical import cost_efficient_combined_portfolios, random_portfolios



def main(verbose = False) -> None:
    subnetworks = ["apt", "jki", "knh", "kuo", "lna", "sij", "skm", "sor", "te", "toi"]
    filenames = ["data/network/dipan_data/" + subnetwork + ".json" for subnetwork in subnetworks]
    travel_volumes = ["model/parameters/subnetwork_volumes/" + subnetwork.lower() + "_volumes.json" for subnetwork in subnetworks]

    all_terminal_nodes = {"apt": ["LNA V0001", "SIJ V0632", "APT V0002", "APT V0001"],
                          "jki": ["KNH V0381", "LUI V0511", "JKI V0411", "JKI V0422"],
                          "knh": ["JKI V0411", "SKM V0262", "KNH V0381"],
                          "kuo": ["SOR V0001", "KRM V0001|V0002", "KUO V0941", "KUO V0002"],
                          "lna": ["TE V0001", "APT V0002", "LNA V0002", "LNA V0001"],
                          "sij": ["SKM V0271", "APT V0001", "TOI V0002", "SIJ V0642", "SIJ V0632", "SIJ V0611"],
                          "skm": ["SIJ V0642", "KNH V0381", "SKM V0262", "SKM V0271"],
                          "sor": ["TOI V0001", "KUO V0002", "SOR V0001"],
                          "te": ["OHM V0002", "LNA V0002", "TE V0001", "TE V0002"],
                          "toi": ["SIJ V0611", "SOR V0001", "TOI V0001", "TOI V0002"]}
    
    subnetwork_transitions = {('krm', 'kuo'): ("KRM V0001|V0002", "KUO V0941"),
                              ('kuo', 'sor'): ("KUO V0002", "SOR V0001"),
                              ('krm', 'sor'): ("SOR V0001", "KRM V0001|V0002"),
                              ('sor', 'toi'): ("TOI V0001", "SOR V0001"),
                              ('kuo', 'toi'): ("TOI V0001", "KUO V0002"),
                              ('toi', 'sij'): ("SIJ V0611", "TOI V0002"),
                              ('sor', 'sij'): ("SIJ V0611", "SOR V0001"),
                              ('toi', 'apt'): ("TOI V0002", "APT V0001"),
                              ('toi', 'skm'): ("TOI V0002", "SKM V0271"),
                              ('sij', 'apt'): ("APT V0001", "SIJ V0632"),
                              ('sij', 'skm'): ("SIJ V0642", "SKM V0271"),
                              ('sij', 'lna'): ("LNA V0001", "SIJ V0632"),
                              ('apt', 'lna'): ("APT V0002", "LNA V0001"),
                              ('apt', 'te'): ("APT V0002", "TE V0001"),
                              ('lna', 'te'): ("LNA V0002", "TE V0001"),
                              ('lna', 'ohm'): ("LNA V0002", "OHM V0002"),
                              ('te', 'ohm'): ("OHM V0002", "TE V0002"),
                              ('sij', 'knh'): ("KNH V0381", "SIJ V0642"),
                              ('skm', 'knh'): ("KNH V0381", "SKM V0262"),
                              ('skm', 'jki'): ("JKI V0411", "SKM V0262"),
                              ('knh', 'jki'): ("KNH V0381", "JKI V0411"),
                              ('knh', 'lui'): ("KNH V0381", "LUI V0511"),
                              ('jki', 'lui'): ("JKI V0422", "LUI V0511"),
                              ('skm', 'apt'): ("SKM V0271", "APT V0001")
    }

    dict_transitions: dict[tuple[str, str], tuple[str, str]] = {}

    for (sub1, sub2), (n1, n2) in subnetwork_transitions.items():
        sorted_sub = sorted([sub1, sub2])
        sorted_pair = sorted([n1, n2])
        sub_pair = (sorted_sub[0], sorted_sub[1])
        node_pair = (sorted_pair[0], sorted_pair[1])
        dict_transitions[sub_pair] = node_pair
 
    dict_Q_CE: dict[str, set[int]] = {}
    dict_portfolio_costs: dict[str, dict[int, list[float]]] = {}
    dict_reinforcement_actions: dict[str, list[tuple[str, float]]] = {}
    dict_reliabilities: dict[str, dict[int, dict[tuple[str, str], float]]] = {}

    dict_Q_random: dict[str, set[int]] = {}
    dict_costs_random: dict[str, dict[int, list[float]]] = {}

    subnetwork_pairs: dict[str, list[tuple[str, str]]] = {}

    budget = [45.0]

    compute_subnetwork = True
    compute_random = False


    if compute_subnetwork:
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
                travel_volumes = read_travel_volumes(travel_volume_path)
                
                #terminal_node_pairs = terminal_pairs(terminal_nodes)
                terminal_node_pairs: list[tuple[str, str]] = list(travel_volumes.keys())

                subnetwork_pairs[subnetwork] = terminal_node_pairs
                
                if verbose:
                    print("Terminal node pairs: ", terminal_node_pairs)
                    print("Number of terminal node pairs: ", len(terminal_node_pairs))
                
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

                

                #print("Computing the feasible portfolios")

                #start = time.time()
                #Q_F, feasible_portfolio_costs = generate_feasible_portfolios(r, costs, budget)
                #end = time.time()
                #print(f"It took {(end - start):.2f} seconds to compute the {len(Q_F)} feasible portfolios")
                

                print("-----------------------------------------")
                start = time.time()
                Q_CE, performances, portfolio_costs, reliabilities = cost_efficient_portfolios(G, paths, node_reinforcements, costs, budget, travel_volumes, False)#, Q_F, feasible_portfolio_costs, travel_volumes, False)
                end = time.time()
                if end - start > 60:
                    print(f"Time to compute cost-efficient portfolios: {(end - start)/60:.2f} minutes")
                else:
                    print(f"Time to compute cost-efficient portfolios: {(end - start):.2f} seconds")

                print(f"Number of resulting cost-efficient portfolios for subnetwork {subnetwork.upper()}: {len(Q_CE)}")

                if compute_random:
                    dominated_performances = {}
                    dominated_costs = {}
                    dominated_portfolios = []
                    for q, p in performances.items():
                        #if q not in Q_CE:
                        dominated_portfolios.append(q)
                        dominated_performances[q] = p
                        dominated_costs[q] = portfolio_costs[q]
                    
                    k = min(4, len(dominated_portfolios))

                    random_sample = set(random.sample(dominated_portfolios, k))
                    save_cost_efficient_portfolios(random_sample, dominated_performances, dominated_costs, node_reinforcements, filename="model/random_results/" + subnetwork + "_random_portfolios.json")
                    dict_Q_random[subnetwork] = random_sample
                    dict_costs_random[subnetwork] = dominated_costs


                dict_Q_CE[subnetwork] = Q_CE
                dict_portfolio_costs[subnetwork] = portfolio_costs
                dict_reinforcement_actions[subnetwork] = node_reinforcements
                dict_reliabilities[subnetwork] = reliabilities

                save_terminal_pair_reliabilities(subnetwork, Q_CE, reliabilities, f"model/parameters/terminal_pair_reliabilities/{subnetwork}_reliabilities.json")
                save_cost_efficient_portfolios(Q_CE, performances, portfolio_costs, node_reinforcements, filename="model/results/" + subnetwork + "_ce_portfolios.json")
    
    else:
        for subnetwork in subnetworks:
            Q_CE, _, portfolio_costs, node_reinforcements = read_cost_efficient_portfolios("model/results/" + subnetwork + "_ce_portfolios.json")
            _, reliabilities = read_terminal_pair_reliabilities(f"model/parameters/terminal_pair_reliabilities/{subnetwork}_reliabilities.json")

            dict_Q_CE[subnetwork] = Q_CE
            dict_portfolio_costs[subnetwork] = portfolio_costs
            dict_reinforcement_actions[subnetwork] = node_reinforcements
            dict_reliabilities[subnetwork] = reliabilities


            q_random, _, costs_random, _ = read_cost_efficient_portfolios("model/random_results/" + subnetwork + "_random_portfolios.json")
            dict_Q_random[subnetwork] = q_random
            dict_costs_random[subnetwork] = costs_random


    filename = "data/network/dipan_data/@network.json"
    G, G_original = construct_graph(filename)

    node_list = sorted(G.nodes())
    
    terminal_nodes = set()
    for _, terminal_nodelist in all_terminal_nodes.items():
        for node in terminal_nodelist:
            terminal_nodes.add(node)

    reliabilities = {node: 0.99 for node in node_list} # Every node gets reliability 0.99 at the start
    artificial_nodes = []
    for node in terminal_nodes:
        node_contains_subnetwork = any(subnetwork in node.lower().replace(" ", "") for subnetwork in subnetworks)
    
        # If it does NOT contain any subnetwork ID, then it's an artificial node
        if not node_contains_subnetwork:  
            artificial_nodes.append(node)
            reliabilities[node] = 1.0
    
    #print("Reliabilities: ", reliabilities)
    #print("Artificial nodes: ", artificial_nodes)

    G = add_reliabilities(G, reliabilities)

    travel_volume_path = "model/parameters/2024_volumes.json"
    travel_volumes = read_travel_volumes(travel_volume_path)

    terminal_nodes = list(terminal_nodes)
    #terminal_node_pairs = terminal_pairs(terminal_nodes)
    terminal_node_pairs: list[tuple[str, str]] = list(travel_volumes.keys())

    combinations = 1
    for _, Q_CE_subnetwork in dict_Q_CE.items():
        combinations *= len(Q_CE_subnetwork)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
    print("\n\n\n")
    print("-"*50)
    print("Hierarchical optimization starting")
    print("Total number of primary nodes ", len(G.nodes()))
    print("Total number of secondary nodes ", len(G_original.nodes()) - len(G.nodes()))
    #print("Terminal node pairs: ", terminal_node_pairs)
    print("A total of ", len(terminal_node_pairs), " terminal node pairs")
    print("In total there are a total of ", combinations, " number of combined portfolios to consider")

    compute_paths = False

    if compute_paths:
        start = time.time()
        path_list = feasible_paths(G_original, G, terminal_node_pairs)
        end = time.time()
        print(f"Elapsed time for identifying the feasible path sets {(end-start):.2f} seconds.")

        save_feasible_paths(path_list, "model/parameters/feasible_paths.json")
    else:
        path_list = read_feasible_paths("model/parameters/feasible_paths.json")

    node_to_subnetwork: dict[str, str] = {}
    
    #for subnetwork, nodes in all_terminal_nodes.items():
    #    for node in nodes:
    #        if node[:3].lower() == subnetwork or node[:2].lower() == subnetwork:
    #            node_to_subnetwork[node] = subnetwork
    #        elif node[:3].lower() == 'ohm' or node[:3].lower() == 'lui' or node[:3].lower() == 'krm':
    #            node_to_subnetwork[node] = node[:3].lower()

    for node in node_list:
        if node[:2].lower() == 'te':
            node_to_subnetwork[node] = node[:2].lower()
        elif node[:3].lower() in subnetworks:
            node_to_subnetwork[node] = node[:3].lower()
        elif node[:3].lower() == 'ohm' or node[:3].lower() == 'lui' or node[:3].lower() == 'krm':
            node_to_subnetwork[node] = node[:3].lower()


    # This now works 13.11.
    partitioned_paths = intermediate_terminal_pairs(path_list, node_to_subnetwork, dict_transitions)
    #subnetwork_travel_volumes(subnetworks, travel_volumes, partitioned_paths)
    #print(f"Partitioned paths: {partitioned_paths}")

    #print(f"Number of keys in partitioned paths: {len(partitioned_paths)}, which should match the number of terminal pairs {len(terminal_node_pairs)}")

    #for (u, v), (path, sub_path) in partitioned_paths.items():
    #    print(f"\nPair {(u, v)} corresponds to partitioned path:")
    #    print(path)
    #    print("with subnetwork path: ")
    #    print(sub_path)
    compute_hierarchical = True
    if compute_hierarchical:
        start = time.time()
        Q_star, combined_performances, combined_costs, dict_reliabilities = cost_efficient_combined_portfolios(partitioned_paths, 
                                                                                        dict_reliabilities, 
                                                                                        travel_volumes, 
                                                                                        dict_Q_CE, 
                                                                                        subnetworks, 
                                                                                        dict_portfolio_costs, 
                                                                                        budget, 
                                                                                        len(subnetworks)
                                                                                        )
        
        end = time.time()
        if end - start > 60:
            print(f"Time to compute cost-efficient combined portfolios: {(end - start)/60:.2f} minutes")
        else:
            print(f"Time to compute cost-efficient combined portfolios: {(end - start):.2f} seconds")
        
        save_combined_portfolios(Q_star, 
                            combined_performances, 
                            combined_costs, 
                            dict_reinforcement_actions, 
                            subnetworks, 
                            filename="model/results/whole_network_ce_portfolios.json")
        
        save_combined_portfolio_reliabilities(Q_star, dict_reliabilities, "model/results/whole_network_reliabilities.json")

    else:
        Q_star, combined_performances, combined_costs, _, _ = read_combined_portfolios("model/results/whole_network_ce_portfolios.json")
        reliabilities = read_combined_portfolio_reliabilities("model/results/whole_network_reliabilities.json")

    print(f"Number of resulting cost-efficient combined portfolios: {len(Q_star)}")
    
    if compute_random:
        portfolios_random, random_performances, random_costs = random_portfolios(partitioned_paths, 
                                                                                    dict_reliabilities, 
                                                                                    travel_volumes, 
                                                                                    dict_Q_random,
                                                                                    subnetworks, 
                                                                                    dict_costs_random,
                                                                                    budget, 
                                                                                    len(subnetworks)
                                                                                    )
        
        save_combined_portfolios(portfolios_random, random_performances, random_costs, dict_reinforcement_actions, subnetworks, filename="model/random_results/combined_portfolios.json")





    
    
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