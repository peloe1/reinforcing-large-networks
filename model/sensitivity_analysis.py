from travel_volumes import *
from result_handling import *
from numpy import random


SCENARIO_TO_PAIR: dict[str, tuple[str, str]] = {
    "1_south_north": ('KRM V0001|V0002', 'OHM V0002'),
    "2_kuo_south": ('KUO V0941', 'KRM V0001|V0002'),
    "3_apt_north": ('APT V0002', 'OHM V0002'),
    "4_south_sij": ('KRM V0001|V0002', 'SIJ V0611'),
    "5_north_kuo": ('OHM V0002', 'KUO V0002'),
    "6_kuo_sor": ('KUO V0002', 'SOR V0001'),
    "7_sij_skm": ('SIJ V0642', 'SKM V0271'),
    "8_north_east": ('OHM V0002', 'LUI V0511'),
    "9_sij_north": ('SIJ V0632', 'OHM V0002'),
    "10_north_lna": ('OHM V0002', 'LNA V0002'),
    "11_kuo_sij": ('KUO V0002', 'SIJ V0611'),
    "12_south_lna": ('KRM V0001|V0002', 'LNA V0001'),
    "13_east_sij": ('LUI V0511', 'SIJ V0642'),
    "14_north_sor": ('OHM V0002', 'SOR V0001'),
    "15_lna_east": ('LNA V0001', 'LUI V0511'),
    "16_kuo_east": ('KUO V0002', 'LUI V0511'),
    "17_south_toi": ('TOI V0001', 'KRM V0001|V0002'),
    "18_jki_east": ('LUI V0511', 'JKI V0422'),
    "19_te_north": ('TE V0002', 'OHM V0002'),
    "20_sij_apt": ('SIJ V0632', 'APT V0001'),
    "21_toi_sij": ('TOI V0002', 'SIJ V0611'),
    "22_kuo_toi": ('KUO V0002', 'TOI V0001'),
    "23_south_jki": ('KRM V0001|V0002', 'JKI V0411'),
    "24_skm_east": ('SKM V0262', 'LUI V0511'),
    "25_south_east": ('KRM V0001|V0002', 'LUI V0511'),
    "26_east_toi": ('LUI V0511', 'TOI V0002'),
    "27_apt_toi": ('APT V0001', 'TOI V0002'),
    "28_lna_apt": ('LNA V0001', 'APT V0002'),
    "29_toi_north": ('TOI V0002', 'OHM V0002'),
    "30_lna_kuo": ('LNA V0001', 'KUO V0002'),
    "31_sij_lna": ('SIJ V0632', 'LNA V0001'),
    "32_te_sij": ('TE V0001', 'SIJ V0632'),
    "33_south_apt": ('KRM V0001|V0002', 'APT V0001'),
    "34_south_sor": ('KRM V0001|V0002', 'SOR V0001'),
    "35_sor_toi": ('SOR V0001', 'TOI V0001'),
    "36_kuo_apt": ('KUO V0002', 'APT V0001')
}


# TODO: Add some random points as well for which we compute the error bars?
def main(Q_star: set[tuple[int, ...]], 
         combined_costs: dict[tuple[int, ...], list[float]], 
         dict_node_reinforcements: dict[str, list[tuple[str, float]]],
         travel_volumes: dict[tuple[str, str], float],
         reliabilities: dict[tuple[int, ...], dict[tuple[str, str], float]],
         directory: str, 
         scenario_name = None,
         direction = None
         ) -> dict[tuple[int, ...], float]:
    
    subnetworks = ["apt", "jki", "knh", "kuo", "lna", "sij", "skm", "sor", "te", "toi"]

    performances: dict[tuple[int, ...], float] = {}

    for combined_portfolio in Q_star:
        reliability_dict = reliabilities[combined_portfolio]
        performance: float = 0.0
        
        for terminal_pair, reliability in reliability_dict.items():
            performance += reliability * travel_volumes[terminal_pair]
        
        performances[combined_portfolio] = performance
    
    if scenario_name is not None and direction is not None:
        save_combined_portfolios(Q_star, performances, combined_costs, dict_node_reinforcements, subnetworks, 
                                filename=f"{directory}/{scenario_name}/whole_network_ce_portfolios_{direction}.json")
    else: # This must be the Monte Carlo simulation
        return performances

    return performances



    
    
    
    
if __name__ == "__main__":
    subnetworks = ["apt", "jki", "knh", "kuo", "lna", "sij", "skm", "sor", "te", "toi"]

    Q_star, _, combined_costs, dict_node_reinforcements, _= read_combined_portfolios(filename="model/results/whole_network_ce_portfolios.json")
    reliabilities = read_combined_portfolio_reliabilities("model/results/whole_network_reliabilities.json")

    # CHANGE THIS
    #directory = "model/sensitivity_analysis/parameter_wise_percentual"
    #directory = "model/sensitivity_analysis/parameter_wise_absolute"
    #directory = "model/sensitivity_analysis/simulated_percentual"
    directory = "model/sensitivity_analysis/simulated_absolute"

    scenarios = ["1_south_north", "2_kuo_south", "3_apt_north", "4_south_sij", "5_north_kuo", "6_kuo_sor", 
                 "7_sij_skm", "8_north_east", "9_sij_north", "10_north_lna", "11_kuo_sij", "12_south_lna",
                 "13_east_sij", "14_north_sor", "15_lna_east", "16_kuo_east", "17_south_toi", "18_jki_east",
                 "19_te_north", "20_sij_apt", "21_toi_sij", "22_kuo_toi", "23_south_jki", "24_skm_east",
                 "25_south_east", "26_east_toi", "27_apt_toi", "28_lna_apt", "29_toi_north", "30_lna_kuo",
                 "31_sij_lna", "32_te_sij", "33_south_apt", "34_south_sor", "35_sor_toi", "36_kuo_apt"]
   
    original_volumes: dict[tuple[str, str], float] = read_travel_volumes(filename="model/sensitivity_analysis/2024_volumes.json")

    if directory == "model/sensitivity_analysis/parameter_wise_percentual":
        for scenario_name in scenarios:
            n1, n2 = SCENARIO_TO_PAIR[scenario_name]
            terminal_pair = tuple(sorted([n1, n2]))

            percent: float = 10.0 / 100.0

            volume_copy = original_volumes.copy()
            volume_copy[terminal_pair] = volume_copy[terminal_pair] * (1 - percent)
            save_frequencies(volume_copy, f"{directory}/{scenario_name}/parameters/2024_volumes_lower.json")

            volume_copy = original_volumes.copy()
            volume_copy[terminal_pair] = volume_copy[terminal_pair] * (1 + percent)
            save_frequencies(volume_copy, f"{directory}/{scenario_name}/parameters/2024_volumes_higher.json")

            direction = "lower"
            travel_volumes_lower = read_travel_volumes(filename=f"{directory}/{scenario_name}/parameters/2024_volumes_{direction}.json")
            performances_lower = main(Q_star, combined_costs, dict_node_reinforcements, travel_volumes_lower, reliabilities, scenario_name, direction, directory)

            direction = "higher"
            travel_volumes_higher = read_travel_volumes(filename=f"{directory}/{scenario_name}/parameters/2024_volumes_{direction}.json")
            performances_higher = main(Q_star, combined_costs, dict_node_reinforcements, travel_volumes_higher, reliabilities, scenario_name, direction, directory)

    elif directory == "model/sensitivity_analysis/parameter_wise_absolute":
        for scenario_name in scenarios:
            n1, n2 = SCENARIO_TO_PAIR[scenario_name]
            terminal_pair = tuple(sorted([n1, n2]))

            N: float = 100

            volume_copy = original_volumes.copy()
            volume_copy[terminal_pair] = max(volume_copy[terminal_pair] - N, 0)
            save_frequencies(volume_copy, f"{directory}/{scenario_name}/parameters/2024_volumes_lower.json")

            volume_copy = original_volumes.copy()
            volume_copy[terminal_pair] = volume_copy[terminal_pair] + N
            save_frequencies(volume_copy, f"{directory}/{scenario_name}/parameters/2024_volumes_higher.json")

            direction = "lower"
            travel_volumes_lower = read_travel_volumes(filename=f"{directory}/{scenario_name}/parameters/2024_volumes_{direction}.json")
            performances_lower = main(Q_star, combined_costs, dict_node_reinforcements, travel_volumes_lower, reliabilities, scenario_name, directory, direction)

            direction = "higher"
            travel_volumes_higher = read_travel_volumes(filename=f"{directory}/{scenario_name}/parameters/2024_volumes_{direction}.json")
            performances_higher = main(Q_star, combined_costs, dict_node_reinforcements, travel_volumes_higher, reliabilities, scenario_name, directory, direction)

            

    # TODO: Figure out how to best do the Monte Carlo simulation here
    # Start with uniform distribution [-10%, +10%]
    elif directory == "model/sensitivity_analysis/simulated_percentual":
        percent: float = 10/100
        sample_size = 10_000

        generator = random.uniform(-percent, percent, size=(sample_size, len(original_volumes)))

        performances: dict[tuple[int, ...], list[float]] = {}

        for idx in range(sample_size):
            volume_copy = original_volumes.copy()
            for j, (terminal_pair, volume) in enumerate(volume_copy.items()):
                variation = generator[idx, j]
                volume_copy[terminal_pair] = volume * (1 + variation)
            
            performance = main(Q_star, combined_costs, dict_node_reinforcements, volume_copy, reliabilities, directory)

            for combined_portfolio, p in performance.items():
                if combined_portfolio in performances:
                    performances[combined_portfolio] = performances[combined_portfolio] + [p]
                else:
                    performances[combined_portfolio] = [p]
        
        save_combined_portfolios_mc(Q_star, performances, combined_costs, dict_node_reinforcements, subnetworks,
                                    f"{directory}/whole_network_ce_portfolios_mc.json")
    
    # TODO: Figure out how to best do the Monte Carlo simulation here
    # Start with uniform distribution [-N, +N]
    elif directory == "model/sensitivity_analysis/simulated_absolute":
        N: float = 100.0

        sample_size = 10_000

        performances: dict[tuple[int, ...], list[float]] = {q: [] for q in Q_star}

        generator = random.uniform(-N, N, size=(sample_size, len(original_volumes)))

        
    
        for idx in range(sample_size):
            volume_copy = original_volumes.copy()
            for j, (terminal_pair, volume) in enumerate(volume_copy.items()):
                variation = generator[idx, j]
                volume_copy[terminal_pair] = max(volume + variation, 0)
            
            performance = main(Q_star, combined_costs, dict_node_reinforcements, volume_copy, reliabilities, directory)

            for combined_portfolio, p in performance.items():
                if combined_portfolio in performances:
                    performances[combined_portfolio] = performances[combined_portfolio] + [p]
                else:
                    performances[combined_portfolio] = [p]
                
        
        save_combined_portfolios_mc(Q_star, performances, combined_costs, dict_node_reinforcements, subnetworks,
                                    f"{directory}/whole_network_ce_portfolios_mc.json")
    
    else: # visualization

        pass



    


    # Use this if you want to do the proper sensitivity analysis on the actual optimal solution / the composition of the optima (core indexes etc.)
    """
    compute_subnetwork_travel_volumes = True
    if compute_subnetwork_travel_volumes:

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
        
        filename = "data/network/dipan_data/@network.json"
        G, G_original = construct_graph(filename)

        node_list = sorted(G.nodes())
        node_to_subnetwork: dict[str, str] = {}
        for node in node_list:
            if node[:2].lower() == 'te':
                node_to_subnetwork[node] = node[:2].lower()
            elif node[:3].lower() in subnetworks:
                node_to_subnetwork[node] = node[:3].lower()
            elif node[:3].lower() == 'ohm' or node[:3].lower() == 'lui' or node[:3].lower() == 'krm':
                node_to_subnetwork[node] = node[:3].lower()

        path_list = read_feasible_paths("model/parameters/feasible_paths.json")
        partitioned_paths = intermediate_terminal_pairs(path_list, node_to_subnetwork, dict_transitions)

        for scenario_name in scenarios:
            travel_volume_path = f"{directory}/{scenario_name}/parameters/2024_volumes.json"
            travel_volume_dict = read_travel_volumes(travel_volume_path)
        
            subnetwork_travel_volumes(subnetworks, travel_volume_dict, partitioned_paths, f"{directory}/{scenario_name}/parameters/subnetwork_volumes")

    """