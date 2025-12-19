from travel_volumes import *
from result_handling import *
from numpy import random
import json
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import pandas as pd

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

def format_station_name(name: str):
    if len(name) <= 3:
        return name.upper()
    else:
        return name.capitalize()


# TODO: Add some random points as well for which we compute the error bars?
def main(Q_star: set[tuple[int, ...]], 
         combined_costs: dict[tuple[int, ...], list[float]], 
         dict_node_reinforcements: dict[str, list[tuple[str, float]]],
         travel_volumes: dict[tuple[str, str], float],
         reliabilities: dict[tuple[int, ...], dict[tuple[str, str], float]],
         directory: str = "", 
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

def plot_pareto_frontier_with_ci(costs: list[float], 
                                 performances: list[list[float]],  # Now a list of lists for multiple samples
                                 random_costs=None, 
                                 random_performances=None, 
                                 sub_network=None, 
                                 normalize=False,
                                 confidence_level=0.95,
                                 show_ci=True,
                                 ci_style='errorbar',  # 'errorbar', 'shaded', or 'boxplot'
                                 plot_mean_line=False):
    """
    Plot Pareto frontier with confidence intervals.
    
    Parameters:
    -----------
    costs : list[float]
        List of costs (number of reinforced switches) for each portfolio
    performances : list[list[float]]
        List of lists containing performance samples for each portfolio
    random_costs : list[float], optional
        Costs for random portfolios
    random_performances : list[list[float]] or list[float], optional
        Performances for random portfolios (can be single values or lists)
    sub_network : str, optional
        Name of subnetwork for title
    normalize : bool, optional
        Whether to normalize the performance values
    confidence_level : float, optional
        Confidence level for intervals (default 0.95 for 95%)
    show_ci : bool, optional
        Whether to show confidence intervals
    ci_style : str, optional
        Style for confidence intervals: 'errorbar', 'shaded', or 'boxplot'
    plot_mean_line : bool, optional
        Whether to connect mean points with a line
        
    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    
    # Calculate means and confidence intervals for each portfolio
    portfolio_means = []
    portfolio_stds = []
    ci_lowers = []
    ci_uppers = []
    margin_errors = []
    
    for perf_samples in performances:
        if isinstance(perf_samples, (int, float)):
            # Single value
            portfolio_means.append(perf_samples)
            portfolio_stds.append(0)
            ci_lowers.append(perf_samples)
            ci_uppers.append(perf_samples)
            margin_errors.append(0)
        else:
            # Multiple samples
            samples = np.array(perf_samples)
            mean_val = np.mean(samples)
            std_val = np.std(samples)
            
            portfolio_means.append(mean_val)
            portfolio_stds.append(std_val)
            
            # Calculate confidence interval
            n = len(samples)
            if n > 1:
                # Use t-distribution for small samples
                t_value = stats.t.ppf((1 + confidence_level) / 2, df=n-1)
                std_err = std_val / np.sqrt(n)
                margin_error = t_value * std_err
            else:
                margin_error = 0
            
            ci_lowers.append(mean_val - margin_error)
            ci_uppers.append(mean_val + margin_error)
            margin_errors.append(margin_error)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot random portfolios (if provided)
    if random_costs is not None and random_performances is not None:
        # Handle both single values and multiple samples for random portfolios
        if isinstance(random_performances[0], (list, np.ndarray)):
            # Calculate means for random portfolios with multiple samples
            random_means = [np.mean(p) if isinstance(p, (list, np.ndarray)) else p 
                           for p in random_performances]
            scatter2 = ax.scatter(x=random_costs, y=random_means, 
                                 c='black', s=1, marker='o', 
                                 alpha=0.5, label='Random Portfolios')
        else:
            # Single values
            scatter2 = ax.scatter(x=random_costs, y=random_performances, 
                                 c='black', s=1, marker='o', 
                                 alpha=0.5, label='Random Portfolios')
    
    # Plot cost-efficient portfolios with confidence intervals
    if show_ci:
        if ci_style == 'errorbar':
            # Plot with error bars
            scatter1 = ax.errorbar(costs, portfolio_means, 
                                   yerr=margin_errors, 
                                   fmt='o', color='red', 
                                   markersize=8, capsize=5,
                                   label='Cost-Efficient Portfolios',
                                   alpha=0.7)
            
            # Connect mean points with line if requested
            if plot_mean_line:
                # Sort by cost for better line connection
                sorted_indices = np.argsort(costs)
                sorted_costs = [costs[i] for i in sorted_indices]
                sorted_means = [portfolio_means[i] for i in sorted_indices]
                ax.plot(sorted_costs, sorted_means, 'r-', alpha=0.5, linewidth=1)
                
        elif ci_style == 'shaded':
            # Plot mean points
            scatter1 = ax.scatter(costs, portfolio_means, 
                                 c='red', s=10, marker='o',
                                 label='Cost-Efficient Portfolios',
                                 zorder=5)
            
            # Connect mean points with line if requested
            if plot_mean_line:
                sorted_indices = np.argsort(costs)
                sorted_costs = [costs[i] for i in sorted_indices]
                sorted_means = [portfolio_means[i] for i in sorted_indices]
                ax.plot(sorted_costs, sorted_means, 'r-', alpha=0.7, linewidth=2, zorder=4)
            
            # Add shaded confidence region
            sorted_indices = np.argsort(costs)
            sorted_costs = [costs[i] for i in sorted_indices]
            sorted_lowers = [ci_lowers[i] for i in sorted_indices]
            sorted_uppers = [ci_uppers[i] for i in sorted_indices]
            
            ax.fill_between(sorted_costs, sorted_lowers, sorted_uppers,
                           alpha=0.2, color='red', label=f'{int(confidence_level*100)}% Confidence Interval')
            
        elif ci_style == 'boxplot':
            # Create positions for boxplots
            unique_costs = sorted(set(costs))
            cost_to_positions = {cost: i for i, cost in enumerate(unique_costs)}
            
            # Group performances by cost
            performances_by_cost = {cost: [] for cost in unique_costs}
            for cost, perf_samples in zip(costs, performances):
                if isinstance(perf_samples, (list, np.ndarray)):
                    performances_by_cost[cost].extend(perf_samples)
                else:
                    performances_by_cost[cost].append(perf_samples)
            
            # Create boxplot
            box_data = [performances_by_cost[cost] for cost in unique_costs]
            positions = [cost_to_positions[cost] for cost in unique_costs]
            
            bp = ax.boxplot(box_data, positions=positions, 
                           widths=0.6, patch_artist=True,
                           showmeans=True, meanline=True,
                           boxprops=dict(facecolor='lightcoral'),
                           medianprops=dict(color='darkred'),
                           meanprops=dict(color='red', linewidth=2))
            
            # Set x-ticks to actual cost values
            ax.set_xticks(positions)
            ax.set_xticklabels([str(cost) for cost in unique_costs])
            
            # Add scatter of means for consistency
            scatter1 = ax.scatter(costs, portfolio_means, 
                                 c='darkred', s=10, marker='D',
                                 label='Portfolio Means', zorder=10)
    else:
        # Original plotting without confidence intervals
        scatter1 = ax.scatter(x=costs, y=portfolio_means, 
                             c='red', s=10, marker='o', 
                             label='Cost-Efficient Portfolios')
    
    # Add legend
    ax.legend(loc='best')
    
    # Set title
    if sub_network is None:
        plt.title(f'Combined Portfolios (with {int(confidence_level*100)}% Confidence Intervals)')
    else:
        plt.title(f'Cost-Efficient Portfolios of Subnetwork {sub_network} (with {int(confidence_level*100)}% CI)')
    
    # Set labels
    plt.xlabel('Number of Reinforced Switches')
    plt.ylabel('Expected Enabled Traffic Volume')
    
    # Set x-ticks (adjust as needed)
    max_cost = int(max(costs)) if costs else 0
    plt.xticks(range(0, max_cost + 5, 5))
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Adjust layout
    plt.tight_layout()
    
    return fig, ax, {
        'means': portfolio_means,
        'ci_lowers': ci_lowers,
        'ci_uppers': ci_uppers,
        'stds': portfolio_stds,
        'margin_errors': margin_errors
    }

def analyze_monte_carlo_results(filename: str):
    """
    Analyze and visualize Monte Carlo results.
    """
    with open(filename, 'r') as f:
        data = json.load(f)
    
    # Extract data
    costs, performances = prepare_data_from_structure(data)
    
    # Print summary statistics
    print("\n" + "="*80)
    print("MONTE CARLO SIMULATION RESULTS")
    print("="*80)
    
    for i, (cost, perf_samples) in enumerate(zip(costs, performances)):
        samples = np.array(perf_samples)
        mean_val = np.mean(samples)
        std_val = np.std(samples)
        ci_low = np.percentile(samples, 2.5)
        ci_high = np.percentile(samples, 97.5)
        cv = std_val / mean_val * 100  # Coefficient of variation
        
        print(f"Portfolio {i+1:2d} (Cost={cost:2d}): "
              f"Mean={mean_val:8.2f}, Std={std_val:6.2f}, "
              f"95% CI=[{ci_low:8.2f}, {ci_high:8.2f}], "
              f"CV={cv:5.2f}%")
    
    # Create comprehensive visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Pareto frontier with CI
    means = [np.mean(p) for p in performances]
    stds = [np.std(p) for p in performances]
    std_errs = [std/np.sqrt(len(p)) for std in stds]
    margin_errors = [1.96 * se for se in std_errs]  # 95% CI
    
    ax1.errorbar(costs, means, yerr=margin_errors, fmt='o-', 
                capsize=5, color='red', alpha=0.7)
    ax1.set_xlabel('Number of Reinforced Switches')
    ax1.set_ylabel('Expected Enabled Traffic Volume')
    ax1.set_title('Pareto Frontier with 95% Confidence Intervals')
    ax1.grid(True, alpha=0.3)
    
    # 2. Distribution of uncertainties
    rel_errors = [me/mean*100 for me, mean in zip(margin_errors, means)]
    ax2.bar(range(len(costs)), rel_errors, color='steelblue', alpha=0.7)
    ax2.set_xlabel('Portfolio Index')
    ax2.set_ylabel('Relative Uncertainty (%)')
    ax2.set_title('Relative 95% CI Width (% of mean)')
    ax2.set_xticks(range(len(costs)))
    ax2.set_xticklabels([str(c) for c in costs])
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Sample distributions for selected portfolios
    selected_indices = [0, len(performances)//2, -1]
    colors = ['red', 'green', 'blue']
    
    for idx, color in zip(selected_indices, colors):
        data = performances[idx]
        ax3.hist(data, bins=50, alpha=0.5, density=True, 
                color=color, label=f'Cost={costs[idx]}')
    
    ax3.set_xlabel('Performance')
    ax3.set_ylabel('Density')
    ax3.set_title('Performance Distributions (Selected Portfolios)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Convergence analysis (if you track intermediate results)
    # Plot how mean converges with sample size
    if len(performances[0]) >= 100:
        sample_sizes = np.logspace(1, np.log10(len(performances[0])), 20).astype(int)
        convergence_data = []
        
        for n in sample_sizes:
            subset_means = [np.mean(samples[:n]) for samples in performances]
            convergence_data.append(np.std(subset_means))
        
        ax4.plot(sample_sizes, convergence_data, 'o-', linewidth=2)
        ax4.set_xlabel('Sample Size')
        ax4.set_ylabel('Std Dev of Portfolio Means')
        ax4.set_title('Monte Carlo Convergence')
        ax4.set_xscale('log')
        ax4.grid(True, alpha=0.3)
    
    plt.suptitle(f'Monte Carlo Analysis: {len(performances[0]):,} samples per portfolio', 
                 fontsize=16)
    plt.tight_layout()
    
    # Save figures
    plt.savefig(f"{filename.replace('.json', '_analysis.png')}", dpi=300, bbox_inches='tight')
    plt.show()
    
    return {
        'costs': costs,
        'means': means,
        'stds': stds,
        'margin_errors': margin_errors,
        'rel_errors': rel_errors
    }


# Helper function to prepare data from your structure
def prepare_data_from_structure(data):
    """
    Extract costs and performances from your data structure.
    
    Parameters:
    -----------
    data : dict
        Your data structure containing portfolios
        
    Returns:
    --------
    costs : list
        List of costs (number of reinforced switches)
    performances : list of lists
        List of performance samples for each portfolio
    """
    costs = []
    performances = []
    
    for portfolio in data['cost_efficient_combined_portfolios']:
        # Calculate cost: number of reinforced nodes
        total_nodes = len(portfolio['total_reinforced_nodes'])
        costs.append(total_nodes)
        
        # Extract performance samples
        # Note: In your data, performance seems to be a single list
        # You might need to adjust this based on your actual data structure
        if 'performance' in portfolio:
            perf_data = portfolio['performance']
            # If performance is a single value, make it a list
            if isinstance(perf_data, (int, float)):
                performances.append([perf_data])
            else:
                performances.append(perf_data)
    
    return costs, performances



def load_portfolio_json(filename, scenario="baseline", direction="base"):
    with open(filename, "r") as f:
        data = json.load(f)

    rows = []
    for p in data["cost_efficient_combined_portfolios"]:
        rows.append({
            "portfolio_id": tuple(p["combined_portfolio_id"]),
            "cost": p["total_cost"],
            "performance": p["performance"],
            "scenario": scenario,
            "direction": direction
        })
    return pd.DataFrame(rows)

def build_sensitivity_dataframe(
    baseline_file,
    scenario_files: dict
):
    """
    scenario_files:
      {
        "1_south_north": {
            "lower": "...json",
            "higher": "...json"
        },
        ...
      }
    """
    dfs = []

    # baseline
    dfs.append(load_portfolio_json(baseline_file))

    # scenarios
    for scen, files in scenario_files.items():
        for direction, fname in files.items():
            dfs.append(load_portfolio_json(
                fname,
                scenario=scen,
                direction=direction
            ))

    return pd.concat(dfs, ignore_index=True)

def compute_univariate_metrics(df):
    base = (
        df[df.direction == "base"]
        .set_index("portfolio_id")
        [["cost", "performance"]]
        .rename(columns={"performance": "base_perf"})
    )

    sens = (
        df[df.direction != "base"]
        .pivot_table(
            index=["portfolio_id", "scenario"],
            columns="direction",
            values="performance"
        )
        .reset_index()
    )

    merged = sens.merge(base, on="portfolio_id")

    merged["delta_plus"]  = merged["higher"] - merged["base_perf"]
    merged["delta_minus"] = merged["lower"]  - merged["base_perf"]
    merged["normalized_sensitivity"] = (
        (merged["higher"] - merged["lower"])
        / (2 * merged["base_perf"])
    )

    return merged, base.reset_index()


def plot_pareto_envelope(base, sens):
    env = sens.groupby("portfolio_id").agg(
        perf_min=("lower", "min"),
        perf_max=("higher", "max")
    )

    df = base.merge(env, on="portfolio_id")

    df = df.sort_values("cost")

    fig, ax = plt.subplots(figsize=(9,6))
    ax.scatter(x=df.cost, y=df.base_perf, marker="o", color="red", label="Cost-Efficient Combined Portfolios")
    ax.fill_between(df.cost, df.perf_min, df.perf_max,
                    alpha=0.25, label="±10% traffic envelope")

    ax.set_xlabel("Cost")
    ax.set_ylabel("Expected enabled traffic volume")
    #ax.set_title("Pareto frontier robustness (univariate ±10%)")
    ax.grid(alpha=0.3)
    ax.legend()

    return fig, ax


def plot_tornado(portfolio_id, sens):
    d = sens[sens.portfolio_id == portfolio_id]

    #order = np.argsort(np.abs(d.delta_plus - d.delta_minus))
    order = np.argsort(
        np.maximum(np.abs(d.delta_plus), np.abs(d.delta_minus))
    )

    fig, ax = plt.subplots(figsize=(12,13))
    ax.barh(
        d.scenario.iloc[order],
        d.delta_plus.iloc[order],
        color="tab:red",
        alpha=0.7,
        label="+10%"
    )
    ax.barh(
        d.scenario.iloc[order],
        d.delta_minus.iloc[order],
        color="tab:blue",
        alpha=0.7,
        label="−10%"
    )

    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel(
    "Difference in Expected Enabled Traffic Volume",
    fontsize=16
)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.legend(fontsize=16)

    return fig, ax






    
if __name__ == "__main__":
    subnetworks = ["apt", "jki", "knh", "kuo", "lna", "sij", "skm", "sor", "te", "toi"]

    Q_star, _, combined_costs, dict_node_reinforcements, _= read_combined_portfolios(filename="model/results/whole_network_ce_portfolios.json")
    reliabilities = read_combined_portfolio_reliabilities("model/results/whole_network_reliabilities.json")

    # CHANGE THIS
    #directory = "model/sensitivity_analysis/parameter_wise_percentual"
    #directory = "model/sensitivity_analysis/parameter_wise_absolute"
    #directory = "model/sensitivity_analysis/simulated_percentual"
    #directory = "model/sensitivity_analysis/simulated_absolute"
    directory = ""

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
        sample_size = 1_000

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

        sample_size = 1_000

        performances: dict[tuple[int, ...], list[float]] = {q: [] for q in Q_star}

        terminal_pairs = list(original_volumes.keys())

        generator = random.uniform(-N, N, size=(sample_size, len(terminal_pairs)))

        for idx in range(sample_size):
            volume_copy = {}
            for j, pair in enumerate(terminal_pairs):
                v = original_volumes[pair]
                variation = generator[idx, j]
                if v < abs(variation):
                    variation = v * variation / abs(variation)
                    volume_copy[pair] = max(v + variation, 0)
                else:
                    volume_copy[pair] = max(v + generator[idx, j], 0)
                
            
            performance = main(Q_star, combined_costs, dict_node_reinforcements, volume_copy, reliabilities, directory)

            for combined_portfolio, p in performance.items():
                if combined_portfolio in performances:
                    performances[combined_portfolio] = performances[combined_portfolio] + [p]
                else:
                    performances[combined_portfolio] = [p]
                
        
        save_combined_portfolios_mc(Q_star, performances, combined_costs, dict_node_reinforcements, subnetworks,
                                    f"{directory}/whole_network_ce_portfolios_mc.json")
    
    else: # visualization
        monte_carlo = False
        if monte_carlo:
            # Load your data
            #directory = "model/sensitivity_analysis/parameter_wise_percentual"
            #directory = "model/sensitivity_analysis/parameter_wise_absolute"
            #directory = "model/sensitivity_analysis/simulated_percentual"
            directory = "model/sensitivity_analysis/simulated_absolute"
            filename = directory + "/whole_network_ce_portfolios_mc.json"
            with open(filename, 'r') as f:
                data = json.load(f)
            
            # Prepare data
            costs, performances = prepare_data_from_structure(data)
            
            # Plot with error bars
            fig1, ax1, stats1 = plot_pareto_frontier_with_ci(
                costs, performances,
                confidence_level=0.95,
                ci_style='errorbar'
            )
            plt.savefig("absolute_pareto_frontier_errorbars.pdf")
            #plt.savefig("percentual_pareto_frontier_errorbars.pdf")
            plt.show()
            
            # Plot with shaded region
            fig2, ax2, stats2 = plot_pareto_frontier_with_ci(
                costs, performances,
                confidence_level=0.95,
                ci_style='shaded'
            )
            plt.savefig("absolute_pareto_frontier_shaded.pdf")
            #plt.savefig("percentual_pareto_frontier_errorbars.pdf")
            plt.show()
            
            # Print statistics
            print("\nPortfolio Statistics:")
            print("Cost\tMean\t95% CI Lower\t95% CI Upper\tStd Dev")
            for i, (cost, mean_val, ci_low, ci_up, std_val) in enumerate(
                zip(costs, stats1['means'], stats1['ci_lowers'], 
                    stats1['ci_uppers'], stats1['stds'])):
                print(f"{cost}\t{mean_val:.2f}\t{ci_low:.2f}\t{ci_up:.2f}\t{std_val:.2f}")
        else:
            # Univariate visualization

            SCENARIO_TO_PAIR_SIMPLE: dict[str, str] = {}

            for key in SCENARIO_TO_PAIR.keys():
                first: str = format_station_name(str(key.split("_")[1]))
                second: str = format_station_name(str(key.split("_")[2]))
                SCENARIO_TO_PAIR_SIMPLE[key] = "(" + first + ", " + second + ")"

            absolute = False
            if absolute:
                directory = "model/sensitivity_analysis/parameter_wise_absolute"
                baseline = "model/results/whole_network_ce_portfolios.json"

                scenario_files = {}
                for scenario in SCENARIO_TO_PAIR.keys():
                    scenario_files[SCENARIO_TO_PAIR_SIMPLE[scenario]] = {
                        "lower": f"{directory}/{scenario}/whole_network_ce_portfolios_lower.json",
                        "higher": f"{directory}/{scenario}/whole_network_ce_portfolios_higher.json"
                    }

                df = build_sensitivity_dataframe(baseline, scenario_files)
                sens, base = compute_univariate_metrics(df)

                # Pareto envelope
                plot_pareto_envelope(base, sens)
                plt.show()

                # Knee portfolio
                knee = base.sort_values("cost").iloc[len(base)//2].portfolio_id

                # Tornado
                plot_tornado(knee, sens)
                plt.show()

            else:
                # Percentual
                directory = "model/sensitivity_analysis/parameter_wise_percentual"
                baseline = "model/results/whole_network_ce_portfolios.json"

                scenario_files = {}
                for scenario in SCENARIO_TO_PAIR.keys():
                    scenario_files[SCENARIO_TO_PAIR_SIMPLE[scenario]] = {
                        "lower": f"{directory}/{scenario}/whole_network_ce_portfolios_lower.json",
                        "higher": f"{directory}/{scenario}/whole_network_ce_portfolios_higher.json"
                    }

                df = build_sensitivity_dataframe(baseline, scenario_files)
                sens, base = compute_univariate_metrics(df)

                # Pareto envelope
                #plot_pareto_envelope(base, sens)
                #plt.show()
                target_costs = [0, 10, 20, 45]

                selected_portfolios = {}
    
                for target_cost in target_costs:
                    if target_cost in base['cost'].values:
                        # Exact match exists
                        portfolio = base[base['cost'] == target_cost].iloc[0]
                    else:
                        # Find closest portfolio
                        cost_diffs = abs(base['cost'] - target_cost)
                        closest_idx = cost_diffs.idxmin()
                        portfolio = base.loc[closest_idx]
                        print(f"Note: No portfolio with exact cost {target_cost}. "
                            f"Using cost {portfolio['cost']} instead.")
                    
                    selected_portfolios[target_cost] = portfolio['portfolio_id']

                # Knee portfolio
                #knee = base.sort_values("cost").iloc[2].portfolio_id

                for cost, knee in selected_portfolios.items():
                    # Tornado
                    plot_tornado(knee, sens)
                    plt.savefig(f"cost_{cost}_parameter_wise_percentual_variation.pdf")
                    plt.show()
                
                # Tornado
                #knee = base.sort_values("cost").iloc[len(base) - 1].portfolio_id
                #plot_tornado(knee, sens)
                #plt.savefig("cost_45_parameter_wise_percentual_variation.pdf")
                #plt.show()



    


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