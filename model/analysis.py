from result_handling import read_combined_portfolios, read_cost_efficient_portfolios, save_combined_portfolios
import matplotlib.pyplot as plt
import numpy as np
import os



def plot_pareto_frontier(costs: list[float], performances: list[float], random_costs = None, random_performances = None, sub_network=None, normalize=False):
    # Cost-Efficient Portfolios

    fig, ax = plt.subplots()

    # Plot cost-efficient portfolios (red dots)
    scatter1 = ax.scatter(x=costs, y=performances, c='red', s=10, marker='o', label='Cost-Efficient Portfolios')

    # The random portfolios (black dots)
    if random_costs is not None and random_performances is not None:
        scatter2 = ax.scatter(x=random_costs, y=random_performances, c='black', s=1, marker='o', label='Random Portfolios')
        ax.legend()
    else:
        # If no random portfolios, still show legend for cost-efficient ones
        ax.legend()

    #if sub_network is None:
    #    plt.title('Combined Portfolios')
    #else:
    #    assert(type(sub_network) == str)
    #    plt.title('Cost-Efficient Portfolios of Subnetwork ' + sub_network)

    plt.xlabel('Cost')
    plt.ylabel('Expected Enabled Traffic Volume')
    #plt.yscale('function', functions=(lambda x: x / 1000, lambda x: x * 1000))
    plt.xticks([0, 5, 10, 15, 20, 25, 30, 35, 40, 45])
    #plt.yticks([14400 + 100 * i for i in range(10)])
    plt.savefig(f"network_pareto_frontier.pdf")
    plt.show()

    return

def plot_subnetwork_pareto_frontier(costs: list[float], performances: list[float], sub_network: str):

    # Cost-Efficient Portfolios
    fig, ax = plt.subplots()

    # Plot cost-efficient portfolios (red dots)
    scatter1 = ax.scatter(x=costs, y=performances, c='red', s=10, marker='o', label='Cost-Efficient Portfolios')

    max_cost = int(max(costs))

   
    # If no random portfolios, still show legend for cost-efficient ones
    ax.legend()

    #plt.title('Cost-Efficient Portfolios for the Subnetwork ' + sub_network.upper())

    plt.xlabel('Cost')
    plt.ylabel('Expected Enabled Traffic Volume')
    #plt.yscale('function', functions=(lambda x: x / 1000, lambda x: x * 1000))
    plt.xticks([i for i in range(max_cost + 1)])
    #plt.yticks([14400 + 100 * i for i in range(10)])
    plt.savefig(f"{sub_network}_pareto_frontier.pdf")
    plt.show()

    return

import numpy as np
import matplotlib.pyplot as plt

def compute_core_indexes_for_all_switches(Q_star, combined_costs, dict_node_reinforcements, selected_budgets):
    # First, map all unique switch IDs across all subnetworks
    all_switch_ids = []
    switch_to_index = {}
    index_to_info = {}
    
    global_id = 0
    subnetworks = sorted(dict_node_reinforcements.keys())
    
    for sub_idx, sub_name in enumerate(subnetworks):
        reinforcements = dict_node_reinforcements[sub_name]
        for local_idx, (node_id, cost) in enumerate(reinforcements):
            all_switch_ids.append(global_id)
            switch_to_index[(sub_idx, local_idx)] = global_id
            index_to_info[global_id] = {
                'subnetwork': sub_name,
                'original_id': node_id,
                'sub_idx': sub_idx,
                'local_idx': local_idx
            }
            global_id += 1
    
    # Initialize dictionary to store core indexes for ALL switches
    core_indexes = {switch_id: [0.0] * len(selected_budgets) for switch_id in all_switch_ids}
    
    # Track which switches are ever reinforced across all portfolios
    switch_reinforced_at_any_budget = {switch_id: False for switch_id in all_switch_ids}
    
    # For each selected budget level
    for budget_idx, budget in enumerate(selected_budgets):
        # Get portfolios at or below this budget
        portfolios_at_budget = []
        for portfolio_tuple in Q_star:
            if combined_costs[portfolio_tuple][0] == budget:
                portfolios_at_budget.append(portfolio_tuple)
        
        if not portfolios_at_budget:
            continue
            
        # For EACH switch, compute frequency at this budget level
        for switch_id in all_switch_ids:
            count = 0
            info = index_to_info[switch_id]
            sub_idx = info['sub_idx']
            local_idx = info['local_idx']
            
            for portfolio_tuple in portfolios_at_budget:
                sub_int = portfolio_tuple[sub_idx]
                num_nodes_in_sub = len(dict_node_reinforcements[info['subnetwork']])
                binary_str = bin(sub_int)[2:].zfill(num_nodes_in_sub)
                
                if binary_str[-(local_idx + 1)] == '1':
                    count += 1
                    switch_reinforced_at_any_budget[switch_id] = True
            
            if portfolios_at_budget:
                core_indexes[switch_id][budget_idx] = count / len(portfolios_at_budget)
    
    # Filter out switches that are never reinforced at any budget level
    filtered_switch_ids = [switch_id for switch_id in all_switch_ids 
                          if switch_reinforced_at_any_budget[switch_id]]
    
    # Create filtered dictionaries
    filtered_core_indexes = {switch_id: core_indexes[switch_id] 
                            for switch_id in filtered_switch_ids}
    filtered_index_to_info = {switch_id: index_to_info[switch_id] 
                             for switch_id in filtered_switch_ids}
    
    print(f"Total switches in network: {len(all_switch_ids)}")
    print(f"Switches reinforced at least once: {len(filtered_switch_ids)}")
    print(f"Switches never reinforced: {len(all_switch_ids) - len(filtered_switch_ids)}")
    
    return filtered_core_indexes, selected_budgets, filtered_switch_ids, filtered_index_to_info


def plot_core_indexes_clean_heatmap(core_indexes, budget_levels, switch_ids, index_to_info):
    # Convert to matrix - ensure we plot ALL switches in order
    switches_sorted = sorted(core_indexes.keys())
    
    # Create matrix
    data = []
    for switch_id in switches_sorted:
        data.append(core_indexes[switch_id])
    
    data_matrix = np.array(data)
    
    # Create figure with appropriate size based on number of switches
    num_switches = len(switches_sorted)
    fig_height = max(8, num_switches * 0.31)
    fig, ax = plt.subplots(figsize=(16, fig_height))
    
    # Plot heatmap with clean style
    cmap = plt.cm.YlOrRd
    #cmap = plt.cm.viridis
    im = ax.imshow(data_matrix, cmap=cmap, aspect='auto', vmin=0, vmax=1)
    
    # Create y-labels with original switch IDs
    y_labels = []
    for switch_id in switches_sorted:
        info = index_to_info[switch_id]
        y_labels.append(f"{info['original_id']}")
    
    ax.set_yticks(np.arange(len(switches_sorted)))
    ax.set_yticklabels(y_labels, fontsize=14)
    
    # Add grid
    #ax.set_xticks(np.arange(len(budget_levels)+1)-0.5, minor=True)
    ax.set_xticks(np.arange(4, len(budget_levels), 5))
    ax.set_xticklabels(budget_levels[4::5], fontsize=14)
    ax.set_yticks(np.arange(num_switches+1)-0.5, minor=True)
    #ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.1, alpha=0.2)
    ax.tick_params(which="minor", size=0)
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.tick_params(labelsize=14)
    cbar.ax.set_ylabel("Core Index", rotation=-90, va="bottom", fontsize=20)
    
    # Set title
    #ax.set_title(f"Core Indexes of Reinforced Switches", fontsize=20, pad=20)
    #ax.set_xlabel("Number of Reinforced Switches", fontsize=16)
    ax.set_xlabel("Cost", fontsize=20)
    #ax.set_ylabel("Switch ID", fontsize=16, labelpad=20)
    plt.subplots_adjust(left=0.30)
    
    plt.tight_layout()
    return fig, ax

def plot_core_indexes(core_indexes, budget_levels, switch_ids, index_to_info):
    # Convert to matrix - ensure we plot ALL switches in order
    switches_sorted = sorted(core_indexes.keys())
    
    # Create matrix
    data = []
    for switch_id in switches_sorted:
        data.append(core_indexes[switch_id])
    
    data_matrix = np.array(data)
    
    # Create figure with appropriate size based on number of switches
    num_switches = len(switches_sorted)
    fig_height = max(8, num_switches * 0.3)
    fig, ax = plt.subplots(figsize=(16, fig_height))
    
    # Create custom colormap for trinary labeling
    # Colors: 0.0, (0.0, 1.0), 1.0
    colors = [
        (0.8, 0.8, 0.8),    # Light gray for 0.0
        (1.0, 0.7, 0.4),    # Orange for (0, 1)
        (0.8, 0.1, 0.1)     # Dark red for 1.0
    ]
    
    # Create a ListedColormap with our colors
    from matplotlib.colors import ListedColormap, BoundaryNorm
    n_bins = 3
    cmap = ListedColormap(colors)
    
    # Define boundaries for our categories:
    # 0: exactly 0
    # 1: between 0 and 1 (exclusive)
    # 2: exactly 1
    bounds = [-0.5, 0.01, 0.99, 1.5]  # Small buffer for floating point precision
    norm = BoundaryNorm(bounds, cmap.N)
    
    # Plot heatmap with custom colormap
    im = ax.imshow(data_matrix, cmap=cmap, norm=norm, aspect='auto')
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(budget_levels)))
    ax.set_xticklabels([f"{b}" for b in budget_levels], fontsize=11)
    
    # Create y-labels with original switch IDs
    y_labels = []
    for switch_id in switches_sorted:
        info = index_to_info[switch_id]
        y_labels.append(f"{info['original_id']}")
    
    ax.set_yticks(np.arange(len(switches_sorted)))
    ax.set_yticklabels(y_labels, fontsize=9)
    
    # Add grid
    ax.set_xticks(np.arange(len(budget_levels)+1)-0.5, minor=True)
    ax.set_yticks(np.arange(num_switches+1)-0.5, minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.3)
    ax.tick_params(which="minor", size=0)
    
    # Create custom colorbar with discrete labels
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8, 
                             ticks=[0, 1, 2],  # Center of each color band
                             format='%d')
    
    # Set colorbar labels
    cbar.ax.set_yticklabels(['Core Index = 0', '0 < Core Index < 1', 'Core Index = 1'])
    cbar.ax.tick_params(labelsize=10)
    
    # Optional: Add text annotations for each cell to show actual values
    # Uncomment if you want to see the actual core index values in each cell
    # for i in range(data_matrix.shape[0]):
    #     for j in range(data_matrix.shape[1]):
    #         val = data_matrix[i, j]
    #         color = 'black' if val > 0.5 else 'black'  # Adjust text color for readability
    #         ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=color, fontsize=8)
    
    # Set title
    ax.set_title(f"Core Indexes of Reinforced Switches", fontsize=14, pad=20)
    ax.set_xlabel("Number of Reinforced Switches", fontsize=12)
    ax.set_ylabel("Switch ID", fontsize=12)
    
    plt.tight_layout()
    return fig, ax


def plot_budget_allocation(
    results_file: str = "model/results/whole_network_ce_portfolios.json",
    output_path: str = "budget_allocation_stacked.pdf",
    figsize: tuple = (12, 7),
    colormap: str = 'tab20c'
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Create a budget allocation plot showing how budget is distributed across subnetworks
    for all cost-efficient portfolios.
    
    Parameters:
    -----------
    results_file : str
        Path to the JSON file with combined portfolios
    output_path : str
        Path to save the output figure
    figsize : Tuple
        Figure size (width, height) in inches
    colormap : str
        Matplotlib colormap name
    
    Returns:
    --------
    budgets : np.ndarray
        Sorted budget values
    allocation_matrix : np.ndarray
        Allocation matrix
    subnetwork_info : Dict
        Information about subnetworks
    """
    
    # Read the combined portfolios data
    Q_star, combined_performances, combined_costs, dict_node_reinforcements, subnetworks = read_combined_portfolios(results_file)
    
    # Convert Q_star to list for easier processing
    portfolio_tuples = list(Q_star)
    
    # Get the number of subnetworks
    n_subnets = len(subnetworks)
    
    # Create subnetwork mapping (index to name)
    # Assuming subnetworks is already a list of names like ["apt", "jki", ...]
    subnetwork_mapping = {i: name.upper() for i, name in enumerate(subnetworks)}
    
    # Initialize allocation matrix
    n_portfolios = len(portfolio_tuples)
    allocation_matrix = np.zeros((n_portfolios, n_subnets))
    budgets = np.zeros(n_portfolios)
    
    print(f"Processing {n_portfolios} cost-efficient portfolios...")
    print(f"Subnetworks: {subnetworks}")
    
    # Process each portfolio
    for i, portfolio_tuple in enumerate(portfolio_tuples):
        # Get total cost (number of reinforced switches)
        total_cost = combined_costs[portfolio_tuple][0]
        budgets[i] = total_cost
        
        # Process each subnetwork in the portfolio
        for sub_idx, sub_portfolio_int in enumerate(portfolio_tuple):
            # Get the cost for this subnetwork portfolio
            # We need to look up the cost of this specific subnetwork portfolio
            subnetwork_name = subnetworks[sub_idx]
            
            # Get the list of node reinforcements for this subnetwork
            node_reinforcements = dict_node_reinforcements.get(subnetwork_name, [])
            num_nodes = len(node_reinforcements)
            
            # Convert integer to binary representation of node selections
            binary_str = bin(sub_portfolio_int)[2:].zfill(num_nodes)
            
            # Count number of reinforced nodes in this subnetwork
            # (1 means reinforced, 0 means not reinforced)
            reinforced_nodes = sum(1 for bit in binary_str if bit == '1')
            
            # Each reinforced node costs 1 unit
            subnetwork_cost = reinforced_nodes
            
            allocation_matrix[i, sub_idx] = subnetwork_cost
    
    # Verify that row sums equal total costs (sanity check)
    row_sums = allocation_matrix.sum(axis=1)
    for i, (budget, row_sum) in enumerate(zip(budgets, row_sums)):
        if abs(budget - row_sum) > 1e-10:
            print(f"Warning: Portfolio {i}: budget={budget}, row_sum={row_sum}")
    
    # Convert to percentages
    for i in range(n_portfolios):
        if budgets[i] > 0:
            allocation_matrix[i, :] = (allocation_matrix[i, :] / budgets[i]) * 100
    
    # Sort by budget
    sort_indices = np.argsort(budgets)
    budgets_sorted = budgets[sort_indices]
    allocation_sorted = allocation_matrix[sort_indices, :]
    
    # Determine subnetwork order by when they first receive funding
    first_funding = np.full(n_subnets, np.inf)
    for j in range(n_subnets):
        nonzero_indices = np.where(allocation_sorted[:, j] > 0)[0]
        if len(nonzero_indices) > 0:
            first_funding[j] = nonzero_indices[0]
    
    # Sort subnetworks by when they first get funded
    subnet_order_indices = np.argsort(first_funding)
    subnetwork_order = np.array(range(n_subnets))[subnet_order_indices]
    allocation_sorted_ordered = allocation_sorted[:, subnet_order_indices]
    
    # Get subnetwork names in the correct order
    subnetwork_names_ordered = [subnetwork_mapping[idx] for idx in subnetwork_order]
    
    # Create the plot
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=figsize)
    
    # Generate colors from colormap
    colors = plt.cm.get_cmap(colormap, n_subnets)
    
    # Create stacked area plot
    ax.stackplot(budgets_sorted,
                 allocation_sorted_ordered.T,
                 labels=subnetwork_names_ordered,
                 colors=[colors(i) for i in range(n_subnets)],
                 alpha=0.85,
                 edgecolor='white',
                 linewidth=0.5)
    
    # Formatting
    ax.set_xlabel('Total Reinforcement Budget (Number of Switches)', 
                  fontsize=12, fontweight='medium')
    ax.set_ylabel('Percentage of Budget Allocated (%)', 
                  fontsize=12, fontweight='medium')
    ax.set_title(f'Budget Allocation Strategy Across {n_subnets} Subnetworks\n' +
                 f'for {n_portfolios} Cost-Efficient Portfolios', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Create legend
    ax.legend(loc='upper left', 
              bbox_to_anchor=(1.02, 1),
              title='Subnetwork',
              fontsize=9,
              title_fontsize=10,
              framealpha=0.9)
    
    # Set limits and grid
    ax.set_xlim(budgets_sorted[0], budgets_sorted[-1])
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Add a subtle horizontal line at 50% for reference
    ax.axhline(y=50, color='gray', linestyle=':', alpha=0.3, linewidth=1)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Prepare subnetwork info for return
    subnetwork_info = {
        'names_ordered': subnetwork_names_ordered,
        'indices_ordered': subnetwork_order.tolist(),
        'first_funding': first_funding[subnet_order_indices].tolist(),
        'average_allocation': allocation_sorted_ordered.mean(axis=0).tolist()
    }
    
    return budgets_sorted, allocation_sorted_ordered, subnetwork_info


def print_allocation_statistics(budgets: np.ndarray, allocation_matrix: np.ndarray, subnetwork_info: dict):
    """
    Print detailed statistics about budget allocation.
    """
    print("\n" + "="*60)
    print("BUDGET ALLOCATION STATISTICS")
    print("="*60)
    
    n_portfolios = len(budgets)
    n_subnets = allocation_matrix.shape[1]
    
    print(f"\nTotal portfolios analyzed: {n_portfolios}")
    print(f"Budget range: {int(budgets[0])} to {int(budgets[-1])} switches")
    print(f"Number of subnetworks: {n_subnets}")
    
    print("\nSubnetwork funding priority (earliest to latest):")
    for i, (subnet_name, first_fund_idx) in enumerate(zip(subnetwork_info['names_ordered'], 
                                                          subnetwork_info['first_funding'])):
        if first_fund_idx < np.inf:
            budget_at_first = int(budgets[int(first_fund_idx)])
            print(f"  {i+1}. {subnet_name}: First funded at budget = {budget_at_first}")
        else:
            print(f"  {i+1}. {subnet_name}: Never funded in these portfolios")
    
    print("\nAverage budget allocation across all portfolios:")
    for i, subnet_name in enumerate(subnetwork_info['names_ordered']):
        avg_alloc = subnetwork_info['average_allocation'][i]
        print(f"  {subnet_name}: {avg_alloc:5.1f}%")
    
    # Calculate allocation at key budget levels
    key_budgets = [5, 10, 15, 20, 25, 30, 35, 40, 45]
    print("\nBudget allocation at key budget levels:")
    print("Budget | Top 3 subnetworks (with percentages)")
    print("-"*50)
    
    for target_budget in key_budgets:
        # Find closest portfolio to this budget
        idx = np.argmin(np.abs(budgets - target_budget))
        actual_budget = int(budgets[idx])
        allocation = allocation_matrix[idx, :]
        
        # Get top 3 subnetworks at this budget
        top_indices = np.argsort(allocation)[-3:][::-1]
        
        top_subnets = []
        for top_idx in top_indices:
            subnet_name = subnetwork_info['names_ordered'][top_idx]
            percentage = allocation[top_idx]
            if percentage > 0.1:  # Only show if > 0.1%
                top_subnets.append(f"{subnet_name} ({percentage:.1f}%)")
        
        if top_subnets:
            print(f"  {actual_budget:2d}   | {', '.join(top_subnets)}")

def plot_budget_allocation_stacked_bar_old(
    results_file: str = "model/results/whole_network_ce_portfolios.json",
    output_path: str = "budget_allocation_stacked_bar.pdf",
    figsize: tuple = (14, 7),
    colormap: str = 'tab20'
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Create a stacked bar chart showing budget allocation across subnetworks
    for all cost-efficient portfolios.
    
    Parameters:
    -----------
    results_file : str
        Path to the JSON file with combined portfolios
    output_path : str
        Path to save the output figure
    figsize : Tuple
        Figure size (width, height) in inches
    colormap : str
        Matplotlib colormap name
    
    Returns:
    --------
    budgets : np.ndarray
        Sorted budget values
    allocation_matrix : np.ndarray
        Allocation matrix
    subnetwork_info : Dict
        Information about subnetworks
    """
    
    # Read the combined portfolios data
    Q_star, combined_performances, combined_costs, dict_node_reinforcements, subnetworks = read_combined_portfolios(results_file)
    
    # Convert Q_star to list for easier processing
    portfolio_tuples = list(Q_star)
    
    # Get the number of subnetworks
    n_subnets = len(subnetworks)
    
    # Create subnetwork mapping (index to name)
    subnetwork_mapping = {i: name.upper() for i, name in enumerate(subnetworks)}
    
    # Initialize allocation matrix
    n_portfolios = len(portfolio_tuples)
    allocation_matrix = np.zeros((n_portfolios, n_subnets))
    budgets = np.zeros(n_portfolios)
    
    print(f"Processing {n_portfolios} cost-efficient portfolios...")
    print(f"Subnetworks: {[name.upper() for name in subnetworks]}")
    
    # Process each portfolio
    for i, portfolio_tuple in enumerate(portfolio_tuples):
        # Get total cost (number of reinforced switches)
        total_cost = combined_costs[portfolio_tuple][0]
        budgets[i] = total_cost
        
        # Process each subnetwork in the portfolio
        for sub_idx, sub_portfolio_int in enumerate(portfolio_tuple):
            subnetwork_name = subnetworks[sub_idx]
            node_reinforcements = dict_node_reinforcements.get(subnetwork_name, [])
            num_nodes = len(node_reinforcements)
            
            # Convert integer to binary representation of node selections
            binary_str = bin(sub_portfolio_int)[2:].zfill(num_nodes)
            
            # Count number of reinforced nodes in this subnetwork
            reinforced_nodes = sum(1 for bit in binary_str if bit == '1')
            
            # Each reinforced node costs 1 unit
            subnetwork_cost = reinforced_nodes
            
            allocation_matrix[i, sub_idx] = subnetwork_cost
    
    # Convert to percentages
    for i in range(n_portfolios):
        if budgets[i] > 0:
            allocation_matrix[i, :] = (allocation_matrix[i, :] / budgets[i]) * 100
    
    # Sort by budget
    sort_indices = np.argsort(budgets)
    budgets_sorted = budgets[sort_indices]
    allocation_sorted = allocation_matrix[sort_indices, :]
    
    # Determine subnetwork order by when they first receive funding
    first_funding = np.full(n_subnets, np.inf)
    for j in range(n_subnets):
        nonzero_indices = np.where(allocation_sorted[:, j] > 0)[0]
        if len(nonzero_indices) > 0:
            first_funding[j] = nonzero_indices[0]
    
    # Sort subnetworks by when they first get funded
    subnet_order_indices = np.argsort(first_funding)
    subnetwork_order = np.array(range(n_subnets))[subnet_order_indices]
    allocation_sorted_ordered = allocation_sorted[:, subnet_order_indices]
    
    # Get subnetwork names in the correct order
    subnetwork_names_ordered = [subnetwork_mapping[idx] for idx in subnetwork_order]
    
    # Create the plot
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=figsize)
    
    # Generate colors from colormap
    try:
        colormap_obj = plt.colormaps.get_cmap(colormap)
    except AttributeError:
        colormap_obj = plt.cm.get_cmap(colormap)
    
    colors = [colormap_obj(i) for i in np.linspace(0, 1, n_subnets)]
    
    # Create stacked bar chart
    bottom = np.zeros(len(budgets_sorted))
    
    # Plot each subnetwork's contribution
    for i in range(n_subnets):
        ax.bar(budgets_sorted, allocation_sorted_ordered[:, i], 
               bottom=bottom, 
               width=0.8,  # Width of each bar
               color=colors[i],
               edgecolor='white',
               linewidth=0.5,
               label=subnetwork_names_ordered[i])
        bottom += allocation_sorted_ordered[:, i]
    
    # Formatting
    ax.set_xlabel('Cost', 
                  fontsize=16, fontweight='medium')
    ax.set_ylabel('Percentage of Budget Allocated (%)', 
                  fontsize=16, fontweight='medium')
    #ax.set_title(f'Budget Allocation Across {n_subnets} Subnetworks\n' +
    #             f'for {n_portfolios} Cost-Efficient Portfolios', 
    #             fontsize=14, fontweight='bold', pad=20)
    
    # Set x-ticks to show every portfolio (every integer budget value)
    ax.set_xticks(budgets_sorted)
    
    # Only label every 5th tick to avoid clutter
    xtick_labels = []
    for i, budget in enumerate(budgets_sorted):
        if i % 5 == 0:  # Every 5th portfolio
            xtick_labels.append(str(int(budget)))
        else:
            #xtick_labels.append('')
            xtick_labels.append(str(int(budget)))
    
    ax.set_xticklabels(xtick_labels, fontsize=14)#, rotation=45)
    
    # Create legend
    ax.legend(loc='upper left', 
              bbox_to_anchor=(1.02, 1),
              title='Subnetwork',
              fontsize=12,
              title_fontsize=14,
              framealpha=0.9)
    
    # Set limits and grid
    ax.set_xlim(budgets_sorted[0] - 0.5, budgets_sorted[-1] + 0.5)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
    ax.set_axisbelow(True)
    
    # Add a subtle horizontal line at 50% for reference
    #ax.axhline(y=50, color='gray', linestyle=':', alpha=0.3, linewidth=1)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.show()
    
    # Prepare subnetwork info for return
    subnetwork_info = {
        'names_ordered': subnetwork_names_ordered,
        'indices_ordered': subnetwork_order.tolist(),
        'first_funding': first_funding[subnet_order_indices].tolist(),
        'average_allocation': allocation_sorted_ordered.mean(axis=0).tolist(),
        'total_budgets': budgets_sorted.tolist()
    }
    
    return budgets_sorted, allocation_sorted_ordered, subnetwork_info

def plot_budget_allocation_stacked_bar(
    results_file: str = "model/results/whole_network_ce_portfolios.json",
    output_path: str = "budget_allocation_stacked_bar.pdf",
    figsize: tuple = (14, 7),
    colormap: str = 'tab20'
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Create a stacked bar chart showing budget allocation across subnetworks
    for all cost-efficient portfolios.
    
    Parameters:
    -----------
    results_file : str
        Path to the JSON file with combined portfolios
    output_path : str
        Path to save the output figure
    figsize : Tuple
        Figure size (width, height) in inches
    colormap : str
        Matplotlib colormap name
    
    Returns:
    --------
    budgets : np.ndarray
        Sorted UNIQUE budget values
    allocation_matrix : np.ndarray
        AVERAGE allocation matrix (one row per unique budget)
    subnetwork_info : Dict
        Information about subnetworks
    """
    
    # Read the combined portfolios data
    Q_star, combined_performances, combined_costs, dict_node_reinforcements, subnetworks = read_combined_portfolios(results_file)
    
    # Convert Q_star to list for easier processing
    portfolio_tuples = list(Q_star)
    
    # Get the number of subnetworks
    n_subnets = len(subnetworks)
    
    # Create subnetwork mapping (index to name)
    subnetwork_mapping = {i: name.upper() for i, name in enumerate(subnetworks)}
    
    # Initialize allocation matrix
    n_portfolios = len(portfolio_tuples)
    allocation_matrix = np.zeros((n_portfolios, n_subnets))
    budgets = np.zeros(n_portfolios)
    
    print(f"Processing {n_portfolios} cost-efficient portfolios...")
    print(f"Subnetworks: {[name.upper() for name in subnetworks]}")
    
    # Process each portfolio
    for i, portfolio_tuple in enumerate(portfolio_tuples):
        # Get total cost (number of reinforced switches)
        total_cost = combined_costs[portfolio_tuple][0]
        budgets[i] = total_cost
        
        # Process each subnetwork in the portfolio
        for sub_idx, sub_portfolio_int in enumerate(portfolio_tuple):
            subnetwork_name = subnetworks[sub_idx]
            node_reinforcements = dict_node_reinforcements.get(subnetwork_name, [])
            num_nodes = len(node_reinforcements)
            
            # Convert integer to binary representation of node selections
            binary_str = bin(sub_portfolio_int)[2:].zfill(num_nodes)
            
            # Count number of reinforced nodes in this subnetwork
            reinforced_nodes = sum(1 for bit in binary_str if bit == '1')
            
            # Each reinforced node costs 1 unit
            subnetwork_cost = reinforced_nodes
            
            allocation_matrix[i, sub_idx] = subnetwork_cost
    
    # Convert to percentages
    for i in range(n_portfolios):
        if budgets[i] > 0:
            allocation_matrix[i, :] = (allocation_matrix[i, :] / budgets[i]) * 100
    
    # --- NEW: AVERAGE PORTFOLIOS WITH SAME BUDGET LEVEL ---
    print("\nAveraging portfolios with same budget level...")
    
    # Get unique budget values
    unique_budgets = np.unique(budgets)
    print(f"Found {len(unique_budgets)} unique budget levels (1-{int(max(budgets))})")
    
    # Initialize averaged allocation matrix
    avg_allocation_matrix = np.zeros((len(unique_budgets), n_subnets))
    
    # For each unique budget level, average all portfolios at that budget
    for i, budget in enumerate(unique_budgets):
        # Find all portfolios with this budget
        indices = np.where(budgets == budget)[0]
        
        if len(indices) > 1:
            print(f"  Budget {int(budget)}: {len(indices)} portfolios → averaging")
        
        # Average the allocations for this budget level
        avg_allocation_matrix[i, :] = np.mean(allocation_matrix[indices, :], axis=0)
    
    # Use averaged data for plotting
    budgets_sorted = unique_budgets
    allocation_sorted = avg_allocation_matrix
    n_budgets = len(unique_budgets)
    
    print(f"\nPlotting {n_budgets} bars (one per budget level)")
    # --- END OF NEW CODE ---
    
    # Determine subnetwork order by when they first receive funding
    first_funding = np.full(n_subnets, np.inf)
    for j in range(n_subnets):
        nonzero_indices = np.where(allocation_sorted[:, j] > 0)[0]
        if len(nonzero_indices) > 0:
            first_funding[j] = nonzero_indices[0]
    
    # Sort subnetworks by when they first get funded
    subnet_order_indices = np.argsort(first_funding)
    subnetwork_order = np.array(range(n_subnets))[subnet_order_indices]
    allocation_sorted_ordered = allocation_sorted[:, subnet_order_indices]
    
    # Get subnetwork names in the correct order
    subnetwork_names_ordered = [subnetwork_mapping[idx] for idx in subnetwork_order]
    
    # Create the plot
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=figsize)
    
    # Generate colors from colormap
    try:
        colormap_obj = plt.colormaps.get_cmap(colormap)
    except AttributeError:
        colormap_obj = plt.cm.get_cmap(colormap)
    
    colors = [colormap_obj(i) for i in np.linspace(0, 1, n_subnets)]
    
    # Create stacked bar chart
    bottom = np.zeros(len(budgets_sorted))
    
    # Plot each subnetwork's contribution
    for i in range(n_subnets):
        ax.bar(budgets_sorted, allocation_sorted_ordered[:, i], 
               bottom=bottom, 
               width=0.8,  # Width of each bar
               color=colors[i],
               edgecolor='white',
               linewidth=0.5,
               label=subnetwork_names_ordered[i])
        bottom += allocation_sorted_ordered[:, i]
    
    # Formatting
    ax.set_xlabel('Cost', 
                  fontsize=18, fontweight='medium')
    ax.set_ylabel('Percentage of Budget Allocated (%)', 
                  fontsize=18, fontweight='medium')
    
    # Set x-ticks to show every unique budget level
    ax.set_xticks(budgets_sorted)
    
    # Label every budget level
    #xtick_labels = [str(int(budget)) for budget in budgets_sorted]
    #ax.set_xticklabels(xtick_labels, fontsize=16)

    xtick_labels = []
    for i, budget in enumerate(budgets_sorted):
        if i % 5 == 0:  # Every 5th portfolio in the sorted list
            xtick_labels.append(str(int(budget)))
        else:
            xtick_labels.append('')

    ax.set_xticklabels(xtick_labels, fontsize=16)

    # Set y-tick labels with custom fontsize
    ax.set_yticks([0, 20, 40, 60, 80, 100])  # Set specific tick positions
    ax.set_yticklabels(['0', '20', '40', '60', '80', '100'], fontsize=16)


    
    # Create legend
    ax.legend(loc='upper left', 
              bbox_to_anchor=(1.02, 1),
              title='Subnetwork',
              fontsize=14,
              title_fontsize=18,
              framealpha=0.9)
    
    # Set limits and grid
    ax.set_xlim(budgets_sorted[0] - 0.5, budgets_sorted[-1] + 0.5)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
    ax.set_axisbelow(True)
    
    # Add vertical grid lines between bars for better distinction
    for x in budgets_sorted[:-1]:
        ax.axvline(x=x + 0.5, color='gray', linestyle=':', alpha=0.2, linewidth=0.5)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.show()
    
    # Prepare subnetwork info for return
    subnetwork_info = {
        'names_ordered': subnetwork_names_ordered,
        'indices_ordered': subnetwork_order.tolist(),
        'first_funding': first_funding[subnet_order_indices].tolist(),
        'average_allocation': allocation_sorted_ordered.mean(axis=0).tolist(),
        'unique_budgets': budgets_sorted.tolist(),
        'n_portfolios_per_budget': [(int(b), np.sum(budgets == b)) for b in unique_budgets]
    }
    
    # Print summary of portfolios per budget
    print("\nPortfolios per budget level:")
    for budget, count in subnetwork_info['n_portfolios_per_budget']:
        if count > 1:
            print(f"  Budget {budget}: {count} portfolios")
    
    return budgets_sorted, allocation_sorted_ordered, subnetwork_info


if __name__ == "__main__":


    network = True

    if network:
        Q_star, combined_performances, combined_costs, dict_node_reinforcements, subnetworks = read_combined_portfolios("model/results/whole_network_ce_portfolios.json")
        

        #normalize = False

        #if normalize:
        #    costs = []
        #    performances = []
        #    dim = None
        #    for Q in Q_star:
        #        dim = len(Q)
        #        break
        #    
        #    if dim is not None:
        #        base_performance = combined_performances[tuple([0 for _ in range(dim)])]

        #        for Q in Q_star:
        #            costs.append(combined_costs[Q])
        #            performances.append(combined_performances[Q] / base_performance - 1)

        #        plot_pareto_frontier(costs, performances)
        #
        #else:
        plot_pareto = False
        if plot_pareto:
            epsilon = 10
            filter_random = False
            random_filename = f"model/random_results/combined_portfolios_epsilon_{epsilon}.json"

            costs = []
            performances = []
            dim = None
            for Q in Q_star:
                dim = len(Q)
                break
            
            if dim is not None:
                for Q in Q_star:
                    costs.append(combined_costs[Q][0])
                    performances.append(combined_performances[Q])

                random_costs = []
                random_performances = []

                if filter_random:
                    seen: set[tuple[float, float]] = set()  # Store (cost, performance) pairs
                    seen_portfolios = set()
                    random_portfolios, dict_random_performances, dict_random_costs, dict_node_reinforcements, subnetworks = read_combined_portfolios("model/random_results/combined_portfolios.json")
                    random_costs = []
                    random_performances = []
                    
                    for portfolio in random_portfolios:
                        # Get the cost (first element of the tuple) which is number of reinforced switches
                        cost = dict_random_costs[portfolio][0]  # Changed: take first element
                        performance = dict_random_performances[portfolio]
                        
                        # Check if we've seen a similar point
                        is_similar = False
                        for seen_cost, seen_perf in seen:
                            # Check if cost is the same AND performance is within epsilon
                            if seen_cost == cost and abs(seen_perf - performance) < epsilon:
                                is_similar = True
                                break
                        
                        # Only add if it's not similar to any already seen point
                        if not is_similar:
                            random_costs.append(cost)  # Changed: store just the cost value
                            random_performances.append(performance)
                            seen.add((cost, performance))
                            seen_portfolios.add(portfolio)
                    
                    # Save filtered results
                    save_combined_portfolios(seen_portfolios, dict_random_performances, dict_random_costs, dict_node_reinforcements, subnetworks, random_filename)
                    # Re-read if needed
                    random_portfolios, dict_random_performances, dict_random_costs, dict_node_reinforcements, subnetworks = read_combined_portfolios(random_filename)
                    
                    plot_pareto_frontier(costs, performances, random_costs, random_performances)

                else:
                    random_portfolios, dict_random_performances, dict_random_costs, dict_node_reinforcements, subnetworks = read_combined_portfolios(random_filename)
                    random_costs = []
                    random_performances = []
                    for portfolio in random_portfolios:
                        random_costs.append(dict_random_costs[portfolio])
                        random_performances.append(dict_random_performances[portfolio])

                    plot_pareto_frontier(costs, performances, random_costs, random_performances)
        else:
            core_index = False
            if core_index:
                # Convert Q_star from dict keys to list if needed
                print("Core indexes: ")
                # Convert Q_star from set to list if needed
                Q_star = list(Q_star)
                
                # Select specific budget levels
                selected_budgets = [i for i in range(1, 46)]#[5, 10, 15, 25, 30, 35, 40, 45]
                
                # Compute core indexes for ALL switches
                core_indexes, budget_levels, all_switch_ids, index_to_info = compute_core_indexes_for_all_switches(
                    Q_star, combined_costs, dict_node_reinforcements, selected_budgets
                )
                
                print(f"\nTotal switches: {len(all_switch_ids)}")
                print(f"Budget levels: {budget_levels}")
                
                # Display some statistics
                print("\nTop 10 most frequently reinforced switches:")
                switch_stats = []
                for switch_id in all_switch_ids:
                    avg_core = np.mean(core_indexes[switch_id])
                    max_core = np.max(core_indexes[switch_id])
                    info = index_to_info[switch_id]
                    switch_stats.append((avg_core, max_core, switch_id, info))
                
                # Sort by average core index
                switch_stats.sort(reverse=True, key=lambda x: x[0])
                
                for i, (avg_core, max_core, switch_id, info) in enumerate(switch_stats[:10]):
                    print(f"{i+1}. Switch {info['original_id']} ({info['subnetwork']}): avg={avg_core:.3f}, max={max_core:.3f}")
                
                # Plot heatmap with ALL switches
                fig, ax = plot_core_indexes_clean_heatmap(core_indexes, budget_levels, all_switch_ids, index_to_info)
                #fig, ax = plot_core_indexes(core_indexes, budget_levels, all_switch_ids, index_to_info)

                plt.savefig("core_indexes_all_switches.png", dpi=500, bbox_inches='tight')
                plt.savefig("core_indexes_all_switches.pdf", dpi=500, bbox_inches='tight')
                plt.show()

            else:
                print("\n" + "="*60)
                print("GENERATING BUDGET ALLOCATION ANALYSIS")
                print("="*60)
                
                # Generate budget allocation plot
                budgets, allocation_matrix, subnetwork_info = plot_budget_allocation(
                    results_file="model/results/whole_network_ce_portfolios.json",
                    output_path="budget_allocation_stacked.pdf",
                    figsize=(12, 7)
                )
                
                # Print detailed statistics
                print_allocation_statistics(budgets, allocation_matrix, subnetwork_info)
                
                # Also save the allocation data for later use
                allocation_data = {
                    'budgets': budgets.tolist(),
                    'allocation_matrix': allocation_matrix.tolist(),
                    'subnetwork_info': subnetwork_info
                }
                
                import json
                with open('budget_allocation_data.json', 'w') as f:
                    json.dump(allocation_data, f, indent=2)
                
                print("\nBudget allocation data saved to 'budget_allocation_data.json'")

                budgets, allocation_matrix, subnetwork_info = plot_budget_allocation_stacked_bar(
                    results_file="model/results/whole_network_ce_portfolios.json",
                    output_path="budget_allocation_stacked_bar.pdf",
                    figsize=(18, 9)
                )

    else:
        subnetworks = ["apt", "jki", "knh", "kuo", "lna", "sij", "skm", "sor", "te", "toi"]
        for sub_network in subnetworks:
            Q, portfolio_performances, portfolio_costs, _ = read_cost_efficient_portfolios(f"model/results/{sub_network}_ce_portfolios.json")

            costs = []
            performances = []
            
            for q in Q:
                costs.append(portfolio_costs[q][0])
                performances.append(portfolio_performances[q])

            plot_subnetwork_pareto_frontier(costs, performances, sub_network)
