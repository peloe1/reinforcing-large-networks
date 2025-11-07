import json
from datetime import datetime
from subnetwork import bitmask_to_portfolio

def portfolio_to_readable(portfolio: int, node_reinforcements: list[tuple[str, float]]) -> dict:
    """Convert a portfolio bitmask to a readable dictionary format."""
    portfolio_list = bitmask_to_portfolio(portfolio, len(node_reinforcements))
    reinforced_nodes = []
    
    for i, (node, _) in enumerate(node_reinforcements):
        if portfolio_list[i] == 1:
            reinforced_nodes.append(node)
    
    return {
        'portfolio_id': portfolio,
        'binary_vector': portfolio_list,
        'reinforced_nodes': reinforced_nodes
    }

def combined_portfolio_to_readable(combined_portfolio: tuple[int, ...], 
                                 dict_node_reinforcements: dict[str, list[tuple[str, float]]],
                                 subnetworks: list[str]) -> dict:
    """Convert a combined portfolio tuple to a readable dictionary format."""
    result = {
        'combined_portfolio_id': combined_portfolio,
        'subnetworks': {}
    }
    
    total_reinforced_nodes = []
    total_binary_vector = []
    
    for j, portfolio in enumerate(combined_portfolio):
        subnetwork = subnetworks[j]
        if portfolio > 0:  # Only process non-zero portfolios
            subnetwork_data = portfolio_to_readable(portfolio, dict_node_reinforcements[subnetwork])
            result['subnetworks'][subnetwork] = subnetwork_data
            total_reinforced_nodes.extend(subnetwork_data['reinforced_nodes'])
            total_binary_vector.append({
                'subnetwork': subnetwork,
                'binary_vector': subnetwork_data['binary_vector']
            })
    
    result['total_reinforced_nodes'] = total_reinforced_nodes
    result['total_binary_vectors'] = total_binary_vector
    
    return result

def save_cost_efficient_portfolios(Q: set[int], 
                                  performances: dict[int, float], 
                                  portfolio_costs: dict[int, list[float]], 
                                  node_reinforcements: list[tuple[str, float]],
                                  filename: str):
    """
    Save cost-efficient portfolios to a JSON file.
    
    Parameters:
        Q: Set of cost-efficient portfolio bitmasks
        performances: Dictionary of portfolio performances
        portfolio_costs: Dictionary of portfolio costs
        node_reinforcements: List of node reinforcement actions
        filename: Output filename (optional)
    """
    
    # Prepare the data for JSON serialization
    output_data = {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'total_portfolios': len(Q),
            'number_of_actions': len(node_reinforcements)
        },
        'node_reinforcements': node_reinforcements,
        'cost_efficient_portfolios': []
    }
    
    # Add each portfolio with its details
    for portfolio in sorted(Q):
        portfolio_data = portfolio_to_readable(portfolio, node_reinforcements)
        portfolio_data.update({
            'performance': performances.get(portfolio, 0.0),
            'cost': portfolio_costs.get(portfolio, [0.0])[0] if portfolio in portfolio_costs else 0.0,
            'total_cost': sum(portfolio_costs.get(portfolio, [0.0]))
        })
        output_data['cost_efficient_portfolios'].append(portfolio_data)
    
    # Write to JSON file
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(output_data, file, indent=2, ensure_ascii=False)
    
    print(f"✅ Saved {len(Q)} cost-efficient portfolios to {filename}")
    return filename

def save_combined_portfolios(Q_star: set[tuple[int, ...]], 
                             performances: dict[tuple[int, ...], float], 
                             combined_costs: dict[tuple[int, ...], list[float]],
                             dict_node_reinforcements: dict[str, list[tuple[str, float]]],
                             subnetworks: list[str],
                             filename: str) -> str:
    """
    Save combined cost-efficient portfolios to a JSON file.
    
    Parameters:
        Q_star: Set of cost-efficient combined portfolio tuples
        performances: Dictionary of portfolio performances
        combined_costs: Dictionary of portfolio costs for each resource
        dict_node_reinforcements: Dictionary mapping subnetwork to node reinforcement actions
        subnetworks: List of subnetwork identifiers
        filename: Output filename (optional)
    """
    
    # Prepare the data for JSON serialization
    output_data = {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'total_combined_portfolios': len(Q_star),
            'number_of_subnetworks': len(subnetworks),
            'subnetworks': subnetworks,
            'budget_dimensions': len(next(iter(combined_costs.values()))) if combined_costs else 0
        },
        'node_reinforcements_by_subnetwork': dict_node_reinforcements,
        'cost_efficient_combined_portfolios': []
    }
    
    # Add each combined portfolio with its details
    for portfolio in sorted(Q_star, key=lambda x: performances.get(x, 0.0), reverse=True):
        portfolio_data = combined_portfolio_to_readable(portfolio, dict_node_reinforcements, subnetworks)
        
        # Add performance and cost information
        portfolio_data.update({
            'performance': performances.get(portfolio, 0.0),
            'cost_per_resource': combined_costs.get(portfolio, [0.0]),
            'total_cost': sum(combined_costs.get(portfolio, [0.0]))
        })
        
        output_data['cost_efficient_combined_portfolios'].append(portfolio_data)
    
    # Write to JSON file
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(output_data, file, indent=2, ensure_ascii=False)
    
    print(f"✅ Saved {len(Q_star)} combined cost-efficient portfolios to {filename}")
    return filename
    
    