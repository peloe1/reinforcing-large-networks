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

def save_cost_efficient_portfolios(Q: set[int], 
                                  performances: dict[int, float], 
                                  portfolio_costs: dict[int, list[float]], 
                                  node_reinforcements: list[tuple[str, float]],
                                  filename: str = None):
    """
    Save cost-efficient portfolios to a JSON file.
    
    Parameters:
        Q: Set of cost-efficient portfolio bitmasks
        performances: Dictionary of portfolio performances
        portfolio_costs: Dictionary of portfolio costs
        node_reinforcements: List of node reinforcement actions
        filename: Output filename (optional)
    """
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"cost_efficient_portfolios_{timestamp}.json"
    
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