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

def read_cost_efficient_portfolios(filename: str) -> tuple[set[int], dict[int, float], dict[int, list[float]], list[tuple[str, float]]]:
    """
    Read cost-efficient portfolios from a JSON file.
    
    Parameters:
        filename: Input filename
        
    Returns:
        Tuple of (Q, performances, portfolio_costs, node_reinforcements) where:
        - Q: Set of cost-efficient portfolio bitmasks
        - performances: Dictionary of portfolio performances
        - portfolio_costs: Dictionary of portfolio costs
        - node_reinforcements: List of node reinforcement actions
    """
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        # Extract node_reinforcements
        node_reinforcements = []
        reinforcements_data = data.get('node_reinforcements', [])
        for reinforcement in reinforcements_data:
            if isinstance(reinforcement, list) and len(reinforcement) == 2:
                node_reinforcements.append((reinforcement[0], reinforcement[1]))
            else:
                print(f"⚠️ Warning: Invalid reinforcement format {reinforcement}, skipping")
        
        # Extract portfolios and their details
        Q = set()
        performances = {}
        portfolio_costs = {}
        
        portfolios_data = data.get('cost_efficient_portfolios', [])
        for portfolio_data in portfolios_data:
            portfolio_id = portfolio_data.get('portfolio_id')
            if portfolio_id is not None:
                Q.add(portfolio_id)
                performances[portfolio_id] = portfolio_data.get('performance', 0.0)
                
                # Reconstruct cost vector - note: we need to handle single cost vs cost vector
                cost = portfolio_data.get('cost', 0.0)
                # If the original had a list of costs, we need to reconstruct it
                # Since the save function stores both 'cost' and 'total_cost', 
                # we'll return a single-element list for compatibility
                portfolio_costs[portfolio_id] = [cost]
        
        print(f"✅ Loaded {len(Q)} cost-efficient portfolios from {filename}")
        print(f"✅ Loaded {len(node_reinforcements)} node reinforcement actions")
        
        return Q, performances, portfolio_costs, node_reinforcements
        
    except FileNotFoundError:
        print(f"❌ Error: File {filename} not found")
        return set(), {}, {}, []
    except json.JSONDecodeError:
        print(f"❌ Error: Invalid JSON format in {filename}")
        return set(), {}, {}, []
    except Exception as e:
        print(f"❌ Error reading {filename}: {str(e)}")
        return set(), {}, {}, []

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

def read_combined_portfolios(filename: str) -> tuple[set[tuple[int, ...]], 
                                                   dict[tuple[int, ...], float], 
                                                   dict[tuple[int, ...], list[float]],
                                                   dict[str, list[tuple[str, float]]],
                                                   list[str]]:
    """
    Read combined cost-efficient portfolios from a JSON file.
    
    Parameters:
        filename: Input filename
        
    Returns:
        Tuple of (Q_star, performances, combined_costs, dict_node_reinforcements, subnetworks) where:
        - Q_star: Set of cost-efficient combined portfolio tuples
        - performances: Dictionary of portfolio performances
        - combined_costs: Dictionary of portfolio costs for each resource
        - dict_node_reinforcements: Dictionary mapping subnetwork to node reinforcement actions
        - subnetworks: List of subnetwork identifiers
    """
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        # Extract metadata
        metadata = data.get('metadata', {})
        subnetworks = metadata.get('subnetworks', [])
        
        # Extract node_reinforcements by subnetwork
        dict_node_reinforcements = {}
        reinforcements_data = data.get('node_reinforcements_by_subnetwork', {})
        for subnetwork, reinforcement_list in reinforcements_data.items():
            dict_node_reinforcements[subnetwork] = []
            for reinforcement in reinforcement_list:
                if isinstance(reinforcement, list) and len(reinforcement) == 2:
                    dict_node_reinforcements[subnetwork].append((reinforcement[0], reinforcement[1]))
                else:
                    print(f"⚠️ Warning: Invalid reinforcement format {reinforcement} for {subnetwork}, skipping")
        
        # Extract combined portfolios and their details
        Q_star = set()
        performances = {}
        combined_costs = {}
        
        portfolios_data = data.get('cost_efficient_combined_portfolios', [])
        for portfolio_data in portfolios_data:
            portfolio_tuple_str = portfolio_data.get('combined_portfolio_id')
            if portfolio_tuple_str is not None:
                # Convert tuple string representation back to actual tuple
                try:
                    # Handle both string representation and direct list
                    if isinstance(portfolio_tuple_str, str):
                        # Convert string like "(0, 1, 2)" to tuple
                        portfolio_tuple = tuple(map(int, portfolio_tuple_str.strip('()').split(',')))
                    else:
                        # Assume it's already a list in the JSON
                        portfolio_tuple = tuple(portfolio_tuple_str)
                    
                    Q_star.add(portfolio_tuple)
                    performances[portfolio_tuple] = portfolio_data.get('performance', 0.0)
                    combined_costs[portfolio_tuple] = portfolio_data.get('cost_per_resource', [0.0])
                    
                except (ValueError, AttributeError) as e:
                    print(f"⚠️ Warning: Could not parse portfolio tuple {portfolio_tuple_str}, skipping: {e}")
        
        print(f"✅ Loaded {len(Q_star)} combined cost-efficient portfolios from {filename}")
        print(f"✅ Loaded {len(subnetworks)} subnetworks: {subnetworks}")
        
        return Q_star, performances, combined_costs, dict_node_reinforcements, subnetworks
        
    except FileNotFoundError:
        print(f"❌ Error: File {filename} not found")
        return set(), {}, {}, {}, []
    except json.JSONDecodeError:
        print(f"❌ Error: Invalid JSON format in {filename}")
        return set(), {}, {}, {}, []
    except Exception as e:
        print(f"❌ Error reading {filename}: {str(e)}")
        return set(), {}, {}, {}, []

def save_feasible_paths(paths: dict[tuple[str, str], list[list[str]]], filename: str):
    path_dict = {}
    for (s, t), path_list in paths.items():
        path_dict[f"({s}, {t})"] = path_list

    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(path_dict, file, ensure_ascii=False)
    
    print(f"✅ Saved {len(paths)} source-target pairs with feasible paths to {filename}")
    
    return filename

def read_feasible_paths(filename: str) -> dict[tuple[str, str], list[list[str]]]:
    """
    Read feasible paths from a JSON file.
    
    Parameters:
        filename: Input filename
        
    Returns:
        Dictionary mapping (source, target) tuples to lists of feasible paths
    """
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        # Convert string keys back to tuple format
        paths = {}
        for key_str, path_list in data.items():
            # Remove the parentheses and split by comma
            # Key is stored as "(source, target)" - remove parentheses and strip whitespace
            key_str_clean = key_str.strip('() ')
            parts = key_str_clean.split(',')
            
            if len(parts) == 2:
                source = parts[0].strip().strip("'\"")
                target = parts[1].strip().strip("'\"")
                paths[(source, target)] = path_list
            else:
                print(f"⚠️ Warning: Could not parse key '{key_str}', skipping")
        
        print(f"✅ Loaded {len(paths)} source-target pairs with feasible paths from {filename}")
        return paths
        
    except FileNotFoundError:
        print(f"❌ Error: File {filename} not found")
        return {}
    except json.JSONDecodeError:
        print(f"❌ Error: Invalid JSON format in {filename}")
        return {}
    except Exception as e:
        print(f"❌ Error reading {filename}: {str(e)}")
        return {}
    
def save_terminal_pair_reliabilities(subnetwork: str, Q: set[int], reliabilities: dict[int, dict[tuple[str, str], float]], filename: str):
    """
    Save terminal pair reliabilities to a JSON file.
    
    Parameters:
        subnetwork: Subnetwork identifier
        reliabilities: Dictionary mapping portfolio IDs to dictionaries of terminal pair reliabilities
        filename: Output filename
    """
    
    # Prepare the data for JSON serialization
    output_data = {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'subnetwork': subnetwork,
            'total_portfolios': len(reliabilities),
            'total_terminal_pairs': len(next(iter(reliabilities.values()))) if reliabilities else 0
        },
        'terminal_pair_reliabilities_by_portfolio': {}
    }
    
    # Convert the nested structure for JSON serialization
    for portfolio_id in sorted(Q):
        pair_reliabilities = reliabilities[portfolio_id]
        portfolio_key = str(portfolio_id)
        output_data['terminal_pair_reliabilities_by_portfolio'][portfolio_key] = {}
        
        # Convert tuple keys to string keys for JSON serialization
        for (source, target), reliability in pair_reliabilities.items():
            key_str = f"({source}, {target})"
            output_data['terminal_pair_reliabilities_by_portfolio'][portfolio_key][key_str] = reliability
    
    # Write to JSON file
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(output_data, file, indent=2, ensure_ascii=False)
    
    print(f"✅ Saved {len(reliabilities)} portfolio reliabilities for subnetwork {subnetwork} to {filename}")
    return filename

def read_terminal_pair_reliabilities(filename: str) -> tuple[str, dict[int, dict[tuple[str, str], float]]]:
    """
    Read terminal pair reliabilities from a JSON file.
    
    Parameters:
        filename: Input filename
        
    Returns:
        Tuple of (subnetwork, reliabilities_dict) where reliabilities_dict maps 
        portfolio IDs to dictionaries of terminal pair reliabilities
    """
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        # Extract metadata
        subnetwork = data.get('metadata', {}).get('subnetwork', 'unknown')
        
        # Convert the nested structure back to original format
        reliabilities = {}
        reliabilities_data = data.get('terminal_pair_reliabilities_by_portfolio', {})
        
        for portfolio_str, pair_reliabilities_data in reliabilities_data.items():
            try:
                portfolio_id = int(portfolio_str)
                reliabilities[portfolio_id] = {}
                
                # Convert string keys back to tuple format
                for key_str, reliability in pair_reliabilities_data.items():
                    # Remove the parentheses and split by comma
                    # Key is stored as "(source, target)" - remove parentheses and strip whitespace
                    key_str_clean = key_str.strip('() ')
                    parts = key_str_clean.split(',')
                    
                    if len(parts) == 2:
                        source = parts[0].strip().strip("'\"")
                        target = parts[1].strip().strip("'\"")
                        reliabilities[portfolio_id][(source, target)] = reliability
                    else:
                        print(f"⚠️ Warning: Could not parse key '{key_str}' for portfolio {portfolio_id}, skipping")
                        
            except ValueError:
                print(f"⚠️ Warning: Could not parse portfolio ID '{portfolio_str}', skipping")
        
        print(f"✅ Loaded {len(reliabilities)} portfolio reliabilities for subnetwork {subnetwork} from {filename}")
        return subnetwork, reliabilities
        
    except FileNotFoundError:
        print(f"❌ Error: File {filename} not found")
        return '', {}
    except json.JSONDecodeError:
        print(f"❌ Error: Invalid JSON format in {filename}")
        return '', {}
    except Exception as e:
        print(f"❌ Error reading {filename}: {str(e)}")
        return '', {}