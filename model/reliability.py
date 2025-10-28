from queue import Queue
import networkx as nx
import itertools
import random
import time
import numpy as np
import matplotlib.pyplot as plt

from path import feasible_paths, terminal_pairs, old_feasible_paths
from graph import generate_random_graph_with_positions
from plotting import plot_network

# Correct implementation
def probability_of_event(G: nx.Graph, event: dict[str, int]) -> float:
    """
        Parameters:
        -----------
        G: nx.Graph
            Graph object
        event: dict[str, int]
            Event as dict consisting of node ids and the states of the nodes 
            +1 represents that the node is used
            -1 represents that the node is disrupted
            0 represents that the node is not used

        Returns:
        --------
        prob: float
            Probability of the state
    """

    prob = 1.0
    for node, e in event.items():
        if e == 1:
            prob *= G.nodes[node]['reliability']
        elif e == -1:
            prob *= (1 - G.nodes[node]['reliability'])
    return prob

def old_terminal_pair_reliability(G: nx.Graph, P_t: list[np.ndarray]) -> float:
    """
        Final corrected Dotson algorithm with improved state exploration
    """
    if len(P_t) == 0:
        return 0.0
    
    R_t = 0.0
    node_list = list(G.nodes())
    
    # Create initial state where all nodes are unused (0)
    X_0 = {node: 0 for node in node_list}
    
    Q = Queue()
    Q.put(X_0)
    
    visited = set()
    visited.add(frozenset(X_0.items()))
    
    # Sort paths by length to prioritize shorter paths
    paths_sorted = sorted(P_t, key=lambda path: len(path))
    
    while not Q.empty():
        X = Q.get()
        
        # Find the first operational path in current state
        operational_path = None
        for path in paths_sorted:
            if all(X[node] != -1 for node in path):
                operational_path = path
                break
        
        if operational_path is not None:
            # Create new state where this path is operational
            X1 = X.copy()
            for node in operational_path:
                X1[node] = 1
            
            # Add probability of this state
            prob = probability_of_event(G, X1)
            R_t += prob
            
            # Generate complement states by disrupting each node in the path
            for node in operational_path:
                complement = X1.copy()
                complement[node] = -1  # Disrupt this specific node
                
                # Reset ALL nodes from the current operational path
                # This is important to avoid carrying over used states
                for path_node in operational_path:
                    complement[path_node] = 0
                
                # Set the disrupted node back to -1
                complement[node] = -1
                
                comp_hash = frozenset(complement.items())
                if comp_hash not in visited:
                    visited.add(comp_hash)
                    Q.put(complement)
    
    return R_t


# Modified Dotson algorithm
# TODO: Perhaps add the list of disruption parameters as a parameter to avoid the G.nodes[node]['reliability'] lookup, which may be unoptimal.
def terminal_pair_reliability(G: nx.Graph, P_t: list[np.ndarray]) -> float:
    """
        Parameters:
        -----------
        G: nx.Graph
            Graph object
        P_t: list[np.ndarray]
            List of feasible paths between some particular terminal pair t represented with a NumPy array of 1D NumPy arrays containing integers
        
        Returns:
        --------
        R: float
            Terminal pair reliability
    """
    if len(P_t) == 0:
        return 0.0
    
    node_list = G.nodes()
    
    R_t = 0
    X_0 = {node: 0 for node in G.nodes()}

    Q = Queue()
    Q.put(X_0)

    visited = set()
    visited.add(tuple(X_0[node] for node in node_list))

    paths: list[np.ndarray] = sorted(P_t, key = lambda path: len(path))
    while not Q.empty():
        X = Q.get()

        for path in paths:
            if all(X[node] != -1 for node in path):
                shortest_path = path #paths.pop(i)
                X1 = X.copy()
                #X1[shortest_path] = 1
                for node in shortest_path:
                    X1[node] = 1
                t = tuple(X1[node] for node in node_list)
                if t not in visited:
                    visited.add(t)
                    #event_list = [(node, state) for node, state in X1.items()] # Avoid this, modify probability of event
                    #R_t += probability_of_event(G, event_list)
                    R_t += probability_of_event(G, X1)

                    #for j in range(2**len(shortest_path)):
                    #    complement = X1.copy()
                    #    for k, node in enumerate(shortest_path):
                    #        complement[node] = 2 * ((j >> k) & 1) - 1
                    #    
                    #    t = tuple(sorted(complement.items()))
                    #    if t not in visited:
                    #        visited.add(t)
                    #        Q.put(complement)

                    # Add the complement events to the queue
                    for node in shortest_path:
                        complement = X1.copy()
                        
                        #complement[complement == 1] = 0
                        for n in shortest_path: #complement:
                           if complement[n] == 1:
                                complement[n] = 0
                        
                        complement[node] = -1
    
                        t = tuple(complement[node] for node in node_list)
                        if t not in visited:
                            visited.add(t)
                            Q.put(complement)
                        
                break

    return R_t

# Redundant helper function
def probability_of_state(G: nx.Graph, state: list[tuple[str, int]]) -> float:
    """
        Parameters:
        -----------
        G: nx.Graph
            Graph object
        state: list[tuple[str, int]]
            List of states

        Returns:
        --------
        prob: float
            Probability of the state
    """

    prob = 1
    for node, e in state:
        prob *= e * G.nodes[node]['reliability'] + (1 - e) * (1 - G.nodes[node]['reliability'])
    return prob

def brute_force_reliability(G: nx.Graph, feasiblePaths: list[np.ndarray]) -> float:
    """
        Corrected brute force reliability calculation
    """
    if len(feasiblePaths) == 0:
        return 0.0
        
    node_list = list(G.nodes())
    n = len(node_list)
    R = 0.0
    
    # Generate all possible node states (working/failed)
    for i in range(2 ** n):
        # Create state vector where:
        # 0 = node is working, 1 = node is failed
        node_failed = {}
        for j, node in enumerate(node_list):
            node_failed[node] = (i >> j) & 1
        
        # Check if any path is operational (all nodes in path are working)
        operational = False
        for path in feasiblePaths:
            if all(node_failed[node] == 0 for node in path):
                operational = True
                break
        
        if operational:
            # Calculate probability of this exact state
            # All failed nodes contribute (1-reliability)
            # All working nodes contribute reliability
            prob = 1.0
            for node in node_list:
                if node_failed[node] == 1:  # Node failed
                    prob *= (1 - G.nodes[node]['reliability'])
                else:  # Node working
                    prob *= G.nodes[node]['reliability']
            R += prob
    
    return R

def compute_speedup_and_accuracy_improved(num_trials=50):
    """
        Improved testing with better numerical stability
    """
    speedups = []
    accuracies = []
    errors = []
    
    for trial in range(num_trials):
        try:
            # Use smaller graphs for more reliable testing
            nodes = random.randint(10, 15)
            edges = random.randint(nodes-1, min(2*nodes, nodes*(nodes-1)//2))
            
            G = generate_random_graph_with_positions(nodes, edges, (-5, 5))
            
            # Ensure reasonable reliabilities and use consistent precision
            for node in G.nodes():
                G.nodes[node]['reliability'] = round(random.uniform(0.7, 0.95), 2)
            
            # Try multiple terminal pairs
            terminal_pairs_to_try = list(itertools.combinations(G.nodes(), 2))
            random.shuffle(terminal_pairs_to_try)
            
            dotson_result = None
            brute_result = None
            
            for term1, term2 in terminal_pairs_to_try[:3]:
                try:
                    paths = list(nx.all_simple_paths(G, term1, term2, cutoff=min(6, nodes)))
                    if len(paths) > 0 and len(paths) < 50:  # Limit path complexity
                        paths_array = [np.array(path) for path in paths]
                        
                        dotson_result = terminal_pair_reliability(G, paths_array)
                        brute_result = brute_force_reliability(G, paths_array)
                        break
                except:
                    continue
            
            if dotson_result is None or brute_result is None:
                continue
                
            # Use relative error for better comparison
            if brute_result > 1e-10:  # Avoid division by zero
                relative_error = abs(dotson_result - brute_result) / brute_result
            else:
                relative_error = abs(dotson_result - brute_result)
            
            accuracy = 1.0 if relative_error < 1e-6 else 0.0
            
            # Timing comparison
            paths_for_timing = [np.array(path) for path in paths]
            
            dotson_time = timeit.timeit(
                lambda: terminal_pair_reliability(G, paths_for_timing), 
                number=3
            ) / 3  # Average over 3 runs
            
            brute_time = timeit.timeit(
                lambda: brute_force_reliability(G, paths_for_timing), 
                number=3  
            ) / 3
            
            speedup = brute_time / max(dotson_time, 1e-10)
            
            speedups.append(min(speedup, 10000))
            accuracies.append(accuracy)
            errors.append(relative_error)
            
            print(f"Trial {trial+1}: RelError = {relative_error:.8f}, "
                  f"Dotson = {dotson_result:.6f}, Brute = {brute_result:.6f}, "
                  f"Speedup = {speedup:.2f}x")
                  
            if relative_error > 1e-6:
                print(f"  DISCREPANCY: Trying simple debug...")
                debug_simple_case(G, paths_array, dotson_result, brute_result)
                
        except Exception as e:
            print(f"Trial {trial+1} failed: {e}")
            continue
    
    if not speedups:
        return 0.0, 0.0, 0.0
    
    avg_speedup = np.mean(speedups)
    avg_accuracy = np.mean(accuracies) 
    avg_error = np.mean(errors)
    
    print(f"\n=== IMPROVED RESULTS ===")
    print(f"Trials: {len(speedups)}")
    print(f"Average Speedup: {avg_speedup:.2f}x")
    print(f"Average Accuracy: {avg_accuracy:.2%}")
    print(f"Average Relative Error: {avg_error:.8f}")
    
    return avg_speedup, avg_accuracy, avg_error

def debug_simple_case(G, paths, dotson_result, brute_result):
    """Debug with a simpler approach"""
    print(f"Simple debug: {len(G.nodes())} nodes, {len(paths)} paths")
    
    # Check if it's a simple case we can verify manually
    if len(paths) <= 3:
        print("Paths:")
        for i, path in enumerate(paths):
            path_rel = 1.0
            for node in path:
                path_rel *= G.nodes[node]['reliability']
            print(f"  Path {i}: {path} (reliability: {path_rel:.6f})")
        
        # For 2 paths, calculate union manually
        if len(paths) == 2:
            p1, p2 = paths
            rel1 = probability_of_event(G, {node: 1 for node in p1})
            rel2 = probability_of_event(G, {node: 1 for node in p2})
            
            common_nodes = set(p1) & set(p2)
            rel_intersection = 1.0
            for node in common_nodes:
                rel_intersection *= G.nodes[node]['reliability']
            
            manual_union = rel1 + rel2 - rel_intersection
            print(f"Manual union: {manual_union:.6f}")
            print(f"Dotson - Manual: {dotson_result - manual_union:.6f}")
            print(f"Brute - Manual: {brute_result - manual_union:.6f}")
            
def compute_speedup_and_accuracy(num_trials=10):
    """
        Enhanced testing with detailed diagnostics
    """
    speedups = []
    accuracies = []
    errors = []
    
    for trial in range(num_trials):
        try:
            # Small graphs for feasible brute force computation
            nodes = random.randint(10, 16)
            edges = random.randint(nodes-1, min(2*nodes, nodes*(nodes-1)//2))
            
            G = generate_random_graph_with_positions(nodes, edges, (-5, 5))
            
            # Ensure reasonable reliabilities
            for node in G.nodes():
                G.nodes[node]['reliability'] = random.uniform(0.7, 0.95)
            
            # Try multiple terminal pairs to find one with paths
            terminal_pairs_to_try = list(itertools.combinations(G.nodes(), 2))
            random.shuffle(terminal_pairs_to_try)
            
            dotson_result = None
            brute_result = None
            
            for term1, term2 in terminal_pairs_to_try[:3]:  # Try up to 3 pairs
                try:
                    paths = list(nx.all_simple_paths(G, term1, term2, cutoff=min(6, nodes)))
                    if len(paths) > 0:
                        paths_array = [np.array(path) for path in paths]
                        
                        dotson_result = terminal_pair_reliability(G, paths_array)
                        brute_result = brute_force_reliability(G, paths_array)
                        break
                except:
                    continue
            
            if dotson_result is None or brute_result is None:
                continue
                
            error = abs(dotson_result - brute_result)
            accuracy = 1.0 if error < 1e-6 else 0.0
            
            # For timing comparison (use perf_counter for better precision)
            paths_for_timing = [np.array(path) for path in paths]
            
            dotson_time = timeit.timeit(
                lambda: terminal_pair_reliability(G, paths_for_timing), 
                number=1
            )
            brute_time = timeit.timeit(
                lambda: brute_force_reliability(G, paths_for_timing), 
                number=1
            )
            
            speedup = brute_time / max(dotson_time, 1e-10)
            
            speedups.append(min(speedup, 10000))  # Cap unrealistic speedups
            accuracies.append(accuracy)
            errors.append(error)
            
            print(f"Trial {trial+1}: Error = {error:.8f}, "
                  f"Dotson = {dotson_result:.6f}, Brute = {brute_result:.6f}, "
                  f"Speedup = {speedup:.2f}x")
                  
            if error > 1e-6:
                print(f"  DISCREPANCY FOUND! Trying to debug...")
                debug_discrepancy(G, paths_array, dotson_result, brute_result)
                
        except Exception as e:
            print(f"Trial {trial+1} failed: {e}")
            continue
    
    if not speedups:
        return 0.0, 0.0, 0.0
    
    avg_speedup = np.mean(speedups)
    avg_accuracy = np.mean(accuracies) 
    avg_error = np.mean(errors)
    
    print(f"\n=== RESULTS ===")
    print(f"Trials: {len(speedups)}")
    print(f"Average Speedup: {avg_speedup:.2f}x")
    print(f"Average Accuracy: {avg_accuracy:.2%}")
    print(f"Average Error: {avg_error:.8f}")
    
    return avg_speedup, avg_accuracy, avg_error

def debug_discrepancy(G, paths, dotson_result, brute_result):
    """Debug when algorithms give different results"""
    print("Debugging discrepancy:")
    print(f"Graph: {len(G.nodes())} nodes, {len(G.edges())} edges")
    print(f"Paths: {len(paths)}")
    for i, path in enumerate(paths):
        print(f"  Path {i}: {path}")
    
    # Check individual path reliabilities
    for i, path in enumerate(paths):
        path_rel = 1.0
        for node in path:
            path_rel *= G.nodes[node]['reliability']
        print(f"  Path {i} reliability: {path_rel:.6f}")
    
    # Manual calculation for 2-path case
    if len(paths) == 2:
        p1, p2 = paths
        # Calculate union probability manually
        rel1 = probability_of_state_unified(G, {node: 1 for node in p1})
        rel2 = probability_of_state_unified(G, {node: 1 for node in p2})
        
        # Find intersection nodes
        common_nodes = set(p1) & set(p2)
        rel_intersection = 1.0
        for node in common_nodes:
            rel_intersection *= G.nodes[node]['reliability']
        
        union_rel = rel1 + rel2 - rel_intersection
        print(f"Manual union: {union_rel:.6f}")

# Run the improved test
if __name__ == '__main__':
    import timeit
    speedup, accuracy, error = compute_speedup_and_accuracy_improved(num_trials=100)