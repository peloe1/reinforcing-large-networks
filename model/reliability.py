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


# Modified Dotson algorithm
# TODO: Perhaps add the list of disruption parameters as a parameter to avoid the G.nodes[node]['reliability'] lookup, which may be unoptimal.
def terminal_pair_reliability(G: nx.Graph, P_t: list[list[str]]) -> float:
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
    # Initialize reliability
    R_t = 0.0
    
    # If no paths exist between the terminal pair 't', the reliability must be zero
    if len(P_t) == 0:
        return R_t
    
    # Sort the node ids for consistent indexing
    node_list = sorted(G.nodes())

    # Neutral element
    X_0 = {node: 0 for node in node_list}

    # Initialize a queue with the neutral element
    Q = Queue()
    Q.put(X_0)

    # Set of visited nodes
    visited = set()
    visited.add(tuple(X_0[node] for node in node_list))

    # Sort the paths in increasing order by the length of the path to hasten the termination of the algorithm
    paths: list[list[str]] = sorted(P_t, key = lambda path: len(path))
    while not Q.empty():
        # Pop out the first element
        X = Q.get()
        found_path = None   
        for path in paths:
            if all(X[n] != -1 for n in path):
               # This path is operational
               found_path = path
               break
        
        # No path was found so move on to the next event in the queue
        if found_path is None:
            continue
       
        # A path was found
        else:
            # Mark the nodes of the found path as used
            X1 = X.copy()
            for node in found_path:
                X1[node] = 1
            
            # Increment the reliability with the probability of this event
            R_t += probability_of_event(G, X1)
           
            # Determine the complement events of X1
            for i, node in enumerate(found_path):
                complement = X1.copy()
                
                #for j in range(i+1, len(found_path)):
                #    complement[found_path[j]] = 0

                # Make all the nodes on the path AFTER 'node' to be unused again
                # The nodes BEFORE 'node' are still marked as used
                for n in found_path[i+1:]:
                    complement[n] = 0

                # Disrupt the node at index i
                complement[node] = -1

                # If this complement event has not yet been seen:
                # 1. Add it to the set of visited events
                # 2. Add it to the queue for further explaining
                t = tuple(complement[n] for n in node_list)
                if t not in visited:
                    visited.add(t)
                    Q.put(complement)
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

def brute_force_reliability(G: nx.Graph, feasiblePaths: list[list[str]]) -> float:
    if len(feasiblePaths) == 0:
        return 0.0
        
    states = itertools.product([0, 1], repeat=len(G.nodes))
    R = 0.0
    node_list = sorted(G.nodes())
    for state in states:
        disruptedNodes = [node for i, node in enumerate(node_list) if state[i] == 0]

        shortest_path = None
        for path in feasiblePaths:
            if all([node not in disruptedNodes for node in path]):
                shortest_path = path
                break
        if shortest_path is not None:
            state = [(node, state[i]) for i, node in enumerate(node_list)]
            R += probability_of_state(G, state)
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
            nodes = random.randint(15, 22)
            edges = random.randint(nodes-1, min(2*nodes, nodes*(nodes-1)//2))
            
            G = generate_random_graph_with_positions(nodes, edges, (-5, 5))
            
            # Ensure reasonable reliabilities and use consistent precision
            for node in G.nodes():
                G.nodes[node]['reliability'] = round(random.uniform(0.9, 0.99), 2)

            # Try multiple terminal pairs
            terminal_pairs_to_try = list(itertools.combinations(G.nodes(), 2))
            random.shuffle(terminal_pairs_to_try)
            
            dotson_result = None
            brute_result = None
            
            for term1, term2 in terminal_pairs_to_try[:3]:
                try:
                    paths = list(nx.all_simple_paths(G, term1, term2, cutoff=min(6, nodes)))
                    if len(paths) > 0 and len(paths) < 50:  # Limit path complexity         
                        dotson_result = terminal_pair_reliability(G, paths)
                        brute_result = brute_force_reliability(G, paths)
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
            
            accuracy = 1.0 if relative_error < 1e-4 else 0.0
            
            # Timing comparison
            
            dotson_time = timeit.timeit(
                lambda: terminal_pair_reliability(G, paths), 
                number=3
            ) / 3  # Average over 3 runs
            
            brute_time = timeit.timeit(
                lambda: brute_force_reliability(G, paths), 
                number=3  
            ) / 3
            
            speedup = brute_time / max(dotson_time, 1e-12)
            
            speedups.append(speedup)
            accuracies.append(accuracy)
            errors.append(relative_error)
            
            print(f"Trial {trial+1}: RelError = {relative_error:.8f}, "
                  f"Dotson = {dotson_result:.6f}, Brute = {brute_result:.6f}, "
                  f"Speedup = {speedup:.2f}x")
                  
            if relative_error > 1e-6:
                print(f"  DISCREPANCY: Trying simple debug...")
                debug_simple_case(G, paths, dotson_result, brute_result)
                
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

# Run the improved test
if __name__ == '__main__':
    import timeit
    speedup, accuracy, error = compute_speedup_and_accuracy_improved(num_trials=100)