from queue import Queue
import networkx as nx
import itertools
import random
import time
import numpy as np
import matplotlib.pyplot as plt

from path import feasible_paths, terminal_pairs
from graph import generate_random_graph_with_positions
from plotting import plot_network

# Correct implementation
def probability_of_event(G: nx.Graph, E: np.ndarray) -> float:
    """
        Parameters:
        -----------
        G: nx.Graph
            Graph object
        event: np.ndarray
            Event as a 1D NumPy array of integers 
            +1 represents that the node is used
            -1 represents that the node is disrupted
            0 represents that the node is not used

        Returns:
        --------
        prob: float
            Probability of the state
    """

    prob = 1.0
    for node, e in enumerate(E):
        if e == 1:
            prob *= G.nodes[node]['reliability']
        elif e == -1:
            prob *= (1 - G.nodes[node]['reliability'])
    return prob


# Modified Dotson algorithm
def terminal_pair_reliability(G: nx.Graph, feasiblePaths: list[np.ndarray]) -> float:
    """
        Parameters:
        -----------
        G: nx.Graph
            Graph object
        feasiblePaths: list[np.ndarray]
            List of feasible paths between some particular terminal pair represented with 1D NumPy arrays
        disruptedNode: int
            Node that is fixed to be disrupted. Used when computing the risk achievement worths

        Returns:
        --------
        R: float
            Terminal pair reliability
    """
    if len(feasiblePaths) == 0:
        return 0.0
    
    R = 0
    E_0 = np.zeros(G.number_of_nodes(), dtype=int)

    Q = Queue()
    Q.put(E_0)

    visited = set()
    visited.add(tuple(E_0))

    paths: list[np.ndarray] = sorted(feasiblePaths, key = lambda path: len(path))
    while not Q.empty():
        E = Q.get()

        for i, path in enumerate(paths):
            if np.all(E[path] != -1):
                shortest_path = paths.pop(i)
                E1 = E.copy()
                E1[shortest_path] = 1
                
                R += probability_of_event(G, E1)

                # Add the complement events to the queue
                for node in shortest_path:
                    complement = E1.copy()
                    complement[node] = -1
                    complement[complement == 1] = 0
                    t = tuple(complement)
                    if t not in visited:
                        visited.add(t)
                        Q.put(complement)

                break
    
    if R > 1:
        print("Reliability = {}".format(R))
        return -1

    return R

# Redundant helper function
def probability_of_state(G: nx.Graph, state: np.ndarray) -> float:
    """
        Parameters:
        -----------
        G: nx.Graph
            Graph object
        state: list[int]
            List of events

        Returns:
        --------
        prob: float
            Probability of the state
    """

    prob = 1
    for node, e in enumerate(state):
        prob *= (1 - e) * G.nodes[node]['reliability'] + e * (1 - G.nodes[node]['reliability'])
    return prob

# Just for double checking the functionality of the modified Dotson algorithm
def brute_force_reliability(G: nx.Graph, feasiblePaths: list[np.ndarray]) -> float:
    """
        Parameters:
        -----------
        G: nx.Graph
            Graph object
        feasiblePaths: list[np.ndarray]
            List of feasible paths between some particular terminal node pair

        Returns:
        --------
        R: float
            Terminal node pair reliability
    """

    states = itertools.product([0, 1], repeat=len(G.nodes))
    R = 0
    for state in states:
        disruptedNodes = [i for i in range(len(state)) if state[i] == 1]

        shortest_path = None
        for path in feasiblePaths:
            if all([node not in disruptedNodes for node in path]):
                shortest_path = path
                break
        if shortest_path is not None:
            R += probability_of_state(G, np.array(state))
    return R

def compute_speedup_and_accuracy():
    speedup = 0.0
    accuracy = 0.0
    error = 0.0
    nodes = random.randint(10, 20)
    #nodes = 7
    edge_numbers = [int(1.2 * nodes), int(1.6 * nodes), int(2.0 * nodes)]
    #edge_numbers = [14]
    for edges in edge_numbers:
        G = generate_random_graph_with_positions(nodes, edges, (-5, 5))
        terminal_nodes: list[int] = [0, 3]
        terminal_node_pairs = terminal_pairs(terminal_nodes)
        
        paths: list[np.ndarray] = feasible_paths(G, terminal_node_pairs)[(terminal_nodes[0], terminal_nodes[1])]

        start = time.time()
        dotson = terminal_pair_reliability(G, paths)
        end = time.time()
        elapsed_dotson = end - start

        start = time.time()
        brute_force = brute_force_reliability(G, paths)
        end = time.time()
        elapsed_brute_force = end - start

        speedup += elapsed_brute_force / (elapsed_dotson + 1e-10)
        
        error += abs(dotson - brute_force)

        if abs(dotson - brute_force) < 0.0001:
            accuracy += 1.0
        else:
            #print("Network G node reliabilities: ")
            #for node in G.nodes:
            #    print("Node {} reliability: {}".format(node, G.nodes[node]['reliability']))
            #print("Network G feasible paths: ")
            #for path in paths:
            #    print(path)

            
            print("-----------------------")
            print("Dotson: {}".format(dotson))
            print("Brute Force: {}".format(brute_force))
            print("-----------------------")

    return speedup / len(edge_numbers), accuracy / len(edge_numbers), error / len(edge_numbers)


if __name__ == '__main__':

    if False:
        k = 50

        terminal_nodes = [0,3]

        average_runtimes_12 = []
        average_runtimes_16 = []
        average_runtimes_20 = []
        for nodes in range(5, 30):
            runtimes_12 = []
            runtimes_16 = []
            runtimes_20 = []
            
            for i in range(k):
                G12 = generate_random_graph_with_positions(nodes, int(1.2 * nodes), (-5, 5))
                G16 = generate_random_graph_with_positions(nodes, int(1.6 * nodes), (-5, 5))
                G20 = generate_random_graph_with_positions(nodes, int(2.0 * nodes), (-5, 5))

                paths12 = feasible_paths(G12, terminal_nodes)[(terminal_nodes[0], terminal_nodes[1])]
                paths16 = feasible_paths(G16, terminal_nodes)[(terminal_nodes[0], terminal_nodes[1])]
                paths20 = feasible_paths(G20, terminal_nodes)[(terminal_nodes[0], terminal_nodes[1])]

                start = time.time()
                terminal_pair_reliability(G12, paths12)
                end = time.time()
                runtimes_12.append(end - start)

                start = time.time()
                terminal_pair_reliability(G16, paths16)
                end = time.time()
                runtimes_16.append(end - start)

                start = time.time()
                terminal_pair_reliability(G20, paths20)
                end = time.time()
                runtimes_20.append(end - start)

            average_runtimes_12.append(np.mean(runtimes_12))
            average_runtimes_16.append(np.mean(runtimes_16))
            average_runtimes_20.append(np.mean(runtimes_20))

        plt.plot(range(5, 30), average_runtimes_12, label="12")
        plt.plot(range(5, 30), average_runtimes_16, label="16")
        plt.plot(range(5, 30), average_runtimes_20, label="20")
        plt.legend()
        plt.show()




    if True:
        k = 1000

        speedup = 0.0
        accuracy = 0.0
        error = 0.0

        for i in range(k):
            print("Iteration: {}".format(i+1))
            out1, out2, out3 = compute_speedup_and_accuracy()
            speedup += out1
            accuracy += out2
            error += out3
        
        print("Accuracy: {} %".format(int(100 * accuracy / k)))
        print("Average speedup: {}x".format(int(speedup / k)))
        print("Average absolute error: {}".format(error / k))

    if False:
        G = nx.Graph()

        # Add nodes with reliability attributes
        G.add_node(0, reliability=0.65)
        G.add_node(1, reliability=0.95)
        G.add_node(2, reliability=0.345)
        G.add_node(3, reliability=0.1290)
        G.add_node(4, reliability=0.80)
        G.add_node(5, reliability=0.99)
        G.add_node(6, reliability=0.50)

        # Add edges between nodes
        G.add_edge(0, 1)
        G.add_edge(1, 2)
        G.add_edge(2, 3)
        G.add_edge(3, 4)
        G.add_edge(1, 3)
        G.add_edge(3, 5)
        G.add_edge(5, 6)
        G.add_edge(6, 4)
        G.add_edge(4, 0)


        terminal_nodes: list[int] = [0, 3]
            
        #paths: list[np.ndarray] = feasible_paths(G, terminal_nodes)[(terminal_nodes[0], terminal_nodes[1])]
        simple_paths = nx.all_simple_paths(G, source=terminal_nodes[0], target=terminal_nodes[1])
        paths = []
        for path in simple_paths:
            paths.append(np.array(path, dtype=int))

        dotson = terminal_pair_reliability(G, paths)
        brute_force = brute_force_reliability(G, paths)

        print("-----------------------")
        print("Dotson: {}".format(dotson))
        print("Brute Force: {}".format(brute_force))
        print("-----------------------")



    if False:
        for i in range(100):
            nodes = random.randint(5,20)
            edge_numbers = [int(1.2 * nodes), int(1.6 * nodes), int(2.0 * nodes)]
            edges = random.choice(edge_numbers)
            
            G = generate_random_graph_with_positions(nodes, edges, (-5, 5))
            terminal_nodes: list[int] = [0, 3]
            
            paths: list[np.ndarray] = feasible_paths(G, terminal_nodes)[(terminal_nodes[0], terminal_nodes[1])]

            dotson = terminal_pair_reliability(G, paths)
            brute_force = brute_force_reliability(G, paths)

            if abs(dotson - brute_force) > 0.001:
                print("-----------------------")
                print("Iteration: {}".format(i+1))
                print("Dotson: {}".format(dotson))
                print("Brute Force: {}".format(brute_force))
                print("-----------------------")
    
    if False:
        E = [-1,1,0,0,1]
        complements = complement_events(np.array(E))
        for event in complements:
            print(event)
    if False:
        G = nx.Graph()
        G.add_node(0, label="0", reliability=0.9, pos=(0,0))
        G.add_node(1, label="1", reliability=0.9, pos=(-1,1))
        G.add_node(2, label="2", reliability=0.9, pos=(0,2))
        G.add_node(3, label="3", reliability=0.9, pos=(3,1))
        G.add_node(4, label="4", reliability=0.9, pos=(1,1.75))
        G.add_node(5, label="5", reliability=0.9, pos=(2,0))
        G.add_node(6, label="6", reliability=0.9, pos=(3,2))

        G.add_edge(0,1)
        G.add_edge(0,2)
        G.add_edge(0,5)
        G.add_edge(0,6)

        G.add_edge(1,2)
        G.add_edge(1,4)
        G.add_edge(1,3)

        G.add_edge(2,4)
        G.add_edge(2,5)
        G.add_edge(2,6)
        
        G.add_edge(6,4)
        G.add_edge(6,3)
        G.add_edge(6,5)

        G.add_edge(5,3)

        paths = [np.array([0,5,3]), np.array([0,1,2,6,3])]

        dotson = terminal_pair_reliability(G, paths)
        brute_force = brute_force_reliability(G, paths)

        print("-----------------------")
        print("Dotson: {}".format(dotson))
        print("Brute Force: {}".format(brute_force))
        print("-----------------------")


        plot_network(G)

    pass