import networkx as nx
import random
import json
import numpy as np
#import plotting
import os
import plotly
from plotly.graph_objects import Figure

def generate_random_graph_with_positions(num_nodes, num_edges, pos_range=(0, 1)):
    """
        Parameters:
        -----------
        num_nodes: int
            Number of nodes in the graph
        num_edges: int
            Number of edges in the graph
        pos_range: tuple[float, float]
            Range of positions for the nodes

        Returns:
        --------
        G: nx.Graph
            A randomly generated Graph object
    
    """

    # Create an empty graph
    G = nx.Graph()
    
    # Add the specified number of nodes with random positions
    for i in range(num_nodes):
        # Generate random positions within the given range
        pos_x = random.uniform(pos_range[0], pos_range[1])
        pos_y = random.uniform(pos_range[0], pos_range[1])
        G.add_node(i, pos=(pos_x, pos_y), label=i, reliability=random.uniform(0.7, 0.95))  # Fixed reliability range
    
    # First ensure the graph is connected by creating a spanning tree
    nodes = list(G.nodes())
    random.shuffle(nodes)
    
    # Add edges to form a spanning tree
    for i in range(1, len(nodes)):
        G.add_edge(nodes[i-1], nodes[i], weight=1, color='gray')
    
    # Calculate remaining edges needed
    edges_added = len(nodes) - 1
    remaining_edges = num_edges - edges_added
    
    if remaining_edges > 0:
        # Get all possible edges that don't exist yet
        possible_edges = []
        for i in range(num_nodes):
            for j in range(i+1, num_nodes):
                if not G.has_edge(i, j):
                    possible_edges.append((i, j))
        
        # Add random edges from the remaining possible edges
        if possible_edges:
            selected_edges = random.sample(possible_edges, min(remaining_edges, len(possible_edges)))
            G.add_edges_from(selected_edges, weight=1, color='gray')
    
    return G


def read_from_json(filename: str) -> nx.Graph:
    """
        Parameters:
        -----------
        filename: str
            Path to the JSON file

        Returns:
        --------
        G: nx.Graph
            Graph object
    """

    with open(filename) as f:
        graph_data = json.load(f)
    
    G = nx.Graph()
            
    for node in graph_data['nodes']:
        G.add_node(node['id'], pos=(node['x'], node['y']), label=node['id'], type=node['type'], reliability=1.0)

    for edge in graph_data['edges']:
        G.add_edge(edge['n0'], edge['n1'])

    return G

# Removes nodes of type "secondary" while preserving the connections
# it connects each neighbour of a secondary node to the other neighbours of that secondary node
def remove_secondary_nodes(G):
    """
        Parameters:
        -----------
        G: nx.Graph
            Graph object

        Returns:
        --------
        G: nx.Graph
            The modified graph without any secondary nodes
    """

    secondary_nodes = [node for node, data in G.nodes(data=True) if data.get('type') == 'secondary']

    for node in secondary_nodes:
        neighbors = list(G.neighbors(node))
        
        for i in range(len(neighbors)):
            for j in range(i+1, len(neighbors)):
                G.add_edge(neighbors[i], neighbors[j])

        G.remove_node(node)

    return G

def add_reliabilities(G: nx.Graph, reliabilities: dict[str, float]) -> nx.Graph:
    """
        Parameters:
        -----------
        G: nx.Graph
            Graph object
        probabilities: dict[int, float]
            Dictionary mapping node IDs to reliabilities

        Returns:
        --------
        G: nx.Graph
            The modified graph

    """

    for node, reliability in reliabilities.items():
        G.nodes[node]['reliability'] = reliability
    return G

def construct_graph(filename: str) -> tuple[nx.Graph, nx.Graph]:
    """
        Parameters:
        -----------
        filename: str
            Path to the JSON file
        terminal_nodes: list[int]
            List of terminal nodes
        travel_data: dict[tuple[int, int], float]
            Dictionary mapping terminal node pairs to travel frequencies
        probabilities: dict[int, float]
            Dictionary mapping node IDs to reliabilities

        Returns:
        --------
        G: nx.Graph
            Graph object
    
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    filename = os.path.join(project_root, filename)

    G_original = read_from_json(filename)
    G = remove_secondary_nodes(G_original.copy())
    
    return G, G_original

def merge_graphs_from_json(file_list: list[str]) -> nx.Graph:
    """
        Parameters:
        -----------
        file_list: list[str]
            List of JSON file paths to merge (networks must share common nodes)

        Returns:
        --------
        G_merged: nx.Graph
            Unified graph with merged nodes and edges
    """

    G_merged = nx.Graph()

    for filename in file_list:
        G_temp = read_from_json(filename)

        # Merge nodes (preserving attributes)
        for node, attrs in G_temp.nodes(data=True):
            if not G_merged.has_node(node):
                G_merged.add_node(node, **attrs)
            else:
                # Update missing attributes only, do not overwrite existing
                for k, v in attrs.items():
                    if k not in G_merged.nodes[node]:
                        G_merged.nodes[node][k] = v

        # Merge edges (preserving weight + other attributes if present)
        for u, v, attrs in G_temp.edges(data=True):
            if not G_merged.has_edge(u, v):
                G_merged.add_edge(u, v, **attrs)

    return G_merged



def plot_network(G: nx.Graph, name: str, manual_clusters=None):
    """
    Plot network with manually specified cluster circles
    
    Parameters:
    -----------
    G : nx.Graph
        Graph object
    name : str
        Name for the output file
    manual_clusters : dict
        Dictionary with cluster specifications
        Example: {
            'Yard A': {'center': (560000, 6990000), 'radius': 5000},
            'Terminal B': {'center': (580000, 6980000), 'radius': 3000}
        }
    """
    nodes = G.nodes(data=True)
    edges = G.edges()

    fig = Figure()

    # Plot edges (railway tracks)
    xs = []
    ys = []

    for e in edges:
        if e[0] in G.nodes and e[1] in G.nodes:
            xs.append(nodes[e[0]]["pos"][0])
            xs.append(nodes[e[1]]["pos"][0])
            xs.append(None)

            ys.append(nodes[e[0]]["pos"][1])
            ys.append(nodes[e[1]]["pos"][1])
            ys.append(None)

    fig.add_scatter(name = "Railway track",
                    x = xs, 
                    y = ys,
                    line_color = "rgba(0, 0, 255, 0.3)")

    # Plot nodes (railway switches)
    primary_nodes_x = []
    primary_nodes_y = []
    primary_nodes_labels = []
    
    for node, data in G.nodes(data=True):
        if data.get('type') == 'primary':
            primary_nodes_x.append(data.get("pos")[0])
            primary_nodes_y.append(data.get("pos")[1])
            primary_nodes_labels.append(str(node))

    if primary_nodes_x:
        fig.add_scatter(name="Railway switches", 
                        x=primary_nodes_x,
                        y=primary_nodes_y,
                        text=primary_nodes_labels,
                        mode="markers")

    # Add manual cluster circles and labels
    if manual_clusters:
        for cluster_name, cluster_data in manual_clusters.items():
            center_x, center_y = cluster_data['center']
            radius = cluster_data['radius']
            location_x, location_y = cluster_data['location']
            color = "black"
            
            # Generate circle points
            theta = np.linspace(0, 2*np.pi, 100)
            circle_x = center_x + radius * np.cos(theta)
            circle_y = center_y + radius * np.sin(theta)
            
            # Add circle to plot
            fig.add_scatter(x=circle_x, y=circle_y, 
                           mode='lines', 
                           line=dict(color=color, width=1, dash='dot'),
                           name=f'{cluster_name} Boundary',
                           showlegend=False)
            
            # Add cluster label
            fig.add_annotation(x=location_x, y=location_y,
                              text=cluster_name,
                              showarrow=False,
                              font=dict(size=10, color=color),#, weight='bold'),
                              bgcolor='white',
                              bordercolor=color,
                              borderwidth=1,
                              borderpad=1)

# Add specific labeled black squares - CLEAN VERSION
    special_points = {
        "KRM V0001|V0002": {"label": "South", "symbol": "square"},
        "LUI V0511": {"label": "East", "symbol": "square"}, 
        "OHM V0002": {"label": "North", "symbol": "square"}
    }
    
    special_x = []
    special_y = []
    special_hover_text = []
    
    for point_id, point_info in special_points.items():
        # Check if the point exists in the graph
        if point_id in G.nodes:
            node_data = G.nodes[point_id]
            special_x.append(node_data.get("pos")[0])
            special_y.append(node_data.get("pos")[1])
            special_hover_text.append(f"{point_id} ({point_info['label']})")
        else:
            print(f"Warning: Special point '{point_id}' not found in graph")
    
    # Add black squares for special points
    if special_x:
        fig.add_scatter(name="Virtual nodes",
                        x=special_x,
                        y=special_y,
                        text=special_hover_text,
                        hoverinfo="text",
                        mode="markers",
                        marker=dict(
                            size=10,
                            color='black', 
                            symbol='square',
                            line=dict(width=1, color='black')
                        ),
                        showlegend=True)
        
        # Add directional labels as clean text annotations (no boxes)
        for i, (x, y, point_id) in enumerate(zip(special_x, special_y, special_points.keys())):
            if point_id in G.nodes:
                label = special_points[point_id]["label"]
                
                # Adjust label position based on direction
                if label == "North":
                    label_y_offset = 2400
                    label_x_offset = 0
                elif label == "South":
                    label_y_offset = -2200
                    label_x_offset = 0
                elif label == "East":
                    label_y_offset = 0
                    label_x_offset = 3800
                else:
                    label_y_offset = 500
                    label_x_offset = 0
                
                fig.add_annotation(
                    x=x + label_x_offset, 
                    y=y + label_y_offset,
                    text=label,
                    showarrow=False,
                    font=dict(size=10, color='black', weight='bold'),
                    # Removed all box styling parameters
                    bgcolor='rgba(0,0,0,0)',  # Transparent background
                    bordercolor='rgba(0,0,0,0)',  # Transparent border
                    borderwidth=0,  # No border
                    borderpad=0,  # No padding
                    opacity=1  # Fully opaque text
                )

    fig.update_layout(
        xaxis_title="<b>X Coord. (ETRS-TM35FIN)</b>",
        yaxis_title="<b>Y Coord. (ETRS-TM35FIN)</b>",
        legend_orientation="h",
        height=700,
        yaxis_scaleanchor="x", 
        yaxis_scaleratio=1,
        showlegend=True
    )
    
    fig.write_image(r"C:\Users\Elias\Desktop\Dippa\msc-thesis\Figures\4-Case Study\{}.pdf".format(name))
    fig.show()



def plot_siilinjarvi(G: nx.Graph, name: str):
    """
    Plot Siilinjärvi station with focused view - virtual nodes moved closer
    """
    # Create a copy of the graph to modify
    G_focused = G.copy()
    
    # Define new positions for virtual nodes to bring them closer to the main station
    # Adjust these coordinates based on where you want them relative to the main station
    VIRTUAL_NODE_POSITIONS = {
        "SKM V0271": (537493.8, 6997555),  # Moved closer to main station
        "APT V0001": (532800, 6995879),  # Moved closer to main station
        "TOI V0002": (534343, 6992939),  # Moved closer to main station
    }
    
    # Define important switches to highlight with manual text positioning
    IMPORTANT_SWITCHES = {
        "SIJ V0611": {"label": "SIJ V0611", "color": "black", "size": 6, "text_offset": (400, 0)},
        "SIJ V0642": {"label": "SIJ V0642", "color": "black", "size": 6, "text_offset": (300, -150)},
        "SIJ V0632": {"label": "SIJ V0632", "color": "black", "size": 6, "text_offset": (350, 100)},
    }
    
    # Reposition virtual nodes in the graph copy
    for node_id, new_pos in VIRTUAL_NODE_POSITIONS.items():
        if node_id in G_focused.nodes:
            G_focused.nodes[node_id]['pos'] = new_pos
            print(f"Repositioned {node_id} to {new_pos}")
    
    # Get the repositioned coordinates for filtering
    skm_x, skm_y = VIRTUAL_NODE_POSITIONS["SKM V0271"]
    apt_x, apt_y = VIRTUAL_NODE_POSITIONS["APT V0001"] 
    toi_x, toi_y = VIRTUAL_NODE_POSITIONS["TOI V0002"]
    
    # Filter nodes: keep only those that are NOT in the redundant branch areas
    nodes_to_keep = set()
    
    for node_id, data in G_focused.nodes(data=True):
        pos = data.get('pos')
        if pos:
            x, y = pos
            
            # Rule 1: For APT branch - filter out nodes that are too far west and north of APT
            # (remove nodes that are further west and higher up than APT)
            apt_condition = not (x < apt_x and y > apt_y)
            
            # Rule 2: For SKM branch - filter out nodes that are too far east and north of SKM  
            # (remove nodes that are further east and higher up than SKM)
            skm_condition = not (x > skm_x and y > skm_y)
            
            # Rule 3: For TOI branch - filter out nodes that are too far south and west of TOI
            # (remove nodes that are further south and west than TOI)
            toi_condition = not (x > toi_x and y < toi_y)
            
            # Keep node if it passes all three filtering rules
            if apt_condition and skm_condition and toi_condition:
                nodes_to_keep.add(node_id)
            else:
                print(f"Filtered out node {node_id} at ({x:.1f}, {y:.1f})")
    
    # Also always keep the virtual nodes themselves
    for node_id in VIRTUAL_NODE_POSITIONS.keys():
        if node_id in G_focused.nodes:
            nodes_to_keep.add(node_id)
    
    # Create subgraph with only the filtered nodes
    G_focused = G_focused.subgraph(nodes_to_keep)
    print(f"Filtered graph: {len(nodes_to_keep)} nodes kept")
    
    # Use the modified graph for plotting
    nodes = G_focused.nodes(data=True)
    edges = G_focused.edges()

    fig = Figure()

    # Plot edges (railway tracks)
    xs = []
    ys = []

    for e in edges:
        if e[0] in G_focused.nodes and e[1] in G_focused.nodes:
            xs.append(nodes[e[0]]["pos"][0])
            xs.append(nodes[e[1]]["pos"][0])
            xs.append(None)

            ys.append(nodes[e[0]]["pos"][1])
            ys.append(nodes[e[1]]["pos"][1])
            ys.append(None)

    fig.add_scatter(name = "Railway track",
                    x = xs, 
                    y = ys,
                    line_color = "rgba(0, 0, 255, 0.3)")

    # Plot nodes (railway switches) - separate regular switches from important ones
    regular_primary_x = []
    regular_primary_y = []
    regular_primary_labels = []
    
    important_switches_x = []
    important_switches_y = []
    important_switches_labels = []
    
    for node, data in G_focused.nodes(data=True):
        if data.get('type') == 'primary':
            x = data.get("pos")[0]
            y = data.get("pos")[1]
            
            if node in IMPORTANT_SWITCHES:
                important_switches_x.append(x)
                important_switches_y.append(y)
                important_switches_labels.append(node)
            else:
                regular_primary_x.append(x)
                regular_primary_y.append(y)
                regular_primary_labels.append(str(node))

    # Plot regular primary nodes
    if regular_primary_x:
        fig.add_scatter(name="Railway switches", 
                        x=regular_primary_x,
                        y=regular_primary_y,
                        text=regular_primary_labels,
                        mode="markers",
                        marker=dict(size=6, color='red'))

    # Plot important switches as markers only (no text)
    if important_switches_x:
        fig.add_scatter(name="Terminal nodes", 
                        x=important_switches_x,
                        y=important_switches_y,
                        text=important_switches_labels,
                        hoverinfo="text",
                        mode="markers",
                        marker=dict(
                            size=6,
                            color='black',
                            symbol='circle',
                            line=dict(width=1, color='black')
                        ),
                        showlegend=True)

    # Add manual text annotations for important switches
    for node_id, switch_info in IMPORTANT_SWITCHES.items():
        if node_id in G_focused.nodes:
            node_data = G_focused.nodes[node_id]
            x = node_data.get("pos")[0]
            y = node_data.get("pos")[1]
            label = switch_info["label"]
            x_offset, y_offset = switch_info["text_offset"]
            
            fig.add_annotation(
                x=x + x_offset, 
                y=y + y_offset,
                text=label,
                showarrow=False,
                font=dict(size=12, color='black'),
                bgcolor='rgba(0,0,0,0)',
                bordercolor='rgba(0,0,0,0)',
                borderwidth=0,
                borderpad=0,
                opacity=1
            )

    # Add specific labeled black squares - with new positions
    special_points = {
        "SKM V0271": {"label": "SKM", "symbol": "square"},
        "APT V0001": {"label": "APT", "symbol": "square"}, 
        "TOI V0002": {"label": "TOI", "symbol": "square"}
    }
    
    special_x = []
    special_y = []
    special_hover_text = []
    
    for point_id, point_info in special_points.items():
        # Check if the point exists in the graph
        if point_id in G_focused.nodes:
            node_data = G_focused.nodes[point_id]
            special_x.append(node_data.get("pos")[0])
            special_y.append(node_data.get("pos")[1])
            special_hover_text.append(f"{point_id} ({point_info['label']})")
        else:
            print(f"Warning: Special point '{point_id}' not found in graph")
    
    # Add black squares for special points
    if special_x:
        fig.add_scatter(name="Virtual nodes",
                        x=special_x,
                        y=special_y,
                        text=special_hover_text,
                        hoverinfo="text",
                        mode="markers",
                        marker=dict(
                            size=10,
                            color='black', 
                            symbol='square',
                            line=dict(width=1, color='black')
                        ),
                        showlegend=True)
        
        # Add labels as clean text annotations (no boxes)
        for i, (x, y, point_id) in enumerate(zip(special_x, special_y, special_points.keys())):
            if point_id in G_focused.nodes:
                label = special_points[point_id]["label"]
                
                # Adjust label position based on which node
                if label == "APT":
                    label_y_offset = 200
                    label_x_offset = 0
                elif label == "TOI":
                    label_y_offset = -200
                    label_x_offset = 0
                elif label == "SKM":
                    label_y_offset = 0
                    label_x_offset = 300
                else:
                    label_y_offset = 500
                    label_x_offset = 0
                
                fig.add_annotation(
                    x=x + label_x_offset, 
                    y=y + label_y_offset,
                    text=label,
                    showarrow=False,
                    font=dict(size=10, color='black', weight='bold'),
                    bgcolor='rgba(0,0,0,0)',
                    bordercolor='rgba(0,0,0,0)',
                    borderwidth=0,
                    borderpad=0,
                    opacity=1
                )

    fig.update_layout(
        xaxis_title="<b>X Coord. (ETRS-TM35FIN)</b>",
        yaxis_title="<b>Y Coord. (ETRS-TM35FIN)</b>",
        legend_orientation="h",
        height=700,
        yaxis_scaleanchor="x", 
        yaxis_scaleratio=1,
        showlegend=True
    )
    
    fig.write_image(r"C:\Users\Elias\Desktop\Dippa\msc-thesis\Figures\4-Case Study\{}.pdf".format(name))
    fig.show()
    
    return G_focused

if __name__ == '__main__':
    
    
    network = False
    
    if network:
        filename = 'data/network/dipan_data/@network.json'
        G, G_original = construct_graph(filename)

        clusters = { 
             'TE': {'center': (515600, 7036900), 'radius': 2500, 'location': (515600 + 1.0 * 2500, 7036900 + 1.0 * 2500)},
             'LNA': {'center': (519814, 7025500), 'radius': 2500, 'location': (519814 + 1.2 * 2500, 7025500 + 1.2 * 2500)},
             'APT': {'center': (526939, 7008000), 'radius': 2500, 'location': (526939 + 1.2 * 2500, 7008000 + 1.2 * 2500)},
             'SIJ': {'center': (535000, 6995800), 'radius': 4000, 'location': (535000, 6995800 + 1.3 * 4000)},
             'SKM': {'center': (544100, 7002700), 'radius': 2000, 'location': (544100 - 0.5 * 2000, 7002700 + 1.6 * 2000)},
             'KNH': {'center': (548320, 7002350), 'radius': 2000, 'location': (548320 + 1.4 * 2000, 7002350 + 1.8 * 2000)},
             'JKI': {'center': (566584, 6993210), 'radius': 2000, 'location': (566584 + 1.2 * 2000, 6993210 + 1.2 * 2000)},
             'TOI': {'center': (536700, 6985450), 'radius': 2000, 'location': (536700 + 2.2 * 2000, 6985450)},
             'SOR': {'center': (534770, 6980652), 'radius': 2000, 'location': (534770 - 2.2 * 2000, 6980652)},
             'KUO': {'center': (535000, 6974420), 'radius': 3500, 'location': (535000 + 1.4 * 3500, 6974420)},
        }

        plot_network(G_original, "network", manual_clusters=clusters)

    else:
        filename = 'data/network/dipan_data/sij.json'
        G, G_original = construct_graph(filename)
        plot_siilinjarvi(G_original, "siilinjarvi")

    

    #reliabilities = {node: 0.99 for node in G.nodes()}
    #G = add_reliabilities(G, reliabilities)

    #for node, data in G.nodes(data=True):
    #    print(type(node))
    #    print("Node: ", node, " of type: ", data.get("type"), " has reliability: ", data.get("reliability"))

    #plotting.plot_network(G)

    
        