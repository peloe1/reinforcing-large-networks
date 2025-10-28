# %%
#import builtins
#from builtins import *
#import jsons
#import json
from graph import construct_graph

import plotly
from plotly.graph_objects import Figure

# %%
#network_file_path = f"{__file__}/../../data/network/sij.json"
filename = 'data/network/sij.json'

network, _ = construct_graph(filename)


nodes = network.nodes(data=True)
edges = network.edges(data=True)

# %%
fig = Figure()

xs = []
ys = []

for e in edges:
    xs.append(e[0]["x"])
    xs.append(e[1]["x"])
    xs.append(None)

    ys.append(e[0]["y"])
    ys.append(e[1]["y"])
    ys.append(None)

fig = Figure()

fig.add_scatter(name = "Track Geom.",
                x = xs, 
                y = ys,
                line_color = "rgba(0, 0, 255, 0.3)")

fig.add_scatter(name = "Track Infra", 
                x = [n["x"] for n in nodes if n["type"] == "primary"],
                y = [n["y"] for n in nodes if n["type"] == "primary"],
                text = [n["id"] for n in nodes if n["type"] == "primary"],
                mode = "markers")

fig.update_layout(
                    xaxis_title = "<b>X Coord. (ETRS-TM35FIN)</b>",
                    yaxis_title = "<b>Y Coord. (ETRS-TM35FIN)</b>",
                    legend_orientation = "h",
                    height = 700,
                    yaxis_scaleanchor = "x", 
                    yaxis_scaleratio = 1,
                    )



    
# %%
