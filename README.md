# Reinforcing large transportation networks

This is a Python project related to my MSc. thesis, which can be examined [here](https://github.com/peloe1/msc-thesis).

## Installation

A Python 3.10 or newer is required. The required libraries and packages can be installed via the *requirements.txt* file. 

## Usage

```python
filename = 'data/network/sij.json'
num_nodes = 40

reliabilities = {i: 0.99 for i in range(num_nodes)}
terminal_nodes = [2, 11, 32] # East, West, South
for node in terminal_nodes:
    reliabilities[node] = 1.0

G = construct_graph(filename, reliabilities)


for node in terminal_nodes:
    G.nodes[node]['reliability'] = 1

terminal_node_pairs = terminal_pairs(terminal_nodes)
traffic_volumes = {t: 100.0 for t in terminal_node_pairs}

paths = feasible_paths(G, terminal_node_pairs)
paths = {t: p for t, p in paths.items() if len(p) > 0}

node_reinforcements = [(i, 0.995) for i in range(30)]
costs = [random.choice([1.0,2.0]) for _ in range(30)]
budget = 12.0

start = time.time()
Q, portfolio_costs = costEfficientPortfolios(G, terminal_node_pairs, paths, traffic_volumes, extremePoints, node_reinforcements, costs, budget)
end = time.time()
print(f"Time to compute cost-efficient portfolios: {(end - start):.2f}")
print(f"Number of cost-efficient portfolios: {len(Q)}")
```