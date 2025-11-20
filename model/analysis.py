from result_handling import read_combined_portfolios
import matplotlib.pyplot as plt



def plot_pareto_frontier(costs: list[float], performances: list[float], sub_network=None):
    plt.scatter(x=costs, y=performances, c='black', s=3, marker='o')
    if sub_network is None:
        plt.title('Cost-Efficient Portfolios for the Whole Network')
    else:
        assert(type(sub_network) == str)
        plt.title('Cost-Efficient Portfolios for the Subnetwork ' + sub_network)
    
    plt.xlabel('Cost')
    plt.ylabel('Normalized xPerformance')
    plt.yscale('function', functions=(lambda x: x / 1000, lambda x: x * 1000))
    plt.show()

    return










if __name__ == "__main__":
    Q_star, combined_performances, combined_costs, _, _ = read_combined_portfolios("model/results/whole_network_ce_portfolios.json")

    costs = []
    performances = []

    dim = None
    for Q in Q_star:
        dim = len(Q)
        break
    
    if dim is not None:
        base_performance = combined_performances[tuple([0 for _ in range(dim)])]

        for Q in Q_star:
            costs.append(combined_costs[Q])
            performances.append(combined_performances[Q] / base_performance - 1)

        plot_pareto_frontier(costs, performances)