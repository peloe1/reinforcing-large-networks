from result_handling import read_combined_portfolios
import matplotlib.pyplot as plt



def plot_pareto_frontier(costs: list[float], performances: list[float], random_costs: list[float] = None, random_performances: list[float] = None, sub_network=None, normalize=False):
    #plt.scatter(x=costs, y=performances, c='black', s=3, marker='o')
    
    
    if normalize:

        maximum = max(performances)
        minimum = min(performances)
        assert(maximum > minimum)
        normalized_performances = []
        for p in performances:
            normalized_performances.append((p - minimum) / (maximum - minimum))

        plt.scatter(x=costs, y=normalized_performances, c='black', s=3, marker='o')

        if sub_network is None:
            plt.title('Combined Portfolios')
        else:
            assert(type(sub_network) == str)
            plt.title('Cost-Efficient Portfolios for the Subnetwork ' + sub_network)

        plt.xlabel('Cost')
        plt.ylabel('Normalized Improvement to Performance')
        plt.yscale('function', functions=(lambda x: x / 1000, lambda x: x * 1000))
        plt.xticks([0, 5, 10, 15, 20, 25, 30, 35, 40, 45])
        plt.yticks([i / 10.0 for i in range(11)])
        plt.show()
    
    else:
        # Cost-Efficient Portfolios

        fig, ax = plt.subplots()

        # Plot cost-efficient portfolios (red dots)
        scatter1 = ax.scatter(x=costs, y=performances, c='red', s=5, marker='o', label='Cost-Efficient Portfolios')

        # The random portfolios (black dots)
        if random_costs is not None and random_performances is not None:
            scatter2 = ax.scatter(x=random_costs, y=random_performances, c='black', s=1, marker='o', label='Random Portfolios')
            ax.legend()
        else:
            # If no random portfolios, still show legend for cost-efficient ones
            ax.legend()

        if sub_network is None:
            plt.title('Combined Portfolios')
        else:
            assert(type(sub_network) == str)
            plt.title('Cost-Efficient Portfolios for the Subnetwork ' + sub_network)

        plt.xlabel('Cost')
        plt.ylabel('Expected Enabled Traffic Volume')
        #plt.yscale('function', functions=(lambda x: x / 1000, lambda x: x * 1000))
        plt.xticks([0, 5, 10, 15, 20, 25, 30, 35, 40, 45])
        plt.yticks([14400 + 100 * i for i in range(10)])
        plt.show()

    return










if __name__ == "__main__":
    Q_star, combined_performances, combined_costs, _, _ = read_combined_portfolios("model/results/whole_network_ce_portfolios.json")
    random_portfolios, dict_random_performances, dict_random_costs, _, _ = read_combined_portfolios("model/random_results/combined_portfolios.json")

    #normalize = False

    #if normalize:
    #    costs = []
    #    performances = []
    #    dim = None
    #    for Q in Q_star:
    #        dim = len(Q)
    #        break
    #    
    #    if dim is not None:
    #        base_performance = combined_performances[tuple([0 for _ in range(dim)])]

    #        for Q in Q_star:
    #            costs.append(combined_costs[Q])
    #            performances.append(combined_performances[Q] / base_performance - 1)

    #        plot_pareto_frontier(costs, performances)
    #
    #else:

    costs = []
    performances = []
    dim = None
    for Q in Q_star:
        dim = len(Q)
        break
    
    if dim is not None:
        for Q in Q_star:
            costs.append(combined_costs[Q])
            performances.append(combined_performances[Q])

        seen: set[tuple[tuple[int, ...], int]] = set()

        random_costs = []
        random_performances = []
        for portfolio in random_portfolios:
            if (tuple(dict_random_costs[portfolio]), dict_random_performances[portfolio]) not in seen:
                if not any(abs(perf - dict_random_performances[portfolio]) < 0.1 for cost_vec, perf in seen if cost_vec == dict_random_costs[portfolio]):
                    random_costs.append(dict_random_costs[portfolio])
                    random_performances.append(dict_random_performances[portfolio])
                    seen.add((tuple(dict_random_costs[portfolio]), dict_random_performances[portfolio]))
            

        plot_pareto_frontier(costs, performances, random_costs, random_performances)