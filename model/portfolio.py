import numpy as np
import time
import itertools
from queue import Queue
import random
import matplotlib.pyplot as plt

def generate_all_portfolios(numberOfProjects: int) -> list[list[int]]:
    """
        Parameters:
            numberOfProjects (int): Number of projects, which are identified by index
        
        Returns:
            list[list[int]]: List of all possible portfolios, each portfolio is represented as a list of 
                             binary values signifying if a project is selected or not in the portfolio
    """

    # The generation of all portfolios when there are 25 projects took 30s on my PC
    # This is not yet a problem since this is run only once in the algorithm
    # But this maybe unviable later on since memory might run out
    # 2^25 = 33 554 432

    #return np.array(np.meshgrid(*([0,1] for _ in range(numberOfProjects)))).T.reshape(-1, numberOfProjects)
    #return list(map(list, itertools.product([0,1], repeat = numberOfProjects)))
    return list(map(list, itertools.product([0,1], repeat = numberOfProjects)))

# Old deprecated inefficient version
def feasible_portfolios_filter(noOfActions: int, costs: list[float], budget: float) -> tuple[list[list[int]], dict[tuple[int, ...], float]]:
    """
        Parameters:
            noOfActions (int): Number of possible reinforcement actions.
            costs (list[float]): The costs of the reinforcement actions.
            budget (float): The budget available.
            
        Returns:
            tuple[list[list[int]], dict[tuple[int, ...], float]]: The set of feasible portfolios as a list of portfolios and their costs in a dictionary.
    """

    feasible = []
    allPortfolios = np.array(generate_all_portfolios(noOfActions))
    costsArray = np.array(costs)
    budgetArray = np.full(2**noOfActions, budget)
    costsOfPortfolios = np.sum(allPortfolios * costsArray, axis=1)
    
    mask = costsOfPortfolios <= budgetArray
    feasible = allPortfolios[mask]

    feasible = feasible.tolist()

    feasibleCosts = costsOfPortfolios[mask]
    portfolioCosts = {}

    for i, q in enumerate(feasible):
        portfolioCosts[tuple(q)] = feasibleCosts[i]

    return feasible, portfolioCosts

# Old inefficient version, which used lists of integers (zeros and ones) to represent portfolios
def generate_feasible_portfolios_old(noOfActions: int, costs: list[float], budget: float) -> tuple[list[list[int]], dict[tuple[int, ...], float]]:
    """
        Parameters:
            noOfActions (int): Number of possible reinforcement actions.
            costs (list[float]): The costs of the reinforcement actions.
            budget (float): The budget available.
            
        Returns:
            tuple[list[list[int]], dict[tuple[int, ...], float]]: The set of feasible portfolios and their costs in a dictionary.
    """

    feasible = []
    costsOfPortfolios = {}

    visited = set()

    q = [0] * noOfActions
    feasible.append(q)
    costsOfPortfolios[tuple(q)] = 0
    visited.add(tuple(q))

    Q = Queue()
    Q.put(q)
    while not Q.empty():
        q = Q.get()
        for i in range(noOfActions):
            if q[i] == 0:
                q_copy = q.copy()
                q_copy[i] = 1
                
                cost = np.sum(np.array(q_copy) * np.array(costs))
                if tuple(q_copy) not in visited and cost <= budget:
                    feasible.append(q_copy)
                    costsOfPortfolios[tuple(q_copy)] = cost
                    visited.add(tuple(q_copy))
                    if cost < budget:
                        Q.put(q_copy)

    return feasible, costsOfPortfolios

def portfolio_as_bitmask(portfolio: list[int]) -> int:
    return sum((bit << i) for i, bit in enumerate(portfolio))

# A new improved version, which hashes the portfolios (binary vectors) to integers (using the trivial choice for hash function) to speed up memory accesses and computations
# q_i = 1 refers the the ith reinforcement action being selected in portfolio q, its corresponding node it is associated with is abstracted away here, but is 
# accessible in nodeReinforcements variable in cost_efficient_portfolios function in ce_portfolios.py, the indexes of that list are the indexes i here.
def generate_feasible_portfolios(noOfActions: int, costs: dict[str, list[float]], budget: list[float]) -> tuple[set[int], dict[int, list[float]]]:
    """
        Parameters:
            noOfActions (int): Number of possible reinforcement actions.
            costs (list[float]): The costs of the reinforcement actions.
            budget (float): The budget available.
            
        Returns:
            tuple[set[int], dict[int, float]]: The set of feasible portfolios as integers and their costs in a dictionary.
    """
    r = len(budget)

    actions: dict[int, str] = {i: node for i, (node, _) in enumerate(costs.items())}

    feasible: set[int] = set()
    costsOfPortfolios: dict[int, list[float]] = {}

    visited: set[int] = set()

    q = 0
    feasible.add(q)
    costsOfPortfolios[q] = [0 for _ in range(len(budget))]
    visited.add(q)

    Q: Queue[int] = Queue()
    Q.put(q)
    while not Q.empty():
        q = Q.get()
        for i in range(noOfActions):
            if not (q >> i) & 1: # True if the ith bit is zero
                # Set the ith bit (which was zero) to one
                q_copy = q | (1 << i)

                action = actions[i]
                cost_vector = [sum(costs[action][j] * ((q_copy >> i) & 1) for i in range(noOfActions)) for j in range(r)]
                
                #sum(costs[i] * ((q_copy >> i) & 1) for i in range(noOfActions))
                if q_copy not in visited and all(cost_vector[j] <= budget[j] for j in range(r)):
                    visited.add(q_copy)
                    feasible.add(q_copy)
                    costsOfPortfolios[q_copy] = cost_vector
                    
                    # If strictly smaller then we explore the child nodes of this portfolio
                    if any(cost_vector[j] < budget[j] for j in range(r)):
                        Q.put(q_copy)

    return feasible, costsOfPortfolios

def dominates_with_cost(e1: float, e2: float, c1: list[float], c2: list[float]) -> bool:
    """
    Parameters:
        e1 (list[float]): Utilities when portfolio q1 has been applied.
        e2 (list[float]): Utilities when portfolio q2 has been applied.
        c1 (float): Cost of portfolio q1.
        c2 (float): Cost of portfolio q2.

    Returns:
        bool: True if portfolio q1 dominates portfolio q2 taking account the cost, false otherwise.
    """

    return (e1 > e2 and all(c1[j] <= c2[j] for j in range(len(c1)))) or (e1 == e2 and all(c1[j] <= c2[j] for j in range(len(c1))) and any(c1[j] < c2[j] for j in range(len(c1))))

def cost_efficient(e1: float, c1: list[float], feasiblePortfolios: list[tuple[float, list[float]]]) -> bool:
    """
    Parameters:
        e1 (list[float]): Utilities when portfolio q1 has been applied.
        c1 (float): Cost of portfolio q1.
        feasiblePortfolios (list[tuple[list[int], list[float], float]]): List of pairs consisting of a portfolio, its expected values for each extreme point and its cost.

    Returns:
        bool: True if portfolio q1 is cost-efficient, false otherwise.
    """
    #c1 = sum([costs[i] * q1[i] for i in range(len(costs))])
    #costOfPortfolios = {tuple(portfolio): sum([costs[i] * portfolio[i] for i in range(len(portfolio))]) for portfolio, _ in feasiblePortfolios}
    #return not (any(dominates(e2, e1) and costOfPortfolios[q2] <= c1 for (q2, e2) in feasiblePortfolios if q1 != q2)) or not (any(equal(e2, e1) and costOfPortfolios[q2] < c1 for (q2, e2) in feasiblePortfolios if q1 != q2))

    #dominated = False
    #for q2, e2 in feasiblePortfolios:
        #if q1 != q2:
            #if dominatesWithCost(e2, e1, costOfPortfolios[tuple(q2)], c1):
                #print(f"{q1} is dominated by {q2}, c1 = {c1}, e1 = {e1}, c2 = {costOfPortfolios[tuple(q2)]}, e2 = {e2}")
                #dominated = True
                #break
    #return not dominated
    #return not any(dominatesWithCost(e2, e1, costOfPortfolios[tuple(q2)], c1) for q2, e2 in feasiblePortfolios if q1 != q2)

    return not any(dominates_with_cost(e2, e1, c2, c1) for e2, c2 in feasiblePortfolios)# if q1 != q2) # Removing this check made it work? WTF


if __name__ == "__main__":
    print("This file is not meant to be run directly!")

    if False:
        n = 25
        start = time.time()
        portfolios_old, costs_old = generate_feasible_portfolios(n, [1.0 for _ in range(n)], n / 2)
        end = time.time()
        print(f"The old version took {(end-start):.2f} seconds")

        start = time.time()
        portfolios_new, costs_new = generateFeasiblePortfolios_bits(n, [1.0 for _ in range(n)], n / 2)
        end = time.time()
        print(f"The new algorithm took {(end-start):.2f} seconds")

        portfolio_set = set()
        for portfolio in portfolios_old:
            portfolio_set.add(portfolio_as_bitmask(portfolio))
        
        print(f"The generated sets of feasible portfolios are equal: {portfolios_new == portfolio_set}")
        print(f"The associated costs are also equal: {all([costs_old[tuple(portfolio)] == costs_new[portfolio_as_bitmask(portfolio)] for portfolio in portfolios_old])}")

        # The code above yielded these results
        """
        The old version took 1668.15 seconds
        The new algorithm took 318.89 seconds
        The generated sets of feasible portfolios are equal: True
        The associated costs are also equal: True
        """


    if False:
        n = 20

        start = time.time()
        portfolios = generate_all_portfolios(n) # This works
        end = time.time()
        
        print(f"Time to generate all portfolios: {(end - start):.2f}")

        #for p in portfolios:
        #    print(p)
    
    if False:
        times_old = []
        times_parallel = []
        for k in range(20, 31):
            print(f"Iteration {k-19}: ")
            average_old = 0
            average_parallel = 0
            for _ in range(5):
                costs = [random.choice([1.0,2.0,3.0]) for _ in range(k)]
                numberOfActions = len(costs)
                budget = k / 2.5

                start = time.time()
                feasible, feasibleCosts = generate_feasible_portfolios(numberOfActions, costs, budget)
                end = time.time()
                #print(f"Time to generate all feasible portfolios with the old method: {(end - start):.2f}")

                average_old += end - start
                
                #print(f"Number of feasible portfolios the old method: {len(feasible)}")
                #print(f"Number of feasible portfolios the new method: {len(feasible_new)}")
            print(f"Average with the old method: {average_old / 5}")
            print(f"Average with the new method: {average_parallel / 5}")
            times_old.append(average_old / 5)
            times_parallel.append(average_parallel / 5)
        
        plt.plot([k for k in range(20, 36)], times_old, label="Old method")
        plt.plot([k for k in range(20, 36)], times_parallel, label="New method")
        plt.xlabel("Number of actions")
        plt.ylabel("Time (s)")
        plt.legend()
        plt.show()

    if False:
        costs = [2.0,2,3]

        feasible, feasibleCosts = feasible_portfolios_filter(3, costs, 5) # This works
        for f in feasible:
            print(f"{f}: {sum(costs[i] * f[i] for i in range(len(costs)))}")
        
        #print(feasibleCosts)

        e1 = [1, 0.75]
        e2 = [1, 0.75]
        c1 = 0.75
        c2 = 0.75
        #print(dominates(e1, e2)) # Works
        print(equal(e1, e2)) # Works
        print(dominates_with_cost(e1, e2, c1, c2)) # Works

        performances = [[1,2], [2,4], [2,2], [2,5], [2,1.5], [3,5], [3,5]]
        costsOfPortfolios = [sum(costs[i] * portfolio[i] for i in range(len(portfolio))) for portfolio in feasible] # Correct
        feasibleOnes = list(zip(feasible, performances, costsOfPortfolios)) # Correct

        feasibleDict = {}

        for portfolio, performance, cost in feasibleOnes:
            feasibleDict[tuple(portfolio)] = performance
            #print(f"{portfolio}: {(performance, cost)}")

        #print(feasibleDict) # Correct

        q1 = [1,0,1]
        e1 = feasibleDict[tuple(q1)]
        c1 = sum(costs[i] * q1[i] for i in range(len(costs)))
        #print(costEfficient(e1, c1, list(feasibleOnes)))

        q2 = [0,1,1]
        e2 = feasibleDict[tuple(q2)]
        c2 = sum(costs[i] * q2[i] for i in range(len(costs)))
        
        #print(dominatesWithCost(e1, e2, c1, c2))