import numpy as np
import time
import itertools
from queue import Queue
import random
import matplotlib.pyplot as plt

def generateAllPortfolios(numberOfProjects: int) -> list[list[int]]:
    """
        Parameters:
            projects (int): Number of projects, which are identified by index
        
        Returns:
            list[tuple[int, ..., int]]: List of all possible portfolios, each portfolio is represented as a list of 
                                        binary values signifying if a project is selected or not in the portfolio
    """

    # The generation of all portfolios when there are 25 projects took 30s on my PC
    # This is not yet a problem since this is run only once in the algorithm
    # But this maybe unviable later on since memory might run out
    # 2^25 = 33 554 432

    #return np.array(np.meshgrid(*([0,1] for _ in range(numberOfProjects)))).T.reshape(-1, numberOfProjects)
    #return list(map(list, itertools.product([0,1], repeat = numberOfProjects)))
    return list(map(list, itertools.product([0,1], repeat = numberOfProjects)))

def feasiblePortfolios(noOfActions: int, costs: list[float], budget: float) -> tuple[list[list[int]], dict]:
    """
        Parameters:
            noOfActions (int): Number of possible reinforcement actions.
            costs (list[float]): The costs of the reinforcement actions.
            budget (float): The budget available.
            
        Returns:
            list[tuple[int, ..., int]]: The set of feasible portfolios.
    """

    feasible = []
    allPortfolios = np.array(generateAllPortfolios(noOfActions))
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
def generateFeasiblePortfolios_old(noOfActions: int, costs: list[float], budget: float) -> tuple[list[list[int]], dict[tuple[int, ...], float]]:
    """
        Parameters:
            noOfActions (int): Number of possible reinforcement actions.
            costs (list[float]): The costs of the reinforcement actions.
            budget (float): The budget available.
            
        Returns:
            tuple[list[int], dict[tuple[int, ...], float]]: The set of feasible portfolios.
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

# A new improved version, which hashes the portfolios (binary vectors) to be just integers to speed up memory accesses and computations
def generateFeasiblePortfolios(noOfActions: int, costs: list[float], budget: float) -> tuple[set[int], dict[int, float]]:
    """
        Parameters:
            noOfActions (int): Number of possible reinforcement actions.
            costs (list[float]): The costs of the reinforcement actions.
            budget (float): The budget available.
            
        Returns:
            tuple[list[int], dict[tuple[int, ...], float]]: The set of feasible portfolios.
    """

    feasible: set[int] = set()
    costsOfPortfolios: dict[int, float] = {}

    visited: set[int] = set()

    q = 0
    feasible.add(q)
    costsOfPortfolios[q] = 0
    visited.add(q)

    Q: Queue[int] = Queue()
    Q.put(q)
    while not Q.empty():
        q = Q.get()
        for i in range(noOfActions):
            if (q >> i) & 1 == 0:
                # Set the i:th bit (which was zero) to one
                q_copy = q | (1 << i)
                
                cost = sum(costs[i] * ((q_copy >> i) & 1) for i in range(noOfActions))
                if q_copy not in visited and cost <= budget:
                    visited.add(q_copy)
                    feasible.add(q_copy)
                    costsOfPortfolios[q_copy] = cost
                    
                    # If strictly smaller then we explore the child nodes of this portfolio
                    if cost < budget:
                        Q.put(q_copy)

    return feasible, costsOfPortfolios

def dominates(e1: list[float], e2: list[float]) -> bool:
    """
    Parameters:
        e1 (float): Expected values of portfolio q1.
        e2 (float): Expected values of portfolio q2.

    Returns:
        bool: True if portfolio q1 dominates portfolio q2, false otherwise.
    """

    return all(e1[i] >= e2[i] for i in range(len(e1))) and any(e1[i] > e2[i] for i in range(len(e1)))

def equal(e1: list[float], e2: list[float]) -> bool:
    """
    Parameters:
        e1 (list[float]): Expected values of portfolio q1.
        e2 (list[float]): Expected values of portfolio q2.

    Returns:
        bool: True if portfolios q1 and q2 are equally efficient.
    """

    return all(e1[i] == e2[i] for i in range(len(e1))) # alternate way to check this
    #return np.array_equal(e1, e2)

def dominatesWithCost(e1: list[float], e2: list[float], c1: float, c2: float) -> bool:
    """
    Parameters:
        e1 (list[float]): Expected performances of portfolio q1.
        e2 (list[float]): Expected performances of portfolio q2.
        c1 (float): Cost of portfolio q1.
        c2 (float): Cost of portfolio q2.

    Returns:
        bool: True if portfolio q1 dominates portfolio q2 taking account the cost, false otherwise.
    """

    return (dominates(e1, e2) and c1 <= c2) or (equal(e1, e2) and c1 < c2)

def costEfficient(e1: list[float], c1: float, feasiblePortfolios: list[tuple[list[float], float]]) -> bool:
    """
    Parameters:
        e1 (list[float]): Expected values of portfolio q1 for each extreme point.
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

    return not any(dominatesWithCost(e2, e1, c2, c1) for e2, c2 in feasiblePortfolios)# if q1 != q2) # Removing this check made it work? WTF


if __name__ == "__main__":
    print("This file is not meant to be run directly!")

    if False:
        n = 25
        start = time.time()
        portfolios_old, costs_old = generateFeasiblePortfolios(n, [1.0 for _ in range(n)], n / 2)
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
        portfolios = generateAllPortfolios(n) # This works
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
                feasible, feasibleCosts = generateFeasiblePortfolios(numberOfActions, costs, budget)
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

        feasible, feasibleCosts = feasiblePortfolios(3, costs, 5) # This works
        for f in feasible:
            print(f"{f}: {sum(costs[i] * f[i] for i in range(len(costs)))}")
        
        #print(feasibleCosts)

        e1 = [1, 0.75]
        e2 = [1, 0.75]
        c1 = 0.75
        c2 = 0.75
        #print(dominates(e1, e2)) # Works
        print(equal(e1, e2)) # Works
        print(dominatesWithCost(e1, e2, c1, c2)) # Works

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