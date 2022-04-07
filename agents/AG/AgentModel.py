from mesa import Agent, Model
from AG.AGalgorithm import AG_Algorithm
from AG.classes import Customer
from typing import List
import numpy as np
import cost

def convertGlobalSolToAGSol(solution: List[List], capacities):
    """
    Convert solution of type:
    [[1, 2, 5], [4, 3]]
    to:
    [1, 2, 5, 4, 3]
    """
    sol_AG = []
    sol_converted = [item for sublist in solution for item in sublist]
    for id in sol_converted:
        customer = Customer(id, 0, 0, capacities[id-1])
        # print(customer)
        sol_AG.append(customer)
    return sol_AG


class AGent(Agent):
    """
    An agent with initial solution.
    With AG algorithm metaheuristic
    For step get the best Solution of last Population

    matrice: matrice des distances
    capacities: c'est la demande de chaque customer, tableu indice id customer
    w: weight cost
    max_capacity chaque camion a max cap globale

    """

    def __init__(self, unique_id, sol_init, matrice, w, capacities, max_capacity, model):
        super().__init__(unique_id, model)
        self.sol_init, self.matrice, self.w, self.capacities, self.max_capacity = sol_init, matrice, w, capacities, max_capacity
        self.result_cost = np.Inf

    def step(self):
        sol = self.model.selectSol()
        sol_init_converted = convertGlobalSolToAGSol(sol, self.capacities)
        agAlgorithm = AG_Algorithm(sol_init_converted, self.matrice, self.max_capacity)
        result = agAlgorithm.perform()
        self.result_cost = result.calculateScore()
        # print(result.chromosome)
        # print(result.calculateScore())
        self.result_sol = result.solution.convertGlobalSolution()
        # Print with the general function of cout
        print("AG:", cost.cout(self.result_sol, self.matrice, self.w), self.result_sol)
        self.model.insertSolStep(self.result_sol)


# class MyModel(Model):
#     """A model with some number of agents."""

#     def __init__(self, N, matrice, w, capacities, max_capacity, sol_init):
#         self.num_agents = N
#         self.schedule = RandomActivation(self)
#         # Create agents
#         for i in range(self.num_agents):
#             a = AGent(i, sol_init, matrice, w, capacities, max_capacity, self)
#             self.schedule.add(a)

#     def step(self):
#         """Advance the model by one step."""
#         self.schedule.step()


if __name__ == "__main__":
    COST_MATRIX = \
        np.matrix([[0, 14, 18, 9, 5, 7],
                [14, 0, 12, 4, 17, 1],
                [18, 12, 0, 3, 2, 1],
                [9, 4, 3, 0, 4, 8],
                [5, 17, 2, 4, 0, 11],
                [7, 1, 1, 8, 11, 0]])

    capacities_example = [20]*6
    max_capacity = 10
    sol_example = [[1, 2, 5], [4, 3]]

    model = MyModel(1, COST_MATRIX, 5, capacities_example, max_capacity, sol_example)
    for i in range(5):
        model.step()


