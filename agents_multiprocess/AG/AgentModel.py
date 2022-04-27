from mesa import Agent, Model
from AG.AGalgorithm import AG_Algorithm
from AG.classes import Customer
from typing import List
import numpy as np
import cost
from AG.AGQlearning import AGQlearning

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

    def __init__(self, unique_id, sol_init, matrice, w, capacities, max_capacity, model, Qlearning = False):
        super().__init__(unique_id, model)
        self.sol_init, self.matrice, self.w, self.capacities, self.max_capacity = sol_init, matrice, w, capacities, max_capacity
        self.result_cost = np.Inf

        self.P_MUT = 0.6
        self.P_CROSS = 0.6
        self.Qlearning = None
        if (Qlearning):
            self.Qlearning = AGQlearning(self)

    def step(self, return_dict, id):

        if(self.Qlearning):
            self.P_CROSS, self.P_MUT = self.Qlearning.episode()

        sol = self.model.selectSol(self.unique_id)
        sol_init_converted = convertGlobalSolToAGSol(sol, self.capacities)
        # Fix algorithm with parameters
        agAlgorithm = AG_Algorithm(sol_init_converted, self.matrice, self.max_capacity)
        agAlgorithm.P_MUT = self.P_MUT
        agAlgorithm.P_CROSS = self.P_CROSS

        result = agAlgorithm.perform()
        self.result_cost = result.calculateScore()
        # print(result.chromosome)
        # print(result.calculateScore())
        self.result_sol = result.solution.convertGlobalSolution()
        # Print with the general function of cout
        if (self.Qlearning):
            print("AG QL:", cost.cout(self.result_sol, self.matrice, self.w), self.result_sol)
        else:
            print("AG:", cost.cout(self.result_sol, self.matrice, self.w), self.result_sol)
        self.model.insertSolStep(self.result_sol, self.unique_id)

        if(self.Qlearning):
            self.Qlearning.learn_Q(self.result_cost)
        
        return_dict[id] = self.result_sol
        return return_dict

    def plot(self):
        self.Qlearning.plot()


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


