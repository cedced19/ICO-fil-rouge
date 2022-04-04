from mesa import Model
from mesa.time import RandomActivation
from AG.AgentModel import AGent
from tabou_agent import TabouAgent
from rs_agent import RSAgent
import numpy as np
from random import choice

class MyModel(Model):
    """A model with some number of agents."""
    
    def __init__(self, N, matrice, w, capacities, max_capacity, sol_init, log = False):
        # Pool of possible solution, commence avec une solution initiale
        self.pool = [sol_init]
        self.num_agents = N
        self.schedule = RandomActivation(self)

        # Creation des trois agent 1 pour chaque algo.
        # Tabou Agent
        self.schedule.add(TabouAgent(1, sol_init, matrice, w, capacities, max_capacity, log, self))
        # Recuit Simule
        self.schedule.add(RSAgent(2, sol_init, matrice, w, capacities, max_capacity, log, self))
        # Genetique
        self.schedule.add(AGent(3, sol_init, matrice, w, capacities, max_capacity, self))


    def step(self):
        """Advance the model by one step."""
        self.schedule.step()

    def selectSol(self):
        sol = choice(self.pool)
        return sol

    def insertSol(self, sol):
        self.pool.append(sol)

if __name__ == "__main__":
    matrice_example = np.matrix([[0, 14, 18, 9, 5, 7], 
           [14, 0, 12, 4, 17, 1],
           [18, 12, 0, 3, 2, 1],
           [9, 4, 3, 0, 4, 8],
           [5, 17, 2, 4, 0, 11],
           [7, 1, 1, 8, 11, 0]])

    sol_example = [[1, 2, 5], [4, 3]]

    max_capacity = 100
    capacities_example = [30]*6

    model = MyModel(1, matrice_example, 5, capacities_example, max_capacity, sol_example)
    for i in range(3):
        model.step()