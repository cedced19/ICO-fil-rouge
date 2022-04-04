from mesa import Agent
from cost import *
from tabou_fct import tabu
import numpy as np

class TabouAgent(Agent):
    """An agent with initial solution."""
    n_max = 50
    aspiration = 1000
    def __init__(self, unique_id, sol_init, matrice, w, capacities, max_capacity, log, model):
        super().__init__(unique_id, model)
        self.sol_init, self.matrice, self.w, self.capacities, self.max_capacity, self.log = sol_init, matrice, w, capacities, max_capacity, log
        self.result_cost = np.Inf

    def step(self):
        sol = self.model.selectSol()
        result = tabu(sol, self.matrice, self.n_max, self.aspiration, self.w, self.capacities, self.max_capacity, log = self.log)
        self.result_cost = result[1]
        self.result_sol = result[0]
        print("Tabou", self.result_cost, self.result_sol)
        self.model.insertSol(self.result_sol)