from mesa import Agent
from cost import *
from tabou_fct import tabu
from tabou_ql_fct import tabu as tabu_ql
import numpy as np

class TabouAgent(Agent):
    """An agent with initial solution."""
    n_max = 50
    aspiration = 1000
    def __init__(self, unique_id, sol_init, matrice, w, capacities, max_capacity, log, ql, model):
        super().__init__(unique_id, model)
        self.sol_init, self.matrice, self.w, self.capacities, self.max_capacity, self.log, self.ql = sol_init, matrice, w, capacities, max_capacity, log, ql
        self.result_cost = np.Inf

    def step(self, return_dict, id):
        sol = self.model.selectSol(self.unique_id)
        if (self.ql):
            result = tabu_ql(sol, self.matrice, self.n_max, self.aspiration, self.w, self.capacities, self.max_capacity, log = self.log)
            print("Tabou QL", result[1], result[0])
        else:
            result = tabu(sol, self.matrice, self.n_max, self.aspiration, self.w, self.capacities, self.max_capacity, log = self.log)
            print("Tabou", result[1], result[0])
        self.result_cost = result[1]
        self.result_sol = result[0]
        self.model.insertSolStep(self.result_sol, self.unique_id)
        return_dict[id] = self.result_sol
        return return_dict