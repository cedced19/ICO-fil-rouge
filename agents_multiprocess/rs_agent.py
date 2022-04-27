from mesa import Agent
from cost import *
from rs_fct import *
from rs_fct import recuit as recuit_ql
import numpy as np

class RSAgent(Agent):
    """An agent with initial solution."""
    t0 = 200
    a = 0.2
    n_iter_cycle = 5
    def __init__(self, unique_id, sol_init, matrice, w, capacities, max_capacity, log, ql, model):
        super().__init__(unique_id, model)
        self.sol_init, self.matrice, self.w, self.capacities, self.max_capacity, self.log, self.ql = sol_init, matrice, w, capacities, max_capacity, log, ql
        self.result_cost = np.Inf

    def step(self, return_dict, id):
        sol = self.model.selectSol(self.unique_id)
        if (self.ql):
            result = recuit_ql(sol, self.matrice, self.w, self.t0, self.a, self.n_iter_cycle, self.capacities, self.max_capacity, log = self.log)
            print("RS QL", result[1], result[0])
        else:
            result = recuit(sol, self.matrice, self.w, self.t0, self.a, self.n_iter_cycle, self.capacities, self.max_capacity, log = self.log)
            print("RS", result[1], result[0])
        self.result_cost = result[1]
        self.result_sol = result[0]
        self.model.insertSolStep(self.result_sol, self.unique_id)
        return_dict[id] = self.result_sol
        return return_dict

    def plot(self):
        pass