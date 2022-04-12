from mesa import Agent
from cost import *
from rs_fct import *
import numpy as np

class RSAgent(Agent):
    """An agent with initial solution."""
    t0 = 200
    a = 0.2
    n_iter_cycle = 10
    def __init__(self, unique_id, sol_init, matrice, w, capacities, max_capacity, log, model):
        super().__init__(unique_id, model)
        self.sol_init, self.matrice, self.w, self.capacities, self.max_capacity, self.log = sol_init, matrice, w, capacities, max_capacity, log
        self.result_cost = np.Inf

    def step(self, return_dict, id):
        sol = self.model.selectSol()
        result = recuit(sol, self.matrice, self.w, self.t0, self.a, self.n_iter_cycle, self.capacities, self.max_capacity, log = self.log)
        self.result_cost = result[1]
        self.result_sol = result[0]
        print("RS", self.result_cost, self.result_sol)
        return_dict[id] = self.result_sol
        return return_dict