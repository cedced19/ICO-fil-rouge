from multiprocessing import pool
from mesa import Model
from mesa.time import BaseScheduler
from AG.AgentModel import AGent
from tabou_agent import TabouAgent
from rs_agent import RSAgent
import numpy as np
from random import choice
from compare_sol import compare_sol
import multiprocessing
import queue


class MultiProcessActivation(BaseScheduler):
    """A scheduler which activates each agent once per step, in random order,
    with the order reshuffled every step.
    This is equivalent to the NetLogo 'ask agents...' and is generally the
    default behavior for an ABM.
    Assumes that all agents have a step(model) method.
    """
    
    def step(self) -> None:
        """Executes the step of all agents, one at a time, in
        random order.
        """
        self.pool_step = []
        q = queue.Queue()
        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        for count, agent in enumerate(self.agent_buffer(shuffled=True)):
            p = multiprocessing.Process(target=agent.step, args=(return_dict, count))
            p.start()
            q.put((count, p))
        while not q.empty():
            r = q.get()
            p = r[1]
            p.join()
            self.pool_step.append(return_dict[r[0]])
            

        self.steps += 1
        self.time += 1

class MyModel(Model):
    """A model with some number of agents."""

    def __init__(self, N, matrice, w, capacities, max_capacity, sol_init, ql, log = False):
        # Pool of possible solution, commence avec une solution initiale
        self.pool = [sol_init]
        self.pool_step = []
        self.num_agents = N
        self.schedule = MultiProcessActivation(self)

        # Creation des trois agent 1 pour chaque algo.
        # Tabou Agent
        self.schedule.add(TabouAgent(1, sol_init, matrice, w, capacities, max_capacity, log, ql, self))
        # Recuit Simule
        self.schedule.add(RSAgent(2, sol_init, matrice, w, capacities, max_capacity, log, ql, self))
        # Genetique
        self.schedule.add(AGent(3, sol_init, matrice, w, capacities, max_capacity, self, ql))


    def step(self):
        """Advance the model by one step."""
        self.pool_step = []
        self.schedule.step()
        self.pool_step = self.schedule.pool_step
        self.insertStep()

    def selectSol(self):
        sol = choice(self.pool)
        return sol

    def insertSolStep(self, sol):
        self.pool_step.append(sol)

    def insertStep(self):
        #self.pool = self.pool_step.copy()
        #self.pool_step = []

        def sort_by_g(elem):
            return elem[0]

        POOL_RADIUS = 5 # TODO HOW choose a good radius???
        # Calculate the distance between solution for add only the effective
        gSols = []
        for sol in self.pool_step:
            for pool_sol in self.pool:
                gSol = 0
                distance = compare_sol(sol, pool_sol)
                if (distance > POOL_RADIUS):
                    pass
                else:
                    gSol += 1 - distance/POOL_RADIUS
            gSols.append([gSol, sol])
        gSols.sort(key=sort_by_g)
        self.pool.append(gSols[0][1])
        print(f"Solution added to the pool: {gSols[0][1]} with a g={gSols[0][0]}")



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

    model = MyModel(1, matrice_example, 5, capacities_example, max_capacity, sol_example, True)
    for i in range(3):
        model.step()