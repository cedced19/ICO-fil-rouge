from cmath import log
from multiprocessing import pool
from matplotlib.cm import get_cmap
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
import matplotlib.pyplot as plt
from cost import cout


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
            q.put((count, p, agent.unique_id, agent.model.differentPool)) # definition of the r vector
            if (agent.model.differentPool):
                self.pool_step.append([])
        while not q.empty():
            r = q.get()
            p = r[1]
            p.join()
            if (r[3]):
                self.pool_step[r[2]].append(return_dict[r[0]])
            else:
                self.pool_step.append(return_dict[r[0]])

        self.steps += 1
        self.time += 1

class MyModel(Model):
    """A model with some number of agents."""

    def __init__(self, N, matrice, w, capacities, max_capacity, sol_init, differentPool, ql, log = False):
        # Pool of possible solution, commence avec une solution initiale
        self.num_agents = N
        self.differentPool = differentPool
        if (not self.differentPool):
            self.pool = [sol_init]
            self.pool_step = []
            self.scores_pool = []
        else:
            self.pool = []
            self.pool_step = []
            for i in range(0,self.num_agents):
                self.pool.append([sol_init])
                self.pool_step.append([])
            self.scores_pool = [[], [], []]
        self.schedule = MultiProcessActivation(self)

        

        self.matrice = matrice
        self.w = w

        # Creation des trois agent 1 pour chaque algo.
        # Tabou Agent
        self.schedule.add(TabouAgent(0, sol_init, matrice, w, capacities, max_capacity, log, ql, self))
        # Recuit Simule
        self.schedule.add(RSAgent(1, sol_init, matrice, w, capacities, max_capacity, log, ql, self))
        # Genetique
        self.schedule.add(AGent(2, sol_init, matrice, w, capacities, max_capacity, self, ql))


    def step(self):
        """Advance the model by one step."""
        if (not self.differentPool):
            self.pool_step = []
        else:
            self.pool_step = []
            for i in range(0,self.num_agents):
                self.pool_step.append([])
        self.schedule.step()
        self.pool_step = self.schedule.pool_step
        self.insertStep()
        print("POOL: ", self.pool, len(self.pool))
        self.scorePool()

    def scorePool(self):
        if not self.differentPool:
            min = np.Inf
            total = 0
            for sol in self.pool:
                print(sol)
                coutSol = cout(sol, self.matrice, self.w)
                total += coutSol
                if coutSol < min:
                    min = coutSol
            avg = total/len(self.pool)
            self.scores_pool.append([avg, min])
        else:
            i = 0
            for pool in self.pool:
                min = np.Inf
                total = 0
                for sol in pool:
                    print(sol)
                    coutSol = cout(sol, self.matrice, self.w)
                    total += coutSol
                    if coutSol < min:
                        min = coutSol
                avg = total/len(self.pool)
                self.scores_pool[i].append([avg, min])
                i += 1
        print(self.scores_pool)

    def selectSol(self, unique_id):
        if (not self.differentPool):
            sol = choice(self.pool)
        else:
            sol = choice(self.pool[unique_id])
        return sol

    def insertSolStep(self, sol, unique_id):
        if (not self.differentPool):
            self.pool_step.append(sol)
        else:
            self.pool_step[unique_id].append(sol)

    def insertStep(self):
        #self.pool = self.pool_step.copy()
        #self.pool_step = []

        def sort_by_g(elem):
            return elem[0]

        POOL_RADIUS = 5 # TODO HOW choose a good radius???
        # Calculate the distance between solution for add only the effective
        if (self.differentPool):
            for i in range(0,self.num_agents):
                gSols = []
                print(self.pool_step[i], self.pool[i])
                for sol in self.pool_step[i]:
                    for pool_sol in self.pool[i]:
                        gSol = 0
                        distance = compare_sol(sol, pool_sol)
                        if (distance > POOL_RADIUS):
                            pass
                        else:
                            gSol += 1 - distance/POOL_RADIUS
                    gSols.append([gSol, sol])
                gSols.sort(key=sort_by_g)
                # print(gSols)
                self.pool[i].append(gSols[0][1])
        else:
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

    def plot(self):
        if not self.differentPool:
            plt.plot([x[0] for x in self.scores_pool], 'r', label="AVG Score in pool")
            plt.plot([x[1] for x in self.scores_pool], 'g', label="MIN Score in pool")
            plt.legend()
            plt.show()
        else:
            cmap = get_cmap('gist_rainbow')
            print(self.pool)
            print()
            print(self.scores_pool)

            plt.plot([x[0] for x in self.scores_pool[0]], color=cmap(1/6), label="AVG Score in pool Tabou")
            plt.plot([x[1] for x in self.scores_pool[0]], color=cmap(2/6), label="MIN Score in pool Tabou")
            plt.plot([x[0] for x in self.scores_pool[1]], color=cmap(3/6), label="AVG Score in pool RS")
            plt.plot([x[1] for x in self.scores_pool[1]], color=cmap(4/6), label="MIN Score in pool RS")
            plt.plot([x[0] for x in self.scores_pool[2]], color=cmap(5/6), label="AVG Score in pool AG")
            plt.plot([x[1] for x in self.scores_pool[2]], color=cmap(6/6), label="MIN Score in pool AG")
            plt.legend()
            plt.show()
            print()


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

    enemyApproach = True
    QLearning = True
    model = MyModel(3, matrice_example, 5, capacities_example, max_capacity, sol_example, enemyApproach, QLearning)
    for i in range(3):
        model.step()
    model.plot()