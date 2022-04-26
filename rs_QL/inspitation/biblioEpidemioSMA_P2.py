import enum
import random
from mesa import Agent, Model
from mesa.space import MultiGrid
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
import numpy as np

# Constant a utiliser
ACTIONS_N = 8 
STATES_N = 3
LR = .85
Y = .99
PTJ_INFECTED = 50
PTJ_NOT_INFECT = 20
PTJ_SURVIVE = 5



# définition de la classe InfectionModel qui hérite de Model
class InfectionModel(Model):
    """A model for infection spread."""
    def __init__(self, N=10, width=10, height=10, ptrans=0.5, death_rate=0.02, recovery_days=21, recovery_sd=7):
        # QLearning
        self.Q = np.zeros((STATES_N, ACTIONS_N)) # Initialisation de la Q matrix
        self.I = 0 # Le step dans le quelle on se trouve (reduire la probabilite d'une action random)

        self.num_agents = N
        # paramètres pour calculer avec la loi normale la durée de récupération d'un individu infecté
        self.recovery_days = recovery_days
        self.recovery_sd = recovery_sd
        # probabilité de transformation du modèle
        self.ptrans = ptrans
        # taux permettant de décès
        self.death_rate = death_rate
        self.grid = MultiGrid(width, height, True)
        self.schedule = RandomActivation(self)
        # self.running = True
        self.dead_agents = []
        # Create agents
        for i in range(self.num_agents):
            a = MyAgent(i, self)
            self.schedule.add(a)
            # Add the agent to a random grid cell
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(a, (x, y))
            # make some agents infected at start
            infected = np.random.choice([0, 1], p=[0.98, 0.02])
            if infected == 1:
                a.state = State.INFECTED
                a.recovery_time = self.get_recovery_time()
        # La classe DataCollector permet de garder une trace de l'état de chaque agent à travers la simulation.
        self.datacollector = DataCollector(agent_reporters={"State": lambda a: a.state})

    # calcul du temps de récupération avec la loi normale 21 jours+/- 7 jours:
    def get_recovery_time(self):
        return int(self.random.normalvariate(self.recovery_days,
                                             self.recovery_sd))

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()
        self.I += 1

    def learn_Q(self, s, s1, action, reward):
        # Fonction pour train la matrix Q, avec les donnes de l'action que on a decide.
        self.Q[s, action] = self.Q[s, action] + LR*(reward + Y * np.max(self.Q[s1, :]) - self.Q[s, action])


class MyAgent(Agent):
    """ An agent in an epidemic model."""
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.age = self.random.normalvariate(20, 40)
        self.state = State.SUSCEPTIBLE
        self.infection_time = 0

    def move(self):
        """Move the agent"""
        # Take best decision from QTable
        Q = self.model.Q
        # Definir la nouvelle action
        Q2 = Q[self.state.numerator, :] + np.random.randn(1, ACTIONS_N)*(1. / (self.model.I + 1))
        # On prend la meilleur que on connait
        action = np.argmax(Q2)
        if random.randint(1,100) < 3:
            action = random.randint(0,7)
        possible_steps = self.model.grid.get_neighborhood(
            self.pos,
            moore=True,
            include_center=False)
        # print("Steps Possible: ", possible_steps)
        # print("Action decide: ", action, self.state )
        # De toutes les possible on prend celle que on avait decider
        new_position = possible_steps[action]
        # new_position = self.random.choice(possible_steps)
        self.model.grid.move_agent(self, new_position)
        return action

    def status(self):
        """Check infection status"""
        if self.state == State.INFECTED:
            drate = self.model.death_rate
            alive = np.random.choice([0, 1], p=[drate, 1-drate])
            if alive == 0:
                self.model.schedule.remove(self)
            t = self.model.schedule.time-self.infection_time
            if t >= self.recovery_time:
                self.state = State.REMOVED

    def contact(self):
        """Find close contacts and infect"""
        # Garde si l'agent a infecte un autre
        i_infected = False
        ptj = 0
        cellmates = self.model.grid.get_cell_list_contents([self.pos])
        if len(cellmates) > 1:
            for other in cellmates:
                if self.random.random() > self.model.ptrans:
                    continue
                if self.state is State.INFECTED and other.state is State.SUSCEPTIBLE:
                    other.state = State.INFECTED
                    other.infection_time = self.model.schedule.time
                    other.recovery_time = self.model.get_recovery_time()
                    i_infected = True
                    # Chaque fois qu'il infecte il perd des points
                    ptj -= PTJ_INFECTED
        if self.state == State.INFECTED and not i_infected:
            # Si il pouvait infecter mais il n'a pas infecter
            ptj += PTJ_NOT_INFECT
        elif not i_infected:
            # Si il peut pas infecter, juste il survie
            ptj += PTJ_SURVIVE
        return ptj


    def step(self):
        self.status()
        s = self.state # On garde le state dans le qu'il etait
        action = self.move()
        reward = self.contact()
        s1 = self.state # On garde le nouveau state
        self.model.learn_Q(s, s1, action, reward) # On train notre Q

class State(enum.IntEnum):
    SUSCEPTIBLE = 0
    INFECTED = 1
    REMOVED = 2