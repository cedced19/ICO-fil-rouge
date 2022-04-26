from random import randint, random
from mesa import Agent
import numpy as np


class AGQlearning:

    states_n = 10 # nombre d'etats: 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
    actions_n = 3 # nombre d'actions: 0: augmenter, 1: diminuer, 2: rester
    ACTIONS = {0: 0.1, 1: -0.1, 2: 0}
    I = 0

    # parametres d'apprentissage
    lr=.85 #appelé alpha, vitesse d'apprentissage : degré d’acceptation de la nouvelle valeur par rapport à l’ancienne
    y=.95 #appelé aussi gamma, facteur d'actualisation,  utilisé pour équilibrer la récompense immédiate et future. nous appliquons la décote à la récompense future. En général, cette valeur peut varier entre 0,8 et 0,99

    # Greedy Epsilon
    epsilon = 0.8

    def __init__(self, agent) -> None:
        self.Qmut = np.zeros((self.states_n, self.actions_n))
        self.Qcross = np.zeros((self.states_n, self.actions_n))
        self.cumul_reward_list = []
        self.actions_list = []
        self.states_list = []

        self.agent = agent

    def step(self):
        self.episode()

    def episode(self):
        '''
        Decide from a initial state, new actions for P_MUT and P_CROSS
        return:
        newStateCross: float , newStateMut: float
        '''
        self.I += 1
        # Mutation
        stateMut = self.agent.P_MUT
        Q2Mut = self.Qmut[int(stateMut*10), :] + np.random.randn(1, self.actions_n)*(1. / (self.I + 1))
        actionMut = np.argmax(Q2Mut)

        if random() < self.epsilon:
            actionMut = randint(0, 2)
        newStateMut = self.agent.P_MUT + self.ACTIONS[actionMut]

        # Cross
        stateCross = self.agent.P_CROSS
        Q2Cross = self.Qcross[int(10*stateCross), :] + np.random.randn(1, self.actions_n)*(1. / (self.I + 1))
        actionCross = np.argmax(Q2Cross)
        if random() < self.epsilon:
            actionCross = randint(0, 2)
        newStateCross = self.agent.P_CROSS + self.ACTIONS[actionCross]

        self.states_list.append([(stateMut, newStateMut), (stateCross, newStateCross)])
        self.actions_list.append([actionMut, actionCross])

        if (self.epsilon > 0.1):
            self.epsilon -= 0.05

        return newStateCross, newStateMut

    def learn_Q(self, reward):

        s, s1 = self.states_list[-1][0]
        action = self.actions_list[-1][0]
        self.Qmut[int(10*s), action] = self.Qmut[int(10*s), action] + self.lr*(reward + self.y * np.max(self.Qmut[int(10*s1), :]) - self.Qmut[int(10*s), action])

        s, s1 = self.states_list[-1][0]
        action = self.actions_list[-1][0]
        self.Qcross[int(10*s), action] = self.Qcross[int(10*s), action] + self.lr*(reward + self.y * np.max(self.Qcross[int(10*s1), :]) - self.Qcross[int(10*s), action])

        self.cumul_reward_list.append(reward)
        print(f"rewards: {self.cumul_reward_list}")
        print(f"actions: {self.actions_list}")
        print(f"states: {self.states_list}")
