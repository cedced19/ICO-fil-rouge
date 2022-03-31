from classes import Customer, Solution
from random import shuffle, random, randint, seed
import numpy as np
from copy import copy
from typing import List
from time import process_time  # CPU Time
import matplotlib.pyplot as plt
from alive_progress import alive_bar  # Progress Bar
from big_example import matrice_example2  # Matrice distance avec 52 customer
from mesa import Agent


class Individu:

    def __init__(self, chromosome: list) -> None:
        self.chromosome = chromosome
        self.solution = Solution(COST_MATRIX)
        self.score = self.calculateScore()

    def calculateScore(self):
        chromosomeCopy = self.chromosome.copy()
        while chromosomeCopy:
            customer = chromosomeCopy.pop()
            self.solution.addCustomer(customer)
        return self.solution.calculateTotalCost()

    def mutation(self):
        # Transposition de deux genes consecutifs
        pos = randint(0, len(self.chromosome)-2)
        temp = self.chromosome[pos]
        self.chromosome[pos] = self.chromosome[pos + 1]
        self.chromosome[pos + 1] = temp

    def __copy__(self):
        return Individu(self.chromosome)

    def __repr__(self) -> str:
        return f"Individu{'{ '}Chromosome: {self.chromosome} \
            Score: {self.score}{'}'}\n"


class AG_Algorithm:

    def cross(self, individu1: Individu, individu2: Individu):

        # Trouver trois clients aléatoire qui ne sont pas identiques
        chif1 = randint(1, len(individu1.chromosome))
        chif2 = chif1
        while(chif2 == chif1):
            chif2 = randint(1, len(individu1.chromosome))
        chif3 = chif1
        while(chif3 == chif1) or (chif3 == chif2):
            chif3 = randint(1, len(individu1.chromosome))

        # Trouver leurs emplacements dans les solutions
        def find_customer(individu, id):
            return [individu.chromosome.index(customer) for
                    customer in individu.chromosome if customer.id == id][0]

        place1 = [find_customer(individu1, chif1), find_customer(individu1, chif2),
                find_customer(individu1, chif3)]
        place2 = [find_customer(individu2, chif1), find_customer(individu2, chif2),
                find_customer(individu2, chif3)]

        # Ordonner leur emplacements
        place1.sort()
        place2.sort()

        individu1_copy = individu1.chromosome.copy()
        individu2_copy = individu2.chromosome.copy()

        # Remplacer le premier client concerné de solution 1 avec ce de solution 2
        # Remplacer le premier client concerné de solution 2 avec ce de solution 1
        for i in range(3):
            individu1_copy[place1[i]] = individu2.chromosome[place2[i]]
            individu2_copy[place2[i]] = individu1.chromosome[place1[i]]

        individu1.chromosome = individu1_copy
        individu2.chromosome = individu2_copy

        # Renvoyer les deux individus
        return individu1, individu2

    P_MUT = 0.6
    P_CROSS = 0.6
    N_ITERATION = 20
    N_POPULATION = 4

    def __init__(self, base_individu) -> None:
        self.population = []
        self.population_initial = []
        self.baseIndividu = base_individu
        for i in range(self.N_POPULATION):
            shuffle(self.baseIndividu)
            newIndividu = self.baseIndividu.copy()
            self.population_initial.append(Individu(newIndividu))

        self.population.append(self.population_initial)

        self.score_by_iteration = {}
        self.score_by_time = {}
        self.bestIndividu = None

    def ScorePopulation(self, population: List[Individu]):
        sum_score = 0
        min_score = None
        for individu in population:
            sum_score += individu.score
            if ((not min_score) or individu.score < min_score):
                min_score = individu.score
        return (sum_score.max()/len(population)), min_score.max()

    def perform(self) -> Individu:
        start = process_time()

        S_total = [[]]
        with alive_bar(self.N_ITERATION, title="AG Algorithm:", ctrl_c=False, theme="smooth") as bar:
            for t in range(0, self.N_ITERATION):
                # Rajoute P[t+1] ; S[t+1]
                self.population.append([])
                S_total.append([])
                # Selection N individus de notre P(t)
                S = S_total[t]

                def by_score(individu: Individu):
                    return individu.score

                self.population[t].sort(key=by_score)
                if ((len(self.population[t])//2 + len(S_total[t])) % 2 == 0):
                    N_SELECTION = len(self.population[t])//2
                else:
                    N_SELECTION = len(self.population[t])//2 + 1

                for i in range(N_SELECTION):
                    S.append(copy(self.population[t][i]))

                # Group by pairs
                pairs = list(zip(S[::2], S[1::2]))
                for pair in pairs:
                    if random() < self.P_CROSS:
                        child1, child2 = self.cross(*pair)
                        S_total[t+1].append(copy(child1))
                        S_total[t+1].append(copy(child2))
                    else:
                        S_total[t+1].append(copy(pair[0]))
                        S_total[t+1].append(copy(pair[1]))

                for individu in S_total[t+1]:
                    if random() < self.P_MUT:
                        individu.mutation()
                    self.population[t+1].append(copy(individu))

                score_population = self.ScorePopulation(self.population[t])
                self.score_by_iteration[t] = score_population
                self.score_by_time[round(process_time() - start, 2)] = score_population

                bar()

        for pop in self.population:
            # print("Population n:", i)
            # print(pop)
            min = 0
            min_ind = None
            for ind in pop:
                if (ind.score < min) or (not min_ind):
                    min = ind.score
                    min_ind = ind
                    self.bestIndividu = ind
            # print(min_ind.solution)
            # print()
            i += 1

        return self.bestIndividu

    def performance_plot(self):
        mean_score_iteration = [self.score_by_iteration[key][0] for
                                key in self.score_by_iteration]
        min_score_iteration = [self.score_by_iteration[key][1] for
                               key in self.score_by_iteration]

        mean_score_time = [self.score_by_time[key][0] for
                           key in self.score_by_time]
        min_score_time = [self.score_by_time[key][1] for
                          key in self.score_by_time]

        plt.plot(mean_score_iteration, color="red", label="mean score")
        plt.plot(min_score_iteration, color="green", label="min score")
        plt.legend()
        plt.title("Score AG par iteration")
        plt.xlabel("N. Iteration")
        plt.ylabel("Score")
        plt.show()

        plt.plot(list(self.score_by_time), mean_score_time, color="red", label="mean score")
        plt.plot(list(self.score_by_time), min_score_time,  color="green", label="min score")
        plt.legend()
        # plt.xticks(list(score_by_time.keys()))
        plt.title("Score AG par temps")
        plt.xlabel("temps [s]")
        plt.ylabel("Score")
        plt.show()


def convertGlobalSolToAGSol(solution: List[List]):
    """
    Convert solution of type:
    [[1, 2, 5], [4, 3]]
    to:
    [1, 2, 5, 4, 3]
    """

    return [item for sublist in solution for item in sublist]


class AGent(Agent):
    """
    An agent with initial solution.
    With AG algorithm methauristic
    """

    def __init__(self, unique_id, sol_init, matrice, n_max, aspiration, w, capacities, max_capacity, model):
        super().__init__(unique_id, model)
        self.sol_init, self.matrice, self.n_max, self.aspiration, self.w, self.capacities, self.max_capacity = sol_init, matrice, n_max, aspiration, w, capacities, max_capacity
        self.result_cost = np.Inf

    def step(self):
        print("step")
        sol_init_converted = convertGlobalSolToAGSol(self.sol_init)
        agAlgorithm = AG_Algorithm(sol_init_converted)
        result = agAlgorithm.perform()
        self.result_cost = result.calculateScore()
        self.result_sol = result.solution.convertGlobalSolution()


if __name__ == "__main__":
    COST_MATRIX = \
        np.matrix([[0, 14, 18, 9, 5, 7],
                [14, 0, 12, 4, 17, 1],
                [18, 12, 0, 3, 2, 1],
                [9, 4, 3, 0, 4, 8],
                [5, 17, 2, 4, 0, 11],
                [7, 1, 1, 8, 11, 0]])


    # Example plus grand avec 52 clients, la demande de chaqun est aleatoire
    # baseInidividu = [Customer(i, 0, 0, randint(10, 99)) for i in range(1, 52)]
    # COST_MATRIX = matrice_example2


    # Création de population et evaluer F(x):
    gene1 = Customer(1, 0, 0, 1)
    gene2 = Customer(2, 0, 0, 1)
    gene3 = Customer(3, 0, 0, 1)
    gene4 = Customer(4, 0, 0, 1)
    gene5 = Customer(5, 0, 0, 1)
    baseInidividu = [gene1, gene2, gene3, gene4, gene5]


    # ag = AG_Algorithm(baseInidividu)
    # ag.perform()
    # ag.performance_plot()

    print(convertGlobalSolToAGSol([[1, 2, 5], [4, 3]]))
