from classes import Customer, Solution
from random import shuffle, random, randint, seed
import numpy as np
from copy import copy
from typing import List
from time import process_time  # CPU Time
import matplotlib.pyplot as plt
from alive_progress import alive_bar  # Progress Bar
from big_example import matrice_example2  # Matrice distance avec 52 customer


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


def ScorePopulation(population: List[Individu]):
    sum_score = 0
    min_score = None
    for individu in population:
        sum_score += individu.score
        if ((not min_score) or individu.score < min_score):
            min_score = individu.score
    return (sum_score.max()/len(population)), min_score.max()


def cross(individu1: Individu, individu2: Individu):

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


seed(100)

P_MUT = 0.6
P_CROSS = 0.6
N_ITERATION = 20
N_POPULATION = 4
COST_MATRIX = \
    np.matrix([[0, 14, 18, 9, 5, 7],
              [14, 0, 12, 4, 17, 1],
              [18, 12, 0, 3, 2, 1],
              [9, 4, 3, 0, 4, 8],
              [5, 17, 2, 4, 0, 11],
              [7, 1, 1, 8, 11, 0]])


# Example plus grand avec 52 clients, la demande de chaqun est aleatoire
baseInidividu = [Customer(i, 0, 0, randint(10, 99)) for i in range(1, 52)]
COST_MATRIX = matrice_example2


# Création de population et evaluer F(x):
gene1 = Customer(1, 0, 0, 1)
gene2 = Customer(2, 0, 0, 1)
gene3 = Customer(3, 0, 0, 1)
gene4 = Customer(4, 0, 0, 1)
gene5 = Customer(5, 0, 0, 1)
# baseInidividu = [gene1, gene2, gene3, gene4, gene5]

population = []

population_initial = []

for i in range(N_POPULATION):
    shuffle(baseInidividu)
    newIndividu = baseInidividu.copy()
    population_initial.append(Individu(newIndividu))

population.append(population_initial)


score_by_iteration = {}
score_by_time = {}

start = process_time()

S_total = [[]]
with alive_bar(N_ITERATION, title="AG Algorithm:", ctrl_c=False, theme="smooth") as bar:
    for t in range(0, N_ITERATION):
        # Rajoute P[t+1] ; S[t+1]
        population.append([])
        S_total.append([])
        # Selection N individus de notre P(t)
        S = S_total[t]

        def by_score(individu: Individu):
            return individu.score

        population[t].sort(key=by_score)
        if ((len(population[t])//2 + len(S_total[t])) % 2 == 0):
            N_SELECTION = len(population[t])//2
        else:
            N_SELECTION = len(population[t])//2 + 1

        for i in range(N_SELECTION):
            S.append(copy(population[t][i]))

        # Group by pairs
        pairs = list(zip(S[::2], S[1::2]))
        for pair in pairs:
            if random() < P_CROSS:
                child1, child2 = cross(*pair)
                S_total[t+1].append(copy(child1))
                S_total[t+1].append(copy(child2))
            else:
                S_total[t+1].append(copy(pair[0]))
                S_total[t+1].append(copy(pair[1]))

        for individu in S_total[t+1]:
            if random() < P_MUT:
                individu.mutation()
            population[t+1].append(copy(individu))

        score_population = ScorePopulation(population[t])
        score_by_iteration[t] = score_population
        score_by_time[round(process_time() - start, 2)] = score_population

        bar()

for pop in population:
    # print("Population n:", i)
    # print(pop)
    min = 0
    min_ind = None
    for ind in pop:
        if (ind.score < min) or (not min_ind):
            min = ind.score
            min_ind = ind
    # print(min_ind.solution)
    # print()
    i += 1

mean_score_iteration = [score_by_iteration[key][0] for
                        key in score_by_iteration]
min_score_iteration = [score_by_iteration[key][1] for
                       key in score_by_iteration]

mean_score_time = [score_by_time[key][0] for
                   key in score_by_time]
min_score_time = [score_by_time[key][1] for
                  key in score_by_time]

plt.plot(mean_score_iteration, color="red", label="mean score")
plt.plot(min_score_iteration, color="green", label="min score")
plt.legend()
plt.title("Score AG par iteration")
plt.xlabel("N. Iteration")
plt.ylabel("Score")
plt.show()

plt.plot(list(score_by_time), mean_score_iteration, color="red", label="mean score")
plt.plot(list(score_by_time), min_score_iteration,  color="green", label="min score")
plt.legend()
# plt.xticks(list(score_by_time.keys()))
plt.title("Score AG par temps")
plt.xlabel("temps [s]")
plt.ylabel("Score")
plt.show()