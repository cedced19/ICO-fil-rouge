from classes import Customer, Solution
from random import shuffle, choice, random, randint
import numpy as np


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

    def __repr__(self) -> str:
        return f"Individu{'{'}Chromosome: {self.chromosome} Score: {str(self.score)}{'}'}\n"


def cross(individu1: Individu, individu2: Individu):

    #Trouver trois clients aléatoire qui ne sont pas identiques
    chif1 = randint(0,len(individu1.chromosome)-1)
    chif2 = chif1
    while(chif2 == chif1):
        chif2 = randint(0,len(individu1.chromosome)-1)
    chif3 = chif1
    while(chif3==chif1)or(chif3==chif2) :
        chif3 = randint(0,len(individu1.chromosome)-1)
    # print(individu1.chromosome, chif1)
    # Trouver leurs emplacements dans les solutions

    # print(individu1.chromosome, individu2.chromosome, chif1, chif2, chif3)

    def find_customer(individu, id):
        return [individu.chromosome.index(customer) for customer in individu.chromosome if customer.id == id][0]

    place1 = [find_customer(individu1, chif1), find_customer(individu1, chif2), find_customer(individu1, chif3)]
    place2 = [find_customer(individu2, chif1), find_customer(individu2, chif2), find_customer(individu2, chif3)]
    # place1 = [individu1.chromosome.index(chif1),individu1.chromosome.index(chif2),individu1.chromosome.index(chif3)]
    # place2 = [individu2.chromosome.index(chif1),individu2.chromosome.index(chif2),individu2.chromosome.index(chif3)]

    # Ordonner leur emplacements
    place1.sort()
    place2.sort()

    # print(place1, place2)

    individu1_copy = individu1.chromosome.copy()
    individu2_copy = individu2.chromosome.copy()
    # Remplacer le premier client concerné de solution 1 avec ce de solution 2
    # Remplacer le premier client concerné de solution 2 avec ce de solution 1
    for i in range(3):
        individu1_copy[place1[i]] = individu2.chromosome[place2[i]]
        individu2_copy[place2[i]] = individu1.chromosome[place1[i]]

    individu1.chromosome = individu1_copy
    individu2.chromosome = individu2_copy

    # print(individu1.chromosome, individu2.chromosome)

    #Renvoyer la solution 1
    return individu1, individu2


P_MUT = 0.5
P_CROSS = 0.5
N_ITERATION = 20
N_POPULATION = 10
N_SELECTION = (N_POPULATION-8)
COST_MATRIX = \
    np.matrix([[0, 14, 18, 9, 5, 7],
              [14, 0, 12, 4, 17, 1],
              [18, 12, 0, 3, 2, 1],
              [9, 4, 3, 0, 4, 8],
              [5, 17, 2, 4, 0, 11],
              [7, 1, 1, 8, 11, 0]])

gene1 = Customer(0, 0, 0, 10)
gene2 = Customer(1, 0, 0, 100)
gene3 = Customer(2, 0, 0, 25)
gene4 = Customer(3, 0, 0, 15)
gene5 = Customer(4, 0, 0, 5)

population = []

from big_example import matrice_example2

# Création de population et evaluer F(x):
# baseInidividu = [gene1, gene2, gene3, gene4, gene5]
COST_MATRIX = matrice_example2
baseInidividu = [Customer(i, 0, 0, randint(10, 99)) for i in range(52)]
population_initial = []
n_error = 0
for i in range(N_POPULATION):
    shuffle(baseInidividu)
    if (baseInidividu) not in population:
        newIndividu = baseInidividu.copy()
        population_initial.append(Individu(newIndividu))
    else:
        n_error += 1
    if n_error > 10:
        break
population.append(population_initial)

S_total = [[]]
for t in range(0, N_ITERATION):
    # Rajoute P[t+1] ; S[t+1]
    population.append([])
    S_total.append([])
    # Selection N individus de notre P(t)
    S = S_total[t]

    def by_score(individu: Individu):
        return individu.score

    population[t].sort(key=by_score)
    # print(population[t])

    if ((len(population[t])//2 + len(S_total)) & 2 == 0):
        N_SELECTION = len(population[t])//2
    else:
        N_SELECTION = len(population[t])//2 + 1
    N_SELECTION = len(population[t])//2

    for i in range(N_SELECTION):
        S.append(population[t][i])

    # Group by pairs
    pairs = list(zip(S[::2], S[1::2]))
    for pair in pairs:
        if random() < P_CROSS:
            child1, child2 = cross(*pair)
            S_total[t+1].append(child1)
            S_total[t+1].append(child2)
        else:
            S_total[t+1].append(pair[0])
            S_total[t+1].append(pair[1])

    for individu in S_total[t+1]:
        if random() < P_MUT:
            individu.mutation()
        population[t+1].append(individu)

i = 0
for pop in population:
    print("Population n:", i)
    # print(pop)
    min = 0
    min_ind = None
    for ind in pop:
        if (ind.score < min) or (not min_ind):
            min = ind.score
            min_ind = ind
    print(min_ind.solution)
    print()
    i += 1

# print(pop)