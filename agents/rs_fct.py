import numpy as np
import matplotlib.pyplot as plt
from time import process_time
from itertools import combinations as comb
from copy import deepcopy
from cost import *
import random

def compute_capacities(route, capacities):
    total = 0
    for i in route:
        total += capacities[i-1]
    return total

def capacity_compatible(sol, capacities, max_capacity):
    for route in sol:
        total = compute_capacities(route, capacities)
        if (total > max_capacity):
            return False
    return True


def filter_on_capacities(solutions, capacities, max_capacity):
    result = []
    for sol in solutions:
        if (capacity_compatible(sol[0], capacities, max_capacity)):
            result.append(sol)
    return result

def exchange(solution, matrice, w, capacities, max_capacity):
    neighbours = []
    for combinaison in list(comb(enumerate(solution), 2)):
        for i in combinaison[0][1]:
            for j in combinaison[1][1]:
                tmp_sol = deepcopy(solution)
                pair0 = deepcopy(combinaison[0][1])     
                pair1 = deepcopy(combinaison[1][1])
                ind0 = combinaison[0][0]
                ind1 = combinaison[1][0]
                pair0.insert(pair0.index(i), j)
                pair0.remove(i)
                pair1.insert(pair1.index(j), i)
                pair1.remove(j)
                tmp_sol[ind0] = pair0
                tmp_sol[ind1] = pair1
                tmp_cost = cout(tmp_sol, matrice, w)
                # on définit un mouvement, par ("type d'opération", "index de la première paire", "index de la seconde paire", "élément 1 changé", "élément 2 changé")
                neighbours.append((tmp_sol,tmp_cost))
    neighbours = filter_on_capacities(neighbours, capacities, max_capacity)
    return neighbours

def relocate(solution, matrice, w, capacities, max_capacity):
    neighbours = []
    combinations = list(comb(enumerate(solution), 2))
    

    for combinaison in combinations: 
        if (len(combinaison[0][1]) == 0):
            for j in combinaison[1][1]:
                tmp_sol = deepcopy(solution)
                pair0 = deepcopy(combinaison[0][1])     
                pair1 = deepcopy(combinaison[1][1])
                ind0 = combinaison[0][0]
                ind1 = combinaison[1][0]
                pair1.remove(j)
                pair0.insert(0, j)
                tmp_sol[ind0] = pair0
                tmp_sol[ind1] = pair1
                tmp_cost = cout(tmp_sol, matrice, w)
                neighbours.append((tmp_sol,tmp_cost))

        for i in combinaison[0][1]:
            for j in combinaison[1][1]:
                tmp_sol = deepcopy(solution)
                pair0 = deepcopy(combinaison[0][1])     
                pair1 = deepcopy(combinaison[1][1])
                ind0 = combinaison[0][0]
                ind1 = combinaison[1][0]
                pair1.remove(j)
                pair0.insert(pair0.index(i) + 1, j)
                tmp_sol[ind0] = pair0
                tmp_sol[ind1] = pair1
                tmp_cost = cout(tmp_sol, matrice, w)
                neighbours.append((tmp_sol,tmp_cost))

        if (len(combinaison[1][1]) == 0):
            for j in combinaison[0][1]:
                tmp_sol = deepcopy(solution)
                pair0 = deepcopy(combinaison[1][1])     
                pair1 = deepcopy(combinaison[0][1])
                ind0 = combinaison[1][0]
                ind1 = combinaison[0][0]
                pair1.remove(j)
                pair0.insert(0, j)
                tmp_sol[ind0] = pair0
                tmp_sol[ind1] = pair1
                tmp_cost = cout(tmp_sol, matrice, w)
                neighbours.append((tmp_sol,tmp_cost))

        for i in combinaison[1][1]:
            for j in combinaison[0][1]:
                tmp_sol = deepcopy(solution)
                pair0 = deepcopy(combinaison[1][1])     
                pair1 = deepcopy(combinaison[0][1])
                ind0 = combinaison[1][0]
                ind1 = combinaison[0][0]
                pair1.remove(j)
                pair0.insert(pair0.index(i) + 1, j)
                tmp_sol[ind0] = pair0
                tmp_sol[ind1] = pair1
                tmp_cost = cout(tmp_sol, matrice, w)
                neighbours.append((tmp_sol,tmp_cost))
        # à ce stade, il y a des redondances du fait des listes vides, mais vu qu'on va utiliser le max, on va avoir qu'une solution unique même si elles ont le mêmes coûts
    if (len(neighbours)):
        neighbours = filter_on_capacities(neighbours, capacities, max_capacity)

        # il serait peut-être nécessaire d'ajouter une fonction pour pouvoir explorer la solution avec ajout d'un camion à coup sûr
        if (len(neighbours) == 0):
            tmp_sol = deepcopy(solution)
            tmp_sol.append([])
            neighbours = relocate(tmp_sol, matrice, w, capacities, max_capacity)
    return neighbours

def inverser(L,i,j):
    M=L.copy()
    M[j], M[i] = L[i], L[j]
    return M
def all_permutations(L):
    V=[]
    l=len(L)
    for i in range (l):
        for j in range (i+1,l):
            V.append(inverser(L,i,j))
    return(V)
def exchange_inside(solution, matrice, w, capacities, max_capacity):
    neighbours = []
    
    for i in range(len(solution)):
        route = solution[i]
        for permut in all_permutations(route):
            tmp_sol = deepcopy(solution)
            tmp_sol[i] = permut
            tmp_cost = cout(tmp_sol, matrice, w)
            neighbours.append((tmp_sol,tmp_cost))
        # à ce stade, il y a des redondances du fait des listes vides, mais vu qu'on va utiliser le max, on va avoir qu'une solution unique même si elles ont le mêmes coûts
    neighbours = filter_on_capacities(neighbours, capacities, max_capacity)
    return neighbours

def recuit(sol_init, matrice, w, t0, a, n_iter_cycle, capacities, max_capacity, plot = False):
    best_sol = sol_init
    current_sol = sol_init
    best_sol_cost = cout(best_sol, matrice, w)
    current_sol_cost = best_sol_cost
    n_iter = 0
    start = process_time()
    cout_iter = []
    cout_best_iter = []
    temps_cout_iter = []
    n_iter_total = 0
    n_iter_total_list = []
    new_cycle = True
    t=t0
    while(new_cycle):
        n_iter=0
        new_cycle = False
        while n_iter < n_iter_cycle:
            n_iter_total += 1
            n_iter_total_list.append(n_iter_total)
            n_iter += 1
            tmp = relocate(current_sol, matrice, w, capacities, max_capacity) + exchange(current_sol, matrice, w, capacities, max_capacity) + exchange_inside(current_sol, matrice, w, capacities, max_capacity)
            if (len(tmp)):
                picked_sol = random.choice(tmp)
                df = cout(picked_sol[0], matrice, w) - cout(current_sol, matrice, w)
                if (df < 0):
                    current_sol = picked_sol[0]
                    current_sol_cost = cout(current_sol, matrice, w)
                    new_cycle = True
                else:
                    prob = np.exp(-df/t)
                    q = random.uniform(0, 1)
                    if (q < prob):
                        current_sol = picked_sol[0]
                        current_sol_cost = cout(current_sol, matrice, w)
                        new_cycle = True
                temps_cout_iter.append(process_time() - start)        
                if (current_sol_cost < best_sol_cost):
                    best_sol = current_sol
                    best_sol_cost = current_sol_cost

                cout_iter.append(cout(current_sol, matrice, w))
                cout_best_iter.append(cout(best_sol, matrice, w))    
          
            else:
                break
            print("best: {}, current: {}".format(best_sol_cost,current_sol_cost))
            
        t = a * t
    if (plot):
        # plot1 : coût par itération 
        #plt.subplot(1,2,1)
        plt.plot(n_iter_total_list, cout_iter, 'b-', label="Cout_iter")
        plt.legend()
        plt.title("Coût RS par iteration")
        plt.xlabel("N. Iteration", fontsize=10)
        plt.xticks(rotation=90, fontsize=7)
        plt.ylabel("Coût")
        # x_major_locator = MultipleLocator(100)
        # y_major_locator = MultipleLocator(20)
        # ax = plt.gca()
        # ax.xaxis.set_major_locator(x_major_locator)
        # ax.yaxis.set_major_locator(y_major_locator)
        plt.show()    

        # plot2 : coût optimal par itération  
        #plt.subplot(1,2,2)      
        plt.plot(n_iter_total_list, cout_best_iter, 'r-', label="Best_Cout")
        plt.legend()
        plt.title("Coût optimal RS par iteration")
        plt.xlabel("N. Iteration")
        plt.xticks(rotation=90, fontsize=7)
        plt.ylabel("Coût optimal")
        # x_major_locator = MultipleLocator(100)
        # y_major_locator = MultipleLocator(5)
        # ax = plt.gca()
        # ax.xaxis.set_major_locator(x_major_locator)
        # ax.yaxis.set_major_locator(y_major_locator)
        plt.show() 

        #plot3: temps par cout courrent
        plt.plot(temps_cout_iter, cout_iter,'g-', label="Cout-Temps")
        plt.legend()
        plt.title("Temps pour trouver la solution courante RS")
        plt.xlabel("Temps")
        plt.xticks(rotation=90, fontsize=5)
        plt.ylabel("Coût")
        # x_major_locator = MultipleLocator(0.05)
        # y_major_locator = MultipleLocator(10)
        # ax = plt.gca()
        # ax.xaxis.set_major_locator(x_major_locator)
        # ax.yaxis.set_major_locator(y_major_locator)
        plt.show() 

        #plot4: temps par cout optimal
        plt.plot(temps_cout_iter, cout_best_iter,'y-', label="Cout Optimal-Temps")
        plt.legend()
        plt.title("Temps pour trouver la solution optimale RS")
        plt.xlabel("Temps")
        plt.ylabel("Coût optimal")
        plt.xticks(rotation=90, fontsize=5)
        # x_major_locator = MultipleLocator(0.05)
        # y_major_locator = MultipleLocator(5)
        # ax = plt.gca()
        # ax.xaxis.set_major_locator(x_major_locator)
        # ax.yaxis.set_major_locator(y_major_locator)
        # plt.figure(figsize=(15,15))    
        plt.show() 

    return best_sol, best_sol_cost