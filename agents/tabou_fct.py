import numpy as np
import matplotlib.pyplot as plt
from time import process_time
from itertools import combinations as comb
from copy import deepcopy
from cost import *

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

def acceptable(tabu_list, movement, current_sol_cost, prev_sol_cost, aspiration):
    if (not (current_sol_cost - prev_sol_cost <= aspiration)):
        return False
    else:
        if (movement in tabu_list):
            return False
        else:
            # tabu_list.append(movement) # mettre un max ?
            return True

def exchange(solution, matrice, w, tabu_list, aspiration, capacities, max_capacity):
    neighbours = []
    sol_cost = cout(solution, matrice, w)
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
                movement = (0, ind0, ind1, i, j)
                if(acceptable(tabu_list, movement, tmp_cost, sol_cost, aspiration)):
                    neighbours.append((tmp_sol,movement,tmp_cost))
                    
    neighbours = filter_on_capacities(neighbours, capacities, max_capacity)

    return neighbours

def flatten(t):
    return [item for sublist in t for item in sublist]
def relocate(solution, matrice, w, tabu_list, aspiration, capacities, max_capacity):
    neighbours = []
    sol_cost = cout(solution, matrice, w)
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
                # on définit un mouvement, par ("type d'opération", "index de la première paire", "index de la seconde paire", "index de là où on place le nouvel élément", "élément deplacé")
                movement = (1, ind0, ind1, 0, j)
                if(acceptable(tabu_list, movement, tmp_cost, sol_cost, aspiration)):
                    neighbours.append((tmp_sol, movement,tmp_cost))

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
                # on définit un mouvement, par ("type d'opération", "index de la première paire", "index de la seconde paire", "index de là où on place le nouvel élément", "élément deplacé")
                movement = (1, ind0, ind1, pair0.index(i) + 1, j)
                if(acceptable(tabu_list, movement, tmp_cost, sol_cost, aspiration)):
                    neighbours.append((tmp_sol, movement,tmp_cost))

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
                # on définit un mouvement, par ("type d'opération", "index de la première paire", "index de la seconde paire", "index de là où on place le nouvel élément", "élément deplacé")
                movement = (1, ind0, ind1, 0, j)
                if(acceptable(tabu_list, movement, tmp_cost, sol_cost, aspiration)):
                    neighbours.append((tmp_sol, movement, tmp_cost))

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
                # on définit un mouvement, par ("type d'opération", "index de la première paire", "index de la seconde paire", "index de là où on place le nouvel élément", "élément deplacé")
                movement = (1, ind0, ind1, pair0.index(i) + 1, j)
                if(acceptable(tabu_list, movement, tmp_cost, sol_cost, aspiration)):
                    neighbours.append((tmp_sol, movement, tmp_cost))
        # à ce stade, il y a des redondances du fait des listes vides, mais vu qu'on va utiliser le max, on va avoir qu'une solution unique même si elles ont le mêmes coûts

    if (len(neighbours)):
        neighbours = filter_on_capacities(neighbours, capacities, max_capacity)

        # il serait peut-être nécessaire d'ajouter une fonction pour pouvoir explorer la solution avec ajout d'un camion à coup sûr
        if (len(neighbours) == 0):
            tmp_sol = deepcopy(solution)
            tmp_sol.append([])
            neighbours = relocate(tmp_sol, matrice, w, tabu_list, aspiration, capacities, max_capacity)
    
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
def exchange_inside(solution, matrice, w, tabu_list, aspiration, capacities, max_capacity):
    if (not capacity_compatible(solution, capacities, max_capacity)):
        return []
    neighbours = []
    sol_cost = cout(solution, matrice, w)
    
    for i in range(len(solution)):
        route = solution[i]
        for permut in all_permutations(route):
            tmp_sol = deepcopy(solution)
            tmp_sol[i] = permut
            tmp_cost = cout(tmp_sol, matrice, w)
            movement = (2, i, permut)
            if(acceptable(tabu_list, movement, tmp_cost, sol_cost, aspiration)):
                    neighbours.append((tmp_sol, movement,tmp_cost))
        # à ce stade, il y a des redondances du fait des listes vides, mais vu qu'on va utiliser le max, on va avoir qu'une solution unique même si elles ont le mêmes coûts
    neighbours = filter_on_capacities(neighbours, capacities, max_capacity)
    
    return neighbours

def tabu(sol_init, matrice, n_max, aspiration, w, capacities, max_capacity, plot = False):
    start = process_time()
    cout_iter = []
    cout_best_iter = []
    temps_cout_iter = []
    nb_iter = []
    best_sol = sol_init
    current_sol = sol_init
    n_iter = 0
    tabu_list = []
    best_iter = 0
    while((cout(current_sol, matrice, w) >= cout(best_sol, matrice, w)) and (n_iter - best_iter < n_max)):
        n_iter+=1
        tmp = relocate(current_sol, matrice, w, tabu_list, aspiration, capacities, max_capacity) + exchange(current_sol, matrice, w, tabu_list, aspiration, capacities, max_capacity) + exchange_inside(current_sol, matrice, w, tabu_list, aspiration, capacities, max_capacity)
        tmp.sort(key=lambda x: x[2])
        if (len(tmp)):
            current_sol = tmp[0][0]
            temps_cout_iter.append(process_time() - start)
            tabu_list.append(tmp[0][1])
            #print(tmp[0][2])
        else:
            break
        if (cout(current_sol, matrice, w) < cout(best_sol, matrice, w)):
            best_sol = current_sol
            best_iter = n_iter

        cout_iter.append(cout(current_sol, matrice, w))
        nb_iter.append(n_iter)
        cout_best_iter.append(cout(best_sol, matrice, w))
    
    if (plot):
        # plot1 : coût par itération 
        #plt.subplot(1,2,1)
        plt.plot(nb_iter, cout_iter, 'b-', label="Cout_iter")
        plt.legend()
        plt.title("Coût TB par iteration")
        plt.xlabel("N. Iteration", fontsize=10)
        plt.xticks(rotation=90, fontsize=7)
        plt.ylabel("Coût")
        # x_major_locator = MultipleLocator(5)
        # y_major_locator = MultipleLocator(5)
        # ax = plt.gca()
        # ax.xaxis.set_major_locator(x_major_locator)
        # ax.yaxis.set_major_locator(y_major_locator)
        plt.show()    

        # plot2 : coût optimal par itération  
        #plt.subplot(1,2,2)      
        plt.plot(nb_iter, cout_best_iter, 'r-', label="Best_Cout")
        plt.legend()
        plt.title("Coût optimal TB par iteration")
        plt.xlabel("N. Iteration")
        plt.xticks(rotation=90, fontsize=7)
        plt.ylabel("Coût optimal")
        # x_major_locator = MultipleLocator(5)
        # y_major_locator = MultipleLocator(2)
        # ax = plt.gca()
        # ax.xaxis.set_major_locator(x_major_locator)
        # ax.yaxis.set_major_locator(y_major_locator)
        plt.show() 

        #plot3: temps par cout courrent
        plt.plot(temps_cout_iter, cout_iter,'g-', label="Cout-Temps")
        plt.legend()
        plt.title("Temps pour trouver la solution courante TB")
        plt.xlabel("Temps")
        plt.xticks(rotation=90, fontsize=7)
        plt.ylabel("Coût")
        # x_major_locator = MultipleLocator(0.005)
        # y_major_locator = MultipleLocator(5)
        # ax = plt.gca()
        # ax.xaxis.set_major_locator(x_major_locator)
        # ax.yaxis.set_major_locator(y_major_locator)
        plt.show() 

        #plot4: temps par cout optimal
        plt.plot(temps_cout_iter, cout_best_iter,'y-', label="Cout Optimal-Temps")
        plt.legend()
        plt.title("Temps pour trouver la solution optimale TB")
        plt.xlabel("Temps")
        plt.ylabel("Coût optimal")
        plt.xticks(rotation=90, fontsize=7)
        # x_major_locator = MultipleLocator(0.005)
        # y_major_locator = MultipleLocator(2)
        # ax = plt.gca()
        # ax.xaxis.set_major_locator(x_major_locator)
        # ax.yaxis.set_major_locator(y_major_locator)   
        plt.show() 

    return best_sol, cout(best_sol, matrice, w)