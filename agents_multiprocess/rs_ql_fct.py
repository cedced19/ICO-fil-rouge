import numpy as np
import matplotlib.pyplot as plt
from time import process_time
from itertools import combinations as comb
from copy import deepcopy
from cost import *
import random
from voisinage import *
LR = .85
Y = .9
def learn_Q(Q, s, s1, action, reward):
        # Fonction pour train la matrix Q, avec les donnes de l'action que on a decide.
        Q[s, action] = Q[s, action] + LR*(reward + Y * np.max(Q[s1, :]) - Q[s, action])
def take_action(st, Q, eps, action_nb):
    # Take an action
    if random.uniform(0, 1) < eps:
        #exploration
        action = random.randint(0, action_nb-1)
    else: # Or greedy action
        #exploitation
        action = np.argmax(Q[st])
    return action 

def recuit(sol_init, matrice, w, t0, a, n_iter_cycle, capacities, max_capacity, plot = False, log = False):
    start = process_time()

    actions = [intra_route_swap, inter_route_swap,intra_route_shift,two_intra_route_swap,two_intra_route_shift,del_small_route_w_capacity,del_random_route_w_capacity]
    action_nb = len(actions)
    Q = np.random.rand(action_nb, action_nb)
    generateVoisin=intra_route_swap
    state = 0

    best_sol = sol_init
    current_sol = sol_init
    best_sol_cost = cout(best_sol, matrice, w)
    current_sol_cost = best_sol_cost
    n_iter = 0
    
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
            tmp_sol = deepcopy(current_sol)
            new_state = take_action(state, Q, 0.8, action_nb)
            generateVoisin = actions[new_state]
            #print(generateVoisin, tmp_sol, capacities, max_capacity)
            tmp_sol = generateVoisin(tmp_sol, capacities, max_capacity)
           
            state = new_state
            learn_Q(Q, state, new_state, new_state, 2)
 
            tmp = [tmp_sol]
            #tmp = relocate(current_sol, matrice, w, capacities, max_capacity) + exchange(current_sol, matrice, w, capacities, max_capacity) + exchange_inside(current_sol, matrice, w, capacities, max_capacity)
            if (len(tmp)):
                picked_sol = random.choice(tmp)
                df = cout(picked_sol, matrice, w) - cout(current_sol, matrice, w)
                if (df < 0):
                    current_sol = picked_sol
                    current_sol_cost = cout(current_sol, matrice, w)
                    new_cycle = True
                else:
                    prob = np.exp(-df/t)
                    q = random.uniform(0, 1)
                    if (q < prob):
                        current_sol = picked_sol
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
            if (log):
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
        plt.show()    

        # plot2 : coût optimal par itération  
        #plt.subplot(1,2,2)      
        plt.plot(n_iter_total_list, cout_best_iter, 'r-', label="Best_Cout")
        plt.legend()
        plt.title("Coût optimal RS par iteration")
        plt.xlabel("N. Iteration")
        plt.xticks(rotation=90, fontsize=7)
        plt.ylabel("Coût optimal")
        plt.show() 

        #plot3: temps par cout courrent
        plt.plot(temps_cout_iter, cout_iter,'g-', label="Cout-Temps")
        plt.legend()
        plt.title("Temps pour trouver la solution courante RS")
        plt.xlabel("Temps")
        plt.xticks(rotation=90, fontsize=5)
        plt.ylabel("Coût")
        plt.show() 

        #plot4: temps par cout optimal
        plt.plot(temps_cout_iter, cout_best_iter,'y-', label="Cout Optimal-Temps")
        plt.legend()
        plt.title("Temps pour trouver la solution optimale RS")
        plt.xlabel("Temps")
        plt.ylabel("Coût optimal")
        plt.xticks(rotation=90, fontsize=5)
        plt.show() 

    return best_sol, best_sol_cost