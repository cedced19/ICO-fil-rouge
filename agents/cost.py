def cout(solution, matrice, w):
    '''
    Calculate cout d'une solution
    '''
    K = len(solution) - solution.count([])
    sum_cost = 0
    for route in solution:
        if (len(route)):
            sum_cost += matrice[0, route[0]] # ajouter 0->premier el
            sum_cost += matrice[0, route[-1]] # ajouter dernier el->0
            for i in range(len(route)-1):
                sum_cost += matrice[route[i], route[i+1]]
    return K*w + sum_cost