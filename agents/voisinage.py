import random

def intra_route_swap(s):
    i=random.randint(0,len(s)-1)
    if len(s[i])<=2:
         return(s)
    else:
        j=random.randint(0,len(s[i])-1)
        k=j
        while k==j:
            k=random.randint(0,len(s[i])-1)
        a=s[i][j]
        s[i][j]=s[i][k]
        s[i][k]=a
        return(s)

def inter_route_swap(s):
    i=random.randint(0,len(s)-1)
    if len(s)<=2:
             return(s)
    else:
             j=i
        while j==i:
            j=random.randint(0,len(s)-1)
        k=random.randint(0,len(s[i]-1))
        h=random.randint(0,len(s[j]-1))
        a=s[i][k]
        s[i][k]=s[j][h]
        s[j][h]=a
        return(s)

def intra_route_shift(s):
    i=random.randint(0,len(s)-1)
    if len(s[i])<=2:
         return(s)
    else:
        j=random.randint(0,len(s[i])-1)
        k=j
        while k==j:
            k=random.randint(0,len(s[i])-1)
        a=s[i][j]
        del s[i][j]
        s[i].insert(k,a)
        return(s)

def inter_route_shift(s):
    i=random.randint(0,len(s)-1)
    if len(s)<=2:
             return(s)
    else:
        h=i
        while h==i:
            h=random.randint(0,len(s)-1)
        j=random.randint(0,len(s[i])-1)
        k=random.randint(0,len(s[h])-1)
        a=s[i][j]
        del s[i][j]
        s[h].insert(k,a)
        return(s)

def two_intra_route_swap(s):
    i=random.randint(0,len(s)-1)
    if len(s[i])<=2:
         return(s)
    else:
        j=random.randint(0,len(s[i])-2)
        k=j
        while k==j:
            k=random.randint(0,len(s[i])-2)
        a=s[i][j]
        b=s[i][j+1]
        s[i][j]=s[i][k]
        s[i][j+1]=s[i][k+1]
        s[i][k]=a
        s[i][k+1]=b
        return(s)

def two_intra_route_shift(s):
    i=random.randint(0,len(s)-1)
    if len(s[i])<=2:
         return(s)
    else:
        j=random.randint(0,len(s[i])-2)
        k=j
        while k==j:
            k=random.randint(0,len(s[i])-2)
        a=s[i][j]
        b=s[i][j+1]
        del s[i][j]
        del s[i][j+1]
        s[i].insert(k,a)
        s[i].insert(k+1,b)
        return(s)

def del_small_route(s):
    l=len(s)
    size_route=[]
    for i in range(l):
        size_route.append(len(s[i]))
    id_min = size_route.index(min(size_route))
    a = s[id_min]
    del s[id_min]
    r = random.randint(0,l-2)
    s[r]=s[r]+a
    return(s)

def del_random_route(s):
    l=len(s)
    i=random.randint(0,l-1)
    a = s[i]
    del s[i]
    r = random.randint(0,l-2)
    s[r]=s[r]+a
    return(s)

def test_capacity(s,capa,max_capa):
    res_test = True
    l = len(s)
    for i in range(l):
        ll = len(s[i])
        sum_capa = 0
        for j in range(ll):
            sum_capa = sum_capa + capa[j+1]
        if sum_capa > max_capa[0]:
            res_test = False
    return(res_test)

def del_small_route_w_capacity(s,capa,max_capa):
    l = len(s)
    S = copy.deepcopy(s)
    vois = del_small_route(s)
    for i in range(l):
        if test_capacity(vois):
            return(vois)
        else:
            s = S
            vois = del_small_route(s)
    return(s)


def del_random_route_w_capacity(s,capa,max_capa):
    l = len(s)
    S = copy.deepcopy(s)
    vois = del_random_route(s)
    for i in range(l):
        if test_capacity(vois):
            return(vois)
        else:
            s = S
            vois = del_random_route(s)
    return(s)


def inter_route_swap_w_capacity(s,capa,max_capa):
    l = len(s)
    S = copy.deepcopy(s)
    vois = inter_route_swap(s)
    for i in range(l):
        if test_capacity(vois):
            return(vois)
        else:
            s = S
            vois = inter_route_swap(s)
    return(s)


def inter_route_shift_w_capacity(s,capa,max_capa):
    l = len(s)
    S = copy.deepcopy(s)
    vois = inter_route_shift(s)
    for i in range(l):
        if test_capacity(vois):
            return(vois)
        else:
            s = S
            vois = inter_route_shift(s)
    return(s)
