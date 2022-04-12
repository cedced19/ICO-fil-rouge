import random

def intra_route_swap(s):
    i=random.randint(0,len(s)-1)
    j=random.randint(0,len(s[i])-1)
    k=j
    while k==j:
        k=random.randint(0,len(s[i])-1)
    a=s[i][j]
    s[i][j]=s[i][k]
    s[i][k]=a
    return()

def inter_route_swap(s):
    i=random.randint(0,len(s)-1)
    j=i
    while j==i:
        j=random.randint(0,len(s)-1)
    k=random.randint(0,len(s[i]-1))
    h=random.randint(0,len(s[j]-1))
    a=s[i][k]
    s[i][k]=s[j][h]
    s[j][h]=a
    return()

def intra_route_shift(s):
    i=random.randint(0,len(s)-1)
    j=random.randint(0,len(s[i])-1)
    k=j
    while k==j:
        k=random.randint(0,len(s[i])-1)
    a=s[i][j]
    del s[i][j]
    s[i].insert(k,a)
    return()

def inter_route_shift(s):
    i=random.randint(0,len(s)-1)
    h=i
    while h==i:
        h=random.randint(0,len(s)-1)
    j=random.randint(0,len(s[i])-1)
    k=random.randint(0,len(s[h])-1)
    a=s[i][j]
    del s[i][j]
    s[h].insert(k,a)
    return()

def two_intra_route_swap(s):
    i=random.randint(0,len(s)-1)
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
    return()

def two_intra_route_shift(s):
    i=random.randint(0,len(s)-1)
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
    return()

def del_small_route(s):
    l=len(s)
    size_route=[]
    for i in range(l):
        size_route.append(len(s[i]))
    id_min = size_route.index(min(size_route))
    del s[id_min]
    return()

def del_random_route(s):
    l=len(s)
    i=random.randint(0,l-1)
    del s[i]
    return()
