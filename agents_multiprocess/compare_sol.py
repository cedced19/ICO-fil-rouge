def index_list(a,s):
    index1=0
    for i in range(len(s)):
        index2=0
        for j in range(len(s[i])):
            if a==s[i][j]:
                return(index1,index2)
            else:
                index2=index2+1
        index1=index1+1
    return(index1,index2)
    
def add_1_couple(a):
    return((a[0],a[1]+1))

def compare_sol(s1,s2):
    nb_sol_diff = 0;
    for ss1 in s1: 
        for i in range(len(ss1)-1):
            if add_1_couple(index_list(ss1[i],s2))!=index_list(ss1[i+1],s2):
                nb_sol_diff=nb_sol_diff+1
    return(nb_sol_diff)


if __name__ == '__main__':
    print(compare_sol([[1, 2, 9], [5, 8], [3, 6, 7, 4]], [[1, 2, 9], [5, 6, 7, 8], [3, 4]]))
