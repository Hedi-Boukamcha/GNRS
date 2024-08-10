import json
import numpy as np
import pandas as pd
import random

from ortools.sat.python import cp_model

file_path = '1st_instance.json'

with open(file_path, 'r') as file:
    data = json.load(file)

print(data)

jobs = []
operations = []
types = []
operation_type = []
operations_by_job = []
job_types = []
modes = []
stations = []

# Initialiser des compteurs pour les opérations et les types
nombre_jobs = len(data)
nombre_operations = 0
nombre_types = 0
num_operations_by_job = 0

print(f"Number of Jobs: {len(data)}")


for job in data:
    for operation in job['operations']:
        operations.append(operation)
        nombre_operations += 1
print(f"Number of Operations: {nombre_operations}")


'''for job in data:
    for operation in job['operations']:
        types.append(operation['type'])
        nombre_types += 1
print(f"Number of Types: {nombre_types}")'''


for job in data:
    job_operations = len(job['operations'])
    operations_by_job.append(job_operations)
    num_operations_by_job += 1
print(f"Operations by Job: {num_operations_by_job}")

for job in data:
    row = []
    for operation in job['operations']:
        row.append(operation['type'])
    job_types.append(row)
print(f"Types of jobs: {job_types}")
    

jobs = [nombre_jobs]
types = [[1, 0], [0, 1]]
operations = [nombre_operations]
modes = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]     
stations = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]


#====================================================================================================================
#                                                  =*= I. Parameters =*=
#====================================================================================================================

print("\n ##__parametre 1__##")

print(f"nombre_jobs: {nombre_jobs}")
print(f"nombre_operations: {nombre_operations}")
print(f"nombre_types: {nombre_types}")


needed_proc = [[[0 for ty in range(len(types))] for o in range(operations_by_job[j])] for j in range(nombre_jobs)]

for i, job in enumerate(data):
    for j, operation in enumerate(job['operations']):
        type_value = operation['type']
        if type_value == 1:
            needed_proc[i][j][0] = [1, 0]
        elif type_value == 2:
            needed_proc[i][j][1] = [0, 1]    
for row in needed_proc:
    print(row)

print("\n ##__parametre 2__##")
lp = []
for job in data:
    lp.append(job['big'])
print(lp,"\n")


print("\n ##__parametre 3__##")
fj = [0 for p in range(nombre_jobs)]


print("\n ##__parametre 4__##")
'''
v = []
for j in range(nombre_jobs):
    v[j] = []
    for o in range(nombre_operations):
        if o != fj[j]:
            v[j][o] = operations[o]
        else:
            v[j][o] = None
'''


print("\n ##__parametre 5__##")
ddp = []
for job in data:
    ddp.append(job['due_date'])
print(ddp,"\n")


print(" \n ##__parametre 6__##")
welding_time = [[0 for o in range(operations_by_job[j])] for j in range(nombre_jobs)]
for row in welding_time: 
    print(row)


print("\n ##__parametre 7__##")
pos_j = []
for job in data:
    pos_j.append(job['pos_time'])
print(pos_j,"\n")


print("\n ##__parametre 8__##")
L = 2


print("\n ##__parametre 9__##")
M = 3


print("\n ##__parametre 10__##")
I = 0
# Calculer la borne supérieure I
for j, job in enumerate(data):
    for o, op in enumerate(job['operations']):
        welding_time_value = welding_time[j][o]
        I += (welding_time_value + pos_j[j] + 3 * M + 2 * L)

# Afficher la borne supérieure I
print(f"\n La borne supérieure I = {I}")



#====================================================================================================================
#                                    =*= III. Decision variables =*=
#====================================================================================================================


print("\n ##__Decision variable 1__##")
entry_station_date = [[0 for _ in range(len(stations))] for _ in range(nombre_jobs)]
for row in entry_station_date:
    print(row)


print("\n ##__Decision variable 2__##")
delay = [0 for _ in range(nombre_jobs)]


print("\n ##__Decision variable 3__##")
exe_start = [[0 for o in range(operations_by_job[j])] for j in range(nombre_jobs)]
for row in exe_start:
    print(row)


print("\n ##__Decision variable 4__##")
job_loaded = [[0 for _ in range(len(stations))] for _ in range(nombre_jobs)]
for j, job in enumerate(data):
    if job["big"] == 1:
        station = 1  # La station 2 pour les pièces avec "big" = 1
    else:
        # Répartition aléatoire pour les autres stations
        station = random.choice(range(len(stations)))  # Par exemple, choisir entre la station 0 et 1
    job_loaded[j][station] = 1
for row in job_loaded:
    print(row)

   
print("\n ##__Decision variable 5__##")
exe_mode = [[[0 for _ in range(len(modes))] for o in range(operations_by_job[j])] for j in range(nombre_jobs)]
for j, job in enumerate(data):
    for o, operation in enumerate(job['operations']):
        if operation['type'] == 1:
            m = random.choice(range(len(modes)))  # Mode A or B random
        elif operation['type'] == 2:
            m = 2  # Mode C
        exe_mode[j][o][m] = 1
for row in exe_mode:
    print(row)


print("\n ##__Decision variable 6__##")
exe_before = [[[[1 for o_prime in range(operations_by_job[j_prime])] for o in range(operations_by_job[j])] for j_prime in range(nombre_jobs)] for j in range(nombre_jobs)]
for row in exe_before:
    print(row)


print("\n ##__Decision variable 7__##")
exe_parallel = [[0 for o in range(operations_by_job[j])] for j in range(nombre_jobs)]
for row in exe_parallel:
    print(row)


print("\n ##__Decision variable 8__##")
job_unload = [[0 for _ in range(3)] for _ in range(nombre_jobs)]
for row in job_unload:
    print(row)


#====================================================================================================================
#                                    =*= II. Historical data for dynamic scheduling =*=
#====================================================================================================================


print("\n ##__Historical data 1__##")
# nbr of stations = 3
job_station = [[0 for _ in range(3)] for _ in range(nombre_jobs)]
for j, job in enumerate(data):
    big = job['big']
    if big == 1:
        job_station[j][1] = 1
for row in job_station:
    print(row)


print("\n ##__Historical data 2__##")
job_modeB = [0 for _ in range(nombre_jobs)]
for j in range(nombre_jobs):
    for o in range(operations_by_job[j]):
        if exe_mode[j][o][1] == 1:  # Vérifie si l'opération o du job j est en mode B
            job_modeB[j] = 1
            break
print(job_modeB,"\n")


print("\n ##__Historical data 3__##")
job_robot = [0 for _ in range(nombre_jobs)]
print(job_robot,"\n")


#====================================================================================================================
#                                    =*= IV. Objective Fonction =*=
#====================================================================================================================

print("\n ##__Objective Fonction__##")
min_Z = 0
for j, job in enumerate(data):
    min_Z += delay[j]
print(f"min Z = {min_Z}")

#====================================================================================================================
#                                    =*= IV. Constraints =*=
#====================================================================================================================

print("\n ##__Constraint 22__##")
def prec(j, j_prime, s):
    # Calculer la borne supérieure
    result_prec = I * (3 - exe_before[j][j_prime][0][0] - job_loaded[j][s] - job_loaded[j_prime][s])
    return result_prec  


print("\n ##__Constraint 23__##")
def end(j, o):
    if o == 0:
        result_end = exe_start[j][o] + welding_time[j][o] + ((pos_j[j] * exe_mode[j][o][2]) * (1 - job_modeB[j]))
    else:
        result_end = exe_start[j][o] + welding_time[j][o] + (pos_j[j] * exe_mode[j][o][2])
    return result_end


print("\n ##__Constraint 24__##")
def free(n, o, o_prime, j, j_prime):
    f = 0
    terms = []
    for q in range(nombre_jobs):
        if (q == j) and (q == j_prime):
            for x in range(num_operations_by_job):
                term1 = terms.append(needed_proc[q][x][1] * exe_before[j][q][o][x])
                term2 = terms.append(needed_proc[q][x][1] * exe_before[q][j_prime][x][o_prime])
                term3 = terms.append( (-1) * (needed_proc[q][x][1] * exe_before[j][j_prime][o][o_prime]))
    f = sum(terms)
    if n==2 :
        result_free = end(o_prime, j_prime) - I * (3 - exe_before[j][j_prime][o][o_prime] - exe_mode[j][o][2] - exe_mode[j][o_prime][3])
    else:
        result_free = end(o_prime, j_prime) - I * (4 - exe_before[j][j_prime][o][o_prime] - exe_mode[j][o][2] - exe_mode[j][o_prime][3] 
                                               - exe_parallel[j][o_prime] + f)
    return result_free


'''print("\n ##__Constraint 24__##")
def free(n, o, o_prime, j, j_prime, ty):
    f = 0
    if (j == j_prime):
        print("verify (j != j_prime)")
    else :
        for x in range(operations_by_job[j]):
            f = +needed_proc[j][o][ty] * (exe_before[o][x] + exe_before[x][o_prime] - exe_before[o][o_prime])
    if n==2 :
        result_free = end(o_prime, j_prime) - (3 - exe_before[o][o_prime] - exe_mode[j][o][2] - exe_mode[j][o_prime][3] )
    else:
        result_free = end(o_prime, j_prime) - (4 - exe_before[o][o_prime] - exe_mode[j][o][2] - exe_mode[j][o_prime][3] 
                                               - exe_parallel[j][o_prime] + f)
    return result_free'''


print("\n ##__Constraint 2__##")
def c2(j, j_prime, o, o_prime):
    for j in range(nombre_jobs):
        for j_prime in range(nombre_jobs):
            for o in range(num_operations_by_job):
                for o_prime in range(num_operations_by_job):
                    if (o != o_prime):
                        res1 = exe_before[j][j_prime][o][o_prime] + exe_before[j_prime][j][o_prime][o]
                        res1 = 1
                    else :
                        print("error c2")
    return res1


print("\n ##__Constraint 3__##")
def c3(j, o):
    for j in range(nombre_jobs):
        for o in range(num_operations_by_job):
                if (o == 0):
                    print("error c3")
                else:
                    exe_start[j][o-1] >= end(o-1) + M * (3 * exe_parallel[j][o-1] + 1) 
    return exe_start


print("\n ##__Constraint 4__##")
def c4(j, j_prime, o, o_prime):
    for j in range(nombre_jobs):
        for j_prime in range(nombre_jobs):
            for o in range(num_operations_by_job):
                for o_prime in range(num_operations_by_job):
                    if (o == o_prime):
                        print("error c4")
                    else:
                        exe_start[j][o] >= end(o_prime) + 2 * M  - I * (1 + exe_mode[j_prime][o_prime][1] - exe_before[j_prime][j][o_prime][o])
    return exe_start


print("\n ##__Constraint 5__##")
def c5(j, j_prime, o, o_prime):
    for j in range(nombre_jobs):
        for j_prime in range(nombre_jobs):
            for o in range(num_operations_by_job):
                for o_prime in range(num_operations_by_job):
                    if (o == o_prime):
                        print("error c5")
                    else:
                        exe_start[j][o] >= end(o_prime) + 2 * M  - I * (1 + exe_mode[j_prime][o_prime][2] - exe_before[j_prime][j][o_prime][o])
    return exe_start


print("\n ##__Constraint 6__##")
def c6(j, j_prime, o, o_prime):
    for j in range(nombre_jobs):
        for j_prime in range(nombre_jobs):
            for o in range(num_operations_by_job):
                for o_prime in range(num_operations_by_job):
                    if (o == o_prime):
                        print("error c6")
                    else:
                        exe_start[j][o] >= end(o_prime) + 2 * M  - I * (1 - exe_before[j_prime][j][o_prime][o] - exe_parallel[j][o])
    return exe_start


print("\n ##__Constraint 7__##")
def c7(j, j_prime, o, o_prime):
    for j in range(nombre_jobs):
        for j_prime in range(nombre_jobs):
            for o in range(num_operations_by_job):
                for o_prime in range(num_operations_by_job):
                    if (o == o_prime):
                        print("error c7")
                    else:
                        exe_start[j][o] >= exe_start[j_prime][o_prime] + (((pos_j[j] * exe_mode[j_prime][o_prime][1]) + 2 * M) * (1- job_modeB[j])) - I * (1 - exe_before[j_prime][j][o_prime][o])
    return exe_start


print("\n ##__Constraint 8__##")
def c8(j, o):
    for j in range(nombre_jobs):
        for o in range(num_operations_by_job):
                exe_parallel[j][o] >= needed_proc[j][o][1]
    return exe_parallel


print("\n ##__Constraint 9__##")
def c9(j):
    terms = []
    for j in range(nombre_jobs):
        for s in range(len(stations)):
                terms.append(entry_station_date[j][s] - M * job_station[j][s] * (1 - job_unload[j][s]) * (pos_j[j] + job_robot[j]))
                exe_start[j][0] >= sum(terms) + M
    return exe_start


print("\n ##__Constraint 10__##")
def c10(j, o, m):
    res = 1
    terms = []
    for j in range(nombre_jobs):
        for o in range(num_operations_by_job):
            for m in range(len(modes)):
                terms.append(exe_mode[j][o][m])
                res = sum(terms)
    return res


print("\n ##__Constraint 11__##")
def c11(j, o):
    for j in range(nombre_jobs):
        for o in range(num_operations_by_job):
            exe_mode[j][o][2] = needed_proc[j][i][1]
    return exe_mode


print("\n ##__Constraint 12__##")
def c12(j, o):
    res = 1
    terms = []
    for j in range(nombre_jobs):
        for s in range(len(stations)):
            terms.append(job_loaded[j][s])
            res = sum(terms)
    return res


'''
print("\n ##__Constraint 24__##") 
def free(o, o_prime, p, j_prime, q):
    for q in range(nombre_jobs):
        for x in range(nombre_jobs):
            if q != p and q != j_prime:

    if (nombre_jobs == 2) :
        result_free = end(o_prime, j_prime) - (3 - exe_before[o][o_prime] - exe_mode[p][o][2] - exe_mode[p][o_prime][3])
    else :
        result_free = end(o_prime, j_prime) - (4 - exe_before[o][o_prime] - exe_mode[p][o][2] - exe_mode[p][o_prime][3]
                                            - exe_parallel[j_prime][o_prime] )
    return result_free


#return jobs #, np.array(due_dates), np.array(pos_times), np.array(bigs), operations
if __name__ == "__main__":
    nom_fichier = '1st_instance.json'
    data = lire_fichier_json(nom_fichier)
    a = extraire_caracteristiques(data)
    # jobs, due_dates, pos_times, bigs, operations = extraire_caracteristiques(data)'''

    