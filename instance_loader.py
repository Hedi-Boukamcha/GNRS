import json
import numpy as np
import pandas as pd
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

# Initialiser des compteurs pour les opérations et les types
nombre_jobs = len(data)
nombre_operations = 0
nombre_types = 0
num_operations_by_job = 0
s = 3


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


for i, job in enumerate(data):
    job_operations = len(job['operations'])
    operations_by_job.append(job_operations)
    num_operations_by_job += 1
print(f"Operations by Job: {len(operations_by_job)}")


for i, job in enumerate(data):
    row = []
    for operation in job['operations']:
        row.append(operation['type'])
    job_types.append(row)
print(f"Types of jobs: {job_types}")
    

jobs = [nombre_jobs]
types = [2]
operations = [nombre_operations]
          

#====================================================================================================================
#                                                  =*= I. Parameters =*=
#====================================================================================================================

print("\n ##__parametre 1__##")

print(f"nombre_jobs: {nombre_jobs}")
print(f"nombre_operations: {nombre_operations}")
print(f"nombre_types: {nombre_types}")


needed_proc = [[[0 for _ in range(2)] for p in range(operations_by_job[p])] for p in range(nombre_jobs)]
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
welding_time = [[0 for p in range(operations_by_job[p])] for p in range(nombre_jobs)]
for row in welding_time: 
    print(row)


print("\n ##__parametre 7__##")
pos_p = []
for job in data:
    pos_p.append(job['pos_time'])
print(pos_p,"\n")


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
        I += (welding_time_value + pos_p[j] + 3 * M + 2 * L)

# Afficher la borne supérieure I
print(f"\n La borne supérieure I est : {I}")


#====================================================================================================================
#                                    =*= II. Historical data for dynamic scheduling =*=
#====================================================================================================================


print("\n ##__Historical data 1__##")
# nbr of stations = 3
job_station = [[0 for _ in range(3)] for _ in range(nombre_jobs)]
for job_index, (job_key, job_data) in enumerate(data.items()):
    big = job_data[0]['big']
    if big == 1:
        job_station[job_index][1] = 1

for row in job_station:
    print(row)


print("\n ##__Historical data 2__##")
job_modeB = [0 for _ in range(nombre_jobs)]
print(job_modeB,"\n")


print("\n ##__Historical data 3__##")
job_robot = [0 for _ in range(nombre_jobs)]
print(job_robot,"\n")


#====================================================================================================================
#                                    =*= III. Decision variables =*=
#====================================================================================================================


print("\n ##__Decision variable 1__##")
entry_station_date = [[0 for _ in range(3)] for _ in range(nombre_jobs)]
for row in entry_station_date:
    print(row)


print("\n ##__Decision variable 2__##")
delay = [0 for _ in range(nombre_jobs)]


print("\n ##__Decision variable 3__##")
exe_start = [[0 for p in range(operations_by_job[p])] for p in range(nombre_jobs)]
for row in exe_start:
    print(row)


print("\n ##__Decision variable 4__##")
job_loaded = [[0 for _ in range(3)] for _ in range(nombre_jobs)]
for row in job_loaded:
    print(row)


print("\n ##__Decision variable 5__##")
exe_mode = [[[0 for _ in range(nombre_operations)] for _ in range(3)] for _ in range(nombre_jobs)]
for row in exe_mode:
    print(row)


print("\n ##__Decision variable 6__##")
exe_before = [[0 for o_prime in range(nombre_operations)] for o in range(nombre_operations) for j in range(nombre_jobs)]
for row in exe_before:
    print(row)


print("\n ##__Decision variable 7__##")
exe_parallel = [[0 for j in range(operations_by_job[j])] for j in range(nombre_jobs)]
for row in exe_parallel:
    print(row)


print("\n ##__Decision variable 8__##")
job_unload = [[0 for _ in range(3)] for _ in range(nombre_jobs)]
for row in job_unload:
    print(row)


#====================================================================================================================
#                                    =*= IV. Objective Fonction =*=
#====================================================================================================================

print("\n ##__Objective Fonction__##")
min_Z = 0
for job_index, (job_key, job_data) in enumerate(data.items()):
    #delay = job_data[0]['delay']
    min_Z += delay[job_index]


#====================================================================================================================
#                                    =*= IV. Constraints =*=
#====================================================================================================================

print("\n ##__Constraint 22__##")
def prec(o, o_prime, p, p_prime, s):
    # Calculer la borne supérieure
    result_prec = 3 - exe_before[o][o_prime] - job_loaded[p][s] - job_loaded[p_prime][s]
    return result_prec  

print("\n ##__Constraint 23__##")
def end(o, p):
    result_end = exe_start[p][o] + welding_time[p][o] + (pos_p[p] * exe_mode[p][o][2])
    return result_end

print("\n ##__Constraint 24__##")
def free(n, o, o_prime, p, p_prime, ty):
    f = 0
    for q in jobs :
        if (q == p) and (q == p_prime):
            print("verify (q == p) or (q == p_prime)")
        else :
            for x in operations:
                f = +a[p][o][ty] * (exe_before[o][x] + exe_before[x][o_prime] - exe_before[o][o_prime])
    if n==2 :
        result_free = end(o_prime, p_prime) - (3 - exe_before[o][o_prime] - exe_mode[p][o][2] - exe_mode[p][o_prime][3] )
    else:
        result_free = end(o_prime, p_prime) - (4 - exe_before[o][o_prime] - exe_mode[p][o][2] - exe_mode[p][o_prime][3] 
                                               - exe_parallel[p][o_prime] + f)
    return result_free

print("\n ##__Constraint 2__##")
def c1(j, o, o_prime):
    for j in jobs:
        for o , o_prime in operations:
            if (o != o_prime):
                res1 = exe_before[j][o][o_prime] + exe_before[j][o_prime][o]
                res1 = 1
            else :
                print("error c2")
    return res1


print("\n ##__Constraint 3__##")
def c2(o, o_prime):

    return 


'''
print("\n ##__Constraint 24__##") 
def free(o, o_prime, p, p_prime, q):
    for q in range(nombre_jobs):
        for x in range(nombre_jobs):
            if q != p and q != p_prime:

    if (nombre_jobs == 2) :
        result_free = end(o_prime, p_prime) - (3 - exe_before[o][o_prime] - exe_mode[p][o][2] - exe_mode[p][o_prime][3])
    else :
        result_free = end(o_prime, p_prime) - (4 - exe_before[o][o_prime] - exe_mode[p][o][2] - exe_mode[p][o_prime][3]
                                            - exe_parallel[p_prime][o_prime] )
    return result_free






#return jobs #, np.array(due_dates), np.array(pos_times), np.array(bigs), operations
if __name__ == "__main__":
    nom_fichier = '1st_instance.json'
    data = lire_fichier_json(nom_fichier)
    a = extraire_caracteristiques(data)
    # jobs, due_dates, pos_times, bigs, operations = extraire_caracteristiques(data)'''

    