import json
import numpy as np
import pandas as pd
from ortools.sat.python import cp_model


def lire_fichier_json(nom_fichier):
    data = pd.read_json(nom_fichier)
    return data

nom_fichier = '1st_instance.json'
data = lire_fichier_json(nom_fichier)
jobs = list(data.keys())
print(jobs)
nombre_jobs = len(jobs)
print(nombre_jobs)

# job = data['jobs']
# print(job)

# Initialiser des compteurs pour les opérations et les types
nombre_jobs = 0
nombre_operations = 0
nombre_types = 0
s = 3

operations =[nombre_operations]
jobs = [nombre_jobs]
types = [nombre_types]

jobs_set = set()
operations_set = set()
types_set = set()

#====================================================================================================================
#                                                  =*= I. Parameters =*=
#====================================================================================================================


print("\n ##__parametre 1__##")

# Collecter les valeurs uniques de jobs, opérations et types
for job_key, job_data in data.items():
    jobs_set.add(job_key)  # Ajouter le nom du job à l'ensemble des jobs
    for operation in job_data[0]["operations"]:
        nombre_operations += 1
        operations_set.add(operation["type"])  # Ajouter le type de l'opération à l'ensemble des opérations
        types_set.add(operation["type"])  # Ajouter le type de l'opération à l'ensemble des types

# Déterminer les nombres à partir des ensembles
nombre_jobs = len(jobs_set)
nombre_types = len(types_set)

print(f"nombre_jobs: {nombre_jobs}")
print(f"nombre_operations: {nombre_operations}")
print(f"nombre_types: {nombre_types}")

operations_by_job = []
for job_key, job_data in data.items():
    nombre_operations = len(job_data[0]['operations'])
    operations_by_job.append(nombre_operations)
print(operations_by_job)


a = [[[0 for _ in range(nombre_types)] for p in range(operations_by_job[p])] for p in range(nombre_jobs)]
for row in a:
    print(row)

print("\n ##__parametre 2__##")

lp = [0] * nombre_jobs

for job_index, (job, operations) in enumerate(data.items()):
    lp[job_index] = operations[0]['big']

print(lp,"\n")


print("\n ##__parametre 5__##")

ddp = [0] * nombre_jobs

for job_index, (job, operations) in enumerate(data.items()):
    ddp[job_index] = operations[0]['due_date']

print(ddp,"\n")


print(" \n ##__parametre 6__##")

welding_time = [[0 for p in range(operations_by_job[p])] for p in range(nombre_jobs)]
for row in welding_time: 
    print(row)

print("\n ##__parametre 7__##")

pos_p = [0] * nombre_jobs
for job_index, (job, operations) in enumerate(data.items()):
    pos_p[job_index] = operations[0]['pos_time']

print(pos_p,"\n")


print("\n ##__parametre 8__##")
L = 0 


print("\n ##__parametre 9__##")
M = 0


print("\n ##__parametre 10__##")
I = 0
# Calculer la borne supérieure I
for job_index, (job_key, job_data) in enumerate(data.items()):
    for operation_index, operation in enumerate(job_data[0]['operations']):
        pos_p = job_data[0]['pos_time']
        welding_time_value = welding_time[job_index][operation_index]
        I += (welding_time_value + pos_p + 3 * M + 2 * L)

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
delay = 0


print("\n ##__Decision variable 3__##")
exe_start = [[0 for p in range(operations_by_job[p])] for p in range(nombre_jobs)]
for row in exe_start:
    print(row)


print("\n ##__Decision variable 4__##")
job_loaded = [[0 for _ in range(3)] for _ in range(nombre_jobs)]
for row in job_loaded:
    print(row)


print("\n ##__Decision variable 5__##")
exe_mode = [[[0 for _ in range(nombre_operations)] for _ in range(nombre_jobs)] for _ in range(3)]
for row in exe_mode:
    print(row)


print("\n ##__Decision variable 6__##")
exe_before = [[0 for _ in range(nombre_operations)] for _ in range(nombre_operations)]
for row in exe_before:
    print(row)


print("\n ##__Decision variable 7__##")
exe_parallel = [[0 for _ in range(nombre_operations) for _ in range(nombre_jobs)]]
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
    min_Z += delay


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

print("\n ##__Constraint 1__##")
def c1(o, o_prime):
    for o , o_prime in operations:
        if (o == o_prime):
            res1 = exe_before[o][o_prime] + exe_before[o_prime][o]
            res1 = 1
        else :
            print("error c1")
    return res1



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

    