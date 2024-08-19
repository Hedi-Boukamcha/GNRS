from model import Solution, Instance
from ortools.sat.python import cp_model
import random


def init_vars(model: cp_model.CpModel, i: Instance):
    operations = [nombre_operations]
    operations_by_job = []
    job_types = []

    # Initialiser des compteurs pour les opérations et les types
    nombre_operations = 0
    num_operations_by_job = 0

    print(f"Number of Jobs: {len(i.data)}")

    for job in i.data:
        for operation in job['operations']:
            operations.append(operation)
            nombre_operations += 1
    print(f"Number of Operations: {nombre_operations}")

    operations_by_job = [len(job['operations']) for job in i.data]

    for job in i.data:
        job_operations = len(job['operations'])
        operations_by_job.append(job_operations)
        num_operations_by_job += 1
    print(f"Operations by Job: {num_operations_by_job}")

    for job in i.data:
        row = []
        for operation in job['operations']:
            row.append(operation['type'])
        job_types.append(row)
    print(f"Types of jobs: {job_types}")

    
    i.s.entry_station_date = [[model.NewIntVar(0, f'entry_station_date_{j}_{s}') for s in range(3)] for j in range(i.nb_jobs)]
    i.s.delay = [model.NewIntVar(0, f'delay_{j}') for j in range(i.nb_jobs)]  
    i.s.exe_start = [[model.NewIntVar(0, f'exe_start_{j}_{o}') for o in range(operations_by_job[j])] for j in range(i.nb_jobs)]
    i.s.job_loaded = [[model.NewBoolVar(0, f'job_loaded_{j}_{s}') for s in range(3)] for j in range(i.nb_jobs)]
    i.s.exe_mode = [[[model.NewBoolVar(0, f'exe_mode_{j}_{o}_{m}') for m in range(3)] for o in range(operations_by_job[j])] for j in range(i.nb_jobs)]
    i.s.exe_before = [[[[model.NewBoolVar(0, f'exe_before_{j}_{o}_{j_prime}_{o_prime}') for o_prime in range(operations_by_job[j_prime])] for o in range(operations_by_job[j])] for j_prime in range(i.nb_jobs)] for j in range(i.nb_jobs)]
    i.s.exe_parallel = [[model.NewBoolVar(0, f'exe_parallel_{j}_{o}') for o in range(operations_by_job[j])] for j in range(i.nb_jobs)]
    i.s.job_unload = [[model.NewBoolVar(0, f'job_unload_{j}_{s}') for s in range(3)] for j in range(i.nb_jobs)]
    
    return model, i.s


def init_objective_function(model: cp_model.CpModel, i: Instance):
    terms = []
    for j in range(i.nb_jobs):        
        terms.append(i.s.delay[j])
    model.Minimize(sum(terms))
    return model, i.s


def prec(i: Instance, j: int, j_prime: int, s: int):
    for j in range(i.nb_jobs):
        for j_prime in range(i.nb_jobs):
            for s in range(i.nb_stations):
                result_prec = i.I * (3 - i.s.exe_before[j][j_prime][0][0] - i.s.job_loaded[j][s] - i.s.job_loaded[j_prime][s])
    return result_prec 


def end(i: Instance, j: int, o: int):
    for j in range(i.nb_jobs):
        for o in range(i.operations_by_job[j]):
            if (o == 0):
                result_end = i.s.exe_start[j][o] + i.welding_time[j][o] + ((i.pos_j[j] * i.s.exe_mode[j][o][2]) * (1 - i.job_modeB[j]))
            else:
                result_end = i.s.exe_start[j][o] + i.welding_time[j][o] + (i.pos_j[j] * i.s.exe_mode[j][o][2])
    return result_end


def free(i: Instance, j: int, j_prime: int, o: int, o_prime: int):
    f = 0
    terms = []
    for q in range(i.nb_jobs):
        if (q == j) and (q == j_prime):
            for x in range(i.operations_by_job[j]):
                term1 = i.needed_proc[q][x][0]
                term2 = i.s.exe_before[j][q][o][x] + i.s.exe_before[q][j_prime][x][o_prime] - i.s.exe_before[j][j_prime][o][o_prime]
                terms.append(term1 * term2)
            if (i.nb_jobs == 2) :
                result_free = end(i, j_prime, o_prime) - i.I * (3 - i.s.exe_before[j][j_prime][o][o_prime] - i.s.exe_mode[j][o][1] - i.s.exe_mode[j][o_prime][2])
            else:
                result_free = end(i, j_prime, o_prime) - i.I * (4 - i.s.exe_before[j][j_prime][o][o_prime] - i.s.exe_mode[j][o][1] - i.s.exe_mode[j][o_prime][2] 
                                                - i.s.exe_parallel[j][o_prime] + sum(terms))
    return result_free
