from model import Solution, Instance
from ortools.sat.python import cp_model
import random


def init_vars(model: cp_model.CpModel, i: Instance):
    jobs = [nombre_jobs]
    operations = [nombre_operations]
    operations_by_job = []
    job_types = []

    # Initialiser des compteurs pour les opérations et les types
    nombre_jobs = len(i.data)
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

    
    i.s.entry_station_date = [[model.NewIntVar(0, f'entry_station_date_{j}_{s}') for s in range(3)] for j in range(nombre_jobs)]
    i.s.delay = [model.NewIntVar(0, f'delay_{j}') for j in range(nombre_jobs)]  
    i.s.exe_start = [[model.NewIntVar(0, f'exe_start_{j}_{o}') for o in range(operations_by_job[j])] for j in range(nombre_jobs)]
    i.s.job_loaded = [[model.NewBoolVar(0, f'job_loaded_{j}_{s}') for s in range(3)] for j in range(nombre_jobs)]
    i.s.exe_mode = [[[model.NewBoolVar(0, f'exe_mode_{j}_{o}_{m}') for m in range(3)] for o in range(operations_by_job[j])] for j in range(nombre_jobs)]
    i.s.exe_before = [[[[model.NewBoolVar(0, f'exe_before_{j}_{o}_{j_prime}_{o_prime}') for o_prime in range(operations_by_job[j_prime])] for o in range(operations_by_job[j])] for j_prime in range(nombre_jobs)] for j in range(nombre_jobs)]
    i.s.exe_parallel = [[model.NewBoolVar(0, f'exe_parallel_{j}_{o}') for o in range(operations_by_job[j])] for j in range(nombre_jobs)]
    i.s.job_unload = [[model.NewBoolVar(0, f'job_unload_{j}_{s}') for s in range(3)] for j in range(nombre_jobs)]
    
    return model, i.s


def init_objective_function(model: cp_model.CpModel, i: Instance):
    terms = []
    nombre_jobs = len(i.data)
    for j in range(nombre_jobs):        
        terms.append(i.s.delay[j])
    model.Minimize(sum(terms))
    return model, i.s


def prec(model: cp_model.CpModel, i: Instance):
    i.I = 0
    result_prec = model.NewIntVar(0, 'result_prec')
    for j in range(i.nombre_jobs):
        for j_prime in range(i.nombre_jobs):
            for s in range(len(i.stations)):
                term1 = s.exe_before[j][j_prime][0][0]
                term2 = s.job_loaded[j][s]
                term3 = s.job_loaded[j_prime][s]

                model.Add(result_prec == i.I * (3 - term1 - term2 - term3))
    return model, i.s 


def end(model: cp_model.CpModel, i: Instance):
    for j in range(i.nombre_jobs):
        for o in range(i.operations_by_job[j]):
            if (o == 0):
                end_value = model.NewIntVar(0, f'end_{j}_{o}')
                model.Add(end_value == i.s.exe_start[j][o] + i.welding_time[j][o] + ((i.pos_j[j] * i.s.exe_mode[j][o][2]) * (1 - i.job_modeB[j])))
            else:
                end_value = model.NewIntVar(0, 10000, f'end_{j}_{o}')
                model.Add(end_value == i.s.exe_start[j][o] + i.welding_time[j][o] + (i.pos_j[j] * i.s.exe_mode[j][o][2]))
    return model, i.s


def free(model: cp_model.CpModel, i: Instance, s: Solution, I, o, o_prime, j, j_prime):
    f = 0
    terms = []
    for q in range(i.nombre_jobs):
        if (q == j) and (q == j_prime):
            for x in range(i.operations_by_job[j]):
                term1 = terms.append(i.needed_proc[q][x][1] * s.exe_before[j][q][o][x])
                term2 = terms.append(i.needed_proc[q][x][1] * s.exe_before[q][j_prime][x][o_prime])
                term3 = terms.append( (-1) * (i.needed_proc[q][x][1] * s.exe_before[j][j_prime][o][o_prime]))
                f = terms.append(term1 + term2 + term3)
    if (i.nombre_jobs == 2) :
        result_free = end(o_prime, j_prime) - I * (3 - s.exe_before[j][j_prime][o][o_prime] - s.exe_mode[j][o][2] - s.exe_mode[j][o_prime][3])
    else:
        result_free = end(o_prime, j_prime) - I * (4 - s.exe_before[j][j_prime][o][o_prime] - s.exe_mode[j][o][2] - s.exe_mode[j][o_prime][3] 
                                               - s.exe_parallel[j][o_prime] + f)
    return result_free
