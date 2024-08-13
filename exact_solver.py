from model import Solution, Instance
from ortools.sat.python import cp_model
import random


def init_vars(data, model: cp_model.CpModel, i: Instance):
    s = Solution()
    jobs = [nombre_jobs]
    operations = [nombre_operations]
    types = []
    operation_type = []
    operations_by_job = []
    job_types = []
    types = [[1, 0], [0, 1]]
    modes = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    stations = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    # Initialiser des compteurs pour les opérations et les types
    nombre_jobs = len(data)
    nombre_operations = 0
    num_operations_by_job = 0

    print(f"Number of Jobs: {len(data)}")

    for job in data:
        for operation in job['operations']:
            operations.append(operation)
            nombre_operations += 1
    print(f"Number of Operations: {nombre_operations}")

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

    
    s.entry_station_date = [[model.NewIntVar(0, f'entry_station_date_{j}_{s}') for s in range(len(stations))] for j in range(nombre_jobs)]
    s.delay = [model.NewIntVar(0, f'delay_{j}') for j in range(nombre_jobs)]  
    s.exe_start = [[model.NewIntVar(0, f'exe_start_{j}_{o}') for o in range(operations_by_job[j])] for j in range(nombre_jobs)]
    s.job_loaded = [[model.NewBoolVar(f'job_loaded_{j}_{s}') for s in range(len(stations))] for j in range(nombre_jobs)]
    s.exe_mode = [[[model.NewBoolVar(f'exe_mode_{j}_{o}_{m}') for m in range(len(modes))] for o in range(operations_by_job[j])] for j in range(nombre_jobs)]
    s.exe_before = [[[[model.NewBoolVar(f'exe_before_{j}_{o}_{j_prime}_{o_prime}') for o_prime in range(operations_by_job[j_prime])] for o in range(operations_by_job[j])] for j_prime in range(nombre_jobs)] for j in range(nombre_jobs)]
    s.exe_parallel = [[model.NewBoolVar(f'exe_parallel_{j}_{o}') for o in range(operations_by_job[j])] for j in range(nombre_jobs)]
    s.job_unload = [[model.NewBoolVar(f'job_unload_{j}_{s}') for s in range(len(stations))] for j in range(nombre_jobs)]
    

    return model, s


   
    
