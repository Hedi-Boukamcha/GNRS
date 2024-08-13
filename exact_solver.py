from model import Solution, Instance
from ortools.sat.python import cp_model
import random


def initialize_data(data):
    jobs = []
    operations = []
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

    jobs = [nombre_jobs]
    operations = [nombre_operations]
    

    return {
        'jobs': jobs,
        'operations': operations,
        'types': types,
        'operation_type': operation_type,
        'operations_by_job': operations_by_job,
        'job_types': job_types,
        'modes': modes,
        'stations': stations
    }
