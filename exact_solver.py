from model import Instance
from ortools.sat.python import cp_model
import random
import json



def init_vars(model: cp_model.CpModel, i: Instance):
    operations = []
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

    
    i.s.entry_station_date = [[model.NewIntVar(0,  1000, f'entry_station_date_{j}_{c}') for c in range(3)] for j in range(i.nb_jobs)]
    i.s.delay = [model.NewIntVar(0,   1000, f'delay_{j}') for j in range(i.nb_jobs)]  
    i.s.exe_start = [[model.NewIntVar(0,   1000, f'exe_start_{j}_{o}') for o in range(operations_by_job[j])] for j in range(i.nb_jobs)]
    i.s.job_loaded = [[model.NewBoolVar(f'job_loaded_{j}_{c}') for c in range(3)] for j in range(i.nb_jobs)]
    i.s.exe_mode = [[[model.NewBoolVar(f'exe_mode_{j}_{o}_{m}') for m in range(3)] for o in range(operations_by_job[j])] for j in range(i.nb_jobs)]
    i.s.exe_before = [[[[model.NewBoolVar(f'exe_before_{j}_{o}_{j_prime}_{o_prime}') for o_prime in range(operations_by_job[j_prime])] for o in range(operations_by_job[j])] for j_prime in range(i.nb_jobs)] for j in range(i.nb_jobs)]
    i.s.exe_parallel = [[model.NewBoolVar(f'exe_parallel_{j}_{o}') for o in range(operations_by_job[j])] for j in range(i.nb_jobs)]
    i.s.job_unload = [[model.NewBoolVar(f'job_unload_{j}_{c}') for c in range(3)] for j in range(i.nb_jobs)]
    
    return model, i.s


def init_objective_function(model: cp_model.CpModel, i: Instance):
    terms = []
    for j in range(i.nb_jobs):        
        terms.append(i.s.delay[j])
    model.Minimize(sum(terms))
    return model, i.s


def prec(i: Instance, j: int, j_prime: int, c: int): #c = station
    for j in range(i.nb_jobs):
        for j_prime in range(i.nb_jobs):
            for s in range(i.nb_stations):
                result_prec = i.I * (3 - i.s.exe_before[j][j_prime][0][0] - i.s.job_loaded[j][c] - i.s.job_loaded[j_prime][c])
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
    terms = []
    result_free = 0
    j = 0 
    j_prime = 0 
    for o in range(i.operations_by_job[j]):
        for o_prime in range(i.operations_by_job[j_prime]):
            for q in range(i.nb_jobs):
                if (q != j) and (q != j_prime):
                    for x in range(i.operations_by_job[q]):
                        print("q", q)
                        print("x", x)
                        print("a",i.needed_proc[q][x][0])
                        term1 = i.needed_proc[q][x][0]
                        term2 = (i.s.exe_before[j][q][o][x] + i.s.exe_before[q][j_prime][x][o_prime]) - i.s.exe_before[j][j_prime][o][o_prime]
                        print(term1)
                        print(term2)
                        terms.append(term1 * term2)
                    if (i.nb_jobs == 2):
                        result_free = end(i, j_prime, o_prime) - i.I * (3 - i.s.exe_before[j][j_prime][o][o_prime] - i.s.exe_mode[j][o][1] - i.s.exe_mode[j][o_prime][2])
                    else:
                        result_free = end(i, j_prime, o_prime) - i.I * (4 - i.s.exe_before[j][j_prime][o][o_prime] - i.s.exe_mode[j][o][1] - i.s.exe_mode[j][o_prime][2] 
                                                        - i.s.exe_parallel[j][o_prime] + sum(terms))
    return result_free


def c2(model: cp_model.CpModel, i: Instance):
    for j in range(i.nb_jobs):
        for j_prime in range(i.nb_jobs):
            for o in range(i.operations_by_job[j]):
                for o_prime in range(i.operations_by_job[j_prime]):
                    if (o != o_prime):
                        model.Add(1 == i.s.exe_before[j][j_prime][o][o_prime] + i.s.exe_before[j_prime][j][o_prime][o])
                    else :
                        print("error c2")
    return model, i.s


def c3(model: cp_model.CpModel, i: Instance):
    for j in range(i.nb_jobs):
        for o in range(1, i.operations_by_job[j]):
            model.Add(i.s.exe_start[j][o-1] >= end(i, j, o-1) + i.M * (3 * i.s.exe_parallel[j][o-1] + 1)) 
    return model, i.s


def c4(model: cp_model.CpModel, i: Instance):
    for j in range(i.nb_jobs):
        for j_prime in range(i.nb_jobs):
            for o in range(i.operations_by_job[j]):
                for o_prime in range(i.operations_by_job[j_prime]):
                    if (o == o_prime):
                        print("error c4")
                    else:
                        model.Add(i.s.exe_start[j][o] >= end(i, j, o_prime) + 2 * i.M  - i.I * (1 + i.s.exe_mode[j_prime][o_prime][1] - i.s.exe_before[j_prime][j][o_prime][o]))
    return model, i.s


def c5(model: cp_model.CpModel, i: Instance):
    for j in range(i.nb_jobs):
        for j_prime in range(i.nb_jobs):
            for o in range(i.operations_by_job[j]):
                for o_prime in range(i.operations_by_job[j_prime]):
                    if (o == o_prime):
                        print("error c5")
                    else:
                        model.Add(i.s.exe_start[j][o] >= end(i, j, o_prime) + 2 * i.M  - i.I * (1 + i.s.exe_mode[j_prime][o_prime][2] - i.s.exe_before[j_prime][j][o_prime][o]))
    return model, i.s


def c6(model: cp_model.CpModel, i: Instance):
    for j in range(i.nb_jobs):
        for j_prime in range(i.nb_jobs):
            for o in range(i.operations_by_job[j]):
                for o_prime in range(i.operations_by_job[j_prime]):
                    if (o == o_prime):
                        print("error c6")
                    else:
                        model.Add(i.s.exe_start[j][o] >= end(i, j, o_prime) + 2 * i.M  - i.I * (1 - i.s.exe_before[j_prime][j][o_prime][o] - i.s.exe_parallel[j][o]))
    return model, i.s


def c7(model: cp_model.CpModel, i: Instance):
    for j in range(i.nb_jobs):
        for j_prime in range(i.nb_jobs):
            for o in range(i.operations_by_job[j]):
                for o_prime in range(i.operations_by_job[j_prime]):
                    if (o == o_prime):
                        print("error c7")
                    else:
                        model.Add(i.s.exe_start[j][o] >= i.s.exe_start[j_prime][o_prime] + (((i.pos_j[j] * i.s.exe_mode[j_prime][o_prime][1]) + 2 * i.M) * (1- i.job_modeB[j])) - i.I * (1 - i.s.exe_before[j_prime][j][o_prime][o]))
    return model, i.s


def c8(model: cp_model.CpModel, i: Instance):
    for j in range(i.nb_jobs):
        for o in range(i.operations_by_job[j]):
            model.Add(i.s.exe_parallel[j][o] >= i.needed_proc[j][o][1])
    return model, i.s


def c9(model: cp_model.CpModel, i: Instance):
    terms = []
    for j in range(i.nb_jobs):
        for c in range(i.nb_stations):
                terms.append(i.s.entry_station_date[j][c] - i.M * i.job_station[j][c] * (1 - i.s.job_unload[j][c]) * (i.pos_j[j] + i.job_robot[j]))
                model.Add(i.s.exe_start[j][0] >= sum(terms) + i.M)
    return model, i.s


def c10(model: cp_model.CpModel, i: Instance):
    terms = []
    for j in range(i.nb_jobs):
        for o in range(i.operations_by_job[j]):
            for m in range(i.nb_modes):
                terms.append(i.s.exe_mode[j][o][m])
                model.Add(1 == sum(terms))
    return model, i.s


def c11(model: cp_model.CpModel, i: Instance):
    for j in range(i.nb_jobs):
        for o in range(i.operations_by_job[j]):
            model.Add(i.s.exe_mode[j][o][2] == i.needed_proc[j][o][1])
    return model, i.s


def c12(model: cp_model.CpModel, i: Instance):
    terms = []
    for j in range(i.nb_jobs):
        for c in range(i.nb_stations):
            terms.append(i.s.job_loaded[j][c])
            model.Add(1 == sum(terms))
    return model, i.s


def c13(model: cp_model.CpModel, i: Instance):
    for j in range(i.nb_jobs):
        model.Add(i.s.job_loaded[j][2] >= i.lp[j])
    return model, i.s


def c14(model: cp_model.CpModel, i: Instance):
    for j in range(i.nb_jobs):
        for o in range(i.operations_by_job[j]):
            model.Add(i.s.delay[j] >= end(i, j, o) + i.L + i.M - i.due_date[j])
    return model, i.s


def c15(model: cp_model.CpModel, i: Instance):
    for j in range(i.nb_jobs):
        for j_prime in range(1, i.nb_jobs):
            for o in range(i.operations_by_job[j]):
                for o_prime in range(i.operations_by_job[j_prime]):
                    if (j == j_prime):
                        print("!!! Error c15 !!!")
                    else:
                        model.Add(i.s.delay[j] >= free(i, o, o_prime, j, j_prime) + i.L + 3 * i.M - i.due_date[j])
                        print("c15", i.s.delay[j])
    return model, i.s


def c16(model: cp_model.CpModel, i: Instance):
    for j in range(i.nb_jobs):
        for j_prime in range(1, i.nb_jobs):
            if (j == j_prime):
                print("!!! Error c16 !!!")
            else:
                for o_prime in range(i.operations_by_job[j_prime]):
                    for c in range(i.nb_stations):
                            model.Add(i.s.entry_station_date[j][c] >= end(i, j_prime, o_prime) - prec(i, j_prime, j, c) + 2 * i.L + i.M)
                            print("c16", i.s.entry_station_date[j][c])
    return model, i.s


def c17(model: cp_model.CpModel, i: Instance):
    for j in range(i.nb_jobs):
        for j_prime in range(i.nb_jobs):
            for j_second in range(i.nb_jobs):
                for o_prime in range(i.operations_by_job[j_prime]):
                    for o_second in range(i.operations_by_job[j_second]):
                        for c in range(i.nb_stations):
                            if (j != j_prime) and (j != j_second) and (j_second != j_prime):
                                model.Add(i.s.entry_station_date[j][c] >= free(i, o_prime, o_second, j_prime, j_second) - prec(i, j_prime, j, c) + 2 * i.L + 3 * i.M)
                            else:
                                print("!!! Error c17 !!!")
    return model, i.s


def c18(model: cp_model.CpModel, i: Instance):
    terms = []
    f1 = 0
    for j in range(i.nb_jobs):
        for c in range(i.nb_stations):
            term1 = terms.append(i.s.job_unload[j][c] * i.M * i.job_robot[j])
            term2 = terms.append(i.s.job_unload[j][c] * 3 * i.M * i.job_modeB[j])
            term3 = terms.append(i.s.job_unload[j][c] * 2 * i.L)
            f1 = terms.append(i.s.job_unload[j][c] * i.M * i.job_robot[j] + 
                              i.s.job_unload[j][c] * 3 * i.M * i.job_modeB[j] + 
                              i.s.job_unload[j][c] * 2 * i.L)
            for s_prime in range(i.nb_stations):
                model.Add(i.s.entry_station_date[j][c] >= sum(terms) + ( - i.I + i.I * i.s.job_loaded[j][s_prime]))
    return model, i.s


def c19(model: cp_model.CpModel, i: Instance):
    terms = []
    for j in range(i.nb_jobs):
        for j_prime in range(i.nb_jobs):
            for c in range(i.nb_stations):
                if (j == j_prime):
                    print("!!! Error c19 !!!")
                else:
                    term1 = terms.append(i.job_station[j][c] * 2 * i.L)
                    term2 = terms.append(i.job_station[j][c] * i.M * i.job_robot[j_prime])
                    term3 = terms.append(i.job_station[j][c] * 3 * i.M * i.job_modeB[j])
                    term4 = terms.append(- i.I + i.I * i.s.job_loaded[j][c])
                    model.Add(i.s.entry_station_date[j][c] >= i.job_station[j][c] * 2 * i.L
                            + i.job_station[j][c] * i.M * i.job_robot[j_prime]
                            + i.job_station[j][c] * 3 * i.M * i.job_modeB[j]
                            + (- i.I + i.I * i.s.job_loaded[j][c]))
    return model, i.s


def c20(model: cp_model.CpModel, i: Instance):
    terms = []
    for j in range(i.nb_jobs):
        for j_prime in range(i.nb_jobs):
            for c in range(i.nb_stations):
                if (j == j_prime):
                    print("!!! Error c20 !!!")
                else:
                    term1 = terms.append(i.s.entry_station_date[j][c] + i.I * 3)
                    term2 = terms.append(i.s.entry_station_date[j][c] - i.I * i.job_station[j][c])
                    term3 = terms.append(i.s.entry_station_date[j][c] - i.I * i.s.exe_before[j_prime][j][0][0])
                    term4 = terms.append(i.s.entry_station_date[j][c] - i.I * i.s.job_loaded[j_prime][c])
                    model.Add(1 >= i.s.entry_station_date[j][c] + i.s.entry_station_date[j][c] + i.I * 3
                               + i.s.entry_station_date[j][c] - i.I * i.job_station[j][c]
                                 + i.s.entry_station_date[j][c] - i.I * i.s.exe_before[j_prime][j][0][0]
                                   + i.s.entry_station_date[j][c] - i.I * i.s.job_loaded[j_prime][c])
    return model, i.s


def c21(model: cp_model.CpModel, i: Instance):
    terms = []
    for j in range(i.nb_jobs):
        for o in range(i.operations_by_job[j]):
            for c in range(i.nb_stations):
                term1 = terms.append(i.s.job_unload[j][c] * i.M * i.job_robot[j])
                term2 = terms.append(i.s.job_unload[j][c] * 2 * i.M * i.job_modeB[j])
                model.Add(i.s.exe_start[j][o] >= sum(terms))
    return model, i.s


def solver(instance_file):
    with open(instance_file, 'r') as file:
        data = json.load(file)

    i = Instance(data)

    model = cp_model.CpModel()

    solver = cp_model.CpSolver()

    init_vars(model, i)
    init_objective_function(model, i)

    for constraint in [c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15,c16,c17,c18,c19,c20,c21]:
        model, i.s = constraint(model, i)

    '''c2(model, i)
    c3(model, i)
    c4(model, i)
    c5(model, i)
    c6(model, i)
    c7(model, i)
    c8(model, i)
    c9(model, i)
    c10(model, i)
    c11(model, i)
    c12(model, i)
    c13(model, i)
    c14(model, i)
    c15(model, i)
    c16(model, i)
    c17(model, i)
    c18(model, i)
    c19(model, i)
    c20(model, i)
    c21(model, i)'''


    status = solver.Solve(model)

    if status == cp_model.OPTIMAL:
        print("Solution optimale trouvée!")
        # Vous pouvez extraire et retourner les valeurs des variables ici
        result_values = {
            'variable_1': solver.Value(i.s.entry_station_date),
            'variable_2': solver.Value(i.s.delay),
            'variable_3': solver.Value(i.s.exe_start),
            'variable_4': solver.Value(i.s.job_loaded),
            'variable_5': solver.Value(i.s.exe_mode),
            'variable_6': solver.Value(i.s.exe_before),
            'variable_7': solver.Value(i.s.exe_parallel),
            'variable_8': solver.Value(i.s.job_unload),
            'fn obj': solver.Value(init_objective_function)
        }
    else:
        print(f"Pas de solution optimale trouvée. Statut: {status}")
        result_values = None
    
    # Retourner le statut, les valeurs des variables, le modèle et le solveur
    return status, result_values, model, solver

if __name__ == "__main__":
    status, result_values, model, solver = solver('1st_instance.json')
