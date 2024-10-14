from model import Instance, MathInstance, FIRST_OP, PROCEDE_1_SEQ_MODE_A, PROCEDE_1_PARALLEL_MODE_B, PROCEDE_2_MODE_C, STATION_1, STATION_2, STATION_3, PROCEDE_1, PROCEDE_2
from ortools.sat.python import cp_model
import random
import json

STATUS_MEANING = ["UNKNOWN", "MODEL_INVALID", "FEASIBLE", "INFEASIBLE", "OPTIMAL"]

def init_vars(model: cp_model.CpModel, i: MathInstance):
    i.s.entry_station_date = [[model.NewIntVar(0, i.I, f'entry_station_date_{j}_{c}') for c in i.loop_stations()] for j in i.loop_jobs()]
    i.s.delay = [model.NewIntVar(0, i.I, f'delay_{j}') for j in i.loop_jobs()]  
    i.s.exe_start = [[model.NewIntVar(0, i.I, f'exe_start_{j}_{o}') for o in i.loop_operations(j)] for j in i.loop_jobs()]
    i.s.job_loaded = [[model.NewBoolVar(f'job_loaded_{j}_{c}') for c in i.loop_stations()] for j in i.loop_jobs()]
    i.s.exe_mode = [[[model.NewBoolVar(f'exe_mode_{j}_{o}_{m}') for m in i.loop_modes()] for o in i.loop_operations(j)] for j in i.loop_jobs()]
    i.s.exe_before = [[[[model.NewBoolVar(f'exe_before_{j}_{j_prime}_{o}_{o_prime}') for o_prime in i.loop_operations(j_prime)] for o in i.loop_operations(j)] for j_prime in i.loop_jobs()] for j in i.loop_jobs()]
    i.s.exe_parallel = [[model.NewBoolVar(f'exe_parallel_{j}_{o}') for o in i.loop_operations(j)] for j in i.loop_jobs()]
    i.s.job_unload = [[model.NewBoolVar(f'job_unload_{j}_{c}') for c in i.loop_stations()] for j in i.loop_jobs()]
    return model, i.s

def is_same(j: int, j_prime: int, o: int, o_prime):
    return (j == j_prime) and (o == o_prime)

def init_objective_function(model: cp_model.CpModel, i: MathInstance):
    terms = []
    for j in i.loop_jobs():        
        terms.append(i.s.delay[j])
    model.Minimize(sum(terms))
    return model, i.s

def prec(i: MathInstance, j: int, j_prime: int, c: int): #c = station
    return i.I * (3 - i.s.exe_before[j][j_prime][FIRST_OP][FIRST_OP] - i.s.job_loaded[j][c] - i.s.job_loaded[j_prime][c])

def end(i: MathInstance, j: int, o: int):
    if (o == 0):
        return i.s.exe_start[j][o] + i.welding_time[j][o] + ((i.pos_j[j] * i.s.exe_mode[j][o][PROCEDE_1_PARALLEL_MODE_B]) * (1 - i.job_modeB[j]))
    else:
        return i.s.exe_start[j][o] + i.welding_time[j][o] + (i.pos_j[j] * i.s.exe_mode[j][o][PROCEDE_1_PARALLEL_MODE_B])

def free(i: MathInstance, j: int, j_prime: int, o: int, o_prime: int):
    if (i.nb_jobs == 2):
        return end(i, j_prime, o_prime) - i.I * (3 - i.s.exe_before[j][j_prime][o][o_prime] - i.s.exe_mode[j][o][PROCEDE_1_PARALLEL_MODE_B] - i.s.exe_mode[j][o_prime][PROCEDE_2_MODE_C])
    else:
        terms = []
        for q in i.loop_jobs():
            if (q != j) and (q != j_prime):
                for x in range(i.operations_by_job[q]):
                    term = (i.s.exe_before[j][q][o][x] + i.s.exe_before[q][j_prime][x][o_prime]) - i.s.exe_before[j][j_prime][o][o_prime]
                    terms.append(i.needed_proc[q][x][PROCEDE_1] * term)
        return end(i, j_prime, o_prime) - i.I * (4 - i.s.exe_before[j][j_prime][o][o_prime] - i.s.exe_mode[j][o][PROCEDE_1_PARALLEL_MODE_B] - i.s.exe_mode[j_prime][o_prime][PROCEDE_2_MODE_C] 
                                     - i.s.exe_parallel[j_prime][o_prime] + sum(terms))

def c2(model: cp_model.CpModel, i: MathInstance):
    for j in i.loop_jobs():
        for j_prime in i.loop_jobs():
            for o in i.loop_operations(j):
                for o_prime in i.loop_operations(j_prime):
                    if not is_same(j, j_prime, o, o_prime):
                        model.Add(1 == i.s.exe_before[j][j_prime][o][o_prime] + i.s.exe_before[j_prime][j][o_prime][o])
    return model, i.s

def c3(model: cp_model.CpModel, i: MathInstance):
    for j in i.loop_jobs():
        for o in i.loop_operations(j, exclude_first=True):
            model.Add(i.s.exe_start[j][o] >= end(i, j, o-1) + i.M * (3 * i.s.exe_parallel[j][o-1] + 1)) 
    return model, i.s

def c4(model: cp_model.CpModel, i: MathInstance):
    for j in i.loop_jobs():
        for j_prime in i.loop_jobs():
            for o in i.loop_operations(j):
                for o_prime in i.loop_operations(j_prime):
                    if not is_same(j, j_prime, o, o_prime):
                        model.Add(i.s.exe_start[j][o] >= end(i, j_prime, o_prime) + 2*i.M  - i.I*(1 + i.s.exe_mode[j_prime][o_prime][PROCEDE_1_PARALLEL_MODE_B] - i.s.exe_before[j_prime][j][o_prime][o]))
    return model, i.s

def c5(model: cp_model.CpModel, i: MathInstance):
    for j in i.loop_jobs():
        for j_prime in i.loop_jobs():
            for o in i.loop_operations(j):
                for o_prime in i.loop_operations(j_prime):
                    if not is_same(j, j_prime, o, o_prime):
                        model.Add(i.s.exe_start[j][o] >= end(i, j_prime, o_prime) + 2 * i.M  - i.I * (1 + i.s.exe_mode[j][o][PROCEDE_2_MODE_C] - i.s.exe_before[j_prime][j][o_prime][o]))
    return model, i.s

def c6(model: cp_model.CpModel, i: MathInstance):
    for j in i.loop_jobs():
        for j_prime in i.loop_jobs():
            for o in i.loop_operations(j):
                for o_prime in i.loop_operations(j_prime):
                    if not is_same(j, j_prime, o, o_prime):
                        model.Add(i.s.exe_start[j][o] >= end(i, j_prime, o_prime) + 2 * i.M  - i.I * (1 - i.s.exe_before[j_prime][j][o_prime][o] - i.s.exe_parallel[j][o]))
    return model, i.s

def c7(model: cp_model.CpModel, i: MathInstance):
    for j in i.loop_jobs():
        for j_prime in i.loop_jobs():
            for o in i.loop_operations(j):
                for o_prime in i.loop_operations(j_prime):
                    if not is_same(j, j_prime, o, o_prime):
                        model.Add(i.s.exe_start[j][o] >= i.s.exe_start[j_prime][o_prime] + 
                                (i.pos_j[j]*i.s.exe_mode[j_prime][o_prime][PROCEDE_1_PARALLEL_MODE_B] + 2*i.M) * (1-i.job_modeB[j])
                                - i.I * (1-i.s.exe_before[j_prime][j][o_prime][o]))
    return model, i.s

def c8(model: cp_model.CpModel, i: MathInstance):
    for j in i.loop_jobs():
        for o in i.loop_operations(j):
            model.Add(i.s.exe_parallel[j][o] >= i.needed_proc[j][o][PROCEDE_2])
    return model, i.s

def c9(model: cp_model.CpModel, i: MathInstance):
    for j in i.loop_jobs():
        terms = []
        for c in i.loop_stations():
            terms.append(i.s.entry_station_date[j][c] - i.M*i.job_station[j][c] * (1-i.s.job_unload[j][c]) * (i.pos_j[j]+i.job_robot[j]))
        model.Add(i.s.exe_start[j][FIRST_OP] >= sum(terms) + i.M)
    return model, i.s

def c10(model: cp_model.CpModel, i: MathInstance):
    for j in i.loop_jobs():
        for o in i.loop_operations(j):
            terms = []
            for m in i.loop_modes():
                terms.append(i.s.exe_mode[j][o][m])
            model.Add(1 == sum(terms))
    return model, i.s

def c11(model: cp_model.CpModel, i: MathInstance):
    for j in i.loop_jobs():
        for o in i.loop_operations(j):
            model.Add(i.s.exe_mode[j][o][PROCEDE_2_MODE_C] == i.needed_proc[j][o][PROCEDE_2])
    return model, i.s

def c12(model: cp_model.CpModel, i: MathInstance):
    for j in i.loop_jobs():
        terms = []
        for c in i.loop_stations():
            terms.append(i.s.job_loaded[j][c])
        model.Add(1 == sum(terms))
    return model, i.s

def c13(model: cp_model.CpModel, i: MathInstance):
    for j in i.loop_jobs():
        model.Add(i.s.job_loaded[j][STATION_2] >= i.lp[j])
    return model, i.s

def c14(model: cp_model.CpModel, i: MathInstance):
    for j in i.loop_jobs():
        for o in i.loop_operations(j):
            model.Add(i.s.delay[j] >= end(i, j, o) + i.L + i.M - i.due_date[j])
    return model, i.s

def c15(model: cp_model.CpModel, i: MathInstance):
    for j in i.loop_jobs():
        for j_prime in i.loop_jobs():
            if (j != j_prime):
                for o in i.loop_operations(j):
                    for o_prime in i.loop_operations(j_prime):
                        model.Add(i.s.delay[j] >= free(i, j, j_prime, o, o_prime) + i.L + 3*i.M - i.due_date[j])
    return model, i.s

def c16(model: cp_model.CpModel, i: MathInstance):
    for j in i.loop_jobs():
        for j_prime in i.loop_jobs():
            if (j != j_prime):
                for o_prime in i.loop_operations(j_prime):
                    for c in i.loop_stations():
                        model.Add(i.s.entry_station_date[j][c] >= end(i, j_prime, o_prime) - prec(i, j_prime, j, c) + 2*i.L + i.M)
    return model, i.s

def c17(model: cp_model.CpModel, i: MathInstance):
    for j in i.loop_jobs():
        for j_prime in i.loop_jobs():
            for j_second in i.loop_jobs():
                if (j!=j_prime) and (j!=j_second) and (j_prime!=j_second):
                    for o_prime in i.loop_operations(j_prime):
                        for o_second in i.loop_operations(j_second):
                            for c in i.loop_stations():
                                model.Add(i.s.entry_station_date[j][c] >= free(i, j_prime, j_second, o_prime, o_second) - prec(i, j_prime, j, c) + 2*i.L + 3*i.M)
    return model, i.s

def c18(model: cp_model.CpModel, i: MathInstance):
    for j in i.loop_jobs():
        for c_prime in i.loop_stations():
            terms = []
            for c in i.loop_stations():
                terms.append(i.s.job_unload[j][c] * (i.M*i.job_robot[j] + 3*i.M*i.job_modeB[j] + 2*i.L))
            model.Add(i.s.entry_station_date[j][c] >= sum(terms) - i.I*(1-i.s.job_loaded[j][c_prime]))
    return model, i.s

def c19(model: cp_model.CpModel, i: MathInstance):
    for j in i.loop_jobs():
        for j_prime in i.loop_jobs():
            if (j != j_prime):
                for c in i.loop_stations():
                    model.Add(i.s.entry_station_date[j][c] >= i.job_station[j_prime][c]*(2*i.L + i.M*i.job_robot[j_prime] + 3*i.M*i.job_modeB[j_prime]) - i.I*(1-i.s.job_loaded[j][c]))
    return model, i.s

def c20(model: cp_model.CpModel, i: MathInstance):
    for j in i.loop_jobs():
        for j_prime in i.loop_jobs():
            if (j != j_prime):
                for c in i.loop_stations():
                    model.Add(1 <= i.s.entry_station_date[j][c] + i.I*(3 - i.job_station[j][c] - i.s.exe_before[j_prime][j][FIRST_OP][FIRST_OP] - i.s.job_loaded[j_prime][c]))
    return model, i.s

def c21(model: cp_model.CpModel, i: MathInstance):
    for j in i.loop_jobs():
        for o in i.loop_operations(j):
            terms = []
            for p in i.loop_jobs():
                for c in i.loop_stations():
                    terms.append(i.s.job_unload[p][c] * (i.M*i.job_robot[p] + 2*i.M*i.job_modeB[p]))
            model.Add(i.s.exe_start[j][o] >= sum(terms))
    return model, i.s

def solver(instance_file, debug: bool=True):
    instance: Instance = Instance.load(instance_file) # PPO instance
    print("---------------------------------------")
    print("=*= DISPLAY INSTANCE IN OOP MODE (_mode readable for human_)=*=")
    print(instance)
    print(instance.jobs[0])
    print(instance.jobs[0].operations[0])
    print("---------------------------------------")
    i: MathInstance = MathInstance(instance.jobs) # Math instance
    print("=*= DISPLAY INSTANCE IN CP MODE (_mode usable for math_)=*=")
    print(i.welding_time)
    print(i.has_history)
    print(i.lp)
    print(i.needed_proc)
    print("---------------------------------------")

    model = cp_model.CpModel()
    solver = cp_model.CpSolver()
    init_vars(model, i)
    init_objective_function(model, i)
    for constraint in [c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15,c16,c17,c18,c19,c20,c21]:
        model, i.s = constraint(model, i)

    if debug:
        solver.parameters.log_search_progress = True
        solver.parameters.cp_model_probing_level = 0
        solver.parameters.enumerate_all_solutions = True

    status = solver.Solve(model)
    
    if status == cp_model.OPTIMAL:
        print("Solution optimale trouvée!")
        print(f'fn obj= {solver.ObjectiveValue()}')
    else:
        print(f"Pas de solution optimale trouvée. Statut: {STATUS_MEANING[status]}")
        
if __name__ == "__main__":
    #solver('./mini_instance_1.json') # OPTIMAL
    solver('./mini_instance_2.json') # INFEASIBLE
    #solver('./1st_instance.json') # INFEASIBLE
