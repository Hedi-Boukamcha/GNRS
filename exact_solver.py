import csv
import os
from models.instance import Instance, MathInstance, FIRST_OP, PROCEDE_1_SEQ_MODE_A, PROCEDE_1_PARALLEL_MODE_B, PROCEDE_2_MODE_C, STATION_1, STATION_2, STATION_3, PROCEDE_1, PROCEDE_2
from ortools.sat.python import cp_model
from simulators.cp_simulator import gantt_cp_solution, simulate_schedule, simulate_instance
import random
import json

STATUS_MEANING = ["UNKNOWN", "MODEL_INVALID", "FEASIBLE", "INFEASIBLE", "OPTIMAL"]
W_Cmax: int = 1
W_delay: int = 99

def init_vars(model: cp_model.CpModel, i: MathInstance):
    i.s.entry_station_date = [[model.NewIntVar(0, i.I, f'entry_station_date_{j}_{c}') for c in i.loop_stations()] for j in i.loop_jobs()]
    i.s.C_max = model.NewIntVar(0, i.I, "C_max")
    i.s.delay = [model.NewIntVar(0, i.I, f'delay_{j}') for j in i.loop_jobs()] 
    i.s.end_j = [model.NewIntVar(0, i.I, f'end_j_{j}') for j in i.loop_jobs()] 
    i.s.end_o =  [[model.NewIntVar(0, i.I, f'end_o_{j}_{o}') for o in i.loop_operations(j)] for j in i.loop_jobs()]
    i.s.free_j = [model.NewIntVar(0, i.I, f'free_j_{j}') for j in i.loop_jobs()]  
    i.s.free_o =  [[model.NewIntVar(0, i.I, f'free_o_{j}_{o}') for o in i.loop_operations(j)] for j in i.loop_jobs()]
    i.s.end_o =  [[model.NewIntVar(0, i.I, f'end_o_j{j}_{o}') for o in i.loop_operations(j)] for j in i.loop_jobs()]
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
    terms.append(W_Cmax * i.s.C_max)
    for j in i.loop_jobs():        
        terms.append( W_delay * i.s.delay[j])
    model.Minimize(sum(terms))
    return model, i.s

# Check if job j is loaded on station c and if its loaded before j'
def prec(i: MathInstance, j: int, j_prime: int, c: int): #c = station
    return i.I * (3 - i.s.exe_before[j][j_prime][FIRST_OP][FIRST_OP] - i.s.job_loaded[j][c] - i.s.job_loaded[j_prime][c])

# Check the real end date of operation o of job j considering the mandatory robot travel and position time (if mode B)
def end(i: MathInstance, j: int, o: int):
    if (o == 0):
        return i.s.exe_start[j][o] + i.welding_time[j][o] + ((i.pos_j[j] * i.s.exe_mode[j][o][PROCEDE_1_PARALLEL_MODE_B]) * (1 - i.job_modeB[j]))
    else:
        return i.s.exe_start[j][o] + i.welding_time[j][o] + (i.pos_j[j] * i.s.exe_mode[j][o][PROCEDE_1_PARALLEL_MODE_B])

# Check the time at which the robot could be free to move job j after its operation o considering anther operation o' of job j' that could occupy the robot arm
def free(i: MathInstance, j: int, j_prime: int, o: int, o_prime: int):
    if (i.nb_jobs == 2):
        return end(i, j_prime, o_prime) - i.I * (3 - i.s.exe_before[j][j_prime][o][o_prime] - i.s.exe_mode[j][o][PROCEDE_1_PARALLEL_MODE_B] - i.s.exe_mode[j_prime][o_prime][PROCEDE_2_MODE_C])
    else:
        terms = []
        for q in i.loop_jobs():
            if (q != j) and (q != j_prime):
                for x in range(i.operations_by_job[q]):
                    term = (i.s.exe_before[j][q][o][x] + i.s.exe_before[q][j_prime][x][o_prime]) - i.s.exe_before[j][j_prime][o][o_prime]
                    terms.append(i.needed_proc[q][x][PROCEDE_1] * term)
        return end(i, j_prime, o_prime) - i.I * (4 - i.s.exe_before[j][j_prime][o][o_prime] - i.s.exe_mode[j][o][PROCEDE_1_PARALLEL_MODE_B] - i.s.exe_mode[j_prime][o_prime][PROCEDE_2_MODE_C] 
                                     - i.s.exe_parallel[j_prime][o_prime] + sum(terms))

# Cmax computation (case 1: no parallelism)
def c1_s(model: cp_model.CpModel, i: MathInstance):
    for j in i.loop_jobs():
        model.Add(i.s.C_max >= i.s.end_j[j] + i.L + i.M)
    return model, i.s

# Cmax computation (case 2: with parallelism)
def c1_p(model: cp_model.CpModel, i: MathInstance):
    for j in i.loop_jobs():
        model.Add(i.s.C_max >= i.s.free_j[j] + i.L + 3*i.M)
    return model, i.s

def c_end_and_free(model: cp_model.CpModel, i: MathInstance):
    for j in i.loop_jobs():
        for o in i.loop_operations(j):
            model.Add(i.s.end_o[j][o] == end(i,j,o))
            model.Add(i.s.end_j[j] >= i.s.end_o[j][o])
            for j_prime in i.loop_jobs():
                for o_prime in i.loop_operations(j_prime):
                    model.Add(i.s.free_o[j][o] >= free(i, j, j_prime, o, o_prime))
                    model.Add(i.s.free_j[j] >= i.s.free_o[j][o])
    return model, i.s

# Either o_prime before o ; Or o before o_prime [one and only one priority]
def c2(model: cp_model.CpModel, i: MathInstance):
    for j in i.loop_jobs():
        for j_prime in i.loop_jobs():
            for o in i.loop_operations(j):
                for o_prime in i.loop_operations(j_prime):
                    if not is_same(j, j_prime, o, o_prime):
                        model.Add(1 == i.s.exe_before[j][j_prime][o][o_prime] + i.s.exe_before[j_prime][j][o_prime][o])
    return model, i.s

# An operation o starts after the end of the previous one o-1 of the same job (plus 1 to 4 robot moves in case of parallel)
def c3(model: cp_model.CpModel, i: MathInstance):
    for j in i.loop_jobs():
        for o in i.loop_operations(j, exclude_first=True):
            model.Add(i.s.exe_start[j][o] - end(i, j, o-1) - i.M * (3*i.s.exe_parallel[j][o-1] + 1) >= 0) 
    return model, i.s

# An operation o starts after the end of the previous one o_prime of the same job (plus 3 robot moves): the previous one was bloqued by another operation
def c3_b(model: cp_model.CpModel, i: MathInstance):
    for j in i.loop_jobs():
        for o in i.loop_operations(j, exclude_first=True):
            for j_prime in i.loop_jobs():
                if j_prime != j:
                    for o_prime in i.loop_operations(j_prime):
                        model.Add(i.s.exe_start[j][o] - free(i, j, j_prime, o-1, o_prime) >= 3*i.M) 
    return model, i.s

# An operation o starts after the end of the previous one o_prime according to decided priority
# Part 1/3: Except if o is in parrallel (mode B)
def c4(model: cp_model.CpModel, i: MathInstance):
    for j in i.loop_jobs():
        for j_prime in i.loop_jobs():
            for o in i.loop_operations(j):
                for o_prime in i.loop_operations(j_prime):
                    if not is_same(j, j_prime, o, o_prime):
                        model.Add(i.s.exe_start[j][o] - end(i, j_prime, o_prime) + i.I*(1 + i.s.exe_mode[j_prime][o_prime][PROCEDE_1_PARALLEL_MODE_B] - i.s.exe_before[j_prime][j][o_prime][o]) >= 2*i.M)
    return model, i.s

# An operation o starts after the end of the previous one o_prime according to decided priority
# Part 2/3: Except if o_prime can have someone in parrallel (mode C)
def c5(model: cp_model.CpModel, i: MathInstance):
    for j in i.loop_jobs():
        for j_prime in i.loop_jobs():
            for o in i.loop_operations(j):
                for o_prime in i.loop_operations(j_prime):
                    if not is_same(j, j_prime, o, o_prime):
                        model.Add(i.s.exe_start[j][o] - end(i, j_prime, o_prime) + i.I*(1 + i.s.exe_mode[j][o][PROCEDE_2_MODE_C] - i.s.exe_before[j_prime][j][o_prime][o]) >= 2*i.M)
    return model, i.s

# A non-parallel operation o starts after the end of the previous one o_prime according to decided priority
# Part 3/3: Case of no exeption (the robot was really busy with the previous op..)
def c6(model: cp_model.CpModel, i: MathInstance):
    for j in i.loop_jobs():
        for j_prime in i.loop_jobs():
            for o in i.loop_operations(j):
                for o_prime in i.loop_operations(j_prime):
                    if not is_same(j, j_prime, o, o_prime):
                        model.Add(i.s.exe_start[j][o] - end(i, j_prime, o_prime) + i.I * (1 - i.s.exe_before[j_prime][j][o_prime][o] + i.s.exe_parallel[j][o]) >= 2*i.M)
    return model, i.s

# An operation starts only after a previous operation started + two robot moves + possibly a positioner time (only if the previous job is not already on the positioner)
def c7(model: cp_model.CpModel, i: MathInstance):
    for j in i.loop_jobs():
        for j_prime in i.loop_jobs():
            for o in i.loop_operations(j):
                for o_prime in i.loop_operations(j_prime):
                    if not is_same(j, j_prime, o, o_prime):
                        model.Add(i.s.exe_start[j][o] - i.s.exe_start[j_prime][o_prime] - (i.pos_j[j_prime] + i.M) * i.s.exe_mode[j_prime][o_prime][PROCEDE_1_PARALLEL_MODE_B] * (1-i.job_modeB[j_prime]) + i.I * (1-i.s.exe_before[j_prime][j][o_prime][o]) >= i.M * (1 - i.job_robot[j]))
    return model, i.s

# Only operation needing Process 2 can be executed in parallel (meaning: there is another job in the positioner for Process 1)
def c8(model: cp_model.CpModel, i: MathInstance):
    for j in i.loop_jobs():
        for o in i.loop_operations(j):
            model.Add(i.s.exe_parallel[j][o] <= i.needed_proc[j][o][PROCEDE_2])
    return model, i.s

# The first operation of a job (that is not removed form a station & has no history) starts its first operation after being loaded + one robot move
def c9(model: cp_model.CpModel, i: MathInstance):
    for j in i.loop_jobs():
        terms = []
        for c in i.loop_stations():
            terms.append(i.s.entry_station_date[j][c] - i.M*i.job_station[j][c] * (1-i.s.job_unload[j][c]) * (i.job_modeB[j]+i.job_robot[j]))
        model.Add(i.s.exe_start[j][FIRST_OP] - sum(terms) >= i.M)
    return model, i.s

# Operations are executed in one and exactly one mode
def c10(model: cp_model.CpModel, i: MathInstance):
    for j in i.loop_jobs():
        for o in i.loop_operations(j):
            terms = []
            for m in i.loop_modes():
                terms.append(i.s.exe_mode[j][o][m])
            model.Add(1 == sum(terms))
    return model, i.s

# Operation requirering Process 2 are executed in Mode C (neither A or B - reserved for Process 1)
def c11(model: cp_model.CpModel, i: MathInstance):
    for j in i.loop_jobs():
        for o in i.loop_operations(j):
            model.Add(i.s.exe_mode[j][o][PROCEDE_2_MODE_C] == i.needed_proc[j][o][PROCEDE_2])
    return model, i.s

# Each must enter one and exactly one loading station!
def c12(model: cp_model.CpModel, i: MathInstance):
    for j in i.loop_jobs():
        terms = []
        for c in i.loop_stations():
            terms.append(i.s.job_loaded[j][c])
        model.Add(1 == sum(terms))
    return model, i.s

# A large job should enter only the station 2! Other jobs have the choice...
def c13(model: cp_model.CpModel, i: MathInstance):
    for j in i.loop_jobs():
        model.Add(i.s.job_loaded[j][STATION_2] >= i.lp[j])
    return model, i.s

# Delay case 1: |end last operation - deadline|
def c14(model: cp_model.CpModel, i: MathInstance):
    for j in i.loop_jobs():
        model.Add(i.s.delay[j] - i.s.end_j[j] >= i.L + i.M - i.due_date[j])
        # model.Add(i.s.delay[j] - end(i, j, i.last_operations(j)) >= i.L + i.M - i.due_date[j])
    return model, i.s

# Delay case 2: still equals |end - deadline| but this time, the job is blocked in mode B and waits for the end of another job
def c15(model: cp_model.CpModel, i: MathInstance):
    for j in i.loop_jobs():
        for j_prime in i.loop_jobs():
            if (j != j_prime):
                for o in i.loop_operations(j):
                    for o_prime in i.loop_operations(j_prime):
                        model.Add(i.s.delay[j] - i.s.free_j[j] >= i.L + 3*i.M - i.due_date[j])
                        # model.Add(i.s.delay[j] - free(i, j, j_prime, o, o_prime) >= i.L + 3*i.M - i.due_date[j])
    return model, i.s

# A Job enters into a station only after the previous has exited (case 1: no parallelism)
def c16(model: cp_model.CpModel, i: MathInstance):
    for j in i.loop_jobs():
        for j_prime in i.loop_jobs():
            if (j != j_prime):
                for o_prime in i.loop_operations(j_prime):
                    for c in i.loop_stations():
                        model.Add(i.s.entry_station_date[j][c] - end(i, j_prime, o_prime) + prec(i, j_prime, j, c) >= 2*i.L + i.M)
    return model, i.s

# A Job enters into a station only after the previous has exited (which sometimes requires waiting for it to be freed from a third job - in case of last operation in B mode)
def c17(model: cp_model.CpModel, i: MathInstance):
    for j in i.loop_jobs():
        for j_prime in i.loop_jobs():
            for j_second in i.loop_jobs():
                if (j!=j_prime) and (j!=j_second) and (j_prime!=j_second):
                    for o_prime in i.loop_operations(j_prime):
                        for o_second in i.loop_operations(j_second):
                            for c in i.loop_stations():
                                model.Add(i.s.entry_station_date[j][c] - free(i, j_prime, j_second, o_prime, o_second) + prec(i, j_prime, j, c) >= 2*i.L + 3*i.M)
    return model, i.s

# A possible entering date into a loading station must wait for unloading times (part 1): same job is unloaded from another station
def c18(model: cp_model.CpModel, i: MathInstance):
    for j in i.loop_jobs():
        for c_prime in i.loop_stations():
            terms = []
            for c in i.loop_stations():
                sub_terms = []
                for j_prime in i.loop_jobs():
                    if j_prime != j:
                        sub_terms.append(i.job_robot[j_prime])
                terms.append(i.s.job_unload[j][c] * (i.M*i.job_robot[j] + (2*i.M + i.M*sum(sub_terms))*i.job_modeB[j] + 2*i.L))
            model.Add(0 >= sum(terms) - i.I*(1-i.s.job_loaded[j][c_prime]) - i.s.entry_station_date[j][c_prime])
    return model, i.s

# A possible entering date into a loading station must wait for unloading times (part 2): another job is unloaded from the same station
def c19(model: cp_model.CpModel, i: MathInstance):
    for j in i.loop_jobs():
        for j_prime in i.loop_jobs():
            if (j != j_prime):
                for c in i.loop_stations():
                    terms = []
                    for j_sec in i.loop_jobs():
                        if j_sec != j_prime and j_sec != j:
                            terms.append(i.job_robot[j_sec])
                    model.Add(0 >= i.job_station[j_prime][c]*(2*i.L + i.M*i.job_robot[j_prime] + (2*i.M + i.M*sum(terms))*i.job_modeB[j_prime]) - i.I*(1-i.s.job_loaded[j][c]) - i.s.entry_station_date[j][c])
    return model, i.s

# If a job is executed before one that have an history (already either in robot or positioner), the later should be removed
def c20(model: cp_model.CpModel, i: MathInstance):
    for j in i.loop_jobs():
        for j_prime in i.loop_jobs():
            if (j != j_prime):
                for c in i.loop_stations():
                    model.Add(1 <= i.s.job_unload[j][c] + i.I*(3 - i.job_station[j][c] - i.s.exe_before[j_prime][j][FIRST_OP][FIRST_OP] - i.s.job_loaded[j_prime][c]))
    return model, i.s

# The start of the first operation of any job should wait for possible unloading time of another job (either from robot or positioner)
def c21(model: cp_model.CpModel, i: MathInstance):
    for j in i.loop_jobs():
        for o in i.loop_operations(j):
            terms = []
            for p in i.loop_jobs():
                for c in i.loop_stations():
                    terms.append(i.s.job_unload[p][c] * (i.M*i.job_robot[p]*(1-i.job_modeB[j]) + 2*i.M*i.job_modeB[p]))
            model.Add(i.s.exe_start[j][o] - sum(terms) >= 0)
    return model, i.s

def solver_per_file(instance_file, debug: bool=True):
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
    print("-----------------------------HISTORY", i.has_history)
    print(i.lp)
    print(i.needed_proc)
    print(i.job_robot)
    print(i.job_modeB)
    print(i.job_station)
    print("---------------------------------------")
    print([i.last_operations(j) for j in i.loop_jobs()])

    model = cp_model.CpModel()
    solver = cp_model.CpSolver()
    init_vars(model, i)
    init_objective_function(model, i)
    for constraint in [c_end_and_free,c1_s,c1_p,c2,c3,c3_b,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15,c16,c17,c18,c19,c20,c21]:
        model, i.s = constraint(model, i)

    if debug:
        solver.parameters.max_time_in_seconds = 60.0 * 60.0
        solver.parameters.relative_gap_limit = 0.0
        solver.parameters.absolute_gap_limit = 0.0
        # solver.parameters.use_implied_bounds = True
        # solver.parameters.use_probing_search = True
        # solver.parameters.cp_model_presolve = True
        # solver.parameters.optimize_with_core = True
        # solver.parameters.log_search_progress = True
        # solver.parameters.enumerate_all_solutions = False
        # solver.parameters.log_search_progress = True
        # solver.parameters.cp_model_probing_level = 0
        # solver.parameters.enumerate_all_solutions = True

    status = solver.Solve(model)
    
    if status == cp_model.OPTIMAL:
        print("Solution optimale trouvée!")
        total_delay = sum(solver.Value(i.s.delay[j]) for j in i.loop_jobs())
        print("Total des delays =", total_delay)
        print("C_max =", solver.Value(i.s.C_max))
        print(f'fn obj= {solver.ObjectiveValue()}')
        instance.display()
        for j in i.loop_jobs():
            print(f"END J{j} = {solver.Value(i.s.end_j[j])} - FREE J{j} = {solver.Value(i.s.free_j[j])}")
            for o in i.loop_operations(j):
                print(f"--> END O{o} = {solver.Value(i.s.end_o[j][o])} - FREE O{o} = {solver.Value(i.s.free_o[j][o])} - PARALLEL = {solver.BooleanValue(i.s.exe_parallel[j][o])}")
        gantt_cp_solution(instance, i, solver, instance_file)

    else:
        print(f"Pas de solution optimale trouvée. Statut: {STATUS_MEANING[status]}")

def solver(instances_folder='data/instances/controled_sizes', debug: bool=True):

    instance_files = sorted([
        os.path.join(instances_folder, f)
        for f in os.listdir(instances_folder)
        if f.endswith('.json')
    ]) # PPO instance

    for idx, file in enumerate(instance_files):
        print(f"\n=== Résolution de {file} ===")
        instance: Instance = Instance.load(file)
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

    for idx, file in enumerate(instance_files):
        model = cp_model.CpModel()
        solver = cp_model.CpSolver()
        init_vars(model, i)
        init_objective_function(model, i)
        for constraint in [c1_p,c1_s,c2,c3,c3_b,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15,c16,c17,c18,c19,c20,c21]:
            model, i.s = constraint(model, i)

        if debug:
            solver.parameters.log_search_progress = True
            solver.parameters.cp_model_probing_level = 0
            solver.parameters.enumerate_all_solutions = True

        status = solver.Solve(model)
    
        print(f"\n=== Résolution de {file} ===")
        if status == cp_model.OPTIMAL:
            print("Solution optimale trouvée!")
            total_delay = sum(solver.Value(i.s.delay[j]) for j in i.loop_jobs())
            print("Total des delays =", total_delay)
            print("C_max =", solver.Value(i.s.C_max))
            print(f'fn obj= {solver.ObjectiveValue()}')
            
            # Sauvegarder la solution dans un fichier CSV
            instance_type = file.split('/')[2]  # Type d'instance, extrait du chemin
            num_instance = int(file.split('_')[-1].split('.')[0])  # Numéro de l'instance extrait du nom du fichier
            #simulate_schedule(instance, i, solver, instance_type, num_instance)  # Sauvegarder les résultats
        else:
            print(f"Pas de solution optimale trouvée. Statut: {STATUS_MEANING[status]}")

# python3 exact_solver.py
if __name__ == "__main__":
    #solver('./mini_instance_1.json')
    #solver('./mini_instance_2.json')
    solver_per_file('data/instances/debug/3rd_instance.json')
    #solver_per_file('data/instances/train/controled_sizes/instance_2.json')
    #solver()

