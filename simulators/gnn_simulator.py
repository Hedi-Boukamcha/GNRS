from models.instance import Instance, MathInstance
from models.state import *
from conf import * 

import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx


# ##################################
# =*= STEP-BY-STEP GNN SIMULATOR =*=
# ##################################
__author__  = "Hedi Boukamcha; Anas Neumann"
__email__   = "hedi.boukamcha.1@ulaval.ca; anas.neumann@polymtl.ca"
__version__ = "1.0.0" 
__license__ = "MIT"


# (0/4) GANTT FOR CNN SIMULATOR ############################################################################

def plot_gantt_chart(tasks, instance_file: str):
    fig, ax = plt.subplots(figsize=(10, 3))
    for task in tasks:
        level = 0 if task["proc_type"] == MACHINE_1 else 1
        ax.barh(level, task["duration"], left=task["start"], color=task["color"], edgecolor='black')
        ax.text(task["start"] + 0.2, level, f'{task["label"]}', va='bottom', fontsize=8, color='black')
        time_points = sorted(set([task["start"] for task in tasks] + [task["end"] for task in tasks]))
       
    ax.set_xticks(time_points)
    ax.set_xticklabels([f'{t:.1f}' for t in time_points], rotation=45, fontsize=8)

    ax.set_xlim(0, max(time_points) + 1)
    ax.set_xlabel("Temps")
    ax.set_yticks([0, 1]) 
    ax.set_yticklabels(["Procédé 1", "Procédé 2"])
    ax.set_title(f"Diagramme de Gantt - {instance_file}")
    plt.tight_layout()
    plt.grid(True, axis='x', linestyle='--', alpha=0.3)
    plt.show()


def gantt_cp_solution(instance: Instance, i: MathInstance, solver, instance_file: str):
    print("\n--- Simulation Gantt à partir du modèle mathématique ---")
    tasks = []
    job_colors = ['#8dd3c7', '#80b1d3', '#fb8072', '#fdb462', '#b3de69', '#fccde5']
    
    for j, job in enumerate(instance.jobs):
        assigned_station = None
        for c in [0, 1, 2]:
            if solver.BooleanValue(i.s.job_loaded[j][c]):
                assigned_station = c
                break
        if assigned_station is None:
            assigned_station = 0

        for o, op in enumerate(job.operations):
            is_removed: bool = False
            modeB = solver.BooleanValue(i.s.exe_mode[j][o][1])
            for c in i.loop_stations():
                is_removed = is_removed or solver.BooleanValue(i.s.job_unload[j][c])
            has_pos_j = modeB and (o>0 or is_removed or not i.job_modeB[j])
            start = solver.Value(i.s.exe_start[j][o])
            duration = i.welding_time[j][o] + (i.pos_j[j] if has_pos_j else 0)
            end = start + duration
            station = f"n°{assigned_station + 1}"
            if op.type == 2:
                mode_str = "Mode C"
            else:
                mode_str = "Mode B" if modeB else "Mode A"
            proc_type = 1 if op.type == 2 else 0
            tasks.append({
                "label": f"J{j+1} O{o+1} S{station} ({mode_str})",
                "start": start,
                "end": end,
                "duration": duration,
                "color": job_colors[j % len(job_colors)],
                "station": assigned_station,
                "proc_type": proc_type
            })

    plot_gantt_chart(tasks, instance_file)


def simulate(previous_state: State, d: Decision, clone: bool=False) -> State:
    state: State      = previous_state.clone() if clone else previous_state
    j: JobState       = state.get_job_by_id(d.job_id)
    o: OperationState = j.operation_states[d.operation_id]
    M: int            = state.M
    L: int            = state.L
    robot             = state.robot
    process: Process  = o.get_target_process(state)

    # 1. Search for a possible parralel job that needs to stay on "positioner" (cancel previous unloading actions)
    job_on_pos_to_unload: JobState = None
    forbidden_station: StationState = None
    if d.parallel and o.operation.type == MACHINE_2:
        job_on_pos_to_unload, forbidden_station = cancel_unloading_last_parallel_if_exist(state, j.is_big())

    # 2. Search the possible start time (either load the job or wait for its previous op to finish)
    time: int = search_start_time(state, j, d, forbidden_station)
    
    # 3. Unload previous job if the target process ain't free
    time = previous_job_back_to_station(state, robot, j, process, M, time)

    # 4. Move the robot if its not at already at job location
    time = robot_move_to_job(j, o, robot, M, max(time, robot.free_at))

    # 5. Robot moves job to target process
    time = robot_move_to_process(j, o, robot, process, M, time)

    # 6. Job needs to be placed on the positioner
    if d.parallel and o.operation.type == MACHINE_1:
        time = position_job(j, o, robot, process, time)

    # 7. Execute the operation
    parallel=(d.parallel and o.operation.type == MACHINE_1)
    time = execute_operation(j, o, robot, process, parallel, time)

    # 8. If the operation is the last of the job, we remove the job from the system
    max_time = time
    if o.is_last:
        time     = robot_move_job_to_station(state, robot, j, o, process, M, time)
        max_time = unload(state, j, o, L, time)
    else:
        max_time = simulate_station_min_free_at(state.robot, j, o, state.M, state.L, time)

    # 9. If a parallel job was waiting (to be unloaded) on the positioner, unload it
    if job_on_pos_to_unload is not None:
        if not o.is_last:
            time = robot_move_job_to_station(state, robot, j, o, process, M, time)
        last_op: OperationState = job_on_pos_to_unload.operation_states[-1]
        last_op.end    = time
        time           = robot_move_job_to_station(state, robot, job_on_pos_to_unload, last_op, state.process1, M, time)
        unloading_time = unload(state, job_on_pos_to_unload, last_op, L, time)
        max_time       = max(max_time, unloading_time)

    state.compute_reward_values(max_time)
    state.decisions.append(d)
    return state

# (1/4) SEARCH START TIME AND LOAD A JOB ###################################################################

def search_start_time(state: State, j: JobState, d: Decision, forbidden_station: StationState) -> int:
    start: int = 0
    if j.location == None:
        start = search_best_station_and_load_job(state, j, forbidden_station)
    if d.operation_id > 0:
        start = j.calendar.get_last_event().end
    return start

def search_best_station_and_load_job(state: State, j: JobState, forbidden_station: StationState) -> int:
    min_possible_loaded_time: int = -1 
    selected_station: StationState = None
    for s in state.all_stations.get_possible_stations(j.is_big()):
        if forbidden_station is None or s.id != forbidden_station.id:
            possible_loading_time = test_loading_time(state, s) 
            if min_possible_loaded_time < 0 or possible_loading_time < min_possible_loaded_time:
                selected_station = s
                min_possible_loaded_time = possible_loading_time    
    time: int = get_loading_time_and_force_unloading_previous(state, selected_station)
    time      = load_job_into_station(state, j, selected_station, state.L, time)
    return time

def load_job_into_station(state: State, job: JobState, station: StationState, L: int, time: int):
    loaded_time: int   = time + L if station.calendar.has_events() else 0
    station.calendar.add(Event(start=time, end=loaded_time, event_type=LOAD, job=job, station=station, source=state.all_stations, dest=state.all_stations))
    job.calendar.add(Event(start=time, end=loaded_time, event_type=LOAD, job=job, station=station, source=state.all_stations, dest=state.all_stations))
    job.location        = state.all_stations
    job.status          = IN_SYSTEM
    job.current_station = station
    station.current_job = job
    return loaded_time

def get_loading_time_and_force_unloading_previous(state: State, station: StationState) -> int: 
    if station.current_job == None: # Case 1: station is free
        return max(0, station.free_at)
    else: # Case 2: unload the blocking job!
        current_job: JobState   = station.current_job
        last_op: OperationState = current_job.get_last_executed_operation()
        time: int               = last_op.end if last_op else 0
        if current_job.location.position_type == POS_PROCESS_1:
            time = robot_move_job_to_station(state, state.robot, current_job, last_op, state.process1, state.M, time)
        elif current_job.location.position_type == POS_PROCESS_2:
            time = robot_move_job_to_station(state, state.robot, current_job, last_op, state.process2, state.M, time)
        time = unload(state, current_job, last_op, L, time)
        return time

def test_loading_time(state: State, station: StationState) -> int: 
    if station.current_job == None: # Case 1: station is free
        return max(0, station.free_at)
    else: # Case 2: station is not free => what time to unload its job?
        current_job: JobState   = station.current_job
        last_op: OperationState = current_job.get_last_executed_operation()
        time: int               = last_op.end if last_op else 0
        if current_job.location.position_type == POS_PROCESS_1 or current_job.location.position_type == POS_PROCESS_2:
            time +=2* state.M if state.robot.location != current_job.location else state.M
        time += state.L
        return time

# (2/4) CANCEL THE UNLOADING OF LAST PARALLEL MODE B #######################################################

def cancel_unloading_last_parallel_if_exist(state: State, needs_station_2: bool):
    if state.process1.calendar.len() >= 2:
        previous_last_event: Event = state.process1.calendar.get(-2)
        operation: OperationState = previous_last_event.operation
        j: JobState = previous_last_event.job
        if previous_last_event.event_type == POS and operation.is_last and (not needs_station_2 or j.current_station.id != STATION_2):
            # Rollback the job
            j.calendar.events.pop()
            j.calendar.events.pop()
            j.status                      = IN_SYSTEM
            j.location                    = state.process1
            j.operation_states[-1].status = IN_EXECUTION

            # Rollback the station
            j.current_station.calendar.events.pop()
            j.current_station.current_job = j
            
            # Rollback the robot
            e: Event    = state.robot.calendar.events.pop()
            prev: Event = state.robot.calendar.events[-1]
            if prev.event_type == MOVE : # if the robot was not already at station 1
                e = state.robot.calendar.events.pop()
            state.robot.location    = e.source
            state.robot.current_job = None
            state.robot.free_at     = state.robot.calendar.get(-1).end
            return j, j.current_station
    return None, None

# (3/4) FREE THE TARGET PROCESS IF STILL BUSY ##############################################################

def previous_job_back_to_station(state: State, robot: RobotState, j: JobState, process: Process, M: int, time: int) -> int:
    time = max(time, process.free_at)
    if process.calendar.has_events(): 
        previous_job: JobState = process.calendar.get(-1).job
        previous_op: OperationState = process.calendar.get(-1).operation
        if previous_job.id != j.id and previous_job.location.position_type == process.position_type:
            time = robot_move_job_to_station(state, robot, previous_job, previous_op, process, M, time)
    return time

def simulate_station_min_free_at(robot: RobotState, j: JobState, o: OperationState, M: int, L: int, time: int) -> int:
    simulated_time = time + M + L
    if robot.location != j.location:
        simulated_time += M
    j.current_station.free_at = simulated_time # min time at which the station could be free
    simulated_time = simulated_time + M + j.operation_states[o.id + 1].operation.processing_time
    j.end                     = simulated_time # min time at which the job could end
    j.delay                   = max(0, j.end - j.job.due_date)
    return simulated_time

def robot_move_job_to_station(state: State, robot: RobotState, j: JobState, o: OperationState, process: Process, M: int, time: int):
    time            = robot_move_to_job(j, o, robot, M, time)
    robot.calendar.add(Event(start=time, end=(time + M), event_type=MOVE, job=j, source=process, dest=state.all_stations, operation=o, station=j.current_station))
    j.calendar.add(Event(start=time, end=(time + M), event_type=MOVE, job=j, source=process, dest=state.all_stations, operation=o, station=j.current_station))
    process.free_at = time
    time           += M
    robot.location  = state.all_stations
    j.location      = state.all_stations
    robot.free_at   = time
    return time

def unload(state: State, j: JobState, o: OperationState, L: int, time: int) ->int:
    unloading_end: int = time + L
    s: StationState    = j.current_station
    j.calendar.add(Event(start=time, end=unloading_end, event_type=UNLOAD, job=j, source=state.all_stations, dest=state.all_stations, operation=o, station=j.current_station))
    j.current_station.calendar.add(Event(start=time, end=unloading_end, event_type=UNLOAD, job=j, source=state.all_stations, dest=state.all_stations, operation=o, station=j.current_station))
    s.free_at          = unloading_end
    s.current_job      = None
    j.end              = unloading_end
    j.delay            = max(0, j.end - j.job.due_date)
    j.status           = DONE if o and o.is_last else NOT_YET
    return unloading_end

# (4/4) EXECUTE ONE OPERATION ##############################################################################

def execute_operation(j: JobState, o: OperationState, robot: RobotState, process: Process, parallel: bool, time: int) -> int:
    execution_time: int = o.operation.processing_time
    o.start             = time
    if not parallel:
        robot.free_at   = time
        robot.calendar.add(Event(start=time, end=(time + execution_time), event_type=HOLD, job=j, source=process, dest=process, operation=o, station=None))
    j.calendar.add(Event(start=time, end=(time + execution_time), event_type=EXECUTE, job=j, source=process, dest=process, operation=o, station=None))
    process.calendar.add(Event(start=time, end=(time + execution_time), event_type=EXECUTE, job=j, source=process, dest=process, operation=o, station=None))
    time               += execution_time
    process.free_at     = time
    o.end               = time
    o.status            = DONE
    o.remaining_time    = 0
    return time

def robot_move_to_job(j: JobState, o: OperationState, robot: RobotState, M: int, time: int) -> int:
    if robot.location != j.location:
        s: StationState = j.current_station if j.location.position_type == POS_STATION else None
        robot.calendar.add(Event(start=time, end=(time + M), event_type=MOVE, job=j, source=robot.location, dest=j.location, operation=o, station=s))
        robot.location  = j.location
        time           += M
    return time

def robot_move_to_process(j: JobState, o: OperationState, robot: RobotState, process: Process, M: int, time: int) -> int:
    s: StationState = j.current_station if j.location.position_type == POS_STATION else None
    robot.calendar.add(Event(start=time, end=(time + M), event_type=MOVE, job=j, source=robot.location, dest=process, operation=o, station=s))
    j.calendar.add(Event(start=time, end=(time + M), event_type=MOVE, job=j, source=robot.location, dest=process, operation=o, station=s))
    robot.location  = process
    j.location      = process
    time           += M
    robot.free_at   = time
    return time

def position_job(j: JobState, o: OperationState, robot: RobotState, process: Process, time: int) -> int:
    j.calendar.add(Event(start=time, end=(time + j.job.pos_time), event_type=POS, job=j, source=process, dest=process, operation=o, station=None))
    process.calendar.add(Event(start=time, end=(time + j.job.pos_time), event_type=POS, job=j, source=process, dest=process, operation=o, station=None))
    robot.calendar.add(Event(start=time, end=(time + j.job.pos_time), event_type=POS, job=j, source=process, dest=process, operation=o, station=None))
    time           += j.job.pos_time
    robot.free_at   = time
    return time

# END OF FILE! ##########################################################################################