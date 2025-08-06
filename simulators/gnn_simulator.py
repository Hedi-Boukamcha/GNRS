from models.state import *
from conf import * 

# ##################################
# =*= STEP-BY-STEP GNN SIMULATOR =*=
# ##################################
__author__  = "Hedi Boukamcha; Anas Neumann"
__email__   = "hedi.boukamcha.1@ulaval.ca; anas.neumann@polymtl.ca"
__version__ = "1.0.0" 
__license__ = "MIT"

def simulate(previous_state: State, d: Decision, clone: bool=False) -> State:
    state: State      = previous_state.clone() if clone else previous_state
    j: JobState       = state.get_job_by_id(d.job_id)
    o: OperationState = j.operation_states[d.operation_id]
    M: int            = state.M
    L: int            = state.L
    robot             = state.robot
    machine: Machine = o.get_target_machine(state)

    # 1. Search for a possible parralel job that needs to stay on "positioner" (cancel previous unloading actions)
    job_on_pos_to_unload: JobState = None
    forbidden_station: StationState = None
    if d.parallel and o.operation.type == MACHINE_2:
        job_on_pos_to_unload, forbidden_station = cancel_unloading_last_parallel_if_exist(state, j.is_big())

    # 2. Search the possible start time (either load the job or wait for its previous op to finish)
    target_job_ready_time: int = search_start_time(state, j, d, forbidden_station)
    
    # 3. Unload previous job if the target machine ain't free
    previous_job_back_to_station(state, robot, j, machine, M)

    # 4. If M2 not in parallel, unload the job in positioner
    if not d.parallel:
        free_positioner(state, robot, M, current_job=j.id)

    # 5. Move the robot if its not at already at job location
    robot_move_to_job(j, o, robot, M)

    # 6. Robot moves job to target machine
    target_job_at_machine_time: int = robot_move_to_machine(j, o, robot, machine, M, job_ready_time=target_job_ready_time)

    # 7. Job needs to be placed on the positioner
    if d.parallel and o.operation.type == MACHINE_1:
        pos_time: int                = position_job(j, o, robot, machine, target_job_at_machine_time)
        time_start_of_execution: int = pos_time
    else:
        time_start_of_execution: int = target_job_at_machine_time

    # 8. Execute the operation
    parallel: bool        = (d.parallel and o.operation.type == MACHINE_1)
    time_end_of_execution = execute_operation(j, o, robot, machine, parallel, time_start_of_execution)

    # 9. If the operation is the last of the job, we remove the job from the system
    if o.is_last:
        robot_move_job_to_station(state, robot, j, o, machine, M)
        unloading_time_target = unload(state, j, o, L, unloading_start=robot.free_at)
    else:
        unloading_time_target = simulate_station_min_free_at(state.robot, j, o, state.M, state.L, time_end_of_execution)

    # 10. If a parallel job was waiting (to be unloaded) on the positioner, unload it
    unloading_time_pos_job: int = -1
    if job_on_pos_to_unload is not None:
        if not o.is_last:
            robot_move_job_to_station(state, robot, j, o, machine, M) 
        last_op: OperationState = job_on_pos_to_unload.operation_states[-1]
        robot_move_job_to_station(state, robot, job_on_pos_to_unload, last_op, state.machine1, M)
        unloading_time_pos_job = unload(state, job_on_pos_to_unload, last_op, L, unloading_start=robot.free_at) 

    state.compute_obj_values_and_upper_bounds(unloading_time=max(unloading_time_target, unloading_time_pos_job), current_time=robot.free_at)
    state.decisions.append(d)
    return state

# (1/4) SEARCH START TIME AND LOAD A JOB ###################################################################

def search_start_time(state: State, j: JobState, d: Decision, forbidden_station: StationState) -> int:
    start_time: int = 0
    if j.location == None:
        start_time = search_best_station_and_load_job(state, j, forbidden_station)
    if d.operation_id > 0:
        start_time = max(start_time, j.calendar.get_last_event().end)
    return start_time

def search_best_station_and_load_job(state: State, j: JobState, forbidden_station: StationState) -> int:
    min_possible_loaded_time: int = -1
    selected_station: StationState = None
    for s in state.all_stations.get_possible_stations(j.is_big()):
        if forbidden_station is None or s.id != forbidden_station.id:
            possible_loading_time = test_loading_time(state, s)
            if min_possible_loaded_time < 0 or possible_loading_time < min_possible_loaded_time or (selected_station.accept_big and not s.accept_big and possible_loading_time <= (1.015 * min_possible_loaded_time)):
                selected_station = s
                min_possible_loaded_time = possible_loading_time
    prev_unload_time: int = get_loading_time_and_force_unloading_previous(state, j, selected_station)
    load_time: int        = load_job_into_station(state, j, selected_station, state.L, prev_unload_time)
    return load_time

def load_job_into_station(state: State, job: JobState, station: StationState, L: int, start_loading_time: int):
    loaded_time: int    = start_loading_time + L if station.calendar.has_events() else 0
    station.calendar.add(Event(start=start_loading_time, end=loaded_time, event_type=LOAD, job=job, station=station, source=state.all_stations, dest=state.all_stations))
    job.calendar.add(Event(start=start_loading_time, end=loaded_time, event_type=LOAD, job=job, station=station, source=state.all_stations, dest=state.all_stations))
    job.location        = state.all_stations
    job.status          = IN_SYSTEM
    job.current_station = station
    station.current_job = job
    return loaded_time

def get_loading_time_and_force_unloading_previous(state: State, j: JobState, station: StationState) -> int: 
    if station.current_job == None: # Case 1: station is free
        return max(0, station.free_at)
    else: # Case 2: unload the blocking job!
        current_job: JobState       = station.current_job
        last_op: OperationState     = current_job.get_last_executed_operation()
        if current_job.location.position_type == POS_MACHINE_1:
            robot_move_job_to_station(state, state.robot, current_job, last_op, state.machine1, state.M)
        elif current_job.location.position_type == POS_MACHINE_2:
            robot_move_job_to_station(state, state.robot, current_job, last_op, state.machine2, state.M)
        print(f"UNLOAD PREVIOUS JOB J{current_job.id + 1} FROM STATION S{station.id +1} TO INSTEAD LOAD JOB J{j.id +1} AT TIME {current_job.calendar.events[-1].end}")
        prev_unload_time: int = unload(state, current_job, last_op, L, unloading_start=current_job.calendar.events[-1].end)
        return prev_unload_time

def test_loading_time(state: State, station: StationState) -> int: 
    if station.current_job == None: # Case 1: station is free
        return max(0, station.free_at)
    else: # Case 2: station is not free => what time to unload its job?
        current_job: JobState   = station.current_job
        last_op: OperationState = current_job.get_last_executed_operation()
        time: int               = last_op.end if last_op else 0
        if current_job.location.position_type == POS_MACHINE_1 or current_job.location.position_type == POS_MACHINE_2:
            time +=2* state.M if state.robot.location != current_job.location else state.M
        time += state.L
        return time

# (2/4) CANCEL THE UNLOADING OF LAST PARALLEL MODE B #######################################################

def cancel_unloading_last_parallel_if_exist(state: State, needs_station_2: bool):
    if state.machine1.calendar.len() >= 2:
        previous_last_event: Event = state.machine1.calendar.get(-2)
        operation: OperationState = previous_last_event.operation
        j: JobState = previous_last_event.job
        if previous_last_event.event_type == POS and operation.is_last and (not needs_station_2 or j.current_station.id != STATION_2):
            # Rollback the job
            j.calendar.events.pop()
            j.calendar.events.pop()
            j.status                      = IN_SYSTEM
            j.location                    = state.machine1
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

# (3/4) FREE THE TARGET MACHINE IF STILL BUSY ##############################################################

def previous_job_back_to_station(state: State, robot: RobotState, j: JobState, machine: Machine, M: int):
    if machine.calendar.has_events():
        previous_job: JobState = machine.calendar.get(-1).job
        previous_op: OperationState = machine.calendar.get(-1).operation
        if previous_job.id != j.id and previous_job.location is not None and previous_job.location.position_type == machine.position_type:
            return robot_move_job_to_station(state, robot, previous_job, previous_op, machine, M)

def simulate_station_min_free_at(robot: RobotState, j: JobState, o: OperationState, M: int, L: int, time_end_of_execution: int) -> int:
    simulated_time = time_end_of_execution + M + L
    if robot.location != j.location:
        simulated_time += M
    j.current_station.free_at = simulated_time # min time at which the station could be free
    simulated_time            = simulated_time + M + j.operation_states[o.id + 1].operation.processing_time
    j.end                     = simulated_time # min time at which the job could end
    j.delay                   = max(0, j.end - j.job.due_date)
    return simulated_time

def robot_move_job_to_station(state: State, robot: RobotState, j: JobState, o: OperationState, machine: Machine, M: int):
    robot_move_to_job(j, o, robot, M)
    time            = max(o.end, robot.free_at, machine.free_at)
    robot.calendar.add(Event(start=time, end=(time + M), event_type=MOVE, job=j, source=machine, dest=state.all_stations, operation=o, station=j.current_station))
    j.calendar.add(Event(start=time, end=(time + M), event_type=MOVE, job=j, source=machine, dest=state.all_stations, operation=o, station=j.current_station))
    machine.free_at = time
    time           += M
    robot.location  = state.all_stations
    j.location      = state.all_stations
    robot.free_at   = time

def free_positioner(state: State, robot: RobotState, M: int, current_job: int):
    if state.machine1.calendar.len() >= 2:
        previous_last_event: Event = state.machine1.calendar.get(-2)
        j: JobState                = previous_last_event.job
        o: OperationState          = previous_last_event.operation
        if current_job != j.id and previous_last_event.event_type == POS and j.location == state.machine1:
            if robot.location != state.machine1:
                robot.calendar.add(Event(start=robot.free_at, end=(robot.free_at + M), event_type=MOVE, job=j, source=robot.location, dest=state.machine1, operation=o, station=j.current_station))
                robot.free_at += M
            time            = max(robot.free_at, state.machine1.free_at, o.end)
            robot.calendar.add(Event(start=time, end=(time + M), event_type=MOVE, job=j, source=state.machine1, dest=state.all_stations, operation=o, station=j.current_station))
            j.calendar.add(Event(start=time, end=(time + M), event_type=MOVE, job=j, source=state.machine1, dest=state.all_stations, operation=o, station=j.current_station))
            time           += M
            robot.location  = state.all_stations
            j.location      = state.all_stations
            robot.free_at   = time

def unload(state: State, j: JobState, o: OperationState, L: int, unloading_start: int) ->int:
    s: StationState      = j.current_station
    unloading_end: int   = unloading_start + L
    j.calendar.add(Event(start=unloading_start, end=unloading_end, event_type=UNLOAD, job=j, source=state.all_stations, dest=state.all_stations, operation=o, station=j.current_station))
    j.current_station.calendar.add(Event(start=j.current_station.calendar.events[-1].end, end=unloading_start, event_type=AWAIT, job=j, source=state.all_stations, dest=state.all_stations, operation=o, station=j.current_station))
    j.current_station.calendar.add(Event(start=unloading_start, end=unloading_end, event_type=UNLOAD, job=j, source=state.all_stations, dest=state.all_stations, operation=o, station=j.current_station))
    s.free_at          = unloading_end
    s.current_job      = None
    j.end              = unloading_end
    #j.current_station = None
    j.delay            = max(0, j.end - j.job.due_date)
    j.status           = DONE if o and o.is_last else NOT_YET
    j.location         = None
    return unloading_end

# (4/4) EXECUTE ONE OPERATION ##############################################################################

def execute_operation(j: JobState, o: OperationState, robot: RobotState, machine: Machine, parallel: bool, time: int) -> int:
    execution_time: int = o.operation.processing_time
    o.start             = time
    if not parallel:
        robot.free_at   = time + execution_time
        robot.calendar.add(Event(start=time, end=(time + execution_time), event_type=HOLD, job=j, source=machine, dest=machine, operation=o, station=None))
    j.calendar.add(Event(start=time, end=(time + execution_time), event_type=EXECUTE, job=j, source=machine, dest=machine, operation=o, station=None))
    machine.calendar.add(Event(start=time, end=(time + execution_time), event_type=EXECUTE, job=j, source=machine, dest=machine, operation=o, station=None))
    time               += execution_time
    machine.free_at     = time
    o.end               = time
    o.status            = DONE
    o.remaining_time    = 0
    return time

def robot_move_to_job(j: JobState, o: OperationState, robot: RobotState, M: int):
    if robot.location  != j.location:
        s: StationState = j.current_station if j.location.position_type == POS_STATION else None
        robot.calendar.add(Event(start=robot.free_at, end=(robot.free_at + M), event_type=MOVE, job=j, source=robot.location, dest=j.location, operation=o, station=s))
        robot.location  = j.location
        robot.free_at  += M

def robot_move_to_machine(j: JobState, o: OperationState, robot: RobotState, machine: Machine, M: int, job_ready_time: int) -> int:
    time = max(job_ready_time, robot.free_at, machine.free_at)
    s: StationState = j.current_station if j.location.position_type == POS_STATION else None
    robot.calendar.add(Event(start=time, end=(time + M), event_type=MOVE, job=j, source=robot.location, dest=machine, operation=o, station=s))
    j.calendar.add(Event(start=time, end=(time + M), event_type=MOVE, job=j, source=robot.location, dest=machine, operation=o, station=s))
    robot.location  = machine
    j.location      = machine
    time           += M
    robot.free_at   = time
    return time

def position_job(j: JobState, o: OperationState, robot: RobotState, machine: Machine, time: int) -> int:
    j.calendar.add(Event(start=time, end=(time + j.job.pos_time), event_type=POS, job=j, source=machine, dest=machine, operation=o, station=None))
    machine.calendar.add(Event(start=time, end=(time + j.job.pos_time), event_type=POS, job=j, source=machine, dest=machine, operation=o, station=None))
    robot.calendar.add(Event(start=time, end=(time + j.job.pos_time), event_type=POS, job=j, source=machine, dest=machine, operation=o, station=None))
    time           += j.job.pos_time
    robot.free_at   = time
    return time

# END OF FILE! ##########################################################################################