from models.state import *
from models.instance import Instance
from conf import *

# ##################################
# =*= STEP-BY-STEP GNN SIMULATOR =*=
# ##################################
__author__ = "Hedi Boukamcha - hedi.boukamcha.1@ulaval.ca"
__version__ = "1.0.0"
__license__ = "MIT"

def simulate_all(i: Instance, decisions: list[Decision]) -> list[State]:
    states: list[State] = []
    states.append(State(i, M, L, NB_STATIONS, BIG_STATION, [], automatic_build=True))
    for d in decisions:
        states.append(simulate(state=states[-1], d=d))
    return states

def simulate(s: State, d: Decision) -> State:
    state: State      = s.clone()
    j: JobState       = state.get_job_by_id(d.job_id)
    o: OperationState = j.operation_states[d.operation_id]
    M: int            = state.M
    L: int            = state.L
    robot             = state.robot
    process: Process  = o.get_target_process()

    # 1. Search the possible start time (either load the job or wait for its previous op to finish)
    time: int = search_start_time(state, j, d)

    # 2. Search for a possible parralel job that needs to stay on "positioner" (cancel previous unloading actions)
    job_on_pos_to_unload: JobState = None
    if d.parallel and o.operation.type == PROCEDE_2:
        job_on_pos_to_unload = cancel_unloading_last_parallel_if_exist(state)
    
    # 3. Unload previous job if the target process ain't free
    time = previous_job_back_to_station(robot, j, process, M, time)

    # 4. Move the robot if its not at already at job location
    time = robot_move_to_job(j, o, robot, M, max(time, robot.free_at))

    # 5. Robot moves job to target process
    time = robot_move_to_process(j, o, robot, process, M, time)

    # 6. Job needs to be placed on the positioner
    if d.parallel and o.operation.type == PROCEDE_1:
        time = position_job(j, o, robot, process, time)

    # 7. Execute the operation
    parallel=(d.parallel and o.operation.type == PROCEDE_1)
    time = execute_operation(j, o, robot, process, parallel, time)

    # 8. If the operation is the last of the job, we remove the job from the system
    if o.is_last:
        time = robot_move_job_to_station(robot, j, o, process, M, time)
        unload(j, o, L, time)

    # 9. If a parallel job was waiting (to be unloaded) on the positioner, unload it
    if job_on_pos_to_unload is not None:
        last_op: OperationState = job_on_pos_to_unload.operation_states[-1]
        last_op.end = time
        time = robot_move_job_to_station(robot, job_on_pos_to_unload, last_op, state.process1, M, time)
        unload(job_on_pos_to_unload, last_op, L, time)
    return state

# (1/4) SEARCH START TIME AND LOAD A JOB ###################################################################

def search_start_time(state: State, j: JobState, d: Decision) -> int:
    start: int = 0
    if j.location == None:
        start = search_best_station_and_load_job(state, j)
    if d.operation_id > 0:
        start = j.calendar.get_last_event().end
    return start

def search_best_station_and_load_job(state: State, j: JobState) -> int:
    min_possible_loaded_time: int = -1 
    selected_station: StationState = None
    for s in state.all_stations.get_possible_stations(j.id):
        possible_loading_time = test_loading_time(state, s) 
        if min_possible_loaded_time < 0 or possible_loading_time < min_possible_loaded_time:
            selected_station = s
            min_possible_loaded_time = possible_loading_time
    time:int = get_loading_time_and_force_unloading_previous(state, selected_station)
    time     = load_job_into_station(j, selected_station, state.L, time)
    return time

def load_job_into_station(job: JobState, station: StationState, L: int, time: int):
    loaded_time: int   = time + L
    job.location        = POS_STATION
    job.status          = IN_SYSTEM
    job.current_station = station
    station.current_job = job
    station.calendar.add(Event(start=time, end=loaded_time, event_type=LOAD, job=job, station=station, source=POS_STATION, dest=POS_STATION))
    job.calendar.add(Event(start=time, end=loaded_time, event_type=LOAD, job=job, station=station, source=POS_STATION, dest=POS_STATION))
    return loaded_time

def get_loading_time_and_force_unloading_previous(state: State, station: StationState) -> int: 
    if station.current_job == None: # Case 1: station is free
        return max(0, station.free_at)
    else: # Case 2: unload the blocking job!
        current_job: JobState   = station.current_job
        last_op: OperationState = current_job.get_last_executed_operation()
        time: int               = last_op.end
        if current_job.location == POS_PROCESS_1:
            time = robot_move_job_to_station(state.robot, current_job, last_op, state.process1, state.M, time)
        elif current_job.location == POS_PROCESS_2:
            time = robot_move_job_to_station(state.robot, current_job, last_op, state.process2, state.M, time)
        time = unload(current_job, last_op, L, time)
        return time

def test_loading_time(state: State, station: StationState) -> int: 
    if station.current_job == None: # Case 1: station is free
        return max(0, station.free_at)
    else: # Case 2: station is not free => what time to unload its job?
        current_job: JobState   = station.current_job
        time: int               = current_job.get_last_executed_operation().end
        if current_job.location == POS_PROCESS_1 or current_job.location == POS_PROCESS_2:
            time +=2* state.M if state.robot.location != current_job.location else state.M
        time += state.L
        return time

# (2/4) CANCEL THE UNLOADING OF LAST PARALLEL MODE B #######################################################

def cancel_unloading_last_parallel_if_exist(state: State):
    if state.process1.calendar.len() >= 2:
        previous_last_event: Event = state.process1.calendar.get(-2)
        operation: OperationState = previous_last_event.operation
        j: JobState = previous_last_event.job
        if previous_last_event.event_type == POS and operation.is_last:
            # Rollback the job
            j.calendar.search_and_delete(event_type=MOVE, source=POS_PROCESS_1, dest=POS_STATION)
            j.calendar.search_and_delete(event_type=UNLOAD)
            j.status                      = IN_SYSTEM
            j.location                    = POS_PROCESS_1
            j.operation_states[-1].status = IN_EXECUTION

            # Rollback the station
            j.current_station.calendar.search_and_delete(event_type=UNLOAD, job_id=j.id)
            j.current_station.current_job = j
            
            # Rollback the robot
            state.robot.calendar.search_and_delete(event_type=MOVE, job_id=j.id, source=POS_PROCESS_1, dest=POS_STATION)
            i, e = state.robot.calendar.search(event_type=MOVE, dest=POS_PROCESS_1, job_id=j.id)
            if i>=0:
                del state.robot.calendar[i]
                e: Event
                state.robot.location    = e.source
                state.robot.current_job = None
                state.robot.free_at     = state.robot.calendar.get(-1).end
            return j
    return None

# (3/4) FREE THE TARGET PROCESS IF STILL BUSY ##############################################################

def previous_job_back_to_station(robot: RobotState, j: JobState, process: Process, M: int, time: int) -> int:
    time = max(time, process.free_at)
    if process.calendar.has_events(): 
        previous_job: JobState = process.calendar.get(-1).job
        previous_op: OperationState = process.calendar.get(-1).operation
        if previous_job.job.id != j.id and previous_job.location == process.position_type:
            time = robot_move_job_to_station(robot, previous_job, previous_op, process, M, time)
    return time

def robot_move_job_to_station(robot: RobotState, j: JobState, o: OperationState, process: Process, M: int, time: int):
    time            = robot_move_to_job(j, o, robot, M, time)
    time           += M
    robot.location  = POS_STATION
    j.location      = POS_STATION
    robot.free_at   = time
    robot.calendar.add(Event(start=time, end=(time + M), event_type=MOVE, job=j, source=process.position_type, dest=POS_STATION, operation=o, station=j.current_station))
    j.calendar.add(Event(start=time, end=(time + M), event_type=MOVE, job=j, source=process.position_type, dest=POS_STATION, operation=o, station=j.current_station))
    return time

def unload(j: JobState, o: OperationState, L: int, time: int) ->int:
    unloading_end: int = time + L
    s: StationState    = j.current_station
    s.free_at          = unloading_end
    s.current_job      = None
    j.end              = unloading_end
    j.delay            = max(0, j.end - j.job.due_date)
    j.status           = DONE if o.is_last else NOT_YET
    j.calendar.add(Event(start=time, end=unloading_end, event_type=UNLOAD, job=j, source=POS_STATION, dest=POS_STATION, operation=o, station=j.current_station))
    j.current_station.calendar.add(Event(start=time, end=unloading_end, event_type=UNLOAD, job=j, source=POS_STATION, dest=POS_STATION, operation=o, station=j.current_station))
    return unloading_end

# (4/4) EXECUTE ONE OPERATION ##############################################################################

def execute_operation(j: JobState, o: OperationState, robot: RobotState, process: Process, parallel: bool, time: int) -> int:
    execution_time: int = o.operation.processing_time
    o.start             = time
    time               += execution_time
    process.free_at     = time
    o.end               = time
    o.status            = DONE
    o.remaining_time    = 0
    if not parallel:
        robot.free_at   = time
        robot.add(Event(start=time, end=(time + execution_time), event_type=HOLD, job=j, source=POS_PROCESS_1, dest=POS_PROCESS_1, operation=o, station=None))
    j.calendar.add(Event(start=time, end=(time + execution_time), event_type=EXECUTE, job=j, source=POS_PROCESS_1, dest=POS_PROCESS_1, operation=o, station=None))
    process.calendar.add(Event(start=time, end=(time + execution_time), event_type=EXECUTE, job=j, source=POS_PROCESS_1, dest=POS_PROCESS_1, operation=o, station=None))

def robot_move_to_job(j: JobState, o: OperationState, robot: RobotState, M: int, time: int) -> int:
    if robot.location != j.location:
        s: StationState = j.current_station if j.location == POS_STATION else None
        robot.location  = j.location
        time           += M
        robot.calendar.add(Event(start=time, end=(time + M), event_type=MOVE, job=j, source=robot.location, dest=j.location, operation=o, station=s))
    return time

def robot_move_to_process(j: JobState, o: OperationState, robot: RobotState, process: Process, M: int, time: int) -> int:
    s: StationState = j.current_station if j.location == POS_STATION else None
    time           += M
    robot.free_at   = time
    robot.location  = process.position_type
    j.location      = process.position_type
    robot.calendar.add(Event(start=time, end=(time + M), event_type=MOVE, job=j, source=robot.location, dest=process.position_type, operation=o, station=s))
    j.calendar.add(Event(start=time, end=(time + M), event_type=MOVE, job=j, source=robot.location, dest=process.position_type, operation=o, station=s))
    return time

def position_job(j: JobState, o: OperationState, robot: RobotState, process: Process, time: int) -> int:
    time           += j.job.pos_time
    robot.free_at   = time
    j.calendar.add(Event(start=time, end=(time + j.job.pos_time), event_type=POS, job=j, source=POS_PROCESS_1, dest=POS_PROCESS_1, operation=o, station=None))
    process.calendar.add(Event(start=time, end=(time + j.job.pos_time), event_type=POS, job=j, source=POS_PROCESS_1, dest=POS_PROCESS_1, operation=o, station=None))
    return time

# END OF FILE! ##########################################################################################