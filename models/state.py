from dataclasses import dataclass
from typing import Union
from models.instance import Operation, Job, Instance
from conf import *
from torch import Tensor
from torch_geometric.data import HeteroData
import torch

# ######################################################
# =*= GNN STATE: INSTANCE DATA AND PARTIAL SOLUTION =*=
# #####################################################
__author__ = "Hedi Boukamcha - hedi.boukamcha.1@ulaval.ca"
__version__ = "1.0.0"
__license__ = "MIT"

# Events and Decision data ---------------------------------------------------------------------------------------
Position = Union['Process', 'Process1', 'Process2', 'Stations']
 
@dataclass
class Decision: 
    def __init__(self, job_id: int, operation_id: int, parallel: bool = False):
        self.job_id: int       = job_id
        self.operation_id: int = operation_id
        self.parallel: bool    = parallel

    def clone(self) -> 'Decision':
        return Decision(self.job_id, self.operation_id, self.parallel)
    
    def __str__(self) -> str:
        return f"Decision(job_id={self.job_id}, operation_id={self.operation_id}, parallel={self.parallel})"

@dataclass
class Event:
    def __init__(self, start: int, end: int, event_type: int, job: 'JobState', source: Position = None, dest: Position = None, operation: 'OperationState' = None, station: 'StationState' = None):
        self.job: JobState             = job
        self.operation: OperationState = operation
        self.station: StationState     = station
        self.start: int                = start
        self.end: int                  = end
        self.event_type: int           = event_type
        self.source: Position          = source
        self.dest: Position            = dest

    def clone(self, job: 'JobState', source: Position = None, dest: Position = None, operation: 'OperationState' = None, station: 'StationState' = None):
        return Event(start=self.start, end=self.end, event_type=self.event_type, job=job, source=source, dest=dest, operation=operation, station=station)

    def duration(self) -> int:
        return self.end - self.start
    
    def __str__(self) -> str:
        station: str = f", station={(self.station.id+1)}" if self.station is not None else ""
        job: str = f", job={(self.job.id+1)}" if self.job is not None else ""
        operation: str = f", operation={(self.operation.id+1)}" if self.operation is not None else ""
        source: str = f", source={LOCATION_NAMES[self.source.position_type]}" if self.source is not None else ""
        dest: str = f", dest={LOCATION_NAMES[self.dest.position_type]}" if self.dest is not None else ""
        return f"Event(start={self.start}, end={self.end}, type={EVENT_NAMES[self.event_type]}{source}{dest}{station}{job}{operation})"

@dataclass
class Calendar:
    def __init__(self):
        self.events: list[Event] = []
    
    def get_last_event(self, start_time: int=-1) -> 'Event':
        if start_time < 0:
            return self.events[-1] if self.events else None
        for i, e in enumerate(self.events):
            if e.start > start_time:
                return self.events[i-1] if i>0 else None
        return None

    def get(self, i: int) -> 'Event':
        return self.events[i]
    
    def add(self, e: Event):
        self.events.append(e)

    def len(self):
        return len(self.events)
    
    def has_events(self) -> bool:
        return self.len()>0
    
    def display_calendar(self, title: str):
        print(f"==*== {title} Calendar ==*==")
        for e in self.events:
            print(e)
        print("-----------------------------")

    def clone(self, new_state: 'State') -> 'Calendar':
        c: Calendar = Calendar()
        for e in self.events:
            job: JobState         = new_state.get_job(e.job)
            source: Position      = new_state.process1 if e.source and e.source.position_type == POS_PROCESS_1 \
                                        else new_state.process2 if e.source and e.source.position_type == POS_PROCESS_2 \
                                        else new_state.all_stations if e.source and e.source.position_type == POS_STATION \
                                        else None
            dest: Position        = new_state.process1 if e.dest and e.dest.position_type == POS_PROCESS_1 \
                                        else new_state.process2 if e.dest and e.dest.position_type == POS_PROCESS_2 \
                                        else new_state.all_stations if e.dest and e.dest.position_type == POS_STATION \
                                        else None
            operation: Operation  = job.get_operation(e.operation.id) if job is not None and e.operation is not None else None
            station: StationState = new_state.get_station(e.station) if e.station else None
            new_e: Event = e.clone(job=job, source=source, dest=dest, operation=operation, station=station)
            c.add(new_e)
        return c

# State data ---------------------------------------------------------------------------------------

@dataclass
class State:
    def __init__(self, i: Instance, M: int, L: int, nb_stations: int=0, station_large: int=0, decisions: list[Decision] = [], automatic_build: bool=True):
        self.i: Instance              = i
        self.L: int                    = L
        self.M: int                    = M
        self.cmax: int                 = 0
        self.total_delay: int          = 0
        self.reward: Tensor            = None
        self.decisions: list[Decision] = decisions
        self.process1: Process1        = None
        self.process2: Process2        = None
        self.all_stations: Stations    = None
        self.robot: RobotState         = None
        self.job_states: list[JobState] = []
        self.processess: list[Process] = []
        if automatic_build:
            self.process1      = Process1()
            self.process2      = Process2()
            self.processess    = [self.process1, self.process2]
            self.all_stations  = Stations(nb_stations=nb_stations, station_large=station_large)
            self.robot         = RobotState(init_position=self.all_stations)
            self.job_states     = [JobState(state=self, id=id, stations=self.all_stations.stations, robot=self.robot, job=job) for id, job in enumerate(i.jobs)]
    
    def min_action_time(self) -> int:
        time: int = min(self.process1.free_at, self.process2.free_at, self.robot.free_at, min([s.free_at for s in self.all_stations.stations]))
        return time

    def compute_reward_values(self, end_time: int):
        self.cmax = max(self.cmax, end_time)
        self.total_delay = 0
        for j in self.job_states:
            j.delay = max(0, j.end - j.job.due_date) # real delay or minimal expected delay
            self.total_delay += j.delay

    def display_calendars(self):
        self.process1.calendar.display_calendar("PROCESS #1")
        self.process2.calendar.display_calendar("PROCESS #2")
        self.robot.calendar.display_calendar("ROBOT")
        for s in self.all_stations.stations:
            s.calendar.display_calendar(f"LOADING STATION #{(s.id +1)}")
        for j in self.job_states:
            j.calendar.display_calendar(f"JOB #{(j.id +1)}")

    def clone(self) -> 'State':
        # Stage 1: clone without links
        c: State       = State(self.i, self.M, self.L, automatic_build=False)
        c.decisions    = [d.clone() for d in self.decisions]
        c.process1     = self.process1.clone()
        c.process2     = self.process2.clone()
        c.processess   = [c.process1, c.process2]
        c.reward       = self.reward.clone() if self.reward is not None else None
        c.robot        = self.robot.clone()
        c.job_states    = [j.clone(c) for j in self.job_states]
        c.all_stations = self.all_stations.clone()

        # State 2: clone all OOP links
        self.robot.clone_calendar_and_current_job_and_location(c)
        self.process1.clone_calendar_and_current_job(c)
        self.process2.clone_calendar_and_current_job(c)
        self.all_stations.clone_calendar_and_current_jobs(c)
        for j in self.job_states:
            new_job: JobState = c.get_job(j)
            j.clone_calendar_and_location_and_current_station(c, new_job)
        return c
    
    def get_job(self, job: 'JobState'=None) -> 'JobState':
        if job:
            return self.get_job_by_id(job.id)
        return None
    
    def get_job_by_id(self, id: int) -> 'JobState':
        for j in self.job_states:
            if j.id == id:
                return j
        return None

    def get_station(self, station: 'StationState'=None) -> 'StationState':
        if station:
            return self.all_stations.get(station.id)
        return None
    
    def std_time(self, time: int, min: int, max: int):
        if max == min:
            return 0.0
        return 1.0 * (time - min) / (max - min)
    
    def check_location(position: Position, location: str) -> float:
        return float(getattr(position, "position_type", None) == location)

    def to_hyper_graph(self, last_job_in_pos: int, current_time: int):
        data = HeteroData()

        job_features: list = []
        min_time: int       = -1
        max_time: int       = -1
        job_machine_1: int  = -1
        job_machine_2: int  = -1
        job_robot:     int  = -1
        nb_first_op_m1: int = 0
        nb_last_op_m1: int  = 0
        nb_first_op_m2: int = 0
        nb_last_op_m2: int  = 0
        job_station_1: int  = -1
        job_station_2: int  = -1
        job_station_3: int  = -1
        poss_jobs_s1: list  = []
        poss_jobs_s2: list  = []
        poss_jobs_s3: list  = []

        # 1. Create job features
        for j in self.job_states:
            if not j.is_done() or j.id == last_job_in_pos:
                poss_jobs_s2.append(j.id)
                if not j.job.big:
                    poss_jobs_s1.append(j.id)
                    poss_jobs_s3.append(j.id)
                cs1, cs2, cs3 = 0.0, 0.0, 0.0
                if j.current_station.id == STATION_1:
                    cs1 = 1.0
                    job_station_1 = j.id
                elif j.current_station.id == STATION_2:
                    cs2 = 1.0
                    job_station_2 = j.id
                elif j.current_station.id == STATION_3:
                    cs3 = 1.0
                    job_station_3 = j.id
                m1, m2, robot = 0.0, 0.0, 0.0
                if self.check_location(j.location, POS_PROCESS_1):
                    m1 = 1.0
                    job_machine_1 = j.id
                    if j.id != last_job_in_pos:
                        robot = 1.0
                        job_robot = j.id
                elif self.check_location(j.location, POS_PROCESS_2):
                    m2 = 1.0
                    job_machine_2 = j.id
                    robot = 1.0
                    job_robot = j.id
                machine_1_is_first: float = float(j.operation_states[0].operation.type == PROCEDE_1)
                is_pos: float             = float(j.id == last_job_in_pos)
                remaining_time_dd: int    = current_time - j.job.due_date
                min_time                  = min(min_time, abs(remaining_time_dd)) if min_time >= 0 else abs(remaining_time_dd)
                max_time                  = max(min_time, abs(remaining_time_dd)) if max_time >= 0 else abs(remaining_time_dd)
                min_time                  = min(min_time, j.job.pos_time)
                max_time                  = max(min_time, j.job.pos_time)
                job_features.append([
                        float(j.job.big),                             # 0. Is it a big job that can only use station 2?
                                                                      # 1. remaining time in machine 1
                                                                      # 2. remaining time in machine 2
                        machine_1_is_first,                           # 3. machine 1 before machine 2?
                        float(j.job.pos_time),                        # 4. Time to place the job on the poisitioner
                        float(remaining_time_dd),                     # 5. Remaining time before due date (or current delay)
                        cs1,                                          # 6. Is it loaded in station 1?
                        cs2,                                          # 7. Is it loaded in station 2?
                        cs3,                                          # 8. Is it loaded in station 3?
                        m1,                                           # 9. Hold by robot at machine 1?
                        m2,                                           # 10. Hold by robot at machine 2?
                        self.check_location(j.location, POS_STATION), # 11. Is the job on the stations?
                        is_pos])                                      # 12. Is the job on the positionner 
                
        # I. Create job features as an array
            # Information of the job in general
            # 0. Is it a big job? (0||1)
            # 1. remaining time in machine 1 (0->1) [we maintain a min and max variable]
            # 2. remaining time in machine 2 (0->1) [we use the same variable]
            # 3. machine 1 before machine 2? (0||1)
            # 4. poisition time (0->1) [we use the same variable as remaining time]
            # 5. time before due date (or delay), dd - use current_time (0->1) [we use the same variable as remaining time]

            # Information on the current state
            # 6. Is it loaded in station 1? (0||1)
            # 7. Is it loaded in station 2? (0||1)
            # 8. Is it loaded in station 3? (0||1)
            # 9. Is it hold by robot at machine 1? (0||1)
            # 10. Is it hold by robot at machine 1? (0||1)
            # 11. Is it on the positionner? (0||1)

            # --> While creating, count numbers of first and last operations in the two machines (4 variables)
            # --> While creating, get the id of the job in the robot
            # --> While creating, get the id of the job in the machine 1 and machine 2

        # II. create station features as an array
            # 0. can process big? (0||1)
            # 1. remaining time before free, max(0, free_at - current_time) [we use the same variable as remaining time]

        # III. create links between stations and jobs
            # 0. could be chosen link without feature
            # 1. has been chosen link without feature
        
        # IV. create machine features as an array
            # 0. remaining numbers of first operations (0->1) [std with the current number of operations]
            # 1. remaining numbers of last operations (0->1) [std with the current number of operations]
            # 2. remaining time before free, max(0, free_at - current_time) [we use the same variable as remaining time]

        # V. create links between jobs and machines
            # 0. job need machine without feature
            # 1. job is in machine without feature
 
        # VI. create robot features as an array
            # 0. is machine 1? (0||1)
            # 1. is machine 2? (0||1)
            # 2. is stations? (0||1)
            # 3. remaining time before free, max(0, free_at - current_time) [we use the same variable as remaining time]

        # VII. create one job and the robot link without feautre

        # VIII. STD job features 1, 2, 4, 5 with min and max + station feature 1 + machine feature 2 + robot feature 3
        # IX. transform arrays as node tensors
        data["job"].x = torch.tensor(job_features, dtype=torch.float)
            



        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        is_free_robot = float(self.robot.current_job is None)
        data["robot"].x = torch.tensor([[
            getattr(self.robot.location, "position_type", None) == POS_PROCESS_1,
            getattr(self.robot.location, "position_type", None) == POS_PROCESS_2,
            getattr(self.robot.location, "position_type", None) == POS_STATION,
            self.robot.free_at / self.cmax]], dtype=torch.float)

        # --- Procédés 1 --------------------------------------------------------------
        is_free_proc1 = float(self.process1.current_job is None)
        data["proc_1"].x = torch.tensor([
            [is_free_proc1, 
             self.process1.free_at] ])

        # --- Procédés 2 --------------------------------------------------------------
        is_free_proc2 = float(self.process2.current_job is None)
        data["proc_2"].x = torch.tensor([
            [is_free_proc2, 
            self.process2.free_at] ])

        # --- Stations --------------------------------------------------------------
        stations = [self.all_stations.get(i) for i in range(3)]
        data["station"].x = torch.tensor([[s.accept_big, 
                                        s.current_job is not None, 
                                        s.free_at]for s in stations])

        # --- job --------------------------------------------------------------
        job_x, id_map = [], {}                    # id logique ➜ index PyG
        for idx, job in enumerate(self.job_states):
            id_map[job.id] = idx
            j = job

            first_op = j.operation_states[0]
            for op in j.operation_states:
                if op.start:                
                    first_op = op
                    break
            start_time = first_op.start

            last_op = j.operation_states[-1]
            if last_op.remaining_time == 0:
                end_time = last_op.end
            else:
                end_time = 0.0
                        
            job_x.append([
                float(j.job.big),
                float(j.job.due_date),
                float(getattr(j.location, "position_type", None) == POS_PROCESS_1),
                float(getattr(j.location, "position_type", None) == POS_PROCESS_2),
                float(getattr(j.location, "position_type", None) == POS),
                float(j.job.blocked),
                float(j.is_done()),
                float(j.job.pos_time),
                start_time,
                end_time
            ])
        data["job"].x = torch.tensor(job_x)

        # ========= 2. EDGES ==========================================================
        
        # (1) job ─operation→ proc --------------------------------------------------
        src, dest, rem, next, curr = [], [], [], [], []

        j_idx = id_map[j.id]
        ops_proc1 = sum(op.operation.type == 1 for op in j.operation_states)
        ops_proc2 = sum(op.operation.type == 2 for op in j.operation_states)

        if ops_proc1:
            src.append(j_idx); dest.append(0); rem.append(float(ops_proc1))
        if ops_proc2:
            src.append(j_idx); dest.append(1); rem.append(float(ops_proc2))

        if src:
            data["job", "operation_line", "proc"].edge_index = torch.tensor([src, dest])
            data["job", "operation_line", "proc"].edge_attr  = torch.tensor(
                rem, dtype=torch.float).unsqueeze(-1)

        # (2) job ─possible_load→ station -------------------------------------------
        for j in self.job_states:
            j_idx = id_map[j.id]
            for s_idx, st in enumerate(stations):
                if j.is_big() and not st.accept_big:
                    continue
                src.append(j_idx); dest.append(s_idx)

        if src:
            data["job", "possible_load_line", "station"].edge_index = torch.tensor(
                [src, dest], dtype=torch.long
            )
            zeros = torch.zeros((len(src), 2), dtype=torch.float)
            data["job", "possible_load_line", "station"].edge_attr = zeros

        # (3) robot ─hold→ job  ------------------------------------------------------
        r = self.robot
        hold_line = (
            r.current_job is not None
            and getattr(r.location, "position_type", None) == POS_PROCESS_1
        )

        if hold_line and r.current_job.id in id_map:
            data["robot", "hold_line", "job"].edge_index = torch.tensor([[0], [id_map[r.current_job.id]]], dtype=torch.long)
            # un seul attribut binaire « is_holding » fixé à 1.0
            data["robot", "hold_line", "job"].edge_attr  = torch.tensor([[1.0]], dtype=torch.float)

        return data

@dataclass
class RobotState:
    def __init__(self, init_position: Position):
        self.free_at: int           = 0
        self.current_job: JobState  = None
        self.location: Position     = init_position
        self.calendar: Calendar     = Calendar()

    def clone(self) -> 'RobotState':
        r: RobotState = RobotState(self.location)
        r.free_at = self.free_at
        return r

    def clone_calendar_and_current_job_and_location(self, new_state: State):
        new_state.robot.current_job = new_state.get_job(self.current_job)
        new_state.robot.calendar    = self.calendar.clone(new_state)
        new_state.robot.location    = new_state.process1 if self.location.position_type == POS_PROCESS_1 \
                                        else new_state.process2 if self.location.position_type == POS_PROCESS_2 \
                                        else new_state.all_stations

@dataclass
class Stations:
    def __init__(self, nb_stations: int=3, station_large: int=1):
        self.stations: list[StationState]       = [StationState(id, big=False) for id in range(nb_stations)]
        self.stations[station_large].accept_big = True
        self.position_type: int                 = POS_STATION

    def get(self, id: int) -> 'StationState':
        return self.stations[id]
    
    def get_possible_stations(self, big_only: bool) -> list['StationState']:
        return [s for s in self.stations if s.accept_big or not big_only]
    
    def clone(self) -> 'Stations':
        s: Stations     = Stations()
        s.position_type = self.position_type
        s.stations      = [station.clone() for station in self.stations]
        return s
    
    def clone_calendar_and_current_jobs(self, new_state: 'State'):
        for s in self.stations:
            new_station: StationState = new_state.get_station(s)
            s.clone_calendar_and_current_job(new_state, new_station)

@dataclass
class StationState:
    def __init__(self, id: int, big: bool):
        self.id: int                = id
        self.accept_big: bool       = big
        self.free_at: int           = 0
        self.current_job: JobState  = None
        self.calendar: Calendar     = Calendar()

    def clone(self) -> 'StationState':
        s: StationState = StationState(self.id, self.accept_big)
        s.free_at       = self.free_at
        s.accept_big    = self.accept_big
        return s
    
    def clone_calendar_and_current_job(self, new_state: 'State', new_station: 'StationState'):
        new_station.current_job   = new_state.get_job(self.current_job)
        new_station.calendar      = self.calendar.clone(new_state)

@dataclass
class Process:
    def __init__(self):
        self.free_at: int           = 0
        self.current_job: JobState  = None
        self.calendar: Calendar     = Calendar()
        self.position_type: int     = -1

@dataclass
class Process1(Process):
    def __init__(self):
        super().__init__()
        self.pos_is_full: bool  = False
        self.position_type: int = POS_PROCESS_1
        
    def clone(self) -> 'Process1':
        p1: Process1     = Process1()
        p1.pos_is_full   = self.pos_is_full
        p1.position_type = self.position_type
        return p1

    def clone_calendar_and_current_job(self, new_state: 'State'):
        new_state.process1.current_job = new_state.get_job(self.current_job)
        new_state.process1.calendar    = self.calendar.clone(new_state)

@dataclass
class Process2(Process):
    def __init__(self):
        super().__init__()
        self.position_type: int = POS_PROCESS_2

    def clone(self) -> 'Process2':
        p2: Process2     = Process2()
        p2.position_type = self.position_type
        return p2
    
    def clone_calendar_and_current_job(self, new_state: 'State'):
        new_state.process2.current_job = new_state.get_job(self.current_job)
        new_state.process2.calendar    = self.calendar.clone(new_state)

@dataclass
class JobState:
    def __init__(self, state: State, id: int, stations: list['StationState']=None, robot: RobotState=None, job: Job=None, build_operations: bool=True):
        self.id: int                       = id
        self.job: Job                      = job
        self.status: int                   = NOT_YET
        self.location: Position            = None
        self.end: int                      = 0
        self.delay: int                    = 0
        self.calendar: Calendar            = Calendar() 
        self.current_station: StationState = None
        self.operation_states: list[OperationState] = []
        if build_operations:
            self.operation_states: list[OperationState] = [OperationState(id=id, job=self, operation=op) for id, op in enumerate(job.operations)] if job else []

    def get_last_executed_operation(self) -> 'OperationState':
        for o in reversed(self.operation_states):
            if o.remaining_time == 0:
                return o
        return None
    
    def is_big(self) -> bool:
        return self.job.big == 1

    def clone(self, state: State) -> 'JobState':
        j: JobState        = JobState(state=state, id=self.id, job=self.job, build_operations=False)
        j.status           = self.status
        j.end              = self.end
        j.delay            = self.delay
        j.operation_states = [op.clone(j) for op in self.operation_states]
        return j
    
    def clone_calendar_and_location_and_current_station(self, new_state: 'State', new_job: 'JobState'):
        new_job.current_station = new_state.get_station(self.current_station)
        new_job.calendar        = self.calendar.clone(new_state)
        if self.location:
            new_job.location = new_state.process1 if self.location.position_type == POS_PROCESS_1 \
                                else new_state.process2 if self.location.position_type == POS_PROCESS_2 \
                                else new_state.all_stations if self.location.position_type == POS_STATION \
                                else None

    def get_operation(self, id: int) -> 'OperationState':
        for o in self.operation_states:
            if o.id == id:
                return o
        return None
    
    def is_done(self) -> bool:
        for o in self.operation_states:
            if o.remaining_time > 0:
                return False
        return True

@dataclass
class OperationState:
    def __init__(self, id: int, job: 'JobState', operation: Operation=None):
        self.operation: Operation = operation
        self.id: int              = id
        self.start: int           = 0
        self.end: int             = 0
        self.is_last: bool        = self.id == len(job.job.operations) - 1
        self.remaining_time: int  = operation.processing_time if operation else 0
        self.status: int          = NOT_YET
        self.job: JobState        = job

    def clone(self, job: 'JobState') -> 'OperationState':
        o: OperationState = OperationState(self.id, job, self.operation)
        o.remaining_time  = self.remaining_time
        o.status          = self.status
        o.start           = self.start
        o.end             = self.end
        return o

    def get_target_process(self, state: 'State') -> Process:
        return state.processess[self.operation.type]