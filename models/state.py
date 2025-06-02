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
    
    def std_column(self, graph: HeteroData, node_type: str, feature_idx: int, min: int, max: int):
        old_value = graph[node_type].x[:, feature_idx]
        graph[node_type].x[:, feature_idx] = 1.0 * (old_value - min) / (max - min)

    def fix_column(self, graph: HeteroData, node_type: str, feature_idx: int):
        old_value = graph[node_type].x[:, feature_idx]
        graph[node_type].x[:, feature_idx] = (old_value > 0).float()
    
    def check_location(self, position: Position, location: str) -> float:
        if position is None:
            return 0.0
        return float(position.position_type == location)

    def to_hyper_graph(self, last_job_in_pos: int, current_time: int) -> HeteroData:
        graph = HeteroData()
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
        poss_jobs_m1: list  = []
        poss_jobs_m2: list  = []

        # 1. Create job features
        job_features: list = []
        for j in self.job_states:
            if not j.is_done() or j.id == last_job_in_pos:
                poss_jobs_s2.append(j.id)
                if not j.job.big:
                    poss_jobs_s1.append(j.id)
                    poss_jobs_s3.append(j.id)
                cs1, cs2, cs3 = 0.0, 0.0, 0.0
                if (j.current_station is not None) and (j.current_station.id == STATION_1):
                    cs1 = 1.0
                    job_station_1 = j.id
                elif (j.current_station is not None) and  j.current_station.id == STATION_2:
                    cs2 = 1.0
                    job_station_2 = j.id
                elif (j.current_station is not None) and  j.current_station.id == STATION_3:
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
                remaining_time_dd: int    = float(current_time - j.job.due_date)
                min_time                  = min(min_time, abs(remaining_time_dd)) if min_time >= 0 else abs(remaining_time_dd)
                max_time                  = max(max_time, abs(remaining_time_dd)) if max_time >= 0 else abs(remaining_time_dd)
                min_time                  = min(min_time, j.job.pos_time)
                max_time                  = max(max_time, j.job.pos_time)
                remaining_time_m1: float  = 0.0
                remaining_time_m2: float  = 0.0
                for idx, o in enumerate(j.operation_states):
                    if o.operation.type == PROCEDE_1:
                        if o.remaining_time > 0:
                            poss_jobs_m1.append(j.id)
                        remaining_time_m1 = o.remaining_time
                        min_time          = min(min_time, remaining_time_m1)
                        max_time          = max(max_time, remaining_time_m1)
                        if idx == 0:
                            nb_first_op_m1 += 1
                        if idx == len(j.operation_states) -1:
                            nb_last_op_m1  += 1
                    else:
                        if o.remaining_time > 0:
                            poss_jobs_m2.append(j.id)
                        remaining_time_m2 = o.remaining_time
                        min_time          = min(min_time, remaining_time_m2)
                        max_time          = max(max_time, remaining_time_m2)
                        if idx == 0:
                            nb_first_op_m2 += 1
                        if idx == len(j.operation_states) -1:
                            nb_last_op_m2  += 1
                job_features.append([
                        float(j.job.big),                             # 0. Is it a big job that can only use station 2?
                        remaining_time_m1,                            # 1. remaining time in machine 1
                        remaining_time_m2,                            # 2. remaining time in machine 2
                        machine_1_is_first,                           # 3. machine 1 before machine 2?
                        float(j.job.pos_time),                        # 4. Time to place the job on the poisitioner
                        remaining_time_dd,                            # 5. Remaining time before due date (or current delay)
                        cs1,                                          # 6. Is it loaded in station 1?
                        cs2,                                          # 7. Is it loaded in station 2?
                        cs3,                                          # 8. Is it loaded in station 3?
                        m1,                                           # 9. Hold by robot at machine 1?
                        m2,                                           # 10. Hold by robot at machine 2?
                        self.check_location(j.location, POS_STATION), # 11. Is the job on the stations?
                        is_pos])                                      # 12. Is the job on the positionner?
        graph["job"].x = torch.tensor(job_features, dtype=torch.float)

        # II. create station features
        station_features: list = []
        for s in self.all_stations.stations:
               time_before_free: float = max(0.0, s.free_at - current_time)
               min_time                = min(min_time, time_before_free)
               max_time                = max(max_time, time_before_free)
               station_features.append([float(s.accept_big),         # 0. Can this station process big jobs?
                                        time_before_free])           # 1. Estimated (or real) remaining time before free
        graph["station"].x = torch.tensor(station_features, dtype=torch.float)
        
        # III. create links between stations and jobs
        scj_src: list = []
        scj_dest: list = []
        slj_src: list = []
        slj_dest: list = []
        for s in self.all_stations.stations:
            jobs = poss_jobs_s1 if s.id == STATION_1 else poss_jobs_s2 if s.id == STATION_2 else poss_jobs_s3
            loaded_job = job_station_1 if s.id == STATION_1 else job_station_2 if s.id == STATION_2 else job_station_3
            for j in jobs:
                scj_src.append(s.id)
                scj_dest.append(j)  
            if loaded_job >= 0: 
                slj_src.append(s.id)
                slj_dest.append(loaded_job)  
        graph["station", "can_load", "job"].edge_index = torch.tensor([scj_src, scj_dest], dtype=torch.long)
        graph["job", "could_be_loaded", "station"].edge_index = graph["station", "can_load", "job"].edge_index.flip(0)
        if slj_src:
            graph["station", "loaded", "job"].edge_index = torch.tensor([slj_src, slj_dest], dtype=torch.long)
            graph["job", "loaded_in", "station"].edge_index = graph["station", "loaded", "job"].edge_index.flip(0)

        # IV. create machine features
        machine_features: list = []
        time_before_free_m1: float = max(0.0, self.process1.free_at - current_time)
        min_time                   = min(min_time, time_before_free_m1)
        max_time                   = max(max_time, time_before_free_m1)
        machine_features.append([nb_first_op_m1,                    # 0. remaining numbers of first operations
                                 nb_last_op_m1,                     # 1. remaining numbers of last operations
                                 time_before_free_m1])              # 2. Estimated (or real) remaining time before free
        time_before_free_m2: float = max(0.0, self.process2.free_at - current_time)
        min_time                   = min(min_time, time_before_free_m2)
        max_time                   = max(max_time, time_before_free_m2)
        machine_features.append([nb_first_op_m2,                    # 0. remaining numbers of first operations
                                 nb_last_op_m2,                     # 1. remaining numbers of last operations
                                 time_before_free_m2])              # 2. Estimated (or real) remaining time before free
        graph["machine"].x = torch.tensor(machine_features, dtype=torch.float)

        # V. create edges between jobs and machines
        mnj_src: list = []
        mnj_dest: list = []
        for j in poss_jobs_m1:
            mnj_src.append(PROCEDE_1)
            mnj_dest.append(j)
        for j in poss_jobs_m2:
            mnj_src.append(PROCEDE_2)
            mnj_dest.append(j)
        graph["machine", "will_execute", "job"].edge_index = torch.tensor([mnj_src, mnj_dest], dtype=torch.long)
        graph["job", "needs", "machine"].edge_index        = graph["machine", "will_execute", "job"].edge_index.flip(0)
        mej_src: list = []
        mej_dest: list = []
        if job_machine_1 >= 0:
            mej_src.append(PROCEDE_1)
            mej_dest.append(job_machine_1)
        if job_machine_2 >= 0:
            mej_src.append(PROCEDE_2)
            mej_dest.append(job_machine_2)
        if mej_src:
            graph["machine", "execute", "job"].edge_index     = torch.tensor([mej_src, mej_dest], dtype=torch.long)
            graph["job", "executed_by", "machine"].edge_index = graph["machine", "execute", "job"].edge_index.flip(0)
 
        # VI. create robot features
        robot_features: list = []
        time_before_free: float = max(0.0, self.robot.free_at - current_time)
        min_time                = min(min_time, time_before_free)
        max_time                = max(max_time, time_before_free)
        robot_features.append([self.check_location(self.robot.location, POS_STATION),   # 0. Is the robot on the stations?
                               self.check_location(self.robot.location, POS_PROCESS_1), # 1. Is the robot on machine 1
                               self.check_location(self.robot.location, POS_PROCESS_2), # 2. Is the robot on machine 2
                               time_before_free])                                       # 3. Estimated (or real) remaining time before free
        graph["robot"].x = torch.tensor(robot_features, dtype=torch.float)

        # VII. create the links between the robot and the job it holds
        if job_robot >= 0:
            graph["robot", "hold", "job"].edge_index = torch.tensor([[0], [job_robot]], dtype=torch.long)
            graph["job", "hold_by", "robot"].edge_index = graph["robot", "hold", "job"].edge_index.flip(0)

        # VIII. Standardize time-related features
        if max_time > min_time:
            self.std_column(graph=graph, node_type="job", feature_idx=1, min=min_time, max=max_time)
            self.std_column(graph=graph, node_type="job", feature_idx=2, min=min_time, max=max_time)
            self.std_column(graph=graph, node_type="job", feature_idx=4, min=min_time, max=max_time)
            self.std_column(graph=graph, node_type="job", feature_idx=5, min=min_time, max=max_time)
            self.std_column(graph=graph, node_type="station", feature_idx=1, min=min_time, max=max_time)
            self.std_column(graph=graph, node_type="machine", feature_idx=2, min=min_time, max=max_time)
            self.std_column(graph=graph, node_type="robot", feature_idx=3, min=min_time, max=max_time)
        else:
            self.fix_column(graph=graph, node_type="job", feature_idx=1)
            self.fix_column(graph=graph, node_type="job", feature_idx=2)
            self.fix_column(graph=graph, node_type="job", feature_idx=4)
            self.fix_column(graph=graph, node_type="job", feature_idx=5)
            self.fix_column(graph=graph, node_type="station", feature_idx=1)
            self.fix_column(graph=graph, node_type="machine", feature_idx=2)
            self.fix_column(graph=graph, node_type="robot", feature_idx=3)
        return graph

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