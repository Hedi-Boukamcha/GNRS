from dataclasses import dataclass
from typing import Union
from models.instance import Operation, Job, Instance
from conf import *
from torch import Tensor

# ######################################################
# =*= GNN STATE: INSTANCE DATA AND PARTIAL SOLUTION =*=
# #####################################################
__author__ = "Hedi Boukamcha - hedi.boukamcha.1@ulaval.ca"
__version__ = "1.0.0"
__license__ = "MIT"

# Events and Decision data ---------------------------------------------------------------------------------------
Position = Union['Process1', 'Process2', 'Stations']

@dataclass
class Decision: 
    def __init__(self, job_id: int, operation_id: int, parallel: bool = False):
        self.job_id: int       = job_id
        self.operation_id: int = operation_id
        self.parallel: bool    = parallel

    def clone(self) -> 'Decision':
        return Decision(self.job_id, self.operation_id, self.parallel)

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
        e: Event    = Event(self.start, self.end, self.event_type)
        e.job       = job
        e.source    = source
        e.dest      = dest
        e.operation = operation
        e.station   = station
        return e

    def duration(self) -> int:
        return self.end - self.start

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
    
    def search(self, type: int=-1, job_id: int=-1, source: Position=None, dest: Position=None) -> tuple[int, Event]:
        for rev_i, e in enumerate(reversed(self.events)):
            if (type<0 or e.event_type==type) and (job_id<0 or e.job.id==job_id) and (source is None or e.source==source) and (dest is None or e.dest==dest):
                position: int = len(self.events) - 1 - rev_i
                return position, e
        return -1, e
    
    def search_and_delete(self, type: int=-1, job_id: int=-1, source: Position=None, dest: Position=None):
        i, _ = self.search(type, job_id, source, dest)
        if i>=0:
            del self.events[i]

    def get(self, i: int) -> 'Event':
        return self.events[i]
    
    def add(self, e: Event):
        self.events.append(e)

    def len(self):
        return len(self.events)
    
    def has_events(self) -> bool:
        return self.len()>0

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
        self.job_sates: list[JobState] = []
        self.processess: list[Process] = []
        if automatic_build:
            self.process1      = Process1()
            self.process2      = Process2()
            self.processess    = [self.process1, self.process2]
            self.all_stations  = Stations(nb_stations=nb_stations, station_large=station_large)
            self.robot         = RobotState(init_position=self.all_stations)
            self.job_sates     = [JobState(id, self.all_stations.stations, self.robot, job) for id, job in enumerate(i.jobs)]

    def clone(self) -> 'State':
        # Stage 1: clone without links
        c: State       = State(self.i, self.M, self.L, automatic_build=False)
        c.decisions    = [d.clone() for d in self.decisions]
        c.process1     = self.process1.clone()
        c.process2     = self.process2.clone()
        c.processess   = [c.process1, c.process2]
        c.reward       = self.reward.clone()
        c.robot        = self.robot.clone()
        c.job_sates    = [j.clone() for j in self.job_sates]
        c.all_stations = self.all_stations.clone()

        # State 2: clone all OOP links
        self.robot.clone_calendar_and_current_job_and_location(c)
        self.process1.clone_calendar_and_current_job(c)
        self.process2.clone_calendar_and_current_job(c)
        self.all_stations.clone_calendar_and_current_jobs(c)
        for j in self.job_sates:
            new_job: JobState = c.get_job(j)
            j.clone_calendar_and_location_and_current_station(c, new_job)
        return c
    
    def get_job(self, job: 'JobState'=None) -> 'JobState':
        if job:
            return self.get_job_by_id(job.id)
        return None
    
    def get_job_by_id(self, id: int) -> 'JobState':
        for j in self.job_sates:
            if j.id == id:
                return j
        return None

    def get_station(self, station: 'StationState'=None) -> 'StationState':
        if station:
            return self.all_stations.get(station.id)
        return None

    def to_hyper_graph(self):
        # TODO translate state into hyper graph (Pytorch Geometric HeteroData)
        pass

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

@dataclass
class Stations:
    def __init__(self, nb_stations: int=0, station_large: int=0):
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
        s: StationState = StationState(self.id, self.big)
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
        self.pos_is_full: bool      = False
        self.position_type: int     = POS_PROCESS_1
        
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
        self.position_type: int     = POS_PROCESS_2

    def clone(self) -> 'Process2':
        p2: Process2     = Process2()
        p2.position_type = self.position_type
        return p2
    
    def clone_calendar_and_current_job(self, new_state: 'State'):
        new_state.process2.current_job = new_state.get_job(self.current_job)
        new_state.process2.calendar    = self.calendar.clone(new_state)

@dataclass
class JobState:
    def __init__(self, id: int, stations: list['StationState']=None, robot: RobotState=None, job: Job=None, build_history: bool=True):
        self.id: int                       = id
        self.job: Job                      = job
        self.status: int                   = NOT_YET
        self.location: Position            = None
        self.end: int                      = 0
        self.delay: int                    = 0
        self.calendar: Calendar            = Calendar() 
        self.current_station: StationState = None
        self.operation_states: list[OperationState] = []
        if build_history and job and job.status > 0:
            self.status = IN_SYSTEM
            robot.location = POS_PROCESS_1 if job.status == 2 and job.operations[0].type == PROCEDE_1 \
                else POS_PROCESS_2 if job.status == 2 and job.operations[0].type == PROCEDE_2 \
                else POS_STATION
            self.location = POS_STATION if job.status == 1 \
                else POS_PROCESS_1 if job.status == 3 \
                else POS_PROCESS_1 if job.status == 2 and job.operations[0].type == PROCEDE_1 \
                else POS_PROCESS_2 if job.status == 2 and job.operations[0].type == PROCEDE_2 \
                else None
            self.current_station = stations[job.blocked] if job.status > 0 else None
            self.operation_states: list[OperationState] = [OperationState(id=id, job=self, operation=op) for id, op in enumerate(job.operations)] if job else []

    def get_last_executed_operation(self) -> 'OperationState':
        for o in self.operation_states:
            if o.status == DONE and o.remaining_time == 0:
                return o
        return None

    def clone(self) -> 'JobState':
        j: JobState        = JobState(id=self.id, job=self.job, build_history=False)
        j.status           = self.status
        j.end              = self.end
        j.delay            = self.delay
        j.operation_states = [OperationState(job=j, operation=op) for op in j.job.operations] if j.job else []
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
        self.is_last: bool        = self.id == len(job.operation_states) - 1
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