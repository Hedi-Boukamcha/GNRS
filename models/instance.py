from conf import *
import json

# #####################################
# =*= INSTANCE DATA FOR MATH SOLVER =*=
# #####################################
__author__  = "Hedi Boukamcha; Anas Neumann"
__email__   = "hedi.boukamcha.1@ulaval.ca; anas.neumann@polymtl.ca"
__version__ = "1.0.0"
__license__ = "MIT"

class Operation:
    def __init__(self, type: int = 0, processing_time: int = 0):
        self.type = type
        self.processing_time = processing_time
    
    def __str__(self):
        return f"{{'type':{self.type}, 'processing_time':{self.processing_time}}}"

class Job:
    def __init__(self, big: int = 0, due_date: int = 0, pos_time: int = 0, operations: list[Operation] = [], status: int = 0, blocked: int = 0):
        self.operations: list[Operation] = operations
        self.big: int = big
        self.due_date: int = due_date
        self.pos_time: int = pos_time
        self.status: int = status
        self.blocked: int = blocked
    
    def __str__(self):
        return f"{{'big':{self.big}, 'due_date':{self.due_date}, 'pos_time':{self.pos_time}, 'status':{self.status}, 'blocked':{self.blocked}, 'operations':{[o.__str__() for o in self.operations]}}}"

class Instance:
    def __init__(self, jobs: list[Job] = []):
        self.jobs: list[Job] = jobs

    def __str__(self):
        return f"{self.jobs}"

    def load(path: str): 
        with open(path, 'r') as f:
            _data = json.load(f)
        jobs = []
        for job_data in _data["jobs"]:
            operations = [Operation(type=op["type"], processing_time=op["processing_time"]) for op in job_data["operations"]]
            job = Job(big=job_data["big"], due_date=job_data["due_date"], pos_time=job_data["pos_time"], operations=operations, status=job_data["status"], blocked=job_data["blocked"])
            jobs.append(job)
        return Instance(jobs=jobs)
    
    def display(self):
        for j in self.jobs:
            print(j)

class MathSolution:
    def __init__(self):
        self.entry_station_date = [], []
        self.exe_start = [], []
        self.job_loaded = [], []
        self.exe_parallel = [], []
        self.job_unload = [], []
        self.C_max = None,
        self.end_j = []
        self.end_o = []
        self.free_j = []
        self.free_o = []
        self.delay = []
        self.exe_mode = [], [], []
        self.exe_before = [], [], [], []

class MathInstance:
    def __init__(self, jobs: list[Job]):
        self.s = MathSolution()
        self.nb_jobs: int = len(jobs)
        self.nb_types: int = 2
        self.nb_stations: int = NB_STATIONS
        self.nb_modes: int = 3
        self.has_history: bool = False
        self.L = L 
        self.M = M 
        self.I = 0 #10_000

        self.lp = [job.big for job in jobs]
        self.operations_by_job = [len(job.operations) for job in jobs]
        self.due_date = [job.due_date for job in jobs]
        self.pos_j = [job.pos_time for job in jobs]

        self.needed_proc = [[[0 for _ in range(self.nb_types)] for _ in job.operations] for job in jobs]
        self.welding_time = [[0 for _ in job.operations] for job in jobs]
        self.job_station = [[0 for _ in range(self.nb_stations)] for _ in jobs]
        self.job_modeB = [0 for _ in jobs]
        self.job_robot = [0 for _ in jobs]
        for j, job in enumerate(jobs):
            self.job_modeB[j] = 1 if (job.status == 3) else 0
            self.job_robot[j] = 1 if (job.status == 2) else 0
            if job.status > 0:
                self.has_history = True
                self.job_station[j][job.blocked-1] = 1
            for o, operation in enumerate(job.operations):
                self.needed_proc[j][o][operation.type-1] = 1  
                self.welding_time[j][o] = operation.processing_time
                self.I += operation.processing_time + self.pos_j[j] + 3*self.M + 2*self.L

    def loop_modes(self):
        return range(self.nb_modes)

    def loop_stations(self):
        return range(self.nb_stations)

    def loop_jobs(self):
        return range(self.nb_jobs)
    
    def last_operations(self, j: int):
        return self.operations_by_job[j] - 1

    def loop_operations(self, j: int, exclude_first: bool =False):
        return range(1, self.operations_by_job[j]) if exclude_first else range(self.operations_by_job[j])