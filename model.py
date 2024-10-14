from ortools.sat.python import cp_model
import numpy as np

PROCEDE_1: int = 0
PROCEDE_2: int = 1

PROCEDE_1_SEQ_MODE_A: int = 0
PROCEDE_1_PARALLEL_MODE_B: int = 1
PROCEDE_2_MODE_C: int = 2

STATION_1: int = 0
STATION_2: int = 1
STATION_3: int = 2

FIRST_OP: int = 0

class Solution:
    def __init__(self):
        self.entry_station_date = [], []
        self.exe_start = [], []
        self.job_loaded = [], []
        self.exe_parallel = [], []
        self.job_unload = [], []
        self.delay = []
        self.exe_mode = [], [], []
        self.exe_before = [], [], [], []

class Instance:
    def __init__(self, data):
        self.data = data
        self.nb_jobs = len(data)
        self.nb_types = 2
        self.nb_stations = 3
        self.nb_modes = 3
        self.has_history = False
        self.s = Solution()
        
        self.operations_by_job = [len(job['operations']) for job in self.data]
        self.needed_proc = self.initialize_needed_proc()
        self.lp = [job['big'] for job in self.data] # self.initialize_lp()
        #self.fj = self.initialize_fj()
        self.due_date = [job['due_date'] for job in self.data] # self.initialize_ddp()
        self.welding_time = self.initialize_welding_time()
        self.pos_j = [job['pos_time'] for job in self.data] #self.initialize_pos_j()
        self.L = 2  
        self.M = 3  
        self.I = self.calculate_upper_bound()
        
        self.job_station = self.initialize_job_station()
        self.job_modeB = self.initialize_job_modeB()
        self.job_robot = self.initialize_job_robot()

    def loop_modes(self):
        return range(self.nb_modes)

    def loop_stations(self):
        return range(self.nb_stations)

    def loop_jobs(self):
        return range(self.nb_jobs)

    def loop_operations(self, j, exclude_first=False):
        return range(1, self.operations_by_job[j]) if exclude_first else range(self.operations_by_job[j])

    def initialize_needed_proc(self):
        needed_proc = [[[0 for ty in range(self.nb_types)] for o in range(self.operations_by_job[j])] for j in range(self.nb_jobs)]
        for j, job in enumerate(self.data):
            for o, operation in enumerate(job['operations']):
                needed_proc[j][o][operation['type']-1] = 1  
        return needed_proc

    '''def initialize_lp(self):
        lp = []
        for job in self.data:
            lp.append(job['big'])
        return lp

    def initialize_fj(self):
        fj = []
        return fj

    def initialize_ddp(self):
        due_date = []
        for job in self.data:
            due_date.append(job['due_date'])
        return due_date'''

    def initialize_welding_time(self):
        welding_time = [[0 for _ in range(self.operations_by_job[j])] for j in range(self.nb_jobs)]
        for j, job in enumerate(self.data):
            for o, operation in enumerate(job['operations']):
                welding_time[j][o] = operation['pocessing_time']
        return welding_time
      
    '''
    def initialize_pos_j(self):
        pos_j = []
        for job in self.data:
            pos_j.append(job['pos_time'])
        return pos_j'''

    def calculate_upper_bound(self):
        I = 0
        for j, job in enumerate(self.data):
            for o, _ in enumerate(job['operations']):
                welding_time_value = self.welding_time[j][o]
                I += (welding_time_value + self.pos_j[j] + 3 * self.M + 2 * self.L)
        return I

    def initialize_job_station(self):
        job_station = [[0 for _ in range(self.nb_stations)] for _ in range(self.nb_jobs)]
        return job_station
        
    def initialize_job_modeB(self):
        job_modeB = [0 for _ in range(self.nb_jobs)]
        return job_modeB

    def initialize_job_robot(self):
        job_robot = [0 for _ in range(self.nb_jobs)]
        return job_robot














