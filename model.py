from ortools.sat.python import cp_model
import numpy as np


class Solution:
    def __init__(self):
        self.entry_station_date, self.exe_start, self.job_loaded , self.exe_parallel, self.job_unload = [], []
        self.delay = []
        self.exe_mode = [], [], []
        self.exe_before = [], [], [], []


class Instance:
    def __init__(self, data, nombre_jobs, operations_by_job, types):
        self.data = data
        self.nombre_jobs = nombre_jobs
        self.operations_by_job = operations_by_job
        self.types = types
        
        self.needed_proc = self.initialize_needed_proc()
        self.lp = self.initialize_lp()
        self.fj = self.initialize_fj()
        self.ddp = self.initialize_ddp()
        self.welding_time = self.initialize_welding_time()
        self.pos_j = self.initialize_pos_j()
        self.L = 2  
        self.M = 3  
        self.I = self.calculate_upper_bound()


    def initialize_needed_proc(self):
        needed_proc = [[[0 for _ in range(len(self.types))] for _ in range(self.operations_by_job[j])] for j in range(self.nombre_jobs)]
        for i, job in enumerate(self.data):
            for j, operation in enumerate(job['operations']):
                type_value = operation['type']
                if type_value == 1:
                    needed_proc[i][j][0] = [1, 0]
                elif type_value == 2:
                    needed_proc[i][j][1] = [0, 1]    
        return needed_proc


    def initialize_lp(self):
        lp = []
        for job in self.data:
            lp.append(job['big'])
        return lp


    def initialize_fj(self):
        return [0 for _ in range(self.nombre_jobs)]


    def initialize_ddp(self):
        ddp = []
        for job in self.data:
            ddp.append(job['due_date'])
        return ddp


    def initialize_welding_time(self):
        welding_time = [[0 for _ in range(self.operations_by_job[j])] for j in range(self.nombre_jobs)]
        return welding_time


    def initialize_pos_j(self):
        pos_j = []
        for job in self.data:
            pos_j.append(job['pos_time'])
        return pos_j


    def calculate_upper_bound(self):
        I = 0
        for j, job in enumerate(self.data):
            for o, _ in enumerate(job['operations']):
                welding_time_value = self.welding_time[j][o]
                I += (welding_time_value + self.pos_j[j] + 3 * self.M + 2 * self.L)
        return I

        




















