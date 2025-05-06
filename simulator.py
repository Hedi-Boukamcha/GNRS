# -*- coding: utf-8 -*-
import sys
import json
from typing import List, Optional

M = 1.0  # Temps de mouvement
L = 2.0  # Temps de chargement/déchargement

# --- Modèles de base ---
class Operation:
    def __init__(self, proc_type: int, duration: float, mode: str = "A"):
        self.proc_type = proc_type
        self.duration = duration
        self.mode = mode

class Job:
    def __init__(self, job_id: int, big: bool, operations: List[Operation], pos_time: float = 0):
        self.id = job_id
        self.big = big
        self.operations = operations
        self.loaded = False
        self.station_id = None
        self.start_time = None
        self.end_time = None
        self.pos_time = pos_time
        self.status = "waiting" #"waiting", "running", "done"
        self.step = "not_started"  # "loading", "executing", "unloading", "done"
        self.current_op = 0

class Station:
    def __init__(self, id: int):
        self.id = id
        self.state = 0
        self.job_id = None
        self.busy_until = 0

class Robot:
    def __init__(self):
        self.position = 0
        self.free_at = 0
        self.occupied = False

    def move_to(self, dest: int):
        distance = abs(self.position - dest)
        self.position = dest
        return distance * M

class SystemState:
    def __init__(self, num_stations: int):
        self.time = 0
        self.robot = Robot()
        self.stations = [Station(i) for i in range(num_stations)]
        self.events = []

    def get_available_station(self, big: bool) -> Optional[Station]:
        for s in self.stations:
            if big and s.id == 1 and s.state == 0:
                return s
            elif not big and s.state == 0 and s.id != 1:
                return s
        return None

    def occupy_station(self, station: Station, job_id: int):
        station.state = 1
        station.job_id = job_id

    def discharge_piece(self, station: Station, mode: str):
        # D1 (mode A/C) or D2 (mode B)
        if mode in ["A", "C"]:
            delay = M + L
        elif mode == "B":
            delay = M + L + M  # D2 = D1 + M
        else:
            delay = M + L
        self.time += delay
        self.robot.free_at = self.time
        station.state = 0
        station.job_id = None
        self.robot.occupied = False

# --- Simulation par job ---
def simulate_job(job: Job, state: SystemState):
    if not job.loaded:
        # Choisir une station
        station = None
        while not station:
            station = state.get_available_station(job.big)
            if not station:
                state.time += 1
        state.time = max(state.time, state.robot.free_at)

        # Chargement : C1 ou C2
        if not state.robot.occupied:
            state.time += state.robot.move_to(station.id) + L  # C1
        else:
            state.time += (M + L) + M  # C2 = D1 or D2 + L
            state.robot.occupied = False

        state.occupy_station(station, job.id)
        job.station_id = station.id
        job.loaded = True
        job.start_time = state.time
        state.robot.occupied = True
        state.robot.free_at = state.time

    for index, op in enumerate(job.operations):
        state.time = max(state.time, state.robot.free_at)

        if index == 0:
            # E3 = E1 or E2 + L
            if not state.robot.occupied:
                state.time += state.robot.move_to(job.station_id) + L  # E1 + L
            else:
                state.time += (M + L) + M  # E2 + L
                state.robot.occupied = False
        else:
            if not state.robot.occupied:
                state.time += state.robot.move_to(job.station_id)  # E1
            else:
                state.time += (M + L) + M  # E2
                state.robot.occupied = False

        if op.mode == "B":
            state.time += job.pos_time

        state.time += op.duration
        state.robot.occupied = True
        state.robot.free_at = state.time

    # Déchargement : D1 ou D2
    station = state.stations[job.station_id]
    state.discharge_piece(station, op.mode)
    job.end_time = state.time
    state.events.append(f"J{job.id} terminé à t={state.time:.1f}")

# --- Simulation globale ---
def simulate_all_jobs(jobs: List[Job], state: SystemState):
    while any(job.status != "done" for job in jobs):
        for job in jobs:
            if job.status == "waiting":
                can_start = True  # ici on pourra vérifier station/robot dispo
                if can_start:
                    simulate_job(job, state)
        state.time += 1
    return state

# --- Chargement instance .json ---
def load_instance_as_jobs(path: str) -> List[Job]:
    with open(path) as f:
        data = json.load(f)

    jobs = []
    for j_id, job_data in enumerate(data):
        big = job_data["big"]
        pos_time = job_data["pos_time"]
        operations = []
        for op in job_data["operations"]:
            proc_type = op["type"]
            duration = op["processing_time"]
            mode = "C" if proc_type == 2 else ("B" if pos_time > 0 else "A")
            operations.append(Operation(proc_type, duration, mode))
        jobs.append(Job(j_id, big, operations, pos_time))
    return jobs

# --- Interface manuelle ---
def simulator(instance_path: str):
    jobs = load_instance_as_jobs(instance_path)
    state = SystemState(num_stations=3)
    simulate_all_jobs(jobs, state)
    print("\nRésultats de simulation :")
    for job in jobs:
        print(f"J{job.id} : début à t={job.start_time:.1f}, fin à t={job.end_time:.1f}, station S{job.station_id}")
    print("\nÉvénements du système :")
    for e in state.events:
        print("•", e)

# --- Exécution CLI ---
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage : python simulator.py <instance.json>")
        exit(1)
    simulator(sys.argv[1])
