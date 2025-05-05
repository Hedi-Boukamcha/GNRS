import json
import sys
from typing import List, Optional

M = 1  
L = 2  


class Operation:
    def __init__(self, proc_type: int, duration: float, mode: str = "A"):
        self.proc_type = proc_type
        self.duration = duration
        self.mode = mode  # "A", "B", "C"

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

    def move_to(self, dest: int):
        distance = abs(self.position - dest)
        time = distance * M
        self.position = dest
        return time

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

    def occupy_station(self, station: Station, job_id: int, duration: float):
        station.state = 1
        station.job_id = job_id
        station.busy_until = self.time + duration

    def discharge_piece(self, station: Station, is_large: bool):
        delay = 3 * M + L if is_large else M + L
        self.time += delay
        station.state = 0
        station.job_id = None
        self.robot.free_at = self.time


def simulate_job(job: Job, state: SystemState):
    if not job.loaded:
        station = None
        while not station:
            station = state.get_available_station(job.big)
            if not station:
                state.time += 1

        state.time = max(state.time, state.robot.free_at)
        move = state.robot.move_to(station.id)
        state.time += move + L

        state.occupy_station(station, job.id, 0)
        job.loaded = True
        job.station_id = station.id
        job.start_time = state.time


    for op in job.operations:
        move = state.robot.move_to(job.station_id)
        state.time = max(state.time, state.robot.free_at)
        state.time += move

        if op.mode == "B":
            state.time += job.pos_time

        state.time += op.duration
        state.robot.free_at = state.time

    station = state.stations[job.station_id]
    state.discharge_piece(station, job.big)
    job.end_time = state.time
    state.events.append(f"J{job.id} terminé à t={state.time:.1f}")

def simulate_all_jobs(jobs: List[Job], state: SystemState):
    for job in jobs:
        simulate_job(job, state)
    return state


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
            if proc_type == 2:
                mode = "C"
            else:
                mode = "B" if pos_time > 0 else "A"
            operations.append(Operation(proc_type, duration, mode))

        job = Job(j_id, big, operations, pos_time)
        jobs.append(job)

    return jobs


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage : python simulator.py <instance.json>")
        exit(1)

    instance_path = sys.argv[1]
    jobs = load_instance_as_jobs(instance_path)
    state = SystemState(num_stations=3)
    simulate_all_jobs(jobs, state)

    print("\nRésultats de simulation :")
    for job in jobs:
        print(f"J{job.id} : début à t={job.start_time:.1f}, fin à t={job.end_time:.1f}, station S{job.station_id + 1}")

    print("\nÉvénements du système :")
    for e in state.events:
        print("•", e)
    