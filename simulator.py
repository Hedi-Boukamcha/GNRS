# simulateur_robotique.py
from pathlib import Path
import json
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import argparse

# Constantes par défaut
M_DEFAULT = 3  # Durée de déplacement
L_DEFAULT = 2  # Temps de chargement/déchargement

# Événement générique
dataclass
class Event:
    start: int
    end: int
    event_type: str
    source: Optional[str] = None
    dest: Optional[str] = None
    job_id: Optional[int] = None
    op_id: Optional[int] = None


class Calendar:
    def __init__(self):
        self.events: List[Dict] = []

    def add_event(self, **kwargs):
        self.events.append(kwargs)

    def get_last_event(self):
        return self.events[-1] if self.events else None

    def clear_after(self, time: int):
        self.events = [e for e in self.events if e["start"] <= time]

    def next_available(self) -> int:
        return self.events[-1]["end"] if self.events else 0


class Station:
    def __init__(self, station_id: int):
        self.id = station_id
        self.calendar = Calendar()


class Robot:
    def __init__(self):
        self.calendar = Calendar()


class Positioner:
    def __init__(self):
        self.calendar = Calendar()


class Proc1:
    def __init__(self):
        self.calendar = Calendar()


class Proc2:
    def __init__(self):
        self.calendar = Calendar()


class Operation:
    def __init__(self, op_type: int, duration: int):
        self.type = op_type
        self.processing_time = duration


class Job:
    def __init__(self, job_id: int, data: Dict):
        self.id = job_id
        self.big = data.get("big", 0)
        self.due_date = data.get("due_date", 0)
        self.operations = [Operation(op["type"], op["processing_time"]) for op in data["operations"]]
        self.calendar = Calendar()


class SystemState:
    def __init__(self, jobs: List[Job]):
        self.robot = Robot()
        self.positioner = Positioner()
        self.proc1 = Proc1()
        self.proc2 = Proc2()
        self.stations = {i: Station(i) for i in [1, 2, 3]}
        self.jobs = {job.id: job for job in jobs}


class Simulator:
    def __init__(self, job_data: List[Dict], M=M_DEFAULT, L=L_DEFAULT):
        self.jobs = [Job(i, job_data[i]) for i in range(len(job_data))]
        self.state = SystemState(self.jobs)
        self.M = M
        self.L = L

    def apply_decision(self, job_id: int, op_id: int, parallel: bool, start_time: int, station_id: Optional[int]):
        job = self.state.jobs[job_id]
        op = job.operations[op_id]
        duration = op.processing_time

        if op_id == 0:
            # Chargement sur station
            station = self.state.stations[station_id]
            load_start = start_time
            load_end = load_start + self.L

            station.calendar.add_event(start=load_start, end=load_end, event_type="load", job_id=job_id)
            job.calendar.add_event(start=load_start, end=load_end, event_type="load", source=f"S{station_id}", job_id=job_id, op_id=op_id)

            # Mouvement robot vers station → positionneur ou proc1/proc2
            move_start = load_end
            move_end = move_start + self.M

            self.state.robot.calendar.add_event(start=move_start, end=move_end, source=f"S{station_id}", dest="proc", event_type="move", job_id=job_id)

            if parallel:
                # Mode B : déposer au positionneur → Proc1
                pos_start = move_end
                pos_end = pos_start + self.L
                self.state.positioner.calendar.add_event(start=pos_start, end=pos_end, event_type="pos", job_id=job_id)

                exe_start = pos_end
                exe_end = exe_start + duration
                self.state.proc1.calendar.add_event(start=exe_start, end=exe_end, event_type="execution", job_id=job_id)
                job.calendar.add_event(start=exe_start, end=exe_end, event_type="execution", dest="Proc1", job_id=job_id, op_id=op_id)
            else:
                # Mode A : robot tient → Proc1
                exe_start = move_end
                exe_end = exe_start + duration
                self.state.proc1.calendar.add_event(start=exe_start, end=exe_end, event_type="execution", job_id=job_id)
                self.state.robot.calendar.add_event(start=exe_start, end=exe_end, source="holding", dest="Proc1", event_type="hold", job_id=job_id)
                job.calendar.add_event(start=exe_start, end=exe_end, event_type="execution", dest="Proc1", job_id=job_id, op_id=op_id)

        else:
            # Étape suivante : robot tient la pièce (Mode C → Proc2)
            move_start = start_time
            move_end = move_start + self.M
            self.state.robot.calendar.add_event(start=move_start, end=move_end, source="pos", dest="proc2", event_type="move", job_id=job_id)

            exe_start = move_end
            exe_end = exe_start + duration
            self.state.proc2.calendar.add_event(start=exe_start, end=exe_end, event_type="execution", job_id=job_id)
            self.state.robot.calendar.add_event(start=exe_start, end=exe_end, source="holding", dest="Proc2", event_type="hold", job_id=job_id)
            job.calendar.add_event(start=exe_start, end=exe_end, event_type="execution", dest="Proc2", job_id=job_id, op_id=op_id)

            # Mouvement retour vers station (déchargement fictif pour l’instant)
            unload_start = exe_end
            unload_end = unload_start + self.M
            self.state.robot.calendar.add_event(start=unload_start, end=unload_end, source="proc2", dest="S?", event_type="move", job_id=job_id)

    def find_earliest_start(self, job_id: int, op_id: int, parallel: bool) -> Tuple[int, Optional[int]]:
        """
        Étape 1 : Trouve la date de départ possible pour une opération (chargement si op 0, sinon exécution)
        """
        job = self.state.jobs[job_id]
        op = job.operations[op_id]

        if op_id == 0:
            # Chercher une station libre (selon big ou pas)
            station_ids = [1, 2, 3] if job.big == 0 else [2]  # Ex : seulement station 2 pour grosse pièce
            earliest_times = []
            for sid in station_ids:
                station = self.state.stations[sid]
                available = station.calendar.next_available()
                earliest_times.append((available, sid))

            # Choisir la première station libre
            chosen_time, chosen_station = min(earliest_times, key=lambda x: x[0])
            robot_available = self.state.robot.calendar.next_available()
            start_time = max(chosen_time, robot_available)

            return start_time, chosen_station

        else:
            # Opération suivante
            if not parallel:
                # Mode A (robot tient la pièce dans Proc1)
                proc1_time = self.state.proc1.calendar.next_available()
                robot_time = self.state.robot.calendar.next_available()
                return max(proc1_time, robot_time), None

            else:
                # Mode C (robot tient la pièce dans Proc2)
                proc2_time = self.state.proc2.calendar.next_available()
                robot_time = self.state.robot.calendar.next_available()
                return max(proc2_time, robot_time), None                        




# Chargement de l'instance JSON
def load_jobs_from_json(path: Path) -> List[Dict]:
    with open(path, 'r') as f:
        return json.load(f)


    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("instance", type=Path)
    args = parser.parse_args()

    job_data = load_jobs_from_json(args.instance)
    sim = Simulator(job_data)

    # Exemple de décisions (job_id, op_id, mode)
    decisions = [
        (1, 0, True),
        (0, 0, True),
        (1, 1, False),
        (2, 0, False)
    ]

    for decision in decisions:
        job_id, op_id, parallel = decision
        start_time, station = sim.find_earliest_start(job_id, op_id, parallel)
        print(f"Décision {decision} → start={start_time}, station={station}")
    
    print("\n==== CALENDRIERS ====\n")

    print("-- Robot --")
    for e in sim.state.robot.calendar.events:
        print(e)

    print("-- Positionneur --")
    for e in sim.state.positioner.calendar.events:
        print(e)

    print("-- Proc1 --")
    for e in sim.state.proc1.calendar.events:
        print(e)

    print("-- Proc2 --")
    for e in sim.state.proc2.calendar.events:
        print(e)

    for sid, station in sim.state.stations.items():
        print(f"-- Station S{sid} --")
        for e in station.calendar.events:
            print(e)

    for jid, job in sim.state.jobs.items():
        print(f"-- Job {jid} --")
        for e in job.calendar.events:
            print(e)


if __name__ == "__main__":
        main()