from __future__ import annotations
import json, csv
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Deque
from collections import deque
from pathlib import Path

# Constantes globales -------------------------------------------------------
L = 2
M = 3     # déplacement Robot (s)
POS_T = 5 # fixation en mode B (s)
DEFAULT_CSV = Path("data/calendars/Robot_calender.csv")

# Data classes --------------------------------------------------------------
@dataclass
class Event:
    source: str
    dest: str
    start: int
    event: str   # move | pos | hold
    end: int
    duration: int
    job: Optional[int]

@dataclass
class Operation:
    proc: int
    processing_time: int
    pos_time: int

@dataclass
class Job:
    job_id: int
    big: bool
    pos_time: int
    operations: List[Operation]
    station: str = ""    # choisi à l’entrée
    op_index: int = 0     # prochaine opération

    def next_op(self) -> Operation | None:
        return self.operations[self.op_index] if self.op_index < len(self.operations) else None

    def advance(self):
        self.op_index += 1

@dataclass
class Robot:
    loc: str = "S1"
    free_at: int = 0

#############################################
# Robot Calendar
#############################################
@dataclass
class RobotCalendar:
    def __init__(self): self.events = []
    def add(self, ev): self.events.append(ev)
    def __iter__(self): return iter(self.events)

#############################################
# Station Calendar
#############################################
@dataclass
class StationCalendar:
    def __init__(self): self.events = []
    def add(self, start, end, event_type, job):
        self.events.append({"start": start, "end": end, "event_type": event_type, "job": job})
    def __iter__(self): return iter(self.events)
    def __iter__(self): return iter(self.events)

#############################################
# Pos + Proc1 Calendar
#############################################
@dataclass
class Proc1PosCalendar:
    def __init__(self): self.events = []
    def add_pos(self, start, end, job):
        self.events.append({"start": start, "end": end, "event_type": "pos", "job": job})
    def add_exec(self, start, end, job):
        self.events.append({"start": start, "end": end, "event_type": "execution", "job": job})
    def __iter__(self): return iter(self.events)

#############################################
# Proc 2 Calendar
#############################################
@dataclass
class Proc2Calendar:
    def __init__(self): self.events = []
    def add_exec(self, start, end, job):
        self.events.append({"start": start, "end": end, "event_type": "execution", "job": job})
    def __iter__(self): return iter(self.events)

#############################################
# Job Calendar
#############################################
@dataclass
class JobCalendar:
    def __init__(self): self.events = []
    def add(self, start, end, source, dest, event_type):
        self.events.append({"start": start, "end": end, "source": source, "dest": dest, "event_type": event_type})
    def __iter__(self): return iter(self.events)


# Simulator -----------------------------------------------------------------
class Simulator:
    def __init__(self, jobs: List[Job]):
        self.jobs: Dict[int, Job] = {j.job_id: j for j in jobs}
        self.events: List[Event] = []
        self.robot = Robot()
        self.station_locked: Dict[str, bool] = {"S1": False, "S2": False, "S3": False}
        self.locked: Dict[str, bool] = {"S1": False, "S2": False, "S3": False}
        self.wait_B: Deque[Tuple[int, float]] = deque()  # (job_id, ready)
        self.proc_busy: Dict[str, float] = {"Proc1": 0, "Proc2": 0}
        self.robot_calendar = RobotCalendar()
        self.proc1_calendar = Proc1PosCalendar()
        self.proc2_calendar = Proc2Calendar()
        self.stations = {s: StationCalendar() for s in ("S1","S2","S3")}
        self.job_calendars = {j.job_id: JobCalendar() for j in jobs}


    def _add_event_line(self, src, dest, evt, dur, job_id):
        if src == dest and dur == 0: return
        start, end = self.robot.free_at, self.robot.free_at + dur
        ev = Event(src, dest, start, evt, end, dur, job_id)
        self.robot_calendar.add(ev)
        if evt == "move":
            if dest in self.stations:
                self.stations[dest].add(start=end, end=end+L, event_type="unload", job=job_id)
                self.job_calendars[job_id].add(start, end, src, dest, "move")
            if src in self.stations:
                self.stations[src].add(start=start-L, end=start, event_type="load", job=job_id)
        elif evt == "hold":
            if dest == "Proc1": self.proc1_calendar.add_exec(start, end, job_id)
            if dest == "Proc2": self.proc2_calendar.add_exec(start, end, job_id)
            self.job_calendars[job_id].add(start, end, dest, dest, "execution")
        elif evt == "pos":
            self.proc1_calendar.add_pos(start, end, job_id)
        self.robot.loc, self.robot.free_at = dest, end

    def _move(self, dest, job_id):
        if self.robot.loc == dest: return
        if self.robot.loc.startswith("S") and [e.dest.startswith("S") for e in self.events]:
            self.robot.loc = dest; return
        self._add_event_line(self.robot.loc, dest, "move", M, job_id)

    # Attend que la station soit libre
    def _wait_station(self, st: str):
        if not self.locked[st]:
            return
        # chercher la prochaine fin d’événement concernant cette station
        fin = [e.end for e in self.events
            if e.dest == st and e.event in {"move", "hold"} and e.end > self.robot.free_at]
        if fin:
            self.robot.free_at = min(fin)
        else:
            self.robot.free_at += M

    @staticmethod
    def _mode(proc: int, parallel: bool):
        if proc == 1:
            return "B" if parallel else "A"
        return "C"

    # choix de la station
    def _default_station(self, job: Job):
        if job.big:
            return "S2"
        # Choisir S1 si libre, sinon S3
        for station in ("S1", "S3"):
            if not self.locked[station]:
                return station
        # Si les deux sont verrouillées, retourner S1 par défaut
        return "S1"

    # Gestion des pièces mode B finies -------------------------------------
    def _collect_finished_B(self):
        while self.wait_B and self.wait_B[0][1] <= self.robot.free_at:
            job_id, ready = self.wait_B.popleft()
            job = self.jobs[job_id]
            # attendre que sa station soit libre (elle est déjà verrouillée pour lui)
            self._move("Pos", job_id)
            self._move(job.station, job_id)
            # dépose finale déverrouille la station
            self.station_locked[job.station] = False

    def _flush_B(self):
        while self.wait_B:
            self._collect_finished_B()
                
    
    def _do_decision(self, job_id: int, op_idx: int, parallel: bool):
        job = self.jobs[job_id]
        if job.station == "":
            job.station = self._default_station(job)
        op = job.operations[op_idx]
        mode = self._mode(op.proc, parallel)

        # Si la pièce est encore sur Pos et l’opération est Proc2 ➜ déplacer direct si libre
        if self.robot.loc == "Pos" and job.next_op() == op:
            if op.proc == 2 and self.robot.free_at >= self.proc_busy["Proc2"]:
                self._move("Proc2", job_id)
            else:
                # Proc2 pas libre → on attend (calendrier Robot reste). Décision refile plus tard
                self.robot.free_at = max(self.robot.free_at, self.proc_busy["Proc2"])
                self._move("Proc2", job_id)
        else:
            # Cas normal : aller à la station (et attendre si verrouillée)
            self._wait_station(job.station)
            self.locked[job.station] = True   # on (re)verrouille pour cette pièce
                #self._collect_B_ready()
            self.locked[job.station] = True
            self._move(job.station, job_id)
            # move station -> Pos/ProcX
            dest = "Pos" if mode == "B" else f"Proc{op.proc}"
            self._move(dest, job_id)


        # Si la station n'est pas libre est la piece en cours doit se decharger pour qu'une nouvelle piece en attente se charge 

        # Exécution ---------------------------------------------------------
        if mode == "B":
            self._add_event_line("Pos", "Pos", "pos", op.pos_time or job.pos_time or POS_T, job_id)
            ready = self.robot.free_at + op.processing_time
            self.wait_B.append((job_id, ready))
            self.proc_busy["Proc1"] = ready
        else:
            hold_dest = f"Proc{op.proc}"
            self._add_event_line(hold_dest, hold_dest, "hold", op.processing_time, job_id)
            self.proc_busy[hold_dest] = self.robot.free_at
            # retour station et déverrouille
            self._move(job.station, job_id)
            self.locked[job.station] = False

        job.advance()


        # Main loop -------------------------------------------------------------
    def execute(self, decisions: List[Tuple[int, int, bool]]):
        for job_id, op_id, parallel in decisions:
            self._collect_finished_B()
            self._do_decision(job_id, op_id, parallel)
        self._flush_B()


    # Export CSV ------------------------------------------------------------
    def to_csv(self, path: Path | str = DEFAULT_CSV):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["job_id", "source", "destination", "start", "end", "duration", "event"])
            for e in self.events:
                w.writerow([e.job, e.source, e.dest, f"{e.start:.2f}", f"{e.end:.2f}", f"{e.duration:.2f}", e.event ])
        return p


    def show(self):
        print("--- Robot Calendar ---")
        for e in self.robot_calendar:
            print(f"{e.source:>3} -> {e.dest:<5} {e.start:6.2f} {e.end:6.2f} {e.duration:6.2f} {e.event:>4} {e.job}")
        print("\n--- Job Calendars ---")
        for jid, lst in self.job_calendars.items():
            print(f"Job {jid}:")
            for ev in lst:
                print(f" {ev['start']:6.2f}-{ev['end']:6.2f} {ev['event_type']:>10} {ev['source']}→{ev['dest']}")
        for k, cal in self.stations.items():
            print(f"\n--- {k} Calendar ---")
            for ev in cal:
                print(f" {ev['start']:6.2f}-{ev['end']:6.2f} {ev['event_type']:>10} job {ev['job']}")
        print("\n--- Proc1 + Pos Calendar ---")
        for ev in self.proc1_calendar:
            print(f" {ev['start']:6.2f}-{ev['end']:6.2f} {ev['event_type']:>10} job {ev['job']}")
        print("\n--- Proc2 Calendar ---")
        for ev in self.proc2_calendar:
            print(f" {ev['start']:6.2f}-{ev['end']:6.2f} {ev['event_type']:>10} job {ev['job']}")


            

# Chargement JSON -----------------------------------------------------------

def load_jobs_from_json(p: str | Path) -> List[Job]:
    data = json.loads(Path(p).read_text())
    jobs: List[Job] = []
    for idx, j in enumerate(data):
        ops = [Operation(o["type"], o["processing_time"], j["pos_time"]) for o in j["operations"]]
        jobs.append(Job(idx, bool(j["big"]), j["pos_time"], ops))
    return jobs

# CLI ----------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("instance", type=Path)
    args = parser.parse_args()

    job_data = load_jobs_from_json(args.instance)
    sim = Simulator(job_data)
    
    decisions1 = [
        (1, 0, True),
        (0, 0, True),
        (1, 1, False),
        (2, 0, False)
    ]
    decisions2 = [
        (0, 0, True),  # Job 1 / op 1 – mode C
        (1, 0, True),  # Job 2 / op 1 – mode B
        (2, 0, True),  # Job 3 / op 1 – mode C
        (2, 1, True),  # Job 3 / op 2 – mode B
        (3, 0, True),  # Job 4 / op 1 – mode C
        (3, 1, False),  # Job 4 / op 2 – mode A
    ]
    decisions3 = [
        (0, 0, True),  # Job 1 / op 1 – mode Bje ve
        (2, 0, True),  # Job 3 / op 1 – mode C
        (1, 0, True),  # Job 2 / op 1 – mode C
    ]
    sim = Simulator(job_data)
    sim.execute(decisions1)
    sim.show()

# python3 test.py data/instances/debug/1st_instance.json
# python3 test.py data/instances/debug/2nd_instance.json
# python3 test.py data/instances/debug/3rd_instance.json