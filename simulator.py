from __future__ import annotations
import csv
import json
from pathlib import Path


"""

Execution debug instance
1- python3 simulator.py data/instances/debug/1st_instance.json
2- python3 simulator.py data/instances/debug/2nd_instance.json
3- python3 simulator.py data/instances/debug/3rd_instance.json

"""

"""Robot calendar simulator
--------------------------------------------------
Implements the rules converged on with the user (Instance 1 timeline)
* single timeline for the robot (list of Event)
* object‑oriented organisation (RobotState, Job, Station, Simulator)
* works for any instance + decision list (job_id, op_id, parallel)

Key constants (can be tuned):
    M   : robot displacement time between two nodes (s)
    POS : positioning time on the positioner (mode B) (s)

Loading / unloading time L is NOT counted in the robot calendar
(stations handle it in their own timeline).

Usage example is provided in __main__ at the bottom.
"""

import argparse
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Deque
from collections import deque

# ─────────────────────────────────────────────────────────────── constants ──
M: float = 3   # movement robot
POS: float = 5  # fixation mode B
DEFAULT_CSV = Path("data/calendars/robot_calender.csv")


# ---------------------------------------------------------------- events ──
@dataclass
class Event:
    source: str
    dest: str
    start: float
    event: str  # "move" | "pos" | "hold"
    end: float
    duration: float
    job: Optional[int] = None

@dataclass
class Operation:
    proc: int
    processing_time: float
    pos_time: float

@dataclass
class Job:
    job_id: int
    big: bool
    due_date: float
    pos_time: float
    operations: List[Operation]
    status: int = 0
    blocked: int = 0
    current_station: str = "S1"
    current_op: int = 0

    def advance(self):
        self.current_op += 1

# ───────────────────────────────────────────── robot ──
@dataclass
class RobotState:
    loc: str = "S1"
    free_at: float = 0.0

# ───────────────────────────────────────────── core ──
class Simulator:
    def __init__(self, jobs: List[Job]):
        self.jobs: Dict[int, Job] = {j.job_id: j for j in jobs}
        self.events: List[Event] = []
        self.robot = RobotState()
        self.wait_B: Deque[Tuple[int, float]] = deque()  # (job_id, ready_at)

    # ---------------- internal helper ----------------
    def _append(self, src: str, dst: str, start: float, evt: str, dur: float, job: Optional[int] = None):
        end = start + dur
        self.events.append(Event(src, dst, start, evt, end, dur, job))
        return end

    @staticmethod
    def _mode(proc: int, parallel: bool) -> str:
        return "B" if (proc == 1 and parallel) else ("C" if proc == 2 else "A")

    @staticmethod
    def _station_for(job: Job) -> str:
        return "S2" if job.big else job.current_station or "S1"

    # ---------------- algorithm ----------------------
    def execute(self, decisions: List[Tuple[int, int, bool]]):
        for j, op_idx, par in decisions:
            self._collect_B_if_ready()
            self._process_decision(j, op_idx, par)
        self._flush_B()

    def _collect_B_if_ready(self):
        while self.wait_B and self.wait_B[0][1] <= self.robot.free_at:
            self._pickup_B()

    def _pickup_B(self):
        job_id, ready = self.wait_B.popleft()
        target = self._station_for(self.jobs[job_id])
        self.robot.free_at = self._append("Pos", target, max(self.robot.free_at, ready), "move", M, job_id)
        self.robot.loc = target

    def _flush_B(self):
        while self.wait_B:
            self._pickup_B()

    def _process_decision(self, job_id: int, op_idx: int, parallel: bool):
        job = self.jobs[job_id]
        op = job.operations[op_idx]
        mode = self._mode(op.proc, parallel)

        # amener robot à la station de la pièce (0 s Sx→Sy)
        if not self.robot.loc.startswith("S"):
            self.robot.free_at = self._append(self.robot.loc, job.current_station, self.robot.free_at, "move", M)
            self.robot.loc = job.current_station
        if self.robot.loc != job.current_station:
            self.robot.free_at = self._append(self.robot.loc, job.current_station, self.robot.free_at, "move", 0.0)
            self.robot.loc = job.current_station

        # move station -> Pos / ProcX
        dest = "Pos" if mode == "B" else f"Proc{op.proc}"
        self.robot.free_at = self._append(self.robot.loc, dest, self.robot.free_at, "move", M, job_id)
        self.robot.loc = dest

        if mode == "B":
            fix_end = self._append("Pos", "Pos", self.robot.free_at, "pos", op.pos_time or job.pos_time or job_id)
            ready = fix_end + op.processing_time
            self.wait_B.append((job_id, ready))
            self.robot.free_at = fix_end
            self.robot.loc = "Pos"
        else:
            hold_end = self._append(dest, dest, self.robot.free_at, "hold", op.processing_time, job_id)
            drop = self._station_for(job)
            self.robot.free_at = self._append(dest, drop, hold_end, "move", M, job_id)
            self.robot.loc = drop

        job.advance()
        job.current_station = self.robot.loc

    # ---------------- output -------------------------
    def to_csv(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["source", "destination", "start", "end", "duration", "event", "job_id"])
            for e in self.events:
                w.writerow([e.source, e.dest, f"{e.start:.2f}", f"{e.end:.2f}", f"{e.duration:.2f}", e.event, e.job])
        return path
    
    def show(self):
        head = f"{'src':>4} -> {'dst':>4}  start   end   dur  evt  job"
        print(head) ; print("-" * len(head))
        for e in self.events:
            print(f"{e.job} {e.source:>4} -> {e.dest:>4}  {e.start:5.2f}  {e.end:5.2f}  {e.duration:4.2f}  {e.event:>4} ")

# ──────────────────────────────────────── loader ──

def load_jobs_from_json(path: str | Path) -> List[Job]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    jobs: List[Job] = []
    for idx, j in enumerate(data):
        ops = [Operation(proc=o["type"], processing_time=o["processing_time"], pos_time=j["pos_time"]) for o in j["operations"]]
        jobs.append(Job(
            job_id=idx,
            big=bool(j["big"]),
            due_date=j["due_date"],
            pos_time=j["pos_time"],
            operations=ops,
            status=j.get("status", 0),
            blocked=j.get("blocked", 0),
            current_station=j.get("station", "S1")
        ))
    return jobs

# ───────────────────────────────────────────────────────────── example ──
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("instance", type=Path)
    args = parser.parse_args()

    job_data = load_jobs_from_json(args.instance)
    sim = Simulator(job_data)
    
    decisions = [
        (1, 0, True),
        (0, 0, True),
        (1, 1, False),
        (2, 0, False)
    ]
    sim = Simulator(job_data)
    sim.execute(decisions)
    sim.show()

    # csv_path = args.csv or DEFAULT_CSV
    # exported = sim.to_csv(csv_path)
    # print(f"\n✔ Calendrier enregistré dans {exported}")

