from __future__ import annotations

"""Robot Calendar Simulator (v7)
────────────────────────────────────────────────────────────────────────────
Règles finales intégrées :
1. **Le robot n’intervient jamais dans le (dé)chargement L**.  Il ne fait
   que déplacer les pièces entre : station ▸ Pos ▸ Proc1 ▸ Proc2.
2. **Une pièce entre et sort par UNE SEULE station** (S2 si `big = true`,
   sinon S1 ou S3).  La station reste **bloquée tant que la pièce est dans
   le système** (du premier chargement jusqu’à la dépose finale).
3. Si la station est bloquée, toute nouvelle pièce ciblant cette station
   **attend** que la précédente soit complètement terminée.
4. 0 s entre deux stations (Sx→Sy) comme validé, pas d’événement durée 0.
5. CSV export vers `data/calendars/robot_calender.csv` (modifiable via
   `--csv`).

Décisions : [(job_id, op_id, parallel)]   parallel ⇒ mode :
    * Proc1 :  False → A   True → B
    * Proc2 :  always C  (parallel flag ignoré)
"""

import json, csv
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Deque
from collections import deque
from pathlib import Path

# Constantes globales -------------------------------------------------------
M = 3     # déplacement robot (s)
POS_T = 5 # fixation en mode B (s)
DEFAULT_CSV = Path("data/calendars/robot_calender.csv")

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

# Simulator -----------------------------------------------------------------
class Simulator:
    def __init__(self, jobs: List[Job]):
        self.jobs: Dict[int, Job] = {j.job_id: j for j in jobs}
        self.events: List[Event] = []
        self.robot = Robot()
        self.station_locked: Dict[str, bool] = {"S1": False, "S2": False, "S3": False}
        self.wait_B: Deque[Tuple[int, int]] = deque()  # (job_id, ready_at)

    # Helpers ---------------------------------------------------------------
    def _add_evt(self, src: str, dst: str, evt: str, dur: int, job: Optional[int]):
        if src == dst and dur == 0:
            return  # skip
        start = self.robot.free_at
        end = start + dur
        self.events.append(Event(src, dst, start, evt, end, dur, job))
        self.robot.free_at = end
        self.robot.loc = dst

    @staticmethod
    def _mode(proc: int, parallel: bool):
        if proc == 1:
            return "B" if parallel else "A"
        return "C"

    @staticmethod
    def _default_station(job: Job):
        return "S2" if job.big else "S1"

    # Main loop -------------------------------------------------------------
    def execute(self, decisions: List[Tuple[int, int, bool]]):
        for job_id, op_id, parallel in decisions:
            self._collect_finished_B()
            self._do_decision(job_id, op_id, parallel)
        self._flush_B()

    # Gestion des pièces mode B finies -------------------------------------
    def _collect_finished_B(self):
        while self.wait_B and self.wait_B[0][1] <= self.robot.free_at:
            job_id, ready = self.wait_B.popleft()
            job = self.jobs[job_id]
            # attendre que sa station soit libre (elle est déjà verrouillée pour lui)
            self._move_to("Pos")
            self._move_to(job.station)
            # dépose finale déverrouille la station
            self.station_locked[job.station] = False

    def _flush_B(self):
        while self.wait_B:
            self._collect_finished_B()

    # Méthodes internes -----------------------------------------------------
    def _move_to(self, dst: str, job: Optional[int] = None):
        if self.robot.loc == dst:
            return
        # Passage instantané entre stations (Sx → Sy) : aucun événement ni durée
        if self.robot.loc.startswith("S") and dst.startswith("S"):
            self.robot.loc = dst
            return
        # Autres déplacements (station ↔ Pos/Proc) : durée M
        self._add_evt(self.robot.loc, dst, "move", M, job)
                

    def _wait_station(self, station: str):
        while self.station_locked[station]:
            # avancer le temps jusqu’à la libération la plus proche
            next_time = min(evt.end for evt in self.events if evt.dest == station and evt.event == "move" and evt.end > self.robot.free_at)
            self.robot.free_at = next_time
            self._collect_finished_B()

    def _do_decision(self, job_id: int, op_idx: int, par: bool):
        job = self.jobs[job_id]
        op = job.operations[op_idx]
        if job.station == "":
            job.station = self._default_station(job)
        # attendre station dispo pour CHARGEMENT initial (si première op)
        self._wait_station(job.station)
        self.station_locked[job.station] = True  # station bloquée jusqu’à fin pièce

        # aller à la station (0 s si déjà dessus)
        self._move_to(job.station)

        # mode & déplacement vers ressource
        mode = self._mode(op.proc, par)
        dest = "Pos" if mode == "B" else f"Proc{op.proc}"
        self._move_to(dest, job_id)

        if mode == "B":
            # fixation
            self._add_evt("Pos", "Pos", "pos", op.pos_time or job.pos_time or POS_T, job_id)
            ready = self.robot.free_at + op.processing_time
            self.wait_B.append((job_id, ready))
        else:
            # hold (robot immobilisé pendant le procédé)
            self._add_evt(dest, dest, "hold", op.processing_time, job_id)
            # retour station + dépose finale → déverrouille station
            self._move_to(job.station, job_id)
            self.station_locked[job.station] = False

        job.advance()

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
        print(f"job {'src':>3} -> {'dst':>5}  start   end   dur  evt")
        print("-"*46)
        for e in self.events:
            print(f" {e.job} {e.source:>3} -> {e.dest:>5}  {e.start:6.2f} {e.end:6.2f} {e.duration:5.2f}  {e.event:>4}")

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
    
    decisions = [
        (1, 0, True),
        (0, 0, True),
        (1, 1, False),
        (2, 0, False)
    ]
    sim = Simulator(job_data)
    sim.execute(decisions)
    sim.show()
