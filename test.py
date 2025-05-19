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
        self.locked: Dict[str, bool] = {"S1": False, "S2": False, "S3": False}
        self.wait_B: Deque[Tuple[int, float]] = deque()  # (job_id, ready)
        self.proc_busy: Dict[str, float] = {"Proc1": 0.0, "Proc2": 0.0}


    # Helpers ---------------------------------------------------------------
    def _add_evt(self, src: str, dst: str, evt: str, dur: int, job: Optional[int]):
        if src == dst and dur == 0:
            return  # skip
        start = self.robot.free_at
        end = start + dur
        self.events.append(Event(src, dst, start, evt, end, dur, job))
        self.robot.free_at = end
        self.robot.loc = dst
    
    def _move(self, dst: str, job_id=None):
        if self.robot.loc == dst:
            return
        # Aucune durée entre deux stations
        if self.robot.loc.startswith("S") and dst.startswith("S"):
            self.robot.loc = dst
            return
        self._add_evt(self.robot.loc, dst, "move", M, job_id)

    def _wait_station(self, st: str):
        if not self.locked[st]:
            return
        # chercher la prochaine fin d’événement concernant cette station
        fut = [e.end for e in self.events
            if e.dest == st and e.event in {"move", "hold"} and e.end > self.robot.free_at]
        if fut:
            self.robot.free_at = min(fut)
        else:
            self.robot.free_at += M  # garde-fou
        self._collect_B_ready()     # on libère peut-être la station


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
                # Proc2 pas libre → on attend (calendrier robot reste). Décision refile plus tard
                self.robot.free_at = max(self.robot.free_at, self.proc_busy["Proc2"])
                self._move("Proc2", job_id)
        else:
            # Cas normal : aller à la station (et attendre si verrouillée)
            self._wait_station(job.station)
            self.locked[job.station] = True   # on (re)verrouille pour cette pièce
                #self._collect_B_ready()
            self.locked[job.station] = True
            self._move(job.station)
            # move station -> Pos/ProcX
            dest = "Pos" if mode == "B" else f"Proc{op.proc}"
            self._move(dest, job_id)

        # Si la station n'est pas libre est la piece en cours doit se decharger pour qu'une nouvelle piece en attente se charge 


        # Exécution ---------------------------------------------------------
        if mode == "B":
            self._add_evt("Pos", "Pos", "pos", op.pos_time or job.pos_time or POS_T, job_id)
            ready = self.robot.free_at + op.processing_time
            self.wait_B.append((job_id, ready))
            self.proc_busy["Proc1"] = ready
        else:
            hold_dest = f"Proc{op.proc}"
            self._add_evt(hold_dest, hold_dest, "hold", op.processing_time, job_id)
            self.proc_busy[hold_dest] = self.robot.free_at
            # retour station et déverrouille
            self._move(job.station, job_id)
            self.locked[job.station] = False

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