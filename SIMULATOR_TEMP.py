from __future__ import annotations
import json, argparse
import random
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

M_DEFAULT: int = 3   # déplacement du robot entre deux nœuds (min)
L_DEFAULT: int = 2   # temps de charge/décharge (min)
BIG_T: int   = 1_000_000  # grande valeur pour bloquer une ressource « indéfiniment »


###########################################################################
# Ressource 
###########################################################################
class Resource:
    def __init__(self, name: str):
        self.name = name
        self.busy_until: int = 0

    def reserve(self, earliest: int, duration: int) -> Tuple[int, int]:
        start = max(earliest, self.busy_until)
        end   = start + duration
        self.busy_until = end
        return start, end

    def __repr__(self) -> str:
        return f"{self.name}(busy_until={self.busy_until})"

###########################################################################
# Données 
###########################################################################
@dataclass
class Operation:
    proc: int
    processing_time: int
    pos_time: int = 0
    mode:   Optional[int] = None
    start:  Optional[int] = None
    end:    Optional[int] = None

    def is_finished(self) -> bool:
        return self.end is not None

@dataclass
class Job:
    j_id: int
    big: bool
    due_date: int
    pos_time: int
    operations: List[Operation]
    status: int = 0      # 0=E0, 1=E1, 2=E2, 3=E3
    blocked: int = 0     # station bloquée (0 si aucune)
    loaded_station: Optional[int] = None
    is_parallele = False
    first_move_pending: bool = True

###########################################################################
# Simulateur principal
###########################################################################
class Simulator:
    def __init__(self, jobs: List[Job], M: int = M_DEFAULT, L: int = L_DEFAULT):
        self.jobs = jobs
        self.M = M
        self.L = L
        self.time = 0
        self.robot_loc   = "idle" 
        self.robot_trace = []
        self.positioner_state   = "free" 
        self.positioner_trace = []
        # ressources physiques
        self.robot = Resource("robot")
        self.stations = {i: Resource(f"station-{i}") for i in (1, 2, 3)}
        self.proc1 = Resource("process-1")
        self.proc2 = Resource("process-2")
        self.positioner = Resource("positioner")

    
        '''# blocages initiaux
        for job in self.jobs:
            if job.blocked:
                self.stations[job.blocked].busy_until = BIG_T
                job.loaded_station = job.blocked
            if job.status == 3:
                self.positioner.busy_until = BIG_T'''
    
    # ------------------------------------------------------------------
    def _select_station(self, job: Job) -> int:
        if job.big:
            return 2
        pref = [1, 3, 2]
        ready = {i: self.stations[i].busy_until for i in pref if self.stations[i].busy_until < BIG_T}
        earliest = min(ready.values())
        for i in pref:
            if ready.get(i) == earliest:
                return i
        return 1
    
    def _next_event_time(self)-> int:
        """Instant le plus proche où *toutes* les ressources critiques
       peuvent accepter une nouvelle décision."""
        return min(
        self.robot.busy_until,
        self.positioner.busy_until,
        self.proc1.busy_until,
        self.proc2.busy_until,
        *(s.busy_until for s in self.stations.values())
    )

    def _hold_and_process(self, process: Resource, earliest: int, duration: int) -> Tuple[int, int]:
        self.robot.reserve(earliest, duration)            # robot occupé toute la durée
        start = max(earliest, process.busy_until)
        end   = start + duration
        self.robot.busy_until = end
        process.busy_until = end
        return start, end

    # ------------------------------------------------------------------
    def apply_decision(self, job_id: int, op_id: int, mode: int) -> Dict[str, int]:
        job = self.jobs[job_id]
        op  = job.operations[op_id]

        # -------- station : allocation ou rappel ----------------------
        if job.loaded_station is None:
            job.loaded_station = self._select_station(job)
        station = self.stations[job.loaded_station]

        # -------- prise en compte du status ---------------------------
        skip_load = skip_travel = skip_pos = False
        if job.first_move_pending and job.status in (1, 2, 3):
            if job.status == 1:                      # pièce déjà sur station
                skip_load = True
            elif job.status == 2:                    # pièce déjà sur le robot
                skip_load = True; 
                skip_travel = True
            elif job.status == 3:                    # pièce déjà sur positionneur
                skip_load = True; skip_travel = True; skip_pos = True
                if mode != 2:
                    raise ValueError("First op of a status‑3 job must use mode B")
                
            # libère les blocages infinis
            if station.busy_until >= BIG_T:
                station.busy_until = 0
            if job.status == 3 and self.positioner.busy_until >= BIG_T:
                self.positioner.busy_until = 0
            job.first_move_pending = False

        # ------------------------------------------------------------------
        # 1) robot → station & chargement
        initial_call = self.time == 0 and self.robot.busy_until == 0
        self._note_robot(self.time, f"moving to station-{job.loaded_station}")
        # première pièce E0 déjà posée → pas de L
        if initial_call and job.status == 0 and not skip_load:
            skip_load = True

        load_duration  = 0 if skip_load else self.L
        travel1 = 0 if (skip_travel or initial_call) else self.M
        t0 = max(self.time, station.busy_until)
        self.robot.reserve(t0, travel1)
        _, end_load = self.robot.reserve(t0 + travel1, load_duration)
        station.busy_until = end_load
        self._note_robot(t0 + travel1, f"loading at station-{job.loaded_station}")

        # ------------------------------------------------------------------
        # 2) station → process
        travel2 = 0 if skip_travel else self.M
        _, end_tr2 = self.robot.reserve(end_load, travel2)

        # ------------------------------------------------------------------
        # 3) processing selon le mode
        if mode == 1:  # Mode A
            self._note_robot(end_tr2, f"holding in proc1")
            p_start, p_end = self._hold_and_process(self.proc1, end_tr2, op.processing_time)
            robot_free = p_end
        elif mode == 2:  # Mode B
            pos_duration = 0 if skip_pos else op.pos_time
            self._note_robot(end_tr2, f"moving to positioner")
            if not skip_pos:                # la pièce n'y est pas déjà
                self._note_positioner(end_tr2, f"job-{job.j_id}")
            _, pos_end = self.positioner.reserve(end_tr2, pos_duration)
            p_start, p_end = self.proc1.reserve(pos_end, op.processing_time)
             # Après le process, juste avant le trajet retour vers la station
            self._note_positioner(p_end, "free")      # libère le positionneur
            self._note_robot(p_end, f"moving from positioner to station-{job.loaded_station}")
            self.robot.busy_until = pos_end + self.M
            robot_free = pos_end + self.M
            self._note_robot(pos_end, f"idle (proc1 en cours)")   # robot libre
        else:  # Mode C
            self._note_robot(end_tr2, f"holding in proc2")
            p_start, p_end = self._hold_and_process(self.proc2, end_tr2, op.processing_time)
            robot_free = p_end

        # ------------------------------------------------------------------
        # 4) Déchargement : seul le trajet retour M
        self._note_robot(robot_free, f"unloading at station-{job.loaded_station}")
        unload_duration = self.M
        self.robot.reserve(robot_free, unload_duration)
        station.busy_until = robot_free + unload_duration
        self._note_robot(robot_free + unload_duration, "idle")

        # enregistrement
        op.mode, op.start, op.end = mode, p_start, p_end
        self.time = self._next_event_time()
        return {"station": job.loaded_station, "start": p_start, "end": p_end}

    # ------------------------------------------------------------------

    # ------------------------------------------------------------------

    def _note_robot(self, t: int, loc: str):
        """Enregistre la nouvelle position si elle change."""
        if loc != self.robot_loc:               
            self.robot_loc = loc
            self.robot_trace.append((t, loc))

    def dump_robot_trace(self):
        print("\nROBOT TIMELINE")
        for t, msg in sorted(self.robot_trace, key=lambda x: x[0]):
            print(f"[{t:>5}]  {msg}")
    
    def _note_positioner(self, t: int, state: str):
        if state != self.positioner_state:
            self.positioner_state = state
            self.positioner_trace.append((t, state))

    def dump_positioner_trace(self):
        print("\nPOSITIONNEUR TIMELINE")
        for t, st in self.positioner_trace:
            print(f"[{t:>5}]  {st}")
    
    def summary(self):
        print("\nJOBS")
        for job in self.jobs:
            print(f"\nJob {job.j_id + 1} (station={job.loaded_station}, status={job.status})")
            for i, op in enumerate(job.operations):
                print(f"  op {i+1} – proc={op.proc}, mode={op.mode}, start={op.start}, end={op.end}")

###########################################################################
# Loader 
###########################################################################

def load_jobs_from_json(path: str | Path) -> List[Job]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    jobs: List[Job] = []
    for idx, j in enumerate(raw):
        ops = [Operation(proc=o["type"], processing_time=o["processing_time"], pos_time=j["pos_time"]) for o in j["operations"]]
        jobs.append(Job(
            idx, 
            bool(j["big"]), 
            j["due_date"], 
            j["pos_time"], 
            operations=ops, 
            status=j.get("status", 0), 
            blocked=j.get("blocked", 0)))
    return jobs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("instance", type=Path)
    args = parser.parse_args()

    sim = Simulator(load_jobs_from_json(args.instance), M_DEFAULT, L_DEFAULT)

    # Décisions d'exemple – même séquence que précédemment
    decisions1 = [
        (1, 0, 2),  # Job 2 / op 1 – mode B
        (0, 0, 3),  # Job 1 / op 1 – mode C
        (1, 1, 3),  # Job 2 / op 2 – mode C
        (2, 0, 1),  # Job 3 / op 1 – mode A
    ]
    decisions2 = [
        (0, 0, 3),  # Job 1 / op 1 – mode C
        (1, 0, 2),  # Job 2 / op 1 – mode B
        (2, 0, 3),  # Job 3 / op 1 – mode C
        (2, 1, 2),  # Job 3 / op 2 – mode B
        (3, 0, 3),  # Job 4 / op 1 – mode C
        (3, 1, 1),  # Job 4 / op 2 – mode A
    ]
    decisions3 = [
        (0, 0, 2),  # Job 1 / op 1 – mode B
        (2, 0, 3),  # Job 3 / op 1 – mode C
        (1, 0, 3),  # Job 2 / op 1 – mode C
    ]
    for d in decisions1:
        res = sim.apply_decision(*d)
        print(f"Decision (job={d[0]+1}, op={d[1]+1}, mode={d[2]}) → start={res['start']}, end={res['end']}, station={res['station']}")
        print("-----",sim._next_event_time())

    sim.summary()
    sim.dump_robot_trace()
    # sim.dump_positioner_trace()
