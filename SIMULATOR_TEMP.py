from __future__ import annotations
"""
Robot welding‑cell simulator (ALSTOM case‑study)
------------------------------------------------

**Version 1.3 – 2025‑05‑10**

Changes
~~~~~~~
* **Intégration complète des champs existants `status` et `blocked`** décrits
  dans l’article :  
  * `status = 0` (E0) : pièce hors du système (comportement standard)  
  * `status = 1` (E1) : pièce déjà posée sur la station `blocked` (on saute
    le temps de chargement `L` la première fois)  
  * `status = 2` (E2) : pièce tenue par le robot (modes A/C) : on saute le
    chargement `L` **et** le déplacement vers la station ; la première action
    du robot sera directement le déplacement vers le procédé.  
  * `status = 3` (E3) : pièce déjà fixée sur le positionneur (mode B) : on
    saute chargement, positionnement et on considère que le positionneur est
    occupé dès `t = 0`. Le premier procédé peut démarrer immédiatement si
    `proc‑1` est libre.
* Le champ `blocked` indique la station occupée à `t = 0` ; plusieurs pièces
  peuvent bloquer les trois stations.
* **Suppression** des anciens champs expérimentaux `preloaded` et `JobState`.
* Numérotation d’affichage **à partir de 1** pour jobs, opérations et stations
  (l’API interne reste indexée à 0 pour la simplicité du code Python).
"""

###########################################################################
# Imports
###########################################################################
import json
import random
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

###########################################################################
# Paramètres par défaut (modifiable à l'initialisation)
###########################################################################
M_DEFAULT: int = 3   # déplacement du robot entre deux nœuds (min)
L_DEFAULT: int = 2   # temps de charge/décharge (min)
BIG_T: int   = 1_000_000  # grande valeur pour bloquer une ressource « indéfiniment »

###########################################################################
# Imports & data models
###########################################################################
import json, argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

###########################################################################
# Ressource unaire générique
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
# Données métier
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
    jid: int
    big: bool
    due_date: int
    pos_time: int
    operations: List[Operation]
    status: int = 0      # 0=E0, 1=E1, 2=E2, 3=E3
    blocked: int = 0     # station bloquée (0 si aucune)
    loaded_station: Optional[int] = None
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
        # ressources physiques
        self.robot = Resource("robot")
        self.stations = {i: Resource(f"station-{i}") for i in (1, 2, 3)}
        self.proc1 = Resource("process-1")
        self.proc2 = Resource("process-2")
        self.positioner = Resource("positioner")

        # blocages initiaux
        for job in self.jobs:
            if job.blocked:
                self.stations[job.blocked].busy_until = BIG_T
                job.loaded_station = job.blocked
            if job.status == 3:
                self.positioner.busy_until = BIG_T

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
            elif job.status == 2:                    # pièce déjà dans la pince
                skip_load = True; skip_travel = True
            elif job.status == 3:                    # pièce déjà sur positionneur
                if mode != 2:
                    raise ValueError("First op of a status‑3 job must use mode B")
                skip_load = True; skip_travel = True; skip_pos = True
            # libère les blocages infinis
            if station.busy_until >= BIG_T:
                station.busy_until = 0
            if job.status == 3 and self.positioner.busy_until >= BIG_T:
                self.positioner.busy_until = 0
            job.first_move_pending = False

        # ------------------------------------------------------------------
        # 1) robot → station & chargement
        initial_call = self.time == 0 and self.robot.busy_until == 0
        # première pièce E0 déjà posée → pas de L
        if initial_call and job.status == 0 and not skip_load:
            skip_load = True

        travel1 = 0 if (skip_travel or initial_call) else self.M
        load_d  = 0 if skip_load else self.L
        t0 = max(self.time, station.busy_until)
        self.robot.reserve(t0, travel1)
        _, e_load = self.robot.reserve(t0 + travel1, load_d)
        station.busy_until = e_load

        # ------------------------------------------------------------------
        # 2) station → process
        travel2 = 0 if skip_travel else self.M
        _, e_tr2 = self.robot.reserve(e_load, travel2)

        # ------------------------------------------------------------------
        # 3) processing selon le mode
        if mode == 1:  # Mode A
            p_s, p_e = self._hold_and_process(self.proc1, e_tr2, op.processing_time)
            robot_free = p_e
        elif mode == 2:  # Mode B
            pos_d = 0 if skip_pos else op.pos_time
            _, pos_e = self.positioner.reserve(e_tr2, pos_d)
            p_s, p_e = self.proc1.reserve(pos_e, op.processing_time)
            self.robot.busy_until = pos_e + self.M
            robot_free = pos_e + self.M
        else:  # Mode C
            p_s, p_e = self._hold_and_process(self.proc2, e_tr2, op.processing_time)
            robot_free = p_e

        # ------------------------------------------------------------------
        # 4) Déchargement : **seul le trajet retour M**
        unload_d = self.M
        self.robot.reserve(robot_free, unload_d)
        station.busy_until = robot_free + unload_d

        # enregistrement
        op.mode, op.start, op.end = mode, p_s, p_e
        self.time = min(s.busy_until for s in self.stations.values())
        return {"station": job.loaded_station, "start": p_s, "end": p_e}

    # ------------------------------------------------------------------
    # Helpers
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

    def _hold_and_process(self, process: Resource, earliest: int, duration: int) -> Tuple[int, int]:
        self.robot.reserve(earliest, duration)              # robot occupé toute la durée
        start = max(earliest, process.busy_until)
        end   = start + duration
        self.robot.busy_until = end
        process.busy_until = end
        return start, end

    # ------------------------------------------------------------------
    def summary(self):
        print("\nJOBS")
        for job in self.jobs:
            print(f"\nJob {job.jid + 1} (station={job.loaded_station}, status={job.status})")
            for i, op in enumerate(job.operations):
                print(f"  op {i+1} – proc={op.proc}, mode={op.mode}, start={op.start}, end={op.end}")

###########################################################################
# Loader + CLI
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
    decisions = [
        (0, 0, 3),  # Job 1 / op 1 – mode C
        (1, 0, 2),  # Job 2 / op 1 – mode B
        (2, 0, 3),  # Job 3 / op 1 – mode C
        (2, 1, 2),  # Job 3 / op 2 – mode B
        (3, 0, 3),  # Job 4 / op 1 – mode C
        (3, 1, 1),  # Job 4 / op 2 – mode A
    ]
    for d in decisions:
        res = sim.apply_decision(*d)
        print(f"Decision (job={d[0]+1}, op={d[1]+1}, mode={d[2]}) → start={res['start']}, end={res['end']}, station={res['station']}")

    sim.summary()
