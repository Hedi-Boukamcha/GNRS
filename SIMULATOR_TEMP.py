"""
    Discrete‑event simulator for the two‑process robotic welding cell
    described in Neumann & al. (2025).  It executes **one scheduling
    decision at a time** and keeps an internal calendar for all physical
    resources (robot arm, positioner, process 1, process 2, three loading
    stations).  After each call the simulator knows exactly where every
    job/part is and when each resource will become free.

    A *decision* is a tuple  (job_id, op_id, mode)  where:
    • job_id  – integer identifier of the part/job
    • op_id   – identifier of the next technological operation of that job
    • mode    – 'A', 'B' or 'C'   (must be feasible for that op)

    The object returns a dict with the realised start/end dates of the
    operation and the station actually used.  All timing rules of the cell
    (load/unload times, robot travel M, clamp‑time bp, parallelism between
    mode B & C) are respected.

    The code is intentionally **data‑agnostic**: you just have to create
    the list of Job objects (with their Operation lists filled in) and pass
    it to the constructor.
    
    REQUEST
    -------
    Now, imagine I had a heuristic method that decide after each past decision the new decision: this decision consist of (next_job_id, next_op_id, next_mode) 
    next_mode is feasible with next_op, and  next_op is feasible with the current state of execution and the precedence relation of next_job. So I have this decision,
    and I want to code a "simulator" to obtain the execution dates of next_op, the date related to next_job, and the loading station. The simulator, coded in an OOP
        style, has an internal memory saving the current state and dates for each loading station, for the robot arm, both processes, and the "positioner". The simulator
        also randomly decides the loading station, but favor stations 1 and 3 over 2 when it is possible (to keep it free for a possible next large job). The simulator 
        also keeps in memory the state and dates of job and operation already inside the system (that are either on a loading station, carried by the arm, on a process, 
        or inside the positioner). The simulator must take into account the arm moves of Ms, the loading/unloding time of Ls, and the positioner time of the current job.
            If there is no history at first, the simulator assumes the robot arm is already positioned in front of the loading stations (no M time to come and get 
            the first op)... Finally, and it is very important, the simulator should be able to consider the possibility of simultaneous mode B (process 1 over positioner)
            and C (process 2 in parallel) and the fact that the job on mode B can't be free until the robot arm is also free and comes get it. 
"""

from __future__ import annotations

import random
import heapq
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple, Optional

# ---------------------------------------------------------------------------
# Basic timing constants (minutes) – tune to your shop‑floor values
# ---------------------------------------------------------------------------
M_DEFAULT: int = 10   # one robot travel between any two nodes
L_DEFAULT: int = 5    # load OR unload time at a station (arm stays)

# ---------------------------------------------------------------------------
# Generic single‑server resource with an availability date
# ---------------------------------------------------------------------------

@dataclass
class Resource:
    """A unary resource whose state is entirely described by the
    instant *busy_until* when it becomes free again."""
    name: str
    busy_until: int = 0

    def reserve(self, earliest: int, duration: int) -> Tuple[int, int]:
        """Book this resource *as soon as possible not before *earliest*.
        Returns (start, end)."""
        start = max(earliest, self.busy_until)
        end   = start + duration
        self.busy_until = end
        return start, end


@dataclass
class Station(Resource):
    idx: int = field(default=0)
    wide: bool = field(default=False)  # station 2 is the only *wide* one


class RobotArm(Resource):
    pass  # no extra attributes


class Process(Resource):
    pass  # distinguishes P1 and P2 by name only


class Positioner(Resource):
    pass


# ---------------------------------------------------------------------------
# Job / Operation data model (you build those from your input data)
# ---------------------------------------------------------------------------

@dataclass
class Operation:
    """A technological operation (weld) inside a job."""
    op_id: int
    process: int             # 1 or 2
    weld_time: int           # pure welding time
    bp: int = 0              # extra clamp time if executed in mode B
    mode: Optional[str] = None  # to be filled at execution ('A','B','C')

    # execution info (filled by the simulator)
    station: Optional[int] = None
    start:   Optional[int] = None
    end:     Optional[int] = None
    done:    bool = False


@dataclass
class Job:
    """A part going through 1 or 2 welding operations."""
    job_id: int
    large: bool                    # needs the wide station 2 when True
    ops: List[Operation]

    # runtime status ---------------------------------------------------------
    loaded_station: Optional[int] = None

    def next_pending_op(self) -> Operation | None:
        return next((o for o in self.ops if not o.done), None)

    @property
    def done(self) -> bool:
        return all(o.done for o in self.ops)


# ---------------------------------------------------------------------------
# The discrete‑event simulator
# ---------------------------------------------------------------------------

class Simulator:
    """Handles the temporal evolution of the cell under a sequence of
    *dispatch* decisions coming from a heuristic."""

    def __init__(
        self,
        jobs: List[Job],
        move_time: int = M_DEFAULT,
        load_time: int = L_DEFAULT,
        seed: int | None = None,
    ) -> None:
        self.M = move_time
        self.L = load_time

        # global time of the simulation
        self._time: int = 0

        # resources ----------------------------------------------------------
        self._arm        = RobotArm("Arm")
        self._process1   = Process("P1")
        self._process2   = Process("P2")
        self._positioner = Positioner("Positioner")
        self._stations   = [
            Station("S1", idx=1, wide=False),
            Station("S2", idx=2, wide=True),   # only one that fits *large*
            Station("S3", idx=3, wide=False),
        ]

        # job book and quick access map
        self._jobs: Dict[int, Job] = {j.job_id: j for j in jobs}

        # future callbacks:  (time, callable)
        self._events: List[Tuple[int, Callable[[], None]]] = []

        # textual history, for debugging or visualisation purposes
        self._history: List[str] = []

        self._rng = random.Random(seed)

    # ------------------------------------------------------------------
    # public interface
    # ------------------------------------------------------------------

    def schedule(self, decision: Tuple[int, int, str]) -> Dict[str, int]:
        """Accept a (job_id, op_id, mode) decision, update the agenda and
        return {'start': t_s, 'end': t_e, 'station': c}."""

        job_id, op_id, mode = decision
        if mode not in ("A", "B", "C"):
            raise ValueError("mode must be 'A', 'B' or 'C'")

        job = self._jobs.get(job_id)
        if job is None:
            raise RuntimeError(f"unknown job {job_id}")

        # precedence satisfied?
        op = next((o for o in job.ops if o.op_id == op_id), None)
        if op is None:
            raise RuntimeError(f"job {job_id} has no op {op_id}")
        if op.done:
            raise RuntimeError("operation already executed")
        earlier_unsolved = any(not o.done for o in job.ops if o.op_id < op_id)
        if earlier_unsolved:
            raise RuntimeError("technological predecessor not finished yet")

        # feasibility of the chosen mode wrt process type
        if op.process == 2 and mode != "C":
            raise RuntimeError("process 2 ops must run in mode C")

        # fast‑forward time by executing due events ----------------------
        self._flush_events()

        # reserve / pick a loading station ------------------------------
        st = self._choose_station(job)

        # compose the micro‑timeline for the selected mode --------------
        if mode == "A":
            start, end = self._play_mode_A(op, st)
        elif mode == "B":
            start, end = self._play_mode_B(op, st)
        else:  # mode C
            start, end = self._play_mode_C(op, st)

        # update job & op records --------------------------------------
        op.mode    = mode
        op.station = st.idx
        op.start   = start
        op.end     = end
        op.done    = True

        # print‑like history line (optional) ---------------------------
        self._history.append(
            f"[{start:>5}→{end:>5}]  job {job_id:>3}  op {op_id}  mode {mode}  station {st.idx}")

        return {"start": start, "end": end, "station": st.idx}

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _flush_events(self) -> None:
        """Execute all future events whose date ≤ current time."""
        while self._events and self._events[0][0] <= self._time:
            t, cb = heapq.heappop(self._events)
            self._time = t
            cb()  # the callback usually frees / books resources

    # ------------------------------------------------------------------
    # station selection: prefer 1 & 3 for small jobs, keep 2 free for *large*
    # ------------------------------------------------------------------

    def _choose_station(self, job: Job) -> Station:
        if job.loaded_station is not None:
            return self._stations[job.loaded_station - 1]

        # candidate list of currently free stations
        candidates = [s for s in self._stations if s.busy_until <= self._time and (s.wide or not job.large)]

        if not candidates:
            # wait for the very first station to become free
            soonest = min(self._stations, key=lambda s: s.busy_until)
            self._time = soonest.busy_until
            return self._choose_station(job)  # retry after time jump

        # prefer side stations for small parts to keep #2 available
        side_stations = [s for s in candidates if s.idx != 2]
        chosen = self._rng.choice(side_stations or candidates)
        job.loaded_station = chosen.idx
        return chosen

    # ------------------------------------------------------------------
    # micro‑planners for the three execution modes
    # ------------------------------------------------------------------

    def _travel_time_if_needed(self) -> int:
        """First call at time 0 – arm is already at the stations."""
        return 0 if self._time == 0 else self.M

    def _play_mode_A(self, op: Operation, st: Station) -> Tuple[int, int]:
        """Process 1, robot holds the part: arm busy from load till unload."""
        t0 = self._time + self._travel_time_if_needed()
        load_s, load_e = self._arm.reserve(t0, self.L + self.M)
        weld_s, weld_e = self._process1.reserve(load_e, op.weld_time)
        self._arm.busy_until = weld_e  # holds the torch/part
        unload_s, unload_e = self._arm.reserve(weld_e, self.L + self.M)
        st.busy_until = unload_e
        self._time = load_s  # timeline anchored at load start
        return load_s, unload_e

    def _play_mode_B(self, op: Operation, st: Station) -> Tuple[int, int]:
        """Process 1 on the positioner: arm busy only during load & unload.
        Weld + clamp done on the positioner; can overlap with process 2."""
        t0 = self._time + self._travel_time_if_needed()
        load_s, load_e = self._arm.reserve(t0, self.L + self.M)

        clamp_time = op.weld_time + op.bp  # bp already includes clamp setup
        clamp_s, clamp_e = self._positioner.reserve(load_e, clamp_time)

        # schedule the unload as *future event* so that it waits for both the
        # arm AND the completion of the clamp.
        def _perform_unload():
            unload_s, unload_e = self._arm.reserve(max(self._arm.busy_until, clamp_e), self.L + self.M)
            st.busy_until = unload_e
        heapq.heappush(self._events, (clamp_e, _perform_unload))

        self._time = load_s
        real_end = clamp_e + self.L + self.M  # after the future unload
        return load_s, real_end

    def _play_mode_C(self, op: Operation, st: Station) -> Tuple[int, int]:
        """Process 2, robot holds the part (can overlap with mode B weld)."""
        t0 = self._time + self._travel_time_if_needed()
        load_s, load_e = self._arm.reserve(t0, self.L + self.M)
        weld_s, weld_e = self._process2.reserve(load_e, op.weld_time)
        self._arm.busy_until = weld_e
        unload_s, unload_e = self._arm.reserve(weld_e, self.L + self.M)
        st.busy_until = unload_e
        self._time = load_s
        return load_s, unload_e

    # ------------------------------------------------------------------
    # read‑only helpers for the caller
    # ------------------------------------------------------------------

    @property
    def now(self) -> int:
        return self._time

    @property
    def history(self) -> List[str]:
        return self._history


# ---------------------------------------------------------------------------
# Example usage (remove or adapt to your testing framework)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Tiny demo data : 2 jobs, 2 ops each -----------------------------------
    j0 = Job(
        job_id=0,
        large=False,
        ops=[
            Operation(0, process=1, weld_time=30, bp=8),  # will run in B
            Operation(1, process=2, weld_time=25),        # must run in C
        ],
    )
    j1 = Job(
        job_id=1,
        large=False,
        ops=[
            Operation(0, process=2, weld_time=20),        # C
            Operation(1, process=1, weld_time=40, bp=10), # choose A or B
        ],
    )

    sim = Simulator([j0, j1], move_time=10, load_time=5, seed=42)

    decisions = [
        (0, 0, "B"),  # J0 first op on positioner
        (1, 0, "C"),  # J1 first op, can overlap with above
        (0, 1, "C"),  # J0 second op after its weld is unclamped
        (1, 1, "A"),  # etc.
    ]

    for d in decisions:
        info = sim.schedule(d)
        print(f"decision {d}  ->  {info}")

    print("\n--- HISTORY ---")
    for line in sim.history:
        print(line)
