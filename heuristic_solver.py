import argparse
from dataclasses import dataclass
import os
import re
import pickle
import time
import pandas as pd
from typing import  List

from conf import *
from models.instance import Instance, Job
from heuristic.local_search import ls as LS
from models.state import State, Decision
from gantt.gnn_gantt import gnn_gantt

@dataclass(frozen=True)
class Indexed:
    idx: int
    job: Job

def solve_all_test(path: str, gantt_path: str):
    for folder, _, _ in INSTANCES_SIZES:
        p: str = path+folder+"/"
        for i in os.listdir(p):
            if i.endswith('.json'):
                idx = re.search(r"instance_(\d+)\.json", i)
                for id in idx.groups():
                    gp = gantt_path + "/LS_"+folder+"_"+id+".png"
                    solve_one(path=path, gantt_path=gp, size=folder, id=id)

def solve_one(path: str, gantt_path: str, size: str, id: str):
    i: Instance = Instance.load(path + size + "/instance_" +id+ ".json")
    start_time = time.time()

    # State 1. Build and sort decisions by due dates (earliest due date first)
    decisions: list[Decision]   = []
    indexed_jobs: List[Indexed] = [Indexed(i, job) for i, job in enumerate(i.jobs)]
    sorted_jobs: list[Indexed]  = sorted(indexed_jobs, key=lambda i: i.job.due_date)
    for index in sorted_jobs:
        j: Job = index.job
        for o_id, o in enumerate(j.operations):
            decisions.append(Decision(job_id=index.idx, operation_id=o_id, job_id_in_graph=0, machine=o.type, parallel=False, comp=-1))

    # Stage 2. Improve the solution using our local search!
    state: State = LS(i, decisions)

    # State 3. Save the results
    computing_time = time.time() - start_time
    with open(path+size+"/heuristic_state_"+id+'.pkl', 'wb') as f:
        pickle.dump(state, f)
    obj: int = state.total_delay + state.cmax
    results = pd.DataFrame({'id': [id], 'obj': [obj], 'delay': [state.total_delay], 'cmax': [state.cmax], 'computing_time': [computing_time]})
    results.to_csv(path+size+"/heuristic_solution_"+id+".csv", index=False)
    gnn_gantt(path=gantt_path, state=state, instance=f"instance_{id}")

# TEST ONE WITH: python heuristic_solver.py --mode=test_one --size=s --id=1 --path=./
# SOLVE ALL WITH: python heuristic_solver.py --mode=test_all --path=.
if __name__ == "__main__":
    parser  = argparse.ArgumentParser(description="Exact solver (CP OR-tools version)")
    parser.add_argument("--path", help="path to load the instances", required=True)
    parser.add_argument("--mode", help="LS use mode, either train, test_one, or test_all", required=True)
    parser.add_argument("--size", help="size of the instance, either s, m, l or xl", required=False)
    parser.add_argument("--id", help="id of the instance to solve", required=False)
    args               = parser.parse_args()
    base_path: str     = args.path
    path: str          = base_path + "/data/instances/test/"
    gantt_path: str    = base_path + "/data/gantts"
    if args.mode == "test_all":
        solve_all_test(path=path, gantt_path=gantt_path)
    else:
        gantt_path = gantt_path + "/LS_"+args.size+"_"+args.id+".png"
        solve_one(path=path, gantt_path=gantt_path, size=args.size , id=args.id) 