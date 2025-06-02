import os
import pickle
import re
import argparse
import random
import pandas as pd
import time

from models.instance import Instance
from simulators.gnn_simulator import *
from utils.common import *
from conf import INSTANCES_SIZES
from models.state import State

# #################################
# =*= GNN + e-greedy DQN SOLVER =*=
# #################################
__author__ = "Hedi Boukamcha - hedi.boukamcha.1@ulaval.ca"
__version__ = "1.0.0"
__license__ = "MIT"

def search_possible_decisions(instance: Instance, state: State) -> list[Decision]:
    decisions: list[Decision] = []
    for j in state.job_states:
        for o in j.operation_states:
            if o.remaining_time > 0:
                decisions.append(Decision(job_id=j.id, operation_id=o.id, parallel=True))
                decisions.append(Decision(job_id=j.id, operation_id=o.id, parallel=False))
                break
    return decisions

def solve_one(path: str, size: str, id: str):
    i: Instance = Instance.load(path+size+"/instance_"+id+'.json')
    start_time = time.time()
    states: list[State] = []
    last_job_in_pos: int = -1
    action_time: int = 0
    states.append(State(i, M, L, NB_STATIONS, BIG_STATION, [], automatic_build=True))
    possible_decisions: list[Decision] = search_possible_decisions(instance=i, state=states[-1])
    while possible_decisions:
        states[-1].to_hyper_graph(last_job_in_pos, action_time)
        d: Decision   = random.choice(possible_decisions)
        if d.parallel:
            if states[-1].get_job_by_id(d.job_id).operation_states[d.operation_id].operation.type == PROCEDE_1:
                last_job_in_pos = d.job_id
        else:
            last_job_in_pos = -1
        s_next: State = simulate(states[-1], d=d) 
        action_time = s_next.min_action_time()
        states.append(s_next)
        possible_decisions = search_possible_decisions(instance=i, state=states[-1])
    final: State = states[-1]
    final.display_calendars()
    computing_time = time.time() - start_time
    with open(path+size+"/gnn_state_"+id+'.pkl', 'wb') as f:
        pickle.dump(final, f)
    obj: int = (final.total_delay * (100 - i.a)) + (final.cmax * i.a)
    results = pd.DataFrame({'id': [id], 'obj': [obj], 'a': [i.a],'delay': [final.total_delay], 'cmax': [final.cmax], 'computing_time': [computing_time]})
    results.to_csv(path+"exact_solution_"+id+".csv", index=False)

def solve_all_test(path: str):
    for folder, _, _ in INSTANCES_SIZES:
        p: str = path+folder+"/"
        for i in os.listdir(p):
            if i.endswith('.json'):
                id: str = re.search(r"instance_(\d+)\.json", i)
                solve_one(path=path, size=folder, id=id)

def train(path: str, interactive: bool):
    # DQN final task
    pass

# TRAIN WITH: python gnn_solver.py --mode=train --interactive=false --path=./
# TEST ONE WITH: python gnn_solver.py --mode=test_one --size=s --id=1 --interactive=false --path=./
# SOLVE ALL WITH: python gnn_solver.py --mode=test_all --interactive=false --path=./
if __name__ == "__main__":
    parser  = argparse.ArgumentParser(description="Exact solver (CP OR-tools version)")
    parser.add_argument("--path", help="path to load the instances", required=True)
    parser.add_argument("--interactive", help="display loss in real time", required=True)
    parser.add_argument("--mode", help="GNN use mode, either train, test_one, or test_all", required=True)
    parser.add_argument("--size", help="size of the instance, either s, m, l or xl", required=False)
    parser.add_argument("--id", help="id of the instance to solve", required=False)
    args = parser.parse_args()
    base_path: str = args.path
    path: str      = base_path + "/data/instances/test/"
    if args.mode == "train":
        interactive: bool = args.interactive.lower() in ['true', 't', 'yes', '1']
        train(path=base_path, interactive=interactive)
    elif args.mode == "test_all":
        solve_all_test(path=path)
    else:
        size = args.size 
        id = args.id
        solve_one(path=path, size=size, id=id)
