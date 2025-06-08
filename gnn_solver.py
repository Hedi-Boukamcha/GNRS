import glob
import os
import pathlib
import pickle
import re
import argparse
import random
import pandas as pd
import time
from torch import Tensor

from models.instance import Instance
from simulators.gnn_simulator import *
from utils.common import *
from conf import INSTANCES_SIZES
from models.state import State
from utils.common import to_bool
from heuristic.local_search import ls as LS
from models.agent import Agent
from models.memory import ReplayMemory, Transition

# #################################
# =*= GNN + e-greedy DQN SOLVER =*=
# #################################
__author__  = "Hedi Boukamcha; Anas Neumann"
__email__   = "hedi.boukamcha.1@ulaval.ca; anas.neumann@polymtl.ca"
__version__ = "1.0.0"
__license__ = "MIT"

class Environment:
    def __init__(self, graph: HeteroData, possible_decisions: list[Decision], decisionsT: Tensor, cmax: int=0, delay: int=0):
        self.graph              = graph
        self.decisionsT         = decisionsT
        self.cmax               = cmax
        self.delay              = delay
        self.possible_decisions = possible_decisions

    def update(self, graph: HeteroData, possible_decisions: list[Decision], decisionsT: Tensor, cmax: int, delay: int):
        self.graph              = graph
        self.decisionsT         = decisionsT
        self.cmax               = cmax
        self.delay              = delay
        self.possible_decisions = possible_decisions

def search_possible_decisions(state: State, device: str) -> list[Decision]:
    decisions: list[Decision] = []
    for j in state.job_states:
        for o in j.operation_states:
            if o.remaining_time > 0:
                decisions.append(Decision(job_id=j.id, operation_id=o.id, process=o.operation.type, parallel=True))
                decisions.append(Decision(job_id=j.id, operation_id=o.id, process=o.operation.type, parallel=False))
                break
    decisionsT: Tensor = torch.tensor([[d.job_id, d.process, float(d.parallel)] for d in decisions], dtype=torch.float32, device=device)
    return decisions, decisionsT

def select_next_decision(agent: Agent, graph: HeteroData, alpha: Tensor, possible_decisions: list[Decision], decisionsT: Tensor, eps_threshold: float, train: bool) -> int:
    if not train or random.random() > eps_threshold:
        Q_values: Tensor = agent.policy_net(graph, decisionsT, alpha)
        return torch.argmax(Q_values.view(-1)).item()
    else:
        return random.randint(0, len(possible_decisions)-1)

def reward(duration: int, cmax_old: int, cmax_new: int, delay_old: int, delay_new: int, alpha: float, device: str) -> Tensor:
    return torch.tensor([-1.0 * (alpha * (cmax_new - cmax_old - duration) + (1-alpha) * (delay_new - delay_old))], dtype=torch.float32, device=device)

def solve_one(agent: Agent, path: str, size: str, id: str, improve: bool, device: str, train: bool=False, memory: ReplayMemory=None, eps_threshold: float=0.0):
    i: Instance = Instance.load(path + size + "/instance_" +id+ ".json")
    start_time = time.time()
    last_job_in_pos: int = -1
    action_time: int = 0
    alpha: Tensor = torch.tensor([i.a], dtype=torch.float32, device=device)
    state: State = State(i, M, L, NB_STATIONS, BIG_STATION, [], automatic_build=True)
    possible_decisions, decisionT = search_possible_decisions(state=state, device=device)
    env: Environment = Environment(graph=state.to_hyper_graph(last_job_in_pos, action_time), possible_decisions=possible_decisions, decisionsT=decisionT)
    while env.possible_decisions:
        action_id: int = select_next_decision(agent, env.graph, env.possible_decisions, env.decisionsT, alpha, eps_threshold, train)
        d: Decision = possible_decisions[action_id]
        if d.parallel:
            if state.get_job_by_id(d.job_id).operation_states[d.operation_id].operation.type == PROCEDE_1:
                last_job_in_pos = d.job_id
        else:
            last_job_in_pos = -1
        state = simulate(state, d=d)
        next_graph: HeteroData = state.to_hyper_graph(last_job_in_pos, action_time)
        next_possible_decisions, next_decisionT = search_possible_decisions(state=state, device=device)
        if train and memory is not None:
            duration: int = state.get_job_by_id(d.job_id).operation_states[d.operation_id].operation.processing_time
            _r: Tensor = reward(duration, env.cmax, state.cmax, env.delay, state.total_delay, i.a, device)
            memory.push(Transition(graph=env.graph, action_id=action_id,  alpha=alpha, possible_actions=decisionT, next_graph=next_graph, reward=_r))
        action_time = state.min_action_time()
        env.update(next_graph, next_possible_decisions, next_decisionT, state.cmax, state.total_delay)
    if not train:
        if improve:
            state = LS(i, state.decisions) # improve with local search
        state.display_calendars()
        computing_time = time.time() - start_time
        with open(path+size+"/gnn_state_"+id+'.pkl', 'wb') as f:
            pickle.dump(state, f)
        obj: int = (state.total_delay * (100 - i.a)) + (state.cmax * i.a)
        results = pd.DataFrame({'id': [id], 'obj': [obj], 'a': [i.a],'delay': [state.total_delay], 'cmax': [state.cmax], 'computing_time': [computing_time]})
        extension: str = "improved_" if improve else ""
        results.to_csv(path+"exact_solution_"+extension+id+".csv", index=False)

def solve_all_test(path: str, improve: bool):
    for folder, _, _ in INSTANCES_SIZES:
        p: str = path+folder+"/"
        for i in os.listdir(p):
            if i.endswith('.json'):
                id: str = re.search(r"instance_(\d+)\.json", i)
                solve_one(path=path, size=folder, id=id, improve=improve)

# TRAIN WITH: python gnn_solver.py --mode=train --interactive=false --path=./
# TEST ONE WITH: python gnn_solver.py --mode=test_one --size=s --id=1 --improve=true  --interactive=false --path=./
# SOLVE ALL WITH: python gnn_solver.py --mode=test_all --improve=true --interactive=false --path=./
if __name__ == "__main__":
    parser  = argparse.ArgumentParser(description="Exact solver (CP OR-tools version)")
    parser.add_argument("--path", help="path to load the instances", required=True)
    parser.add_argument("--interactive", help="display loss in real time", required=True)
    parser.add_argument("--mode", help="GNN use mode, either train, test_one, or test_all", required=True)
    parser.add_argument("--size", help="size of the instance, either s, m, l or xl", required=False)
    parser.add_argument("--id", help="id of the instance to solve", required=False)
    parser.add_argument("--improve", help="improve the solution using local improvement operator", required=False)
    args = parser.parse_args()
    base_path: str = args.path
    path: str      = base_path + "/data/instances/debug/" #"/data/instances/debug OR test"
    if args.mode == "train":
        interactive: bool = to_bool(args.interactive)
        # train(path=base_path, interactive=interactive)
    elif args.mode == "test_all":
        improve: bool = to_bool(args.improve)
        solve_all_test(path=path, improve=improve)
    else:
        size = args.size 
        id = args.id
        improve: bool = to_bool(args.improve)
        solve_one(path=path, size=size, id=id, improve=improve)
