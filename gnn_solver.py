
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_ENABLE_INCOMPLETE_DYNAMIC_SHAPES"] = "1"
import math
import glob
import pathlib
import pickle
import re
import argparse
import random
import pandas as pd
import time
from torch import Tensor
from torch_geometric.data import HeteroData
import warnings
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module=r"torch_geometric\.nn\.conv\.hetero_conv",
)
from typing import Tuple, List

from models.instance import Instance
from simulators.gnn_simulator import *
from utils.common import *
from conf import INSTANCES_SIZES
from models.state import State
from utils.common import to_bool
from heuristic.local_search import ls as LS
from models.agent import Agent
from models.memory import Transition
from gantt.gnn_gantt import gnn_gantt

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
                decisions.append(Decision(job_id=j.id, job_id_in_graph=j.graph_id, operation_id=o.id, process=o.operation.type, parallel=True))
                decisions.append(Decision(job_id=j.id, job_id_in_graph=j.graph_id, operation_id=o.id, process=o.operation.type, parallel=False))
                break
    decisionsT: Tensor = torch.tensor([[d.job_id_in_graph, d.process, float(d.parallel)] for d in decisions], dtype=torch.float32, device=device)
    return decisions, decisionsT

def compute_upper_bounds(i: Instance)-> Tuple[int, int]:
    jobs: List["Job"] = i.jobs
    nb_jobs: int      = len(jobs)
    per_j_term        = sum(j.pos_time for j in jobs)
    per_op_term       = sum((2*M) + op.processing_time for j in jobs for op in j.operations)
    ub_cmax           = 2 * L * nb_jobs + per_op_term + (per_j_term/2)
    sorted_jobs       = sorted(jobs, key=lambda j: j.due_date)
    delays: int       = 0
    for idx, job in enumerate(sorted_jobs):
        rank    = (idx * 0.9)/ nb_jobs if nb_jobs else 0.0
        delay   = max(0, ub_cmax * (1 - rank) - job.due_date)
        delays += delay
    ub_delay    = max(1, delays)
    return ub_cmax, ub_delay

def reward(duration: int, cmax_old: int, cmax_new: int, delay_old: int, delay_new: int, ub_cmax: int, ub_delay: int, device: str) -> Tensor:
    return torch.tensor([-1.0 * ((cmax_new - cmax_old - duration)/ub_cmax + (delay_new - delay_old)/ub_delay)], dtype=torch.float32, device=device)

def solve_one(agent: Agent, path: str, size: str, id: str, improve: bool, device: str, train: bool=False, eps_threshold: float=0.0):
    i: Instance = Instance.load(path + size + "/instance_" +id+ ".json")
    start_time = time.time()
    last_job_in_pos: int = -1
    action_time: int = 0
    state: State = State(i, M, L, NB_STATIONS, BIG_STATION, [], automatic_build=True)
    graph: HeteroData = state.to_hyper_graph(last_job_in_pos, action_time, device)
    poss_dess, dessT = search_possible_decisions(state=state, device=device)
    ub_cmax: int = 0
    ub_delay: int = 0
    if train:
        ub_cmax, ub_delay = compute_upper_bounds(i)
    env: Environment = Environment(graph=graph, possible_decisions=poss_dess, decisionsT=dessT)
    while env.possible_decisions:
        action_id: int = agent.select_next_decision(graph=env.graph, possible_decisions=env.possible_decisions, decisionsT=env.decisionsT, eps_threshold=eps_threshold, train=train)
        d: Decision = env.possible_decisions[action_id]
        if d.parallel:
            if state.get_job_by_id(d.job_id).operation_states[d.operation_id].operation.type == MACHINE_1:
                last_job_in_pos = d.job_id
        else:
            last_job_in_pos = -1
        state = simulate(state, d=d, clone=False)
        next_graph: HeteroData = state.to_hyper_graph(last_job_in_pos=last_job_in_pos, current_time=action_time, device=device)
        next_possible_decisions, next_decisionT = search_possible_decisions(state=state, device=device)
        if train:
            final: bool   = len(next_possible_decisions) == 0
            duration: int = state.get_job_by_id(d.job_id).operation_states[d.operation_id].operation.processing_time
            _r: Tensor    = reward(duration=duration, cmax_old=env.cmax, cmax_new=state.cmax, delay_old=env.delay, delay_new=state.total_delay, ub_cmax=ub_cmax, ub_delay=ub_delay, device=device)
            agent.memory.push(Transition(graph=env.graph, action_id=action_id, possible_actions=env.decisionsT, next_graph=next_graph, next_possible_actions=next_decisionT, reward=_r, final=final, nb_actions=len(env.possible_decisions)))
        action_time = state.min_action_time()
        env.update(next_graph, next_possible_decisions, next_decisionT, state.cmax, state.total_delay)
    if not train:
        if improve:
            state = LS(i, state.decisions) # improve with local search
            gnn_gantt(state, f"instance_{id}")
        state.display_calendars()
        computing_time = time.time() - start_time
        with open(path+size+"/gnn_state_"+id+'.pkl', 'wb') as f:
            pickle.dump(state, f)
        obj: int = state.total_delay + state.cmax
        results = pd.DataFrame({'id': [id], 'obj': [obj], 'delay': [state.total_delay], 'cmax': [state.cmax], 'computing_time': [computing_time]})
        extension: str = "improved_" if improve else ""
        results.to_csv(path+"gnn_solution_"+extension+id+".csv", index=False)

def solve_all_test(agent: Agent, path: str, improve: bool, device: str):
    for folder, _, _ in INSTANCES_SIZES:
        p: str = path+folder+"/"
        for i in os.listdir(p):
            if i.endswith('.json'):
                id: str = re.search(r"instance_(\d+)\.json", i)
                solve_one(agent=agent, path=path, size=folder, id=id, improve=improve, device=device, train=False, eps_threshold=0.0)

def train(agent: Agent, path: str, device: str):
    start_time = time.time()
    print("Loading dataset....")
    sizes: list[str]      = ["s", "m", "l", "xl"]
    complexity_limit: int = 1
    size: str             = "s"
    instance_id: str      = "1"
    lr_decay_factor       = 0.5
    for episode in range(1, NB_EPISODES+1):
        if episode % SWITCH_RATE == 0:
            size             = random.choice(sizes[:complexity_limit])
            instance_id: str = str(random.randint(1, 150))
        eps_threshold: float = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * episode / EPS_DECAY_RATE)
        solve_one(agent=agent, path=path, size=size, id=instance_id, improve=False, device=device, train=True, eps_threshold=eps_threshold)
        computing_time = time.time() - start_time
        agent.diversity.update(eps_threshold)
        if episode % COMPLEXITY_RATE == 0 and complexity_limit<len(sizes):
            complexity_limit += 1
            for g in agent.optimizer.param_groups:
                g['lr'] *= lr_decay_factor
        if len(agent.memory) > BATCH_SIZE:
            loss: float = agent.optimize_policy()
            agent.optimize_target()
            print(f"Training episode: {episode} [time={computing_time:.2f}] -- instance: ({size}, {instance_id}) -- diversity rate (epsilion): {eps_threshold:.3f} -- loss: {loss:.5f}")
        else:
            print(f"Training episode: {episode} [time={computing_time:.2f}] -- instance: ({size}, {instance_id}) -- diversity rate (epsilion): {eps_threshold:.3f} -- No optimization yet...")
        if episode % SAVING_RATE == 0 or episode == NB_EPISODES:
            agent.save()
    print("End!")

# TRAIN WITH: python gnn_solver.py --mode=train --interactive=true --load=False --path=./
# TEST ONE WITH: python gnn_solver.py --mode=test_one --size=s --id=1 --improve=true --interactive=false --load=False --path=./
# SOLVE ALL WITH: python gnn_solver.py --mode=test_all --improve=true --interactive=false --load=False --path=./
if __name__ == "__main__":
    parser  = argparse.ArgumentParser(description="Exact solver (CP OR-tools version)")
    parser.add_argument("--path", help="path to load the instances", required=True)
    parser.add_argument("--interactive", help="display loss in real time", required=True)
    parser.add_argument("--mode", help="GNN use mode, either train, test_one, or test_all", required=True)
    parser.add_argument("--size", help="size of the instance, either s, m, l or xl", required=False)
    parser.add_argument("--id", help="id of the instance to solve", required=False)
    parser.add_argument("--load", help="do we load the weights of policy_net", required=True)
    parser.add_argument("--improve", help="improve the solution using local improvement operator", required=False)
    args               = parser.parse_args()
    base_path: str     = args.path
    instance_type: str = "debug/" if args.mode=="debug" else "train/" if args.mode == "train" else "test/"
    path: str          = base_path + "/data/instances/" + instance_type
    load_weights: bool = to_bool(args.load)
    interactive: bool  = to_bool(args.interactive)
    # device: str      = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    device: str        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Current computing device is: {device}...")
    agent: Agent       = Agent(device=device, interactive=interactive, load=load_weights, path=base_path+'/data/training/', train=(args.mode == "train"))
    if args.mode == "train":
        train(agent=agent, path=path, device=device)
    elif args.mode == "test_all":
        solve_all_test(agent=agent, path=path, improve=to_bool(args.improve), device=device)
    else:
        solve_one(agent=agent, path=path, size=args.size , id=args.id, improve=to_bool(args.improve), device=device, train=False, eps_threshold=0.0) 
