
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
    def __init__(self, graph: HeteroData, possible_decisions: list[Decision], decisionsT: Tensor, cmax: int=0, delay: int=0, ub_cmax: int=0, ub_delay: int=0, n: int=0):
        self.graph              = graph
        self.decisionsT         = decisionsT
        self.cmax               = cmax
        self.delay              = delay
        self.possible_decisions = possible_decisions
        self.total_jobs         = n
        self.ub_cmax            = ub_cmax
        self.ub_delay           = ub_delay
        self.rm_jobs            = n
        self.m2_parallel        = 0
        self.action_time        = 0

    def update(self, graph: HeteroData, possible_decisions: list[Decision], decisionsT: Tensor, cmax: int, delay: int, m2_parallel: int, m2: int):
        self.graph              = graph
        self.decisionsT         = decisionsT
        self.cmax               = cmax
        self.delay              = delay
        self.possible_decisions = possible_decisions
        self.rm_jobs            = graph['job'].x.shape[0] if graph['job'].x is not None else 0
        self.m2_parallel        = m2_parallel / m2 if m2 > 0 else 0

def search_possible_decisions(state: State, possible_parallel: bool, needed_parallel: bool, device: str, env: Environment) -> list[Decision]:
    decisions: list[Decision] = []
    for j in state.job_states:
        for o in j.operation_states:
            if o.remaining_time > 0:
                if possible_parallel or o.operation.type == MACHINE_1:
                    decisions.append(Decision(job_id=j.id, job_id_in_graph=j.graph_id, operation_id=o.id, machine=o.operation.type, parallel=True))
                if not needed_parallel or o.operation.type == MACHINE_1:
                    decisions.append(Decision(job_id=j.id, job_id_in_graph=j.graph_id, operation_id=o.id, machine=o.operation.type, parallel=False))
                break
    decisionsT: Tensor = torch.tensor([[d.job_id_in_graph, d.machine, float(d.parallel), env.ub_cmax, env.ub_delay, env.cmax, env.delay, env.total_jobs, env.rm_jobs, env.m2_parallel, env.action_time] for d in decisions], dtype=torch.float32, device=device)
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

def reward(duration: int, cmax_old: int, cmax_new: int, delay_old: int, delay_new: int, ub_cmax: int, ub_delay: int, device: str) -> Tensor:#
    return torch.tensor([-REWARD_SCALE * ((cmax_new - cmax_old - duration)/ub_cmax + (delay_new - delay_old)/ub_delay)], dtype=torch.float32, device=device)

def solve_one(agent: Agent, gantt_path: str, path: str, size: str, id: str, improve: bool, device: str, train: bool=False, greedy: bool=False, retires: int=RETRIES, mimic_LS: bool=False, eps_threshold: float=0.0):
    i: Instance       = Instance.load(path + size + "/instance_" +id+ ".json")
    start_time        = time.time()
    best_state: State = None
    best_obj: int     = -1
    next_M2_parallel  = False
    m2: int           = 0
    m2_parallel: int  = 0
    for retry in range(retires):
        g: bool = (retry == 1) if not train else greedy
        last_job_in_pos: int = -1
        state: State = State(i, M, L, NB_STATIONS, BIG_STATION, [], automatic_build=True)
        graph: HeteroData = state.to_hyper_graph(last_job_in_pos=last_job_in_pos, current_time=0, device=device)
        ub_cmax, ub_delay = compute_upper_bounds(i)
        env: Environment = Environment(graph=graph, possible_decisions=None, decisionsT=None, ub_cmax=ub_cmax, ub_delay=ub_delay, n=len(i.jobs))
        env.possible_decisions, env.decisionsT = search_possible_decisions(state=state, possible_parallel=(last_job_in_pos>=0), needed_parallel=next_M2_parallel, env=env, device=device)
        while env.possible_decisions:
            action_id: int = agent.select_next_decision(state=state, graph=env.graph, possible_decisions=env.possible_decisions, decisionsT=env.decisionsT, eps_threshold=eps_threshold, train=train, greedy=g, mimic_LS=mimic_LS)
            d: Decision = env.possible_decisions[action_id]
            if d.parallel:
                if state.get_job_by_id(d.job_id).operation_states[d.operation_id].operation.type == MACHINE_1:
                    last_job_in_pos = d.job_id
                    next_M2_parallel = True
                else:
                    m2_parallel += 1
                    m2 += 1
                    next_M2_parallel = False
            else:
                next_M2_parallel = False
                last_job_in_pos = -1
                if state.get_job_by_id(d.job_id).operation_states[d.operation_id].operation.type == MACHINE_2:
                    m2 += 1
            state = simulate(state, d=d, clone=False)
            next_graph: HeteroData = state.to_hyper_graph(last_job_in_pos=last_job_in_pos, current_time=env.action_time, device=device)
            next_possible_decisions, next_decisionT = search_possible_decisions(state=state, possible_parallel=(last_job_in_pos>=0), needed_parallel=next_M2_parallel, env=env, device=device)
            if train:
                final: bool   = len(next_possible_decisions) == 0
                duration: int = state.get_job_by_id(d.job_id).operation_states[d.operation_id].operation.processing_time
                _r: Tensor    = reward(duration=duration, cmax_old=env.cmax, cmax_new=state.cmax, delay_old=env.delay, delay_new=state.total_delay, ub_cmax=ub_cmax, ub_delay=ub_delay, device=device)
                agent.memory.push(Transition(graph=env.graph, action_id=action_id, possible_actions=env.decisionsT, next_graph=next_graph, next_possible_actions=next_decisionT, reward=_r, final=final, nb_actions=len(env.possible_decisions)))
            env.action_time = state.min_action_time()
            env.update(graph=next_graph, possible_decisions=next_possible_decisions, decisionsT=next_decisionT, cmax=state.cmax, delay=state.total_delay, m2_parallel=m2_parallel, m2=m2)
        if improve:
            state = LS(i, state.decisions) # improve with local search
        obj: int = state.total_delay + state.cmax
        if best_state is None or obj < best_obj:
            best_obj   = obj
            best_state = state
    if not train:
        computing_time = time.time() - start_time
        with open(path+size+"/gnn_state_"+id+'.pkl', 'wb') as f:
            pickle.dump(best_state, f)
        gnn_gantt(gantt_path, best_state, f"instance_{id}")
        results = pd.DataFrame({'id': [id], 'obj': [best_obj], 'delay': [best_state.total_delay], 'cmax': [best_state.cmax], 'computing_time': [computing_time]})
        extension: str = "improved_" if improve else ""
        results.to_csv(path+size+"/gnn_solution_"+extension+id+".csv", index=False)

def solve_all_test(agent: Agent, gantt_path:str, path: str, improve: bool, device: str):
    extension: str = "improved_gnn" if improve else "gnn"
    for folder, _, _ in INSTANCES_SIZES:
        p: str = path+folder+"/"
        for i in os.listdir(p):
            if i.endswith('.json'):
                idx = re.search(r"instance_(\d+)\.json", i)
                for id in idx.groups():
                    solve_one(agent=agent, gantt_path=gantt_path+extension+"_"+folder+"_"+id+".png", path=path, size=folder, id=id, improve=improve, device=device, retires=RETRIES, train=False, mimic_LS=False, eps_threshold=0.0)

def train(agent: Agent, path: str, device: str):
    start_time = time.time()
    print("Loading dataset....")
    sizes: list[str]      = ["s", "m", "l", "xl"]
    complexity_limit: int = 1
    size: str             = "s"
    instance_id: str      = "1"
    for episode in range(1, NB_EPISODES+1):
        if episode % SWITCH_RATE == 0:
            size             = random.choice(sizes[:complexity_limit])
            instance_id: str = str(random.randint(1, 150))
        eps_threshold: float = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * episode / EPS_DECAY_RATE)
        greedy = True if episode < (0.85 * NB_EPISODES) else random.random() > 0.7
        solve_one(agent=agent, path=path, gantt_path="", size=size, id=instance_id, improve=False, device=device, retires=1, train=True, greedy=greedy, eps_threshold=eps_threshold, mimic_LS=(episode % SWITCH_RATE == 0))
        computing_time = time.time() - start_time
        agent.diversity.update(eps_threshold)
        if episode % COMPLEXITY_RATE == 0 and complexity_limit<len(sizes):
            complexity_limit += 1
        if len(agent.memory) > BATCH_SIZE:
            loss: float = agent.optimize_policy()
            agent.optimize_target()
            print(f"Training episode: {episode} [time={computing_time:.2f}] -- instance: ({size}, {instance_id}) -- diversity rate (epsilion): {eps_threshold:.3f} -- loss: {loss:.5f} -- LR: {agent.optimizer.param_groups[0]['lr']:.6f}")
        else:
            print(f"Training episode: {episode} [time={computing_time:.2f}] -- instance: ({size}, {instance_id}) -- diversity rate (epsilion): {eps_threshold:.3f} -- No optimization yet...")
        if episode % SAVING_RATE == 0 or episode == NB_EPISODES:
            agent.save()
    print("End!")

# TRAIN WITH: python gnn_solver.py --mode=train --interactive=true --load=false --path=.
# TEST ONE WITH: python gnn_solver.py --mode=test_one --size=s --id=1 --improve=true --interactive=false --load=true --path=.
# SOLVE ALL WITH: python gnn_solver.py --mode=test_all --improve=true --interactive=false --load=true --path=.
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
    gantt_path: str    = base_path + "/data/gantts/"
    load_weights: bool = to_bool(args.load)
    interactive: bool  = to_bool(args.interactive)
    # device: str      = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    device: str        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Current computing device is: {device}...")
    agent: Agent       = Agent(device=device, interactive=interactive, load=load_weights, path=base_path+'/data/training/', train=(args.mode == "train"))
    if args.mode == "train":
        train(agent=agent, path=path, device=device)
    elif args.mode == "test_all":
        solve_all_test(agent=agent, path=path, gantt_path=gantt_path, improve=to_bool(args.improve), device=device)
    else:
        improve: bool = to_bool(args.improve)
        extension: str = "improved_gnn_" if improve else "gnn_"
        solve_one(agent=agent, path=path, gantt_path=gantt_path+extension+"_"+args.size+"_"+args.id+".png", size=args.size , id=args.id, improve=improve, retires=RETRIES, device=device, train=False, mimic_LS=False, eps_threshold=0.0) 
