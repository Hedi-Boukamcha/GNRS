
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
from gantt_builder.gnn_gantt import gnn_gantt

# #################################
# =*= GNN + e-greedy DQN SOLVER =*=
# #################################
__author__  = "Hedi Boukamcha; Anas Neumann"
__email__   = "hedi.boukamcha.1@ulaval.ca; anas.neumann@polymtl.ca"
__version__ = "1.0.0"
__license__ = "MIT"

class Environment:
    def __init__(self, graph: HeteroData, possible_decisions: list[Decision], decisionsT: Tensor, cmax: int=0, delay: int=0, init_UB_cmax: int=0, init_UB_delay: int=0, n: int=0):
        self.graph              = graph
        self.decisionsT         = decisionsT
        self.cmax               = cmax
        self.delay              = delay
        self.possible_decisions = possible_decisions
        self.total_jobs         = n
        self.init_UB_cmax       = init_UB_cmax
        self.init_UB_delay      = init_UB_delay
        self.rm_jobs            = n
        self.m2_parallel        = 0
        self.action_time        = 0
        self.ub_cmax            = init_UB_cmax
        self.ub_delay           = init_UB_delay

    def update(self, graph: HeteroData, possible_decisions: list[Decision], decisionsT: Tensor, cmax: int, delay: int, ub_cmax: int, ub_delay: int, m2_parallel: int, m2: int):
        self.graph              = graph
        self.decisionsT         = decisionsT
        self.cmax               = cmax
        self.delay              = delay
        self.possible_decisions = possible_decisions
        self.rm_jobs            = graph['job'].x.shape[0] if graph['job'].x is not None else 0
        self.m2_parallel        = m2_parallel / m2 if m2 > 0 else 0
        self.ub_cmax            = ub_cmax
        self.ub_delay           = ub_delay

def search_possible_decisions(state: State, possible_parallel: bool, needed_parallel: bool, device: str, env: Environment, comp: int=None) -> list[Decision]:
    M1_decisions: list[Decision] = []
    M2_decisions: list[Decision] = []
    for j in state.job_states:
        for o in j.operation_states:
            if o.remaining_time > 0:
                if o.operation.type == MACHINE_1:
                    M1_decisions.append(Decision(job_id=j.id, job_id_in_graph=j.graph_id, operation_id=o.id, machine=o.operation.type, parallel=True, comp=-1))
                    M1_decisions.append(Decision(job_id=j.id, job_id_in_graph=j.graph_id, operation_id=o.id, machine=o.operation.type, parallel=False, comp=-1))
                else:
                    if possible_parallel:
                        M2_decisions.append(Decision(job_id=j.id, job_id_in_graph=j.graph_id, operation_id=o.id, machine=o.operation.type, parallel=True, comp=comp))
                    if not needed_parallel :
                        M2_decisions.append(Decision(job_id=j.id, job_id_in_graph=j.graph_id, operation_id=o.id, machine=o.operation.type, parallel=False, comp=-1))
                break
    decisions: list[Decision] = M2_decisions + (M1_decisions if (not needed_parallel or len(M2_decisions)==0) else [])
    decisionsT: Tensor = torch.tensor([[d.job_id_in_graph, d.machine, float(d.parallel), env.init_UB_cmax, env.init_UB_delay, env.ub_cmax, env.ub_delay, env.cmax, env.delay, env.total_jobs, env.rm_jobs, env.m2_parallel, env.action_time] for d in decisions], dtype=torch.float32, device=device)
    return decisions, decisionsT

def reward(env: Environment, cmax_new: int, delay_new: int, ub_cmax_new: int, ub_delay_new: int, device: str) -> Tensor:
    delta_cmax: float  = (TRADE_OFF * (ub_cmax_new - env.ub_cmax) + cmax_new - env.cmax)/(1 + TRADE_OFF)
    delta_delay: float = (TRADE_OFF * (ub_delay_new - env.ub_delay) + delay_new - env.delay)/(1 + TRADE_OFF)
    return torch.tensor([-REWARD_SCALE * (delta_cmax + delta_delay)], dtype=torch.float32, device=device)

def solve_one(agent: Agent, gantt_path: str, path: str, size: str, id: str, improve: bool, device: str, train: bool=False, greedy: bool=False, retires: int=RETRIES, eps_threshold: float=0.0):
    start_time        = time.time()
    best_state: State = None
    best_obj: int     = -1
    for retry in range(retires):
        g: bool              = (retry == 0) if not train else greedy
        # print("Retrying with greedy:", g, "for instance:", id, "retry:", retry+1, "/", retires)
        i: Instance          = Instance.load(path + size + "/instance_" +id+ ".json")
        next_M2_parallel     = False
        m2: int              = 0
        m2_parallel: int     = 0
        last_job_in_pos: int = -1
        state: State         = State(i, M, L, NB_STATIONS, BIG_STATION, [], automatic_build=True)
        state.compute_obj_values_and_upper_bounds(unloading_time=0, current_time=0)
        graph: HeteroData    = state.to_hyper_graph(last_job_in_pos=last_job_in_pos, current_time=0, device=device)
        env: Environment     = Environment(graph=graph, possible_decisions=None, decisionsT=None, init_UB_cmax=state.ub_cmax, init_UB_delay=state.ub_delay, n=len(i.jobs))
        env.possible_decisions, env.decisionsT = search_possible_decisions(state=state, possible_parallel=(last_job_in_pos>=0), needed_parallel=next_M2_parallel, env=env, comp=-1, device=device)
        while env.possible_decisions:
            action_id: int = agent.select_next_decision(graph=env.graph, possible_decisions=env.possible_decisions, decisionsT=env.decisionsT, eps_threshold=eps_threshold, train=train, greedy=g)
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
            next_possible_decisions, next_decisionT = search_possible_decisions(state=state, possible_parallel=(last_job_in_pos>=0), needed_parallel=next_M2_parallel, env=env, comp=last_job_in_pos, device=device)
            if train:
                final: bool   = len(next_possible_decisions) == 0
                _r: Tensor    = reward(env=env, cmax_new=state.start_time, delay_new=state.total_delay, ub_cmax_new=state.ub_cmax, ub_delay_new=state.ub_delay, device=device)
                agent.memory.push(Transition(graph=env.graph, action_id=action_id, possible_actions=env.decisionsT, next_graph=next_graph, next_possible_actions=next_decisionT, reward=_r, final=final, nb_actions=len(env.possible_decisions), weight=T_WEIGTHS[size]))
            env.action_time = state.min_action_time()
            env.update(graph=next_graph, possible_decisions=next_possible_decisions, decisionsT=next_decisionT, cmax=state.cmax, delay=state.total_delay, ub_cmax=state.ub_cmax, ub_delay=state.ub_delay, m2_parallel=m2_parallel, m2=m2)
        if improve:
            state = LS(i, state.decisions) # improve with local search
        obj: int = state.total_delay + state.cmax
        if best_state is None or obj < best_obj:
            best_obj   = obj
            best_state = state
    if not train:
        extension: str = "improved_" if improve else ""
        computing_time = time.time() - start_time
        with open(path+size+"/gnn_state_"+extension+id+'.pkl', 'wb') as f:
            pickle.dump(best_state, f)
        gnn_gantt(gantt_path, best_state, f"instance_{id}")
        results = pd.DataFrame({'id': [id], 'obj': [best_obj], 'delay': [best_state.total_delay], 'cmax': [best_state.cmax], 'computing_time': [computing_time]})
        results.to_csv(path+size+"/gnn_solution_"+extension+id+".csv", index=False)
    return best_obj, (env.init_UB_cmax+env.init_UB_delay)

def solve_all_test(agent: Agent, gantt_path:str, path: str, improve: bool, device: str):
    extension: str = "improved_gnn" if improve else "gnn"
    for folder, _, _ in INSTANCES_SIZES:
        p: str = path+folder+"/"
        for i in os.listdir(p):
            if i.endswith('.json'):
                idx = re.search(r"instance_(\d+)\.json", i)
                for id in idx.groups():
                    solve_one(agent=agent, gantt_path=gantt_path+extension+"_"+folder+"_"+id+".png", path=path, size=folder, id=id, improve=improve, device=device, retires=RETRIES, train=False, eps_threshold=0.0)

def train(agent: Agent, path: str, device: str):
    start_time = time.time()
    print("Loading dataset....")
    sizes: list[str]      = ["s", "m", "l", "xl"]
    complexity_limit: int = 1
    size: str             = "s"
    instance_id: str      = "1"
    sr: int               = SWITCH_RATE
    for episode in range(1, NB_EPISODES+1):
        if episode % sr == 0:
            sr              += 1000
            size             = random.choice(sizes[:complexity_limit])
            instance_id: str = str(random.randint(1, 150))
        eps_threshold: float = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * episode / EPS_DECAY_RATE)
        greedy = True if episode < (0.8 * NB_EPISODES) else random.random() > 0.7
        obj, ub = solve_one(agent=agent, path=path, gantt_path="", size=size, id=instance_id, improve=False, device=device, retires=1, train=True, greedy=greedy, eps_threshold=eps_threshold)
        computing_time = time.time() - start_time
        agent.diversity.update(eps_threshold)
        if episode == 1 or episode % VALIDATE_RATE == 0:
            for vs in sizes[:complexity_limit]:
                print(f"Validating size {vs}...")
                val_obj = 0
                for id in range(1, 100):
                    v_id: str = str(id)
                    vo,_ = solve_one(agent=agent, path=path, gantt_path="", size=vs, id=v_id, improve=False, device=device, retires=1, train=True, greedy=True, eps_threshold=0.0)
                    val_obj += vo
                val_obj /= 100.0
                agent.add_obj(size=vs, obj=val_obj)
                print(f"Valdation of size {vs} = AVG = {val_obj}...") 
        if episode % COMPLEXITY_RATE == 0 and complexity_limit<len(sizes):
            complexity_limit += 1
        if len(agent.memory) > BATCH_SIZE:
            loss: float = agent.optimize_policy()
            agent.optimize_target()
            if episode>WARMUP_EPISODES and episode%LR_REDUCE_RATE==0 and agent.optimizer.param_groups[0]['lr']>MIN_LR:                    
                agent.optimizer.param_groups[0]['lr'] *= 0.5
            print(f"Training episode: {episode} [time={computing_time:.2f}] -- instance: ({size}, {instance_id}) -- obj: ({obj} / {int(ub)}) -- diversity rate (epsilion): {eps_threshold:.3f} -- loss: {loss:.5f} -- LR: {agent.optimizer.param_groups[0]['lr']:.0e}")
        else:
            print(f"Training episode: {episode} [time={computing_time:.2f}] -- instance: ({size}, {instance_id}) -- obj: ({obj} / {int(ub)}) -- diversity rate (epsilion): {eps_threshold:.3f} -- No optimization yet...")
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
        solve_one(agent=agent, path=path, gantt_path=gantt_path+extension+args.size+"_"+args.id+".png", size=args.size , id=args.id, improve=improve, retires=RETRIES, device=device, train=False, eps_threshold=0.0) 
