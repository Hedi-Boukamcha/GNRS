
import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_ENABLE_INCOMPLETE_DYNAMIC_SHAPES"] = "1"
import math
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
import ray
from ray import ObjectRef

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
from models.environment import Candidate, Environment

# #################################
# =*= GNN + e-greedy DQN SOLVER =*=
# #################################
__author__  = "Hedi Boukamcha; Anas Neumann"
__email__   = "hedi.boukamcha.1@ulaval.ca; anas.neumann@polymtl.ca"
__version__ = "2.0.0"
__license__ = "MIT"

def search_possible_decisions(env: Environment, device: str) -> list[Decision]:
    M1_decisions: list[Decision] = []
    M2_decisions: list[Decision] = []
    for j in env.state.job_states:
        for o in j.operation_states:
            if o.remaining_time > 0:
                if o.operation.type == MACHINE_1:
                    M1_decisions.append(Decision(job_id=j.id, job_id_in_graph=j.graph_id, operation_id=o.id, machine=o.operation.type, parallel=True, comp=-1))
                    M1_decisions.append(Decision(job_id=j.id, job_id_in_graph=j.graph_id, operation_id=o.id, machine=o.operation.type, parallel=False, comp=-1))
                else:
                    if env.last_job_in_pos>=0:
                        M2_decisions.append(Decision(job_id=j.id, job_id_in_graph=j.graph_id, operation_id=o.id, machine=o.operation.type, parallel=True, comp=env.last_job_in_pos))
                    if not env.next_M2_parallel:
                        M2_decisions.append(Decision(job_id=j.id, job_id_in_graph=j.graph_id, operation_id=o.id, machine=o.operation.type, parallel=False, comp=-1))
                break
    decisions: list[Decision] = M2_decisions + (M1_decisions if (not env.next_M2_parallel or len(M2_decisions)==0) else [])
    decisionsT: Tensor = torch.tensor([[d.job_id_in_graph, d.machine, float(d.parallel), env.init_UB_cmax, env.init_UB_delay, env.ub_cmax, env.ub_delay, env.cmax, env.delay, env.total_jobs, env.rm_jobs, env.m2_parallel, env.action_time] for d in decisions], dtype=torch.float32, device=device)
    return decisions, decisionsT

def reward(env: Environment, device: str) -> Tensor:
    delta_cmax: float  = (TRADE_OFF * (env.state.ub_cmax - env.ub_cmax) + env.state.start_time - env.cmax)/(1 + TRADE_OFF)
    delta_delay: float = (TRADE_OFF * (env.state.ub_delay - env.ub_delay) + env.state.total_delay - env.delay)/(1 + TRADE_OFF)
    return torch.tensor([-REWARD_SCALE * (delta_cmax + delta_delay)], dtype=torch.float32, device=device)

@ray.remote
def step_as_task(agent: Agent, last_env: Environment, action_id: int, device: str, clone: bool=False, train: bool=False, pb_size: int=0) -> Environment:
    return take_one_step(agent=agent, last_env=last_env, action_id=action_id, device=device, clone=clone, train=train, pb_size=pb_size)

def take_one_step(agent: Agent, last_env: Environment, action_id: int, device: str, clone: bool=False, train: bool=False, pb_size: int=0) -> Environment:
    next_env: Environment = last_env.clone() if clone else last_env
    d: Decision           = next_env.possible_decisions[action_id]
    if d.parallel:
        if next_env.state.get_job_by_id(d.job_id).operation_states[d.operation_id].operation.type == MACHINE_1:
            next_env.last_job_in_pos  = d.job_id
            next_env.next_M2_parallel = True
        else:
            next_env.total_m2_parallel += 1
            next_env.m2                += 1
            next_env.next_M2_parallel   = False
    else:
        next_env.next_M2_parallel = False
        next_env.last_job_in_pos  = -1
        if next_env.state.get_job_by_id(d.job_id).operation_states[d.operation_id].operation.type == MACHINE_2:
            next_env.m2 += 1
    next_env.state         = simulate(next_env.state, d=d, clone=False)
    next_graph: HeteroData = next_env.state.to_hyper_graph(last_job_in_pos=next_env.last_job_in_pos, current_time=next_env.action_time, device=device)
    next_possible_decisions, next_decisionT = search_possible_decisions(env=next_env, device=device)
    if train:
        final: bool   = len(next_possible_decisions) == 0
        _r: Tensor    = reward(env=next_env, device=device)
        agent.memory.push(Transition(graph=next_env.graph, action_id=action_id, possible_actions=next_env.decisionsT, next_graph=next_graph, next_possible_actions=next_decisionT, reward=_r, final=final, nb_actions=len(next_env.possible_decisions), weight=T_WEIGTHS[pb_size]))
    next_env.update(graph=next_graph, possible_decisions=next_possible_decisions, decisionsT=next_decisionT)
    return next_env

def build_scores_and_filter(candidates: list[Candidate], limit: int) -> list[Candidate]:
    candidates.sort(key=lambda x: x.Q_value, reverse=True)
    for rank, c in enumerate(candidates):
        c.combined_score = 0.65 * rank
    candidates.sort(key=lambda x: x.ub_cmax)
    for rank, c in enumerate(candidates):
        c.combined_score += 0.1 * rank
    candidates.sort(key=lambda x: x.lb_delay)
    for rank, c in enumerate(candidates):
        c.combined_score += 0.25 * rank
    candidates.sort(key=lambda x: x.combined_score)
    return candidates[:limit]

def beam_solve_one(agent: Agent, gantt_path: str, path: str, size: str, id: str, improve: bool, device: str, beam_width: int=BEAM_WIDTH):
    start_time                  = time.time()
    i: Instance                 = Instance.load(path + size + "/instance_" +id+ ".json")
    init_state: State           = State(i, M, L, NB_STATIONS, BIG_STATION, [], automatic_build=True)
    init_state.compute_obj_values_and_upper_bounds(unloading_time=0, current_time=0)
    graph: HeteroData           = init_state.to_hyper_graph(last_job_in_pos=-1, current_time=0, device=device)
    env: Environment            = Environment(graph=graph, state=init_state, possible_decisions=None,
                                              decisionsT=None, init_UB_cmax=init_state.ub_cmax,
                                              init_UB_delay=init_state.ub_delay, n=len(i.jobs))
    env.possible_decisions, env.decisionsT = search_possible_decisions(env=env, device=device)
    beam: list[Environment]     = [env]
    finished: list[Environment] = []
    step: int                   = 0
    print(f"🚀 Starting Beam Search (k={beam_width}) for {size}.{id}...")
    while beam:
        limit: int = beam_width + i.n - step
        step += 1
        candidates: list[Candidate] = []
        for p_idx, parent_env in enumerate(beam): # 1. expansion
            q_values: Tensor = agent.get_all_q_values(parent_env.graph, parent_env.decisionsT)
            for a_idx, q_val in enumerate(q_values.tolist()):
                candidates.append(Candidate(parent_idx=p_idx,
                                            action_idx=a_idx,
                                            Q_value=q_val,
                                            ub_cmax=parent_env.state.ub_cmax,
                                            lb_delay=parent_env.state.total_delay))

        # 2. selection and prunning
        top_k = build_scores_and_filter(candidates=candidates, limit=limit)
        futures: list[ObjectRef] = []
        for c in top_k:
            parent_env = beam[c.parent_idx]
            futures.append(step_as_task.remote(agent=agent, last_env=parent_env, action_id=c.action_idx, clone=True, device='cpu')) # 3. simulations in parallel
        new_environments = ray.get(futures) # 4. wait for all simulations
        next_beam: list[Environment] = []
        for next_env in new_environments:
            if next_env.graph:
                next_env.graph = next_env.graph.to(device)
            if next_env.decisionsT is not None:
                next_env.decisionsT = next_env.decisionsT.to(device)
            if not next_env.possible_decisions:
                if improve: # improve finished states with local search
                    next_env.state = LS(i, next_env.state.decisions)
                finished.append(next_env)
            else:
                next_beam.append(next_env)
        beam = next_beam
        if step > 1000: break # Safety break to prevent infinite loops in broken instances
    if not finished:
        print("Warning: No finished solution found. Using last active beam.")
        best_env = new_environments[0]
    else:
        print(len(finished), "finished solutions found.")
        best_env = min(finished, key=lambda e: e.state.total_delay + e.state.cmax) # 5. select best among finished
    obj: int = best_env.state.total_delay + best_env.state.cmax
    print(f"Instance {size}.{id}: OBJ={obj}...")
    extension: str = "improved_beam_" if improve else "beam_"
    computing_time = time.time() - start_time
    gnn_gantt(gantt_path, best_env.state, f"instance {size}.{id}")
    results = pd.DataFrame({'id': [id], 
                            'obj': [obj], 
                            'delay': [best_env.state.total_delay], 
                            'cmax': [best_env.state.cmax], 
                            'computing_time': [computing_time]})
    results.to_csv(f"{path}{size}/{agent.prefix}gnn_solution_{extension}{id}.csv", index=False)

def repeated_solve_one(agent: Agent, gantt_path: str, path: str, size: str, id: str, improve: bool, device: str, retires: int=RETRIES):
    start_time        = time.time()
    best_state: State = None
    best_obj: int     = -1
    for retry in range(retires):
        g: bool              = (retry == 0)
        i: Instance          = Instance.load(path + size + "/instance_" +id+ ".json")
        init_state: State    = State(i, M, L, NB_STATIONS, BIG_STATION, [], automatic_build=True)
        init_state.compute_obj_values_and_upper_bounds(unloading_time=0, current_time=0)
        graph: HeteroData    = init_state.to_hyper_graph(last_job_in_pos=-1, current_time=0, device=device)
        env: Environment     = Environment(graph=graph, state=init_state, n=len(i.jobs))
        env.possible_decisions, env.decisionsT = search_possible_decisions(env=env, device=device)
        while env.possible_decisions:
            action_id: int = agent.select_next_decision(graph=env.graph, decisionsT=env.decisionsT)
            env            = take_one_step(agent=agent, last_env=env, action_id=action_id, device=device)
        if improve:
            env.state = LS(i, env.state.decisions) # improve with local search
        obj: int = env.state.total_delay + env.state.cmax
        print(f"Instance {size}.{id} (retry #{retry+1}/{retires}): OBJ={obj}...")
        if best_state is None or obj < best_obj:
            best_obj   = obj
            best_state = env.state
    extension: str = "improved_" if improve else ""
    computing_time = time.time() - start_time
    gnn_gantt(gantt_path, best_state, f"instance {size}.{id}")
    results = pd.DataFrame({'id': [id], 'obj': [best_obj], 'delay': [best_state.total_delay], 'cmax': [best_state.cmax], 'computing_time': [computing_time]})
    results.to_csv(f"{path}{size}/{agent.prefix}gnn_solution_{extension}{id}.csv", index=False)

def solve_one_for_training(agent: Agent, path: str, size: str, id: str, device: str, greedy: bool=False, eps_threshold: float=0.0) -> tuple[int, int]:
    i: Instance          = Instance.load(path + size + "/instance_" +id+ ".json")
    init_state: State    = State(i, M, L, NB_STATIONS, BIG_STATION, [], automatic_build=True)
    init_state.compute_obj_values_and_upper_bounds(unloading_time=0, current_time=0)
    graph: HeteroData    = init_state.to_hyper_graph(last_job_in_pos=-1, current_time=0, device=device)
    env: Environment     = Environment(graph=graph, state=init_state, n=len(i.jobs))
    env.possible_decisions, env.decisionsT = search_possible_decisions(env=env, device=device)
    while env.possible_decisions:
        action_id: int = agent.select_next_decision(graph=env.graph, decisionsT=env.decisionsT, possible_decisions=env.possible_decisions, greedy=greedy, eps_threshold=eps_threshold)
        env = take_one_step(agent=agent, last_env=env, action_id=action_id, pb_size=size, device=device, train=True)
    obj: int = env.state.total_delay + env.state.cmax
    return obj, (env.init_UB_cmax + env.init_UB_delay)

def solve_all_test(agent: Agent, gantt_path:str, path: str, improve: bool, beam: bool, device: str):
    extension: str = "beam_gnn" if beam else "improved_gnn" if improve else "gnn"
    for folder, _, _ in INSTANCES_SIZES:
        p: str = path+folder+"/"
        for i in os.listdir(p):
            if i.endswith('.json'):
                idx = re.search(r"instance_(\d+)\.json", i)
                for id in idx.groups():
                    if beam:
                        beam_solve_one(agent=agent, gantt_path=f"{gantt_path}{extension}_{folder}_{id}.png", path=path, size=folder, id=id, improve=improve, device=device) 
                    else:   
                        repeated_solve_one(agent=agent, gantt_path=f"{gantt_path}{agent.prefix}{extension}_{folder}_{id}.png", path=path, size=folder, id=id, improve=improve, device=device, retires=RETRIES)

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
        obj, ub = solve_one_for_training(agent=agent, path=path, size=size, id=instance_id, device=device, greedy=greedy, eps_threshold=eps_threshold)
        computing_time = time.time() - start_time
        agent.diversity.update(eps_threshold)
        if episode == 1 or episode % VALIDATE_RATE == 0:
            for vs in sizes[:complexity_limit]:
                print(f"Validating size {vs}...")
                val_obj = 0
                for id in range(1, 100):
                    v_id: str = str(id)
                    vo,_ = solve_one_for_training(agent=agent, path=path, size=vs, id=v_id, device=device, greedy=True, eps_threshold=0.0)
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

# TRAIN WITH: python gnn_solver.py --mode=train --interactive=true --load=false --path=. --custom=true
# TEST ONE WITH: python gnn_solver.py --mode=test_one --size=s --id=1 --improve=true --interactive=false --load=true --path=. --custom=true --beam=true
# SOLVE ALL WITH: python gnn_solver.py --mode=test_all --improve=true --interactive=false --load=true --path=. --custom=true --beam=true
if __name__ == "__main__":
    parser  = argparse.ArgumentParser(description="Exact solver (CP OR-tools version)")
    parser.add_argument("--path", help="path to load the instances", required=True)
    parser.add_argument("--interactive", help="display loss in real time", required=True)
    parser.add_argument("--mode", help="GNN use mode, either train, test_one, or test_all", required=True)
    parser.add_argument("--size", help="size of the instance, either s, m, l or xl", required=False)
    parser.add_argument("--id", help="id of the instance to solve", required=False)
    parser.add_argument("--load", help="do we load the weights of policy_net", required=True)
    parser.add_argument("--custom", help="use the custom Q-net instead of a basic one", required=True)
    parser.add_argument("--beam", help="use the gnn-guided beam search", required=False)
    parser.add_argument("--improve", help="improve the solution using local improvement operator", required=False)
    args               = parser.parse_args()
    base_path: str     = args.path
    instance_type: str = "debug/" if args.mode=="debug" else "train/" if args.mode == "train" else "test/"
    path: str          = base_path + "/data/instances/" + instance_type
    gantt_path: str    = base_path + "/data/gantts/"
    load_weights: bool = to_bool(args.load)
    custom: bool       = to_bool(args.custom)

    interactive: bool  = to_bool(args.interactive)
    # device: str      = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    device: str        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Current computing device is: {device}...")
    ray.init(num_cpus=8, ignore_reinit_error=True)
    agent: Agent       = Agent(device=device, interactive=interactive, load=load_weights, path=base_path+'/data/training/', train=(args.mode == "train"), custom=custom)
    if args.mode == "train":
        train(agent=agent, path=path, device=device)
    elif args.mode == "test_all":
        beam: bool = to_bool(args.beam)
        solve_all_test(agent=agent, path=path, gantt_path=gantt_path, improve=to_bool(args.improve), beam=beam, device=device)
    else:
        beam: bool     = to_bool(args.beam)
        improve: bool  = to_bool(args.improve)
        extension: str = "improved_gnn_" if improve else "gnn_"
        if beam:
            beam_solve_one(agent=agent, path=path, gantt_path=gantt_path+extension+args.size+"_"+args.id+".png", size=args.size , id=args.id, improve=improve, retires=RETRIES, device=device, train=False, eps_threshold=0.0) 
        else:
            repeated_solve_one(agent=agent, path=path, gantt_path=gantt_path+extension+args.size+"_"+args.id+".png", size=args.size , id=args.id, improve=improve, retires=RETRIES, device=device, train=False, eps_threshold=0.0) 
