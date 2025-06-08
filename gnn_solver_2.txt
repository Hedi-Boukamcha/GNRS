import glob
import os
import pathlib
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
from utils.common import to_bool
from heuristic.local_search import ls as LS
from models.agent import DQNAgent

# #################################
# =*= GNN + e-greedy DQN SOLVER =*=
# #################################
__author__  = "Hedi Boukamcha; Anas Neumann"
__email__   = "hedi.boukamcha.1@ulaval.ca; anas.neumann@polymtl.ca"
__version__ = "1.0.0"
__license__ = "MIT"


# ---------- Hyper-paramètres ------------------
ALPHA               = 0.5
BATCH_SIZE          = 256
CAPACITY            = 100_000
NODE_EMBED_DIM      = 64
TARGET_UPDATE       = 2_000
GAMMA               = 0.99
EPISODES            = 10_000
EVAL_EVERY          = 200
CHECKPOINT_EVERY    = 500
LEARNING_RATE       = 1e-4
# -------------------------------------------------------------------------

def search_possible_decisions(instance: Instance, state: State) -> list[Decision]:
    decisions: list[Decision] = []
    for j in state.job_states:
        for o in j.operation_states:
            if o.remaining_time > 0:
                decisions.append(Decision(job_id=j.id, operation_id=o.id, process=o.operation.type, parallel=True))
                decisions.append(Decision(job_id=j.id, operation_id=o.id, process=o.operation.type, parallel=False))
                break
    return decisions

def solve_one(path: str, size: str, id: str, improve: bool):
    i: Instance = Instance.load(path + size + "/instance_" +id+ ".json")
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
            if state.get_job_by_id(d.job_id).operation_states[d.operation_id].operation.type == PROCEDE_1:
                last_job_in_pos = d.job_id
        else:
            last_job_in_pos = -1
        s_next: State = simulate(states[-1], d=d) 
        action_time = s_next.min_action_time()
        states.append(s_next)
        possible_decisions = search_possible_decisions(instance=i, state=states[-1])
    final: State = states[-1]
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
    return transitions

def solve_all_test(path: str, improve: bool):
    for folder, _, _ in INSTANCES_SIZES:
        p: str = path+folder+"/"
        for i in os.listdir(p):
            if i.endswith('.json'):
                id: str = re.search(r"instance_(\d+)\.json", i)
                solve_one(path=path, size=folder, id=id, improve=improve)

def train(path: str, interactive: bool):
    # DQN final task
    json_paths = sorted(glob.glob(f"{path}/*/instance_*.json"))

    # Init agent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent  = DQNAgent(node_embed_dim=NODE_EMBED_DIM,
                      gamma=GAMMA,
                      lr=LEARNING_RATE,
                      target_update=TARGET_UPDATE,
                      capacity=CAPACITY,
                      device=device)
    
    for ep in range(1, EPISODES + 1):
        # --------- choix aléatoire d'une instance -----------------
        file_path = random.choice(json_paths)
        path         = pathlib.Path(file_path)
        size      = path.parent.name           # ex. "S", "M", "L"
        id_       = path.stem.split("_")[1]    # numéro après 'instance_'

        transitions = solve_one(path, size, id_, agent, ALPHA)
        
        # Cumulative rewards ---------------------------------------
        cum_reward = 0.0
        for s, action, r, s_next, done in transitions:
            agent.memory.push(s, action, r, s_next, done)
            agent.optimize(BATCH_SIZE)
            cum_reward += r

        # --------- LOG console basique ----------------------------
        if ep % 10 == 0:
            print(f"[{time.strftime('%H:%M:%S')}] "
                  f"Ep {ep:05d} | return {cum_reward:8.1f} | ε={agent.eps:.3f}")

        # Epsilon greedy -------------------------------------------
        if interactive and ep % EVAL_EVERY == 0:
            print(f"\n=== Évaluation greedy sur {path.name} ===")
            solve_one(path=path, size=size, id=id_,
                      improve=True,
                      agent=agent,
                      alpha=ALPHA)

        #  CHECKPOINT périodique -----------------------------------
        if ep % CHECKPOINT_EVERY == 0:
            ckpt = {
                "policy_state":  agent.policy_net.state_dict(),
                "target_state":  agent.target_net.state_dict(),
                "optimizer":     agent.optimizer.state_dict(),
                "epsilon":       agent.eps,
                "episode":       ep
            }
            torch.save(ckpt, f"checkpoints/dqn_episode{ep}.pt")
            print(f"Checkpoint enregistré (episode {ep})")

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
        train(path=base_path, interactive=interactive)
    elif args.mode == "test_all":
        improve: bool = to_bool(args.improve)
        solve_all_test(path=path, improve=improve)
    else:
        size = args.size 
        id = args.id
        improve: bool = to_bool(args.improve)
        solve_one(path=path, size=size, id=id, improve=improve)
