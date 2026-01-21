import random
import copy
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

from models import instance
from models.state import State, Decision
from models.instance import Instance
from conf import *
from simulators.gnn_simulator import simulate
from heuristic.local_search import ls as LS

# ##################################
# =*= TABU SEARCH METHOD =*=
# ##################################
__author__  = "Hedi Boukamcha; Anas Neumann"
__email__   = "hedi.boukamcha.1@ulaval.ca; anas.neumann@polymtl.ca"
__version__ = "1.0.0"
__license__ = "MIT"

@dataclass
class SmartMove:
    job_id: int
    op1_move: Optional[Tuple[int, int]] = None
    op2_move: Optional[Tuple[int, int]] = None
    
    def __str__(self):
        return f"Move(Job {self.job_id}: Op1={self.op1_move}, Op2={self.op2_move})"

def _get_possible_neighbors(decisions: List[Decision], sample_size: int) -> List[SmartMove]:
    n = len(decisions)
    moves: List[SmartMove] = []
    job_indices: Dict[int, Dict[int, int]] = {}
    for idx, d in enumerate(decisions):
        if d.job_id not in job_indices:
            job_indices[d.job_id] = {}
        job_indices[d.job_id][d.operation_id] = idx
    unique_job_ids = list(job_indices.keys())
    attempts = 0
    while len(moves) < sample_size and attempts < sample_size * 5:
        attempts += 1
        job_id = random.choice(unique_job_ids)
        ops = job_indices[job_id]
        has_op2 = 1 in ops
        mode = random.choice(['MOVE_OP1', 'MOVE_OP2'])
        if not has_op2: mode = 'MOVE_OP1'
        if mode == 'MOVE_OP2' and has_op2:       # move the 2nd operation
            curr_op2  = ops[1]
            curr_op1  = ops[0]
            min_bound = curr_op1 + 1             # cannot go before Op 1
            max_bound = min(curr_op2 + 4, n - 1) # move in the future (max 4 positions ahead)
            if min_bound >= max_bound: continue  # No valid moves
            new_pos = random.randint(min_bound, max_bound)
            if new_pos == curr_op2: continue     # No change
            moves.append(SmartMove(job_id=job_id, op2_move=(curr_op2, new_pos)))
        elif mode == 'MOVE_OP1':                 # move the 1st operation
            curr_op1 = ops[0]
            target_op1 = random.randint(0, n - 1)
            if target_op1 == curr_op1: continue
            if has_op2:                          # Adjust Op 2 accordingly   
                curr_op2   = ops[1]
                gap        = curr_op2 - curr_op1 
                target_op2 = target_op1 + gap
                if target_op2 >= n: 
                    diff        = target_op2 - (n - 1)
                    target_op1 -= diff
                    target_op2  = n - 1
                if target_op1 < 0: continue
                moves.append(SmartMove(
                    job_id=job_id, 
                    op1_move=(curr_op1, target_op1), 
                    op2_move=(curr_op2, target_op2)))
            else:
                moves.append(SmartMove(job_id=job_id, op1_move=(curr_op1, target_op1))) 
    return moves

def _apply_move(decisions: List[Decision], move: SmartMove) -> List[Decision]:
    new_decisions = decisions.copy()
    actions = []
    if move.op1_move: actions.append(move.op1_move)
    if move.op2_move: actions.append(move.op2_move)
    actions.sort(key=lambda x: x[0], reverse=True) # Remove from highest OLD index to lowest OLD to avoid index shifting issues
    popped_items = {}                              # map original_index -> decision_object
    for curr_idx, _ in actions:
        d = new_decisions.pop(curr_idx)
        popped_items[curr_idx] = d
    actions.sort(key=lambda x: x[1]) # Insert in ascending order of NEW position to respect the intended order 
    for curr_idx, new_idx in actions:
        d = popped_items[curr_idx]
        new_decisions.insert(new_idx, d)
    return new_decisions

def _run_ls_and_evaluate(instance: Instance, decisions: List[Decision]) -> Tuple[State, int, bool]:
    if _check_if_all_possible(instance, decisions):
        improved_state: State = LS(instance, decisions)
        return improved_state, (improved_state.total_delay + improved_state.cmax), True
    return None, float('inf'), False

def _check_if_one_possible(state: State, d: Decision) -> bool:
    for j in state.job_states:
        for o in j.operation_states:
            if o.remaining_time > 0:
                if j.id == d.job_id and o.id == d.operation_id:
                    return True
                break
    return False

def _check_if_all_possible(instance: Instance, decisions: list[Decision]) -> bool:
    state: State = State(instance, M, L, NB_STATIONS, BIG_STATION, [], automatic_build=True)
    for d in decisions:
        if not _check_if_one_possible(state, d):
            return False
        state = simulate(state, d=d, clone=False) 
    return True

def tabu_search(instance: Instance, init_decisions: list[Decision]) -> State:
    best_decisions: list[Decision]       = [d.clone() for d in init_decisions]
    current_state, current_obj, feasible = _run_ls_and_evaluate(instance, init_decisions)
    best_obj: int = current_obj
    tabu_list: Dict[Tuple[int, int], int] = {} # Tabu Memory: Map (job_id, operation_id) -> iteration_index where it becomes free
    iter_count = 0
    no_improvement_count = 0
    print(f"Start Tabu Search. Initial Obj: {best_obj}")
    while iter_count < MAX_ITERATIONS and no_improvement_count < MAX_NO_IMPROVEMENT:
        iter_count += 1
        possible_moves: List[SmartMove]         = _get_possible_neighbors(current_state.decisions, NEIGHBOR_SAMPLE_SIZE)
        best_neighbor_decisions: List[Decision] = None
        best_neighbor_obj: float                = float('inf')
        best_move_ref: Optional[SmartMove]      = None
        found_valid_move: bool                  = False
        for move in possible_moves:
            is_tabu: bool = False
            if move.op1_move:
                sig1 = (move.job_id, 0) 
                if sig1 in tabu_list and tabu_list[sig1] > iter_count:
                    is_tabu = True
            if move.op2_move:
                sig2 = (move.job_id, 1)
                if sig2 in tabu_list and tabu_list[sig2] > iter_count:
                    is_tabu = True
            if is_tabu:
                continue
            candidate_decisions: list[Decision] = _apply_move(current_state.decisions, move)
            opt_candidate, candidate_obj, feasible = _run_ls_and_evaluate(instance, candidate_decisions)
            if not feasible:
                continue
            if candidate_obj < best_neighbor_obj:
                best_neighbor_decisions = opt_candidate
                best_neighbor_obj       = candidate_obj
                best_move_ref           = move
                found_valid_move        = True
        if found_valid_move:
            current_state.decisions = best_neighbor_decisions
            current_obj             = best_neighbor_obj
            if best_move_ref.op1_move:
                tabu_list[(best_move_ref.job_id, 0)] = iter_count + TABU_TENURE
            if best_move_ref.op2_move:
                tabu_list[(best_move_ref.job_id, 1)] = iter_count + TABU_TENURE
            if current_obj < best_obj:
                best_obj             = current_obj
                best_decisions       = [d.clone() for d in current_state.decisions]
                no_improvement_count = 0
                print(f"Iter {iter_count}: New Best found {best_obj}")
            else:
                no_improvement_count += 1
        else:
            no_improvement_count += 1    
    print(f"Tabu Search Finished. Best Obj: {best_obj}")
    final_state, _, _ = _run_ls_and_evaluate(instance, best_decisions)
    return final_state