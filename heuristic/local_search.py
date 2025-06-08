from models.state import State, Decision
from models.instance import Instance
from conf import *
from simulators.gnn_simulator import simulate

# ##################################
# =*= LOCAL IMPROVEMENT OPERATOR =*=
# ##################################
__author__  = "Hedi Boukamcha; Anas Neumann"
__email__   = "hedi.boukamcha.1@ulaval.ca; anas.neumann@polymtl.ca"
__version__ = "1.0.0"
__license__ = "MIT"

def ls(instance: Instance, decisions: list[Decision]):
    state, obj  = _simulate_one(instance, decisions)
    idx: int    = 0
    while idx < len(decisions):
        d: Decision  = decisions[idx].clone()
        to_test: list[Decision] = decisions[:idx] + d + decisions[idx+1:]
        if d.parallel == True:
            d.parallel = False
            idx += 1
            # TODO while the next where True and process 2 put them as False too (all at once)
            # update idx to the end of while
        elif d.parallel == False and idx < len(decisions) -1:
            d.parallel = True
            idx += 1
            # TODO while the next where process 2 and False put them as True too (test one by one)
            # update idx to the end of while
            pass
        new_state, new_obj = _simulate_one(instance, to_test)
        if new_obj <= obj:
            decisions = to_test
            state = new_state
            obj = new_obj
    return state

def _simulate_one(instance: Instance, decisions: list[Decision]) -> tuple[State, int]:
    state: State = State(instance, M, L, NB_STATIONS, BIG_STATION, [], automatic_build=True)
    for d in decisions:
        state = simulate(state, d=d, clone=False) 
    obj: int = (state.total_delay * (100 - instance.a)) + (state.cmax * instance.a)
    return state, obj