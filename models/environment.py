from torch import Tensor
from torch_geometric.data import HeteroData

from models.state import Decision, State

# ###########################
# =*= SOLVING ENVIRONMENT =*=
# ###########################
__author__  = "Hedi Boukamcha; Anas Neumann"
__email__   = "hedi.boukamcha.1@ulaval.ca; anas.neumann@polymtl.ca"
__version__ = "2.0.0"
__license__ = "MIT"

class Environment:
    def __init__(self, graph: HeteroData, state: State, possible_decisions: list[Decision]=None, decisionsT: Tensor=None, cmax: int=0, delay: int=0, init_UB_cmax: int=0, init_UB_delay: int=0, n: int=0, total_m2_parallel: int=0, m2: int=0, next_M2_parallel: bool=False, action_time: int=0, ub_cmax: int=0, ub_delay: int=0, m2_parallel: int=0, reward: float=0.0, last_job_in_pos: int=-1, Qvalue: float=0.0):
        self.graph              = graph
        self.decisionsT         = decisionsT
        self.cmax               = cmax
        self.delay              = delay
        self.possible_decisions = possible_decisions
        self.total_jobs         = n
        self.init_UB_cmax       = init_UB_cmax if init_UB_cmax > 0 else state.ub_cmax
        self.init_UB_delay      = init_UB_delay if init_UB_delay > 0 else state.ub_delay
        self.rm_jobs            = n
        self.m2_parallel        = m2_parallel
        self.action_time        = action_time
        self.ub_cmax            = ub_cmax if ub_cmax > 0 else state.ub_cmax
        self.ub_delay           = ub_delay if ub_delay > 0 else state.ub_delay
        self.next_M2_parallel   = next_M2_parallel
        self.m2                 = m2
        self.total_m2_parallel  = total_m2_parallel
        self.state              = state
        self.reward             = reward
        self.last_job_in_pos    = last_job_in_pos
        self.Qvalue             = Qvalue

    def update(self, graph: HeteroData, possible_decisions: list[Decision], decisionsT: Tensor):
        self.graph              = graph
        self.decisionsT         = decisionsT
        self.cmax               = self.state.cmax
        self.delay              = self.state.total_delay
        self.possible_decisions = possible_decisions
        self.rm_jobs            = graph['job'].x.shape[0] if graph['job'].x is not None else 0
        self.m2_parallel        = self.total_m2_parallel / self.m2 if self.m2 > 0 else 0
        self.ub_cmax            = self.state.ub_cmax
        self.ub_delay           = self.state.ub_delay
        self.action_time        = self.state.min_action_time()

    def clone(self):
        return Environment(graph=self.graph, state=self.state.clone(), possible_decisions=self.possible_decisions.copy(), decisionsT=self.decisionsT.clone(), cmax=self.cmax, delay=self.delay, init_UB_cmax=self.init_UB_cmax, init_UB_delay=self.init_UB_delay, n=self.total_jobs, total_m2_parallel=self.total_m2_parallel, m2=self.m2, next_M2_parallel=self.next_M2_parallel, action_time=self.action_time, ub_cmax=self.ub_cmax, ub_delay=self.ub_delay, m2_parallel=self.m2_parallel, reward=self.reward, last_job_in_pos=self.last_job_in_pos, Qvalue=self.Qvalue)
    
class Candidate:
    def __init__(self, parent_idx: int, action_idx: int, Q_value: float, env: 'Environment', cmax: int=0, delay: int=0):
        self.parent_idx      = parent_idx
        self.action_idx      = action_idx
        self.Q_value         = Q_value
        self.cmax            = cmax
        self.delay           = delay
        self.env             = env
        self.combined_score  = 0