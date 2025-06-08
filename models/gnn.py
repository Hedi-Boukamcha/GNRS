import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GATConv, HeteroConv, Linear
from torch_geometric.data import HeteroData

from conf import *

# #######################################################
# =*= ARCHITECTURE OF THE DEEP-Q GRAPH NEURAL NETWORK =*=
# #######################################################
__author__  = "Hedi Boukamcha; Anas Neumann"
__email__   = "hedi.boukamcha.1@ulaval.ca; anas.neumann@polymtl.ca"
__version__ = "1.0.0"
__license__ = "MIT"

# Build a GNN-transformer (GAT) block
def GAT(in_dim: int, out_dim: int, attention_heads: int=4, dropout: float=0.0):
    return GATConv(in_dim, out_dim // attention_heads, heads=attention_heads, dropout=dropout, concat=True)

# Job embedding with message passing: (station, machine, robot) -> job
class JobEmbedding(nn.Module):
    def __init__(self, dimensions: tuple, out_dim: int, heads: int=ATTENTION_HEADS, dropout: float=DROPOUT):
        super().__init__()
        dj, ds, dm, dr = dimensions
        self.conv      = HeteroConv({
                                ('station', 'can_load',    'job'): GAT(ds, out_dim, heads, dropout),
                                ('station', 'loaded',      'job'): GAT(ds, out_dim, heads, dropout),
                                ('machine', 'will_execute','job'): GAT(dm, out_dim, heads, dropout),
                                ('machine', 'execute',     'job'): GAT(dm, out_dim, heads, dropout),
                                ('robot',   'hold',        'job'): GAT(dr, out_dim, heads, dropout),
                            }, aggr='sum')
        self.residual  = Linear(dj, out_dim, bias=False) if dj != out_dim else nn.Identity()
        self.norm      = nn.LayerNorm(out_dim)

    def forward(self, nodes, edges):
        out_dict = self.conv(nodes, edges)
        h = out_dict['job'] + self.residual(nodes['job'])
        return {'job': self.norm(F.relu(h))}

# Other nodes embedding with message passing: job -> (station, machine, robot)
class OtherEmbedding(nn.Module):
    def __init__(self, dimensions: tuple, out_dim: int, heads: int=ATTENTION_HEADS, dropout: float=DROPOUT):
        super().__init__()
        dj, ds, dm, dr = dimensions
        self.conv = HeteroConv({
                        ('job', 'could_be_loaded', 'station'): GAT(dj, out_dim, heads, dropout),
                        ('job', 'loaded_in',       'station'): GAT(dj, out_dim, heads, dropout),
                        ('job', 'needs',           'machine'): GAT(dj, out_dim, heads, dropout),
                        ('job', 'executed_by',     'machine'): GAT(dj, out_dim, heads, dropout),
                        ('job', 'hold_by',         'robot'):   GAT(dj, out_dim, heads, dropout),
                    }, aggr='sum')
        self.res_station  = Linear(ds, out_dim, bias=False) if ds!=out_dim else nn.Identity()
        self.res_machine  = Linear(dm, out_dim, bias=False) if dm!=out_dim else nn.Identity()
        self.res_robot    = Linear(dr, out_dim, bias=False) if dr!=out_dim else nn.Identity()
        self.norm_station = nn.LayerNorm(out_dim)
        self.norm_machine = nn.LayerNorm(out_dim)
        self.norm_robot   = nn.LayerNorm(out_dim)

    def forward(self, x_dict, edge_index_dict):
        out_dict = self.conv(x_dict, edge_index_dict)
        h_s = self.norm_station(F.relu(out_dict['station'] + self.res_station(x_dict['station'])))
        h_m = self.norm_machine(F.relu(out_dict['machine'] + self.res_machine(x_dict['machine'])))
        h_r = self.norm_robot(  F.relu(out_dict['robot']   + self.res_robot(  x_dict['robot']  )))
        return {'station': h_s, 'machine': h_m, 'robot': h_r}

# Main Deep-Q Network: embedding stack + final MLP to output Q value
class QNet(nn.Module):
    def __init__(self, job_in: int=JOB_FEATURES, robot_in: int=ROBOT_FEATURES, machine_in: int=MACHINE_FEATURES, station_in: int=STATION_FEATURES,
                 d_job: int = JOB_DIM, d_other = NODE_DIM, heads = ATTENTION_HEADS, dropout: float=DROPOUT):
        super().__init__()
        self.d_job             = d_job
        self.d_other           = d_other
        self.lin_job           = Linear(job_in, d_job)
        self.lin_station       = Linear(station_in, d_other)
        self.lin_machine       = Linear(machine_in, d_other)
        self.lin_robot         = Linear(robot_in, d_other)
        self.job_up_1          = JobEmbedding(dimensions=(d_job, d_other, d_other, d_other), out_dim=d_job, heads=heads, dropout=dropout)
        self.other_up_1        = OtherEmbedding(dimensions=(d_job, d_other, d_other, d_other), out_dim=d_other, heads=heads, dropout=dropout)
        self.job_up_2          = JobEmbedding((d_job,d_other,d_other,d_other), out_dim=d_job, heads=heads, dropout=dropout)
        self.other_up_2        = OtherEmbedding((d_job,d_other,d_other,d_other), out_dim=d_other, heads=heads, dropout=dropout)
        graph_vector_size: int = d_other*6 + d_job # shape = 3 stations + 2 machines + robot + mean-jobs = 64
        self.global_lin        = Linear(graph_vector_size, GRAPH_DIM)
        self.Q_mlp = nn.Sequential(
            nn.Linear(d_job + GRAPH_DIM + 3, 64),   # +3 for [parallel, alpha, process]
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1))

    def _init_node_feats(self, data: HeteroData):
        x = {}
        x['job']     = F.relu(self.lin_job(data['job'].x))
        x['station'] = F.relu(self.lin_station(data['station'].x))
        x['machine'] = F.relu(self.lin_machine(data['machine'].x))
        x['robot']   = F.relu(self.lin_robot(data['robot'].x))
        return x

    def forward(self, data: HeteroData, actions: Tensor, alpha: Tensor): # action shape = shape (A,3); alpha shape = (1,)
        # 1. Embedding stacks (stack size=2)
        nodes           = self._init_node_feats(data)
        edges           = data.edge_index_dict
        
        updated_job     = self.job_up_1(nodes, edges)
        nodes['job']    = updated_job['job']
        updated_others  = self.other_up_1({**nodes}, edges)
        nodes.update(updated_others)

        updated_job2    = self.job_up_2(nodes, edges)
        nodes['job']    = updated_job2['job']
        updated_others2 = self.other_up_2({**nodes}, edges)
        nodes.update(updated_others2)

        # 2. Graph-level embedding: mean of all jobs + 3 stations + 2 machines + 1 robot
        h_job           = nodes['job']
        mean_jobs       = h_job.mean(dim=0, keepdim=True)
        h_nodes         = torch.cat([nodes['station'][0:3].reshape(-1), nodes['machine'][0:2].reshape(-1), nodes['robot'][0].reshape(-1)], dim=0).unsqueeze(0) # (1,6*d_other)
        h_global        = self.global_lin(torch.cat([h_nodes, mean_jobs], dim=1))
        h_global        = F.relu(h_global).squeeze(0)

        # 3. Build per-action tensors and final Q values (all in parrallel)
        job_ids   = actions[:,0].long()
        parallel  = actions[:,2].unsqueeze(1).float()
        process   = actions[:,1].unsqueeze(1).float()
        A         = actions.size(0)
        emb_jobs  = h_job[job_ids]
        h_globalA = h_global.expand(A, -1)
        alphaA    = alpha.expand(A,1)
        action_feat = torch.cat([emb_jobs, h_globalA, process, parallel, alphaA], dim=1)
        Q_values = self.Q_mlp(action_feat).squeeze(1)
        return Q_values