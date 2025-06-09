import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GATConv, HeteroConv, Linear, AttentionalAggregation
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
def GAT(src_dim: int, dst_dim: int, out_dim: int, attention_heads: int=4, dropout: float=0.0):
    return GATConv((src_dim, dst_dim), out_dim // attention_heads, heads=attention_heads, dropout=dropout, concat=True, add_self_loops=False )

# Job embedding with message passing: (station, machine, robot) -> job
class JobEmbedding(nn.Module):
    def __init__(self, dimensions: tuple, out_dim: int, heads: int=ATTENTION_HEADS, dropout: float=DROPOUT):
        super().__init__()
        dj, ds, dm, dr = dimensions
        self.conv      = HeteroConv({
                                ('station', 'can_load',    'job'): GAT(src_dim=ds, dst_dim=dj, out_dim=out_dim, attention_heads=heads, dropout=dropout),
                                ('station', 'loaded',      'job'): GAT(src_dim=ds, dst_dim=dj, out_dim=out_dim, attention_heads=heads, dropout=dropout),
                                ('machine', 'will_execute','job'): GAT(src_dim=dm, dst_dim=dj, out_dim=out_dim, attention_heads=heads, dropout=dropout),
                                ('machine', 'execute',     'job'): GAT(src_dim=dm, dst_dim=dj, out_dim=out_dim, attention_heads=heads, dropout=dropout),
                                ('robot',   'hold',        'job'): GAT(src_dim=dr, dst_dim=dj, out_dim=out_dim, attention_heads=heads, dropout=dropout),
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
                        ('job', 'could_be_loaded', 'station'): GAT(src_dim=dj, dst_dim=ds, out_dim=out_dim, attention_heads=heads, dropout=dropout),
                        ('job', 'loaded_in',       'station'): GAT(src_dim=dj, dst_dim=ds, out_dim=out_dim, attention_heads=heads, dropout=dropout),
                        ('job', 'needs',           'machine'): GAT(src_dim=dj, dst_dim=dm, out_dim=out_dim, attention_heads=heads, dropout=dropout),
                        ('job', 'executed_by',     'machine'): GAT(src_dim=dj, dst_dim=dm, out_dim=out_dim, attention_heads=heads, dropout=dropout),
                        ('job', 'hold_by',         'robot'):   GAT(src_dim=dj, dst_dim=dr, out_dim=out_dim, attention_heads=heads, dropout=dropout),
                    }, aggr='sum')
        self.res_station  = Linear(ds, out_dim, bias=False) if ds!=out_dim else nn.Identity()
        self.res_machine  = Linear(dm, out_dim, bias=False) if dm!=out_dim else nn.Identity()
        self.res_robot    = Linear(dr, out_dim, bias=False) if dr!=out_dim else nn.Identity()
        self.norm_station = nn.LayerNorm(out_dim)
        self.norm_machine = nn.LayerNorm(out_dim)
        self.norm_robot   = nn.LayerNorm(out_dim)

    def forward(self, x_dict, edge_index_dict):
        out_dict = self.conv(x_dict, edge_index_dict)
        h_s_msg = out_dict.get('station', torch.zeros_like(x_dict['station']))
        h_m_msg = out_dict.get('machine', torch.zeros_like(x_dict['machine']))
        h_r_msg = out_dict.get('robot', torch.zeros_like(x_dict['robot']))
        h_s = self.norm_station(F.relu(h_s_msg + self.res_station(x_dict['station'])))
        h_m = self.norm_machine(F.relu(h_m_msg + self.res_machine(x_dict['machine'])))
        h_r = self.norm_robot(F.relu(h_r_msg + self.res_robot(x_dict['robot'])))
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
        self.job_up_2          = JobEmbedding(dimensions=(d_job,d_other,d_other,d_other), out_dim=d_job, heads=heads, dropout=dropout)
        self.other_up_2        = OtherEmbedding(dimensions=(d_job,d_other,d_other,d_other), out_dim=d_other, heads=heads, dropout=dropout)
        graph_vector_size: int = d_other*6 + d_job # shape = 3 stations + 2 machines + robot + mean-jobs = 64
        self.global_lin        = Linear(graph_vector_size, GRAPH_DIM)
        self.job_pooling       = AttentionalAggregation(gate_nn=Linear(d_job, 1), nn = None)
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
        B              = data.num_graphs
        h_station_flat = nodes['station'].reshape(B, 3 * self.d_other)
        h_machine_flat = nodes['machine'].reshape(B, 2 * self.d_other)
        h_robot_flat   = nodes['robot'].reshape(B, 1 * self.d_other)
        h_nodes        = torch.cat([h_station_flat, h_machine_flat, h_robot_flat], dim=1)
        batch_job      = data['job'].batch
        mean_jobs      = self.job_pooling(nodes['job'], batch_job)
        graph_vec      = torch.cat([h_nodes, mean_jobs], dim=1)
        h_global       = F.relu(self.global_lin(graph_vec))

        # 3. Build per-action tensors and final Q values (all in parrallel)
        job_ids        = actions[:,0].long()
        parallel       = actions[:,2].unsqueeze(1).float()
        process        = actions[:,1].unsqueeze(1).float()
        job_ptr        = data['job'].ptr[:-1]
        graph_ids      = batch_job[job_ids]
        global_job_ids = job_ids + job_ptr[graph_ids]
        emb_jobs       = nodes['job'][global_job_ids]
        h_globalA      = h_global[graph_ids]      
        alphaA         = alpha.view(1, 1).expand(actions.size(0), 1)
        action_feat = torch.cat([emb_jobs, h_globalA, process, parallel, alphaA], dim=1)
        Q_values = self.Q_mlp(action_feat).squeeze(1)
        return Q_values