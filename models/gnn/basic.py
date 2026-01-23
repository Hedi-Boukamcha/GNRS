import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import SAGEConv, HeteroConv, Linear, global_mean_pool
from torch_geometric.data import HeteroData

from conf import *

# #######################################################
# =*= ARCHITECTURE OF A BASIC DEEP-Q GRAPH NEURAL NET =*=
# #######################################################
__author__  = "Hedi Boukamcha; Anas Neumann"
__email__   = "hedi.boukamcha.1@ulaval.ca; anas.neumann@polymtl.ca"
__version__ = "2.0.0"
__license__ = "MIT"

class JobEmbedding(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.conv = HeteroConv({
            ('station', 'can_load',    'job'): SAGEConv(hidden_dim, hidden_dim, aggr='mean'),
            ('station', 'loaded',      'job'): SAGEConv(hidden_dim, hidden_dim, aggr='mean'),
            ('machine', 'will_execute','job'): SAGEConv(hidden_dim, hidden_dim, aggr='mean'),
            ('machine', 'execute',     'job'): SAGEConv(hidden_dim, hidden_dim, aggr='mean'),
            ('robot',   'hold',        'job'): SAGEConv(hidden_dim, hidden_dim, aggr='mean'),
        }, aggr='sum')

    def forward(self, nodes, edges):
        out_dict = self.conv(nodes, edges)
        return {'job': F.relu(out_dict['job'])}

class OtherEmbedding(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.conv = HeteroConv({
            ('job', 'could_be_loaded', 'station'): SAGEConv(hidden_dim, hidden_dim, aggr='mean'),
            ('job', 'loaded_in',       'station'): SAGEConv(hidden_dim, hidden_dim, aggr='mean'),
            ('job', 'needs',           'machine'): SAGEConv(hidden_dim, hidden_dim, aggr='mean'),
            ('job', 'executed_by',     'machine'): SAGEConv(hidden_dim, hidden_dim, aggr='mean'),
            ('job', 'hold_by',         'robot'):   SAGEConv(hidden_dim, hidden_dim, aggr='mean'),
        }, aggr='sum')

    def forward(self, x_dict, edge_index_dict):
        out_dict = self.conv(x_dict, edge_index_dict)
        h_s = F.relu(out_dict.get('station', torch.zeros_like(x_dict['station'])))
        h_m = F.relu(out_dict.get('machine', torch.zeros_like(x_dict['machine'])))
        h_r = F.relu(out_dict.get('robot',   torch.zeros_like(x_dict['robot'])))
        return {'station': h_s, 'machine': h_m, 'robot': h_r}

class QNet(nn.Module):
    def __init__(self, job_in: int=JOB_FEATURES, robot_in: int=ROBOT_FEATURES, machine_in: int=MACHINE_FEATURES, station_in: int=STATION_FEATURES, hidden_dim: int = 64):
        super().__init__()
        self.lin_job     = Linear(job_in,     hidden_dim)
        self.lin_station = Linear(station_in, hidden_dim)
        self.lin_machine = Linear(machine_in, hidden_dim)
        self.lin_robot   = Linear(robot_in,   hidden_dim)
        self.job_up_1    = JobEmbedding(hidden_dim)
        self.other_up_1  = OtherEmbedding(hidden_dim)
        self.job_up_2    = JobEmbedding(hidden_dim)
        mlp_input_dim    = hidden_dim + hidden_dim + 2 
        self.Q_mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, 64),
            nn.ReLU(),
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

    def forward(self, data: HeteroData, actions: Tensor): 
        # 1. Embedding stacks (stack size=2)
        nodes          = self._init_node_feats(data)
        edges          = data.edge_index_dict
        
        updated_job    = self.job_up_1(nodes, edges)
        nodes['job']   = updated_job['job']
        updated_others = self.other_up_1(nodes, edges)
        nodes.update(updated_others)
        
        updated_job2    = self.job_up_2(nodes, edges)
        nodes['job']    = updated_job2['job']

        # 2. Graph-level embedding: mean of all jobs
        batch_job       = data['job'].batch
        h_global        = global_mean_pool(nodes['job'], batch_job) # Shape [B, hidden_dim]
        
        # 3. Build per-action tensors and final Q values (all in parrallel)
        job_ids         = actions[:,0].long()
        machine         = actions[:,1].unsqueeze(1).float()
        parallel        = actions[:,2].unsqueeze(1).float()
        emb_jobs        = nodes["job"][job_ids]       # (A, hidden_dim)
        graph_ids       = batch_job[job_ids]
        h_globalA       = h_global[graph_ids]         # (A, hidden_dim)
        action_feat     = torch.cat([emb_jobs, h_globalA, machine, parallel], dim=1)
        Q_values        = self.Q_mlp(action_feat).squeeze(1)
        return Q_values