import pickle
import random
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.optim import Adam
from torch.nn.utils        import clip_grad_norm_
from torch_geometric.data import Batch, HeteroData

import matplotlib.pyplot as plt

from models.gnn import QNet
from models.memory import ReplayMemory, Transition
from conf import *
from models.state import Decision
from utils.common import top_k_Q_to_probs

# #################
# =*= DQN Agent =*=
# #################
__author__  = "Hedi Boukamcha; Anas Neumann"
__email__   = "hedi.boukamcha.1@ulaval.ca; anas.neumann@polymtl.ca"
__version__ = "1.0.0"
__license__ = "MIT"

class Loss():
    def __init__(self, xlabel: str, ylabel: str, title: str, color: str, show: bool = True, width=7.04, height=4.80):
        self.show = show
        if self.show:
            plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(width, height))
        self.x_data = []
        self.y_data = []
        self.episode = 0
        self.line, = self.ax.plot(self.x_data, self.y_data, label=title, color=color)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.ax.legend()
        if self.show:
            plt.ioff()
    
    def update(self, loss_value: float):
        self.episode = self.episode + 1
        self.x_data.append(self.episode)
        self.y_data.append(loss_value)
        self.line.set_xdata(self.x_data)
        self.line.set_ydata(self.y_data)
        self.ax.relim()
        self.ax.autoscale_view()
        if self.show:
            plt.pause(0.0001)

    def save(self, filepath: str):
        self.fig.savefig(filepath + ".png")
        with open(filepath + '_x_data.pkl', 'wb') as f:
            pickle.dump(self.x_data, f)
        with open(filepath + '_y_data.pkl', 'wb') as f:
            pickle.dump(self.y_data, f)

class Agent:
    def __init__(self, device: str, interactive: bool, path: str, load: bool=False, train: bool=False):
        self.policy_net: QNet     = QNet()
        self.memory: ReplayMemory = ReplayMemory()
        self.path: str            = path
        self.device: str          = device
        if load:
            self.load(device=device)
        self.policy_net.to(device=device)
        if train:
            self.target_net: QNet     = QNet()
            self.target_net.to(device=device)
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.policy_net.train()
            self.target_net.eval()
            self.optimizer       = Adam(list(self.policy_net.parameters()), lr=LR)
            self.scheduler       = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.5, cooldown=500, patience=LR_PATIENCE, threshold=LR_THRESHOLD)
            self.loss: Loss      = Loss(xlabel="Episode", ylabel="Loss", title="Huber Loss (policy network)", color="blue", show=interactive)
            self.diversity: Loss = Loss(xlabel="Episode", ylabel="Diversity probability", title="Epsilon threshold", color="green", show=interactive)
        else:
            self.policy_net.eval()

    def select_next_decision(self, graph: HeteroData, possible_decisions: list[Decision], decisionsT: Tensor, eps_threshold: float, train: bool, greedy: bool) -> int:
        if train:
            if random.random() > eps_threshold:
                Q_values: Tensor = self.policy_net(Batch.from_data_list([graph]).to(self.device), decisionsT)
                return torch.argmax(Q_values.view(-1)).item()
            else:
                return random.randint(0, len(possible_decisions)-1) 
        else:
            with torch.no_grad():
                Q_values: Tensor = self.policy_net(Batch.from_data_list([graph]).to(self.device), decisionsT)
                # print("Nb possible decisions:", len(possible_decisions))
                # print("Q  mean:", Q_values.mean().item())
                # print("Q  max:", Q_values.max().item())
                # print("Q  min:", Q_values.min().item())
                # print("Q  std :", Q_values.std().item())
                return torch.argmax(Q_values.view(-1)).item() if greedy else top_k_Q_to_probs(Q=Q_values.view(-1), topk=min(5, len(possible_decisions)), temperature=0.5)

    def save(self):
        print(f"Saving policy_net and current loss...")
        torch.save(self.policy_net.state_dict(), f"{self.path}policy_net.pth")
        self.diversity.save(f"{self.path}epsilon")
        self.loss.save(f"{self.path}loss")

    def load(self, device: str):
        print(f"Loading policy_net...")
        self.policy_net.load_state_dict(torch.load(f"{self.path}policy_net.pth", map_location=torch.device(device), weights_only=True))

    def optimize_target(self):
        for target_p, policy_p in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_p.data.mul_(1.0 - TAU).add_(policy_p.data, alpha=TAU)

    def _shift_actions(self, action_tensors: list[Tensor], graphs_batch: Batch) -> Tensor: # action_tensors is a list[(nᵢ,3)]
        """
            Convert job-local ids -> global row ids, list order == graph order.
            Returns (Σ|Aᵢ|, 3) tensor [job_global , machine , parallel].
        """
        ptr     = graphs_batch['job'].ptr                # [B+1] Get the number of jobs in each state
        decisions_with_batch_ids = []                    # List to save the results                                
        for batch_idx, pa in enumerate(action_tensors):  # keep list order
            offset     = ptr[batch_idx]                  # scalar, get the offset in indices provoked by the previous graphs (and their jobs)
            local_id   = pa[:, 0].long()                 # Get the local id before offset
            global_id  = local_id + offset               # Build the global (batch-level) ids for every row needs shift
            pa_adj     = torch.stack([global_id.to(pa.dtype), # the final tensor would have the global id,
                                    pa[:,1],                  # the machine,
                                    pa[:,2],                  # the parallel option
                                    pa[:,3], pa[:,4], pa[:,5], pa[:,6], pa[:,7], pa[:,8], pa[:,9], pa[:,10]], dim=1) # and the problem size features!
            decisions_with_batch_ids.append(pa_adj)      # Add the new "possible decision"
        out = torch.cat(decisions_with_batch_ids, dim=0) # Translate into tensor (Σ|Aᵢ|, 3) 
        return out

    def optimize_policy(self) -> float:
        """
            Update the policy network with a mini-batch from replay memory.
        """
        # ---------- 0. sample & unpack ----------------------------------------------------------------------
        transitions   = self.memory.sample(BATCH_SIZE)                                     # Sample transitions
        batch         = Transition(*zip(*transitions))                                     # Regroup by attributes (rewards together, graph together, etc.)
        graphs_cur    = Batch.from_data_list(list(batch.graph)).to(self.device)            # Build one big graph for the whole batch of current state (nodes idx, like jobs, changed!!)
        pa_cur_list   = list(batch.possible_actions)                                       # All possible actions in the batch (hard at this point to distinguish by graph!!)
        act_idx_local = torch.as_tensor(batch.action_id, device=self.device)               # Local ID of the action (one big tensor for the whole batch -> for now with a problem of wrong indices!!)
        rewards       = torch.as_tensor(batch.reward, device=self.device)                  # reward of each currentstate (one big tensor for the whole batch)
        finals        = torch.as_tensor(batch.final, device=self.device, dtype=torch.bool) # boolean check if the graph is final (one big tensor for the whole batch)

        # ---------- 1. build current action tensor (with batch-level indices) ------------------------------
        actions_cur   = self._shift_actions(pa_cur_list, graphs_cur)                       # Create a new tensor with action but this time with global (batch-level) indices to replace pa_cur_list
        lengths_cur   = torch.as_tensor([pa.size(0) for pa in pa_cur_list],    
                                        device=self.device, dtype=torch.long)              # Get the number of possible actions for each current state
        offsets_cur   = torch.cat((torch.zeros(1, device=self.device, dtype=torch.long), 
                                        torch.cumsum(lengths_cur, dim=0)[:-1]))            # The the offset of possible actions by graph (increase with the number of possible actions from the previous graph)
        
        # ---------- 2. Q(s,·) ------------------------------------------------------------------------------
        q_all_cur = self.policy_net(graphs_cur, actions_cur)                               # Use the policy_net to get the Q-value of all possible actions of all state [used once for the whole batch and all possible actions]
        q_sa_cur  = q_all_cur[offsets_cur + act_idx_local]                                 # Get the Q-value of the action that was taken: use the local (act_idx_local) + the offsets_cur to retrieve its batch-level index

        # ---------- 3. next-state branch -------------------------------------------------------------------
        non_final = ~finals
        if non_final.any():                                                                             # If some transitions are not final
            nf_graphs   = [batch.next_graph[i]            for i in range(BATCH_SIZE) if non_final[i]]   # Get the list of non-final graph 
            nf_pa_list  = [batch.next_possible_actions[i] for i in range(BATCH_SIZE) if non_final[i]]   # Get the list of non-final graph' possible actions (with wrong indices: local only for the graph!!)
            graphs_nxt  = Batch.from_data_list(nf_graphs).to(self.device)                               # Like before, build one big graph for the whole batch of next state (nodes idx, like jobs, changed!!)
            actions_nxt = self._shift_actions(nf_pa_list, graphs_nxt)                                   # Replace the list of next possible actions nf_pa_list) with correct batch-level indices!
            len_nxt     = torch.as_tensor([p.size(0) for p in nf_pa_list],                   
                                        device=self.device, dtype=torch.long)                           # Get the number of possible actions for each next state
            with torch.no_grad():
                q_all_nxt = self.target_net(graphs_nxt, actions_nxt)                                    # Use the target_net to get the Q-value of all possible actions of all next state [used once for the whole batch and all possible actions]
            q_split  = torch.split(q_all_nxt, len_nxt.tolist())                                         # Split the Q-value by next state (instead of batch) so we can compute the MAX
            max_q_n  = torch.stack([q.max() for q in q_split])                                          # Get the max Q-value of the next state
        else:
            max_q_n = torch.zeros(0, device=self.device)                                                # In case of final state, the y would be only the reward (+0)

        # ---------- 4. compute (and display) the huber loss & optimize ------------------------------------------
        y = rewards.clone()                                            # y = reward r (start with that)
        y[non_final] += GAMMA * max_q_n                                # y = reward r + discounted factor γ x MAX_Q_VALUES(state s+1) predicted with Q_target [or 0 if final]
        loss = F.smooth_l1_loss(q_sa_cur, y, beta=BETA)                # L(x, y) = 1/2 (x-y)^2 for small errors (|x-y| ≤ δ) else δ|x-y| - 1/2 x δ^2 | here x (q_sa_cur) = predicted quality of (s, a) using the policy network
        self.optimizer.zero_grad()                                     # reset gradients ∇ℓ = 0
        loss.backward()                                                # Build gradients ∇ℓ(f(θi, x), y) with backprop
        clip_grad_norm_(self.policy_net.parameters(), MAX_GRAD_NORM)   # Normalize to avoid exploding gradients
        self.optimizer.step()                                          # Do a gradient step and update parameters -> θi+1 = θi - α∑∇ℓ(f(θi, x), y)
        self.scheduler.step(loss.item())                               # Reduce the learning rate if the loss does not improve
        self.loss.update(loss.item())                                  # Display the loss in the chart!
        return loss.item()
                       