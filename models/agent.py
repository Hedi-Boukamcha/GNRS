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
            self.load(path=path, device=device)
        self.policy_net.to(device=device)
        if train:
            self.target_net: QNet     = QNet()
            self.target_net.to(device=device)
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.policy_net.train()
            self.target_net.eval()
            self.optimizer       = Adam(list(self.policy_net.parameters()), lr=LR)
            self.loss: Loss      = Loss(xlabel="Episode", ylabel="Loss", title="Huber Loss (policy network)", color="blue", show=interactive)
            self.diversity: Loss = Loss(xlabel="Episode", ylabel="Diversity probability", title="Epsilon threshold", color="green", show=interactive)
        else:
            self.policy_net.eval()

    def select_next_decision(self, graph: HeteroData, alpha: Tensor, possible_decisions: list[Decision], decisionsT: Tensor, eps_threshold: float, train: bool) -> int:
        if not train or random.random() > eps_threshold:
            Q_values: Tensor = self.policy_net(Batch.from_data_list([graph]).to(self.device), decisionsT, alpha)
            return torch.argmax(Q_values.view(-1)).item()
        else:
            return random.randint(0, len(possible_decisions)-1)

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

    def optimize_policy(self) -> float:
        """
            Optimize the polict network using the Huber loss between selected action and expected best action (based on approx Q-value)
                y = reward r + discounted factor γ x MAX_Q_VALUES(state s+1) predicted with Q_target
                x = predicted quality of (s, a) using the policy network
                L(x, y) = 1/2 (x-y)^2 for small errors (|x-y| ≤ δ) else δ|x-y| - 1/2 x δ^2
        """
        transitions            = self.memory.sample(BATCH_SIZE)
        batch                  = Transition(*zip(*transitions))
        batch_of_graphs        = Batch.from_data_list(list(batch.graph)).to(self.device)
        possible_actions_list  = list(batch.possible_actions)
        action_indices_local   = torch.as_tensor(batch.action_id, device=self.device, dtype=torch.long)
        alphas_list            = [a.to(self.device) for a in batch.alpha] # each a is 1×1
        rewards                = torch.as_tensor(batch.reward, device=self.device, dtype=torch.float32)
        finals                 = torch.as_tensor(batch.final, device=self.device, dtype=torch.bool)
        lengths_cur            = torch.as_tensor([pa.size(0) for pa in possible_actions_list], device=self.device, dtype=torch.long)       # [B]
        offsets_cur            = torch.cat((torch.zeros(1, device=self.device, dtype=torch.long), torch.cumsum(lengths_cur, dim=0)[:-1]))  # [B]
        actions_cat_cur        = torch.cat(possible_actions_list, dim=0).to(self.device)                       # Σ|A_i| × feat_dim
        alpha_vals_cur         = torch.cat(alphas_list, dim=0).view(-1)                                        # [B]
        alphas_cat_cur         = torch.repeat_interleave(alpha_vals_cur, lengths_cur).unsqueeze(1)             # Σ|A_i| × 1
        q_all_cur              = self.policy_net(batch_of_graphs, actions_cat_cur, alphas_cat_cur).squeeze(1)  # Σ|A_i|
        global_action_indices  = offsets_cur + action_indices_local                                            # [B]
        q_sa_cur               = q_all_cur[global_action_indices]                                              # [B]
        non_final_mask         = ~finals                                                                       # [B]
        if non_final_mask.any():
            nf_graphs                = [batch.next_graph[i]            for i in range(BATCH_SIZE) if non_final_mask[i]]
            nf_actions_list          = [batch.next_possible_actions[i] for i in range(BATCH_SIZE) if non_final_mask[i]]
            nf_alphas_list           = [batch.alpha[i].to(self.device) for i in range(BATCH_SIZE) if non_final_mask[i]]
            lengths_nxt              = torch.as_tensor([pa.size(0) for pa in nf_actions_list], device=self.device, dtype=torch.long)
            actions_cat_nxt          = torch.cat(nf_actions_list, dim=0).to(self.device)
            alpha_vals_nxt           = torch.cat(nf_alphas_list, dim=0).view(-1)
            alphas_cat_nxt           = torch.repeat_interleave(alpha_vals_nxt, lengths_nxt).unsqueeze(1)
            batched_graph_nxt        = Batch.from_data_list(nf_graphs).to(self.device)
            with torch.no_grad():
                q_all_nxt = self.target_net(batched_graph_nxt, actions_cat_nxt, alphas_cat_nxt).squeeze(1)    # Σ|A'_i|
            q_splits     = torch.split(q_all_nxt, lengths_nxt.tolist())                                       # list of tensors
            max_q_nxt    = torch.stack([q.max() for q in q_splits])                                           # [n_non_final]
        else:
            max_q_nxt = torch.zeros(0, device=self.device)
        y = rewards.clone() # start with r_t                                                           
        if non_final_mask.any():
            y[non_final_mask] += GAMMA * max_q_nxt
        loss = F.smooth_l1_loss(q_sa_cur, y)
        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.policy_net.parameters(), MAX_GRAD_NORM)
        self.optimizer.step()
        printed_loss = loss.detach().cpu().item()
        self.loss.update(printed_loss)
        return printed_loss