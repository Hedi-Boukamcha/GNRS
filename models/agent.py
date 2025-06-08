import pickle
import random
import torch
from torch import Tensor
from torch.optim import Adam
from torch_geometric.data import HeteroData

import matplotlib.pyplot as plt

from gnn import QNet
from memory import ReplayMemory
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
    def __init__(self, device: str, interactive: bool, path: str, load: bool=False):
        self.policy_net: QNet     = QNet()
        self.target_net: QNet     = QNet()
        self.memory: ReplayMemory = ReplayMemory()
        self.path: str            = path
        if load:
            self.load(path=path, device=device)
        self.policy_net.to(device=device)
        self.target_net.to(device=device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.policy_net.train()
        self.target_net.eval()
        self.policy_net = torch.compile(self.policy_net)
        self.target_net = torch.compile(self.target_net)
        self.optimizer = Adam(list(self.policy_net.parameters()), lr=LR)
        self.loss: Loss = Loss(xlabel="Episode", ylabel="Loss", title="Huber Loss (policy network)", color="blue", show=interactive)
        self.diversity: Loss = Loss(xlabel="Episode", ylabel="Diversity probability", title="Epsilon threshold", color="green", show=interactive)

    def select_next_decision(self, graph: HeteroData, alpha: Tensor, possible_decisions: list[Decision], decisionsT: Tensor, eps_threshold: float, train: bool) -> int:
        if not train or random.random() > eps_threshold:
            Q_values: Tensor = self.policy_net(graph, decisionsT, alpha)
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
        _target_weights = self.target_net.state_dict()
        _policy_weights = self.policy_net.state_dict()
        for param in _policy_weights:
            _target_weights[param] = _policy_weights[param] * TAU + _target_weights[param] * (1 - TAU)
        self.target_net.load_state_dict(_target_weights)

    def optimize_policy(self) -> float:
        # TODO
        pass