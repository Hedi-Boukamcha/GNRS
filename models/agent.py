import random, torch
from torch.optim import Adam

from gnn import QNet
from memory import ReplayMemory
from gnn_solver import search_possible_decisions


# ##################
# =*= DQN SAgent =*=
# ##################
__author__ = "Hedi Boukamcha - hedi.boukamcha.1@ulaval.ca, Anas Neumann - anas.neumann@polymtl.ca"
__version__ = "1.0.0"
__license__ = "MIT"

class Agent:
    def __init__(self, node_embed_dim, 
                 gamma=0.99, lr=1e-4,
                 eps_start=1., eps_end=0.05, eps_decay=0.995,
                 target_update=2000, capacity=100_000, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Réseaux ----------------------------------------------------
        self.policy_net = QNet(node_embed_dim=node_embed_dim).to(self.device)
        self.target_net = QNet(node_embed_dim=node_embed_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = Adam(self.policy_net.parameters(), lr=lr)

        # Mémoire & divers ------------------------------------------
        self.memory = ReplayMemory(capacity)
        self.gamma  = gamma
        self.eps, self.eps_min, self.eps_decay = eps_start, eps_end, eps_decay
        self.target_update = target_update
        self.learn_step = 0

    # Epsilon_Greedy -------------------------------------------------
    def select_action(self, graph, actions, alpha):
        # Exploration
        if random.random() < self.eps:
            return random.choice(actions)

        # Exploration
        a_tensor = torch.tensor(actions, device=self.device)
        with torch.no_grad():
            qvals = self.policy_net(graph.to(self.device), a_tensor, torch.tensor([alpha], device=self.device))
            return actions[qvals.argmax().item()]

    # --------------------------------------------------------------
    def optimize(self, batch_size):
        if len(self.memory) < batch_size:
            return

        graphs, actions, rewards, next_graphs, done = self.memory.sample(batch_size)

        q_predicted = self.policy_net(graphs, actions)
        with torch.no_grad():
            q_next = self.target_net(next_graphs, actions).max(1)[0]
            q_target = rewards + self.gamma * q_next * (1 - done)

        loss = torch.nn.functional.smooth_l1_loss(q_predicted, q_target, beta=1.0, reduction="mean") # huber loss function (smooth_l1_loss), beta = 1 seuil default
        self.optimizer.zero_grad(); 
        loss.backward(); 
        self.optimizer.step()

        self.learn_step += 1
        if self.learn_step % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        self.eps = max(self.eps_min, self.eps * self.eps_decay)
