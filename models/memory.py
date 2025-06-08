from collections import deque, namedtuple
import random, torch


# ##################################
# =*= Replay Memmory Tree Format =*=
# ##################################
__author__ = "Hedi Boukamcha - hedi.boukamcha.1@ulaval.ca, Anas Neumann - anas.neumann@polymtl.ca"
__version__ = "1.0.0"
__license__ = "MIT"



Transition = namedtuple('Transition', ('graph', 'action', 'next_graph', 'reward'))

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
