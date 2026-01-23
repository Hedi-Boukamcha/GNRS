from collections import namedtuple, deque
import random
from conf import *

# ##################################
# =*= Replay Memmory Tree Format =*=
# ##################################
__author__  = "Hedi Boukamcha; Anas Neumann"
__email__   = "hedi.boukamcha.1@ulaval.ca; anas.neumann@polymtl.ca"
__version__ = "2.0.0"
__license__ = "MIT"

Transition = namedtuple('Transition', ('graph', 'action_id',  'possible_actions', 'next_graph', 'next_possible_actions', 'reward', 'final', 'nb_actions', 'weight'))

class ReplayMemory:
    def __init__(self, capacity: int=CAPACITY):
        self.memory = deque([], maxlen=capacity)

    def push(self, t: Transition):
        self.memory.append(t)

    def sample(self, batch_size: int=BATCH_SIZE):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)