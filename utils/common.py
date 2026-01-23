import torch
from torch import Tensor
import torch.nn.functional as F

# ###################################################
# =*= COMMON FUNCTIONS SHARED ACCROSS THE PROJECT =*=
# ###################################################
__author__  = "Hedi Boukamcha; Anas Neumann"
__email__   = "hedi.boukamcha.1@ulaval.ca; anas.neumann@polymtl.ca"
__version__ = "2.0.0"
__license__ = "MIT"
    
def to_bool(v: str) -> bool:
    return v.lower() in ['true', 't', 'yes', '1']

def top_k_Q_to_probs(Q: Tensor, temperature: float = 0.95, topk: int = 5) -> int:                
    topk = min(topk, len(Q))                                           # robust value     
    vals, idx = torch.topk(Q, k=topk)                                  # largest-Q actions
    vals = torch.nan_to_num(vals, nan=-1e9, posinf=1e9, neginf=-1e9)   # finite
    vals = vals - vals.max()                                           # improves soft‑max stability
    p = torch.softmax(vals / temperature, dim=0)                       # Boltzmann exploration
    choice = idx[torch.multinomial(p, 1)].item()
    return choice