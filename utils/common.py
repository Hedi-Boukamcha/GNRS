import torch
from torch import Tensor
import torch.nn.functional as F

# ###################################################
# =*= COMMON FUNCTIONS SHARED ACCROSS THE PROJECT =*=
# ###################################################
__author__  = "Hedi Boukamcha; Anas Neumann"
__email__   = "hedi.boukamcha.1@ulaval.ca; anas.neumann@polymtl.ca"
__version__ = "1.0.0"
__license__ = "MIT"

def to_bool(v: str) -> bool:
    return v.lower() in ['true', 't', 'yes', '1']

def tensors_to_probs(q: Tensor, temperature: float = 1.0) -> Tensor:
    q = torch.nan_to_num(q, nan=-1e9, posinf=1e9, neginf=-1e9)  # finite
    probs = F.softmax(q / temperature, dim=0)
    if torch.allclose(probs.sum(), torch.tensor(0.0, device=probs.device)):
        probs = torch.ones_like(probs) / probs.numel()
    return probs