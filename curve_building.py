from models.agent import Loss
import argparse

# #################################
# =*= GNN + e-greedy DQN SOLVER =*=
# #################################
__author__  = "Hedi Boukamcha; Anas Neumann"
__email__   = "hedi.boukamcha.1@ulaval.ca; anas.neumann@polymtl.ca"
__version__ = "2.0.0"
__license__ = "MIT"

# TEST WITH: python curve_building.py --path=.
if __name__ == "__main__":
    # CONFIG
    parser  = argparse.ArgumentParser(description="Curve improvement")
    parser.add_argument("--path", help="path to load the curves", required=True)
    args               = parser.parse_args()
    path: str          = args.path +'/data/training/'   
    interactive: bool  = False
    W: float           = 14.628
    H: float           = 9.936

    # CREATE CURVES
    loss: Loss        = Loss(xlabel="Episode", ylabel="Loss", title="Huber Loss (policy network)", color="blue", show=interactive, width=W, height=H)
    diversity: Loss   = Loss(xlabel="Episode", ylabel="Diversity probability", title="Epsilon threshold", color="green", show=interactive, width=W, height=H)
    s_obj: Loss       = Loss(xlabel="Episode", ylabel="Objective value (cmax + delay)", title="Avg objective value for S instances", color="orange", show=interactive, width=W, height=H)
    m_obj: Loss       = Loss(xlabel="Episode", ylabel="Objective value (cmax + delay)", title="Avg objective value for M instances", color="orange", show=interactive, width=W, height=H)
    l_obj: Loss       = Loss(xlabel="Episode", ylabel="Objective value (cmax + delay)", title="Avg objective value for L instances", color="orange", show=interactive, width=W, height=H)
    xl_obj: Loss      = Loss(xlabel="Episode", ylabel="Objective value (cmax + delay)", title="Avg objective value for XL instances", color="orange", show=interactive, width=W, height=H)

    # LOAD AND UDAPTE   
    diversity.load_and_update(f"{path}basic_epsilon")
    loss.load_and_update(f"{path}basic_loss")
    l_obj.load_and_update(f"{path}basic_l_obj")
    s_obj.load_and_update(f"{path}basic_s_obj")
    m_obj.load_and_update(f"{path}basic_m_obj")
    xl_obj.load_and_update(f"{path}basic_xl_obj") 