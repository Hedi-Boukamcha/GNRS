import argparse
import pickle
from models.state import State
from gantt.gnn_gantt import gnn_gantt

# #################################
# =*= GNN + e-greedy DQN SOLVER =*=
# #################################
__author__  = "Hedi Boukamcha; Anas Neumann"
__email__   = "hedi.boukamcha.1@ulaval.ca; anas.neumann@polymtl.ca"
__version__ = "1.0.0"
__license__ = "MIT"

# TEST WITH: python gantt_building.py --path=. --type=gnn --size=s --id=13
if __name__ == "__main__":
    parser  = argparse.ArgumentParser(description="Curve improvement")
    parser.add_argument("--path", help="path to load the curves", required=True)
    parser.add_argument("--size", help="instance size", required=True)
    parser.add_argument("--type", help="solution type", required=True)
    parser.add_argument("--id", help="id", required=True)
    args = parser.parse_args()
    saving_path: str = f"{args.path}/data/gantts/temp/{args.type}_{args.size}_{args.id}.png"
    loading_path: str = f"{args.path}/data/instances/test/{args.size}/{args.type}_state_{args.id}.pkl"
    print(f"Loading state from {loading_path}")
    solution: State = pickle.load(loading_path)
    print(f"Saving Gantt chart to {saving_path}")
    gnn_gantt(saving_path, solution, f"instance_{id}")