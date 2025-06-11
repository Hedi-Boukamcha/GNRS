import argparse

# ####################################
# =*= EXACT MIP SOLVER (IBM Cplex) =*=
# ####################################
__author__  = "Hedi Boukamcha; Anas Neumann"
__email__   = "hedi.boukamcha.1@ulaval.ca; anas.neumann@polymtl.ca"
__version__ = "1.0.0"
__license__ = "MIT"

def solve(path: str, id: str):
    pass

# TEST WITH: python mip_solver.py --type=train --size=s --id=1 path=./
if __name__ == "__main__":
    parser  = argparse.ArgumentParser(description="Exact solver (CP OR-tools version)")
    parser.add_argument("--path", help="path to load the instances", required=True)
    parser.add_argument("--type", help="type of the instance, either train or test", required=True)
    parser.add_argument("--size", help="size of the instance, either s, m, l or xl", required=True)
    parser.add_argument("--id", help="id of the instance to solve", required=True)
    args = parser.parse_args()
    solve(path=args.path+"data/instances/"+args.type+"/"+args.size, id=args.id)