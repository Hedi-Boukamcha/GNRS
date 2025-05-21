import argparse
from pathlib import Path
import json

from models.instance import Instance, MathInstance
from exact_solver import solver_per_file
from simulators.cp_simulator import gantt_cp_solution

def load_instance(path: Path) -> Instance:
    with open(path) as f:
        data = json.load(f)
    return Instance.from_json(data)

def main():
    parser = argparse.ArgumentParser(description="ALSTOM Robot")
    parser.add_argument("--instance", type=Path, required=True, help="JSON instance file")
    args = parser.parse_args()

    # Load instance
    instance = load_instance(args.instance)

    # Transform to MathInstance
    math_instance = MathInstance(instance)
    
    # Resolution with exact solver
    solver = solver_per_file(str(args.instance), debug=args.debug)

    # Gantt
    gantt_cp_solution(instance, math_instance, solver, instance_file=str(args.instance))

if __name__ == "__main__":
    main()
