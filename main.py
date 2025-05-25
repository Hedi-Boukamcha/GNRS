import argparse
from pathlib import Path
import json

from models.instance import Instance, MathInstance
from simulators.cp_simulator import *
from simulators.gnn_simulator import *
from utils.common import *
import argparse

def main(instance_file):
    instance: Instance = Instance.load(instance_file)
    decisions1 = [
        Decision(1, 0, True),
        Decision(0, 0, True),
        Decision(1, 1, False),
        Decision(2, 0, False)
    ]
    decisions2 = [
        Decision(0, 0, False),  # Job 1 / op 1 – mode C
        Decision(1, 0, True),  # Job 2 / op 1 – mode B
        Decision(2, 0, True),  # Job 3 / op 1 – mode C
        Decision(2, 1, True),  # Job 3 / op 2 – mode B
        Decision(3, 0, True),  # Job 4 / op 1 – mode C
        Decision(3, 1, False),  # Job 4 / op 2 – mode A
    ]
    decisions3 = [
        Decision(0, 0, True),  # Job 1 / op 1 – mode Bje ve
        Decision(2, 0, True),  # Job 3 / op 1 – mode C
        Decision(1, 0, False),  # Job 2 / op 1 – mode C
    ]
    sim = simulate_all(instance, decisions3)

# python3 main.py data/instances/debug/1st_instance.json
# python3 main.py data/instances/debug/2nd_instance.json
# python3 main.py data/instances/debug/3rd_instance.json
if __name__ == "__main__":
    main("data/instances/debug/3rd_instance.json")
