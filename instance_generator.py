import json
import os
import random
import argparse
from conf import INSTANCES_SIZES, M, L

# ##########################
# =*= INSTANCE GENERATOR =*=
# ##########################
__author__ = "Hedi Boukamcha - hedi.boukamcha.1@ulaval.ca"
__version__ = "1.0.0"
__license__ = "MIT"

# TODO
# 1. station, procede etc. commencent a 0
# 2. genere les due dates logiques avec le nb jobs (certains jobs on des dues dates loin ... nb ops + procesing times)

def generate_controledSize_instance(
        nombre_jobs: int = 5,
        max_operations_par_job: int = 2,
        types_operations: list = [0, 1],
        duree_min: int = 10,
        duree_max: int = 60,
        pos_time: int = 5,
        due_date_min: int = 50,
    ):
    
    jobs = []
    for _ in range(nombre_jobs):
        nb_operations = random.randint(1, max_operations_par_job)
        last_type = None 
        due_date_max = (2 * L + pos_time) * nombre_jobs + nb_operations * (2 * M + 60)
        borne_inf = max(due_date_min, due_date_max // 3)
        due_date = random.randint(borne_inf, due_date_max)
        job = {
            "big": random.randint(0, 1),
            "due_date": due_date,
            "pos_time": pos_time,
            "status": random.randint(0, 3),
            "blocked": random.randint(0, 2),
            "operations": []
        }

        for _ in range(nb_operations):
            available_types = [t for t in types_operations if t != last_type]
            chosen_type = random.choice(available_types)
            proc_time = random.randint(duree_min, duree_max)
            op = {
                "type": chosen_type,
                "processing_time": proc_time
            }
            last_type = chosen_type
            job["operations"].append(op)
        jobs.append(job)
    return {'a': random.randint(0, 10) * 10, 'jobs': jobs}

def save_instances_json(folder, instances):
    os.makedirs(folder, exist_ok=True)
    for i, instance in enumerate(instances):
        file_name = f"instance_{i+1}.json"
        path = os.path.join(folder, file_name)
        with open(path, "w") as f:
            json.dump(instance, f, indent=4)

# TEST WITH: python instance_generator.py --train=150 --test=50 --path=./
if __name__ == "__main__":
    parser  = argparse.ArgumentParser(description="Instance Generator")
    parser.add_argument("--path", help="path to save the instances", required=True)
    parser.add_argument("--train", help="number of training instances", required=True)
    parser.add_argument("--test", help="number of test instances", required=True)
    args = parser.parse_args()
    nb_train: int = int(args.train)
    nb_test: int = int(args.test)
    base_path: str = args.path + "data/instances/"
    for size_name, job_min, job_max in INSTANCES_SIZES:
        train_instances: list = []
        for i in range(nb_train):
            train_instances.append(generate_controledSize_instance(nombre_jobs=random.randint(job_min, job_max), max_operations_par_job=2))
        save_instances_json(base_path + "train/" + size_name + "/", train_instances)
        test_instances: list = []
        for i in range(nb_test):
            test_instances.append(generate_controledSize_instance(nombre_jobs=random.randint(job_min, job_max), max_operations_par_job=2))
        save_instances_json(base_path + "test/" + size_name + "/", test_instances)
