import json
import os
import random
import argparse
from conf import INSTANCES_SIZES

def generate_controledSize_instance(
        nombre_jobs: int = 5,
        max_operations_par_job: int = 20,
        types_operations: list = [1, 2],
        duree_min: int = 10,
        duree_max: int = 60,
        due_date_min: int = 50,
        due_date_max: int = 250
    ):
    jobs = []
    for _ in range(nombre_jobs):
        job = {
            "big": random.randint(0, 1),
            "due_date": random.randint(due_date_min, due_date_max),
            "pos_time": 5,
            "status": random.randint(0, 3),
            "blocked": random.randint(0, 2),
            "operations": []
        }
        nb_operations = random.randint(1, max_operations_par_job)
        last_type = None 
        for _ in range(nb_operations):
            # Exclure le type précédent
            available_types = [t for t in types_operations if t != last_type]
            chosen_type = random.choice(available_types)
            op = {
                "type": chosen_type,
                "processing_time": random.randint(duree_min, duree_max)
            }
            job["operations"].append(op)
            last_type = chosen_type
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
    