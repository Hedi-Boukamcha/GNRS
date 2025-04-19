import json
import os
import random

def generate_instance(
    nombre_jobs: int = 5,
    max_operations_par_job: int = 20,
    types_operations: list = [1, 2],
    duree_min: int = 10,
    duree_max: int = 60,
    due_date_min: int = 50,
    due_date_max: int = 250
):
    instance = []

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
        for _ in range(nb_operations):
            op = {
                "type": random.choice(types_operations),
                "processing_time": random.randint(duree_min, duree_max)
            }
            job["operations"].append(op)

        instance.append(job)
    return instance

def download_instances(dossier, fichier, instances):
    os.makedirs(dossier, exist_ok=True)
    chemin = os.path.join(dossier, fichier)

    with open(chemin, "w") as f:
        f.write("instances = [\n")
        for inst in instances:
            f.write(f"    {inst},\n")
        f.write("]\n")


if __name__ == "__main__":
    # Exemple : générer 3 instances de taille 4 jobs / 3 opérations max
    
    all_instances = [
        generate_instance(4, 3),
        generate_instance(5, 2),
        generate_instance(3, 4),
    ]
    
    download_instances("data", "instances.py", all_instances)
