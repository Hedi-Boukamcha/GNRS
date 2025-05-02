import json
import os
import random


def generate_controledSize_instance(
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

        instance.append(job)
    return instance

def generate_loadVariants_instance(min_jobs: int = 1, max_jobs: int = 15):

    nomber_jobs = random.randint(min_jobs, max_jobs)
    instance = []

    for _ in range(nomber_jobs):
        variant = random.choice(["light", "heavy", "dense", "mixed"])

        job = {
            "big": random.randint(0, 1),
            "due_date": 0,  # sera ajustée après calcul
            "pos_time": 5,
            "status": random.randint(0, 3),
            "blocked": random.randint(0, 1),
            "operations": []
        }

        # Profil léger → peu d’opérations, durées faibles
        if variant == "light":
            nb_operations = max(2, random.randint(1, 2))
            proc_min, proc_max = 10, 25

        # Profil lourd → 1 à 3 opérations, durées très longues
        elif variant == "heavy":
            nb_operations = max(2, random.randint(1, 3))
            proc_min, proc_max = 50, 70

        # Profil dense → beaucoup d’opérations mais petites
        elif variant == "dense":
            nb_operations = random.randint(6, 10)
            proc_min, proc_max = 5, 15

        # Profil équilibré (mixed)
        elif variant == "mixed":
            nb_operations = random.randint(3, 6)
            proc_min, proc_max = 20, 40

        total_processing = 0
        for _ in range(nb_operations):
            op = {
                "type": random.choice([1, 2]),
                "processing_time": random.randint(proc_min, proc_max)
            }
            total_processing += op["processing_time"]
            job["operations"].append(op)

        # due_date plus ou moins éloignée du total_processing
        job["due_date"] = total_processing + random.randint(20, 80)

        instance.append(job)

    return instance

def download_instances(folder, file, instances):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, file)

    with open(path, "w") as f:
        f.write("instances = [\n")
        for inst in instances:
            f.write(f"    {inst},\n")
        f.write("]\n")

def download_instances_json(folder, file, instances):
    os.makedirs(folder, exist_ok=True)

    for i, instance in enumerate(instances):
        file_name = f"{file}_{i+1}.json"
        path = os.path.join(folder, file_name)
        with open(path, "w") as f:
            json.dump(instance, f, indent=4)



if __name__ == "__main__":
    
    all_controledSize_instances = [
        generate_controledSize_instance(2, 3),
        generate_controledSize_instance(4, 3),
        generate_controledSize_instance(6, 3),
    ]
    download_instances_json("data/instances/controled_sizes", "instance", all_controledSize_instances)


    """all_loadVariants_instances = [
        generate_loadVariants_instance(1, 5),
        generate_loadVariants_instance(1, 10),
        generate_loadVariants_instance(1, 15),
    ]"""
    #download_instances_json("data/instances/load_variants", "instance", all_loadVariants_instances)
