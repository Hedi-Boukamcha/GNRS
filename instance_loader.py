import json
import numpy as np
import pandas as pd

def lire_fichier_json(nom_fichier):
    data = pd.read_json(nom_fichier)
    return data

nom_fichier = '1st_instance.json'
data = lire_fichier_json(nom_fichier)
jobs = list(data.keys())
print(jobs)
nombre_jobs = len(jobs)
print(nombre_jobs)

# job = data['jobs']
# print(job)
# Parcourir chaque job

# Initialiser des compteurs pour les opérations et les types
nombre_jobs = 0
nombre_operations = 0
nombre_types = 0

operations =[nombre_operations]
jobs = [nombre_jobs]
types = [nombre_types]

jobs_set = set()
operations_set = set()
types_set = set()

print("##__parametre 1__##\n")

# Collecter les valeurs uniques de jobs, opérations et types
for job_key, job_data in data.items():
    jobs_set.add(job_key)  # Ajouter le nom du job à l'ensemble des jobs
    for operation in job_data[0]["operations"]:
        nombre_operations += 1
        operations_set.add(operation["type"])  # Ajouter le type de l'opération à l'ensemble des opérations
        types_set.add(operation["type"])  # Ajouter le type de l'opération à l'ensemble des types

# Déterminer les nombres à partir des ensembles
nombre_jobs = len(jobs_set)
nombre_types = len(types_set)

print(f"nombre_jobs: {nombre_jobs}")
print(f"nombre_operations: {nombre_operations}")
print(f"nombre_types: {nombre_types}")

a = [[[0 for piece in range(nombre_operations)] for job in range(nombre_jobs)] for type_ in range(nombre_types)]
print(a)



print("##__parametre 2__##\n")

lp = [0] * nombre_jobs

for job_index, (job, operations) in enumerate(data.items()):
    lp[job_index] = operations[0]['big']

print(lp)


#return jobs #, np.array(due_dates), np.array(pos_times), np.array(bigs), operations



'''if __name__ == "__main__":
    nom_fichier = '1st_instance.json'
    data = lire_fichier_json(nom_fichier)
    a = extraire_caracteristiques(data)
    # jobs, due_dates, pos_times, bigs, operations = extraire_caracteristiques(data)'''

    