import json
import numpy as np
import pandas as pd

def lire_fichier_json(nom_fichier):
    data = pd.read_json(nom_fichier)
    return data

def extraire_caracteristiques(data):
    jobs = list(data.keys())
    nombre_jobs = len(jobs)
    
    # Extraire les dates de livraison, les temps de positionnement et les tailles
    due_dates = []
    pos_times = []
    bigs = []
    operations = []

    for job in jobs:
        job_data = data[job][0]
        due_dates.append(job_data["due_date"])
        pos_times.append(job_data["pos_time"])
        bigs.append(job_data["big"])

    
    due_dates = np.array(due_dates)
    pos_times = np.array(pos_times)
    bigs = np.array(bigs)



    print("Jobs:", jobs)
    print("Due dates:", due_dates)
    print("Positioning times:", pos_times)
    print("Operations:", operations)
    
    print("##__parametre 1__##\n")
    inst_1 = np.array([[1, 0], [1, 1], [1, 0]]) # lignes: pieces, colonnes: operations
    print("pro_needed:\n", inst_1)
    
    print("##__parametre 2__##\n")
    
    print("pro_needed:\n", inst_1)
    print("##__parametre 3__##\n")
    # Initialiser le tableau `firstop`
    first_op = np.full(len(bigs), -1, dtype=int)


    # Remplir le tableau `firstop` avec les indices de la première opération dans chaque job
    for job_index, job_data in enumerate(data.keys()):
        first_op[job_index] = 1  # On suppose que le premier index est 0, à ajuster selon la structure réelle des données
    print("first operation :")
    print(first_op)  
    print("##__parametre 4__##\n")

    print("##__parametre 5__##\n")

    print("##__parametre 6__##\n")

    print("##__parametre 7__##\n")

    print("##__parametre 8__##\n")

    print("##__parametre 9__##\n")

    print("##__parametre 10__##\n")

    print("##__parametre 11__##\n")

    print("##__parametre 12__##\n")

    print("##__parametre 13__##\n")

    print("##__parametre 14__##\n")






    return jobs, np.array(due_dates), np.array(pos_times), np.array(bigs), operations



if __name__ == "__main__":
    nom_fichier = '1st_instance.json'
    data = lire_fichier_json(nom_fichier)
    jobs, due_dates, pos_times, bigs, operations = extraire_caracteristiques(data)

    