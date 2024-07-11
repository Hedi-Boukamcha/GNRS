import json
import numpy as np
import pandas as pd

def lire_fichier_json(nom_fichier):
    data = pd.read_json(nom_fichier)
    return data

nom_fichier = '1st_instance.json'
data = lire_fichier_json(nom_fichier)
jobs = list(data.keys())
#jobs = np.array(data["jobs"])
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
types_operations = set()

operations =[nombre_operations]
jobs = [nombre_jobs]
types = [nombre_types]


for job_key in data.keys():
    if job_key.startswith('job_'):
        job_data = data[job_key][0]  # Accéder au premier élément de la liste job_x
        nombre_jobs += 1
        print(f"\nInformations pour {job_key} :")
        print(f"big: {job_data['big']}")
        print(f"due_date: {job_data['due_date']}")
        print(f"pos_time: {job_data['pos_time']}")
        
        # Accéder aux opérations pour chaque job
        if 'operations' in job_data:
            operations = job_data['operations']
            nombre_operations += len(job_data['operations'])
            print(nombre_operations)
            #for operation in job_data['operations']:
                #nombre_types += len(operation['type'])
            print(f"\nOpérations pour {job_key} :")
            for operation in operations:
                print(f"Type: {operation['type']}")
                print(f"Processing Time: {operation['pocessing_time']}")
                print()
        else:
            print(f"Aucune opération trouvée pour {job_key}.")

for job_key in data.keys():
        if job_key.startswith('job_'):
            job_data = data[job_key][0]  # Accéder au premier élément de la liste job_x
            nombre_jobs += 1
            
            # Compter le nombre d'opérations pour chaque job
            if 'operations' in job_data:
                nombre_operations += len(job_data['operations'])
                
                # Parcourir les opérations pour compter le nombre de types
                for index, operation in enumerate(job_data['operations'], start=1):
                    if 'type' in operation:
                        types_operations.add(operation['type'])  # Ajouter le type à l'ensemble
    
  
nombre_types = len(types_operations)    
# Afficher les résultats
print(f"Nombre total de jobs : {nombre_jobs}")
print(f"Nombre total d'opérations : {nombre_operations}")
print(f"Nombre total de types : {nombre_types}")

a= [[[ 0 for op in range(nombre_operations)] for j in range(nombre_jobs)] for pro in range(nombre_types)]
print(a)
    
'''    # Extraire les dates de livraison, les temps de positionnement et les tailles
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
    max_operations = max(len(job[0]['operations']) for job in data.values())
    
    # Calculer le nombre de pièces (jobs)
    num_pieces = len(data)

    procedures = set()
    for job in data.values():  
        for operation in job[0][len(operations)]:
            procedures.add(operation['type'])

    num_procedures = len(procedures)

# Initialiser le tableau 3D avec des zéros
    a = np.zeros((num_pieces, max_operations, num_procedures), dtype=int)

# Remplir le tableau avec les valeurs binaires selon la présence des opérations
    for piece_index, (job_key, job_data) in enumerate(data.items()):
        for op_index, operation in enumerate(job_data[0][len(operations)]):
            procedure_type = operation['type']
            # Trouver l'index du procédé dans la liste des procédés uniques
            procedure_index = list(procedures).index(procedure_type)
            a[piece_index, op_index, procedure_index] = 1

# Afficher le tableau
    print("Tableau 3D initialisé à zéro puis mis à jour :")
    print(a)
    # Remplir le tableau `firstop` avec les indices de la première opération dans chaque job
    
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





'''
#return jobs #, np.array(due_dates), np.array(pos_times), np.array(bigs), operations



'''if __name__ == "__main__":
    nom_fichier = '1st_instance.json'
    data = lire_fichier_json(nom_fichier)
    a = extraire_caracteristiques(data)
    # jobs, due_dates, pos_times, bigs, operations = extraire_caracteristiques(data)'''

    