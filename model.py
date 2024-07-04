from ortools.sat.python import cp_model
import numpy as np


class MathematicModel:
    def __init__(self):
        # model instance
        self.model = cp_model.CpModel()
        self.variables = {}
        self._define_sets()
        self._create_variables()
        self._add_constraints()
        self._set_objective()

    def _define_sets(self):
        # Définir les ensembles
        self.P = np.arange(self.n)  # Pièces à traiter
        self.O = np.array([['op1', 'op2', 'op3'] for _ in range(self.n)])  # Opérations pour chaque pièce
        self.Pro = np.array(['pro1', 'pro2'])  # Procédés
        self.M = np.array(['mA', 'mB', 'mC'])  # Modes d'exécution
        self.S = np.array(['s1', 's2', 's3'])  # Stations de chargement

    def _create_variables(self):
        # Variables de décision
        self.variables['X'] = self.model.NewIntVar(0, 1, 'x')
        self.variables['Y'] = self.model.NewIntVar(0, 5, 'y')
        self.variables['Z'] = self.model.NewIntVar(0, 7, 'z')

    def _define_parameters(self):
        
        # Définir le paramètre proc_needed
        # 1 si l’opération o ∈ O nécessite le procédé pro ∈ Pro , 0 sinon
        self.parameters['proc_needed'] = {}
        for op in self.O:
            for pro in self.Pro:
                self.parameters['proc_needed'][(op, pro)] = 1 if self.operation_needs_procedure(op, pro) else 0
        

        # Définir le paramètre is_large
        # 1 si la pièce p ∈ P est large, 0 sinon
        self.parameters['is_large'] = {}
        # # Initialiser toutes les pieces a 0, 'ne sont pas large'
        for p in self.P:
            self.parameters['is_large'][p] = 0
        

        # Définir la premiere operation de la piece
        self.parameters['first_operation'] = {}
        for p in self.P:
            # Définir la première opération comme étant 'op0' pour toutes les pièces p
            self.parameters['first_operation'][p] = 'op0'
        

        # Définir l'operation qui precede l'operation o
        self.parameters['preceding_operation'] = {}
        for p in self.P:
            for op in self.O:
                # Exclure op0 comme opération précédente si op est la première opération
                if op != self.parameters['first_operation'][p]:
                    self.parameters['preceding_operation'][(p, op)] = self.parameters['first_operation'][p]

        
        # Définir la date due de la piece p
        self.parameters['date_due'] = {}


        # Définir la duree de soudure de l'operation o
        self.parameters['welding_duration'] = {}


        # Définir la duree de positionnement de la piece p en mode B
        self.parameters['posB'] = {}


        # Définir la duree de chargement / dechargement
        self.parameters['L'] = {}


        # Définir la duree d'un mouvement de RM (Robot Manipulateur)
        self.parameters['M'] = {}

        # Définir la borne superieur I
        self.parameters['upper_bound_I'] = 0
        for p in self.P:
            for o in self.O:
                welding_dur = self.parameters['welding_duration'][o] if o in self.parameters['welding_duration'] else 0
                setup_dur = self.parameters['setup_duration'][p] if p in self.parameters['setup_duration'] else 0
                M = self.parameters['M'][o] if o in self.parameters['M'] else 0
                L = self.parameters['L'][p] if p in self.parameters['L'] else 0
                self.parameters['upper_bound_I'] += welding_dur + setup_dur + 3 * M + 2 * L


    def _add_constraints(self):
        # Contraintes
        self.model.Add(self.variables['X'] + self.variables['Y'] + self.variables['Z'] <= 20)
        self.model.Add(self.variables['X'] - 2 * self.variables['Y'] + 3 * self.variables['Z'] >= 5)
        self.model.Add(self.variables['X'] + self.variables['Y'] >= self.variables['Z'])

    def _set_objective(self):
        # Fonction objectif
        self.model.Minimize(self.variables['X'] + 2 * self.variables['Y'] + 3 * self.variables['Z'])
    
    def solve(self):
        # Résoudre le modèle
        solver = cp_model.CpSolver()
        status = solver.Solve(self.model)
        if status == cp_model.OPTIMAL:
            print('Solution optimale trouvée')
            for var_name, var in self.variables.items():
                print(f'Valeur de {var_name}:', solver.Value(var))
            print('Valeur de la fonction objectif:', solver.ObjectiveValue())
        else:
            print('La solution optimale n\'a pas été trouvée')

# Utiliser la classe pour résoudre le modèle
optimization_model = MathematicModel()
optimization_model.solve()























