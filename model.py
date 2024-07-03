from ortools.sat.python import cp_model


class MathematicModel:
    def __init__(self):
        # model instance
        self.model = cp_model.CpModel()
        self.variables = {}
        self._create_variables()
        self._add_constraints()
        self._set_objective()
    
    def _create_variables(self):
        # Variables de décision
        self.variables['X'] = self.model.NewIntVar(0, 1, 'x')
        self.variables['Y'] = self.model.NewIntVar(0, 5, 'y')
        self.variables['Z'] = self.model.NewIntVar(0, 7, 'z')

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























