import pandas as pd
import os

# ##############################
# =*= FINAL RESULTS ANALYSIS =*=
# ##############################
__author__  = "Hedi Boukamcha; Anas Neumann"
__email__   = "hedi.boukamcha.1@ulaval.ca; anas.neumann@polymtl.ca"
__version__ = "1.0.0"
__license__ = "MIT"




def results_tables(result_type: str, output_file: str):
    path         = "./data/instances/test/"
    results_path = "./data/results/"
    sizes        = ['s', 'm', 'l', 'xl',]
    columns      = ['Size', 'Instance ID', 'Status', 'Obj', 'Delay', 'Cmax', 'Computing_time']
    rows         = []
    status_exists_globally = False
    for size in sizes:
        for inst_id in range(51):
            filename = os.path.join(path, f"{size}/{result_type}_{inst_id}.csv")
            row = {
                'Size':           size,
                'Instance ID':    inst_id,
                'Obj':            None,
                'Delay':          None,
                'Cmax':           None,
                'Computing_time': None
            }
            if os.path.exists(filename):
                df                    = pd.read_csv(filename)
                row_data              = df.iloc[0].to_dict()
                row['Obj']            = row_data.get('obj')
                row['Delay']          = row_data.get('delay')
                row['Cmax']           = row_data.get('cmax')
                row['Computing_time'] = row_data.get('computing_time')
                if 'status' in row_data:
                    row['Status']          = row_data['status']
                    status_exists_globally = True
            else:
                pass
            rows.append(row)
    df_exact         = pd.DataFrame(rows)
    df_exact['Size'] = pd.Categorical(df_exact['Size'], categories=sizes, ordered=True)
    df_exact         = df_exact.sort_values(['Size', 'Instance ID'])
    columns          = ['Size', 'Instance ID']
    if status_exists_globally and 'Status' in df_exact.columns:
        columns.append('Status')
    columns += ['Obj', 'Delay', 'Cmax', 'Computing_time']
    df_exact = df_exact[columns]
    df_exact.to_csv(results_path + output_file, index=False)




def detailed_results_per_method(
    file_per_methode: dict,
    variables: list,
    sizes=('s', 'm', 'l', 'xl'),
    output='./data/results/'
    ):
    """
    Crée un fichier CSV par taille avec une ligne par instance (0-50) et une colonne par méthode.

    Args:
        fichiers_par_methode: dict {nom_methode: chemin_fichier_csv}
        variable: le nom de la colonne à extraire ('Delay', 'Cmax', etc.)
        tailles: les tailles à traiter
        output_dir: dossier de sortie
    """
    os.makedirs(output, exist_ok=True)

    for size in sizes:
        df_result = pd.DataFrame({'Instance ID': list(range(51))}) 

        for method, csv_file in file_per_methode.items():
            if not os.path.exists(csv_file):
                print(f"Non existing file for {method}: {csv_file}")
                continue

            df = pd.read_csv(csv_file)
            df_taille = df[df['Size'] == size]

            for var in variables:
                value = df_taille.set_index('Instance ID')[var]
                df_result[f'{method}_{var}'] = df_result['Instance ID'].map(value)
            
            if method == 'exact' and 'Status' in df.columns and 'Instance ID' in df.columns:
                df = df.drop_duplicates(subset='Instance ID')
                df_result['exact_Status'] = df_result['Instance ID'].map(
                    df.set_index('Instance ID')['Status']
                )

        # Sauvegarde par taille
        df_result.to_csv(os.path.join(output, f"detailed_results_{size}.csv"), index=False)







def construire_tableau_latex_agrégé(
    method: dict,
    variable: str,
    output_path: str,
    sizes=('s', 'm', 'l', 'xl'),
    colonnes_source=('Size', 'Delay', 'Cmax', 'Computing_time')
    ):
    """
    Construit un tableau LaTeX où les lignes sont les méthodes,
    et les colonnes sont les tailles avec min, avg, max pour la variable donnée.
    """
    lignes_tableau = []

    for methode, chemin_csv in method.items():
        df = pd.read_csv(chemin_csv)
        df = df[df['Size'].isin(sizes)]

        ligne = {'Méthode': methode}
        for size in sizes:
            sous_df = df[df['Size'] == size]
            if sous_df.empty:
                ligne[f'{size}_min'] = ''
                ligne[f'{size}_avg'] = ''
                ligne[f'{size}_max'] = ''
            else:
                ligne[f'{size}_min'] = sous_df[variable].min()
                ligne[f'{size}_avg'] = sous_df[variable].mean()
                ligne[f'{size}_max'] = sous_df[variable].max()
        lignes_tableau.append(ligne)

    df_latex = pd.DataFrame(lignes_tableau)

    # Ordre des colonnes
    colonnes = ['Méthode']
    for size in sizes:
        for suffix in ['min', 'avg', 'max']:
            colonnes.append(f'{size}_{suffix}')
    df_latex = df_latex[colonnes]

    # Génération du LaTeX
    latex_code = df_latex.to_latex(index=False, column_format='l' + 'c' * (len(colonnes) - 1), float_format="%.2f")

    with open(output_path, 'w') as f:
        f.write(latex_code)



# python3 result_analysis.py
if __name__ == "__main__":

    variables = ['Delay', 'Cmax', 'Obj', 'Computing_time']

    results_tables(result_type='exact_solution', output_file='exact_solution_results.csv')
    results_tables(result_type='gnn_solution', output_file='gnn_solution_results.csv')
    results_tables(result_type='gnn_solution_improved', output_file='gnn_solution_improved_results.csv')
    results_tables(result_type='heuristic_solution', output_file='heuristic_solution_results.csv')

    result_files = {
        'exact': './data/results/exact_solution_results.csv',
        'gnn': './data/results/gnn_solution_results.csv',
        'heuristic': './data/results/heuristic_solution_results.csv',
        'gnn + ls': './data/results/gnn_solution_improved_results.csv'
    }

    # Exemple pour delay :
    construire_tableau_latex_agrégé(
        method=result_files,
        variable='Delay',
        output_path='./data/results/table_delay_aggregated.tex'
    )

    detailed_results_per_method(
    file_per_methode = result_files,
    variables =variables,
    output='./data/results/'
    )