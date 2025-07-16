import pandas as pd
import os

# ##############################
# =*= FINAL RESULTS ANALYSIS =*=
# ##############################
__author__  = "Hedi Boukamcha; Anas Neumann"
__email__   = "hedi.boukamcha.1@ulaval.ca; anas.neumann@polymtl.ca"
__version__ = "1.0.0"
__license__ = "MIT"



def csv_to_latex_table(input_csv: str, output_tex: str, float_format="%.3f"):
    df = pd.read_csv(input_csv)

    # Colonnes exactes (Math)
    math_cols = ['exact_Status', 'exact_Obj', 'exact_Computing_time', 'exact_Cmax', 'exact_Delay', 'exact_Gap']
    ls_cols   = ['heuristic_Computing_time', 'heuristic_dev_Cmax', 'heuristic_dev_Delay', 'heuristic_dev_Obj']
    gnn_cols  = ['gnn_Computing_time', 'gnn_dev_Cmax', 'gnn_dev_Delay', 'gnn_dev_Obj']
    gnnls_cols = ['gnn + ls_Computing_time', 'gnn + ls_dev_Cmax', 'gnn + ls_dev_Delay', 'gnn + ls_dev_Obj']

    # Vérifie la présence des colonnes pour éviter les erreurs
    columns_present = df.columns.tolist()
    math_cols = [col for col in math_cols if col in columns_present]
    ls_cols   = [col for col in ls_cols if col in columns_present]
    gnn_cols  = [col for col in gnn_cols if col in columns_present]
    gnnls_cols = [col for col in gnnls_cols if col in columns_present]

    # Final column order
    ordered_columns = ['Instance ID'] + math_cols + ls_cols + gnn_cols + gnnls_cols
    df = df[ordered_columns]

    # En-têtes pour LaTeX
    col_labels = (
        ['Instance ID'] +
        ['Status', 'Obj', 'CT', 'Cmax', 'Delay', 'Gap'][:len(math_cols)] +
        ['CT', 'Dev_Cmax', 'Dev_Delay', 'Dev_Obj'][:len(ls_cols)] +
        ['CT', 'Dev_Cmax', 'Dev_Delay', 'Dev_Obj'][:len(gnn_cols)] +
        ['CT', 'Dev_Cmax', 'Dev_Delay', 'Dev_Obj'][:len(gnnls_cols)]
    )

    # Niveau 1 (groupe)
    top_headers = (
        [''] +  # Instance ID
        ['Math'] * len(math_cols) +
        ['LS'] * len(ls_cols) +
        ['GNN'] * len(gnn_cols) +
        ['GNN+LS'] * len(gnnls_cols)
    )

    # Construction MultiIndex
    multi_index = pd.MultiIndex.from_arrays([top_headers, col_labels])
    df.columns = multi_index

    # Export LaTeX
    latex_code = df.to_latex(index=False, multicolumn=True, multicolumn_format='c', float_format="%.3f", escape=False)

    # Sauvegarde
    with open(output_tex, 'w') as f:
        f.write(latex_code)

def results_tables(result_type: str, output_file: str):
    path         = "./data/instances/test/"
    results_path = "./data/results/"
    sizes        = ['s', 'm', 'l', 'xl',]
    columns      = ['Size', 'Instance ID', 'Obj', 'Delay', 'Cmax', 'Computing_time']
    rows         = []
    status_exists_globally = False
    gap_exists_globally = False
    for size in sizes:
        for inst_id in range(51):
            filename = os.path.join(path, f"{size}/{result_type}_{inst_id}.csv")
            row = {
                'Size':           size,
                'Instance ID':    inst_id,
                'Obj':            None,
                'Delay':          None,
                'Cmax':           None,
                'Computing_time': None,
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
                if 'gap' in row_data:
                    row['Gap']          = row_data['gap']
                    gap_exists_globally = True
            else:
                pass
            rows.append(row)
    df_result         = pd.DataFrame(rows)
    df_result['Size'] = pd.Categorical(df_result['Size'], categories=sizes, ordered=True)
    df_result         = df_result.sort_values(['Size', 'Instance ID'])
    columns          = ['Size', 'Instance ID']
    if status_exists_globally and 'Status' in df_result.columns:
        columns.append('Status')
    if gap_exists_globally and 'Gap' in df_result.columns:
        columns.append('Gap')
    columns += ['Obj', 'Delay', 'Cmax', 'Computing_time']
    df_result = df_result[columns]
    df_result.to_csv(results_path + output_file, index=False)

def detailed_results_per_method(file_per_methode: dict, variables: list, sizes=('s', 'm', 'l', 'xl'), output='./data/results/'):

    os.makedirs(output, exist_ok=True)
    for size in sizes:
        df_result = pd.DataFrame({'Instance ID': list(range(51))})
        exact_data = {}

        for method, csv_file in file_per_methode.items():
            if method == 'exact':
                if not os.path.exists(csv_file):
                    print(f"No file named : {csv_file}")
                    continue
                df_exact = pd.read_csv(csv_file)
                df_exact = df_exact[df_exact['Size'] == size].drop_duplicates(subset='Instance ID')
                exact_data = df_exact.set_index('Instance ID')

                for var in variables:
                    if var in exact_data.columns:
                        df_result[f'exact_{var}'] = df_result['Instance ID'].map(exact_data[var])

                if 'Status' in exact_data.columns:
                    df_result['exact_Status'] = df_result['Instance ID'].map(exact_data['Status'])

                if 'Gap' in exact_data.columns:
                    df_result['exact_Gap'] = df_result['Instance ID'].map(exact_data['Gap'])

        for method, csv_file in file_per_methode.items():
            if method == 'exact':
                continue 

            if not os.path.exists(csv_file):
                print(f"Fichier introuvable pour {method}: {csv_file}")
                continue

            df = pd.read_csv(csv_file)
            df = df[df['Size'] == size].drop_duplicates(subset='Instance ID')
            df_method = df.set_index('Instance ID')

            for var in ['Cmax', 'Delay', 'Obj']:
                if var in df_method.columns and f'exact_{var}' in df_result.columns:
                    diff = df_result['Instance ID'].map(df_method[var])
                    base = df_result[f'exact_{var}']
                    deviation = (diff - base) / base
                    df_result[f'{method}_dev_{var}'] = deviation

            if 'Computing_time' in df_method.columns:
                df_result[f'{method}_Computing_time'] = df_result['Instance ID'].map(df_method['Computing_time'])

        output_file = os.path.join(output, f'detailed_results_{size}.csv')
        df_result.to_csv(output_file, index=False)

        for size in sizes:
            input_csv = f'./data/results/detailed_results_{size}.csv'
            output_tex = f'./data/results/detailed_results_{size}.tex'
            csv_to_latex_table(input_csv, output_tex)

def construire_tableau_latex_agrégé(
    method: dict,
    variable: str,
    output_path: str,
    sizes=('s', 'm', 'l', 'xl'),
    colonnes_source=('Size', 'Delay', 'Cmax', 'Computing_time', 'Gap')
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