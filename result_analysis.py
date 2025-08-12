import pandas as pd
import numpy as np
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

    math_cols  = ['exact_Status', 'exact_Computing_time', 'exact_Cmax', 'exact_Delay', 'exact_Obj', 'exact_Gap']
    ls_cols    = ['heuristic_Computing_time', 'heuristic_dev_Cmax', 'heuristic_dev_Delay', 'heuristic_dev_Obj']
    gnn_cols   = ['gnn_Computing_time', 'gnn_dev_Cmax', 'gnn_dev_Delay', 'gnn_dev_Obj']
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

    # --- Format Status ---
    if 'exact_Status' in df.columns:
        df['exact_Status'] = df['exact_Status'].apply(lambda x: 'F' if str(x).lower().startswith('f') else 'O')

    # --- Format Obj, Cmax, Delay (Exact) ---
    def format_time(value):
        try:
            v = float(value)
            if v >= 3600:
                return f"{int(v // 3600)}h"
            elif v >= 60:
                return f"{int(v // 60)}m"
            else:
                return f"{int(v)}s"
        except:
            return value

    for col in ['exact_Computing_time']:
        if col in df.columns:
            df[col] = df[col].apply(format_time)

    # --- Format Gaps & Deviations ---
    def format_percent(value):
        try:
            return f"{float(value)*100:.2f}\\%"
        except:
            return value

    deviation_cols = [
        'exact_Gap',
        'heuristic_dev_Cmax', 'heuristic_dev_Delay', 'heuristic_dev_Obj',
        'gnn_dev_Cmax', 'gnn_dev_Delay', 'gnn_dev_Obj',
        'gnn + ls_dev_Cmax', 'gnn + ls_dev_Delay', 'gnn + ls_dev_Obj'
    ]
    for col in deviation_cols:
        if col in df.columns:
            df[col] = df[col].apply(format_percent)

    # --- Format entiere ---
    def format_int(value):
        try:
            return str(int(float(value)))
        except:
            return value
    
    int_cols = ['exact_Obj', 'exact_Cmax', 'exact_Delay']
    for col in int_cols:
        if col in df.columns:
            df[col] = df[col].apply(format_int)

    # --- Format  ---
    def format(value):
        try:
            return f"{float(value):.2f}s"
        except:
            return value
        
    format_cols = ['gnn_Computing_time', 'gnn + ls_Computing_time', 'heuristic_Computing_time']
    for col in format_cols:
        if col in df.columns:
            df[col] = df[col].apply(format)

    # En-têtes pour LaTeX
    col_labels = (
        ['Instance ID'] +
        ['Status', 'Obj', 'CT', 'Cmax', 'Delay', 'Gap'][:len(math_cols)] +
        ['CT', 'Dev_Cmax', 'Dev_Delay', 'Dev_Obj'][:len(ls_cols)] +
        ['CT', 'Dev_Cmax', 'Dev_Delay', 'Dev_Obj'][:len(gnn_cols)] +
        ['CT', 'Dev_Cmax', 'Dev_Delay', 'Dev_Obj'][:len(gnnls_cols)]
    )

    df.columns = col_labels

    # Export to LaTeX
    latex_code = df.to_latex(index=False, float_format=float_format, escape=False)

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
    columns += ['Computing_time', 'Obj', 'Cmax', 'Delay']
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
                    deviation = (diff - base) / (base + 1e-8)
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
    output_path: str,
    sizes = ('s', 'm', 'l', 'xl'),
    variables=('Delay', 'Cmax', 'Computing_time', 'Gap'),
    ):

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
                ligne[f'{size}_min'] = sous_df[variables].min()
                ligne[f'{size}_avg'] = sous_df[variables].mean()
                ligne[f'{size}_max'] = sous_df[variables].max()
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



def _safe_num(s): return pd.to_numeric(s, errors="coerce")
def _q(x, p): return np.nanpercentile(x, p) if len(x) else np.nan

def fmt_int(x):
    try: return str(int(round(float(x))))
    except: return ""

def fmt_hours(x):
    try: return f"{(float(x)/3600):.2f}h"
    except: return ""

def six_stats(series):
    if series is None or len(series)==0:
        return [np.nan]*6
    s = _safe_num(series).dropna()
    if s.empty:
        return [np.nan]*6
    return [s.min(), _q(s,25), _q(s,50), s.mean(), _q(s,75), s.max()]

# ---------- coeur ----------
def generate_latex_from_size_files(size_files: dict, output_tex: str):
    """
    size_files: dict comme {"S":"s.csv", "M":"m.csv", "L":"l.csv", "XL":"xl.csv"}
    Produit le tableau LaTeX au format demandé.
    """
    # Charger tous les CSV par taille (insensible à la casse des clés)
    sizes_order = ["S","M","L","XL"]
    dfs_by_size = {}
    for k in sizes_order:
        path = size_files.get(k) or size_files.get(k.lower())
        if path:
            dfs_by_size[k] = pd.read_csv(path)
        else:
            dfs_by_size[k] = None  # pas de fichier pour cette taille

    # Méthodes + libellés
    methods = ["exact", "gnn", "heuristic", "gnn + ls"]
    method_labels = ["Math model", "GNN", "LS", "Improved GNN"]  # seulement pour l'entête

    # Colonnes attendues par méthode
    delay_cols = {
        "exact":"exact_Delay", "gnn":"gnn_Delay",
        "heuristic":"heuristic_Delay", "gnn + ls":"gnn + ls_Delay"
    }
    cmax_cols = {
        "exact":"exact_Cmax", "gnn":"gnn_Cmax",
        "heuristic":"heuristic_Cmax", "gnn + ls":"gnn + ls_Cmax"
    }
    obj_cols = {
        "exact":"exact_Obj", "gnn":"gnn_Obj",
        "heuristic":"heuristic_Obj", "gnn + ls":"gnn + ls_Obj"
    }
    ct_cols = {
        "exact":"exact_Computing_time", "gnn":"gnn_Computing_time",
        "heuristic":"heuristic_Computing_time", "gnn + ls":"gnn + ls_Computing_time"
    }

    def block_rows(colmap, formatter, label_prefix):
        """
        Retourne 6 lignes LaTeX (Min/Q1/MEDIAN/Avg/Q3/Max) pour un bloc (Delay/Cmax/Obj/CT).
        Ordre des 16 cellules : 4 méthodes x 4 tailles.
        """
        # Construire les 16 séries (methode, taille)
        grid = []
        for m in methods:
            for sz in sizes_order:
                df = dfs_by_size[sz]
                col = colmap[m]
                series = None
                if df is not None and col in df.columns:
                    series = df[col]
                grid.append(series)

        # Calculer stats (6) pour chaque cellule (16) puis pivoter en 6 lignes
        stats_per_cell = [six_stats(s) for s in grid]  # liste de 16 x [6]
        # pivoter: pour chaque rang 0..5, prendre toutes les cellules et formatter
        labels = [
            f"Min {label_prefix}",
            "Q1",
            "MEDIAN",
            f"Avg {label_prefix}",
            "Q3",
            f"Max {label_prefix}",
        ]
        lines = []
        for stat_idx, lbl in enumerate(labels):
            vals = []
            for cell in stats_per_cell:
                v = cell[stat_idx]
                vals.append("" if pd.isna(v) else formatter(v))
            line = "            \\textbf{" + lbl + "} & " + " & ".join(vals) + r" \\"
            lines.append(line)
        return lines

    # Header (méthodes/tailles)
    class_line = (
        "            \\textbf{Class}: & "
        + " & ".join(sizes_order)
        + " & " + " & ".join(sizes_order)
        + " & " + " & ".join(sizes_order)
        + " & " + " & ".join(sizes_order)
        + r" \\"
        "\n            \\hline"
    )

    # Construire les 4 blocs
    delay_block = "\n".join(block_rows(delay_cols, fmt_int, "delay")) + "\n            \\hline"
    cmax_block  = "\n".join(block_rows(cmax_cols,  fmt_int, "Cmax")) + "\n            \\hline"
    obj_block   = "\n".join(block_rows(obj_cols,   fmt_int, "objective")) + "\n            \\hline"
    ct_block    = "\n".join(block_rows(ct_cols,    fmt_hours, "comp. time")) + "\n            \\hline"

    latex = delay_block + cmax_block + obj_block + ct_block
    with open(output_tex, "w", encoding="utf-8") as f:
        f.write(latex)


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

    construire_tableau_latex_agrégé(
        method=result_files,
        variables=variables,
        sizes=('s', 'm', 'l', 'xl'),
        output_path='./data/results/aggregated_results.tex'
    )

    detailed_results_per_method(
    file_per_methode = result_files,
    variables =variables,
    output='./data/results/'
    )
    
    size_files = {
        "S": "./data/results/detailed_results_s.csv",
        "M": "./data/results/detailed_results_m.csv",
        "L": "./data/results/detailed_results_l.csv",
        "XL":"./data/results/detailed_results_xl.csv",
    }

    generate_latex_from_size_files(size_files, output_tex="./data/results/table_results.tex")
