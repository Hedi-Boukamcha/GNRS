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

# --- Format  ---
def format_seconds(value):
    try:
        return f"{float(value):.2f}s"
    except:
        return value

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

# --- Format Gaps & Deviations ---
def format_percent(value):
    try:
        return f"{float(value)*100:.2f}\\%"
    except:
        return value

# --- Format entiere ---
def format_int(value):
    try:
        return str(int(float(value)))
    except:
        return value

def _stats_6(series: pd.Series):
    """min, Q1, median, mean, Q3, max sur une série numérique (ignore NaN)."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return [np.nan]*6
    return [
        s.min(),
        s.quantile(0.25),
        s.median(),
        s.mean(),
        s.quantile(0.75),
        s.max(),
    ]

def _place_block(df_final, col_name, var_prefix, values_as_list, formatter=lambda x: x, exact=False):
    rows = [f"min_{var_prefix}", f"Q1_{var_prefix}", f"median_{var_prefix}",
            f"avg_{var_prefix}", f"Q3_{var_prefix}", f"max_{var_prefix}"]
    for row, v in zip(rows, values_as_list):
        if exact:
            df_final.loc[row, col_name] = "" if (v is None or (isinstance(v, float) and np.isnan(v))) else str(v)
        else:
            df_final.loc[row, col_name] = formatter(v)
    

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

    for col in ['exact_Computing_time']:
        if col in df.columns:
            df[col] = df[col].apply(format_time)

    deviation_cols = [
        'exact_Gap',
        'heuristic_dev_Cmax', 'heuristic_dev_Delay', 'heuristic_dev_Obj',
        'gnn_dev_Cmax', 'gnn_dev_Delay', 'gnn_dev_Obj',
        'gnn + ls_dev_Cmax', 'gnn + ls_dev_Delay', 'gnn + ls_dev_Obj'
    ]
    for col in deviation_cols:
        if col in df.columns:
            df[col] = df[col].apply(format_percent)
    
    int_cols = ['exact_Obj', 'exact_Cmax', 'exact_Delay']
    for col in int_cols:
        if col in df.columns:
            df[col] = df[col].apply(format_int)

        
    format_cols = ['gnn_Computing_time', 'gnn + ls_Computing_time', 'heuristic_Computing_time']
    for col in format_cols:
        if col in df.columns:
            df[col] = df[col].apply(format_seconds)

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

def aggregated_results_table(
    file_per_method: dict,
    output_tex_path: str,
    sizes=('s','m','l','xl'),
    output_path='./data/results/',
    methods_order=None 
):
    # Déterminer l’ordre des méthodes
    if methods_order is None:
        # exact d'abord, puis les autres dans l'ordre fourni par file_per_method
        keys = list(file_per_method.keys())
        methods_order = (['exact'] if 'exact' in keys else []) + [m for m in keys if m != 'exact']

    # Lignes (index)
    row_labels = [
        "min_delay", "Q1_delay", "median_delay", "avg_delay", "Q3_delay", "max_delay",
        "min_cmax",  "Q1_cmax",  "median_cmax",  "avg_cmax",  "Q3_cmax",  "max_cmax",
        "min_obj",   "Q1_obj",   "median_obj",   "avg_obj",   "Q3_obj",   "max_obj",
        "min_comp_time", "Q1_comp_time", "median_comp_time", "avg_comp_time", "Q3_comp_time", "max_comp_time",
        "nbr_best_solutions", "nbr_optimal_solutions"
    ]
    df_final = pd.DataFrame(index=row_labels)

    # Déterminer l’ordre des méthodes (par défaut: exact puis les autres dans l'ordre fourni)
    if methods_order is None:
        keys = list(file_per_method.keys())
        methods_order = (['exact'] if 'exact' in keys else []) + [m for m in keys if m != 'exact']

    # Charger les detailed_results_{size}.csv
    detailed_by_size = {}
    for size in sizes:
        path = os.path.join(output_path, f'detailed_results_{size}.csv')
        detailed_by_size[size] = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()

    # Agrégation par méthode et taille
    for method in methods_order:
        for size in sizes:
            col = f"{size}_{method}"
            df = detailed_by_size.get(size, pd.DataFrame())

            if df.empty:
                df_final[col] = [""] * len(row_labels)
                continue

            # --------- EXact ----------
            if method == 'exact':
                delay_col = 'exact_Delay' if 'exact_Delay' in df.columns else None
                cmax_col  = 'exact_Cmax'  if 'exact_Cmax'  in df.columns else None
                obj_col   = 'exact_Obj'   if 'exact_Obj'   in df.columns else None
                time_col  = 'exact_Computing_time' if 'exact_Computing_time' in df.columns else None

                # Delay
                delay_stats = _stats_6(df[delay_col]) if delay_col else [np.nan]*6
                # Cmax
                cmax_stats = _stats_6(df[cmax_col])  if cmax_col  else [np.nan]*6
                # Obj
                obj_stats = _stats_6(df[obj_col])   if obj_col   else [np.nan]*6
                # Computing time (6 stats)
                time_stats = _stats_6(df[time_col])  if time_col  else [np.nan]*6

                _place_block(df_final, col, "delay",     delay_stats, exact=True)
                _place_block(df_final, col, "cmax",      cmax_stats,  exact=True)
                _place_block(df_final, col, "obj",       obj_stats,   exact=True)
                _place_block(df_final, col, "comp_time", time_stats,  exact=True)

                for var_prefix in ["delay", "cmax", "obj", "comp_time"]:
                    for row in [f"min_{var_prefix}", f"max_{var_prefix}"]:
                        val = df_final.loc[row, col]
                        df_final.loc[row, col] = format_int(val)
                
                for var_prefix in ["comp_time"]:
                    for row in row_labels:
                        val = df_final.loc[row, col]
                        df_final.loc[row, col] = format_time(val)

                # Comptages exact
                if 'exact_Gap' in df.columns:
                    best_count = (pd.to_numeric(df['exact_Gap'], errors="coerce") == 0).sum()
                elif 'exact_Status' in df.columns:
                    best_count = (df['exact_Status'].astype(str).str.upper() == 'OPTIMAL').sum()
                else:
                    best_count = 0

                opt_count = 0
                if 'exact_Status' in df.columns:
                    opt_count = (df['exact_Status'].astype(str).str.upper() == 'OPTIMAL').sum()
                
                df_final.loc["nbr_best_solutions", col]   = str(best_count)
                df_final.loc["nbr_optimal_solutions", col] = str(opt_count)

            # --------- Méthodes approchées (heuristic/gnn/gnnls/...) ----------
            else:
                delay_dev = f'{method}_dev_Delay'
                cmax_dev  = f'{method}_dev_Cmax'
                obj_dev   = f'{method}_dev_Obj'
                time_col  = f'{method}_Computing_time'

                delay_stats = _stats_6(df[delay_dev]) if delay_dev in df.columns else [np.nan]*6
                cmax_stats  = _stats_6(df[cmax_dev])  if cmax_dev  in df.columns else [np.nan]*6
                obj_stats   = _stats_6(df[obj_dev])   if obj_dev   in df.columns else [np.nan]*6
                time_stats  = _stats_6(df[time_col])  if time_col  in df.columns else [np.nan]*6

                # Placement explicite (formatages adaptés)
                _place_block(df_final, col, "delay",     delay_stats, formatter=format_percent)
                _place_block(df_final, col, "cmax",      cmax_stats,  formatter=format_percent)
                _place_block(df_final, col, "obj",       obj_stats,   formatter=format_percent)
                _place_block(df_final, col, "comp_time", time_stats,  formatter=format_seconds)


                # Comptages
                tol = 1e-12
                # best: dev_Obj == 0
                if obj_dev in df.columns:
                    s_obj_dev = pd.to_numeric(df[obj_dev], errors="coerce")
                    best_count = (s_obj_dev.abs() <= tol).sum()
                else:
                    best_count = 0

                # optimal: dev_Obj==0 ET exact optimal
                have_exact_opt = False
                if 'exact_Status' in df.columns:
                    exact_opt_mask = (df['exact_Status'].astype(str).str.upper() == 'OPTIMAL')
                    have_exact_opt = True
                elif 'exact_Gap' in df.columns:
                    exact_opt_mask = (pd.to_numeric(df['exact_Gap'], errors="coerce") == 0)
                    have_exact_opt = True
                else:
                    exact_opt_mask = pd.Series([False]*len(df))

                if have_exact_opt and obj_dev in df.columns:
                    s = pd.to_numeric(df[obj_dev], errors="coerce")
                    opt_count = ((s.abs() <= tol) & exact_opt_mask).sum()
                else:
                    opt_count = 0

                df_final.loc["nbr_best_solutions", col]   = format_int(best_count)
                df_final.loc["nbr_optimal_solutions", col] = format_int(opt_count)

    # Ordonner les colonnes: tailles x méthodes (exact d'abord)
    ordered_cols = []
    for method in methods_order:
        for size in sizes:
            ordered_cols.append(f"{size}_{method}")
    df_final = df_final[ordered_cols]

    # Export LaTeX
    latex = df_final.to_latex(escape=False, na_rep="", column_format='l' + 'c'*len(df_final.columns))
    os.makedirs(os.path.dirname(output_tex_path), exist_ok=True)
    with open(output_tex_path, 'w') as f:
        f.write(latex)

    return df_final      


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

    df_agg = aggregated_results_table(
        file_per_method=result_files,
        sizes=('s', 'm', 'l', 'xl'),
        output_path='./data/results/',
        output_tex_path = './data/results/aggregated_results.tex',
        methods_order= ['exact', 'gnn', 'heuristic', 'gnn + ls']
    )

    detailed_results_per_method(
    file_per_methode = result_files,
    variables =variables,
    output='./data/results/'
    )
