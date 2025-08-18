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

# --- Format second  ---
def format_seconds(value):
    try:
        return f"{float(value):.2f}s"
    except:
        return value

# --- Format comma  ---
def format_comma(value):
    try:
        return f"{float(value):.2f}"
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
    
def count_best_by_instance(df: pd.DataFrame, methods_order, tol=1e-12):

    counts = {m: 0 for m in methods_order}
    # colonnes attendues par méthode
    col_for_method = { 'exact': 'exact_Obj', **{m: f'{m}_Obj' for m in methods_order if m != 'exact'} }

    # itère sur chaque instance (ligne)
    for _, row in df.iterrows():
        # collecte des valeurs Obj dispo pour cette instance
        vals = []
        for m in methods_order:
            col = col_for_method.get(m)
            if col in df.columns:
                v = row[col]
                if pd.notna(v):
                    vals.append((m, float(v)))

        if not vals:
            continue

        # min de la ligne
        min_val = min(v for _, v in vals)

        # incrémente +1 pour chaque méthode à min (ex æquo inclus)
        for m, v in vals:
            if abs(v - min_val) <= tol:
                counts[m] += 1

    return counts

def csv_to_latex_table(input_csv: str, output_tex: str, float_format="%.3f"):
    df = pd.read_csv(input_csv)

    math_cols_src  = ['exact_Status', 'exact_Computing_time', 'exact_Obj', 'exact_Cmax', 'exact_Delay', 'exact_Gap']
    ls_cols_src    = ['heuristic_Computing_time', 'heuristic_dev_Obj', 'heuristic_dev_Cmax', 'heuristic_dev_Delay']
    gnn_cols_src   = ['gnn_Computing_time',       'gnn_dev_Obj',       'gnn_dev_Cmax',       'gnn_dev_Delay']
    gnnls_cols_src = ['gnn + ls_Computing_time',  'gnn + ls_dev_Obj',  'gnn + ls_dev_Cmax',  'gnn + ls_dev_Delay']

    # Vérifie la présence des colonnes pour éviter les erreurs
    columns_present = df.columns.tolist()
    math_cols_src = [col for col in math_cols_src if col in columns_present]
    ls_cols_src   = [col for col in ls_cols_src if col in columns_present]
    gnn_cols_src  = [col for col in gnn_cols_src if col in columns_present]
    gnnls_cols_src = [col for col in gnnls_cols_src if col in columns_present]

    # Final column order
    ordered_columns = ['Instance ID'] + math_cols_src + ls_cols_src + gnn_cols_src + gnnls_cols_src
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
        ['Status', 'exact_CT', 'exact_Obj', 'exact_Cmax', 'exact_Delay', 'exact_Gap'][:len(math_cols_src)] +
        ['ls_CT', 'ls_Dev_Obj', 'ls_Dev_Cmax', 'ls_Dev_Delay'][:len(ls_cols_src)] +
        ['gnn_CT', 'gnn_Dev_Obj', 'gnn_Dev_Cmax', 'gnn_Dev_Delay'][:len(gnn_cols_src)] +
        ['gnn+ls_CT', 'gnn+ls_Dev_Obj', 'gnn+ls_Dev_Cmax', 'gnn+ls_Dev_Delay'][:len(gnnls_cols_src)]
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

    # Ordre de base pour sauvegarde CSV détaillé (on filtrera celles présentes)
    base_order = [
        'Instance ID',
        'exact_Status', 'exact_Computing_time', 'exact_Obj', 'exact_Cmax', 'exact_Delay', 'exact_Gap',
        'heuristic_Computing_time', 'heuristic_dev_Obj', 'heuristic_dev_Cmax', 'heuristic_dev_Delay',
        'gnn_Computing_time', 'gnn_dev_Obj', 'gnn_dev_Cmax', 'gnn_dev_Delay',
        'gnn + ls_Computing_time', 'gnn + ls_dev_Obj', 'gnn + ls_dev_Cmax', 'gnn + ls_dev_Delay',
    ]

    for size in sizes:
        # squelette: toutes les instances 0..50
        df_result = pd.DataFrame({'Instance ID': list(range(51))})

        # -------- exact --------
        exact_df = None
        if 'exact' in file_per_methode:
            csv_file = file_per_methode['exact']
            if os.path.exists(csv_file):
                df = pd.read_csv(csv_file)
                # filtre par taille si présent
                if 'Size' in df.columns:
                    df = df[df['Size'] == size]
                # 1 ligne par instance
                df = df.drop_duplicates(subset='Instance ID')
                if 'Instance ID' in df.columns:
                    exact_df = df.set_index('Instance ID')

                    # variables de base
                    for var in variables:
                        if var in exact_df.columns:
                            df_result[f'exact_{var}'] = df_result['Instance ID'].map(exact_df[var])

                    # status / gap si présents
                    if 'Status' in exact_df.columns:
                        df_result['exact_Status'] = df_result['Instance ID'].map(exact_df['Status'])
                    if 'Gap' in exact_df.columns:
                        df_result['exact_Gap'] = df_result['Instance ID'].map(exact_df['Gap'])
            else:
                print(f"[WARN] exact file not found: {csv_file}")

        # -------- autres méthodes (déviations + CT) --------
        for method, csv_file in file_per_methode.items():
            if method == 'exact':
                continue
            if not os.path.exists(csv_file):
                print(f"[WARN] file not found for {method}: {csv_file}")
                continue

            df = pd.read_csv(csv_file)
            if 'Size' in df.columns:
                df = df[df['Size'] == size]
            df = df.drop_duplicates(subset='Instance ID')
            if 'Instance ID' not in df.columns:
                print(f"[WARN] '{method}' missing 'Instance ID' for size={size}")
                continue

            df_m = df.set_index('Instance ID')

            # déviations signées pour Obj/Cmax/Delay
            for var in ['Obj', 'Cmax', 'Delay']:
                exact_col = f'exact_{var}'
                if var in df_m.columns and exact_col in df_result.columns:
                    approx_vals = df_result['Instance ID'].map(df_m[var])
                    base_vals   = df_result[exact_col]
                    dev = (approx_vals - base_vals) / (base_vals + 1e-8)
                    df_result[f'{method}_dev_{var}'] = dev

            # computing time brut
            if 'Computing_time' in df_m.columns:
                df_result[f'{method}_Computing_time'] = df_result['Instance ID'].map(df_m['Computing_time'])

        # -------- ordre de colonnes stable à l'écriture --------
        present_cols = [c for c in base_order if c in df_result.columns]
        df_out = df_result[['Instance ID'] + [c for c in present_cols if c != 'Instance ID']]

        # sauvegarde CSV
        out_csv = os.path.join(output, f'detailed_results_{size}.csv')
        df_out.to_csv(out_csv, index=False)

        # génération LaTeX associée (utilise ton csv_to_latex_table avec col_labels fixes)
        out_tex = os.path.join(output, f'detailed_results_{size}.tex')
        csv_to_latex_table(out_csv, out_tex)

def aggregated_results_table(
    file_per_method: dict,
    output_tex_path: str,
    sizes=('s','m','l','xl'),
    output_path='./data/results/',
    methods_order=None 
):
    # 1) ordre des méthodes (exact d'abord par défaut)
    if methods_order is None:
        keys = list(file_per_method.keys())
        methods_order = (['exact'] if 'exact' in keys else []) + [m for m in keys if m != 'exact']

    # 2) lignes (index) -> on supprime "nbr_best_solutions"
    row_labels = [
        "min_delay", "Q1_delay", "median_delay", "avg_delay", "Q3_delay", "max_delay",
        "min_cmax",  "Q1_cmax",  "median_cmax",  "avg_cmax",  "Q3_cmax",  "max_cmax",
        "min_obj",   "Q1_obj",   "median_obj",   "avg_obj",   "Q3_obj",   "max_obj",
        "min_comp_time", "Q1_comp_time", "median_comp_time", "avg_comp_time", "Q3_comp_time", "max_comp_time",
        "nbr_best_solutions", "nbr_optimal_solutions"
    ]
    df_final = pd.DataFrame(index=row_labels)

    # charger les detailed_results_{size}.csv
    detailed_by_size = {}
    for size in sizes:
        path = os.path.join(output_path, f'detailed_results_{size}.csv')
        detailed_by_size[size] = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()

    # ---- 4) agrégation ----
    for method in methods_order:
        for size in sizes:
            col = f"{size}_{method}"
            df = detailed_by_size.get(size, pd.DataFrame())

            if df.empty:
                df_final[col] = [""] * len(row_labels)
                continue
            if col not in df_final.columns:
                df_final[col] = ""

            if method == 'exact':
                # colonnes attendues dans detailed_results_*.csv
                delay_col = 'exact_Delay' if 'exact_Delay' in df.columns else None
                cmax_col  = 'exact_Cmax'  if 'exact_Cmax'  in df.columns else None
                obj_col   = 'exact_Obj'   if 'exact_Obj'   in df.columns else None
                time_col  = 'exact_Computing_time' if 'exact_Computing_time' in df.columns else None

                delay_stats = _stats_6(df[delay_col]) if delay_col else [np.nan]*6
                cmax_stats  = _stats_6(df[cmax_col])  if cmax_col  else [np.nan]*6
                obj_stats   = _stats_6(df[obj_col])   if obj_col   else [np.nan]*6
                time_stats  = _stats_6(df[time_col])  if time_col  else [np.nan]*6

                # placer les blocs
                _place_block(df_final, col, "delay",     delay_stats, exact=True)
                _place_block(df_final, col, "cmax",      cmax_stats,  exact=True)
                _place_block(df_final, col, "obj",       obj_stats,   exact=True)
                _place_block(df_final, col, "comp_time", time_stats,  exact=True)

                # min/max en entier uniquement
                for var_prefix in ["delay", "cmax", "obj", "comp_time"]:
                    for row in (f"min_{var_prefix}", f"max_{var_prefix}"):
                        df_final.loc[row, col] = format_int(df_final.loc[row, col])
                
                # temps formaté
                for row in (f"min_comp_time", f"Q1_comp_time", f"median_comp_time", 
                            f"avg_comp_time", f"Q3_comp_time", f"max_comp_time"):
                    df_final.loc[row, col] = format_time(df_final.loc[row, col])

                # Comptage optimal uniquement
                opt_count = 0
                if 'exact_Status' in df.columns:
                    opt_count = (df['exact_Status'].astype(str).str.upper() == 'OPTIMAL').sum()
                elif 'exact_Gap' in df.columns:
                    opt_count = (pd.to_numeric(df['exact_Gap'], errors="coerce") == 0).sum()

                df_final.loc["nbr_optimal_solutions", col] = str(opt_count)

            else:
                delay_dev = f'{method}_dev_Delay'
                cmax_dev  = f'{method}_dev_Cmax'
                obj_dev   = f'{method}_dev_Obj'
                time_col  = f'{method}_Computing_time'

                delay_stats = _stats_6(df[delay_dev]) if delay_dev in df.columns else [np.nan]*6
                cmax_stats  = _stats_6(df[cmax_dev])  if cmax_dev  in df.columns else [np.nan]*6
                obj_stats   = _stats_6(df[obj_dev])   if obj_dev   in df.columns else [np.nan]*6
                time_stats  = _stats_6(df[time_col])  if time_col  in df.columns else [np.nan]*6

                _place_block(df_final, col, "delay",     delay_stats, formatter=format_percent)
                _place_block(df_final, col, "cmax",      cmax_stats,  formatter=format_percent)
                _place_block(df_final, col, "obj",       obj_stats,   formatter=format_percent)
                _place_block(df_final, col, "comp_time", time_stats,  formatter=format_seconds)

                # Comptage optimal uniquement
                opt_count = 0
                if 'exact_Status' in df.columns and obj_dev in df.columns:
                    exact_opt_mask = (df['exact_Status'].astype(str).str.upper() == 'OPTIMAL')
                    s = pd.to_numeric(df[obj_dev], errors="coerce")
                    tol = 1e-12
                    opt_count = ((s.abs() <= tol) & exact_opt_mask).sum()

                df_final.loc["nbr_optimal_solutions", col] = str(opt_count)

    # 5) ordre des colonnes: tailles x méthodes
    ordered_cols = [f"{size}_{method}" for method in methods_order for size in sizes]
    df_final = df_final[ordered_cols]

    # 6) export LaTeX
    os.makedirs(os.path.dirname(output_tex_path), exist_ok=True)
    latex = df_final.to_latex(escape=False, na_rep="", column_format='l' + 'c'*len(df_final.columns))
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
