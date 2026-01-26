import pandas as pd
import numpy as np
import os

from utils.common import format_seconds, format_time, format_percent, format_int

# ##############################
# =*= FINAL RESULTS ANALYSIS =*=
# ##############################
__author__  = "Hedi Boukamcha; Anas Neumann"
__email__   = "hedi.boukamcha.1@ulaval.ca; anas.neumann@polymtl.ca"
__version__ = "2.0.0"
__license__ = "MIT"

INSTANCE_ID: str   = 'Instance ID'
instances_path:str = "./data/instances/test/"
results_path: str  = "./data/results/"

sizes:     list[str]     = ['s', 'm', 'l', 'xl']
few_variables: list[str] = ['Delay', 'Cmax', 'Obj', 'Computing_time']
all_variables: list[str] = ['Size', INSTANCE_ID, 'Obj', 'Delay', 'Cmax', 'Computing_time']

gnn_approches:        list[str] = ["basic_gnn_solution", "basic_gnn_solution_improved", "basic_gnn_solution_improved_beam", "gnn_solution", "gnn_solution_improved", ]
exact_approaches:     list[str] = ["exact_solution"]
heuristic_approaches: list[str] = ["heuristic_solution", "TS_solution"]
all_approaches:       list[str] = gnn_approches + exact_approaches + heuristic_approaches

def _stats_6(series: pd.Series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty: return [np.nan]*6
    return [
        s.min(),
        s.quantile(0.25), # Q1
        s.median(),
        s.mean(),
        s.quantile(0.75), #Q3
        s.max()]

def _place_block(df_final: pd.DataFrame, col_name: str, var_prefix: str, values_as_list: list, formatter=lambda x: x, exact: bool=False):
    rows: list[str] = [f"min_{var_prefix}", f"Q1_{var_prefix}", f"median_{var_prefix}", f"avg_{var_prefix}", f"Q3_{var_prefix}", f"max_{var_prefix}"]
    for row, v in zip(rows, values_as_list):
        if exact: df_final.loc[row, col_name] = "" if (v is None or (isinstance(v, float) and np.isnan(v))) else str(v)
        else: df_final.loc[row, col_name] = formatter(v)
    
def count_best_by_instance(df: pd.DataFrame, methods_order: list[str], tol=1e-12):
    counts = {m: 0 for m in methods_order}
    col_for_method = { 'exact_solution': 'exact_solution_Obj', **{m: f'{m}_Obj' for m in methods_order if m != 'exact_solution'} }
    for _, row in df.iterrows():
        vals = []
        for m in methods_order:
            col = col_for_method.get(m)
            if col in df.columns:
                v = row[col]
                if pd.notna(v):
                    vals.append((m, float(v)))
        if not vals:
            continue
        min_val = min(v for _, v in vals)
        for m, v in vals:
            if abs(v - min_val) <= tol:
                counts[m] += 1
    return counts

def csv_to_latex_table(input_csv: str, output_tex: str, float_format="%.3f"):
    df = pd.read_csv(input_csv)
    if 'exact_solution_Status' in df.columns:
        df['exact_solution_Status'] = df['exact_solution_Status'].apply(lambda x: 'F' if str(x).lower().startswith('f') else 'O')
    for col in df.columns:
        if 'time' in col.lower():
            if 'exact' in col.lower():
                df[col] = df[col].apply(format_time)
            else:
                df[col] = df[col].apply(format_seconds)
        elif '_Gap' in col.lower() or '_dev' in col.lower():
            df[col] = df[col].apply(format_percent)
        elif 'exact_' in col.lower() and 'time' not in col.lower():
            df[col] = df[col].apply(format_int)
    latex_code = df.to_latex(index=False, float_format=float_format, escape=False)
    with open(output_tex, 'w') as f:
        f.write(latex_code)

def detailed_results_per_method(file_per_method: dict, variables: list):
    base_order = [INSTANCE_ID, 'exact_Status', 'exact_Computing_time', 'exact_Obj', 'exact_Cmax', 'exact_Delay', 'exact_Gap'] 
    + [col for a in all_approaches[1:]  for col in (f'{a}_Computing_time', f'{a}_dev_Obj', f'{a}_dev_Cmax', f'{a}_dev_Delay')]
    for size in sizes:
        df_result = pd.DataFrame({INSTANCE_ID: list(range(51))})
        exact_df = None
        if 'exact' in file_per_method: # EXACT CP SOLUTION
            csv_file = file_per_method['exact_solution']
            if os.path.exists(csv_file):
                df = pd.read_csv(csv_file)
                if 'Size' in df.columns:
                    df = df[df['Size'] == size]
                df = df.drop_duplicates(subset=INSTANCE_ID)
                if INSTANCE_ID in df.columns:
                    exact_df = df.set_index(INSTANCE_ID)
                    for var in variables:
                        if var in exact_df.columns:
                            df_result[f'exact_{var}'] = df_result[INSTANCE_ID].map(exact_df[var])
                    if 'Status' in exact_df.columns:
                        df_result['exact_Status'] = df_result[INSTANCE_ID].map(exact_df['Status'])
                    if 'Gap' in exact_df.columns:
                        df_result['exact_Gap'] = df_result[INSTANCE_ID].map(exact_df['Gap'])
            else:
                print(f"[WARN] exact file not found: {csv_file}")
        for method, csv_file in file_per_method.items(): # OTHER METHODS
            if method == 'exact':
                continue
            if not os.path.exists(csv_file):
                print(f"[WARN] file not found for {method}: {csv_file}")
                continue
            df = pd.read_csv(csv_file)
            if 'Size' in df.columns:
                df = df[df['Size'] == size]
            df = df.drop_duplicates(subset=INSTANCE_ID)
            if INSTANCE_ID not in df.columns:
                print(f"[WARN] '{method}' missing INSTANCE_ID for size={size}")
                continue
            df_m = df.set_index(INSTANCE_ID)
            for var in ['Obj', 'Cmax', 'Delay']:
                exact_col = f'exact_{var}'
                if var in df_m.columns and exact_col in df_result.columns:
                    approx_vals = df_result[INSTANCE_ID].map(df_m[var])
                    base_vals   = df_result[exact_col]
                    dev = (approx_vals - base_vals) / (base_vals + 1e-8)
                    df_result[f'{method}_dev_{var}'] = dev
            if 'Computing_time' in df_m.columns:
                df_result[f'{method}_Computing_time'] = df_result[INSTANCE_ID].map(df_m['Computing_time'])
        present_cols = [c for c in base_order if c in df_result.columns]
        df_out = df_result[[INSTANCE_ID] + [c for c in present_cols if c != INSTANCE_ID]]
        out_csv = os.path.join(results_path, f'detailed_results_{size}.csv')
        df_out.to_csv(out_csv, index=False)
        out_tex = os.path.join(results_path, f'detailed_results_{size}.tex')
        csv_to_latex_table(out_csv, out_tex)

def aggregated_results_table(file_per_method: dict, output_tex_path: str):
    keys:          list[str] = list(file_per_method.keys())
    methods_order: list[str] = (['exact'] if 'exact' in keys else []) + [m for m in keys if m != 'exact']
    row_labels = [
        "min_delay", "Q1_delay", "median_delay", "avg_delay", "Q3_delay", "max_delay",
        "min_cmax",  "Q1_cmax",  "median_cmax",  "avg_cmax",  "Q3_cmax",  "max_cmax",
        "min_obj",   "Q1_obj",   "median_obj",   "avg_obj",   "Q3_obj",   "max_obj",
        "min_comp_time", "Q1_comp_time", "median_comp_time", "avg_comp_time", "Q3_comp_time", "max_comp_time",
        "nbr_best_solutions", "nbr_optimal_solutions"]
    df_final = pd.DataFrame(index=row_labels)
    detailed_by_size = {}
    for size in sizes:
        path = os.path.join(results_path, f'detailed_results_{size}.csv')
        detailed_by_size[size] = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()
    for method in methods_order:
        for size in sizes:
            col = f"{size}_{method}"
            df = detailed_by_size.get(size, pd.DataFrame())
            if df.empty:
                df_final[col] = [""] * len(row_labels)
                continue
            if col not in df_final.columns:
                df_final[col] = ""
            objs = pd.DataFrame(index=df.index)
            if 'exact_Obj' in df.columns:
                objs['exact'] = pd.to_numeric(df['exact_Obj'], errors="coerce")
            for m in gnn_approches + heuristic_approaches:
                obj_col = f'{m}_Obj'
                dev_col = f'{m}_dev_Obj'
                if obj_col in df.columns:
                    objs[m] = pd.to_numeric(df[obj_col], errors="coerce")
                elif dev_col in df.columns and 'exact_Obj' in df.columns:
                    objs[m] = (1.0 + pd.to_numeric(df[dev_col], errors="coerce")) * pd.to_numeric(df['exact_Obj'], errors="coerce")
            best_counts = {m: 0 for m in all_approaches}
            for _, row in objs.iterrows():
                row = row.dropna()
                if not row.empty:
                    min_val = row.min()
                    bests = row[row == min_val].index
                    for best in bests:
                        best_counts[best] += 1
            if method == exact_approaches[0]: # EXACT CP SOLVER
                delay_col = 'exact_Delay' if 'exact_Delay' in df.columns else None
                cmax_col  = 'exact_Cmax'  if 'exact_Cmax'  in df.columns else None
                obj_col   = 'exact_Obj'   if 'exact_Obj'   in df.columns else None
                time_col  = 'exact_Computing_time' if 'exact_Computing_time' in df.columns else None

                delay_stats = _stats_6(df[delay_col]) if delay_col else [np.nan]*6
                cmax_stats  = _stats_6(df[cmax_col])  if cmax_col  else [np.nan]*6
                obj_stats   = _stats_6(df[obj_col])   if obj_col   else [np.nan]*6
                time_stats  = _stats_6(df[time_col])  if time_col  else [np.nan]*6

                _place_block(df_final, col, "delay",     delay_stats, exact=True)
                _place_block(df_final, col, "cmax",      cmax_stats,  exact=True)
                _place_block(df_final, col, "obj",       obj_stats,   exact=True)
                _place_block(df_final, col, "comp_time", time_stats,  exact=True)

                for var_prefix in ["delay", "cmax", "obj", "comp_time"]:
                    for row in (f"min_{var_prefix}", f"max_{var_prefix}"):
                        df_final.loc[row, col] = format_int(df_final.loc[row, col])
                
                for row in (f"min_comp_time", f"Q1_comp_time", f"median_comp_time", 
                            f"avg_comp_time", f"Q3_comp_time", f"max_comp_time"):
                    df_final.loc[row, col] = format_time(df_final.loc[row, col])

                opt_count = 0
                if 'exact_Status' in df.columns:
                    opt_count = (df['exact_Status'].astype(str).str.upper() == 'OPTIMAL').sum()
                elif 'exact_Gap' in df.columns:
                    opt_count = (pd.to_numeric(df['exact_Gap'], errors="coerce") == 0).sum()
                df_final.loc["nbr_optimal_solutions", col] = str(opt_count)
                df_final.loc["nbr_best_solutions", col] = format_int(best_counts.get(method, 0))

            else: # OTHER APPROACHES
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

                opt_count = 0
                if 'exact_Status' in df.columns and obj_dev in df.columns:
                    exact_opt_mask = (df['exact_Status'].astype(str).str.upper() == 'OPTIMAL')
                    s = pd.to_numeric(df[obj_dev], errors="coerce")
                    tol = 1e-12
                    opt_count = ((s.abs() <= tol) & exact_opt_mask).sum()
                df_final.loc["nbr_optimal_solutions", col] = str(opt_count)
                df_final.loc["nbr_best_solutions", col] = format_int(best_counts.get(method, 0))
    ordered_cols = [f"{size}_{method}" for method in methods_order for size in sizes]
    df_final = df_final[ordered_cols]
    os.makedirs(os.path.dirname(output_tex_path), exist_ok=True)
    latex = df_final.to_latex(escape=False, na_rep="", column_format='l' + 'c'*len(df_final.columns))
    with open(output_tex_path, 'w') as f:
        f.write(latex)

def results_tables(result_type: str, output_file: str):
    rows = []
    status_exists_globally = False
    gap_exists_globally = False
    for size in sizes:
        for inst_id in range(51):
            filename = os.path.join(instances_path, f"{size}/{result_type}_{inst_id}.csv")
            row = {
                'Size':           size,
                INSTANCE_ID:    inst_id,
                'Obj':            None,
                'Delay':          None,
                'Cmax':           None,
                'Computing_time': None}
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
    df_result         = df_result.sort_values(['Size', INSTANCE_ID])
    columns           = ['Size', INSTANCE_ID]
    if status_exists_globally and 'Status' in df_result.columns:
        columns.append('Status')
    if gap_exists_globally and 'Gap' in df_result.columns:
        columns.append('Gap')
    columns += ['Computing_time', 'Obj', 'Cmax', 'Delay']
    df_result = df_result[columns]
    df_result.to_csv(results_path + output_file, index=False)

# python3 result_analysis.py
if __name__ == "__main__":
    for a in all_approaches:
         results_tables(result_type=a, output_file=f'{a}_results.csv')
    result_files: dict = { **{a: f'{results_path}{a}_results.csv' for a in all_approaches}}
    aggregated_results_table(file_per_method=result_files, output_tex_path=results_path+'aggregated_results.tex')
    detailed_results_per_method(file_per_method=result_files, variables=few_variables)