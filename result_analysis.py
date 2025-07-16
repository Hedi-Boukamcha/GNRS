import pandas as pd
import os

# ##############################
# =*= FINAL RESULTS ANALYSIS =*=
# ##############################
__author__  = "Hedi Boukamcha; Anas Neumann"
__email__   = "hedi.boukamcha.1@ulaval.ca; anas.neumann@polymtl.ca"
__version__ = "1.0.0"
__license__ = "MIT"




def exact_solver_results(result_type: str, output_file: str):
    path = "./data/instances/test/"
    results_path = "./data/results/"
    sizes = ['s', 'm', 'l', 'xl',]
    
    columns = ['Size', 'Instance ID', 'Status', 'Obj', 'Delay', 'Cmax', 'Computing_time']
    rows = []

    for size in sizes:
        for inst_id in range(51):
            filename = os.path.join(path, f"{size}/{result_type}_{inst_id}.csv")
            if os.path.exists(filename):
                df = pd.read_csv(filename)
                row_data = df.iloc[0].to_dict()
                row = {
                    'Size': size,
                    'Instance ID': inst_id,
                    'Status': row_data.get('status'),
                    'Obj': row_data.get('obj'),
                    'Delay': row_data.get('delay'),
                    'Cmax': row_data.get('cmax'),
                    'Computing_time': row_data.get('computing_time')
                }
            else:
                row = {
                    'Size': size,
                    'Instance ID': inst_id,
                    'Status': None,
                    'Obj': None,
                    'Delay': None,
                    'Cmax': None,
                    'Computing_time': None
                }
            rows.append(row)
    df_exact = pd.DataFrame(rows, columns=columns)
    df_exact['Size'] = pd.Categorical(df_exact['Size'], categories=sizes, ordered=True)
    df_exact = df_exact.sort_values(['Size', 'Instance ID'])
    df_exact.to_csv(results_path + output_file, index=False)


if __name__ == "__main__":
    exact_solver_results(result_type='exact_solution', output_file='exact_solution_results.csv')
    exact_solver_results(result_type='gnn_solution', output_file='gnn_solution_results.csv')
    exact_solver_results(result_type='gnn_solution_improved', output_file='gnn_solution_improved_results.csv')
    exact_solver_results(result_type='heuristic_solution', output_file='heuristic_solution_results.csv')
