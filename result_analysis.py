import pandas as pd
import os

# ##############################
# =*= FINAL RESULTS ANALYSIS =*=
# ##############################
__author__  = "Hedi Boukamcha; Anas Neumann"
__email__   = "hedi.boukamcha.1@ulaval.ca; anas.neumann@polymtl.ca"
__version__ = "1.0.0"
__license__ = "MIT"




def exact_solver_results():
    path = "./data/instances/test/"
    results_path = "./data/results/"
    sizes = ['s', 'm', 'l', 'xl',]

    columns = ['Size', 'Instance ID', 'Status', 'Obj', 'Delay', 'Cmax', 'Computing_time']
    rows = []

    for size in sizes:
        for inst_id in range(51):
            filename = os.path.join(path, f"{size}/exact_solution_{inst_id}.csv")
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
    df_exact.to_csv(results_path+"exact_solver_results.csv", index=False)


if __name__ == "__main__":
    exact_solver_results()
    pass