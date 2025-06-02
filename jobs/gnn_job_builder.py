import argparse

# ######################################################
# =*= SINGLE JOB GENERATOR TO TRAIN THE GNN WITH DQN =*=
# ######################################################
__author__ = "Hedi Boukamcha - hedi.boukamcha.1@ulaval.ca"
__version__ = "1.0.0"
__license__ = "MIT"

'''
    TEST WITH
    python gnn_job_builder.py --account=x --parent=y --mail=x@mail.com --time=20 --memory=187 --cpu=16
'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="EPSIII job builder")
    parser.add_argument("--account", help="Compute Canada Account", required=True)
    parser.add_argument("--parent", help="Compute Canada Parent Account", required=True)
    parser.add_argument("--mail", help="Compute Canada Email Adress", required=True)
    parser.add_argument("--time", help="Computing time", required=True)
    parser.add_argument("--memory", help="Computing RAM", required=True)
    parser.add_argument("--cpu", help="Computing CPUs", required=True)
    args = parser.parse_args()
    BASIC_PATH = "/home/"+args.account+"/projects/def-"+args.parent+"/"+args.account+"/gns/"

    f = open("./scripts/train_gnn.sh", "w+")
    f.write("#!/bin/bash\n")
    f.write("#SBATCH --nodes 1\n")
    f.write(f"#SBATCH --time={args.time}:00:00\n")
    f.write(f"#SBATCH --mem={args.memory}G\n")
    f.write(f"#SBATCH --cpus-per-task={args.cpu}\n")
    f.write(f"#SBATCH --account=def-{args.parent}\n")
    f.write(f"#SBATCH --mail-user={args.mail}\n")
    f.write("#SBATCH --gres=gpu:1\n")
    f.write("#SBATCH --mail-type=FAIL\n")
    f.write(f"#SBATCH --output={BASIC_PATH}data/out/train_gns.out\n")  
    f.write("module load python/3.12\n")
    f.write("module load cuda/10.2\n")
    f.write("virtualenv --no-download $SLURM_TMPDIR/env\n")
    f.write("source $SLURM_TMPDIR/env/bin/activate\n")
    f.write("pip install --upgrade pip --no-index\n")
    f.write("pip install --no-index -r "+BASIC_PATH+"requirements/drl.txt\n")
    f.write(f"python {BASIC_PATH}gnn_solver.py --interactive=false --mode=train --path="+BASIC_PATH+" \n")
    f.write("deactivate\n")
    f.close()
