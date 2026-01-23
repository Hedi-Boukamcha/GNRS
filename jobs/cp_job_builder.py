import argparse
import os

# ############################################################
# =*= JOB GENERATOR TO SOLVE TEST INSTANCES WITH CP SOLVER =*=
# ############################################################
__author__  = "Hedi Boukamcha; Anas Neumann"
__email__   = "hedi.boukamcha.1@ulaval.ca; anas.neumann@polymtl.ca"
__version__ = "2.0.0"
__license__ = "MIT"

START_IDX: int = 1
END_IDX: int = 50
INSTANCES_SIZES = ["s", "m", "l", "xl"]

# TEST WITH: python cp_job_builder.py --account=x --parent=y --mail=x@mail.com --time=2 --memory=12
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="EPSIII job builder")
    parser.add_argument("--account", help="Compute Canada Account", required=True)
    parser.add_argument("--time", help="Time limit in hours", required=True)
    parser.add_argument("--memory", help="Memory (RAM) limit in Giga", required=True)
    parser.add_argument("--parent", help="Compute Canada Parent Account", required=True)
    parser.add_argument("--mail", help="Compute Canada Email Adress", required=True)
    args = parser.parse_args()
    BASIC_PATH = "/home/"+args.account+"/projects/def-"+args.parent+"/"+args.account+"/GNRS/"
    for size in INSTANCES_SIZES:
        os.makedirs(f"./scripts/{size}/", exist_ok=True) 
        for id in range(START_IDX, END_IDX+1):
            f = open(f"./scripts/{size}/exact_{id}.sh", "w+")
            f.write("#!/bin/bash\n")
            f.write("#SBATCH --nodes 1\n")
            f.write(f"#SBATCH --time={args.time}:20:00\n")
            f.write(f"#SBATCH --mem={args.memory}G\n")
            f.write(f"#SBATCH --cpus-per-task=32\n")
            f.write(f"#SBATCH --account=def-{args.parent}\n")
            f.write(f"#SBATCH --mail-user={args.mail}\n")
            f.write("#SBATCH --mail-type=FAIL\n")
            f.write(f"#SBATCH --output={BASIC_PATH}data/out/exact_{size}_{id}.out\n")  
            f.write("module load python/3.12\n")
            f.write("virtualenv --no-download $SLURM_TMPDIR/env\n")
            f.write("source $SLURM_TMPDIR/env/bin/activate\n")
            f.write("pip install --upgrade pip --no-index\n")
            f.write(f"pip install {BASIC_PATH}wheels/protobuf-5.28.3-*.whl\n")
            f.write(f"pip install {BASIC_PATH}wheels/immutabledict-4.2.0-*.whl\n")
            f.write("pip install --no-index -r "+BASIC_PATH+"requirements/or.txt\n")
            f.write(f"python {BASIC_PATH}cp_solver.py --type=test --size={size} --id={id} --path="+BASIC_PATH+" \n")
            f.write("deactivate\n")
            f.close()
