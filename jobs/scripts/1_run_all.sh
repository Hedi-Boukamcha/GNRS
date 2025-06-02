#!/bin/bash
args=("$@")

slurm() {
    folder=$1
    for f in "$folder"/*.sh; do
        sbatch "$f"
    done
}

for arg in "${args[@]}"; do
    case $arg in
        "s")
            echo "Running small instances with exact solver..."
            slurm "./s"
            ;;
        "m")
            echo "Running medium instances with exact solver..."
            slurm "./m"
            ;;
        "l")
            echo "Running large instances with exact solver..."
            slurm "./l"
            ;;
        "xl")
            echo "Running extra-large instances with exact solver..."
            slurm "./xl"
            ;;
    esac
done
