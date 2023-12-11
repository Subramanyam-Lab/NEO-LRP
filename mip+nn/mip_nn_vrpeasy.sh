#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=64GB
#SBATCH --job-name=mip_nn_vrpeasy
#SBATCH --output=mip_nn_vrpeasy.out
#SBATCH --time=5-00:00:00
#SBATCH --error=mip_nn_vrpeasy.err
#SBATCH --account=azs7266_sc
#SBATCH --partition=sla-prio

source ~/.bashrc
module load anaconda
module load gurobi/10.0.3
conda activate vrpsolvereasy

name_patterns=(
    "coord20-5-1"
    "coord20-5-1b"
    "coord20-5-2"
    "coord20-5-2b"
    "coord50-5-1"
    "coord50-5-1b"
    "coord50-5-2"
    "coord50-5-2b"
    "coord50-5-2BIS"
    "coord50-5-2bBIS"
    "coord50-5-3"
    "coord50-5-3b"
    "coord100-5-1"
    "coord100-5-1b"
    "coord100-5-2"
    "coord100-5-2b"
    "coord100-5-3"
    "coord100-5-3b"
    "coord100-10-1"
    "coord100-10-1b"
    "coord100-10-2"
    "coord100-10-2b"
    "coord100-10-3"
    "coord100-10-3b"
    "coord200-10-1"
    "coord200-10-1b"
    "coord200-10-2"
    "coord200-10-2b"
    "coord200-10-3"
    "coord200-10-3b"
)

# name_patterns=(
#     "coord20-5-1b"
# )



for name_pattern in "${name_patterns[@]}"
do
    export name_pattern="$name_pattern"    
    /storage/home/wzk5140/.conda/envs/weber/bin/python mip_nn_vrpeasy.py
done

echo "All instances have completed running."