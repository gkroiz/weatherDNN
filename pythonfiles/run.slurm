#!/bin/bash
#SBATCH -N 1       ##number of nodes
#SBATCH -t 12:00:00 ##time
#SBATCH -A CLI900 ##research group name
#SBATCH -J main   ##job name
##SBATCH \-\-mem=0  ##declares all avaialble memory of node
#SBATCH -o slurm-main.out
#SBATCH -e slurm-main.err
#SBATCH -p batch

source /autofs/nccs-svm1_home1/gkroiz1/.bash_init-miniconda
conda activate weatherCNNEnv
## module load python/3.6

## srun -n16 -N1 -c1 --cpu-binds=cores --exclusive python3 ../pythonscripts/preprocessing.py

##$1 is first commandline argument
# srun -N16 python3 main.py &
srun python3 main.py &


wait