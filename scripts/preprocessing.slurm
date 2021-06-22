#!/bin/bash
#SBATCH -N 1       ##number of nodes
#SBATCH -t 1:00:00 ##time
#SBATCH -A CLI900 ##research group name
#SBATCH -J test    ##job name
#SBATCH \-\-mem=0  ##declares all avaialble memory of node
#SBATCH -o slurm.out
#SBATCH -e slurm.err
#SBATCH -p batch

module load python/3.6

# srun -n16 -N1 -c1 --cpu-binds=cores --exclusive python3 ../pythonscripts/preprocessing.py
srun -N1 python3 ../pythonscripts/preprocessing.py

wait
