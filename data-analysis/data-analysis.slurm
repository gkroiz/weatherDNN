#!/bin/bash
#SBATCH -N 5       ##number of nodes
#SBATCH -t 3:00:00 ##time
#SBATCH -A CLI900 ##research group name
#SBATCH -J data-analysis    ##job name
#SBATCH \-\-mem=0  ##declares all avaialble memory of node
#SBATCH -o slurm-da-1.out
#SBATCH -e slurm-da-1.err
#SBATCH -p batch

source /autofs/nccs-svm1_home1/gkroiz1/.bash_init-miniconda
conda activate weatherCNNEnv
## module load python/3.6

## srun -n16 -N1 -c1 --cpu-binds=cores --exclusive python3 ../pythonscripts/preprocessing.py

##$1 is first commandline argument
# srun -N1 python3 data-analysis.py $1 &
# srun -N1 python3 data-analysis.py $2 &
# srun -N1 python3 data-analysis.py $3 &
# # srun -N1 python3 data-analysis.py $4 &
srun -N1 python3 data-analysis.py 2001 &
srun -N1 python3 data-analysis.py 2002 &
srun -N1 python3 data-analysis.py 2003 &
srun -N1 python3 data-analysis.py 2004 &
srun -N1 python3 data-analysis.py 2005 &
# srun -N1 python3 data-analysis.py 2006 &
# srun -N1 python3 data-analysis.py 2007 &
# srun -N1 python3 data-analysis.py 2008 &
# srun -N1 python3 data-analysis.py 2009 &
# srun -N1 python3 data-analysis.py 2010 &
# srun -N1 python3 data-analysis.py 2011 &

wait