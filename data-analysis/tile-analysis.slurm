#!/bin/bash
#SBATCH -N 11       ##number of nodes
#SBATCH -t 24:00:00 ##time
#SBATCH -A CLI900 ##research group name
#SBATCH -J tile-analy   ##job name
##SBATCH \-\-mem=0  ##declares all avaialble memory of node
#SBATCH -o slurm-ta.out
#SBATCH -e slurm-ta.err
#SBATCH -p batch

source /autofs/nccs-svm1_home1/gkroiz1/.bash_init-miniconda
conda activate weatherCNNEnv
## module load python/3.6

##$1 is first commandline argument
srun -n16 -N1 python3 tile-analysis.py 2001 &
srun -n16 -N1 python3 tile-analysis.py 2002 &
srun -n16 -N1 python3 tile-analysis.py 2003 &
srun -n16 -N1 python3 tile-analysis.py 2004 &
srun -n16 -N1 python3 tile-analysis.py 2005 &
srun -n16 -N1 python3 tile-analysis.py 2006 &
srun -n16 -N1 python3 tile-analysis.py 2007 &
srun -n16 -N1 python3 tile-analysis.py 2008 &
srun -n16 -N1 python3 tile-analysis.py 2009 &
srun -n16 -N1 python3 tile-analysis.py 2010 &
srun -n16 -N1 python3 tile-analysis.py 2011 &

wait