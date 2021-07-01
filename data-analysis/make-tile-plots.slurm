#!/bin/bash
#SBATCH -N 1       ##number of nodes
#SBATCH -t 01:00:00 ##time
#SBATCH -A CLI900 ##research group name
#SBATCH -J tile-plots   ##job name
##SBATCH \-\-mem=0  ##declares all avaialble memory of node
#SBATCH -o slurm-mtp.out
#SBATCH -e slurm-mtp.err
#SBATCH -p batch

source /autofs/nccs-svm1_home1/gkroiz1/.bash_init-miniconda
conda activate weatherCNNEnv
## module load python/3.6

##$1 is first commandline argument
srun -N1 python3 make-tile-plots.py 2001 &
srun -N1 python3 make-tile-plots.py 2002 &
srun -N1 python3 make-tile-plots.py 2003 &
srun -N1 python3 make-tile-plots.py 2004 &
srun -N1 python3 make-tile-plots.py 2005 &
srun -N1 python3 make-tile-plots.py 2006 &
srun -N1 python3 make-tile-plots.py 2007 &
srun -N1 python3 make-tile-plots.py 2008 &
srun -N1 python3 make-tile-plots.py 2009 &
srun -N1 python3 make-tile-plots.py 2010 &
srun -N1 python3 make-tile-plots.py 2011 &

wait