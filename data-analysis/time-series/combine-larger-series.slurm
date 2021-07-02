#!/bin/bash
#SBATCH -N 11       ##number of nodes
#SBATCH -t 24:00:00 ##time
#SBATCH -A CLI900 ##research group name
#SBATCH -J combine-tiles   ##job name
##SBATCH \-\-mem=0  ##declares all avaialble memory of node
#SBATCH -o slurm-cls.out
#SBATCH -e slurm-cls.err
#SBATCH -p batch

source /autofs/nccs-svm1_home1/gkroiz1/.bash_init-miniconda
conda activate weatherCNNEnv
## module load python/3.6

## srun -n16 -N1 -c1 --cpu-binds=cores --exclusive python3 ../pythonscripts/preprocessing.py

##$1 is first commandline argument
srun -n16 -N1 python3 combine-larger-series.py 2001 &
srun -n16 -N1 python3 combine-larger-series.py 2002 &
srun -n16 -N1 python3 combine-larger-series.py 2003 &
srun -n16 -N1 python3 combine-larger-series.py 2004 &
srun -n16 -N1 python3 combine-larger-series.py 2005 &
srun -n16 -N1 python3 combine-larger-series.py 2006 &
srun -n16 -N1 python3 combine-larger-series.py 2007 &
srun -n16 -N1 python3 combine-larger-series.py 2008 &
srun -n16 -N1 python3 combine-larger-series.py 2009 &
srun -n16 -N1 python3 combine-larger-series.py 2010 &
srun -n16 -N1 python3 combine-larger-series.py 2011 &


wait