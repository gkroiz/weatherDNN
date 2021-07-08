# weatherDNN
deep neural network for SULI program at ORNL 2021

### data-analysis 
This directory includes files that 
1) analyze data (data-analysis.py/slurm)
2) take 1024x1024 grid and create 256 subtiles (subsampling.py/slurm)
3) analyze tiles (tile-analysis.py/slurm)
4) combine tiles into months and years (time-series)

### pythonfiles
This directory includes 
1) the convLSTM (model.py) 
2) the main code to create train/val/test datasets, and use these on the model (main.py/run.slurm)
3) remaining files are empty

To run this on andes, please use run.slurm, which runs main.py as a slurm job.
Alternatively, you can run directly using python3

### Conda Environment Dependencies
dependencies:
  - xarray
  - tensorflow
  - seaborn
  - pickleshare
  - jupyter
  - jupyterlab
  - netcdf4
  - dask
  - python=3.7
  - mpi4py
  - nco
  - cdo=1.9.8
  - h5netcdf=0.11.0

