# PrecipRate Benchmark
PrecipRate Benchmark was designed for the SULI program at ORNL (2021). Conceptually, this dataset provides a benchmark dataset of precipitaion rates. Additionally, there is a baseline model that is provided. These two features are included to facilitate comparision between different deep learning architectures. The data is from the Multi-RADAR/Multi-sensor System (MRMS). A more in-depth description is provided in the technical report ```SULI_Report.pdf```.

To use this benchmark dataset, you must install the conda environment with the dependencies below. Using different versions of the dependencies may result in errors.


## Conda Environment Dependencies
dependencies:
  - python=3.7
  - tensorflow=2.4.1
  - tensorflow-gpu=2.4.1
  - seaborn=0.11.1
  - netcdf4=1.5.7
  - dask=2021.7.0
  - cdo=1.9.8
  - h5netcdf=0.11.0
  - pandas=1.1.3
  - xarray=0.18.2
  - scikit-image=0.18.1
  - numpy=1.19.5


## Repository structure
1) ```tile-generation```
2) ```deep-learning```

The ```tile-generation``` directory contains files needed to create the tiles. The ```deep-learning``` directory contains the baseline model, which is a convolutional LSTM (convLSTM).

## Step 1: tile generation 
For this step, run the code in ```tile-generation```. The entirety of this step will take place in ```tiles-generation```. The original MRMS dataset consists of 11 years of netcdf files of a 1024x1024 region. Each netcdf file is a 5 minute interval within the 11 years. The directory structure of the provided data is as follows:

```bash
$ tree -L 1 .
.
├── 2001
├── 2002
├── 2003
├── 2004
├── 2005
├── 2006
├── 2007
├── 2008
├── 2009
├── 2010
└── 2011
```

The current code only works via slurm scripting. This is to parallelize the process. However, there are explanations on how this can be changed to your settings.

### Step 1.1: Run create-tiles.py
First, we want to create the tiles from the 1024x1024 region. You first need to fill in the json parameters for the ```create-tiles.json``` file. Next, run the following:

```
srun -n{NUMBER OF PROCESSES PER NODE} -N1 python3 create-tiles.py {YEAR} &
```

```NUMBER OF PROCESSES PER NODE``` will determine the rank within ```create-tiles.py```

```YEAR``` will determine what year you are creating tiles for. This is a required command line argument for 

To create tiles for all 11 years of data, you will need to run the ```srun``` command for each year from 2001-2011. Running this will result in the same directory structure as shown previously

### Step 1.2: Run combine-single-tiles.py
Once you have created the tiles in Step 1.1, you will need to combine the tiles. First fill in the json parameters for the ```combine-tiles.json``` file. Next, run the following:

```
srun -n{NUMBER OF PROCESSES PER NODE} -N1 python3 combine-single-tiles.py {YEAR} &
```

To create tiles for all 11 years of data, you will need to run the ```srun``` command for each year from 2001-2011. Running this will result in the same directory structure as shown previously. Within each year directory (for ex: 2001), you will have 12 netcdf files, one for each month of the year. 


### Step 1.3: Run combine-monthly-tiles.py
The next step after Step 1.2 is to merge the monthly netcdf files into one annual netcdf file. To do this, you need to run the following:

```
srun -n{NUMBER OF PROCESSES PER NODE} -N1 python3 combine-monthly-tiles.py {YEAR} &
```

After step 1.3, for each tile, there will be 1 file that consists of all of the time frames per year.


## Step 2: deep learning
You must complete the entirety of Step 1 in order for Step 2 to work. The entirety of this section takes place within the ```deep-learning``` directory.

### Step 2.1: Run preprocessing.py
This step takes the tiles, and creates datasets stored as npy files. You must first fill the json parameters within ```preprocessing.json```. Then you can run the preprocessing process via the following:

```
python3 preprocessing.py
```

Currently, there is no parallel implementation for this step. However, one alternative is to run this script several times, and instead of random tile generation within each script, have a set list of tileIDs for each script.

### Step 2.2: Run main.py
After Step 2.1, you need to fill the json parameters within ```main.json```.  Then you can run ```main.py``` via the following:

```
CUDA_VISIBLE_DEVICES={GPU_IDS} python3 main.py 1> out.txt 2> err.txt &
```

```GPU_IDS``` is a list of numbers. For example if there are 8 GPUs available, the IDs will be 0,1,2,3,4,5,6,7. If you want to use 4 GPUs, fill ```GPU_IDS``` with ```0,1,2,3```, where you are using the GPUs with IDs 0,1,2, and 3.
This will run the training process based on the datasets created in Step 2.1. Any errors will show in ```err.txt``` and the general output from the script will show in ```out.txt```. When running this, be careful of your batch size, as you can easily overload memory.

### Step 2.3: Run predictions.py
After Step 2.2, you will have a saved model. If you want to run predictions on the model, first fill the json parameters within ```predictions.json```. Then, you can run ```predictions.py``` via the following:

```
CUDA_VISIBLE_DEVICES={GPU_IDS} python3 predictions.py 1> out.txt 2> err.txt &
```

## Acknowledgements

The code was developed by Gerson Kroiz, mentored by Valentine Anantharaj. Special thanks to Junqi Yin and Aristeidia Tsaris for their technical guidance. This research used resources of the Oak Ridge Leadership Computing Facility, which is a DOE Office of Science User Facility, and the resources of the Compute and Data Environment for Science (CADES) at the Oak Ridge National Laboratory, both supported by the Office of Science of the U.S. Department of Energy under Contract DE-AC05-00OR22725. For any questions or concerns, please contact gkroiz1@umbc.edu