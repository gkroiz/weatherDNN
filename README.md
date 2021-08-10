# PrecipRate Benchmark
PrecipRate Benchmark was designed for the SULI program at ORNL (2021). Conceptually, this dataset provides a benchmark dataset of precipitaion rates. Additionally, there is a baseline model that is provided. These two features are included to facilitate comparision between different deep learning architectures. The data is from the Multi-RADAR/Multi-sensor System (MRMS). A more in-depth description is provided in the technical report ```SULI_Report.pdf```.

To use this benchmark dataset, you must install the conda environment with the dependencies below. Using different versions of the dependencies may result in errors. After install the conda dependencies, you can follow the steps to create tiles and use the baseline model. Please note that some of these steps will take multiple hours, depending on you computational setup. This is NOT recommended for a personal laptop. Multiple-GPUs strongly suggested.


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
2) ```baseline-model```

The ```tile-generation``` directory contains files needed to create the tiles. The ```baseline-model``` directory contains the baseline model, which is a convolutional LSTM (convLSTM).

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


The code used within Step 1 is designed to run with slurm. This is to parallelize the process. However, if you do not use slurm, each file within ```tile-generation``` provides some insight on what to change (namely, the ```rank``` and ```size``` variables).

### Step 1.1: Fill tile-generation.json
Before Step 1.2, fill the json parameters for ```tile-generation.json```. Here is a description of the file:

* ```data_dir_loc``` - This is the location of where the entire dataset is (should have tree structure as shown above). DO NOT END WITH '/'.
* ```tiles_dir_loc``` - This is the location of where to save the individual tiles. DO NOT END WITH '/'.
* ```combined_tiles_dir_loc``` - This is the location of where to save the combined tiles (both the monthly and annual combinations). DO NOT END WITH '/'.
* ```tile_dim``` - This is the dimension size of a single tile. Suggested value is 64.
* ```entire_dataset_dim``` - This is the dimension size of the entire dataset. If using the same dataset from the report, use value 1024.


### Step 1.2: Run create-tiles.py
After Step 1.1, we want to create the tiles netcdf files for each time frame for the 1024x1024 region. To create the individual tiles, run the following:

```
srun -n{NUMBER OF PROCESSES PER NODE} -N1 python3 create-tiles.py {YEAR} &
```

```NUMBER OF PROCESSES PER NODE``` will determine the rank within ```create-tiles.py```

```YEAR``` will determine what year you are creating tiles for. This is a required command line argument for 

To create tiles for all 11 years of data, you will need to run the ```srun``` command for each year from 2001-2011. Running this will create the tree directory structure shown above within ```tiles_dir_loc```. Each year directory (ex: 2001) will include folders for each tile, where within the tile folders is the netcdf data.

### Step 1.3: Run combine-single-tiles.py
Once you have created the tiles in Step 1.2, you will need to combine the individual time frames into monthly netcdf files for each tile. For this step, run the following:

```
srun -n{NUMBER OF PROCESSES PER NODE} -N1 python3 combine-single-tiles.py {YEAR} &
```

To create tiles for all 11 years of data, you will need to run the ```srun``` command for each year from 2001-2011. Running this will result in the tree directory structure shown above within ```combined_tiles_dir_loc```. Each year directory (ex: 2001) will include folders for each tile. Each of these tiles folders will have 12 netcdf files, one for each month of the year. 


### Step 1.4: Run combine-monthly-tiles.py
The next step after Step 1.3 is to merge the monthly netcdf files into one annual netcdf file. To do this, you need to run the following:

```
srun -n{NUMBER OF PROCESSES PER NODE} -N1 python3 combine-monthly-tiles.py {YEAR} &
```

To create tiles for all 11 years of data, you will need to run the ```srun``` command for each year from 2001-2011. The ```combined_tiles_dir_loc``` directory structure will be the same as before. The only difference is that within each tile folder, in addition to the 12 netcdf files for each month, there will be another netcdf file for the entire year.

## Step 2: deep learning
You must complete the entirety of Step 1 in order for Step 2 to work. The entirety of this section takes place within the ```baseline-model``` directory.

### Step 2.1: Fill baseline-model.json
Before Step 2.2, fill the json parameters for ```baseline-model.json```. This will be used by ```preprocessing.py```, ```main.py```, and ```predictions.py```. Here is a description of the file:

* ```train_years``` - array of years in string format (ex: ["2001", "2002", "2003"]). The training dataset will be created based on these years.
* ```val_years``` - array of years in string format. The validation dataset will be created based on these years.
* ```test_years``` - array of years in string format. The testing dataset will be created based on these years.
* ```tiles_dir_loc``` - location of where to read tiles. This should be the same as ```combined_tiles_dir_loc``` from ```tile-generation.json```. DO NOT END WITH '/'.
* ```train_loc``` - location of where training data should be stored. DO NOT END WITH '/'.
* ```val_loc``` - location of where validation data should be stored. DO NOT END WITH '/'.
* ```test_loc``` - location of where testing data should be stored. DO NOT END WITH '/'.
* ```meta_data_loc``` - location of where the meta_data is stored. Must end as '.csv'
* ```saved_model_loc```: - where to save model. Must end as '.h5'
* ```lead_time_x``` - this is the amount of consecutive time (in minutes) for x_data (input to model).
* ```lead_time_y``` - this is the amount of consecutive time (in minutes) for y_data (output to model).
* ```num_tiles``` - number of tiles to include in the datasets. Note that the train, val, and test datasets do not care for tileID. As such, increasing ```num_tiles``` will increase the number of samples within each dataset.


### Step 2.2: Run preprocessing.py
This step takes the annual combined tiles and creates datasets stored as npy files based on locations specified in Step 2.1. This preprocessing step is executable via the following:

```
python3 preprocessing.py
```

Currently, there is no parallel implementation for this step. However, one alternative is to run this script several times, and instead of random tile generation within each script, have a set list of tileIDs for each script.

### Step 2.4: Run main.py
This step will run the training process of the model. This step is executable via the following: 

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