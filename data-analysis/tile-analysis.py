import numpy as np
import xarray as xr
import time
import os, os.path
import matplotlib.pyplot as plt
import sys

if __name__ == "__main__":
    rank = int(os.environ["SLURM_PROCID"])
    size = int(os.environ["SLURM_NPROCS"])

    year = sys.argv[1]

    # print('in file for year ' + str(year))

    start = time.time()

    counter = 0

    #count maximum precipitation amount:
    max_precip = 0

    #array for histogram:
    sums = np.zeros(160)
    num_files_counter = np.zeros(1)

    #ON ANDES
    # TILESDIR = '/autofs/nccs-svm1_home1/gkroiz1/netcdf/tiles'
    TILESDIR = '/gpfs/alpine/cli900/world-shared/users/gkroiz1/tile-sums/' + str(year)
    if rank == 0:
        # print("Here, rank: " + str(rank))
        if not os.path.isdir(TILESDIR):
            os.mkdir(TILESDIR)
            os.chdir(TILESDIR)
            for tile in range(16*16):
                os.mkdir('tile'+str(tile))
    
    # comm.Barrier()

    time.sleep(60)

    for tile in range(int(256/size)):
        tileID = int(tile + rank * size)
        print('rank: ' + str(rank) + ', tileID: ' + str(tileID))
        #on andes
        #full
        DATADIR = '/gpfs/alpine/cli900/world-shared/users/gkroiz1/tiles/' + str(year) + '/tile' + str(tileID)
        
        #sample
        # DATADIR = '/autofs/nccs-svm1_home1/gkroiz1/netcdf/limited_tiles/tile' + str(tileID)
        #iterate through each file in the directory
        num_files = len([name for name in os.listdir(DATADIR) if os.path.isfile(os.path.join(DATADIR, name))])
        num_files_counter[0] = num_files

        for filename in os.listdir(DATADIR):
            # print('file: ' + str(counter) + ', year: ' + year)
            #open file
            data = xr.open_dataset(f'{DATADIR}/{filename}')

            #calculate values

            data.PrecipRate_surface.data[data.PrecipRate_surface.data > 160] = 160
            sums += np.histogram(data.PrecipRate_surface.data, 160, (0,160))[0]
        np.save(f'{TILESDIR}/tile'+str(tileID)+'/sums.npy', sums)
        np.save(f'{TILESDIR}/tile'+str(tileID)+'/num_files.npy', num_files_counter)

            # counter += 1

    end = time.time()
    
    print('time elapsed = ' + str(end - start))

