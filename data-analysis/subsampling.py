import numpy as np
import xarray as xr
import time
import os, os.path
from mpi4py import MPI
import sys

# import 


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    print('rank: ' + str(rank) + ' in file')

    year = sys.argv[1]
    # print(size)
    #ON ANDES
    DATADIR = '/gpfs/alpine/cli900/world-shared/data/collections/mrms/subset_1024x1024/netcdf/' + str(year)
    # DATADIR = '/autofs/nccs-svm1_home1/gkroiz1/netcdf/2004'

    #ON LOCAL
    # DATADIR = '/Users/gersonkroiz/research/SULI_2021/weatherDNN/netcdf/2004'

    #find number of files in data directory (should be 5 * 12 * 24)
    num_files = len([name for name in os.listdir(DATADIR) if os.path.isfile(os.path.join(DATADIR, name))])
    file_name = 'MRMS_PrecipRate_00.00_20040613-000000.nc'


    #make directory fore tiles

    #ON LOCAL
    # TILESDIR = '/Users/gersonkroiz/research/SULI_2021/weatherDNN/tiles'

    #ON ANDES
    # TILESDIR = '/autofs/nccs-svm1_home1/gkroiz1/netcdf/tiles'
    TILESDIR = '/gpfs/alpine/cli900/world-shared/users/gkroiz1/tiles'
    if rank == 0:
        if not os.path.isdir(TILESDIR):
            os.mkdir(TILESDIR)
            os.chdir(TILESDIR)
            for tile in range(16*16):
                os.mkdir('tile'+str(tile))
    
    comm.Barrier()

    print('rank: ' + str(rank) + ' after comm barrier')
    overallStart = time.time()
    #goes through each file
    for filename in os.listdir(DATADIR):
        start = time.time()

        data = xr.open_dataset(f'{DATADIR}/{filename}')

        #goes through each latitude coord
        # print('int(16/size): ' + str(int(16/size)))
        for lat in range(int(16/size)):

            #goes through each longitude coord
            for long in range(16):
                mpi_lat = lat + int(rank)*int(16/size)
                # print(mpi_lat)
                #make a subdataset of the lat, long, and prec of the tile
                tile = xr.Dataset(
                    data_vars=dict(
                        PrecipRate_surface=data.PrecipRate_surface[0, 1023-mpi_lat*64-64:1023-mpi_lat*64, long*64:long*64+64]
                    ),
                    coords = dict(
                       time=data.time
                    ),
                    attrs = dict(Conventions='COARDS', History='created by wgrib2', GRIB2_grid_template = '0')
                    )
    #             print('tile: ' + str(lat * 16 + long) + ', lat: ' + str(data.latitude.data[lat * 64]) + ', long: ' + str(data.longitude.data[long*64]))

                #save dataset
                os.chdir(TILESDIR + '/tile' + str(mpi_lat * 16 + long))
                tile.to_netcdf(f't-' + str(mpi_lat * 16 + long) + '-' + str(filename), mode='w', format='netcdf4')

        end = time.time()
        # print('rank: ' + str(rank) + ', time elapsed = ' + str(end - start))

    overallEnd = time.time()
    print('year: ' + str(year) + ', rank: ' + str(rank) + ', FINAL TIME = ' + str(overallEnd - overallStart))

