import numpy as np
import xarray as xr
import time
import os, os.path

# import 


if __name__ == "__main__":
    #ON ANDES
    DATADIR = '/autofs/nccs-svm1_home1/gkroiz1/netcdf/2004'

    #ON LOCAL
    DATADIR = '/Users/gersonkroiz/research/SULI_2021/weatherDNN/netcdf/2004'

    #find number of files in data directory (should be 5 * 12 * 24)
    num_files = len([name for name in os.listdir(DATADIR) if os.path.isfile(os.path.join(DATADIR, name))])
    file_name = 'MRMS_PrecipRate_00.00_20040613-000000.nc'

    year = '2004'
    month = '06'
    day = '13'

    #make directory fore tiles
    TILESDIR = '/Users/gersonkroiz/research/SULI_2021/weatherDNN/tiles'
    if not os.path.isdir(TILESDIR):
        os.mkdir(TILESDIR)
        os.chdir(TILESDIR)
        for tile in range(16*16):
            os.mkdir('tile'+str(tile))

    start = time.time()
    #goes through each file
    for file in range(1):
        data = xr.open_dataset(f'{DATADIR}/MRMS_PrecipRate_00.00_' + year + month + day + '-000000.nc')

        #goes through each latitude coord
        for lat in range(16):

            #goes through each longitude coord
            for long in range(16):

                #make a subdataset of the lat, long, and prec of the tile
                tile = xr.Dataset(
                    data_vars=dict(
                        PrecipRate_surface=data.PrecipRate_surface[0, 1023-lat*64-63:1023-lat*64, long*64:long*64+63]
                    ),
                    coords = dict(
                       time=data2.time
                    ),
                    attrs = dict(Conventions='COARDS', History='created by wgrib2', GRIB2_grid_template = '0')
                    )
    #             print('tile: ' + str(lat * 16 + long) + ', lat: ' + str(data.latitude.data[lat * 64]) + ', long: ' + str(data.longitude.data[long*64]))

                #save dataset
                os.chdir(TILESDIR + '/tile' + str(lat * 16 + long))
                tile.to_netcdf(f'MRMS_PrecipRate_00.00_'+year+month+day+'-000000-'+ str(lat * 16 + long) + '.nc', mode='w', format='netcdf4')

    end = time.time()

    print('time elapsed = ' + str(end - start))