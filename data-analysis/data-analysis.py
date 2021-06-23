import numpy as np
import xarray as xr
import time
import os, os.path
import matplotlib.pyplot as plt

if __name__ == "__main__":
    
    #local
    DATADIR = '/Users/gersonkroiz/research/SULI_2021/weatherDNN/netcdf/2004'

    #on andes
    # DATADIR = '/gpfs/alpine/cli900/world-shared/data/collections/mrms/subset_1024x1024/netcdf'


    #rain intensities based on https://www.baranidesign.com/faq-articles/2020/1/19/rain-rate-intensity-classification
    start = time.time()

    num_files = len([name for name in os.listdir(DATADIR) if os.path.isfile(os.path.join(DATADIR, name))])

    arr_num_zeroes = np.empty(num_files)
    arr_num_light_rain = np.empty(num_files)
    arr_num_moderate_rain = np.empty(num_files)
    arr_num_heavy_rain = np.empty(num_files)
    arr_num_violent_rain = np.empty(num_files)

    counter = 0
    #iterate through each file in the directory
    for filename in os.listdir(DATADIR):
        
        #open file
        data = xr.open_dataset(f'{DATADIR}/{filename}')

        #calculate values
        num_zeroes = 1024*1024 - np.count_nonzero(data.PrecipRate_surface.data)
        nonzeroes = data.PrecipRate_surface.data[data.PrecipRate_surface.data != 0]
        light_rain = nonzeroes[nonzeroes < 2.5]
        moderate_rain = nonzeroes[nonzeroes > 2.5]
        moderate_rain = moderate_rain[moderate_rain < 7.5]
        heavy_rain = nonzeroes[nonzeroes > 7.5]
        heavy_rain = heavy_rain[heavy_rain < 50]
        violent_rain = nonzeroes[nonzeroes > 50]

        #store values in arrays
        arr_num_zeroes[counter] = num_zeroes
        arr_num_light_rain[counter] = light_rain.size
        arr_num_moderate_rain[counter] = moderate_rain.size
        arr_num_heavy_rain[counter] = heavy_rain.size
        arr_num_violent_rain[counter] = violent_rain.size

        counter += 1
    end = time.time()
    
    print('time elapsed = ' + str(end - start))
    
    
    #make plots
    if not os.path.isdir('figures'):
        os.mkdir('figures')
    os.chdir('figures')
    #1024x1024 is dim of each netcdf file
    plot_data = [arr_num_zeroes/(1024*1024), arr_num_light_rain/(1024*1024), arr_num_moderate_rain/(1024*1024), arr_num_heavy_rain/(1024*1024), arr_num_violent_rain/(1024*1024)]
    titles = ['No Rain', 'Light Rain', 'Moderate Rain', 'Heavy Rain', 'Violent Rain']
    savefig_names = ['no_rain.pdf', 'light_rain.pdf', 'moderate_rain.pdf', 'heavy_rain.pdf', 'violent_rain.pdf']
    for i in range(5):
        plt.figure(i)
        plt.boxplot(plot_data[i],  0, '+')
        plt.ylabel('Percent of pixels')
        plt.title(titles[i])
        plt.savefig(savefig_names[i])
