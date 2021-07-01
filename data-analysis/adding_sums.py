import numpy as np
import xarray as xr
import time
import os, os.path
import matplotlib.pyplot as plt
import sys

if __name__ == "__main__":
    
    SUMSDIR = '/autofs/nccs-svm1_home1/gkroiz1/weatherDNN/data-analysis/sums'
    sums = np.zeros(160)
    for filename in os.listdir(SUMSDIR):
        #open file
        print(filename)
        data = np.load(f'{SUMSDIR}/{filename}')
        sums += data
    total_num_files = 99942+103600+104057+104208+103804+99386+104821+104801+104916+104653+104963
    
    #for bin size of 5
    sums_bin_5 = np.zeros(32)
    count = 0
    index = 0
    for i in range(len(sums)):
        sums_bin_5[index] += sums[i]
        count += 1
        if count == 5:
            count = 0
            index += 1



    os.chdir('/autofs/nccs-svm1_home1/gkroiz1/weatherDNN/data-analysis/figures/line_hist_total')
    #for bin size 1
    # plt.plot(sums/(1024*1024*total_num_files))

    #for bin size 5
    plt.plot(sums_bin_5/(1024*1024*total_num_files))

    plt.ylim([10**(-10), 10**0])
    plt.yscale('log')
    plt.ylabel('PDF')
    plt.xlabel('Precipitation Rate (mm/hr)')
    plt.title('Precipitation Line Histogram for 11 year')
    plt.locator_params(axis='y', numticks=20)

    #for bin size 1
    # plt.savefig('line_hist_total_bin_1.pdf')

    #for bin size 5
    plt.savefig('line_hist_total_bin_5.pdf')