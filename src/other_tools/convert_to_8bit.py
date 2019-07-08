#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 13:51:43 2018

@author: avanetten

scp -r /Users/avanetten/Documents/cosmiq/basiss/albu_inference_mod_new/src 10.123.1.70:/raid/local/src/basiss/albu_inference_mod_new

"""

import os
import json
import argparse
from collections import defaultdict
from osgeo import gdal
import numpy as np
import shutil
from config import Config

# add path and import apls_tools
from other_tools import apls_tools



###############################################################################
# numbers retrieved from all_dems_min_max.py, and give min, max value for 
# each band over the entire city (e.g.: 2' corresponds to Vegas)
# 'tot' is mean of each value across cities
rescale = {
    '2': {
        1: [25.48938322, 1468.79676441],
        2: [145.74823054, 1804.91911021],
        3: [155.47927199, 1239.49848332]
    },
    '4': {
        1: [79.29799666, 978.35058431],
        2: [196.66026711, 1143.74207012],
        3: [170.72954925, 822.32387312]
    },
    '3': {
        1: [46.26129032, 1088.43225806],
        2: [127.54516129, 1002.20322581],
        3: [141.64516129, 681.90967742]
    },
    '5': {
        1: [101.63250883, 1178.05300353],
        2: [165.38869258, 1190.5229682 ],
        3: [126.5335689, 776.70671378]
    },
    'tot_3band': {
        1: [63,  1178],
        2: [158, 1285],
        3: [148, 880]
    },
    # RGB corresponds to bands: 5, 3, 2
    'tot_8band': {
            1: [154, 669], 
            2: [122, 1061], 
            3: [119, 1520], 
            4: [62, 1497], 
            5: [20, 1342], 
            6: [36, 1505], 
            7: [17, 1853], 
            8: [7, 1559]}

}
    

###############################################################################
def calc_rescale(im_file_raw, m, percentiles):
    srcRaster = gdal.Open(im_file_raw)
    for band in range(1, 4):
        b = srcRaster.GetRasterBand(band)
        band_arr_tmp = b.ReadAsArray()
        bmin = np.percentile(band_arr_tmp.flatten(),
                             percentiles[0])
        bmax= np.percentile(band_arr_tmp.flatten(),
                            percentiles[1])
        m[band].append((bmin, bmax))

    # for k, v in m.items():
    #     print(k, np.mean(v, axis=0))
    return m


###############################################################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('im_file_raw')
    parser.add_argument('im_file_out')
    parser.add_argument('--rescale_type', type=str, default='tot_8band',
                        help="Can be tot_3band or tot_8band")
    args = parser.parse_args()


    apls_tools.convert_to_8Bit(args.im_file_raw, args.im_file_out,
                                           outputPixType='Byte',
                                           outputFormat='GTiff',
                                           rescale_type=rescale[args.rescale_type])
    return


###############################################################################
if __name__ == "__main__":
    main()