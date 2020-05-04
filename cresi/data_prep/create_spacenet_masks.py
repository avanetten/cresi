#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 16:56:31 2017

@author: avanetten

adapted from # https://github.com/CosmiQ/apls/tree/master/src
"""

import os
import cv2
import json
import argparse
from collections import defaultdict
from osgeo import gdal
import numpy as np
import shutil
from config import Config

# add path and import apls_tools
from other_tools import apls_tools

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
def gamma_correction(image, gamma=1.66):
    '''https://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/'''
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
 
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path')
    parser.add_argument('--training', action='store_true')
    #parser.add_argument('datasets', type=str, nargs='+')
    #parser.add_argument('mode', type=str, default='test', help='test or train')
    #parser.add_argument('--buffer_meters', type=float, default=2, help='road width (m)')
    args = parser.parse_args()
    #buffer_meters = args.buffer_meters

    # get config
    with open(args.config_path, 'r') as f:
        cfg = json.load(f)
    config = Config(**cfg)


    # set values
    if config.num_channels == 3:
        image_format_path = 'RGB-PanSharpen'
    else:
        image_format_path = 'MUL-PanSharpen'
        
    imfile_prefix = image_format_path + '_'
    label_path_extra = 'geojson/spacenetroads'
    geojson_prefix = 'spacenetroads_'    
    burnValue = 255

    buffer_meters = float(config.mask_width_m)
    buffer_meters_str = str(np.round(buffer_meters,1)).replace('.', 'p')
    test = not args.training

    paths_data_raw = []
    #############
    # output directories
    
    # put all training images in one directory so training can find em all   
    if not test:
        path_masks =       os.path.join(config.path_data_root, config.train_data_refined_dir, 'masks{}m'.format(buffer_meters_str))
        path_images_8bit = os.path.join(config.path_data_root, config.train_data_refined_dir, 'images')
        # make dirs
        for d in [path_masks, path_images_8bit]:
            print ("cleaning and remaking:", d)
            shutil.rmtree(d, ignore_errors=True)
            os.makedirs(d, exist_ok=True)

        # set path_data_raw
        for dpart in config.data_train_raw_parts.split(','):
            paths_data_raw.append(os.path.join(config.path_data_root, dpart))
            
    else:
        path_masks =       os.path.join(config.path_data_root, config.test_data_refined_dir, 'masks{}m'.format(buffer_meters_str))
        path_images_8bit = os.path.join(config.path_data_root, config.test_data_refined_dir)
        # make dirs
        for d in [path_images_8bit]:
            print ("cleaning and remaking:", d)
            shutil.rmtree(d, ignore_errors=True)
            os.makedirs(d, exist_ok=True)
        # set path_data_raw
        for dpart in config.data_test_raw_parts.split(','):
            paths_data_raw.append(os.path.join(config.path_data_root, dpart))

    #path_masks =       os.path.join(config.path_data_root, config.data_refined_name+'_train' if not test else config.data_refined_name+'_test', 'masks{}m'.format(buffer_meters))
    #path_images_8bit = os.path.join(config.path_data_root, config.data_refined_name+'_train' if not test else config.data_refined_name+'_test', 'images')

    # make dirs
    for d in [path_masks, path_images_8bit]:
        print ("cleaning and remaking:", d)
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d, exist_ok=True)

    # iterate through dirs
    for path_data in paths_data_raw:
        
        path_data = path_data.strip().rstrip('/')
        # get test_data_name for rescaling (if needed)
        #test_data_name = os.path.split(path_data)[-1]
        #test_data_name = '_'.join(test_data_name.split('_')[:3]) + '_'
        path_images_raw = os.path.join(path_data, image_format_path)
        path_labels = os.path.join(path_data, label_path_extra)

        # iterate through images, convert to 8-bit, and create masks
        im_files = os.listdir(path_images_raw)
        #m = defaultdict(list)
        for im_file in im_files:
            if not im_file.endswith('.tif'):
                continue

            #name_root_small = im_file.split('_')[-1].split('.')[0]
            name_root_full = im_file.split(imfile_prefix)[-1].split('.')[0]

            # create 8-bit image
            im_file_raw = os.path.join(path_images_raw, im_file)
            im_file_out = os.path.join(path_images_8bit, im_file)
            #im_file_out = os.path.join(path_images_8bit, test_data_name + name_root + '.tif')
            # convert to 8bit
            # m = calc_rescale(im_file_raw, m, percentiles=[2,98])
            # continue
            
            ####################
            # SET RESCALE TYPE
            #rescale_type = test_data_name.split('_')[1]
            if config.num_channels == 3:
                rescale_type = 'tot_3band' #test_data_name.split('_')[1]
            else:
                rescale_type = 'tot_8band' #test_data_name.split('_')[1]
            ####################

            if not os.path.isfile(im_file_out):
                apls_tools.convert_to_8Bit(im_file_raw, im_file_out,
                                           outputPixType='Byte',
                                           outputFormat='GTiff',
                                           rescale_type=rescale[rescale_type])

            if test:
                continue
            
            else:
                # determine mask output files
                label_name = geojson_prefix + name_root_full + '.geojson'
                label_file_tot = os.path.join(path_labels, label_name)
                output_raster = os.path.join(path_masks, im_file)
                print("\nname_root:", name_root_full)
                print("  output_mask_raster:", output_raster)
                
                # create masks
                mask, gdf_buffer = apls_tools.get_road_buffer(label_file_tot, im_file_out,
                                                              output_raster,
                                                              buffer_meters=buffer_meters,
                                                              burnValue=burnValue,
                                                              bufferRoundness=6,
                                                              plot_file=None,
                                                              figsize= (6,6),  #(13,4),
                                                              fontsize=8,
                                                              dpi=200, show_plot=False,
                                                              verbose=False)

        #for k, v in m.items():
        #    print(test_data_name, k, np.mean(v, axis=0))


###############################################################################
if __name__ == "__main__":
    main()