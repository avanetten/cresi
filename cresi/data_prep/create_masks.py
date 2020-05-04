#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 16:56:31 2017

@author: avanetten

adapted from # https://github.com/CosmiQ/apls/tree/master/src
"""

import os
import sys
import json
import argparse
import skimage.io
import numpy as np
import shutil
# from config import Config

# add path and import apls_tools
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
import apls_tools, AddGeoReferencing, save_array_gdal
# from utils import apls_tools, AddGeoReferencing


###############################################################################
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--im_dir', default='', type=str,
        help='Image directory')
    parser.add_argument('--labels_dir', default='', type=str,
        help='Edges directory (geojsons)')
    parser.add_argument('--out_dir_mask', default='', type=str,
        help='Output folder for masks')
    parser.add_argument('--out_dir_comb', default='', type=str,
        help='Output folder for masks appended to image')
    parser.add_argument('--im_prefix', default='', type=str,
        help='prefix to match geojson with tif')
    parser.add_argument('--geojson_prefix', default='osmroads_', type=str,
        help='prefix to match geojson with tif')
    parser.add_argument('--buffer_meters', default=2, type=float,
        help='Road buffer')
    parser.add_argument('--burnValue', default=255, type=int,
        help='mask burn value')
    
    args = parser.parse_args()
    print ("args:", args )
    
    verbose = False #
    path_images = args.im_dir
    path_labels = args.labels_dir
    path_masks = args.out_dir_mask
    path_comb = args.out_dir_comb
    buffer_meters = args.buffer_meters
    geojson_prefix = args.geojson_prefix
    burnValue = args.burnValue 
    buffer_meters_str = str(np.round(buffer_meters, 1)).replace('.', 'p')

    # make dirs
    for d in [path_masks]:
        # print("Cleaning and remaking:", d)
        # shutil.rmtree(d, ignore_errors=True)
        print("making:", d)
        os.makedirs(d, exist_ok=True)
    if len(path_comb) > 0:
        os.makedirs(path_comb, exist_ok=True)

    # iterate through images and create masks
    im_files = sorted(os.listdir(path_images))
    # m = defaultdict(list)
    for i, im_file in enumerate(im_files):
        if not im_file.endswith('.tif'):
            continue

        name_root_full = im_file.split('.')[0]
        # name_root_full = im_file.split(imfile_prefix)[-1].split('.')[0]

        im_path = os.path.join(path_images, im_file)

        # determine mask output files
        label_name = geojson_prefix + name_root_full + '.geojson'
        label_file_tot = os.path.join(path_labels, label_name)
        output_raster = os.path.join(path_masks, im_file)
        # print("\n")
        print(i, "/", len(im_files), "name_root:", name_root_full)
        print("  im_file:", im_path)
        print("  label_file:", label_file_tot)
        print("  output_mask_raster:", output_raster)

        # create masks
        mask, gdf_buffer = apls_tools.get_road_buffer(
            label_file_tot, im_path,
            output_raster,
            buffer_meters=buffer_meters,
            burnValue=burnValue,
            bufferRoundness=6,
            plot_file=None,
            figsize=(6, 6),  # (13,4),
            fontsize=8,
            dpi=200, show_plot=False,
            verbose=False)
    
        # append mask to image?
        if len(path_comb) > 0:
            output_raster_comb = os.path.join(path_comb, im_file)
            im = skimage.io.imread(im_path)
            im_comb = np.dstack((im, mask)).astype(np.uint8)
            if verbose:
                print("    im.shape:", im.shape)
                print("    im", im)
                print("    mask.shape:", mask.shape)
                print("    mask", mask)
                print("    im_comb.shape:", im_comb.shape)
                print("    im_comb", im_comb)
            # move channels to front
            #print("im_comb.shape (init):", im_comb.shape)
            im_comb = np.moveaxis(im_comb, -1, 0)
            if verbose:
                print("    im_comb.shape (final):", im_comb.shape)
            # save
            save_array_gdal.CreateMultiBandGeoTiff(output_raster_comb,
                                                   im_comb)
           # skimage.io.imsave(output_raster_comb, im_comb)

    # add geo referencing
    AddGeoReferencing.geo_that_raster(path_masks, path_images)
    if len(path_comb) > 0:  
        AddGeoReferencing.geo_that_raster(path_comb, path_images)

#    ### USE CONFIG ###
#    parser = argparse.ArgumentParser()
#    parser.add_argument('config_path')
#    parser.add_argument('--training', action='store_true')
#    args = parser.parse_args()
#
#    label_path_extra = 'trainsat_edges'
#
#    # get config
#    with open(args.config_path, 'r') as f:
#        cfg = json.load(f)
#    config = Config(**cfg)
#
#    # set values
##    if config.num_channels == 3:
##        image_format_path = 'RGB-PanSharpen'
##    else:
##        image_format_path = 'MUL-PanSharpen'
##    imfile_prefix = image_format_path + '_'
#    imfile_prefix = ''  # image_format_path + '_'
#
#    label_path_extra = 'trainsat_edges'
#    geojson_prefix = 'osmroads_'
#    # geojson_prefix = 'spacenetroads_'
#    burnValue = 255
#
#    buffer_meters = float(config.mask_width_m)
#    buffer_meters_str = str(np.round(buffer_meters, 1)).replace('.', 'p')
#    test = not args.training
#
#    paths_data_raw = []
#    #############
#    # output directories
#
#    # put all training images in one directory so training can find em all
#    if not test:
#        path_masks =       os.path.join(config.path_data_root, config.train_data_refined_dir, 'masks{}m'.format(buffer_meters_str))
#        path_images_8bit = os.path.join(config.path_data_root, config.train_data_refined_dir, 'images')
#        # make dirs
#        for d in [path_masks, path_images_8bit]:
#            print("cleaning and remaking:", d)
#            shutil.rmtree(d, ignore_errors=True)
#            os.makedirs(d, exist_ok=True)
#
#        # set path_data_raw
#        for dpart in config.data_train_raw_parts.split(','):
#            paths_data_raw.append(os.path.join(config.path_data_root, dpart))
#
#    else:
#        path_masks =       os.path.join(config.path_data_root, config.test_data_refined_dir, 'masks{}m'.format(buffer_meters_str))
#        path_images_8bit = os.path.join(config.path_data_root, config.test_data_refined_dir)
#        # make dirs
#        for d in [path_images_8bit]:
#            print("Cleaning and remaking:", d)
#            shutil.rmtree(d, ignore_errors=True)
#            os.makedirs(d, exist_ok=True)
#        # set path_data_raw
#        for dpart in config.data_test_raw_parts.split(','):
#            paths_data_raw.append(os.path.join(config.path_data_root, dpart))
#           
#    # make dirs
#    for d in [path_masks]:
#        print("Cleaning and remaking:", d)
#        shutil.rmtree(d, ignore_errors=True)
#        os.makedirs(d, exist_ok=True)
#
#    # iterate through dirs
#    for path_data in paths_data_raw:
#
#        print("path_data:", path_data)
#        path_data = path_data.strip().rstrip('/')
#        path_labels = os.path.join(path_data, label_path_extra)
#
#        # iterate through images and create masks
#        im_files = os.listdir(path_images_8bit)
#        # m = defaultdict(list)
#        for im_file in im_files:
#            if not im_file.endswith('.tif'):
#                continue
#
#            name_root_full = im_file.split(imfile_prefix)[-1].split('.')[0]
#
#            im_file = os.path.join(path_images_8bit, im_file)
#
#            # determine mask output files
#            label_name = geojson_prefix + name_root_full + '.geojson'
#            label_file_tot = os.path.join(path_labels, label_name)
#            output_raster = os.path.join(path_masks, im_file)
#            print("\nname_root:", name_root_full)
#            print("  output_mask_raster:", output_raster)
#
#            # create masks
#            mask, gdf_buffer = apls_tools.get_road_buffer(
#                label_file_tot, im_file,
#                output_raster,
#                buffer_meters=buffer_meters,
#                burnValue=burnValue,
#                bufferRoundness=6,
#                plot_file=None,
#                figsize=(6, 6),  # (13,4),
#                fontsize=8,
#                dpi=200, show_plot=False,
#                verbose=False)


###############################################################################
if __name__ == "__main__":
    main()
