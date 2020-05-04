#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 09:03:15 2019

@author: avanetten
"""

import os
import cv2
import gdal
import math
import road_speed
import argparse
import numpy as np
import pandas as pd
import skimage.io
import warnings
# also see ipynb/_speed_data.prep.ipynb


###############################################################################
def gauss_blur_arr(in_arr, kernel_blur=7):
    '''Assume shape is (channels, h, w)'''

    n_channels, h, w = in_arr.shape
    out_arr = np.zeros(in_arr.shape)
    for band in range(n_channels):
        im_channel = in_arr[band, :, :]
        im_blur = cv2.GaussianBlur(im_channel, (kernel_blur, kernel_blur), 0)
        out_arr[band, :, :] = im_blur
        
    return out_arr
        
        
###############################################################################
def convert_array_to_multichannel(in_arr, n_channels=7, burnValue=255, 
                                  append_total_band=False, verbose=False):
    '''Take input array with multiple values, and make each value a unique
    channel.  Assume a zero value is background, while value of 1 is the 
    first channel, 2 the second channel, etc.'''
    
    h,w = in_arr.shape[:2]
    # scikit image wants it in this format by default
    out_arr = np.zeros((n_channels, h,w), dtype=np.uint8)
    #out_arr = np.zeros((h,w,n_channels), dtype=np.uint8)
    
    for band in range(n_channels):
        val = band + 1
        band_out = np.zeros((h, w), dtype=np.uint8)
        if verbose:
            print ("band:", band)
        band_arr_bool = np.where(in_arr == val)
        band_out[band_arr_bool] = burnValue
        out_arr[band,:,:] = band_out
        #out_arr[:,:,band] = band_out
 
    if append_total_band:
        tot_band = np.zeros((h,w), dtype=np.uint8)
        band_arr_bool = np.where(in_arr > 0)
        tot_band[band_arr_bool] = burnValue
        tot_band = tot_band.reshape(1,h,w)
        out_arr = np.concatenate((out_arr, tot_band), axis=0).astype(np.uint8)
    
    if verbose:
        print ("out_arr.shape:", out_arr.shape)
    return out_arr


###############################################################################
def CreateMultiBandGeoTiff(OutPath, Array):
    '''
    Author: Jake Shermeyer
    Array has shape:
        Channels, Y, X?
    '''
    driver = gdal.GetDriverByName('GTiff')
    DataSet = driver.Create(OutPath, Array.shape[2], Array.shape[1],
                            Array.shape[0], gdal.GDT_Byte,
                            ['COMPRESS=LZW'])
    for i, image in enumerate(Array, 1):
        DataSet.GetRasterBand(i).WriteArray(image)
    del DataSet

    return OutPath



###############################################################################
def speed_mask_dir(geojson_dir, image_dir, output_dir,
                             speed_to_burn_func,
                             mask_burn_val_key='burnValue',
                             buffer_distance_meters=2,
                             buffer_roundness=1,
                             dissolve_by='speed_m/s',
                             bin_conversion_key='speed_mph',
                             verbose=True,
                             # below here is all variables for binned speed
                             output_dir_multidim='',
                             channel_value_mult=1,
                             n_channels=8,
                             channel_burnValue=255,
                             append_total_band=True,
                             label_type='SN5',
                             crs=None,
                             ):
    """Create continuous speed masks for entire dir"""

    images = sorted([z for z in os.listdir(image_dir) if z.endswith('.tif')])
    for j, image_name in enumerate(images):

        image_root = image_name.split('.')[0]
        # image_root = image_name.split('RGB-PanSharpen_')[-1].split('.')[0]
        image_path = os.path.join(image_dir, image_name)

        mask_path_out = os.path.join(output_dir, image_name)

        # Get geojson path
        
        if label_type == 'SN3' or label_type == 'SN5': 
            # SpaceNet chips
            geojson_path = os.path.join(
                geojson_dir, image_root.replace('PS-RGB', 'geojson_roads_speed').replace('PS-MS', 'geojson_roads_speed')
                # geojson_dir, image_root.replace('PS-RGB', 'geojson_roads_speed')
                + '.geojson')
            # # Contiguous files
            # geojson_path = os.path.join(geojson_dir, image_root + '.geojson')
        elif label_type == 'SN4':
            # example im: Pan-Sharpen_Atlanta_nadir7_catid_1030010003D22F00_748451_3743589.tif
            # example json: spacenet-roads_748451_3743589_speed.geojson
            name_root = '_'.join(image_root.split('_')[-2:])
            geojson_path = os.path.join(geojson_dir,
                                        'spacenet-roads_' + name_root + '_speed.geojson')
        
        else:
            print("Uknown label type (expecting SN3, SN4 or SN5)', returning...")
            return
        
        # if (j % 100) == 0:
        if (j % 1) == 0:
            print(j+1, "/", len(images), "image:", image_name,
                  "geojson:", geojson_path)
        if j > 0:
            verbose = False

        gdf_buffer = road_speed.create_speed_gdf(
            image_path, geojson_path, mask_path_out, speed_to_burn_func,
            mask_burn_val_key=mask_burn_val_key,
            buffer_distance_meters=buffer_distance_meters,
            buffer_roundness=buffer_roundness,
            dissolve_by=dissolve_by,
            bin_conversion_key=bin_conversion_key,
            crs=crs,
            verbose=verbose)

        # If Binning...
        if output_dir_multidim:
            mask_path_out_md = os.path.join(output_dir_multidim, image_name)

            # Convert array to a multi-channel image
            mask_bins = skimage.io.imread(mask_path_out)
            mask_bins = (mask_bins / channel_value_mult).astype(int)
            if verbose:
                print("mask_bins.shape:", mask_bins.shape)
                print("np unique mask_bins:", np.unique(mask_bins))
                # print ("mask_bins:", mask_bins)
            # define mask_channels
            if np.max(mask_bins) == 0:
                h, w = skimage.io.imread(mask_path_out).shape[:2]
                # h, w = cv2.imread(mask_path_out, 0).shape[:2]
                if append_total_band:
                    mask_channels = np.zeros((n_channels+1, h, w)).astype(np.uint8)
                else:
                    mask_channels = np.zeros((n_channels, h, w)).astype(np.uint8)
    
            else:
                mask_channels = convert_array_to_multichannel(
                    mask_bins, n_channels=n_channels,
                    burnValue=channel_burnValue,
                    append_total_band=append_total_band,
                    verbose=verbose)
            if verbose:
                print("mask_channels.shape:", mask_channels.shape)
                print("mask_channels.dtype:", mask_channels.dtype)

            # write to file
            # skimage version...
            #skimage.io.imsave(mask_path_out_md, mask_channels, compress=1)  # , plugin='tifffile')
            # gdal version
            CreateMultiBandGeoTiff(mask_path_out_md, mask_channels)


###############################################################################
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--geojson_dir', default='', type=str,
                        help='location of geojson labels')
    parser.add_argument('--image_dir', default='', type=str,
                        help='location of geotiffs')
    parser.add_argument('--output_conversion_csv_contin', default='', type=str,
                        help='location of output conversion file')
    parser.add_argument('--output_mask_dir_contin', default='', type=str,
                        help='location of output masks')
    parser.add_argument('--output_conversion_csv_binned', default='', type=str,
                        help='location of output conversion file')
    parser.add_argument('--output_mask_multidim_dir', default='', type=str,
                        help='location of output masks for binned case '
                        ' set to '' to use continuous case')
    parser.add_argument('--buffer_distance_meters', default=2, type=float,
                        help='Mask buffer')
#    parser.add_argument('--output_conversion_csv_contin', default='', type=str,
#                        help='location of output conversion file')
#    parser.add_argument('--output_mask_dir_contin', default='', type=str,
#                        help='location of output masks')
#    parser.add_argument('--output_conversion_csv_binned', default='', type=str,
#                        help='location of output conversion file')
#    parser.add_argument('--output_mask_multidim_dir', default='', type=str,
#                        help='location of output masks for binned case '
#                        ' set to '' to use continuous case')
    parser.add_argument('--label_type', default='SN5', type=str,
                        help='SpaceNet data label formate (SN3, SN4, or SN5)')
    parser.add_argument('--crs', default='None', type=str,
                        help='crs of input data, use None to compute it '
                        'SN4 Atlanta is EPSG::32616')
    args = parser.parse_args()

    if args.crs == 'None':
        args.crs = None
        
    # ATLANTA
    # args.crs = {'init': 'epsg:32616'}
    
    # hardcoded for now...
    buffer_roundness = 1
    mask_burn_val_key = 'burnValue'
    dissolve_by = 'inferred_speed_mps'  # 'speed_m/s'
    bin_conversion_key = 'inferred_speed_mph'  # 'speed_mph'
    verbose = True
    # resave_pkl = False  # True

    # skimage throws an annoying "low contrast warning, so ignore"
    # ignore skimage warnings
    warnings.filterwarnings("ignore")

    # CREATE CONVERSION CSVS
    
    ###########################################################################
    # CONTINUOUS
    ###########################################################################
    if len(args.output_mask_multidim_dir) == 0:
                    
        min_road_burn_val = 0   # 80  #50  # 127
        min_speed_contin = 0  # 15   #15
        max_speed_contin = 65  # 65
        mask_max = 255
        verbose = True
        # placeholder variables for binned case
        channel_value_mult, n_channels, channel_burnValue, append_total_band \
            = 0, 0, 0, 0

        # make output dir
        os.makedirs(args.output_mask_dir_contin, exist_ok=True)

        #######################################################################
        def speed_to_burn_func(speed):
            '''Convert speed estimate to mask burn value between
            0 and mask_max'''
            bw = mask_max - min_road_burn_val
            burn_val = min(min_road_burn_val + bw * ((speed-min_speed_contin)/(max_speed_contin-min_speed_contin)), mask_max)
            return max(burn_val, min_road_burn_val)

        speed_arr_contin = np.arange(min_speed_contin, max_speed_contin + 1, 1)
        burn_val_arr = [speed_to_burn_func(s) for s in speed_arr_contin]
        d = {'burn_val': burn_val_arr, 'speed': speed_arr_contin}
        df_s = pd.DataFrame(d)

        # make conversion dataframe (optional)
        if not os.path.exists(args.output_conversion_csv_contin):
            print("Write burn_val -> speed conversion to:",
                  args.output_conversion_csv_contin)
            df_s.to_csv(args.output_conversion_csv_contin)
        else:
            print("path already exists, not overwriting...",
                  args.output_conversion_csv_contin)



    ###########################################################################
    # BINNED 10
    ###########################################################################
    else:
        
        min_speed_bin = 1
        max_speed_bin = 65
        channel_burnValue = 255
        channel_value_mult = 1
        append_total_band = True
        speed_arr_bin = np.arange(min_speed_bin, max_speed_bin + 1, 1)

        # make output dir
        if len(args.output_mask_dir_contin) > 0:
            os.makedirs(args.output_mask_dir_contin, exist_ok=True)
        if len(args.output_mask_multidim_dir) > 0:            
            os.makedirs(args.output_mask_multidim_dir, exist_ok=True)

        # # BINNED 10
        # #######################################################################
        # def speed_to_burn_func(speed_mph):
        #     '''bin every 10 mph or so
        #     Convert speed estimate to appropriate channel
        #     bin = 0 if speed = 0'''
        #     speed_bins = [10, 15, 18.75, 20, 25, 30, 35, 45, 55, 65]
        #     for i, speed_tmp in enumerate(speed_bins):
        #         if speed_mph <= speed_tmp:
        #             return int(255 * float(i + 1) / len(speed_bins))
        # # determine num_channels
        # n_channels = len(np.unique([int(speed_to_burn_func(z)) for z in speed_arr_bin]))
 
        # BINNED 7
        #######################################################################
        bin_size_mph = 10.0
        def speed_to_burn_func(speed_mph):
            '''bin every 10 mph or so
            Convert speed estimate to appropriate channel
            bin = 0 if speed = 0'''
            return int( int(math.ceil(speed_mph / bin_size_mph)) * channel_value_mult) 
        # determine num_channels
        n_channels = len(np.unique([int(speed_to_burn_func(z)) for z in speed_arr_bin]))
        # n_channels = int(speed_to_burn_func(max_speed_bin))
        
        print("n_channels:", n_channels)
        # update channel_value_mult
        channel_value_mult = int(255/n_channels)

        # make conversion dataframe
        print("speed_arr_bin:", speed_arr_bin)
        burn_val_arr = np.array([speed_to_burn_func(s) for s in speed_arr_bin])
        print("burn_val_arr:", burn_val_arr)
        d = {'burn_val': burn_val_arr, 'speed': speed_arr_bin}
        df_s_bin = pd.DataFrame(d)
        # add a couple columns, first the channel that the speed corresponds to
        channel_val = (burn_val_arr / channel_value_mult).astype(int) - 1
        print("channel_val:", channel_val)
        df_s_bin['channel'] = channel_val
        # burn_uni = np.sort(np.unique(burn_val_arr))
        # print ("burn_uni:", burn_uni)
        if not os.path.exists(args.output_conversion_csv_binned):
            print("Write burn_val -> speed conversion to:",
                  args.output_conversion_csv_binned)
            df_s_bin.to_csv(args.output_conversion_csv_binned)
        else:
            print("path already exists, not overwriting...",
                  args.output_conversion_csv_binned)

    ###########################################################################

    ###########################################################################
    speed_mask_dir(args.geojson_dir, args.image_dir,
                   args.output_mask_dir_contin,
                   speed_to_burn_func,
                   mask_burn_val_key=mask_burn_val_key,
                   buffer_distance_meters=args.buffer_distance_meters,
                   buffer_roundness=buffer_roundness,
                   dissolve_by=dissolve_by,
                   bin_conversion_key=bin_conversion_key,
                   verbose=verbose,
                   # below here is all variables for binned speed
                   output_dir_multidim=args.output_mask_multidim_dir,
                   channel_value_mult=channel_value_mult,
                   n_channels=n_channels,
                   channel_burnValue=channel_burnValue,
                   append_total_band=append_total_band,
                   label_type=args.label_type,
                   crs=args.crs
                   )


###############################################################################
if __name__ == "__main__":
    main()
