#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 07:46:39 2019

@author: avanetten
"""

import os
import sys
import cv2
import subprocess
import gdal
import numpy as np
import argparse
from multiprocessing.pool import Pool

# numbers retrieved from all_dems_min_max.py, and give min, max value for 
# each band over the entirety of SN3
rescale = {
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
def convert_to_8Bit(inputRaster, outputRaster,
                    outputPixType="Byte",
                    outputFormat="GTiff",
                    rescale_type="perc",
                    percentiles=[2, 98],
                    band_order=[],
                    nodata_val=0,
                    max_zero_frac=0.3,
                    ):
    '''
    Convert 16bit image to 8bit
    rescale_type = [clip, perc, <dict>key
        if clip, scaling is done sctricly between 0 65535
        if rescale, each band is rescaled to a min and max
        set by percentiles
        if dict, access the 'rescale' dict at the beginning for rescaling
    percentiles, if using rescale_type=rescale, otherwise ignored
    if the array has greater than max_zero_frac == 0, then skip

    band_order determines which bands and in what order to create them.
        If band_order == [], use all bands.
        for WV3 8-band,  RGB corresponds to bands: 5, 3, 2
    https://gdal.org/programs/gdal_translate.html
    '''

    srcRaster = gdal.Open(inputRaster)
    if len(band_order) == 0:
        nbands = srcRaster.RasterCount
    else:
        nbands = len(band_order)

    if nbands == 3:
        cmd = ['gdal_translate', '-ot', outputPixType, '-of', outputFormat, '-a_nodata', str(nodata_val),
               '-co', '"PHOTOMETRIC=rgb"']
    else:
        cmd = ['gdal_translate', '-ot', outputPixType, '-of', outputFormat, '-a_nodata', str(nodata_val)]

    # get bands
    if len(band_order) == 0:
        band_list = range(1, srcRaster.RasterCount + 1)
    else:
        band_list = band_order
    # iterate through bands
    # for bandId in range(srcRaster.RasterCount):
    #    bandId = bandId+1
    for j, bandId in enumerate(band_list):
        band = srcRaster.GetRasterBand(bandId)
        if rescale_type == "perc":
            # print(j, "band:", band)
            bmin = band.GetMinimum()
            bmax = band.GetMaximum()
            # print("bmin, bmax:", bmin, bmax)
            # # if not exist minimum and maximum values
            # if bmin is None or bmax is None:
            #     (bmin, bmax) = band.ComputeRasterMinMax(1)
            # else, rescale to percentiles, ignoring all null or 0 values
            band_arr_tmp = band.ReadAsArray()
            band_arr_flat = band_arr_tmp.flatten()
            # if bmin, bmax not exist, use numpy to get min and max
            if bmin is None or bmax is None:
                bmin, bmax = np.min(band_arr_flat), np.max(band_arr_flat)

            ####################
            # ensure bmin is 1 or greater!
            #bmin = max(1, bmin)
            ####################
            
            # print("np min, max for band_arr_flat:", np.min(band_arr_flat), np.max(band_arr_flat))
            print("bmin, bmax:", bmin, bmax)
            band_arr_pos = band_arr_flat[band_arr_flat > 0]

            ####################
            # test zero frac
            zero_frac = 1.0 - (len(band_arr_pos) / (1.0 * len(band_arr_flat)))
            print("zero_frac = ", zero_frac)
            if zero_frac >= max_zero_frac:
                cmd_str = 'echo ' + inputRaster + ' too many zeros, skpping'
                print("zero_frac = ", zero_frac, "skipping...")
                return cmd_str
            ####################
            
            if len(band_arr_pos) == 0:
                (bmin, bmax) = band.ComputeRasterMinMax(1)
            else:
                bmin = np.percentile(band_arr_pos, percentiles[0])
                bmax = np.percentile(band_arr_pos, percentiles[1])
        elif rescale_type == 'clip':
            bmin, bmax = 0, 65535
        else: 
            # d = rescale[rescale_type]
            bmin, bmax = rescale[rescale_type][bandId]
            # isinstance(rescale_type, dict):
            # bmin, bmax = rescale_type[bandId]

        ####################
        # ensure bmin is 1 or greater!
        bmin = max(1, bmin)
        ####################        print(j, "bmin, bmax:", bmin, bmax)

        cmd.append("-b {}".format(bandId))
        # scale must denote the output bantd
        cmd.append("-scale_{}".format(j+1))
        cmd.append("{}".format(bmin))
        cmd.append("{}".format(bmax))
        cmd.append("{}".format(0))
        cmd.append("{}".format(255))

    cmd.append(inputRaster)
    cmd.append(outputRaster)
    cmd_str = ' '.join(cmd)
    # cmd_str = '"' + ' '.join(cmd) + '"'
    print("Conversion_command list:", cmd)
    print("Conversion_command str:", cmd_str)

    try:
        os.system(cmd_str)
        # subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
        return cmd_str
    except:
        return cmd_str


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

    return m


###############################################################################
def process_image(params):
    im_file, im_file_raw, im_file_out, \
    outputPixType, outputFormat, rescale_type, percentiles, \
    max_zero_frac, band_order \
    = params

    if not im_file.endswith('.tif'):
        return

    if not os.path.isfile(im_file_out):
        #apls_tools.convert_to_8Bit(im_file_raw, im_file_out,
        # print ("isinstance(rescale_type, dict):", isinstance(rescale[rescale_type], dict))
        cmd_str = convert_to_8Bit(im_file_raw, im_file_out,
                                   outputPixType=outputPixType,
                                   outputFormat=outputFormat,
                                   rescale_type=rescale_type,
                                   percentiles=percentiles,
                                   band_order=band_order,
                                   max_zero_frac=max_zero_frac)
    else:
        print ("File exists, skipping!", im_file_out)

    
###############################################################################
def dir_to_8bit(path_images_raw, path_images_8bit,
                command_file_loc='',
                outputPixType="Byte",
                outputFormat="GTiff",
                rescale_type="perc",
                percentiles=[2, 98],
                max_zero_frac=0.3,
                band_order=[],
                n_threads=12):
    '''Create directory of 8bit images'''

    # iterate through images, convert to 8-bit, and create masks
    im_files = [z for z in sorted(os.listdir(path_images_raw)) if z.endswith('.tif')]
    print("im_files:", im_files)

    params = []
    for i, im_file in enumerate(im_files):
           
        # create 8-bit image
        im_file_raw = os.path.join(path_images_raw, im_file)
        im_file_out = os.path.join(path_images_8bit, im_file)

        params.append((im_file, im_file_raw, im_file_out, \
                        outputPixType, outputFormat, rescale_type, percentiles, \
                        max_zero_frac, band_order))
                        
    pool = Pool(n_threads)
    pool.map(process_image, params)

    return

###############################################################################
def dir_to_8bit_single_threaded(path_images_raw, path_images_8bit,
                command_file_loc='',
                outputPixType="Byte",
                outputFormat="GTiff",
                rescale_type="perc",
                percentiles=[2, 98],
                max_zero_frac=0.3,
                band_order=[],
                n_threads=12):
    '''Create directory of 8bit images'''

    # os.makedirs(path_images_8bit, exist_ok=True)
    if len(command_file_loc) > 0:
        f = open(command_file_loc, 'w')

    # iterate through images, convert to 8-bit, and create masks
    im_files = sorted(os.listdir(path_images_raw))
    print("im_files:", im_files)

    # m = defaultdict(list)
    for i, im_file in enumerate(im_files):
    
        if not im_file.endswith('.tif'):
            continue

        if (i % 1) == 0:
            print("\n")
            print (i, im_file)
            
        # create 8-bit image
        im_file_raw = os.path.join(path_images_raw, im_file)
        im_file_out = os.path.join(path_images_8bit, im_file)
        #im_file_out = os.path.join(path_images_8bit, test_data_name + name_root + '.tif')
        # convert to 8bit
        # m = calc_rescale(im_file_raw, m, percentiles=[2,98])
        # continue
        
        if not os.path.isfile(im_file_out):
            #apls_tools.convert_to_8Bit(im_file_raw, im_file_out,
            # print ("isinstance(rescale_type, dict):", isinstance(rescale[rescale_type], dict))
            cmd_str = convert_to_8Bit(im_file_raw, im_file_out,
                                       outputPixType=outputPixType,
                                       outputFormat=outputFormat,
                                       rescale_type=rescale_type,
                                       percentiles=percentiles,
                                       band_order=band_order,
                                       max_zero_frac=max_zero_frac)

            if len(command_file_loc) > 0:
                f.write(cmd_str + '\n')
        else:
            print ("File exists, skipping!", im_file_out)

    if len(command_file_loc) > 0:
        f.close()

    return


###############################################################################
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', type=str, default='')
    parser.add_argument('--outdir', type=str, default='')
    parser.add_argument('--command_file_loc', type=str, default='')
    parser.add_argument('--rescale_type', type=str, default='perc',
                        help="clip, perc, tot_8band, tot_3band")
    parser.add_argument('--band_order', type=str, default='5,3,2',
                        help="',' separated list "
                        " set to '' to use default band order, 1-indexed")
    parser.add_argument('--percentiles', type=str, default='2,98',
                        help="',' separated list of min,max percentiles")
    parser.add_argument('--max_zero_frac', type=float, default=0.3,
                        help="max percentage of image we allow to be null")
    parser.add_argument('--n_threads', type=int, default=12,
                           help="num threads for multiprocessing")
    args = parser.parse_args()

    # parse band_order
    if len(args.band_order) == 0:
        band_order = []
    else:
        band_order_str = args.band_order.split(',')
        band_order = [int(z) for z in band_order_str]
    percentiles = [int(z) for z in args.percentiles.split(',')]

    # values that should remain constant
    outputPixType = "Byte"
    outputFormat = "GTiff"

    os.makedirs(args.outdir, exist_ok=True)

    dir_to_8bit(args.indir, args.outdir,
                command_file_loc=args.command_file_loc,
                rescale_type=args.rescale_type,
                band_order=band_order,
                outputPixType=outputPixType,
                outputFormat=outputFormat,
                percentiles=percentiles,
                max_zero_frac=args.max_zero_frac,
                n_threads=args.n_threads)
