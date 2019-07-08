#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 17 14:07:03 2018

@author: avanetten
"""

from __future__ import print_function
import os
import json
import argparse
import pandas as pd
import numpy as np
import skimage.io
import cv2
import time
from config import Config
import logging
from other_tools import make_logger
# import gdal

#import sys
#path_basiss = os.path.dirname(os.path.realpath(__file__))
#sys.path.append(path_basiss)
#import basiss

###############################################################################
def post_process_image(df_pos_, data_dir, num_classes=1, im_prefix='',
                       super_verbose=False):
    '''
    For a dataframe of image positions (df_pos_), and the tiles of that image,
    reconstruct the image. Image can be a single band mask, a 3-band image, a
    multiband mask, or a multiband image.  Adapted from basiss.py
    Assume that only one image root is in df_pos_   
    '''
    
    # make sure we don't saturate overlapping, images, so rescale by a factor
    # of 4 (this allows us to not use np.uint16 for mask_raw)
    rescale_factor = 1
    
    # get image width and height
    w, h = df_pos_['im_x'].values[0], df_pos_['im_y'].values[0]
    
    if num_classes == 1:
        # create numpy zeros of appropriate shape
        #mask_raw = np.zeros((h,w), dtype=np.uint8)  # dtype=np.uint16)
        mask_raw = np.zeros((h,w), dtype=np.uint16)
        #  = create another zero array to record which pixels are overlaid
        mask_norm = np.zeros((h,w), dtype=np.uint8)  # dtype=np.uint16)
    else:
        # create numpy zeros of appropriate shape
        #mask_raw = np.zeros((h,w), dtype=np.uint8)  # dtype=np.uint16)
        mask_raw = np.zeros((h,w, num_classes), dtype=np.uint16)
        #  = create another zero array to record which pixels are overlaid
        mask_norm = np.zeros((h,w, num_classes), dtype=np.uint8)  # dtype=np.uint16)
    
    overlay_count = np.zeros((h,w), dtype=np.uint8)
    
    # iterate through slices
    for i, (idx_tmp, item) in enumerate(df_pos_.iterrows()):
        
        if (i % 50) == 0:
            print (i, "/", len(df_pos_))
            #print (i, "\n", idx_tmp, "\n", item)
        [row_val, idx, name, name_full, xmin, ymin, slice_x, slice_y, im_x, im_y] = item
        # add prefix, if required
        if len(im_prefix) > 0:
            name = im_prefix + name
            name_full = im_prefix + name_full
        if num_classes == 1:
            mask_slice_refine = cv2.imread(os.path.join(data_dir, name_full), 0)
        elif num_classes == 3:
            mask_slice_refine = cv2.imread(os.path.join(data_dir, name_full), 1)
        else:
            t01 = time.time()
#            mask_slice_refine = gdal.Open(os.path.join(data_dir, name_full)).ReadAsArray() 
#            # make sure image is h,w,channels (assume less than 20 channels)
#            if (len(mask_slice_refine.shape) == 3) and (mask_slice_refine.shape[0] < 20): 
#                mask_slice_refine = np.moveaxis(mask_slice_refine, 0, -1)

            # gdal is much faster, just needs some work !!!!
            # skimage
            plugin = 'tifffile'
            mask_slice_refine = skimage.io.imread(os.path.join(data_dir, name_full), plugin=plugin)
            # assume channels, h, w, so we want to reorder to h,w, channels?
            mask_slice_refine = np.moveaxis(mask_slice_refine, 0, -1)
            #print ("mask_slice_refine.shape:", mask_slice_refine.shape)
            #print ("Time to read image:", time.time() - t01, "seconds")

#                # we want skimage to read in (channels, h, w) for multi-channel
#                #   assume less than 20 channels
#                #print ("mask_channels.shape:", mask_channels.shape)
#                if prob_arr_tmp.shape[0] > 20: 
#                    #print ("mask_channels.shape:", mask_channels.shape)
#                    prob_arr = np.moveaxis(prob_arr_tmp, 0, -1)
#                    #print ("mask.shape:", mask.shape)  

        # rescale make slice?
        if rescale_factor != 1:
            mask_slice_refine = (mask_slice_refine / rescale_factor).astype(np.uint8)
        
        #print ("mask_slice_refine:", mask_slice_refine)
        if super_verbose:
            print ("item:", item)
            
        x0, x1 = xmin, xmin + slice_x
        y0, y1 = ymin, ymin + slice_y

        if num_classes == 1:
            # add mask to mask_raw
            mask_raw[y0:y1, x0:x1] += mask_slice_refine
        else:
            # add mask to mask_raw
            mask_raw[y0:y1, x0:x1, :] += mask_slice_refine
            # per channel
            #for c in range(num_classes):
            #    mask_raw[y0:y1, x0:x1, c] += mask_slice_refine[:,:,c]
        # update count
        overlay_count[y0:y1, x0:x1] += np.ones((slice_y, slice_x), dtype=np.uint8)


    # compute normalized mask
    # if overlay_count == 0, reset to 1
    overlay_count[np.where(overlay_count == 0)] = 1
    if rescale_factor != 1:
        mask_raw = mask_raw.astype(np.uint8)
    
    #print ("np.max(overlay_count):", np.max(overlay_count))
    #print ("np.min(overlay_count):", np.min(overlay_count))
                  
    # throws a memory error if using np.divide...
    if (w < 60000) and (h < 60000):
        if num_classes == 1:
            mask_norm = np.divide(mask_raw, overlay_count).astype(np.uint8)
        else:
            for j in range(num_classes):
                mask_norm[:,:,j] = np.divide(mask_raw[:,:,j], overlay_count).astype(np.uint8)
    else:
        for j in range(h):
            #print ("j:", j)
            mask_norm[j] = (mask_raw[j] / overlay_count[j]).astype(np.uint8)

#    # throws a memory error if using np.divide...
#    if (w < 60000) and (h < 60000):
#        mask_norm = np.divide(mask_raw, overlay_count).astype(np.uint8)
#    else:
#        for j in range(h):
#            #print ("j:", j)
#            mask_norm[j] = (mask_raw[j] / overlay_count[j]).astype(np.uint8)
#            #for k in range(w): 
#            #    mask_norm[j,k] = (mask_raw[j,k] / overlay_count[j,k]).astype(np.uint8)
    
    # rescale mask_norm
    if rescale_factor != 1:
        mask_norm = (mask_norm * rescale_factor).astype(np.uint8)
    
    #print ("mask_norm.shape:", mask_norm.shape)
    #print ("mask_norm.dtype:", mask_norm.dtype)

    return name, mask_norm, mask_raw, overlay_count   
        

###############################################################################
def post_process_image_3band(df_pos_, data_dir, n_bands=3, im_prefix='',
                             super_verbose=False):
    '''
    For a dataframe of image positions (df_pos_), and the tiles of that image,
    reconstruct the image. Adapted from basiss.py
    Assume that only one image root is in df_pos_
    '''
    
    # make sure we don't saturate overlapping, images, so rescale by a factor
    # of 4 (this allows us to not use np.uint16 for im_raw)
    rescale_factor = 1
    
    # get image width and height
    w, h = df_pos_['im_x'].values[0], df_pos_['im_y'].values[0]
    
    # create numpy zeros of appropriate shape
    #im_raw = np.zeros((h,w), dtype=np.uint8)  # dtype=np.uint16)
    im_raw = np.zeros((h,w,n_bands), dtype=np.uint16)

    #  = create another zero array to record which pixels are overlaid
    im_norm = np.zeros((h,w,n_bands), dtype=np.uint8)  # dtype=np.uint16)
    overlay_count = np.zeros((h,w), dtype=np.uint8)

    # iterate through slices
    for i, (idx_tmp, item) in enumerate(df_pos_.iterrows()):
        
        if (i % 100) == 0:
            print (i, "/", len(df_pos_))
            #print (i, "\n", idx_tmp, "\n", item)
        [row_val, idx, name, name_full, xmin, ymin, slice_x, slice_y, im_x, im_y] = item
 
        if len(im_prefix) > 0:
            name = im_prefix + name
            name_full = im_prefix + name_full
       
        # read in image
        if n_bands == 3:
            im_slice_refine = cv2.imread(os.path.join(data_dir, name_full), 1)
        else:
            print ("Still need to write code to handle multispecral data...")
            return
        
        # rescale make slice?
        if rescale_factor != 1:
            im_slice_refine = (im_slice_refine / rescale_factor).astype(np.uint8)
        
        #print ("im_slice_refine:", im_slice_refine)
        if super_verbose:
            print ("item:", item)
            
        x0, x1 = xmin, xmin + slice_x
        y0, y1 = ymin, ymin + slice_y
        #print ("x0, x1, y0, y1,", x0, x1, y0, y1)

        # add data to im_raw for each band
        for j in range(n_bands):
            im_raw[y0:y1, x0:x1, j] += im_slice_refine[:,:,j]

        # update count
        overlay_count[y0:y1, x0:x1] += np.ones((slice_y, slice_x), dtype=np.uint8)

    # compute normalized im
    # if overlay_count == 0, reset to 1
    overlay_count[np.where(overlay_count == 0)] = 1
    if rescale_factor != 1:
        im_raw = im_raw.astype(np.uint8)
    
    #print ("np.max(overlay_count):", np.max(overlay_count))
    #print ("np.min(overlay_count):", np.min(overlay_count))
                  
    # throws a memory error if using np.divide...
    if h < 60000:
        for j in range(n_bands):
            im_norm[:,:,j] = np.divide(im_raw[:,:,j], overlay_count).astype(np.uint8)
    else:
        for j in range(h):
            #print ("j:", j)
            im_norm[j] = (im_raw[j] / overlay_count[j]).astype(np.uint8)
    
    # rescale im_norm
    if rescale_factor != 1:
        im_norm = (im_norm * rescale_factor).astype(np.uint8)
    
    #print ("im_norm.shape:", im_norm.shape)
    #print ("im_norm.dtype:", im_norm.dtype)

    return name, im_norm, im_raw, overlay_count   

###############################################################################
def main():

    # if using config instead of argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path')
    args = parser.parse_args()
    with open(args.config_path, 'r') as f:
        cfg = json.load(f)
        config = Config(**cfg)

    print ("Running stitch.py...")
    
    
    save_overlay_and_raw = False  # switch to save the stitchin overlay and 
                                  # non-normalized image
    
    
    # compression 0 to 9 (most compressed)
    compression_params = [cv2.IMWRITE_PNG_COMPRESSION, 5]


    folds_dir = os.path.join(config.path_results_root, 
                             config.test_results_dir, 
                             config.folds_save_dir)
    merge_dir = os.path.join(config.path_results_root, 
                             config.test_results_dir, 
                             config.merged_dir)
    
    if config.num_folds > 1:
        im_dir = merge_dir
        im_prefix = ''
    else:
        im_dir = folds_dir
        im_prefix = 'fold0_'
    
    
    # output dirs
    out_dir_mask_raw = os.path.join(config.path_results_root, config.test_results_dir, config.stitched_dir_raw)
    out_dir_count = os.path.join(config.path_results_root, config.test_results_dir, config.stitched_dir_count)
    out_dir_mask_norm = os.path.join(config.path_results_root, config.test_results_dir, config.stitched_dir_norm)
    
    # assume tile csv is in data dir, not root dir
    path_tile_df_csv = os.path.join(config.path_data_root, os.path.dirname(config.test_sliced_dir), config.tile_df_csv)
    # try tile_df_csv in results path
    #path_tile_df_csv = os.path.join(config.path_results_root, config.test_results_dir, config.tile_df_csv)

    #out_dir_mask_norm = config.stitched_dir_norm #os.path.join(config.stitched_dir ,'mask_norm')
    #out_dir_mask_raw = config.stitched_dir_raw #os.path.join(config.stitched_dir, 'mask_raw')
    #out_dir_count = config.stitched_dir_count #os.path.join(config.stitched_dir, 'mask_count')
    #res_root_dir = os.path.dirname(config.merged_dir)
    ##out_dir_root = os.path.join(res_root_dir, 'stitched')
    #out_dir_mask_norm = os.path.join(res_root_dir, 'stitched/mask_norm')
    #out_dir_mask_raw = os.path.join(res_root_dir, 'stitched/mask_raw')
    #out_dir_count = os.path.join(res_root_dir, 'stitched/mask_count')
    
    # make dirs
    os.makedirs(out_dir_mask_norm, exist_ok=True)  
    os.makedirs(out_dir_mask_raw, exist_ok=True)  
    os.makedirs(out_dir_count, exist_ok=True)  
 
    res_root_dir = os.path.join(config.path_results_root, config.test_results_dir)
    log_file = os.path.join(res_root_dir, 'stitch.log')
    console, logger1 = make_logger.make_logger(log_file, logger_name='log')
#    ###############################################################################
#    # https://docs.python.org/3/howto/logging-cookbook.html#logging-to-multiple-destinations
#    # set up logging to file - see previous section for more details
#    res_root_dir = os.path.join(config.path_results_root, config.test_results_dir)
#    log_file = os.path.join(res_root_dir, 'stitch.log')
#    logging.basicConfig(level=logging.DEBUG,
#                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
#                        datefmt='%m-%d %H:%M',
#                        filename=log_file,
#                        filemode='w')
#    # define a Handler which writes INFO messages or higher to the sys.stderr
#    console = logging.StreamHandler()
#    console.setLevel(logging.INFO)
#    # set a format which is simpler for console use
#    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
#    #formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
#    # tell the handler to use this format
#    console.setFormatter(formatter)
#    # add the handler to the root logger
#    logging.getLogger('').addHandler(console)
#    logger1 = logging.getLogger('log')
#    logger1.info("log file: {x}".format(x=log_file))
#    ###############################################################################  
    
    
    # read in df_pos
    #df_file = os.path.join(out_dir_root, 'tile_df.csv')
    df_pos_tot = pd.read_csv(path_tile_df_csv)
    logger1.info("len df_pos_tot: {x}".format(x=len(df_pos_tot)))
    #print ("len df_pos_tot:", len(df_pos_tot))
    t0 = time.time()
    ttot = 0
    
    # save for each individual image
    idxs = np.sort(np.unique(df_pos_tot['idx']))
    logger1.info("image idxs: {x}".format(x=idxs))
    #print ("image idxs:", idxs)
    for idx in idxs:
        logger1.info("\n")
        logger1.info("idx: {x} / {y}".format(x=idx+1, y=len(idxs)))
        #print ("\nidx:", idx, "/", len(idxs))
        # filter by idx
        df_pos = df_pos_tot.loc[df_pos_tot['idx'] == idx]
        logger1.info("len df_pos: {x}".format(x=len(df_pos)))
        #print ("len df_pos:", len(df_pos))
        # execute
        t1 = time.time()
        name, mask_norm, mask_raw, overlay_count = \
                post_process_image(df_pos, im_dir, 
                                   im_prefix=im_prefix,
                                   num_classes=config.num_classes,
                                   super_verbose=False)
        t2 = time.time()
        ttot += t2-t1
        logger1.info("Time to run stitch for idx: {x} = {y} seconds".format(x=idx, y=t2-t1))
        #print ("Time to run stitch for idx:", idx, "=", t2 - t1, "seconds")
        logger1.info("mask_norm.shape: {x}".format(x=mask_norm.shape))
        print ("mask_norm.dtype:", mask_norm.dtype)
        print ("mask_raw.dtype:", mask_raw.dtype)
        print ("overlay_count.dtype:", overlay_count.dtype)
        print ("np.max(overlay_count):", np.max(overlay_count))
        print ("np.min(overlay_count):", np.min(overlay_count))
    
        # write to files (cv2 can't handle reading enormous files, can write large ones)
        print ("Saving to files...")
        # remove prefix, if required
        if len(im_prefix) > 0:
            out_file_root = name.split(im_prefix)[-1] + '.tif'
        else:
            out_file_root = name + '.tif'
            
        logger1.info("out_file_root {x}:".format(x=out_file_root))
        #print ("out_file_root:", out_file_root)
        out_file_mask_norm = os.path.join(out_dir_mask_norm, out_file_root)
        out_file_mask_raw = os.path.join(out_dir_mask_raw, out_file_root)
        out_file_count = os.path.join(out_dir_count, out_file_root)
        
        if config.num_classes == 1:
            cv2.imwrite(out_file_mask_norm, mask_norm.astype(np.uint8), compression_params)
            del mask_norm
            if save_overlay_and_raw:
                cv2.imwrite(out_file_mask_raw, mask_raw.astype(np.uint8), compression_params)
            del mask_raw
        else:
            mask_norm = np.moveaxis(mask_norm, -1, 0).astype(np.uint8)
            skimage.io.imsave(out_file_mask_norm, mask_norm, compress=1)
            del mask_norm
            if save_overlay_and_raw:    
                mask_raw = np.moveaxis(mask_raw, -1, 0).astype(np.uint8)
                skimage.io.imsave(out_file_mask_raw, mask_raw, compress=1)
            del mask_raw

        if save_overlay_and_raw:
            cv2.imwrite(out_file_count, overlay_count, compression_params)
        #cv2.imwrite(out_file_count, overlay_count.astype(np.uint8), compression_params)
        del overlay_count
        #skimage.io.imsave(out_file_mask_norm, mask_norm)
        #skimage.io.imsave(out_file_mask_raw, mask_raw)
        #skimage.io.imsave(out_file_count, overlay_count)
    
    t3 = time.time()
    logger1.info("Time to run stitch.py and create large masks: {} seconds".format(ttot))
    logger1.info("Time to run stitch.py and create large masks (and save): {} seconds".format(t3-t0))
    #print ("Time to run stitch.py and create large masks:", ttot, "seconds")
    #print ("Time to run stitch.py and create large masks (and save):", t3 - t0, "seconds")

    return



if __name__ == "__main__":
    main()