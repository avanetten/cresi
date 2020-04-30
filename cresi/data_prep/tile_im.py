#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 15 16:16:58 2018

@author: avanetten
"""

from __future__ import print_function

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import cv2
# cv2 can't load large files, so need to import skimage too
import skimage.io 

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'configs'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from configs.config import Config


###############################################################################
def slice_ims(im_dir, out_dir, slice_x, slice_y, 
                    stride_x, stride_y,
                    pos_columns = ['idx', 'name', 'name_full', 'xmin', 
                                   'ymin', 'slice_x', 
                                   'slice_y', 'im_x', 'im_y'],
                    sep='__', verbose=True):
    '''Slice images into patches, assume ground truth masks 
        are present
    Adapted from basiss.py'''
    
    if verbose:
        print("Slicing images in:", im_dir)
        
    t0 = time.time()    
    count = 0
    pos_list, name_list = [], []
    tot_pixels = 0
    #nims,h,w,nbands = im_arr.shape

    im_roots = np.sort([z for z in os.listdir(im_dir) if z.endswith('.tif')])
    for i,im_root in enumerate(im_roots):

        im_path =  os.path.join(im_dir, im_root)
        if verbose:
            print("im_path:", im_path)
        name = im_root.split('.')[0]
        
        # load with skimage, and reverse order of bands
        im = skimage.io.imread(im_path)#[::-1]
        # cv2 can't load large files
        #im = cv2.imread(im_path)
        h, w, nbands = im.shape
        n_pix = h * w
        print("im.shape:", im.shape)
        print("n pixels:", n_pix)
        tot_pixels += n_pix 
        
        seen_coords = set()
        
        #if verbose and (i % 10) == 0:
        #    print(i, "im_root:", im_root)
                
        # dice it up
        # after resize, iterate through image 
        #     and bin it up appropriately
        for x in range(0, w - 1, stride_x):  
            for y in range(0, h - 1, stride_y): 
                
                xmin = min(x, w-slice_x)
                ymin = min(y, h - slice_y) 
                coords = (xmin, ymin)
                
                # check if we've already seen these coords
                if coords in seen_coords:
                    continue
                else:
                    seen_coords.add(coords)
                
                # check if we screwed up binning
                if (xmin + slice_x > w) or (ymin + slice_y > h):
                    print("Improperly binned image,")
                    return

                # get satellite image cutout
                im_cutout = im[ymin:ymin + slice_y, 
                               xmin:xmin + slice_x]
                
                ##############
                # skip if the whole thing is black
                if np.max(im_cutout) < 1.:
                    continue
                else:
                    count += 1
                
                if verbose and (count % 50) == 0:
                    print("count:", count, "x:", x, "y:", y) 
                ###############
                                

                # set slice name
                name_full = str(i) + sep + name + sep \
                    + str(xmin) + sep + str(ymin) + sep \
                    + str(slice_x)  + sep + str(slice_y) \
                    + sep + str(w) + sep + str(h) \
                    + '.tif'
                    
                pos = [i, name, name_full, xmin, ymin, slice_x, slice_y, w, h]
                # add to arrays
                #idx_list.append(idx_full)
                name_list.append(name_full)
                #im_list.append(im_cutout)
                #mask_list.append(mask_cutout) 
                pos_list.append(pos)
                
                name_out = os.path.join(out_dir, name_full)
                # if we read in with skimage, need to reverse colors
                cv2.imwrite(name_out, cv2.cvtColor(im_cutout, cv2.COLOR_RGB2BGR))
                #cv2.imwrite(name_out, im_cutout)
    
    # create position datataframe
    df_pos = pd.DataFrame(pos_list, columns=pos_columns)
    df_pos.index = np.arange(len(df_pos))
    
    if verbose:
        print("  len df;", len(df_pos))
        print("  Time to slice arrays:", time.time() - t0, "seconds")
        print("  Total pixels in test image(s):", tot_pixels)
        
    return df_pos


##############################################################################
def main():
    
    # # construct the argument parse and parse the arguments
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--im_dir', type=str, default='',
    #                     help="images location")
    # parser.add_argument('--out_dir', type=str, default='',
    #                     help="output_images location")
    # parser.add_argument('--slice_x', type=int, default=1300)
    # parser.add_argument('--slice_y', type=int, default=1300)
    # parser.add_argument('--stride_x', type=int, default=1200)
    # parser.add_argument('--stride_y', type=int, default=1200)
    # args = parser.parse_args()

    # if not os.path.exists(args.out_dir):
    #     os.mkdir(args.out_dir)

    # df_pos = slice_ims(args.im_dir, args.out_dir, args.slice_x, args.slice_y, 
    #                 args.stride_x, args.stride_y,
    #                 pos_columns = ['idx', 'name', 'xmin', 
    #                               'ymin', 'slice_x', 
    #                               'slice_y', 'im_x', 'im_y'],
    #                 verbose=True)

    # use config file
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path')
    args = parser.parse_args()
    # get config
    with open(args.config_path, 'r') as f:
        cfg = json.load(f)
        config = Config(**cfg)

    # get input dir
    path_images_8bit = os.path.join(config.test_data_refined_dir)

    # make output dirs
    # first, results dir
    res_dir = os.path.join(config.path_results_root, config.test_results_dir)
    os.makedirs(res_dir, exist_ok=True)
    path_tile_df_csv = os.path.join(config.path_results_root, config.test_results_dir, config.tile_df_csv)
    # path_tile_df_csv2 = os.path.join(config.path_data_root, os.path.dirname(config.test_sliced_dir), config.tile_df_csv)

    # path for sliced data
    path_sliced = config.test_sliced_dir
    # path_sliced = os.path.join(config.path_data_root, config.test_sliced_dir)
    print("Output path for sliced images:", path_sliced)
    
    # only run if nonzero tile and sliced_dir
    if (len(config.test_sliced_dir) > 0) and (config.slice_x > 0):
        os.makedirs(path_sliced, exist_ok=True)
   
        df_pos = slice_ims(path_images_8bit, path_sliced, 
                       config.slice_x, config.slice_y, 
                       config.stride_x, config.stride_y,
                       pos_columns = ['idx', 'name', 'name_full', 'xmin', 
                                   'ymin', 'slice_x', 
                                   'slice_y', 'im_x', 'im_y'],
                       verbose=True)
        # save to file
        df_pos.to_csv(path_tile_df_csv)
        print("df saved to file:", path_tile_df_csv)
        # also csv save to data dir
        # df_pos.to_csv(path_tile_df_csv2)


 ###############################################################################
if __name__ == '__main__':
    main()