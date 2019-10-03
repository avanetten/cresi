#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 09:16:00 2018

@author: avanetten
"""

from __future__ import print_function

import os
import time
import argparse
import numpy as np
import cv2
# cv2 can't load large files, so need to import skimage too
import skimage.io 


###############################################################################
if __name__ == '__main__':
        
    # use config file
    # use config file? 
    from config import Config
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path')
    parser.add_argument('--gsd_m', type=float, default=0.3, 
                        help="Image GSD in meters")
    args = parser.parse_args()

    # get config
    with open(args.config_path, 'r') as f:
        cfg = json.load(f)
        config = Config(**cfg)

    # get input dir
    path_images_8bit = os.path.join(config.path_data_root, config.test_data_refined_dir)

    im_roots = np.sort([z for z in os.listdir(path_images_8bit) if z.endswith('.tif')])
    tot_pixels = 0
    for i,im_root in enumerate(im_roots):

        im_path =  os.path.join(path_images_8bit, im_root)
        print ("im_path:", im_path)
        
        # load with skimage, and reverse order of bands
        im = skimage.io.imread(im_path)#[::-1]
        # cv2 can't load large files
        #im = cv2.imread(im_path)
        h, w, nbands = im.shape
        n_pix = h * w
        print ("  im.shape:", im.shape)
        print ("  n pixels:", n_pix)
        print ("  area (km2):", n_pix * args.gsd_m**2 / 1000**2)
        tot_pixels += n_pix 

    tot_area_m2 = tot_pixels * args.gsd_m**2
    tot_area_km2 = tot_area_m2 / (1000**2)
    
    print ("Total pixels in test image(s):", tot_pixels)
    print ("GSD (meter):", args.gsd_m)
    print ("Total area (square meters) in test image(s):", tot_area_m2)
    print ("Total area (square km) in test image(s):", tot_area_km2)
