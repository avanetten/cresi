#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 26 10:05:24 2018

@author: Jake Shermeyer

"""

import numpy as np
import gdal
import os
import glob
import sys
from tqdm import tqdm

def geo_that_raster(norm_folder, original_image_folder):
    os.chdir(norm_folder)
    SR_output=glob.glob("*.tif")

    os.chdir(original_image_folder)
    Originals=glob.glob("*.tif")

    SR_output.sort()
    Originals.sort()
    #print(SR_output,Originals)

    if len(SR_output) != len(Originals):  ### could add more error checking as needed, I think simple sorting should work
        print("Inequal number of images in each folder, check your data")
        print("Exiting")
    else:
        for image,SR in tqdm(zip(Originals, SR_output)):
            #print(image,SR)
            os.chdir(original_image_folder)
            raster=gdal.Open(image)
            geo=raster.GetGeoTransform()
            proj=raster.GetProjection()
            os.chdir(norm_folder)
            O=gdal.Open(SR, gdal.GA_Update)
            O.SetProjection( proj )
            O.SetGeoTransform( geo )
            del O
            
            
if __name__ == "__main__":
    geo_that_raster(sys.argv[1], sys.argv[2])