#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 14:51:46 2019

@author: Jake Shermeyer
"""

import gdal

def CreateMultiBandGeoTiff(OutPath, Array, nodatavalue=0):
    '''
    Array has shape:
        Channels, Y, X? 
    '''
    driver=gdal.GetDriverByName('GTiff')
    DataSet = driver.Create(OutPath, Array.shape[2], Array.shape[1], 
                            Array.shape[0], gdal.GDT_Byte,
                            ['COMPRESS=LZW']
                            )
    
    for i, image in enumerate(Array, 1):
        DataSet.GetRasterBand(i).WriteArray( image )
        DataSet.GetRasterBand(i).SetNoDataValue(nodatavalue)

    del DataSet
    return OutPath


#def CreateMultiBandGeoTiff(OutPath, Array):
#     '''
#     Array has shape:
#         Channels, Y, X
#     '''
#     driver=gdal.GetDriverByName('GTiff')
#     DataSet = driver.Create(OutPath, Array.shape[2], Array.shape[1], 
#                             Array.shape[0], gdal.GDT_Byte,
#                             ['COMPRESS=LZW'])
#     for i, image in enumerate(Array, 1):
#         DataSet.GetRasterBand(i).WriteArray( image )
#     del DataSet
    
#     return OutPath