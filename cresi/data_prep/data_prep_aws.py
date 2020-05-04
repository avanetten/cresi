#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 12:36:51 2019

@author: avanetten
"""

import os

#######################
path_cresi = '/home/ubuntu/src/cresi/cresi/'

im_dirs_train = [
        '/data/SN3_roads/train/AOI_2_Vegas/PS-MS/',
        '/data/SN3_roads/train/AOI_3_Paris/PS-MS/',
        '/data/SN3_roads/train/AOI_4_Shanghai/PS-MS/',
        '/data/SN3_roads/train/AOI_5_Khartoum/PS-MS/',
        '/data/SN5_roads/train/AOI_7_Moscow/PS-MS/',
        '/data/SN5_roads/train/AOI_8_Mumbai/PS-MS/',
        ]

geojson_dirs_train = [
        '/data/SN3_roads/train/AOI_2_Vegas/geojson_roads_speed/',
        '/data/SN3_roads/train/AOI_3_Paris/geojson_roads_speed/',
        '/data/SN3_roads/train/AOI_4_Shanghai/geojson_roads_speed/',
        '/data/SN3_roads/train/AOI_5_Khartoum/geojson_roads_speed/',
        '/data/SN5_roads/train/AOI_7_Moscow/geojson_roads_speed/',
        '/data/SN5_roads/train/AOI_8_Mumbai/geojson_roads_speed/'
        ]

im_dir_test_root = '/data/SN5_roads/test_private'


# outputs (train)
eightbit_im_dir = '/home/ubuntu/data/cresi_data/train/8bit/PS-RGB'
output_conversion_csv = '/home/ubuntu/data/cresi_data/SN5_roads_train_speed_conversion_binned.csv'
speed_mask_dir = '/home/ubuntu/data/cresi_data/train/train_mask_binned'
speed_mask_multidim_dir = '/home/ubuntu/data/cresi_data/train/train_mask_binned_mc'
os.makedirs(eightbit_im_dir)
os.makedirs(speed_mask_dir)
os.makedirs(speed_mask_multidim_dir)

# outputs (test)
eightbit_im_dir_test = '/home/ubuntu/data/cresi_data/test/8bit/PS-RGB'
os.makedirs(eightbit_im_dir_test)
#######################

#######################
# 0. Data Prep for Training

#######################
# Build 8bit training images from 16bit SpaceNet images
for im_dir in im_dirs_train:
    cmd = 'python ' \
          + os.path.join(path_cresi, 'data_prep/create_8bit_images.py') \
          + ' --indir=' + im_dir \
          + ' --outdir=' + eightbit_im_dir \
          + ' --rescale_type=perc' \
          + ' --percentiles=2,98'\
          + ' --band_order=5,3,2'
    print(repr(cmd))
    print("images to 8bit cmd:", cmd)
    os.system(cmd)

#######################
# prep speed masks
for im_dir, geojson_dir in zip(im_dirs_train, geojson_dirs_train):
    cmd = 'python ' \
          + os.path.join(path_cresi, 'data_prep/speed_masks.py') + ' ' \
          + '--geojson_dir=' + geojson_dir + ' ' \
          + '--image_dir=' + im_dir + ' ' \
          + '--output_conversion_csv=' + output_conversion_csv + ' ' \
          + '--output_mask_dir=' + speed_mask_dir + ' ' \
          + '--output_mask_multidim_dir=' + speed_mask_multidim_dir

    print("speed mask cmd:", cmd)
    os.system(cmd)


#######################
# 1. Data Prep for Testing

#######################
# Build 8bit test images from 16bit SpaceNet images
im_dirs_test_part = [z for z in os.listdir(im_dir_test_root)
                     if z.startswith('AOI')]
for im_dir_part in im_dirs_test_part:
    im_dir = os.path.join(im_dir_test_root, im_dir_part, 'PS-MS')
    cmd = 'python ' \
          + os.path.join(path_cresi, 'data_prep/create_8bit_images.py') + ' ' \
          + '--indir=' + im_dir + ' ' \
          + '--outdir=' + eightbit_im_dir_test + ' ' \
          + '--rescale_type=perc' + ' ' \
          + '--percentiles=2,98' + ' ' \
          + '--band_order=5,3,2'
    print("test images to 8bit cmd:", cmd)
    os.system(cmd)
