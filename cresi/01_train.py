#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat May  5 02:18:38 2018

@author: avanetten
  
Inspired by:
https://github.com/SpaceNetChallenge/RoadDetector/tree/master/albu-solution
"""

import time
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import os
#import shutil
import torch
import logging
import json
import argparse

#https://discuss.pytorch.org/t/cuda-freezes-the-python/9651/5
torch.randn(10).cuda()

############
# need the following to avoid the following error:
#  TqdmSynchronisationWarning: Set changed size during iteration (see https://github.com/tqdm/tqdm/issues/481
from tqdm import tqdm
tqdm.monitor_interval = 0
############

from net.augmentations.transforms import get_flips_colors_augmentation, get_flips_shifts_augmentation
from net.dataset.reading_image_provider import ReadingImageProvider
from net.dataset.raw_image import RawImageType
from net.pytorch_utils.train import train
from utils.utils import update_config, get_csv_folds
from configs.config import Config
from utils import make_logger

###############################################################################
class RawImageTypePad(RawImageType):
    global config
    def finalyze(self, data):
        # border reflection of 22 yields a field size of 1344 for 1300 pix inputs
        return self.reflect_border(data, config.padding)  #22)

###############################################################################
def train_cresi(config, paths, fn_mapping, image_suffix, folds_file_loc,
                save_path, log_path, num_channels=3, logger=None):
    #t0 = time.time()
    ds = ReadingImageProvider(RawImageType, paths, fn_mapping, 
                              image_suffix=image_suffix, num_channels=num_channels)
    if logger:
        logger.info("len ds: {}".format(len(ds)))
        logger.info("folds_file_loc: {}".format(folds_file_loc))
        logger.info("save_path: {}".format(save_path))
    else:
        print("len ds:", len(ds))
        print("folds_file_loc:", folds_file_loc)
        print("save_path:", save_path)

    folds = get_csv_folds(folds_file_loc, ds.im_names)
    num_workers = 0 if os.name == 'nt' else 2
    for fold, (train_idx, val_idx) in enumerate(folds):
        if args.fold is not None and int(args.fold) != fold:
            continue
        if logger:
            logger.info("num workers: {}".format(num_workers))
            logger.info("fold: {}".format(fold))
            # logger.info("(train_idx, val_idx):", (train_idx, val_idx))
            logger.info("len(train_idx): {}".format(len(train_idx)))
            logger.info("len(val_idx): {}".format(len(val_idx)))

        if config.num_channels == 3:
            transforms = get_flips_colors_augmentation()
        else:
            # can't do hsv rescaling with multiband imagery, so skip this part
            transforms = get_flips_shifts_augmentation()
    
        train(ds, fold, train_idx, val_idx, config, save_path, log_path,
              num_workers=num_workers, transforms=transforms, logger=logger)


###############################################################################
if __name__ == "__main__":
    
    save_im_gdal_format =  True 
    
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path')
    # parser.add_argument('--training', action='store_true')
    parser.add_argument('--fold', type=int)
    args = parser.parse_args()
    
    # get config
    with open(args.config_path, 'r') as f:
        cfg = json.load(f)
    config = Config(**cfg)

    # set some vals
    ###################
    path_images_train = config.train_data_refined_dir_ims
    path_masks_train = config.train_data_refined_dir_masks


    paths = {
            'masks': path_masks_train,
            'images': path_images_train
            }
    
    log_file = os.path.join(config.path_results_root, 'weights', config.save_weights_dir, 'train.log')
    print("log_file:", log_file)

    fn_mapping = {
        'masks': lambda name: os.path.splitext(name)[0] + '.tif'  #'.png'
    }
    image_suffix = ''#'img'
    # set folds
    skip_folds = []
    if args.fold is not None:
        skip_folds = [i for i in range(4) if i != int(args.fold)]
    print ("paths:", paths)
    print ("fn_mapping:", fn_mapping)
    print ("image_suffix:", image_suffix)
    ###################

    # set up logging
    console, logger = make_logger.make_logger(log_file, logger_name='log')

    # set paths
    weight_save_path = os.path.join(config.path_results_root,
                                    'weights', config.save_weights_dir)
    log_train_path = os.path.join(config.path_results_root, 'logs')
    os.makedirs(weight_save_path, exist_ok=True)
    os.makedirs(log_train_path, exist_ok=True)
    folds_save_path = os.path.join(weight_save_path,
                                   config.folds_file_name)
    t0 = time.time()
    logger.info("Training: weight_save_path: {x}".format(x=weight_save_path))
    # print ("Training: weight_save_path:", weight_save_path)
    try:
        train_cresi(config, paths, fn_mapping, image_suffix,
                    folds_save_path, weight_save_path, log_train_path,
                    num_channels=config.num_channels, logger=logger)
    except Exception as e:
        logger.error('Error occurred: {}'.format(str(e)))
        print("log loc:", log_file)

    logger.info("Time to train: {x} seconds".format(x=time.time() - t0))
    print ("Time to train:", time.time() - t0, "seconds")
