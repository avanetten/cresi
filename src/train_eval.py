#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat May  5 02:18:38 2018

@author: avanetten
  
Adapted from:
https://github.com/SpaceNetChallenge/RoadDetector/tree/master/albu-solution
"""

import time
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import os
import numpy as np
#import shutil
import torch
import logging


#https://discuss.pytorch.org/t/cuda-freezes-the-python/9651/5
torch.randn(10).cuda()

############
# need the following to avoid the following error:
#  TqdmSynchronisationWarning: Set changed size during iteration (see https://github.com/tqdm/tqdm/issues/481
from tqdm import tqdm
tqdm.monitor_interval = 0
############

from augmentations.transforms import get_flips_colors_augmentation, get_flips_shifts_augmentation
from dataset.reading_image_provider import ReadingImageProvider
from dataset.raw_image import RawImageType
from pytorch_utils.train import train
from pytorch_utils.concrete_eval import FullImageEvaluator
from other_tools.utils import update_config, get_csv_folds
import argparse
import json
from config import Config
from other_tools import make_logger

###############################################################################
class RawImageTypePad(RawImageType):
    global config
    def finalyze(self, data):
        # border reflection of 22 yields a field size of 1344 for 1300 pix inputs
        return self.reflect_border(data, config.padding)  #22)

###############################################################################
def train_roads(config, paths, fn_mapping, image_suffix, folds_file_loc,
                save_path, log_path, num_channels=3):
    #t0 = time.time()
    ds = ReadingImageProvider(RawImageType, paths, fn_mapping, 
                              image_suffix=image_suffix, num_channels=num_channels)
    print ("len ds:", len(ds))
    print ("folds_file_loc:", folds_file_loc)
    print ("save_path:", save_path)
    folds = get_csv_folds(folds_file_loc, ds.im_names)
    #folds = get_csv_folds('folds4.csv', ds.im_names)
    num_workers = 0 if os.name == 'nt' else 2
    for fold, (train_idx, val_idx) in enumerate(folds):
        if args.fold is not None and int(args.fold) != fold:
            continue
        print ("num workers:", num_workers)
        print ("fold:", fold)
        #print ("(train_idx, val_idx):", (train_idx, val_idx))
        print ("len(train_idx):", len(train_idx))
        print ("len(val_idx):", len(val_idx))
        
        if config.num_channels == 3:
            transforms = get_flips_colors_augmentation()
        else:
            # can't do hsv rescaling with multiband imagery, so skip this part
            transforms = get_flips_shifts_augmentation()
    
        train(ds, fold, train_idx, val_idx, config, save_path, log_path,
              num_workers=num_workers, transforms=transforms)
    
###############################################################################
def eval_roads(config, paths, fn_mapping, image_suffix, save_dir, test=True,
               num_channels=3, weight_dir='', nfolds=4, 
               save_im_gdal_format=False):
    
    #t0 = time.time()
    ds = ReadingImageProvider(RawImageTypePad, paths, fn_mapping, 
                              image_suffix=image_suffix, num_channels=num_channels)
    
    folds = [([], list(range(len(ds)))) for i in range(nfolds)]
    num_workers = 0 if os.name == 'nt' else 2
    #print ("folds:", folds)
    print ("num_workers:", num_workers)
    keval = FullImageEvaluator(config, ds, save_dir=save_dir, test=test, 
                               flips=3, num_workers=num_workers, 
                               border=config.padding,
                               save_im_gdal_format=save_im_gdal_format)
    for fold, (t, e) in enumerate(folds):
       print ("fold:", fold)
       if args.fold is not None and int(args.fold) != fold:
           print ("ummmm....")
           continue
       keval.predict(fold, e, weight_dir)
           
    #t1 = time.time()
    #nfiles = len(os.listdir(args.path_images_8bit))
    #print ("Time to run", len(folds), "folds for", nfiles, "=", t1 - t0, "seconds")
    return folds

###############################################################################
if __name__ == "__main__":
    
    save_im_gdal_format =  True #False
    #save_im_skimage = False
    
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path')
    parser.add_argument('--training', action='store_true')
    #parser.add_argument('mode', type=str, default='test', help='test or train')
    parser.add_argument('--fold', type=int)
    args = parser.parse_args()
    
    # get config
    with open(args.config_path, 'r') as f:
        cfg = json.load(f)
    config = Config(**cfg)
    

    # set some vals
    ###################
    buffer_meters = float(config.mask_width_m)
    buffer_meters_str = str(np.round(buffer_meters,1)).replace('.', 'p')
    test = not args.training

    if test:
        # update config file (only if t esting!!!)
        #rows, cols = 1344, 1344
        config = update_config(config, target_rows=config.eval_rows, target_cols=config.eval_cols)
        #config = update_config(config, dataset_path=os.path.join(config.dataset_path, 'test' if test else 'train'))

        # set images folder (depending on if we are slicing or not)
        if (config.test_sliced_dir) and (config.slice_x > 0):
            path_images = path_sliced = os.path.join(config.path_data_root, config.test_sliced_dir)
            #path_images = config.path_sliced
        else:
            path_images = os.path.join(config.path_data_root, config.test_data_refined_dir)
        paths = {
                'masks': '',
                'images': path_images
                }
        # set weights_dir (same as weight_save_path)
        weight_dir = os.path.join(config.path_results_root, 'weights', config.save_weights_dir)
        log_file = os.path.join(config.path_results_root, config.test_results_dir, 'test.log')

    else:
        path_masks_train =       os.path.join(config.path_data_root, config.train_data_refined_dir_masks)
        path_images_8bit_train = os.path.join(config.path_data_root, config.train_data_refined_dir_ims)
        #path_masks_train =       os.path.join(config.path_data_root, config.train_data_refined_dir, 'masks{}m'.format(buffer_meters_str))
        #path_images_8bit_train = os.path.join(config.path_data_root, config.train_data_refined_dir, 'images')
        paths = {
                'masks': path_masks_train,
                'images': path_images_8bit_train
                }
        log_file = os.path.join(config.path_results_root, 'weights', 'train.log')

    
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
    console, logger1 = make_logger.make_logger(log_file, logger_name='log')
#    ###############################################################################
#    # https://docs.python.org/3/howto/logging-cookbook.html#logging-to-multiple-destinations
#    # set up logging to file - see previous section for more details
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


    if test:
        # make sure output folders exist
        save_dir = os.path.join(config.path_results_root, config.test_results_dir, config.folds_save_dir)
        os.makedirs(save_dir, exist_ok=True)
        logger1.info("Testing: weight_dir: {x}".format(x=weight_dir))
        #print ("Testing: weight_dir:", weight_dir)
        # execute
        t0 = time.time()
        logging.info("Saving eval outputs to: {x}".format(x=save_dir))
        #print ("Saving eval outputs to:", save_dir)
        folds = eval_roads(config, paths, fn_mapping, image_suffix, save_dir,
                           test=test, weight_dir=weight_dir, 
                           num_channels=config.num_channels,
                           nfolds=config.num_folds,
                           save_im_gdal_format=save_im_gdal_format)
        t1 = time.time()
        logger1.info("Time to run {x} folds for {y} = {z} seconds".format(x=len(folds), 
                     y=len(os.listdir(path_images)), z=t1-t0))
        #print ("Time to run", len(folds), "folds for", len(os.listdir(path_images)), "=", t1 - t0, "seconds")

    else:

        # set paths
        weight_save_path = os.path.join(config.path_results_root, 'weights', config.save_weights_dir)
        log_train_path = os.path.join(config.path_results_root, 'logs')
        os.makedirs(weight_save_path, exist_ok=True)
        os.makedirs(log_train_path, exist_ok=True)
        folds_save_path = os.path.join(weight_save_path, config.folds_file_name)
        t0 = time.time()
        logger1.info("Training: weight_save_path: {x}".format(x=weight_save_path))
        #print ("Training: weight_save_path:", weight_save_path)
        train_roads(config, paths, fn_mapping, image_suffix, 
                    folds_save_path, weight_save_path, log_train_path, 
                    num_channels=config.num_channels)
        logger1.info("Time to train: {x} seconds".format(x=time.time() - t0))
        #print ("Time to train:", time.time() - t0, "seconds")
