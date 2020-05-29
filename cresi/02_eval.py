#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat May  5 02:18:38 2018

@author: avanetten

Adapted from:
https://github.com/SpaceNetChallenge/RoadDetector/tree/master/albu-solution
"""
# Standard library imports.
import argparse
import glob
import json
import logging
import os
import time

# Related third party imports.
import cv2
import torch
from tqdm import tqdm

# Local application/library specific imports.
from configs.config import Config
from net.augmentations.transforms import get_flips_colors_augmentation, \
    get_flips_shifts_augmentation
from net.dataset.reading_image_provider import ReadingImageProvider
from net.dataset.raw_image import RawImageType
from net.pytorch_utils.concrete_eval import FullImageEvaluator
from utils.utils import update_config, get_csv_folds
from utils import make_logger

# Need the following to avoid the following error:
# TqdmSynchronisationWarning: Set changed size during iteration
# see https://github.com/tqdm/tqdm/issues/481
tqdm.monitor_interval = 0

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

# https://github.com/pytorch/pytorch/issues/1668
#device = torch.device('cuda:0' if torch.cuda.is_avaliable() else 'cpu')
if torch.cuda.is_available():
    print("Executing inference with GPUs")
    # pytorch 0.3
    # torch.cuda.device(0)
    ## pytorch 0.4
    #device = "cuda"
        #https://discuss.pytorch.org/t/cuda-freezes-the-python/9651/5
    # torch.cuda.empty_cache()
    torch.randn(10).cuda()
else:
    print("Executing inference on the CPU")
    # pytorch 0.3
    torch.cuda.device(-1)
    ## pytorch 0.4
    #device = "cpu"


###############################################################################
class RawImageTypePad(RawImageType):
    global config
    def finalyze(self, data):
        # Border reflection of 22 yields a field size of 1344 for 1300 pix inputs
        return self.reflect_border(data, config.padding)  #22)


###############################################################################
def eval_cresi(config,
               paths,
               fn_mapping,
               image_suffix,
               save_dir,
               test=True,
               num_channels=3,
               weight_dir='',
               nfolds=4,
               save_im_gdal_format=False):
    
    # no grad needed for test, and uses less memory?
    with torch.no_grad():
    # if False:
        #t0 = time.time()
        ds = ReadingImageProvider(RawImageTypePad, paths, fn_mapping, 
                                  image_suffix=image_suffix, num_channels=num_channels)
        
        folds = [([], list(range(len(ds)))) for i in range(nfolds)]
        if torch.cuda.is_available():
            num_workers = 0 if os.name == 'nt' else 2
        else:            
            # Get connection error if more than 0 workers and cpu:
            # https://discuss.pytorch.org/t/data-loader-crashes-during-training-something-to-do-with-multiprocessing-in-docker/4379/5
            num_workers = 0

        print("num_workers:", num_workers)
        keval = FullImageEvaluator(config, ds, save_dir=save_dir, test=test, 
                                   flips=3, num_workers=num_workers, 
                                   border=config.padding,
                                   save_im_gdal_format=save_im_gdal_format)
        for fold, (t, e) in enumerate(folds):
           print("fold:", fold)
           if args.fold is not None and int(args.fold) != fold:
               print("ummmm....")
               continue
           keval.predict(fold, e, weight_dir)
    return folds


###############################################################################
if __name__ == "__main__":
    save_im_gdal_format = False #True
    #save_im_skimage = False

    parser = argparse.ArgumentParser()
    parser.add_argument('config_path')
    parser.add_argument('--fold', type=int)
    args = parser.parse_args()

    # Get config
    with open(args.config_path, 'r') as f:
        cfg = json.load(f)
    config = Config(**cfg)
    config = update_config(config, target_rows=config.eval_rows, target_cols=config.eval_cols)

    # Set images folder (depending on if we are slicing or not)
    if config.test_sliced_dir and config.slice_x:
        print("(len(config.test_sliced_dir) > 0) and (config.slice_x > 0), executing tile_im.py..")
        # path_images = path_sliced = os.path.join(config.path_data_root, config.test_sliced_dir)
        cmd = 'python ' + config.path_src + '/data_prep/tile_im.py ' + args.config_path
        print("slice command:", cmd)
        os.system(cmd)
        path_images = config.test_sliced_dir
    else:
        path_images = config.test_data_refined_dir
 
    # Check image files
    exts = ('*.tif', '*.tiff', '*.jpg', '*.JPEG', '*.JPG', '*.png', ) # the tuple of file types
    files_grabbed = []
    for ext in exts:
        files_grabbed.extend(glob.glob(os.path.join(path_images, ext)))
    if not files_grabbed:
        print("02_eval.py: No valid image files to process, returning...")
    else:
        paths = {
                'masks': '',
                'images': path_images
                }
        # Set weights_dir (same as weight_save_path)
        if config.save_weights_dir.startswith("/"):
            weight_dir = config.save_weights_dir
        else:
            weight_dir = os.path.join(config.path_results_root, 'weights', config.save_weights_dir)
            
        log_file = os.path.join(config.path_results_root,
                                config.test_results_dir,
                                'test.log')
        print("log_file:", log_file)
        # make sure output folders exist
        save_dir = os.path.join(config.path_results_root, config.test_results_dir, config.folds_save_dir)
        print("save_dir:", save_dir)
        os.makedirs(save_dir, exist_ok=True)

        fn_mapping = {'masks': lambda name: os.path.splitext(name)[0] + '.tif'}
        image_suffix = ''#'img'
        # Set folds
        skip_folds = []
        if args.fold is not None:
            skip_folds = [i for i in range(4) if i != int(args.fold)]
        print("paths:", paths)
        print("fn_mapping:", fn_mapping)
        print("image_suffix:", image_suffix)
        ###################

        # set up logging
        console, logger = make_logger.make_logger(log_file,
                                                  logger_name='log',
                                                  write_to_console=bool(config.log_to_console))

        logger.info("Testing: weight_dir: {x}".format(x=weight_dir))
        # execute
        eval_start = time.time()
        logging.info("Saving eval outputs to: {x}".format(x=save_dir))
        folds = eval_cresi(config,
                           paths,
                           fn_mapping,
                           image_suffix,
                           save_dir,
                           test=True,
                           weight_dir=weight_dir,
                           num_channels=config.num_channels,
                           nfolds=config.num_folds,
                           save_im_gdal_format=save_im_gdal_format)
        eval_end = time.time()
        eval_time = eval_end - eval_start
        logger.info("Time to run {x} folds for {y} = {z} seconds".format(x=len(folds), 
                     y=len(os.listdir(path_images)), z=eval_time))
        print("Time to run", len(folds), "folds for",
               len(os.listdir(path_images)), "=", eval_time, "seconds")
