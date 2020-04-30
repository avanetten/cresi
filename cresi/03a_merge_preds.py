import os
import numpy as np
import argparse
import json
import cv2
import skimage.io
import shutil
import time
from configs.config import Config
from utils import make_logger
from utils.save_array_gdal import CreateMultiBandGeoTiff

# skimage gives really annoying warnings
import warnings
warnings.filterwarnings("ignore")
#with warnings.catch_warnings():
#    warnings.simplefilter("ignore")


############
# need the following to ovoid the following error:
#  TqdmSynchronisationWarning: Set changed size during iteration (see https://github.com/tqdm/tqdm/issues/481
from tqdm import tqdm
tqdm.monitor_interval = 0
############


##############################################################################
def merge_tiffs(root, out_dir, out_dir_gdal=None, num_classes=1, 
                verbose=True):

    '''
    Surprisingly, merge_tiffs works fine with multichannel, e.g.:
    # explore taking mean of multiple images (relates to merge_preds.py)
    import numpy as np
    
    # set first channel of a to 5, rest to 0
    a = np.zeros((7,20,20))
    a[0,:,:] = 5
    # set b as ones
    b = np.ones((7,20,20))
    # set c as 3, except set 7th channel as -10
    c = 3 * np.ones((7,20,20))
    c[6,:,:] = -10
    
    probs = []
    for prob_arr in [a,b,c]:
        print ("prob_arr.shape:", prob_arr.shape)
        probs.append(prob_arr)
    prob_arr = np.mean(probs, axis=0)
    print ("prob_arr.shape:", prob_arr.shape)
    # first channel should be mean of (5, 1, 3) = 3
    print ("prob_arr[0,:,:]:", prob_arr[0,:,:])
    # third channel should be mean of (0, 1, 3)
    print ("prob_arr[3,:,:]:", prob_arr[3,:,:])
    # 7th channel should be mean of (0, 1, -10) = -3
    print ("prob_arr[6,:,:]:", prob_arr[6,:,:])
    #print ("prob_arr:", prob_arr)
    '''
    
    prob_files = {f for f in os.listdir(root) if os.path.splitext(f)[1] in ['.tif', '.tiff']}
    print ("prob_files:", prob_files)
    unfolded = {f[6:] for f in prob_files if f.startswith('fold')}
    #print ("unfolded:", unfolded)
    if not unfolded:
        unfolded = prob_files

    for prob_file in tqdm(unfolded):
        probs = []
        for fold in range(4):
            prob_path = os.path.join(root, 'fold{}_'.format(fold) + prob_file)
            
            if num_classes == 1:
                prob_arr = cv2.imread(prob_path, cv2.IMREAD_GRAYSCALE)
            elif num_classes == 3:
                prob_arr = cv2.imread(prob_path, 1)
            else:
                prob_arr_tmp = skimage.io.imread(prob_path)
                # we want skimage to read in (channels, h, w) for multi-channel
                #   assume less than 20 channels
                #print ("mask_channels.shape:", mask_channels.shape)
                if prob_arr_tmp.shape[0] > 20: 
                    #print ("mask_channels.shape:", mask_channels.shape)
                    prob_arr = np.moveaxis(prob_arr_tmp, 0, -1)
                    #print ("mask.shape:", mask.shape)         
                else:
                    prob_arr = prob_arr_tmp
                    
            if verbose:
                print ("prob_path:", prob_path)
                print ("prob_arr.shape:", prob_arr.shape)
            probs.append(prob_arr)
        
        prob_arr_mean = np.mean(probs, axis=0).astype(np.uint8)
        if verbose:
            print ("prob_arr_mean.shape:", prob_arr_mean.shape)
            print ("prob_arr_mean.dtype:", prob_arr_mean.dtype)

        #res_path_geo = os.path.join(root, 'merged', prob_file)
        res_path_geo = os.path.join(out_dir, prob_file)
        if num_classes == 1 or num_classes == 3:
            cv2.imwrite(res_path_geo, prob_arr_mean)
        else:
            # skimage reads in (channels, h, w) for multi-channel
            # assume less than 20 channels
            #print ("mask_channels.shape:", mask_channels.shape)
            if prob_arr_mean.shape[0] > 20: 
                #print ("mask_channels.shape:", mask_channels.shape)
                prob_arr_mean_skimage = np.moveaxis(prob_arr_mean, -1, 0)
            else:
                prob_arr_mean_skimage = prob_arr_mean
            skimage.io.imsave(res_path_geo, prob_arr_mean_skimage, compress=1)
            
            # save gdal too?
            if out_dir_gdal:
                outpath_gdal = os.path.join(out_dir_gdal, prob_file)
                #outpath_gdal = os.path.join(out_dir_gdal, prob_file)
                # want chabnnels first
                # assume less than 20 channels
                if prob_arr_mean.shape[0] > 20: 
                    #print ("mask_channels.shape:", mask_channels.shape)
                    mask_gdal = np.moveaxis(prob_arr_mean, -1, 0)
                    #print ("mask.shape:", mask.shape)         
                else:
                    mask_gdal = prob_arr_mean
                CreateMultiBandGeoTiff(outpath_gdal, mask_gdal)


##############################################################################
def merge_tiffs_defferent_folders(roots, res):
    '''Need to update to handle multiple bands!'''
    os.makedirs(os.path.join(res), exist_ok=True)
    prob_files = {f for f in os.listdir(roots[0]) if os.path.splitext(f)[1] in ['.tif', '.tiff']}

    for prob_file in tqdm(prob_files):
        probs = []
        for root in roots:
            prob_arr = cv2.imread(os.path.join(root, prob_file), cv2.IMREAD_GRAYSCALE)
            probs.append(prob_arr)
        prob_arr = np.mean(probs, axis=0)
        # prob_arr = np.clip(probs[0] * 0.7 + probs[1] * 0.3, 0, 1.)

        res_path_geo = os.path.join(res, prob_file)
        cv2.imwrite(res_path_geo, prob_arr)


##############################################################################
def execute():

    # # if using argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--folds_save_dir', type=str, default='/raid/local/src/apls/albu_inference_mod/results',
    #                         help="path to predicted folds")
    # parser.add_argument('--out_dir', type=str, default='/raid/local/src/apls/albu_inference_mod/results',
    #                         help="path to merged predictions")
    # args = parser.parse_args()
    # #out_dir = os.path.join(os.path.dirname(root), 'merged')
    # os.makedirs(args.out_dir, exist_ok=True)  #os.path.join(root, 'merged'), exist_ok=True)
    
    # t0 = time.time()
    # merge_tiffs(args.folds_save_dir, args.out_dir)
    # t1 = time.time()
    # print ("Time to merge", len(os.listdir(args.folds_save_dir)), "files:", t1-t0, "seconds")
    
    # # compress original folds
    # output_filename = args.folds_save_dir
    # print ("output_filename:", output_filename)
    # shutil.make_archive(output_filename, 'gztar', args.folds_save_dir) #'zip', res_dir)
    # # remove folds
    # #shutil.rmtree(args.folds_save_dir, ignore_errors=True)
    
    # if using config instead of argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path')
    args = parser.parse_args()
    with open(args.config_path, 'r') as f:
        cfg = json.load(f)
        config = Config(**cfg)
    verbose = False

        
    # nothing to do if only one fold
    if config.num_folds == 1:
        print ("num_folds = 1, no need to merge")
        return
        
    folds_dir = os.path.join(config.path_results_root, config.test_results_dir, config.folds_save_dir)
    merge_dir = os.path.join(config.path_results_root, config.test_results_dir, config.merged_dir)
    
    # make gdal folder?
    merge_dir_gdal = merge_dir + '_gdal'
    #merge_dir_gdal = None

    #res_dir = config.folds_save_dir
    #res_dir = os.path.join(config.results_dir, config.folder + config.out_suff + '/folds')
    print ("folds_save_dir used in merge_preds():", folds_dir)

    out_dir = merge_dir
    os.makedirs(out_dir, exist_ok=True)  #os.path.join(root, 'merged'), exist_ok=True)
    # set output dir to: os.path.join(config.results_dir, config.folder + config.out_suff, 'merged')
    print ("out_dir used in merge_preds():", out_dir)

    if merge_dir_gdal:
        os.makedirs(merge_dir_gdal, exist_ok=True)
        
    t0 = time.time()
    merge_tiffs(folds_dir, out_dir, num_classes=config.num_classes,
                out_dir_gdal=merge_dir_gdal,
                verbose=verbose)
    t1 = time.time()
    print ("Time to merge", len(os.listdir(folds_dir)), "files:", t1-t0, "seconds")
    
    #root = '/results/results'
    #merge_tiffs(os.path.join(root, '2m_4fold_512_30e_d0.2_g0.2_test'))

    print ("Compress original folds...")
    output_filename = folds_dir  
    #output_filename = os.path.join(config.results_dir, config.folder + config.out_suff + '/folds')
    print ("output_filename:", output_filename)
    shutil.make_archive(output_filename, 'gztar', folds_dir) #'zip', res_dir)
    # remove folds
    shutil.rmtree(folds_dir, ignore_errors=True)
    
    print ("Compress original gdal folds...")
    output_filename = folds_dir  + '_gdal'
    if os.path.exists(output_filename):
        #output_filename = os.path.join(config.results_dir, config.folder + config.out_suff + '/folds')
        print ("output_filename:", output_filename)
        shutil.make_archive(output_filename, 'gztar', folds_dir + '_gdal') #'zip', res_dir)
        # remove folds
        shutil.rmtree(folds_dir + '_gdal', ignore_errors=True)


##############################################################################
if __name__ == "__main__":
    execute()