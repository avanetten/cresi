#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 14:29:41 2020

@author: avanetten
"""

import os
import shutil
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

###############################################################################
def split_data(im_dir, mask_dir, test_frac=0.2,
               outfile_csv='', 
               outdir_im_train='', outdir_mask_train='',
               outdir_im_test='', outdir_mask_test=''
               # replace_dict={'PS-RGB': 'mask_intersections'}
               ):
    '''Split imagery and masks into train/test directories'''

    os.makedirs(outdir_im_train, exist_ok=True)
    os.makedirs(outdir_mask_train, exist_ok=True)
    os.makedirs(outdir_im_test, exist_ok=True)
    os.makedirs(outdir_mask_test, exist_ok=True)
    
    im_list = sorted([z for z in os.listdir(im_dir) if z.endswith('.tif')])
    idxs = range(len(im_list))
    idx_train, idx_test = train_test_split(idxs, test_size=test_frac)
    
    columns = ['idx', 'im_name', 'im_path', 'mask_path', 
                'outdir_im', 'outdir_mask']
    out_arr = []
    print("len im_list:", len(im_list))
    for i, im_name in enumerate(im_list):
        if (i % 100) == 0:
            print(i, im_name)
            
        im_path = os.path.join(im_dir, im_name)
        # intersection_name = image_name.replace('PS-RGB', 'mask_intersections').replace('PS-MS', 'mask_intersections')      
        mask_path = os.path.join(mask_dir, im_name)
        
        # set outputs
        if i in idx_train:
            outdir_im = outdir_im_train
            outdir_mask = outdir_mask_train
        else:
            outdir_im = outdir_im_test
            outdir_mask = outdir_mask_test
        
        # copy to destination
        shutil.copy(im_path, outdir_im)
        shutil.copy(mask_path, outdir_mask)
        
        out_arr.append([i, im_name, im_path, mask_path, outdir_im, outdir_mask])
            
    df = pd.DataFrame(out_arr, columns=columns)
    print("df:", df)
    df.to_csv(outfile_csv)
    
    return

    
##############################################################################
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--im_dir', type=str, default='')
    parser.add_argument('--mask_dir', type=str, default='')
    parser.add_argument('--test_frac', type=float, default=0.2)
    parser.add_argument('--outfile_csv', type=str, default='')
    parser.add_argument('--outdir_im_train', type=str, default='')
    parser.add_argument('--outdir_mask_train', type=str, default='')
    parser.add_argument('--outdir_im_test', type=str, default='')
    parser.add_argument('--outdir_mask_test', type=str, default='')
    args = parser.parse_args()

    if os.path.exists(args.outfile_csv):
        print(args.outfile_csv, "already exists, no need to split")
    else:
        split_data(args.im_dir, args.mask_dir, 
               test_frac=args.test_frac,
               outfile_csv=args.outfile_csv, 
               outdir_im_train=args.outdir_im_train, 
               outdir_mask_train=args.outdir_mask_train,
               outdir_im_test=args.outdir_im_test, 
               outdir_mask_test=args.outdir_mask_test)


##############################################################################
if __name__ == "__main__":
    main()
