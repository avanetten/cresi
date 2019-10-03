import pandas as pd
import numpy as np
import os
import random
import json
import argparse
from random import shuffle
random.seed(42)
from jsons.config import Config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path')
    args = parser.parse_args()
    
    # get config
    with open(args.config_path, 'r') as f:
        cfg = json.load(f)
    config = Config(**cfg)
    
    ###################
    # set paths
    path_images_8bit_train = os.path.join(config.path_data_root, config.train_data_refined_dir_ims)
    # path_images_8bit_train = os.path.join(config.path_data_root, config.train_data_refined_dir, 'images')
    print ("gen+folds.py: path_images_8bit_train:", path_images_8bit_train)
    files = os.listdir(path_images_8bit_train)
    print ("files[:10]:", files[:10])
    weight_save_path = os.path.join(config.path_results_root, 'weights', config.save_weights_dir)
    os.makedirs(weight_save_path, exist_ok=True)
    folds_save_path = os.path.join(weight_save_path, config.folds_file_name)
    
    if os.path.exists(folds_save_path):
        print("folds csv already exists:", folds_save_path)
        return
        
    else:
        print ("folds_save_path:", folds_save_path)

        # # set values
        # if config.num_channels == 3:
        #     image_format_path = 'RGB-PanSharpen'
        # else:
        #     image_format_path = 'MUL-PanSharpen'
        # imfile_prefix = image_format_path + '_'
        ###################    

        shuffle(files)
    
        s = {k.split('_')[0] for k in files}
        d = {k: [v for v in files] for k in s}
    
        folds = {}
    
        if config.num_folds == 1:
            nfolds = int(np.rint(1. / config.default_val_perc))
        else:
            nfolds = config.num_folds
    
        idx = 0
        for v in d.values():
            for val in v:
                folds[val] = idx % nfolds
                idx+=1
    
        df = pd.Series(folds, name='fold')
        df.to_csv(folds_save_path, header=['fold'], index=True)


###############################################################################
if __name__ == "__main__":
    main()
