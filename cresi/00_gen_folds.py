import pandas as pd
import numpy as np
import os
import random
import json
import argparse
from random import shuffle
random.seed(42)
from configs.config import Config


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
    paths_images_train = config.train_data_refined_dir_ims.split(',')
    print("00_gen_folds.py: path_images_train:", paths_images_train)
    train_files = []
    for p in paths_images_train:
        train_files.extend(os.listdir(p))
    print("train_files[:10]:", train_files[:10])
    weight_save_path = os.path.join(config.path_results_root, 'weights', config.save_weights_dir)
    os.makedirs(weight_save_path, exist_ok=True)
    folds_save_path = os.path.join(weight_save_path, config.folds_file_name)
    
    if os.path.exists(folds_save_path):
        print("folds csv already exists:", folds_save_path)
        return
        
    else:
        print ("folds_save_path:", folds_save_path)

        shuffle(train_files)
    
        s = {k.split('_')[0] for k in train_files}
        d = {k: [v for v in train_files] for k in s}
    
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
