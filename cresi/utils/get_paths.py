#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 04:37:50 2018

@author: avanetten
"""

        weight_save_path = os.path.join(config.results_dir, 'weights', config.save_weights_name)
        log_train_path = os.path.join(config.results_dir, 'logs')


create_spacenet_masks.py

    # set values
    burnValue = 255
    image_format_path = 'RGB-PanSharpen'
    label_path_extra = 'geojson/spacenetroads'
    geojson_prefix = 'spacenetroads_'
    
    imfile_prefix = image_format_path + '_'


    buffer_meters = float(config.mask_width_m)
    buffer_meters_str = str(np.round(buffer_meters,1)).replace('.', 'p')
    test = not args.training


    paths_data_raw = []
    #############
    # output directories
    if test:
        path_masks =       os.path.join(config.path_data_root, config.data_refined_name+'_test', 'masks{}m'.format(buffer_meters_str))
        path_images_8bit = os.path.join(config.path_data_root, config.data_refined_name+'_test', 'images')
        # set path_data_raw
        for dpart in config.data_train_raw_part.split(','):
            paths_data_raw.append(os.path.join(config.path_data_root,
                                                 dpart))

    else:
        path_masks =       os.path.join(config.path_data_root, config.data_refined_name+'_train', 'masks{}m'.format(buffer_meters_str))
        path_images_8bit = os.path.join(config.path_data_root, config.data_refined_name+'_train', 'images')       
        # set path_data_raw
        for dpart in config.data_test_raw_part.split(','):
            paths_data_raw.append(os.path.join(config.path_data_root,
                                                 dpart))
    #path_masks =       os.path.join(config.path_data_root, config.data_refined_name+'_train' if not test else config.data_refined_name+'_test', 'masks{}m'.format(buffer_meters))
    #path_images_8bit = os.path.join(config.path_data_root, config.data_refined_name+'_train' if not test else config.data_refined_name+'_test', 'images')


gen_folds.py

    path_images_8bit = os.path.join(config.path_data_root, config.data_refined_name+'_train', 'images')       
