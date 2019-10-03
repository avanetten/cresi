#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 16:34:02 2019

@author: avanetten
"""
# plot graph on image, plus ground_truth

import os
import cv2
import sys
import math
import shapely
import warnings
import importlib
import skimage.io
import scipy.misc 
import numpy as np
import osmnx as ox
import networkx as nx
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

# local?
local = True
if local:
    cresi_dir = '/raid/cosmiq/cresi/src'
    apls_dir = '/raid/cosmiq/apls/apls/src'
# dev box
else:
    cresi_dir = '/raid/local/src/cresi/src'
    apls_dir = '/raid/local/src/apls/apls/src'
sys.path.append(cresi_dir)
sys.path.append(apls_dir)
#from other_tools import apls_tools
import apls
import apls_tools 
#import speed
#import skeleton_speed
import graphTools
importlib.reload(graphTools)
importlib.reload(apls_tools)
importlib.reload(apls)

image_dir = '/raid/cosmiq/spacenet/data/spacenetv2/basiss_rgb_8bit_test_400m/images'
geojson_dir_gt = '/raid/cosmiq/spacenet/data/spacenetv2/basiss_rgb_8bit_test_400m/geojson/spacenetroads_noveau'
res_dir = '/raid/cosmiq/cresi/results/resnet34_ave_speed_mc_focal_totband_test_sn3chips'
graph_dir_out = os.path.join(res_dir, 'graphs_speed')
plot_out_dir = os.path.join(res_dir, 'graphs_speed_plots')

print ("plot_out_dir:", plot_out_dir)
os.makedirs(plot_out_dir, exist_ok=True)
N_plots = 6
verbose = False


#width_key, width_mult = 'travel_time', 2
width_key, width_mult = 'speed_m/s', 0.5
#width_key, width_mult = 'speed_mph', 0.3
#width_key, width_mult = 4, 1   # constant widths

gt_color, prop_color = 'cyan', 'lime'

# ground truth props
osmidx, osmNodeidx = 0, 0
speed_key = 'speed_m/s'
travel_time_key = 'travel_time'
weight = 'travel_time'  # 'length'
gt_subgraph_filter_weight = 'length'
gt_min_subgraph_length = 5
prop_subgraph_filter_weight = 'length_pix'
prop_min_subgraph_length = 10 # GSD = 0.3
default_speed=13.41
use_pix_coords = False


#image_list0 = ['RGB-PanSharpen_AOI_4_Shanghai_img555.tif', 'RGB-PanSharpen_AOI_2_Vegas_img1056.tif',  'RGB-PanSharpen_AOI_2_Vegas_img500.tif']
image_list0 = ['RGB-PanSharpen_AOI_2_Vegas_img232.tif']  
#image_list0 = [ 'RGB-PanSharpen_AOI_2_Vegas_img1005.tif']

image_list = os.listdir(image_dir)
N_extra = N_plots - len(image_list0)
image_names = image_list0 + list(np.random.choice(image_list, N_extra))
#print ("image_names:", image_names)

for i,image_name in enumerate(image_names):
    print ("image_name:", image_name)
    image_root = image_name.split('RGB-PanSharpen_')[-1].split('.tif')[0]
    image_path = os.path.join(image_dir, image_name)
    pkl_prop_path = os.path.join(graph_dir_out, image_name.split('.')[0] + '.gpickle')
    figname = os.path.join(plot_out_dir, image_name)
    if not os.path.exists(pkl_prop_path):
        continue
    print ("load proposal...")
    G_prop = nx.read_gpickle(pkl_prop_path)
    
    # get ground truth graph from geojson
    print ("load ground truth from geojson...")
    gt_path = os.path.join(geojson_dir_gt, 'spacenetroads_' + image_root + '.geojson')
    if verbose:
        print ("gt_path:", gt_path)
    # ground truth
    G_gt, _ = \
        apls.create_gt_graph(gt_path, image_path, network_type='all_private',
             valid_road_types=set([]),
             subgraph_filter_weight=gt_subgraph_filter_weight,
             min_subgraph_length=gt_min_subgraph_length,
             use_pix_coords=use_pix_coords,
             osmidx=osmidx, osmNodeidx=osmNodeidx,
             speed_key=speed_key,
             travel_time_key=travel_time_key,
             verbose=verbose)

    if verbose:
        # print a node?
        node_tmp = list(G_gt.nodes())[-1]
        print (node_tmp, "random node props:", G_gt.nodes[node_tmp])
        # print an edge
        edge_tmp = list(G_gt.edges())[-1]
        print ("random edge props for edge:", edge_tmp, " = ", 
                   G_gt.edges[edge_tmp[0], edge_tmp[1], 0]) #G.edge[edge_tmp[0]][edge_tmp[1]])
        geom_latlon =  G_gt.edges[edge_tmp[0], edge_tmp[1], 0]['geometry_latlon']
        print ("geom_latlon:", geom_latlon)
        geom_wkt =  G_gt.edges[edge_tmp[0], edge_tmp[1], 0]['geometry']
        print ("geom_wkt:", geom_wkt)
        geom_pix =  G_gt.edges[edge_tmp[0], edge_tmp[1], 0]['geometry_pix']
        print ("geom_pix:", geom_pix)
    
        #print (image_name)
        #print (G_gt.nodes)
    
    _ = apls_tools.plot_gt_prop_graphs(G_gt, G_prop, image_path, figsize=(16, 8), 
                      show_endnodes=True,
                      width_key=width_key, width_mult=width_mult,
                      gt_color=gt_color, prop_color=prop_color, 
                      title=image_name, adjust=False, 
                      figname=figname, verbose=verbose)
    
plt.show()


'''

python /raid/cosmiq/cresi/src/plot_graphs_on_im.py

'''