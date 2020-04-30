#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun May  6 08:51:36 2018

@author: avanetten

Inspired by:
    https://github.com/SpaceNetChallenge/RoadDetector/blob/master/albu-solution/src/skeleton.py
"""

from skimage.morphology import skeletonize, remove_small_objects, remove_small_holes
from skimage.morphology import erosion, dilation, opening, closing, disk
from skimage.feature import blob_dog, blob_log, blob_doh
import numpy as np
from scipy import ndimage as ndi
from matplotlib.pylab import plt
from utils import sknw, sknw_int64
import os
import pandas as pd
from itertools import tee
from scipy.spatial.distance import pdist, squareform
from collections import OrderedDict #, defaultdict
import json
import time
import random
import argparse
import networkx as nx
import logging
from multiprocessing.pool import Pool
import skimage
import skimage.draw
import scipy.spatial
import skimage.io 
import cv2

from utils import make_logger, medial_axis_weight
from configs.config import Config
wkt_to_G = __import__('05_wkt_to_G')

logger1 = None
linestring = "LINESTRING {}"



# from apls.py
###############################################################################
def clean_sub_graphs(G_, min_length=150, max_nodes_to_skip=100,
                     weight='length_pix', verbose=True,
                     super_verbose=False):
    '''Remove subgraphs with a max path length less than min_length,
    if the subgraph has more than max_noxes_to_skip, don't check length 
       (this step great improves processing time)'''
    
    if len(G_.nodes()) == 0:
        return G_
    
    if verbose:
        print("Running clean_sub_graphs...")
    try:
        sub_graphs = list(nx.connected_component_subgraphs(G_))
    except:
        sub_graph_nodes = nx.connected_components(G_)
        sub_graphs = [G_.subgraph(c).copy() for c in sub_graph_nodes]
    
    if verbose:
        print("  N sub_graphs:", len([z.nodes for z in sub_graphs]))
        
    bad_nodes = []
    if verbose:
        print(" len(G_.nodes()):", len(G_.nodes()) )
        print(" len(G_.edges()):", len(G_.edges()) )
    if super_verbose:
        print("G_.nodes:", G_.nodes())
        edge_tmp = G_.edges()[np.random.randint(len(G_.edges()))]
        print(edge_tmp, "G.edge props:", G_.edge[edge_tmp[0]][edge_tmp[1]])

    for G_sub in sub_graphs:
        # don't check length if too many nodes in subgraph
        if len(G_sub.nodes()) > max_nodes_to_skip:
            continue
        
        else:
            all_lengths = dict(nx.all_pairs_dijkstra_path_length(G_sub, weight=weight))
            if super_verbose:
                print("  \nGs.nodes:", G_sub.nodes() )
                print("  all_lengths:", all_lengths )
            # get all lenghts
            lens = []
            #for u,v in all_lengths.iteritems():
            for u in all_lengths.keys():
                v = all_lengths[u]
                #for uprime, vprime in v.iteritems():
                for uprime in v.keys():
                    vprime = v[uprime]
                    lens.append(vprime)
                    if super_verbose:
                        print("  u, v", u,v )
                        print("    uprime, vprime:", uprime, vprime )
            max_len = np.max(lens)
            if super_verbose:
                print("  Max length of path:", max_len)
            if max_len < min_length:
                bad_nodes.extend(G_sub.nodes())
                if super_verbose:
                    print(" appending to bad_nodes:", G_sub.nodes())

    # remove bad_nodes
    G_.remove_nodes_from(bad_nodes)
    if verbose:
        print(" num bad_nodes:", len(bad_nodes))
        #print("bad_nodes:", bad_nodes)
        print(" len(G'.nodes()):", len(G_.nodes()))
        print(" len(G'.edges()):", len(G_.edges()))
    if super_verbose:
        print("  G_.nodes:", G_.nodes())
        
    return G_


# From road_raster.py
###############################################################################
def dl_post_process_pred(mask, glob_thresh=80, kernel_size=9,
                         min_area=2000, contour_smoothing=0.001,
                         adapt_kernel=85, adapt_const=-3,
                         use_glob_thresh=False,
                         kernel_open=19, verbose=False):
    '''Refine mask file and return both refined mask and skeleton'''
  
    t0 = time.time()
    kernel_blur = kernel_size #9
    kernel_close = kernel_size #9
    #kernel_open = kernel_size #9

    kernel_close = np.ones((kernel_close,kernel_close), np.uint8)
    kernel_open = np.ones((kernel_open, kernel_open), np.uint8)
        
    blur = cv2.medianBlur(mask, kernel_blur)
    
    # global thresh
    glob_thresh_arr = cv2.threshold(blur, glob_thresh, 1, cv2.THRESH_BINARY)[1]
    glob_thresh_arr_smooth = cv2.medianBlur(glob_thresh_arr, kernel_blur)

    ## skimage
    #block_size = 81
    #mask_thresh = threshold_adaptive(mask, block_size, offset=10)
    #mask_thresh= mask_thresh.astype(int)
    #plt.imshow(mask_thresh)
    
    t1 = time.time()
    if verbose:
        print("Time to compute open(), close(), and get thresholds:", t1-t0, "seconds")
    
    if use_glob_thresh:
        mask_thresh = glob_thresh_arr_smooth
    else:
        # adaptive thresholding
    #    Python: cv2.adaptiveThreshold(src, maxValue, adaptiveMethod, thresholdType, blockSize, C[, dst]) → dst
    #    Parameters:
    #    src – Source 8-bit single-channel image.
    #    dst – Destination image of the same size and the same type as src .
    #    maxValue – Non-zero value assigned to the pixels for which the condition is satisfied. See the details below.
    #    adaptiveMethod – Adaptive thresholding algorithm to use, ADAPTIVE_THRESH_MEAN_C or ADAPTIVE_THRESH_GAUSSIAN_C . See the details below.
    #    thresholdType – Thresholding type that must be either THRESH_BINARY or THRESH_BINARY_INV .
    #    blockSize – Size of a pixel neighborhood that is used to calculate a threshold value for the pixel: 3, 5, 7, and so on.
    #    C – Constant subtracted from the mean or weighted mean (see the details below). Normally, it is positive but may be zero or negative as well.
        adapt_thresh = cv2.adaptiveThreshold(mask,1,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                    cv2.THRESH_BINARY,adapt_kernel, adapt_const)
        #adapt_kernel = 501
        #adapt_thresh = cv2.adaptiveThreshold(mask,150,cv2.ADAPTIVE_THRESH_MEAN_C,\
        #            cv2.THRESH_BINARY,adapt_kernel,2)
        # resmooth
        adapt_thresh_smooth = cv2.medianBlur(adapt_thresh, kernel_blur)

        mask_thresh = adapt_thresh_smooth 

    
    ## background equalization
    ## https://stackoverflow.com/questions/39231534/get-darker-lines-of-an-image-using-opencv
    #max_value = np.max(mask)
    #backgroundRemoved = mask.astype(float)
    #bg_kern = 101
    #blur_eq = cv2.GaussianBlur(backgroundRemoved, (bg_kern, bg_kern), 200)
    #backgroundRemoved = backgroundRemoved/blur_eq
    #backgroundRemoved = (backgroundRemoved*max_value/np.max(backgroundRemoved)).astype(np.uint8)
    #closing_bg = cv2.morphologyEx(backgroundRemoved, cv2.MORPH_CLOSE, kernel_close)
    
    # opening and closing
    # http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
    #gradient = cv2.morphologyEx(mask_thresh, cv2.MORPH_GRADIENT, kernel)
    closing = cv2.morphologyEx(mask_thresh, cv2.MORPH_CLOSE, kernel_close)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel_open)
    # try on bgRemoved?
    
    t2 = time.time()
    if verbose:
        print("Time to compute adaptive_thresh, second open(), close():", t2-t1, "seconds")

    # set output
    if contour_smoothing < 0:
        final_mask = opening
    else:
        # contours
        # remove small items
        contours, cont_plot, hole_idxs = get_contours_complex(opening, 
                                            min_thresh=glob_thresh, 
                                           min_area=min_area, 
                                           contour_smoothing=contour_smoothing)
        #contours, cont_plot = get_contours(opening, min_thresh=glob_thresh, 
        #                                   min_area=min_area, 
        #                                   contour_smoothing=contour_smoothing)
    
        # for some reason contours don't extend to the edge, so clip the edge
        # and resize
        mask_filt_raw = get_mask(mask_thresh, cont_plot, hole_idxs=hole_idxs)
        shape_tmp = mask_filt_raw.shape
        mask_filt1 = 200 * cv2.resize(mask_filt_raw[2:-2, 2:-2], shape_tmp).astype(np.uint8)
        if verbose:
            print("mask:", mask)
            print("mask.dtype:", mask.dtype)
            print("mask_fi1t1.dtype:", mask_filt1.dtype)
            print("mask.shape == mask_filt1.shape:", mask.shape == mask_filt1.shape )
            print("mask_filt1.shape:", mask_filt1.shape)
            print("mask_filt1", mask_filt1)
        # thresh and resmooth
        mask_filt = cv2.GaussianBlur(mask_filt1, (kernel_blur, kernel_blur), 0)
        #mask_filt = cv2.threshold(mask_filt2, glob_thresh, 1, cv2.THRESH_BINARY)
        final_mask = mask_filt    

    t3 = time.time()
    if verbose:
        print("Time to smooth contours:", t3-t2, "seconds")
    
    # skeletonize
    #medial = medial_axis(final_mask)
    #medial_int = medial.astype(np.uint8)
    medial_int = medial_axis(final_mask).astype(np.uint8)
    if verbose:
        print("Time to compute medial_axis:", time.time() - t3, "seconds")
        print("Time to run dl_post_process_pred():", time.time() - t0, "seconds")
    
    return final_mask, medial_int


###############################################################################
def cv2_skeletonize(img):
    """ OpenCV function to return a skeletonized version of img, a Mat object
    https://gist.github.com/jsheedy/3913ab49d344fac4d02bcc887ba4277d"""
    #  hat tip to http://felix.abecassis.me/2011/09/opencv-morphological-skeleton/

    img = img.copy() # don't clobber original
    skel = img.copy()

    skel[:,:] = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))

    while True:
        eroded = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)
        temp = cv2.morphologyEx(eroded, cv2.MORPH_DILATE, kernel)
        temp  = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img[:,:] = eroded[:,:]
        if cv2.countNonZero(img) == 0:
            break

    return skel


###############################################################################
def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


###############################################################################
def remove_sequential_duplicates(seq):
    # todo
    res = [seq[0]]
    for elem in seq[1:]:
        if elem == res[-1]:
            continue
        res.append(elem)
    return res


###############################################################################
def remove_duplicate_segments(seq):
    seq = remove_sequential_duplicates(seq)
    segments = set()
    split_seg = []
    res = []
    for idx, (s, e) in enumerate(pairwise(seq)):
        if (s, e) not in segments and (e, s) not in segments:
            segments.add((s, e))
            segments.add((e, s))
        else:
            split_seg.append(idx+1)
    for idx, v in enumerate(split_seg):
        if idx == 0:
            res.append(seq[:v])
        if idx == len(split_seg) - 1:
            res.append(seq[v:])
        else:
            s = seq[split_seg[idx-1]:v]
            if len(s) > 1:
                res.append(s)
    if not len(split_seg):
        res.append(seq)
    return res


###############################################################################
def flatten(l):
    return [item for sublist in l for item in sublist]


###############################################################################
def get_angle(p0, p1=np.array([0, 0]), p2=None):
    """ compute angle (in degrees) for p0p1p2 corner
    Inputs:
        p0,p1,p2 - points in the form of [x,y]
    """
    if p2 is None:
        p2 = p1 + np.array([1, 0])
    v0 = np.array(p0) - np.array(p1) 
    v1 = np.array(p2) - np.array(p1)

    angle = np.math.atan2(np.linalg.det([v0,v1]),np.dot(v0,v1))
    return np.degrees(angle)


###############################################################################
def preprocess(img, thresh, img_mult=255, hole_size=300,
               cv2_kernel_close=7, cv2_kernel_open=7, verbose=False):
    '''
    http://scikit-image.org/docs/dev/api/skimage.morphology.html#skimage.morphology.remove_small_holes
    hole_size in remove_small_objects is the maximum area, in pixels of the
    hole
    '''

    # sometimes get a memory error with this approach
    if img.size < 10000000000:
    # if img.size < 0:
        if verbose:
            print("Run preprocess() with skimage")
        img = (img > (img_mult * thresh)).astype(np.bool)        
        remove_small_objects(img, hole_size, in_place=True)
        remove_small_holes(img, hole_size, in_place=True)
        # img = cv2.dilate(img.astype(np.uint8), np.ones((7, 7)))

    # cv2 is generally far faster and more memory efficient (though less
    #  effective)
    else:
        if verbose:
            print("Run preprocess() with cv2")

        #from road_raster.py, dl_post_process_pred() function
        kernel_close = np.ones((cv2_kernel_close, cv2_kernel_close), np.uint8)
        kernel_open = np.ones((cv2_kernel_open, cv2_kernel_open), np.uint8)
        kernel_blur = cv2_kernel_close
   
        # global thresh
        #mask_thresh = (img > (img_mult * thresh))#.astype(np.bool)        
        blur = cv2.medianBlur( (img * img_mult).astype(np.uint8), kernel_blur)
        glob_thresh_arr = cv2.threshold(blur, thresh, 1, cv2.THRESH_BINARY)[1]
        glob_thresh_arr_smooth = cv2.medianBlur(glob_thresh_arr, kernel_blur)
        mask_thresh = glob_thresh_arr_smooth      
    
        # opening and closing
        # http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
        #gradient = cv2.morphologyEx(mask_thresh, cv2.MORPH_GRADIENT, kernel)
        closing_t = cv2.morphologyEx(mask_thresh, cv2.MORPH_CLOSE, kernel_close)
        opening_t = cv2.morphologyEx(closing_t, cv2.MORPH_OPEN, kernel_open)
        img = opening_t.astype(np.bool)
        #img = opening

    return img

###############################################################################
def graph2lines(G):
    node_lines = []
    edges = list(G.edges())
    if len(edges) < 1:
        return []
    prev_e = edges[0][1]
    current_line = list(edges[0])
    added_edges = {edges[0]}
    for s, e in edges[1:]:
        if (s, e) in added_edges:
            continue
        if s == prev_e:
            current_line.append(e)
        else:
            node_lines.append(current_line)
            current_line = [s, e]
        added_edges.add((s, e))
        prev_e = e
    if current_line:
        node_lines.append(current_line)
    return node_lines


###############################################################################
def visualize(img, G, vertices):
    plt.imshow(img, cmap='gray')

    # draw edges by pts
    for (s, e) in G.edges():
        vals = flatten([[v] for v in G[s][e].values()])
        for val in vals:
            ps = val.get('pts', [])
            plt.plot(ps[:, 1], ps[:, 0], 'green')

    # draw node by o
    node, nodes = G.node(), G.nodes
    # deg = G.degree
    # ps = np.array([node[i]['o'] for i in nodes])
    ps = np.array(vertices)
    plt.plot(ps[:, 1], ps[:, 0], 'r.')

    # title and show
    plt.title('Build Graph')
    plt.show()


###############################################################################
def line_points_dist(line1, pts):
    return np.cross(line1[1] - line1[0], pts - line1[0]) / np.linalg.norm(line1[1] - line1[0])


###############################################################################
def remove_small_terminal(G, weight='weight', min_weight_val=30, 
                          pix_extent=1300, edge_buffer=4, verbose=False):
    '''Remove small terminals, if a node in the terminal is within edge_buffer
    of the the graph edge, keep it'''
    deg = dict(G.degree())
    terminal_points = [i for i, d in deg.items() if d == 1]
    if verbose:
        print("remove_small_terminal() - N terminal_points:", len(terminal_points))
    edges = list(G.edges())
    for s, e in edges:
        if s == e:
            sum_len = 0
            vals = flatten([[v] for v in G[s][s].values()])
            for ix, val in enumerate(vals):
                sum_len += len(val['pts'])
            if sum_len < 3:
                G.remove_edge(s, e)
                continue
            
        # check if at edge
        sx, sy = G.nodes[s]['o']
        ex, ey = G.nodes[e]['o']
        edge_point = False
        for ptmp in [sx, sy, ex, ey]:
            if (ptmp < (0 + edge_buffer)) or (ptmp > (pix_extent - edge_buffer)):
                if verbose:
                    print("ptmp:", ptmp)
                    print("(pix_extent - edge_buffer):", (pix_extent - edge_buffer))
                    print("(ptmp > (pix_extent - edge_buffer):", (ptmp > (pix_extent - edge_buffer)))
                    print("ptmp < (0 + edge_buffer):", (ptmp < (0 + edge_buffer)))
                edge_point = True
            else:
                continue
        # don't remove edges near the edge of the image
        if edge_point:
            if verbose:
                print("(pix_extent - edge_buffer):", (pix_extent - edge_buffer))
                print("edge_point:", sx, sy, ex, ey, "continue")
            continue

        vals = flatten([[v] for v in G[s][e].values()])
        for ix, val in enumerate(vals):
            if verbose:
                print("val.get(weight, 0):", val.get(weight, 0) )
            if s in terminal_points and val.get(weight, 0) < min_weight_val:
                G.remove_node(s)
            if e in terminal_points and val.get(weight, 0) < min_weight_val:
                G.remove_node(e)
    return


###############################################################################
def add_direction_change_nodes(pts, s, e, s_coord, e_coord):
    if len(pts) > 3:
        ps = pts.reshape(pts.shape[0], 1, 2).astype(np.int32)
        approx = 2
        ps = cv2.approxPolyDP(ps, approx, False)
        ps = np.squeeze(ps, 1)
        st_dist = np.linalg.norm(ps[0] - s_coord)
        en_dist = np.linalg.norm(ps[-1] - s_coord)
        if st_dist > en_dist:
            s, e = e, s
            s_coord, e_coord = e_coord, s_coord
        ps[0] = s_coord
        ps[-1] = e_coord
    else:
        ps = np.array([s_coord, e_coord], dtype=np.int32)
    return ps


###############################################################################
def add_small_segments(G, terminal_points, terminal_lines, 
                       dist1=24, dist2=80,
                       angle1=30, angle2=150, 
                       verbose=False):
    '''Connect small, missing segments
    terminal points are the end of edges.  This function tries to pair small
    gaps in roads.  It will not try to connect a missed T-junction, as the 
    crossroad will not have a terminal point'''
    
    print("Running add_small_segments()")
    try:
        node = G.node
    except:
        node = G.nodes
    # if verbose:
    #   print("node:", node)

    term = [node[t]['o'] for t in terminal_points]
    # print("term:", term)
    dists = squareform(pdist(term))
    possible = np.argwhere((dists > 0) & (dists < dist1))
    good_pairs = []
    for s, e in possible:
        if s > e:
            continue
        s, e = terminal_points[s], terminal_points[e]

        if G.has_edge(s, e):
            continue
        good_pairs.append((s, e))

    possible2 = np.argwhere((dists > dist1) & (dists < dist2))
    for s, e in possible2:
        if s > e:
            continue
        s, e = terminal_points[s], terminal_points[e]
        if G.has_edge(s, e):
            continue
        l1 = terminal_lines[s]
        l2 = terminal_lines[e]
        d = line_points_dist(l1, l2[0])

        if abs(d) > dist1:
            continue
        angle = get_angle(l1[1] - l1[0], np.array((0, 0)), l2[1] - l2[0])
        if (-1*angle1 < angle < angle1) or (angle < -1*angle2) or (angle > angle2):
            good_pairs.append((s, e))

    if verbose:
        print("  good_pairs:", good_pairs)
        
    dists = {}
    for s, e in good_pairs:
        s_d, e_d = [G.nodes[s]['o'], G.nodes[e]['o']]
        # print("s_d", s_d)
        # print("type s_d", type(s_d))
        # print("s_d - e_d", s_d - e_d)
        # return
        dists[(s, e)] = np.linalg.norm(s_d - e_d)

    dists = OrderedDict(sorted(dists.items(), key=lambda x: x[1]))

    wkt = []
    added = set()
    good_coords = []
    for s, e in dists.keys():
        if s not in added and e not in added:
            added.add(s)
            added.add(e)
            s_d, e_d = G.nodes[s]['o'].astype(np.int32), G.nodes[e]['o'].astype(np.int32)
            line_strings = ["{1:.1f} {0:.1f}".format(*c.tolist()) for c in [s_d, e_d]]
            line = '(' + ", ".join(line_strings) + ')'
            wkt.append(linestring.format(line))
            good_coords.append( (tuple(s_d), tuple(e_d)) )
    return wkt, good_pairs, good_coords


###############################################################################
def make_skeleton(img_loc, thresh, debug, fix_borders, replicate=5,
                  clip=2, img_shape=(1300, 1300), img_mult=255, hole_size=300,
                  cv2_kernel_close=7, cv2_kernel_open=7,
                  use_medial_axis=False,
                  max_out_size=(200000, 200000),
                  num_classes=1,
                  skeleton_band='all',
                  kernel_blur=27,
                  min_background_frac=0.2,
                  verbose=False
                  ):
    '''
    Extract a skeleton from a mask.
    skeleton_band is the index of the band of the mask to use for 
        skeleton extraction, set to string 'all' to use all bands
    '''
    
    if verbose:
        print("Executing make_skeleton...")
    t0 = time.time()
    #replicate = 5
    #clip = 2
    rec = replicate + clip
    weight_arr = None


    # read in data
    if num_classes == 1:
        try:
            img = cv2.imread(img_loc, cv2.IMREAD_GRAYSCALE)
        except:
            img = skimage.io.imread(img_loc, as_gray=True).astype(np.uint8)#[::-1]
            
    else:
        # ensure 8bit?
        img_tmp = skimage.io.imread(img_loc).astype(np.uint8)
        # we want skimage to read in (channels, h, w) for multi-channel
        #   assume less than 20 channels
        if img_tmp.shape[0] > 20: 
            img_full = np.moveaxis(img_tmp, 0, -1)
        else:
            img_full = img_tmp
        # select the desired band for skeleton extraction
        #  if < 0, sum all bands
        if type(skeleton_band) == str:  #skeleton_band < 0:
            img = np.sum(img_full, axis=0).astype(np.int8)
        else:
            img = img_full[skeleton_band, :, :]
    if verbose:
        print("make_skeleton(), input img_shape:", img_shape)
        print("make_skeleton(), img.shape:", img.shape)
        print("make_skeleton(), img.size:", img.size)
        print("make_skeleton(), img dtype:", img.dtype)
        #print("make_skeleton(), img unique:", np.unique(img))

    ##########
    # potentially keep only subset of data 
    shape0 = img.shape
    img = img[:max_out_size[0], :max_out_size[1]]
    if img.shape != shape0:
        print("Using only subset of data!!!!!!!!")
        print("make_skeletion() new img.shape:", img.shape)
    ##########

    #if len(img_shape) > 0:
    #    assert img.shape == img_shape #(1300, 1300)
    
    if fix_borders:
        img = cv2.copyMakeBorder(img, replicate, replicate, replicate, 
                                 replicate, cv2.BORDER_REPLICATE)        
    img_copy = None
    if debug:
        if fix_borders:
            img_copy = np.copy(img[replicate:-replicate,replicate:-replicate])
        else:
            img_copy = np.copy(img)
        
    if verbose:
        print("Run preprocess()...")
    t1 = time.time()
    img = preprocess(img, thresh, img_mult=img_mult, hole_size=hole_size,
                     cv2_kernel_close=cv2_kernel_close, 
                     cv2_kernel_open=cv2_kernel_open)
    
    # img, _ = dl_post_process_pred(img)
    
    t2 = time.time()
    if verbose:
        print("Time to run preprocess():", t2-t1, "seconds")
    if not np.any(img):
        return None, None
    
    if not use_medial_axis:
        if verbose:
            print("skeletonize...")
        ske = skeletonize(img).astype(np.uint16)
        t3 = time.time()
        if verbose:
            print("Time to run skimage.skeletonize():", t3-t2, "seconds")

    else:
        if verbose:
            print("running updated skimage.medial_axis...")
        ske = medial_axis_weight.medial_axis_weight(
                img, weight_arr=weight_arr,
                return_distance=False).astype(np.uint16)
        # ske = skimage.morphology.medial_axis(img).astype(np.uint16)
        t3 = time.time()
        if verbose:
            print("Time to run medial_axis_weight():", t3-t2, "seconds")
        # print("Time to run skimage.medial_axis():", t3-t2, "seconds")

    if fix_borders:
        if verbose:
            print("fix_borders...")
        ske = ske[rec:-rec, rec:-rec]
        ske = cv2.copyMakeBorder(ske, clip, clip, clip, clip, cv2.BORDER_CONSTANT, value=0)
        # ske = ske[replicate:-replicate,replicate:-replicate]  
        img = img[replicate:-replicate,replicate:-replicate]
        t4 = time.time()
        if verbose:
            print("Time fix borders:", t4-t3, "seconds")
    
    t1 = time.time()
    if verbose:
        print("ske.shape:", ske.shape)
        print("Time to run make_skeleton:", t1-t0, "seconds")
    
    #print("make_skeletion(), ske.shape:", ske.shape)
    #print("make_skeletion(), ske.size:", ske.size)
    #print("make_skeletion(), ske dtype:", ske.dtype)
    #print("make_skeletion(), ske unique:", np.unique(ske))
    #return

    return img, ske


###############################################################################
def img_to_ske_G(params):

    img_loc, out_ske_file, out_gpickle, thresh, \
                debug, fix_borders, \
                img_shape,\
                skel_replicate, skel_clip, \
                img_mult, hole_size, \
                cv2_kernel_close, cv2_kernel_open,\
                min_subgraph_length_pix,\
                min_spur_length_pix,\
                max_out_size,\
                use_medial_axis,\
                num_classes,\
                skeleton_band, \
                kernel_blur,\
                min_background_frac,\
                verbose\
        = params

    # create skeleton
    img_refine, ske = make_skeleton(img_loc, thresh, debug, fix_borders, 
                      replicate=skel_replicate, clip=skel_clip, 
                      img_shape=img_shape, 
                      img_mult=img_mult, hole_size=hole_size,
                      cv2_kernel_close=cv2_kernel_close,
                      cv2_kernel_open=cv2_kernel_open,
                      max_out_size=max_out_size,
                      skeleton_band=skeleton_band,
                      num_classes=num_classes,
                      use_medial_axis=use_medial_axis,
                      kernel_blur=kernel_blur,
                      min_background_frac=min_background_frac,
                      verbose=verbose)

    # print("img_loc:", img_loc)

    #img_copy, ske = make_skeleton(root, fn, debug, threshes, fix_borders)
    if ske is None:
        return [linestring.format("EMPTY"), [], []]
    # save to file
    if out_ske_file:
        cv2.imwrite(out_ske_file, ske.astype(np.uint8)*255)
        ## write portion to file?
        #out_ske_part = out_ske_file.split('.tif')[0] + '_part.tif'
        #print('out_ske_part_path', out_ske_part)
        #cv2.imwrite(out_ske_part, ske[0:2000, 0:2000].astype(np.uint8)*220)
    
    # create graph
    if verbose:
        print("Execute sknw...")
    # if the file is too large, use sknw_int64 to accomodate high numbers
    #   for coordinates
    if np.max(ske.shape) > 32767:
        G = sknw_int64.build_sknw(ske, multi=True)
    else:
        G = sknw.build_sknw(ske, multi=True)

   # print a random node and edge
    if verbose:
        node_tmp = list(G.nodes())[-1]
        print(node_tmp, "random node props:", G.nodes[node_tmp])
        # print an edge
        edge_tmp = list(G.edges())[-1]
        #print("random edge props for edge:", edge_tmp, " = ",
        #      G.edges[edge_tmp[0], edge_tmp[1], 0]) #G.edge[edge_tmp[0]][edge_tmp[1]])

    # iteratively clean out small terminals
    for itmp in range(8):
        ntmp0 = len(G.nodes())
        if verbose:
            print("Clean out small terminals - round", itmp)
            print("Clean out small terminals - round", itmp, "num nodes:", ntmp0)
        # sknw attaches a 'weight' property that is the length in pixels
        pix_extent = np.max(ske.shape)
        remove_small_terminal(G, weight='weight',
                              min_weight_val=min_spur_length_pix,
                              pix_extent=pix_extent)
        # kill the loop if we stopped removing nodes
        ntmp1 = len(G.nodes())
        if ntmp0 == ntmp1:
            break
        else:
            continue

    if verbose:
        print("len G.nodes():", len(G.nodes()))
        print("len G.edges():", len(G.edges()))
    if len(G.edges()) == 0:
        return [linestring.format("EMPTY"), [], []]

    # print a random node and edge
    if verbose:
        node_tmp = list(G.nodes())[-1]
        print(node_tmp, "random node props:", G.nodes[node_tmp])
        # print an edge
        edge_tmp = list(G.edges())[-1]
        print("random edge props for edge:", edge_tmp, " = ",
              G.edges[edge_tmp[0], edge_tmp[1], 0]) #G.edge[edge_tmp[0]][edge_tmp[1]])
        # node_tmp = list(G.nodes())[np.random.randint(len(G.nodes()))]
        # print(node_tmp, "G.node props:", G.nodes[node_tmp])
        # edge_tmp = list(G.edges())[np.random.randint(len(G.edges()))]
        # print(edge_tmp, "G.edge props:", G.edges(edge_tmp))
        # print(edge_tmp, "G.edge props:", G.edges[edge_tmp[0]][edge_tmp[1]])

    # # let's not clean out subgraphs yet, since we still need to run
    # # add_small_segments() and terminal_points_to_crossroads()
    # if verbose:
    #     print("Clean out short subgraphs")
    #     try:
    #         sub_graphs = list(nx.connected_component_subgraphs(G))
    #     except:
    #         sub_graphs = list(nx.conncted_components(G))
    #     # print("sub_graphs:", sub_graphs)
    # # sknw attaches a 'weight' property that is the length in pixels
    # t01 = time.time()
    # G = clean_sub_graphs(G, min_length=min_subgraph_length_pix,
    #                  max_nodes_to_skip=100,
    #                  weight='weight', verbose=verbose,
    #                  super_verbose=False)
    # t02 = time.time()
    # if verbose:
    #     print("Time to run clean_sub_graphs():", t02-t01, "seconds")
    #     print("len G_sub.nodes():", len(G.nodes()))
    #     print("len G_sub.edges():", len(G.edges()))

    # remove self loops
    ebunch = nx.selfloop_edges(G)
    G.remove_edges_from(list(ebunch))

    # save G
    if len(out_gpickle) > 0:
        nx.write_gpickle(G, out_gpickle)

    return G, ske, img_refine


###############################################################################
def G_to_wkt(G, add_small=True, connect_crossroads=True,
             img_copy=None, debug=False, verbose=False, super_verbose=False):
    """Transform G to wkt"""

    # print("G:", G)
    if G == [linestring.format("EMPTY")] or type(G) == str:
        return [linestring.format("EMPTY")]

    node_lines = graph2lines(G)
    # if verbose:
    #    print("node_lines:", node_lines)

    if not node_lines:
        return [linestring.format("EMPTY")]
    try:
        node = G.node
    except:
        node = G.nodes
    # print("node:", node)
    deg = dict(G.degree())
    wkt = []
    terminal_points = [i for i, d in deg.items() if d == 1]

    # refine wkt
    if verbose:
        print("Refine wkt...")
    terminal_lines = {}
    vertices = []
    for i, w in enumerate(node_lines):
        if ((i % 10000) == 0) and (i > 0) and verbose:
            print("  ", i, "/", len(node_lines))
        coord_list = []
        additional_paths = []
        for s, e in pairwise(w):
            vals = flatten([[v] for v in G[s][e].values()])
            for ix, val in enumerate(vals):

                s_coord, e_coord = node[s]['o'], node[e]['o']
                # print("s_coord:", s_coord, "e_coord:", e_coord)
                pts = val.get('pts', [])
                if s in terminal_points:
                    terminal_lines[s] = (s_coord, e_coord)
                if e in terminal_points:
                    terminal_lines[e] = (e_coord, s_coord)

                ps = add_direction_change_nodes(pts, s, e, s_coord, e_coord)

                if len(ps.shape) < 2 or len(ps) < 2:
                    continue

                if len(ps) == 2 and np.all(ps[0] == ps[1]):
                    continue

                line_strings = ["{1:.1f} {0:.1f}".format(*c.tolist()) for c in ps]
                if ix == 0:
                    coord_list.extend(line_strings)
                else:
                    additional_paths.append(line_strings)

                vertices.append(ps)

        if not len(coord_list):
            continue
        segments = remove_duplicate_segments(coord_list)
        # print("segments:", segments)
        # return
    
        for coord_list in segments:
            if len(coord_list) > 1:
                line = '(' + ", ".join(coord_list) + ')'
                wkt.append(linestring.format(line))
        for line_strings in additional_paths:
            line = ", ".join(line_strings)
            line_rev = ", ".join(reversed(line_strings))
            for s in wkt:
                if line in s or line_rev in s:
                    break
            else:
                wkt.append(linestring.format('(' + line + ')'))

    if add_small and len(terminal_points) > 1:
        small_segs, good_pairs, good_coords = add_small_segments(
            G, terminal_points, terminal_lines, verbose=verbose)
        print("small_segs", small_segs)
        wkt.extend(small_segs)

    if debug:
        vertices = flatten(vertices)
        visualize(img_copy, G, vertices)

    if not wkt:
        return [linestring.format("EMPTY")]

    #return cross_segs
    return wkt



###############################################################################
def build_wkt_dir(indir, outfile, out_ske_dir, out_gdir='', thresh=0.3,
                  # threshes={'2': .3, '3': .3, '4': .3, '5': .2},
                  im_prefix='',
                  debug=False, 
                  add_small=True, 
                  fix_borders=True,
                  img_shape=(1300, 1300),
                  skel_replicate=5, skel_clip=2,
                  img_mult=255,
                  hole_size=300, cv2_kernel_close=7, cv2_kernel_open=7,
                  min_subgraph_length_pix=50,
                  min_spur_length_pix=16,
                  spacenet_naming_convention=False,
                  num_classes=1,
                  max_out_size=(100000, 100000),
                  use_medial_axis=True,
                  skeleton_band='all',
                  kernel_blur=27,
                  min_background_frac=0.2,
                  n_threads=12,
                  verbose=False,
                  super_verbose=False):
    '''Execute built_graph_wkt for an entire folder
    Split image name on AOI, keep only name after AOI.  This is necessary for 
    scoring'''

    im_files = np.sort([z for z in os.listdir(indir) if z.endswith('.tif')])
    nfiles = len(im_files)
    n_threads = min(n_threads, nfiles)
    params = []
    for i, imfile in enumerate(im_files):
                
        t1 = time.time()
        if verbose:
            print("\n", i+1, "/", nfiles, ":", imfile)
        logger1.info("{x} / {y} : {z}".format(x=i+1, y=nfiles, z=imfile))
        img_loc = os.path.join(indir, imfile)

        if spacenet_naming_convention:
            im_root = 'AOI' + imfile.split('AOI')[-1].split('.')[0]
        else:
            im_root = imfile.split('.')[0]
        if len(im_prefix) > 0:
            im_root = im_root.split(im_prefix)[-1]

        if verbose:
            print("  img_loc:", img_loc)
            print("  im_root:", im_root)
        if out_ske_dir:
            out_ske_file = os.path.join(out_ske_dir, imfile)
        else:
            out_ske_file = ''
        if verbose:
            print("  out_ske_file:", out_ske_file)
        if len(out_gdir) > 0:
            out_gpickle = os.path.join(out_gdir,
                                       imfile.split('.')[0] + '.gpickle')
        else:
            out_gpickle = ''

        param_row = (img_loc, out_ske_file, out_gpickle, thresh, \
                debug, fix_borders, \
                img_shape,\
                skel_replicate, skel_clip, \
                img_mult, hole_size, \
                cv2_kernel_close, cv2_kernel_open,\
                min_subgraph_length_pix,\
                min_spur_length_pix,\
                max_out_size,\
                use_medial_axis,\
                num_classes,\
                skeleton_band, \
                kernel_blur,\
                min_background_frac,\
                verbose)
        params.append(param_row)

    # execute
    if n_threads > 1:
        pool = Pool(n_threads)
        pool.map(img_to_ske_G, params)
    else:
        img_to_ske_G(params[0])

    # now build wkt_list (single-threaded)
    all_data = []
    for gpickle in os.listdir(out_gdir):
        t1 = time.time()
        gpath = os.path.join(out_gdir, gpickle)
        imfile = gpickle.split('.')[0] + '.tif'
        if spacenet_naming_convention:
            im_root = 'AOI' + imfile.split('AOI')[-1].split('.')[0]
        else:
            im_root = imfile.split('.')[0]
        if len(im_prefix) > 0:
            im_root = im_root.split(im_prefix)[-1]

        G = nx.read_gpickle(gpath)
        wkt_list = G_to_wkt(G, add_small=add_small, 
                            verbose=verbose, super_verbose=super_verbose)

        # add to all_data
        for v in wkt_list:
            all_data.append((im_root, v))
            # all_data.append((imfile, v))
        t2 = time.time()
        logger1.info("Time to build graph: {} seconds".format(t2-t1))

    # save to csv
    df = pd.DataFrame(all_data, columns=['ImageId', 'WKT_Pix'])
    df.to_csv(outfile, index=False)

    return df


###############################################################################
def main():

    global logger1
    add_small=True
    verbose = True
    super_verbose = False
    spacenet_naming_convention = False  # True
    debug=False
    fix_borders=True
    img_shape=() # (1300, 1300)
    skel_replicate=5
    skel_clip=2
    img_mult=255
    hole_size=300
    cv2_kernel_close=7
    cv2_kernel_open=7
    kernel_blur=-1  # 25
    min_background_frac=-1  # 0.2
    max_out_size=(2000000, 2000000)
    n_threads = 12
    im_prefix = ''
    
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path')
    args = parser.parse_args()
    with open(args.config_path, 'r') as f:
        cfg = json.load(f)
        config = Config(**cfg)

    min_spur_length_pix = int(np.rint(config.min_spur_length_m / config.GSD))
    print("min_spur_length_pix:", min_spur_length_pix)
    use_medial_axis = bool(config.use_medial_axis)
    print("Use_medial_axis?", use_medial_axis)
    pix_extent = config.eval_rows - (2 * config.padding)

    # check if we are stitching together large images or not
    out_dir_mask_norm = os.path.join(config.path_results_root, 
                                     config.test_results_dir, 
                                     config.stitched_dir_norm)
    folds_dir = os.path.join(config.path_results_root, 
                             config.test_results_dir, 
                             config.folds_save_dir)
    merge_dir = os.path.join(config.path_results_root, 
                             config.test_results_dir, 
                             config.merged_dir)

    if os.path.exists(out_dir_mask_norm):
        im_dir = out_dir_mask_norm
    else:
        if config.num_folds > 1:
            im_dir = merge_dir
        else:
            im_dir = folds_dir
            im_prefix = 'fold0_'
            
    os.makedirs(im_dir, exist_ok=True)
  
    # outut files
    res_root_dir = os.path.join(config.path_results_root, 
                                config.test_results_dir)
    outfile_csv = os.path.join(res_root_dir, config.wkt_submission)
    #outfile_gpickle = os.path.join(res_root_dir, 'G_sknw.gpickle')
    out_ske_dir = os.path.join(res_root_dir, config.skeleton_dir)  # set to '' to not save
    os.makedirs(out_ske_dir, exist_ok=True)
    if len(config.skeleton_pkl_dir) > 0:
        out_gdir = os.path.join(res_root_dir, config.skeleton_pkl_dir)  # set to '' to not save
        os.makedirs(out_gdir, exist_ok=True)
    else:
        out_gdir = ''
         
    print("im_dir:", im_dir)
    print("out_ske_dir:", out_ske_dir)
    print("out_gdir:", out_gdir)
        
    thresh = config.skeleton_thresh
#    # thresholds for each aoi
#    threshes={'2': .3, '3': .3, '4': .3, '5': .2}  
#    thresh = threshes[config.aoi]
    min_subgraph_length_pix = config.min_subgraph_length_pix
    #min_subgraph_length_pix=200

    log_file = os.path.join(res_root_dir, 'skeleton.log')
    console, logger1 = make_logger.make_logger(log_file, logger_name='log',
                                               write_to_console=bool(config.log_to_console))   
    
    # print("Building wkts...")
    t0 = time.time()
    df = build_wkt_dir(im_dir, outfile_csv, out_ske_dir, out_gdir, thresh, 
                debug=debug, 
                add_small=add_small, 
                fix_borders=fix_borders,
                img_shape=img_shape,
                skel_replicate=skel_replicate, skel_clip=skel_clip,
                img_mult=img_mult, hole_size=hole_size,
                min_subgraph_length_pix=min_subgraph_length_pix,
                min_spur_length_pix=min_spur_length_pix,
                cv2_kernel_close=cv2_kernel_close, cv2_kernel_open=cv2_kernel_open,
                max_out_size=max_out_size,
                skeleton_band=config.skeleton_band,
                num_classes=config.num_classes,
                im_prefix=im_prefix,
                spacenet_naming_convention=spacenet_naming_convention,
                use_medial_axis=use_medial_axis,
                kernel_blur=kernel_blur,
                min_background_frac=min_background_frac,
                n_threads=n_threads,
                verbose=verbose,
                super_verbose=super_verbose
                )        

    print("len df:", len(df))
    print("outfile:", outfile_csv)
    t1 = time.time()
    logger1.info("Total time to run build_wkt_dir: {} seconds".format(t1-t0))
    print("Total time to run build_wkt_dir:", t1-t0, "seconds")


##############################################################################
if __name__ == "__main__":
    main()
