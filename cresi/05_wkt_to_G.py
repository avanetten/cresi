#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May  8 00:10:40 2018

@author: avanetten

Read in a list of wkt linestrings, render to networkx graph
"""

from __future__ import print_function
import os
import utm
import shapely.wkt
import shapely.ops
from shapely.geometry import mapping, Point, LineString
import fiona
import networkx as nx
import osmnx as ox
from osgeo import gdal, ogr, osr
import argparse
import json
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import logging
# import cv2

from utils import make_logger
from jsons.config import Config

logger1 = None


###############################################################################
def clean_sub_graphs(G_, min_length=150, max_nodes_to_skip=30,
                     weight='length_pix', verbose=True,
                     super_verbose=False):
    '''Remove subgraphs with a max path length less than min_length,
    if the subgraph has more than max_noxes_to_skip, don't check length 
       (this step great improves processing time)'''
    
    if len(list(G_.nodes())) == 0:
        return G_
    
    print ("Running clean_sub_graphs...")
    sub_graphs = list(nx.connected_component_subgraphs(G_))
    bad_nodes = []
    if verbose:
        print (" len(G_.nodes()):", len(list(G_.nodes())) )
        print (" len(G_.edges()):", len(list(G_.edges())) )
    if super_verbose:
        print ("G_.nodes:", G_.nodes())
        edge_tmp = G_.edges()[np.random.randint(len(G_.edges()))]
        print (edge_tmp, "G.edge props:", G_.edge[edge_tmp[0]][edge_tmp[1]])

    for G_sub in sub_graphs:
        # don't check length if too many nodes in subgraph
        if len(G_sub.nodes()) > max_nodes_to_skip:
            continue
        
        else:
            all_lengths = dict(nx.all_pairs_dijkstra_path_length(G_sub, weight=weight))
            if super_verbose:
                        print ("  \nGs.nodes:", G_sub.nodes() )
                        print ("  all_lengths:", all_lengths )
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
                        print ("  u, v", u,v )
                        print ("    uprime, vprime:", uprime, vprime )
            max_len = np.max(lens)
            if super_verbose:
                print ("  Max length of path:", max_len)
            if max_len < min_length:
                bad_nodes.extend(G_sub.nodes())
                if super_verbose:
                    print (" appending to bad_nodes:", G_sub.nodes())

    # remove bad_nodes
    G_.remove_nodes_from(bad_nodes)
    if verbose:
        print (" num bad_nodes:", len(bad_nodes))
        #print ("bad_nodes:", bad_nodes)
        print (" len(G'.nodes()):", len(G_.nodes()))
        print (" len(G'.edges()):", len(G_.edges()))
    if super_verbose:
        print ("  G_.nodes:", G_.nodes())
        
    return G_


###############################################################################
def remove_short_edges(G_, min_spur_length_m=100, verbose=True):
    """Remove unconnected edges shorter than the desired length"""
    if verbose:
        print("Remove shoert edges")
    deg_list = list(G_.degree)
    # iterate through list
    bad_nodes = []
    for i, (n, deg) in enumerate(deg_list):
        # if verbose and (i % 500) == 0:
        #     print(n, deg)
        # check if node has only one neighbor
        if deg == 1:
            # get edge
            edge = list(G_.edges(n))
            u, v = edge[0]
            # get edge length
            edge_props = G_.get_edge_data(u, v, 0)
            length = edge_props['length']
            # edge_props = G_.edges([u, v])

            if length < min_spur_length_m:
                bad_nodes.append(n)
                if verbose:
                    print(i, "/", len(list(G_.nodes())),
                          "n, deg, u, v, length:", n, deg, u, v, length)

    if verbose:
        print("bad_nodes:", bad_nodes)
    G_.remove_nodes_from(bad_nodes)
    if verbose:
        print("num remaining nodes:", len(list(G_.nodes())))
    return G_


###############################################################################
def wkt_list_to_nodes_edges(wkt_list):
    '''Convert wkt list to nodes and edges
    Make an edge between each node in linestring. Since one linestring
    may contain multiple edges, this is the safest approach'''
    
    node_loc_set = set()    # set of edge locations
    node_loc_dic = {}       # key = node idx, val = location
    node_loc_dic_rev = {}   # key = location, val = node idx
    edge_loc_set = set()    # set of edge locations
    edge_dic = {}           # edge properties
    node_iter = 0
    edge_iter = 0
    
    for i,lstring in enumerate(wkt_list):
        # get lstring properties
        shape = shapely.wkt.loads(lstring)
        xs, ys = shape.coords.xy
        length_orig = shape.length
        
        # iterate through coords in line to create edges between every point
        for j,(x,y) in enumerate(zip(xs, ys)):
            loc = (x,y)
            # for first item just make node, not edge
            if j == 0:
                # if not yet seen, create new node
                if loc not in node_loc_set:
                    node_loc_set.add(loc)
                    node_loc_dic[node_iter] = loc
                    node_loc_dic_rev[loc] = node_iter
                    node = node_iter
                    node_iter += 1
                    
            # if not first node in edge, retrieve previous node and build edge
            else:
                prev_loc = (xs[j-1], ys[j-1])
                #print ("prev_loc:", prev_loc)
                prev_node = node_loc_dic_rev[prev_loc]

                # if new, create new node
                if loc not in node_loc_set:
                    node_loc_set.add(loc)
                    node_loc_dic[node_iter] = loc
                    node_loc_dic_rev[loc] = node_iter
                    node = node_iter
                    node_iter += 1
                # if seen before, retrieve node properties
                else:
                    node = node_loc_dic_rev[loc]

                
                # add edge, which is start_node to end_node
                edge_loc = (loc, prev_loc)
                edge_loc_rev = (prev_loc, loc)
                # shouldn't be duplicate edges, so break if we see one
                if (edge_loc in edge_loc_set) or (edge_loc_rev in edge_loc_set):
                    print ("Oops, edge already seen, returning:", edge_loc)
                    return
                
                # get distance to prev_loc and current loc
                proj_prev = shape.project(Point(prev_loc))
                proj = shape.project(Point(loc))
                # edge length is the diffence of the two projected lengths
                #   along the linestring
                edge_length = abs(proj - proj_prev)
                # make linestring
                line_out = LineString([prev_loc, loc])
                line_out_wkt = line_out.wkt
                
                edge_props = {'start': prev_node,
                              'start_loc_pix': prev_loc,
                              'end': node,
                              'end_loc_pix': loc,
                              'length_pix': edge_length,
                              'wkt_pix': line_out_wkt,
                              'geometry_pix': line_out,
                              'osmid': i}
                #print ("edge_props", edge_props)
                
                edge_loc_set.add(edge_loc)
                edge_dic[edge_iter] = edge_props
                edge_iter += 1

    return node_loc_dic, edge_dic
        

############################################################################### 
def wkt_list_to_nodes_edges_sloppy(wkt_list):
    '''Convert wkt list to nodes and edges
    Assumes each linestring corresponds to a unique edge
    Since this is not always the case, this function fails if a linestring
    contains multiple edges'''
    
    node_loc_set = set()    # set of edge locations
    node_loc_dic = {}       # key = node idx, val = location
    node_loc_dic_rev = {}   # key = location, val = node idx
    edge_dic = {}           # edge properties
    node_iter = 0
    edge_iter = 0
    
    for lstring in wkt_list:
        # get lstring properties
        shape = shapely.wkt.loads(lstring)
        x, y = shape.coords.xy
        length = shape.length
        
        # set start node
        start_loc = (x[0], y[0])
        # if new, create new node
        if start_loc not in node_loc_set:
            node_loc_set.add(start_loc)
            node_loc_dic[node_iter] = start_loc
            node_loc_dic_rev[start_loc] = node_iter
            start_node = node_iter
            node_iter += 1
        # if seen before, retrieve node properties
        else:
            start_node = node_loc_dic_rev[start_loc]
        
        # set end node (just like start node)
        end_loc = (x[-1], y[-1])
        # if new, create new node
        if end_loc not in node_loc_set:
            node_loc_set.add(end_loc)
            node_loc_dic[node_iter] = end_loc
            node_loc_dic_rev[end_loc] = node_iter
            end_node = node_iter
            node_iter += 1
        # if seen before, retrieve node properties
        else:
            end_node = node_loc_dic_rev[end_loc]
        
            
        # add edge, which is start_node to end_node
        edge_props = {'start': start_node,
                      'start_loc_pix': start_loc,
                      'end': end_node,
                      'end_loc_pix': end_loc,
                      'length_pix': length,
                      'wkt_pix': lstring,
                      'geometry_pix': shape}

        edge_dic[edge_iter] = edge_props
        edge_iter += 1

    return node_loc_dic, edge_dic
        

###############################################################################
def nodes_edges_to_G(node_loc_dic, edge_dic, name='glurp'):
    '''Take output of wkt_list_to_nodes_edges(wkt_list) and create networkx 
    graph'''
    
    G = nx.MultiDiGraph()
    # set graph crs and name
    G.graph = {'name': name,
               'crs': {'init': 'epsg:4326'}
               }
    
    # add nodes
    #for key,val in node_loc_dic.iteritems():
    for key in node_loc_dic.keys():
        val = node_loc_dic[key]
        attr_dict = {'osmid': key,
                     'x_pix': val[0],
                     'y_pix': val[1]}
        G.add_node(key, **attr_dict)
    
    # add edges
    #for key,val in edge_dic.iteritems():
    for key in edge_dic.keys():
        val = edge_dic[key]
        attr_dict = val
        u = attr_dict['start']
        v = attr_dict['end']
        #attr_dict['osmid'] = str(i)
        
        #print ("nodes_edges_to_G:", u, v, "attr_dict:", attr_dict)
        if type(attr_dict['start_loc_pix']) == list:
            return
        
        G.add_edge(u, v, **attr_dict)
        
        ## always set edge key to zero?  (for nx 1.X)
        ## THIS SEEMS NECESSARY FOR OSMNX SIMPLIFY COMMAND
        #G.add_edge(u, v, key=0, attr_dict=attr_dict)
        ##G.add_edge(u, v, key=key, attr_dict=attr_dict)
        
    #G1 = ox.simplify_graph(G)
    
    G2 = G.to_undirected()
    
    return G2

###############################################################################
def wkt_to_shp(wkt_list, shp_file):
    '''Take output of build_graph_wkt() and render the list of linestrings
    into a shapefile
    # https://gis.stackexchange.com/questions/52705/how-to-write-shapely-geometries-to-shapefiles
    '''
        
    # Define a linestring feature geometry with one attribute
    schema = {
        'geometry': 'LineString',
        'properties': {'id': 'int'},
    }
    
    # Write a new shapefile
    with fiona.open(shp_file, 'w', 'ESRI Shapefile', schema) as c:
        for i,line in enumerate(wkt_list):
            shape = shapely.wkt.loads(line)
            c.write({
                    'geometry': mapping(shape),
                    'properties': {'id': i},
                    })
        
    return

###############################################################################
def shp_to_G(shp_file):
    '''Ingest G from shapefile
    DOES NOT APPEAR TO WORK CORRECTLY'''
    
    G = nx.read_shp(shp_file)
    
    return G
        

###############################################################################
def pixelToGeoCoord(xPix, yPix, inputRaster, sourceSR='', geomTransform='', targetSR=''):
    '''from spacenet geotools'''

    if targetSR =='':
        performReprojection=False
        targetSR = osr.SpatialReference()
        targetSR.ImportFromEPSG(4326)
    else:
        performReprojection=True

    if geomTransform=='':
        srcRaster = gdal.Open(inputRaster)
        geomTransform = srcRaster.GetGeoTransform()

        source_sr = osr.SpatialReference()
        source_sr.ImportFromWkt(srcRaster.GetProjectionRef())

    geom = ogr.Geometry(ogr.wkbPoint)
    xOrigin = geomTransform[0]
    yOrigin = geomTransform[3]
    pixelWidth = geomTransform[1]
    pixelHeight = geomTransform[5]

    xCoord = (xPix * pixelWidth) + xOrigin
    yCoord = (yPix * pixelHeight) + yOrigin
    geom.AddPoint(xCoord, yCoord)

    if performReprojection:
        if sourceSR=='':
            srcRaster = gdal.Open(inputRaster)
            sourceSR = osr.SpatialReference()
            sourceSR.ImportFromWkt(srcRaster.GetProjectionRef())
        coord_trans = osr.CoordinateTransformation(sourceSR, targetSR)
        geom.Transform(coord_trans)

    return (geom.GetX(), geom.GetY())

###############################################################################
def get_node_geo_coords(G, im_file, verbose=False):
    
    nn = len(G.nodes())
    for i,(n,attr_dict) in enumerate(G.nodes(data=True)):
        if verbose:
            print ("node:", n)
        # if (i % 1000) == 0:
        #     print ("node", i, "/", nn, attr_dict)
        x_pix, y_pix = attr_dict['x_pix'], attr_dict['y_pix']
        
        targetSR = osr.SpatialReference()
        targetSR.ImportFromEPSG(4326)
        lon, lat = pixelToGeoCoord(x_pix, y_pix, im_file, targetSR=targetSR)
        
        # fix zone
        if i == 0:
            [utm_east, utm_north, utm_zone, utm_letter] =\
                        utm.from_latlon(lat, lon)
        else:
            [utm_east, utm_north, _, _] = utm.from_latlon(lat, lon,
                force_zone_number=utm_zone, force_zone_letter=utm_letter)
                        
        if lat > 90:
            print(n, attr_dict)
            return
        attr_dict['lon'] = lon
        attr_dict['lat'] = lat
        attr_dict['utm_east'] = utm_east
        attr_dict['utm_zone'] = utm_zone
        attr_dict['utm_letter'] = utm_letter
        attr_dict['utm_north'] = utm_north        
        attr_dict['x'] = lon
        attr_dict['y'] = lat

        if (i % 1000) == 0:
            print ("node", i, "/", nn, attr_dict)

        if verbose:
            print (" ", n, attr_dict)

    return G


###############################################################################
def convert_pix_lstring_to_geo(wkt_lstring, im_file, 
                               utm_zone=None, utm_letter=None, verbose=False):
    '''Convert linestring in pixel coords to geo coords
    If zone or letter changes inthe middle of line, it's all screwed up, so
    force zone and letter based on first point
    (latitude, longitude, force_zone_number=None, force_zone_letter=None)
    Or just force utm zone and letter explicitly
        '''
    shape = wkt_lstring  #shapely.wkt.loads(lstring)
    x_pixs, y_pixs = shape.coords.xy
    coords_latlon = []
    coords_utm = []
    for i,(x,y) in enumerate(zip(x_pixs, y_pixs)):
        
        targetSR = osr.SpatialReference()
        targetSR.ImportFromEPSG(4326)
        lon, lat = pixelToGeoCoord(x, y, im_file, targetSR=targetSR)

        if utm_zone and utm_letter:
            [utm_east, utm_north, _, _] = utm.from_latlon(lat, lon,
                force_zone_number=utm_zone, force_zone_letter=utm_letter)
        else:
            [utm_east, utm_north, utm_zone, utm_letter] = utm.from_latlon(lat, lon)
        
        if verbose:
            print("lat lon, utm_east, utm_north, utm_zone, utm_letter]",
                [lat, lon, utm_east, utm_north, utm_zone, utm_letter])
        coords_utm.append([utm_east, utm_north])
        coords_latlon.append([lon, lat])
    
    lstring_latlon = LineString([Point(z) for z in coords_latlon])
    lstring_utm = LineString([Point(z) for z in coords_utm])
    
    return lstring_latlon, lstring_utm, utm_zone, utm_letter                   


###############################################################################
def get_edge_geo_coords(G, im_file, remove_pix_geom=True,
                        verbose=False):
    
    ne = len(list(G.edges()))
    for i,(u,v,attr_dict) in enumerate(G.edges(data=True)):
        if verbose:
            print ("edge:", u,v)
            print ("  attr_dict_init:", attr_dict)

        if (i % 1000) == 0:
            print ("edge", i, "/", ne)
        geom_pix = attr_dict['geometry_pix']
        
        # fix utm zone and letter to first item seen
        if i == 0:
            lstring_latlon, lstring_utm, utm_zone, utm_letter = convert_pix_lstring_to_geo(geom_pix, im_file)
        else:   
            lstring_latlon, lstring_utm, _, _  \
                = convert_pix_lstring_to_geo(geom_pix, im_file,
                                             utm_zone=utm_zone,
                                             utm_letter=utm_letter)
        # lstring_latlon, lstring_utm, utm_zone, utm_letter = convert_pix_lstring_to_geo(geom_pix, im_file)
        attr_dict['geometry_latlon_wkt'] = lstring_latlon.wkt
        attr_dict['geometry_utm_wkt'] = lstring_utm.wkt
        attr_dict['length_latlon'] = lstring_latlon.length
        attr_dict['length_utm'] = lstring_utm.length
        attr_dict['length'] = lstring_utm.length
        attr_dict['utm_zone'] = utm_zone
        attr_dict['utm_letter'] = utm_letter
        if verbose:
            print ("  attr_dict_final:", attr_dict)
            
        # geometry screws up osmnx.simplify function
        if remove_pix_geom:
            #attr_dict['geometry_wkt'] = lstring_latlon.wkt
            attr_dict['geometry_pix'] = geom_pix.wkt
            
        # ensure utm length isn't excessive
        if lstring_utm.length > 5000:
            print(u, v, "edge length too long!:", attr_dict)
            return
            
    return G


###############################################################################
def wkt_to_G(wkt_list, im_file=None, min_subgraph_length_pix=30, 
             min_spur_length_m=5, simplify_graph=True, verbose=False):
    '''Execute all functions'''

    t0 = time.time()
    print ("Running wkt_list_to_nodes_edges()...")
    node_loc_dic, edge_dic = wkt_list_to_nodes_edges(wkt_list)
    t1 = time.time()
    print ("Time to run wkt_list_to_nodes_egdes():", t1 - t0, "seconds")
    
    #print ("node_loc_dic:", node_loc_dic)
    #print ("edge_dic:", edge_dic)
    
    print ("Creating G...")
    G0 = nodes_edges_to_G(node_loc_dic, edge_dic)  
    print ("  len(G.nodes():", len(G0.nodes()))
    print ("  len(G.edges():", len(G0.edges()))
    #for edge_tmp in G0.edges():
    #    print ("\n 0 wtk_to_G():", edge_tmp, G0.edge[edge_tmp[0]][edge_tmp[1]])
    
    
    t2 = time.time()
    print ("Time to run nodes_edges_to_G():", t2-t1, "seconds")
    
    print ("Clean out short subgraphs")
    G0 = clean_sub_graphs(G0, min_length=min_subgraph_length_pix, 
                     max_nodes_to_skip=30,
                     weight='length_pix', verbose=True,
                     super_verbose=False)
    t3 = time.time()
    print ("Time to run clean_sub_graphs():", t3-t2, "seconds")

    if len(G0) == 0:
        return G0
        
    # geo coords
    if im_file:
        print ("Running get_node_geo_coords()...")
        G1 = get_node_geo_coords(G0, im_file, verbose=verbose)
        t4= time.time()
        print ("Time to run get_node_geo_coords():", t4-t3, "seconds")

        print ("Running get_edge_geo_coords()...")
        G1 = get_edge_geo_coords(G1, im_file, verbose=verbose)
        t5 = time.time()
        print ("Time to run get_edge_geo_coords():", t5-t4, "seconds")

        print("pre projection...")
        node = list(G1.nodes())[-1]
        print(node, "random node props:", G1.nodes[node])
        # print an edge
        edge_tmp = list(G1.edges())[-1]
        print(edge_tmp, "random edge props:", G1.get_edge_data(edge_tmp[0], edge_tmp[1]))

        print ("projecting graph...")
        G_projected = ox.project_graph(G1)
        
        print("post projection...")
        node = list(G_projected.nodes())[-1]
        print(node, "random node props:", G_projected.nodes[node])
        # print an edge
        edge_tmp = list(G_projected.edges())[-1]
        print(edge_tmp, "random edge props:", G_projected.get_edge_data(edge_tmp[0], edge_tmp[1]))

        t6 = time.time()
        print ("Time to project graph:", t6-t5, "seconds")

        # simplify
        #G_simp = ox.simplify_graph(G_projected.to_directed())
        #ox.plot_graph(G_projected)
        #G1.edge[19][22]
        
        Gout = G_projected #G_simp
    
    else:
        Gout = G0


    ###########################################################################
    # remove short edges
    t31 = time.time()
    Gout = remove_short_edges(Gout, min_spur_length_m=min_spur_length_m)
    t32 = time.time()
    print("Time to remove_short_edges():", t32 - t31, "seconds")
    ###########################################################################
    
    if simplify_graph:
        print ("Simplifying graph")
        t7 = time.time()
        G0 = ox.simplify_graph(Gout.to_directed())
        Gout = G0.to_undirected()
        #Gout = ox.project_graph(G0)
        
        t8 = time.time()
        print ("Time to run simplify graph:", t8-t7, "seconds")
        # When the simplify funciton combines edges, it concats multiple
        #  edge properties into a list.  This means that 'geometry_pix' is now
        #  a list of geoms.  Convert this to a linestring with 
        #   shaply.ops.linemergeconcats 
        print ("Merge 'geometry' linestrings...")
        keys_tmp = ['geometry_pix', 'geometry_latlon_wkt', 'geometry_utm_wkt']
        for key_tmp in keys_tmp:
            print ("Merge", key_tmp, "...")
            for i,(u,v,attr_dict) in enumerate(Gout.edges(data=True)):
                if (i % 10000) == 0:
                    print (i, u , v)
                geom = attr_dict[key_tmp]
                #print (i, u, v, "geom:", geom)
                #print ("  type(geom):", type(geom))
                
                if type(geom) == list:
                    # check if the list items are wkt strings, if so, create
                    #   linestrigs
                    if (type(geom[0]) == str):# or (type(geom_pix[0]) == unicode):
                        geom = [shapely.wkt.loads(ztmp) for ztmp in geom]
                    # merge geoms
                    #geom = shapely.ops.linemerge(geom)
                    #attr_dict[key_tmp] =  geom
                    attr_dict[key_tmp] = shapely.ops.linemerge(geom)
                elif type(geom) == str:
                    attr_dict[key_tmp] = shapely.wkt.loads(geom)
                else:
                    pass

        # assign 'geometry' tag to geometry_utm_wkt
        for i,(u,v,attr_dict) in enumerate(Gout.edges(data=True)):
            if verbose:
                print ("Create 'geometry' field in edges...")
            #geom_pix = attr_dict[key_tmp]
            line = attr_dict['geometry_utm_wkt']       
            if type(line) == str:# or type(line) == unicode:
                attr_dict['geometry'] = shapely.wkt.loads(line) 
            else:
                attr_dict['geometry'] = attr_dict['geometry_utm_wkt']       
            # update wkt_pix?
            #print ("attr_dict['geometry_pix':", attr_dict['geometry_pix'])
            attr_dict['wkt_pix'] = attr_dict['geometry_pix'].wkt
        
            # update 'length_pix'
            attr_dict['length_pix'] = np.sum([attr_dict['length_pix']])
                
    # get a few stats (and set to graph properties)
    logger1.info("Number of nodes: {}".format(len(Gout.nodes())))
    logger1.info("Number of edges: {}".format(len(Gout.edges())))
    #print ("Number of nodes:", len(Gout.nodes()))
    #print ("Number of edges:", len(Gout.edges()))
    Gout.graph['N_nodes'] = len(Gout.nodes())
    Gout.graph['N_edges'] = len(Gout.edges())
    
    # get total length of edges
    tot_meters = 0
    for i,(u,v,attr_dict) in enumerate(Gout.edges(data=True)):
        tot_meters  += attr_dict['length'] 
    print ("Length of edges (km):", tot_meters/1000)
    Gout.graph['Tot_edge_km'] = tot_meters/1000

    print ("G.graph:", Gout.graph)
    
    t7 = time.time()
    print ("Total time to run wkt_to_G():", t7-t0, "seconds")
        
    return Gout

################################################################################
def main():
    
    global logger1 
    
    # min_subgraph_length_pix = 300
    min_spur_length_m = 0.001  # default = 5
    local = False #True
    verbose = True
    super_verbose = False
    make_plots = False #True
    save_shapefiles = True #False
    pickle_protocol = 4     # 4 is most recent, python 2.7 can't read 4
    
    # local
    if local:
        pass
    
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument('config_path')
        args = parser.parse_args()
        with open(args.config_path, 'r') as f:
            cfg = json.load(f)
            config = Config(**cfg)
            
        # outut files
        res_root_dir = os.path.join(config.path_results_root, config.test_results_dir)
        path_images = os.path.join(config.path_data_root, config.test_data_refined_dir)
        csv_file = os.path.join(res_root_dir, config.wkt_submission)
        graph_dir = os.path.join(res_root_dir, config.graph_dir)
        log_file = os.path.join(res_root_dir, 'wkt_to_G.log')
        os.makedirs(graph_dir, exist_ok=True)

        min_subgraph_length_pix = config.min_subgraph_length_pix
        min_spur_length_m = config.min_spur_length_m

    console, logger1 = make_logger.make_logger(log_file, logger_name='log')

    # read in wkt list
    logger1.info("df_wkt at: {}".format(csv_file))
    #print ("df_wkt at:", csv_file)
    df_wkt = pd.read_csv(csv_file)
    # columns=['ImageId', 'WKT_Pix'])

    # iterate through image ids and create graphs
    t0 = time.time()
    image_ids = np.sort(np.unique(df_wkt['ImageId']))
    print("image_ids:", image_ids)
    print("len image_ids:", len(image_ids))

    for i,image_id in enumerate(image_ids):
        
        #if image_id != 'AOI_2_Vegas_img586':
        #    continue
        out_file = os.path.join(graph_dir, image_id.split('.')[0] + '.gpickle')
        
        logger1.info("\n{x} / {y}, {z}".format(x=i+1, y=len(image_ids), z=image_id))
        #print ("\n")
        #print (i, "/", len(image_ids), image_id)
                    
        # for geo referencing, im_file should be the raw image
        if config.num_channels == 3:
            im_file = os.path.join(path_images, 'RGB-PanSharpen_' + image_id + '.tif')
        else:
            im_file = os.path.join(path_images, 'MUL-PanSharpen_' + image_id + '.tif')   
        #im_file = os.path.join(path_images, image_id)
        if not os.path.exists(im_file):
            im_file = os.path.join(path_images, image_id + '.tif')
        
        # filter 
        df_filt = df_wkt['WKT_Pix'][df_wkt['ImageId'] == image_id]
        wkt_list = df_filt.values
        #wkt_list = [z[1] for z in df_filt_vals]
        
        # print a few values
        logger1.info("\n{x} / {y}, num linestrings: {z}".format(x=i+1, y=len(image_ids), z=len(wkt_list)))
        #print ("\n", i, "/", len(image_ids), "num linestrings:", len(wkt_list))
        if verbose:
            print ("image_file:", im_file)
            print ("  wkt_list[:2]", wkt_list[:2])
    
        if (len(wkt_list) == 0) or (wkt_list[0] == 'LINESTRING EMPTY'):
            G = nx.MultiDiGraph()
            nx.write_gpickle(G, out_file, protocol=pickle_protocol)
            continue
        
        # create graph
        t1 = time.time()
        G = wkt_to_G(wkt_list, im_file=im_file, 
                     min_subgraph_length_pix=min_subgraph_length_pix,
                     min_spur_length_m=min_spur_length_m,
                     verbose=super_verbose)
        t2 = time.time()
        if verbose:
            logger1.info("Time to create graph: {} seconds".format(t2-t1))
            #print ("Time to create graph:", t2-t1, "seconds")
            
        if len(G.nodes()) == 0:
            nx.write_gpickle(G, out_file, protocol=pickle_protocol)
            continue
        
        # print a node
        node = list(G.nodes())[-1]
        print (node, "random node props:", G.nodes[node])
        # print an edge
        edge_tmp = list(G.edges())[-1]
        #print (edge_tmp, "random edge props:", G.edges([edge_tmp[0], edge_tmp[1]])) #G.edge[edge_tmp[0]][edge_tmp[1]])
        print (edge_tmp, "random edge props:", G.get_edge_data(edge_tmp[0], edge_tmp[1]))

        # save graph
        logger1.info("Saving graph to directory: {}".format(graph_dir))
        #print ("Saving graph to directory:", graph_dir)
        nx.write_gpickle(G, out_file, protocol=pickle_protocol)
        
        # save shapefile as well?
        if save_shapefiles:
            logger1.info("Saving shapefile to directory: {}".format(graph_dir))
            try:
                ox.save_graph_shapefile(G, filename=image_id.split('.')[0] , folder=graph_dir, encoding='utf-8')
            except:
                print("Cannot save shapefile...")
            #out_file2 = os.path.join(graph_dir, image_id.split('.')[0] + '.graphml')
            #ox.save_graphml(G, image_id.split('.')[0] + '.graphml', folder=graph_dir)

        # plot, if desired
        if make_plots:
            print ("Plotting graph...")
            outfile_plot = os.path.join(graph_dir, image_id)
            print ("outfile_plot:", outfile_plot)
            ox.plot_graph(G, fig_height=9, fig_width=9, 
                          #save=True, filename=outfile_plot, margin=0.01)
                          )
            #plt.tight_layout()
            plt.savefig(outfile_plot, dpi=400)
            
        #if i > 30:
        #    break
        
    tf = time.time()
    logger1.info("Time to run wkt_to_G.py: {} seconds".format(tf - t0))
    #print ("Time to run wkt_to_G.py:", tf - t0, "seconds")
         
    
###############################################################################
if __name__ == "__main__":
    main()