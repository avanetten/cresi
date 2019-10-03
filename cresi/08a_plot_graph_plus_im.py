#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May 21 12:55:47 2018

@author: avanetten

plotting adapted from:
    https://github.com/gboeing/osmnx/blob/master/osmnx/plot.py

Combine with plot_prop_dos from road_raster.py

"""


from __future__ import print_function

import matplotlib
# if in docker, the line below is necessary
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

import time
import os
import json
import argparse
import random
import numpy as np
import networkx as nx
import osmnx as ox
import ast
from shapely import wkt
import geopandas as gpd
from shapely.geometry import Point
from shapely.geometry import LineString

#import pandas as pd
#import matplotlib.cm as cm
#from descartes import PolygonPatch
#from shapely.geometry import Polygon
#from shapely.geometry import MultiPolygon

# cv2 can't load large files, so need to import skimage too
import skimage.io 
import cv2

from osmnx.utils import log, make_str
import osmnx.settings as ox_settings

###############################################################################
def load_graphml_v0p5(filename, folder=None):
    """
    Load a GraphML file from disk and convert the node/edge attributes to
    correct data types.  Assume filename holds entire path to file s
    from: https://github.com/gboeing/osmnx/blob/master/osmnx/save_load.py

    Parameters
    ----------
    filename : string
        the name of the graphml file (including file extension)
    folder : string
        the folder containing the file, if None, use default data folder
    Returns
    -------
    networkx multidigraph
    
    change to undirected...
    """
    start_time = time.time()

    G = nx.MultiGraph(nx.read_graphml(filename, node_type=int))
#    # read the graph from disk
#    if folder is None:
#        folder = settings.data_folder
#    path = os.path.join(folder, filename)
#    G = nx.MultiDiGraph(nx.read_graphml(path, node_type=int))

    # convert graph crs attribute from saved string to correct dict data type
    G.graph['crs'] = ast.literal_eval(G.graph['crs'])

    if 'streets_per_node' in G.graph:
        G.graph['streets_per_node'] = ast.literal_eval(G.graph['streets_per_node'])

    # convert numeric node tags from string to numeric data types
    print('Converting node and edge attribute data types')
    for _, data in G.nodes(data=True):
        data['osmid'] = int(data['osmid'])
        data['x'] = float(data['x'])
        data['y'] = float(data['y'])
        data['x_pix'] = float(data['x_pix'])
        data['y_pix'] = float(data['y_pix'])
        
    # convert numeric, bool, and list node tags from string to correct data types
    for _, _, data in G.edges(data=True, keys=False):

#        # first parse oneway to bool and length to float - they should always
#        # have only 1 value each
#        data['oneway'] = ast.literal_eval(data['oneway'])
        
        data['length'] = float(data['length'])

        # these attributes might have a single value, or a list if edge's
        # topology was simplified
        for attr in ['highway', 'name', 'bridge', 'tunnel', 'lanes', 'ref', 'maxspeed', 'service', 'access', 'area', 'landuse', 'width', 'est_width']:
            # if this edge has this attribute, and it starts with '[' and ends
            # with ']', then it's a list to be parsed
            if attr in data and data[attr][0] == '[' and data[attr][-1] == ']':
                # try to convert the string list to a list type, else leave as
                # single-value string (and leave as string if error)
                try:
                    data[attr] = ast.literal_eval(data[attr])
                except:
                    pass

        # osmid might have a single value or a list, but if single value, then
        # parse int
        if 'osmid' in data:
            if data['osmid'][0] == '[' and data['osmid'][-1] == ']':
                data['osmid'] = ast.literal_eval(data['osmid'])
            else:
                data['osmid'] = int(data['osmid'])

        # if geometry attribute exists, load the string as well-known text to
        # shapely LineString
        if 'geometry' in data:
            data['geometry'] = wkt.loads(data['geometry'])

    # remove node_default and edge_default metadata keys if they exist
    if 'node_default' in G.graph:
        del G.graph['node_default']
    if 'edge_default' in G.graph:
        del G.graph['edge_default']

    print('Loaded graph with {:,} nodes and {:,} edges in {:,.2f} seconds from "{}"'.format(len(list(G.nodes())),
                                                                                          len(list(G.edges())),
                                                                                          time.time()-start_time,
                                                                                          filename))
    return G

################################################################################
#def gdfs_to_graph(gdf_nodes, gdf_edges):
#    """
#    Convert node and edge GeoDataFrames into a graph
#    Parameters
#    ----------
#    gdf_nodes : GeoDataFrame
#    gdf_edges : GeoDataFrame
#    Returns
#    -------
#    networkx multidigraph
#    """
#
#    G = nx.MultiDiGraph()
#    G.graph['crs'] = gdf_nodes.crs
#    G.graph['name'] = gdf_nodes.gdf_name.rstrip('_nodes')
#
#    # add the nodes and their attributes to the graph
#    G.add_nodes_from(gdf_nodes.index)
#    attributes = gdf_nodes.to_dict()
#    for attribute_name in gdf_nodes.columns:
#        # only add this attribute to nodes which have a non-null value for it
#        attribute_values = {k:v for k, v in attributes[attribute_name].items() if pd.notnull(v)}
#        nx.set_node_attributes(G, name=attribute_name, values=attribute_values)
#
#    # add the edges and attributes that are not u, v, key (as they're added
#    # separately) or null
#    for _, row in gdf_edges.iterrows():
#        attrs = {}
#        for label, value in row.iteritems():
#            if (label not in ['u', 'v', 'key']) and (isinstance(value, list) or pd.notnull(value)):
#                attrs[label] = value
#        G.add_edge(row['u'], row['v'], key=row['key'], **attrs)
#
#    return G

###############################################################################
def graph_to_gdfs_pix(G, nodes=True, edges=True, node_geometry=True, fill_edge_geometry=True):
    """
    Convert a graph into node and/or edge GeoDataFrames
    Parameters
    ----------
    G : networkx multidigraph
    nodes : bool
        if True, convert graph nodes to a GeoDataFrame and return it
    edges : bool
        if True, convert graph edges to a GeoDataFrame and return it
    node_geometry : bool
        if True, create a geometry column from node x and y data
    fill_edge_geometry : bool
        if True, fill in missing edge geometry fields using origin and
        destination nodes
    Returns
    -------
    GeoDataFrame or tuple
        gdf_nodes or gdf_edges or both as a tuple
    """

    if not (nodes or edges):
        raise ValueError('You must request nodes or edges, or both.')

    to_return = []

    if nodes:

        start_time = time.time()

        nodes = {node:data for node, data in G.nodes(data=True)}
        gdf_nodes = gpd.GeoDataFrame(nodes).T
        if node_geometry:
            #gdf_nodes['geometry'] = gdf_nodes.apply(lambda row: Point(row['x'], row['y']), axis=1)
            gdf_nodes['geometry_pix'] = gdf_nodes.apply(lambda row: Point(row['x_pix'], row['y_pix']), axis=1)

        gdf_nodes.crs = G.graph['crs']
        gdf_nodes.gdf_name = '{}_nodes'.format(G.graph['name'])
        gdf_nodes['osmid'] = gdf_nodes['osmid'].astype(np.int64).map(make_str)

        to_return.append(gdf_nodes)
        log('Created GeoDataFrame "{}" from graph in {:,.2f} seconds'.format(gdf_nodes.gdf_name, time.time()-start_time))

    if edges:

        start_time = time.time()

        # create a list to hold our edges, then loop through each edge in the
        # graph
        edges = []
        for u, v, key, data in G.edges(keys=True, data=True):

            # for each edge, add key and all attributes in data dict to the
            # edge_details
            edge_details = {'u':u, 'v':v, 'key':key}
            for attr_key in data:
                edge_details[attr_key] = data[attr_key]

             # if edge doesn't already have a geometry attribute, create one now
            # if fill_edge_geometry==True
            if 'geometry_pix' not in data:
                if fill_edge_geometry:
                    point_u = Point((G.nodes[u]['x_pix'], G.nodes[u]['y_pix']))
                    point_v = Point((G.nodes[v]['x_pix'], G.nodes[v]['y_pix']))
                    edge_details['geometry_pix'] = LineString([point_u, point_v])
                else:
                    edge_details['geometry_pix'] = np.nan

#            # if edge doesn't already have a geometry attribute, create one now
#            # if fill_edge_geometry==True
#            if 'geometry' not in data:
#                if fill_edge_geometry:
#                    point_u = Point((G.nodes[u]['x'], G.nodes[u]['y']))
#                    point_v = Point((G.nodes[v]['x'], G.nodes[v]['y']))
#                    edge_details['geometry'] = LineString([point_u, point_v])
#                else:
#                    edge_details['geometry'] = np.nan

            edges.append(edge_details)

        # create a GeoDataFrame from the list of edges and set the CRS
        gdf_edges = gpd.GeoDataFrame(edges)
        gdf_edges.crs = G.graph['crs']
        gdf_edges.gdf_name = '{}_edges'.format(G.graph['name'])

        to_return.append(gdf_edges)
        log('Created GeoDataFrame "{}" from graph in {:,.2f} seconds'.format(gdf_edges.gdf_name, time.time()-start_time))

    if len(to_return) > 1:
        return tuple(to_return)
    else:
        return to_return[0]




###############################################################################
def graph_to_gdfs_pix_v0p5(G, nodes=True, edges=True, node_geometry=True, 
                  fill_edge_geometry=True):
    """
    Convert a graph into node and/or edge GeoDataFrames
    Parameters
    Use pixel coords
    https://github.com/gboeing/osmnx/blob/v0.5.4/osmnx/save_load.py

    ----------
    G : networkx multidigraph
    nodes : bool
        if True, convert graph nodes to a GeoDataFrame and return it
    edges : bool
        if True, convert graph edges to a GeoDataFrame and return it
    node_geometry : bool
        if True, create a geometry column from node x and y data
    fill_edge_geometry : bool
        if True, fill in missing edge geometry fields using origin and
        destination nodes
    Returns
    -------
    GeoDataFrame or tuple
        gdf_nodes or gdf_edges or both as a tuple
    """

    if not (nodes or edges):
        raise ValueError('You must request nodes or edges, or both.')

    to_return = []

    if nodes:

        start_time = time.time()

        nodes = {node:data for node, data in G.nodes(data=True)}
        gdf_nodes = gpd.GeoDataFrame(nodes).T
        if node_geometry:
            gdf_nodes['geometry_pix'] = gdf_nodes.apply(lambda row: Point(row['x_pix'], row['y_pix']), axis=1)
        gdf_nodes.crs = G.graph['crs']
        gdf_nodes.gdf_name = '{}_nodes'.format(G.graph['name'])
        gdf_nodes['osmid'] = gdf_nodes['osmid'].astype(np.int64).map(ox.utils.make_str)

        to_return.append(gdf_nodes)
        print('Created GeoDataFrame "{}" from graph in {:,.2f} seconds'.format(gdf_nodes.gdf_name, time.time()-start_time))

    if edges:

        start_time = time.time()

        # create a list to hold our edges, then loop through each edge in the
        # graph
        edges = []
        for u, v, key, data in G.edges(keys=True, data=True):

            # for each edge, add key and all attributes in data dict to the
            # edge_details
            edge_details = {'u':u, 'v':v, 'key':key}
            for attr_key in data:
                
                edge_details[attr_key] = data[attr_key]

            # if edge doesn't already have a geometry attribute, create one now
            # if fill_edge_geometry==True
            if 'geometry_pix' not in data:
                if fill_edge_geometry:
                    point_u = Point((G.node[u]['x_pix'], G.node[u]['y_pix']))
                    point_v = Point((G.node[v]['x_pix'], G.node[v]['y_pix']))
                    edge_details['geometry_pix'] = LineString([point_u, point_v])
                else:
                    edge_details['geometry_pix'] = np.nan

            edges.append(edge_details)

        # create a GeoDataFrame from the list of edges and set the CRS
        gdf_edges = gpd.GeoDataFrame(edges)
        gdf_edges.crs = G.graph['crs']
        gdf_edges.gdf_name = '{}_edges'.format(G.graph['name'])

        to_return.append(gdf_edges)
        print('Created GeoDataFrame "{}" from graph in {:,.2f} seconds'.format(gdf_edges.gdf_name, time.time()-start_time))

    if len(to_return) > 1:
        return tuple(to_return)
    else:
        return to_return[0]

###############################################################################
def plot_graph_pix(G, im=None, bbox=None, fig_height=6, fig_width=None, margin=0.02,
               axis_off=True, equal_aspect=False, bgcolor='w', show=True,
               save=False, close=True, file_format='png', filename='temp',
               default_dpi=300, annotate=False, node_color='#66ccff', node_size=15,
               node_alpha=1, node_edgecolor='none', node_zorder=1,
               edge_color='#999999', edge_linewidth=1, edge_alpha=1,
               edge_width_key='speed_mph', 
               edge_width_mult=1./25,
               use_geom=True):
    """
    Plot a networkx spatial graph.
    Parameters
    ----------
    G : networkx multidigraph
    bbox : tuple
        bounding box as north,south,east,west - if None will calculate from
        spatial extents of data. if passing a bbox, you probably also want to
        pass margin=0 to constrain it.
    fig_height : int
        matplotlib figure height in inches
    fig_width : int
        matplotlib figure width in inches
    margin : float
        relative margin around the figure
    axis_off : bool
        if True turn off the matplotlib axis
    equal_aspect : bool
        if True set the axis aspect ratio equal
    bgcolor : string
        the background color of the figure and axis
    show : bool
        if True, show the figure
    save : bool
        if True, save the figure as an image file to disk
    close : bool
        close the figure (only if show equals False) to prevent display
    file_format : string
        the format of the file to save (e.g., 'jpg', 'png', 'svg')
    filename : string
        the name of the file if saving
    default_dpi : int
        the resolution of the image file if saving (may get altered for
        large images)
    annotate : bool
        if True, annotate the nodes in the figure
    node_color : string
        the color of the nodes
    node_size : int
        the size of the nodes
    node_alpha : float
        the opacity of the nodes
    node_edgecolor : string
        the color of the node's marker's border
    node_zorder : int
        zorder to plot nodes, edges are always 2, so make node_zorder 1 to plot
        nodes beneath them or 3 to plot nodes atop them
    edge_color : string
        the color of the edges' lines
    edge_linewidth : float
        the width of the edges' lines
    edge_alpha : float
        the opacity of the edges' lines
    edge_width_key : str
        optional: key in edge propwerties to determine edge width,
        supercedes edge_linewidth, default to "speed_mph"
    edge_width_mult : float
        factor to rescale width for plotting, default to 1./25, which gives
        a line width of 1 for 25 mph speed limit.
    use_geom : bool
        if True, use the spatial geometry attribute of the edges to draw
        geographically accurate edges, rather than just lines straight from node
        to node
    Returns
    -------
    fig, ax : tuple
    """

    log('Begin plotting the graph...')
    node_Xs = [float(x) for _, x in G.nodes(data='x_pix')]
    node_Ys = [float(y) for _, y in G.nodes(data='y_pix')]
    #node_Xs = [float(x) for _, x in G.nodes(data='x')]
    #node_Ys = [float(y) for _, y in G.nodes(data='y')]

   # get north, south, east, west values either from bbox parameter or from the
    # spatial extent of the edges' geometries
    if bbox is None:
        edges = graph_to_gdfs_pix(G, nodes=False, fill_edge_geometry=True)
        #print ("plot_graph_pix():, edges:", edges)
        print ("plot_graph_pix():, edges.columns:", edges.columns)
        #print ("plot_graph_pix(): edges['geometry_pix']:", edges['geometry_pix'])
        #print ("plot_graph_pix(): edges['geometry']:", edges['geometry'])
        print ("type edges['geometry_pix'].:", type(edges['geometry_pix']))
        print ("type gpd.GeoSeries(edges['geometry_pix']):", type(gpd.GeoSeries(edges['geometry_pix'])))
        print ("type gpd.GeoSeries(edges['geometry_pix'][0]):", type(gpd.GeoSeries(edges['geometry_pix']).iloc[0]))
        west, south, east, north = gpd.GeoSeries(edges['geometry_pix']).total_bounds
        #west, south, east, north = edges.total_bounds
    else:
        north, south, east, west = bbox

#    # get north, south, east, west values either from bbox parameter or from the
#    # spatial extent of the edges' geometries
#    if bbox is None:
#        edges = graph_to_gdfs_pix(G, nodes=False, fill_edge_geometry=True)
#        west, south, east, north = edges.total_bounds
#    else:
#        north, south, east, west = bbox

    # if caller did not pass in a fig_width, calculate it proportionately from
    # the fig_height and bounding box aspect ratio
    bbox_aspect_ratio = (north-south)/(east-west)
    if fig_width is None:
        fig_width = fig_height / bbox_aspect_ratio

    # create the figure and axis
    print ("Creating figure and axis...")
    if im is not None:
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.imshow(im)
        print ("im.shape:", im.shape)
        #fig, ax = save_and_show(fig, ax, save, show, close, filename, file_format, dpi, axis_off)
        #return
    else:
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), facecolor=bgcolor)
        ax.set_facecolor(bgcolor)
    ## create the figure and axis
    #fig, ax = plt.subplots(figsize=(fig_width, fig_height), facecolor=bgcolor)
    #ax.set_facecolor(bgcolor)

    # draw the edges as lines from node to node
    start_time = time.time()
    lines = []
    widths = []
    for u, v, data in G.edges(keys=False, data=True):
        if 'geometry_pix' in data and use_geom:
            # if it has a geometry attribute (a list of line segments), add them
            # to the list of lines to plot
            xs, ys = data['geometry_pix'].xy
            lines.append(list(zip(xs, ys)))
        else:
            # if it doesn't have a geometry attribute, the edge is a straight
            # line from node to node
            x1 = G.nodes[u]['x_pix']
            y1 = G.nodes[u]['y_pix']
            x2 = G.nodes[v]['x_pix']
            y2 = G.nodes[v]['y_pix']
            line = [(x1, y1), (x2, y2)]
            lines.append(line)
            
        # get widths
        if edge_width_key in data:
            width = int(np.rint(data[edge_width_key] * edge_width_mult))
        else:
            width = edge_linewidth
        widths.append(width)

    # add the lines to the axis as a linecollection
    lc = LineCollection(lines, colors=edge_color, 
                        linewidths=widths,
                        alpha=edge_alpha, zorder=2)
    ax.add_collection(lc)
    log('Drew the graph edges in {:,.2f} seconds'.format(time.time()-start_time))

    # scatter plot the nodes
    ax.scatter(node_Xs, node_Ys, s=node_size, c=node_color, alpha=node_alpha, 
               edgecolor=node_edgecolor, zorder=node_zorder)

    # set the extent of the figure
    margin_ns = (north - south) * margin
    margin_ew = (east - west) * margin
    ax.set_ylim((south - margin_ns, north + margin_ns))
    ax.set_xlim((west - margin_ew, east + margin_ew))

    # configure axis appearance
    xaxis = ax.get_xaxis()
    yaxis = ax.get_yaxis()

    xaxis.get_major_formatter().set_useOffset(False)
    yaxis.get_major_formatter().set_useOffset(False)

    # if axis_off, turn off the axis display set the margins to zero and point
    # the ticks in so there's no space around the plot
    if axis_off:
        ax.axis('off')
        ax.margins(0)
        ax.tick_params(which='both', direction='in')
        xaxis.set_visible(False)
        yaxis.set_visible(False)
        fig.canvas.draw()

    if equal_aspect:
        # make everything square
        ax.set_aspect('equal')
        fig.canvas.draw()
    else:
        # if the graph is not projected, conform the aspect ratio to not stretch the plot
        if G.graph['crs'] == ox_settings.default_crs:
            coslat = np.cos((min(node_Ys) + max(node_Ys)) / 2. / 180. * np.pi)
            ax.set_aspect(1. / coslat)
            fig.canvas.draw()

    # annotate the axis with node IDs if annotate=True
    if annotate:
        for node, data in G.nodes(data=True):
            ax.annotate(node, xy=(data['x_pix'], data['y_pix']))

    # update dpi, if image
    if im is not None:
        #   mpl can handle a max of 2^29 pixels, or 23170 on a side
        # recompute max_dpi
        max_dpi = int(23000 / max(fig_height, fig_width))
        h, w = im.shape[:2]
        # try to set dpi to native resolution of imagery
        desired_dpi = max(default_dpi, 1.0 * h / fig_height) 
        #desired_dpi = max(default_dpi, int( np.max(im.shape) / max(fig_height, fig_width) ) )
        dpi = int(np.min([max_dpi, desired_dpi ]))

    # save and show the figure as specified
    fig, ax = save_and_show(fig, ax, save, show, close, filename, 
                            file_format, dpi, axis_off)
    return fig, ax

###############################################################################
def plot_graph_pix_v0p5(G, im=None, bbox=None, fig_height=6, fig_width=None, margin=0.01,
               axis_off=True, equal_aspect=False, bgcolor='w', show=True,
               save=False, close=True, file_format='png', filename='temp',
               dpi=300, annotate=False, node_color='#66ccff', node_size=15,
               node_alpha=1, node_edgecolor='none', node_zorder=1,
               edge_color='#999999', edge_linewidth=1, edge_alpha=1,
               use_geom=True):
    """
    Plot a networkx spatial graph. Use pixel coords
    From: https://github.com/gboeing/osmnx/blob/v0.5.4/osmnx/plot.py
    
    Parameters
    ----------
    G : networkx multidigraph
    bbox : tuple
        bounding box as north,south,east,west - if None will calculate from
        spatial extents of data
    fig_height : int
        matplotlib figure height in inches
    fig_width : int
        matplotlib figure width in inches
    margin : float
        relative margin around the figure
    axis_off : bool
        if True turn off the matplotlib axis
    equal_aspect : bool
        if True set the axis aspect ratio equal
    bgcolor : string
        the background color of the figure and axis
    show : bool
        if True, show the figure
    save : bool
        if True, save the figure as an image file to disk
    close : bool
        close the figure (only if show equals False) to prevent display
    file_format : string
        the format of the file to save (e.g., 'jpg', 'png', 'svg')
    filename : string
        the name of the file if saving
    dpi : int
        the resolution of the image file if saving
    annotate : bool
        if True, annotate the nodes in the figure
    node_color : string
        the color of the nodes
    node_size : int
        the size of the nodes
    node_alpha : float
        the opacity of the nodes
    node_edgecolor : string
        the color of the node's marker's border
    node_zorder : int
        zorder to plot nodes, edges are always 2, so make node_zorder 1 to plot
        nodes beneath them or 3 to plot nodes atop them
    edge_color : string
        the color of the edges' lines
    edge_linewidth : float
        the width of the edges' lines
    edge_alpha : float
        the opacity of the edges' lines
    use_geom : bool
        if True, use the spatial geometry attribute of the edges to draw
        geographically accurate edges, rather than just lines straight from node
        to node
    Returns
    -------
    fig, ax : tuple
    """

    print('Begin plotting the graph...')
    node_Xs = [float(node['x_pix']) for node in G.node.values()]
    node_Ys = [float(node['y_pix']) for node in G.node.values()]

    # get north, south, east, west values either from bbox parameter or from the
    # spatial extent of the edges' geometries
    if bbox is None:
        edges = graph_to_gdfs_pix(G, nodes=False, fill_edge_geometry=True)
        #print ("plot_graph_pix():, edges:", edges)
        print ("plot_graph_pix():, edges.columns:", edges.columns)
        #print ("plot_graph_pix(): edges['geometry_pix']:", edges['geometry_pix'])
        #print ("plot_graph_pix(): edges['geometry']:", edges['geometry'])
        print ("type edges['geometry_pix'].:", type(edges['geometry_pix']))
        print ("type gpd.GeoSeries(edges['geometry_pix']):", type(gpd.GeoSeries(edges['geometry_pix'])))
        print ("type gpd.GeoSeries(edges['geometry_pix'][0]):", type(gpd.GeoSeries(edges['geometry_pix']).iloc[0]))
        west, south, east, north = gpd.GeoSeries(edges['geometry_pix']).total_bounds
        #west, south, east, north = edges.total_bounds
    else:
        north, south, east, west = bbox

    print ("north:", north)
    print ("south:", south)
    print ("east:", east)
    print ("west:", west)

    # if caller did not pass in a fig_width, calculate it proportionately from
    # the fig_height and bounding box aspect ratio
    bbox_aspect_ratio = (north-south)/(east-west)
    if fig_width is None:
        fig_width = fig_height / bbox_aspect_ratio


    # create the figure and axis
    print ("Creating figure and axis...")
    if im is not None:
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.imshow(im)
        print ("im.shape:", im.shape)
        #fig, ax = save_and_show(fig, ax, save, show, close, filename, file_format, dpi, axis_off)
        #return
    else:
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), facecolor=bgcolor)
        ax.set_facecolor(bgcolor)

    # create the figure and axis
    #fig, ax = plt.subplots(figsize=(fig_width, fig_height), facecolor=bgcolor)
    #ax.set_facecolor(bgcolor)


    # draw the edges as lines from node to node
    print ("Draw the edges as lines from node to node...")
    start_time = time.time()
    lines = []
    x_all, y_all = [], []
    for u, v, data in G.edges(keys=False, data=True):
        if 'geometry_pix' in data and use_geom:
            # if it has a geometry attribute (a list of line segments), add them
            # to the list of lines to plot
            xs, ys = data['geometry_pix'].xy
            lines.append(list(zip(xs, ys)))
            x_all.extend(xs)
            y_all.extend(ys)
        else:
            # if it doesn't have a geometry attribute, the edge is a straight
            # line from node to node
            x1 = G.node[u]['x_pix']
            y1 = G.node[u]['y_pix']
            x2 = G.node[v]['x_pix']
            y2 = G.node[v]['y_pix']
            line = [(x1, y1), (x2, y2)]
            lines.append(line)
            x_all.append([x1, x2])
            y_all.append([y1, y2])
    print ("lines[-1]:", lines[-1])

    # reset n,s,e,w?
    print ("Reseting north, south, east, west...")
    x_all = np.array(x_all)
    y_all = np.array(y_all)
    north, south = np.min(y_all), np.max(y_all)
    west, east = np.min(x_all), np.max(x_all)
    print ("north, south:", north, south)
    print ("east, west:", east, west)

    # add the lines to the axis as a linecollection
    lc = LineCollection(lines, colors=edge_color, linewidths=edge_linewidth, alpha=edge_alpha, zorder=2)
    ax.add_collection(lc)
    print('Drew the graph edges in {:,.2f} seconds'.format(time.time()-start_time))

    # scatter plot the nodes
    print ("Scatter plot the nodes")
    ax.scatter(node_Xs, node_Ys, s=node_size, c=node_color, alpha=node_alpha, 
               edgecolor=node_edgecolor, zorder=node_zorder)

    # set the extent of the figure
    margin_ns = (north - south) * margin
    margin_ew = (east - west) * margin
    ax.set_ylim((south - margin_ns, north + margin_ns))
    ax.set_xlim((west - margin_ew, east + margin_ew))

    # configure axis appearance
    ax.get_xaxis().get_major_formatter().set_useOffset(False)
    ax.get_yaxis().get_major_formatter().set_useOffset(False)

    # if axis_off, turn off the axis display set the margins to zero and point
    # the ticks in so there's no space around the plot
    # actually, keep margin
    if axis_off:
        ax.axis('off')
        #ax.margins(0)
        ax.tick_params(which='both', direction='in')
        fig.canvas.draw()

    if equal_aspect:
        # make everything square
        ax.set_aspect('equal')
        fig.canvas.draw()

    # annotate the axis with node IDs if annotate=True
    if annotate:
        for node, data in G.nodes(data=True):
            ax.annotate(node, xy=(data['x_pix'], data['y_pix']))

    # save and show the figure as specified
    print("Saving plot...")
    fig, ax = save_and_show(fig, ax, save, show, close, filename, file_format, dpi, axis_off)
    return fig, ax


###############################################################################
def plot_graph_route_pix(G, route, im=None, bbox=None, fig_height=6, fig_width=None,
                     margin=0.02, bgcolor='w', axis_off=True, show=True,
                     save=False, close=True, file_format='png', filename='temp',
                     dpi=300, annotate=False, node_color='#999999',
                     node_size=15, node_alpha=1, node_edgecolor='none',
                     node_zorder=1, edge_color='#999999', edge_linewidth=1,
                     edge_alpha=1,
                     edge_width_key='speed_mph',
                     edge_width_mult=1./25,
                     use_geom=True, origin_point=None,
                     destination_point=None, route_color='r', route_linewidth=4,
                     route_alpha=0.5, orig_dest_node_alpha=0.5,
                     orig_dest_node_size=100, 
                     orig_dest_node_color='r'):
    """
    Plot a route along a networkx spatial graph.
    Parameters
    ----------
    G : networkx multidigraph
    route : list
        the route as a list of nodes
    bbox : tuple
        bounding box as north,south,east,west - if None will calculate from
        spatial extents of data. if passing a bbox, you probably also want to
        pass margin=0 to constrain it.
    fig_height : int
        matplotlib figure height in inches
    fig_width : int
        matplotlib figure width in inches
    margin : float
        relative margin around the figure
    axis_off : bool
        if True turn off the matplotlib axis
    bgcolor : string
        the background color of the figure and axis
    show : bool
        if True, show the figure
    save : bool
        if True, save the figure as an image file to disk
    close : bool
        close the figure (only if show equals False) to prevent display
    file_format : string
        the format of the file to save (e.g., 'jpg', 'png', 'svg')
    filename : string
        the name of the file if saving
    dpi : int
        the resolution of the image file if saving
    annotate : bool
        if True, annotate the nodes in the figure
    node_color : string
        the color of the nodes
    node_size : int
        the size of the nodes
    node_alpha : float
        the opacity of the nodes
    node_edgecolor : string
        the color of the node's marker's border
    node_zorder : int
        zorder to plot nodes, edges are always 2, so make node_zorder 1 to plot
        nodes beneath them or 3 to plot nodes atop them
    edge_color : string
        the color of the edges' lines
    edge_linewidth : float
        the width of the edges' lines
    edge_alpha : float
        the opacity of the edges' lines
    use_geom : bool
        if True, use the spatial geometry attribute of the edges to draw
        geographically accurate edges, rather than just lines straight from node
        to node
    origin_point : tuple
        optional, an origin (lat, lon) point to plot instead of the origin node
    destination_point : tuple
        optional, a destination (lat, lon) point to plot instead of the
        destination node
    route_color : string
        the color of the route
    route_linewidth : int
        the width of the route line
    route_alpha : float
        the opacity of the route line
    orig_dest_node_alpha : float
        the opacity of the origin and destination nodes
    orig_dest_node_size : int
        the size of the origin and destination nodes
    orig_dest_node_color : string
        the color of the origin and destination nodes 
        (can be a string or list with (origin_color, dest_color))
        of nodes
    Returns
    -------
    fig, ax : tuple
    """

    # plot the graph but not the route
    fig, ax = plot_graph_pix(G, im=im, bbox=bbox, fig_height=fig_height, fig_width=fig_width,
                         margin=margin, axis_off=axis_off, bgcolor=bgcolor,
                         show=False, save=False, close=False, filename=filename,
                         default_dpi=dpi, annotate=annotate, node_color=node_color,
                         node_size=node_size, node_alpha=node_alpha,
                         node_edgecolor=node_edgecolor, node_zorder=node_zorder,
                         edge_color=edge_color, edge_linewidth=edge_linewidth,
                         edge_alpha=edge_alpha, edge_width_key=edge_width_key,
                         edge_width_mult=edge_width_mult,
                         use_geom=use_geom)

    # the origin and destination nodes are the first and last nodes in the route
    origin_node = route[0]
    destination_node = route[-1]

    if origin_point is None or destination_point is None:
        # if caller didn't pass points, use the first and last node in route as
        # origin/destination
        origin_destination_ys = (G.nodes[origin_node]['y_pix'],
                                 G.nodes[destination_node]['y_pix'])
        origin_destination_xs = (G.nodes[origin_node]['x_pix'],
                                 G.nodes[destination_node]['x_pix'])
    else:
        # otherwise, use the passed points as origin/destination
        origin_destination_xs = (origin_point[0], destination_point[0])
        origin_destination_ys = (origin_point[1], destination_point[1])

    # scatter the origin and destination points
    ax.scatter(origin_destination_xs, origin_destination_ys,
               s=orig_dest_node_size, 
               c=orig_dest_node_color,
               alpha=orig_dest_node_alpha, edgecolor=node_edgecolor, zorder=4)

    # plot the route lines
    edge_nodes = list(zip(route[:-1], route[1:]))
    lines = []
    for u, v in edge_nodes:
        # if there are parallel edges, select the shortest in length
        data = min(G.get_edge_data(u, v).values(), key=lambda x: x['length'])

        # if it has a geometry attribute (ie, a list of line segments)
        if 'geometry_pix' in data and use_geom:
            # add them to the list of lines to plot
            xs, ys = data['geometry_pix'].xy
            lines.append(list(zip(xs, ys)))
        else:
            # if it doesn't have a geometry attribute, the edge is a straight
            # line from node to node
            x1 = G.nodes[u]['x_pix']
            y1 = G.nodes[u]['y_pix']
            x2 = G.nodes[v]['x_pix']
            y2 = G.nodes[v]['y_pix']
            line = [(x1, y1), (x2, y2)]
            lines.append(line)

    # add the lines to the axis as a linecollection
    lc = LineCollection(lines, colors=route_color, linewidths=route_linewidth, alpha=route_alpha, zorder=3)
    ax.add_collection(lc)

    # save and show the figure as specified
    fig, ax = save_and_show(fig, ax, save, show, close, filename, file_format, dpi, axis_off)
    return fig, ax

###############################################################################
def plot_graph_route_pix_v0p5(G, route, im=None, bbox=None, fig_height=6, fig_width=None,
                     margin=0.01, bgcolor='w', axis_off=True, show=True,
                     save=False, close=True, file_format='png', filename='temp',
                     dpi=300, annotate=False, node_color='#999999',
                     node_size=15, node_alpha=1, node_edgecolor='none',
                     node_zorder=1, edge_color='#999999', edge_linewidth=1,
                     edge_alpha=1, use_geom=True, origin_point=None,
                     destination_point=None, route_color='r', route_linewidth=4,
                     route_alpha=0.5, orig_dest_node_alpha=0.5,
                     orig_dest_node_size=100, orig_dest_node_color='r',
                     orig_dest_point_color='b'):
    """
    Plot a route along a networkx spatial graph. Use pixel coords
    From: https://github.com/gboeing/osmnx/blob/v0.5.4/osmnx/plot.py

    Parameters
    ----------
    G : networkx multidigraph
    route : list
        the route as a list of nodes
    bbox : tuple
        bounding box as north,south,east,west - if None will calculate from
        spatial extents of data
    fig_height : int
        matplotlib figure height in inches
    fig_width : int
        matplotlib figure width in inches
    margin : float
        relative margin around the figure
    axis_off : bool
        if True turn off the matplotlib axis
    bgcolor : string
        the background color of the figure and axis
    show : bool
        if True, show the figure
    save : bool
        if True, save the figure as an image file to disk
    close : bool
        close the figure (only if show equals False) to prevent display
    file_format : string
        the format of the file to save (e.g., 'jpg', 'png', 'svg')
    filename : string
        the name of the file if saving
    dpi : int
        the resolution of the image file if saving
    annotate : bool
        if True, annotate the nodes in the figure
    node_color : string
        the color of the nodes
    node_size : int
        the size of the nodes
    node_alpha : float
        the opacity of the nodes
    node_edgecolor : string
        the color of the node's marker's border
    node_zorder : int
        zorder to plot nodes, edges are always 2, so make node_zorder 1 to plot
        nodes beneath them or 3 to plot nodes atop them
    edge_color : string
        the color of the edges' lines
    edge_linewidth : float
        the width of the edges' lines
    edge_alpha : float
        the opacity of the edges' lines
    use_geom : bool
        if True, use the spatial geometry attribute of the edges to draw
        geographically accurate edges, rather than just lines straight from node
        to node
    origin_point : tuple
        optional, an origin (lat, lon) point to plot instead of the origin node
    destination_point : tuple
        optional, a destination (lat, lon) point to plot instead of the
        destination node
    route_color : string
        the color of the route
    route_linewidth : int
        the width of the route line
    route_alpha : float
        the opacity of the route line
    orig_dest_node_alpha : float
        the opacity of the origin and destination nodes
    orig_dest_node_size : int
        the size of the origin and destination nodes
    orig_dest_node_color : string
        the color of the origin and destination nodes
    orig_dest_point_color : string
        the color of the origin and destination points if being plotted instead
        of nodes
    Returns
    -------
    fig, ax : tuple
    """

    # plot the graph but not the route
    fig, ax = plot_graph_pix(G, im=im, bbox=bbox, fig_height=fig_height, fig_width=fig_width,
                         margin=margin, axis_off=axis_off, bgcolor=bgcolor,
                         show=False, save=False, close=False, filename=filename,
                         dpi=dpi, annotate=annotate, node_color=node_color,
                         node_size=node_size, node_alpha=node_alpha,
                         node_edgecolor=node_edgecolor, node_zorder=node_zorder,
                         edge_color=edge_color, edge_linewidth=edge_linewidth,
                         edge_alpha=edge_alpha, use_geom=use_geom)

    # the origin and destination nodes are the first and last nodes in the route
    origin_node = route[0]
    destination_node = route[-1]

    if origin_point is None or destination_point is None:
        # if caller didn't pass points, use the first and last node in route as
        # origin/destination
        origin_destination_lats = (G.node[origin_node]['y_pix'], G.node[destination_node]['y_pix'])
        origin_destination_lons = (G.node[origin_node]['x_pix'], G.node[destination_node]['x_pix'])
    else:
        # otherwise, use the passed points as origin/destination
        origin_destination_lats = (origin_point[0], destination_point[0])
        origin_destination_lons = (origin_point[1], destination_point[1])
        orig_dest_node_color = orig_dest_point_color

    # scatter the origin and destination points
    ax.scatter(origin_destination_lons, origin_destination_lats, s=orig_dest_node_size,
               c=orig_dest_node_color, alpha=orig_dest_node_alpha, edgecolor=node_edgecolor, zorder=4)

    # plot the route lines
    print ("Plotting route lines...")
    edge_nodes = list(zip(route[:-1], route[1:]))
    lines = []
    for u, v in edge_nodes:
        # if there are parallel edges, select the shortest in length
        data = min([data for data in G.edges([u][v]).values()], key=lambda x: x['length'])
        #data = min([data for data in G.edge[u][v].values()], key=lambda x: x['length'])

        # if it has a geometry attribute (ie, a list of line segments)
        if 'geometry_pix' in data and use_geom:
            # add them to the list of lines to plot
            xs, ys = data['geometry_pix'].xy
            lines.append(list(zip(xs, ys)))
        else:
            # if it doesn't have a geometry attribute, the edge is a straight
            # line from node to node
            x1 = G.node[u]['x_pix']
            y1 = G.node[u]['y_pix']
            x2 = G.node[v]['x_pix']
            y2 = G.node[v]['y_pix']
            line = [(x1, y1), (x2, y2)]
            lines.append(line)

    # add the lines to the axis as a linecollection
    lc = LineCollection(lines, colors=route_color, linewidths=route_linewidth, alpha=route_alpha, zorder=3)
    ax.add_collection(lc)

    # save and show the figure as specified
    print ("Saving route plot...")
    fig, ax = save_and_show(fig, ax, save, show, close, filename, file_format, dpi, axis_off)
    return fig, ax


###############################################################################
def save_and_show(fig, ax, save, show, close, filename, file_format, dpi, 
                  axis_off, tight_layout=False):
    """
    Save a figure to disk and show it, as specified.
    Assume filename holds entire path to file
    
    Parameters
    ----------
    fig : figure
    ax : axis
    save : bool
        whether to save the figure to disk or not
    show : bool
        whether to display the figure or not
    close : bool
        close the figure (only if show equals False) to prevent display
    filename : string
        the name of the file to save
    file_format : string
        the format of the file to save (e.g., 'jpg', 'png', 'svg')
    dpi : int
        the resolution of the image file if saving
    axis_off : bool
        if True matplotlib axis was turned off by plot_graph so constrain the
        saved figure's extent to the interior of the axis
    Returns
    -------
    fig, ax : tuple
    """
    # save the figure if specified
    if save:
        start_time = time.time()

        # create the save folder if it doesn't already exist
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        path_filename = filename #os.path.join(settings.imgs_folder, os.extsep.join([filename, file_format]))

        if file_format == 'svg':
            # if the file_format is svg, prep the fig/ax a bit for saving
            ax.axis('off')
            ax.set_position([0, 0, 1, 1])
            ax.patch.set_alpha(0.)
            fig.patch.set_alpha(0.)
            fig.savefig(path_filename, bbox_inches=0, format=file_format, facecolor=fig.get_facecolor(), transparent=True)
        else:
            if axis_off:
                # if axis is turned off, constrain the saved figure's extent to
                # the interior of the axis
                extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            else:
                extent = 'tight'
                
            if tight_layout:
                # extent = 'tight'
                fig.gca().set_axis_off()
                fig.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                                hspace = 0, wspace = 0)
                plt.margins(0,0)
                # fig.gca().xaxis.set_major_locator(NullLocator())
                # fig.gca().yaxis.set_major_locator(NullLocator())
                fig.savefig(path_filename, dpi=dpi, bbox_inches=extent,
                            format=file_format, facecolor=fig.get_facecolor(),
                            transparent=True, pad_inches=0)
            else:
                
                fig.savefig(path_filename, dpi=dpi, bbox_inches=extent,
                            format=file_format, facecolor=fig.get_facecolor(),
                            transparent=True)


        print('Saved the figure to disk in {:,.2f} seconds'.format(time.time()-start_time))

    # show the figure if specified
    if show:
        start_time = time.time()
        plt.show()
        print('Showed the plot in {:,.2f} seconds'.format(time.time()-start_time))
    # if show=False, close the figure if close=True to prevent display
    elif close:
        plt.close()

    return fig, ax

###############################################################################
def main():
    
    default_crs = {'init':'epsg:4326'}

#    # Vegas settings
#    local = False
#    fig_height=12
#    fig_width=12
#    node_color='#ffdd1a'
#    edge_color='#ffdd1a'
#    node_size=0.2
#    node_alpha=0.7
#    edge_linewidth=0.3
#    edge_alpha=0.8
#    orig_dest_node_size=4.5*node_size    
#    save_only_route_png = False  # True

    # 60cm Google imagery 
    local = False
    max_plots = 12
    fig_height=12
    fig_width=12
    node_color='#ffdd1a'
    edge_color='#ffdd1a'
    node_size=0.4
    node_alpha=0.75
    edge_linewidth=0.6
    edge_alpha=0.85
    orig_dest_node_size=4.5*node_size    
    save_only_route_png = False  # True
    shuffle = True
    
    # new york
    node_color, edge_color = 'yellow', 'yellow'
    edge_linewidth=0.9
 
#    # Khartoum settings (metrics)
#    save_only_route_png = False#True
#    local = False
#    fig_height=12
#    fig_width=12
#    node_color='#66ccff'  # light blue
#    node_size=0.6
#    node_alpha=0.7
#    edge_color='#bfefff'   # lightblue1
#    edge_linewidth=0.4
#    edge_alpha=0.7
#    orig_dest_node_size=8*node_size
  
#    # Vegas0 settings
#    local = False
#    fig_height=12
#    fig_width=12
#    node_color='#66ccff'  # light blue
#    #node_color='#8b3626' # tomato4
#    node_size=0.4
#    node_alpha=0.6
#    #edge_color='#999999'  # gray
#    #edge_color='#ee5c42'  # tomato2
#    edge_color='#bfefff'   # lightblue1
#    edge_linewidth=0.2
#    edge_alpha=0.5
#    orig_dest_node_size=4.5*node_size
    
#    # shanghai yuge settings 
#    save_only_route_png = False#True
#    local = False
#    fig_height=12
#    fig_width=12
#    node_color='#66ccff'  # light blue
#    node_size=0.4
#    node_alpha=0.6
#    edge_color='#bfefff'   # lightblue1
#    edge_linewidth=0.3
#    edge_alpha=0.6
#    orig_dest_node_size=8*node_size

    route_color='r'
    orig_dest_node_color='r'
    route_linewidth=4*edge_linewidth
    
    #dpi=1000 # set dpi latet using image size, to approximate native resolution


    # local
    if local:
        albu_path = '/Users/avanetten/Documents/cosmiq/apls/albu_inference_mod'
        path_images_raw = '/Users/avanetten/Documents/cosmiq/spacenet/data/spacenetv2/AOI_2_Vegas_Test/400m/RGB-PanSharpen'
        res_root_dir = os.path.join(albu_path, 'results/2m_4fold_512_30e_d0.2_g0.2_AOI_2_Vegas_Test')
    
    # deployed
    else:
        from config import Config
        parser = argparse.ArgumentParser()
        parser.add_argument('config_path')
        args = parser.parse_args()
        with open(args.config_path, 'r') as f:
            cfg = json.load(f)
            config = Config(**cfg)
    
        # outut files
        res_root_dir = os.path.join(config.path_results_root, config.test_results_dir)
        path_images_8bit = os.path.join(config.path_data_root, config.test_data_refined_dir)
        graph_dir = os.path.join(res_root_dir, config.graph_dir)
        out_dir = graph_dir.strip() + '_plots'
        #res_root_dir = config.results_dir #os.path.dirname(im_dir)
        #path_images_8bit = config.path_images_8bit
        #graph_dir = os.path.join(res_root_dir, 'graphs')

    
    # iterate through images and graphs, plot routes
    im_list = sorted([z for z in os.listdir(path_images_8bit) if z.endswith('.tif')])
    # im_list = [z for z in os.listdir(path_images_8bit) if z.startswith('khartoum')]
    #im_list = [z for z in os.listdir(path_images_8bit) if z.startswith('paris')]
    #im_list = [z for z in os.listdir(path_images_8bit) if z.startswith('new')]
    
    if shuffle:
        random.shuffle(im_list)

    for i,im_root in enumerate(im_list):#enumerate(os.listdir(path_images_8bit)):
        if not im_root.endswith('.tif'):
            continue
    
        if i >= max_plots:
            break
        
        im_root_no_ext = im_root.split('.tif')[0]
        im_file = os.path.join(path_images_8bit, im_root)
        graph_pkl = os.path.join(graph_dir, im_root_no_ext + '.gpickle')
        print ("\n\n", i, "im_root:", im_root)
        print ("  im_file:", im_file)
        print ("  graph_pkl:", graph_pkl)
        
        # read G
        # graphml
        #G = load_graphml(im_root_no_ext + '.graphml', folder=graph_dir)
        
        # gpickle?
        print ("Reading gpickle...")
        G = nx.read_gpickle(graph_pkl)
        
        # make sure geometries are not just strings
        print ("Make sure geometries are not just strings...")
        for u, v, key, data in G.edges(keys=True, data=True):
            for attr_key in data:
                if (attr_key == 'geometry') and (type(data[attr_key]) == str):
                    #print ("update geometry...")
                    data[attr_key] = wkt.loads(data[attr_key])
                elif (attr_key == 'geometry_pix') and (type(data[attr_key]) == str):
                    data[attr_key] = wkt.loads(data[attr_key])                
                else:
                    continue
                
        # G = ox.project_graph(G)
        # G = ox.simplify_graph(G.to_directed())
        
        # G.to_directed()

        # project??
        #G = ox.project_graph(G)

        # print a node
        # nx v2
        #print (G.nodes)
        #node = G.nodes[0]
        #print ("G.nodes(data='x'):", G.nodes(data='x'))
        # nx v 1.11

        # print a node
        node = list(G.nodes())[-1]
        print (node, "random node props:", G.nodes[node])
        # print an edge
        edge_tmp = list(G.edges())[-1]
        print (edge_tmp, "random edge props:", G.edges([edge_tmp[0], edge_tmp[1]])) #G.edge[edge_tmp[0]][edge_tmp[1]])

        #node = G.nodes()[-1]
        #print ("node:", node, "props:", G.node[node])
        #u,v = G.edges()[-1]
        #print ("edge:", u,v, "props:", G.edge[u][v])


        # read in image, cv2 fails on large files
        print ("Read in image...")
        try:
            #convert to rgb (cv2 reads in bgr)
            img_cv2 = cv2.imread(im_file, 1)
            print ("img_cv2.shape:", img_cv2.shape)
            im = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
        except:
            im = skimage.io.imread(im_file, as_grey=False).astype(np.uint8)#[::-1]

        # set dpi to approximate native resolution
        print ("im.shape:", im.shape)
        desired_dpi = int( np.max(im.shape) / np.max([fig_height, fig_width]) ) 
        print ("desired dpi:", desired_dpi)
        # max out dpi at 3500
        dpi = int(np.min([3500, desired_dpi ]))
        print ("plot dpi:", dpi)

#        # plot graph 
#        if not save_only_route_png:
#            #out_file_plot = os.path.join(graph_dir, im_root_no_ext + '_ox_raw_plot')
#            out_file_plot = im_root_no_ext + '_ox_raw_plot'
#            print ("outfile_plot:", out_file_plot)
#            ox.plot_graph(G, fig_height=fig_height, fig_width=fig_width, 
#                           filename=out_file_plot, dpi=dpi,                        
#                           show=False, save=True)


        # plot graph with image backround
        if not save_only_route_png:
            out_file_plot = os.path.join(out_dir, im_root_no_ext + '_ox_plot.tif')
            print ("outfile_plot:", out_file_plot)
            plot_graph_pix(G, im, fig_height=fig_height, fig_width=fig_width, 
                           node_size=node_size, node_alpha=node_alpha, node_color=node_color, 
                           edge_linewidth=edge_linewidth, edge_alpha=edge_alpha, edge_color=edge_color,
                           filename=out_file_plot, default_dpi=dpi,                        
                           show=False, save=True)
            
#        ################            
#        # plot graph route
#        print ("\nPlot a random route on the graph...")
#        t0 = time.time()
#        # set source
#        source_idx = np.random.randint(0,len(G.nodes()))
#        source = list(G.nodes())[source_idx]
#        # get routes
#        lengths, paths = nx.single_source_dijkstra(G, source=source, weight='length')
#        # random target
#        targ_idx = np.random.randint(0, len(list(lengths.keys())))
#        target = list(lengths.keys())[targ_idx]
#        # specific route
#        route = paths[target]
# 
#        print ("source:", source)
#        print ("target:", target)
#        print ("route:", route)
#        
#        # plot route       
#        out_file_route = os.path.join(out_dir, im_root_no_ext + '_ox_route_r0.tif')
#        print ("outfile_route:", out_file_route)
#        plot_graph_route_pix(G, route, im=im, fig_height=fig_height, fig_width=fig_width, 
#                       node_size=node_size, node_alpha=node_alpha, node_color=node_color, 
#                       edge_linewidth=edge_linewidth, edge_alpha=edge_alpha, edge_color=edge_color,
#                       orig_dest_node_size=orig_dest_node_size,
#                       route_color=route_color, 
#                       orig_dest_node_color=orig_dest_node_color,
#                       route_linewidth=route_linewidth,
#                       filename=out_file_route, dpi=dpi,                        
#                       show=False, save=True)
#        t1 = time.time()
#        print ("Time to run plot_graph_route_pix():", t1-t0, "seconds")
#
#
#        ################
#        # plot another graph route?
#        print ("\nPlot another random route on the graph...")
#        t0 = time.time()
#        # set source
#        source_idx = np.random.randint(0,len(G.nodes()))
#        source = list(G.nodes())[source_idx]
#        # get routes
#        lengths, paths = nx.single_source_dijkstra(G, source=source, weight='length')
#        # random target
#        targ_idx = np.random.randint(0, len(list(lengths.keys())))
#        target = list(lengths.keys())[targ_idx]
#        # specific route
#        route = paths[target]
# 
#        print ("source:", source)
#        print ("target:", target)
#        print ("route:", route)
#        
#        # plot another route       
#        out_file_route = os.path.join(out_dir, im_root_no_ext + '_ox_route_r1.tif')
#        print ("outfile_route:", out_file_route)
#        plot_graph_route_pix(G, route, im=im, fig_height=fig_height, fig_width=fig_width, 
#                       node_size=node_size, node_alpha=node_alpha, node_color=node_color, 
#                       edge_linewidth=edge_linewidth, edge_alpha=edge_alpha, edge_color=edge_color,
#                       orig_dest_node_size=orig_dest_node_size,
#                       route_color=route_color, 
#                       orig_dest_node_color=orig_dest_node_color,
#                       route_linewidth=route_linewidth,
#                       filename=out_file_route, dpi=dpi,                        
#                       show=False, save=True)
#        t1 = time.time()
#        print("Time to run plot_graph_route_pix():", t1-t0, "seconds")

        
###############################################################################
if __name__ == "__main__":
    main()
